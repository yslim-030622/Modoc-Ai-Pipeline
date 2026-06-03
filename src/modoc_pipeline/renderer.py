"""FFmpeg rendering helpers for final vertical MP4 assembly."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


XFADE_DURATION = 0.5


class RenderError(RuntimeError):
    """Raised when local rendering cannot proceed."""


@dataclass(frozen=True)
class RenderedVideo:
    language: str
    path: Path


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RenderError(
            "ffmpeg is not installed. Install it before rendering final MP4 files:\n"
            "  brew install ffmpeg"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public render entry points
# ─────────────────────────────────────────────────────────────────────────────

def render_language_videos(
    *,
    run_dir: Path,
    scenes: dict[str, list[dict[str, Any]]],
    video_output_dir: Path | None = None,
) -> list[RenderedVideo]:
    ensure_ffmpeg()

    rendered: list[RenderedVideo] = []
    videos_dir = video_output_dir or run_dir / "videos"
    assets_dir = run_dir / "render_assets"
    videos_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    for language, language_scenes in scenes.items():
        clip_paths = [
            run_dir / "veo" / language / f"{scene['scene_id']}.mp4"
            for scene in language_scenes
        ]
        missing = [path for path in clip_paths if not path.exists()]
        if missing:
            raise RenderError(
                f"Missing Veo clips for {language}. Run `python3 -m modoc_pipeline veo --run {run_dir}` first. "
                f"First missing file: {missing[0]}"
            )

        language_assets_dir = assets_dir / language
        srt_path = language_assets_dir / f"{language}.srt"
        concat_path = language_assets_dir / f"{language}_concat.txt"
        output_path = final_video_path(videos_dir, run_dir, language)
        captions_dir = language_assets_dir / "captions"
        language_assets_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        srt_path.write_text(build_srt(language_scenes), encoding="utf-8")
        concat_path.write_text(build_concat_file(clip_paths), encoding="utf-8")
        caption_images = write_caption_images(captions_dir, language_scenes)

        command = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_path),
        ]
        for image_path in caption_images:
            command.extend(["-loop", "1", "-i", str(image_path)])

        if has_ffmpeg_filter("subtitles"):
            command.extend([
                "-vf", _subtitles_filter(srt_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
            ])
        elif has_ffmpeg_filter("overlay"):
            filter_complex, video_label = build_overlay_filter(language_scenes)
            command.extend([
                "-filter_complex", filter_complex,
                "-map", video_label, "-map", "0:a?",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
            ])
        else:
            command.extend(["-c", "copy"])
        command.append(str(output_path))
        subprocess.run(command, check=True)
        rendered.append(RenderedVideo(language=language, path=output_path))

    return rendered


def render_shared_language_videos(
    *,
    run_dir: Path,
    visual_scenes: list[dict[str, Any]],
    localized_tracks: dict[str, list[dict[str, Any]]],
    languages: set[str] | None = None,
    video_output_dir: Path | None = None,
) -> list[RenderedVideo]:
    """Render final videos, preferring per-scene audio sync over combined audio.

    Per-scene audio (audio/{language}/scene_01.wav, …) is the preferred path:
    - Each Veo clip is extended (freeze last frame) to match its scene's TTS duration
    - Subtitles are timed to actual TTS audio, not fixed scene durations
    - No audio tempo compression — TTS plays at natural speed

    Falls back to combined audio (audio/{language}.wav) for backward compat.
    """
    ensure_ffmpeg()

    rendered: list[RenderedVideo] = []
    videos_dir = video_output_dir or run_dir / "videos"
    assets_dir = run_dir / "render_assets"
    videos_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    for language, language_scenes in localized_tracks.items():
        if languages is not None and language not in languages:
            continue

        scene_lookup = {scene["scene_id"]: scene for scene in visual_scenes}
        selected_visuals = [scene_lookup[scene["scene_id"]] for scene in language_scenes]
        clip_paths = [
            run_dir / "veo" / "shared" / f"{scene['scene_id']}.mp4"
            for scene in selected_visuals
        ]
        missing = [p for p in clip_paths if not p.exists()]
        if missing:
            raise RenderError(f"Missing shared Veo clip: {missing[0]}")

        language_assets_dir = assets_dir / language
        language_assets_dir.mkdir(parents=True, exist_ok=True)
        output_path = final_video_path(videos_dir, run_dir, language)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer per-scene audio for frame-accurate sync
        per_scene_audio = _collect_per_scene_audio(run_dir / "audio" / language, language_scenes)
        if per_scene_audio:
            rendered.append(_render_per_scene_sync(
                language=language,
                language_scenes=language_scenes,
                clip_paths=clip_paths,
                per_scene_audio=per_scene_audio,
                assets_dir=language_assets_dir,
                output_path=output_path,
            ))
        else:
            # Fallback: combined audio with tempo adjustment
            combined_audio = run_dir / "audio" / f"{language}.wav"
            if not combined_audio.exists():
                raise RenderError(f"Missing TTS audio for {language}: {combined_audio}")
            rendered.append(_render_combined_audio(
                language=language,
                language_scenes=language_scenes,
                clip_paths=clip_paths,
                audio_path=combined_audio,
                assets_dir=language_assets_dir,
                output_path=output_path,
            ))

    return rendered


# ─────────────────────────────────────────────────────────────────────────────
# Per-scene sync render (primary path)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_per_scene_audio(
    lang_audio_dir: Path,
    language_scenes: list[dict[str, Any]],
) -> dict[str, Path] | None:
    """Return {scene_id: wav_path} if all per-scene audio files exist, else None."""
    result: dict[str, Path] = {}
    for scene in language_scenes:
        scene_id = scene["scene_id"]
        candidate = lang_audio_dir / f"{scene_id}.wav"
        if not candidate.exists():
            return None
        result[scene_id] = candidate
    return result or None


def _render_per_scene_sync(
    *,
    language: str,
    language_scenes: list[dict[str, Any]],
    clip_paths: list[Path],
    per_scene_audio: dict[str, Path],
    assets_dir: Path,
    output_path: Path,
) -> RenderedVideo:
    """Render using per-scene audio for frame-accurate TTS sync.

    Strategy:
    1. Probe each scene's audio duration.
    2. Extend each Veo clip to match (freeze last frame) using ffmpeg tpad.
    3. Build subtitle SRT where timestamps come from actual audio durations.
    4. Concatenate extended clips (video only) + concatenated audio.
    5. Overlay caption images timed to audio durations.
    """
    extended_dir = assets_dir / "extended_clips"
    extended_dir.mkdir(parents=True, exist_ok=True)
    captions_dir = assets_dir / "captions"

    # Step 1: measure audio durations and determine effective scene durations
    audio_durations: list[float] = []
    for scene, clip_path in zip(language_scenes, clip_paths):
        scene_id = scene["scene_id"]
        audio_dur = probe_media_duration(per_scene_audio[scene_id]) or float(scene.get("duration_seconds", 6))
        audio_durations.append(audio_dur)

    # Step 2: extend each clip to its audio duration (freeze last frame, strip audio)
    extended_paths: list[Path] = []
    for scene, clip_path, audio_dur in zip(language_scenes, clip_paths, audio_durations):
        scene_id = scene["scene_id"]
        extended = extended_dir / f"{scene_id}.mp4"
        clip_dur = probe_media_duration(clip_path) or float(scene.get("duration_seconds", 6))
        _extend_clip(clip_path, extended, target_duration=audio_dur, clip_duration=clip_dur)
        extended_paths.append(extended)

    # Step 3: build scenes-with-audio-durations for SRT + caption timing
    audio_timed_scenes = [
        {**scene, "duration_seconds": dur}
        for scene, dur in zip(language_scenes, audio_durations)
    ]

    srt_path = assets_dir / f"{language}.srt"
    srt_path.write_text(build_srt_float(audio_timed_scenes), encoding="utf-8")

    concat_path = assets_dir / f"{language}_concat.txt"
    concat_path.write_text(build_concat_file(extended_paths), encoding="utf-8")

    caption_images = write_caption_images(captions_dir, audio_timed_scenes)

    # Step 4: concatenate per-scene audio files
    combined_audio_path = assets_dir / f"{language}_combined.wav"
    _concatenate_wav_files(list(per_scene_audio[s["scene_id"]] for s in language_scenes), combined_audio_path)

    # Step 5: assemble final MP4 with xfade transitions between clips
    total_audio_duration = sum(audio_durations)
    _assemble_with_xfade(
        clip_paths=extended_paths,
        audio_path=combined_audio_path,
        caption_images=caption_images,
        audio_durations=audio_durations,
        output_path=output_path,
        total_audio_duration=total_audio_duration,
    )
    print(f"  {language}: {total_audio_duration:.1f}s video — per-scene xfade sync")
    return RenderedVideo(language=language, path=output_path)


def _extend_clip(clip_path: Path, output_path: Path, *, target_duration: float, clip_duration: float) -> None:
    """Extend a clip to target_duration by freezing its last frame.

    Strips audio from the Veo clip (TTS audio is provided separately).
    If the clip is already long enough, just strips audio and copies.
    """
    extra = max(0.0, target_duration - clip_duration)
    if extra < 0.05:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(clip_path),
            "-an",
            "-c:v", "copy",
            str(output_path),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(clip_path),
            "-vf", f"tpad=stop_mode=clone:stop_duration={extra:.3f}",
            "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(output_path),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _concatenate_wav_files(wav_paths: list[Path], output_path: Path) -> None:
    """Concatenate WAV files using ffmpeg concat demuxer."""
    list_path = output_path.parent / "audio_list.txt"
    list_path.write_text(
        "\n".join(f"file '{_escape_concat_path(p)}'" for p in wav_paths) + "\n",
        encoding="utf-8",
    )
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(output_path),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _assemble_with_xfade(
    *,
    clip_paths: list[Path],
    audio_path: Path,
    caption_images: list[Path],
    audio_durations: list[float],
    output_path: Path,
    total_audio_duration: float,
) -> None:
    """Merge video clips with xfade transitions + TTS audio + caption overlays.

    Clips are cross-faded with a short fade transition to eliminate hard cuts
    and make freeze-frame extensions invisible at scene boundaries.
    """
    n = len(clip_paths)
    xfade_dur = XFADE_DURATION if n > 1 else 0.0

    command = ["ffmpeg", "-y"]
    for clip in clip_paths:
        command.extend(["-i", str(clip)])
    command.extend(["-i", str(audio_path)])
    for img in caption_images:
        command.extend(["-loop", "1", "-i", str(img)])

    audio_input = n
    img_input_start = n + 1
    parts: list[str] = []

    # Build xfade chain between all clips
    current = "0:v"
    xfade_offset_acc = 0.0
    for i in range(1, n):
        xfade_offset_acc += audio_durations[i - 1] - xfade_dur
        out = f"xf{i}"
        parts.append(
            f"[{current}][{i}:v]xfade=transition=fade"
            f":duration={xfade_dur:.3f}:offset={xfade_offset_acc:.3f}[{out}]"
        )
        current = out

    # Build caption overlay chain on top of xfade result.
    # Caption timing uses cumulative audio durations (not xfade-adjusted) to prevent
    # adjacent captions from overlapping when the xfade compresses the video timeline.
    caption_cursor = 0.0
    for idx, dur in enumerate(audio_durations):
        caption_start = caption_cursor
        caption_end = caption_cursor + dur
        img_input = img_input_start + idx
        out = f"v{idx + 1}"
        parts.append(
            f"[{current}][{img_input}:v]overlay=0:0"
            f":enable='between(t,{caption_start:.3f},{caption_end:.3f})':format=auto[{out}]"
        )
        current = out
        caption_cursor += dur  # advance by full audio duration — no xfade offset

    command.extend([
        "-filter_complex", ";".join(parts),
        "-map", f"[{current}]",
        "-map", f"{audio_input}:a",
        "-t", f"{total_audio_duration:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(output_path),
    ])
    subprocess.run(command, check=True)


# ─────────────────────────────────────────────────────────────────────────────
# Combined-audio fallback render (backward compat)
# ─────────────────────────────────────────────────────────────────────────────

def _render_combined_audio(
    *,
    language: str,
    language_scenes: list[dict[str, Any]],
    clip_paths: list[Path],
    audio_path: Path,
    assets_dir: Path,
    output_path: Path,
) -> RenderedVideo:
    srt_path = assets_dir / f"{language}.srt"
    concat_path = assets_dir / f"{language}_concat.txt"
    captions_dir = assets_dir / "captions"

    srt_path.write_text(build_srt(language_scenes), encoding="utf-8")
    concat_path.write_text(build_concat_file(clip_paths), encoding="utf-8")
    caption_images = write_caption_images(captions_dir, language_scenes)

    target_duration = total_scene_duration(language_scenes)
    audio_duration = probe_media_duration(audio_path)
    audio_filter = build_audio_fit_filter(audio_duration, target_duration)

    command = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-i", str(audio_path),
    ]
    for img in caption_images:
        command.extend(["-loop", "1", "-i", str(img)])

    filter_complex, video_label = build_overlay_filter_with_offset(language_scenes, image_input_offset=2)
    command.extend([
        "-filter_complex", filter_complex,
        "-map", video_label,
        "-map", "1:a",
    ])
    if audio_filter:
        command.extend(["-af", audio_filter])
    command.extend([
        "-t", f"{target_duration:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
    ])
    command.append(str(output_path))
    subprocess.run(command, check=True)
    print(f"  {language}: {target_duration}s video — combined audio fallback")
    return RenderedVideo(language=language, path=output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Subtitle / caption builders
# ─────────────────────────────────────────────────────────────────────────────

def build_srt(scenes: list[dict[str, Any]]) -> str:
    entries: list[str] = []
    cursor = 0.0
    for index, scene in enumerate(scenes, start=1):
        duration = float(scene.get("duration_seconds", 6))
        start = cursor
        end = cursor + duration
        cursor = end
        entries.extend([
            str(index),
            f"{_srt_time_float(start)} --> {_srt_time_float(end)}",
            str(scene.get("subtitle_text") or scene.get("caption_text") or "").strip(),
            "",
        ])
    return "\n".join(entries)


def build_srt_float(scenes: list[dict[str, Any]]) -> str:
    """Build SRT with float-precision durations (for per-scene audio timing)."""
    return build_srt(scenes)


def write_caption_images(captions_dir: Path, scenes: list[dict[str, Any]]) -> list[Path]:
    """Render clean YouTube-Shorts style captions: white bold text + dark outline.

    No background box — outlined text sits directly on video.
    Eliminates the dark pill shadow and avoids alpha-accumulation artifacts
    when chaining overlay filters across scenes.
    """
    captions_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for index, scene in enumerate(scenes, start=1):
        text = str(scene.get("subtitle_text") or scene.get("caption_text") or "").strip()
        image = Image.new("RGBA", (720, 1280), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        active_font, lines, line_height = fit_caption_text(draw, text, max_width=640, max_lines=4)
        total_text_height = len(lines) * line_height
        y = 1280 - total_text_height - 80

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=active_font)
            line_width = bbox[2] - bbox[0]
            x = (720 - line_width) / 2

            # 8-direction outline for solid-looking stroke without background box
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, -2), (-2, 2), (2, 2)):
                draw.text((x + dx, y + dy), line, font=active_font, fill=(0, 0, 0, 210))
            draw.text((x, y), line, font=active_font, fill=(255, 255, 255, 255))
            y += line_height

        path = captions_dir / f"caption_{index:02d}.png"
        image.save(path)
        paths.append(path)

    return paths


def fit_caption_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    max_width: int,
    max_lines: int,
) -> tuple[ImageFont.ImageFont, list[str], int]:
    for size, line_height in ((38, 50), (34, 46), (30, 42), (26, 38), (22, 34)):
        font = load_caption_font(size=size)
        lines = wrap_text(draw, text, font, max_width=max_width)
        if len(lines) <= max_lines and all(text_width(draw, line, font) <= max_width for line in lines):
            return font, lines, line_height

    font = load_caption_font(size=20)
    lines = wrap_text(draw, text, font, max_width=max_width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = truncate_to_width(draw, lines[-1] + "...", font, max_width)
    return font, lines, 30


# ─────────────────────────────────────────────────────────────────────────────
# FFmpeg filter builders
# ─────────────────────────────────────────────────────────────────────────────

def build_overlay_filter(scenes: list[dict[str, Any]]) -> tuple[str, str]:
    return build_overlay_filter_with_offset(scenes, image_input_offset=1)


def build_overlay_filter_with_offset(
    scenes: list[dict[str, Any]], *, image_input_offset: int
) -> tuple[str, str]:
    """Build chained overlay filter with proper RGBA alpha handling.

    format=auto ensures FFmpeg respects the PNG alpha channel instead of
    treating the overlay as opaque, preventing ghost / shadow accumulation
    across the filter chain.
    """
    parts: list[str] = []
    current = "0:v"
    cursor = 0.0
    for index, scene in enumerate(scenes, start=1):
        duration = float(scene.get("duration_seconds", 6))
        start = cursor
        end = cursor + duration
        cursor = end
        output = f"v{index}"
        image_input = image_input_offset + index - 1
        parts.append(
            f"[{current}][{image_input}:v]overlay=0:0"
            f":enable='between(t,{start:.3f},{end:.3f})':format=auto[{output}]"
        )
        current = output
    return ";".join(parts), f"[{current}]"


# ─────────────────────────────────────────────────────────────────────────────
# Path / file helpers
# ─────────────────────────────────────────────────────────────────────────────

def final_video_path(videos_dir: Path, run_dir: Path, language: str) -> Path:
    source = run_dir / "source.json"
    row_label = run_dir.name
    if source.exists():
        try:
            import json
            row_number = json.loads(source.read_text(encoding="utf-8")).get("row_number")
            if row_number:
                row_label = f"Row_{row_number}"
        except Exception:
            pass
    # All files go flat into videos_dir — no language subfolders, all-English names.
    file_name = f"{row_label}_{language}.mp4"
    return videos_dir / file_name


def language_folder_name(language: str) -> str:
    return language


def build_concat_file(clip_paths: list[Path]) -> str:
    return "\n".join(f"file '{_escape_concat_path(path)}'" for path in clip_paths) + "\n"


def total_scene_duration(scenes: list[dict[str, Any]]) -> int:
    return sum(int(scene.get("duration_seconds", 6)) for scene in scenes)


def probe_media_duration(path: Path) -> float | None:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    try:
        return float(probe.stdout.strip())
    except ValueError:
        return None


def build_audio_fit_filter(audio_duration: float | None, target_duration: float) -> str | None:
    if not audio_duration or audio_duration <= target_duration * 1.01:
        return None
    ratio = audio_duration / target_duration
    parts: list[str] = []
    while ratio > 2.0:
        parts.append("atempo=2.0")
        ratio /= 2.0
    while ratio < 0.5:
        parts.append("atempo=0.5")
        ratio /= 0.5
    parts.append(f"atempo={ratio:.3f}")
    return ",".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Text wrapping / font helpers
# ─────────────────────────────────────────────────────────────────────────────

def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = split_caption_units(text)
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    original_has_spaces = " " in text
    for word in words[1:]:
        separator = "" if is_cjk_caption(current + word) and not original_has_spaces else " "
        candidate = f"{current}{separator}{word}"
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            if text_width(draw, current, font) > max_width:
                lines.extend(break_long_unit(draw, current, font, max_width))
            else:
                lines.append(current)
            current = word
    if text_width(draw, current, font) > max_width:
        lines.extend(break_long_unit(draw, current, font, max_width))
    else:
        lines.append(current)
    return lines


def split_caption_units(text: str) -> list[str]:
    if not text:
        return []
    if " " not in text and is_cjk_caption(text):
        return list(text)
    units: list[str] = []
    for word in text.split():
        units.extend(break_cjk_mixed_word(word))
    return units


def break_cjk_mixed_word(word: str) -> list[str]:
    if is_cjk_caption(word) and len(word) > 8:
        return list(word)
    return [word]


def is_cjk_caption(text: str) -> bool:
    return any("㄰" <= char <= "㆏" or "가" <= char <= "힣" for char in text)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def break_long_unit(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and text_width(draw, candidate, font) > max_width:
            parts.append(current)
            current = char
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts or [text]


def truncate_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    while text and text_width(draw, text, font) > max_width:
        base = text[:-4].rstrip()
        text = f"{base}..." if base else "..."
    return text


def load_caption_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default(size=size)


# ─────────────────────────────────────────────────────────────────────────────
# FFmpeg capability detection
# ─────────────────────────────────────────────────────────────────────────────

def has_ffmpeg_filter(name: str) -> bool:
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return any(line.split()[1:2] == [name] for line in probe.stdout.splitlines())


# ─────────────────────────────────────────────────────────────────────────────
# String / path escaping
# ─────────────────────────────────────────────────────────────────────────────

def _srt_time_float(seconds: float) -> str:
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    secs = int(seconds) % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _escape_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def _escape_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _subtitles_filter(path: Path) -> str:
    style = "FontSize=18,Alignment=2,MarginV=70,Outline=2"
    return f"subtitles=filename='{_escape_filter_path(path)}':force_style='{style}'"


# ─────────────────────────────────────────────────────────────────────────────
# Meme slideshow render (photo → Ken Burns → xfade → optional audio)
# ─────────────────────────────────────────────────────────────────────────────

MEME_VIDEO_SIZE = "1080x1920"
MEME_FPS = 25
MEME_SPEED_FACTOR = 1.15  # TTS playback speedup; clips are also shortened proportionally


def _frame_align(duration: float, fps: int = MEME_FPS) -> float:
    """Round duration to an exact frame boundary to eliminate timestamp drift."""
    frames = max(1, round(duration * fps))
    return frames / fps


def render_meme_slideshow(
    *,
    run_dir: Path,
    meme_plan: dict,
    image_dir: Path,
    video_output_dir: Path,
    languages: set[str] | None = None,
    zoom_enabled: bool = False,
    bgm_dir: Path | None = None,
) -> list[RenderedVideo]:
    """Assemble meme slideshow MP4s from generated PNGs.

    Pipeline per language:
      1. Bake top/bottom meme text onto each PNG (PIL)
      2. Probe actual TTS audio duration per scene → use as clip length (sync fix)
      3. Convert each baked PNG to a static or subtle-zoom MP4 clip
      4. Concatenate clips with xfade crossfade + TTS audio + fade-to-black ending
    """
    ensure_ffmpeg()
    rendered: list[RenderedVideo] = []
    video_output_dir.mkdir(parents=True, exist_ok=True)

    for lang_key in ("english", "korean", "spanish"):
        if languages is not None and lang_key not in languages:
            continue
        lang_plan = meme_plan.get(lang_key)
        if not lang_plan:
            continue

        scenes = lang_plan.get("scenes", [])
        lang_image_dir = image_dir / lang_key
        if not lang_image_dir.exists():
            raise RenderError(
                f"Missing meme images for {lang_key}. "
                f"Run `python3 -m modoc_pipeline imagen --run {run_dir}` first."
            )

        assets_dir = run_dir / "meme_render_assets" / lang_key
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: bake meme text onto images
        baked_dir = assets_dir / "baked"
        baked_images = bake_meme_text_onto_images(
            scenes=scenes,
            raw_images_dir=lang_image_dir,
            baked_dir=baked_dir,
            language=lang_key,
        )
        if not baked_images:
            raise RenderError(f"No baked images produced for {lang_key}.")

        active_scenes = scenes[:len(baked_images)]

        # Step 2: resolve per-scene durations from actual TTS audio (sync fix)
        # If TTS exists, each clip length = actual audio duration (+ small tail buffer).
        # Fallback to plan's duration_seconds if no audio found.
        scene_durations = _resolve_scene_durations(
            run_dir=run_dir,
            lang_key=lang_key,
            scenes=active_scenes,
        )

        # Step 3: convert baked PNGs to MP4 clips (static by default, subtle zoom optional)
        clips_dir = assets_dir / "zoompan_clips"
        clips = _build_zoompan_clips(
            baked_images=baked_images,
            scene_durations=scene_durations,
            clips_dir=clips_dir,
            zoom_enabled=zoom_enabled,
        )

        # Step 4: concatenate per-scene audio, assemble with fade-to-black ending
        audio_path = _find_meme_audio(run_dir=run_dir, lang_key=lang_key, scenes=active_scenes, assets_dir=assets_dir)

        # Look for language-specific BGM from Lyria
        bgm_path: Path | None = None
        if bgm_dir:
            for ext in (".mp3", ".wav", ".m4a"):
                candidate = bgm_dir / f"{lang_key}_bgm{ext}"
                if candidate.exists():
                    bgm_path = candidate
                    break

        output_path = _meme_video_path(video_output_dir, run_dir, lang_key)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _assemble_meme_video(
            clips=clips,
            audio_path=audio_path,
            scene_durations=scene_durations,
            output_path=output_path,
            bgm_path=bgm_path,
        )
        total = sum(scene_durations)
        bgm_label = " + BGM" if bgm_path else ""
        audio_label = f"with audio{bgm_label}" if audio_path else "silent"
        print(f"  {lang_key}: {total:.1f}s meme slideshow ({audio_label}) → {output_path.name}")
        rendered.append(RenderedVideo(language=lang_key, path=output_path))

    return rendered


def bake_meme_text_onto_images(
    *,
    scenes: list[dict],
    raw_images_dir: Path,
    baked_dir: Path,
    language: str,
) -> list[Path]:
    """Render top_text / bottom_text onto each scene PNG (Impact meme style).

    Text is baked permanently into the image (not an FFmpeg overlay layer),
    matching real meme aesthetics. Returns only paths where source PNG exists.
    """
    baked_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for scene in scenes:
        scene_id = scene.get("scene_id", "")
        src = raw_images_dir / f"{scene_id}.png"
        if not src.exists():
            continue

        img = Image.open(src).convert("RGBA")
        img = _fit_to_meme_canvas(img)
        draw = ImageDraw.Draw(img)

        top_text = str(scene.get("top_text", "")).strip()
        bottom_text = str(scene.get("bottom_text", "")).strip()

        if top_text:
            _draw_meme_text(draw, top_text, position="top", language=language)
        if bottom_text:
            _draw_meme_text(draw, bottom_text, position="bottom", language=language)

        # Convert to RGB for JPEG-safe PNG output
        final = Image.new("RGB", img.size, (0, 0, 0))
        final.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
        out_path = baked_dir / f"{scene_id}.png"
        final.save(out_path, "PNG")
        paths.append(out_path)

    return paths


def _fit_to_meme_canvas(img: Image.Image) -> Image.Image:
    """Resize and center-crop to 1080×1920 (9:16)."""
    target_w, target_h = 1080, 1920
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


_CANVAS_W = 1080
_CANVAS_H = 1920
_TEXT_MARGIN = 50     # px from edge
_TEXT_SAFE_TOP = 40   # minimum y for top text
_TEXT_SAFE_BOTTOM = _CANVAS_H - _TEXT_MARGIN  # maximum y+height for bottom text


def _draw_meme_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    position: str,
    language: str,
) -> None:
    """Draw Impact-style meme text with bounds-checked placement.

    Shrinks font until all lines fit within the safe zone for the given position.
    Never draws outside the canvas — clips to _TEXT_SAFE_BOTTOM / _TEXT_SAFE_TOP.
    """
    max_text_w = _CANVAS_W - 80

    # Auto-size: start at 80px, shrink until fits in ≤3 lines within safe zone
    font_size = 80
    font = _load_meme_font(size=font_size, language=language)
    lines = wrap_text(draw, text, font, max_width=max_text_w)

    while font_size > 32:
        line_height = int(font_size * 1.25)
        total_h = len(lines) * line_height
        fits_lines = len(lines) <= 3
        # Check that the block fits in its safe zone
        if position == "top":
            fits_space = (_TEXT_SAFE_TOP + total_h) < (_CANVAS_H // 2 - 50)
        else:
            fits_space = (_CANVAS_H - total_h - _TEXT_MARGIN) >= (_CANVAS_H // 2 + 50)
        if fits_lines and fits_space:
            break
        font_size -= 6
        font = _load_meme_font(size=font_size, language=language)
        lines = wrap_text(draw, text, font, max_width=max_text_w)
        if len(lines) > 3:
            lines = lines[:3]

    line_height = int(font_size * 1.25)
    total_h = len(lines) * line_height

    if position == "top":
        y = _TEXT_SAFE_TOP
    else:
        y = _CANVAS_H - total_h - _TEXT_MARGIN

    for line in lines:
        # Bounds guard: skip lines that would render outside the canvas
        if y + line_height > _CANVAS_H or y < 0:
            break

        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        x = (_CANVAS_W - line_w) // 2

        # 8-direction black outline for legibility on any background
        for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3), (-3, -3), (3, -3), (-3, 3), (3, 3)):
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height


def _load_meme_font(size: int, language: str) -> ImageFont.ImageFont:
    """Load a bold font suitable for meme text; CJK-aware for Korean."""
    cjk_candidates = [
        "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Bold.otf",
    ]
    latin_bold_candidates = [
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/Library/Fonts/Impact.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    ]
    fallback_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]

    candidates = (cjk_candidates if language == "korean" else latin_bold_candidates) + fallback_candidates
    for path in candidates:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default(size=size)


def _resolve_scene_durations(
    *,
    run_dir: Path,
    lang_key: str,
    scenes: list[dict],
) -> list[float]:
    """Return per-scene clip durations adjusted for TTS speed and frame boundaries.

    Formula: clip_dur = (audio_dur + 0.3s_tail) / MEME_SPEED_FACTOR

    Dividing by MEME_SPEED_FACTOR means clips are proportionally shorter,
    perfectly matching the atempo-sped-up audio applied during assembly.
    """
    audio_base = run_dir / "meme_audio" / lang_key
    durations: list[float] = []
    for scene in scenes:
        scene_id = scene.get("scene_id", "")
        resolved = False
        for ext in (".wav", ".mp3"):
            candidate = audio_base / f"{scene_id}{ext}"
            if candidate.exists():
                actual = probe_media_duration(candidate)
                if actual and actual > 0:
                    durations.append(_frame_align((actual + 0.3) / MEME_SPEED_FACTOR))
                    resolved = True
                    break
        if not resolved:
            durations.append(_frame_align(float(scene.get("duration_seconds", 4.0))))
    return durations


def _build_zoompan_clips(
    *,
    baked_images: list[Path],
    scene_durations: list[float],
    clips_dir: Path,
    zoom_enabled: bool,
) -> list[Path]:
    """Convert each baked PNG into a precisely-framed MP4 clip.

    Uses -frames:v (exact frame count) instead of -t (float seconds) so every
    clip has a precise, deterministic duration aligned to frame boundaries.
    This eliminates the timestamp drift that causes xfade shimmer/shake.

    zoom_enabled=False (default): sharp static image.
    zoom_enabled=True: very subtle 1.04x zoom (only if source is large enough).
    """
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []

    for img_path, dur in zip(baked_images, scene_durations):
        clip_path = clips_dir / img_path.with_suffix(".mp4").name
        n_frames = max(1, round(dur * MEME_FPS))  # exact, matches _frame_align

        # 0.15s fade-in per clip gives a smooth visual entry at each cut point.
        # This replaces xfade (which caused black screen due to offset miscalculation).
        fade_in = f",fade=t=in:st=0:d=0.15"
        if zoom_enabled:
            vf = (
                f"scale=1200:2133:flags=lanczos,"
                f"zoompan=z='min(zoom+0.0003,1.04)':d={n_frames}"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={MEME_VIDEO_SIZE},"
                f"fps={MEME_FPS}{fade_in}"
            )
        else:
            vf = f"scale={MEME_VIDEO_SIZE}:flags=lanczos,fps={MEME_FPS}{fade_in}"

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-r", str(MEME_FPS),  # force constant input framerate
                "-i", str(img_path),
                "-vf", vf,
                "-frames:v", str(n_frames),          # exact frame count, no float rounding
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-g", str(MEME_FPS),                 # keyframe every second for clean xfade cuts
                str(clip_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        clips.append(clip_path)

    return clips


def _find_meme_audio(
    *,
    run_dir: Path,
    lang_key: str,
    scenes: list[dict],
    assets_dir: Path,
) -> Path | None:
    """Concatenate per-scene MP3/WAV files into one WAV for FFmpeg. Returns None if no audio found."""
    audio_base = run_dir / "meme_audio" / lang_key
    scene_audio: list[Path] = []

    for scene in scenes:
        scene_id = scene.get("scene_id", "")
        for ext in (".mp3", ".wav"):
            candidate = audio_base / f"{scene_id}{ext}"
            if candidate.exists():
                scene_audio.append(candidate)
                break

    if not scene_audio:
        return None

    combined = assets_dir / f"{lang_key}_combined_audio.wav"
    list_file = assets_dir / "audio_list.txt"
    list_file.write_text(
        "\n".join(f"file '{_escape_concat_path(p)}'" for p in scene_audio) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(combined),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return combined if combined.exists() else None


def _assemble_meme_video(
    *,
    clips: list[Path],
    audio_path: Path | None,
    scene_durations: list[float],
    output_path: Path,
    bgm_path: Path | None = None,
) -> None:
    """Concatenate clips via concat demuxer + optional atempo-sped audio + optional BGM.

    Per-clip fade-in is already baked into each clip by _build_zoompan_clips.
    Text is baked into images by bake_meme_text_onto_images — no overlay needed.
    """
    if audio_path:
        _assemble_meme_concat_audio(
            clips=clips,
            audio_path=audio_path,
            scene_durations=scene_durations,
            output_path=output_path,
            bgm_path=bgm_path,
        )
        return

    # Silent assembly: simple concat, no audio track
    total_dur = sum(scene_durations)
    fade_out_start = max(0.0, total_dur - 0.4)
    concat_file = output_path.parent / f"{output_path.stem}_concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{_escape_concat_path(c)}'" for c in clips) + "\n",
        encoding="utf-8",
    )
    fcomplex = f"[0:v]fade=t=out:st={fade_out_start:.6f}:d=0.4[vout]"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-filter_complex", fcomplex,
            "-map", "[vout]",
            "-t", f"{total_dur:.6f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(output_path),
        ],
        check=True,
    )


def _assemble_meme_concat_audio(
    *,
    clips: list[Path],
    audio_path: Path,
    scene_durations: list[float],
    output_path: Path,
    bgm_path: Path | None = None,
    bgm_volume: float = 0.32,
) -> None:
    """Assemble meme video: concat demuxer + atempo TTS + optional Lyria BGM mix + fade-to-black.

    If bgm_path is provided, mixes BGM at bgm_volume under the TTS voiceover.
    BGM fades in over 0.3s so it punches in immediately without a hard start.
    BGM is looped so it always covers the full video duration.
    Both TTS and BGM fade out smoothly at the end.
    """
    total_dur = sum(scene_durations)
    fade_out_start = max(0.0, total_dur - 0.4)

    concat_file = output_path.parent / f"{output_path.stem}_concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{_escape_concat_path(c)}'" for c in clips) + "\n",
        encoding="utf-8",
    )

    if bgm_path and bgm_path.exists():
        # Video: input 0 (concat clips), TTS: input 1, BGM: input 2 (looped)
        fcomplex = (
            f"[0:v]fade=t=out:st={fade_out_start:.6f}:d=0.4[vout];"
            f"[1:a]atempo={MEME_SPEED_FACTOR}[tts_fast];"
            f"[2:a]afade=t=in:st=0:d=0.3,volume={bgm_volume},afade=t=out:st={fade_out_start:.6f}:d=0.5[bgm_faded];"
            f"[tts_fast][bgm_faded]amix=inputs=2:duration=first:normalize=0[aout]"
        )
        command = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(audio_path),
            "-stream_loop", "-1", "-i", str(bgm_path),
            "-filter_complex", fcomplex,
            "-map", "[vout]", "-map", "[aout]",
            "-t", f"{total_dur:.6f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(output_path),
        ]
    else:
        # No BGM: TTS only with atempo speedup
        fcomplex = f"[0:v]fade=t=out:st={fade_out_start:.6f}:d=0.4[vout]"
        command = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(audio_path),
            "-filter_complex", fcomplex,
            "-map", "[vout]", "-map", "1:a",
            "-af", f"atempo={MEME_SPEED_FACTOR}",
            "-t", f"{total_dur:.6f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(output_path),
        ]

    subprocess.run(command, check=True)


def _meme_video_path(videos_dir: Path, run_dir: Path, language: str) -> Path:
    source = run_dir / "source.json"
    row_label = run_dir.name
    if source.exists():
        try:
            import json
            row_number = json.loads(source.read_text(encoding="utf-8")).get("row_number")
            if row_number:
                row_label = f"Row_{row_number}"
        except Exception:
            pass
    return videos_dir / f"{row_label}_{language}_meme.mp4"
