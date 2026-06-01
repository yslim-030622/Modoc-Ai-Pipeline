"""FFmpeg rendering helpers for final vertical MP4 assembly."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


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

    # Step 5: assemble final MP4
    total_duration = sum(audio_durations)
    _assemble_final_mp4(
        concat_path=concat_path,
        audio_path=combined_audio_path,
        caption_images=caption_images,
        audio_timed_scenes=audio_timed_scenes,
        output_path=output_path,
        total_duration=total_duration,
        image_input_offset=2,
    )
    print(f"  {language}: {total_duration:.1f}s video — per-scene sync")
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


def _assemble_final_mp4(
    *,
    concat_path: Path,
    audio_path: Path,
    caption_images: list[Path],
    audio_timed_scenes: list[dict[str, Any]],
    output_path: Path,
    total_duration: float,
    image_input_offset: int,
) -> None:
    """Merge extended video clips + TTS audio + caption overlays into final MP4."""
    command = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-i", str(audio_path),
    ]
    for img in caption_images:
        command.extend(["-loop", "1", "-i", str(img)])

    filter_complex, video_label = build_overlay_filter_with_offset(
        audio_timed_scenes, image_input_offset=image_input_offset
    )
    command.extend([
        "-filter_complex", filter_complex,
        "-map", video_label,
        "-map", "1:a",
        "-t", f"{total_duration:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
    ])
    command.append(str(output_path))
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
    for size, line_height in ((46, 58), (42, 54), (38, 50), (34, 46), (30, 42)):
        font = load_caption_font(size=size)
        lines = wrap_text(draw, text, font, max_width=max_width)
        if len(lines) <= max_lines and all(text_width(draw, line, font) <= max_width for line in lines):
            return font, lines, line_height

    font = load_caption_font(size=26)
    lines = wrap_text(draw, text, font, max_width=max_width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = truncate_to_width(draw, lines[-1] + "...", font, max_width)
    return font, lines, 38


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
