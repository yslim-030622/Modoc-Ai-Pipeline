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


def render_language_videos(
    *,
    run_dir: Path,
    scenes: dict[str, list[dict[str, Any]]],
) -> list[RenderedVideo]:
    ensure_ffmpeg()

    rendered: list[RenderedVideo] = []
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

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

        srt_path = videos_dir / f"{language}.srt"
        concat_path = videos_dir / f"{language}_concat.txt"
        output_path = videos_dir / f"{language}.mp4"
        captions_dir = videos_dir / f"{language}_captions"

        srt_path.write_text(build_srt(language_scenes), encoding="utf-8")
        concat_path.write_text(build_concat_file(clip_paths), encoding="utf-8")
        caption_images = write_caption_images(captions_dir, language_scenes)

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
        ]
        for image_path in caption_images:
            command.extend(["-loop", "1", "-i", str(image_path)])

        if has_ffmpeg_filter("subtitles"):
            # Re-encode video because burning subtitles is a video filter. Audio
            # is preserved as AAC so the generated Veo sound stays in the file.
            command.extend(
                [
                    "-vf",
                    _subtitles_filter(srt_path),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                ]
            )
        elif has_ffmpeg_filter("overlay"):
            filter_complex, video_label = build_overlay_filter(language_scenes)
            command.extend(
                [
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    video_label,
                    "-map",
                    "0:a?",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                ]
            )
        else:
            # Extremely minimal FFmpeg builds may lack both subtitles and
            # overlay. In that case we still produce a valid MP4 and keep the
            # SRT sidecar, rather than blocking the whole pipeline.
            command.extend(["-c", "copy"])
        command.append(str(output_path))
        subprocess.run(command, check=True)
        rendered.append(RenderedVideo(language=language, path=output_path))

    return rendered


def build_srt(scenes: list[dict[str, Any]]) -> str:
    entries: list[str] = []
    cursor = 0
    for index, scene in enumerate(scenes, start=1):
        duration = int(scene.get("duration_seconds", 6))
        start = cursor
        end = cursor + duration
        cursor = end
        entries.extend(
            [
                str(index),
                f"{_srt_time(start)} --> {_srt_time(end)}",
                str(scene.get("subtitle_text", "")).strip(),
                "",
            ]
        )
    return "\n".join(entries)


def build_concat_file(clip_paths: list[Path]) -> str:
    return "\n".join(f"file '{_escape_concat_path(path)}'" for path in clip_paths) + "\n"


def write_caption_images(captions_dir: Path, scenes: list[dict[str, Any]]) -> list[Path]:
    captions_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    font = load_caption_font(size=42)
    small_font = load_caption_font(size=34)

    for index, scene in enumerate(scenes, start=1):
        text = str(scene.get("subtitle_text", "")).strip()
        image = Image.new("RGBA", (720, 1280), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        lines = wrap_text(draw, text, font, max_width=600)
        line_height = 54
        box_padding_x = 34
        box_padding_y = 24
        box_width = 640
        box_height = max(150, len(lines) * line_height + box_padding_y * 2)
        box_x = 40
        box_y = 1280 - box_height - 92

        draw.rounded_rectangle(
            [box_x, box_y, box_x + box_width, box_y + box_height],
            radius=26,
            fill=(0, 0, 0, 184),
        )

        y = box_y + box_padding_y
        active_font = font if len(lines) <= 3 else small_font
        if active_font is not font:
            lines = wrap_text(draw, text, active_font, max_width=600)
            line_height = 46
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=active_font)
            line_width = bbox[2] - bbox[0]
            x = box_x + (box_width - line_width) / 2
            draw.text((x, y), line, font=active_font, fill=(255, 255, 255, 255))
            y += line_height

        path = captions_dir / f"caption_{index:02d}.png"
        image.save(path)
        paths.append(path)

    return paths


def build_overlay_filter(scenes: list[dict[str, Any]]) -> tuple[str, str]:
    parts: list[str] = []
    current = "0:v"
    cursor = 0
    for index, scene in enumerate(scenes, start=1):
        duration = int(scene.get("duration_seconds", 6))
        start = cursor
        end = cursor + duration
        cursor = end
        output = f"v{index}"
        parts.append(
            f"[{current}][{index}:v]overlay=0:0:enable='between(t,{start},{end})'[{output}]"
        )
        current = output
    return ";".join(parts), f"[{current}]"


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


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


def _srt_time(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02},000"


def _escape_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def _escape_filter_path(path: Path) -> str:
    # FFmpeg filter arguments use ':' as separators, so escape path characters
    # that are common in absolute file names.
    return str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _subtitles_filter(path: Path) -> str:
    style = "FontSize=18,Alignment=2,MarginV=70,Outline=2"
    return f"subtitles=filename='{_escape_filter_path(path)}':force_style='{style}'"


def has_ffmpeg_filter(name: str) -> bool:
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return any(line.split()[1:2] == [name] for line in probe.stdout.splitlines())
