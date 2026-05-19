"""FFmpeg rendering helpers for final vertical MP4 assembly."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

        srt_path.write_text(build_srt(language_scenes), encoding="utf-8")
        concat_path.write_text(build_concat_file(clip_paths), encoding="utf-8")

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
        else:
            # Some Homebrew FFmpeg builds omit libass/subtitles support. In that
            # case we still produce a valid MP4 and keep the SRT sidecar, rather
            # than blocking the pipeline on a local codec build detail.
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
