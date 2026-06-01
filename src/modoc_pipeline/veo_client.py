"""Veo client helpers for the Gemini API route."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

log = logging.getLogger(__name__)


class DependencyError(RuntimeError):
    """Raised when a local executable or credential prerequisite is missing."""


@dataclass(frozen=True)
class GeminiVeoConfig:
    api_key: str
    model: str
    person_generation: str
    generate_audio: bool


LATEST_HIGH_QUALITY_VEO_MODEL = "veo-3.1-generate-preview"


def load_gemini_veo_config() -> GeminiVeoConfig:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise DependencyError("GEMINI_API_KEY is missing from .env.")
    return GeminiVeoConfig(
        api_key=api_key,
        model=LATEST_HIGH_QUALITY_VEO_MODEL,
        person_generation=os.getenv("GEMINI_VEO_PERSON_GENERATION", "allow_all").strip() or "allow_all",
        generate_audio=_env_bool("GEMINI_VEO_GENERATE_AUDIO", default=False),
    )


def extract_last_frame(clip_path: Path, frame_path: Path) -> bool:
    """Extract the last frame of a video clip to JPEG for frame-continuation.

    Frame continuation is the primary mechanism for visual consistency: the last
    frame of clip N becomes the first frame of clip N+1, forcing Veo to treat
    it as a continuation of the same scene rather than a fresh generation.
    Without this, Veo generates a new random character and environment every clip.
    """
    if not clip_path.exists():
        return False
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-sseof", "-0.5",
            "-i", str(clip_path),
            "-vframes", "1",
            "-q:v", "2",
            "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2",
            str(frame_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0 and frame_path.exists()


def _poll_to_completion(client: genai.Client, operation: Any, poll_seconds: int) -> Any:
    """Poll until done and return the completed operation.

    Raises RuntimeError if the operation finishes with a failed result (result
    is None or generated_videos is empty), so callers can catch and retry.
    """
    while not operation.done:
        time.sleep(poll_seconds)
        operation = client.operations.get(operation)

    result = getattr(operation, "result", None)
    if result is None:
        error = getattr(operation, "error", None)
        raise RuntimeError(f"Veo operation completed with no result. error={error}")
    videos = getattr(result, "generated_videos", None)
    if not videos:
        raise RuntimeError("Veo operation completed but generated_videos is empty.")
    return operation


def _generate_video_with_first_frame(
    client: genai.Client,
    *,
    model: str,
    prompt: str,
    config_kwargs: dict[str, Any],
    first_frame_path: Path | None,
    poll_seconds: int = 10,
) -> Any:
    """Call generate_videos, optionally anchoring to a first frame.

    Falls back to text-only generation if image-to-video fails or if the
    completed operation has no result (the Developer API may not support
    image-to-video for all model variants / API key tiers).
    """
    if first_frame_path and first_frame_path.exists():
        try:
            frame_bytes = first_frame_path.read_bytes()
            operation = client.models.generate_videos(
                model=model,
                prompt=prompt,
                image=types.Image(image_bytes=frame_bytes, mime_type="image/jpeg"),
                config=types.GenerateVideosConfig(**config_kwargs),
            )
            return _poll_to_completion(client, operation, poll_seconds)
        except Exception as exc:
            log.warning(
                "Image-to-video failed (%s); retrying as text-only generation.", exc
            )

    operation = client.models.generate_videos(
        model=model,
        prompt=prompt,
        config=types.GenerateVideosConfig(**config_kwargs),
    )
    return _poll_to_completion(client, operation, poll_seconds)


def generate_gemini_veo_clips(
    *,
    scenes: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    config: GeminiVeoConfig,
    poll_seconds: int = 10,
) -> list[Path]:
    """Generate Veo clips through the Gemini Developer API (per-language mode)."""

    client = genai.Client(api_key=config.api_key)
    written: list[Path] = []

    for language, language_scenes in scenes.items():
        prev_frame_path: Path | None = None

        for scene in language_scenes:
            scene_id = scene["scene_id"]
            target = output_dir / language / f"{scene_id}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)

            config_kwargs: dict[str, Any] = {
                "number_of_videos": 1,
                "duration_seconds": int(scene.get("duration_seconds", 6)),
                "aspect_ratio": "9:16",
                "negative_prompt": scene.get("negative_prompt") or None,
                "person_generation": config.person_generation,
            }
            if config.generate_audio:
                config_kwargs["generate_audio"] = True

            operation = _generate_video_with_first_frame(
                client,
                model=config.model,
                prompt=scene["visual_prompt"],
                config_kwargs=config_kwargs,
                first_frame_path=prev_frame_path,
                poll_seconds=poll_seconds,
            )

            generated = operation.result.generated_videos[0]
            video = generated.video
            video_bytes = video.video_bytes if video.video_bytes else client.files.download(file=video)
            target.write_bytes(video_bytes)
            written.append(target)

            frame_path = output_dir / language / f"{scene_id}_last_frame.jpg"
            if extract_last_frame(target, frame_path):
                prev_frame_path = frame_path
            else:
                log.warning("Could not extract last frame from %s; next clip will be text-only.", target)
                prev_frame_path = None

    return written


def generate_gemini_shared_veo_clips(
    *,
    visual_scenes: list[dict[str, Any]],
    output_dir: Path,
    config: GeminiVeoConfig,
    poll_seconds: int = 10,
    max_clips: int | None = None,
    initial_frame_path: Path | None = None,
) -> list[Path]:
    """Generate shared Veo clips with sequential frame continuation.

    Frame continuation strategy:
    - Clip 1 (or first selected clip): uses initial_frame_path if provided, else text-only
    - Each subsequent clip: last frame of the previous clip as first frame

    initial_frame_path allows partial re-generation (e.g. scene_02..04 only) to
    still benefit from frame continuation by seeding with scene_01's last frame.
    """
    client = genai.Client(api_key=config.api_key)
    written: list[Path] = []
    selected_scenes = visual_scenes[:max_clips] if max_clips is not None else visual_scenes

    shared_dir = output_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    prev_frame_path: Path | None = initial_frame_path

    for scene in selected_scenes:
        scene_id = scene["scene_id"]
        target = shared_dir / f"{scene_id}.mp4"

        config_kwargs: dict[str, Any] = {
            "number_of_videos": 1,
            "duration_seconds": int(scene.get("duration_seconds", 6)),
            "aspect_ratio": "9:16",
            "negative_prompt": scene.get("negative_prompt") or None,
            "person_generation": config.person_generation,
        }
        if config.generate_audio:
            config_kwargs["generate_audio"] = True

        print(f"  Generating {scene_id} {'(with frame continuation)' if prev_frame_path else '(fresh start)'}...")

        operation = _generate_video_with_first_frame(
            client,
            model=config.model,
            prompt=scene["visual_prompt"],
            config_kwargs=config_kwargs,
            first_frame_path=prev_frame_path,
            poll_seconds=poll_seconds,
        )

        generated = operation.result.generated_videos[0]
        video = generated.video
        video_bytes = video.video_bytes if video.video_bytes else client.files.download(file=video)
        target.write_bytes(video_bytes)
        written.append(target)

        frame_path = shared_dir / f"{scene_id}_last_frame.jpg"
        if extract_last_frame(target, frame_path):
            prev_frame_path = frame_path
        else:
            log.warning("Could not extract last frame from %s; next clip will be text-only.", target)
            prev_frame_path = None

    return written


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
