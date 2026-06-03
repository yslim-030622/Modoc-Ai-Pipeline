"""Veo client helpers for the Gemini API route."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import types

log = logging.getLogger(__name__)


class DependencyError(RuntimeError):
    """Raised when a local executable or credential prerequisite is missing."""


class VeoGenerationError(RuntimeError):
    """Raised with serializable context when Veo generation fails."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


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
        model=os.getenv("GEMINI_VEO_MODEL", LATEST_HIGH_QUALITY_VEO_MODEL).strip() or LATEST_HIGH_QUALITY_VEO_MODEL,
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

    result = _operation_videos_result(operation)
    if result is None:
        error = getattr(operation, "error", None)
        raise VeoGenerationError(
            f"Veo operation completed with no result. error={error}",
            {"operation": _serialize_operation(operation), "error": str(error)},
        )
    videos = getattr(result, "generated_videos", None)
    if not videos:
        raise VeoGenerationError(
            "Veo operation completed but generated_videos is empty.",
            {"operation": _serialize_operation(operation), "result": _serialize_value(result)},
        )
    return operation


def _generate_video_with_first_frame(
    client: genai.Client,
    *,
    model: str,
    prompt: str,
    config_kwargs: dict[str, Any],
    first_frame_path: Path | None,
    scene_id: str,
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
            operation = _call_generate_videos_with_retries(
                client,
                model=model,
                prompt=prompt,
                image=types.Image(image_bytes=frame_bytes, mime_type=_image_mime_type(first_frame_path)),
                config=types.GenerateVideosConfig(**config_kwargs),
            )
            return _poll_to_completion(client, operation, poll_seconds)
        except Exception as exc:
            log.warning(
                "Image-to-video failed (%s); retrying as text-only generation.", exc
            )

    try:
        operation = _call_generate_videos_with_retries(
            client,
            model=model,
            prompt=prompt,
            config=types.GenerateVideosConfig(**config_kwargs),
        )
        return _poll_to_completion(client, operation, poll_seconds)
    except Exception as exc:
        safe_prompt = build_safe_retry_prompt(prompt)
        retry_kwargs = {**config_kwargs, "negative_prompt": None}
        try:
            retry_operation = _call_generate_videos_with_retries(
                client,
                model=model,
                prompt=safe_prompt,
                config=types.GenerateVideosConfig(**retry_kwargs),
            )
            return _poll_to_completion(client, retry_operation, poll_seconds)
        except Exception as retry_exc:
            report = {
                "scene_id": scene_id,
                "model": model,
                "prompt": prompt,
                "safe_retry_prompt": safe_prompt,
                "config": config_kwargs,
                "first_error": _error_payload(exc),
                "retry_error": _error_payload(retry_exc),
            }
            raise VeoGenerationError(f"Veo generation failed for {scene_id}: {retry_exc}", report) from retry_exc


def _call_generate_videos_with_retries(
    client: genai.Client,
    *,
    model: str,
    prompt: str,
    config: types.GenerateVideosConfig,
    image: types.Image | None = None,
    attempts: int = 3,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            kwargs: dict[str, Any] = {"model": model, "prompt": prompt, "config": config}
            if image is not None:
                kwargs["image"] = image
            return client.models.generate_videos(**kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts - 1 or not _is_transient_network_error(exc):
                break
            time.sleep(3 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Veo generate_videos failed without an exception.")


def _image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _is_transient_network_error(exc: Exception) -> bool:
    message = str(exc).lower()
    transient_markers = (
        "connecterror",
        "nodename nor servname",
        "temporary failure",
        "name or service not known",
        "connection reset",
        "connection aborted",
        "timed out",
        "timeout",
        "503",
        "504",
        "unavailable",
    )
    return any(marker in message for marker in transient_markers) or isinstance(exc, OSError)


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
                scene_id=scene_id,
                poll_seconds=poll_seconds,
            )

            generated = _operation_videos_result(operation).generated_videos[0]
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
    visual_qa_callback: Callable[[dict[str, Any], Path], dict[str, Any]] | None = None,
    max_scene_retries: int = 0,
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

    visual_qa_reports: list[dict[str, Any]] = []

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

        base_prompt = scene["visual_prompt"]
        attempt = 0
        accepted = False
        last_report: dict[str, Any] | None = None

        while attempt <= max_scene_retries and not accepted:
            prompt = base_prompt if attempt == 0 else build_visual_repair_prompt(base_prompt, last_report)
            print(
                f"  Generating {scene_id} "
                f"{'(with frame continuation)' if prev_frame_path else '(fresh start)'}"
                f"{'' if attempt == 0 else f' visual-retry-{attempt}'}..."
            )

            operation = _generate_video_with_first_frame(
                client,
                model=config.model,
                prompt=prompt,
                config_kwargs=config_kwargs,
                first_frame_path=prev_frame_path,
                scene_id=scene_id,
                poll_seconds=poll_seconds,
            )

            generated = _operation_videos_result(operation).generated_videos[0]
            video = generated.video
            video_bytes = video.video_bytes if video.video_bytes else client.files.download(file=video)
            target.write_bytes(video_bytes)

            if visual_qa_callback is None:
                accepted = True
                break

            last_report = visual_qa_callback(scene, target)
            last_report["attempt"] = attempt
            visual_qa_reports.append(last_report)
            accepted = last_report.get("status") == "passed"
            attempt += 1

        if not accepted:
            raise VeoGenerationError(
                f"Visual QA failed for {scene_id} after {max_scene_retries + 1} attempt(s).",
                {
                    "scene_id": scene_id,
                    "prompt": base_prompt,
                    "last_visual_qa_report": last_report,
                    "all_visual_qa_reports": visual_qa_reports,
                },
            )

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


def build_safe_retry_prompt(prompt: str) -> str:
    return (
        "Controlled 2D character-plus-infographic vertical pediatric health scene. "
        "A small faceless teal MoDoc guide marker appears beside one clear infographic prop, "
        "plain warm paper texture background, no rooms, no people, no children, no doctors, "
        "no medical devices, no eyes, no mouth, no lips, no teeth, no hands, no limbs, no text, "
        "NO TEXT ANYWHERE IN FRAME, clean 9:16 video."
    )


def build_visual_repair_prompt(base_prompt: str, report: dict[str, Any] | None) -> str:
    repair = ""
    if report:
        repair = str(report.get("repair_instruction") or report.get("summary") or "")
        forbidden = ", ".join(report.get("detected_forbidden_elements") or [])
        random_objects = ", ".join(report.get("detected_random_objects") or [])
        if forbidden:
            repair += f" Remove forbidden elements: {forbidden}."
        if random_objects:
            repair += f" Remove unrelated random objects: {random_objects}."
    return (
        f"{base_prompt}. Visual QA repair: {repair} "
        "Make the scene visually match the scene contract with concrete infographic props. "
        "Keep the same mouthless MoDoc Guide icon, no mouth/lips/teeth/hands, no rooms, no random props, "
        "NO TEXT ANYWHERE IN FRAME."
    )


def _operation_videos_result(operation: Any) -> Any:
    result = getattr(operation, "result", None)
    if result is not None:
        return result
    response = getattr(operation, "response", None)
    if response is not None:
        return response
    return None


def _serialize_operation(operation: Any) -> dict[str, Any]:
    return _serialize_value(operation)


def _serialize_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return value
    return {"repr": repr(value)}


def _error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, VeoGenerationError):
        return {"message": str(exc), "report": exc.report}
    return {"type": exc.__class__.__name__, "message": str(exc)}
