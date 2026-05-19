"""Veo client helpers for Vertex AI and Gemini API routes."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


class DependencyError(RuntimeError):
    """Raised when a local executable or credential prerequisite is missing."""


@dataclass(frozen=True)
class VertexConfig:
    project_id: str
    location: str
    model: str
    person_generation: str
    generate_audio: bool


@dataclass(frozen=True)
class GeminiVeoConfig:
    api_key: str
    model: str
    person_generation: str
    generate_audio: bool


def load_vertex_config() -> VertexConfig:
    project_id = os.getenv("VERTEX_PROJECT_ID", "").strip()
    if not project_id:
        raise DependencyError("VERTEX_PROJECT_ID is missing from .env.")
    return VertexConfig(
        project_id=project_id,
        location=os.getenv("VERTEX_LOCATION", "us-central1").strip() or "us-central1",
        model=os.getenv("VEO_MODEL", "veo-3.0-generate-001").strip() or "veo-3.0-generate-001",
        # The project scope document requires allow_all for child-content allowlisting.
        # It remains configurable because some SDK/project combinations may only
        # expose allow_adult unless the allowlist is active.
        person_generation=os.getenv("VEO_PERSON_GENERATION", "allow_all").strip() or "allow_all",
        generate_audio=_env_bool("VEO_GENERATE_AUDIO", default=True),
    )


def load_gemini_veo_config() -> GeminiVeoConfig:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise DependencyError("GEMINI_API_KEY is missing from .env.")
    return GeminiVeoConfig(
        api_key=api_key,
        model=os.getenv("GEMINI_VEO_MODEL", "veo-3.1-generate-preview").strip()
        or "veo-3.1-generate-preview",
        # Gemini API may reject child-related person generation depending on
        # account/model access. Keeping this configurable lets the team test
        # stricter values without changing code.
        person_generation=os.getenv("GEMINI_VEO_PERSON_GENERATION", "allow_all").strip() or "allow_all",
        # The Gemini Developer API rejects generate_audio in the current SDK
        # mode. Vertex remains the route for audio-first Veo generation.
        generate_audio=_env_bool("GEMINI_VEO_GENERATE_AUDIO", default=False),
    )


def ensure_vertex_prerequisites() -> None:
    """Check local auth prerequisites before creating expensive Veo operations."""

    service_account = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if service_account:
        if Path(service_account).exists():
            return
        raise DependencyError(f"GOOGLE_APPLICATION_CREDENTIALS does not exist: {service_account}")

    if not shutil.which("gcloud"):
        raise DependencyError(_vertex_setup_message("gcloud is not installed."))

    probe = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise DependencyError(_vertex_setup_message("Application-default credentials are not configured."))


def generate_veo_clips(
    *,
    scenes: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    config: VertexConfig,
    poll_seconds: int = 10,
) -> list[Path]:
    client = genai.Client(
        vertexai=True,
        project=config.project_id,
        location=config.location,
    )
    written: list[Path] = []

    for language, language_scenes in scenes.items():
        for scene in language_scenes:
            scene_id = scene["scene_id"]
            target = output_dir / language / f"{scene_id}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)

            operation = client.models.generate_videos(
                model=config.model,
                prompt=scene["visual_prompt"],
                config=types.GenerateVideosConfig(
                    number_of_videos=1,
                    duration_seconds=int(scene.get("duration_seconds", 6)),
                    aspect_ratio="9:16",
                    negative_prompt=scene.get("negative_prompt") or None,
                    person_generation=config.person_generation,
                    generate_audio=config.generate_audio,
                ),
            )

            while not operation.done:
                time.sleep(poll_seconds)
                operation = client.operations.get(operation)

            generated = operation.result.generated_videos[0]
            video = generated.video
            if video.video_bytes:
                video_bytes = video.video_bytes
            else:
                video_bytes = client.files.download(file=video)
            target.write_bytes(video_bytes)
            written.append(target)

    return written


def generate_gemini_veo_clips(
    *,
    scenes: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    config: GeminiVeoConfig,
    poll_seconds: int = 10,
) -> list[Path]:
    """Generate Veo clips through the Gemini Developer API.

    This route uses only GEMINI_API_KEY and does not require gcloud/ADC. It is
    useful for quick validation, while the Vertex route remains available for
    MoDoc's allowlisted child-content workflow.
    """

    client = genai.Client(api_key=config.api_key)
    written: list[Path] = []

    for language, language_scenes in scenes.items():
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

            operation = client.models.generate_videos(
                model=config.model,
                prompt=scene["visual_prompt"],
                config=types.GenerateVideosConfig(**config_kwargs),
            )

            while not operation.done:
                time.sleep(poll_seconds)
                operation = client.operations.get(operation)

            generated = operation.result.generated_videos[0]
            video = generated.video
            if video.video_bytes:
                video_bytes = video.video_bytes
            else:
                video_bytes = client.files.download(file=video)
            target.write_bytes(video_bytes)
            written.append(target)

    return written


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _vertex_setup_message(reason: str) -> str:
    return (
        f"{reason}\n"
        "Install and authenticate Google Cloud SDK before running Veo:\n"
        "  brew install --cask google-cloud-sdk\n"
        "  gcloud auth application-default login\n"
        "  gcloud config set project 47589570665"
    )
