"""Gemini image generation client for the meme slideshow pipeline.

Uses gemini-2.0-flash-preview-image-generation (free tier).
Generates one PNG per MemeScene per language.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"
_FREE_TIER_SLEEP_SECONDS = 6


class ImagenError(RuntimeError):
    """Raised when Gemini returns no image for a scene."""


@dataclass(frozen=True)
class ImagenConfig:
    api_key: str
    model: str


def load_imagen_config() -> ImagenConfig:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from .env.")
    model = os.getenv("GEMINI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL).strip() or DEFAULT_IMAGE_MODEL
    return ImagenConfig(api_key=api_key, model=model)


def generate_meme_images(
    *,
    meme_plan: dict[str, Any],
    output_dir: Path,
    config: ImagenConfig,
    languages: set[str] | None = None,
    sleep_between_requests: float = _FREE_TIER_SLEEP_SECONDS,
) -> dict[str, dict[str, Path]]:
    """Generate one PNG per scene per language.

    Returns {language: {scene_id: png_path}}.

    Free tier RPM limit: sleep_between_requests (default 6s) keeps throughput
    well under 10 RPM for a typical 4-scene × 3-language run (~72s total).
    """
    client = genai.Client(api_key=config.api_key)
    result: dict[str, dict[str, Path]] = {}

    for lang_key in ("english", "korean", "spanish"):
        if languages is not None and lang_key not in languages:
            continue
        lang_plan = meme_plan.get(lang_key)
        if not lang_plan:
            continue

        lang_dir = output_dir / lang_key
        lang_dir.mkdir(parents=True, exist_ok=True)
        scene_paths: dict[str, Path] = {}

        visual_style_anchor = str(lang_plan.get("visual_style_anchor", "")).strip()
        character_sheet = str(lang_plan.get("character_sheet", "")).strip()
        scenes = lang_plan.get("scenes", [])
        reference_image_bytes: bytes | None = None
        for idx, scene in enumerate(scenes):
            scene_id = scene.get("scene_id", f"scene_{idx + 1:02d}")
            output_path = lang_dir / f"{scene_id}.png"

            print(f"  Generating image {lang_key}/{scene_id}...")
            try:
                _generate_scene_image(
                    client=client,
                    config=config,
                    scene=scene,
                    output_path=output_path,
                    visual_style_anchor=visual_style_anchor,
                    character_sheet=character_sheet,
                    reference_image_bytes=reference_image_bytes,
                )
                scene_paths[scene_id] = output_path
                # Use scene 1 as visual reference for all subsequent scenes
                if reference_image_bytes is None and output_path.exists():
                    reference_image_bytes = output_path.read_bytes()
            except Exception as exc:
                log.warning("Image generation failed for %s/%s: %s", lang_key, scene_id, exc)

            if idx < len(scenes) - 1 or lang_key != "spanish":
                time.sleep(sleep_between_requests)

        result[lang_key] = scene_paths

    return result


def _generate_scene_image(
    *,
    client: genai.Client,
    config: ImagenConfig,
    scene: dict[str, Any],
    output_path: Path,
    visual_style_anchor: str = "",
    character_sheet: str = "",
    reference_image_bytes: bytes | None = None,
) -> None:
    prompt = _build_image_prompt(scene, visual_style_anchor=visual_style_anchor, character_sheet=character_sheet)

    if reference_image_bytes is not None:
        contents: Any = [
            types.Part.from_bytes(data=reference_image_bytes, mime_type="image/png"),
            types.Part.from_text(text=(
                "Generate a new scene in the EXACT same art style and with the EXACT same character "
                "(same face shape, hair, clothing, skin tone) as shown in the reference image above. "
                "Only the pose, expression, action, and background change. "
                + prompt
            )),
        ]
    else:
        contents = prompt

    response = client.models.generate_content(
        model=config.model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    candidates = getattr(response, "candidates", []) or []
    if not candidates:
        raise ImagenError(f"No candidates returned for scene {scene.get('scene_id')}")

    for part in candidates[0].content.parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "mime_type", "").startswith("image/"):
            image_bytes = inline.data
            if isinstance(image_bytes, str):
                import base64
                image_bytes = base64.b64decode(image_bytes)
            output_path.write_bytes(image_bytes)
            return

    raise ImagenError(
        f"Gemini returned no image part for scene {scene.get('scene_id')}. "
        f"Response: {response}"
    )


def _build_image_prompt(scene: dict[str, Any], *, visual_style_anchor: str = "", character_sheet: str = "") -> str:
    base_prompt = scene.get("image_prompt", "").strip()

    # Inject anchor + character_sheet if the planner omitted them
    anchor_prefix = ""
    if visual_style_anchor and not base_prompt.startswith(visual_style_anchor[:40]):
        anchor_prefix = visual_style_anchor
    if character_sheet and character_sheet[:30] not in base_prompt:
        anchor_prefix = f"{anchor_prefix}. {character_sheet}" if anchor_prefix else character_sheet
    if anchor_prefix:
        base_prompt = f"{anchor_prefix}. {base_prompt}"

    no_text_prefix = "ABSOLUTELY NO text, words, letters, or signs in the image. "
    style_suffix = (
        "9:16 vertical. Flat 2D illustration style. "
        "No realistic human faces or children. No medical devices."
    )

    return " ".join(p for p in [no_text_prefix, base_prompt, style_suffix] if p)
