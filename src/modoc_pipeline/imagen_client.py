"""Gemini image generation client for the meme slideshow pipeline.

Uses Gemini image generation for meme scene images.
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

from .gemini_client import parse_json_response

log = logging.getLogger(__name__)

DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"
DEFAULT_IMAGE_REVIEW_MODEL = "gemini-3.5-flash"
_FREE_TIER_SLEEP_SECONDS = 6


class ImagenError(RuntimeError):
    """Raised when Gemini returns no image for a scene."""


@dataclass(frozen=True)
class ImagenConfig:
    api_key: str
    model: str
    review_model: str = DEFAULT_IMAGE_REVIEW_MODEL
    image_qa_enabled: bool = True
    image_qa_fail_closed: bool = False
    max_regenerations: int = 1


def load_imagen_config() -> ImagenConfig:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from .env.")
    model = os.getenv("GEMINI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL).strip() or DEFAULT_IMAGE_MODEL
    review_model = os.getenv("GEMINI_IMAGE_REVIEW_MODEL", DEFAULT_IMAGE_REVIEW_MODEL).strip() or DEFAULT_IMAGE_REVIEW_MODEL
    image_qa_enabled = os.getenv("MODOC_IMAGE_QA", "1").strip().lower() not in {"0", "false", "no"}
    image_qa_fail_closed = os.getenv("MODOC_IMAGE_QA_FAIL_CLOSED", "0").strip().lower() in {"1", "true", "yes"}
    return ImagenConfig(
        api_key=api_key,
        model=model,
        review_model=review_model,
        image_qa_enabled=image_qa_enabled,
        image_qa_fail_closed=image_qa_fail_closed,
    )


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
        visual_brief = meme_plan.get("visual_brief", {}) if isinstance(meme_plan, dict) else {}
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
                if config.image_qa_enabled:
                    review = _review_scene_image(
                        client=client,
                        config=config,
                        image_path=output_path,
                        scene=scene,
                        visual_brief=visual_brief,
                        character_sheet=character_sheet,
                    )
                    if review.get("status") == "failed":
                        log.warning(
                            "Image QA failed for %s/%s: %s",
                            lang_key,
                            scene_id,
                            "; ".join(review.get("issues", [])),
                        )
                        for attempt in range(config.max_regenerations):
                            _generate_scene_image(
                                client=client,
                                config=config,
                                scene=scene,
                                output_path=output_path,
                                visual_style_anchor=visual_style_anchor,
                                character_sheet=character_sheet,
                                reference_image_bytes=reference_image_bytes,
                                extra_instruction=_image_repair_instruction(review),
                            )
                            review = _review_scene_image(
                                client=client,
                                config=config,
                                image_path=output_path,
                                scene=scene,
                                visual_brief=visual_brief,
                                character_sheet=character_sheet,
                            )
                            if review.get("status") != "failed":
                                break
                            log.warning(
                                "Image QA retry %s still failed for %s/%s: %s",
                                attempt + 1,
                                lang_key,
                                scene_id,
                                "; ".join(review.get("issues", [])),
                            )
                        if review.get("status") == "failed" and config.image_qa_fail_closed:
                            raise ImagenError(
                                f"Image QA failed for {lang_key}/{scene_id}: "
                                f"{'; '.join(review.get('issues', []))}"
                            )
                        if review.get("status") == "failed":
                            log.warning(
                                "Continuing despite image QA failure for %s/%s: %s",
                                lang_key,
                                scene_id,
                                "; ".join(review.get("issues", [])),
                            )
                scene_paths[scene_id] = output_path
                # Lock the first successfully generated scene as the visual reference.
                # All subsequent scenes receive this image to anchor character consistency.
                if reference_image_bytes is None and output_path.exists():
                    reference_image_bytes = output_path.read_bytes()
                    log.debug("Locked reference image from %s/%s", lang_key, scene_id)
            except Exception as exc:
                log.warning("Image generation failed for %s/%s: %s", lang_key, scene_id, exc)
                raise

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
    extra_instruction: str = "",
) -> None:
    prompt = _build_image_prompt(scene, visual_style_anchor=visual_style_anchor, character_sheet=character_sheet)
    if extra_instruction:
        prompt = f"{prompt} {extra_instruction}"

    if reference_image_bytes is not None:
        contents: Any = [
            types.Part.from_bytes(data=reference_image_bytes, mime_type="image/png"),
            types.Part.from_text(text=(
                "Generate a new scene that preserves ONLY the character identity and art style from the reference image: "
                "same face shape, hair, skin tone, and outfit. "
                "Do NOT preserve the reference pose, prop, room, camera angle, or background. "
                "Change the composition to match the new scene's content-specific action, props, background, and shot type. "
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


def _review_scene_image(
    *,
    client: genai.Client,
    config: ImagenConfig,
    image_path: Path,
    scene: dict[str, Any],
    visual_brief: dict[str, Any],
    character_sheet: str = "",
) -> dict[str, Any]:
    try:
        character_sheet = character_sheet or str(scene.get("character_sheet", "")).strip()
        prompt = f"""
Review this generated illustration for a pediatric short-form video scene.

Scene JSON:
{scene}

Row-specific visual brief:
{visual_brief}

Character sheet (must be consistent across all scenes):
{character_sheet}

Return JSON only:
{{
  "status": "passed" or "failed",
  "issues": ["short issue strings"],
  "repair_instruction": "short concrete instruction"
}}

Fail if:
- the image contains readable text, letters, numbers, labels, logos, charts, or UI;
- the image shows visible medicine, medication bottles, oral syringes, droppers, measuring spoons, pills, tablets, capsules, needles, or medication preparation;
- it uses forbidden visuals from the visual brief;
- it does not clearly match the spoken tts_text;
- it does not show the scene_visual_action or primary_prop;
- primary_prop is provided but not visibly present;
- it adds irrelevant generic filler instead of row-specific visual anchors;
- it could fit an unrelated pediatric topic with no changes;
- it is nearly the same static pose/prop/background as a generic caregiver scene;
- a child appears but their gender differs from the character_sheet (e.g. girl described but boy shown, or vice versa);
- a child's hair color, hair style, or clothing color visibly differs from the character_sheet;
- a child appears to be in distress, pain, fear, or receiving medical treatment.

Pass only if the image is text-free, content-specific, safe, visually coherent, and all characters match the character_sheet.
""".strip()
        response = client.models.generate_content(
            model=config.review_model,
            contents=[
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/png"),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        raw_text = response.text or ""
        if not raw_text.strip():
            return {
                "status": "failed",
                "issues": ["Image QA returned an empty response."],
                "repair_instruction": "Regenerate with a content-specific, text-free scene.",
            }
        parsed_raw = parse_json_response(raw_text)
        parsed = parsed_raw if isinstance(parsed_raw, dict) else {}
        status = str(parsed.get("status", "")).lower()
        if status not in {"passed", "failed"}:
            return {
                "status": "failed",
                "issues": [f"Image QA returned invalid status: {status or 'missing'}."],
                "repair_instruction": "Regenerate and return a valid passed/failed QA decision.",
            }
        return {
            "status": status,
            "issues": [str(issue) for issue in parsed.get("issues", []) or []],
            "repair_instruction": str(parsed.get("repair_instruction", "")),
        }
    except Exception as exc:
        log.warning("Image QA failed for %s: %s", image_path, exc)
        return {
            "status": "failed",
            "issues": [f"Image QA could not parse or validate the review response: {exc}"],
            "repair_instruction": "Regenerate with a content-specific, text-free scene and rerun QA.",
        }


def _image_repair_instruction(review: dict[str, Any]) -> str:
    issues = "; ".join(str(issue) for issue in review.get("issues", []) or [])
    repair = str(review.get("repair_instruction", "")).strip()
    return (
        "Regenerate and fix these image QA failures. "
        f"Issues: {issues}. {repair} "
        "Keep the same character identity and art style, but change pose, props, and background to match the scene."
    ).strip()


def _build_image_prompt(scene: dict[str, Any], *, visual_style_anchor: str = "", character_sheet: str = "") -> str:
    base_prompt = scene.get("image_prompt", "").strip()
    tts_text = str(scene.get("tts_text", "")).strip()

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
        "You may add a few safe setting details that support the scene, but no readable text, labels, logos, dosage numbers, graphic symptoms, or alarming medical props. "
        "Do not show visible medicine, medication bottles, oral syringes, droppers, measuring spoons, pills, tablets, capsules, needles, or medication preparation."
    )
    spoken_line_prefix = f'This image must visually match the spoken line: "{tts_text}".' if tts_text else ""

    return " ".join(p for p in [no_text_prefix, spoken_line_prefix, base_prompt, style_suffix] if p)
