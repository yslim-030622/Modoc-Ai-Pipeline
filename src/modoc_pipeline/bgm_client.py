"""Lyria BGM generation per language for the meme slideshow pipeline.

Uses lyria-3-clip-preview (30s clips, MP3 output).
Each language gets a culturally-tuned BGM prompt for viral short-form content.
"""

from __future__ import annotations

import logging
from pathlib import Path

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

LYRIA_MODEL = "lyria-3-clip-preview"

# Culturally-tuned BGM prompts for viral short-form parenting/health content.
# Research basis: Korean R&B/trap trending on 틱톡; US indie-pop/lo-fi for parenting content;
# Latino reggaeton-dembow for Latino parent community TikTok.
LANGUAGE_BGM_PROMPTS: dict[str, str] = {
    "korean": (
        "Smooth Korean R&B with warm piano chords and light trap-influenced beat at 95 BPM. "
        "Soft emotional feel, intimate and cozy atmosphere. "
        "Nostalgic yet modern, family-friendly. "
        "Instrumental only — no lyrics, no vocals, no voice."
    ),
    "english": (
        "Upbeat wholesome indie-pop with bright acoustic guitar and warm piano at 110 BPM. "
        "Joyful, relatable millennial-parent energy. Light percussion, feel-good vibe. "
        "Instrumental only — no lyrics, no vocals, no voice."
    ),
    "spanish": (
        "Reggaeton-inspired beat with dembow rhythm at 92 BPM. "
        "Warm Latin percussion, light marimba melody, uplifting family-friendly celebration feel. "
        "Instrumental only — no lyrics, no vocals, no voice."
    ),
}


class BgmError(RuntimeError):
    """Raised when Lyria returns no audio."""


def generate_meme_bgm(
    *,
    language: str,
    output_path: Path,
    api_key: str,
) -> Path:
    """Generate a 30-second BGM clip for the given language using Lyria.

    Returns the path to the written MP3 file.
    """
    client = genai.Client(api_key=api_key)
    prompt = LANGUAGE_BGM_PROMPTS.get(language, LANGUAGE_BGM_PROMPTS["english"])

    print(f"  Generating BGM for {language} ({LYRIA_MODEL})...")
    response = client.models.generate_content(
        model=LYRIA_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
        ),
    )

    candidates = getattr(response, "candidates", []) or []
    if not candidates:
        raise BgmError(f"Lyria returned no candidates for {language}")

    for part in candidates[0].content.parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "mime_type", "").startswith("audio/"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(inline.data)
            size_kb = len(inline.data) // 1024
            print(f"  BGM {language}: {size_kb}KB ({inline.mime_type})")
            return output_path

    raise BgmError(f"Lyria returned no audio part for {language}. Response: {response}")


def generate_all_bgm(
    *,
    output_dir: Path,
    api_key: str,
    languages: set[str] | None = None,
) -> dict[str, Path]:
    """Generate one BGM file per language. Returns {language: mp3_path}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}

    for lang in ("english", "korean", "spanish"):
        if languages is not None and lang not in languages:
            continue
        output_path = output_dir / f"{lang}_bgm.mp3"
        try:
            generate_meme_bgm(language=lang, output_path=output_path, api_key=api_key)
            result[lang] = output_path
        except Exception as exc:
            log.warning("BGM generation failed for %s: %s", lang, exc)

    return result
