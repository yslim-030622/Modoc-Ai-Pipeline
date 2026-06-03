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

LYRIA_MODEL = "lyria-realtime"

# Culturally-tuned BGM prompts engineered for viral short-form parenting content.
#
# Research basis (2025):
#   Korean: K-pop synth-pop / 감성 trap trending on 틱톡 for parenting 짤 — punchy 8-bar loop,
#           bright pluck synth + emotional piano, ~100 BPM. NewJeans/aespa production aesthetic.
#   English: Hyperpop-lite / bright upbeat indie at 115 BPM dominant for US millennial-parent
#            Reels (think Olivia Rodrigo "good 4 u" energy but softer). Hook from bar 1.
#   Spanish: Modern reggaeton-pop dembow 94 BPM + marimba/vibraphone melody hook; "sad reggaeton"
#            / emotional urbano crossover trend for Latino family TikTok (Bad Bunny soft-side energy).
#
# Prompt rules for Lyria:
#   - Lead with BPM + genre so the model locks tempo immediately
#   - Say "punchy kick" / "hook from the first bar" — Lyria responds to these cues
#   - End with explicit no-vocals guardrail
LANGUAGE_BGM_PROMPTS: dict[str, str] = {
    "korean": (
        "K-pop synth-pop at 100 BPM. "
        "Punchy 808 kick drum on every beat, bright pluck synth hook starts on bar 1, "
        "warm emotional piano chords underneath, light trap hi-hat roll. "
        "Catchy and energetic from the very first second. "
        "Cute uplifting feel — perfect for viral Korean parenting content. "
        "Instrumental only — absolutely no lyrics, no vocals, no humming, no voice whatsoever."
    ),
    "english": (
        "Upbeat hyperpop-lite indie-pop at 115 BPM. "
        "Bright punchy synth hook punches in immediately on bar 1, "
        "four-on-the-floor kick drum, warm strummed acoustic guitar layer, "
        "playful xylophone accent on the offbeat. "
        "Energetic, fun, millennial-parent TikTok Reels energy — builds excitement across 30 seconds. "
        "Instrumental only — absolutely no lyrics, no vocals, no humming, no voice whatsoever."
    ),
    "spanish": (
        "Modern reggaeton-pop at 94 BPM. "
        "Dembow rhythm kicks in from bar 1, bright marimba melody hook over warm Latin bass, "
        "emotional synth pad underneath, light shaker and cowbell percussion. "
        "Uplifting, festive, warm — viral Latino family TikTok energy. "
        "Instrumental only — absolutely no lyrics, no vocals, no humming, no voice whatsoever."
    ),
}


class BgmError(RuntimeError):
    """Raised when Lyria returns no audio."""


def generate_meme_bgm(
    *,
    language: str,
    output_path: Path,
    api_key: str,
    max_retries: int = 2,
) -> Path:
    """Generate a BGM clip for the given language using Lyria.

    Retries up to max_retries times on failure. If the language-specific
    prompt fails after retries, falls back to the English prompt so we
    always produce audio rather than silently skipping.

    Returns the path to the written audio file.
    """
    client = genai.Client(api_key=api_key)
    prompts_to_try = [
        LANGUAGE_BGM_PROMPTS.get(language, LANGUAGE_BGM_PROMPTS["english"]),
        LANGUAGE_BGM_PROMPTS["english"],  # fallback
    ]

    last_exc: Exception = BgmError("No attempts made")
    for attempt_idx, prompt in enumerate(prompts_to_try):
        label = language if attempt_idx == 0 else f"{language}→english-fallback"
        for retry in range(max_retries):
            try:
                print(f"  Generating BGM for {label} ({LYRIA_MODEL}){'  retry ' + str(retry) if retry else ''}...")
                audio_bytes = _call_lyria(client, prompt)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(audio_bytes)
                size_kb = len(audio_bytes) // 1024
                print(f"  BGM {label}: {size_kb}KB")
                return output_path
            except Exception as exc:
                last_exc = exc
                log.warning("BGM attempt failed (%s, retry %d): %s", label, retry, exc)

    raise BgmError(f"All BGM attempts failed for {language}: {last_exc}") from last_exc


def _call_lyria(client: genai.Client, prompt: str) -> bytes:
    """Make a single Lyria API call and return raw audio bytes."""
    response = client.models.generate_content(
        model=LYRIA_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
        ),
    )

    candidates = getattr(response, "candidates", []) or []
    if not candidates:
        raise BgmError("Lyria returned no candidates")

    for part in candidates[0].content.parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "mime_type", "").startswith("audio/"):
            data = inline.data
            if isinstance(data, str):
                import base64
                data = base64.b64decode(data)
            return data

    raise BgmError(f"Lyria returned no audio part. Response text: {getattr(response, 'text', '')!r}")


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
