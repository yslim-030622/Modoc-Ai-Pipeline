"""Gemini TTS helpers for controlled language-specific narration."""

from __future__ import annotations

import logging
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

from google import genai
from google.genai import types


@dataclass(frozen=True)
class TtsConfig:
    api_key: str
    model: str
    voice_name: str


# Voices chosen for warmth and conversational feel (2025 preferred list).
# Fenrir = warm/approachable English, Kore = natural Korean, Aoede = expressive Spanish.
LANGUAGE_VOICES: dict[str, str] = {
    "korean": "Kore",
    "english": "Fenrir",
    "spanish": "Aoede",
}

# Meme pipeline voices — slightly more upbeat/energetic tone for short-form content.
MEME_LANGUAGE_VOICES: dict[str, str] = {
    "korean": "Kore",
    "english": "Puck",
    "spanish": "Leda",
}

LANGUAGE_NARRATION_STYLE: dict[str, str] = {
    "korean": (
        "부드럽고 따뜻한 목소리로 소아과 교육 영상을 읽어주세요. "
        "침착하고 명확하게, 일정한 속도로 읽어주세요. 단어를 추가하지 마세요. 내용: "
    ),
    "english": (
        "Read this pediatric education short in a calm, warm, clear voice. "
        "Use a steady conversational pace. Do not add any words. Content: "
    ),
    "spanish": (
        "Lee este contenido educativo pediátrico con una voz cálida, clara y tranquila. "
        "Usa un ritmo constante y no añadas palabras. Contenido: "
    ),
}


DEFAULT_TTS_MODEL = "gemini-3.1-flash-tts-preview"


def load_tts_config() -> TtsConfig:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from .env.")
    return TtsConfig(
        api_key=api_key,
        model=os.getenv("GEMINI_TTS_MODEL", DEFAULT_TTS_MODEL).strip() or DEFAULT_TTS_MODEL,
        voice_name=os.getenv("GEMINI_TTS_VOICE", "Kore").strip() or "Kore",
    )


def _synthesize(
    client: genai.Client,
    text: str,
    *,
    model: str,
    voice: str,
    language: str,
    style_instruction: str = "",
) -> bytes:
    """Call Gemini TTS and return raw PCM bytes."""
    style_prefix = style_instruction.strip() or LANGUAGE_NARRATION_STYLE.get(language, LANGUAGE_NARRATION_STYLE["english"])
    if "내용:" not in style_prefix and "Content:" not in style_prefix and "Contenido:" not in style_prefix:
        style_prefix = f"{style_prefix.strip()} Content: "
    prompt = f"{style_prefix}{text}"
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise RuntimeError("Gemini TTS returned no candidates")
    parts = getattr(candidates[0].content, "parts", None) or []
    if not parts:
        raise RuntimeError("Gemini TTS returned no audio parts")
    data = parts[0].inline_data.data
    if not data:
        raise RuntimeError("Gemini TTS returned empty audio data")
    return data



_RETRY_DELAYS = (3.0, 8.0, 20.0)  # 3 attempts: wait 3s, then 8s, then 20s before giving up


def _synthesize_with_retry(
    client: genai.Client,
    text: str,
    *,
    model: str,
    voice: str,
    language: str,
    style_instruction: str = "",
    scene_label: str = "",
) -> bytes | None:
    last_exc: Exception | None = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            return _synthesize(client, text, model=model, voice=voice, language=language, style_instruction=style_instruction)
        except Exception as exc:
            last_exc = exc
            if delay is None:
                break
            log.warning("TTS %s attempt %d failed (%s) — retrying in %.0fs", scene_label, attempt, exc, delay)
            print(f"  TTS {scene_label}: attempt {attempt} failed, retrying in {delay:.0f}s...")
            time.sleep(delay)
    log.warning("TTS %s gave up after %d attempts: %s", scene_label, len(_RETRY_DELAYS) + 1, last_exc)
    print(f"  TTS {scene_label}: skipped after {len(_RETRY_DELAYS) + 1} attempts ({last_exc})")
    return None


def generate_meme_gemini_tts(
    *,
    meme_plan: dict[str, Any],
    output_dir: Path,
    config: TtsConfig,
    languages: set[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """Generate per-scene Gemini TTS WAV files from a MemePlan.

    Reads tts_text from each MemeScene and synthesizes with Gemini TTS.
    Returns {language: {scene_id: wav_path}}.
    Skips scenes with empty tts_text.
    """
    client = genai.Client(api_key=config.api_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, Path]] = {}

    for lang_key in ("english", "korean", "spanish"):
        if languages is not None and lang_key not in languages:
            continue
        lang_plan = meme_plan.get(lang_key)
        if not lang_plan:
            continue

        voice = MEME_LANGUAGE_VOICES.get(lang_key, LANGUAGE_VOICES.get(lang_key, config.voice_name))
        tts_style = str(lang_plan.get("tts_style", "")).strip()
        lang_dir = output_dir / lang_key
        lang_dir.mkdir(parents=True, exist_ok=True)
        scene_paths: dict[str, Path] = {}

        for scene in lang_plan.get("scenes", []):
            scene_id = scene.get("scene_id", "")
            tts_text = str(scene.get("tts_text", "")).strip()
            if not tts_text or not scene_id:
                continue
            pcm = _synthesize_with_retry(
                client,
                tts_text,
                model=config.model,
                voice=voice,
                language=lang_key,
                style_instruction=tts_style,
                scene_label=f"{lang_key}/{scene_id}",
            )
            if pcm is None:
                continue
            path = lang_dir / f"{scene_id}.wav"
            write_wave(path, pcm)
            scene_paths[scene_id] = path
            print(f"  TTS {lang_key}/{scene_id}: {len(pcm) // 48000:.1f}s")

        if scene_paths:
            result[lang_key] = scene_paths

    return result


def write_wave(path: Path, pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(rate)
        handle.writeframes(pcm)
