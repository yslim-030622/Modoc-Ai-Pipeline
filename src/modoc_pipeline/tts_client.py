"""Gemini TTS helpers for controlled language-specific narration."""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


@dataclass(frozen=True)
class TtsConfig:
    api_key: str
    model: str
    voice_name: str


# Language-specific voices for natural pronunciation.
# Kore = Korean female, Puck = English male natural, Leda = Spanish female natural.
LANGUAGE_VOICES: dict[str, str] = {
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


def load_tts_config() -> TtsConfig:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from .env.")
    return TtsConfig(
        api_key=api_key,
        model=os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts").strip()
        or "gemini-2.5-flash-preview-tts",
        voice_name=os.getenv("GEMINI_TTS_VOICE", "Kore").strip() or "Kore",
    )


def _synthesize(
    client: genai.Client,
    text: str,
    *,
    model: str,
    voice: str,
    language: str,
) -> bytes:
    """Call Gemini TTS and return raw PCM bytes."""
    style_prefix = LANGUAGE_NARRATION_STYLE.get(language, LANGUAGE_NARRATION_STYLE["english"])
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
    return response.candidates[0].content.parts[0].inline_data.data


def generate_tts_per_scene(
    *,
    localized_tracks: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    config: TtsConfig,
    languages: set[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """Generate one WAV file per scene per language.

    Returns {language: {scene_id: wav_path}}.

    Per-scene audio is the foundation for frame-accurate subtitle sync:
    each Veo clip is extended (freeze last frame) to match its scene's audio
    duration, so speech, subtitles, and video are aligned without any
    tempo compression.
    """
    client = genai.Client(api_key=config.api_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, Path]] = {}

    for language, scenes in localized_tracks.items():
        if languages is not None and language not in languages:
            continue
        voice = LANGUAGE_VOICES.get(language, config.voice_name)
        lang_dir = output_dir / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        scene_paths: dict[str, Path] = {}

        for scene in scenes:
            scene_id = scene["scene_id"]
            tts_text = str(scene.get("tts_text") or scene.get("caption_text") or "").strip()
            if not tts_text:
                continue
            pcm = _synthesize(client, tts_text, model=config.model, voice=voice, language=language)
            path = lang_dir / f"{scene_id}.wav"
            write_wave(path, pcm)
            scene_paths[scene_id] = path
            print(f"  TTS {language}/{scene_id}: {len(pcm)//48000:.1f}s audio")

        result[language] = scene_paths

    return result


def generate_tts_audio(
    *,
    localized_tracks: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    config: TtsConfig,
    languages: set[str] | None = None,
) -> dict[str, Path]:
    """Legacy: generate one combined WAV per language. Used as fallback only."""
    client = genai.Client(api_key=config.api_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for language, scenes in localized_tracks.items():
        if languages is not None and language not in languages:
            continue
        voice = LANGUAGE_VOICES.get(language, config.voice_name)
        transcript = " ".join(
            str(scene.get("tts_text") or scene.get("caption_text") or "").strip()
            for scene in scenes
        )
        if not transcript.strip():
            raise ValueError(f"No TTS text found for language: {language}")
        pcm = _synthesize(client, transcript, model=config.model, voice=voice, language=language)
        path = output_dir / f"{language}.wav"
        write_wave(path, pcm)
        written[language] = path

    return written


def write_wave(path: Path, pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(rate)
        handle.writeframes(pcm)
