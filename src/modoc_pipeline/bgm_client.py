"""Lyria BGM generation per language for the meme slideshow pipeline.

Uses Lyria RealTime streaming music generation and writes WAV output.
Each language gets a culturally-tuned BGM prompt for viral short-form content.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

log = logging.getLogger(__name__)

LYRIA_MODEL = "models/lyria-realtime-exp"
LYRIA_SAMPLE_RATE = 44100
LYRIA_CHANNELS = 2
LYRIA_SAMPLE_WIDTH = 2
DEFAULT_BGM_SECONDS = 30.0
DEFAULT_STREAM_TIMEOUT_SECONDS = 45.0

# Culturally-tuned BGM prompts engineered for viral short-form parenting content.
#
# These defaults are conservative fallbacks. Topic-specific MemePlan bgm_prompt
# should normally override them. Keep defaults voiceover-safe and medically calm.
#
# Prompt rules for Lyria:
#   - Lead with BPM + genre so the model locks tempo immediately
#   - Say "punchy kick" / "hook from the first bar" — Lyria responds to these cues
#   - End with explicit no-vocals guardrail
LANGUAGE_BGM_PROMPTS: dict[str, str] = {
    "korean": (
        "Gentle Korean social-video educational underscore at 92 BPM. "
        "Soft electric piano, light muted percussion, warm pad, subtle pluck accents. "
        "Calm and reassuring, not cute, not comedic, leaving clear space for voiceover. "
        "Instrumental only — absolutely no lyrics, no vocals, no humming, no voice whatsoever."
    ),
    "english": (
        "Gentle educational short-form underscore at 96 BPM. "
        "Soft piano, muted marimba accents, light brushed percussion, warm low synth pad. "
        "Reassuring, curious, parent-friendly, never festive or dramatic, leaving clear space for voiceover. "
        "Instrumental only — absolutely no lyrics, no vocals, no humming, no voice whatsoever."
    ),
    "spanish": (
        "Warm Spanish-language social-video educational underscore at 94 BPM. "
        "Soft nylon guitar, gentle marimba accents, light shaker, warm pad, no dembow rhythm. "
        "Human, calm, reassuring, not festive, not party-like, leaving clear space for voiceover. "
        "Instrumental only — absolutely no lyrics, no vocals, no humming, no voice whatsoever."
    ),
}


LANGUAGE_BGM_CONFIG: dict[str, types.LiveMusicGenerationConfig] = {
    "korean": types.LiveMusicGenerationConfig(
        bpm=92,
        temperature=1.0,
        guidance=4.0,
        density=0.56,
        brightness=0.62,
        music_generation_mode=types.MusicGenerationMode.QUALITY,
    ),
    "english": types.LiveMusicGenerationConfig(
        bpm=96,
        temperature=1.0,
        guidance=4.0,
        density=0.58,
        brightness=0.65,
        music_generation_mode=types.MusicGenerationMode.QUALITY,
    ),
    "spanish": types.LiveMusicGenerationConfig(
        bpm=94,
        temperature=1.0,
        guidance=4.0,
        density=0.58,
        brightness=0.66,
        music_generation_mode=types.MusicGenerationMode.QUALITY,
    ),
}


@dataclass(frozen=True)
class BgmConfig:
    model: str = LYRIA_MODEL
    seconds: float = DEFAULT_BGM_SECONDS
    stream_timeout_seconds: float = DEFAULT_STREAM_TIMEOUT_SECONDS


class BgmError(RuntimeError):
    """Raised when Lyria returns no audio."""


def generate_meme_bgm(
    *,
    language: str,
    output_path: Path,
    api_key: str,
    max_retries: int = 2,
    config: BgmConfig | None = None,
    prompt_override: str | None = None,
    music_config_override: dict[str, Any] | None = None,
) -> Path:
    """Generate a BGM clip for the given language using Lyria.

    Retries up to max_retries times on failure. If the language-specific
    prompt fails after retries, falls back to the English prompt so we
    always produce audio rather than silently skipping.

    Returns the path to the written audio file.
    """
    _ensure_ssl_cert_file()
    bgm_config = config or BgmConfig()
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1alpha"),
    )
    prompts_to_try = [
        _with_instrumental_guardrail(prompt_override) if prompt_override else LANGUAGE_BGM_PROMPTS.get(language, LANGUAGE_BGM_PROMPTS["english"]),
        LANGUAGE_BGM_PROMPTS["english"],  # fallback
    ]

    last_exc: Exception = BgmError("No attempts made")
    for attempt_idx, prompt in enumerate(prompts_to_try):
        label = language if attempt_idx == 0 else f"{language}→english-fallback"
        for retry in range(max_retries):
            try:
                print(f"  Generating BGM for {label} ({bgm_config.model}){'  retry ' + str(retry) if retry else ''}...")
                audio_bytes = _call_lyria(
                    client,
                    prompt,
                    language=language if attempt_idx == 0 else "english",
                    config=bgm_config,
                    music_config_override=music_config_override if attempt_idx == 0 else None,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                write_wave(output_path, audio_bytes)
                size_kb = len(audio_bytes) // 1024
                print(f"  BGM {label}: {size_kb}KB")
                return output_path
            except Exception as exc:
                last_exc = exc
                log.warning("BGM attempt failed (%s, retry %d): %s", label, retry, exc)

    raise BgmError(f"All BGM attempts failed for {language}: {last_exc}") from last_exc


def _call_lyria(
    client: genai.Client,
    prompt: str,
    *,
    language: str,
    config: BgmConfig,
    music_config_override: dict[str, Any] | None = None,
) -> bytes:
    """Stream Lyria RealTime PCM audio and return raw PCM bytes."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            _call_lyria_async(
                client,
                prompt,
                language=language,
                config=config,
                music_config_override=music_config_override,
            )
        )
    raise BgmError("BGM generation cannot run inside an existing asyncio event loop.")


async def _call_lyria_async(
    client: genai.Client,
    prompt: str,
    *,
    language: str,
    config: BgmConfig,
    music_config_override: dict[str, Any] | None = None,
) -> bytes:
    target_bytes = int(config.seconds * LYRIA_SAMPLE_RATE * LYRIA_CHANNELS * LYRIA_SAMPLE_WIDTH)
    chunks: list[bytes] = []
    music_config = _resolve_music_config(language, music_config_override)

    async def receive_audio(session) -> None:
        async for message in session.receive():
            filtered = getattr(message, "filtered_prompt", None)
            if filtered:
                raise BgmError(f"Lyria filtered prompt: {filtered}")
            server_content = getattr(message, "server_content", None)
            audio_chunks = getattr(server_content, "audio_chunks", None) if server_content else None
            for chunk in audio_chunks or []:
                data = getattr(chunk, "data", None)
                if not data:
                    continue
                if isinstance(data, str):
                    data = base64.b64decode(data)
                chunks.append(data)
                if sum(len(item) for item in chunks) >= target_bytes:
                    return

    async with client.aio.live.music.connect(model=config.model) as session:
        await session.set_weighted_prompts([types.WeightedPrompt(text=prompt, weight=1.0)])
        await session.set_music_generation_config(music_config)
        await session.play()
        try:
            await asyncio.wait_for(receive_audio(session), timeout=config.stream_timeout_seconds)
        except TimeoutError:
            if not chunks:
                raise
            log.warning("Lyria stream timed out after %.1fs; using partial BGM audio.", config.stream_timeout_seconds)
        finally:
            await session.stop()

    audio = b"".join(chunks)
    if not audio:
        raise BgmError("Lyria returned no audio chunks")
    return audio[:target_bytes]


def generate_all_bgm(
    *,
    output_dir: Path,
    api_key: str,
    meme_plan: dict[str, Any] | None = None,
    languages: set[str] | None = None,
) -> dict[str, Path]:
    """Generate one BGM file per language. Returns {language: wav_path}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}

    for lang in ("english", "korean", "spanish"):
        if languages is not None and lang not in languages:
            continue
        output_path = output_dir / f"{lang}_bgm.wav"
        lang_plan = (meme_plan or {}).get(lang, {}) if isinstance(meme_plan, dict) else {}
        try:
            generate_meme_bgm(
                language=lang,
                output_path=output_path,
                api_key=api_key,
                prompt_override=str(lang_plan.get("bgm_prompt", "")).strip() or None,
                music_config_override=lang_plan.get("bgm_config") if isinstance(lang_plan.get("bgm_config"), dict) else None,
            )
            result[lang] = output_path
        except Exception as exc:
            log.warning("BGM generation failed for %s: %s", lang, exc)

    return result


def write_wave(
    path: Path,
    pcm: bytes,
    *,
    channels: int = LYRIA_CHANNELS,
    rate: int = LYRIA_SAMPLE_RATE,
    sample_width: int = LYRIA_SAMPLE_WIDTH,
) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def _ensure_ssl_cert_file() -> None:
    """Point WebSocket TLS verification at certifi when Python lacks system CAs."""
    if os.getenv("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except Exception:
        return
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())


def _with_instrumental_guardrail(prompt: str | None) -> str:
    text = (prompt or "").strip()
    guardrail = "Instrumental only — absolutely no lyrics, no vocals, no humming, no voice whatsoever."
    if not text:
        return guardrail
    lowered = text.lower()
    if "instrumental only" in lowered or "no vocals" in lowered:
        return text
    return f"{text} {guardrail}"


def _resolve_music_config(
    language: str,
    override: dict[str, Any] | None,
) -> types.LiveMusicGenerationConfig:
    if not override:
        return LANGUAGE_BGM_CONFIG.get(language, LANGUAGE_BGM_CONFIG["english"])

    fallback = LANGUAGE_BGM_CONFIG.get(language, LANGUAGE_BGM_CONFIG["english"])

    def _float(name: str, default: float) -> float:
        try:
            return float(override.get(name, default))
        except Exception:
            return default

    def _int(name: str, default: int) -> int:
        try:
            return int(override.get(name, default))
        except Exception:
            return default

    return types.LiveMusicGenerationConfig(
        bpm=max(60, min(200, _int("bpm", getattr(fallback, "bpm", 110) or 110))),
        temperature=max(0.0, min(3.0, _float("temperature", getattr(fallback, "temperature", 1.0) or 1.0))),
        guidance=max(0.0, min(6.0, _float("guidance", getattr(fallback, "guidance", 4.0) or 4.0))),
        density=max(0.0, min(1.0, _float("density", getattr(fallback, "density", 0.72) or 0.72))),
        brightness=max(0.0, min(1.0, _float("brightness", getattr(fallback, "brightness", 0.75) or 0.75))),
        music_generation_mode=types.MusicGenerationMode.QUALITY,
    )
