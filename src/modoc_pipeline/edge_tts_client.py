"""Free TTS via Microsoft Edge TTS (edge-tts library).

edge-tts is completely free — no API key required.
Install: pip install edge-tts  OR  pip install -e ".[meme]"

Voices used:
  Korean:  ko-KR-SunHiNeural  (warm female)
  English: en-US-JennyNeural  (friendly female)
  Spanish: es-MX-DaliaNeural  (natural female, Latin American)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

EDGE_TTS_VOICES: dict[str, str] = {
    "korean": "ko-KR-SunHiNeural",
    "english": "en-US-JennyNeural",
    "spanish": "es-MX-DaliaNeural",
}


def generate_meme_tts(
    *,
    meme_plan: dict[str, Any],
    output_dir: Path,
    languages: set[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """Generate one MP3 per scene per language using edge-tts.

    Returns {language: {scene_id: mp3_path}}.
    Scenes with empty tts_text are skipped silently.
    Raises ImportError with a helpful message if edge-tts is not installed.
    """
    try:
        import edge_tts  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "edge-tts is not installed. Run: pip install edge-tts\n"
            "Or: pip install -e '.[meme]'"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, Path]] = {}

    for lang_key in ("english", "korean", "spanish"):
        if languages is not None and lang_key not in languages:
            continue
        lang_plan = meme_plan.get(lang_key)
        if not lang_plan:
            continue

        voice = EDGE_TTS_VOICES.get(lang_key, EDGE_TTS_VOICES["english"])
        lang_dir = output_dir / lang_key
        lang_dir.mkdir(parents=True, exist_ok=True)
        scene_paths: dict[str, Path] = {}

        for scene in lang_plan.get("scenes", []):
            scene_id = scene.get("scene_id", "")
            tts_text = str(scene.get("tts_text", "")).strip()
            if not tts_text or not scene_id:
                continue

            output_path = lang_dir / f"{scene_id}.mp3"
            print(f"  TTS {lang_key}/{scene_id}: {tts_text[:60]}...")
            try:
                asyncio.run(_synthesize(text=tts_text, voice=voice, output_path=output_path))
                scene_paths[scene_id] = output_path
            except Exception as exc:
                log.warning("edge-tts failed for %s/%s: %s", lang_key, scene_id, exc)

        if scene_paths:
            result[lang_key] = scene_paths

    return result


async def _synthesize(*, text: str, voice: str, output_path: Path) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))
