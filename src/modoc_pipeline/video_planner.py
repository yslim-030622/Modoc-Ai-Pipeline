"""Scene planning for turning generated scripts into Veo prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from .gemini_client import GeminiResponseParseError, parse_json_response


@dataclass(frozen=True)
class VideoPlanGeneration:
    """Parsed video plan plus raw model text for debugging."""

    parsed: dict[str, Any]
    raw_text: str


def generate_video_plan(
    *,
    api_key: str,
    model: str,
    scripts: dict[str, Any],
    scene_count: int,
    scene_duration: int,
) -> VideoPlanGeneration:
    client = genai.Client(api_key=api_key)
    prompt = build_video_plan_prompt(
        scripts=scripts,
        scene_count=scene_count,
        scene_duration=scene_duration,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.5,
        ),
    )
    raw_text = response.text or ""
    try:
        parsed = parse_json_response(raw_text)
    except Exception as exc:
        raise GeminiResponseParseError(raw_text, f"Gemini returned invalid video-plan JSON: {exc}") from exc
    return VideoPlanGeneration(parsed=parsed, raw_text=raw_text)


def build_video_plan_prompt(
    *,
    scripts: dict[str, Any],
    scene_count: int,
    scene_duration: int,
) -> str:
    """Create a constrained prompt for child-safe vertical video planning.

    The plan is intentionally scene-based because Veo is better suited for
    short clips than a single long Shorts-length generation. Keeping subtitles
    and voiceover intent next to the prompt lets later stages render a usable
    MP4 even if the generated audio needs human review.
    """

    return f"""
You are planning vertical short-form pediatric education videos.

Create exactly {scene_count} scenes per language from the provided scripts.
Each scene must be {scene_duration} seconds. The visual prompts will be sent to
Veo for 9:16 video generation.

Safety and content constraints:
- Do not request real children, identifiable minors, hospital patients, or
  realistic depictions of distressed children.
- Prefer bright, calm, parent-friendly educational visuals: clean illustrated
  diagrams, household care objects, simple clinic-safe environments, and gentle
  animated educational metaphors.
- Do not add new medical advice beyond the script.
- Avoid scary, graphic, invasive, or diagnosis-like visuals.
- Avoid dark backgrounds, red X warnings, panic imagery, emergency rooms,
  needles, realistic organs, and alarming icons unless the script explicitly
  requires urgent care.
- Do not ask Veo to generate readable text, numbers, calendars, labels, or UI
  copy inside the image; rendered captions will carry the message.
- Preserve medical uncertainty. Never rewrite "less likely" as "no", "never",
  "definitely", or any other diagnosis-like certainty.
- Keep each prompt concise enough for a video model.
- Keep subtitle_text short, plain, and medically cautious. Each subtitle should
  be one sentence or sentence fragment under 80 characters when possible.

Return valid JSON only. Do not include Markdown fences.
Use this exact top-level shape:
{{
  "video_plan": {{
    "aspect_ratio": "9:16",
    "scene_count": {scene_count},
    "scene_duration_seconds": {scene_duration},
    "audio_strategy": "veo_generated_audio"
  }},
  "scenes": {{
    "english": [
      {{
        "scene_id": "scene_01",
        "language": "english",
        "duration_seconds": {scene_duration},
        "visual_prompt": "...",
        "negative_prompt": "...",
        "subtitle_text": "...",
        "voiceover_intent": "..."
      }}
    ],
    "korean": [],
    "spanish": []
  }}
}}

Scripts JSON:
{json.dumps(scripts, ensure_ascii=False, indent=2)}
""".strip()
