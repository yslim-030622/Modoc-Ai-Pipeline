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
            temperature=0.2,
        ),
    )
    raw_text = response.text or ""
    try:
        parsed = parse_json_response(raw_text)
    except Exception as exc:
        raise GeminiResponseParseError(raw_text, f"Gemini returned invalid video-plan JSON: {exc}") from exc
    return VideoPlanGeneration(parsed=parsed, raw_text=raw_text)


# ─────────────────────────────────────────────────────────────────────────────
# Compact character / environment anchors
# ─────────────────────────────────────────────────────────────────────────────

CHARACTER_ANCHOR = (
    "East Asian 3D-illustrated cartoon child (age 4-6), oversized round head, "
    "large soft expressive eyes, jet-black straight fringe hair, warm honey-beige skin, "
    "sage-green knit long-sleeve top, cream straight-leg pants. "
    "Premium soft 3D illustration style — NOT anime, NOT photo-realistic, NOT flat vector. "
    "IDENTICAL character appearance in every scene."
)

# No fixed environment — Gemini selects per-row based on medical topic.
ENVIRONMENT_OPTIONS = (
    "warm Korean living room | child's cozy bedroom | bright modern kitchen | "
    "sunlit outdoor garden | warm bathroom with soft towels"
)

NEGATIVE_BASE = (
    "any text, any writing, any letters, any characters, any words, any numbers, "
    "Korean text, Korean characters, Korean subtitles, Korean captions, Korean writing, "
    "CJK characters, Asian script, on-screen text, baked-in subtitles, hardcoded captions, "
    "lower-third text, speech bubbles, dialogue boxes, watermark, logo, "
    "realistic child, real person, Caucasian child, blonde hair, blue eyes, "
    "different outfit, different hair, talking mouth, open mouth, speech, lip movement, "
    "extra limbs, deformed hands, morphing body, distorted face, body horror, "
    "floating objects, random props, unexpected furniture, clutter, "
    "doctor, nurse, hospital, clinic, stethoscope, syringe, IV bag, x-ray, "
    "anatomy diagram, blood, needles, multiple characters, "
    "sudden brightness change, flickering light, harsh shadows, "
    "smartphone, screen, sci-fi effects, glowing orbs, dark lighting, "
    "abrupt cut, motion blur artifact, duplicate character"
)


def build_video_plan_prompt(
    *,
    scripts: dict[str, Any],
    scene_count: int,
    scene_duration: int,
) -> str:
    """Build a cinematic Veo 3.1 video plan using the 5-part prompt formula.

    Key principles:
    - Character NEVER talks (TTS narrates, mouth stays closed)
    - Environment varies per medical topic — no more same-room-every-time
    - Visual prompts are short & cinematic (5-part formula) for best Veo quality
    - TTS length limits are calculated from scene_duration to target ~20s total
    """
    korean_char_limit = max(16, int(scene_duration * 3.0))
    english_word_limit = max(9, int(scene_duration * 2.0))
    spanish_word_limit = max(10, int(scene_duration * 2.2))

    return f"""
You are a senior 3D animated video director for a pediatric health education channel.
Plan a {scene_count}-scene Veo 3.1 vertical short (9:16). Each scene is {scene_duration}s.
Total target: {scene_count * scene_duration}s.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — PICK AN ENVIRONMENT (do this FIRST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Read the scripts. Choose the ONE environment from the list below that best matches
the medical topic. Use this SAME environment in all {scene_count} scenes.

Available environments:
  {ENVIRONMENT_OPTIONS}

Rules:
- Match the topic: fever/rest → bedroom | eating/feeding → kitchen | outdoor activity → garden
- NEVER: hospital, clinic, doctor's office, medical setting
- The chosen environment gives each video its unique visual identity

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE NO-TALKING RULE (never violate)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The character MUST NOT appear to speak. Mouth always CLOSED. No lip movement.
The TTS audio narrates — unsynced mouth animation looks broken.

Express content through: facial expressions | body actions | care gestures | eye contact.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHARACTER (copy verbatim into every visual_prompt)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHARACTER_ANCHOR: "{CHARACTER_ANCHOR}"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL PROMPT FORMAT — 5-PART CINEMATIC FORMULA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every visual_prompt = [Camera] + [CHARACTER_ANCHOR] + [Action+Expression] + [Environment] + [End tag]

[Camera]: Wide shot | Medium shot (waist up) | Medium close-up (chest up) | Close-up
[CHARACTER_ANCHOR]: copy the full CHARACTER_ANCHOR text above verbatim
[Action+Expression]: ONE specific action + emotion that illustrates the scene content
  - scene_01: initial symptom or relatable moment — mildly concerned, eyebrows raised
  - scene_02: care action or key fact — curious/attentive, focused
  - scene_03: moment of understanding — wide eyes, soft expression
  - scene_04: calm resolution — gentle smile, relaxed
[Environment]: Your chosen environment, described warmly and specifically
[End tag]: always end with "consistent warm soft lighting, smooth continuous motion, mouth closed, NO TEXT ANYWHERE IN FRAME, completely clean video"

CAMERA PROGRESSION — include movement for cinematic flow:
  scene_01 → "Wide shot, slow gentle push-in" (establish full body + environment)
  scene_02 → "Medium shot, subtle pan left to right" (action shot, waist up)
  scene_03 → "Medium close-up, slow dolly in" (expressive reaction, chest up)
  scene_04 → "Wide shot, slow gentle pull-back" (resolved, calm atmosphere)

VISUAL CONTINUITY (critical for smooth clip stitching):
  - Same environment, same lighting angle across all 4 scenes
  - Add "consistent warm soft lighting" to every prompt
  - The character should feel like they are in one continuous moment
  - Avoid hard pose resets between scenes — use gradual posture changes

OBJECTS (max one per scene, home objects only): soft blanket | small thermometer | cup of water
NEVER: medicine, pills, syringes, medical devices, phones, screens, random background items

Keep each visual_prompt under 280 characters total. Concise prompts produce better Veo quality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENE CONTENT MAPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  scene_01 → hook
  scene_02 → body[0]
  scene_03 → body[1]
  scene_04 → safety_caveat

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TTS TEXT RULES — NO HALLUCINATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tts_text = what the TTS voice says aloud. MUST come from the scripts. No invented facts.

HARD LIMITS per scene (clip is {scene_duration}s — stay under):
  Korean:  ≤ {korean_char_limit} characters  (Korean TTS is slow — cut aggressively)
  English: ≤ {english_word_limit} words
  Spanish: ≤ {spanish_word_limit} words

Korean example for {scene_duration}s scene:
  BAD (too long): "새로운 음식이나 변비가 역류를 유발할 수 있어요" → cuts off
  GOOD (fits):    "변비가 역류 원인이 될 수 있어요" → natural pace

caption_text = tts_text (identical — used as subtitle overlay)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELF-CHECK BEFORE RETURNING JSON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ ] Environment chosen from the options list and used consistently
[ ] CHARACTER_ANCHOR appears verbatim in every visual_prompt
[ ] Each scene has a DIFFERENT camera angle AND different action
[ ] No visual_prompt mentions talking, speaking, mouth moving, lip movement
[ ] Each visual_prompt ends with "mouth closed, NO TEXT ANYWHERE IN FRAME, completely clean video"
[ ] Korean tts_text ≤ {korean_char_limit} characters
[ ] English tts_text ≤ {english_word_limit} words
[ ] No tts_text invents medical claims not in scripts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — valid JSON only, no Markdown fences
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
  "video_plan": {{
    "aspect_ratio": "9:16",
    "scene_count": {scene_count},
    "scene_duration_seconds": {scene_duration},
    "chosen_environment": "...",
    "audio_strategy": "gemini_tts",
    "visual_strategy": "cinematic_5part_no_talking"
  }},
  "character_bible": "{CHARACTER_ANCHOR}",
  "visual_scenes": [
    {{
      "scene_id": "scene_01",
      "duration_seconds": {scene_duration},
      "visual_prompt": "Wide shot, [CHARACTER_ANCHOR], [action + expression], [chosen environment described specifically], consistent warm soft lighting, smooth continuous motion, mouth closed, NO TEXT ANYWHERE IN FRAME, completely clean video.",
      "negative_prompt": "{NEGATIVE_BASE}"
    }}
  ],
  "localized_tracks": {{
    "english": [
      {{
        "scene_id": "scene_01",
        "language": "english",
        "duration_seconds": {scene_duration},
        "caption_text": "...",
        "tts_text": "..."
      }}
    ],
    "korean": [],
    "spanish": []
  }}
}}

Scripts:
{json.dumps(scripts, ensure_ascii=False, indent=2)}
""".strip()
