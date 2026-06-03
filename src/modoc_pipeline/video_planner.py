"""Scene planning for turning generated scripts into Veo prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from .gemini_client import GeminiResponseParseError, parse_json_response
from .schemas import VideoPlanPackage


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
    grounding_report: dict[str, Any] | None = None,
    scene_count: int,
    scene_duration: int,
    repair_instructions: str = "",
    previous_payload: dict[str, Any] | None = None,
) -> VideoPlanGeneration:
    client = genai.Client(api_key=api_key)
    prompt = build_video_plan_prompt(
        scripts=scripts,
        grounding_report=grounding_report,
        scene_count=scene_count,
        scene_duration=scene_duration,
        repair_instructions=repair_instructions,
        previous_payload=previous_payload,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VideoPlanPackage,
            temperature=0.2,
        ),
    )
    raw_text = response.text or ""
    try:
        plan = VideoPlanPackage.model_validate_json(raw_text)
    except ValidationError as exc:
        try:
            plan = VideoPlanPackage.model_validate(parse_json_response(raw_text))
        except Exception as inner:
            raise GeminiResponseParseError(
                raw_text,
                f"Gemini returned invalid video-plan JSON: {exc}; fallback parse failed: {inner}",
            ) from inner
    except Exception as exc:
        raise GeminiResponseParseError(raw_text, f"Gemini returned invalid video-plan JSON: {exc}") from exc
    parsed = assemble_video_plan(plan, scripts=scripts)
    return VideoPlanGeneration(parsed=parsed, raw_text=raw_text)


# ─────────────────────────────────────────────────────────────────────────────
# Compact visual anchors
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CHARACTER_BIBLE = {
    "name": "MoDoc Guide",
    "style_anchor": (
        "controlled 2D character-plus-infographic animation, warm pediatric health palette, paper texture"
    ),
    "character_description": (
        "faceless teal MoDoc guide marker, simple round blank icon, no facial features, no limbs, flat 2D"
    ),
    "palette": ["warm off-white", "teal", "coral", "sunny yellow", "soft navy"],
    "background": "plain warm off-white paper texture background, no rooms, no furniture",
    "forbidden_elements": [
        "realistic child", "baby", "doctor", "hospital", "clinic", "nursery", "crib",
        "toy", "mouth", "lips", "teeth", "speech bubble", "text", "logo", "watermark",
    ],
}

# Infographic props by topic — Gemini picks 2-3 that match the narration.
PROP_GUIDE = """\
  Motor / physical development →
    footprint trail, stepping-stone path, balance arc, strength meter bar, milestone marker, neural web pattern

  Nutrition / supplements →
    calcium particle cluster, vitamin D sunburst beam, absorption flow arrow,
    nutrient stack icon, deficiency gap shape, sufficiency fill bar

  Vaccines / immunity →
    shield icon, calendar grid with checkmark, protection dome, schedule timeline, immunity ring

  Symptoms / concerns →
    alert triangle, temperature gradient bar, symptom checklist icon, severity scale, color-coded warning zone

  Digestion / hydration →
    water droplet cluster, hydration fill bar, flow arc, fiber particle stream,
    gut flow wave (abstract, no anatomy), smooth passage indicator, liquid level icon

  Medical evaluation / checkup →
    evaluation clipboard icon, assessment grid, growth curve arc (no numbers), checkup meter

  General (use only when nothing above fits) →
    question-to-answer arc, progress path, soft reveal circle\
"""

NEGATIVE_BASE = (
    "text, captions, subtitles, letters, numbers, logo, watermark, speech bubble, "
    "mouth, lips, teeth, tongue, speaking mouth, lip sync, realistic face, eyes with pupils, "
    "human, person, child, baby, infant, doctor, nurse, patient, "
    "hands, fingers, arms, 3D render, clay render, plush toy, "
    "crib, cot, nursery, baby bed, bed rails, toy blocks, toy, apple, plant, picture frame, "
    "bedroom, living room, kitchen, hospital, clinic, doctor, nurse, medicine, syringe, "
    "stethoscope, pill, capsule, tablet, injection, bandage, "
    "blood, anatomy diagram, organ, stomach, body part, scary mood, clutter, flicker"
)


def build_video_plan_prompt(
    *,
    scripts: dict[str, Any],
    grounding_report: dict[str, Any] | None = None,
    scene_count: int,
    scene_duration: int,
    repair_instructions: str = "",
    previous_payload: dict[str, Any] | None = None,
) -> str:
    """Build a structured planning prompt; code assembles final Veo prompts."""
    # The renderer extends each visual clip to actual TTS duration, so these
    # limits should prevent rambling without forcing medically meaningful words
    # out of compact Korean/Spanish translations.
    korean_char_limit = max(34, int(scene_duration * 5.5))
    english_word_limit = max(16, int(scene_duration * 2.7))
    spanish_word_limit = max(18, int(scene_duration * 3.0))
    grounding_text = json.dumps(grounding_report or {}, ensure_ascii=False, indent=2)
    repair_text = ""
    if repair_instructions:
        repair_text = (
            "\nRepair instructions from Gemini quality gate:\n"
            f"{repair_instructions}\n\n"
            "Previous failed JSON payload to repair:\n"
            f"{json.dumps(previous_payload or {}, ensure_ascii=False, indent=2)}\n"
            "Repair the previous payload instead of starting over. Preserve scene IDs, "
            "visual continuity, and safe medical meaning. Change only the fields needed "
            "to satisfy the quality gate.\n"
        )

    return f"""
You are a senior short-form creative director for a pediatric health education channel.
Plan a {scene_count}-scene Veo 3.1 vertical short (9:16). Each scene is {scene_duration}s.
Total target: {scene_count * scene_duration}s.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — READ THE SCRIPT AND PICK VISUAL BEATS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Read the scripts. Identify the medical topic. Pick 2-3 concrete infographic PROPS
from the list below that directly represent what the narration describes. Use them
across all {scene_count} scenes with the same recurring faceless MoDoc guide marker.

{PROP_GUIDE}

ACTION WRITING RULE — this is the most important rule:
Each scene's "action" field must describe what the viewer SEES, not what the concept MEANS.
Use physical motion verbs: flows, grows, fills, splits, expands, pulses, reveals, fades, rises.

  WRONG: "gut flow wave represents constipation difficulty"   ← Veo ignores abstract concepts
  RIGHT: "a rounded tube icon with a smooth blue stream slows to a stop as a red blockage
          wedge slides in from the right, the stream pools behind it"

  WRONG: "calcium interacts with vitamin D"
  RIGHT: "small calcium dot cluster orbits slowly, then a warm golden sunburst beam sweeps in
          from the upper right, the dots are pulled toward it and dissolve into the beam"

Each scene needs a mini story arc — establish the prop, show the change, show the result:
  [0-2s] Establish the prop in resting state
  [2-4s] Something changes (blockage appears, beam arrives, meter fills, triangle reveals)
  [4-6s] Show the result or consequence

VISUAL STYLE:
  - Use one recurring faceless teal MoDoc guide marker plus infographic props.
  - The guide marker has no eyes, mouth, lips, teeth, hands, fingers, or limbs.
  - The guide marker shifts, slides, or pulses beside props; it never acts with a face or hands.
  - No realistic people, no children, no doctors, no rooms, no furniture.
  - The medical meaning must be visible through props and motion, not through random background objects.

ENGAGEMENT STYLE:
  - Add a soft retention beat in every scene: gentle surprise, myth-vs-fact, parent POV, or tiny visual joke.
  - Keep it parent-friendly and calm. No fear hooks and no exaggerated medical certainty.

CAMERA vocabulary — assign one per scene, vary across scenes (Veo 3.1 responds to cinematography language):
  "static camera, slow dolly-in, wide vertical composition"
  "camera pans left to right, split composition"
  "static camera, centered composition, slow reveal"
  "camera slowly pulls back, clean vertical composition"

VISUAL CONTINUITY:
  - Same faceless MoDoc guide marker, same props, same warm off-white paper texture across all scenes
  - Each scene shows the SAME props in a new state or phase
  - No hard resets — scenes feel like one continuous story

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCENE CONTENT MAPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  scene_01 → hook
  scene_02 → body[0]
  scene_03 → body[1]
  scene_04 → safety_caveat
  scene_05 → cta (only if {scene_count} scenes are requested)

For scene_01 hook, preserve the same core meaning in every language:
age + growth + language/speech concern. Compact wording is fine, but do not
drop one of those three concepts in Korean or Spanish.

SCENE CONTRACT RULE:
For every scene, create scene_contract:
  narration_source: hook, body_0, body_1, body_2, safety_caveat, or cta
  viewer_should_understand: what should be clear even with subtitles muted
  must_show: 2-4 concrete visible objects/actions, never abstract nouns only
  must_not_show: forbidden objects likely to hallucinate for this topic
  meme_device: gentle_surprise, myth_vs_fact, parent_pov, or tiny_visual_joke

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TTS TEXT RULES — NO HALLUCINATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tts_text = what the TTS voice says aloud. MUST come from the scripts. No invented facts.
For each scene, first define meaning_contract: 2-4 short English concepts that
MUST appear in every language's tts_text/caption_text for that scene.
Examples:
  scene_01 hook → ["three-year-old", "growth", "language or speech"]
  safety scene → ["pediatrician or clinician", "growth and language evaluation"]

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

[ ] Props chosen from the topic list match the narration content
[ ] Each scene has a DIFFERENT camera angle AND different action
[ ] Korean tts_text ≤ {korean_char_limit} characters
[ ] English tts_text ≤ {english_word_limit} words
[ ] Spanish tts_text ≤ {spanish_word_limit} words
[ ] No tts_text invents medical claims not in scripts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — valid JSON only, no Markdown fences
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
  "video_plan": {{
    "aspect_ratio": "9:16",
    "scene_count": {scene_count},
    "scene_duration_seconds": {scene_duration},
    "chosen_environment": "plain paper background + recurring guide + topic props",
    "audio_strategy": "gemini_tts",
    "visual_strategy": "controlled_character_infographic"
  }},
  "character_bible": {{
    "name": "MoDoc Guide",
    "style_anchor": "controlled 2D character-plus-infographic short-form animation, clean crisp vector shapes, warm pediatric health palette, soft paper texture",
    "character_description": "faceless teal MoDoc guide marker, simple round blank icon, no facial features, no limbs, flat 2D",
    "palette": ["warm off-white", "teal", "coral", "sunny yellow", "soft navy"],
    "background": "plain warm off-white paper texture background, no rooms, no furniture",
    "forbidden_elements": ["realistic child", "baby", "doctor", "hospital", "nursery", "crib", "toy", "mouth", "lips", "teeth", "text"]
  }},
  "visual_blueprints": [
    {{
      "scene_id": "scene_01",
      "meaning_contract": ["constipation", "hard stool", "after solid foods"],
      "scene_contract": {{
        "narration_source": "hook",
        "viewer_should_understand": "A parent question appears: solid foods can change stool comfort.",
        "must_show": ["faceless teal MoDoc guide marker beside a rounded tube icon", "blue flow slows", "amber blocks stack up"],
        "must_not_show": ["baby", "crib", "doctor", "medicine", "text"],
        "meme_device": "parent_pov"
      }},
      "camera": "static camera, slow dolly-in, wide vertical composition",
      "action": "a rounded tube icon sits centered, a smooth blue stream flows through it steadily, then slows and thickens into dark amber blocks that stack up — the tube strains and bulges slightly",
      "expression": "slow and deliberate, building tension",
      "environment_detail": "plain warm off-white paper texture background",
      "prop": "tube icon, blue stream, amber blockage blocks"
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

Grounding report:
{grounding_text}
{repair_text}
""".strip()


def assemble_video_plan(plan: VideoPlanPackage, *, scripts: dict[str, Any]) -> dict[str, Any]:
    selected_ids = [scene.scene_id for scene in plan.visual_blueprints]
    tracks = build_localized_tracks_from_scripts(
        scripts=scripts,
        scene_ids=selected_ids,
        scene_duration=plan.video_plan.scene_duration_seconds,
    )

    character_bible = plan.character_bible.model_dump()
    return {
        "video_plan": plan.video_plan.model_dump(),
        "character_bible": character_bible,
        "visual_scenes": [
            {
                "scene_id": scene.scene_id,
                "meaning_contract": scene.meaning_contract,
                "scene_contract": scene.scene_contract.model_dump(),
                "duration_seconds": plan.video_plan.scene_duration_seconds,
                "visual_prompt": build_veo_prompt(
                    scene.model_dump(),
                    plan.video_plan.chosen_environment,
                    character_bible=character_bible,
                ),
                "negative_prompt": NEGATIVE_BASE,
            }
            for scene in plan.visual_blueprints
        ],
        "localized_tracks": tracks,
        "visual_blueprints": [scene.model_dump() for scene in plan.visual_blueprints],
    }


def build_localized_tracks_from_scripts(
    *,
    scripts: dict[str, Any],
    scene_ids: list[str],
    scene_duration: int,
) -> dict[str, list[dict[str, Any]]]:
    """Create narration tracks directly from the approved scripts.

    This avoids a second Gemini summarization pass dropping medical meaning in
    one language while trying to fit short scene-level limits.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for language in ("english", "korean", "spanish"):
        script = scripts.get(language, {})
        scene_text = {
            "scene_01": str(script.get("hook") or "").strip(),
            "scene_02": _body_item(script, 0),
            "scene_03": _body_item(script, 1),
            "scene_04": str(script.get("safety_caveat") or "").strip(),
            "scene_05": str(script.get("cta") or "").strip(),
        }
        tracks: list[dict[str, Any]] = []
        for scene_id in scene_ids:
            text = scene_text.get(scene_id) or _fallback_scene_text(script)
            text = normalize_scene_text(language, scene_id, text, scripts)
            tracks.append(
                {
                    "scene_id": scene_id,
                    "language": language,
                    "duration_seconds": scene_duration,
                    "caption_text": text,
                    "tts_text": text,
                }
            )
        result[language] = tracks
    return result


def _body_item(script: dict[str, Any], index: int) -> str:
    body = script.get("body") or []
    if index < len(body):
        return str(body[index]).strip()
    return ""


def _fallback_scene_text(script: dict[str, Any]) -> str:
    for key in ("safety_caveat", "cta", "hook", "title"):
        text = str(script.get(key) or "").strip()
        if text:
            return text
    return ""


def normalize_scene_text(
    language: str,
    scene_id: str,
    text: str,
    scripts: dict[str, Any],
) -> str:
    if scene_id != "scene_05":
        return text
    english_cta = str(scripts.get("english", {}).get("cta") or "").lower()
    if not any(term in english_cta for term in ("fluid", "hydrat", "water")):
        return text
    lowered = text.lower()
    if language == "korean" and "수분" not in text and "물" not in text:
        return f"{text.rstrip(' .。요')} 수분도 충분히 챙겨 주세요."
    if language == "spanish" and "líquid" not in lowered and "hidrat" not in lowered and "agua" not in lowered:
        return f"{text.rstrip(' .')} Mantenga buena hidratación."
    return text


def build_veo_prompt(
    scene: dict[str, Any],
    chosen_environment: str,
    *,
    character_bible: dict[str, Any] | None = None,
) -> str:
    bible = character_bible or DEFAULT_CHARACTER_BIBLE
    style_anchor = clean_prompt_piece(bible.get("style_anchor") or DEFAULT_CHARACTER_BIBLE["style_anchor"])
    character = sanitize_character_piece(bible.get("character_description") or DEFAULT_CHARACTER_BIBLE["character_description"])
    background = clean_prompt_piece(bible.get("background") or DEFAULT_CHARACTER_BIBLE["background"])
    camera = clean_prompt_piece(scene.get("camera") or "")
    action = sanitize_visual_terms(clean_prompt_piece(scene.get("action") or ""))
    tone = clean_prompt_piece(scene.get("expression") or "")
    prop = clean_prompt_piece(scene.get("prop") or "")
    contract = scene.get("scene_contract") or {}
    must_show = sanitize_visual_terms(clean_prompt_piece(", ".join(contract.get("must_show") or [])))
    meme_device = clean_prompt_piece(contract.get("meme_device") or "")

    # Merge prop into action if not already mentioned — avoids duplication
    content = action
    if prop and prop.lower() not in action.lower():
        content = f"{action}, featuring {prop}"
    if must_show and must_show.lower() not in content.lower():
        content = f"{content}, show {must_show}"

    # Research-backed order: Style → Camera → Content → Light source → Background → Safety
    components = [
        style_anchor,               # Style first — Veo style recognition
        character,                  # Stable recurring visual subject
        camera,                     # Camera second — cinematography control
        content,                    # Physical action + props — content description
        f"soft meme beat: {meme_device}" if meme_device else "",
        tone,                       # Motion mood
        "soft warm light",
        background,
        "mouth-free, lip-sync-free, hand-free, text-free, NO TEXT ANYWHERE IN FRAME",
        "clean vertical 9:16 video",
    ]
    return trim_prompt(", ".join(c for c in components if c))


def clean_prompt_piece(value: Any) -> str:
    text = str(value).strip().strip(".")
    for phrase in (
        "consistent warm soft lighting",
        "smooth continuous motion",
        "slow subtle motion",
        "smooth simple 2D motion",
        "mouth closed",
        "symbol-only visuals",
        "face-free",
        "hand-free",
        "flat 2D only",
        "NO TEXT ANYWHERE IN FRAME",
        "clean vertical video",
    ):
        text = text.replace(phrase, "").replace(phrase.lower(), "")
    banned_words = (
        "crib", "cot", "nursery", "baby bed", "bed rails", "toy blocks",
        "toy", "plant", "apple", "picture frame", "shelf",
        "hands folded", "hand", "finger", "smile",
        "3d", "bedroom", "living room",
        "kitchen", "hospital", "clinic", "doctor", "nurse", "syringe",
        "needle", "medicine",
    )
    lowered = text.lower()
    if any(word in lowered for word in banned_words):
        return ""
    return " ".join(text.replace(",,", ",").split()).strip(" ,")


def sanitize_character_piece(value: Any) -> str:
    text = str(value).strip().strip(".")
    text = sanitize_visual_terms(text)
    # Avoid words that push Veo toward expressive faces or lip movement.
    replacements = {
        "mouthless": "faceless",
        "friendly": "neutral",
        "calm helpful posture": "static marker posture",
        "no lips, no teeth, no hands": "no facial features, no limbs",
        "no hands": "no limbs",
        "hands": "limbs",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return " ".join(text.split())


def sanitize_visual_terms(value: str) -> str:
    replacements = {
        "mouthless MoDoc Guide": "faceless teal MoDoc guide marker",
        "mouthless guide": "faceless guide marker",
        "MoDoc Guide bobs": "MoDoc guide marker gently shifts",
        "guide bobs": "guide marker gently shifts",
        "guide leans": "guide marker slides",
        "The mouthless": "The faceless",
        "No mouth, no speaking": "No facial features or limbs",
        "mouthless": "faceless",
    }
    text = value
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def trim_prompt(prompt: str, *, max_chars: int = 980, max_words: int = 175) -> str:
    required_tail = "mouth-free, lip-sync-free, hand-free, text-free, NO TEXT ANYWHERE IN FRAME, clean vertical 9:16 video"
    words = prompt.split()
    if len(words) > max_words:
        prompt = " ".join(words[:max_words])
    for phrase in ("mouth-free", "lip-sync-free", "hand-free", "text-free", "NO TEXT ANYWHERE IN FRAME", "clean vertical 9:16 video"):
        prompt = prompt.replace(phrase, "")
    prompt = " ".join(prompt.replace(",,", ",").split()).strip(" ,")
    budget = max_chars - len(required_tail) - 2
    if len(prompt) > budget:
        prompt = prompt[:budget].rsplit(" ", 1)[0]
    prompt = f"{prompt}, {required_tail}"
    return prompt.strip(" ,")
