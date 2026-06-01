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
            temperature=0.1,
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
    """Build a strict, anti-hallucination prompt for multi-clip Veo video planning.

    Each Veo call is stateless — it has zero memory of any previous generation.
    The only way to maintain visual consistency across clips is to:
    1. Define a single Character Bible and embed it VERBATIM in every visual_prompt.
    2. Use frame-continuation at generation time (last frame → next clip's first frame).
    3. Keep the environment anchor identical in every prompt.
    Without these, Veo will generate a different-looking character every clip.
    """

    return f"""
You are a senior AI video director designing a {scene_count}-scene Veo 3 vertical video
for pediatric health education. Each scene is {scene_duration} seconds. The scenes will be
generated INDEPENDENTLY by Veo — the model has NO memory across calls. Visual consistency
depends entirely on how you write the prompts.

═══════════════════════════════════════════════════════
PHASE 1 — DEFINE SHARED VISUAL IDENTITY (do this first)
═══════════════════════════════════════════════════════

Invent one stylized, non-identifiable East Asian animated child character and one fixed environment.
Write these as CHARACTER_BIBLE and ENVIRONMENT_BIBLE strings — then embed them VERBATIM
(copy-paste exactly) inside EVERY visual_prompt you write.

CHARACTER_BIBLE must include ALL of the following — be extremely specific:
- ETHNICITY: East Asian cartoon-stylized features (slightly almond-shaped eyes, soft rounded face, warm-toned skin)
- HAIR: exact color + exact cut (e.g. "short straight jet-black hair with blunt fringe, clean cut above the ears")
- SKIN TONE: a 3D-render descriptor (e.g. "warm honey-beige smooth 3D-illustrated skin, consistent colour across all scenes")
- OUTFIT TOP: exact color + fabric (e.g. "sage-green ribbed knit long-sleeve top")
- OUTFIT BOTTOM: exact color + fabric (e.g. "cream cotton straight-leg pants")
- AGE: 4–6 years, oversized rounded head, large soft eyes, NO realistic face
- STYLE: "premium soft 3D illustrated" — NEVER anime, NEVER realistic, NEVER flat vector, NEVER CGI-movie style
- Add: "character appearance MUST NOT change between scenes — identical hair, skin, outfit, and face shape in every shot"

ENVIRONMENT_BIBLE must include ALL of the following:
- Setting: warm modern Korean-style living room, cream walls, light oak wood floor
- Furniture: cream-white round upholstered sofa only — no other furniture visible
- Lighting: large window on the LEFT side only, soft warm daylight, gentle ambient fill, NO harsh shadows, NO overhead lighting, NO multiple light sources
- No hospital, clinic, outdoor, or any other setting — SAME ROOM in every scene
- Add: "environment layout MUST NOT change between scenes — same wall colour, same floor, same window position"

═══════════════════════════════════════════════════════
PHASE 2 — DESIGN {scene_count} VISUAL SCENES
═══════════════════════════════════════════════════════

Scene progression (follow this exact camera arc):
- Scene 01: Wide establishing shot — child visible in full room environment, calm resting pose
- Scene 02: Medium shot — gentle parent care action (only hands visible, no parent face), child expression changes
- Scene 03: Close-up of care object or body detail — thermometer, warm cup, blanket texture, or hands-on-forehead
- Scene 04: Wide resolution shot — child with calm resolved expression, room still visible

For each visual_prompt:
1. START with the literal style anchor: "Premium 3D animated pediatric explainer, vertical 9:16, "
2. IMMEDIATELY follow with CHARACTER_BIBLE verbatim (copy the exact string you defined in Phase 1)
3. IMMEDIATELY follow with ENVIRONMENT_BIBLE verbatim (copy the exact string you defined in Phase 1)
4. Then describe the scene-specific action, camera angle, movement, and lighting emphasis
5. END with: "no on-screen text, no labels, no readable letters, lower third left clean for subtitles."
6. Keep total prompt length between 60–100 words

═══════════════════════════════════════════════════════
ANTI-HALLUCINATION ENFORCEMENT — MANDATORY FOR EVERY SCENE
═══════════════════════════════════════════════════════

These rules override everything else. Violating any rule is a generation failure.

CHARACTER LOCK (zero tolerance for drift):
- Every visual_prompt MUST start with CHARACTER_BIBLE verbatim (copy exactly — no paraphrasing, no shortening)
- Every visual_prompt MUST include ENVIRONMENT_BIBLE verbatim immediately after CHARACTER_BIBLE
- NEVER write "the child" or "the same child" — always include the full character description
- NEVER change: hair color, hair length, hair style, skin tone, outfit color, outfit type, face shape
- NEVER add new characters of any kind: no doctors, nurses, parents (faces), siblings, strangers, background figures, other children
- NEVER change the character's apparent ethnicity or racial features between scenes
- The character MUST look East Asian cartoon-stylized in every scene — not Caucasian, not ambiguous

ENVIRONMENT LOCK (zero tolerance for scene change):
- SAME ROOM in every scene — never change to outdoor, hospital, clinic, school, or any other setting
- SAME window position (left side) and lighting direction in every scene
- NEVER add objects not present in ENVIRONMENT_BIBLE: no toys on floor, no tables, no bookshelves, no decorations
- Care objects (thermometer, warm cup, blanket) must be held in frame — not placed on new surfaces

OBJECT HALLUCINATION PREVENTION:
- If a care object is needed (thermometer, cup), describe it VERY specifically: size, color, material
- Use ONE object per close-up scene — never multiple care objects in the same frame
- NEVER show: stethoscopes, IV bags, syringes, pills, hospital beds, monitors, oxygen masks, medical charts
- NEVER show: anatomy diagrams, lung cross-sections, throat interior, X-rays, body tissue
- NEVER show: smartphones, screens, tablets, or any device with a display

STYLE LOCK (no style drift):
- Premium soft 3D illustrated style ONLY — never photo-realistic, never anime, never flat vector
- NEVER use words in prompts that imply realism: "photorealistic", "real skin", "lifelike", "detailed texture"
- NEVER use: "glowing", "magical", "sci-fi", "orbs", "energy waves", "aura", "sparkles", "neon"
- Every prompt must name: exact camera position, exact subject action, exact lighting

NEGATIVE PROMPT (must include this base in every scene, add scene-specific items):
"text, captions, labels, letters, numbers, watermark, logo, photorealistic child, real person, Caucasian child, western cartoon style, blonde hair, blue eyes, different skin tone, different outfit, different hair, new character, doctor, nurse, hospital, clinic, emergency room, stethoscope, IV bag, syringe, x-ray, lung diagram, anatomy, infected tissue, mucus, blood, needles, fear expression, panic, glowing orbs, sci-fi effects, dark lighting, red warning signs, multiple characters, background people, outdoor, different room, smartphone, screen, tablet"

═══════════════════════════════════════════════════════
PHASE 3 — WRITE LOCALIZED CAPTION AND TTS TRACKS
═══════════════════════════════════════════════════════

Write three localized tracks (english, korean, spanish) aligned to the same scene_id order.

CAPTION TEXT RULES (shown on screen):
- caption_text is a short on-screen subtitle — one clear phrase per scene, naturally readable
- Write it as a COMPLETE thought, not a truncated sentence — if it doesn't fit, shorten the idea not the words
- NO character limits imposed — but keep it short enough to read in {scene_duration} seconds
- Natural language only: no abbreviated medical jargon, no hashtags, no emojis, no markdown
- Korean captions: write in natural spoken Korean (구어체), not formal written Korean (문어체)
  - Use 이에요/예요 not 입니다, use 거예요 not 것입니다, use 줄 수 있어요 not 줄 수 있습니다
  - Use particles and sentence endings that sound natural when heard aloud, not read
  - Example bad: "항생제는 세균성 감염에만 효과가 있습니다" → Good: "항생제는 세균 감염에만 효과가 있어요"
- English captions: conversational, warm parent-facing tone — "Your child..." not "Children..."
- Spanish captions: natural spoken Spanish — use indicative present, avoid overly formal subjunctive
- Medically cautious language: use "can", "may", "podría", "수 있어요" — not "is", "will", "always"

TTS TEXT RULES (narrated aloud, not shown):
- tts_text is what the voice will say out loud for this scene
- Write it as a complete, naturally speakable sentence — this becomes the AUDIO for this scene
- Length target: 1–3 sentences speakable in {scene_duration}–{scene_duration + 4} seconds at natural pace
- tts_text MUST match the meaning of caption_text — it may add natural connective phrases but NOT new medical claims
- Korean tts_text: 완전한 자연스러운 구어체 문장. 정보 전달이 명확하고 부드럽게 들려야 함
- DO NOT add any claim absent from the script. Preserve medical uncertainty from the source.
- Must cover the medical core across the 4 scenes: 1) what the symptom may mean, 2) antibiotic scope, 3) cautious lung/chest point if present, 4) red flags/follow-up
- Final scene tts_text must mention the safety caveat / doctor follow-up if the script contains one

═══════════════════════════════════════════════════════
SELF-CHECK BEFORE RETURNING JSON
═══════════════════════════════════════════════════════

Before outputting JSON, verify:
[ ] CHARACTER_BIBLE appears verbatim in every visual_prompt
[ ] ENVIRONMENT_BIBLE appears verbatim in every visual_prompt
[ ] No prompt references "same character as before" or "same room as before" — it must be explicit
[ ] No readable text, labels, or numbers requested in any prompt
[ ] Negative prompts are specific and present in all scenes
[ ] All caption_text values are within character limits
[ ] localized_tracks have exactly {scene_count} entries each in the same scene_id order

═══════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════

Return valid JSON only. No Markdown fences. Use this exact top-level shape:

{{
  "video_plan": {{
    "aspect_ratio": "9:16",
    "scene_count": {scene_count},
    "scene_duration_seconds": {scene_duration},
    "audio_strategy": "gemini_tts",
    "visual_strategy": "shared_language_neutral_veo"
  }},
  "character_bible": "...",
  "environment_bible": "...",
  "visual_scenes": [
    {{
      "scene_id": "scene_01",
      "duration_seconds": {scene_duration},
      "visual_prompt": "Premium 3D animated pediatric explainer, vertical 9:16, [CHARACTER_BIBLE verbatim], [ENVIRONMENT_BIBLE verbatim], [scene-specific action and camera], no on-screen text, no labels, no readable letters, lower third left clean for subtitles.",
      "negative_prompt": "text, captions, labels, letters, numbers, watermark, logo, realistic child, identifiable minor, hospital patient, emergency room, stethoscope, IV bag, syringe, x-ray, infected lungs, mucus close-up, blood, needles, panic, glowing orbs, sci-fi effects, dark lighting, red warning signs, doctor in frame, new characters"
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

Scripts JSON:
{json.dumps(scripts, ensure_ascii=False, indent=2)}
""".strip()
