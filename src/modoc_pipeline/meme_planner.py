"""Culture-aware meme plan generation via two-stage Gemini call.

Stage 1: Google Search grounding to find trending meme formats per language.
Stage 2: Structured MemePlan generation using those trends + script content.

Gemini SDK does not allow response_schema + google_search tool in the same
request (same constraint as grounding.py), so the two stages are separated.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from .gemini_client import parse_json_response, GeminiResponseParseError
from .schemas import MemePlan

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemePlanGeneration:
    parsed: dict[str, Any]
    raw_text: str
    search_metadata: dict[str, Any]


def generate_meme_plan(
    *,
    api_key: str,
    model: str,
    scripts: dict[str, Any],
    topic: str,
) -> MemePlanGeneration:
    """Research trending memes per language then generate a MemePlan.

    Two-stage approach:
      1. Google Search grounding → trending meme formats (free-form JSON)
      2. Structured MemePlan generation using Stage 1 results as context
    """
    client = genai.Client(api_key=api_key)

    trends_json, search_metadata = _stage1_search_trends(client, model=model, topic=topic)
    raw_text, parsed = _stage2_generate_plan(
        client,
        model=model,
        scripts=scripts,
        topic=topic,
        trends_json=trends_json,
    )
    return MemePlanGeneration(parsed=parsed, raw_text=raw_text, search_metadata=search_metadata)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Google Search grounding
# ─────────────────────────────────────────────────────────────────────────────

def _stage1_search_trends(
    client: genai.Client,
    *,
    model: str,
    topic: str,
) -> tuple[str, dict[str, Any]]:
    """Return (trends_json_str, search_metadata_dict)."""
    prompt = _build_search_prompt(topic)
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.3,
            ),
        )
        raw = response.text or "{}"
        search_metadata = _extract_search_metadata(response)
        # Strip markdown if present
        try:
            parse_json_response(raw)
            return raw, search_metadata
        except Exception:
            return raw, search_metadata
    except Exception as exc:
        log.warning("Meme trend search failed (%s); using empty trends.", exc)
        return "{}", {}


def _build_search_prompt(topic: str) -> str:
    return f"""
Research current trending meme formats used for health and parenting content
on social media in 2025. Focus on these three cultural contexts:

- Korean (한국): 짤방 (jjal) formats on 에브리타임, 인스타그램, 트위터/X.
  What visual styles and text patterns are trending for health/parenting 짤?
- English (US/global): TikTok, Instagram Reels, Twitter/X.
  What short-form meme formats (POV, reaction, before/after, text-on-video)
  work best for health/parenting education in 2025?
- Spanish (Latin America/Spain): TikTok Latino, Instagram.
  What relatability meme formats ("cuando...", "mamás cuando...") are
  trending for health and parenting content?

Topic context: {topic}

Return a JSON object with exactly three keys: "english", "korean", "spanish".
Each value is a list of 3-5 short strings describing trending meme formats
suitable for health/parenting edutainment. Be specific and current.
Example format:
{{
  "english": ["POV: your toddler has a fever at 2am", "before/after reaction meme"],
  "korean": ["짤: 아이 증상 공감 밈", "SNS 부모 공감 카드뉴스"],
  "spanish": ["cuando el pediatra dice que es normal", "mamás latinas reaccionando"]
}}

Return only the JSON object. No explanation.
""".strip()


def _extract_search_metadata(response: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"queries": [], "citations": []}
    try:
        candidates = getattr(response, "candidates", []) or []
        for candidate in candidates:
            grounding = (
                getattr(candidate, "grounding_metadata", None)
                or getattr(candidate, "groundingMetadata", None)
            )
            if not grounding:
                continue
            for chunk in getattr(grounding, "grounding_chunks", []) or []:
                web = getattr(chunk, "web", None)
                if web:
                    uri = getattr(web, "uri", "") or ""
                    title = getattr(web, "title", "") or ""
                    if uri and {"uri": uri, "title": title} not in metadata["citations"]:
                        metadata["citations"].append({"uri": uri, "title": title})
            for query in getattr(grounding, "web_search_queries", []) or []:
                if query not in metadata["queries"]:
                    metadata["queries"].append(query)
    except Exception as exc:
        log.debug("Could not extract search metadata: %s", exc)
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Structured MemePlan generation
# ─────────────────────────────────────────────────────────────────────────────

def _stage2_generate_plan(
    client: genai.Client,
    *,
    model: str,
    scripts: dict[str, Any],
    topic: str,
    trends_json: str,
) -> tuple[str, dict[str, Any]]:
    prompt = _build_plan_prompt(scripts=scripts, topic=topic, trends_json=trends_json)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MemePlan,
            temperature=0.6,
        ),
    )
    raw_text = response.text or ""
    try:
        parsed = MemePlan.model_validate_json(raw_text).model_dump()
    except ValidationError as exc:
        try:
            parsed = MemePlan.model_validate(parse_json_response(raw_text)).model_dump()
        except Exception as inner:
            raise GeminiResponseParseError(
                raw_text,
                f"Meme plan parse failed: {exc}; fallback failed: {inner}",
            ) from inner
    except Exception as exc:
        raise GeminiResponseParseError(raw_text, f"Meme plan invalid JSON: {exc}") from exc
    return raw_text, parsed


def _build_plan_prompt(
    *,
    scripts: dict[str, Any],
    topic: str,
    trends_json: str,
) -> str:
    scripts_str = json.dumps(scripts, ensure_ascii=False, indent=2)
    return f"""
Create a 4-scene VIRAL meme slideshow for pediatric health education. Each language version must feel native to that culture — humor-first, then education.

Topic: {topic}
Medical facts (use ONLY these, never invent): {scripts_str}
Trending formats found: {trends_json}

STORY ARC (4 scenes):
  Scene 1 HOOK: shocking/funny parent panic moment — must make someone stop scrolling
  Scene 2 TENSION: the relatable wrong assumption parents make (humor angle)
  Scene 3 INSIGHT: the "wait, really?!" medical truth — surprising, reassuring
  Scene 4 RELIEF: simple action + warm close — leaves parent feeling empowered

VISUAL CONSISTENCY (critical — this is the #1 quality issue):
Define one visual_style_anchor AND one character_sheet per language.

character_sheet = locked character description used VERBATIM in every scene prompt.
It must include ALL of:
  - exact hair: color, length, style (e.g. "shoulder-length wavy black hair in a loose bun")
  - exact clothing: color, type, pattern (e.g. "oversized grey hoodie, dark blue joggers")
  - skin tone (e.g. "warm light beige skin")
  - body shape / build (e.g. "petite, slightly rounded face")
  - defining prop (e.g. "always holding a white ceramic mug")
Example: "petite woman, warm light beige skin, shoulder-length wavy black hair in a loose bun, oversized grey hoodie, dark blue joggers, white ceramic mug in hand"

IMAGE PROMPT RULES:
- Every scene prompt MUST open with: "[visual_style_anchor]. [character_sheet full text]. [new action/emotion this scene]."
- ZERO text, letters, signs in the image — pure illustration
- Describe the EMOTIONAL MOMENT: expression, body language, action, lighting
- Good: "[anchor]. [character_sheet]. Now jaw-dropped, pointing at phone with free hand, blue phone glow, dark 3am bedroom."
- Bad: "Same character as scene 1" (useless — model has no memory), "background with bold colors" (too vague)
- Max 90 words. No realistic faces. No medical equipment. 9:16 vertical.

TEXT (displayed over image — keep SHORT and PUNCHY):
- top_text: ≤ 6 words label or "" — cultural format label (see below)
- bottom_text: ≤ 8 words — the punchline or key fact, conversational
- tts_text: ONE spoken sentence, ≤ 10 words, warm and quick — medically accurate

KOREAN style (바이럴 짤방):
- meme_format: "jjal" or "caption_only"
- 반전 format: starts relatable, ends with surprising medical truth
- All text in Korean
- top_text: "새벽 2시 엄마:", "소아과 가기 전:", "육아 꿀팁:" style
- bottom_text: texting-a-friend Korean — "이거 진짜야?" tone
- visual_style_anchor: cute flat 2D Korean webtoon style, soft sage green and warm cream palette
- character_sheet: generate a locked description for a petite young Korean woman. Include: warm ivory skin, shoulder-length straight black hair in a messy bun, oversized mint-green hoodie, light grey sweatpants, holding a white ceramic mug
- tts_text: 자연스럽고 빠른 어투, 10단어 이내

ENGLISH style (TikTok/Reels viral):
- meme_format: "pov" or "reaction"
- "POV:", "Hot take:", or "Nobody warns you:" format
- top_text: "POV:", "Hot take:", "Nobody warns you:" style
- bottom_text: Gen Z/millennial parent tone — slightly snarky then warm
- visual_style_anchor: modern flat illustration, bold teal and warm white palette
- character_sheet: generate a locked description for a tired millennial woman. Include: light peach skin, shoulder-length brown hair in a high messy bun, oversized teal crewneck sweatshirt, black leggings, dark circles under eyes
- tts_text: casual friend voice, ≤ 10 words

SPANISH style (TikTok Latino viral):
- meme_format: "relatability"
- "Cuando el doctor dice..." or "Mamá latina:" format
- All text in Spanish
- top_text: "Cuando:", "Mamá latina:", "El pediatra dice:" style
- bottom_text: warm community voice — validates the mom's instinct while sharing the fact
- visual_style_anchor: warm vibrant flat illustration, ochre and terracotta palette
- character_sheet: generate a locked description for an expressive Latina woman. Include: warm golden-brown skin, voluminous curly dark brown hair past shoulders, floral terracotta blouse, jeans, large gold hoop earrings
- tts_text: cálido y rápido, ≤ 10 palabras

SAFETY: every medical fact in tts_text/bottom_text must trace to source scripts. No fear-based framing.
source_topic: short English phrase.
""".strip()
