"""Culture-aware meme plan generation via staged Gemini calls.

Stage 1: Google Search grounding to find 2026 trends per language.
Stage 2: Generate three creative candidates per language.
Stage 3: Judge and select one candidate per language.
Stage 4: Generate a structured MemePlan from the winning candidates.

Gemini SDK does not allow response_schema + google_search tool in the same
request (same constraint as grounding.py), so Stage 1 stays free-form.
Stages 2-4 use response_schema to enforce structure at the API level.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from .gemini_client import gemini_schema, parse_json_response
from .schemas import CreativeCandidateSet, CreativeScoreSet, MemePlan, TrendResearch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemePlanGeneration:
    parsed: dict[str, Any]
    raw_text: str
    search_metadata: dict[str, Any]
    trend_research: dict[str, Any]
    creative_candidates: dict[str, Any]
    creative_scores: dict[str, Any]


def generate_meme_plan(
    *,
    api_key: str,
    model: str,
    scripts: dict[str, Any],
    topic: str,
) -> MemePlanGeneration:
    client = genai.Client(api_key=api_key)

    trends_json, trend_research, search_metadata = _stage1_search_trends(client, model=model, topic=topic)
    candidates_raw, candidates = _stage2_generate_candidates(
        client,
        model=model,
        scripts=scripts,
        topic=topic,
        trends=trend_research,
    )
    scores_raw, scores = _stage3_judge_candidates(
        client,
        model=model,
        scripts=scripts,
        topic=topic,
        candidates=candidates,
    )
    raw_text, parsed = _stage4_generate_plan(
        client,
        model=model,
        scripts=scripts,
        topic=topic,
        trends_json=trends_json,
        candidates=candidates,
        scores=scores,
    )
    combined_raw = json.dumps(
        {
            "trend_research_raw": trends_json,
            "creative_candidates_raw": candidates_raw,
            "creative_scores_raw": scores_raw,
            "final_meme_plan_raw": raw_text,
        },
        ensure_ascii=False,
        indent=2,
    )
    return MemePlanGeneration(
        parsed=parsed,
        raw_text=combined_raw,
        search_metadata=search_metadata,
        trend_research=trend_research,
        creative_candidates=candidates,
        creative_scores=scores,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Google Search grounding (cannot use response_schema with search tool)
# ─────────────────────────────────────────────────────────────────────────────

def _stage1_search_trends(
    client: genai.Client,
    *,
    model: str,
    topic: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
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
        try:
            parsed = TrendResearch.model_validate(parse_json_response(raw)).model_dump()
        except Exception:
            parsed = TrendResearch().model_dump()
        return raw, parsed, search_metadata
    except Exception as exc:
        log.warning("Meme trend search failed (%s); using empty trends.", exc)
        return "{}", TrendResearch().model_dump(), {}


def _build_search_prompt(topic: str) -> str:
    return f"""
Research current 2026 short-form formats for pediatric health / parenting
edutainment. Use current web results and prioritize the last 30-60 days when
possible. Consider TikTok Next 2026 signals: Reali-TEA, Curiosity Detours, and
Emotional ROI. Focus on native, non-cringe formats for:

- English (US/global): TikTok, Instagram Reels, YouTube Shorts.
- Korean (한국): Instagram Reels, YouTube Shorts, TikTok, parenting communities,
  KakaoTalk-like conversational captions, and Korean 짤/밈 behavior.
- Spanish (Latin America/Spain): TikTok Latino, Instagram Reels, YouTube Shorts.

Topic context: {topic}

Return a JSON object with exactly three keys: "english", "korean", "spanish".
Each value is a list of 3-5 objects with exactly these keys:
- format_name
- platform
- hook_templates: list of short native caption/hook templates
- caption_style
- visual_editing_style
- audio_mood
- why_it_works
- avoid: list of cliches or stale formats to avoid

Example format:
{{
  "english": [{{
    "format_name": "Parent at 2am POV",
    "platform": "TikTok/Reels/Shorts",
    "hook_templates": ["POV: it is 2am and you searched..."],
    "caption_style": "large clean first-frame hook plus short bottom reveal",
    "visual_editing_style": "fast 4-beat faceless illustrated story",
    "audio_mood": "upbeat but soft, no panic",
    "why_it_works": "real parent moment plus useful relief",
    "avoid": ["fake doctor voice", "fear bait"]
  }}],
  "korean": [],
  "spanish": []
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
# Stage 2: Creative candidate generation
# ─────────────────────────────────────────────────────────────────────────────

def _stage2_generate_candidates(
    client: genai.Client,
    *,
    model: str,
    scripts: dict[str, Any],
    topic: str,
    trends: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    prompt = _build_candidate_prompt(scripts=scripts, topic=topic, trends=trends)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_schema(CreativeCandidateSet),
            temperature=0.75,
        ),
    )
    raw_text = response.text or ""
    try:
        parsed = CreativeCandidateSet.model_validate_json(raw_text).model_dump()
    except Exception:
        parsed = _parse_candidate_set(raw_text)
    return raw_text, parsed


def _build_candidate_prompt(*, scripts: dict[str, Any], topic: str, trends: dict[str, Any]) -> str:
    scripts_str = json.dumps(scripts, ensure_ascii=False, indent=2)
    trends_str = json.dumps(trends, ensure_ascii=False, indent=2)
    return f"""
Create exactly 3 creative candidates per language for a Gemini-only pediatric
health short-form slideshow. This is for 2026 short-form culture, not generic
medical education.

Topic: {topic}
Medical facts (use ONLY these; never add claims): {scripts_str}
Trend research: {trends_str}

Rules:
- Output valid JSON only.
- Each candidate must be medically safe, parent-facing, and culturally native.
- Medical safety beats humor. Never use fear bait, diagnosis bait, or definitive reassurance.
- English: use parent-realistic hooks like POV, Nobody warns you, parent at 2am, save this. Avoid overdone Gen-Z slang.
- Korean: use 새벽 육아, 검색하다 더 불안, 맘카페/카톡 말투, 반전 짤.
- Spanish: use Cuando..., Lo que nadie te explica..., El pediatra dijo..., Mamá latina: format.
- visual_style_anchor and character_sheet must be concrete and fully described for image consistency.
- bgm_prompt must be instrumental-only social video music, with no artist names and no vocals.
- BGM must match the medical seriousness. For symptoms, postpartum complications,
  swelling, blood pressure, blood clot concerns, diagnosis, or referral topics,
  use gentle low-density educational music. Do NOT use party, festive, cute,
  comedy, hyperpop, reggaeton, dembow, club, huge drop, or high-energy K-pop cues.
- bgm_config must include bpm, density, brightness, guidance, temperature.
- For serious medical topics, set bgm_config density <= 0.68 and brightness <= 0.76.
- scene_beats must be exactly 4 short beats: hook, tension, insight, relief.
- Captions must not mention unsupported home remedies, folk treatments, emergency
  rooms, or urgent care unless those concepts appear in the medical script.

Return CreativeCandidateSet JSON.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Candidate judging / scoring
# ─────────────────────────────────────────────────────────────────────────────

def _stage3_judge_candidates(
    client: genai.Client,
    *,
    model: str,
    scripts: dict[str, Any],
    topic: str,
    candidates: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    prompt = _build_score_prompt(scripts=scripts, topic=topic, candidates=candidates)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_schema(CreativeScoreSet),
            temperature=0.2,
        ),
    )
    raw_text = response.text or ""
    try:
        parsed = CreativeScoreSet.model_validate_json(raw_text).model_dump()
    except Exception:
        parsed = _parse_score_set(raw_text)
    return raw_text, _ensure_one_selected_per_language(parsed)


def _build_score_prompt(*, scripts: dict[str, Any], topic: str, candidates: dict[str, Any]) -> str:
    return f"""
Judge these creative candidates for a 2026 pediatric health short-form video.

Topic: {topic}
Medical scripts/facts: {json.dumps(scripts, ensure_ascii=False, indent=2)}
Candidates: {json.dumps(candidates, ensure_ascii=False, indent=2)}

Score every candidate 1-5 for:
- hook_strength
- native_fit
- retention
- medical_safety
- not_cringe
- visual_feasibility

Rules:
- Select exactly one candidate per language.
- If medical_safety is below 4, selected must be false even if it is funny.
- Prefer real parent situations, curiosity, and emotional payoff over generic advice.
- Penalize stale templates, stereotypes, visible-text-in-image concepts, and fear framing.

Return CreativeScoreSet JSON only.
""".strip()


def _ensure_one_selected_per_language(scores: dict[str, Any]) -> dict[str, Any]:
    for lang in ("english", "korean", "spanish"):
        items = list(scores.get(lang, []) or [])
        selected = [item for item in items if item.get("selected")]
        if len(selected) == 1:
            continue
        for item in items:
            item["selected"] = False
        safe_items = [item for item in items if int(item.get("medical_safety", 0)) >= 4]
        ranked = safe_items or items
        if ranked:
            best = max(
                ranked,
                key=lambda item: (
                    int(item.get("medical_safety", 0)),
                    int(item.get("hook_strength", 0))
                    + int(item.get("native_fit", 0))
                    + int(item.get("retention", 0))
                    + int(item.get("not_cringe", 0))
                    + int(item.get("visual_feasibility", 0)),
                ),
            )
            best["selected"] = True
        scores[lang] = items
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Final MemePlan generation
# ─────────────────────────────────────────────────────────────────────────────

def _stage4_generate_plan(
    client: genai.Client,
    *,
    model: str,
    scripts: dict[str, Any],
    topic: str,
    trends_json: str,
    candidates: dict[str, Any],
    scores: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    prompt = _build_plan_prompt(
        scripts=scripts,
        topic=topic,
        trends_json=trends_json,
        candidates=candidates,
        scores=scores,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_schema(MemePlan),
            temperature=0.6,
        ),
    )
    raw_text = response.text or ""
    try:
        parsed = MemePlan.model_validate_json(raw_text).model_dump()
    except Exception:
        parsed = _parse_meme_plan(raw_text, topic=topic, candidates=candidates, scores=scores)
    return raw_text, parsed


def _build_plan_prompt(
    *,
    scripts: dict[str, Any],
    topic: str,
    trends_json: str,
    candidates: dict[str, Any],
    scores: dict[str, Any],
) -> str:
    scripts_str = json.dumps(scripts, ensure_ascii=False, indent=2)
    candidates_str = json.dumps(candidates, ensure_ascii=False, indent=2)
    scores_str = json.dumps(scores, ensure_ascii=False, indent=2)
    return f"""
Create the final 4-scene 2026 short-form slideshow plan for pediatric health
education. Use the selected creative candidate per language; do not invent a
new direction.

Topic: {topic}
Medical facts (use ONLY these, never invent): {scripts_str}
Trend research found: {trends_json}
Creative candidates: {candidates_str}
Creative scores and selections: {scores_str}

STORY ARC (4 scenes):
  Scene 1 HOOK: real parent situation / curiosity gap; caption_role="hook"
  Scene 2 TENSION: relatable wrong assumption; caption_role="tension"
  Scene 3 INSIGHT: supported medical truth; caption_role="insight"
  Scene 4 RELIEF: simple action + warm close; caption_role="relief"

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
- top_text: ≤ 6 words label or ""; Korean about ≤12 characters.
- bottom_text: ≤ 8 words; Korean about ≤18 characters.
- tts_text: ONE spoken sentence, ≤ 10 words, warm and quick, medically accurate.
- first scene caption must be readable in about 1 second.

LANGUAGE FIELDS:
- Fill creative_angle, trend_rationale, caption_style, tts_style, bgm_prompt, bgm_config, avoid_cliches from the selected candidate.
- caption_style must be one of: impact, clean_reels, korean_jjal, spanish_social.
- bgm_prompt must be instrumental only, no vocals, no humming, no artist names.
- bgm_config should fit the selected BGM prompt.
- For serious medical topics, BGM must be gentle, low-density, reassuring, and
  voiceover-safe. Avoid festive, cute, dance, reggaeton/dembow, hyperpop, 808,
  club, party, huge-drop, or comedic prompts unless the medical script is clearly
  low-risk and playful.
- Captions must not add unsupported urgency or remedies. Do not mention emergency
  rooms, urgent care, alcohol/romero, massage, 호박즙, or other folk remedies unless
  the source script explicitly supports them.
- English: parent-realistic, not try-hard Gen Z.
- Korean: native Korean text, 새벽 육아/search anxiety/Kakao-like phrasing allowed.
- Spanish: native Spanish text, warm discovery/relief tone.

PER-LANGUAGE VISUAL IDENTITY (use these locked designs for consistency):

Korean:
  - visual_style_anchor: "Cute flat 2D Korean webtoon illustration style, soft sage green and warm cream palette"
  - character_sheet: "petite woman, warm ivory skin, shoulder-length straight black hair in a messy bun, oversized mint-green hoodie, light grey sweatpants, white ceramic mug in hand"
  - caption_style: korean_jjal
  - meme_format: "jjal"

English:
  - visual_style_anchor: "Modern flat 2D illustration style, bold teal and warm white palette"
  - character_sheet: "tired millennial woman, light peach skin, shoulder-length brown hair in a high messy bun, oversized teal crewneck sweatshirt, black leggings, dark circles under eyes, white ceramic mug in hand"
  - caption_style: impact
  - meme_format: "pov"

Spanish:
  - visual_style_anchor: "Warm vibrant flat 2D illustration style, ochre and terracotta palette"
  - character_sheet: "expressive Latina woman, warm golden-brown skin, voluminous curly dark brown hair past shoulders, floral terracotta blouse, jeans, large gold hoop earrings, coffee cup in hand"
  - caption_style: spanish_social
  - meme_format: "relatability"

SAFETY: every medical fact in tts_text/bottom_text must trace to source scripts. No fear-based framing.
source_topic: short English phrase.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Flexible fallback parsers (handle non-standard Gemini output structures)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_candidate_set(raw_text: str) -> dict[str, Any]:
    """Normalize Gemini candidate output to {english: [...], korean: [...], spanish: [...]}."""
    data = parse_json_response(raw_text)

    if "candidates" in data and isinstance(data["candidates"], list):
        grouped: dict[str, list] = {"english": [], "korean": [], "spanish": []}
        for item in data["candidates"]:
            lang = str(item.get("language", "")).lower()
            if lang in grouped:
                grouped[lang].append(_normalize_candidate(item, lang, index=len(grouped[lang])))
        return grouped

    grouped = {}
    for lang in ("english", "korean", "spanish"):
        items = data.get(lang) or []
        grouped[lang] = [_normalize_candidate(item, lang, index=i) for i, item in enumerate(items)]
    return grouped


def _parse_score_set(raw_text: str) -> dict[str, Any]:
    """Normalize Gemini score output to {english: [...], korean: [...], spanish: [...]}."""
    data = parse_json_response(raw_text)

    if "scores" in data and isinstance(data["scores"], list):
        return _group_scores(data["scores"])

    if "evaluations" in data and isinstance(data["evaluations"], list):
        items = []
        for ev in data["evaluations"]:
            nested = ev.get("scores") or {}
            items.append({
                "candidate_id": ev.get("candidate_id") or "",
                "language": _lang_from_candidate_id(ev.get("candidate_id") or ""),
                "hook_strength": int(nested.get("hook_strength") or ev.get("hook_strength") or 3),
                "native_fit": int(nested.get("native_fit") or ev.get("native_fit") or 3),
                "retention": int(nested.get("retention") or ev.get("retention") or 3),
                "medical_safety": int(nested.get("medical_safety") or ev.get("medical_safety") or 3),
                "not_cringe": int(nested.get("not_cringe") or ev.get("not_cringe") or 3),
                "visual_feasibility": int(nested.get("visual_feasibility") or ev.get("visual_feasibility") or 3),
                "selected": bool(ev.get("winner") or ev.get("selected") or False),
                "rationale": ev.get("rationale") or "",
            })
        return _group_scores(items)

    grouped = {}
    for lang in ("english", "korean", "spanish"):
        items = data.get(lang) or []
        grouped[lang] = [_normalize_score(item, lang) for item in items]
    return grouped


def _parse_meme_plan(
    raw_text: str,
    *,
    topic: str,
    candidates: dict[str, Any] | None = None,
    scores: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize Gemini meme plan output, filling missing fields from selected candidates."""
    data = parse_json_response(raw_text)
    result: dict[str, Any] = {"source_topic": data.get("source_topic") or topic}
    for lang in ("english", "korean", "spanish"):
        lang_data = data.get(lang)
        if not isinstance(lang_data, dict):
            continue
        selected = _get_selected_candidate(candidates or {}, scores or {}, lang)
        result[lang] = _normalize_lang_plan(lang_data, lang, selected)
    return result


def _normalize_candidate(item: dict[str, Any], lang: str, *, index: int = 0) -> dict[str, Any]:
    hook_templates = item.get("hook_templates") or []
    hook_template = item.get("hook_template") or item.get("hook") or (hook_templates[0] if hook_templates else "")
    beats_raw = item.get("scene_beats") or item.get("beats") or []
    scene_beats = [
        str(b.get("visual") or b.get("beat_name") or b) if isinstance(b, dict) else str(b)
        for b in beats_raw
    ]
    return {
        "candidate_id": item.get("candidate_id") or item.get("id") or f"{lang}_{index + 1}",
        "language": lang,
        "creative_angle": item.get("creative_angle") or item.get("format_name") or "",
        "trend_rationale": item.get("trend_rationale") or item.get("why_it_works") or "",
        "hook_template": hook_template,
        "caption_style": item.get("caption_style") or "impact",
        "visual_style_anchor": item.get("visual_style_anchor") or "",
        "character_sheet": item.get("character_sheet") or "",
        "tts_style": item.get("tts_style") or "",
        "bgm_prompt": item.get("bgm_prompt") or "",
        "bgm_config": _normalize_bgm_config(item.get("bgm_config") or {}),
        "avoid_cliches": item.get("avoid_cliches") or item.get("avoid") or [],
        "scene_beats": scene_beats,
    }


def _normalize_score(item: dict[str, Any], lang: str) -> dict[str, Any]:
    return {
        "candidate_id": item.get("candidate_id") or "",
        "language": item.get("language") or lang,
        "hook_strength": int(item.get("hook_strength") or 3),
        "native_fit": int(item.get("native_fit") or 3),
        "retention": int(item.get("retention") or 3),
        "medical_safety": int(item.get("medical_safety") or 3),
        "not_cringe": int(item.get("not_cringe") or 3),
        "visual_feasibility": int(item.get("visual_feasibility") or 3),
        "selected": bool(item.get("selected") or item.get("winner") or False),
        "rationale": item.get("rationale") or "",
    }


def _group_scores(items: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list] = {"english": [], "korean": [], "spanish": []}
    for item in items:
        lang = str(item.get("language") or _lang_from_candidate_id(item.get("candidate_id") or "")).lower()
        if lang in grouped:
            grouped[lang].append(_normalize_score(item, lang))
    return grouped


def _get_selected_candidate(
    candidates: dict[str, Any],
    scores: dict[str, Any],
    lang: str,
) -> dict[str, Any]:
    lang_scores = scores.get(lang) or []
    selected_id = next((s.get("candidate_id") for s in lang_scores if s.get("selected")), None)
    lang_candidates = candidates.get(lang) or []
    if selected_id:
        for c in lang_candidates:
            if c.get("candidate_id") == selected_id:
                return c
    return lang_candidates[0] if lang_candidates else {}


def _normalize_lang_plan(
    lang_data: dict[str, Any],
    lang: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    scenes_raw = lang_data.get("scenes") or []
    scenes = [_normalize_scene(s, i) for i, s in enumerate(scenes_raw)]
    hook_template = candidate.get("hook_template") or ""
    return {
        "language": lang,
        "cultural_context": lang_data.get("cultural_context") or "",
        "trending_hooks": lang_data.get("trending_hooks") or ([hook_template] if hook_template else [""]),
        "creative_angle": lang_data.get("creative_angle") or candidate.get("creative_angle") or "",
        "trend_rationale": lang_data.get("trend_rationale") or candidate.get("trend_rationale") or "",
        "caption_style": lang_data.get("caption_style") or candidate.get("caption_style") or "impact",
        "tts_style": lang_data.get("tts_style") or candidate.get("tts_style") or "",
        "bgm_prompt": lang_data.get("bgm_prompt") or candidate.get("bgm_prompt") or "",
        "bgm_config": _normalize_bgm_config(lang_data.get("bgm_config") or candidate.get("bgm_config") or {}),
        "avoid_cliches": lang_data.get("avoid_cliches") or candidate.get("avoid_cliches") or [],
        "visual_style_anchor": lang_data.get("visual_style_anchor") or candidate.get("visual_style_anchor") or "",
        "character_sheet": lang_data.get("character_sheet") or candidate.get("character_sheet") or "",
        "scenes": scenes,
    }


def _normalize_scene(scene: dict[str, Any], index: int) -> dict[str, Any]:
    scene_num = scene.get("scene_num") or scene.get("scene_number") or (index + 1)
    scene_id = scene.get("scene_id") or f"scene_{int(scene_num):02d}"
    return {
        "scene_id": scene_id,
        "meme_format": scene.get("meme_format") or "pov",
        "caption_role": scene.get("caption_role") or "hook",
        "top_text": scene.get("top_text") or "",
        "bottom_text": scene.get("bottom_text") or scene.get("caption_text") or "",
        "image_prompt": scene.get("image_prompt") or "",
        "duration_seconds": float(scene.get("duration_seconds") or 3.0),
        "tts_text": scene.get("tts_text") or "",
    }


def _normalize_bgm_config(config: dict[str, Any]) -> dict[str, Any]:
    def _to_float(v: Any, default: float) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _to_int(v: Any, default: int) -> int:
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default

    return {
        "bpm": _to_int(config.get("bpm"), 110),
        "density": _to_float(config.get("density"), 0.72),
        "brightness": _to_float(config.get("brightness"), 0.75),
        "guidance": _to_float(config.get("guidance"), 4.0),
        "temperature": _to_float(config.get("temperature"), 1.0),
    }


def _lang_from_candidate_id(candidate_id: str) -> str:
    cid = candidate_id.lower()
    for lang in ("english", "korean", "spanish"):
        if cid.startswith(lang):
            return lang
    return ""
