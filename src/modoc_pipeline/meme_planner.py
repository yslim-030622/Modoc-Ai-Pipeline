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
from .prompt_profiles import load_campaign_profile, profile_json
from .schemas import ContentVisualBrief, CreativeCandidateSet, CreativeScoreSet, MemePlan, TrendResearch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemePlanGeneration:
    parsed: dict[str, Any]
    raw_text: str
    search_metadata: dict[str, Any]
    trend_research: dict[str, Any]
    visual_brief: dict[str, Any]
    creative_candidates: dict[str, Any]
    creative_scores: dict[str, Any]


def generate_meme_plan(
    *,
    api_key: str,
    model: str,
    scripts: dict[str, Any],
    topic: str,
    campaign_profile: dict[str, Any] | None = None,
    campaign_profile_path: str | None = None,
) -> MemePlanGeneration:
    client = genai.Client(api_key=api_key)
    profile = campaign_profile or load_campaign_profile(campaign_profile_path)

    trends_json, trend_research, search_metadata = _stage1_search_trends(
        client,
        model=model,
        topic=topic,
        campaign_profile=profile,
    )
    visual_brief_raw, visual_brief = _stage2_generate_visual_brief(
        client,
        model=model,
        scripts=scripts,
        topic=topic,
        campaign_profile=profile,
    )
    candidates_raw, candidates = _stage2_generate_candidates(
        client,
        model=model,
        scripts=scripts,
        topic=topic,
        trends=trend_research,
        visual_brief=visual_brief,
        campaign_profile=profile,
    )
    scores_raw, scores = _stage3_judge_candidates(
        client,
        model=model,
        scripts=scripts,
        topic=topic,
        candidates=candidates,
        visual_brief=visual_brief,
        campaign_profile=profile,
    )
    raw_text, parsed = _stage4_generate_plan(
        client,
        model=model,
        scripts=scripts,
        topic=topic,
        trends_json=trends_json,
        candidates=candidates,
        scores=scores,
        visual_brief=visual_brief,
        campaign_profile=profile,
    )
    combined_raw = json.dumps(
        {
            "trend_research_raw": trends_json,
            "visual_brief_raw": visual_brief_raw,
            "creative_candidates_raw": candidates_raw,
            "creative_scores_raw": scores_raw,
            "final_meme_plan_raw": raw_text,
            "campaign_profile": profile,
        },
        ensure_ascii=False,
        indent=2,
    )
    return MemePlanGeneration(
        parsed=parsed,
        raw_text=combined_raw,
        search_metadata=search_metadata,
        trend_research=trend_research,
        visual_brief=visual_brief,
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
    campaign_profile: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    prompt = _build_search_prompt(topic, campaign_profile=campaign_profile)
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


def _build_search_prompt(topic: str, *, campaign_profile: dict[str, Any] | None = None) -> str:
    profile = campaign_profile or load_campaign_profile()
    return f"""
Research current short-form formats for pediatric health / parenting
edutainment. Use current web results and prioritize {profile.get("trend_window")}.
Do not assume last month's or last year's named trend frameworks are still
current unless search results support them.

Campaign profile:
{profile_json(profile)}

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

Use the campaign profile's language platform lists for search focus. The example
below is structural only; do not copy its creative content unless current search
results support it.

Example structure:
{{
  "english": [{{
    "format_name": "current searched format name",
    "platform": "platform where it currently appears",
    "hook_templates": ["native hook pattern from current search"],
    "caption_style": "current caption style",
    "visual_editing_style": "current editing/composition style",
    "audio_mood": "current audio mood",
    "why_it_works": "why this is current and fits parent education",
    "avoid": ["stale or unsafe nearby template"]
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
# Stage 2: Row-specific visual brief
# ─────────────────────────────────────────────────────────────────────────────

CONTENT_VISUAL_GRAMMAR: dict[str, dict[str, list[str]]] = {
    "fever_triage": {
        "allowed_props": ["blank digital thermometer", "water cup", "phone", "light blanket", "sofa", "blank note"],
        "forbidden_visuals": ["thermometer numbers", "red alarm graphics", "hospital bed unless source says hospital", "pills unless medication is in script"],
    },
    "medication_dosing": {
        "allowed_props": ["unlabeled medicine bottle", "measuring spoon", "oral syringe without numbers", "phone timer", "blank dosing note"],
        "forbidden_visuals": ["readable labels", "dosage numbers inside image", "loose pills", "brand logos", "needle"],
    },
    "vaccine_reaction": {
        "allowed_props": ["baby blanket", "tiny bandage sticker", "blank thermometer", "water cup", "phone", "caregiver lap"],
        "forbidden_visuals": ["needle injection scene", "clinic panic", "vaccine vial text", "sick child close-up"],
    },
    "gi_hydration": {
        "allowed_props": ["water cup", "small bowl", "towel", "sofa", "phone", "blank sticky note"],
        "forbidden_visuals": ["vomit detail", "toilet scene", "graphic fluids", "medicine unless source says medicine"],
    },
    "puberty_period": {
        "allowed_props": ["closed planner", "plain pouch", "folded towel", "phone", "private bedroom desk", "clinic folder"],
        "forbidden_visuals": ["blood drops", "uterus graphics", "visible calendar numbers", "embarrassed facial exaggeration"],
    },
    "rash_skin": {
        "allowed_props": ["phone photo view without visible rash detail", "soft towel", "clinic folder", "lamp", "blank note"],
        "forbidden_visuals": ["graphic lesions", "diagnosis labels", "magnified skin cells", "fearful red warning symbols"],
    },
    "lab_result": {
        "allowed_props": ["blank folder", "blank paper", "phone call", "doctor desk", "pen", "two blank cards"],
        "forbidden_visuals": ["readable lab report", "numbers", "charts", "balance scale", "blood drop icon"],
    },
    "development_behavior": {
        "allowed_props": ["blocks", "picture book", "play mat", "desk lamp", "caregiver notebook", "phone"],
        "forbidden_visuals": ["diagnosis stamp", "brain icon", "test score", "worried child close-up"],
    },
    "injury_urgent": {
        "allowed_props": ["ice pack wrapped in towel", "phone", "sofa", "blank note", "clinic bag"],
        "forbidden_visuals": ["blood", "open wound", "ambulance unless source says emergency", "panic symbols"],
    },
    "newborn_feeding_sleep": {
        "allowed_props": ["baby blanket", "feeding bottle without labels", "burp cloth", "dim lamp", "crib silhouette", "phone"],
        "forbidden_visuals": ["unsafe sleep position", "exhausted dark circles unless source says sleep deprivation", "clock numbers"],
    },
    "general": {
        "allowed_props": ["phone", "blank note", "water cup", "sofa", "desk", "clinic folder"],
        "forbidden_visuals": ["medical infographic", "readable text", "numbers", "logos", "fear symbols"],
    },
}


def _stage2_generate_visual_brief(
    client: genai.Client,
    *,
    model: str,
    scripts: dict[str, Any],
    topic: str,
    campaign_profile: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    prompt = _build_visual_brief_prompt(
        scripts=scripts,
        topic=topic,
        campaign_profile=campaign_profile,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_schema(ContentVisualBrief),
            temperature=0.25,
        ),
    )
    raw_text = response.text or ""
    try:
        parsed = ContentVisualBrief.model_validate_json(raw_text).model_dump()
    except Exception:
        parsed = _parse_visual_brief(raw_text)
    return raw_text, _merge_visual_grammar(parsed)


def _build_visual_brief_prompt(
    *,
    scripts: dict[str, Any],
    topic: str,
    campaign_profile: dict[str, Any] | None = None,
) -> str:
    profile = campaign_profile or load_campaign_profile()
    return f"""
Classify this pediatric Q&A row into a row-specific visual brief for an
automated short-form slideshow. This is NOT the final storyboard. It is the
visual decision layer that prevents every row from looking like the same
generic caregiver-at-home scene.

Topic: {topic}
Medical scripts/facts: {json.dumps(scripts, ensure_ascii=False, indent=2)}
Campaign profile: {profile_json(profile)}

Allowed content_type values:
- fever_triage
- medication_dosing
- vaccine_reaction
- gi_hydration
- puberty_period
- rash_skin
- lab_result
- development_behavior
- injury_urgent
- newborn_feeding_sleep
- general

Content-type visual grammar:
{json.dumps(CONTENT_VISUAL_GRAMMAR, ensure_ascii=False, indent=2)}

Rules:
- Pick exactly one content_type based on the source row, not on generic parenting vibes.
- allowed_props must come mostly from that content_type's allowed_props.
- forbidden_visuals must include the content_type's forbidden visuals plus any row-specific hallucination risks.
- must_show should name concrete visible anchors that make this exact row recognizable.
- must_not_show should forbid unsupported or risky visuals from the source.
- Use props only when medically/contextually relevant. Do not default to plants,
  coffee mugs, sunny windows, or generic living rooms unless the source row calls for them.
- Do not put readable text, labels, dosage numbers, calendar numbers, or charts
  inside generated images. Captions are rendered later by the video renderer.
- Medical precision belongs in captions/TTS; the image should show safe observable action.

Return ContentVisualBrief JSON only.
""".strip()


def _parse_visual_brief(raw_text: str) -> dict[str, Any]:
    data = parse_json_response(raw_text)
    return ContentVisualBrief.model_validate(data).model_dump()


def _merge_visual_grammar(brief: dict[str, Any]) -> dict[str, Any]:
    content_type = str(brief.get("content_type") or "general")
    grammar = CONTENT_VISUAL_GRAMMAR.get(content_type) or CONTENT_VISUAL_GRAMMAR["general"]
    allowed = _dedupe_strings([*(brief.get("allowed_props") or []), *grammar["allowed_props"]])[:8]
    forbidden = _dedupe_strings([*(brief.get("forbidden_visuals") or []), *grammar["forbidden_visuals"]])[:12]
    normalized = ContentVisualBrief.model_validate({
        **brief,
        "content_type": content_type if content_type in CONTENT_VISUAL_GRAMMAR else "general",
        "allowed_props": allowed,
        "forbidden_visuals": forbidden,
    }).model_dump()
    return normalized


def _dedupe_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Creative candidate generation
# ─────────────────────────────────────────────────────────────────────────────

def _stage2_generate_candidates(
    client: genai.Client,
    *,
    model: str,
    scripts: dict[str, Any],
    topic: str,
    trends: dict[str, Any],
    visual_brief: dict[str, Any],
    campaign_profile: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    prompt = _build_candidate_prompt(
        scripts=scripts,
        topic=topic,
        trends=trends,
        visual_brief=visual_brief,
        campaign_profile=campaign_profile,
    )
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


def _build_candidate_prompt(
    *,
    scripts: dict[str, Any],
    topic: str,
    trends: dict[str, Any],
    visual_brief: dict[str, Any] | None = None,
    campaign_profile: dict[str, Any] | None = None,
) -> str:
    scripts_str = json.dumps(scripts, ensure_ascii=False, indent=2)
    trends_str = json.dumps(trends, ensure_ascii=False, indent=2)
    visual_brief_str = json.dumps(visual_brief or {}, ensure_ascii=False, indent=2)
    profile = campaign_profile or load_campaign_profile()
    return f"""
Create exactly 3 creative candidates per language for a Gemini-only pediatric
health short-form slideshow. Use the trend research as the primary creative
source; use the campaign profile for safety, visual, and brand constraints.

Topic: {topic}
Medical facts (use ONLY these; never add claims): {scripts_str}
Trend research: {trends_str}
Row-specific visual brief: {visual_brief_str}
Campaign profile: {profile_json(profile)}

Rules:
- Output valid JSON only.
- Each candidate must be medically safe, parent-facing, current, and culturally native.
- Every candidate's trend_rationale must explain which trend research item it adapts.
- Do not use banned_templates from the campaign profile.
- If trend research is empty or weak, choose a neutral evergreen short-form pattern
  and say that clearly in trend_rationale.
- Medical safety beats humor. Never use fear bait, diagnosis bait, or definitive reassurance.
- Use language notes from the campaign profile, not stereotype templates.
- The visual concept and scene_beats must fit the row-specific visual brief's
  content_type, allowed_props, must_show, and must_not_show.
- Do not default to generic plants, coffee mugs, sunny windows, or static
  caregiver poses unless the visual brief makes them relevant.
- visual_style_anchor and character_sheet must be concrete and fully described for image consistency.
- Prefer one reusable video-level caregiver identity across all languages. Keep
  character_sheet neutral and well-rested unless the source medical script
  specifically requires fatigue or a night scenario.
- bgm_prompt should specify a trending, upbeat short-form style: ukulele, claps, light percussion,
  bouncy indie-pop or Latin-pop, immediate catchy hook in the first 2 seconds, cheerful viral energy.
  No vocals, no humming, no lyrics, room for voiceover. Avoid generic "parent-friendly" clichés.
- BGM must feel like a trending TikTok/Reels track — bright, bouncy, and instantly engaging.
  For serious medical topics, stay upbeat and cheerful; never go somber or dark.
- bgm_config must include bpm, density, brightness, guidance, temperature.
- Prefer bpm 116-122, density 0.62-0.68, brightness 0.90-0.96, guidance 3.3-3.8.
- scene_beats must be exactly 4 short beats: hook, tension, insight, relief.
- Captions must not mention unsupported home remedies, folk treatments, emergency
  rooms, or urgent care unless those concepts appear in the medical script.

Return CreativeCandidateSet JSON.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Candidate judging / scoring
# ─────────────────────────────────────────────────────────────────────────────

def _stage3_judge_candidates(
    client: genai.Client,
    *,
    model: str,
    scripts: dict[str, Any],
    topic: str,
    candidates: dict[str, Any],
    visual_brief: dict[str, Any],
    campaign_profile: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    prompt = _build_score_prompt(
        scripts=scripts,
        topic=topic,
        candidates=candidates,
        visual_brief=visual_brief,
        campaign_profile=campaign_profile,
    )
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


def _build_score_prompt(
    *,
    scripts: dict[str, Any],
    topic: str,
    candidates: dict[str, Any],
    visual_brief: dict[str, Any] | None = None,
    campaign_profile: dict[str, Any] | None = None,
) -> str:
    profile = campaign_profile or load_campaign_profile()
    return f"""
Judge these creative candidates for a current pediatric health short-form video.

Topic: {topic}
Medical scripts/facts: {json.dumps(scripts, ensure_ascii=False, indent=2)}
Candidates: {json.dumps(candidates, ensure_ascii=False, indent=2)}
Row-specific visual brief: {json.dumps(visual_brief or {}, ensure_ascii=False, indent=2)}
Campaign profile: {profile_json(profile)}

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
- Penalize candidates that ignore trend research or copy banned_templates.
- Penalize candidates whose scene_beats ignore the visual brief content_type,
  allowed_props, must_show, or must_not_show.
- Penalize visual concepts that make one language a different character identity
  when the campaign profile asks for one video-level caregiver.
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
# Stage 5: Final MemePlan generation
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
    visual_brief: dict[str, Any],
    campaign_profile: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    prompt = _build_plan_prompt(
        scripts=scripts,
        topic=topic,
        trends_json=trends_json,
        candidates=candidates,
        scores=scores,
        visual_brief=visual_brief,
        campaign_profile=campaign_profile,
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
    parsed["visual_brief"] = _merge_visual_grammar(parsed.get("visual_brief") or visual_brief)
    return raw_text, parsed


def _build_plan_prompt(
    *,
    scripts: dict[str, Any],
    topic: str,
    trends_json: str,
    candidates: dict[str, Any],
    scores: dict[str, Any],
    visual_brief: dict[str, Any] | None = None,
    campaign_profile: dict[str, Any] | None = None,
) -> str:
    scripts_str = json.dumps(scripts, ensure_ascii=False, indent=2)
    candidates_str = json.dumps(candidates, ensure_ascii=False, indent=2)
    scores_str = json.dumps(scores, ensure_ascii=False, indent=2)
    visual_brief_str = json.dumps(visual_brief or {}, ensure_ascii=False, indent=2)
    profile = campaign_profile or load_campaign_profile()
    return f"""
Create the final 4-scene current short-form slideshow plan for pediatric health
education. Use the selected creative candidate per language; do not invent a
new direction. Current trend research and the campaign profile are binding
creative inputs.

Topic: {topic}
Medical facts (use ONLY these, never invent): {scripts_str}
Trend research found: {trends_json}
Creative candidates: {candidates_str}
Creative scores and selections: {scores_str}
Row-specific visual brief: {visual_brief_str}
Campaign profile: {profile_json(profile)}

STORY ARC (4 scenes):
  Scene 1 HOOK: real parent situation / curiosity gap; caption_role="hook"
  Scene 2 TENSION: relatable wrong assumption; caption_role="tension"
  Scene 3 INSIGHT: supported medical truth; caption_role="insight"
  Scene 4 RELIEF: simple action + warm close; caption_role="relief"

VISUAL CONSISTENCY (critical):
Define a visual_style_anchor AND character_sheet that can work across the full
video. By default, reuse the same visual_style_anchor and character_sheet in
English, Korean, and Spanish so the same caregiver carries the topic across
localized videos. Only diverge if the campaign profile explicitly asks for
language-specific characters.

character_sheet = stable character description used VERBATIM in every scene prompt.
If the video features both a caregiver and a child, character_sheet MUST describe BOTH separately and lock ALL of the following:

For the caregiver:
  - exact hair: color, length, style (e.g. "shoulder-length wavy black hair in a loose bun")
  - exact clothing: color, type, pattern (e.g. "oversized grey hoodie, dark blue joggers")
  - skin tone (e.g. "warm light beige skin")
  - body shape / build (e.g. "petite, slightly rounded face")

For the child (if appearing in any scene):
  - gender: explicitly state "girl" or "boy" — NEVER leave ambiguous
  - approximate age range (e.g. "toddler around 2 years old", "school-age girl around 6")
  - exact hair: color, length, style (e.g. "short curly black hair")
  - exact clothing: color, type (e.g. "pastel yellow long-sleeve top, white pants")
  - skin tone matching caregiver family

These descriptions must be IDENTICAL across all 4 scenes and all languages.
It must NOT include props, backgrounds, actions, emotions, or phrases like "always holding..." because those belong to individual scenes.
Example with child: "petite adult caregiver, warm light beige skin, shoulder-length wavy black hair in a loose bun, oversized sage cardigan, white shirt, olive trousers; toddler girl around 2 years old, warm light beige skin, short wavy black hair, pastel yellow top, white pants"

VISUAL SCENE WORKFLOW:
Before writing any language scene, use the Row-specific visual brief as the
source of truth for visual decisions:
- content_type decides the visual grammar.
- allowed_props are the main prop pool.
- forbidden_visuals and must_not_show override generic creativity.
- must_show identifies row-specific anchors that should appear across the 4 scenes.
- Do not replace the brief with generic home, coffee, plant, or sunny-window imagery.

For every scene, fill these fields before writing image_prompt:
- medical_message: the single medically supported idea this scene communicates.
- scene_visual_action: a concrete, safe action or situation that supports the
  message without trying to explain the medicine inside the picture.
- safe_props: 0-3 props chosen from the visual brief's allowed_props unless the
  source script requires another safe prop.
- shot_type: one of close_up, medium_action, wide_scene, over_the_shoulder,
  tabletop_action, conversation.
- primary_prop: the main row-relevant prop, or "" if none.
- background: the scene-specific setting.

The image_prompt must be derived from scene_visual_action + safe_props + shot_type.
Images should support mood, pacing, and scene variety. Captions and TTS carry
the precise medical explanation.

CONTENT-SPECIFIC VISUAL ANCHORS:
- Fever rows may use a blank digital thermometer, water cup, blanket, phone, or
  caregiver checking temperature. Do not show readable temperature numbers.
- Medication rows may use an unlabeled medicine bottle, measuring spoon, oral
  syringe without numbers, phone timer, or blank dosing note. Do not show labels,
  pills, dosage numbers, brands, or needles.
- Vaccine reaction rows may use a baby blanket, tiny bandage sticker, blank
  thermometer, water cup, phone, or caregiver lap. Do not show injection needles
  or vaccine vial text.
- GI hydration rows may use water cup, small bowl, towel, sofa, or caregiver
  offering fluids. Do not show vomit, toilet scenes, or graphic fluids.
- Puberty/period rows may use a closed planner, plain pouch, folded towel, phone,
  private desk, or clinic folder. Do not show blood drops, uterus graphics, or
  visible calendar numbers.
- Rash rows may use a phone photo posture without detailed rash, towel, clinic
  folder, or lamp. Do not show graphic lesions or diagnosis labels.
- Lab/result rows may use blank paper, folder, phone call, doctor desk, pen, or
  two blank cards. Do not show readable lab reports, charts, numbers, balance
  scales, or blood drop icons.
- Development/behavior rows may use blocks, picture books, play mat, notebook,
  or caregiver observing. Do not show diagnosis stamps, brain icons, or scores.
- Injury/urgent rows may use an ice pack wrapped in towel, phone, sofa, clinic
  bag, or calm caregiver action. Do not show blood, open wounds, ambulances, or
  panic symbols unless the source explicitly says so.
- Newborn feeding/sleep rows may use baby blanket, feeding bottle without labels,
  burp cloth, dim lamp, crib silhouette, or phone. Do not show unsafe sleep.

IMAGE SIMPLICITY:
- Do not make medical infographics, charts, scales, traffic lights, lab cards,
  diagnosis symbols, or symbolic medical metaphors unless the visual brief
  explicitly allows a concrete non-text prop.
- If you need to show time passing, use blank reminder cards, a closed planner,
  or a phone timer with no visible numbers.
- If you need to show concern or comparison, use caregiver body language, two
  blank cards, or a simple tabletop arrangement with no symbols or labels.

VISUAL VARIETY:
- The 4 scenes must not all be the same shot type.
- Do not repeat the same primary_prop in more than 2 scenes.
- Do not repeat the same background in more than 2 scenes.
- Vary camera distance, action, and composition across the story arc.
- Preserve only character identity across scenes; pose, prop, location, camera
  angle, and composition must change to match each scene_visual_action.

IMAGE PROMPT RULES:
- Every scene prompt MUST open with: "[visual_style_anchor]. [character_sheet full text]. [new action/emotion this scene]."
- CHARACTER CONSISTENCY IS MANDATORY: every scene must describe the child with the EXACT same gender, age, hair, clothing, and skin tone from character_sheet. Never switch girl↔boy or change hair/outfit between scenes.
- ZERO text, letters, signs in the image — pure illustration
- Describe a concrete scene: shot type, composition, ordinary prop, action,
  background, and calm expression.
- Good: "[anchor]. [character_sheet]. Over-the-shoulder shot at a desk, pointing to a closed planner and two blank sticky notes, soft daylight, calm focused posture."
- Bad: "Same character as scene 1" (useless — model has no memory), "background with bold colors" (too vague)
- Bad: four scenes of the caregiver standing still with the same mug or phone.
- Bad: balance scale, blood drop, plant/leaf symbol, traffic light, numbered calendar,
  lab report, medical chart, icon labels, or any text inside the image.
- Max 90 words. 9:16 vertical.
- Do not add dark circles, exhausted features, messy-night imagery, or 3am bedroom
  scenes unless the source script specifically needs a night scenario.

TEXT (displayed over image — keep SHORT and PUNCHY):
- top_text: ≤ 6 words label or ""; Korean about ≤12 characters.
- bottom_text: ≤ 8 words; Korean about ≤18 characters.
- tts_text: ONE spoken sentence, ≤ 10 words, warm and quick, medically accurate.
- first scene caption must be readable in about 1 second.

LANGUAGE FIELDS:
- Fill creative_angle, trend_rationale, caption_style, tts_style, bgm_prompt, bgm_config, avoid_cliches from the selected candidate.
- caption_style must be one of: impact, clean_reels, korean_jjal, spanish_social.
- bgm_prompt must sound like a trending TikTok/Reels track: ukulele, claps, bouncy indie-pop or Latin-pop,
  immediate hook in the first 2 seconds. No vocals, no humming, no artist names, room for voiceover.
- bgm_config should fit the selected BGM prompt: prefer bpm 116-122, brightness 0.90-0.96, density 0.62-0.68.
- For ALL topics including serious medical ones, BGM must stay bright, bouncy, and cheerful — never somber.
- Captions must not add unsupported urgency or remedies. Do not mention emergency
  rooms, urgent care, alcohol/romero, massage, 호박즙, or other folk remedies unless
  the source script explicitly supports them.
- Follow each language's campaign profile notes and selected trend pattern.
- Do not use banned_templates from the campaign profile.
- Keep character identity consistent across languages; localize text and cultural
  delivery without turning the person into a different stereotype.

SAFETY: every medical fact in tts_text/bottom_text must trace to source scripts. No fear-based framing.
Set visual_brief in the final JSON to the exact row-specific visual brief above,
with only minimal cleanup if needed.
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
    result: dict[str, Any] = {
        "source_topic": data.get("source_topic") or topic,
        "visual_brief": _merge_visual_grammar(data.get("visual_brief") or {}),
    }
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
        "medical_message": scene.get("medical_message") or "",
        "scene_visual_action": scene.get("scene_visual_action") or scene.get("visual_job") or "",
        "safe_props": scene.get("safe_props") or [],
        "shot_type": scene.get("shot_type") or "medium_action",
        "primary_prop": scene.get("primary_prop") or "",
        "background": scene.get("background") or "",
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
