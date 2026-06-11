"""Deterministic quality validators for meme plan content."""

from __future__ import annotations

import json
from typing import Any

from .orchestration.state import ReviewIssue, ReviewReport


def deterministic_creative_report(
    meme_plan: dict[str, Any],
    creative_scores: dict[str, Any],
    *,
    scripts: dict[str, Any] | None = None,
) -> ReviewReport:
    issues: list[dict[str, Any]] = []
    serious_topic = _is_serious_medical_topic(scripts or {}, meme_plan)
    for lang in ("english", "korean", "spanish"):
        lang_plan = meme_plan.get(lang, {}) if isinstance(meme_plan, dict) else {}
        subject = lang
        for field in ("creative_angle", "trend_rationale", "caption_style", "tts_style", "bgm_prompt"):
            if not str(lang_plan.get(field, "")).strip():
                issues.append(_issue(subject, f"{field} is required for 2026 creative planning."))
        if lang_plan.get("caption_style") not in {"impact", "clean_reels", "korean_jjal", "spanish_social"}:
            issues.append(_issue(subject, "caption_style must be a supported renderer style."))
        scenes = lang_plan.get("scenes", []) or []
        roles = [scene.get("caption_role") for scene in scenes if isinstance(scene, dict)]
        for expected in ("hook", "tension", "insight", "relief"):
            if expected not in roles:
                issues.append(_issue(subject, f"missing caption_role={expected}."))
        if not _has_selected_safe_score(creative_scores, lang):
            issues.append(_issue(subject, "creative_scores must select one candidate with medical_safety >= 4."))
        bgm_prompt = str(lang_plan.get("bgm_prompt", "")).lower()
        if not _has_any(bgm_prompt, ("instrumental", "no vocals", "no lyrics")):
            issues.append(_issue(subject, "BGM prompt should explicitly keep the track instrumental and voiceover-safe."))
        bgm_config = lang_plan.get("bgm_config", {}) if isinstance(lang_plan.get("bgm_config"), dict) else {}
        if serious_topic:
            if float(bgm_config.get("density", 0.0) or 0.0) > 0.86:
                issues.append(_issue(subject, "BGM density is too high for voiceover clarity. Use density <= 0.86."))
            if float(bgm_config.get("brightness", 0.0) or 0.0) > 0.92:
                issues.append(_issue(subject, "BGM brightness is too high for voiceover clarity. Use brightness <= 0.92."))

    if issues:
        return ReviewReport(
            status="failed",
            stage="creative_review",
            summary="Creative review failed.",
            issues=issues,
            repair_instruction=(
                "Rebuild the MemePlan from a selected safe creative candidate. "
                "Fill creative metadata, use all four caption roles, and avoid generic medical explainer framing."
            ),
        )
    return ReviewReport(status="passed", stage="creative_review", summary="Creative review passed.")


def deterministic_meme_text_report(
    scripts: dict[str, Any],
    meme_plan: dict[str, Any],
) -> ReviewReport:
    issues: list[dict[str, Any]] = []
    source_text = json.dumps(scripts, ensure_ascii=False).lower()
    urgency_allowed = _has_any(source_text, ("emergency", "er", "응급", "urgencias", "urgent", "urgente"))
    for lang in ("english", "korean", "spanish"):
        lang_plan = meme_plan.get(lang, {}) if isinstance(meme_plan, dict) else {}
        scenes = lang_plan.get("scenes", []) or []
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            subject = f"{lang}/{scene.get('scene_id', '')}"
            top = str(scene.get("top_text", "")).strip()
            bottom = str(scene.get("bottom_text", "")).strip()
            tts = str(scene.get("tts_text", "")).strip()
            combined = f"{top} {bottom} {tts}".lower()

            if not urgency_allowed and _has_any(combined, ("emergency room", " er ", "응급실", "urgencias", "sala de emergencia")):
                issues.append(_issue(subject, "caption/tts adds emergency-room urgency not supported by the script."))
            if _has_any(combined, ("호박즙", "마사지", "alcohol de romero", "romero", "home remedy", "remedio casero")):
                issues.append(_issue(subject, "caption/tts mentions a remedy or folk treatment that is not supported by the script."))
            if "mamá latina" in combined:
                issues.append(_issue(subject, "caption uses the stale/stereotyped 'Mamá latina' template. Use a situation-specific Spanish hook instead."))
            if lang == "korean":
                if len(top) > 14:
                    issues.append(_issue(subject, "Korean top_text is too long for first-frame reading; keep it around 12 characters."))
                if len(bottom) > 24:
                    issues.append(_issue(subject, "Korean bottom_text is too long for mobile readability; keep it around 18 characters."))
            else:
                if _word_count(top) > 6:
                    issues.append(_issue(subject, "top_text is too long; keep it to 6 words or fewer."))
                if _word_count(bottom) > 8:
                    issues.append(_issue(subject, "bottom_text is too long; keep it to 8 words or fewer."))

    if issues:
        return ReviewReport(
            status="failed",
            stage="meme_plan_review",
            summary="Deterministic meme text validation failed.",
            issues=issues,
            repair_instruction=(
                "Rewrite captions and tts_text to remove unsupported urgency, remedies, stereotypes, "
                "and overlong text while preserving the medical meaning."
            ),
        )
    return ReviewReport(status="passed", stage="meme_plan_review", summary="Deterministic meme text validation passed.")


def deterministic_visual_prompt_report(meme_plan: dict[str, Any]) -> ReviewReport:
    issues: list[dict[str, Any]] = []
    visual_brief = meme_plan.get("visual_brief", {}) if isinstance(meme_plan, dict) else {}
    allowed_props = [str(p).strip().lower() for p in visual_brief.get("allowed_props", []) or [] if str(p).strip()]
    forbidden_visuals = [str(v).strip().lower() for v in visual_brief.get("forbidden_visuals", []) or [] if str(v).strip()]
    content_type = str(visual_brief.get("content_type", "")).strip()
    if not content_type:
        issues.append(_issue("visual_brief", "content_type is required for row-specific visual planning."))
    if content_type and not allowed_props:
        issues.append(_issue("visual_brief", "allowed_props is required for content-specific visual grammar."))
    for lang in ("english", "korean", "spanish"):
        lang_plan = meme_plan.get(lang)
        if not isinstance(lang_plan, dict):
            continue
        anchor = str(lang_plan.get("visual_style_anchor", "")).strip()
        character = str(lang_plan.get("character_sheet", "")).strip()
        character_probe = character[:30]
        scenes = lang_plan.get("scenes", []) or []
        shot_types: list[str] = []
        primary_props: list[str] = []
        backgrounds: list[str] = []
        for scene in scenes:
            scene_id = str(scene.get("scene_id", ""))
            prompt = str(scene.get("image_prompt", ""))
            subject = f"{lang}/{scene_id}"
            shot_type = str(scene.get("shot_type", "")).strip()
            primary_prop = str(scene.get("primary_prop", "")).strip().lower()
            background = str(scene.get("background", "")).strip().lower()
            if shot_type:
                shot_types.append(shot_type)
            if primary_prop:
                primary_props.append(primary_prop)
                if allowed_props and not _prop_matches_allowed(primary_prop, allowed_props):
                    issues.append(_issue(subject, f"primary_prop should come from visual_brief.allowed_props: {primary_prop}."))
                if primary_prop in {"plant", "potted plant", "coffee mug", "mug", "tea cup"} and not _prop_matches_allowed(primary_prop, allowed_props):
                    issues.append(_issue(subject, f"primary_prop looks like generic filler instead of a row-specific visual anchor: {primary_prop}."))
            if background:
                backgrounds.append(background)
            for field in ("medical_message", "scene_visual_action", "shot_type"):
                if not str(scene.get(field, "")).strip():
                    issues.append(_issue(subject, f"{field} is required for visual scene planning."))
            safe_props = [str(p).strip().lower() for p in scene.get("safe_props", []) or [] if str(p).strip()]
            for prop in safe_props:
                if allowed_props and not _prop_matches_allowed(prop, allowed_props):
                    issues.append(_issue(subject, f"safe_props should come from visual_brief.allowed_props: {prop}."))
            if anchor and not prompt.startswith(anchor[:40]):
                issues.append(_issue(subject, "image_prompt must start with the visual_style_anchor."))
            if character_probe and character_probe not in prompt:
                issues.append(_issue(subject, "image_prompt must include the stable character_sheet."))
            lowered = prompt.lower()
            if primary_prop and primary_prop not in lowered:
                issues.append(_issue(subject, f"image_prompt must visibly include primary_prop: {primary_prop}."))
            semantic_issue = _semantic_visual_mismatch(scene, lowered)
            if semantic_issue:
                issues.append(_issue(subject, semantic_issue))
            forbidden_pairs = {
                "text": "image_prompt requests or permits visible text.",
                "letters": "image_prompt requests or permits visible letters.",
                "sign": "image_prompt requests or permits visible signs.",
                "logo": "image_prompt requests or permits visible logos.",
                "realistic face": "image_prompt includes realistic faces.",
            }
            for token, message in forbidden_pairs.items():
                if token in lowered and "no " + token not in lowered and "zero " + token not in lowered:
                    issues.append(_issue(subject, message))
            repetitive_phrases = (
                "always holding",
                "holding the warm mug",
                "holding a warm mug",
                "same character as scene",
            )
            for phrase in repetitive_phrases:
                if phrase in lowered:
                    issues.append(_issue(subject, f"image_prompt over-preserves a repeated prop or prior scene: {phrase}."))
            hallucination_prone = (
                "balance scale",
                "blood drop",
                "traffic light",
                "lab report",
                "lab card",
                "medical chart",
                "calendar with",
                "numbered calendar",
                "visible numbers",
                "leaf symbol",
                "organs",
                "pills",
            )
            for phrase in hallucination_prone:
                if phrase in lowered:
                    issues.append(_issue(subject, f"image_prompt uses hallucination-prone medical infographic imagery: {phrase}."))
            for phrase in forbidden_visuals:
                if phrase and phrase in lowered:
                    issues.append(_issue(subject, f"image_prompt conflicts with visual_brief.forbidden_visuals: {phrase}."))

        if len(scenes) >= 4:
            subject = f"{lang}/visual_variety"
            if len(set(shot_types)) < 3:
                issues.append(_issue(subject, "Use at least 3 distinct shot_type values across the 4 scenes."))
            if _has_repeated_nonempty(primary_props, max_allowed=2):
                issues.append(_issue(subject, "Do not repeat the same primary_prop in more than 2 scenes."))
            if _has_repeated_nonempty(backgrounds, max_allowed=2):
                issues.append(_issue(subject, "Do not repeat the same background in more than 2 scenes."))

    if issues:
        return ReviewReport(
            status="failed",
            stage="visual_prompt_review",
            summary="Deterministic visual prompt validation failed.",
            issues=issues,
            repair_instruction="Rewrite failed visual fields so visual_brief has a content_type and allowed_props, each scene uses row-specific allowed props, has medical_message/scene_visual_action, varies shot_type/prop/background, includes anchor and character sheet, and avoids visible text, logos, realistic faces, generic filler props, forbidden visuals, and medical infographic symbols.",
        )
    return ReviewReport(
        status="passed",
        stage="visual_prompt_review",
        summary="Deterministic visual prompt validation passed.",
    )


def validate_meme_plan_quality(
    *,
    meme_plan: dict[str, Any],
    creative_scores: dict[str, Any],
    scripts: dict[str, Any],
) -> dict[str, Any]:
    payload = build_meme_plan_quality_reports(
        meme_plan=meme_plan,
        creative_scores=creative_scores,
        scripts=scripts,
    )
    assert_meme_plan_quality(payload)
    return payload


def build_meme_plan_quality_reports(
    *,
    meme_plan: dict[str, Any],
    creative_scores: dict[str, Any],
    scripts: dict[str, Any],
) -> dict[str, Any]:
    reports = [
        deterministic_creative_report(meme_plan, creative_scores, scripts=scripts),
        deterministic_meme_text_report(scripts, meme_plan),
        deterministic_visual_prompt_report(meme_plan),
    ]
    return {report.stage: report.model_dump() for report in reports}


def assert_meme_plan_quality(reports: dict[str, Any]) -> None:
    failed = [
        report
        for report in reports.values()
        if isinstance(report, dict) and report.get("status") == "failed"
    ]
    if failed:
        summary = "; ".join(
            f"{report.get('stage')}: {report.get('summary') or report.get('repair_instruction')}"
            for report in failed
        )
        raise ValueError(f"Meme plan quality validation failed: {summary}")


def expected_scene_audio_ids(
    meme_plan: dict[str, Any],
    languages: set[str] | None = None,
) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = {}
    for lang in _requested_languages(meme_plan, languages):
        scenes = meme_plan.get(lang, {}).get("scenes", []) or []
        scene_ids = {
            str(scene.get("scene_id", "")).strip()
            for scene in scenes
            if isinstance(scene, dict)
            and str(scene.get("scene_id", "")).strip()
            and str(scene.get("tts_text", "")).strip()
        }
        if scene_ids:
            expected[lang] = scene_ids
    return expected


def assert_tts_complete(
    *,
    audio_paths: dict[str, dict[str, Any]],
    meme_plan: dict[str, Any],
    languages: set[str] | None = None,
) -> None:
    expected = expected_scene_audio_ids(meme_plan, languages)
    missing: list[str] = []
    for lang, scene_ids in expected.items():
        generated = set((audio_paths.get(lang) or {}).keys())
        for scene_id in sorted(scene_ids - generated):
            missing.append(f"{lang}/{scene_id}")
    if missing:
        raise RuntimeError(f"TTS generation missing required scene audio: {', '.join(missing)}")


def assert_bgm_complete(
    *,
    bgm_paths: dict[str, Any],
    meme_plan: dict[str, Any],
    languages: set[str] | None = None,
) -> None:
    expected = _requested_languages(meme_plan, languages)
    missing = [lang for lang in expected if lang not in bgm_paths]
    if missing:
        raise RuntimeError(f"BGM generation missing required languages: {', '.join(missing)}")


def _requested_languages(meme_plan: dict[str, Any], languages: set[str] | None) -> list[str]:
    requested = languages or {"english", "korean", "spanish"}
    return [
        lang
        for lang in ("english", "korean", "spanish")
        if lang in requested and isinstance(meme_plan.get(lang), dict)
    ]


def _semantic_visual_mismatch(scene: dict[str, Any], lowered_prompt: str) -> str:
    tts = str(scene.get("tts_text", "")).strip().lower()
    if not tts:
        return ""

    checks = (
        (
            ("fever", "열", "fiebre", "temperature", "40 degrees", "40도"),
            ("thermometer", "water", "medicine", "medication", "light clothing", "blanket", "monitoring", "해열", "체온", "물", "termómetro", "agua"),
            "fever-related tts_text needs a visible fever-care anchor in image_prompt.",
        ),
        (
            ("x-ray", "xray", "stethoscope", "청진", "폐사진", "radiografía", "estetoscopio"),
            ("x-ray", "xray", "stethoscope", "folder", "doctor desk", "comparison", "청진", "폐사진", "radiografía", "estetoscopio"),
            "x-ray/stethoscope tts_text needs a visible diagnostic comparison anchor in image_prompt.",
        ),
        (
            ("er", "emergency", "urgent", "응급", "urgencias", "unresponsive", "trouble breathing", "breathing", "경련", "호흡"),
            ("phone", "pediatric", "doctor", "clinic", "warning", "monitoring", "caregiver watching", "breathing", "응급", "소아과", "urgencias", "médico"),
            "emergency-warning tts_text needs a visible triage or warning-sign anchor, not only routine care.",
        ),
    )

    for triggers, anchors, message in checks:
        if _has_any(tts, triggers) and not _has_any(lowered_prompt, anchors):
            return message
    return ""


def _has_repeated_nonempty(values: list[str], *, max_allowed: int) -> bool:
    for value in set(v for v in values if v):
        if values.count(value) > max_allowed:
            return True
    return False


def _prop_matches_allowed(prop: str, allowed_props: list[str]) -> bool:
    prop = prop.strip().lower()
    if not prop:
        return True
    return any(prop in allowed or allowed in prop for allowed in allowed_props)


def _has_selected_safe_score(creative_scores: dict[str, Any], language: str) -> bool:
    for score in creative_scores.get(language, []) or []:
        if score.get("selected") and int(score.get("medical_safety", 0)) >= 4:
            return True
    return False


def _is_serious_medical_topic(scripts: dict[str, Any], meme_plan: dict[str, Any]) -> bool:
    text = f"{json.dumps(scripts, ensure_ascii=False)} {json.dumps(meme_plan, ensure_ascii=False)}".lower()
    return _has_any(
        text,
        (
            "blood clot",
            "high blood pressure",
            "persistent swelling",
            "lymphedema",
            "ultrasound",
            "혈전",
            "고혈압",
            "부종",
            "림프",
            "초음파",
            "coágulo",
            "presión arterial",
            "hinchazón",
            "linfedema",
            "ecografía",
        ),
    )


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _word_count(text: str) -> int:
    return len([part for part in text.replace(":", " ").split() if part.strip()])


def _issue(subject: str, text: str) -> dict[str, Any]:
    return {
        "severity": "high",
        "issue": f"{subject}: {text}",
        "repair_instruction": f"Repair {subject}: {text}",
    }
