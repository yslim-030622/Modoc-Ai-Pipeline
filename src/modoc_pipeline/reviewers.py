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
        if serious_topic and _has_any(
            bgm_prompt,
            (
                "party",
                "festive",
                "cute",
                "comedy",
                "hyperpop",
                "reggaeton",
                "dembow",
                "dance",
                "808",
                "huge drop",
                "club",
                "k-pop synth-pop",
            ),
        ):
            issues.append(_issue(subject, "BGM prompt is too playful or dance-focused for a serious medical topic. Use gentle, low-density, reassuring educational music."))
        bgm_config = lang_plan.get("bgm_config", {}) if isinstance(lang_plan.get("bgm_config"), dict) else {}
        if serious_topic:
            if float(bgm_config.get("density", 0.0) or 0.0) > 0.68:
                issues.append(_issue(subject, "BGM density is too high for a serious medical topic. Use density <= 0.68."))
            if float(bgm_config.get("brightness", 0.0) or 0.0) > 0.76:
                issues.append(_issue(subject, "BGM brightness is too high for a serious medical topic. Use brightness <= 0.76."))

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
    for lang in ("english", "korean", "spanish"):
        lang_plan = meme_plan.get(lang)
        if not isinstance(lang_plan, dict):
            continue
        anchor = str(lang_plan.get("visual_style_anchor", "")).strip()
        character = str(lang_plan.get("character_sheet", "")).strip()
        character_probe = character[:30]
        for scene in lang_plan.get("scenes", []) or []:
            scene_id = str(scene.get("scene_id", ""))
            prompt = str(scene.get("image_prompt", ""))
            subject = f"{lang}/{scene_id}"
            if anchor and not prompt.startswith(anchor[:40]):
                issues.append(_issue(subject, "image_prompt must start with the visual_style_anchor."))
            if character_probe and character_probe not in prompt:
                issues.append(_issue(subject, "image_prompt must include the locked character_sheet."))
            lowered = prompt.lower()
            forbidden_pairs = {
                "text": "image_prompt requests or permits visible text.",
                "letters": "image_prompt requests or permits visible letters.",
                "sign": "image_prompt requests or permits visible signs.",
                "logo": "image_prompt requests or permits visible logos.",
                "medical device": "image_prompt includes medical devices.",
                "realistic face": "image_prompt includes realistic faces.",
            }
            for token, message in forbidden_pairs.items():
                if token in lowered and "no " + token not in lowered and "zero " + token not in lowered:
                    issues.append(_issue(subject, message))

    if issues:
        return ReviewReport(
            status="failed",
            stage="visual_prompt_review",
            summary="Deterministic visual prompt validation failed.",
            issues=issues,
            repair_instruction="Rewrite failed image_prompt fields to include the anchor and character sheet, and ban visible text, logos, realistic faces, children, and medical devices.",
        )
    return ReviewReport(
        status="passed",
        stage="visual_prompt_review",
        summary="Deterministic visual prompt validation passed.",
    )


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
