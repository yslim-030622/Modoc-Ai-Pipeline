"""Gemini quality gates and lightweight deterministic validations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from google import genai
from google.genai import types
from pydantic import ValidationError

from .gemini_client import GeminiResponseParseError, parse_json_response
from .schemas import QualityIssue, QualityReport


@dataclass(frozen=True)
class JudgedGeneration:
    payload: dict[str, Any]
    quality_reports: list[dict[str, Any]]
    raw_text: str


class QualityGateError(RuntimeError):
    def __init__(self, message: str, reports: list[dict[str, Any]], payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.reports = reports
        self.payload = payload


RepairFn = Callable[[str, dict[str, Any]], tuple[dict[str, Any], str]]


def judge_with_gemini(
    *,
    api_key: str,
    model: str,
    stage: str,
    payload: dict[str, Any],
    context: dict[str, Any],
    quality_gate: str,
) -> QualityReport:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=build_judge_prompt(
            stage=stage,
            payload=payload,
            context=context,
            quality_gate=quality_gate,
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QualityReport,
            temperature=0.1,
        ),
    )
    raw_text = response.text or ""
    try:
        return QualityReport.model_validate_json(raw_text)
    except ValidationError as exc:
        try:
            return QualityReport.model_validate(parse_json_response(raw_text))
        except Exception as inner:
            raise GeminiResponseParseError(
                raw_text,
                f"Gemini returned invalid quality report: {exc}; fallback parse failed: {inner}",
            ) from inner


def build_judge_prompt(
    *,
    stage: str,
    payload: dict[str, Any],
    context: dict[str, Any],
    quality_gate: str,
) -> str:
    return f"""
You are the internal Gemini-only quality gate for an automated MoDoc pediatric
video pipeline. Judge the payload for stage "{stage}" using a "{quality_gate}"
quality gate.

Pass only if:
- Medical claims are supported by the expert answer or grounding facts.
- No unsupported diagnosis, treatment, timeline, risk factor, or reassurance is added.
- Scripts are parent-friendly, accurate, and not fear-based.
- Scripts use short-form structure: quick hook, parent-relatable question or
  gentle myth-vs-fact turn, medically supported correction, and calm CTA.
- Localized text is semantically equivalent across the languages present in the
  payload. Do not fail for languages or scenes intentionally omitted by
  --languages or --max-clips; judge only what is present in Payload.
- Veo prompts contain no placeholders, no text-on-screen request, no realistic child,
  and are concise enough for the Veo prompt budget.
- Veo prompts use one stable mouthless MoDoc Guide icon plus at least one concrete
  infographic prop (e.g. footprint trail,
  calcium particle cluster, shield icon, clipboard, sunburst beam, balance arc)
  that directly represents the narration concept. A prompt composed only of generic
  geometric shapes (dot, ring, path, arc, circle, soft shape) with no recognizable
  prop fails visual clarity — flag it with a repair instruction naming a specific prop.
- Scene contracts must say what a muted viewer should understand and list concrete
  must_show objects/actions that directly match the narration.
- TTS/caption text is clear spoken narration. Do not fail only because narration
  is longer than the nominal scene duration; the renderer extends clips to match
  actual Gemini TTS duration.

Return JSON only. If failed, every issue must include a concrete repair_instruction.
When a localized track fails semantic equivalence, the repair_instruction must
include exact replacement tts_text/caption_text for every affected language.
Prefer repairing the localized track over regenerating the whole plan.

Context:
{json.dumps(context, ensure_ascii=False, indent=2)}

Payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def judge_and_repair(
    *,
    api_key: str,
    judge_model: str,
    stage: str,
    payload: dict[str, Any],
    raw_text: str,
    context: dict[str, Any],
    repair_fn: RepairFn,
    max_repair_attempts: int,
    quality_gate: str,
) -> JudgedGeneration:
    reports: list[dict[str, Any]] = []
    current_payload = payload
    current_raw_text = raw_text

    for attempt in range(max_repair_attempts + 1):
        deterministic = deterministic_quality_report(stage, current_payload)
        if deterministic.status == "failed":
            report = deterministic
        else:
            report = judge_with_gemini(
                api_key=api_key,
                model=judge_model,
                stage=stage,
                payload=current_payload,
                context=context,
                quality_gate=quality_gate,
            )
        report_payload = report.model_dump()
        report_payload["attempt"] = attempt
        reports.append(report_payload)
        if report.status == "passed":
            return JudgedGeneration(
                payload=current_payload,
                quality_reports=reports,
                raw_text=current_raw_text,
            )
        if attempt >= max_repair_attempts:
            break
        instruction = "\n".join(
            issue.repair_instruction or issue.issue
            for issue in report.issues
        )
        current_payload, current_raw_text = repair_fn(instruction, current_payload)

    raise QualityGateError(
        f"{stage} failed Gemini quality gate after {max_repair_attempts + 1} attempt(s): "
        f"{reports[-1].get('summary') or reports[-1].get('issues')}",
        reports,
        current_payload,
    )



def deterministic_quality_report(stage: str, payload: dict[str, Any]) -> QualityReport:
    issues: list[QualityIssue] = []
    if stage == "video_plan":
        for scene in payload.get("visual_scenes", []):
            scene_id = str(scene.get("scene_id") or "")
            prompt = str(scene.get("visual_prompt") or "")
            if not scene.get("meaning_contract"):
                issues.append(_issue(stage, scene_id, "Scene lacks meaning_contract for cross-language equivalence."))
            contract = scene.get("scene_contract") or {}
            if not contract:
                issues.append(_issue(stage, scene_id, "Scene lacks scene_contract for visual clarity."))
            else:
                if not str(contract.get("viewer_should_understand") or "").strip():
                    issues.append(_issue(stage, scene_id, "Scene contract lacks viewer_should_understand."))
                if len(contract.get("must_show") or []) < 1:
                    issues.append(_issue(stage, scene_id, "Scene contract lacks concrete must_show objects."))
            if "[" in prompt or "]" in prompt:
                issues.append(_issue(stage, scene_id, "Veo prompt contains placeholders."))
            if len(prompt.split()) > 180 or len(prompt) > 1024:
                issues.append(_issue(stage, scene_id, "Veo prompt is too long."))
            lowered = prompt.lower()
            if "realistic child" in lowered or "real child" in lowered or "photo-realistic child" in lowered:
                issues.append(_issue(stage, scene_id, "Veo prompt requests a realistic child."))
            if "no text anywhere in frame" not in lowered:
                issues.append(_issue(stage, scene_id, "Veo prompt lacks no-text instruction."))
            for banned in (
                "crib", "nursery", "baby bed", "toy", "plant",
                "apple", "picture frame", "shelf", "mouth closed", "smile",
                "hands", "fingers", "cartoon face", "eyes with pupils",
                "3d render", "clay render", "bedroom", "living room", "kitchen", "hospital",
                "clinic", "doctor", "nurse", "syringe", "needle", "medicine",
                "lip sync", "speaking mouth",
            ):
                if banned in lowered:
                    issues.append(_issue(stage, scene_id, f"Veo prompt contains banned visual element: {banned}."))
            for required in ("mouth-free", "lip-sync-free", "hand-free", "text-free", "no text anywhere in frame"):
                if required not in lowered:
                    issues.append(_issue(stage, scene_id, f"Veo prompt must specify {required}."))
            if "modoc guide" not in lowered and "mouthless" not in lowered:
                issues.append(_issue(stage, scene_id, "Veo prompt must include the stable mouthless MoDoc Guide."))
        tracks = payload.get("localized_tracks", {})
        visual_ids = {str(scene.get("scene_id") or "") for scene in payload.get("visual_scenes", [])}
        for lang, scenes in tracks.items():
            if not scenes:
                issues.append(_issue(stage, lang, "Localized track is empty."))
            for scene in scenes:
                text = str(scene.get("tts_text") or "")
                scene_id = str(scene.get("scene_id") or "")
                if scene_id not in visual_ids:
                    issues.append(_issue(stage, scene_id, f"{lang} track references a missing visual scene."))
                if lang == "english" and len(text.split()) > 34:
                    issues.append(_issue(stage, scene_id, "English TTS text is too long."))
                if lang == "spanish" and len(text.split()) > 38:
                    issues.append(_issue(stage, scene_id, "Spanish TTS text is too long."))
                if lang == "korean" and len(text) > 95:
                    issues.append(_issue(stage, scene_id, "Korean TTS text is too long."))
    elif stage == "script_package":
        for claim in payload.get("medical_claims", []):
            if not claim.get("evidence_from_expert_answer") and not claim.get("grounding_fact_indices"):
                issues.append(_issue(stage, claim.get("claim", ""), "Medical claim has no evidence link."))

    if issues:
        return QualityReport(status="failed", stage=stage, issues=issues, summary="Deterministic validation failed.")
    return QualityReport(status="passed", stage=stage, summary="Deterministic validation passed.")


def final_quality_report(
    *,
    source_row: int,
    videos: list[dict[str, str]],
    judge_reports: dict[str, Any],
    visual_qa_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return QualityReport(
        status="passed",
        stage="run-all",
        summary=f"Automated Gemini quality gates passed for source row {source_row}. Human final review is still required.",
        issues=[],
        human_review_checklist=[
            "Watch every final MP4 end to end.",
            "Confirm captions match narration and are readable.",
            "Confirm no visible text, logo, watermark, or unwanted medical props appear in Veo clips.",
            "Confirm medical statements match the expert answer and cited grounding facts.",
            "Confirm the video is suitable for parent education before publishing.",
        ],
    ).model_dump() | {
        "videos": videos,
        "judge_reports": judge_reports,
        "visual_qa_reports": visual_qa_reports or [],
    }


def _issue(stage: str, subject: str, text: str) -> QualityIssue:
    prefix = f"{subject}: " if subject else ""
    return QualityIssue(
        stage=stage,
        severity="high",
        issue=f"{prefix}{text}",
        repair_instruction=f"Repair {prefix}{text.lower()}",
    )
