"""Gemini generation client and response parsing."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from .excel_source import QnaSource
from .schemas import ScriptPackage


# Keywords that cause 400 INVALID_ARGUMENT when passed to Gemini response_schema.
# NUMBER type fields with minimum/maximum (especially minimum: 0.0) and numeric
# default values are rejected by the API. Stripping these keeps the structural
# schema intact while letting prompts + Pydantic handle the value constraints.
_GEMINI_STRIP_KEYS = frozenset({
    "default", "minimum", "maximum", "minItems", "maxItems",
    "exclusiveMinimum", "exclusiveMaximum",
})


def gemini_schema(model_cls: type) -> dict[str, Any]:
    """Return a Gemini-compatible schema dict from a Pydantic model class.

    Resolves $refs inline and strips keywords that cause 400 INVALID_ARGUMENT.
    Use this whenever passing a schema that contains NUMBER-typed fields
    (floats) or numeric default values to GenerateContentConfig.
    """
    raw = copy.deepcopy(model_cls.model_json_schema())
    defs = raw.pop("$defs", {})
    resolved = _resolve_refs(raw, defs)
    return _strip_keys(resolved)


def _resolve_refs(obj: Any, defs: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        if "$ref" in obj:
            ref_name = obj["$ref"].split("/")[-1]
            return _resolve_refs(copy.deepcopy(defs.get(ref_name, {})), defs)
        return {k: _resolve_refs(v, defs) for k, v in obj.items() if k != "$defs"}
    if isinstance(obj, list):
        return [_resolve_refs(item, defs) for item in obj]
    return obj


def _strip_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_keys(v) for k, v in obj.items() if k not in _GEMINI_STRIP_KEYS}
    if isinstance(obj, list):
        return [_strip_keys(item) for item in obj]
    return obj


@dataclass(frozen=True)
class GeminiGeneration:
    """Parsed model output plus the raw response for auditability."""

    parsed: dict[str, Any]
    raw_text: str


class GeminiResponseParseError(ValueError):
    """Raised when Gemini returns text that cannot be parsed as JSON."""

    def __init__(self, raw_text: str, message: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


def generate_short_form_package(
    *,
    api_key: str,
    model: str,
    source: QnaSource,
    grounding_report: dict[str, Any] | None = None,
    repair_instructions: str = "",
    previous_payload: dict[str, Any] | None = None,
) -> GeminiGeneration:
    client = genai.Client(api_key=api_key)
    prompt = build_generation_prompt(
        source,
        grounding_report=grounding_report,
        repair_instructions=repair_instructions,
        previous_payload=previous_payload,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ScriptPackage,
            temperature=0.2,
        ),
    )

    raw_text = response.text or ""
    try:
        parsed = ScriptPackage.model_validate_json(raw_text).model_dump()
    except ValidationError as exc:
        try:
            parsed = ScriptPackage.model_validate(parse_json_response(raw_text)).model_dump()
        except Exception as inner:
            raise GeminiResponseParseError(
                raw_text,
                f"Gemini returned invalid script package: {exc}; fallback parse failed: {inner}",
            ) from inner
    except Exception as exc:
        raise GeminiResponseParseError(raw_text, f"Gemini returned invalid JSON: {exc}") from exc
    return GeminiGeneration(parsed=parsed, raw_text=raw_text)


def build_generation_prompt(
    source: QnaSource,
    *,
    grounding_report: dict[str, Any] | None = None,
    repair_instructions: str = "",
    previous_payload: dict[str, Any] | None = None,
) -> str:
    """Build a single prompt that produces scripts and review artifacts together.

    Medical claims are separated from the scripts because reviewers need a
    concise evidence map, while editors need language-specific creative copy.
    Keeping both in one model call preserves consistency between the generated
    scripts and the review packet.
    """

    grounding_text = json.dumps(grounding_report or {}, ensure_ascii=False, indent=2)
    repair_text = ""
    if repair_instructions:
        repair_text = (
            "\nRepair instructions from Gemini quality gate:\n"
            f"{repair_instructions}\n\n"
            "Previous failed JSON payload to repair:\n"
            f"{json.dumps(previous_payload or {}, ensure_ascii=False, indent=2)}\n"
            "Repair the previous payload instead of starting over. Keep medically "
            "supported content and change only fields needed to pass the gate.\n"
        )

    return f"""
You are helping build a semi-automated short-form video pipeline for pediatric
health education. Convert the source Q&A into short-form scripts and a medical
review aid.

Hard constraints:
- Return valid JSON only. Do not include Markdown fences.
- Create scripts in English, Korean, and Spanish.
- Each script must target 35 seconds or less.
- Only use facts explicitly present in the expert answer. If a grounding report
  is provided, supported_facts may also be used. Do not add diagnoses, treatments,
  timelines, or risk factors that are absent from those sources.
- Preserve the expert answer's uncertainty word-for-word. "Less likely" must not
  become "safe" or "nothing to worry about". Use "can", "may", "ask a clinician".
- Include a safety caveat when symptoms require medical attention.
- Keep the parent-facing tone calm and direct. No fear hooks, clickbait, or
  definitive reassurance beyond what the expert answer supports.
- Structure every script for retention:
  1) 0-2 second hook: a parent-recognizable question or "wait, this can happen?" moment.
  2) A gentle tension beat: myth-vs-fact, parent POV, or "the key detail is..." turn.
  3) A medically supported correction in plain spoken language.
  4) A calm safety caveat / CTA.
- Use short spoken lines with natural pauses. Sound like a social video, not a blog.
- Keep each language semantically equivalent — same medical meaning, natural phrasing.
- Keep each script compact: one hook, 2-3 body lines, one safety caveat, one CTA.

Return JSON with this exact top-level shape:
{{
  "scripts": {{
    "english": {{
      "language": "English",
      "title": "...",
      "hook": "...",
      "body": ["...", "..."],
      "safety_caveat": "...",
      "cta": "...",
      "estimated_seconds": 30
    }},
    "korean": {{
      "language": "Korean",
      "title": "...",
      "hook": "...",
      "body": ["...", "..."],
      "safety_caveat": "...",
      "cta": "...",
      "estimated_seconds": 30
    }},
    "spanish": {{
      "language": "Spanish",
      "title": "...",
      "hook": "...",
      "body": ["...", "..."],
      "safety_caveat": "...",
      "cta": "...",
      "estimated_seconds": 30
    }}
  }},
  "medical_claims": [
    {{
      "claim": "...",
      "evidence_from_expert_answer": "...",
      "grounding_fact_indices": [0],
      "citation_indices": [0],
      "appears_in_languages": ["english", "korean", "spanish"],
      "risk_level": "low|medium|high"
    }}
  ],
  "editor_notes": ["..."],
  "reviewer_notes": ["..."]
}}

Source row: {source.row_number}
Blog status: {source.status_english}
Blog date: {source.english_date}
Blog URL: {source.wix_post_url_english}

Question:
{source.question_text}

Expert answer:
{source.expert_answer_text}

Grounding report:
{grounding_text}
{repair_text}
""".strip()


def parse_json_response(raw_text: str) -> dict[str, Any]:
    """Parse model JSON while tolerating occasional Markdown-style wrapping.

    The raw response is still preserved by the caller on failures. That makes
    bad model output debuggable instead of silently losing the only evidence of
    what Gemini returned.
    """

    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)
