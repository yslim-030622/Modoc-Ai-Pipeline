"""Gemini generation client and response parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from .excel_source import QnaSource


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
) -> GeminiGeneration:
    client = genai.Client(api_key=api_key)
    prompt = build_generation_prompt(source)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.4,
        ),
    )

    raw_text = response.text or ""
    try:
        parsed = parse_json_response(raw_text)
    except Exception as exc:
        raise GeminiResponseParseError(raw_text, f"Gemini returned invalid JSON: {exc}") from exc
    return GeminiGeneration(parsed=parsed, raw_text=raw_text)


def build_generation_prompt(source: QnaSource) -> str:
    """Build a single prompt that produces scripts and review artifacts together.

    Medical claims are separated from the scripts because reviewers need a
    concise evidence map, while editors need language-specific creative copy.
    Keeping both in one model call preserves consistency between the generated
    scripts and the review packet.
    """

    return f"""
You are helping build a semi-automated short-form video pipeline for pediatric
health education. Convert the source Q&A into short-form scripts and a medical
review aid.

Hard constraints:
- Return valid JSON only. Do not include Markdown fences.
- Create scripts in English, Korean, and Spanish.
- Each script must target 60 seconds or less.
- Do not diagnose the child. Do not invent facts not present in the expert answer.
- Avoid unsafe certainty. Use cautious language such as "can", "may", and
  "ask a clinician" where appropriate.
- Include a safety caveat when symptoms require medical attention.
- Medical claims must be traceable to the expert answer text.

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
      "estimated_seconds": 45
    }},
    "korean": {{
      "language": "Korean",
      "title": "...",
      "hook": "...",
      "body": ["...", "..."],
      "safety_caveat": "...",
      "cta": "...",
      "estimated_seconds": 45
    }},
    "spanish": {{
      "language": "Spanish",
      "title": "...",
      "hook": "...",
      "body": ["...", "..."],
      "safety_caveat": "...",
      "cta": "...",
      "estimated_seconds": 45
    }}
  }},
  "medical_claims": [
    {{
      "claim": "...",
      "evidence_from_expert_answer": "...",
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
