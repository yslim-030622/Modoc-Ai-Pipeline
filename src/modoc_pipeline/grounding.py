"""Gemini Search grounding for medical facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from .excel_source import QnaSource
from .gemini_client import GeminiResponseParseError, parse_json_response
from .schemas import GroundingCitation, GroundingReport


@dataclass(frozen=True)
class GroundingGeneration:
    parsed: dict[str, Any]
    raw_text: str
    grounding_metadata: dict[str, Any]


def generate_grounding_report(
    *,
    api_key: str,
    model: str,
    source: QnaSource,
    enable_search: bool,
) -> GroundingGeneration:
    if not enable_search:
        report = GroundingReport(
            status="expert_only",
            supported_facts=[
                {
                    "fact": "Use only facts explicitly present in the expert answer.",
                    "source": "expert_answer",
                    "source_indices": [],
                }
            ],
            notes=["Google Search grounding was skipped by configuration."],
        )
        return GroundingGeneration(parsed=report.model_dump(), raw_text=report.model_dump_json(), grounding_metadata={})

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=build_grounding_prompt(source),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GroundingReport,
            temperature=0.1,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    raw_text = response.text or ""
    metadata = _extract_grounding_metadata(response)
    try:
        parsed = GroundingReport.model_validate_json(raw_text)
    except ValidationError as exc:
        try:
            parsed = GroundingReport.model_validate(parse_json_response(raw_text))
        except Exception as inner:
            raise GeminiResponseParseError(
                raw_text,
                f"Gemini returned invalid grounding report: {exc}; fallback parse failed: {inner}",
            ) from inner

    payload = parsed.model_dump()
    _merge_grounding_metadata(payload, metadata)
    return GroundingGeneration(parsed=payload, raw_text=raw_text, grounding_metadata=metadata)


def build_grounding_prompt(source: QnaSource) -> str:
    return f"""
You are preparing medical grounding for a pediatric short-form video.
Use Google Search only to verify general pediatric facts that are relevant to
the source question and expert answer. Do not invent diagnosis or treatment.

Return a compact JSON grounding report. Supported facts must be short and safe
for parent education. If web grounding is weak, mark status as "insufficient"
and keep the final content grounded in the expert answer only.

Source row: {source.row_number}

Question:
{source.question_text}

Expert answer:
{source.expert_answer_text}
""".strip()


def _extract_grounding_metadata(response: Any) -> dict[str, Any]:
    try:
        candidate = response.candidates[0]
        metadata = getattr(candidate, "grounding_metadata", None)
        if metadata is None:
            metadata = getattr(candidate, "groundingMetadata", None)
        if metadata is None:
            return {}
        if hasattr(metadata, "model_dump"):
            return metadata.model_dump(exclude_none=True)
        if isinstance(metadata, dict):
            return metadata
    except Exception:
        return {}
    return {}


def _merge_grounding_metadata(payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    if not metadata:
        return
    queries = metadata.get("web_search_queries") or metadata.get("webSearchQueries") or []
    if queries and not payload.get("search_queries"):
        payload["search_queries"] = queries

    chunks = metadata.get("grounding_chunks") or metadata.get("groundingChunks") or []
    citations: list[GroundingCitation] = []
    for chunk in chunks:
        web = chunk.get("web", {}) if isinstance(chunk, dict) else {}
        uri = str(web.get("uri") or "").strip()
        title = str(web.get("title") or "").strip()
        if uri or title:
            citations.append(GroundingCitation(uri=uri, title=title))
    if citations:
        existing = {
            (item.get("uri", ""), item.get("title", ""))
            for item in payload.get("citations", [])
            if isinstance(item, dict)
        }
        for citation in citations:
            key = (citation.uri, citation.title)
            if key not in existing:
                payload.setdefault("citations", []).append(citation.model_dump())

