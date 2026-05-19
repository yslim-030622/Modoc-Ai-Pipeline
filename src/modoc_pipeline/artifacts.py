"""Artifact writing and KPI timing log support."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .excel_source import QnaSource


TIMING_FIELDS = [
    "run_id",
    "source_row",
    "content_selection_minutes",
    "format_decision_minutes",
    "script_generation_minutes",
    "medical_review_minutes",
    "upload_publish_minutes",
    "human_total_minutes",
    "notes",
]


@dataclass(frozen=True)
class HumanTiming:
    """Human intervention minutes used by the project KPI denominator."""

    content_selection_minutes: float = 0.0
    format_decision_minutes: float = 0.0
    script_generation_minutes: float = 0.0
    medical_review_minutes: float = 0.0
    upload_publish_minutes: float = 0.0
    notes: str = ""

    @property
    def human_total_minutes(self) -> float:
        return round(
            self.content_selection_minutes
            + self.format_decision_minutes
            + self.script_generation_minutes
            + self.medical_review_minutes
            + self.upload_publish_minutes,
            2,
        )


def make_run_id(source: QnaSource) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_row{source.row_number}"


def write_success_artifacts(
    *,
    run_dir: Path,
    source: QnaSource,
    generation: dict[str, Any],
    raw_text: str,
    status: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "source.json", source.to_dict())
    _write_json(run_dir / "scripts.json", generation.get("scripts", {}))
    _write_json(run_dir / "claims.json", generation.get("medical_claims", []))
    _write_text(run_dir / "review_packet.md", build_review_packet(source, generation))
    _write_json(run_dir / "status.json", status)

    # The successful raw response is useful when prompt changes affect output
    # shape, but it is not the primary artifact reviewers should consume.
    _write_text(run_dir / "raw_gemini_response.txt", raw_text)


def write_failure_artifacts(
    *,
    run_dir: Path,
    source: QnaSource,
    raw_text: str,
    status: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "source.json", source.to_dict())
    _write_text(run_dir / "raw_gemini_response.txt", raw_text)
    _write_json(run_dir / "status.json", status)


def build_review_packet(source: QnaSource, generation: dict[str, Any]) -> str:
    scripts = generation.get("scripts", {})
    claims = generation.get("medical_claims", [])
    reviewer_notes = generation.get("reviewer_notes", [])

    lines = [
        "# Medical Review Packet",
        "",
        f"- Source row: {source.row_number}",
        f"- Blog status: {source.status_english or 'N/A'}",
        f"- Blog date: {source.english_date or 'N/A'}",
        f"- Blog URL: {source.wix_post_url_english or 'N/A'}",
        "",
        "## Source Question",
        "",
        source.question_text,
        "",
        "## Expert Answer",
        "",
        source.expert_answer_text,
        "",
        "## Generated Scripts",
        "",
    ]

    for key in ("english", "korean", "spanish"):
        script = scripts.get(key, {})
        lines.extend(
            [
                f"### {script.get('language', key.title())}",
                "",
                f"**Title:** {script.get('title', '')}",
                "",
                f"**Hook:** {script.get('hook', '')}",
                "",
                "**Body:**",
                "",
            ]
        )
        for item in script.get("body", []):
            lines.append(f"- {item}")
        lines.extend(
            [
                "",
                f"**Safety caveat:** {script.get('safety_caveat', '')}",
                "",
                f"**CTA:** {script.get('cta', '')}",
                "",
                f"**Estimated seconds:** {script.get('estimated_seconds', '')}",
                "",
            ]
        )

    lines.extend(["## Medical Claims to Review", ""])
    for index, claim in enumerate(claims, start=1):
        lines.extend(
            [
                f"### Claim {index}",
                "",
                f"- Claim: {claim.get('claim', '')}",
                f"- Evidence: {claim.get('evidence_from_expert_answer', '')}",
                f"- Languages: {', '.join(claim.get('appears_in_languages', []))}",
                f"- Risk level: {claim.get('risk_level', '')}",
                "",
            ]
        )

    lines.extend(["## Reviewer Notes", ""])
    for note in reviewer_notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def append_timing_log(
    *,
    log_path: Path,
    run_id: str,
    source_row: int,
    timing: HumanTiming,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = log_path.exists()

    # These fields intentionally capture human intervention time, not API wait
    # time. The internship KPI denominator is meant to measure repeatable human
    # effort required to move one video through the pipeline.
    row = {
        **asdict(timing),
        "run_id": run_id,
        "source_row": source_row,
        "human_total_minutes": timing.human_total_minutes,
    }

    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in TIMING_FIELDS})


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

