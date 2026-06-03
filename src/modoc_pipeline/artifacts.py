"""Artifact writing and KPI timing log support."""

from __future__ import annotations

import csv
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

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

# ─────────────────────────────────────────────────────────────────────────────
# Automated stage timing
# ─────────────────────────────────────────────────────────────────────────────

STAGE_TIMING_FILE = "timing.json"


def record_stage(run_dir: Path, stage: str, duration_seconds: float, status: str = "ok") -> None:
    """Write one stage's timing into run_dir/timing.json.

    Each run writes only to its own run_dir, so concurrent runs never
    conflict. The file is read-modify-write but only from the single
    process that owns this run_dir.
    """
    path = run_dir / STAGE_TIMING_FILE
    run_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    data.setdefault("stages", {})[stage] = {
        "duration_seconds": round(duration_seconds, 2),
        "status": status,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finalize_run_timing(run_dir: Path, *, source_row: int, total_seconds: float, logs_dir: Path) -> None:
    """Append the completed run's timing summary to logs/run_timings.jsonl.

    Uses O_APPEND so each line write is atomic on macOS/Linux (POSIX guarantees
    atomicity for writes under PIPE_BUF, which is >= 512 bytes everywhere).
    Concurrent runs append independently without locking.
    """
    path = run_dir / STAGE_TIMING_FILE
    stages: dict[str, Any] = {}
    if path.exists():
        try:
            stages = json.loads(path.read_text(encoding="utf-8")).get("stages", {})
        except Exception:
            pass

    summary = {
        "run_id": run_dir.name,
        "source_row": source_row,
        "total_seconds": round(total_seconds, 2),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "stages": stages,
    }

    jsonl_path = logs_dir / "run_timings.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


@contextmanager
def timed_stage(run_dir: Path, stage: str) -> Generator[None, None, None]:
    """Context manager that records stage duration to timing.json on exit.

    Usage:
        with timed_stage(run_dir, "meme_plan"):
            ... do work ...
    """
    started = time.monotonic()
    status = "ok"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        record_stage(run_dir, stage, time.monotonic() - started, status)


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
    grounding_report: dict[str, Any] | None = None,
    quality_reports: dict[str, Any] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "source.json", source.to_dict())
    _write_json(run_dir / "scripts.json", generation.get("scripts", {}))
    _write_json(run_dir / "claims.json", generation.get("medical_claims", []))
    if grounding_report is not None:
        _write_json(run_dir / "grounding_report.json", grounding_report)
    _write_text(
        run_dir / "review_packet.md",
        build_review_packet(
            source,
            generation,
            grounding_report=grounding_report,
            quality_reports=quality_reports,
        ),
    )
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


def build_review_packet(
    source: QnaSource,
    generation: dict[str, Any],
    *,
    grounding_report: dict[str, Any] | None = None,
    quality_reports: dict[str, Any] | None = None,
    final_videos: list[dict[str, str]] | None = None,
    visual_qa_reports: list[dict[str, Any]] | None = None,
) -> str:
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
        "## Grounding",
        "",
    ]

    if grounding_report:
        lines.extend(
            [
                f"- Status: {grounding_report.get('status', '')}",
                f"- Search queries: {', '.join(grounding_report.get('search_queries', [])) or 'N/A'}",
                "",
                "### Citations",
                "",
            ]
        )
        for index, citation in enumerate(grounding_report.get("citations", []), start=1):
            lines.append(f"{index}. {citation.get('title', '')}: {citation.get('uri', '')}")
        lines.extend(["", "### Supported Facts", ""])
        for index, fact in enumerate(grounding_report.get("supported_facts", []), start=1):
            lines.append(f"{index}. {fact.get('fact', '')}")
        unsupported = grounding_report.get("unsupported_or_unsafe_claims", [])
        if unsupported:
            lines.extend(["", "### Unsupported or Unsafe Claims", ""])
            for item in unsupported:
                lines.append(f"- {item}")
        lines.append("")
    else:
        lines.extend(["Google Search grounding was not recorded for this run.", ""])

    lines.extend(
        [
        "## Generated Scripts",
        "",
        ]
    )

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
                f"- Grounding fact indices: {', '.join(str(i) for i in claim.get('grounding_fact_indices', [])) or 'N/A'}",
                f"- Citation indices: {', '.join(str(i) for i in claim.get('citation_indices', [])) or 'N/A'}",
                f"- Languages: {', '.join(claim.get('appears_in_languages', []))}",
                f"- Risk level: {claim.get('risk_level', '')}",
                "",
            ]
        )

    lines.extend(["## Gemini Quality Gate", ""])
    if quality_reports:
        for stage, reports in quality_reports.items():
            lines.extend([f"### {stage}", ""])
            if isinstance(reports, list):
                for report in reports:
                    lines.append(
                        f"- Attempt {report.get('attempt', '')}: "
                        f"{report.get('status', '')} - {report.get('summary', '')}"
                    )
            else:
                lines.append(f"- {reports}")
            lines.append("")
    else:
        lines.extend(["No quality reports were recorded yet.", ""])

    if visual_qa_reports:
        lines.extend(["## Gemini Visual QA", ""])
        for report in visual_qa_reports:
            lines.extend(
                [
                    f"### {report.get('scene_id', '')}",
                    "",
                    f"- Status: {report.get('status', '')}",
                    f"- Summary: {report.get('summary', '')}",
                    f"- Random objects: {', '.join(report.get('detected_random_objects', [])) or 'None reported'}",
                    f"- Forbidden elements: {', '.join(report.get('detected_forbidden_elements', [])) or 'None reported'}",
                    "",
                ]
            )
    else:
        lines.extend(["## Gemini Visual QA", "", "No visual QA reports were recorded.", ""])

    if final_videos:
        lines.extend(["## Final Videos", ""])
        for item in final_videos:
            lines.append(f"- {item.get('language', '')}: {item.get('path', '')}")
        lines.append("")

    lines.extend(
        [
            "## Human Final Review Checklist",
            "",
            "- Watch every final MP4 end to end.",
            "- Confirm captions match narration and are readable.",
            "- Confirm no visible text, logo, watermark, or unwanted medical props appear in Veo clips.",
            "- Confirm medical statements match the expert answer and cited grounding facts.",
            "- Confirm the video is suitable for parent education before publishing.",
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
