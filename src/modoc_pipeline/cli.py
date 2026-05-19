"""Command-line interface for the MoDoc pipeline MVP."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from .artifacts import (
    HumanTiming,
    append_timing_log,
    make_run_id,
    write_failure_artifacts,
    write_success_artifacts,
)
from .excel_source import load_qna_sources
from .gemini_client import GeminiResponseParseError, generate_short_form_package


DEFAULT_INPUT = "Q&A Blog Contents List.xlsx"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_LOG_PATH = "logs/pipeline_runs.csv"
DEFAULT_MODEL = "gemini-2.5-flash"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        return run_generate(args)

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modoc_pipeline",
        description="Generate short-form script and medical-review artifacts from MoDoc Q&A rows.",
    )
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser(
        "generate",
        help="Generate scripts, claims, review packets, and timing log rows.",
    )
    generate.add_argument("--input", default=DEFAULT_INPUT, help="Path to the Q&A Excel workbook.")
    generate.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for run artifacts.")
    generate.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="CSV path for KPI timing logs.")
    generate.add_argument("--limit", type=int, default=1, help="Number of source rows to process.")
    generate.add_argument("--row", type=int, help="Specific Excel row number to process.")
    generate.add_argument(
        "--status",
        default="Published",
        help="Status filter for source rows. Use 'any' to disable filtering.",
    )
    generate.add_argument(
        "--model",
        default=None,
        help="Gemini model name. Defaults to GEMINI_MODEL or gemini-2.5-flash.",
    )

    generate.add_argument("--content-selection-minutes", type=float, default=0.0)
    generate.add_argument("--format-decision-minutes", type=float, default=0.0)
    generate.add_argument("--script-generation-minutes", type=float, default=0.0)
    generate.add_argument("--medical-review-minutes", type=float, default=0.0)
    generate.add_argument("--upload-publish-minutes", type=float, default=0.0)
    generate.add_argument("--notes", default="", help="Free-form timing notes for the CSV log.")

    return parser


def run_generate(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is missing. Add it to .env before running generation.")
        return 2

    model = args.model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    log_path = Path(args.log_path)

    timing = HumanTiming(
        content_selection_minutes=args.content_selection_minutes,
        format_decision_minutes=args.format_decision_minutes,
        script_generation_minutes=args.script_generation_minutes,
        medical_review_minutes=args.medical_review_minutes,
        upload_publish_minutes=args.upload_publish_minutes,
        notes=args.notes,
    )

    try:
        sources = load_qna_sources(
            input_path,
            status_filter=args.status,
            row_number=args.row,
            limit=args.limit,
        )
    except Exception as exc:
        print(f"Failed to load source workbook: {exc}")
        return 1

    print(f"Loaded {len(sources)} source row(s) from {input_path}.")

    failures = 0
    for source in sources:
        run_id = make_run_id(source)
        run_dir = output_dir / run_id
        print(f"Generating artifacts for source row {source.row_number} ({run_id})...")

        started = time.monotonic()
        raw_text = ""
        try:
            generation = generate_short_form_package(
                api_key=api_key,
                model=model,
                source=source,
            )
            elapsed_seconds = round(time.monotonic() - started, 2)
            status = {
                "run_id": run_id,
                "status": "succeeded",
                "source_row": source.row_number,
                "model": model,
                "prompt_version": "script_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": elapsed_seconds,
            }
            write_success_artifacts(
                run_dir=run_dir,
                source=source,
                generation=generation.parsed,
                raw_text=generation.raw_text,
                status=status,
            )
        except GeminiResponseParseError as exc:
            raw_text = exc.raw_text
            failures += 1
            elapsed_seconds = round(time.monotonic() - started, 2)
            status = {
                "run_id": run_id,
                "status": "failed",
                "source_row": source.row_number,
                "model": model,
                "prompt_version": "script_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": elapsed_seconds,
                "error": str(exc),
            }
            write_failure_artifacts(
                run_dir=run_dir,
                source=source,
                raw_text=raw_text,
                status=status,
            )
            print(f"Generation failed for row {source.row_number}: {exc}")
            continue
        except Exception as exc:
            failures += 1
            elapsed_seconds = round(time.monotonic() - started, 2)
            status = {
                "run_id": run_id,
                "status": "failed",
                "source_row": source.row_number,
                "model": model,
                "prompt_version": "script_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": elapsed_seconds,
                "error": str(exc),
            }
            write_failure_artifacts(
                run_dir=run_dir,
                source=source,
                raw_text=raw_text,
                status=status,
            )
            print(f"Generation failed for row {source.row_number}: {exc}")
            continue

        append_timing_log(
            log_path=log_path,
            run_id=run_id,
            source_row=source.row_number,
            timing=timing,
        )
        print(f"Wrote artifacts to {run_dir}.")

    if failures:
        print(f"Completed with {failures} failure(s). Check each status.json for details.")
        return 1

    print(f"Timing log updated at {log_path}.")
    return 0
