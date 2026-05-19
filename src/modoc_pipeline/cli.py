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
from .io_utils import read_json, write_json, write_text
from .renderer import render_language_videos
from .veo_client import (
    ensure_vertex_prerequisites,
    generate_gemini_veo_clips,
    generate_veo_clips,
    load_gemini_veo_config,
    load_vertex_config,
)
from .video_planner import generate_video_plan


DEFAULT_INPUT = "Q&A Blog Contents List.xlsx"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_LOG_PATH = "logs/pipeline_runs.csv"
DEFAULT_MODEL = "gemini-2.5-flash"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        return run_generate(args)
    if args.command == "plan-video":
        return run_plan_video(args)
    if args.command == "veo":
        return run_veo(args)
    if args.command == "veo-gemini":
        return run_veo_gemini(args)
    if args.command == "render":
        return run_render(args)

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

    plan_video = subparsers.add_parser(
        "plan-video",
        help="Create scene prompts from scripts.json for Vertex AI Veo.",
    )
    plan_video.add_argument("--run", required=True, help="Path to an outputs/<run_id> directory.")
    plan_video.add_argument(
        "--model",
        default=None,
        help="Gemini model name. Defaults to GEMINI_MODEL or gemini-2.5-flash.",
    )
    plan_video.add_argument("--scenes", type=int, default=3, help="Number of scenes per language.")
    plan_video.add_argument("--scene-duration", type=int, default=6, help="Seconds per scene.")

    veo = subparsers.add_parser(
        "veo",
        help="Generate Veo clips from scene_prompts.json using Vertex AI.",
    )
    veo.add_argument("--run", required=True, help="Path to an outputs/<run_id> directory.")
    veo.add_argument("--poll-seconds", type=int, default=10, help="Polling interval for Veo operations.")
    veo.add_argument("--languages", help="Comma-separated languages to generate, e.g. english,korean.")
    veo.add_argument("--max-clips", type=int, help="Maximum number of clips to generate for smoke tests.")

    veo_gemini = subparsers.add_parser(
        "veo-gemini",
        help="Generate Veo clips from scene_prompts.json using GEMINI_API_KEY.",
    )
    veo_gemini.add_argument("--run", required=True, help="Path to an outputs/<run_id> directory.")
    veo_gemini.add_argument("--poll-seconds", type=int, default=10, help="Polling interval for Veo operations.")
    veo_gemini.add_argument("--languages", help="Comma-separated languages to generate, e.g. english,korean.")
    veo_gemini.add_argument("--max-clips", type=int, help="Maximum number of clips to generate for smoke tests.")

    render = subparsers.add_parser(
        "render",
        help="Render final 9:16 MP4 files from Veo clips with burned subtitles.",
    )
    render.add_argument("--run", required=True, help="Path to an outputs/<run_id> directory.")
    render.add_argument("--languages", help="Comma-separated languages to render, e.g. english,korean.")
    render.add_argument("--max-clips", type=int, help="Maximum number of clips to render for smoke tests.")

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


def run_plan_video(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is missing. Add it to .env before planning video scenes.")
        return 2

    run_dir = Path(args.run)
    scripts_path = run_dir / "scripts.json"
    if not scripts_path.exists():
        print(f"Missing scripts.json. Run `generate` first or check the run path: {run_dir}")
        return 1

    model = args.model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    started = time.monotonic()
    try:
        scripts = read_json(scripts_path)
        plan_generation = generate_video_plan(
            api_key=api_key,
            model=model,
            scripts=scripts,
            scene_count=args.scenes,
            scene_duration=args.scene_duration,
        )
        parsed = plan_generation.parsed
        write_json(run_dir / "video_plan.json", parsed.get("video_plan", {}))
        write_json(run_dir / "scene_prompts.json", parsed.get("scenes", {}))
        write_text(run_dir / "raw_video_plan_response.txt", plan_generation.raw_text)
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "plan-video",
                "status": "succeeded",
                "model": model,
                "scene_count": args.scenes,
                "scene_duration_seconds": args.scene_duration,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
            },
        )
    except GeminiResponseParseError as exc:
        write_text(run_dir / "raw_video_plan_response.txt", exc.raw_text)
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "plan-video",
                "status": "failed",
                "model": model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "error": str(exc),
            },
        )
        print(f"Video planning failed: {exc}")
        return 1
    except Exception as exc:
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "plan-video",
                "status": "failed",
                "model": model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "error": str(exc),
            },
        )
        print(f"Video planning failed: {exc}")
        return 1

    print(f"Wrote video plan artifacts to {run_dir}.")
    return 0


def run_veo(args: argparse.Namespace) -> int:
    load_dotenv()
    run_dir = Path(args.run)
    scene_path = run_dir / "scene_prompts.json"
    if not scene_path.exists():
        print(f"Missing scene_prompts.json. Run `plan-video` first or check the run path: {run_dir}")
        return 1

    started = time.monotonic()
    try:
        ensure_vertex_prerequisites()
        config = load_vertex_config()
        scenes = filter_scenes(read_json(scene_path), languages=args.languages, max_clips=args.max_clips)
        written = generate_veo_clips(
            scenes=scenes,
            output_dir=run_dir / "veo",
            config=config,
            poll_seconds=args.poll_seconds,
        )
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "veo",
                "status": "succeeded",
                "model": config.model,
                "vertex_project_id": config.project_id,
                "vertex_location": config.location,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "clip_count": len(written),
                "clips": [str(path) for path in written],
            },
        )
    except Exception as exc:
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "veo",
                "status": "failed",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "error": str(exc),
            },
        )
        print(f"Veo generation failed: {exc}")
        return 1

    print(f"Wrote Veo clips to {run_dir / 'veo'}.")
    return 0


def run_veo_gemini(args: argparse.Namespace) -> int:
    load_dotenv()
    run_dir = Path(args.run)
    scene_path = run_dir / "scene_prompts.json"
    if not scene_path.exists():
        print(f"Missing scene_prompts.json. Run `plan-video` first or check the run path: {run_dir}")
        return 1

    started = time.monotonic()
    try:
        config = load_gemini_veo_config()
        scenes = filter_scenes(read_json(scene_path), languages=args.languages, max_clips=args.max_clips)
        written = generate_gemini_veo_clips(
            scenes=scenes,
            output_dir=run_dir / "veo",
            config=config,
            poll_seconds=args.poll_seconds,
        )
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "veo-gemini",
                "status": "succeeded",
                "model": config.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "clip_count": len(written),
                "clips": [str(path) for path in written],
            },
        )
    except Exception as exc:
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "veo-gemini",
                "status": "failed",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "error": str(exc),
            },
        )
        print(f"Gemini API Veo generation failed: {exc}")
        return 1

    print(f"Wrote Gemini API Veo clips to {run_dir / 'veo'}.")
    return 0


def run_render(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    scene_path = run_dir / "scene_prompts.json"
    if not scene_path.exists():
        print(f"Missing scene_prompts.json. Run `plan-video` first or check the run path: {run_dir}")
        return 1

    started = time.monotonic()
    try:
        scenes = filter_scenes(read_json(scene_path), languages=args.languages, max_clips=args.max_clips)
        rendered = render_language_videos(run_dir=run_dir, scenes=scenes)
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "render",
                "status": "succeeded",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "videos": [{"language": item.language, "path": str(item.path)} for item in rendered],
            },
        )
    except Exception as exc:
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "render",
                "status": "failed",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "error": str(exc),
            },
        )
        print(f"Render failed: {exc}")
        return 1

    print(f"Wrote final MP4 files to {run_dir / 'videos'}.")
    return 0


def filter_scenes(
    scenes: dict[str, list[dict]],
    *,
    languages: str | None,
    max_clips: int | None,
) -> dict[str, list[dict]]:
    selected_languages = None
    if languages:
        selected_languages = {item.strip() for item in languages.split(",") if item.strip()}

    filtered: dict[str, list[dict]] = {}
    remaining = max_clips
    for language, language_scenes in scenes.items():
        if selected_languages is not None and language not in selected_languages:
            continue
        chosen = list(language_scenes)
        if remaining is not None:
            if remaining <= 0:
                break
            chosen = chosen[:remaining]
            remaining -= len(chosen)
        if chosen:
            filtered[language] = chosen

    if not filtered:
        raise ValueError("No scenes selected. Check --languages and --max-clips.")
    return filtered
