"""Command-line interface for the MoDoc pipeline MVP."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from .artifacts import (
    HumanTiming,
    append_timing_log,
    build_review_packet,
    make_run_id,
    write_failure_artifacts,
    write_success_artifacts,
)
from .excel_source import load_qna_sources
from .gemini_client import GeminiResponseParseError, generate_short_form_package
from .grounding import generate_grounding_report
from .io_utils import read_json, write_json, write_text
from .quality import final_quality_report, judge_and_repair
from .renderer import render_language_videos, render_shared_language_videos
from .tts_client import generate_tts_per_scene, generate_meme_gemini_tts, load_tts_config
from .veo_client import (
    VeoGenerationError,
    generate_gemini_veo_clips,
    generate_gemini_shared_veo_clips,
    load_gemini_veo_config,
)
from .video_planner import generate_video_plan
from .meme_planner import generate_meme_plan
from .imagen_client import generate_meme_images, load_imagen_config
from .renderer import render_meme_slideshow
from .bgm_client import generate_all_bgm


DEFAULT_INPUT = "Q&A Blog Contents List.xlsx"
DEFAULT_OUTPUT_DIR = "logs"
DEFAULT_VIDEO_DIR = "videos"
DEFAULT_LOG_PATH = "logs/pipeline_runs.csv"
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_JUDGE_MODEL = "gemini-3.5-flash"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        return run_generate(args)
    if args.command == "plan-video":
        return run_plan_video(args)
    if args.command == "veo":
        return run_veo(args)
    if args.command == "render":
        return run_render(args)
    if args.command == "tts":
        return run_tts(args)
    if args.command == "run-all":
        return run_all(args)
    if args.command == "meme-plan":
        return run_meme_plan(args)
    if args.command == "imagen":
        return run_imagen(args)
    if args.command == "meme-tts":
        return run_meme_tts(args)
    if args.command == "render-meme":
        return run_render_meme(args)
    if args.command == "meme-run-all":
        return run_meme_all(args)

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
    generate.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for JSON run logs.")
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
        help="Gemini model name. Defaults to GEMINI_MODEL or gemini-3.5-flash-lite.",
    )
    add_quality_arguments(generate)

    generate.add_argument("--content-selection-minutes", type=float, default=0.0)
    generate.add_argument("--format-decision-minutes", type=float, default=0.0)
    generate.add_argument("--script-generation-minutes", type=float, default=0.0)
    generate.add_argument("--medical-review-minutes", type=float, default=0.0)
    generate.add_argument("--upload-publish-minutes", type=float, default=0.0)
    generate.add_argument("--notes", default="", help="Free-form timing notes for the CSV log.")

    plan_video = subparsers.add_parser(
        "plan-video",
        help="Create scene prompts from scripts.json for Veo.",
    )
    plan_video.add_argument("--run", required=True, help="Path to a logs/<run_id> directory.")
    plan_video.add_argument(
        "--model",
        default=None,
        help="Gemini model name. Defaults to GEMINI_MODEL or gemini-3.5-flash-lite.",
    )
    plan_video.add_argument("--scenes", type=int, default=5, help="Number of scenes per language.")
    plan_video.add_argument("--scene-duration", type=int, default=6, help="Seconds per scene.")

    veo = subparsers.add_parser(
        "veo",
        help="Generate Veo clips from scene_prompts.json using the latest high-quality Veo model.",
    )
    veo.add_argument("--run", required=True, help="Path to a logs/<run_id> directory.")
    veo.add_argument("--poll-seconds", type=int, default=10, help="Polling interval for Veo operations.")
    veo.add_argument("--languages", help="Comma-separated languages to generate, e.g. english,korean.")
    veo.add_argument("--scene-ids", help="Comma-separated scene IDs to generate, e.g. scene_03.")
    veo.add_argument("--max-clips", type=int, help="Maximum number of clips to generate for smoke tests.")

    render = subparsers.add_parser(
        "render",
        help="Render final 9:16 MP4 files from Veo clips with burned subtitles.",
    )
    render.add_argument("--run", required=True, help="Path to a logs/<run_id> directory.")
    render.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR, help="Directory for final videos.")
    render.add_argument("--languages", help="Comma-separated languages to render, e.g. english,korean.")
    render.add_argument("--max-clips", type=int, help="Maximum number of clips to render for smoke tests.")

    tts = subparsers.add_parser(
        "tts",
        help="Generate Gemini TTS narration for localized tracks.",
    )
    tts.add_argument("--run", required=True, help="Path to a logs/<run_id> directory.")
    tts.add_argument("--languages", help="Comma-separated languages to synthesize.")

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Run generate, plan-video, Veo, TTS, and render in one command.",
    )
    add_source_arguments(run_all_parser)
    run_all_parser.add_argument("--scenes", type=int, default=4, help="Number of scenes per language.")
    run_all_parser.add_argument("--scene-duration", type=int, default=6, help="Seconds per scene.")
    run_all_parser.add_argument("--poll-seconds", type=int, default=10, help="Polling interval for Veo operations.")
    run_all_parser.add_argument("--languages", help="Comma-separated languages to generate/render.")
    run_all_parser.add_argument("--max-clips", type=int, help="Maximum number of clips for smoke tests.")
    run_all_parser.add_argument("--notes", default="run-all", help="Free-form timing notes for the CSV log.")
    run_all_parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR, help="Directory for final videos.")
    add_quality_arguments(run_all_parser)

    # ── Meme slideshow pipeline (free tier) ──────────────────────────────────

    meme_plan_parser = subparsers.add_parser(
        "meme-plan",
        help="Research trending memes per language and generate a MemePlan (free tier).",
    )
    meme_plan_parser.add_argument("--run", required=True, help="Path to a logs/<run_id> directory.")
    meme_plan_parser.add_argument(
        "--model",
        default=None,
        help="Gemini model for meme planning. Defaults to GEMINI_MEME_MODEL or gemini-2.0-flash.",
    )
    meme_plan_parser.add_argument("--scenes", type=int, default=4, help="Number of scenes per language (3-5).")

    imagen_parser = subparsers.add_parser(
        "imagen",
        help="Generate meme-style images using Gemini image generation (free tier).",
    )
    imagen_parser.add_argument("--run", required=True, help="Path to a logs/<run_id> directory.")
    imagen_parser.add_argument("--languages", help="Comma-separated languages, e.g. english,korean.")
    imagen_parser.add_argument("--max-images", type=int, help="Max images per language (for smoke tests).")

    meme_tts_parser = subparsers.add_parser(
        "meme-tts",
        help="Generate free voiceover using edge-tts (no API key needed). Install: pip install edge-tts",
    )
    meme_tts_parser.add_argument("--run", required=True, help="Path to a logs/<run_id> directory.")
    meme_tts_parser.add_argument("--languages", help="Comma-separated languages to synthesize.")

    render_meme_parser = subparsers.add_parser(
        "render-meme",
        help="Assemble meme slideshow MP4s (Ken Burns + xfade + optional edge-tts audio).",
    )
    render_meme_parser.add_argument("--run", required=True, help="Path to a logs/<run_id> directory.")
    render_meme_parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR, help="Directory for final videos.")
    render_meme_parser.add_argument("--languages", help="Comma-separated languages to render.")
    render_meme_parser.add_argument("--no-zoom", action="store_true", help="Disable Ken Burns zoom effect.")

    meme_all_parser = subparsers.add_parser(
        "meme-run-all",
        help="Run meme-plan → imagen → [meme-tts] → render-meme in one command (free tier).",
    )
    meme_all_parser.add_argument("--run", required=True, help="Path to a logs/<run_id> directory with scripts.json.")
    meme_all_parser.add_argument(
        "--model",
        default=None,
        help="Gemini model for meme planning. Defaults to GEMINI_MEME_MODEL or gemini-2.0-flash.",
    )
    meme_all_parser.add_argument("--languages", help="Comma-separated languages to generate/render.")
    meme_all_parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR, help="Directory for final videos.")
    meme_all_parser.add_argument("--skip-tts", action="store_true", help="Skip edge-tts (produce silent videos).")
    meme_all_parser.add_argument("--no-zoom", action="store_true", help="Disable Ken Burns zoom effect.")

    return parser


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to the Q&A Excel workbook.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for JSON run logs.")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="CSV path for KPI timing logs.")
    parser.add_argument("--row", type=int, help="Specific Excel row number to process.")
    parser.add_argument(
        "--status",
        default="Published",
        help="Status filter for source rows. Use 'any' to disable filtering.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model name. Defaults to GEMINI_MODEL or gemini-3.5-flash-lite.",
    )


def add_quality_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skip-search", action="store_true", help="Skip Gemini Google Search grounding.")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Gemini judge model. Defaults to GEMINI_JUDGE_MODEL or gemini-3.5-flash-lite.",
    )
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument("--quality-gate", default="strict", choices=("strict", "standard"))


def run_generate(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is missing. Add it to .env before running generation.")
        return 2

    model = args.model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    judge_model = args.judge_model or os.getenv("GEMINI_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
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
            stage_result = run_grounded_script_stage(
                api_key=api_key,
                model=model,
                judge_model=judge_model,
                source=source,
                enable_search=not args.skip_search and _env_bool("GEMINI_ENABLE_SEARCH", default=True),
                max_repair_attempts=args.max_repair_attempts,
                quality_gate=args.quality_gate,
                run_dir=run_dir,
            )
            generation = stage_result["generation"]
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
                generation=generation["parsed"],
                raw_text=generation["raw_text"],
                status=status,
                grounding_report=stage_result["grounding_report"],
                quality_reports=stage_result["quality_reports"],
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


def run_grounded_script_stage(
    *,
    api_key: str,
    model: str,
    judge_model: str,
    source,
    enable_search: bool,
    max_repair_attempts: int,
    quality_gate: str,
    run_dir: Path,
) -> dict[str, Any]:
    grounding = generate_grounding_report(
        api_key=api_key,
        model=judge_model,
        source=source,
        enable_search=enable_search,
    )
    write_json(run_dir / "grounding_report.json", grounding.parsed)
    write_text(run_dir / "raw_grounding_response.txt", grounding.raw_text)

    def generate_once(
        repair_instructions: str = "",
        previous_payload: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str]:
        generation = generate_short_form_package(
            api_key=api_key,
            model=model,
            source=source,
            grounding_report=grounding.parsed,
            repair_instructions=repair_instructions,
            previous_payload=previous_payload,
        )
        return generation.parsed, generation.raw_text

    initial_payload, initial_raw = generate_once()
    judged = judge_and_repair(
        api_key=api_key,
        judge_model=judge_model,
        stage="script_package",
        payload=initial_payload,
        raw_text=initial_raw,
        context={"source": source.to_dict(), "grounding_report": grounding.parsed},
        repair_fn=generate_once,
        max_repair_attempts=max_repair_attempts,
        quality_gate=quality_gate,
    )
    return {
        "grounding_report": grounding.parsed,
        "generation": {"parsed": judged.payload, "raw_text": judged.raw_text},
        "quality_reports": {"script_package": judged.quality_reports},
    }


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
        write_json(run_dir / "visual_scenes.json", parsed.get("visual_scenes", []))
        write_json(run_dir / "localized_tracks.json", parsed.get("localized_tracks", {}))
        write_json(run_dir / "scene_prompts.json", parsed)
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
        config = load_gemini_veo_config()
        visual_path = run_dir / "visual_scenes.json"
        if visual_path.exists():
            all_visual_scenes = read_json(visual_path)
            visual_scenes = filter_visual_scenes(
                all_visual_scenes,
                scene_ids=args.scene_ids,
                max_clips=args.max_clips,
            )
            initial_frame_path = _find_initial_frame(
                run_dir / "veo" / "shared",
                all_scenes=all_visual_scenes,
                selected_scenes=visual_scenes,
            )
            written = generate_gemini_shared_veo_clips(
                visual_scenes=visual_scenes,
                output_dir=run_dir / "veo",
                config=config,
                poll_seconds=args.poll_seconds,
                initial_frame_path=initial_frame_path,
            )
        else:
            scenes = filter_scenes(
                read_json(scene_path),
                languages=args.languages,
                scene_ids=args.scene_ids,
                max_clips=args.max_clips,
            )
            written = generate_gemini_veo_clips(
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
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "clip_count": len(written),
                "clips": [str(path) for path in written],
            },
        )
    except VeoGenerationError as exc:
        write_json(run_dir / "veo_error_report.json", exc.report)
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


def run_render(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    scene_path = run_dir / "scene_prompts.json"
    if not scene_path.exists():
        print(f"Missing scene_prompts.json. Run `plan-video` first or check the run path: {run_dir}")
        return 1

    started = time.monotonic()
    try:
        visual_path = run_dir / "visual_scenes.json"
        tracks_path = run_dir / "localized_tracks.json"
        if visual_path.exists() and tracks_path.exists():
            visual_scenes, localized_tracks = select_shared_plan(
                visual_scenes=read_json(visual_path),
                localized_tracks=read_json(tracks_path),
                languages=parse_csv(args.languages),
                max_clips=args.max_clips,
            )
            rendered = render_shared_language_videos(
                run_dir=run_dir,
                visual_scenes=visual_scenes,
                localized_tracks=localized_tracks,
                video_output_dir=video_output_dir_for_run(run_dir, Path(args.video_dir)),
            )
        else:
            scenes = filter_scenes(
                read_json(scene_path),
                languages=args.languages,
                scene_ids=None,
                max_clips=args.max_clips,
            )
            rendered = render_language_videos(
                run_dir=run_dir,
                scenes=scenes,
                video_output_dir=video_output_dir_for_run(run_dir, Path(args.video_dir)),
            )
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

    print(f"Wrote final MP4 files to {video_output_dir_for_run(run_dir, Path(args.video_dir))}.")
    return 0


def run_tts(args: argparse.Namespace) -> int:
    load_dotenv()
    run_dir = Path(args.run)
    tracks_path = run_dir / "localized_tracks.json"
    if not tracks_path.exists():
        print(f"Missing localized_tracks.json. Run `plan-video` first or check the run path: {run_dir}")
        return 1
    started = time.monotonic()
    try:
        config = load_tts_config()
        languages = parse_csv(args.languages)
        # Per-scene TTS: each scene gets its own WAV for frame-accurate sync.
        # The renderer auto-detects per-scene files and uses them preferentially.
        scene_audio = generate_tts_per_scene(
            localized_tracks=read_json(tracks_path),
            output_dir=run_dir / "audio",
            config=config,
            languages=languages,
        )
        audio_summary = {
            language: {sid: str(p) for sid, p in scenes.items()}
            for language, scenes in scene_audio.items()
        }
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "tts",
                "status": "succeeded",
                "model": config.model,
                "voice": "per-language",
                "mode": "per-scene",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "audio": audio_summary,
            },
        )
    except Exception as exc:
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "tts",
                "status": "failed",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "error": str(exc),
            },
        )
        print(f"TTS failed: {exc}")
        return 1

    print(f"Wrote per-scene TTS audio to {run_dir / 'audio'}.")
    return 0


def run_all(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is missing. Add it to .env before running the full pipeline.")
        return 2

    model = args.model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    judge_model = args.judge_model or os.getenv("GEMINI_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    log_path = Path(args.log_path)

    try:
        sources = load_qna_sources(
            input_path,
            status_filter=args.status,
            row_number=args.row,
            limit=1,
        )
    except Exception as exc:
        print(f"Failed to load source workbook: {exc}")
        return 1

    source = sources[0]
    run_id = make_run_id(source)
    run_dir = output_dir / run_id
    total_started = time.monotonic()
    quality_reports: dict[str, Any] = {}
    grounding_report: dict[str, Any] = {}
    print(f"Running full pipeline for source row {source.row_number} ({run_id})...")

    try:
        generation_started = time.monotonic()
        stage_result = run_grounded_script_stage(
            api_key=api_key,
            model=model,
            judge_model=judge_model,
            source=source,
            enable_search=not args.skip_search and _env_bool("GEMINI_ENABLE_SEARCH", default=True),
            max_repair_attempts=args.max_repair_attempts,
            quality_gate=args.quality_gate,
            run_dir=run_dir,
        )
        generation = stage_result["generation"]
        grounding_report = stage_result["grounding_report"]
        quality_reports.update(stage_result["quality_reports"])
        write_success_artifacts(
            run_dir=run_dir,
            source=source,
            generation=generation["parsed"],
            raw_text=generation["raw_text"],
            status={
                "run_id": run_id,
                "status": "succeeded",
                "source_row": source.row_number,
                "model": model,
                "prompt_version": "script_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - generation_started, 2),
            },
            grounding_report=grounding_report,
            quality_reports=quality_reports,
        )
        append_timing_log(
            log_path=log_path,
            run_id=run_id,
            source_row=source.row_number,
            timing=HumanTiming(notes=args.notes),
        )
        print("Stage complete: generate")

        plan_started = time.monotonic()
        def plan_once(
            repair_instructions: str = "",
            previous_payload: dict[str, Any] | None = None,
        ) -> tuple[dict[str, Any], str]:
            plan = generate_video_plan(
                api_key=api_key,
                model=model,
                scripts=generation["parsed"].get("scripts", {}),
                grounding_report=grounding_report,
                scene_count=args.scenes,
                scene_duration=args.scene_duration,
                repair_instructions=repair_instructions,
                previous_payload=previous_payload,
            )
            selected_visuals, selected_tracks = select_shared_plan(
                visual_scenes=plan.parsed.get("visual_scenes", []),
                localized_tracks=plan.parsed.get("localized_tracks", {}),
                languages=parse_csv(args.languages),
                max_clips=args.max_clips,
            )
            selected_video_plan = {
                **plan.parsed.get("video_plan", {}),
                "scene_count": len(selected_visuals),
            }
            selected = {
                **plan.parsed,
                "video_plan": selected_video_plan,
                "visual_scenes": selected_visuals,
                "localized_tracks": selected_tracks,
            }
            return selected, plan.raw_text

        initial_plan, initial_plan_raw = plan_once()
        judged_plan = judge_and_repair(
            api_key=api_key,
            judge_model=judge_model,
            stage="video_plan",
            payload=initial_plan,
            raw_text=initial_plan_raw,
            context={
                "source": source.to_dict(),
                "scripts": generation["parsed"].get("scripts", {}),
                "grounding_report": grounding_report,
                "selected_languages": sorted(parse_csv(args.languages) or initial_plan.get("localized_tracks", {}).keys()),
                "selected_scene_ids": [scene.get("scene_id") for scene in initial_plan.get("visual_scenes", [])],
                "selection_note": "Only selected languages and scenes are present in this payload.",
            },
            repair_fn=plan_once,
            max_repair_attempts=args.max_repair_attempts,
            quality_gate=args.quality_gate,
        )
        parsed_plan = judged_plan.payload
        quality_reports["video_plan"] = judged_plan.quality_reports
        visual_scenes = parsed_plan.get("visual_scenes", [])
        localized_tracks = parsed_plan.get("localized_tracks", {})
        write_json(run_dir / "video_plan.json", parsed_plan.get("video_plan", {}))
        write_json(run_dir / "visual_scenes.json", visual_scenes)
        write_json(run_dir / "localized_tracks.json", localized_tracks)
        # Keep the full model response for debugging prompt drift and schema changes.
        write_json(run_dir / "scene_prompts.json", parsed_plan)
        write_text(run_dir / "raw_video_plan_response.txt", judged_plan.raw_text)
        write_json(run_dir / "quality_report.json", {"stage_reports": quality_reports})
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "plan-video",
                "status": "succeeded",
                "model": model,
                "scene_count": args.scenes,
                "scene_duration_seconds": args.scene_duration,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - plan_started, 2),
            },
        )
        print("Stage complete: plan-video")

        veo_started = time.monotonic()
        veo_config = load_gemini_veo_config()
        try:
            clips = generate_gemini_shared_veo_clips(
                visual_scenes=visual_scenes,
                output_dir=run_dir / "veo",
                config=veo_config,
                poll_seconds=args.poll_seconds,
            )
        except VeoGenerationError as exc:
            write_json(run_dir / "veo_error_report.json", exc.report)
            raise
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "veo",
                "status": "succeeded",
                "model": veo_config.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - veo_started, 2),
                "clip_count": len(clips),
                "clips": [str(path) for path in clips],
            },
        )
        print("Stage complete: veo")

        tts_started = time.monotonic()
        tts_config = load_tts_config()
        scene_audio = generate_tts_per_scene(
            localized_tracks=localized_tracks,
            output_dir=run_dir / "audio",
            config=tts_config,
        )
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "tts",
                "status": "succeeded",
                "model": tts_config.model,
                "voice": "per-language",
                "mode": "per-scene",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - tts_started, 2),
                "audio": {lang: {sid: str(p) for sid, p in scenes.items()} for lang, scenes in scene_audio.items()},
            },
        )
        print("Stage complete: tts")

        render_started = time.monotonic()
        rendered = render_shared_language_videos(
            run_dir=run_dir,
            visual_scenes=visual_scenes,
            localized_tracks=localized_tracks,
            video_output_dir=video_output_dir_for_run(run_dir, Path(args.video_dir)),
        )
        rendered_payload = [{"language": item.language, "path": str(item.path)} for item in rendered]
        final_report = final_quality_report(
            source_row=source.row_number,
            videos=rendered_payload,
            judge_reports=quality_reports,
        )
        write_json(run_dir / "quality_report.json", final_report)
        write_text(
            run_dir / "review_packet.md",
            build_review_packet(
                source,
                generation["parsed"],
                grounding_report=grounding_report,
                quality_reports=quality_reports,
                final_videos=rendered_payload,
            ),
        )
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "run-all",
                "status": "succeeded",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - total_started, 2),
                "render_seconds": round(time.monotonic() - render_started, 2),
                "videos": rendered_payload,
            },
        )
        print("Stage complete: render")
    except Exception as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            run_dir / "video_status.json",
            {
                "stage": "run-all",
                "status": "failed",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - total_started, 2),
                "error": str(exc),
            },
        )
        print(f"run-all failed: {exc}")
        print(f"Partial artifacts, if any, are in {run_dir}.")
        return 1

    print(f"Full pipeline complete: {run_dir}")
    for item in rendered:
        print(f"- {item.language}: {item.path}")
    return 0


def filter_scenes(
    scenes: dict[str, list[dict]],
    *,
    languages: str | None,
    scene_ids: str | None,
    max_clips: int | None,
) -> dict[str, list[dict]]:
    selected_languages = None
    if languages:
        selected_languages = {item.strip() for item in languages.split(",") if item.strip()}
    selected_scene_ids = None
    if scene_ids:
        selected_scene_ids = {item.strip() for item in scene_ids.split(",") if item.strip()}

    filtered: dict[str, list[dict]] = {}
    remaining = max_clips
    for language, language_scenes in scenes.items():
        if selected_languages is not None and language not in selected_languages:
            continue
        chosen = [
            scene for scene in language_scenes
            if selected_scene_ids is None or scene.get("scene_id") in selected_scene_ids
        ]
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


def parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    return parsed or None


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def filter_visual_scenes(
    visual_scenes: list[dict],
    *,
    scene_ids: str | None,
    max_clips: int | None,
) -> list[dict]:
    selected_scene_ids = parse_csv(scene_ids)
    selected = [
        scene for scene in visual_scenes
        if selected_scene_ids is None or scene.get("scene_id") in selected_scene_ids
    ]
    if max_clips is not None:
        selected = selected[:max_clips]
    if not selected:
        raise ValueError("No visual scenes selected. Check --scene-ids and --max-clips.")
    return selected


def select_shared_plan(
    *,
    visual_scenes: list[dict],
    localized_tracks: dict[str, list[dict]],
    languages: set[str] | None,
    max_clips: int | None,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Select a consistent shared-visual subset across all localized tracks.

    The same Veo clips must be reused for every language; otherwise the final
    English/Korean/Spanish videos drift visually and become hard to compare.
    This validation fails early when the planner omits a track or invents a
    scene ID that has no matching visual clip.
    """

    selected_visuals = visual_scenes[:max_clips] if max_clips is not None else visual_scenes
    if not selected_visuals:
        raise ValueError("Video plan does not contain visual_scenes.")

    selected_ids = [str(scene.get("scene_id", "")).strip() for scene in selected_visuals]
    selected_id_set = set(selected_ids)
    if len(selected_id_set) != len(selected_ids) or "" in selected_id_set:
        raise ValueError("visual_scenes must have unique non-empty scene_id values.")

    selected_tracks: dict[str, list[dict]] = {}
    for language, tracks in localized_tracks.items():
        if languages is not None and language not in languages:
            continue
        by_scene_id = {str(track.get("scene_id", "")).strip(): track for track in tracks}
        missing = [scene_id for scene_id in selected_ids if scene_id not in by_scene_id]
        if missing:
            raise ValueError(f"localized_tracks.{language} is missing scene(s): {', '.join(missing)}")
        selected_tracks[language] = [by_scene_id[scene_id] for scene_id in selected_ids]

    if not selected_tracks:
        raise ValueError("No localized tracks selected. Check --languages.")
    return selected_visuals, selected_tracks


def _find_initial_frame(
    shared_dir: Path,
    *,
    all_scenes: list[dict],
    selected_scenes: list[dict],
) -> Path | None:
    """Find the last-frame JPEG of the scene immediately before the first selected scene.

    When re-generating a subset of clips (e.g. scene_02..04 only), this seeds
    frame continuation so scene_02 still uses scene_01's last frame as its
    first frame, rather than starting from scratch with no continuity anchor.
    """
    if not selected_scenes:
        return None
    first_selected_id = selected_scenes[0].get("scene_id", "")
    all_ids = [s.get("scene_id", "") for s in all_scenes]
    try:
        idx = all_ids.index(first_selected_id)
    except ValueError:
        return None
    if idx == 0:
        return None
    prev_id = all_ids[idx - 1]
    candidate = shared_dir / f"{prev_id}_last_frame.jpg"
    return candidate if candidate.exists() else None


def video_output_dir_for_run(run_dir: Path, video_root: Path = Path(DEFAULT_VIDEO_DIR)) -> Path:
    source_path = run_dir / "source.json"
    if source_path.exists():
        try:
            source = read_json(source_path)
            row_number = source.get("row_number")
            if row_number:
                return video_root / f"Row_{row_number}"
        except Exception:
            pass
    return video_root / run_dir.name


# ─────────────────────────────────────────────────────────────────────────────
# Meme slideshow pipeline handlers (free tier)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MEME_MODEL = "gemini-3.5-flash"


def run_meme_plan(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is missing. Add it to .env before running meme-plan.")
        return 2

    run_dir = Path(args.run)
    scripts_path = run_dir / "scripts.json"
    if not scripts_path.exists():
        print(f"Missing scripts.json. Run `generate` first: {run_dir}")
        return 1

    model = args.model or os.getenv("GEMINI_MEME_MODEL", DEFAULT_MEME_MODEL)
    started = time.monotonic()
    try:
        scripts = read_json(scripts_path)
        topic = _derive_topic(scripts)
        print(f"Researching meme trends and generating plan for: {topic}")
        result = generate_meme_plan(api_key=api_key, model=model, scripts=scripts, topic=topic)
        write_json(run_dir / "meme_plan.json", result.parsed)
        write_text(run_dir / "raw_meme_plan_response.txt", result.raw_text)
        write_json(
            run_dir / "meme_status.json",
            {
                "stage": "meme-plan",
                "status": "succeeded",
                "model": model,
                "topic": topic,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
            },
        )
        print(f"Wrote meme_plan.json to {run_dir}.")
    except Exception as exc:
        write_json(
            run_dir / "meme_status.json",
            {
                "stage": "meme-plan",
                "status": "failed",
                "error": str(exc),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
            },
        )
        print(f"meme-plan failed: {exc}")
        return 1
    return 0


def run_imagen(args: argparse.Namespace) -> int:
    load_dotenv()
    run_dir = Path(args.run)
    meme_plan_path = run_dir / "meme_plan.json"
    if not meme_plan_path.exists():
        print(f"Missing meme_plan.json. Run `meme-plan` first: {run_dir}")
        return 1

    started = time.monotonic()
    try:
        config = load_imagen_config()
        meme_plan = read_json(meme_plan_path)
        languages = parse_csv(args.languages)

        if hasattr(args, "max_images") and args.max_images:
            for lang_key in meme_plan:
                if isinstance(meme_plan[lang_key], dict):
                    meme_plan[lang_key]["scenes"] = meme_plan[lang_key].get("scenes", [])[:args.max_images]

        print(f"Generating meme images using {config.model}...")
        result = generate_meme_images(
            meme_plan=meme_plan,
            output_dir=run_dir / "meme_images",
            config=config,
            languages=languages,
        )
        image_summary = {lang: {sid: str(p) for sid, p in scenes.items()} for lang, scenes in result.items()}
        write_json(
            run_dir / "meme_status.json",
            {
                "stage": "imagen",
                "status": "succeeded",
                "model": config.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "images": image_summary,
            },
        )
        total = sum(len(v) for v in result.values())
        print(f"Generated {total} image(s) → {run_dir / 'meme_images'}.")
    except Exception as exc:
        write_json(
            run_dir / "meme_status.json",
            {
                "stage": "imagen",
                "status": "failed",
                "error": str(exc),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
            },
        )
        print(f"imagen failed: {exc}")
        return 1
    return 0


def run_meme_tts(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    meme_plan_path = run_dir / "meme_plan.json"
    if not meme_plan_path.exists():
        print(f"Missing meme_plan.json. Run `meme-plan` first: {run_dir}")
        return 1

    started = time.monotonic()
    try:
        from .edge_tts_client import generate_meme_tts
    except ImportError as exc:
        print(f"edge-tts not installed: {exc}")
        print("Install with: pip install edge-tts  OR  pip install -e '.[meme]'")
        return 2

    try:
        meme_plan = read_json(meme_plan_path)
        languages = parse_csv(args.languages)
        print("Generating voiceover with edge-tts (free)...")
        result = generate_meme_tts(
            meme_plan=meme_plan,
            output_dir=run_dir / "meme_audio",
            languages=languages,
        )
        audio_summary = {lang: {sid: str(p) for sid, p in scenes.items()} for lang, scenes in result.items()}
        write_json(
            run_dir / "meme_status.json",
            {
                "stage": "meme-tts",
                "status": "succeeded",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "audio": audio_summary,
            },
        )
        total = sum(len(v) for v in result.values())
        print(f"Generated {total} audio file(s) → {run_dir / 'meme_audio'}.")
    except Exception as exc:
        write_json(
            run_dir / "meme_status.json",
            {
                "stage": "meme-tts",
                "status": "failed",
                "error": str(exc),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
            },
        )
        print(f"meme-tts failed: {exc}")
        return 1
    return 0


def run_render_meme(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    meme_plan_path = run_dir / "meme_plan.json"
    if not meme_plan_path.exists():
        print(f"Missing meme_plan.json. Run `meme-plan` first: {run_dir}")
        return 1

    started = time.monotonic()
    try:
        meme_plan = read_json(meme_plan_path)
        languages = parse_csv(args.languages)
        video_dir = video_output_dir_for_run(run_dir, Path(args.video_dir))
        print("Rendering meme slideshow videos...")
        rendered = render_meme_slideshow(
            run_dir=run_dir,
            meme_plan=meme_plan,
            image_dir=run_dir / "meme_images",
            video_output_dir=video_dir,
            languages=languages,
            zoom_enabled=not args.no_zoom,
        )
        write_json(
            run_dir / "meme_status.json",
            {
                "stage": "render-meme",
                "status": "succeeded",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
                "videos": [{"language": item.language, "path": str(item.path)} for item in rendered],
            },
        )
        print(f"Wrote meme videos to {video_dir}:")
        for item in rendered:
            print(f"  {item.language}: {item.path}")
    except Exception as exc:
        write_json(
            run_dir / "meme_status.json",
            {
                "stage": "render-meme",
                "status": "failed",
                "error": str(exc),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - started, 2),
            },
        )
        print(f"render-meme failed: {exc}")
        return 1
    return 0


def run_meme_all(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is missing. Add it to .env before running meme-run-all.")
        return 2

    run_dir = Path(args.run)
    if not (run_dir / "scripts.json").exists():
        print(f"Missing scripts.json. Run `generate` first: {run_dir}")
        return 1

    model = args.model or os.getenv("GEMINI_MEME_MODEL", DEFAULT_MEME_MODEL)
    languages = parse_csv(args.languages)
    total_started = time.monotonic()

    print(f"[meme-run-all] Starting free-tier meme pipeline for {run_dir.name}...")

    # Stage 1: meme-plan
    scripts = read_json(run_dir / "scripts.json")
    topic = _derive_topic(scripts)
    try:
        print(f"\n[1/4] meme-plan — researching trends and generating plan...")
        result = generate_meme_plan(api_key=api_key, model=model, scripts=scripts, topic=topic)
        write_json(run_dir / "meme_plan.json", result.parsed)
        write_text(run_dir / "raw_meme_plan_response.txt", result.raw_text)
        meme_plan = result.parsed
        print("  Done.")
    except Exception as exc:
        print(f"  meme-plan failed: {exc}")
        return 1

    # Stage 2: imagen
    try:
        print(f"\n[2/4] imagen — generating meme images (free tier, ~6s per image)...")
        imagen_config = load_imagen_config()
        image_result = generate_meme_images(
            meme_plan=meme_plan,
            output_dir=run_dir / "meme_images",
            config=imagen_config,
            languages=languages,
        )
        total_images = sum(len(v) for v in image_result.values())
        print(f"  Generated {total_images} image(s).")
    except Exception as exc:
        print(f"  imagen failed: {exc}")
        return 1

    # Stage 3: meme-tts — Gemini TTS (default) or skip
    if not args.skip_tts:
        try:
            print(f"\n[3/5] meme-tts — generating voiceover with Gemini TTS...")
            tts_config = load_tts_config()
            generate_meme_gemini_tts(
                meme_plan=meme_plan,
                output_dir=run_dir / "meme_audio",
                config=tts_config,
                languages=languages,
            )
            print("  Done.")
        except Exception as exc:
            print(f"  meme-tts failed (non-fatal, continuing without audio): {exc}")
    else:
        print("\n[3/5] meme-tts — skipped (--skip-tts).")

    # Stage 4: BGM via Lyria
    bgm_dir: Path | None = None
    if not args.skip_tts:
        try:
            print(f"\n[4/5] bgm — generating background music with Lyria...")
            bgm_dir = run_dir / "meme_bgm"
            bgm_result = generate_all_bgm(
                output_dir=bgm_dir,
                api_key=api_key,
                languages=languages,
            )
            print(f"  Generated BGM for: {', '.join(bgm_result.keys())}")
        except Exception as exc:
            print(f"  BGM generation failed (non-fatal, continuing without BGM): {exc}")
            bgm_dir = None
    else:
        print("\n[4/5] bgm — skipped (--skip-tts).")

    # Stage 5: render-meme
    try:
        print(f"\n[5/5] render-meme — assembling slideshow videos...")
        video_dir = video_output_dir_for_run(run_dir, Path(args.video_dir))
        rendered = render_meme_slideshow(
            run_dir=run_dir,
            meme_plan=meme_plan,
            image_dir=run_dir / "meme_images",
            video_output_dir=video_dir,
            languages=languages,
            zoom_enabled=not args.no_zoom,
            bgm_dir=bgm_dir,
        )
        write_json(
            run_dir / "meme_status.json",
            {
                "stage": "meme-run-all",
                "status": "succeeded",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - total_started, 2),
                "videos": [{"language": item.language, "path": str(item.path)} for item in rendered],
            },
        )
    except Exception as exc:
        write_json(
            run_dir / "meme_status.json",
            {
                "stage": "meme-run-all",
                "status": "failed",
                "error": str(exc),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": round(time.monotonic() - total_started, 2),
            },
        )
        print(f"  render-meme failed: {exc}")
        return 1

    elapsed = round(time.monotonic() - total_started, 1)
    print(f"\n[meme-run-all] Complete in {elapsed}s:")
    for item in rendered:
        print(f"  {item.language}: {item.path}")
    return 0


def _derive_topic(scripts: dict) -> str:
    """Extract a short topic label from the scripts dict (english title preferred)."""
    try:
        return scripts.get("scripts", {}).get("english", {}).get("title", "pediatric health")
    except Exception:
        return "pediatric health"
