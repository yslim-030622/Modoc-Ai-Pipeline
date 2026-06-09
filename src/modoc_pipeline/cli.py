"""Command-line interface for the MoDoc pipeline MVP."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

from .artifacts import timed_stage
from .io_utils import read_json, write_json, write_text
from .tts_client import generate_meme_gemini_tts, load_tts_config
from .meme_planner import generate_meme_plan
from .imagen_client import generate_meme_images, load_imagen_config
from .orchestration.graph import build_pipeline_graph
from .renderer import render_meme_slideshow
from .bgm_client import generate_all_bgm
from .dashboard import serve_dashboard


DEFAULT_INPUT = "Q&A Blog Contents List.xlsx"
DEFAULT_OUTPUT_DIR = "logs"
DEFAULT_VIDEO_DIR = "videos"
DEFAULT_LOG_PATH = "logs/pipeline_runs.csv"
DEFAULT_MODEL = "gemini-3.5-flash"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        return run_generate(args)
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
    if args.command == "dashboard":
        return run_dashboard(args)

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
        help="Generate meme videos from one or more Q&A rows. E.g.: generate 24  or  generate 23 12 25",
    )
    generate.add_argument("rows", nargs="+", type=int, help="Row number(s) to generate.")
    generate.add_argument("--input", default=DEFAULT_INPUT, help="Path to the Q&A Excel workbook.")
    generate.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for JSON run logs.")
    generate.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="CSV path for KPI timing logs.")
    generate.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR, help="Directory for final videos.")
    generate.add_argument("--languages", help="Comma-separated languages, e.g. english,korean.")
    generate.add_argument("--skip-tts", action="store_true", help="Skip TTS (produce silent videos).")
    generate.add_argument("--no-zoom", action="store_true", help="Disable Ken Burns zoom effect.")
    generate.add_argument("--model", default=None, help="Gemini model. Defaults to GEMINI_MODEL env var.")
    generate.add_argument("--skip-search", action="store_true", help="Skip Gemini Google Search grounding.")
    generate.add_argument(
        "--campaign-profile",
        default=None,
        help="Path to a campaign profile JSON for meme planning. Defaults to MODOC_CAMPAIGN_PROFILE.",
    )

    # ── Meme slideshow pipeline (individual stages for power users) ──────────

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
    meme_plan_parser.add_argument(
        "--campaign-profile",
        default=None,
        help="Path to a campaign profile JSON. Defaults to MODOC_CAMPAIGN_PROFILE.",
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
    meme_all_parser.add_argument(
        "--campaign-profile",
        default=None,
        help="Path to a campaign profile JSON. Defaults to MODOC_CAMPAIGN_PROFILE.",
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Serve a localhost dashboard for live agent pipeline status.",
    )
    dashboard_parser.add_argument("--host", default="localhost", help="Dashboard bind host.")
    dashboard_parser.add_argument("--port", type=int, default=8765, help="Dashboard port.")
    dashboard_parser.add_argument("--logs-dir", default=DEFAULT_OUTPUT_DIR, help="Logs directory to watch.")

    return parser





def run_generate(args: argparse.Namespace) -> int:
    """Full meme pipeline for one or more row numbers.

    Usage: generate 24          → row 24
           generate 23 12 25    → rows 23, 12, 25
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is missing. Add it to .env before running generation.")
        return 2

    model = args.model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    log_path = Path(args.log_path)
    languages = parse_csv(args.languages)
    graph = build_pipeline_graph()

    failures = 0
    for row_number in args.rows:
        print(f"\n{'='*60}")
        print(f"[generate] Row {row_number}")
        print(f"{'='*60}")
        total_started = time.monotonic()
        print("\n[agentic] Running LangGraph production pipeline...")
        initial_state = {
            "row_number": row_number,
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "log_path": str(log_path),
            "video_dir": str(Path(args.video_dir)),
            "languages": sorted(languages) if languages else None,
            "api_key": api_key,
            "model": model,
            "enable_search": not args.skip_search and _env_bool("GEMINI_ENABLE_SEARCH", default=True),
            "skip_tts": args.skip_tts,
            "no_zoom": args.no_zoom,
            "campaign_profile_path": args.campaign_profile or os.getenv("MODOC_CAMPAIGN_PROFILE", "").strip() or None,
            "total_started": total_started,
            "agent_trace": [],
        }
        final_state = graph.invoke(initial_state)
        if final_state.get("failure_status"):
            failures += 1
            print(f"  Failed closed: {final_state.get('failure_message', 'unknown error')}")
            continue

        total_elapsed = round(time.monotonic() - total_started, 1)
        output_path = video_output_dir_for_run(Path(final_state["run_dir"]), Path(args.video_dir))
        print(f"\n[generate] Row {row_number} complete in {total_elapsed}s → {output_path}")

    if failures:
        print(f"\nCompleted with {failures} failure(s).")
        return 1
    return 0


def run_dashboard(args: argparse.Namespace) -> int:
    serve_dashboard(host=args.host, port=args.port, logs_dir=Path(args.logs_dir))
    return 0




def _run_meme_for_run_dir(
    *,
    run_dir: Path,
    api_key: str,
    model: str,
    languages: set[str] | None,
    skip_tts: bool,
    no_zoom: bool,
    video_dir: Path,
    campaign_profile_path: str | None = None,
) -> bool:
    """Run meme-plan → imagen → tts → bgm → render for an existing run_dir.

    Returns True on success, False on failure.
    """
    scripts_path = run_dir / "scripts.json"
    if not scripts_path.exists():
        print(f"  Missing scripts.json in {run_dir}")
        return False

    meme_model = model or os.getenv("GEMINI_MEME_MODEL", DEFAULT_MEME_MODEL)
    scripts = read_json(scripts_path)
    topic = _derive_topic(scripts)

    # Stage 2: meme-plan
    print(f"\n[2/5] meme-plan — researching trends and generating plan...")
    try:
        with timed_stage(run_dir, "meme_plan"):
            result = generate_meme_plan(
                api_key=api_key,
                model=meme_model,
                scripts=scripts,
                topic=topic,
                campaign_profile_path=campaign_profile_path or os.getenv("MODOC_CAMPAIGN_PROFILE", "").strip() or None,
            )
            write_json(run_dir / "meme_plan.json", result.parsed)
            write_json(run_dir / "trend_research.json", result.trend_research)
            write_json(run_dir / "visual_brief.json", result.visual_brief)
            write_json(run_dir / "creative_candidates.json", result.creative_candidates)
            write_json(run_dir / "creative_scores.json", result.creative_scores)
            write_text(run_dir / "raw_meme_plan_response.txt", result.raw_text)
            meme_plan = result.parsed
        print("  Done.")
    except Exception as exc:
        print(f"  meme-plan failed: {exc}")
        return False

    # Stage 3: imagen
    print(f"\n[3/5] imagen — generating meme images...")
    try:
        with timed_stage(run_dir, "imagen"):
            imagen_config = load_imagen_config()
            image_result = generate_meme_images(
                meme_plan=meme_plan,
                output_dir=run_dir / "meme_images",
                config=imagen_config,
                languages=languages,
            )
        print(f"  Generated {sum(len(v) for v in image_result.values())} image(s).")
    except Exception as exc:
        print(f"  imagen failed: {exc}")
        return False

    # Stage 4: TTS + BGM
    bgm_dir: Path | None = None
    if not skip_tts:
        print(f"\n[4/5] tts + bgm — generating voiceover and background music...")
        try:
            with timed_stage(run_dir, "tts"):
                tts_config = load_tts_config()
                generate_meme_gemini_tts(
                    meme_plan=meme_plan,
                    output_dir=run_dir / "meme_audio",
                    config=tts_config,
                    languages=languages,
                )
        except Exception as exc:
            print(f"  TTS failed (non-fatal, continuing without audio): {exc}")

        try:
            with timed_stage(run_dir, "bgm"):
                bgm_dir = run_dir / "meme_bgm"
                bgm_result = generate_all_bgm(output_dir=bgm_dir, api_key=api_key, meme_plan=meme_plan, languages=languages)
            print(f"  BGM generated for: {', '.join(bgm_result.keys())}")
        except Exception as exc:
            print(f"  BGM failed (non-fatal, continuing without BGM): {exc}")
            bgm_dir = None
    else:
        print("\n[4/5] tts + bgm — skipped (--skip-tts).")

    # Stage 5: render
    print(f"\n[5/5] render — assembling slideshow videos...")
    try:
        with timed_stage(run_dir, "render"):
            out_dir = video_output_dir_for_run(run_dir, video_dir)
            rendered = render_meme_slideshow(
                run_dir=run_dir,
                meme_plan=meme_plan,
                image_dir=run_dir / "meme_images",
                video_output_dir=out_dir,
                languages=languages,
                zoom_enabled=not no_zoom,
                bgm_dir=bgm_dir,
            )
        write_json(run_dir / "meme_status.json", {
            "stage": "generate",
            "status": "succeeded",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "videos": [{"language": v.language, "path": str(v.path)} for v in rendered],
        })
        for v in rendered:
            print(f"  {v.language}: {v.path}")
        return True
    except Exception as exc:
        write_json(run_dir / "meme_status.json", {
            "stage": "generate",
            "status": "failed",
            "error": str(exc),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"  render failed: {exc}")
        return False


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
        result = generate_meme_plan(
            api_key=api_key,
            model=model,
            scripts=scripts,
            topic=topic,
            campaign_profile_path=args.campaign_profile or os.getenv("MODOC_CAMPAIGN_PROFILE", "").strip() or None,
        )
        write_json(run_dir / "meme_plan.json", result.parsed)
        write_json(run_dir / "trend_research.json", result.trend_research)
        write_json(run_dir / "visual_brief.json", result.visual_brief)
        write_json(run_dir / "creative_candidates.json", result.creative_candidates)
        write_json(run_dir / "creative_scores.json", result.creative_scores)
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
        bgm_dir = run_dir / "meme_bgm"
        bgm_dir = bgm_dir if bgm_dir.exists() and any(bgm_dir.iterdir()) else None
        print("Rendering meme slideshow videos...")
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
        result = generate_meme_plan(
            api_key=api_key,
            model=model,
            scripts=scripts,
            topic=topic,
            campaign_profile_path=args.campaign_profile or os.getenv("MODOC_CAMPAIGN_PROFILE", "").strip() or None,
        )
        write_json(run_dir / "meme_plan.json", result.parsed)
        write_json(run_dir / "trend_research.json", result.trend_research)
        write_json(run_dir / "visual_brief.json", result.visual_brief)
        write_json(run_dir / "creative_candidates.json", result.creative_candidates)
        write_json(run_dir / "creative_scores.json", result.creative_scores)
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
                meme_plan=meme_plan,
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


def _derive_topic(scripts: dict) -> str:
    """Extract a short topic label from the scripts dict (english title preferred)."""
    try:
        return scripts.get("scripts", {}).get("english", {}).get("title", "pediatric health")
    except Exception:
        return "pediatric health"
