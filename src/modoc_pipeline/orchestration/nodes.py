"""LangGraph node implementations for the production pipeline."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..artifacts import (
    HumanTiming,
    append_timing_log,
    finalize_run_timing,
    make_run_id,
    timed_stage,
    write_failure_artifacts,
    write_success_artifacts,
)
from ..bgm_client import generate_all_bgm
from ..excel_source import QnaSource, load_qna_sources
from ..gemini_client import generate_short_form_package
from ..grounding import generate_grounding_report
from ..imagen_client import generate_meme_images, load_imagen_config
from ..io_utils import write_json, write_text
from ..meme_planner import generate_meme_plan
from ..renderer import render_meme_slideshow
from ..reviewers import (
    assert_bgm_complete,
    assert_meme_plan_quality,
    assert_tts_complete,
    build_meme_plan_quality_reports,
)
from ..tts_client import generate_meme_gemini_tts, load_tts_config
from .state import PipelineState


def load_source_node(state: PipelineState) -> PipelineState:
    started = time.monotonic()
    try:
        sources = load_qna_sources(
            Path(state["input_path"]),
            status_filter="any",
            row_number=state["row_number"],
            limit=1,
        )
        source = sources[0]
        run_id = make_run_id(source)
        run_dir = str(Path(state["output_dir"]) / run_id)
        updated = dict(state)
        updated.update(
            {
                "source": source.to_dict(),
                "run_id": run_id,
                "run_dir": run_dir,
            }
        )
        return _trace(updated, "load_source", "succeeded", f"row {source.row_number}", started)
    except Exception as exc:
        return _fail(state, "load_source", exc, started)


def grounding_agent_node(state: PipelineState) -> PipelineState:
    started = time.monotonic()
    try:
        source = _source_from_state(state)
        with timed_stage(Path(state["run_dir"]), "grounding"):
            result = generate_grounding_report(
                api_key=state["api_key"],
                model=state["model"],
                source=source,
                enable_search=state["enable_search"],
            )
        write_json(Path(state["run_dir"]) / "grounding_report.json", result.parsed)
        write_text(Path(state["run_dir"]) / "raw_grounding_response.txt", result.raw_text)
        updated = dict(state)
        updated.update({"grounding_report": result.parsed, "grounding_raw_text": result.raw_text})
        return _trace(updated, "grounding_agent", "succeeded", result.parsed.get("status", ""), started)
    except Exception as exc:
        return _fail(state, "grounding_agent", exc, started)


def script_writer_agent_node(state: PipelineState) -> PipelineState:
    started = time.monotonic()
    try:
        source = _source_from_state(state)
        with timed_stage(Path(state["run_dir"]), "script_writer"):
            result = generate_short_form_package(
                api_key=state["api_key"],
                model=state["model"],
                source=source,
                grounding_report=state.get("grounding_report", {}),
            )
        updated = dict(state)
        updated.update(
            {
                "scripts": result.parsed.get("scripts", {}),
                "claims": result.parsed.get("medical_claims", []),
                "script_raw_text": result.raw_text,
            }
        )
        return _trace(updated, "script_writer_agent", "succeeded", "scripts generated", started)
    except Exception as exc:
        return _fail(state, "script_writer_agent", exc, started)


def persist_script_artifacts_node(state: PipelineState) -> PipelineState:
    started = time.monotonic()
    try:
        source = _source_from_state(state)
        generation = {
            "scripts": state.get("scripts", {}),
            "medical_claims": state.get("claims", []),
            "reviewer_notes": ["Clinician review required before publication."],
        }
        write_success_artifacts(
            run_dir=Path(state["run_dir"]),
            source=source,
            generation=generation,
            raw_text=state.get("script_raw_text", ""),
            status={
                "run_id": state["run_id"],
                "status": "scripts_generated",
                "source_row": source.row_number,
                "model": state["model"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            grounding_report=state.get("grounding_report", {}),
            quality_reports={},
        )
        append_timing_log(
            log_path=Path(state["log_path"]),
            run_id=state["run_id"],
            source_row=source.row_number,
            timing=HumanTiming(),
        )
        _write_agent_artifacts(state)
        return _trace(state, "persist_script_artifacts", "succeeded", "script artifacts written", started)
    except Exception as exc:
        return _fail(state, "persist_script_artifacts", exc, started)


def meme_planner_agent_node(state: PipelineState) -> PipelineState:
    started = time.monotonic()
    try:
        topic = _derive_topic(state.get("scripts", {}))
        with timed_stage(Path(state["run_dir"]), "meme_plan"):
            result = generate_meme_plan(
                api_key=state["api_key"],
                model=state["model"],
                scripts=state.get("scripts", {}),
                topic=topic,
                campaign_profile_path=state.get("campaign_profile_path"),
            )
        write_json(Path(state["run_dir"]) / "meme_plan.json", result.parsed)
        write_json(Path(state["run_dir"]) / "trend_research.json", result.trend_research)
        write_json(Path(state["run_dir"]) / "trend_sources.json", result.trend_sources)
        write_json(Path(state["run_dir"]) / "visual_brief.json", result.visual_brief)
        write_json(Path(state["run_dir"]) / "creative_candidates.json", result.creative_candidates)
        write_json(Path(state["run_dir"]) / "creative_scores.json", result.creative_scores)
        write_text(Path(state["run_dir"]) / "raw_meme_plan_response.txt", result.raw_text)
        quality_reports = build_meme_plan_quality_reports(
            meme_plan=result.parsed,
            creative_scores=result.creative_scores,
            scripts=state.get("scripts", {}),
        )
        write_json(Path(state["run_dir"]) / "meme_plan_quality_reports.json", quality_reports)
        assert_meme_plan_quality(quality_reports)
        updated = dict(state)
        updated.update(
            {
                "meme_plan": result.parsed,
                "meme_plan_raw_text": result.raw_text,
                "trend_research": result.trend_research,
                "trend_sources": result.trend_sources,
                "visual_brief": result.visual_brief,
                "creative_candidates": result.creative_candidates,
                "creative_scores": result.creative_scores,
                "meme_plan_quality_reports": quality_reports,
                "topic": topic,
            }
        )
        return _trace(updated, "meme_planner_agent", "succeeded", topic, started)
    except Exception as exc:
        return _fail(state, "meme_planner_agent", exc, started)


def image_generation_agent_node(state: PipelineState) -> PipelineState:
    started = time.monotonic()
    try:
        with timed_stage(Path(state["run_dir"]), "imagen"):
            image_result = generate_meme_images(
                meme_plan=state.get("meme_plan", {}),
                output_dir=Path(state["run_dir"]) / "meme_images",
                config=load_imagen_config(),
                languages=_languages_set(state),
            )
        image_paths = {lang: {sid: str(path) for sid, path in scenes.items()} for lang, scenes in image_result.items()}
        updated = dict(state)
        updated["image_paths"] = image_paths
        return _trace(updated, "image_generation_agent", "succeeded", f"{sum(len(v) for v in image_paths.values())} images", started)
    except Exception as exc:
        return _fail(state, "image_generation_agent", exc, started)


def tts_agent_node(state: PipelineState) -> PipelineState:
    started = time.monotonic()
    if state.get("skip_tts"):
        updated = dict(state)
        updated["audio_paths"] = {}
        return _trace(updated, "tts_agent", "skipped", "--skip-tts", started)
    try:
        with timed_stage(Path(state["run_dir"]), "tts"):
            result = generate_meme_gemini_tts(
                meme_plan=state.get("meme_plan", {}),
                output_dir=Path(state["run_dir"]) / "meme_audio",
                config=load_tts_config(),
                languages=_languages_set(state),
            )
        audio_paths = {lang: {sid: str(path) for sid, path in scenes.items()} for lang, scenes in result.items()}
        assert_tts_complete(
            audio_paths=audio_paths,
            meme_plan=state.get("meme_plan", {}),
            languages=_languages_set(state),
        )
        updated = dict(state)
        updated["audio_paths"] = audio_paths
        return _trace(updated, "tts_agent", "succeeded", f"{sum(len(v) for v in audio_paths.values())} audio files", started)
    except Exception as exc:
        return _fail(state, "tts_agent", exc, started)


def bgm_agent_node(state: PipelineState) -> PipelineState:
    started = time.monotonic()
    if state.get("skip_tts"):
        updated = dict(state)
        updated["bgm_paths"] = {}
        return _trace(updated, "bgm_agent", "skipped", "--skip-tts", started)
    try:
        with timed_stage(Path(state["run_dir"]), "bgm"):
            result = generate_all_bgm(
                output_dir=Path(state["run_dir"]) / "meme_bgm",
                api_key=state["api_key"],
                meme_plan=state.get("meme_plan", {}),
                languages=_languages_set(state),
            )
        bgm_paths = {lang: str(path) for lang, path in result.items()}
        assert_bgm_complete(
            bgm_paths=bgm_paths,
            meme_plan=state.get("meme_plan", {}),
            languages=_languages_set(state),
        )
        updated = dict(state)
        updated["bgm_paths"] = bgm_paths
        return _trace(updated, "bgm_agent", "succeeded", ", ".join(bgm_paths) or "no bgm", started)
    except Exception as exc:
        return _fail(state, "bgm_agent", exc, started)


def render_agent_node(state: PipelineState) -> PipelineState:
    started = time.monotonic()
    try:
        with timed_stage(Path(state["run_dir"]), "render"):
            rendered = render_meme_slideshow(
                run_dir=Path(state["run_dir"]),
                meme_plan=state.get("meme_plan", {}),
                image_dir=Path(state["run_dir"]) / "meme_images",
                video_output_dir=_video_output_dir_for_run(Path(state["run_dir"]), Path(state["video_dir"])),
                languages=_languages_set(state),
                zoom_enabled=not state.get("no_zoom", False),
                bgm_dir=Path(state["run_dir"]) / "meme_bgm" if state.get("bgm_paths") else None,
            )
        videos = [{"language": item.language, "path": str(item.path)} for item in rendered]
        updated = dict(state)
        updated["rendered_videos"] = videos
        return _trace(updated, "render_agent", "succeeded", f"{len(videos)} videos", started)
    except Exception as exc:
        return _fail(state, "render_agent", exc, started)


def finalize_artifacts_node(state: PipelineState) -> PipelineState:
    started = time.monotonic()
    try:
        source = _source_from_state(state)
        total_elapsed = round(time.monotonic() - float(state.get("total_started", started)), 1)
        write_json(
            Path(state["run_dir"]) / "meme_status.json",
            {
                "stage": "generate",
                "status": "succeeded",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "videos": state.get("rendered_videos", []),
            },
        )
        write_json(
            Path(state["run_dir"]) / "status.json",
            {
                "run_id": state["run_id"],
                "status": "succeeded",
                "source_row": source.row_number,
                "model": state["model"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "automated_generation_seconds": total_elapsed,
            },
        )
        finalize_run_timing(
            Path(state["run_dir"]),
            source_row=source.row_number,
            total_seconds=total_elapsed,
            logs_dir=Path(state["output_dir"]),
        )
        _write_agent_artifacts(state)
        return _trace(state, "finalize_artifacts", "succeeded", f"{total_elapsed}s", started)
    except Exception as exc:
        return _fail(state, "finalize_artifacts", exc, started)


def fail_closed_node(state: PipelineState) -> PipelineState:
    started = time.monotonic()
    updated = dict(state)
    updated.setdefault("failure_status", "failed")
    updated.setdefault("failure_message", "Pipeline failed.")
    try:
        if "source" in updated and "run_dir" in updated:
            source = _source_from_state(updated)
            write_failure_artifacts(
                run_dir=Path(updated["run_dir"]),
                source=source,
                raw_text=updated.get("script_raw_text", "") or updated.get("meme_plan_raw_text", ""),
                status={
                    "run_id": updated.get("run_id", ""),
                    "status": updated["failure_status"],
                    "source_row": source.row_number,
                    "error": updated["failure_message"],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            _write_agent_artifacts(updated)
    finally:
        return _trace(updated, "fail_closed", "failed", updated.get("failure_message", ""), started)


def route_after_failure(state: PipelineState) -> str:
    return "fail_closed" if state.get("failure_status") else "continue"


def _fail(state: PipelineState, agent: str, exc: Exception, started: float) -> PipelineState:
    updated = dict(state)
    updated["failure_status"] = updated.get("failure_status") or "failed"
    updated["failure_message"] = str(exc)
    return _trace(updated, agent, "failed", str(exc), started)


def _trace(state: PipelineState, agent: str, status: str, message: str, started: float) -> PipelineState:
    updated = dict(state)
    trace = list(updated.get("agent_trace", []))
    trace.append(
        {
            "agent": agent,
            "status": status,
            "message": message,
            "duration_seconds": round(time.monotonic() - started, 2),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    updated["agent_trace"] = trace
    if updated.get("run_dir"):
        _write_agent_artifacts(updated)
    return updated


def _source_from_state(state: PipelineState) -> QnaSource:
    return QnaSource(**state["source"])


def _languages_set(state: PipelineState) -> set[str] | None:
    languages = state.get("languages")
    return set(languages) if languages else None


def _derive_topic(scripts: dict[str, Any]) -> str:
    try:
        return scripts.get("english", {}).get("title") or "pediatric health"
    except Exception:
        return "pediatric health"


def _write_agent_artifacts(state: PipelineState) -> None:
    if not state.get("run_dir"):
        return
    run_dir = Path(state["run_dir"])
    write_json(run_dir / "agent_trace.json", state.get("agent_trace", []))
    write_json(run_dir / "pipeline_state.json", _sanitize_state(state))


def _sanitize_state(state: PipelineState) -> dict[str, Any]:
    omitted = {"api_key", "total_started"}
    return {key: value for key, value in state.items() if key not in omitted}


def _video_output_dir_for_run(run_dir: Path, video_root: Path) -> Path:
    source_path = run_dir / "source.json"
    if source_path.exists():
        try:
            import json

            source = json.loads(source_path.read_text(encoding="utf-8"))
            row_number = source.get("row_number")
            if row_number:
                return video_root / f"Row_{row_number}"
        except Exception:
            pass
    return video_root / run_dir.name
