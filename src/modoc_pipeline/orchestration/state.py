"""Shared state types for the LangGraph production pipeline."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


ReviewDecision = Literal["passed", "failed"]


class AgentStatus(BaseModel):
    agent: str
    status: Literal["started", "succeeded", "failed", "skipped"]
    message: str = ""
    duration_seconds: float | None = None


class ReviewIssue(BaseModel):
    severity: Literal["low", "medium", "high"] = "medium"
    issue: str
    repair_instruction: str = ""


class PipelineState(TypedDict, total=False):
    row_number: int
    input_path: str
    output_dir: str
    log_path: str
    video_dir: str
    languages: list[str] | None
    api_key: str
    model: str
    enable_search: bool
    skip_tts: bool
    no_zoom: bool
    total_started: float

    source: dict[str, Any]
    run_id: str
    run_dir: str
    topic: str

    grounding_report: dict[str, Any]
    grounding_raw_text: str
    scripts: dict[str, Any]
    claims: list[dict[str, Any]]
    script_raw_text: str
    trend_research: dict[str, Any]
    creative_candidates: dict[str, Any]
    creative_scores: dict[str, Any]
    meme_plan: dict[str, Any]
    meme_plan_raw_text: str
    image_paths: dict[str, dict[str, str]]
    audio_paths: dict[str, dict[str, str]]
    bgm_paths: dict[str, str]
    rendered_videos: list[dict[str, str]]

    failure_status: str
    failure_message: str
    agent_trace: list[dict[str, Any]]


class ReviewReport(BaseModel):
    status: ReviewDecision
    stage: str
    summary: str = ""
    issues: list[ReviewIssue] = Field(default_factory=list)
    repair_instruction: str = ""
