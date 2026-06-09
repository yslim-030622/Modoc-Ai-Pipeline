"""Pydantic schemas for Gemini structured-output stages."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


LanguageKey = Literal["english", "korean", "spanish"]
CaptionRole = Literal["hook", "tension", "insight", "relief"]
CaptionStyle = Literal["impact", "clean_reels", "korean_jjal", "spanish_social"]
ShotType = Literal["close_up", "medium_action", "wide_scene", "over_the_shoulder", "tabletop_action", "conversation"]
ContentType = Literal[
    "fever_triage",
    "medication_dosing",
    "vaccine_reaction",
    "gi_hydration",
    "puberty_period",
    "rash_skin",
    "lab_result",
    "development_behavior",
    "injury_urgent",
    "newborn_feeding_sleep",
    "general",
]


class GroundingCitation(BaseModel):
    uri: str = Field(default="", description="Source URL returned by Google Search grounding.")
    title: str = Field(default="", description="Short source title.")


class GroundedMedicalFact(BaseModel):
    fact: str
    source_indices: list[int] = Field(default_factory=list)
    source: str = Field(default="expert_answer", description="expert_answer or google_search")


class GroundingReport(BaseModel):
    status: Literal["grounded", "expert_only", "insufficient"]
    search_queries: list[str] = Field(default_factory=list)
    citations: list[GroundingCitation] = Field(default_factory=list)
    supported_facts: list[GroundedMedicalFact] = Field(default_factory=list)
    unsupported_or_unsafe_claims: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Script(BaseModel):
    language: str
    title: str
    hook: str
    body: list[str] = Field(min_length=1, max_length=3)
    safety_caveat: str
    cta: str
    estimated_seconds: int = Field(ge=1, le=35)


class Scripts(BaseModel):
    english: Script
    korean: Script
    spanish: Script


class MedicalClaim(BaseModel):
    claim: str
    evidence_from_expert_answer: str = ""
    grounding_fact_indices: list[int] = Field(default_factory=list)
    citation_indices: list[int] = Field(default_factory=list)
    appears_in_languages: list[LanguageKey]
    risk_level: Literal["low", "medium", "high"]


class ScriptPackage(BaseModel):
    scripts: Scripts
    medical_claims: list[MedicalClaim]
    editor_notes: list[str] = Field(default_factory=list)
    reviewer_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def claims_must_have_grounding(self) -> "ScriptPackage":
        for claim in self.medical_claims:
            if not claim.evidence_from_expert_answer.strip() and not claim.grounding_fact_indices:
                raise ValueError(f"medical claim lacks evidence: {claim.claim}")
        return self



class QualityIssue(BaseModel):
    stage: str
    severity: Literal["low", "medium", "high"]
    issue: str
    repair_instruction: str = ""


class QualityReport(BaseModel):
    status: Literal["passed", "failed"]
    stage: str
    issues: list[QualityIssue] = Field(default_factory=list)
    summary: str = ""
    human_review_checklist: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Meme slideshow pipeline schemas
# ─────────────────────────────────────────────────────────────────────────────

class MemeFormat(str, Enum):
    POV = "pov"
    BEFORE_AFTER = "before_after"
    REACTION = "reaction"
    JJAL = "jjal"
    CAPTION_ONLY = "caption_only"
    RELATABILITY = "relatability"


class BgmConfigSpec(BaseModel):
    bpm: int = Field(default=110, ge=60, le=200)
    density: float = Field(default=0.72, ge=0.0, le=1.0)
    brightness: float = Field(default=0.75, ge=0.0, le=1.0)
    guidance: float = Field(default=4.0, ge=0.0, le=6.0)
    temperature: float = Field(default=1.0, ge=0.0, le=3.0)


class MemeScene(BaseModel):
    scene_id: str
    meme_format: MemeFormat
    caption_role: CaptionRole = "hook"
    medical_message: str = ""
    scene_visual_action: str = ""
    safe_props: list[str] = Field(default_factory=list)
    shot_type: ShotType = "medium_action"
    primary_prop: str = ""
    background: str = ""
    top_text: str = ""
    bottom_text: str
    image_prompt: str
    duration_seconds: float = Field(default=3.0, ge=1.0, le=10.0)

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def clamp_duration(cls, v: Any) -> float:
        return max(1.0, min(10.0, float(v)))
    tts_text: str = ""

    @field_validator("scene_id")
    @classmethod
    def scene_id_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scene_id is required")
        return value


class ContentVisualBrief(BaseModel):
    content_type: ContentType = "general"
    care_context: str = ""
    target_subject: str = ""
    risk_level: Literal["low", "medium", "high"] = "low"
    allowed_props: list[str] = Field(default_factory=list, max_length=8)
    forbidden_visuals: list[str] = Field(default_factory=list, max_length=12)
    must_show: list[str] = Field(default_factory=list, max_length=6)
    must_not_show: list[str] = Field(default_factory=list, max_length=8)
    visual_tone: str = ""


class MemeLangPlan(BaseModel):
    language: LanguageKey
    cultural_context: str
    trending_hooks: list[str] = Field(min_length=1, max_length=5)
    creative_angle: str = ""
    trend_rationale: str = ""
    caption_style: CaptionStyle = "impact"
    tts_style: str = ""
    bgm_prompt: str = ""
    bgm_config: BgmConfigSpec = Field(default_factory=BgmConfigSpec)
    avoid_cliches: list[str] = Field(default_factory=list)
    visual_style_anchor: str  # shared art style + character + palette applied to ALL scenes
    character_sheet: str  # locked character description: exact hair, clothing, skin tone, body shape — repeated verbatim in every scene prompt
    scenes: list[MemeScene] = Field(min_length=3, max_length=5)


class MemePlan(BaseModel):
    english: MemeLangPlan
    korean: MemeLangPlan
    spanish: MemeLangPlan
    source_topic: str
    visual_brief: ContentVisualBrief = Field(default_factory=ContentVisualBrief)

    @model_validator(mode="before")
    @classmethod
    def inject_language_keys(cls, data: Any) -> Any:
        for key in ("english", "korean", "spanish"):
            if isinstance(data.get(key), dict) and "language" not in data[key]:
                data[key] = {**data[key], "language": key}
        return data


class TrendFormat(BaseModel):
    format_name: str
    platform: str
    hook_templates: list[str] = Field(default_factory=list)
    caption_style: str = ""
    visual_editing_style: str = ""
    audio_mood: str = ""
    why_it_works: str = ""
    avoid: list[str] = Field(default_factory=list)


class TrendResearch(BaseModel):
    english: list[TrendFormat] = Field(default_factory=list)
    korean: list[TrendFormat] = Field(default_factory=list)
    spanish: list[TrendFormat] = Field(default_factory=list)


class CreativeCandidate(BaseModel):
    candidate_id: str
    language: LanguageKey
    creative_angle: str
    trend_rationale: str
    hook_template: str
    caption_style: CaptionStyle
    visual_style_anchor: str
    character_sheet: str
    tts_style: str
    bgm_prompt: str
    bgm_config: BgmConfigSpec = Field(default_factory=BgmConfigSpec)
    avoid_cliches: list[str] = Field(default_factory=list)
    scene_beats: list[str] = Field(min_length=4, max_length=4)


class CreativeCandidateSet(BaseModel):
    english: list[CreativeCandidate] = Field(min_length=3, max_length=3)
    korean: list[CreativeCandidate] = Field(min_length=3, max_length=3)
    spanish: list[CreativeCandidate] = Field(min_length=3, max_length=3)


class CreativeScore(BaseModel):
    candidate_id: str
    language: LanguageKey
    hook_strength: int = Field(ge=1, le=5)
    native_fit: int = Field(ge=1, le=5)
    retention: int = Field(ge=1, le=5)
    medical_safety: int = Field(ge=1, le=5)
    not_cringe: int = Field(ge=1, le=5)
    visual_feasibility: int = Field(ge=1, le=5)
    selected: bool = False
    rationale: str = ""


class CreativeScoreSet(BaseModel):
    english: list[CreativeScore] = Field(min_length=3, max_length=3)
    korean: list[CreativeScore] = Field(min_length=3, max_length=3)
    spanish: list[CreativeScore] = Field(min_length=3, max_length=3)
