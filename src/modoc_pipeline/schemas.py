"""Pydantic schemas for Gemini structured-output stages."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


LanguageKey = Literal["english", "korean", "spanish"]


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


class VideoPlanMeta(BaseModel):
    aspect_ratio: Literal["9:16"]
    scene_count: int = Field(ge=1, le=8)
    scene_duration_seconds: int = Field(ge=4, le=8)
    chosen_environment: str
    audio_strategy: Literal["gemini_tts"]
    visual_strategy: str


class CharacterBible(BaseModel):
    name: str = "MoDoc Guide"
    style_anchor: str
    character_description: str
    palette: list[str] = Field(default_factory=list)
    background: str
    forbidden_elements: list[str] = Field(default_factory=list)


class SceneContract(BaseModel):
    narration_source: Literal["hook", "body_0", "body_1", "body_2", "safety_caveat", "cta"]
    viewer_should_understand: str
    must_show: list[str] = Field(min_length=1, max_length=5)
    must_not_show: list[str] = Field(default_factory=list)
    meme_device: Literal["gentle_surprise", "myth_vs_fact", "parent_pov", "tiny_visual_joke"]

    @field_validator("viewer_should_understand")
    @classmethod
    def viewer_goal_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("viewer_should_understand is required")
        return value


class SceneBlueprint(BaseModel):
    scene_id: str
    meaning_contract: list[str] = Field(default_factory=list)
    scene_contract: SceneContract
    camera: str
    action: str
    expression: str
    environment_detail: str
    prop: str = Field(default="")

    @field_validator("scene_id")
    @classmethod
    def scene_id_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scene_id is required")
        return value

    @field_validator("camera", "action", "expression", "environment_detail", "prop")
    @classmethod
    def no_placeholders(cls, value: str) -> str:
        if "[" in value or "]" in value:
            raise ValueError("placeholders are not allowed")
        return value


class LocalizedTrack(BaseModel):
    scene_id: str
    language: LanguageKey
    duration_seconds: int = Field(ge=4, le=8)
    caption_text: str
    tts_text: str


class LocalizedTracks(BaseModel):
    english: list[LocalizedTrack]
    korean: list[LocalizedTrack]
    spanish: list[LocalizedTrack]


class VideoPlanPackage(BaseModel):
    video_plan: VideoPlanMeta
    character_bible: CharacterBible
    visual_blueprints: list[SceneBlueprint]
    localized_tracks: LocalizedTracks


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


class MemeScene(BaseModel):
    scene_id: str
    meme_format: MemeFormat
    top_text: str = ""
    bottom_text: str
    image_prompt: str
    duration_seconds: float = Field(default=3.0, ge=2.0, le=6.0)
    tts_text: str = ""

    @field_validator("scene_id")
    @classmethod
    def scene_id_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scene_id is required")
        return value


class MemeLangPlan(BaseModel):
    language: LanguageKey
    cultural_context: str
    trending_hooks: list[str] = Field(min_length=1, max_length=5)
    visual_style_anchor: str  # shared art style + character + palette applied to ALL scenes
    scenes: list[MemeScene] = Field(min_length=3, max_length=5)


class MemePlan(BaseModel):
    english: MemeLangPlan
    korean: MemeLangPlan
    spanish: MemeLangPlan
    source_topic: str
