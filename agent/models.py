from __future__ import annotations
from dataclasses import dataclass
from pydantic import BaseModel, field_validator, model_validator


class CurrentRole(BaseModel):
    title: str
    employer: str
    tenure_years: float


class MobilitySignal(BaseModel):
    score: int
    rationale: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: int) -> int:
        # Clamp rather than reject — a 0 from the model means "insufficient data",
        # which maps cleanly to 1 (lowest valid signal). Prevents a costly retry.
        return max(1, min(v, 5))


class RoleFit(BaseModel):
    role: str  # populated dynamically from the role spec — no default
    score: int
    justification: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: int) -> int:
        # Same clamping rationale as MobilitySignal.
        return max(1, min(v, 10))


class CandidateBriefing(BaseModel):
    candidate_id: str
    full_name: str
    current_role: CurrentRole
    career_narrative: str
    experience_tags: list[str]
    firm_aum_context: str
    mobility_signal: MobilitySignal
    role_fit: RoleFit
    outreach_hook: str

    @field_validator("experience_tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        if len(v) < 3:
            raise ValueError("experience_tags must contain at least 3 items")
        return v

    @field_validator("outreach_hook")
    @classmethod
    def validate_hook(cls, v: str) -> str:
        cleaned = v.strip()
        if "\n" in cleaned:
            raise ValueError("outreach_hook must be a single sentence with no line breaks")
        return cleaned

    @model_validator(mode="after")
    def validate_narrative_length(self) -> CandidateBriefing:
        sentences = [s.strip() for s in self.career_narrative.split(".") if s.strip()]
        if len(sentences) < 2:
            raise ValueError("career_narrative must contain at least 3-4 sentences")
        return self


class OutputSchema(BaseModel):
    candidates: list[CandidateBriefing]


@dataclass
class CandidateInput:
    """Raw data from CSV — not validated beyond presence."""
    full_name: str
    current_employer: str
    current_title: str
    linkedin_url: str
    candidate_id: str  # assigned at parse time: "candidate_1" etc.
