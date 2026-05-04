from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Finding(BaseModel):
    title: str = Field(..., description="Short title of the finding")
    insight: str = Field(..., description="Specific market insight backed by evidence")
    evidence: list[str] = Field(default_factory=list, description="Evidence snippets or cited facts")
    risk_level: str | None = Field(default=None, description="Optional risk level: low/medium/high")


class DraftReport(BaseModel):
    executive_summary: str
    findings: list[Finding]
    sources: list[str]
    data_points: list[str]


class CriticFeedback(BaseModel):
    verdict: Literal["APPROVED", "NEEDS_REVISION"]
    issues: list[str]
    missing_perspectives: list[str]
    fact_check_results: list[str]
    score: float = Field(..., ge=0.0, le=1.0)


class FinalReport(BaseModel):
    executive_summary: str
    key_findings: list[str]
    recommendations: list[str]
    sources: list[str]
    methodology: str