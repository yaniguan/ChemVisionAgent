"""Structured audit result schemas."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class SkillScore(BaseModel):
    """Score for a single capability skill."""

    skill_name: str
    accuracy: float = Field(ge=0.0, le=1.0)
    num_samples: int = Field(ge=0)
    notes: str = ""


class AuditReport(BaseModel):
    """Full audit report aggregating scores across all probed skills."""

    model_name: str
    run_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    skill_scores: list[SkillScore] = Field(default_factory=list)
    overall_accuracy: float = Field(0.0, ge=0.0, le=1.0)
    metadata: dict[str, object] = Field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        n = len(self.skill_scores)
        return (
            f"AuditReport | model={self.model_name} | "
            f"skills={n} | overall={self.overall_accuracy:.2%}"
        )
