"""Structured log entry for a single skill invocation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ToolCallLog(BaseModel):
    """Record of a single skill/tool invocation by the agent."""

    skill_name: str = Field(..., description="Name of the skill that was called.")
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments passed to the skill (image paths serialised as strings).",
    )
    output_summary: str = Field(
        ..., description="Human-readable summary of the skill output."
    )
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence score returned by the skill (0–1)."
    )
    low_confidence: bool = Field(
        False,
        description="True when confidence is below the agent's configured threshold.",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_output: str = Field("", description="Raw text returned by the vision model.")
