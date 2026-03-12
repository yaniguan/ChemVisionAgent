"""Data structures for recording an agent reasoning trace."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class StepType(StrEnum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


class AgentStep(BaseModel):
    """A single step in the ReAct reasoning loop."""

    step_index: int
    step_type: StepType
    content: str
    skill_name: str | None = None  # populated for ACTION steps
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentTrace(BaseModel):
    """Complete reasoning trace for one agent invocation."""

    query: str
    image_paths: list[str] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    final_answer: str | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None

    def append(self, step: AgentStep) -> None:
        """Add a step to the trace."""
        self.steps.append(step)

    def num_steps(self) -> int:
        """Return total number of recorded steps."""
        return len(self.steps)
