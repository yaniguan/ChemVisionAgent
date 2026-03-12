"""Agent configuration schema."""

from __future__ import annotations

from pydantic import BaseModel, Field

from chemvision.models.config import ModelConfig


class AgentConfig(BaseModel):
    """Configuration for the ChemVision ReAct agent."""

    # Vision model for skill execution (None → Anthropic API fallback)
    model: ModelConfig | None = Field(
        None,
        description="Vision model for skill execution. None uses the Anthropic API fallback.",
    )

    # Planning LLM
    planning_model: str = Field(
        "claude-sonnet-4-20250514",
        description="Anthropic model ID used for ReAct planning.",
    )
    anthropic_api_key: str | None = Field(
        None,
        description="Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.",
    )

    # Confidence gating
    confidence_threshold: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="Skill outputs with confidence below this value are flagged.",
    )

    max_steps: int = Field(10, gt=0, description="Hard cap on ReAct iterations.")
    skill_names: list[str] = Field(
        default_factory=list,
        description="Skills available to the agent; empty = all registered.",
    )
    verbose: bool = Field(False, description="Stream Thought/Action/Observation to stdout.")
