"""Structured analysis report synthesised from an agent trace."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from chemvision.agent.tool_log import ToolCallLog


class AnalysisReport(BaseModel):
    """Final structured report produced by the ChemVision agent.

    Captures the synthesised answer, reasoning trace metadata, all tool
    calls that were made, and a confidence-propagation flag.

    Example
    -------
    >>> report = AnalysisReport.build(
    ...     question="What phase is present?",
    ...     image_paths=["img.png"],
    ...     final_answer="The dominant phase is anatase TiO2.",
    ...     tool_logs=[...],
    ...     confidence_threshold=0.75,
    ... )
    >>> report.low_confidence_flag
    False
    """

    question: str
    image_paths: list[str]
    final_answer: str
    tool_logs: list[ToolCallLog] = Field(default_factory=list)

    # Confidence propagation
    low_confidence_flag: bool = Field(
        False,
        description=(
            "True when ANY intermediate skill call had confidence below the threshold. "
            "Indicates conclusions should be treated with caution."
        ),
    )
    min_intermediate_confidence: float | None = Field(
        None,
        description="Lowest confidence observed across all skill calls.",
    )

    num_steps: int = Field(0, description="Total number of ReAct steps taken.")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Full reasoning trace (thought + action + observation steps)
    trace_steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Serialised AgentStep records for full trace display including THOUGHT steps.",
    )

    # Structured data parsed from final_answer when it contains JSON
    structured_data: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Key-value pairs extracted when final_answer is a JSON object. "
            "Useful for downstream assertions (e.g. phase_transition_temperature)."
        ),
    )

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        question: str,
        image_paths: list[str],
        final_answer: str,
        tool_logs: list[ToolCallLog],
        num_steps: int = 0,
        confidence_threshold: float = 0.75,
        trace_steps: list[dict[str, Any]] | None = None,
    ) -> "AnalysisReport":
        """Construct an :class:`AnalysisReport` with confidence propagation.

        Scans *tool_logs* for any call whose ``confidence`` is below
        *confidence_threshold* and sets :attr:`low_confidence_flag` accordingly.

        Parameters
        ----------
        question:
            The original natural-language question.
        image_paths:
            Paths to images that were analysed.
        final_answer:
            Synthesised answer text.
        tool_logs:
            All :class:`ToolCallLog` records from the run.
        num_steps:
            Total ReAct steps (thought + action + observation).
        confidence_threshold:
            Agent-configured threshold; any skill below this flags the report.
        """
        confidences: list[float] = [
            log.confidence
            for log in tool_logs
            if log.confidence is not None
        ]

        min_conf: float | None = min(confidences) if confidences else None
        low_flag = min_conf is not None and min_conf < confidence_threshold

        # Back-fill low_confidence on individual logs
        for log in tool_logs:
            if log.confidence is not None and log.confidence < confidence_threshold:
                log.low_confidence = True

        # Try to parse structured fields from the final answer
        structured: dict[str, Any] = {}
        stripped = final_answer.strip()
        if stripped.startswith("{"):
            try:
                structured = json.loads(stripped)
            except json.JSONDecodeError:
                pass

        return cls(
            question=question,
            image_paths=image_paths,
            final_answer=final_answer,
            tool_logs=tool_logs,
            low_confidence_flag=low_flag,
            min_intermediate_confidence=min_conf,
            num_steps=num_steps,
            structured_data=structured,
            trace_steps=trace_steps or [],
        )

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return self.model_dump(mode="json")
