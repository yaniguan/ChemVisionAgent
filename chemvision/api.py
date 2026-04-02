"""FastAPI application exposing ChemVision Agent as a REST service.

Endpoints
---------
POST /analyze   — run the ReAct agent on images + question
GET  /audit     — return the latest audit report for the loaded model
GET  /health    — liveness check
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chemvision.agent.agent import ChemVisionAgent
from chemvision.agent.config import AgentConfig
from chemvision.agent.report import AnalysisReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-safe app state
# ---------------------------------------------------------------------------

_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB per image
_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class _AppState:
    """Thread-safe container for singleton agent and latest audit."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.agent: ChemVisionAgent | None = None
        self.latest_audit: dict[str, Any] | None = None

    async def set_audit(self, report: dict[str, Any]) -> None:
        async with self._lock:
            self.latest_audit = report

    async def get_audit(self) -> dict[str, Any] | None:
        async with self._lock:
            return self.latest_audit


_STATE = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Initialise the agent once at startup; clean up on shutdown."""
    cfg = AgentConfig()  # reads ANTHROPIC_API_KEY from env
    _STATE.agent = ChemVisionAgent(cfg)
    yield
    _STATE.agent = None


app = FastAPI(
    title="ChemVision Agent API",
    description="Multimodal scientific image reasoning platform.",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze."""

    question: str = Field(..., description="Natural-language question about the image(s).")
    image_paths: list[str] = Field(
        ...,
        min_length=1,
        description="Absolute or relative paths to the image files to analyse.",
    )


class ToolCallSummary(BaseModel):
    """Compact summary of a single tool call for the API response."""

    skill_name: str
    confidence: float | None
    low_confidence: bool


class AnalyzeResponse(BaseModel):
    """Response schema for POST /analyze."""

    question: str
    image_paths: list[str]
    final_answer: str
    num_steps: int
    low_confidence_flag: bool
    min_intermediate_confidence: float | None
    tool_calls: list[ToolCallSummary]


class ReasonResponse(BaseModel):
    """Legacy response schema for /reason (kept for backward compatibility)."""

    answer: str
    num_steps: int
    model_used: str


class AuditResponse(BaseModel):
    """Response schema for GET /audit."""

    available: bool
    report: dict[str, Any] | None = None
    message: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "ok", "version": "0.2.0"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    """Run the ChemVision ReAct agent on images + a question.

    The agent plans which vision skills to invoke, executes them in order,
    and synthesises a final structured answer.  Any intermediate skill
    confidence below 0.75 sets ``low_confidence_flag`` on the response.
    """
    agent = _STATE.agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")

    # Validate image paths: existence, extension, size
    for p_str in body.image_paths:
        p = Path(p_str)
        if not p.exists():
            raise HTTPException(status_code=422, detail=f"Image not found: {p_str}")
        if p.suffix.lower() not in _ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=422, detail=f"Unsupported image format: {p.suffix}")
        if p.stat().st_size > _MAX_IMAGE_BYTES:
            raise HTTPException(status_code=422, detail=f"Image too large (>{_MAX_IMAGE_BYTES // 1024 // 1024} MB): {p_str}")

    try:
        report: AnalysisReport = agent.run(
            question=body.question,
            image_paths=body.image_paths,
        )
    except (ValueError, TypeError, KeyError) as exc:
        logger.exception("Agent failed with data error")
        raise HTTPException(status_code=422, detail=f"Analysis failed: {exc}") from exc
    except Exception as exc:
        logger.exception("Agent failed with unexpected error")
        raise HTTPException(status_code=500, detail=f"Internal error: {type(exc).__name__}") from exc

    # Store as latest audit report (thread-safe)
    await _STATE.set_audit(report.to_dict())

    return AnalyzeResponse(
        question=report.question,
        image_paths=report.image_paths,
        final_answer=report.final_answer,
        num_steps=report.num_steps,
        low_confidence_flag=report.low_confidence_flag,
        min_intermediate_confidence=report.min_intermediate_confidence,
        tool_calls=[
            ToolCallSummary(
                skill_name=log.skill_name,
                confidence=log.confidence,
                low_confidence=log.low_confidence,
            )
            for log in report.tool_logs
        ],
    )


@app.get("/audit", response_model=AuditResponse)
async def get_audit() -> AuditResponse:
    """Return the latest analysis report produced by the agent.

    The report is cached in memory from the most recent ``POST /analyze``
    call.  Returns 404-style JSON if no run has completed yet.
    """
    report_dict: dict[str, Any] | None = await _STATE.get_audit()
    if report_dict is None:
        return AuditResponse(
            available=False,
            message="No analysis has been run yet. Call POST /analyze first.",
        )

    return AuditResponse(available=True, report=report_dict)


# ---------------------------------------------------------------------------
# Legacy endpoint (kept for backward compatibility with existing tests)
# ---------------------------------------------------------------------------


@app.get("/reason", response_model=ReasonResponse, deprecated=True)
async def reason_get() -> ReasonResponse:
    """Deprecated placeholder — use POST /analyze instead."""
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use POST /analyze.",
    )
