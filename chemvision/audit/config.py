"""Audit configuration schema."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class AuditConfig(BaseModel):
    """Configuration for a capability audit run."""

    model_name: str = Field(..., description="Identifier of the model under audit.")
    benchmark_dir: Path = Field(..., description="Directory with benchmark image/QA pairs.")
    skill_names: list[str] = Field(default_factory=list, description="Skills to probe; empty = all registered.")
    output_dir: Path = Field(Path("audit_results/"))
    batch_size: int = Field(8, gt=0)
    seed: int = Field(42)
