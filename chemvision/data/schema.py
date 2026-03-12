"""Pydantic schemas for dataset records and build configuration."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ImageDomain(StrEnum):
    SPECTROSCOPY = "spectroscopy"
    MICROSCOPY = "microscopy"
    CHROMATOGRAPHY = "chromatography"
    MOLECULAR_DIAGRAM = "molecular_diagram"
    CRYSTAL_STRUCTURE = "crystal_structure"
    SIMULATION = "simulation"
    OTHER = "other"


class ImageRecord(BaseModel):
    """A single annotated scientific image sample."""

    id: str = Field(..., description="Unique sample identifier.")
    image_path: Path = Field(..., description="Absolute path to the image file.")
    domain: ImageDomain = Field(..., description="Scientific domain of the image.")
    question: str = Field(..., description="Natural-language question about the image.")
    answer: str = Field(..., description="Ground-truth answer string.")
    # Optional fields — all default to None for backward compatibility
    bbox: list[float] | None = Field(
        None,
        description="Bounding box [x0, y0, x1, y1] in normalised [0, 1] coordinates, "
        "or None when the question refers to the full image.",
    )
    difficulty: Literal["easy", "medium", "hard"] | None = Field(
        None,
        description="Question difficulty tier.",
    )
    source: str | None = Field(
        None,
        description="Provenance tag, e.g. 'synthetic_vasp', 'synthetic_lammps', "
        "'literature_arxiv', 'literature_doi'.",
    )
    metadata: dict[str, object] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Source-specific configuration models
# ---------------------------------------------------------------------------


class SyntheticConfig(BaseModel):
    """Configuration for the synthetic VASP/LAMMPS data generator."""

    files: list[Path] = Field(
        default_factory=list,
        description="VASP OUTCAR or LAMMPS dump files to process.",
    )
    n_questions_per_difficulty: int = Field(2, gt=0, description="QA pairs per difficulty tier.")
    output_subdir: str = Field("synthetic", description="Sub-directory under output_dir for images.")
    seed: int = Field(42)


class ScraperConfig(BaseModel):
    """Configuration for the literature figure scraper."""

    identifiers: list[str] = Field(
        default_factory=list,
        description="DOI strings (e.g. '10.1038/...') or arXiv IDs (e.g. '2301.00001').",
    )
    max_figures_per_paper: int = Field(10, gt=0)
    request_delay: float = Field(1.0, ge=0.0, description="Seconds to wait between requests.")
    include_image_in_haiku_call: bool = Field(
        True,
        description="Whether to send the figure image to Claude Haiku for visual grounding.",
    )
    output_subdir: str = Field("literature", description="Sub-directory under output_dir for images.")
    api_key: str | None = Field(
        None,
        description="Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.",
    )


# ---------------------------------------------------------------------------
# Top-level dataset build configuration
# ---------------------------------------------------------------------------


class DatasetConfig(BaseModel):
    """Configuration for a dataset build run."""

    source_dir: Path = Field(..., description="Root directory containing raw images.")
    output_dir: Path = Field(..., description="Where to write processed dataset files.")
    train_ratio: float = Field(0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(0.1, ge=0.0, le=1.0)
    seed: int = Field(42)
    domains: list[ImageDomain] = Field(
        default_factory=list,
        description="Filter by domain; empty = all.",
    )
    # Optional sub-pipeline configs — if provided, they take precedence over
    # the raw-image collect+annotate fallback path.
    synthetic: SyntheticConfig | None = Field(
        None,
        description="Synthetic generator config. Set to enable VASP/LAMMPS pipeline.",
    )
    scraper: ScraperConfig | None = Field(
        None,
        description="Literature scraper config. Set to enable arXiv/DOI pipeline.",
    )
