"""Typed Pydantic output models for each composable vision skill.

Every model inherits from :class:`~chemvision.skills.base.SkillResult` so it
remains compatible with the existing skill registry and audit pipeline while
exposing strongly-typed, domain-specific fields.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from chemvision.skills.base import SkillResult


# ---------------------------------------------------------------------------
# analyze_structure
# ---------------------------------------------------------------------------


class LatticeParams(SkillResult):
    """Crystallographic lattice parameters extracted from a structural image."""

    skill_name: str = "analyze_structure"
    raw_output: str = ""

    a: float | None = Field(None, description="Lattice constant a (Å).")
    b: float | None = Field(None, description="Lattice constant b (Å).")
    c: float | None = Field(None, description="Lattice constant c (Å).")
    alpha: float | None = Field(None, description="Angle α (°).")
    beta: float | None = Field(None, description="Angle β (°).")
    gamma: float | None = Field(None, description="Angle γ (°).")
    unit: str = Field("Å", description="Unit for a, b, c.")


class DefectLocation(SkillResult):
    """Single detected defect with normalised image coordinates."""

    skill_name: str = "analyze_structure"
    raw_output: str = ""

    x: float = Field(0.0, ge=0.0, le=1.0, description="Normalised x position [0, 1].")
    y: float = Field(0.0, ge=0.0, le=1.0, description="Normalised y position [0, 1].")
    defect_type: str = Field("unknown", description="E.g. vacancy, dislocation, grain_boundary.")
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class StructureAnalysis(SkillResult):
    """Full output of the ``analyze_structure`` skill."""

    lattice_params: LatticeParams | None = None
    symmetry: str = Field("", description="Crystal system or space group.")
    defect_locations: list[DefectLocation] = Field(default_factory=list)
    defect_density: float | None = Field(
        None, description="Estimated defect density (defects / nm²)."
    )


# ---------------------------------------------------------------------------
# extract_spectrum_data
# ---------------------------------------------------------------------------


class Peak(SkillResult):
    """A single resolved peak in a spectrum."""

    skill_name: str = "extract_spectrum_data"
    raw_output: str = ""

    position: float = Field(0.0, description="Peak position on the x-axis.")
    intensity: float = Field(
        0.0, ge=0.0, description="Normalised peak intensity [0, 1] or raw counts."
    )
    assignment: str = Field("", description="Chemical assignment, e.g. 'C=O stretch'.")
    fwhm: float | None = Field(None, description="Full-width at half-maximum.")


class SpectrumData(SkillResult):
    """Full output of the ``extract_spectrum_data`` skill."""

    peaks: list[Peak] = Field(default_factory=list)
    background_level: float | None = Field(None, description="Estimated background level [0, 1].")
    snr: float | None = Field(None, description="Signal-to-noise ratio estimate.")
    spectrum_type: str = Field("", description="Detected spectrum type, e.g. 'XRD'.")


# ---------------------------------------------------------------------------
# compare_structures
# ---------------------------------------------------------------------------


class DiffRegion(SkillResult):
    """Bounding region in the compared image where a difference was detected."""

    skill_name: str = "compare_structures"
    raw_output: str = ""

    x: float = Field(0.0, ge=0.0, le=1.0, description="Normalised left edge [0, 1].")
    y: float = Field(0.0, ge=0.0, le=1.0, description="Normalised top edge [0, 1].")
    width: float = Field(0.0, ge=0.0, le=1.0)
    height: float = Field(0.0, ge=0.0, le=1.0)
    description: str = ""


class QuantitativeChange(SkillResult):
    """One measurable metric that changed between compared structures."""

    skill_name: str = "compare_structures"
    raw_output: str = ""

    metric: str = ""
    before: float | None = None
    after: float | None = None
    delta: float | None = None
    unit: str = ""


class StructureComparison(SkillResult):
    """Full output of the ``compare_structures`` skill."""

    diff_regions: list[DiffRegion] = Field(default_factory=list)
    quantitative_changes: list[QuantitativeChange] = Field(default_factory=list)
    trend: str = Field("", description="Overall trend description.")


# ---------------------------------------------------------------------------
# validate_figure_caption
# ---------------------------------------------------------------------------


class CaptionValidation(SkillResult):
    """Full output of the ``validate_figure_caption`` skill."""

    consistency_score: float = Field(
        0.0, ge=0.0, le=1.0, description="1.0 = fully consistent with the image."
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Specific inconsistencies or inaccuracies found.",
    )


# ---------------------------------------------------------------------------
# detect_anomaly
# ---------------------------------------------------------------------------


class Anomaly(SkillResult):
    """A single detected anomaly with location and severity."""

    skill_name: str = "detect_anomaly"
    raw_output: str = ""

    location_x: float = Field(0.0, ge=0.0, le=1.0)
    location_y: float = Field(0.0, ge=0.0, le=1.0)
    anomaly_type: str = Field(
        "other",
        description="E.g. crack, void, contamination, phase_inclusion, surface_damage.",
    )
    description: str = ""
    severity: Literal["low", "medium", "high"] = "low"
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class AnomalyReport(SkillResult):
    """Full output of the ``detect_anomaly`` skill."""

    anomalies: list[Anomaly] = Field(default_factory=list)
    severity: Literal["none", "low", "medium", "high"] = Field(
        "none", description="Overall severity across all detected anomalies."
    )
    recommendations: list[str] = Field(default_factory=list)
