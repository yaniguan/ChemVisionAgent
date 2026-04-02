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


# ---------------------------------------------------------------------------
# molecular_structure
# ---------------------------------------------------------------------------


class FunctionalGroup(SkillResult):
    """A single functional group identified in a molecular structure."""

    skill_name: str = "molecular_structure"
    raw_output: str = ""

    name: str = Field("", description="E.g. 'hydroxyl', 'carboxyl', 'amine'.")
    smarts: str | None = Field(None, description="SMARTS pattern if determinable.")
    count: int = Field(1, ge=1, description="Number of occurrences in the molecule.")


class StereocenterInfo(SkillResult):
    """A single stereocenter or geometric isomerism element."""

    skill_name: str = "molecular_structure"
    raw_output: str = ""

    atom_or_bond: str = Field("", description="Atom label or bond description, e.g. 'C3', 'C2=C3'.")
    descriptor: str = Field("", description="R/S for chiral centres; E/Z for double bonds.")
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class MolecularStructureData(SkillResult):
    """Full output of the ``molecular_structure`` skill."""

    smiles: str | None = Field(None, description="SMILES string of the molecule.")
    iupac_name: str | None = Field(None, description="IUPAC name if determinable.")
    common_name: str | None = Field(None, description="Common or trade name if shown.")
    molecular_formula: str | None = Field(None, description="E.g. 'C6H12O6'.")
    molecular_weight: float | None = Field(None, description="Estimated MW in g/mol.")
    functional_groups: list[FunctionalGroup] = Field(default_factory=list)
    stereocenters: list[StereocenterInfo] = Field(default_factory=list)
    ring_systems: list[str] = Field(
        default_factory=list,
        description="E.g. ['benzene', 'cyclohexane', 'pyridine'].",
    )
    num_rings: int | None = Field(None, ge=0)


class AnomalyReport(SkillResult):
    """Full output of the ``detect_anomaly`` skill."""

    anomalies: list[Anomaly] = Field(default_factory=list)
    severity: Literal["none", "low", "medium", "high"] = Field(
        "none", description="Overall severity across all detected anomalies."
    )
    recommendations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# extract_reaction
# ---------------------------------------------------------------------------


class Molecule(SkillResult):
    """A molecule identified in a reaction scheme."""

    skill_name: str = "extract_reaction"
    raw_output: str = ""

    name: str = Field("", description="IUPAC name or common name.")
    smiles: str | None = Field(None, description="SMILES string if determinable.")
    role: Literal["reactant", "product", "reagent", "catalyst", "solvent", "unknown"] = "unknown"


class ReactionConditions(SkillResult):
    """Experimental conditions extracted from a reaction scheme or table."""

    skill_name: str = "extract_reaction"
    raw_output: str = ""

    temperature: str | None = Field(None, description="E.g. '80 °C', 'rt', '-78 °C'.")
    pressure: str | None = Field(None, description="E.g. '1 atm', '5 bar'.")
    solvent: str | None = Field(None, description="E.g. 'THF', 'MeOH/H2O'.")
    time: str | None = Field(None, description="E.g. '12 h', '30 min'.")
    atmosphere: str | None = Field(None, description="E.g. 'N2', 'Ar', 'air'.")
    yield_percent: float | None = Field(None, description="Reported yield 0–100.")


class ReactionData(SkillResult):
    """Full output of the ``extract_reaction`` skill."""

    reaction_type: str = Field(
        "", description="E.g. 'Suzuki coupling', 'aldol condensation', 'oxidation'."
    )
    molecules: list[Molecule] = Field(default_factory=list)
    conditions: ReactionConditions | None = None
    arrow_type: str = Field(
        "", description="E.g. 'single-step', 'retrosynthetic', 'equilibrium'."
    )


# ---------------------------------------------------------------------------
# analyze_microscopy
# ---------------------------------------------------------------------------


class ParticleMeasurement(SkillResult):
    """Dimensions measured on one individual particle."""

    skill_name: str = "analyze_microscopy"
    raw_output: str = ""

    diameter: float | None = Field(None, description="Diameter or longest axis in calibrated units.")
    aspect_ratio: float | None = Field(None, ge=0.0, description="Length / width (1.0 = sphere).")
    shape: str = Field("unknown", description="E.g. 'spherical', 'rod', 'platelet'.")
    location_x: float = Field(0.0, ge=0.0, le=1.0, description="Normalised x centre [0, 1].")
    location_y: float = Field(0.0, ge=0.0, le=1.0, description="Normalised y centre [0, 1].")


class SizeStatistics(SkillResult):
    """Aggregate particle-size statistics across all measured particles."""

    skill_name: str = "analyze_microscopy"
    raw_output: str = ""

    mean_diameter: float | None = Field(None, description="Mean diameter in *unit*.")
    std_diameter: float | None = Field(None, ge=0.0, description="Standard deviation of diameter.")
    min_diameter: float | None = Field(None, description="Smallest measured diameter.")
    max_diameter: float | None = Field(None, description="Largest measured diameter.")
    unit: str = Field("nm", description="Unit for all diameter values.")
    distribution: Literal["monodisperse", "polydisperse", "bimodal", "unknown"] = "unknown"
    particle_count: int | None = Field(None, ge=0, description="Number of particles measured.")


class ScaleBar(SkillResult):
    """Scale bar metadata read directly from the image."""

    skill_name: str = "analyze_microscopy"
    raw_output: str = ""

    value: float | None = Field(None, description="Numeric value shown on the scale bar.")
    unit: str = Field("nm", description="Unit of the scale bar value.")
    pixel_length: int | None = Field(None, ge=0, description="Pixel length of the scale bar segment.")
    nm_per_pixel: float | None = Field(None, ge=0.0, description="Calibrated nm / pixel ratio.")


class MorphologyInfo(SkillResult):
    """Qualitative morphology description of the imaged material."""

    skill_name: str = "analyze_microscopy"
    raw_output: str = ""

    shape: str = Field(
        "unknown",
        description="Dominant particle shape: spherical, rod, platelet, dendritic, porous, "
        "irregular, core-shell, or other.",
    )
    surface_texture: str = Field(
        "unknown",
        description="Surface finish: smooth, rough, faceted, porous, or other.",
    )
    aggregation: str = Field(
        "unknown",
        description="Dispersion state: dispersed, agglomerated, sintered, or clustered.",
    )
    description: str = Field("", description="Free-text morphology summary.")


class MicroscopyAnalysis(SkillResult):
    """Full output of the ``analyze_microscopy`` skill."""

    morphology: MorphologyInfo | None = None
    particles: list[ParticleMeasurement] = Field(
        default_factory=list,
        description="Per-particle measurements (up to ~20 representative particles).",
    )
    size_statistics: SizeStatistics | None = None
    scale_bar: ScaleBar | None = None
    imaging_modality: str = Field(
        "unknown",
        description="Detected imaging modality: SEM, TEM, STEM, AFM, OM, or other.",
    )
    magnification: str | None = Field(None, description="Magnification label if visible, e.g. '50000x'.")
