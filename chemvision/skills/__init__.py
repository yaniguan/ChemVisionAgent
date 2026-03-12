"""Composable vision skill modules for scientific image understanding.

Each skill is a self-contained callable that accepts an image + optional
kwargs and returns a strongly-typed Pydantic output model.

Public API
----------
BaseSkill                -- abstract base all skills must inherit
SkillResult              -- generic output container (base of all typed outputs)

Built-in skills
~~~~~~~~~~~~~~~
AnalyzeStructureSkill    -- lattice params, symmetry, defect mapping
ExtractSpectrumSkill     -- peak extraction from XRD / Raman / XPS
CompareStructuresSkill   -- quantitative diff between multiple structure images
ValidateCaptionSkill     -- consistency check between figure and caption
DetectAnomalySkill       -- anomaly detection and severity ranking

Legacy skills (stub implementations kept for backward compatibility)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SpectrumReadingSkill     -- basic spectrum reading (generic, unstructured)
MolecularStructureSkill  -- SMILES / functional group extraction
MicroscopySkill          -- morphology description

Typed output models
~~~~~~~~~~~~~~~~~~~
StructureAnalysis, LatticeParams, DefectLocation
SpectrumData, Peak
StructureComparison, DiffRegion, QuantitativeChange
CaptionValidation
AnomalyReport, Anomaly

Registry
~~~~~~~~
SkillRegistry            -- OOP registry class (shares state with functional API)
DEFAULT_REGISTRY         -- pre-built instance with all five skills registered
register_skill           -- register a custom skill at runtime
get_skill                -- retrieve a skill by name
list_skills              -- list all registered skill names
"""

from chemvision.skills.analyze_structure import AnalyzeStructureSkill
from chemvision.skills.base import BaseSkill, SkillResult
from chemvision.skills.compare_structures import CompareStructuresSkill
from chemvision.skills.detect_anomaly import DetectAnomalySkill
from chemvision.skills.extract_spectrum import ExtractSpectrumSkill
from chemvision.skills.microscopy import MicroscopySkill
from chemvision.skills.molecular import MolecularStructureSkill
from chemvision.skills.outputs import (
    Anomaly,
    AnomalyReport,
    CaptionValidation,
    DefectLocation,
    DiffRegion,
    LatticeParams,
    Peak,
    QuantitativeChange,
    SpectrumData,
    StructureAnalysis,
    StructureComparison,
)
from chemvision.skills.registry import get_skill, list_skills, register_skill
from chemvision.skills.skill_registry import DEFAULT_REGISTRY, SkillRegistry
from chemvision.skills.spectrum import SpectrumReadingSkill
from chemvision.skills.validate_caption import ValidateCaptionSkill

__all__ = [
    # Base
    "BaseSkill",
    "SkillResult",
    # New typed skills
    "AnalyzeStructureSkill",
    "ExtractSpectrumSkill",
    "CompareStructuresSkill",
    "ValidateCaptionSkill",
    "DetectAnomalySkill",
    # Legacy skills
    "SpectrumReadingSkill",
    "MolecularStructureSkill",
    "MicroscopySkill",
    # Typed outputs
    "StructureAnalysis",
    "LatticeParams",
    "DefectLocation",
    "SpectrumData",
    "Peak",
    "StructureComparison",
    "DiffRegion",
    "QuantitativeChange",
    "CaptionValidation",
    "AnomalyReport",
    "Anomaly",
    # Registry
    "SkillRegistry",
    "DEFAULT_REGISTRY",
    "register_skill",
    "get_skill",
    "list_skills",
]
