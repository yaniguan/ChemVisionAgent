"""Skill: extract structural information from molecular diagrams."""

from __future__ import annotations

from typing import Any

from PIL.Image import Image

from chemvision.skills.base import BaseSkill, SkillResult


class MolecularStructureSkill(BaseSkill):
    """Extract SMILES, functional groups, and structural features from 2-D diagrams.

    Handles hand-drawn and computer-generated skeletal structures.
    """

    name = "molecular_structure"

    def build_prompt(self, **kwargs: Any) -> str:
        """Return the default molecular-structure extraction instruction."""
        return (
            "You are an expert organic chemist. "
            "Examine this molecular structure diagram and provide: "
            "(1) SMILES string if determinable, "
            "(2) molecular formula, "
            "(3) list of functional groups present, "
            "(4) key stereocentres or geometric isomerism notes. "
            "Return structured JSON."
        )

    def __call__(self, image: Image, model: Any, **kwargs: Any) -> SkillResult:
        """Run molecular structure extraction on a single image."""
        raise NotImplementedError
