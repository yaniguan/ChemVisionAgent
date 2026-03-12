"""Skill: describe morphology and features in microscopy images."""

from __future__ import annotations

from typing import Any

from PIL.Image import Image

from chemvision.skills.base import BaseSkill, SkillResult


class MicroscopySkill(BaseSkill):
    """Analyse SEM, TEM, optical, and fluorescence microscopy images.

    Returns morphological descriptions, scale-bar readings, and particle
    statistics where applicable.
    """

    name = "microscopy"

    def build_prompt(self, **kwargs: Any) -> str:
        """Return the default microscopy description instruction."""
        return (
            "You are an expert materials scientist. "
            "Examine this microscopy image and describe: "
            "(1) imaging modality (SEM/TEM/optical/fluorescence), "
            "(2) scale bar value if visible, "
            "(3) morphology of observed structures (shape, size range, distribution), "
            "(4) any notable defects, phases, or features. "
            "Be precise and quantitative where possible."
        )

    def __call__(self, image: Image, model: Any, **kwargs: Any) -> SkillResult:
        """Run microscopy analysis on a single image."""
        raise NotImplementedError
