"""Skill: read and interpret spectroscopy images (IR, NMR, UV-Vis, MS)."""

from __future__ import annotations

from typing import Any

from PIL.Image import Image

from chemvision.skills.base import BaseSkill, SkillResult


class SpectrumReadingSkill(BaseSkill):
    """Identify peaks, axes, and spectral features in spectroscopy plots.

    Supports IR, 1H-NMR, 13C-NMR, UV-Vis, and mass spectrometry images.
    """

    name = "spectrum_reading"

    def build_prompt(self, **kwargs: Any) -> str:
        """Return the default spectrum-analysis instruction."""
        return (
            "You are an expert analytical chemist. "
            "Examine this spectrum image and identify: "
            "(1) the type of spectrum, "
            "(2) major peaks with approximate positions, "
            "(3) functional groups or fragments they indicate, "
            "(4) any notable features or artefacts. "
            "Be concise and structured."
        )

    def __call__(self, image: Image, model: Any, **kwargs: Any) -> SkillResult:
        """Run spectrum reading on a single image."""
        raise NotImplementedError
