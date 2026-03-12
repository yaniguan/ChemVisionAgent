"""Skill: extract quantitative peak data from XRD, Raman, or XPS spectra."""

from __future__ import annotations

from typing import Any, Literal

from PIL.Image import Image

from chemvision.skills._parse import extract_json, to_float, to_float_required, to_list, to_str
from chemvision.skills.base import BaseSkill, SkillResult
from chemvision.skills.outputs import Peak, SpectrumData

_PROMPT_TEMPLATE = """\
You are an expert spectroscopist.
Analyse this {spectrum_type} spectrum image and extract quantitative data.

Your output MUST be a single valid JSON object matching this schema exactly:
{{
  "peaks": [
    {{
      "position": <x-axis value, e.g. 2θ in ° for XRD, cm⁻¹ for Raman, eV for XPS>,
      "intensity": <0.0–1.0 normalised to the strongest peak>,
      "assignment": "<phase, bond, or core-level, e.g. α-Fe2O3 (104), C=O, Fe 2p3/2>",
      "fwhm": <full-width at half-maximum in the same x-axis units, or null>
    }}
  ],
  "background_level": <estimated baseline level 0.0–1.0, or null>,
  "snr": <signal-to-noise ratio estimate as a positive number, or null>,
  "confidence": <0.0–1.0>
}}

List peaks in order of decreasing intensity.
Use null for values that cannot be determined from the image.
Respond with only the JSON object, no other text.\
"""

_VALID_TYPES: frozenset[str] = frozenset({"XRD", "Raman", "XPS"})


class ExtractSpectrumSkill(BaseSkill):
    """Extract peak positions, intensities, and SNR from spectroscopy images.

    Supports XRD (2θ), Raman (cm⁻¹), and XPS (eV) spectra.  The
    ``spectrum_type`` kwarg selects the axis-unit hint in the prompt.

    Example
    -------
    >>> skill = ExtractSpectrumSkill()
    >>> result = skill(image, model, spectrum_type="Raman")
    >>> result.peaks[0].position
    1348.0
    >>> result.snr
    24.5
    """

    name = "extract_spectrum_data"

    def build_prompt(self, **kwargs: Any) -> str:
        spectrum_type = to_str(kwargs.get("spectrum_type"), "spectrum")
        if spectrum_type.upper() in _VALID_TYPES:
            spectrum_type = spectrum_type.upper()
        return _PROMPT_TEMPLATE.format(spectrum_type=spectrum_type)

    def __call__(self, image: Image, model: Any, **kwargs: Any) -> SpectrumData:
        """Run spectrum extraction and return a typed :class:`SpectrumData`.

        Parameters
        ----------
        image:
            RGB PIL image of the spectrum plot.
        model:
            Loaded :class:`~chemvision.models.base.BaseVisionModel`.
        spectrum_type:
            One of ``"XRD"``, ``"Raman"``, ``"XPS"`` (kwarg, case-insensitive).
        """
        spectrum_type = to_str(kwargs.get("spectrum_type"), "spectrum")
        prompt = self.build_prompt(**kwargs)
        raw = model.generate(image, prompt)
        data = extract_json(raw) or {}

        peaks: list[Peak] = []
        for p in to_list(data.get("peaks")):
            if isinstance(p, dict):
                peaks.append(
                    Peak(
                        position=to_float_required(p.get("position"), 0.0),
                        intensity=to_float_required(p.get("intensity"), 0.0),
                        assignment=to_str(p.get("assignment")),
                        fwhm=to_float(p.get("fwhm")),
                        raw_output=raw,
                    )
                )

        return SpectrumData(
            skill_name=self.name,
            raw_output=raw,
            parsed=data,
            confidence=to_float(data.get("confidence")),
            peaks=peaks,
            background_level=to_float(data.get("background_level")),
            snr=to_float(data.get("snr")),
            spectrum_type=spectrum_type,
        )
