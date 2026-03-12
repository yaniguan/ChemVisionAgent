"""Skill: analyse crystal structure, symmetry, and defects from a materials image."""

from __future__ import annotations

from typing import Any

from PIL.Image import Image

from chemvision.skills._parse import extract_json, to_float, to_float_required, to_list, to_str
from chemvision.skills.base import BaseSkill, SkillResult
from chemvision.skills.outputs import DefectLocation, LatticeParams, StructureAnalysis

_PROMPT_TEMPLATE = """\
You are an expert materials scientist and crystallographer.
Analyse this {material_type} sample image.

Your output MUST be a single valid JSON object matching this schema exactly:
{{
  "lattice_params": {{
    "a": <float or null>,
    "b": <float or null>,
    "c": <float or null>,
    "alpha": <float or null>,
    "beta": <float or null>,
    "gamma": <float or null>,
    "unit": "Å"
  }},
  "symmetry": "<crystal system or space group, e.g. cubic, Fm-3m>",
  "defect_locations": [
    {{
      "x": <0.0–1.0 normalised>,
      "y": <0.0–1.0 normalised>,
      "defect_type": "<vacancy|interstitial|dislocation|grain_boundary|stacking_fault|other>",
      "confidence": <0.0–1.0>
    }}
  ],
  "defect_density": <float or null>,
  "confidence": <0.0–1.0>
}}

Use null for any field that cannot be determined from the image.
Respond with only the JSON object, no other text.\
"""


class AnalyzeStructureSkill(BaseSkill):
    """Identify lattice parameters, symmetry, and defect sites in materials images.

    Accepts SEM, TEM, HRTEM, SAED, and simulated structure images.  The
    ``material_type`` kwarg (e.g. ``"perovskite"``, ``"FCC metal"``) is
    interpolated into the prompt to prime the model's domain knowledge.

    Example
    -------
    >>> skill = AnalyzeStructureSkill()
    >>> result = skill(image, model, material_type="BaTiO3 perovskite")
    >>> result.symmetry
    'tetragonal'
    >>> result.lattice_params.a
    3.99
    """

    name = "analyze_structure"

    def build_prompt(self, **kwargs: Any) -> str:
        material_type = to_str(kwargs.get("material_type"), "crystalline material")
        return _PROMPT_TEMPLATE.format(material_type=material_type)

    def __call__(self, image: Image, model: Any, **kwargs: Any) -> StructureAnalysis:
        """Run structure analysis and return a typed :class:`StructureAnalysis`.

        Parameters
        ----------
        image:
            RGB PIL image of the material.
        model:
            Loaded :class:`~chemvision.models.base.BaseVisionModel`.
        material_type:
            Optional string describing the material (kwarg).
        """
        prompt = self.build_prompt(**kwargs)
        raw = model.generate(image, prompt)
        data = extract_json(raw) or {}

        lattice_raw = data.get("lattice_params") or {}
        lattice = LatticeParams(
            a=to_float(lattice_raw.get("a")),
            b=to_float(lattice_raw.get("b")),
            c=to_float(lattice_raw.get("c")),
            alpha=to_float(lattice_raw.get("alpha")),
            beta=to_float(lattice_raw.get("beta")),
            gamma=to_float(lattice_raw.get("gamma")),
            unit=to_str(lattice_raw.get("unit"), "Å"),
            raw_output=raw,
        )

        defects: list[DefectLocation] = []
        for d in to_list(data.get("defect_locations")):
            if isinstance(d, dict):
                defects.append(
                    DefectLocation(
                        x=to_float_required(d.get("x"), 0.0),
                        y=to_float_required(d.get("y"), 0.0),
                        defect_type=to_str(d.get("defect_type"), "unknown"),
                        confidence=to_float(d.get("confidence")),
                        raw_output=raw,
                    )
                )

        return StructureAnalysis(
            skill_name=self.name,
            raw_output=raw,
            parsed=data,
            confidence=to_float(data.get("confidence")),
            lattice_params=lattice,
            symmetry=to_str(data.get("symmetry")),
            defect_locations=defects,
            defect_density=to_float(data.get("defect_density")),
        )
