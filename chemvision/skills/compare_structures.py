"""Skill: quantitatively compare two or more structural images side-by-side."""

from __future__ import annotations

from typing import Any

from PIL.Image import Image

from chemvision.skills._parse import extract_json, to_float, to_float_required, to_list, to_str
from chemvision.skills.base import BaseSkill, SkillResult
from chemvision.skills.outputs import DiffRegion, QuantitativeChange, StructureComparison

_PROMPT_TEMPLATE = """\
You are an expert materials scientist performing comparative structural analysis.
The image shows {n_images} structure(s) arranged side-by-side.
Comparison type: {comparison_type}.

Identify all structural differences and measurable changes between the panels.

Your output MUST be a single valid JSON object matching this schema exactly:
{{
  "diff_regions": [
    {{
      "x": <0.0–1.0 normalised left edge of the difference region>,
      "y": <0.0–1.0 normalised top edge>,
      "width": <0.0–1.0>,
      "height": <0.0–1.0>,
      "description": "<concise description of the difference>"
    }}
  ],
  "quantitative_changes": [
    {{
      "metric": "<name, e.g. grain_size, d-spacing, peak_shift>",
      "before": <numeric value or null>,
      "after": <numeric value or null>,
      "delta": <after minus before, or null>,
      "unit": "<unit string, e.g. nm, Å, °>"
    }}
  ],
  "trend": "<overall structural trend, e.g. grain_growth, phase_separation, amorphisation>",
  "confidence": <0.0–1.0>
}}

Use null for fields that cannot be determined.
Respond with only the JSON object, no other text.\
"""


def _concat_images(images: list[Image]) -> Image:
    """Paste *images* side-by-side into a single RGB canvas."""
    from PIL import Image as PILImage

    if len(images) == 1:
        return images[0]
    total_w = sum(img.width for img in images)
    max_h = max(img.height for img in images)
    canvas = PILImage.new("RGB", (total_w, max_h), color=(255, 255, 255))
    x = 0
    for img in images:
        canvas.paste(img.convert("RGB"), (x, 0))
        x += img.width
    return canvas


class CompareStructuresSkill(BaseSkill):
    """Detect and quantify structural differences across multiple images.

    Pass the full list of images via the ``images`` kwarg.  The first
    positional ``image`` argument is used as a fallback when ``images`` is
    absent so the skill stays compatible with the :class:`BaseSkill`
    single-image interface.

    All images are concatenated horizontally before inference so the model
    receives a single compound view.

    Example
    -------
    >>> skill = CompareStructuresSkill()
    >>> result = skill(
    ...     images[0], model,
    ...     images=images,
    ...     comparison_type="grain size evolution under annealing",
    ... )
    >>> result.trend
    'grain growth'
    """

    name = "compare_structures"

    def build_prompt(self, **kwargs: Any) -> str:
        n_images = int(kwargs.get("n_images", 2))
        comparison_type = to_str(kwargs.get("comparison_type"), "structural comparison")
        return _PROMPT_TEMPLATE.format(n_images=n_images, comparison_type=comparison_type)

    def __call__(self, image: Image, model: Any, **kwargs: Any) -> StructureComparison:
        """Run comparative analysis and return a typed :class:`StructureComparison`.

        Parameters
        ----------
        image:
            First (or only) image; used as fallback when ``images`` kwarg is absent.
        model:
            Loaded :class:`~chemvision.models.base.BaseVisionModel`.
        images:
            Full list of images to compare (kwarg).  Overrides *image* when present.
        comparison_type:
            Free-text description of the comparison context (kwarg).
        """
        images: list[Image] = list(kwargs.get("images") or [image])
        if not images:
            images = [image]

        compound = _concat_images(images)
        prompt = self.build_prompt(n_images=len(images), **kwargs)
        raw = model.generate(compound, prompt)
        data = extract_json(raw) or {}

        diff_regions: list[DiffRegion] = []
        for r in to_list(data.get("diff_regions")):
            if isinstance(r, dict):
                diff_regions.append(
                    DiffRegion(
                        x=to_float_required(r.get("x"), 0.0),
                        y=to_float_required(r.get("y"), 0.0),
                        width=to_float_required(r.get("width"), 0.0),
                        height=to_float_required(r.get("height"), 0.0),
                        description=to_str(r.get("description")),
                        raw_output=raw,
                    )
                )

        changes: list[QuantitativeChange] = []
        for c in to_list(data.get("quantitative_changes")):
            if isinstance(c, dict):
                changes.append(
                    QuantitativeChange(
                        metric=to_str(c.get("metric")),
                        before=to_float(c.get("before")),
                        after=to_float(c.get("after")),
                        delta=to_float(c.get("delta")),
                        unit=to_str(c.get("unit")),
                        raw_output=raw,
                    )
                )

        return StructureComparison(
            skill_name=self.name,
            raw_output=raw,
            parsed=data,
            confidence=to_float(data.get("confidence")),
            diff_regions=diff_regions,
            quantitative_changes=changes,
            trend=to_str(data.get("trend")),
        )
