"""Skill: validate scientific figure captions for consistency with image content."""

from __future__ import annotations

from typing import Any

from PIL.Image import Image

from chemvision.skills._parse import extract_json, to_float, to_list, to_str
from chemvision.skills.base import BaseSkill, SkillResult
from chemvision.skills.outputs import CaptionValidation

_PROMPT_TEMPLATE = """\
You are a rigorous scientific peer reviewer.
Examine the provided figure and validate the following caption.

Caption:
\"\"\"{caption}\"\"\"

Assess whether every claim, measurement, scale, or label in the caption is
accurately and completely represented in the figure.

Your output MUST be a single valid JSON object matching this schema exactly:
{{
  "consistency_score": <0.0–1.0, where 1.0 = fully consistent>,
  "contradictions": [
    "<description of a specific inconsistency, mis-labelling, or unsupported claim>"
  ],
  "confidence": <0.0–1.0>
}}

List only concrete, verifiable contradictions.
Leave \"contradictions\" as an empty list if the caption is fully consistent.
Respond with only the JSON object, no other text.\
"""


class ValidateCaptionSkill(BaseSkill):
    """Check whether a figure caption accurately describes the image.

    Computes a ``consistency_score`` in [0, 1] and enumerates specific
    contradictions found by the model.

    Example
    -------
    >>> skill = ValidateCaptionSkill()
    >>> result = skill(image, model, caption="XRD pattern showing cubic phase.")
    >>> result.consistency_score
    0.85
    >>> result.contradictions
    ['Caption claims cubic phase but the pattern shows tetragonal splitting.']
    """

    name = "validate_figure_caption"

    def build_prompt(self, **kwargs: Any) -> str:
        caption = to_str(kwargs.get("caption"), "(no caption provided)")
        return _PROMPT_TEMPLATE.format(caption=caption)

    def __call__(self, image: Image, model: Any, **kwargs: Any) -> CaptionValidation:
        """Run caption validation and return a typed :class:`CaptionValidation`.

        Parameters
        ----------
        image:
            The figure image to validate against.
        model:
            Loaded :class:`~chemvision.models.base.BaseVisionModel`.
        caption:
            The caption text to check (kwarg, required).
        """
        prompt = self.build_prompt(**kwargs)
        raw = model.generate(image, prompt)
        data = extract_json(raw) or {}

        contradictions = [
            to_str(item)
            for item in to_list(data.get("contradictions"))
            if item is not None
        ]

        return CaptionValidation(
            skill_name=self.name,
            raw_output=raw,
            parsed=data,
            confidence=to_float(data.get("confidence")),
            consistency_score=min(1.0, max(0.0, float(data.get("consistency_score", 0.0)))),
            contradictions=contradictions,
        )
