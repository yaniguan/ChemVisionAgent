"""Skill: detect anomalies, defects, and unexpected features in scientific images."""

from __future__ import annotations

from typing import Any, Literal, get_args

from PIL.Image import Image

from chemvision.skills._parse import extract_json, to_float, to_list, to_str
from chemvision.skills.base import BaseSkill, SkillResult
from chemvision.skills.outputs import Anomaly, AnomalyReport

_PROMPT_TEMPLATE = """\
You are a quality-control and failure-analysis expert in materials science.
Examine this image in the context of: {domain_context}

Identify every anomaly, defect, or unexpected feature visible in the image.

Your output MUST be a single valid JSON object matching this schema exactly:
{{
  "anomalies": [
    {{
      "location_x": <0.0–1.0 normalised x position of the anomaly centre>,
      "location_y": <0.0–1.0 normalised y position>,
      "anomaly_type": "<crack|void|contamination|phase_inclusion|surface_damage|other>",
      "description": "<concise description of the anomaly>",
      "severity": "<low|medium|high>",
      "confidence": <0.0–1.0>
    }}
  ],
  "severity": "<none|low|medium|high>  (overall severity across all detected anomalies)",
  "recommendations": [
    "<actionable recommendation for mitigation or further investigation>"
  ],
  "confidence": <0.0–1.0>
}}

If no anomalies are found, set severity to "none" and leave anomalies as an empty list.
List recommendations in order of priority.
Respond with only the JSON object, no other text.\
"""

_ANOMALY_SEVERITIES: frozenset[str] = frozenset(get_args(Literal["low", "medium", "high"]))
_REPORT_SEVERITIES: frozenset[str] = frozenset(
    get_args(Literal["none", "low", "medium", "high"])
)


class DetectAnomalySkill(BaseSkill):
    """Locate anomalies in scientific images and rank their severity.

    Uses the ``domain_context`` kwarg (e.g. ``"SEM image of alumina coating"``)
    to prime the model's domain knowledge before anomaly detection.

    Example
    -------
    >>> skill = DetectAnomalySkill()
    >>> result = skill(image, model, domain_context="cross-section SEM of TBC coating")
    >>> result.severity
    'medium'
    >>> result.anomalies[0].anomaly_type
    'crack'
    """

    name = "detect_anomaly"

    def build_prompt(self, **kwargs: Any) -> str:
        domain_context = to_str(kwargs.get("domain_context"), "scientific materials image")
        return _PROMPT_TEMPLATE.format(domain_context=domain_context)

    def __call__(self, image: Image, model: Any, **kwargs: Any) -> AnomalyReport:
        """Run anomaly detection and return a typed :class:`AnomalyReport`.

        Parameters
        ----------
        image:
            RGB PIL image to inspect.
        model:
            Loaded :class:`~chemvision.models.base.BaseVisionModel`.
        domain_context:
            Description of the imaging modality and material (kwarg).
        """
        prompt = self.build_prompt(**kwargs)
        raw = model.generate(image, prompt)
        data = extract_json(raw) or {}

        anomalies: list[Anomaly] = []
        for a in to_list(data.get("anomalies")):
            if not isinstance(a, dict):
                continue
            raw_sev = to_str(a.get("severity"), "low").lower()
            sev: Literal["low", "medium", "high"] = (
                raw_sev if raw_sev in _ANOMALY_SEVERITIES else "low"  # type: ignore[assignment]
            )
            anomalies.append(
                Anomaly(
                    location_x=min(1.0, max(0.0, float(a.get("location_x") or 0.0))),
                    location_y=min(1.0, max(0.0, float(a.get("location_y") or 0.0))),
                    anomaly_type=to_str(a.get("anomaly_type"), "other"),
                    description=to_str(a.get("description")),
                    severity=sev,
                    confidence=to_float(a.get("confidence")),
                    raw_output=raw,
                )
            )

        raw_report_sev = to_str(data.get("severity"), "none").lower()
        report_sev: Literal["none", "low", "medium", "high"] = (
            raw_report_sev  # type: ignore[assignment]
            if raw_report_sev in _REPORT_SEVERITIES
            else "none"
        )

        recommendations = [
            to_str(r) for r in to_list(data.get("recommendations")) if r is not None
        ]

        return AnomalyReport(
            skill_name=self.name,
            raw_output=raw,
            parsed=data,
            confidence=to_float(data.get("confidence")),
            anomalies=anomalies,
            severity=report_sev,
            recommendations=recommendations,
        )
