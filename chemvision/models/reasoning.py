"""Chain-of-vision reasoning wrapper enforcing structured 3-step output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from PIL.Image import Image

from chemvision.models.base import BaseVisionModel


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates (top-left origin)."""

    region_id: str
    x: float
    y: float
    width: float
    height: float
    label: str = ""


@dataclass
class LocalizationResult:
    """Step 1 output: detected regions of interest with bounding boxes."""

    boxes: list[BoundingBox] = field(default_factory=list)
    raw: str = ""


@dataclass
class AnalysisResult:
    """Step 2 output: detailed per-region descriptions."""

    descriptions: dict[str, str] = field(default_factory=dict)
    raw: str = ""


@dataclass
class ConclusionResult:
    """Step 3 output: structured numerical findings."""

    findings: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class ChainOfVisionOutput:
    """Full 3-step reasoning trace for a single image query."""

    localization: LocalizationResult
    analysis: AnalysisResult
    conclusion: ConclusionResult
    raw_steps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a scientific image analysis assistant specialised in chemistry and materials science.
Reason step-by-step using the following **strict** format:

STEP 1 - LOCALIZE:
Return a JSON array of bounding boxes for every region of interest.
Each entry must have keys: "region_id" (str), "x" (px), "y" (px),
"width" (px), "height" (px), "label" (str).
Wrap the array in <localize>...</localize> tags.

STEP 2 - ANALYZE:
For each region_id produced in Step 1, write a detailed scientific description.
Format as a JSON object mapping region_id → description string.
Wrap it in <analyze>...</analyze> tags.

STEP 3 - CONCLUDE:
Provide structured numerical findings as a JSON object.
Keys are measurement names; values are numbers or arrays of numbers.
Wrap it in <conclude>...</conclude> tags.

Do not skip any step. Do not output text outside the three tagged blocks.\
"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ChainOfVisionReasoning:
    """Prompting wrapper that enforces a 3-step chain-of-vision reasoning output.

    Wraps any :class:`~chemvision.models.base.BaseVisionModel` and injects
    :data:`_SYSTEM_PROMPT` to force the model to produce:

    1. **LOCALIZE** – bounding boxes around regions of interest as JSON.
    2. **ANALYZE** – detailed description of each located region.
    3. **CONCLUDE** – structured numerical findings.

    Each step is delimited by XML-style tags and parsed independently.
    Malformed JSON in any step is recovered gracefully.

    Example
    -------
    >>> cov = ChainOfVisionReasoning(model)
    >>> out = cov.reason(image, "What are the peak positions in this spectrum?")
    >>> out.conclusion.findings
    {'peak_wavenumbers_cm1': [1720.0, 2960.0]}
    >>> out.localization.boxes[0].label
    'carbonyl_peak'
    """

    def __init__(self, model: BaseVisionModel) -> None:
        self.model = model

    def reason(self, image: Image, question: str) -> ChainOfVisionOutput:
        """Run 3-step chain-of-vision reasoning on *image*.

        Parameters
        ----------
        image:
            RGB PIL image to analyse.
        question:
            Scientific question about the image.

        Returns
        -------
        ChainOfVisionOutput
            Parsed localization, analysis, and conclusion.
        """
        prompt = f"{_SYSTEM_PROMPT}\n\nQuestion: {question}"
        raw = self.model.generate(image, prompt)
        result = self._parse(raw)
        result.raw_steps = [raw]
        return result

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, raw: str) -> ChainOfVisionOutput:
        return ChainOfVisionOutput(
            localization=self._parse_localize(raw),
            analysis=self._parse_analyze(raw),
            conclusion=self._parse_conclude(raw),
        )

    def _parse_localize(self, text: str) -> LocalizationResult:
        raw = self._extract_tag(text, "localize")
        if not raw:
            return LocalizationResult(raw=raw)
        try:
            data = json.loads(raw)
            boxes = [
                BoundingBox(
                    region_id=str(entry.get("region_id", "")),
                    x=float(entry.get("x", 0)),
                    y=float(entry.get("y", 0)),
                    width=float(entry.get("width", 0)),
                    height=float(entry.get("height", 0)),
                    label=str(entry.get("label", "")),
                )
                for entry in data
                if isinstance(entry, dict)
            ]
        except (json.JSONDecodeError, TypeError, ValueError):
            boxes = []
        return LocalizationResult(boxes=boxes, raw=raw)

    def _parse_analyze(self, text: str) -> AnalysisResult:
        raw = self._extract_tag(text, "analyze")
        if not raw:
            return AnalysisResult(raw=raw)
        try:
            data = json.loads(raw)
            descriptions = (
                {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
            )
        except json.JSONDecodeError:
            descriptions = {}
        return AnalysisResult(descriptions=descriptions, raw=raw)

    def _parse_conclude(self, text: str) -> ConclusionResult:
        raw = self._extract_tag(text, "conclude")
        if not raw:
            return ConclusionResult(raw=raw)
        try:
            data = json.loads(raw)
            findings = data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            findings = {}
        return ConclusionResult(findings=findings, raw=raw)

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        """Extract content between ``<tag>…</tag>``, stripped of whitespace."""
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else ""
