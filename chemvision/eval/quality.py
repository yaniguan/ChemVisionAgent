"""AI quality scoring framework.

Provides a composite quality score (0–100) across five dimensions:

  1. **Accuracy**       — Are the extracted values correct?
  2. **Calibration**    — Do confidence scores match actual accuracy?
  3. **Completeness**   — Are all requested fields populated (non-null)?
  4. **Consistency**    — Do repeated runs on the same input agree?
  5. **Latency**        — Is the response fast enough for the use case?

Each dimension gets a 0–20 sub-score. The composite score is the sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DimensionScore:
    """Score for one quality dimension."""

    name: str
    score: float                    # [0, 20]
    max_score: float = 20.0
    details: str = ""


@dataclass
class QualityReport:
    """Composite AI quality assessment."""

    dimensions: list[DimensionScore] = field(default_factory=list)
    composite_score: float = 0.0    # [0, 100]
    grade: str = "F"                # A/B/C/D/F

    def summary(self) -> str:
        lines = [f"AI Quality Score: {self.composite_score:.0f}/100 (Grade: {self.grade})"]
        for d in self.dimensions:
            bar = "█" * int(d.score) + "░" * int(d.max_score - d.score)
            lines.append(f"  {d.name:15s} {bar} {d.score:.1f}/{d.max_score:.0f}  {d.details}")
        return "\n".join(lines)


class AIQualityScorer:
    """Compute a composite quality score for a set of skill outputs.

    Example
    -------
    >>> scorer = AIQualityScorer()
    >>> scorer.add_result(correct=True, confidence=0.9, completeness=0.8, latency_ms=1200)
    >>> scorer.add_result(correct=False, confidence=0.3, completeness=1.0, latency_ms=800)
    >>> report = scorer.score()
    >>> report.grade
    'B'
    """

    def __init__(self, target_latency_ms: float = 5000.0) -> None:
        self._target_latency = target_latency_ms
        self._correct: list[bool] = []
        self._confidences: list[float] = []
        self._completeness: list[float] = []
        self._latencies: list[float] = []
        self._repeat_groups: dict[str, list[str]] = {}  # key → list of outputs

    def add_result(
        self,
        correct: bool,
        confidence: float = 0.5,
        completeness: float = 1.0,
        latency_ms: float = 0.0,
    ) -> None:
        self._correct.append(correct)
        self._confidences.append(confidence)
        self._completeness.append(completeness)
        self._latencies.append(latency_ms)

    def add_consistency_pair(self, key: str, output: str) -> None:
        """Register repeated runs of the same input for consistency scoring."""
        self._repeat_groups.setdefault(key, []).append(output)

    def score(self) -> QualityReport:
        dims: list[DimensionScore] = []

        # 1. Accuracy (0–20)
        if self._correct:
            acc = sum(self._correct) / len(self._correct)
            acc_score = acc * 20
            dims.append(DimensionScore("Accuracy", acc_score, details=f"{acc:.0%} correct"))
        else:
            dims.append(DimensionScore("Accuracy", 0.0, details="no data"))

        # 2. Calibration (0–20): low ECE = high score
        if self._confidences and self._correct:
            confs = np.array(self._confidences)
            accs = np.array([1.0 if c else 0.0 for c in self._correct])
            ece = self._compute_ece(confs, accs)
            cal_score = max(0, (1 - ece * 5)) * 20  # ECE=0 → 20, ECE≥0.2 → 0
            dims.append(DimensionScore("Calibration", cal_score, details=f"ECE={ece:.3f}"))
        else:
            dims.append(DimensionScore("Calibration", 10.0, details="no confidence data"))

        # 3. Completeness (0–20)
        if self._completeness:
            comp = float(np.mean(self._completeness))
            comp_score = comp * 20
            dims.append(DimensionScore("Completeness", comp_score, details=f"{comp:.0%} fields filled"))
        else:
            dims.append(DimensionScore("Completeness", 0.0, details="no data"))

        # 4. Consistency (0–20): agreement across repeated runs
        if self._repeat_groups:
            agreements = []
            for key, outputs in self._repeat_groups.items():
                if len(outputs) >= 2:
                    # Fraction of pairs that agree
                    n = len(outputs)
                    agree = sum(
                        1 for i in range(n) for j in range(i + 1, n)
                        if outputs[i] == outputs[j]
                    )
                    total_pairs = n * (n - 1) / 2
                    agreements.append(agree / total_pairs if total_pairs > 0 else 1.0)
            cons = float(np.mean(agreements)) if agreements else 1.0
            cons_score = cons * 20
            dims.append(DimensionScore("Consistency", cons_score, details=f"{cons:.0%} agreement"))
        else:
            dims.append(DimensionScore("Consistency", 15.0, details="no repeat data"))

        # 5. Latency (0–20): fraction within target
        if self._latencies:
            within = sum(1 for l in self._latencies if l <= self._target_latency) / len(self._latencies)
            lat_score = within * 20
            p50 = float(np.percentile(self._latencies, 50))
            dims.append(DimensionScore("Latency", lat_score, details=f"p50={p50:.0f}ms, {within:.0%} within target"))
        else:
            dims.append(DimensionScore("Latency", 10.0, details="no data"))

        composite = sum(d.score for d in dims)
        grade = self._grade(composite)

        return QualityReport(dimensions=dims, composite_score=composite, grade=grade)

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 90:
            return "A"
        if score >= 75:
            return "B"
        if score >= 60:
            return "C"
        if score >= 40:
            return "D"
        return "F"

    def _compute_ece(self, confs: np.ndarray, accs: np.ndarray, n_bins: int = 10) -> float:
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confs > bins[i]) & (confs <= bins[i + 1])
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / len(confs)) * abs(accs[mask].mean() - confs[mask].mean())
        return float(ece)
