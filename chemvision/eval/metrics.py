"""Per-skill and aggregate evaluation metrics.

Goes beyond the existing CapabilityMatrix (accuracy only) to provide:
  - Per-skill precision / recall / F1
  - Confidence-error correlation (ECE — Expected Calibration Error)
  - Regression metrics for numeric outputs (MAE, RMSE, R²)
  - Latency percentiles (p50, p95, p99)

All metrics are computed from a list of (predicted, ground_truth, confidence,
latency_ms) tuples — no network calls or model inference required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SkillMetrics:
    """Evaluation metrics for one skill."""

    skill_name: str
    n_samples: int = 0

    # Classification metrics (exact/substring match)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Regression metrics (for numeric outputs like peak positions, MW, etc.)
    mae: float | None = None
    rmse: float | None = None
    r_squared: float | None = None
    max_error: float | None = None

    # Calibration metrics
    ece: float | None = None              # Expected Calibration Error
    mean_confidence: float | None = None
    confidence_accuracy_gap: float | None = None  # |mean_conf - accuracy|

    # Latency
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class MetricsSuiteResult:
    """Aggregate metrics across all skills."""

    per_skill: dict[str, SkillMetrics] = field(default_factory=dict)
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    overall_accuracy: float = 0.0
    overall_ece: float | None = None
    n_total: int = 0


@dataclass
class EvalSample:
    """One evaluation sample: predicted vs ground truth."""

    skill_name: str
    predicted: str | float
    ground_truth: str | float
    confidence: float | None = None
    latency_ms: float | None = None
    is_numeric: bool = False


class MetricsSuite:
    """Compute comprehensive evaluation metrics from prediction/truth pairs.

    Example
    -------
    >>> suite = MetricsSuite()
    >>> suite.add("extract_spectrum_data", predicted="25.3", ground_truth="25.3", confidence=0.92)
    >>> suite.add("extract_spectrum_data", predicted="25.5", ground_truth="25.3", confidence=0.85)
    >>> result = suite.compute()
    >>> result.per_skill["extract_spectrum_data"].accuracy
    0.5
    """

    def __init__(self, n_calibration_bins: int = 10) -> None:
        self._samples: list[EvalSample] = []
        self._n_bins = n_calibration_bins

    def add(
        self,
        skill_name: str,
        predicted: str | float,
        ground_truth: str | float,
        confidence: float | None = None,
        latency_ms: float | None = None,
        is_numeric: bool = False,
    ) -> None:
        self._samples.append(EvalSample(
            skill_name=skill_name, predicted=predicted, ground_truth=ground_truth,
            confidence=confidence, latency_ms=latency_ms, is_numeric=is_numeric,
        ))

    def add_numeric(
        self,
        skill_name: str,
        predicted: float,
        ground_truth: float,
        confidence: float | None = None,
        latency_ms: float | None = None,
    ) -> None:
        self.add(skill_name, predicted, ground_truth, confidence, latency_ms, is_numeric=True)

    def compute(self) -> MetricsSuiteResult:
        result = MetricsSuiteResult()
        by_skill: dict[str, list[EvalSample]] = {}
        for s in self._samples:
            by_skill.setdefault(s.skill_name, []).append(s)

        all_correct = 0
        all_total = 0

        for skill_name, samples in by_skill.items():
            sm = SkillMetrics(skill_name=skill_name, n_samples=len(samples))

            # Classification: exact match (string comparison)
            correct = sum(
                1 for s in samples
                if str(s.predicted).strip().lower() == str(s.ground_truth).strip().lower()
            )
            sm.accuracy = correct / len(samples) if samples else 0.0
            # For binary-style: precision = recall = accuracy when treating as exact match
            sm.precision = sm.accuracy
            sm.recall = sm.accuracy
            sm.f1 = sm.accuracy  # exact match F1 = accuracy

            all_correct += correct
            all_total += len(samples)

            # Regression metrics (numeric samples)
            numeric = [s for s in samples if s.is_numeric]
            if numeric:
                preds = np.array([float(s.predicted) for s in numeric])
                truths = np.array([float(s.ground_truth) for s in numeric])
                errors = preds - truths
                sm.mae = float(np.mean(np.abs(errors)))
                sm.rmse = float(np.sqrt(np.mean(errors ** 2)))
                sm.max_error = float(np.max(np.abs(errors)))
                ss_res = np.sum(errors ** 2)
                ss_tot = np.sum((truths - np.mean(truths)) ** 2)
                sm.r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None

            # Calibration: Expected Calibration Error
            with_conf = [s for s in samples if s.confidence is not None]
            if with_conf:
                confs = np.array([s.confidence for s in with_conf])
                accs = np.array([
                    1.0 if str(s.predicted).strip().lower() == str(s.ground_truth).strip().lower() else 0.0
                    for s in with_conf
                ])
                sm.mean_confidence = float(np.mean(confs))
                sm.confidence_accuracy_gap = abs(sm.mean_confidence - sm.accuracy)
                sm.ece = self._compute_ece(confs, accs)

            # Latency percentiles
            latencies = [s.latency_ms for s in samples if s.latency_ms is not None]
            if latencies:
                lat_arr = np.array(latencies)
                sm.latency_p50_ms = float(np.percentile(lat_arr, 50))
                sm.latency_p95_ms = float(np.percentile(lat_arr, 95))
                sm.latency_p99_ms = float(np.percentile(lat_arr, 99))

            result.per_skill[skill_name] = sm

        result.n_total = all_total
        result.overall_accuracy = all_correct / all_total if all_total > 0 else 0.0

        # Macro F1 (unweighted average across skills)
        f1_values = [m.f1 for m in result.per_skill.values()]
        result.macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0

        # Weighted F1 (weighted by sample count)
        if all_total > 0:
            result.weighted_f1 = sum(
                m.f1 * m.n_samples / all_total for m in result.per_skill.values()
            )

        # Overall ECE
        all_confs = [s.confidence for s in self._samples if s.confidence is not None]
        all_accs = [
            1.0 if str(s.predicted).strip().lower() == str(s.ground_truth).strip().lower() else 0.0
            for s in self._samples if s.confidence is not None
        ]
        if all_confs:
            result.overall_ece = self._compute_ece(np.array(all_confs), np.array(all_accs))

        return result

    def _compute_ece(self, confidences: np.ndarray, accuracies: np.ndarray) -> float:
        """Expected Calibration Error: weighted average of |accuracy - confidence| per bin."""
        bin_boundaries = np.linspace(0, 1, self._n_bins + 1)
        ece = 0.0
        for i in range(self._n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            n_bin = mask.sum()
            if n_bin == 0:
                continue
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += (n_bin / len(confidences)) * abs(avg_acc - avg_conf)
        return float(ece)
