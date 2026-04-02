"""Confidence calibration via isotonic regression and Platt scaling.

VLM skill confidence scores are often poorly calibrated (overconfident or
underconfident). This module learns a mapping from raw confidence → calibrated
probability using held-out validation data.

Methods
-------
1. Isotonic regression (non-parametric, monotone): best for 100+ samples.
2. Platt scaling (logistic sigmoid): works with fewer samples.

After calibrating, skill output confidences become *real probabilities*:
a confidence of 0.8 means the skill is correct 80% of the time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibrationResult:
    """Output of calibration fitting."""

    method: str                                # "isotonic" or "platt"
    n_train_samples: int = 0
    ece_before: float = 0.0                    # ECE on training set, before calibration
    ece_after: float = 0.0                     # ECE on training set, after calibration
    improvement_pct: float = 0.0
    params: dict[str, float] = field(default_factory=dict)  # Platt: {a, b}


class ConfidenceCalibrator:
    """Map raw VLM confidence → calibrated probability.

    Example
    -------
    >>> cal = ConfidenceCalibrator()
    >>> cal.fit(raw_confs=[0.9, 0.8, 0.7, 0.6], correct=[1, 1, 0, 0])
    >>> cal.calibrate(0.85)
    0.72
    """

    def __init__(self, method: str = "isotonic", n_bins: int = 10) -> None:
        if method not in ("isotonic", "platt"):
            raise ValueError(f"Unknown method: {method!r}")
        self._method = method
        self._n_bins = n_bins
        self._fitted = False

        # Isotonic state
        self._iso_x: np.ndarray | None = None
        self._iso_y: np.ndarray | None = None

        # Platt state
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        raw_confs: list[float] | np.ndarray,
        correct: list[int] | np.ndarray,
    ) -> CalibrationResult:
        """Fit the calibrator on labelled (raw_confidence, was_correct) pairs.

        Parameters
        ----------
        raw_confs:
            Skill output confidence values [0, 1].
        correct:
            Binary labels: 1 if the prediction was correct, 0 otherwise.
        """
        confs = np.array(raw_confs, dtype=np.float64)
        labels = np.array(correct, dtype=np.float64)

        ece_before = self._ece(confs, labels)

        if self._method == "isotonic":
            self._fit_isotonic(confs, labels)
        else:
            self._fit_platt(confs, labels)

        calibrated = np.array([self.calibrate(c) for c in confs])
        ece_after = self._ece(calibrated, labels)
        improvement = ((ece_before - ece_after) / max(ece_before, 1e-9)) * 100

        self._fitted = True
        return CalibrationResult(
            method=self._method,
            n_train_samples=len(confs),
            ece_before=ece_before,
            ece_after=ece_after,
            improvement_pct=improvement,
            params={"a": self._platt_a, "b": self._platt_b} if self._method == "platt" else {},
        )

    # ------------------------------------------------------------------
    # Calibrate
    # ------------------------------------------------------------------

    def calibrate(self, raw_confidence: float) -> float:
        """Return calibrated probability for a raw confidence value."""
        if not self._fitted:
            return raw_confidence  # pass-through before fitting

        if self._method == "isotonic":
            return self._apply_isotonic(raw_confidence)
        else:
            return self._apply_platt(raw_confidence)

    def calibrate_batch(self, confs: list[float]) -> list[float]:
        return [self.calibrate(c) for c in confs]

    # ------------------------------------------------------------------
    # Isotonic regression (Pool Adjacent Violators Algorithm)
    # ------------------------------------------------------------------

    def _fit_isotonic(self, confs: np.ndarray, labels: np.ndarray) -> None:
        """Non-parametric monotone calibration via PAV algorithm."""
        order = np.argsort(confs)
        x_sorted = confs[order]
        y_sorted = labels[order].astype(float)

        # Pool Adjacent Violators
        n = len(y_sorted)
        result = y_sorted.copy()
        weight = np.ones(n, dtype=float)
        i = 0
        while i < n - 1:
            if result[i] > result[i + 1]:
                # Pool: merge with next
                total = result[i] * weight[i] + result[i + 1] * weight[i + 1]
                w = weight[i] + weight[i + 1]
                result[i] = total / w
                weight[i] = w
                result = np.delete(result, i + 1)
                weight = np.delete(weight, i + 1)
                x_sorted = np.delete(x_sorted, i + 1)
                n -= 1
                if i > 0:
                    i -= 1
            else:
                i += 1

        self._iso_x = x_sorted
        self._iso_y = result

    def _apply_isotonic(self, conf: float) -> float:
        if self._iso_x is None or self._iso_y is None:
            return conf
        idx = np.searchsorted(self._iso_x, conf, side="right") - 1
        idx = max(0, min(idx, len(self._iso_y) - 1))
        return float(np.clip(self._iso_y[idx], 0.0, 1.0))

    # ------------------------------------------------------------------
    # Platt scaling (logistic sigmoid)
    # ------------------------------------------------------------------

    def _fit_platt(self, confs: np.ndarray, labels: np.ndarray) -> None:
        """Fit a logistic sigmoid: P(correct) = 1 / (1 + exp(a * conf + b))."""
        # Simple gradient descent (no scipy dependency)
        a, b = 1.0, 0.0
        lr = 0.01
        for _ in range(1000):
            logits = a * confs + b
            probs = 1.0 / (1.0 + np.exp(-logits))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            # Binary cross-entropy gradient
            grad_a = np.mean((probs - labels) * confs)
            grad_b = np.mean(probs - labels)
            a -= lr * grad_a
            b -= lr * grad_b
        self._platt_a = float(a)
        self._platt_b = float(b)

    def _apply_platt(self, conf: float) -> float:
        logit = self._platt_a * conf + self._platt_b
        return float(1.0 / (1.0 + np.exp(-logit)))

    # ------------------------------------------------------------------
    # ECE helper
    # ------------------------------------------------------------------

    def _ece(self, confs: np.ndarray, labels: np.ndarray) -> float:
        bins = np.linspace(0, 1, self._n_bins + 1)
        ece = 0.0
        for i in range(self._n_bins):
            mask = (confs > bins[i]) & (confs <= bins[i + 1])
            n_bin = mask.sum()
            if n_bin == 0:
                continue
            ece += (n_bin / len(confs)) * abs(labels[mask].mean() - confs[mask].mean())
        return float(ece)
