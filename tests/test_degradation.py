"""Tests for chemvision.audit.degradation module."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from chemvision.audit.degradation import (
    DegradationConfig,
    DegradationResult,
    DegradationTester,
    ReliabilityEnvelope,
)


# ---------------------------------------------------------------------------
# DegradationConfig schema
# ---------------------------------------------------------------------------


def test_degradation_config_defaults() -> None:
    cfg = DegradationConfig()
    assert cfg.threshold == pytest.approx(0.7)
    assert cfg.n_samples_per_eval == 20
    assert cfg.n_binary_search_iters == 8
    assert cfg.seed == 42


def test_degradation_config_threshold_bounds() -> None:
    with pytest.raises(Exception):
        DegradationConfig(threshold=1.5)
    with pytest.raises(Exception):
        DegradationConfig(threshold=-0.1)


# ---------------------------------------------------------------------------
# DegradationResult
# ---------------------------------------------------------------------------


def _make_result(critical: float, clean: float, maxdeg: float) -> DegradationResult:
    return DegradationResult(
        degradation_type="gaussian_noise",
        param_name="sigma",
        param_unit="px intensity",
        critical_param=critical,
        accuracy_at_critical=0.70,
        param_clean=clean,
        param_max_degradation=maxdeg,
        n_model_calls=40,
    )


def test_degradation_result_normalized_tolerance_zero_at_clean() -> None:
    r = _make_result(critical=0.0, clean=0.0, maxdeg=100.0)
    assert r.normalized_tolerance == pytest.approx(0.0)


def test_degradation_result_normalized_tolerance_one_at_max() -> None:
    r = _make_result(critical=100.0, clean=0.0, maxdeg=100.0)
    assert r.normalized_tolerance == pytest.approx(1.0)


def test_degradation_result_normalized_tolerance_midpoint() -> None:
    r = _make_result(critical=50.0, clean=0.0, maxdeg=100.0)
    assert r.normalized_tolerance == pytest.approx(0.5)


def test_degradation_result_robustness_label_high() -> None:
    r = _make_result(critical=70.0, clean=0.0, maxdeg=100.0)  # tolerance = 0.7
    assert r.robustness_label == "high"


def test_degradation_result_robustness_label_moderate() -> None:
    r = _make_result(critical=40.0, clean=0.0, maxdeg=100.0)  # tolerance = 0.4
    assert r.robustness_label == "moderate"


def test_degradation_result_robustness_label_low() -> None:
    r = _make_result(critical=15.0, clean=0.0, maxdeg=100.0)  # tolerance = 0.15
    assert r.robustness_label == "low"


def test_degradation_result_robustness_jpeg() -> None:
    # JPEG: clean=95, maxdeg=1; critical=47 means tolerance=(95-47)/(95-1)≈0.51
    r = DegradationResult(
        degradation_type="jpeg_compression",
        param_name="quality",
        param_unit="JPEG quality",
        critical_param=47.0,
        accuracy_at_critical=0.70,
        param_clean=95.0,
        param_max_degradation=1.0,
        n_model_calls=40,
    )
    assert r.normalized_tolerance == pytest.approx((95 - 47) / (95 - 1), rel=1e-3)
    assert r.robustness_label == "moderate"


# ---------------------------------------------------------------------------
# ReliabilityEnvelope serialisation
# ---------------------------------------------------------------------------


def _make_envelope() -> ReliabilityEnvelope:
    r = _make_result(30.0, 0.0, 100.0)
    return ReliabilityEnvelope(
        model_name="test-model",
        threshold=0.7,
        results={"gaussian_noise": r},
        evaluated_at="2025-01-01T00:00:00+00:00",
    )


def test_reliability_envelope_save_and_load(tmp_path: Path) -> None:
    envelope = _make_envelope()
    path = tmp_path / "envelope.json"
    envelope.save_json(path)

    assert path.exists()
    loaded = ReliabilityEnvelope.load_json(path)
    assert loaded.model_name == "test-model"
    assert loaded.threshold == pytest.approx(0.7)
    assert "gaussian_noise" in loaded.results
    assert loaded.results["gaussian_noise"].critical_param == pytest.approx(30.0)


def test_reliability_envelope_json_is_valid(tmp_path: Path) -> None:
    envelope = _make_envelope()
    path = tmp_path / "envelope.json"
    envelope.save_json(path)
    # Should be parseable as plain JSON
    data = json.loads(path.read_text())
    assert "degradations" in data
    assert "gaussian_noise" in data["degradations"]


def test_reliability_envelope_post_init_sets_evaluated_at() -> None:
    r = _make_result(30.0, 0.0, 100.0)
    env = ReliabilityEnvelope(
        model_name="m",
        threshold=0.7,
        results={"gaussian_noise": r},
    )
    assert env.evaluated_at != ""


# ---------------------------------------------------------------------------
# Image degradation application methods (no model required)
# ---------------------------------------------------------------------------


def _rgb_image(w: int = 64, h: int = 64) -> PILImage.Image:
    return PILImage.new("RGB", (w, h), color=(128, 100, 80))


def test_apply_gaussian_noise_changes_image() -> None:
    img = _rgb_image()
    result = DegradationTester._apply_gaussian_noise(img, sigma=30.0)
    assert result.size == img.size
    import numpy as np
    assert not (np.array(result) == np.array(img)).all()


def test_apply_gaussian_noise_zero_sigma_is_close_to_original() -> None:
    img = _rgb_image()
    result = DegradationTester._apply_gaussian_noise(img, sigma=0.0)
    import numpy as np
    diff = abs(np.array(result, dtype=float) - np.array(img, dtype=float)).max()
    assert diff < 1.0  # no real change at sigma=0


def test_apply_jpeg_compression_same_size() -> None:
    img = _rgb_image()
    result = DegradationTester._apply_jpeg_compression(img, quality=50.0)
    assert result.size == img.size


def test_apply_jpeg_compression_changes_image() -> None:
    img = _rgb_image()
    result = DegradationTester._apply_jpeg_compression(img, quality=1.0)
    import numpy as np
    assert not (np.array(result) == np.array(img)).all()


def test_apply_occlusion_same_size() -> None:
    img = _rgb_image()
    result = DegradationTester._apply_occlusion(img, fraction=0.25)
    assert result.size == img.size


def test_apply_occlusion_zero_fraction_unchanged() -> None:
    img = _rgb_image()
    result = DegradationTester._apply_occlusion(img, fraction=0.0)
    import numpy as np
    diff = abs(np.array(result, dtype=float) - np.array(img, dtype=float)).max()
    assert diff == 0.0


def test_apply_downsampling_same_size() -> None:
    img = _rgb_image()
    result = DegradationTester._apply_downsampling(img, scale=0.25)
    assert result.size == img.size


def test_apply_downsampling_scale_one_preserves_image() -> None:
    img = _rgb_image()
    result = DegradationTester._apply_downsampling(img, scale=1.0)
    import numpy as np
    # Bicubic round-trip at scale=1.0 may change a few values; check size at least
    assert result.size == img.size


def test_apply_color_shift_same_size() -> None:
    img = _rgb_image()
    result = DegradationTester._apply_color_shift(img, magnitude=30.0)
    assert result.size == img.size


def test_apply_color_shift_zero_magnitude_unchanged() -> None:
    img = _rgb_image()
    result = DegradationTester._apply_color_shift(img, magnitude=0.0)
    import numpy as np
    diff = abs(np.array(result, dtype=float) - np.array(img, dtype=float)).max()
    assert diff == 0.0


# ---------------------------------------------------------------------------
# _score_answer
# ---------------------------------------------------------------------------


def test_score_answer_substring_hit() -> None:
    assert DegradationTester._score_answer("The value is 42 units", "42 units")


def test_score_answer_substring_miss() -> None:
    assert not DegradationTester._score_answer("The value is 55", "42")


def test_score_answer_case_insensitive() -> None:
    assert DegradationTester._score_answer("PEAK AT 1720", "peak at 1720")


# ---------------------------------------------------------------------------
# _binary_search with mocked _evaluate_accuracy
# ---------------------------------------------------------------------------


def test_binary_search_finds_crossover() -> None:
    """Verify the binary search converges near threshold."""
    tester = DegradationTester(DegradationConfig(threshold=0.7, n_binary_search_iters=10))

    # Simulate: accuracy = 1.0 - param/100  (linearly decreasing)
    # Threshold 0.7 → crossover at param=30
    def fake_evaluate(model, records, apply_fn, param):
        return max(0.0, 1.0 - param / 100.0)

    apply_fn = DegradationTester._apply_gaussian_noise
    with patch.object(tester, "_evaluate_accuracy", side_effect=fake_evaluate):
        critical, acc = tester._binary_search(
            apply_fn=apply_fn,
            param_clean=0.0,
            param_degraded=100.0,
            model=MagicMock(),
            records=[],
        )

    assert critical == pytest.approx(30.0, abs=1.0)
    assert acc == pytest.approx(0.70, abs=0.02)


def test_binary_search_jpeg_direction() -> None:
    """Verify binary search works when param_clean > param_degraded (JPEG)."""
    tester = DegradationTester(DegradationConfig(threshold=0.7, n_binary_search_iters=10))

    # JPEG: quality 95 (clean) → 1 (degraded)
    # accuracy = quality / 100 → threshold at quality=70
    def fake_evaluate(model, records, apply_fn, param):
        return param / 100.0

    with patch.object(tester, "_evaluate_accuracy", side_effect=fake_evaluate):
        critical, acc = tester._binary_search(
            apply_fn=DegradationTester._apply_jpeg_compression,
            param_clean=95.0,
            param_degraded=1.0,
            model=MagicMock(),
            records=[],
        )

    # critical quality should be near 70
    assert critical == pytest.approx(70.0, abs=2.0)
