"""Tests for chemvision.audit.report_generator module."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from chemvision.audit.degradation import DegradationResult, ReliabilityEnvelope
from chemvision.audit.matrix import CapabilityMatrix, MatrixConfig
from chemvision.audit.report_generator import (
    AuditReportGenerator,
    _accuracy_recommendation,
    _fmt_pct,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_matrix(tmp_path: Path) -> CapabilityMatrix:
    matrix = CapabilityMatrix(MatrixConfig(output_dir=tmp_path))
    # Inject some scores
    matrix._cells[("counting", "easy")].update(True)
    matrix._cells[("counting", "easy")].update(True)
    matrix._cells[("counting", "medium")].update(False)
    matrix._cells[("spatial_reasoning", "easy")].update(True)
    return matrix


def _make_envelope() -> ReliabilityEnvelope:
    results = {
        "gaussian_noise": DegradationResult(
            degradation_type="gaussian_noise",
            param_name="sigma",
            param_unit="px intensity",
            critical_param=45.0,
            accuracy_at_critical=0.70,
            param_clean=0.0,
            param_max_degradation=100.0,
            n_model_calls=160,
        ),
        "jpeg_compression": DegradationResult(
            degradation_type="jpeg_compression",
            param_name="quality",
            param_unit="JPEG quality",
            critical_param=55.0,
            accuracy_at_critical=0.70,
            param_clean=95.0,
            param_max_degradation=1.0,
            n_model_calls=160,
        ),
    }
    return ReliabilityEnvelope(
        model_name="test-vlm",
        threshold=0.7,
        results=results,
        evaluated_at="2025-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# _accuracy_recommendation helper
# ---------------------------------------------------------------------------


def test_accuracy_recommendation_production_ready() -> None:
    _, rec = _accuracy_recommendation(0.90)
    assert "Production" in rec


def test_accuracy_recommendation_supervised() -> None:
    _, rec = _accuracy_recommendation(0.75)
    assert "Supervised" in rec


def test_accuracy_recommendation_research() -> None:
    _, rec = _accuracy_recommendation(0.55)
    assert "Research" in rec


def test_accuracy_recommendation_not_recommended() -> None:
    _, rec = _accuracy_recommendation(0.20)
    assert "Not recommended" in rec


def test_accuracy_recommendation_nan() -> None:
    _, rec = _accuracy_recommendation(math.nan)
    assert "No samples" in rec


def test_fmt_pct_normal() -> None:
    assert _fmt_pct(0.875) == "88%"


def test_fmt_pct_nan() -> None:
    assert _fmt_pct(math.nan) == "n/a"


# ---------------------------------------------------------------------------
# AuditReportGenerator.generate
# ---------------------------------------------------------------------------


def test_generate_creates_report_file(tmp_path: Path) -> None:
    matrix = _make_matrix(tmp_path)
    envelope = _make_envelope()
    gen = AuditReportGenerator(matrix, envelope)
    report_path = gen.generate(output_dir=tmp_path)
    assert report_path.exists()
    assert report_path.suffix == ".md"
    assert report_path.stat().st_size > 0


def test_generate_report_contains_model_name(tmp_path: Path) -> None:
    matrix = _make_matrix(tmp_path)
    envelope = _make_envelope()
    gen = AuditReportGenerator(matrix, envelope)
    report_path = gen.generate(output_dir=tmp_path)
    content = report_path.read_text()
    assert "test-vlm" in content


def test_generate_report_contains_heatmap_base64(tmp_path: Path) -> None:
    matrix = _make_matrix(tmp_path)
    gen = AuditReportGenerator(matrix)
    report_path = gen.generate(output_dir=tmp_path)
    content = report_path.read_text()
    assert "data:image/png;base64," in content


def test_generate_report_contains_all_task_types(tmp_path: Path) -> None:
    matrix = _make_matrix(tmp_path)
    gen = AuditReportGenerator(matrix)
    report_path = gen.generate(output_dir=tmp_path)
    content = report_path.read_text()
    for task in CapabilityMatrix.TASK_TYPES:
        assert task in content


def test_generate_report_contains_degradation_table(tmp_path: Path) -> None:
    matrix = _make_matrix(tmp_path)
    envelope = _make_envelope()
    gen = AuditReportGenerator(matrix, envelope)
    report_path = gen.generate(output_dir=tmp_path)
    content = report_path.read_text()
    assert "gaussian_noise" in content
    assert "jpeg_compression" in content


def test_generate_report_contains_recommendations(tmp_path: Path) -> None:
    matrix = _make_matrix(tmp_path)
    gen = AuditReportGenerator(matrix)
    report_path = gen.generate(output_dir=tmp_path)
    content = report_path.read_text()
    assert "Deployment Recommendations" in content


def test_generate_without_envelope_skips_degradation_section(tmp_path: Path) -> None:
    matrix = _make_matrix(tmp_path)
    gen = AuditReportGenerator(matrix, envelope=None)
    report_path = gen.generate(output_dir=tmp_path)
    content = report_path.read_text()
    # Section 3 header should not appear
    assert "Robustness to Image Degradations" not in content


def test_generate_uses_prebuilt_heatmap(tmp_path: Path) -> None:
    """When heatmap_path is provided, generate() should use it directly."""
    matrix = _make_matrix(tmp_path)
    # Pre-export heatmap
    heatmap_path = matrix.export_heatmap(tmp_path)
    original_mtime = heatmap_path.stat().st_mtime

    gen = AuditReportGenerator(matrix)
    gen.generate(output_dir=tmp_path, heatmap_path=heatmap_path)

    # Heatmap file should not have been re-generated (same mtime)
    assert heatmap_path.stat().st_mtime == original_mtime
