"""Tests for chemvision.audit.matrix module."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from chemvision.audit.matrix import (
    CapabilityMatrix,
    CellResult,
    Difficulty,
    MatrixConfig,
    TaskType,
)


# ---------------------------------------------------------------------------
# MatrixConfig schema
# ---------------------------------------------------------------------------


def test_matrix_config_defaults() -> None:
    cfg = MatrixConfig()
    assert cfg.score_fn == "substring"
    assert cfg.output_dir == Path("reports/")


def test_matrix_config_custom() -> None:
    cfg = MatrixConfig(score_fn="exact", output_dir=Path("/tmp/out"))
    assert cfg.score_fn == "exact"
    assert cfg.output_dir == Path("/tmp/out")


# ---------------------------------------------------------------------------
# CellResult
# ---------------------------------------------------------------------------


def test_cell_result_accuracy_empty() -> None:
    cell = CellResult("counting", "easy")
    assert math.isnan(cell.accuracy)


def test_cell_result_accuracy_after_updates() -> None:
    cell = CellResult("counting", "easy")
    cell.update(correct=True)
    cell.update(correct=True)
    cell.update(correct=False)
    assert cell.accuracy == pytest.approx(2 / 3)


def test_cell_result_update_increments_totals() -> None:
    cell = CellResult("anomaly_detection", "hard")
    cell.update(True)
    cell.update(False)
    assert cell.num_total == 2
    assert cell.num_correct == 1


# ---------------------------------------------------------------------------
# CapabilityMatrix initialisation
# ---------------------------------------------------------------------------


def test_capability_matrix_has_correct_dimensions() -> None:
    matrix = CapabilityMatrix()
    assert len(matrix.TASK_TYPES) == 5
    assert len(matrix.DIFFICULTIES) == 3


def test_capability_matrix_task_types_are_correct() -> None:
    matrix = CapabilityMatrix()
    expected = {
        "spatial_reasoning",
        "counting",
        "cross_image_comparison",
        "anomaly_detection",
        "caption_validation",
    }
    assert set(matrix.TASK_TYPES) == expected


def test_capability_matrix_all_cells_start_as_nan() -> None:
    matrix = CapabilityMatrix()
    for t in matrix.TASK_TYPES:
        for d in matrix.DIFFICULTIES:
            assert math.isnan(matrix.get_score(t, d))


def test_capability_matrix_to_array_shape() -> None:
    matrix = CapabilityMatrix()
    arr = matrix.to_array()
    assert len(arr) == 5
    assert all(len(row) == 3 for row in arr)


def test_capability_matrix_to_dict_structure() -> None:
    matrix = CapabilityMatrix()
    d = matrix.to_dict()
    assert set(d.keys()) == set(matrix.TASK_TYPES)
    for task_scores in d.values():
        assert set(task_scores.keys()) == set(matrix.DIFFICULTIES)


# ---------------------------------------------------------------------------
# _score_answer
# ---------------------------------------------------------------------------


def test_score_answer_substring_match() -> None:
    assert CapabilityMatrix._score_answer(
        "The peak is at 1720 cm-1 for carbonyl", "1720 cm-1", strategy="substring"
    )


def test_score_answer_exact_match() -> None:
    assert CapabilityMatrix._score_answer(
        "1720 cm-1", "1720 cm-1", strategy="exact"
    )


def test_score_answer_exact_fails_substring() -> None:
    assert not CapabilityMatrix._score_answer(
        "The answer is 1720 cm-1 approximately", "1720 cm-1", strategy="exact"
    )


def test_score_answer_case_insensitive() -> None:
    assert CapabilityMatrix._score_answer("PEAK AT 1720", "peak at 1720", strategy="exact")


def test_score_answer_wrong_answer() -> None:
    assert not CapabilityMatrix._score_answer("3000 cm-1", "1720 cm-1", strategy="substring")


# ---------------------------------------------------------------------------
# run_evaluation with a mock model + tmp images
# ---------------------------------------------------------------------------


def _make_record(tmp_path: Path, task_type: str, difficulty: str, answer: str, idx: int):
    """Create a minimal ImageRecord backed by a real temporary image."""
    from chemvision.data.schema import ImageRecord, ImageDomain

    img_path = tmp_path / f"img_{idx}.png"
    PILImage.new("RGB", (32, 32), color=(idx * 30 % 256, 0, 0)).save(img_path)
    return ImageRecord(
        id=f"rec_{idx}",
        image_path=img_path,
        domain=ImageDomain.SPECTROSCOPY,
        question="What is the peak wavenumber?",
        answer=answer,
        difficulty=difficulty,
        metadata={"task_type": task_type},
    )


def test_run_evaluation_fills_cells(tmp_path: Path) -> None:
    records = [
        _make_record(tmp_path, "counting", "easy", "three peaks", 0),
        _make_record(tmp_path, "counting", "easy", "three peaks", 1),
        _make_record(tmp_path, "counting", "hard", "five bands", 2),
    ]
    mock_model = MagicMock()
    mock_model.generate.return_value = "I see three peaks in the spectrum"

    matrix = CapabilityMatrix().run_evaluation(mock_model, records)

    assert matrix.get_cell("counting", "easy").num_total == 2
    assert matrix.get_cell("counting", "hard").num_total == 1


def test_run_evaluation_scores_correctly(tmp_path: Path) -> None:
    records = [
        _make_record(tmp_path, "counting", "easy", "1720", 0),  # correct
        _make_record(tmp_path, "counting", "easy", "1720", 1),  # correct
        _make_record(tmp_path, "counting", "easy", "3300", 2),  # wrong answer
    ]
    mock_model = MagicMock()
    # Model always predicts "1720 cm-1"
    mock_model.generate.return_value = "The peak is at 1720 cm-1"

    matrix = CapabilityMatrix().run_evaluation(mock_model, records)

    cell = matrix.get_cell("counting", "easy")
    assert cell.num_correct == 2  # "1720" in output, but "3300" is not
    assert cell.num_total == 3
    assert cell.accuracy == pytest.approx(2 / 3)


def test_run_evaluation_unknown_task_type_uses_fallback(tmp_path: Path) -> None:
    from chemvision.data.schema import ImageRecord, ImageDomain

    img_path = tmp_path / "img.png"
    PILImage.new("RGB", (32, 32)).save(img_path)
    record = ImageRecord(
        id="r1",
        image_path=img_path,
        domain=ImageDomain.SPECTROSCOPY,
        question="q",
        answer="a",
        difficulty="easy",
        metadata={},  # no task_type key
    )
    mock_model = MagicMock()
    mock_model.generate.return_value = "a"

    cfg = MatrixConfig(unknown_task_type="spatial_reasoning")
    matrix = CapabilityMatrix(cfg).run_evaluation(mock_model, [record])

    assert matrix.get_cell("spatial_reasoning", "easy").num_total == 1


def test_run_evaluation_returns_self(tmp_path: Path) -> None:
    records = [_make_record(tmp_path, "counting", "easy", "x", 0)]
    mock_model = MagicMock()
    mock_model.generate.return_value = "x"

    matrix = CapabilityMatrix()
    result = matrix.run_evaluation(mock_model, records)
    assert result is matrix


# ---------------------------------------------------------------------------
# export_heatmap (monkeypatched matplotlib so no display required)
# ---------------------------------------------------------------------------


def test_export_heatmap_creates_file(tmp_path: Path) -> None:
    matrix = CapabilityMatrix(MatrixConfig(output_dir=tmp_path))

    # Inject some scores manually
    matrix._cells[("counting", "easy")].update(True)
    matrix._cells[("counting", "easy")].update(False)

    out = matrix.export_heatmap(tmp_path)
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 0
