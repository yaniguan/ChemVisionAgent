"""Tests for chemvision.models.reasoning module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

from chemvision.models.reasoning import (
    AnalysisResult,
    BoundingBox,
    ChainOfVisionOutput,
    ChainOfVisionReasoning,
    ConclusionResult,
    LocalizationResult,
)


# ---------------------------------------------------------------------------
# Data structure defaults
# ---------------------------------------------------------------------------


def test_bounding_box_label_defaults_to_empty() -> None:
    bb = BoundingBox(region_id="r1", x=10.0, y=20.0, width=50.0, height=30.0)
    assert bb.label == ""


def test_localization_result_empty_defaults() -> None:
    lr = LocalizationResult()
    assert lr.boxes == []
    assert lr.raw == ""


def test_analysis_result_empty_defaults() -> None:
    ar = AnalysisResult()
    assert ar.descriptions == {}
    assert ar.raw == ""


def test_conclusion_result_empty_defaults() -> None:
    cr = ConclusionResult()
    assert cr.findings == {}
    assert cr.raw == ""


def test_chain_output_raw_steps_defaults_to_empty() -> None:
    out = ChainOfVisionOutput(
        localization=LocalizationResult(),
        analysis=AnalysisResult(),
        conclusion=ConclusionResult(),
    )
    assert out.raw_steps == []


# ---------------------------------------------------------------------------
# _extract_tag static method
# ---------------------------------------------------------------------------


def test_extract_tag_found() -> None:
    text = "some text <localize>[1,2,3]</localize> more"
    result = ChainOfVisionReasoning._extract_tag(text, "localize")
    assert result == "[1,2,3]"


def test_extract_tag_multiline() -> None:
    text = '<analyze>\n{"r1": "bright region"}\n</analyze>'
    result = ChainOfVisionReasoning._extract_tag(text, "analyze")
    assert result == '{"r1": "bright region"}'


def test_extract_tag_missing_returns_empty_string() -> None:
    result = ChainOfVisionReasoning._extract_tag("no tags here", "localize")
    assert result == ""


def test_extract_tag_strips_whitespace() -> None:
    text = '<conclude>  {"x": 1}  </conclude>'
    result = ChainOfVisionReasoning._extract_tag(text, "conclude")
    assert result == '{"x": 1}'


def test_extract_tag_nested_content_preserved() -> None:
    inner = json.dumps([{"region_id": "r1", "x": 0, "y": 0, "width": 10, "height": 10}])
    text = f"<localize>{inner}</localize>"
    result = ChainOfVisionReasoning._extract_tag(text, "localize")
    assert result == inner


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _make_reasoning() -> ChainOfVisionReasoning:
    return ChainOfVisionReasoning(MagicMock())


def test_parse_localize_valid_json() -> None:
    cov = _make_reasoning()
    boxes_json = json.dumps([
        {"region_id": "r1", "x": 10, "y": 20, "width": 50, "height": 30, "label": "peak"},
    ])
    result = cov._parse_localize(f"<localize>{boxes_json}</localize>")
    assert len(result.boxes) == 1
    assert result.boxes[0].region_id == "r1"
    assert result.boxes[0].label == "peak"
    assert result.boxes[0].x == pytest.approx(10.0)
    assert result.boxes[0].y == pytest.approx(20.0)


def test_parse_localize_multiple_boxes() -> None:
    cov = _make_reasoning()
    data = [
        {"region_id": "r1", "x": 0, "y": 0, "width": 50, "height": 50, "label": "A"},
        {"region_id": "r2", "x": 50, "y": 0, "width": 50, "height": 50, "label": "B"},
    ]
    result = cov._parse_localize(f"<localize>{json.dumps(data)}</localize>")
    assert len(result.boxes) == 2
    assert result.boxes[1].region_id == "r2"


def test_parse_localize_invalid_json_returns_empty_boxes() -> None:
    cov = _make_reasoning()
    result = cov._parse_localize("<localize>not valid json</localize>")
    assert result.boxes == []
    assert result.raw == "not valid json"


def test_parse_localize_no_tag_returns_empty() -> None:
    cov = _make_reasoning()
    result = cov._parse_localize("no localize tag here")
    assert result.boxes == []


def test_parse_analyze_valid_json() -> None:
    cov = _make_reasoning()
    data = {"r1": "Sharp absorption at 1720 cm-1", "r2": "Broad OH stretch"}
    result = cov._parse_analyze(f"<analyze>{json.dumps(data)}</analyze>")
    assert result.descriptions["r1"] == "Sharp absorption at 1720 cm-1"
    assert result.descriptions["r2"] == "Broad OH stretch"


def test_parse_analyze_invalid_json_returns_empty() -> None:
    cov = _make_reasoning()
    result = cov._parse_analyze("<analyze>bad json</analyze>")
    assert result.descriptions == {}


def test_parse_analyze_non_dict_json_returns_empty() -> None:
    cov = _make_reasoning()
    result = cov._parse_analyze("<analyze>[1, 2, 3]</analyze>")
    assert result.descriptions == {}


def test_parse_conclude_valid_json() -> None:
    cov = _make_reasoning()
    findings = {"peak_wavenumbers_cm1": [1720.0, 2960.0], "purity": 0.98}
    result = cov._parse_conclude(f"<conclude>{json.dumps(findings)}</conclude>")
    assert result.findings["purity"] == pytest.approx(0.98)
    assert result.findings["peak_wavenumbers_cm1"] == [1720.0, 2960.0]


def test_parse_conclude_invalid_json_returns_empty() -> None:
    cov = _make_reasoning()
    result = cov._parse_conclude("<conclude>oops</conclude>")
    assert result.findings == {}


def test_parse_conclude_non_dict_json_returns_empty() -> None:
    cov = _make_reasoning()
    result = cov._parse_conclude("<conclude>[1, 2]</conclude>")
    assert result.findings == {}


# ---------------------------------------------------------------------------
# Full round-trip with a mock model
# ---------------------------------------------------------------------------


def _make_raw_output() -> str:
    boxes = [{"region_id": "r1", "x": 0, "y": 0, "width": 100, "height": 50, "label": "IR peak"}]
    analysis = {"r1": "Carbonyl stretch at 1720 cm-1 indicating ester group"}
    conclusion = {"peak_cm1": 1720.0, "peak_intensity": 0.85}
    return (
        f"<localize>{json.dumps(boxes)}</localize>\n"
        f"<analyze>{json.dumps(analysis)}</analyze>\n"
        f"<conclude>{json.dumps(conclusion)}</conclude>"
    )


def test_reason_full_output_parsed_correctly() -> None:
    mock_model = MagicMock()
    mock_model.generate.return_value = _make_raw_output()
    cov = ChainOfVisionReasoning(mock_model)
    image = PILImage.new("RGB", (100, 50))

    out = cov.reason(image, "What are the IR peaks?")

    assert len(out.localization.boxes) == 1
    assert out.localization.boxes[0].label == "IR peak"
    assert "r1" in out.analysis.descriptions
    assert "ester" in out.analysis.descriptions["r1"]
    assert out.conclusion.findings["peak_cm1"] == pytest.approx(1720.0)
    assert out.conclusion.findings["peak_intensity"] == pytest.approx(0.85)


def test_reason_raw_steps_populated() -> None:
    mock_model = MagicMock()
    mock_model.generate.return_value = _make_raw_output()
    cov = ChainOfVisionReasoning(mock_model)
    image = PILImage.new("RGB", (10, 10))

    out = cov.reason(image, "question")
    assert len(out.raw_steps) == 1
    assert "<localize>" in out.raw_steps[0]


def test_reason_prompt_includes_question() -> None:
    mock_model = MagicMock()
    mock_model.generate.return_value = ""
    cov = ChainOfVisionReasoning(mock_model)
    image = PILImage.new("RGB", (10, 10))

    cov.reason(image, "unique_test_question_xyz")

    # Second positional argument to generate() is the prompt
    prompt_arg = mock_model.generate.call_args[0][1]
    assert "unique_test_question_xyz" in prompt_arg


def test_reason_prompt_includes_system_instructions() -> None:
    mock_model = MagicMock()
    mock_model.generate.return_value = ""
    cov = ChainOfVisionReasoning(mock_model)
    image = PILImage.new("RGB", (10, 10))

    cov.reason(image, "any question")

    prompt_arg = mock_model.generate.call_args[0][1]
    assert "LOCALIZE" in prompt_arg
    assert "ANALYZE" in prompt_arg
    assert "CONCLUDE" in prompt_arg


def test_reason_empty_model_output_returns_empty_results() -> None:
    mock_model = MagicMock()
    mock_model.generate.return_value = ""
    cov = ChainOfVisionReasoning(mock_model)
    image = PILImage.new("RGB", (10, 10))

    out = cov.reason(image, "question")

    assert out.localization.boxes == []
    assert out.analysis.descriptions == {}
    assert out.conclusion.findings == {}
