"""Tests for chemvision.skills typed skill implementations."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

from chemvision.skills._parse import extract_json, to_float, to_list, to_str
from chemvision.skills.analyze_structure import AnalyzeStructureSkill
from chemvision.skills.compare_structures import CompareStructuresSkill, _concat_images
from chemvision.skills.detect_anomaly import DetectAnomalySkill
from chemvision.skills.extract_spectrum import ExtractSpectrumSkill
from chemvision.skills.outputs import (
    Anomaly,
    AnomalyReport,
    CaptionValidation,
    DefectLocation,
    LatticeParams,
    Peak,
    SpectrumData,
    StructureAnalysis,
    StructureComparison,
)
from chemvision.skills.skill_registry import DEFAULT_REGISTRY, SkillRegistry
from chemvision.skills.validate_caption import ValidateCaptionSkill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image(w: int = 64, h: int = 64) -> PILImage.Image:
    return PILImage.new("RGB", (w, h), color=(100, 120, 140))


def _mock_model(response: str) -> Any:
    m = MagicMock()
    m.generate.return_value = response
    return m


def _json_response(**kwargs: Any) -> str:
    return json.dumps(kwargs)


# ---------------------------------------------------------------------------
# _parse utilities
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_direct_json(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_code_block(self) -> None:
        text = '```json\n{"x": 42}\n```'
        assert extract_json(text) == {"x": 42}

    def test_fenced_no_language_tag(self) -> None:
        text = '```\n{"x": 42}\n```'
        assert extract_json(text) == {"x": 42}

    def test_json_embedded_in_prose(self) -> None:
        text = 'Here is the result: {"k": "v"} — that is all.'
        assert extract_json(text) == {"k": "v"}

    def test_returns_none_on_total_failure(self) -> None:
        assert extract_json("no json here at all") is None

    def test_returns_none_for_list_root(self) -> None:
        # Root lists are not dicts — should fall through
        assert extract_json("[1, 2, 3]") is None

    def test_whitespace_padded(self) -> None:
        assert extract_json('  {"z": 0}  ') == {"z": 0}


class TestToFloat:
    def test_none_returns_default(self) -> None:
        assert to_float(None) is None
        assert to_float(None, default=0.5) == pytest.approx(0.5)

    def test_valid_float(self) -> None:
        assert to_float(3.14) == pytest.approx(3.14)

    def test_string_number(self) -> None:
        assert to_float("2.5") == pytest.approx(2.5)

    def test_invalid_string_returns_default(self) -> None:
        assert to_float("not a number", default=9.9) == pytest.approx(9.9)


class TestToList:
    def test_list_passthrough(self) -> None:
        assert to_list([1, 2]) == [1, 2]

    def test_non_list_returns_empty(self) -> None:
        assert to_list(None) == []
        assert to_list("string") == []
        assert to_list(42) == []


class TestToStr:
    def test_none_returns_default(self) -> None:
        assert to_str(None) == ""
        assert to_str(None, default="x") == "x"

    def test_converts_types(self) -> None:
        assert to_str(3) == "3"


# ---------------------------------------------------------------------------
# Typed output models
# ---------------------------------------------------------------------------


class TestOutputModels:
    def test_lattice_params_defaults(self) -> None:
        lp = LatticeParams()
        assert lp.a is None
        assert lp.unit == "Å"

    def test_structure_analysis_defaults(self) -> None:
        sa = StructureAnalysis(skill_name="analyze_structure", raw_output="")
        assert sa.defect_locations == []
        assert sa.symmetry == ""
        assert sa.defect_density is None

    def test_peak_defaults(self) -> None:
        p = Peak(raw_output="")
        assert p.position == 0.0
        assert p.fwhm is None

    def test_spectrum_data_defaults(self) -> None:
        sd = SpectrumData(skill_name="extract_spectrum_data", raw_output="")
        assert sd.peaks == []
        assert sd.snr is None

    def test_structure_comparison_defaults(self) -> None:
        sc = StructureComparison(skill_name="compare_structures", raw_output="")
        assert sc.diff_regions == []
        assert sc.trend == ""

    def test_caption_validation_bounds(self) -> None:
        cv = CaptionValidation(
            skill_name="validate_figure_caption",
            raw_output="",
            consistency_score=0.95,
        )
        assert cv.consistency_score == pytest.approx(0.95)

    def test_anomaly_report_defaults(self) -> None:
        ar = AnomalyReport(skill_name="detect_anomaly", raw_output="")
        assert ar.anomalies == []
        assert ar.severity == "none"
        assert ar.recommendations == []

    def test_anomaly_severity_literal(self) -> None:
        a = Anomaly(raw_output="", severity="high")
        assert a.severity == "high"

    def test_anomaly_report_severity_none(self) -> None:
        ar = AnomalyReport(skill_name="detect_anomaly", raw_output="", severity="none")
        assert ar.severity == "none"


# ---------------------------------------------------------------------------
# AnalyzeStructureSkill
# ---------------------------------------------------------------------------


class TestAnalyzeStructureSkill:
    def _skill(self) -> AnalyzeStructureSkill:
        return AnalyzeStructureSkill()

    def test_name(self) -> None:
        assert self._skill().name == "analyze_structure"

    def test_build_prompt_contains_material_type(self) -> None:
        prompt = self._skill().build_prompt(material_type="rutile TiO2")
        assert "rutile TiO2" in prompt

    def test_build_prompt_default_material(self) -> None:
        prompt = self._skill().build_prompt()
        assert len(prompt) > 20

    def test_call_parses_valid_json(self) -> None:
        payload = {
            "lattice_params": {"a": 4.59, "b": 4.59, "c": 2.96, "unit": "Å"},
            "symmetry": "tetragonal",
            "defect_locations": [
                {"x": 0.3, "y": 0.4, "defect_type": "vacancy", "confidence": 0.85}
            ],
            "defect_density": 0.05,
            "confidence": 0.9,
        }
        model = _mock_model(_json_response(**payload))
        result = self._skill()(_image(), model, material_type="rutile")

        assert isinstance(result, StructureAnalysis)
        assert result.symmetry == "tetragonal"
        assert result.lattice_params is not None
        assert result.lattice_params.a == pytest.approx(4.59)
        assert result.lattice_params.c == pytest.approx(2.96)
        assert len(result.defect_locations) == 1
        assert result.defect_locations[0].defect_type == "vacancy"
        assert result.defect_locations[0].confidence == pytest.approx(0.85)
        assert result.defect_density == pytest.approx(0.05)
        assert result.confidence == pytest.approx(0.9)

    def test_call_handles_invalid_json_gracefully(self) -> None:
        model = _mock_model("I cannot determine the structure.")
        result = self._skill()(_image(), model)

        assert isinstance(result, StructureAnalysis)
        assert result.symmetry == ""
        assert result.defect_locations == []
        assert result.confidence is None

    def test_call_raw_output_preserved(self) -> None:
        model = _mock_model('{"symmetry": "cubic", "confidence": 0.7}')
        result = self._skill()(_image(), model)
        assert "cubic" in result.raw_output

    def test_extends_skill_result(self) -> None:
        from chemvision.skills.base import SkillResult
        model = _mock_model("{}")
        result = self._skill()(_image(), model)
        assert isinstance(result, SkillResult)


# ---------------------------------------------------------------------------
# ExtractSpectrumSkill
# ---------------------------------------------------------------------------


class TestExtractSpectrumSkill:
    def _skill(self) -> ExtractSpectrumSkill:
        return ExtractSpectrumSkill()

    def test_name(self) -> None:
        assert self._skill().name == "extract_spectrum_data"

    def test_build_prompt_xrd(self) -> None:
        prompt = self._skill().build_prompt(spectrum_type="XRD")
        assert "XRD" in prompt

    def test_call_parses_peaks(self) -> None:
        payload = {
            "peaks": [
                {"position": 33.2, "intensity": 1.0, "assignment": "α-Fe2O3 (104)", "fwhm": 0.3},
                {"position": 35.6, "intensity": 0.6, "assignment": "α-Fe2O3 (110)", "fwhm": 0.4},
            ],
            "background_level": 0.05,
            "snr": 18.5,
            "confidence": 0.88,
        }
        model = _mock_model(json.dumps(payload))
        result = self._skill()(_image(), model, spectrum_type="XRD")

        assert isinstance(result, SpectrumData)
        assert len(result.peaks) == 2
        assert result.peaks[0].position == pytest.approx(33.2)
        assert result.peaks[0].assignment == "α-Fe2O3 (104)"
        assert result.peaks[0].fwhm == pytest.approx(0.3)
        assert result.background_level == pytest.approx(0.05)
        assert result.snr == pytest.approx(18.5)
        assert result.spectrum_type == "XRD"

    def test_call_empty_peaks_on_bad_json(self) -> None:
        model = _mock_model("No spectrum detected.")
        result = self._skill()(_image(), model, spectrum_type="Raman")
        assert result.peaks == []
        assert result.confidence is None

    def test_spectrum_type_stored(self) -> None:
        model = _mock_model('{"peaks": [], "confidence": 0.5}')
        result = self._skill()(_image(), model, spectrum_type="XPS")
        assert result.spectrum_type == "XPS"


# ---------------------------------------------------------------------------
# CompareStructuresSkill
# ---------------------------------------------------------------------------


class TestCompareStructuresSkill:
    def _skill(self) -> CompareStructuresSkill:
        return CompareStructuresSkill()

    def test_name(self) -> None:
        assert self._skill().name == "compare_structures"

    def test_build_prompt_includes_n_images(self) -> None:
        prompt = self._skill().build_prompt(n_images=3, comparison_type="grain growth")
        assert "3" in prompt
        assert "grain growth" in prompt

    def test_concat_images_single(self) -> None:
        img = _image(60, 40)
        result = _concat_images([img])
        assert result is img

    def test_concat_images_two(self) -> None:
        a, b = _image(60, 40), _image(80, 50)
        result = _concat_images([a, b])
        assert result.width == 140   # 60 + 80
        assert result.height == 50  # max(40, 50)

    def test_call_parses_comparison(self) -> None:
        payload = {
            "diff_regions": [
                {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.15, "description": "grain boundary shift"}
            ],
            "quantitative_changes": [
                {"metric": "grain_size", "before": 12.0, "after": 18.5, "delta": 6.5, "unit": "nm"}
            ],
            "trend": "grain_growth",
            "confidence": 0.82,
        }
        images = [_image(), _image()]
        model = _mock_model(json.dumps(payload))
        result = self._skill()(_image(), model, images=images, comparison_type="annealing")

        assert isinstance(result, StructureComparison)
        assert result.trend == "grain_growth"
        assert len(result.diff_regions) == 1
        assert result.diff_regions[0].description == "grain boundary shift"
        assert len(result.quantitative_changes) == 1
        assert result.quantitative_changes[0].delta == pytest.approx(6.5)
        assert result.confidence == pytest.approx(0.82)

    def test_call_uses_single_image_fallback(self) -> None:
        model = _mock_model('{"trend": "stable", "confidence": 0.6}')
        # No "images" kwarg — should fall back to the positional image
        result = self._skill()(_image(), model)
        assert isinstance(result, StructureComparison)

    def test_call_bad_json_returns_empty(self) -> None:
        model = _mock_model("Cannot compare.")
        result = self._skill()(_image(), model, images=[_image(), _image()])
        assert result.diff_regions == []
        assert result.trend == ""


# ---------------------------------------------------------------------------
# ValidateCaptionSkill
# ---------------------------------------------------------------------------


class TestValidateCaptionSkill:
    def _skill(self) -> ValidateCaptionSkill:
        return ValidateCaptionSkill()

    def test_name(self) -> None:
        assert self._skill().name == "validate_figure_caption"

    def test_build_prompt_contains_caption(self) -> None:
        prompt = self._skill().build_prompt(caption="XRD showing cubic phase")
        assert "XRD showing cubic phase" in prompt

    def test_call_consistent_caption(self) -> None:
        payload = {
            "consistency_score": 0.95,
            "contradictions": [],
            "confidence": 0.9,
        }
        model = _mock_model(json.dumps(payload))
        result = self._skill()(_image(), model, caption="SEM image of ZnO nanorods.")

        assert isinstance(result, CaptionValidation)
        assert result.consistency_score == pytest.approx(0.95)
        assert result.contradictions == []
        assert result.confidence == pytest.approx(0.9)

    def test_call_with_contradictions(self) -> None:
        payload = {
            "consistency_score": 0.3,
            "contradictions": [
                "Caption claims FCC but pattern shows BCC peaks.",
                "Scale bar reads 500 nm but caption says 1 µm.",
            ],
            "confidence": 0.85,
        }
        model = _mock_model(json.dumps(payload))
        result = self._skill()(_image(), model, caption="FCC nanoparticles, scale bar 1 µm.")

        assert len(result.contradictions) == 2
        assert "FCC" in result.contradictions[0]
        assert result.consistency_score == pytest.approx(0.3)

    def test_consistency_score_clamped_to_range(self) -> None:
        model = _mock_model('{"consistency_score": 1.5, "contradictions": []}')
        result = self._skill()(_image(), model, caption="test")
        assert result.consistency_score <= 1.0

    def test_bad_json_returns_zero_score(self) -> None:
        model = _mock_model("No caption issues found.")
        result = self._skill()(_image(), model, caption="test")
        assert result.consistency_score == pytest.approx(0.0)
        assert result.contradictions == []


# ---------------------------------------------------------------------------
# DetectAnomalySkill
# ---------------------------------------------------------------------------


class TestDetectAnomalySkill:
    def _skill(self) -> DetectAnomalySkill:
        return DetectAnomalySkill()

    def test_name(self) -> None:
        assert self._skill().name == "detect_anomaly"

    def test_build_prompt_contains_context(self) -> None:
        prompt = self._skill().build_prompt(domain_context="SEM of alumina coating")
        assert "SEM of alumina coating" in prompt

    def test_call_parses_anomalies(self) -> None:
        payload = {
            "anomalies": [
                {
                    "location_x": 0.45,
                    "location_y": 0.62,
                    "anomaly_type": "crack",
                    "description": "Transverse crack through coating layer.",
                    "severity": "high",
                    "confidence": 0.93,
                }
            ],
            "severity": "high",
            "recommendations": ["Halt process; perform root-cause analysis."],
            "confidence": 0.91,
        }
        model = _mock_model(json.dumps(payload))
        result = self._skill()(_image(), model, domain_context="cross-section TEM")

        assert isinstance(result, AnomalyReport)
        assert result.severity == "high"
        assert len(result.anomalies) == 1
        assert result.anomalies[0].anomaly_type == "crack"
        assert result.anomalies[0].severity == "high"
        assert result.anomalies[0].confidence == pytest.approx(0.93)
        assert len(result.recommendations) == 1
        assert result.confidence == pytest.approx(0.91)

    def test_call_no_anomalies(self) -> None:
        payload = {
            "anomalies": [],
            "severity": "none",
            "recommendations": [],
            "confidence": 0.88,
        }
        model = _mock_model(json.dumps(payload))
        result = self._skill()(_image(), model)
        assert result.severity == "none"
        assert result.anomalies == []

    def test_unknown_severity_falls_back_to_none(self) -> None:
        model = _mock_model('{"anomalies": [], "severity": "critical"}')
        result = self._skill()(_image(), model)
        assert result.severity == "none"

    def test_bad_json_returns_empty_report(self) -> None:
        model = _mock_model("No anomalies detected.")
        result = self._skill()(_image(), model)
        assert result.anomalies == []
        assert result.severity == "none"


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------


class TestSkillRegistry:
    def test_default_registry_contains_all_five(self) -> None:
        expected = {
            "analyze_structure",
            "extract_spectrum_data",
            "compare_structures",
            "validate_figure_caption",
            "detect_anomaly",
        }
        assert expected.issubset(set(DEFAULT_REGISTRY.list_skills()))

    def test_contains_operator(self) -> None:
        assert "analyze_structure" in DEFAULT_REGISTRY
        assert "no_such_skill" not in DEFAULT_REGISTRY

    def test_getitem(self) -> None:
        skill = DEFAULT_REGISTRY["detect_anomaly"]
        assert isinstance(skill, DetectAnomalySkill)

    def test_get_raises_for_unknown(self) -> None:
        with pytest.raises(KeyError):
            DEFAULT_REGISTRY.get("no_such_skill")

    def test_len_at_least_five(self) -> None:
        assert len(DEFAULT_REGISTRY) >= 5

    def test_iter_yields_sorted_names(self) -> None:
        names = list(DEFAULT_REGISTRY)
        assert names == sorted(names)

    def test_register_custom_skill(self) -> None:
        from chemvision.skills.base import SkillResult

        class _Custom(SkillRegistry.__mro__[0]):  # just to get a unique type
            pass

        class _TestSkill(AnalyzeStructureSkill):
            name = "_test_registry_skill_xyz"

            def __call__(self, image: Any, model: Any, **kwargs: Any) -> SkillResult:
                return StructureAnalysis(skill_name=self.name, raw_output="")

        registry = SkillRegistry()
        s = _TestSkill()
        registry.register(s)
        assert "_test_registry_skill_xyz" in registry
        assert registry["_test_registry_skill_xyz"] is s

    def test_registry_syncs_with_functional_api(self) -> None:
        from chemvision.skills.registry import get_skill, list_skills

        assert "analyze_structure" in list_skills()
        skill = get_skill("analyze_structure")
        assert isinstance(skill, AnalyzeStructureSkill)

    def test_repr_contains_skill_names(self) -> None:
        r = repr(DEFAULT_REGISTRY)
        assert "analyze_structure" in r

    def test_register_returns_skill(self) -> None:
        registry = SkillRegistry()
        skill = AnalyzeStructureSkill()
        skill.name = "_return_test_skill"  # type: ignore[assignment]
        returned = registry.register(skill)
        assert returned is skill
