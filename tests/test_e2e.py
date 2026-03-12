"""End-to-end integration test: XRD phase-transition analysis.

Scenario
--------
"Analyze these 3 XRD images taken at 200°C, 400°C, 600°C.
Identify the phase transition temperature and plot grain size vs temperature."

The Anthropic API and local vision model are mocked so the test runs fully
offline, but the real agent ReAct loop, skill dispatch, confidence
propagation, and report assembly are all exercised end-to-end.

Assertions
----------
✓ Agent calls extract_spectrum at least 3 times (once per image)
✓ Agent calls compare_structures at least once
✓ Final report contains a phase_transition_temperature field
✓ All intermediate confidence scores are logged (non-None)
✓ Total wall-clock latency is below 30 seconds
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from chemvision.agent.agent import ChemVisionAgent
from chemvision.agent.config import AgentConfig
from chemvision.agent.planner import AgentPlanner
from chemvision.agent.report import AnalysisReport
from chemvision.data.synthetic_generator import XRDImageGenerator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEMPERATURES = [200.0, 400.0, 600.0]
QUESTION = (
    "Analyze these 3 XRD images taken at 200°C, 400°C, 600°C. "
    "Identify the phase transition temperature and plot grain size vs temperature."
)
MAX_LATENCY_S = 30.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def xrd_images(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """Generate three synthetic XRD PNG images and return their paths."""
    out_dir = tmp_path_factory.mktemp("xrd_series")
    gen = XRDImageGenerator(seed=0)
    samples = gen.generate_temperature_series(
        temperatures=TEMPERATURES,
        output_dir=out_dir,
    )
    paths = [str(s.image_path) for s in samples]
    assert len(paths) == 3, "Expected exactly 3 images"
    assert all(Path(p).exists() for p in paths), "All image files must exist on disk"
    return paths


# ---------------------------------------------------------------------------
# Mock vision model responses keyed by skill + image index
# ---------------------------------------------------------------------------

def _xrd_peaks_json(temperature: float, confidence: float) -> str:
    """Return a realistic-looking extract_spectrum JSON response."""
    if temperature <= 200:
        peaks = [
            {"two_theta": 25.28, "intensity": 1.00, "fwhm": 0.80, "assignment": "A(101)"},
            {"two_theta": 48.05, "intensity": 0.35, "fwhm": 0.78, "assignment": "A(200)"},
            {"two_theta": 55.06, "intensity": 0.22, "fwhm": 0.82, "assignment": "A(211)"},
        ]
        phase = "anatase"
        grain_nm = 18.5
    elif temperature <= 400:
        peaks = [
            {"two_theta": 25.28, "intensity": 0.60, "fwhm": 0.55, "assignment": "A(101)"},
            {"two_theta": 27.45, "intensity": 0.40, "fwhm": 0.50, "assignment": "R(110)"},
            {"two_theta": 36.09, "intensity": 0.25, "fwhm": 0.52, "assignment": "R(101)"},
        ]
        phase = "mixed"
        grain_nm = 28.1
    else:
        peaks = [
            {"two_theta": 27.45, "intensity": 1.00, "fwhm": 0.30, "assignment": "R(110)"},
            {"two_theta": 36.09, "intensity": 0.55, "fwhm": 0.28, "assignment": "R(101)"},
            {"two_theta": 54.33, "intensity": 0.55, "fwhm": 0.32, "assignment": "R(211)"},
        ]
        phase = "rutile"
        grain_nm = 45.2

    return json.dumps({
        "spectrum_type": "XRD",
        "phase": phase,
        "peaks": peaks,
        "grain_size_nm": grain_nm,
        "confidence": confidence,
    })


def _compare_json(confidence: float = 0.87) -> str:
    return json.dumps({
        "summary": (
            "Phase transformation from anatase to rutile observed between 400°C and 600°C. "
            "Grain size increases monotonically: 18.5 nm (200°C) → 28.1 nm (400°C) → 45.2 nm (600°C)."
        ),
        "diff_regions": [
            {"region": "25.3° peak", "change": "decreasing", "magnitude": 0.6},
            {"region": "27.5° peak", "change": "increasing", "magnitude": 1.0},
        ],
        "quantitative_changes": [
            {"parameter": "rutile_fraction", "at_200C": 0.05, "at_400C": 0.40, "at_600C": 0.95},
            {"parameter": "grain_size_nm",   "at_200C": 18.5, "at_400C": 28.1, "at_600C": 45.2},
        ],
        "confidence": confidence,
    })


_FINAL_ANSWER = json.dumps({
    "phase_transition_temperature": 500,
    "phase_transition_temperature_unit": "°C",
    "transition_description": (
        "Anatase → rutile phase transition occurs near 500°C, "
        "based on the emergence of the rutile 110 peak at 27.45° 2θ."
    ),
    "grain_size_vs_temperature": [
        {"temperature_c": 200, "grain_size_nm": 18.5},
        {"temperature_c": 400, "grain_size_nm": 28.1},
        {"temperature_c": 600, "grain_size_nm": 45.2},
    ],
    "dominant_phase_at_200C": "anatase",
    "dominant_phase_at_400C": "mixed",
    "dominant_phase_at_600C": "rutile",
})


# ---------------------------------------------------------------------------
# Mock planner factory
# ---------------------------------------------------------------------------


def _build_mock_response(
    tool_calls: list[dict[str, Any]],
    text: str = "",
    stop_reason: str = "tool_use",
) -> MagicMock:
    """Create a mock Anthropic response with the given tool calls."""
    response = MagicMock()
    response.stop_reason = stop_reason

    content_blocks = []

    if text:
        tb = MagicMock(spec=["text"])
        tb.text = text
        content_blocks.append(tb)

    for tc in tool_calls:
        block = MagicMock()
        block.type = "tool_use"
        block.id = tc["id"]
        block.name = tc["name"]
        block.input = tc.get("input", {})
        content_blocks.append(block)

    response.content = content_blocks
    response._tool_calls = tool_calls  # used by mock extract_tool_calls
    return response


def _build_planner_sequence() -> list[MagicMock]:
    """Return the ordered list of mock planner responses for the scenario."""
    return [
        # Step 1: extract spectrum from image 0 (200°C)
        _build_mock_response(
            text="I'll extract XRD peaks from each image starting with the 200°C scan.",
            tool_calls=[{"id": "c1", "name": "extract_spectrum", "input": {"image_index": 0, "spectrum_type": "XRD"}}],
        ),
        # Step 2: extract spectrum from image 1 (400°C)
        _build_mock_response(
            text="Now extracting peaks from the 400°C scan.",
            tool_calls=[{"id": "c2", "name": "extract_spectrum", "input": {"image_index": 1, "spectrum_type": "XRD"}}],
        ),
        # Step 3: extract spectrum from image 2 (600°C)
        _build_mock_response(
            text="Now extracting peaks from the 600°C scan.",
            tool_calls=[{"id": "c3", "name": "extract_spectrum", "input": {"image_index": 2, "spectrum_type": "XRD"}}],
        ),
        # Step 4: compare all three spectra
        _build_mock_response(
            text="I'll compare the three patterns to identify the transition temperature.",
            tool_calls=[{"id": "c4", "name": "compare_structures", "input": {"image_indices": [0, 1, 2], "comparison_type": "phase"}}],
        ),
        # Step 5: emit final answer
        _build_mock_response(
            text="Based on the observations, I can now synthesise the final answer.",
            tool_calls=[{"id": "c5", "name": "final_answer", "input": {"answer": _FINAL_ANSWER}}],
        ),
    ]


# ---------------------------------------------------------------------------
# Vision model mock
# ---------------------------------------------------------------------------


def _mock_vision_generate(image: Any, prompt: str) -> str:
    """Dispatch to the right JSON response based on the prompt content."""
    prompt_lower = prompt.lower()
    if "extract" in prompt_lower or "peak" in prompt_lower or "spectrum" in prompt_lower:
        # We can't know the image index from the prompt alone, so we return
        # temperature-differentiated responses based on call count via side_effect.
        return _xrd_peaks_json(200, 0.91)  # default; overridden via side_effect
    if "compar" in prompt_lower or "diff" in prompt_lower:
        return _compare_json(0.87)
    return json.dumps({"result": "ok", "confidence": 0.85})


# ---------------------------------------------------------------------------
# Main E2E test
# ---------------------------------------------------------------------------


class TestXRDPhaseTransitionE2E:
    """End-to-end test: 3-image XRD phase transition analysis."""

    def _build_agent(self, planner_responses: list[MagicMock]) -> tuple[ChemVisionAgent, MagicMock, MagicMock]:
        cfg = AgentConfig(anthropic_api_key="fake-key", verbose=False, max_steps=10)
        agent = ChemVisionAgent(cfg)

        # --- Mock planner ---------------------------------------------------
        mock_planner = MagicMock(spec=AgentPlanner)
        mock_planner.plan.side_effect = planner_responses
        mock_planner.build_initial_message.return_value = [{"role": "user", "content": []}]
        mock_planner.build_tool_result_message.return_value = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "obs"}]}
        ]
        mock_planner.assistant_message.return_value = {"role": "assistant", "content": []}
        mock_planner.extract_text.side_effect = lambda r: getattr(r, "_text", "")
        mock_planner.extract_tool_calls.side_effect = lambda r: r._tool_calls
        mock_planner.is_final.side_effect = lambda r: r.stop_reason in ("end_turn", "max_tokens")
        agent._planner = mock_planner

        # Set the text on each mock via _text attribute
        for resp in planner_responses:
            for block in resp.content:
                if hasattr(block, "text"):
                    resp._text = block.text
                    break
            else:
                resp._text = ""

        # --- Mock vision model with temperature-indexed responses -----------
        spectrum_responses = [
            _xrd_peaks_json(200, 0.91),
            _xrd_peaks_json(400, 0.85),
            _xrd_peaks_json(600, 0.93),
            _compare_json(0.87),
        ]
        call_counter = {"n": 0}

        def vision_generate_side_effect(image: Any, prompt: str) -> str:
            idx = call_counter["n"]
            call_counter["n"] += 1
            if idx < len(spectrum_responses):
                return spectrum_responses[idx]
            return json.dumps({"result": "ok", "confidence": 0.80})

        mock_vision = MagicMock()
        mock_vision.generate.side_effect = vision_generate_side_effect
        agent._vision_model = mock_vision

        return agent, mock_planner, mock_vision

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_extract_spectrum_called_at_least_3_times(self, xrd_images: list[str]) -> None:
        """Agent must call extract_spectrum for each of the 3 temperature images."""
        responses = _build_planner_sequence()
        agent, mock_planner, mock_vision = self._build_agent(responses)

        report = agent.run(question=QUESTION, image_paths=xrd_images)

        spectrum_calls = [
            log for log in report.tool_logs if log.skill_name == "extract_spectrum"
        ]
        assert len(spectrum_calls) >= 3, (
            f"Expected ≥3 extract_spectrum calls, got {len(spectrum_calls)}"
        )

    def test_compare_structures_called_at_least_once(self, xrd_images: list[str]) -> None:
        """Agent must call compare_structures to identify the phase transition."""
        responses = _build_planner_sequence()
        agent, _, _ = self._build_agent(responses)

        report = agent.run(question=QUESTION, image_paths=xrd_images)

        compare_calls = [
            log for log in report.tool_logs if log.skill_name == "compare_structures"
        ]
        assert len(compare_calls) >= 1, (
            f"Expected ≥1 compare_structures call, got {len(compare_calls)}"
        )

    def test_final_report_contains_phase_transition_temperature(
        self, xrd_images: list[str]
    ) -> None:
        """Final report structured_data must include phase_transition_temperature."""
        responses = _build_planner_sequence()
        agent, _, _ = self._build_agent(responses)

        report = agent.run(question=QUESTION, image_paths=xrd_images)

        assert "phase_transition_temperature" in report.structured_data, (
            "structured_data is missing 'phase_transition_temperature'. "
            f"Got keys: {list(report.structured_data.keys())}"
        )
        ptt = report.structured_data["phase_transition_temperature"]
        assert isinstance(ptt, (int, float)), (
            f"phase_transition_temperature must be numeric, got {type(ptt)}"
        )
        assert 300 <= ptt <= 700, (
            f"Physically unreasonable transition temp: {ptt}°C"
        )

    def test_all_intermediate_confidence_scores_logged(
        self, xrd_images: list[str]
    ) -> None:
        """Every skill tool log must record a non-None confidence score."""
        responses = _build_planner_sequence()
        agent, _, _ = self._build_agent(responses)

        report = agent.run(question=QUESTION, image_paths=xrd_images)

        # exclude final_answer pseudo-tool (it has no confidence)
        skill_logs = [
            log for log in report.tool_logs if log.skill_name != "final_answer"
        ]
        assert len(skill_logs) >= 1, "No skill logs recorded."

        missing = [log.skill_name for log in skill_logs if log.confidence is None]
        assert not missing, (
            f"These skill calls have no confidence logged: {missing}"
        )

    def test_total_latency_below_30_seconds(self, xrd_images: list[str]) -> None:
        """The entire agent run (with mocked I/O) must complete in < 30 s."""
        responses = _build_planner_sequence()
        agent, _, _ = self._build_agent(responses)

        start = time.perf_counter()
        agent.run(question=QUESTION, image_paths=xrd_images)
        elapsed = time.perf_counter() - start

        assert elapsed < MAX_LATENCY_S, (
            f"Agent run took {elapsed:.2f}s, exceeds {MAX_LATENCY_S}s limit"
        )

    def test_grain_size_vs_temperature_data_present(self, xrd_images: list[str]) -> None:
        """Structured data should include grain_size_vs_temperature list."""
        responses = _build_planner_sequence()
        agent, _, _ = self._build_agent(responses)

        report = agent.run(question=QUESTION, image_paths=xrd_images)

        assert "grain_size_vs_temperature" in report.structured_data
        gsvt = report.structured_data["grain_size_vs_temperature"]
        assert isinstance(gsvt, list) and len(gsvt) == 3

        temps = [entry["temperature_c"] for entry in gsvt]
        grain_sizes = [entry["grain_size_nm"] for entry in gsvt]

        # Grain size must increase with temperature (grain growth)
        assert grain_sizes == sorted(grain_sizes), (
            f"Grain sizes not monotonically increasing: {grain_sizes}"
        )
        assert temps == sorted(temps)

    def test_confidence_propagation_flag(self, xrd_images: list[str]) -> None:
        """low_confidence_flag must be False when all skills return high confidence."""
        responses = _build_planner_sequence()
        agent, _, _ = self._build_agent(responses)

        report = agent.run(question=QUESTION, image_paths=xrd_images)

        # All mocked confidences are ≥ 0.85, above threshold=0.75
        assert report.low_confidence_flag is False

    def test_image_paths_recorded_in_report(self, xrd_images: list[str]) -> None:
        """Report must record the original image paths."""
        responses = _build_planner_sequence()
        agent, _, _ = self._build_agent(responses)

        report = agent.run(question=QUESTION, image_paths=xrd_images)

        assert report.image_paths == xrd_images

    def test_num_steps_matches_expected(self, xrd_images: list[str]) -> None:
        """Step count must reflect 3 extractions + 1 compare + 1 final_answer."""
        responses = _build_planner_sequence()
        agent, _, _ = self._build_agent(responses)

        report = agent.run(question=QUESTION, image_paths=xrd_images)

        # 5 planning steps, each producing at least 1 action step + 1 observation
        assert report.num_steps >= 5

    def test_full_scenario_combined(self, xrd_images: list[str]) -> None:
        """Run the full scenario once and verify all invariants together."""
        responses = _build_planner_sequence()
        agent, _, _ = self._build_agent(responses)

        start = time.perf_counter()
        report = agent.run(question=QUESTION, image_paths=xrd_images)
        elapsed = time.perf_counter() - start

        # Latency
        assert elapsed < MAX_LATENCY_S

        # Tool call counts
        spectrum_count = sum(1 for l in report.tool_logs if l.skill_name == "extract_spectrum")
        compare_count = sum(1 for l in report.tool_logs if l.skill_name == "compare_structures")
        assert spectrum_count >= 3
        assert compare_count >= 1

        # Phase transition temperature
        assert "phase_transition_temperature" in report.structured_data
        assert 300 <= report.structured_data["phase_transition_temperature"] <= 700

        # All confidences logged
        for log in report.tool_logs:
            assert log.confidence is not None, f"Missing confidence on {log.skill_name}"

        # Confidence flag
        assert report.low_confidence_flag is False


# ---------------------------------------------------------------------------
# Synthetic generator sanity tests
# ---------------------------------------------------------------------------


class TestXRDImageGenerator:
    def test_generates_correct_number_of_files(self, tmp_path: Path) -> None:
        gen = XRDImageGenerator(seed=1)
        samples = gen.generate_temperature_series(TEMPERATURES, output_dir=tmp_path)
        assert len(samples) == 3
        for s in samples:
            assert s.image_path is not None
            assert Path(s.image_path).exists()
            assert Path(s.image_path).stat().st_size > 1000  # non-trivial PNG

    def test_phase_labels_match_temperature(self, tmp_path: Path) -> None:
        gen = XRDImageGenerator(seed=2, transition_temp=500.0)
        samples = gen.generate_temperature_series([200.0, 400.0, 600.0], output_dir=tmp_path)
        phases = [s.dominant_phase for s in samples]
        assert phases[0] == "anatase"   # 200°C — well below transition
        assert phases[2] == "rutile"    # 600°C — well above transition

    def test_grain_size_increases_with_temperature(self, tmp_path: Path) -> None:
        gen = XRDImageGenerator(seed=3)
        samples = gen.generate_temperature_series([200.0, 400.0, 600.0], output_dir=tmp_path)
        grains = [s.grain_size_nm for s in samples]
        assert grains == sorted(grains), f"Grain sizes not monotonic: {grains}"

    def test_anatase_fraction_decreases_with_temperature(self, tmp_path: Path) -> None:
        gen = XRDImageGenerator(seed=4)
        samples = gen.generate_temperature_series([100.0, 300.0, 500.0, 700.0], output_dir=tmp_path)
        fractions = [s.anatase_fraction for s in samples]
        assert fractions == sorted(fractions, reverse=True)

    def test_single_image_generation(self, tmp_path: Path) -> None:
        gen = XRDImageGenerator()
        sample = gen.generate_single(450.0, output_dir=tmp_path)
        assert sample.image_path is not None
        assert Path(sample.image_path).exists()
        assert sample.temperature_c == 450.0

    def test_structured_data_parsed_from_final_answer(self) -> None:
        """AnalysisReport.build parses JSON from final_answer into structured_data."""
        from chemvision.agent.report import AnalysisReport

        report = AnalysisReport.build(
            question="Q",
            image_paths=[],
            final_answer=_FINAL_ANSWER,
            tool_logs=[],
        )
        assert report.structured_data["phase_transition_temperature"] == 500
        assert len(report.structured_data["grain_size_vs_temperature"]) == 3

    def test_non_json_final_answer_leaves_structured_data_empty(self) -> None:
        from chemvision.agent.report import AnalysisReport

        report = AnalysisReport.build(
            question="Q",
            image_paths=[],
            final_answer="The phase transition is near 500°C.",
            tool_logs=[],
        )
        assert report.structured_data == {}
