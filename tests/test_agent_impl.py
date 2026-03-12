"""Comprehensive tests for the ChemVision ReAct agent implementation.

All tests mock external dependencies (Anthropic API, vision model, PIL)
so the suite runs offline without requiring API keys or GPU.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chemvision.agent.adapter import AnthropicVisionFallback
from chemvision.agent.agent import ChemVisionAgent
from chemvision.agent.config import AgentConfig
from chemvision.agent.planner import AgentPlanner
from chemvision.agent.report import AnalysisReport
from chemvision.agent.tool_log import ToolCallLog
from chemvision.agent.trace import AgentStep, AgentTrace, StepType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dummy_image(tmp_path: Path):
    """Create a small real PNG image on disk and return its path."""
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (64, 64), color=(128, 200, 100))
    p = tmp_path / "test_image.png"
    img.save(p)
    return str(p)


@pytest.fixture()
def tool_log_high():
    return ToolCallLog(
        skill_name="analyze_structure",
        inputs={"material_type": "ceramic"},
        output_summary="Lattice params extracted.",
        confidence=0.92,
        raw_output='{"lattice_params": {}}',
    )


@pytest.fixture()
def tool_log_low():
    return ToolCallLog(
        skill_name="detect_anomaly",
        inputs={"domain_context": "SEM"},
        output_summary="One crack detected.",
        confidence=0.45,
        raw_output='{"anomalies": []}',
    )


# ---------------------------------------------------------------------------
# ToolCallLog tests
# ---------------------------------------------------------------------------


class TestToolCallLog:
    def test_default_low_confidence_flag(self, tool_log_high):
        assert tool_log_high.low_confidence is False

    def test_has_timestamp(self, tool_log_high):
        assert isinstance(tool_log_high.timestamp, datetime)

    def test_serialisable(self, tool_log_high):
        d = tool_log_high.model_dump(mode="json")
        assert d["skill_name"] == "analyze_structure"
        assert d["confidence"] == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# AnalysisReport tests
# ---------------------------------------------------------------------------


class TestAnalysisReport:
    def test_no_tool_logs_no_flag(self):
        report = AnalysisReport.build(
            question="What is this?",
            image_paths=["img.png"],
            final_answer="It's a crystal.",
            tool_logs=[],
            confidence_threshold=0.75,
        )
        assert report.low_confidence_flag is False
        assert report.min_intermediate_confidence is None

    def test_all_high_confidence_no_flag(self, tool_log_high):
        report = AnalysisReport.build(
            question="Q",
            image_paths=["a.png"],
            final_answer="A",
            tool_logs=[tool_log_high],
            confidence_threshold=0.75,
        )
        assert report.low_confidence_flag is False
        assert report.min_intermediate_confidence == pytest.approx(0.92)

    def test_low_confidence_sets_flag(self, tool_log_high, tool_log_low):
        report = AnalysisReport.build(
            question="Q",
            image_paths=["a.png"],
            final_answer="A",
            tool_logs=[tool_log_high, tool_log_low],
            confidence_threshold=0.75,
        )
        assert report.low_confidence_flag is True
        assert report.min_intermediate_confidence == pytest.approx(0.45)

    def test_back_fills_low_confidence_on_logs(self, tool_log_high, tool_log_low):
        AnalysisReport.build(
            question="Q",
            image_paths=["a.png"],
            final_answer="A",
            tool_logs=[tool_log_high, tool_log_low],
            confidence_threshold=0.75,
        )
        assert tool_log_high.low_confidence is False
        assert tool_log_low.low_confidence is True

    def test_to_dict_is_json_serialisable(self, tool_log_high):
        report = AnalysisReport.build(
            question="Q",
            image_paths=["a.png"],
            final_answer="A",
            tool_logs=[tool_log_high],
            num_steps=3,
        )
        d = report.to_dict()
        assert json.dumps(d)  # must not raise
        assert d["num_steps"] == 3

    def test_none_confidence_logs_ignored_for_flag(self):
        log = ToolCallLog(
            skill_name="validate_caption",
            inputs={},
            output_summary="ok",
            confidence=None,
        )
        report = AnalysisReport.build(
            question="Q",
            image_paths=[],
            final_answer="A",
            tool_logs=[log],
            confidence_threshold=0.75,
        )
        assert report.low_confidence_flag is False
        assert report.min_intermediate_confidence is None


# ---------------------------------------------------------------------------
# AgentTrace tests
# ---------------------------------------------------------------------------


class TestAgentTrace:
    def test_append_and_count(self):
        trace = AgentTrace(query="test", image_paths=["img.png"])
        trace.append(AgentStep(step_index=0, step_type=StepType.THOUGHT, content="thinking"))
        assert trace.num_steps() == 1

    def test_multiple_image_paths(self):
        trace = AgentTrace(query="compare", image_paths=["a.png", "b.png"])
        assert len(trace.image_paths) == 2

    def test_empty_initially(self):
        trace = AgentTrace(query="q", image_paths=[])
        assert trace.num_steps() == 0
        assert trace.final_answer is None


# ---------------------------------------------------------------------------
# AgentConfig tests
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.model is None
        assert cfg.planning_model == "claude-sonnet-4-20250514"
        assert cfg.confidence_threshold == pytest.approx(0.75)
        assert cfg.max_steps == 10
        assert cfg.skill_names == []
        assert cfg.verbose is False

    def test_confidence_threshold_validation(self):
        with pytest.raises(Exception):
            AgentConfig(confidence_threshold=1.5)

    def test_max_steps_validation(self):
        with pytest.raises(Exception):
            AgentConfig(max_steps=0)


# ---------------------------------------------------------------------------
# AnthropicVisionFallback tests
# ---------------------------------------------------------------------------


class TestAnthropicVisionFallback:
    def test_init_uses_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
        fb = AnthropicVisionFallback()
        assert fb._api_key == "test-key-123"

    def test_generate_encodes_image_and_calls_api(self, tmp_path):
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (32, 32), color=(0, 0, 0))
        img_path = tmp_path / "img.png"
        img.save(img_path)

        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "Test response from Claude"
        mock_response.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        fb = AnthropicVisionFallback(api_key="fake-key")
        fb._client = mock_client

        result = fb.generate(img, "What is in this image?")
        assert result == "Test response from Claude"
        mock_client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# AgentPlanner tests
# ---------------------------------------------------------------------------


class TestAgentPlanner:
    def test_build_initial_message_structure(self, dummy_image):
        planner = AgentPlanner(api_key="fake-key")
        msgs = planner.build_initial_message("What phase?", [dummy_image])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        # Should contain image block + question
        content = msgs[0]["content"]
        assert any(c.get("type") == "image" for c in content)
        assert any(c.get("type") == "text" and "What phase?" in c["text"] for c in content)

    def test_build_tool_result_message(self):
        planner = AgentPlanner(api_key="fake-key")
        msgs = planner.build_tool_result_message("tool_abc", "Found cracks.")
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["type"] == "tool_result"
        assert msgs[0]["content"][0]["tool_use_id"] == "tool_abc"
        assert msgs[0]["content"][0]["content"] == "Found cracks."

    def test_extract_tool_calls(self):
        mock_response = MagicMock()
        block = MagicMock()
        block.type = "tool_use"
        block.id = "call_001"
        block.name = "analyze_structure"
        block.input = {"material_type": "metal"}
        mock_response.content = [block]

        calls = AgentPlanner.extract_tool_calls(mock_response)
        assert len(calls) == 1
        assert calls[0]["name"] == "analyze_structure"
        assert calls[0]["input"]["material_type"] == "metal"

    def test_extract_text(self):
        mock_response = MagicMock()
        b1 = MagicMock(spec=["text"])
        b1.text = "Hello"
        b2 = MagicMock(spec=["text"])
        b2.text = "World"
        mock_response.content = [b1, b2]

        text = AgentPlanner.extract_text(mock_response)
        assert "Hello" in text
        assert "World" in text

    def test_is_final_end_turn(self):
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        assert AgentPlanner.is_final(mock_response) is True

    def test_is_final_tool_use_not_final(self):
        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        assert AgentPlanner.is_final(mock_response) is False

    def test_plan_filters_tools_by_skill_names(self):
        planner = AgentPlanner(api_key="fake-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock()
        planner._client = mock_client

        planner.plan(messages=[], available_skill_names=["analyze_structure"])
        call_kwargs = mock_client.messages.create.call_args[1]
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "analyze_structure" in tool_names
        assert "final_answer" in tool_names
        assert "detect_anomaly" not in tool_names


# ---------------------------------------------------------------------------
# ChemVisionAgent integration tests (fully mocked)
# ---------------------------------------------------------------------------


def _make_mock_response(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_id: str = "call_001",
    stop_reason: str = "tool_use",
    text: str = "",
) -> MagicMock:
    """Build a mock Anthropic response that calls one tool."""
    response = MagicMock()
    response.stop_reason = stop_reason

    text_block = MagicMock(spec=["text"])
    text_block.text = text

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = tool_id
    tool_block.name = tool_name
    tool_block.input = tool_input

    response.content = [text_block, tool_block]
    return response


def _make_final_response(answer: str) -> MagicMock:
    return _make_mock_response(
        "final_answer",
        {"answer": answer},
        tool_id="call_final",
        stop_reason="tool_use",
    )


class TestChemVisionAgentRun:
    """Integration-style tests with all external calls mocked."""

    def _build_agent_with_mocks(self, planner_responses: list[MagicMock]) -> ChemVisionAgent:
        cfg = AgentConfig(anthropic_api_key="fake-key", verbose=False)
        agent = ChemVisionAgent(cfg)

        # Mock planner
        mock_planner = MagicMock(spec=AgentPlanner)
        mock_planner.plan.side_effect = planner_responses
        mock_planner.build_initial_message.return_value = [{"role": "user", "content": []}]
        mock_planner.build_tool_result_message.return_value = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "obs"}]}
        ]
        mock_planner.assistant_message.return_value = {"role": "assistant", "content": []}
        mock_planner.extract_text.return_value = "I will analyse the structure."
        mock_planner.extract_tool_calls.side_effect = lambda r: r._tool_calls
        mock_planner.is_final.side_effect = lambda r: r.stop_reason in ("end_turn", "max_tokens")
        agent._planner = mock_planner

        # Mock vision model
        mock_vision = MagicMock()
        mock_vision.generate.return_value = json.dumps({
            "lattice_params": {"a": 3.5, "b": 3.5, "c": 3.5},
            "symmetry": "cubic",
            "defect_locations": [],
            "defect_density": 0.01,
            "confidence": 0.88,
        })
        agent._vision_model = mock_vision

        return agent

    def test_run_single_skill_then_final_answer(self, dummy_image):
        # First call: analyze_structure; Second call: final_answer
        r1 = MagicMock()
        r1.stop_reason = "tool_use"
        r1.content = []
        r1._tool_calls = [{"id": "c1", "name": "analyze_structure", "input": {}}]

        r2 = MagicMock()
        r2.stop_reason = "tool_use"
        r2.content = []
        r2._tool_calls = [{"id": "c2", "name": "final_answer", "input": {"answer": "Cubic TiO2."}}]

        agent = self._build_agent_with_mocks([r1, r2])
        report = agent.run(question="What phase?", image_paths=[dummy_image])

        assert report.final_answer == "Cubic TiO2."
        assert len(report.tool_logs) == 1
        assert report.tool_logs[0].skill_name == "analyze_structure"
        assert report.num_steps > 0

    def test_confidence_flag_propagates(self, dummy_image):
        """Low-confidence skill output should set low_confidence_flag on report."""
        mock_vision = MagicMock()
        # Return low confidence
        mock_vision.generate.return_value = json.dumps({"anomalies": [], "severity": "none", "confidence": 0.3})

        r1 = MagicMock()
        r1.stop_reason = "tool_use"
        r1.content = []
        r1._tool_calls = [{"id": "c1", "name": "detect_anomaly", "input": {"domain_context": "SEM"}}]

        r2 = MagicMock()
        r2.stop_reason = "tool_use"
        r2.content = []
        r2._tool_calls = [{"id": "c2", "name": "final_answer", "input": {"answer": "No anomaly."}}]

        agent = self._build_agent_with_mocks([r1, r2])
        agent._vision_model = mock_vision
        report = agent.run(question="Any defects?", image_paths=[dummy_image])

        assert report.low_confidence_flag is True
        assert report.tool_logs[0].low_confidence is True

    def test_unknown_skill_returns_error_observation(self, dummy_image):
        r1 = MagicMock()
        r1.stop_reason = "tool_use"
        r1.content = []
        r1._tool_calls = [{"id": "c1", "name": "nonexistent_skill", "input": {}}]

        r2 = MagicMock()
        r2.stop_reason = "tool_use"
        r2.content = []
        r2._tool_calls = [{"id": "c2", "name": "final_answer", "input": {"answer": "Unknown."}}]

        agent = self._build_agent_with_mocks([r1, r2])
        # nonexistent_skill maps to itself and won't be in registry
        report = agent.run(question="?", image_paths=[dummy_image])
        assert "not registered" in report.tool_logs[0].output_summary

    def test_respects_max_steps(self, dummy_image):
        """Agent must stop after max_steps even without final_answer."""
        # All responses keep calling a skill (no final_answer)
        def make_skill_response():
            r = MagicMock()
            r.stop_reason = "tool_use"
            r.content = []
            r._tool_calls = [{"id": "cx", "name": "analyze_structure", "input": {}}]
            return r

        cfg = AgentConfig(anthropic_api_key="fake-key", max_steps=3, verbose=False)
        agent = ChemVisionAgent(cfg)

        mock_planner = MagicMock(spec=AgentPlanner)
        mock_planner.plan.side_effect = [make_skill_response() for _ in range(10)]
        mock_planner.build_initial_message.return_value = [{"role": "user", "content": []}]
        mock_planner.build_tool_result_message.return_value = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "obs"}]}
        ]
        mock_planner.assistant_message.return_value = {"role": "assistant", "content": []}
        mock_planner.extract_text.return_value = ""
        mock_planner.extract_tool_calls.side_effect = lambda r: r._tool_calls
        mock_planner.is_final.return_value = False
        agent._planner = mock_planner

        mock_vision = MagicMock()
        mock_vision.generate.return_value = '{"lattice_params": {}, "confidence": 0.8}'
        agent._vision_model = mock_vision

        report = agent.run(question="?", image_paths=[dummy_image])
        # Should have stopped after max_steps=3 iterations
        assert mock_planner.plan.call_count <= cfg.max_steps


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------


class TestAPIEndpoints:
    def test_health(self):
        from fastapi.testclient import TestClient

        from chemvision.api import app

        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_audit_no_run(self):
        from fastapi.testclient import TestClient

        from chemvision.api import _STATE, app

        _STATE["latest_audit"] = None
        with TestClient(app) as client:
            resp = client.get("/audit")
            assert resp.status_code == 200
            data = resp.json()
            assert data["available"] is False

    def test_analyze_missing_image(self):
        from fastapi.testclient import TestClient

        from chemvision.api import _STATE, app

        _STATE["agent"] = MagicMock()
        with TestClient(app) as client:
            resp = client.post(
                "/analyze",
                json={"question": "What?", "image_paths": ["/nonexistent/image.png"]},
            )
            assert resp.status_code == 422

    def test_analyze_calls_agent_and_caches_report(self, dummy_image):
        from fastapi.testclient import TestClient

        from chemvision.api import _STATE, app

        mock_report = AnalysisReport.build(
            question="Q",
            image_paths=[dummy_image],
            final_answer="It is anatase.",
            tool_logs=[],
            num_steps=2,
        )

        mock_agent = MagicMock(spec=ChemVisionAgent)
        mock_agent.run.return_value = mock_report

        with TestClient(app) as client:
            # Override after lifespan has initialised (lifespan creates its own agent)
            _STATE["agent"] = mock_agent
            _STATE["latest_audit"] = None

            resp = client.post(
                "/analyze",
                json={"question": "Q", "image_paths": [dummy_image]},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["final_answer"] == "It is anatase."
            assert data["low_confidence_flag"] is False
            assert _STATE["latest_audit"] is not None
