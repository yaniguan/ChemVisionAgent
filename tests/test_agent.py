"""Tests for chemvision.agent module."""

from chemvision.agent.trace import AgentStep, AgentTrace, StepType


def test_agent_trace_append_and_count() -> None:
    trace = AgentTrace(query="What is this?", image_paths=["/tmp/img.png"])
    step = AgentStep(step_index=0, step_type=StepType.THOUGHT, content="I need to inspect the image.")
    trace.append(step)
    assert trace.num_steps() == 1
    assert trace.steps[0].step_type == StepType.THOUGHT


def test_agent_trace_no_steps_initially() -> None:
    trace = AgentTrace(query="test", image_paths=["img.png"])
    assert trace.num_steps() == 0
    assert trace.final_answer is None


def test_agent_trace_multiple_image_paths() -> None:
    trace = AgentTrace(query="compare", image_paths=["a.png", "b.png"])
    assert len(trace.image_paths) == 2
