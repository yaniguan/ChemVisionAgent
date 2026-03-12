"""ReAct-based agent orchestration for multi-step scientific image reasoning.

Public API
----------
ChemVisionAgent     -- top-level agent that plans and executes skill chains
AgentConfig         -- agent-level configuration
AgentPlanner        -- Anthropic tool-use planner
AgentStep           -- single Thought/Action/Observation record
AgentTrace          -- full reasoning trace for one query
AnalysisReport      -- structured synthesis report with confidence flags
ToolCallLog         -- per-skill invocation log entry
AnthropicVisionFallback -- duck-typed vision backend using Anthropic API
"""

from chemvision.agent.adapter import AnthropicVisionFallback
from chemvision.agent.agent import ChemVisionAgent
from chemvision.agent.config import AgentConfig
from chemvision.agent.planner import AgentPlanner
from chemvision.agent.report import AnalysisReport
from chemvision.agent.tool_log import ToolCallLog
from chemvision.agent.trace import AgentStep, AgentTrace, StepType

__all__ = [
    "AgentConfig",
    "AgentPlanner",
    "AgentStep",
    "AgentTrace",
    "AnalysisReport",
    "AnthropicVisionFallback",
    "ChemVisionAgent",
    "StepType",
    "ToolCallLog",
]
