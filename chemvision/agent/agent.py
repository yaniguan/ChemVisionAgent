"""ChemVisionAgent — ReAct loop that chains vision skills to answer a query."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from chemvision.agent.adapter import AnthropicVisionFallback
from chemvision.agent.config import AgentConfig
from chemvision.agent.planner import AgentPlanner
from chemvision.agent.report import AnalysisReport
from chemvision.agent.tool_log import ToolCallLog
from chemvision.agent.trace import AgentStep, AgentTrace, StepType
from chemvision.skills.base import SkillResult
from chemvision.skills.skill_registry import DEFAULT_REGISTRY, SkillRegistry


# Skill name in planner → registry name mapping
# Planner uses short friendly names; registry uses skill.name attributes.
_PLANNER_TO_REGISTRY: dict[str, str] = {
    "analyze_structure": "analyze_structure",
    "extract_spectrum": "extract_spectrum_data",
    "compare_structures": "compare_structures",
    "validate_caption": "validate_figure_caption",
    "detect_anomaly": "detect_anomaly",
}


class ChemVisionAgent:
    """Multi-step ReAct agent for scientific image reasoning.

    Uses the Anthropic tool-use API (``claude-sonnet-4-20250514``) for
    planning and dispatches to registered vision skills for execution.

    The agent iterates Thought → Action (skill call) → Observation until
    it produces a Final Answer or reaches ``config.max_steps``.

    Example
    -------
    >>> cfg = AgentConfig()
    >>> agent = ChemVisionAgent(cfg)
    >>> report = agent.run(
    ...     question="What crystal phase is dominant?",
    ...     image_paths=["xrd.png"],
    ... )
    >>> print(report.final_answer)
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        registry: SkillRegistry | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self._registry = registry or DEFAULT_REGISTRY
        self._vision_model: Any = None  # lazy
        self._planner: AgentPlanner | None = None  # lazy

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_vision_model(self) -> Any:
        """Return a vision model; loads lazily on first call."""
        if self._vision_model is not None:
            return self._vision_model

        if self.config.model is not None:
            # Load the configured local vision model
            from chemvision.models.llava import LLaVAWrapper  # noqa: PLC0415

            self._vision_model = LLaVAWrapper(self.config.model)
            self._vision_model.load()
        else:
            # Fall back to Anthropic vision API
            self._vision_model = AnthropicVisionFallback(
                api_key=self.config.anthropic_api_key,
            )

        return self._vision_model

    def _get_planner(self) -> AgentPlanner:
        if self._planner is None:
            self._planner = AgentPlanner(
                api_key=self.config.anthropic_api_key,
                planning_model=self.config.planning_model,
            )
        return self._planner

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        image_paths: list[str],
    ) -> AnalysisReport:
        """Execute the ReAct loop and return a structured :class:`AnalysisReport`.

        Parameters
        ----------
        question:
            Natural-language question to answer.
        image_paths:
            Paths to one or more scientific images.

        Returns
        -------
        AnalysisReport
            Synthesised answer with full tool call log and confidence flags.
        """
        planner = self._get_planner()
        vision_model = self._get_vision_model()

        # Load images eagerly so skills can access them by index
        images: list[Image.Image] = [
            Image.open(p).convert("RGB") for p in image_paths
        ]

        trace = AgentTrace(
            query=question,
            image_paths=list(image_paths),
        )
        tool_logs: list[ToolCallLog] = []
        messages = planner.build_initial_message(question, image_paths)

        step_index = 0
        final_answer: str = ""
        available_skills = (
            self.config.skill_names if self.config.skill_names else list(_PLANNER_TO_REGISTRY.keys())
        )

        for _iteration in range(self.config.max_steps):
            # --- Thought: ask Claude what to do next --------------------
            response = planner.plan(messages, available_skill_names=available_skills)

            # Accumulate assistant message into history
            messages.append(planner.assistant_message(response))

            thought_text = planner.extract_text(response)
            if thought_text:
                step_index += 1
                thought_step = AgentStep(
                    step_index=step_index,
                    step_type=StepType.THOUGHT,
                    content=thought_text,
                )
                trace.append(thought_step)
                if self.config.verbose:
                    print(f"[THOUGHT] {thought_text}")

            # --- Action: parse tool calls --------------------------------
            tool_calls = planner.extract_tool_calls(response)

            if not tool_calls:
                # Model stopped without calling a tool — treat as done
                if not final_answer:
                    final_answer = thought_text or "Unable to determine an answer."
                break

            all_done = False
            tool_result_messages: list[dict[str, Any]] = []

            for call in tool_calls:
                tool_name: str = call["name"]
                tool_input: dict[str, Any] = call.get("input") or {}
                tool_id: str = call["id"]

                step_index += 1
                action_step = AgentStep(
                    step_index=step_index,
                    step_type=StepType.ACTION,
                    content=json.dumps({"tool": tool_name, "input": tool_input}),
                    skill_name=tool_name,
                )
                trace.append(action_step)
                if self.config.verbose:
                    print(f"[ACTION] {tool_name}({tool_input})")

                # Handle final_answer pseudo-tool
                if tool_name == "final_answer":
                    final_answer = tool_input.get("answer", "")
                    all_done = True
                    observation = "Answer recorded."
                    tool_result_messages.extend(
                        planner.build_tool_result_message(tool_id, observation)
                    )
                    break

                # --- Execute skill ----------------------------------------
                observation, log = self._execute_skill(
                    tool_name=tool_name,
                    tool_input=tool_input,
                    images=images,
                    image_paths=image_paths,
                    vision_model=vision_model,
                )
                tool_logs.append(log)

                step_index += 1
                obs_step = AgentStep(
                    step_index=step_index,
                    step_type=StepType.OBSERVATION,
                    content=observation,
                    skill_name=tool_name,
                )
                trace.append(obs_step)
                if self.config.verbose:
                    print(f"[OBSERVATION] {observation[:200]}")

                tool_result_messages.extend(
                    planner.build_tool_result_message(tool_id, observation)
                )

            # Append all tool results as a single batch
            messages.extend(tool_result_messages)

            if all_done:
                break

            # Stop early if model signals end_turn with no tool calls
            if planner.is_final(response) and not tool_calls:
                break

        trace.final_answer = final_answer
        trace.finished_at = datetime.utcnow()

        return AnalysisReport.build(
            question=question,
            image_paths=list(image_paths),
            final_answer=final_answer,
            tool_logs=tool_logs,
            num_steps=step_index,
            confidence_threshold=self.config.confidence_threshold,
        )

    # ------------------------------------------------------------------
    # Skill execution
    # ------------------------------------------------------------------

    def _execute_skill(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        images: list[Image.Image],
        image_paths: list[str],
        vision_model: Any,
    ) -> tuple[str, ToolCallLog]:
        """Run a single skill and return (observation_text, ToolCallLog)."""
        registry_name = _PLANNER_TO_REGISTRY.get(tool_name, tool_name)

        try:
            skill = self._registry.get(registry_name)
        except KeyError:
            observation = f"Error: skill '{registry_name}' is not registered."
            log = ToolCallLog(
                skill_name=tool_name,
                inputs=tool_input,
                output_summary=observation,
                raw_output="",
            )
            return observation, log

        # Resolve image(s) from indices
        primary_image, extra_images = self._resolve_images(tool_input, images)

        # Build kwargs for the skill
        kwargs: dict[str, Any] = {
            k: v
            for k, v in tool_input.items()
            if k not in ("image_index", "image_indices")
        }
        if extra_images:
            kwargs["images"] = [primary_image] + extra_images

        try:
            result: SkillResult = skill(primary_image, vision_model, **kwargs)
        except Exception as exc:  # noqa: BLE001
            observation = f"Skill execution error: {exc}"
            log = ToolCallLog(
                skill_name=tool_name,
                inputs=tool_input,
                output_summary=observation,
                raw_output=str(exc),
            )
            return observation, log

        observation = self._format_observation(result)
        log = ToolCallLog(
            skill_name=tool_name,
            inputs=tool_input,
            output_summary=observation[:500],
            confidence=result.confidence,
            raw_output=result.raw_output,
        )
        return observation, log

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_images(
        tool_input: dict[str, Any],
        images: list[Image.Image],
    ) -> tuple[Image.Image, list[Image.Image]]:
        """Select primary image and any additional images from tool_input."""
        if not images:
            raise ValueError("No images provided to the agent.")

        # Multi-image skill (compare_structures)
        indices = tool_input.get("image_indices")
        if indices is not None:
            selected = [images[i] for i in indices if i < len(images)]
            if not selected:
                selected = images
            return selected[0], selected[1:]

        # Single-image skill
        idx = int(tool_input.get("image_index") or 0)
        idx = min(idx, len(images) - 1)
        return images[idx], []

    @staticmethod
    def _format_observation(result: SkillResult) -> str:
        """Summarise a SkillResult as a text observation for the planner."""
        conf_str = (
            f" (confidence: {result.confidence:.2f})"
            if result.confidence is not None
            else ""
        )
        summary_parts = [f"Skill '{result.skill_name}' result{conf_str}:"]

        # Include key parsed fields if available
        if result.parsed:
            try:
                summary_parts.append(json.dumps(result.parsed, indent=2))
            except (TypeError, ValueError):
                summary_parts.append(str(result.parsed))
        else:
            summary_parts.append(result.raw_output[:1000])

        return "\n".join(summary_parts)
