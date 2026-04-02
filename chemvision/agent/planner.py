"""AgentPlanner: wraps the Anthropic tool-use API to drive the ReAct loop.

The planner converts the five registered skills into Anthropic tool
definitions, sends Thought prompts to ``claude-sonnet-4-20250514``, and
parses the model's tool-call response back into (skill_name, kwargs) pairs.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

from PIL import Image


# ---------------------------------------------------------------------------
# Anthropic tool definitions for each skill
# ---------------------------------------------------------------------------

_SKILL_TOOLS: list[dict[str, Any]] = [
    {
        "name": "analyze_structure",
        "description": (
            "Analyse the crystallographic structure of a materials science image. "
            "Returns lattice parameters, symmetry group, and defect locations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "Zero-based index into the images list (default 0).",
                },
                "material_type": {
                    "type": "string",
                    "description": "Material class, e.g. 'ceramic', 'metal', 'polymer'.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "extract_spectrum",
        "description": (
            "Extract peak positions, intensities, and assignments from a spectrum image "
            "(XRD, Raman, XPS, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "Zero-based index into the images list (default 0).",
                },
                "spectrum_type": {
                    "type": "string",
                    "enum": ["XRD", "Raman", "XPS", "spectrum"],
                    "description": "Type of spectrum in the image.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "compare_structures",
        "description": (
            "Quantitatively compare two or more structure images side-by-side. "
            "Returns diff regions and quantitative changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Indices of images to compare (default: all images).",
                },
                "comparison_type": {
                    "type": "string",
                    "description": "Focus of comparison, e.g. 'morphology', 'phase', 'defects'.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "validate_caption",
        "description": (
            "Check whether the caption text is consistent with the image content. "
            "Returns a consistency score and list of contradictions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "Zero-based index into the images list (default 0).",
                },
                "caption": {
                    "type": "string",
                    "description": "Caption text to validate against the image.",
                },
            },
            "required": ["caption"],
        },
    },
    {
        "name": "detect_anomaly",
        "description": (
            "Detect anomalies, defects, and unexpected features in a scientific image. "
            "Returns a ranked list of anomalies with severity scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "Zero-based index into the images list (default 0).",
                },
                "domain_context": {
                    "type": "string",
                    "description": (
                        "Description of the imaging modality and material, "
                        "e.g. 'SEM image of alumina coating'."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "identify_molecule",
        "description": (
            "Extract the SMILES string, IUPAC name, molecular formula, molecular weight, "
            "functional groups (with counts), stereocenters (R/S, E/Z), and ring systems "
            "from a 2-D skeletal formula, ball-and-stick model, or hand-drawn molecular diagram."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "Zero-based index into the images list (default 0).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "extract_reaction",
        "description": (
            "Extract reaction type, all molecules (reactants, reagents, catalysts, products), "
            "and experimental conditions (temperature, solvent, time, yield, atmosphere) "
            "from a chemistry literature figure such as a reaction scheme or conditions table."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "Zero-based index into the images list (default 0).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "analyze_microscopy",
        "description": (
            "Analyse morphology, particle size distribution, and scale bar in a microscopy image "
            "(SEM, TEM, STEM, AFM, or optical). Returns dominant particle shape, surface texture, "
            "aggregation state, per-particle measurements (diameter, aspect ratio), aggregate "
            "size statistics (mean, std, min, max, distribution type), and calibrated scale-bar info."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "Zero-based index into the images list (default 0).",
                },
                "imaging_context": {
                    "type": "string",
                    "description": (
                        "Brief description of the sample and instrument, e.g. "
                        "'SEM image of ZnO nanoparticles' or 'TEM of Au nanorods in solution'."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "final_answer",
        "description": (
            "Emit the final synthesised answer when sufficient information has been gathered. "
            "Call this tool once you are ready to conclude."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The complete, concise answer to the user's question.",
                }
            },
            "required": ["answer"],
        },
    },
]

_SYSTEM_PROMPT = """\
You are ChemVision, an expert AI agent for scientific image analysis.

You have access to seven vision skills:
  • analyze_structure    — crystallographic analysis, lattice params, defects
  • extract_spectrum     — peak extraction from XRD / Raman / XPS spectra
  • compare_structures   — quantitative comparison of multiple structure images
  • validate_caption     — figure/caption consistency check
  • detect_anomaly       — anomaly and defect detection with severity ranking
  • extract_reaction     — reaction type, molecules (SMILES), and conditions from chemistry figures
  • analyze_microscopy   — morphology, particle size distribution, and scale bar (SEM/TEM/AFM/OM)
  • identify_molecule    — SMILES, IUPAC name, functional groups, stereocenters, ring systems from structure diagrams

Strategy
--------
1. Think step-by-step about which skill(s) best answer the question.
2. Call the most relevant skill(s) in order.
3. Use the observation from each skill call to refine your understanding.
4. When you have enough information, call `final_answer` with a complete response.

Important
---------
- Always call at least one vision skill before `final_answer`.
- Do NOT repeat a skill with identical arguments.
- Base your final answer exclusively on skill observations, not prior knowledge.
"""


class AgentPlanner:
    """Drive the ReAct loop via Anthropic tool-use.

    Parameters
    ----------
    api_key:
        Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    planning_model:
        Claude model ID to use for planning.
    max_tokens:
        Maximum tokens per planning response.
    """

    def __init__(
        self,
        api_key: str | None = None,
        planning_model: str = "claude-sonnet-4-6",
        max_tokens: int = 2048,
        use_extended_thinking: bool = False,
        thinking_budget_tokens: int = 8000,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._planning_model = planning_model
        self._max_tokens = max_tokens
        self._use_extended_thinking = use_extended_thinking
        self._thinking_budget_tokens = thinking_budget_tokens
        self._client: Any = None

    # ------------------------------------------------------------------
    # Lazy Anthropic client
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "anthropic package is required for AgentPlanner. "
                    "Install it with: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # Image encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(path: str | Path) -> dict[str, Any]:
        """Return an Anthropic image content block for *path*."""
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        }

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def build_initial_message(
        self,
        question: str,
        image_paths: list[str],
    ) -> list[dict[str, Any]]:
        """Construct the first user message (images + question)."""
        content: list[dict[str, Any]] = []

        for i, path in enumerate(image_paths):
            content.append({"type": "text", "text": f"Image {i}:"})
            content.append(self._encode_image(path))

        content.append({"type": "text", "text": f"Question: {question}"})
        return [{"role": "user", "content": content}]

    def build_tool_result_message(
        self,
        tool_use_id: str,
        observation: str,
    ) -> list[dict[str, Any]]:
        """Wrap a skill observation as a tool_result message pair."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": observation,
                    }
                ],
            }
        ]

    # ------------------------------------------------------------------
    # Planning step
    # ------------------------------------------------------------------

    def plan(
        self,
        messages: list[dict[str, Any]],
        available_skill_names: list[str] | None = None,
    ) -> Any:
        """Send *messages* to Claude and return the raw API response.

        Parameters
        ----------
        messages:
            Accumulated conversation history.
        available_skill_names:
            If provided, only include tool definitions for these skills
            (plus ``final_answer`` which is always included).
        """
        client = self._get_client()

        # Filter tools to those available in this agent's registry
        tools = _SKILL_TOOLS
        if available_skill_names is not None:
            allowed = set(available_skill_names) | {"final_answer"}
            tools = [t for t in _SKILL_TOOLS if t["name"] in allowed]

        kwargs: dict[str, Any] = dict(
            model=self._planning_model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if self._use_extended_thinking:
            budget = self._thinking_budget_tokens
            # max_tokens must strictly exceed the thinking budget
            if kwargs["max_tokens"] <= budget:
                kwargs["max_tokens"] = budget + 2048
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        return client.messages.create(**kwargs)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def extract_tool_calls(response: Any) -> list[dict[str, Any]]:
        """Extract all tool_use blocks from a Claude response.

        Returns a list of dicts with keys ``id``, ``name``, ``input``.
        """
        calls = []
        for block in response.content:
            if block.type == "tool_use":
                calls.append(
                    {"id": block.id, "name": block.name, "input": block.input}
                )
        return calls

    @staticmethod
    def extract_text(response: Any) -> str:
        """Extract concatenated text blocks from a Claude response (excludes thinking)."""
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "thinking":
                continue
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    @staticmethod
    def extract_thinking(response: Any) -> str:
        """Extract extended thinking content if present (empty string otherwise)."""
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "thinking" and hasattr(block, "thinking"):
                parts.append(block.thinking)
        return "\n".join(parts)

    @staticmethod
    def is_final(response: Any) -> bool:
        """Return True when the model's stop_reason signals end_turn."""
        return response.stop_reason in ("end_turn", "max_tokens")

    @staticmethod
    def assistant_message(response: Any) -> dict[str, Any]:
        """Convert a Claude response into an assistant message for history."""
        return {"role": "assistant", "content": response.content}
