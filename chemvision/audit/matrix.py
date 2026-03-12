"""CapabilityMatrix: 2-axis accuracy matrix across task_type × difficulty."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from chemvision.data.schema import ImageRecord
from chemvision.models.base import BaseVisionModel


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskType(StrEnum):
    SPATIAL_REASONING = "spatial_reasoning"
    COUNTING = "counting"
    CROSS_IMAGE_COMPARISON = "cross_image_comparison"
    ANOMALY_DETECTION = "anomaly_detection"
    CAPTION_VALIDATION = "caption_validation"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Config + result containers
# ---------------------------------------------------------------------------


class MatrixConfig(BaseModel):
    """Configuration for :class:`CapabilityMatrix`."""

    output_dir: Path = Field(Path("reports/"), description="Directory to save heatmap PNG.")
    score_fn: str = Field(
        "substring",
        description="Answer scoring strategy: 'exact' or 'substring'.",
    )
    unknown_task_type: str = Field(
        "spatial_reasoning",
        description="Fallback task_type when metadata key is absent.",
    )
    unknown_difficulty: str = Field(
        "medium",
        description="Fallback difficulty when ImageRecord.difficulty is absent.",
    )


@dataclass
class CellResult:
    """Accumulated accuracy for a single (task_type, difficulty) cell."""

    task_type: str
    difficulty: str
    num_correct: int = 0
    num_total: int = 0

    @property
    def accuracy(self) -> float:
        """Return accuracy in [0, 1], or NaN when no samples were evaluated."""
        return self.num_correct / self.num_total if self.num_total > 0 else math.nan

    def update(self, correct: bool) -> None:
        self.num_total += 1
        if correct:
            self.num_correct += 1


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CapabilityMatrix:
    """2D accuracy matrix across task_type × difficulty.

    The matrix is filled by :meth:`run_evaluation`, which drives the model on
    every :class:`~chemvision.data.schema.ImageRecord` in the provided
    dataset and accumulates binary correct/incorrect scores.

    Task type is read from ``record.metadata["task_type"]``; difficulty from
    ``record.difficulty``.  Unknown values fall back to the
    :class:`MatrixConfig` defaults.

    Example
    -------
    >>> cfg = MatrixConfig(output_dir=Path("reports/"))
    >>> matrix = CapabilityMatrix(cfg).run_evaluation(model, dataset)
    >>> matrix.get_score("counting", "easy")
    0.875
    >>> heatmap_path = matrix.export_heatmap()
    """

    TASK_TYPES: list[str] = [t.value for t in TaskType]
    DIFFICULTIES: list[str] = [d.value for d in Difficulty]

    def __init__(self, config: MatrixConfig | None = None) -> None:
        self.config = config or MatrixConfig()
        self._cells: dict[tuple[str, str], CellResult] = {
            (t, d): CellResult(task_type=t, difficulty=d)
            for t in self.TASK_TYPES
            for d in self.DIFFICULTIES
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_evaluation(
        self,
        model: BaseVisionModel,
        dataset: list[ImageRecord],
    ) -> CapabilityMatrix:
        """Evaluate *model* on every record in *dataset* and fill the matrix.

        Parameters
        ----------
        model:
            A loaded :class:`~chemvision.models.base.BaseVisionModel`.
        dataset:
            List of annotated :class:`~chemvision.data.schema.ImageRecord`.

        Returns
        -------
        CapabilityMatrix
            ``self``, so calls can be chained.
        """
        from PIL import Image as PILImage

        for record in dataset:
            task_type = str(
                record.metadata.get("task_type", self.config.unknown_task_type)
            )
            difficulty = record.difficulty or self.config.unknown_difficulty

            # Clamp to known values
            if task_type not in self.TASK_TYPES:
                task_type = self.config.unknown_task_type
            if difficulty not in self.DIFFICULTIES:
                difficulty = self.config.unknown_difficulty

            image = PILImage.open(record.image_path).convert("RGB")
            predicted = model.generate(image, record.question)
            correct = self._score_answer(predicted, record.answer, self.config.score_fn)
            self._cells[(task_type, difficulty)].update(correct)

        return self

    def get_score(self, task_type: str, difficulty: str) -> float:
        """Return accuracy for a single cell, or NaN if no samples."""
        return self._cells[(task_type, difficulty)].accuracy

    def get_cell(self, task_type: str, difficulty: str) -> CellResult:
        """Return the raw :class:`CellResult` for inspection."""
        return self._cells[(task_type, difficulty)]

    def to_array(self) -> list[list[float]]:
        """Return accuracy values as ``[task_types × difficulties]`` float list."""
        return [
            [self._cells[(t, d)].accuracy for d in self.DIFFICULTIES]
            for t in self.TASK_TYPES
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the matrix as a nested dict for JSON export."""
        return {
            t: {d: self._cells[(t, d)].accuracy for d in self.DIFFICULTIES}
            for t in self.TASK_TYPES
        }

    def export_heatmap(self, output_dir: Path | None = None) -> Path:
        """Render the capability matrix as a colour-coded heatmap PNG.

        NaN cells (no samples evaluated) are shown in a neutral grey.

        Parameters
        ----------
        output_dir:
            Override for :attr:`MatrixConfig.output_dir`.

        Returns
        -------
        Path
            Path to the saved PNG file.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required. Install with: pip install matplotlib"
            ) from exc

        out_dir = output_dir or self.config.output_dir
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        raw = self.to_array()
        data = np.array(raw, dtype=float)  # shape: [n_tasks, n_diffs]

        # Build a masked array so NaN cells render differently
        mask = np.isnan(data)
        display = np.where(mask, 0.0, data)

        fig, ax = plt.subplots(figsize=(6, 7))
        cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
        im = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

        # Overlay grey for NaN cells
        grey_cmap = matplotlib.colors.ListedColormap(["lightgrey"])  # type: ignore[attr-defined]
        ax.imshow(
            mask.astype(float),
            cmap=grey_cmap,
            vmin=0,
            vmax=1,
            alpha=mask.astype(float),
            aspect="auto",
        )

        # Axis labels
        ax.set_xticks(range(len(self.DIFFICULTIES)))
        ax.set_xticklabels(self.DIFFICULTIES, fontsize=10)
        ax.set_yticks(range(len(self.TASK_TYPES)))
        ax.set_yticklabels(
            [t.replace("_", "\n") for t in self.TASK_TYPES], fontsize=8
        )
        ax.set_xlabel("Difficulty", fontsize=11)
        ax.set_ylabel("Task Type", fontsize=11)
        ax.set_title("VLM Capability Matrix", fontsize=13, fontweight="bold")

        # Annotate cells with accuracy values
        for i, t in enumerate(self.TASK_TYPES):
            for j, d in enumerate(self.DIFFICULTIES):
                val = self._cells[(t, d)].accuracy
                label = f"{val:.2f}" if not math.isnan(val) else "—"
                color = "black" if (math.isnan(val) or 0.3 <= val <= 0.7) else "white"
                ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color)

        fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
        fig.tight_layout()

        out_path = out_dir / "capability_matrix.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_answer(predicted: str, ground_truth: str, strategy: str = "substring") -> bool:
        """Return True if *predicted* is considered correct for *ground_truth*.

        ``'exact'``     — case-insensitive exact match after stripping.
        ``'substring'`` — ground truth appears anywhere in the prediction.
        """
        pred = predicted.strip().lower()
        gt = ground_truth.strip().lower()
        if strategy == "exact":
            return pred == gt
        return gt in pred  # default: substring match
