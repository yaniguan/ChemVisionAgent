"""Model registry and experiment tracker.

Provides versioned model storage and structured experiment logging
so every training run is reproducible and comparable.

ModelRegistry: save/load/list/compare trained model checkpoints.
ExperimentTracker: log hyperparams, metrics, and artifacts per run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Experiment Tracker
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRun:
    """One training or evaluation run."""

    run_id: str = ""
    experiment_name: str = ""
    started_at: str = ""
    finished_at: str = ""
    status: str = "running"                # running | completed | failed
    hyperparams: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def duration_s(self) -> float:
        if not self.started_at or not self.finished_at:
            return 0.0
        t0 = datetime.fromisoformat(self.started_at)
        t1 = datetime.fromisoformat(self.finished_at)
        return (t1 - t0).total_seconds()


class ExperimentTracker:
    """JSON-backed experiment tracker.

    Example
    -------
    >>> tracker = ExperimentTracker(Path("experiments"))
    >>> run = tracker.start_run("csca_v1", hyperparams={"lr": 1e-3, "epochs": 100})
    >>> tracker.log_metrics(run.run_id, {"loss": 0.12, "recall_at_1": 0.85})
    >>> tracker.end_run(run.run_id, status="completed")
    >>> tracker.list_runs()
    [ExperimentRun(run_id='csca_v1_20260402_...', ...)]
    """

    def __init__(self, base_dir: Path | str) -> None:
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "runs.json"

    def start_run(
        self,
        experiment_name: str,
        hyperparams: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str = "",
    ) -> ExperimentRun:
        """Create a new experiment run."""
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%dT%H%M%S%f")
        run_id = f"{experiment_name}_{ts}"
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            started_at=now,
            hyperparams=hyperparams or {},
            tags=tags or [],
            notes=notes,
        )
        self._save_run(run)
        return run

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        """Append metrics to an existing run."""
        run = self._load_run(run_id)
        if run is None:
            raise KeyError(f"Run not found: {run_id}")
        run.metrics.update(metrics)
        self._save_run(run)

    def log_artifact(self, run_id: str, artifact_path: str) -> None:
        """Register an artifact (model checkpoint, figure, etc.) for a run."""
        run = self._load_run(run_id)
        if run is None:
            raise KeyError(f"Run not found: {run_id}")
        run.artifacts.append(artifact_path)
        self._save_run(run)

    def end_run(self, run_id: str, status: str = "completed") -> ExperimentRun:
        """Mark a run as finished."""
        run = self._load_run(run_id)
        if run is None:
            raise KeyError(f"Run not found: {run_id}")
        run.finished_at = datetime.now(timezone.utc).isoformat()
        run.status = status
        self._save_run(run)
        return run

    def list_runs(self, experiment_name: str | None = None) -> list[ExperimentRun]:
        """List all runs, optionally filtered by experiment name."""
        runs = self._load_all()
        if experiment_name:
            runs = [r for r in runs if r.experiment_name == experiment_name]
        return sorted(runs, key=lambda r: r.started_at, reverse=True)

    def get_run(self, run_id: str) -> ExperimentRun | None:
        return self._load_run(run_id)

    def compare_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        """Return a comparison table of metrics across runs."""
        rows = []
        for rid in run_ids:
            run = self._load_run(rid)
            if run:
                rows.append({"run_id": rid, "status": run.status, **run.hyperparams, **run.metrics})
        return rows

    def _save_run(self, run: ExperimentRun) -> None:
        all_runs = self._load_all()
        all_runs = [r for r in all_runs if r.run_id != run.run_id]
        all_runs.append(run)
        with open(self._index_path, "w") as f:
            json.dump([asdict(r) for r in all_runs], f, indent=2, default=str)

    def _load_run(self, run_id: str) -> ExperimentRun | None:
        for run in self._load_all():
            if run.run_id == run_id:
                return run
        return None

    def _load_all(self) -> list[ExperimentRun]:
        if not self._index_path.exists():
            return []
        with open(self._index_path) as f:
            data = json.load(f)
        return [ExperimentRun(**d) for d in data]


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


@dataclass
class ModelVersion:
    """Metadata for one registered model version."""

    model_name: str
    version: int
    created_at: str
    checkpoint_path: str
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    sha256: str = ""
    run_id: str = ""                       # link to experiment tracker


class ModelRegistry:
    """Versioned model checkpoint storage.

    Example
    -------
    >>> registry = ModelRegistry(Path("model_registry"))
    >>> registry.register("csca", "/tmp/csca_checkpoint.pt",
    ...                   config={"latent_dim": 128}, metrics={"R@1": 0.85})
    >>> registry.latest("csca")
    ModelVersion(model_name='csca', version=1, ...)
    >>> registry.list_versions("csca")
    [ModelVersion(...)]
    """

    def __init__(self, base_dir: Path | str) -> None:
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "registry.json"

    def register(
        self,
        model_name: str,
        checkpoint_path: str | Path,
        config: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        run_id: str = "",
    ) -> ModelVersion:
        """Register a new model version. Copies checkpoint to registry dir."""
        cp = Path(checkpoint_path)
        if not cp.exists():
            raise FileNotFoundError(f"Checkpoint not found: {cp}")

        versions = self.list_versions(model_name)
        next_version = max((v.version for v in versions), default=0) + 1

        # Copy checkpoint to registry
        dest_dir = self._dir / model_name / f"v{next_version}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / cp.name
        shutil.copy2(cp, dest)

        # Compute checksum
        sha = hashlib.sha256(cp.read_bytes()).hexdigest()[:16]

        mv = ModelVersion(
            model_name=model_name,
            version=next_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            checkpoint_path=str(dest),
            config=config or {},
            metrics=metrics or {},
            sha256=sha,
            run_id=run_id,
        )

        all_versions = self._load_all()
        all_versions.append(mv)
        self._save_all(all_versions)
        return mv

    def latest(self, model_name: str) -> ModelVersion | None:
        """Return the latest version of a model."""
        versions = self.list_versions(model_name)
        return versions[-1] if versions else None

    def get_version(self, model_name: str, version: int) -> ModelVersion | None:
        """Return a specific version."""
        for v in self.list_versions(model_name):
            if v.version == version:
                return v
        return None

    def list_versions(self, model_name: str) -> list[ModelVersion]:
        """List all versions of a model, sorted by version number."""
        all_v = self._load_all()
        return sorted(
            [v for v in all_v if v.model_name == model_name],
            key=lambda v: v.version,
        )

    def list_models(self) -> list[str]:
        """Return all registered model names."""
        return sorted(set(v.model_name for v in self._load_all()))

    def compare(self, model_name: str) -> list[dict[str, Any]]:
        """Return a comparison table across all versions of a model."""
        return [
            {"version": v.version, "sha256": v.sha256, **v.config, **v.metrics}
            for v in self.list_versions(model_name)
        ]

    def _save_all(self, versions: list[ModelVersion]) -> None:
        with open(self._index_path, "w") as f:
            json.dump([asdict(v) for v in versions], f, indent=2, default=str)

    def _load_all(self) -> list[ModelVersion]:
        if not self._index_path.exists():
            return []
        with open(self._index_path) as f:
            data = json.load(f)
        return [ModelVersion(**d) for d in data]
