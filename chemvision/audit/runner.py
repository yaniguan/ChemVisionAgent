"""AuditRunner — evaluates a model against registered skill probes."""

from __future__ import annotations

import json
from pathlib import Path

from chemvision.audit.config import AuditConfig
from chemvision.audit.matrix import CapabilityMatrix, MatrixConfig
from chemvision.audit.report import AuditReport, SkillScore
from chemvision.data.schema import ImageRecord
from chemvision.models.base import BaseVisionModel
from chemvision.skills.registry import list_skills


class AuditRunner:
    """Drive a full audit run across all configured skill probes.

    Discovers :class:`~chemvision.skills.base.BaseSkill` probes from the
    global registry, runs them against the benchmark dataset, and returns a
    structured :class:`AuditReport`.  Optionally also fills a
    :class:`~chemvision.audit.matrix.CapabilityMatrix`.

    Example
    -------
    >>> cfg = AuditConfig(model_name="qwen-vl-7b", benchmark_dir=Path("benchmarks/"))
    >>> runner = AuditRunner(cfg, model)
    >>> report = runner.run()
    >>> print(report.summary())
    """

    def __init__(self, config: AuditConfig, model: BaseVisionModel) -> None:
        self.config = config
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_probes(self) -> list[str]:
        """Return skill probe names available for the configured skill list.

        When ``config.skill_names`` is empty all registered skills are used.
        """
        all_skills = list_skills()
        if self.config.skill_names:
            return [s for s in self.config.skill_names if s in all_skills]
        return all_skills

    def run(self) -> AuditReport:
        """Execute all probes and aggregate results into an :class:`AuditReport`."""
        from chemvision.skills.registry import get_skill
        from PIL import Image as PILImage

        records = self._load_records()
        probe_names = self.discover_probes()
        skill_scores: list[SkillScore] = []

        for probe_name in probe_names:
            skill = get_skill(probe_name)
            num_correct = 0
            num_total = 0
            for record in records:
                image = PILImage.open(record.image_path).convert("RGB")
                result = skill(image, self.model)
                # Use substring match against ground-truth answer
                predicted = result.raw_output.strip().lower()
                gt = record.answer.strip().lower()
                if gt in predicted:
                    num_correct += 1
                num_total += 1

            accuracy = num_correct / num_total if num_total > 0 else 0.0
            skill_scores.append(
                SkillScore(
                    skill_name=probe_name,
                    accuracy=accuracy,
                    num_samples=num_total,
                )
            )

        overall = (
            sum(s.accuracy for s in skill_scores) / len(skill_scores)
            if skill_scores
            else 0.0
        )
        report = AuditReport(
            model_name=self.config.model_name,
            skill_scores=skill_scores,
            overall_accuracy=overall,
        )
        self.save_report(report)
        return report

    def save_report(self, report: AuditReport) -> None:
        """Persist *report* as JSON to ``config.output_dir``."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "audit_report.json"
        path.write_text(report.model_dump_json(indent=2))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_records(self) -> list[ImageRecord]:
        """Load ImageRecord objects from the benchmark directory JSONL files."""
        bench_dir = Path(self.config.benchmark_dir)
        records: list[ImageRecord] = []

        for jsonl in sorted(bench_dir.glob("*.jsonl")):
            with open(jsonl) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(ImageRecord.model_validate_json(line))

        return records
