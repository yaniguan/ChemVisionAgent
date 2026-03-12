"""AuditReportGenerator: markdown report with embedded heatmap and recommendations."""

from __future__ import annotations

import base64
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chemvision.audit.degradation import DegradationResult, ReliabilityEnvelope
from chemvision.audit.matrix import CapabilityMatrix


# ---------------------------------------------------------------------------
# Deployment recommendation thresholds
# ---------------------------------------------------------------------------

_ACCURACY_TIERS = [
    (0.85, "**Production-ready** — deploy with standard monitoring."),
    (0.70, "**Supervised deployment** — require human review for edge cases."),
    (0.50, "**Research / exploration** — not suitable for automated production."),
    (0.0,  "**Not recommended** — insufficient capability for this task."),
]

_ROBUSTNESS_TIERS = {
    "high":     "Robust ✔",
    "moderate": "Moderate ⚠",
    "low":      "Sensitive ✘",
}

_DEGRADATION_GUIDANCE: dict[str, dict[str, str]] = {
    "gaussian_noise": {
        "high":     "Deploy confidently in noisy acquisition environments.",
        "moderate": "Pre-process with denoising before inference.",
        "low":      "Ensure clean image acquisition; model is noise-sensitive.",
    },
    "jpeg_compression": {
        "high":     "Tolerates heavily compressed images; safe for web delivery.",
        "moderate": "Use quality ≥ 60 for reliable results.",
        "low":      "Require lossless or near-lossless image formats.",
    },
    "occlusion": {
        "high":     "Handles partial obstructions; suitable for microscopy with debris.",
        "moderate": "Acceptable when occlusion < 20 % of the region of interest.",
        "low":      "Ensure full visibility of target region before inference.",
    },
    "downsampling": {
        "high":     "Robust to low-resolution inputs; suitable for thumbnail analysis.",
        "moderate": "Recommend ≥ 50 % of native resolution for reliable inference.",
        "low":      "Use full-resolution images; model degrades rapidly when downsampled.",
    },
    "color_shift": {
        "high":     "Stable across staining variations and colour-calibration drift.",
        "moderate": "Apply colour normalisation for cross-instrument comparisons.",
        "low":      "Standardise colour calibration; model is colour-sensitive.",
    },
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class AuditReportGenerator:
    """Assemble a markdown audit report from matrix and degradation results.

    The report includes:

    * A Base64-embedded capability heatmap PNG.
    * An accuracy table per task × difficulty.
    * A robustness summary table with critical parameters.
    * Per-task deployment recommendations based on accuracy tiers.
    * Per-degradation guidance strings.

    Example
    -------
    >>> gen = AuditReportGenerator(matrix, envelope)
    >>> report_path = gen.generate(output_dir=Path("reports/"))
    """

    def __init__(
        self,
        matrix: CapabilityMatrix,
        envelope: ReliabilityEnvelope | None = None,
    ) -> None:
        self.matrix = matrix
        self.envelope = envelope

    def generate(
        self,
        output_dir: Path | None = None,
        heatmap_path: Path | None = None,
    ) -> Path:
        """Render the full markdown report and write it to disk.

        Parameters
        ----------
        output_dir:
            Directory where ``audit_report.md`` will be written.
            Defaults to :attr:`CapabilityMatrix.config.output_dir`.
        heatmap_path:
            Pre-generated heatmap PNG.  When ``None``, :meth:`export_heatmap`
            is called automatically.

        Returns
        -------
        Path
            Path to the written ``audit_report.md``.
        """
        out_dir = Path(output_dir or self.matrix.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Ensure heatmap exists
        if heatmap_path is None:
            heatmap_path = self.matrix.export_heatmap(out_dir)

        sections: list[str] = [
            self._section_header(),
            self._section_heatmap(heatmap_path),
            self._section_accuracy_table(),
            self._section_degradation_table(),
            self._section_recommendations(),
        ]

        content = "\n\n".join(s for s in sections if s)
        report_path = out_dir / "audit_report.md"
        report_path.write_text(content, encoding="utf-8")
        return report_path

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _section_header(self) -> str:
        model_name = (
            self.envelope.model_name
            if self.envelope
            else "unknown"
        )
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"# ChemVision VLM Capability Audit Report\n\n"
            f"**Model:** `{model_name}`  \n"
            f"**Generated:** {now}"
        )

    def _section_heatmap(self, heatmap_path: Path) -> str:
        b64 = _encode_image_base64(heatmap_path)
        return (
            "## 1. Capability Matrix\n\n"
            "The heatmap below shows model accuracy across five task types "
            "(rows) and three difficulty levels (columns).  "
            "Green = high accuracy; red = low accuracy; grey = no samples.\n\n"
            f'<img src="data:image/png;base64,{b64}" '
            f'alt="Capability Matrix Heatmap" width="520"/>'
        )

    def _section_accuracy_table(self) -> str:
        header = (
            "## 2. Accuracy by Task × Difficulty\n\n"
            "| Task Type | Easy | Medium | Hard | Row Mean |\n"
            "|-----------|------|--------|------|----------|\n"
        )
        rows: list[str] = []
        for t in self.matrix.TASK_TYPES:
            scores = [self.matrix.get_score(t, d) for d in self.matrix.DIFFICULTIES]
            valid = [s for s in scores if not math.isnan(s)]
            mean = sum(valid) / len(valid) if valid else math.nan
            cells = [f"{s:.2f}" if not math.isnan(s) else "—" for s in scores]
            mean_str = f"{mean:.2f}" if not math.isnan(mean) else "—"
            rows.append(f"| `{t}` | {' | '.join(cells)} | {mean_str} |")
        return header + "\n".join(rows)

    def _section_degradation_table(self) -> str:
        if self.envelope is None:
            return ""

        header = (
            "## 3. Robustness to Image Degradations\n\n"
            f"Accuracy threshold: **{self.envelope.threshold:.0%}**\n\n"
            "| Degradation | Critical Parameter | Accuracy @ Critical | Robustness |\n"
            "|-------------|-------------------|---------------------|------------|\n"
        )
        rows: list[str] = []
        for deg_type, res in self.envelope.results.items():
            label = _ROBUSTNESS_TIERS.get(res.robustness_label, res.robustness_label)
            param_str = f"{res.param_name} = {res.critical_param:.2f} {res.param_unit}"
            acc_str = f"{res.accuracy_at_critical:.2f}"
            rows.append(f"| `{deg_type}` | {param_str} | {acc_str} | {label} |")
        return header + "\n".join(rows)

    def _section_recommendations(self) -> str:
        parts = ["## 4. Deployment Recommendations\n"]

        # Per-task accuracy-based recommendations
        parts.append("### 4.1 By Task Type\n")
        for t in self.matrix.TASK_TYPES:
            scores = [
                self.matrix.get_score(t, d)
                for d in self.matrix.DIFFICULTIES
            ]
            valid = [s for s in scores if not math.isnan(s)]
            mean = sum(valid) / len(valid) if valid else math.nan

            label, rec = _accuracy_recommendation(mean)
            parts.append(f"**`{t}`** ({_fmt_pct(mean)}) — {rec}")

        # Per-degradation recommendations
        if self.envelope is not None:
            parts.append("\n### 4.2 By Degradation Type\n")
            for deg_type, res in self.envelope.results.items():
                guidance = _DEGRADATION_GUIDANCE.get(deg_type, {}).get(
                    res.robustness_label, ""
                )
                rob = _ROBUSTNESS_TIERS.get(res.robustness_label, res.robustness_label)
                parts.append(f"**`{deg_type}`** [{rob}] — {guidance}")

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_image_base64(path: Path) -> str:
    """Read a PNG/JPEG file and return its Base64-encoded string."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _accuracy_recommendation(mean_accuracy: float) -> tuple[str, str]:
    """Return (tier_label, recommendation_text) for *mean_accuracy*."""
    if math.isnan(mean_accuracy):
        return ("no data", "No samples evaluated — cannot assess.")
    for threshold, text in _ACCURACY_TIERS:
        if mean_accuracy >= threshold:
            return (f"{mean_accuracy:.0%}", text)
    return ("0%", _ACCURACY_TIERS[-1][1])


def _fmt_pct(value: float) -> str:
    return f"{value:.0%}" if not math.isnan(value) else "n/a"
