"""CLI entry point for the ChemVision capability audit framework.

Usage
-----
python -m chemvision.audit.run --model <model_id_or_path> --dataset <jsonl_dir>

Optional flags
--------------
--output        Output directory for reports and JSON envelope (default: reports/)
--threshold     Accuracy threshold for degradation binary search (default: 0.7)
--backend       Model backend class: 'llava' (default) or 'qwen'
--no-degrade    Skip the DegradationTester and only run CapabilityMatrix
--score-fn      Answer scoring strategy: 'substring' (default) or 'exact'
--seed          Random seed for degradation sampling (default: 42)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m chemvision.audit.run",
        description="Evaluate a VLM against the ChemVision capability audit suite.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local path to the vision-language model.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Directory containing train.jsonl / val.jsonl (or a single JSONL file).",
    )
    p.add_argument(
        "--output",
        default="reports/",
        type=Path,
        help="Output directory for audit_report.md and reliability_envelope.json.",
    )
    p.add_argument(
        "--threshold",
        default=0.7,
        type=float,
        help="Accuracy threshold used by DegradationTester (default: 0.7).",
    )
    p.add_argument(
        "--backend",
        default="llava",
        choices=["llava", "qwen"],
        help="Model backend class to use.",
    )
    p.add_argument(
        "--no-degrade",
        action="store_true",
        help="Skip degradation testing (CapabilityMatrix only).",
    )
    p.add_argument(
        "--score-fn",
        default="substring",
        choices=["substring", "exact"],
        dest="score_fn",
        help="Answer scoring strategy (default: substring).",
    )
    p.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for dataset sampling (default: 42).",
    )
    return p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_records(dataset_path: Path) -> list:
    """Load ImageRecord objects from a JSONL file or directory of JSONL files."""
    from chemvision.data.schema import ImageRecord

    if dataset_path.is_dir():
        # Try val.jsonl first (smaller); fall back to train.jsonl
        for candidate in ("val.jsonl", "test.jsonl", "train.jsonl"):
            jsonl = dataset_path / candidate
            if jsonl.exists():
                dataset_path = jsonl
                break
        else:
            raise FileNotFoundError(
                f"No JSONL file found in {dataset_path}. "
                "Expected val.jsonl, test.jsonl, or train.jsonl."
            )

    records = []
    with open(dataset_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(ImageRecord.model_validate_json(line))
    return records


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(model_id: str, backend: str):
    from chemvision.models.config import ModelConfig

    cfg = ModelConfig(
        model_name_or_path=model_id,
        device="cuda",
        dtype="bfloat16",
    )
    if backend == "qwen":
        from chemvision.models.qwen_vl import QwenVLWrapper
        model = QwenVLWrapper(cfg)
    else:
        from chemvision.models.llava import LLaVAWrapper
        model = LLaVAWrapper(cfg)

    model.load()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data -------------------------------------------------------
    print(f"[audit] Loading dataset from {args.dataset} …")
    records = _load_records(Path(args.dataset))
    print(f"[audit] Loaded {len(records)} records.")

    # ---- Load model ------------------------------------------------------
    print(f"[audit] Loading model '{args.model}' (backend={args.backend}) …")
    model = _load_model(args.model, args.backend)
    print(f"[audit] Model ready: {model!r}")

    # ---- Capability matrix -----------------------------------------------
    from chemvision.audit.matrix import CapabilityMatrix, MatrixConfig

    print("[audit] Running CapabilityMatrix evaluation …")
    matrix_cfg = MatrixConfig(output_dir=output_dir, score_fn=args.score_fn)
    matrix = CapabilityMatrix(matrix_cfg).run_evaluation(model, records)

    # Print quick summary
    print("[audit] Capability matrix:")
    for task in matrix.TASK_TYPES:
        row = {d: matrix.get_score(task, d) for d in matrix.DIFFICULTIES}
        row_str = "  ".join(
            f"{d[:3]}={v:.2f}" if v == v else f"{d[:3]}=N/A"
            for d, v in row.items()
        )
        print(f"  {task:<30} {row_str}")

    # ---- Degradation testing ---------------------------------------------
    envelope = None
    if not args.no_degrade:
        from chemvision.audit.degradation import DegradationConfig, DegradationTester

        print(f"[audit] Running DegradationTester (threshold={args.threshold}) …")
        deg_cfg = DegradationConfig(
            threshold=args.threshold,
            seed=args.seed,
            output_dir=output_dir,
        )
        tester = DegradationTester(deg_cfg)
        envelope = tester.run(model, records)

        print("[audit] Reliability envelope:")
        for dtype, res in envelope.results.items():
            print(
                f"  {dtype:<22} critical={res.critical_param:.2f} "
                f"{res.param_unit:<14} robustness={res.robustness_label}"
            )
    else:
        print("[audit] Degradation testing skipped (--no-degrade).")

    # ---- Generate report -------------------------------------------------
    from chemvision.audit.report_generator import AuditReportGenerator

    print("[audit] Generating audit report …")
    gen = AuditReportGenerator(matrix, envelope)
    report_path = gen.generate(output_dir=output_dir)
    print(f"[audit] Report written to: {report_path}")

    envelope_path = output_dir / "reliability_envelope.json"
    if envelope_path.exists():
        print(f"[audit] Reliability envelope written to: {envelope_path}")

    print("[audit] Done.")


if __name__ == "__main__":
    main()
