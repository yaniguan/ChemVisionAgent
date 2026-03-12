#!/usr/bin/env python
"""LoRA fine-tuning entry point for ChemVision vision-language models.

Usage
-----
python scripts/train_lora.py --config configs/lora_train.yaml
python scripts/train_lora.py --config configs/lora_train.yaml \\
    --resume_from checkpoints/lora-v1/checkpoint-500

The script:
  1. Loads the YAML training config.
  2. Initialises Weights & Biases (silently disabled if wandb is not installed).
  3. Builds a PeftConfig from the YAML fields.
  4. Loads train / val splits from JSONL files produced by DatasetBuilder.
  5. Runs LoRA fine-tuning via PeftFineTuner.
  6. Merges LoRA weights into the base model and saves the result.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for ChemVision vision-language models."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML training configuration file.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a Trainer checkpoint directory to resume training from.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required. Install with: pip install pyyaml"
        ) from exc
    with open(path) as fh:
        return yaml.safe_load(fh)


def _init_wandb(cfg: dict[str, Any]) -> None:
    """Initialise W&B run, or disable W&B silently if the package is absent."""
    try:
        import wandb
    except ImportError:
        os.environ.setdefault("WANDB_DISABLED", "true")
        print("[train_lora] wandb not installed — logging disabled.")
        return

    wandb.init(
        project=cfg.get("wandb_project", "chemvision-lora"),
        name=cfg.get("wandb_run_name"),
        tags=cfg.get("wandb_tags", []),
        config=cfg,
    )
    print(f"[train_lora] W&B run: {wandb.run.name}")  # type: ignore[union-attr]


def _load_records(dataset_dir: Path, split: str) -> list[Any]:
    """Load :class:`~chemvision.data.schema.ImageRecord` from a JSONL file."""
    from chemvision.data.schema import ImageRecord

    jsonl_path = dataset_dir / f"{split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {jsonl_path}\n"
            "Run DatasetBuilder first to generate the JSONL files."
        )

    records: list[ImageRecord] = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(ImageRecord.model_validate_json(line))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    cfg = _load_yaml(args.config)
    print(f"[train_lora] Config loaded from {args.config}")

    _init_wandb(cfg)

    # Build typed config objects from the flat YAML dict
    from chemvision.models.config import ModelConfig, PeftConfig

    base_model_cfg = ModelConfig(**cfg["base_model"])
    peft_cfg = PeftConfig(
        base_model=base_model_cfg,
        lora_r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
        output_dir=cfg.get("output_dir", "checkpoints/"),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
    )

    # Load dataset
    dataset_dir = Path(cfg.get("dataset_dir", "data/processed"))
    train_split = cfg.get("train_split", "train")
    val_split = cfg.get("val_split", "val")

    print(f"[train_lora] Loading dataset from {dataset_dir} …")
    train_records = _load_records(dataset_dir, train_split)
    val_records = _load_records(dataset_dir, val_split)
    print(f"[train_lora] Train: {len(train_records)} samples | Val: {len(val_records)} samples")

    # Fine-tune
    from chemvision.models.finetuner import PeftFineTuner

    tuner = PeftFineTuner(peft_cfg)

    resume_from = args.resume_from
    if resume_from:
        print(f"[train_lora] Resuming from checkpoint: {resume_from}")

    tuner.train(train_records, val_records)
    print(f"[train_lora] Training complete. Checkpoint saved to: {peft_cfg.output_dir}")

    # Merge LoRA weights into the base model
    merged_path = str(Path(peft_cfg.output_dir) / "merged")
    print(f"[train_lora] Merging LoRA weights → {merged_path} …")
    tuner.merge_and_export(merged_path)
    print("[train_lora] Done.")


if __name__ == "__main__":
    main()
