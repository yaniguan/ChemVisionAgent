#!/usr/bin/env python
"""Conditioning ablation benchmark for the upgraded SELFIES generator.

Compares 4 model variants across 3 seeds to measure the impact of:
  - FiLM conditioning (always on in the new architecture)
  - Auxiliary property prediction loss (prop_loss_weight)
  - Classifier-free guidance (guidance_scale + cfg_drop_prob)

Saves results to benchmarks/results/conditioning_ablation.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chemvision.data.zinc250k import load_zinc250k, zinc250k_splits
from chemvision.eval.moses_metrics import compute_moses_metrics
from chemvision.generation.selfies_gen import SELFIESGenConfig, SELFIESGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_MOLECULES = 12000
SEEDS = [42, 123, 456]
EPOCHS = 40
BATCH_SIZE = 64
EMBED_DIM = 128
N_LAYERS = 3
N_GEN_PER_TARGET = 1       # molecules per target property vector
N_TEST_TARGETS = 200        # number of test property vectors to condition on

# The ZINC250K dataset has 3 properties: LogP, QED, SAS
COND_DIM = 3

VARIANTS: dict[str, dict] = {
    "baseline": {
        "description": "FiLM present but no aux loss, no CFG (old behavior equivalent)",
        "guidance_scale": 0.0,
        "prop_loss_weight": 0.0,
        "cfg_drop_prob": 0.0,
    },
    "film_only": {
        "description": "FiLM with gradient signal via cross-attention only (new arch baseline)",
        "guidance_scale": 0.0,
        "prop_loss_weight": 0.0,
        "cfg_drop_prob": 0.0,
    },
    "film_aux": {
        "description": "FiLM + auxiliary property prediction loss",
        "guidance_scale": 0.0,
        "prop_loss_weight": 0.1,
        "cfg_drop_prob": 0.0,
    },
    "film_aux_cfg": {
        "description": "FiLM + aux loss + classifier-free guidance",
        "guidance_scale": 1.5,
        "prop_loss_weight": 0.1,
        "cfg_drop_prob": 0.1,
    },
}


def compute_property_alignment(
    generated_smiles: list[str | None],
    target_qeds: np.ndarray,
    tolerance: float = 0.1,
) -> dict[str, float]:
    """Compute property alignment metrics between generated and target QED values.

    Returns dict with: qed_mae, qed_pearson, qed_hit_rate
    """
    from rdkit import Chem
    from rdkit.Chem import QED

    gen_qeds = []
    matched_targets = []

    for smi, tgt_qed in zip(generated_smiles, target_qeds):
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            gen_qeds.append(QED.qed(mol))
            matched_targets.append(tgt_qed)
        except (ValueError, RuntimeError):
            continue

    if len(gen_qeds) < 2:
        return {"qed_mae": float("nan"), "qed_pearson": float("nan"), "qed_hit_rate": 0.0}

    gen_qeds = np.array(gen_qeds)
    matched_targets = np.array(matched_targets)

    mae = float(np.mean(np.abs(gen_qeds - matched_targets)))
    corr, _ = pearsonr(gen_qeds, matched_targets)
    hit_rate = float(np.mean(np.abs(gen_qeds - matched_targets) < tolerance))

    return {
        "qed_mae": round(mae, 4),
        "qed_pearson": round(float(corr), 4),
        "qed_hit_rate": round(hit_rate, 4),
    }


def run_single(
    variant_name: str,
    variant_cfg: dict,
    seed: int,
    train_smiles: list[str],
    train_props: np.ndarray,
    test_smiles: list[str],
    test_props: np.ndarray,
) -> dict:
    """Train one variant with one seed, generate, and evaluate."""
    print(f"\n{'='*70}", flush=True)
    print(f"  Variant: {variant_name} | Seed: {seed}", flush=True)
    print(f"  {variant_cfg.get('description', '')}", flush=True)
    print(f"{'='*70}", flush=True)

    config = SELFIESGenConfig(
        cond_dim=COND_DIM,
        embed_dim=EMBED_DIM,
        n_heads=4,
        n_layers=N_LAYERS,
        max_len=80,
        dropout=0.1,
        learning_rate=3e-4,
        temperature=1.0,
        seed=seed,
        prop_loss_weight=variant_cfg["prop_loss_weight"],
        guidance_scale=variant_cfg["guidance_scale"],
        cfg_drop_prob=variant_cfg["cfg_drop_prob"],
    )

    gen = SELFIESGenerator(config)

    # Train
    t0 = time.time()
    train_result = gen.train(
        train_smiles,
        train_props,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True,
    )
    train_time = time.time() - t0
    print(
        f"  Training done: {train_result.epochs} epochs, "
        f"final_loss={train_result.final_loss:.4f}, "
        f"time={train_time:.1f}s",
        flush=True,
    )

    # Select test targets for generation
    rng = np.random.RandomState(seed)
    n_targets = min(N_TEST_TARGETS, len(test_props))
    target_idx = rng.choice(len(test_props), n_targets, replace=False)
    target_props = test_props[target_idx]

    # Generate molecules
    gs_for_gen = variant_cfg["guidance_scale"]
    t1 = time.time()
    results = gen.generate(
        target_properties=target_props,
        n_per_target=N_GEN_PER_TARGET,
        guidance_scale=gs_for_gen,
    )
    gen_time = time.time() - t1
    print(f"  Generated {len(results)} molecules in {gen_time:.1f}s", flush=True)

    # Extract SMILES
    generated_smiles = [r.smiles for r in results]
    valid_smiles = [s for s in generated_smiles if s is not None]
    print(f"  Valid: {len(valid_smiles)}/{len(generated_smiles)}", flush=True)

    # MOSES metrics
    moses = compute_moses_metrics(generated_smiles, training_smiles=train_smiles)
    print(f"  MOSES: validity={moses.validity:.3f}, uniqueness={moses.uniqueness:.3f}, "
          f"novelty={moses.novelty:.3f}, int_div={moses.int_div_1:.4f}, "
          f"scaffold_div={moses.scaffold_diversity:.4f}, fcd={moses.fcd:.2f}",
          flush=True)

    # Property alignment (QED is index 1 in [LogP, QED, SAS])
    target_qeds = target_props[:, 1]
    # Expand targets to match n_per_target
    expanded_target_qeds = np.repeat(target_qeds, N_GEN_PER_TARGET)
    prop_align = compute_property_alignment(generated_smiles, expanded_target_qeds)
    print(f"  Property alignment: MAE={prop_align['qed_mae']:.4f}, "
          f"Pearson={prop_align['qed_pearson']:.4f}, "
          f"hit_rate={prop_align['qed_hit_rate']:.3f}",
          flush=True)

    return {
        "train_epochs": train_result.epochs,
        "train_final_loss": round(train_result.final_loss, 4),
        "train_time_s": round(train_time, 1),
        "gen_time_s": round(gen_time, 1),
        "n_generated": len(generated_smiles),
        "n_valid": len(valid_smiles),
        "validity": round(moses.validity, 4),
        "uniqueness": round(moses.uniqueness, 4),
        "novelty": round(moses.novelty, 4),
        "int_div": round(moses.int_div_1, 4),
        "int_div_2": round(moses.int_div_2, 4),
        "scaffold_diversity": round(moses.scaffold_diversity, 4),
        "fcd": round(moses.fcd, 4),
        "qed_mae": prop_align["qed_mae"],
        "qed_pearson": prop_align["qed_pearson"],
        "qed_hit_rate": prop_align["qed_hit_rate"],
    }


def aggregate_seeds(seed_results: dict[int, dict]) -> tuple[dict, dict]:
    """Compute mean and std across seeds for all numeric metrics."""
    keys = [k for k in list(seed_results.values())[0] if isinstance(list(seed_results.values())[0][k], (int, float))]
    mean_dict = {}
    std_dict = {}
    for k in keys:
        vals = [seed_results[s][k] for s in seed_results]
        # Skip NaN values
        valid_vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if valid_vals:
            mean_dict[k] = round(float(np.mean(valid_vals)), 4)
            std_dict[k] = round(float(np.std(valid_vals)), 4)
        else:
            mean_dict[k] = float("nan")
            std_dict[k] = float("nan")
    return mean_dict, std_dict


def print_summary_table(all_results: dict) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 100, flush=True)
    print("CONDITIONING ABLATION RESULTS SUMMARY", flush=True)
    print("=" * 100, flush=True)

    metrics = [
        ("validity", "Validity", ".3f"),
        ("uniqueness", "Unique", ".3f"),
        ("novelty", "Novelty", ".3f"),
        ("int_div", "IntDiv", ".4f"),
        ("scaffold_diversity", "ScaffDiv", ".4f"),
        ("fcd", "FCD", ".2f"),
        ("qed_mae", "QED MAE", ".4f"),
        ("qed_pearson", "QED r", ".4f"),
        ("qed_hit_rate", "Hit Rate", ".3f"),
        ("train_final_loss", "Loss", ".4f"),
    ]

    # Header
    header = f"{'Variant':<20}"
    for _, label, _ in metrics:
        header += f" {label:>14}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for variant_name in all_results:
        vdata = all_results[variant_name]
        m = vdata["mean"]
        s = vdata["std"]
        row = f"{variant_name:<20}"
        for key, _, fmt in metrics:
            mv = m.get(key, float("nan"))
            sv = s.get(key, float("nan"))
            if np.isnan(mv):
                row += f" {'N/A':>14}"
            else:
                row += f" {f'{mv:{fmt}}+/-{sv:{fmt}}':>14}"
        print(row, flush=True)

    print("=" * 100, flush=True)
    print("FCD: lower is better. Hit Rate: fraction within 0.1 QED tolerance.", flush=True)


def main() -> None:
    print("=" * 70, flush=True)
    print("CONDITIONING ABLATION BENCHMARK", flush=True)
    print(f"Dataset: ZINC250K (max {MAX_MOLECULES} molecules)", flush=True)
    print(f"Variants: {list(VARIANTS.keys())}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
          f"embed_dim={EMBED_DIM}, n_layers={N_LAYERS}", flush=True)
    print("=" * 70, flush=True)

    # Load data
    print("\nLoading ZINC250K...", flush=True)
    smiles, properties = load_zinc250k(max_molecules=MAX_MOLECULES)
    print(f"Loaded {len(smiles)} molecules, properties shape: {properties.shape}", flush=True)

    # Split
    splits = zinc250k_splits(smiles, properties, train_ratio=0.8, val_ratio=0.1, seed=42)
    train_smiles, train_props = splits["train"]
    test_smiles, test_props = splits["test"]
    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}", flush=True)

    # Run all variants and seeds
    all_results: dict[str, dict] = {}
    total_start = time.time()

    for variant_name, variant_cfg in VARIANTS.items():
        seed_results: dict[int, dict] = {}

        for seed in SEEDS:
            result = run_single(
                variant_name=variant_name,
                variant_cfg=variant_cfg,
                seed=seed,
                train_smiles=train_smiles,
                train_props=train_props,
                test_smiles=test_smiles,
                test_props=test_props,
            )
            seed_results[seed] = result

        mean_metrics, std_metrics = aggregate_seeds(seed_results)
        all_results[variant_name] = {
            "seeds": seed_results,
            "mean": mean_metrics,
            "std": std_metrics,
        }

    total_time = time.time() - total_start
    print(f"\nTotal benchmark time: {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)

    # Print summary
    print_summary_table(all_results)

    # Save results
    output = {
        "metadata": {
            "dataset": "zinc250k",
            "max_molecules": MAX_MOLECULES,
            "n_train": len(train_smiles),
            "n_test": len(test_smiles),
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "n_layers": N_LAYERS,
            "n_gen_per_target": N_GEN_PER_TARGET,
            "n_test_targets": N_TEST_TARGETS,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_time_s": round(total_time, 1),
        },
        "variants": all_results,
    }

    results_dir = REPO_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "conditioning_ablation.json"

    # Convert any numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            converted = _convert(obj)
            if converted is not obj:
                return converted
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
