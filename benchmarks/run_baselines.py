#!/usr/bin/env python3
"""Run standard baselines and compute MOSES metrics for fair comparison.

Loads ZINC250K, trains each baseline (CharRNN, Fragment recombination,
SMILES augmentation), generates molecules, and computes MOSES metrics.

Results are saved to benchmarks/results/baseline_results.json.

Usage
-----
    python benchmarks/run_baselines.py [--n_train 10000] [--n_gen 1000] [--epochs 30]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chemvision.core.reproducibility import set_global_seed

set_global_seed(42)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run molecular generation baselines.")
    p.add_argument("--n_train", type=int, default=10000, help="Training set size from ZINC250K")
    p.add_argument("--n_gen", type=int, default=1000, help="Molecules to generate per baseline")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs for CharRNN")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for CharRNN")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    all_results: dict[str, dict] = {}

    # ===================================================================
    # PHASE 1: LOAD ZINC250K
    # ===================================================================

    print("=" * 70)
    print(f"PHASE 1: LOADING ZINC250K (using {args.n_train} molecules)")
    print("=" * 70)

    from chemvision.data.zinc250k import load_zinc250k, zinc250k_splits

    smiles_all, props_all = load_zinc250k(max_molecules=args.n_train + 2000)
    splits = zinc250k_splits(smiles_all, props_all, seed=42)

    train_smiles, _ = splits["train"]
    test_smiles, _ = splits["test"]
    print(f"  Train: {len(train_smiles)}, Test: {len(test_smiles)}")
    print()

    from chemvision.eval.moses_metrics import compute_moses_metrics

    # ===================================================================
    # PHASE 2: CharRNN BASELINE
    # ===================================================================

    print("=" * 70)
    print("PHASE 2: CharRNN BASELINE")
    print("=" * 70)

    from benchmarks.baselines import CharRNNBaseline

    charrnn = CharRNNBaseline(hidden_dim=256, n_layers=2, dropout=0.2)

    print(f"  Training CharRNN ({args.epochs} epochs)...")
    t0 = time.perf_counter()
    train_info = charrnn.train(
        train_smiles,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    train_time = time.perf_counter() - t0
    print(f"  Trained in {train_time:.1f}s, "
          f"epochs={train_info['epochs']}, "
          f"final_loss={train_info['final_loss']:.4f}")

    print(f"  Generating {args.n_gen} molecules...")
    t0 = time.perf_counter()
    charrnn_smiles = charrnn.generate(n=args.n_gen, temperature=0.8)
    gen_time = time.perf_counter() - t0
    print(f"  Generated in {gen_time:.1f}s")

    m_charrnn = compute_moses_metrics(charrnn_smiles, train_smiles)
    all_results["CharRNN"] = {
        **asdict(m_charrnn),
        "train_time_s": train_time,
        "gen_time_s": gen_time,
        "train_info": train_info,
    }
    print(f"  V={m_charrnn.validity:.1%} U={m_charrnn.uniqueness:.1%} "
          f"N={m_charrnn.novelty:.1%} IntDiv={m_charrnn.int_div_1:.3f}")
    print()

    # ===================================================================
    # PHASE 3: FRAGMENT RECOMBINATION BASELINE
    # ===================================================================

    print("=" * 70)
    print("PHASE 3: FRAGMENT RECOMBINATION BASELINE")
    print("=" * 70)

    from benchmarks.baselines import FragmentBaseline

    fragment = FragmentBaseline()

    print("  Extracting BRICS fragments...")
    t0 = time.perf_counter()
    fragment.fit(train_smiles)
    fit_time = time.perf_counter() - t0
    print(f"  Fitted in {fit_time:.1f}s")

    print(f"  Generating {args.n_gen} molecules...")
    t0 = time.perf_counter()
    fragment_smiles = fragment.generate(n=args.n_gen)
    gen_time = time.perf_counter() - t0
    print(f"  Generated in {gen_time:.1f}s")

    m_fragment = compute_moses_metrics(fragment_smiles, train_smiles)
    all_results["Fragment Recombination"] = {
        **asdict(m_fragment),
        "fit_time_s": fit_time,
        "gen_time_s": gen_time,
    }
    print(f"  V={m_fragment.validity:.1%} U={m_fragment.uniqueness:.1%} "
          f"N={m_fragment.novelty:.1%} IntDiv={m_fragment.int_div_1:.3f}")
    print()

    # ===================================================================
    # PHASE 4: SMILES AUGMENTATION BASELINE
    # ===================================================================

    print("=" * 70)
    print("PHASE 4: SMILES AUGMENTATION BASELINE")
    print("=" * 70)

    from benchmarks.baselines import AugmentationBaseline

    augment = AugmentationBaseline()

    print("  Fitting augmentation baseline...")
    t0 = time.perf_counter()
    augment.fit(train_smiles)
    fit_time = time.perf_counter() - t0
    print(f"  Fitted in {fit_time:.1f}s")

    print(f"  Generating {args.n_gen} molecules...")
    t0 = time.perf_counter()
    augment_smiles = augment.generate(n=args.n_gen)
    gen_time = time.perf_counter() - t0
    print(f"  Generated in {gen_time:.1f}s")

    m_augment = compute_moses_metrics(augment_smiles, train_smiles)
    all_results["SMILES Augmentation"] = {
        **asdict(m_augment),
        "fit_time_s": fit_time,
        "gen_time_s": gen_time,
    }
    print(f"  V={m_augment.validity:.1%} U={m_augment.uniqueness:.1%} "
          f"N={m_augment.novelty:.1%} IntDiv={m_augment.int_div_1:.3f}")
    print()

    # ===================================================================
    # PHASE 5: TRAIN SET RESAMPLE (trivial baseline)
    # ===================================================================

    print("=" * 70)
    print("PHASE 5: TRAIN SET RESAMPLE (trivial baseline)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    resample_smiles = [rng.choice(train_smiles) for _ in range(args.n_gen)]
    m_resample = compute_moses_metrics(resample_smiles, train_smiles)
    all_results["Train Resample"] = asdict(m_resample)
    print(f"  V={m_resample.validity:.1%} U={m_resample.uniqueness:.1%} "
          f"N={m_resample.novelty:.1%} IntDiv={m_resample.int_div_1:.3f}")
    print()

    # ===================================================================
    # SUMMARY TABLE
    # ===================================================================

    print("=" * 70)
    print("SUMMARY: Baseline MOSES Metrics")
    print("=" * 70)
    print()
    print(f"{'Method':<25s} {'Valid':>7s} {'Unique':>7s} {'Novel':>7s} "
          f"{'IntDiv':>7s} {'FCD':>7s} {'ScafDiv':>8s}")
    print("-" * 75)
    for name in ["CharRNN", "Fragment Recombination", "SMILES Augmentation", "Train Resample"]:
        r = all_results[name]
        fcd = f"{r['fcd_proxy']:.1f}" if r.get("fcd_proxy", 0) > 0 else "N/A"
        print(f"{name:<25s} {r['validity']:>6.1%} {r['uniqueness']:>6.1%} "
              f"{r['novelty']:>6.1%} {r['int_div_1']:>6.3f} {fcd:>7s} "
              f"{r['scaffold_diversity']:>7.3f}")

    # ===================================================================
    # SAVE RESULTS
    # ===================================================================

    output_path = OUT / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
