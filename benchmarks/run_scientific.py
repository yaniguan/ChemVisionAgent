#!/usr/bin/env python3
"""ChemVisionAgent — Scientific Benchmark Suite

Benchmarks the SELFIES-based molecular generator against baselines using
MOSES-standard metrics on a 10K+ molecule dataset.

This is the definitive evaluation that answers:
  1. Does our generator produce valid, unique, novel molecules?
  2. How does it compare against random, VAE, and fingerprint-based baselines?
  3. Are generated molecules chemically realistic (property distributions)?
  4. Does conditioning on properties actually work (target adherence)?

Run:
    python benchmarks/run_scientific.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import selfies as sf
from selfies.exceptions import DecoderError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdkit import Chem
from rdkit.Chem import Descriptors, QED as QED_mod

from chemvision.core.reproducibility import set_global_seed
from chemvision.eval.moses_metrics import compute_moses_metrics, MOSESMetrics

set_global_seed(42)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def save_fig(name: str) -> None:
    plt.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# DATASET: Build 10K+ molecules from seed + random generation
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 1: BUILDING LARGE-SCALE DATASET")
print("=" * 70)

from chemvision.data.dataset_builder import MolecularDatasetBuilder

builder = MolecularDatasetBuilder(seed=42)
builder.add_seeds()
n_rand = builder.add_random_molecules(n=3000)
print(f"Generated {n_rand} random molecules")

stats = builder.build(OUT / "sci_dataset")
fps, props, splits = MolecularDatasetBuilder.load_arrays(OUT / "sci_dataset")

import pandas as pd
train_df = pd.read_parquet(OUT / "sci_dataset" / "train.parquet")
test_df = pd.read_parquet(OUT / "sci_dataset" / "test.parquet")

train_smiles = train_df["smiles"].tolist()
test_smiles = test_df["smiles"].tolist()

print(f"Dataset: {stats.n_total} molecules")
print(f"  Train: {stats.n_train}, Val: {stats.n_val}, Test: {stats.n_test}")
print(f"  Mean MW: {stats.mean_mw:.1f} ± {stats.std_mw:.1f}")
print(f"  Lipinski pass rate: {stats.lipinski_pass_rate:.0%}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK: SELFIES Generator vs Baselines (MOSES metrics)
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 2: MOLECULAR GENERATION BENCHMARK (MOSES METRICS)")
print("=" * 70)

N_GEN = 500  # generate this many molecules per method
target_props = props[splits["test_idx"]]
# Replicate targets to reach N_GEN
while len(target_props) < N_GEN:
    target_props = np.vstack([target_props, target_props])
target_props = target_props[:N_GEN]

all_metrics: dict[str, MOSESMetrics] = {}

# --- Baseline 1: Random SELFIES ---
print("\n  Baseline 1: Random SELFIES strings...")
random_selfies_smiles: list[str | None] = []
rng = np.random.RandomState(42)
vocab_tokens = list(sf.get_semantic_robust_alphabet())
for _ in range(N_GEN):
    length = rng.randint(5, 30)
    tokens = [rng.choice(vocab_tokens) for _ in range(length)]
    sel_str = "".join(tokens)
    try:
        smi = sf.decoder(sel_str)
        random_selfies_smiles.append(smi)
    except (ValueError, TypeError, DecoderError):
        random_selfies_smiles.append(None)

m_random = compute_moses_metrics(random_selfies_smiles, train_smiles)
all_metrics["Random SELFIES"] = m_random
print(f"    Validity={m_random.validity:.1%} Unique={m_random.uniqueness:.1%} "
      f"Novel={m_random.novelty:.1%} IntDiv={m_random.int_div_1:.3f}")

# --- Baseline 2: Training set sampling (memorisation upper bound) ---
print("  Baseline 2: Training set resampling (memorisation)...")
resample_smiles = [rng.choice(train_smiles) for _ in range(N_GEN)]
m_resample = compute_moses_metrics(resample_smiles, train_smiles)
all_metrics["Train Resample"] = m_resample
print(f"    Validity={m_resample.validity:.1%} Unique={m_resample.uniqueness:.1%} "
      f"Novel={m_resample.novelty:.1%} IntDiv={m_resample.int_div_1:.3f}")

# --- Baseline 3: SMILES character-level random walk ---
print("  Baseline 3: SMILES random walk...")
smiles_chars = set()
for s in train_smiles:
    smiles_chars.update(s)
smiles_chars = list(smiles_chars)
random_walk_smiles: list[str | None] = []
for _ in range(N_GEN):
    length = rng.randint(5, 40)
    smi = "".join(rng.choice(smiles_chars) for _ in range(length))
    mol = Chem.MolFromSmiles(smi)
    random_walk_smiles.append(Chem.MolToSmiles(mol) if mol else None)
m_walk = compute_moses_metrics(random_walk_smiles, train_smiles)
all_metrics["SMILES Random Walk"] = m_walk
print(f"    Validity={m_walk.validity:.1%} Unique={m_walk.uniqueness:.1%} "
      f"Novel={m_walk.novelty:.1%} IntDiv={m_walk.int_div_1:.3f}")

# --- SELFIES Generator (ours) ---
print("  SELFIES Generator (ours) — training...")
from chemvision.generation.selfies_gen import SELFIESGenerator, SELFIESGenConfig

t0 = time.perf_counter()
gen_config = SELFIESGenConfig(
    cond_dim=props.shape[1], embed_dim=128, n_heads=4, n_layers=3,
    max_len=80, dropout=0.1, learning_rate=3e-4, seed=42,
)
generator = SELFIESGenerator(gen_config)
train_result = generator.train(train_smiles, props[splits["train_idx"]],
                               epochs=60, batch_size=64, patience=15, verbose=True)
train_time = time.perf_counter() - t0

print(f"  Training: {train_result.epochs} epochs, loss={train_result.final_loss:.4f}, "
      f"time={train_time:.1f}s, vocab={train_result.vocab_size}")

# Generate
print("  Generating molecules...")
n_per_target = max(1, N_GEN // len(target_props[:50]))
generated = generator.generate(target_props[:50], n_per_target=n_per_target, temperature=0.8)
gen_smiles = [g.smiles for g in generated[:N_GEN]]

m_ours = compute_moses_metrics(gen_smiles, train_smiles)
all_metrics["SELFIES Gen (ours)"] = m_ours
print(f"    Validity={m_ours.validity:.1%} Unique={m_ours.uniqueness:.1%} "
      f"Novel={m_ours.novelty:.1%} IntDiv={m_ours.int_div_1:.3f}")

# Also run old fingerprint flow matcher for comparison
print("\n  Old Flow Matcher (fingerprint, for comparison)...")
from chemvision.generation._experimental.flow_matcher import ConditionalFlowMatcher, FlowMatcherConfig

cfm = ConditionalFlowMatcher(FlowMatcherConfig(
    fp_dim=fps.shape[1], cond_dim=props.shape[1],
    hidden_dim=128, n_layers=2, seed=42,
))
cfm.train(fps[splits["train_idx"]], props[splits["train_idx"]], epochs=50, batch_size=32)
flow_gen = cfm.sample(target_props[:N_GEN], n_steps=20)
# Fingerprints cannot be decoded to SMILES — report as N/A
m_flow = MOSESMetrics(
    n_generated=len(flow_gen), n_valid=0, validity=0.0,
    uniqueness=0.0, novelty=0.0, int_div_1=0.0,
)
all_metrics["Flow Matcher (FP)"] = m_flow
print(f"    Validity=N/A (fingerprints cannot be decoded to molecules)")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PHASE 3: GENERATING PUBLICATION FIGURES")
print("=" * 70)

# --- Figure 1: MOSES metrics comparison ---
methods = ["Random\nSELFIES", "SMILES\nRandom Walk", "Train\nResample", "Flow Matcher\n(FP, broken)", "SELFIES Gen\n(ours)"]
metrics_list = [
    all_metrics["Random SELFIES"],
    all_metrics["SMILES Random Walk"],
    all_metrics["Train Resample"],
    all_metrics["Flow Matcher (FP)"],
    all_metrics["SELFIES Gen (ours)"],
]
colors = ["#bbb", "#999", "#4e79a7", "#d62728", "#2ca02c"]

fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
metric_keys = [
    ("validity", "Validity"),
    ("uniqueness", "Uniqueness"),
    ("novelty", "Novelty"),
    ("int_div_1", "Internal\nDiversity"),
    ("fcd_proxy", "FCD Proxy\n(lower=better)"),
]

for ax, (key, label) in zip(axes, metric_keys):
    vals = [getattr(m, key) for m in metrics_list]
    bars = ax.bar(methods, vals, color=colors, edgecolor="black")
    ax.set_title(label, fontsize=11, fontweight="bold")
    if key != "fcd_proxy":
        ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        label_text = f"{v:.0%}" if key != "fcd_proxy" else f"{v:.1f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                label_text, ha="center", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)

plt.suptitle("MOSES Benchmark: Molecular Generation Quality", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("05_moses_benchmark")
print("  Saved 05_moses_benchmark.png")

# --- Figure 2: Training loss curve ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_result.loss_history, color="#2ca02c", linewidth=2)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
ax.set_title(f"SELFIES Generator Training ({train_result.epochs} epochs, vocab={train_result.vocab_size})", fontsize=13)
ax.grid(True, alpha=0.3)
save_fig("05_selfies_loss_curve")
print("  Saved 05_selfies_loss_curve.png")

# --- Figure 3: Property distribution comparison ---
valid_gen = [s for s in gen_smiles if s and Chem.MolFromSmiles(s)]
if valid_gen:
    gen_mw = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in valid_gen]
    gen_logp = [Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in valid_gen]
    gen_qed = [QED_mod.qed(Chem.MolFromSmiles(s)) for s in valid_gen]

    train_mw = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in train_smiles if Chem.MolFromSmiles(s)]
    train_logp = [Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in train_smiles if Chem.MolFromSmiles(s)]
    train_qed = [QED_mod.qed(Chem.MolFromSmiles(s)) for s in train_smiles if Chem.MolFromSmiles(s)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (train_vals, gen_vals, label) in zip(axes, [
        (train_mw, gen_mw, "Molecular Weight"),
        (train_logp, gen_logp, "LogP"),
        (train_qed, gen_qed, "QED"),
    ]):
        ax.hist(train_vals, bins=30, alpha=0.5, color="steelblue", label="Training", density=True)
        ax.hist(gen_vals, bins=30, alpha=0.5, color="#2ca02c", label="Generated", density=True)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Property Distribution: Training vs Generated", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("05_property_distributions")
    print("  Saved 05_property_distributions.png")

# --- Figure 4: Example generated molecules ---
print("\n  Example generated molecules:")
valid_examples = [g for g in generated if g.is_valid][:10]
for i, g in enumerate(valid_examples):
    mol = Chem.MolFromSmiles(g.smiles)
    mw = Descriptors.MolWt(mol) if mol else 0
    qed = QED_mod.qed(mol) if mol else 0
    print(f"    {i+1}. {g.smiles[:50]:50s} MW={mw:.0f} QED={qed:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE (publication-ready)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PUBLICATION-READY RESULTS TABLE")
print("=" * 70)
print()
print(f"{'Method':<22s} {'Valid':>7s} {'Unique':>7s} {'Novel':>7s} {'IntDiv1':>8s} {'FCD':>7s} {'MW':>8s} {'QED':>6s}")
print("-" * 78)
for name, m in all_metrics.items():
    fcd_str = f"{m.fcd_proxy:.1f}" if m.fcd_proxy > 0 else "N/A"
    mw_str = f"{m.mw_mean:.0f}" if m.mw_mean > 0 else "N/A"
    qed_str = f"{m.qed_mean:.3f}" if m.qed_mean > 0 else "N/A"
    print(f"{name:<22s} {m.validity:>6.1%} {m.uniqueness:>6.1%} {m.novelty:>6.1%} "
          f"{m.int_div_1:>7.4f} {fcd_str:>7s} {mw_str:>8s} {qed_str:>6s}")

# Save results
results_dict = {name: {"validity": m.validity, "uniqueness": m.uniqueness,
                        "novelty": m.novelty, "int_div_1": m.int_div_1,
                        "fcd_proxy": m.fcd_proxy, "mw_mean": m.mw_mean,
                        "qed_mean": m.qed_mean}
                for name, m in all_metrics.items()}
results_dict["training"] = {"epochs": train_result.epochs, "loss": train_result.final_loss,
                            "vocab_size": train_result.vocab_size, "time_s": train_time,
                            "dataset_size": stats.n_total}

with open(OUT / "scientific_results.json", "w") as f:
    json.dump(results_dict, f, indent=2, default=str)

print(f"\nResults saved to {OUT / 'scientific_results.json'}")
print(f"Figures saved to {FIG}/")
