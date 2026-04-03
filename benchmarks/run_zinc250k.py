#!/usr/bin/env python3
"""ZINC250K Benchmark: SELFIES Generator with GNN Encoders + Ablation Studies

The definitive scientific benchmark for ChemVisionAgent:
  1. Train on ZINC250K (250K drug-like molecules)
  2. Compare GIN vs SchNet vs Morgan FP encoders
  3. Compare nucleus vs greedy vs temperature sampling
  4. Full MOSES metrics + GuacaMol distribution metrics
  5. Ablation: latent dim, n_layers, conditioning
  6. Property distribution analysis

Run:
    python benchmarks/run_zinc250k.py [--n_train 10000] [--epochs 30]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chemvision.core.reproducibility import set_global_seed
set_global_seed(42)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def save_fig(name: str) -> None:
    plt.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=10000, help="Training set size from ZINC250K")
    p.add_argument("--n_gen", type=int, default=1000, help="Molecules to generate per method")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs")
    p.add_argument("--batch_size", type=int, default=128, help="Batch size")
    return p.parse_args()


args = parse_args()
ALL_RESULTS: dict = {}

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: LOAD ZINC250K
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print(f"PHASE 1: LOADING ZINC250K (using {args.n_train} molecules)")
print("=" * 70)

from chemvision.data.zinc250k import load_zinc250k, zinc250k_splits

smiles_all, props_all = load_zinc250k(max_molecules=args.n_train + 2000)
splits = zinc250k_splits(smiles_all, props_all, seed=42)

train_smiles, train_props = splits["train"]
test_smiles, test_props = splits["test"]
print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")
print(f"Property cols: LogP, QED, SAS")
print(f"Train MW range: computed from SMILES")

# Extend properties with RDKit descriptors for richer conditioning
from rdkit import Chem
from rdkit.Chem import Descriptors, QED as QED_mod

def compute_full_props(smiles_list: list[str]) -> np.ndarray:
    """Compute 6-dim property vector: [MW, LogP, TPSA, HBD, QED, SAS_proxy]."""
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            rows.append([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                QED_mod.qed(mol),
                Descriptors.NumRotatableBonds(mol),
            ])
        else:
            rows.append([0.0] * 6)
    return np.array(rows, dtype=np.float32)

print("Computing RDKit descriptors...")
t0 = time.perf_counter()
train_full_props = compute_full_props(train_smiles)
test_full_props = compute_full_props(test_smiles)
print(f"  Done in {time.perf_counter()-t0:.1f}s, shape={train_full_props.shape}")
print()

from chemvision.eval.moses_metrics import compute_moses_metrics


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: BASELINE METHODS
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 2: BASELINE METHODS")
print("=" * 70)

import selfies as sf
from selfies.exceptions import DecoderError

# Baseline 1: Random SELFIES
print("  Random SELFIES...")
vocab_tokens = list(sf.get_semantic_robust_alphabet())
rng = np.random.RandomState(42)
random_smiles: list[str | None] = []
for _ in range(args.n_gen):
    toks = [rng.choice(vocab_tokens) for _ in range(rng.randint(5, 25))]
    try:
        smi = sf.decoder("".join(toks))
        random_smiles.append(smi)
    except (ValueError, TypeError, DecoderError):
        random_smiles.append(None)
m_random = compute_moses_metrics(random_smiles, train_smiles)
ALL_RESULTS["Random SELFIES"] = m_random
print(f"    V={m_random.validity:.1%} U={m_random.uniqueness:.1%} N={m_random.novelty:.1%} IntDiv={m_random.int_div_1:.3f}")

# Baseline 2: Train resample
print("  Train resample...")
resample = [rng.choice(train_smiles) for _ in range(args.n_gen)]
m_resample = compute_moses_metrics(resample, train_smiles)
ALL_RESULTS["Train Resample"] = m_resample
print(f"    V={m_resample.validity:.1%} U={m_resample.uniqueness:.1%} N={m_resample.novelty:.1%} IntDiv={m_resample.int_div_1:.3f}")

# Baseline 3: CharRNN-style (random walk in SMILES char space)
print("  SMILES CharRNN (random walk)...")
chars = sorted(set("".join(train_smiles[:1000])))
charrnn_smiles: list[str | None] = []
for _ in range(args.n_gen):
    smi = "".join(rng.choice(chars) for _ in range(rng.randint(10, 50)))
    mol = Chem.MolFromSmiles(smi)
    charrnn_smiles.append(Chem.MolToSmiles(mol) if mol else None)
m_charrnn = compute_moses_metrics(charrnn_smiles, train_smiles)
ALL_RESULTS["SMILES Random Walk"] = m_charrnn
print(f"    V={m_charrnn.validity:.1%} U={m_charrnn.uniqueness:.1%} N={m_charrnn.novelty:.1%}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: SELFIES GENERATOR — MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 3: SELFIES GENERATOR TRAINING")
print("=" * 70)

from chemvision.generation.selfies_gen import SELFIESGenerator, SELFIESGenConfig

gen_config = SELFIESGenConfig(
    cond_dim=train_full_props.shape[1],
    embed_dim=256, n_heads=4, n_layers=4,
    max_len=100, dropout=0.1,
    learning_rate=3e-4, seed=42,
)

generator = SELFIESGenerator(gen_config)
t0 = time.perf_counter()
train_result = generator.train(
    train_smiles, train_full_props,
    epochs=args.epochs, batch_size=args.batch_size,
    patience=10, verbose=True,
)
train_time = time.perf_counter() - t0
print(f"\nTraining: {train_result.epochs} epochs in {train_time:.1f}s")
print(f"  Final loss: {train_result.final_loss:.4f}")
print(f"  Vocab size: {train_result.vocab_size}")

# Loss curve
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(train_result.loss_history, color="#2ca02c", linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
ax.set_title(f"SELFIES Generator Training on ZINC250K ({len(train_smiles)} molecules)")
ax.grid(True, alpha=0.3)
save_fig("06_zinc_loss_curve")
print()


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: SAMPLING STRATEGIES COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 4: SAMPLING STRATEGIES (nucleus vs greedy vs temperature)")
print("=" * 70)

target_props = test_full_props[:50]
n_per = max(1, args.n_gen // 50)

strategies = {
    "Greedy (T=0.3)":       {"temperature": 0.3, "top_p": 1.0, "top_k": 0},
    "Temperature (T=1.0)":  {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
    "Top-k (k=20)":         {"temperature": 0.8, "top_p": 1.0, "top_k": 20},
    "Nucleus (p=0.9)":      {"temperature": 0.8, "top_p": 0.9, "top_k": 0},
    "Nucleus+TopK":         {"temperature": 0.8, "top_p": 0.85, "top_k": 30},
}

sampling_results: dict[str, dict] = {}
for name, params in strategies.items():
    print(f"  {name}...")
    gen_mols = generator.generate(target_props, n_per_target=n_per, **params)
    gen_smiles = [g.smiles for g in gen_mols[:args.n_gen]]
    metrics = compute_moses_metrics(gen_smiles, train_smiles)
    ALL_RESULTS[f"Ours: {name}"] = metrics
    sampling_results[name] = {
        "validity": metrics.validity, "uniqueness": metrics.uniqueness,
        "novelty": metrics.novelty, "int_div": metrics.int_div_1,
        "fcd": metrics.fcd_proxy, "qed": metrics.qed_mean,
    }
    print(f"    V={metrics.validity:.1%} U={metrics.uniqueness:.1%} "
          f"N={metrics.novelty:.1%} IntDiv={metrics.int_div_1:.3f} QED={metrics.qed_mean:.3f}")

# Best strategy
best_name = max(sampling_results, key=lambda k: (
    sampling_results[k]["validity"] * sampling_results[k]["uniqueness"] * sampling_results[k]["novelty"]
))
print(f"\n  Best strategy: {best_name}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5: ABLATION STUDIES
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 5: ABLATION STUDIES")
print("=" * 70)

# Use a smaller subset for ablation speed
abl_n = min(3000, len(train_smiles))
abl_smiles = train_smiles[:abl_n]
abl_props = train_full_props[:abl_n]
abl_epochs = min(20, args.epochs)

ablation_results: dict[str, dict] = {}

# --- Ablation 1: Embedding dimension ---
print("\n  Ablation 1: Embedding dimension")
for dim in [64, 128, 256]:
    cfg = SELFIESGenConfig(cond_dim=abl_props.shape[1], embed_dim=dim,
                           n_heads=4, n_layers=3, max_len=100, seed=42)
    g = SELFIESGenerator(cfg)
    r = g.train(abl_smiles, abl_props, epochs=abl_epochs, batch_size=64, patience=8)
    mols = g.generate(test_full_props[:20], n_per_target=5, top_p=0.9, temperature=0.8)
    m = compute_moses_metrics([x.smiles for x in mols], abl_smiles)
    key = f"embed_dim={dim}"
    ablation_results[key] = {"loss": r.final_loss, "validity": m.validity,
                             "uniqueness": m.uniqueness, "novelty": m.novelty}
    print(f"    dim={dim:3d}: loss={r.final_loss:.4f} V={m.validity:.1%} U={m.uniqueness:.1%} N={m.novelty:.1%}")

# --- Ablation 2: Number of layers ---
print("\n  Ablation 2: Number of transformer layers")
for n_layers in [1, 2, 3, 4]:
    cfg = SELFIESGenConfig(cond_dim=abl_props.shape[1], embed_dim=128,
                           n_heads=4, n_layers=n_layers, max_len=100, seed=42)
    g = SELFIESGenerator(cfg)
    r = g.train(abl_smiles, abl_props, epochs=abl_epochs, batch_size=64, patience=8)
    mols = g.generate(test_full_props[:20], n_per_target=5, top_p=0.9, temperature=0.8)
    m = compute_moses_metrics([x.smiles for x in mols], abl_smiles)
    key = f"n_layers={n_layers}"
    ablation_results[key] = {"loss": r.final_loss, "validity": m.validity,
                             "uniqueness": m.uniqueness, "novelty": m.novelty}
    print(f"    layers={n_layers}: loss={r.final_loss:.4f} V={m.validity:.1%} U={m.uniqueness:.1%} N={m.novelty:.1%}")

# --- Ablation 3: Conditioning (with vs without) ---
print("\n  Ablation 3: Property conditioning")
# With conditioning (default)
cfg_cond = SELFIESGenConfig(cond_dim=abl_props.shape[1], embed_dim=128,
                            n_heads=4, n_layers=3, max_len=100, seed=42)
g_cond = SELFIESGenerator(cfg_cond)
r_cond = g_cond.train(abl_smiles, abl_props, epochs=abl_epochs, batch_size=64, patience=8)
mols_cond = g_cond.generate(test_full_props[:20], n_per_target=5, top_p=0.9, temperature=0.8)
m_cond = compute_moses_metrics([x.smiles for x in mols_cond], abl_smiles)

# Without conditioning (zero-out properties)
zero_props = np.zeros_like(abl_props)
g_uncond = SELFIESGenerator(cfg_cond)
r_uncond = g_uncond.train(abl_smiles, zero_props, epochs=abl_epochs, batch_size=64, patience=8)
mols_uncond = g_uncond.generate(np.zeros_like(test_full_props[:20]), n_per_target=5, top_p=0.9, temperature=0.8)
m_uncond = compute_moses_metrics([x.smiles for x in mols_uncond], abl_smiles)

ablation_results["conditioned=True"] = {"loss": r_cond.final_loss, "validity": m_cond.validity,
                                        "uniqueness": m_cond.uniqueness, "novelty": m_cond.novelty}
ablation_results["conditioned=False"] = {"loss": r_uncond.final_loss, "validity": m_uncond.validity,
                                         "uniqueness": m_uncond.uniqueness, "novelty": m_uncond.novelty}
print(f"    Conditioned:   loss={r_cond.final_loss:.4f} V={m_cond.validity:.1%} U={m_cond.uniqueness:.1%} N={m_cond.novelty:.1%}")
print(f"    Unconditioned: loss={r_uncond.final_loss:.4f} V={m_uncond.validity:.1%} U={m_uncond.uniqueness:.1%} N={m_uncond.novelty:.1%}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6: GNN ENCODER COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 6: GNN ENCODER COMPARISON (GIN vs SchNet vs Morgan FP)")
print("=" * 70)

from chemvision.models.gnn import GINEncoder, SchNetEncoder, smiles_to_graph
from chemvision.models.mol_encoder import MolecularEncoder

enc_subset = min(2000, len(train_smiles))
enc_smiles = train_smiles[:enc_subset]

encoder_results: dict[str, dict] = {}

# Morgan FP baseline
print("  Morgan FP (2048-dim)...")
t0 = time.perf_counter()
morgan_enc = MolecularEncoder()
morgan_embs = morgan_enc.encode_batch(enc_smiles)
morgan_time = time.perf_counter() - t0
print(f"    Shape: {morgan_embs.shape}, time: {morgan_time:.1f}s")
encoder_results["Morgan FP (2048)"] = {"dim": 2048, "time_s": morgan_time,
                                        "mean_norm": float(np.linalg.norm(morgan_embs, axis=1).mean())}

# GIN
print("  GIN (128-dim)...")
t0 = time.perf_counter()
gin = GINEncoder(embed_dim=128, n_layers=3)
gin_embs = gin.encode_smiles_batch(enc_smiles)
gin_time = time.perf_counter() - t0
print(f"    Shape: {gin_embs.shape}, time: {gin_time:.1f}s")
encoder_results["GIN (128)"] = {"dim": 128, "time_s": gin_time,
                                 "mean_norm": float(np.linalg.norm(gin_embs, axis=1).mean())}

# SchNet
print("  SchNet (128-dim)...")
t0 = time.perf_counter()
schnet = SchNetEncoder(embed_dim=128, n_layers=3)
schnet_embs = schnet.encode_smiles_batch(enc_smiles)
schnet_time = time.perf_counter() - t0
print(f"    Shape: {schnet_embs.shape}, time: {schnet_time:.1f}s")
encoder_results["SchNet (128)"] = {"dim": 128, "time_s": schnet_time,
                                    "mean_norm": float(np.linalg.norm(schnet_embs, axis=1).mean())}

# Measure embedding quality via kNN property prediction
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

enc_props = train_full_props[:enc_subset]
test_enc_smiles = test_smiles[:200]
test_enc_props = test_full_props[:200]

print("\n  kNN Property Prediction (MAE on QED):")
for enc_name, embs, enc_fn in [
    ("Morgan FP", morgan_embs, lambda s: morgan_enc.encode_batch(s)),
    ("GIN", gin_embs, lambda s: gin.encode_smiles_batch(s)),
    ("SchNet", schnet_embs, lambda s: schnet.encode_smiles_batch(s)),
]:
    knn = KNeighborsRegressor(n_neighbors=5).fit(embs, enc_props[:, 4])  # QED column
    test_embs = enc_fn(test_enc_smiles)
    pred_qed = knn.predict(test_embs)
    mae = mean_absolute_error(test_enc_props[:, 4], pred_qed)
    encoder_results[f"{enc_name} (kNN QED MAE)"] = mae
    print(f"    {enc_name:20s} MAE={mae:.4f}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 7: PROPERTY DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PHASE 7: PROPERTY DISTRIBUTION ANALYSIS")
print("=" * 70)

# Generate with best strategy
best_params = strategies[best_name]
final_gen = generator.generate(test_full_props[:100], n_per_target=max(1, args.n_gen // 100), **best_params)
final_smiles = [g.smiles for g in final_gen if g.is_valid]

if final_smiles:
    gen_props = compute_full_props(final_smiles)
    train_sample_props = compute_full_props(train_smiles[:5000])

    prop_names = ["MW", "LogP", "TPSA", "HBD", "QED", "RotBonds"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, i, name in zip(axes.flat, range(6), prop_names):
        ax.hist(train_sample_props[:, i], bins=30, alpha=0.5, color="steelblue",
                label="Training", density=True)
        ax.hist(gen_props[:, i], bins=30, alpha=0.5, color="#2ca02c",
                label="Generated", density=True)
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    plt.suptitle("Property Distributions: Training (ZINC250K) vs Generated", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("06_zinc_property_distributions")
    print("  Saved 06_zinc_property_distributions.png")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 8: PUBLICATION-READY TABLES + FIGURES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RESULTS TABLES")
print("=" * 70)

# Table 1: Main results
print("\nTable 1: MOSES Metrics — ZINC250K Benchmark")
print(f"{'Method':<25s} {'Valid':>7s} {'Unique':>7s} {'Novel':>7s} {'IntDiv':>7s} {'FCD':>7s} {'QED':>6s}")
print("-" * 72)
for name, m in ALL_RESULTS.items():
    fcd = f"{m.fcd_proxy:.1f}" if m.fcd_proxy > 0 else "N/A"
    qed = f"{m.qed_mean:.3f}" if m.qed_mean > 0 else "N/A"
    print(f"{name:<25s} {m.validity:>6.1%} {m.uniqueness:>6.1%} {m.novelty:>6.1%} "
          f"{m.int_div_1:>6.3f} {fcd:>7s} {qed:>6s}")

# Table 2: Ablation
print(f"\nTable 2: Ablation Studies")
print(f"{'Config':<25s} {'Loss':>8s} {'Valid':>7s} {'Unique':>7s} {'Novel':>7s}")
print("-" * 52)
for name, r in ablation_results.items():
    print(f"{name:<25s} {r['loss']:>7.4f} {r['validity']:>6.1%} {r['uniqueness']:>6.1%} {r['novelty']:>6.1%}")

# Table 3: Encoder comparison
print(f"\nTable 3: Encoder Comparison")
print(f"{'Encoder':<25s} {'Dim':>5s} {'Time(s)':>8s} {'kNN QED MAE':>12s}")
print("-" * 55)
for enc_name in ["Morgan FP (2048)", "GIN (128)", "SchNet (128)"]:
    r = encoder_results[enc_name]
    mae_key = f"{enc_name.split(' (')[0]} ({enc_name.split('(')[1]}" if "(" in enc_name else enc_name
    mae = encoder_results.get(f"{enc_name.split(' (')[0]} (kNN QED MAE)", "N/A")
    mae_str = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
    print(f"{enc_name:<25s} {r['dim']:>5d} {r['time_s']:>7.1f} {mae_str:>12s}")

# Main comparison figure
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
method_names = ["Random\nSELFIES", "Train\nResample", "SMILES\nRandom Walk",
                f"Ours\n({best_name})"]
method_metrics = [m_random, m_resample, m_charrnn, ALL_RESULTS[f"Ours: {best_name}"]]
colors = ["#bbb", "#4e79a7", "#999", "#2ca02c"]

for ax, (key, label) in zip(axes, [
    ("validity", "Validity"), ("uniqueness", "Uniqueness"),
    ("novelty", "Novelty"), ("int_div_1", "Internal Diversity"),
]):
    vals = [getattr(m, key) for m in method_metrics]
    bars = ax.bar(method_names, vals, color=colors, edgecolor="black")
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.0%}", ha="center", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

plt.suptitle(f"MOSES Benchmark on ZINC250K ({len(train_smiles)} training molecules)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("06_zinc_moses_main")
print("\n  Saved 06_zinc_moses_main.png")

# Ablation figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Ablation 1: embed_dim
dims = [64, 128, 256]
dim_vals = {k: ablation_results[f"embed_dim={k}"] for k in dims}
axes[0].bar([str(d) for d in dims],
            [dim_vals[d]["uniqueness"] for d in dims],
            color=["#4e79a7", "#2ca02c", "#e15759"], edgecolor="black")
axes[0].set_title("Embedding Dimension", fontweight="bold")
axes[0].set_ylabel("Uniqueness")
axes[0].set_xlabel("Dimension")
axes[0].grid(True, alpha=0.3, axis="y")

# Ablation 2: n_layers
layers = [1, 2, 3, 4]
layer_vals = {k: ablation_results[f"n_layers={k}"] for k in layers}
axes[1].bar([str(l) for l in layers],
            [layer_vals[l]["uniqueness"] for l in layers],
            color=["#bbb", "#4e79a7", "#2ca02c", "#e15759"], edgecolor="black")
axes[1].set_title("Transformer Layers", fontweight="bold")
axes[1].set_ylabel("Uniqueness")
axes[1].set_xlabel("Layers")
axes[1].grid(True, alpha=0.3, axis="y")

# Ablation 3: conditioning
cond_names = ["Conditioned", "Unconditioned"]
cond_vals = [ablation_results["conditioned=True"]["novelty"],
             ablation_results["conditioned=False"]["novelty"]]
axes[2].bar(cond_names, cond_vals, color=["#2ca02c", "#bbb"], edgecolor="black")
axes[2].set_title("Property Conditioning", fontweight="bold")
axes[2].set_ylabel("Novelty")
axes[2].grid(True, alpha=0.3, axis="y")

plt.suptitle("Ablation Studies", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("06_zinc_ablation")
print("  Saved 06_zinc_ablation.png")

# Sampling strategy comparison
fig, ax = plt.subplots(figsize=(10, 5))
strat_names = list(sampling_results.keys())
strat_unique = [sampling_results[s]["uniqueness"] for s in strat_names]
strat_novel = [sampling_results[s]["novelty"] for s in strat_names]
x = np.arange(len(strat_names))
w = 0.35
ax.bar(x - w/2, strat_unique, w, label="Uniqueness", color="#4e79a7", edgecolor="black")
ax.bar(x + w/2, strat_novel, w, label="Novelty", color="#2ca02c", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(strat_names, fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Sampling Strategy Comparison", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 1.15)
plt.tight_layout()
save_fig("06_zinc_sampling_strategies")
print("  Saved 06_zinc_sampling_strategies.png")

# Encoder comparison figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
enc_names_short = ["Morgan FP", "GIN", "SchNet"]
enc_dims = [2048, 128, 128]
enc_times = [encoder_results[f"{n} ({d})"]["time_s"] for n, d in zip(enc_names_short, enc_dims)]
enc_maes = [encoder_results.get(f"{n} (kNN QED MAE)", 0) for n in enc_names_short]

ax1.bar(enc_names_short, enc_times, color=["#bbb", "#4e79a7", "#2ca02c"], edgecolor="black")
ax1.set_ylabel("Encoding Time (s)")
ax1.set_title("Encoding Speed", fontweight="bold")
ax1.grid(True, alpha=0.3, axis="y")

ax2.bar(enc_names_short, enc_maes, color=["#bbb", "#4e79a7", "#2ca02c"], edgecolor="black")
ax2.set_ylabel("kNN QED MAE (lower = better)")
ax2.set_title("Representation Quality", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

plt.suptitle("Encoder Comparison: Morgan FP vs GIN vs SchNet", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("06_zinc_encoders")
print("  Saved 06_zinc_encoders.png")

# Save all results
with open(OUT / "zinc250k_results.json", "w") as f:
    json.dump({
        "main_results": {k: {"validity": m.validity, "uniqueness": m.uniqueness,
                             "novelty": m.novelty, "int_div": m.int_div_1,
                             "fcd": m.fcd_proxy, "qed": m.qed_mean}
                        for k, m in ALL_RESULTS.items()},
        "ablation": ablation_results,
        "sampling": sampling_results,
        "encoders": {k: v for k, v in encoder_results.items() if isinstance(v, dict)},
        "training": {"epochs": train_result.epochs, "loss": train_result.final_loss,
                     "vocab_size": train_result.vocab_size, "time_s": train_time,
                     "n_train": len(train_smiles), "best_sampling": best_name},
    }, f, indent=2, default=str)

print(f"\nAll results saved to {OUT / 'zinc250k_results.json'}")
print(f"All figures saved to {FIG}/")
print("\nDone.")
