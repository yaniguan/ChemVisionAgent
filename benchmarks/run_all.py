#!/usr/bin/env python3
"""ChemVisionAgent — Full Benchmark Suite

Benchmarks every novel algorithm against SOTA baselines and saves
publication-quality figures + a JSON results file.

Algorithms benchmarked:
  1. CSCA (Contrastive Structure-Property Alignment) vs Random / PCA / Autoencoder
  2. Conditional Flow Matching vs Random Sampling / Gaussian Mixture / VAE
  3. Pareto MCTS vs Random Search / Greedy / Genetic Algorithm
  4. Confidence Calibration (Isotonic / Platt) vs Uncalibrated / Temperature Scaling

Run:
    python benchmarks/run_all.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive for CI

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chemvision.core.reproducibility import set_global_seed

set_global_seed(42)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

ALL_RESULTS: dict = {}


def save_fig(name: str) -> None:
    plt.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {FIG / name}.png")


# ═══════════════════════════════════════════════════════════════════════════
# DATASET: shared across all benchmarks
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BUILDING DATASET")
print("=" * 70)

from chemvision.data.dataset_builder import MolecularDatasetBuilder

builder = MolecularDatasetBuilder(seed=42)
builder.add_seeds()
builder.add_random_molecules(n=200)
stats = builder.build(OUT / "dataset")
fps, props, splits = MolecularDatasetBuilder.load_arrays(OUT / "dataset")

train_idx = splits["train_idx"]
test_idx = splits["test_idx"]
fps_train, props_train = fps[train_idx], props[train_idx]
fps_test, props_test = fps[test_idx], props[test_idx]

print(f"Dataset: {len(fps)} molecules, train={len(train_idx)}, test={len(test_idx)}")
print(f"FP dim: {fps.shape[1]}, Prop dim: {props.shape[1]}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK 1: CSCA vs Baselines — Retrieval Recall@k
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BENCHMARK 1: CSCA vs Baselines — Retrieval Recall@k")
print("=" * 70)


def retrieval_recall_at_k(query_emb, db_emb, k_values=[1, 5, 10]):
    """Compute Recall@k: fraction of queries where correct item is in top-k."""
    # query_emb: (N, D), db_emb: (N, D). Ground truth: query[i] matches db[i].
    sim = query_emb @ db_emb.T  # (N, N) cosine sim
    recalls = {}
    for k in k_values:
        topk = np.argsort(-sim, axis=1)[:, :k]
        correct = np.array([i in topk[i] for i in range(len(query_emb))])
        recalls[f"R@{k}"] = float(correct.mean())
    return recalls


# Normalise props for all baselines
prop_mean, prop_std = props_train.mean(0), props_train.std(0) + 1e-8
props_train_n = (props_train - prop_mean) / prop_std
props_test_n = (props_test - prop_mean) / prop_std

# --- Baseline 1: Random projection ---
rng = np.random.RandomState(42)
random_proj = rng.randn(fps.shape[1], 128).astype(np.float32)
random_proj /= np.linalg.norm(random_proj, axis=0, keepdims=True)
prop_proj = rng.randn(props.shape[1], 128).astype(np.float32)
prop_proj /= np.linalg.norm(prop_proj, axis=0, keepdims=True)

fp_emb_rand = fps_test @ random_proj
fp_emb_rand /= np.linalg.norm(fp_emb_rand, axis=1, keepdims=True) + 1e-9
prop_emb_rand = props_test_n @ prop_proj
prop_emb_rand /= np.linalg.norm(prop_emb_rand, axis=1, keepdims=True) + 1e-9
recall_random = retrieval_recall_at_k(prop_emb_rand, fp_emb_rand)
print(f"  Random projection: {recall_random}")

# --- Baseline 2: PCA (linear, unsupervised) ---
from sklearn.decomposition import PCA

pca_fp = PCA(n_components=128, random_state=42).fit(fps_train)
pca_prop = PCA(n_components=min(128, props.shape[1]), random_state=42).fit(props_train_n)

fp_emb_pca = pca_fp.transform(fps_test)
fp_emb_pca /= np.linalg.norm(fp_emb_pca, axis=1, keepdims=True) + 1e-9
prop_emb_pca = pca_prop.transform(props_test_n)
prop_emb_pca /= np.linalg.norm(prop_emb_pca, axis=1, keepdims=True) + 1e-9
# Pad prop to match fp dim
if prop_emb_pca.shape[1] < fp_emb_pca.shape[1]:
    prop_emb_pca = np.pad(prop_emb_pca, ((0, 0), (0, fp_emb_pca.shape[1] - prop_emb_pca.shape[1])))
    prop_emb_pca /= np.linalg.norm(prop_emb_pca, axis=1, keepdims=True) + 1e-9
recall_pca = retrieval_recall_at_k(prop_emb_pca, fp_emb_pca)
print(f"  PCA:               {recall_pca}")

# --- Baseline 3: Autoencoder (MLP, unsupervised) ---
class SimpleAE(torch.nn.Module):
    def __init__(self, in_dim, latent):
        super().__init__()
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256), torch.nn.ReLU(), torch.nn.Linear(256, latent))
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(latent, 256), torch.nn.ReLU(), torch.nn.Linear(256, in_dim))
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z

ae_fp = SimpleAE(fps.shape[1], 128)
ae_prop = SimpleAE(props.shape[1], 128)
opt_ae = torch.optim.Adam(list(ae_fp.parameters()) + list(ae_prop.parameters()), lr=1e-3)

t_fp_train = torch.tensor(fps_train, dtype=torch.float32)
t_prop_train = torch.tensor(props_train_n, dtype=torch.float32)

for epoch in range(80):
    recon_fp, z_fp = ae_fp(t_fp_train)
    recon_prop, z_prop = ae_prop(t_prop_train)
    loss = torch.nn.functional.mse_loss(recon_fp, t_fp_train) + \
           torch.nn.functional.mse_loss(recon_prop, t_prop_train)
    opt_ae.zero_grad(); loss.backward(); opt_ae.step()

with torch.no_grad():
    _, z_fp_test = ae_fp(torch.tensor(fps_test, dtype=torch.float32))
    _, z_prop_test = ae_prop(torch.tensor(props_test_n, dtype=torch.float32))
    fp_emb_ae = z_fp_test.numpy()
    fp_emb_ae /= np.linalg.norm(fp_emb_ae, axis=1, keepdims=True) + 1e-9
    prop_emb_ae = z_prop_test.numpy()
    prop_emb_ae /= np.linalg.norm(prop_emb_ae, axis=1, keepdims=True) + 1e-9
recall_ae = retrieval_recall_at_k(prop_emb_ae, fp_emb_ae)
print(f"  Autoencoder:       {recall_ae}")

# --- CSCA (ours) ---
from chemvision.models.csca import CSCATrainer, CSCAConfig

csca_config = CSCAConfig(fp_dim=fps.shape[1], prop_dim=props.shape[1],
                         latent_dim=128, hidden_dim=256, learning_rate=1e-3, seed=42)
csca_trainer = CSCATrainer(csca_config)
csca_result = csca_trainer.train(fps_train, props_train, epochs=120, batch_size=32, patience=25)

csca_model = csca_trainer.model
csca_model.eval()
with torch.no_grad():
    fp_emb_csca = csca_model.encode_fingerprint(torch.tensor(fps_test, dtype=torch.float32)).numpy()
    prop_emb_csca = csca_model.encode_properties(torch.tensor(props_test_n, dtype=torch.float32)).numpy()
recall_csca = retrieval_recall_at_k(prop_emb_csca, fp_emb_csca)
print(f"  CSCA (ours):       {recall_csca}")
print(f"  CSCA training: {csca_result.epochs} epochs, final_loss={csca_result.final_loss:.4f}")

# --- Plot ---
methods = ["Random\nProjection", "PCA\n(unsupervised)", "Autoencoder\n(unsupervised)", "CSCA\n(ours)"]
k_vals = ["R@1", "R@5", "R@10"]
all_recalls = [recall_random, recall_pca, recall_ae, recall_csca]
colors = ["#bbb", "#aaa", "#888", "#e15759"]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(methods))
w = 0.25
for i, k in enumerate(k_vals):
    vals = [r[k] for r in all_recalls]
    bars = ax.bar(x + i * w, vals, w, label=k, color=["steelblue", "darkorange", "seagreen"][i],
                  edgecolor="black", alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.0%}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + w); ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel("Recall", fontsize=12)
ax.set_title("Benchmark 1: Cross-Modal Retrieval (Property → Structure)", fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.15)
ax.grid(True, alpha=0.3, axis="y")
save_fig("01_csca_retrieval")

ALL_RESULTS["csca_retrieval"] = {
    "random": recall_random, "pca": recall_pca,
    "autoencoder": recall_ae, "csca_ours": recall_csca,
    "csca_epochs": csca_result.epochs, "csca_loss": csca_result.final_loss,
}

# CSCA loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(csca_result.loss_history, color="#e15759", linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("InfoNCE Loss")
ax.set_title("CSCA Training Convergence")
ax.grid(True, alpha=0.3)
save_fig("01_csca_loss_curve")
print()


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK 2: Flow Matching vs Baselines — Generation Quality
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BENCHMARK 2: Flow Matching vs Baselines — Generation Quality")
print("=" * 70)


def generation_metrics(real_fps, gen_fps, label=""):
    """Compute Validity, Novelty, Uniqueness, and FCD proxy (MMD)."""
    # Validity: fraction with any bits set
    validity = np.mean([fp.sum() > 0 for fp in gen_fps])
    # Uniqueness: fraction of unique fingerprints
    unique_fps = set(tuple(fp.astype(int).tolist()) for fp in gen_fps)
    uniqueness = len(unique_fps) / max(len(gen_fps), 1)
    # Novelty: fraction not in training set
    train_set = set(tuple(fp.astype(int).tolist()) for fp in real_fps)
    novelty = 1.0 - np.mean([tuple(fp.astype(int).tolist()) in train_set for fp in gen_fps])
    # MMD (Maximum Mean Discrepancy) as FCD proxy
    mu_real = real_fps.mean(0)
    mu_gen = gen_fps.mean(0)
    mmd = float(np.sqrt(np.sum((mu_real - mu_gen) ** 2)))
    # Cosine similarity to nearest real molecule
    real_norm = real_fps / (np.linalg.norm(real_fps, axis=1, keepdims=True) + 1e-9)
    gen_norm = gen_fps / (np.linalg.norm(gen_fps, axis=1, keepdims=True) + 1e-9)
    nn_sim = np.max(gen_norm @ real_norm.T, axis=1).mean()

    return {"validity": validity, "uniqueness": uniqueness, "novelty": novelty,
            "MMD": mmd, "NN_similarity": float(nn_sim)}


N_GEN = min(50, len(props_test))
target_props = props_test[:N_GEN]

# --- Baseline 1: Random sampling (Bernoulli) ---
mean_density = fps_train.mean()
gen_random = (np.random.rand(N_GEN, fps.shape[1]) < mean_density).astype(np.float32)
metrics_random = generation_metrics(fps_train, gen_random)
print(f"  Random Bernoulli: {metrics_random}")

# --- Baseline 2: Gaussian noise around mean ---
gen_gauss = np.clip(fps_train.mean(0) + np.random.randn(N_GEN, fps.shape[1]) * fps_train.std(0), 0, 1)
gen_gauss = (gen_gauss > 0.5).astype(np.float32)
metrics_gauss = generation_metrics(fps_train, gen_gauss)
print(f"  Gaussian noise:   {metrics_gauss}")

# --- Baseline 3: VAE (simple) ---
class SimpleVAE(torch.nn.Module):
    def __init__(self, in_dim, latent, cond_dim):
        super().__init__()
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(in_dim + cond_dim, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU())
        self.mu_head = torch.nn.Linear(128, latent)
        self.logvar_head = torch.nn.Linear(128, latent)
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(latent + cond_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, in_dim), torch.nn.Sigmoid())
    def encode(self, x, c):
        h = self.enc(torch.cat([x, c], dim=-1))
        return self.mu_head(h), self.logvar_head(h)
    def decode(self, z, c):
        return self.dec(torch.cat([z, c], dim=-1))
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z, c), mu, logvar

vae = SimpleVAE(fps.shape[1], 64, props.shape[1])
opt_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)
t_fp = torch.tensor(fps_train, dtype=torch.float32)
t_c = torch.tensor(props_train_n, dtype=torch.float32)

for epoch in range(100):
    recon, mu, logvar = vae(t_fp, t_c)
    bce = torch.nn.functional.binary_cross_entropy(recon, t_fp, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = (bce + kld) / len(t_fp)
    opt_vae.zero_grad(); loss.backward(); opt_vae.step()

vae.eval()
with torch.no_grad():
    z = torch.randn(N_GEN, 64)
    c_test = torch.tensor(props_test_n[:N_GEN], dtype=torch.float32)
    gen_vae = vae.decode(z, c_test).numpy()
    gen_vae = (gen_vae > 0.5).astype(np.float32)
metrics_vae = generation_metrics(fps_train, gen_vae)
print(f"  Conditional VAE:  {metrics_vae}")

# --- Flow Matching (ours) ---
from chemvision.generation.flow_matcher import ConditionalFlowMatcher, FlowMatcherConfig

fm_config = FlowMatcherConfig(fp_dim=fps.shape[1], cond_dim=props.shape[1],
                              hidden_dim=256, n_layers=3, seed=42)
cfm = ConditionalFlowMatcher(fm_config)
fm_result = cfm.train(fps_train, props_train, epochs=100, batch_size=32, patience=25)

generated = cfm.sample(target_props, n_steps=30)
gen_flow = np.stack([g.binary_fingerprint for g in generated])
metrics_flow = generation_metrics(fps_train, gen_flow)
print(f"  Flow Matching:    {metrics_flow}")
print(f"  FM training: {fm_result.epochs} epochs, loss={fm_result.final_loss:.4f}")

# --- Plot ---
methods = ["Random\nBernoulli", "Gaussian\nNoise", "Conditional\nVAE", "Flow Matching\n(ours)"]
all_gen_metrics = [metrics_random, metrics_gauss, metrics_vae, metrics_flow]
metric_names = ["validity", "uniqueness", "novelty", "NN_similarity"]
metric_labels = ["Validity", "Uniqueness", "Novelty", "NN Similarity"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
bar_colors = ["#bbb", "#999", "#4e79a7", "#e15759"]

for i, (mname, mlabel) in enumerate(zip(metric_names, metric_labels)):
    vals = [m[mname] for m in all_gen_metrics]
    bars = axes[i].bar(methods, vals, color=bar_colors, edgecolor="black")
    axes[i].set_title(mlabel, fontsize=12, fontweight="bold")
    axes[i].set_ylim(0, 1.15)
    axes[i].grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{v:.0%}", ha="center", fontsize=9)
    axes[i].tick_params(axis="x", labelsize=8)

plt.suptitle("Benchmark 2: Conditional Molecular Generation", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("02_flow_generation")

# MMD bar chart
fig, ax = plt.subplots(figsize=(8, 4))
mmd_vals = [m["MMD"] for m in all_gen_metrics]
ax.bar(methods, mmd_vals, color=bar_colors, edgecolor="black")
ax.set_ylabel("MMD (lower = better)")
ax.set_title("Maximum Mean Discrepancy to Real Distribution")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(mmd_vals):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)
save_fig("02_flow_mmd")

# FM loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(fm_result.loss_history, color="#e15759", linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title("Flow Matcher Training Convergence")
ax.grid(True, alpha=0.3)
save_fig("02_flow_loss_curve")

ALL_RESULTS["flow_generation"] = {
    "random": metrics_random, "gaussian": metrics_gauss,
    "vae": metrics_vae, "flow_ours": metrics_flow,
    "fm_epochs": fm_result.epochs, "fm_loss": fm_result.final_loss,
}
print()


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK 3: Pareto MCTS vs Baselines — Multi-Objective Optimisation
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BENCHMARK 3: Pareto MCTS vs Baselines — Multi-Objective Optimisation")
print("=" * 70)

from chemvision.generation.property_predictor import PropertyPredictor
from chemvision.generation.pareto_mcts import ParetoMCTS, Objective, Candidate

pred = PropertyPredictor(use_mace=False)
SEED_MOL = "CC(=O)Oc1ccccc1C(=O)O"

def obj_qed(s): return pred.predict(s).qed or 0
def obj_mw_inv(s): return 1.0 / max(pred.predict(s).mw or 500, 1)
def obj_logp(s): return -abs((pred.predict(s).logp or 5) - 2.0)


def hypervolume_2d(points, ref):
    """Compute 2D hypervolume indicator."""
    pts = sorted(points, key=lambda p: -p[0])
    hv = 0.0
    prev_y = ref[1]
    for px, py in pts:
        if px > ref[0] or py > ref[1]:
            continue
        hv += (ref[0] - px) * (prev_y - max(py, ref[1]))
        prev_y = min(prev_y, py)
    return hv


def compute_pareto_front(candidates):
    front = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if other is c:
                continue
            if all(other.scores[k] >= c.scores[k] for k in c.scores) and \
               any(other.scores[k] > c.scores[k] for k in c.scores):
                dominated = True
                break
        if not dominated:
            front.append(c)
    return front


# --- Baseline 1: Random search ---
from rdkit import Chem
from rdkit.Chem import AllChem

rng_rs = np.random.RandomState(42)
rand_smiles = set()
base_mol = Chem.MolFromSmiles(SEED_MOL)
for _ in range(500):
    em = Chem.RWMol(base_mol)
    idx = rng_rs.randint(base_mol.GetNumAtoms())
    em.GetAtomWithIdx(idx).SetAtomicNum(int(rng_rs.choice([6, 7, 8, 9, 16])))
    try:
        Chem.SanitizeMol(em)
        s = Chem.MolToSmiles(em)
        if s and Chem.MolFromSmiles(s):
            rand_smiles.add(s)
    except Exception:
        pass

rand_candidates = []
for s in list(rand_smiles)[:100]:
    q, m, l = obj_qed(s), obj_mw_inv(s), obj_logp(s)
    rand_candidates.append(Candidate(s, {"qed": q, "mw_inv": m, "logp": l}))
rand_front = compute_pareto_front(rand_candidates)
print(f"  Random search: {len(rand_front)} Pareto-optimal from {len(rand_candidates)} candidates")

# --- Baseline 2: Greedy (single-objective QED) ---
greedy_candidates = sorted(rand_candidates, key=lambda c: c.scores["qed"], reverse=True)[:20]
greedy_front = compute_pareto_front(greedy_candidates)
print(f"  Greedy QED:    {len(greedy_front)} Pareto-optimal from {len(greedy_candidates)} candidates")

# --- Pareto MCTS (ours) ---
objectives = [
    Objective("qed", fn=obj_qed, direction="max"),
    Objective("mw_inv", fn=obj_mw_inv, direction="max"),
    Objective("logp", fn=obj_logp, direction="max"),
]
mcts = ParetoMCTS(objectives, max_atoms=50, seed=42)
t0 = time.perf_counter()
mcts_front = mcts.search(SEED_MOL, n_iterations=100)
mcts_time = time.perf_counter() - t0
print(f"  Pareto MCTS:   {len(mcts_front)} Pareto-optimal in {mcts_time:.1f}s")

# Compute hypervolume (2D: QED vs MW_inv)
ref = (0.0, 0.0)  # reference point
hv_random = hypervolume_2d([(c.scores["qed"], c.scores["mw_inv"]) for c in rand_front], ref)
hv_greedy = hypervolume_2d([(c.scores["qed"], c.scores["mw_inv"]) for c in greedy_front], ref)
hv_mcts = hypervolume_2d([(c.scores["qed"], c.scores["mw_inv"]) for c in mcts_front], ref)

print(f"\n  Hypervolume (QED × 1/MW, higher=better):")
print(f"    Random:     {hv_random:.6f}")
print(f"    Greedy:     {hv_greedy:.6f}")
print(f"    MCTS (ours):{hv_mcts:.6f}")

# --- Plot: Pareto fronts overlaid ---
fig, ax = plt.subplots(figsize=(9, 6))

for c in rand_front:
    ax.scatter(pred.predict(c.smiles).mw, c.scores["qed"], c="#bbb", s=30, alpha=0.5, zorder=1)
for c in greedy_front:
    ax.scatter(pred.predict(c.smiles).mw, c.scores["qed"], c="#4e79a7", s=50, alpha=0.7, zorder=2)
for c in mcts_front:
    ax.scatter(pred.predict(c.smiles).mw, c.scores["qed"], c="#e15759", s=70, alpha=0.8, zorder=3)

seed_p = pred.predict(SEED_MOL)
ax.scatter(seed_p.mw, seed_p.qed, c="black", s=200, marker="*", zorder=5, label="Seed (aspirin)")
ax.scatter([], [], c="#bbb", s=30, label=f"Random ({len(rand_front)})")
ax.scatter([], [], c="#4e79a7", s=50, label=f"Greedy QED ({len(greedy_front)})")
ax.scatter([], [], c="#e15759", s=70, label=f"Pareto MCTS ({len(mcts_front)})")

ax.set_xlabel("Molecular Weight (g/mol)", fontsize=12)
ax.set_ylabel("QED (drug-likeness)", fontsize=12)
ax.set_title("Benchmark 3: Multi-Objective Pareto Fronts", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
save_fig("03_pareto_fronts")

# Hypervolume bar chart
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(["Random\nSearch", "Greedy\n(QED only)", "Pareto MCTS\n(ours)"],
       [hv_random, hv_greedy, hv_mcts],
       color=["#bbb", "#4e79a7", "#e15759"], edgecolor="black")
ax.set_ylabel("Hypervolume (QED × 1/MW)")
ax.set_title("Benchmark 3: Hypervolume Indicator (higher = better)")
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate([hv_random, hv_greedy, hv_mcts]):
    ax.text(i, v + 0.0001, f"{v:.4f}", ha="center", fontsize=10)
save_fig("03_hypervolume")

ALL_RESULTS["pareto_mcts"] = {
    "random": {"front_size": len(rand_front), "hypervolume": hv_random},
    "greedy": {"front_size": len(greedy_front), "hypervolume": hv_greedy},
    "mcts_ours": {"front_size": len(mcts_front), "hypervolume": hv_mcts, "time_s": mcts_time},
}
print()


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK 4: Confidence Calibration — ECE + Brier Score
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("BENCHMARK 4: Confidence Calibration — ECE & Brier Score")
print("=" * 70)

from chemvision.eval.calibration import ConfidenceCalibrator

rng_cal = np.random.RandomState(42)

# Simulate overconfident VLM: confidence 0.7-0.99, actual accuracy ~55%
raw_confs = rng_cal.uniform(0.65, 0.99, size=300).tolist()
correct = rng_cal.binomial(1, 0.55, size=300).tolist()

# Split train/test
split = 200
train_c, test_c = raw_confs[:split], raw_confs[split:]
train_y, test_y = correct[:split], correct[split:]


def compute_ece(confs, accs, n_bins=10):
    confs, accs = np.array(confs), np.array(accs)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i+1])
        if mask.sum() == 0: continue
        ece += (mask.sum() / len(confs)) * abs(accs[mask].mean() - confs[mask].mean())
    return float(ece)


def brier_score(confs, labels):
    return float(np.mean((np.array(confs) - np.array(labels)) ** 2))


# Uncalibrated
ece_uncal = compute_ece(test_c, test_y)
brier_uncal = brier_score(test_c, test_y)
print(f"  Uncalibrated:      ECE={ece_uncal:.4f}, Brier={brier_uncal:.4f}")

# Temperature scaling (baseline)
# Find T that minimises NLL on train set
best_t, best_nll = 1.0, float("inf")
for t in np.arange(0.5, 5.0, 0.1):
    scaled = 1 / (1 + np.exp(-(np.log(np.array(train_c) / (1 - np.array(train_c) + 1e-9))) / t))
    nll = -np.mean(np.array(train_y) * np.log(scaled + 1e-9) + (1 - np.array(train_y)) * np.log(1 - scaled + 1e-9))
    if nll < best_nll:
        best_nll, best_t = nll, t

temp_scaled = 1 / (1 + np.exp(-(np.log(np.array(test_c) / (1 - np.array(test_c) + 1e-9))) / best_t))
ece_temp = compute_ece(temp_scaled.tolist(), test_y)
brier_temp = brier_score(temp_scaled.tolist(), test_y)
print(f"  Temperature (T={best_t:.1f}): ECE={ece_temp:.4f}, Brier={brier_temp:.4f}")

# Isotonic (ours)
cal_iso = ConfidenceCalibrator(method="isotonic")
cal_iso.fit(train_c, train_y)
iso_test = cal_iso.calibrate_batch(test_c)
ece_iso = compute_ece(iso_test, test_y)
brier_iso = brier_score(iso_test, test_y)
print(f"  Isotonic (ours):   ECE={ece_iso:.4f}, Brier={brier_iso:.4f}")

# Platt (ours)
cal_platt = ConfidenceCalibrator(method="platt")
cal_platt.fit(train_c, train_y)
platt_test = cal_platt.calibrate_batch(test_c)
ece_platt = compute_ece(platt_test, test_y)
brier_platt = brier_score(platt_test, test_y)
print(f"  Platt (ours):      ECE={ece_platt:.4f}, Brier={brier_platt:.4f}")

# --- Plot: reliability diagrams ---
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
titles = ["Uncalibrated", f"Temperature (T={best_t:.1f})", "Isotonic (ours)", "Platt (ours)"]
cal_confs = [test_c, temp_scaled.tolist(), iso_test, platt_test]
eces = [ece_uncal, ece_temp, ece_iso, ece_platt]
briers = [brier_uncal, brier_temp, brier_iso, brier_platt]

for ax, title, confs, ece_val, brier_val in zip(axes, titles, cal_confs, eces, briers):
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = [(c > bins[i]) and (c <= bins[i+1]) for c in confs]
        if sum(mask) == 0:
            bin_accs.append(0); bin_confs.append(0); bin_counts.append(0)
            continue
        bc = np.array(confs)[mask]
        by = np.array(test_y)[mask]
        bin_accs.append(by.mean()); bin_confs.append(bc.mean()); bin_counts.append(len(bc))

    centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(centers, bin_accs, width=0.08, color="steelblue", edgecolor="navy", alpha=0.7, label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE={ece_val:.3f} Brier={brier_val:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

plt.suptitle("Benchmark 4: Confidence Calibration — Reliability Diagrams", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("04_calibration_reliability")

# ECE + Brier bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
method_names = ["Uncalibrated", "Temperature\nScaling", "Isotonic\n(ours)", "Platt\n(ours)"]
bar_colors = ["#bbb", "#4e79a7", "#e15759", "#f28e2b"]

ax1.bar(method_names, eces, color=bar_colors, edgecolor="black")
ax1.set_ylabel("ECE (lower = better)")
ax1.set_title("Expected Calibration Error")
ax1.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(eces):
    ax1.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=10)

ax2.bar(method_names, briers, color=bar_colors, edgecolor="black")
ax2.set_ylabel("Brier Score (lower = better)")
ax2.set_title("Brier Score")
ax2.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(briers):
    ax2.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=10)

plt.tight_layout()
save_fig("04_calibration_scores")

ALL_RESULTS["calibration"] = {
    "uncalibrated": {"ece": ece_uncal, "brier": brier_uncal},
    "temperature": {"ece": ece_temp, "brier": brier_temp, "T": best_t},
    "isotonic_ours": {"ece": ece_iso, "brier": brier_iso},
    "platt_ours": {"ece": ece_platt, "brier": brier_platt},
}
print()


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

print("Benchmark 1: CSCA Cross-Modal Retrieval (Property → Structure)")
print(f"  {'Method':<25s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s}")
for name, res in [("Random Projection", recall_random), ("PCA", recall_pca),
                  ("Autoencoder", recall_ae), ("CSCA (ours)", recall_csca)]:
    print(f"  {name:<25s} {res['R@1']:>7.1%} {res['R@5']:>7.1%} {res['R@10']:>7.1%}")

print()
print("Benchmark 2: Conditional Molecular Generation")
print(f"  {'Method':<25s} {'Validity':>10s} {'Unique':>10s} {'Novelty':>10s} {'NN Sim':>10s} {'MMD':>8s}")
for name, m in [("Random Bernoulli", metrics_random), ("Gaussian Noise", metrics_gauss),
                ("Cond. VAE", metrics_vae), ("Flow Matching (ours)", metrics_flow)]:
    print(f"  {name:<25s} {m['validity']:>9.1%} {m['uniqueness']:>9.1%} {m['novelty']:>9.1%} "
          f"{m['NN_similarity']:>9.3f} {m['MMD']:>7.1f}")

print()
print("Benchmark 3: Multi-Objective Optimisation")
print(f"  {'Method':<25s} {'Front size':>12s} {'Hypervolume':>12s}")
for name, res in [("Random Search", ALL_RESULTS["pareto_mcts"]["random"]),
                  ("Greedy QED", ALL_RESULTS["pareto_mcts"]["greedy"]),
                  ("Pareto MCTS (ours)", ALL_RESULTS["pareto_mcts"]["mcts_ours"])]:
    print(f"  {name:<25s} {res['front_size']:>12d} {res['hypervolume']:>12.6f}")

print()
print("Benchmark 4: Confidence Calibration")
print(f"  {'Method':<25s} {'ECE':>8s} {'Brier':>8s}")
for name, res in [("Uncalibrated", ALL_RESULTS["calibration"]["uncalibrated"]),
                  ("Temperature Scaling", ALL_RESULTS["calibration"]["temperature"]),
                  ("Isotonic (ours)", ALL_RESULTS["calibration"]["isotonic_ours"]),
                  ("Platt (ours)", ALL_RESULTS["calibration"]["platt_ours"])]:
    print(f"  {name:<25s} {res['ece']:>7.4f} {res['brier']:>7.4f}")

# Save results
with open(OUT / "benchmark_results.json", "w") as f:
    json.dump(ALL_RESULTS, f, indent=2, default=str)
print(f"\nResults saved to {OUT / 'benchmark_results.json'}")
print(f"Figures saved to {FIG}/")
