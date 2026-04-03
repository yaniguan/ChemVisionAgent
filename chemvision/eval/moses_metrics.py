"""MOSES-standard molecular generation metrics.

Implements the 6 core metrics from the Molecular Sets (MOSES) benchmark
(Polykovskiy et al., Frontiers in Pharmacology, 2020):

  1. Validity     — fraction of generated SMILES that parse to valid molecules
  2. Uniqueness   — fraction of valid molecules that are unique
  3. Novelty      — fraction of valid+unique molecules NOT in the training set
  4. IntDiv (p=1) — internal diversity: 1 - mean Tanimoto similarity within generated set
  5. IntDiv (p=2) — sqrt variant for sensitivity to clusters
  6. FCD proxy    — Frechet ChemNet Distance approximated via Morgan FP statistics

These metrics are the standard for evaluating molecular generative models.
Any paper submission requires reporting these numbers.

Reference: https://github.com/molecularsets/moses
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs import TanimotoSimilarity

logger = logging.getLogger(__name__)


@dataclass
class MOSESMetrics:
    """Standard MOSES benchmark metrics."""

    n_generated: int = 0
    n_valid: int = 0
    n_unique: int = 0
    n_novel: int = 0

    validity: float = 0.0          # valid / generated
    uniqueness: float = 0.0        # unique / valid
    novelty: float = 0.0           # novel / unique
    int_div_1: float = 0.0         # 1 - mean pairwise Tanimoto
    int_div_2: float = 0.0         # sqrt variant
    fcd: float = 0.0               # proper Frechet ChemNet Distance (lower = more realistic)
    fcd_proxy: float = 0.0         # old proxy: norm of mean difference (kept for backwards compat)
    scaffold_diversity: float = 0.0  # fraction of unique Murcko scaffolds

    # Property distribution metrics
    mw_mean: float = 0.0
    mw_std: float = 0.0
    logp_mean: float = 0.0
    logp_std: float = 0.0
    qed_mean: float = 0.0

    def summary(self) -> str:
        return (
            f"MOSES Metrics (n={self.n_generated}):\n"
            f"  Validity:   {self.validity:.1%}\n"
            f"  Uniqueness: {self.uniqueness:.1%}\n"
            f"  Novelty:    {self.novelty:.1%}\n"
            f"  IntDiv1:    {self.int_div_1:.4f}\n"
            f"  IntDiv2:    {self.int_div_2:.4f}\n"
            f"  FCD:        {self.fcd:.2f}\n"
            f"  FCD proxy:  {self.fcd_proxy:.2f}\n"
            f"  Scaffold diversity: {self.scaffold_diversity:.4f}\n"
            f"  MW:         {self.mw_mean:.1f} ± {self.mw_std:.1f}\n"
            f"  LogP:       {self.logp_mean:.2f}\n"
            f"  QED:        {self.qed_mean:.3f}"
        )


def _compute_fcd(fps_train: np.ndarray, fps_gen: np.ndarray) -> float:
    """Compute Frechet ChemNet Distance (proper Frechet distance).

    FCD = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*(C1*C2)^(1/2))

    where mu, C are mean and covariance of fingerprint distributions.
    Uses Morgan fingerprints as proxy for ChemNet activations.
    """
    mu1 = np.mean(fps_train, axis=0)
    mu2 = np.mean(fps_gen, axis=0)
    sigma1 = np.cov(fps_train, rowvar=False)
    sigma2 = np.cov(fps_gen, rowvar=False)

    diff = mu1 - mu2

    # Product of covariances
    from scipy.linalg import sqrtm
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fcd = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fcd)


def _scaffold_diversity(smiles_list: list[str]) -> float:
    """Fraction of unique Murcko scaffolds."""
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    scaffolds = set()
    valid = 0
    for smi in smiles_list:
        try:
            scaf = MurckoScaffoldSmiles(smi)
            scaffolds.add(scaf)
            valid += 1
        except ValueError:
            pass  # invalid SMILES for scaffold computation
    return len(scaffolds) / max(valid, 1)


def compute_moses_metrics(
    generated_smiles: list[str | None],
    training_smiles: list[str] | None = None,
    n_jobs: int = 1,
) -> MOSESMetrics:
    """Compute full MOSES benchmark metrics for a set of generated SMILES.

    Parameters
    ----------
    generated_smiles:
        List of generated SMILES strings (None entries count as invalid).
    training_smiles:
        Training set SMILES for novelty computation. If None, novelty = 1.0.

    Returns
    -------
    MOSESMetrics with all 6 core metrics + property statistics.
    """
    result = MOSESMetrics(n_generated=len(generated_smiles))

    # 1. Validity
    valid_mols: list[Any] = []
    valid_smiles: list[str] = []
    for smi in generated_smiles:
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canon = Chem.MolToSmiles(mol)
            valid_mols.append(mol)
            valid_smiles.append(canon)

    result.n_valid = len(valid_smiles)
    result.validity = result.n_valid / max(result.n_generated, 1)

    if not valid_smiles:
        return result

    # 2. Uniqueness
    unique_smiles = list(set(valid_smiles))
    result.n_unique = len(unique_smiles)
    result.uniqueness = result.n_unique / max(result.n_valid, 1)

    # 3. Novelty
    if training_smiles is not None:
        train_set = set(Chem.MolToSmiles(Chem.MolFromSmiles(s))
                       for s in training_smiles if Chem.MolFromSmiles(s) is not None)
        novel = [s for s in unique_smiles if s not in train_set]
        result.n_novel = len(novel)
        result.novelty = result.n_novel / max(result.n_unique, 1)
    else:
        result.n_novel = result.n_unique
        result.novelty = 1.0

    # 4 & 5. Internal Diversity (Tanimoto-based)
    unique_mols = [Chem.MolFromSmiles(s) for s in unique_smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in unique_mols if m]
    n_fps = len(fps)

    if n_fps >= 2:
        # Sample pairwise similarities (cap at 5000 pairs for speed)
        max_pairs = min(5000, n_fps * (n_fps - 1) // 2)
        sims = []
        rng = np.random.RandomState(42)
        for _ in range(max_pairs):
            i, j = rng.randint(n_fps, size=2)
            if i != j:
                sims.append(TanimotoSimilarity(fps[i], fps[j]))
        if sims:
            mean_sim = np.mean(sims)
            result.int_div_1 = 1.0 - mean_sim
            result.int_div_2 = 1.0 - np.sqrt(np.mean(np.array(sims) ** 2))

    # 6. FCD — proper Frechet distance + legacy proxy
    if training_smiles is not None and len(fps) >= 2:
        from rdkit.DataStructs import ConvertToNumpyArray

        train_fps = []
        for s in training_smiles[:5000]:
            mol = Chem.MolFromSmiles(s)
            if mol:
                fp_arr = np.zeros(2048)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                ConvertToNumpyArray(fp, fp_arr)
                train_fps.append(fp_arr)

        gen_fps = []
        for fp in fps[:5000]:
            fp_arr = np.zeros(2048)
            ConvertToNumpyArray(fp, fp_arr)
            gen_fps.append(fp_arr)

        if train_fps and gen_fps:
            train_fps_arr = np.array(train_fps)
            gen_fps_arr = np.array(gen_fps)

            # Legacy proxy (norm of mean difference)
            mu_train = np.mean(train_fps_arr, axis=0)
            mu_gen = np.mean(gen_fps_arr, axis=0)
            result.fcd_proxy = float(np.linalg.norm(mu_train - mu_gen))

            # Proper Frechet distance
            try:
                result.fcd = _compute_fcd(train_fps_arr, gen_fps_arr)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                logger.warning("FCD computation failed, falling back to proxy: %s", e)
                result.fcd = result.fcd_proxy

    # 7. Scaffold diversity
    if unique_smiles:
        result.scaffold_diversity = _scaffold_diversity(unique_smiles)

    # Property statistics
    mws, logps, qeds = [], [], []
    for mol in unique_mols:
        if mol:
            mws.append(Descriptors.MolWt(mol))
            logps.append(Descriptors.MolLogP(mol))
            try:
                from rdkit.Chem import QED
                qeds.append(QED.qed(mol))
            except ImportError:
                break  # no QED module — skip all
            except ValueError:
                pass  # QED computation failed for this molecule

    if mws:
        result.mw_mean = float(np.mean(mws))
        result.mw_std = float(np.std(mws))
    if logps:
        result.logp_mean = float(np.mean(logps))
    if qeds:
        result.qed_mean = float(np.mean(qeds))

    return result
