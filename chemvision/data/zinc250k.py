"""ZINC250K dataset loader.

Downloads and caches the ZINC250K dataset (Irwin et al., J. Chem. Inf. Model. 2012)
— 249,455 drug-like molecules with precomputed LogP, QED, and SAS properties.

This is the standard dataset used by:
  - Chemical VAE (Gomez-Bombarelli et al., ACS Cent. Sci. 2018)
  - JT-VAE (Jin et al., ICML 2018)
  - MOSES benchmark (Polykovskiy et al., Front. Pharmacol. 2020)
  - GuacaMol benchmark (Brown et al., J. Chem. Inf. Model. 2019)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ZINC_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)
_CACHE_DIR = Path.home() / ".cache" / "chemvision"


def load_zinc250k(
    cache_dir: Path | str | None = None,
    max_molecules: int | None = None,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Load ZINC250K SMILES + properties.

    Returns
    -------
    (smiles_list, properties)
        smiles_list: list of canonical SMILES (up to 249K)
        properties: (N, 3) array of [LogP, QED, SAS]
    """
    cache = Path(cache_dir) if cache_dir else _CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    csv_path = cache / "zinc250k.csv"

    if not csv_path.exists():
        logger.info("Downloading ZINC250K to %s ...", csv_path)
        import requests
        resp = requests.get(_ZINC_URL, timeout=60)
        resp.raise_for_status()
        csv_path.write_bytes(resp.content)
        logger.info("Downloaded %.1f MB", len(resp.content) / 1e6)

    df = pd.read_csv(csv_path)
    # Columns: smiles, logP, qed, SAS
    smiles_col = [c for c in df.columns if "smile" in c.lower()][0]
    smiles_list = [s.strip() for s in df[smiles_col].tolist()]

    # Build property array
    prop_cols = []
    for name in ["logP", "qed", "SAS"]:
        matches = [c for c in df.columns if name.lower() in c.lower()]
        if matches:
            prop_cols.append(matches[0])
    properties = df[prop_cols].values.astype(np.float32)

    if max_molecules and max_molecules < len(smiles_list):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(smiles_list), max_molecules, replace=False)
        smiles_list = [smiles_list[i] for i in idx]
        properties = properties[idx]

    logger.info("Loaded %d molecules from ZINC250K (%d properties each)",
                len(smiles_list), properties.shape[1])
    return smiles_list, properties


def zinc250k_splits(
    smiles: list[str],
    properties: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Deterministic train/val/test split."""
    rng = np.random.RandomState(seed)
    n = len(smiles)
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    def _select(idx: np.ndarray) -> tuple[list[str], np.ndarray]:
        return [smiles[i] for i in idx], properties[idx]

    return {
        "train": _select(perm[:n_train]),
        "val": _select(perm[n_train:n_train + n_val]),
        "test": _select(perm[n_train + n_val:]),
    }
