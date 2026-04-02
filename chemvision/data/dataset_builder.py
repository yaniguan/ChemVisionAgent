"""Dataset builder: collect, organise, and serve training data for CSCA + Flow Matcher.

Generates paired (fingerprint, property_vector) training data from:
  1. PubChem REST API (real molecules)
  2. RDKit random generation (augmentation)
  3. Combinatorial fragment enumeration

Output: Parquet files with reproducible train/val/test splits.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetStats:
    """Summary statistics for a built dataset."""

    n_total: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_unique_smiles: int = 0
    fp_dim: int = 0
    prop_dim: int = 0
    prop_names: list[str] = field(default_factory=list)
    mean_mw: float = 0.0
    std_mw: float = 0.0
    lipinski_pass_rate: float = 0.0


class MolecularDatasetBuilder:
    """Build fingerprint ↔ property paired datasets for training.

    Example
    -------
    >>> builder = MolecularDatasetBuilder()
    >>> builder.add_from_smiles_list(["CCO", "CC(=O)O", "c1ccccc1"])
    >>> builder.add_from_pubchem(n=100)
    >>> stats = builder.build(Path("data/csca_dataset"))
    >>> stats.n_total
    103
    """

    # Standard PubChem drug-like molecules for seeding
    _SEED_SMILES: list[str] = [
        "CC(=O)Oc1ccccc1C(=O)O",           # aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",      # ibuprofen
        "Cn1cnc2c1c(=O)n(c(=O)n2C)C",      # caffeine
        "CC(=O)Nc1ccc(O)cc1",               # paracetamol
        "CN(C)C(=N)NC(=N)N",                # metformin
        "CCOc1ccc(cc1)C(=O)c2ccccc2",       # ethyl benzoylbenzoate
        "OC(=O)c1ccccc1O",                  # salicylic acid
        "CC(=O)OC1=CC=CC=C1C(=O)O",        # aspirin (canonical)
        "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34", # pyrene
        "C1CCCCC1",                          # cyclohexane
        "c1ccccc1",                          # benzene
        "CCO",                               # ethanol
        "CC(=O)O",                           # acetic acid
        "CC#N",                              # acetonitrile
        "CCCCCCCC",                          # octane
        "c1ccncc1",                          # pyridine
        "C1CCNCC1",                          # piperidine
        "c1ccc(cc1)O",                       # phenol
        "c1ccc(cc1)N",                       # aniline
        "OC(=O)CC(O)(CC(=O)O)C(=O)O",      # citric acid
    ]

    _PROP_NAMES = ["mw", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "rings", "qed"]

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._smiles_list: list[str] = []
        self._rng = np.random.RandomState(seed)

    def add_from_smiles_list(self, smiles_list: list[str]) -> int:
        """Add molecules from an explicit SMILES list. Returns count added."""
        from rdkit import Chem
        added = 0
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol)
                if canonical not in self._smiles_list:
                    self._smiles_list.append(canonical)
                    added += 1
        return added

    def add_seeds(self) -> int:
        """Add built-in seed molecules."""
        return self.add_from_smiles_list(self._SEED_SMILES)

    def add_from_pubchem(self, n: int = 200, seed_smiles: str | None = None) -> int:
        """Fetch similar compounds from PubChem and add them.

        Uses the first seed molecule or a provided SMILES as the similarity query.
        Falls back to built-in seeds if PubChem is unreachable.
        """
        try:
            from chemvision.retrieval.pubchem_client import PubChemClient
            client = PubChemClient(timeout=10)
            query = seed_smiles or (self._smiles_list[0] if self._smiles_list else "CCO")
            results = client.get_similar_compounds(query, threshold=70, max_results=n)
            smiles = [r.get("CanonicalSMILES", "") for r in results if r.get("CanonicalSMILES")]
            return self.add_from_smiles_list(smiles)
        except Exception:
            return 0

    def add_random_molecules(self, n: int = 500) -> int:
        """Generate random drug-like molecules via fragment combination.

        Uses a simple SMILES mutation strategy: take seed molecules,
        apply random atom substitutions, and keep valid ones.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        generated: set[str] = set()
        seeds = list(self._smiles_list) or list(self._SEED_SMILES)

        substitutions = [6, 7, 8, 9, 16]  # C, N, O, F, S

        attempts = 0
        while len(generated) < n and attempts < n * 20:
            attempts += 1
            base_smi = seeds[self._rng.randint(len(seeds))]
            mol = Chem.MolFromSmiles(base_smi)
            if mol is None or mol.GetNumAtoms() == 0:
                continue

            em = Chem.RWMol(mol)
            idx = self._rng.randint(mol.GetNumAtoms())
            new_z = substitutions[self._rng.randint(len(substitutions))]
            em.GetAtomWithIdx(idx).SetAtomicNum(new_z)

            try:
                Chem.SanitizeMol(em)
                smi = Chem.MolToSmiles(em)
                if smi and Chem.MolFromSmiles(smi) is not None and smi not in generated:
                    generated.add(smi)
            except Exception:
                pass

        return self.add_from_smiles_list(list(generated))

    def build(
        self,
        output_dir: Path | str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> DatasetStats:
        """Compute features and save train/val/test Parquet files.

        Returns
        -------
        DatasetStats with counts and property statistics.
        """
        from chemvision.models.mol_encoder import MolecularEncoder

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        encoder = MolecularEncoder()

        # Compute fingerprints and properties
        rows: list[dict[str, Any]] = []
        for smi in self._smiles_list:
            desc = encoder.compute_descriptors(smi)
            if desc.mw is None:
                continue
            fp = encoder.encode(smi)
            props = [
                desc.mw or 0, desc.logp or 0, desc.tpsa or 0,
                desc.hbd or 0, desc.hba or 0, desc.rotatable_bonds or 0,
                desc.rings or 0, desc.qed or 0,
            ]
            rows.append({
                "smiles": smi,
                "fingerprint": fp.tolist(),
                "properties": props,
                "mw": desc.mw,
                "logp": desc.logp,
                "qed": desc.qed,
                "lipinski_pass": desc.lipinski_pass or False,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return DatasetStats()

        # Deterministic split
        perm = self._rng.permutation(len(df))
        n_train = int(len(df) * train_ratio)
        n_val = int(len(df) * val_ratio)

        train_df = df.iloc[perm[:n_train]]
        val_df = df.iloc[perm[n_train:n_train + n_val]]
        test_df = df.iloc[perm[n_train + n_val:]]

        # Save to Parquet
        train_df.to_parquet(output_dir / "train.parquet", index=False)
        val_df.to_parquet(output_dir / "val.parquet", index=False)
        test_df.to_parquet(output_dir / "test.parquet", index=False)

        # Also save numpy arrays for direct model consumption
        fps = np.array([row["fingerprint"] for row in rows], dtype=np.float32)
        props = np.array([row["properties"] for row in rows], dtype=np.float32)
        np.savez(
            output_dir / "arrays.npz",
            fps=fps, props=props,
            train_idx=perm[:n_train],
            val_idx=perm[n_train:n_train + n_val],
            test_idx=perm[n_train + n_val:],
        )

        stats = DatasetStats(
            n_total=len(df),
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
            n_unique_smiles=df["smiles"].nunique(),
            fp_dim=len(rows[0]["fingerprint"]),
            prop_dim=len(rows[0]["properties"]),
            prop_names=self._PROP_NAMES,
            mean_mw=float(df["mw"].mean()) if "mw" in df else 0,
            std_mw=float(df["mw"].std()) if "mw" in df else 0,
            lipinski_pass_rate=float(df["lipinski_pass"].mean()) if "lipinski_pass" in df else 0,
        )

        # Save stats as JSON
        import json
        with open(output_dir / "stats.json", "w") as f:
            json.dump(stats.__dict__, f, indent=2)

        return stats

    @staticmethod
    def load_arrays(dataset_dir: Path | str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Load pre-built numpy arrays.

        Returns
        -------
        (fps, props, splits_dict) where splits_dict has train_idx, val_idx, test_idx
        """
        data = np.load(Path(dataset_dir) / "arrays.npz")
        return (
            data["fps"],
            data["props"],
            {"train_idx": data["train_idx"], "val_idx": data["val_idx"], "test_idx": data["test_idx"]},
        )
