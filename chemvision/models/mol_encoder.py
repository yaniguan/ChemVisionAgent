"""Molecular encoder: RDKit-based 3D conformer generation and fingerprint embedding.

This module is the representation backbone for ChemVisionAgent's scientific pipeline.

Architecture
------------
1. **Fingerprint embedding** — 2048-bit Morgan (ECFP4) fingerprint as a float32 vector.
   Used as the embedding for the vector store and as input to downstream predictors.

2. **3D conformer generation** — ETKDG + UFF energy minimisation via RDKit.
   Produces atomic coordinates that can be fed to MACE or other 3D models.

3. **Descriptor computation** — Lipinski / ADMET descriptors via RDKit.

4. **Uni-Mol2 shim** — Optional: if ``unimol_tools`` is installed the encoder
   can delegate to Uni-Mol2 for SE(3)-equivariant embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# RDKit (required)
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


@dataclass
class ConformerResult:
    """3D atomic geometry for one molecule."""

    smiles: str
    atomic_numbers: list[int] = field(default_factory=list)
    coordinates: list[list[float]] = field(default_factory=list)  # Å, shape (N, 3)
    energy_kcal: float | None = None
    success: bool = False

    @property
    def num_atoms(self) -> int:
        return len(self.atomic_numbers)


@dataclass
class MolDescriptors:
    """Physicochemical descriptors for a molecule."""

    smiles: str
    mw: float | None = None
    logp: float | None = None
    tpsa: float | None = None
    hbd: int | None = None          # H-bond donors
    hba: int | None = None          # H-bond acceptors
    rotatable_bonds: int | None = None
    rings: int | None = None
    aromatic_rings: int | None = None
    heavy_atoms: int | None = None
    qed: float | None = None        # Quantitative Estimate of Drug-likeness (0–1)
    sa_score: float | None = None   # Synthetic Accessibility (1=easy, 10=hard)

    # Lipinski pass (MW<500, LogP<5, HBD≤5, HBA≤10)
    @property
    def lipinski_pass(self) -> bool | None:
        if None in (self.mw, self.logp, self.hbd, self.hba):
            return None
        return (
            self.mw < 500
            and self.logp < 5
            and self.hbd <= 5
            and self.hba <= 10
        )


class MolecularEncoder:
    """Convert SMILES strings into fingerprint embeddings, 3D conformers, and descriptors.

    Example
    -------
    >>> encoder = MolecularEncoder()
    >>> emb = encoder.encode("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    >>> emb.shape
    (2048,)
    >>> conf = encoder.generate_conformer("CC(=O)Oc1ccccc1C(=O)O")
    >>> conf.success
    True
    >>> desc = encoder.compute_descriptors("CC(=O)Oc1ccccc1C(=O)O")
    >>> desc.lipinski_pass
    True
    """

    def __init__(
        self,
        fp_radius: int = 2,
        fp_bits: int = 2048,
        n_conformers: int = 10,
        use_unimol: bool = False,
    ) -> None:
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.n_conformers = n_conformers
        self._unimol: Any = None

        if use_unimol:
            try:
                from unimol_tools import UniMolRepr  # type: ignore[import]

                self._unimol = UniMolRepr(data_type="molecule", remove_hs=True)
            except ImportError:
                pass  # graceful fallback to Morgan fingerprint

    # ------------------------------------------------------------------
    # Fingerprint embedding
    # ------------------------------------------------------------------

    def encode(self, smiles: str) -> np.ndarray:
        """Return a float32 Morgan fingerprint (2048-dim) for *smiles*.

        If Uni-Mol2 is available it returns a 512-dim SE(3)-equivariant
        embedding instead.

        Returns all-zeros on parse failure so downstream code stays simple.
        """
        if self._unimol is not None:
            return self._encode_unimol(smiles)
        return self._encode_morgan(smiles)

    def encode_batch(self, smiles_list: list[str]) -> np.ndarray:
        """Return (N, D) embedding matrix for a list of SMILES."""
        return np.stack([self.encode(s) for s in smiles_list])

    def _encode_morgan(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.fp_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, nBits=self.fp_bits)
        arr = np.zeros(self.fp_bits, dtype=np.float32)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        return arr

    def _encode_unimol(self, smiles: str) -> np.ndarray:
        try:
            result = self._unimol.get_repr([smiles], return_atomic_reprs=False)
            return np.array(result["cls_repr"][0], dtype=np.float32)
        except Exception:
            return self._encode_morgan(smiles)

    # ------------------------------------------------------------------
    # 3D conformer generation
    # ------------------------------------------------------------------

    def generate_conformer(self, smiles: str, seed: int = 42) -> ConformerResult:
        """Generate a low-energy 3D conformer via ETKDG + UFF minimisation.

        Parameters
        ----------
        smiles:
            Input SMILES string.
        seed:
            Random seed for reproducible ETKDG embedding.

        Returns
        -------
        ConformerResult
            Contains atomic numbers, 3D coordinates (Å), and UFF energy.
            ``success=False`` if embedding fails.
        """
        result = ConformerResult(smiles=smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return result

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 0  # use all available
        params.maxIterations = 200

        cids = AllChem.EmbedMultipleConfs(mol, numConfs=self.n_conformers, params=params)
        if not cids:
            return result

        # UFF energy minimisation — keep lowest-energy conformer
        energies: list[tuple[float, int]] = []
        ff_results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=1000)
        for cid, (converged, energy) in enumerate(ff_results):
            energies.append((energy, cid))
        energies.sort()
        best_cid = energies[0][1]

        conf = mol.GetConformer(best_cid)
        result.atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        result.coordinates = [
            list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())
        ]
        result.energy_kcal = energies[0][0]
        result.success = True
        return result

    # ------------------------------------------------------------------
    # Descriptor computation
    # ------------------------------------------------------------------

    def compute_descriptors(self, smiles: str) -> MolDescriptors:
        """Compute Lipinski / ADMET physicochemical descriptors.

        Uses RDKit Descriptors module. SA score is computed via the
        Ertl & Schuffenhauer (2009) algorithm if available in rdkit.
        """
        desc = MolDescriptors(smiles=smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return desc

        desc.mw = Descriptors.MolWt(mol)
        desc.logp = Descriptors.MolLogP(mol)
        desc.tpsa = Descriptors.TPSA(mol)
        desc.hbd = rdMolDescriptors.CalcNumHBD(mol)
        desc.hba = rdMolDescriptors.CalcNumHBA(mol)
        desc.rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        desc.rings = rdMolDescriptors.CalcNumRings(mol)
        desc.aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        desc.heavy_atoms = mol.GetNumHeavyAtoms()

        # QED (built into RDKit >= 2020)
        try:
            from rdkit.Chem import QED
            desc.qed = QED.qed(mol)
        except Exception:
            pass

        # SA score (Ertl & Schuffenhauer) — available via rdkit.Contrib.SA_Score
        try:
            from rdkit.Chem.rdMolDescriptors import CalcCrippenDescriptors  # noqa
            # Try sascorer from RDKit contrib
            import importlib
            sa_mod = importlib.import_module("rdkit.Contrib.SA_Score.sascorer")
            desc.sa_score = sa_mod.calculateScore(mol)
        except Exception:
            pass

        return desc

    # ------------------------------------------------------------------
    # Tanimoto similarity
    # ------------------------------------------------------------------

    def tanimoto(self, smiles_a: str, smiles_b: str) -> float:
        """Return the Tanimoto coefficient between two SMILES strings."""
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return 0.0
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, self.fp_radius, self.fp_bits)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, self.fp_radius, self.fp_bits)
        from rdkit.DataStructs import TanimotoSimilarity
        return float(TanimotoSimilarity(fp_a, fp_b))
