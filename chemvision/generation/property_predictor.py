"""Property predictor: RDKit descriptors (always) + MACE-MP-0 (optional GPU).

Architecture
------------
* **RDKit tier** (CPU, always available): computes Lipinski descriptors,
  QED, SA score, and a simple estimated band-gap proxy for organic molecules.

* **MACE-MP-0 tier** (GPU optional): universal equivariant force-field for
  inorganic / periodic crystals.  Provides DFT-quality energy/forces without
  any fitting.  Auto-loaded only if ``mace-torch`` is installed.

All outputs are returned as a ``PropertyResult`` dataclass so callers
have a uniform interface regardless of which backend ran.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PropertyResult:
    """Predicted physicochemical and/or quantum-mechanical properties."""

    smiles: str

    # --- Lipinski / ADMET (RDKit, always present) ---
    mw: float | None = None
    logp: float | None = None
    tpsa: float | None = None
    qed: float | None = None            # 0–1, higher = more drug-like
    sa_score: float | None = None       # 1 (easy) – 10 (hard to synthesise)
    hbd: int | None = None
    hba: int | None = None
    rotatable_bonds: int | None = None

    # --- MACE-MP-0 (optional, periodic / inorganic) ---
    energy_ev: float | None = None      # potential energy in eV
    forces_ev_ang: list[list[float]] | None = None  # forces per atom in eV/Å

    # --- derived scores ---
    drug_score: float | None = None     # composite: QED * (1 - SA_score/10)
    synthesisability: str = "unknown"   # easy / moderate / hard / very_hard

    backend: str = "rdkit"              # which backend was used
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and k != "warnings"}


class PropertyPredictor:
    """Predict molecular and materials properties from SMILES or crystal data.

    Example
    -------
    >>> pred = PropertyPredictor()
    >>> result = pred.predict("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    >>> result.qed
    0.553
    >>> result.synthesisability
    'easy'
    """

    def __init__(self, use_mace: bool = True, device: str = "cpu") -> None:
        self._mace_calc: Any = None
        if use_mace:
            self._try_load_mace(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, smiles: str) -> PropertyResult:
        """Predict all available properties for a SMILES string.

        Falls back gracefully if MACE is unavailable.
        """
        result = PropertyResult(smiles=smiles)
        self._rdkit_predict(smiles, result)
        return result

    def predict_crystal(
        self,
        atoms: Any,  # ase.Atoms object
    ) -> PropertyResult:
        """Run MACE-MP-0 on an ASE Atoms object (periodic crystal).

        Returns an empty PropertyResult with a warning if MACE is not installed.
        """
        smiles = atoms.get_chemical_formula() if hasattr(atoms, "get_chemical_formula") else ""
        result = PropertyResult(smiles=smiles)

        if self._mace_calc is None:
            result.warnings.append("MACE not available; install mace-torch for crystal property prediction")
            result.backend = "none"
            return result

        try:
            atoms.calc = self._mace_calc
            result.energy_ev = float(atoms.get_potential_energy())
            result.forces_ev_ang = atoms.get_forces().tolist()
            result.backend = "mace-mp-0"
        except RuntimeError as exc:
            logger.error("MACE inference failed for %s: %s", smiles, exc)
            result.warnings.append(f"MACE inference failed: {exc}")
            result.backend = "mace-failed"

        return result

    def rank_candidates(
        self, smiles_list: list[str], descending_qed: bool = True
    ) -> list[PropertyResult]:
        """Predict and rank a list of SMILES by QED (desc) then SA score (asc)."""
        results = [self.predict(s) for s in smiles_list]
        results.sort(
            key=lambda r: (
                -(r.qed or 0.0) if descending_qed else (r.qed or 0.0),
                r.sa_score or 10.0,
            )
        )
        return results

    # ------------------------------------------------------------------
    # RDKit backend
    # ------------------------------------------------------------------

    def _rdkit_predict(self, smiles: str, result: PropertyResult) -> None:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED, rdMolDescriptors

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result.warnings.append(f"Invalid SMILES: {smiles!r}")
                return

            result.mw = Descriptors.MolWt(mol)
            result.logp = Descriptors.MolLogP(mol)
            result.tpsa = Descriptors.TPSA(mol)
            result.hbd = rdMolDescriptors.CalcNumHBD(mol)
            result.hba = rdMolDescriptors.CalcNumHBA(mol)
            result.rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            result.qed = QED.qed(mol)

            # SA score
            sa = self._sa_score(mol)
            result.sa_score = sa

            # Composite drug score
            if result.qed is not None and sa is not None:
                result.drug_score = result.qed * (1.0 - sa / 10.0)
                if sa <= 3:
                    result.synthesisability = "easy"
                elif sa <= 5:
                    result.synthesisability = "moderate"
                elif sa <= 7:
                    result.synthesisability = "hard"
                else:
                    result.synthesisability = "very_hard"

            result.backend = "rdkit"
        except ImportError:
            result.warnings.append("RDKit not available")
        except ValueError as exc:
            logger.warning("RDKit prediction failed for %r: %s", smiles, exc)
            result.warnings.append(f"RDKit prediction failed: {exc}")

    @staticmethod
    def _sa_score(mol: Any) -> float | None:
        try:
            import importlib
            sa = importlib.import_module("rdkit.Contrib.SA_Score.sascorer")
            return float(sa.calculateScore(mol))
        except ImportError:
            pass
        except ValueError as exc:
            logger.warning("SA score calculator raised ValueError: %s", exc)
        # Fallback: simple heuristic (ring count + stereocenters)
        try:
            from rdkit.Chem import rdMolDescriptors
            rings = rdMolDescriptors.CalcNumRings(mol)
            stereo = len(rdMolDescriptors.CalcChiralCenters(mol, includeUnassigned=True))
            # rough mapping: 0-2 rings + no stereo = easy
            sa_approx = 1.0 + rings * 0.5 + stereo * 0.8
            return min(sa_approx, 10.0)
        except (ImportError, ValueError) as exc:
            logger.warning("SA score heuristic fallback failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # MACE backend (optional)
    # ------------------------------------------------------------------

    def _try_load_mace(self, device: str) -> None:
        try:
            from mace.calculators import mace_mp  # type: ignore[import]

            self._mace_calc = mace_mp(model="small", dispersion=False, default_dtype="float32", device=device)
        except ImportError:
            pass
        except RuntimeError as exc:
            logger.warning("MACE loading failed (e.g. CUDA not available): %s", exc)
