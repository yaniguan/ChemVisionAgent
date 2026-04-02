"""Skill: full scientific pipeline — retrieval → encode → predict → optimise.

This skill chains all new scientific modules into a single composable unit:

  1. Extract SMILES from the image (delegates to MolecularStructureSkill).
  2. Fetch PubChem ground-truth properties (retrieval grounding).
  3. Encode the molecule as a Morgan fingerprint embedding.
  4. Predict physicochemical / drug-likeness properties (RDKit + optional MACE).
  5. Run Pareto MCTS to propose optimised analogues.
  6. Return a unified PropertyPredictionResult.

The skill can also be called with an explicit ``smiles`` kwarg to skip step 1
(useful for testing or when SMILES is already known).
"""

from __future__ import annotations

from typing import Any

from PIL.Image import Image
from pydantic import Field

from chemvision.generation.pareto_mcts import Objective, ParetoMCTS
from chemvision.generation.property_predictor import PropertyPredictor, PropertyResult
from chemvision.models.mol_encoder import MolecularEncoder
from chemvision.retrieval.pubchem_client import PubChemClient
from chemvision.retrieval.vector_store import MoleculeVectorStore
from chemvision.skills.base import BaseSkill, SkillResult


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class OptimisedCandidate(SkillResult):
    """One Pareto-optimal molecule from the MCTS optimisation."""

    skill_name: str = "property_prediction"
    raw_output: str = ""

    smiles: str = ""
    qed: float | None = None
    mw: float | None = None
    logp: float | None = None
    sa_score: float | None = None
    drug_score: float | None = None
    synthesisability: str = "unknown"
    pareto_rank: int = 0
    scores: dict[str, float] = Field(default_factory=dict)


class PropertyPredictionResult(SkillResult):
    """Full output of the ``property_prediction`` skill."""

    # Input molecule
    input_smiles: str = ""

    # PubChem grounding
    pubchem_name: str | None = None
    pubchem_formula: str | None = None
    pubchem_mw: float | None = None
    pubchem_logp: float | None = None

    # Predicted properties
    predicted: PropertyResult | None = None

    # Similar molecules from the vector store
    similar_molecules: list[dict[str, Any]] = Field(default_factory=list)

    # Pareto-optimal optimised candidates
    pareto_candidates: list[OptimisedCandidate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------


class PropertyPredictionSkill(BaseSkill):
    """Chain retrieval + encoding + prediction + Pareto MCTS in one skill call.

    Parameters (kwargs)
    -------------------
    smiles : str, optional
        If provided, skip the VLM SMILES extraction step.
    n_mcts_iterations : int, default 80
        Number of MCTS iterations for the optimisation step.
    run_optimisation : bool, default True
        Whether to run Pareto MCTS (can be disabled for speed).
    """

    name = "property_prediction"

    # Shared instances — initialised once per process
    _encoder: MolecularEncoder | None = None
    _predictor: PropertyPredictor | None = None
    _pubchem: PubChemClient | None = None
    _store: MoleculeVectorStore | None = None

    @classmethod
    def _get_encoder(cls) -> MolecularEncoder:
        if cls._encoder is None:
            cls._encoder = MolecularEncoder()
        return cls._encoder

    @classmethod
    def _get_predictor(cls) -> PropertyPredictor:
        if cls._predictor is None:
            cls._predictor = PropertyPredictor(use_mace=False)
        return cls._predictor

    @classmethod
    def _get_pubchem(cls) -> PubChemClient:
        if cls._pubchem is None:
            cls._pubchem = PubChemClient()
        return cls._pubchem

    @classmethod
    def _get_store(cls) -> MoleculeVectorStore:
        if cls._store is None:
            cls._store = MoleculeVectorStore()
        return cls._store

    # ------------------------------------------------------------------

    def build_prompt(self, **kwargs: Any) -> str:
        return (
            "Extract the SMILES string for every molecule visible in this image. "
            "Output a valid JSON object with a single key 'smiles' whose value is "
            "a list of SMILES strings. If no molecule is visible, return {\"smiles\": []}."
        )

    def __call__(self, image: Image, model: Any, **kwargs: Any) -> PropertyPredictionResult:
        """Run the full scientific pipeline.

        Parameters
        ----------
        image:
            PIL image (may be a molecular diagram or any scientific image).
        model:
            Loaded vision model — used only if *smiles* kwarg is absent.
        smiles:
            Override SMILES (skip VLM extraction).
        n_mcts_iterations:
            MCTS budget (default 80).
        run_optimisation:
            Set False to skip MCTS (faster).
        """
        from chemvision.skills._parse import extract_json

        smiles: str | None = kwargs.get("smiles")
        n_iter: int = int(kwargs.get("n_mcts_iterations", 80))
        run_opt: bool = bool(kwargs.get("run_optimisation", True))

        # --- Step 1: Extract SMILES from image (if not provided) -------
        if not smiles:
            raw = model.generate(image, self.build_prompt())
            data = extract_json(raw) or {}
            candidates = data.get("smiles", [])
            smiles = candidates[0] if candidates else ""

        result = PropertyPredictionResult(
            skill_name=self.name,
            raw_output=smiles or "",
            input_smiles=smiles or "",
        )

        if not smiles:
            result.confidence = 0.0
            return result

        # --- Step 2: PubChem retrieval grounding -----------------------
        pubchem_data = self._get_pubchem().fetch_by_smiles(smiles)
        if pubchem_data:
            result.pubchem_name = pubchem_data.get("IUPACName") or pubchem_data.get("canonical_smiles")
            result.pubchem_formula = pubchem_data.get("MolecularFormula")
            result.pubchem_mw = _safe_float(pubchem_data.get("MolecularWeight"))
            result.pubchem_logp = _safe_float(pubchem_data.get("XLogP"))

        # --- Step 3: Encode + vector store -----------------------------
        encoder = self._get_encoder()
        embedding = encoder.encode(smiles)
        store = self._get_store()

        # Add this molecule so future queries can retrieve it
        store.add(smiles, embedding, {"smiles": smiles, "name": result.pubchem_name or ""})
        result.similar_molecules = store.search(embedding, k=5)

        # --- Step 4: Property prediction -------------------------------
        predictor = self._get_predictor()
        pred = predictor.predict(smiles)
        result.predicted = pred
        result.confidence = pred.qed  # use QED as proxy for overall confidence

        # --- Step 5: Pareto MCTS optimisation --------------------------
        if run_opt:
            objectives = [
                Objective("qed",    fn=lambda s: predictor.predict(s).qed or 0.0,    direction="max"),
                Objective("mw_inv", fn=lambda s: 1.0 / max(predictor.predict(s).mw or 500, 1), direction="max"),
                Objective("logp",   fn=lambda s: _clamp_logp(predictor.predict(s).logp), direction="max"),
            ]
            mcts = ParetoMCTS(objectives, seed=42)
            front = mcts.search(smiles, n_iterations=n_iter)
            result.pareto_candidates = [
                OptimisedCandidate(
                    smiles=c.smiles,
                    qed=predictor.predict(c.smiles).qed,
                    mw=predictor.predict(c.smiles).mw,
                    logp=predictor.predict(c.smiles).logp,
                    sa_score=predictor.predict(c.smiles).sa_score,
                    drug_score=predictor.predict(c.smiles).drug_score,
                    synthesisability=predictor.predict(c.smiles).synthesisability,
                    pareto_rank=c.pareto_rank,
                    scores=c.scores,
                )
                for c in front[:10]
            ]

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _clamp_logp(logp: float | None) -> float:
    """Score LogP: penalise values outside the ideal 1–3 window."""
    if logp is None:
        return -1.0
    return -abs(logp - 2.0)   # peak score at LogP=2
