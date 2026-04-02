"""End-to-end tests for the new scientific pipeline modules.

Coverage
--------
Layer 1 — Molecular Encoder (RDKit)
Layer 2 — PubChem Retrieval  (live HTTP, skipped if offline)
Layer 3 — Vector Store        (numpy cosine similarity)
Layer 4 — Physics: Symmetry   (spglib)
Layer 5 — Physics: Scherrer   (pure maths)
Layer 6 — Property Predictor  (RDKit)
Layer 7 — Pareto MCTS         (multi-objective optimisation)
Layer 8 — PropertyPredictionSkill (full chain, no VLM required)
Layer 9 — Skill registry integration

All tests run on CPU, require NO API keys, and complete in < 60 s.
"""

from __future__ import annotations

import math
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Molecules used throughout the tests
# ---------------------------------------------------------------------------
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
INVALID_SMILES = "this_is_not_a_smiles_$$"

# FCC Cu crystal
CU_LATTICE = [[3.615, 0, 0], [0, 3.615, 0], [0, 0, 3.615]]
CU_POSITIONS = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
CU_NUMBERS = [29, 29, 29, 29]  # Cu = 29


# ===========================================================================
# Layer 1: Molecular Encoder
# ===========================================================================

class TestMolecularEncoder:

    def setup_method(self) -> None:
        from chemvision.models.mol_encoder import MolecularEncoder
        self.enc = MolecularEncoder()

    def test_encode_returns_correct_shape(self) -> None:
        emb = self.enc.encode(ASPIRIN)
        assert emb.shape == (2048,), f"Expected (2048,), got {emb.shape}"
        assert emb.dtype == np.float32

    def test_encode_invalid_smiles_returns_zeros(self) -> None:
        emb = self.enc.encode(INVALID_SMILES)
        assert np.all(emb == 0.0), "Invalid SMILES should produce zero vector"

    def test_different_molecules_have_different_embeddings(self) -> None:
        e1 = self.enc.encode(ASPIRIN)
        e2 = self.enc.encode(CAFFEINE)
        assert not np.allclose(e1, e2), "Distinct molecules must have distinct fingerprints"

    def test_tanimoto_self_similarity(self) -> None:
        sim = self.enc.tanimoto(ASPIRIN, ASPIRIN)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_tanimoto_different_molecules(self) -> None:
        sim = self.enc.tanimoto(ASPIRIN, CAFFEINE)
        assert 0.0 <= sim < 1.0

    def test_conformer_generation(self) -> None:
        result = self.enc.generate_conformer(ASPIRIN)
        assert result.success, "ETKDG conformer generation failed for aspirin"
        assert result.num_atoms > 0
        assert len(result.coordinates) == result.num_atoms
        # Check coordinates are physically reasonable (< 50 Å from origin)
        coords = np.array(result.coordinates)
        assert np.all(np.abs(coords) < 50.0)

    def test_conformer_invalid_smiles(self) -> None:
        result = self.enc.generate_conformer(INVALID_SMILES)
        assert not result.success

    def test_descriptors_aspirin(self) -> None:
        desc = self.enc.compute_descriptors(ASPIRIN)
        assert desc.mw is not None
        assert 175 < desc.mw < 185, f"Aspirin MW should be ~180, got {desc.mw}"
        assert desc.logp is not None
        assert desc.lipinski_pass is True

    def test_descriptors_invalid_smiles(self) -> None:
        desc = self.enc.compute_descriptors(INVALID_SMILES)
        assert desc.mw is None

    def test_encode_batch(self) -> None:
        smiles_list = [ASPIRIN, IBUPROFEN, CAFFEINE, PARACETAMOL]
        matrix = self.enc.encode_batch(smiles_list)
        assert matrix.shape == (4, 2048)


# ===========================================================================
# Layer 2: PubChem Retrieval (skip if offline / rate-limited)
# ===========================================================================

@pytest.mark.network
class TestPubChemClient:
    """Marked 'network' so CI can skip with -m 'not network'."""

    def setup_method(self) -> None:
        from chemvision.retrieval.pubchem_client import PubChemClient
        self.client = PubChemClient(timeout=10)

    def test_fetch_by_name_aspirin(self) -> None:
        props = self.client.fetch_by_name("aspirin")
        if not props:
            pytest.skip("PubChem unreachable")
        assert "MolecularFormula" in props
        assert props["MolecularFormula"] == "C9H8O4"

    def test_fetch_by_smiles_aspirin(self) -> None:
        props = self.client.fetch_by_smiles(ASPIRIN)
        if not props:
            pytest.skip("PubChem unreachable")
        assert "MolecularWeight" in props
        mw = float(props["MolecularWeight"])
        assert 179 < mw < 182

    def test_fetch_invalid_smiles_returns_empty(self) -> None:
        props = self.client.fetch_by_smiles(INVALID_SMILES)
        assert props == {}, "Invalid SMILES should return empty dict, not raise"

    def test_similar_compounds_returns_list(self) -> None:
        results = self.client.get_similar_compounds(ASPIRIN, threshold=90, max_results=3)
        if not results:
            pytest.skip("PubChem unreachable or no results")
        assert isinstance(results, list)
        assert all("MolecularFormula" in r for r in results)


# ===========================================================================
# Layer 3: Vector Store
# ===========================================================================

class TestMoleculeVectorStore:

    def setup_method(self) -> None:
        from chemvision.models.mol_encoder import MolecularEncoder
        from chemvision.retrieval.vector_store import MoleculeVectorStore
        self.store = MoleculeVectorStore()
        self.enc = MolecularEncoder()

    def _add(self, smiles: str, name: str) -> None:
        emb = self.enc.encode(smiles)
        self.store.add(name, emb, {"smiles": smiles})

    def test_add_and_len(self) -> None:
        assert len(self.store) == 0
        self._add(ASPIRIN, "aspirin")
        assert len(self.store) == 1

    def test_search_returns_self_as_top_hit(self) -> None:
        for smi, name in [(ASPIRIN, "aspirin"), (IBUPROFEN, "ibuprofen"), (CAFFEINE, "caffeine")]:
            self._add(smi, name)
        query = self.enc.encode(ASPIRIN)
        hits = self.store.search(query, k=3)
        assert hits[0]["name"] == "aspirin"
        assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_search_orders_by_similarity(self) -> None:
        for smi, name in [(ASPIRIN, "aspirin"), (IBUPROFEN, "ibuprofen"), (CAFFEINE, "caffeine")]:
            self._add(smi, name)
        query = self.enc.encode(IBUPROFEN)
        hits = self.store.search(query, k=3)
        assert hits[0]["name"] == "ibuprofen"

    def test_search_empty_store(self) -> None:
        query = self.enc.encode(ASPIRIN)
        hits = self.store.search(query, k=3)
        assert hits == []

    def test_save_and_load(self, tmp_path: Any) -> None:
        self._add(ASPIRIN, "aspirin")
        self._add(CAFFEINE, "caffeine")
        save_path = tmp_path / "store"
        self.store.save(str(save_path))

        from chemvision.retrieval.vector_store import MoleculeVectorStore
        store2 = MoleculeVectorStore()
        store2.load(str(save_path))
        assert len(store2) == 2
        query = self.enc.encode(ASPIRIN)
        hits = store2.search(query, k=1)
        assert hits[0]["name"] == "aspirin"


# ===========================================================================
# Layer 4: Crystal Symmetry (spglib)
# ===========================================================================

class TestCrystalSymmetryAnalyzer:

    def setup_method(self) -> None:
        from chemvision.physics.symmetry import CrystalSymmetryAnalyzer
        self.analyzer = CrystalSymmetryAnalyzer()

    def test_fcc_cu_space_group(self) -> None:
        result = self.analyzer.analyze(CU_LATTICE, CU_POSITIONS, CU_NUMBERS)
        assert result.is_valid, "spglib failed for FCC Cu"
        assert result.space_group_number == 225, f"FCC Cu should be Fm-3m (#225), got #{result.space_group_number}"
        assert result.crystal_system == "cubic"

    def test_fcc_cu_point_group(self) -> None:
        result = self.analyzer.analyze(CU_LATTICE, CU_POSITIONS, CU_NUMBERS)
        assert "m" in result.point_group.lower() or "3" in result.point_group

    def test_from_lattice_params_cubic(self) -> None:
        # Simple cubic Si-like, a=b=c, all angles 90°
        result = self.analyzer.from_lattice_params(
            a=5.43, b=5.43, c=5.43,
            alpha=90, beta=90, gamma=90,
            species=[14],
            fractional_positions=[[0.0, 0.0, 0.0]],
        )
        assert result.is_valid

    def test_invalid_structure_returns_invalid(self) -> None:
        # Pass nonsensical data
        result = self.analyzer.analyze([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0]], [1])
        assert not result.is_valid

    def test_summary_string(self) -> None:
        result = self.analyzer.analyze(CU_LATTICE, CU_POSITIONS, CU_NUMBERS)
        summary = result.summary
        assert "cubic" in summary.lower()
        assert "225" in summary


# ===========================================================================
# Layer 5: Scherrer grain-size
# ===========================================================================

class TestScherrerAnalyzer:

    def setup_method(self) -> None:
        from chemvision.physics.scherrer import ScherrerAnalyzer
        self.analyzer = ScherrerAnalyzer()

    def test_single_peak(self) -> None:
        # TiO2 anatase (101) at 2θ=25.3°, FWHM=0.35°
        results = self.analyzer.analyze_peaks([(25.3, 0.35)])
        assert len(results) == 1
        r = results[0]
        assert r.valid
        # D should be roughly 24 nm for these parameters
        assert 20 < r.grain_size_nm < 30, f"Expected ~24 nm, got {r.grain_size_nm:.1f} nm"

    def test_fwhm_zero_is_invalid(self) -> None:
        results = self.analyzer.analyze_peaks([(25.3, 0.0)])
        assert not results[0].valid

    def test_mean_grain_size(self) -> None:
        peaks = [(25.3, 0.35), (48.0, 0.42), (53.8, 0.45)]
        mean = self.analyzer.mean_grain_size_nm(peaks)
        assert mean is not None
        assert 10 < mean < 40

    def test_scherrer_equation_correctness(self) -> None:
        """Manual calculation for Cu Kα, 2θ=38.5°, FWHM=0.5°."""
        K, lam = 0.9, 1.5406
        theta = math.radians(38.5 / 2)
        beta = math.radians(0.5)
        expected_ang = K * lam / (beta * math.cos(theta))
        expected_nm = expected_ang / 10.0

        results = self.analyzer.analyze_peaks([(38.5, 0.5)])
        assert results[0].valid
        assert results[0].grain_size_nm == pytest.approx(expected_nm, rel=1e-4)


# ===========================================================================
# Layer 6: Property Predictor (RDKit)
# ===========================================================================

class TestPropertyPredictor:

    def setup_method(self) -> None:
        from chemvision.generation.property_predictor import PropertyPredictor
        self.pred = PropertyPredictor(use_mace=False)

    def test_predict_aspirin(self) -> None:
        result = self.pred.predict(ASPIRIN)
        assert result.mw is not None
        assert result.qed is not None
        assert 0.0 < result.qed < 1.0
        assert result.logp is not None
        assert result.backend == "rdkit"

    def test_predict_invalid_smiles(self) -> None:
        result = self.pred.predict(INVALID_SMILES)
        assert result.mw is None
        assert len(result.warnings) > 0

    def test_predict_caffeine_lipinski(self) -> None:
        result = self.pred.predict(CAFFEINE)
        # Caffeine: MW=194, HBD=0, HBA=3, LogP=-0.07 → should pass Lipinski
        from chemvision.models.mol_encoder import MolecularEncoder
        desc = MolecularEncoder().compute_descriptors(CAFFEINE)
        assert desc.lipinski_pass is True

    def test_synthesisability_label(self) -> None:
        result = self.pred.predict(ASPIRIN)
        assert result.synthesisability in ("easy", "moderate", "hard", "very_hard")

    def test_drug_score_range(self) -> None:
        result = self.pred.predict(ASPIRIN)
        if result.drug_score is not None:
            assert 0.0 <= result.drug_score <= 1.0

    def test_rank_candidates(self) -> None:
        smiles_list = [ASPIRIN, CAFFEINE, IBUPROFEN, PARACETAMOL]
        ranked = self.pred.rank_candidates(smiles_list)
        assert len(ranked) == 4
        # First result should have highest (or equal) QED
        assert all(
            (ranked[0].qed or 0) >= (ranked[i].qed or 0)
            for i in range(1, len(ranked))
        )


# ===========================================================================
# Layer 7: Pareto MCTS
# ===========================================================================

class TestParetoMCTS:

    def setup_method(self) -> None:
        from chemvision.generation.property_predictor import PropertyPredictor
        from chemvision.generation.pareto_mcts import ParetoMCTS, Objective
        pred = PropertyPredictor(use_mace=False)
        self.objectives = [
            Objective("qed",    fn=lambda s: pred.predict(s).qed or 0.0, direction="max"),
            Objective("mw_inv", fn=lambda s: 1.0 / max(pred.predict(s).mw or 500, 1), direction="max"),
        ]
        self.mcts = ParetoMCTS(self.objectives, seed=0)

    def test_search_returns_candidates(self) -> None:
        front = self.mcts.search(ASPIRIN, n_iterations=20)
        assert len(front) > 0

    def test_pareto_front_non_dominated(self) -> None:
        front = self.mcts.search(ASPIRIN, n_iterations=20)
        # All candidates on the Pareto front should have rank 0
        assert all(c.pareto_rank == 0 for c in front)

    def test_candidates_are_valid_smiles(self) -> None:
        from rdkit import Chem
        front = self.mcts.search(PARACETAMOL, n_iterations=20)
        for c in front:
            mol = Chem.MolFromSmiles(c.smiles)
            assert mol is not None, f"Invalid SMILES on Pareto front: {c.smiles!r}"

    def test_scores_match_objectives(self) -> None:
        front = self.mcts.search(ASPIRIN, n_iterations=15)
        for c in front:
            assert "qed" in c.scores
            assert "mw_inv" in c.scores

    def test_search_is_deterministic(self) -> None:
        """Same seed → same Pareto front."""
        from chemvision.generation.property_predictor import PropertyPredictor
        from chemvision.generation.pareto_mcts import ParetoMCTS, Objective
        pred = PropertyPredictor(use_mace=False)
        objs = [Objective("qed", fn=lambda s: pred.predict(s).qed or 0, direction="max")]
        m1 = ParetoMCTS(objs, seed=99).search(ASPIRIN, n_iterations=10)
        m2 = ParetoMCTS(objs, seed=99).search(ASPIRIN, n_iterations=10)
        assert [c.smiles for c in m1] == [c.smiles for c in m2]


# ===========================================================================
# Layer 8: PropertyPredictionSkill (full pipeline, mocked VLM)
# ===========================================================================

class TestPropertyPredictionSkill:
    """Test the full PropertyPredictionSkill without a real VLM.

    The VLM model is mocked to return a fixed SMILES so the rest of the
    pipeline (retrieval, encoding, prediction, MCTS) runs for real.
    """

    def _make_mock_model(self, smiles: str = ASPIRIN) -> MagicMock:
        model = MagicMock()
        model.generate.return_value = f'{{"smiles": ["{smiles}"]}}'
        return model

    def _dummy_image(self) -> Any:
        from PIL import Image
        return Image.new("RGB", (64, 64), color=(200, 200, 200))

    def test_skill_name_in_registry(self) -> None:
        from chemvision.skills.skill_registry import DEFAULT_REGISTRY
        assert "property_prediction" in DEFAULT_REGISTRY

    def test_full_pipeline_with_smiles_kwarg(self) -> None:
        from chemvision.skills.property_prediction import PropertyPredictionSkill
        skill = PropertyPredictionSkill()
        image = self._dummy_image()
        model = self._make_mock_model()

        result = skill(image, model, smiles=ASPIRIN, run_optimisation=True, n_mcts_iterations=20)

        assert result.input_smiles == ASPIRIN
        assert result.predicted is not None
        assert result.predicted.mw is not None
        assert result.predicted.qed is not None
        assert len(result.pareto_candidates) > 0

    def test_full_pipeline_via_vlm(self) -> None:
        from chemvision.skills.property_prediction import PropertyPredictionSkill
        skill = PropertyPredictionSkill()
        image = self._dummy_image()
        model = self._make_mock_model(CAFFEINE)

        result = skill(image, model, run_optimisation=False)

        assert result.input_smiles == CAFFEINE
        assert result.predicted is not None
        assert result.predicted.backend == "rdkit"

    def test_no_smiles_extracted_returns_low_confidence(self) -> None:
        from chemvision.skills.property_prediction import PropertyPredictionSkill
        skill = PropertyPredictionSkill()
        image = self._dummy_image()
        model = MagicMock()
        model.generate.return_value = '{"smiles": []}'

        result = skill(image, model)
        assert result.confidence == 0.0

    def test_pareto_candidates_are_typed(self) -> None:
        from chemvision.skills.property_prediction import PropertyPredictionSkill, OptimisedCandidate
        skill = PropertyPredictionSkill()
        image = self._dummy_image()
        model = self._make_mock_model(IBUPROFEN)

        result = skill(image, model, smiles=IBUPROFEN, run_optimisation=True, n_mcts_iterations=15)
        for c in result.pareto_candidates:
            assert isinstance(c, OptimisedCandidate)
            assert c.smiles != ""
            assert c.pareto_rank == 0


# ===========================================================================
# Layer 9: Integration — full pipeline timing
# ===========================================================================

class TestEndToEndTiming:
    """Smoke test the full pipeline and report latencies."""

    def test_full_pipeline_completes_in_time(self) -> None:
        """All layers combined should complete in < 30 s on a laptop CPU."""
        from chemvision.models.mol_encoder import MolecularEncoder
        from chemvision.generation.property_predictor import PropertyPredictor
        from chemvision.generation.pareto_mcts import ParetoMCTS, Objective
        from chemvision.physics.symmetry import CrystalSymmetryAnalyzer
        from chemvision.physics.scherrer import ScherrerAnalyzer
        from chemvision.retrieval.vector_store import MoleculeVectorStore

        t0 = time.perf_counter()

        # --- Encode ---
        enc = MolecularEncoder()
        embeddings = enc.encode_batch([ASPIRIN, IBUPROFEN, CAFFEINE, PARACETAMOL])
        assert embeddings.shape == (4, 2048)

        # --- Vector store ---
        store = MoleculeVectorStore()
        for i, (s, name) in enumerate([(ASPIRIN, "asp"), (IBUPROFEN, "ibu"), (CAFFEINE, "caf")]):
            store.add(name, embeddings[i])
        hits = store.search(embeddings[0], k=3)
        assert hits[0]["name"] == "asp"

        # --- Property prediction ---
        pred = PropertyPredictor(use_mace=False)
        results = pred.rank_candidates([ASPIRIN, IBUPROFEN, CAFFEINE, PARACETAMOL])
        assert len(results) == 4

        # --- Pareto MCTS ---
        objectives = [
            Objective("qed", fn=lambda s: pred.predict(s).qed or 0, direction="max"),
            Objective("mw",  fn=lambda s: -(pred.predict(s).mw or 500), direction="max"),
        ]
        front = ParetoMCTS(objectives, seed=7).search(ASPIRIN, n_iterations=30)
        assert len(front) > 0

        # --- Crystal symmetry ---
        sym = CrystalSymmetryAnalyzer().analyze(CU_LATTICE, CU_POSITIONS, CU_NUMBERS)
        assert sym.space_group_number == 225

        # --- Scherrer ---
        grain = ScherrerAnalyzer().mean_grain_size_nm([(25.3, 0.35), (48.0, 0.42)])
        assert grain is not None

        elapsed = time.perf_counter() - t0
        print(f"\n  Full pipeline completed in {elapsed:.2f} s")
        assert elapsed < 30.0, f"Pipeline too slow: {elapsed:.1f} s > 30 s limit"
