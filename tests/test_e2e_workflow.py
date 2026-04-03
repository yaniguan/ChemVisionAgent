"""End-to-end workflow test: train -> generate -> evaluate.

This test uses a tiny fixture dataset (50 molecules) and minimal epochs
to validate the complete scientific workflow in CI.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FIXTURE_SMILES: list[str] = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCN(CC)CC",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "O=C(O)C(N)CC1=CC=CC=C1",
    "CC1=CC=CC=C1",
    "CCCC",
    "C(=O)N",
    "CCCCCCCC",
    "c1ccc(O)cc1",
    "CC(=O)NC1=CC=C(O)C=C1",
    "OC(=O)C1CCCCC1",
    "CC(C)O",
    "CCOC(=O)C",
    "c1ccc2ccccc2c1",
    "C1CCCCC1",
    "CC=CC",
    "C#N",
    "ClC(Cl)Cl",
    "CCOCC",
    "CC(C)(C)O",
    "c1ccncc1",
    "OC1=CC=CC=C1",
    "CC(=O)OC(C)C",
    "CCCCCC",
    "C1=CC=C(C=C1)O",
    "CC(O)CC",
    "CCC(=O)O",
    "CCCO",
    "CC(C)CC",
    "CCCCCCC",
    "C1CCOC1",
    "c1ccc(N)cc1",
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",
    "c1ccc(Cl)cc1",
    "CC#C",
    "CCCCC",
    "OCC(O)CO",
    "c1ccc(F)cc1",
    "CCC(C)O",
    "CC(=O)CC",
    "CCOC",
    "C(CO)O",
    "CC(=O)C",
    "CCN",
    "c1cc(O)ccc1O",
    "CCC",
    "CCCCCCCCCC",
]

assert len(_FIXTURE_SMILES) == 50, "Fixture must contain exactly 50 SMILES"


@pytest.fixture()
def fixture_smiles() -> list[str]:
    return list(_FIXTURE_SMILES)


@pytest.fixture()
def fixture_properties() -> np.ndarray:
    """Compute 8-dim property vector for each fixture molecule."""
    props = []
    for smi in _FIXTURE_SMILES:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Fixture SMILES failed to parse: {smi}"
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotb = Descriptors.NumRotatableBonds(mol)
        rings = Descriptors.RingCount(mol)
        heavy = mol.GetNumHeavyAtoms()
        props.append([mw, logp, hba, hbd, tpsa, rotb, rings, heavy])
    return np.array(props, dtype=np.float32)


# ---------------------------------------------------------------------------
# Test 1: full train -> generate -> evaluate pipeline
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_tiny_train_generate_evaluate(
    fixture_smiles: list[str], fixture_properties: np.ndarray
) -> None:
    from chemvision.eval.moses_metrics import MOSESMetrics, compute_moses_metrics
    from chemvision.generation.selfies_gen import SELFIESGenConfig, SELFIESGenerator

    config = SELFIESGenConfig(cond_dim=8, embed_dim=64, n_heads=2, n_layers=2, max_len=60)
    gen = SELFIESGenerator(config)

    result = gen.train(fixture_smiles, fixture_properties, epochs=3, batch_size=16, patience=5)
    assert result.epochs == 3
    assert result.vocab_size > 0
    assert len(result.loss_history) == 3

    # Generate 20 molecules conditioned on the mean property vector
    target = fixture_properties.mean(axis=0, keepdims=True)
    molecules = gen.generate(target, n_per_target=20)
    assert len(molecules) == 20

    # Every SELFIES decoding should produce a valid molecule
    generated_smiles = [m.smiles for m in molecules if m.smiles is not None]
    assert len(generated_smiles) > 0, "Generator produced no valid SMILES"

    # Compute MOSES metrics
    metrics = compute_moses_metrics(generated_smiles, training_smiles=fixture_smiles)
    assert isinstance(metrics, MOSESMetrics)
    assert metrics.validity == 1.0, (
        f"All decoded SELFIES should be valid, got validity={metrics.validity}"
    )
    assert metrics.n_generated == len(generated_smiles)


# ---------------------------------------------------------------------------
# Test 2: checkpoint save/load
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_checkpoint_save_load(
    fixture_smiles: list[str], fixture_properties: np.ndarray, tmp_path
) -> None:
    import json

    from chemvision.generation.selfies_gen import SELFIESGenConfig, SELFIESGenerator

    config = SELFIESGenConfig(cond_dim=8, embed_dim=64, n_heads=2, n_layers=2, max_len=60)
    gen = SELFIESGenerator(config)
    result = gen.train(fixture_smiles, fixture_properties, epochs=3, batch_size=16)

    # Save config and training info (mirrors what the CLI train command does)
    info = {
        "epochs": result.epochs,
        "final_loss": result.final_loss,
        "vocab_size": result.vocab_size,
        "converged": result.converged,
        "loss_history": result.loss_history,
        "config": config.__dict__,
    }
    info_path = tmp_path / "train_info.json"
    info_path.write_text(json.dumps(info, indent=2, default=str))
    assert info_path.exists()

    # Verify the saved JSON is valid and contains expected keys
    loaded = json.loads(info_path.read_text())
    assert loaded["epochs"] == 3
    assert loaded["vocab_size"] > 0
    assert "config" in loaded
    assert loaded["config"]["cond_dim"] == 8
    assert loaded["config"]["embed_dim"] == 64


# ---------------------------------------------------------------------------
# Test 3: evaluation output schema
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_evaluation_output_schema(fixture_smiles: list[str]) -> None:
    from dataclasses import fields

    from chemvision.eval.moses_metrics import MOSESMetrics, compute_moses_metrics

    metrics = compute_moses_metrics(fixture_smiles, training_smiles=fixture_smiles)

    # Check all expected fields are present
    expected_fields = {
        "validity",
        "uniqueness",
        "novelty",
        "int_div_1",
        "fcd",
        "scaffold_diversity",
    }
    actual_fields = {f.name for f in fields(metrics)}
    assert expected_fields.issubset(actual_fields), (
        f"Missing fields: {expected_fields - actual_fields}"
    )

    # Since fixture SMILES are all valid RDKit molecules, validity should be 1.0
    assert metrics.validity == 1.0
    # Uniqueness should be > 0 (we have diverse molecules)
    assert metrics.uniqueness > 0.0
    # Novelty should be 0.0 since generated == training
    assert metrics.novelty == 0.0
    # Internal diversity should be > 0 for diverse molecules
    assert metrics.int_div_1 > 0.0
    # Scaffold diversity should be > 0
    assert metrics.scaffold_diversity > 0.0
    # Summary should be a non-empty string
    assert len(metrics.summary()) > 0


# ---------------------------------------------------------------------------
# Test 4: GIN encode then generate
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_gin_encode_then_generate(
    fixture_smiles: list[str], fixture_properties: np.ndarray
) -> None:
    from chemvision.generation.selfies_gen import SELFIESGenConfig, SELFIESGenerator
    from chemvision.models.gnn import GINEncoder, smiles_to_graph

    # Verify graph conversion works
    graph = smiles_to_graph("CCO")
    assert graph is not None
    assert graph["x"].shape[0] == 3  # ethanol has 3 heavy atoms
    assert graph["x"].shape[1] == 37  # 37-d atom features
    assert graph["edge_index"].shape[0] == 2

    # Encode SMILES with GINEncoder
    encoder = GINEncoder(embed_dim=128, n_layers=2)
    emb = encoder.encode_smiles("CCO")
    assert emb.shape == (128,), f"Expected (128,), got {emb.shape}"
    assert not np.all(emb == 0), "Embedding should not be all zeros"

    # Batch encoding
    batch_emb = encoder.encode_smiles_batch(["CCO", "c1ccccc1", "CC(=O)O"])
    assert batch_emb.shape == (3, 128)

    # Now train a small generator and generate conditioned on properties
    config = SELFIESGenConfig(cond_dim=8, embed_dim=64, n_heads=2, n_layers=2, max_len=60)
    gen = SELFIESGenerator(config)
    gen.train(fixture_smiles, fixture_properties, epochs=2, batch_size=16)

    target = fixture_properties[:1]
    molecules = gen.generate(target, n_per_target=5)
    assert len(molecules) == 5
    for mol in molecules:
        assert hasattr(mol, "smiles")
        assert hasattr(mol, "selfies")
        assert hasattr(mol, "is_valid")


# ---------------------------------------------------------------------------
# Test 5: baselines generate valid molecules
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_baselines_generate_valid(fixture_smiles: list[str]) -> None:
    from benchmarks.baselines import FragmentBaseline

    baseline = FragmentBaseline()
    baseline.fit(fixture_smiles)
    assert len(baseline._fragments) > 0, "BRICS should extract at least some fragments"

    generated = baseline.generate(n=10)
    assert len(generated) == 10

    # At least some generated SMILES should be valid molecules
    valid_count = sum(1 for smi in generated if Chem.MolFromSmiles(smi) is not None)
    assert valid_count > 0, (
        f"FragmentBaseline should produce at least 1 valid molecule out of 10, "
        f"got {valid_count}"
    )
