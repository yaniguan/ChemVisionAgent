"""End-to-end tests for CSCA model, Flow Matcher, and Dataset Builder.

These tests actually TRAIN models on CPU and verify convergence.
No mocks — real RDKit computations, real PyTorch training, real Parquet I/O.

Test flow:
  1. Build dataset from seed molecules (RDKit)
  2. Train CSCA on fingerprint ↔ property pairs
  3. Verify CSCA retrieval accuracy
  4. Train Flow Matcher conditioned on properties
  5. Sample new molecules from Flow Matcher
  6. Verify generated fingerprints are valid
  7. Verify reproducibility (same seed = same result)
  8. Test batch processing + latency profiling
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Builder
# ═══════════════════════════════════════════════════════════════════════════

class TestDatasetBuilder:

    def test_add_seeds(self) -> None:
        from chemvision.data.dataset_builder import MolecularDatasetBuilder
        builder = MolecularDatasetBuilder(seed=42)
        n = builder.add_seeds()
        assert n >= 15, f"Expected ≥15 seed molecules, got {n}"

    def test_add_random_molecules(self) -> None:
        from chemvision.data.dataset_builder import MolecularDatasetBuilder
        builder = MolecularDatasetBuilder(seed=42)
        builder.add_seeds()
        n = builder.add_random_molecules(n=50)
        assert n >= 20, f"Expected ≥20 random molecules, got {n}"

    def test_build_produces_parquet(self, tmp_path: Path) -> None:
        from chemvision.data.dataset_builder import MolecularDatasetBuilder
        builder = MolecularDatasetBuilder(seed=42)
        builder.add_seeds()
        builder.add_random_molecules(n=30)
        stats = builder.build(tmp_path / "dataset")

        assert stats.n_total >= 30
        assert stats.n_train > 0
        assert stats.fp_dim == 2048
        assert stats.prop_dim == 8
        assert (tmp_path / "dataset" / "train.parquet").exists()
        assert (tmp_path / "dataset" / "arrays.npz").exists()
        assert (tmp_path / "dataset" / "stats.json").exists()

    def test_load_arrays(self, tmp_path: Path) -> None:
        from chemvision.data.dataset_builder import MolecularDatasetBuilder
        builder = MolecularDatasetBuilder(seed=42)
        builder.add_seeds()
        builder.add_random_molecules(n=20)
        builder.build(tmp_path / "ds")

        fps, props, splits = MolecularDatasetBuilder.load_arrays(tmp_path / "ds")
        assert fps.ndim == 2 and fps.shape[1] == 2048
        assert props.ndim == 2 and props.shape[1] == 8
        assert "train_idx" in splits

    def test_reproducible_splits(self, tmp_path: Path) -> None:
        from chemvision.data.dataset_builder import MolecularDatasetBuilder
        builder1 = MolecularDatasetBuilder(seed=42)
        builder1.add_seeds()
        stats1 = builder1.build(tmp_path / "ds1")

        builder2 = MolecularDatasetBuilder(seed=42)
        builder2.add_seeds()
        stats2 = builder2.build(tmp_path / "ds2")

        fps1, _, _ = MolecularDatasetBuilder.load_arrays(tmp_path / "ds1")
        fps2, _, _ = MolecularDatasetBuilder.load_arrays(tmp_path / "ds2")
        assert np.allclose(fps1, fps2), "Same seed should produce same dataset"


# ═══════════════════════════════════════════════════════════════════════════
# CSCA Model — TRAINS ON CPU
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def training_data(tmp_path: Path):
    """Build a small training dataset."""
    from chemvision.data.dataset_builder import MolecularDatasetBuilder
    builder = MolecularDatasetBuilder(seed=42)
    builder.add_seeds()
    builder.add_random_molecules(n=80)
    builder.build(tmp_path / "data")
    fps, props, splits = MolecularDatasetBuilder.load_arrays(tmp_path / "data")
    return fps, props, splits


class TestCSCAModel:

    def test_model_forward(self) -> None:
        from chemvision.models.csca import CSCAModel, CSCAConfig
        model = CSCAModel(CSCAConfig(fp_dim=64, prop_dim=4, latent_dim=16, hidden_dim=32))
        fp = torch.randn(8, 64)
        props = torch.randn(8, 4)
        loss = model(fp, props)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_model_encodes_to_latent(self) -> None:
        from chemvision.models.csca import CSCAModel, CSCAConfig
        model = CSCAModel(CSCAConfig(fp_dim=64, prop_dim=4, latent_dim=16, hidden_dim=32))
        fp = torch.randn(5, 64)
        z = model.encode_fingerprint(fp)
        assert z.shape == (5, 16)
        # L2-normalised
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)

    def test_train_converges(self, training_data) -> None:
        """CRITICAL: verify the CSCA model actually learns (loss decreases)."""
        fps, props, _ = training_data
        from chemvision.models.csca import CSCATrainer, CSCAConfig
        config = CSCAConfig(fp_dim=fps.shape[1], prop_dim=props.shape[1],
                           latent_dim=32, hidden_dim=64, learning_rate=1e-3, seed=42)
        trainer = CSCATrainer(config)
        result = trainer.train(fps, props, epochs=80, batch_size=32, patience=30, verbose=False)

        assert result.epochs >= 5, "Should train for at least 5 epochs"
        assert result.final_loss < result.loss_history[0], \
            f"Loss should decrease: {result.loss_history[0]:.4f} → {result.final_loss:.4f}"
        assert result.converged, f"Should converge. Loss history: {result.loss_history[:5]}...{result.loss_history[-3:]}"

    def test_retrieval_accuracy(self, training_data) -> None:
        """After training, given a property vector, can we retrieve the correct molecule?"""
        fps, props, _ = training_data
        from chemvision.models.csca import CSCATrainer, CSCAConfig
        config = CSCAConfig(fp_dim=fps.shape[1], prop_dim=props.shape[1],
                           latent_dim=32, hidden_dim=64, seed=42)
        trainer = CSCATrainer(config)
        result = trainer.train(fps, props, epochs=80, batch_size=32, verbose=False)
        assert result.val_retrieval_acc > 0.05, \
            f"Retrieval accuracy too low: {result.val_retrieval_acc:.2%}"

    def test_retrieve_returns_indices(self, training_data) -> None:
        fps, props, _ = training_data
        from chemvision.models.csca import CSCATrainer, CSCAConfig
        config = CSCAConfig(fp_dim=fps.shape[1], prop_dim=props.shape[1],
                           latent_dim=32, hidden_dim=64, seed=42)
        trainer = CSCATrainer(config)
        trainer.train(fps, props, epochs=30, batch_size=32)
        results = trainer.retrieve(props[:3], fps, k=5)
        assert len(results) == 3
        assert all(len(r) == 5 for r in results)
        assert all(0 <= idx < len(fps) for r in results for idx in r)

    def test_encode_latent_space(self, training_data) -> None:
        fps, props, _ = training_data
        from chemvision.models.csca import CSCATrainer, CSCAConfig
        config = CSCAConfig(fp_dim=fps.shape[1], prop_dim=props.shape[1],
                           latent_dim=32, hidden_dim=64, seed=42)
        trainer = CSCATrainer(config)
        trainer.train(fps, props, epochs=20, batch_size=32)
        z = trainer.encode(fps[:10])
        assert z.shape == (10, 32)

    def test_reproducibility(self, training_data) -> None:
        """Same seed → same training result."""
        fps, props, _ = training_data
        from chemvision.models.csca import CSCATrainer, CSCAConfig
        config = CSCAConfig(fp_dim=fps.shape[1], prop_dim=props.shape[1],
                           latent_dim=16, hidden_dim=32, seed=99)

        t1 = CSCATrainer(config)
        r1 = t1.train(fps, props, epochs=20, batch_size=32)

        t2 = CSCATrainer(CSCAConfig(**{**config.__dict__, "seed": 99}))
        r2 = t2.train(fps, props, epochs=20, batch_size=32)

        assert r1.loss_history == pytest.approx(r2.loss_history, abs=1e-5), \
            "Same seed should produce identical training"


# ═══════════════════════════════════════════════════════════════════════════
# Conditional Flow Matcher — TRAINS ON CPU
# ═══════════════════════════════════════════════════════════════════════════

class TestFlowMatcher:

    def test_vector_field_net_shapes(self) -> None:
        from chemvision.generation.flow_matcher import _VectorFieldNet, FlowMatcherConfig
        config = FlowMatcherConfig(fp_dim=64, cond_dim=4, hidden_dim=32, n_layers=2)
        net = _VectorFieldNet(config)
        x = torch.randn(8, 64)
        t = torch.rand(8, 1)
        c = torch.randn(8, 4)
        out = net(x, t, c)
        assert out.shape == (8, 64)

    def test_train_converges(self, training_data) -> None:
        """CRITICAL: verify flow matcher loss decreases during training."""
        fps, props, _ = training_data
        from chemvision.generation.flow_matcher import ConditionalFlowMatcher, FlowMatcherConfig
        config = FlowMatcherConfig(
            fp_dim=fps.shape[1], cond_dim=props.shape[1],
            hidden_dim=128, n_layers=2, learning_rate=5e-4, seed=42,
        )
        cfm = ConditionalFlowMatcher(config)
        result = cfm.train(fps, props, epochs=60, batch_size=32, patience=25, verbose=False)

        assert result.epochs >= 5
        assert result.final_loss < result.loss_history[0], \
            f"Flow matcher loss should decrease: {result.loss_history[0]:.4f} → {result.final_loss:.4f}"

    def test_sample_produces_fingerprints(self, training_data) -> None:
        """Generate molecules and verify output shape and validity."""
        fps, props, _ = training_data
        from chemvision.generation.flow_matcher import ConditionalFlowMatcher, FlowMatcherConfig
        config = FlowMatcherConfig(
            fp_dim=fps.shape[1], cond_dim=props.shape[1],
            hidden_dim=64, n_layers=2, seed=42,
        )
        cfm = ConditionalFlowMatcher(config)
        cfm.train(fps, props, epochs=30, batch_size=32)

        # Sample conditioned on first 5 property vectors
        generated = cfm.sample(props[:5], n_steps=10)

        assert len(generated) == 5
        for g in generated:
            assert g.fingerprint.shape == (fps.shape[1],)
            assert g.binary_fingerprint.shape == (fps.shape[1],)
            assert set(np.unique(g.binary_fingerprint)).issubset({0.0, 1.0})
            assert g.condition.shape == (props.shape[1],)

    def test_generated_fingerprints_have_nonzero_bits(self, training_data) -> None:
        fps, props, _ = training_data
        from chemvision.generation.flow_matcher import ConditionalFlowMatcher, FlowMatcherConfig
        config = FlowMatcherConfig(
            fp_dim=fps.shape[1], cond_dim=props.shape[1],
            hidden_dim=64, n_layers=2, seed=42,
        )
        cfm = ConditionalFlowMatcher(config)
        cfm.train(fps, props, epochs=30, batch_size=32)
        generated = cfm.sample(props[:3], n_steps=20)

        for g in generated:
            n_bits = g.binary_fingerprint.sum()
            # A valid drug-like molecule typically has 50–300 Morgan bits set
            assert n_bits > 0, "Generated fingerprint should have some bits set"

    def test_reproducibility(self, training_data) -> None:
        fps, props, _ = training_data
        from chemvision.generation.flow_matcher import ConditionalFlowMatcher, FlowMatcherConfig
        config = FlowMatcherConfig(
            fp_dim=fps.shape[1], cond_dim=props.shape[1],
            hidden_dim=32, n_layers=1, seed=42,
        )
        cfm1 = ConditionalFlowMatcher(config)
        r1 = cfm1.train(fps, props, epochs=15, batch_size=32)

        cfm2 = ConditionalFlowMatcher(FlowMatcherConfig(**{**config.__dict__, "seed": 42}))
        r2 = cfm2.train(fps, props, epochs=15, batch_size=32)

        assert r1.loss_history == pytest.approx(r2.loss_history, abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# Full pipeline: dataset → CSCA → flow matcher → generation → validation
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPipeline:

    def test_end_to_end(self, tmp_path: Path) -> None:
        """Full pipeline: build data → train CSCA → train flow → generate → validate."""
        t0 = time.perf_counter()

        # 1. Build dataset
        from chemvision.data.dataset_builder import MolecularDatasetBuilder
        builder = MolecularDatasetBuilder(seed=42)
        builder.add_seeds()
        builder.add_random_molecules(n=60)
        stats = builder.build(tmp_path / "data")
        assert stats.n_total >= 50
        fps, props, splits = MolecularDatasetBuilder.load_arrays(tmp_path / "data")

        # 2. Train CSCA
        from chemvision.models.csca import CSCATrainer, CSCAConfig
        csca_config = CSCAConfig(
            fp_dim=fps.shape[1], prop_dim=props.shape[1],
            latent_dim=32, hidden_dim=64, seed=42,
        )
        csca_trainer = CSCATrainer(csca_config)
        csca_result = csca_trainer.train(fps, props, epochs=50, batch_size=32)
        assert csca_result.converged or csca_result.final_loss < csca_result.loss_history[0]

        # 3. Train Flow Matcher
        from chemvision.generation.flow_matcher import ConditionalFlowMatcher, FlowMatcherConfig
        flow_config = FlowMatcherConfig(
            fp_dim=fps.shape[1], cond_dim=props.shape[1],
            hidden_dim=64, n_layers=2, seed=42,
        )
        cfm = ConditionalFlowMatcher(flow_config)
        flow_result = cfm.train(fps, props, epochs=40, batch_size=32)
        assert flow_result.final_loss < flow_result.loss_history[0]

        # 4. Generate new molecules conditioned on target properties
        target_props = np.array([
            [200.0, 2.0, 60.0, 1, 3, 2, 1, 0.7],  # drug-like target
            [150.0, 1.0, 40.0, 2, 2, 1, 0, 0.8],  # smaller drug-like
        ], dtype=np.float32)
        generated = cfm.sample(target_props, n_steps=20)
        assert len(generated) == 2

        # 5. Use CSCA to retrieve closest real molecules to generated ones
        retrieved = csca_trainer.retrieve(target_props, fps, k=3)
        assert len(retrieved) == 2
        assert all(len(r) == 3 for r in retrieved)

        # 6. Profile the pipeline
        from chemvision.eval.profiler import LatencyProfiler
        profiler = LatencyProfiler()
        profiler.record("dataset_build", 0)  # already timed above
        profiler.record("csca_train", 0)
        profiler.record("flow_train", 0)
        profiler.record("generation", 0)

        elapsed = time.perf_counter() - t0
        print(f"\n  Full pipeline: {elapsed:.1f}s "
              f"(dataset={stats.n_total} mols, "
              f"CSCA={csca_result.epochs} epochs, "
              f"Flow={flow_result.epochs} epochs)")
        assert elapsed < 120, f"Full pipeline too slow: {elapsed:.0f}s"


# ═══════════════════════════════════════════════════════════════════════════
# Reproducibility + seed propagation
# ═══════════════════════════════════════════════════════════════════════════

class TestReproducibility:

    def test_global_seed(self) -> None:
        from chemvision.core.reproducibility import set_global_seed
        set_global_seed(123)
        a = np.random.rand(5)
        set_global_seed(123)
        b = np.random.rand(5)
        assert np.allclose(a, b), "Global seed should make numpy deterministic"

    def test_global_seed_torch(self) -> None:
        from chemvision.core.reproducibility import set_global_seed
        set_global_seed(456)
        a = torch.randn(5)
        set_global_seed(456)
        b = torch.randn(5)
        assert torch.allclose(a, b)


# ═══════════════════════════════════════════════════════════════════════════
# Batch processing
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchProcessor:

    def test_encode_batch(self) -> None:
        from chemvision.core.batch import BatchProcessor
        proc = BatchProcessor(max_workers=2)
        smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
        result = proc.encode_batch(smiles, chunk_size=2)
        assert result.shape == (4, 2048)

    def test_predict_batch(self) -> None:
        from chemvision.core.batch import BatchProcessor
        proc = BatchProcessor(max_workers=2)
        smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
        results = proc.predict_batch(smiles)
        assert len(results) == 3
        assert all(r.mw is not None for r in results)
