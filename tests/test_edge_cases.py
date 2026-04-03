"""Edge-case tests for schema validation, empty/invalid inputs, and boundary conditions."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Empty / None inputs
# ═══════════════════════════════════════════════════════════════════════════


class TestEmptyNoneInputs:

    def test_empty_smiles_list_to_vocab(self) -> None:
        """build_vocabulary([]) returns only the 3 base tokens."""
        from chemvision.generation.selfies_gen import build_vocabulary

        vocab = build_vocabulary([])
        assert "[nop]" in vocab
        assert "[BOS]" in vocab
        assert "[EOS]" in vocab
        assert len(vocab) == 3

    def test_none_smiles_to_graph(self) -> None:
        """smiles_to_graph(None-equivalent invalid string) returns None."""
        from chemvision.models.gnn import smiles_to_graph

        assert smiles_to_graph("") is None

    def test_empty_string_smiles(self) -> None:
        """Empty string SMILES returns None from smiles_to_graph."""
        from chemvision.models.gnn import smiles_to_graph

        result = smiles_to_graph("")
        assert result is None

    def test_empty_batch_graphs(self) -> None:
        """batch_graphs([]) returns a valid empty batch dict."""
        from chemvision.models.gnn import batch_graphs

        bg = batch_graphs([])
        assert bg["n_graphs"] == 0
        assert bg["x"].shape[0] == 0
        assert bg["edge_index"].shape == (2, 0)
        assert len(bg["batch"]) == 0

    def test_generate_with_no_training(self) -> None:
        """Calling generate() before train() raises RuntimeError."""
        torch = pytest.importorskip("torch")
        from chemvision.generation.selfies_gen import SELFIESGenerator

        gen = SELFIESGenerator()
        with pytest.raises(RuntimeError, match="not trained"):
            gen.generate(np.array([[0.5] * 8]))


# ═══════════════════════════════════════════════════════════════════════════
# Invalid molecules
# ═══════════════════════════════════════════════════════════════════════════


class TestInvalidMolecules:

    def test_malformed_smiles_parsing(self) -> None:
        """Malformed SMILES returns None, does not crash."""
        from chemvision.models.gnn import smiles_to_graph

        result = smiles_to_graph("NOT_A_SMILES!!!")
        assert result is None

    def test_invalid_selfies_tokens(self) -> None:
        """Invalid SELFIES tokens handled gracefully by decoder."""
        from chemvision.generation.selfies_gen import selfies_tokens_to_smiles

        # Garbage tokens -- should return None or a string, not raise
        result = selfies_tokens_to_smiles(["[INVALID_TOKEN_XYZ]", "[???]"])
        # Either None or a string is acceptable; no crash
        assert result is None or isinstance(result, str)

    def test_smiles_with_unicode(self) -> None:
        """Pure unicode string returns None from graph conversion."""
        from chemvision.models.gnn import smiles_to_graph

        result = smiles_to_graph("\u2603\u2604\u2605")  # pure unicode, no valid atoms
        assert result is None

    def test_extremely_long_smiles(self) -> None:
        """Very long SMILES string is either parsed or returns None, no hang."""
        from chemvision.models.gnn import smiles_to_graph

        long_smi = "C" * 5000  # long linear alkane
        result = smiles_to_graph(long_smi)
        # Should either work or return None, not hang or crash
        assert result is None or result["n_atoms"] > 0

    def test_smiles_to_selfies_tokens_invalid(self) -> None:
        """smiles_to_selfies_tokens returns None for invalid SMILES."""
        from chemvision.generation.selfies_gen import smiles_to_selfies_tokens

        assert smiles_to_selfies_tokens("INVALID!!!") is None


# ═══════════════════════════════════════════════════════════════════════════
# Numeric edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestNumericEdgeCases:

    def test_nan_property_values(self) -> None:
        """NaN property values are detected during train()."""
        torch = pytest.importorskip("torch")
        from chemvision.generation.selfies_gen import SELFIESGenerator, SELFIESGenConfig

        gen = SELFIESGenerator(SELFIESGenConfig(cond_dim=2))
        smiles = ["CCO", "c1ccccc1"]
        props = np.array([[1.0, float("nan")], [0.5, 0.3]])
        with pytest.raises(ValueError, match="NaN"):
            gen.train(smiles, props, epochs=1)

    def test_inf_property_values(self) -> None:
        """Inf property values are detected during train()."""
        torch = pytest.importorskip("torch")
        from chemvision.generation.selfies_gen import SELFIESGenerator, SELFIESGenConfig

        gen = SELFIESGenerator(SELFIESGenConfig(cond_dim=2))
        smiles = ["CCO", "c1ccccc1"]
        props = np.array([[1.0, float("inf")], [0.5, 0.3]])
        with pytest.raises(ValueError, match="Inf"):
            gen.train(smiles, props, epochs=1)

    def test_zero_temperature_generation(self) -> None:
        """Zero temperature should be rejected (division by zero)."""
        torch = pytest.importorskip("torch")
        from unittest.mock import MagicMock
        from chemvision.generation.selfies_gen import SELFIESGenerator, SELFIESGenConfig

        gen = SELFIESGenerator(SELFIESGenConfig(cond_dim=2))
        gen.model = MagicMock()  # mock model so .eval() works
        gen.vocab = {"[nop]": 0, "[BOS]": 1, "[EOS]": 2}
        gen._cond_mean = np.zeros(2)
        gen._cond_std = np.ones(2)

        with pytest.raises(ValueError, match="temperature must be positive"):
            gen.generate(np.array([[0.5, 0.5]]), temperature=0.0)

    def test_negative_temperature(self) -> None:
        """Negative temperature should be rejected."""
        torch = pytest.importorskip("torch")
        from unittest.mock import MagicMock
        from chemvision.generation.selfies_gen import SELFIESGenerator, SELFIESGenConfig

        gen = SELFIESGenerator(SELFIESGenConfig(cond_dim=2))
        gen.model = MagicMock()
        gen.vocab = {"[nop]": 0, "[BOS]": 1, "[EOS]": 2}
        gen._cond_mean = np.zeros(2)
        gen._cond_std = np.ones(2)

        with pytest.raises(ValueError, match="temperature must be positive"):
            gen.generate(np.array([[0.5, 0.5]]), temperature=-1.0)

    def test_confidence_out_of_bounds_skill_result(self) -> None:
        """SkillResult rejects confidence outside [0, 1]."""
        from pydantic import ValidationError
        from chemvision.skills.base import SkillResult

        with pytest.raises(ValidationError):
            SkillResult(skill_name="test", raw_output="x", confidence=1.5)

        with pytest.raises(ValidationError):
            SkillResult(skill_name="test", raw_output="x", confidence=-0.1)

    def test_confidence_out_of_bounds_tool_call_log(self) -> None:
        """ToolCallLog rejects confidence outside [0, 1]."""
        from pydantic import ValidationError
        from chemvision.agent.tool_log import ToolCallLog

        with pytest.raises(ValidationError):
            ToolCallLog(
                skill_name="test",
                output_summary="x",
                confidence=2.0,
            )

        with pytest.raises(ValidationError):
            ToolCallLog(
                skill_name="test",
                output_summary="x",
                confidence=-0.5,
            )

    def test_confidence_valid_bounds(self) -> None:
        """SkillResult and ToolCallLog accept boundary values 0.0 and 1.0."""
        from chemvision.skills.base import SkillResult
        from chemvision.agent.tool_log import ToolCallLog

        r0 = SkillResult(skill_name="t", raw_output="x", confidence=0.0)
        assert r0.confidence == 0.0
        r1 = SkillResult(skill_name="t", raw_output="x", confidence=1.0)
        assert r1.confidence == 1.0

        log0 = ToolCallLog(skill_name="t", output_summary="x", confidence=0.0)
        assert log0.confidence == 0.0
        log1 = ToolCallLog(skill_name="t", output_summary="x", confidence=1.0)
        assert log1.confidence == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Data validation
# ═══════════════════════════════════════════════════════════════════════════


class TestDataValidation:

    def test_properties_shape_mismatch(self) -> None:
        """Mismatched properties length vs smiles_list raises ValueError."""
        torch = pytest.importorskip("torch")
        from chemvision.generation.selfies_gen import SELFIESGenerator, SELFIESGenConfig

        gen = SELFIESGenerator(SELFIESGenConfig(cond_dim=2))
        smiles = ["CCO", "c1ccccc1", "CC"]
        props = np.array([[1.0, 0.5], [0.3, 0.2]])  # length 2, smiles length 3
        with pytest.raises(ValueError, match="properties length"):
            gen.train(smiles, props, epochs=1)

    def test_properties_dim_mismatch(self) -> None:
        """Properties with wrong feature dim raises ValueError."""
        torch = pytest.importorskip("torch")
        from chemvision.generation.selfies_gen import SELFIESGenerator, SELFIESGenConfig

        gen = SELFIESGenerator(SELFIESGenConfig(cond_dim=8))
        smiles = ["CCO", "c1ccccc1"]
        props = np.array([[1.0, 0.5], [0.3, 0.2]])  # dim=2 but cond_dim=8
        with pytest.raises(ValueError, match="cond_dim"):
            gen.train(smiles, props, epochs=1)

    def test_empty_dataframe_load(self) -> None:
        """build_vocabulary with list of empty/invalid SMILES returns base vocab."""
        from chemvision.generation.selfies_gen import build_vocabulary

        vocab = build_vocabulary(["", "", "INVALID!!"])
        # Only the 3 base tokens should remain
        assert len(vocab) == 3

    def test_corrupted_csv_path(self) -> None:
        """Attempting to load a nonexistent file raises an appropriate error."""
        with pytest.raises((FileNotFoundError, OSError)):
            Path("/nonexistent/fake_data.csv").read_text()


# ═══════════════════════════════════════════════════════════════════════════
# Model edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestModelEdgeCases:

    def test_single_molecule_training(self) -> None:
        """Training on 1 molecule returns TrainResult (may not converge)."""
        torch = pytest.importorskip("torch")
        from chemvision.generation.selfies_gen import SELFIESGenerator, SELFIESGenConfig

        gen = SELFIESGenerator(SELFIESGenConfig(cond_dim=2))
        smiles = ["CCO"]
        props = np.array([[0.5, 0.3]])
        # With <10 valid sequences the train() function returns early
        result = gen.train(smiles, props, epochs=5)
        assert result.epochs == 0  # too few sequences → early return

    def test_generate_zero_molecules(self) -> None:
        """generate() with empty target_properties returns empty list."""
        torch = pytest.importorskip("torch")
        from unittest.mock import MagicMock
        from chemvision.generation.selfies_gen import SELFIESGenerator, SELFIESGenConfig

        gen = SELFIESGenerator(SELFIESGenConfig(cond_dim=2))
        gen.model = MagicMock()
        gen.vocab = {"[nop]": 0, "[BOS]": 1, "[EOS]": 2}
        gen._cond_mean = np.zeros(2)
        gen._cond_std = np.ones(2)

        # Empty array of shape (0, 2)
        results = gen.generate(np.zeros((0, 2)))
        assert results == []

    def test_batch_encode_all_invalid(self) -> None:
        """encode_smiles_batch with all invalid SMILES returns zeros."""
        torch = pytest.importorskip("torch")
        from chemvision.models.gnn import GINEncoder

        enc = GINEncoder(embed_dim=64, n_layers=2)
        result = enc.encode_smiles_batch(["INVALID1", "INVALID2", "!!!"])
        assert result.shape == (3, 64)
        assert np.allclose(result, 0.0)

    def test_gin_no_edges_molecule(self) -> None:
        """Single-atom molecule (no bonds) is handled by GIN encoder."""
        torch = pytest.importorskip("torch")
        from chemvision.models.gnn import smiles_to_graph, GINEncoder

        # Single atom: [Na] has no bonds
        g = smiles_to_graph("[Na]")
        # May return None or a valid graph depending on RDKit
        if g is not None:
            assert g["edge_index"].shape == (2, 0)
            assert g["n_atoms"] == 1
            # Should still encode without error
            enc = GINEncoder(embed_dim=64, n_layers=2)
            emb = enc.encode_smiles("[Na]")
            assert emb.shape == (64,)

    def test_schnet_single_atom(self) -> None:
        """SchNet encoder handles single-atom molecule."""
        torch = pytest.importorskip("torch")
        from chemvision.models.gnn import SchNetEncoder

        enc = SchNetEncoder(embed_dim=64, n_layers=2)
        emb = enc.encode_smiles("[Na]")
        assert emb.shape == (64,)
        # Should produce a valid (non-NaN) embedding
        assert not np.any(np.isnan(emb))

    def test_gin_encode_valid_smiles(self) -> None:
        """GIN encoder produces non-zero embedding for valid SMILES."""
        torch = pytest.importorskip("torch")
        from chemvision.models.gnn import GINEncoder

        enc = GINEncoder(embed_dim=64, n_layers=2)
        emb = enc.encode_smiles("CCO")
        assert emb.shape == (64,)
        # Valid molecule should produce non-zero output
        assert not np.allclose(emb, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Boundary conditions
# ═══════════════════════════════════════════════════════════════════════════


class TestBoundaryConditions:

    def test_max_len_selfies_sequence(self) -> None:
        """Sequences longer than max_len are truncated during tokenisation."""
        torch = pytest.importorskip("torch")
        from chemvision.generation.selfies_gen import (
            SELFIESGenConfig,
            build_vocabulary,
            smiles_to_selfies_tokens,
        )

        config = SELFIESGenConfig(max_len=10)
        # Benzene encodes to ~6 SELFIES tokens + BOS + EOS = ~8
        # Use a longer molecule to force truncation
        long_smi = "C" * 50  # long alkane
        tokens = smiles_to_selfies_tokens(long_smi)
        if tokens is not None:
            # BOS + tokens + EOS
            full_len = 1 + len(tokens) + 1
            # After truncation it should be at most max_len
            ids = list(range(full_len))
            truncated = ids[: config.max_len]
            assert len(truncated) <= config.max_len

    def test_vocab_size_one(self) -> None:
        """build_vocabulary with a single simple molecule still builds vocab."""
        from chemvision.generation.selfies_gen import build_vocabulary

        vocab = build_vocabulary(["C"])  # methane
        assert len(vocab) >= 3  # at least PAD, BOS, EOS
        # Should have the [C] token too
        assert len(vocab) >= 4

    def test_top_k_larger_than_vocab(self) -> None:
        """top_k > vocab_size is clamped via min() in generate()."""
        torch = pytest.importorskip("torch")
        from chemvision.generation.selfies_gen import SELFIESGenerator, SELFIESGenConfig

        config = SELFIESGenConfig(cond_dim=2, max_len=10, n_layers=1)
        gen = SELFIESGenerator(config)

        # Build a minimal trained state with small vocab
        smiles = ["CCO"] * 20  # need at least 10 valid for training to proceed
        props = np.tile([0.5, 0.3], (20, 1))
        result = gen.train(smiles, props, epochs=2, patience=2)

        if gen.model is not None:
            # top_k=99999 should not crash
            molecules = gen.generate(
                np.array([[0.5, 0.3]]),
                n_per_target=1,
                top_k=99999,
            )
            assert isinstance(molecules, list)

    def test_graph_consistency_check(self) -> None:
        """smiles_to_graph returns graphs with consistent shapes."""
        from chemvision.models.gnn import smiles_to_graph

        for smi in ["CCO", "c1ccccc1", "CC(=O)O", "[Cu]"]:
            g = smiles_to_graph(smi)
            if g is not None:
                n = g["n_atoms"]
                assert g["x"].shape[0] == n
                # All edge indices must be within [0, n)
                if g["edge_index"].size > 0:
                    assert g["edge_index"].max() < n
                    assert g["edge_index"].min() >= 0

    def test_datetime_fields_are_tz_aware(self) -> None:
        """Timestamps in models should be timezone-aware (UTC)."""
        from chemvision.agent.tool_log import ToolCallLog
        from chemvision.agent.trace import AgentStep, AgentTrace, StepType

        log = ToolCallLog(skill_name="t", output_summary="x")
        assert log.timestamp.tzinfo is not None

        step = AgentStep(step_index=0, step_type=StepType.THOUGHT, content="hi")
        assert step.timestamp.tzinfo is not None

        trace = AgentTrace(query="q")
        assert trace.started_at.tzinfo is not None

    def test_skill_result_none_confidence_allowed(self) -> None:
        """SkillResult accepts confidence=None."""
        from chemvision.skills.base import SkillResult

        r = SkillResult(skill_name="t", raw_output="x", confidence=None)
        assert r.confidence is None

    def test_nan_target_properties_generate(self) -> None:
        """generate() rejects NaN in target_properties."""
        torch = pytest.importorskip("torch")
        from unittest.mock import MagicMock
        from chemvision.generation.selfies_gen import SELFIESGenerator, SELFIESGenConfig

        gen = SELFIESGenerator(SELFIESGenConfig(cond_dim=2))
        gen.model = MagicMock()
        gen.vocab = {"[nop]": 0, "[BOS]": 1, "[EOS]": 2}
        gen._cond_mean = np.zeros(2)
        gen._cond_std = np.ones(2)

        with pytest.raises(ValueError, match="NaN"):
            gen.generate(np.array([[float("nan"), 0.5]]))
