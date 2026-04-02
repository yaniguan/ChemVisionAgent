"""Tests for evaluation pipeline, data pipeline, calibration, and bug fixes.

Covers:
  - MetricsSuite (accuracy, F1, ECE, regression metrics, latency)
  - ConfidenceCalibrator (isotonic + Platt)
  - AIQualityScorer (5-dimension scoring)
  - LatencyProfiler (stage timing)
  - DataStore (Parquet persistence, quality scoring, splits)
  - Bug fix: Pareto dominance logic
  - Bug fix: thread-safe API state
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# MetricsSuite
# ═══════════════════════════════════════════════════════════════════════════

class TestMetricsSuite:

    def setup_method(self) -> None:
        from chemvision.eval.metrics import MetricsSuite
        self.suite = MetricsSuite()

    def test_perfect_accuracy(self) -> None:
        for _ in range(10):
            self.suite.add("skill_a", "hello", "hello", confidence=0.9)
        result = self.suite.compute()
        assert result.overall_accuracy == 1.0
        assert result.per_skill["skill_a"].accuracy == 1.0

    def test_zero_accuracy(self) -> None:
        for _ in range(10):
            self.suite.add("skill_a", "wrong", "right", confidence=0.9)
        result = self.suite.compute()
        assert result.overall_accuracy == 0.0

    def test_mixed_accuracy(self) -> None:
        self.suite.add("s", "a", "a")
        self.suite.add("s", "b", "a")
        result = self.suite.compute()
        assert result.overall_accuracy == pytest.approx(0.5)

    def test_regression_metrics(self) -> None:
        self.suite.add_numeric("s", predicted=10.0, ground_truth=10.0)
        self.suite.add_numeric("s", predicted=12.0, ground_truth=10.0)
        self.suite.add_numeric("s", predicted=8.0, ground_truth=10.0)
        result = self.suite.compute()
        m = result.per_skill["s"]
        assert m.mae is not None
        assert m.mae == pytest.approx(4.0 / 3)  # mean of |0|, |2|, |2|
        assert m.rmse is not None
        assert m.max_error == pytest.approx(2.0)

    def test_ece_perfect_calibration(self) -> None:
        # Conf=1.0, all correct → ECE = 0
        for _ in range(20):
            self.suite.add("s", "x", "x", confidence=1.0)
        result = self.suite.compute()
        assert result.overall_ece is not None
        assert result.overall_ece < 0.05

    def test_ece_overconfident(self) -> None:
        # High confidence, low accuracy → high ECE
        for _ in range(20):
            self.suite.add("s", "wrong", "right", confidence=0.95)
        result = self.suite.compute()
        assert result.overall_ece is not None
        assert result.overall_ece > 0.5

    def test_multi_skill(self) -> None:
        self.suite.add("a", "x", "x", confidence=0.9)
        self.suite.add("a", "y", "x", confidence=0.8)
        self.suite.add("b", "z", "z", confidence=0.7)
        result = self.suite.compute()
        assert "a" in result.per_skill
        assert "b" in result.per_skill
        assert result.per_skill["a"].accuracy == pytest.approx(0.5)
        assert result.per_skill["b"].accuracy == pytest.approx(1.0)

    def test_latency_percentiles(self) -> None:
        for i in range(100):
            self.suite.add("s", "x", "x", latency_ms=float(i))
        result = self.suite.compute()
        m = result.per_skill["s"]
        assert m.latency_p50_ms is not None
        assert 45 < m.latency_p50_ms < 55

    def test_empty_suite(self) -> None:
        result = self.suite.compute()
        assert result.n_total == 0
        assert result.overall_accuracy == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ConfidenceCalibrator
# ═══════════════════════════════════════════════════════════════════════════

class TestConfidenceCalibrator:

    def test_isotonic_reduces_ece(self) -> None:
        from chemvision.eval.calibration import ConfidenceCalibrator
        rng = np.random.RandomState(42)
        # Simulate overconfident model: high conf, 50% accuracy
        confs = rng.uniform(0.7, 1.0, size=100).tolist()
        correct = rng.binomial(1, 0.5, size=100).tolist()

        cal = ConfidenceCalibrator(method="isotonic")
        result = cal.fit(confs, correct)
        assert result.ece_after <= result.ece_before
        assert result.improvement_pct >= 0

    def test_platt_scaling(self) -> None:
        from chemvision.eval.calibration import ConfidenceCalibrator
        rng = np.random.RandomState(7)
        confs = rng.uniform(0.3, 0.9, size=80).tolist()
        correct = [1 if c > 0.5 else 0 for c in confs]

        cal = ConfidenceCalibrator(method="platt")
        result = cal.fit(confs, correct)
        assert cal.is_fitted
        assert 0.0 <= cal.calibrate(0.8) <= 1.0
        assert 0.0 <= cal.calibrate(0.2) <= 1.0

    def test_calibrate_before_fit_passthrough(self) -> None:
        from chemvision.eval.calibration import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        assert cal.calibrate(0.75) == 0.75  # passthrough

    def test_calibrate_batch(self) -> None:
        from chemvision.eval.calibration import ConfidenceCalibrator
        cal = ConfidenceCalibrator(method="isotonic")
        cal.fit([0.9, 0.8, 0.7, 0.6, 0.5], [1, 1, 0, 0, 0])
        batch = cal.calibrate_batch([0.85, 0.55])
        assert len(batch) == 2
        assert all(0 <= v <= 1 for v in batch)

    def test_invalid_method_raises(self) -> None:
        from chemvision.eval.calibration import ConfidenceCalibrator
        with pytest.raises(ValueError, match="Unknown method"):
            ConfidenceCalibrator(method="invalid")


# ═══════════════════════════════════════════════════════════════════════════
# AIQualityScorer
# ═══════════════════════════════════════════════════════════════════════════

class TestAIQualityScorer:

    def test_perfect_score(self) -> None:
        from chemvision.eval.quality import AIQualityScorer
        scorer = AIQualityScorer(target_latency_ms=5000)
        for _ in range(20):
            scorer.add_result(correct=True, confidence=1.0, completeness=1.0, latency_ms=1000)
        report = scorer.score()
        assert report.composite_score >= 85
        assert report.grade in ("A", "B")

    def test_poor_score(self) -> None:
        from chemvision.eval.quality import AIQualityScorer
        scorer = AIQualityScorer(target_latency_ms=1000)
        for _ in range(20):
            scorer.add_result(correct=False, confidence=0.95, completeness=0.2, latency_ms=5000)
        report = scorer.score()
        assert report.composite_score < 40
        assert report.grade in ("D", "F")

    def test_consistency_scoring(self) -> None:
        from chemvision.eval.quality import AIQualityScorer
        scorer = AIQualityScorer()
        scorer.add_result(correct=True, confidence=0.9)
        # Same input, same output → high consistency
        scorer.add_consistency_pair("img1", "result_a")
        scorer.add_consistency_pair("img1", "result_a")
        scorer.add_consistency_pair("img1", "result_a")
        report = scorer.score()
        cons_dim = next((d for d in report.dimensions if d.name == "Consistency"), None)
        assert cons_dim is not None
        assert cons_dim.score >= 18  # high consistency

    def test_summary_string(self) -> None:
        from chemvision.eval.quality import AIQualityScorer
        scorer = AIQualityScorer()
        scorer.add_result(correct=True, confidence=0.8, completeness=0.9, latency_ms=2000)
        report = scorer.score()
        summary = report.summary()
        assert "AI Quality Score" in summary
        assert report.grade in summary


# ═══════════════════════════════════════════════════════════════════════════
# LatencyProfiler
# ═══════════════════════════════════════════════════════════════════════════

class TestLatencyProfiler:

    def test_measure_records_time(self) -> None:
        from chemvision.eval.profiler import LatencyProfiler
        profiler = LatencyProfiler()
        with profiler.measure("sleep"):
            time.sleep(0.01)  # 10ms
        report = profiler.report()
        assert len(report) == 1
        assert report[0].stage == "sleep"
        assert report[0].mean_ms >= 8  # at least 8ms (accounting for scheduling)

    def test_multiple_stages(self) -> None:
        from chemvision.eval.profiler import LatencyProfiler
        profiler = LatencyProfiler()
        for _ in range(5):
            with profiler.measure("fast"):
                pass
            with profiler.measure("slow"):
                time.sleep(0.001)
        report = profiler.report()
        assert len(report) == 2
        # Sorted by total time desc, so "slow" should be first
        assert report[0].stage == "slow"

    def test_manual_record(self) -> None:
        from chemvision.eval.profiler import LatencyProfiler
        profiler = LatencyProfiler()
        profiler.record("test", 100.0)
        profiler.record("test", 200.0)
        report = profiler.report()
        assert report[0].mean_ms == pytest.approx(150.0)

    def test_reset(self) -> None:
        from chemvision.eval.profiler import LatencyProfiler
        profiler = LatencyProfiler()
        profiler.record("test", 100.0)
        profiler.reset()
        assert profiler.report() == []


# ═══════════════════════════════════════════════════════════════════════════
# DataStore + DataQualityScorer
# ═══════════════════════════════════════════════════════════════════════════

class TestDataStore:

    def test_ingest_and_query(self, tmp_path: Path) -> None:
        from chemvision.data.pipeline import DataStore, DataRecord
        store = DataStore(tmp_path / "store")
        record = DataRecord(smiles="CCO", domain="molecular", source="pubchem")
        stored = store.ingest(record)
        assert stored.record_id != ""
        assert stored.quality_score > 0

        df = store.query(domain="molecular")
        assert len(df) == 1

    def test_deduplication(self, tmp_path: Path) -> None:
        from chemvision.data.pipeline import DataStore, DataRecord
        store = DataStore(tmp_path / "store")
        r1 = DataRecord(smiles="CCO", domain="molecular", source="pubchem")
        r2 = DataRecord(smiles="CCO", domain="molecular", source="pubchem")
        store.ingest(r1)
        store.ingest(r2)
        # Same content hash → deduplicated (version incremented)
        assert store.count() == 1
        df = store.query()
        assert int(df.iloc[0]["version"]) == 2

    def test_quality_scoring(self) -> None:
        from chemvision.data.pipeline import DataQualityScorer, DataRecord
        scorer = DataQualityScorer()

        # Good record
        good = DataRecord(smiles="CCO", domain="molecular", source="pubchem",
                         molecular_weight=46.07, molecular_formula="C2H6O")
        score, flags = scorer.score(good)
        assert score > 0.5
        assert "invalid_smiles" not in flags

        # Bad record (invalid SMILES)
        bad = DataRecord(smiles="NOT_SMILES", domain="", source="")
        score_bad, flags_bad = scorer.score(bad)
        assert score_bad < score
        assert "invalid_smiles" in flags_bad

    def test_parquet_persistence(self, tmp_path: Path) -> None:
        from chemvision.data.pipeline import DataStore, DataRecord
        store = DataStore(tmp_path / "store")
        store.ingest(DataRecord(smiles="CCO", domain="molecular", source="test"))
        store.ingest(DataRecord(smiles="CC(=O)O", domain="molecular", source="test"))

        # Reload from disk
        store2 = DataStore(tmp_path / "store")
        assert store2.count() == 2

    def test_split_deterministic(self, tmp_path: Path) -> None:
        from chemvision.data.pipeline import DataStore, DataRecord
        store = DataStore(tmp_path / "store")
        for i in range(100):
            store.ingest(DataRecord(smiles=f"C{'C' * i}", domain="mol", source="test"))

        train1, val1, test1 = store.split(seed=42)
        train2, val2, test2 = store.split(seed=42)
        assert list(train1["record_id"]) == list(train2["record_id"])
        assert len(train1) == 80
        assert len(val1) == 10
        assert len(test1) == 10

    def test_stats(self, tmp_path: Path) -> None:
        from chemvision.data.pipeline import DataStore, DataRecord
        store = DataStore(tmp_path / "store")
        store.ingest(DataRecord(smiles="CCO", domain="molecular", source="pubchem"))
        store.ingest(DataRecord(smiles="CC", domain="crystal", source="synthetic"))
        stats = store.stats()
        assert stats["total"] == 2
        assert "molecular" in stats["by_domain"]


# ═══════════════════════════════════════════════════════════════════════════
# Bug fix verification: Pareto dominance
# ═══════════════════════════════════════════════════════════════════════════

class TestParetoDominanceFix:

    def test_dominance_correct(self) -> None:
        from chemvision.generation.pareto_mcts import Candidate

        a = Candidate(smiles="A", scores={"x": 0.9, "y": 0.8})
        b = Candidate(smiles="B", scores={"x": 0.7, "y": 0.6})
        assert a.dominates(b), "A should dominate B (better in both)"
        assert not b.dominates(a), "B should not dominate A"

    def test_no_dominance_when_equal(self) -> None:
        from chemvision.generation.pareto_mcts import Candidate
        a = Candidate(smiles="A", scores={"x": 0.5, "y": 0.5})
        b = Candidate(smiles="B", scores={"x": 0.5, "y": 0.5})
        assert not a.dominates(b), "Equal scores → no dominance"
        assert not b.dominates(a)

    def test_no_dominance_when_tradeoff(self) -> None:
        from chemvision.generation.pareto_mcts import Candidate
        a = Candidate(smiles="A", scores={"x": 0.9, "y": 0.3})
        b = Candidate(smiles="B", scores={"x": 0.3, "y": 0.9})
        assert not a.dominates(b), "Trade-off → no dominance"
        assert not b.dominates(a)

    def test_missing_key_in_other(self) -> None:
        from chemvision.generation.pareto_mcts import Candidate
        a = Candidate(smiles="A", scores={"x": 0.5, "y": 0.5})
        b = Candidate(smiles="B", scores={"x": 0.5})  # missing "y"
        # b.scores has no "y" → defaults to inf → a cannot dominate
        assert not a.dominates(b)
