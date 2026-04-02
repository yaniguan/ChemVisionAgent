"""Latency profiler for pipeline stage benchmarking.

Wraps individual pipeline stages and records wall-clock times for each call.
Produces percentile breakdowns and identifies bottleneck stages.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ProfileResult:
    """Profiling results for one pipeline stage."""

    stage: str
    n_calls: int = 0
    total_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    _pct_of_total: float = 0.0

    @property
    def pct_of_total(self) -> float:
        return self._pct_of_total


class LatencyProfiler:
    """Collect and report per-stage latency distributions.

    Example
    -------
    >>> profiler = LatencyProfiler()
    >>> with profiler.measure("encoding"):
    ...     encoder.encode(smiles)
    >>> with profiler.measure("prediction"):
    ...     predictor.predict(smiles)
    >>> report = profiler.report()
    >>> report[0].stage
    'encoding'
    """

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = defaultdict(list)

    class _Timer:
        def __init__(self, timings: list[float]) -> None:
            self._timings = timings
            self._start: float = 0.0

        def __enter__(self) -> "_Timer":
            self._start = time.perf_counter()
            return self

        def __exit__(self, *_: Any) -> None:
            elapsed_ms = (time.perf_counter() - self._start) * 1000
            self._timings.append(elapsed_ms)

    def measure(self, stage: str) -> _Timer:
        """Context manager that times a code block."""
        return self._Timer(self._timings[stage])

    def record(self, stage: str, elapsed_ms: float) -> None:
        """Manually record a timing."""
        self._timings[stage].append(elapsed_ms)

    def report(self) -> list[ProfileResult]:
        """Compute per-stage statistics and return sorted by total time (desc)."""
        results: list[ProfileResult] = []
        grand_total = sum(sum(ts) for ts in self._timings.values())

        for stage, timings in self._timings.items():
            arr = np.array(timings)
            r = ProfileResult(
                stage=stage,
                n_calls=len(arr),
                total_ms=float(arr.sum()),
                mean_ms=float(arr.mean()),
                p50_ms=float(np.percentile(arr, 50)),
                p95_ms=float(np.percentile(arr, 95)),
                p99_ms=float(np.percentile(arr, 99)),
                min_ms=float(arr.min()),
                max_ms=float(arr.max()),
            )
            results.append(r)

        results.sort(key=lambda r: r.total_ms, reverse=True)
        # Compute pct_of_total
        for r in results:
            r._pct_of_total = r.total_ms / grand_total * 100 if grand_total > 0 else 0
        return results

    def reset(self) -> None:
        self._timings.clear()
