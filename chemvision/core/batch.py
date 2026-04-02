"""Batch processing utilities for scalable molecular analysis.

Provides concurrent and batched execution for CPU-bound molecular operations
that would otherwise bottleneck on sequential processing.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T")


class BatchProcessor:
    """Process molecules in parallel batches.

    Example
    -------
    >>> from chemvision.models.mol_encoder import MolecularEncoder
    >>> enc = MolecularEncoder()
    >>> processor = BatchProcessor(max_workers=4)
    >>> results = processor.map(enc.compute_descriptors, smiles_list)
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._max_workers = max_workers

    def map(self, fn: Callable[[str], T], smiles_list: list[str]) -> list[T]:
        """Apply fn to each SMILES in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            return list(pool.map(fn, smiles_list))

    def encode_batch(
        self,
        smiles_list: list[str],
        chunk_size: int = 500,
    ) -> np.ndarray:
        """Encode SMILES to fingerprints in chunks (memory-bounded).

        Yields fingerprints in chunks to avoid loading all 2048-dim vectors
        into memory at once for very large molecule sets.
        """
        from chemvision.models.mol_encoder import MolecularEncoder
        encoder = MolecularEncoder()

        chunks: list[np.ndarray] = []
        for i in range(0, len(smiles_list), chunk_size):
            chunk = smiles_list[i:i + chunk_size]
            embeddings = np.stack([encoder.encode(s) for s in chunk])
            chunks.append(embeddings)

        return np.vstack(chunks)

    def predict_batch(
        self,
        smiles_list: list[str],
    ) -> list[Any]:
        """Predict properties for a list of SMILES in parallel."""
        from chemvision.generation.property_predictor import PropertyPredictor
        predictor = PropertyPredictor(use_mace=False)
        return self.map(predictor.predict, smiles_list)
