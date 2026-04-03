"""Molecule vector store with O(log N) HNSW search.

Backends (auto-selected):
  1. **hnswlib** (default if installed): O(log N) approximate NN, <1ms at 100K+.
  2. **numpy fallback**: O(N) exact cosine similarity, fine for <10K molecules.
  3. **ChromaDB** (persistent=True): disk-backed HNSW for production use.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Try to import hnswlib for fast ANN search
try:
    import hnswlib
    _HAS_HNSW = True
except ImportError:
    _HAS_HNSW = False


class MoleculeVectorStore:
    """Cosine-similarity retrieval over molecular fingerprint embeddings.

    Automatically uses HNSW (O(log N)) when hnswlib is installed,
    falls back to exact numpy search (O(N)) otherwise.

    Example
    -------
    >>> store = MoleculeVectorStore()
    >>> store.add("aspirin", embedding, {"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    >>> hits = store.search(query_embedding, k=3)
    >>> hits[0]["name"]
    'aspirin'
    """

    def __init__(
        self,
        dim: int = 2048,
        max_elements: int = 100_000,
        ef_construction: int = 200,
        M: int = 16,
        persistent: bool = False,
        persist_dir: str = ".chroma_store",
        use_hnsw: bool = True,
    ) -> None:
        self._dim = dim
        self._names: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._metadata: list[dict[str, Any]] = []
        self._persistent = persistent

        # HNSW index (built lazily on first search if >1000 elements)
        self._hnsw_index: Any = None
        self._use_hnsw = use_hnsw and _HAS_HNSW
        self._max_elements = max_elements
        self._ef_construction = ef_construction
        self._M = M
        self._hnsw_dirty = True  # index needs rebuild

        if persistent:
            try:
                import chromadb
                self._chroma = chromadb.PersistentClient(path=persist_dir)
                self._col = self._chroma.get_or_create_collection(
                    "molecules", metadata={"hnsw:space": "cosine"}
                )
            except ImportError:
                self._persistent = False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, name: str, embedding: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        """Add one molecule to the store."""
        meta = metadata or {}
        norm = embedding / (np.linalg.norm(embedding) + 1e-9)
        self._names.append(name)
        self._embeddings.append(norm)
        self._metadata.append(meta)
        self._hnsw_dirty = True

        if self._persistent:
            self._col.upsert(
                ids=[name],
                embeddings=[norm.tolist()],
                metadatas=[{k: str(v) for k, v in meta.items()}],
            )

    def add_batch(
        self,
        names: list[str],
        embeddings: list[np.ndarray],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add multiple molecules at once."""
        metas = metadatas or [{} for _ in names]
        for name, emb, meta in zip(names, embeddings, metas):
            self.add(name, emb, meta)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        """Return the *k* most similar stored molecules.

        Uses HNSW (O(log N)) if available and store has >100 elements,
        otherwise falls back to exact numpy search.
        """
        if not self._embeddings:
            return []

        k = min(k, len(self._embeddings))
        q = query / (np.linalg.norm(query) + 1e-9)

        # Use HNSW for large stores
        if self._use_hnsw and len(self._embeddings) > 100:
            return self._search_hnsw(q, k)

        # Exact numpy search (small stores)
        return self._search_exact(q, k)

    def _search_exact(self, q: np.ndarray, k: int) -> list[dict[str, Any]]:
        """O(N) exact cosine similarity search."""
        matrix = np.stack(self._embeddings)
        scores = matrix @ q
        idx = np.argsort(-scores)[:k]
        return [
            {"name": self._names[i], "score": float(scores[i]), **self._metadata[i]}
            for i in idx
        ]

    def _search_hnsw(self, q: np.ndarray, k: int) -> list[dict[str, Any]]:
        """O(log N) approximate nearest neighbor via hnswlib."""
        if self._hnsw_dirty:
            self._rebuild_hnsw()

        labels, distances = self._hnsw_index.knn_query(q.reshape(1, -1), k=k)
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            # hnswlib with cosine space returns 1 - cosine_similarity
            score = 1.0 - dist
            results.append(
                {"name": self._names[idx], "score": float(score), **self._metadata[idx]}
            )
        return results

    def _rebuild_hnsw(self) -> None:
        """Rebuild the HNSW index from current embeddings."""
        n = len(self._embeddings)
        dim = self._embeddings[0].shape[0] if self._embeddings else self._dim

        self._hnsw_index = hnswlib.Index(space="cosine", dim=dim)
        self._hnsw_index.init_index(
            max_elements=max(n * 2, self._max_elements),
            ef_construction=self._ef_construction,
            M=self._M,
        )
        matrix = np.stack(self._embeddings)
        self._hnsw_index.add_items(matrix, list(range(n)))
        self._hnsw_index.set_ef(50)  # search-time ef
        self._hnsw_dirty = False
        logger.debug("HNSW index rebuilt with %d elements (dim=%d)", n, dim)

    @property
    def backend(self) -> str:
        """Return which search backend is active."""
        if self._persistent:
            return "chromadb"
        if self._use_hnsw and len(self._embeddings) > 100:
            return "hnsw"
        return "numpy"

    def __len__(self) -> int:
        return len(self._names)

    # ------------------------------------------------------------------
    # Persistence (numpy fallback)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save store to a JSON + numpy npz file pair."""
        path = Path(path)
        np.savez(path.with_suffix(".npz"), embeddings=np.stack(self._embeddings))
        with open(path.with_suffix(".json"), "w") as f:
            json.dump({"names": self._names, "metadata": self._metadata}, f)

    def load(self, path: str | Path) -> None:
        """Restore a previously saved store (clears current contents)."""
        path = Path(path)
        data = np.load(path.with_suffix(".npz"))
        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)
        self._embeddings = [data["embeddings"][i] for i in range(len(data["embeddings"]))]
        self._names = meta["names"]
        self._metadata = meta["metadata"]
        self._hnsw_dirty = True
