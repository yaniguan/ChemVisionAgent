"""In-memory molecule vector store with optional ChromaDB persistence.

Stores molecular fingerprint embeddings (2048-bit Morgan) and allows
fast cosine-similarity retrieval.  Backs a RAG layer so the agent can
ground answers in known compounds rather than hallucinating.

Backend selection
-----------------
* Default: pure numpy (no extra deps, ephemeral).
* ``persistent=True``: uses ChromaDB for disk-backed storage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class MoleculeVectorStore:
    """Cosine-similarity retrieval over molecular fingerprint embeddings.

    Example
    -------
    >>> store = MoleculeVectorStore()
    >>> store.add("aspirin", embedding, {"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    >>> hits = store.search(query_embedding, k=3)
    >>> hits[0]["name"]
    'aspirin'
    """

    def __init__(self, persistent: bool = False, persist_dir: str = ".chroma_store") -> None:
        self._persistent = persistent
        self._names: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._metadata: list[dict[str, Any]] = []

        if persistent:
            try:
                import chromadb

                self._chroma = chromadb.PersistentClient(path=persist_dir)
                self._col = self._chroma.get_or_create_collection(
                    "molecules", metadata={"hnsw:space": "cosine"}
                )
            except ImportError:
                self._persistent = False  # graceful fallback

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, name: str, embedding: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        """Add one molecule to the store.

        Parameters
        ----------
        name:
            Unique identifier (SMILES, CID, or common name).
        embedding:
            Float32 numpy array (e.g. 2048-dim Morgan fingerprint).
        metadata:
            Arbitrary JSON-serialisable dict stored alongside the vector.
        """
        meta = metadata or {}
        norm = embedding / (np.linalg.norm(embedding) + 1e-9)
        self._names.append(name)
        self._embeddings.append(norm)
        self._metadata.append(meta)

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

        Returns
        -------
        list[dict]
            Each entry has keys ``name``, ``score``, and the stored metadata fields.
        """
        if not self._embeddings:
            return []

        q = query / (np.linalg.norm(query) + 1e-9)
        matrix = np.stack(self._embeddings)  # (N, D)
        scores = matrix @ q  # cosine similarity
        idx = np.argsort(-scores)[:k]
        return [
            {"name": self._names[i], "score": float(scores[i]), **self._metadata[i]}
            for i in idx
        ]

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
