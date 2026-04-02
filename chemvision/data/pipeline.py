"""Structured data pipeline: collect, version, store, and serve scientific data.

Architecture
------------
  Raw input (images, SMILES, spectra, papers)
      ↓
  DataIngestor  (validate, hash, assign provenance)
      ↓
  DataStore     (Parquet backend with schema enforcement)
      ↓
  DataLoader    (filtered, batched, reproducible splits)

Key features:
  - Content-addressable hashing (SHA-256) for deduplication
  - Schema validation via Pydantic before storage
  - Parquet persistence (columnar, fast, compressible)
  - Train/val/test splits with deterministic seeding
  - Data quality scoring per record
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class DataRecord(BaseModel):
    """One record in the structured data store."""

    record_id: str = ""                     # auto-generated SHA-256 hash
    source: str = ""                        # "pubchem", "synthetic", "literature", "user_upload"
    domain: str = ""                        # "spectroscopy", "microscopy", "molecular", "crystal"
    created_at: str = ""                    # ISO timestamp
    version: int = 1

    # Molecular data
    smiles: str | None = None
    inchi_key: str | None = None
    molecular_formula: str | None = None
    molecular_weight: float | None = None

    # Spectral data
    spectrum_type: str | None = None        # "XRD", "Raman", "XPS", "IR", "NMR"
    peak_positions: list[float] = Field(default_factory=list)
    peak_intensities: list[float] = Field(default_factory=list)

    # Crystal data
    space_group: str | None = None
    lattice_params: dict[str, float] = Field(default_factory=dict)  # a, b, c, alpha, beta, gamma

    # Image reference
    image_path: str | None = None
    image_hash: str | None = None           # SHA-256 of image bytes

    # Quality
    quality_score: float = 0.0              # [0, 1] computed by DataQualityScorer
    quality_flags: list[str] = Field(default_factory=list)

    # Provenance
    provenance: dict[str, Any] = Field(default_factory=dict)  # arbitrary metadata

    def compute_id(self) -> str:
        """Deterministic content-addressable hash for deduplication."""
        key_fields = f"{self.smiles}|{self.spectrum_type}|{self.image_hash}|{self.space_group}"
        return hashlib.sha256(key_fields.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------


class DataQualityScorer:
    """Score a DataRecord's quality on a 0–1 scale.

    Checks:
      - Field completeness (30%)
      - Value range validity (30%)
      - Internal consistency (20%)
      - Provenance richness (20%)
    """

    _CRITICAL_FIELDS = ["smiles", "domain", "source"]
    _OPTIONAL_FIELDS = ["molecular_formula", "molecular_weight", "spectrum_type", "space_group"]

    def score(self, record: DataRecord) -> tuple[float, list[str]]:
        """Return (quality_score, quality_flags)."""
        score = 0.0
        flags: list[str] = []

        # 1. Completeness (0.3)
        filled = sum(1 for f in self._CRITICAL_FIELDS if getattr(record, f))
        optional_filled = sum(1 for f in self._OPTIONAL_FIELDS if getattr(record, f))
        completeness = (filled / len(self._CRITICAL_FIELDS)) * 0.5 + \
                       (optional_filled / max(len(self._OPTIONAL_FIELDS), 1)) * 0.5
        score += completeness * 0.3
        if filled < len(self._CRITICAL_FIELDS):
            flags.append(f"missing_critical_fields:{len(self._CRITICAL_FIELDS) - filled}")

        # 2. Value range validity (0.3)
        range_score = 1.0
        if record.molecular_weight is not None:
            if record.molecular_weight <= 0 or record.molecular_weight > 5000:
                range_score -= 0.5
                flags.append("mw_out_of_range")
        if record.quality_score < 0 or record.quality_score > 1:
            range_score -= 0.5
            flags.append("quality_score_invalid")
        # Validate SMILES if present
        if record.smiles:
            try:
                from rdkit import Chem
                if Chem.MolFromSmiles(record.smiles) is None:
                    range_score -= 0.5
                    flags.append("invalid_smiles")
            except ImportError:
                pass
        score += max(range_score, 0) * 0.3

        # 3. Internal consistency (0.2)
        consistency = 1.0
        if record.peak_positions and record.peak_intensities:
            if len(record.peak_positions) != len(record.peak_intensities):
                consistency -= 0.5
                flags.append("peak_count_mismatch")
        score += consistency * 0.2

        # 4. Provenance (0.2)
        prov_score = min(len(record.provenance) / 3, 1.0)  # 3+ fields = full score
        if record.source:
            prov_score = max(prov_score, 0.5)
        score += prov_score * 0.2

        return round(score, 3), flags


# ---------------------------------------------------------------------------
# Data store (Parquet backend)
# ---------------------------------------------------------------------------


class DataStore:
    """Parquet-backed data store with schema enforcement and deduplication.

    Example
    -------
    >>> store = DataStore(Path("data/store"))
    >>> store.ingest(DataRecord(smiles="CCO", domain="molecular", source="pubchem"))
    >>> df = store.query(domain="molecular")
    >>> len(df)
    1
    """

    def __init__(self, store_dir: Path | str) -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._parquet_path = self._dir / "records.parquet"
        self._scorer = DataQualityScorer()
        self._df: pd.DataFrame | None = None

    @property
    def parquet_path(self) -> Path:
        return self._parquet_path

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, record: DataRecord) -> DataRecord:
        """Validate, score, deduplicate, and store a record."""
        # Auto-generate ID and timestamp
        if not record.record_id:
            record.record_id = record.compute_id()
        if not record.created_at:
            record.created_at = datetime.now(timezone.utc).isoformat()

        # Quality scoring
        quality, flags = self._scorer.score(record)
        record.quality_score = quality
        record.quality_flags = flags

        # Append to store
        df = self._load()
        if record.record_id in df["record_id"].values:
            # Update existing record (increment version)
            mask = df["record_id"] == record.record_id
            record.version = int(df.loc[mask, "version"].iloc[0]) + 1
            df = df[~mask]

        new_row = pd.DataFrame([record.model_dump()])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save(df)
        return record

    def ingest_batch(self, records: list[DataRecord]) -> int:
        """Ingest multiple records. Returns count of records stored."""
        for r in records:
            self.ingest(r)
        return len(records)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        domain: str | None = None,
        source: str | None = None,
        min_quality: float = 0.0,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Query records with optional filters."""
        df = self._load()
        if domain:
            df = df[df["domain"] == domain]
        if source:
            df = df[df["source"] == source]
        if min_quality > 0:
            df = df[df["quality_score"] >= min_quality]
        return df.head(limit)

    def get_by_id(self, record_id: str) -> DataRecord | None:
        df = self._load()
        match = df[df["record_id"] == record_id]
        if match.empty:
            return None
        row = match.iloc[0].to_dict()
        return DataRecord(**row)

    def count(self) -> int:
        return len(self._load())

    def stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        df = self._load()
        if df.empty:
            return {"total": 0}
        return {
            "total": len(df),
            "by_domain": df["domain"].value_counts().to_dict() if "domain" in df else {},
            "by_source": df["source"].value_counts().to_dict() if "source" in df else {},
            "mean_quality": float(df["quality_score"].mean()) if "quality_score" in df else 0,
            "quality_distribution": {
                "high (>0.7)": int((df["quality_score"] > 0.7).sum()),
                "medium (0.4-0.7)": int(((df["quality_score"] >= 0.4) & (df["quality_score"] <= 0.7)).sum()),
                "low (<0.4)": int((df["quality_score"] < 0.4).sum()),
            } if "quality_score" in df else {},
        }

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        min_quality: float = 0.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Deterministic train/val/test split.

        Returns
        -------
        (train_df, val_df, test_df)
        """
        df = self._load()
        if min_quality > 0:
            df = df[df["quality_score"] >= min_quality]

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(df))
        n_train = int(len(df) * train_ratio)
        n_val = int(len(df) * val_ratio)

        train = df.iloc[indices[:n_train]]
        val = df.iloc[indices[n_train:n_train + n_val]]
        test = df.iloc[indices[n_train + n_val:]]
        return train, val, test

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        if self._parquet_path.exists():
            self._df = pd.read_parquet(self._parquet_path)
        else:
            # Empty DataFrame with correct column names
            self._df = pd.DataFrame(columns=list(DataRecord.model_fields.keys()))
        return self._df

    def _save(self, df: pd.DataFrame) -> None:
        self._df = df
        # Convert complex columns to JSON strings for Parquet compatibility
        for col in ("peak_positions", "peak_intensities", "lattice_params",
                     "quality_flags", "provenance"):
            if col in df.columns:
                df[col] = df[col].apply(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
        df.to_parquet(self._parquet_path, index=False)
