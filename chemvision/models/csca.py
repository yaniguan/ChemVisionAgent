"""CSCA — Characterization-Structure Contrastive Alignment.

Novel contribution: learn a shared latent space between molecular structure
descriptors and physicochemical property vectors via contrastive learning.

Architecture
------------
  MoleculeEncoder: Morgan FP (2048) → MLP → latent (128)
  PropertyEncoder: property vector (D) → MLP → latent (128)
  InfoNCE loss:    pull matched pairs together, push mismatched apart

After training, the shared latent space enables:
  - Zero-shot property prediction from structure alone
  - Structure retrieval given target property profile
  - Interpolation between structures in latent space

This is analogous to MoleculeSTM (Nature MI 2023) but aligns
structure ↔ property instead of structure ↔ text.

Training data: PubChem molecules with RDKit-computed descriptors.
No images needed for the base model — image encoder added in Phase 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@dataclass
class CSCAConfig:
    """Configuration for CSCA model."""

    fp_dim: int = 2048              # Morgan fingerprint dimension
    prop_dim: int = 8               # Number of property features
    latent_dim: int = 128           # Shared latent space dimension
    hidden_dim: int = 256           # MLP hidden layer size
    temperature: float = 0.07      # InfoNCE temperature (CLIP default)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42


@dataclass
class CSCATrainResult:
    """Training result from CSCA."""

    epochs: int = 0
    final_loss: float = float("inf")
    loss_history: list[float] = field(default_factory=list)
    val_retrieval_acc: float = 0.0  # top-1 retrieval accuracy on val set
    converged: bool = False


if _HAS_TORCH:

    class _MLP(nn.Module):
        """2-layer MLP with LayerNorm and dropout."""

        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class CSCAModel(nn.Module):
        """Characterization-Structure Contrastive Alignment model.

        Two separate MLP encoders project fingerprints and property vectors
        into a shared latent space. InfoNCE loss aligns matched pairs.

        Example
        -------
        >>> model = CSCAModel(CSCAConfig())
        >>> fp = torch.randn(32, 2048)     # batch of 32 fingerprints
        >>> props = torch.randn(32, 8)     # matching property vectors
        >>> loss = model(fp, props)
        >>> loss.backward()
        """

        def __init__(self, config: CSCAConfig) -> None:
            super().__init__()
            self.config = config
            self.fp_encoder = _MLP(config.fp_dim, config.hidden_dim, config.latent_dim, config.dropout)
            self.prop_encoder = _MLP(config.prop_dim, config.hidden_dim, config.latent_dim, config.dropout)
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / config.temperature)))

        def encode_fingerprint(self, fp: torch.Tensor) -> torch.Tensor:
            """Encode fingerprints to L2-normalised latent vectors."""
            z = self.fp_encoder(fp)
            return F.normalize(z, dim=-1)

        def encode_properties(self, props: torch.Tensor) -> torch.Tensor:
            """Encode property vectors to L2-normalised latent vectors."""
            z = self.prop_encoder(props)
            return F.normalize(z, dim=-1)

        def forward(self, fp: torch.Tensor, props: torch.Tensor) -> torch.Tensor:
            """Compute symmetric InfoNCE loss.

            Parameters
            ----------
            fp : (B, fp_dim) fingerprint tensor
            props : (B, prop_dim) property tensor

            Returns
            -------
            Scalar loss (lower = better alignment).
            """
            z_fp = self.encode_fingerprint(fp)
            z_prop = self.encode_properties(props)

            # Scaled cosine similarity matrix (B, B)
            logit_scale = self.logit_scale.exp().clamp(max=100)
            logits = logit_scale * z_fp @ z_prop.T

            # Symmetric cross-entropy (CLIP-style)
            labels = torch.arange(len(fp), device=fp.device)
            loss_fp = F.cross_entropy(logits, labels)
            loss_prop = F.cross_entropy(logits.T, labels)
            return (loss_fp + loss_prop) / 2

        def similarity(self, fp: torch.Tensor, props: torch.Tensor) -> torch.Tensor:
            """Cosine similarity between fingerprint and property embeddings."""
            z_fp = self.encode_fingerprint(fp)
            z_prop = self.encode_properties(props)
            return z_fp @ z_prop.T

else:
    # CPU-only fallback: no PyTorch
    class CSCAModel:  # type: ignore[no-redef]
        def __init__(self, config: CSCAConfig) -> None:
            raise ImportError("CSCAModel requires PyTorch. Install with: pip install torch")


class CSCATrainer:
    """Train and evaluate the CSCA model.

    Example
    -------
    >>> trainer = CSCATrainer(CSCAConfig())
    >>> # fps: (N, 2048) fingerprints, props: (N, 8) property vectors
    >>> result = trainer.train(fps, props, epochs=50, batch_size=64)
    >>> result.converged
    True
    >>> # Retrieve: given a property vector, find closest fingerprint
    >>> query_props = props[:1]
    >>> top_k = trainer.retrieve(query_props, fps, k=5)
    """

    def __init__(self, config: CSCAConfig | None = None) -> None:
        if not _HAS_TORCH:
            raise ImportError("CSCATrainer requires PyTorch")
        self.config = config or CSCAConfig()
        self._seed()
        self.model = CSCAModel(self.config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _seed(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def train(
        self,
        fps: np.ndarray,
        props: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        val_split: float = 0.1,
        patience: int = 15,
        verbose: bool = False,
    ) -> CSCATrainResult:
        """Train the CSCA model on fingerprint ↔ property pairs.

        Parameters
        ----------
        fps : (N, fp_dim) Morgan fingerprints as float32
        props : (N, prop_dim) property vectors as float32
        epochs : max training epochs
        batch_size : mini-batch size
        val_split : fraction held out for validation
        patience : early stopping patience (epochs without improvement)
        verbose : print loss every 10 epochs

        Returns
        -------
        CSCATrainResult with loss history and retrieval accuracy.
        """
        # Normalise properties to zero-mean, unit-variance
        prop_mean = props.mean(axis=0)
        prop_std = props.std(axis=0) + 1e-8
        props_norm = (props - prop_mean) / prop_std

        # Train/val split
        n = len(fps)
        n_val = max(int(n * val_split), batch_size)
        rng = np.random.RandomState(self.config.seed)
        perm = rng.permutation(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        fp_train = torch.tensor(fps[train_idx], dtype=torch.float32)
        prop_train = torch.tensor(props_norm[train_idx], dtype=torch.float32)
        fp_val = torch.tensor(fps[val_idx], dtype=torch.float32)
        prop_val = torch.tensor(props_norm[val_idx], dtype=torch.float32)

        result = CSCATrainResult()
        best_loss = float("inf")
        wait = 0

        self.model.train()
        for epoch in range(epochs):
            # Shuffle training data
            perm_t = torch.randperm(len(fp_train))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(fp_train), batch_size):
                idx = perm_t[i:i + batch_size]
                if len(idx) < 4:  # need at least 4 for contrastive
                    continue
                fp_batch = fp_train[idx]
                prop_batch = prop_train[idx]

                self.optimizer.zero_grad()
                loss = self.model(fp_batch, prop_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            result.loss_history.append(avg_loss)

            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        result.epochs = len(result.loss_history)
        result.final_loss = result.loss_history[-1] if result.loss_history else float("inf")
        result.converged = result.final_loss < result.loss_history[0] * 0.5 if result.loss_history else False

        # Validation: top-1 retrieval accuracy
        self.model.eval()
        with torch.no_grad():
            sim = self.model.similarity(fp_val, prop_val)  # (n_val, n_val)
            top1_fp = sim.argmax(dim=1)  # for each fp, which prop is closest?
            correct = (top1_fp == torch.arange(len(fp_val))).float().mean()
            result.val_retrieval_acc = float(correct)

        # Store normalisation params for inference
        self._prop_mean = prop_mean
        self._prop_std = prop_std

        return result

    def retrieve(
        self,
        query_props: np.ndarray,
        candidate_fps: np.ndarray,
        k: int = 5,
    ) -> list[list[int]]:
        """Given property queries, retrieve top-k closest fingerprints.

        Parameters
        ----------
        query_props : (Q, prop_dim) property queries
        candidate_fps : (N, fp_dim) candidate fingerprints
        k : number of results per query

        Returns
        -------
        list of lists of indices into candidate_fps, length Q × k
        """
        props_norm = (query_props - self._prop_mean) / self._prop_std

        self.model.eval()
        with torch.no_grad():
            q = torch.tensor(props_norm, dtype=torch.float32)
            c = torch.tensor(candidate_fps, dtype=torch.float32)
            sim = self.model.similarity(c, q).T  # (Q, N)
            topk = sim.topk(min(k, sim.shape[1]), dim=1)
        return topk.indices.tolist()

    def encode(self, fps: np.ndarray) -> np.ndarray:
        """Encode fingerprints to latent space (for downstream use)."""
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode_fingerprint(torch.tensor(fps, dtype=torch.float32))
        return z.numpy()
