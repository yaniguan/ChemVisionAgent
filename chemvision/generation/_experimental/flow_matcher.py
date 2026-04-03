"""EXPERIMENTAL: Flow Matcher for molecular fingerprint space.

WARNING -- QUARANTINED -- This module generates fingerprint vectors that cannot be
decoded back to valid molecules. It produces 0% validity in benchmarks.
Retained for research reference only. Do NOT use in production pipelines.

To re-enter the benchmark pipeline, this needs:
  1. A latent-space representation with a trained decoder
  2. Validity-preserving decoding (e.g., SELFIES decoder on latent codes)
  3. Demonstrated validity > 70% on ZINC250K
"""

from __future__ import annotations

import math
import logging

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@dataclass
class FlowMatcherConfig:
    """Configuration for the Conditional Flow Matcher."""

    fp_dim: int = 2048              # fingerprint dimension
    cond_dim: int = 8               # conditioning vector dimension (properties)
    hidden_dim: int = 512           # network hidden size
    n_layers: int = 3               # number of residual blocks
    sigma_min: float = 1e-4         # minimum noise scale
    learning_rate: float = 5e-4
    seed: int = 42


@dataclass
class FlowMatcherTrainResult:
    """Training output."""

    epochs: int = 0
    final_loss: float = float("inf")
    loss_history: list[float] = field(default_factory=list)
    converged: bool = False


@dataclass
class GeneratedMolecule:
    """One generated fingerprint with its conditioning."""

    fingerprint: np.ndarray         # (fp_dim,) continuous
    binary_fingerprint: np.ndarray  # (fp_dim,) thresholded to {0, 1}
    condition: np.ndarray           # (cond_dim,) target properties used
    log_likelihood: float = 0.0     # approximate from ODE trace


if _HAS_TORCH:

    class _ResBlock(nn.Module):
        """Residual block with time and condition injection."""

        def __init__(self, dim: int, cond_dim: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.linear1 = nn.Linear(dim, dim)
            self.linear2 = nn.Linear(dim, dim)
            self.time_emb = nn.Linear(1, dim)
            self.cond_emb = nn.Linear(cond_dim, dim)
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            h = self.linear1(h)
            h = h + self.time_emb(t) + self.cond_emb(c)
            h = self.act(h)
            h = self.linear2(h)
            return x + h  # residual

    class _VectorFieldNet(nn.Module):
        """Neural network approximating the conditional vector field v_theta(x, t, c).

        Architecture: input projection -> N residual blocks -> output projection.
        """

        def __init__(self, config: FlowMatcherConfig) -> None:
            super().__init__()
            self.input_proj = nn.Linear(config.fp_dim, config.hidden_dim)
            self.blocks = nn.ModuleList([
                _ResBlock(config.hidden_dim, config.cond_dim)
                for _ in range(config.n_layers)
            ])
            self.output_proj = nn.Linear(config.hidden_dim, config.fp_dim)

        def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            """Predict the vector field at (x, t) conditioned on c.

            Parameters
            ----------
            x : (B, fp_dim) noisy fingerprint
            t : (B, 1) time step
            c : (B, cond_dim) conditioning
            """
            h = self.input_proj(x)
            for block in self.blocks:
                h = block(h, t, c)
            return self.output_proj(h)

    class ConditionalFlowMatcher:
        """Trainable conditional flow matching model for molecular generation.

        Example
        -------
        >>> cfm = ConditionalFlowMatcher(FlowMatcherConfig(fp_dim=2048, cond_dim=8))
        >>> result = cfm.train(fps, props, epochs=50)
        >>> generated = cfm.sample(target_props, n_samples=10)
        >>> generated[0].binary_fingerprint.shape
        (2048,)
        """

        def __init__(self, config: FlowMatcherConfig | None = None) -> None:
            self.config = config or FlowMatcherConfig()
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            self.net = _VectorFieldNet(self.config)
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(), lr=self.config.learning_rate
            )
            self._fp_mean: np.ndarray | None = None
            self._fp_std: np.ndarray | None = None
            self._cond_mean: np.ndarray | None = None
            self._cond_std: np.ndarray | None = None

        def train(
            self,
            fps: np.ndarray,
            conditions: np.ndarray,
            epochs: int = 100,
            batch_size: int = 64,
            patience: int = 20,
            verbose: bool = False,
        ) -> FlowMatcherTrainResult:
            """Train the flow matcher on fingerprint <-> property pairs.

            Parameters
            ----------
            fps : (N, fp_dim) molecular fingerprints
            conditions : (N, cond_dim) target property vectors
            """
            # Normalise inputs
            self._fp_mean = fps.mean(axis=0)
            self._fp_std = fps.std(axis=0) + 1e-8
            self._cond_mean = conditions.mean(axis=0)
            self._cond_std = conditions.std(axis=0) + 1e-8

            x1 = torch.tensor((fps - self._fp_mean) / self._fp_std, dtype=torch.float32)
            c = torch.tensor((conditions - self._cond_mean) / self._cond_std, dtype=torch.float32)

            result = FlowMatcherTrainResult()
            best_loss = float("inf")
            wait = 0
            sigma = self.config.sigma_min

            self.net.train()
            for epoch in range(epochs):
                perm = torch.randperm(len(x1))
                epoch_loss = 0.0
                n_batches = 0

                for i in range(0, len(x1), batch_size):
                    idx = perm[i:i + batch_size]
                    if len(idx) < 4:
                        continue
                    x1_batch = x1[idx]
                    c_batch = c[idx]
                    B = len(x1_batch)

                    # Sample time uniformly
                    t = torch.rand(B, 1)

                    # Sample noise (source distribution)
                    x0 = torch.randn_like(x1_batch)

                    # Optimal transport conditional path:
                    # x_t = (1 - (1-sigma)*t) * x0 + t * x1
                    mu_t = t * x1_batch + (1 - (1 - sigma) * t) * x0

                    # Target vector field: u_t = x1 - (1-sigma) * x0
                    target = x1_batch - (1 - sigma) * x0

                    # Predict
                    self.optimizer.zero_grad()
                    pred = self.net(mu_t, t, c_batch)

                    # MSE loss on vector field
                    loss = F.mse_loss(pred, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                result.loss_history.append(avg_loss)

                if verbose and epoch % 10 == 0:
                    logger.info("Epoch %3d | loss=%.4f", epoch, avg_loss)

                if avg_loss < best_loss - 1e-5:
                    best_loss = avg_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

            result.epochs = len(result.loss_history)
            result.final_loss = result.loss_history[-1] if result.loss_history else float("inf")
            result.converged = len(result.loss_history) > 5 and result.final_loss < result.loss_history[0] * 0.8

            return result

        @torch.no_grad()
        def sample(
            self,
            target_conditions: np.ndarray,
            n_steps: int = 20,
            threshold: float = 0.5,
        ) -> list[GeneratedMolecule]:
            """Generate molecules by integrating the learned vector field.

            Parameters
            ----------
            target_conditions : (Q, cond_dim) desired property vectors
            n_steps : Euler integration steps (more = better quality)
            threshold : binarisation threshold for fingerprint bits

            Returns
            -------
            List of GeneratedMolecule with continuous and binary fingerprints.
            """
            self.net.eval()
            c_norm = (target_conditions - self._cond_mean) / self._cond_std
            c = torch.tensor(c_norm, dtype=torch.float32)
            Q = len(c)

            # Start from noise
            x = torch.randn(Q, self.config.fp_dim)
            dt = 1.0 / n_steps

            # Euler integration: dx/dt = v_theta(x, t, c)
            for step in range(n_steps):
                t = torch.full((Q, 1), step * dt)
                v = self.net(x, t, c)
                x = x + v * dt

            # Denormalise
            x_np = x.numpy() * self._fp_std + self._fp_mean

            results: list[GeneratedMolecule] = []
            for i in range(Q):
                fp_cont = x_np[i]
                fp_bin = (fp_cont > threshold).astype(np.float32)
                results.append(GeneratedMolecule(
                    fingerprint=fp_cont,
                    binary_fingerprint=fp_bin,
                    condition=target_conditions[i],
                ))
            return results

else:
    class ConditionalFlowMatcher:  # type: ignore[no-redef]
        def __init__(self, config: FlowMatcherConfig | None = None) -> None:
            raise ImportError("ConditionalFlowMatcher requires PyTorch")
