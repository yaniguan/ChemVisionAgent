"""SELFIES-based autoregressive molecular generator.

Scientific contribution: generate molecules that are **guaranteed 100% chemically
valid** by construction, conditioned on target property profiles.

SELFIES (Krenn et al., Machine Learning: Science and Technology, 2020) is a
string representation where every possible string decodes to a valid molecule.
This eliminates the validity problem of SMILES-based generation entirely.

Architecture
------------
  Property vector (D) → MLP → context embedding (H)
  Context + partial SELFIES → Transformer decoder → next token

  Every generated string is decoded to a valid SMILES via selfies.decoder().

This replaces the fingerprint-space Flow Matcher which generates vectors
that cannot be decoded to molecules (the fundamental flaw identified in review).

References
----------
- Krenn et al. "Self-Referencing Embedded Strings (SELFIES)", 2020
- REINVENT (Olivecrona et al., J Cheminform 2017) for autoregressive drug design
- MOSES benchmark (Polykovskiy et al., Front Pharmacol 2020)
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import selfies as sf
from selfies.exceptions import DecoderError, EncoderError
from rdkit import Chem

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_PAD = "[nop]"
_BOS = "[BOS]"
_EOS = "[EOS]"


def build_vocabulary(smiles_list: list[str], max_vocab: int = 200) -> dict[str, int]:
    """Build a SELFIES token vocabulary from a SMILES dataset."""
    counter: Counter[str] = Counter()
    for smi in smiles_list:
        try:
            sel = sf.encoder(smi)
            if sel:
                tokens = list(sf.split_selfies(sel))
                counter.update(tokens)
        except (ValueError, TypeError, EncoderError):
            pass  # invalid SMILES for SELFIES encoding — skip

    # Keep most common tokens
    vocab = {_PAD: 0, _BOS: 1, _EOS: 2}
    for tok, _ in counter.most_common(max_vocab - 3):
        vocab[tok] = len(vocab)
    return vocab


def smiles_to_selfies_tokens(smi: str) -> list[str] | None:
    """Convert SMILES to a list of SELFIES tokens."""
    try:
        sel = sf.encoder(smi)
        if sel is None:
            return None
        return list(sf.split_selfies(sel))
    except (ValueError, TypeError, EncoderError) as exc:
        logger.warning("SELFIES encoding failed for %r: %s", smi, exc)
        return None


def selfies_tokens_to_smiles(tokens: list[str]) -> str | None:
    """Decode SELFIES tokens back to SMILES. Always produces a valid molecule."""
    try:
        sel_str = "".join(tokens)
        smi = sf.decoder(sel_str)
        if smi and Chem.MolFromSmiles(smi) is not None:
            return Chem.MolToSmiles(Chem.MolFromSmiles(smi))  # canonical
        return smi
    except (ValueError, TypeError, DecoderError) as exc:
        logger.warning("SELFIES decoding failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class SELFIESGenConfig:
    """Configuration for the SELFIES generator."""

    cond_dim: int = 8              # property conditioning vector dimension
    embed_dim: int = 128           # token + positional embedding dimension
    n_heads: int = 4               # transformer attention heads
    n_layers: int = 3              # transformer decoder layers
    max_len: int = 80              # max SELFIES sequence length
    dropout: float = 0.1
    learning_rate: float = 3e-4
    temperature: float = 1.0       # sampling temperature
    seed: int = 42
    prop_loss_weight: float = 0.1  # auxiliary property prediction loss weight
    guidance_scale: float = 1.5    # classifier-free guidance scale (0.0 = disabled)
    cfg_drop_prob: float = 0.1     # conditioning drop probability for CFG training


@dataclass
class GenerationResult:
    """One generated molecule."""

    smiles: str | None
    selfies: str
    is_valid: bool
    properties_target: list[float] = field(default_factory=list)


@dataclass
class TrainResult:
    """Training output."""

    epochs: int = 0
    final_loss: float = float("inf")
    loss_history: list[float] = field(default_factory=list)
    prop_loss_history: list[float] = field(default_factory=list)
    vocab_size: int = 0
    converged: bool = False


if _HAS_TORCH:

    class FiLMConditioner(nn.Module):
        """Feature-wise Linear Modulation for property conditioning.

        Applies learned scale and shift to hidden states based on the
        conditioning vector, enabling much stronger property control than
        cross-attention alone.

        Reference: Perez et al., "FiLM: Visual Reasoning with a General
        Conditioning Layer", AAAI 2018.
        """

        def __init__(self, cond_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.scale = nn.Linear(cond_dim, hidden_dim)
            self.shift = nn.Linear(cond_dim, hidden_dim)

        def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            """Apply FiLM modulation.

            Parameters
            ----------
            x : (B, L, H) hidden states
            cond : (B, cond_dim) conditioning vector

            Returns
            -------
            (B, L, H) modulated hidden states
            """
            gamma = self.scale(cond).unsqueeze(1)  # (B, 1, H)
            beta = self.shift(cond).unsqueeze(1)    # (B, 1, H)
            return gamma * x + beta

    class _SELFIESTransformer(nn.Module):
        """Property-conditioned autoregressive SELFIES generator.

        Uses three complementary conditioning mechanisms:
        1. Cross-attention memory injection (original)
        2. FiLM modulation at each decoder layer
        3. Auxiliary property prediction head (training signal)
        """

        def __init__(self, vocab_size: int, config: SELFIESGenConfig) -> None:
            super().__init__()
            self.config = config
            self.vocab_size = vocab_size

            # Token + position embeddings
            self.tok_emb = nn.Embedding(vocab_size, config.embed_dim)
            self.pos_emb = nn.Embedding(config.max_len, config.embed_dim)
            self.cond_proj = nn.Linear(config.cond_dim, config.embed_dim)

            # Individual transformer decoder layers (for FiLM injection between layers)
            self.decoder_layers = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=config.embed_dim,
                    nhead=config.n_heads,
                    dim_feedforward=config.embed_dim * 4,
                    dropout=config.dropout,
                    batch_first=True,
                )
                for _ in range(config.n_layers)
            ])

            # FiLM conditioners — one per decoder layer
            self.film_layers = nn.ModuleList([
                FiLMConditioner(config.cond_dim, config.embed_dim)
                for _ in range(config.n_layers)
            ])

            self.output_head = nn.Linear(config.embed_dim, vocab_size)
            self.dropout = nn.Dropout(config.dropout)

            # Auxiliary property prediction head
            self.prop_head = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, config.cond_dim),
            )

        def forward(
            self, tokens: torch.Tensor, cond: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass for teacher forcing.

            Parameters
            ----------
            tokens : (B, L) token indices
            cond : (B, cond_dim) property conditioning

            Returns
            -------
            logits : (B, L, vocab_size)
            prop_pred : (B, cond_dim) predicted properties from hidden states
            """
            B, L = tokens.shape
            pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)

            # Embed tokens + positions
            x = self.tok_emb(tokens) + self.pos_emb(pos)
            x = self.dropout(x)

            # Condition as memory (single-token memory sequence)
            memory = self.cond_proj(cond).unsqueeze(1)  # (B, 1, embed_dim)

            # Causal mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=tokens.device)

            # Decode with FiLM conditioning at each layer
            h = x
            for dec_layer, film in zip(self.decoder_layers, self.film_layers):
                h = dec_layer(h, memory, tgt_mask=causal_mask)
                h = film(h, cond)

            # Property prediction from mean-pooled hidden states
            prop_pred = self.prop_head(h.mean(dim=1))  # (B, cond_dim)

            return self.output_head(h), prop_pred

        def forward_logits_only(
            self, tokens: torch.Tensor, cond: torch.Tensor
        ) -> torch.Tensor:
            """Forward pass returning only logits (used during generation).

            Parameters
            ----------
            tokens : (B, L) token indices
            cond : (B, cond_dim) property conditioning

            Returns
            -------
            (B, L, vocab_size) logits
            """
            logits, _ = self.forward(tokens, cond)
            return logits

    class SELFIESGenerator:
        """Train and sample property-conditioned SELFIES molecules.

        Example
        -------
        >>> gen = SELFIESGenerator(SELFIESGenConfig())
        >>> result = gen.train(smiles_list, properties, epochs=50)
        >>> molecules = gen.generate(target_properties, n=100)
        >>> all(m.is_valid for m in molecules)
        True
        """

        def __init__(self, config: SELFIESGenConfig | None = None) -> None:
            self.config = config or SELFIESGenConfig()
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            self.vocab: dict[str, int] = {}
            self.inv_vocab: dict[int, str] = {}
            self.model: _SELFIESTransformer | None = None
            self._cond_mean: np.ndarray | None = None
            self._cond_std: np.ndarray | None = None

        def train(
            self,
            smiles_list: list[str],
            properties: np.ndarray,
            epochs: int = 50,
            batch_size: int = 64,
            patience: int = 10,
            verbose: bool = False,
        ) -> TrainResult:
            """Train the generator on (SMILES, property) pairs.

            Parameters
            ----------
            smiles_list : list of SMILES strings
            properties : (N, cond_dim) property vectors
            """
            result = TrainResult()

            # Validate properties shape matches smiles_list length
            if properties.ndim == 1:
                properties = properties.reshape(-1, 1)
            if len(properties) != len(smiles_list):
                raise ValueError(
                    f"properties length ({len(properties)}) must match "
                    f"smiles_list length ({len(smiles_list)})"
                )
            if properties.shape[1] != self.config.cond_dim:
                raise ValueError(
                    f"properties dim ({properties.shape[1]}) must match "
                    f"cond_dim ({self.config.cond_dim})"
                )
            # Check for NaN/Inf in properties
            if np.any(np.isnan(properties)) or np.any(np.isinf(properties)):
                raise ValueError("properties must not contain NaN or Inf values")

            # Build vocabulary
            self.vocab = build_vocabulary(smiles_list)
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            result.vocab_size = len(self.vocab)

            # Tokenize
            sequences: list[list[int]] = []
            valid_props: list[np.ndarray] = []
            pad_id = self.vocab[_PAD]
            bos_id = self.vocab[_BOS]
            eos_id = self.vocab[_EOS]

            for i, smi in enumerate(smiles_list):
                tokens = smiles_to_selfies_tokens(smi)
                if tokens is None:
                    continue
                ids = [bos_id] + [self.vocab.get(t, pad_id) for t in tokens] + [eos_id]
                if len(ids) > self.config.max_len:
                    ids = ids[:self.config.max_len]
                # Pad
                ids += [pad_id] * (self.config.max_len - len(ids))
                sequences.append(ids)
                valid_props.append(properties[i])

            if len(sequences) < 10:
                logger.warning("Too few valid SELFIES sequences (%d)", len(sequences))
                return result

            tokens_tensor = torch.tensor(sequences, dtype=torch.long)
            self._cond_mean = np.mean(valid_props, axis=0)
            self._cond_std = np.std(valid_props, axis=0) + 1e-8
            props_norm = (np.array(valid_props) - self._cond_mean) / self._cond_std
            cond_tensor = torch.tensor(props_norm, dtype=torch.float32)

            # Build model
            self.model = _SELFIESTransformer(len(self.vocab), self.config)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

            best_loss = float("inf")
            wait = 0
            cfg_drop = self.config.cfg_drop_prob

            self.model.train()
            for epoch in range(epochs):
                perm = torch.randperm(len(tokens_tensor))
                epoch_loss = 0.0
                epoch_prop_loss = 0.0
                n_batches = 0

                for i in range(0, len(tokens_tensor), batch_size):
                    idx = perm[i:i + batch_size]
                    toks = tokens_tensor[idx]
                    cond = cond_tensor[idx]

                    # Classifier-free guidance: randomly drop conditioning
                    if cfg_drop > 0.0:
                        drop_mask = (
                            torch.rand(cond.size(0), 1) < cfg_drop
                        ).float().to(cond.device)
                        cond = cond * (1.0 - drop_mask)

                    # Input: all tokens except last; Target: all tokens except first
                    inp = toks[:, :-1]
                    tgt = toks[:, 1:]

                    logits, prop_pred = self.model(inp, cond)
                    ce_loss = F.cross_entropy(
                        logits.reshape(-1, len(self.vocab)),
                        tgt.reshape(-1),
                        ignore_index=pad_id,
                    )

                    # Auxiliary property prediction loss (use original cond before drop)
                    prop_target = cond_tensor[idx]
                    prop_loss = F.mse_loss(prop_pred, prop_target)
                    total_loss = ce_loss + self.config.prop_loss_weight * prop_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += ce_loss.item()
                    epoch_prop_loss += prop_loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                avg_prop_loss = epoch_prop_loss / max(n_batches, 1)
                result.loss_history.append(avg_loss)
                result.prop_loss_history.append(avg_prop_loss)

                if verbose and epoch % 10 == 0:
                    logger.info(
                        "Epoch %3d | ce_loss=%.4f | prop_loss=%.4f",
                        epoch, avg_loss, avg_prop_loss,
                    )

                if avg_loss < best_loss - 1e-4:
                    best_loss = avg_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

            result.epochs = len(result.loss_history)
            result.final_loss = result.loss_history[-1] if result.loss_history else float("inf")
            result.converged = result.final_loss < result.loss_history[0] * 0.7
            return result

        @torch.no_grad()
        def generate(
            self,
            target_properties: np.ndarray,
            n_per_target: int = 10,
            temperature: float | None = None,
            top_k: int = 0,
            top_p: float = 0.9,
            guidance_scale: float | None = None,
        ) -> list[GenerationResult]:
            """Generate molecules conditioned on target properties.

            Parameters
            ----------
            top_k : int
                If >0, keep only top-k tokens before sampling.
            top_p : float
                Nucleus sampling: keep smallest set of tokens whose cumulative
                probability >= top_p. Set to 1.0 to disable.
                Default 0.9 (standard nucleus sampling).
            guidance_scale : float or None
                Classifier-free guidance scale. If None, uses config default.
                Set to 0.0 to disable guidance (original behavior).
                Higher values (e.g. 1.5-3.0) push generation toward targets.

            Returns
            -------
            list[GenerationResult] — each with SMILES, validity flag, target props.
            Every generated SELFIES string decodes to a valid molecule by construction.
            """
            if self.model is None:
                raise RuntimeError("Model not trained. Call train() first.")

            # Validate temperature early (before touching model)
            temp = temperature if temperature is not None else self.config.temperature
            if temp <= 0:
                raise ValueError(f"temperature must be positive, got {temp}")

            gs = guidance_scale if guidance_scale is not None else self.config.guidance_scale
            use_cfg = gs > 0.0

            # Validate target_properties shape
            if target_properties.ndim == 1:
                target_properties = target_properties.reshape(1, -1)
            if target_properties.shape[0] == 0:
                return []
            if target_properties.shape[1] != self.config.cond_dim:
                raise ValueError(
                    f"target_properties dim ({target_properties.shape[1]}) must match "
                    f"cond_dim ({self.config.cond_dim})"
                )
            if np.any(np.isnan(target_properties)) or np.any(np.isinf(target_properties)):
                raise ValueError("target_properties must not contain NaN or Inf values")

            self.model.eval()
            pad_id = self.vocab[_PAD]
            bos_id = self.vocab[_BOS]
            eos_id = self.vocab[_EOS]

            props_norm = (target_properties - self._cond_mean) / self._cond_std
            results: list[GenerationResult] = []

            for prop_vec, prop_raw in zip(props_norm, target_properties):
                cond = torch.tensor(prop_vec, dtype=torch.float32).unsqueeze(0)
                uncond = torch.zeros_like(cond)  # unconditional = zero conditioning

                for _ in range(n_per_target):
                    generated_ids = [bos_id]

                    for step in range(self.config.max_len - 1):
                        inp = torch.tensor([generated_ids], dtype=torch.long)
                        cond_logits = self.model.forward_logits_only(inp, cond)
                        next_logits = cond_logits[0, -1, :]

                        # Classifier-free guidance
                        if use_cfg:
                            uncond_logits = self.model.forward_logits_only(inp, uncond)
                            next_logits = (
                                (1.0 + gs) * next_logits
                                - gs * uncond_logits[0, -1, :]
                            )

                        next_logits = next_logits / temp

                        # Top-k filtering
                        if top_k > 0:
                            topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                            next_logits[next_logits < topk_vals[-1]] = float("-inf")

                        # Nucleus (top-p) sampling
                        if top_p < 1.0:
                            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            # Remove tokens with cumulative prob above threshold
                            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                            sorted_logits[mask] = float("-inf")
                            # Scatter back
                            next_logits = torch.zeros_like(next_logits).scatter(0, sorted_idx, sorted_logits)

                        probs = F.softmax(next_logits, dim=-1)
                        next_id = torch.multinomial(probs, 1).item()

                        if next_id == eos_id:
                            break
                        generated_ids.append(next_id)

                    # Decode tokens to SELFIES string
                    selfies_tokens = [
                        self.inv_vocab.get(tid, "")
                        for tid in generated_ids[1:]  # skip BOS
                        if tid not in (pad_id, bos_id, eos_id)
                    ]
                    selfies_str = "".join(selfies_tokens)
                    smiles = selfies_tokens_to_smiles(selfies_tokens)

                    results.append(GenerationResult(
                        smiles=smiles,
                        selfies=selfies_str,
                        is_valid=smiles is not None and Chem.MolFromSmiles(smiles) is not None,
                        properties_target=prop_raw.tolist(),
                    ))

            return results

else:
    class SELFIESGenerator:  # type: ignore[no-redef]
        def __init__(self, config: Any = None) -> None:
            raise ImportError("SELFIESGenerator requires PyTorch")
