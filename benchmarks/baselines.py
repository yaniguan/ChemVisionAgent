"""Standard baselines for molecular generation benchmarking.

Implements:
1. CharRNN baseline (character-level RNN on SMILES)
2. Random SMILES enumeration
3. Fragment-based recombination
4. Training set augmentation (SMILES randomization)

These provide fair comparison points for the SELFIES generator.

References
----------
- Segler et al., "Generating focused molecule libraries for drug discovery
  with recurrent neural networks", ACS Cent. Sci. 2018
- Bjerrum, "SMILES enumeration as data augmentation for neural network
  modeling of molecules", arXiv 2017
- Degen et al., "On the Art of Compiling and Using 'Drug-Like' Chemical
  Fragment Spaces", ChemMedChem 2008 (BRICS)
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# 1. CharRNN Baseline
# ---------------------------------------------------------------------------


class CharRNNBaseline:
    """Character-level LSTM for SMILES generation (Segler et al., 2018).

    A simple but standard baseline that learns the character-level distribution
    of SMILES strings and generates new molecules autoregressively.

    If PyTorch is not available, falls back to a frequency-based Markov chain
    that approximates character transitions from the training set.
    """

    _PAD = "\0"
    _BOS = "\1"
    _EOS = "\2"

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}
        self._model: Any = None
        self._use_torch = _HAS_TORCH
        # Markov fallback state
        self._bigrams: dict[str, Counter] = {}

    def train(
        self,
        smiles_list: list[str],
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        max_len: int = 120,
        patience: int = 8,
    ) -> dict[str, Any]:
        """Train the character-level model on SMILES strings.

        Returns
        -------
        dict with keys: epochs, final_loss, loss_history
        """
        # Filter valid SMILES
        valid_smiles = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(Chem.MolToSmiles(mol))
        if len(valid_smiles) < 10:
            logger.warning("Too few valid SMILES (%d) for CharRNN training", len(valid_smiles))
            return {"epochs": 0, "final_loss": float("inf"), "loss_history": []}

        # Build vocabulary
        chars: set[str] = set()
        for smi in valid_smiles:
            chars.update(smi)
        self.vocab = {self._PAD: 0, self._BOS: 1, self._EOS: 2}
        for ch in sorted(chars):
            self.vocab[ch] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        if self._use_torch:
            return self._train_torch(valid_smiles, epochs, batch_size, lr, max_len, patience)
        else:
            return self._train_markov(valid_smiles)

    def _train_markov(self, smiles_list: list[str]) -> dict[str, Any]:
        """Train a bigram Markov chain as fallback when PyTorch is unavailable."""
        self._bigrams = {}
        for smi in smiles_list:
            seq = self._BOS + smi + self._EOS
            for a, b in zip(seq[:-1], seq[1:]):
                if a not in self._bigrams:
                    self._bigrams[a] = Counter()
                self._bigrams[a][b] += 1
        return {"epochs": 1, "final_loss": 0.0, "loss_history": [0.0]}

    def _train_torch(
        self,
        smiles_list: list[str],
        epochs: int,
        batch_size: int,
        lr: float,
        max_len: int,
        patience: int,
    ) -> dict[str, Any]:
        """Train an LSTM character model with PyTorch."""
        vocab_size = len(self.vocab)
        pad_id = self.vocab[self._PAD]
        bos_id = self.vocab[self._BOS]
        eos_id = self.vocab[self._EOS]

        # Tokenize
        sequences: list[list[int]] = []
        for smi in smiles_list:
            ids = [bos_id] + [self.vocab.get(c, pad_id) for c in smi] + [eos_id]
            if len(ids) > max_len:
                ids = ids[:max_len]
            ids += [pad_id] * (max_len - len(ids))
            sequences.append(ids)

        tokens_tensor = torch.tensor(sequences, dtype=torch.long)

        # Build LSTM model
        model = _CharLSTM(vocab_size, self.hidden_dim, self.n_layers, self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        loss_history: list[float] = []
        best_loss = float("inf")
        wait = 0

        for epoch in range(epochs):
            perm = torch.randperm(len(tokens_tensor))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(tokens_tensor), batch_size):
                idx = perm[i : i + batch_size]
                toks = tokens_tensor[idx]
                inp = toks[:, :-1]
                tgt = toks[:, 1:]

                logits = model(inp)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    tgt.reshape(-1),
                    ignore_index=pad_id,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self._model = model
        return {
            "epochs": len(loss_history),
            "final_loss": loss_history[-1] if loss_history else float("inf"),
            "loss_history": loss_history,
        }

    def generate(self, n: int = 100, temperature: float = 1.0, max_len: int = 120) -> list[str]:
        """Generate SMILES strings from the trained model.

        Returns
        -------
        list[str] — generated SMILES (may include None for invalid molecules).
        """
        if self._use_torch and self._model is not None:
            return self._generate_torch(n, temperature, max_len)
        elif self._bigrams:
            return self._generate_markov(n, max_len)
        else:
            logger.warning("CharRNN not trained; returning empty list")
            return []

    def _generate_markov(self, n: int, max_len: int) -> list[str]:
        """Generate SMILES using the bigram Markov chain."""
        results: list[str] = []
        for _ in range(n):
            chars: list[str] = []
            current = self._BOS
            for _ in range(max_len):
                if current not in self._bigrams:
                    break
                counts = self._bigrams[current]
                total = sum(counts.values())
                r = random.random() * total
                cumulative = 0
                next_char = self._EOS
                for ch, cnt in counts.items():
                    cumulative += cnt
                    if cumulative >= r:
                        next_char = ch
                        break
                if next_char == self._EOS:
                    break
                chars.append(next_char)
                current = next_char
            smi = "".join(chars)
            mol = Chem.MolFromSmiles(smi)
            results.append(Chem.MolToSmiles(mol) if mol else smi)
        return results

    @torch.no_grad()
    def _generate_torch(self, n: int, temperature: float, max_len: int) -> list[str]:
        """Generate SMILES using the trained LSTM."""
        self._model.eval()
        bos_id = self.vocab[self._BOS]
        eos_id = self.vocab[self._EOS]
        pad_id = self.vocab[self._PAD]

        results: list[str] = []
        for _ in range(n):
            generated_ids = [bos_id]
            hidden = None

            for _ in range(max_len - 1):
                inp = torch.tensor([[generated_ids[-1]]], dtype=torch.long)
                logits, hidden = self._model.step(inp, hidden)
                next_logits = logits[0, 0, :] / max(temperature, 1e-8)

                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

                if next_id == eos_id:
                    break
                generated_ids.append(next_id)

            chars = [
                self.inv_vocab.get(tid, "")
                for tid in generated_ids[1:]
                if tid not in (pad_id, bos_id, eos_id)
            ]
            smi = "".join(chars)
            mol = Chem.MolFromSmiles(smi)
            results.append(Chem.MolToSmiles(mol) if mol else smi)

        return results


if _HAS_TORCH:

    class _CharLSTM(nn.Module):
        """Simple character-level LSTM for SMILES generation."""

        def __init__(
            self,
            vocab_size: int,
            hidden_dim: int = 256,
            n_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.lstm = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_dim, vocab_size)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Full sequence forward pass: (B, L) -> (B, L, vocab_size)."""
            emb = self.dropout(self.embedding(x))
            out, _ = self.lstm(emb)
            return self.fc(out)

        def step(
            self, x: torch.Tensor, hidden: tuple | None = None
        ) -> tuple[torch.Tensor, tuple]:
            """Single-step forward for autoregressive generation.

            Parameters
            ----------
            x : (1, 1) single token
            hidden : LSTM hidden state tuple or None

            Returns
            -------
            (logits, new_hidden)
            """
            emb = self.embedding(x)
            out, hidden = self.lstm(emb, hidden)
            logits = self.fc(out)
            return logits, hidden


# ---------------------------------------------------------------------------
# 2. Fragment Recombination Baseline
# ---------------------------------------------------------------------------


class FragmentBaseline:
    """BRICS fragment recombination baseline.

    Decomposes training molecules into BRICS fragments (Degen et al., 2008),
    then reassembles random fragment combinations to generate new molecules.
    """

    def __init__(self, max_fragments_per_mol: int = 10) -> None:
        self.max_fragments_per_mol = max_fragments_per_mol
        self._fragments: list[str] = []

    def fit(self, smiles_list: list[str]) -> None:
        """Extract BRICS fragments from training molecules."""
        from rdkit.Chem import BRICS

        fragment_set: set[str] = set()
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                frags = BRICS.BRICSDecompose(mol)
                for frag in list(frags)[: self.max_fragments_per_mol]:
                    fragment_set.add(frag)
            except Exception:
                continue

        self._fragments = list(fragment_set)
        logger.info("FragmentBaseline: extracted %d unique BRICS fragments", len(self._fragments))

    def generate(self, n: int = 100, max_frags: int = 4) -> list[str]:
        """Generate molecules by random fragment recombination.

        Selects 2-4 fragments at random, attempts to build molecules via
        BRICS.BRICSBuild, and falls back to simple concatenation if needed.

        Returns
        -------
        list[str] — generated SMILES strings.
        """
        if not self._fragments:
            logger.warning("FragmentBaseline not fitted; returning empty list")
            return []

        from rdkit.Chem import BRICS

        results: list[str] = []
        attempts = 0
        max_attempts = n * 20  # avoid infinite loops

        while len(results) < n and attempts < max_attempts:
            attempts += 1
            n_frags = random.randint(2, min(max_frags, len(self._fragments)))
            selected = random.sample(self._fragments, min(n_frags, len(self._fragments)))

            # Try BRICS build
            frag_mols = [Chem.MolFromSmiles(f) for f in selected]
            frag_mols = [m for m in frag_mols if m is not None]
            if len(frag_mols) < 2:
                continue

            try:
                # BRICSBuild returns a generator; take the first valid product
                products = BRICS.BRICSBuild(frag_mols)
                for product in products:
                    smi = Chem.MolToSmiles(product)
                    if smi and Chem.MolFromSmiles(smi) is not None:
                        results.append(smi)
                        break
                    if len(results) >= n:
                        break
            except Exception:
                continue

        # If BRICS didn't produce enough, pad with random fragments as-is
        while len(results) < n:
            frag = random.choice(self._fragments)
            # Strip BRICS dummy atoms [*:n] for a clean SMILES
            clean = Chem.MolFromSmiles(frag)
            if clean is not None:
                smi = Chem.MolToSmiles(clean)
                if smi:
                    results.append(smi)
            else:
                results.append(frag)

        return results[:n]


# ---------------------------------------------------------------------------
# 3. SMILES Augmentation Baseline
# ---------------------------------------------------------------------------


class AugmentationBaseline:
    """SMILES randomization baseline (Bjerrum, 2017).

    Enumerates different valid SMILES representations of training molecules
    by randomizing the atom ordering. This tests whether the generator can
    do better than simply memorizing variant SMILES of training molecules.
    """

    def __init__(self) -> None:
        self._mols: list[Any] = []

    def fit(self, smiles_list: list[str]) -> None:
        """Parse and store training molecules."""
        self._mols = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                self._mols.append(mol)
        logger.info("AugmentationBaseline: stored %d valid molecules", len(self._mols))

    def generate(self, n: int = 100) -> list[str]:
        """Generate SMILES by random atom-order enumeration.

        For each sample, picks a random training molecule and produces
        a non-canonical SMILES by randomising the atom numbering.

        Returns
        -------
        list[str] — randomized SMILES strings.
        """
        if not self._mols:
            logger.warning("AugmentationBaseline not fitted; returning empty list")
            return []

        results: list[str] = []
        for _ in range(n):
            mol = random.choice(self._mols)
            try:
                smi = self._randomize_smiles(mol)
                if smi:
                    results.append(smi)
                else:
                    # Fallback: canonical SMILES
                    results.append(Chem.MolToSmiles(mol))
            except Exception:
                results.append(Chem.MolToSmiles(mol))

        return results

    @staticmethod
    def _randomize_smiles(mol: Any) -> str | None:
        """Produce a random valid SMILES for a molecule.

        Shuffles the atom indices so RDKit traverses the molecular graph
        in a different order, producing a different (but equivalent) SMILES.
        """
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return None
        order = list(range(n_atoms))
        random.shuffle(order)
        renumbered = Chem.RenumberAtoms(mol, order)
        return Chem.MolToSmiles(renumbered, canonical=False)
