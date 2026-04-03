"""Graph Neural Network encoders for molecular graphs.

Implements two architectures for molecular representation learning:

1. **GIN** (Graph Isomorphism Network, Xu et al., ICLR 2019)
   - Sum-aggregation with MLP update, maximally expressive among MPNNs
   - Best for: molecular classification, fingerprint-level tasks

2. **SchNet** (Schutt et al., NeurIPS 2017)
   - Continuous-filter convolution on 3D atomic distances
   - Best for: energy/force prediction, conformer-dependent properties
   - 2D fallback uses shortest-path (hop) distances as proxy

Both produce a fixed-size molecular embedding from a variable-size graph,
suitable as drop-in replacements for Morgan fingerprints.

These operate on RDKit molecular graphs — no torch_geometric dependency
for GIN (pure PyTorch), optional for SchNet with 3D coordinates.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Atom features: atomic number one-hot (common elements)
_ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I

# Feature dimensions (computed from feature vector structure):
# atomic_num one-hot: 10 (9 + other)
# degree one-hot: 7 (0-5 + other)
# formal_charge one-hot: 6 (-2 to +2 + other)
# hybridization one-hot: 6 (5 types + other)
# is_aromatic: 1
# is_in_ring: 1
# num_Hs one-hot: 6 (0-4 + other)
_ATOM_DIM = 10 + 7 + 6 + 6 + 1 + 1 + 6  # = 37

_BOND_DIM = 6  # single, double, triple, aromatic, conjugated, in_ring


def one_hot(value: Any, allowable: list[Any]) -> list[float]:
    """One-hot encode *value* against *allowable* set, with an 'other' bin."""
    encoding = [0.0] * (len(allowable) + 1)
    if value in allowable:
        encoding[allowable.index(value)] = 1.0
    else:
        encoding[-1] = 1.0  # "other"
    return encoding


def atom_features(atom) -> list[float]:
    """Rich atom feature vector (37-d)."""
    from rdkit import Chem

    features: list[float] = []
    # Atomic number one-hot (9 common + other)
    features.extend(one_hot(atom.GetAtomicNum(), _ATOM_LIST))
    # Degree one-hot (0-5+)
    features.extend(one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5]))
    # Formal charge one-hot (-2 to +2)
    features.extend(one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2]))
    # Hybridization one-hot
    features.extend(one_hot(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]))
    # Boolean features
    features.append(float(atom.GetIsAromatic()))
    features.append(float(atom.IsInRing()))
    # Num H (0-4+)
    features.extend(one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))
    return features


def bond_features(bond) -> list[float]:
    """Bond feature vector (6-d)."""
    from rdkit import Chem

    bt = bond.GetBondType()
    return [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def smiles_to_graph(smiles: str) -> dict[str, Any] | None:
    """Convert SMILES to a graph dict with node features, edge index, and edge attributes.

    Returns
    -------
    dict with keys:
      x: (N, atom_dim) node feature matrix
      edge_index: (2, E) edge index (undirected)
      edge_attr: (E, bond_dim) bond feature matrix
      n_atoms: int
    or None if SMILES is invalid.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n = mol.GetNumAtoms()
    if n == 0:
        return None

    # Node features: rich atom features
    x = np.zeros((n, _ATOM_DIM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        x[i] = atom_features(atom)

    # Edge index (undirected) and bond features
    edges: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges.append([i, j])
        edge_attrs.append(bf)
        edges.append([j, i])
        edge_attrs.append(bf)

    if edges:
        edge_index = np.array(edges, dtype=np.int64).T  # (2, E)
        edge_attr = np.array(edge_attrs, dtype=np.float32)  # (E, bond_dim)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, _BOND_DIM), dtype=np.float32)

    # Validate consistency: edge indices must reference valid atoms
    if edge_index.size > 0:
        assert edge_index.max() < n, "edge_index references atom beyond n_atoms"
        assert edge_index.min() >= 0, "edge_index contains negative indices"
    assert x.shape == (n, _ATOM_DIM), f"x shape {x.shape} inconsistent with n_atoms={n}"

    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "n_atoms": n}


def batch_graphs(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch multiple graphs into one big graph with a batch vector."""
    if not graphs:
        return {
            "x": np.zeros((0, _ATOM_DIM), dtype=np.float32),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_attr": np.zeros((0, _BOND_DIM), dtype=np.float32),
            "batch": np.zeros(0, dtype=np.int64),
            "n_graphs": 0,
        }
    xs, edges, edge_attrs, batch = [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        xs.append(g["x"])
        e = g["edge_index"].copy()
        e += offset
        edges.append(e)
        if "edge_attr" in g:
            edge_attrs.append(g["edge_attr"])
        batch.extend([i] * g["n_atoms"])
        offset += g["n_atoms"]

    result: dict[str, Any] = {
        "x": np.vstack(xs).astype(np.float32),
        "edge_index": np.hstack(edges).astype(np.int64) if edges else np.zeros((2, 0), dtype=np.int64),
        "batch": np.array(batch, dtype=np.int64),
        "n_graphs": len(graphs),
    }
    if edge_attrs:
        result["edge_attr"] = np.vstack(edge_attrs).astype(np.float32)
    else:
        result["edge_attr"] = np.zeros((0, _BOND_DIM), dtype=np.float32)
    return result


def _shortest_path_distances(edge_index: np.ndarray, n_atoms: int) -> np.ndarray:
    """Compute all-pairs shortest path distances via BFS on the graph.

    Returns an (n_atoms, n_atoms) matrix of hop counts.
    Unreachable pairs get distance = n_atoms (a large sentinel).
    """
    from collections import deque

    dist = np.full((n_atoms, n_atoms), n_atoms, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)

    # Build adjacency list
    adj: dict[int, list[int]] = {i: [] for i in range(n_atoms)}
    if edge_index.size > 0:
        for k in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, k]), int(edge_index[1, k])
            adj[src].append(dst)

    # BFS from each node
    for start in range(n_atoms):
        visited = {start}
        queue: deque[tuple[int, int]] = deque([(start, 0)])
        while queue:
            node, d = queue.popleft()
            dist[start, node] = d
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, d + 1))

    return dist


if _HAS_TORCH:

    class GINLayer(nn.Module):
        """One GIN layer: aggregate neighbors with sum, update with MLP.

        Optionally incorporates edge (bond) features via an edge MLP.
        """

        def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0, eps: float = 0.0) -> None:
            super().__init__()
            self.eps = nn.Parameter(torch.tensor(eps))
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
                nn.ReLU(),
            )
            self.edge_mlp: nn.Module | None = None
            if edge_dim > 0:
                self.edge_mlp = nn.Linear(edge_dim, in_dim)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """
            x: (N, in_dim) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) optional bond features
            """
            src, dst = edge_index[0], edge_index[1]
            # Aggregate: sum of neighbor features (optionally modulated by edge features)
            agg = torch.zeros_like(x)
            if src.numel() > 0:
                msg = x[src]
                if self.edge_mlp is not None and edge_attr is not None:
                    msg = msg + self.edge_mlp(edge_attr)
                agg.index_add_(0, dst, msg)
            # Update: (1 + eps) * x + aggregate
            out = (1 + self.eps) * x + agg
            return self.mlp(out)

    class GINEncoder(nn.Module):
        """Graph Isomorphism Network for molecular encoding.

        Produces a fixed-size embedding from a variable-size molecular graph.

        Example
        -------
        >>> enc = GINEncoder(embed_dim=128, n_layers=3)
        >>> g = smiles_to_graph("CCO")
        >>> emb = enc.encode_smiles("CCO")  # (128,)
        """

        def __init__(
            self,
            embed_dim: int = 128,
            n_layers: int = 3,
            atom_dim: int = _ATOM_DIM,
            dropout: float = 0.1,
            edge_dim: int = _BOND_DIM,
        ) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.edge_dim = edge_dim
            self.input_proj = nn.Linear(atom_dim, embed_dim)
            self.layers = nn.ModuleList([
                GINLayer(embed_dim, embed_dim, edge_dim=edge_dim) for _ in range(n_layers)
            ])
            self.dropout = nn.Dropout(dropout)
            self.output_proj = nn.Linear(embed_dim, embed_dim)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            batch: torch.Tensor,
            edge_attr: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Forward pass: graph -> embedding.

            Parameters
            ----------
            x: (N_total, atom_dim) all node features (batched)
            edge_index: (2, E_total) all edges (batched with offsets)
            batch: (N_total,) graph membership for each node
            edge_attr: (E_total, bond_dim) optional bond features

            Returns
            -------
            (B, embed_dim) one embedding per graph
            """
            h = self.input_proj(x)
            for layer in self.layers:
                h = layer(h, edge_index, edge_attr=edge_attr)
                h = self.dropout(h)

            # Global mean pooling per graph
            n_graphs = int(batch.max().item()) + 1
            out = torch.zeros(n_graphs, self.embed_dim, device=x.device)
            counts = torch.zeros(n_graphs, 1, device=x.device)
            out.index_add_(0, batch, h)
            counts.index_add_(0, batch, torch.ones(len(batch), 1, device=x.device))
            out = out / counts.clamp(min=1)

            return self.output_proj(out)

        @torch.no_grad()
        def encode_smiles(self, smiles: str) -> np.ndarray:
            """Encode a single SMILES to an embedding vector."""
            self.eval()
            g = smiles_to_graph(smiles)
            if g is None:
                return np.zeros(self.embed_dim, dtype=np.float32)
            bg = batch_graphs([g])
            x = torch.tensor(bg["x"])
            ei = torch.tensor(bg["edge_index"])
            b = torch.tensor(bg["batch"])
            ea = torch.tensor(bg["edge_attr"]) if bg["edge_attr"].size > 0 else None
            emb = self(x, ei, b, edge_attr=ea)
            return emb[0].numpy()

        @torch.no_grad()
        def encode_smiles_batch(self, smiles_list: list[str]) -> np.ndarray:
            """Encode a batch of SMILES to embeddings."""
            self.eval()
            graphs = [smiles_to_graph(s) for s in smiles_list]
            valid = [(i, g) for i, g in enumerate(graphs) if g is not None]
            if not valid:
                return np.zeros((len(smiles_list), self.embed_dim), dtype=np.float32)

            bg = batch_graphs([g for _, g in valid])
            x = torch.tensor(bg["x"])
            ei = torch.tensor(bg["edge_index"])
            b = torch.tensor(bg["batch"])
            ea = torch.tensor(bg["edge_attr"]) if bg["edge_attr"].size > 0 else None
            embs = self(x, ei, b, edge_attr=ea).numpy()

            result = np.zeros((len(smiles_list), self.embed_dim), dtype=np.float32)
            for (orig_idx, _), emb in zip(valid, embs):
                result[orig_idx] = emb
            return result

    # ---- SchNet ----

    class GaussianRBF(nn.Module):
        """Gaussian radial basis function expansion for distances."""

        def __init__(self, n_rbf: int = 20, cutoff: float = 10.0) -> None:
            super().__init__()
            self.n_rbf = n_rbf
            offsets = torch.linspace(0.0, cutoff, n_rbf)
            self.register_buffer("offsets", offsets)
            self.width = (offsets[1] - offsets[0]).item() if n_rbf > 1 else 1.0

        def forward(self, dist: torch.Tensor) -> torch.Tensor:
            """dist: (E,) -> (E, n_rbf)"""
            return torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.offsets) / self.width) ** 2)

    class SchNetLayer(nn.Module):
        """SchNet interaction layer.

        Supports two modes:
        - **3D mode** (use_3d=True): continuous-filter convolution on real
          inter-atomic distances expanded via Gaussian RBF.
        - **2D mode** (use_3d=False, default): uses shortest-path (hop count)
          distances as a proxy, expanded the same way.
        """

        def __init__(self, hidden_dim: int, n_rbf: int = 20, use_3d: bool = False) -> None:
            super().__init__()
            self.use_3d = use_3d
            self.n_rbf = n_rbf
            self.rbf = GaussianRBF(n_rbf=n_rbf, cutoff=10.0 if use_3d else 20.0)
            self.filter_net = nn.Sequential(
                nn.Linear(n_rbf, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.atom_update = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_dist: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """
            x: (N, hidden_dim)
            edge_index: (2, E)
            edge_dist: (E,) pairwise distances (3D Euclidean or hop count)
            """
            src, dst = edge_index[0], edge_index[1]
            if src.numel() == 0:
                return x + self.atom_update(x)

            if edge_dist is None:
                # Fallback: all edges at distance 1 (immediate neighbors)
                edge_dist = torch.ones(src.shape[0], device=x.device)

            # Continuous-filter convolution
            rbf = self.rbf(edge_dist)  # (E, n_rbf)
            W = self.filter_net(rbf)   # (E, hidden_dim)
            msg = x[src] * W           # element-wise modulation
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, msg)
            return x + self.atom_update(agg)

    class SchNetEncoder(nn.Module):
        """SchNet-style encoder for molecular graphs.

        Parameters
        ----------
        use_3d : bool
            If True, expects 3D coordinates and computes real Euclidean distances.
            If False (default), uses shortest-path hop distances as a 2D proxy.
        """

        def __init__(
            self,
            embed_dim: int = 128,
            n_layers: int = 3,
            atom_dim: int = _ATOM_DIM,
            use_3d: bool = False,
            n_rbf: int = 20,
        ) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.use_3d = use_3d
            self.input_proj = nn.Linear(atom_dim, embed_dim)
            self.layers = nn.ModuleList([
                SchNetLayer(embed_dim, n_rbf=n_rbf, use_3d=use_3d)
                for _ in range(n_layers)
            ])
            self.output_proj = nn.Linear(embed_dim, embed_dim)

        def _compute_edge_distances(
            self,
            edge_index: torch.Tensor,
            n_atoms: int,
            pos: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Compute per-edge distances.

            In 3D mode, uses Euclidean distances from *pos*.
            In 2D mode, uses shortest-path hop counts.
            """
            src, dst = edge_index[0], edge_index[1]
            if self.use_3d and pos is not None:
                # Real 3D Euclidean distances
                return (pos[src] - pos[dst]).norm(dim=-1)

            # 2D fallback: shortest-path distances via BFS
            ei_np = edge_index.detach().cpu().numpy()
            sp = _shortest_path_distances(ei_np, n_atoms)
            sp_tensor = torch.tensor(sp, device=edge_index.device, dtype=torch.float32)
            return sp_tensor[src, dst]

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            batch: torch.Tensor,
            pos: torch.Tensor | None = None,
            edge_attr: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Forward pass.

            Parameters
            ----------
            x : (N_total, atom_dim)
            edge_index : (2, E_total)
            batch : (N_total,)
            pos : (N_total, 3) optional 3D coordinates (required if use_3d=True)
            edge_attr : ignored (accepted for interface compatibility)
            """
            h = self.input_proj(x)

            n_atoms = int(x.shape[0])
            if edge_index.shape[1] > 0:
                edge_dist = self._compute_edge_distances(edge_index, n_atoms, pos=pos)
            else:
                edge_dist = None

            for layer in self.layers:
                h = layer(h, edge_index, edge_dist=edge_dist)

            # Global mean pool
            n_graphs = int(batch.max().item()) + 1
            out = torch.zeros(n_graphs, self.embed_dim, device=x.device)
            counts = torch.zeros(n_graphs, 1, device=x.device)
            out.index_add_(0, batch, h)
            counts.index_add_(0, batch, torch.ones(len(batch), 1, device=x.device))
            return self.output_proj(out / counts.clamp(min=1))

        @torch.no_grad()
        def encode_smiles(self, smiles: str) -> np.ndarray:
            self.eval()
            g = smiles_to_graph(smiles)
            if g is None:
                return np.zeros(self.embed_dim, dtype=np.float32)
            bg = batch_graphs([g])
            return self(torch.tensor(bg["x"]), torch.tensor(bg["edge_index"]),
                       torch.tensor(bg["batch"]))[0].numpy()

        @torch.no_grad()
        def encode_smiles_batch(self, smiles_list: list[str]) -> np.ndarray:
            self.eval()
            graphs = [smiles_to_graph(s) for s in smiles_list]
            valid = [(i, g) for i, g in enumerate(graphs) if g is not None]
            if not valid:
                return np.zeros((len(smiles_list), self.embed_dim), dtype=np.float32)
            bg = batch_graphs([g for _, g in valid])
            embs = self(torch.tensor(bg["x"]), torch.tensor(bg["edge_index"]),
                       torch.tensor(bg["batch"])).numpy()
            result = np.zeros((len(smiles_list), self.embed_dim), dtype=np.float32)
            for (idx, _), emb in zip(valid, embs):
                result[idx] = emb
            return result

else:
    class GINEncoder:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            raise ImportError("GINEncoder requires PyTorch")

    class SchNetEncoder:  # type: ignore[no-redef]
        def __init__(self, **kw: Any) -> None:
            raise ImportError("SchNetEncoder requires PyTorch")
