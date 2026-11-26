"""GSA-V1 core for NK-LSTM-PATH.

Geometric self-attention with a learned positive edge action on the N-dimensional
binary hypercube.  The model consumes forced paths, not Ogata trajectories.

Project integration (from repository root):
    from gsa_v1_core import build_forced_pair_dataset, GSAPathModel

Tensor conventions:
    x_seq   : (B, L, N), values in {-1, +1}; L = T + 1 states
    act_idx : (B, T), integer flipped-site indices; action t maps x[t] -> x[t+1]
    j0      : (B, N*(N-1)//2), upper-triangle packing of the initial symmetric J
    pad_mask: (B, L), bool, True means padding (right-padding only)
    pos     : (B, L), integer position indices

For a query at position i and key at position j <= i, the geometric bias is
    - beta_action * sum_{t=j}^{i-1} c_theta(x[t], act[t]).
It therefore depends on the observed path segment.  The all-pairs shortest-path
matrix S* is provided as a diagnostic/reference, not substituted for the
observed path action in the attention bias.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TransportMode = Literal["integrable", "symmetric"]
GeometryMode = Literal["none", "fixed", "learned"]


# ---------------------------------------------------------------------------
# J <-> vector convention
# ---------------------------------------------------------------------------

def _upper_indices(n_sites: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n_sites, k=1)


def pack_J(J: np.ndarray) -> np.ndarray:
    """Pack a symmetric, zero-diagonal J into lexicographic upper-triangle order."""
    arr = np.asarray(J, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("J doit être une matrice carrée.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("J contient des valeurs non finies.")
    if not np.allclose(arr, arr.T, rtol=1e-10, atol=1e-12):
        raise ValueError("J doit être symétrique.")
    if not np.allclose(np.diag(arr), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("La diagonale de J doit être nulle.")
    rows, cols = _upper_indices(arr.shape[0])
    return arr[rows, cols].copy()


def unpack_J(vector: np.ndarray | torch.Tensor, n_sites: int):
    """Unpack upper-triangle vector to a symmetric zero-diagonal matrix."""
    expected = n_sites * (n_sites - 1) // 2
    if isinstance(vector, torch.Tensor):
        if vector.shape[-1] != expected:
            raise ValueError(f"Dernière dimension attendue: {expected}, reçue: {vector.shape[-1]}.")
        out = vector.new_zeros((*vector.shape[:-1], n_sites, n_sites))
        rows, cols = _upper_indices(n_sites)
        r = torch.as_tensor(rows, device=vector.device)
        c = torch.as_tensor(cols, device=vector.device)
        out[..., r, c] = vector
        out[..., c, r] = vector
        return out
    arr = np.asarray(vector)
    if arr.shape[-1] != expected:
        raise ValueError(f"Dernière dimension attendue: {expected}, reçue: {arr.shape[-1]}.")
    out = np.zeros((*arr.shape[:-1], n_sites, n_sites), dtype=arr.dtype)
    rows, cols = _upper_indices(n_sites)
    out[..., rows, cols] = arr
    out[..., cols, rows] = arr
    return out


# ---------------------------------------------------------------------------
# Binary hypercube indexing and edge geometry
# ---------------------------------------------------------------------------

def states_to_indices(x_seq: torch.Tensor) -> torch.Tensor:
    """Map {-1,+1}^N states to integer vertices in [0, 2**N)."""
    if x_seq.ndim != 3:
        raise ValueError("x_seq doit avoir la forme (B,L,N).")
    bits = (x_seq > 0).to(torch.long)
    n_sites = x_seq.shape[-1]
    weights = (1 << torch.arange(n_sites, device=x_seq.device, dtype=torch.long))
    return torch.sum(bits * weights, dim=-1)


def _undirected_edge_index(n_sites: int) -> tuple[torch.Tensor, int]:
    """Return an (2**N,N) lookup; reverse traversals share one edge parameter."""
    n_states = 1 << n_sites
    edge_to_id: dict[tuple[int, int], int] = {}
    lookup = np.empty((n_states, n_sites), dtype=np.int64)
    for state in range(n_states):
        for site in range(n_sites):
            neighbour = state ^ (1 << site)
            key = (min(state, neighbour), max(state, neighbour))
            if key not in edge_to_id:
                edge_to_id[key] = len(edge_to_id)
            lookup[state, site] = edge_to_id[key]
    return torch.as_tensor(lookup, dtype=torch.long), len(edge_to_id)


class UndirectedPositiveEdgeCosts(nn.Module):
    """Learn one positive cost per undirected edge of the binary hypercube.

    Softplus followed by a bounded transform guarantees strictly positive,
    bounded costs.  Mean normalization fixes their global scale so beta_action
    remains interpretable.  For N=6 this gives 192 undirected edge parameters.
    """

    def __init__(self, n_sites: int, eps: float = 1e-4) -> None:
        super().__init__()
        if n_sites < 1:
            raise ValueError("n_sites doit être >= 1.")
        if eps <= 0:
            raise ValueError("eps doit être strictement positif.")
        self.n_sites = int(n_sites)
        self.n_states = 1 << self.n_sites
        self.eps = float(eps)
        edge_lookup, n_edges = _undirected_edge_index(self.n_sites)
        self.register_buffer("edge_lookup", edge_lookup)
        self.edge_logits = nn.Parameter(torch.zeros(n_edges))

    def unique_costs(self) -> torch.Tensor:
        positive = F.softplus(self.edge_logits)
        bounded_raw = 0.5 + 1.5 * (positive / (1.0 + positive))
        # bounded_raw is in (0.5, 2); after mean normalization costs lie in
        # (0.25, 4), with global mean exactly one (up to floating-point error).
        return bounded_raw / bounded_raw.mean().clamp_min(self.eps)

    def table(self) -> torch.Tensor:
        """Return c_theta(state, site), shape (2**N,N), with reverse symmetry."""
        return self.unique_costs()[self.edge_lookup]

    def regularizer(self) -> torch.Tensor:
        """Penalize excessive edge-cost heterogeneity, not the fixed mean scale."""
        log_cost = torch.log(self.unique_costs().clamp_min(self.eps))
        return torch.mean((log_cost - log_cost.mean()) ** 2)

    def all_pairs_shortest_paths(self) -> torch.Tensor:
        """Differentiable min-plus Floyd-Warshall; returns S* of shape (2**N,2**N)."""
        costs = self.table()
        n_states = self.n_states
        device, dtype = costs.device, costs.dtype
        inf = torch.tensor(float("inf"), device=device, dtype=dtype)
        dist = torch.full((n_states, n_states), inf, device=device, dtype=dtype)
        idx = torch.arange(n_states, device=device)
        dist[idx, idx] = 0.0
        for site in range(self.n_sites):
            neighbours = idx ^ (1 << site)
            dist[idx, neighbours] = costs[:, site]
        for pivot in range(n_states):
            through = dist[:, pivot:pivot + 1] + dist[pivot:pivot + 1, :]
            dist = torch.minimum(dist, through)
        return dist


def fixed_unit_edge_table(n_sites: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Unit edge costs: a fixed hypercube metric (shortest distance = Hamming)."""
    return torch.ones((1 << n_sites, n_sites), device=device, dtype=dtype)


def cumulative_observed_action(
    x_seq: torch.Tensor,
    act_idx: torch.Tensor,
    edge_cost_table: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build S_obs[b,i,j] = action cost from token j to token i for j <= i.

    Returns (B,L,L). Entries above the causal diagonal are set to zero because
    those locations are masked in attention.
    """
    batch, length, n_sites = x_seq.shape
    if act_idx.shape != (batch, length - 1):
        raise ValueError("act_idx doit avoir la forme (B,L-1).")
    if edge_cost_table.shape != (1 << n_sites, n_sites):
        raise ValueError("edge_cost_table doit avoir la forme (2**N,N).")
    state_idx = states_to_indices(x_seq)
    if pad_mask is None:
        valid_transition = torch.ones((batch, length - 1), device=x_seq.device, dtype=torch.bool)
    else:
        if pad_mask.shape != (batch, length) or pad_mask.dtype != torch.bool:
            raise ValueError("pad_mask doit être booléen et de forme (B,L).")
        valid_transition = (~pad_mask[:, :-1]) & (~pad_mask[:, 1:])
    if act_idx.numel() and torch.any(valid_transition & ((act_idx < 0) | (act_idx >= n_sites))):
        raise ValueError("Chaque action valide doit être un indice entre 0 et N-1.")
    safe_actions = act_idx.clamp(min=0, max=n_sites - 1)
    edge_costs = edge_cost_table[state_idx[:, :-1], safe_actions]
    edge_costs = edge_costs * valid_transition.to(edge_costs.dtype)
    prefix = torch.cat(
        [edge_costs.new_zeros((batch, 1)), torch.cumsum(edge_costs, dim=1)],
        dim=1,
    )
    # Query i is row i; key j is column j.
    action_matrix = prefix[:, :, None] - prefix[:, None, :]
    causal = torch.tril(torch.ones((length, length), device=x_seq.device, dtype=torch.bool))
    return action_matrix.masked_fill(~causal[None, :, :], 0.0)


def observed_action_excess(
    x_seq: torch.Tensor,
    observed_action: torch.Tensor,
    shortest_paths: torch.Tensor,
) -> torch.Tensor:
    """Return max(0, S_obs - S*(endpoints)); diagnostic path excess (B,L,L)."""
    state_idx = states_to_indices(x_seq)
    # For query i, key j: compare the observed j->i segment with S*(x_j,x_i).
    start = state_idx[:, None, :]
    end = state_idx[:, :, None]
    shortest = shortest_paths[start, end]
    return torch.clamp(observed_action - shortest, min=0.0)


# ---------------------------------------------------------------------------
# Attention and path model
# ---------------------------------------------------------------------------

class GeometricSelfAttention(nn.Module):
    """Causal multi-head self-attention with an optional observed-action bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model doit être divisible par n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        action_bias: torch.Tensor | None,
        pad_mask: torch.Tensor | None,
        beta_action: float,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if action_bias is not None:
            if action_bias.shape != (batch, length, length):
                raise ValueError("action_bias doit avoir la forme (B,L,L).")
            scores = scores - float(beta_action) * action_bias[:, None, :, :]

        causal_mask = torch.triu(
            torch.ones((length, length), device=hidden.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))
        if pad_mask is not None:
            # True in pad_mask means the key is padding and cannot be attended to.
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous()
        attended = attended.view(batch, length, self.d_model)
        out = self.out_dropout(self.out_proj(attended))
        if pad_mask is not None:
            out = out.masked_fill(pad_mask[:, :, None], 0.0)
        return out


class GSAEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, ff_mult: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = GeometricSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, hidden, action_bias, pad_mask, beta_action):
        hidden = hidden + self.attention(self.norm1(hidden), action_bias, pad_mask, beta_action)
        hidden = hidden + self.ff(self.norm2(hidden))
        if pad_mask is not None:
            hidden = hidden.masked_fill(pad_mask[:, :, None], 0.0)
        return hidden


@dataclass(frozen=True)
class GSAConfig:
    n_sites: int = 6
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.0
    beta_action: float = 1.0
    max_len: int = 32
    edge_eps: float = 1e-4
    geometry_mode: GeometryMode = "learned"

    def __post_init__(self) -> None:
        if self.n_sites < 2:
            raise ValueError("n_sites doit être >= 2.")
        if self.d_model < 1 or self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model doit être positif et divisible par n_heads.")
        if self.n_layers < 1 or self.ff_mult < 1 or self.max_len < 2:
            raise ValueError("n_layers, ff_mult et max_len doivent être positifs.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout doit appartenir à [0,1).")
        if self.beta_action < 0.0 or self.edge_eps <= 0.0:
            raise ValueError("beta_action doit être >= 0 et edge_eps > 0.")
        if self.geometry_mode not in ("none", "fixed", "learned"):
            raise ValueError("geometry_mode doit être none, fixed ou learned.")


class GSAPathModel(nn.Module):
    """Predict J_final from a forced path using shared path encoders.

    A path is encoded once; for a pair, call the same model on both orders and
    subtract predictions.  This makes Omega_hat(a,b) = -Omega_hat(b,a) by
    construction.
    """

    def __init__(self, config: GSAConfig = GSAConfig()) -> None:
        super().__init__()
        self.config = config
        n = config.n_sites
        d_j = n * (n - 1) // 2
        self.x_proj = nn.Linear(n, config.d_model)
        self.j0_proj = nn.Linear(d_j, config.d_model)
        self.action_embedding = nn.Embedding(n + 1, config.d_model)  # n = terminal/no-action token
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.blocks = nn.ModuleList([
            GSAEncoderBlock(config.d_model, config.n_heads, config.dropout, config.ff_mult)
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.delta_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, d_j),
        )
        self.edge_costs = (
            UndirectedPositiveEdgeCosts(n, config.edge_eps)
            if config.geometry_mode == "learned" else None
        )

    @property
    def d_J(self) -> int:
        n = self.config.n_sites
        return n * (n - 1) // 2

    def edge_cost_table(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.config.geometry_mode == "none":
            return None
        if self.config.geometry_mode == "fixed":
            return fixed_unit_edge_table(self.config.n_sites, device=device, dtype=dtype)
        assert self.edge_costs is not None
        return self.edge_costs.table().to(device=device, dtype=dtype)

    def path_action_bias(
        self,
        x_seq: torch.Tensor,
        act_idx: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        table = self.edge_cost_table(device=x_seq.device, dtype=x_seq.dtype)
        if table is None:
            return None
        return cumulative_observed_action(x_seq, act_idx, table, pad_mask)

    def forward(
        self,
        x_seq: torch.Tensor,
        act_idx: torch.Tensor,
        j0: torch.Tensor,
        pos: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_seq.ndim != 3:
            raise ValueError("x_seq doit avoir la forme (B,L,N).")
        batch, length, n_sites = x_seq.shape
        if n_sites != self.config.n_sites:
            raise ValueError(f"N attendu={self.config.n_sites}, reçu={n_sites}.")
        if length < 2 or length > self.config.max_len:
            raise ValueError("La longueur L doit appartenir à [2,max_len].")
        if act_idx.shape != (batch, length - 1):
            raise ValueError("act_idx doit avoir la forme (B,L-1).")
        if j0.shape != (batch, self.d_J):
            raise ValueError(f"j0 doit avoir la forme (B,{self.d_J}).")
        if not torch.is_floating_point(x_seq) or not torch.is_floating_point(j0):
            raise TypeError("x_seq et j0 doivent être des tenseurs flottants.")
        if pos is None:
            pos = torch.arange(length, device=x_seq.device).expand(batch, length)
        if pos.shape != (batch, length) or pos.dtype != torch.long:
            raise ValueError("pos doit être long et de forme (B,L).")
        if torch.any(pos < 0) or torch.any(pos >= self.config.max_len):
            raise ValueError("pos contient un indice hors de [0,max_len).")
        if pad_mask is None:
            pad_mask = torch.zeros((batch, length), device=x_seq.device, dtype=torch.bool)
        elif pad_mask.shape != (batch, length) or pad_mask.dtype != torch.bool:
            raise ValueError("pad_mask doit être booléen et de forme (B,L).")
        if torch.any(pad_mask.all(dim=1)):
            raise ValueError("Chaque séquence doit contenir au moins un état non rembourré.")
        # Padding must be on the right: once padded, all later positions are padded.
        if length > 1 and torch.any(pad_mask[:, :-1] & ~pad_mask[:, 1:]):
            raise ValueError("pad_mask doit utiliser un remplissage à droite.")
        if torch.any((x_seq != 1.0) & (x_seq != -1.0)):
            raise ValueError("x_seq doit contenir uniquement -1 et +1.")

        action_tokens = torch.full((batch, length), self.config.n_sites,
                                   device=x_seq.device, dtype=torch.long)
        if length > 1:
            valid_actions = (~pad_mask[:, :-1]) & (~pad_mask[:, 1:])
            safe_actions = act_idx.clamp(min=0, max=self.config.n_sites - 1)
            action_tokens[:, :-1] = torch.where(
                valid_actions, safe_actions, torch.full_like(safe_actions, self.config.n_sites)
            )
        action_tokens = action_tokens.masked_fill(pad_mask, self.config.n_sites)
        hidden = (
            self.x_proj(x_seq)
            + self.action_embedding(action_tokens)
            + self.position_embedding(pos)
            + self.j0_proj(j0).unsqueeze(1)
        )
        hidden = self.input_norm(hidden)
        if pad_mask is not None:
            hidden = hidden.masked_fill(pad_mask[:, :, None], 0.0)
        action_bias = self.path_action_bias(x_seq, act_idx, pad_mask)
        for block in self.blocks:
            hidden = block(hidden, action_bias, pad_mask, self.config.beta_action)
        hidden = self.final_norm(hidden)
        lengths = (~pad_mask).sum(dim=1)
        last_valid = lengths - 1
        pooled = hidden[torch.arange(batch, device=x_seq.device), last_valid]
        delta_j = self.delta_head(pooled)
        return j0 + delta_j

    def geometry_regularizer(self) -> torch.Tensor:
        if self.edge_costs is None:
            return next(self.parameters()).new_zeros(())
        return self.edge_costs.regularizer()

    def shortest_path_matrix(self) -> torch.Tensor | None:
        """Return learned S*; fixed mode returns the Hamming-distance matrix."""
        if self.config.geometry_mode == "none":
            return None
        if self.config.geometry_mode == "learned":
            assert self.edge_costs is not None
            return self.edge_costs.all_pairs_shortest_paths()
        table = fixed_unit_edge_table(
            self.config.n_sites,
            device=self.position_embedding.weight.device,
            dtype=self.position_embedding.weight.dtype,
        )
        # A temporary fixed-edge instance is unnecessary: direct min-plus closure.
        n_states = 1 << self.config.n_sites
        inf = torch.tensor(float("inf"), device=table.device, dtype=table.dtype)
        dist = torch.full((n_states, n_states), inf, device=table.device, dtype=table.dtype)
        idx = torch.arange(n_states, device=table.device)
        dist[idx, idx] = 0.0
        for site in range(self.config.n_sites):
            dist[idx, idx ^ (1 << site)] = table[:, site]
        for pivot in range(n_states):
            dist = torch.minimum(dist, dist[:, pivot:pivot + 1] + dist[pivot:pivot + 1, :])
        return dist


# ---------------------------------------------------------------------------
# Paired forced-path dataset using the project's real transition function
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForcedPairDataset:
    x_a: np.ndarray          # (P,T+1,N)
    x_b: np.ndarray          # (P,T+1,N)
    act_a: np.ndarray        # (P,T)
    act_b: np.ndarray        # (P,T)
    j0: np.ndarray           # (P,d_J)
    j_seq_a: np.ndarray      # (P,T+1,d_J), retained for a coupled-state extension
    j_seq_b: np.ndarray      # (P,T+1,d_J), retained for a coupled-state extension
    j_final_a: np.ndarray    # (P,d_J)
    j_final_b: np.ndarray    # (P,d_J)
    omega: np.ndarray        # (P,d_J) = j_final_a - j_final_b
    initial_state_id: np.ndarray  # (P,)
    site_pair: np.ndarray        # (P,2)
    mode: str

    @property
    def n_pairs(self) -> int:
        return int(self.x_a.shape[0])


def _rollout_forced_path(X0, J0, actions: Sequence[int], eta: float, mode: TransportMode):
    # Delayed import lets the model and its unit tests run outside the repository.
    from balayage_commutateur_nk_v2 import step_flip_node

    x = np.asarray(X0, dtype=np.float64).copy()
    j = np.asarray(J0, dtype=np.float64).copy()
    xs = [x.copy()]
    js = [j.copy()]
    for site in actions:
        x, j = step_flip_node(x, j, int(site), float(eta), mode)
        x = np.asarray(x, dtype=np.float64).copy()
        j = np.asarray(j, dtype=np.float64).copy()
        xs.append(x.copy())
        js.append(j.copy())
    return np.stack(xs), np.stack(js)


def build_forced_pair_dataset(
    initial_states: Sequence[tuple[np.ndarray, np.ndarray]],
    eta: float = 0.25,
    mode: TransportMode = "symmetric",
) -> ForcedPairDataset:
    """Build matched (k->m, m->k) pairs using step_flip_node from the project.

    No NKLandscape or Ogata thinning is called: the action order is imposed.
    The landscape affects stochastic action selection, not the forced mutation
    operator used to define these paired commutator targets.
    """
    if mode not in ("integrable", "symmetric"):
        raise ValueError("mode doit être integrable ou symmetric.")
    if not initial_states:
        raise ValueError("initial_states ne peut pas être vide.")
    n_sites = int(np.asarray(initial_states[0][0]).size)
    if n_sites < 2:
        raise ValueError("Il faut au moins deux sites.")
    rows: dict[str, list] = {
        "x_a": [], "x_b": [], "act_a": [], "act_b": [], "j0": [],
        "jseq_a": [], "jseq_b": [], "ja": [], "jb": [], "omega": [],
        "state_id": [], "pair": [],
    }
    for state_id, (X0, J0) in enumerate(initial_states):
        X0 = np.asarray(X0, dtype=np.float64)
        J0 = np.asarray(J0, dtype=np.float64)
        if X0.shape != (n_sites,) or not np.all(np.isin(X0, (-1.0, 1.0))):
            raise ValueError(f"X0 invalide à l'indice {state_id}.")
        if J0.shape != (n_sites, n_sites):
            raise ValueError(f"J0 invalide à l'indice {state_id}.")
        j0_vec = pack_J(J0)
        for k in range(n_sites):
            for m in range(k + 1, n_sites):
                xa, ja_seq = _rollout_forced_path(X0, J0, (k, m), eta, mode)
                xb, jb_seq = _rollout_forced_path(X0, J0, (m, k), eta, mode)
                if not np.array_equal(xa[-1], xb[-1]):
                    raise RuntimeError("Les deux ordres n'aboutissent pas au même X final.")
                ja_vec, jb_vec = pack_J(ja_seq[-1]), pack_J(jb_seq[-1])
                rows["x_a"].append(xa)
                rows["x_b"].append(xb)
                rows["jseq_a"].append(np.stack([pack_J(j) for j in ja_seq]))
                rows["jseq_b"].append(np.stack([pack_J(j) for j in jb_seq]))
                rows["act_a"].append((k, m))
                rows["act_b"].append((m, k))
                rows["j0"].append(j0_vec)
                rows["ja"].append(ja_vec)
                rows["jb"].append(jb_vec)
                rows["omega"].append(ja_vec - jb_vec)
                rows["state_id"].append(state_id)
                rows["pair"].append((k, m))
    return ForcedPairDataset(
        x_a=np.asarray(rows["x_a"], dtype=np.float32),
        x_b=np.asarray(rows["x_b"], dtype=np.float32),
        act_a=np.asarray(rows["act_a"], dtype=np.int64),
        act_b=np.asarray(rows["act_b"], dtype=np.int64),
        j0=np.asarray(rows["j0"], dtype=np.float32),
        j_seq_a=np.asarray(rows["jseq_a"], dtype=np.float32),
        j_seq_b=np.asarray(rows["jseq_b"], dtype=np.float32),
        j_final_a=np.asarray(rows["ja"], dtype=np.float32),
        j_final_b=np.asarray(rows["jb"], dtype=np.float32),
        omega=np.asarray(rows["omega"], dtype=np.float32),
        initial_state_id=np.asarray(rows["state_id"], dtype=np.int64),
        site_pair=np.asarray(rows["pair"], dtype=np.int64),
        mode=mode,
    )


def split_indices_by_initial_state(
    dataset: ForcedPairDataset,
    seed: int = 7,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> dict[str, np.ndarray]:
    """Split by initial-state group to prevent sibling path-pair leakage."""
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction doit être dans ]0,1[.")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction doit être dans ]0,1[.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Les fractions train + validation doivent être < 1.")
    groups = np.unique(dataset.initial_state_id)
    if groups.size < 3:
        raise ValueError("Il faut au moins trois états initiaux pour train/validation/test.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(groups)
    n_train = max(1, int(np.floor(groups.size * train_fraction)))
    n_val = max(1, int(np.floor(groups.size * validation_fraction)))
    if n_train + n_val >= groups.size:
        n_train = groups.size - 2
        n_val = 1
    train_groups = shuffled[:n_train]
    val_groups = shuffled[n_train:n_train + n_val]
    test_groups = shuffled[n_train + n_val:]
    ids = dataset.initial_state_id
    return {
        "train": np.flatnonzero(np.isin(ids, train_groups)),
        "validation": np.flatnonzero(np.isin(ids, val_groups)),
        "test": np.flatnonzero(np.isin(ids, test_groups)),
    }


# ---------------------------------------------------------------------------
# Shared-head paired prediction, objective and evaluation metrics
# ---------------------------------------------------------------------------

def predict_path_pair(
    model: GSAPathModel,
    x_a: torch.Tensor,
    act_a: torch.Tensor,
    x_b: torch.Tensor,
    act_b: torch.Tensor,
    j0: torch.Tensor,
    pos_a: torch.Tensor | None = None,
    pos_b: torch.Tensor | None = None,
    pad_mask_a: torch.Tensor | None = None,
    pad_mask_b: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict both orders with identical weights and return Omega_hat = Jhat_a-Jhat_b."""
    pred_a = model(x_a, act_a, j0, pos=pos_a, pad_mask=pad_mask_a)
    pred_b = model(x_b, act_b, j0, pos=pos_b, pad_mask=pad_mask_b)
    return pred_a, pred_b, pred_a - pred_b


def paired_gsa_loss(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    lambda_omega: float = 1.0,
    lambda_geometry: float = 1e-3,
    model: GSAPathModel | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Terminal loss + pair-closure loss + optional edge-cost regularization."""
    if pred_a.shape != pred_b.shape or pred_a.shape != target_a.shape or pred_b.shape != target_b.shape:
        raise ValueError("Les quatre tenseurs de J terminal doivent avoir la même forme.")
    omega_hat = pred_a - pred_b  # shared head => exact antisymmetry by construction
    omega_true = target_a - target_b
    loss_j = 0.5 * (F.mse_loss(pred_a, target_a) + F.mse_loss(pred_b, target_b))
    loss_omega = F.mse_loss(omega_hat, omega_true)
    if model is None:
        loss_geometry = loss_j.new_zeros(())
    else:
        loss_geometry = model.geometry_regularizer()
    loss = loss_j + float(lambda_omega) * loss_omega + float(lambda_geometry) * loss_geometry
    info = {
        "loss_total": float(loss.detach().cpu()),
        "loss_terminal": float(loss_j.detach().cpu()),
        "loss_closure": float(loss_omega.detach().cpu()),
        "loss_geometry_regularizer": float(loss_geometry.detach().cpu()),
    }
    return loss, info


def closure_metrics(omega_hat: np.ndarray, omega_true: np.ndarray, eps: float = 1e-12) -> dict[str, float | int]:
    """Report MSE, norm-relative error, cosine and count of nonzero targets."""
    pred = np.asarray(omega_hat, dtype=np.float64)
    true = np.asarray(omega_true, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 2:
        raise ValueError("omega_hat et omega_true doivent être de même forme (P,d_J).")
    errors = pred - true
    norms = np.linalg.norm(true, axis=1)
    pred_norms = np.linalg.norm(pred, axis=1)
    valid = norms > eps
    relative = np.linalg.norm(errors, axis=1)[valid] / norms[valid]
    cosine = np.sum(pred[valid] * true[valid], axis=1) / np.maximum(pred_norms[valid] * norms[valid], eps)
    return {
        "mse": float(np.mean(errors ** 2)),
        "mean_relative_norm_error_nonzero": float(np.mean(relative)) if np.any(valid) else float("nan"),
        "mean_cosine_nonzero": float(np.mean(cosine)) if np.any(valid) else float("nan"),
        "n_pairs": int(pred.shape[0]),
        "n_nonzero_targets": int(np.count_nonzero(valid)),
    }


# ---------------------------------------------------------------------------
# Numerical property test: S* is endpoint-only; observed action is path-aware
# ---------------------------------------------------------------------------

def test_path_vs_shortest_action_property() -> dict[str, float]:
    """Tiny exact example on a 6-cube proving property (1) numerically."""
    n_sites = 6
    n_states = 1 << n_sites
    costs = np.full((n_states, n_sites), 10.0, dtype=np.float64)

    def set_undirected(state: int, site: int, value: float) -> None:
        neighbour = state ^ (1 << site)
        costs[state, site] = value
        costs[neighbour, site] = value

    # Start vertex 0; common endpoint 0 xor 1 xor 2 = 3.
    # Path A: 0 --site0(cost 1)--> 1 --site1(cost 4)--> 3; total 5.
    # Path B: 0 --site1(cost 2)--> 2 --site0(cost 1.5)--> 3; total 3.5.
    set_undirected(0, 0, 1.0)
    set_undirected(1, 1, 4.0)
    set_undirected(0, 1, 2.0)
    set_undirected(2, 0, 1.5)

    dist = np.full((n_states, n_states), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for state in range(n_states):
        for site in range(n_sites):
            neighbour = state ^ (1 << site)
            dist[state, neighbour] = costs[state, site]
    for pivot in range(n_states):
        dist = np.minimum(dist, dist[:, pivot:pivot + 1] + dist[pivot:pivot + 1, :])

    action_a = costs[0, 0] + costs[1, 1]
    action_b = costs[0, 1] + costs[2, 0]
    shortest_ab = dist[0, 3]
    assert np.isclose(shortest_ab, min(action_a, action_b))
    assert np.isclose(shortest_ab, dist[0, 3])  # same two endpoints => same S*
    assert not np.isclose(action_a, action_b)    # observed action distinguishes order
    return {
        "observed_action_k_then_m": float(action_a),
        "observed_action_m_then_k": float(action_b),
        "shortest_action_between_common_endpoints": float(shortest_ab),
    }


def self_test() -> None:
    property_result = test_path_vs_shortest_action_property()
    config = GSAConfig(n_sites=6, d_model=32, n_heads=4, n_layers=2, max_len=8, geometry_mode="learned")
    model = GSAPathModel(config)
    batch, length, n_sites = 5, 3, 6
    x_seq = torch.where(torch.rand(batch, length, n_sites) > 0.5, 1.0, -1.0)
    act_idx = torch.randint(0, n_sites, (batch, length - 1))
    j0 = torch.randn(batch, config.n_sites * (config.n_sites - 1) // 2)
    pred = model(x_seq, act_idx, j0)
    assert pred.shape == j0.shape
    assert torch.isfinite(pred).all()
    table = model.edge_costs.table()
    assert table.shape == (1 << n_sites, n_sites)
    assert torch.all(table > 0)
    # Reverse traversal of every undirected edge has exactly the same cost.
    for state in range(1 << n_sites):
        for site in range(n_sites):
            neighbour = state ^ (1 << site)
            assert torch.allclose(table[state, site], table[neighbour, site])
    shortest = model.shortest_path_matrix()
    assert shortest is not None and shortest.shape == (1 << n_sites, 1 << n_sites)
    assert torch.isfinite(shortest).all()
    assert torch.allclose(shortest, shortest.T, atol=1e-6, rtol=1e-6)
    loss = pred.square().mean() + model.geometry_regularizer()
    loss.backward()
    assert model.edge_costs.edge_logits.grad is not None
    assert torch.isfinite(model.edge_costs.edge_logits.grad).all()
    print("GSA-V1 SELF-TEST: OK")
    print("Property S* / action observée:", property_result)
    print("Model output:", tuple(pred.shape))
    print("Undirected edges:", model.edge_costs.edge_logits.numel())
    print("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


def integration_smoke() -> None:
    """Exercise the real project step_flip_node API on a tiny paired dataset."""
    from balayage_commutateur_nk_v2 import (
        ExperimentConfig, sample_binary_symmetric_states
    )

    cfg = ExperimentConfig(n_initial_states=3)
    rng = np.random.default_rng(cfg.init_seed)
    initial_states = sample_binary_symmetric_states(
        n_sites=cfg.n_sites, n_states=cfg.n_initial_states, rng=rng, scale=cfg.scale_J
    )
    expected_pairs = cfg.n_initial_states * (cfg.n_sites * (cfg.n_sites - 1) // 2)
    for mode in ("integrable", "symmetric"):
        dataset = build_forced_pair_dataset(initial_states, eta=cfg.eta, mode=mode)
        assert dataset.n_pairs == expected_pairs
        assert dataset.x_a.shape == (expected_pairs, 3, cfg.n_sites)
        assert dataset.j_seq_a.shape == (expected_pairs, 3, cfg.n_sites * (cfg.n_sites - 1) // 2)
        assert dataset.omega.shape == (expected_pairs, cfg.n_sites * (cfg.n_sites - 1) // 2)
        if mode == "integrable":
            max_abs = float(np.max(np.abs(dataset.omega)))
            if max_abs > 1e-12:
                raise AssertionError(f"Contrôle intégrable non nul: max |Omega|={max_abs:.3e}")
        print(
            f"{mode}: pairs={dataset.n_pairs}, x={dataset.x_a.shape}, "
            f"j_seq={dataset.j_seq_a.shape}, omega={dataset.omega.shape}, "
            f"max|Omega|={np.max(np.abs(dataset.omega)):.3e}"
        )
    print("GSA-V1 PROJECT INTEGRATION SMOKE: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tests du noyau GSA-V1.")
    parser.add_argument("--self-test", action="store_true", help="Exécute les tests numériques locaux.")
    parser.add_argument("--integration-smoke", action="store_true", help="Teste step_flip_node dans le dépôt NK-LSTM-PATH.")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.integration_smoke:
        integration_smoke()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
