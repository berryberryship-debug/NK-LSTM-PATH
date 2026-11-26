#!/usr/bin/env python3
"""
gsa_v1_core.py

Noyau géométrique et causal pour l'encodage de trajectoires sur l'hypercube
{-1, +1}^N et la prédiction antisymétrique du commutateur d'ordre Omega.

Invariants garantis par le noyau :
1. Source unique d'embedding causal (`GSAPathModel.embed_inputs`) :
   - Token 0 : jeton spécial `n_sites` (aucune action entrante).
   - Token t (t >= 1) : `act_idx[:, t - 1]` (transition entrante x[t-1] -> x[t]).
2. Validation stricte des indices d'actions et des transitions à un seul flip
   avant tout clamp ou indexation.
3. Coûts d'arêtes non orientés strictement positifs sur l'hypercube (en mode
   `learned`), coûts unitaires fixes (`fixed`), ou sans biais géométrique (`none`).
4. Action géométrique observée le long du chemin :
       S_obs[b, i, j] = sum_{t=j}^{i-1} c_theta(x[t], act[t])   pour j <= i
5. Antisymétrie exacte par construction via différence algébrique d'un encodeur
   partagé : Omega_hat(a, b) = J_hat(a) - J_hat(b) = -Omega_hat(b, a).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


TransportMode = Literal["integrable", "symmetric"]
GeometryMode = Literal["none", "fixed", "learned"]


# ---------------------------------------------------------------------------
# Packing / unpacking symmetric coupling matrices J (zero diagonal)
# ---------------------------------------------------------------------------

def pack_J(J: np.ndarray) -> np.ndarray:
    """Pack upper-triangular off-diagonal entries of symmetric (N, N) J into (d_J,)."""
    mat = np.asarray(J, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] < 2:
        raise ValueError(f"J doit être une matrice carrée (N, N) avec N >= 2, reçu {mat.shape}.")
    r, c = np.triu_indices(mat.shape[0], k=1)
    return mat[r, c].copy()


def unpack_J(j_vec: np.ndarray, n_sites: int) -> np.ndarray:
    """Reconstruct symmetric (N, N) matrix with zero diagonal from (d_J,) vector."""
    vec = np.asarray(j_vec, dtype=np.float64)
    d_j = n_sites * (n_sites - 1) // 2
    if vec.shape != (d_j,):
        raise ValueError(f"j_vec doit avoir la forme ({d_j},), reçu {vec.shape}.")
    mat = np.zeros((n_sites, n_sites), dtype=np.float64)
    r, c = np.triu_indices(n_sites, k=1)
    mat[r, c] = vec
    mat[c, r] = vec
    return mat


# ---------------------------------------------------------------------------
# Hypercube indexing and strict transition validation
# ---------------------------------------------------------------------------

def states_to_indices(x_seq: torch.Tensor) -> torch.Tensor:
    """Convert binary states in {-1, +1}^N of shape (..., N) to integer indices in [0, 2^N)."""
    n_sites = x_seq.shape[-1]
    bits = (x_seq > 0).to(torch.long)
    weights = (1 << torch.arange(n_sites, device=x_seq.device, dtype=torch.long))
    return (bits * weights).sum(dim=-1)


def _validate_actions_and_transitions(
    x_seq: torch.Tensor,
    act_idx: torch.Tensor,
    valid_transition: torch.Tensor,
) -> torch.Tensor:
    """Validate action domain and single-site hypercube flips before any indexing."""
    if torch.is_floating_point(act_idx) or torch.is_complex(act_idx) or act_idx.dtype == torch.bool:
        raise TypeError(f"act_idx doit être un tenseur d'entiers, reçu {act_idx.dtype}.")
    batch, length, n_sites = x_seq.shape
    if act_idx.shape != (batch, length - 1):
        raise ValueError(f"act_idx doit avoir la forme {(batch, length - 1)}, reçu {act_idx.shape}.")
    if valid_transition.shape != (batch, length - 1) or valid_transition.dtype != torch.bool:
        raise ValueError("valid_transition doit être booléen et de forme (B, L-1).")

    if torch.any(valid_transition):
        active_actions = act_idx[valid_transition]
        if torch.any((active_actions < 0) | (active_actions >= n_sites)):
            raise ValueError(f"act_idx contient une action hors du domaine [0, {n_sites - 1}].")

    safe_actions = torch.where(
        valid_transition,
        act_idx.to(torch.long),
        torch.zeros_like(act_idx, dtype=torch.long),
    )

    if length > 1 and torch.any(valid_transition):
        expected_next = x_seq[:, :-1].clone()
        b_idx = torch.arange(batch, device=x_seq.device)[:, None].expand(batch, length - 1)
        t_idx = torch.arange(length - 1, device=x_seq.device)[None, :].expand(batch, length - 1)
        flip_sign = torch.where(valid_transition, -1.0, 1.0).to(x_seq.dtype)
        expected_next[b_idx, t_idx, safe_actions] = (
            expected_next[b_idx, t_idx, safe_actions] * flip_sign
        )
        if not torch.equal(expected_next[valid_transition], x_seq[:, 1:][valid_transition]):
            raise ValueError(
                "Transition hypercube invalide : x_seq[:, t+1] doit correspondre "
                "au flip exact du seul site act_idx[:, t]."
            )

    return safe_actions


# ---------------------------------------------------------------------------
# Undirected edge costs and observed path action
# ---------------------------------------------------------------------------

class UndirectedPositiveEdgeCosts(nn.Module):
    """Learnable strictly positive undirected edge costs on the N-cube."""

    def __init__(self, n_sites: int, edge_eps: float = 1e-4) -> None:
        super().__init__()
        if n_sites < 2:
            raise ValueError("n_sites doit être >= 2.")
        if edge_eps <= 0.0:
            raise ValueError("edge_eps doit être > 0.")
        self.n_sites = int(n_sites)
        self.edge_eps = float(edge_eps)

        n_states = 1 << self.n_sites
        n_undirected = self.n_sites * (1 << (self.n_sites - 1))
        edge_id_table = torch.full((n_states, self.n_sites), -1, dtype=torch.long)

        next_id = 0
        for state in range(n_states):
            for site in range(self.n_sites):
                neighbour = state ^ (1 << site)
                if state < neighbour:
                    edge_id_table[state, site] = next_id
                    edge_id_table[neighbour, site] = next_id
                    next_id += 1

        assert next_id == n_undirected
        self.register_buffer("edge_id_table", edge_id_table, persistent=False)
        self.edge_logits = nn.Parameter(torch.zeros(n_undirected, dtype=torch.float32))

    def table(self) -> torch.Tensor:
        """Return symmetric (2^N, N) table of strictly positive edge traversal costs."""
        costs = F.softplus(self.edge_logits) + self.edge_eps
        return costs[self.edge_id_table]

    def regularizer(self) -> torch.Tensor:
        """Smooth L2 regularization keeping edge logits well-conditioned."""
        return (self.edge_logits ** 2).mean()

    def all_pairs_shortest_paths(self) -> torch.Tensor:
        """Compute exact all-pairs shortest-path matrix S* on the N-cube (2^N, 2^N)."""
        table = self.table()
        n_states = 1 << self.n_sites
        inf = torch.tensor(float("inf"), device=table.device, dtype=table.dtype)
        dist = torch.full((n_states, n_states), inf, device=table.device, dtype=table.dtype)
        idx = torch.arange(n_states, device=table.device)
        dist[idx, idx] = 0.0
        for site in range(self.n_sites):
            dist[idx, idx ^ (1 << site)] = table[:, site]
        for pivot in range(n_states):
            dist = torch.minimum(dist, dist[:, pivot:pivot + 1] + dist[pivot:pivot + 1, :])
        return dist


def fixed_unit_edge_table(
    n_sites: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return unit-cost edge table (2^N, N) corresponding to Hamming step cost 1."""
    return torch.ones((1 << n_sites, n_sites), device=device, dtype=dtype)


def cumulative_observed_action(
    x_seq: torch.Tensor,
    act_idx: torch.Tensor,
    edge_cost_table: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute causal observed path action matrix S_obs[b, i, j] = sum_{t=j}^{i-1} c(x_t, a_t)."""
    if x_seq.ndim != 3:
        raise ValueError("x_seq doit avoir la forme (B, L, N).")
    batch, length, n_sites = x_seq.shape
    if edge_cost_table.shape != (1 << n_sites, n_sites):
        raise ValueError(
            f"edge_cost_table doit avoir la forme {(1 << n_sites, n_sites)}, "
            f"reçu {edge_cost_table.shape}."
        )
    if pad_mask is None:
        valid_transition = torch.ones((batch, length - 1), device=x_seq.device, dtype=torch.bool)
    else:
        if pad_mask.shape != (batch, length) or pad_mask.dtype != torch.bool:
            raise ValueError("pad_mask doit être booléen et de forme (B, L).")
        valid_transition = (~pad_mask[:, :-1]) & (~pad_mask[:, 1:])

    safe_actions = _validate_actions_and_transitions(x_seq, act_idx, valid_transition)
    state_idx = states_to_indices(x_seq)
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
    """Return max(0, S_obs - S*(endpoints)); diagnostic path excess (B, L, L)."""
    state_idx = states_to_indices(x_seq)
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
                raise ValueError("action_bias doit avoir la forme (B, L, L).")
            scores = scores - float(beta_action) * action_bias[:, None, :, :]

        causal_mask = torch.triu(
            torch.ones((length, length), device=hidden.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))
        if pad_mask is not None:
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
            raise ValueError("dropout doit appartenir à [0, 1).")
        if self.beta_action < 0.0 or self.edge_eps <= 0.0:
            raise ValueError("beta_action doit être >= 0 et edge_eps > 0.")
        if self.geometry_mode not in ("none", "fixed", "learned"):
            raise ValueError("geometry_mode doit être none, fixed ou learned.")


class GSAPathModel(nn.Module):
    """Predict J_final from a forced path using shared path encoders."""

    def __init__(self, config: GSAConfig = GSAConfig()) -> None:
        super().__init__()
        self.config = config
        n = config.n_sites
        d_j = n * (n - 1) // 2
        self.x_proj = nn.Linear(n, config.d_model)
        self.j0_proj = nn.Linear(d_j, config.d_model)
        self.action_embedding = nn.Embedding(n + 1, config.d_model)
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

    def embed_inputs(
        self,
        x_seq: torch.Tensor,
        act_idx: torch.Tensor,
        j0: torch.Tensor,
        pos: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single source of truth for causal path embeddings.

        Token 0 receives special_action = n_sites.
        Token t (t >= 1) receives act_idx[:, t - 1] (incoming transition x[t-1] -> x[t]).
        """
        if x_seq.ndim != 3:
            raise ValueError("x_seq doit avoir la forme (B, L, N).")
        batch, length, n_sites = x_seq.shape
        if n_sites != self.config.n_sites:
            raise ValueError(f"N attendu={self.config.n_sites}, reçu={n_sites}.")
        if length < 2 or length > self.config.max_len:
            raise ValueError(f"La longueur L doit appartenir à [2, {self.config.max_len}].")
        if act_idx.shape != (batch, length - 1):
            raise ValueError("act_idx doit avoir la forme (B, L-1).")
        if j0.shape != (batch, self.d_J):
            raise ValueError(f"j0 doit avoir la forme (B, {self.d_J}).")
        if not torch.is_floating_point(x_seq) or not torch.is_floating_point(j0):
            raise TypeError("x_seq et j0 doivent être des tenseurs flottants.")
        if not torch.isfinite(x_seq).all() or not torch.isfinite(j0).all():
            raise ValueError("x_seq et j0 doivent contenir uniquement des valeurs finies.")

        if pos is None:
            pos = torch.arange(length, device=x_seq.device, dtype=torch.long).expand(batch, length)
        if pos.shape != (batch, length) or pos.dtype != torch.long:
            raise ValueError("pos doit être long et de forme (B, L).")
        if torch.any(pos < 0) or torch.any(pos >= self.config.max_len):
            raise ValueError("pos contient un indice hors de [0, max_len).")

        if pad_mask is None:
            pad_mask = torch.zeros((batch, length), device=x_seq.device, dtype=torch.bool)
        elif pad_mask.shape != (batch, length) or pad_mask.dtype != torch.bool:
            raise ValueError("pad_mask doit être booléen et de forme (B, L).")
        if torch.any(pad_mask.all(dim=1)):
            raise ValueError("Chaque séquence doit contenir au moins un état non rembourré.")
        if length > 1 and torch.any(pad_mask[:, :-1] & ~pad_mask[:, 1:]):
            raise ValueError("pad_mask doit utiliser un remplissage à droite.")
        if torch.any((x_seq != 1.0) & (x_seq != -1.0)):
            raise ValueError("x_seq doit contenir uniquement -1 et +1.")

        valid_incoming = (~pad_mask[:, :-1]) & (~pad_mask[:, 1:])
        safe_actions = _validate_actions_and_transitions(x_seq, act_idx, valid_incoming)

        special_action = self.config.n_sites
        action_tokens = torch.full(
            (batch, length),
            special_action,
            device=x_seq.device,
            dtype=torch.long,
        )
        if length > 1:
            action_tokens[:, 1:] = torch.where(
                valid_incoming,
                safe_actions,
                torch.full_like(safe_actions, special_action),
            )
        action_tokens = action_tokens.masked_fill(pad_mask, special_action)

        hidden = (
            self.x_proj(x_seq)
            + self.action_embedding(action_tokens)
            + self.position_embedding(pos)
            + self.j0_proj(j0).unsqueeze(1)
        )
        hidden = self.input_norm(hidden)
        hidden = hidden.masked_fill(pad_mask[:, :, None], 0.0)
        return hidden, pad_mask

    def forward(
        self,
        x_seq: torch.Tensor,
        act_idx: torch.Tensor,
        j0: torch.Tensor,
        pos: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden, pad_mask = self.embed_inputs(
            x_seq=x_seq,
            act_idx=act_idx,
            j0=j0,
            pos=pos,
            pad_mask=pad_mask,
        )
        batch = hidden.shape[0]
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
    x_a: np.ndarray          # (P, T+1, N)
    x_b: np.ndarray          # (P, T+1, N)
    act_a: np.ndarray        # (P, T)
    act_b: np.ndarray        # (P, T)
    j0: np.ndarray           # (P, d_J)
    j_seq_a: np.ndarray      # (P, T+1, d_J), retained for a coupled-state extension
    j_seq_b: np.ndarray      # (P, T+1, d_J), retained for a coupled-state extension
    j_final_a: np.ndarray    # (P, d_J)
    j_final_b: np.ndarray    # (P, d_J)
    omega: np.ndarray        # (P, d_J) = j_final_a - j_final_b
    initial_state_id: np.ndarray  # (P,)
    site_pair: np.ndarray        # (P, 2)
    mode: str

    @property
    def n_pairs(self) -> int:
        return int(self.x_a.shape[0])


def _rollout_forced_path(X0, J0, actions: Sequence[int], eta: float, mode: TransportMode):
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
    """Build matched (k->m, m->k) pairs using step_flip_node from the project."""
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
        raise ValueError("train_fraction doit être dans ]0, 1[.")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction doit être dans ]0, 1[.")
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
    splits = {
        "train": np.flatnonzero(np.isin(ids, train_groups)),
        "validation": np.flatnonzero(np.isin(ids, val_groups)),
        "test": np.flatnonzero(np.isin(ids, test_groups)),
    }
    train_ids = set(ids[splits["train"]].tolist())
    val_ids = set(ids[splits["validation"]].tolist())
    test_ids = set(ids[splits["test"]].tolist())
    if not (train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)):
        raise RuntimeError("Chevauchement d'initial_state_id détecté entre partitions.")
    return splits


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
    """Predict both orders with identical weights and return Omega_hat = Jhat_a - Jhat_b."""
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
    omega_hat = pred_a - pred_b
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
        raise ValueError("omega_hat et omega_true doivent être de même forme (P, d_J).")
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
    assert np.isclose(shortest_ab, dist[0, 3])
    assert not np.isclose(action_a, action_b)
    return {
        "observed_action_k_then_m": float(action_a),
        "observed_action_m_then_k": float(action_b),
        "shortest_action_between_common_endpoints": float(shortest_ab),
    }


def _build_valid_test_paths(
    batch: int,
    length: int,
    n_sites: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_seq = torch.empty((batch, length, n_sites), dtype=torch.float32)
    act_idx = torch.randint(0, n_sites, (batch, length - 1), generator=gen, dtype=torch.long)
    x_seq[:, 0] = torch.where(torch.rand((batch, n_sites), generator=gen) > 0.5, 1.0, -1.0)
    for t in range(length - 1):
        x_seq[:, t + 1] = x_seq[:, t].clone()
        x_seq[torch.arange(batch), t + 1, act_idx[:, t]] *= -1.0
    return x_seq, act_idx


def _build_same_state_different_incoming_action_path(
    x_seq: torch.Tensor,
    act_idx: torch.Tensor,
    target_token: int,
    n_sites: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a valid hypercube trajectory with identical x[:, target_token]
    but a distinct incoming action act_alt[:, target_token - 1] != act_idx[:, target_token - 1].
    """
    batch = x_seq.shape[0]
    batch_idx = torch.arange(batch, device=x_seq.device)
    x_alt = x_seq.clone()
    act_alt = act_idx.clone()

    new_incoming = (act_idx[:, target_token - 1] + 1) % n_sites
    act_alt[:, target_token - 1] = new_incoming

    x_alt[:, target_token - 1] = x_alt[:, target_token].clone()
    x_alt[batch_idx, target_token - 1, new_incoming] *= -1.0

    for step in range(target_token - 2, -1, -1):
        x_alt[:, step] = x_alt[:, step + 1].clone()
        x_alt[batch_idx, step, act_alt[:, step]] *= -1.0

    return x_alt, act_alt


def self_test() -> None:
    property_result = test_path_vs_shortest_action_property()
    gen = torch.Generator().manual_seed(42)
    batch, length, n_sites = 5, 4, 6
    x_seq, act_idx = _build_valid_test_paths(batch, length, n_sites, gen)
    j0 = torch.randn(batch, n_sites * (n_sites - 1) // 2, generator=gen)

    learned_model = None
    for mode in ("none", "fixed", "learned"):
        cfg = GSAConfig(n_sites=n_sites, d_model=32, n_heads=4, n_layers=2, max_len=8, geometry_mode=mode)
        m = GSAPathModel(cfg)
        m.train()
        pred = m(x_seq, act_idx, j0)
        assert pred.shape == j0.shape
        assert torch.isfinite(pred).all()
        loss = pred.square().mean() + m.geometry_regularizer()
        loss.backward()

        assert any(p.grad is not None for p in m.blocks.parameters()), (
            f"Gradient absent des blocs GSA en geometry_mode={mode}."
        )
        if mode == "learned":
            assert m.edge_costs is not None
            assert m.edge_costs.edge_logits.grad is not None
            assert torch.isfinite(m.edge_costs.edge_logits.grad).all()
            learned_model = m
        else:
            assert m.edge_costs is None

    assert learned_model is not None
    model = learned_model
    model.eval()

    with torch.no_grad():
        emb_ref, _ = model.embed_inputs(x_seq, act_idx, j0)
        for t in range(length - 1):
            act_alt = act_idx.clone()
            act_alt[:, t:] = (act_alt[:, t:] + 1) % n_sites
            x_alt = x_seq.clone()
            for step in range(t, length - 1):
                x_alt[:, step + 1] = x_alt[:, step].clone()
                x_alt[torch.arange(batch), step + 1, act_alt[:, step]] *= -1.0
            emb_alt, _ = model.embed_inputs(x_alt, act_alt, j0)
            torch.testing.assert_close(emb_ref[:, :t + 1], emb_alt[:, :t + 1], atol=1e-6, rtol=0.0)

        for t in range(1, length):
            x_same_state, act_diff_in = _build_same_state_different_incoming_action_path(
                x_seq, act_idx, target_token=t, n_sites=n_sites
            )
            torch.testing.assert_close(x_seq[:, t], x_same_state[:, t], atol=1e-6, rtol=0.0)
            emb_iso, _ = model.embed_inputs(x_same_state, act_diff_in, j0)
            assert not torch.allclose(emb_ref[:, t], emb_iso[:, t], atol=1e-6), (
                f"Le token {t} doit refléter l'action entrante act[:, {t - 1}] à état x_t identique."
            )

    for mode in ("none", "fixed", "learned"):
        m_check = GSAPathModel(GSAConfig(n_sites=6, d_model=16, n_heads=2, n_layers=1, geometry_mode=mode))
        bad_act = act_idx.clone()
        bad_act[0, 0] = -1
        try:
            m_check(x_seq, bad_act, j0)
            raise AssertionError(f"Action négative acceptée en mode {mode}.")
        except ValueError:
            pass

    table = model.edge_costs.table()
    assert table.shape == (1 << n_sites, n_sites)
    assert torch.all(table > 0)
    for state in range(1 << n_sites):
        for site in range(n_sites):
            neighbour = state ^ (1 << site)
            assert torch.allclose(table[state, site], table[neighbour, site])
    shortest = model.shortest_path_matrix()
    assert shortest is not None and shortest.shape == (1 << n_sites, 1 << n_sites)
    assert torch.isfinite(shortest).all()
    assert torch.allclose(shortest, shortest.T, atol=1e-6, rtol=1e-6)
    print("GSA-V1 SELF-TEST: OK")
    print("Property S* / action observée:", property_result)
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
