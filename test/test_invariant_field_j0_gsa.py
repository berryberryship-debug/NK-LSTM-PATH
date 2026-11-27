python -m pip install mlflow --quiet
$invScript = @'
#!/usr/bin/env python3
"""
test_invariant_field_j0_gsa.py

Coûts d'arêtes invariants S_6 x Z_2 gouvernés par le champ local physique :
  c_theta(x, k | J0) = softplus(w0 + f_theta(|h_k(x, J0)|, sqrt(sum_{j!=k} J0_{kj}^2))) + 1e-4
Avec suivi expérimental MLflow (paramètres, métriques par graine et artefacts).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from balayage_commutateur_nk_v2 import ExperimentConfig, sample_binary_symmetric_states
from gsa_v1_core import (
    GSAConfig,
    GSAPathModel,
    build_forced_pair_dataset,
    closure_metrics,
    split_indices_by_initial_state,
)
from test_quenched_j0_t3_gsa import build_t3_quenched_dataset


class InvariantFieldEdgeCosts(nn.Module):
    """Coûts d'arêtes non orientés invariants par permutation et renversement de spins."""

    def __init__(self, n_sites: int = 6, hidden_dim: int = 16, edge_eps: float = 1e-4) -> None:
        super().__init__()
        self.n_sites = n_sites
        self.edge_eps = float(edge_eps)
        n_states = 1 << n_sites

        edge_id_table = torch.full((n_states, n_sites), -1, dtype=torch.long)
        mid_spins: list[torch.Tensor] = []
        edge_sites: list[int] = []
        next_id = 0
        for state in range(n_states):
            for site in range(n_sites):
                neighbour = state ^ (1 << site)
                if state < neighbour:
                    edge_id_table[state, site] = next_id
                    edge_id_table[neighbour, site] = next_id
                    x_u = torch.tensor(
                        [1.0 if ((state >> s) & 1) else -1.0 for s in range(n_sites)],
                        dtype=torch.float32,
                    )
                    x_v = torch.tensor(
                        [1.0 if ((neighbour >> s) & 1) else -1.0 for s in range(n_sites)],
                        dtype=torch.float32,
                    )
                    mid_spins.append(0.5 * (x_u + x_v))
                    edge_sites.append(site)
                    next_id += 1

        triu_rows, triu_cols = torch.triu_indices(n_sites, n_sites, offset=1)
        self.register_buffer("edge_id_table", edge_id_table)
        self.register_buffer("mid_spins", torch.stack(mid_spins, dim=0))
        self.register_buffer("edge_sites", torch.tensor(edge_sites, dtype=torch.long))
        self.register_buffer("triu_rows", triu_rows)
        self.register_buffer("triu_cols", triu_cols)

        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def unpack_symmetric_j0(self, j0: torch.Tensor) -> torch.Tensor:
        batch = j0.shape[0]
        mat = torch.zeros((batch, self.n_sites, self.n_sites), device=j0.device, dtype=j0.dtype)
        mat[:, self.triu_rows, self.triu_cols] = j0
        mat[:, self.triu_cols, self.triu_rows] = j0
        return mat

    def edge_logits(self, j0: torch.Tensor) -> torch.Tensor:
        j_mat = self.unpack_symmetric_j0(j0)
        j_edge_rows = j_mat[:, self.edge_sites, :]
        h_local = (j_edge_rows * self.mid_spins.unsqueeze(0)).sum(dim=-1)
        q_local = torch.sqrt((j_edge_rows ** 2).sum(dim=-1) + 1e-12)
        feats = torch.stack([h_local.abs(), q_local], dim=-1)
        return self.mlp(feats).squeeze(-1)

    def edge_costs(self, j0: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.edge_logits(j0)) + self.edge_eps

    def tables(self, j0: torch.Tensor) -> torch.Tensor:
        return self.edge_costs(j0)[:, self.edge_id_table]

    def regularizer(self, j0: torch.Tensor) -> torch.Tensor:
        return (self.edge_logits(j0) ** 2).mean()


class InvariantFieldGSAPathModel(GSAPathModel):
    def __init__(self, config: GSAConfig, hidden_dim: int = 16) -> None:
        base_cfg = GSAConfig(
            n_sites=config.n_sites,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            ff_mult=config.ff_mult,
            dropout=config.dropout,
            beta_action=config.beta_action,
            max_len=config.max_len,
            edge_eps=config.edge_eps,
            geometry_mode="none",
        )
        super().__init__(base_cfg)
        self.cond_edge_costs = InvariantFieldEdgeCosts(
            n_sites=config.n_sites,
            hidden_dim=hidden_dim,
            edge_eps=config.edge_eps,
        )

    def conditioned_action_bias(
        self,
        x_seq: torch.Tensor,
        act_idx: torch.Tensor,
        j0: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, n_sites = x_seq.shape
        tables = self.cond_edge_costs.tables(j0)
        bits = (x_seq[:, :-1, :] > 0).to(torch.long)
        weights = 1 << torch.arange(n_sites, device=x_seq.device, dtype=torch.long)
        state_idx = (bits * weights).sum(dim=-1)
        b_idx = torch.arange(batch, device=x_seq.device).unsqueeze(1).expand_as(state_idx)
        step_costs = tables[b_idx, state_idx, act_idx]
        if pad_mask is not None:
            valid = (~pad_mask[:, :-1]) & (~pad_mask[:, 1:])
            step_costs = step_costs.masked_fill(~valid, 0.0)
        prefix = torch.zeros((batch, length), device=x_seq.device, dtype=x_seq.dtype)
        prefix[:, 1:] = torch.cumsum(step_costs, dim=1)
        return torch.clamp(prefix[:, :, None] - prefix[:, None, :], min=0.0)

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
        action_bias = self.conditioned_action_bias(x_seq, act_idx, j0, pad_mask)
        for block in self.blocks:
            hidden = block(hidden, action_bias, pad_mask, self.config.beta_action)
        hidden = self.final_norm(hidden)
        lengths = (~pad_mask).sum(dim=1)
        last_valid = lengths - 1
        pooled = hidden[torch.arange(batch, device=x_seq.device), last_valid]
        return j0 + self.delta_head(pooled)


def train_and_eval(
    dataset,
    label: str,
    horizon_tag: str,
    device: torch.device,
    seeds: list[int],
    out_dir: Path,
) -> dict:
    splits = split_indices_by_initial_state(dataset, seed=7, train_fraction=0.60, validation_fraction=0.20)
    tensors = {
        "x_a": torch.as_tensor(dataset.x_a, device=device, dtype=torch.float32),
        "x
