#!/usr/bin/env python3
"""
train_nk_lstm_gsa_v2.py

Entraînement et ablation de l'architecture hybride NK-LSTM-GSA V2,
directement adossée au noyau géométrique `gsa_v1_core.py`.

Architectures testées :
    lstm_only | gsa_only | hybrid_concat | hybrid_gated
"""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gsa_v1_core import (
    ForcedPairDataset,
    GSAConfig,
    GSAPathModel,
    GeometryMode,
    TransportMode,
    build_forced_pair_dataset,
    closure_metrics,
    paired_gsa_loss,
    predict_path_pair,
    split_indices_by_initial_state,
)

ArchMode = Literal["lstm_only", "gsa_only", "hybrid_concat", "hybrid_gated"]


# ============================================================================
# CONFIGURATION (VALIDATION STRICTE)
# ============================================================================


@dataclass(frozen=True)
class TrainV2Config:
    arch_mode: ArchMode = "hybrid_concat"
    geometry_mode: GeometryMode = "learned"

    n_sites: int = 6
    d_model: int = 64
    n_heads: int = 4
    gsa_layers: int = 2
    lstm_layers: int = 2

    ff_mult: int = 4
    dropout: float = 0.0
    beta_action: float = 1.0
    max_len: int = 32
    edge_eps: float = 1e-4

    transport_mode: TransportMode = "symmetric"
    n_initial_states: int = 64
    eta: float = 0.25
    scale_j: float = 0.5

    dataset_npz: str | None = None
    synthetic_smoke: bool = False

    train_fraction: float = 0.60
    validation_fraction: float = 0.20

    batch_size: int = 32
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    lambda_omega: float = 1.0
    lambda_geometry: float = 1e-3

    early_stopping_patience: int = 15
    min_delta: float = 1e-6

    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    self_test: bool = False
    output_dir: str = "runs/nk_lstm_gsa_v2"

    def __post_init__(self) -> None:
        if self.arch_mode not in ("lstm_only", "gsa_only", "hybrid_concat", "hybrid_gated"):
            raise ValueError(f"arch_mode invalide : {self.arch_mode}")
        if self.geometry_mode not in ("none", "fixed", "learned"):
            raise ValueError(f"geometry_mode invalide : {self.geometry_mode}")
        if self.transport_mode not in ("symmetric", "integrable"):
            raise ValueError(f"transport_mode invalide : {self.transport_mode}")

        if self.n_sites < 2:
            raise ValueError(f"n_sites doit être >= 2, reçu {self.n_sites}.")
        if self.d_model < 1 or self.n_heads < 1 or self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) doit être > 0 et divisible par n_heads ({self.n_heads})."
            )
        if self.gsa_layers < 1:
            raise ValueError(f"gsa_layers doit être >= 1, reçu {self.gsa_layers}.")
        if self.lstm_layers < 1:
            raise ValueError(f"lstm_layers doit être >= 1, reçu {self.lstm_layers}.")
        if self.ff_mult < 1:
            raise ValueError(f"ff_mult doit être >= 1, reçu {self.ff_mult}.")
        if self.max_len < 3:
            raise ValueError(f"max_len doit être >= 3 pour ForcedPairDataset (L=3), reçu {self.max_len}.")

        if self.batch_size < 1:
            raise ValueError(f"batch_size doit être >= 1, reçu {self.batch_size}.")
        if self.epochs < 1:
            raise ValueError(f"epochs doit être >= 1, reçu {self.epochs}.")
        if self.early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience doit être >= 1, reçu {self.early_stopping_patience}.")
        if self.dataset_npz is None and not self.self_test and self.n_initial_states < 3:
            raise ValueError(
                f"n_initial_states doit être >= 3 pour construire train/val/test, reçu {self.n_initial_states}."
            )

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout doit appartenir à [0, 1), reçu {self.dropout}.")
        if not (0.0 < self.train_fraction < 1.0):
            raise ValueError(f"train_fraction doit être dans ]0, 1[, reçu {self.train_fraction}.")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError(f"validation_fraction doit être dans ]0, 1[, reçu {self.validation_fraction}.")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction doit être strictement inférieur à 1.0.")

        numeric_fields = {
            "dropout": self.dropout,
            "beta_action": self.beta_action,
            "edge_eps": self.edge_eps,
            "eta": self.eta,
            "scale_j": self.scale_j,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "lambda_omega": self.lambda_omega,
            "lambda_geometry": self.lambda_geometry,
            "min_delta": self.min_delta,
        }
        for field_name, val in numeric_fields.items():
            if not math.isfinite(float(val)):
                raise ValueError(f"Paramètre numérique non fini : {field_name}={val}.")

        if self.beta_action < 0.0:
            raise ValueError(f"beta_action doit être >= 0, reçu {self.beta_action}.")
        if self.edge_eps <= 0.0:
            raise ValueError(f"edge_eps doit être > 0, reçu {self.edge_eps}.")
        if self.lr <= 0.0:
            raise ValueError(f"lr doit être > 0, reçu {self.lr}.")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay doit être >= 0, reçu {self.weight_decay}.")
        if self.grad_clip <= 0.0:
            raise ValueError(f"grad_clip doit être > 0, reçu {self.grad_clip}.")
        if self.lambda_omega < 0.0 or self.lambda_geometry < 0.0:
            raise ValueError("lambda_omega et lambda_geometry doivent être >= 0.")
        if self.min_delta < 0.0:
            raise ValueError(f"min_delta doit être >= 0, reçu {self.min_delta}.")


# ============================================================================
# REPRODUCTIBILITÉ & VALIDATION DATASET
# ============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_binary_states(x: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contient des valeurs non finies.")
    if not np.all((x == 1.0) | (x == -1.0)):
        raise ValueError(f"{name} contient des valeurs autres que -1/+1.")


def _validate_finite_array(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contient des valeurs non finies (NaN ou Inf).")


def _validate_single_flip_transitions(x_seq: np.ndarray, act_idx: np.ndarray, name: str) -> None:
    if x_seq.ndim != 3:
        raise ValueError(f"{name} doit avoir la forme (B, L, N).")
    if act_idx.ndim != 2:
        raise ValueError(f"{name}.act doit avoir la forme (B, L-1).")
    if not np.issubdtype(act_idx.dtype, np.integer):
        raise TypeError(f"{name}: act_idx doit être un tableau d'entiers, reçu {act_idx.dtype}.")

    B, L, N = x_seq.shape
    if act_idx.shape != (B, L - 1):
        raise ValueError(f"{name}: act_idx.shape={act_idx.shape}, attendu={(B, L - 1)}.")

    rows = np.arange(B)
    for t in range(L - 1):
        actions = act_idx[:, t]
        if np.any(actions < 0) or np.any(actions >= N):
            raise ValueError(f"{name}: action hors domaine [0, {N - 1}] à t={t}.")
        expected = x_seq[:, t].copy()
        expected[rows, actions] *= -1.0
        if not np.array_equal(expected, x_seq[:, t + 1]):
            raise ValueError(
                f"{name}: transition invalide à t={t}. "
                f"La transition doit être exactement le flip du site act_idx[:, {t}]."
            )


def validate_forced_pair_dataset(
    dataset: ForcedPairDataset,
    expected_n_sites: int,
    expected_mode: TransportMode | None = None,
    max_len: int = 32,
) -> None:
    required = [
        "x_a", "x_b", "act_a", "act_b", "j0",
        "j_seq_a", "j_seq_b", "j_final_a", "j_final_b",
        "omega", "initial_state_id", "site_pair", "mode",
    ]
    for name in required:
        if not hasattr(dataset, name) or getattr(dataset, name) is None:
            raise ValueError(f"Champ obligatoire absent du ForcedPairDataset : {name}")

    x_a = np.asarray(dataset.x_a)
    x_b = np.asarray(dataset.x_b)
    act_a = np.asarray(dataset.act_a)
    act_b = np.asarray(dataset.act_b)
    j0 = np.asarray(dataset.j0)
    j_seq_a = np.asarray(dataset.j_seq_a)
    j_seq_b = np.asarray(dataset.j_seq_b)
    j_final_a = np.asarray(dataset.j_final_a)
    j_final_b = np.asarray(dataset.j_final_b)
    omega = np.asarray(dataset.omega)
    initial_state_id = np.asarray(dataset.initial_state_id)
    site_pair = np.asarray(dataset.site_pair)

    if x_a.ndim != 3 or x_b.ndim != 3:
        raise ValueError("x_a et x_b doivent être de dimension 3 : (B, L, N).")
    B, L, N = x_a.shape
    if B < 1:
        raise ValueError("Le dataset ne peut pas être vide (B >= 1 requis).")
    if N != expected_n_sites:
        raise ValueError(f"n_sites incohérent : dataset={N}, config={expected_n_sites}.")
    if x_b.shape != x_a.shape:
        raise ValueError(f"x_a/x_b incompatibles : {x_a.shape} vs {x_b.shape}.")
    if L > max_len:
        raise ValueError(f"Longueur de trajectoire L={L} supérieure à max_len={max_len}.")
    if L != 3:
        raise ValueError(f"Le contrat ForcedPairDataset exige exactement L=3 états, reçu L={L}.")

    if not np.issubdtype(act_a.dtype, np.integer) or not np.issubdtype(act_b.dtype, np.integer):
        raise TypeError("act_a et act_b doivent être des tableaux d'entiers.")
    if act_a.shape != (B, 2) or act_b.shape != (B, 2):
        raise ValueError(f"act_a/act_b doivent avoir la forme {(B, 2)}.")

    d_j = N * (N - 1) // 2
    if j0.shape != (B, d_j):
        raise ValueError(f"j0.shape={j0.shape}, attendu={(B, d_j)}.")
    if j_seq_a.shape != (B, 3, d_j) or j_seq_b.shape != (B, 3, d_j):
        raise ValueError(f"j_seq_a/j_seq_b doivent avoir la forme {(B, 3, d_j)}.")
    if j_final_a.shape != (B, d_j) or j_final_b.shape != (B, d_j) or omega.shape != (B, d_j):
        raise ValueError(f"j_final_a/j_final_b/omega doivent avoir la forme {(B, d_j)}.")

    for arr_name, arr in (
        ("j0", j0), ("j_seq_a", j_seq_a), ("j_seq_b", j_seq_b),
        ("j_final_a", j_final_a), ("j_final_b", j_final_b), ("omega", omega),
    ):
        _validate_finite_array(arr, arr_name)

    if not np.issubdtype(initial_state_id.dtype, np.integer) or initial_state_id.shape != (B,):
        raise ValueError(f"initial_state_id invalide : shape={initial_state_id.shape}, dtype={initial_state_id.dtype}.")
    if not np.issubdtype(site_pair.dtype, np.integer) or site_pair.shape != (B, 2):
        raise ValueError(f"site_pair invalide : shape={site_pair.shape}, dtype={site_pair.dtype}.")
    if np.any(site_pair < 0) or np.any(site_pair >= N):
        raise ValueError(f"site_pair contient des indices hors de [0, {N - 1}].")
    if np.any(site_pair[:, 0] == site_pair[:, 1]):
        raise ValueError("site_pair doit contenir deux sites distincts k != m.")

    _validate_binary_states(x_a, "x_a")
    _validate_binary_states(x_b, "x_b")
    _validate_single_flip_transitions(x_a, act_a, "x_a/act_a")
    _validate_single_flip_transitions(x_b, act_b, "x_b/act_b")

    if not np.array_equal(x_a[:, 0], x_b[:, 0]):
        raise ValueError("x_a[:,0] et x_b[:,0] doivent être identiques.")
    if not np.array_equal(x_a[:, -1], x_b[:, -1]):
        raise ValueError("Les chemins A et B n'aboutissent pas au même état final.")

    if np.any(act_a[:, 0] == act_a[:, 1]) or np.any(act_b[:, 0] == act_b[:, 1]):
        raise ValueError("Les deux actions d'un chemin doivent porter sur des sites distincts.")
    pair_expected = np.sort(site_pair, axis=1)
    if not np.array_equal(np.sort(act_a, axis=1), pair_expected):
        raise ValueError("site_pair incohérent avec les deux actions du chemin A.")
    if not np.array_equal(np.sort(act_b, axis=1), pair_expected):
        raise ValueError("site_pair incohérent avec les deux actions du chemin B.")
    if not np.array_equal(act_a[:, 0], act_b[:, 1]) or not np.array_equal(act_a[:, 1], act_b[:, 0]):
        raise ValueError("Les deux chemins doivent inverser exactement leur ordre (k->m vs m->k).")

    np.testing.assert_allclose(j_seq_a[:, 0], j0, atol=1e-6, rtol=0.0, err_msg="j_seq_a[:,0] != j0")
    np.testing.assert_allclose(j_seq_b[:, 0], j0, atol=1e-6, rtol=0.0, err_msg="j_seq_b[:,0] != j0")
    np.testing.assert_allclose(j_seq_a[:, -1], j_final_a, atol=1e-6, rtol=0.0, err_msg="j_seq_a[:,-1] != j_final_a")
    np.testing.assert_allclose(j_seq_b[:, -1], j_final_b, atol=1e-6, rtol=0.0, err_msg="j_seq_b[:,-1] != j_final_b")
    np.testing.assert_allclose(omega, j_final_a - j_final_b, atol=1e-6, rtol=0.0, err_msg="omega != j_final_a - j_final_b")

    dataset_mode = str(dataset.mode)
    if dataset_mode not in ("integrable", "symmetric"):
        raise ValueError(f"Transport mode inconnu dans le dataset : {dataset_mode}.")
    if expected_mode is not None and dataset_mode != str(expected_mode):
        raise ValueError(f"Transport mode incohérent : dataset={dataset_mode}, config={expected_mode}.")

    print(
        f"Dataset validé : B={B}, L={L}, N={N}, d_J={d_j}, "
        f"mode={dataset_mode}, états initiaux={len(np.unique(initial_state_id))}"
    )


# ============================================================================
# MODÈLE HYBRIDE
# ============================================================================


class HybridPathModel(nn.Module):
    """Encodeur de chemin unifié compatible avec predict_path_pair."""

    def __init__(
        self,
        gsa_config: GSAConfig,
        arch_mode: ArchMode = "hybrid_concat",
        lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        if arch_mode not in ("lstm_only", "gsa_only", "hybrid_concat", "hybrid_gated"):
            raise ValueError(f"arch_mode invalide : {arch_mode}")
        if lstm_layers < 1:
            raise ValueError("lstm_layers doit être >= 1.")

        self.config = gsa_config
        self.arch_mode = arch_mode
        self.lstm_layers = int(lstm_layers)
        self.gsa = GSAPathModel(gsa_config)
        d_model = gsa_config.d_model

        self.use_lstm = arch_mode in ("lstm_only", "hybrid_concat", "hybrid_gated")
        self.use_gsa = arch_mode in ("gsa_only", "hybrid_concat", "hybrid_gated")

        if not self.use_gsa:
            self.gsa.blocks = nn.ModuleList()
            self.gsa.final_norm = None
            self.gsa.edge_costs = None

        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=self.lstm_layers,
                batch_first=True,
                dropout=gsa_config.dropout if self.lstm_layers > 1 else 0.0,
            )
            self.lstm_norm = nn.LayerNorm(d_model)
        else:
            self.lstm = None
            self.lstm_norm = None

        if arch_mode == "hybrid_concat":
            self.fusion = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.fusion_gate = None
        elif arch_mode == "hybrid_gated":
            self.fusion_gate = nn.Linear(2 * d_model, d_model)
            self.fusion = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.fusion = None
            self.fusion_gate = None

    @property
    def d_J(self) -> int:
        return self.gsa.d_J

    def geometry_regularizer(self) -> torch.Tensor:
        if not self.use_gsa:
            return next(self.parameters()).new_zeros(())
        return self.gsa.geometry_regularizer()

    def shortest_path_matrix(self) -> torch.Tensor | None:
        if not self.use_gsa:
            return None
        return self.gsa.shortest_path_matrix()

    def _embed_inputs(
        self,
        x_seq: torch.Tensor,
        act_idx: torch.Tensor,
        j0: torch.Tensor,
        pos: torch.Tensor | None,
        pad_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gsa.embed_inputs(x_seq=x_seq, act_idx=act_idx, j0=j0, pos=pos, pad_mask=pad_mask)

    def _encode_branches(
        self,
        x_seq: torch.Tensor,
        act_idx: torch.Tensor,
        j0: torch.Tensor,
        pos: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        embedded, pad_mask = self._embed_inputs(x_seq=x_seq, act_idx=act_idx, j0=j0, pos=pos, pad_mask=pad_mask)
        length = embedded.shape[1]
        lengths = (~pad_mask).sum(dim=1)

        seq_temp = None
        if self.use_lstm:
            assert self.lstm is not None and self.lstm_norm is not None
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            seq_temp, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=length
            )
            seq_temp = self.lstm_norm(seq_temp)
            seq_temp = seq_temp.masked_fill(pad_mask[:, :, None], 0.0)

        seq_geom = None
        if self.use_gsa:
            assert self.gsa.final_norm is not None
            action_bias = self.gsa.path_action_bias(x_seq, act_idx, pad_mask)
            seq_geom = embedded
            for block in self.gsa.blocks:
                seq_geom = block(seq_geom, action_bias, pad_mask, self.config.beta_action)
            seq_geom = self.gsa.final_norm(seq_geom)
            seq_geom = seq_geom.masked_fill(pad_mask[:, :, None], 0.0)

        return embedded, seq_temp, seq_geom, pad_mask

    def forward(
        self,
        x_seq: torch.Tensor,
        act_idx: torch.Tensor,
        j0: torch.Tensor,
        pos: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded, seq_temp, seq_geom, pad_mask = self._encode_branches(
            x_seq=x_seq, act_idx=act_idx, j0=j0, pos=pos, pad_mask=pad_mask
        )
        batch = embedded.shape[0]
        lengths = (~pad_mask).sum(dim=1)
        last_valid = torch.clamp(lengths - 1, min=0)
        batch_idx = torch.arange(batch, device=x_seq.device)

        h_temp = seq_temp[batch_idx, last_valid] if seq_temp is not None else None
        h_geom = seq_geom[batch_idx, last_valid] if seq_geom is not None else None

        if self.arch_mode == "lstm_only":
            assert h_temp is not None
            fused = h_temp
        elif self.arch_mode == "gsa_only":
            assert h_geom is not None
            fused = h_geom
        elif self.arch_mode == "hybrid_concat":
            assert h_temp is not None and h_geom is not None and self.fusion is not None
            fused = self.fusion(torch.cat([h_temp, h_geom], dim=-1))
        elif self.arch_mode == "hybrid_gated":
            assert h_temp is not None and h_geom is not None and self.fusion_gate is not None and self.fusion is not None
            gate = torch.sigmoid(self.fusion_gate(torch.cat([h_temp, h_geom], dim=-1)))
            fused = self.fusion(gate * h_temp + (1.0 - gate) * h_geom)
        else:
            raise RuntimeError(f"Mode architectural non géré : {self.arch_mode}")

        delta_j = self.gsa.delta_head(fused)
        return j0 + delta_j


# ============================================================================
# DATASET PYTORCH, SAVE/LOAD & PRÉPARATION
# ============================================================================


class PairedPathSubset(Dataset):
    def __init__(self, dataset: ForcedPairDataset, indices: np.ndarray) -> None:
        self.x_a = torch.as_tensor(dataset.x_a[indices], dtype=torch.float32)
        self.x_b = torch.as_tensor(dataset.x_b[indices], dtype=torch.float32)
        self.act_a = torch.as_tensor(dataset.act_a[indices], dtype=torch.long)
        self.act_b = torch.as_tensor(dataset.act_b[indices], dtype=torch.long)
        self.j0 = torch.as_tensor(dataset.j0[indices], dtype=torch.float32)
        self.j_seq_a = torch.as_tensor(dataset.j_seq_a[indices], dtype=torch.float32)
        self.j_seq_b = torch.as_tensor(dataset.j_seq_b[indices], dtype=torch.float32)
        self.j_final_a = torch.as_tensor(dataset.j_final_a[indices], dtype=torch.float32)
        self.j_final_b = torch.as_tensor(dataset.j_final_b[indices], dtype=torch.float32)
        self.omega = torch.as_tensor(dataset.omega[indices], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.j0.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_a": self.x_a[idx], "x_b": self.x_b[idx],
            "act_a": self.act_a[idx], "act_b": self.act_b[idx],
            "j0": self.j0[idx],
            "j_seq_a": self.j_seq_a[idx], "j_seq_b": self.j_seq_b[idx],
            "j_final_a": self.j_final_a[idx], "j_final_b": self.j_final_b[idx],
            "omega": self.omega[idx],
        }


def save_forced_pair_dataset(path: Path, dataset: ForcedPairDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_a=dataset.x_a, x_b=dataset.x_b,
        act_a=dataset.act_a, act_b=dataset.act_b,
        j0=dataset.j0, j_seq_a=dataset.j_seq_a, j_seq_b=dataset.j_seq_b,
        j_final_a=dataset.j_final_a, j_final_b=dataset.j_final_b,
        omega=dataset.omega,
        initial_state_id=dataset.initial_state_id,
        site_pair=dataset.site_pair,
        mode=np.array(dataset.mode),
    )


def load_forced_pair_dataset(path: Path) -> ForcedPairDataset:
    with np.load(path, allow_pickle=False) as data:
        required = [
            "x_a", "x_b", "act_a", "act_b", "j0",
            "j_seq_a", "j_seq_b", "j_final_a", "j_final_b",
            "omega", "initial_state_id", "site_pair", "mode",
        ]
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError("NPZ incomplet : champs manquants : " + ", ".join(missing))
        return ForcedPairDataset(
            x_a=data["x_a"], x_b=data["x_b"],
            act_a=data["act_a"], act_b=data["act_b"],
            j0=data["j0"], j_seq_a=data["j_seq_a"], j_seq_b=data["j_seq_b"],
            j_final_a=data["j_final_a"], j_final_b=data["j_final_b"],
            omega=data["omega"],
            initial_state_id=data["initial_state_id"],
            site_pair=data["site_pair"],
            mode=str(data["mode"].item()),
        )


def build_synthetic_smoke_dataset(
    n_sites: int,
    n_initial_states: int,
    eta: float,
    mode: TransportMode,
    seed: int,
) -> ForcedPairDataset:
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {
        "x_a": [], "x_b": [], "act_a": [], "act_b": [], "j0": [],
        "jseq_a": [], "jseq_b": [], "ja": [], "jb": [], "omega": [],
        "state_id": [], "pair": [],
    }
    triu_r, triu_c = np.triu_indices(n_sites, k=1)

    for state_id in range(n_initial_states):
        x0 = rng.choice([-1.0, 1.0], size=(n_sites,))
        raw = rng.normal(0.0, 0.5, size=(n_sites, n_sites))
        j0_mat = 0.5 * (raw + raw.T)
        np.fill_diagonal(j0_mat, 0.0)
        j0_vec = j0_mat[triu_r, triu_c].astype(np.float32)

        for k in range(n_sites):
            for m in range(k + 1, n_sites):
                xa = np.stack([x0, x0.copy(), x0.copy()])
                xa[1, k] *= -1.0
                xa[2, k] *= -1.0
                xa[2, m] *= -1.0

                xb = np.stack([x0, x0.copy(), x0.copy()])
                xb[1, m] *= -1.0
                xb[2, m] *= -1.0
                xb[2, k] *= -1.0

                ja = j0_vec.copy()
                jb = j0_vec.copy()
                if mode == "symmetric":
                    comm_scale = eta * float(x0[k] * x0[m])
                    edge_idx = int(np.flatnonzero((triu_r == k) & (triu_c == m))[0])
                    ja[edge_idx] += comm_scale
                    jb[edge_idx] -= comm_scale

                rows["x_a"].append(xa)
                rows["x_b"].append(xb)
                rows["act_a"].append((k, m))
                rows["act_b"].append((m, k))
                rows["j0"].append(j0_vec)
                rows["jseq_a"].append(np.stack([j0_vec, j0_vec, ja]))
                rows["jseq_b"].append(np.stack([j0_vec, j0_vec, jb]))
                rows["ja"].append(ja)
                rows["jb"].append(jb)
                rows["omega"].append(ja - jb)
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


def prepare_dataset(config: TrainV2Config) -> ForcedPairDataset:
    if config.dataset_npz is not None:
        path = Path(config.dataset_npz)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        dataset = load_forced_pair_dataset(path)
    elif config.synthetic_smoke:
        dataset = build_synthetic_smoke_dataset(
            n_sites=config.n_sites,
            n_initial_states=config.n_initial_states,
            eta=config.eta,
            mode=config.transport_mode,
            seed=config.seed,
        )
    else:
        from balayage_commutateur_nk_v2 import sample_binary_symmetric_states

        rng = np.random.default_rng(config.seed)
        initial_states = sample_binary_symmetric_states(
            n_sites=config.n_sites,
            n_states=config.n_initial_states,
            rng=rng,
            scale=config.scale_j,
        )
        dataset = build_forced_pair_dataset(
            initial_states=initial_states,
            eta=config.eta,
            mode=config.transport_mode,
        )

    validate_forced_pair_dataset(
        dataset,
        expected_n_sites=config.n_sites,
        expected_mode=config.transport_mode,
        max_len=config.max_len,
    )
    return dataset


# ============================================================================
# BASELINE & ENTRAÎNEMENT PAR ÉPOQUE
# ============================================================================


def compute_baseline_metrics(
    j0: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
) -> dict[str, float]:
    baseline_omega = torch.zeros_like(j0)
    true_omega = target_a - target_b
    mse_a = torch.mean((j0 - target_a) ** 2).item()
    mse_b = torch.mean((j0 - target_b) ** 2).item()
    mse_omega = torch.mean((baseline_omega - true_omega) ** 2).item()
    return {
        "baseline_mse_a": float(mse_a),
        "baseline_mse_b": float(mse_b),
        "baseline_mse_terminal": float(0.5 * (mse_a + mse_b)),
        "baseline_mse_omega": float(mse_omega),
    }


def run_epoch(
    model: HybridPathModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    config: TrainV2Config,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float | int]]:
    training = optimizer is not None
    model.train(training)

    totals = {"loss_total": 0.0, "loss_terminal": 0.0, "loss_omega": 0.0, "loss_geometry_regularizer": 0.0}
    baseline_totals = {"baseline_mse_a": 0.0, "baseline_mse_b": 0.0, "baseline_mse_terminal": 0.0, "baseline_mse_omega": 0.0}
    n_samples = 0
    omega_hats: list[np.ndarray] = []
    omega_trues: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            x_a = batch["x_a"].to(device)
            x_b = batch["x_b"].to(device)
            act_a = batch["act_a"].to(device)
            act_b = batch["act_b"].to(device)
            j0 = batch["j0"].to(device)
            target_a = batch["j_final_a"].to(device)
            target_b = batch["j_final_b"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            pred_a, pred_b, omega_hat = predict_path_pair(model, x_a, act_a, x_b, act_b, j0)
            loss, info = paired_gsa_loss(
                pred_a=pred_a,
                pred_b=pred_b,
                target_a=target_a,
                target_b=target_b,
                lambda_omega=config.lambda_omega,
                lambda_geometry=config.lambda_geometry,
                model=model,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError("Perte non finie détectée pendant run_epoch.")

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            bs = int(j0.shape[0])
            n_samples += bs
            totals["loss_total"] += float(info["loss_total"]) * bs
            totals["loss_terminal"] += float(info["loss_terminal"]) * bs
            omega_loss = float(info.get("loss_closure", info.get("loss_omega", 0.0)))
            totals["loss_omega"] += omega_loss * bs
            totals["loss_geometry_regularizer"] += float(info["loss_geometry_regularizer"]) * bs

            batch_baseline = compute_baseline_metrics(j0=j0, target_a=target_a, target_b=target_b)
            for key in baseline_totals:
                baseline_totals[key] += batch_baseline[key] * bs

            omega_hats.append(omega_hat.detach().cpu().numpy())
            omega_trues.append((target_a - target_b).detach().cpu().numpy())

    denom = max(n_samples, 1)
    avg_losses = {k: v / denom for k, v in totals.items()}
    avg_baseline = {k: v / denom for k, v in baseline_totals.items()}

    omega_pred = np.concatenate(omega_hats, axis=0)
    omega_true = np.concatenate(omega_trues, axis=0)
    omega_metrics = closure_metrics(omega_pred, omega_true)

    metrics = {**omega_metrics, **avg_baseline}
    metrics["delta_mse_omega_vs_baseline"] = float(omega_metrics["mse"]) - avg_baseline["baseline_mse_omega"]
    return avg_losses, metrics


# ============================================================================
# SELF-TEST COMPLET
# ============================================================================


def build_self_test_batch(
    n_sites: int,
    batch_size: int = 3,
    length: int = 4,
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if length < 2:
        raise ValueError("length doit être >= 2.")
    rng = np.random.default_rng(seed)
    x = np.empty((batch_size, length, n_sites), dtype=np.float32)
    act = np.empty((batch_size, length - 1), dtype=np.int64)
    d_j = n_sites * (n_sites - 1) // 2
    j0 = rng.normal(0.0, 0.5, size=(batch_size, d_j)).astype(np.float32)

    for b in range(batch_size):
        x[b, 0] = rng.choice([-1.0, 1.0], size=n_sites)
        for t in range(length - 1):
            a = int(rng.integers(0, n_sites))
            act[b, t] = a
            x[b, t + 1] = x[b, t].copy()
            x[b, t + 1, a] *= -1.0

    return torch.from_numpy(x), torch.from_numpy(act), torch.from_numpy(j0)


def _rollout_from_prefix(x_base: torch.Tensor, act_alt: torch.Tensor, start_step: int) -> torch.Tensor:
    x_alt = x_base.clone()
    batch, length, _ = x_alt.shape
    batch_idx = torch.arange(batch, device=x_alt.device)
    for t in range(start_step, length - 1):
        x_alt[:, t + 1] = x_alt[:, t].clone()
        x_alt[batch_idx, t + 1, act_alt[:, t]] *= -1.0
    return x_alt


def _build_same_state_different_incoming_action_path(
    x_base: torch.Tensor,
    act_base: torch.Tensor,
    target_token: int,
    n_sites: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construit une trajectoire valide atteignant exactement x_base[:, target_token]
    via une action entrante différente act_alt[:, target_token - 1] != act_base[:, target_token - 1].
    """
    batch = x_base.shape[0]
    batch_idx = torch.arange(batch, device=x_base.device)
    x_alt = x_base.clone()
    act_alt = act_base.clone()

    new_incoming = (act_base[:, target_token - 1] + 1) % n_sites
    act_alt[:, target_token - 1] = new_incoming

    x_alt[:, target_token - 1] = x_alt[:, target_token].clone()
    x_alt[batch_idx, target_token - 1, new_incoming] *= -1.0

    for step in range(target_token - 2, -1, -1):
        x_alt[:, step] = x_alt[:, step + 1].clone()
        x_alt[batch_idx, step, act_alt[:, step]] *= -1.0

    return x_alt, act_alt


def self_test(config: TrainV2Config) -> None:
    if config.max_len < 4:
        raise ValueError(f"--self-test exige max_len >= 4, reçu max_len={config.max_len}.")

    print()
    print("=" * 72)
    print("SELF-TEST NK-LSTM-GSA V2 (CAUSALITÉ & SYNCHRONISATION NOYAU)")
    print("=" * 72)

    device = torch.device(config.device)
    gsa_cfg = GSAConfig(
        n_sites=config.n_sites,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.gsa_layers,
        ff_mult=config.ff_mult,
        dropout=0.0,
        beta_action=config.beta_action,
        max_len=config.max_len,
        edge_eps=config.edge_eps,
        geometry_mode=config.geometry_mode,
    )

    x, act, j0 = build_self_test_batch(n_sites=config.n_sites, batch_size=3, length=4, seed=config.seed)
    x, act, j0 = x.to(device), act.to(device), j0.to(device)
    pad_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=device)

    # 1. Test des 4 architectures + gradients conditionnés au mode géométrique
    for arch_mode in ("lstm_only", "gsa_only", "hybrid_concat", "hybrid_gated"):
        print(f"\n[TEST] arch_mode={arch_mode} (geometry_mode={gsa_cfg.geometry_mode})")
        model = HybridPathModel(gsa_config=gsa_cfg, arch_mode=arch_mode, lstm_layers=config.lstm_layers).to(device)
        model.train()

        out = model(x_seq=x, act_idx=act, j0=j0, pad_mask=pad_mask)
        assert out.shape == (x.shape[0], j0.shape[-1])
        loss = out.square().mean() + model.geometry_regularizer()
        loss.backward()

        if model.use_lstm:
            assert model.lstm is not None
            assert any(p.grad is not None for p in model.lstm.parameters())

        if model.use_gsa:
            assert any(p.grad is not None for p in model.gsa.blocks.parameters())
            if gsa_cfg.geometry_mode == "learned":
                assert model.gsa.edge_costs is not None
                assert model.gsa.edge_costs.edge_logits.grad is not None
            else:
                assert model.gsa.edge_costs is None
        else:
            assert len(model.gsa.blocks) == 0 and model.gsa.final_norm is None and model.gsa.edge_costs is None

        print("  forward & gradient : OK")

    # Vérification explicite des 3 régimes géométriques none / fixed / learned
    print("\n[TEST] régimes géométriques (none, fixed, learned)")
    for geom_mode in ("none", "fixed", "learned"):
        geom_cfg = GSAConfig(
            n_sites=config.n_sites, d_model=config.d_model, n_heads=config.n_heads,
            n_layers=config.gsa_layers, ff_mult=config.ff_mult, dropout=0.0,
            beta_action=config.beta_action, max_len=config.max_len,
            edge_eps=config.edge_eps, geometry_mode=geom_mode,
        )
        geom_model = HybridPathModel(gsa_config=geom_cfg, arch_mode="gsa_only", lstm_layers=config.lstm_layers).to(device)
        geom_model.train()
        bias = geom_model.gsa.path_action_bias(x, act, pad_mask)
        if geom_mode == "none":
            assert bias is None and geom_model.gsa.edge_costs is None
        elif geom_mode == "fixed":
            assert bias is not None and geom_model.gsa.edge_costs is None
        else:
            assert bias is not None and geom_model.gsa.edge_costs is not None

        pred_geom = geom_model(x_seq=x, act_idx=act, j0=j0, pad_mask=pad_mask)
        (pred_geom.square().mean() + geom_model.geometry_regularizer()).backward()
        assert any(p.grad is not None for p in geom_model.gsa.blocks.parameters())
        if geom_mode == "learned":
            assert geom_model.gsa.edge_costs.edge_logits.grad is not None
    print("  régimes géométriques : OK")

    # 2. Synchronisation exacte noyau GSAPathModel <-> HybridPathModel (gsa_only)
    print("\n[TEST] synchronisation GSAPathModel <-> HybridPathModel (gsa_only)")
    hybrid_gsa = HybridPathModel(gsa_config=gsa_cfg, arch_mode="gsa_only", lstm_layers=config.lstm_layers).to(device)
    hybrid_gsa.eval()
    with torch.no_grad():
        out_core = hybrid_gsa.gsa(x_seq=x, act_idx=act, j0=j0, pad_mask=pad_mask)
        out_hybrid = hybrid_gsa(x_seq=x, act_idx=act, j0=j0, pad_mask=pad_mask)
    torch.testing.assert_close(out_core, out_hybrid, atol=1e-6, rtol=1e-6)
    print("  équivalence noyau : OK")

    # 3. Antisymétrie exacte Omega
    print("\n[TEST] antisymétrie Omega")
    model = HybridPathModel(gsa_config=gsa_cfg, arch_mode="hybrid_concat", lstm_layers=config.lstm_layers).to(device)
    model.eval()
    act_b = act.clone()
    act_b[:, 0] = act[:, 1]
    act_b[:, 1] = act[:, 0]
    x_b = _rollout_from_prefix(x, act_b, start_step=0)
    _, _, omega_ab = predict_path_pair(model, x, act, x_b, act_b, j0)
    _, _, omega_ba = predict_path_pair(model, x_b, act_b, x, act, j0)
    torch.testing.assert_close(omega_ab, -omega_ba, atol=1e-6, rtol=1e-6)
    print("  antisymétrie : OK")

    # 4. Causalité complète : préfixes (embeddings + LSTM + GSA) + cohérence + canal d'action isolé
    print("\n[TEST] causalité stricte sur tous les préfixes (embeddings, LSTM, GSA)")
    length = x.shape[1]
    with torch.no_grad():
        emb_ref, lstm_ref, gsa_ref, _ = model._encode_branches(x, act, j0, pos=None, pad_mask=pad_mask)
        assert lstm_ref is not None and gsa_ref is not None

        for t in range(length - 1):
            act_future = act.clone()
            act_future[:, t:] = (act_future[:, t:] + 1) % config.n_sites
            x_future = _rollout_from_prefix(x, act_future, start_step=t)
            emb_alt, lstm_alt, gsa_alt, _ = model._encode_branches(x_future, act_future, j0, pos=None, pad_mask=pad_mask)
            assert lstm_alt is not None and gsa_alt is not None

            torch.testing.assert_close(emb_ref[:, : t + 1], emb_alt[:, : t + 1], atol=1e-6, rtol=0.0)
            torch.testing.assert_close(lstm_ref[:, : t + 1], lstm_alt[:, : t + 1], atol=1e-6, rtol=0.0)
            torch.testing.assert_close(gsa_ref[:, : t + 1], gsa_alt[:, : t + 1], atol=1e-6, rtol=0.0)
            assert not torch.allclose(lstm_ref[:, t + 1 :], lstm_alt[:, t + 1 :], atol=1e-6)

        for t in range(1, length):
            act_phys = act.clone()
            act_phys[:, t - 1] = (act_phys[:, t - 1] + 1) % config.n_sites
            x_phys = _rollout_from_prefix(x, act_phys, start_step=t - 1)
            emb_phys, _, _, _ = model._encode_branches(x_phys, act_phys, j0, pos=None, pad_mask=pad_mask)
            torch.testing.assert_close(emb_ref[:, :t], emb_phys[:, :t], atol=1e-6, rtol=0.0)
            assert not torch.allclose(emb_ref[:, t], emb_phys[:, t], atol=1e-6)

        for t in range(1, length):
            x_same_state, act_diff_in = _build_same_state_different_incoming_action_path(
                x_base=x, act_base=act, target_token=t, n_sites=config.n_sites
            )
            torch.testing.assert_close(x[:, t], x_same_state[:, t], atol=1e-6, rtol=0.0)
            emb_iso, _, _, _ = model._encode_branches(x_same_state, act_diff_in, j0, pos=None, pad_mask=pad_mask)
            assert not torch.allclose(emb_ref[:, t], emb_iso[:, t], atol=1e-6)

    print("  filtration causale (préfixes + LSTM + GSA + cohérence + canal d'action) : OK")

    # 5. Rejet strict des actions invalides & disjonction du split
    print("\n[TEST] rejet strict des actions invalides & disjonction du split")
    for bad_val in (-1, config.n_sites):
        bad_act = act.clone()
        bad_act[0, 0] = bad_val
        try:
            model(x_seq=x, act_idx=bad_act, j0=j0, pad_mask=pad_mask)
            raise AssertionError(f"Action invalide ({bad_val}) acceptée.")
        except ValueError:
            pass

    smoke_ds = build_synthetic_smoke_dataset(
        n_sites=config.n_sites, n_initial_states=6, eta=config.eta, mode="symmetric", seed=config.seed
    )
    validate_forced_pair_dataset(smoke_ds, expected_n_sites=config.n_sites, expected_mode="symmetric", max_len=config.max_len)
    smoke_splits = split_indices_by_initial_state(
        smoke_ds, seed=config.seed, train_fraction=config.train_fraction, validation_fraction=config.validation_fraction
    )
    tr_ids = set(smoke_ds.initial_state_id[smoke_splits["train"]].tolist())
    va_ids = set(smoke_ds.initial_state_id[smoke_splits["validation"]].tolist())
    te_ids = set(smoke_ds.initial_state_id[smoke_splits["test"]].tolist())
    assert tr_ids.isdisjoint(va_ids) and tr_ids.isdisjoint(te_ids) and va_ids.isdisjoint(te_ids)

    print("  validation actions & split : OK")
    print()
    print("=" * 72)
    print("SELF-TEST : TOUS LES TESTS SONT VALIDES")
    print("=" * 72)


# ============================================================================
# CLI & MAIN
# ============================================================================


def parse_args() -> TrainV2Config:
    parser = argparse.ArgumentParser(description="Entraînement NK-LSTM-GSA V2 sur ForcedPairDataset.")
    parser.add_argument("--arch-mode", choices=["lstm_only", "gsa_only", "hybrid_concat", "hybrid_gated"], default="hybrid_concat")
    parser.add_argument("--geometry-mode", choices=["none", "fixed", "learned"], default="learned")
    parser.add_argument("--transport-mode", choices=["symmetric", "integrable"], default="symmetric")
    parser.add_argument("--n-sites", type=int, default=6)
    parser.add_argument("--n-initial-states", type=int, default=64)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--scale-j", type=float, default=0.5)
    parser.add_argument("--dataset-npz", type=str, default=None)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--gsa-layers", type=int, default=2)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--beta-action", type=float, default=1.0)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--edge-eps", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-omega", type=float, default=1.0)
    parser.add_argument("--lambda-geometry", type=float, default=1e-3)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--early-stopping-patience", "--epochs-patience", dest="early_stopping_patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="runs/nk_lstm_gsa_v2")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    return TrainV2Config(
        arch_mode=args.arch_mode,
        geometry_mode=args.geometry_mode,
        transport_mode=args.transport_mode,
        n_sites=args.n_sites,
        d_model=args.d_model,
        n_heads=args.n_heads,
        gsa_layers=args.gsa_layers,
        lstm_layers=args.lstm_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        beta_action=args.beta_action,
        max_len=args.max_len,
        edge_eps=args.edge_eps,
        n_initial_states=args.n_initial_states,
        eta=args.eta,
        scale_j=args.scale_j,
        dataset_npz=args.dataset_npz,
        synthetic_smoke=args.synthetic_smoke,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lambda_omega=args.lambda_omega,
        lambda_geometry=args.lambda_geometry,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        seed=args.seed,
        device=args.device,
        self_test=args.self_test,
        output_dir=args.output_dir,
    )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    if config.self_test:
        self_test(config)
        return

    device = torch.device(config.device)
    print("\n" + "=" * 72)
    print("NK-LSTM-GSA V2")
    print("=" * 72)
    print(f"Architecture : {config.arch_mode}")
    print(f"Géométrie   : {config.geometry_mode}")
    print(f"Transport    : {config.transport_mode}")
    print(f"Device       : {device}")
    print(f"Seed         : {config.seed}")

    if config.arch_mode == "lstm_only" and config.geometry_mode == "learned":
        warnings.warn(
            "geometry_mode='learned' est sans effet sur le forward en arch_mode='lstm_only' : la voie GSA est désactivée."
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    dataset = prepare_dataset(config)
    if int(dataset.x_a.shape[-1]) != config.n_sites:
        raise ValueError(f"Incohérence n_sites : dataset={dataset.x_a.shape[-1]}, config={config.n_sites}.")

    splits = split_indices_by_initial_state(
        dataset=dataset,
        seed=config.seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    if len(splits["train"]) == 0 or len(splits["validation"]) == 0 or len(splits["test"]) == 0:
        raise ValueError("Un des splits est vide.")

    train_ids = set(dataset.initial_state_id[splits["train"]].tolist())
    val_ids = set(dataset.initial_state_id[splits["validation"]].tolist())
    test_ids = set(dataset.initial_state_id[splits["test"]].tolist())
    if not (train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)):
        raise RuntimeError("Fuite de groupe détectée : chevauchement d'initial_state_id entre partitions.")

    print("\nSplit groupé (disjonction des états initiaux vérifiée) :")
    print(f"  train      = {len(splits['train'])} ({len(train_ids)} états)")
    print(f"  validation = {len(splits['validation'])} ({len(val_ids)} états)")
    print(f"  test       = {len(splits['test'])} ({len(test_ids)} états)")

    gen = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(PairedPathSubset(dataset, splits["train"]), batch_size=config.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(PairedPathSubset(dataset, splits["validation"]), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(PairedPathSubset(dataset, splits["test"]), batch_size=config.batch_size, shuffle=False)

    gsa_cfg = GSAConfig(
        n_sites=config.n_sites,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.gsa_layers,
        ff_mult=config.ff_mult,
        dropout=config.dropout,
        beta_action=config.beta_action,
        max_len=config.max_len,
        edge_eps=config.edge_eps,
        geometry_mode=config.geometry_mode,
    )

    model = HybridPathModel(gsa_config=gsa_cfg, arch_mode=config.arch_mode, lstm_layers=config.lstm_layers).to(device)
    active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParamètres entraînables actifs : {active_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    patience = 0
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        train_losses, train_metrics = run_epoch(model, train_loader, optimizer, config, device)
        val_losses, val_metrics = run_epoch(model, val_loader, None, config, device)

        history.append({
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        })

        print(
            f"Epoch {epoch:03d} | train={train_losses['loss_total']:.6f} | "
            f"val={val_losses['loss_total']:.6f} | Omega MSE={val_metrics['mse']:.6f} | "
            f"Baseline Omega MSE={val_metrics['baseline_mse_omega']:.6f} | "
            f"ΔOmega={val_metrics['delta_mse_omega_vs_baseline']:.6f}"
        )

        if val_losses["loss_total"] < best_val_loss - config.min_delta:
            best_val_loss = val_losses["loss_total"]
            patience = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": asdict(config),
            }, output_dir / "best.pt")
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print(f"Arrêt anticipé à l'époque {epoch} (meilleure val={best_val_loss:.6f}).")
                break

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best_path = output_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("Aucun checkpoint best.pt n'a été produit.")

    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    test_losses, test_metrics = run_epoch(model, test_loader, None, config, device)
    summary = {
        "arch_mode": config.arch_mode,
        "geometry_mode": config.geometry_mode,
        "transport_mode": config.transport_mode,
        "active_trainable_parameters": active_params,
        "best_val_loss": best_val_loss,
        "test_losses": test_losses,
        "test_metrics": test_metrics,
        "baseline": {
            "mse_terminal": test_metrics["baseline_mse_terminal"],
            "mse_omega": test_metrics["baseline_mse_omega"],
        },
        "model_vs_baseline": {
            "delta_mse_omega": test_metrics["delta_mse_omega_vs_baseline"],
        },
    }

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("RÉSULTATS TEST")
    print("=" * 72)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
