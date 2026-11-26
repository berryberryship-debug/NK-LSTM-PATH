#!/usr/bin/env python3
"""
train_nk_lstm_gsa_v2.py

Entraînement et ablation de l'architecture hybride NK-LSTM-GSA V2,
directement adossée au noyau géométrique `gsa_v1_core.py`.

Architecture testée :
    lstm_only
    gsa_only
    hybrid_concat
    hybrid_gated

Invariants :
- États x_seq dans {-1, +1}^N.
- Causalité stricte (filtration temporelle) :
  Le token 0 ne reçoit aucune action (jeton spécial N).
  Le token t (t >= 1) reçoit act_idx[:, t-1], c'est-à-dire l'action
  qui a permis la transition de x[t-1] vers x[t].
  Le token t ne reçoit donc aucune action future act[t] ou ultérieure.
- Action géométrique observée :
      S_obs[b, i, j] = sum_{t=j}^{i-1} c_theta(x[t], act[t])
- Antisymétrie exacte :
      Omega_hat(a,b) = -Omega_hat(b,a)
  via la différence algébrique des deux prédictions de chemin.
- Split groupé par initial_state_id.
- Baseline explicite :
      J_hat(t) = J0
      Omega_hat(t) = 0

IMPORTANT :
Cette V2 entraîne ici la prédiction terminale de J et du commutateur
d'ordre Omega. Elle ne constitue pas encore la version complète de
navigation spatiale à pertes multiples.
"""

from __future__ import annotations

import argparse
import json
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


ArchMode = Literal[
    "lstm_only",
    "gsa_only",
    "hybrid_concat",
    "hybrid_gated",
]


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class TrainV2Config:
    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # NK / dataset
    # ------------------------------------------------------------------
    transport_mode: TransportMode = "symmetric"

    n_initial_states: int = 64
    eta: float = 0.25
    scale_j: float = 0.5

    dataset_npz: str | None = None
    synthetic_smoke: bool = False

    train_fraction: float = 0.60
    validation_fraction: float = 0.20

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    batch_size: int = 32
    epochs: int = 60

    lr: float = 1e-3
    weight_decay: float = 1e-5

    grad_clip: float = 1.0

    lambda_omega: float = 1.0
    lambda_geometry: float = 1e-3

    early_stopping_patience: int = 15
    min_delta: float = 1e-6

    # ------------------------------------------------------------------
    # Reproductibilité / sorties / exécution
    # ------------------------------------------------------------------
    seed: int = 7

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    self_test: bool = False
    output_dir: str = "runs/nk_lstm_gsa_v2"


# ============================================================================
# REPRODUCTIBILITÉ
# ============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# VALIDATION DATASET
# ============================================================================


def _assert_same_shape(name: str, a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Incohérence de forme pour {name} : "
            f"{a.shape} != {b.shape}"
        )


def _validate_binary_states(x: np.ndarray, name: str) -> None:
    if not np.all((x == 1.0) | (x == -1.0)):
        raise ValueError(
            f"{name} contient des valeurs autres que -1/+1."
        )


def _validate_single_flip_transitions(
    x_seq: np.ndarray,
    act_idx: np.ndarray,
    name: str,
) -> None:
    """
    Vérifie que chaque transition t -> t+1 correspond exactement
    au flip du seul site act_idx[t].
    """
    if x_seq.ndim != 3:
        raise ValueError(
            f"{name} doit avoir la forme (B,L,N)."
        )

    if act_idx.ndim != 2:
        raise ValueError(
            f"{name}.act doit avoir la forme (B,L-1)."
        )

    B, L, N = x_seq.shape

    if act_idx.shape != (B, L - 1):
        raise ValueError(
            f"{name}: act_idx.shape={act_idx.shape}, "
            f"attendu={(B, L - 1)}."
        )

    for t in range(L - 1):
        actions = act_idx[:, t]

        if np.any(actions < 0) or np.any(actions >= N):
            raise ValueError(
                f"{name}: action hors domaine à t={t}."
            )

        expected = x_seq[:, t].copy()

        rows = np.arange(B)
        expected[rows, actions] *= -1.0

        if not np.array_equal(expected, x_seq[:, t + 1]):
            raise ValueError(
                f"{name}: transition invalide à t={t}. "
                f"La transition doit être exactement le flip "
                f"du site act_idx[:, {t}]."
            )


def validate_forced_pair_dataset(
    dataset: ForcedPairDataset,
    expected_n_sites: int,
    expected_mode: TransportMode | None = None,
) -> None:
    """
    Validation structurelle complète avant entraînement.

    Vérifie :
    - dimensions ;
    - états binaires ;
    - transitions hypercube ;
    - J0 au premier état ;
    - J_final cohérent avec j_seq[-1] ;
    - omega = J_final_a - J_final_b ;
    - état initial identique entre les deux ordres ;
    - état final identique entre les deux ordres ;
    - cohérence des deux actions forcées.
    """

    required = [
        "x_a",
        "x_b",
        "act_a",
        "act_b",
        "j0",
        "j_seq_a",
        "j_seq_b",
        "j_final_a",
        "j_final_b",
        "omega",
        "initial_state_id",
        "site_pair",
    ]

    for name in required:
        if not hasattr(dataset, name):
            raise ValueError(
                f"Champ obligatoire absent du ForcedPairDataset : {name}"
            )

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

    # ------------------------------------------------------------------
    # Dimensions principales
    # ------------------------------------------------------------------

    if x_a.ndim != 3 or x_b.ndim != 3:
        raise ValueError(
            "x_a et x_b doivent être de dimension 3 : (B,L,N)."
        )

    B, L, N = x_a.shape

    if N != expected_n_sites:
        raise ValueError(
            f"n_sites incohérent : dataset={N}, "
            f"config={expected_n_sites}."
        )

    if x_b.shape != x_a.shape:
        raise ValueError(
            f"x_a/x_b incompatibles : "
            f"{x_a.shape} vs {x_b.shape}."
        )

    if L < 2:
        raise ValueError("Une trajectoire doit contenir au moins 2 états.")

    if L > 32:
        warnings.warn(
            f"Le dataset contient L={L} états. "
            "Vérifier max_len avant entraînement."
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    if act_a.shape != (B, L - 1):
        raise ValueError(
            f"act_a.shape={act_a.shape}, attendu={(B, L - 1)}."
        )

    if act_b.shape != (B, L - 1):
        raise ValueError(
            f"act_b.shape={act_b.shape}, attendu={(B, L - 1)}."
        )

    # ------------------------------------------------------------------
    # J
    # ------------------------------------------------------------------

    if j0.ndim != 2:
        raise ValueError(
            f"j0 doit être 2D, reçu {j0.shape}."
        )

    d_j = N * (N - 1) // 2

    if j0.shape != (B, d_j):
        raise ValueError(
            f"j0.shape={j0.shape}, attendu={(B, d_j)}."
        )

    if j_seq_a.shape != (B, L, d_j):
        raise ValueError(
            f"j_seq_a.shape={j_seq_a.shape}, "
            f"attendu={(B, L, d_j)}."
        )

    if j_seq_b.shape != (B, L, d_j):
        raise ValueError(
            f"j_seq_b.shape={j_seq_b.shape}, "
            f"attendu={(B, L, d_j)}."
        )

    if j_final_a.shape != (B, d_j):
        raise ValueError(
            f"j_final_a.shape={j_final_a.shape}, "
            f"attendu={(B, d_j)}."
        )

    if j_final_b.shape != (B, d_j):
        raise ValueError(
            f"j_final_b.shape={j_final_b.shape}, "
            f"attendu={(B, d_j)}."
        )

    if omega.shape != (B, d_j):
        raise ValueError(
            f"omega.shape={omega.shape}, "
            f"attendu={(B, d_j)}."
        )

    # ------------------------------------------------------------------
    # IDs
    # ------------------------------------------------------------------

    if initial_state_id.shape != (B,):
        raise ValueError(
            f"initial_state_id.shape={initial_state_id.shape}, "
            f"attendu={(B,)}."
        )

    if site_pair.ndim != 2 or site_pair.shape != (B, 2):
        raise ValueError(
            f"site_pair.shape={site_pair.shape}, "
            f"attendu={(B, 2)}."
        )

    # ------------------------------------------------------------------
    # États
    # ------------------------------------------------------------------

    _validate_binary_states(x_a, "x_a")
    _validate_binary_states(x_b, "x_b")

    # ------------------------------------------------------------------
    # Trajectoires : une action = un flip
    # ------------------------------------------------------------------

    _validate_single_flip_transitions(
        x_a,
        act_a,
        "x_a/act_a",
    )

    _validate_single_flip_transitions(
        x_b,
        act_b,
        "x_b/act_b",
    )

    # ------------------------------------------------------------------
    # Les deux chemins doivent partir du même état
    # ------------------------------------------------------------------

    if not np.array_equal(x_a[:, 0], x_b[:, 0]):
        raise ValueError(
            "x_a[:,0] et x_b[:,0] doivent être identiques."
        )

    # ------------------------------------------------------------------
    # Les deux ordres doivent conduire au même état final
    # ------------------------------------------------------------------

    if not np.array_equal(x_a[:, -1], x_b[:, -1]):
        raise ValueError(
            "Les chemins A et B n'aboutissent pas au même état final."
        )

    # ------------------------------------------------------------------
    # Site pair cohérent avec les deux premiers actes
    # ------------------------------------------------------------------

    if L >= 3:
        pair_a = np.sort(act_a[:, :2], axis=1)
        pair_b = np.sort(act_b[:, :2], axis=1)

        if not np.array_equal(pair_a, site_pair):
            raise ValueError(
                "site_pair incohérent avec les deux actions du chemin A."
            )

        if not np.array_equal(pair_b, site_pair):
            raise ValueError(
                "site_pair incohérent avec les deux actions du chemin B."
            )

        if not np.array_equal(
            act_a[:, 0],
            act_b[:, 1],
        ):
            raise ValueError(
                "Le chemin A doit commencer par l'action finale "
                "du chemin B dans une paire k→m / m→k."
            )

        if not np.array_equal(
            act_a[:, 1],
            act_b[:, 0],
        ):
            raise ValueError(
                "Les deux chemins doivent inverser exactement leur ordre."
            )

    # ------------------------------------------------------------------
    # J initial
    # ------------------------------------------------------------------

    np.testing.assert_allclose(
        j_seq_a[:, 0],
        j0,
        atol=1e-6,
        rtol=0.0,
        err_msg="j_seq_a[:,0] != j0",
    )

    np.testing.assert_allclose(
        j_seq_b[:, 0],
        j0,
        atol=1e-6,
        rtol=0.0,
        err_msg="j_seq_b[:,0] != j0",
    )

    # ------------------------------------------------------------------
    # J final
    # ------------------------------------------------------------------

    np.testing.assert_allclose(
        j_seq_a[:, -1],
        j_final_a,
        atol=1e-6,
        rtol=0.0,
        err_msg="j_seq_a[:,-1] != j_final_a",
    )

    np.testing.assert_allclose(
        j_seq_b[:, -1],
        j_final_b,
        atol=1e-6,
        rtol=0.0,
        err_msg="j_seq_b[:,-1] != j_final_b",
    )

    # ------------------------------------------------------------------
    # Omega
    # ------------------------------------------------------------------

    np.testing.assert_allclose(
        omega,
        j_final_a - j_final_b,
        atol=1e-6,
        rtol=0.0,
        err_msg="omega != j_final_a - j_final_b",
    )

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    dataset_mode = str(dataset.mode)

    if expected_mode is not None and dataset_mode != str(expected_mode):
        raise ValueError(
            f"Transport mode incohérent : "
            f"dataset={dataset_mode}, config={expected_mode}."
        )

    print(
        f"Dataset validé : "
        f"B={B}, L={L}, N={N}, d_J={d_j}, "
        f"mode={dataset_mode}, "
        f"états initiaux={len(np.unique(initial_state_id))}"
    )


# ============================================================================
# MODÈLE HYBRIDE
# ============================================================================


class HybridPathModel(nn.Module):
    """
    Encodeur de chemin unifié compatible avec predict_path_pair.

    Important :
    `lstm_only` ne signifie pas que toute l'infrastructure GSAPathModel
    est supprimée. Les projections, embeddings et la tête J sont partagés.
    Seule la voie d'attention GSA est désactivée.

    La géométrie apprise n'est donc active que lorsque use_gsa=True.
    """

    def __init__(
        self,
        gsa_config: GSAConfig,
        arch_mode: ArchMode = "hybrid_concat",
        lstm_layers: int = 2,
    ) -> None:

        super().__init__()

        valid_modes = (
            "lstm_only",
            "gsa_only",
            "hybrid_concat",
            "hybrid_gated",
        )

        if arch_mode not in valid_modes:
            raise ValueError(
                f"arch_mode invalide : {arch_mode}"
            )

        if lstm_layers < 1:
            raise ValueError(
                "lstm_layers doit être >= 1."
            )

        self.config = gsa_config
        self.arch_mode = arch_mode
        self.lstm_layers = int(lstm_layers)

        # Infrastructure partagée avec le noyau GSA.
        self.gsa = GSAPathModel(gsa_config)

        d_model = gsa_config.d_model

        self.use_lstm = arch_mode in (
            "lstm_only",
            "hybrid_concat",
            "hybrid_gated",
        )

        self.use_gsa = arch_mode in (
            "gsa_only",
            "hybrid_concat",
            "hybrid_gated",
        )

        # ------------------------------------------------------------------
        # LSTM
        # ------------------------------------------------------------------

        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=self.lstm_layers,
                batch_first=True,
                dropout=(
                    gsa_config.dropout
                    if self.lstm_layers > 1
                    else 0.0
                ),
            )

            self.lstm_norm = nn.LayerNorm(d_model)

        else:
            self.lstm = None
            self.lstm_norm = None

        # ------------------------------------------------------------------
        # Fusion
        # ------------------------------------------------------------------

        if arch_mode == "hybrid_concat":

            self.fusion = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

            self.fusion_gate = None

        elif arch_mode == "hybrid_gated":

            self.fusion_gate = nn.Linear(
                2 * d_model,
                d_model,
            )

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

    # ------------------------------------------------------------------
    # Embedding causal
    # ------------------------------------------------------------------

    def _embed_inputs(
        self,
        x_seq: torch.Tensor,
        act_idx: torch.Tensor,
        j0: torch.Tensor,
        pos: torch.Tensor | None,
        pad_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if x_seq.ndim != 3:
            raise ValueError(
                "x_seq doit avoir la forme (B,L,N)."
            )

        batch, length, n_sites = x_seq.shape

        if n_sites != self.config.n_sites:
            raise ValueError(
                f"N attendu={self.config.n_sites}, reçu={n_sites}."
            )

        if length < 2 or length > self.config.max_len:
            raise ValueError(
                f"La longueur L doit appartenir à [2, {self.config.max_len}]."
            )

        if act_idx.shape != (
            batch,
            length - 1,
        ):
            raise ValueError(
                "act_idx doit avoir la forme (B,L-1)."
            )

        if j0.shape != (
            batch,
            self.d_J,
        ):
            raise ValueError(
                f"j0 doit avoir la forme "
                f"(B,{self.d_J})."
            )

        if (
            not torch.is_floating_point(x_seq)
            or not torch.is_floating_point(j0)
        ):
            raise TypeError(
                "x_seq et j0 doivent être flottants."
            )

        # --------------------------------------------------------------
        # Positions
        # --------------------------------------------------------------

        if pos is None:
            pos = torch.arange(
                length,
                device=x_seq.device,
                dtype=torch.long,
            ).expand(batch, length)

        if (
            pos.shape != (batch, length)
            or pos.dtype != torch.long
        ):
            raise ValueError(
                "pos doit être long et de forme (B,L)."
            )

        if torch.any(pos < 0):
            raise ValueError(
                "pos contient une valeur négative."
            )

        if torch.any(pos >= self.config.max_len):
            raise ValueError(
                "pos contient un indice hors de max_len."
            )

        # --------------------------------------------------------------
        # Padding
        # --------------------------------------------------------------

        if pad_mask is None:

            pad_mask = torch.zeros(
                (batch, length),
                device=x_seq.device,
                dtype=torch.bool,
            )

        elif (
            pad_mask.shape != (batch, length)
            or pad_mask.dtype != torch.bool
        ):
            raise ValueError(
                "pad_mask doit être booléen et de forme (B,L)."
            )

        if torch.any(pad_mask.all(dim=1)):
            raise ValueError(
                "Chaque séquence doit contenir au moins "
                "un état non rembourré."
            )

        # Padding à droite uniquement.
        if length > 1:
            invalid_padding = (
                pad_mask[:, :-1]
                & ~pad_mask[:, 1:]
            )

            if torch.any(invalid_padding):
                raise ValueError(
                    "pad_mask doit utiliser un padding à droite."
                )

        # --------------------------------------------------------------
        # États
        # --------------------------------------------------------------

        if torch.any(
            (x_seq != 1.0)
            & (x_seq != -1.0)
        ):
            raise ValueError(
                "x_seq doit contenir uniquement -1 et +1."
            )

        # --------------------------------------------------------------
        # ACTIONS CAUSALES : Filtration temporelle
        #
        # Token 0 :
        #     aucun acte entrant -> indice spécial n_sites.
        #
        # Token t (t >= 1) :
        #     reçoit act[t-1], c'est-à-dire l'action qui a produit
        #     x_t à partir de x_{t-1}.
        #
        # Ainsi :
        #   token t = f(x_0,...,x_t, a_0,...,a_{t-1}, J0)
        #
        # aucune action future act[t:] n'est accessible.
        # --------------------------------------------------------------

        special_action = self.config.n_sites

        action_tokens = torch.full(
            (batch, length),
            special_action,
            device=x_seq.device,
            dtype=torch.long,
        )

        if length > 1:
            valid_incoming = (
                ~pad_mask[:, 1:]
                & ~pad_mask[:, :-1]
            )

            safe_actions = act_idx.clamp(
                min=0,
                max=self.config.n_sites - 1,
            )

            # Injection sur les indices 1 à L-1
            action_tokens[:, 1:] = torch.where(
                valid_incoming,
                safe_actions,
                torch.full_like(
                    safe_actions,
                    special_action,
                ),
            )

        action_tokens = action_tokens.masked_fill(
            pad_mask,
            special_action,
        )

        # --------------------------------------------------------------
        # Embedding partagé
        # --------------------------------------------------------------

        hidden = (
            self.gsa.x_proj(x_seq)
            + self.gsa.action_embedding(action_tokens)
            + self.gsa.position_embedding(pos)
            + self.gsa.j0_proj(j0).unsqueeze(1)
        )

        hidden = self.gsa.input_norm(hidden)

        hidden = hidden.masked_fill(
            pad_mask[:, :, None],
            0.0,
        )

        return hidden, pad_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_seq: torch.Tensor,
        act_idx: torch.Tensor,
        j0: torch.Tensor,
        pos: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        embedded, pad_mask = self._embed_inputs(
            x_seq=x_seq,
            act_idx=act_idx,
            j0=j0,
            pos=pos,
            pad_mask=pad_mask,
        )

        batch, length, _ = embedded.shape

        lengths = (~pad_mask).sum(dim=1)
        last_valid = torch.clamp(lengths - 1, min=0)

        batch_idx = torch.arange(
            batch,
            device=x_seq.device,
        )

        # ==============================================================
        # VOIE TEMPORELLE
        # ==============================================================

        h_temp = None

        if self.use_lstm:

            assert (
                self.lstm is not None
                and self.lstm_norm is not None
            )

            packed = nn.utils.rnn.pack_padded_sequence(
                embedded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            packed_out, _ = self.lstm(packed)

            seq_temp, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=length,
            )

            seq_temp = self.lstm_norm(seq_temp)

            h_temp = seq_temp[
                batch_idx,
                last_valid,
            ]

        # ==============================================================
        # VOIE GÉOMÉTRIQUE
        # ==============================================================

        h_geom = None

        if self.use_gsa:

            action_bias = self.gsa.path_action_bias(
                x_seq,
                act_idx,
                pad_mask,
            )

            seq_geom = embedded

            for block in self.gsa.blocks:
                seq_geom = block(
                    seq_geom,
                    action_bias,
                    pad_mask,
                    self.config.beta_action,
                )

            seq_geom = self.gsa.final_norm(seq_geom)

            h_geom = seq_geom[
                batch_idx,
                last_valid,
            ]

        # ==============================================================
        # FUSION
        # ==============================================================

        if self.arch_mode == "lstm_only":

            assert h_temp is not None
            fused = h_temp

        elif self.arch_mode == "gsa_only":

            assert h_geom is not None
            fused = h_geom

        elif self.arch_mode == "hybrid_concat":

            assert (
                h_temp is not None
                and h_geom is not None
                and self.fusion is not None
            )

            fused = self.fusion(
                torch.cat(
                    [h_temp, h_geom],
                    dim=-1,
                )
            )

        elif self.arch_mode == "hybrid_gated":

            assert (
                h_temp is not None
                and h_geom is not None
                and self.fusion_gate is not None
                and self.fusion is not None
            )

            gate = torch.sigmoid(
                self.fusion_gate(
                    torch.cat(
                        [h_temp, h_geom],
                        dim=-1,
                    )
                )
            )

            fused = self.fusion(
                gate * h_temp
                + (1.0 - gate) * h_geom
            )

        else:
            raise RuntimeError(
                f"Mode architectural non géré : "
                f"{self.arch_mode}"
            )

        # ==============================================================
        # PRÉDICTION RÉSIDUELLE
        #
        # J_hat = J0 + Delta_J
        #
        # Ce n'est PAS une intégration cumulative de deltas temporels.
        # ==============================================================

        delta_j = self.gsa.delta_head(fused)

        return j0 + delta_j


# ============================================================================
# DATASET PYTORCH
# ============================================================================


class PairedPathSubset(Dataset):
    """
    Sous-ensemble de ForcedPairDataset.

    Les j_seq sont conservés pour validation et audit futur,
    même si la fonction de perte actuelle de gsa_v1_core
    supervise principalement la sortie terminale.
    """

    def __init__(
        self,
        dataset: ForcedPairDataset,
        indices: np.ndarray,
    ) -> None:

        self.x_a = torch.as_tensor(
            dataset.x_a[indices],
            dtype=torch.float32,
        )

        self.x_b = torch.as_tensor(
            dataset.x_b[indices],
            dtype=torch.float32,
        )

        self.act_a = torch.as_tensor(
            dataset.act_a[indices],
            dtype=torch.long,
        )

        self.act_b = torch.as_tensor(
            dataset.act_b[indices],
            dtype=torch.long,
        )

        self.j0 = torch.as_tensor(
            dataset.j0[indices],
            dtype=torch.float32,
        )

        self.j_seq_a = torch.as_tensor(
            dataset.j_seq_a[indices],
            dtype=torch.float32,
        )

        self.j_seq_b = torch.as_tensor(
            dataset.j_seq_b[indices],
            dtype=torch.float32,
        )

        self.j_final_a = torch.as_tensor(
            dataset.j_final_a[indices],
            dtype=torch.float32,
        )

        self.j_final_b = torch.as_tensor(
            dataset.j_final_b[indices],
            dtype=torch.float32,
        )

        self.omega = torch.as_tensor(
            dataset.omega[indices],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return int(self.j0.shape[0])

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:

        return {
            "x_a": self.x_a[idx],
            "x_b": self.x_b[idx],

            "act_a": self.act_a[idx],
            "act_b": self.act_b[idx],

            "j0": self.j0[idx],

            "j_seq_a": self.j_seq_a[idx],
            "j_seq_b": self.j_seq_b[idx],

            "j_final_a": self.j_final_a[idx],
            "j_final_b": self.j_final_b[idx],

            "omega": self.omega[idx],
        }


# ============================================================================
# SAVE / LOAD DATASET
# ============================================================================


def save_forced_pair_dataset(
    path: Path,
    dataset: ForcedPairDataset,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,

        x_a=dataset.x_a,
        x_b=dataset.x_b,

        act_a=dataset.act_a,
        act_b=dataset.act_b,

        j0=dataset.j0,

        j_seq_a=dataset.j_seq_a,
        j_seq_b=dataset.j_seq_b,

        j_final_a=dataset.j_final_a,
        j_final_b=dataset.j_final_b,

        omega=dataset.omega,

        initial_state_id=dataset.initial_state_id,
        site_pair=dataset.site_pair,

        mode=np.array(
            dataset.mode
        ),
    )


def load_forced_pair_dataset(
    path: Path,
) -> ForcedPairDataset:

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        required = [
            "x_a",
            "x_b",
            "act_a",
            "act_b",
            "j0",
            "j_seq_a",
            "j_seq_b",
            "j_final_a",
            "j_final_b",
            "omega",
            "initial_state_id",
            "site_pair",
            "mode",
        ]

        missing = [
            name
            for name in required
            if name not in data
        ]

        if missing:
            raise ValueError(
                "NPZ incomplet : champs manquants : "
                + ", ".join(missing)
            )

        return ForcedPairDataset(
            x_a=data["x_a"],
            x_b=data["x_b"],
            act_a=data["act_a"],
            act_b=data["act_b"],
            j0=data["j0"],
            j_seq_a=data["j_seq_a"],
            j_seq_b=data["j_seq_b"],
            j_final_a=data["j_final_a"],
            j_final_b=data["j_final_b"],
            omega=data["omega"],
            initial_state_id=data["initial_state_id"],
            site_pair=data["site_pair"],
            mode=str(data["mode"].item()),
        )


# ============================================================================
# DATASET SYNTHÉTIQUE SMOKE
# ============================================================================


def build_synthetic_smoke_dataset(
    n_sites: int,
    n_initial_states: int,
    eta: float,
    mode: TransportMode,
    seed: int,
) -> ForcedPairDataset:

    rng = np.random.default_rng(seed)

    rows: dict[str, list] = {
        "x_a": [],
        "x_b": [],

        "act_a": [],
        "act_b": [],

        "j0": [],

        "jseq_a": [],
        "jseq_b": [],

        "ja": [],
        "jb": [],

        "omega": [],

        "state_id": [],
        "pair": [],
    }

    d_j = (
        n_sites
        * (n_sites - 1)
        // 2
    )

    triu_r, triu_c = np.triu_indices(
        n_sites,
        k=1,
    )

    for state_id in range(
        n_initial_states
    ):

        x0 = rng.choice(
            [-1.0, 1.0],
            size=(n_sites,),
        )

        raw = rng.normal(
            0.0,
            0.5,
            size=(n_sites, n_sites),
        )

        j0_mat = (
            0.5
            * (raw + raw.T)
        )

        np.fill_diagonal(
            j0_mat,
            0.0,
        )

        j0_vec = (
            j0_mat[
                triu_r,
                triu_c
            ]
            .astype(np.float32)
        )

        for k in range(n_sites):

            for m in range(
                k + 1,
                n_sites,
            ):

                # ------------------------------------------------------
                # Chemin A : k -> m
                # ------------------------------------------------------

                xa = np.stack(
                    [
                        x0,
                        x0.copy(),
                        x0.copy(),
                    ]
                )

                xa[1, k] *= -1.0
                xa[2, k] *= -1.0
                xa[2, m] *= -1.0

                # ------------------------------------------------------
                # Chemin B : m -> k
                # ------------------------------------------------------

                xb = np.stack(
                    [
                        x0,
                        x0.copy(),
                        x0.copy(),
                    ]
                )

                xb[1, m] *= -1.0
                xb[2, m] *= -1.0
                xb[2, k] *= -1.0

                ja = j0_vec.copy()
                jb = j0_vec.copy()

                if mode == "symmetric":

                    # Terme explicitement dépendant de l'ordre.
                    comm_scale = (
                        eta
                        * float(
                            x0[k] * x0[m]
                        )
                    )

                    edge_idx = int(
                        np.flatnonzero(
                            (triu_r == k)
                            & (triu_c == m)
                        )[0]
                    )

                    ja[edge_idx] += comm_scale
                    jb[edge_idx] -= comm_scale

                rows["x_a"].append(xa)
                rows["x_b"].append(xb)

                rows["act_a"].append(
                    (k, m)
                )

                rows["act_b"].append(
                    (m, k)
                )

                rows["j0"].append(
                    j0_vec
                )

                rows["jseq_a"].append(
                    np.stack(
                        [
                            j0_vec,
                            j0_vec,
                            ja,
                        ]
                    )
                )

                rows["jseq_b"].append(
                    np.stack(
                        [
                            j0_vec,
                            j0_vec,
                            jb,
                        ]
                    )
                )

                rows["ja"].append(
                    ja
                )

                rows["jb"].append(
                    jb
                )

                rows["omega"].append(
                    ja - jb
                )

                rows["state_id"].append(
                    state_id
                )

                rows["pair"].append(
                    (k, m)
                )

    return ForcedPairDataset(
        x_a=np.asarray(
            rows["x_a"],
            dtype=np.float32,
        ),

        x_b=np.asarray(
            rows["x_b"],
            dtype=np.float32,
        ),

        act_a=np.asarray(
            rows["act_a"],
            dtype=np.int64,
        ),

        act_b=np.asarray(
            rows["act_b"],
            dtype=np.int64,
        ),

        j0=np.asarray(
            rows["j0"],
            dtype=np.float32,
        ),

        j_seq_a=np.asarray(
            rows["jseq_a"],
            dtype=np.float32,
        ),

        j_seq_b=np.asarray(
            rows["jseq_b"],
            dtype=np.float32,
        ),

        j_final_a=np.asarray(
            rows["ja"],
            dtype=np.float32,
        ),

        j_final_b=np.asarray(
            rows["jb"],
            dtype=np.float32,
        ),

        omega=np.asarray(
            rows["omega"],
            dtype=np.float32,
        ),

        initial_state_id=np.asarray(
            rows["state_id"],
            dtype=np.int64,
        ),

        site_pair=np.asarray(
            rows["pair"],
            dtype=np.int64,
        ),

        mode=mode,
    )


# ============================================================================
# PRÉPARATION DATASET
# ============================================================================


def prepare_dataset(
    config: TrainV2Config,
) -> ForcedPairDataset:

    if config.dataset_npz is not None:

        path = Path(
            config.dataset_npz
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {path}"
            )

        dataset = load_forced_pair_dataset(
            path
        )

        validate_forced_pair_dataset(
            dataset,
            expected_n_sites=config.n_sites,
            expected_mode=config.transport_mode,
        )

        return dataset

    if config.synthetic_smoke:

        dataset = build_synthetic_smoke_dataset(
            n_sites=config.n_sites,
            n_initial_states=config.n_initial_states,
            eta=config.eta,
            mode=config.transport_mode,
            seed=config.seed,
        )

        validate_forced_pair_dataset(
            dataset,
            expected_n_sites=config.n_sites,
            expected_mode=config.transport_mode,
        )

        return dataset

    from balayage_commutateur_nk_v2 import (
        sample_binary_symmetric_states,
    )

    rng = np.random.default_rng(
        config.seed
    )

    initial_states = (
        sample_binary_symmetric_states(
            n_sites=config.n_sites,
            n_states=config.n_initial_states,
            rng=rng,
            scale=config.scale_j,
        )
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
    )

    return dataset


# ============================================================================
# BASELINE
# ============================================================================


def compute_baseline_metrics(
    j0: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
) -> dict[str, float]:

    """
    Baseline prédictive :

        J_hat_a = J0
        J_hat_b = J0
        Omega_hat = 0
    """

    baseline_a = j0
    baseline_b = j0

    baseline_omega = torch.zeros_like(
        baseline_a
    )

    true_omega = (
        target_a - target_b
    )

    mse_a = torch.mean(
        (baseline_a - target_a) ** 2
    ).item()

    mse_b = torch.mean(
        (baseline_b - target_b) ** 2
    ).item()

    mse_terminal = 0.5 * (
        mse_a + mse_b
    )

    mse_omega = torch.mean(
        (baseline_omega - true_omega) ** 2
    ).item()

    return {
        "baseline_mse_a": float(mse_a),
        "baseline_mse_b": float(mse_b),
        "baseline_mse_terminal": float(
            mse_terminal
        ),
        "baseline_mse_omega": float(
            mse_omega
        ),
    }


# ============================================================================
# EPOCH
# ============================================================================


def run_epoch(
    model: HybridPathModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    config: TrainV2Config,
    device: torch.device,
) -> tuple[
    dict[str, float],
    dict[str, float | int],
]:

    training = optimizer is not None

    model.train(training)

    totals = {
        "loss_total": 0.0,
        "loss_terminal": 0.0,
        "loss_omega": 0.0,
        "loss_geometry_regularizer": 0.0,
    }

    baseline_totals = {
        "baseline_mse_a": 0.0,
        "baseline_mse_b": 0.0,
        "baseline_mse_terminal": 0.0,
        "baseline_mse_omega": 0.0,
    }

    n_samples = 0

    omega_hats: list[np.ndarray] = []
    omega_trues: list[np.ndarray] = []

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for batch in loader:

            x_a = batch["x_a"].to(
                device
            )

            x_b = batch["x_b"].to(
                device
            )

            act_a = batch["act_a"].to(
                device
            )

            act_b = batch["act_b"].to(
                device
            )

            j0 = batch["j0"].to(
                device
            )

            target_a = batch[
                "j_final_a"
            ].to(device)

            target_b = batch[
                "j_final_b"
            ].to(device)

            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

            # ----------------------------------------------------------
            # Modèle
            # ----------------------------------------------------------

            pred_a, pred_b, omega_hat = (
                predict_path_pair(
                    model,
                    x_a,
                    act_a,
                    x_b,
                    act_b,
                    j0,
                )
            )

            loss, info = paired_gsa_loss(
                pred_a=pred_a,
                pred_b=pred_b,
                target_a=target_a,
                target_b=target_b,
                lambda_omega=config.lambda_omega,
                lambda_geometry=config.lambda_geometry,
                model=model,
            )

            if training:

                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip,
                )

                optimizer.step()

            # ----------------------------------------------------------
            # Pertes
            # ----------------------------------------------------------

            bs = int(
                j0.shape[0]
            )

            n_samples += bs

            totals["loss_total"] += (
                float(info["loss_total"])
                * bs
            )

            totals["loss_terminal"] += (
                float(info["loss_terminal"])
                * bs
            )

            if "loss_closure" in info:
                omega_loss = float(
                    info["loss_closure"]
                )

            elif "loss_omega" in info:
                omega_loss = float(
                    info["loss_omega"]
                )

            else:
                true_omega = (
                    target_a
                    - target_b
                )

                omega_loss = torch.mean(
                    (
                        omega_hat
                        - true_omega
                    ) ** 2
                ).item()

            totals["loss_omega"] += (
                omega_loss
                * bs
            )

            totals[
                "loss_geometry_regularizer"
            ] += (
                float(
                    info[
                        "loss_geometry_regularizer"
                    ]
                )
                * bs
            )

            # ----------------------------------------------------------
            # Baseline
            # ----------------------------------------------------------

            batch_baseline = (
                compute_baseline_metrics(
                    j0=j0,
                    target_a=target_a,
                    target_b=target_b,
                )
            )

            for key in baseline_totals:
                baseline_totals[key] += (
                    batch_baseline[key]
                    * bs
                )

            # ----------------------------------------------------------
            # Omega
            # ----------------------------------------------------------

            omega_hats.append(
                omega_hat
                .detach()
                .cpu()
                .numpy()
            )

            omega_trues.append(
                (
                    target_a
                    - target_b
                )
                .detach()
                .cpu()
                .numpy()
            )

    # --------------------------------------------------------------
    # Moyennes
    # --------------------------------------------------------------

    denominator = max(
        n_samples,
        1,
    )

    avg_losses = {
        key: value / denominator
        for key, value in totals.items()
    }

    avg_baseline = {
        key: value / denominator
        for key, value in baseline_totals.items()
    }

    # --------------------------------------------------------------
    # Métriques Omega
    # --------------------------------------------------------------

    omega_pred = np.concatenate(
        omega_hats,
        axis=0,
    )

    omega_true = np.concatenate(
        omega_trues,
        axis=0,
    )

    omega_metrics = closure_metrics(
        omega_pred,
        omega_true,
    )

    metrics = {
        **omega_metrics,
        **avg_baseline,
    }

    metrics[
        "delta_mse_omega_vs_baseline"
    ] = (
        float(
            omega_metrics["mse"]
        )
        - avg_baseline[
            "baseline_mse_omega"
        ]
    )

    return (
        avg_losses,
        metrics,
    )


# ============================================================================
# SELF TEST
# ============================================================================


def build_self_test_batch(
    n_sites: int,
    batch_size: int = 3,
    length: int = 4,
    seed: int = 123,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    if length < 2:
        raise ValueError(
            "length doit être >= 2."
        )

    rng = np.random.default_rng(
        seed
    )

    x = np.empty(
        (
            batch_size,
            length,
            n_sites,
        ),
        dtype=np.float32,
    )

    act = np.empty(
        (
            batch_size,
            length - 1,
        ),
        dtype=np.int64,
    )

    d_j = (
        n_sites
        * (n_sites - 1)
        // 2
    )

    j0 = rng.normal(
        0.0,
        0.5,
        size=(
            batch_size,
            d_j,
        ),
    ).astype(np.float32)

    for b in range(
        batch_size
    ):

        x[b, 0] = rng.choice(
            [-1.0, 1.0],
            size=n_sites,
        )

        for t in range(
            length - 1
        ):

            a = int(
                rng.integers(
                    0,
                    n_sites,
                )
            )

            act[b, t] = a

            x[b, t + 1] = (
                x[b, t]
                .copy()
            )

            x[b, t + 1, a] *= -1.0

    return (
        torch.from_numpy(x),
        torch.from_numpy(act),
        torch.from_numpy(j0),
    )


def self_test(
    config: TrainV2Config,
) -> None:

    print()
    print("=" * 72)
    print("SELF-TEST NK-LSTM-GSA V2 (AVEC CAUSALITÉ STRICTE)")
    print("=" * 72)

    device = torch.device(
        config.device
    )

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

    x, act, j0 = (
        build_self_test_batch(
            n_sites=config.n_sites,
            batch_size=3,
            length=4,
            seed=config.seed,
        )
    )

    x = x.to(device)
    act = act.to(device)
    j0 = j0.to(device)

    pad_mask = torch.zeros(
        x.shape[:2],
        dtype=torch.bool,
        device=device,
    )

    # --------------------------------------------------------------
    # Test des quatre architectures
    # --------------------------------------------------------------

    for arch_mode in (
        "lstm_only",
        "gsa_only",
        "hybrid_concat",
        "hybrid_gated",
    ):

        print(
            f"\n[TEST] arch_mode={arch_mode}"
        )

        model = HybridPathModel(
            gsa_config=gsa_cfg,
            arch_mode=arch_mode,
            lstm_layers=config.lstm_layers,
        ).to(device)

        model.train()

        out = model(
            x_seq=x,
            act_idx=act,
            j0=j0,
            pad_mask=pad_mask,
        )

        expected_shape = (
            x.shape[0],
            j0.shape[-1],
        )

        assert (
            out.shape
            == expected_shape
        ), (
            f"Shape sortie incorrecte : "
            f"{out.shape} != "
            f"{expected_shape}"
        )

        loss = out.square().mean()
        loss.backward()

        # ----------------------------------------------------------
        # Gradient LSTM
        # ----------------------------------------------------------

        if model.use_lstm:

            assert (
                model.lstm is not None
            )

            lstm_has_grad = any(
                parameter.grad is not None
                for parameter in (
                    model.lstm.parameters()
                )
            )

            assert lstm_has_grad, (
                "Aucun gradient détecté "
                "dans la branche LSTM."
            )

        # ----------------------------------------------------------
        # Gradient GSA
        # ----------------------------------------------------------

        if (
            model.use_gsa
            and model.gsa.edge_costs
            is not None
        ):

            edge_grad = (
                model
                .gsa
                .edge_costs
                .edge_logits
                .grad
            )

            assert (
                edge_grad is not None
            ), (
                "Gradient absent des "
                "coûts géométriques "
                "alors que GSA est active."
            )

        else:

            if (
                model.gsa.edge_costs
                is not None
            ):
                assert (
                    model
                    .gsa
                    .edge_costs
                    .edge_logits
                    .grad
                    is None
                ), (
                    "Un gradient GSA inattendu "
                    "est apparu en lstm_only."
                )

        print(
            "  forward        : OK"
        )

        print(
            "  gradient       : OK"
        )

    # --------------------------------------------------------------
    # Test antisymétrie exacte
    # --------------------------------------------------------------

    print(
        "\n[TEST] antisymétrie Omega"
    )

    model = HybridPathModel(
        gsa_config=gsa_cfg,
        arch_mode="hybrid_concat",
        lstm_layers=config.lstm_layers,
    ).to(device)

    x_a = x
    x_b = x.clone()

    act_a = act
    act_b = act.clone()

    if act.shape[1] >= 2:

        act_b[:, 0] = act[:, 1]
        act_b[:, 1] = act[:, 0]

        x_b = x.clone()

        for b in range(
            x.shape[0]
        ):

            x_b[b, 0] = x[b, 0]

            x_b[b, 1] = (
                x_b[b, 0]
                .clone()
            )

            x_b[
                b,
                1,
                act_b[b, 0],
            ] *= -1.0

            x_b[b, 2] = (
                x_b[b, 1]
                .clone()
            )

            x_b[
                b,
                2,
                act_b[b, 1],
            ] *= -1.0

            if x.shape[1] > 3:
                for t in range(
                    2,
                    x.shape[1] - 1,
                ):
                    x_b[
                        b,
                        t + 1,
                    ] = (
                        x_b[
                            b,
                            t,
                        ].clone()
                    )

                    x_b[
                        b,
                        t + 1,
                        act_b[
                            b,
                            t,
                        ],
                    ] *= -1.0

    _, _, omega_ab = (
        predict_path_pair(
            model,
            x_a,
            act_a,
            x_b,
            act_b,
            j0,
        )
    )

    _, _, omega_ba = (
        predict_path_pair(
            model,
            x_b,
            act_b,
            x_a,
            act_a,
            j0,
        )
    )

    torch.testing.assert_close(
        omega_ab,
        -omega_ba,
        atol=1e-6,
        rtol=1e-6,
    )

    print(
        "  antisymétrie    : OK"
    )

    # --------------------------------------------------------------
    # Test causalité d'embedding stricte
    # --------------------------------------------------------------

    print(
        "\n[TEST] causalité stricte des actions"
    )

    model = HybridPathModel(
        gsa_config=gsa_cfg,
        arch_mode="lstm_only",
        lstm_layers=config.lstm_layers,
    ).to(device)

    embedded_1, _ = (
        model._embed_inputs(
            x_seq=x,
            act_idx=act,
            j0=j0,
            pos=None,
            pad_mask=pad_mask,
        )
    )

    act_future = act.clone()

    if act_future.shape[1] >= 2:
        act_future[:, 1:] = (
            (
                act_future[:, 1:]
                + 1
            )
            % config.n_sites
        )

    embedded_2, _ = (
        model._embed_inputs(
            x_seq=x,
            act_idx=act_future,
            j0=j0,
            pos=None,
            pad_mask=pad_mask,
        )
    )

    # Le token 0 ne reçoit aucune action (décalage causal).
    torch.testing.assert_close(
        embedded_1[:, 0],
        embedded_2[:, 0],
        atol=1e-6,
        rtol=0.0,
    )

    # Le token 1 dépend uniquement de act[:, 0],
    # donc perturber act[:, 1:] ne doit pas modifier embedded[:, 1].
    torch.testing.assert_close(
        embedded_1[:, 1],
        embedded_2[:, 1],
        atol=1e-6,
        rtol=0.0,
    )

    print(
        "  filtration causale : OK (aucun impact d'action future)"
    )

    print()
    print(
        "=" * 72
    )
    print(
        "SELF-TEST : TOUS LES TESTS SONT VALIDES"
    )
    print(
        "=" * 72
    )


# ============================================================================
# ARGUMENTS CLI
# ============================================================================


def parse_args() -> TrainV2Config:

    parser = argparse.ArgumentParser(
        description=(
            "Entraînement NK-LSTM-GSA V2 "
            "sur ForcedPairDataset."
        )
    )

    parser.add_argument(
        "--arch-mode",
        choices=[
            "lstm_only",
            "gsa_only",
            "hybrid_concat",
            "hybrid_gated",
        ],
        default="hybrid_concat",
    )

    parser.add_argument(
        "--geometry-mode",
        choices=[
            "none",
            "fixed",
            "learned",
        ],
        default="learned",
    )

    parser.add_argument(
        "--transport-mode",
        choices=[
            "symmetric",
            "integrable",
        ],
        default="symmetric",
    )

    parser.add_argument(
        "--n-sites",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--n-initial-states",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--scale-j",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--dataset-npz",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gsa-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lstm-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--ff-mult",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--beta-action",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-len",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--edge-eps",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda-omega",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda-geometry",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--early-stopping-patience",
        "--epochs-patience",
        dest="early_stopping_patience",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/nk_lstm_gsa_v2",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

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


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    config = parse_args()

    set_seed(
        config.seed
    )

    # --------------------------------------------------------------
    # Self-test direct
    # --------------------------------------------------------------

    if config.self_test:
        self_test(config)
        return

    device = torch.device(
        config.device
    )

    print()
    print(
        "=" * 72
    )
    print(
        "NK-LSTM-GSA V2"
    )
    print(
        "=" * 72
    )

    print(
        f"Architecture : {config.arch_mode}"
    )

    print(
        f"Géométrie   : {config.geometry_mode}"
    )

    print(
        f"Transport    : {config.transport_mode}"
    )

    print(
        f"Device       : {device}"
    )

    print(
        f"Seed         : {config.seed}"
    )

    # --------------------------------------------------------------
    # Avertissement sémantique
    # --------------------------------------------------------------

    if (
        config.arch_mode == "lstm_only"
        and config.geometry_mode == "learned"
    ):
        warnings.warn(
            "geometry_mode='learned' est sans effet "
            "sur le forward en arch_mode='lstm_only' : "
            "la voie GSA n'est pas utilisée."
        )

    # --------------------------------------------------------------
    # Sorties
    # --------------------------------------------------------------

    output_dir = Path(
        config.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            asdict(config),
            f,
            indent=2,
        )

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    dataset = prepare_dataset(
        config
    )

    actual_n_sites = (
        int(dataset.x_a.shape[-1])
    )

    if (
        actual_n_sites
        != config.n_sites
    ):
        raise ValueError(
            "Incohérence n_sites : "
            f"dataset={actual_n_sites}, "
            f"config={config.n_sites}."
        )

    # --------------------------------------------------------------
    # Split groupé
    # --------------------------------------------------------------

    splits = (
        split_indices_by_initial_state(
            dataset=dataset,
            seed=config.seed,
            train_fraction=(
                config.train_fraction
            ),
            validation_fraction=(
                config.validation_fraction
            ),
        )
    )

    if (
        len(splits["train"]) == 0
        or len(splits["validation"]) == 0
        or len(splits["test"]) == 0
    ):
        raise ValueError(
            "Un des splits est vide."
        )

    print()
    print(
        "Split groupé :"
    )

    print(
        f"  train      = {len(splits['train'])}"
    )

    print(
        f"  validation = {len(splits['validation'])}"
    )

    print(
        f"  test       = {len(splits['test'])}"
    )

    # --------------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------------

    gen = (
        torch.Generator()
        .manual_seed(
            config.seed
        )
    )

    train_loader = DataLoader(
        PairedPathSubset(
            dataset,
            splits["train"],
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=gen,
    )

    val_loader = DataLoader(
        PairedPathSubset(
            dataset,
            splits["validation"],
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        PairedPathSubset(
            dataset,
            splits["test"],
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    # --------------------------------------------------------------
    # Configuration GSA
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Modèle
    # --------------------------------------------------------------

    model = HybridPathModel(
        gsa_config=gsa_cfg,
        arch_mode=config.arch_mode,
        lstm_layers=config.lstm_layers,
    ).to(device)

    print()
    print(
        f"Paramètres entraînables : "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # --------------------------------------------------------------
    # Optimiseur
    #
    # zero_grad(set_to_none=True) assure qu'un paramètre dont le
    # gradient est None n'est pas modifié par AdamW (weight decay inclus).
    # --------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # --------------------------------------------------------------
    # Entraînement
    # --------------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    patience = 0

    history: list[dict] = []

    for epoch in range(
        1,
        config.epochs + 1,
    ):

        train_losses, train_metrics = (
            run_epoch(
                model,
                train_loader,
                optimizer,
                config,
                device,
            )
        )

        val_losses, val_metrics = (
            run_epoch(
                model,
                val_loader,
                None,
                config,
                device,
            )
        )

        history.append(
            {
                "epoch": epoch,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_losses['loss_total']:.6f} | "
            f"val={val_losses['loss_total']:.6f} | "
            f"Omega MSE={val_metrics['mse']:.6f} | "
            f"Baseline Omega MSE="
            f"{val_metrics['baseline_mse_omega']:.6f} | "
            f"ΔOmega="
            f"{val_metrics['delta_mse_omega_vs_baseline']:.6f}"
        )

        # ----------------------------------------------------------
        # Early stopping
        # ----------------------------------------------------------

        if (
            val_losses["loss_total"]
            < best_val_loss
            - config.min_delta
        ):

            best_val_loss = (
                val_losses[
                    "loss_total"
                ]
            )

            patience = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "best_val_loss": (
                        best_val_loss
                    ),
                    "config": (
                        asdict(config)
                    ),
                },
                output_dir
                / "best.pt",
            )

        else:

            patience += 1

            if (
                patience
                >= config.early_stopping_patience
            ):

                print(
                    f"Arrêt anticipé à "
                    f"l'époque {epoch} "
                    f"(meilleure val="
                    f"{best_val_loss:.6f})."
                )

                break

    # --------------------------------------------------------------
    # Historique
    # --------------------------------------------------------------

    with open(
        output_dir
        / "history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    # --------------------------------------------------------------
    # Recharge meilleur modèle
    # --------------------------------------------------------------

    best_path = (
        output_dir
        / "best.pt"
    )

    if not best_path.exists():
        raise RuntimeError(
            "Aucun checkpoint best.pt n'a été produit."
        )

    ckpt = torch.load(
        best_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------------
    # Test final
    # --------------------------------------------------------------

    test_losses, test_metrics = (
        run_epoch(
            model,
            test_loader,
            None,
            config,
            device,
        )
    )

    # --------------------------------------------------------------
    # Résumé
    # --------------------------------------------------------------

    summary = {
        "arch_mode": (
            config.arch_mode
        ),

        "geometry_mode": (
            config.geometry_mode
        ),

        "transport_mode": (
            config.transport_mode
        ),

        "best_val_loss": (
            best_val_loss
        ),

        "test_losses": (
            test_losses
        ),

        "test_metrics": (
            test_metrics
        ),

        "baseline": {
            "mse_terminal": (
                test_metrics[
                    "baseline_mse_terminal"
                ]
            ),
            "mse_omega": (
                test_metrics[
                    "baseline_mse_omega"
                ]
            ),
        },

        "model_vs_baseline": {
            "delta_mse_omega": (
                test_metrics[
                    "delta_mse_omega_vs_baseline"
                ]
            ),
        },
    }

    with open(
        output_dir
        / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        "=" * 72
    )

    print(
        "RÉSULTATS TEST"
    )

    print(
        "=" * 72
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
