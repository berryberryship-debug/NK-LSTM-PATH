#!/usr/bin/env python3
"""
train_nk_lstm_gsa_v2.py

Entraînement et ablation de l'architecture hybride NK-LSTM-GSA V2,
directement adossée au noyau géométrique `gsa_v1_core.py`.

Invariants préservés :
- Entrées exactes sur l'hypercube {-1, +1}^N : (x_seq, act_idx, j0, pad_mask).
- Action géométrique observée S_obs[b, i, j] = sum_{t=j}^{i-1} c_theta(x[t], act[t]).
- Antisymétrie exacte Omega_hat(a, b) = -Omega_hat(b, a) via `predict_path_pair`.
- Découpage groupé par `initial_state_id` via `split_indices_by_initial_state`.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainV2Config:
    # Architecture et ablations
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

    # Données et physique NK
    transport_mode: TransportMode = "symmetric"
    n_initial_states: int = 64
    eta: float = 0.25
    scale_j: float = 0.5
    dataset_npz: str | None = None
    synthetic_smoke: bool = False
    train_fraction: float = 0.60
    validation_fraction: float = 0.20

    # Optimisation et pertes
    batch_size: int = 32
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    lambda_omega: float = 1.0
    lambda_geometry: float = 1e-3
    early_stopping_patience: int = 15
    min_delta: float = 1e-6

    # Reproductibilité et sorties
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "runs/nk_lstm_gsa_v2"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Modèle hybride : voie temporelle (LSTM) + voie géométrique (GSAPathModel)
# ---------------------------------------------------------------------------

class HybridPathModel(nn.Module):
    """Encodeur de chemin unifié compatible avec `predict_path_pair` et `paired_gsa_loss`."""

    def __init__(self, gsa_config: GSAConfig, arch_mode: ArchMode = "hybrid_concat", lstm_layers: int = 2) -> None:
        super().__init__()
        if arch_mode not in ("lstm_only", "gsa_only", "hybrid_concat", "hybrid_gated"):
            raise ValueError(f"arch_mode invalide : {arch_mode}")
        if lstm_layers < 1:
            raise ValueError("lstm_layers doit être >= 1.")

        self.config = gsa_config
        self.arch_mode: ArchMode = arch_mode
        self.lstm_layers = int(lstm_layers)

        # Réutilisation directe des projections, embeddings et blocs GSA de gsa_v1_core
        self.gsa = GSAPathModel(gsa_config)
        d_model = gsa_config.d_model

        self.use_lstm = arch_mode in ("lstm_only", "hybrid_concat", "hybrid_gated")
        self.use_gsa = arch_mode in ("gsa_only", "hybrid_concat", "hybrid_gated")

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
        if x_seq.ndim != 3:
            raise ValueError("x_seq doit avoir la forme (B, L, N).")
        batch, length, n_sites = x_seq.shape
        if n_sites != self.config.n_sites:
            raise ValueError(f"N attendu={self.config.n_sites}, reçu={n_sites}.")
        if length < 2 or length > self.config.max_len:
            raise ValueError("La longueur L doit appartenir à [2, max_len].")
        if act_idx.shape != (batch, length - 1):
            raise ValueError("act_idx doit avoir la forme (B, L-1).")
        if j0.shape != (batch, self.d_J):
            raise ValueError(f"j0 doit avoir la forme (B, {self.d_J}).")
        if not torch.is_floating_point(x_seq) or not torch.is_floating_point(j0):
            raise TypeError("x_seq et j0 doivent être des tenseurs flottants.")

        if pos is None:
            pos = torch.arange(length, device=x_seq.device).expand(batch, length)
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

        action_tokens = torch.full(
            (batch, length), self.config.n_sites, device=x_seq.device, dtype=torch.long
        )
        valid_actions = (~pad_mask[:, :-1]) & (~pad_mask[:, 1:])
        safe_actions = act_idx.clamp(min=0, max=self.config.n_sites - 1)
        action_tokens[:, :-1] = torch.where(
            valid_actions, safe_actions, torch.full_like(safe_actions, self.config.n_sites)
        )
        action_tokens = action_tokens.masked_fill(pad_mask, self.config.n_sites)

        hidden = (
            self.gsa.x_proj(x_seq)
            + self.gsa.action_embedding(action_tokens)
            + self.gsa.position_embedding(pos)
            + self.gsa.j0_proj(j0).unsqueeze(1)
        )
        hidden = self.gsa.input_norm(hidden)
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
        embedded, pad_mask = self._embed_inputs(x_seq, act_idx, j0, pos, pad_mask)
        batch, length, _ = embedded.shape
        lengths = (~pad_mask).sum(dim=1)
        last_valid = lengths - 1
        batch_idx = torch.arange(batch, device=x_seq.device)

        h_temp = None
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
            h_temp = seq_temp[batch_idx, last_valid]

        h_geom = None
        if self.use_gsa:
            action_bias = self.gsa.path_action_bias(x_seq, act_idx, pad_mask)
            seq_geom = embedded
            for block in self.gsa.blocks:
                seq_geom = block(seq_geom, action_bias, pad_mask, self.config.beta_action)
            seq_geom = self.gsa.final_norm(seq_geom)
            h_geom = seq_geom[batch_idx, last_valid]

        if self.arch_mode == "lstm_only":
            assert h_temp is not None
            fused = h_temp
        elif self.arch_mode == "gsa_only":
            assert h_geom is not None
            fused = h_geom
        elif self.arch_mode == "hybrid_concat":
            assert h_temp is not None and h_geom is not None and self.fusion is not None
            fused = self.fusion(torch.cat([h_temp, h_geom], dim=-1))
        else:
            assert (
                h_temp is not None
                and h_geom is not None
                and self.fusion_gate is not None
                and self.fusion is not None
            )
            gate = torch.sigmoid(self.fusion_gate(torch.cat([h_temp, h_geom], dim=-1)))
            fused = self.fusion(gate * h_temp + (1.0 - gate) * h_geom)

        delta_j = self.gsa.delta_head(fused)
        return j0 + delta_j


# ---------------------------------------------------------------------------
# Dataset PyTorch sur ForcedPairDataset
# ---------------------------------------------------------------------------

class PairedPathSubset(Dataset):
    """Sous-ensemble indexé de ForcedPairDataset sans fuite entre paires sœurs."""

    def __init__(self, dataset: ForcedPairDataset, indices: np.ndarray) -> None:
        self.x_a = torch.as_tensor(dataset.x_a[indices], dtype=torch.float32)
        self.x_b = torch.as_tensor(dataset.x_b[indices], dtype=torch.float32)
        self.act_a = torch.as_tensor(dataset.act_a[indices], dtype=torch.long)
        self.act_b = torch.as_tensor(dataset.act_b[indices], dtype=torch.long)
        self.j0 = torch.as_tensor(dataset.j0[indices], dtype=torch.float32)
        self.j_final_a = torch.as_tensor(dataset.j_final_a[indices], dtype=torch.float32)
        self.j_final_b = torch.as_tensor(dataset.j_final_b[indices], dtype=torch.float32)
        self.omega = torch.as_tensor(dataset.omega[indices], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.j0.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_a": self.x_a[idx],
            "x_b": self.x_b[idx],
            "act_a": self.act_a[idx],
            "act_b": self.act_b[idx],
            "j0": self.j0[idx],
            "j_final_a": self.j_final_a[idx],
            "j_final_b": self.j_final_b[idx],
            "omega": self.omega[idx],
        }


def save_forced_pair_dataset(path: Path, dataset: ForcedPairDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        mode=np.array(dataset.mode),
    )


def load_forced_pair_dataset(path: Path) -> ForcedPairDataset:
    with np.load(path, allow_pickle=False) as data:
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


def build_synthetic_smoke_dataset(
    n_sites: int, n_initial_states: int, eta: float, mode: TransportMode, seed: int
) -> ForcedPairDataset:
    """Jeu synthétique déterministe pour tester le pipeline hors dépôt."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {
        "x_a": [], "x_b": [], "act_a": [], "act_b": [], "j0": [],
        "jseq_a": [], "jseq_b": [], "ja": [], "jb": [], "omega": [],
        "state_id": [], "pair": [],
    }
    d_j = n_sites * (n_sites - 1) // 2
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
                    # Terme dépendant de l'ordre (non commutatif)
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
        return load_forced_pair_dataset(path)

    if config.synthetic_smoke:
        return build_synthetic_smoke_dataset(
            n_sites=config.n_sites,
            n_initial_states=config.n_initial_states,
            eta=config.eta,
            mode=config.transport_mode,
            seed=config.seed,
        )

    from balayage_commutateur_nk_v2 import sample_binary_symmetric_states

    rng = np.random.default_rng(config.seed)
    initial_states = sample_binary_symmetric_states(
        n_sites=config.n_sites,
        n_states=config.n_initial_states,
        rng=rng,
        scale=config.scale_j,
    )
    return build_forced_pair_dataset(
        initial_states=initial_states,
        eta=config.eta,
        mode=config.transport_mode,
    )


# ---------------------------------------------------------------------------
# Boucle d'entraînement et d'évaluation
# ---------------------------------------------------------------------------

def run_epoch(
    model: HybridPathModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    config: TrainV2Config,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float | int]]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "loss_total": 0.0,
        "loss_terminal": 0.0,
        "loss_closure": 0.0,
        "loss_geometry_regularizer": 0.0,
    }
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

            pred_a, pred_b, omega_hat = predict_path_pair(
                model, x_a, act_a, x_b, act_b, j0
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
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            bs = int(j0.shape[0])
            n_samples += bs
            for k in totals:
                totals[k] += info[k] * bs

            omega_hats.append(omega_hat.detach().cpu().numpy())
            omega_trues.append((target_a - target_b).detach().cpu().numpy())

    avg_losses = {k: v / max(n_samples, 1) for k, v in totals.items()}
    metrics = closure_metrics(
        np.concatenate(omega_hats, axis=0),
        np.concatenate(omega_trues, axis=0),
    )
    return avg_losses, metrics


def parse_args() -> TrainV2Config:
    parser = argparse.ArgumentParser(description="Entraînement NK-LSTM-GSA V2 sur ForcedPairDataset.")
    parser.add_argument(
        "--arch-mode",
        choices=["lstm_only", "gsa_only", "hybrid_concat", "hybrid_gated"],
        default="hybrid_concat",
    )
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
    parser.add_argument("--beta-action", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-omega", type=float, default=1.0)
    parser.add_argument("--lambda-geometry", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="runs/nk_lstm_gsa_v2")
    args = parser.parse_args()

    return TrainV2Config(
        arch_mode=args.arch_mode,
        geometry_mode=args.geometry_mode,
        transport_mode=args.transport_mode,
        n_sites=args.n_sites,
        n_initial_states=args.n_initial_states,
        eta=args.eta,
        scale_j=args.scale_j,
        dataset_npz=args.dataset_npz,
        synthetic_smoke=args.synthetic_smoke,
        d_model=args.d_model,
        n_heads=args.n_heads,
        gsa_layers=args.gsa_layers,
        lstm_layers=args.lstm_layers,
        beta_action=args.beta_action,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lambda_omega=args.lambda_omega,
        lambda_geometry=args.lambda_geometry,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = torch.device(config.device)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    dataset = prepare_dataset(config)
    if dataset.n_sites != config.n_sites if hasattr(dataset, "n_sites") else dataset.x_a.shape[-1] != config.n_sites:
        raise ValueError("Incohérence sur n_sites entre la configuration et le dataset.")

    splits = split_indices_by_initial_state(
        dataset=dataset,
        seed=config.seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )

    gen = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        PairedPathSubset(dataset, splits["train"]),
        batch_size=config.batch_size,
        shuffle=True,
        generator=gen,
    )
    val_loader = DataLoader(
        PairedPathSubset(dataset, splits["validation"]),
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        PairedPathSubset(dataset, splits["test"]),
        batch_size=config.batch_size,
        shuffle=False,
    )

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
    model = HybridPathModel(
        gsa_config=gsa_cfg,
        arch_mode=config.arch_mode,
        lstm_layers=config.lstm_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    best_val_loss = float("inf")
    patience = 0
    history = []

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
            f"Epoch {epoch:03d} | "
            f"train={train_losses['loss_total']:.6f} | "
            f"val={val_losses['loss_total']:.6f} | "
            f"omega_mse={val_metrics['mse']:.6f} | "
            f"cos={val_metrics['mean_cosine_nonzero']:.4f}"
        )

        if val_losses["loss_total"] < best_val_loss - config.min_delta:
            best_val_loss = val_losses["loss_total"]
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": asdict(config),
                },
                output_dir / "best.pt",
            )
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print(f"Arrêt anticipé à l'époque {epoch} (meilleure val={best_val_loss:.6f}).")
                break

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_losses, test_metrics = run_epoch(model, test_loader, None, config, device)

    summary = {
        "arch_mode": config.arch_mode,
        "geometry_mode": config.geometry_mode,
        "transport_mode": config.transport_mode,
        "best_val_loss": best_val_loss,
        "test_losses": test_losses,
        "test_metrics": test_metrics,
    }
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Résultats Test :", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
