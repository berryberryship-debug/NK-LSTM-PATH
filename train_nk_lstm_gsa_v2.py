#!/usr/bin/env python3
"""
train_nk_lstm_gsa_v2.py

NK-LSTM-GSA V2 : architecture unifiée temporelle (LSTM) et géométrique (GSA).

Périmètre V2 :
- Entraînement sur les données NK existantes.
- Prédiction terminale.
- Prédiction du chemin / action suivante.
- Dépendance à l'ordre temporel.
- Régularisation de la GSA.

Périmètre V3, implémenté mais désactivé :
- Fermeture / retour au départ.
- Métrique riemannienne.
- Potentiel d'obstacle.
- Supervision de transition.
- Environnement hypercube.

Le script ne fabrique pas de cibles spatiales artificielles.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class V2Config:
    # Données
    data_path: str = "data/nk_sequences.npz"
    obs_key: str = "observations"
    action_key: str = "actions"
    target_key: str = "targets"
    lengths_key: str = "lengths"
    val_fraction: float = 0.15
    test_fraction: float = 0.10

    # Architecture
    obs_dim: int = 64
    action_dim: int = 8
    target_dim: int = 64
    d_model: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.10
    gsa_layers: int = 1
    n_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.10
    fusion: str = "concat"  # "concat" ou "gated"
    causal_attention: bool = True

    # GSA
    beta_init: float = 0.5
    learn_beta: bool = True
    action_bias_dim: int = 32

    # Entraînement
    batch_size: int = 64
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    early_stopping_patience: int = 12
    min_delta: float = 1e-4

    # Pertes V2
    lambda_terminal: float = 1.0
    lambda_path: float = 1.0
    lambda_order: float = 0.25
    lambda_gsa_reg: float = 1e-3
    epsilon: float = 1e-3

    # Pertes V3 : désactivées en V2
    use_closure_loss: bool = False
    use_metric_loss: bool = False
    use_potential_loss: bool = False
    use_transition_supervision: bool = False
    use_navigation_env: bool = False

    lambda_closure: float = 0.0
    lambda_metric: float = 0.0
    lambda_potential: float = 0.0
    lambda_transition: float = 0.0

    # Reproductibilité et sortie
    seed: int = 1337
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "runs/nk_lstm_gsa_v2"
    save_every_epoch: bool = True
    amp: bool = False


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset NK
# ---------------------------------------------------------------------------

class NKSequenceDataset(Dataset):
    """
    Format attendu dans le fichier .npz :

    observations : (N, T, obs_dim), float32
    actions      : (N, T, action_dim), float32 ou int64
    targets      : (N, target_dim), float32
    lengths      : (N,), int64

    Les séquences sont paddées à droite. Les positions >= length[b]
    sont ignorées par les masques et les pertes.
    """

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        lengths: np.ndarray,
    ):
        self.observations = torch.as_tensor(observations, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)

        if actions.dtype in (np.int64, np.int32, int):
            self.actions = torch.as_tensor(actions, dtype=torch.long)
            self.actions_are_indices = True
        else:
            self.actions = torch.as_tensor(actions, dtype=torch.float32)
            self.actions_are_indices = False

        assert self.observations.ndim == 3
        assert self.actions.ndim == 3
        assert self.targets.ndim == 2
        assert self.lengths.ndim == 1

        self.max_length = int(self.lengths.max().item())
        assert self.observations.shape[1] >= self.max_length
        assert self.actions.shape[1] >= self.max_length

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        length = int(self.lengths[index].item())

        obs = self.observations[index, :length]
        target = self.targets[index]
        length_tensor = torch.tensor(length, dtype=torch.long)

        if self.actions_are_indices:
            action = self.actions[index, :length]
        else:
            action = self.actions[index, :length]

        return {
            "observations": obs,
            "actions": action,
            "targets": target,
            "lengths": length_tensor,
            "actions_are_indices": torch.tensor(
                self.actions_are_indices, dtype=torch.bool
            ),
        }


def collate_sequences(batch):
    max_length = max(item["lengths"].item() for item in batch)
    batch_size = len(batch)

    obs_dim = batch[0]["observations"].shape[-1]
    target_dim = batch[0]["targets"].shape[-1]
    action_dim = batch[0]["actions"].shape[-1]

    observations = torch.zeros(batch_size, max_length, obs_dim)
    actions = torch.zeros(batch_size, max_length, action_dim)
    targets = torch.stack([item["targets"] for item in batch])
    lengths = torch.stack([item["lengths"] for item in batch])
    valid_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    actions_are_indices = batch[0]["actions_are_indices"].item()

    for i, item in enumerate(batch):
        length = int(item["lengths"].item())
        observations[i, :length] = item["observations"]
        actions[i, :length] = item["actions"]
        valid_mask[i, :length] = True

    if actions_are_indices:
        actions = actions.long()
        next_action = torch.full(
            (batch_size, max_length), -100, dtype=torch.long
        )

        for i, item in enumerate(batch):
            length = int(item["lengths"].item())
            if length > 1:
                next_action[i, : length - 1] = item["actions"][1:length]

        return {
            "observations": observations,
            "actions": actions,
            "targets": targets,
            "lengths": lengths,
            "valid_mask": valid_mask,
            "next_action": next_action,
            "actions_are_indices": torch.tensor(True),
        }

    next_state = torch.zeros_like(observations)

    for i, item in enumerate(batch):
        length = int(item["lengths"].item())
        if length > 1:
            next_state[i, : length - 1] = item["observations"][1:length]

    return {
        "observations": observations,
        "actions": actions,
        "targets": targets,
        "lengths": lengths,
        "valid_mask": valid_mask,
        "next_state": next_state,
        "actions_are_indices": torch.tensor(False),
    }


def load_nk_dataset(config: V2Config) -> Tuple[Dataset, Dataset, Dataset]:
    path = Path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de données introuvable : {path}. "
            "V2 n'utilise que les données NK existantes."
        )

    data = np.load(path, allow_pickle=False)

    required = {
        config.obs_key,
        config.action_key,
        config.target_key,
        config.lengths_key,
    }
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"Clés manquantes dans {path} : {sorted(missing)}")

    observations = data[config.obs_key]
    actions = data[config.action_key]
    targets = data[config.target_key]
    lengths = data[config.lengths_key]

    n = len(lengths)
    indices = np.arange(n)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(indices)

    n_val = int(math.floor(n * config.val_fraction))
    n_test = int(math.floor(n * config.test_fraction))
    n_train = n - n_val - n_test

    if n_train <= 0:
        raise ValueError("Pas assez d'exemples pour entraîner.")

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def subset(idx):
        return NKSequenceDataset(
            observations[idx],
            actions[idx],
            targets[idx],
            lengths[idx],
        )

    return subset(train_idx), subset(val_idx), subset(test_idx)


# ---------------------------------------------------------------------------
# GSA : attention géométrique V2
# ---------------------------------------------------------------------------

class GeometricSelfAttention(nn.Module):
    """
    Attention causale avec biais de coût de transition.

    score_ij = Q_i K_j / sqrt(d_head) - beta * S_ij + gamma * A_ij

    S_ij : coût de transition estimé.
    A_ij : compatibilité action-état, si disponible.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        action_dim: int,
        beta_init: float = 0.5,
        learn_beta: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model doit être divisible par n_heads.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.action_proj = nn.Linear(action_dim, self.d_head)
        self.action_bias_proj = nn.Linear(self.d_head, 1)

        if learn_beta:
            self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init)))
        else:
            self.register_buffer(
                "log_beta",
                torch.tensor(math.log(max(beta_init, 1e-8))),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        transition_cost: torch.Tensor,
        actions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        device = x.device

        q = self.q_proj(x).view(
            batch_size, seq_len, self.n_heads, self.d_head
        ).transpose(1, 2)

        k = self.k_proj(x).view(
            batch_size, seq_len, self.n_heads, self.d_head
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            batch_size, seq_len, self.n_heads, self.d_head
        ).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_head)

        beta = torch.exp(self.log_beta)
        scores = scores - beta * transition_cost.unsqueeze(1)

        action_embed = self.action_proj(actions)
        action_bias = self.action_bias_proj(action_embed).squeeze(-1)
        scores = scores + action_bias.unsqueeze(1)

        causal_mask = torch.tril(
            torch.ones(
                seq_len, seq_len, dtype=torch.bool, device=device
            )
        )
        mask = causal_mask.unsqueeze(0) & valid_mask.unsqueeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        output = self.out_proj(context)

        return output, attention, transition_cost


class GSAEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        action_dim: int,
        ffn_dim: int,
        beta_init: float,
        learn_beta: bool,
        dropout: float,
    ):
        super().__init__()
        self.attention = GeometricSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            action_dim=action_dim,
            beta_init=beta_init,
            learn_beta=learn_beta,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        self.cost_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def estimate_transition_cost(
        self,
        hidden_states: torch.Tensor,
        actions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = hidden_states.shape

        left = hidden_states.unsqueeze(2).expand(
            batch_size, seq_len, seq_len, d_model
        )
        right = hidden_states.unsqueeze(1).expand(
            batch_size, seq_len, seq_len, d_model
        )

        pair = torch.cat([left, right], dim=-1)
        raw_cost = self.cost_head(pair).squeeze(-1)
        cost = F.softplus(raw_cost)

        invalid = ~valid_mask.unsqueeze(-1) | ~valid_mask.unsqueeze(-2)
        cost = cost.masked_fill(invalid, 0.0)

        return cost

    def forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        transition_cost = self.estimate_transition_cost(
            x, actions, valid_mask
        )

        attended, attention, cost = self.attention(
            x=x,
            transition_cost=transition_cost,
            actions=actions,
            valid_mask=valid_mask,
        )

        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x, attention, cost


# ---------------------------------------------------------------------------
# NK-LSTM-GSA V2
# ---------------------------------------------------------------------------

class NKLSTMGSAV2(nn.Module):
    def __init__(self, config: V2Config):
        super().__init__()
        self.config = config

        input_dim = config.obs_dim + config.action_dim
        self.input_proj = nn.Linear(input_dim, config.d_model)

        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0.0,
        )

        self.gsa_layers = nn.ModuleList(
            [
                GSAEncoderLayer(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    action_dim=config.action_dim,
                    ffn_dim=config.ffn_dim,
                    beta_init=config.beta_init,
                    learn_beta=config.learn_beta,
                    dropout=config.dropout,
                )
                for _ in range(config.gsa_layers)
            ]
        )

        if config.fusion == "concat":
            fusion_dim = config.d_model * 2
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.d_model),
            )
        elif config.fusion == "gated":
            fusion_dim = config.d_model * 2
            self.fusion_gate = nn.Linear(fusion_dim, 1)
            self.fusion = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.d_model),
            )
        else:
            raise ValueError("fusion doit être 'concat' ou 'gated'.")

        self.terminal_head = nn.Linear(config.d_model, config.target_dim)
        self.next_state_head = nn.Linear(config.d_model, config.obs_dim)
        self.next_action_head = nn.Linear(config.d_model, config.action_dim)
        self.order_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        lengths: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = observations.shape

        inputs = torch.cat([observations, actions], dim=-1)
        embedded = self.input_proj(inputs)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.lstm(packed)

        temporal, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=seq_len,
        )

        geometric = temporal
        attention_maps = []
        transition_costs = []

        for layer in self.gsa_layers:
            geometric, attention, cost = layer(
                geometric, actions, valid_mask
            )
            attention_maps.append(attention)
            transition_costs.append(cost)

        if self.config.fusion == "concat":
            fused_input = torch.cat([temporal, geometric], dim=-1)
            fused = self.fusion(fused_input)
        else:
            gate_input = torch.cat([temporal, geometric], dim=-1)
            gate = torch.sigmoid(self.fusion_gate(gate_input))
            fused = gate * temporal + (1.0 - gate) * geometric
            fused = self.fusion(fused)

        terminal = self.terminal_head(fused[:, -1])
        next_state = self.next_state_head(fused)
        next_action_logits = self.next_action_head(fused)

        order_logits = self.order_head(fused).squeeze(-1)

        return {
            "temporal": temporal,
            "geometric": geometric,
            "fused": fused,
            "terminal": terminal,
            "next_state": next_state,
            "next_action_logits": next_action_logits,
            "order_logits": order_logits,
            "attention": attention_maps[-1],
            "transition_cost": transition_costs[-1],
        }


# ---------------------------------------------------------------------------
# Pertes V2
# ---------------------------------------------------------------------------

def sequence_mean(loss: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    loss: (B, T)
    valid_mask: (B, T)
    """
    masked = loss * valid_mask.float()
    denominator = valid_mask.float().sum().clamp_min(1.0)
    return masked.sum() / denominator


def terminal_loss(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    return F.mse_loss(output["terminal"], batch["targets"])


def path_loss(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if "next_action" in batch:
        logits = output["next_action_logits"]
        target = batch["next_action"]

        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )

    next_state = batch["next_state"]
    predicted = output["next_state"]

    error = (predicted - next_state).abs().sum(dim=-1)
    shifted_mask = valid_mask.clone()
    shifted_mask[:, -1] = False

    return sequence_mean(error, shifted_mask)


def order_loss(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len = output["order_logits"].shape
    device = output["order_logits"].device

    positions = torch.arange(seq_len, device=device).float()
    positions = positions.unsqueeze(0).expand(batch_size, seq_len)

    max_lengths = batch["lengths"].float().unsqueeze(-1).clamp_min(2.0)
    normalized_positions = positions / (max_lengths - 1.0)

    predicted = torch.sigmoid(output["order_logits"])
    error = (predicted - normalized_positions) ** 2

    return sequence_mean(error, valid_mask)


def gsa_regularization(
    output: Dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    cost = output["transition_cost"]

    positive_penalty = F.softplus(epsilon - cost)
    magnitude_penalty = cost ** 2

    pair_valid = (
        valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)
    )

    positive_loss = sequence_mean(
        positive_penalty, pair_valid.float()
    )
    magnitude_loss = sequence_mean(
        magnitude_penalty, pair_valid.float()
    )

    return positive_loss + magnitude_loss


def nk_v2_loss(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: V2Config,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    valid_mask = batch["valid_mask"]

    terminal = terminal_loss(output, batch)
    path = path_loss(output, batch, valid_mask)
    order = order_loss(output, batch, valid_mask)
    gsa_reg = gsa_regularization(output, valid_mask, config.epsilon)

    total = (
        config.lambda_terminal * terminal
        + config.lambda_path * path
        + config.lambda_order * order
        + config.lambda_gsa_reg * gsa_reg
    )

    metrics = {
        "loss": total.item(),
        "terminal": terminal.item(),
        "path": path.item(),
        "order": order.item(),
        "gsa_reg": gsa_reg.item(),
    }

    return total, metrics


# ---------------------------------------------------------------------------
# Entraînement et validation
# ---------------------------------------------------------------------------

def move_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def run_epoch(
    model: NKLSTMGSAV2,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    config: V2Config,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
) -> Tuple[float, Dict[str, float]]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "loss": 0.0,
        "terminal": 0.0,
        "path": 0.0,
        "order": 0.0,
        "gsa_reg": 0.0,
    }
    count = 0

    for batch in loader:
        batch = move_batch(batch, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        if config.amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(
                    observations=batch["observations"],
                    actions=batch["actions"],
                    lengths=batch["lengths"],
                    valid_mask=batch["valid_mask"],
                )
                loss, metrics = nk_v2_loss(output, batch, config)
        else:
            output = model(
                observations=batch["observations"],
                actions=batch["actions"],
                lengths=batch["lengths"],
                valid_mask=batch["valid_mask"],
            )
            loss, metrics = nk_v2_loss(output, batch, config)

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )
                optimizer.step()

        for key in totals:
            totals[key] += metrics[key]
        count += 1

    return totals["loss"] / max(count, 1), {
        key: value / max(count, 1) for key, value in totals.items()
    }


@torch.no_grad()
def evaluate(
    model: NKLSTMGSAV2,
    loader: DataLoader,
    config: V2Config,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    return run_epoch(model, loader, None, config, device, None)


def save_checkpoint(
    path: Path,
    model: NKLSTMGSAV2,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    config: V2Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "best_val_loss": best_val_loss,
            "config": asdict(config),
        },
        path,
    )


def build_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: V2Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_sequences,
        generator=generator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_sequences,
        pin_memory=True,
    )

    return train_loader, val_loader, val_loader


def build_optimizer_and_scheduler(
    model: NKLSTMGSAV2,
    config: V2Config,
    steps_per_epoch: int,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.epochs - config.warmup_epochs, 1),
        )
    elif config.scheduler == "none":
        scheduler = None
    else:
        raise ValueError("scheduler doit être 'cosine' ou 'none'.")

    return optimizer, scheduler


def parse_args() -> V2Config:
    parser = argparse.ArgumentParser(
        description="Entraînement NK-LSTM-GSA V2."
    )

    parser.add_argument("--data-path", default="data/nk_sequences.npz")
    parser.add_argument("--output-dir", default="runs/nk_lstm_gsa_v2")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--gsa-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--fusion", choices=["concat", "gated"], default="concat")
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-save-every-epoch", action="store_true")

    args = parser.parse_args()

    config = V2Config(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        lstm_layers=args.lstm_layers,
        gsa_layers=args.gsa_layers,
        n_heads=args.n_heads,
        beta_init=args.beta,
        fusion=args.fusion,
        scheduler=args.scheduler,
        amp=args.amp,
        save_every_epoch=not args.no_save_every_epoch,
    )

    return config


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    device = torch.device(config.device)

    train_dataset, val_dataset, test_dataset = load_nk_dataset(config)
    train_loader, val_loader, test_loader = build_loaders(
        train_dataset, val_dataset, config
    )

    model = NKLSTMGSAV2(config).to(device)

    steps_per_epoch = max(len(train_loader), 1)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, config, steps_per_epoch
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if config.amp and device.type == "cuda"
        else None
    )

    best_val_loss = float("inf")
    patience = 0

    history = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, optimizer, config, device, scaler
        )

        if scheduler is not None and epoch > config.warmup_epochs:
            scheduler.step()

        val_loss, val_metrics = evaluate(
            model, val_loader, config, device
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "validation": val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"train {train_loss:.5f} | "
            f"val {val_loss:.5f} | "
            f"terminal {val_metrics['terminal']:.5f} | "
            f"path {val_metrics['path']:.5f} | "
            f"order {val_metrics['order']:.5f}"
        )

        improved = val_loss < best_val_loss - config.min_delta

        if improved:
            best_val_loss = val_loss
            patience = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                config,
            )
        else:
            patience += 1

        if config.save_every_epoch:
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                config,
            )

        with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        if patience >= config.early_stopping_patience:
            print(
                f"Early stopping à l'époque {epoch} ; "
                f"meilleure val_loss = {best_val_loss:.5f}."
            )
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics = evaluate(
        model, test_loader, config, device
    )

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "test_loss": test_loss,
                "metrics": test_metrics,
                "best_val_loss": best_val_loss,
            },
            handle,
            indent=2,
        )

    print(f"Test loss : {test_loss:.5f}")
    print(f"Résultats sauvegardés dans : {output_dir}")


if __name__ == "__main__":
    main()
