"""
nk_lstm_v1.py
Projet : NK-LSTM-PATH

V1 — apprentissage séquentiel de l'extrémité fibrée.

INTERFACES RÉELLES UTILISÉES
----------------------------
- balayage_commutateur_nk_v2.py : génération des états initiaux, paires,
  opérateur intégrable et wrapper step_flip_node().
- fiber_bundle.py : psi_sym() et mutate_postflip() via le wrapper.
- trajectories.py : NKLandscape et simulate_ogata_thinning().

EXPÉRIENCE A — CONTRÔLE GÉOMÉTRIQUE À DEUX MUTATIONS
---------------------------------------------------
Construit k -> m et m -> k avec step_flip_node() pour les deux transports.
Vérifie la fermeture du transport intégrable et mesure l'écart de fermeture
symétrique. Un LSTM X-seul non entraîné mesure séparément sa sensibilité à
l'ordre des configurations.

EXPÉRIENCE B — TRAJECTOIRES NK ADAPTATIVES
-----------------------------------------
Les trajectoires sont simulées par simulate_ogata_thinning() avec un unique
paysage NK partagé entre les deux modes. La règle de transport est choisie
par psi_fn : psi_integrable ou psi_sym.

Le LSTM reçoit, à chaque événement accepté :
    - X_t : configuration juste avant le flip ;
    - a_t : site muté, encodé one-hot.
La fibre initiale J0 est encodée une seule fois, au premier pas.
La cible est J_final. Les fibres intermédiaires J_t ne sont jamais fournies.

Pour gamma=0 (fixé dans cette V1), les temps d'attente ne modifient pas J ;
le chemin d'actions accepté et les configurations suffisent à déterminer
la composition du transport. Une trajectoire sans événement reçoit un token
sans action et doit prédire J_final = J0.

La validation sépare des états initiaux complets : toutes les trajectoires
issues d'un même (X0,J0) restent dans la même partition.

Exécution depuis la racine du dépôt :
    python .\\nk_lstm_v1.py

Exemple :
    python .\\nk_lstm_v1.py --n-initial-states 20 \\
        --n-trajectories-per-state 8 --epochs 100 \\
        --json-out .\\nk_lstm_v1_resultats.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from balayage_commutateur_nk_v2 import (  # noqa: E402
    ExperimentConfig,
    iter_site_pairs,
    psi_integrable,
    sample_binary_symmetric_states,
    step_flip_node,
    validate_config,
)
from fiber_bundle import psi_sym  # noqa: E402
from trajectories import NKLandscape, simulate_ogata_thinning  # noqa: E402

TransportMode = Literal["integrable", "symmetric"]


# ======================================================================
# 1. CONTRÔLE GÉOMÉTRIQUE : CARRÉS À DEUX MUTATIONS
# ======================================================================

@dataclass(frozen=True)
class PathRecord:
    initial_state_idx: int
    k: int
    m: int
    order: tuple[int, int]
    X_states: np.ndarray       # (3, N) : X0, X1, X2
    J_states: np.ndarray       # (3, N, N) : J0, J1, J2
    actions: np.ndarray        # (2, N) : sites mutés, one-hot


def build_two_step_path(
    X0: np.ndarray,
    J0: np.ndarray,
    order: tuple[int, int],
    eta: float,
    mode: TransportMode,
    initial_state_idx: int,
    k: int,
    m: int,
) -> PathRecord:
    """Construit un chemin via step_flip_node() du projet."""
    if len(order) != 2 or order[0] == order[1]:
        raise ValueError("order doit contenir exactement deux sites distincts.")

    X = np.asarray(X0, dtype=np.float64).copy()
    J = np.asarray(J0, dtype=np.float64).copy()
    if X.ndim != 1 or J.shape != (X.size, X.size):
        raise ValueError("Dimensions invalides pour l'état initial (X0, J0).")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(J)):
        raise ValueError("L'état initial contient des valeurs non finies.")

    X_values = [X.copy()]
    J_values = [J.copy()]
    actions = np.zeros((2, X.size), dtype=np.float32)

    for t, site in enumerate(order):
        if not 0 <= site < X.size:
            raise IndexError(f"Site de mutation hors limites : {site}.")
        actions[t, site] = 1.0
        X, J = step_flip_node(
            X, J, site=int(site), eta=float(eta), mode=mode
        )
        X = np.asarray(X, dtype=np.float64).copy()
        J = np.asarray(J, dtype=np.float64).copy()
        if X.shape != X_values[0].shape or J.shape != (X.size, X.size):
            raise ValueError("Le transport a retourné des dimensions inattendues.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(J)):
            raise ValueError("Le transport a produit des valeurs non finies.")
        X_values.append(X)
        J_values.append(J)

    return PathRecord(
        initial_state_idx=initial_state_idx,
        k=k,
        m=m,
        order=order,
        X_states=np.stack(X_values),
        J_states=np.stack(J_values),
        actions=actions,
    )


def build_two_step_records(
    initial_states: list[tuple[np.ndarray, np.ndarray]],
    cfg: ExperimentConfig,
    mode: TransportMode,
) -> list[PathRecord]:
    records: list[PathRecord] = []
    for state_idx, (X0, J0) in enumerate(initial_states):
        for k, m in iter_site_pairs(cfg.n_sites, unordered_only=True):
            records.append(build_two_step_path(
                X0, J0, (k, m), cfg.eta, mode, state_idx, k, m
            ))
            records.append(build_two_step_path(
                X0, J0, (m, k), cfg.eta, mode, state_idx, k, m
            ))
    return records


def paired_records(records: list[PathRecord]):
    grouped: dict[tuple[int, int, int], dict[tuple[int, int], PathRecord]] = {}
    for record in records:
        key = (record.initial_state_idx, record.k, record.m)
        grouped.setdefault(key, {})[record.order] = record
    for key, paths in grouped.items():
        state_idx, k, m = key
        a = paths.get((k, m))
        b = paths.get((m, k))
        if a is None or b is None:
            raise RuntimeError(f"Paire de chemins incomplète : {key}.")
        yield a, b


def verify_two_step_endpoints(
    records_int: list[PathRecord],
    records_sym: list[PathRecord],
    rtol: float,
    atol: float,
) -> dict[str, float]:
    gaps: dict[str, list[float]] = {"integrable": [], "symmetric": []}
    for mode, records in (("integrable", records_int), ("symmetric", records_sym)):
        for a, b in paired_records(records):
            if not np.array_equal(a.X_states[-1], b.X_states[-1]):
                raise RuntimeError(
                    f"Les extrémités X diffèrent pour {mode}, "
                    f"état {a.initial_state_idx}, paire ({a.k},{a.m})."
                )
            gap = float(np.linalg.norm(a.J_states[-1] - b.J_states[-1], ord="fro"))
            gaps[mode].append(gap)
            if mode == "integrable" and not np.allclose(
                a.J_states[-1], b.J_states[-1], rtol=rtol, atol=atol
            ):
                raise RuntimeError(
                    "Le contrôle intégrable ne ferme pas : "
                    f"état {a.initial_state_idx}, paire ({a.k},{a.m}), écart={gap:.6e}."
                )
    return {
        "integrable_gap_mean": float(np.mean(gaps["integrable"])),
        "integrable_gap_max": float(np.max(gaps["integrable"])),
        "symmetric_gap_mean": float(np.mean(gaps["symmetric"])),
        "symmetric_gap_max": float(np.max(gaps["symmetric"])),
    }


# ======================================================================
# 2. TRAJECTOIRES ADAPTATIVES : SIMULATE_OGATA_THINNING
# ======================================================================

@dataclass(frozen=True)
class AdaptiveRecord:
    initial_state_idx: int
    replicate_idx: int
    X_seq: np.ndarray       # (T, N), configurations avant chaque événement
    J0: np.ndarray          # (N, N), fibre initiale seulement
    actions: np.ndarray     # (T, N), site de chaque événement, one-hot
    J_final: np.ndarray     # (N, N), fibre cible
    n_events: int
    t_final: float


def make_adaptive_record(
    X0: np.ndarray,
    J0: np.ndarray,
    initial_state_idx: int,
    replicate_idx: int,
    eta: float,
    mode: TransportMode,
    landscape: NKLandscape,
    cfg: ExperimentConfig,
    t_max: float,
    rng_seed: int,
    max_events: int,
) -> AdaptiveRecord:
    """Simule une trajectoire adaptative réelle pour un mode de transport."""
    psi_fn = psi_integrable if mode == "integrable" else psi_sym
    rng = np.random.default_rng(rng_seed)

    simulation = simulate_ogata_thinning(
        X0=X0,
        J0=J0,
        t_max=t_max,
        mu=cfg.mu,
        beta=cfg.beta,
        theta_A=cfg.theta_A,
        lam=cfg.lam,
        fnk_fn=landscape,
        rng=rng,
        eta=eta,
        gamma=0.0,              # V1 : isole le transport, sans dissipation continue
        J_anchor=J0,
        max_events=max_events,
        psi_fn=psi_fn,
    )

    if simulation.n_accepted >= max_events and simulation.t_final < t_max - 1e-12:
        raise RuntimeError(
            f"Trajectoire tronquée par max_events={max_events} "
            f"(état {initial_state_idx}, réplication {replicate_idx}, mode {mode})."
        )

    x_pre: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    current_X = np.asarray(X0, dtype=np.float64).copy()

    for event in simulation.events:
        x_pre.append(current_X.copy())
        action = np.zeros(current_X.size, dtype=np.float32)
        action[int(event.node)] = 1.0
        action_rows.append(action)
        current_X = np.asarray(event.X, dtype=np.float64).copy()

    if not x_pre:
        # Token neutre : permet de traiter une trajectoire sans événement.
        x_pre.append(np.asarray(X0, dtype=np.float64).copy())
        action_rows.append(np.zeros(np.asarray(X0).size, dtype=np.float32))

    if not np.array_equal(current_X, np.asarray(simulation.X_final, dtype=np.float64)):
        raise RuntimeError("Reconstruction des configurations incohérente avec X_final.")

    return AdaptiveRecord(
        initial_state_idx=initial_state_idx,
        replicate_idx=replicate_idx,
        X_seq=np.stack(x_pre),
        J0=np.asarray(J0, dtype=np.float64).copy(),
        actions=np.stack(action_rows),
        J_final=np.asarray(simulation.J_final, dtype=np.float64).copy(),
        n_events=simulation.n_accepted,
        t_final=float(simulation.t_final),
    )


def build_adaptive_records(
    initial_states: list[tuple[np.ndarray, np.ndarray]],
    cfg: ExperimentConfig,
    mode: TransportMode,
    landscape: NKLandscape,
    n_trajectories_per_state: int,
    t_max: float,
    sim_seed: int,
    max_events: int,
) -> list[AdaptiveRecord]:
    """Génère les trajectoires en gardant paysage et conditions initiales communs."""
    records: list[AdaptiveRecord] = []
    for state_idx, (X0, J0) in enumerate(initial_states):
        for replicate_idx in range(n_trajectories_per_state):
            # Même graine par (état, réplication) pour les deux modes :
            # nombres aléatoires communs, malgré des trajectoires qui divergent.
            seed = sim_seed + state_idx * 100_003 + replicate_idx * 997
            records.append(make_adaptive_record(
                X0=X0,
                J0=J0,
                initial_state_idx=state_idx,
                replicate_idx=replicate_idx,
                eta=cfg.eta,
                mode=mode,
                landscape=landscape,
                cfg=cfg,
                t_max=t_max,
                rng_seed=seed,
                max_events=max_events,
            ))
    return records


# ======================================================================
# 3. ENCODAGE ET MODÈLES LSTM
# ======================================================================

def numpy_matrix_to_upper_triangle(J: np.ndarray) -> np.ndarray:
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J doit être une matrice carrée.")
    return J[np.triu_indices(J.shape[0], k=1)].copy()


def matrix_to_upper_triangle_torch(J: torch.Tensor) -> torch.Tensor:
    if J.ndim < 2 or J.shape[-1] != J.shape[-2]:
        raise ValueError("J doit avoir la forme (..., N, N).")
    n = J.shape[-1]
    indices = torch.triu_indices(n, n, offset=1, device=J.device)
    return J[..., indices[0], indices[1]]


class NKLSTM_X(nn.Module):
    """LSTM X-seul non entraîné, utilisé comme contrôle de sensibilité à l'ordre."""
    def __init__(self, n_sites: int, d_x: int = 32, hidden: int = 64):
        super().__init__()
        self.n_sites = n_sites
        self.embed_x = nn.Linear(n_sites, d_x)
        self.lstm = nn.LSTM(d_x, hidden, batch_first=True)

    def forward(self, x_seq: torch.Tensor, state=None):
        if x_seq.ndim != 3 or x_seq.shape[-1] != self.n_sites:
            raise ValueError("x_seq doit avoir la forme (B,T,N).")
        z = torch.tanh(self.embed_x(x_seq))
        return self.lstm(z, state)


class NKLSTM_XJA(nn.Module):
    """
    LSTM endpoint : (séquence X, actions, fibre initiale J0) -> fibre finale.

    J0 est injectée au premier pas seulement. Aucune fibre intermédiaire
    n'est fournie au modèle.
    """
    def __init__(
        self,
        n_sites: int,
        d_x: int = 32,
        d_j: int = 64,
        d_action: int = 16,
        hidden: int = 64,
    ):
        super().__init__()
        self.n_sites = n_sites
        self.d_j_input = n_sites * (n_sites - 1) // 2
        self.embed_x = nn.Linear(n_sites, d_x)
        self.embed_j0 = nn.Linear(self.d_j_input, d_j)
        self.embed_action = nn.Linear(n_sites, d_action)
        fused_dim = d_x + d_j + d_action
        self.fusion = nn.Linear(fused_dim, fused_dim)
        self.lstm = nn.LSTM(fused_dim, hidden, batch_first=True)
        self.head_j_final = nn.Linear(hidden, self.d_j_input)

    def forward(
        self,
        x_seq: torch.Tensor,
        j0_triangle: torch.Tensor,
        action_seq: torch.Tensor,
        lengths: torch.Tensor | None = None,
        state=None,
    ):
        if x_seq.ndim != 3 or x_seq.shape[-1] != self.n_sites:
            raise ValueError("x_seq doit avoir la forme (B,T,N).")
        if j0_triangle.ndim != 2 or j0_triangle.shape[-1] != self.d_j_input:
            raise ValueError("j0_triangle doit avoir la forme (B,d_J).")
        if action_seq.ndim != 3 or action_seq.shape[-1] != self.n_sites:
            raise ValueError("action_seq doit avoir la forme (B,T,N).")
        if x_seq.shape[:2] != action_seq.shape[:2]:
            raise ValueError("Les dimensions batch/temps de X et action doivent correspondre.")
        if x_seq.shape[0] != j0_triangle.shape[0] or x_seq.shape[1] < 1:
            raise ValueError("Dimensions batch invalides ou séquence vide.")

        batch_size, steps, _ = x_seq.shape
        if lengths is None:
            lengths = torch.full((batch_size,), steps, dtype=torch.long)
        if lengths.ndim != 1 or lengths.numel() != batch_size:
            raise ValueError("lengths doit contenir une longueur par séquence.")
        if torch.any(lengths < 1) or torch.any(lengths > steps):
            raise ValueError("Chaque longueur doit être comprise entre 1 et T.")

        x = torch.tanh(self.embed_x(x_seq))
        j0 = torch.tanh(self.embed_j0(j0_triangle))
        action = torch.tanh(self.embed_action(action_seq))

        j_context = torch.zeros(
            batch_size, steps, j0.shape[-1], dtype=j0.dtype, device=j0.device
        )
        j_context[:, 0, :] = j0

        z = torch.cat((x, j_context, action), dim=-1)
        z = torch.tanh(self.fusion(z))
        packed = pack_padded_sequence(
            z, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, final_state = self.lstm(packed, state)
        predicted_j_final = self.head_j_final(final_state[0][-1])
        return predicted_j_final, final_state


# ======================================================================
# 4. DATASET VARIABLE-LONGUEUR ET PARTITION
# ======================================================================

class EndpointDataset(Dataset):
    def __init__(self, records: list[AdaptiveRecord]):
        self.records = records
        if not records:
            raise ValueError("Dataset vide.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        x = torch.from_numpy(record.X_seq.astype(np.float32, copy=False))
        j0 = torch.from_numpy(numpy_matrix_to_upper_triangle(record.J0).astype(np.float32))
        action = torch.from_numpy(record.actions.astype(np.float32, copy=False))
        target = torch.from_numpy(numpy_matrix_to_upper_triangle(record.J_final).astype(np.float32))
        return x, j0, action, target


def collate_endpoint_batch(batch):
    xs, j0s, actions, targets = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    action_pad = pad_sequence(actions, batch_first=True, padding_value=0.0)
    return (
        x_pad,
        torch.stack(j0s),
        action_pad,
        lengths,
        torch.stack(targets),
    )


def split_initial_states(
    n_initial_states: int,
    validation_fraction: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    if n_initial_states < 2:
        raise ValueError("Il faut au moins deux états initiaux pour la validation indépendante.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction doit être comprise entre 0 et 1.")
    indices = np.arange(n_initial_states)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_val = max(1, int(math.ceil(n_initial_states * validation_fraction)))
    n_val = min(n_val, n_initial_states - 1)
    val = set(int(i) for i in indices[:n_val])
    train = set(int(i) for i in indices[n_val:])
    return train, val


def select_records(records: list[AdaptiveRecord], indices: set[int]) -> list[AdaptiveRecord]:
    selected = [r for r in records if r.initial_state_idx in indices]
    if not selected:
        raise ValueError("Aucune trajectoire dans la partition demandée.")
    return selected


# ======================================================================
# 5. ENTRAÎNEMENT ET ÉVALUATION
# ======================================================================

@dataclass
class TrainingResult:
    mode: str
    n_train_trajectories: int
    n_validation_trajectories: int
    n_train_initial_states: int
    n_validation_initial_states: int
    best_epoch: int
    train_mse_at_best: float
    validation_mse: float
    validation_J0_baseline_mse: float


def make_loader(records: list[AdaptiveRecord], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        EndpointDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=collate_endpoint_batch,
    )


def evaluate_mse_and_baseline(
    model: NKLSTM_XJA,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    squared_error = 0.0
    baseline_error = 0.0
    total_elements = 0
    with torch.no_grad():
        for x, j0, action, lengths, target in loader:
            x, j0, action, target = (
                x.to(device), j0.to(device), action.to(device), target.to(device)
            )
            prediction, _ = model(x, j0, action, lengths)
            squared_error += torch.sum((prediction - target) ** 2).item()
            baseline_error += torch.sum((j0 - target) ** 2).item()
            total_elements += target.numel()
    if total_elements == 0:
        raise ValueError("Ensemble de validation vide.")
    return squared_error / total_elements, baseline_error / total_elements


def train_mode_model(
    mode: TransportMode,
    train_records: list[AdaptiveRecord],
    validation_records: list[AdaptiveRecord],
    n_sites: int,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    torch_seed: int,
    device: torch.device,
    train_state_count: int,
    validation_state_count: int,
) -> tuple[NKLSTM_XJA, TrainingResult]:
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    model = NKLSTM_XJA(n_sites=n_sites, hidden=hidden).to(device)
    train_loader = make_loader(train_records, batch_size, True, torch_seed)
    validation_loader = make_loader(validation_records, batch_size, False, torch_seed)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    best_validation = float("inf")
    best_train = float("inf")
    best_baseline = float("inf")
    best_epoch = 0
    best_state = None
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_squared_error = 0.0
        total_elements = 0
        for x, j0, action, lengths, target in train_loader:
            x, j0, action, target = (
                x.to(device), j0.to(device), action.to(device), target.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(x, j0, action, lengths)
            loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Perte non finie pour {mode}, époque {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_squared_error += torch.sum((prediction.detach() - target) ** 2).item()
            total_elements += target.numel()

        train_mse = total_squared_error / max(total_elements, 1)
        val_mse, baseline_mse = evaluate_mse_and_baseline(
            model, validation_loader, device
        )
        if val_mse < best_validation:
            best_validation = val_mse
            best_train = train_mse
            best_baseline = baseline_mse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"Aucun état de modèle valide pour {mode}.")
    model.load_state_dict(best_state)
    model.to(device)

    result = TrainingResult(
        mode=mode,
        n_train_trajectories=len(train_records),
        n_validation_trajectories=len(validation_records),
        n_train_initial_states=train_state_count,
        n_validation_initial_states=validation_state_count,
        best_epoch=best_epoch,
        train_mse_at_best=best_train,
        validation_mse=best_validation,
        validation_J0_baseline_mse=best_baseline,
    )
    return model, result


# ======================================================================
# 6. DIAGNOSTICS D'ÉTATS CACHÉS ET PRÉDICTION SUR LE CONTRÔLE À DEUX FLIPS
# ======================================================================

@torch.no_grad()
def state_distances(state_a, state_b) -> dict[str, float]:
    h_a, c_a = state_a
    h_b, c_b = state_b
    d_h = torch.linalg.vector_norm(h_a - h_b)
    d_c = torch.linalg.vector_norm(c_a - c_b)
    d_total = torch.sqrt(d_h.square() + d_c.square())
    return {"D_h": float(d_h.item()), "D_c": float(d_c.item()), "D_LSTM": float(d_total.item())}


def summarize_distances(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("Aucune distance à résumer.")
    out = {"n_pairs": float(len(values))}
    for name in ("D_h", "D_c", "D_LSTM"):
        arr = np.asarray([v[name] for v in values], dtype=np.float64)
        out[f"mean_{name}"] = float(np.mean(arr))
        out[f"max_{name}"] = float(np.max(arr))
    return out


def _controlled_inputs(record: PathRecord, device: torch.device):
    x = torch.from_numpy(record.X_states[:-1].astype(np.float32))[None, ...].to(device)
    j0 = torch.from_numpy(numpy_matrix_to_upper_triangle(record.J_states[0]).astype(np.float32))[None, ...].to(device)
    action = torch.from_numpy(record.actions.astype(np.float32))[None, ...].to(device)
    lengths = torch.tensor([2], dtype=torch.long)
    return x, j0, action, lengths


@torch.no_grad()
def evaluate_x_only_order(
    model: NKLSTM_X,
    records: list[PathRecord],
    selected_indices: set[int],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    selected = [r for r in records if r.initial_state_idx in selected_indices]
    # Les enregistrements sont groupés par paire et par ordre.
    grouped: dict[tuple[int, int, int], dict[tuple[int, int], PathRecord]] = {}
    for r in selected:
        grouped.setdefault((r.initial_state_idx, r.k, r.m), {})[r.order] = r
    values: list[dict[str, float]] = []
    for (state_idx, k, m), paths in grouped.items():
        a, b = paths.get((k, m)), paths.get((m, k))
        if a is None or b is None:
            raise RuntimeError("Chemin inverse absent pour le contrôle X-seul.")
        xa = torch.from_numpy(a.X_states.astype(np.float32))[None, ...].to(device)
        xb = torch.from_numpy(b.X_states.astype(np.float32))[None, ...].to(device)
        _, state_a = model(xa)
        _, state_b = model(xb)
        values.append(state_distances(state_a, state_b))
    return summarize_distances(values)


@torch.no_grad()
def evaluate_trained_order_sensitivity(
    model: NKLSTM_XJA,
    records: list[PathRecord],
    selected_indices: set[int],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    selected = [r for r in records if r.initial_state_idx in selected_indices]
    grouped: dict[tuple[int, int, int], dict[tuple[int, int], PathRecord]] = {}
    for r in selected:
        grouped.setdefault((r.initial_state_idx, r.k, r.m), {})[r.order] = r
    values: list[dict[str, float]] = []
    for (state_idx, k, m), paths in grouped.items():
        a, b = paths.get((k, m)), paths.get((m, k))
        if a is None or b is None:
            raise RuntimeError("Chemin inverse absent pour le diagnostic XJA.")
        xa, ja, aa, la = _controlled_inputs(a, device)
        xb, jb, ab, lb = _controlled_inputs(b, device)
        _, state_a = model(xa, ja, aa, la)
        _, state_b = model(xb, jb, ab, lb)
        values.append(state_distances(state_a, state_b))
    return summarize_distances(values)


@torch.no_grad()
def evaluate_adaptive_validation(
    model: NKLSTM_XJA,
    records: list[AdaptiveRecord],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = make_loader(records, batch_size, False, 0)
    model.eval()
    errors: list[float] = []
    baselines: list[float] = []
    with torch.no_grad():
        for x, j0, action, lengths, target in loader:
            x, j0, action, target = (
                x.to(device), j0.to(device), action.to(device), target.to(device)
            )
            prediction, _ = model(x, j0, action, lengths)
            per_sample = torch.mean((prediction - target) ** 2, dim=1)
            base_sample = torch.mean((j0 - target) ** 2, dim=1)
            errors.extend(per_sample.cpu().numpy().tolist())
            baselines.extend(base_sample.cpu().numpy().tolist())

    event_counts = np.asarray([r.n_events for r in records], dtype=np.float64)
    return {
        "trajectory_endpoint_mse_mean": float(np.mean(errors)),
        "trajectory_endpoint_mse_max": float(np.max(errors)),
        "J0_baseline_mse_mean": float(np.mean(baselines)),
        "n_validation_trajectories": float(len(records)),
        "mean_accepted_events": float(np.mean(event_counts)),
        "zero_event_fraction": float(np.mean(event_counts == 0.0)),
    }


# ======================================================================
# 7. CONFIGURATION ET EXÉCUTION
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apprentissage LSTM sur les transports NK-LSTM-PATH réels."
    )
    parser.add_argument("--n-sites", type=int, default=6)
    parser.add_argument("--n-initial-states", type=int, default=20)
    parser.add_argument("--init-seed", type=int, default=7)
    parser.add_argument("--scale-j", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--theta-a", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--nk-k", type=int, default=3)
    parser.add_argument("--nk-seed", type=int, default=7)
    parser.add_argument("--t-max", type=float, default=5.0)
    parser.add_argument("--n-trajectories-per-state", type=int, default=8)
    parser.add_argument("--sim-seed", type=int, default=2026)
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> dict:
    if args.epochs < 1 or args.batch_size < 1 or args.hidden < 1:
        raise ValueError("epochs, batch-size et hidden doivent être >= 1.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0 or args.patience < 1:
        raise ValueError("Hyperparamètres d'entraînement non valides.")
    if args.n_trajectories_per_state < 1 or args.max_events < 1:
        raise ValueError("n-trajectories-per-state et max-events doivent être >= 1.")
    if not np.isfinite(args.t_max) or args.t_max < 0.0:
        raise ValueError("t-max doit être fini et >= 0.")

    cfg = ExperimentConfig(
        n_sites=args.n_sites,
        n_initial_states=args.n_initial_states,
        init_seed=args.init_seed,
        scale_J=args.scale_j,
        eta=args.eta,
        mu=args.mu,
        beta=args.beta,
        theta_A=args.theta_a,
        lam=args.lam,
        nk_K=args.nk_k,
        nk_seed=args.nk_seed,
    )
    validate_config(cfg)
    if cfg.n_initial_states < 2:
        raise ValueError("n-initial-states doit être >= 2.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé, mais aucun périphérique CUDA n'est disponible.")
    device = torch.device(args.device)

    initial_rng = np.random.default_rng(cfg.init_seed)
    initial_states = sample_binary_symmetric_states(
        n_sites=cfg.n_sites,
        n_states=cfg.n_initial_states,
        rng=initial_rng,
        scale=cfg.scale_J,
    )

    # --------------------------------------------------------------
    # A. Contrôle déterministe du carré de mutations
    # --------------------------------------------------------------
    square_int = build_two_step_records(initial_states, cfg, "integrable")
    square_sym = build_two_step_records(initial_states, cfg, "symmetric")
    square_checks = verify_two_step_endpoints(
        square_int, square_sym, rtol=cfg.rtol, atol=cfg.atol
    )

    # --------------------------------------------------------------
    # B. Trajectoires adaptatives sur un paysage NK unique et partagé
    # --------------------------------------------------------------
    landscape = NKLandscape(N=cfg.n_sites, K=cfg.nk_K, seed=cfg.nk_seed)
    adaptive_records: dict[str, list[AdaptiveRecord]] = {}
    for mode in ("integrable", "symmetric"):
        adaptive_records[mode] = build_adaptive_records(
            initial_states=initial_states,
            cfg=cfg,
            mode=mode,  # type: ignore[arg-type]
            landscape=landscape,
            n_trajectories_per_state=args.n_trajectories_per_state,
            t_max=args.t_max,
            sim_seed=args.sim_seed,
            max_events=args.max_events,
        )

    train_indices, validation_indices = split_initial_states(
        cfg.n_initial_states,
        args.validation_fraction,
        seed=args.torch_seed,
    )

    trained_models: dict[str, NKLSTM_XJA] = {}
    training_results: dict[str, TrainingResult] = {}
    adaptive_validation: dict[str, dict[str, float]] = {}

    for mode in ("integrable", "symmetric"):
        train_records = select_records(adaptive_records[mode], train_indices)
        val_records = select_records(adaptive_records[mode], validation_indices)
        model, train_result = train_mode_model(
            mode=mode,  # type: ignore[arg-type]
            train_records=train_records,
            validation_records=val_records,
            n_sites=cfg.n_sites,
            hidden=args.hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            torch_seed=args.torch_seed,
            device=device,
            train_state_count=len(train_indices),
            validation_state_count=len(validation_indices),
        )
        trained_models[mode] = model
        training_results[mode] = train_result
        adaptive_validation[mode] = evaluate_adaptive_validation(
            model, val_records, args.batch_size, device
        )

    # --------------------------------------------------------------
    # C. Diagnostics d'ordre sur les états initiaux réservés à la validation
    # --------------------------------------------------------------
    torch.manual_seed(args.torch_seed)
    model_x = NKLSTM_X(cfg.n_sites, hidden=args.hidden).to(device)
    order_x = evaluate_x_only_order(
        model_x, square_int, validation_indices, device
    )
    order_xja = {
        mode: evaluate_trained_order_sensitivity(
            trained_models[mode], square_records, validation_indices, device
        )
        for mode, square_records in (("integrable", square_int), ("symmetric", square_sym))
    }

    event_stats = {}
    for mode in ("integrable", "symmetric"):
        counts = np.asarray([r.n_events for r in adaptive_records[mode]], dtype=np.float64)
        event_stats[mode] = {
            "n_trajectories": int(len(counts)),
            "mean_accepted_events": float(np.mean(counts)),
            "max_accepted_events": int(np.max(counts)),
            "zero_event_fraction": float(np.mean(counts == 0.0)),
        }

    payload = {
        "experiment": "NK-LSTM-PATH V1 — controlled order square + adaptive NK trajectory endpoint prediction",
        "config": {
            "n_sites": cfg.n_sites,
            "n_initial_states": cfg.n_initial_states,
            "init_seed": cfg.init_seed,
            "scale_J": cfg.scale_J,
            "eta": cfg.eta,
            "mu": cfg.mu,
            "beta": cfg.beta,
            "theta_A": cfg.theta_A,
            "lambda": cfg.lam,
            "nk_K": cfg.nk_K,
            "nk_seed": cfg.nk_seed,
            "t_max": args.t_max,
            "gamma": 0.0,
            "n_trajectories_per_state": args.n_trajectories_per_state,
            "sim_seed": args.sim_seed,
            "max_events": args.max_events,
            "hidden": args.hidden,
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "validation_fraction": args.validation_fraction,
            "torch_seed": args.torch_seed,
            "device": str(device),
        },
        "split": {
            "train_initial_state_indices": sorted(train_indices),
            "validation_initial_state_indices": sorted(validation_indices),
        },
        "controlled_two_step_endpoint_checks": square_checks,
        "adaptive_trajectory_event_statistics": event_stats,
        "training": {mode: asdict(value) for mode, value in training_results.items()},
        "adaptive_validation": adaptive_validation,
        "controlled_X_only_untrained_order_sensitivity": order_x,
        "controlled_X_J0_action_trained_order_sensitivity": order_xja,
        "interpretation_limits": [
            "Le carré à deux mutations est un contrôle déterministe; ses chemins ne sont pas sélectionnés par les taux NK.",
            "Les trajectoires adaptatives sont générées par simulate_ogata_thinning avec un paysage NK unique partagé.",
            "Les modèles adaptatifs reçoivent X_t et le site muté; J0 est fourni une seule fois et les J intermédiaires sont masqués.",
            "La cible est J_final : l'expérience évalue l'apprentissage de la composition séquentielle du transport sur des chemins adaptatifs.",
            "gamma=0 dans cette V1; la dissipation continue et l'information des temps d'attente ne sont pas modélisées par l'entrée LSTM.",
            "La validation sépare les états initiaux complets; elle ne sépare pas les réalisations aléatoires d'un même état entre les partitions.",
            "Les modèles intégrable et symétrique sont entraînés séparément; leurs coordonnées cachées ne sont pas directement comparables entre modèles.",
            "Les transports peuvent générer des distributions de trajectoires différentes malgré les mêmes états initiaux, le même paysage et des graines de simulation appariées; les MSE des deux modes ne sont donc pas une comparaison appariée de difficulté.",
            "Le LSTM reçoit les événements acceptés (sites mutés), pas les propositions rejetées ni les taux complets; il prédit le transport de J le long d'une histoire réalisée, pas la règle de sélection stochastique.",
            "Une erreur prédictive faible ne prouve pas à elle seule une reconstruction générale de l'histoire ni une holonomie géométrique.",
            "Le LSTM X-seul est non entraîné : sa distance cachée est un diagnostic de sensibilité à l'ordre, non une mémoire apprise.",
        ],
    }

    print("\nNK-LSTM-PATH — V1 : CARRÉ CONTRÔLÉ + TRAJECTOIRES NK ADAPTATIVES")
    print("=" * 78)
    print(
        f"N={cfg.n_sites} | états initiaux={cfg.n_initial_states} | "
        f"réplications/état={args.n_trajectories_per_state} | "
        f"eta={cfg.eta:g} | dim(Sym_0)={cfg.n_sites * (cfg.n_sites - 1) // 2}"
    )
    print(
        f"Paysage NK : K={cfg.nk_K} | seed={cfg.nk_seed} | "
        f"mu={cfg.mu:g} | beta={cfg.beta:g} | theta_A={cfg.theta_A:g} | lambda={cfg.lam:g}"
    )
    print(
        f"Validation par état initial : entraînement={len(train_indices)} | "
        f"validation={len(validation_indices)}"
    )

    print("\nA — CONTRÔLE GÉOMÉTRIQUE À DEUX MUTATIONS")
    for key, value in square_checks.items():
        print(f"{key} : {value:.6e}")

    print("\nDIAGNOSTIC X-SEUL NON ENTRAÎNÉ, SUR LES ÉTATS DE VALIDATION")
    for key, value in order_x.items():
        print(f"{key} : {int(value) if key == 'n_pairs' else f'{value:.6e}'}")

    for mode in ("integrable", "symmetric"):
        tr = training_results[mode]
        print(f"\nB — TRANSPORT {mode.upper()} : TRAJECTOIRES ADAPTATIVES")
        print(f"Trajectoires entraînement / validation : {tr.n_train_trajectories} / {tr.n_validation_trajectories}")
        print(f"États initiaux entraînement / validation : {tr.n_train_initial_states} / {tr.n_validation_initial_states}")
        print(f"Meilleure époque : {tr.best_epoch}")
        print(f"MSE entraînement au meilleur modèle : {tr.train_mse_at_best:.6e}")
        print(f"MSE validation : {tr.validation_mse:.6e}")
        print(f"MSE baseline J_final = J0 : {tr.validation_J0_baseline_mse:.6e}")
        print(f"Événements acceptés moyens : {event_stats[mode]['mean_accepted_events']:.3f}")
        print(f"Fraction de trajectoires sans événement : {event_stats[mode]['zero_event_fraction']:.3f}")
        print("Validation :")
        for key, value in adaptive_validation[mode].items():
            print(f"  {key} : {value:.6e}")
        print("Sensibilité de l'état récurrent sur les carrés de validation :")
        for key, value in order_xja[mode].items():
            print(f"  {key} : {int(value) if key == 'n_pairs' else f'{value:.6e}'}")

    print("\nPrécaution : la comparaison des modèles cachés entre transports n'est pas une comparaison dans un repère commun.")
    print("La prédiction de J_final doit être comparée à la baseline J_final = J0.")
    print("=" * 78)
    return payload


def main() -> None:
    args = parse_args()
    payload = run_experiment(args)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\nExport JSON : {args.json_out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExpérience interrompue par l'utilisateur.")
        raise
