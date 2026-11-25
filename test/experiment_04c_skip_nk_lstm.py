"""
experiment_04c_skip_nk_lstm.py
Protocole 04c (NK-LSTM-PATH v1.4 - Skip-connection NK) :
Sépare strictement l'intégration de l'histoire observable (X_{0:t}, dt_{0:t})
dans le LSTM de l'injection directe (skip-connection) des features locales
instantanées nk_feat_t dans la tête de sortie.
Conserve intact experiment_04b_known_j0_results.json et enregistre dans
experiment_04c_skip_nk_results.json.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "PyTorch (torch) est requis pour experiment_04c_skip_nk_lstm.py."
    ) from exc


# ============================================================================
# 1. Configuration centralisée (identique à 04b pour comparaison exacte)
# ============================================================================


@dataclass
class Experiment04cConfig:
    n_spins: int = 4
    k_epistasis: int = 2
    neighbor_mode: str = "random"
    train_nk_seeds: Tuple[int, ...] = (101, 102, 103, 104, 105, 106)
    test_nk_seeds: Tuple[int, ...] = (201, 202, 203)

    eta: float = 0.25
    theta_a: float = 0.35
    gamma: float = 0.5
    mu: float = 1.0
    beta: float = 2.0
    coupled_lambda: float = 0.25

    horizon_steps: int = 12
    max_proposals_per_step: int = 2000
    n_train_trajectories: int = 600
    n_test_free_trajectories: int = 240

    paired_seq_A: Tuple[int, ...] = (0, 1, 2, 3)
    paired_seq_B: Tuple[int, ...] = (3, 2, 1, 0)
    n_test_paired_trials: int = 240

    m3_windows: Tuple[int, ...] = (2, 4, 6)
    hidden_dim: int = 64
    epochs: int = 28
    batch_size: int = 128
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4

    integrable_equiv_tol: float = 0.02
    global_seed: int = 2026
    out_json: str = "experiment_04c_skip_nk_results.json"


# ============================================================================
# 2. Paysage NK canonique et fibres initiales déterministes
# ============================================================================


def spins_to_bits(X: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("X doit être un vecteur unidimensionnel.")
    if not np.all(np.isin(arr, [-1.0, 1.0])):
        raise ValueError("Les spins doivent appartenir strictement à {-1, +1}.")
    return (arr > 0.0).astype(np.int64)


def spins_to_state_code(X: np.ndarray) -> int:
    code = 0
    for val in X:
        code = (code << 1) | (1 if val > 0.0 else 0)
    return code


@dataclass
class NKLandscape:
    n_spins: int
    k_epistasis: int
    neighbors: np.ndarray
    tables: np.ndarray
    seed: int = 2026
    neighbor_mode: str = "random"

    def __post_init__(self) -> None:
        if self.n_spins < 1:
            raise ValueError("n_spins doit être >= 1.")
        if not (0 <= self.k_epistasis < self.n_spins):
            raise ValueError(
                f"k_epistasis ({self.k_epistasis}) doit vérifier "
                f"0 <= K < n_spins ({self.n_spins})."
            )
        self.neighbors = np.asarray(self.neighbors, dtype=np.int64)
        self.tables = np.asarray(self.tables, dtype=np.float64)
        if self.neighbors.shape != (self.n_spins, self.k_epistasis):
            raise ValueError("Forme invalide pour neighbors.")
        if self.tables.shape != (self.n_spins, 1 << (self.k_epistasis + 1)):
            raise ValueError("Forme invalide pour tables.")
        if not np.all(np.isfinite(self.tables)):
            raise ValueError("Tables NK non finies.")
        if self.k_epistasis > 0 and (
            np.any(self.neighbors < 0) or np.any(self.neighbors >= self.n_spins)
        ):
            raise ValueError("Indices de voisins hors limites.")
        for i in range(self.n_spins):
            row = self.neighbors[i]
            if i in row or len(np.unique(row)) != len(row):
                raise ValueError(f"Voisinage invalide pour le site {i}.")

        self._affected_sites: List[np.ndarray] = []
        for k in range(self.n_spins):
            dependents = [k]
            for i in range(self.n_spins):
                if k in self.neighbors[i]:
                    dependents.append(i)
            self._affected_sites.append(
                np.array(sorted(set(dependents)), dtype=np.int64)
            )

        n_states = 1 << self.n_spins
        self._fitness_cache = np.zeros(n_states, dtype=np.float64)
        self._df_cache = np.zeros((n_states, self.n_spins), dtype=np.float64)
        for code in range(n_states):
            X = np.array(
                [
                    1.0 if ((code >> (self.n_spins - 1 - i)) & 1) else -1.0
                    for i in range(self.n_spins)
                ],
                dtype=np.float64,
            )
            self._fitness_cache[code] = self.fitness(X)
            for k in range(self.n_spins):
                self._df_cache[code, k] = self.delta_fitness_flip(X, k)

    @classmethod
    def generate(
        cls,
        n_spins: int,
        k_epistasis: int,
        seed: int = 2026,
        neighbor_mode: str = "random",
    ) -> "NKLandscape":
        rng = np.random.default_rng(seed)
        neighbors = np.zeros((n_spins, k_epistasis), dtype=np.int64)
        if neighbor_mode == "random":
            for i in range(n_spins):
                candidates = [idx for idx in range(n_spins) if idx != i]
                if k_epistasis > 0:
                    neighbors[i] = np.sort(
                        rng.choice(candidates, size=k_epistasis, replace=False)
                    )
        elif neighbor_mode == "adjacent":
            for i in range(n_spins):
                if k_epistasis > 0:
                    neighbors[i] = np.array(
                        [(i + off) % n_spins for off in range(1, k_epistasis + 1)],
                        dtype=np.int64,
                    )
        else:
            raise ValueError(f"neighbor_mode inconnu : {neighbor_mode}")

        tables = rng.uniform(0.0, 1.0, size=(n_spins, 1 << (k_epistasis + 1)))
        return cls(
            n_spins=n_spins,
            k_epistasis=k_epistasis,
            neighbors=neighbors,
            tables=tables,
            seed=seed,
            neighbor_mode=neighbor_mode,
        )

    def _entry_index(self, site_i: int, bits: np.ndarray) -> int:
        idx = int(bits[site_i])
        for nb in self.neighbors[site_i]:
            idx = (idx << 1) | int(bits[nb])
        return idx

    def fitness(self, X: Sequence[float] | np.ndarray) -> float:
        bits = spins_to_bits(X)
        total = 0.0
        for i in range(self.n_spins):
            total += self.tables[i, self._entry_index(i, bits)]
        return float(total / self.n_spins)

    def delta_fitness_flip(
        self, X: Sequence[float] | np.ndarray, node_k: int
    ) -> float:
        if not (0 <= node_k < self.n_spins):
            raise IndexError(f"node_k ({node_k}) hors limites.")
        bits = spins_to_bits(X)
        affected = self._affected_sites[node_k]
        f_before = sum(
            self.tables[i, self._entry_index(i, bits)] for i in affected
        )
        bits[node_k] ^= 1
        f_after = sum(
            self.tables[i, self._entry_index(i, bits)] for i in affected
        )
        return float((f_after - f_before) / self.n_spins)


def make_symmetric_zero_diag(mat: np.ndarray) -> np.ndarray:
    m = 0.5 * (
        np.asarray(mat, dtype=np.float64) + np.asarray(mat, dtype=np.float64).T
    )
    np.fill_diagonal(m, 0.0)
    return m


def build_deterministic_j0_bank(
    n_spins: int, theta_a: float
) -> Dict[str, np.ndarray]:
    j0_uniform = np.full((n_spins, n_spins), 1.0, dtype=np.float64)
    np.fill_diagonal(j0_uniform, 0.0)

    j0_alternating = np.zeros((n_spins, n_spins), dtype=np.float64)
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            sign = 1.0 if ((i + j) % 2 == 1) else -1.0
            val = sign * (0.7 + 0.25 * abs(i - j))
            j0_alternating[i, j] = val
            j0_alternating[j, i] = val

    j0_graded = np.zeros((n_spins, n_spins), dtype=np.float64)
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            val = (1.2 / (1.0 + 0.35 * (i + j))) * (-1.0 if i == 1 else 1.0)
            j0_graded[i, j] = val
            j0_graded[j, i] = val

    raw_bank = {
        "J0_uniform_pos": make_symmetric_zero_diag(j0_uniform),
        "J0_alternating_signed": make_symmetric_zero_diag(j0_alternating),
        "J0_graded_hub": make_symmetric_zero_diag(j0_graded),
    }

    calibrated: Dict[str, np.ndarray] = {}
    target_min_acc = 1.4 * theta_a
    for name, mat in raw_bank.items():
        min_acc = float(np.min(np.sum(mat ** 2, axis=1)))
        scale = np.sqrt(target_min_acc / max(min_acc, 1e-12))
        calibrated[name] = make_symmetric_zero_diag(mat * scale)

    return calibrated


def upper_tri_vector(J: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(J.shape[0], k=1)
    return np.asarray(J[idx], dtype=np.float64)


def psi_symmetric_post_flip(X_after: np.ndarray, J: np.ndarray) -> np.ndarray:
    v = J @ X_after
    psi = 0.5 * (
        (v[:, None] + v[None, :]) - J * (X_after[:, None] + X_after[None, :])
    )
    np.fill_diagonal(psi, 0.0)
    return psi


def psi_integrable_potential(
    X_before: np.ndarray, X_after: np.ndarray
) -> np.ndarray:
    psi = -0.5 * (np.outer(X_after, X_after) - np.outer(X_before, X_before))
    np.fill_diagonal(psi, 0.0)
    return psi


def step_flip_node(
    X: np.ndarray,
    J: np.ndarray,
    node_k: int,
    eta: float,
    mode: str = "symmetric",
) -> Tuple[np.ndarray, np.ndarray]:
    X_next = X.copy()
    X_next[node_k] = -X_next[node_k]
    J_next = J.copy()
    if mode == "symmetric":
        psi = psi_symmetric_post_flip(X_next, J_next)
        row = eta * psi[node_k]
        row[node_k] = 0.0
        J_next[node_k, :] += row
        J_next[:, node_k] += row
    elif mode == "integrable":
        psi_int = psi_integrable_potential(X, X_next)
        row = eta * psi_int[node_k]
        row[node_k] = 0.0
        J_next[node_k, :] += row
        J_next[:, node_k] += row
    else:
        raise ValueError(f"Mode inconnu : {mode}")
    np.fill_diagonal(J_next, 0.0)
    return X_next, J_next


def relax_fiber(
    J_t: np.ndarray, J0: np.ndarray, gamma: float, dt: float
) -> np.ndarray:
    out = J0 + (J_t - J0) * float(np.exp(-gamma * dt))
    np.fill_diagonal(out, 0.0)
    return out


def compute_instantaneous_state(
    X: np.ndarray,
    J: np.ndarray,
    nk: NKLandscape,
    cfg: Experiment04cConfig,
) -> Dict[str, np.ndarray | float]:
    code = spins_to_state_code(X)
    A = np.sum(J * J, axis=1)
    mask = A > cfg.theta_a
    f_val = float(nk._fitness_cache[code])
    df_nk_vec = nk._df_cache[code]

    field_h = J @ X
    delta_u = -2.0 * X * field_h
    delta_f = df_nk_vec + cfg.coupled_lambda * delta_u

    acc_prob = np.minimum(1.0, np.exp(np.clip(cfg.beta * delta_f, -60.0, 60.0)))
    rates = np.where(mask, cfg.mu * acc_prob, 0.0)

    total_rate = float(np.sum(rates))
    probs = (
        rates / total_rate
        if total_rate > 0.0
        else np.zeros_like(rates, dtype=np.float64)
    )
    return {
        "fitness": f_val,
        "df_nk": df_nk_vec,
        "J_vec": upper_tri_vector(J),
        "A_vec": A,
        "rates": rates,
        "total_rate": total_rate,
        "probs": probs,
    }


# ============================================================================
# 3. Génération par amincissement d'Ogata et banc apparié
# ============================================================================


def simulate_free_trajectory(
    nk: NKLandscape,
    J0_fixed: np.ndarray,
    cfg: Experiment04cConfig,
    rng: np.random.Generator,
    mode: str,
) -> Optional[Dict[str, np.ndarray]]:
    n = cfg.n_spins
    T = cfg.horizon_steps
    lambda_max = float(n * cfg.mu)

    X_curr = rng.choice(np.array([-1.0, 1.0]), size=n)
    J_curr = J0_fixed.copy()

    seq_X = np.zeros((T, n), dtype=np.float32)
    seq_dt = np.zeros((T, 1), dtype=np.float32)
    seq_nk_feat = np.zeros((T, n + 1), dtype=np.float32)
    seq_J = np.zeros((T, n * (n - 1) // 2), dtype=np.float32)
    seq_A = np.zeros((T, n), dtype=np.float32)
    seq_w = np.zeros((T, n), dtype=np.float32)

    dt_since_last_event = 0.0

    for step in range(T):
        st = compute_instantaneous_state(X_curr, J_curr, nk, cfg)
        seq_X[step] = X_curr.astype(np.float32)
        seq_dt[step, 0] = np.float32(dt_since_last_event)
        seq_nk_feat[step, 0] = np.float32(st["fitness"])
        seq_nk_feat[step, 1:] = np.asarray(st["df_nk"], dtype=np.float32)
        seq_J[step] = np.asarray(st["J_vec"], dtype=np.float32)
        seq_A[step] = np.asarray(st["A_vec"], dtype=np.float32)
        seq_w[step] = np.asarray(st["rates"], dtype=np.float32)

        if step == T - 1:
            break

        acc_dt = 0.0
        accepted = False
        for _ in range(cfg.max_proposals_per_step):
            dt_prop = float(rng.exponential(1.0 / lambda_max))
            acc_dt += dt_prop
            J_curr = relax_fiber(J_curr, J0_fixed, gamma=cfg.gamma, dt=dt_prop)

            st_prop = compute_instantaneous_state(X_curr, J_curr, nk, cfg)
            w_tot = float(st_prop["total_rate"])
            if w_tot > 0.0 and rng.uniform(0.0, 1.0) < (w_tot / lambda_max):
                probs = np.asarray(st_prop["probs"], dtype=np.float64)
                chosen_k = int(rng.choice(n, p=probs))
                X_curr, J_curr = step_flip_node(
                    X_curr, J_curr, node_k=chosen_k, eta=cfg.eta, mode=mode
                )
                dt_since_last_event = acc_dt
                accepted = True
                break

            if w_tot == 0.0 and np.max(np.abs(J_curr - J0_fixed)) < 1e-6:
                return None

        if not accepted:
            return None

    return {
        "X": seq_X,
        "dt": seq_dt,
        "nk_feat": seq_nk_feat,
        "J": seq_J,
        "A": seq_A,
        "w": seq_w,
    }


def build_free_dataset(
    nk_seeds: Sequence[int],
    J0_fixed: np.ndarray,
    n_trajectories: int,
    cfg: Experiment04cConfig,
    rng: np.random.Generator,
    mode: str,
) -> Dict[str, np.ndarray]:
    landscapes = [
        NKLandscape.generate(
            n_spins=cfg.n_spins,
            k_epistasis=cfg.k_epistasis,
            seed=int(sd),
            neighbor_mode=cfg.neighbor_mode,
        )
        for sd in nk_seeds
    ]
    trajs: List[Dict[str, np.ndarray]] = []
    attempts = 0
    max_attempts = n_trajectories * 25
    while len(trajs) < n_trajectories and attempts < max_attempts:
        nk = landscapes[attempts % len(landscapes)]
        attempts += 1
        tr = simulate_free_trajectory(nk, J0_fixed, cfg, rng, mode=mode)
        if tr is not None:
            trajs.append(tr)

    if len(trajs) < n_trajectories:
        raise RuntimeError(
            f"Seulement {len(trajs)}/{n_trajectories} trajectoires non absorbantes "
            f"générées en mode {mode}."
        )

    return {
        key: np.stack([tr[key] for tr in trajs], axis=0)
        for key in ("X", "dt", "nk_feat", "J", "A", "w")
    }


def build_paired_test_bank(
    nk_seeds: Sequence[int],
    J0_fixed: np.ndarray,
    n_pairs: int,
    cfg: Experiment04cConfig,
    rng: np.random.Generator,
    mode: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    n = cfg.n_spins
    suffix_len = len(cfg.paired_seq_A)
    prefix_len = max(1, cfg.horizon_steps - suffix_len)
    T = prefix_len + suffix_len

    landscapes = [
        NKLandscape.generate(
            n_spins=n,
            k_epistasis=cfg.k_epistasis,
            seed=int(sd),
            neighbor_mode=cfg.neighbor_mode,
        )
        for sd in nk_seeds
    ]

    data_A: Dict[str, List[np.ndarray]] = {
        k: [] for k in ("X", "dt", "nk_feat", "J", "A", "w")
    }
    data_B: Dict[str, List[np.ndarray]] = {
        k: [] for k in ("X", "dt", "nk_feat", "J", "A", "w")
    }

    for idx in range(n_pairs):
        nk = landscapes[idx % len(landscapes)]
        X0 = rng.choice(np.array([-1.0, 1.0]), size=n)

        prefix_flips = [int(rng.integers(0, n)) for _ in range(prefix_len - 1)]
        prefix_dts = [
            float(rng.uniform(0.05, 0.4)) for _ in range(prefix_len - 1)
        ]
        suffix_dts = [float(rng.uniform(0.05, 0.4)) for _ in range(suffix_len)]

        flips_A = prefix_flips + list(cfg.paired_seq_A)
        flips_B = prefix_flips + list(cfg.paired_seq_B)
        dts_all = prefix_dts + suffix_dts

        branches = []
        for flips in (flips_A, flips_B):
            X_c = X0.copy()
            J_c = J0_fixed.copy()
            seq_X = np.zeros((T, n), dtype=np.float32)
            seq_dt = np.zeros((T, 1), dtype=np.float32)
            seq_nk = np.zeros((T, n + 1), dtype=np.float32)
            seq_J = np.zeros((T, n * (n - 1) // 2), dtype=np.float32)
            seq_A = np.zeros((T, n), dtype=np.float32)
            seq_w = np.zeros((T, n), dtype=np.float32)

            dt_prev = 0.0
            for step in range(T):
                st = compute_instantaneous_state(X_c, J_c, nk, cfg)
                seq_X[step] = X_c.astype(np.float32)
                seq_dt[step, 0] = np.float32(dt_prev)
                seq_nk[step, 0] = np.float32(st["fitness"])
                seq_nk[step, 1:] = np.asarray(st["df_nk"], dtype=np.float32)
                seq_J[step] = np.asarray(st["J_vec"], dtype=np.float32)
                seq_A[step] = np.asarray(st["A_vec"], dtype=np.float32)
                seq_w[step] = np.asarray(st["rates"], dtype=np.float32)

                if step < T - 1:
                    dt_step = dts_all[step]
                    J_rel = relax_fiber(J_c, J0_fixed, gamma=cfg.gamma, dt=dt_step)
                    X_c, J_c = step_flip_node(
                        X_c,
                        J_rel,
                        node_k=flips[step],
                        eta=cfg.eta,
                        mode=mode,
                    )
                    dt_prev = dt_step

            branches.append(
                {
                    "X": seq_X,
                    "dt": seq_dt,
                    "nk_feat": seq_nk,
                    "J": seq_J,
                    "A": seq_A,
                    "w": seq_w,
                }
            )

        assert np.array_equal(branches[0]["X"][-1], branches[1]["X"][-1]), (
            f"Paire {idx} : X_T^A != X_T^B"
        )
        for key in data_A:
            data_A[key].append(branches[0][key])
            data_B[key].append(branches[1][key])

    return {
        "A": {k: np.stack(v, axis=0) for k, v in data_A.items()},
        "B": {k: np.stack(v, axis=0) for k, v in data_B.items()},
    }


# ============================================================================
# 4. Modèles M1-M4 : SeqLSTM avec skip-connection directe pour nk_feat_t
# ============================================================================


class StepMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SkipNKSeqLSTM(nn.Module):
    """
    La récurrence intègre uniquement l'histoire observable de la trajectoire
    (X_{0:t} ou (X_{0:t}, dt_{0:t})). Les features locales instantanées
    nk_feat_t sont injectées directement (skip-connection) dans la tête de
    sortie au pas t, empêchant le LSTM de sur-apprendre l'historique des
    paysages NK d'entraînement.
    """

    def __init__(
        self, seq_dim: int, nk_dim: int, hidden_dim: int, out_dim: int
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + nk_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self, x_seq: torch.Tensor, nk_inst: torch.Tensor
    ) -> torch.Tensor:
        h, _ = self.lstm(x_seq)
        fused = torch.cat([h, nk_inst], dim=-1)
        return self.head(fused)


def build_window_with_current_nk(
    X: np.ndarray, nk_feat: np.ndarray, window_len: int
) -> np.ndarray:
    B, T, n = X.shape
    padded = np.concatenate(
        [np.zeros((B, window_len - 1, n), dtype=np.float32), X], axis=1
    )
    win_x = np.zeros((B, T, window_len * n), dtype=np.float32)
    for t in range(T):
        win_x[:, t, :] = padded[:, t : t + window_len, :].reshape(
            B, window_len * n
        )
    return np.concatenate([win_x, nk_feat], axis=-1)


def prepare_mlp_inputs(
    dataset: Dict[str, np.ndarray],
    model_name: str,
    window_len: int = 1,
) -> np.ndarray:
    X = dataset["X"]
    nk_feat = dataset["nk_feat"]
    if model_name == "M1":
        return X
    if model_name == "M2":
        return np.concatenate([X, nk_feat], axis=-1)
    if model_name == "M3_X_NK":
        return build_window_with_current_nk(X, nk_feat, window_len=window_len)
    raise ValueError(f"Modèle MLP inconnu : {model_name}")


def prepare_lstm_inputs(
    dataset: Dict[str, np.ndarray], model_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    if model_name == "M4_skipNK_X":
        return dataset["X"], dataset["nk_feat"]
    if model_name == "M4b_skipNK_X_dt":
        seq = np.concatenate([dataset["X"], dataset["dt"]], axis=-1)
        return seq, dataset["nk_feat"]
    raise ValueError(f"Modèle LSTM inconnu : {model_name}")


def train_and_predict(
    model_name: str,
    train_ds: Dict[str, np.ndarray],
    test_free_ds: Dict[str, np.ndarray],
    paired_ds: Dict[str, Dict[str, np.ndarray]],
    target_key: str,
    cfg: Experiment04cConfig,
    device: torch.device,
    window_len: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_tr = train_ds[target_key]
    out_dim = y_tr.shape[-1]
    loss_fn = nn.MSELoss()

    if model_name in ("M4_skipNK_X", "M4b_skipNK_X_dt"):
        seq_tr, nk_tr = prepare_lstm_inputs(train_ds, model_name)
        seq_te, nk_te = prepare_lstm_inputs(test_free_ds, model_name)
        seq_pa_A, nk_pa_A = prepare_lstm_inputs(paired_ds["A"], model_name)
        seq_pa_B, nk_pa_B = prepare_lstm_inputs(paired_ds["B"], model_name)

        net = SkipNKSeqLSTM(
            seq_dim=seq_tr.shape[-1],
            nk_dim=nk_tr.shape[-1],
            hidden_dim=cfg.hidden_dim,
            out_dim=out_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(seq_tr),
                torch.from_numpy(nk_tr),
                torch.from_numpy(y_tr),
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
        )

        net.train()
        for _ in range(cfg.epochs):
            for b_seq, b_nk, by in loader:
                b_seq = b_seq.to(device)
                b_nk = b_nk.to(device)
                by = by.to(device)
                optimizer.zero_grad()
                pred = net(b_seq, b_nk)
                loss = loss_fn(pred, by)
                loss.backward()
                optimizer.step()

        net.eval()
        with torch.no_grad():
            pred_free = (
                net(
                    torch.from_numpy(seq_te).to(device),
                    torch.from_numpy(nk_te).to(device),
                )
                .cpu()
                .numpy()
            )
            pred_pa_A = (
                net(
                    torch.from_numpy(seq_pa_A).to(device),
                    torch.from_numpy(nk_pa_A).to(device),
                )
                .cpu()
                .numpy()
            )
            pred_pa_B = (
                net(
                    torch.from_numpy(seq_pa_B).to(device),
                    torch.from_numpy(nk_pa_B).to(device),
                )
                .cpu()
                .numpy()
            )
        return pred_free, pred_pa_A, pred_pa_B

    x_tr = prepare_mlp_inputs(train_ds, model_name, window_len=window_len)
    x_te = prepare_mlp_inputs(test_free_ds, model_name, window_len=window_len)
    x_pa_A = prepare_mlp_inputs(paired_ds["A"], model_name, window_len=window_len)
    x_pa_B = prepare_mlp_inputs(paired_ds["B"], model_name, window_len=window_len)

    mlp = StepMLP(x_tr.shape[-1], cfg.hidden_dim, out_dim).to(device)
    optimizer = torch.optim.AdamW(
        mlp.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    mlp.train()
    for _ in range(cfg.epochs):
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            pred = mlp(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()

    mlp.eval()
    with torch.no_grad():
        pred_free = mlp(torch.from_numpy(x_te).to(device)).cpu().numpy()
        pred_pa_A = mlp(torch.from_numpy(x_pa_A).to(device)).cpu().numpy()
        pred_pa_B = mlp(torch.from_numpy(x_pa_B).to(device)).cpu().numpy()

    return pred_free, pred_pa_A, pred_pa_B


# ============================================================================
# 5. Métriques et orchestration
# ============================================================================


def r2_score_flat(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.reshape(-1, y_true.shape[-1])
    yp = y_pred.reshape(-1, y_pred.shape[-1])
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt, axis=0, keepdims=True)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def evaluate_predictions(
    target_key: str,
    y_free_true: np.ndarray,
    y_free_pred: np.ndarray,
    y_pa_A_true: np.ndarray,
    y_pa_B_true: np.ndarray,
    y_pa_A_pred: np.ndarray,
    y_pa_B_pred: np.ndarray,
    theta_a: float,
) -> Dict[str, float]:
    free_mse = float(np.mean((y_free_true - y_free_pred) ** 2))
    free_r2 = r2_score_flat(y_free_true, y_free_pred)

    yt_A = y_pa_A_true[:, -1, :]
    yt_B = y_pa_B_true[:, -1, :]
    yp_A = y_pa_A_pred[:, -1, :]
    yp_B = y_pa_B_pred[:, -1, :]

    paired_mse = 0.5 * float(
        np.mean((yt_A - yp_A) ** 2) + np.mean((yt_B - yp_B) ** 2)
    )

    true_diff = yt_A - yt_B
    pred_diff = yp_A - yp_B
    true_diff_norm = float(np.mean(np.linalg.norm(true_diff, axis=-1)))
    pred_diff_norm = float(np.mean(np.linalg.norm(pred_diff, axis=-1)))
    diff_mse = float(np.mean((true_diff - pred_diff) ** 2))
    diff_r2 = r2_score_flat(true_diff[:, None, :], pred_diff[:, None, :])

    num = np.sum(true_diff * pred_diff, axis=-1)
    den = np.linalg.norm(true_diff, axis=-1) * np.linalg.norm(pred_diff, axis=-1)
    valid = den > 1e-12
    diff_cosine = float(np.mean(num[valid] / den[valid])) if np.any(valid) else 0.0

    metrics: Dict[str, float] = {
        "free_mse": free_mse,
        "free_r2": free_r2,
        "paired_final_mse": paired_mse,
        "paired_true_diff_norm": true_diff_norm,
        "paired_pred_diff_norm": pred_diff_norm,
        "paired_diff_mse": diff_mse,
        "paired_diff_r2": diff_r2,
        "paired_diff_cosine": diff_cosine,
    }

    if target_key == "A":
        metrics["free_access_mask_acc"] = float(
            np.mean((y_free_true > theta_a) == (y_free_pred > theta_a))
        )
        metrics["paired_access_mask_acc"] = 0.5 * float(
            np.mean((yt_A > theta_a) == (yp_A > theta_a))
            + np.mean((yt_B > theta_a) == (yp_B > theta_a))
        )

    if target_key == "w":
        w_true_pos = np.clip(y_free_true, 0.0, None)
        w_pred_pos = np.clip(y_free_pred, 0.0, None)
        s_t = np.sum(w_true_pos, axis=-1, keepdims=True)
        s_p = np.sum(w_pred_pos, axis=-1, keepdims=True)
        p_t = np.where(s_t > 1e-12, w_true_pos / np.maximum(s_t, 1e-12), 0.0)
        p_p = np.where(s_p > 1e-12, w_pred_pos / np.maximum(s_p, 1e-12), 0.0)
        metrics["free_jump_prob_tv"] = 0.5 * float(
            np.mean(np.sum(np.abs(p_t - p_p), axis=-1))
        )

    return metrics


def run_for_j0_and_mode(
    j0_name: str,
    J0_fixed: np.ndarray,
    mode: str,
    cfg: Experiment04cConfig,
    device: torch.device,
    seed_offset: int,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.global_seed + seed_offset)

    train_ds = build_free_dataset(
        cfg.train_nk_seeds,
        J0_fixed,
        cfg.n_train_trajectories,
        cfg,
        rng,
        mode=mode,
    )
    test_free_ds = build_free_dataset(
        cfg.test_nk_seeds,
        J0_fixed,
        cfg.n_test_free_trajectories,
        cfg,
        rng,
        mode=mode,
    )
    paired_ds = build_paired_test_bank(
        cfg.test_nk_seeds,
        J0_fixed,
        cfg.n_test_paired_trials,
        cfg,
        rng,
        mode=mode,
    )
    print(
        f"[{j0_name} | {mode}] Données générées en {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    model_specs: List[Tuple[str, str, int]] = [
        ("M1_instant_X", "M1", 1),
        ("M2_instant_X_NK", "M2", 1),
    ]
    for w_len in cfg.m3_windows:
        model_specs.append((f"M3_window_X_NK_L{w_len}", "M3_X_NK", int(w_len)))
    model_specs.append(("M4_skipNK_LSTM_X", "M4_skipNK_X", 1))
    model_specs.append(("M4b_skipNK_LSTM_X_dt", "M4b_skipNK_X_dt", 1))

    results_by_target: Dict[str, Dict[str, Dict[str, float]]] = {}

    for target_key in ("J", "A", "w"):
        t_tgt = time.perf_counter()
        results_by_target[target_key] = {}
        for label, m_type, w_len in model_specs:
            torch.manual_seed(cfg.global_seed)
            np.random.seed(cfg.global_seed)
            pred_free, pred_pa_A, pred_pa_B = train_and_predict(
                model_name=m_type,
                train_ds=train_ds,
                test_free_ds=test_free_ds,
                paired_ds=paired_ds,
                target_key=target_key,
                cfg=cfg,
                device=device,
                window_len=w_len,
            )
            results_by_target[target_key][label] = evaluate_predictions(
                target_key=target_key,
                y_free_true=test_free_ds[target_key],
                y_free_pred=pred_free,
                y_pa_A_true=paired_ds["A"][target_key],
                y_pa_B_true=paired_ds["B"][target_key],
                y_pa_A_pred=pred_pa_A,
                y_pa_B_pred=pred_pa_B,
                theta_a=cfg.theta_a,
            )
        print(
            f"[{j0_name} | {mode}] Cible {target_key} terminée en "
            f"{time.perf_counter() - t_tgt:.1f}s",
            flush=True,
        )

    equiv_check: Dict[str, object] = {}
    if mode == "integrable":
        for target_key in ("J", "A", "w"):
            m2_mse = results_by_target[target_key]["M2_instant_X_NK"]["free_mse"]
            m4_mse = results_by_target[target_key]["M4_skipNK_LSTM_X"]["free_mse"]
            gap = abs(m2_mse - m4_mse)
            equiv_check[target_key] = {
                "M2_free_mse": m2_mse,
                "M4_skipNK_free_mse": m4_mse,
                "abs_gap": gap,
                "within_tolerance": bool(gap <= cfg.integrable_equiv_tol),
            }

    return {
        "j0_name": j0_name,
        "mode": mode,
        "mean_inter_event_dt_train": float(np.mean(train_ds["dt"][:, 1:, 0])),
        "targets": results_by_target,
        "integrable_equivalence": equiv_check if mode == "integrable" else None,
    }


def run_experiment_04c(
    cfg: Optional[Experiment04cConfig] = None,
) -> Dict[str, object]:
    if cfg is None:
        cfg = Experiment04cConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Démarrage Experiment 04c (Skip-NK LSTM) sur {device}...", flush=True)
    j0_bank = build_deterministic_j0_bank(cfg.n_spins, cfg.theta_a)

    runs_by_j0: Dict[str, Dict[str, object]] = {}
    for idx, (j0_name, J0_mat) in enumerate(j0_bank.items()):
        sym_res = run_for_j0_and_mode(
            j0_name=j0_name,
            J0_fixed=J0_mat,
            mode="symmetric",
            cfg=cfg,
            device=device,
            seed_offset=100 * (idx + 1) + 11,
        )
        int_res = run_for_j0_and_mode(
            j0_name=j0_name,
            J0_fixed=J0_mat,
            mode="integrable",
            cfg=cfg,
            device=device,
            seed_offset=100 * (idx + 1) + 29,
        )
        runs_by_j0[j0_name] = {
            "J0_upper_tri": upper_tri_vector(J0_mat).tolist(),
            "symmetric": sym_res,
            "integrable": int_res,
        }

    summary_across_j0: Dict[str, Dict[str, float]] = {}
    for target_key in ("J", "A", "w"):
        m2_sym_r2 = np.mean(
            [
                runs_by_j0[jn]["symmetric"]["targets"][target_key]["M2_instant_X_NK"]["free_r2"]
                for jn in j0_bank
            ]
        )
        m4_sym_r2 = np.mean(
            [
                runs_by_j0[jn]["symmetric"]["targets"][target_key]["M4_skipNK_LSTM_X"]["free_r2"]
                for jn in j0_bank
            ]
        )
        m4b_sym_r2 = np.mean(
            [
                runs_by_j0[jn]["symmetric"]["targets"][target_key]["M4b_skipNK_LSTM_X_dt"]["free_r2"]
                for jn in j0_bank
            ]
        )
        m4_sym_cos = np.mean(
            [
                runs_by_j0[jn]["symmetric"]["targets"][target_key]["M4_skipNK_LSTM_X"]["paired_diff_cosine"]
                for jn in j0_bank
            ]
        )
        m4b_sym_cos = np.mean(
            [
                runs_by_j0[jn]["symmetric"]["targets"][target_key]["M4b_skipNK_LSTM_X_dt"]["paired_diff_cosine"]
                for jn in j0_bank
            ]
        )
        m4_int_r2 = np.mean(
            [
                runs_by_j0[jn]["integrable"]["targets"][target_key]["M4_skipNK_LSTM_X"]["free_r2"]
                for jn in j0_bank
            ]
        )
        summary_across_j0[target_key] = {
            "mean_M2_sym_free_R2": float(m2_sym_r2),
            "mean_M4_skipNK_sym_free_R2": float(m4_sym_r2),
            "mean_M4b_skipNK_sym_free_R2": float(m4b_sym_r2),
            "mean_M4_skipNK_sym_paired_cosine": float(m4_sym_cos),
            "mean_M4b_skipNK_sym_paired_cosine": float(m4b_sym_cos),
            "mean_M4_skipNK_int_free_R2": float(m4_int_r2),
            "holonomy_R2_gain_M4_over_M2": float(m4_sym_r2 - m2_sym_r2),
            "sym_vs_int_R2_surplus_M4": float(m4_sym_r2 - m4_int_r2),
        }

    payload = {
        "protocol": "M1_M4_known_deterministic_J0_skipNK_LSTM",
        "device": str(device),
        "config": asdict(cfg),
        "summary_across_j0": summary_across_j0,
        "runs_by_j0": runs_by_j0,
    }

    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


if __name__ == "__main__":
    config = Experiment04cConfig()
    report = run_experiment_04c(config)

    print("\n================ SYNTHÈSE 04c (SKIP-NK LSTM) SUR LES 3 J0 ================")
    for t_key, s_vals in report["summary_across_j0"].items():
        print(f"\n--- Cible : {t_key} ---")
        for k, v in s_vals.items():
            print(f"  {k:<36} : {v:+.6f}")

    for j0_name, j0_block in report["runs_by_j0"].items():
        print(f"\n==================== FIBRE INITIALE : {j0_name} ====================")
        for mode_key in ("symmetric", "integrable"):
            print(f"\n  [Mode : {mode_key}]")
            targets = j0_block[mode_key]["targets"]
            for t_key, models_dict in targets.items():
                print(f"  --- Cible : {t_key} ---")
                for m_label, m_stats in models_dict.items():
                    line = (
                        f"    {m_label:<22} | free_MSE={m_stats['free_mse']:.5f} "
                        f"| free_R2={m_stats['free_r2']:+.4f} "
                        f"| pair_MSE={m_stats['paired_final_mse']:.5f} "
                        f"| pred_diff={m_stats['paired_pred_diff_norm']:.4f} "
                        f"(true={m_stats['paired_true_diff_norm']:.4f}, "
                        f"cos={m_stats['paired_diff_cosine']:+.3f})"
                    )
                    if "free_access_mask_acc" in m_stats:
                        line += f" | mask_acc={m_stats['free_access_mask_acc']:.3f}"
                    if "free_jump_prob_tv" in m_stats:
                        line += f" | jump_TV={m_stats['free_jump_prob_tv']:.4f}"
                    print(line)

    print(f"\nRésultats enregistrés dans : {config.out_json}")
