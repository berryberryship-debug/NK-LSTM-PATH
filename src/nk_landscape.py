"""
balayage_commutateur_nk_v2.py
Projet : NK-LSTM-PATH

COMPARAISON DU TRANSPORT INTÉGRABLE ET DU TRANSPORT SYMÉTRIQUE

Pour chaque état initial (X0, J0) et chaque paire non ordonnée (k, m),
compare les deux chemins : k -> m et m -> k.

Deux règles de transport sont évaluées :
  - integrable : Psi_ij(X') = s'_i s'_j pour i != j ;
  - symmetric  : opérateur natif psi_sym(X', J).

Pour chaque règle, le script mesure :
  - le commutateur final en norme de Frobenius ;
  - l'égalité des états X et des états complets (X, J) ;
  - la symétrie et la diagonale nulle sur les matrices brutes ;
  - la différence d'accessibilité A(J), en norme et par composante ;
  - la différence des taux w(X, J), en norme et par composante ;
  - les désaccords entre les masques d'accessibilité finaux.

Le paysage NK canonique et les états initiaux sont générés une seule fois
par exécution. Les deux transports utilisent les mêmes données.

Chaque paire non ordonnée est évaluée une seule fois : les deux ordres de
mutation sont déjà comparés à l'intérieur de chaque observation.

Exécution depuis la racine du dépôt :
    python .\balayage_commutateur_nk_v2.py

Exemples :
    python .\balayage_commutateur_nk_v2.py --eta 0.25 --lam 0.5
    python .\balayage_commutateur_nk_v2.py --json-out resultats_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import numpy as np


# ======================================================================
# 0. CHEMINS ET IMPORTS DES MODULES DU PROJET
# ======================================================================

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"

if not _SRC_DIR.is_dir():
    raise FileNotFoundError(
        f"Dossier source introuvable : {_SRC_DIR}"
    )

_src_text = str(_SRC_DIR)
if _src_text not in sys.path:
    sys.path.insert(0, _src_text)

from fiber_bundle import (  # noqa: E402
    _validate_J,
    _validate_state,
    accessibility_A,
    mutate_postflip,
    psi_sym,
)
from nk_landscape import NKLandscape  # noqa: E402
from trajectories import transition_rates  # noqa: E402


TransportMode = Literal["integrable", "symmetric"]


# ======================================================================
# 1. VALIDATION ET CONFIGURATION EXPÉRIMENTALE
# ======================================================================

def _strict_int(name: str, value: object) -> int:
    """Valide un entier sans accepter les booléens ni tronquer les flottants."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} doit être un entier.")
    return int(value)


def _finite_float(name: str, value: object) -> float:
    """Convertit une valeur en flottant et refuse NaN/inf."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} doit être numérique, pas booléen.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} doit être numérique.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} doit être fini.")
    return result


def _validate_site_index(name: str, value: object, n_sites: int) -> int:
    """Valide un indice de site sans conversion silencieuse."""
    index = _strict_int(name, value)
    if not 0 <= index < n_sites:
        raise IndexError(
            f"{name}={index} hors limites pour N={n_sites}."
        )
    return index


@dataclass(frozen=True)
class ExperimentConfig:
    n_sites: int = 6
    n_initial_states: int = 10
    init_seed: int = 7
    scale_J: float = 0.2

    eta: float = 0.25
    mu: float = 1.0
    beta: float = 1.0
    theta_A: float = 0.1
    lam: float = 0.5

    nk_K: int = 3
    nk_seed: int = 7
    neighbor_mode: str = "random"

    tol_w: float = 1e-12
    rtol: float = 1e-12
    atol: float = 1e-14


def validate_config(cfg: ExperimentConfig) -> None:
    """Valide tous les paramètres avant toute génération ou simulation."""
    n_sites = _strict_int("n_sites", cfg.n_sites)
    n_initial_states = _strict_int(
        "n_initial_states", cfg.n_initial_states
    )
    nk_K = _strict_int("nk_K", cfg.nk_K)
    init_seed = _strict_int("init_seed", cfg.init_seed)
    nk_seed = _strict_int("nk_seed", cfg.nk_seed)

    if n_sites < 2:
        raise ValueError("n_sites doit être >= 2.")
    if n_initial_states < 1:
        raise ValueError("n_initial_states doit être >= 1.")
    if not 0 <= nk_K < n_sites:
        raise ValueError("nk_K doit satisfaire 0 <= K < N.")
    if init_seed < 0 or nk_seed < 0:
        raise ValueError("Les graines doivent être positives ou nulles.")

    finite_values = {
        "scale_J": cfg.scale_J,
        "eta": cfg.eta,
        "mu": cfg.mu,
        "beta": cfg.beta,
        "theta_A": cfg.theta_A,
        "lam": cfg.lam,
        "tol_w": cfg.tol_w,
        "rtol": cfg.rtol,
        "atol": cfg.atol,
    }
    converted = {
        name: _finite_float(name, value)
        for name, value in finite_values.items()
    }

    if converted["scale_J"] < 0.0:
        raise ValueError("scale_J doit être >= 0.")
    if converted["mu"] < 0.0:
        raise ValueError("mu doit être >= 0.")
    for name in ("tol_w", "rtol", "atol"):
        if converted[name] < 0.0:
            raise ValueError(f"{name} doit être >= 0.")

    if cfg.neighbor_mode not in ("random", "adjacent"):
        raise ValueError(
            "neighbor_mode doit être 'random' ou 'adjacent'."
        )


# ======================================================================
# 2. OPÉRATEURS DE TRANSPORT
# ======================================================================

def psi_integrable(X: np.ndarray, J: np.ndarray) -> np.ndarray:
    r"""Contrôle intégrable : Psi_ij = s_i s_j pour i != j, Psi_ii = 0."""
    state = _validate_state(X)
    mat = _validate_J(J)
    if mat.shape != (state.size, state.size):
        raise ValueError("Dimensions incompatibles entre X et J.")

    psi = np.outer(state, state).astype(np.float64, copy=False)
    np.fill_diagonal(psi, 0.0)
    return psi


def step_flip_node(
    X: np.ndarray,
    J: np.ndarray,
    site: int,
    eta: float,
    mode: TransportMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Applique un flip avec l'un des deux opérateurs de transport."""
    state = _validate_state(X)
    mat = _validate_J(J)
    if mat.shape != (state.size, state.size):
        raise ValueError("Dimensions incompatibles entre X et J.")

    site_idx = _validate_site_index("site", site, state.size)
    eta_value = _finite_float("eta", eta)

    if mode == "integrable":
        psi_fn = psi_integrable
    elif mode == "symmetric":
        psi_fn = psi_sym
    else:
        raise ValueError(f"Mode de transport inconnu : {mode!r}.")

    X_next, J_next = mutate_postflip(
        state,
        mat,
        k=site_idx,
        eta=eta_value,
        psi_fn=psi_fn,
    )

    return np.asarray(X_next, dtype=np.float64), np.asarray(
        J_next, dtype=np.float64
    )


# ======================================================================
# 3. GÉNÉRATION DES ÉTATS INITIAUX ET PAIRES DE MUTATIONS
# ======================================================================

def sample_binary_symmetric_states(
    n_sites: int,
    n_states: int,
    rng: np.random.Generator,
    scale: float = 0.2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Génère des X binaires et des J symétriques à diagonale nulle."""
    n_sites = _strict_int("n_sites", n_sites)
    n_states = _strict_int("n_states", n_states)
    scale_value = _finite_float("scale", scale)

    if n_sites < 2 or n_states < 1:
        raise ValueError("Il faut n_sites >= 2 et n_states >= 1.")
    if scale_value < 0.0:
        raise ValueError("scale doit être >= 0.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng doit être un numpy.random.Generator.")

    states: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_states):
        X0 = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64),
            size=n_sites,
        )
        raw_J = rng.normal(
            0.0,
            scale_value,
            size=(n_sites, n_sites),
        )
        J0 = 0.5 * (raw_J + raw_J.T)
        np.fill_diagonal(J0, 0.0)
        states.append((X0, J0))

    return states


def iter_site_pairs(n_sites: int) -> Iterable[tuple[int, int]]:
    """Génère une seule fois chaque paire non ordonnée de sites distincts."""
    n_sites = _strict_int("n_sites", n_sites)
    if n_sites < 2:
        raise ValueError("n_sites doit être >= 2.")

    for k in range(n_sites):
        for m in range(k + 1, n_sites):
            yield k, m


# ======================================================================
# 4. COMPARAISON DES DEUX ORDRES POUR UN TRANSPORT DONNÉ
# ======================================================================

@dataclass(frozen=True)
class PathPairResult:
    XA: np.ndarray
    XB: np.ndarray
    JA: np.ndarray
    JB: np.ndarray
    same_X_endpoint: bool
    same_full_endpoint_exact: bool
    same_full_endpoint_close: bool
    symmetry_error_A: float
    symmetry_error_B: float
    diagonal_error_A: float
    diagonal_error_B: float
    commutator_fro: float


def _raw_matrix_for_diagnostics(
    J: np.ndarray,
    expected_n: int,
    name: str,
) -> np.ndarray:
    """Vérifie forme et finitude sans symétriser ni vider la diagonale."""
    raw = np.asarray(J, dtype=np.float64)
    if raw.shape != (expected_n, expected_n):
        raise ValueError(
            f"{name} doit avoir la forme {(expected_n, expected_n)}, "
            f"reçu {raw.shape}."
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} contient des valeurs non finies.")
    return raw.copy()


def matrix_symmetry_error(J: np.ndarray) -> float:
    """Mesure l'erreur de symétrie sans normaliser la matrice."""
    raw = np.asarray(J, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("J doit être carrée.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("J contient des valeurs non finies.")
    return float(np.linalg.norm(raw - raw.T, ord="fro"))


def matrix_diagonal_error(J: np.ndarray) -> float:
    """Mesure le maximum absolu de la diagonale, sans la modifier."""
    raw = np.asarray(J, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("J doit être carrée.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("J contient des valeurs non finies.")
    return float(np.max(np.abs(np.diag(raw))))


def compute_two_step_commutator(
    X0: np.ndarray,
    J0: np.ndarray,
    k: int,
    m: int,
    eta: float,
    mode: TransportMode,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> PathPairResult:
    """Compare k -> m et m -> k pour le mode de transport sélectionné."""
    X0v = _validate_state(X0)
    J0v = _validate_J(J0)

    if J0v.shape != (X0v.size, X0v.size):
        raise ValueError("Dimensions incompatibles entre X0 et J0.")

    k_idx = _validate_site_index("k", k, X0v.size)
    m_idx = _validate_site_index("m", m, X0v.size)
    if k_idx == m_idx:
        raise ValueError("Les sites k et m doivent être distincts.")

    eta_value = _finite_float("eta", eta)
    rtol_value = _finite_float("rtol", rtol)
    atol_value = _finite_float("atol", atol)
    if rtol_value < 0.0 or atol_value < 0.0:
        raise ValueError("rtol et atol doivent être >= 0.")

    # Chemin A : k puis m.
    XA, JA_raw = step_flip_node(
        X0v.copy(), J0v.copy(), k_idx, eta_value, mode
    )
    XA, JA_raw = step_flip_node(
        XA, JA_raw, m_idx, eta_value, mode
    )

    # Chemin B : m puis k.
    XB, JB_raw = step_flip_node(
        X0v.copy(), J0v.copy(), m_idx, eta_value, mode
    )
    XB, JB_raw = step_flip_node(
        XB, JB_raw, k_idx, eta_value, mode
    )

    XA = _validate_state(XA)
    XB = _validate_state(XB)
    JA = _raw_matrix_for_diagnostics(JA_raw, X0v.size, "JA")
    JB = _raw_matrix_for_diagnostics(JB_raw, X0v.size, "JB")

    # Les mesures sont faites sur les matrices brutes, AVANT _validate_J.
    symmetry_A = matrix_symmetry_error(JA)
    symmetry_B = matrix_symmetry_error(JB)
    diagonal_A = matrix_diagonal_error(JA)
    diagonal_B = matrix_diagonal_error(JB)

    same_X = bool(np.array_equal(XA, XB))
    same_exact = bool(same_X and np.array_equal(JA, JB))
    same_close = bool(
        same_X and np.allclose(JA, JB, rtol=rtol_value, atol=atol_value)
    )
    commutator = float(np.linalg.norm(JA - JB, ord="fro"))

    # Validation structurelle après les diagnostics : cette fonction peut
    # lever une erreur, elle n'est pas utilisée pour les mesures ci-dessus.
    _validate_J(JA)
    _validate_J(JB)

    return PathPairResult(
        XA=XA.copy(),
        XB=XB.copy(),
        JA=JA,
        JB=JB,
        same_X_endpoint=same_X,
        same_full_endpoint_exact=same_exact,
        same_full_endpoint_close=same_close,
        symmetry_error_A=symmetry_A,
        symmetry_error_B=symmetry_B,
        diagonal_error_A=diagonal_A,
        diagonal_error_B=diagonal_B,
        commutator_fro=commutator,
    )


# ======================================================================
# 5. OBSERVABLES : ACCESSIBILITÉ, TAUX ET MASQUES
# ======================================================================

@dataclass(frozen=True)
class ObservableGap:
    delta_A_norm: float
    delta_A_components: tuple[float, ...]
    delta_w_norm: float
    delta_w_max: float
    delta_w_components: tuple[float, ...]
    mask_disagreements: int


def evaluate_observable_gap(
    XA: np.ndarray,
    JA: np.ndarray,
    XB: np.ndarray,
    JB: np.ndarray,
    transition_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta_A: float,
) -> ObservableGap:
    """Compare A(J), w(X,J) et les masques finaux entre deux chemins."""
    theta = _finite_float("theta_A", theta_A)
    XA_v = _validate_state(XA)
    XB_v = _validate_state(XB)
    if XA_v.shape != XB_v.shape:
        raise ValueError("Les états X comparés n'ont pas la même dimension.")

    JA_v = _raw_matrix_for_diagnostics(JA, XA_v.size, "JA")
    JB_v = _raw_matrix_for_diagnostics(JB, XB_v.size, "JB")

    A_A = np.asarray(accessibility_A(JA_v), dtype=np.float64)
    A_B = np.asarray(accessibility_A(JB_v), dtype=np.float64)
    w_A = np.asarray(transition_fn(XA_v, JA_v), dtype=np.float64)
    w_B = np.asarray(transition_fn(XB_v, JB_v), dtype=np.float64)

    expected_shape = (XA_v.size,)
    for name, values in (
        ("A_A", A_A),
        ("A_B", A_B),
        ("w_A", w_A),
        ("w_B", w_B),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} doit avoir la forme {expected_shape}, "
                f"reçu {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contient des valeurs non finies.")

    if np.any(A_A < 0.0) or np.any(A_B < 0.0):
        raise ValueError("Les accessibilités doivent être non négatives.")
    if np.any(w_A < 0.0) or np.any(w_B < 0.0):
        raise ValueError("Les taux de transition doivent être non négatifs.")

    delta_A = A_A - A_B
    delta_w = w_A - w_B
    mask_A = A_A > theta
    mask_B = A_B > theta

    return ObservableGap(
        delta_A_norm=float(np.linalg.norm(delta_A)),
        delta_A_components=tuple(float(v) for v in delta_A),
        delta_w_norm=float(np.linalg.norm(delta_w)),
        delta_w_max=float(np.max(np.abs(delta_w))),
        delta_w_components=tuple(float(v) for v in delta_w),
        mask_disagreements=int(np.count_nonzero(mask_A != mask_B)),
    )


# ======================================================================
# 6. STRUCTURES DES RÉSULTATS
# ======================================================================

@dataclass(frozen=True)
class SweepSampleRecord:
    seed_idx: int
    k: int
    m: int

    same_X_endpoint_int: bool
    same_X_endpoint_sym: bool
    same_full_endpoint_exact_int: bool
    same_full_endpoint_exact_sym: bool
    same_full_endpoint_close_int: bool
    same_full_endpoint_close_sym: bool

    symmetry_error_max_int: float
    symmetry_error_max_sym: float
    diagonal_error_max_int: float
    diagonal_error_max_sym: float

    commutator_int_fro: float
    commutator_sym_fro: float

    delta_A_norm_int: float
    delta_A_norm_sym: float
    delta_A_components_int: tuple[float, ...]
    delta_A_components_sym: tuple[float, ...]

    delta_w_norm_int: float
    delta_w_norm_sym: float
    delta_w_max_int: float
    delta_w_max_sym: float
    delta_w_components_int: tuple[float, ...]
    delta_w_components_sym: tuple[float, ...]

    mask_disagreements_int: int
    mask_disagreements_sym: int


@dataclass(frozen=True)
class SweepSummary:
    n_samples: int

    mean_commutator_int_fro: float
    max_commutator_int_fro: float
    mean_commutator_sym_fro: float
    max_commutator_sym_fro: float

    max_symmetry_error_int: float
    max_symmetry_error_sym: float
    max_diagonal_error_int: float
    max_diagonal_error_sym: float

    frac_same_X_endpoint_int: float
    frac_same_X_endpoint_sym: float
    frac_same_full_endpoint_exact_int: float
    frac_same_full_endpoint_exact_sym: float
    frac_same_full_endpoint_close_int: float
    frac_same_full_endpoint_close_sym: float

    mean_delta_A_norm_int: float
    mean_delta_A_norm_sym: float
    max_delta_A_norm_int: float
    max_delta_A_norm_sym: float

    mean_delta_w_norm_int: float
    mean_delta_w_norm_sym: float
    max_delta_w_norm_int: float
    max_delta_w_norm_sym: float

    frac_nonzero_delta_w_int: float
    frac_nonzero_delta_w_sym: float

    total_mask_disagreements_int: int
    total_mask_disagreements_sym: int
    records: list[SweepSampleRecord]


# ======================================================================
# 7. BALAYAGE COMPARATIF
# ======================================================================

def sweep_commutator_and_dynamics(
    initial_states: Sequence[tuple[np.ndarray, np.ndarray]],
    cfg: ExperimentConfig,
    transition_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> SweepSummary:
    """Exécute les deux transports sur les mêmes états et paires de sites."""
    if len(initial_states) == 0:
        raise ValueError("initial_states ne peut pas être vide.")
    if not callable(transition_fn):
        raise TypeError("transition_fn doit être callable.")

    records: list[SweepSampleRecord] = []

    for seed_idx, (X0, J0) in enumerate(initial_states):
        X0v = _validate_state(X0)
        J0v = _validate_J(J0)
        if X0v.size != cfg.n_sites or J0v.shape != (
            cfg.n_sites,
            cfg.n_sites,
        ):
            raise ValueError(
                f"Dimensions initiales incompatibles à l'indice {seed_idx}."
            )

        for k, m in iter_site_pairs(cfg.n_sites):
            result_int = compute_two_step_commutator(
                X0v,
                J0v,
                k,
                m,
                cfg.eta,
                mode="integrable",
                rtol=cfg.rtol,
                atol=cfg.atol,
            )
            result_sym = compute_two_step_commutator(
                X0v,
                J0v,
                k,
                m,
                cfg.eta,
                mode="symmetric",
                rtol=cfg.rtol,
                atol=cfg.atol,
            )

            # Le transport intégrable est le contrôle de commutation.
            if (
                not result_int.same_X_endpoint
                or not result_int.same_full_endpoint_close
            ):
                raise RuntimeError(
                    "Le contrôle intégrable ne commute pas à la tolérance "
                    "demandée "
                    f"(état {seed_idx}, paire ({k}, {m}), "
                    f"écart={result_int.commutator_fro:.3e})."
                )

            gap_int = evaluate_observable_gap(
                result_int.XA,
                result_int.JA,
                result_int.XB,
                result_int.JB,
                transition_fn,
                cfg.theta_A,
            )
            gap_sym = evaluate_observable_gap(
                result_sym.XA,
                result_sym.JA,
                result_sym.XB,
                result_sym.JB,
                transition_fn,
                cfg.theta_A,
            )

            records.append(
                SweepSampleRecord(
                    seed_idx=seed_idx,
                    k=k,
                    m=m,
                    same_X_endpoint_int=result_int.same_X_endpoint,
                    same_X_endpoint_sym=result_sym.same_X_endpoint,
                    same_full_endpoint_exact_int=(
                        result_int.same_full_endpoint_exact
                    ),
                    same_full_endpoint_exact_sym=(
                        result_sym.same_full_endpoint_exact
                    ),
                    same_full_endpoint_close_int=(
                        result_int.same_full_endpoint_close
                    ),
                    same_full_endpoint_close_sym=(
                        result_sym.same_full_endpoint_close
                    ),
                    symmetry_error_max_int=max(
                        result_int.symmetry_error_A,
                        result_int.symmetry_error_B,
                    ),
                    symmetry_error_max_sym=max(
                        result_sym.symmetry_error_A,
                        result_sym.symmetry_error_B,
                    ),
                    diagonal_error_max_int=max(
                        result_int.diagonal_error_A,
                        result_int.diagonal_error_B,
                    ),
                    diagonal_error_max_sym=max(
                        result_sym.diagonal_error_A,
                        result_sym.diagonal_error_B,
                    ),
                    commutator_int_fro=result_int.commutator_fro,
                    commutator_sym_fro=result_sym.commutator_fro,
                    delta_A_norm_int=gap_int.delta_A_norm,
                    delta_A_norm_sym=gap_sym.delta_A_norm,
                    delta_A_components_int=gap_int.delta_A_components,
                    delta_A_components_sym=gap_sym.delta_A_components,
                    delta_w_norm_int=gap_int.delta_w_norm,
                    delta_w_norm_sym=gap_sym.delta_w_norm,
                    delta_w_max_int=gap_int.delta_w_max,
                    delta_w_max_sym=gap_sym.delta_w_max,
                    delta_w_components_int=gap_int.delta_w_components,
                    delta_w_components_sym=gap_sym.delta_w_components,
                    mask_disagreements_int=gap_int.mask_disagreements,
                    mask_disagreements_sym=gap_sym.mask_disagreements,
                )
            )

    if not records:
        raise RuntimeError("Le balayage n'a produit aucune comparaison.")

    def values(name: str) -> np.ndarray:
        return np.asarray(
            [getattr(record, name) for record in records],
            dtype=np.float64,
        )

    def mean(name: str) -> float:
        return float(np.mean(values(name)))

    def maximum(name: str) -> float:
        return float(np.max(values(name)))

    def fraction_nonzero(name: str) -> float:
        return float(np.mean(values(name) > cfg.tol_w))

    return SweepSummary(
        n_samples=len(records),
        mean_commutator_int_fro=mean("commutator_int_fro"),
        max_commutator_int_fro=maximum("commutator_int_fro"),
        mean_commutator_sym_fro=mean("commutator_sym_fro"),
        max_commutator_sym_fro=maximum("commutator_sym_fro"),
        max_symmetry_error_int=maximum("symmetry_error_max_int"),
        max_symmetry_error_sym=maximum("symmetry_error_max_sym"),
        max_diagonal_error_int=maximum("diagonal_error_max_int"),
        max_diagonal_error_sym=maximum("diagonal_error_max_sym"),
        frac_same_X_endpoint_int=float(
            np.mean(values("same_X_endpoint_int"))
        ),
        frac_same_X_endpoint_sym=float(
            np.mean(values("same_X_endpoint_sym"))
        ),
        frac_same_full_endpoint_exact_int=float(
            np.mean(values("same_full_endpoint_exact_int"))
        ),
        frac_same_full_endpoint_exact_sym=float(
            np.mean(values("same_full_endpoint_exact_sym"))
        ),
        frac_same_full_endpoint_close_int=float(
            np.mean(values("same_full_endpoint_close_int"))
        ),
        frac_same_full_endpoint_close_sym=float(
            np.mean(values("same_full_endpoint_close_sym"))
        ),
        mean_delta_A_norm_int=mean("delta_A_norm_int"),
        mean_delta_A_norm_sym=mean("delta_A_norm_sym"),
        max_delta_A_norm_int=maximum("delta_A_norm_int"),
        max_delta_A_norm_sym=maximum("delta_A_norm_sym"),
        mean_delta_w_norm_int=mean("delta_w_norm_int"),
        mean_delta_w_norm_sym=mean("delta_w_norm_sym"),
        max_delta_w_norm_int=maximum("delta_w_norm_int"),
        max_delta_w_norm_sym=maximum("delta_w_norm_sym"),
        frac_nonzero_delta_w_int=fraction_nonzero("delta_w_max_int"),
        frac_nonzero_delta_w_sym=fraction_nonzero("delta_w_max_sym"),
        total_mask_disagreements_int=int(
            np.sum(values("mask_disagreements_int"))
        ),
        total_mask_disagreements_sym=int(
            np.sum(values("mask_disagreements_sym"))
        ),
        records=records,
    )


# ======================================================================
# 8. FITNESS NK ET TAUX DE TRANSITION
# ======================================================================

def build_transition_function(
    cfg: ExperimentConfig,
    landscape: NKLandscape,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Construit les taux à partir du paysage NK canonique partagé."""
    if not isinstance(landscape, NKLandscape):
        raise TypeError(
            "landscape doit être une instance du NKLandscape canonique."
        )

    def compute_w(X: np.ndarray, J: np.ndarray) -> np.ndarray:
        return transition_rates(
            X=X,
            J=J,
            mu=cfg.mu,
            beta=cfg.beta,
            theta_A=cfg.theta_A,
            lam=cfg.lam,
            fnk_fn=landscape,
        )

    return compute_w


# ======================================================================
# 9. AFFICHAGE ET EXPORT
# ======================================================================

def print_summary(summary: SweepSummary, cfg: ExperimentConfig) -> None:
    total_site_comparisons = summary.n_samples * cfg.n_sites

    print("\n" + "=" * 76)
    print("NK-LSTM-PATH — CONTRÔLE INTÉGRABLE / TRANSPORT SYMÉTRIQUE")
    print("=" * 76)
    print(
        f"N={cfg.n_sites} | états initiaux={cfg.n_initial_states} | "
        f"comparaisons={summary.n_samples}"
    )
    print(
        f"eta={cfg.eta:g} | mu={cfg.mu:g} | beta={cfg.beta:g} | "
        f"theta_A={cfg.theta_A:g} | lambda={cfg.lam:g} | K={cfg.nk_K}"
    )
    print(
        f"Graine initiale={cfg.init_seed} | graine NK={cfg.nk_seed} | "
        f"mode voisinage={cfg.neighbor_mode} | échelle J0={cfg.scale_J:g}"
    )

    print("\nCONTRÔLE INTÉGRABLE")
    print(
        f"Commutateur moyen / max       : "
        f"{summary.mean_commutator_int_fro:.6e} / "
        f"{summary.max_commutator_int_fro:.6e}"
    )
    print(
        f"États complets identiques     : "
        f"{100.0 * summary.frac_same_full_endpoint_exact_int:.1f}% exacts"
    )
    print(
        f"États complets proches        : "
        f"{100.0 * summary.frac_same_full_endpoint_close_int:.1f}% à tolérance"
    )
    print(
        f"Erreur max de symétrie        : "
        f"{summary.max_symmetry_error_int:.6e}"
    )
    print(
        f"Erreur max de diagonale       : "
        f"{summary.max_diagonal_error_int:.6e}"
    )
    print(
        f"Écart moyen d'accessibilité   : "
        f"{summary.mean_delta_A_norm_int:.6e}"
    )
    print(
        f"Écart moyen des taux          : "
        f"{summary.mean_delta_w_norm_int:.6e}"
    )
    print(
        f"Comparaisons avec Δw non nul  : "
        f"{100.0 * summary.frac_nonzero_delta_w_int:.2f}%"
    )
    print(
        f"Désaccords de masques         : "
        f"{summary.total_mask_disagreements_int}/"
        f"{total_site_comparisons} "
        f"({100.0 * summary.total_mask_disagreements_int / total_site_comparisons:.2f}%)"
    )

    print("\nTRANSPORT SYMÉTRIQUE")
    print(
        f"Commutateur moyen / max       : "
        f"{summary.mean_commutator_sym_fro:.6e} / "
        f"{summary.max_commutator_sym_fro:.6e}"
    )
    print(
        f"États X identiques            : "
        f"{100.0 * summary.frac_same_X_endpoint_sym:.1f}%"
    )
    print(
        f"États complets identiques     : "
        f"{100.0 * summary.frac_same_full_endpoint_exact_sym:.1f}% exacts"
    )
    print(
        f"États complets proches        : "
        f"{100.0 * summary.frac_same_full_endpoint_close_sym:.1f}% à tolérance"
    )
    print(
        f"Erreur max de symétrie        : "
        f"{summary.max_symmetry_error_sym:.6e}"
    )
    print(
        f"Erreur max de diagonale       : "
        f"{summary.max_diagonal_error_sym:.6e}"
    )
    print(
        f"Écart moyen d'accessibilité   : "
        f"{summary.mean_delta_A_norm_sym:.6e}"
    )
    print(
        f"Écart maximal d'accessibilité : "
        f"{summary.max_delta_A_norm_sym:.6e}"
    )
    print(
        f"Écart moyen des taux          : "
        f"{summary.mean_delta_w_norm_sym:.6e}"
    )
    print(
        f"Écart maximal des taux        : "
        f"{summary.max_delta_w_norm_sym:.6e}"
    )
    print(
        f"Comparaisons avec Δw non nul  : "
        f"{100.0 * summary.frac_nonzero_delta_w_sym:.2f}%"
    )
    print(
        f"Désaccords de masques         : "
        f"{summary.total_mask_disagreements_sym}/"
        f"{total_site_comparisons} "
        f"({100.0 * summary.total_mask_disagreements_sym / total_site_comparisons:.2f}%)"
    )

    print("\nDéfinition de Δw non nul : max_k |w_A,k - w_B,k| > tol_w.")
    print(
        "Les vecteurs ΔA et Δw par site sont conservés dans chaque "
        "enregistrement JSON."
    )
    print(
        "Les désaccords comptent les sites dont les masques finaux "
        "diffèrent entre les deux ordres."
    )
    print("Ils ne représentent pas des franchissements temporels du seuil.")
    print("=" * 76)


def export_json(
    path: Path,
    cfg: ExperimentConfig,
    summary: SweepSummary,
) -> None:
    """Exporte paramètres, synthèse et enregistrements individuels en JSON."""
    payload = {
        "project": "NK-LSTM-PATH",
        "script": "balayage_commutateur_nk_v2.py",
        "nk_backend": "nk_landscape.NKLandscape.generate",
        "config": asdict(cfg),
        "summary": asdict(summary),
        "definitions": {
            "integrable_transport": (
                "Psi_ij(X') = X'_i*X'_j for i != j; Psi_ii = 0"
            ),
            "symmetric_transport": "fiber_bundle.psi_sym",
            "matrix_diagnostics": (
                "Symmetry and diagonal errors are measured on raw output "
                "matrices before _validate_J normalization."
            ),
            "nonzero_rate_difference": (
                "max_k(abs(w_A[k] - w_B[k])) > tol_w"
            ),
            "rate_and_accessibility_components": (
                "Per-site delta_A and delta_w vectors are stored per sample."
            ),
            "mask_disagreements": (
                "sitewise XOR of final masks A_k(J) > theta_A"
            ),
            "pair_counting": (
                "Each unordered pair is evaluated once; both mutation orders "
                "are compared within that evaluation."
            ),
        },
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    print(f"\nRésultats détaillés enregistrés : {path}")


# ======================================================================
# 10. INTERFACE DE LIGNE DE COMMANDE
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare le transport intégrable et le transport symétrique "
            "de NK-LSTM-PATH."
        )
    )
    parser.add_argument("--n-sites", type=int, default=6)
    parser.add_argument("--n-initial-states", type=int, default=10)
    parser.add_argument("--init-seed", type=int, default=7)
    parser.add_argument("--scale-j", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--theta-a", type=float, default=0.1)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--nk-k", type=int, default=3)
    parser.add_argument("--nk-seed", type=int, default=7)
    parser.add_argument(
        "--neighbor-mode",
        choices=("random", "adjacent"),
        default="random",
    )
    parser.add_argument("--tol-w", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-14)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Chemin facultatif d'export JSON.",
    )
    return parser.parse_args()


def main() -> SweepSummary:
    args = parse_args()
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
        neighbor_mode=args.neighbor_mode,
        tol_w=args.tol_w,
        rtol=args.rtol,
        atol=args.atol,
    )
    validate_config(cfg)

    rng = np.random.default_rng(cfg.init_seed)
    initial_states = sample_binary_symmetric_states(
        n_sites=cfg.n_sites,
        n_states=cfg.n_initial_states,
        rng=rng,
        scale=cfg.scale_J,
    )

    # Paysage NK canonique : aucune génération de secours silencieuse.
    landscape = NKLandscape.generate(
        n_spins=cfg.n_sites,
        k_epistasis=cfg.nk_K,
        seed=cfg.nk_seed,
        neighbor_mode=cfg.neighbor_mode,
    )
    transition_fn = build_transition_function(cfg, landscape)

    summary = sweep_commutator_and_dynamics(
        initial_states=initial_states,
        cfg=cfg,
        transition_fn=transition_fn,
    )
    print_summary(summary, cfg)

    if args.json_out is not None:
        export_json(args.json_out, cfg, summary)

    return summary


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExpérience interrompue par l'utilisateur.")
        raise
