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
  - la conservation de la symétrie et de la diagonale nulle ;
  - la différence d'accessibilité A(J) entre les deux chemins ;
  - la différence des taux w(X, J), en norme et composante par composante ;
  - les désaccords entre les masques d'accessibilité finaux.

Le paysage NK et les états initiaux sont générés une seule fois par exécution.
Les deux règles de transport utilisent exactement les mêmes états initiaux,
les mêmes paires de mutations et le même paysage NK.

Exécution depuis la racine du dépôt :
    python .\balayage_commutateur_nk_v2.py

Paramètres modifiables en ligne de commande :
    python .\balayage_commutateur_nk_v2.py --eta 0.25 --lam 0.5

Export facultatif :
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

if _SRC_DIR.is_dir():
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
from trajectories import NKLandscape, transition_rates  # noqa: E402


TransportMode = Literal["integrable", "symmetric"]


# ======================================================================
# 1. CONFIGURATION EXPÉRIMENTALE
# ======================================================================

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

    tol_w: float = 1e-12
    rtol: float = 1e-12
    atol: float = 1e-14
    unordered_pairs_only: bool = True


def validate_config(cfg: ExperimentConfig) -> None:
    """Valide les paramètres avant toute génération ou simulation."""
    if cfg.n_sites < 2:
        raise ValueError("n_sites doit être >= 2.")
    if cfg.n_initial_states < 1:
        raise ValueError("n_initial_states doit être >= 1.")
    if not (0 <= cfg.nk_K < cfg.n_sites):
        raise ValueError("nk_K doit satisfaire 0 <= K < N.")

    for name in ("init_seed", "nk_seed"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} doit être un entier.")
        if int(value) < 0:
            raise ValueError(f"{name} doit être positif ou nul.")

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
    for name, value in finite_values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} doit être fini.")

    if cfg.scale_J < 0.0:
        raise ValueError("scale_J doit être >= 0.")
    if cfg.mu < 0.0:
        raise ValueError("mu doit être >= 0.")
    for name in ("tol_w", "rtol", "atol"):
        if getattr(cfg, name) < 0.0:
            raise ValueError(f"{name} doit être >= 0.")


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
    if mode == "integrable":
        psi_fn = psi_integrable
    elif mode == "symmetric":
        psi_fn = psi_sym
    else:
        raise ValueError(f"Mode de transport inconnu : {mode!r}.")

    return mutate_postflip(
        X,
        J,
        k=int(site),
        eta=float(eta),
        psi_fn=psi_fn,
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
    if n_sites < 2 or n_states < 1:
        raise ValueError("Il faut n_sites >= 2 et n_states >= 1.")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("scale doit être fini et >= 0.")

    states: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_states):
        X0 = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64),
            size=n_sites,
        )
        raw_J = rng.normal(0.0, scale, size=(n_sites, n_sites))
        J0 = 0.5 * (raw_J + raw_J.T)
        np.fill_diagonal(J0, 0.0)
        states.append((X0, J0))
    return states


def iter_site_pairs(
    n_sites: int,
    unordered_only: bool = True,
) -> Iterable[tuple[int, int]]:
    """Génère les paires distinctes de sites, sans doublons par défaut."""
    for k in range(n_sites):
        start_m = k + 1 if unordered_only else 0
        for m in range(start_m, n_sites):
            if k != m:
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


def matrix_symmetry_error(J: np.ndarray) -> float:
    return float(np.linalg.norm(J - J.T, ord="fro"))


def matrix_diagonal_error(J: np.ndarray) -> float:
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J doit être carrée.")
    return float(np.max(np.abs(np.diag(J))))


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
    if k == m:
        raise ValueError("Les sites k et m doivent être distincts.")

    X0v = _validate_state(X0)
    J0v = _validate_J(J0)
    if J0v.shape != (X0v.size, X0v.size):
        raise ValueError("Dimensions incompatibles entre X0 et J0.")
    if not np.isfinite(eta):
        raise ValueError("eta doit être fini.")
    if not (0 <= k < X0v.size and 0 <= m < X0v.size):
        raise IndexError("k et m doivent être des indices valides.")

    # Chemin A : k puis m.
    XA, JA = step_flip_node(X0v.copy(), J0v.copy(), k, eta, mode)
    XA, JA = step_flip_node(XA, JA, m, eta, mode)

    # Chemin B : m puis k.
    XB, JB = step_flip_node(X0v.copy(), J0v.copy(), m, eta, mode)
    XB, JB = step_flip_node(XB, JB, k, eta, mode)

    XA = _validate_state(XA)
    XB = _validate_state(XB)
    JA = _validate_J(JA)
    JB = _validate_J(JB)

    same_X = bool(np.array_equal(XA, XB))
    same_exact = bool(same_X and np.array_equal(JA, JB))
    same_close = bool(same_X and np.allclose(JA, JB, rtol=rtol, atol=atol))

    return PathPairResult(
        XA=XA,
        XB=XB,
        JA=JA,
        JB=JB,
        same_X_endpoint=same_X,
        same_full_endpoint_exact=same_exact,
        same_full_endpoint_close=same_close,
        symmetry_error_A=matrix_symmetry_error(JA),
        symmetry_error_B=matrix_symmetry_error(JB),
        diagonal_error_A=matrix_diagonal_error(JA),
        diagonal_error_B=matrix_diagonal_error(JB),
        commutator_fro=float(np.linalg.norm(JA - JB, ord="fro")),
    )


# ======================================================================
# 5. OBSERVABLES : ACCESSIBILITÉ, TAUX ET MASQUES
# ======================================================================

@dataclass(frozen=True)
class ObservableGap:
    delta_A_norm: float
    delta_w_norm: float
    delta_w_max: float
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
    A_A = np.asarray(accessibility_A(JA), dtype=np.float64)
    A_B = np.asarray(accessibility_A(JB), dtype=np.float64)
    w_A = np.asarray(transition_fn(XA, JA), dtype=np.float64)
    w_B = np.asarray(transition_fn(XB, JB), dtype=np.float64)

    if A_A.shape != A_B.shape or w_A.shape != w_B.shape:
        raise ValueError("Les dimensions des observables comparées diffèrent.")
    for name, values in (("A_A", A_A), ("A_B", A_B), ("w_A", w_A), ("w_B", w_B)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contient des valeurs non finies.")

    delta_A = A_A - A_B
    delta_w = w_A - w_B
    mask_A = A_A > theta_A
    mask_B = A_B > theta_A

    return ObservableGap(
        delta_A_norm=float(np.linalg.norm(delta_A)),
        delta_w_norm=float(np.linalg.norm(delta_w)),
        delta_w_max=float(np.max(np.abs(delta_w))),
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
    delta_w_norm_int: float
    delta_w_norm_sym: float
    delta_w_max_int: float
    delta_w_max_sym: float

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
    if not initial_states:
        raise ValueError("initial_states ne peut pas être vide.")

    records: list[SweepSampleRecord] = []

    for seed_idx, (X0, J0) in enumerate(initial_states):
        X0v = _validate_state(X0)
        J0v = _validate_J(J0)
        if X0v.size != cfg.n_sites or J0v.shape != (cfg.n_sites, cfg.n_sites):
            raise ValueError(f"Dimensions initiales incompatibles à l'indice {seed_idx}.")

        for k, m in iter_site_pairs(cfg.n_sites, cfg.unordered_pairs_only):
            result_int = compute_two_step_commutator(
                X0v, J0v, k, m, cfg.eta, mode="integrable", rtol=cfg.rtol, atol=cfg.atol
            )
            result_sym = compute_two_step_commutator(
                X0v, J0v, k, m, cfg.eta, mode="symmetric", rtol=cfg.rtol, atol=cfg.atol
            )

            # Contrôle attendu : le transport intégrable commute.
            if not result_int.same_X_endpoint or not result_int.same_full_endpoint_close:
                raise RuntimeError(
                    "Le contrôle intégrable ne commute pas à la tolérance demandée "
                    f"(état {seed_idx}, paire ({k}, {m}), "
                    f"écart={result_int.commutator_fro:.3e})."
                )

            gap_int = evaluate_observable_gap(
                result_int.XA, result_int.JA, result_int.XB, result_int.JB,
                transition_fn, cfg.theta_A,
            )
            gap_sym = evaluate_observable_gap(
                result_sym.XA, result_sym.JA, result_sym.XB, result_sym.JB,
                transition_fn, cfg.theta_A,
            )

            records.append(
                SweepSampleRecord(
                    seed_idx=seed_idx,
                    k=k,
                    m=m,
                    same_X_endpoint_int=result_int.same_X_endpoint,
                    same_X_endpoint_sym=result_sym.same_X_endpoint,
                    same_full_endpoint_exact_int=result_int.same_full_endpoint_exact,
                    same_full_endpoint_exact_sym=result_sym.same_full_endpoint_exact,
                    same_full_endpoint_close_int=result_int.same_full_endpoint_close,
                    same_full_endpoint_close_sym=result_sym.same_full_endpoint_close,
                    symmetry_error_max_int=max(result_int.symmetry_error_A, result_int.symmetry_error_B),
                    symmetry_error_max_sym=max(result_sym.symmetry_error_A, result_sym.symmetry_error_B),
                    diagonal_error_max_int=max(result_int.diagonal_error_A, result_int.diagonal_error_B),
                    diagonal_error_max_sym=max(result_sym.diagonal_error_A, result_sym.diagonal_error_B),
                    commutator_int_fro=result_int.commutator_fro,
                    commutator_sym_fro=result_sym.commutator_fro,
                    delta_A_norm_int=gap_int.delta_A_norm,
                    delta_A_norm_sym=gap_sym.delta_A_norm,
                    delta_w_norm_int=gap_int.delta_w_norm,
                    delta_w_norm_sym=gap_sym.delta_w_norm,
                    delta_w_max_int=gap_int.delta_w_max,
                    delta_w_max_sym=gap_sym.delta_w_max,
                    mask_disagreements_int=gap_int.mask_disagreements,
                    mask_disagreements_sym=gap_sym.mask_disagreements,
                )
            )

    def values(name: str) -> np.ndarray:
        return np.asarray([getattr(record, name) for record in records], dtype=np.float64)

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
        frac_same_X_endpoint_int=float(np.mean(values("same_X_endpoint_int"))),
        frac_same_X_endpoint_sym=float(np.mean(values("same_X_endpoint_sym"))),
        frac_same_full_endpoint_exact_int=float(np.mean(values("same_full_endpoint_exact_int"))),
        frac_same_full_endpoint_exact_sym=float(np.mean(values("same_full_endpoint_exact_sym"))),
        frac_same_full_endpoint_close_int=float(np.mean(values("same_full_endpoint_close_int"))),
        frac_same_full_endpoint_close_sym=float(np.mean(values("same_full_endpoint_close_sym"))),
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
        total_mask_disagreements_int=int(np.sum(values("mask_disagreements_int"))),
        total_mask_disagreements_sym=int(np.sum(values("mask_disagreements_sym"))),
        records=records,
    )


# ======================================================================
# 8. FITNESS NK ET TAUX DE TRANSITION
# ======================================================================

def build_transition_function(
    cfg: ExperimentConfig,
    landscape: NKLandscape,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Construit le calcul des taux à partir du paysage NK partagé."""
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
    print(f"N={cfg.n_sites} | états initiaux={cfg.n_initial_states} | comparaisons={summary.n_samples}")
    print(
        f"eta={cfg.eta:g} | mu={cfg.mu:g} | beta={cfg.beta:g} | "
        f"theta_A={cfg.theta_A:g} | lambda={cfg.lam:g} | K={cfg.nk_K}"
    )
    print(f"Graine initiale={cfg.init_seed} | graine NK={cfg.nk_seed} | échelle J0={cfg.scale_J:g}")

    print("\nCONTRÔLE INTÉGRABLE")
    print(f"Commutateur moyen / max       : {summary.mean_commutator_int_fro:.6e} / {summary.max_commutator_int_fro:.6e}")
    print(f"États complets identiques     : {100.0 * summary.frac_same_full_endpoint_exact_int:.1f}% exacts")
    print(f"États complets proches        : {100.0 * summary.frac_same_full_endpoint_close_int:.1f}% à tolérance")
    print(f"Erreur max de symétrie        : {summary.max_symmetry_error_int:.6e}")
    print(f"Erreur max de diagonale       : {summary.max_diagonal_error_int:.6e}")
    print(f"Écart moyen d'accessibilité   : {summary.mean_delta_A_norm_int:.6e}")
    print(f"Écart moyen des taux          : {summary.mean_delta_w_norm_int:.6e}")
    print(f"Comparaisons avec Δw non nul  : {100.0 * summary.frac_nonzero_delta_w_int:.2f}%")
    print(
        f"Désaccords de masques         : {summary.total_mask_disagreements_int}/"
        f"{total_site_comparisons} ({100.0 * summary.total_mask_disagreements_int / total_site_comparisons:.2f}%)"
    )

    print("\nTRANSPORT SYMÉTRIQUE")
    print(f"Commutateur moyen / max       : {summary.mean_commutator_sym_fro:.6e} / {summary.max_commutator_sym_fro:.6e}")
    print(f"États X identiques            : {100.0 * summary.frac_same_X_endpoint_sym:.1f}%")
    print(f"États complets identiques     : {100.0 * summary.frac_same_full_endpoint_exact_sym:.1f}% exacts")
    print(f"États complets proches        : {100.0 * summary.frac_same_full_endpoint_close_sym:.1f}% à tolérance")
    print(f"Erreur max de symétrie        : {summary.max_symmetry_error_sym:.6e}")
    print(f"Erreur max de diagonale       : {summary.max_diagonal_error_sym:.6e}")
    print(f"Écart moyen d'accessibilité   : {summary.mean_delta_A_norm_sym:.6e}")
    print(f"Écart maximal d'accessibilité : {summary.max_delta_A_norm_sym:.6e}")
    print(f"Écart moyen des taux          : {summary.mean_delta_w_norm_sym:.6e}")
    print(f"Écart maximal des taux        : {summary.max_delta_w_norm_sym:.6e}")
    print(f"Comparaisons avec Δw non nul  : {100.0 * summary.frac_nonzero_delta_w_sym:.2f}%")
    print(
        f"Désaccords de masques         : {summary.total_mask_disagreements_sym}/"
        f"{total_site_comparisons} ({100.0 * summary.total_mask_disagreements_sym / total_site_comparisons:.2f}%)"
    )

    print("\nDéfinition de Δw non nul : max_k |w_A,k - w_B,k| > tol_w.")
    print("Les désaccords comptent les sites dont les masques finaux diffèrent entre les deux ordres.")
    print("Ils ne représentent pas des franchissements temporels du seuil.")
    print("=" * 76)


def export_json(
    path: Path,
    cfg: ExperimentConfig,
    summary: SweepSummary,
) -> None:
    """Exporte paramètres, synthèse et enregistrements individuels en JSON."""
    payload = {
        "config": asdict(cfg),
        "summary": asdict(summary),
        "definitions": {
            "integrable_transport": "Psi_ij(X') = s'_i*s'_j for i != j; Psi_ii = 0",
            "symmetric_transport": "fiber_bundle.psi_sym",
            "nonzero_rate_difference": "max_k(abs(w_A[k] - w_B[k])) > tol_w",
            "mask_disagreements": "sitewise XOR of final masks A_k(J) > theta_A",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"\nRésultats détaillés enregistrés : {path}")


# ======================================================================
# 10. INTERFACE DE LIGNE DE COMMANDE
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare le transport intégrable et le transport symétrique de NK-LSTM-PATH."
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
    parser.add_argument("--tol-w", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-14)
    parser.add_argument("--ordered-pairs", action="store_true", help="Inclut les deux orientations des paires.")
    parser.add_argument("--json-out", type=Path, default=None, help="Chemin facultatif d'export JSON.")
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
        tol_w=args.tol_w,
        rtol=args.rtol,
        atol=args.atol,
        unordered_pairs_only=not args.ordered_pairs,
    )
    validate_config(cfg)

    rng = np.random.default_rng(cfg.init_seed)
    initial_states = sample_binary_symmetric_states(
        n_sites=cfg.n_sites,
        n_states=cfg.n_initial_states,
        rng=rng,
        scale=cfg.scale_J,
    )

    # Un seul paysage NK, partagé par les deux modes et toutes les paires.
    landscape = NKLandscape(N=cfg.n_sites, K=cfg.nk_K, seed=cfg.nk_seed)
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
