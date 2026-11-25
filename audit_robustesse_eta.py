"""
audit_robustesse_eta.py
Projet : NK-LSTM-PATH

AUDIT DE ROBUSTESSE SUR L'INTENSITÉ DE TRANSPORT ETA

Réutilise :
    fiber_bundle.py
    trajectories.py
    balayage_commutateur_nk_v3.py

Tous les paramètres sauf eta sont maintenus identiques à l'expérience de référence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from balayage_commutateur_nk_v3 import (
    ExperimentConfig,
    SymmetricOrderSweepSummary,
    build_transition_adapter,
    prompt_float,
    prompt_int,
    sample_binary_symmetric_states,
    sweep_symmetric_order,
    validate_config,
)
from trajectories import NKLandscape


DEFAULT_ETA_GRID: List[float] = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]


@dataclass(frozen=True)
class EtaAuditRow:
    eta: float
    n_samples: int
    mean_gap_fro: float
    max_gap_fro: float
    max_sym_err: float
    max_diag_err: float
    frac_same_X: float
    mean_delta_A: float
    mean_delta_w: float
    frac_nonzero_delta_w: float
    total_crossings: int


def parse_eta_grid_input() -> List[float]:
    """
    Demande une liste de valeurs de eta séparées par des virgules ou espaces.
    Une entrée vide conserve DEFAULT_ETA_GRID.
    """
    while True:
        raw = input(
            f"Valeurs de eta à auditer [défaut : {', '.join(str(v) for v in DEFAULT_ETA_GRID)}] : "
        ).strip()

        if not raw:
            return list(DEFAULT_ETA_GRID)

        cleaned = raw.replace(";", " ").replace(",", " ")
        tokens = [tok for tok in cleaned.split() if tok]
        if not tokens:
            print("Entrée vide ou invalide.")
            continue

        values: List[float] = []
        valid = True
        for tok in tokens:
            try:
                val = float(tok)
            except ValueError:
                print(f"Valeur invalide ignorée : '{tok}'. Entrer des nombres.")
                valid = False
                break
            if not np.isfinite(val):
                print(f"Valeur non finie : '{tok}'.")
                valid = False
                break
            values.append(val)

        if valid:
            return values


def collect_audit_parameters() -> tuple[List[float], float, float, float, float, int]:
    """
    Collecte la grille de eta ainsi que les paramètres fixes du protocole.
    """
    print()
    print("=" * 72)
    print("NK-LSTM-PATH — AUDIT DE ROBUSTESSE SUR ETA")
    print("=" * 72)
    print()
    print("Paramètres structurels fixés :")
    print("  N = 6 | Nombre de graines = 10 | Graine initiale = 7 | Échelle J = 0.2")
    print("  Graine du paysage NK = 7")
    print()

    eta_values = parse_eta_grid_input()
    mu = prompt_float("mu — échelle des taux (fixe)")
    beta = prompt_float("beta — sensibilité à la fitness (fixe)")
    theta_A = prompt_float("theta_A — seuil d'accessibilité (fixe)")
    lam = prompt_float("lam — pondération du couplage (fixe)")
    nk_K = prompt_int("K — nombre de dépendances épistatiques NK (fixe)")

    return eta_values, mu, beta, theta_A, lam, nk_K


def run_eta_robustness_audit(
    eta_values: List[float],
    mu: float,
    beta: float,
    theta_A: float,
    lam: float,
    nk_K: int,
) -> List[EtaAuditRow]:
    """
    Exécute le balayage de commutation pour chaque eta sur exactement
    les mêmes états initiaux et le même paysage NK.
    """
    n_sites = 6
    n_seeds = 10
    random_seed = 7
    scale_J = 0.2
    nk_seed = 7

    rng = np.random.default_rng(random_seed)
    initial_states = sample_binary_symmetric_states(
        n_sites=n_sites,
        n_seeds=n_seeds,
        rng=rng,
        scale=scale_J,
    )

    nk_landscape = NKLandscape(
        N=n_sites,
        K=nk_K,
        seed=nk_seed,
    )

    rows: List[EtaAuditRow] = []

    for eta_val in eta_values:
        cfg = ExperimentConfig(
            n_sites=n_sites,
            n_seeds=n_seeds,
            random_seed=random_seed,
            scale_J=scale_J,
            eta=eta_val,
            mu=mu,
            beta=beta,
            theta_A=theta_A,
            lam=lam,
            nk_K=nk_K,
            nk_seed=nk_seed,
        )
        validate_config(cfg)

        adapter = build_transition_adapter(cfg, fnk_fn=nk_landscape.fitness)
        summary: SymmetricOrderSweepSummary = sweep_symmetric_order(
            initial_states=initial_states,
            eta=cfg.eta,
            transition_fn=adapter,
            theta_A=cfg.theta_A,
            unordered_pairs_only=cfg.unordered_pairs_only,
            tol_w=cfg.tol_w,
            rtol=cfg.rtol,
            atol=cfg.atol,
        )

        rows.append(
            EtaAuditRow(
                eta=float(cfg.eta),
                n_samples=summary.n_samples,
                mean_gap_fro=summary.mean_non_commut_gap_fro,
                max_gap_fro=summary.max_non_commut_gap_fro,
                max_sym_err=summary.max_symmetry_error,
                max_diag_err=summary.max_diagonal_error,
                frac_same_X=summary.frac_same_X_endpoint,
                mean_delta_A=summary.mean_delta_A_norm,
                mean_delta_w=summary.mean_delta_w_norm,
                frac_nonzero_delta_w=summary.frac_nonzero_delta_w,
                total_crossings=summary.total_threshold_crossings,
            )
        )

    return rows


def print_audit_table(rows: List[EtaAuditRow]) -> None:
    """Affiche le tableau comparatif complet de robustesse sur eta."""
    print()
    print("=" * 116)
    print("TABLEAU DE ROBUSTESSE — VARIATION DE ETA (AUTRES PARAMÈTRES FIXES)")
    print("=" * 116)
    header = (
        f"{'eta':>8} | {'N_comp':>6} | {'Écart moy J':>12} | {'Écart max J':>12} | "
        f"{'Err sym':>9} | {'Err diag':>9} | {'X id.':>7} | "
        f"{'Δw≠0':>7} | {'Seuils':>6}"
    )
    print(header)
    print("-" * 116)

    for r in rows:
        print(
            f"{r.eta:8.4g} | {r.n_samples:6d} | {r.mean_gap_fro:12.6f} | {r.max_gap_fro:12.6f} | "
            f"{r.max_sym_err:9.2e} | {r.max_diag_err:9.2e} | {r.frac_same_X * 100:6.1f}% | "
            f"{r.frac_nonzero_delta_w * 100:6.1f}% | {r.total_crossings:6d}"
        )

    print("=" * 116)


def main() -> List[EtaAuditRow]:
    eta_values, mu, beta, theta_A, lam, nk_K = collect_audit_parameters()
    rows = run_eta_robustness_audit(
        eta_values=eta_values,
        mu=mu,
        beta=beta,
        theta_A=theta_A,
        lam=lam,
        nk_K=nk_K,
    )
    print_audit_table(rows)
    return rows


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Audit interrompu par l'utilisateur.")
    except Exception as exc:
        print()
        print("ÉCHEC DE L'AUDIT")
        print(f"{type(exc).__name__} : {exc}")
        raise
