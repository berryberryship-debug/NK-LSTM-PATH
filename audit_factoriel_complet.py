
"""
audit_factoriel_complet.py
Projet : NK-LSTM-PATH

AUDIT FACTORIEL CROISÉ :
    init_seed x nk_seed x eta

Objectifs
---------
1. Mesurer l'effet de eta.
2. Mesurer la sensibilité aux configurations initiales.
3. Mesurer la sensibilité aux paysages NK.
4. Examiner descriptivement l'interaction init_seed x nk_seed.
5. Exporter les résultats détaillés en JSON.

Protocole par défaut
-------------------
N = 6
10 configurations initiales par init_seed
5 graines initiales
5 graines NK
6 valeurs de eta
15 paires non ordonnées par configuration.

Nombre attendu :
    5 x 5 x 6 x 10 x 15 = 22 500 comparaisons.

Le même ensemble de configurations initiales est réutilisé
pour tous les paysages NK et toutes les valeurs de eta associées
à une même graine initiale.

Le même paysage NK est réutilisé pour toutes les configurations
initiales et toutes les valeurs de eta associées à une même
graine NK.

Les statistiques entre cellules sont descriptives.
Les 22 500 comparaisons ne sont pas 22 500 réplications
indépendantes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from balayage_commutateur_nk_v3 import (
    ExperimentConfig,
    build_transition_adapter,
    sample_binary_symmetric_states,
    sweep_symmetric_order,
    validate_config,
)

from trajectories import NKLandscape


# ================================================================
# 1. PARAMÈTRES PAR DÉFAUT
# ================================================================

DEFAULT_ETA_GRID = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]

DEFAULT_INIT_SEEDS = [7, 13, 21, 42, 99]
DEFAULT_NK_SEEDS = [7, 13, 21, 42, 99]

DEFAULT_N_SITES = 6
DEFAULT_N_INITIAL_STATES = 10
DEFAULT_SCALE_J = 0.2

DEFAULT_MU = 1.0
DEFAULT_BETA = 1.0
DEFAULT_THETA_A = 0.1
DEFAULT_LAM = 0.5
DEFAULT_NK_K = 3


# ================================================================
# 2. VALIDATION DES LISTES
# ================================================================

def parse_float_list(raw: str, name: str) -> list[float]:
    """Convertit une liste de réels séparés par virgules ou espaces."""

    tokens = raw.replace(";", " ").replace(",", " ").split()

    if not tokens:
        raise ValueError(f"{name} ne peut pas être vide.")

    values = []

    for token in tokens:
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                f"Valeur invalide dans {name} : {token}"
            ) from exc

        if not np.isfinite(value):
            raise ValueError(
                f"Valeur non finie dans {name} : {token}"
            )

        if value in values:
            raise ValueError(
                f"Valeur dupliquée dans {name} : {value}"
            )

        values.append(value)

    return values


def parse_seed_list(raw: str, name: str) -> list[int]:
    """Convertit et valide une liste de graines entières."""

    tokens = raw.replace(";", " ").replace(",", " ").split()

    if not tokens:
        raise ValueError(f"{name} ne peut pas être vide.")

    values = []

    for token in tokens:
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Graine invalide dans {name} : {token}"
            ) from exc

        if value < 0:
            raise ValueError(
                f"Les graines doivent être >= 0 : {value}"
            )

        if value in values:
            raise ValueError(
                f"Graine dupliquée dans {name} : {value}"
            )

        values.append(value)

    return values


def validate_global_parameters(args: argparse.Namespace) -> None:
    """Valide les paramètres communs avant la génération des données."""

    integer_values = {
        "n_sites": args.n_sites,
        "n_initial_states": args.n_initial_states,
        "nk_K": args.nk_K,
    }

    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} doit être un entier.")

    if args.n_sites < 2:
        raise ValueError("n_sites doit être >= 2.")

    if args.n_initial_states < 1:
        raise ValueError("n_initial_states doit être >= 1.")

    if not (0 <= args.nk_K < args.n_sites):
        raise ValueError("nk_K doit vérifier 0 <= K < N.")

    real_values = {
        "scale_J": args.scale_J,
        "mu": args.mu,
        "beta": args.beta,
        "theta_A": args.theta_A,
        "lam": args.lam,
    }

    for name, value in real_values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} doit être fini.")

    if args.scale_J < 0.0:
        raise ValueError("scale_J doit être >= 0.")

    if args.mu < 0.0:
        raise ValueError("mu doit être >= 0.")


# ================================================================
# 3. ARGUMENTS DE LIGNE DE COMMANDE
# ================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit factoriel croisé NK-LSTM-PATH : "
            "init_seed x nk_seed x eta."
        )
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Exécute avec les paramètres fournis ou les valeurs par défaut.",
    )

    parser.add_argument(
        "--eta-grid",
        type=str,
        default="0,0.05,0.1,0.25,0.5,1.0",
    )

    parser.add_argument(
        "--init-seeds",
        type=str,
        default="7,13,21,42,99",
    )

    parser.add_argument(
        "--nk-seeds",
        type=str,
        default="7,13,21,42,99",
    )

    parser.add_argument("--n-sites", type=int, default=DEFAULT_N_SITES)
    parser.add_argument(
        "--n-initial-states",
        type=int,
        default=DEFAULT_N_INITIAL_STATES,
    )
    parser.add_argument("--scale-J", type=float, default=DEFAULT_SCALE_J)

    parser.add_argument("--mu", type=float, default=DEFAULT_MU)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--theta-A", type=float, default=DEFAULT_THETA_A)
    parser.add_argument("--lam", type=float, default=DEFAULT_LAM)
    parser.add_argument("--nk-K", type=int, default=DEFAULT_NK_K)

    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Chemin d'export JSON des résultats.",
    )

    return parser.parse_args()


# ================================================================
# 4. CONFIGURATION EXPÉRIMENTALE
# ================================================================

def make_config(
    args: argparse.Namespace,
    init_seed: int,
    nk_seed: int,
    eta: float,
) -> ExperimentConfig:
    """Construit une configuration validée pour une cellule factorielle."""

    cfg = ExperimentConfig(
        n_sites=args.n_sites,
        n_seeds=args.n_initial_states,
        random_seed=init_seed,
        scale_J=args.scale_J,

        eta=eta,
        mu=args.mu,
        beta=args.beta,
        theta_A=args.theta_A,
        lam=args.lam,

        nk_K=args.nk_K,
        nk_seed=nk_seed,
    )

    validate_config(cfg)

    return cfg


# ================================================================
# 5. INVARIANTS STRUCTURELS
# ================================================================

def validate_summary(
    summary: Any,
    cfg: ExperimentConfig,
    expected_comparisons: int,
) -> None:
    """Vérifie les invariants algorithmiques de chaque cellule."""

    if summary.n_samples != expected_comparisons:
        raise RuntimeError(
            f"Nombre de comparaisons incorrect : "
            f"{summary.n_samples}, attendu : {expected_comparisons}."
        )

    if summary.frac_same_X_endpoint != 1.0:
        raise RuntimeError(
            "Les deux ordres ne produisent pas le même état X final."
        )

    metrics = [
        summary.mean_non_commut_gap_fro,
        summary.max_non_commut_gap_fro,
        summary.max_symmetry_error,
        summary.max_diagonal_error,
        summary.mean_delta_A_norm,
        summary.max_delta_A_norm,
        summary.mean_delta_w_norm,
        summary.max_delta_w_norm,
        summary.frac_nonzero_delta_w,
    ]

    if not np.all(np.isfinite(metrics)):
        raise RuntimeError("Une métrique contient NaN ou infini.")

    tolerance = max(
        1e-12,
        10.0 * cfg.rtol,
        10.0 * cfg.atol,
    )

    if summary.max_symmetry_error > tolerance:
        raise RuntimeError("La symétrie de J n'est pas conservée.")

    if summary.max_diagonal_error > tolerance:
        raise RuntimeError("La diagonale de J n'est pas nulle.")

    # Contrôle analytique du cas eta = 0.
    if cfg.eta == 0.0:

        zero_metrics = [
            summary.mean_non_commut_gap_fro,
            summary.max_non_commut_gap_fro,
            summary.mean_delta_A_norm,
            summary.max_delta_A_norm,
            summary.mean_delta_w_norm,
            summary.max_delta_w_norm,
        ]

        if any(abs(float(value)) > tolerance for value in zero_metrics):
            raise RuntimeError(
                "Invariant eta=0 violé : les deux ordres devraient "
                "produire des états complets identiques."
            )

        if summary.frac_same_full_endpoint_exact != 1.0:
            raise RuntimeError(
                "Les états complets ne sont pas identiques à eta=0."
            )

        if summary.total_threshold_crossings != 0:
            raise RuntimeError(
                "Désaccord de masque actif détecté à eta=0."
            )


# ================================================================
# 6. CONSTRUCTION D'UNE CELLULE FACTORIELLE
# ================================================================

def run_factorial_cell(
    args: argparse.Namespace,
    initial_states: list[tuple[np.ndarray, np.ndarray]],
    landscape: NKLandscape,
    init_seed: int,
    nk_seed: int,
    eta: float,
    expected_comparisons: int,
) -> dict:
    """
    Exécute une cellule :

        (init_seed, nk_seed, eta)

    Les états initiaux et le paysage NK sont déjà construits
    et réutilisés pour les autres cellules.
    """

    cfg = make_config(
        args=args,
        init_seed=init_seed,
        nk_seed=nk_seed,
        eta=eta,
    )

    if landscape.N != args.n_sites:
        raise RuntimeError("Dimension incorrecte du paysage NK.")

    if landscape.K != args.nk_K:
        raise RuntimeError("Valeur K incorrecte du paysage NK.")

    # Transmission de l'instance :
    # le calcul incrémental delta_fitness_flip reste accessible.
    adapter = build_transition_adapter(
        cfg,
        fnk_fn=landscape,
    )

    summary = sweep_symmetric_order(
        initial_states=initial_states,
        eta=cfg.eta,
        transition_fn=adapter,
        theta_A=cfg.theta_A,
        unordered_pairs_only=cfg.unordered_pairs_only,
        tol_w=cfg.tol_w,
        rtol=cfg.rtol,
        atol=cfg.atol,
    )

    validate_summary(
        summary=summary,
        cfg=cfg,
        expected_comparisons=expected_comparisons,
    )

    denominator = summary.n_samples * args.n_sites

    return {
        "init_seed": int(init_seed),
        "nk_seed": int(nk_seed),
        "eta": float(eta),

        "n_samples": int(summary.n_samples),

        "mean_gap_fro": float(summary.mean_non_commut_gap_fro),
        "max_gap_fro": float(summary.max_non_commut_gap_fro),

        "max_symmetry_error": float(summary.max_symmetry_error),
        "max_diagonal_error": float(summary.max_diagonal_error),

        "frac_same_X": float(summary.frac_same_X_endpoint),
        "frac_same_full_exact": float(
            summary.frac_same_full_endpoint_exact
        ),
        "frac_same_full_close": float(
            summary.frac_same_full_endpoint_close
        ),

        "mean_delta_A": float(summary.mean_delta_A_norm),
        "max_delta_A": float(summary.max_delta_A_norm),

        "mean_delta_w": float(summary.mean_delta_w_norm),
        "max_delta_w": float(summary.max_delta_w_norm),

        "frac_nonzero_delta_w": float(
            summary.frac_nonzero_delta_w
        ),

        "active_mask_disagreements": int(
            summary.total_threshold_crossings
        ),

        "active_mask_disagreements_pct": float(
            100.0 * summary.total_threshold_crossings / denominator
        ),
    }


# ================================================================
# 7. AGRÉGATION DES CELLULES
# ================================================================

METRICS = (
    "mean_gap_fro",
    "max_gap_fro",
    "mean_delta_A",
    "mean_delta_w",
    "frac_nonzero_delta_w",
    "active_mask_disagreements_pct",
)


def mean_sd(values: list[float]) -> dict:
    """Moyenne et écart-type descriptifs d'une série de cellules."""

    array = np.asarray(values, dtype=np.float64)

    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Série vide ou non finie dans l'agrégation.")

    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }


def aggregate_by_factor(
    rows: list[dict],
    factor: str,
    levels: list,
) -> list[dict]:
    """
    Agrège les cellules selon un facteur en moyennant
    sur tous les niveaux des autres facteurs.
    """

    output = []

    for level in levels:
        group = [row for row in rows if row[factor] == level]

        if not group:
            raise RuntimeError(
                f"Aucune cellule trouvée pour {factor}={level}."
            )

        record = {
            factor: level,
            "n_cells": len(group),
        }

        for metric in METRICS:
            record[metric] = mean_sd(
                [float(row[metric]) for row in group]
            )

        output.append(record)

    return output


def calculate_init_nk_interactions(
    rows: list[dict],
    eta_grid: list[float],
    init_seeds: list[int],
    nk_seeds: list[int],
) -> list[dict]:
    """
    Calcule un résidu d'interaction descriptif init_seed x nk_seed.

    Pour chaque eta et chaque métrique :

        R_ij = Y_ij - moyenne_ligne_i - moyenne_colonne_j + moyenne_globale

    RMS(R) et max(abs(R)) décrivent l'écart au modèle additif
    à deux facteurs.

    Ces statistiques ne constituent pas, à elles seules,
    un test statistique d'interaction.
    """

    output = []

    for eta in eta_grid:
        eta_rows = [row for row in rows if row["eta"] == eta]

        if len(eta_rows) != len(init_seeds) * len(nk_seeds):
            raise RuntimeError(
                f"Cellules manquantes pour eta={eta}."
            )

        item = {"eta": float(eta)}

        for metric in ("mean_gap_fro", "mean_delta_A", "mean_delta_w"):
            matrix = np.empty(
                (len(init_seeds), len(nk_seeds)),
                dtype=np.float64,
            )

            for i, init_seed in enumerate(init_seeds):
                for j, nk_seed in enumerate(nk_seeds):
                    matches = [
                        row for row in eta_rows
                        if row["init_seed"] == init_seed
                        and row["nk_seed"] == nk_seed
                    ]

                    if len(matches) != 1:
                        raise RuntimeError(
                            "Cellule factorielle manquante ou dupliquée : "
                            f"init={init_seed}, NK={nk_seed}, eta={eta}."
                        )

                    matrix[i, j] = matches[0][metric]

            residual = (
                matrix
                - np.mean(matrix, axis=1, keepdims=True)
                - np.mean(matrix, axis=0, keepdims=True)
                + np.mean(matrix)
            )

            item[f"{metric}_interaction_rms"] = float(
                np.sqrt(np.mean(residual ** 2))
            )

            item[f"{metric}_interaction_max_abs"] = float(
                np.max(np.abs(residual))
            )

        output.append(item)

    return output


# ================================================================
# 8. AFFICHAGE
# ================================================================

def print_eta_summary(rows: list[dict]) -> None:
    """Affiche les effets descriptifs selon eta."""

    print()
    print("=" * 150)
    print("SYNTHÈSE PAR ETA — MOYENNE ± ÉCART-TYPE ENTRE CELLULES DE GRAINES")
    print("=" * 150)

    print(
        f"{'eta':>7} | {'Cellules':>8} | "
        f"{'Gap J':>23} | {'Delta A':>23} | "
        f"{'Delta w':>23} | {'Paires Δw != 0':>17} | "
        f"{'Désaccords (%)':>17}"
    )
    print("-" * 150)

    for row in rows:
        gap = row["mean_gap_fro"]
        delta_a = row["mean_delta_A"]
        delta_w = row["mean_delta_w"]
        nonzero = row["frac_nonzero_delta_w"]
        masks = row["active_mask_disagreements_pct"]

        print(
            f"{row['eta']:7.3g} | "
            f"{row['n_cells']:8d} | "
            f"{gap['mean']:.5e} ± {gap['sd']:.2e} | "
            f"{delta_a['mean']:.5e} ± {delta_a['sd']:.2e} | "
            f"{delta_w['mean']:.5e} ± {delta_w['sd']:.2e} | "
            f"{100.0 * nonzero['mean']:7.2f}% ± {100.0 * nonzero['sd']:.2f}% | "
            f"{masks['mean']:7.2f} ± {masks['sd']:.2f}"
        )

    print("=" * 150)


def print_seed_summary(
    title: str,
    rows: list[dict],
    factor: str,
) -> None:
    """Affiche les effets marginaux descriptifs d'une graine."""

    print()
    print("=" * 126)
    print(title)
    print("=" * 126)

    print(
        f"{factor:>10} | {'Cellules':>8} | "
        f"{'Gap J':>22} | {'Delta A':>22} | "
        f"{'Delta w':>22} | {'Désaccords (%)':>18}"
    )
    print("-" * 126)

    for row in rows:
        gap = row["mean_gap_fro"]
        delta_a = row["mean_delta_A"]
        delta_w = row["mean_delta_w"]
        masks = row["active_mask_disagreements_pct"]

        print(
            f"{row[factor]:10d} | "
            f"{row['n_cells']:8d} | "
            f"{gap['mean']:.5e} ± {gap['sd']:.2e} | "
            f"{delta_a['mean']:.5e} ± {delta_a['sd']:.2e} | "
            f"{delta_w['mean']:.5e} ± {delta_w['sd']:.2e} | "
            f"{masks['mean']:8.2f} ± {masks['sd']:.2f}"
        )

    print("=" * 126)


def print_interaction_summary(rows: list[dict]) -> None:
    """Affiche les résidus d'interaction init x NK."""

    print()
    print("=" * 110)
    print("INTERACTION DESCRIPTIVE init_seed x nk_seed")
    print("RMS des résidus après retrait des effets additifs des deux facteurs")
    print("=" * 110)

    print(
        f"{'eta':>7} | {'RMS interaction Gap J':>24} | "
        f"{'RMS interaction Delta A':>25} | "
        f"{'RMS interaction Delta w':>25}"
    )
    print("-" * 110)

    for row in rows:
        print(
            f"{row['eta']:7.3g} | "
            f"{row['mean_gap_fro_interaction_rms']:24.5e} | "
            f"{row['mean_delta_A_interaction_rms']:25.5e} | "
            f"{row['mean_delta_w_interaction_rms']:25.5e}"
        )

    print("=" * 110)


# ================================================================
# 9. EXPORT JSON
# ================================================================

def export_json(
    output_path: str,
    metadata: dict,
    cells: list[dict],
    by_eta: list[dict],
    by_init_seed: list[dict],
    by_nk_seed: list[dict],
    interactions: list[dict],
) -> None:
    """Exporte les données détaillées et les agrégations."""

    path = Path(output_path)

    if path.suffix.lower() != ".json":
        raise ValueError("Le fichier de sortie doit porter l'extension .json.")

    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "project": "NK-LSTM-PATH",
        "audit": "factoriel_complet",
        "metadata": metadata,
        "number_of_cells": len(cells),
        "number_of_pair_comparisons": int(
            sum(row["n_samples"] for row in cells)
        ),
        "interpretation": {
            "statistics": "Descriptives, non assimilables à des tests d'inférence.",
            "pair_comparisons": (
                "Les comparaisons de mutations sont regroupées dans des cellules "
                "de graines et ne sont pas toutes indépendantes."
            ),
            "mask_disagreements": (
                "Désaccords entre les masques d'accessibilité finaux, "
                "pas des franchissements temporels."
            ),
            "interaction": (
                "Résidu descriptif du modèle additif init_seed + nk_seed."
            ),
        },
        "cells": cells,
        "summary_by_eta": by_eta,
        "summary_by_init_seed": by_init_seed,
        "summary_by_nk_seed": by_nk_seed,
        "init_nk_interactions_by_eta": interactions,
    }

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print()
    print(f"Export JSON enregistré : {path.resolve()}")


# ================================================================
# 10. AUDIT FACTORIEL COMPLET
# ================================================================

def run_factorial_audit(args: argparse.Namespace) -> dict:
    """Exécute les 25 x 6 cellules du plan factoriel."""

    validate_global_parameters(args)

    eta_grid = parse_float_list(args.eta_grid, "eta_grid")
    init_seeds = parse_seed_list(args.init_seeds, "init_seeds")
    nk_seeds = parse_seed_list(args.nk_seeds, "nk_seeds")

    if len(init_seeds) < 2:
        raise ValueError("Au moins deux graines initiales sont nécessaires.")

    if len(nk_seeds) < 2:
        raise ValueError("Au moins deux graines NK sont nécessaires.")

    n_sites = args.n_sites
    n_initial_states = args.n_initial_states

    if n_sites >= 2:
        pairs_per_state = n_sites * (n_sites - 1) // 2
    else:
        raise ValueError("N doit être >= 2.")

    # ------------------------------------------------------------
    # Génération unique des configurations initiales
    # ------------------------------------------------------------

    print()
    print("=" * 100)
    print("NK-LSTM-PATH — AUDIT FACTORIEL COMPLET")
    print("=" * 100)

    print(f"N                              : {n_sites}")
    print(f"Configurations par init_seed   : {n_initial_states}")
    print(f"Échelle initiale de J          : {args.scale_J}")
    print(f"Graines initiales              : {init_seeds}")
    print(f"Graines NK                     : {nk_seeds}")
    print(f"Grille eta                     : {eta_grid}")
    print(f"mu                             : {args.mu}")
    print(f"beta                           : {args.beta}")
    print(f"theta_A                        : {args.theta_A}")
    print(f"lambda                         : {args.lam}")
    print(f"K                              : {args.nk_K}")

    initial_states_by_seed = {}

    print()
    print("Génération des configurations initiales...")

    for seed in init_seeds:
        rng = np.random.default_rng(seed)

        states = sample_binary_symmetric_states(
            n_sites=n_sites,
            n_seeds=n_initial_states,
            rng=rng,
            scale=args.scale_J,
        )

        if len(states) != n_initial_states:
            raise RuntimeError(
                f"Génération incomplète pour init_seed={seed}."
            )

        initial_states_by_seed[seed] = states

    # ------------------------------------------------------------
    # Génération unique des paysages NK
    # ------------------------------------------------------------

    print("Génération des paysages NK...")

    landscapes_by_seed = {}

    for seed in nk_seeds:
        landscape = NKLandscape(
            N=n_sites,
            K=args.nk_K,
            seed=seed,
        )

        landscapes_by_seed[seed] = landscape

    # ------------------------------------------------------------
    # Taille du plan factoriel
    # ------------------------------------------------------------

    unordered_only = True
    comparisons_per_cell = n_initial_states * pairs_per_state

    n_cells = (
        len(init_seeds)
        * len(nk_seeds)
        * len(eta_grid)
    )

    expected_total = n_cells * comparisons_per_cell

    print()
    print("STRUCTURE DU PLAN")
    print(f"Cellules factorielles             : {n_cells}")
    print(f"Comparaisons par cellule          : {comparisons_per_cell}")
    print(f"Comparaisons totales attendues    : {expected_total}")
    print()

    # ------------------------------------------------------------
    # Boucle factorielle
    # ------------------------------------------------------------

    cells = []
    cell_index = 0

    for init_seed in init_seeds:
        initial_states = initial_states_by_seed[init_seed]

        for nk_seed in nk_seeds:
            landscape = landscapes_by_seed[nk_seed]

            for eta in eta_grid:
                cell_index += 1

                print(
                    f"[CELL {cell_index}/{n_cells}] "
                    f"init_seed={init_seed}, "
                    f"nk_seed={nk_seed}, "
                    f"eta={eta:g}"
                )

                cell = run_factorial_cell(
                    args=args,
                    initial_states=initial_states,
                    landscape=landscape,
                    init_seed=init_seed,
                    nk_seed=nk_seed,
                    eta=eta,
                    expected_comparisons=comparisons_per_cell,
                )

                cells.append(cell)

    # ------------------------------------------------------------
    # Vérification globale du nombre de comparaisons
    # ------------------------------------------------------------

    actual_total = sum(
        row["n_samples"] for row in cells
    )

    if len(cells) != n_cells:
        raise RuntimeError(
            f"Nombre de cellules incorrect : {len(cells)} != {n_cells}."
        )

    if actual_total != expected_total:
        raise RuntimeError(
            f"Nombre total incorrect : {actual_total} != {expected_total}."
        )

    # ------------------------------------------------------------
    # Agrégations équilibrées
    # ------------------------------------------------------------

    by_eta = aggregate_by_factor(
        cells,
        factor="eta",
        levels=eta_grid,
    )

    by_init_seed = aggregate_by_factor(
        cells,
        factor="init_seed",
        levels=init_seeds,
    )

    by_nk_seed = aggregate_by_factor(
        cells,
        factor="nk_seed",
        levels=nk_seeds,
    )

    interactions = calculate_init_nk_interactions(
        rows=cells,
        eta_grid=eta_grid,
        init_seeds=init_seeds,
        nk_seeds=nk_seeds,
    )

    # ------------------------------------------------------------
    # Métadonnées
    # ------------------------------------------------------------

    metadata = {
        "n_sites": n_sites,
        "n_initial_states_per_init_seed": n_initial_states,
        "scale_J": float(args.scale_J),

        "init_seeds": init_seeds,
        "nk_seeds": nk_seeds,
        "eta_grid": eta_grid,

        "mu": float(args.mu),
        "beta": float(args.beta),
        "theta_A": float(args.theta_A),
        "lam": float(args.lam),
        "nk_K": int(args.nk_K),

        "pairs_per_state": int(pairs_per_state),
        "comparisons_per_cell": int(comparisons_per_cell),
        "number_of_cells": int(n_cells),
        "expected_total_comparisons": int(expected_total),

        "unordered_pairs_only": unordered_only,
    }

    result = {
        "metadata": metadata,
        "cells": cells,
        "by_eta": by_eta,
        "by_init_seed": by_init_seed,
        "by_nk_seed": by_nk_seed,
        "interactions": interactions,
    }

    return result


# ================================================================
# 11. POINT D'ENTRÉE
# ================================================================

def main() -> None:
    args = parse_arguments()

    result = run_factorial_audit(args)

    print_eta_summary(result["by_eta"])

    print_seed_summary(
        "EFFET MARGINAL DES GRAINES INITIALES",
        result["by_init_seed"],
        "init_seed",
    )

    print_seed_summary(
        "EFFET MARGINAL DES GRAINES NK",
        result["by_nk_seed"],
        "nk_seed",
    )

    print_interaction_summary(
        result["interactions"]
    )

    print()
    print("=" * 100)
    print("BILAN DU PLAN FACTORIEL")
    print("=" * 100)

    print(
        f"Cellules calculées             : "
        f"{len(result['cells'])}"
    )

    print(
        f"Comparaisons de paires         : "
        f"{sum(row['n_samples'] for row in result['cells'])}"
    )

    print("Invariants structurels : contrôlés dans chaque cellule.")

    if args.json_out:
        export_json(
            output_path=args.json_out,
            metadata=result["metadata"],
            cells=result["cells"],
            by_eta=result["by_eta"],
            by_init_seed=result["by_init_seed"],
            by_nk_seed=result["by_nk_seed"],
            interactions=result["interactions"],
        )

    print()
    print("AUDIT FACTORIEL TERMINÉ.")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("Audit interrompu par l'utilisateur.")

    except Exception as exc:
        print()
        print("ÉCHEC DE L'AUDIT FACTORIEL")
        print(f"{type(exc).__name__} : {exc}")
        raise
