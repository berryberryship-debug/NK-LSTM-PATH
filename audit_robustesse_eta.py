
"""
audit_robustesse_eta.py
Projet : NK-LSTM-PATH

AUDIT DE ROBUSTESSE SUR L'INTENSITÉ DE TRANSPORT ETA

Objectif
--------
Mesurer l'effet de eta sur la non-commutativité de deux mutations
successives, en maintenant les autres paramètres expérimentaux fixes.

Réutilise
---------
fiber_bundle.py
trajectories.py
balayage_commutateur_nk_v3.py

Protocole
---------
N = 6
Nombre de configurations initiales = 10
Graine des configurations initiales = 7
Échelle initiale de J = 0.2
Graine du paysage NK = 7

Le même ensemble de configurations initiales et la même instance
NKLandscape sont réutilisés pour toutes les valeurs de eta.

Les deux ordres comparés sont :

    A : M_m o M_k
    B : M_k o M_m

Les observables sont évaluées sur les états finaux :

    (X, J) -> A(J) -> w(X, J)

Attention
---------
active_mask_disagreements compte les désaccords entre les masques
d'accessibilité finaux des deux ordres.

Ce compteur ne représente pas le nombre de franchissements
temporels du seuil.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Sequence

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


# ================================================================
# 1. CONFIGURATION STRUCTURELLE FIXE
# ================================================================

N_SITES = 6
N_INITIAL_STATES = 10

INITIAL_STATES_SEED = 7
INITIAL_J_SCALE = 0.2

NK_SEED = 7

DEFAULT_ETA_GRID = (
    0.0,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)


# ================================================================
# 2. STRUCTURE DES RÉSULTATS
# ================================================================

@dataclass(frozen=True)
class EtaAuditRow:
    eta: float

    n_samples: int

    mean_gap_fro: float
    max_gap_fro: float

    max_sym_err: float
    max_diag_err: float

    frac_same_X: float
    frac_same_full_exact: float
    frac_same_full_close: float

    mean_delta_A: float
    max_delta_A: float

    mean_delta_w: float
    max_delta_w: float

    frac_nonzero_delta_w: float

    active_mask_disagreements: int

    @property
    def total_crossings(self) -> int:
        """
        Alias conservé pour compatibilité avec les anciens usages.

        Ce champ désigne les désaccords entre masques actifs finaux,
        et non des franchissements temporels.
        """
        return self.active_mask_disagreements


# ================================================================
# 3. VALIDATION DES ENTRÉES
# ================================================================

def finite_real(name: str, value: float) -> float:
    """Convertit un scalaire en float et refuse NaN et infini."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} doit être un réel, pas un booléen.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} doit être un réel.") from exc

    if not np.isfinite(result):
        raise ValueError(f"{name} doit être fini.")

    return result


def validate_eta_grid(
    eta_values: Sequence[float],
) -> List[float]:
    """
    Valide la grille de eta.

    - Au moins une valeur.
    - Toutes les valeurs sont réelles et finies.
    - Pas de doublons.
    - L'ordre fourni est conservé.

    Les valeurs négatives ne sont pas interdites ici :
    eta est traité comme un paramètre de transport signé.
    """

    if isinstance(eta_values, (str, bytes)):
        raise TypeError("La grille eta doit être une séquence numérique.")

    try:
        raw_values = list(eta_values)
    except TypeError as exc:
        raise TypeError("La grille eta doit être itérable.") from exc

    if not raw_values:
        raise ValueError("La grille eta ne peut pas être vide.")

    values: List[float] = []

    for index, raw in enumerate(raw_values):
        value = finite_real(f"eta[{index}]", raw)

        if value in values:
            raise ValueError(
                f"Valeur eta dupliquée : {value}. "
                "Chaque valeur doit apparaître une seule fois."
            )

        values.append(value)

    return values


def validate_audit_arguments(
    mu: float,
    beta: float,
    theta_A: float,
    lam: float,
    nk_K: int,
) -> tuple[float, float, float, float, int]:
    """Valide les paramètres communs à toute la grille eta."""

    mu = finite_real("mu", mu)
    beta = finite_real("beta", beta)
    theta_A = finite_real("theta_A", theta_A)
    lam = finite_real("lam", lam)

    if isinstance(nk_K, (bool, np.bool_)) or not isinstance(
        nk_K, (int, np.integer)
    ):
        raise TypeError("nk_K doit être un entier.")

    nk_K = int(nk_K)

    if mu < 0.0:
        raise ValueError("mu doit être >= 0.")

    if not (0 <= nk_K < N_SITES):
        raise ValueError(
            f"nk_K doit vérifier 0 <= K < {N_SITES}."
        )

    return mu, beta, theta_A, lam, nk_K


# ================================================================
# 4. SAISIE DE LA GRILLE ETA
# ================================================================

def parse_eta_grid_input() -> List[float]:
    """
    Demande une liste de eta séparés par des virgules,
    des espaces ou des points-virgules.

    Entrée vide : grille par défaut.
    """

    while True:
        default_text = ", ".join(
            str(value) for value in DEFAULT_ETA_GRID
        )

        raw = input(
            f"Valeurs de eta à auditer [défaut : {default_text}] : "
        ).strip()

        if not raw:
            return list(DEFAULT_ETA_GRID)

        cleaned = raw.replace(";", " ").replace(",", " ")
        tokens = cleaned.split()

        try:
            values = validate_eta_grid(
                [float(token) for token in tokens]
            )
        except (TypeError, ValueError) as exc:
            print(f"Grille invalide : {exc}")
            continue

        return values


# ================================================================
# 5. COLLECTE DES PARAMÈTRES
# ================================================================

def collect_audit_parameters(
) -> tuple[List[float], float, float, float, float, int]:
    """
    Collecte eta et les paramètres fixes.

    Les valeurs par défaut proviennent de ExperimentConfig,
    afin de conserver le protocole de référence.
    """

    reference = ExperimentConfig()

    print()
    print("=" * 76)
    print("NK-LSTM-PATH — AUDIT DE ROBUSTESSE SUR ETA")
    print("=" * 76)

    print()
    print("PARAMÈTRES STRUCTURELS FIXES")
    print(f"N                         : {N_SITES}")
    print(f"Configurations initiales  : {N_INITIAL_STATES}")
    print(f"Graine initiale           : {INITIAL_STATES_SEED}")
    print(f"Échelle initiale de J     : {INITIAL_J_SCALE}")
    print(f"Graine du paysage NK      : {NK_SEED}")

    print()
    print("GRILLE DE TRANSPORT")

    eta_values = parse_eta_grid_input()

    print()
    print("PARAMÈTRES FIXES DU MODÈLE")
    print("(Entrée conserve la valeur de référence.)")

    mu = prompt_float(
        "mu — échelle des taux",
        default=reference.mu,
    )

    beta = prompt_float(
        "beta — sensibilité à la fitness",
        default=reference.beta,
    )

    theta_A = prompt_float(
        "theta_A — seuil d'accessibilité",
        default=reference.theta_A,
    )

    lam = prompt_float(
        "lam — pondération du couplage",
        default=reference.lam,
    )

    nk_K = prompt_int(
        "K — dépendances épistatiques NK",
        default=reference.nk_K,
    )

    return eta_values, mu, beta, theta_A, lam, nk_K


# ================================================================
# 6. CONFIGURATION D'UNE VALEUR ETA
# ================================================================

def make_eta_config(
    eta: float,
    mu: float,
    beta: float,
    theta_A: float,
    lam: float,
    nk_K: int,
) -> ExperimentConfig:
    """Construit une configuration expérimentale pour une valeur de eta."""

    cfg = ExperimentConfig(
        n_sites=N_SITES,
        n_seeds=N_INITIAL_STATES,
        random_seed=INITIAL_STATES_SEED,
        scale_J=INITIAL_J_SCALE,

        eta=eta,
        mu=mu,
        beta=beta,
        theta_A=theta_A,
        lam=lam,

        nk_K=nk_K,
        nk_seed=NK_SEED,
    )

    validate_config(cfg)

    return cfg


# ================================================================
# 7. VÉRIFICATION DES INVARIANTS
# ================================================================

def validate_sweep_summary(
    summary: SymmetricOrderSweepSummary,
    cfg: ExperimentConfig,
    expected_samples: int,
) -> None:
    """
    Vérifie les invariants algorithmiques du balayage.

    Pour deux mutations distinctes :
      - X final doit être identique dans les deux ordres ;
      - J doit rester symétrique ;
      - la diagonale de J doit rester nulle.

    À eta = 0 :
      - les deux chemins doivent être identiques ;
      - les différences géométriques et les différences de taux
        doivent être nulles ;
      - les masques actifs doivent être identiques.
    """

    if summary.n_samples != expected_samples:
        raise RuntimeError(
            "Nombre de comparaisons incorrect : "
            f"{summary.n_samples}, attendu : {expected_samples}."
        )

    if not np.isclose(
        summary.frac_same_X_endpoint,
        1.0,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError(
            "Invariant violé : les deux ordres ne produisent pas "
            "le même état X final."
        )

    structure_tol = max(
        1e-12,
        10.0 * cfg.atol,
        10.0 * cfg.rtol,
    )

    if summary.max_symmetry_error > structure_tol:
        raise RuntimeError(
            "La symétrie de J n'est pas conservée : "
            f"{summary.max_symmetry_error:.6e}."
        )

    if summary.max_diagonal_error > structure_tol:
        raise RuntimeError(
            "La diagonale de J n'est pas nulle : "
            f"{summary.max_diagonal_error:.6e}."
        )

    if cfg.eta == 0.0:
        zero_metrics = {
            "écart moyen de J": summary.mean_non_commut_gap_fro,
            "écart maximal de J": summary.max_non_commut_gap_fro,
            "différence moyenne A": summary.mean_delta_A_norm,
            "différence maximale A": summary.max_delta_A_norm,
            "différence moyenne w": summary.mean_delta_w_norm,
            "différence maximale w": summary.max_delta_w_norm,
        }

        for name, value in zero_metrics.items():
            if abs(float(value)) > structure_tol:
                raise RuntimeError(
                    f"Invariant eta=0 violé : {name} = "
                    f"{value:.6e}."
                )

        if summary.frac_same_full_endpoint_exact != 1.0:
            raise RuntimeError(
                "À eta=0, les états finaux complets devraient "
                "être exactement identiques."
            )

        if summary.total_threshold_crossings != 0:
            raise RuntimeError(
                "À eta=0, aucun désaccord de masque actif "
                "ne devrait être observé."
            )


# ================================================================
# 8. AUDIT PRINCIPAL
# ================================================================

def run_eta_robustness_audit(
    eta_values: Sequence[float],
    mu: float,
    beta: float,
    theta_A: float,
    lam: float,
    nk_K: int,
) -> List[EtaAuditRow]:
    """
    Exécute le balayage de commutation sur plusieurs valeurs de eta.

    Toutes les valeurs de eta utilisent :
      - les mêmes configurations initiales ;
      - le même paysage NK ;
      - les mêmes paramètres mu, beta, theta_A et lambda ;
      - les mêmes tolérances numériques.

    Seul eta varie.
    """

    eta_grid = validate_eta_grid(eta_values)

    mu, beta, theta_A, lam, nk_K = validate_audit_arguments(
        mu=mu,
        beta=beta,
        theta_A=theta_A,
        lam=lam,
        nk_K=nk_K,
    )

    # ------------------------------------------------------------
    # Configuration de référence et validation préalable
    # ------------------------------------------------------------

    base_cfg = make_eta_config(
        eta=eta_grid[0],
        mu=mu,
        beta=beta,
        theta_A=theta_A,
        lam=lam,
        nk_K=nk_K,
    )

    # ------------------------------------------------------------
    # Génération unique des configurations initiales
    # ------------------------------------------------------------

    rng = np.random.default_rng(INITIAL_STATES_SEED)

    initial_states = sample_binary_symmetric_states(
        n_sites=N_SITES,
        n_seeds=N_INITIAL_STATES,
        rng=rng,
        scale=INITIAL_J_SCALE,
    )

    if len(initial_states) != N_INITIAL_STATES:
        raise RuntimeError(
            "Nombre incorrect de configurations initiales générées."
        )

    # ------------------------------------------------------------
    # Construction unique du paysage NK canonique
    # ------------------------------------------------------------

    nk_landscape = NKLandscape(
        N=N_SITES,
        K=nk_K,
        seed=NK_SEED,
    )

    # Contrôle dimensionnel du paysage.
    if nk_landscape.N != N_SITES:
        raise RuntimeError(
            "Dimension incorrecte du paysage NK."
        )

    if nk_landscape.K != nk_K:
        raise RuntimeError(
            "Nombre de dépendances NK incorrect."
        )

    # ------------------------------------------------------------
    # Nombre attendu de comparaisons
    # ------------------------------------------------------------

    if base_cfg.unordered_pairs_only:
        pairs_per_state = N_SITES * (N_SITES - 1) // 2
    else:
        pairs_per_state = N_SITES * (N_SITES - 1)

    expected_samples = N_INITIAL_STATES * pairs_per_state

    # ------------------------------------------------------------
    # Balayage de eta
    # ------------------------------------------------------------

    rows: List[EtaAuditRow] = []

    for eta_index, eta_value in enumerate(eta_grid, start=1):

        cfg = replace(
            base_cfg,
            eta=eta_value,
        )

        validate_config(cfg)

        print()
        print(
            f"[ETA {eta_index}/{len(eta_grid)}] "
            f"eta = {cfg.eta:g}"
        )

        # IMPORTANT :
        # On transmet l'objet NKLandscape, pas sa méthode .fitness.
        # Cela conserve l'accès à delta_fitness_flip().
        transition_adapter = build_transition_adapter(
            cfg,
            fnk_fn=nk_landscape,
        )

        summary: SymmetricOrderSweepSummary = sweep_symmetric_order(
            initial_states=initial_states,
            eta=cfg.eta,
            transition_fn=transition_adapter,
            theta_A=cfg.theta_A,
            unordered_pairs_only=cfg.unordered_pairs_only,
            tol_w=cfg.tol_w,
            rtol=cfg.rtol,
            atol=cfg.atol,
        )

        # --------------------------------------------------------
        # Validation des résultats
        # --------------------------------------------------------

        validate_sweep_summary(
            summary=summary,
            cfg=cfg,
            expected_samples=expected_samples,
        )

        # --------------------------------------------------------
        # Enregistrement des métriques
        # --------------------------------------------------------

        row = EtaAuditRow(
            eta=float(cfg.eta),

            n_samples=summary.n_samples,

            mean_gap_fro=summary.mean_non_commut_gap_fro,
            max_gap_fro=summary.max_non_commut_gap_fro,

            max_sym_err=summary.max_symmetry_error,
            max_diag_err=summary.max_diagonal_error,

            frac_same_X=summary.frac_same_X_endpoint,
            frac_same_full_exact=summary.frac_same_full_endpoint_exact,
            frac_same_full_close=summary.frac_same_full_endpoint_close,

            mean_delta_A=summary.mean_delta_A_norm,
            max_delta_A=summary.max_delta_A_norm,

            mean_delta_w=summary.mean_delta_w_norm,
            max_delta_w=summary.max_delta_w_norm,

            frac_nonzero_delta_w=summary.frac_nonzero_delta_w,

            active_mask_disagreements=summary.total_threshold_crossings,
        )

        rows.append(row)

        print(
            f"  Comparaisons : {row.n_samples}/{expected_samples}"
        )

        if cfg.eta == 0.0:
            print(
                "  Contrôle eta=0 : invariants vérifiés."
            )

    return rows


# ================================================================
# 9. AFFICHAGE DES RÉSULTATS
# ================================================================

def print_audit_table(
    rows: Sequence[EtaAuditRow],
) -> None:
    """Affiche les métriques comparatives de robustesse sur eta."""

    if not rows:
        raise ValueError("Aucun résultat à afficher.")

    print()
    print("=" * 174)
    print("NK-LSTM-PATH — ROBUSTESSE À L'INTENSITÉ DE TRANSPORT")
    print("=" * 174)

    header = (
        f"{'eta':>7} | "
        f"{'N_comp':>6} | "
        f"{'Gap moy J':>12} | "
        f"{'Gap max J':>12} | "
        f"{'Err sym':>10} | "
        f"{'Err diag':>10} | "
        f"{'X id.':>7} | "
        f"{'J exact':>8} | "
        f"{'Delta A moy':>12} | "
        f"{'Delta A max':>12} | "
        f"{'Delta w moy':>12} | "
        f"{'Delta w max':>12} | "
        f"{'Delta w != 0':>13} | "
        f"{'Désaccords':>11}"
    )

    print(header)
    print("-" * 174)

    for row in rows:
        print(
            f"{row.eta:7.3g} | "
            f"{row.n_samples:6d} | "
            f"{row.mean_gap_fro:12.4e} | "
            f"{row.max_gap_fro:12.4e} | "
            f"{row.max_sym_err:10.2e} | "
            f"{row.max_diag_err:10.2e} | "
            f"{100.0 * row.frac_same_X:6.1f}% | "
            f"{100.0 * row.frac_same_full_exact:7.1f}% | "
            f"{row.mean_delta_A:12.4e} | "
            f"{row.max_delta_A:12.4e} | "
            f"{row.mean_delta_w:12.4e} | "
            f"{row.max_delta_w:12.4e} | "
            f"{100.0 * row.frac_nonzero_delta_w:12.1f}% | "
            f"{row.active_mask_disagreements:11d}"
        )

    print("=" * 174)

    print()
    print("INTERPRÉTATION DES COLONNES")
    print("Gap J       : norme de Frobenius moyenne/maximale de JA - JB.")
    print("Delta A     : norme de la différence des accessibilités finales.")
    print("Delta w     : norme de la différence des taux finaux.")
    print("X id.       : proportion des configurations avec X_A = X_B.")
    print("J exact     : proportion des extrémités complètes exactement égales.")
    print("Désaccords  : nombre cumulé de désaccords entre masques actifs finaux.")
    print()
    print("Le compteur de désaccords n'est pas un nombre de franchissements temporels.")


# ================================================================
# 10. EXÉCUTION PRINCIPALE
# ================================================================

def main() -> List[EtaAuditRow]:
    """Point d'entrée interactif de l'audit."""

    (
        eta_values,
        mu,
        beta,
        theta_A,
        lam,
        nk_K,
    ) = collect_audit_parameters()

    print()
    print("=" * 76)
    print("PROTOCOLE EXPÉRIMENTAL")
    print("=" * 76)

    print(f"N                         : {N_SITES}")
    print(f"Configurations initiales  : {N_INITIAL_STATES}")
    print(f"Graine initiale           : {INITIAL_STATES_SEED}")
    print(f"Échelle initiale de J     : {INITIAL_J_SCALE}")
    print(f"Graine NK                 : {NK_SEED}")
    print(f"K                         : {nk_K}")
    print(f"mu                        : {mu}")
    print(f"beta                      : {beta}")
    print(f"theta_A                   : {theta_A}")
    print(f"lambda                    : {lam}")
    print(f"Grille eta                : {eta_values}")

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


# ================================================================
# 11. POINT D'ENTRÉE
# ================================================================

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
