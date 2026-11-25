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

Protocole structurel
--------------------
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
d'accessibilité finaux des deux ordres. Il ne compte pas des événements
temporels.

La fraction Delta w non nul est calculée composante par composante :
une comparaison est déclarée non nulle si max(abs(w_A - w_B)) > tol_w.
Cette convention est cohérente avec l'audit multi-graines.

Aucune valeur numérique par défaut n'est attribuée à mu, beta, theta_A,
lam ou K : ExperimentConfig ne définit pas de valeurs par défaut pour
ces paramètres. Ils sont donc demandés explicitement au lancement.
"""

from __future__ import annotations

from dataclasses import dataclass
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

# Tolérance absolue réservée aux invariants structurels de J.
# Elle est volontairement indépendante de rtol/atol des calculs numériques.
STRUCTURE_ATOL = 1e-12

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

    # Fraction des comparaisons pour lesquelles au moins une
    # composante de w diffère de plus de tol_w.
    frac_nonzero_delta_w: float

    # Somme des désaccords site par site entre les masques finaux.
    active_mask_disagreements: int

    @property
    def total_crossings(self) -> int:
        """Alias historique ; ce n'est pas un nombre de franchissements temporels."""
        return self.active_mask_disagreements


# ================================================================
# 3. VALIDATION DES ENTRÉES
# ================================================================

def finite_real(name: str, value: float) -> float:
    """Convertit un scalaire en float et refuse les booléens, NaN et infinis."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} doit être un réel, pas un booléen.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} doit être un réel.") from exc

    if not np.isfinite(result):
        raise ValueError(f"{name} doit être fini.")

    return result


def validate_eta_grid(eta_values: Sequence[float]) -> List[float]:
    """
    Valide la grille de eta.

    - Au moins une valeur.
    - Toutes les valeurs sont réelles et finies.
    - Pas de doublons.
    - L'ordre fourni est conservé.

    Les valeurs négatives sont admises : eta est ici traité comme un
    paramètre de transport signé. Cette convention doit rester celle
    du protocole physique du projet.
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
    Les décimales doivent utiliser le point, car la virgule sépare
    les valeurs de la grille.
    """

    default_text = ", ".join(str(value) for value in DEFAULT_ETA_GRID)

    while True:
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

    Important : ExperimentConfig ne définit aucun défaut pour mu,
    beta, theta_A, lam ou nk_K. Ces valeurs sont donc demandées
    explicitement au moyen des fonctions prompt_float/prompt_int
    du moteur, avec leur signature réelle (un seul argument).
    """

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
    print("Entrer les valeurs retenues dans le protocole de référence.")
    print("Aucune valeur par défaut n'est définie pour ces paramètres.")

    # Signatures conformes à balayage_commutateur_nk_v3.py.
    mu = prompt_float("mu — échelle des taux")
    beta = prompt_float("beta — sensibilité à la fitness")
    theta_A = prompt_float("theta_A — seuil d'accessibilité")
    lam = prompt_float("lam — pondération du couplage")
    nk_K = prompt_int("K — dépendances épistatiques NK")

    return eta_values, mu, beta, theta_A, lam, nk_K


# ================================================================
# 6. CONSTRUCTION D'UNE CONFIGURATION ETA
# ================================================================

def make_eta_config(
    eta: float,
    mu: float,
    beta: float,
    theta_A: float,
    lam: float,
    nk_K: int,
) -> ExperimentConfig:
    """Construit une configuration ; seul eta varie entre les passages."""

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
# 7. MÉTRIQUES COMPLÉMENTAIRES ET INVARIANTS
# ================================================================

def fraction_nonzero_delta_w(
    records: Sequence,
    tol_w: float,
) -> float:
    """
    Fraction de comparaisons où au moins une composante des taux diffère
    de plus de tol_w.

    Le critère max(abs(delta_w)) > tol_w est cohérent avec l'audit
    multi-graines. Il ne doit pas être confondu avec ||delta_w||_2 > tol_w.
    """

    if not records:
        raise ValueError("Impossible de calculer Delta w sans enregistrements.")

    maxima = np.asarray(
        [record.delta_w_max for record in records],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(maxima)):
        raise RuntimeError("Les écarts maximaux de taux contiennent une valeur non finie.")

    return float(np.mean(maxima > tol_w))


def validate_sweep_summary(
    summary: SymmetricOrderSweepSummary,
    cfg: ExperimentConfig,
    expected_samples: int,
) -> None:
    """Vérifie les invariants géométriques, les compteurs et le cas eta=0."""

    if summary.n_samples != expected_samples:
        raise RuntimeError(
            "Nombre de comparaisons incorrect : "
            f"{summary.n_samples}, attendu : {expected_samples}."
        )

    if len(summary.records) != expected_samples:
        raise RuntimeError(
            "Le nombre d'enregistrements ne correspond pas au nombre attendu : "
            f"{len(summary.records)} au lieu de {expected_samples}."
        )

    if summary.n_samples != len(summary.records):
        raise RuntimeError(
            "Incohérence entre summary.n_samples et summary.records."
        )

    numeric_metrics = {
        "écart moyen de J": summary.mean_non_commut_gap_fro,
        "écart maximal de J": summary.max_non_commut_gap_fro,
        "erreur maximale de symétrie": summary.max_symmetry_error,
        "erreur maximale de diagonale": summary.max_diagonal_error,
        "fraction X identique": summary.frac_same_X_endpoint,
        "fraction endpoints exacts": summary.frac_same_full_endpoint_exact,
        "fraction endpoints proches": summary.frac_same_full_endpoint_close,
        "différence moyenne A": summary.mean_delta_A_norm,
        "différence maximale A": summary.max_delta_A_norm,
        "différence moyenne w": summary.mean_delta_w_norm,
        "différence maximale w": summary.max_delta_w_norm,
    }

    for name, value in numeric_metrics.items():
        if not np.isfinite(float(value)):
            raise RuntimeError(f"Métrique non finie : {name} = {value}.")

    nonnegative_metrics = (
        "mean_non_commut_gap_fro",
        "max_non_commut_gap_fro",
        "max_symmetry_error",
        "max_diagonal_error",
        "mean_delta_A_norm",
        "max_delta_A_norm",
        "mean_delta_w_norm",
        "max_delta_w_norm",
    )
    for name in nonnegative_metrics:
        if float(getattr(summary, name)) < 0.0:
            raise RuntimeError(f"Métrique négative inattendue : {name}.")

    fraction_metrics = (
        "frac_same_X_endpoint",
        "frac_same_full_endpoint_exact",
        "frac_same_full_endpoint_close",
    )
    for name in fraction_metrics:
        value = float(getattr(summary, name))
        if not (0.0 <= value <= 1.0):
            raise RuntimeError(f"Fraction hors de [0, 1] : {name} = {value}.")

    if not all(record.same_X_endpoint for record in summary.records):
        raise RuntimeError(
            "Invariant violé : au moins une paire d'ordres produit un X final différent."
        )

    if summary.frac_same_X_endpoint != 1.0:
        raise RuntimeError(
            "Invariant violé : les deux ordres ne produisent pas "
            "systématiquement le même état X final."
        )

    if summary.total_threshold_crossings < 0:
        raise RuntimeError("Le compteur de désaccords de masques est négatif.")

    record_disagreements = sum(
        int(record.threshold_crossing_count)
        for record in summary.records
    )
    if record_disagreements != summary.total_threshold_crossings:
        raise RuntimeError(
            "Incohérence du compteur de désaccords entre la synthèse "
            "et les enregistrements individuels."
        )

    # Tolérance absolue dédiée aux invariants structurels de J.
    if summary.max_symmetry_error > STRUCTURE_ATOL:
        raise RuntimeError(
            "La symétrie de J n'est pas conservée : "
            f"{summary.max_symmetry_error:.6e} > {STRUCTURE_ATOL:.1e}."
        )

    if summary.max_diagonal_error > STRUCTURE_ATOL:
        raise RuntimeError(
            "La diagonale de J n'est pas nulle : "
            f"{summary.max_diagonal_error:.6e} > {STRUCTURE_ATOL:.1e}."
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
            if abs(float(value)) > STRUCTURE_ATOL:
                raise RuntimeError(
                    f"Invariant eta=0 violé : {name} = {value:.6e}."
                )

        if summary.frac_same_full_endpoint_exact != 1.0:
            raise RuntimeError(
                "À eta=0, les états finaux complets devraient être "
                "exactement identiques."
            )

        if summary.frac_same_full_endpoint_close != 1.0:
            raise RuntimeError(
                "À eta=0, tous les états finaux devraient aussi être proches."
            )

        if summary.total_threshold_crossings != 0:
            raise RuntimeError(
                "À eta=0, aucun désaccord de masque actif ne devrait être observé."
            )

        if fraction_nonzero_delta_w(summary.records, cfg.tol_w) != 0.0:
            raise RuntimeError(
                "À eta=0, aucune comparaison ne devrait présenter "
                "de différence de taux supérieure à tol_w."
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
      - la même instance NKLandscape ;
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

    # Configuration de référence pour les paramètres communs et les
    # conventions par défaut du moteur (tolérances, paires non ordonnées).
    reference_cfg = make_eta_config(
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

    if nk_landscape.N != N_SITES:
        raise RuntimeError("Dimension incorrecte du paysage NK.")

    if nk_landscape.K != nk_K:
        raise RuntimeError("Nombre de dépendances NK incorrect.")

    # ------------------------------------------------------------
    # Nombre attendu de comparaisons
    # ------------------------------------------------------------

    if reference_cfg.unordered_pairs_only:
        pairs_per_state = N_SITES * (N_SITES - 1) // 2
    else:
        pairs_per_state = N_SITES * (N_SITES - 1)

    expected_samples = N_INITIAL_STATES * pairs_per_state

    # ------------------------------------------------------------
    # Balayage de eta
    # ------------------------------------------------------------

    rows: List[EtaAuditRow] = []

    for eta_index, eta_value in enumerate(eta_grid, start=1):
        cfg = make_eta_config(
            eta=eta_value,
            mu=mu,
            beta=beta,
            theta_A=theta_A,
            lam=lam,
            nk_K=nk_K,
        )

        print()
        print(f"[ETA {eta_index}/{len(eta_grid)}] eta = {cfg.eta:g}")

        # On transmet l'objet NKLandscape afin que transition_rates
        # puisse reconnaître le paysage canonique et utiliser, lorsque
        # disponible, son calcul spécialisé de delta_fitness_flip.
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

        validate_sweep_summary(
            summary=summary,
            cfg=cfg,
            expected_samples=expected_samples,
        )

        # Recalcul de la fraction selon le critère composante par
        # composante, commun à l'audit multi-graines.
        frac_delta_w = fraction_nonzero_delta_w(
            summary.records,
            cfg.tol_w,
        )

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
            frac_nonzero_delta_w=frac_delta_w,
            active_mask_disagreements=summary.total_threshold_crossings,
        )

        rows.append(row)

        print(f"  Comparaisons : {row.n_samples}/{expected_samples}")
        if cfg.eta == 0.0:
            print("  Contrôle eta=0 : invariants vérifiés.")

    return rows


# ================================================================
# 9. AFFICHAGE DES RÉSULTATS
# ================================================================

def print_audit_table(rows: Sequence[EtaAuditRow]) -> None:
    """Affiche les métriques comparatives de robustesse sur eta."""

    if not rows:
        raise ValueError("Aucun résultat à afficher.")

    print()
    print("=" * 190)
    print("NK-LSTM-PATH — ROBUSTESSE À L'INTENSITÉ DE TRANSPORT")
    print("=" * 190)

    header = (
        f"{'eta':>7} | "
        f"{'N_comp':>6} | "
        f"{'Gap moy J':>12} | "
        f"{'Gap max J':>12} | "
        f"{'Err sym':>10} | "
        f"{'Err diag':>10} | "
        f"{'X id.':>7} | "
        f"{'J exact':>8} | "
        f"{'J proche':>8} | "
        f"{'Delta A moy':>12} | "
        f"{'Delta A max':>12} | "
        f"{'Delta w moy':>12} | "
        f"{'Delta w max':>12} | "
        f"{'Delta w != 0':>13} | "
        f"{'Désaccords':>11}"
    )

    print(header)
    print("-" * 190)

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
            f"{100.0 * row.frac_same_full_close:7.1f}% | "
            f"{row.mean_delta_A:12.4e} | "
            f"{row.max_delta_A:12.4e} | "
            f"{row.mean_delta_w:12.4e} | "
            f"{row.max_delta_w:12.4e} | "
            f"{100.0 * row.frac_nonzero_delta_w:12.1f}% | "
            f"{row.active_mask_disagreements:11d}"
        )

    print("=" * 190)
    print()
    print("INTERPRÉTATION DES COLONNES")
    print("Gap J       : norme de Frobenius de JA - JB, moyenne et maximum.")
    print("Delta A     : norme de la différence des accessibilités finales.")
    print("Delta w     : norme de la différence des vecteurs de taux finaux.")
    print("Delta w != 0: fraction où max(abs(w_A - w_B)) > tol_w.")
    print("X id.       : proportion des comparaisons avec X_A = X_B.")
    print("J exact     : proportion des extrémités complètes exactement égales.")
    print("J proche    : proportion des extrémités complètes égales aux tolérances près.")
    print(
        "Désaccords  : somme des désaccords site par site entre les masques "
        "d'accessibilité finaux."
    )
    print("Ce compteur ne représente pas des franchissements temporels du seuil.")


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
    print(f"Tolérance structurelle J  : {STRUCTURE_ATOL:.1e}")

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
