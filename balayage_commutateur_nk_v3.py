"""
balayage_commutateur_nk_v3.py
Projet : NK-LSTM-PATH

BALAYAGE DU DÉFAUT DE COMMUTATION DISCRÈTE

Ordre A :
    M_m o M_k

Ordre B :
    M_k o M_m

Chaîne d'observables :
    (X, J) -> A(J) -> w(X, J)

Modules utilisés :
    src/fiber_bundle.py
    src/trajectories.py

Le dossier src/ est ajouté automatiquement au chemin d'importation.
Le script peut donc être lancé depuis la racine du dépôt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple
import sys

import numpy as np


# ================================================================
# 0. CHEMIN DES MODULES DU PROJET
# ================================================================

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"

# Dans le dépôt, fiber_bundle.py et trajectories.py résident dans src/.
# Le test is_dir() conserve aussi la compatibilité avec une disposition
# locale où ces modules seraient directement à côté de ce script.
if _SRC_DIR.is_dir():
    src_path = str(_SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


from fiber_bundle import (  # noqa: E402
    mutate_postflip,
    accessibility_A,
    _validate_state,
    _validate_J,
)

from trajectories import (  # noqa: E402
    NKLandscape,
    transition_rates,
)


# ================================================================
# 1. CONFIGURATION EXPÉRIMENTALE
# ================================================================

@dataclass(frozen=True)
class ExperimentConfig:

    # Dimension du système
    n_sites: int

    # Nombre d'états initiaux
    n_seeds: int

    # Graine des états initiaux
    random_seed: int

    # Échelle de la matrice J initiale
    scale_J: float

    # Paramètres du transport et de la transition
    eta: float
    mu: float
    beta: float
    theta_A: float
    lam: float

    # Paramètre épistatique NK
    nk_K: int

    # Graine du paysage NK
    nk_seed: int

    # Tolérances numériques
    rtol: float = 1e-12
    atol: float = 1e-14
    tol_w: float = 1e-12

    # Comparaison des paires non ordonnées
    unordered_pairs_only: bool = True


# ================================================================
# 2. SAISIE DES PARAMÈTRES
# ================================================================

def prompt_float(label: str) -> float:
    """
    Demande un réel fini dans le terminal.
    Accepte le point ou la virgule comme séparateur décimal.
    """

    while True:

        raw = input(f"{label} : ").strip()
        raw = raw.replace(",", ".")

        try:
            value = float(raw)
        except ValueError:
            print("Valeur invalide. Entrer un nombre.")
            continue

        if not np.isfinite(value):
            print("La valeur doit être finie.")
            continue

        return value


def prompt_int(label: str) -> int:
    """Demande un entier dans le terminal."""

    while True:

        raw = input(f"{label} : ").strip()

        try:
            value = int(raw)
        except ValueError:
            print("Valeur invalide. Entrer un entier.")
            continue

        return value


def collect_config() -> ExperimentConfig:
    """
    Collecte les paramètres qui ne sont pas définis dans le code.

    Les valeurs structurelles du balayage initial sont conservées :
        N = 6
        n_seeds = 10
        random_seed = 7
        scale_J = 0.2

    Les paramètres physiques et NK sont demandés au lancement.
    """

    print()
    print("=" * 66)
    print("NK-LSTM-PATH — CONFIGURATION DU BALAYAGE")
    print("=" * 66)

    print()
    print("Paramètres structurels conservés :")
    print("N = 6")
    print("Nombre de graines = 10")
    print("Graine initiale = 7")
    print("Échelle initiale de J = 0.2")

    print()
    print("Entrer les paramètres du protocole expérimental :")
    print("Utiliser les valeurs déjà validées dans le projet.")
    print()

    eta = prompt_float("eta — intensité du transport")
    mu = prompt_float("mu — échelle des taux")
    beta = prompt_float("beta — sensibilité à la fitness")
    theta_A = prompt_float("theta_A — seuil d'accessibilité")
    lam = prompt_float("lam — pondération du couplage")
    nk_K = prompt_int("K — nombre de dépendances épistatiques NK")

    cfg = ExperimentConfig(
        n_sites=6,
        n_seeds=10,
        random_seed=7,
        scale_J=0.2,
        eta=eta,
        mu=mu,
        beta=beta,
        theta_A=theta_A,
        lam=lam,
        nk_K=nk_K,
        nk_seed=7,
    )

    validate_config(cfg)
    return cfg


# ================================================================
# 3. VALIDATION DE LA CONFIGURATION
# ================================================================

def validate_config(
    cfg: ExperimentConfig,
) -> None:

    if cfg.n_sites < 2:
        raise ValueError(
            "n_sites doit être supérieur ou égal à 2."
        )

    if cfg.n_seeds < 1:
        raise ValueError(
            "n_seeds doit être positif."
        )

    if not np.isfinite(cfg.scale_J) or cfg.scale_J < 0.0:
        raise ValueError(
            "scale_J doit être fini et positif ou nul."
        )

    if cfg.nk_K < 0 or cfg.nk_K >= cfg.n_sites:
        raise ValueError(
            "K doit satisfaire 0 <= K < N."
        )

    parameters = {
        "eta": cfg.eta,
        "mu": cfg.mu,
        "beta": cfg.beta,
        "theta_A": cfg.theta_A,
        "lam": cfg.lam,
    }

    for name, value in parameters.items():
        if not np.isfinite(value):
            raise ValueError(
                f"Le paramètre {name} doit être fini."
            )

    if cfg.mu < 0.0:
        raise ValueError(
            "mu doit être supérieur ou égal à zéro."
        )

    tolerances = {
        "rtol": cfg.rtol,
        "atol": cfg.atol,
        "tol_w": cfg.tol_w,
    }

    for name, value in tolerances.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} doit être fini et positif ou nul."
            )


# ================================================================
# 4. ADAPTATEUR DES TAUX DE TRANSITION
# ================================================================

def build_transition_adapter(
    cfg: ExperimentConfig,
    fnk_fn: Callable[[np.ndarray], float],
):
    """
    Adapte la fonction native :

        transition_rates(
            X, J, mu, beta, theta_A, lam, fnk_fn
        )

    à l'interface du balayage :

        transition_adapter(X, J)

    fnk_fn peut être une fonction de fitness ou l'objet NKLandscape,
    qui est lui-même appelable et expose delta_fitness_flip.
    """

    def transition_adapter(
        X: np.ndarray,
        J: np.ndarray,
    ) -> np.ndarray:

        rates = transition_rates(
            X=X,
            J=J,
            mu=cfg.mu,
            beta=cfg.beta,
            theta_A=cfg.theta_A,
            lam=cfg.lam,
            fnk_fn=fnk_fn,
        )

        return np.asarray(
            rates,
            dtype=np.float64,
        )

    return transition_adapter


# ================================================================
# 5. UTILITAIRES GÉOMÉTRIQUES
# ================================================================

def matrix_symmetry_error(
    J: np.ndarray,
) -> float:
    """Norme de Frobenius de J - J.T."""

    return float(
        np.linalg.norm(
            J - J.T,
            ord="fro",
        )
    )


def matrix_diagonal_error(
    J: np.ndarray,
) -> float:
    """Écart maximal de la diagonale à zéro."""

    return float(
        np.max(
            np.abs(np.diag(J))
        )
    )


def validate_state_pair(
    X: np.ndarray,
    J: np.ndarray,
    n_sites: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validation par les fonctions natives de fiber_bundle.py."""

    Xv = _validate_state(X)
    Jv = _validate_J(J)

    if Xv.shape != (n_sites,):
        raise ValueError(
            f"Dimension de X invalide : {Xv.shape}."
        )

    if Jv.shape != (n_sites, n_sites):
        raise ValueError(
            f"Dimension de J invalide : {Jv.shape}."
        )

    return Xv, Jv


# ================================================================
# 6. STRUCTURES DE RÉSULTATS
# ================================================================

@dataclass
class PairOrderRecord:

    seed_idx: int
    k: int
    m: int

    same_X_endpoint: bool
    same_full_endpoint_exact: bool
    same_full_endpoint_close: bool

    symmetry_error_A: float
    symmetry_error_B: float

    diagonal_error_A: float
    diagonal_error_B: float

    non_commut_gap_fro: float

    delta_A_norm: float
    delta_w_norm: float
    delta_w_max: float

    # Nom historique : compte les désaccords entre masques finaux,
    # pas des événements temporels de franchissement.
    threshold_crossing_count: int


@dataclass
class SymmetricOrderSweepSummary:

    n_samples: int

    mean_non_commut_gap_fro: float
    max_non_commut_gap_fro: float

    max_symmetry_error: float
    max_diagonal_error: float

    frac_same_X_endpoint: float
    frac_same_full_endpoint_exact: float
    frac_same_full_endpoint_close: float

    mean_delta_A_norm: float
    max_delta_A_norm: float

    mean_delta_w_norm: float
    max_delta_w_norm: float

    # Fraction des comparaisons où max(abs(w_A - w_B)) > tol_w.
    frac_nonzero_delta_w: float

    # Nom historique : somme des désaccords site par site des masques finaux.
    total_threshold_crossings: int

    records: List[PairOrderRecord]


# ================================================================
# 7. COMPARAISON DES DEUX ORDRES DE MUTATION
# ================================================================

def compare_symmetric_mutation_order(
    X0: np.ndarray,
    J0: np.ndarray,
    k: int,
    m: int,
    eta: float,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> Dict[str, object]:
    """
    Compare :

        Ordre A : M_m(M_k(X0, J0))
        Ordre B : M_k(M_m(X0, J0))

    Les deux chemins utilisent exactement mutate_postflip.
    """

    if k == m:
        raise ValueError(
            "Les sites k et m doivent être distincts."
        )

    # ------------------------------------------------------------
    # ORDRE A : k puis m
    # ------------------------------------------------------------

    XA, JA = mutate_postflip(
        X0.copy(),
        J0.copy(),
        int(k),
        eta,
    )

    XA, JA = mutate_postflip(
        XA,
        JA,
        int(m),
        eta,
    )

    # ------------------------------------------------------------
    # ORDRE B : m puis k
    # ------------------------------------------------------------

    XB, JB = mutate_postflip(
        X0.copy(),
        J0.copy(),
        int(m),
        eta,
    )

    XB, JB = mutate_postflip(
        XB,
        JB,
        int(k),
        eta,
    )

    XA = np.asarray(XA)
    XB = np.asarray(XB)

    JA = np.asarray(
        JA,
        dtype=np.float64,
    )

    JB = np.asarray(
        JB,
        dtype=np.float64,
    )

    if XA.shape != XB.shape:
        raise ValueError(
            "Les dimensions des états X finaux diffèrent."
        )

    if JA.shape != JB.shape:
        raise ValueError(
            "Les dimensions des matrices J finales diffèrent."
        )

    if not np.all(np.isfinite(JA)):
        raise ValueError(
            "JA contient des valeurs non finies."
        )

    if not np.all(np.isfinite(JB)):
        raise ValueError(
            "JB contient des valeurs non finies."
        )

    same_X = bool(
        np.array_equal(XA, XB)
    )

    same_exact = bool(
        same_X and np.array_equal(JA, JB)
    )

    same_close = bool(
        same_X
        and np.allclose(
            JA,
            JB,
            rtol=rtol,
            atol=atol,
        )
    )

    gap = JA - JB

    return {
        "XA": XA,
        "XB": XB,
        "JA": JA,
        "JB": JB,
        "same_X_endpoint": same_X,
        "same_full_endpoint_exact": same_exact,
        "same_full_endpoint_close": same_close,
        "sym_err_A": matrix_symmetry_error(JA),
        "sym_err_B": matrix_symmetry_error(JB),
        "diag_err_A": matrix_diagonal_error(JA),
        "diag_err_B": matrix_diagonal_error(JB),
        "gap_matrix": gap,
        "non_commut_gap_fro": float(
            np.linalg.norm(
                gap,
                ord="fro",
            )
        ),
    }


# ================================================================
# 8. TRANSMISSION DES OBSERVABLES
# ================================================================

def evaluate_observable_transmission(
    XA: np.ndarray,
    JA: np.ndarray,
    XB: np.ndarray,
    JB: np.ndarray,
    transition_fn: Callable,
    theta_A: float,
) -> Tuple[float, float, float, int]:
    """
    Évalue :

        J -> A(J) -> w(X, J)

    Mesures :
        ||A_A - A_B||
        ||w_A - w_B||
        max |w_A - w_B|

    Le compteur compare les masques binaires finaux
    (A_k > theta_A). Il ne compte pas d'événements temporels.
    """

    A_A = np.asarray(
        accessibility_A(JA),
        dtype=np.float64,
    )

    A_B = np.asarray(
        accessibility_A(JB),
        dtype=np.float64,
    )

    if A_A.shape != A_B.shape:
        raise ValueError(
            "Les dimensions de A_A et A_B diffèrent."
        )

    w_A = np.asarray(
        transition_fn(XA, JA),
        dtype=np.float64,
    )

    w_B = np.asarray(
        transition_fn(XB, JB),
        dtype=np.float64,
    )

    if w_A.shape != w_B.shape:
        raise ValueError(
            "Les dimensions de w_A et w_B diffèrent."
        )

    for name, value in (
        ("A_A", A_A),
        ("A_B", A_B),
        ("w_A", w_A),
        ("w_B", w_B),
    ):

        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"{name} contient des valeurs non finies."
            )

    delta_A = A_A - A_B
    delta_w = w_A - w_B

    active_A = A_A > theta_A
    active_B = A_B > theta_A

    mask_disagreements = int(
        np.sum(active_A != active_B)
    )

    return (
        float(np.linalg.norm(delta_A)),
        float(np.linalg.norm(delta_w)),
        float(np.max(np.abs(delta_w))),
        mask_disagreements,
    )


# ================================================================
# 9. GÉNÉRATION DES PAIRES DE SITES
# ================================================================

def iter_site_pairs(
    n_sites: int,
    unordered_only: bool = True,
) -> Iterable[Tuple[int, int]]:
    """Génère les paires distinctes de sites."""

    for k in range(n_sites):

        start_m = (
            k + 1
            if unordered_only
            else 0
        )

        for m in range(start_m, n_sites):

            if k != m:
                yield k, m


# ================================================================
# 10. BALAYAGE MULTI-GRAINES
# ================================================================

def sweep_symmetric_order(
    initial_states: List[Tuple[np.ndarray, np.ndarray]],
    eta: float,
    transition_fn: Callable,
    theta_A: float,
    unordered_pairs_only: bool = True,
    tol_w: float = 1e-12,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> SymmetricOrderSweepSummary:
    """
    Exécute le balayage des deux ordres de mutation.

    La fraction Delta w non nulle est définie composante par composante :

        max(abs(w_A - w_B)) > tol_w

    Cette définition est partagée avec audit_robustesse_eta.py.
    """

    for name, value in (
        ("tol_w", tol_w),
        ("rtol", rtol),
        ("atol", atol),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} doit être fini et positif ou nul."
            )

    records: List[PairOrderRecord] = []

    for seed_idx, (X0_raw, J0_raw) in enumerate(initial_states):

        X0 = _validate_state(X0_raw)
        J0 = _validate_J(J0_raw)

        n_sites = X0.size

        if J0.shape != (n_sites, n_sites):
            raise ValueError(
                "Dimensions incompatibles dans un état initial."
            )

        for k, m in iter_site_pairs(
            n_sites,
            unordered_only=unordered_pairs_only,
        ):

            res = compare_symmetric_mutation_order(
                X0=X0,
                J0=J0,
                k=k,
                m=m,
                eta=eta,
                rtol=rtol,
                atol=atol,
            )

            dA, dw, dw_max, mask_disagreements = (
                evaluate_observable_transmission(
                    XA=res["XA"],
                    JA=res["JA"],
                    XB=res["XB"],
                    JB=res["JB"],
                    transition_fn=transition_fn,
                    theta_A=theta_A,
                )
            )

            records.append(
                PairOrderRecord(
                    seed_idx=seed_idx,
                    k=k,
                    m=m,
                    same_X_endpoint=res["same_X_endpoint"],
                    same_full_endpoint_exact=res[
                        "same_full_endpoint_exact"
                    ],
                    same_full_endpoint_close=res[
                        "same_full_endpoint_close"
                    ],
                    symmetry_error_A=res["sym_err_A"],
                    symmetry_error_B=res["sym_err_B"],
                    diagonal_error_A=res["diag_err_A"],
                    diagonal_error_B=res["diag_err_B"],
                    non_commut_gap_fro=res[
                        "non_commut_gap_fro"
                    ],
                    delta_A_norm=dA,
                    delta_w_norm=dw,
                    delta_w_max=dw_max,
                    threshold_crossing_count=mask_disagreements,
                )
            )

    if not records:
        raise RuntimeError(
            "Aucune comparaison n'a été effectuée."
        )

    # ------------------------------------------------------------
    # VECTEURS STATISTIQUES
    # ------------------------------------------------------------

    gaps = np.array(
        [r.non_commut_gap_fro for r in records],
        dtype=np.float64,
    )

    sym_errs = np.array(
        [
            max(
                r.symmetry_error_A,
                r.symmetry_error_B,
            )
            for r in records
        ],
        dtype=np.float64,
    )

    diag_errs = np.array(
        [
            max(
                r.diagonal_error_A,
                r.diagonal_error_B,
            )
            for r in records
        ],
        dtype=np.float64,
    )

    same_x = np.array(
        [r.same_X_endpoint for r in records],
        dtype=bool,
    )

    full_exact = np.array(
        [r.same_full_endpoint_exact for r in records],
        dtype=bool,
    )

    full_close = np.array(
        [r.same_full_endpoint_close for r in records],
        dtype=bool,
    )

    dA_vals = np.array(
        [r.delta_A_norm for r in records],
        dtype=np.float64,
    )

    dw_vals = np.array(
        [r.delta_w_norm for r in records],
        dtype=np.float64,
    )

    # Critère composante par composante, identique à l'audit eta.
    dw_max_vals = np.array(
        [r.delta_w_max for r in records],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(dw_max_vals)):
        raise RuntimeError(
            "Les écarts maximaux de taux contiennent une valeur non finie."
        )

    mask_disagreement_vals = np.array(
        [r.threshold_crossing_count for r in records],
        dtype=np.int64,
    )

    # ------------------------------------------------------------
    # SYNTHÈSE
    # ------------------------------------------------------------

    return SymmetricOrderSweepSummary(
        n_samples=len(records),
        mean_non_commut_gap_fro=float(
            np.mean(gaps)
        ),
        max_non_commut_gap_fro=float(
            np.max(gaps)
        ),
        max_symmetry_error=float(
            np.max(sym_errs)
        ),
        max_diagonal_error=float(
            np.max(diag_errs)
        ),
        frac_same_X_endpoint=float(
            np.mean(same_x)
        ),
        frac_same_full_endpoint_exact=float(
            np.mean(full_exact)
        ),
        frac_same_full_endpoint_close=float(
            np.mean(full_close)
        ),
        mean_delta_A_norm=float(
            np.mean(dA_vals)
        ),
        max_delta_A_norm=float(
            np.max(dA_vals)
        ),
        mean_delta_w_norm=float(
            np.mean(dw_vals)
        ),
        max_delta_w_norm=float(
            np.max(dw_vals)
        ),
        frac_nonzero_delta_w=float(
            np.mean(dw_max_vals > tol_w)
        ),
        total_threshold_crossings=int(
            np.sum(mask_disagreement_vals)
        ),
        records=records,
    )


# ================================================================
# 11. GÉNÉRATION DES ÉTATS INITIAUX
# ================================================================

def sample_binary_symmetric_states(
    n_sites: int,
    n_seeds: int,
    rng: np.random.Generator,
    scale: float = 0.2,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Génère :
        X0 in {-1, +1}^N
        J0 symétrique à diagonale nulle.

    scale est l'écart-type des coefficients de la matrice brute avant
    symétrisation. Pour des coefficients bruts indépendants de même
    écart-type, les coefficients hors diagonale de J0 ont donc un
    écart-type égal à scale / sqrt(2).
    """

    if n_sites < 2:
        raise ValueError("n_sites doit être >= 2.")
    if n_seeds < 1:
        raise ValueError("n_seeds doit être >= 1.")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("scale doit être fini et positif ou nul.")

    states = []

    for _ in range(n_seeds):

        X0 = rng.choice(
            np.array([-1, 1], dtype=np.int8),
            size=(n_sites,),
        )

        raw_J = rng.normal(
            0.0,
            scale,
            size=(n_sites, n_sites),
        )

        J0 = 0.5 * (
            raw_J + raw_J.T
        )

        np.fill_diagonal(
            J0,
            0.0,
        )

        states.append(
            (X0, J0)
        )

    return states


# ================================================================
# 12. AFFICHAGE DES RÉSULTATS
# ================================================================

def print_summary(
    summary: SymmetricOrderSweepSummary,
) -> None:

    print()
    print("=" * 68)
    print("RÉSULTATS — BALAYAGE DE NON-COMMUTATIVITÉ")
    print("=" * 68)

    print(
        f"Nombre de comparaisons       : "
        f"{summary.n_samples}"
    )

    print(
        f"Écart moyen ||JA-JB||F       : "
        f"{summary.mean_non_commut_gap_fro:.12e}"
    )

    print(
        f"Écart maximal ||JA-JB||F     : "
        f"{summary.max_non_commut_gap_fro:.12e}"
    )

    print(
        f"Erreur maximale symétrie     : "
        f"{summary.max_symmetry_error:.12e}"
    )

    print(
        f"Erreur maximale diagonale    : "
        f"{summary.max_diagonal_error:.12e}"
    )

    print(
        f"Fraction X identique         : "
        f"{summary.frac_same_X_endpoint:.6f}"
    )

    print(
        f"Fraction endpoints exacts    : "
        f"{summary.frac_same_full_endpoint_exact:.6f}"
    )

    print(
        f"Fraction endpoints proches   : "
        f"{summary.frac_same_full_endpoint_close:.6f}"
    )

    print(
        f"Delta A moyen                : "
        f"{summary.mean_delta_A_norm:.12e}"
    )

    print(
        f"Delta A maximal              : "
        f"{summary.max_delta_A_norm:.12e}"
    )

    print(
        f"Delta w moyen                : "
        f"{summary.mean_delta_w_norm:.12e}"
    )

    print(
        f"Delta w maximal              : "
        f"{summary.max_delta_w_norm:.12e}"
    )

    print(
        f"Fraction Delta w non nul     : "
        f"{summary.frac_nonzero_delta_w:.6f}"
    )

    print(
        f"Désaccords masques finaux    : "
        f"{summary.total_threshold_crossings}"
    )

    print("=" * 68)
    print("Définition Delta w non nul : max(abs(w_A - w_B)) > tol_w.")
    print("Désaccords masques finaux : comparaison site par site de A_k > theta_A.")
    print("Ce compteur ne représente pas des franchissements temporels du seuil.")


# ================================================================
# 13. EXÉCUTION PRINCIPALE
# ================================================================

def main() -> SymmetricOrderSweepSummary:
    """
    Initialise un paysage NK unique puis exécute le balayage.
    """

    cfg = collect_config()
    validate_config(cfg)

    # ------------------------------------------------------------
    # Générateur des états initiaux
    # ------------------------------------------------------------

    rng = np.random.default_rng(
        cfg.random_seed
    )

    initial_states = sample_binary_symmetric_states(
        n_sites=cfg.n_sites,
        n_seeds=cfg.n_seeds,
        rng=rng,
        scale=cfg.scale_J,
    )

    # ------------------------------------------------------------
    # Paysage NK unique pour toute l'expérience
    # ------------------------------------------------------------

    nk_landscape = NKLandscape(
        N=cfg.n_sites,
        K=cfg.nk_K,
        seed=cfg.nk_seed,
    )

    # On transmet l'objet NKLandscape, pas seulement sa méthode fitness.
    # transition_rates peut ainsi appeler delta_fitness_flip directement.
    transition_adapter = build_transition_adapter(
        cfg,
        fnk_fn=nk_landscape,
    )

    # ------------------------------------------------------------
    # Nombre de comparaisons attendues
    # ------------------------------------------------------------

    if cfg.unordered_pairs_only:

        n_pairs = (
            cfg.n_sites * (cfg.n_sites - 1)
        ) // 2

    else:

        n_pairs = (
            cfg.n_sites * (cfg.n_sites - 1)
        )

    expected = (
        len(initial_states) * n_pairs
    )

    # ------------------------------------------------------------
    # Présentation du protocole exécuté
    # ------------------------------------------------------------

    print()
    print("=" * 68)
    print("NK-LSTM-PATH — DÉMARRAGE DE L'EXPÉRIENCE")
    print("=" * 68)

    print(f"Nombre de sites N             : {cfg.n_sites}")
    print(f"Nombre de graines             : {cfg.n_seeds}")
    print(f"Nombre de paires par graine   : {n_pairs}")
    print(f"Comparaisons attendues        : {expected}")

    print()
    print("Paramètres utilisés :")
    print(f"eta                            : {cfg.eta}")
    print(f"mu                             : {cfg.mu}")
    print(f"beta                           : {cfg.beta}")
    print(f"theta_A                        : {cfg.theta_A}")
    print(f"lam                            : {cfg.lam}")
    print(f"K NK                           : {cfg.nk_K}")
    print(f"Graine du paysage NK           : {cfg.nk_seed}")

    print()
    print("Le paysage NK est maintenu identique")
    print("pour toutes les comparaisons.")
    print()

    # ------------------------------------------------------------
    # Exécution
    # ------------------------------------------------------------

    summary = sweep_symmetric_order(
        initial_states=initial_states,
        eta=cfg.eta,
        transition_fn=transition_adapter,
        theta_A=cfg.theta_A,
        unordered_pairs_only=cfg.unordered_pairs_only,
        tol_w=cfg.tol_w,
        rtol=cfg.rtol,
        atol=cfg.atol,
    )

    # ------------------------------------------------------------
    # Contrôle du nombre de comparaisons
    # ------------------------------------------------------------

    if summary.n_samples != expected:
        raise RuntimeError(
            f"Nombre de comparaisons incorrect : "
            f"{summary.n_samples} au lieu de {expected}."
        )

    print_summary(summary)
    return summary


# ================================================================
# 14. POINT D'ENTRÉE
# ================================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print()
        print("Expérience interrompue par l'utilisateur.")

    except Exception as exc:
        print()
        print("ÉCHEC DE L'EXPÉRIENCE")
        print(f"{type(exc).__name__} : {exc}")
        raise
