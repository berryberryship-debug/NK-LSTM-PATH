
"""
fiber_bundle.py
Projet : NK-LSTM-PATH

Modélisation géométrique de l'espace fibré discret :

    {-1, +1}^N x Sym_0(N)

Fonctions :
    - Validation des états X et matrices J
    - pack_J et unpack_J
    - Accessibilité locale A_k(J)
    - Transport symétrique psi_sym
    - Mutation avec transport post-flip mutate_postflip
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np


PsiFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ================================================================
# 1. VALIDATION DE L'ÉTAT X
# ================================================================

def _validate_state(
    X: np.ndarray,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Valide et normalise un vecteur binaire :

        X in {-1, +1}^N, N >= 2.
    """

    state = np.asarray(X, dtype=np.float64)

    if state.ndim != 1:
        raise ValueError(
            f"L'état X doit être un vecteur 1D, reçu ndim={state.ndim}."
        )

    if state.size < 2:
        raise ValueError(
            f"La dimension N doit être >= 2, reçu N={state.size}."
        )

    if not np.all(np.isfinite(state)):
        raise ValueError(
            "L'état X contient des valeurs non finies."
        )

    if not np.all(
        np.isclose(np.abs(state), 1.0, atol=atol)
    ):
        raise ValueError(
            "Chaque composante de X doit appartenir à {-1, +1}."
        )

    return np.where(state >= 0.0, 1.0, -1.0)


# ================================================================
# 2. VALIDATION DE LA MATRICE J
# ================================================================

def _validate_J(
    J: np.ndarray,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Valide une matrice de couplage :

        J in Sym_0(N)

    avec J symétrique et diagonale nulle.
    """

    mat = np.asarray(J, dtype=np.float64)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"J doit être une matrice carrée 2D, reçu shape={mat.shape}."
        )

    n = mat.shape[0]

    if n < 2:
        raise ValueError(
            f"La dimension N de J doit être >= 2, reçu N={n}."
        )

    if not np.all(np.isfinite(mat)):
        raise ValueError(
            "La matrice J contient des valeurs non finies."
        )

    if not np.allclose(mat, mat.T, atol=atol):
        raise ValueError(
            "La matrice de couplage J doit être symétrique."
        )

    if not np.allclose(
        np.diag(mat),
        0.0,
        atol=atol,
    ):
        raise ValueError(
            "La diagonale de la matrice J doit être nulle."
        )

    clean = 0.5 * (mat + mat.T)

    np.fill_diagonal(clean, 0.0)

    return clean


# ================================================================
# 3. DÉPAQUETAGE DE J
# ================================================================

def unpack_J(
    couplings: Sequence[float],
    n: int,
) -> np.ndarray:
    """
    Construit une matrice symétrique N x N à diagonale nulle
    à partir de N * (N - 1) / 2 composantes.

    Ordre lexicographique du triangle supérieur strict.
    """

    n_int = int(n)

    if n_int < 2:
        raise ValueError(
            f"N doit être >= 2, reçu N={n_int}."
        )

    vec = np.asarray(
        couplings,
        dtype=np.float64,
    ).ravel()

    expected = n_int * (n_int - 1) // 2

    if vec.size != expected:
        raise ValueError(
            f"Longueur attendue {expected} pour N={n_int}, "
            f"reçu {vec.size}."
        )

    if not np.all(np.isfinite(vec)):
        raise ValueError(
            "Les couplages contiennent des valeurs non finies."
        )

    mat = np.zeros(
        (n_int, n_int),
        dtype=np.float64,
    )

    iu = np.triu_indices(n_int, k=1)

    mat[iu] = vec
    mat[(iu[1], iu[0])] = vec

    return mat


# ================================================================
# 4. EMPAQUETAGE DE J
# ================================================================

def pack_J(J: np.ndarray) -> np.ndarray:
    """
    Extrait les composantes du triangle supérieur strict de J.
    """

    mat = _validate_J(J)

    n = mat.shape[0]

    iu = np.triu_indices(n, k=1)

    return mat[iu].copy()


# ================================================================
# 5. ACCESSIBILITÉ LOCALE
# ================================================================

def accessibility_A(J: np.ndarray) -> np.ndarray:
    """
    Calcule :

        A_k(J) = sum_{j != k} J_{kj}^2
    """

    mat = _validate_J(J)

    return np.sum(
        mat * mat,
        axis=1,
    )


# ================================================================
# 6. OPÉRATEUR DE TRANSPORT SYMÉTRIQUE
# ================================================================

def psi_sym(
    X: np.ndarray,
    J: np.ndarray,
) -> np.ndarray:
    """
    Opérateur de transport discret symétrique :

        Psi(X, J) in Sym_0(N)

    Pour i != j :

        Psi_ij = (1/2) * sum_{r != i,j}
                 (J_jr + J_ir) s_r

    Et :

        Psi_ii = 0

    Identité utilisée :

        sum_{r != i,j} (J_ir + J_jr) s_r
        =
        (J s)_i + (J s)_j - J_ij (s_i + s_j)
    """

    state = _validate_state(X)
    mat = _validate_J(J)

    n = state.size

    if mat.shape[0] != n:
        raise ValueError(
            "Dimensions incompatibles entre X et J."
        )

    field = mat @ state

    psi = 0.5 * (
        field[:, None]
        + field[None, :]
        - mat * (
            state[:, None]
            + state[None, :]
        )
    )

    psi = 0.5 * (
        psi + psi.T
    )

    np.fill_diagonal(
        psi,
        0.0,
    )

    return psi


# ================================================================
# 7. MUTATION AVEC TRANSPORT POST-FLIP
# ================================================================

def mutate_postflip(
    X: np.ndarray,
    J: np.ndarray,
    k: int,
    eta: float,
    psi_fn: PsiFn = psi_sym,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mutation élémentaire sur le site k.

    1. Basculement du spin :

        X^(k)_k = -X_k

    2. Calcul de Psi sur l'état post-flip :

        Psi = psi_fn(X^(k), J)

    3. Actualisation des seules arêtes incidentes à k :

        J'_{kj} = J'_{jk} = J_{kj} + eta * Psi_{kj}

    Les arêtes non incidentes restent invariantes.
    """

    state = _validate_state(X)
    mat = _validate_J(J)

    n = state.size

    if mat.shape[0] != n:
        raise ValueError(
            "Dimensions incompatibles entre X et J."
        )

    if not (0 <= k < n):
        raise IndexError(
            f"Indice de nœud k={k} hors limites pour N={n}."
        )

    if not np.isfinite(eta):
        raise ValueError(
            "Le pas de transport eta doit être fini."
        )

    # 1. Flip du spin
    X_next = state.copy()
    X_next[k] = -X_next[k]

    # 2. Transport post-flip
    psi = np.asarray(
        psi_fn(X_next, mat),
        dtype=np.float64,
    )

    if psi.shape != mat.shape or not np.all(np.isfinite(psi)):
        raise ValueError(
            "L'opérateur psi_fn doit retourner une matrice N x N finie."
        )

    # 3. Mise à jour des arêtes incidentes
    J_next = mat.copy()

    step = float(eta)

    if step != 0.0:
        incident = np.arange(n) != k

        J_next[k, incident] = (
            mat[k, incident]
            + step * psi[k, incident]
        )

        J_next[incident, k] = J_next[k, incident]

    np.fill_diagonal(
        J_next,
        0.0,
    )

    return X_next, J_next
