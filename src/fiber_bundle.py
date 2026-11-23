"""
src/fiber_bundle.py
Projet : NK-LSTM-PATH

Modélisation géométrique de l'espace fibré discret sur {-1, +1}^N x Sym_0(N) :
  - Validation des configurations de spins X in {-1, +1}^N et des matrices
    de couplage symétriques à diagonale nulle J in Sym_0(N),
  - Empaquetage / dépaquetage du triangle supérieur (`pack_J`, `unpack_J`),
  - Calcul de l'accessibilité locale quadratique A_k(J) = sum_{j != k} J_{kj}^2,
  - Opérateur de transport discret symétrique `psi_sym` :
        Psi_{ij}(X, J) = (1 / 2) * sum_{r != i, j} (J_{jr} + J_{ir}) s_r  (i != j),
        Psi_{ii}(X, J) = 0,
  - Mutation avec transport post-flip `mutate_postflip` : basculement strict
    du spin s_k -> -s_k avant l'évaluation de Psi(X^{(k)}, J) et mise à jour
    des seules arêtes incidentes au nœud muté :
        J'_{kj} = J'_{jk} = J_{kj} + eta * Psi_{kj}(X^{(k)}, J)  (j != k),
    les arêtes non incidentes demeurant strictement invariantes.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple
import numpy as np

PsiFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _validate_state(X: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """
    Valide et normalise un vecteur d'état binaire X in {-1, +1}^N (N >= 2).
    """
    state = np.asarray(X, dtype=np.float64)
    if state.ndim != 1:
        raise ValueError(f"L'état X doit être un vecteur 1D, reçu ndim={state.ndim}.")
    if state.size < 2:
        raise ValueError(f"La dimension N doit être >= 2, reçu N={state.size}.")
    if not np.all(np.isfinite(state)):
        raise ValueError("L'état X contient des valeurs non finies.")
    if not np.all(np.isclose(np.abs(state), 1.0, atol=atol)):
        raise ValueError("Chaque composante de X doit appartenir à {-1, +1}.")
    return np.where(state >= 0.0, 1.0, -1.0)


def _validate_J(J: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """
    Valide et renvoie une copie propre d'une matrice de couplage symétrique
    à diagonale nulle J in R^{N x N} (N >= 2).
    """
    mat = np.asarray(J, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"J doit être une matrice carrée 2D, reçu shape={mat.shape}.")
    n = mat.shape[0]
    if n < 2:
        raise ValueError(f"La dimension N de J doit être >= 2, reçu N={n}.")
    if not np.all(np.isfinite(mat)):
        raise ValueError("La matrice J contient des valeurs non finies.")
    if not np.allclose(mat, mat.T, atol=atol):
        raise ValueError("La matrice de couplage J doit être symétrique.")
    if not np.allclose(np.diag(mat), 0.0, atol=atol):
        raise ValueError("La diagonale de la matrice J doit être nulle.")

    clean = 0.5 * (mat + mat.T)
    np.fill_diagonal(clean, 0.0)
    return clean


def unpack_J(couplings: Sequence[float], n: int) -> np.ndarray:
    """
    Construit une matrice symétrique N x N à diagonale nulle à partir des
    N * (N - 1) / 2 composantes du triangle supérieur strict (ordre lexicographique
    par lignes : (0, 1), ..., (0, N-1), (1, 2), ..., (N-2, N-1)).
    """
    n_int = int(n)
    if n_int < 2:
        raise ValueError(f"N doit être >= 2, reçu N={n_int}.")
    vec = np.asarray(couplings, dtype=np.float64).ravel()
    expected = n_int * (n_int - 1) // 2
    if vec.size != expected:
        raise ValueError(
            f"Longueur attendue {expected} pour N={n_int}, reçu {vec.size}."
        )
    if not np.all(np.isfinite(vec)):
        raise ValueError("Les couplages contiennent des valeurs non finies.")

    mat = np.zeros((n_int, n_int), dtype=np.float64)
    iu = np.triu_indices(n_int, k=1)
    mat[iu] = vec
    mat[(iu[1], iu[0])] = vec
    return mat


def pack_J(J: np.ndarray) -> np.ndarray:
    """
    Extrait les N * (N - 1) / 2 composantes du triangle supérieur strict de J
    dans l'ordre lexicographique par lignes.
    """
    mat = _validate_J(J)
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu].copy()


def accessibility_A(J: np.ndarray) -> np.ndarray:
    """
    Calcule le vecteur des accessibilités locales quadratiques pour k = 0, ..., N-1 :
        A_k(J) = sum_{j != k} J_{kj}^2.
    """
    mat = _validate_J(J)
    return np.sum(mat * mat, axis=1)


def psi_sym(X: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Opérateur de transport discret symétrique Psi(X, J) in Sym_0(N) :
        Psi_{ij} = (1 / 2) * sum_{r != i, j} (J_{jr} + J_{ir}) s_r  pour i != j,
        Psi_{ii} = 0.

    Utilise l'identité algébrique exacte (pour J_{rr} = 0) :
        sum_{r != i, j} (J_{ir} + J_{jr}) s_r = (J s)_i + (J s)_j - J_{ij} (s_i + s_j).
    """
    state = _validate_state(X)
    mat = _validate_J(J)
    n = state.size
    if mat.shape[0] != n:
        raise ValueError("Dimensions incompatibles entre X et J.")

    field = mat @ state
    psi = 0.5 * (
        field[:, None] + field[None, :] - mat * (state[:, None] + state[None, :])
    )
    psi = 0.5 * (psi + psi.T)
    np.fill_diagonal(psi, 0.0)
    return psi


def mutate_postflip(
    X: np.ndarray,
    J: np.ndarray,
    k: int,
    eta: float,
    psi_fn: PsiFn = psi_sym,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applique une mutation élémentaire sur le nœud k selon la convention post-flip :
      1. Basculement strict du spin muté : X^{(k)}_k = -X_k, X^{(k)}_j = X_j (j != k).
      2. Évaluation de l'opérateur de transport sur l'état post-flip :
             Psi = psi_fn(X^{(k)}, J).
      3. Transport des seules arêtes incidentes au nœud k :
             J'_{kj} = J'_{jk} = J_{kj} + eta * Psi_{kj}  pour j != k,
         les arêtes non incidentes J'_{uv} = J_{uv} (u != k, v != k) restant
         strictement invariantes.
    """
    state = _validate_state(X)
    mat = _validate_J(J)
    n = state.size
    if mat.shape[0] != n:
        raise ValueError("Dimensions incompatibles entre X et J.")
    if not (0 <= k < n):
        raise IndexError(f"Indice de nœud k={k} hors limites pour N={n}.")
    if not np.isfinite(eta):
        raise ValueError("Le pas de transport eta doit être fini.")

    X_next = state.copy()
    X_next[k] = -X_next[k]

    psi = np.asarray(psi_fn(X_next, mat), dtype=np.float64)
    if psi.shape != mat.shape or not np.all(np.isfinite(psi)):
        raise ValueError("L'opérateur psi_fn doit retourner une matrice N x N finie.")

    J_next = mat.copy()
    step = float(eta)
    if step != 0.0:
        incident = np.arange(n) != k
        J_next[k, incident] = mat[k, incident] + step * psi[k, incident]
        J_next[incident, k] = J_next[k, incident]
    np.fill_diagonal(J_next, 0.0)
    return X_next, J_next
