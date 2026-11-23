"""
src/fiber_bundle.py -- Projet NK-LSTM-PATH
Transport discret symétrique (convention stricte post-flip), chemins sur la
face (1,2), commutateur, accessibilité A_k(J), oracle analytique exact N=3.
"""
from __future__ import annotations
from typing import Callable, Iterable, Tuple
import numpy as np

PsiFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _validate_state(X) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("L'état X doit être un vecteur 1D non vide.")
    if not np.all(np.isin(arr, (-1.0, 1.0))):
        raise ValueError(f"Spins hors de {{-1,+1}} : {arr}")
    return arr


def _validate_J(J, atol: float = 1e-12) -> np.ndarray:
    mat = np.asarray(J, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] < 2:
        raise ValueError("J doit être carrée, N >= 2.")
    if not np.allclose(mat, mat.T, atol=atol, rtol=0.0):
        raise ValueError("J doit être symétrique.")
    if not np.allclose(np.diag(mat), 0.0, atol=atol, rtol=0.0):
        raise ValueError("Diagonale de J non nulle.")
    return mat


def pack_J(J) -> np.ndarray:
    mat = _validate_J(J)
    return mat[np.triu_indices(mat.shape[0], k=1)].copy()


def unpack_J(j_vec, n: int = 3) -> np.ndarray:
    vec = np.asarray(list(j_vec), dtype=np.float64)
    expected = n * (n - 1) // 2
    if vec.ndim != 1 or vec.size != expected:
        raise ValueError(f"Attendu {expected} composantes pour N={n}, reçu {vec.size}.")
    mat = np.zeros((n, n))
    mat[np.triu_indices(n, k=1)] = vec
    return mat + mat.T


def psi_sym(X, J) -> np.ndarray:
    """Psi_ij = 1/2 sum_{r not in {i,j}} (J_jr + J_ir) s_r ; Psi_ii = 0."""
    s = _validate_state(X)
    m = _validate_J(J)
    if s.size != m.shape[0]:
        raise ValueError("Dimensions incompatibles entre X et J.")
    w = m @ s
    psi = 0.5 * ((w[:, None] + w[None, :]) - m * (s[:, None] + s[None, :]))
    np.fill_diagonal(psi, 0.0)
    return psi


def psi_integrable(X, J) -> np.ndarray:
    """Contrôle intégrable : Psi_ij = s_i s_j ; Psi_ii = 0."""
    s = _validate_state(X)
    m = _validate_J(J)
    if s.size != m.shape[0]:
        raise ValueError("Dimensions incompatibles entre X et J.")
    psi = np.outer(s, s)
    np.fill_diagonal(psi, 0.0)
    return psi


def mutate_postflip(X, J, k: int, eta: float, psi_fn: PsiFn = psi_sym):
    s = _validate_state(X)
    m = _validate_J(J)
    n = s.size
    if m.shape[0] != n:
        raise ValueError("Dimensions incompatibles entre X et J.")
    if not (0 <= k < n):
        raise IndexError(f"k={k} hors limites pour N={n}.")
    X_next = s.copy()
    X_next[k] = -X_next[k]
    psi = _validate_J(psi_fn(X_next, m))
    J_next = m.copy()
    for j in range(n):
        if j != k:
            v = m[k, j] + eta * psi[k, j]
            J_next[k, j] = v
            J_next[j, k] = v
    return X_next, J_next


def apply_path(X, J, nodes: Iterable[int], eta: float, psi_fn: PsiFn = psi_sym):
    cX, cJ = _validate_state(X), _validate_J(J)
    for k in nodes:
        cX, cJ = mutate_postflip(cX, cJ, k=k, eta=eta, psi_fn=psi_fn)
    return cX, cJ


def path_12(X, J, eta, psi_fn: PsiFn = psi_sym):
    return apply_path(X, J, (0, 1), eta, psi_fn)


def path_21(X, J, eta, psi_fn: PsiFn = psi_sym):
    return apply_path(X, J, (1, 0), eta, psi_fn)


def commutator(X, J, eta, k=0, m=1, psi_fn: PsiFn = psi_sym) -> np.ndarray:
    """Delta J = J^(k->m) - J^(m->k). Les spins finaux doivent coïncider."""
    XA, JA = apply_path(X, J, (k, m), eta, psi_fn)
    XB, JB = apply_path(X, J, (m, k), eta, psi_fn)
    if not np.array_equal(XA, XB):
        raise AssertionError("Spins finaux différents entre les deux chemins.")
    return JA - JB


def accessibility_A(J) -> np.ndarray:
    m = _validate_J(J)
    return np.sum(m * m, axis=1)


def accessibility_set(J, theta_A: float) -> np.ndarray:
    return accessibility_A(J) > float(theta_A)


def accessibility_margin(J, theta_A: float) -> np.ndarray:
    return np.abs(accessibility_A(J) - float(theta_A))


def analytic_commutator_N3(X0, J0, eta) -> dict:
    """Oracle analytique exact N=3, face (1,2), post-flip."""
    s = _validate_state(X0)
    m = _validate_J(J0)
    if s.size != 3:
        raise ValueError("N = 3 uniquement.")
    s1, s2, s3 = (float(v) for v in s)
    a, b, c = (float(v) for v in pack_J(m))
    h = float(eta)
    q = 0.25 * h * h

    jA = np.array([
        a + h * (b + c) * s3 + q * (a + c) * s2 * s3,
        b + 0.5 * h * (a + c) * s2,
        c - 0.5 * h * (a + b) * s1 - q * s1 * ((a + c) * s2 + (b + c) * s3),
    ])
    jB = np.array([
        a + h * (b + c) * s3 + q * (a + b) * s1 * s3,
        b - 0.5 * h * (a + c) * s2 - q * s2 * ((a + b) * s1 + (b + c) * s3),
        c + 0.5 * h * (a + b) * s1,
    ])
    dj = np.array([
        q * s3 * ((a + c) * s2 - (a + b) * s1),
        h * (a + c) * s2 + q * s2 * ((a + b) * s1 + (b + c) * s3),
        -h * (a + b) * s1 - q * s1 * ((a + c) * s2 + (b + c) * s3),
    ])
    # coefficients dominants : DeltaJ12/eta^2, DeltaJ13/eta, DeltaJ23/eta
    lead_coef = np.array([
        0.25 * s3 * ((a + c) * s2 - (a + b) * s1),
        (a + c) * s2,
        -(a + b) * s1,
    ])
    dA3_exact = dj[1] * (jA[1] + jB[1]) + dj[2] * (jA[2] + jB[2])
    dA3_first = 2.0 * h * (b * (a + c) * s2 - c * (a + b) * s1)
    return {
        "j_A_exact": jA,
        "j_B_exact": jB,
        "delta_j_exact": dj,
        "leading_coef": lead_coef,
        "delta_A3_exact": float(dA3_exact),
        "delta_A3_first_order": float(dA3_first),
    }
