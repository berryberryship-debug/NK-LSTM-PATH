"""
trajectories.py
Projet : NK-LSTM-PATH

Trajectoires adaptatives sur l'espace d'états couplé :

    X in {-1, +1}^N
    J in Sym_0(N)

Le paysage NK fournit la composante combinatoire de la fitness.
La matrice J porte les couplages géométriques évolutifs.

Chaîne dynamique :

    (X, J)
       -> mutation de X
       -> transport local post-flip de J
       -> accessibilité A(J)
       -> taux de transition w(X, J)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
import os
import sys
from typing import Any, Callable, Sequence

import numpy as np

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
for _path in (_SRC_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fiber_bundle import (  # noqa: E402
    PsiFn,
    _validate_J,
    _validate_state,
    accessibility_A,
    mutate_postflip,
    psi_sym,
)

try:
    from nk_landscape import NKLandscape as _CanonicalNKLandscape  # noqa: E402
except ImportError:  # pragma: no cover
    _CanonicalNKLandscape = None


FitnessFn = Callable[[np.ndarray], float]


# ============================================================
# 0. STRUCTURES DE RÉSULTATS ET VALIDATION
# ============================================================

@dataclass(frozen=True)
class TrajectoryEvent:
    """Événement de saut accepté le long d'une trajectoire."""

    time: float
    node: int
    X: np.ndarray
    J: np.ndarray
    rates: np.ndarray


@dataclass(frozen=True)
class SimulationResult:
    """Résultat complet d'une simulation de trajectoire."""

    events: list[TrajectoryEvent]
    X_final: np.ndarray
    J_final: np.ndarray
    t_final: float

    @property
    def n_accepted(self) -> int:
        return len(self.events)


@dataclass(frozen=True)
class ThresholdCrossingInfo:
    """Racines de traversée de seuil sur [0, 1]."""

    interior_z: np.ndarray
    boundary_z: np.ndarray
    events: list[tuple[float, int, str]]


def _finite_scalar(name: str, value: float) -> float:
    """Convertit un scalaire en float et refuse NaN/inf."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} doit être un réel fini.")
    return value


def _evaluate_fitness(fnk_fn: FitnessFn, X: np.ndarray) -> float:
    """Évalue une fitness NK et vérifie que le résultat est fini."""
    return _finite_scalar("La fitness", fnk_fn(X))


def _validate_pair(
    X: np.ndarray,
    J: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Valide conjointement X et J."""
    Xv = _validate_state(X)
    Jv = _validate_J(J)
    if Jv.shape != (Xv.size, Xv.size):
        raise ValueError(
            f"Dimensions incompatibles : X a {Xv.size} sites, "
            f"J a la forme {Jv.shape}."
        )
    return Xv, Jv


def _draw_unit_uniform(rng: Any) -> float:
    """Tire u in [0, 1) via l'interface `uniform(0.0, 1.0)`."""
    if hasattr(rng, "uniform"):
        u = float(rng.uniform(0.0, 1.0))
    elif hasattr(rng, "random"):
        u = float(rng.random())
    else:  # pragma: no cover
        raise TypeError("rng doit exposer `uniform(0.0, 1.0)` ou `random()`.")
    return min(max(u, 0.0), 1.0 - np.finfo(np.float64).eps)


def _draw_exponential(rng: Any, rate: float) -> float:
    """Tire un temps d'attente Exp(rate) à partir de `uniform(0.0, 1.0)`."""
    u = _draw_unit_uniform(rng)
    return float(-log(1.0 - u) / rate)


def _select_channel_from_rates(
    rng: Any,
    rates: np.ndarray,
    total_rate: float,
) -> int:
    """Sélectionne un canal k selon les taux cumulés avec `side='right'`."""
    rates = np.asarray(rates, dtype=float)
    total_rate = float(total_rate)

    if rates.ndim != 1 or rates.size == 0:
        raise ValueError("rates doit être un tableau unidimensionnel non vide.")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("Les taux doivent être finis et positifs ou nuls.")
    if not np.isfinite(total_rate) or total_rate <= 0.0:
        raise ValueError("total_rate doit être strictement positif et fini.")

    # Tirage directement dans l'intervalle des taux cumulés.
    if hasattr(rng, "uniform"):
        u2 = float(rng.uniform(0.0, total_rate))
    elif hasattr(rng, "random"):
        u2 = float(rng.random()) * total_rate
    else:  # pragma: no cover
        raise TypeError("rng doit exposer `uniform(0.0, total_rate)` ou `random()`.")

    if not np.isfinite(u2):
        raise ValueError("Le tirage du canal doit être fini.")

    # Protection contre les éventuelles bornes numériques.
    u2 = min(max(u2, 0.0), np.nextafter(total_rate, 0.0))
    cumulative = np.cumsum(rates)

    # side='right' écarte les canaux de taux nul lorsque u2 == 0.
    k = int(np.searchsorted(cumulative, u2, side="right"))
    if k >= rates.size:
        positive = np.flatnonzero(rates > 0.0)
        if positive.size == 0:
            raise ValueError("Aucun canal actif.")
        k = int(positive[-1])

    return k


# ============================================================
# 1. PAYSAGE NK UNIFIÉ (DÉLÉGUÉ AU MOTEUR CANONIQUE O(K))
# ============================================================

class NKLandscape:
    """
    Interface unifiée du paysage NK de Kauffman.

    Accepte indistinctement les arguments `(N, K)` ou `(n, k)` et délègue
    au moteur canonique `nk_landscape.NKLandscape.generate` lorsqu'il est
    disponible.
    """

    def __init__(
        self,
        N: int | None = None,
        K: int | None = None,
        seed: int | None = None,
        *,
        n: int | None = None,
        k: int | None = None,
        neighbor_mode: str = "random",
    ) -> None:
        n_val = N if N is not None else n
        k_val = K if K is not None else k

        if n_val is None or k_val is None:
            raise ValueError("Spécifier (N, K) ou (n, k) pour construire NKLandscape.")

        n_int = int(n_val)
        k_int = int(k_val)

        if n_int < 2:
            raise ValueError("N doit être >= 2.")
        if not (0 <= k_int < n_int):
            raise ValueError("K doit satisfaire 0 <= K < N.")

        self.N = n_int
        self.K = k_int
        self.n_spins = n_int
        self.k_epistasis = k_int
        self.seed = seed
        self.neighbor_mode = neighbor_mode

        if _CanonicalNKLandscape is not None:
            canon_seed = 2026 if seed is None else int(seed)
            self._backend = _CanonicalNKLandscape.generate(
                n_spins=n_int,
                k_epistasis=k_int,
                seed=canon_seed,
                neighbor_mode=neighbor_mode,
            )
            self.neighbors = self._backend.neighbors
            self.tables = self._backend.tables
        else:  # pragma: no cover
            self._backend = None
            rng = np.random.default_rng(seed)
            self.neighbors = np.empty((n_int, k_int), dtype=np.int64)
            for i in range(n_int):
                candidates = np.array([j for j in range(n_int) if j != i], dtype=np.int64)
                if k_int > 0:
                    self.neighbors[i] = np.sort(rng.choice(candidates, size=k_int, replace=False))
            self.tables = rng.uniform(0.0, 1.0, size=(n_int, 1 << (k_int + 1)))

    def local_contribution(self, X: np.ndarray, i: int) -> float:
        Xv = _validate_state(X)
        if Xv.size != self.N:
            raise ValueError(f"Taille de X incompatible : {Xv.size} != {self.N}.")
        if not (0 <= i < self.N):
            raise IndexError(f"Indice i hors limites : {i}.")

        if self._backend is not None:
            x_int = np.where(Xv >= 0.0, 1, -1).astype(np.int8)
            return self._backend.local_fitness(x_int, i)

        indices = np.concatenate(([i], self.neighbors[i]))
        bits = ((Xv[indices] + 1.0) / 2.0).astype(np.int8)
        index = 0
        for bit in bits:
            index = (index << 1) | int(bit)
        return float(self.tables[i, index])

    def fitness(self, X: np.ndarray) -> float:
        Xv = _validate_state(X)
        if Xv.size != self.N:
            raise ValueError(f"Taille de X incompatible : {Xv.size} != {self.N}.")
        if self._backend is not None:
            x_int = np.where(Xv >= 0.0, 1, -1).astype(np.int8)
            return self._backend.fitness(x_int)
        total = sum(self.local_contribution(Xv, i) for i in range(self.N))
        return float(total / self.N)

    def delta_fitness_flip(self, X: np.ndarray, k: int) -> float:
        Xv = _validate_state(X)
        if self._backend is not None:
            x_int = np.where(Xv >= 0.0, 1, -1).astype(np.int8)
            return self._backend.delta_fitness_flip(x_int, int(k))
        X_next = Xv.copy()
        X_next[k] *= -1.0
        return float(self.fitness(X_next) - self.fitness(Xv))

    __call__ = fitness


# ============================================================
# 2. FITNESS NK NULLE
# ============================================================

def zero_fnk(X: np.ndarray) -> float:
    """Fitness NK nulle, utile pour isoler le terme de couplage."""
    _validate_state(X)
    return 0.0


# ============================================================
# 3. FITNESS ÉTENDUE F(X, J)
# ============================================================

def extended_fitness(
    X: np.ndarray,
    J: np.ndarray,
    lam: float,
    fnk_fn: FitnessFn = zero_fnk,
) -> float:
    r"""
    Fitness étendue :

        F(X, J) = F_NK(X) + (lambda/2) X^T J X
    """
    X, J = _validate_pair(X, J)
    lam = _finite_scalar("lam", lam)
    f_nk = _evaluate_fitness(fnk_fn, X)
    coupling = 0.5 * float(X @ J @ X)
    return float(f_nk + lam * coupling)


# ============================================================
# 4. VARIATION DE FITNESS POUR UN FLIP
# ============================================================

def delta_fitness_k(
    X: np.ndarray,
    J: np.ndarray,
    k: int,
    lam: float,
    fnk_fn: FitnessFn = zero_fnk,
) -> float:
    r"""
    Variation de fitness lors du flip du site k,
    J étant maintenu fixe :

        Delta F_k =
            Delta F_NK - 2 * lambda * X_k * sum_j(J_kj X_j)
    """
    X, J = _validate_pair(X, J)
    N = X.size
    if not (0 <= k < N):
        raise IndexError(f"Indice k hors limites : {k}.")
    lam = _finite_scalar("lam", lam)

    if isinstance(fnk_fn, NKLandscape):
        delta_nk = fnk_fn.delta_fitness_flip(X, k)
    elif _CanonicalNKLandscape is not None and isinstance(fnk_fn, _CanonicalNKLandscape):
        x_int = np.where(X >= 0.0, 1, -1).astype(np.int8)
        delta_nk = fnk_fn.delta_fitness_flip(x_int, int(k))
    else:
        X_next = X.copy()
        X_next[k] *= -1.0
        delta_nk = _evaluate_fitness(fnk_fn, X_next) - _evaluate_fitness(fnk_fn, X)

    local_field = float(np.dot(J[k], X))
    delta_J = -2.0 * lam * float(X[k]) * local_field
    return float(delta_nk + delta_J)


# ============================================================
# 5. TAUX DE TRANSITION
# ============================================================

def transition_rates(
    X: np.ndarray,
    J: np.ndarray,
    mu: float,
    beta: float,
    theta_A: float,
    lam: float,
    fnk_fn: FitnessFn = zero_fnk,
) -> np.ndarray:
    r"""
    Calcule les taux :

        w_k(X, J) =
            mu * 1_{A_k(J) > theta_A}
            * min(1, exp(beta * Delta F_k))
    """
    X, J = _validate_pair(X, J)
    mu = _finite_scalar("mu", mu)
    beta = _finite_scalar("beta", beta)
    theta_A = _finite_scalar("theta_A", theta_A)
    lam = _finite_scalar("lam", lam)

    if mu < 0.0:
        raise ValueError("mu doit être >= 0.")

    rates = np.zeros(X.size, dtype=np.float64)
    if mu == 0.0:
        return rates

    A = accessibility_A(J)
    active = A > theta_A

    for k in range(X.size):
        if not active[k]:
            continue
        delta = delta_fitness_k(X=X, J=J, k=k, lam=lam, fnk_fn=fnk_fn)
        exponent = beta * delta
        boltzmann = 1.0 if exponent >= 0.0 else float(np.exp(exponent))
        rates[k] = mu * boltzmann

    return rates


# ============================================================
# 6. DISSIPATION EXPONENTIELLE DE J
# ============================================================

def dissipate_J(
    J: np.ndarray,
    J_anchor: np.ndarray,
    dt: float,
    gamma: float,
) -> np.ndarray:
    r"""
    Dissipation exponentielle vers la matrice d'ancrage :

        J(t + dt) =
            J_anchor + (J(t) - J_anchor) * exp(-gamma * dt)
    """
    J = _validate_J(J)
    J_anchor = _validate_J(J_anchor)

    if J.shape != J_anchor.shape:
        raise ValueError("J et J_anchor doivent avoir la même forme.")

    dt = _finite_scalar("dt", dt)
    gamma = _finite_scalar("gamma", gamma)

    if dt < 0.0:
        raise ValueError("dt doit être >= 0.")
    if gamma < 0.0:
        raise ValueError("gamma doit être >= 0.")

    if dt == 0.0 or gamma == 0.0:
        return J.copy()

    decay = float(np.exp(-gamma * dt))
    J_next = J_anchor + (J - J_anchor) * decay
    return _validate_J(J_next)


# ============================================================
# 7. RACINES RÉELLES D'UN POLYNÔME QUADRATIQUE
# ============================================================

def _stable_quadratic_real_roots(
    a: float,
    b: float,
    c: float,
    atol: float = 1e-14,
) -> list[float]:
    """
    Résout a*z^2 + b*z + c = 0 avec mise à l'échelle relative.
    Conserve le degré 2 tant que |a| / S ne sous-déborde pas en binary64.
    """
    a = _finite_scalar("a", a)
    b = _finite_scalar("b", b)
    c = _finite_scalar("c", c)
    atol = _finite_scalar("atol", atol)

    if atol < 0.0:
        raise ValueError("atol doit être >= 0.")

    scale = max(abs(a), abs(b), abs(c))
    if scale == 0.0:
        return []

    a_s = a / scale
    b_s = b / scale
    c_s = c / scale

    tiny_quad = np.finfo(np.float64).tiny
    if abs(a_s) <= tiny_quad:
        if abs(b_s) <= tiny_quad:
            return []
        return [float(-c_s / b_s)]

    discriminant = b_s * b_s - 4.0 * a_s * c_s
    disc_scale = abs(b_s * b_s) + abs(4.0 * a_s * c_s) + 1.0

    if discriminant < 0.0:
        if abs(discriminant) <= atol * disc_scale:
            discriminant = 0.0
        else:
            return []

    sqrt_disc = sqrt(max(0.0, discriminant))
    if sqrt_disc == 0.0:
        return [float(-b_s / (2.0 * a_s))]

    q = -0.5 * (b_s + np.copysign(sqrt_disc, b_s))
    roots = [float(q / a_s)]
    if abs(q) > tiny_quad:
        roots.append(float(c_s / q))

    roots = sorted(r for r in roots if np.isfinite(r))
    unique: list[float] = []
    for r in roots:
        if not unique or not np.isclose(r, unique[-1], rtol=0.0, atol=atol):
            unique.append(r)

    return unique


# ============================================================
# 8. RÉSOLUTION ET CLASSIFICATION DES TRAVERSÉES DE SEUIL
# ============================================================

def _compute_threshold_events_z(
    J0: np.ndarray,
    J_anchor: np.ndarray,
    z0: float = 1.0,
    z1: float = 0.0,
    theta_A: float = 0.0,
    atol: float = 1e-12,
) -> tuple[list[tuple[float, int, str]], np.ndarray]:
    """
    Calcule les événements de traversée et les racines frontières {z1, z0}
    pour A_k(z) = theta_A avec J(z) = J_anchor + (J0 - J_anchor) * z.
    """
    J0 = _validate_J(J0)
    J_anchor = _validate_J(J_anchor)

    if J0.shape != J_anchor.shape:
        raise ValueError("J0 et J_anchor doivent avoir la même forme.")

    z0 = _finite_scalar("z0", z0)
    z1 = _finite_scalar("z1", z1)
    theta_A = _finite_scalar("theta_A", theta_A)
    atol = _finite_scalar("atol", atol)

    if z0 < z1:
        raise ValueError("z0 doit être >= z1.")
    if z1 < 0.0:
        raise ValueError("z1 doit être >= 0.")
    if atol <= 0.0:
        raise ValueError("atol doit être strictement positif.")

    D = J0 - J_anchor
    interior_events: list[tuple[float, int, str]] = []
    boundary_roots: list[float] = []

    for k in range(J0.shape[0]):
        row_anchor = J_anchor[k]
        row_delta = D[k]

        a = float(np.dot(row_delta, row_delta))
        b = float(2.0 * np.dot(row_anchor, row_delta))
        c = float(np.dot(row_anchor, row_anchor) - theta_A)

        roots = _stable_quadratic_real_roots(a, b, c, atol=1e-15)

        for root in roots:
            if root < z1 - atol or root > z0 + atol:
                continue

            if abs(root - z0) <= atol:
                boundary_roots.append(float(z0))
                continue
            if abs(root - z1) <= atol:
                boundary_roots.append(float(z1))
                continue

            z = float(min(z0, max(z1, root)))
            eps = min(max(1e-9, 10.0 * atol), max((z0 - z1) * 0.25, 1e-15))
            zl = max(z1, z - eps)
            zr = min(z0, z + eps)

            poly_l = a * zl * zl + b * zl + c
            poly_r = a * zr * zr + b * zr + c

            if poly_l < 0.0 and poly_r > 0.0:
                kind = "up"
            elif poly_l > 0.0 and poly_r < 0.0:
                kind = "down"
            else:
                kind = "touch"

            interior_events.append((z, k, kind))

    interior_events.sort(key=lambda item: (-item[0], item[1]))
    unique_events: list[tuple[float, int, str]] = []
    for event in interior_events:
        z, k, _kind = event
        if not any(k == pk and abs(z - pz) <= atol for pz, pk, _ in unique_events):
            unique_events.append(event)

    boundary_sorted = sorted(set(boundary_roots), reverse=True)
    unique_bounds: list[float] = []
    for zb in boundary_sorted:
        if not unique_bounds or abs(zb - unique_bounds[-1]) > atol:
            unique_bounds.append(zb)

    return unique_events, np.asarray(unique_bounds, dtype=np.float64)


def classify_threshold_crossings_z(
    J0: np.ndarray,
    J_anchor: np.ndarray,
    theta_A: float,
    z0: float = 1.0,
    z1: float = 0.0,
    atol: float = 1e-12,
) -> ThresholdCrossingInfo:
    """
    Classe les racines de A_k(z) = theta_A sur [z1, z0] en racines intérieures
    strictes (z1 < z < z0) et racines frontières {z0, z1}.
    """
    events, boundary_z = _compute_threshold_events_z(
        J0=J0,
        J_anchor=J_anchor,
        z0=z0,
        z1=z1,
        theta_A=theta_A,
        atol=atol,
    )
    interior_vals: list[float] = []
    for z, _k, _kind in events:
        if not interior_vals or abs(z - interior_vals[-1]) > atol:
            interior_vals.append(float(z))

    return ThresholdCrossingInfo(
        interior_z=np.asarray(interior_vals, dtype=np.float64),
        boundary_z=boundary_z,
        events=events,
    )


def solve_threshold_crossings_z(
    J0: np.ndarray,
    J_anchor: np.ndarray,
    theta_A: float,
    z0: float = 1.0,
    z1: float = 0.0,
    atol: float = 1e-12,
) -> np.ndarray:
    r"""
    Résout A_k(z) = theta_A sur l'intérieur strict (z1, z0) pour :

        J(z) = J_anchor + (J0 - J_anchor) * z

    Retourne les racines intérieures triées par z décroissant.
    """
    info = classify_threshold_crossings_z(
        J0=J0,
        J_anchor=J_anchor,
        theta_A=theta_A,
        z0=z0,
        z1=z1,
        atol=atol,
    )
    return info.interior_z


# ============================================================
# 9. SIMULATION PAR THINNING D'OGATA
# ============================================================

def simulate_ogata_thinning(
    X0: np.ndarray,
    J0: np.ndarray,
    t_max: float,
    mu: float,
    beta: float,
    theta_A: float,
    lam: float = 0.0,
    fnk_fn: FitnessFn = zero_fnk,
    rng: Any = None,
    eta: float = 0.0,
    gamma: float = 0.0,
    J_anchor: np.ndarray | None = None,
    max_events: int = 100000,
    psi_fn: PsiFn = psi_sym,
) -> SimulationResult:
    """Simulation stochastique exacte par rejet d'Ogata."""
    X = _validate_state(X0).copy()
    J = _validate_J(J0).copy()

    if J.shape != (X.size, X.size):
        raise ValueError("Dimensions incompatibles entre X0 et J0.")

    t_max = _finite_scalar("t_max", t_max)
    mu = _finite_scalar("mu", mu)
    beta = _finite_scalar("beta", beta)
    theta_A = _finite_scalar("theta_A", theta_A)
    lam = _finite_scalar("lam", lam)
    eta = _finite_scalar("eta", eta)
    gamma = _finite_scalar("gamma", gamma)

    if t_max < 0.0 or mu < 0.0 or gamma < 0.0 or max_events < 0:
        raise ValueError("Les paramètres de taux, temps et compteurs doivent être positifs.")

    if J_anchor is None:
        J_anchor = J.copy()
    else:
        J_anchor = _validate_J(J_anchor)

    if J_anchor.shape != J.shape:
        raise ValueError("J_anchor doit avoir la même forme que J0.")

    if rng is None:
        rng = np.random.default_rng()

    events: list[TrajectoryEvent] = []
    t = 0.0
    majorant = float(X.size * mu)

    if majorant == 0.0 or t_max == 0.0 or max_events == 0:
        J_end = dissipate_J(J, J_anchor, dt=t_max, gamma=gamma)
        return SimulationResult(events=events, X_final=X, J_final=J_end, t_final=t_max)

    for _ in range(max_events):
        waiting = _draw_exponential(rng, majorant)
        t_candidate = t + waiting

        if t_candidate > t_max:
            J = dissipate_J(J, J_anchor, dt=t_max - t, gamma=gamma)
            t = t_max
            break

        J_candidate = dissipate_J(J, J_anchor, dt=waiting, gamma=gamma)
        rates = transition_rates(
            X=X,
            J=J_candidate,
            mu=mu,
            beta=beta,
            theta_A=theta_A,
            lam=lam,
            fnk_fn=fnk_fn,
        )
        total_rate = float(np.sum(rates))
        acceptance = min(1.0, max(0.0, total_rate / majorant))

        u_acc = _draw_unit_uniform(rng)
        if total_rate > 0.0 and u_acc < acceptance:
            k = _select_channel_from_rates(rng, rates, total_rate)
            X_after, J_after = mutate_postflip(
                X,
                J_candidate,
                k=k,
                eta=eta,
                psi_fn=psi_fn,
            )
            X = X_after
            J = J_after
            t = t_candidate
            events.append(
                TrajectoryEvent(
                    time=t,
                    node=k,
                    X=X.copy(),
                    J=J.copy(),
                    rates=rates.copy(),
                )
            )
        else:
            J = J_candidate
            t = t_candidate

    return SimulationResult(events=events, X_final=X, J_final=J, t_final=t)


# ============================================================
# 10. SOLVEUR EXACT PAR MORCEAUX POUR lambda = 0
# ============================================================

def simulate_exact_piecewise_lambda0(
    X0: np.ndarray,
    J0: np.ndarray,
    t_max: float,
    mu: float,
    beta: float,
    theta_A: float,
    eta: float = 0.0,
    gamma: float = 0.0,
    J_anchor: np.ndarray | None = None,
    rng: Any = None,
    max_events: int = 100000,
    fnk_fn: FitnessFn = zero_fnk,
    psi_fn: PsiFn = psi_sym,
) -> SimulationResult:
    r"""
    Simulation exacte par morceaux pour lambda = 0.

    Entre deux mutations, J(t) dissipe vers J_anchor.
    Le masque d'accessibilité est constant par morceaux
    en z = exp(-gamma * Delta t).
    """
    X = _validate_state(X0).copy()
    J = _validate_J(J0).copy()

    if J.shape != (X.size, X.size):
        raise ValueError("Dimensions incompatibles entre X0 et J0.")

    t_max = _finite_scalar("t_max", t_max)
    mu = _finite_scalar("mu", mu)
    beta = _finite_scalar("beta", beta)
    theta_A = _finite_scalar("theta_A", theta_A)
    eta = _finite_scalar("eta", eta)
    gamma = _finite_scalar("gamma", gamma)

    if t_max < 0.0 or mu < 0.0 or gamma < 0.0 or max_events < 0:
        raise ValueError("Paramètres non valides.")

    if J_anchor is None:
        J_anchor = J.copy()
    else:
        J_anchor = _validate_J(J_anchor)

    if J_anchor.shape != J.shape:
        raise ValueError("J_anchor doit avoir la même forme que J0.")

    if rng is None:
        rng = np.random.default_rng()

    events: list[TrajectoryEvent] = []
    t = 0.0

    if t_max == 0.0 or mu == 0.0 or max_events == 0:
        J_end = dissipate_J(J, J_anchor, dt=t_max, gamma=gamma)
        return SimulationResult(events=events, X_final=X, J_final=J_end, t_final=t_max)

    event_count = 0

    while t < t_max and event_count < max_events:
        remaining = t_max - t

        # --------------------------------------------------------
        # Cas non dissipatif
        # --------------------------------------------------------
        if gamma == 0.0:
            rates = transition_rates(
                X=X,
                J=J,
                mu=mu,
                beta=beta,
                theta_A=theta_A,
                lam=0.0,
                fnk_fn=fnk_fn,
            )
            total_rate = float(np.sum(rates))
            if total_rate <= 0.0:
                t = t_max
                break

            waiting = _draw_exponential(rng, total_rate)
            if waiting >= remaining:
                t = t_max
                break

            k = _select_channel_from_rates(rng, rates, total_rate)
            X, J = mutate_postflip(X, J, k=k, eta=eta, psi_fn=psi_fn)
            t += waiting
            event_count += 1
            events.append(
                TrajectoryEvent(
                    time=t,
                    node=k,
                    X=X.copy(),
                    J=J.copy(),
                    rates=rates.copy(),
                )
            )
            continue

        # --------------------------------------------------------
        # Cas dissipatif : z = exp(-gamma * (tau - t))
        # --------------------------------------------------------
        z_start = 1.0
        z_end = float(exp(-gamma * remaining))

        crossings, _ = _compute_threshold_events_z(
            J0=J,
            J_anchor=J_anchor,
            z0=z_start,
            z1=z_end,
            theta_A=theta_A,
        )

        boundaries = [z_start]
        boundaries.extend(z for z, _, _ in crossings if z_end < z < z_start)
        boundaries.append(z_end)
        boundaries = sorted(set(boundaries), reverse=True)

        mutation_occurred = False
        J_ref = J.copy()
        t_ref = t

        for z_a, z_b in zip(boundaries[:-1], boundaries[1:]):
            if z_a <= 0.0 or z_b <= 0.0:
                continue

            dt_segment = log(z_a / z_b) / gamma
            if dt_segment <= 0.0:
                continue

            z_mid = 0.5 * (z_a + z_b)
            J_mid = _validate_J(J_anchor + (J_ref - J_anchor) * z_mid)

            rates = transition_rates(
                X=X,
                J=J_mid,
                mu=mu,
                beta=beta,
                theta_A=theta_A,
                lam=0.0,
                fnk_fn=fnk_fn,
            )
            total_rate = float(np.sum(rates))

            if total_rate > 0.0:
                waiting = _draw_exponential(rng, total_rate)
                if waiting < dt_segment:
                    t_event = t + waiting
                    tau = t_event - t_ref
                    J_event = dissipate_J(J_ref, J_anchor, dt=tau, gamma=gamma)
                    k = _select_channel_from_rates(rng, rates, total_rate)
                    X, J = mutate_postflip(X, J_event, k=k, eta=eta, psi_fn=psi_fn)
                    t = t_event
                    event_count += 1
                    mutation_occurred = True
                    events.append(
                        TrajectoryEvent(
                            time=t,
                            node=k,
                            X=X.copy(),
                            J=J.copy(),
                            rates=rates.copy(),
                        )
                    )
                    break

            t += dt_segment
            J = dissipate_J(J_ref, J_anchor, dt=t - t_ref, gamma=gamma)

        if mutation_occurred:
            continue

        J = dissipate_J(J_ref, J_anchor, dt=t_max - t_ref, gamma=gamma)
        t = t_max
        break

    return SimulationResult(events=events, X_final=X, J_final=J, t_final=t)


# ============================================================
# 11. WASSERSTEIN-1 EMPIRIQUE EN DIMENSION 1
# ============================================================

def wasserstein_1d(
    x: Sequence[float],
    y: Sequence[float],
) -> float:
    r"""
    Distance de Wasserstein-1 unidimensionnelle
    sur deux échantillons empiriques.
    """
    x_arr = np.sort(np.asarray(x, dtype=np.float64).ravel())
    y_arr = np.sort(np.asarray(y, dtype=np.float64).ravel())

    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("Les deux échantillons doivent être non vides.")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("Les échantillons doivent contenir des valeurs finies.")

    i = 0
    j = 0
    mass_x = 1.0 / x_arr.size
    mass_y = 1.0 / y_arr.size
    remaining_x = mass_x
    remaining_y = mass_y
    distance = 0.0

    while i < x_arr.size and j < y_arr.size:
        transported = min(remaining_x, remaining_y)
        distance += transported * abs(x_arr[i] - y_arr[j])
        remaining_x -= transported
        remaining_y -= transported

        if remaining_x <= 1e-15:
            i += 1
            remaining_x = mass_x if i < x_arr.size else 0.0
        if remaining_y <= 1e-15:
            j += 1
            remaining_y = mass_y if j < y_arr.size else 0.0

    return float(distance)


# ============================================================
# 12. SLICED WASSERSTEIN-1
# ============================================================

def sliced_wasserstein_1(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 128,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> float:
    """Distance Sliced Wasserstein-1 par projections aléatoires."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X et Y doivent être bidimensionnels.")
    if X.shape[0] == 0 or Y.shape[0] == 0:
        raise ValueError("X et Y doivent contenir au moins un point.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X et Y doivent avoir la même dimension de caractéristiques.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("X et Y doivent contenir des valeurs finies.")
    if n_projections < 1:
        raise ValueError("n_projections doit être >= 1.")

    if rng is None:
        rng = np.random.default_rng(seed)

    dimension = X.shape[1]
    directions = rng.normal(size=(n_projections, dimension))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    zero_rows = norms[:, 0] == 0.0

    while np.any(zero_rows):
        directions[zero_rows] = rng.normal(size=(int(np.sum(zero_rows)), dimension))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        zero_rows = norms[:, 0] == 0.0

    directions /= norms
    distances = [wasserstein_1d(X @ direction, Y @ direction) for direction in directions]
    return float(np.mean(distances))


# ============================================================
# 13. ENERGY DISTANCE
# ============================================================

def energy_distance(
    X: np.ndarray,
    Y: np.ndarray,
    block_size: int | None = None,
) -> float:
    r"""
    Energy distance empirique entre deux distributions
    multivariées (calcul par blocs mémoire).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X et Y doivent être bidimensionnels.")
    if X.shape[0] == 0 or Y.shape[0] == 0:
        raise ValueError("X et Y doivent contenir au moins un point.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimensions incompatibles entre X et Y.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("X et Y doivent contenir des valeurs finies.")
    if block_size is not None and block_size < 1:
        raise ValueError("block_size doit être >= 1.")

    def mean_pairwise_dist(A: np.ndarray, B: np.ndarray) -> float:
        n_a = A.shape[0]
        n_b = B.shape[0]
        step = n_a if block_size is None else min(int(block_size), n_a)
        total = 0.0
        for i0 in range(0, n_a, step):
            i1 = min(i0 + step, n_a)
            diffs = A[i0:i1, None, :] - B[None, :, :]
            total += float(np.sum(np.linalg.norm(diffs, axis=-1)))
        return total / float(n_a * n_b)

    cross = mean_pairwise_dist(X, Y)
    within_x = mean_pairwise_dist(X, X)
    within_y = mean_pairwise_dist(Y, Y)

    squared_dist = 2.0 * cross - within_x - within_y
    return float(sqrt(max(0.0, squared_dist)))
