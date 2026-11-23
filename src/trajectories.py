"""
src/trajectories.py
Projet : NK-LSTM-PATH

Simulateur de la dynamique jointe (mutations, transport post-flip, dissipation
exponentielle) par amincissement exact d'Ogata (thinning) avec horloge majorante
Lambda_max = N * mu, accompagné du solveur analytique exact par morceaux pour
le régime sans rétroaction (lambda = 0) et de la distance de Wasserstein-1 (W_1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np

from fiber_bundle import (
    PsiFn,
    _validate_J,
    _validate_state,
    accessibility_A,
    mutate_postflip,
    psi_sym,
)

FitnessNKFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class JumpEvent:
    """Enregistrement d'une mutation acceptée et du transport post-flip associé."""

    time: float
    node: int
    X: np.ndarray
    J: np.ndarray


@dataclass(frozen=True)
class TrajectoryResult:
    """Résultat complet d'une trajectoire simulée sur [0, t_max]."""

    t_max: float
    X_final: np.ndarray
    J_final: np.ndarray
    events: List[JumpEvent]
    n_accepted: int
    n_candidates: int
    method: str


class NKLandscape:
    """
    Paysage adaptatif NK natif sur {-1, +1}^N (converti en {0, 1}^N).
    Pour chaque composant i, la contribution dépend de s_i et de K voisins.
    """

    def __init__(self, n: int, k: int, seed: int = 0) -> None:
        if n < 2:
            raise ValueError("N doit être >= 2.")
        if not (0 <= k < n):
            raise ValueError(f"K={k} doit vérifier 0 <= K < N={n}.")
        self.n = int(n)
        self.k = int(k)
        rng = np.random.default_rng(seed)

        self.neighbors = np.zeros((self.n, self.k), dtype=np.int64)
        for i in range(self.n):
            pool = [j for j in range(self.n) if j != i]
            if self.k > 0:
                self.neighbors[i] = rng.choice(pool, size=self.k, replace=False)

        self.table = rng.uniform(0.0, 1.0, size=(self.n, 1 << (self.k + 1)))
        self._powers = (1 << np.arange(self.k + 1, dtype=np.int64))

    def __call__(self, X: np.ndarray) -> float:
        state = _validate_state(X)
        if state.size != self.n:
            raise ValueError(f"Dimension attendue N={self.n}, reçu {state.size}.")
        bits = (state > 0.0).astype(np.int64)
        total = 0.0
        for i in range(self.n):
            if self.k == 0:
                idx = int(bits[i])
            else:
                local_bits = np.empty(self.k + 1, dtype=np.int64)
                local_bits[0] = bits[i]
                local_bits[1:] = bits[self.neighbors[i]]
                idx = int(np.dot(local_bits, self._powers))
            total += float(self.table[i, idx])
        return total / float(self.n)


def zero_fnk(X: np.ndarray) -> float:
    """Paysage natif nul F_NK(X) = 0 (contrôle neutre)."""
    _validate_state(X)
    return 0.0


def extended_fitness(
    X: np.ndarray,
    J: np.ndarray,
    lam: float,
    fnk_fn: FitnessNKFn = zero_fnk,
) -> float:
    """
    Fitness étendue :
        F(X, J) = F_NK(X) + lambda * sum_{i < j} J_{ij} s_i s_j.
    """
    state = _validate_state(X)
    mat = _validate_J(J)
    if state.size != mat.shape[0]:
        raise ValueError("Dimensions incompatibles entre X et J.")
    f_nk = float(fnk_fn(state))
    if lam == 0.0:
        return f_nk
    # Comme J est symétrique à diagonale nulle, sum_{i<j} J_ij s_i s_j = 0.5 * s^T J s
    coupling = 0.5 * float(state @ (mat @ state))
    return f_nk + float(lam) * coupling


def delta_fitness_k(
    X: np.ndarray,
    J: np.ndarray,
    k: int,
    lam: float,
    fnk_fn: FitnessNKFn = zero_fnk,
) -> float:
    """
    Variation exacte de fitness lors du basculement du nœud k (s_k -> -s_k),
    évaluée à fibre J fixée avant le transport :
        Delta F_k = F_NK(X^{(k)}) - F_NK(X) - 2 * lambda * s_k * sum_{j != k} J_{kj} s_j.
    """
    state = _validate_state(X)
    mat = _validate_J(J)
    n = state.size
    if mat.shape[0] != n:
        raise ValueError("Dimensions incompatibles entre X et J.")
    if not (0 <= k < n):
        raise IndexError(f"Indice de nœud k={k} hors limites pour N={n}.")

    X_flipped = state.copy()
    X_flipped[k] = -X_flipped[k]
    df_nk = float(fnk_fn(X_flipped)) - float(fnk_fn(state))
    if lam == 0.0:
        return df_nk

    # J_{kk} == 0 => sum_{j != k} J_{kj} s_j == (J @ s)_k
    local_field = float(np.dot(mat[k], state))
    return df_nk - 2.0 * float(lam) * float(state[k]) * local_field


def transition_rates(
    X: np.ndarray,
    J: np.ndarray,
    mu: float,
    beta: float,
    lam: float,
    theta_A: float,
    fnk_fn: FitnessNKFn = zero_fnk,
) -> np.ndarray:
    """
    Calcule le vecteur des taux de mutation w_k(X, J) pour k = 0, ..., N-1 :
        w_k(X, J) = mu * a_k(J) * min(1, exp(beta * Delta F_k(X, J))),
    où a_k(J) = 1[A_k(J) > theta_A]. Vérifie 0 <= w_k <= mu.
    """
    if mu < 0.0:
        raise ValueError("Le taux de base mu doit être >= 0.")
    if beta < 0.0:
        raise ValueError("La sensibilité sélective beta doit être >= 0.")

    state = _validate_state(X)
    mat = _validate_J(J)
    n = state.size
    if mat.shape[0] != n:
        raise ValueError("Dimensions incompatibles entre X et J.")

    rates = np.zeros(n, dtype=np.float64)
    if mu == 0.0:
        return rates

    active = accessibility_A(mat) > float(theta_A)
    for k in range(n):
        if not active[k]:
            continue
        df_k = delta_fitness_k(state, mat, k=k, lam=lam, fnk_fn=fnk_fn)
        exponent = float(beta) * df_k
        boltzmann = 1.0 if exponent >= 0.0 else float(np.exp(exponent))
        rates[k] = float(mu) * boltzmann
    return rates


def dissipate_J(
    J_curr: np.ndarray,
    J_anchor: np.ndarray,
    gamma: float,
    dt: float,
) -> np.ndarray:
    """
    Évolution analytique exacte par dissipation linéaire sur une durée dt >= 0 :
        J(t_0 + dt) = J^{(0)} + (J(t_0) - J^{(0)}) * exp(-gamma * dt).
    """
    if gamma < 0.0:
        raise ValueError("Le taux de dissipation gamma doit être >= 0.")
    if dt < 0.0:
        raise ValueError("L'incrément temporel dt doit être >= 0.")

    mat_curr = _validate_J(J_curr)
    mat_anchor = _validate_J(J_anchor)
    if mat_curr.shape != mat_anchor.shape:
        raise ValueError("Dimensions incompatibles entre J_curr et J_anchor.")

    if dt == 0.0 or gamma == 0.0:
        return mat_curr.copy()

    decay = float(np.exp(-float(gamma) * float(dt)))
    J_next = mat_anchor + (mat_curr - mat_anchor) * decay
    # Garantie de symétrie numérique exacte
    J_next = 0.5 * (J_next + J_next.T)
    np.fill_diagonal(J_next, 0.0)
    return J_next


def solve_threshold_crossings_z(
    J_curr: np.ndarray,
    J_anchor: np.ndarray,
    theta_A: float,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Résout analytiquement, pour chaque nœud k in {0, ..., N-1}, l'équation
    quadratique en z = exp(-gamma * tau) in (0, 1) :
        A_k(J^{(0)} + (J_curr - J^{(0)}) * z) - theta_A = 0
        <=> c_2 * z^2 + c_1 * z + c_0 = 0,
    avec :
        c_2 = sum_{j != k} D_{kj}^2 >= 0,
        c_1 = 2 * sum_{j != k} J^{(0)}_{kj} D_{kj},
        c_0 = sum_{j != k} (J^{(0)}_{kj})^2 - theta_A,
        D = J_curr - J^{(0)}.

    Gère explicitement les cas dégénérés (c_2 = 0), les racines doubles
    (Delta = 0), les racines hors de (0, 1) et les égalités au seuil en z = 1.
    Retourne les racines distinctes triées par ordre décroissant (z proche de 1 d'abord).
    """
    mat_curr = _validate_J(J_curr)
    mat_anchor = _validate_J(J_anchor)
    if mat_curr.shape != mat_anchor.shape:
        raise ValueError("Dimensions incompatibles entre J_curr et J_anchor.")

    diff = mat_curr - mat_anchor
    n = mat_curr.shape[0]
    roots: List[float] = []
    eps_coef = 1e-28

    for k in range(n):
        d_row = diff[k]
        j0_row = mat_anchor[k]

        c2 = float(np.dot(d_row, d_row))
        c1 = 2.0 * float(np.dot(j0_row, d_row))
        c0 = float(np.dot(j0_row, j0_row)) - float(theta_A)

        if c2 <= eps_coef:
            if abs(c1) > eps_coef:
                z = -c0 / c1
                if tol < z < 1.0 - tol:
                    roots.append(float(z))
            continue

        disc = c1 * c1 - 4.0 * c2 * c0
        if disc < -eps_coef:
            continue
        if abs(disc) <= eps_coef:
            z = -c1 / (2.0 * c2)
            if tol < z < 1.0 - tol:
                roots.append(float(z))
            continue

        sqrt_disc = float(np.sqrt(disc))
        z1 = (-c1 - sqrt_disc) / (2.0 * c2)
        z2 = (-c1 + sqrt_disc) / (2.0 * c2)
        for z in (z1, z2):
            if tol < z < 1.0 - tol:
                roots.append(float(z))

    if not roots:
        return np.empty(0, dtype=np.float64)

    roots_sorted = sorted(roots, reverse=True)
    dedup: List[float] = [roots_sorted[0]]
    for z in roots_sorted[1:]:
        if abs(z - dedup[-1]) > tol:
            dedup.append(z)
    return np.asarray(dedup, dtype=np.float64)


def simulate_ogata_thinning(
    X0: np.ndarray,
    J0: np.ndarray,
    t_max: float,
    mu: float,
    gamma: float,
    beta: float,
    lam: float,
    eta: float,
    theta_A: float,
    fnk_fn: FitnessNKFn = zero_fnk,
    psi_fn: PsiFn = psi_sym,
    J_anchor: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    max_candidates: int = 5_000_000,
) -> TrajectoryResult:
    """
    Simule en loi exacte la dynamique jointe (mutations, transport post-flip,
    dissipation) sur [0, t_max] par amincissement d'Ogata avec horloge majorante
    uniforme Lambda_max = N * mu.
    """
    if t_max < 0.0:
        raise ValueError("t_max doit être >= 0.")
    if mu < 0.0 or gamma < 0.0 or beta < 0.0:
        raise ValueError("Les paramètres mu, gamma et beta doivent être >= 0.")

    X = _validate_state(X0).copy()
    J = _validate_J(J0).copy()
    anchor = J.copy() if J_anchor is None else _validate_J(J_anchor).copy()
    n = X.size
    if J.shape[0] != n or anchor.shape[0] != n:
        raise ValueError("Dimensions incompatibles entre X0, J0 et J_anchor.")

    gen = np.random.default_rng() if rng is None else rng
    events: List[JumpEvent] = []
    t = 0.0
    n_candidates = 0
    lambda_max = float(n) * float(mu)

    if lambda_max == 0.0 or t_max == 0.0:
        J_end = dissipate_J(J, anchor, gamma=gamma, dt=t_max)
        return TrajectoryResult(
            t_max=float(t_max),
            X_final=X,
            J_final=J_end,
            events=events,
            n_accepted=0,
            n_candidates=0,
            method="ogata_thinning",
        )

    while t < t_max:
        if n_candidates >= max_candidates:
            raise RuntimeError(
                f"Nombre maximal d'événements candidats atteint ({max_candidates})."
            )

        u1 = float(gen.uniform(0.0, 1.0))
        while u1 == 0.0:
            u1 = float(gen.uniform(0.0, 1.0))
        dt = -float(np.log(u1)) / lambda_max
        n_candidates += 1

        if t + dt > t_max:
            J = dissipate_J(J, anchor, gamma=gamma, dt=t_max - t)
            t = float(t_max)
            break

        t += dt
        J = dissipate_J(J, anchor, gamma=gamma, dt=dt)

        rates = transition_rates(
            X, J, mu=mu, beta=beta, lam=lam, theta_A=theta_A, fnk_fn=fnk_fn
        )
        total_rate = float(np.sum(rates))
        if total_rate > lambda_max * (1.0 + 1e-12):
            raise AssertionError(
                f"Violation de la borne majorante : Lambda={total_rate} > N*mu={lambda_max}."
            )

        u2 = float(gen.uniform(0.0, lambda_max))
        if u2 <= total_rate and total_rate > 0.0:
            cum_rates = np.cumsum(rates)
            k = int(np.searchsorted(cum_rates, u2, side="left"))
            if k >= n:
                k = n - 1
            X, J = mutate_postflip(X, J, k=k, eta=eta, psi_fn=psi_fn)
            events.append(JumpEvent(time=t, node=k, X=X.copy(), J=J.copy()))

    return TrajectoryResult(
        t_max=float(t_max),
        X_final=X,
        J_final=J,
        events=events,
        n_accepted=len(events),
        n_candidates=n_candidates,
        method="ogata_thinning",
    )


def simulate_exact_piecewise_lambda0(
    X0: np.ndarray,
    J0: np.ndarray,
    t_max: float,
    mu: float,
    gamma: float,
    beta: float,
    eta: float,
    theta_A: float,
    fnk_fn: FitnessNKFn = zero_fnk,
    psi_fn: PsiFn = psi_sym,
    J_anchor: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    max_events: int = 1_000_000,
) -> TrajectoryResult:
    """
    Solveur de Gillespie exact par morceaux pour le régime sans rétroaction
    (lambda = 0) : entre deux mutations, les poids de Boltzmann W_k(X) sont
    constants et les instants de franchissement du seuil A_k(J(t)) = theta_A
    sont calculés analytiquement via `solve_threshold_crossings_z`.
    """
    if t_max < 0.0:
        raise ValueError("t_max doit être >= 0.")
    if mu < 0.0 or gamma < 0.0 or beta < 0.0:
        raise ValueError("Les paramètres mu, gamma et beta doivent être >= 0.")

    X = _validate_state(X0).copy()
    J = _validate_J(J0).copy()
    anchor = J.copy() if J_anchor is None else _validate_J(J_anchor).copy()
    n = X.size
    if J.shape[0] != n or anchor.shape[0] != n:
        raise ValueError("Dimensions incompatibles entre X0, J0 et J_anchor.")

    gen = np.random.default_rng() if rng is None else rng
    events: List[JumpEvent] = []
    t = 0.0

    if mu == 0.0 or t_max == 0.0:
        J_end = dissipate_J(J, anchor, gamma=gamma, dt=t_max)
        return TrajectoryResult(
            t_max=float(t_max),
            X_final=X,
            J_final=J_end,
            events=events,
            n_accepted=0,
            n_candidates=0,
            method="exact_piecewise_lambda0",
        )

    while t < t_max:
        if len(events) >= max_events:
            raise RuntimeError(f"Nombre maximal d'événements atteint ({max_events}).")

        remaining_t = float(t_max) - t

        # Poids de Boltzmann constants entre deux mutations (lambda = 0)
        boltz = np.empty(n, dtype=np.float64)
        for k in range(n):
            df_k = delta_fitness_k(X, J, k=k, lam=0.0, fnk_fn=fnk_fn)
            exponent = float(beta) * df_k
            boltz[k] = 1.0 if exponent >= 0.0 else float(np.exp(exponent))

        if gamma == 0.0:
            active = (accessibility_A(J) > float(theta_A)).astype(np.float64)
            rates = float(mu) * active * boltz
            total_rate = float(np.sum(rates))
            if total_rate <= 0.0:
                t = float(t_max)
                break
            u1 = float(gen.uniform(0.0, 1.0))
            while u1 == 0.0:
                u1 = float(gen.uniform(0.0, 1.0))
            dt = -float(np.log(u1)) / total_rate
            if t + dt > t_max:
                t = float(t_max)
                break
            t += dt
            u2 = float(gen.uniform(0.0, total_rate))
            k = int(np.searchsorted(np.cumsum(rates), u2, side="left"))
            if k >= n:
                k = n - 1
            X, J = mutate_postflip(X, J, k=k, eta=eta, psi_fn=psi_fn)
            events.append(JumpEvent(time=t, node=k, X=X.copy(), J=J.copy()))
            continue

        # Résolution analytique des franchissements de seuil sur (0, remaining_t)
        z_roots = solve_threshold_crossings_z(J, anchor, theta_A=theta_A)
        if z_roots.size > 0:
            taus_all = -np.log(z_roots) / float(gamma)
            taus = taus_all[(taus_all > 0.0) & (taus_all < remaining_t)]
        else:
            taus = np.empty(0, dtype=np.float64)

        boundaries = np.concatenate(([0.0], taus, [remaining_t]))
        jumped = False

        for m in range(boundaries.size - 1):
            tau_start = float(boundaries[m])
            tau_end = float(boundaries[m + 1])
            seg_len = tau_end - tau_start
            if seg_len <= 0.0:
                continue

            # Sur l'ouvert (tau_start, tau_end), aucun seuil n'est franchi :
            # l'évaluation au point milieu donne le masque exact sur tout le sous-intervalle.
            tau_mid = 0.5 * (tau_start + tau_end)
            J_mid = dissipate_J(J, anchor, gamma=gamma, dt=tau_mid)
            active_mid = (accessibility_A(J_mid) > float(theta_A)).astype(np.float64)
            rates_m = float(mu) * active_mid * boltz
            lam_m = float(np.sum(rates_m))

            if lam_m <= 0.0:
                continue

            u1 = float(gen.uniform(0.0, 1.0))
            while u1 == 0.0:
                u1 = float(gen.uniform(0.0, 1.0))
            wait = -float(np.log(u1)) / lam_m

            if wait < seg_len:
                tau_hit = tau_start + wait
                t += tau_hit
                J = dissipate_J(J, anchor, gamma=gamma, dt=tau_hit)
                u2 = float(gen.uniform(0.0, lam_m))
                k = int(np.searchsorted(np.cumsum(rates_m), u2, side="left"))
                if k >= n:
                    k = n - 1
                X, J = mutate_postflip(X, J, k=k, eta=eta, psi_fn=psi_fn)
                events.append(JumpEvent(time=t, node=k, X=X.copy(), J=J.copy()))
                jumped = True
                break

        if not jumped:
            J = dissipate_J(J, anchor, gamma=gamma, dt=remaining_t)
            t = float(t_max)
            break

    return TrajectoryResult(
        t_max=float(t_max),
        X_
