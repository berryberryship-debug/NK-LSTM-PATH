"""
src/trajectories.py
Projet : NK-LSTM-PATH

Simulateur de la dynamique jointe (mutations, transport post-flip, dissipation
exponentielle) par amincissement exact d'Ogata (thinning) avec horloge majorante
Lambda_max = N * mu, accompagné du solveur analytique exact par morceaux pour
le régime sans rétroaction (lambda = 0) à horloge de risque intégré, ainsi que
des métriques de comparaison en loi (Wasserstein-1 1D, Sliced Wasserstein-1 et
distance d'énergie multivariée calculée par blocs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence
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


@dataclass(frozen=True)
class ThresholdCrossingsZ:
    """
    Classification des racines z = exp(-gamma * tau) de A_k(J(tau)) = theta_A :
      - interior_z : racines strictement intérieures z in (0, 1), correspondant
        à des instants finis tau = -ln(z) / gamma > 0 qui partitionnent
        l'intervalle entre deux mutations.
      - boundary_z : racines situées aux bornes z = 1 (tau = 0, égalité initiale
        au seuil) ou z = 0 (tau -> +inf, égalité asymptotique à l'ancre J^(0)).
        Volontairement exclues de `interior_z` car elles ne créent pas de
        sous-segment intérieur supplémentaire.
    """

    interior_z: np.ndarray
    boundary_z: np.ndarray


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
        self._powers = 1 << np.arange(self.k + 1, dtype=np.int64)

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
    J_next = 0.5 * (J_next + J_next.T)
    np.fill_diagonal(J_next, 0.0)
    return J_next


def _stable_quadratic_real_roots(
    c2: float,
    c1: float,
    c0: float,
) -> List[float]:
    """
    Résout c2 * z^2 + c1 * z + c0 = 0 dans R en mettant les coefficients à
    l'échelle S = max(|c2|, |c1|, |c0|) pour éviter tout débordement flottant,
    sans réduction arbitraire du degré quadratique lorsque c2 != 0, et en
    utilisant la factorisation de Viète :
        q = -0.5 * (b + sign(b) * sqrt(Delta)),
        z_1 = q / a,   z_2 = c / q.
    """
    scale = max(abs(c2), abs(c1), abs(c0))
    if scale == 0.0 or not np.isfinite(scale):
        return []

    a = c2 / scale
    b = c1 / scale
    c = c0 / scale

    if a == 0.0:
        if b == 0.0:
            return []
        return [float(-c / b)]

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        disc_scale = max(b * b, abs(4.0 * a * c))
        if disc >= -1e-15 * disc_scale:
            return [float(-b / (2.0 * a))]
        return []
    if disc == 0.0:
        return [float(-b / (2.0 * a))]

    sqrt_disc = float(np.sqrt(disc))
    sign_b = 1.0 if b >= 0.0 else -1.0
    q = -0.5 * (b + sign_b * sqrt_disc)
    if q == 0.0:
        return [float(-b / (2.0 * a))]

    z1 = float(q / a)
    z2 = float(c / q)
    return [z1, z2]


def classify_threshold_crossings_z(
    J_curr: np.ndarray,
    J_anchor: np.ndarray,
    theta_A: float,
    eps_boundary: float = 1e-15,
    dedup_tol: float = 1e-14,
) -> ThresholdCrossingsZ:
    """
    Résout et classe pour chaque nœud k in {0, ..., N-1} les racines de :
        A_k(J^{(0)} + (J_curr - J^{(0)}) * z) - theta_A = 0
        <=> c_2 * z^2 + c_1 * z + c_0 = 0,
    avec :
        c_2 = sum_{j != k} D_{kj}^2 >= 0,
        c_1 = 2 * sum_{j != k} J^{(0)}_{kj} D_{kj},
        c_0 = sum_{j != k} (J^{(0)}_{kj})^2 - theta_A,
        D = J_curr - J^{(0)}.

    Distingue :
      - `interior_z` : racines dans (eps_boundary, 1 - eps_boundary), définissant
        les instants tau = -ln(z)/gamma in (0, +inf) de changement de masque.
      - `boundary_z` : racines aux bornes |z| <= eps_boundary ou |z - 1| <= eps_boundary,
        volontairement exclues des points de coupure intérieurs.
    """
    mat_curr = _validate_J(J_curr)
    mat_anchor = _validate_J(J_anchor)
    if mat_curr.shape != mat_anchor.shape:
        raise ValueError("Dimensions incompatibles entre J_curr et J_anchor.")

    diff = mat_curr - mat_anchor
    n = mat_curr.shape[0]
    interior: List[float] = []
    boundary: List[float] = []

    for k in range(n):
        d_row = diff[k]
        j0_row = mat_anchor[k]

        c2 = float(np.dot(d_row, d_row))
        c1 = 2.0 * float(np.dot(j0_row, d_row))
        c0 = float(np.dot(j0_row, j0_row)) - float(theta_A)

        for z in _stable_quadratic_real_roots(c2, c1, c0):
            if eps_boundary < z < 1.0 - eps_boundary:
                interior.append(z)
            elif abs(z) <= eps_boundary or abs(z - 1.0) <= eps_boundary:
                boundary.append(float(np.clip(z, 0.0, 1.0)))

    def _dedup_desc(vals: List[float]) -> np.ndarray:
        if not vals:
            return np.empty(0, dtype=np.float64)
        s = sorted(vals, reverse=True)
        out = [s[0]]
        for v in s[1:]:
            if abs(v - out[-1]) > dedup_tol:
                out.append(v)
        return np.asarray(out, dtype=np.float64)

    return ThresholdCrossingsZ(
        interior_z=_dedup_desc(interior),
        boundary_z=_dedup_desc(boundary),
    )


def solve_threshold_crossings_z(
    J_curr: np.ndarray,
    J_anchor: np.ndarray,
    theta_A: float,
    eps_boundary: float = 1e-15,
    dedup_tol: float = 1e-14,
) -> np.ndarray:
    """
    Retourne les racines strictement intérieures z in (0, 1) triées par ordre
    décroissant (instants tau = -ln(z) / gamma croissants). Les racines aux
    bornes z in {0, 1} sont volontairement exclues.
    """
    return classify_threshold_crossings_z(
        J_curr=J_curr,
        J_anchor=J_anchor,
        theta_A=theta_A,
        eps_boundary=eps_boundary,
        dedup_tol=dedup_tol,
    ).interior_z


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
        if u2 < total_rate and total_rate > 0.0:
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
    (lambda = 0) :
      - Entre deux mutations, les poids de Boltzmann W_k(X) sont constants.
      - Une unique horloge exponentielle E = -ln(u) ~ Exp(1) est tirée au
        début de chaque intervalle inter-mutations ; le risque intégré
        E_m = Lambda_m * Delta t_m est soustrait à chaque changement de masque
        sans réinitialiser l'horloge.
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

        # Résolution analytique des franchissements strictement intérieurs sur (0, remaining_t)
        z_roots = solve_threshold_crossings_z(J, anchor, theta_A=theta_A)
        if z_roots.size > 0:
            taus_all = -np.log(z_roots) / float(gamma)
            taus = taus_all[(taus_all > 0.0) & (taus_all < remaining_t)]
        else:
            taus = np.empty(0, dtype=np.float64)

        boundaries = np.concatenate(([0.0], taus, [remaining_t]))

        # Horloge exponentielle unique conservée à travers tous les sous-segments
        u = float(gen.uniform(0.0, 1.0))
        while u == 0.0:
            u = float(gen.uniform(0.0, 1.0))
        hazard_remaining = -float(np.log(u))
        jumped = False

        for m in range(boundaries.size - 1):
            tau_start = float(boundaries[m])
            tau_end = float(boundaries[m + 1])
            seg_len = tau_end - tau_start
            if seg_len <= 0.0:
                continue

            tau_mid = 0.5 * (tau_start + tau_end)
            J_mid = dissipate_J(J, anchor, gamma=gamma, dt=tau_mid)
            active_mid = (accessibility_A(J_mid) > float(theta_A)).astype(np.float64)
            rates_m = float(mu) * active_mid * boltz
            lam_m = float(np.sum(rates_m))

            if lam_m <= 0.0:
                continue

            hazard_segment = lam_m * seg_len
            if hazard_remaining < hazard_segment:
                wait = hazard_remaining / lam_m
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

            hazard_remaining -= hazard_segment

        if not jumped:
            J = dissipate_J(J, anchor, gamma=gamma, dt=remaining_t)
            t = float(t_max)
            break

    return TrajectoryResult(
        t_max=float(t_max),
        X_final=X,
        J_final=J,
        events=events,
        n_accepted=len(events),
        n_candidates=len(events),
        method="exact_piecewise_lambda0",
    )


def wasserstein_1d(u_samples: Sequence[float], v_samples: Sequence[float]) -> float:
    """
    Calcule la distance de Wasserstein-1 (W_1) exacte entre deux échantillons
    empiriques unidimensionnels de tailles quelconques (n_u, n_v >= 1).
    """
    u = np.sort(np.asarray(u_samples, dtype=np.float64).ravel())
    v = np.sort(np.asarray(v_samples, dtype=np.float64).ravel())
    if u.size == 0 or v.size == 0:
        raise ValueError("Les échantillons doivent être non vides.")
    if u.size == v.size:
        return float(np.mean(np.abs(u - v)))

    all_values = np.concatenate([u, v])
    all_values.sort()
    deltas = np.diff(all_values)
    if deltas.size == 0:
        return 0.0

    cdf_u = np.searchsorted(u, all_values[:-1], side="right") / float(u.size)
    cdf_v = np.searchsorted(v, all_values[:-1], side="right") / float(v.size)
    return float(np.sum(np.abs(cdf_u - cdf_v) * deltas))


def sliced_wasserstein_1(
    U: np.ndarray,
    V: np.ndarray,
    n_projections: int = 64,
    seed: int = 0,
) -> float:
    """
    Distance de Wasserstein-1 projetée (Sliced Wasserstein-1) entre deux
    nuages multivariés U in R^{n x d} et V in R^{m x d}.
    """
    arr_u = np.asarray(U, dtype=np.float64)
    arr_v = np.asarray(V, dtype=np.float64)
    if arr_u.ndim != 2 or arr_v.ndim != 2 or arr_u.shape[1] != arr_v.shape[1]:
        raise ValueError("U et V doivent être des matrices 2D de même dimension d.")
    if arr_u.shape[0] == 0 or arr_v.shape[0] == 0:
        raise ValueError("Les échantillons U et V doivent être non vides.")

    d = arr_u.shape[1]
    rng = np.random.default_rng(seed)
    dirs = rng.normal(0.0, 1.0, size=(n_projections, d))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-15)

    proj_u = arr_u @ dirs.T
    proj_v = arr_v @ dirs.T
    total = 0.0
    for p in range(n_projections):
        total += wasserstein_1d(proj_u[:, p], proj_v[:, p])
    return total / float(n_projections)


def energy_distance(
    U: np.ndarray,
    V: np.ndarray,
    block_size: int = 256,
) -> float:
    """
    Distance d'énergie multivariée D_E(U, V) = sqrt(2 E||X - Y|| - E||X - X'|| - E||Y - Y'||),
    calculée par blocs de taille `block_size` afin de borner l'empreinte mémoire
    à O(block_size * max(n, m) * d).
    """
    arr_u = np.asarray(U, dtype=np.float64)
    arr_v = np.asarray(V, dtype=np.float64)
    if arr_u.ndim != 2 or arr_v.ndim != 2 or arr_u.shape[1] != arr_v.shape[1]:
        raise ValueError("U et V doivent être des matrices 2D de même dimension d.")
    if arr_u.shape[0] < 2 or arr_v.shape[0] < 2:
        raise ValueError("Au moins 2 échantillons requis dans U et V.")
    if block_size < 1:
        raise ValueError("block_size doit être >= 1.")

    def _mean_cross_blocked(A: np.ndarray, B: np.ndarray) -> float:
        n_a, n_b = A.shape[0], B.shape[0]
        total = 0.0
        for start in range(0, n_a, block_size):
            end = min(start + block_size, n_a)
            diff = A[start:end, None, :] - B[None, :, :]
            total += float(np.sum(np.linalg.norm(diff, axis=-1)))
        return total / float(n_a * n_b)

    def _mean_within_blocked(A: np.ndarray) -> float:
        n_a = A.shape[0]
        total = 0.0
        for start in range(0, n_a, block_size):
            end = min(start + block_size, n_a)
            diff = A[start:end, None, :] - A[None, :, :]
            total += float(np.sum(np.linalg.norm(diff, axis=-1)))
        return total / float(n_a * (n_a - 1))

    e_xy = _mean_cross_blocked(arr_u, arr_v)
    e_xx = _mean_within_blocked(arr_u)
    e_yy = _mean_within_blocked(arr_v)
    val = max(0.0, 2.0 * e_xy - e_xx - e_yy)
    return float(np.sqrt(val))
