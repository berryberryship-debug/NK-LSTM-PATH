
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

from math import exp, log, sqrt
from typing import Callable, Sequence

import numpy as np

from fiber_bundle import (
    _validate_J,
    _validate_state,
    accessibility_A,
    mutate_postflip,
)


FitnessFn = Callable[[np.ndarray], float]


# ============================================================
# 0. VALIDATION DES PARAMÈTRES ET DES FITNESS
# ============================================================

def _finite_scalar(name: str, value: float) -> float:
    """Convertit un scalaire en float et refuse NaN/inf."""

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            f"{name} doit être un réel fini."
        )

    return value


def _evaluate_fitness(
    fnk_fn: FitnessFn,
    X: np.ndarray,
) -> float:
    """Évalue une fitness NK et vérifie que le résultat est fini."""

    return _finite_scalar(
        "La fitness",
        fnk_fn(X),
    )


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


# ============================================================
# 1. PAYSAGE NK CANONIQUE DE KAUFFMAN
# ============================================================

class NKLandscape:
    """
    Paysage NK canonique.

    Chaque site i possède une contribution locale f_i dépendant :
    - de son propre état ;
    - de K autres sites.
    """

    def __init__(
        self,
        N: int,
        K: int,
        seed: int | None = None,
    ) -> None:

        if N < 2:
            raise ValueError("N doit être >= 2.")

        if not (0 <= K < N):
            raise ValueError(
                "K doit satisfaire 0 <= K < N."
            )

        self.N = int(N)
        self.K = int(K)

        rng = np.random.default_rng(seed)

        self.neighbors: list[np.ndarray] = []
        self.tables: list[np.ndarray] = []

        table_size = 2 ** (self.K + 1)

        for i in range(self.N):

            candidates = np.array(
                [j for j in range(self.N) if j != i],
                dtype=int,
            )

            if self.K > 0:
                neigh = rng.choice(
                    candidates,
                    size=self.K,
                    replace=False,
                )
            else:
                neigh = np.empty(0, dtype=int)

            self.neighbors.append(
                np.asarray(neigh, dtype=int)
            )

            self.tables.append(
                rng.random(table_size)
            )

    def local_contribution(
        self,
        X: np.ndarray,
        i: int,
    ) -> float:

        X = _validate_state(X)

        if X.size != self.N:
            raise ValueError(
                f"Taille de X incompatible : {X.size} != {self.N}."
            )

        if not (0 <= i < self.N):
            raise IndexError(
                f"Indice i hors limites : {i}."
            )

        indices = np.concatenate(
            ([i], self.neighbors[i])
        )

        bits = (
            (X[indices] + 1.0) / 2.0
        ).astype(np.int8)

        index = 0

        for bit in bits:
            index = (index << 1) | int(bit)

        return float(
            self.tables[i][index]
        )

    def fitness(
        self,
        X: np.ndarray,
    ) -> float:

        X = _validate_state(X)

        if X.size != self.N:
            raise ValueError(
                f"Taille de X incompatible : {X.size} != {self.N}."
            )

        total = sum(
            self.local_contribution(X, i)
            for i in range(self.N)
        )

        return float(total / self.N)

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
    fnk_fn: FitnessFn,
) -> float:
    r"""
    Fitness étendue :

        F(X, J) = F_NK(X) + (lambda/2) X^T J X
    """

    X, J = _validate_pair(X, J)

    lam = _finite_scalar("lam", lam)

    f_nk = _evaluate_fitness(
        fnk_fn,
        X,
    )

    coupling = 0.5 * float(
        X @ J @ X
    )

    return float(
        f_nk + lam * coupling
    )


# ============================================================
# 4. VARIATION DE FITNESS POUR UN FLIP
# ============================================================

def delta_fitness_k(
    X: np.ndarray,
    J: np.ndarray,
    k: int,
    lam: float,
    fnk_fn: FitnessFn,
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
        raise IndexError(
            f"Indice k hors limites : {k}."
        )

    lam = _finite_scalar("lam", lam)

    X_next = X.copy()
    X_next[k] *= -1.0

    delta_nk = (
        _evaluate_fitness(fnk_fn, X_next)
        - _evaluate_fitness(fnk_fn, X)
    )

    local_field = float(
        np.dot(J[k], X)
    )

    delta_J = (
        -2.0 * lam * float(X[k]) * local_field
    )

    return float(
        delta_nk + delta_J
    )


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
    fnk_fn: FitnessFn,
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

    rates = np.zeros(
        X.size,
        dtype=float,
    )

    if mu == 0.0:
        return rates

    A = accessibility_A(J)
    active = A > theta_A

    for k in range(X.size):

        if not active[k]:
            continue

        delta = delta_fitness_k(
            X=X,
            J=J,
            k=k,
            lam=lam,
            fnk_fn=fnk_fn,
        )

        exponent = beta * delta

        boltzmann = (
            1.0
            if exponent >= 0.0
            else float(np.exp(exponent))
        )

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
        raise ValueError(
            "J et J_anchor doivent avoir la même forme."
        )

    dt = _finite_scalar("dt", dt)
    gamma = _finite_scalar("gamma", gamma)

    if dt < 0.0:
        raise ValueError("dt doit être >= 0.")

    if gamma < 0.0:
        raise ValueError("gamma doit être >= 0.")

    if dt == 0.0 or gamma == 0.0:
        return J.copy()

    decay = float(
        np.exp(-gamma * dt)
    )

    J_next = (
        J_anchor + (J - J_anchor) * decay
    )

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
    """Résout a*z^2 + b*z + c = 0 de façon stable."""

    a = _finite_scalar("a", a)
    b = _finite_scalar("b", b)
    c = _finite_scalar("c", c)
    atol = _finite_scalar("atol", atol)

    if atol < 0.0:
        raise ValueError("atol doit être >= 0.")

    if abs(a) <= atol:

        if abs(b) <= atol:
            return []

        return [-c / b]

    discriminant = b * b - 4.0 * a * c

    scale = (
        abs(b * b)
        + abs(4.0 * a * c)
        + 1.0
    )

    if discriminant < 0.0:

        if abs(discriminant) <= atol * scale:
            discriminant = 0.0
        else:
            return []

    sqrt_disc = sqrt(
        max(0.0, discriminant)
    )

    if sqrt_disc == 0.0:
        return [-b / (2.0 * a)]

    q = -0.5 * (
        b + np.copysign(sqrt_disc, b)
    )

    roots = [q / a]

    if abs(q) > atol:
        roots.append(c / q)

    roots = sorted(
        float(r)
        for r in roots
        if np.isfinite(r)
    )

    unique: list[float] = []

    for r in roots:

        if (
            not unique
            or not np.isclose(
                r,
                unique[-1],
                rtol=0.0,
                atol=atol,
            )
        ):
            unique.append(r)

    return unique


# ============================================================
# 8. ANCIENNE INTERFACE DE CLASSIFICATION
# ============================================================

def classify_threshold_crossings_z(
    A0: np.ndarray,
    A_anchor: np.ndarray,
    z0: float,
    z1: float,
    theta_A: float,
) -> list[tuple[float, int, str]]:
    """Conservé pour rétro-compatibilité."""

    _ = (
        A0,
        A_anchor,
        z0,
        z1,
        theta_A,
    )

    raise NotImplementedError(
        "A(z0) et A(z1) ne suffisent pas à déterminer les "
        "traversées quadratiques. Utiliser "
        "solve_threshold_crossings_z(J0, J_anchor, ...)."
    )


# ============================================================
# 9. RÉSOLUTION DES TRAVERSÉES DE SEUIL
# ============================================================

def solve_threshold_crossings_z(
    J0: np.ndarray,
    J_anchor: np.ndarray,
    z0: float,
    z1: float,
    theta_A: float,
    atol: float = 1e-12,
) -> list[tuple[float, int, str]]:
    r"""
    Résout A_k(z) = theta_A sur [z1, z0] pour :

        J(z) = J_anchor + (J0 - J_anchor) * z
    """

    J0 = _validate_J(J0)
    J_anchor = _validate_J(J_anchor)

    if J0.shape != J_anchor.shape:
        raise ValueError(
            "J0 et J_anchor doivent avoir la même forme."
        )

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

    events: list[tuple[float, int, str]] = []

    for k in range(J0.shape[0]):

        row_anchor = J_anchor[k]
        row_delta = D[k]

        a = float(
            np.dot(row_delta, row_delta)
        )

        b = float(
            2.0 * np.dot(row_anchor, row_delta)
        )

        c = float(
            np.dot(row_anchor, row_anchor) - theta_A
        )

        roots = _stable_quadratic_real_roots(
            a,
            b,
            c,
            atol=atol,
        )

        for root in roots:

            if root < z1 - atol or root > z0 + atol:
                continue

            z = float(
                min(z0, max(z1, root))
            )

            eps = min(
                max(1e-8, 100.0 * atol),
                max((z0 - z1) * 0.25, atol),
            )

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

            events.append(
                (z, k, kind)
            )

    # Tri par z décroissant :
    # ordre chronologique de la dissipation.

    events.sort(
        key=lambda item: (-item[0], item[1])
    )

    unique: list[tuple[float, int, str]] = []

    for event in events:

        z, k, _kind = event

        duplicate = any(
            k == pk and abs(z - pz) <= atol
            for pz, pk, _ in unique
        )

        if not duplicate:
            unique.append(event)

    return unique


# ============================================================
# 10. SIMULATION PAR THINNING D'OGATA
# ============================================================

def simulate_ogata_thinning(
    X0: np.ndarray,
    J0: np.ndarray,
    t_max: float,
    mu: float,
    beta: float,
    theta_A: float,
    lam: float,
    fnk_fn: FitnessFn,
    rng: np.random.Generator | None = None,
    eta: float = 0.0,
    gamma: float = 0.0,
    J_anchor: np.ndarray | None = None,
    max_events: int = 100000,
) -> list[dict]:
    """Simulation stochastique par rejet d'Ogata."""

    X = _validate_state(X0).copy()
    J = _validate_J(J0).copy()

    if J.shape != (X.size, X.size):
        raise ValueError(
            "Dimensions incompatibles entre X0 et J0."
        )

    t_max = _finite_scalar("t_max", t_max)
    mu = _finite_scalar("mu", mu)
    beta = _finite_scalar("beta", beta)
    theta_A = _finite_scalar("theta_A", theta_A)
    lam = _finite_scalar("lam", lam)
    eta = _finite_scalar("eta", eta)
    gamma = _finite_scalar("gamma", gamma)

    if (
        t_max < 0.0
        or mu < 0.0
        or gamma < 0.0
        or max_events < 0
    ):
        raise ValueError(
            "Les paramètres de taux, temps et compteurs doivent être positifs."
        )

    if J_anchor is None:
        J_anchor = J.copy()
    else:
        J_anchor = _validate_J(J_anchor)

    if J_anchor.shape != J.shape:
        raise ValueError(
            "J_anchor doit avoir la même forme que J0."
        )

    if rng is None:
        rng = np.random.default_rng()

    events: list[dict] = []

    t = 0.0

    majorant = float(
        X.size * mu
    )

    if majorant == 0.0 or t_max == 0.0 or max_events == 0:
        return events

    for _ in range(max_events):

        waiting = float(
            rng.exponential(1.0 / majorant)
        )

        t_candidate = t + waiting

        if t_candidate > t_max:
            break

        J_candidate = dissipate_J(
            J,
            J_anchor,
            dt=waiting,
            gamma=gamma,
        )

        rates = transition_rates(
            X=X,
            J=J_candidate,
            mu=mu,
            beta=beta,
            theta_A=theta_A,
            lam=lam,
            fnk_fn=fnk_fn,
        )

        total_rate = float(
            np.sum(rates)
        )

        acceptance = min(
            1.0,
            max(0.0, total_rate / majorant),
        )

        if rng.random() < acceptance and total_rate > 0.0:

            k = int(
                rng.choice(
                    X.size,
                    p=rates / total_rate,
                )
            )

            X_before = X.copy()
            J_before = J_candidate.copy()

            X_after, J_after = mutate_postflip(
                X,
                J_candidate,
                k=k,
                eta=eta,
            )

            X = X_after
            J = J_after

            t = t_candidate

            events.append(
                {
                    "t": t,
                    "site": k,
                    "X_before": X_before,
                    "J_before": J_before,
                    "X_after": X.copy(),
                    "J_after": J.copy(),
                    "rates": rates.copy(),
                    "total_rate": total_rate,
                    "accepted": True,
                }
            )

        else:

            J = J_candidate
            t = t_candidate

    return events


# ============================================================
# 11. SOLVEUR EXACT PAR MORCEAUX POUR lambda = 0
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
    rng: np.random.Generator | None = None,
    max_events: int = 100000,
    fnk_fn: FitnessFn = zero_fnk,
) -> list[dict]:
    r"""
    Simulation exacte par morceaux pour lambda = 0.

    Entre deux mutations, J(t) dissipe vers J_anchor.
    Le masque d'accessibilité est constant par morceaux
    en z = exp(-gamma * Delta t).
    """

    X = _validate_state(X0).copy()
    J = _validate_J(J0).copy()

    if J.shape != (X.size, X.size):
        raise ValueError(
            "Dimensions incompatibles entre X0 et J0."
        )

    t_max = _finite_scalar("t_max", t_max)
    mu = _finite_scalar("mu", mu)
    beta = _finite_scalar("beta", beta)
    theta_A = _finite_scalar("theta_A", theta_A)
    eta = _finite_scalar("eta", eta)
    gamma = _finite_scalar("gamma", gamma)

    if (
        t_max < 0.0
        or mu < 0.0
        or gamma < 0.0
        or max_events < 0
    ):
        raise ValueError(
            "Paramètres non valides."
        )

    if J_anchor is None:
        J_anchor = J.copy()
    else:
        J_anchor = _validate_J(J_anchor)

    if J_anchor.shape != J.shape:
        raise ValueError(
            "J_anchor doit avoir la même forme que J0."
        )

    if rng is None:
        rng = np.random.default_rng()

    events: list[dict] = []

    t = 0.0

    if t_max == 0.0 or mu == 0.0 or max_events == 0:
        return events

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

            total_rate = float(
                np.sum(rates)
            )

            if total_rate <= 0.0:
                break

            waiting = float(
                rng.exponential(1.0 / total_rate)
            )

            if waiting >= remaining:
                break

            k = int(
                rng.choice(
                    X.size,
                    p=rates / total_rate,
                )
            )

            X_before = X.copy()
            J_before = J.copy()

            X, J = mutate_postflip(
                X,
                J,
                k=k,
                eta=eta,
            )

            t += waiting
            event_count += 1

            events.append(
                {
                    "t": t,
                    "site": k,
                    "X_before": X_before,
                    "J_before": J_before,
                    "X_after": X.copy(),
                    "J_after": J.copy(),
                    "rates": rates.copy(),
                    "total_rate": total_rate,
                    "accepted": True,
                }
            )

            continue

        # --------------------------------------------------------
        # Cas dissipatif : z = exp(-gamma * (tau - t))
        # --------------------------------------------------------

        z_start = 1.0

        z_end = float(
            exp(-gamma * remaining)
        )

        crossings = solve_threshold_crossings_z(
            J0=J,
            J_anchor=J_anchor,
            z0=z_start,
            z1=z_end,
            theta_A=theta_A,
        )

        boundaries = [z_start]

        boundaries.extend(
            z
            for z, _, _ in crossings
            if z_end < z < z_start
        )

        boundaries.append(z_end)

        boundaries = sorted(
            set(boundaries),
            reverse=True,
        )

        mutation_occurred = False

        J_ref = J.copy()
        t_ref = t

        for z_a, z_b in zip(
            boundaries[:-1],
            boundaries[1:],
        ):

            if z_a <= 0.0 or z_b <= 0.0:
                continue

            dt_segment = (
                log(z_a / z_b) / gamma
            )

            if dt_segment <= 0.0:
                continue

            z_mid = 0.5 * (z_a + z_b)

            J_mid = _validate_J(
                J_anchor
                + (J_ref - J_anchor) * z_mid
            )

            rates = transition_rates(
                X=X,
                J=J_mid,
                mu=mu,
                beta=beta,
                theta_A=theta_A,
                lam=0.0,
                fnk_fn=fnk_fn,
            )

            total_rate = float(
                np.sum(rates)
            )

            if total_rate > 0.0:

                waiting = float(
                    rng.exponential(1.0 / total_rate)
                )

                if waiting < dt_segment:

                    t_event = t + waiting

                    tau = t_event - t_ref

                    J_event = dissipate_J(
                        J_ref,
                        J_anchor,
                        dt=tau,
                        gamma=gamma,
                    )

                    k = int(
                        rng.choice(
                            X.size,
                            p=rates / total_rate,
                        )
                    )

                    X_before = X.copy()
                    J_before = J_event.copy()

                    X, J = mutate_postflip(
                        X,
                        J_event,
                        k=k,
                        eta=eta,
                    )

                    t = t_event
                    event_count += 1
                    mutation_occurred = True

                    events.append(
                        {
                            "t": t,
                            "site": k,
                            "X_before": X_before,
                            "J_before": J_before,
                            "X_after": X.copy(),
                            "J_after": J.copy(),
                            "rates": rates.copy(),
                            "total_rate": total_rate,
                            "accepted": True,
                        }
                    )

                    break

            # Aucun saut : progression jusqu'au segment suivant.

            t += dt_segment

            J = dissipate_J(
                J_ref,
                J_anchor,
                dt=t - t_ref,
                gamma=gamma,
            )

        if mutation_occurred:
            continue

        J = dissipate_J(
            J_ref,
            J_anchor,
            dt=t_max - t_ref,
            gamma=gamma,
        )

        t = t_max
        break

    return events


# ============================================================
# 12. WASSERSTEIN-1 EMPIRIQUE EN DIMENSION 1
# ============================================================

def wasserstein_1d(
    x: Sequence[float],
    y: Sequence[float],
) -> float:
    r"""
    Distance de Wasserstein-1 unidimensionnelle
    sur deux échantillons empiriques.
    """

    x_arr = np.sort(
        np.asarray(x, dtype=float).ravel()
    )

    y_arr = np.sort(
        np.asarray(y, dtype=float).ravel()
    )

    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError(
            "Les deux échantillons doivent être non vides."
        )

    if (
        not np.all(np.isfinite(x_arr))
        or not np.all(np.isfinite(y_arr))
    ):
        raise ValueError(
            "Les échantillons doivent contenir des valeurs finies."
        )

    i = 0
    j = 0

    mass_x = 1.0 / x_arr.size
    mass_y = 1.0 / y_arr.size

    remaining_x = mass_x
    remaining_y = mass_y

    distance = 0.0

    while i < x_arr.size and j < y_arr.size:

        transported = min(
            remaining_x,
            remaining_y,
        )

        distance += (
            transported
            * abs(x_arr[i] - y_arr[j])
        )

        remaining_x -= transported
        remaining_y -= transported

        if remaining_x <= 1e-15:
            i += 1
            remaining_x = (
                mass_x if i < x_arr.size else 0.0
            )

        if remaining_y <= 1e-15:
            j += 1
            remaining_y = (
                mass_y if j < y_arr.size else 0.0
            )

    return float(distance)


# ============================================================
# 13. SLICED WASSERSTEIN-1
# ============================================================

def sliced_wasserstein_1(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 128,
    rng: np.random.Generator | None = None,
) -> float:
    """Distance Sliced Wasserstein-1 par projections aléatoires."""

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(
            "X et Y doivent être bidimensionnels."
        )

    if X.shape[0] == 0 or Y.shape[0] == 0:
        raise ValueError(
            "X et Y doivent contenir au moins un point."
        )

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "X et Y doivent avoir la même dimension de caractéristiques."
        )

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError(
            "X et Y doivent contenir des valeurs finies."
        )

    if n_projections < 1:
        raise ValueError(
            "n_projections doit être >= 1."
        )

    if rng is None:
        rng = np.random.default_rng()

    dimension = X.shape[1]

    directions = rng.normal(
        size=(n_projections, dimension)
    )

    norms = np.linalg.norm(
        directions,
        axis=1,
        keepdims=True,
    )

    zero_rows = norms[:, 0] == 0.0

    while np.any(zero_rows):

        directions[zero_rows] = rng.normal(
            size=(
                int(np.sum(zero_rows)),
                dimension,
            )
        )

        norms = np.linalg.norm(
            directions,
            axis=1,
            keepdims=True,
        )

        zero_rows = norms[:, 0] == 0.0

    directions /= norms

    distances = [
        wasserstein_1d(
            X @ direction,
            Y @ direction,
        )
        for direction in directions
    ]

    return float(
        np.mean(distances)
    )


# ============================================================
# 14. ENERGY DISTANCE
# ============================================================

def energy_distance(
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    r"""
    Energy distance empirique entre deux distributions
    multivariées.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(
            "X et Y doivent être bidimensionnels."
        )

    if X.shape[0] == 0 or Y.shape[0] == 0:
        raise ValueError(
            "X et Y doivent contenir au moins un point."
        )

    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Dimensions incompatibles entre X et Y."
        )

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError(
            "X et Y doivent contenir des valeurs finies."
        )

    def mean_pairwise_dist(
        A: np.ndarray,
        B: np.ndarray,
    ) -> float:

        diffs = (
            A[:, None, :]
            - B[None, :, :]
        )

        return float(
            np.mean(
                np.linalg.norm(
                    diffs,
                    axis=-1,
                )
            )
        )

    cross = mean_pairwise_dist(X, Y)
    within_x = mean_pairwise_dist(X, X)
    within_y = mean_pairwise_dist(Y, Y)

    squared_dist = (
        2.0 * cross
        - within_x
        - within_y
    )

    return float(
        sqrt(max(0.0, squared_dist))
    )
