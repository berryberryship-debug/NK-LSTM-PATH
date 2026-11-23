"""
src/trajectories.py -- Dynamique jointe (mutation + transport post-flip + dissipation).
Taux : w_k = mu * a_k(J) * min(1, exp(beta*dF_k)), dF_k = F(X^(k),J) - F(X,J) (J pre-transport).
F(X,J) = F_NK(X) + lam * sum_{i<j} J_ij s_i s_j.   Dissipation : J(t)=Jref+(J-Jref)exp(-gamma dt).
  - simulate_thinning  : amincissement d'Ogata, horloge majorante N*mu (tout lam).
  - simulate_piecewise : exact par morceaux, lam = 0 uniquement (racines d'un polynome de degre 2 en z=exp(-gamma tau)).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import fiber_bundle as fb


@dataclass(frozen=True)
class Params:
    mu: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5
    eta: float = 0.3
    lam: float = 0.0
    theta_A: float = 0.4


def delta_F(X, J, k, land, lam):
    Xk = np.array(X, float); Xk[k] = -Xk[k]
    dF = land.F(Xk) - land.F(X)
    if lam != 0.0:
        dF += -2.0 * lam * X[k] * float(J[k] @ X)   # J_kk = 0
    return dF


def rates(X, J, p: Params, land):
    N = len(X)
    A = np.sum(J * J, axis=1)
    w = np.zeros(N)
    for k in range(N):
        if A[k] > p.theta_A:
            w[k] = p.mu * np.exp(min(0.0, p.beta * delta_F(X, J, k, land, p.lam)))
    return w


def simulate_thinning(X0, J0, p: Params, land, T, rng, psi_fn=fb.psi_sym):
    X = np.array(X0, float); Jref = np.array(J0, float); Jc = Jref.copy()
    N = len(X); lam_max = N * p.mu
    t = 0.0; events = []
    while True:
        dt = rng.exponential(1.0 / lam_max)
        if t + dt > T:
            break
        t += dt
        Jc = Jref + (Jc - Jref) * np.exp(-p.gamma * dt)
        w = rates(X, Jc, p, land); tot = w.sum()
        if tot > lam_max * (1 + 1e-12):
            raise AssertionError("Borne de l'amincissement violée.")
        if rng.uniform(0.0, lam_max) <= tot:
            k = int(np.searchsorted(np.cumsum(w), rng.uniform(0.0, tot), side="right"))
            k = min(k, N - 1)
            X, Jc = fb.mutate_postflip(X, Jc, k, p.eta, psi_fn)
            events.append((t, k, X.copy(), Jc.copy()))
    return events


def _breakpoints(D, Jref, theta, gamma):
    N = D.shape[0]; bps = []
    for k in range(N):
        idx = [j for j in range(N) if j != k]
        d, j0 = D[k, idx], Jref[k, idx]
        a2, a1, a0 = float(d @ d), 2.0 * float(j0 @ d), float(j0 @ j0) - theta
        if abs(a2) < 1e-14:
            roots = [-a0 / a1] if abs(a1) > 1e-14 else []
        else:
            disc = a1 * a1 - 4 * a2 * a0
            roots = [] if disc < 0 else [(-a1 + s * np.sqrt(disc)) / (2 * a2) for s in (1, -1)]
        for z in roots:
            if 0.0 < z < 1.0:
                bps.append(-np.log(z) / gamma)
    return sorted(set(bps))


def simulate_piecewise(X0, J0, p: Params, land, T, rng, psi_fn=fb.psi_sym):
    if p.lam != 0.0:
        raise ValueError("simulate_piecewise : lam = 0 uniquement.")
    X = np.array(X0, float); Jref = np.array(J0, float); Jc = Jref.copy()
    N = len(X); t = 0.0; events = []
    while t < T:
        D = Jc - Jref
        c = np.array([np.exp(min(0.0, p.beta * delta_F(X, Jc, k, land, 0.0))) for k in range(N)])
        edges = [0.0] + _breakpoints(D, Jref, p.theta_A, p.gamma) + [np.inf]
        E = rng.exponential(); tau_ev = None; weights = None
        for s, e in zip(edges[:-1], edges[1:]):
            mid = s + 1.0 if np.isinf(e) else 0.5 * (s + e)
            Jm = Jref + D * np.exp(-p.gamma * mid)
            wts = p.mu * c * (np.sum(Jm * Jm, axis=1) > p.theta_A)
            rate = wts.sum()
            if rate == 0.0:
                if np.isinf(e):
                    break
                continue
            if np.isinf(e) or rate * (e - s) >= E:
                tau_ev = s + E / rate; weights = wts; break
            E -= rate * (e - s)
        if tau_ev is None or t + tau_ev > T:
            break
        t += tau_ev
        Jt = Jref + D * np.exp(-p.gamma * tau_ev)
        k = min(int(np.searchsorted(np.cumsum(weights), rng.uniform(0.0, weights.sum()), side="right")), N - 1)
        X, Jc = fb.mutate_postflip(X, Jt, k, p.eta, psi_fn)
        events.append((t, k, X.copy(), Jc.copy()))
    return events


def fiber_at(events, J0, p: Params, t, J_init=None):
    """Fibre à l'instant t (relaxation exacte depuis le dernier événement <= t)."""
    Jref = np.array(J0, float)
    t_last, J_last = 0.0, (Jref if J_init is None else np.array(J_init, float))
    for (te, _, _, Je) in events:
        if te <= t:
            t_last, J_last = te, Je
        else:
            break
    return Jref + (J_last - Jref) * np.exp(-p.gamma * (t - t_last))
