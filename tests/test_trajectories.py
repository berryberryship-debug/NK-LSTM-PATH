"""
tests/test_trajectories.py
Projet : NK-LSTM-PATH

Suite de validation de `src/trajectories.py` :
  1. Borne supérieure uniforme Lambda(X, J) <= N * mu et invariance de Delta F_k
     entre fibres distinctes à lambda = 0.
  2. Dissipation analytique exacte (propriété de semi-groupe, symétrie, limites).
  3. Transport post-flip lors des mutations (flip avant Psi, dissipation pure des
     arêtes non incidentes).
  4. Lois élémentaires et relation exacte de bilan détaillé de Metropolis :
     w(X -> X^{(k)}) / w(X^{(k)} -> X) = exp(beta * Delta F_k).
  5. Stabilité numérique du solveur quadratique à mise à l'échelle relative
     (coefficients 1e-16 et 1e+16, coefficient quadratique 0 < c2 / S < 1e-14
     sans réduction artificielle du degré tant que c2 / S ne sous-déborde pas
     en binary64, racines proches 0.5 +- 1e-7, racines proches des bornes,
     tangence avec tolérance d'arrondi documentée, exclusion des bornes {0, 1}).
  6. Conservation du risque intégré int_0^t Lambda(u) du = E à travers un changement
     de masque d'accessibilité :
       - fermeture d'un canal (3*mu -> 2*mu à tau_c = ln(2)/gamma avec theta_A = 0.25),
       - ouverture retardée de deux canaux (0 -> 2*mu à tau_c = ln(2)/gamma avec theta_A = 0.25).
  7. Validation croisée en loi entre `simulate_ogata_thinning` et
     `simulate_exact_piecewise_lambda0` à lambda = 0 reposant sur :
       (i)   un test de permutation Monte-Carlo à deux échantillons avec correction
             de rang p = (1 + n_>=) / (1 + n_perm) sur n_perm = 9999 tirages
             (résolution minimale 1e-4, seuil alpha = 0.01) ; un résultat p > 0.01
             indique la non-rejection de l'égalité des lois (absence de divergence
             détectée) sans prouver à lui seul leur équivalence,
       (ii)  une règle d'équivalence pratique exigeant que la borne supérieure
             bootstrap percentile q_0.95(D^*), à couverture nominale de 95 %,
             soit strictement inférieure à la marge d'équivalence delta_eq
             définie a priori,
       (iii) des statistiques temporelles calculées individuellement par
             trajectoire : observables globales (T_1, T_dernier, étendue
             T_dernier - T_1, proportion de trajectoires à >= 2 événements) et
             observables conditionnelles (moyenne, médiane et quantile 75 % des
             intervalles inter-sauts conditionnellement à N_events >= 2).
  8. Sélection du canal au cas limite u2 == 0.0 (side="right") vérifiée par
     générateur contrôlé lorsque les premiers canaux ont un taux nul.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence
import unittest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import fiber_bundle as fb  # noqa: E402
import trajectories as traj  # noqa: E402


class _ControlledZeroU2Generator:
    """
    Générateur déterministe minimaliste reproduisant l'interface `uniform(low, high)`
    pour injecter exactement `u2 = 0.0` lors de la sélection du canal tout en
    contrôlant les tirages d'attente temporelle `u1`.
    """

    def __init__(self, u1_sequence: Sequence[float], u2_value: float = 0.0) -> None:
        self._u1_iter = iter(u1_sequence)
        self._u2_value = float(u2_value)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        if low == 0.0 and high == 1.0:
            return float(next(self._u1_iter))
        return self._u2_value


def _mc_permutation_test_1d(
    u_samples: np.ndarray,
    v_samples: np.ndarray,
    n_permutations: int = 9999,
    seed: int = 2026,
    chunk_size: int = 500,
) -> tuple[float, float]:
    """
    Test de permutation Monte-Carlo à deux échantillons (tailles n_u, n_v quelconques)
    pour W_1 avec correction de rang p = (1 + n_>=) / (1 + n_perm).
    """
    u = np.asarray(u_samples, dtype=np.float64).ravel()
    v = np.asarray(v_samples, dtype=np.float64).ravel()
    n_u = u.size
    n_v = v.size
    w1_obs = traj.wasserstein_1d(u, v)

    pooled = np.concatenate([u, v])
    n_tot = pooled.size
    order = np.argsort(pooled, kind="mergesort")
    deltas = np.diff(pooled[order])
    if deltas.size == 0:
        return 0.0, 1.0

    rng = np.random.default_rng(seed)
    ge_count = 0
    done = 0
    while done < n_permutations:
        m = min(chunk_size, n_permutations - done)
        noise = rng.random((m, n_tot))
        perms = np.argpartition(noise, kth=n_u - 1, axis=1)
        is_u = np.zeros((m, n_tot), dtype=np.bool_)
        rows = np.arange(m)[:, None]
        is_u[rows, perms[:, :n_u]] = True

        is_u_sorted = is_u[:, order][:, :-1]
        cdf_u = np.cumsum(is_u_sorted, axis=1) / float(n_u)
        cdf_v = np.cumsum(~is_u_sorted, axis=1) / float(n_v)
        w1_perm = np.abs(cdf_u - cdf_v) @ deltas
        ge_count += int(np.sum(w1_perm >= w1_obs))
        done += m

    p_value = (1.0 + ge_count) / (1.0 + n_permutations)
    return w1_obs, p_value


def _bootstrap_percentile_ucb95_1d(
    u_samples: np.ndarray,
    v_samples: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 2026,
) -> float:
    """
    Borne supérieure bootstrap percentile q_0.95(W_1^*) à couverture nominale
    de 95 %, obtenue par rééchantillonnage indépendant au niveau des trajectoires.
    """
    u = np.asarray(u_samples, dtype=np.float64).ravel()
    v = np.asarray(v_samples, dtype=np.float64).ravel()
    n_u = u.size
    n_v = v.size
    rng = np.random.default_rng(seed)

    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        iu = rng.integers(0, n_u, size=n_u)
        iv = rng.integers(0, n_v, size=n_v)
        boot_stats[b] = traj.wasserstein_1d(u[iu], v[iv])
    return float(np.quantile(boot_stats, 0.95))


def _mc_permutation_sliced_w1(
    U: np.ndarray,
    V: np.ndarray,
    n_projections: int = 32,
    n_permutations: int = 9999,
    seed: int = 2026,
    chunk_size: int = 500,
) -> tuple[float, float]:
    """
    Test de permutation Monte-Carlo à deux échantillons (n_perm = 9999,
    correction de rang) pour Sliced Wasserstein-1 sur R^d.
    """
    n_u = U.shape[0]
    n_v = V.shape[0]
    d = U.shape[1]
    sw1_obs = traj.sliced_wasserstein_1(U, V, n_projections=n_projections, seed=42)

    rng_dir = np.random.default_rng(42)
    dirs = rng_dir.normal(0.0, 1.0, size=(n_projections, d))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-15)

    pooled = np.vstack([U, V])
    n_tot = pooled.shape[0]
    proj = pooled @ dirs.T

    orders = [np.argsort(proj[:, q], kind="mergesort") for q in range(n_projections)]
    deltas_list = [np.diff(proj[orders[q], q]) for q in range(n_projections)]

    rng = np.random.default_rng(seed)
    ge_count = 0
    done = 0
    while done < n_permutations:
        m = min(chunk_size, n_permutations - done)
        noise = rng.random((m, n_tot))
        perms = np.argpartition(noise, kth=n_u - 1, axis=1)
        is_u = np.zeros((m, n_tot), dtype=np.bool_)
        rows = np.arange(m)[:, None]
        is_u[rows, perms[:, :n_u]] = True

        acc = np.zeros(m, dtype=np.float64)
        for q in range(n_projections):
            is_u_sorted = is_u[:, orders[q]][:, :-1]
            cdf_u = np.cumsum(is_u_sorted, axis=1) / float(n_u)
            cdf_v = np.cumsum(~is_u_sorted, axis=1) / float(n_v)
            acc += np.abs(cdf_u - cdf_v) @ deltas_list[q]

        sw1_perm = acc / float(n_projections)
        ge_count += int(np.sum(sw1_perm >= sw1_obs))
        done += m

    p_value = (1.0 + ge_count) / (1.0 + n_permutations)
    return sw1_obs, p_value


def _bootstrap_percentile_ucb95_sliced_w1(
    U: np.ndarray,
    V: np.ndarray,
    n_projections: int = 32,
    n_bootstrap: int = 1000,
    seed: int = 2026,
) -> float:
    """Borne supérieure bootstrap percentile q_0.95(Sliced-W1^*) (couverture nominale 95 %)."""
    n_u = U.shape[0]
    n_v = V.shape[0]
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        iu = rng.integers(0, n_u, size=n_u)
        iv = rng.integers(0, n_v, size=n_v)
        boot_stats[b] = traj.sliced_wasserstein_1(
            U[iu], V[iv], n_projections=n_projections, seed=42
        )
    return float(np.quantile(boot_stats, 0.95))


def _mc_permutation_energy_blocked(
    U: np.ndarray,
    V: np.ndarray,
    n_permutations: int = 9999,
    block_size: int = 64,
    perm_batch_size: int = 250,
    seed: int = 2026,
) -> tuple[float, float]:
    """
    Test de permutation Monte-Carlo à deux échantillons (n_perm = 9999,
    correction de rang p = (1 + n_>=) / (1 + n_perm)) pour la distance d'énergie
    D_E(U, V).
    Les masques d'appartenance booléens `in_u` de dimension (n_permutations, n_tot)
    sont générés une seule fois par tranches de taille B_p = `perm_batch_size`,
    puis réutilisés lors de l'unique parcours des blocs de distances euclidiennes
    de taille B_r = `block_size`.
    L'empreinte mémoire auxiliaire est bornée en :
        O(n_permutations * n_tot [bool] + B_p * n_tot [float64]
          + B_r * n_tot * d + B_p * B_r + n_permutations).
    """
    ed_obs = traj.energy_distance(U, V, block_size=block_size)
    n_u = U.shape[0]
    n_v = V.shape[0]
    pooled = np.vstack([U, V])
    n_tot = pooled.shape[0]

    # Génération unique des masques d'appartenance booléens par tranches
    in_u = np.zeros((n_permutations, n_tot), dtype=np.bool_)
    rng = np.random.default_rng(seed)
    for p0 in range(0, n_permutations, perm_batch_size):
        p1 = min(p0 + perm_batch_size, n_permutations)
        m = p1 - p0
        noise = rng.random((m, n_tot))
        perms = np.argpartition(noise, kth=n_u - 1, axis=1)
        rows = np.arange(m)[:, None]
        in_u[p0:p1][rows, perms[:, :n_u]] = True

    sum_uu = np.zeros(n_permutations, dtype=np.float64)
    sum_vv = np.zeros(n_permutations, dtype=np.float64)
    sum_uv = np.zeros(n_permutations, dtype=np.float64)

    # Parcours unique des blocs de lignes de distances euclidiennes
    for i0 in range(0, n_tot, block_size):
        i1 = min(i0 + block_size, n_tot)
        diff = pooled[i0:i1, None, :] - pooled[None, :, :]
        dist_block = np.linalg.norm(diff, axis=-1)  # (B_r, n_tot)
        s_all = np.sum(dist_block, axis=1)          # (B_r,)

        for p0 in range(0, n_permutations, perm_batch_size):
            p1 = min(p0 + perm_batch_size, n_permutations)
            in_u_batch = in_u[p0:p1].astype(np.float64)  # (B_p, n_tot)
            row_in_u = in_u_batch[:, i0:i1]              # (B_p, B_r)
            row_in_v = 1.0 - row_in_u                    # (B_p, B_r)

            s_u = dist_block @ in_u_batch.T              # (B_r, B_p)
            s_v = s_all[:, None] - s_u                   # (B_r, B_p)

            sum_uu[p0:p1] += np.sum(row_in_u * s_u.T, axis=1)
            sum_uv[p0:p1] += np.sum(row_in_u * s_v.T, axis=1)
            sum_vv[p0:p1] += np.sum(row_in_v * s_v.T, axis=1)

    e_xy = sum_uv / float(n_u * n_v)
    e_xx = sum_uu / float(n_u * (n_u - 1))
    e_yy = sum_vv / float(n_v * (n_v - 1))
    ed_perm = np.sqrt(np.maximum(0.0, 2.0 * e_xy - e_xx - e_yy))

    ge_count = int(np.sum(ed_perm >= ed_obs))
    p_value = (1.0 + ge_count) / (1.0 + n_permutations)
    return ed_obs, p_value


def _bootstrap_percentile_ucb95_energy_blocked(
    U: np.ndarray,
    V: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int = 128,
    seed: int = 2026,
) -> float:
    """Borne supérieure bootstrap percentile q_0.95(D_E^*) (couverture nominale 95 %)."""
    n_u = U.shape[0]
    n_v = V.shape[0]
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        iu = rng.integers(0, n_u, size=n_u)
        iv = rng.integers(0, n_v, size=n_v)
        boot_stats[b] = traj.energy_distance(U[iu], V[iv], block_size=block_size)
    return float(np.quantile(boot_stats, 0.95))


def _mc_permutation_cov_fro(
    U: np.ndarray,
    V: np.ndarray,
    n_permutations: int = 9999,
    seed: int = 2026,
) -> tuple[float, float]:
    """
    Test de permutation Monte-Carlo à deux échantillons (n_perm = 9999,
    correction de rang) pour la norme de Frobenius de l'écart des matrices
    de covariance empiriques.
    """
    cov_obs = float(
        np.linalg.norm(np.cov(U, rowvar=False) - np.cov(V, rowvar=False), ord="fro")
    )
    n_u = U.shape[0]
    pooled = np.vstack([U, V])
    n_tot = pooled.shape[0]
    rng = np.random.default_rng(seed)

    ge_count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n_tot)
        c_u = np.cov(pooled[perm[:n_u]], rowvar=False)
        c_v = np.cov(pooled[perm[n_u:]], rowvar=False)
        if float(np.linalg.norm(c_u - c_v, ord="fro")) >= cov_obs:
            ge_count += 1

    p_value = (1.0 + ge_count) / (1.0 + n_permutations)
    return cov_obs, p_value


def _bootstrap_percentile_ucb95_cov_fro(
    U: np.ndarray,
    V: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 2026,
) -> float:
    """Borne supérieure bootstrap percentile q_0.95(||Sigma_U - Sigma_V||_F^*) (95 % nominal)."""
    n_u = U.shape[0]
    n_v = V.shape[0]
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        iu = rng.integers(0, n_u, size=n_u)
        iv = rng.integers(0, n_v, size=n_v)
        c_u = np.cov(U[iu], rowvar=False)
        c_v = np.cov(V[iv], rowvar=False)
        boot_stats[b] = float(np.linalg.norm(c_u - c_v, ord="fro"))
    return float(np.quantile(boot_stats, 0.95))


class TestTrajectories(unittest.TestCase):
    """Validation du simulateur joint et du contrôle analytique par morceaux."""

    def test_01_rate_upper_bound_and_lambda0_invariance(self) -> None:
        """
        Vérifie :
        - 0 <= w_k <= mu et Sum_k w_k <= N * mu,
        - l'invariance exacte de Delta F_k entre deux matrices J distinctes à lambda = 0,
        - la concordance entre Delta F_k et F(X^{(k)}, J) - F(X, J) pour lambda != 0.
        """
        rng = np.random.default_rng(101)
        for n, k_nk in ((3, 2), (6, 3)):
            nk = traj.NKLandscape(n=n, k=k_nk, seed=7)
            mu = 2.4
            beta = 3.5
            for _ in range(30):
                X = rng.choice([-1.0, 1.0], size=n)
                raw = rng.normal(0.0, 0.8, size=(n, n))
                J = 0.5 * (raw + raw.T)
                np.fill_diagonal(J, 0.0)

                raw2 = rng.normal(0.0, 1.5, size=(n, n))
                J2 = 0.5 * (raw2 + raw2.T)
                np.fill_diagonal(J2, 0.0)
                for node in range(n):
                    df_1 = traj.delta_fitness_k(X, J, k=node, lam=0.0, fnk_fn=nk)
                    df_2 = traj.delta_fitness_k(X, J2, k=node, lam=0.0, fnk_fn=nk)
                    self.assertAlmostEqual(df_1, df_2, places=15)

                lam = 1.7
                f_before = traj.extended_fitness(X, J, lam=lam, fnk_fn=nk)
                for node in range(n):
                    X_flip = X.copy()
                    X_flip[node] = -X_flip[node]
                    f_after = traj.extended_fitness(X_flip, J, lam=lam, fnk_fn=nk)
                    df_analytic = traj.delta_fitness_k(X, J, k=node, lam=lam, fnk_fn=nk)
                    self.assertAlmostEqual(df_analytic, f_after - f_before, places=13)

                for theta_A in (-1.0, 0.2, 2.0):
                    rates = traj.transition_rates(
                        X, J, mu=mu, beta=beta, lam=lam, theta_A=theta_A, fnk_fn=nk
                    )
                    self.assertTrue(np.all(rates >= 0.0))
                    self.assertTrue(np.all(rates <= mu + 1e-15))
                    self.assertLessEqual(float(np.sum(rates)), n * mu + 1e-14)

    def test_02_analytical_dissipation(self) -> None:
        """Vérifie la propriété de semi-groupe D_{t2}(D_{t1}(J)) = D_{t1+t2}(J) et les limites."""
        J_anchor = fb.unpack_J((0.30, -0.45, 0.20), n=3)
        J_start = fb.unpack_J((-0.80, 0.90, -0.50), n=3)
        gamma = 1.35

        dt1, dt2 = 0.4, 0.7
        J_mid = traj.dissipate_J(J_start, J_anchor, gamma=gamma, dt=dt1)
        J_seq = traj.dissipate_J(J_mid, J_anchor, gamma=gamma, dt=dt2)
        J_dir = traj.dissipate_J(J_start, J_anchor, gamma=gamma, dt=dt1 + dt2)

        np.testing.assert_allclose(J_seq, J_dir, atol=1e-15)
        np.testing.assert_allclose(J_dir, J_dir.T, atol=1e-15)
        np.testing.assert_allclose(np.diag(J_dir), 0.0, atol=1e-15)

        J_nodiss = traj.dissipate_J(J_start, J_anchor, gamma=0.0, dt=5.0)
        np.testing.assert_allclose(J_nodiss, J_start, atol=1e-15)

        J_asymp = traj.dissipate_J(J_start, J_anchor, gamma=gamma, dt=40.0)
        np.testing.assert_allclose(J_asymp, J_anchor, atol=1e-15)

    def test_03_postflip_transport_along_trajectory(self) -> None:
        """Vérifie événement par événement la dissipation, le flip et le transport post-flip."""
        X0 = np.array([1.0, -1.0, 1.0])
        J0 = fb.unpack_J((0.70, -0.60, 0.80), n=3)
        anchor = fb.unpack_J((0.40, 0.40, 0.40), n=3)
        gamma = 0.5
        eta = 0.15

        res = traj.simulate_ogata_thinning(
            X0=X0,
            J0=J0,
            J_anchor=anchor,
            t_max=3.0,
            mu=3.0,
            gamma=gamma,
            beta=0.0,
            lam=0.0,
            eta=eta,
            theta_A=-1.0,
            rng=np.random.default_rng(42),
        )
        self.assertGreater(res.n_accepted, 5)

        prev_t = 0.0
        prev_X = X0.copy()
        prev_J = J0.copy()

        for ev in res.events:
            dt = ev.time - prev_t
            J_pre_jump = traj.dissipate_J(prev_J, anchor, gamma=gamma, dt=dt)
            X_expected, J_expected = fb.mutate_postflip(
                prev_X, J_pre_jump, k=ev.node, eta=eta, psi_fn=fb.psi_sym
            )
            np.testing.assert_array_equal(ev.X, X_expected)
            np.testing.assert_allclose(ev.J, J_expected, atol=1e-14)

            other_nodes = [i for i in range(3) if i != ev.node]
            u, v = other_nodes[0], other_nodes[1]
            self.assertAlmostEqual(ev.J[u, v], J_pre_jump[u, v], places=14)

            prev_t = ev.time
            prev_X = ev.X.copy()
            prev_J = ev.J.copy()

    def test_04_metropolis_detailed_balance_and_event_laws(self) -> None:
        """
        Vérifie :
        1. La relation exacte de bilan détaillé de Metropolis à fibre J fixée et
           nœuds accessibles :
               w(X -> X^{(k)}) / w(X^{(k)} -> X) = exp(beta * Delta F_k(X, J)).
        2. Le silence total sous seuil inaccessible.
        3. La loi de Poisson(N * mu * t_max) et l'uniformité des canaux en régime neutre.
        """
        nk = traj.NKLandscape(n=4, k=2, seed=31)
        rng = np.random.default_rng(55)
        mu = 1.9
        beta = 2.3
        lam = 1.4

        for _ in range(25):
            X = rng.choice([-1.0, 1.0], size=4)
            raw = rng.normal(0.0, 0.7, size=(4, 4))
            J = 0.5 * (raw + raw.T)
            np.fill_diagonal(J, 0.0)

            rates_fwd = traj.transition_rates(
                X, J, mu=mu, beta=beta, lam=lam, theta_A=-1.0, fnk_fn=nk
            )
            for k in range(4):
                X_flip = X.copy()
                X_flip[k] = -X_flip[k]
                rates_bwd = traj.transition_rates(
                    X_flip, J, mu=mu, beta=beta, lam=lam, theta_A=-1.0, fnk_fn=nk
                )
                w_fwd = float(rates_fwd[k])
                w_bwd = float(rates_bwd[k])
                self.assertGreater(w_fwd, 0.0)
                self.assertGreater(w_bwd, 0.0)

                df_k = traj.delta_fitness_k(X, J, k=k, lam=lam, fnk_fn=nk)
                expected_ratio = float(np.exp(beta * df_k))
                self.assertAlmostEqual(w_fwd / w_bwd, expected_ratio, places=12)

        X0 = np.array([1.0, 1.0, 1.0])
        J0 = fb.unpack_J((0.30, -0.20, 0.40), n=3)
        res_silent = traj.simulate_ogata_thinning(
            X0=X0,
            J0=J0,
            t_max=5.0,
            mu=4.0,
            gamma=1.0,
            beta=1.0,
            lam=1.0,
            eta=0.2,
            theta_A=10.0,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(res_silent.n_accepted, 0)

        rng_pois = np.random.default_rng(123)
        counts = []
        node_hist = np.zeros(3, dtype=np.int64)
        for _ in range(1500):
            res = traj.simulate_ogata_thinning(
                X0=X0,
                J0=J0,
                t_max=2.0,
                mu=2.0,
                gamma=0.5,
                beta=0.0,
                lam=0.0,
                eta=0.0,
                theta_A=-1.0,
                rng=rng_pois,
            )
            counts.append(res.n_accepted)
            for ev in res.events:
                node_hist[ev.node] += 1

        self.assertAlmostEqual(float(np.mean(counts)), 12.0, delta=0.35)
        np.testing.assert_allclose(
            node_hist / float(np.sum(node_hist)), np.full(3, 1.0 / 3.0), atol=0.02
        )

    def test_05_quadratic_solver_conditioning_and_root_preservation(self) -> None:
        """
        Vérifie `_stable_quadratic_real_roots` et `classify_threshold_crossings_z` :
        - invariance d'échelle de 1e-16 à 1e+16,
        - conservation des racines intérieures lorsque c2 / S = 1e-15 (sans
          réduction artificielle du degré tant que c2 / S ne sous-déborde pas
          en binary64),
        - deux racines très proches sans être identiques (z = 0.5 +- 1e-7),
        - racines proches des bornes (1e-11 et 1 - 1e-11),
        - exclusion explicite et documentée des bornes z in {0, 1}.
        """
        for scale in (1e-16, 1.0, 1e16):
            roots = traj._stable_quadratic_real_roots(
                1.0 * scale, -1.0 * scale, 0.1875 * scale
            )
            np.testing.assert_allclose(sorted(roots), [0.25, 0.75], atol=1e-13)

        c2_tiny = 1e-15
        roots_tiny = traj._stable_quadratic_real_roots(c2_tiny, -1.0, 0.4)
        self.assertEqual(len(roots_tiny), 2)
        z_small, z_large = sorted(roots_tiny)
        self.assertAlmostEqual(c2_tiny * (z_small ** 2) - z_small + 0.4, 0.0, places=15)
        self.assertGreater(z_large, 1e14)

        J_anchor = fb.unpack_J((1.0,), n=2)
        J_curr = fb.unpack_J((-1.0,), n=2)
        delta = 1e-7
        z_close = traj.solve_threshold_crossings_z(
            J_curr, J_anchor, theta_A=4.0 * (delta ** 2)
        )
        self.assertEqual(z_close.size, 2)
        np.testing.assert_allclose(
            z_close, np.array([0.5 + delta, 0.5 - delta]), atol=1e-11
        )

        theta_near_edge = (1.0 - 2e-11) ** 2
        info_near = traj.classify_threshold_crossings_z(
            J_curr, J_anchor, theta_A=theta_near_edge
        )
        np.testing.assert_allclose(
            info_near.interior_z, np.array([1.0 - 1e-11, 1e-11]), atol=1e-14
        )
        self.assertEqual(info_near.boundary_z.size, 0)

        info_bound = traj.classify_threshold_crossings_z(J_curr, J_anchor, theta_A=1.0)
        self.assertEqual(info_bound.interior_z.size, 0)
        np.testing.assert_allclose(info_bound.boundary_z, np.array([1.0, 0.0]), atol=1e-15)

    def test_06_integrated_hazard_conservation_across_threshold(self) -> None:
        """
        Vérifie la conservation du risque intégré int_0^t Lambda(u) du = E à travers
        un changement de masque d'accessibilité :
          - Cas 1 (fermeture d'un canal) : J(0) = (1, 1, 0), J^{(0)} = (0, 1, 0),
            theta_A = 0.25. L'accessibilité du nœud médian vaut A_1(t) = e^{-2 gamma t},
            d'où e^{-2 gamma tau_c} = 0.25 <=> tau_c = ln(2) / gamma, et une transition
            de taux 3*mu -> 2*mu à tau_c.
          - Cas 2 (ouverture retardée de deux canaux) : J(0) = (0, 0, 0),
            J^{(0)} = (1, 0, 0), theta_A = 0.25 => transition 0 -> 2*mu à tau_c = ln(2)/gamma.
        """
        mu = 1.0
        gamma = 1.0
        theta_A_close = 0.25
        tau_c = float(np.log(2.0) / gamma)

        X0 = np.array([1.0, 1.0, 1.0])
        J0 = fb.unpack_J((1.0, 1.0, 0.0), n=3)
        J_anchor = fb.unpack_J((0.0, 1.0, 0.0), n=3)

        t_before = 0.40
        t_after = 1.00
        surv_exact_before = float(np.exp(-3.0 * mu * t_before))
        surv_exact_after = float(
            np.exp(-3.0 * mu * tau_c - 2.0 * mu * (t_after - tau_c))
        )

        n_mc = 12_000
        rng_ex = np.random.default_rng(777)
        rng_og = np.random.default_rng(888)
        t1_ex = np.empty(n_mc, dtype=np.float64)
        t1_og = np.empty(n_mc, dtype=np.float64)

        for i in range(n_mc):
            res_ex = traj.simulate_exact_piecewise_lambda0(
                X0=X0,
                J0=J0,
                J_anchor=J_anchor,
                t_max=2.0,
                mu=mu,
                gamma=gamma,
                beta=0.0,
                eta=0.0,
                theta_A=theta_A_close,
                rng=rng_ex,
            )
            t1_ex[i] = res_ex.events[0].time if res_ex.events else np.inf

            res_og = traj.simulate_ogata_thinning(
                X0=X0,
                J0=J0,
                J_anchor=J_anchor,
                t_max=2.0,
                mu=mu,
                gamma=gamma,
                beta=0.0,
                lam=0.0,
                eta=0.0,
                theta_A=theta_A_close,
                rng=rng_og,
            )
            t1_og[i] = res_og.events[0].time if res_og.events else np.inf

        self.assertAlmostEqual(float(np.mean(t1_ex > t_before)), surv_exact_before, delta=0.016)
        self.assertAlmostEqual(float(np.mean(t1_ex > t_after)), surv_exact_after, delta=0.016)
        self.assertAlmostEqual(float(np.mean(t1_og > t_after)), surv_exact_after, delta=0.016)

        J0_open = fb.unpack_J((0.0, 0.0, 0.0), n=3)
        J_anchor_open = fb.unpack_J((1.0, 0.0, 0.0), n=3)
        t_obs = 1.30
        surv_open_exact = float(np.exp(-2.0 * mu * (t_obs - tau_c)))
        rng_open = np.random.default_rng(999)
        zero_jumps = 0
        for _ in range(n_mc):
            res_op = traj.simulate_exact_piecewise_lambda0(
                X0=X0,
                J0=J0_open,
                J_anchor=J_anchor_open,
                t_max=t_obs,
                mu=mu,
                gamma=gamma,
                beta=0.0,
                eta=0.0,
                theta_A=0.25,
                rng=rng_open,
            )
            if res_op.n_accepted == 0:
                zero_jumps += 1
        self.assertAlmostEqual(zero_jumps / n_mc, surv_open_exact, delta=0.016)

    def test_07_cross_validation_equivalence_and_trajectory_temporal_laws(self) -> None:
        """
        Validation croisée en loi entre `simulate_ogata_thinning` et
        `simulate_exact_piecewise_lambda0` à lambda = 0 :
          1. Test de permutation Monte-Carlo à deux échantillons avec correction
             de rang p = (1 + n_>=) / (1 + n_perm) sur n_perm = 9999 tirages
             (résolution minimale 1e-4, seuil alpha = 0.01). p > 0.01 indique
             la non-rejection de l'égalité en loi (absence de divergence détectée),
             sans constituer à lui seul une preuve d'équivalence.
          2. L'argument d'équivalence pratique repose sur la borne supérieure
             bootstrap percentile q_0.95(D^*), à couverture nominale de 95 %
             (dont la couverture finie exacte n'est pas indépendamment établie),
             comparée à la marge d'équivalence delta_eq définie a priori :
                 q_0.95(D^*) < delta_eq.
          3. Statistiques temporelles définies individuellement par trajectoire :
             - prop_ge2_events compare la fréquence de sélection de la
               population conditionnelle (N_events >= 2) ;
             - cond_ge2_traj_inter_* comparent les résumés inter-sauts à
               l'intérieur de cette sous-population conditionnelle. Ces résultats
               sont complémentaires et ne suffisent pas à eux seuls à établir
               l'égalité de la loi complète sur l'espace des trajectoires.
        """
        nk = traj.NKLandscape(n=3, k=2, seed=19)
        X0 = np.array([1.0, -1.0, 1.0])
        J0 = fb.unpack_J((0.85, -0.70, 0.65), n=3)
        J_anchor = fb.unpack_J((0.20, -0.15, 0.25), n=3)
        t_max = 2.5

        params = dict(
            t_max=t_max,
            mu=2.2,
            gamma=0.9,
            beta=1.5,
            eta=0.18,
            theta_A=0.35,
            fnk_fn=nk,
            J_anchor=J_anchor,
        )

        seeds = [11, 22, 33, 44]
        n_per_seed = 600
        n_total = len(seeds) * n_per_seed

        joint_og = np.empty((n_total, 7), dtype=np.float64)
        joint_ex = np.empty((n_total, 7), dtype=np.float64)

        t_first_og = np.empty(n_total, dtype=np.float64)
        t_first_ex = np.empty(n_total, dtype=np.float64)
        t_last_og = np.empty(n_total, dtype=np.float64)
        t_last_ex = np.empty(n_total, dtype=np.float64)
        t_span_og = np.empty(n_total, dtype=np.float64)
        t_span_ex = np.empty(n_total, dtype=np.float64)
        has_ge2_og = np.zeros(n_total, dtype=np.float64)
        has_ge2_ex = np.zeros(n_total, dtype=np.float64)

        cond_inter_mean_og: list[float] = []
        cond_inter_med_og: list[float] = []
        cond_inter_q75_og: list[float] = []

        cond_inter_mean_ex: list[float] = []
        cond_inter_med_ex: list[float] = []
        cond_inter_q75_ex: list[float] = []

        idx = 0
        for seed in seeds:
            rng_og = np.random.default_rng(seed)
            rng_ex = np.random.default_rng(seed + 10_000)

            for _ in range(n_per_seed):
                res_og = traj.simulate_ogata_thinning(
                    X0=X0, J0=J0, lam=0.0, rng=rng_og, **params
                )
                res_ex = traj.simulate_exact_piecewise_lambda0(
                    X0=X0, J0=J0, rng=rng_ex, **params
                )

                joint_og[idx, :3] = fb.pack_J(res_og.J_final)
                joint_og[idx, 3:6] = res_og.X_final
                joint_og[idx, 6] = float(res_og.n_accepted)

                joint_ex[idx, :3] = fb.pack_J(res_ex.J_final)
                joint_ex[idx, 3:6] = res_ex.X_final
                joint_ex[idx, 6] = float(res_ex.n_accepted)

                if res_og.events:
                    times_og = np.asarray([ev.time for ev in res_og.events], dtype=np.float64)
                    t_first_og[idx] = float(times_og[0])
                    t_last_og[idx] = float(times_og[-1])
                    t_span_og[idx] = float(times_og[-1] - times_og[0])
                    if times_og.size >= 2:
                        has_ge2_og[idx] = 1.0
                        dts_og = np.diff(times_og)
                        cond_inter_mean_og.append(float(np.mean(dts_og)))
                        cond_inter_med_og.append(float(np.median(dts_og)))
                        cond_inter_q75_og.append(float(np.quantile(dts_og, 0.75)))
                else:
                    t_first_og[idx] = t_max
                    t_last_og[idx] = 0.0
                    t_span_og[idx] = 0.0

                if res_ex.events:
                    times_ex = np.asarray([ev.time for ev in res_ex.events], dtype=np.float64)
                    t_first_ex[idx] = float(times_ex[0])
                    t_last_ex[idx] = float(times_ex[-1])
                    t_span_ex[idx] = float(times_ex[-1] - times_ex[0])
                    if times_ex.size >= 2:
                        has_ge2_ex[idx] = 1.0
                        dts_ex = np.diff(times_ex)
                        cond_inter_mean_ex.append(float(np.mean(dts_ex)))
                        cond_inter_med_ex.append(float(np.median(dts_ex)))
                        cond_inter_q75_ex.append(float(np.quantile(dts_ex, 0.75)))
                else:
                    t_first_ex[idx] = t_max
                    t_last_ex[idx] = 0.0
                    t_span_ex[idx] = 0.0

                idx += 1

        alpha_level = 0.01
        n_perm = 9999

        # (a) Lois marginales : permutation Monte-Carlo (9999) + q_0.95(W_1^*) < delta_eq
        marginal_specs = [
            ("J_12", joint_og[:, 0], joint_ex[:, 0], 0.035),
            ("J_13", joint_og[:, 1], joint_ex[:, 1], 0.035),
            ("J_23", joint_og[:, 2], joint_ex[:, 2], 0.035),
            ("n_events", joint_og[:, 6], joint_ex[:, 6], 0.250),
        ]
        for name, u_vec, v_vec, delta_eq in marginal_specs:
            w1_obs, p_val = _mc_permutation_test_1d(u_vec, v_vec, n_permutations=n_perm)
            ucb95 = _bootstrap_percentile_ucb95_1d(u_vec, v_vec, n_bootstrap=2000)
            self.assertGreater(
                p_val,
                alpha_level,
                msg=f"Marginal '{name}': rejet permutation MC (W1={w1_obs:.5f}, p={p_val:.5f})",
            )
            self.assertLess(
                ucb95,
                delta_eq,
                msg=(
                    f"Marginal '{name}': borne bootstrap percentile q_0.95={ucb95:.5f} "
                    f">= delta_eq={delta_eq} (W1={w1_obs:.5f})"
                ),
            )

        # (b) Loi jointe multivariée sur R^7 standardisé (sous-échantillon reproductible m=600)
        sub_rng = np.random.default_rng(2026)
        sub_idx_og = sub_rng.choice(n_total, size=600, replace=False)
        sub_idx_ex = sub_rng.choice(n_total, size=600, replace=False)

        pooled = np.vstack([joint_og, joint_ex])
        means = np.mean(pooled, axis=0)
        stds = np.maximum(np.std(pooled, axis=0), 1e-12)
        Z_og = (joint_og[sub_idx_og] - means) / stds
        Z_ex = (joint_ex[sub_idx_ex] - means) / stds

        sw1_obs, sw1_p = _mc_permutation_sliced_w1(
            Z_og, Z_ex, n_projections=32, n_permutations=n_perm
        )
        sw1_ucb95 = _bootstrap_percentile_ucb95_sliced_w1(
            Z_og, Z_ex, n_projections=32, n_bootstrap=1000
        )
        self.assertGreater(sw1_p, alpha_level, msg=f"Sliced-W1 rejeté : {sw1_obs:.5f} (p={sw1_p:.5f})")
        self.assertLess(
            sw1_ucb95,
            0.11,
            msg=f"Sliced-W1 : borne bootstrap percentile q_0.95={sw1_ucb95:.5f} >= 0.11",
        )

        ed_obs, ed_p = _mc_permutation_energy_blocked(
            Z_og, Z_ex, n_permutations=n_perm, block_size=64, perm_batch_size=250
        )
        ed_ucb95 = _bootstrap_percentile_ucb95_energy_blocked(
            Z_og, Z_ex, n_bootstrap=1000, block_size=128
        )
        self.assertGreater(ed_p, alpha_level, msg=f"EnergyDist rejetée : {ed_obs:.5f} (p={ed_p:.5f})")
        self.assertLess(
            ed_ucb95,
            0.12,
            msg=f"EnergyDist : borne bootstrap percentile q_0.95={ed_ucb95:.5f} >= 0.12",
        )

        cov_obs, cov_p = _mc_permutation_cov_fro(Z_og, Z_ex, n_permutations=n_perm)
        cov_ucb95 = _bootstrap_percentile_ucb95_cov_fro(Z_og, Z_ex, n_bootstrap=2000)
        self.assertGreater(cov_p, alpha_level, msg=f"Cov Fro rejetée : {cov_obs:.5f} (p={cov_p:.5f})")
        self.assertLess(
            cov_ucb95,
            0.45,
            msg=f"Cov Fro : borne bootstrap percentile q_0.95={cov_ucb95:.5f} >= 0.45",
        )

        # (c) Statistiques temporelles par trajectoire
        temporal_specs = [
            ("T_first_capped", t_first_og, t_first_ex, 0.035),
            ("T_last_capped", t_last_og, t_last_ex, 0.045),
            ("event_span", t_span_og, t_span_ex, 0.055),
            ("prop_ge2_events", has_ge2_og, has_ge2_ex, 0.040),
            (
                "cond_ge2_traj_inter_mean",
                np.asarray(cond_inter_mean_og, dtype=np.float64),
                np.asarray(cond_inter_mean_ex, dtype=np.float64),
                0.045,
            ),
            (
                "cond_ge2_traj_inter_median",
                np.asarray(cond_inter_med_og, dtype=np.float64),
                np.asarray(cond_inter_med_ex, dtype=np.float64),
                0.045,
            ),
            (
                "cond_ge2_traj_inter_q75",
                np.asarray(cond_inter_q75_og, dtype=np.float64),
                np.asarray(cond_inter_q75_ex, dtype=np.float64),
                0.045,
            ),
        ]
        for name, u_vec, v_vec, delta_eq in temporal_specs:
            self.assertGreater(u_vec.size, 100)
            self.assertGreater(v_vec.size, 100)
            w1_t, p_t = _mc_permutation_test_1d(u_vec, v_vec, n_permutations=n_perm)
            ucb95_t = _bootstrap_percentile_ucb95_1d(u_vec, v_vec, n_bootstrap=2000)
            self.assertGreater(
                p_t,
                alpha_level,
                msg=f"Temporal '{name}': rejet permutation MC (W1={w1_t:.5f}, p={p_t:.5f})",
            )
            self.assertLess(
                ucb95_t,
                delta_eq,
                msg=(
                    f"Temporal '{name}': borne bootstrap percentile q_0.95={ucb95_t:.5f} "
                    f">= delta_eq={delta_eq} (W1={w1_t:.5f})"
                ),
            )

    def test_08_zero_draw_channel_selection_boundary(self) -> None:
        """
        Vérifie avec un générateur contrôlé retournant exactement `u2 = 0.0` que
        `np.searchsorted(..., side="right")` sélectionne le premier canal de taux
        strictement positif (ici k = 2) et ne sélectionne jamais un canal de taux
        nul (k = 0 ou k = 1), tant dans `simulate_ogata_thinning` que dans
        `simulate_exact_piecewise_lambda0` (pour gamma = 0 et gamma > 0).
        """
        X0 = np.array([1.0, 1.0, 1.0])
        J0 = fb.unpack_J((0.0, 0.4, 0.4), n=3)
        mu = 1.5
        theta_A = 0.25

        gen_og = _ControlledZeroU2Generator(u1_sequence=[0.5, 1e-6], u2_value=0.0)
        res_og = traj.simulate_ogata_thinning(
            X0=X0,
            J0=J0,
            t_max=1.0,
            mu=mu,
            gamma=0.0,
            beta=0.0,
            lam=0.0,
            eta=0.0,
            theta_A=theta_A,
            rng=gen_og,  # type: ignore[arg-type]
        )
        self.assertEqual(res_og.n_accepted, 1)
        self.assertEqual(res_og.events[0].node, 2)

        gen_ex0 = _ControlledZeroU2Generator(u1_sequence=[0.5, 1e-6], u2_value=0.0)
        res_ex0 = traj.simulate_exact_piecewise_lambda0(
            X0=X0,
            J0=J0,
            t_max=1.0,
            mu=mu,
            gamma=0.0,
            beta=0.0,
            eta=0.0,
            theta_A=theta_A,
            rng=gen_ex0,  # type: ignore[arg-type]
        )
        self.assertEqual(res_ex0.n_accepted, 1)
        self.assertEqual(res_ex0.events[0].node, 2)

        gen_ex_pos = _ControlledZeroU2Generator(u1_sequence=[0.9, 1e-6], u2_value=0.0)
        res_ex_pos = traj.simulate_exact_piecewise_lambda0(
            X0=X0,
            J0=J0,
            t_max=1.0,
            mu=mu,
            gamma=0.2,
            beta=0.0,
            eta=0.0,
            theta_A=theta_A,
            rng=gen_ex_pos,  # type: ignore[arg-type]
        )
        self.assertEqual(res_ex_pos.n_accepted, 1)
        self.assertEqual(res_ex_pos.events[0].node, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
