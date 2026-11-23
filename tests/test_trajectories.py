"""
tests/test_trajectories.py
Projet : NK-LSTM-PATH

Suite de validation complète de `src/trajectories.py` :
  1. Borne supérieure uniforme Lambda(X, J) <= N * mu et invariance de Delta F_k
     entre fibres distinctes à lambda = 0.
  2. Dissipation analytique exacte (propriété de semi-groupe, symétrie, limites).
  3. Transport post-flip lors des mutations (flip avant Psi, dissipation pure des
     arêtes non incidentes).
  4. Lois élémentaires et relation exacte de bilan détaillé de Metropolis :
     w(X -> X^{(k)}) / w(X^{(k)} -> X) = exp(beta * Delta F_k).
  5. Stabilité numérique du solveur quadratique à tolérance relative (coefficients
     très petits 1e-16 et très grands 1e+16, racines proches 0.5 +- 1e-7, racines
     près de 0 et 1, tangence, exclusion documentée des bornes {0, 1}).
  6. Conservation du risque intégré int_0^t Lambda(u) du = E à travers un changement
     de masque d'accessibilité (fermeture et ouverture de canaux).
  7. Validation croisée en loi auto-calibrée (enveloppes Monte-Carlo internes avec
     IC à 99 %) entre Ogata et le solveur analytique par morceaux à lambda = 0 :
       (a) distributions marginales (J_12, J_13, J_23, n_événements),
       (b) distribution jointe multivariée (Sliced Wasserstein-1, Energy Distance
           et matrice de covariance sur R^7),
       (c) statistiques temporelles des événements (T_1, T_dernier, inter-sauts).
"""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import fiber_bundle as fb  # noqa: E402
import trajectories as traj  # noqa: E402


def _self_calibrated_1d_bound(
    pool_a: np.ndarray,
    pool_b: np.ndarray,
    n_bootstrap: int = 250,
    seed: int = 2026,
) -> tuple[float, float]:
    """
    Calibre l'enveloppe nulle (quantile 99 % + 2*std) de W_1 sous H0 en comparant
    des demi-échantillons indépendants issus d'un même simulateur, corrigée par le
    facteur d'échelle 1/sqrt(2) correspondant à la taille complète m.
    """
    rng = np.random.default_rng(seed)
    m = pool_a.size
    half = m // 2
    null_vals = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        pa = rng.permutation(m)
        pb = rng.permutation(m)
        w_a = traj.wasserstein_1d(pool_a[pa[:half]], pool_a[pa[half : 2 * half]])
        w_b = traj.wasserstein_1d(pool_b[pb[:half]], pool_b[pb[half : 2 * half]])
        null_vals[b] = 0.5 * (w_a + w_b) / np.sqrt(2.0)
    q99 = float(np.quantile(null_vals, 0.99))
    std = float(np.std(null_vals))
    return q99 + 2.0 * std, float(np.mean(null_vals))


def _self_calibrated_multivariate_bound(
    U: np.ndarray,
    V: np.ndarray,
    metric_fn,
    n_bootstrap: int = 120,
    seed: int = 2026,
) -> tuple[float, float]:
    """Calibre l'enveloppe nulle (quantile 99 % + 2*std) d'une métrique multivariée."""
    rng = np.random.default_rng(seed)
    m = U.shape[0]
    half = m // 2
    null_vals = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        pa = rng.permutation(m)
        pb = rng.permutation(m)
        d_a = metric_fn(U[pa[:half]], U[pa[half : 2 * half]])
        d_b = metric_fn(V[pb[:half]], V[pb[half : 2 * half]])
        null_vals[b] = 0.5 * (d_a + d_b) / np.sqrt(2.0)
    q99 = float(np.quantile(null_vals, 0.99))
    std = float(np.std(null_vals))
    return q99 + 2.0 * std, float(np.mean(null_vals))


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

        # 1. Bilan détaillé exact sur 25 couples (X, J) aléatoires
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

        # 2. Silence total sous seuil élevé
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

        # 3. Loi de Poisson(N*mu*t_max) et uniformité des canaux (beta=0, eta=0)
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

    def test_05_scale_relative_quadratic_solver_and_close_roots(self) -> None:
        """
        Vérifie `_stable_quadratic_real_roots` et `classify_threshold_crossings_z` :
        - coefficients très petits (échelle 1e-16) et très grands (échelle 1e+16),
        - coefficients fortement déséquilibrés (c2 = 1e-10, c1 = 1, c0 = -0.5),
        - deux racines très proches sans être identiques (z = 0.5 +- 1e-7),
        - racines proches des bornes (1e-11 et 1 - 1e-11),
        - exclusion volontaire et documentée des bornes z in {0, 1}.
        """
        # 1. Coefficients très petits (1e-16) : (z - 0.25)(z - 0.75) * 1e-16 = 0
        for scale in (1e-16, 1.0, 1e16):
            roots = traj._stable_quadratic_real_roots(
                1.0 * scale, -1.0 * scale, 0.1875 * scale
            )
            np.testing.assert_allclose(sorted(roots), [0.25, 0.75], atol=1e-13)

        # 2. Coefficients fortement déséquilibrés
        roots_unbal = traj._stable_quadratic_real_roots(1e-10, 1.0, -0.5)
        z_pos = max(roots_unbal)
        self.assertAlmostEqual(1e-10 * (z_pos ** 2) + z_pos - 0.5, 0.0, places=15)

        # 3. Deux racines très proches mais distinctes : z = 0.5 - 1e-7 et z = 0.5 + 1e-7
        # Pour N=2, j0 = 1, d = -2 => A(z) = (1 - 2z)^2.
        # À z = 0.5 +- delta, A(z) = 4 * delta^2. Pour delta = 1e-7, theta_A = 4e-14.
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

        # 4. Racines très proches des bornes (1 - 1e-11 et 1e-11)
        theta_near_edge = (1.0 - 2e-11) ** 2
        info_near = traj.classify_threshold_crossings_z(
            J_curr, J_anchor, theta_A=theta_near_edge
        )
        np.testing.assert_allclose(
            info_near.interior_z, np.array([1.0 - 1e-11, 1e-11]), atol=1e-14
        )
        self.assertEqual(info_near.boundary_z.size, 0)

        # 5. Exclusion explicite des bornes z in {0, 1}
        info_bound = traj.classify_threshold_crossings_z(J_curr, J_anchor, theta_A=1.0)
        self.assertEqual(info_bound.interior_z.size, 0)
        np.testing.assert_allclose(info_bound.boundary_z, np.array([1.0, 0.0]), atol=1e-15)

    def test_06_integrated_hazard_conservation_across_threshold(self) -> None:
        """
        Vérifie la conservation du risque intégré int_0^t Lambda(u) du = E lorsqu'un
        changement de masque modifie Lambda(u) en cours d'intervalle sans réinitialiser
        l'horloge exponentielle (fermeture et ouverture de canaux).
        """
        mu = 1.5
        gamma = 1.0
        theta_A = 0.5
        tau_cross = float(np.log(2.0) / gamma)

        # Cas 1 : fermeture d'un canal (Lambda_1 = 2*mu -> Lambda_2 = mu à tau_cross)
        X0 = np.array([1.0, 1.0, 1.0])
        J0 = fb.unpack_J((1.0, 1.0, 0.0), n=3)
        J_anchor = fb.unpack_J((0.0, 1.0, 0.0), n=3)

        t_before = 0.40
        t_after = 1.20
        surv_exact_before = float(np.exp(-2.0 * mu * t_before))
        surv_exact_after = float(
            np.exp(-2.0 * mu * tau_cross - mu * (t_after - tau_cross))
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
                t_max=2.5,
                mu=mu,
                gamma=gamma,
                beta=0.0,
                eta=0.0,
                theta_A=theta_A,
                rng=rng_ex,
            )
            t1_ex[i] = res_ex.events[0].time if res_ex.events else np.inf

            res_og = traj.simulate_ogata_thinning(
                X0=X0,
                J0=J0,
                J_anchor=J_anchor,
                t_max=2.5,
                mu=mu,
                gamma=gamma,
                beta=0.0,
                lam=0.0,
                eta=0.0,
                theta_A=theta_A,
                rng=rng_og,
            )
            t1_og[i] = res_og.events[0].time if res_og.events else np.inf

        # Erreur standard binomiale <= 0.5 / sqrt(12000) ~ 0.0046 ; tolérance 3.5 sigma = 0.016
        self.assertAlmostEqual(float(np.mean(t1_ex > t_before)), surv_exact_before, delta=0.016)
        self.assertAlmostEqual(float(np.mean(t1_ex > t_after)), surv_exact_after, delta=0.016)
        self.assertAlmostEqual(float(np.mean(t1_og > t_after)), surv_exact_after, delta=0.016)

        # Cas 2 : ouverture retardée de deux canaux (Lambda_1 = 0 -> Lambda_2 = 2*mu à tau_cross)
        J0_open = fb.unpack_J((0.0, 0.0, 0.0), n=3)
        J_anchor_open = fb.unpack_J((1.0, 0.0, 0.0), n=3)
        t_obs = 1.30
        surv_open_exact = float(np.exp(-2.0 * mu * (t_obs - tau_cross)))
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

    def test_07_cross_validation_marginal_joint_and_temporal_self_calibrated(self) -> None:
        """
        Validation croisée complète en loi (auto-calibrée par référence Monte-Carlo
        interne avec IC à 99 %) entre `simulate_ogata_thinning` et
        `simulate_exact_piecewise_lambda0` à lambda = 0, en distinguant :
          (a) les lois marginales (J_12, J_13, J_23, n_événements),
          (b) la loi jointe multivariée (J_12, J_13, J_23, s_1, s_2, s_3, n_événements)
              via Sliced Wasserstein-1, Energy Distance et matrice de covariance,
          (c) les statistiques temporelles (instant du 1er saut T_1, instant du
              dernier saut T_dernier, durée moyenne inter-sauts).
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
        n_per_seed = 900
        n_total = len(seeds) * n_per_seed

        # Vecteur joint d'état final : (J_12, J_13, J_23, s_1, s_2, s_3, n_accepted)
        joint_og = np.empty((n_total, 7), dtype=np.float64)
        joint_ex = np.empty((n_total, 7), dtype=np.float64)

        t_first_og = np.empty(n_total, dtype=np.float64)
        t_first_ex = np.empty(n_total, dtype=np.float64)
        t_last_og = np.empty(n_total, dtype=np.float64)
        t_last_ex = np.empty(n_total, dtype=np.float64)
        inter_og: list[float] = []
        inter_ex: list[float] = []

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
                    times_og = [ev.time for ev in res_og.events]
                    t_first_og[idx] = times_og[0]
                    t_last_og[idx] = times_og[-1]
                    if len(times_og) >= 2:
                        inter_og.extend(np.diff(times_og).tolist())
                else:
                    t_first_og[idx] = t_max
                    t_last_og[idx] = 0.0

                if res_ex.events:
                    times_ex = [ev.time for ev in res_ex.events]
                    t_first_ex[idx] = times_ex[0]
                    t_last_ex[idx] = times_ex[-1]
                    if len(times_ex) >= 2:
                        inter_ex.extend(np.diff(times_ex).tolist())
                else:
                    t_first_ex[idx] = t_max
                    t_last_ex[idx] = 0.0

                idx += 1

        # (a) Comparaison des distributions marginales (auto-calibrées)
        marginal_specs = [
            ("J_12", joint_og[:, 0], joint_ex[:, 0]),
            ("J_13", joint_og[:, 1], joint_ex[:, 1]),
            ("J_23", joint_og[:, 2], joint_ex[:, 2]),
            ("n_events", joint_og[:, 6], joint_ex[:, 6]),
        ]
        for name, u_vec, v_vec in marginal_specs:
            w1_cross = traj.wasserstein_1d(u_vec, v_vec)
            bound, null_mean = _self_calibrated_1d_bound(u_vec, v_vec)
            self.assertLessEqual(
                w1_cross,
                bound,
                msg=f"Marginal '{name}': W1={w1_cross:.5f} > borne MC={bound:.5f} (null={null_mean:.5f})",
            )

        # (b) Comparaison de la distribution jointe multivariée (R^7 standardisé)
        pooled = np.vstack([joint_og, joint_ex])
        means = np.mean(pooled, axis=0)
        stds = np.maximum(np.std(pooled, axis=0), 1e-12)
        Z_og = (joint_og - means) / stds
        Z_ex = (joint_ex - means) / stds

        sw1_cross = traj.sliced_wasserstein_1(Z_og, Z_ex, n_projections=64, seed=42)
        sw1_bound, sw1_null = _self_calibrated_multivariate_bound(
            Z_og, Z_ex, lambda A, B: traj.sliced_wasserstein_1(A, B, n_projections=64, seed=42)
        )
        self.assertLessEqual(
            sw1_cross,
            sw1_bound,
            msg=f"Joint Sliced-W1={sw1_cross:.5f} > borne MC={sw1_bound:.5f} (null={sw1_null:.5f})",
        )

        ed_cross = traj.energy_distance(Z_og, Z_ex)
        ed_bound, ed_null = _self_calibrated_multivariate_bound(
            Z_og, Z_ex, traj.energy_distance, n_bootstrap=60
        )
        self.assertLessEqual(
            ed_cross,
            ed_bound,
            msg=f"Joint EnergyDist={ed_cross:.5f} > borne MC={ed_bound:.5f} (null={ed_null:.5f})",
        )

        # Préservation des dépendances croisées (matrice de corrélation / covariance sur Z)
        cov_og = np.cov(Z_og, rowvar=False)
        cov_ex = np.cov(Z_ex, rowvar=False)
        cov_diff = float(np.linalg.norm(cov_og - cov_ex, ord="fro"))
        cov_bound, cov_null = _self_calibrated_multivariate_bound(
            Z_og,
            Z_ex,
            lambda A, B: float(
                np.linalg.norm(np.cov(A, rowvar=False) - np.cov(B, rowvar=False), ord="fro")
            ),
        )
        self.assertLessEqual(
            cov_diff,
            cov_bound,
            msg=f"Joint Cov Fro={cov_diff:.5f} > borne MC={cov_bound:.5f} (null={cov_null:.5f})",
        )

        # (c) Comparaison des statistiques temporelles des événements
        temporal_specs = [
            ("T_first_capped", t_first_og, t_first_ex),
            ("T_last_capped", t_last_og, t_last_ex),
            ("inter_jump_dt", np.asarray(inter_og), np.asarray(inter_ex)),
        ]
        for name, u_vec, v_vec in temporal_specs:
            w1_t = traj.wasserstein_1d(u_vec, v_vec)
            bound_t, null_t = _self_calibrated_1d_bound(u_vec, v_vec)
            self.assertLessEqual(
                w1_t,
                bound_t,
                msg=f"Temporal '{name}': W1={w1_t:.5f} > borne MC={bound_t:.5f} (null={null_t:.5f})",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
