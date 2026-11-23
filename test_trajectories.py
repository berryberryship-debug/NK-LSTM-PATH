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
     (coefficients 1e-16 et 1e+16, coefficient quadratique 0 < c2/S < 1e-14 avec
     racine intérieure conservée, racines proches 0.5 +- 1e-7, racines proches des
     bornes, tangence, exclusion documentée des bornes {0, 1}).
  6. Conservation du risque intégré int_0^t Lambda(u) du = E à travers un changement
     de masque d'accessibilité :
       - fermeture d'un canal (3*mu -> 2*mu à tau_c = ln(2)/gamma),
       - ouverture retardée de deux canaux (0 -> 2*mu à tau_c = ln(2)/gamma).
  7. Validation croisée en loi par test de permutation exact à deux échantillons
     (alpha = 0.01) doublé d'une marge d'équivalence pratique explicite delta_eq
     entre `simulate_ogata_thinning` et `simulate_exact_piecewise_lambda0` à lambda = 0 :
       (a) lois marginales (J_12, J_13, J_23, n_événements),
       (b) loi jointe multivariée (Sliced Wasserstein-1, Energy Distance précalculée
           et écart de covariance sur R^7),
       (c) statistiques temporelles des événements (T_1, T_dernier, inter-sauts de
           longueurs n_U, n_V potentiellement distinctes).
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


def _two_sample_permutation_test_1d(
    u_samples: np.ndarray,
    v_samples: np.ndarray,
    n_permutations: int = 400,
    seed: int = 2026,
) -> tuple[float, float]:
    """
    Test de permutation exact à deux échantillons (tailles n_u, n_v quelconques)
    pour la statistique W_1. Retourne (w1_obs, p_value).
    """
    u = np.asarray(u_samples, dtype=np.float64).ravel()
    v = np.asarray(v_samples, dtype=np.float64).ravel()
    n_u = u.size
    w1_obs = traj.wasserstein_1d(u, v)

    pooled = np.concatenate([u, v])
    n_tot = pooled.size
    rng = np.random.default_rng(seed)

    ge_count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n_tot)
        w1_perm = traj.wasserstein_1d(pooled[perm[:n_u]], pooled[perm[n_u:]])
        if w1_perm >= w1_obs:
            ge_count += 1

    p_value = (1.0 + ge_count) / (1.0 + n_permutations)
    return w1_obs, p_value


def _two_sample_permutation_test_multivariate(
    U: np.ndarray,
    V: np.ndarray,
    metric_fn,
    n_permutations: int = 300,
    seed: int = 2026,
) -> tuple[float, float]:
    """
    Test de permutation exact à deux échantillons sur nuages multivariés U, V.
    Retourne (stat_obs, p_value).
    """
    n_u = U.shape[0]
    stat_obs = float(metric_fn(U, V))
    pooled = np.vstack([U, V])
    n_tot = pooled.shape[0]
    rng = np.random.default_rng(seed)

    ge_count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n_tot)
        stat_perm = float(metric_fn(pooled[perm[:n_u]], pooled[perm[n_u:]]))
        if stat_perm >= stat_obs:
            ge_count += 1

    p_value = (1.0 + ge_count) / (1.0 + n_permutations)
    return stat_obs, p_value


def _energy_distance_permutation_fast(
    U: np.ndarray,
    V: np.ndarray,
    n_permutations: int = 400,
    seed: int = 2026,
) -> tuple[float, float]:
    """
    Calcule D_E(U, V) exact par blocs et effectue un test de permutation rapide
    en précalculant une seule fois la matrice des distances euclidiennes sur
    l'échantillon réuni.
    """
    ed_obs = traj.energy_distance(U, V, block_size=256)
    n_u = U.shape[0]
    n_v = V.shape[0]
    pooled = np.vstack([U, V])
    n_tot = pooled.shape[0]

    # Matrice des distances euclidiennes calculée par blocs (empreinte O(n_tot^2))
    dist_mat = np.zeros((n_tot, n_tot), dtype=np.float64)
    block = 256
    for i in range(0, n_tot, block):
        ie = min(i + block, n_tot)
        diff = pooled[i:ie, None, :] - pooled[None, :, :]
        dist_mat[i:ie, :] = np.linalg.norm(diff, axis=-1)

    rng = np.random.default_rng(seed)
    ge_count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n_tot)
        iu = perm[:n_u]
        iv = perm[n_u:]
        e_xy = float(np.mean(dist_mat[np.ix_(iu, iv)]))
        e_xx = float(np.sum(dist_mat[np.ix_(iu, iu)]) / (n_u * (n_u - 1)))
        e_yy = float(np.sum(dist_mat[np.ix_(iv, iv)]) / (n_v * (n_v - 1)))
        ed_perm = float(np.sqrt(max(0.0, 2.0 * e_xy - e_xx - e_yy)))
        if ed_perm >= ed_obs:
            ge_count += 1

    p_value = (1.0 + ge_count) / (1.0 + n_permutations)
    return ed_obs, p_value


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
        - conservation des racines intérieures lorsque 0 < c2 / S < 1e-14 (pas de
          réduction artificielle de degré tant que c2 != 0),
        - deux racines très proches sans être identiques (z = 0.5 +- 1e-7),
        - racines proches des bornes (1e-11 et 1 - 1e-11),
        - exclusion explicite et documentée des bornes z in {0, 1}.
        """
        # 1. Échelles extrêmes (1e-16, 1, 1e16)
        for scale in (1e-16, 1.0, 1e16):
            roots = traj._stable_quadratic_real_roots(
                1.0 * scale, -1.0 * scale, 0.1875 * scale
            )
            np.testing.assert_allclose(sorted(roots), [0.25, 0.75], atol=1e-13)

        # 2. Coefficient quadratique très petit (c2 / S = 1e-15 < 1e-14) :
        #    c2 * z^2 - 1.0 * z + 0.4 = 0 possède deux racines réelles :
        #    z_2 ~ 0.4 in (0, 1) et z_1 ~ 1e15 hors de (0, 1). Aucune n'est perdue.
        c2_tiny = 1e-15
        roots_tiny = traj._stable_quadratic_real_roots(c2_tiny, -1.0, 0.4)
        self.assertEqual(len(roots_tiny), 2)
        z_small, z_large = sorted(roots_tiny)
        self.assertAlmostEqual(c2_tiny * (z_small ** 2) - z_small + 0.4, 0.0, places=15)
        self.assertGreater(z_large, 1e14)

        # 3. Deux racines très proches mais distinctes : z = 0.5 - 1e-7 et z = 0.5 + 1e-7
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
        Vérifie la conservation du risque intégré int_0^t Lambda(u) du = E à travers
        un changement de masque d'accessibilité :

        Cas 1 (fermeture d'un canal sur N = 3) :
          J^{(0)} = (0, 1, 0), J(0) = (1, 1, 0), theta_A = 0.5, tau_c = ln(2)/gamma.
          - Pour t <= tau_c : A_1(t) = e^{-2 gamma t} + 1 > 0.5, A_2(t) = e^{-2 gamma t} > 0.5,
            A_3(t) = 1 > 0.5  =>  3 nœuds accessibles  =>  Lambda_1 = 3 * mu.
          - Pour t > tau_c  : A_1(t) > 1 > 0.5, A_2(t) < 0.5, A_3(t) = 1 > 0.5
            =>  2 nœuds accessibles  =>  Lambda_2 = 2 * mu.
          Survie exacte :
            P(T_1 > t) = exp(-3 * mu * t)                           si t <= tau_c,
            P(T_1 > t) = exp(-3 * mu * tau_c - 2 * mu * (t - tau_c)) si t > tau_c.

        Cas 2 (ouverture retardée de deux canaux sur N = 3) :
          J^{(0)} = (1, 0, 0), J(0) = (0, 0, 0), theta_A = 0.25, tau_c = ln(2)/gamma.
          - Pour t <= tau_c : 0 nœud accessible  =>  Lambda_1 = 0.
          - Pour t > tau_c  : 2 nœuds accessibles =>  Lambda_2 = 2 * mu.
          Survie exacte : P(T_1 > t) = exp(-2 * mu * (t - tau_c)) pour t > tau_c.
        """
        mu = 1.0
        gamma = 1.0
        theta_A = 0.5
        tau_c = float(np.log(2.0) / gamma)

        # Cas 1 : fermeture d'un canal (3*mu -> 2*mu à tau_c)
        X0 = np.array([1.0, 1.0, 1.0])
        J0 = fb.unpack_J((1.0, 1.0, 0.0), n=3)
        J_anchor = fb.unpack_J((0.0, 1.0, 0.0), n=3)

        t_before = 0.40  # < tau_c ~ 0.69315
        t_after = 1.00   # > tau_c ~ 0.69315
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
                theta_A=theta_A,
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
                theta_A=theta_A,
                rng=rng_og,
            )
            t1_og[i] = res_og.events[0].time if res_og.events else np.inf

        # Erreur standard binomiale <= 0.5 / sqrt(12000) ~ 0.0046 ; tolérance 3.5 sigma = 0.016
        self.assertAlmostEqual(float(np.mean(t1_ex > t_before)), surv_exact_before, delta=0.016)
        self.assertAlmostEqual(float(np.mean(t1_ex > t_after)), surv_exact_after, delta=0.016)
        self.assertAlmostEqual(float(np.mean(t1_og > t_after)), surv_exact_after, delta=0.016)

        # Cas 2 : ouverture retardée de deux canaux (0 -> 2*mu à tau_c)
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

    def test_07_cross_validation_permutation_and_equivalence_margins(self) -> None:
        """
        Validation croisée en loi entre `simulate_ogata_thinning` et
        `simulate_exact_piecewise_lambda0` à lambda = 0 combinant :
          1. Un test de permutation exact à deux échantillons (alpha = 0.01,
             i.e. p_value > 0.01) sans hypothèse d'égalité de taille d'échantillon.
          2. Une marge d'équivalence pratique explicite delta_eq sur chaque observable :
               - marginales W_1 : delta_eq = 0.035 sur J_ij, 0.15 sur n_events,
               - jointe multivariée (R^7 standardisé) : Sliced W_1 <= 0.08,
                 Energy Distance <= 0.08, Frobenius(Cov_Og - Cov_Ex) <= 0.25,
               - temporelles W_1 (tailles n_U != n_V gérées nativement) :
                 delta_eq = 0.04 sur T_1 et T_dernier, 0.03 sur inter-sauts.
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

        alpha_level = 0.01

        # (a) Lois marginales : test de permutation + marge d'équivalence explicite
        marginal_specs = [
            ("J_12", joint_og[:, 0], joint_ex[:, 0], 0.035),
            ("J_13", joint_og[:, 1], joint_ex[:, 1], 0.035),
            ("J_23", joint_og[:, 2], joint_ex[:, 2], 0.035),
            ("n_events", joint_og[:, 6], joint_ex[:, 6], 0.150),
        ]
        for name, u_vec, v_vec, delta_eq in marginal_specs:
            w1_obs, p_val = _two_sample_permutation_test_1d(u_vec, v_vec, n_permutations=400)
            self.assertGreater(
                p_val,
                alpha_level,
                msg=f"Marginal '{name}': rejet par permutation (W1={w1_obs:.5f}, p={p_val:.4f})",
            )
            self.assertLessEqual(
                w1_obs,
                delta_eq,
                msg=f"Marginal '{name}': W1={w1_obs:.5f} > marge d'équivalence {delta_eq}",
            )

        # (b) Loi jointe multivariée sur R^7 standardisé (sous-échantillon reproductible m=800)
        sub_rng = np.random.default_rng(2026)
        sub_idx_og = sub_rng.choice(n_total, size=800, replace=False)
        sub_idx_ex = sub_rng.choice(n_total, size=800, replace=False)

        pooled = np.vstack([joint_og, joint_ex])
        means = np.mean(pooled, axis=0)
        stds = np.maximum(np.std(pooled, axis=0), 1e-12)
        Z_og = (joint_og[sub_idx_og] - means) / stds
        Z_ex = (joint_ex[sub_idx_ex] - means) / stds

        sw1_obs, sw1_p = _two_sample_permutation_test_multivariate(
            Z_og,
            Z_ex,
            lambda A, B: traj.sliced_wasserstein_1(A, B, n_projections=64, seed=42),
            n_permutations=300,
        )
        self.assertGreater(sw1_p, alpha_level, msg=f"Sliced-W1 rejeté : {sw1_obs:.5f} (p={sw1_p:.4f})")
        self.assertLessEqual(sw1_obs, 0.08, msg=f"Sliced-W1={sw1_obs:.5f} > delta_eq=0.08")

        ed_obs, ed_p = _energy_distance_permutation_fast(Z_og, Z_ex, n_permutations=400)
        self.assertGreater(ed_p, alpha_level, msg=f"EnergyDist rejetée : {ed_obs:.5f} (p={ed_p:.4f})")
        self.assertLessEqual(ed_obs, 0.08, msg=f"EnergyDist={ed_obs:.5f} > delta_eq=0.08")

        cov_obs, cov_p = _two_sample_permutation_test_multivariate(
            Z_og,
            Z_ex,
            lambda A, B: float(
                np.linalg.norm(np.cov(A, rowvar=False) - np.cov(B, rowvar=False), ord="fro")
            ),
            n_permutations=300,
        )
        self.assertGreater(cov_p, alpha_level, msg=f"Cov Fro rejetée : {cov_obs:.5f} (p={cov_p:.4f})")
        self.assertLessEqual(cov_obs, 0.25, msg=f"Cov Fro={cov_obs:.5f} > delta_eq=0.25")

        # (c) Statistiques temporelles (tailles n_U != n_V gérées nativement sans troncature)
        arr_inter_og = np.asarray(inter_og, dtype=np.float64)
        arr_inter_ex = np.asarray(inter_ex, dtype=np.float64)
        self.assertGreater(arr_inter_og.size, 100)
        self.assertGreater(arr_inter_ex.size, 100)

        temporal_specs = [
            ("T_first_capped", t_first_og, t_first_ex, 0.040),
            ("T_last_capped", t_last_og, t_last_ex, 0.040),
            ("inter_jump_dt", arr_inter_og, arr_inter_ex, 0.030),
        ]
        for name, u_vec, v_vec, delta_eq in temporal_specs:
            w1_t, p_t = _two_sample_permutation_test_1d(u_vec, v_vec, n_permutations=400)
            self.assertGreater(
                p_t,
                alpha_level,
                msg=f"Temporal '{name}': rejet par permutation (W1={w1_t:.5f}, p={p_t:.4f})",
            )
            self.assertLessEqual(
                w1_t,
                delta_eq,
                msg=f"Temporal '{name}': W1={w1_t:.5f} > marge d'équivalence {delta_eq}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
