"""
tests/test_trajectories.py
Projet : NK-LSTM-PATH

Tests unitaires et statistiques de `src/trajectories.py` :
  1. Borne supérieure uniforme Lambda(X, J) <= N * mu et découplage lambda = 0.
  2. Dissipation analytique exacte (symétrie, diagonale nulle, limites gamma -> 0 et t -> inf).
  3. Transport post-flip lors des mutations (flip avant Psi, arêtes non incidentes
     uniquement dissipées).
  4. Distribution des événements (horloge exponentielle Exp(N*mu), uniformité des canaux,
     silence sous seuil inaccessible, biais sélectif Metropolis pour lambda != 0).
  5. Résolution analytique des franchissements de seuil en z = exp(-gamma * tau) in (0, 1] :
     0 racine, 1 racine, 2 racines distinctes, racine double (tangence), racines hors (0, 1),
     égalité exacte au seuil à t = 0.
  6. Validation croisée en loi (distance de Wasserstein-1 W_1) entre l'amincissement
     d'Ogata et le solveur analytique exact par morceaux à lambda = 0 sur plusieurs graines.
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


class TestTrajectories(unittest.TestCase):
    """Validation du simulateur joint et du contrôle analytique par morceaux."""

    def test_01_rate_upper_bound_and_decoupling(self) -> None:
        """Vérifie 0 <= w_k <= mu, Sum w_k <= N*mu et d(Delta F_k)/dJ == 0 pour lambda = 0."""
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

                # Découplage sélectif à lambda = 0
                raw2 = rng.normal(0.0, 1.5, size=(n, n))
                J2 = 0.5 * (raw2 + raw2.T)
                np.fill_diagonal(J2, 0.0)
                for node in range(n):
                    df_1 = traj.delta_fitness_k(X, J, k=node, lam=0.0, fnk_fn=nk)
                    df_2 = traj.delta_fitness_k(X, J2, k=node, lam=0.0, fnk_fn=nk)
                    self.assertAlmostEqual(df_1, df_2, places=15)

                # Cohérence de Delta F_k avec extended_fitness pour lambda != 0
                lam = 1.7
                f_before = traj.extended_fitness(X, J, lam=lam, fnk_fn=nk)
                for node in range(n):
                    X_flip = X.copy()
                    X_flip[node] = -X_flip[node]
                    f_after = traj.extended_fitness(X_flip, J, lam=lam, fnk_fn=nk)
                    df_analytic = traj.delta_fitness_k(X, J, k=node, lam=lam, fnk_fn=nk)
                    self.assertAlmostEqual(df_analytic, f_after - f_before, places=13)

                # Borne supérieure uniforme Lambda <= N * mu
                for theta_A in (-1.0, 0.2, 2.0):
                    rates = traj.transition_rates(
                        X, J, mu=mu, beta=beta, lam=lam, theta_A=theta_A, fnk_fn=nk
                    )
                    self.assertTrue(np.all(rates >= 0.0))
                    self.assertTrue(np.all(rates <= mu + 1e-15))
                    self.assertLessEqual(float(np.sum(rates)), n * mu + 1e-14)

    def test_02_analytical_dissipation(self) -> None:
        """Vérifie J(t) = J^(0) + (J(t_0) - J^(0)) e^{-gamma dt}, symétrie et semi-groupe."""
        J_anchor = fb.unpack_J((0.30, -0.45, 0.20), n=3)
        J_start = fb.unpack_J((-0.80, 0.90, -0.50), n=3)
        gamma = 1.35

        # Semi-groupe : dissipation de dt1 puis dt2 == dissipation directe de dt1 + dt2
        dt1, dt2 = 0.4, 0.7
        J_mid = traj.dissipate_J(J_start, J_anchor, gamma=gamma, dt=dt1)
        J_seq = traj.dissipate_J(J_mid, J_anchor, gamma=gamma, dt=dt2)
        J_dir = traj.dissipate_J(J_start, J_anchor, gamma=gamma, dt=dt1 + dt2)

        np.testing.assert_allclose(J_seq, J_dir, atol=1e-15)
        np.testing.assert_allclose(J_dir, J_dir.T, atol=1e-15)
        np.testing.assert_allclose(np.diag(J_dir), 0.0, atol=1e-15)

        # Limite gamma = 0 -> invariance
        J_nodiss = traj.dissipate_J(J_start, J_anchor, gamma=0.0, dt=5.0)
        np.testing.assert_allclose(J_nodiss, J_start, atol=1e-15)

        # Relaxation asymptotique t -> infini
        J_asymp = traj.dissipate_J(J_start, J_anchor, gamma=gamma, dt=40.0)
        np.testing.assert_allclose(J_asymp, J_anchor, atol=1e-15)

    def test_03_postflip_transport_along_trajectory(self) -> None:
        """
        Vérifie qu'à chaque événement enregistré dans une trajectoire :
        - le spin du nœud muté bascule,
        - les arêtes non incidentes n'ont subi que la dissipation pure,
        - les arêtes incidentes correspondent exactement à `mutate_postflip`.
        """
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
            theta_A=-1.0,  # toujours actif
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

            # L'arête opposée au nœud muté ne subit aucun saut de transport
            other_nodes = [i for i in range(3) if i != ev.node]
            u, v = other_nodes[0], other_nodes[1]
            self.assertAlmostEqual(ev.J[u, v], J_pre_jump[u, v], places=14)

            prev_t = ev.time
            prev_X = ev.X.copy()
            prev_J = ev.J.copy()

    def test_04_event_distributions_and_barriers(self) -> None:
        """
        Vérifie les lois statistiques élémentaires :
        - Silence total si theta_A > max_k A_k(J(t)) sur tout [0, t_max].
        - Régime neutre (beta = 0, a_k = 1) : nombre d'événements ~ Poisson(N * mu * t_max)
          et répartition uniforme sur les N nœuds.
        - Biais sélectif (lambda != 0, beta grand) : favorise la configuration qui maximise F.
        """
        X0 = np.array([1.0, 1.0, 1.0])
        J0 = fb.unpack_J((0.30, -0.20, 0.40), n=3)

        # 1. Silence total sous seuil élevé
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
        np.testing.assert_array_equal(res_silent.X_final, X0)
        np.testing.assert_allclose(
            res_silent.J_final, traj.dissipate_J(J0, J0, gamma=1.0, dt=5.0), atol=1e-15
        )

        # 2. Régime neutre actif : taux total = N * mu = 3 * 2.0 = 6.0 sur t_max = 2.0 -> moyenne = 12
        rng = np.random.default_rng(123)
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
                eta=0.0,  # fibre figée => a_k reste = 1
                theta_A=-1.0,
                rng=rng,
            )
            counts.append(res.n_accepted)
            for ev in res.events:
                node_hist[ev.node] += 1

        mean_count = float(np.mean(counts))
        self.assertAlmostEqual(mean_count, 12.0, delta=0.35)
        freqs = node_hist / float(np.sum(node_hist))
        np.testing.assert_allclose(freqs, np.full(3, 1.0 / 3.0), atol=0.02)

        # 3. Biais sélectif ferromagnétique (J_ij = +1, lambda = 2, beta = 3, eta = 0)
        J_ferro = fb.unpack_J((1.0, 1.0, 1.0), n=3)
        rng_sel = np.random.default_rng(999)
        aligned_count = 0
        n_trials = 400
        for _ in range(n_trials):
            res_sel = traj.simulate_ogata_thinning(
                X0=np.array([1.0, -1.0, 1.0]),
                J0=J_ferro,
                t_max=3.0,
                mu=3.0,
                gamma=0.0,
                beta=3.0,
                lam=2.0,
                eta=0.0,
                theta_A=0.5,
                rng=rng_sel,
            )
            if abs(float(np.sum(res_sel.X_final))) == 3.0:
                aligned_count += 1
        self.assertGreater(aligned_count / n_trials, 0.90)

    def test_05_threshold_root_solver_edge_cases(self) -> None:
        """
        Vérifie `solve_threshold_crossings_z` sur tous les régimes algébriques :
        - 0 racine dans (0, 1),
        - 1 racine simple dans (0, 1),
        - 2 racines distinctes dans (0, 1) (traversée non monotone de 0),
        - racine double (tangence sans changement de signe),
        - racines hors de (0, 1],
        - égalité exacte au seuil à t = 0 (z = 1).
        """
        # N = 2 : une seule arête J_{01} = j(z) = j0 + d * z, donc A_0(z) = A_1(z) = (j0 + d*z)^2.
        # Posons j0 = 1.0, d = -2.0 => j(z) = 1 - 2z => A_k(z) = (1 - 2z)^2.
        J_anchor = fb.unpack_J((1.0,), n=2)
        J_curr = fb.unpack_J((-1.0,), n=2)

        # 1. Deux racines distinctes : (1 - 2z)^2 = 0.25 => z in {0.25, 0.75}
        roots_2 = traj.solve_threshold_crossings_z(J_curr, J_anchor, theta_A=0.25)
        np.testing.assert_allclose(roots_2, np.array([0.75, 0.25]), atol=1e-12)

        # 2. Tangence / racine double : (1 - 2z)^2 = 0.0 => z = 0.5
        roots_double = traj.solve_threshold_crossings_z(J_curr, J_anchor, theta_A=0.0)
        np.testing.assert_allclose(roots_double, np.array([0.5]), atol=1e-12)

        # 3. Aucune racine réelle : (1 - 2z)^2 = -0.1
        roots_none = traj.solve_threshold_crossings_z(J_curr, J_anchor, theta_A=-0.1)
        self.assertEqual(roots_none.size, 0)

        # 4. Égalité exacte au seuil en z = 1 : (1 - 2z)^2 = 1.0 => z = 1 (exclu de (0,1)) et z = 0 (exclu)
        roots_boundary = traj.solve_threshold_crossings_z(J_curr, J_anchor, theta_A=1.0)
        self.assertEqual(roots_boundary.size, 0)

        # 5. Une seule racine dans (0, 1) : j0 = 0.2, d = 0.8 => j(z) = 0.2 + 0.8z
        # A_k(z) = 0.25 => z = (0.5 - 0.2) / 0.8 = 0.375 (l'autre racine est négative)
        roots_1 = traj.solve_threshold_crossings_z(
            fb.unpack_J((1.0,), n=2), fb.unpack_J((0.2,), n=2), theta_A=0.25
        )
        np.testing.assert_allclose(roots_1, np.array([0.375]), atol=1e-12)

    def test_06_piecewise_exact_vs_ogata_thinning_lambda0(self) -> None:
        """
        Validation croisée en loi (distance de Wasserstein-1 W_1) entre
        `simulate_ogata_thinning` et `simulate_exact_piecewise_lambda0` à lambda = 0,
        incluant des franchissements dynamiques de seuil theta_A pendant la relaxation,
        sur plusieurs graines aléatoires.
        """
        nk = traj.NKLandscape(n=3, k=2, seed=19)
        X0 = np.array([1.0, -1.0, 1.0])
        J0 = fb.unpack_J((0.85, -0.70, 0.65), n=3)
        J_anchor = fb.unpack_J((0.20, -0.15, 0.25), n=3)

        # Au départ : A_0 = 0.85^2 + (-0.70)^2 = 1.2125 > theta_A = 0.35
        # À l'équilibre J_anchor : A_0 = 0.20^2 + (-0.15)^2 = 0.0625 < theta_A = 0.35
        # => Franchissements multiples de seuil garantis au cours des trajectoires.
        params = dict(
            t_max=2.5,
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

        ogata_j12, ogata_j13, ogata_j23, ogata_counts = [], [], [], []
        exact_j12, exact_j13, exact_j23, exact_counts = [], [], [], []

        for seed in seeds:
            rng_og = np.random.default_rng(seed)
            rng_ex = np.random.default_rng(seed + 10_000)

            for _ in range(n_per_seed):
                res_og = traj.simulate_ogata_thinning(X0=X0, J0=J0, lam=0.0, rng=rng_og, **params)
                res_ex = traj.simulate_exact_piecewise_lambda0(X0=X0, J0=J0, rng=rng_ex, **params)

                j_og = fb.pack_J(res_og.J_final)
                j_ex = fb.pack_J(res_ex.J_final)

                ogata_j12.append(j_og[0])
                ogata_j13.append(j_og[1])
                ogata_j23.append(j_og[2])
                ogata_counts.append(float(res_og.n_accepted))

                exact_j12.append(j_ex[0])
                exact_j13.append(j_ex[1])
                exact_j23.append(j_ex[2])
                exact_counts.append(float(res_ex.n_accepted))

        w1_j12 = traj.wasserstein_1d(ogata_j12, exact_j12)
        w1_j13 = traj.wasserstein_1d(ogata_j13, exact_j13)
        w1_j23 = traj.wasserstein_1d(ogata_j23, exact_j23)
        w1_cnt = traj.wasserstein_1d(ogata_counts, exact_counts)

        # Sur N_total = 2400 trajectoires, l'écart W_1 empirique entre deux lois identiques
        # est borné par l'erreur Monte-Carlo O(1 / sqrt(2400)) ~ 0.02
        self.assertLess(w1_j12, 0.025)
        self.assertLess(w1_j13, 0.025)
        self.assertLess(w1_j23, 0.025)
        self.assertLess(w1_cnt, 0.15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
