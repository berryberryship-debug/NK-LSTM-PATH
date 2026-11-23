import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fiber_bundle as fb
from nk_landscape import NKLandscape
import trajectories as tr

LAND = NKLandscape(3, 1, seed=7)
J0 = fb.unpack_J((0.8, -0.6, 0.7))
X0 = np.array([1., -1., 1.])


def stats(sim, p, M, seed, T):
    rng = np.random.default_rng(seed)
    n, dj = [], []
    for _ in range(M):
        ev = sim(X0, J0, p, LAND, T, rng)
        n.append(len(ev))
        Jf = tr.fiber_at(ev, J0, p, T)
        dj.append(np.linalg.norm(Jf - J0))
    return np.array(n, float), np.array(dj)


def w1(a, b):
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


class TestLandscapeAndRates(unittest.TestCase):
    def test_nk_deterministic_and_range(self):
        L2 = NKLandscape(3, 1, seed=7)
        for s in [(1, 1, 1), (-1, 1, -1)]:
            self.assertEqual(LAND.F(s), L2.F(s)); self.assertTrue(0 <= LAND.F(s) <= 1)

    def test_deltaF_J_term_matches_direct(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            J = fb.unpack_J(rng.uniform(-1, 1, 3)); X = rng.choice([-1., 1.], 3); lam = 0.7
            F = lambda Y: LAND.F(Y) + lam * sum(J[i, j] * Y[i] * Y[j] for i in range(3) for j in range(i + 1, 3))
            for k in range(3):
                Xk = X.copy(); Xk[k] = -Xk[k]
                self.assertAlmostEqual(tr.delta_F(X, J, k, LAND, lam), F(Xk) - F(X), places=12)

    def test_rates_bounded_by_mu(self):
        rng = np.random.default_rng(1)
        p = tr.Params(lam=1.5, theta_A=0.2)
        for _ in range(500):
            J = fb.unpack_J(rng.uniform(-2, 2, 3)); X = rng.choice([-1., 1.], 3)
            w = tr.rates(X, J, p, LAND)
            self.assertTrue(np.all(w >= 0) and np.all(w <= p.mu + 1e-15))

    def test_lam0_decouples_from_J(self):
        X = np.array([1., 1., -1.])
        a = tr.delta_F(X, fb.unpack_J((1, 2, 3)), 0, LAND, 0.0)
        b = tr.delta_F(X, fb.unpack_J((-1, 0.1, 5)), 0, LAND, 0.0)
        self.assertEqual(a, b)


class TestSimulators(unittest.TestCase):
    def test_invariants(self):
        rng = np.random.default_rng(3); p = tr.Params(lam=0.8)
        ev = tr.simulate_thinning(X0, J0, p, LAND, 20.0, rng)
        Jprev = None
        for (t, k, X, J) in ev:
            np.testing.assert_allclose(J, J.T, atol=1e-12); self.assertTrue(np.all(np.diag(J) == 0))
            self.assertTrue(np.all(np.isin(X, (-1., 1.))))
        ts = [e[0] for e in ev]; self.assertEqual(ts, sorted(ts))

    def test_thinning_vs_piecewise_lam0(self):
        p = tr.Params(lam=0.0, theta_A=1.0, eta=0.4, gamma=0.5)  # theta traverse l'amplitude de A_k
        T, M = 8.0, 3000
        n1, d1 = stats(tr.simulate_thinning, p, M, 11, T)
        n2, d2 = stats(tr.simulate_piecewise, p, M, 12, T)
        n3, d3 = stats(tr.simulate_thinning, p, M, 13, T)   # base de bruit : meme methode, autre graine
        self.assertGreater(n1.mean(), 1.0)
        base_n, base_d = w1(n1, n3), w1(d1, d3)
        self.assertLessEqual(w1(n1, n2), 2.5 * base_n + 0.05)
        self.assertLessEqual(w1(d1, d2), 2.5 * base_d + 0.01)
        se = np.sqrt(n1.var() / M + n2.var() / M)
        self.assertLess(abs(n1.mean() - n2.mean()), 4 * se)

    def test_thinning_vs_euler_lam_nonzero(self):
        p = tr.Params(lam=0.8, theta_A=0.9, eta=0.3, gamma=0.5, beta=1.0)
        T, dt, M = 4.0, 0.01, 600
        rng = np.random.default_rng(21); counts = []
        for _ in range(M):
            X, J, n, t = X0.copy(), J0.copy(), 0, 0.0
            while t < T:
                J = J0 + (J - J0) * np.exp(-p.gamma * dt)
                w = tr.rates(X, J, p, LAND)
                u = rng.random()
                if u < w.sum() * dt:
                    k = min(int(np.searchsorted(np.cumsum(w), rng.uniform(0, w.sum()), side="right")), 2)
                    X, J = fb.mutate_postflip(X, J, k, p.eta); n += 1
                t += dt
            counts.append(n)
        counts = np.array(counts, float)
        nt, _ = stats(tr.simulate_thinning, p, 3000, 22, T)
        se = np.sqrt(counts.var() / M + nt.var() / 3000)
        self.assertLess(abs(counts.mean() - nt.mean()), 4 * se + 0.05 * nt.mean())  # biais Euler O(dt) toléré
        self.assertGreater(nt.mean(), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
