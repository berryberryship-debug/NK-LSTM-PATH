import itertools
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import fiber_bundle as fb

SPINS = [np.array(s, float) for s in itertools.product((-1, 1), repeat=3)]
COUPLINGS = [
    (0.7, -1.3, 0.4),
    (1.0, 2.0, -0.5),
    (-0.9, 0.6, 1.7),
    (0.3, 0.3, 0.3),
    (1.0, -1.0, 2.5),      # a+b=0
    (1.0, 2.5, -1.0),      # a+c=0
    (1.0, -1.0, -1.0),     # a+b=a+c=0
    (0.0, 0.0, 0.0),
]
ETAS = [0.5, 0.1, 0.01, 1e-3]
TOL = 1e-12


def cases():
    for X in SPINS:
        for c in COUPLINGS:
            for eta in ETAS:
                yield X, fb.unpack_J(c), eta, c


class TestCurvatureN3(unittest.TestCase):
    def test_01_pack_unpack(self):
        for c in COUPLINGS:
            np.testing.assert_array_equal(fb.pack_J(fb.unpack_J(c)), c)

    def test_02_symmetry_after_transport(self):
        for X, J, eta, _ in cases():
            for k in range(3):
                _, J1 = fb.mutate_postflip(X, J, k, eta)
                np.testing.assert_allclose(J1, J1.T, atol=TOL, rtol=0)
                self.assertTrue(np.all(np.diag(J1) == 0))

    def test_03_non_incident_edges_unchanged(self):
        for X, J, eta, _ in cases():
            for k in range(3):
                _, J1 = fb.mutate_postflip(X, J, k, eta)
                for i, j in itertools.combinations(range(3), 2):
                    if k not in (i, j):
                        self.assertEqual(J1[i, j], J[i, j])

    def test_04_spin_states_equal_after_both_paths(self):
        # Critère 6 corrigé : ce sont les SPINS qui coïncident, pas les fibres.
        for X, J, eta, _ in cases():
            XA, _ = fb.path_12(X, J, eta)
            XB, _ = fb.path_21(X, J, eta)
            np.testing.assert_array_equal(XA, XB)
            np.testing.assert_array_equal(XA[:2], -X[:2])
            self.assertEqual(XA[2], X[2])

    def test_05_fibres_match_analytic_exact(self):
        for X, J, eta, c in cases():
            o = fb.analytic_commutator_N3(X, J, eta)
            _, JA = fb.path_12(X, J, eta)
            _, JB = fb.path_21(X, J, eta)
            np.testing.assert_allclose(fb.pack_J(JA), o["j_A_exact"], atol=TOL, rtol=0)
            np.testing.assert_allclose(fb.pack_J(JB), o["j_B_exact"], atol=TOL, rtol=0)

    def test_06_commutator_equals_analytic(self):
        for X, J, eta, c in cases():
            o = fb.analytic_commutator_N3(X, J, eta)
            dJ = fb.commutator(X, J, eta)
            np.testing.assert_allclose(fb.pack_J(dJ), o["delta_j_exact"], atol=TOL, rtol=0)

    def test_07_leading_orders(self):
        # DeltaJ12/eta^2, DeltaJ13/eta, DeltaJ23/eta -> coefficient dominant (erreur O(eta))
        eta = 1e-4
        for X in SPINS:
            for c in COUPLINGS:
                J = fb.unpack_J(c)
                dj = fb.pack_J(fb.commutator(X, J, eta))
                ratios = dj / np.array([eta ** 2, eta, eta])
                lead = fb.analytic_commutator_N3(X, J, eta)["leading_coef"]
                scale = 1.0 + np.max(np.abs(c)) ** 2 * 4
                np.testing.assert_allclose(ratios, lead, atol=10 * eta * scale, rtol=0)

    def test_08_order_scaling_generic(self):
        # Cas générique : log-pente 2 pour J12, 1 pour J13 et J23.
        X, c = SPINS[0], (0.7, -1.3, 0.4)
        J = fb.unpack_J(c)
        e1, e2 = 1e-3, 1e-4
        d1 = np.abs(fb.pack_J(fb.commutator(X, J, e1)))
        d2 = np.abs(fb.pack_J(fb.commutator(X, J, e2)))
        slopes = np.log10(d1 / d2)
        np.testing.assert_allclose(slopes, [2, 1, 1], atol=0.05)

    def test_09_dominant_term_vanishing_is_not_exact_vanishing(self):
        # a+c=0 : coef dominant de DeltaJ13 nul, mais DeltaJ13 != 0 en général (terme eta^2).
        X, c, eta = np.array([1.0, 1.0, 1.0]), (1.0, 2.5, -1.0), 0.1
        o = fb.analytic_commutator_N3(X, fb.unpack_J(c), eta)
        self.assertEqual(o["leading_coef"][1], 0.0)
        self.assertNotEqual(o["delta_j_exact"][1], 0.0)
        dJ = fb.pack_J(fb.commutator(X, fb.unpack_J(c), eta))
        self.assertGreater(abs(dJ[1]), 1e-6)

    def test_10_delta_A3_factorisation_exact(self):
        for X, J, eta, c in cases():
            _, JA = fb.path_12(X, J, eta)
            _, JB = fb.path_21(X, J, eta)
            direct = fb.accessibility_A(JA)[2] - fb.accessibility_A(JB)[2]
            o = fb.analytic_commutator_N3(X, J, eta)
            self.assertAlmostEqual(direct, o["delta_A3_exact"], delta=1e-11)

    def test_11_delta_A3_first_order(self):
        for X in SPINS:
            for c in COUPLINGS:
                J = fb.unpack_J(c)
                errs = []
                for eta in (1e-3, 1e-4):
                    _, JA = fb.path_12(X, J, eta)
                    _, JB = fb.path_21(X, J, eta)
                    direct = fb.accessibility_A(JA)[2] - fb.accessibility_A(JB)[2]
                    errs.append(
                        abs(direct - fb.analytic_commutator_N3(X, J, eta)["delta_A3_first_order"])
                    )
                # reste O(eta^2) : l'erreur doit chuter ~100x (ou être ~0)
                self.assertTrue(errs[1] <= errs[0] / 50 or errs[1] < 1e-12, (X, c, errs))

    def test_12_integrable_control_commutes(self):
        for X, J, eta, _ in cases():
            dJ = fb.commutator(X, J, eta, psi_fn=fb.psi_integrable)
            np.testing.assert_allclose(dJ, 0.0, atol=TOL, rtol=0)

    def test_13_sym_transport_really_noncommuting(self):
        X, J = SPINS[0], fb.unpack_J((0.7, -1.3, 0.4))
        self.assertGreater(np.max(np.abs(fb.commutator(X, J, 0.1))), 1e-3)

    def test_14_threshold_crossing_and_margin(self):
        X, J, eta = SPINS[0], fb.unpack_J((0.7, -1.3, 0.4)), 0.3
        _, JA = fb.path_12(X, J, eta)
        _, JB = fb.path_21(X, J, eta)
        A3a, A3b = fb.accessibility_A(JA)[2], fb.accessibility_A(JB)[2]
        theta = 0.5 * (A3a + A3b)
        self.assertNotEqual(
            bool(fb.accessibility_set(JA, theta)[2]),
            bool(fb.accessibility_set(JB, theta)[2]),
        )
        self.assertAlmostEqual(fb.accessibility_margin(JA, theta)[2], abs(A3a - A3b) / 2, places=12)

    def test_15_input_validation(self):
        with self.assertRaises(ValueError):
            fb._validate_state([1, 0, 1])
        with self.assertRaises(ValueError):
            fb._validate_J(np.array([[0, 1], [2, 0]]))
        with self.assertRaises(IndexError):
            fb.mutate_postflip(SPINS[0], np.zeros((3, 3)), 5, 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
