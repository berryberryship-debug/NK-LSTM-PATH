import numpy as np
from itertools import combinations

from nk_landscape import NKLandscape
from fiber_bundle import mutate_postflip, accessibility_A
from trajectories import transition_rates


def create_nk_landscape(n_sites: int, nk_K: int, nk_seed: int):
    return NKLandscape.generate(
        n_spins=n_sites,
        k_epistasis=nk_K,
        seed=nk_seed,
        neighbor_mode="random",
    )


def sample_initial_states(n_sites: int, n_states: int, seed: int, scale_J: float = 0.2):
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n_states):
        X0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n_sites,))
        raw_J = rng.normal(0.0, scale_J, size=(n_sites, n_sites))
        J0 = 0.5 * (raw_J + raw_J.T)
        np.fill_diagonal(J0, 0.0)
        states.append((X0, J0))
    return states


def run_single_pair_audit(
    n_sites: int,
    n_states: int,
    init_seed: int,
    nk_seed: int,
    scale_J: float,
    eta: float,
    mu: float,
    beta: float,
    theta_A: float,
    lam: float,
    nk_K: int,
    tol_w: float = 1e-12,
):
    states = sample_initial_states(n_sites, n_states, init_seed, scale_J)
    landscape = create_nk_landscape(n_sites, nk_K, nk_seed)
    site_pairs = list(combinations(range(n_sites), 2))

    diffs_J = []
    max_sym_err = 0.0
    max_diag_err = 0.0
    x_identical_count = 0
    dw_nonzero_count = 0
    total_mask_disagreements = 0
    n_comp = 0

    for X0, J0 in states:
        for k, m in site_pairs:
            n_comp += 1

            X_A1, J_A1 = mutate_postflip(X0.copy(), J0.copy(), k, eta)
            X_A, J_A = mutate_postflip(X_A1, J_A1, m, eta)

            X_B1, J_B1 = mutate_postflip(X0.copy(), J0.copy(), m, eta)
            X_B, J_B = mutate_postflip(X_B1, J_B1, k, eta)

            diffs_J.append(float(np.linalg.norm(J_A - J_B, ord="fro")))
            max_sym_err = max(
                max_sym_err,
                float(np.max(np.abs(J_A - J_A.T))),
                float(np.max(np.abs(J_B - J_B.T))),
            )
            max_diag_err = max(
                max_diag_err,
                float(np.max(np.abs(np.diag(J_A)))),
                float(np.max(np.abs(np.diag(J_B)))),
            )

            if np.array_equal(X_A, X_B):
                x_identical_count += 1

            active_A = accessibility_A(J_A) > theta_A
            active_B = accessibility_A(J_B) > theta_A
            total_mask_disagreements += int(np.sum(active_A != active_B))

            w_A = transition_rates(
                X_A,
                J_A,
                mu=mu,
                beta=beta,
                theta_A=theta_A,
                lam=lam,
                fnk_fn=landscape,
            )
            w_B = transition_rates(
                X_B,
                J_B,
                mu=mu,
                beta=beta,
                theta_A=theta_A,
                lam=lam,
                fnk_fn=landscape,
            )
            if np.max(np.abs(w_A - w_B)) > tol_w:
                dw_nonzero_count += 1

    return {
        "n_comp": n_comp,
        "mean_J": float(np.mean(diffs_J)),
        "max_J": float(np.max(diffs_J)),
        "err_sym": max_sym_err,
        "err_diag": max_diag_err,
        "x_id_pct": 100.0 * x_identical_count / n_comp,
        "dw_nonzero_pct": 100.0 * dw_nonzero_count / n_comp,
        "mask_disagreements": total_mask_disagreements,
        "mask_disagreements_pct": 100.0 * total_mask_disagreements / (n_comp * n_sites),
    }


def main():
    n_sites = 6
    n_states = 10
    scale_J = 0.2

    print("=" * 90)
    print("NK-LSTM-PATH — AUDIT DE REPRODUCTIBILITÉ MULTI-GRAINES (N = 6)")
    print("=" * 90)

    eta = float(input("1. eta fixe [défaut : 0.25] : ").strip() or "0.25")
    mu = float(input("2. mu — échelle des taux [défaut : 1.0] : ").strip() or "1.0")
    beta = float(input("3. beta — sensibilité fitness [défaut : 1.0] : ").strip() or "1.0")
    theta_A = float(input("4. theta_A — seuil d'accessibilité [défaut : 0.1] : ").strip() or "0.1")
    lam = float(input("5. lam — pondération du couplage [défaut : 0.5] : ").strip() or "0.5")
    nk_K = int(input("6. K — dépendances NK [défaut : 3] : ").strip() or "3")

    raw_init = input("7. Graines initiales (X0, J0) [défaut : 7, 13, 21, 42, 99] : ").strip()
    init_seeds = [int(s) for s in raw_init.split(",")] if raw_init else [7, 13, 21, 42, 99]

    raw_nk = input("8. Graines du paysage NK [défaut : 7, 13, 21, 42, 99] : ").strip()
    nk_seeds = [int(s) for s in raw_nk.split(",")] if raw_nk else [7, 13, 21, 42, 99]

    header = (
        f"{'init':>5} | {'nk':>4} | {'Écart moy J':>11} | {'Écart max J':>11} | "
        f"{'Err sym':>8} | {'Err diag':>8} | {'X id.':>6} | {'Δw≠0':>6} | {'Désaccords D_T':>16}"
    )

    fixed_nk = nk_seeds[0]
    print("\n" + "-" * len(header))
    print(f"BLOC A — VARIATION DE init_seed (nk_seed fixé à {fixed_nk}, eta = {eta})")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    res_A = []
    for s_init in init_seeds:
        r = run_single_pair_audit(
            n_sites, n_states, s_init, fixed_nk, scale_J,
            eta, mu, beta, theta_A, lam, nk_K
        )
        res_A.append(r)
        print(
            f"{s_init:>5} | {fixed_nk:>4} | {r['mean_J']:>11.6f} | {r['max_J']:>11.6f} | "
            f"{r['err_sym']:>8.2e} | {r['err_diag']:>8.2e} | {r['x_id_pct']:>5.1f}% | "
            f"{r['dw_nonzero_pct']:>5.1f}% | {r['mask_disagreements']:>4} ({r['mask_disagreements_pct']:>4.1f}%)"
        )

    fixed_init = init_seeds[0]
    print("\n" + "-" * len(header))
    print(f"BLOC B — VARIATION DE nk_seed (init_seed fixé à {fixed_init}, eta = {eta})")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    res_B = []
    for s_nk in nk_seeds:
        r = run_single_pair_audit(
            n_sites, n_states, fixed_init, s_nk, scale_J,
            eta, mu, beta, theta_A, lam, nk_K
        )
        res_B.append(r)
        print(
            f"{fixed_init:>5} | {s_nk:>4} | {r['mean_J']:>11.6f} | {r['max_J']:>11.6f} | "
            f"{r['err_sym']:>8.2e} | {r['err_diag']:>8.2e} | {r['x_id_pct']:>5.1f}% | "
            f"{r['dw_nonzero_pct']:>5.1f}% | {r['mask_disagreements']:>4} ({r['mask_disagreements_pct']:>4.1f}%)"
        )

    print("\n" + "=" * len(header))
    print("SYNTHÈSE MULTI-GRAINES (MOYENNE ± ÉCART-TYPE)")
    print("=" * len(header))
    for label, group in [("Bloc A (init_seed variable)", res_A), ("Bloc B (nk_seed variable)", res_B)]:
        m_J = [x["mean_J"] for x in group]
        m_dw = [x["dw_nonzero_pct"] for x in group]
        m_md = [x["mask_disagreements"] for x in group]
        print(
            f"{label:<28} : Écart moy J = {np.mean(m_J):.6f} ± {np.std(m_J):.6f} | "
            f"Δw≠0 = {np.mean(m_dw):.1f}% ± {np.std(m_dw):.1f}% | "
            f"Désaccords D_T = {np.mean(m_md):.1f} ± {np.std(m_md):.1f}"
        )


if __name__ == "__main__":
    main()
