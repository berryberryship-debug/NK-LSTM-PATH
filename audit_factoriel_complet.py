"""
analyse_hierarchique_factoriel.py
Projet : NK-LSTM-PATH

ANALYSE HIÉRARCHIQUE POST-HOC AUTO-ADAPTATIVE SUR res_factoriel_22500.json
(SANS NOUVELLE SIMULATION)

Détecte automatiquement le schéma JSON des résultats factoriels et réalise :
- La vérification d'invariance structurelle de J et A vis-à-vis de nk_seed
- Le modèle mixte croisé (init_seed & nk_seed aléatoires, eta fixe répété sur n = 25 unités)
- L'estimation des pentes individuelles et de leur dispersion sur les 25 unités.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


METRIC_ALIASES: Dict[str, List[str]] = {
    "mean_gap_fro": ["mean_gap_fro", "mean_non_commut_gap_fro", "mean_J_gap", "gap_mean"],
    "max_gap_fro": ["max_gap_fro", "max_non_commut_gap_fro", "max_J_gap", "gap_max"],
    "mean_delta_A": ["mean_delta_A", "mean_delta_A_norm", "delta_A_mean"],
    "max_delta_A": ["max_delta_A", "max_delta_A_norm", "delta_A_max"],
    "mean_delta_w": ["mean_delta_w", "mean_delta_w_norm", "delta_w_mean"],
    "max_delta_w": ["max_delta_w", "max_delta_w_norm", "delta_w_max"],
    "frac_nonzero_delta_w": [
        "frac_nonzero_delta_w",
        "frac_dw_nonzero",
        "dw_nonzero_frac",
        "dw_nonzero_pct",
    ],
    "total_crossings": [
        "total_crossings",
        "total_threshold_crossings",
        "crossings",
        "mask_disagreements",
    ],
}

INIT_SEED_KEYS = ["init_seed", "random_seed", "seed_init", "init"]
NK_SEED_KEYS = ["nk_seed", "seed_nk", "nk"]
ETA_KEYS = ["eta", "eta_val", "eta_value"]


def _find_first_key(mapping: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in mapping:
            return key
    lower_map = {k.lower(): k for k in mapping.keys()}
    for key in candidates:
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return None


def _extract_cell_records(payload: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extrait la liste des cellules factorielles et un dictionnaire de métadonnées."""
    meta: Dict[str, Any] = {}
    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) for item in payload):
            return payload, meta
        raise ValueError("La liste JSON racine ne contient pas de dictionnaires de cellules.")

    if not isinstance(payload, dict):
        raise ValueError("Format JSON racine non reconnu.")

    for k, v in payload.items():
        if isinstance(v, dict):
            meta[k] = v
        elif not isinstance(v, list):
            meta[k] = v

    for preferred in ("cells", "results", "records", "data", "entries", "rows", "factorial_cells"):
        val = payload.get(preferred)
        if isinstance(val, list) and val and all(isinstance(item, dict) for item in val):
            return val, meta

    for k, v in payload.items():
        if isinstance(v, list) and v and all(isinstance(item, dict) for item in v):
            return v, meta

    raise ValueError(
        f"Impossible de localiser une liste de cellules dans les clés JSON : {list(payload.keys())}"
    )


def _extract_grid_from_meta(
    meta: Dict[str, Any],
    cells: List[Dict[str, Any]],
    key_candidates: List[str],
    list_candidates: List[str],
    cast_fn: Any,
) -> List[Any]:
    for container in [meta] + [v for v in meta.values() if isinstance(v, dict)]:
        if not isinstance(container, dict):
            continue
        found_list_key = _find_first_key(container, list_candidates)
        if found_list_key and isinstance(container[found_list_key], list) and container[found_list_key]:
            return [cast_fn(x) for x in container[found_list_key]]

    cell_key = _find_first_key(cells[0], key_candidates)
    if cell_key is None:
        raise ValueError(f"Clé introuvable parmi {key_candidates} dans les cellules : {list(cells[0].keys())}")

    unique_vals: List[Any] = []
    for c in cells:
        val = cast_fn(c[cell_key])
        if not any(
            (val == existing) or (isinstance(val, float) and isinstance(existing, float) and np.isclose(val, existing))
            for existing in unique_vals
        ):
            unique_vals.append(val)

    unique_vals.sort()
    return unique_vals


def _resolve_metric_value(cell: Dict[str, Any], canonical_name: str, aliases: List[str]) -> float:
    key = _find_first_key(cell, aliases)
    if key is None:
        raise KeyError(
            f"Métrique '{canonical_name}' introuvable (cherché : {aliases}) dans : {list(cell.keys())}"
        )
    val = float(cell[key])
    if canonical_name == "frac_nonzero_delta_w" and "pct" in key.lower() and val > 1.0:
        val = val / 100.0
    return val


def load_factorial_tensor(
    json_path: str | Path,
) -> Tuple[Dict[str, Any], List[int], List[int], List[float], Dict[str, np.ndarray]]:
    raw_obj = json.loads(Path(json_path).read_text(encoding="utf-8"))
    cells, meta = _extract_cell_records(raw_obj)

    init_seeds = [
        int(x)
        for x in _extract_grid_from_meta(
            meta, cells, INIT_SEED_KEYS, ["init_seeds", "random_seeds", "seeds_init"], int
        )
    ]
    nk_seeds = [
        int(x)
        for x in _extract_grid_from_meta(
            meta, cells, NK_SEED_KEYS, ["nk_seeds", "seeds_nk"], int
        )
    ]
    eta_values = [
        float(x)
        for x in _extract_grid_from_meta(
            meta, cells, ETA_KEYS, ["eta_values", "etas", "eta_grid"], float
        )
    ]

    init_key = _find_first_key(cells[0], INIT_SEED_KEYS)
    nk_key = _find_first_key(cells[0], NK_SEED_KEYS)
    eta_key = _find_first_key(cells[0], ETA_KEYS)
    assert init_key is not None and nk_key is not None and eta_key is not None

    n_i, n_k, n_e = len(init_seeds), len(nk_seeds), len(eta_values)
    idx_i = {s: idx for idx, s in enumerate(init_seeds)}
    idx_k = {s: idx for idx, s in enumerate(nk_seeds)}

    tensors = {
        m: np.full((n_i, n_k, n_e), np.nan, dtype=np.float64) for m in METRIC_ALIASES
    }

    for c in cells:
        i = idx_i[int(c[init_key])]
        k = idx_k[int(c[nk_key])]
        c_eta = float(c[eta_key])
        e_matches = [idx for idx, val in enumerate(eta_values) if np.isclose(val, c_eta)]
        if len(e_matches) != 1:
            raise ValueError(f"Ambiguïté ou valeur absente sur eta={c_eta} dans {eta_values}")
        e = e_matches[0]

        for canonical_name, aliases in METRIC_ALIASES.items():
            tensors[canonical_name][i, k, e] = _resolve_metric_value(c, canonical_name, aliases)

    for m, arr in tensors.items():
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Valeurs manquantes ou non finies dans le tenseur '{m}'.")

    design_info = {
        "n_init_seeds": n_i,
        "n_nk_seeds": n_k,
        "n_eta_values": n_e,
        "n_units": n_i * n_k,
        "init_seeds": init_seeds,
        "nk_seeds": nk_seeds,
        "eta_values": eta_values,
        "meta_detected": meta,
    }
    return design_info, init_seeds, nk_seeds, eta_values, tensors


def verify_structural_nk_invariance(tensors: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Vérifie l'invariance structurelle de J et A vis-à-vis de nk_seed."""
    results: Dict[str, float] = {}
    for key in ["mean_gap_fro", "max_gap_fro", "mean_delta_A", "max_delta_A"]:
        results[key] = float(np.max(np.ptp(tensors[key], axis=1)))
    return results


def mixed_two_way_random_repeated_anova(Y: np.ndarray) -> Dict[str, float]:
    """
    Décomposition hiérarchique pour un plan croisé à 2 facteurs aléatoires
    (I: init_seed, K: nk_seed) et 1 facteur fixe répété (E: eta),
    sans réplication interne par cellule (n_units = I * K).
    """
    I, K, E = Y.shape
    grand = float(np.mean(Y))

    m_i = np.mean(Y, axis=(1, 2), keepdims=True)
    m_k = np.mean(Y, axis=(0, 2), keepdims=True)
    m_e = np.mean(Y, axis=(0, 1), keepdims=True)

    m_ik = np.mean(Y, axis=2, keepdims=True)
    m_ie = np.mean(Y, axis=1, keepdims=True)
    m_ke = np.mean(Y, axis=0, keepdims=True)

    ss_i = float(np.sum((m_i - grand) ** 2))
    ss_k = float(np.sum((m_k - grand) ** 2))
    ss_e = float(np.sum((m_e - grand) ** 2))

    ss_ik = float(np.sum((m_ik - m_i - m_k + grand) ** 2))
    ss_ie = float(np.sum((m_ie - m_i - m_e + grand) ** 2))
    ss_ke = float(np.sum((m_ke - m_k - m_e + grand) ** 2))

    res_ike = Y - m_ik - m_ie - m_ke + m_i + m_k + m_e - grand
    ss_ike = float(np.sum(res_ike ** 2))

    df_i = I - 1
    df_k = K - 1
    df_e = E - 1
    df_ik = df_i * df_k
    df_ie = df_i * df_e
    df_ke = df_k * df_e
    df_ike = df_i * df_k * df_e

    ms_i = ss_i / df_i
    ms_k = ss_k / df_k
    ms_e = ss_e / df_e
    ms_ik = ss_ik / df_ik
    ms_ie = ss_ie / df_ie
    ms_ke = ss_ke / df_ke
    ms_ike = ss_ike / df_ike

    f_i = ms_i / ms_ik if ms_ik > 0 else np.nan
    p_i = float(stats.f.sf(f_i, df_i, df_ik)) if np.isfinite(f_i) else np.nan

    f_k = ms_k / ms_ik if ms_ik > 0 else np.nan
    p_k = float(stats.f.sf(f_k, df_k, df_ik)) if np.isfinite(f_k) else np.nan

    f_ik = ms_ik / ms_ike if ms_ike > 0 else np.nan
    p_ik = float(stats.f.sf(f_ik, df_ik, df_ike)) if np.isfinite(f_ik) else np.nan

    f_ie = ms_ie / ms_ike if ms_ike > 0 else np.nan
    p_ie = float(stats.f.sf(f_ie, df_ie, df_ike)) if np.isfinite(f_ie) else np.nan

    f_ke = ms_ke / ms_ike if ms_ike > 0 else np.nan
    p_ke = float(stats.f.sf(f_ke, df_ke, df_ike)) if np.isfinite(f_ke) else np.nan

    denom_e = ms_ie + ms_ke - ms_ike
    if denom_e > 0:
        f_e = ms_e / denom_e
        num_df = denom_e ** 2
        den_df = (ms_ie ** 2) / df_ie + (ms_ke ** 2) / df_ke + (ms_ike ** 2) / df_ike
        df_e_denom = num_df / den_df if den_df > 0 else np.nan
        p_e = float(stats.f.sf(f_e, df_e, df_e_denom)) if np.isfinite(df_e_denom) else np.nan
    else:
        f_e, df_e_denom, p_e = np.nan, np.nan, np.nan

    var_ike = max(0.0, ms_ike)
    var_ik = max(0.0, (ms_ik - ms_ike) / E)
    var_i = max(0.0, (ms_i - ms_ik) / (K * E))
    var_k = max(0.0, (ms_k - ms_ik) / (I * E))
    var_ie = max(0.0, (ms_ie - ms_ike) / K)
    var_ke = max(0.0, (ms_ke - ms_ike) / I)

    var_random_total = var_i + var_k + var_ik + var_ie + var_ke + var_ike
    icc_init = var_i / var_random_total if var_random_total > 0 else 0.0
    icc_nk = var_k / var_random_total if var_random_total > 0 else 0.0
    icc_ik = var_ik / var_random_total if var_random_total > 0 else 0.0

    return {
        "df_i": df_i, "df_k": df_k, "df_e": df_e,
        "df_ik": df_ik, "df_ie": df_ie, "df_ke": df_ke, "df_ike": df_ike,
        "ms_e": ms_e, "ms_i": ms_i, "ms_k": ms_k,
        "ms_ik": ms_ik, "ms_ie": ms_ie, "ms_ke": ms_ke, "ms_ike": ms_ike,
        "F_eta": f_e, "df2_eta": df_e_denom, "p_eta": p_e,
        "F_init": f_i, "p_init": p_i,
        "F_nk": f_k, "p_nk": p_k,
        "F_init_x_nk": f_ik, "p_init_x_nk": p_ik,
        "F_eta_x_init": f_ie, "p_eta_x_init": p_ie,
        "F_eta_x_nk": f_ke, "p_eta_x_nk": p_ke,
        "var_init": var_i,
        "var_nk": var_k,
        "var_init_x_nk": var_ik,
        "var_eta_x_init": var_ie,
        "var_eta_x_nk": var_ke,
        "var_resid": var_ike,
        "var_random_total": var_random_total,
        "icc_init": icc_init,
        "icc_nk": icc_nk,
        "icc_init_x_nk": icc_ik,
    }


def analyze_unit_slopes(eta_values: List[float], Y: np.ndarray) -> Dict[str, Any]:
    """Calcule les pentes individuelles par unité (i, k) et leur dispersion."""
    etas = np.asarray(eta_values, dtype=np.float64)
    I, K, _ = Y.shape

    def fit_slopes(mask: np.ndarray) -> Dict[str, Any]:
        x = etas[mask]
        xc = x - np.mean(x)
        denom = float(np.sum(xc ** 2))
        slopes = np.empty((I, K), dtype=np.float64)
        r2_vals = np.empty((I, K), dtype=np.float64)
        for i in range(I):
            for k in range(K):
                y = Y[i, k, mask]
                yc = y - np.mean(y)
                b = float(np.sum(xc * yc)) / denom if denom > 0 else 0.0
                slopes[i, k] = b
                ss_tot = float(np.sum(yc ** 2))
                ss_res = float(np.sum((yc - b * xc) ** 2))
                r2_vals[i, k] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        flat = slopes.ravel()
        n = flat.size
        mean_b = float(np.mean(flat))
        sd_b = float(np.std(flat, ddof=1)) if n > 1 else 0.0
        se_b = sd_b / np.sqrt(n) if n > 0 else 0.0
        t_crit = float(stats.t.ppf(0.975, df=max(1, n - 1)))

        grand_b = np.mean(slopes)
        ss_tot_b = float(np.sum((slopes - grand_b) ** 2))
        ss_i_b = float(np.sum((np.mean(slopes, axis=1, keepdims=True) - grand_b) ** 2))
        ss_k_b = float(np.sum((np.mean(slopes, axis=0, keepdims=True) - grand_b) ** 2))
        ss_ik_b = ss_tot_b - ss_i_b - ss_k_b

        return {
            "n_units": n,
            "mean_slope": mean_b,
            "sd_slope": sd_b,
            "ci95_low": mean_b - t_crit * se_b,
            "ci95_high": mean_b + t_crit * se_b,
            "min_slope": float(np.min(flat)),
            "max_slope": float(np.max(flat)),
            "mean_r2": float(np.mean(r2_vals)),
            "pct_var_slope_init": 100.0 * ss_i_b / ss_tot_b if ss_tot_b > 0 else 0.0,
            "pct_var_slope_nk": 100.0 * ss_k_b / ss_tot_b if ss_tot_b > 0 else 0.0,
            "pct_var_slope_inter": 100.0 * ss_ik_b / ss_tot_b if ss_tot_b > 0 else 0.0,
        }

    return {
        "full_grid": fit_slopes(np.ones_like(etas, dtype=bool)),
        "low_eta_le_025": fit_slopes(etas <= 0.25 + 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse hiérarchique post-hoc auto-adaptative sur res_factoriel_22500.json"
    )
    parser.add_argument("--json-in", type=str, default="res_factoriel_22500.json")
    parser.add_argument("--json-out", type=str, default="res_analyse_hierarchique.json")
    args = parser.parse_args()

    design, init_seeds, nk_seeds, eta_values, tensors = load_factorial_tensor(args.json_in)

    print("=" * 118)
    print(
        f"ANALYSE HIÉRARCHIQUE POST-HOC — {design['n_units']} UNITÉS CROISÉES "
        f"({len(init_seeds)} init × {len(nk_seeds)} nk) × {len(eta_values)} RÉGIMES DE ETA"
    )
    print("=" * 118)
    print(f"Source : {args.json_in}")
    print(f"Grilles : init={init_seeds} | nk={nk_seeds} | eta={eta_values}")

    inv = verify_structural_nk_invariance(tensors)
    print("\n" + "-" * 118)
    print("1. INVARIANCE STRUCTURELLE VIS-À-VIS DE nk_seed (ÉCART MAX ENTRE PAYSAGES)")
    print("-" * 118)
    for k, v in inv.items():
        print(f"  {k:<22} : max_ptp_across_nk = {v:.3e}")

    print("\n" + "-" * 118)
    print("2. MODÈLE MIXTE CROISÉ (init_seed & nk_seed ALÉATOIRES, eta FIXE RÉPÉTÉ)")
    print("-" * 118)
    mixed_summary: Dict[str, Any] = {}
    for key, label in [
        ("mean_delta_w", "Norme ||Δw||2"),
        ("max_delta_w", "Max |Δw|"),
        ("frac_nonzero_delta_w", "Fraction Δw≠0"),
        ("total_crossings", "Désaccords D_T"),
    ]:
        res = mixed_two_way_random_repeated_anova(tensors[key])
        mixed_summary[key] = res
        print(f"\n  >>> Observable : {label}")
        print(
            f"      Effet fixe eta     : F({res['df_e']}, {res['df2_eta']:.2f}) = {res['F_eta']:.3f} "
            f"(p = {res['p_eta']:.3e})"
        )
        print(
            f"      Effet aléat. init  : F({res['df_i']}, {res['df_ik']}) = {res['F_init']:.3f} "
            f"(p = {res['p_init']:.3e}) | Var = {res['var_init']:.6e} (ICC_init = {res['icc_init']*100:.2f}%)"
        )
        print(
            f"      Effet aléat. nk    : F({res['df_k']}, {res['df_ik']}) = {res['F_nk']:.3f} "
            f"(p = {res['p_nk']:.3e}) | Var = {res['var_nk']:.6e} (ICC_nk = {res['icc_nk']*100:.2f}%)"
        )
        print(
            f"      Interaction init×nk: F({res['df_ik']}, {res['df_ike']}) = {res['F_init_x_nk']:.3f} "
            f"(p = {res['p_init_x_nk']:.3e}) | Var = {res['var_init_x_nk']:.6e} (ICC_ik = {res['icc_init_x_nk']*100:.2f}%)"
        )
        print(
            f"      Inter. eta×init    : F({res['df_ie']}, {res['df_ike']}) = {res['F_eta_x_init']:.3f} "
            f"(p = {res['p_eta_x_init']:.3e}) | Var = {res['var_eta_x_init']:.6e}"
        )
        print(
            f"      Inter. eta×nk      : F({res['df_ke']}, {res['df_ike']}) = {res['F_eta_x_nk']:.3f} "
            f"(p = {res['p_eta_x_nk']:.3e}) | Var = {res['var_eta_x_nk']:.6e}"
        )

    print("\n" + "-" * 118)
    print("3. DISTRIBUTION DES PENTES INDIVIDUELLES PAR UNITÉ (i, k)")
    print("-" * 118)
    slopes_summary: Dict[str, Any] = {}
    for key, label in [
        ("mean_gap_fro", "Écart ||JA-JB||F"),
        ("mean_delta_A", "Norme ||ΔA||2"),
        ("mean_delta_w", "Norme ||Δw||2"),
        ("total_crossings", "Désaccords D_T"),
    ]:
        sl = analyze_unit_slopes(eta_values, tensors[key])
        slopes_summary[key] = sl
        fg = sl["full_grid"]
        lo = sl["low_eta_le_025"]
        print(f"\n  >>> {label}")
        print(
            f"      Grille complète  : pente = {fg['mean_slope']:.5f} ± {fg['sd_slope']:.5f} "
            f"(IC95% [{fg['ci95_low']:.5f}, {fg['ci95_high']:.5f}]) | R² moy = {fg['mean_r2']:.4f} | "
            f"Var(pente): init={fg['pct_var_slope_init']:.1f}%, nk={fg['pct_var_slope_nk']:.1f}%, inter={fg['pct_var_slope_inter']:.1f}%"
        )
        print(
            f"      Régime eta<=0.25 : pente = {lo['mean_slope']:.5f} ± {lo['sd_slope']:.5f} "
            f"(IC95% [{lo['ci95_low']:.5f}, {lo['ci95_high']:.5f}]) | R² moy = {lo['mean_r2']:.4f} | "
            f"Var(pente): init={lo['pct_var_slope_init']:.1f}%, nk={lo['pct_var_slope_nk']:.1f}%, inter={lo['pct_var_slope_inter']:.1f}%"
        )

    print("\n" + "=" * 118)

    out_payload = {
        "source_json": args.json_in,
        "design": design,
        "structural_nk_invariance": inv,
        "mixed_model_25_units": mixed_summary,
        "unit_slopes_25_units": slopes_summary,
    }
    Path(args.json_out).write_text(json.dumps(out_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
