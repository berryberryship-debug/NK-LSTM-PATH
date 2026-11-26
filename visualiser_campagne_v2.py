#!/usr/bin/env python3
"""
visualiser_campagne_v2.py

Outil d'analyse et de visualisation pour NK-LSTM-GSA V2 :
1. Fusionne et dédoublonne les exécutions de `runs/campagne_v2` et `runs/v2_*`.
2. Vérifie l'homogénéité stricte des hyperparamètres avant toute agrégation multi-graines.
3. Sélectionne le checkpoint à inspecter indépendamment du jeu de test (`best_val_loss`
   ou graine pré-enregistrée).
4. Extrait directement les paramètres `edge_logits` de `UndirectedPositiveEdgeCosts`
   pour reconstruire la table (2^N, N) des coûts d'arêtes c_theta(x, a) et la matrice
   géodésique exacte S* (2^N, 2^N) sur l'hypercube.
5. Isole le Gramien de `gsa.x_proj.weight` dans un diagnostic de projection séparé.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


LABELS: dict[tuple[str, str], str] = {
    ("hybrid_concat", "learned"): "Hybrid Concat (GSA apprise)",
    ("hybrid_concat", "fixed"): "Hybrid Concat (GSA fixe)",
    ("hybrid_concat", "none"): "Hybrid Concat (sans GSA)",
    ("gsa_only", "learned"): "GSA Only (apprise)",
    ("gsa_only", "fixed"): "GSA Only (fixe)",
    ("gsa_only", "none"): "GSA Only (sans géométrie)",
    ("lstm_only", "none"): "LSTM Only",
    ("hybrid_gated", "learned"): "Hybrid Gated (apprise)",
}

COLORS: dict[tuple[str, str], str] = {
    ("hybrid_concat", "learned"): "#1b9e77",
    ("hybrid_concat", "fixed"): "#66a61e",
    ("hybrid_concat", "none"): "#a6761d",
    ("gsa_only", "learned"): "#d95f02",
    ("gsa_only", "fixed"): "#e6ab02",
    ("gsa_only", "none"): "#b15928",
    ("lstm_only", "none"): "#7570b3",
    ("hybrid_gated", "learned"): "#e7298a",
}

EXCLUDED_CONFIG_KEYS = {
    "seed",
    "output_dir",
    "run_dir",
    "device",
    "num_workers",
    "save_every",
}


def _f(val: Any) -> float | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        v = float(val)
        return v if math.isfinite(v) else None
    if isinstance(val, str):
        try:
            v = float(val)
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


def _mean_std(vals: list[float | None]) -> tuple[float | None, float | None, int]:
    clean = [v for v in vals if v is not None and math.isfinite(v)]
    n = len(clean)
    if n == 0:
        return None, None, 0
    if n == 1:
        return clean[0], 0.0, 1
    return statistics.mean(clean), statistics.stdev(clean), n


def _canonical_hparams(cfg: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Construit une signature canonique des hyperparamètres hors graine aléatoire."""
    items: list[tuple[str, str]] = []
    for k, v in sorted(cfg.items()):
        if k in EXCLUDED_CONFIG_KEYS:
            continue
        if isinstance(v, float):
            items.append((k, f"{v:.10g}"))
        else:
            items.append((k, json.dumps(v, sort_keys=True, ensure_ascii=False)))
    return tuple(items)


def load_run_bundle(run_dir: Path, source_priority: int = 0) -> dict[str, Any] | None:
    """Charge un dossier d'exécution complet (config, test_metrics, history, best.pt)."""
    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        return None

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    hist_path = run_dir / "history.json"
    hist = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else []

    losses = data.get("test_losses", {})
    metrics = data.get("test_metrics", {})
    base = data.get("baseline", {})

    omega_mse = _f(metrics.get("mse"))
    if omega_mse is None:
        omega_mse = _f(losses.get("loss_omega"))

    base_omega = _f(base.get("mse_omega"))
    if base_omega is None:
        base_omega = _f(metrics.get("baseline_mse_omega"))

    seed_raw = cfg.get("seed")
    seed = int(seed_raw) if isinstance(seed_raw, (int, float)) and not isinstance(seed_raw, bool) else None

    n_init_raw = cfg.get("n_initial_states")
    n_init = int(n_init_raw) if isinstance(n_init_raw, (int, float)) and not isinstance(n_init_raw, bool) else None

    ckpt = run_dir / "best.pt"
    arch_mode = str(data.get("arch_mode", cfg.get("arch_mode", "?")))
    geometry_mode = str(data.get("geometry_mode", cfg.get("geometry_mode", "?")))
    transport_mode = str(data.get("transport_mode", cfg.get("transport_mode", "?")))
    eta = _f(cfg.get("eta"))

    return {
        "dossier": run_dir.name,
        "run_path": str(run_dir),
        "source_priority": source_priority,
        "arch_mode": arch_mode,
        "geometry_mode": geometry_mode,
        "transport_mode": transport_mode,
        "eta": eta,
        "seed": seed,
        "n_sites": int(cfg.get("n_sites", 6)),
        "n_initial_states": n_init,
        "edge_eps": float(cfg.get("edge_eps", 1e-4)),
        "hparams_sig": _canonical_hparams(cfg),
        "config": cfg,
        "best_val": _f(data.get("best_val_loss")),
        "term_mse": _f(losses.get("loss_terminal")),
        "omega_mse": omega_mse,
        "base_omega": base_omega,
        "cos_nonzero": _f(metrics.get("mean_cosine_nonzero")),
        "history": hist if isinstance(hist, list) else [],
        "ckpt_path": ckpt if ckpt.exists() else None,
    }


def discover_runs(
    base_dir: Path,
    extra_dir: Path | None = None,
    extra_pattern: str = "v2_*",
) -> list[dict[str, Any]]:
    """
    Parcourt systématiquement `base_dir` ET `extra_dir/extra_pattern`,
    puis dédoublonne les exécutions identiques (même configuration et même graine).
    """
    raw_bundles: list[dict[str, Any]] = []

    if base_dir.exists():
        for sub in sorted(base_dir.iterdir()):
            if sub.is_dir():
                b = load_run_bundle(sub, source_priority=0)
                if b and b["transport_mode"] == "symmetric":
                    raw_bundles.append(b)

    if extra_dir is not None and extra_dir.exists():
        for sub in sorted(extra_dir.glob(extra_pattern)):
            if not sub.is_dir():
                continue
            if base_dir.exists() and sub.resolve() == base_dir.resolve():
                continue
            b = load_run_bundle(sub, source_priority=1)
            if b and b["transport_mode"] == "symmetric":
                raw_bundles.append(b)

    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicates_skipped = 0
    for b in raw_bundles:
        dedup_key = (
            b["arch_mode"],
            b["geometry_mode"],
            b["transport_mode"],
            b["hparams_sig"],
            b["seed"] if b["seed"] is not None else b["dossier"],
        )
        if dedup_key not in deduped:
            deduped[dedup_key] = b
        else:
            duplicates_skipped += 1

    bundles = sorted(
        deduped.values(),
        key=lambda x: (x["eta"] or 0.0, x["arch_mode"], x["geometry_mode"], x["seed"] or -1, x["dossier"]),
    )
    print(f"Runs symétriques chargés : {len(bundles)} uniques (doublons ignorés : {duplicates_skipped})")
    for b in bundles:
        print(
            f"  - {b['run_path']} | arch={b['arch_mode']} | geom={b['geometry_mode']} | "
            f"eta={b['eta']} | n_init={b['n_initial_states']} | seed={b['seed']} | val={b['best_val']}"
        )
    return bundles


def filter_homogeneous_cohort(
    bundles: list[dict[str, Any]],
    target_n_init: int | None = None,
) -> list[dict[str, Any]]:
    """
    Garantit que les runs comparés partagent exactement la même configuration
    d'hyperparamètres (hors `seed` et `eta`).
    """
    if not bundles:
        return []

    filtered = bundles
    if target_n_init is not None:
        filtered = [b for b in bundles if b["n_initial_states"] == target_n_init]
        excluded = len(bundles) - len(filtered)
        if excluded > 0:
            print(f"[FILTRE COHORTE] {excluded} run(s) exclu(s) car n_initial_states != {target_n_init}.")

    if not filtered:
        return []

    def sig_without_eta(b: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        return tuple((k, v) for k, v in b["hparams_sig"] if k not in {"eta", "arch_mode", "geometry_mode"})

    by_sig: dict[tuple[tuple[str, str], ...], list[dict[str, Any]]] = {}
    for b in filtered:
        by_sig.setdefault(sig_without_eta(b), []).append(b)

    if len(by_sig) > 1:
        _, dominant_runs = max(by_sig.items(), key=lambda kv: len(kv[1]))
        print(
            f"[AVERTISSEMENT COHORTE] {len(by_sig)} configurations d'hyperparamètres distinctes détectées. "
            f"Conservation de la cohorte homogène majoritaire ({len(dominant_runs)} runs)."
        )
        return dominant_runs

    return filtered


def _load_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")

    if isinstance(payload, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            sub = payload.get(k)
            if isinstance(sub, dict):
                return {nk: nv for nk, nv in sub.items() if isinstance(nv, torch.Tensor)}
        return {nk: nv for nk, nv in payload.items() if isinstance(nv, torch.Tensor)}
    return {}


def compute_undirected_hypercube_geometry(
    edge_logits: torch.Tensor,
    n_sites: int,
    edge_eps: float = 1e-4,
) -> dict[str, np.ndarray]:
    """
    Reconstruit exactement les objets géométriques de `UndirectedPositiveEdgeCosts`
    (`gsa_v1_core.py`, lignes 122-172) :
      - undirected_costs : vecteur (N * 2^(N-1),) = softplus(edge_logits) + edge_eps
      - cost_table       : table (2^N, N) symétrique c_theta(x, a)
      - shortest_paths   : matrice géodésique S* (2^N, 2^N) via Floyd-Warshall
    """
    n_states = 1 << n_sites
    n_undirected = n_sites * (1 << (n_sites - 1))
    logits_1d = edge_logits.detach().cpu().float().reshape(-1)
    if logits_1d.numel() != n_undirected:
        raise ValueError(
            f"Dimension inattendue pour edge_logits : {logits_1d.numel()} (attendu {n_undirected} pour N={n_sites})."
        )

    edge_id_table = torch.full((n_states, n_sites), -1, dtype=torch.long)
    next_id = 0
    for state in range(n_states):
        for site in range(n_sites):
            neighbour = state ^ (1 << site)
            if state < neighbour:
                edge_id_table[state, site] = next_id
                edge_id_table[neighbour, site] = next_id
                next_id += 1

    costs = F.softplus(logits_1d) + float(edge_eps)
    table = costs[edge_id_table]

    dist = torch.full((n_states, n_states), float("inf"), dtype=torch.float32)
    idx = torch.arange(n_states)
    dist[idx, idx] = 0.0
    for site in range(n_sites):
        dist[idx, idx ^ (1 << site)] = table[:, site]
    for pivot in range(n_states):
        dist = torch.minimum(dist, dist[:, pivot : pivot + 1] + dist[pivot : pivot + 1, :])

    return {
        "edge_logits": logits_1d.numpy().astype(np.float64),
        "undirected_costs": costs.numpy().astype(np.float64),
        "cost_table": table.numpy().astype(np.float64),
        "shortest_paths": dist.numpy().astype(np.float64),
    }


def inspect_checkpoint_geometry(
    ckpt_path: Path,
    n_sites: int = 6,
    edge_eps: float = 1e-4,
) -> tuple[dict[str, np.ndarray] | None, str, np.ndarray | None, str | None]:
    """
    Extrait :
      1. Les véritables coûts d'arêtes GSA (`edge_logits`) et la matrice géodésique S*.
      2. Séparément, le diagnostic de représentation `gsa.x_proj.weight` (si présent).
    """
    state = _load_state_dict(ckpt_path)
    if not state:
        return None, "checkpoint vide ou illisible", None, None

    n_undirected = n_sites * (1 << (n_sites - 1))
    geom_result: dict[str, np.ndarray] | None = None
    geom_source = "paramètre edge_logits introuvable dans le checkpoint"

    for name, tensor in state.items():
        if name.endswith("edge_logits") and tensor.numel() == n_undirected:
            geom_result = compute_undirected_hypercube_geometry(tensor, n_sites=n_sites, edge_eps=edge_eps)
            geom_source = f"paramètre exact '{name}' (forme {tuple(tensor.shape)}, eps={edge_eps})"
            break

    xproj_gram: np.ndarray | None = None
    xproj_name: str | None = None
    for name, tensor in state.items():
        if name.endswith("x_proj.weight") and tensor.ndim == 2:
            w = tensor.detach().cpu().float().numpy().astype(np.float64)
            xproj_gram = np.abs(w.T @ w)
            xproj_name = name
            break

    return geom_result, geom_source, xproj_gram, xproj_name


def select_checkpoint_blind_to_test(
    bundles: list[dict[str, Any]],
    target_eta: float,
    ckpt_override: str = "",
    select_seed: int | None = None,
) -> tuple[dict[str, Any] | None, Path | None, str]:
    """
    Sélectionne le checkpoint à inspecter SANS jamais consulter les métriques de test :
      1. `--ckpt` explicite si fourni ;
      2. sinon graine pré-enregistrée (`--select-seed`) si spécifiée ;
      3. sinon meilleur score de validation (`best_val_loss`) à `target_eta`.
    """
    if ckpt_override:
        p = Path(ckpt_override)
        return None, p if p.exists() else None, f"sélection explicite (--ckpt {p})"

    candidates = [
        b
        for b in bundles
        if b["geometry_mode"] == "learned"
        and b["ckpt_path"] is not None
        and b["eta"] is not None
        and math.isclose(b["eta"], target_eta, rel_tol=1e-5, abs_tol=1e-7)
    ]
    if not candidates:
        candidates = [b for b in bundles if b["geometry_mode"] == "learned" and b["ckpt_path"] is not None]
    if not candidates:
        return None, None, "aucun checkpoint avec geometry_mode='learned' trouvé"

    if select_seed is not None:
        seeded = [b for b in candidates if b["seed"] == select_seed]
        if seeded:
            chosen = seeded[0]
            return chosen, chosen["ckpt_path"], f"graine pré-enregistrée seed={select_seed} (hors test)"

    with_val = [b for b in candidates if b["best_val"] is not None]
    if with_val:
        with_val.sort(key=lambda b: (b["best_val"], b["seed"] if b["seed"] is not None else 999999))
        chosen = with_val[0]
        return (
            chosen,
            chosen["ckpt_path"],
            f"validation minimale best_val_loss={chosen['best_val']:.6f} (indépendant du test)",
        )

    candidates.sort(key=lambda b: (b["seed"] if b["seed"] is not None else 999999, b["dossier"]))
    chosen = candidates[0]
    return chosen, chosen["ckpt_path"], "ordre déterministe de protocole (hors test)"


def export_gsa_geometry_artifacts(
    geom: dict[str, np.ndarray],
    out_dir: Path,
    n_sites: int,
    edge_eps: float,
) -> tuple[Path, Path]:
    """Exporte la table (2^N, N) des coûts d'arêtes et la matrice géodésique S* (2^N, 2^N)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_table = geom["cost_table"]
    shortest_paths = geom["shortest_paths"]
    n_states = cost_table.shape[0]

    table_csv = out_dir / "table_couts_aretes_gsa_hypercube.csv"
    with open(table_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["state_idx", "state_bits"] + [f"site_{a}_cost" for a in range(n_sites)])
        for s in range(n_states):
            bits = format(s, f"0{n_sites}b")
            w.writerow([s, bits] + [f"{cost_table[s, a]:.6f}" for a in range(n_sites)])

    s_star_csv = out_dir / "matrice_geodesique_S_star_gsa.csv"
    with open(s_star_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["state"] + [f"s{j}_{format(j, f'0{n_sites}b')}" for j in range(n_states)])
        for i in range(n_states):
            row_lbl = f"s{i}_{format(i, f'0{n_sites}b')}"
            w.writerow([row_lbl] + [f"{shortest_paths[i, j]:.6f}" for j in range(n_states)])

    init_cost = math.log(2.0) + edge_eps
    uc = geom["undirected_costs"]
    print("\nStatistiques des 192 coûts d'arêtes appris c_e = softplus(edge_logits_e) + eps :")
    print(f"  - Valeur uniforme initiale (ln(2) + eps) : {init_cost:.6f}")
    print(
        f"  - Coûts appris : min={np.min(uc):.6f}, mean={np.mean(uc):.6f}, "
        f"std={np.std(uc):.6f}, max={np.max(uc):.6f}"
    )
    print(
        f"  - Coût moyen par site (flip bit 0..{n_sites-1}) : "
        + ", ".join(f"S{a}={np.mean(cost_table[:, a]):.4f}" for a in range(n_sites))
    )
    print(f"  - Table (2^N, N) exportée   : {table_csv.resolve()}")
    print(f"  - Matrice S* (2^N, 2^N)     : {s_star_csv.resolve()}")
    return table_csv, s_star_csv


def export_xproj_diagnostic(
    gram: np.ndarray,
    param_name: str,
    out_dir: Path,
) -> Path:
    """Exporte séparément le Gramien absolu |W^T W| de x_proj.weight comme diagnostic de projection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_csv = out_dir / "diagnostic_similarite_projection_x_proj.csv"
    n_cols = gram.shape[0]
    with open(diag_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["col"] + [f"col_{j}" for j in range(n_cols)])
        for i in range(n_cols):
            w.writerow([f"col_{i}"] + [f"{gram[i, j]:.6f}" for j in range(n_cols)])
    print(f"  - Diagnostic projection ({param_name}) : {diag_csv.resolve()}")
    return diag_csv


def generate_figures(
    bundles: list[dict[str, Any]],
    out_dir: Path,
    target_eta: float,
    geom: dict[str, np.ndarray] | None,
    geom_run_label: str,
    geom_criterion: str,
    edge_eps: float,
) -> list[Path]:
    """Génère les figures PNG corrigées."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[INFO] Installe matplotlib pour tracer les PNG : pip install matplotlib")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
        }
    )
    created: list[Path] = []

    eta_runs = [
        b
        for b in bundles
        if b["eta"] is not None and math.isclose(b["eta"], target_eta, rel_tol=1e-5, abs_tol=1e-7)
    ]
    if not eta_runs and bundles:
        etas = [b["eta"] for b in bundles if b["eta"] is not None]
        if etas:
            target_eta = statistics.mode(etas)
            eta_runs = [
                b
                for b in bundles
                if b["eta"] is not None and math.isclose(b["eta"], target_eta, rel_tol=1e-5, abs_tol=1e-7)
            ]

    if eta_runs:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for b in eta_runs:
            groups.setdefault((b["arch_mode"], b["geometry_mode"]), []).append(b)

        rows = []
        for (arch, geom_m), items in groups.items():
            om_m, om_s, n_runs = _mean_std([x["omega_mse"] for x in items])
            tm_m, tm_s, _ = _mean_std([x["term_mse"] for x in items])
            cs_m, cs_s, _ = _mean_std([x["cos_nonzero"] for x in items])
            unique_seeds = sorted({x["seed"] for x in items if x["seed"] is not None})
            n_seeds = len(unique_seeds)
            seed_txt = f"{n_runs} runs, {n_seeds} seeds" if n_seeds > 0 else f"{n_runs} runs"
            if om_m is not None:
                rows.append(
                    {
                        "key": (arch, geom_m),
                        "label": f"{LABELS.get((arch, geom_m), f'{arch} ({geom_m})')}\n({seed_txt})",
                        "om_m": om_m,
                        "om_s": om_s or 0.0,
                        "tm_m": tm_m or 0.0,
                        "tm_s": tm_s or 0.0,
                        "cs_m": cs_m if cs_m is not None else float("nan"),
                        "cs_s": cs_s or 0.0,
                    }
                )

        rows.sort(key=lambda r: r["om_m"])
        if rows:
            fig, axes = plt.subplots(1, 3, figsize=(16.5, max(4.6, 0.9 * len(rows) + 1.6)))
            y = np.arange(len(rows))
            cols = [COLORS.get(r["key"], "#1f77b4") for r in rows]

            vals_om = [r["om_m"] for r in rows]
            errs_om = [r["om_s"] for r in rows]
            bars1 = axes[0].barh(y, vals_om, xerr=errs_om, color=cols, alpha=0.9, capsize=4)
            axes[0].set_yticks(y)
            axes[0].set_yticklabels([r["label"] for r in rows])
            axes[0].invert_yaxis()
            axes[0].set_title(f"Omega MSE (η = {target_eta:.2f})", fontweight="bold")
            axes[0].set_xlim(0, max(vals_om) * 1.32)
            for bar, v in zip(bars1, vals_om):
                axes[0].text(
                    v + max(vals_om) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.6f}",
                    va="center",
                    fontsize=8.5,
                )

            vals_tm = [r["tm_m"] for r in rows]
            errs_tm = [r["tm_s"] for r in rows]
            bars2 = axes[1].barh(y, vals_tm, xerr=errs_tm, color=cols, alpha=0.9, capsize=4)
            axes[1].set_yticks(y)
            axes[1].set_yticklabels([])
            axes[1].invert_yaxis()
            axes[1].set_title(f"Terminal MSE (η = {target_eta:.2f})", fontweight="bold")
            axes[1].set_xlim(0, max(vals_tm) * 1.32)
            for bar, v in zip(bars2, vals_tm):
                axes[1].text(
                    v + max(vals_tm) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.6f}",
                    va="center",
                    fontsize=8.5,
                )

            vals_cs = [r["cs_m"] for r in rows]
            errs_cs = [r["cs_s"] for r in rows]
            bars3 = axes[2].barh(y, vals_cs, xerr=errs_cs, color=cols, alpha=0.9, capsize=4)
            axes[2].axvline(0.0, color="#333333", linewidth=1.1, linestyle="-")
            axes[2].set_yticks(y)
            axes[2].set_yticklabels([])
            axes[2].invert_yaxis()
            axes[2].set_title(f"Alignement Cosinus ∈ [-1, 1] (η = {target_eta:.2f})", fontweight="bold")
            axes[2].set_xlim(-1.05, 1.18)
            for bar, v in zip(bars3, vals_cs):
                if math.isfinite(v):
                    offset = 0.03 if v >= 0 else -0.03
                    ha = "left" if v >= 0 else "right"
                    axes[2].text(
                        v + offset,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.4f}",
                        va="center",
                        ha=ha,
                        fontsize=8.5,
                    )

            fig.suptitle("Comparaison Homogène NK-LSTM-GSA V2", fontsize=12.5, fontweight="bold")
            fig.tight_layout()
            p1 = out_dir / "fig1_ablations_multiseed_v2.png"
            fig.savefig(p1, dpi=180, bbox_inches="tight")
            plt.close(fig)
            created.append(p1)

    distinct_etas = sorted({round(b["eta"], 6) for b in bundles if b["eta"] is not None})
    if len(distinct_etas) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
        for arch, geom_m in sorted({(b["arch_mode"], b["geometry_mode"]) for b in bundles}):
            xs, om_m, om_s, tm_m, tm_s, cs_m, cs_s = [], [], [], [], [], [], []
            for ev in distinct_etas:
                sub = [
                    b
                    for b in bundles
                    if b["arch_mode"] == arch
                    and b["geometry_mode"] == geom_m
                    and b["eta"] is not None
                    and math.isclose(b["eta"], ev, rel_tol=1e-5)
                ]
                if not sub:
                    continue
                m1, s1, _ = _mean_std([x["omega_mse"] for x in sub])
                m2, s2, _ = _mean_std([x["term_mse"] for x in sub])
                m3, s3, _ = _mean_std([x["cos_nonzero"] for x in sub])
                if m1 is not None:
                    xs.append(ev)
                    om_m.append(m1)
                    om_s.append(s1 or 0.0)
                    tm_m.append(m2 or 0.0)
                    tm_s.append(s2 or 0.0)
                    cs_m.append(m3 if m3 is not None else float("nan"))
                    cs_s.append(s3 or 0.0)
            if xs:
                c = COLORS.get((arch, geom_m), "#1f77b4")
                lbl = LABELS.get((arch, geom_m), f"{arch} ({geom_m})")
                xa = np.array(xs)
                for ax, ml, sl in [(axes[0], om_m, om_s), (axes[1], tm_m, tm_s), (axes[2], cs_m, cs_s)]:
                    ma, sa = np.array(ml), np.array(sl)
                    ax.plot(xa, ma, marker="o", linewidth=2, color=c, label=lbl)
                    ax.fill_between(xa, ma - sa, ma + sa, color=c, alpha=0.15)
        axes[0].set_title("Omega MSE vs η", fontweight="bold")
        axes[0].legend(fontsize=8)
        axes[1].set_title("Terminal MSE vs η", fontweight="bold")
        axes[2].axhline(0.0, color="#333333", linewidth=1.0)
        axes[2].set_title("Cosinus ∈ [-1, 1] vs η", fontweight="bold")
        axes[2].set_ylim(-1.05, 1.05)
        fig.tight_layout()
        p2 = out_dir / "fig2_balayage_eta_v2.png"
        fig.savefig(p2, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(p2)

    if geom is not None:
        cost_table = geom["cost_table"]
        shortest_paths = geom["shortest_paths"]
        uc = geom["undirected_costs"]
        n_states, n_sites = cost_table.shape
        init_cost = math.log(2.0) + edge_eps

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(16.5, 5.2),
            gridspec_kw={"width_ratios": [1.0, 1.25, 1.05]},
        )

        im1 = axes[0].imshow(cost_table, cmap="viridis", aspect="auto")
        axes[0].set_xticks(range(n_sites))
        axes[0].set_xticklabels([f"Site {a}" for a in range(n_sites)])
        axes[0].set_ylabel(f"État hypercube x ∈ {{0..{n_states - 1}}}")
        axes[0].set_title("Table des Coûts d'Arêtes c_θ(x, a)\n(2^N × N)")
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(shortest_paths, cmap="magma", aspect="equal")
        axes[1].set_xlabel(f"État cible y ∈ {{0..{n_states - 1}}}")
        axes[1].set_ylabel(f"État source x ∈ {{0..{n_states - 1}}}")
        axes[1].set_title("Distances Géodésiques Exactes S*(x, y)\n(Floyd-Warshall sur le N-cube)")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].hist(uc, bins=22, color="#1b9e77", edgecolor="white", alpha=0.9)
        axes[2].axvline(
            init_cost,
            color="#d95f02",
            linestyle="--",
            linewidth=2.0,
            label=f"Init. uniforme ln(2)+ε = {init_cost:.4f}",
        )
        axes[2].axvline(
            float(np.mean(uc)),
            color="#111111",
            linestyle="-",
            linewidth=1.6,
            label=f"Moyenne apprise = {np.mean(uc):.4f}",
        )
        axes[2].set_xlabel("Coût d'arête c_e = softplus(logit_e) + ε")
        axes[2].set_ylabel("Nombre d'arêtes (sur 192)")
        axes[2].set_title("Distribution des Coûts d'Arêtes Appris")
        axes[2].legend(fontsize=8.5)

        fig.suptitle(
            f"Géométrie GSA Apprise ({geom_run_label}) — {geom_criterion}",
            fontsize=11.5,
            fontweight="bold",
        )
        fig.tight_layout()
        p4 = out_dir / "fig4_geometrie_apprise_gsa_v2.png"
        fig.savefig(p4, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(p4)

    print("\nFigures PNG générées :")
    for p in created:
        print(f"  -> {p.resolve()}")
    return created


def run_self_test() -> None:
    """Valide toutes les corrections d'audit sur un environnement simulé."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        camp_dir = root / "campagne_v2"
        camp_dir.mkdir(parents=True, exist_ok=True)
        n_sites = 6
        n_undirected = n_sites * (1 << (n_sites - 1))

        for folder in [camp_dir / "eta0p25_hybrid_concat_learned_s7", root / "v2_512_hybrid_concat"]:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "config.json").write_text(
                json.dumps({"arch_mode": "hybrid_concat", "geometry_mode": "learned", "eta": 0.25, "seed": 7, "n_sites": 6, "n_initial_states": 512, "edge_eps": 1e-4}),
                encoding="utf-8",
            )
            (folder / "test_metrics.json").write_text(
                json.dumps({
                    "arch_mode": "hybrid_concat",
                    "geometry_mode": "learned",
                    "transport_mode": "symmetric",
                    "best_val_loss": 0.0025,
                    "test_losses": {"loss_terminal": 0.0009, "loss_omega": 0.00072},
                    "test_metrics": {"mse": 0.00072, "mean_cosine_nonzero": 0.956},
                    "baseline": {"mse_omega": 0.00865},
                }),
                encoding="utf-8",
            )
            torch.save(
                {
                    "model_state_dict": {
                        "gsa.edge_costs.edge_logits": torch.linspace(-0.5, 0.8, n_undirected),
                        "gsa.x_proj.weight": torch.randn(32, 6),
                    }
                },
                folder / "best.pt",
            )

        gsa_dir = root / "v2_512_gsa_only"
        gsa_dir.mkdir(parents=True, exist_ok=True)
        (gsa_dir / "config.json").write_text(
            json.dumps({"arch_mode": "gsa_only", "geometry_mode": "learned", "eta": 0.25, "seed": 7, "n_sites": 6, "n_initial_states": 512, "edge_eps": 1e-4}),
            encoding="utf-8",
        )
        (gsa_dir / "test_metrics.json").write_text(
            json.dumps({
                "arch_mode": "gsa_only",
                "geometry_mode": "learned",
                "transport_mode": "symmetric",
                "best_val_loss": 0.0040,
                "test_losses": {"loss_terminal": 0.0020, "loss_omega": 0.0018},
                "test_metrics": {"mse": 0.0018, "mean_cosine_nonzero": -0.026106},
                "baseline": {"mse_omega": 0.00865},
            }),
            encoding="utf-8",
        )
        torch.save(
            {"model_state_dict": {"gsa.edge_costs.edge_logits": torch.zeros(n_undirected)}},
            gsa_dir / "best.pt",
        )

        old_dir = root / "v2_old_128"
        old_dir.mkdir(parents=True, exist_ok=True)
        (old_dir / "config.json").write_text(
            json.dumps({"arch_mode": "lstm_only", "geometry_mode": "none", "eta": 0.25, "seed": 7, "n_sites": 6, "n_initial_states": 128}),
            encoding="utf-8",
        )
        (old_dir / "test_metrics.json").write_text(
            json.dumps({
                "arch_mode": "lstm_only",
                "geometry_mode": "none",
                "transport_mode": "symmetric",
                "best_val_loss": 0.0030,
                "test_losses": {"loss_terminal": 0.0015, "loss_omega": 0.0012},
                "test_metrics": {"mse": 0.0012, "mean_cosine_nonzero": 0.91},
                "baseline": {"mse_omega": 0.00865},
            }),
            encoding="utf-8",
        )

        bundles = discover_runs(camp_dir, extra_dir=root)
        assert len(bundles) == 3, f"Attendu 3 runs dédoublonnés, reçu {len(bundles)}"
        cohort = filter_homogeneous_cohort(bundles, target_n_init=512)
        assert len(cohort) == 2, f"Attendu 2 runs homogènes (n_init=512), reçu {len(cohort)}"

        chosen, ckpt, crit = select_checkpoint_blind_to_test(cohort, target_eta=0.25)
        assert chosen is not None and chosen["arch_mode"] == "hybrid_concat"
        assert "best_val_loss" in crit

        geom, _, gram, xname = inspect_checkpoint_geometry(ckpt, n_sites=6, edge_eps=1e-4)
        assert geom is not None
        assert geom["cost_table"].shape == (64, 6)
        assert geom["shortest_paths"].shape == (64, 64)
        assert gram is not None and xname == "gsa.x_proj.weight"

        figs = generate_figures(cohort, root / "figs", 0.25, geom, chosen["dossier"], crit, 1e-4)
        assert len(figs) == 2

    print("SELF-TEST AUDIT COMPLET : OK")


def main() -> None:
    p = argparse.ArgumentParser(description="Visualisation rigoureuse NK-LSTM-GSA V2.")
    p.add_argument("--base-dir", default="runs/campagne_v2", help="Répertoire de campagne.")
    p.add_argument("--extra-dir", default="runs", help="Répertoire complémentaire (ex: runs/v2_*).")
    p.add_argument("--extra-pattern", default="v2_*", help="Motif dans --extra-dir.")
    p.add_argument("--out-dir", default="runs/campagne_v2/figures", help="Dossier de sortie.")
    p.add_argument("--target-eta", type=float, default=0.25, help="Valeur cible de eta.")
    p.add_argument("--n-initial-states", type=int, default=512, help="Cohorte homogène cible (défaut: 512).")
    p.add_argument("--select-seed", type=int, default=None, help="Graine pré-enregistrée pour inspecter best.pt.")
    p.add_argument("--ckpt", default="", help="Chemin explicite vers best.pt (prioritaire).")
    p.add_argument("--self-test", action="store_true", help="Valide le script sur un jeu simulé.")
    args = p.parse_args()

    if args.self_test:
        run_self_test()
        return

    bundles = discover_runs(
        base_dir=Path(args.base_dir),
        extra_dir=Path(args.extra_dir) if args.extra_dir else None,
        extra_pattern=args.extra_pattern,
    )
    if not bundles:
        print("Aucun run symétrique trouvé.")
        return

    cohort = filter_homogeneous_cohort(bundles, target_n_init=args.n_initial_states)
    if not cohort:
        print("Aucun run homogène retenu après filtrage.")
        return

    out_dir = Path(args.out_dir)
    chosen, ckpt_path, criterion = select_checkpoint_blind_to_test(
        cohort,
        target_eta=args.target_eta,
        ckpt_override=args.ckpt,
        select_seed=args.select_seed,
    )

    geom: dict[str, np.ndarray] | None = None
    run_label = ckpt_path.parent.name if ckpt_path else "inconnu"
    edge_eps = 1e-4

    if ckpt_path is not None and ckpt_path.exists():
        n_sites = chosen["n_sites"] if chosen else 6
        edge_eps = chosen["edge_eps"] if chosen else 1e-4
        run_label = chosen["dossier"] if chosen else ckpt_path.parent.name
        print(f"\nCheckpoint sélectionné ({criterion}) : {ckpt_path.resolve()}")
        geom, geom_src, xproj_gram, xproj_name = inspect_checkpoint_geometry(
            ckpt_path, n_sites=n_sites, edge_eps=edge_eps
        )
        if geom is not None:
            print(f"Extraction géométrique réussie via {geom_src}")
            export_gsa_geometry_artifacts(geom, out_dir, n_sites=n_sites, edge_eps=edge_eps)
        else:
            print(f"[AVERTISSEMENT] {geom_src}")

        if xproj_gram is not None and xproj_name is not None:
            export_xproj_diagnostic(xproj_gram, xproj_name, out_dir)

    generate_figures(
        bundles=cohort,
        out_dir=out_dir,
        target_eta=args.target_eta,
        geom=geom,
        geom_run_label=run_label,
        geom_criterion=criterion,
        edge_eps=edge_eps,
    )


if __name__ == "__main__":
    main()
