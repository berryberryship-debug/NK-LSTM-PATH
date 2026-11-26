#!/usr/bin/env python3
"""
lancer_campagne_v2.py

Orchestrateur autonome pour les expériences multi-graines et le balayage en eta
de `train_nk_lstm_gsa_v2.py`, avec reprise automatique et calcul des moyennes
et écarts-types par architecture.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ABLATION_SPECS: list[tuple[str, str]] = [
    ("hybrid_concat", "learned"),
    ("hybrid_concat", "fixed"),
    ("hybrid_concat", "none"),
    ("gsa_only", "learned"),
    ("gsa_only", "fixed"),
    ("lstm_only", "none"),
]

SWEEP_SPECS: list[tuple[str, str]] = [
    ("hybrid_concat", "learned"),
    ("gsa_only", "learned"),
    ("lstm_only", "none"),
]


def _to_finite_float(val: Any) -> float | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        fval = float(val)
        return fval if math.isfinite(fval) else None
    if isinstance(val, str):
        try:
            fval = float(val)
            return fval if math.isfinite(fval) else None
        except ValueError:
            return None
    return None


def _eta_tag(eta: float) -> str:
    return f"{eta:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _run_dir_name(arch_mode: str, geometry_mode: str, eta: float, seed: int) -> str:
    return f"eta{_eta_tag(eta)}_{arch_mode}_{geometry_mode}_s{seed}"


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None, int]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    n = len(clean)
    if n == 0:
        return None, None, 0
    if n == 1:
        return clean[0], 0.0, 1
    return statistics.mean(clean), statistics.stdev(clean), n


def _fmt_ms(
    mean_val: float | None,
    std_val: float | None,
    n: int,
    decimals: int = 6,
    is_pct: bool = False,
) -> str:
    if mean_val is None or not math.isfinite(mean_val):
        return "-"
    if is_pct:
        if n >= 2 and std_val is not None:
            return f"{mean_val:+.2f}% ± {std_val:.2f}%"
        return f"{mean_val:+.2f}%"
    if n >= 2 and std_val is not None:
        return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"
    return f"{mean_val:.{decimals}f}"


def load_single_run(run_dir: Path) -> dict[str, Any] | None:
    """Charge les métriques de test d'un run terminé."""
    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cfg: dict[str, Any] = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    test_losses = data.get("test_losses", {})
    test_metrics = data.get("test_metrics", {})
    baseline = data.get("baseline", {})
    vs_base = data.get("model_vs_baseline", {})

    omega_mse = _to_finite_float(test_metrics.get("mse"))
    if omega_mse is None:
        omega_mse = _to_finite_float(test_losses.get("loss_omega"))

    base_omega = _to_finite_float(baseline.get("mse_omega"))
    if base_omega is None:
        base_omega = _to_finite_float(test_metrics.get("baseline_mse_omega"))

    delta_omega = _to_finite_float(vs_base.get("delta_mse_omega"))
    if delta_omega is None:
        delta_omega = _to_finite_float(test_metrics.get("delta_mse_omega_vs_baseline"))
    if delta_omega is None and omega_mse is not None and base_omega is not None:
        delta_omega = omega_mse - base_omega

    gain_pct: float | None = None
    if base_omega is not None and base_omega > 1e-15 and omega_mse is not None:
        gain_pct = 100.0 * (base_omega - omega_mse) / base_omega

    params_raw = data.get("active_trainable_parameters")
    params = int(params_raw) if isinstance(params_raw, (int, float)) else None

    return {
        "dossier": run_dir.name,
        "arch_mode": str(data.get("arch_mode", cfg.get("arch_mode", "?"))),
        "geometry_mode": str(data.get("geometry_mode", cfg.get("geometry_mode", "?"))),
        "transport_mode": str(data.get("transport_mode", cfg.get("transport_mode", "?"))),
        "eta": _to_finite_float(cfg.get("eta")),
        "seed": int(cfg["seed"]) if "seed" in cfg and cfg["seed"] is not None else None,
        "params": params,
        "best_val": _to_finite_float(data.get("best_val_loss")),
        "term_mse": _to_finite_float(test_losses.get("loss_terminal")),
        "base_term": _to_finite_float(baseline.get("mse_terminal")),
        "omega_mse": omega_mse,
        "base_omega": base_omega,
        "delta_omega": delta_omega,
        "gain_pct": gain_pct,
        "rel_norm_err": _to_finite_float(test_metrics.get("mean_relative_norm_error_nonzero")),
        "cos_nonzero": _to_finite_float(test_metrics.get("mean_cosine_nonzero")),
    }


def build_plan(
    mode: str,
    seeds: list[int],
    ablation_eta: float,
    etas: list[float],
) -> list[dict[str, Any]]:
    """Construit la liste dédoublonnée des runs à exécuter."""
    tasks_by_key: dict[tuple[str, str, float, int], dict[str, Any]] = {}

    if mode in ("all", "ablations"):
        for arch_mode, geometry_mode in ABLATION_SPECS:
            for seed in seeds:
                key = (arch_mode, geometry_mode, round(ablation_eta, 8), seed)
                tasks_by_key[key] = {
                    "arch_mode": arch_mode,
                    "geometry_mode": geometry_mode,
                    "eta": ablation_eta,
                    "seed": seed,
                    "dir_name": _run_dir_name(arch_mode, geometry_mode, ablation_eta, seed),
                }

    if mode in ("all", "eta_sweep"):
        for eta in etas:
            for arch_mode, geometry_mode in SWEEP_SPECS:
                for seed in seeds:
                    key = (arch_mode, geometry_mode, round(eta, 8), seed)
                    if key not in tasks_by_key:
                        tasks_by_key[key] = {
                            "arch_mode": arch_mode,
                            "geometry_mode": geometry_mode,
                            "eta": eta,
                            "seed": seed,
                            "dir_name": _run_dir_name(arch_mode, geometry_mode, eta, seed),
                        }

    return list(tasks_by_key.values())


def execute_plan(
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    base_dir: Path,
) -> None:
    """Exécute séquentiellement les entraînements manquants avec reprise automatique."""
    train_script = Path(args.train_script)
    if not args.dry_run and not train_script.exists():
        raise FileNotFoundError(f"Script d'entraînement introuvable : {train_script.resolve()}")

    total = len(tasks)
    print()
    print("=" * 88)
    print(f"PLAN D'EXÉCUTION CAMPAGNE V2 ({total} combinaisons uniques)")
    print("=" * 88)

    for idx, task in enumerate(tasks, start=1):
        run_dir = base_dir / task["dir_name"]
        metrics_file = run_dir / "test_metrics.json"

        cmd = [
            sys.executable,
            str(train_script),
            "--arch-mode",
            task["arch_mode"],
            "--geometry-mode",
            task["geometry_mode"],
            "--transport-mode",
            "symmetric",
            "--n-sites",
            str(args.n_sites),
            "--n-initial-states",
            str(args.n_initial_states),
            "--eta",
            str(task["eta"]),
            "--dropout",
            str(args.dropout),
            "--weight-decay",
            str(args.weight_decay),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(task["seed"]),
            "--output-dir",
            str(run_dir),
        ]

        if metrics_file.exists() and not args.force:
            print(f"[{idx:02d}/{total:02d}] DÉJÀ TERMINÉ (ignoré) : {task['dir_name']}")
            continue

        if args.dry_run:
            print(f"[{idx:02d}/{total:02d}] DRY-RUN : {' '.join(cmd)}")
            continue

        print()
        print("-" * 88)
        print(
            f"[{idx:02d}/{total:02d}] LANCEMENT : arch={task['arch_mode']} | "
            f"geom={task['geometry_mode']} | eta={task['eta']} | seed={task['seed']}"
        )
        print("-" * 88)
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        elapsed = time.perf_counter() - t0
        print(f"[{idx:02d}/{total:02d}] TERMINÉ en {elapsed:.1f} s -> {run_dir}")


def aggregate_records(
    records: list[dict[str, Any]],
    specs: list[tuple[str, str]],
    target_etas: list[float],
) -> list[dict[str, Any]]:
    """Agrège les runs par (eta, arch_mode, geometry_mode) en moyenne et écart-type."""
    summary_rows: list[dict[str, Any]] = []

    for eta in target_etas:
        for arch_mode, geometry_mode in specs:
            matched = [
                r
                for r in records
                if r["arch_mode"] == arch_mode
                and r["geometry_mode"] == geometry_mode
                and r["eta"] is not None
                and math.isclose(r["eta"], eta, rel_tol=1e-6, abs_tol=1e-8)
            ]
            if not matched:
                continue

            matched.sort(key=lambda x: -1 if x["seed"] is None else x["seed"])
            seeds_found = [str(r["seed"]) for r in matched if r["seed"] is not None]
            params = matched[0]["params"]

            term_m, term_s, n_runs = _mean_std([r["term_mse"] for r in matched])
            omega_m, omega_s, _ = _mean_std([r["omega_mse"] for r in matched])
            base_m, base_s, _ = _mean_std([r["base_omega"] for r in matched])
            delta_m, delta_s, _ = _mean_std([r["delta_omega"] for r in matched])
            gain_m, gain_s, _ = _mean_std([r["gain_pct"] for r in matched])
            rel_m, rel_s, _ = _mean_std([r["rel_norm_err"] for r in matched])
            cos_m, cos_s, _ = _mean_std([r["cos_nonzero"] for r in matched])

            summary_rows.append(
                {
                    "eta": eta,
                    "arch_mode": arch_mode,
                    "geometry_mode": geometry_mode,
                    "n_seeds": n_runs,
                    "seeds": ",".join(seeds_found),
                    "params": params,
                    "term_mse_mean": term_m,
                    "term_mse_std": term_s,
                    "omega_mse_mean": omega_m,
                    "omega_mse_std": omega_s,
                    "base_omega_mean": base_m,
                    "base_omega_std": base_s,
                    "delta_omega_mean": delta_m,
                    "delta_omega_std": delta_s,
                    "gain_pct_mean": gain_m,
                    "gain_pct_std": gain_s,
                    "rel_err_mean": rel_m,
                    "rel_err_std": rel_s,
                    "cos_mean": cos_m,
                    "cos_std": cos_s,
                }
            )

    summary_rows.sort(
        key=lambda r: (
            r["eta"],
            float("inf") if r["omega_mse_mean"] is None else r["omega_mse_mean"],
            r["arch_mode"],
            r["geometry_mode"],
        )
    )
    return summary_rows


def render_summary_table(rows: list[dict[str, Any]]) -> str:
    """Formate un tableau texte aligné avec moyenne ± écart-type."""
    columns: list[tuple[str, str, bool]] = [
        ("Eta", "eta_str", True),
        ("Architecture", "arch_mode", False),
        ("Géométrie", "geometry_mode", False),
        ("Seeds", "n_seeds_str", True),
        ("Params", "params_str", True),
        ("Term MSE (moy ± std)", "term_str", True),
        ("Omega MSE (moy ± std)", "omega_str", True),
        ("Gain % (moy ± std)", "gain_str", True),
        ("Err Rel (moy ± std)", "rel_str", True),
        ("Cosinus (moy ± std)", "cos_str", True),
    ]

    formatted: list[dict[str, str]] = []
    for r in rows:
        n = r["n_seeds"]
        formatted.append(
            {
                "eta_str": f"{r['eta']:.2f}",
                "arch_mode": r["arch_mode"],
                "geometry_mode": r["geometry_mode"],
                "n_seeds_str": f"{n} ({r['seeds']})" if r["seeds"] else str(n),
                "params_str": f"{r['params']:,}" if r["params"] is not None else "-",
                "term_str": _fmt_ms(r["term_mse_mean"], r["term_mse_std"], n, 6),
                "omega_str": _fmt_ms(r["omega_mse_mean"], r["omega_mse_std"], n, 6),
                "gain_str": _fmt_ms(r["gain_pct_mean"], r["gain_pct_std"], n, 2, is_pct=True),
                "rel_str": _fmt_ms(r["rel_err_mean"], r["rel_err_std"], n, 4),
                "cos_str": _fmt_ms(r["cos_mean"], r["cos_std"], n, 4),
            }
        )

    widths: dict[str, int] = {}
    for header, key, _ in columns:
        max_cell = max((len(row[key]) for row in formatted), default=0)
        widths[key] = max(len(header), max_cell)

    sep = "+-" + "-+-".join("-" * widths[key] for _, key, _ in columns) + "-+"
    header_line = (
        "| "
        + " | ".join(
            header.rjust(widths[key]) if align_right else header.ljust(widths[key])
            for header, key, align_right in columns
        )
        + " |"
    )

    lines = [sep, header_line, sep]
    prev_eta: str | None = None
    for idx, row in enumerate(formatted):
        if prev_eta is not None and row["eta_str"] != prev_eta:
            lines.append(sep)
        prev_eta = row["eta_str"]

        lines.append(
            "| "
            + " | ".join(
                row[key].rjust(widths[key]) if align_right else row[key].ljust(widths[key])
                for _, key, align_right in columns
            )
            + " |"
        )
        if idx == len(formatted) - 1:
            lines.append(sep)

    return "\n".join(lines)


def write_summary_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "eta",
        "arch_mode",
        "geometry_mode",
        "n_seeds",
        "seeds",
        "params",
        "term_mse_mean",
        "term_mse_std",
        "omega_mse_mean",
        "omega_mse_std",
        "base_omega_mean",
        "base_omega_std",
        "delta_omega_mean",
        "delta_omega_std",
        "gain_pct_mean",
        "gain_pct_std",
        "rel_err_mean",
        "rel_err_std",
        "cos_mean",
        "cos_std",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})


def write_raw_csv(records: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dossier",
        "eta",
        "arch_mode",
        "geometry_mode",
        "seed",
        "params",
        "best_val",
        "term_mse",
        "base_term",
        "omega_mse",
        "base_omega",
        "delta_omega",
        "gain_pct",
        "rel_norm_err",
        "cos_nonzero",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in fieldnames})


def collect_and_report(
    base_dir: Path,
    ablation_eta: float,
    etas: list[float],
) -> None:
    """Relit tous les runs disponibles dans base_dir et génère tableaux + CSV."""
    if not base_dir.exists():
        print(f"Aucun dossier trouvé : {base_dir}")
        return

    records: list[dict[str, Any]] = []
    for sub in sorted(base_dir.iterdir()):
        if sub.is_dir():
            rec = load_single_run(sub)
            if rec is not None:
                records.append(rec)

    if not records:
        print(f"Aucun fichier test_metrics.json trouvé dans {base_dir}.")
        return

    raw_csv = base_dir / "runs_bruts_campagne_v2.csv"
    write_raw_csv(records, raw_csv)

    ablation_rows = aggregate_records(records, ABLATION_SPECS, [ablation_eta])
    if ablation_rows:
        print()
        print("=" * 116)
        print(f"SYNTHÈSE MULTI-GRAINES DES ABLATIONS (eta = {ablation_eta:.2f})")
        print("=" * 116)
        print(render_summary_table(ablation_rows))
        abl_csv = base_dir / "synthese_ablations_multiseed_v2.csv"
        write_summary_csv(ablation_rows, abl_csv)
        print(f"CSV ablations : {abl_csv.resolve()}")

    sweep_rows = aggregate_records(records, SWEEP_SPECS, etas)
    if sweep_rows:
        print()
        print("=" * 116)
        print("SYNTHÈSE DU BALAYAGE EN ETA (MOYENNE ± ÉCART-TYPE PAR ARCHITECTURE)")
        print("=" * 116)
        print(render_summary_table(sweep_rows))
        sweep_csv = base_dir / "synthese_balayage_eta_v2.csv"
        write_summary_csv(sweep_rows, sweep_csv)
        print(f"CSV balayage eta : {sweep_csv.resolve()}")

    print(f"CSV runs bruts   : {raw_csv.resolve()}")


def run_self_test() -> None:
    """Valide le calcul des moyennes/écarts-types, la déduplication et les exports CSV."""
    plan = build_plan("all", seeds=[7, 17, 27], ablation_eta=0.25, etas=[0.05, 0.15, 0.25, 0.40])
    assert len(plan) == 45, f"Attendu 45 runs dédoublonnés, reçu {len(plan)}"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        samples = [
            ("hybrid_concat", "learned", 0.25, 7, 0.00070, 0.956),
            ("hybrid_concat", "learned", 0.25, 17, 0.00074, 0.952),
            ("hybrid_concat", "learned", 0.25, 27, 0.00072, 0.954),
            ("lstm_only", "none", 0.25, 7, 0.00100, 0.938),
            ("lstm_only", "none", 0.25, 17, 0.00106, 0.934),
            ("lstm_only", "none", 0.25, 27, 0.00103, 0.936),
        ]
        for arch, geom, eta, seed, om_mse, cos_v in samples:
            d = root / _run_dir_name(arch, geom, eta, seed)
            d.mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text(
                json.dumps({"arch_mode": arch, "geometry_mode": geom, "eta": eta, "seed": seed}),
                encoding="utf-8",
            )
            (d / "test_metrics.json").write_text(
                json.dumps(
                    {
                        "arch_mode": arch,
                        "geometry_mode": geom,
                        "transport_mode": "symmetric",
                        "active_trainable_parameters": 188623 if arch == "hybrid_concat" else 75919,
                        "best_val_loss": 0.002,
                        "test_losses": {"loss_terminal": 0.0009, "loss_omega": om_mse},
                        "test_metrics": {
                            "mse": om_mse,
                            "mean_relative_norm_error_nonzero": 0.31,
                            "mean_cosine_nonzero": cos_v,
                        },
                        "baseline": {"mse_terminal": 0.012, "mse_omega": 0.00865},
                        "model_vs_baseline": {"delta_mse_omega": om_mse - 0.00865},
                    }
                ),
                encoding="utf-8",
            )

        recs = [load_single_run(p) for p in root.iterdir() if p.is_dir()]
        recs_clean = [r for r in recs if r is not None]
        agg = aggregate_records(recs_clean, ABLATION_SPECS, [0.25])
        assert len(agg) == 2
        assert agg[0]["arch_mode"] == "hybrid_concat"
        assert agg[0]["n_seeds"] == 3
        assert math.isclose(agg[0]["omega_mse_mean"], 0.00072, rel_tol=1e-6)
        assert math.isclose(agg[0]["omega_mse_std"], 0.00002, rel_tol=1e-6)

        table = render_summary_table(agg)
        assert "0.000720 ± 0.000020" in table

        csv_test = root / "test_agg.csv"
        write_summary_csv(agg, csv_test)
        assert csv_test.exists()

    print("SELF-TEST CAMPAGNE MULTI-GRAINES & ETA : OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrateur multi-graines et balayage en eta pour train_nk_lstm_gsa_v2.py."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "ablations", "eta_sweep", "aggregate_only"],
        default="all",
        help="Sous-ensemble à exécuter (défaut: all).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 17, 27],
        help="Liste des graines aléatoires (défaut: 7 17 27).",
    )
    parser.add_argument(
        "--ablation-eta",
        type=float,
        default=0.25,
        help="Valeur de eta utilisée pour la matrice complète d'ablation (défaut: 0.25).",
    )
    parser.add_argument(
        "--etas",
        type=float,
        nargs="+",
        default=[0.05, 0.15, 0.25, 0.40],
        help="Grille de valeurs de eta pour le balayage (défaut: 0.05 0.15 0.25 0.40).",
    )
    parser.add_argument("--n-sites", type=int, default=6)
    parser.add_argument("--n-initial-states", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--train-script",
        type=str,
        default="train_nk_lstm_gsa_v2.py",
        help="Chemin vers train_nk_lstm_gsa_v2.py.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="runs/campagne_v2",
        help="Dossier racine des sorties de la campagne (défaut: runs/campagne_v2).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ré-exécuter même si test_metrics.json existe déjà.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les commandes sans lancer d'entraînement.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Valide le script sur des runs simulés.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        run_self_test()
        return

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.mode != "aggregate_only":
        tasks = build_plan(
            mode=args.mode,
            seeds=args.seeds,
            ablation_eta=args.ablation_eta,
            etas=args.etas,
        )
        try:
            execute_plan(tasks, args, base_dir)
        except KeyboardInterrupt:
            print("\n[INTERRUPTION] Arrêt demandé par l'utilisateur. Agrégation des runs déjà terminés...")

    if not args.dry_run:
        collect_and_report(
            base_dir=base_dir,
            ablation_eta=args.ablation_eta,
            etas=args.etas,
        )


if __name__ == "__main__":
    main()
