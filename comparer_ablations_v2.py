#!/usr/bin/env python3
"""
comparer_ablations_v2.py

Script autonome pour lire, agréger et comparer sous forme de tableau
les fichiers `test_metrics.json` produits par `train_nk_lstm_gsa_v2.py`
dans les sous-dossiers `runs/v2_*`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any


def _pick_metric(mapping: dict[str, Any], candidates: tuple[str, ...]) -> float | None:
    """Extrait la première métrique numérique finie correspondant aux clés candidates."""
    if not isinstance(mapping, dict):
        return None
    lower_map = {str(k).lower(): v for k, v in mapping.items()}
    for key in candidates:
        val = mapping.get(key, lower_map.get(key.lower()))
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            fval = float(val)
            if math.isfinite(fval):
                return fval
    return None


def _fmt_float(val: float | None, decimals: int = 6) -> str:
    if val is None or not math.isfinite(val):
        return "-"
    return f"{val:.{decimals}f}"


def _fmt_pct(val: float | None) -> str:
    if val is None or not math.isfinite(val):
        return "-"
    return f"{val:+.2f}%"


def _fmt_int(val: int | None) -> str:
    if val is None:
        return "-"
    return f"{val:,}"


def load_run_record(metrics_path: Path) -> dict[str, Any]:
    """Charge un dossier d'expérience à partir de son fichier test_metrics.json."""
    run_dir = metrics_path.parent
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cfg: dict[str, Any] = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    epochs_ran: int | None = None
    history_path = run_dir / "history.json"
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            hist = json.load(f)
            if isinstance(hist, list):
                epochs_ran = len(hist)

    test_losses = data.get("test_losses", {})
    test_metrics = data.get("test_metrics", {})
    baseline = data.get("baseline", {})
    vs_base = data.get("model_vs_baseline", {})

    omega_mse = _pick_metric(test_metrics, ("mse", "omega_mse", "mse_omega"))
    if omega_mse is None:
        omega_mse = _pick_metric(test_losses, ("loss_omega",))

    base_omega = _pick_metric(baseline, ("mse_omega",))
    if base_omega is None:
        base_omega = _pick_metric(test_metrics, ("baseline_mse_omega",))

    delta_omega = _pick_metric(vs_base, ("delta_mse_omega",))
    if delta_omega is None:
        delta_omega = _pick_metric(test_metrics, ("delta_mse_omega_vs_baseline",))
    if delta_omega is None and omega_mse is not None and base_omega is not None:
        delta_omega = omega_mse - base_omega

    gain_pct: float | None = None
    if base_omega is not None and base_omega > 1e-15 and omega_mse is not None:
        gain_pct = 100.0 * (base_omega - omega_mse) / base_omega

    params_raw = data.get("active_trainable_parameters")
    params = int(params_raw) if isinstance(params_raw, (int, float)) else None

    eta_raw = cfg.get("eta")
    eta = float(eta_raw) if isinstance(eta_raw, (int, float)) else None

    seed_raw = cfg.get("seed")
    seed = int(seed_raw) if isinstance(seed_raw, (int, float)) else None

    return {
        "dossier": run_dir.name,
        "arch_mode": str(data.get("arch_mode", cfg.get("arch_mode", "?"))),
        "geometry_mode": str(data.get("geometry_mode", cfg.get("geometry_mode", "?"))),
        "transport_mode": str(data.get("transport_mode", cfg.get("transport_mode", "?"))),
        "eta": eta,
        "seed": seed,
        "params": params,
        "epochs": epochs_ran,
        "best_val": _pick_metric(data, ("best_val_loss",)),
        "test_total": _pick_metric(test_losses, ("loss_total",)),
        "test_term": _pick_metric(test_losses, ("loss_terminal",)),
        "base_term": _pick_metric(baseline, ("mse_terminal",))
        or _pick_metric(test_metrics, ("baseline_mse_terminal",)),
        "omega_mse": omega_mse,
        "base_omega": base_omega,
        "delta_omega": delta_omega,
        "gain_pct": gain_pct,
        "omega_mae": _pick_metric(test_metrics, ("mae", "omega_mae")),
        "r2": _pick_metric(test_metrics, ("r2", "r2_score", "r_squared")),
        "cos_sim": _pick_metric(test_metrics, ("cosine_similarity", "cosine", "cos_sim")),
    }


def collect_runs(runs_dir: Path, pattern: str) -> list[dict[str, Any]]:
    """Collecte et trie tous les fichiers test_metrics.json correspondant au motif."""
    if not runs_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for metrics_file in sorted(runs_dir.glob(f"{pattern}/test_metrics.json")):
        try:
            records.append(load_run_record(metrics_file))
        except Exception as exc:
            print(f"[AVERTISSEMENT] Lecture ignorée pour {metrics_file} : {exc}", file=sys.stderr)

    records.sort(
        key=lambda r: (
            0 if r["transport_mode"] == "integrable" else 1,
            float("inf") if r["omega_mse"] is None else r["omega_mse"],
            r["dossier"],
        )
    )
    return records


def render_table(records: list[dict[str, Any]]) -> str:
    """Construit un tableau texte aligné pour affichage dans le terminal."""
    columns: list[tuple[str, str, bool]] = [
        ("Dossier", "dossier", False),
        ("Architecture", "arch_mode", False),
        ("Géométrie", "geometry_mode", False),
        ("Transport", "transport_mode", False),
        ("Eta", "eta_str", True),
        ("Params", "params_str", True),
        ("Ep.", "epochs_str", True),
        ("Val Best", "best_val_str", True),
        ("Test Term", "test_term_str", True),
        ("Omega MSE", "omega_mse_str", True),
        ("Base Omega", "base_omega_str", True),
        ("Delta Omega", "delta_omega_str", True),
        ("Gain %", "gain_pct_str", True),
        ("R2", "r2_str", True),
        ("Cosinus", "cos_sim_str", True),
    ]

    formatted_rows: list[dict[str, str]] = []
    for r in records:
        formatted_rows.append(
            {
                "dossier": r["dossier"],
                "arch_mode": r["arch_mode"],
                "geometry_mode": r["geometry_mode"],
                "transport_mode": r["transport_mode"],
                "eta_str": _fmt_float(r["eta"], 2),
                "params_str": _fmt_int(r["params"]),
                "epochs_str": _fmt_int(r["epochs"]),
                "best_val_str": _fmt_float(r["best_val"], 6),
                "test_term_str": _fmt_float(r["test_term"], 6),
                "omega_mse_str": _fmt_float(r["omega_mse"], 6),
                "base_omega_str": _fmt_float(r["base_omega"], 6),
                "delta_omega_str": _fmt_float(r["delta_omega"], 6),
                "gain_pct_str": _fmt_pct(r["gain_pct"]),
                "r2_str": _fmt_float(r["r2"], 4),
                "cos_sim_str": _fmt_float(r["cos_sim"], 4),
            }
        )

    widths: dict[str, int] = {}
    for header, key, _ in columns:
        max_cell = max((len(row[key]) for row in formatted_rows), default=0)
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
    prev_transport: str | None = None
    for idx, row in enumerate(formatted_rows):
        curr_transport = row["transport_mode"]
        if prev_transport is not None and curr_transport != prev_transport:
            lines.append(sep)
        prev_transport = curr_transport

        line = (
            "| "
            + " | ".join(
                row[key].rjust(widths[key]) if align_right else row[key].ljust(widths[key])
                for _, key, align_right in columns
            )
            + " |"
        )
        lines.append(line)
        if idx == len(formatted_rows) - 1:
            lines.append(sep)

    return "\n".join(lines)


def export_csv(records: list[dict[str, Any]], csv_path: Path) -> None:
    """Exporte les métriques agrégées vers un fichier CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dossier",
        "arch_mode",
        "geometry_mode",
        "transport_mode",
        "eta",
        "seed",
        "params",
        "epochs",
        "best_val",
        "test_total",
        "test_term",
        "base_term",
        "omega_mse",
        "base_omega",
        "delta_omega",
        "gain_pct",
        "omega_mae",
        "r2",
        "cos_sim",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in fieldnames})


def run_self_test() -> None:
    """Vérifie le parsing, le tri, le rendu tableau et l'export CSV sur des runs factices."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mock_runs = [
            (
                "v2_control_integrable",
                {
                    "arch_mode": "hybrid_concat",
                    "geometry_mode": "learned",
                    "transport_mode": "integrable",
                    "active_trainable_parameters": 95400,
                    "best_val_loss": 0.00012,
                    "test_losses": {"loss_total": 0.00015, "loss_terminal": 0.00011, "loss_omega": 0.00004},
                    "test_metrics": {"mse": 0.00004, "r2": 1.0, "cosine_similarity": 1.0},
                    "baseline": {"mse_terminal": 0.0012, "mse_omega": 0.0},
                    "model_vs_baseline": {"delta_mse_omega": 0.00004},
                },
                {"eta": 0.25, "seed": 7},
            ),
            (
                "v2_hybrid_concat_learned",
                {
                    "arch_mode": "hybrid_concat",
                    "geometry_mode": "learned",
                    "transport_mode": "symmetric",
                    "active_trainable_parameters": 95400,
                    "best_val_loss": 0.0018,
                    "test_losses": {"loss_total": 0.0021, "loss_terminal": 0.0012, "loss_omega": 0.0009},
                    "test_metrics": {"mse": 0.0009, "r2": 0.9421, "cosine_similarity": 0.9712},
                    "baseline": {"mse_terminal": 0.0085, "mse_omega": 0.0160},
                    "model_vs_baseline": {"delta_mse_omega": -0.0151},
                },
                {"eta": 0.25, "seed": 7},
            ),
            (
                "v2_lstm_only",
                {
                    "arch_mode": "lstm_only",
                    "geometry_mode": "none",
                    "transport_mode": "symmetric",
                    "active_trainable_parameters": 68200,
                    "best_val_loss": 0.0054,
                    "test_losses": {"loss_total": 0.0061, "loss_terminal": 0.0033, "loss_omega": 0.0028},
                    "test_metrics": {"mse": 0.0028, "r2": 0.8210, "cosine_similarity": 0.9055},
                    "baseline": {"mse_terminal": 0.0085, "mse_omega": 0.0160},
                    "model_vs_baseline": {"delta_mse_omega": -0.0132},
                },
                {"eta": 0.25, "seed": 7},
            ),
        ]

        for name, metrics_payload, cfg_payload in mock_runs:
            d = root / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "test_metrics.json").write_text(json.dumps(metrics_payload), encoding="utf-8")
            (d / "config.json").write_text(json.dumps(cfg_payload), encoding="utf-8")
            (d / "history.json").write_text(json.dumps([{"epoch": 1}, {"epoch": 2}]), encoding="utf-8")

        records = collect_runs(root, "v2_*")
        assert len(records) == 3, f"Attendu 3 runs, reçu {len(records)}"
        assert records[0]["dossier"] == "v2_control_integrable"
        assert records[1]["dossier"] == "v2_hybrid_concat_learned"
        assert math.isclose(records[1]["gain_pct"], 94.375, rel_tol=1e-5)

        table = render_table(records)
        assert "v2_hybrid_concat_learned" in table

        csv_out = root / "out.csv"
        export_csv(records, csv_out)
        assert csv_out.exists()

    print("SELF-TEST COMPARATEUR : OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare les fichiers test_metrics.json des dossiers runs/v2_*."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Dossier parent contenant les sous-dossiers d'entraînement (défaut: runs).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="v2_*",
        help="Motif des sous-dossiers à inclure (défaut: v2_*).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Inclure tous les sous-dossiers de --runs-dir (équivalent à --pattern '*').",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="runs/comparaison_ablations_v2.csv",
        help="Chemin d'export CSV (laisser vide '' pour désactiver).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Exécute un test de validation interne sur des runs simulés.",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    runs_dir = Path(args.runs_dir)
    pattern = "*" if args.all else args.pattern
    records = collect_runs(runs_dir, pattern)

    if not records:
        print(
            f"Aucun fichier test_metrics.json trouvé dans '{runs_dir / pattern}'.\n"
            "Lance d'abord les entraînements avec train_nk_lstm_gsa_v2.py."
        )
        return

    print()
    print("=" * 80)
    print(f"COMPARAISON DES ABLATIONS NK-LSTM-GSA V2 ({len(records)} exécutions)")
    print("=" * 80)
    print(render_table(records))

    if args.csv:
        csv_path = Path(args.csv)
        export_csv(records, csv_path)
        print(f"\nTableau CSV enregistré dans : {csv_path.resolve()}")


if __name__ == "__main__":
    main()
