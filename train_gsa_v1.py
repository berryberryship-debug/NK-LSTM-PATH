"""Train and compare GSA-V1 conditions on paired, forced NK-LSTM-PATH paths.

Place this file and gsa_v1_core.py at the repository root, beside
balayage_commutateur_nk_v2.py. Run from that directory:

  python .\\train_gsa_v1.py --n-initial-states 100 --epochs 120 --patience 20

This script does not modify existing project files. It creates a result folder
with JSON metrics and one checkpoint per attention geometry condition.

Scientific scope:
- data are forced two-flip squares (k -> m versus m -> k), not Ogata paths;
- the independent split unit is the initial state;
- test states are unseen, while the six-site pair types are represented in train;
- the integrable transport is a null-control check, not a learned-memory task;
- this first version learns edge costs on the X hypercube, not on full (X,J).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from balayage_commutateur_nk_v2 import (  # noqa: E402
    ExperimentConfig,
    sample_binary_symmetric_states,
)
from gsa_v1_core import (  # noqa: E402
    GSAConfig,
    GSAPathModel,
    build_forced_pair_dataset,
    closure_metrics,
    paired_gsa_loss,
    split_indices_by_initial_state,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    """Convert NumPy/PyTorch values and non-finite floats into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    return value


def dataset_tensors(dataset, device: torch.device) -> dict[str, torch.Tensor]:
    def tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(getattr(dataset, name), dtype=dtype, device=device)

    return {
        "x_a": tensor("x_a", torch.float32),
        "x_b": tensor("x_b", torch.float32),
        "act_a": tensor("act_a", torch.long),
        "act_b": tensor("act_b", torch.long),
        "j0": tensor("j0", torch.float32),
        "j_final_a": tensor("j_final_a", torch.float32),
        "j_final_b": tensor("j_final_b", torch.float32),
        "omega": tensor("omega", torch.float32),
    }


def predict_indices(
    model: GSAPathModel,
    tensors: dict[str, torch.Tensor],
    indices: np.ndarray | torch.Tensor,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    idx = torch.as_tensor(indices, dtype=torch.long, device=tensors["j0"].device)
    pred_a_chunks: list[np.ndarray] = []
    pred_b_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, idx.numel(), batch_size):
            take = idx[start:start + batch_size]
            pred_a = model(
                tensors["x_a"][take], tensors["act_a"][take], tensors["j0"][take]
            )
            pred_b = model(
                tensors["x_b"][take], tensors["act_b"][take], tensors["j0"][take]
            )
            pred_a_chunks.append(pred_a.cpu().numpy())
            pred_b_chunks.append(pred_b.cpu().numpy())
    pred_a = np.concatenate(pred_a_chunks, axis=0)
    pred_b = np.concatenate(pred_b_chunks, axis=0)
    return {"pred_a": pred_a, "pred_b": pred_b, "omega_hat": pred_a - pred_b}


def evaluate_predictions(
    predictions: dict[str, np.ndarray],
    dataset,
    indices: np.ndarray,
) -> dict[str, Any]:
    pred_a = predictions["pred_a"]
    pred_b = predictions["pred_b"]
    omega_hat = predictions["omega_hat"]
    target_a = dataset.j_final_a[indices]
    target_b = dataset.j_final_b[indices]
    omega_true = dataset.omega[indices]
    terminal_mse = 0.5 * (
        float(np.mean((pred_a - target_a) ** 2))
        + float(np.mean((pred_b - target_b) ** 2))
    )
    result = closure_metrics(omega_hat, omega_true)
    result.update({
        "terminal_mse": terminal_mse,
        "mean_true_omega_norm": float(np.mean(np.linalg.norm(omega_true, axis=1))),
        "mean_predicted_omega_norm": float(np.mean(np.linalg.norm(omega_hat, axis=1))),
        "max_abs_omega_component": float(np.max(np.abs(omega_true))) if omega_true.size else 0.0,
    })
    return result


def per_initial_state_mse(
    omega_hat: np.ndarray,
    omega_true: np.ndarray,
    state_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(state_ids)
    means = np.asarray([
        np.mean((omega_hat[state_ids == state] - omega_true[state_ids == state]) ** 2)
        for state in unique
    ], dtype=np.float64)
    return unique, means


def bootstrap_improvement(
    model_mse_by_state: np.ndarray,
    zero_mse_by_state: np.ndarray,
    seed: int,
    n_bootstrap: int = 4000,
    n_comparisons: int = 4,
) -> dict[str, Any]:
    """Cluster bootstrap by initial state; Bonferroni-adjusted interval."""
    if model_mse_by_state.shape != zero_mse_by_state.shape:
        raise ValueError("Les vecteurs de MSE par état doivent avoir la même forme.")
    improvement = zero_mse_by_state - model_mse_by_state
    n = improvement.size
    if n < 2:
        return {
            "n_initial_states": int(n),
            "mean_mse_improvement_vs_zero": float(np.mean(improvement)) if n else None,
            "simultaneous_ci_low": None,
            "simultaneous_ci_high": None,
            "ci_method": "cluster bootstrap unavailable: fewer than 2 test states",
        }
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(improvement[draw], axis=1)
    alpha_each = 0.05 / max(1, n_comparisons)
    low_q = alpha_each / 2.0
    high_q = 1.0 - alpha_each / 2.0
    return {
        "n_initial_states": int(n),
        "mean_mse_improvement_vs_zero": float(np.mean(improvement)),
        "simultaneous_ci_low": float(np.quantile(boot_means, low_q)),
        "simultaneous_ci_high": float(np.quantile(boot_means, high_q)),
        "ci_method": (
            f"paired cluster bootstrap by initial state; {n_bootstrap} draws; "
            f"Bonferroni-adjusted percentile interval over {n_comparisons} comparisons"
        ),
    }


def _ridge_features(dataset, eta: float) -> np.ndarray:
    """Construct fixed, interpretable covariates for the tabular Omega baseline."""
    x0 = dataset.x_a[:, 0, :].astype(np.float64)
    j0 = dataset.j0.astype(np.float64)
    pairs = dataset.site_pair
    n_sites = x0.shape[1]
    pair_lookup = {
        pair: i
        for i, pair in enumerate(
            ( (k, m) for k in range(n_sites) for m in range(k + 1, n_sites) )
        )
    }
    pair_onehot = np.zeros((len(pairs), len(pair_lookup)), dtype=np.float64)
    pair_ids = np.asarray([pair_lookup[tuple(map(int, pair))] for pair in pairs])
    pair_onehot[np.arange(len(pairs)), pair_ids] = 1.0
    upper_i, upper_j = np.triu_indices(n_sites, 1)
    interactions = j0 * (x0[:, upper_i] * x0[:, upper_j])
    eta_col = np.full((len(pairs), 1), float(eta), dtype=np.float64)
    return np.concatenate(
        [x0, j0, pair_onehot, interactions, eta_col, eta_col ** 2], axis=1
    )


def fit_ridge_omega_baseline(
    dataset, train_indices: np.ndarray, alpha: float = 1.0, eta: float = 0.25
):
    """Fit a train-only multi-output ridge baseline for Omega.

    Features: X0, packed J0, site-pair one-hot, all X_i*X_j*J_ij interactions,
    eta and eta^2. This is a deliberately simple tabular comparator, not a
    claimed closed-form solution of psi_sym.
    """
    if alpha <= 0.0:
        raise ValueError("ridge alpha doit être strictement positif.")
    features = _ridge_features(dataset, eta)
    target = dataset.omega.astype(np.float64)
    x_train = features[train_indices]
    y_train = target[train_indices]
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale < 1e-10] = 1.0
    z_train = (x_train - x_mean) / x_scale
    y_mean = y_train.mean(axis=0)
    zt_z = z_train.T @ z_train
    reg = float(alpha) * np.eye(zt_z.shape[0], dtype=np.float64)
    coef = np.linalg.solve(zt_z + reg, z_train.T @ (y_train - y_mean))
    return {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "coef": coef}


def predict_ridge_omega_baseline(
    dataset, fitted: dict[str, np.ndarray], indices: np.ndarray, eta: float = 0.25
) -> np.ndarray:
    features = _ridge_features(dataset, eta)
    z = (features[indices] - fitted["x_mean"]) / fitted["x_scale"]
    return (z @ fitted["coef"] + fitted["y_mean"]).astype(np.float64)


def evaluate_direct_omega(
    omega_hat: np.ndarray,
    dataset,
    indices: np.ndarray,
) -> dict[str, Any]:
    omega_true = dataset.omega[indices]
    result = closure_metrics(omega_hat, omega_true)
    result.update({
        "terminal_mse": None,
        "mean_true_omega_norm": float(np.mean(np.linalg.norm(omega_true, axis=1))),
        "mean_predicted_omega_norm": float(np.mean(np.linalg.norm(omega_hat, axis=1))),
    })
    return result


def train_one_condition(
    mode: str,
    config: GSAConfig,
    tensors: dict[str, torch.Tensor],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dataset,
    args,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    # Resetting the seed gives all conditions the same initialization for shared layers.
    set_seed(args.model_seed)
    model = GSAPathModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_closure = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    train_index_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_index_tensor = torch.as_tensor(val_idx, dtype=torch.long, device=device)

    def validation_metrics() -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            pred = predict_indices(model, tensors, val_idx)
            omega_true = dataset.omega[val_idx]
            terminal = 0.5 * (
                np.mean((pred["pred_a"] - dataset.j_final_a[val_idx]) ** 2)
                + np.mean((pred["pred_b"] - dataset.j_final_b[val_idx]) ** 2)
            )
            closure = np.mean((pred["omega_hat"] - omega_true) ** 2)
        return float(closure), float(terminal)

    initial_closure, initial_terminal = validation_metrics()
    best_val_closure = initial_closure
    history.append({
        "epoch": 0,
        "train_loss": float("nan"),
        "validation_closure_mse": initial_closure,
        "validation_terminal_mse": initial_terminal,
    })

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = train_index_tensor[torch.randperm(train_index_tensor.numel(), device=device)]
        sums = {"loss": 0.0, "terminal": 0.0, "closure": 0.0}
        seen = 0
        for start in range(0, permutation.numel(), args.batch_size):
            batch_ids = permutation[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred_a = model(tensors["x_a"][batch_ids], tensors["act_a"][batch_ids], tensors["j0"][batch_ids])
            pred_b = model(tensors["x_b"][batch_ids], tensors["act_b"][batch_ids], tensors["j0"][batch_ids])
            loss, info = paired_gsa_loss(
                pred_a,
                pred_b,
                tensors["j_final_a"][batch_ids],
                tensors["j_final_b"][batch_ids],
                lambda_omega=args.lambda_omega,
                lambda_geometry=args.lambda_geometry,
                model=model,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Perte non finie à l'époque {epoch}, mode={mode}.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            size = int(batch_ids.numel())
            sums["loss"] += info["loss_total"] * size
            sums["terminal"] += info["loss_terminal"] * size
            sums["closure"] += info["loss_closure"] * size
            seen += size

        val_closure, val_terminal = validation_metrics()
        history.append({
            "epoch": float(epoch),
            "train_loss": sums["loss"] / max(1, seen),
            "train_terminal_mse": sums["terminal"] / max(1, seen),
            "train_closure_mse": sums["closure"] / max(1, seen),
            "validation_closure_mse": val_closure,
            "validation_terminal_mse": val_terminal,
        })

        # Primary selection target is closure MSE, because path-memory is the
        # hypothesis under test. Terminal MSE remains a separate reported metric.
        if val_closure < best_val_closure - args.min_delta:
            best_val_closure = val_closure
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    checkpoint_path = output_dir / f"gsa_v1_{mode}_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "gsa_config": asdict(config),
        "best_epoch": best_epoch,
        "best_validation_closure_mse": best_val_closure,
        "model_seed": args.model_seed,
    }, checkpoint_path)

    val_pred = predict_indices(model, tensors, val_idx)
    test_pred = predict_indices(model, tensors, args.test_indices)
    val_metrics = evaluate_predictions(val_pred, dataset, val_idx)
    test_metrics = evaluate_predictions(test_pred, dataset, args.test_indices)

    return {
        "geometry_mode": mode,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_epoch": int(best_epoch),
        "best_validation_closure_mse": float(best_val_closure),
        "validation": val_metrics,
        "test": test_metrics,
        "checkpoint": str(checkpoint_path.name),
        "history": history,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement comparatif NK-LSTM-PATH GSA-V1.")
    parser.add_argument("--n-initial-states", type=int, default=100)
    parser.add_argument("--init-seed", type=int, default=7)
    parser.add_argument("--model-seed", type=int, default=7)
    parser.add_argument("--n-sites", type=int, default=6)
    parser.add_argument("--scale-j", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-9)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-omega", type=float, default=1.0)
    parser.add_argument("--lambda-geometry", type=float, default=1e-3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--beta-action", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--outdir", type=Path, default=Path("gsa_v1_results"))
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_initial_states < 3:
        raise ValueError("Au moins 3 états initiaux sont requis pour les trois partitions.")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience et batch-size doivent être >= 1.")
    if args.bootstrap_draws < 100:
        raise ValueError("bootstrap-draws doit être >= 100 pour une estimation utile.")
    if args.lr <= 0 or args.grad_clip <= 0 or args.weight_decay < 0:
        raise ValueError("lr et grad-clip doivent être > 0; weight-decay doit être >= 0.")
    if args.min_delta < 0:
        raise ValueError("min-delta doit être >= 0.")
    if args.ridge_alpha <= 0 or args.lambda_omega < 0 or args.lambda_geometry < 0:
        raise ValueError("ridge-alpha doit être > 0; les poids de perte doivent être >= 0.")
    if args.n_sites != 6:
        # This first study is deliberately tied to the N=6 protocol and its 15 J coordinates.
        raise ValueError("Cette première configuration GSA-V1 est fixée à N=6.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé, mais indisponible dans cet environnement PyTorch.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.model_seed)

    project_cfg = ExperimentConfig(
        n_sites=args.n_sites,
        n_initial_states=args.n_initial_states,
        init_seed=args.init_seed,
        scale_J=args.scale_j,
        eta=args.eta,
    )
    rng = np.random.default_rng(project_cfg.init_seed)
    initial_states = sample_binary_symmetric_states(
        n_sites=project_cfg.n_sites,
        n_states=project_cfg.n_initial_states,
        rng=rng,
        scale=project_cfg.scale_J,
    )

    # The actual transport implementation is imported through the user's project module.
    symmetric_data = build_forced_pair_dataset(
        initial_states, eta=project_cfg.eta, mode="symmetric"
    )
    integrable_data = build_forced_pair_dataset(
        initial_states, eta=project_cfg.eta, mode="integrable"
    )
    integrable_max_omega = float(np.max(np.abs(integrable_data.omega)))
    if integrable_max_omega > 1e-12:
        raise RuntimeError(
            "Le contrôle intégrable n'est pas nul : "
            f"max|Omega|={integrable_max_omega:.6e}. Arrêt avant entraînement."
        )

    split = split_indices_by_initial_state(symmetric_data, seed=args.init_seed)
    train_idx = split["train"]
    val_idx = split["validation"]
    test_idx = split["test"]
    args.test_indices = test_idx
    tensors = dataset_tensors(symmetric_data, device)

    n_nonzero = int(np.count_nonzero(np.linalg.norm(symmetric_data.omega, axis=1) > 1e-12))
    if n_nonzero == 0:
        raise RuntimeError("Le transport symétrique ne produit aucune fermeture non nulle.")

    print("\nNK-LSTM-PATH — GSA-V1 TRAIN / VALIDATION / TEST")
    print("=" * 72)
    print(f"N=6 | états initiaux={args.n_initial_states} | paires forcées={symmetric_data.n_pairs}")
    print(f"eta={args.eta:g} | scale_J={args.scale_j:g} | device={device}")
    print(
        f"États train/validation/test : "
        f"{len(np.unique(symmetric_data.initial_state_id[train_idx]))}/"
        f"{len(np.unique(symmetric_data.initial_state_id[val_idx]))}/"
        f"{len(np.unique(symmetric_data.initial_state_id[test_idx]))}"
    )
    print(f"Paires avec Omega != 0 (symétrique) : {n_nonzero}/{symmetric_data.n_pairs}")
    print(f"Contrôle intégrable max|Omega| : {integrable_max_omega:.3e}")
    print("Sélection du meilleur epoch : MSE de fermeture sur validation.")

    # A simple, train-only regression gives a non-neural comparator for Omega.
    ridge = fit_ridge_omega_baseline(
        symmetric_data, train_idx, alpha=args.ridge_alpha, eta=args.eta
    )
    ridge_test_hat = predict_ridge_omega_baseline(
        symmetric_data, ridge, test_idx, eta=args.eta
    )
    ridge_test_metrics = evaluate_direct_omega(ridge_test_hat, symmetric_data, test_idx)

    zero_test_hat = np.zeros_like(symmetric_data.omega[test_idx])
    zero_metrics = evaluate_direct_omega(zero_test_hat, symmetric_data, test_idx)

    model_results: dict[str, Any] = {}
    model_metrics_by_state: dict[str, np.ndarray] = {
        "zero": np.asarray([
            np.mean((symmetric_data.omega[test_idx][symmetric_data.initial_state_id[test_idx] == sid]) ** 2)
            for sid in np.unique(symmetric_data.initial_state_id[test_idx])
        ], dtype=np.float64)
    }

    for mode in ("none", "fixed", "learned"):
        config = GSAConfig(
            n_sites=6,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            beta_action=args.beta_action,
            max_len=3,
            geometry_mode=mode,
        )
        print(f"\nEntraînement : geometry_mode={mode}")
        result = train_one_condition(
            mode, config, tensors, train_idx, val_idx, symmetric_data,
            args, device, args.outdir,
        )
        model_results[mode] = result
        pred_test = predict_indices(
            # Reconstruct the selected model from its checkpoint state for the group analysis.
            _load_model_from_saved(result, config, device, args.outdir),
            tensors,
            test_idx,
        )
        _, group_mse = per_initial_state_mse(
            pred_test["omega_hat"],
            symmetric_data.omega[test_idx],
            symmetric_data.initial_state_id[test_idx],
        )
        model_metrics_by_state[mode] = group_mse
        print(
            f"best_epoch={result['best_epoch']} | "
            f"val closure MSE={result['validation']['mse']:.6e} | "
            f"test closure MSE={result['test']['mse']:.6e} | "
            f"test cos={result['test']['mean_cosine_nonzero']}"
        )

    # Baseline/test metrics are evaluated on exactly the same held-out initial states.
    state_ids = np.unique(symmetric_data.initial_state_id[test_idx])
    zero_group_mse = model_metrics_by_state["zero"]
    comparison_names = ["ridge", "none", "fixed", "learned"]
    ridge_group_mse = per_initial_state_mse(
        ridge_test_hat, symmetric_data.omega[test_idx], symmetric_data.initial_state_id[test_idx]
    )[1]
    model_metrics_by_state["ridge"] = ridge_group_mse
    comparisons: dict[str, Any] = {
        "ridge": bootstrap_improvement(
            ridge_group_mse, zero_group_mse, args.model_seed + 101,
            args.bootstrap_draws, len(comparison_names),
        )
    }
    for offset, mode in enumerate(("none", "fixed", "learned"), start=1):
        comparisons[mode] = bootstrap_improvement(
            model_metrics_by_state[mode], zero_group_mse,
            args.model_seed + 101 + offset,
            args.bootstrap_draws,
            len(comparison_names),
        )

    payload = {
        "protocol": "GSA-V1 forced two-flip square; symmetric transport is the learning target",
        "project_config": asdict(project_cfg),
        "model_config": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "beta_action": args.beta_action,
            "max_len": 3,
            "lambda_omega": args.lambda_omega,
            "lambda_geometry": args.lambda_geometry,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs_max": args.epochs,
            "patience": args.patience,
            "selection_metric": "validation closure MSE",
        },
        "split": {
            "unit": "initial_state_id",
            "train_initial_state_ids": np.unique(symmetric_data.initial_state_id[train_idx]).tolist(),
            "validation_initial_state_ids": np.unique(symmetric_data.initial_state_id[val_idx]).tolist(),
            "test_initial_state_ids": np.unique(symmetric_data.initial_state_id[test_idx]).tolist(),
            "train_pairs": int(len(train_idx)),
            "validation_pairs": int(len(val_idx)),
            "test_pairs": int(len(test_idx)),
            "scope_note": "Site-pair types are shared across splits; only initial states are held out.",
        },
        "controls": {
            "integrable_max_abs_omega": integrable_max_omega,
            "symmetric_nonzero_pair_count": n_nonzero,
            "zero_omega_baseline_test": zero_metrics,
            "ridge_omega_baseline_test": ridge_test_metrics,
            "ridge_alpha": args.ridge_alpha,
            "ridge_features": ["X0", "packed_J0", "site_pair_one_hot", "X_i*X_j*J_ij", "eta", "eta_squared"],
        },
        "models": model_results,
        "cluster_bootstrap_improvement_vs_zero": comparisons,
        "statistical_unit": "initial state; paired cluster bootstrap over test initial states",
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_familywise_comparisons": len(comparison_names),
        "limitations": [
            "The experiment uses forced two-flip paths, not adaptive Ogata trajectories.",
            "The test split holds out initial states, but not site-pair types or longer path lengths.",
            "The learned edge action is defined on X in {-1,+1}^6; J_t is retained in the dataset but not provided to the attention model.",
            "The integrable transport is a null-control check and is not scored as evidence of learned path memory.",
            "A positive result is predictive evidence for this protocol, not proof of consciousness or a physical Feynman action.",
        ],
    }
    json_path = args.outdir / "gsa_v1_comparison.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)

    print("\nBASELINES SUR LE TEST")
    print(f"Omega=0      : MSE={zero_metrics['mse']:.6e}")
    print(f"Ridge Omega  : MSE={ridge_test_metrics['mse']:.6e} | cos={ridge_test_metrics['mean_cosine_nonzero']}")
    print("\nINTERVALLES BOOTSTRAP APPARIÉS (amélioration MSE vs Omega=0)")
    for name in comparison_names:
        item = comparisons[name]
        print(
            f"{name:8s}: gain={item['mean_mse_improvement_vs_zero']:.6e} | "
            f"IC simultané=[{item['simultaneous_ci_low']}, {item['simultaneous_ci_high']}]"
        )
    print(f"\nJSON : {json_path}")
    print(f"Checkpoints : {args.outdir}")
    print("=" * 72)


def _load_model_from_saved(result: dict[str, Any], config: GSAConfig,
                           device: torch.device, output_dir: Path) -> GSAPathModel:
    model = GSAPathModel(config).to(device)
    try:
        saved = torch.load(output_dir / result["checkpoint"], map_location=device, weights_only=True)
    except TypeError:  # compatibility with older PyTorch
        saved = torch.load(output_dir / result["checkpoint"], map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    main()
