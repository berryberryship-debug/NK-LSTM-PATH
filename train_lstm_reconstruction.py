"""
train_lstm_reconstruction.py
Projet : NK-LSTM-PATH

Modèle LSTM multi-tâche pour la reconstruction temporelle séquence-à-séquence :
- Cible principale : w_t (taux de transition avec tête Softplus)
- Cible auxiliaire : A_t (accessibilité instantanée avec tête linéaire)
- Évaluation stratifiée : eta <= 0.25 vs eta > 0.25
- Séparation anti-fuite par graines (init_seed / nk_seed)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        targets_w: np.ndarray,
        targets_A: np.ndarray,
    ) -> None:
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets_w = torch.tensor(targets_w, dtype=torch.float32)
        self.targets_A = torch.tensor(targets_A, dtype=torch.float32)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets_w[idx], self.targets_A[idx]


class EpistaticPathLSTM(nn.Module):
    """Réseau récurrent séquence-à-séquence avec double tête de projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_w = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus(),
        )
        self.head_A = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        pred_w = self.head_w(out)
        pred_A = self.head_A(out)
        return pred_w, pred_A


def split_by_disjoint_seeds(
    metadata: List[Dict[str, Any]],
    test_init_seeds: Tuple[int, ...],
    test_nk_seeds: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Place dans le test toute séquence impliquant au moins une graine
    (init_seed ou nk_seed) réservée à l'évaluation.
    """
    train_indices: List[int] = []
    test_indices: List[int] = []

    init_set = {int(s) for s in test_init_seeds}
    nk_set = {int(s) for s in test_nk_seeds}

    for idx, item in enumerate(metadata):
        s_init = int(item["init_seed"])
        s_nk = int(item["nk_seed"])
        if s_init in init_set or s_nk in nk_set:
            test_indices.append(idx)
        else:
            train_indices.append(idx)

    return np.array(train_indices, dtype=np.int64), np.array(test_indices, dtype=np.int64)


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((y_true - y_pred) ** 2).item()
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    denom = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
    num = torch.sum((y_true - y_pred) ** 2).item()
    r2 = 1.0 - (num / (denom + 1e-9))
    return {"mse": float(mse), "mae": float(mae), "r2": float(r2)}


def evaluate_order_gap_mae(
    test_metadata: List[Dict[str, Any]],
    t_w: torch.Tensor,
    p_w: torch.Tensor,
) -> Optional[float]:
    """
    Si des paires ('forward', 'swapped_01') partagent (init_seed, state_idx, nk_seed, eta),
    compare l'écart de taux entre ordres vrai (w_fwd - w_swap) et prédit.
    """
    grouped: Dict[Tuple[int, int, int, float], Dict[str, int]] = {}
    for local_idx, m in enumerate(test_metadata):
        tag = str(m.get("order_tag", ""))
        if tag not in ("forward", "swapped_01"):
            continue
        key = (
            int(m["init_seed"]),
            int(m.get("state_idx", -1)),
            int(m["nk_seed"]),
            float(m["eta"]),
        )
        grouped.setdefault(key, {})[tag] = local_idx

    diffs: List[float] = []
    for pair in grouped.values():
        if "forward" in pair and "swapped_01" in pair:
            i_fwd = pair["forward"]
            i_swp = pair["swapped_01"]
            true_gap = t_w[i_fwd] - t_w[i_swp]
            pred_gap = p_w[i_fwd] - p_w[i_swp]
            diffs.append(torch.mean(torch.abs(true_gap - pred_gap)).item())

    return float(np.mean(diffs)) if diffs else None


def train_model(
    model: EpistaticPathLSTM,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn_w: nn.Module,
    loss_fn_A: nn.Module,
    lambda_aux: float = 0.5,
    device: str = "cpu",
) -> float:
    model.train()
    total_loss = 0.0

    for x, y_w, y_A in train_loader:
        x = x.to(device)
        y_w = y_w.to(device)
        y_A = y_A.to(device)

        optimizer.zero_grad()
        p_w, p_A = model(x)

        l_w = loss_fn_w(p_w, y_w)
        l_A = loss_fn_A(p_A, y_A)
        loss = l_w + lambda_aux * l_A

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(x)

    return total_loss / len(train_loader.dataset)


def evaluate_stratified(
    model: EpistaticPathLSTM,
    test_dataset: SequenceDataset,
    test_metadata: List[Dict[str, Any]],
    device: str = "cpu",
) -> Dict[str, Dict[str, Optional[float]]]:
    model.eval()
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    preds_w, preds_A = [], []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            pw, pa = model(x)
            preds_w.append(pw.cpu())
            preds_A.append(pa.cpu())

    p_w = torch.cat(preds_w, dim=0)
    p_A = torch.cat(preds_A, dim=0)
    t_w = test_dataset.targets_w
    t_A = test_dataset.targets_A

    etas = np.array([float(m["eta"]) for m in test_metadata])
    regimes = {
        "global": np.ones(len(etas), dtype=bool),
        "linear_regime (eta <= 0.25)": etas <= 0.25,
        "saturated_regime (eta > 0.25)": etas > 0.25,
    }

    report: Dict[str, Dict[str, Optional[float]]] = {}
    for name, mask in regimes.items():
        if np.sum(mask) == 0:
            continue
        idx = torch.tensor(mask, dtype=torch.bool)
        m_w = compute_metrics(t_w[idx], p_w[idx])
        m_A = compute_metrics(t_A[idx], p_A[idx])
        sub_meta = [m for m, flag in zip(test_metadata, mask) if flag]
        order_gap_mae = evaluate_order_gap_mae(sub_meta, t_w[idx], p_w[idx])

        report[name] = {
            "n_sequences": int(np.sum(mask)),
            "w_mse": m_w["mse"],
            "w_mae": m_w["mae"],
            "w_r2": m_w["r2"],
            "A_mse": m_A["mse"],
            "A_mae": m_A["mae"],
            "A_r2": m_A["r2"],
            "order_gap_w_mae": order_gap_mae,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement LSTM pour NK-LSTM-PATH")
    parser.add_argument("--data-prefix", type=str, default="dataset_discret_wt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-aux", type=float, default=0.5)
    parser.add_argument("--test-init-seed", type=int, default=99)
    parser.add_argument("--test-nk-seed", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    npz_path = Path(f"{args.data_prefix}.npz")
    json_path = Path(f"{args.data_prefix}_metadata.json")

    data = np.load(npz_path)
    inputs = data["inputs"]
    targets_w = data["targets_w"]
    targets_A = data["targets_A"]

    meta_payload = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = meta_payload["samples"]

    train_idx, test_idx = split_by_disjoint_seeds(
        metadata,
        test_init_seeds=(args.test_init_seed,),
        test_nk_seeds=(args.test_nk_seed,),
    )

    print("=" * 80)
    print(f"CONFIGURATION DE L'ENTRAÎNEMENT (ÉCHANTILLONS TOTAUX : {len(metadata)})")
    print(f"Train : {len(train_idx)} séquences | Test : {len(test_idx)} séquences")
    print(
        f"Graines réservées au test : init_seed={args.test_init_seed}, nk_seed={args.test_nk_seed}"
    )
    print("=" * 80)

    train_set = SequenceDataset(inputs[train_idx], targets_w[train_idx], targets_A[train_idx])
    test_set = SequenceDataset(inputs[test_idx], targets_w[test_idx], targets_A[test_idx])
    test_meta = [metadata[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EpistaticPathLSTM(
        input_dim=inputs.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=targets_w.shape[-1],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn_w = nn.MSELoss()
    loss_fn_A = nn.MSELoss()

    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        loss_epoch = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn_w=loss_fn_w,
            loss_fn_A=loss_fn_A,
            lambda_aux=args.lambda_aux,
            device=device,
        )
        history.append({"epoch": float(epoch), "train_loss": float(loss_epoch)})
        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"Époque {epoch:2d}/{args.epochs:2d} | Perte d'entraînement composite : {loss_epoch:.5f}"
            )

    print("-" * 80)
    print("RÉSULTATS DE L'ÉVALUATION STRATIFIÉE HORS-DISTRIBUTION :")
    results = evaluate_stratified(model, test_set, test_meta, device=device)
    for regime, metrics in results.items():
        print(f"\n--- Régime : {regime} (n={metrics['n_sequences']}) ---")
        print(
            f"  w_t (Taux)        : MSE = {metrics['w_mse']:.5f} | MAE = {metrics['w_mae']:.5f} | R² = {metrics['w_r2']:.4f}"
        )
        print(
            f"  A_t (Accessibilité): MSE = {metrics['A_mse']:.5f} | MAE = {metrics['A_mae']:.5f} | R² = {metrics['A_r2']:.4f}"
        )
        if metrics["order_gap_w_mae"] is not None:
            print(f"  Écart d'ordre |Δw| : MAE = {metrics['order_gap_w_mae']:.5f}")
    print("=" * 80)

    if args.model_out:
        torch.save(model.state_dict(), args.model_out)

    if args.json_out:
        out_data = {
            "data_prefix": args.data_prefix,
            "train_count": int(len(train_idx)),
            "test_count": int(len(test_idx)),
            "test_init_seed": int(args.test_init_seed),
            "test_nk_seed": int(args.test_nk_seed),
            "history": history,
            "stratified_metrics": results,
        }
        Path(args.json_out).write_text(json.dumps(out_data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
