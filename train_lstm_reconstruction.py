"""
train_lstm_reconstruction.py
Projet : NK-LSTM-PATH

Modèle LSTM multi-tâche pour la reconstruction temporelle séquence-à-séquence
à partir des entrées observées (histoire des spins et eta) :
- Cible J_t : géométrie de couplage (symétrisée, diagonale nulle)
- Cible A_t : accessibilité instantanée (tête linéaire)
- Cible w_t : taux de transition (tête Softplus)
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


class TrajectoryReconstructionDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        targets_J: np.ndarray,
        targets_A: np.ndarray,
        targets_w: np.ndarray,
    ) -> None:
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets_J = torch.tensor(targets_J, dtype=torch.float32)
        self.targets_A = torch.tensor(targets_A, dtype=torch.float32)
        self.targets_w = torch.tensor(targets_w, dtype=torch.float32)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.inputs[idx],
            self.targets_J[idx],
            self.targets_A[idx],
            self.targets_w[idx],
        )


def project_symmetric_zero_diag(mat: torch.Tensor) -> torch.Tensor:
    """Projette un tenseur (..., N, N) sur les matrices symétriques à diagonale nulle."""
    sym = 0.5 * (mat + torch.swapaxes(mat, -1, -2))
    n = sym.shape[-1]
    eye = torch.eye(n, device=sym.device, dtype=sym.dtype)
    return sym * (1.0 - eye)


class LatentGeometryPathLSTM(nn.Module):
    """LSTM séquence-à-séquence reconstruisant J_t, A_t et w_t."""

    def __init__(
        self,
        input_dim: int,
        n_sites: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_sites = n_sites
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_J = nn.Linear(hidden_dim, n_sites * n_sites)
        self.head_A = nn.Linear(hidden_dim, n_sites)
        self.head_w = nn.Sequential(
            nn.Linear(hidden_dim, n_sites),
            nn.Softplus(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        batch_size, seq_len, _ = out.shape
        raw_J = self.head_J(out).view(batch_size, seq_len, self.n_sites, self.n_sites)
        pred_J = project_symmetric_zero_diag(raw_J)
        pred_A = self.head_A(out)
        pred_w = self.head_w(out)
        return pred_J, pred_A, pred_w


def split_by_disjoint_seeds(
    metadata: List[Dict[str, Any]],
    test_init_seeds: Tuple[int, ...],
    test_nk_seeds: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray]:
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


def train_one_epoch(
    model: LatentGeometryPathLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    lambda_J: float,
    lambda_A: float,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y_J, y_A, y_w in loader:
        x = x.to(device)
        y_J = y_J.to(device)
        y_A = y_A.to(device)
        y_w = y_w.to(device)

        optimizer.zero_grad()
        p_J, p_A, p_w = model(x)

        l_w = loss_fn(p_w, y_w)
        l_A = loss_fn(p_A, y_A)
        l_J = loss_fn(p_J, y_J)
        loss = l_w + lambda_A * l_A + lambda_J * l_J

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


def evaluate_stratified(
    model: LatentGeometryPathLSTM,
    test_dataset: TrajectoryReconstructionDataset,
    test_metadata: List[Dict[str, Any]],
    device: str,
) -> Dict[str, Dict[str, Any]]:
    model.eval()
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    preds_J, preds_A, preds_w = [], [], []
    with torch.no_grad():
        for x, _, _, _ in loader:
            x = x.to(device)
            pj, pa, pw = model(x)
            preds_J.append(pj.cpu())
            preds_A.append(pa.cpu())
            preds_w.append(pw.cpu())

    p_J = torch.cat(preds_J, dim=0)
    p_A = torch.cat(preds_A, dim=0)
    p_w = torch.cat(preds_w, dim=0)

    t_J = test_dataset.targets_J
    t_A = test_dataset.targets_A
    t_w = test_dataset.targets_w

    etas = np.array([float(m["eta"]) for m in test_metadata])
    regimes = {
        "global": np.ones(len(etas), dtype=bool),
        "linear_regime (eta <= 0.25)": etas <= 0.25,
        "saturated_regime (eta > 0.25)": etas > 0.25,
    }

    report: Dict[str, Dict[str, Any]] = {}
    for name, mask in regimes.items():
        if np.sum(mask) == 0:
            continue
        idx = torch.tensor(mask, dtype=torch.bool)
        m_J = compute_metrics(t_J[idx], p_J[idx])
        m_A = compute_metrics(t_A[idx], p_A[idx])
        m_w = compute_metrics(t_w[idx], p_w[idx])
        report[name] = {
            "n_trajectories": int(np.sum(mask)),
            "J_mse": m_J["mse"],
            "J_mae": m_J["mae"],
            "J_r2": m_J["r2"],
            "A_mse": m_A["mse"],
            "A_mae": m_A["mae"],
            "A_r2": m_A["r2"],
            "w_mse": m_w["mse"],
            "w_mae": m_w["mae"],
            "w_r2": m_w["r2"],
        }
    return report


def load_records(meta_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("trajectories", "samples", "records"):
        if isinstance(meta_payload.get(key), list):
            return meta_payload[key]
    raise KeyError("Aucune liste de trajectoires trouvée dans le fichier de métadonnées JSON.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraînement LSTM pour reconstruire J_t, A_t et w_t sur NK-LSTM-PATH"
    )
    parser.add_argument("--data-prefix", type=str, default="nk_lstm_path_trajectoires")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-J", type=float, default=1.0)
    parser.add_argument("--lambda-A", type=float, default=0.5)
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
    if "targets_J" not in data.files:
        raise KeyError(
            f"'targets_J' est absent de {npz_path} (clés présentes : {data.files}). "
            "Assure-toi d'utiliser l'archive produite par generateur_trajectoires_nk_lstm.py."
        )

    inputs = data["inputs"]
    targets_J = data["targets_J"]
    targets_A = data["targets_A"]
    targets_w = data["targets_w"]

    meta_payload = json.loads(json_path.read_text(encoding="utf-8"))
    metadata = load_records(meta_payload)

    train_idx, test_idx = split_by_disjoint_seeds(
        metadata,
        test_init_seeds=(args.test_init_seed,),
        test_nk_seeds=(args.test_nk_seed,),
    )

    n_sites = targets_w.shape[-1]
    print("=" * 84)
    print(f"RECONSTRUCTION LATENTE NK-LSTM-PATH (TRAJECTOIRES TOTALES : {len(metadata)})")
    print(f"Train : {len(train_idx)} trajectoires | Test : {len(test_idx)} trajectoires")
    print(
        f"Graines réservées au test : init_seed={args.test_init_seed}, nk_seed={args.test_nk_seed}"
    )
    print("=" * 84)

    train_set = TrajectoryReconstructionDataset(
        inputs[train_idx], targets_J[train_idx], targets_A[train_idx], targets_w[train_idx]
    )
    test_set = TrajectoryReconstructionDataset(
        inputs[test_idx], targets_J[test_idx], targets_A[test_idx], targets_w[test_idx]
    )
    test_meta = [metadata[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LatentGeometryPathLSTM(
        input_dim=inputs.shape[-1],
        n_sites=n_sites,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        loss_val = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lambda_J=args.lambda_J,
            lambda_A=args.lambda_A,
            device=device,
        )
        history.append({"epoch": float(epoch), "train_loss": float(loss_val)})
        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"Époque {epoch:2d}/{args.epochs:2d} | Perte d'entraînement composite : {loss_val:.5f}"
            )

    print("-" * 84)
    print("ÉVALUATION STRATIFIÉE SUR LES CIBLES CACHÉES (J_t, A_t, w_t) :")
    results = evaluate_stratified(model, test_set, test_meta, device=device)
    for regime, metrics in results.items():
        print(f"\n--- Régime : {regime} (n={metrics['n_trajectories']}) ---")
        print(
            f"  J_t (Géométrie)    : MSE = {metrics['J_mse']:.5f} | MAE = {metrics['J_mae']:.5f} | R² = {metrics['J_r2']:.4f}"
        )
        print(
            f"  A_t (Accessibilité): MSE = {metrics['A_mse']:.5f} | MAE = {metrics['A_mae']:.5f} | R² = {metrics['A_r2']:.4f}"
        )
        print(
            f"  w_t (Taux)         : MSE = {metrics['w_mse']:.5f} | MAE = {metrics['w_mae']:.5f} | R² = {metrics['w_r2']:.4f}"
        )
    print("=" * 84)

    if args.model_out:
        torch.save(model.state_dict(), args.model_out)

    if args.json_out:
        out_payload = {
            "data_prefix": args.data_prefix,
            "train_count": int(len(train_idx)),
            "test_count": int(len(test_idx)),
            "test_init_seed": int(args.test_init_seed),
            "test_nk_seed": int(args.test_nk_seed),
            "history": history,
            "stratified_metrics": results,
        }
        Path(args.json_out).write_text(json.dumps(out_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
