"""
train_lstm_reconstruction.py
Projet : NK-LSTM-PATH

LSTM multi-tâche pour la reconstruction temporelle.

Entrées :
    - X_0, ..., X_T ;
    - eta ;
    - J0 empaqueté ;
    - représentation explicite du paysage NK.

Cibles :
    - J_1, ..., J_T ;
    - A_1, ..., A_T ;
    - w_1, ..., w_T.

Alignement :
    La sortie LSTM à l'indice 0 correspond à X0.
    Les cibles commencent après la première mutation.
    L'entraînement utilise donc les sorties[:, 1:].

Contraintes des sorties :
    - J : matrice symétrique à diagonale nulle ;
    - A : accessibilité non négative ;
    - w : taux compris entre 0 et mu.

Évaluation :
    - globale ;
    - nouvelle graine initiale uniquement ;
    - nouveau paysage NK uniquement ;
    - nouvelle graine initiale et nouveau paysage NK ;
    - régimes de eta.

Les catégories de généralisation sont définies à partir des graines
effectivement présentes dans l'ensemble d'entraînement.

Les matrices J sont évaluées et entraînées sur leur triangle
supérieur strict afin d'éviter le double comptage des arêtes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ======================================================================
# 1. VALIDATION DES VALEURS DE MÉTADONNÉES
# ======================================================================

def metadata_seed_to_int(
    value: Any,
    field_name: str,
    index: int,
) -> int:
    """
    Convertit une graine de métadonnées en entier sans troncature.

    Accepte :
        - les entiers ;
        - les flottants représentant exactement un entier ;
        - les chaînes contenant un entier.

    Refuse :
        - les booléens ;
        - les valeurs non finies ;
        - les flottants non entiers ;
        - les valeurs non convertibles.
    """

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{field_name} doit être un entier, "
            f"pas un booléen, à l'indice {index}."
        )

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):

        numeric_value = float(value)

        if not np.isfinite(numeric_value):
            raise ValueError(
                f"{field_name} n'est pas fini à l'indice {index}."
            )

        if not numeric_value.is_integer():
            raise ValueError(
                f"{field_name} doit être entier à l'indice {index}. "
                f"Valeur observée : {numeric_value}."
            )

        return int(numeric_value)

    if isinstance(value, str):

        stripped = value.strip()

        if not stripped:
            raise ValueError(
                f"{field_name} est une chaîne vide à l'indice {index}."
            )

        try:
            return int(stripped, 10)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} doit contenir un entier à l'indice {index}. "
                f"Valeur observée : {value!r}."
            ) from exc

    raise TypeError(
        f"Type invalide pour {field_name} à l'indice {index} : "
        f"{type(value).__name__}."
    )


# ======================================================================
# 2. DATASET
# ======================================================================

class TrajectoryReconstructionDataset(Dataset):
    """Dataset séquence-à-séquence pour NK-LSTM-PATH."""

    def __init__(
        self,
        inputs: np.ndarray,
        targets_J: np.ndarray,
        targets_A: np.ndarray,
        targets_w: np.ndarray,
    ) -> None:

        self.inputs = torch.as_tensor(
            inputs,
            dtype=torch.float32,
        )

        self.targets_J = torch.as_tensor(
            targets_J,
            dtype=torch.float32,
        )

        self.targets_A = torch.as_tensor(
            targets_A,
            dtype=torch.float32,
        )

        self.targets_w = torch.as_tensor(
            targets_w,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        return (
            self.inputs[idx],
            self.targets_J[idx],
            self.targets_A[idx],
            self.targets_w[idx],
        )


# ======================================================================
# 3. PROJECTION GÉOMÉTRIQUE DE J
# ======================================================================

def project_symmetric_zero_diag(
    mat: torch.Tensor,
) -> torch.Tensor:
    """
    Projection sur les matrices symétriques à diagonale nulle.

    Accepte les dimensions :
        (..., N, N)
    """

    if mat.ndim < 2:
        raise ValueError(
            "Le tenseur J doit avoir au moins deux dimensions."
        )

    if mat.shape[-1] != mat.shape[-2]:
        raise ValueError(
            "Le tenseur J doit contenir des matrices carrées."
        )

    sym = 0.5 * (
        mat + torch.swapaxes(mat, -1, -2)
    )

    n = sym.shape[-1]

    eye = torch.eye(
        n,
        device=sym.device,
        dtype=sym.dtype,
    )

    return sym * (1.0 - eye)


# ======================================================================
# 4. ARCHITECTURE LSTM
# ======================================================================

class LatentGeometryPathLSTM(nn.Module):
    """
    LSTM séquence-à-séquence avec trois têtes :

        J : géométrie symétrique à diagonale nulle ;
        A : accessibilité non négative ;
        w : taux de transition dans [0, mu].

    mu_max est lu dans la configuration du générateur.
    """

    def __init__(
        self,
        input_dim: int,
        n_sites: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        mu_max: float = 1.0,
    ) -> None:

        super().__init__()

        if input_dim < 1:
            raise ValueError(
                "input_dim doit être >= 1."
            )

        if n_sites < 2:
            raise ValueError(
                "n_sites doit être >= 2."
            )

        if hidden_dim < 1:
            raise ValueError(
                "hidden_dim doit être >= 1."
            )

        if num_layers < 1:
            raise ValueError(
                "num_layers doit être >= 1."
            )

        if not np.isfinite(mu_max) or mu_max < 0.0:
            raise ValueError(
                "mu_max doit être fini et >= 0."
            )

        if not np.isfinite(dropout) or not (0.0 <= dropout < 1.0):
            raise ValueError(
                "dropout doit appartenir à [0, 1)."
            )

        self.n_sites = int(n_sites)
        self.mu_max = float(mu_max)
        self.input_dim = int(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Géométrie J.
        self.head_J = nn.Linear(
            hidden_dim,
            n_sites * n_sites,
        )

        # Accessibilité A >= 0.
        # ReLU permet également de représenter exactement zéro.
        self.head_A = nn.Sequential(
            nn.Linear(hidden_dim, n_sites),
            nn.ReLU(),
        )

        # Taux w >= 0.
        # La borne supérieure mu est appliquée dans forward().
        self.head_w = nn.Sequential(
            nn.Linear(hidden_dim, n_sites),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        if x.ndim != 3:
            raise ValueError(
                "Les entrées LSTM doivent avoir la forme "
                "(batch, séquence, dimension)."
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Dimension d'entrée incorrecte : "
                f"{x.shape[-1]} au lieu de {self.input_dim}."
            )

        out, _ = self.lstm(x)

        batch_size, seq_len, _ = out.shape

        raw_J = self.head_J(out).view(
            batch_size,
            seq_len,
            self.n_sites,
            self.n_sites,
        )

        pred_J = project_symmetric_zero_diag(raw_J)

        pred_A = self.head_A(out)

        # Le modèle respecte la borne théorique :
        # 0 <= w_k <= mu.
        pred_w = torch.clamp(
            self.head_w(out),
            min=0.0,
            max=self.mu_max,
        )

        return pred_J, pred_A, pred_w


# ======================================================================
# 5. SÉPARATION PAR GRAINES DISJOINTES
# ======================================================================

def split_by_disjoint_seeds(
    metadata: List[Dict[str, Any]],
    test_init_seeds: Tuple[int, ...],
    test_nk_seeds: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Réserve au test toute trajectoire dont :

        init_seed appartient à test_init_seeds
        OU
        nk_seed appartient à test_nk_seeds.

    L'entraînement ne contient donc aucune trajectoire
    utilisant une graine initiale ou NK réservée au test.
    """

    train_indices: List[int] = []
    test_indices: List[int] = []

    init_set = {
        metadata_seed_to_int(seed, "test_init_seed", i)
        for i, seed in enumerate(test_init_seeds)
    }

    nk_set = {
        metadata_seed_to_int(seed, "test_nk_seed", i)
        for i, seed in enumerate(test_nk_seeds)
    }

    if not init_set and not nk_set:
        raise ValueError(
            "Au moins une graine de test est nécessaire."
        )

    for idx, item in enumerate(metadata):

        if not isinstance(item, dict):
            raise TypeError(
                f"Les métadonnées de l'indice {idx} "
                "doivent être un objet."
            )

        if "init_seed" not in item or "nk_seed" not in item:
            raise KeyError(
                f"Métadonnées incomplètes à l'indice {idx}."
            )

        s_init = metadata_seed_to_int(
            item["init_seed"],
            "init_seed",
            idx,
        )

        s_nk = metadata_seed_to_int(
            item["nk_seed"],
            "nk_seed",
            idx,
        )

        if s_init in init_set or s_nk in nk_set:
            test_indices.append(idx)
        else:
            train_indices.append(idx)

    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
    )


# ======================================================================
# 6. MÉTRIQUES
# ======================================================================

def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> Dict[str, float]:
    """Calcule MSE, MAE et R²."""

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Dimensions métriques incompatibles : "
            f"{y_true.shape} != {y_pred.shape}."
        )

    if y_true.numel() == 0:
        raise ValueError(
            "Impossible de calculer des métriques sur un tenseur vide."
        )

    if not torch.isfinite(y_true).all().item():
        raise ValueError(
            "Les valeurs de référence contiennent des valeurs non finies."
        )

    if not torch.isfinite(y_pred).all().item():
        raise ValueError(
            "Les prédictions contiennent des valeurs non finies."
        )

    mse = torch.mean(
        (y_true - y_pred) ** 2
    ).item()

    mae = torch.mean(
        torch.abs(y_true - y_pred)
    ).item()

    denom = torch.sum(
        (y_true - torch.mean(y_true)) ** 2
    ).item()

    num = torch.sum(
        (y_true - y_pred) ** 2
    ).item()

    # Convention pour une cible constante :
    # R² = 1 si la prédiction est exacte, sinon R² = 0.
    if denom <= 1e-12:
        r2 = 1.0 if num <= 1e-12 else 0.0
    else:
        r2 = 1.0 - num / denom

    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
    }


def compute_J_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> Dict[str, float]:
    """
    Métriques sur le triangle supérieur strict de J.

    Les éléments diagonaux et les éléments symétriques dupliqués
    ne participent pas au calcul.
    """

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Dimensions de J incompatibles."
        )

    if y_true.ndim < 2:
        raise ValueError(
            "Les tenseurs J doivent avoir au moins deux dimensions."
        )

    if y_true.shape[-1] != y_true.shape[-2]:
        raise ValueError(
            "Les matrices J doivent être carrées."
        )

    n_sites = y_true.shape[-1]

    if n_sites < 2:
        raise ValueError(
            "Au moins deux sites sont nécessaires pour évaluer J."
        )

    upper = torch.triu_indices(
        n_sites,
        n_sites,
        offset=1,
        device=y_true.device,
    )

    true_edges = y_true[
        ...,
        upper[0],
        upper[1],
    ]

    pred_edges = y_pred[
        ...,
        upper[0],
        upper[1],
    ]

    return compute_metrics(
        true_edges,
        pred_edges,
    )


def compute_all_metrics(
    targets_J: torch.Tensor,
    predictions_J: torch.Tensor,
    targets_A: torch.Tensor,
    predictions_A: torch.Tensor,
    targets_w: torch.Tensor,
    predictions_w: torch.Tensor,
) -> Dict[str, float]:
    """Calcule les métriques des trois cibles."""

    m_J = compute_J_metrics(
        targets_J,
        predictions_J,
    )

    m_A = compute_metrics(
        targets_A,
        predictions_A,
    )

    m_w = compute_metrics(
        targets_w,
        predictions_w,
    )

    return {
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


# ======================================================================
# 7. VÉRIFICATION DES DONNÉES
# ======================================================================

def validate_dataset_arrays(
    inputs: np.ndarray,
    targets_J: np.ndarray,
    targets_A: np.ndarray,
    targets_w: np.ndarray,
    metadata: List[Dict[str, Any]],
    mu_max: float,
) -> None:
    """Vérifie les dimensions, valeurs et contraintes des cibles."""

    if not np.isfinite(mu_max) or mu_max < 0.0:
        raise ValueError(
            "mu_max doit être fini et non négatif."
        )

    arrays = {
        "inputs": inputs,
        "targets_J": targets_J,
        "targets_A": targets_A,
        "targets_w": targets_w,
    }

    for name, array in arrays.items():

        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"{name} doit être un ndarray."
            )

        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(
                f"{name} doit contenir des valeurs numériques."
            )

        if np.issubdtype(array.dtype, np.complexfloating):
            raise TypeError(
                f"{name} doit contenir des valeurs réelles, "
                "pas complexes."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} contient des valeurs non finies."
            )

    if inputs.ndim != 3:
        raise ValueError(
            "inputs doit avoir la forme (N_traj, T+1, D_in)."
        )

    if targets_J.ndim != 4:
        raise ValueError(
            "targets_J doit avoir la forme (N_traj, T, N, N)."
        )

    if targets_A.ndim != 3:
        raise ValueError(
            "targets_A doit avoir la forme (N_traj, T, N)."
        )

    if targets_w.ndim != 3:
        raise ValueError(
            "targets_w doit avoir la forme (N_traj, T, N)."
        )

    n_trajectories = inputs.shape[0]

    if n_trajectories == 0:
        raise ValueError(
            "Le jeu de données est vide."
        )

    if inputs.shape[1] < 2:
        raise ValueError(
            "La séquence d'entrée doit contenir au moins X0 et X1."
        )

    if inputs.shape[2] < 1:
        raise ValueError(
            "La dimension d'entrée doit être strictement positive."
        )

    if not isinstance(metadata, list):
        raise TypeError(
            "Les métadonnées doivent être une liste de trajectoires."
        )

    if not (
        len(targets_J)
        == len(targets_A)
        == len(targets_w)
        == len(metadata)
        == n_trajectories
    ):
        raise ValueError(
            "Les nombres de trajectoires ne correspondent pas "
            "entre NPZ et métadonnées JSON."
        )

    seq_len = targets_w.shape[1]
    n_sites = targets_w.shape[2]

    if seq_len < 1:
        raise ValueError(
            "Les cibles doivent contenir au moins un pas temporel."
        )

    if n_sites < 2:
        raise ValueError(
            "Le nombre de sites doit être >= 2."
        )

    if inputs.shape[1] != seq_len + 1:
        raise ValueError(
            "Décalage temporel incorrect : les entrées doivent "
            "contenir T+1 états et les cibles T états."
        )

    if inputs.shape[2] < n_sites:
        raise ValueError(
            "La dimension d'entrée est incompatible avec le nombre de sites."
        )

    if targets_A.shape != (
        n_trajectories,
        seq_len,
        n_sites,
    ):
        raise ValueError(
            "Forme de targets_A incorrecte."
        )

    if targets_J.shape != (
        n_trajectories,
        seq_len,
        n_sites,
        n_sites,
    ):
        raise ValueError(
            "Forme de targets_J incorrecte."
        )

    if np.any(targets_w < -1e-7):
        raise ValueError(
            "Les taux cibles w ne peuvent pas être négatifs."
        )

    if np.any(targets_A < -1e-7):
        raise ValueError(
            "Les accessibilités cibles A ne peuvent pas être négatives."
        )

    # transition_rates garantit :
    # 0 <= w_k <= mu.
    if np.any(targets_w > mu_max + 1e-6):
        raise ValueError(
            "Une valeur cible w dépasse mu. "
            f"Maximum observé : {float(np.max(targets_w)):.8g}, "
            f"mu : {mu_max:.8g}."
        )

    if not np.allclose(
        targets_J,
        np.swapaxes(targets_J, -1, -2),
        atol=1e-6,
    ):
        raise ValueError(
            "Les cibles J ne sont pas symétriques."
        )

    if not np.allclose(
        np.diagonal(
            targets_J,
            axis1=-2,
            axis2=-1,
        ),
        0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Les diagonales des cibles J ne sont pas nulles."
        )

    for idx, item in enumerate(metadata):

        if not isinstance(item, dict):
            raise TypeError(
                f"Les métadonnées de la trajectoire {idx} "
                "doivent être un objet."
            )

        required = (
            "init_seed",
            "nk_seed",
            "eta",
        )

        for key in required:
            if key not in item:
                raise KeyError(
                    f"Clé '{key}' absente des métadonnées "
                    f"de la trajectoire {idx}."
                )

        metadata_seed_to_int(
            item["init_seed"],
            "init_seed",
            idx,
        )

        metadata_seed_to_int(
            item["nk_seed"],
            "nk_seed",
            idx,
        )

        try:
            eta_value = float(item["eta"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Valeur eta non numérique pour la trajectoire {idx}."
            ) from exc

        if not np.isfinite(eta_value):
            raise ValueError(
                f"Valeur eta non finie pour la trajectoire {idx}."
            )


def validate_metadata_schema(
    meta_payload: Dict[str, Any],
    inputs: np.ndarray,
    targets_w: np.ndarray,
) -> float:
    """
    Vérifie les déclarations du jeu de données correspondant au générateur V2.

    Retourne mu, utilisé comme borne supérieure de w.

    Attention :
    les marqueurs input_policy attestent de la déclaration du générateur.
    Ils ne permettent pas, à eux seuls, de vérifier l'ordre effectif
    des caractéristiques dans chaque vecteur d'entrée.
    """

    if not isinstance(meta_payload, dict):
        raise TypeError(
            "La racine du fichier JSON doit être un objet."
        )

    input_policy = meta_payload.get("input_policy")

    if not isinstance(input_policy, str):
        raise TypeError(
            "input_policy doit être une chaîne de caractères."
        )

    if "X_0_to_X_T" not in input_policy:
        raise ValueError(
            "Le jeu de données ne déclare pas l'entrée X_0:T."
        )

    if "packed_J0" not in input_policy:
        raise ValueError(
            "Le jeu de données ne déclare pas J0 empaqueté."
        )

    if "NK_" not in input_policy:
        raise ValueError(
            "Le jeu de données ne déclare pas la représentation NK."
        )

    config = meta_payload.get("config")

    if not isinstance(config, dict):
        raise KeyError(
            "La configuration du générateur est absente des métadonnées."
        )

    if "mu" not in config:
        raise KeyError(
            "Le paramètre mu est absent de la configuration."
        )

    if isinstance(config["mu"], bool):
        raise TypeError(
            "Le paramètre mu doit être numérique, pas booléen."
        )

    try:
        mu_max = float(config["mu"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Le paramètre mu doit être numérique."
        ) from exc

    if not np.isfinite(mu_max) or mu_max < 0.0:
        raise ValueError(
            "Le paramètre mu doit être fini et non négatif."
        )

    if "input_dim" in meta_payload:

        declared_dim = meta_payload["input_dim"]

        if isinstance(declared_dim, bool):
            raise TypeError(
                "input_dim dans les métadonnées doit être un entier."
            )

        try:
            declared_dim = int(declared_dim)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "input_dim dans les métadonnées n'est pas valide."
            ) from exc

        if declared_dim != inputs.shape[-1]:
            raise ValueError(
                "La dimension d'entrée déclarée dans les métadonnées "
                "ne correspond pas à celle de l'archive NPZ."
            )

    shapes = meta_payload.get("shapes", {})

    if shapes is not None and not isinstance(shapes, dict):
        raise TypeError(
            "Le champ shapes des métadonnées doit être un objet."
        )

    if isinstance(shapes, dict) and "targets_w" in shapes:

        declared_shape = shapes["targets_w"]

        if not isinstance(declared_shape, (list, tuple)):
            raise TypeError(
                "La forme déclarée de targets_w doit être une liste."
            )

        try:
            declared_shape = tuple(
                int(value)
                for value in declared_shape
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "La forme déclarée de targets_w est invalide."
            ) from exc

        if list(declared_shape) != list(targets_w.shape):
            raise ValueError(
                "La forme de targets_w dans les métadonnées "
                "ne correspond pas à l'archive NPZ."
            )

    return mu_max


# ======================================================================
# 8. CONSTRUCTION DES CATÉGORIES DE GÉNÉRALISATION
# ======================================================================

def build_generalization_masks(
    train_metadata: List[Dict[str, Any]],
    test_metadata: List[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """
    Classe les trajectoires de test selon les graines réellement
    présentes dans l'ensemble d'entraînement.

    new_initial_seed_only :
        graine initiale absente du train ;
        graine NK présente dans le train.

    new_NK_seed_only :
        graine initiale présente dans le train ;
        graine NK absente du train.

    new_initial_and_NK_seeds :
        graine initiale absente du train ;
        graine NK absente du train.

    Cette définition évite de considérer automatiquement une graine
    comme connue simplement parce qu'elle n'est pas celle qui a été
    explicitement réservée.
    """

    if len(train_metadata) == 0:
        raise ValueError(
            "Impossible de classifier la généralisation : "
            "les métadonnées d'entraînement sont vides."
        )

    if len(test_metadata) == 0:
        raise ValueError(
            "Impossible de classifier la généralisation : "
            "les métadonnées de test sont vides."
        )

    train_init_seeds = {
        metadata_seed_to_int(
            item["init_seed"],
            "init_seed",
            idx,
        )
        for idx, item in enumerate(train_metadata)
    }

    train_nk_seeds = {
        metadata_seed_to_int(
            item["nk_seed"],
            "nk_seed",
            idx,
        )
        for idx, item in enumerate(train_metadata)
    }

    init_seeds = np.asarray(
        [
            metadata_seed_to_int(
                item["init_seed"],
                "init_seed",
                idx,
            )
            for idx, item in enumerate(test_metadata)
        ],
        dtype=np.int64,
    )

    nk_seeds = np.asarray(
        [
            metadata_seed_to_int(
                item["nk_seed"],
                "nk_seed",
                idx,
            )
            for idx, item in enumerate(test_metadata)
        ],
        dtype=np.int64,
    )

    etas = np.asarray(
        [
            float(item["eta"])
            for item in test_metadata
        ],
        dtype=np.float64,
    )

    init_seen_in_train = np.asarray(
        [
            seed in train_init_seeds
            for seed in init_seeds
        ],
        dtype=bool,
    )

    nk_seen_in_train = np.asarray(
        [
            seed in train_nk_seeds
            for seed in nk_seeds
        ],
        dtype=bool,
    )

    masks = {
        "global": np.ones(
            len(test_metadata),
            dtype=bool,
        ),

        "new_initial_seed_only": (
            ~init_seen_in_train & nk_seen_in_train
        ),

        "new_NK_seed_only": (
            init_seen_in_train & ~nk_seen_in_train
        ),

        "new_initial_and_NK_seeds": (
            ~init_seen_in_train & ~nk_seen_in_train
        ),

        "eta_le_0.25": (
            etas <= 0.25
        ),

        "eta_gt_0.25": (
            etas > 0.25
        ),
    }

    return masks


# ======================================================================
# 9. ENTRAÎNEMENT D'UNE ÉPOQUE
# ======================================================================

def train_one_epoch(
    model: LatentGeometryPathLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    lambda_J: float,
    lambda_A: float,
    lambda_w: float,
    device: str,
) -> Dict[str, float]:

    model.train()

    total_loss = 0.0
    total_l_J = 0.0
    total_l_A = 0.0
    total_l_w = 0.0

    total_samples = 0

    for x, y_J, y_A, y_w in loader:

        x = x.to(device)
        y_J = y_J.to(device)
        y_A = y_A.to(device)
        y_w = y_w.to(device)

        optimizer.zero_grad(set_to_none=True)

        p_J_all, p_A_all, p_w_all = model(x)

        # L'indice 0 correspond à X0.
        # Les cibles commencent après la première mutation.
        p_J = p_J_all[:, 1:, :, :]
        p_A = p_A_all[:, 1:, :]
        p_w = p_w_all[:, 1:, :]

        if p_J.shape != y_J.shape:
            raise RuntimeError(
                f"Alignement incorrect pour J : "
                f"{p_J.shape} != {y_J.shape}."
            )

        if p_A.shape != y_A.shape:
            raise RuntimeError(
                f"Alignement incorrect pour A : "
                f"{p_A.shape} != {y_A.shape}."
            )

        if p_w.shape != y_w.shape:
            raise RuntimeError(
                f"Alignement incorrect pour w : "
                f"{p_w.shape} != {y_w.shape}."
            )

        n_sites = p_J.shape[-1]

        upper = torch.triu_indices(
            n_sites,
            n_sites,
            offset=1,
            device=p_J.device,
        )

        # Chaque arête de J est comptée une seule fois.
        l_J = loss_fn(
            p_J[..., upper[0], upper[1]],
            y_J[..., upper[0], upper[1]],
        )

        l_A = loss_fn(
            p_A,
            y_A,
        )

        l_w = loss_fn(
            p_w,
            y_w,
        )

        loss = (
            lambda_J * l_J
            + lambda_A * l_A
            + lambda_w * l_w
        )

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                "La fonction de perte est devenue non finie."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = x.shape[0]

        total_loss += loss.item() * batch_size
        total_l_J += l_J.item() * batch_size
        total_l_A += l_A.item() * batch_size
        total_l_w += l_w.item() * batch_size

        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "Aucun échantillon d'entraînement."
        )

    return {
        "loss": total_loss / total_samples,
        "loss_J": total_l_J / total_samples,
        "loss_A": total_l_A / total_samples,
        "loss_w": total_l_w / total_samples,
    }


# ======================================================================
# 10. ÉVALUATION STRATIFIÉE
# ======================================================================

def evaluate_stratified(
    model: LatentGeometryPathLSTM,
    train_metadata: List[Dict[str, Any]],
    test_dataset: TrajectoryReconstructionDataset,
    test_metadata: List[Dict[str, Any]],
    device: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Évalue séparément :

        global :
            toutes les trajectoires réservées au test ;

        new_initial_seed_only :
            graine initiale absente du train,
            graine NK présente dans le train ;

        new_NK_seed_only :
            graine initiale présente dans le train,
            graine NK absente du train ;

        new_initial_and_NK_seeds :
            graines initiale et NK absentes du train ;

        eta_le_0.25 :
            régime eta <= 0.25 ;

        eta_gt_0.25 :
            régime eta > 0.25.
    """

    if len(test_dataset) == 0:
        raise ValueError(
            "Le jeu de test est vide."
        )

    if len(test_metadata) != len(test_dataset):
        raise ValueError(
            "Les métadonnées de test ne correspondent pas au dataset."
        )

    masks = build_generalization_masks(
        train_metadata=train_metadata,
        test_metadata=test_metadata,
    )

    model.eval()

    loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )

    preds_J: List[torch.Tensor] = []
    preds_A: List[torch.Tensor] = []
    preds_w: List[torch.Tensor] = []

    with torch.no_grad():

        for x, _, _, _ in loader:

            x = x.to(device)

            p_J_all, p_A_all, p_w_all = model(x)

            # Retrait de la sortie correspondant à X0.
            preds_J.append(
                p_J_all[:, 1:].cpu()
            )

            preds_A.append(
                p_A_all[:, 1:].cpu()
            )

            preds_w.append(
                p_w_all[:, 1:].cpu()
            )

    p_J = torch.cat(
        preds_J,
        dim=0,
    )

    p_A = torch.cat(
        preds_A,
        dim=0,
    )

    p_w = torch.cat(
        preds_w,
        dim=0,
    )

    t_J = test_dataset.targets_J
    t_A = test_dataset.targets_A
    t_w = test_dataset.targets_w

    report: Dict[str, Dict[str, Any]] = {}

    for name, mask in masks.items():

        if not np.any(mask):
            continue

        idx = torch.as_tensor(
            mask,
            dtype=torch.bool,
        )

        metrics = compute_all_metrics(
            targets_J=t_J[idx],
            predictions_J=p_J[idx],
            targets_A=t_A[idx],
            predictions_A=p_A[idx],
            targets_w=t_w[idx],
            predictions_w=p_w[idx],
        )

        report[name] = {
            "n_trajectories": int(np.sum(mask)),
            **metrics,
        }

    return report


# ======================================================================
# 11. LECTURE DES MÉTADONNÉES
# ======================================================================

def load_records(
    meta_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:

    for key in (
        "trajectories",
        "samples",
        "records",
    ):

        value = meta_payload.get(key)

        if isinstance(value, list):
            return value

    raise KeyError(
        "Aucune liste de trajectoires trouvée "
        "dans le fichier de métadonnées JSON."
    )


# ======================================================================
# 12. AFFICHAGE DES MÉTRIQUES
# ======================================================================

def print_evaluation_report(
    results: Dict[str, Dict[str, Any]],
) -> None:

    print("-" * 84)
    print("ÉVALUATION SUR LES TRAJECTOIRES RÉSERVÉES")
    print("-" * 84)

    for regime, metrics in results.items():

        print(
            f"\n--- {regime} "
            f"(n={metrics['n_trajectories']}) ---"
        )

        print(
            f"J_t : MSE={metrics['J_mse']:.6f} "
            f"| MAE={metrics['J_mae']:.6f} "
            f"| R²={metrics['J_r2']:.4f}"
        )

        print(
            f"A_t : MSE={metrics['A_mse']:.6f} "
            f"| MAE={metrics['A_mae']:.6f} "
            f"| R²={metrics['A_r2']:.4f}"
        )

        print(
            f"w_t : MSE={metrics['w_mse']:.6f} "
            f"| MAE={metrics['w_mae']:.6f} "
            f"| R²={metrics['w_r2']:.4f}"
        )

    print("-" * 84)


# ======================================================================
# 13. PROGRAMME PRINCIPAL
# ======================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Entraînement LSTM NK-LSTM-PATH V2 "
            "pour reconstruire J, A et w."
        )
    )

    parser.add_argument(
        "--data-prefix",
        type=str,
        default="nk_lstm_path_trajectoires_v2",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--lambda-J",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda-A",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--lambda-w",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--test-init-seed",
        type=int,
        default=99,
    )

    parser.add_argument(
        "--test-nk-seed",
        type=int,
        default=99,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--model-out",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Validation des paramètres
    # --------------------------------------------------------------

    if args.epochs < 1:
        raise ValueError(
            "epochs doit être >= 1."
        )

    if args.batch_size < 1:
        raise ValueError(
            "batch-size doit être >= 1."
        )

    if args.hidden_dim < 1:
        raise ValueError(
            "hidden-dim doit être >= 1."
        )

    if args.num_layers < 1:
        raise ValueError(
            "num-layers doit être >= 1."
        )

    if not np.isfinite(args.dropout):
        raise ValueError(
            "dropout doit être fini."
        )

    if not (0.0 <= args.dropout < 1.0):
        raise ValueError(
            "dropout doit appartenir à [0, 1)."
        )

    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError(
            "lr doit être strictement positif et fini."
        )

    loss_weights = {
        "lambda_J": args.lambda_J,
        "lambda_A": args.lambda_A,
        "lambda_w": args.lambda_w,
    }

    for name, value in loss_weights.items():

        if not np.isfinite(value):
            raise ValueError(
                f"{name} doit être fini."
            )

        if value < 0:
            raise ValueError(
                f"{name} doit être >= 0."
            )

    if sum(loss_weights.values()) <= 0:
        raise ValueError(
            "Au moins un coefficient de perte doit être positif."
        )

    # --------------------------------------------------------------
    # Reproductibilité
    # --------------------------------------------------------------

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --------------------------------------------------------------
    # Chargement des fichiers
    # --------------------------------------------------------------

    npz_path = Path(
        f"{args.data_prefix}.npz"
    )

    json_path = Path(
        f"{args.data_prefix}_metadata.json"
    )

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Archive introuvable : {npz_path}"
        )

    if not json_path.exists():
        raise FileNotFoundError(
            f"Métadonnées introuvables : {json_path}"
        )

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as data:

        required_keys = {
            "inputs",
            "targets_J",
            "targets_A",
            "targets_w",
        }

        missing = required_keys.difference(
            data.files
        )

        if missing:
            raise KeyError(
                f"Clés absentes de l'archive : {sorted(missing)}"
            )

        inputs = data["inputs"]
        targets_J = data["targets_J"]
        targets_A = data["targets_A"]
        targets_w = data["targets_w"]

    meta_payload = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    metadata = load_records(
        meta_payload
    )

    # --------------------------------------------------------------
    # Validation du schéma et des données
    # --------------------------------------------------------------

    mu_max = validate_metadata_schema(
        meta_payload=meta_payload,
        inputs=inputs,
        targets_w=targets_w,
    )

    validate_dataset_arrays(
        inputs=inputs,
        targets_J=targets_J,
        targets_A=targets_A,
        targets_w=targets_w,
        metadata=metadata,
        mu_max=mu_max,
    )

    # --------------------------------------------------------------
    # Séparation entraînement / test
    # --------------------------------------------------------------

    train_idx, test_idx = split_by_disjoint_seeds(
        metadata=metadata,
        test_init_seeds=(
            args.test_init_seed,
        ),
        test_nk_seeds=(
            args.test_nk_seed,
        ),
    )

    if len(train_idx) == 0:
        raise ValueError(
            "Aucune trajectoire d'entraînement après séparation."
        )

    if len(test_idx) == 0:
        raise ValueError(
            "Aucune trajectoire de test après séparation."
        )

    train_set = TrajectoryReconstructionDataset(
        inputs[train_idx],
        targets_J[train_idx],
        targets_A[train_idx],
        targets_w[train_idx],
    )

    test_set = TrajectoryReconstructionDataset(
        inputs[test_idx],
        targets_J[test_idx],
        targets_A[test_idx],
        targets_w[test_idx],
    )

    train_meta = [
        metadata[int(i)]
        for i in train_idx
    ]

    test_meta = [
        metadata[int(i)]
        for i in test_idx
    ]

    # --------------------------------------------------------------
    # Vérification du découpage des graines
    # --------------------------------------------------------------

    train_init_values = {
        metadata_seed_to_int(
            item["init_seed"],
            "init_seed",
            idx,
        )
        for idx, item in enumerate(train_meta)
    }

    train_nk_values = {
        metadata_seed_to_int(
            item["nk_seed"],
            "nk_seed",
            idx,
        )
        for idx, item in enumerate(train_meta)
    }

    if args.test_init_seed in train_init_values:
        raise RuntimeError(
            "Fuite détectée : la graine initiale réservée "
            "est présente dans l'entraînement."
        )

    if args.test_nk_seed in train_nk_values:
        raise RuntimeError(
            "Fuite détectée : la graine NK réservée "
            "est présente dans l'entraînement."
        )

    # --------------------------------------------------------------
    # Comptage des catégories de généralisation
    # --------------------------------------------------------------

    generalization_masks = build_generalization_masks(
        train_metadata=train_meta,
        test_metadata=test_meta,
    )

    count_new_init_only = int(
        np.sum(
            generalization_masks["new_initial_seed_only"]
        )
    )

    count_new_nk_only = int(
        np.sum(
            generalization_masks["new_NK_seed_only"]
        )
    )

    count_new_both = int(
        np.sum(
            generalization_masks["new_initial_and_NK_seeds"]
        )
    )

    # --------------------------------------------------------------
    # Paramètres du modèle
    # --------------------------------------------------------------

    input_dim = int(
        inputs.shape[-1]
    )

    n_sites = int(
        targets_w.shape[-1]
    )

    seq_len = int(
        targets_w.shape[1]
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------------
    # Affichage du protocole
    # --------------------------------------------------------------

    print("=" * 84)
    print("NK-LSTM-PATH — RECONSTRUCTION V2")
    print("=" * 84)

    print(
        f"Trajectoires totales : {len(metadata)}"
    )

    print(
        f"Train                : {len(train_idx)}"
    )

    print(
        f"Test                 : {len(test_idx)}"
    )

    print(
        f"Nouvelle graine initiale uniquement : "
        f"{count_new_init_only}"
    )

    print(
        f"Nouveau paysage NK uniquement       : "
        f"{count_new_nk_only}"
    )

    print(
        f"Nouvelles graines initiale et NK    : "
        f"{count_new_both}"
    )

    print(
        f"Graines réservées : "
        f"init_seed={args.test_init_seed}, "
        f"nk_seed={args.test_nk_seed}"
    )

    print(
        f"Dimension d'entrée : {input_dim}"
    )

    print(
        f"Nombre de sites    : {n_sites}"
    )

    print(
        f"Longueur entrée    : {inputs.shape[1]}"
    )

    print(
        f"Longueur cible     : {seq_len}"
    )

    print(
        f"Borne des taux w  : mu={mu_max}"
    )

    print(
        f"Device             : {device}"
    )

    print(
        "Entrées : X_0:T + eta + J0 + représentation NK"
    )

    print(
        "Cibles  : J_1:T + A_1:T + w_1:T"
    )

    print("=" * 84)

    # --------------------------------------------------------------
    # Initialisation du modèle
    # --------------------------------------------------------------

    model = LatentGeometryPathLSTM(
        input_dim=input_dim,
        n_sites=n_sites,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mu_max=mu_max,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    loss_fn = nn.MSELoss()

    # Générateur dédié pour le mélange des trajectoires.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    # --------------------------------------------------------------
    # Entraînement
    # --------------------------------------------------------------

    history: List[Dict[str, float]] = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        epoch_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lambda_J=args.lambda_J,
            lambda_A=args.lambda_A,
            lambda_w=args.lambda_w,
            device=device,
        )

        history.append(
            {
                "epoch": float(epoch),
                **{
                    key: float(value)
                    for key, value in epoch_metrics.items()
                },
            }
        )

        if (
            epoch % 5 == 0
            or epoch == args.epochs
        ):

            print(
                f"Époque {epoch:3d}/{args.epochs:3d} "
                f"| Loss = {epoch_metrics['loss']:.6f} "
                f"| J = {epoch_metrics['loss_J']:.6f} "
                f"| A = {epoch_metrics['loss_A']:.6f} "
                f"| w = {epoch_metrics['loss_w']:.6f}"
            )

    # --------------------------------------------------------------
    # Évaluation
    # --------------------------------------------------------------

    results = evaluate_stratified(
        model=model,
        train_metadata=train_meta,
        test_dataset=test_set,
        test_metadata=test_meta,
        device=device,
    )

    print_evaluation_report(
        results
    )

    # --------------------------------------------------------------
    # Sauvegarde du modèle
    # --------------------------------------------------------------

    if args.model_out:

        model_path = Path(
            args.model_out
        )

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "n_sites": n_sites,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "mu_max": mu_max,
                "seed": args.seed,
                "test_init_seed": args.test_init_seed,
                "test_nk_seed": args.test_nk_seed,
                "input_policy": (
                    "X_0:T + eta + J0 + NK_landscape"
                ),
                "target_alignment": (
                    "J_1:T + A_1:T + w_1:T"
                ),
                "loss_weights": loss_weights,
            },
            model_path,
        )

        print(
            f"Modèle sauvegardé : {model_path}"
        )

    # --------------------------------------------------------------
    # Sauvegarde des résultats
    # --------------------------------------------------------------

    if args.json_out:

        json_out = Path(
            args.json_out
        )

        json_out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_payload = {
            "project": "NK-LSTM-PATH",
            "version": "0.5",
            "data_prefix": args.data_prefix,

            "input_dim": input_dim,
            "n_sites": n_sites,
            "seq_len": seq_len,

            "train_count": int(len(train_idx)),
            "test_count": int(len(test_idx)),

            "test_init_seed": int(
                args.test_init_seed
            ),

            "test_nk_seed": int(
                args.test_nk_seed
            ),

            "generalization_counts": {
                "new_initial_seed_only": count_new_init_only,
                "new_NK_seed_only": count_new_nk_only,
                "new_initial_and_NK_seeds": count_new_both,
            },

            "mu_max": float(mu_max),

            "input_policy": (
                "X_0:T + eta + J0 + NK_landscape"
            ),

            "target_alignment": (
                "J_1:T + A_1:T + w_1:T"
            ),

            "loss_weights": {
                "lambda_J": float(args.lambda_J),
                "lambda_A": float(args.lambda_A),
                "lambda_w": float(args.lambda_w),
            },

            "training_parameters": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "hidden_dim": int(args.hidden_dim),
                "num_layers": int(args.num_layers),
                "dropout": float(args.dropout),
                "learning_rate": float(args.lr),
                "seed": int(args.seed),
            },

            "history": history,
            "stratified_metrics": results,
        }

        json_out.write_text(
            json.dumps(
                out_payload,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        print(
            f"Résultats sauvegardés : {json_out}"
        )

    print("=" * 84)
    print("FIN DE L'ENTRAÎNEMENT")
    print("=" * 84)


if __name__ == "__main__":
    main()
