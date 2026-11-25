"""
nk_landscape.py
Paysage adaptatif NK canonique (Kauffman) pour NK-LSTM-PATH.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


def spins_to_bits(X: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Convertit strictement X in {-1, +1}^N en bits b in {0, 1}^N.
    Lève une exception si les composantes ne sont pas exactement +/- 1.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("X doit être un vecteur unidimensionnel.")
    if not np.all(np.isin(arr, [-1.0, 1.0])):
        raise ValueError("Les spins doivent appartenir strictement à {-1, +1}.")
    return (arr > 0.0).astype(np.int64)


@dataclass
class NKLandscape:
    n_spins: int
    k_epistasis: int
    neighbors: np.ndarray
    tables: np.ndarray
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.n_spins < 1:
            raise ValueError("n_spins doit être >= 1.")
        if not (0 <= self.k_epistasis < self.n_spins):
            raise ValueError(
                f"k_epistasis ({self.k_epistasis}) doit vérifier 0 <= K < n_spins ({self.n_spins})."
            )
        self.neighbors = np.asarray(self.neighbors, dtype=np.int64)
        self.tables = np.asarray(self.tables, dtype=np.float64)

        if self.neighbors.shape != (self.n_spins, self.k_epistasis):
            raise ValueError(
                f"Forme attendue pour neighbors : {(self.n_spins, self.k_epistasis)}, "
                f"reçu : {self.neighbors.shape}."
            )
        expected_table_shape = (self.n_spins, 1 << (self.k_epistasis + 1))
        if self.tables.shape != expected_table_shape:
            raise ValueError(
                f"Forme attendue pour tables : {expected_table_shape}, "
                f"reçu : {self.tables.shape}."
            )

        # Précalcul de la dépendance inverse S_k : sites dont la contribution
        # dépend de l'état du site k (k lui-même + sites i où k in neighbors[i])
        self._affected_sites: list[np.ndarray] = []
        for k in range(self.n_spins):
            dependents = [k]
            for i in range(self.n_spins):
                if k in self.neighbors[i]:
                    dependents.append(i)
            self._affected_sites.append(np.array(sorted(dependents), dtype=np.int64))

    @classmethod
    def generate(
        cls,
        n_spins: int,
        k_epistasis: int,
        seed: int = 2026,
        neighbor_mode: str = "random",
        table_low: float = 0.0,
        table_high: float = 1.0,
    ) -> "NKLandscape":
        rng = np.random.default_rng(seed)
        neighbors = np.zeros((n_spins, k_epistasis), dtype=np.int64)

        if neighbor_mode == "random":
            for i in range(n_spins):
                candidates = [idx for idx in range(n_spins) if idx != i]
                chosen = rng.choice(candidates, size=k_epistasis, replace=False)
                neighbors[i] = np.sort(chosen)
        elif neighbor_mode == "adjacent":
            for i in range(n_spins):
                neighbors[i] = np.array(
                    [(i + offset) % n_spins for offset in range(1, k_epistasis + 1)],
                    dtype=np.int64,
                )
        else:
            raise ValueError(f"neighbor_mode inconnu : {neighbor_mode}")

        n_entries = 1 << (k_epistasis + 1)
        tables = rng.uniform(table_low, table_high, size=(n_spins, n_entries))

        return cls(
            n_spins=n_spins,
            k_epistasis=k_epistasis,
            neighbors=neighbors,
            tables=tables,
            seed=seed,
        )

    def _entry_index(self, site_i: int, bits: np.ndarray) -> int:
        idx = int(bits[site_i])
        for nb in self.neighbors[site_i]:
            idx = (idx << 1) | int(bits[nb])
        return idx

    def site_contributions(self, X: Sequence[float] | np.ndarray) -> np.ndarray:
        bits = spins_to_bits(X)
        if bits.shape != (self.n_spins,):
            raise ValueError(
                f"Configuration X de dimension {self.n_spins} attendue, reçu : {bits.shape}."
            )
        contribs = np.zeros(self.n_spins, dtype=np.float64)
        for i in range(self.n_spins):
            entry = self._entry_index(i, bits)
            contribs[i] = self.tables[i, entry]
        return contribs

    def fitness(self, X: Sequence[float] | np.ndarray) -> float:
        return float(np.mean(self.site_contributions(X)))

    def delta_fitness_flip(self, X: Sequence[float] | np.ndarray, node_k: int) -> float:
        """
        Calcule Delta F_k = F_NK(X^(k)) - F_NK(X) en temps local O(K).
        Seuls les sites de S_k = {k} U {i : k in V_i} sont réévalués.
        """
        if not (0 <= node_k < self.n_spins):
            raise IndexError(f"node_k ({node_k}) hors limites pour N={self.n_spins}.")

        bits = spins_to_bits(X)
        affected = self._affected_sites[node_k]

        # Contributions initiales sur l'ensemble local affecté
        f_before = 0.0
        for i in affected:
            entry = self._entry_index(i, bits)
            f_before += self.tables[i, entry]

        # Bascule locale du bit k : 0 -> 1 ou 1 -> 0
        bits[node_k] ^= 1

        # Contributions après bascule
        f_after = 0.0
        for i in affected:
            entry = self._entry_index(i, bits)
            f_after += self.tables[i, entry]

        return float((f_after - f_before) / self.n_spins)

    def __call__(self, X: Sequence[float] | np.ndarray) -> float:
        return self.fitness(X)


def make_nk_fitness_fn(
    n_spins: int,
    k_epistasis: int,
    seed: int = 2026,
    neighbor_mode: str = "random",
) -> NKLandscape:
    return NKLandscape.generate(
        n_spins=n_spins,
        k_epistasis=k_epistasis,
        seed=seed,
        neighbor_mode=neighbor_mode,
    )
