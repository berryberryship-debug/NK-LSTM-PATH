
"""
nk_landscape.py
Paysage adaptatif NK canonique (Kauffman) pour NK-LSTM-PATH.

Responsabilités :
- Génération reproductible des voisinages et tables NK.
- Calcul de la fitness globale F_NK(X).
- Calcul local de Delta F_NK lors d'un flip.
- Interface callable compatible avec trajectories.py.

Convention :
    X ∈ {-1, +1}^N
    b ∈ {0, 1}^N

La fitness est la moyenne des contributions locales :
    F_NK(X) = (1/N) * sum_i f_i(X_i, X_{V_i})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def spins_to_bits(X: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Convertit strictement X ∈ {-1, +1}^N en bits b ∈ {0, 1}^N.

    Lève ValueError si X n'est pas un vecteur de spins valide.
    """
    arr = np.asarray(X, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError("X doit être un vecteur unidimensionnel.")

    if arr.size == 0:
        raise ValueError("X ne peut pas être vide.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("X doit contenir uniquement des valeurs finies.")

    if not np.all(np.isin(arr, [-1.0, 1.0])):
        raise ValueError(
            "Les spins doivent appartenir strictement à {-1, +1}."
        )

    return (arr > 0.0).astype(np.int64)


@dataclass
class NKLandscape:
    """
    Paysage NK canonique de Kauffman.

    Attributs :
        n_spins      : nombre de sites N.
        k_epistasis  : nombre de voisins K par site.
        neighbors    : voisinages de chaque site.
        tables       : tables de contributions locales.
        seed         : graine de génération.

    La méthode generate() construit un paysage reproductible.
    """

    n_spins: int
    k_epistasis: int
    neighbors: np.ndarray
    tables: np.ndarray
    seed: int = 2026

    _affected_sites: list[np.ndarray] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.n_spins, (int, np.integer)):
            raise TypeError("n_spins doit être un entier.")

        if not isinstance(self.k_epistasis, (int, np.integer)):
            raise TypeError("k_epistasis doit être un entier.")

        self.n_spins = int(self.n_spins)
        self.k_epistasis = int(self.k_epistasis)

        if self.n_spins < 1:
            raise ValueError("n_spins doit être >= 1.")

        if not (0 <= self.k_epistasis < self.n_spins):
            raise ValueError(
                f"k_epistasis ({self.k_epistasis}) doit vérifier "
                f"0 <= K < n_spins ({self.n_spins})."
            )

        self.neighbors = np.asarray(
            self.neighbors,
            dtype=np.int64,
        )

        self.tables = np.asarray(
            self.tables,
            dtype=np.float64,
        )

        expected_neighbors = (
            self.n_spins,
            self.k_epistasis,
        )

        if self.neighbors.shape != expected_neighbors:
            raise ValueError(
                f"Forme attendue pour neighbors : {expected_neighbors}, "
                f"reçu : {self.neighbors.shape}."
            )

        expected_tables = (
            self.n_spins,
            1 << (self.k_epistasis + 1),
        )

        if self.tables.shape != expected_tables:
            raise ValueError(
                f"Forme attendue pour tables : {expected_tables}, "
                f"reçu : {self.tables.shape}."
            )

        if not np.all(np.isfinite(self.tables)):
            raise ValueError("Les tables doivent contenir des valeurs finies.")

        if np.any(self.neighbors < 0) or np.any(
            self.neighbors >= self.n_spins
        ):
            raise ValueError("Les indices des voisins sont hors limites.")

        for i in range(self.n_spins):
            if np.any(self.neighbors[i] == i):
                raise ValueError(
                    f"Le site {i} ne peut pas être son propre voisin."
                )

            if np.unique(self.neighbors[i]).size != self.k_epistasis:
                raise ValueError(
                    f"Les voisins du site {i} doivent être distincts."
                )

        # Dépendance inverse :
        # S_k = {k} ∪ {i : k appartient aux voisins de i}.
        #
        # Seules ces contributions changent lorsqu'on retourne le spin k.
        self._affected_sites = []

        for k in range(self.n_spins):
            dependents = [k]

            for i in range(self.n_spins):
                if k in self.neighbors[i]:
                    dependents.append(i)

            self._affected_sites.append(
                np.asarray(
                    sorted(set(dependents)),
                    dtype=np.int64,
                )
            )

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
        """
        Génère un paysage NK reproductible.

        neighbor_mode :
            random   : voisins aléatoires sans remise.
            adjacent : voisins cycliques successifs.

        La génération des tables suit celle des voisinages avec
        le même générateur pseudo-aléatoire.
        """
        if not isinstance(n_spins, (int, np.integer)):
            raise TypeError("n_spins doit être un entier.")

        if not isinstance(k_epistasis, (int, np.integer)):
            raise TypeError("k_epistasis doit être un entier.")

        n_spins = int(n_spins)
        k_epistasis = int(k_epistasis)

        if n_spins < 1:
            raise ValueError("n_spins doit être >= 1.")

        if not (0 <= k_epistasis < n_spins):
            raise ValueError(
                "k_epistasis doit vérifier 0 <= K < n_spins."
            )

        table_low = float(table_low)
        table_high = float(table_high)

        if not np.isfinite(table_low) or not np.isfinite(table_high):
            raise ValueError("Les bornes des tables doivent être finies.")

        if table_low >= table_high:
            raise ValueError(
                "table_low doit être strictement inférieur à table_high."
            )

        if neighbor_mode not in ("random", "adjacent"):
            raise ValueError(
                f"neighbor_mode inconnu : {neighbor_mode}"
            )

        rng = np.random.default_rng(seed)

        neighbors = np.zeros(
            (n_spins, k_epistasis),
            dtype=np.int64,
        )

        if neighbor_mode == "random":
            for i in range(n_spins):
                candidates = np.asarray(
                    [idx for idx in range(n_spins) if idx != i],
                    dtype=np.int64,
                )

                chosen = rng.choice(
                    candidates,
                    size=k_epistasis,
                    replace=False,
                )

                neighbors[i] = np.sort(chosen)

        elif neighbor_mode == "adjacent":
            for i in range(n_spins):
                neighbors[i] = np.asarray(
                    [
                        (i + offset) % n_spins
                        for offset in range(1, k_epistasis + 1)
                    ],
                    dtype=np.int64,
                )

        n_entries = 1 << (k_epistasis + 1)

        tables = rng.uniform(
            table_low,
            table_high,
            size=(n_spins, n_entries),
        )

        return cls(
            n_spins=n_spins,
            k_epistasis=k_epistasis,
            neighbors=neighbors,
            tables=tables,
            seed=int(seed),
        )

    def _entry_index(
        self,
        site_i: int,
        bits: np.ndarray,
    ) -> int:
        """Construit l'indice binaire de la table locale du site i."""
        if not (0 <= site_i < self.n_spins):
            raise IndexError("Indice de site hors limites.")

        index = int(bits[site_i])

        for neighbor in self.neighbors[site_i]:
            index = (index << 1) | int(bits[neighbor])

        return index

    def site_contributions(
        self,
        X: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Retourne les N contributions locales f_i."""
        bits = spins_to_bits(X)

        if bits.shape != (self.n_spins,):
            raise ValueError(
                f"Configuration de dimension {self.n_spins} attendue, "
                f"reçu : {bits.shape}."
            )

        contributions = np.zeros(
            self.n_spins,
            dtype=np.float64,
        )

        for i in range(self.n_spins):
            index = self._entry_index(i, bits)
            contributions[i] = self.tables[i, index]

        return contributions

    def fitness(
        self,
        X: Sequence[float] | np.ndarray,
    ) -> float:
        """
        Fitness globale :
            F_NK(X) = mean_i f_i(X).
        """
        return float(np.mean(self.site_contributions(X)))

    def delta_fitness_flip(
        self,
        X: Sequence[float] | np.ndarray,
        node_k: int,
    ) -> float:
        """
        Variation exacte de fitness après inversion du site k :

            Delta F_k = F_NK(X^(k)) - F_NK(X).

        Seules les contributions des sites affectés sont recalculées.

        Complexité :
            O(|S_k| * K), au lieu de O(N * K).
        """
        if not isinstance(node_k, (int, np.integer)):
            raise TypeError("node_k doit être un entier.")

        node_k = int(node_k)

        if not (0 <= node_k < self.n_spins):
            raise IndexError(
                f"node_k ({node_k}) hors limites pour N={self.n_spins}."
            )

        bits = spins_to_bits(X)

        if bits.shape != (self.n_spins,):
            raise ValueError(
                f"Dimension attendue : {self.n_spins}, "
                f"reçu : {bits.size}."
            )

        affected = self._affected_sites[node_k]

        f_before = 0.0

        for i in affected:
            index = self._entry_index(int(i), bits)
            f_before += self.tables[i, index]

        bits_after = bits.copy()
        bits_after[node_k] ^= 1

        f_after = 0.0

        for i in affected:
            index = self._entry_index(int(i), bits_after)
            f_after += self.tables[i, index]

        return float(
            (f_after - f_before) / self.n_spins
        )

    def __call__(
        self,
        X: Sequence[float] | np.ndarray,
    ) -> float:
        """Permet d'utiliser directement le paysage comme fonction."""
        return self.fitness(X)


def make_nk_fitness_fn(
    n_spins: int,
    k_epistasis: int,
    seed: int = 2026,
    neighbor_mode: str = "random",
) -> NKLandscape:
    """Construit et retourne un paysage NK callable."""
    return NKLandscape.generate(
        n_spins=n_spins,
        k_epistasis=k_epistasis,
        seed=seed,
        neighbor_mode=neighbor_mode,
    )
