"""src/nk_landscape.py -- Paysage NK minimal (Kauffman). Spins s in {-1,+1}; x=(s+1)/2 pour F_NK."""
from __future__ import annotations
import numpy as np


class NKLandscape:
    def __init__(self, N: int, K: int, seed: int):
        if not (0 <= K <= N - 1):
            raise ValueError("0 <= K <= N-1 requis.")
        rng = np.random.default_rng(seed)
        self.N, self.K = N, K
        self.neighbors = []
        for i in range(N):
            others = [j for j in range(N) if j != i]
            nb = sorted(rng.choice(others, size=K, replace=False).tolist()) if K > 0 else []
            self.neighbors.append(nb)
        self.tables = rng.random((N, 2 ** (K + 1)))
        self._cache: dict = {}

    def F(self, X) -> float:
        key = tuple(int(v) for v in X)
        if key in self._cache:
            return self._cache[key]
        x = [(v + 1) // 2 for v in key]
        tot = 0.0
        for i in range(self.N):
            idx = x[i]
            for j in self.neighbors[i]:
                idx = 2 * idx + x[j]
            tot += self.tables[i, idx]
        val = tot / self.N
        self._cache[key] = val
        return val
