"""Moteur de trajectoires pour NK-LSTM-PATH.

Ce module centralise :
- l'interface de compatibilite NKLandscape (ancienne et nouvelle signatures) ;
- fitness et taux de transition de Metropolis ;
- dissipation analytique de la fibre ;
- simulation par amincissement d'Ogata ;
- simulation exacte par morceaux lorsque lambda = 0 ;
- resolution robuste des franchissements de seuil ;
- distances statistiques utilisees par les tests de validation.

Le generateur NK canonique est nk_landscape.NKLandscape.generate lorsque le
module racine est disponible. Un repli local reproduit le meme ordre de tirage
aleatoire si le module canonique n'est pas importable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Imports locaux et generateur NK canonique
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from fiber_bundle import (  # type: ignore
        _validate_state as _fb_validate_state,
        _validate_J as _fb_validate_J,
        accessibility_A,
        mutate_postflip,
        psi_sym,
    )
except ImportError:  # Permet aussi l'import lorsque le module est lance seul.
    def _fb_validate_state(X: np.ndarray, atol: float = 1e-12) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
            raise ValueError("X doit etre un vecteur fini de dimension >= 2.")
        if not np.all(np.isclose(np.abs(x), 1.0, atol=atol)):
            raise ValueError("Les spins doivent appartenir a {-1, +1}.")
        return np.where(x >= 0.0, 1.0, -1.0)

    def _fb_validate_J(J: np.ndarray, atol: float = 1e-12) -> np.ndarray:
        j = np.asarray(J, dtype=np.float64)
        if j.ndim != 2 or j.shape[0] != j.shape[1] or j.shape[0] < 2:
            raise ValueError("J doit etre une matrice carree de dimension >= 2.")
        if not np.all(np.isfinite(j)):
            raise ValueError("J contient des valeurs non finies.")
        if not np.allclose(j, j.T, atol=atol):
            raise ValueError("J doit etre symetrique.")
        if not np.allclose(np.diag(j), 0.0, atol=atol):
            raise ValueError("La diagonale de J doit etre nulle.")
        out = 0.5 * (j + j.T)
        np.fill_diagonal(out, 0.0)
        return out

    def accessibility_A(J: np.ndarray) -> np.ndarray:
        j = _fb_validate_J(J)
        return np.sum(j * j, axis=1)

    def psi_sym(X: np.ndarray, J: np.ndarray) -> np.ndarray:
        x = _fb_validate_state(X)
        j = _fb_validate_J(J)
        if j.shape[0] != x.size:
            raise ValueError("Dimensions incompatibles entre X et J.")
        field = j @ x
        p = 0.5 * (field[:, None] + field[None, :] - j * (x[:, None] + x[None, :]))
        p = 0.5 * (p + p.T)
        np.fill_diagonal(p, 0.0)
        return p

    def mutate_postflip(
        X: np.ndarray, J: np.ndarray, k: int, eta: float,
        psi_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = psi_sym,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = _fb_validate_state(X)
        j = _fb_validate_J(J)
        if j.shape[0] != x.size:
            raise ValueError("Dimensions incompatibles entre X et J.")
        if not (0 <= int(k) < x.size):
            raise IndexError("Indice de mutation hors limites.")
        if not np.isfinite(eta):
            raise ValueError("eta doit etre fini.")
        x_next = x.copy()
        x_next[int(k)] *= -1.0
        p = np.asarray(psi_fn(x_next, j), dtype=np.float64)
        if p.shape != j.shape or not np.all(np.isfinite(p)):
            raise ValueError("psi_fn doit retourner une matrice finie de meme dimension.")
        j_next = j.copy()
        row = j[int(k), :] + float(eta) * p[int(k), :]
        j_next[int(k), :] = row
        j_next[:, int(k)] = row
        np.fill_diagonal(j_next, 0.0)
        return x_next, j_next

try:
    from nk_landscape import NKLandscape as _CanonicalNKLandscape  # type: ignore
except ImportError:
    _CanonicalNKLandscape = None


class _FallbackNKLandscape:
    """Repli local reproduisant l'algorithme de nk_landscape.NKLandscape."""

    def __init__(self, n_spins: int, k_epistasis: int, neighbors: np.ndarray,
                 tables: np.ndarray, seed: int = 2026) -> None:
        self.n_spins = int(n_spins)
        self.k_epistasis = int(k_epistasis)
        self.neighbors = np.asarray(neighbors, dtype=np.int64)
        self.tables = np.asarray(tables, dtype=np.float64)
        self.seed = int(seed)
        if self.n_spins < 1 or not (0 <= self.k_epistasis < self.n_spins):
            raise ValueError("Parametres NK invalides : N >= 1 et 0 <= K < N requis.")
        if self.neighbors.shape != (self.n_spins, self.k_epistasis):
            raise ValueError("Forme invalide pour neighbors.")
        if self.tables.shape != (self.n_spins, 1 << (self.k_epistasis + 1)):
            raise ValueError("Forme invalide pour tables.")
        if not np.all(np.isfinite(self.tables)):
            raise ValueError("Tables NK non finies.")
        self._affected_sites: List[np.ndarray] = []
        for k in range(self.n_spins):
            dep = [k]
            for i in range(self.n_spins):
                if k in self.neighbors[i]:
                    dep.append(i)
            self._affected_sites.append(np.asarray(sorted(set(dep)), dtype=np.int64))

    @classmethod
    def generate(cls, n_spins: int, k_epistasis: int, seed: int = 2026,
                 neighbor_mode: str = "random") -> "_FallbackNKLandscape":
        rng = np.random.default_rng(seed)
        neighbors = np.zeros((n_spins, k_epistasis), dtype=np.int64)
        if neighbor_mode == "random":
            for i in range(n_spins):
                candidates = [idx for idx in range(n_spins) if idx != i]
                if k_epistasis > 0:
                    chosen = rng.choice(candidates, size=k_epistasis, replace=False)
                    neighbors[i] = np.sort(chosen)
        elif neighbor_mode == "adjacent":
            for i in range(n_spins):
                if k_epistasis > 0:
                    neighbors[i] = np.asarray(
                        [(i + off) % n_spins for off in range(1, k_epistasis + 1)],
                        dtype=np.int64,
                    )
        else:
            raise ValueError(f"neighbor_mode inconnu : {neighbor_mode}")
        tables = rng.uniform(0.0, 1.0, size=(n_spins, 1 << (k_epistasis + 1)))
        return cls(n_spins, k_epistasis, neighbors, tables, seed)

    @staticmethod
    def _bits(X: Sequence[float] | np.ndarray) -> np.ndarray:
        x = _fb_validate_state(np.asarray(X, dtype=np.float64))
        return (x > 0.0).astype(np.int64)

    def _entry_index(self, site_i: int, bits: np.ndarray) -> int:
        idx = int(bits[site_i])
        for nb in self.neighbors[site_i]:
            idx = (idx << 1) | int(bits[nb])
        return idx

    def fitness(self, X: Sequence[float] | np.ndarray) -> float:
        bits = self._bits(X)
        if bits.shape != (self.n_spins,):
            raise ValueError("Dimension X incompatible avec le paysage NK.")
        return float(sum(self.tables[i, self._entry_index(i, bits)]
                         for i in range(self.n_spins)) / self.n_spins)

    def delta_fitness_flip(self, X: Sequence[float] | np.ndarray, node_k: int) -> float:
        if not (0 <= int(node_k) < self.n_spins):
            raise IndexError("node_k hors limites.")
        bits = self._bits(X)
        affected = self._affected_sites[int(node_k)]
        before = sum(self.tables[i, self._entry_index(int(i), bits)] for i in affected)
        bits[int(node_k)] ^= 1
        after = sum(self.tables[i, self._entry_index(int(i), bits)] for i in affected)
        return float((after - before) / self.n_spins)

    def __call__(self, X: Sequence[float] | np.ndarray) -> float:
        return self.fitness(X)


class NKLandscape:
    """Adaptateur compatible avec les interfaces historiques et canoniques.

    Formes acceptees : NKLandscape(N, K, seed=...),
    NKLandscape(n=N, k=K, seed=...), NKLandscape(N=N, K=K, seed=...),
    NKLandscape(n_spins=N, k_epistasis=K, seed=...) et generate(...).

    La realisation effective est deleguee au generateur canonique a la racine
    du depot, lorsqu'il est disponible.
    """

    def __init__(self, N: Optional[int] = None, K: Optional[int] = None,
                 seed: int = 2026, neighbor_mode: str = "random", *,
                 n: Optional[int] = None, k: Optional[int] = None,
                 n_spins: Optional[int] = None,
                 k_epistasis: Optional[int] = None) -> None:
        n_values = [v for v in (N, n, n_spins) if v is not None]
        k_values = [v for v in (K, k, k_epistasis) if v is not None]
        if not n_values or not k_values:
            raise TypeError("Fournir N/n/n_spins et K/k/k_epistasis.")
        if any(int(v) != int(n_values[0]) for v in n_values):
            raise ValueError("Valeurs contradictoires pour N.")
        if any(int(v) != int(k_values[0]) for v in k_values):
            raise ValueError("Valeurs contradictoires pour K.")
        n_int, k_int = int(n_values[0]), int(k_values[0])
        if _CanonicalNKLandscape is not None:
            self._impl = _CanonicalNKLandscape.generate(
                n_spins=n_int,
                k_epistasis=k_int,
                seed=int(seed),
                neighbor_mode=neighbor_mode,
            )
        else:
            self._impl = _FallbackNKLandscape.generate(
                n_spins=n_int,
                k_epistasis=k_int,
                seed=int(seed),
                neighbor_mode=neighbor_mode,
            )
        self.N = n_int
        self.K = k_int
        self.seed = int(seed)
        self.neighbor_mode = neighbor_mode

    @classmethod
    def generate(cls, n_spins: int, k_epistasis: int, seed: int = 2026,
                 neighbor_mode: str = "random") -> "NKLandscape":
        return cls(n_spins=n_spins, k_epistasis=k_epistasis,
                   seed=seed, neighbor_mode=neighbor_mode)

    def __getattr__(self, name: str):
        return getattr(self._impl, name)

    def fitness(self, X: Sequence[float] | np.ndarray) -> float:
        return float(self._impl.fitness(X))

    def delta_fitness_flip(self, X: Sequence[float] | np.ndarray, node_k: int) -> float:
        return float(self._impl.delta_fitness_flip(X, node_k))

    def __call__(self, X: Sequence[float] | np.ndarray) -> float:
        return self.fitness(X)


# ---------------------------------------------------------------------------
# Fitness et taux de transition
# ---------------------------------------------------------------------------

FitnessFn = Callable[[np.ndarray], float]


def zero_fnk(X: np.ndarray) -> float:
    """Fitness NK nulle, utile pour les tests et controles sans paysage."""
    _fb_validate_state(X)
    return 0.0


def _finite_scalar(name: str, value: float) -> float:
    val = float(value)
    if not np.isfinite(val):
        raise ValueError(f"{name} doit etre fini.")
    return val


def extended_fitness(X: np.ndarray, J: np.ndarray, lam: float = 0.0,
                     fnk_fn: Optional[FitnessFn] = None) -> float:
    x = _fb_validate_state(X)
    j = _fb_validate_J(J)
    if j.shape[0] != x.size:
        raise ValueError("Dimensions incompatibles entre X et J.")
    coupling = _finite_scalar("lam", lam)
    fn = zero_fnk if fnk_fn is None else fnk_fn
    native = _finite_scalar("F_NK", fn(x.copy()))
    return float(native + 0.5 * coupling * (x @ j @ x))


def delta_fitness_k(X: np.ndarray, J: np.ndarray, k: int,
                    lam: float = 0.0,
                    fnk_fn: Optional[FitnessFn] = None) -> float:
    x = _fb_validate_state(X)
    j = _fb_validate_J(J)
    if j.shape[0] != x.size:
        raise ValueError("Dimensions incompatibles entre X et J.")
    node = int(k)
    if not (0 <= node < x.size):
        raise IndexError(f"Indice k={node} hors limites.")
    coupling = _finite_scalar("lam", lam)
    fn = zero_fnk if fnk_fn is None else fnk_fn

    # Si la fonction provient d'un objet NK qui expose le calcul local,
    # utiliser sa mise a jour O(K). Cela couvre NKLandscape comme objet
    # appelable et NKLandscape.fitness comme methode liee.
    nk_owner = getattr(fn, "__self__", None)
    nk_candidate = nk_owner if nk_owner is not None else fn
    local_delta = getattr(nk_candidate, "delta_fitness_flip", None)
    if callable(local_delta):
        delta_native = _finite_scalar("Delta F_NK", local_delta(x, node))
    else:
        # Repli generique pour toute fonction de fitness arbitraire.
        x_flip = x.copy()
        x_flip[node] *= -1.0
        delta_native = _finite_scalar("Delta F_NK", fn(x_flip) - fn(x))
    delta_coupling = -2.0 * coupling * x[node] * float(j[node] @ x)
    return float(delta_native + delta_coupling)


def transition_rates(X: np.ndarray, J: np.ndarray, mu: float,
                     beta: float, theta_A: float, lam: float,
                     fnk_fn: Optional[FitnessFn] = None) -> np.ndarray:
    x = _fb_validate_state(X)
    j = _fb_validate_J(J)
    if j.shape[0] != x.size:
        raise ValueError("Dimensions incompatibles entre X et J.")
    mu_v = _finite_scalar("mu", mu)
    beta_v = _finite_scalar("beta", beta)
    theta_v = _finite_scalar("theta_A", theta_A)
    lam_v = _finite_scalar("lam", lam)
    if mu_v < 0.0:
        raise ValueError("mu doit etre >= 0.")
    rates = np.zeros(x.size, dtype=np.float64)
    if mu_v == 0.0:
        return rates
    active = accessibility_A(j) > theta_v
    fn = zero_fnk if fnk_fn is None else fnk_fn
    for k in range(x.size):
        if not active[k]:
            continue
        exponent = beta_v * delta_fitness_k(x, j, k, lam=lam_v, fnk_fn=fn)
        # min(1, exp(exponent)) sans debordement exponentiel positif.
        acceptance = 1.0 if exponent >= 0.0 else float(np.exp(exponent))
        rates[k] = mu_v * acceptance
    return rates


# ---------------------------------------------------------------------------
# Dissipation et resultats de simulation
# ---------------------------------------------------------------------------


def dissipate_J(J_t: np.ndarray, J_anchor: np.ndarray, gamma: float,
                dt: float) -> np.ndarray:
    j = _fb_validate_J(J_t)
    anchor = _fb_validate_J(J_anchor)
    if j.shape != anchor.shape:
        raise ValueError("J_t et J_anchor doivent avoir la meme forme.")
    g = _finite_scalar("gamma", gamma)
    delta_t = _finite_scalar("dt", dt)
    if g < 0.0 or delta_t < 0.0:
        raise ValueError("gamma et dt doivent etre >= 0.")
    out = anchor + (j - anchor) * float(np.exp(-g * delta_t))
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 0.0)
    return out


@dataclass
class SimulationEvent:
    time: float
    node: int
    X: np.ndarray
    J: np.ndarray


@dataclass
class SimulationResult:
    X_final: np.ndarray
    J_final: np.ndarray
    events: List[SimulationEvent]
    t_final: float

    @property
    def n_accepted(self) -> int:
        return len(self.events)


def _rng_uniform(rng, low: float = 0.0, high: float = 1.0) -> float:
    value = float(rng.uniform(low, high))
    if not np.isfinite(value):
        raise ValueError("Le generateur aleatoire a retourne une valeur non finie.")
    if value < low or value > high:
        raise ValueError("Le generateur aleatoire a retourne une valeur hors intervalle.")
    return value


def _exp_wait_from_unit_uniform(u: float, rate: float) -> float:
    if rate <= 0.0:
        return math.inf
    # numpy.random.Generator.uniform peut produire 0 ; on borne alors le tirage
    # au plus petit flottant strictement positif de binary64.
    u_safe = max(float(u), float(np.nextafter(0.0, 1.0)))
    return float(-math.log(u_safe) / rate)


def _select_channel(rates: np.ndarray, rng) -> int:
    total = float(np.sum(rates))
    if total <= 0.0:
        raise ValueError("Impossible de selectionner un canal de taux total nul.")
    # Le tirage est exprime dans l'unite du taux. side='right' saute les
    # intervalles cumules de longueur nulle lorsque u2 == 0.
    u2 = _rng_uniform(rng, 0.0, total)
    cumulative = np.cumsum(rates, dtype=np.float64)
    node = int(np.searchsorted(cumulative, u2, side="right"))
    if node >= rates.size:
        positive = np.flatnonzero(rates > 0.0)
        if positive.size == 0:
            raise RuntimeError("Aucun canal strictement positif apres sommation positive.")
        node = int(positive[-1])
    if rates[node] <= 0.0:
        # Protection contre un eventuel arrondi en limite d'intervalle.
        positive = np.flatnonzero((rates > 0.0) & (cumulative > u2))
        if positive.size == 0:
            positive = np.flatnonzero(rates > 0.0)
        node = int(positive[0])
    return node


def _make_result(X: np.ndarray, J: np.ndarray, events: List[SimulationEvent],
                 t_final: float) -> SimulationResult:
    return SimulationResult(
        X_final=np.asarray(X, dtype=np.float64).copy(),
        J_final=np.asarray(J, dtype=np.float64).copy(),
        events=events,
        t_final=float(t_final),
    )


def simulate_ogata_thinning(
    X0: np.ndarray,
    J0: np.ndarray,
    t_max: float,
    mu: float,
    gamma: float,
    beta: float,
    lam: float = 0.0,
    eta: float = 0.0,
    theta_A: float = 0.0,
    J_anchor: Optional[np.ndarray] = None,
    fnk_fn: Optional[FitnessFn] = None,
    psi_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = psi_sym,
    rng=None,
) -> SimulationResult:
    """Simule les mutations avec taux variables et amincissement d'Ogata.

    La borne uniforme est Lambda_max = N * mu. Les temps de proposition sont
    tires via rng.uniform(0, 1), puis le test d'acceptation et le canal utilisent
    des tirages exprimes dans leurs intervalles de taux.
    """
    x = _fb_validate_state(X0)
    j = _fb_validate_J(J0)
    if j.shape[0] != x.size:
        raise ValueError("Dimensions incompatibles entre X0 et J0.")
    anchor = j.copy() if J_anchor is None else _fb_validate_J(J_anchor)
    if anchor.shape != j.shape:
        raise ValueError("J_anchor doit avoir la meme forme que J0.")
    t_end = _finite_scalar("t_max", t_max)
    mu_v = _finite_scalar("mu", mu)
    gamma_v = _finite_scalar("gamma", gamma)
    beta_v = _finite_scalar("beta", beta)
    lam_v = _finite_scalar("lam", lam)
    eta_v = _finite_scalar("eta", eta)
    theta_v = _finite_scalar("theta_A", theta_A)
    if t_end < 0.0 or mu_v < 0.0 or gamma_v < 0.0:
        raise ValueError("t_max, mu et gamma doivent etre >= 0.")
    if rng is None:
        rng = np.random.default_rng()
    fn = zero_fnk if fnk_fn is None else fnk_fn
    n = x.size
    lambda_max = float(n * mu_v)
    events: List[SimulationEvent] = []
    t = 0.0
    if t_end == 0.0 or lambda_max == 0.0:
        return _make_result(x, j, events, t_end)

    # Boucle de propositions ; les propositions rejetees ne changent pas X.
    while t < t_end:
        u1 = _rng_uniform(rng, 0.0, 1.0)
        dt_prop = _exp_wait_from_unit_uniform(u1, lambda_max)
        if not np.isfinite(dt_prop) or t + dt_prop >= t_end:
            j = dissipate_J(j, anchor, gamma_v, t_end - t)
            t = t_end
            break

        j = dissipate_J(j, anchor, gamma_v, dt_prop)
        t += dt_prop
        rates = transition_rates(
            x, j, mu=mu_v, beta=beta_v, theta_A=theta_v,
            lam=lam_v, fnk_fn=fn,
        )
        total = float(np.sum(rates))
        if total <= 0.0:
            continue

        # Uniformite dans [0, Lambda_max) pour l'acceptation du candidat.
        u_accept = _rng_uniform(rng, 0.0, lambda_max)
        if u_accept >= total:
            continue

        node = _select_channel(rates, rng)
        x, j = mutate_postflip(x, j, node, eta_v, psi_fn=psi_fn)
        events.append(SimulationEvent(t, node, x.copy(), j.copy()))

    return _make_result(x, j, events, t)


# ---------------------------------------------------------------------------
# Solveur de racines et changements de masque sous dissipation
# ---------------------------------------------------------------------------


def _stable_quadratic_real_roots(c2: float, c1: float, c0: float) -> List[float]:
    """Racines reelles de c2*x^2+c1*x+c0 avec mise a l'echelle relative.

    Le degre quadratique n'est reduit que si le coefficient mis a l'echelle
    est exactement nul. Le discriminant legerement negatif, dans une tolerance
    de 32 eps multipliee par son echelle naturelle, est traite comme une tangence.
    """
    a0 = _finite_scalar("c2", c2)
    b0 = _finite_scalar("c1", c1)
    c0v = _finite_scalar("c0", c0)
    scale = max(abs(a0), abs(b0), abs(c0v))
    if scale == 0.0:
        return []
    a, b, c = a0 / scale, b0 / scale, c0v / scale
    if a == 0.0:
        if b == 0.0:
            return []
        return [float(-c / b)]

    four_ac = 4.0 * a * c
    disc = b * b - four_ac
    disc_scale = b * b + abs(four_ac)
    disc_tol = 32.0 * np.finfo(np.float64).eps * max(disc_scale, np.finfo(np.float64).tiny)
    if disc < -disc_tol:
        return []
    if disc < 0.0:
        disc = 0.0
    sqrt_disc = math.sqrt(disc)
    if sqrt_disc == 0.0:
        return [float(-b / (2.0 * a))]

    q = -0.5 * (b + math.copysign(sqrt_disc, b))
    if q == 0.0:
        roots = [-b / (2.0 * a)]
    else:
        roots = [q / a, c / q]
    roots = [float(r) for r in roots if np.isfinite(r)]
    roots.sort()
    if len(roots) == 2 and roots[0] == roots[1]:
        return [roots[0]]
    return roots


@dataclass
class ThresholdCrossingInfo:
    interior_z: np.ndarray
    boundary_z: np.ndarray


def classify_threshold_crossings_z(
    J_curr: np.ndarray,
    J_anchor: np.ndarray,
    theta_A: float,
    boundary_tol: float = 1e-12,
) -> ThresholdCrossingInfo:
    """Classifie les racines des scores A_k(J(z)) = theta_A.

    J(z) = J_anchor + z * (J_curr - J_anchor), z in [0, 1].
    Les racines strictement interieures sont renvoyees dans l'ordre decroissant
    de z ; les racines numeriquement confondues avec 0 ou 1 sont separees.
    """
    j = _fb_validate_J(J_curr)
    anchor = _fb_validate_J(J_anchor)
    if j.shape != anchor.shape:
        raise ValueError("J_curr et J_anchor doivent avoir la meme forme.")
    theta = _finite_scalar("theta_A", theta_A)
    tol = _finite_scalar("boundary_tol", boundary_tol)
    if tol < 0.0:
        raise ValueError("boundary_tol doit etre >= 0.")

    d = j - anchor
    roots_all: List[float] = []
    for k in range(j.shape[0]):
        a = float(np.dot(d[k], d[k]))
        b = float(2.0 * np.dot(anchor[k], d[k]))
        c = float(np.dot(anchor[k], anchor[k]) - theta)
        roots_all.extend(_stable_quadratic_real_roots(a, b, c))

    roots_all.sort(reverse=True)
    interior: List[float] = []
    boundary: List[float] = []
    for root in roots_all:
        if root < -tol or root > 1.0 + tol:
            continue
        if abs(root) <= tol:
            boundary.append(0.0)
        elif abs(root - 1.0) <= tol:
            boundary.append(1.0)
        elif 0.0 < root < 1.0:
            # Deduplication des franchissements simultanes de plusieurs sites.
            if not interior or abs(root - interior[-1]) > max(1e-13, tol):
                interior.append(float(root))

    boundary_unique: List[float] = []
    for root in boundary:
        if root not in boundary_unique:
            boundary_unique.append(root)
    return ThresholdCrossingInfo(
        interior_z=np.asarray(interior, dtype=np.float64),
        boundary_z=np.asarray(boundary_unique, dtype=np.float64),
    )


def solve_threshold_crossings_z(
    J_curr: np.ndarray,
    J_anchor: np.ndarray,
    theta_A: float,
    boundary_tol: float = 1e-12,
) -> np.ndarray:
    """Retourne uniquement les racines strictement interieures a (0, 1)."""
    return classify_threshold_crossings_z(
        J_curr, J_anchor, theta_A, boundary_tol=boundary_tol
    ).interior_z.copy()


def _next_threshold_interval(J: np.ndarray, anchor: np.ndarray,
                             gamma: float, theta_A: float) -> float:
    """Temps jusqu'au prochain changement potentiel de masque (lambda=0)."""
    if gamma <= 0.0:
        return math.inf
    roots = solve_threshold_crossings_z(J, anchor, theta_A)
    if roots.size == 0:
        return math.inf
    # z decroit de 1 vers 0 ; les racines z decroissantes sont des temps croissants.
    times = -np.log(roots) / gamma
    positive = times[times > 1e-14]
    return float(np.min(positive)) if positive.size else math.inf


def simulate_exact_piecewise_lambda0(
    X0: np.ndarray,
    J0: np.ndarray,
    t_max: float,
    mu: float,
    gamma: float,
    beta: float,
    eta: float,
    theta_A: float,
    J_anchor: Optional[np.ndarray] = None,
    fnk_fn: Optional[FitnessFn] = None,
    psi_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = psi_sym,
    rng=None,
) -> SimulationResult:
    """Simulation exacte par morceaux lorsque lambda=0.

    Entre deux mutations, X est constant et la fitness NK ne depend pas de J.
    Les taux sont donc constants sur chaque intervalle ou le masque A_k(J)>
    theta_A ne change pas. Les bornes des intervalles sont calculees en resolvant
    les equations quadratiques exactes du mouvement de dissipation.
    """
    x = _fb_validate_state(X0)
    j = _fb_validate_J(J0)
    if j.shape[0] != x.size:
        raise ValueError("Dimensions incompatibles entre X0 et J0.")
    anchor = j.copy() if J_anchor is None else _fb_validate_J(J_anchor)
    if anchor.shape != j.shape:
        raise ValueError("J_anchor doit avoir la meme forme que J0.")
    t_end = _finite_scalar("t_max", t_max)
    mu_v = _finite_scalar("mu", mu)
    gamma_v = _finite_scalar("gamma", gamma)
    beta_v = _finite_scalar("beta", beta)
    eta_v = _finite_scalar("eta", eta)
    theta_v = _finite_scalar("theta_A", theta_A)
    if t_end < 0.0 or mu_v < 0.0 or gamma_v < 0.0:
        raise ValueError("t_max, mu et gamma doivent etre >= 0.")
    if rng is None:
        rng = np.random.default_rng()
    fn = zero_fnk if fnk_fn is None else fnk_fn
    events: List[SimulationEvent] = []
    t = 0.0
    if t_end == 0.0 or mu_v == 0.0:
        return _make_result(x, j, events, t_end)

    iteration_guard = 0
    while t < t_end:
        iteration_guard += 1
        if iteration_guard > 2_000_000:
            raise RuntimeError("Garde d'iterations depassee dans le simulateur exact.")

        remaining = t_end - t
        dt_cross = _next_threshold_interval(j, anchor, gamma_v, theta_v)
        interval = min(remaining, dt_cross)
        if interval <= 0.0 or not np.isfinite(interval):
            interval = remaining

        # Evalue le masque a l'interieur de l'intervalle, pas exactement sur
        # une frontiere ou la convention stricte '>' attribue le canal ferme.
        probe_dt = 0.5 * interval if interval > 0.0 else 0.0
        j_probe = dissipate_J(j, anchor, gamma_v, probe_dt)
        rates = transition_rates(
            x, j_probe, mu=mu_v, beta=beta_v,
            theta_A=theta_v, lam=0.0, fnk_fn=fn,
        )
        total = float(np.sum(rates))

        if total > 0.0:
            u_wait = _rng_uniform(rng, 0.0, 1.0)
            dt_event = _exp_wait_from_unit_uniform(u_wait, total)
        else:
            dt_event = math.inf

        if dt_event < interval:
            j = dissipate_J(j, anchor, gamma_v, dt_event)
            t += dt_event
            node = _select_channel(rates, rng)
            x, j = mutate_postflip(x, j, node, eta_v, psi_fn=psi_fn)
            events.append(SimulationEvent(t, node, x.copy(), j.copy()))
        else:
            j = dissipate_J(j, anchor, gamma_v, interval)
            t += interval
            if t >= t_end:
                break
            # Recalcul au voisinage immediat de la frontiere : l'intervalle
            # suivant sera evalue en son milieu au prochain tour.

    return _make_result(x, j, events, t)


# ---------------------------------------------------------------------------
# Distances statistiques
# ---------------------------------------------------------------------------


def wasserstein_1d(u_samples: np.ndarray, v_samples: np.ndarray) -> float:
    """Distance de Wasserstein-1 exacte entre deux distributions empiriques 1D."""
    x = np.sort(np.asarray(u_samples, dtype=np.float64).ravel())
    y = np.sort(np.asarray(v_samples, dtype=np.float64).ravel())
    if x.size == 0 or y.size == 0:
        raise ValueError("Les deux echantillons doivent etre non vides.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Les echantillons doivent etre finis.")

    i = j = 0
    mass_x = 1.0 / x.size
    mass_y = 1.0 / y.size
    rem_x = mass_x
    rem_y = mass_y
    distance = 0.0
    eps = 16.0 * np.finfo(np.float64).eps

    while i < x.size and j < y.size:
        transported = min(rem_x, rem_y)
        distance += transported * abs(float(x[i] - y[j]))
        rem_x -= transported
        rem_y -= transported
        if rem_x <= eps * mass_x:
            i += 1
            rem_x = mass_x if i < x.size else 0.0
        if rem_y <= eps * mass_y:
            j += 1
            rem_y = mass_y if j < y.size else 0.0
    return float(distance)


def sliced_wasserstein_1(
    U: np.ndarray,
    V: np.ndarray,
    n_projections: int = 32,
    seed: Optional[int] = None,
) -> float:
    """Sliced Wasserstein-1 empirique, moyenne sur des directions aleatoires."""
    u = np.asarray(U, dtype=np.float64)
    v = np.asarray(V, dtype=np.float64)
    if u.ndim != 2 or v.ndim != 2 or u.shape[1] != v.shape[1]:
        raise ValueError("U et V doivent etre des matrices 2D de meme dimension.")
    if u.shape[0] == 0 or v.shape[0] == 0 or u.shape[1] == 0:
        raise ValueError("U et V doivent contenir des observations et des dimensions.")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("U et V doivent etre finis.")
    n_proj = int(n_projections)
    if n_proj < 1:
        raise ValueError("n_projections doit etre >= 1.")
    rng = np.random.default_rng(seed)
    dirs = rng.normal(0.0, 1.0, size=(n_proj, u.shape[1]))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs /= np.maximum(norms, 1e-15)
    distances = [wasserstein_1d(u @ d, v @ d) for d in dirs]
    return float(np.mean(distances))


def energy_distance(U: np.ndarray, V: np.ndarray,
                    block_size: Optional[int] = None) -> float:
    """Distance d'energie empirique euclidienne, calculee par blocs.

    Convention : racine carree de max(0, 2 E||U-V|| - E||U-U'|| - E||V-V'||),
    les termes intra-echantillon utilisant les paires distinctes (la diagonale
    nulle est incluse dans la somme puis divisee par n(n-1)).
    """
    u = np.asarray(U, dtype=np.float64)
    v = np.asarray(V, dtype=np.float64)
    if u.ndim != 2 or v.ndim != 2 or u.shape[1] != v.shape[1]:
        raise ValueError("U et V doivent etre des matrices 2D de meme dimension.")
    if u.shape[0] == 0 or v.shape[0] == 0:
        raise ValueError("Les echantillons doivent etre non vides.")
    if u.shape[0] < 2 or v.shape[0] < 2:
        raise ValueError("Chaque echantillon doit contenir au moins deux observations.")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("U et V doivent etre finis.")
    bsz = int(block_size) if block_size is not None else 256
    if bsz < 1:
        raise ValueError("block_size doit etre >= 1.")

    def sum_pairwise(A: np.ndarray, B: np.ndarray) -> float:
        total = 0.0
        for start in range(0, A.shape[0], bsz):
            block = A[start:start + bsz]
            diff = block[:, None, :] - B[None, :, :]
            total += float(np.sum(np.linalg.norm(diff, axis=-1)))
        return total

    cross = sum_pairwise(u, v) / float(u.shape[0] * v.shape[0])
    uu = sum_pairwise(u, u) / float(u.shape[0] * (u.shape[0] - 1))
    vv = sum_pairwise(v, v) / float(v.shape[0] * (v.shape[0] - 1))
    value = 2.0 * cross - uu - vv
    # Le terme peut etre tres legerement negatif par arrondi pour des lois egales.
    return float(math.sqrt(max(0.0, value)))


__all__ = [
    "NKLandscape", "FitnessFn", "zero_fnk", "extended_fitness",
    "delta_fitness_k", "transition_rates", "dissipate_J",
    "SimulationEvent", "SimulationResult", "simulate_ogata_thinning",
    "simulate_exact_piecewise_lambda0", "_stable_quadratic_real_roots",
    "ThresholdCrossingInfo", "classify_threshold_crossings_z",
    "solve_threshold_crossings_z", "wasserstein_1d",
    "sliced_wasserstein_1", "energy_distance", "accessibility_A",
    "mutate_postflip", "psi_sym",
]
