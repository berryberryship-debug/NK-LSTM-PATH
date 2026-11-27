"""
V13.5 — Rapport de participation spatiale (PR) des modes singuliers
du Jacobien tangentiel conditionnel J_{FI|B}.

Le Jacobien est calculé avec B_t fixé pendant la différentiation.
Il ne s'agit pas du Jacobien de l'application hybride complète F-I-B.

Conditions comparées :
  - dyn      : état issu de la trajectoire F-I-B ;
  - fourier  : phase commune appliquée à F et I, puis reconstruction de B ;
  - perm     : permutation spatiale commune de F, I et B.

Les contrôles Fourier conservent analytiquement les modules spectraux
individuels de F et I, leur spectre croisé complexe, leurs normes et
leurs moyennes (phase DC fixée à 1), sous réserve des erreurs flottantes.

Le contrôle permutation conserve les valeurs et associations ponctuelles
des champs, mais réarrange leur disposition par rapport au Laplacien.

Les statistiques sont calculées sur les différences appariées entre graines.
"""

import argparse
import csv
import os

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------
# Paramètres du modèle
# ---------------------------------------------------------------------

ALPHA = 0.25
BETA = 0.05
G = 0.3
D_I = 4.0
GAMMA = 1.0
SEUIL = 0.0

N_MODES = 5
N_BOOT = 2000
BOOTSTRAP_SEED = 7

FOURIER_SEED_OFFSET = 2000
PERMUTATION_SEED_OFFSET = 1000

KAPPA_SMIN_THRESHOLD = 1e-12
VERIFY_TOL = 1e-10
JACOBIAN_VERIFY_TOL = 1e-6


# ---------------------------------------------------------------------
# Modèle dynamique
# ---------------------------------------------------------------------

def lap(Z: np.ndarray) -> np.ndarray:
    """Laplacien discret 2D périodique à cinq points."""
    return (
        np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
        - 4.0 * Z
    )


def pas_fib(
    F: np.ndarray,
    I: np.ndarray,
    B: np.ndarray,
    D_F: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mise à jour séquentielle F-I.

    F_{t+1} est calculé en premier.
    I_{t+1} dépend ensuite de F_{t+1}.
    B est maintenu fixe dans cette fonction.
    """
    Q = 2.0 * B - 1.0

    Fn = (
        (1.0 - ALPHA) * F
        + ALPHA * np.tanh(I + G * Q + D_F * lap(F))
    )

    In = (
        (1.0 - BETA) * I
        + BETA * np.tanh(
            Fn + D_I * lap(I) - GAMMA * (I ** 3)
        )
    )

    return Fn, In


def trajectoire(
    seed: int,
    L: int,
    D_F: float,
    temps: tuple
) -> dict:
    """Génère une trajectoire et conserve les états aux temps demandés."""
    if not temps:
        raise ValueError("La liste des temps ne peut pas être vide.")

    rng = np.random.default_rng(seed)

    I = rng.uniform(-0.2, 0.2, (L, L))
    F = np.zeros((L, L), dtype=np.float64)
    B = rng.integers(0, 2, (L, L)).astype(np.float64)

    temps_demandes = set(temps)
    t_max = max(temps)
    out = {}

    for t in range(t_max + 1):
        if t in temps_demandes:
            out[t] = (F.copy(), I.copy(), B.copy())

        F, I = pas_fib(F, I, B, D_F)

        B = (
            F + 0.5 * I - SEUIL > 0.0
        ).astype(np.float64)

    return out


# ---------------------------------------------------------------------
# Matrice du Laplacien
# ---------------------------------------------------------------------

def matrice_laplacienne(L: int) -> sp.csr_matrix:
    """Construit le Laplacien périodique 2D sous forme creuse CSR."""
    if L < 2:
        raise ValueError("L doit être supérieur ou égal à 2.")

    N = L * L
    idx = np.arange(N).reshape(L, L)

    rows = []
    cols = []

    for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rows.append(idx.ravel())
        cols.append(np.roll(idx, shift, axis=(0, 1)).ravel())

    r = np.concatenate(rows)
    c = np.concatenate(cols)

    Lp = sp.coo_matrix(
        (np.ones(len(r), dtype=np.float64), (r, c)),
        shape=(N, N)
    ).tocsr()

    Lp = Lp - 4.0 * sp.identity(
        N,
        format="csr",
        dtype=np.float64
    )

    return Lp.tocsr()


# ---------------------------------------------------------------------
# Jacobien tangentiel conditionnel à B
# ---------------------------------------------------------------------

def jacobien_conditionnel_B(
    F: np.ndarray,
    I: np.ndarray,
    B: np.ndarray,
    D_F: float,
    Lp: sp.csr_matrix
) -> sp.csr_matrix:
    """
    Jacobien J_{FI|B} = d(F_{t+1}, I_{t+1})/d(F_t, I_t),
    avec B_t maintenu fixe pendant la différentiation.
    """
    N = F.size

    Id = sp.identity(N, format="csr", dtype=np.float64)

    a = (
        I
        + G * (2.0 * B - 1.0)
        + D_F * lap(F)
    )

    Fn = (
        (1.0 - ALPHA) * F
        + ALPHA * np.tanh(a)
    )

    b = (
        Fn
        + D_I * lap(I)
        - GAMMA * (I ** 3)
    )

    sa_values = 1.0 - np.tanh(a) ** 2
    sb_values = 1.0 - np.tanh(b) ** 2

    sa = sp.diags(sa_values.ravel(), format="csr")
    sb = sp.diags(sb_values.ravel(), format="csr")
    dI2 = sp.diags((I ** 2).ravel(), format="csr")

    dFF = (
        (1.0 - ALPHA) * Id
        + ALPHA * D_F * (sa @ Lp)
    )

    dFI = ALPHA * sa

    # Chaîne F_t -> F_{t+1} -> I_{t+1}.
    dIF = BETA * (sb @ dFF)

    dII = (
        (1.0 - BETA) * Id
        + BETA * sb @ (
            dFI + D_I * Lp - 3.0 * GAMMA * dI2
        )
    )

    return sp.bmat(
        [[dFF, dFI], [dIF, dII]],
        format="csr"
    )


# ---------------------------------------------------------------------
# Participation ratio spatial
# ---------------------------------------------------------------------

def pr(w: np.ndarray) -> float:
    """
    PR(w) = (sum_i w_i)^2 / sum_i(w_i^2).

    Pour des poids non négatifs non tous nuls, PR est compris
    entre 1 et le nombre de composantes spatiales.
    """
    s1 = float(np.sum(w))
    s2 = float(np.sum(w ** 2))

    if s2 <= 1e-15:
        return np.nan

    return (s1 * s1) / s2


def mesures_PR(
    J: sp.csr_matrix,
    N: int,
    n_modes: int = N_MODES,
    use_sparse: bool = False
) -> dict:
    """
    Mesure le PR spatial des vecteurs singuliers droits dominants
    et associés aux plus petites valeurs singulières.
    """
    dim = 2 * N

    if n_modes < 1:
        raise ValueError("n_modes doit être supérieur ou égal à 1.")

    if n_modes >= dim:
        raise ValueError("n_modes doit être strictement inférieur à 2N.")

    def poids_spatial(v):
        return v[:N] ** 2 + v[N:] ** 2

    def decomposition_dense():
        _, s, Vt = la.svd(
            J.toarray(),
            full_matrices=False,
            check_finite=True
        )

        Vt_top = Vt[:n_modes]
        Vt_bot = Vt[::-1][:n_modes]

        return Vt_top, Vt_bot, float(s[0]), float(s[-1])

    # Dense de référence ou dimension trop petite pour svds.
    if not use_sparse or dim <= 2 * n_modes + 2:
        Vt_top, Vt_bot, s_max, s_min = decomposition_dense()

    else:
        try:
            # Plus grandes valeurs singulières.
            _, s_hi, Vt_hi = spla.svds(
                J,
                k=n_modes,
                which="LM"
            )
            idx_hi = np.argsort(s_hi)[::-1]
            Vt_top = Vt_hi[idx_hi]
            s_max = float(s_hi[idx_hi[0]])

            # Plus petites valeurs singulières.
            _, s_lo, Vt_lo = spla.svds(
                J,
                k=n_modes,
                which="SM",
                tol=1e-5,
                maxiter=dim * 10
            )
            idx_lo = np.argsort(s_lo)
            Vt_bot = Vt_lo[idx_lo]
            s_min = float(s_lo[idx_lo[0]])

        except (spla.ArpackNoConvergence, spla.ArpackError):
            # Repli dense uniquement pour les exceptions ARPACK interceptées.
            Vt_top, Vt_bot, s_max, s_min = decomposition_dense()

    hi_pr = [pr(poids_spatial(v)) for v in Vt_top]
    lo_pr = [pr(poids_spatial(v)) for v in Vt_bot]

    if s_min > KAPPA_SMIN_THRESHOLD:
        kappa = s_max / s_min
    else:
        kappa = np.inf

    log10_kappa = (
        float(np.log10(kappa))
        if np.isfinite(kappa)
        else np.nan
    )

    return {
        "PR_top1": hi_pr[0] / N,
        "PR_top5": float(np.mean(hi_pr)) / N,
        "PR_bot1": lo_pr[0] / N,
        "PR_bot5": float(np.mean(lo_pr)) / N,
        "log10_kappa": log10_kappa
    }


# ---------------------------------------------------------------------
# Contrôle par permutation
# ---------------------------------------------------------------------

def substitut_permutation(
    F: np.ndarray,
    I: np.ndarray,
    B: np.ndarray,
    seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applique une même permutation spatiale aux trois champs.

    Les valeurs et associations ponctuelles F-I-B sont conservées.
    La disposition sur le réseau est réarrangée, ce qui modifie
    les relations de voisinage relativement au Laplacien périodique.
    """
    rng = np.random.default_rng(seed)
    p = rng.permutation(F.size)

    return tuple(
        X.ravel()[p].reshape(F.shape)
        for X in (F, I, B)
    )


# ---------------------------------------------------------------------
# Multiplicateur Fourier hermitien
# ---------------------------------------------------------------------

def multiplicateur_hermitien(
    L: int,
    seed: int
) -> np.ndarray:
    """
    Construit phi avec |phi(k)|=1 et phi(-k)=conj(phi(k)).

    La phase DC est fixée à 1 afin de préserver les moyennes.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((L, L))

    W = np.fft.fft2(w)
    mag = np.abs(W)

    phi = np.ones_like(W, dtype=np.complex128)
    mask = mag >= 1e-15
    phi[mask] = W[mask] / mag[mask]

    phi[0, 0] = 1.0 + 0.0j

    return phi


# ---------------------------------------------------------------------
# Reconstruction de B par quantile
# ---------------------------------------------------------------------

def reconstruire_B_quantile(
    S: np.ndarray,
    n_actifs: int
) -> np.ndarray:
    """
    Reconstruit B avec exactement n_actifs sites actifs.

    Les valeurs les plus élevées de S sont sélectionnées.
    Le tri stable rend la sélection déterministe en cas d'égalité.
    """
    N = S.size

    if not isinstance(n_actifs, (int, np.integer)):
        raise TypeError("n_actifs doit être un entier.")

    if not 0 <= n_actifs <= N:
        raise ValueError("n_actifs doit être compris entre 0 et N.")

    B_flat = np.zeros(N, dtype=np.float64)

    if n_actifs == 0:
        return B_flat.reshape(S.shape)

    if n_actifs == N:
        return np.ones(S.shape, dtype=np.float64)

    ordre = np.argsort(S.ravel(), kind="stable")
    indices_actifs = ordre[-n_actifs:]
    B_flat[indices_actifs] = 1.0

    return B_flat.reshape(S.shape)


# ---------------------------------------------------------------------
# Contrôle Fourier
# ---------------------------------------------------------------------

def substitut_fourier(
    F: np.ndarray,
    I: np.ndarray,
    B: np.ndarray,
    seed: int,
    mode: str = "quantile"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applique une phase commune à F et I.

    Le mode quantile conserve exactement le nombre de sites actifs de B.
    Le mode theta applique le seuil fixe SEUIL ; la densité peut varier.
    """
    if F.shape != I.shape or F.shape != B.shape:
        raise ValueError("F, I et B doivent avoir la même forme.")

    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("Les champs doivent être des matrices carrées.")

    L = F.shape[0]
    phi = multiplicateur_hermitien(L, seed)

    F_hat = np.fft.fft2(F)
    I_hat = np.fft.fft2(I)

    Fs_complex = np.fft.ifft2(F_hat * phi)
    Is_complex = np.fft.ifft2(I_hat * phi)

    # Les résidus imaginaires sont contrôlés séparément par --verif.
    Fs = Fs_complex.real
    Is = Is_complex.real

    S_sur = Fs + 0.5 * Is

    if mode == "theta":
        Bs = (S_sur - SEUIL > 0.0).astype(np.float64)

    elif mode == "quantile":
        Bs = reconstruire_B_quantile(
            S_sur,
            int(np.count_nonzero(B))
        )

    else:
        raise ValueError(f"Mode B inconnu : {mode}")

    return Fs, Is, Bs


# ---------------------------------------------------------------------
# Statistiques appariées
# ---------------------------------------------------------------------

def stats_apparie(
    diff: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED
) -> dict:
    """
    Bootstrap percentile sur les différences appariées entre graines.

    Les unités statistiques sont les graines, pas les sites du réseau.
    """
    x = np.asarray(diff, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 2:
        return {
            "n": n,
            "moy": np.nan,
            "sd": np.nan,
            "lo": np.nan,
            "hi": np.nan,
            "dz": np.nan,
            "pos": int((x > 0).sum())
        }

    if n_boot < 1:
        raise ValueError("n_boot doit être supérieur ou égal à 1.")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)

    lo, hi = np.quantile(boot_means, [0.025, 0.975])

    sd = float(x.std(ddof=1))
    mean = float(x.mean())

    if sd > 1e-12:
        dz = mean / sd
    elif abs(mean) <= 1e-12:
        dz = np.nan
    else:
        dz = np.copysign(np.inf, mean)

    return {
        "n": n,
        "moy": mean,
        "sd": sd,
        "lo": float(lo),
        "hi": float(hi),
        "dz": float(dz),
        "pos": int((x > 0).sum())
    }


# ---------------------------------------------------------------------
# Diagnostics Fourier avec critères de réussite
# ---------------------------------------------------------------------

def verif_substitution_fourier(
    L: int = 21,
    seed: int = 42
) -> None:
    """
    Vérifie numériquement les invariants Fourier sur des champs
    réels de test indépendants d'une trajectoire.
    """
    print("=== Validation numérique du contrôle Fourier ===")

    phi = multiplicateur_hermitien(L, seed)

    ecart_unimod = float(np.max(np.abs(np.abs(phi) - 1.0)))

    idx0 = np.arange(L)[:, None]
    idx1 = np.arange(L)[None, :]
    phi_neg = phi[(-idx0) % L, (-idx1) % L]
    err_herm = float(np.max(np.abs(phi_neg - np.conj(phi))))

    rng = np.random.default_rng(seed)
    F = rng.standard_normal((L, L))
    I = rng.standard_normal((L, L))

    F_hat = np.fft.fft2(F)
    I_hat = np.fft.fft2(I)
    cross_orig = F_hat * np.conj(I_hat)

    Fs_complex = np.fft.ifft2(F_hat * phi)
    Is_complex = np.fft.ifft2(I_hat * phi)

    err_imag_F = float(np.max(np.abs(Fs_complex.imag)))
    err_imag_I = float(np.max(np.abs(Is_complex.imag)))

    Fs = Fs_complex.real
    Is = Is_complex.real

    Fs_hat_recon = np.fft.fft2(Fs)
    Is_hat_recon = np.fft.fft2(Is)

    cross_sur = Fs_hat_recon * np.conj(Is_hat_recon)

    err_mod_F = float(np.max(np.abs(np.abs(Fs_hat_recon) - np.abs(F_hat))))
    err_mod_I = float(np.max(np.abs(np.abs(Is_hat_recon) - np.abs(I_hat))))
    err_cross = float(np.max(np.abs(cross_sur - cross_orig)))

    err_norm_F = float(abs(np.linalg.norm(Fs) - np.linalg.norm(F)))
    err_norm_I = float(abs(np.linalg.norm(Is) - np.linalg.norm(I)))

    err_mean_F = float(abs(Fs.mean() - F.mean()))
    err_mean_I = float(abs(Is.mean() - I.mean()))

    diagnostics = {
        "unimodularite": ecart_unimod,
        "symetrie_hermitienne": err_herm,
        "residu_imaginaire_F": err_imag_F,
        "residu_imaginaire_I": err_imag_I,
        "modules_spectraux_F": err_mod_F,
        "modules_spectraux_I": err_mod_I,
        "spectre_croise_complexe": err_cross,
        "norme_F": err_norm_F,
        "norme_I": err_norm_I,
        "moyenne_F": err_mean_F,
        "moyenne_I": err_mean_I
    }

    for name, value in diagnostics.items():
        print(f"{name:<32}: {value:.3e}")

    failed = {
        name: value
        for name, value in diagnostics.items()
        if value > VERIFY_TOL
    }

    if failed:
        details = ", ".join(
            f"{name}={value:.3e}"
            for name, value in failed.items()
        )
        raise RuntimeError(
            "Échec du contrôle Fourier (tolérance "
            f"{VERIFY_TOL:.1e}) : {details}"
        )

    print("Résultat : contrôles Fourier réussis sur les champs de test.")
    print(
        "Note : ces diagnostics ne remplacent pas la vérification "
        "des résultats d'une campagne sur trajectoires."
    )


# ---------------------------------------------------------------------
# Validation du Jacobien conditionnel
# ---------------------------------------------------------------------

def verif_jacobien() -> None:
    """
    Compare le Jacobien analytique conditionnel à B fixé
    avec une approximation par différences finies centrées.
    """
    L = 7
    N = L * L
    rng = np.random.default_rng(42)

    F = rng.uniform(-0.5, 0.5, (L, L))
    I = rng.uniform(-0.5, 0.5, (L, L))
    B = (F + 0.5 * I > SEUIL).astype(np.float64)

    Lp = matrice_laplacienne(L)

    print("\n=== Validation du Jacobien conditionnel J_{FI|B} ===")
    print("Comparaison analytique / différences finies centrées")

    for D_F in (0.0, 0.1, 0.4):
        Jf = np.zeros((2 * N, 2 * N), dtype=np.float64)
        eps = 1e-7

        for c in range(2 * N):
            dF = np.zeros(N)
            dI = np.zeros(N)

            if c < N:
                dF[c] = eps
            else:
                dI[c - N] = eps

            Fp, Ip = pas_fib(
                F + dF.reshape(L, L),
                I + dI.reshape(L, L),
                B,
                D_F
            )

            Fm, Im = pas_fib(
                F - dF.reshape(L, L),
                I - dI.reshape(L, L),
                B,
                D_F
            )

            Jf[:, c] = np.concatenate([
                (Fp - Fm).ravel(),
                (Ip - Im).ravel()
            ]) / (2.0 * eps)

        Ja = jacobien_conditionnel_B(
            F, I, B, D_F, Lp
        ).toarray()

        denominateur = np.linalg.norm(Jf, ord="fro")
        if denominateur <= 0.0:
            raise RuntimeError("Norme du Jacobien numérique nulle.")

        err = float(
            np.linalg.norm(Ja - Jf, ord="fro") / denominateur
        )

        print(
            f"D_F = {D_F:<4} | erreur relative Frobenius = {err:.3e}"
        )

        if not np.isfinite(err) or err > JACOBIAN_VERIFY_TOL:
            raise RuntimeError(
                "Échec de la vérification du Jacobien pour "
                f"D_F={D_F} : erreur={err:.3e}, "
                f"tolérance={JACOBIAN_VERIFY_TOL:.1e}"
            )

    print("Résultat : vérification du Jacobien réussie.")


# ---------------------------------------------------------------------
# Fonctions auxiliaires
# ---------------------------------------------------------------------

def moyenne_finie(rows, key):
    """Moyenne des valeurs finies d'une colonne."""
    values = np.asarray([r[key] for r in rows], dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def selectionner(lignes, temps, cas):
    """Sélectionne les lignes et les ordonne par graine."""
    return sorted(
        [r for r in lignes if r["t"] == temps and r["cas"] == cas],
        key=lambda r: r["seed"]
    )


def valeurs_colonne(rows, key):
    return np.asarray([r[key] for r in rows], dtype=np.float64)


def parser_temps(texte: str) -> tuple:
    try:
        temps = tuple(int(t.strip()) for t in texte.split(","))
    except ValueError as exc:
        raise ValueError(
            "Format invalide pour --times. Exemple : 5,100,500"
        ) from exc

    if not temps:
        raise ValueError("Au moins un instant doit être fourni.")

    if any(t < 0 for t in temps):
        raise ValueError("Les instants doivent être positifs ou nuls.")

    if len(set(temps)) != len(temps):
        raise ValueError("Les instants ne doivent pas être dupliqués.")

    return tuple(sorted(temps))


def valider_arguments(args):
    if args.L < 2:
        raise ValueError("L doit être supérieur ou égal à 2.")

    if args.seeds < 1:
        raise ValueError("seeds doit être supérieur ou égal à 1.")

    if not np.isfinite(args.D_F):
        raise ValueError("D_F doit être une valeur finie.")

    if args.n_boot < 1:
        raise ValueError("n_boot doit être supérieur ou égal à 1.")

    return parser_temps(args.times)


def format_stat(q):
    return (
        f"{q['moy']:+.4f} "
        f"[{q['lo']:+.4f}, {q['hi']:+.4f}] "
        f"{q['dz']:+.2f} "
        f"{q['pos']}/{q['n']}"
    )


# ---------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Mesure du PR spatial des modes singuliers "
            "du Jacobien tangentiel conditionnel J_{FI|B}."
        )
    )

    parser.add_argument(
        "--verif",
        action="store_true",
        help="Vérifie le contrôle Fourier et le Jacobien conditionnel."
    )
    parser.add_argument("--L", type=int, default=21)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--times", default="5,100,500")
    parser.add_argument("--D_F", type=float, default=0.1)
    parser.add_argument(
        "--B-mode",
        choices=["quantile", "theta"],
        default="quantile"
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Active la SVD partielle creuse exploratoire."
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOT,
        help="Nombre de réplications bootstrap."
    )
    parser.add_argument(
        "--out",
        default="results/v13_5_pr.csv"
    )

    args = parser.parse_args()

    try:
        temps = valider_arguments(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.verif:
        verif_substitution_fourier(L=args.L)
        verif_jacobien()
        return

    L = args.L
    N = L * L
    Lp = matrice_laplacienne(L)

    dossier = os.path.dirname(args.out)
    if dossier:
        os.makedirs(dossier, exist_ok=True)

    mode_svd = (
        "Partielle creuse LM/SM (exploratoire)"
        if args.sparse
        else "Dense de référence"
    )

    print(
        f"L={L} (N={N}) | graines={args.seeds} | "
        f"D_F={args.D_F} | B-mode={args.B_mode} | SVD={mode_svd}"
    )
    print(f"Temps mesurés : {temps}")
    print("Jacobien : J_{FI|B}, avec B fixé pendant la différentiation.")

    mesures = [
        "PR_top1",
        "PR_top5",
        "PR_bot1",
        "PR_bot5",
        "log10_kappa"
    ]

    lignes = []

    for s in range(args.seeds):
        traj = trajectoire(
            seed=s,
            L=L,
            D_F=args.D_F,
            temps=temps
        )

        for t in temps:
            F, I, B = traj[t]

            cas = {
                "dyn": (F, I, B),
                "fourier": substitut_fourier(
                    F,
                    I,
                    B,
                    FOURIER_SEED_OFFSET + s,
                    mode=args.B_mode
                ),
                "perm": substitut_permutation(
                    F,
                    I,
                    B,
                    PERMUTATION_SEED_OFFSET + s
                )
            }

            for nom, (Fx, Ix, Bx) in cas.items():
                J = jacobien_conditionnel_B(
                    Fx, Ix, Bx, args.D_F, Lp
                )

                m = mesures_PR(
                    J,
                    N,
                    n_modes=N_MODES,
                    use_sparse=args.sparse
                )

                m.update({
                    "seed": s,
                    "t": t,
                    "cas": nom,
                    "densB": float(Bx.mean()),
                    "lapF": float(np.linalg.norm(lap(Fx)))
                })

                lignes.append(m)

        print(
            f"  [Progression] Graine {s + 1}/{args.seeds} calculée",
            flush=True
        )

    colonnes = ["seed", "t", "cas", "densB", "lapF"] + mesures

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colonnes)
        writer.writeheader()
        writer.writerows(lignes)

    for t in temps:
        d = selectionner(lignes, t, "dyn")
        fo = selectionner(lignes, t, "fourier")
        pe = selectionner(lignes, t, "perm")

        print("\n" + "=" * 104)
        print(f"INSTANT t = {t}")
        print("=" * 104)

        print(
            "Densité B moyenne : "
            f"dyn {moyenne_finie(d, 'densB'):.6f} | "
            f"fourier {moyenne_finie(fo, 'densB'):.6f} | "
            f"perm {moyenne_finie(pe, 'densB'):.6f}"
        )
        print(
            "||Lap F|| moyenne : "
            f"dyn {moyenne_finie(d, 'lapF'):.6f} | "
            f"fourier {moyenne_finie(fo, 'lapF'):.6f} | "
            f"perm {moyenne_finie(pe, 'lapF'):.6f}"
        )

        print("-" * 104)
        print(
            f"{'Mesure':<14} | {'Dyn moy.':<10} | "
            f"{'Dyn-Fourier : moy. [IC95] dz +/n':<43} | "
            f"{'Dyn-Perm : moy. [IC95] dz +/n':<39}"
        )
        print("-" * 104)

        for k in mesures:
            vals_dyn = valeurs_colonne(d, k)
            vals_fourier = valeurs_colonne(fo, k)
            vals_perm = valeurs_colonne(pe, k)

            s1 = stats_apparie(
                vals_dyn - vals_fourier,
                n_boot=args.n_boot,
                seed=BOOTSTRAP_SEED
            )
            s2 = stats_apparie(
                vals_dyn - vals_perm,
                n_boot=args.n_boot,
                seed=BOOTSTRAP_SEED + 1
            )

            dyn_mean = moyenne_finie(d, k)

            print(
                f"{k:<14} | {dyn_mean:<10.4f} | "
                f"{format_stat(s1):<43} | {format_stat(s2):<39}"
            )

    print(f"\nDonnées détaillées enregistrées dans : {args.out}")
    print("\nRappel d'interprétation :")
    print(
        "  - Dyn - Fourier : contraste avec modules spectraux "
        "individuels et spectre croisé complexe de F et I conservés."
    )
    print(
        "  - Dyn - Permutation : contraste avec réarrangement spatial "
        "commun des champs."
    )
    print(
        "  - Les différences bootstrap sont calculées sur les graines, "
        "pas sur les sites."
    )
    print("  - Le Jacobien est conditionnel à B fixé.")
    print(
        "  - Le seuil de conditionnement sur s_min est "
        f"{KAPPA_SMIN_THRESHOLD:.1e}."
    )


if __name__ == "__main__":
    main()
