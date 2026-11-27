"""
V13.5 — Rapport de participation spatiale (PR) des modes singuliers
du Jacobien tangentiel conditionnel J_{FI|B}.

Cadre méthodologique
--------------------
1. Jacobien :
   J_{FI|B} = d(F_{t+1}, I_{t+1}) / d(F_t, I_t), avec B_t fixé.

2. Contrôle Fourier :
   - multiplicateur unimodulaire |phi(k)| = 1 ;
   - symétrie hermitienne à la précision flottante ;
   - phase commune appliquée à F et I ;
   - conservation analytique des spectres individuels,
     du spectre croisé complexe et des normes ;
   - reconstruction de B par conservation exacte de sa densité
     en mode quantile.

3. Hiérarchie des contrôles :
   Dynamique -> Substitut Fourier -> Permutation.

4. Analyse spectrale :
   - SVD dense de référence ;
   - SVD creuse partielle à titre exploratoire ;
   - repli dense en cas d'échec de la résolution creuse.

5. Statistiques :
   - différences appariées par graine ;
   - bootstrap percentile descriptif ;
   - d_z descriptif.

Périmètre d'interprétation
--------------------------
Le contraste dynamique - Fourier mesure une différence de participation
spatiale entre la dynamique et un contrôle conservant les invariants
spectraux de F et I.

Il ne permet pas, à lui seul, d'attribuer cette différence exclusivement
à des corrélations d'ordre supérieur : la reconstruction de B et son
couplage au Jacobien contribuent également au contraste.

Le Jacobien étudié est conditionnel à B fixé. Il ne représente pas
le Jacobien classique de l'application hybride complète F-I-B.

Usage
-----
python v13_5_holographie_PR.py --verif
python v13_5_holographie_PR.py --L 21 --seeds 10
python v13_5_holographie_PR.py --L 21 --seeds 10 --sparse
python v13_5_holographie_PR.py --L 21 --seeds 10 --B-mode theta
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
):
    """
    Mise à jour séquentielle du système F-I-B.

    F_{t+1} est calculé en premier.
    I_{t+1} dépend ensuite de F_{t+1}.
    B_{t+1} est actualisé dans trajectoire().
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
):
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
            out[t] = (
                F.copy(),
                I.copy(),
                B.copy()
            )

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
    N = L * L
    idx = np.arange(N).reshape(L, L)

    rows = []
    cols = []

    for shift in (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1)
    ):
        rows.append(idx.ravel())
        cols.append(
            np.roll(idx, shift, axis=(0, 1)).ravel()
        )

    r = np.concatenate(rows)
    c = np.concatenate(cols)

    Lp = sp.coo_matrix(
        (
            np.ones(len(r), dtype=np.float64),
            (r, c)
        ),
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
    Jacobien tangentiel conditionnel :

        J_{FI|B} = d(F_{t+1}, I_{t+1}) / d(F_t, I_t)

    B_t est maintenu fixé pendant la différentiation.

    Ce Jacobien décrit la branche locale du système correspondant
    à une configuration donnée de B_t. Il ne représente pas le
    Jacobien classique de l'application hybride complète F-I-B.
    """
    N = F.size

    Id = sp.identity(
        N,
        format="csr",
        dtype=np.float64
    )

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

    # Dérivée de tanh(x) :
    # sech²(x) = 1 - tanh²(x).
    # Cette écriture évite les débordements éventuels de cosh(x).

    sa_values = 1.0 - np.tanh(a) ** 2
    sb_values = 1.0 - np.tanh(b) ** 2

    sa = sp.diags(
        sa_values.ravel(),
        format="csr"
    )

    sb = sp.diags(
        sb_values.ravel(),
        format="csr"
    )

    dFF = (
        (1.0 - ALPHA) * Id
        + ALPHA * D_F * (sa @ Lp)
    )

    dFI = ALPHA * sa

    dIF = BETA * (sb @ dFF)

    dII = (
        (1.0 - BETA) * Id
        + BETA * (
            sb @ (
                dFI
                + D_I * Lp
                - 3.0 * GAMMA
                * sp.diags(
                    (I ** 2).ravel(),
                    format="csr"
                )
            )
        )
    )

    return sp.bmat(
        [
            [dFF, dFI],
            [dIF, dII]
        ],
        format="csr"
    )


# ---------------------------------------------------------------------
# Participation ratio spatial
# ---------------------------------------------------------------------

def pr(w: np.ndarray) -> float:
    """
    Participation ratio :

        PR(w) = (sum_i w_i)^2 / sum_i(w_i^2)

    La mesure est comprise entre 1 et le nombre de sites,
    pour des poids non négatifs non tous nuls.
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
    Mesure le PR spatial des modes singuliers dominants et contractés.

    Pour chaque vecteur singulier droit v, le poids spatial au site i est :

        w_i = v_F,i² + v_I,i²

    Le PR est ensuite normalisé par N, le nombre de sites.
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
            full_matrices=False
        )

        Vt_top = Vt[:n_modes]
        Vt_bot = Vt[::-1][:n_modes]

        return (
            Vt_top,
            Vt_bot,
            float(s[0]),
            float(s[-1])
        )

    # Référence dense ou cas où la SVD partielle n'est pas applicable.
    if (
        not use_sparse
        or dim <= 2 * n_modes + 2
    ):
        Vt_top, Vt_bot, s_max, s_min = decomposition_dense()

    else:
        try:
            # Modes dominants : plus grandes valeurs singulières.
            _, s_hi, Vt_hi = spla.svds(
                J,
                k=n_modes,
                which="LM"
            )

            idx_hi = np.argsort(s_hi)[::-1]

            Vt_top = Vt_hi[idx_hi]
            s_max = float(s_hi[idx_hi[0]])

            # Modes contractés : plus petites valeurs singulières.
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

        except (
            spla.ArpackNoConvergence,
            spla.ArpackError
        ):
            # Repli dense : résultat de référence si le calcul
            # creux partiel ne converge pas.
            Vt_top, Vt_bot, s_max, s_min = decomposition_dense()

    hi_pr = [
        pr(poids_spatial(v))
        for v in Vt_top
    ]

    lo_pr = [
        pr(poids_spatial(v))
        for v in Vt_bot
    ]

    if s_min > 1e-12:
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
):
    """
    Applique une même permutation spatiale aux trois champs.

    Cette opération conserve les valeurs des champs et leurs
    associations locales entre F, I et B, mais détruit leur
    organisation spatiale sur le réseau.
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
    Construit un multiplicateur de phase tel que :

        |phi(k)| = 1

        phi(-k) = conj(phi(k))

    La symétrie est satisfaite à la précision flottante.

    Le coefficient DC est fixé à 1 afin de préserver la composante
    moyenne des champs.
    """
    rng = np.random.default_rng(seed)

    w = rng.standard_normal((L, L))

    W = np.fft.fft2(w)
    mag = np.abs(W)

    phi = np.ones_like(
        W,
        dtype=np.complex128
    )

    mask = mag >= 1e-15

    phi[mask] = W[mask] / mag[mask]

    # Composante continue : phase nulle.
    phi[0, 0] = 1.0 + 0.0j

    return phi


# ---------------------------------------------------------------------
# Reconstruction de B par quantile
# ---------------------------------------------------------------------

def reconstruire_B_quantile(
    S: np.ndarray,
    densite_cible: float
) -> np.ndarray:
    """
    Reconstruit un champ binaire ayant exactement le même nombre
    de sites actifs que la densité cible.

    Le tri stable garantit un résultat déterministe en présence
    de valeurs ex æquo.
    """
    N = S.size

    n_actifs = int(
        round(densite_cible * N)
    )

    n_actifs = max(
        0,
        min(N, n_actifs)
    )

    B_flat = np.zeros(
        N,
        dtype=np.float64
    )

    if n_actifs == 0:
        return B_flat.reshape(S.shape)

    if n_actifs == N:
        return np.ones(
            S.shape,
            dtype=np.float64
        )

    ordre = np.argsort(
        S.ravel(),
        kind="stable"
    )

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
):
    """
    Applique une phase commune aux spectres de F et I.

    Conservation analytique :
        |F_hat_sur| = |F_hat|
        |I_hat_sur| = |I_hat|
        F_hat_sur * conj(I_hat_sur)
            = F_hat * conj(I_hat)

    B est ensuite reconstruit :
        - mode quantile : densité exactement conservée ;
        - mode theta : seuil fixe appliqué au champ transformé.
    """
    L = F.shape[0]

    phi = multiplicateur_hermitien(
        L,
        seed
    )

    F_hat = np.fft.fft2(F)
    I_hat = np.fft.fft2(I)

    F_hat_sur = F_hat * phi
    I_hat_sur = I_hat * phi

    # IFFT complexe conservée temporairement pour mesurer
    # le résidu imaginaire avant extraction de la partie réelle.
    Fs_complex = np.fft.ifft2(F_hat_sur)
    Is_complex = np.fft.ifft2(I_hat_sur)

    Fs = Fs_complex.real
    Is = Is_complex.real

    S_sur = Fs + 0.5 * Is

    if mode == "theta":

        Bs = (
            S_sur - SEUIL > 0.0
        ).astype(np.float64)

    elif mode == "quantile":

        Bs = reconstruire_B_quantile(
            S_sur,
            float(B.mean())
        )

    else:
        raise ValueError(
            f"Mode B inconnu : {mode}"
        )

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
    Bootstrap percentile descriptif sur les différences appariées
    entre graines.

    Les unités statistiques sont les graines, et non les sites
    individuels du réseau.
    """
    x = np.asarray(
        diff,
        dtype=np.float64
    )

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
        raise ValueError(
            "n_boot doit être supérieur ou égal à 1."
        )

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        n,
        size=(n_boot, n)
    )

    boot_means = x[idx].mean(axis=1)

    lo, hi = np.quantile(
        boot_means,
        [0.025, 0.975]
    )

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
# Diagnostics Fourier
# ---------------------------------------------------------------------

def verif_substitution_fourier(
    L: int = 21,
    seed: int = 42
):
    """
    Vérifie :
      1. l'unimodularité de phi ;
      2. sa symétrie hermitienne ;
      3. le résidu imaginaire des IFFT ;
      4. les spectres individuels ;
      5. le spectre croisé complexe ;
      6. les normes spatiales.
    """
    print("=== Validation numérique du contrôle Fourier ===")

    phi = multiplicateur_hermitien(
        L,
        seed
    )

    # 1. Unimodularité.
    ecart_unimod = np.max(
        np.abs(np.abs(phi) - 1.0)
    )

    print(
        "Écart max à l'unimodularité ||phi| - 1|       : "
        f"{ecart_unimod:.2e}"
    )

    # 2. Symétrie hermitienne.
    idx = (
        np.arange(L)[:, None],
        np.arange(L)[None, :]
    )

    idx_neg = (
        (-idx[0]) % L,
        (-idx[1]) % L
    )

    phi_neg = phi[idx_neg]

    err_herm = np.max(
        np.abs(phi_neg - np.conj(phi))
    )

    print(
        "Écart max symétrie hermitienne                : "
        f"{err_herm:.2e}"
    )

    # 3. Champs tests réels.
    rng = np.random.default_rng(seed)

    F = rng.standard_normal((L, L))
    I = rng.standard_normal((L, L))

    F_hat = np.fft.fft2(F)
    I_hat = np.fft.fft2(I)

    cross_orig = F_hat * np.conj(I_hat)

    # 4. Transformation.
    Fs_hat = F_hat * phi
    Is_hat = I_hat * phi

    Fs_complex = np.fft.ifft2(Fs_hat)
    Is_complex = np.fft.ifft2(Is_hat)

    err_imag_F = np.max(
        np.abs(Fs_complex.imag)
    )

    err_imag_I = np.max(
        np.abs(Is_complex.imag)
    )

    Fs = Fs_complex.real
    Is = Is_complex.real

    print(
        "Résidu imaginaire IFFT(F)                    : "
        f"{err_imag_F:.2e}"
    )

    print(
        "Résidu imaginaire IFFT(I)                    : "
        f"{err_imag_I:.2e}"
    )

    # 5. Recalcul des spectres après reconstruction réelle.
    Fs_hat_reconstruit = np.fft.fft2(Fs)
    Is_hat_reconstruit = np.fft.fft2(Is)

    cross_sur = (
        Fs_hat_reconstruit
        * np.conj(Is_hat_reconstruit)
    )

    err_mod_F = np.max(
        np.abs(
            np.abs(Fs_hat_reconstruit)
            - np.abs(F_hat)
        )
    )

    err_mod_I = np.max(
        np.abs(
            np.abs(Is_hat_reconstruit)
            - np.abs(I_hat)
        )
    )

    err_cross = np.max(
        np.abs(cross_sur - cross_orig)
    )

    print(
        "Conservation |F_hat|                         : "
        f"{err_mod_F:.2e}"
    )

    print(
        "Conservation |I_hat|                         : "
        f"{err_mod_I:.2e}"
    )

    print(
        "Conservation spectre croisé complexe         : "
        f"{err_cross:.2e}"
    )

    # 6. Conservation des normes spatiales.
    norm_F_orig = np.linalg.norm(F)
    norm_F_sur = np.linalg.norm(Fs)

    norm_I_orig = np.linalg.norm(I)
    norm_I_sur = np.linalg.norm(Is)

    err_norm_F = abs(
        norm_F_sur - norm_F_orig
    )

    err_norm_I = abs(
        norm_I_sur - norm_I_orig
    )

    print(
        "Conservation norme spatiale F                : "
        f"{err_norm_F:.2e}"
    )

    print(
        "Conservation norme spatiale I                : "
        f"{err_norm_I:.2e}"
    )

    print(
        "\nCes écarts sont des diagnostics numériques ; "
        "ils ne remplacent pas la vérification des résultats "
        "sur les trajectoires réelles."
    )


# ---------------------------------------------------------------------
# Validation du Jacobien conditionnel
# ---------------------------------------------------------------------

def verif_jacobien():
    """
    Compare le Jacobien analytique conditionnel à B fixé
    avec une approximation par différences finies centrées.
    """
    L = 7
    N = L * L

    rng = np.random.default_rng(42)

    F = rng.uniform(
        -0.5,
        0.5,
        (L, L)
    )

    I = rng.uniform(
        -0.5,
        0.5,
        (L, L)
    )

    B = (
        F + 0.5 * I > SEUIL
    ).astype(np.float64)

    Lp = matrice_laplacienne(L)

    print(
        "\n=== Validation du Jacobien conditionnel "
        "J_{FI|B} ==="
    )

    print(
        "Comparaison analytique / différences finies centrées"
    )

    for D_F in (0.0, 0.1, 0.4):

        Jf = np.zeros(
            (2 * N, 2 * N),
            dtype=np.float64
        )

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

            Jf[:, c] = (
                np.concatenate([
                    (Fp - Fm).ravel(),
                    (Ip - Im).ravel()
                ])
                / (2.0 * eps)
            )

        Ja = jacobien_conditionnel_B(
            F,
            I,
            B,
            D_F,
            Lp
        ).toarray()

        denominateur = np.linalg.norm(
            Jf,
            ord="fro"
        )

        if denominateur <= 0.0:
            raise RuntimeError(
                "Norme du Jacobien numérique nulle."
            )

        err = (
            np.linalg.norm(
                Ja - Jf,
                ord="fro"
            )
            / denominateur
        )

        print(
            f"D_F = {D_F:<4} | "
            f"Erreur relative Frobenius = {err:.3e}"
        )


# ---------------------------------------------------------------------
# Fonctions auxiliaires pour les statistiques
# ---------------------------------------------------------------------

def moyenne_finie(rows, key):
    """Moyenne des valeurs finies d'une colonne."""
    values = np.asarray(
        [r[key] for r in rows],
        dtype=np.float64
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def selectionner(lignes, temps, cas):
    """Sélectionne les lignes et les ordonne par graine."""
    return sorted(
        [
            r for r in lignes
            if r["t"] == temps and r["cas"] == cas
        ],
        key=lambda r: r["seed"]
    )


def valeurs_colonne(rows, key):
    return np.asarray(
        [r[key] for r in rows],
        dtype=np.float64
    )


# ---------------------------------------------------------------------
# Validation des arguments
# ---------------------------------------------------------------------

def parser_temps(texte: str) -> tuple:
    try:
        temps = tuple(
            int(t.strip())
            for t in texte.split(",")
        )
    except ValueError as exc:
        raise ValueError(
            "Format invalide pour --times. "
            "Exemple : 5,100,500"
        ) from exc

    if not temps:
        raise ValueError(
            "Au moins un instant doit être fourni."
        )

    if any(t < 0 for t in temps):
        raise ValueError(
            "Les instants doivent être positifs ou nuls."
        )

    if len(set(temps)) != len(temps):
        raise ValueError(
            "Les instants ne doivent pas être dupliqués."
        )

    return tuple(sorted(temps))


def valider_arguments(args):
    if args.L < 2:
        raise ValueError(
            "L doit être supérieur ou égal à 2."
        )

    if args.seeds < 1:
        raise ValueError(
            "seeds doit être supérieur ou égal à 1."
        )

    if not np.isfinite(args.D_F):
        raise ValueError(
            "D_F doit être une valeur finie."
        )

    temps = parser_temps(args.times)

    return temps


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
        help=(
            "Vérifie le multiplicateur Fourier, "
            "les invariants spectraux et le Jacobien."
        )
    )

    parser.add_argument(
        "--L",
        type=int,
        default=21,
        help="Taille linéaire du réseau."
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Nombre de graines indépendantes."
    )

    parser.add_argument(
        "--times",
        default="5,100,500",
        help="Instants de mesure séparés par des virgules."
    )

    parser.add_argument(
        "--D_F",
        type=float,
        default=0.1,
        help="Coefficient de diffusion de F."
    )

    parser.add_argument(
        "--B-mode",
        choices=["quantile", "theta"],
        default="quantile",
        help=(
            "Reconstruction de B : densité conservée "
            "par quantile ou seuil fixe theta."
        )
    )

    parser.add_argument(
        "--sparse",
        action="store_true",
        help=(
            "Active la SVD partielle creuse exploratoire. "
            "La convergence n'est pas garantie."
        )
    )

    parser.add_argument(
        "--out",
        default="results/v13_5_pr.csv",
        help="Chemin du fichier CSV de sortie."
    )

    args = parser.parse_args()

    try:
        temps = valider_arguments(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.verif:

        verif_substitution_fourier(
            L=args.L
        )

        verif_jacobien()

        return

    L = args.L
    N = L * L

    Lp = matrice_laplacienne(L)

    dossier = os.path.dirname(args.out)

    if dossier:
        os.makedirs(
            dossier,
            exist_ok=True
        )

    mode_svd = (
        "Partielle creuse LM/SM"
        if args.sparse
        else "Dense de référence"
    )

    print(
        f"L={L} (N={N}) | "
        f"Graines={args.seeds} | "
        f"D_F={args.D_F} | "
        f"Mode B={args.B_mode} | "
        f"SVD={mode_svd}"
    )

    print(
        f"Temps mesurés : {temps}"
    )

    print(
        "Jacobien : J_{FI|B}, avec B fixé pendant la différentiation."
    )

    mesures = [
        "PR_top1",
        "PR_top5",
        "PR_bot1",
        "PR_bot5",
        "log10_kappa"
    ]

    lignes = []

    # -------------------------------------------------------------
    # Boucle expérimentale
    # -------------------------------------------------------------

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
                "dyn": (
                    F,
                    I,
                    B
                ),

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
                    Fx,
                    Ix,
                    Bx,
                    args.D_F,
                    Lp
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
                    "lapF": float(
                        np.linalg.norm(lap(Fx))
                    )
                })

                lignes.append(m)

        print(
            f"  [Progression] Graine "
            f"{s + 1}/{args.seeds} calculée",
            flush=True
        )

    # -------------------------------------------------------------
    # Export CSV
    # -------------------------------------------------------------

    colonnes = [
        "seed",
        "t",
        "cas",
        "densB",
        "lapF"
    ] + mesures

    with open(
        args.out,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=colonnes
        )

        writer.writeheader()
        writer.writerows(lignes)

    # -------------------------------------------------------------
    # Synthèse comparative
    # -------------------------------------------------------------

    for t in temps:

        d = selectionner(
            lignes,
            t,
            "dyn"
        )

        fo = selectionner(
            lignes,
            t,
            "fourier"
        )

        pe = selectionner(
            lignes,
            t,
            "perm"
        )

        print(
            "\n"
            + "=" * 88
        )

        print(
            f"INSTANT t = {t}"
        )

        print(
            "=" * 88
        )

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

        print("-" * 112)

        print(
            f"{'Mesure':<14} | "
            f"{'Dyn moy.':<10} | "
            f"{'Dyn-Fourier : moy. [IC95] dz +/n':<43} | "
            f"{'Dyn-Perm : moy. [IC95] dz +/n':<39}"
        )

        print("-" * 112)

        for k in mesures:

            vals_dyn = valeurs_colonne(
                d,
                k
            )

            vals_fourier = valeurs_colonne(
                fo,
                k
            )

            vals_perm = valeurs_colonne(
                pe,
                k
            )

            diff_fourier = (
                vals_dyn - vals_fourier
            )

            diff_perm = (
                vals_dyn - vals_perm
            )

            s1 = stats_apparie(
                diff_fourier
            )

            s2 = stats_apparie(
                diff_perm
            )

            dyn_mean = moyenne_finie(
                d,
                k
            )

            def format_stat(q):

                return (
                    f"{q['moy']:+.4f} "
                    f"[{q['lo']:+.4f}, {q['hi']:+.4f}] "
                    f"{q['dz']:+.2f} "
                    f"{q['pos']}/{q['n']}"
                )

            print(
                f"{k:<14} | "
                f"{dyn_mean:<10.4f} | "
                f"{format_stat(s1):<43} | "
                f"{format_stat(s2):<39}"
            )

    # -------------------------------------------------------------
    # Fin
    # -------------------------------------------------------------

    print(
        "\nDonnées détaillées enregistrées dans : "
        f"{args.out}"
    )

    print(
        "\nRappel d'interprétation :"
    )

    print(
        "  - Dyn - Fourier : contraste avec invariants spectraux "
        "de F et I conservés."
    )

    print(
        "  - Dyn - Permutation : contraste avec destruction "
        "de l'organisation spatiale."
    )

    print(
        "  - Les différences bootstrap sont calculées sur les graines, "
        "pas sur les sites."
    )

    print(
        "  - Le Jacobien est conditionnel à B fixé."
    )


if __name__ == "__main__":
    main()
