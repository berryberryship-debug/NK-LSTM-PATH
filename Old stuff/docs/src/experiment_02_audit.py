"""
src/experiment_02_audit.py
Projet : NK-LSTM-PATH

Audit spectral et pseudospectral de l'opérateur linéarisé gelé M_gelé(r, nu)
en Convention B, et cartographie du plan (r, nu) autour des seuils r_c in {1, 2, 4}.

Statut épistémique :
- Moteur spectral / SVD pseudospectral / constante de Kreiss / cartographie (r, nu) : complet.
- Constructeur `build_M_frozen` : contrat en attente de la formule explicite de M_gelé
  en Convention B (aucune extrapolation silencieuse).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class SpectralDiagnostics:
    """Indicateurs spectraux et de non-normalité d'un opérateur carré M."""

    eigenvalues: np.ndarray
    spectral_abscissa: float        # alpha(M) = max Re(lambda_i)
    spectral_radius: float          # rho(M) = max |lambda_i|
    henrici_departure: float        # sqrt(||M||_F^2 - sum |lambda_i|^2)
    eigenvector_cond: float         # kappa_2(V) = ||V||_2 ||V^{-1}||_2


def build_M_frozen(
    X_frozen: np.ndarray,
    J_frozen: np.ndarray,
    r: float,
    nu: float,
) -> np.ndarray:
    """
    Construit la matrice linéarisée gelée M_gelé(r, nu) sous Convention B.

    CONTRAT EN ATTENTE :
    La forme explicite de M_gelé(r, nu; X_frozen, J_frozen) sous Convention B
    doit être verrouillée ici avant exécution de l'audit sur le modèle NK-LSTM-PATH.
    """
    raise NotImplementedError(
        "La définition matricielle explicite de M_gelé(r, nu) en Convention B "
        "n'est pas encore verrouillée dans `build_M_frozen`."
    )


def analyze_spectrum(M: np.ndarray) -> SpectralDiagnostics:
    """Calcule le spectre exact et les métriques de non-normalité de M."""
    mat = np.asarray(M, dtype=np.complex128)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
        raise ValueError("M doit être une matrice carrée non vide.")

    eigvals, eigvecs = np.linalg.eig(mat)
    alpha = float(np.max(np.real(eigvals)))
    rho = float(np.max(np.abs(eigvals)))

    frob_sq = float(np.sum(np.abs(mat) ** 2))
    eig_sq = float(np.sum(np.abs(eigvals) ** 2))
    henrici = float(np.sqrt(max(0.0, frob_sq - eig_sq)))

    try:
        cond_v = float(np.linalg.cond(eigvecs))
    except np.linalg.LinAlgError:
        cond_v = float("inf")

    return SpectralDiagnostics(
        eigenvalues=eigvals,
        spectral_abscissa=alpha,
        spectral_radius=rho,
        henrici_departure=henrici,
        eigenvector_cond=cond_v,
    )


def compute_pseudospectrum_grid(
    M: np.ndarray,
    re_bounds: Tuple[float, float] = (-2.0, 1.0),
    im_bounds: Tuple[float, float] = (-2.0, 2.0),
    n_re: int = 120,
    n_im: int = 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Évalue sigma_min(z I - M) sur une grille complexe [re_bounds] x [im_bounds].
    Le epsilon-pseudospectre correspond à l'ensemble de sous-niveau :
        Lambda_epsilon(M) = { z in C : sigma_min(z I - M) <= epsilon }.
    """
    mat = np.asarray(M, dtype=np.complex128)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("M doit être carrée.")
    n = mat.shape[0]
    eye = np.eye(n, dtype=np.complex128)

    re_vals = np.linspace(re_bounds[0], re_bounds[1], n_re)
    im_vals = np.linspace(im_bounds[0], im_bounds[1], n_im)
    RE, IM = np.meshgrid(re_vals, im_vals)
    Z = RE + 1j * IM

    # Calcul vectorisé des valeurs singulières minimales sur toute la grille
    shifted = Z[..., None, None] * eye[None, None, :, :] - mat[None, None, :, :]
    singular_vals = np.linalg.svd(shifted, compute_uv=False)
    sigma_min = singular_vals[..., -1]
    return RE, IM, sigma_min


def pseudospectral_abscissa_and_kreiss(
    RE: np.ndarray,
    IM: np.ndarray,
    sigma_min: np.ndarray,
    epsilons: Sequence[float] = (1e-1, 1e-2, 1e-3),
) -> Tuple[dict[float, float], float]:
    """
    Estime sur grille :
      - alpha_epsilon(M) = max { Re(z) : sigma_min(zI - M) <= epsilon }
      - la borne inférieure de la constante de Kreiss :
        K(M) >= max_{Re(z) > 0} Re(z) / sigma_min(zI - M).
    """
    alpha_eps: dict[float, float] = {}
    for eps in epsilons:
        mask = sigma_min <= float(eps)
        alpha_eps[float(eps)] = float(np.max(RE[mask])) if np.any(mask) else float("nan")

    rhp_mask = RE > 0.0
    if np.any(rhp_mask):
        safe_sigma = np.maximum(sigma_min[rhp_mask], 1e-15)
        kreiss_lb = float(np.max(RE[rhp_mask] / safe_sigma))
    else:
        kreiss_lb = float("nan")

    return alpha_eps, kreiss_lb


def scan_r_nu_plane(
    M_builder: Callable[[float, float], np.ndarray],
    r_grid: np.ndarray,
    nu_grid: np.ndarray,
    epsilon_pseudo: float = 1e-2,
    compute_pseudo: bool = False,
) -> dict[str, np.ndarray]:
    """
    Balaye le plan (r, nu) et calcule en chaque point :
      - l'abscisse spectrale alpha(M(r, nu))
      - le rayon spectral rho(M(r, nu))
      - l'écart de non-normalité de Henrici
      - optionnellement l'abscisse pseudospectrale alpha_epsilon(M(r, nu)).
    """
    R, NU = np.meshgrid(r_grid, nu_grid)
    alpha_map = np.zeros_like(R, dtype=np.float64)
    rho_map = np.zeros_like(R, dtype=np.float64)
    henrici_map = np.zeros_like(R, dtype=np.float64)
    alpha_eps_map = np.full_like(R, np.nan, dtype=np.float64)

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            M_ij = M_builder(float(R[i, j]), float(NU[i, j]))
            diag = analyze_spectrum(M_ij)
            alpha_map[i, j] = diag.spectral_abscissa
            rho_map[i, j] = diag.spectral_radius
            henrici_map[i, j] = diag.henrici_departure

            if compute_pseudo:
                pad = max(1.5, 1.5 * diag.spectral_radius + 0.5)
                RE, IM, smin = compute_pseudospectrum_grid(
                    M_ij,
                    re_bounds=(-pad, pad),
                    im_bounds=(-pad, pad),
                    n_re=60,
                    n_im=60,
                )
                a_eps, _ = pseudospectral_abscissa_and_kreiss(
                    RE, IM, smin, epsilons=(epsilon_pseudo,)
                )
                alpha_eps_map[i, j] = a_eps[epsilon_pseudo]

    return {
        "R": R,
        "NU": NU,
        "alpha": alpha_map,
        "rho": rho_map,
        "henrici": henrici_map,
        "alpha_eps": alpha_eps_map,
    }


def plot_audit_figures(
    M_ref: np.ndarray,
    scan_data: dict[str, np.ndarray],
    r_thresholds: Iterable[float] = (1.0, 2.0, 4.0),
    epsilons: Sequence[float] = (1e-1, 1e-2, 1e-3),
    output_path: str = "experiment_02_audit.png",
) -> None:
    """
    Génère la planche à 2 panneaux :
      - Gauche : spectre et contours d'epsilon-pseudospectres de M_ref dans C
      - Droite : diagramme de phase dans le plan (r, nu) avec frontière alpha = 0
                 et repères des seuils critiques r_c in {1, 2, 4}.
    """
    import matplotlib.pyplot as plt

    diag = analyze_spectrum(M_ref)
    pad = max(1.5, 1.5 * diag.spectral_radius + 0.5)
    RE, IM, smin = compute_pseudospectrum_grid(
        M_ref, re_bounds=(-pad, pad), im_bounds=(-pad, pad), n_re=140, n_im=140
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    # Panneau 1 : Spectre et pseudospectres
    ax0 = axes[0]
    levels = sorted(epsilons)
    cs = ax0.contour(RE, IM, smin, levels=levels, cmap="viridis")
    ax0.clabel(cs, inline=True, fontsize=8, fmt="eps=%.0e")
    ax0.scatter(
        np.real(diag.eigenvalues),
        np.imag(diag.eigenvalues),
        color="crimson",
        marker="x",
        s=55,
        zorder=5,
        label="Valeurs propres",
    )
    ax0.axvline(0.0, color="black", linestyle="--", linewidth=0.9, label="Re(z) = 0")
    ax0.set_title(
        f"Spectre & Pseudospectres (alpha={diag.spectral_abscissa:.3f}, "
        f"kappa(V)={diag.eigenvector_cond:.1f})"
    )
    ax0.set_xlabel("Re(z)")
    ax0.set_ylabel("Im(z)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper left", fontsize=8)

    # Panneau 2 : Diagramme (r, nu) et seuils r_c in {1, 2, 4}
    ax1 = axes[1]
    R = scan_data["R"]
    NU = scan_data["NU"]
    alpha_map = scan_data["alpha"]

    im_plot = ax1.pcolormesh(R, NU, alpha_map, shading="auto", cmap="coolwarm")
    fig.colorbar(im_plot, ax=ax1, label="Abscisse spectrale alpha(M_gelé)")

    if np.min(alpha_map) < 0.0 < np.max(alpha_map):
        ax1.contour(R, NU, alpha_map, levels=[0.0], colors="black", linewidths=1.8)

    for rc in r_thresholds:
        ax1.axvline(
            float(rc),
            color="k",
            linestyle=":",
            linewidth=1.4,
            label=f"Seuil r_c = {rc:g}",
        )

    ax1.set_title("Diagramme de stabilité dans le plan (r, nu)")
    ax1.set_xlabel("Paramètre r")
    ax1.set_ylabel("Paramètre nu")
    ax1.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _demo_nonnormal_builder(r: float, nu: float) -> np.ndarray:
    """
    Matrice synthétique non-normale de démonstration pour vérifier le pipeline
    SVD/pseudospectre/grille (r, nu) indépendamment du modèle NK-LSTM-PATH.
    """
    return np.array(
        [
            [-1.0 + 0.25 * r, 4.0 * nu, 1.5],
            [0.0, -1.2 + 0.20 * r, 3.0 * nu],
            [0.0, 0.0, -1.5 + 0.15 * r],
        ],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit spectral et pseudospectral (Expérience 02).")
    parser.add_argument(
        "--demo-nonnormal",
        action="store_true",
        help="Exécute le pipeline sur une matrice non-normale synthétique de contrôle.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_02_audit.png",
        help="Chemin de sauvegarde de la figure.",
    )
    args = parser.parse_args()

    if not args.demo_nonnormal:
        # Appel direct au contrat du modèle : lèvera NotImplementedError tant que
        # build_M_frozen n'est pas verrouillé avec la formule exacte de Convention B.
        build_M_frozen(np.ones(3), np.zeros((3, 3)), r=1.0, nu=0.5)
        return

    r_grid = np.linspace(0.2, 5.0, 60)
    nu_grid = np.linspace(0.05, 2.0, 50)
    scan_data = scan_r_nu_plane(_demo_nonnormal_builder, r_grid, nu_grid, compute_pseudo=False)
    M_ref = _demo_nonnormal_builder(r=2.0, nu=1.0)
    diag = analyze_spectrum(M_ref)
    RE, IM, smin = compute_pseudospectrum_grid(M_ref)
    alpha_eps, kreiss_lb = pseudospectral_abscissa_and_kreiss(RE, IM, smin)

    print(f"[DEMO] alpha(M_ref) = {diag.spectral_abscissa:.6f}")
    print(f"[DEMO] rho(M_ref)   = {diag.spectral_radius:.6f}")
    print(f"[DEMO] Henrici dep. = {diag.henrici_departure:.6f}")
    print(f"[DEMO] kappa_2(V)   = {diag.eigenvector_cond:.6f}")
    print(f"[DEMO] Kreiss LB    = {kreiss_lb:.6f}")
    print(f"[DEMO] alpha_eps    = {alpha_eps}")
    plot_audit_figures(M_ref, scan_data, output_path=args.output)
    print(f"[DEMO] Figure enregistrée : {args.output}")


if __name__ == "__main__":
    main()
