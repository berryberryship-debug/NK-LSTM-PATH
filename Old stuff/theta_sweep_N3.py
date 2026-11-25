"""Balayage statique de theta_A sur la face (1,2), N=3 (une etape de chaque chemin, pas de dynamique).
J0 ~ U[-1,1]^3, X0 uniforme dans {-1,1}^3, 10 graines x 50 000 tirages. Formules exactes de l'oracle
(verifiees contre fiber_bundle sur 300 tirages). IC 95 % : Student (9 ddl) sur les 10 graines."""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fiber_bundle as fb

def exact_A3(s, abc, eta):
    s1, s2, s3 = s.T; a, b, c = abc.T; q = 0.25 * eta**2
    j13A = b + 0.5*eta*(a+c)*s2
    j23A = c - 0.5*eta*(a+b)*s1 - q*s1*((a+c)*s2 + (b+c)*s3)
    j13B = b - 0.5*eta*(a+c)*s2 - q*s2*((a+b)*s1 + (b+c)*s3)
    j23B = c + 0.5*eta*(a+b)*s1
    return j13A**2 + j23A**2, j13B**2 + j23B**2

# verification croisée du vectorisé contre le code
rng = np.random.default_rng(0)
for _ in range(300):
    s = rng.choice([-1., 1.], 3); abc = rng.uniform(-1, 1, 3); eta = 0.2
    J = fb.unpack_J(abc); _, JA = fb.path_12(s, J, eta); _, JB = fb.path_21(s, J, eta)
    A, B = exact_A3(s[None], abc[None], eta)
    assert abs(A[0] - fb.accessibility_A(JA)[2]) < 1e-12 and abs(B[0] - fb.accessibility_A(JB)[2]) < 1e-12

ETAS = [0.4, 0.2, 0.1, 0.05, 0.025]
THETAS = np.round(np.arange(0.05, 2.0001, 0.05), 2)
SEEDS, M, T975 = 10, 50000, 2.262
rows = []
for eta in ETAS:
    fr = np.zeros((SEEDS, len(THETAS))); med = np.full((SEEDS, len(THETAS)), np.nan)
    for sd in range(SEEDS):
        r = np.random.default_rng(1000 + sd)
        s = r.choice([-1., 1.], (M, 3)); abc = r.uniform(-1, 1, (M, 3))
        A, B = exact_A3(s, abc, eta)
        for i, th in enumerate(THETAS):
            cross = (A > th) != (B > th)
            fr[sd, i] = cross.mean()
            if cross.any():
                med[sd, i] = np.median(np.minimum(np.abs(A - th), np.abs(B - th))[cross])
    m = fr.mean(0); se = fr.std(0, ddof=1) / np.sqrt(SEEDS)
    for i, th in enumerate(THETAS):
        rows.append([eta, th, m[i], se[i], m[i]-T975*se[i], m[i]+T975*se[i], np.nanmean(med[:, i])])
out = os.environ.get("OUT", ".")
with open(os.path.join(out, "theta_sweep_N3.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["eta","theta_A","frac_cross","se","ci95_lo","ci95_hi","median_min_margin_given_cross"]); w.writerows(rows)
R = np.array(rows)
print("Fraction de franchissement (moyenne ± IC95), quelques theta :")
for th in (0.25, 0.5, 1.0, 1.5):
    print(" theta=%.2f : " % th + " | ".join("eta=%.3f: %.4f±%.4f" % (r[0], r[2], T975*r[3]) for r in R if abs(r[1]-th) < 1e-9))
print("Pente log-log fraction vs eta (OLS) par theta :")
for th in (0.25, 0.5, 1.0, 1.5):
    sel = R[np.abs(R[:,1]-th) < 1e-9]
    print("  theta=%.2f pente=%.3f" % (th, np.polyfit(np.log(sel[:,0]), np.log(sel[:,2]), 1)[0]))
print("argmax theta (indicatif, depend de la loi d'echantillonnage) :")
for eta in ETAS:
    sel = R[R[:,0] == eta]; print("  eta=%.3f -> theta=%.2f (frac=%.4f)" % (eta, sel[np.argmax(sel[:,2]),1], sel[:,2].max()))
