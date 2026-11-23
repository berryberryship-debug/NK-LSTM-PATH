# NK-LSTM-PATH

**Reconstruction de l'information historique et géométrie endogène dans les paysages adaptatifs de Kauffman**

Modélisation multi-échelle des paysages NK, des trajectoires adaptatives, des structures de fibrés discrets et des représentations temporelles par réseaux LSTM.

- **Auteur :** Xioranne Lessard
- **Projet :** Axiomes Base Labs
- **Version :** 0.3 — Dynamique jointe minimale, audit $N = 3$ validé et protocole expérimental révisé
- **Statut :** Recherche exploratoire — seul l'audit algébrique et statistique statique à $N = 3$ est vérifié ; la dynamique jointe, la persistance et l'inférence LSTM restent à exécuter et tester.

---

## 1. Présentation

**NK-LSTM-PATH** est un projet de recherche computationnelle visant à étudier la structure informationnelle des trajectoires évolutives dans les paysages adaptatifs développés par Stuart Kauffman.

Le modèle NK classique décrit un paysage statique où la contribution de chaque composant dépend de son état et de ses interactions avec $K$ autres composants. Dans ce cadre standard, la dynamique de transition est markovienne : l'état présent détermine entièrement les possibilités futures.

Notre recherche introduit une rupture conceptuelle : **la géométrie endogène des possibilités**. Nous modélisons l'évolution non plus comme un déplacement sur une surface fixe, mais comme le transport d'une structure d'interactions (la fibre) qui modifie sa propre topologie au fur et à mesure de la trajectoire.

Le projet cherche à déterminer si le chemin parcouru contient une information prédictive irréductible, et si cette mémoire historique se manifeste de manière purement métrique (transitoire) ou topologique (persistante via la modification des transitions accessibles sous marge contrôlée).

---

## 2. Question fondamentale

Une configuration présente suffit-elle à caractériser les possibilités de son évolution future ?

Nous distinguons deux formes de mémoire issues de l'histoire d'une trajectoire $H_t$ :

1. **La mémoire métrique :** la distance entre deux états de la fibre adaptative $J$ issue de deux chemins différents s'estompe sous l'effet de la dissipation.
2. **La mémoire topologique :** l'écart entre les fibres modifie de façon persistante le franchissement des seuils d'accessibilité $\mathcal{T}(X, J)$ avec une marge non nulle $m_k(t) = \vert{}A_k(J(t)) - \theta_A\vert{} > 0$.

Nous cherchons à mesurer séparément :

$$D_J(t) = \Vert{} J_A(t) - J_B(t) \Vert{}_F \qquad \text{et} \qquad D_{\mathcal{T}}(t) = \left\vert{}\mathcal{T}(X, J_A(t)) \triangle \mathcal{T}(X, J_B(t))\right\vert{}$$

La persistance de $D_{\mathcal{T}}(t)$ alors que $D_J(t) \to 0$ constituerait un **indice de mémoire topologique, sous marge contrôlée** (afin d'exclure les artefacts liés à la discontinuité du seuil dur).

---

## 3. Architecture conceptuelle et dynamique jointe

Le projet repose sur un formalisme à deux niveaux (Base et Fibre) et quatre composantes dynamiques.

### A — Base combinatoire (Paysage natif)
L'hypercube des configurations $X = (s_1, \dots, s_N) \in \{-1, +1\}^N$ (en bijection avec $\{0, 1\}^N$ via $x_i = (s_i + 1)/2$). La fitness native $F_{\mathrm{NK}}(X)$ est générée par le paysage NK.

### B — Fibre adaptative et Connexion discrète
Au-dessus de chaque configuration, une fibre symétrique $J \in \mathrm{Sym}_N(\mathbb{R})$ (à diagonale nulle) représente la structure d'interactions adaptatives.

En **convention stricte post-flip**, lors de la mutation d'un nœud $k$, le spin bascule d'abord ($s_k' = -s_k$), puis le transport symétrique $\Psi$ est évalué sur les seules arêtes incidentes à $k$ :

$$\Psi_{ij}(X', J) = \frac{1}{2} \sum_{r \notin \{i, j\}} (J_{jr} + J_{ir})\, s_r' \quad (i \neq j), \qquad J'_{kj} = J_{kj} + \eta\,\Psi_{kj}(X', J).$$

La non-commutativité des transports ($[U_k, U_m] \neq 0$) définit une courbure discrète (holonomie) sur les faces élémentaires de l'hypercube.

### C — Dynamique jointe et Sélection ($\mathcal{L} = \mathcal{L}_{\mathrm{mut}} + \mathcal{L}_{\mathrm{transp}} + \mathcal{L}_{\mathrm{diss}}$)
L'évolution stochastique en temps continu couple trois mécanismes explicites :

1. **Fitness étendue :**
   $$F(X, J) = F_{\mathrm{NK}}(X) + \lambda \sum_{i < j} J_{ij} s_i s_j$$
   où $\lambda = 0$ isole le régime sans rétroaction sélective.
2. **Mutation et transport instantané ($\mathcal{L}_{\mathrm{mut}} + \mathcal{L}_{\mathrm{transp}}$) :** le taux de bascule du nœud $k$ ($X \to X^{(k)}$) est
   $$w_k(X, J) = \mu \, a_k(J) \, \min\!\left(1, e^{\beta \Delta F_k(X, J)}\right), \qquad \Delta F_k(X, J) = F(X^{(k)}, J) - F(X, J)$$
   conditionné par le masque d'accessibilité $a_k(J) = \mathbf{1}[A_k(J) > \theta_A]$ avec $A_k(J) = \sum_{j \neq k} J_{kj}^2$ et marge $m_k(J) = \vert{}A_k(J) - \theta_A\vert{}$. Chaque mutation acceptée déclenche immédiatement le transport post-flip $J \mapsto J + \eta \Psi(X^{(k)}, J)$.
3. **Dissipation continue ($\mathcal{L}_{\mathrm{diss}}$) :** entre deux mutations, la fibre relaxe exponentiellement vers $J^{(0)}$ :
   $$\frac{dJ}{dt} = -\gamma\left(J - J^{(0)}\right) \implies J(t) = J^{(0)} + \left(J(t_0) - J^{(0)}\right) e^{-\gamma (t - t_0)}.$$

**Méthode d'intégration stochastique :** dès que $\lambda \neq 0$, les taux $w_k(X, J(t))$ varient continûment entre deux sauts. La simulation repose sur un **schéma par amincissement d'Ogata (*thinning*)** exact en loi, exploitant la borne supérieure uniforme $\sum_{k=1}^N w_k(X, J(t)) \le N\mu$ (pour $\lambda = 0$, les instants de franchissement de seuil se résolvent également de manière analytique par morceaux).

### D — Reconstruction temporelle par LSTM
Les réseaux *Long Short-Term Memory* sont utilisés comme observateurs. Ils doivent inférer l'état latent de la fibre $J_t$ (non observable directement) à partir de la seule séquence des états $X_t$. L'objectif est de tester si l'apprentissage séquentiel capture la mémoire topologique.

---

## 4. Modèles expérimentaux comparés

Pour isoler les effets, nous confrontons les architectures d'apprentissage et les contrôles dynamiques.

### Architectures d'inférence (LSTM et références)
- **M1 :** État présent (Référence instantanée).
- **M2 :** État + fitness (Référence enrichie).
- **M3 :** Fenêtre historique (Mémoire explicite non récurrente).
- **M4 :** Séquence historique (Réseau LSTM).

### Design factoriel de la dynamique ($2 \times 2$)

| Règle de transport / Sélection | Avec rétroaction ($\lambda \neq 0$) | Sans rétroaction ($\lambda = 0$) |
| :--- | :--- | :--- |
| **Transport symétrique (non commutatif, $[U_k, U_m] \neq 0$)** | Modèle complet | Transport symétrique pur |
| **Transport intégrable (commutatif, $[U_k, U_m] = 0$)** | Rétroaction pure | Ligne de base |

---

## 5. Protocole initial et état de validation ($N = 3$)

L'audit commence sur la dimension minimale $N = 3$ avant de monter en échelle. Les paramètres sans dimension sont le taux de mutation normalisé $\nu = \mu N \bar{W} / \gamma$ et le rapport spectral $r = \eta\mu / \gamma$.

| Paramètre | Valeur / Définition |
| :--- | :--- |
| $N$ (composants) | $3$ (audit exact), $8$, $20$ |
| $K$ (interactions) | $2$, $4$, $8$ |
| Transitions | Seuil dur $a_k(J) = \mathbf{1}[A_k(J) > \theta_A]$ |
| Métrique de distribution | Distance de Wasserstein-1 ($W_1$) |

### Résultats acquis sur la face élémentaire $(1, 2)$ ($N = 3$, statique)
Pour $X_0 = (s_1, s_2, s_3) \in \{-1, +1\}^3$, $a = J_{12}^{(0)}$, $b = J_{13}^{(0)}$, $c = J_{23}^{(0)}$ et $A_3(J) = J_{13}^2 + J_{23}^2$, les chemins $A : 1 \to 2$ et $B : 2 \to 1$ en convention stricte post-flip vérifient :

$$\Delta J_{12} = O(\eta^2), \qquad \Delta J_{13} = O(\eta), \qquad \Delta J_{23} = O(\eta)$$

et

$$\Delta A_3 = \Delta J_{13}\left(J_{13}^{(A)} + J_{13}^{(B)}\right) + \Delta J_{23}\left(J_{23}^{(A)} + J_{23}^{(B)}\right) = 2\eta\left[b(a + c)s_2 - c(a + b)s_1\right] + O(\eta^2).$$

### Protocole de seuil ($\theta_A$)
Le seuil $\theta_A$ est traité comme un paramètre explicite et exploré par balayage sur une grille publiée dans son intégralité, sans calibration préalable sur un maximum empirique. Un premier balayage exploratoire sur 20 000 tirages ($J \sim \mathcal{U}([-1, 1]^3)$, grille grossière sans barres d'erreur) a confirmé la dépendance linéaire en $\eta$ des taux de franchissement sur une seule étape, mais **doit être refait sur une grille raffinée avec intervalles de confiance et plusieurs graines aléatoires** avant d'être cité comme référence quantitative.

---

## 6. Hypothèses de recherche

- **H1 — Non-commutativité et courbure :** Le transport symétrique de la fibre brise la parité du paysage et produit une courbure discrète non nulle sur les faces, dont la structure spectrale linéarisée ($M_{\mathrm{gelé}}$) dépend de la classe de configuration.
- **H2 — Mémoire topologique vs métrique :** Il existe un régime de paramètres où $D_J(t) \to 0$ (relaxation) mais où $D_{\mathcal{T}}(t)$ persiste sous marge contrôlée $m_k(t) > 0$ sur une plage explicite de $\theta_A$.
- **H3 — Organisation latente :** Le LSTM (**M4**) parvient à reconstruire la variable latente $J_t$ et à prédire les ruptures d'accessibilité $\mathcal{T}$, surpassant **M1–M3** dans le régime de transport symétrique actif.
- **H4 — Invariance transversale :** Les propriétés spectrales de l'opérateur de relaxation linéarisé ($M_{\mathrm{gelé}}$) et les transitions associées sont invariantes entre différentes réalisations du paysage natif $F_{\mathrm{NK}}$.

---

## 7. Contrôles et falsification

La validité expérimentale repose sur cinq contrôles explicites :

1. **Contrôle de transport intégrable :** Remplacement de la règle de transport symétrique par la 1-forme exacte $\Psi_{kj}^{\mathrm{int}} = s_k s_j$ (commutant identiquement en convention post-flip : $\Delta J \equiv 0$). Toute différence observée avec le modèle complet est attribuable au **choix de la règle de transport** (qui combine courbure et dérive locale).
2. **Mélange de l'ordre temporel :** Test de permutation des séquences historiques.
3. **Contrôle de parité :** Le transport symétrique brise la parité $s \to -s$. Le modèle est évalué sur les deux classes pour mesurer l'asymétrie des spectres et des commutateurs.
4. **Contrôle de seuil :** Remplacement du seuil dur par une sigmoïde lissée $a_k^{(\delta)}$ et publication de la courbe complète en fonction de $\theta_A$ avec intervalles de confiance.
5. **Contrôle des fuites :** Isolation stricte des paysages de test (entraînement supervisé sans accès à $J_t$).

---

## 8. Rapport conceptuel avec l'intégrale de chemin

Dans la formulation de Feynman, l'amplitude d'un processus résulte de la somme des contributions sur l'ensemble des chemins admissibles. Dans notre formalisme, la probabilité de transition dépend de l'holonomie accumulée le long de la trajectoire :

$$\mathcal{A} = \sum_{\text{chemins}} e^{-S(\text{chemin})}$$

où l'action effective $S$ dépend de la topologie du réseau et de l'historique des transports sur la fibre. Nous ne postulons aucune équivalence physique avec la mécanique quantique : le rapprochement est strictement structurel — le passé n'est pas seulement une suite d'états, c'est un transport de structure.

---

## 9. Organisation du dépôt (Mise à jour v0.3)

```text
NK-LSTM-PATH/
│
├── README.md
├── docs/
│   └── audit_courbure_N3.md      # Audit algébrique exact N = 3 (v0.2)
│
├── src/
│   ├── fiber_bundle.py           # Matrices J, règle Psi post-flip, commutateurs, oracle N = 3
│   ├── config.py                 # Paramètres sans dimension et grilles de balayage
│   ├── nk_landscape.py           # Paysage natif F_NK
│   ├── trajectories.py           # Dynamique jointe (amincissement d'Ogata : mutations + transport + dissipation)
│   ├── datasets.py               # Génération des trajectoires et des contrôles
│   └── metrics.py                # Wasserstein-1 (W_1), D_J(t), D_T(t) et marges m_k(t)
│
├── models/
│   ├── baseline_state.py         # Références M1 / M2
│   ├── baseline_history.py       # Référence M3 (fenêtre historique)
│   └── lstm_model.py             # Modèle M4 (inférence séquentielle de J_t)
│
├── experiments/
│   ├── experiment_01_nk.py       # Ligne de base NK statique
│   ├── experiment_02_audit.py    # Spectres de M_gelé, pseudospectres et plan (r, nu)
│   └── controls.py               # Contrôles (transport intégrable, seuil lissé, permutations)
│
└── tests/
    ├── test_curvature_N3.py      # Suite de validation algébrique et asymptotique N = 3 (validée)
    ├── test_nk.py                # Tests du paysage natif
    └── test_leakage.py           # Vérification d'isolation train/test
