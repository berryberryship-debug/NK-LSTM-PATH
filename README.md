# NK-LSTM-PATH

**Reconstruction de l'information historique et géométrie endogène dans les paysages adaptatifs de Kauffman**

Modélisation multi-échelle des paysages NK, des trajectoires adaptatives, des structures de fibrés discrets et des représentations temporelles par réseaux LSTM.

- **Auteur :** Xioranne Lessard
- **Projet :** Axiomes Base Labs
- **Version :** 0.2 — Formalisation géométrique, audit $N = 3$ validé et protocole expérimental révisé
- **Statut :** Recherche exploratoire — hypothèses à tester

---

## 1. Présentation

**NK-LSTM-PATH** est un projet de recherche computationnelle visant à étudier la structure informationnelle des trajectoires évolutives dans les paysages adaptatifs développés par Stuart Kauffman.

Le modèle NK classique décrit un paysage statique où la contribution de chaque composant dépend de son état et de ses interactions avec $K$ autres composants. Dans ce cadre standard, la dynamique de transition est markovienne : l'état présent détermine entièrement les possibilités futures.

Notre recherche introduit une rupture conceptuelle : **la géométrie endogène des possibilités**. Nous modélisons l'évolution non plus comme un déplacement sur une surface fixe, mais comme le transport d'une structure d'interactions (la fibre) qui modifie sa propre topologie au fur et à mesure de la trajectoire.

Le projet cherche à déterminer si le chemin parcouru contient une information prédictive irréductible, et si cette mémoire historique se manifeste de manière purement métrique (transitoire) ou topologique (persistante via la modification des transitions accessibles).

---

## 2. Question fondamentale

Une configuration présente suffit-elle à caractériser les possibilités de son évolution future ?

Nous distinguons deux formes de mémoire issues de l'histoire d'une trajectoire $H_t$ :

1. **La mémoire métrique :** la distance entre deux états de la fibre adaptative $J$ issue de deux chemins différents s'estompe sous l'effet de la dissipation.
2. **La mémoire topologique :** l'écart entre les fibres, même infime, modifie de façon persistante le franchissement des seuils d'accessibilité $\mathcal{T}(X, J)$.

Nous cherchons à mesurer séparément :

$$D_J(t) = \Vert{} J_A(t) - J_B(t) \Vert{}_F \qquad \text{et} \qquad D_{\mathcal{T}}(t) = \left\vert{}\mathcal{T}(X, J_A(t)) \triangle \mathcal{T}(X, J_B(t))\right\vert{}$$

La persistance de $D_{\mathcal{T}}(t)$ alors que $D_J(t) \to 0$ constituerait la preuve d'une irréductibilité topologique du processus.

---

## 3. Architecture conceptuelle

Le projet repose sur un formalisme à deux niveaux (Base et Fibre) et quatre composantes dynamiques.

### A — Base combinatoire (Paysage natif)
L'hypercube des configurations $X \in \{-1, +1\}^N$ (ou $\{0, 1\}^N$ selon le codage). La fitness native $F_{\mathrm{NK}}(X)$ est générée par le paysage NK.

### B — Fibre adaptative et Connexion discrète
Au-dessus de chaque configuration, une fibre symétrique $J \in \mathrm{Sym}_N(\mathbb{R})$ (à diagonale nulle) représente la structure d'interactions adaptatives.

En **convention stricte post-flip**, lors de la mutation d'un nœud $k$, le spin bascule d'abord ($s_k' = -s_k$), puis le transport symétrique $\Psi$ est évalué sur les seules arêtes incidentes à $k$ :

$$\Psi_{ij}(X', J) = \frac{1}{2} \sum_{r \notin \{i, j\}} (J_{jr} + J_{ir})\, s_r' \quad (i \neq j), \qquad J'_{kj} = J_{kj} + \eta\,\Psi_{kj}(X', J).$$

La non-commutativité des transports ($[U_k, U_m] \neq 0$) définit une courbure discrète (holonomie) sur les faces élémentaires de l'hypercube.

### C — Dynamique jointe et Sélection
L'évolution est régie par un générateur de Markov couplé :

$$\mathcal{L} = \mathcal{L}_{\mathrm{mut}} + \mathcal{L}_{\mathrm{transp}} + \mathcal{L}_{\mathrm{diss}}$$

La sélection stochastique dépend d'une fitness étendue $F(X, J)$ qui couple le paysage natif et l'état de la fibre. L'accessibilité binaire d'une mutation sur le nœud $k$ est conditionnée par la perméabilité locale de la fibre :

$$A_k(J) = \sum_{j \neq k} J_{kj}^2, \qquad a_k(J) = \mathbf{1}\left[A_k(J) > \theta_A\right], \qquad m_k(J) = \left\vert{}A_k(J) - \theta_A\right\vert{}.$$

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

| Régime de connexion / Sélection | Rétroaction sélective ($\partial_J W \neq 0$) | Sans rétroaction ($\partial_J W = 0$) |
| :--- | :--- | :--- |
| **Courbure réelle ($[U_k, U_m] \neq 0$)** | Modèle complet | Courbure pure |
| **Transport intégrable ($[U_k, U_m] = 0$)** | Feedback pur | Ligne de base |

---

## 5. Protocole initial et résultats d'audit ($N = 3$)

L'audit algébrique commence sur la dimension minimale $N = 3$ avant de monter en échelle. Les paramètres sans dimension sont le taux de mutation normalisé $\nu = \mu N \bar{W} / \lambda_0$ et le rapport spectral $r = \eta\mu / \lambda_0$.

| Paramètre | Valeur / Définition |
| :--- | :--- |
| $N$ (composants) | $3$ (audit exact), $8$, $20$ |
| $K$ (interactions) | $2$, $4$, $8$ |
| Transitions | Seuil dur $a_k(J) = \mathbf{1}[A_k(J) > \theta_A]$ |
| Métrique de distribution | Distance de Wasserstein-1 ($W_1$) |

### Écart d'accessibilité sur la face élémentaire $(1, 2)$ ($N = 3$)

Pour $X_0 = (s_1, s_2, s_3) \in \{-1, +1\}^3$, $a = J_{12}^{(0)}$, $b = J_{13}^{(0)}$, $c = J_{23}^{(0)}$ et $A_3(J) = J_{13}^2 + J_{23}^2$, la différence entre les chemins $A : 1 \to 2$ et $B : 2 \to 1$ en convention stricte post-flip présente une structure asymétrique :

$$\Delta J_{12} = O(\eta^2), \qquad \Delta J_{13} = O(\eta), \qquad \Delta J_{23} = O(\eta).$$

L'écart exact d'accessibilité du nœud tiers vérifie l'identité algébrique :

$$\Delta A_3 = A_3(J^{(A)}) - A_3(J^{(B)}) = \Delta J_{13}\left(J_{13}^{(A)} + J_{13}^{(B)}\right) + \Delta J_{23}\left(J_{23}^{(A)} + J_{23}^{(B)}\right)$$

avec pour développement au premier ordre en $\eta$ (validé numériquement à précision $O(\eta^2)$) :

$$\Delta A_3 = 2\eta\left[b(a + c)s_2 - c(a + b)s_1\right] + O(\eta^2).$$

### Protocole de seuil ($\theta_A$)
Le seuil d'accessibilité $\theta_A$ est traité comme un paramètre explicite et exploré : il est évalué par **balayage systématique sur une grille explicite publiée dans son intégralité** (avec barres d'erreur), sans calibration préalable sur $J^{(0)}$ ni sélection sur un maximum empirique dépendant de la loi d'échantillonnage.

---

## 6. Hypothèses de recherche

- **H1 — Non-commutativité et courbure :** Le transport symétrique de la fibre brise la parité du paysage et produit une courbure discrète non nulle sur les faces, dont les seuils d'instabilité locale dépendent de la classe de configuration (spectres de $M_{\mathrm{gelé}}$ à $r_c \in \{1, 2, 4\}$).
- **H2 — Mémoire topologique vs métrique :** Il existe un régime de paramètres où $D_J(t) \to 0$ (relaxation) mais où $D_{\mathcal{T}}(t)$ persiste, créant une mémoire discrète de l'histoire dépendante de la plage de $\theta_A$.
- **H3 — Organisation latente :** Le LSTM (**M4**) parvient à reconstruire la variable latente $J_t$ et à prédire les ruptures d'accessibilité $\mathcal{T}$, surpassant **M1–M3** uniquement dans le régime de courbure active.
- **H4 — Invariance transversale :** Les propriétés spectrales de l'opérateur de relaxation linéarisé ($M_{\mathrm{gelé}}$) et les transitions associées sont invariantes entre différentes réalisations du paysage natif $F_{\mathrm{NK}}$.

---

## 7. Contrôles et falsification

La validité expérimentale repose sur cinq contrôles explicites :

1. **Contrôle de transport intégrable :** Remplacement de la règle de transport symétrique par la 1-forme exacte $\Psi_{kj}^{\mathrm{int}} = s_k s_j$ (commutant identiquement en convention post-flip : $\Delta J \equiv 0$). Toute différence observée avec le modèle complet est attribuable à la non-commutativité du transport.
2. **Mélange de l'ordre temporel :** Test de permutation des séquences historiques.
3. **Contrôle de parité :** Le transport symétrique brise la parité $s \to -s$. Le modèle est évalué sur les deux classes pour mesurer l'asymétrie des spectres et des commutateurs.
4. **Contrôle de seuil :** Remplacement du seuil dur par une sigmoïde lissée $a_k^{(\delta)}$ et publication de la courbe complète en fonction de $\theta_A$.
5. **Contrôle des fuites :** Isolation stricte des paysages de test (entraînement supervisé sans accès à $J_t$).

---

## 8. Rapport conceptuel avec l'intégrale de chemin

Dans la formulation de Feynman, l'amplitude d'un processus résulte de la somme des contributions sur l'ensemble des chemins admissibles. Dans notre formalisme, la probabilité de transition dépend de l'holonomie accumulée le long de la trajectoire :

$$\mathcal{A} = \sum_{\text{chemins}} e^{-S(\text{chemin})}$$

où l'action effective $S$ dépend de la topologie du réseau et de l'historique des transports sur la fibre. Nous ne postulons aucune équivalence physique avec la mécanique quantique : le rapprochement est strictement structurel — le passé n'est pas seulement une suite d'états, c'est un transport de structure.

---

## 9. Organisation du dépôt (Mise à jour v0.2)

```text
NK-LSTM-PATH/
│
├── README.md
├── docs/
│   └── audit_courbure_N3.md      # Audit algébrique exact N = 3 (v0.2)
│
├── src/
│   ├── fiber_bundle.py           # Matrices J, règle Psi post-flip, commutateurs, oracle N = 3
│   ├── experiment_02_audit.py    # Spectres de M_gelé, pseudospectres et plan (r, nu)
│   ├── config.py                 # Paramètres sans dimension et grilles de balayage
│   ├── nk_landscape.py           # Paysage natif F_NK
│   ├── trajectories.py           # Dynamique jointe (Gillespie : mutations + transport + dissipation)
│   ├── datasets.py               # Génération des trajectoires et des contrôles
│   └── metrics.py                # Wasserstein-1 (W_1), D_J(t) et D_T(t)
│
├── models/
│   ├── baseline_state.py         # Références M1 / M2
│   ├── baseline_history.py       # Référence M3 (fenêtre historique)
│   └── lstm_model.py             # Modèle M4 (inférence séquentielle de J_t)
│
├── experiments/
│   ├── experiment_01_nk.py       # Ligne de base NK statique
│   └── controls.py               # Contrôles (transport intégrable, seuil lissé, permutations)
│
└── tests/
    ├── test_curvature_N3.py      # Suite de validation algébrique et asymptotique N = 3
    ├── test_nk.py                # Tests du paysage natif
    └── test_leakage.py           # Vérification d'isolation train/test
