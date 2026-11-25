# NK-LSTM-PATH

**Reconstruction de l'information historique et géométrie endogène dans les paysages adaptatifs de Kauffman**

Modélisation multi-échelle des paysages NK, des trajectoires adaptatives, des structures de fibrés discrets et des représentations temporelles par réseaux LSTM.

- **Auteur :** Xioranne Lessard
- **Projet :** Axiomes Base Labs
- **Version :** 0.4 — Audit algébrique \(N=3\) et audit numérique de non-commutativité \(N=6\), avec balayage de l'intensité de transport
- **Statut :** Recherche exploratoire. L'audit algébrique exact à \(N=3\) et le balayage numérique de non-commutativité à \(N=6\) ont été exécutés. La dynamique jointe avec dissipation, la persistance temporelle de la mémoire d'accessibilité et l'inférence LSTM restent à valider.

---

## 1. Présentation

**NK-LSTM-PATH** est un projet de recherche computationnelle visant à étudier la structure informationnelle des trajectoires évolutives dans les paysages adaptatifs développés par Stuart Kauffman.

Le modèle NK classique décrit un paysage de fitness où la contribution de chaque composant dépend de son état et de ses interactions avec \(K\) autres composants. Une dynamique évolutive définie sur ce paysage peut être markovienne lorsque l'état considéré contient toutes les variables nécessaires à la détermination des transitions futures.

Notre recherche introduit une rupture conceptuelle : **la géométrie endogène des possibilités**. Nous modélisons l'évolution non plus comme un déplacement sur une surface fixe, mais comme le transport d'une structure d'interactions — la fibre — qui se modifie au cours de la trajectoire.

Le projet cherche à déterminer si le chemin parcouru contient une information prédictive irréductible et si cette mémoire historique se manifeste de manière :

- **métrique**, par la conservation transitoire d'un écart entre deux fibres ;
- **liée à l'accessibilité**, par une différence persistante entre les transitions accessibles depuis deux histoires distinctes, sous marge contrôlée.

La seconde propriété reste une hypothèse expérimentale : une différence instantanée entre les ensembles de transitions accessibles ne suffit pas à établir une mémoire topologique persistante.

---

## 2. Question fondamentale

Une configuration présente suffit-elle à caractériser les possibilités de son évolution future ?

Nous distinguons deux formes de mémoire issues de l'histoire d'une trajectoire \(H_t\).

1. **Mémoire métrique :** la distance entre deux états de la fibre adaptative \(J\), issus de deux chemins différents, peut persister temporairement ou s'estomper sous l'effet de la dissipation.

2. **Mémoire d'accessibilité :** l'écart entre les fibres peut modifier les transitions accessibles depuis une même configuration. Cette différence est mesurée dans le temps et sous une marge non nulle par rapport au seuil.

Nous définissons l'ensemble des transitions accessibles :

\[
\mathcal{T}(X,J)
=
\{k \mid A_k(J)>\theta_A\}
\]

avec :

\[
A_k(J)=\sum_{j\neq k}J_{kj}^{2}
\]

et la marge au seuil :

\[
m_k(t)=\left|A_k(J(t))-\theta_A\right|
\]

Les deux observables principales sont :

\[
D_J(t)=\left\|J_A(t)-J_B(t)\right\|_F
\]

et :

\[
D_{\mathcal{T}}(t)
=
\left|
\mathcal{T}(X,J_A(t))
\triangle
\mathcal{T}(X,J_B(t))
\right|
\]

où \(\triangle\) désigne la différence symétrique entre ensembles.

La persistance de \(D_{\mathcal{T}}(t)\) alors que \(D_J(t)\to 0\), sous une marge contrôlée \(m_k(t)>0\), constituerait un **indice de mémoire d'accessibilité persistante**. Son interprétation comme mémoire topologique devra être étayée par l'analyse de la structure des ensembles accessibles et par des contrôles contre les effets de seuil.

---

## 3. Architecture conceptuelle et dynamique jointe

Le projet repose sur un formalisme à deux niveaux — Base et Fibre — et quatre composantes dynamiques.

### A — Base combinatoire : paysage natif

L'hypercube des configurations est :

\[
X=(s_1,\ldots,s_N)\in\{-1,+1\}^N
\]

en bijection avec \(\{0,1\}^N\) via :

\[
x_i=\frac{s_i+1}{2}
\]

La fitness native \(F_{\mathrm{NK}}(X)\) est générée par le paysage NK.

### B — Fibre adaptative et connexion discrète

Au-dessus de chaque configuration, une fibre symétrique :

\[
J\in\mathrm{Sym}_N(\mathbb{R})
\]

à diagonale nulle, représente la structure d'interactions adaptatives.

En **convention stricte post-flip**, lors de la mutation d'un nœud \(k\), le spin bascule d'abord :

\[
s_k'=-s_k
\]

Le transport symétrique \(\Psi\) est ensuite évalué sur l'état post-flip :

\[
\Psi_{ij}(X',J)
=
\frac{1}{2}
\sum_{r\notin\{i,j\}}
(J_{jr}+J_{ir})s_r',
\qquad i\neq j
\]

Seules les arêtes incidentes à \(k\) sont mises à jour :

\[
J'_{kj}=J_{kj}+\eta\Psi_{kj}(X',J)
\]

avec conservation de la symétrie :

\[
J'_{jk}=J'_{kj}
\]

et de la diagonale nulle.

Pour deux mutations distinctes \(k\) et \(m\), les deux compositions possibles sont comparées :

\[
U_m\circ U_k
\qquad\text{et}\qquad
U_k\circ U_m
\]

Un écart non nul entre les fibres finales caractérise une dépendance à l'ordre des transports. Dans ce cadre, le commutateur discret est utilisé comme observable de non-commutativité :

\[
\left\|
U_m(U_k(J))-U_k(U_m(J))
\right\|_F
>0
\]

Cette non-commutativité est interprétée comme une courbure discrète ou une holonomie de face dans le modèle. Cette terminologie désigne ici une propriété opérationnelle des compositions de transports ; elle ne constitue pas, à elle seule, une démonstration de toutes les propriétés d'une connexion géométrique continue.

### C — Dynamique jointe et sélection

L'évolution complète est conçue autour de trois mécanismes :

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{mut}}
+
\mathcal{L}_{\mathrm{transp}}
+
\mathcal{L}_{\mathrm{diss}}
\]

**1. Fitness étendue**

\[
F(X,J)
=
F_{\mathrm{NK}}(X)
+
\lambda\sum_{i<j}J_{ij}s_is_j
\]

De manière équivalente, pour une matrice symétrique à diagonale nulle :

\[
F(X,J)
=
F_{\mathrm{NK}}(X)
+
\frac{\lambda}{2}X^TJX
\]

Le paramètre \(\lambda=0\) isole le régime sans rétroaction sélective de la fibre sur la fitness.

**2. Mutation et transport instantané**

Le taux de bascule du nœud \(k\), correspondant à \(X\to X^{(k)}\), est :

\[
w_k(X,J)
=
\mu a_k(J)
\min\left(1,e^{\beta\Delta F_k(X,J)}\right)
\]

avec :

\[
\Delta F_k(X,J)
=
F(X^{(k)},J)-F(X,J)
\]

Le masque d'accessibilité est :

\[
a_k(J)=\mathbf{1}[A_k(J)>\theta_A]
\]

Chaque mutation acceptée déclenche immédiatement le transport post-flip :

\[
J\mapsto J+\eta\Psi(X^{(k)},J)
\]

**3. Dissipation continue**

Entre deux mutations, la fibre relaxe exponentiellement vers sa référence \(J^{(0)}\) :

\[
\frac{dJ}{dt}
=
-\gamma(J-J^{(0)})
\]

d'où :

\[
J(t)
=
J^{(0)}
+
\left(J(t_0)-J^{(0)}\right)e^{-\gamma(t-t_0)}
\]

**Méthode d'intégration stochastique**

Lorsque \(\lambda\neq0\), les taux \(w_k(X,J(t))\) peuvent varier continûment entre deux sauts. La simulation prévue repose sur un schéma par amincissement d'Ogata (*thinning*), avec la borne :

\[
\sum_{k=1}^{N}w_k(X,J(t))\leq N\mu
\]

Cette borne découle de \(0\leq a_k\leq1\) et de :

\[
0\leq\min(1,e^{\beta\Delta F_k})\leq1
\]

La validité de l'implémentation stochastique complète, incluant la dissipation et les changements de taux, reste à vérifier par des tests dédiés.

### D — Reconstruction temporelle par LSTM

Les réseaux *Long Short-Term Memory* sont utilisés comme observateurs.

Ils doivent inférer l'état latent de la fibre \(J_t\), non directement observable, à partir de la seule séquence des états \(X_t\).

L'objectif est de déterminer si l'apprentissage séquentiel capture une information historique qui n'est pas récupérable à partir du seul état présent.

Cette capacité reste à tester. Aucune supériorité du LSTM sur les modèles de référence n'est revendiquée à ce stade.

---

## 4. Modèles expérimentaux comparés

Pour isoler les effets, nous confrontons les architectures d'apprentissage et les contrôles dynamiques.

### Architectures d'inférence

- **M1 — État présent :** référence instantanée.
- **M2 — État + fitness :** référence enrichie.
- **M3 — Fenêtre historique :** mémoire explicite non récurrente.
- **M4 — Séquence historique :** réseau LSTM.

### Design factoriel de la dynamique \(2\times2\)

| Règle de transport / Sélection | Avec rétroaction \((\lambda\neq0)\) | Sans rétroaction \((\lambda=0)\) |
| :--- | :--- | :--- |
| Transport symétrique non commutatif \(([U_k,U_m]\neq0)\) | Modèle complet | Transport symétrique pur |
| Transport intégrable commutatif \(([U_k,U_m]=0)\) | Rétroaction pure | Ligne de base |

Ce plan factoriel définit les comparaisons prévues. Son exécution complète et l'évaluation statistique des quatre conditions restent à réaliser.

---

## 5. Protocoles expérimentaux et état de validation

Le programme distingue deux échelles de validation :

- \(N=3\) : audit algébrique exact sur une face élémentaire.
- \(N=6\) : audit numérique de l'effet de l'ordre des mutations et de sa dépendance à l'intensité du transport.

Les simulations dynamiques de plus grande échelle, notamment \(N=8\) et \(N=20\), restent à exécuter.

### 5.1 — Audit algébrique exact sur la face élémentaire \((1,2)\), \(N=3\)

Pour :

\[
X_0=(s_1,s_2,s_3)\in\{-1,+1\}^3
\]

et :

\[
a=J_{12}^{(0)},\quad
b=J_{13}^{(0)},\quad
c=J_{23}^{(0)}
\]

avec :

\[
A_3(J)=J_{13}^2+J_{23}^2
\]

les chemins :

\[
A:1\to2
\qquad\text{et}\qquad
B:2\to1
\]

en convention stricte post-flip vérifient :

\[
\Delta J_{12}=O(\eta^2)
\]

\[
\Delta J_{13}=O(\eta)
\]

\[
\Delta J_{23}=O(\eta)
\]

et :

\[
\begin{aligned}
\Delta A_3
&=
\Delta J_{13}
\left(J_{13}^{(A)}+J_{13}^{(B)}\right)\\
&\quad+
\Delta J_{23}
\left(J_{23}^{(A)}+J_{23}^{(B)}\right)
\end{aligned}
\]

soit, au premier ordre :

\[
\Delta A_3
=
2\eta
\left[
b(a+c)s_2-c(a+b)s_1
\right]
+
O(\eta^2)
\]

Cet audit établit une expression analytique de l'écart d'accessibilité sur la face élémentaire étudiée. Il ne valide pas, à lui seul, la dynamique jointe en temps continu.

### 5.2 — Audit numérique de non-commutativité, \(N=6\)

Un balayage numérique compare les deux ordres d'application de mutations distinctes :

\[
k\to m
\qquad\text{et}\qquad
m\to k
\]

Le protocole exécuté utilise :

| Paramètre | Valeur |
| :--- | :--- |
| Nombre de sites \(N\) | 6 |
| Nombre de graines | 10 |
| Paires distinctes par graine | 15 |
| Comparaisons par valeur de \(\eta\) | 150 |
| Graine initiale | 7 |
| Échelle initiale de \(J\) | 0.2 |
| Graine du paysage NK | 7 |
| \(K\) | 3 |
| \(\mu\) | 1.0 |
| \(\beta\) | 1.0 |
| \(\theta_A\) | 0.1 |
| \(\lambda\) | 0.5 |

Le paysage NK est maintenu identique dans les comparaisons.

Le balayage de robustesse a été effectué sur :

\[
\eta\in
\{0,\;0.05,\;0.1,\;0.25,\;0.5,\;1.0\}
\]

Chaque ligne du tableau correspond à 150 comparaisons.

#### Résultats du balayage

| \(\eta\) | Comparaisons | Écart moyen \(\|J_A-J_B\|_F\) | Écart maximal | Erreur symétrie | Erreur diagonale | \(X\) identique | \(\Delta w\neq0\) | Franchissements* |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 150 | 0.000000 | 0.000000 | 0 | 0 | 100.0 % | 0.0 % | 0 |
| 0.05 | 150 | 0.037462 | 0.084396 | 0 | 0 | 100.0 % | 68.0 % | 20 |
| 0.10 | 150 | 0.075777 | 0.174236 | 0 | 0 | 100.0 % | 68.7 % | 34 |
| 0.25 | 150 | 0.197552 | 0.478854 | 0 | 0 | 100.0 % | 80.0 % | 91 |
| 0.50 | 150 | 0.431533 | 1.118733 | 0 | 0 | 100.0 % | 93.3 % | 178 |
| 1.00 | 150 | 1.060533 | 3.053254 | 0 | 0 | 100.0 % | 97.3 % | 213 |

\* Le compteur de franchissements est le compteur rapporté par le programme. Comme il dépasse 150 pour certaines valeurs de \(\eta\), il ne représente pas simplement le nombre de comparaisons ayant connu au moins un franchissement. Son unité exacte — événements par site, transitions ou autre comptage interne — doit être confirmée dans l'implémentation avant toute interprétation quantitative supplémentaire.

#### Observations établies

**Contrôle nul**

À \(\eta=0\), l'écart entre les matrices finales est nul :

\[
\left\|J_A-J_B\right\|_F=0
\]

Le transport étant désactivé, les deux ordres ne produisent aucune divergence dans la fibre.

**Augmentation de l'écart avec \(\eta\)**

L'écart moyen et l'écart maximal augmentent à chaque valeur de \(\eta\) testée. Dans cette grille, l'écart moyen passe de 0.037462 à \(\eta=0.05\) à 1.060533 à \(\eta=1\).

Cette observation indique une dépendance croissante de la non-commutativité mesurée à l'intensité du transport. Elle ne suffit pas à établir une loi générale de croissance ni une relation analytique entre l'écart et \(\eta\).

**Conservation de l'état de base**

La fraction d'états \(X\) identiques entre les deux ordres est de 100 % à toutes les valeurs testées.

Pour deux flips distincts, cette conservation est cohérente avec la commutation des flips sur la base combinatoire :

\[
X^{(k)(m)}=X^{(m)(k)}
\qquad(k\neq m)
\]

La divergence observée se situe donc dans la fibre \(J\), et non dans l'état final des spins pour ce protocole.

**Conservation des contraintes de la fibre**

Les erreurs maximales de symétrie et de diagonale sont nulles pour toutes les valeurs de \(\eta\).

La mise à jour conserve donc les invariants structurels testés :

\[
J=J^T
\]

et :

\[
J_{ii}=0
\]

**Propagation vers les taux**

La fraction de comparaisons où \(\Delta w\neq0\) augmente globalement avec \(\eta\), passant de 68.0 % à \(\eta=0.05\) à 97.3 % à \(\eta=1\).

La différence entre les fibres se répercute donc sur les observables de transition calculées par le programme.

Cette propagation peut impliquer les changements d'accessibilité \(A_k(J)\), ainsi que les termes de fitness présents dans les taux. Les contributions respectives devront être isolées par des contrôles spécifiques.

#### Valeurs complémentaires à \(\eta=0.25\)

La première exécution de référence a également rapporté :

| Observable | Résultat |
| :--- | ---: |
| Écart moyen \(\|J_A-J_B\|_F\) | \(1.975517194180\times10^{-1}\) |
| Écart maximal \(\|J_A-J_B\|_F\) | \(4.788543574602\times10^{-1}\) |
| Erreur maximale de symétrie | 0 |
| Erreur maximale diagonale | 0 |
| Fraction \(X\) identique | 100 % |
| \(\Delta A\) moyen | \(5.336970508995\times10^{-2}\) |
| \(\Delta A\) maximal | \(2.641798966565\times10^{-1}\) |
| \(\Delta w\) moyen | \(4.848557077700\times10^{-1}\) |
| \(\Delta w\) maximal | \(1.667627437335\) |
| Fraction \(\Delta w\neq0\) | 80.0 % |
| Franchissements rapportés | 91 |

L'écart moyen et maximal sur \(J\), ainsi que les autres indicateurs reproduits dans le balayage, concordent avec la ligne \(\eta=0.25\) du tableau de robustesse.

#### Portée et limites de cet audit

Cet audit numérique établit, pour le protocole testé :

- une dépendance mesurable à l'ordre des mutations dans la fibre \(J\) ;
- une augmentation de l'écart moyen et maximal sur la grille de \(\eta\) ;
- une transmission de cette différence vers les observables de transition \(w\) ;
- la conservation de \(X\), de la symétrie de \(J\) et de sa diagonale nulle.

Il ne démontre pas encore :

- une mémoire persistante après dissipation ;
- la persistance de \(D_{\mathcal{T}}(t)\) lorsque \(D_J(t)\to0\) ;
- une propriété topologique indépendante du seuil choisi ;
- une supériorité prédictive du LSTM ;
- une loi générale reliant l'intensité du transport à la courbure.

**Condition de reproductibilité à documenter :** la graine initiale et les paramètres sont fixés dans le protocole. L'identité exacte des réalisations \((X_0,J_0)\) réutilisées pour chaque valeur de \(\eta\) doit être confirmée dans le code source. Le seul affichage des graines ne suffit pas à prouver cet appariement.

### 5.3 — Protocole de seuil \(\theta_A\)

Le seuil \(\theta_A\) est traité comme un paramètre explicite et exploré par balayage sur une grille publiée dans son intégralité, sans calibration préalable sur un maximum empirique.

Un premier balayage exploratoire sur 20 000 tirages :

\[
J\sim\mathcal{U}([-1,1]^3)
\]

avec une grille grossière et sans barres d'erreur, avait indiqué une dépendance des taux de franchissement à \(\eta\) sur une seule étape.

Ce résultat doit être refait sur une grille raffinée, avec intervalles de confiance et plusieurs graines aléatoires, avant d'être cité comme référence quantitative.

---

## 6. Hypothèses de recherche

Les hypothèses sont conservées comme propositions à tester. Les observations du protocole \(N=6\) constituent un premier résultat expérimental, et non une validation globale de chacune d'elles.

- **H1 — Non-commutativité et courbure :** le transport symétrique de la fibre brise la parité du paysage et produit une non-commutativité discrète sur les faces élémentaires. La structure spectrale linéarisée (\(M_{\mathrm{gelé}}\)) pourrait dépendre de la classe de configuration.  
  **État :** non-commutativité mesurée sur la fibre dans les audits \(N=3\) et \(N=6\). La dépendance spectrale et la portée générale de l'hypothèse restent à tester.

- **H2 — Mémoire d'accessibilité vs mémoire métrique :** il existe un régime de paramètres où \(D_J(t)\to0\), par relaxation, tandis que \(D_{\mathcal{T}}(t)\) persiste sous marge contrôlée \(m_k(t)>0\) sur une plage explicite de \(\theta_A\).  
  **État :** non testé par les audits de non-commutativité présentés ici. Il nécessite des trajectoires avec dissipation et des mesures temporelles.

- **H3 — Organisation latente :** le LSTM (M4) parvient à reconstruire la variable latente \(J_t\) et à prédire les ruptures d'accessibilité \(\mathcal{T}\), en surpassant M1–M3 dans le régime de transport symétrique actif.  
  **État :** à tester. Aucune performance d'inférence LSTM n'est revendiquée à ce stade.

- **H4 — Invariance transversale :** les propriétés spectrales de l'opérateur de relaxation linéarisé (\(M_{\mathrm{gelé}}\)) et les transitions associées sont invariantes entre différentes réalisations du paysage natif \(F_{\mathrm{NK}}\).  
  **État :** à tester sur plusieurs réalisations indépendantes du paysage NK.

---

## 7. Contrôles et falsification

La validité expérimentale repose sur cinq contrôles explicites.

1. **Contrôle de transport intégrable**

   Remplacement de la règle de transport symétrique par le contrôle candidat :

   \[
   \Psi_{kj}^{\mathrm{int}}=s_k's_j
   \]

   avec \(s_k'\) évalué après le flip.

   L'objectif est de construire un transport dont les compositions associées à deux flips distincts commutent :

   \[
   U_m^{\mathrm{int}}\circ U_k^{\mathrm{int}}
   =
   U_k^{\mathrm{int}}\circ U_m^{\mathrm{int}}
   \]

   L'écart entre les fibres finales doit alors être nul :

   \[
   \left\|
   U_m^{\mathrm{int}}(U_k^{\mathrm{int}}(J))
   -
   U_k^{\mathrm{int}}(U_m^{\mathrm{int}}(J))
   \right\|_F=0
   \]

   Cette égalité ne signifie pas que chaque mise à jour individuelle de \(J\) est nulle. Elle signifie que le résultat final ne dépend pas de l'ordre des deux transports. L'intégrabilité de ce contrôle doit être vérifiée par des tests unitaires.

2. **Mélange de l'ordre temporel**

   Permutation des séquences historiques afin de déterminer si les modèles d'inférence exploitent effectivement l'ordre temporel plutôt qu'une simple distribution des états.

3. **Contrôle de parité**

   Le transport symétrique est examiné sous la transformation :

   \[
   s\to-s
   \]

   Le modèle est évalué sur les deux classes pour mesurer l'éventuelle asymétrie des observables, des spectres et des commutateurs.

4. **Contrôle de seuil**

   Remplacement du seuil dur par une sigmoïde lissée \(a_k^{(\delta)}\), puis publication de la courbe complète en fonction de \(\theta_A\), avec intervalles de confiance.

5. **Contrôle des fuites**

   Isolation stricte des paysages de test. L'entraînement supervisé ne doit pas avoir accès à \(J_t\) lorsque cette variable est la cible d'inférence.

---

## 8. Rapport conceptuel avec l'intégrale de chemin

Le projet établit un rapprochement structurel avec les formulations par chemins : une évolution peut être décrite en considérant l'ensemble des trajectoires admissibles et le poids associé à chacune.

Une écriture formelle possible est :

\[
\mathcal{Z}_{\mathrm{chemins}}
=
\sum_{\text{chemins}}
e^{-S_{\mathrm{eff}}(\text{chemin})}
\]

où \(S_{\mathrm{eff}}\) représente une action effective dépendant de la structure du réseau et de l'historique des transports sur la fibre.

Cette expression est utilisée comme analogie de somme de poids de chemins. Elle ne constitue pas une amplitude de Feynman en temps réel et ne postule aucune équivalence physique avec la mécanique quantique.

Le rapprochement est strictement structurel : le passé n'est pas seulement une suite d'états, il peut aussi correspondre à un transport de structure qui modifie les possibilités ultérieures.

---

## 9. Organisation du dépôt

Arborescence cible du projet :

```text
NK-LSTM-PATH/
│
├── README.md
│
├── docs/
│   ├── audit_courbure_N3.md
│   └── audit_robustesse_eta_N6.md
│
├── src/
│   ├── fiber_bundle.py
│   ├── config.py
│   ├── nk_landscape.py
│   ├── trajectories.py
│   ├── datasets.py
│   └── metrics.py
│
├── models/
│   ├── baseline_state.py
│   ├── baseline_history.py
│   └── lstm_model.py
│
├── experiments/
│   ├── experiment_01_nk.py
│   ├── experiment_02_audit.py
│   ├── balayage_commutateur_nk_v3.py
│   ├── audit_robustesse_eta.py
│   └── controls.py
│
└── tests/
    ├── test_curvature_N3.py
    ├── test_nk.py
    ├── test_leakage.py
    └── test_integrable_transport.py
```

Les deux scripts d'audit \(N=6\) sont inscrits dans l'arborescence cible. Leurs chemins effectifs dans le dépôt devront être harmonisés avec cette structure lors de leur intégration.

---

## 10. État des validations et prochaines étapes

| Élément | État |
| :--- | :--- |
| Audit algébrique exact sur la face \(N=3\) | Vérifié |
| Audit numérique de non-commutativité \(N=6\) | Exécuté |
| Balayage de six intensités \(\eta\) | Exécuté |
| Conservation de la symétrie et de la diagonale de \(J\) | Vérifiée sur le protocole \(N=6\) |
| Conservation de \(X\) entre les deux ordres de flips | Vérifiée sur le protocole \(N=6\) |
| Définition exacte du compteur de franchissements | À auditer |
| Appariement exact des états initiaux entre les valeurs de \(\eta\) | À confirmer dans le code |
| Contrôle de transport intégrable | À implémenter et tester |
| Dynamique jointe avec dissipation | À exécuter et tester |
| Persistance temporelle de \(D_{\mathcal{T}}(t)\) | À tester |
| Robustesse sur plusieurs paysages NK indépendants | À tester |
| Comparaison M1–M4 | À exécuter |
| Généralisation à \(N=8\) et \(N=20\) | À exécuter |

**Critère de progression scientifique :** chaque extension du modèle doit préserver la distinction entre résultat algébrique, observation numérique et hypothèse interprétative. Une non-commutativité mesurée dans la fibre ne suffit pas à établir une mémoire topologique persistante ; une reconstruction numérique de \(J_t\) ne suffit pas, à elle seule, à établir une propriété cognitive du réseau LSTM.

Le projet avance donc par validations successives : géométrie discrète, dynamique jointe, persistance de l'accessibilité, puis reconstruction temporelle.
