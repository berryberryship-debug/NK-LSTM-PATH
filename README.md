# NK-LSTM-PATH

**Reconstruction de l'information historique et géométrie endogène dans les paysages adaptatifs de Kauffman**

Modélisation multi-échelle des paysages NK, des trajectoires adaptatives, des structures de fibrés discrets et des représentations temporelles par réseaux LSTM.

- **Auteur :** Xioranne Lessard
- **Projet :** Axiomes Base Labs
- **Version :** 0.4 — Audit algébrique exact \(N=3\), audits numériques de non-commutativité \(N=6\), balayage de l'intensité de transport et audit de reproductibilité multi-graines.
- **Statut :** Recherche exploratoire. L'audit algébrique à \(N=3\), le balayage numérique à \(N=6\) et l'audit multi-graines ont été exécutés. La dynamique jointe avec dissipation, la persistance temporelle de l'accessibilité et l'inférence LSTM restent à valider.

---

## 1. Présentation

**NK-LSTM-PATH** est un projet de recherche computationnelle consacré à l'étude de l'information historique portée par les trajectoires évolutives dans les paysages adaptatifs de Kauffman.

Le modèle NK classique décrit un paysage de fitness dans lequel la contribution de chaque composant dépend de son état et de ses interactions avec \(K\) autres composants. Une dynamique évolutive construite sur ce paysage peut être markovienne lorsque l'état du système contient toutes les variables nécessaires à la détermination de son évolution future.

NK-LSTM-PATH introduit une structure supplémentaire : une **fibre adaptative**, représentée par une matrice d'interactions \(J\), transportée et transformée au cours des mutations.

Le modèle distingue ainsi deux niveaux :

- une configuration combinatoire \(X\), qui décrit l'état présent des composants ;
- une structure d'interactions \(J\), qui dépend des mises à jour successives et peut donc porter une information sur l'histoire du système.

La question expérimentale consiste à déterminer dans quelle mesure deux trajectoires aboutissant à une même configuration \(X\) peuvent conserver des structures de couplage différentes et produire des observables de transition distinctes.

Le projet examine trois propriétés :

1. **Non-commutativité du transport :** deux mutations appliquées dans des ordres différents peuvent-elles produire des fibres finales distinctes ?
2. **Mémoire métrique :** la différence entre deux fibres peut-elle persister temporairement avant de s'atténuer sous l'effet de la dissipation ?
3. **Mémoire d'accessibilité :** les différences entre les fibres peuvent-elles modifier les transitions accessibles depuis une configuration donnée ?

Il faut distinguer la mémoire portée par une variable d'état de la non-markovianité du système. Si l'état complet est \((X,J)\), l'information historique encodée dans \(J\) peut être intégrée à un processus markovien sur cet espace augmenté. La mémoire est alors portée par la structure présente de la fibre, même si elle provient de l'histoire antérieure.

Les résultats numériques présentés dans cette version établissent une non-commutativité mesurée dans le protocole discret étudié. Ils ne démontrent pas encore une mémoire persistante après dissipation ni une propriété cognitive du réseau LSTM.

---

## 2. Question fondamentale et observables

Une configuration présente \(X\) suffit-elle à caractériser les possibilités d'évolution future, ou faut-il également connaître la structure d'interactions \(J\) construite au cours de l'histoire ?

Nous distinguons deux trajectoires, \(A\) et \(B\), qui partent d'un même état initial et subissent les mêmes mutations dans des ordres différents.

### 2.1 Mémoire métrique

La distance entre les deux fibres est mesurée par :

\[
D_J(t)=\left\|J_A(t)-J_B(t)\right\|_F
\]

où \(\|\cdot\|_F\) désigne la norme de Frobenius.

Cette observable mesure la divergence géométrique entre deux structures d'interactions issues de trajectoires distinctes.

### 2.2 Accessibilité

L'accessibilité d'un nœud \(k\) est définie par :

\[
A_k(J)=\sum_{j\neq k}J_{kj}^{2}
\]

Pour un seuil \(\theta_A\), l'ensemble des transitions accessibles est :

\[
\mathcal{T}(X,J)
=
\{k\in\{1,\ldots,N\}\mid A_k(J)>\theta_A\}
\]

Dans la règle actuelle, l'accessibilité dépend de \(J\), et non directement de \(X\). L'argument \(X\) est conservé pour situer l'observable dans l'état complet du système.

La différence d'accessibilité entre deux trajectoires est :

\[
D_{\mathcal{T}}(t)
=
\operatorname{card}
\left(
\mathcal{T}(X,J_A(t))
\triangle
\mathcal{T}(X,J_B(t))
\right)
\]

où \(\triangle\) représente la différence symétrique entre ensembles.

Une valeur non nulle signifie que les deux trajectoires ne possèdent pas exactement le même ensemble de nœuds accessibles au temps considéré.

### 2.3 Marge au seuil et limite mathématique

La marge au seuil est définie pour chaque trajectoire :

\[
m_{k,A}(t)
=
\left|A_k(J_A(t))-\theta_A\right|
\]

\[
m_{k,B}(t)
=
\left|A_k(J_B(t))-\theta_A\right|
\]

La marge permet de distinguer une différence d'accessibilité robuste d'une différence produite par la proximité immédiate du seuil.

Une contrainte mathématique doit toutefois être prise en compte dans l'interprétation de la mémoire d'accessibilité.

Pour chaque site \(k\), en notant \(r_k^A\) et \(r_k^B\) les lignes correspondantes des deux matrices :

\[
\begin{aligned}
\left|A_k(J_A)-A_k(J_B)\right|
&=
\left|\|r_k^A\|^2-\|r_k^B\|^2\right|\\
&\leq
\left(\|r_k^A\|+\|r_k^B\|\right)
\|r_k^A-r_k^B\|\\
&\leq
\left(\|J_A\|_F+\|J_B\|_F\right)D_J
\end{aligned}
\]

Si les deux masques diffèrent au site \(k\), le seuil \(\theta_A\) se trouve entre les deux valeurs d'accessibilité. Par conséquent :

\[
\min(m_{k,A},m_{k,B})
\leq
\left(\|J_A\|_F+\|J_B\|_F\right)D_J
\]

Cette relation impose une limite à l'hypothèse de persistance initialement envisagée.

Si les normes des fibres restent bornées et que :

\[
D_J(t)\longrightarrow 0
\]

alors toute divergence persistante des masques exige qu'au moins une des deux trajectoires se rapproche du seuil. Une différence d'accessibilité ne peut donc pas persister asymptotiquement avec une marge uniformément positive sur les deux trajectoires.

**Conséquence pour le protocole :**

- Une divergence de \(D_{\mathcal T}\) pendant que \(D_J\) reste non nul peut être observée et mesurée.
- Une divergence de masque lorsque les marges sont très faibles doit être interprétée comme une sensibilité au seuil.
- Une persistance de l'accessibilité après convergence de \(J_A\) et \(J_B\), avec une marge uniformément positive, nécessiterait une règle d'accessibilité dépendante de l'histoire ou une variable supplémentaire, par exemple un mécanisme d'hystérésis.

Cette distinction constitue une contrainte de falsification pour les extensions du modèle.

---

## 3. Architecture conceptuelle et dynamique jointe

Le modèle s'organise autour d'une base combinatoire et d'une fibre adaptative.

### A — Base combinatoire : paysage natif

L'espace des configurations est l'hypercube :

\[
X=(s_1,\ldots,s_N)\in\{-1,+1\}^N
\]

En bijection avec \(\{0,1\}^N\) :

\[
x_i=\frac{s_i+1}{2}
\]

La fitness native \(F_{\mathrm{NK}}(X)\) est déterminée par le paysage NK.

Chaque composant possède une contribution locale dépendant de son état et des \(K\) autres composants associés à son voisinage épistatique.

### B — Fibre adaptative et connexion discrète

Au-dessus de chaque configuration, la fibre est représentée par :

\[
J\in\mathrm{Sym}_N(\mathbb R)
\]

avec :

\[
J=J^T,\qquad J_{ii}=0
\]

La matrice \(J\) représente la structure des interactions adaptatives entre les composants.

#### Convention stricte post-flip

Lorsqu'un nœud \(k\) mute, son spin bascule d'abord :

\[
s_k'=-s_k
\]

Le transport est ensuite évalué sur l'état post-flip \(X'\).

Pour \(i\neq j\), la fonction de transport est :

\[
\Psi_{ij}(X',J)
=
\frac{1}{2}
\sum_{r\notin\{i,j\}}
(J_{jr}+J_{ir})s_r'
\]

Seules les arêtes incidentes au nœud muté \(k\) sont mises à jour :

\[
J'_{kj}
=
J_{kj}+\eta\Psi_{kj}(X',J)
\]

La symétrie est conservée par :

\[
J'_{jk}=J'_{kj}
\]

et la diagonale demeure nulle.

Le paramètre \(\eta\) contrôle l'intensité du transport.

#### Comparaison des ordres de mutation

Pour deux nœuds distincts \(k\) et \(m\), les deux compositions sont :

\[
U_m\circ U_k
\qquad\text{et}\qquad
U_k\circ U_m
\]

Les flips sur la base combinatoire commutent :

\[
X^{(k)(m)}=X^{(m)(k)}
\qquad(k\neq m)
\]

La fibre, en revanche, peut dépendre de l'ordre d'application des transports.

On définit l'observable de non-commutativité :

\[
\mathcal{C}_{km}(X,J)
=
\left\|
J_{k\rightarrow m}
-
J_{m\rightarrow k}
\right\|_F
\]

où les deux matrices finales sont obtenues en appliquant les mutations et les transports dans les ordres indiqués.

Une valeur strictement positive caractérise une dépendance à l'ordre dans la règle de transport étudiée.

Dans ce projet, cette différence est interprétée opérationnellement comme une courbure discrète ou une holonomie de face. Cette terminologie désigne une propriété des compositions de transports. Elle ne constitue pas, à elle seule, une démonstration de toutes les propriétés d'une connexion géométrique continue.

### C — Dynamique jointe et sélection

La dynamique complète envisagée associe trois mécanismes :

1. mutation de la base ;
2. transport instantané de la fibre lors d'une mutation acceptée ;
3. dissipation continue entre les mutations.

#### 1. Fitness étendue

La fitness du système est définie par :

\[
F(X,J)
=
F_{\mathrm{NK}}(X)
+
\lambda\sum_{i<j}J_{ij}s_is_j
\]

Pour une matrice symétrique à diagonale nulle :

\[
F(X,J)
=
F_{\mathrm{NK}}(X)
+
\frac{\lambda}{2}X^TJX
\]

Le paramètre \(\lambda\) pondère la rétroaction de la fibre sur la sélection.

Le cas :

\[
\lambda=0
\]

isole le régime dans lequel la fibre n'intervient pas dans la fitness, tout en pouvant continuer à évoluer par transport.

#### 2. Mutation et taux de transition

Le taux de bascule du nœud \(k\) est :

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

et :

\[
a_k(J)=\mathbf{1}[A_k(J)>\theta_A]
\]

où :

- \(\mu\) est l'échelle des taux ;
- \(\beta\) contrôle la sensibilité à la variation de fitness ;
- \(\theta_A\) est le seuil d'accessibilité ;
- \(\lambda\) pondère le couplage dans la fitness.

Dans cette règle, le taux est évalué sur l'état courant \((X,J)\). Lorsqu'une mutation est acceptée, le spin bascule, puis le transport post-flip met à jour la fibre.

La transition complète est donc une mise à jour de l'état augmenté :

\[
(X,J)
\longrightarrow
\left(X^{(k)},U_k(J)\right)
\]

#### 3. Dissipation continue

Entre deux mutations, la fibre est conçue pour relaxer exponentiellement vers une matrice de référence \(J^{(0)}\) :

\[
\frac{dJ}{dt}
=
-\gamma(J-J^{(0)})
\]

La solution entre deux événements est :

\[
J(t)
=
J^{(0)}
+
\left(J(t_0)-J^{(0)}\right)
e^{-\gamma(t-t_0)}
\]

où \(\gamma\) est le coefficient de dissipation.

Cette évolution préserve la symétrie et la diagonale nulle lorsque \(J(t_0)\) et \(J^{(0)}\) respectent ces contraintes.

#### 4. Méthode d'intégration stochastique envisagée

Lorsque \(\lambda\neq0\), les taux peuvent varier continuellement entre deux mutations en raison de l'évolution de \(J(t)\).

Une simulation par amincissement d'Ogata (*thinning*) est prévue.

Sous les conditions :

\[
\mu\geq0,\qquad 0\leq a_k(J)\leq1
\]

et :

\[
0\leq\min(1,e^{\beta\Delta F_k})\leq1
\]

on obtient :

\[
0\leq w_k(X,J)\leq\mu
\]

et donc :

\[
\sum_{k=1}^{N}w_k(X,J)\leq N\mu
\]

Cette borne fournit une intensité supérieure pour le processus de proposition utilisé par l'amincissement.

La validation de l'intégration stochastique complète reste à effectuer, notamment pour vérifier la gestion des taux variables, des événements acceptés et de la relaxation entre événements.

### D — Reconstruction temporelle par LSTM

Les réseaux *Long Short-Term Memory* sont envisagés comme observateurs temporels de la fibre adaptative.

L'entrée principale est la séquence des configurations :

\[
X_{t_0},X_{t_1},\ldots,X_{t_n}
\]

L'objectif est de reconstruire une estimation de la fibre latente :

\[
\widehat J_t
=
\operatorname{LSTM}(X_{t_0},\ldots,X_t)
\]

Le projet examine si la séquence historique permet d'estimer une structure d'interactions qui n'est pas déterminée par la seule configuration présente \(X_t\).

Cette capacité doit être évaluée par comparaison avec des modèles de référence, sur des trajectoires et des paysages de test séparés de l'entraînement.

Aucune supériorité du LSTM sur les modèles de référence n'est revendiquée à ce stade.

---

## 4. Modèles expérimentaux comparés

Quatre architectures d'inférence sont prévues.

| Modèle | Entrée | Fonction |
| :--- | :--- | :--- |
| M1 — État présent | \(X_t\) | Référence instantanée |
| M2 — État et fitness | \(X_t,F_t\) | Référence enrichie |
| M3 — Fenêtre historique | Fenêtre finie de \(X\) | Mémoire explicite non récurrente |
| M4 — Séquence historique | Séquence de \(X\) | Inférence récurrente par LSTM |

### Plan factoriel \(2\times2\) de la dynamique

Le plan expérimental prévu croise la règle de transport et la rétroaction de la fibre sur la fitness.

| Règle de transport | Avec rétroaction \((\lambda\neq0)\) | Sans rétroaction \((\lambda=0)\) |
| :--- | :--- | :--- |
| Transport symétrique non commutatif | Modèle complet | Transport symétrique pur |
| Transport intégrable commutatif | Rétroaction sans non-commutativité du transport | Ligne de base |

Ce plan doit permettre de distinguer les effets associés à la non-commutativité de ceux associés à la rétroaction sélective.

Son exécution complète et l'évaluation statistique des quatre conditions restent à réaliser.

---

## 5. Protocoles expérimentaux et résultats

Les validations sont organisées selon deux échelles principales :

- \(N=3\) : audit algébrique exact sur une face élémentaire ;
- \(N=6\) : comparaison numérique des ordres de mutation, balayage de \(\eta\) et audits de reproductibilité multi-graines.

Les simulations dynamiques de plus grande échelle, notamment \(N=8\) et \(N=20\), ne sont pas encore validées.

### 5.1 — Audit algébrique exact sur une face élémentaire, \(N=3\)

On considère :

\[
X_0=(s_1,s_2,s_3)\in\{-1,+1\}^3
\]

et la matrice initiale :

\[
J^{(0)}
=
\begin{pmatrix}
0&a&b\\
a&0&c\\
b&c&0
\end{pmatrix}
\]

avec :

\[
a=J_{12}^{(0)},\qquad
b=J_{13}^{(0)},\qquad
c=J_{23}^{(0)}
\]

L'accessibilité du troisième nœud est :

\[
A_3(J)=J_{13}^{2}+J_{23}^{2}
\]

Les deux chemins comparés sont :

\[
A:1\rightarrow2
\qquad\text{et}\qquad
B:2\rightarrow1
\]

La convention stricte post-flip donne les ordres :

\[
J_A=U_2(U_1(J^{(0)}))
\]

\[
J_B=U_1(U_2(J^{(0)}))
\]

En posant :

\[
\Delta J=J_A-J_B
\]

l'audit algébrique exact établit les ordres suivants :

\[
\Delta J_{12}=O(\eta^2)
\]

\[
\Delta J_{13}=O(\eta)
\]

\[
\Delta J_{23}=O(\eta)
\]

L'écart d'accessibilité s'écrit exactement :

\[
\begin{aligned}
\Delta A_3
&=
\left(J_{13}^{(A)}\right)^2
-
\left(J_{13}^{(B)}\right)^2\\
&\quad+
\left(J_{23}^{(A)}\right)^2
-
\left(J_{23}^{(B)}\right)^2
\end{aligned}
\]

soit :

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

Au premier ordre en \(\eta\) :

\[
\boxed{
\Delta A_3
=
2\eta
\left[
b(a+c)s_2-c(a+b)s_1
\right]
+
O(\eta^2)
}
\]

Cet audit établit une expression analytique de l'écart d'accessibilité sur la face élémentaire étudiée.

Il ne valide pas, à lui seul, la dynamique jointe avec dissipation ni la persistance temporelle de l'accessibilité.

### 5.2 — Audit numérique de non-commutativité et balayage de \(\eta\), \(N=6\)

Le protocole compare les deux ordres d'application de deux mutations distinctes :

\[
k\rightarrow m
\qquad\text{et}\qquad
m\rightarrow k
\]

#### Paramètres de référence

| Paramètre | Valeur |
| :--- | ---: |
| Nombre de sites \(N\) | 6 |
| Nombre de configurations initiales | 10 |
| Paires distinctes par configuration | 15 |
| Comparaisons par valeur de \(\eta\) | 150 |
| Échelle initiale de \(J\) | 0.2 |
| Graine du paysage NK | 7 |
| Dépendances NK \(K\) | 3 |
| \(\mu\) | 1.0 |
| \(\beta\) | 1.0 |
| \(\theta_A\) | 0.1 |
| \(\lambda\) | 0.5 |

Le même paysage NK est utilisé pour les comparaisons d'une valeur donnée de \(\eta\).

Le balayage porte sur :

\[
\eta\in
\{0,\;0.05,\;0.1,\;0.25,\;0.5,\;1.0\}
\]

Chaque ligne correspond à 150 comparaisons.

#### Résultats du balayage

| \(\eta\) | Comparaisons | Écart moyen \(\|J_A-J_B\|_F\) | Écart maximal | Erreur symétrie | Erreur diagonale | \(X\) identique | \(\Delta w\neq0\) | Désaccords \(D_{\mathcal T}\) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 150 | 0.000000 | 0.000000 | 0 | 0 | 100.0 % | 0.0 % | 0 / 900 (0.0 %) |
| 0.05 | 150 | 0.037462 | 0.084396 | 0 | 0 | 100.0 % | 68.0 % | 20 / 900 (2.2 %) |
| 0.10 | 150 | 0.075777 | 0.174236 | 0 | 0 | 100.0 % | 68.7 % | 34 / 900 (3.8 %) |
| 0.25 | 150 | 0.197552 | 0.478854 | 0 | 0 | 100.0 % | 80.0 % | 91 / 900 (10.1 %) |
| 0.50 | 150 | 0.431533 | 1.118733 | 0 | 0 | 100.0 % | 93.3 % | 178 / 900 (19.8 %) |
| 1.00 | 150 | 1.060533 | 3.053254 | 0 | 0 | 100.0 % | 97.3 % | 213 / 900 (23.7 %) |

#### Définition exacte des indicateurs

**Écart géométrique**

L'écart moyen et l'écart maximal sont calculés sur les 150 comparaisons de matrices finales :

\[
D_J^{(q)}
=
\|J_A^{(q)}-J_B^{(q)}\|_F
\]

**Différence des taux**

Une comparaison est comptée lorsque :

\[
\max_k
|w_{A,k}-w_{B,k}|
>
10^{-12}
\]

La colonne \(\Delta w\neq0\) représente le pourcentage de comparaisons satisfaisant ce critère. Il ne s'agit pas du pourcentage de composantes individuelles de taux différentes.

**Désaccords d'accessibilité**

Pour chaque comparaison \(q\) :

\[
D_{\mathcal T}^{(q)}
=
\operatorname{card}
\left(
\mathcal T(J_A^{(q)})
\triangle
\mathcal T(J_B^{(q)})
\right)
\]

Le compteur agrégé est :

\[
C_{\mathcal T}
=
\sum_{q=1}^{150}D_{\mathcal T}^{(q)}
\]

Le pourcentage est calculé sur les \(150\times6=900\) comparaisons site par site :

\[
C_{\mathcal T}^{(\%)}
=
100\,
\frac{C_{\mathcal T}}{150\times6}
\]

**Correction terminologique :** le compteur précédemment désigné comme « franchissements de seuil » mesure en réalité les désaccords entre les masques d'accessibilité finaux des deux chemins. Il ne mesure pas des franchissements temporels du seuil le long d'une trajectoire.

#### Observations établies

**Contrôle nul**

À \(\eta=0\), les deux ordres conduisent à la même fibre :

\[
\|J_A-J_B\|_F=0
\]

Aucun désaccord de masque n'est observé.

**Augmentation de l'écart géométrique**

L'écart moyen et l'écart maximal augmentent pour chacune des valeurs de \(\eta\) testées.

Dans cette grille, l'écart moyen passe de :

\[
0.037462\quad(\eta=0.05)
\]

à :

\[
1.060533\quad(\eta=1.0)
\]

Cette observation indique une augmentation de la non-commutativité mesurée avec l'intensité du transport, dans le domaine testé. Elle ne démontre pas une loi générale de croissance ni une relation analytique universelle avec \(\eta\).

**Conservation de la base**

La fraction d'états finaux \(X\) identiques est de 100 % pour toutes les valeurs de \(\eta\).

Cela est cohérent avec la commutation de deux flips distincts sur la base combinatoire :

\[
X^{(k)(m)}=X^{(m)(k)}
\]

La divergence mesurée se situe donc dans la fibre \(J\), et non dans l'état final des spins pour ce protocole.

**Conservation des contraintes structurelles**

Les erreurs maximales de symétrie et de diagonale sont nulles pour toutes les valeurs de \(\eta\) :

\[
J=J^T
\]

\[
J_{ii}=0
\]

Les contraintes structurelles testées sont donc respectées dans les 150 comparaisons de chaque ligne.

**Transmission vers les observables de transition**

La fraction de comparaisons présentant une différence non nulle entre les vecteurs de taux augmente globalement avec \(\eta\), de 68.0 % à \(\eta=0.05\) à 97.3 % à \(\eta=1.0\).

La différence entre les fibres se répercute donc sur les taux calculés.

Cette variation peut provenir des différences d'accessibilité et des termes de fitness. Le protocole présenté ne sépare pas encore quantitativement ces contributions.

### 5.3 — Audit de reproductibilité multi-graines, \(N=6\)

Un deuxième audit a été exécuté afin de séparer l'effet des conditions initiales de celui du paysage NK.

Chaque exécution utilise :

- \(N=6\) ;
- 10 configurations initiales par graine ;
- 15 paires distinctes par configuration ;
- 150 comparaisons par ligne ;
- \(\eta=0.25\) ;
- \(\mu=1.0\) ;
- \(\beta=1.0\) ;
- \(\theta_A=0.1\) ;
- \(\lambda=0.5\) ;
- \(K=3\) ;
- échelle initiale de \(J=0.2\).

Les graines comparées sont :

\[
\{7,13,21,42,99\}
\]

L'audit comprend deux blocs indépendants.

#### Bloc A — Variation de la graine initiale

Le paysage NK est maintenu fixe :

\[
nk\_seed=7
\]

La graine initiale des configurations \(X_0,J_0\) varie.

| Graine initiale | Graine NK | Écart moyen \(J\) | Écart maximal \(J\) | Err. sym. | Err. diag. | \(X\) identique | \(\Delta w\neq0\) | Désaccords \(D_{\mathcal T}\) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 0.197552 | 0.478854 | 0 | 0 | 100.0 % | 82.7 % | 91 (10.1 %) |
| 13 | 7 | 0.205107 | 0.489502 | 0 | 0 | 100.0 % | 86.7 % | 77 (8.6 %) |
| 21 | 7 | 0.188030 | 0.426054 | 0 | 0 | 100.0 % | 79.3 % | 90 (10.0 %) |
| 42 | 7 | 0.198132 | 0.460827 | 0 | 0 | 100.0 % | 85.3 % | 90 (10.0 %) |
| 99 | 7 | 0.217580 | 0.443931 | 0 | 0 | 100.0 % | 94.0 % | 100 (11.1 %) |

Synthèse sur les cinq graines :

| Indicateur | Moyenne ± écart-type |
| :--- | :---: |
| Écart moyen \(J\) | \(0.201280\pm0.009795\) |
| Comparaisons avec \(\Delta w\neq0\) | \(85.6\%\pm4.9\%\) |
| Désaccords \(D_{\mathcal T}\) | \(89.6\pm7.3\) |

Ces résultats montrent une variabilité entre les configurations initiales dans le protocole testé.

Les contraintes suivantes restent satisfaites :

- \(X_A=X_B\) dans 100 % des comparaisons ;
- erreur de symétrie nulle ;
- erreur diagonale nulle.

#### Bloc B — Variation de la graine du paysage NK

La configuration initiale est maintenue fixe :

\[
init\_seed=7
\]

La graine du paysage NK varie.

| Graine initiale | Graine NK | Écart moyen \(J\) | Écart maximal \(J\) | Err. sym. | Err. diag. | \(X\) identique | \(\Delta w\neq0\) | Désaccords \(D_{\mathcal T}\) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 0.197552 | 0.478854 | 0 | 0 | 100.0 % | 82.7 % | 91 (10.1 %) |
| 7 | 13 | 0.197552 | 0.478854 | 0 | 0 | 100.0 % | 81.3 % | 91 (10.1 %) |
| 7 | 21 | 0.197552 | 0.478854 | 0 | 0 | 100.0 % | 81.3 % | 91 (10.1 %) |
| 7 | 42 | 0.197552 | 0.478854 | 0 | 0 | 100.0 % | 80.0 % | 91 (10.1 %) |
| 7 | 99 | 0.197552 | 0.478854 | 0 | 0 | 100.0 % | 81.3 % | 91 (10.1 %) |

Synthèse sur les cinq paysages NK :

| Indicateur | Moyenne ± écart-type |
| :--- | :---: |
| Écart moyen \(J\) | \(0.197552\pm0.000000\) |
| Comparaisons avec \(\Delta w\neq0\) | \(81.3\%\pm0.8\%\) |
| Désaccords \(D_{\mathcal T}\) | \(91.0\pm0.0\) |

Les indicateurs géométriques sont invariants entre les cinq réalisations NK.

Cette invariance est attendue de l'architecture actuelle : la fonction de transport `mutate_postflip` ne reçoit pas le paysage NK comme argument. Lorsque \(X_0,J_0,\eta\) et les paires de mutations sont identiques, changer le paysage NK ne modifie donc pas les fibres finales calculées.

Le résultat constitue un contrôle de cohérence et de séparation des modules du code. Il ne constitue pas une découverte indépendante sur les paysages NK.

Les taux de transition, en revanche, varient selon le paysage utilisé pour calculer la fitness. Le bloc B confirme ainsi une séparation entre :

- la géométrie du transport, déterminée ici par les états initiaux et \(\eta\) ;
- les taux calculés, qui dépendent également de la réalisation du paysage NK.

#### Limite statistique de la synthèse

Les écarts-types indiqués sont des écarts-types descriptifs calculés sur les cinq résultats par graine, avec la convention de population (`np.std`, `ddof=0`).

Ils ne sont ni des intervalles de confiance ni des tests de significativité. Les cinq graines constituent un audit exploratoire de reproductibilité.

#### Écart à réconcilier : 80.0 % contre 82.7 %

Une différence numérique demeure entre deux exécutions réalisées avec les paramètres de référence :

- l'ancien audit et le balayage de \(\eta\) rapportent 80.0 % de comparaisons avec \(\Delta w\neq0\) à \(\eta=0.25\) ;
- le nouvel audit multi-graines rapporte 82.7 % pour `init_seed=7` et `nk_seed=7`.

Les indicateurs géométriques concordent pourtant exactement :

\[
\overline{D_J}=0.197552
\]

\[
D_{J,\max}=0.478854
\]

\[
C_{\mathcal T}=91
\]

Cette différence de 2.7 points de pourcentage n'est pas réconciliée dans la présente version.

Les deux résultats ne doivent pas être fusionnés ni moyennés. Il faut comparer les versions des scripts, la construction du paysage NK, les données initiales et le calcul effectif des taux avant de retenir une valeur de référence unique.

### 5.4 — Protocole de seuil \(\theta_A\)

Le seuil \(\theta_A\) est traité comme un paramètre explicite du modèle.

Un premier balayage exploratoire a été effectué sur 20 000 tirages :

\[
J\sim\mathcal U([-1,1]^3)
\]

pour la représentation à trois sites par les trois coefficients indépendants d'une matrice symétrique à diagonale nulle.

Cette première exploration utilisait une grille grossière et ne comportait pas de barres d'erreur.

Elle suggérait une dépendance des résultats de franchissement du seuil à l'intensité du transport sur une seule étape.

Ce résultat reste exploratoire. Il devra être reproduit avec :

- une grille de seuils publiée intégralement ;
- une grille raffinée ;
- plusieurs graines aléatoires ;
- des intervalles de confiance ;
- une distinction explicite entre les franchissements au cours d'une trajectoire et les désaccords entre deux masques finaux.

Aucun seuil de référence ne doit être calibré a posteriori sur un maximum empirique sans justification indépendante.

---

## 6. Hypothèses de recherche

Les hypothèses sont formulées comme des propositions falsifiables. Leur état est distingué des observations numériques déjà obtenues.

### H1 — Non-commutativité et géométrie discrète

Le transport symétrique de la fibre produit une non-commutativité sur certaines faces élémentaires de l'hypercube.

La structure spectrale linéarisée, notée \(M_{\mathrm{gelé}}\), pourrait également dépendre de la classe de configuration.

**État :** non-commutativité mesurée dans les audits \(N=3\) et \(N=6\). La dépendance spectrale, la relation avec la parité et la portée générale de l'hypothèse restent à tester.

### H2 — Mémoire métrique et accessibilité

La dissipation peut réduire la distance entre deux fibres issues de chemins distincts.

La règle d'accessibilité actuelle étant une fonction déterministe et seuillée de \(J\), une divergence de masque ne peut cependant pas persister asymptotiquement lorsque \(D_J\to0\) si les deux fibres restent bornées et si les marges au seuil sont uniformément positives.

L'hypothèse doit donc être séparée en deux propositions :

- **H2a — Accessibilité transitoire :** une différence d'accessibilité peut persister temporairement lorsque les deux fibres sont distinctes, notamment lorsque certains scores \(A_k(J)\) sont proches du seuil.
- **H2b — Accessibilité avec mémoire propre :** une différence d'accessibilité peut subsister après convergence géométrique uniquement si une variable dépendante de l'histoire, une règle d'hystérésis ou un état interne supplémentaire est introduit.

**État :** non testé par les audits de non-commutativité présentés ici. La dynamique avec dissipation et le suivi temporel de \(D_J(t)\), \(D_{\mathcal T}(t)\) et des marges restent à réaliser.

### H3 — Organisation latente et reconstruction LSTM

Le LSTM (M4) pourrait reconstruire la fibre latente \(J_t\) et prédire les changements d'accessibilité à partir de la séquence des configurations.

La comparaison doit être réalisée contre M1, M2 et M3, sur des trajectoires indépendantes de celles utilisées pour l'entraînement.

**État :** à tester. Aucune performance d'inférence ni supériorité du LSTM n'est revendiquée à ce stade.

### H4 — Invariance transversale au paysage NK

Les propriétés géométriques du transport pourraient rester invariantes entre différentes réalisations du paysage natif \(F_{\mathrm{NK}}\), lorsque les conditions initiales et les règles de transport sont identiques.

**État :** invariance observée dans l'audit statique multi-graines \(N=6\), conformément à la séparation actuelle entre `mutate_postflip` et le calcul de la fitness NK. La généralisation aux propriétés spectrales et à la dynamique jointe reste à tester.

---

## 7. Contrôles et falsification

La validation du modèle repose sur cinq contrôles.

### 1. Contrôle de transport intégrable

Un contrôle candidat remplace la règle de transport symétrique par :

\[
\Psi_{kj}^{\mathrm{int}}(X')
=
s_k's_j'
\]

où les spins sont évalués sur l'état post-flip.

La mise à jour correspondante est :

\[
J'_{kj}=J_{kj}+\eta\Psi_{kj}^{\mathrm{int}}(X')
\]

avec symétrisation de l'arête.

Le contrôle vise à construire un transport dont les compositions associées à deux flips distincts commutent :

\[
U_m^{\mathrm{int}}\circ U_k^{\mathrm{int}}
=
U_k^{\mathrm{int}}\circ U_m^{\mathrm{int}}
\]

L'écart final attendu est :

\[
\left\|
J_{k\rightarrow m}^{\mathrm{int}}
-
J_{m\rightarrow k}^{\mathrm{int}}
\right\|_F=0
\]

Cette égalité ne signifie pas que chaque mise à jour individuelle de \(J\) est nulle. Elle signifie que le résultat final ne dépend pas de l'ordre des deux transports.

L'intégrabilité de ce contrôle doit être vérifiée par des tests unitaires avant son utilisation comme référence expérimentale.

### 2. Mélange de l'ordre temporel

Les séquences historiques sont permutées afin de déterminer si les modèles d'inférence exploitent l'ordre temporel des observations plutôt qu'une simple distribution des états.

La comparaison doit conserver les mêmes ensembles d'entraînement et de test.

### 3. Contrôle de parité

Le transport symétrique est examiné sous la transformation :

\[
X\rightarrow-X
\]

Le protocole compare les observables, les spectres et les commutateurs entre les deux configurations, en explicitant la transformation appliquée à \(J\) et les conditions initiales.

### 4. Contrôle de seuil

Le seuil dur est comparé à une sigmoïde lissée :

\[
a_k^{(\delta)}(J)
=
\frac{1}
{1+\exp\left[-(A_k(J)-\theta_A)/\delta\right]}
\]

Le paramètre \(\delta\) contrôle la largeur de la zone de transition.

L'analyse doit publier les résultats en fonction de \(\theta_A\) et de \(\delta\), avec intervalles d'incertitude et plusieurs graines.

### 5. Contrôle des fuites d'information

Pour l'inférence de \(J_t\) :

- les séquences de test doivent être indépendantes des séquences d'entraînement ;
- les paysages NK de test doivent être séparés de ceux utilisés pour l'entraînement lorsque la généralisation aux paysages est étudiée ;
- les fenêtres provenant d'une même trajectoire ne doivent pas être réparties aléatoirement entre entraînement et test ;
- \(J_t\), lorsqu'il constitue la cible supervisée, ne doit pas être fourni comme variable d'entrée au modèle.

Les performances doivent être rapportées séparément pour la reconstruction de \(J_t\), la prédiction des taux et la prédiction des masques d'accessibilité.

---

## 8. Rapport conceptuel avec l'intégrale de chemin

Le projet établit un rapprochement structurel avec les formulations par chemins : une évolution peut être étudiée à partir de l'ensemble des trajectoires admissibles et des poids associés.

Une écriture formelle possible est :

\[
\mathcal{Z}_{\mathrm{chemins}}
=
\sum_{\text{chemins}}
e^{-S_{\mathrm{eff}}(\text{chemin})}
\]

où \(S_{\mathrm{eff}}\) représenterait une action effective dépendant de la structure du réseau et de l'historique des transports sur la fibre.

Cette écriture est utilisée comme analogie de somme de poids sur des trajectoires. L'action \(S_{\mathrm{eff}}\) n'est pas encore dérivée à partir du processus stochastique défini dans ce projet.

Elle ne constitue pas une amplitude de Feynman en temps réel et ne postule aucune équivalence physique avec la mécanique quantique.

Le rapprochement est strictement structurel : le chemin parcouru peut modifier la fibre et, par conséquent, les possibilités ultérieures du système.

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
│   ├── audit_robustesse_eta_N6.md
│   └── audit_reproductibilite_multigraines_N6.md
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
│   ├── audit_reproductibilite_graines.py
│   └── controls.py
│
└── tests/
    ├── test_curvature_N3.py
    ├── test_nk.py
    ├── test_leakage.py
    └── test_integrable_transport.py
```

Cette arborescence est une cible d'organisation. Les chemins effectifs des fichiers devront être harmonisés lors de leur intégration au dépôt.

Les résultats expérimentaux doivent être accompagnés des versions exactes des scripts qui les produisent.

---

## 10. État des validations

| Élément | État |
| :--- | :--- |
| Audit algébrique exact sur la face \(N=3\) | Vérifié |
| Audit numérique de non-commutativité \(N=6\) | Exécuté |
| Balayage de six intensités \(\eta\) | Exécuté |
| Conservation de la symétrie de \(J\) dans l'audit \(N=6\) | Vérifiée |
| Conservation de la diagonale nulle de \(J\) | Vérifiée |
| Conservation de \(X\) entre les deux ordres de flips | Vérifiée |
| Désaccords entre masques finaux \(D_{\mathcal T}\) | Mesurés |
| Définition du compteur précédemment appelé « franchissements » | Clarifiée : désaccords de masques |
| Audit multi-graines sur les conditions initiales | Exécuté |
| Audit multi-graines sur les paysages NK | Exécuté |
| Invariance géométrique sous variation de `nk_seed` | Observée et conforme à l'architecture du code |
| Écart 80.0 % / 82.7 % sur \(\Delta w\neq0\) | À réconcilier |
| Appariement exact des états initiaux entre les valeurs de \(\eta\) | À confirmer dans le code |
| Contrôle de transport intégrable | À implémenter et tester |
| Dynamique jointe avec dissipation | À exécuter et tester |
| Persistance temporelle de \(D_J(t)\) | À tester |
| Persistance temporelle de \(D_{\mathcal T}(t)\) | À tester |
| Robustesse de l'accessibilité hors proximité du seuil | À tester |
| Comparaison M1–M4 | À exécuter |
| Généralisation à \(N=8\) et \(N=20\) | À exécuter |

---

## 11. Prochaines étapes expérimentales

Les étapes sont ordonnées selon leurs dépendances.

### Étape 1 — Réconciliation de l'audit de référence

Résoudre l'écart entre les valeurs de 80.0 % et 82.7 % pour \(\Delta w\neq0\) à \(\eta=0.25\).

Conserver les mêmes :

- configurations initiales ;
- matrices \(J_0\) ;
- paires de mutations ;
- voisinages et tables du paysage NK ;
- paramètres de taux ;
- tolérance numérique ;
- versions des fonctions de transport et de fitness.

L'objectif est de reproduire exactement le résultat à partir d'un même état initial et d'un même paysage.

### Étape 2 — Verrouillage de la reproductibilité

Pour chaque expérience, enregistrer :

- les graines ;
- les paramètres complets ;
- les versions du code ;
- les configurations \(X_0\) ;
- les matrices \(J_0\) ;
- les voisinages et les tables NK ;
- les sorties numériques brutes.

L'appariement des réalisations initiales entre les valeurs de \(\eta\) doit être confirmé directement dans le code.

### Étape 3 — Validation du contrôle intégrable

Implémenter le transport candidat, vérifier sa commutation par tests unitaires et comparer ses résultats à ceux du transport symétrique.

### Étape 4 — Dynamique jointe et dissipation

Implémenter et tester le processus comprenant les mutations, le transport post-flip et la relaxation continue.

Contrôler la borne d'intensité utilisée par l'amincissement, la gestion des événements et la conservation des contraintes de la fibre.

### Étape 5 — Persistance temporelle de la mémoire métrique et de l'accessibilité

Suivre simultanément :

\[
D_J(t),\qquad
D_{\mathcal T}(t),\qquad
m_{k,A}(t),\qquad
m_{k,B}(t)
\]

Distinguer les divergences transitoires produites par la proximité du seuil d'une éventuelle mémoire d'accessibilité dépendante de l'histoire.

### Étape 6 — Reconstruction LSTM

Construire les jeux de données temporels, entraîner M1–M4 et comparer leurs capacités de reconstruction et de prédiction sur des trajectoires et des paysages indépendants.

### Étape 7 — Extension de l'échelle

Après validation des contrôles précédents, étendre progressivement les expériences à :

\[
N=8
\qquad\text{puis}\qquad
N=20
\]

La complexité numérique ne doit pas précéder la validation des invariants, de la reproductibilité et des métriques.

---

## 12. Critère de progression scientifique

Chaque extension du modèle doit conserver une distinction explicite entre :

1. **Résultat algébrique :** propriété démontrée à partir des règles mathématiques définies.
2. **Observation numérique :** résultat obtenu dans un protocole donné, avec paramètres, graines et limites documentés.
3. **Hypothèse interprétative :** propriété proposée qui exige une expérience discriminante.
4. **Validation logicielle :** conformité de l'implémentation aux spécifications et aux tests.

Une non-commutativité mesurée dans la fibre ne suffit pas à établir une mémoire topologique persistante.

Une différence entre deux masques d'accessibilité ne suffit pas à démontrer une mémoire indépendante du seuil.

Une reconstruction numérique de \(J_t\) ne suffit pas, à elle seule, à établir une propriété cognitive du réseau LSTM.

Le programme de recherche progresse donc par validations successives :

\[
\boxed{
\text{Géométrie discrète}
\longrightarrow
\text{Dynamique jointe}
\longrightarrow
\text{Persistance}
\longrightarrow
\text{Reconstruction temporelle}
}
\]

L'objectif est de déterminer précisément quelles informations historiques sont conservées dans la structure adaptative, dans quelles conditions elles influencent les possibilités futures et jusqu'à quel point elles peuvent être reconstruites à partir des trajectoires observées.
