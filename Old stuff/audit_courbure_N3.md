# Audit algébrique de la courbure discrète — N = 3

**Projet :** NK-LSTM-PATH
**Version :** 0.2
**Statut :** Dérivation analytique et validation numérique effectuées (`src/fiber_bundle.py`, `tests/test_curvature_N3.py`)
**Périmètre :** Face élémentaire de mutations (1, 2), transport symétrique, convention stricte post-flip.

---

## 1. Objet de l'audit

Cet audit examine la dépendance à l'ordre de deux mutations successives dans une fibre de couplages symétriques.

L'objectif est de calculer :

\[
\Delta J = J^{(A)} - J^{(B)}
\]

pour les deux chemins :

\[
A: 1 \rightarrow 2,
\qquad
B: 2 \rightarrow 1.
\]

On cherche également à déterminer les conséquences de cette différence sur le critère d'accessibilité :

\[
A_k(J) = \sum_{j \neq k} J_{kj}^{2}.
\]

Le présent document établit et valide numériquement les résultats algébriques exacts pour \(N = 3\). Il ne constitue pas encore une démonstration de mémoire topologique persistante ni une extension à \(N > 3\).

---

## 2. Définitions et conventions

L'état initial est :

\[
X_0 = (s_1, s_2, s_3),
\qquad s_i \in \{-1, +1\}.
\]

La fibre symétrique possède trois coordonnées indépendantes :

\[
\mathbf{j} = (J_{12}, J_{13}, J_{23}).
\]

On pose :

\[
a = J_{12}^{(0)},\qquad
b = J_{13}^{(0)},\qquad
c = J_{23}^{(0)}.
\]

La règle de transport symétrique est :

\[
\Psi_{ij}(X, J)
=
\frac{1}{2}
\sum_{r \notin \{i, j\}}
(J_{jr} + J_{ir}) s_r
\quad (i \neq j),
\qquad
\Psi_{ii}(X, J) = 0.
\]

Pour \(N = 3\), chaque arête incidente à une mutation possède un seul terme dans cette somme.

**Convention stricte post-flip :** le spin muté est inversé (\(s_k' = -s_k\)) avant l'évaluation de \(\Psi\). Les autres spins conservent leur valeur.

La mise à jour est :

\[
J_{ij}' = J_{ij} + \eta \Psi_{ij}.
\]

Les arêtes non incidentes au nœud muté restent inchangées.

On définit le défaut de commutation par :

\[
\Delta J_{ij} = J_{ij}^{(A)} - J_{ij}^{(B)}.
\]

---

## 3. Chemin A : mutation 1 puis mutation 2

### Première mutation : nœud 1

Le spin devient \(-s_1\).

Les mises à jour sont :

\[
J_{12}^{(1,A)}
=
a + \frac{\eta}{2}(b + c)s_3,
\]

\[
J_{13}^{(1,A)}
=
b + \frac{\eta}{2}(a + c)s_2,
\]

\[
J_{23}^{(1,A)} = c.
\]

### Deuxième mutation : nœud 2

Le spin \(s_2\) devient \(-s_2\). Le spin du nœud 1 est déjà égal à \(-s_1\).

On obtient :

\[
J_{12}^{(A)}
=
J_{12}^{(1,A)}
+ \frac{\eta}{2}
(J_{13}^{(1,A)} + c)s_3,
\]

\[
J_{23}^{(A)}
=
c - \frac{\eta}{2}
(J_{13}^{(1,A)} + J_{12}^{(1,A)})s_1,
\]

\[
J_{13}^{(A)} = J_{13}^{(1,A)}.
\]

Après développement exact :

\[
\boxed{
J_{12}^{(A)}
=
a + \eta(b + c)s_3
+ \frac{\eta^2}{4}(a + c)s_2 s_3
}
\]

\[
\boxed{
J_{13}^{(A)}
=
b + \frac{\eta}{2}(a + c)s_2
}
\]

\[
\boxed{
J_{23}^{(A)}
=
c - \frac{\eta}{2}(a + b)s_1
- \frac{\eta^2}{4}s_1
\left[(a + c)s_2 + (b + c)s_3\right]
}
\]

---

## 4. Chemin B : mutation 2 puis mutation 1

### Première mutation : nœud 2

Le spin devient \(-s_2\).

Les mises à jour sont :

\[
J_{12}^{(1,B)}
=
a + \frac{\eta}{2}(b + c)s_3,
\]

\[
J_{23}^{(1,B)}
=
c + \frac{\eta}{2}(a + b)s_1,
\]

\[
J_{13}^{(1,B)} = b.
\]

### Deuxième mutation : nœud 1

Le spin \(s_1\) devient \(-s_1\). Le spin du nœud 2 est déjà égal à \(-s_2\).

On obtient :

\[
J_{12}^{(B)}
=
J_{12}^{(1,B)}
+ \frac{\eta}{2}
(J_{23}^{(1,B)} + b)s_3,
\]

\[
J_{13}^{(B)}
=
b - \frac{\eta}{2}
(J_{23}^{(1,B)} + J_{12}^{(1,B)})s_2,
\]

\[
J_{23}^{(B)} = J_{23}^{(1,B)}.
\]

Après développement exact :

\[
\boxed{
J_{12}^{(B)}
=
a + \eta(b + c)s_3
+ \frac{\eta^2}{4}(a + b)s_1 s_3
}
\]

\[
\boxed{
J_{13}^{(B)}
=
b - \frac{\eta}{2}(a + c)s_2
- \frac{\eta^2}{4}s_2
\left[(a + b)s_1 + (b + c)s_3\right]
}
\]

\[
\boxed{
J_{23}^{(B)}
=
c + \frac{\eta}{2}(a + b)s_1
}
\]

---

## 5. Défaut de commutation

En soustrayant les résultats des deux chemins (\(X^{(A)} = X^{(B)} = (-s_1, -s_2, s_3)\), tandis que \(J^{(A)} \neq J^{(B)}\)) :

### Arête commune (1, 2)

\[
\boxed{
\Delta J_{12}
=
\frac{\eta^2}{4}s_3
\left[(a + c)s_2 - (a + b)s_1\right]
}
\]

### Arête périphérique (1, 3)

\[
\boxed{
\Delta J_{13}
=
\eta(a + c)s_2
+
\frac{\eta^2}{4}s_2
\left[(a + b)s_1 + (b + c)s_3\right]
}
\]

### Arête périphérique (2, 3)

\[
\boxed{
\Delta J_{23}
=
-\eta(a + b)s_1
- \frac{\eta^2}{4}s_1
\left[(a + c)s_2 + (b + c)s_3\right]
}
\]

Les ordres dominants sont donc asymétriques selon le type d'arête :

\[
\Delta J_{12} = O(\eta^2),
\qquad
\Delta J_{13} = O(\eta),
\qquad
\Delta J_{23} = O(\eta).
\]

L'arête commune aux deux mutations porte un défaut quadratique issu de l'interaction entre les deux mises à jour successives ; les arêtes périphériques portent un défaut linéaire dès le premier ordre en \(\eta\) dû à l'inversion d'ordre d'évaluation des spins mutés.

---

## 6. Conséquence sur l'accessibilité du nœud 3

Le critère d'accessibilité est :

\[
A_3(J) = J_{13}^2 + J_{23}^2.
\]

L'écart exact entre les deux chemins vérifie l'identité algébrique :

\[
\boxed{
\Delta A_3
=
A_3(J^{(A)}) - A_3(J^{(B)})
=
\Delta J_{13}\left(J_{13}^{(A)} + J_{13}^{(B)}\right)
+
\Delta J_{23}\left(J_{23}^{(A)} + J_{23}^{(B)}\right)
}
\]

Au premier ordre en \(\eta\) :

\[
\Delta A_3
=
2b\,\Delta J_{13}
+
2c\,\Delta J_{23}
+ O(\eta^2),
\]

soit :

\[
\boxed{
\Delta A_3
=
2\eta
\left[
b(a + c)s_2 - c(a + b)s_1
\right]
+ O(\eta^2)
}
\]

Le critère binaire d'accessibilité est :

\[
a_3(J) = \mathbf{1}[A_3(J) > \theta_A],
\qquad
m_3(J) = |A_3(J) - \theta_A|.
\]

C'est l'écart \(\Delta A_3\) entre les deux chemins qui est génériquement d'ordre \(O(\eta)\). Un basculement effectif \(a_3(J^{(A)}) \neq a_3(J^{(B)})\) n'a lieu que lorsque \(\theta_A\) sépare \(A_3(J^{(A)})\) et \(A_3(J^{(B)})\) ; dans ce régime calibré, les marges individuelles \(m_3(J^{(A)})\) et \(m_3(J^{(B)})\) sont d'ordre \(O(\eta)\).

---

## 7. Annulations du terme dominant et cas dégénérés

Il convient de distinguer strictement l'**annulation du terme dominant** (limite asymptotique lorsque \(\eta \to 0\)) et l'**annulation exacte** du défaut de commutation à \(\eta > 0\) fini.

### 7.1 Annulation du terme dominant linéaire (\(O(\eta)\))

Les coefficients dominants des arêtes périphériques sont :

\[
\lim_{\eta \to 0} \frac{\Delta J_{13}}{\eta} = (a + c)s_2,
\qquad
\lim_{\eta \to 0} \frac{\Delta J_{23}}{\eta} = -(a + b)s_1.
\]

Comme \(s_1, s_2 \in \{-1, +1\}\), ces termes dominants s'annulent respectivement lorsque :

\[
a + c = 0 \quad (\text{pour } \Delta J_{13}),
\qquad
a + b = 0 \quad (\text{pour } \Delta J_{23}).
\]

Cependant, pour \(\eta > 0\) fini, le terme quadratique subsiste génériquement :

\[
\Delta J_{13}\big|_{a+c=0}
=
\frac{\eta^2}{4}s_2\left[(a + b)s_1 + (b + c)s_3\right] = O(\eta^2),
\]

\[
\Delta J_{23}\big|_{a+b=0}
=
-\frac{\eta^2}{4}s_1\left[(a + c)s_2 + (b + c)s_3\right] = O(\eta^2).
\]

De même, le terme dominant de \(\Delta A_3\) s'annule lorsque \(b(a+c)s_2 = c(a+b)s_1\), sans imposer \(\Delta A_3 = 0\) à l'ordre \(O(\eta^2)\).

### 7.2 Intersection dégénérée \(a + b = a + c = 0\)

Lorsque \(b = c = -a\) (avec \(a \neq 0\)), les deux coefficients linéaires en \(\eta\) s'annulent simultanément **et** l'arête commune devient identiquement nulle à tous les ordres :

\[
\Delta J_{12}\big|_{b=c=-a} \equiv 0 \quad (\forall \eta),
\]

tandis que les arêtes périphériques basculent en régime quadratique pur non nul :

\[
\Delta J_{13}\big|_{b=c=-a} = -\frac{\eta^2}{2}a\,s_2 s_3,
\qquad
\Delta J_{23}\big|_{b=c=-a} = +\frac{\eta^2}{2}a\,s_1 s_3.
\]

### 7.3 Annulation exacte à tous les ordres

L'annulation exacte \(\Delta J = \mathbf{0}_{3\times 3}\) pour tout \(\eta\) et tout \(X_0\) a lieu sur la fibre nulle \(a = b = c = 0\), ou lorsque \(a+c=0\) et \((a+b)s_1 + (b+c)s_3 = 0\) (resp. \(a+b=0\) et \((a+c)s_2 + (b+c)s_3 = 0\)) pour une configuration de spins donnée.

---

## 8. Critères de validation numérique

Le module `src/fiber_bundle.py` et la suite `tests/test_curvature_N3.py` valident l'ensemble des critères suivants :

1. Les huit configurations de spins \(X_0 \in \{-1, +1\}^3\).
2. Plusieurs régimes de couplages initiaux (positifs, négatifs, mixtes, symétriques).
3. Des pas \(\eta\) décroissants (\(\eta \in \{0.5, 0.1, 0.01, 10^{-3}, 10^{-4}\}\)).
4. Les configurations génériques, les annulations du terme dominant (\(a+b=0\), \(a+c=0\)), l'intersection \(a+b=a+c=0\) et la fibre nulle.
5. La symétrie de la matrice \(J\) et la nullité de sa diagonale après chaque transport.
6. La coïncidence des états de spins finaux (\(X^{(A)} = X^{(B)}\)) et l'égalité exacte entre l'écart des fibres finales \(J^{(A)} - J^{(B)}\) et le commutateur analytique \(\Delta J\), composante par composante.
7. La conservation stricte des coordonnées non incidentes à chaque mutation élémentaire.
8. La concordance à précision machine entre résultats numériques et expressions analytiques exactes de \(J^{(A)}\), \(J^{(B)}\) et \(\Delta J\).
9. La convergence des rapports d'ordres dominants :
   \[
   \frac{\Delta J_{12}}{\eta^2} \longrightarrow \frac{1}{4}s_3\left[(a+c)s_2 - (a+b)s_1\right],
   \qquad
   \frac{\Delta J_{13}}{\eta} \longrightarrow (a+c)s_2,
   \qquad
   \frac{\Delta J_{23}}{\eta} \longrightarrow -(a+b)s_1.
   \]
10. La concordance entre calcul direct de \(\Delta A_3\), factorisation exacte et développement au premier ordre, ainsi que l'annulation identique \(\Delta J \equiv \mathbf{0}\) du contrôle intégrable \(\Psi^{\mathrm{int}}_{ij} = s_i s_j\).

---

## 9. Limites et statut épistémique

Les résultats établissent et valident numériquement un défaut de commutation algébrique pour la règle de transport discrète définie ici en dimension \(N = 3\).

Ils ne démontrent pas encore :

- une courbure géométrique au sens d'une connexion différentielle continue ;
- une instabilité dynamique ou un régime de croissance transitoire non-normale ;
- une mémoire topologique persistante sous dynamique stochastique (Gillespie) ;
- une généralisation à \(N > 3\).

---

## 10. Validation effectuée et étape suivante

Les deux modules suivants sont implémentés et validés (15 tests unitaires couvrant \(8 \times 8 \times 4 = 256\) cas sur la grille complète, erreur maximale \(\le 2.22 \times 10^{-16}\)) :

- `src/fiber_bundle.py`
- `tests/test_curvature_N3.py`

**Étape suivante :** verrouiller la définition matricielle explicite de l'opérateur linéarisé gelé \(M_{\mathrm{gelé}}(r, \nu)\) sous Convention B afin de finaliser et exécuter `src/experiment_02_audit.py` (spectres, \(\varepsilon\)-pseudospectres et diagramme des seuils \(r_c \in \{1, 2, 4\}\)).
