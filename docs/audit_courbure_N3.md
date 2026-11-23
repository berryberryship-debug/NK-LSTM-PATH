Audit algébrique de la courbure discrète — N = 3

Projet : NK-LSTM-PATH
Version : 0.1
Statut : Dérivation analytique — validation numérique à effectuer
Périmètre : Face élémentaire de mutations (1, 2), transport symétrique, convention stricte post-flip.

---

1. Objet de l'audit

Cet audit examine la dépendance à l'ordre de deux mutations successives dans une fibre de couplages symétriques.

L'objectif est de calculer :

[
\Delta J=J^{(A)}-J^{(B)}
]

pour les deux chemins :

[
A: 1\rightarrow2,
\qquad
B: 2\rightarrow1.
]

On cherche également à déterminer les conséquences de cette différence sur le critère d'accessibilité :

[
A_k(J)=\sum_{j\neq k}J_{kj}^{2}.
]

Le présent document établit des résultats algébriques pour (N=3). Il ne constitue pas encore une validation numérique ni une démonstration de mémoire topologique persistante.

2. Définitions et conventions

L'état initial est :

[
X_0=(s_1,s_2,s_3),
\qquad s_i\in{-1,+1}.
]

La fibre symétrique possède trois coordonnées indépendantes :

[
\mathbf j=(J_{12},J_{13},J_{23}).
]

On pose :

[
a=J_{12}^{(0)},\qquad
b=J_{13}^{(0)},\qquad
c=J_{23}^{(0)}.
]

La règle de transport est :

[
\Psi_{ij}(X,J)

\frac12
\sum_{r\notin{i,j}}
(J_{jr}+J_{ir})s_r.
]

Pour (N=3), chaque arête incidente à une mutation possède un seul terme dans cette somme.

Convention post-flip : le spin muté est inversé avant l'évaluation de (\Psi). Les autres spins conservent leur valeur.

La mise à jour est :

[
J_{ij}'=J_{ij}+\eta\Psi_{ij}.
]

Les arêtes non incidentes au nœud muté restent inchangées.

On définit le défaut de commutation par :

[
\Delta J_{ij}=J_{ij}^{(A)}-J_{ij}^{(B)}.
]

3. Chemin A : mutation 1 puis mutation 2

Première mutation : nœud 1

Le spin devient (-s_1).

Les mises à jour sont :

[
J_{12}^{(1,A)}

a+\frac{\eta}{2}(b+c)s_3,
]

[
J_{13}^{(1,A)}

b+\frac{\eta}{2}(a+c)s_2,
]

[
J_{23}^{(1,A)}=c.
]

Deuxième mutation : nœud 2

Le spin (s_2) devient (-s_2). Le spin du nœud 1 est déjà égal à (-s_1).

On obtient :

[
J_{12}^{(A)}

J_{12}^{(1,A)}
+\frac{\eta}{2}
(J_{13}^{(1,A)}+c)s_3,
]

[
J_{23}^{(A)}

c-\frac{\eta}{2}
(J_{13}^{(1,A)}+J_{12}^{(1,A)})s_1,
]

[
J_{13}^{(A)}=J_{13}^{(1,A)}.
]

Après développement :

[
\boxed{
J_{12}^{(A)}

a+\eta(b+c)s_3
+\frac{\eta^2}{4}(a+c)s_2s_3
}
]

[
\boxed{
J_{13}^{(A)}

b+\frac{\eta}{2}(a+c)s_2
}
]

[
\boxed{
J_{23}^{(A)}

c-\frac{\eta}{2}(a+b)s_1
-\frac{\eta^2}{4}s_1
\left[(a+c)s_2+(b+c)s_3\right]
}
]

4. Chemin B : mutation 2 puis mutation 1

Première mutation : nœud 2

Le spin devient (-s_2).

Les mises à jour sont :

[
J_{12}^{(1,B)}

a+\frac{\eta}{2}(b+c)s_3,
]

[
J_{23}^{(1,B)}

c+\frac{\eta}{2}(a+b)s_1,
]

[
J_{13}^{(1,B)}=b.
]

Deuxième mutation : nœud 1

Le spin (s_1) devient (-s_1). Le spin du nœud 2 est déjà égal à (-s_2).

On obtient :

[
J_{12}^{(B)}

J_{12}^{(1,B)}
+\frac{\eta}{2}
(J_{23}^{(1,B)}+b)s_3,
]

[
J_{13}^{(B)}

b-\frac{\eta}{2}
(J_{23}^{(1,B)}+J_{12}^{(1,B)})s_2,
]

[
J_{23}^{(B)}=J_{23}^{(1,B)}.
]

Après développement :

[
\boxed{
J_{12}^{(B)}

a+\eta(b+c)s_3
+\frac{\eta^2}{4}(a+b)s_1s_3
}
]

[
\boxed{
J_{13}^{(B)}

b-\frac{\eta}{2}(a+c)s_2
-\frac{\eta^2}{4}s_2
\left[(a+b)s_1+(b+c)s_3\right]
}
]

[
\boxed{
J_{23}^{(B)}

c+\frac{\eta}{2}(a+b)s_1
}
]

5. Défaut de commutation

En soustrayant les résultats des deux chemins :

Arête commune (1, 2)

[
\boxed{
\Delta J_{12}

\frac{\eta^2}{4}s_3
\left[(a+c)s_2-(a+b)s_1\right]
}
]

Arête périphérique (1, 3)

[
\boxed{
\Delta J_{13}

\eta(a+c)s_2
+
\frac{\eta^2}{4}s_2
\left[(a+b)s_1+(b+c)s_3\right]
}
]

Arête périphérique (2, 3)

[
\boxed{
\Delta J_{23}

-\eta(a+b)s_1
-\frac{\eta^2}{4}s_1
\left[(a+c)s_2+(b+c)s_3\right]
}
]

Les ordres dominants sont donc :

[
\Delta J_{12}=O(\eta^2),
]

[
\Delta J_{13}=O(\eta),
\qquad
\Delta J_{23}=O(\eta).
]

La non-commutativité est donc génériquement présente dès le premier ordre en (\eta) dans les coordonnées périphériques.

L'arête commune présente un défaut d'ordre deux.

Cette distinction est propre à la règle de transport discrète et à la convention post-flip adoptées ici.

6. Conséquence sur l'accessibilité du nœud 3

Le critère d'accessibilité est :

[
A_3(J)=J_{13}^2+J_{23}^2.
]

La différence entre les deux chemins s'écrit :

[
\Delta A_3

(J_{13}^{(A)})^2-(J_{13}^{(B)})^2
+
(J_{23}^{(A)})^2-(J_{23}^{(B)})^2.
]

Au premier ordre :

[
\Delta A_3

2b\Delta J_{13}
+
2c\Delta J_{23}
+O(\eta^2).
]

Ainsi :

[
\boxed{
\Delta A_3

2\eta
\left[
b(a+c)s_2-c(a+b)s_1
\right]
+O(\eta^2)
}
]

Cette expression peut être non nulle dès le premier ordre.

Le critère d'accessibilité est :

[
a_3(J)=\mathbf 1[A_3(J)>\theta_A].
]

Une différence de (A_3) ne suffit toutefois pas à démontrer un changement d'accessibilité : il faut vérifier que les deux valeurs se trouvent effectivement de part et d'autre de (\theta_A).

7. Conditions de dégénérescence

Le défaut de commutation peut s'annuler pour certaines configurations ou certains couplages.

À l'ordre dominant :

[
\Delta J_{13}=0
\quad\Longleftrightarrow\quad
(a+c)s_2=0,
]

[
\Delta J_{23}=0
\quad\Longleftrightarrow\quad
(a+b)s_1=0.
]

Comme les spins valent toujours (+1) ou (-1), ces conditions correspondent respectivement à :

[
a+c=0,
\qquad
a+b=0.
]

Le terme dominant de l'écart d'accessibilité s'annule lorsque :

[
b(a+c)s_2=c(a+b)s_1.
]

Ces configurations doivent être conservées dans les tests. Elles constituent des cas de contrôle importants.

8. Critères de validation numérique

Le fichier "fiber_bundle.py" devra reproduire les expressions précédentes sans approximation supplémentaire.

Les tests devront couvrir :

1. Les huit configurations de spins.
2. Plusieurs valeurs positives et négatives des couplages initiaux.
3. Des valeurs de (\eta) décroissantes.
4. Les configurations génériques et les cas de dégénérescence.
5. La symétrie de la matrice après chaque transport.
6. L'égalité des états finaux après les deux chemins.
7. La conservation des coordonnées non incidentes à chaque mutation.
8. La concordance des résultats numériques et analytiques pour les trois composantes de (\Delta J).
9. La vérification des ordres dominants en (\eta).
10. La concordance entre calcul direct et développement analytique de (\Delta A_3).

Les rapports suivants devront notamment converger vers des valeurs finies lorsque les coefficients dominants sont non nuls :

[
\frac{\Delta J_{12}}{\eta^2},
\qquad
\frac{\Delta J_{13}}{\eta},
\qquad
\frac{\Delta J_{23}}{\eta}.
]

9. Limites et statut épistémique

Les résultats établissent un défaut de commutation algébrique pour la règle de transport définie ici.

Ils ne démontrent pas encore :

- une courbure géométrique au sens d'une connexion différentielle continue ;
- une instabilité dynamique ;
- une mémoire topologique persistante ;
- une irréductibilité opérationnelle ;
- une généralisation à (N>3).

Le terme « courbure discrète » est utilisé ici pour désigner le défaut de commutation des transports autour de la face élémentaire. Une identification à une notion géométrique plus générale nécessitera une construction supplémentaire.

10. Étape suivante

Construire :

- "src/fiber_bundle.py"
- "tests/test_curvature_N3.py"

Le premier module implémentera les transports et les compositions de chemins. Le second comparera systématiquement les calculs numériques aux expressions analytiques de référence établies dans ce document.

Critère de sortie : validation des huit configurations de spins et reproduction des trois composantes du défaut de commutation, y compris leurs ordres dominants respectifs en (\eta).

---

Référence de travail : audit analytique interne du modèle NK-LSTM-PATH, version 0.1. Les résultats numériques restent à produire.
