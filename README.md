# NK-LSTM-PATH
## Reconstruction de l'information historique et géométrie endogène dans les paysages adaptatifs de Kauffman

## Modélisation multi-échelle des paysages NK, des trajectoires adaptatives, des structures de fibrés discrets et des représentations temporelles par réseaux LSTM.

Auteur : Xioranne Lessard

Projet : Axiomes Base Labs

Version : 0.4 — Audit algébrique exact 
N
=
3
N=3, audits numériques de non-commutativité 
N
=
6
N=6, balayage de l'intensité de transport, audit de reproductibilité multi-graines et validation logicielle du noyau computationnel.

Statut : Recherche exploratoire. L'audit algébrique à 
N
=
3
N=3, le balayage numérique à 
N
=
6
N=6, l'audit multi-graines et la suite de huit tests de trajectories.py ont été exécutés. La validation de la dynamique jointe avec dissipation, de la persistance temporelle de l'accessibilité et de l'inférence LSTM demeure ouverte.

1. Présentation
NK-LSTM-PATH est un projet de recherche computationnelle consacré à l'étude de l'information historique portée par les trajectoires évolutives dans les paysages adaptatifs de Kauffman.

Le modèle NK classique décrit un paysage de fitness dans lequel la contribution de chaque composant dépend de son état et de ses interactions avec 
K
K autres composants. Une dynamique évolutive construite sur ce paysage peut être markovienne lorsque l'état du système contient toutes les variables nécessaires à la détermination de son évolution future.

NK-LSTM-PATH introduit une structure supplémentaire : une fibre adaptative, représentée par une matrice d'interactions 
J
J, transportée et transformée au cours des mutations.

Le modèle distingue ainsi deux niveaux :

une configuration combinatoire 
X
X, qui décrit l'état présent des composants ;

une structure d'interactions 
J
J, qui dépend des mises à jour successives et peut donc porter une information sur l'histoire du système.

La question expérimentale consiste à déterminer dans quelle mesure deux trajectoires aboutissant à une même configuration 
X
X peuvent conserver des structures de couplage différentes et produire des observables de transition distinctes.

Le projet examine trois propriétés :

Non-commutativité du transport : deux mutations appliquées dans des ordres différents peuvent-elles produire des fibres finales distinctes ?

Mémoire métrique : la différence entre deux fibres peut-elle persister temporairement avant de s'atténuer sous l'effet de la dissipation ?

Mémoire d'accessibilité : les différences entre les fibres peuvent-elles modifier les transitions accessibles depuis une configuration donnée ?

Il faut distinguer la mémoire portée par une variable d'état de la non-markovianité du système. Si l'état complet est 
(
X
,
J
)
(X,J), l'information historique encodée dans 
J
J peut être intégrée à un processus markovien sur cet espace augmenté. La mémoire est alors portée par la structure présente de la fibre, même si elle provient de l'histoire antérieure.

Les résultats numériques présentés dans cette version établissent une non-commutativité mesurée dans le protocole discret étudié. Ils ne démontrent pas encore une mémoire persistante après dissipation ni une propriété cognitive du réseau LSTM.

2. Question fondamentale et observables
Une configuration présente 
X
X suffit-elle à caractériser les possibilités d'évolution future, ou faut-il également connaître la structure d'interactions 
J
J construite au cours de l'histoire ?

Nous distinguons deux trajectoires, 
A
A et 
B
B, qui partent d'un même état initial et subissent les mêmes mutations dans des ordres différents.

2.1 Mémoire métrique
La distance entre les deux fibres est mesurée par :

D
J
(
t
)
=
∥
J
A
(
t
)
−
J
B
(
t
)
∥
F
D 
J
​
 (t)=∥J 
A
​
 (t)−J 
B
​
 (t)∥ 
F
​
 
où 
∥
⋅
∥
F
∥⋅∥ 
F
​
  désigne la norme de Frobenius.

Cette observable mesure la divergence géométrique entre deux structures d'interactions issues de trajectoires distinctes.

2.2 Accessibilité
L'accessibilité d'un nœud 
k
k est définie par :

A
k
(
J
)
=
∑
j
≠
k
J
k
j
2
A 
k
​
 (J)= 
j

=k
∑
​
 J 
kj
2
​
 
Pour un seuil 
θ
A
θ 
A
​
 , l'ensemble des transitions accessibles est :

T
(
X
,
J
)
=
{
k
∈
{
1
,
…
,
N
}
∣
A
k
(
J
)
>
θ
A
}
T(X,J)={k∈{1,…,N}∣A 
k
​
 (J)>θ 
A
​
 }
Dans la règle actuelle, l'accessibilité dépend de 
J
J, et non directement de 
X
X. L'argument 
X
X est conservé pour situer l'observable dans l'état complet du système.

La différence d'accessibilité entre deux trajectoires est :

D
T
(
t
)
=
card
⁡
(
T
(
X
,
J
A
(
t
)
)
△
T
(
X
,
J
B
(
t
)
)
)
D 
T
​
 (t)=card(T(X,J 
A
​
 (t))△T(X,J 
B
​
 (t)))
où 
△
△ représente la différence symétrique entre ensembles.

Une valeur non nulle signifie que les deux trajectoires ne possèdent pas exactement le même ensemble de nœuds accessibles au temps considéré.

2.3 Marge au seuil et limite mathématique
La marge au seuil est définie pour chaque trajectoire :

m
k
,
A
(
t
)
=
∣
A
k
(
J
A
(
t
)
)
−
θ
A
∣
m 
k,A
​
 (t)=∣A 
k
​
 (J 
A
​
 (t))−θ 
A
​
 ∣
m
k
,
B
(
t
)
=
∣
A
k
(
J
B
(
t
)
)
−
θ
A
∣
m 
k,B
​
 (t)=∣A 
k
​
 (J 
B
​
 (t))−θ 
A
​
 ∣
La marge permet de distinguer une différence d'accessibilité robuste d'une différence produite par la proximité immédiate du seuil.

Une contrainte mathématique doit toutefois être prise en compte dans l'interprétation de la mémoire d'accessibilité.

Pour chaque site 
k
k, en notant 
r
k
A
r 
k
A
​
  et 
r
k
B
r 
k
B
​
  les lignes correspondantes des deux matrices :

∣
A
k
(
J
A
)
−
A
k
(
J
B
)
∣
=
∣
∥
r
k
A
∥
2
−
∥
r
k
B
∥
2
∣
≤
(
∥
r
k
A
∥
+
∥
r
k
B
∥
)
∥
r
k
A
−
r
k
B
∥
≤
(
∥
J
A
∥
F
+
∥
J
B
∥
F
)
D
J
∣A 
k
​
 (J 
A
​
 )−A 
k
​
 (J 
B
​
 )∣
​
  
= 
​
 ∥r 
k
A
​
 ∥ 
2
 −∥r 
k
B
​
 ∥ 
2
  
​
 
≤(∥r 
k
A
​
 ∥+∥r 
k
B
​
 ∥)∥r 
k
A
​
 −r 
k
B
​
 ∥
≤(∥J 
A
​
 ∥ 
F
​
 +∥J 
B
​
 ∥ 
F
​
 )D 
J
​
 
​
 
Si les deux masques diffèrent au site 
k
k, le seuil 
θ
A
θ 
A
​
  se trouve entre les deux valeurs d'accessibilité. Par conséquent :

min
⁡
(
m
k
,
A
,
m
k
,
B
)
≤
(
∥
J
A
∥
F
+
∥
J
B
∥
F
)
D
J
min(m 
k,A
​
 ,m 
k,B
​
 )≤(∥J 
A
​
 ∥ 
F
​
 +∥J 
B
​
 ∥ 
F
​
 )D 
J
​
 
Cette relation impose une limite à l'hypothèse de persistance initialement envisagée.

Si les normes des fibres restent bornées et que :

D
J
(
t
)
⟶
0
D 
J
​
 (t)⟶0
alors toute divergence persistante des masques exige qu'au moins une des deux trajectoires se rapproche du seuil. Une différence d'accessibilité ne peut donc pas persister asymptotiquement avec une marge uniformément positive sur les deux trajectoires.

Conséquence pour le protocole :

Une divergence de 
D
T
D 
T
​
  pendant que 
D
J
D 
J
​
  reste non nul peut être observée et mesurée.

Une divergence de masque lorsque les marges sont très faibles doit être interprétée comme une sensibilité au seuil.

Une persistance de l'accessibilité après convergence de 
J
A
J 
A
​
  et 
J
B
J 
B
​
 , avec une marge uniformément positive, nécessiterait une règle d'accessibilité dépendante de l'histoire ou une variable supplémentaire, par exemple un mécanisme d'hystérésis.

Cette distinction constitue une contrainte de falsification pour les extensions du modèle.

3. Architecture conceptuelle et dynamique jointe
Le modèle s'organise autour d'une base combinatoire et d'une fibre adaptative.

A — Base combinatoire : paysage natif
L'espace des configurations est l'hypercube :

X
=
(
s
1
,
…
,
s
N
)
∈
{
−
1
,
+
1
}
N
X=(s 
1
​
 ,…,s 
N
​
 )∈{−1,+1} 
N
 
En bijection avec 
{
0
,
1
}
N
{0,1} 
N
  :

x
i
=
s
i
+
1
2
x 
i
​
 = 
2
s 
i
​
 +1
​
 
La fitness native 
F
N
K
(
X
)
F 
NK
​
 (X) est déterminée par le paysage NK.

Chaque composant possède une contribution locale dépendant de son état et des 
K
K autres composants associés à son voisinage épistatique.

B — Fibre adaptative et connexion discrète
Au-dessus de chaque configuration, la fibre est représentée par :

J
∈
S
y
m
N
(
R
)
J∈Sym 
N
​
 (R)
avec :

J
=
J
T
,
J
i
i
=
0
J=J 
T
 ,J 
ii
​
 =0
La matrice 
J
J représente la structure des interactions adaptatives entre les composants.

Convention stricte post-flip
Lorsqu'un nœud 
k
k mute, son spin bascule d'abord :

s
k
′
=
−
s
k
s 
k
′
​
 =−s 
k
​
 
Le transport est ensuite évalué sur l'état post-flip 
X
′
X 
′
 .

Pour 
i
≠
j
i

=j, la fonction de transport est :

Ψ
i
j
(
X
′
,
J
)
=
1
2
∑
r
∉
{
i
,
j
}
(
J
j
r
+
J
i
r
)
s
r
′
Ψ 
ij
​
 (X 
′
 ,J)= 
2
1
​
  
r∈
/
{i,j}
∑
​
 (J 
jr
​
 +J 
ir
​
 )s 
r
′
​
 
Seules les arêtes incidentes au nœud muté 
k
k sont mises à jour :

J
k
j
′
=
J
k
j
+
η
Ψ
k
j
(
X
′
,
J
)
J 
kj
′
​
 =J 
kj
​
 +ηΨ 
kj
​
 (X 
′
 ,J)
La symétrie est conservée par :

J
j
k
′
=
J
k
j
′
J 
jk
′
​
 =J 
kj
′
​
 
et la diagonale demeure nulle.

Le paramètre 
η
η contrôle l'intensité du transport.

Comparaison des ordres de mutation
Pour deux nœuds distincts 
k
k et 
m
m, les deux compositions sont :

U
m
∘
U
k
et
U
k
∘
U
m
U 
m
​
 ∘U 
k
​
 etU 
k
​
 ∘U 
m
​
 
Les flips sur la base combinatoire commutent :

X
(
k
)
(
m
)
=
X
(
m
)
(
k
)
(
k
≠
m
)
X 
(k)(m)
 =X 
(m)(k)
 (k

=m)
La fibre, en revanche, peut dépendre de l'ordre d'application des transports.

On définit l'observable de non-commutativité :

C
k
m
(
X
,
J
)
=
∥
J
k
→
m
−
J
m
→
k
∥
F
C 
km
​
 (X,J)=∥J 
k→m
​
 −J 
m→k
​
 ∥ 
F
​
 
où les deux matrices finales sont obtenues en appliquant les mutations et les transports dans les ordres indiqués.

Une valeur strictement positive caractérise une dépendance à l'ordre dans la règle de transport étudiée.

Dans ce projet, cette différence est interprétée opérationnellement comme une courbure discrète ou une holonomie de face. Cette terminologie désigne une propriété des compositions de transports. Elle ne constitue pas, à elle seule, une démonstration de toutes les propriétés d'une connexion géométrique continue.

C — Dynamique jointe et sélection
La dynamique complète envisagée associe trois mécanismes :

mutation de la base ;

transport instantané de la fibre lors d'une mutation acceptée ;

dissipation continue entre les mutations.

1. Fitness étendue
La fitness du système est définie par :

F
(
X
,
J
)
=
F
N
K
(
X
)
+
λ
∑
i
<
j
J
i
j
s
i
s
j
F(X,J)=F 
NK
​
 (X)+λ 
i<j
∑
​
 J 
ij
​
 s 
i
​
 s 
j
​
 
Pour une matrice symétrique à diagonale nulle :

F
(
X
,
J
)
=
F
N
K
(
X
)
+
λ
2
X
T
J
X
F(X,J)=F 
NK
​
 (X)+ 
2
λ
​
 X 
T
 JX
Le paramètre 
λ
λ pondère la rétroaction de la fibre sur la sélection.

Le cas :

λ
=
0
λ=0
isole le régime dans lequel la fibre n'intervient pas dans la fitness, tout en pouvant continuer à évoluer par transport.

2. Mutation et taux de transition
Le taux de bascule du nœud 
k
k est :

w
k
(
X
,
J
)
=
μ
a
k
(
J
)
min
⁡
(
1
,
e
β
Δ
F
k
(
X
,
J
)
)
w 
k
​
 (X,J)=μa 
k
​
 (J)min(1,e 
βΔF 
k
​
 (X,J)
 )
avec :

Δ
F
k
(
X
,
J
)
=
F
(
X
(
k
)
,
J
)
−
F
(
X
,
J
)
ΔF 
k
​
 (X,J)=F(X 
(k)
 ,J)−F(X,J)
et :

a
k
(
J
)
=
1
[
A
k
(
J
)
>
θ
A
]
a 
k
​
 (J)=1[A 
k
​
 (J)>θ 
A
​
 ]
où :

μ
μ est l'échelle des taux ;

β
β contrôle la sensibilité à la variation de fitness ;

θ
A
θ 
A
​
  est le seuil d'accessibilité ;

λ
λ pondère le couplage dans la fitness.

Dans cette règle, le taux est évalué sur l'état courant 
(
X
,
J
)
(X,J). Lorsqu'une mutation est acceptée, le spin bascule, puis le transport post-flip met à jour la fibre.

La transition complète est donc une mise à jour de l'état augmenté :

(
X
,
J
)
⟶
(
X
(
k
)
,
U
k
(
J
)
)
(X,J)⟶(X 
(k)
 ,U 
k
​
 (J))
3. Dissipation continue
Entre deux mutations, la fibre est conçue pour relaxer exponentiellement vers une matrice de référence 
J
(
0
)
J 
(0)
  :

d
J
d
t
=
−
γ
(
J
−
J
(
0
)
)
dt
dJ
​
 =−γ(J−J 
(0)
 )
La solution entre deux événements est :

J
(
t
)
=
J
(
0
)
+
(
J
(
t
0
)
−
J
(
0
)
)
e
−
γ
(
t
−
t
0
)
J(t)=J 
(0)
 +(J(t 
0
​
 )−J 
(0)
 )e 
−γ(t−t 
0
​
 )
 
où 
γ
γ est le coefficient de dissipation.

Cette évolution préserve la symétrie et la diagonale nulle lorsque 
J
(
t
0
)
J(t 
0
​
 ) et 
J
(
0
)
J 
(0)
  respectent ces contraintes.

4. Méthode d'intégration stochastique
La simulation par amincissement d'Ogata (thinning) est implémentée dans le moteur de trajectoires. Un simulateur analytique par morceaux, simulate_exact_piecewise_lambda0, est également prévu comme contrôle lorsque 
λ
=
0
λ=0.

Sous les conditions :

μ
≥
0
,
0
≤
a
k
(
J
)
≤
1
μ≥0,0≤a 
k
​
 (J)≤1
et :

0
≤
min
⁡
(
1
,
e
β
Δ
F
k
)
≤
1
0≤min(1,e 
βΔF 
k
​
 
 )≤1
on obtient :

0
≤
w
k
(
X
,
J
)
≤
μ
0≤w 
k
​
 (X,J)≤μ
et donc :

∑
k
=
1
N
w
k
(
X
,
J
)
≤
N
μ
k=1
∑
N
​
 w 
k
​
 (X,J)≤Nμ
Cette borne fournit une intensité supérieure pour le processus de proposition utilisé par l'amincissement.

La suite de tests de trajectories.py vérifie notamment la borne d'intensité, la dissipation analytique, le transport post-flip, la conservation du risque intégré et la comparaison en loi des deux simulateurs à 
λ
=
0
λ=0. Les huit tests de cette suite ont été exécutés avec succès dans l'environnement documenté à la fin du présent README.

D — Reconstruction temporelle par LSTM
Les réseaux Long Short-Term Memory sont envisagés comme observateurs temporels de la fibre adaptative.

L'entrée principale est la séquence des configurations :

X
t
0
,
X
t
1
,
…
,
X
t
n
X 
t 
0
​
 
​
 ,X 
t 
1
​
 
​
 ,…,X 
t 
n
​
 
​
 
L'objectif est de reconstruire une estimation de la fibre latente :

J
^
t
=
LSTM
⁡
(
X
t
0
,
…
,
X
t
)
J
  
t
​
 =LSTM(X 
t 
0
​
 
​
 ,…,X 
t
​
 )
Le projet examine si la séquence historique permet d'estimer une structure d'interactions qui n'est pas déterminée par la seule configuration présente 
X
t
X 
t
​
 .

Cette capacité doit être évaluée par comparaison avec des modèles de référence, sur des trajectoires et des paysages de test séparés de l'entraînement.

Aucune supériorité du LSTM sur les modèles de référence n'est revendiquée à ce stade.

4. Modèles expérimentaux comparés
Quatre architectures d'inférence sont prévues.

Modèle	Entrée	Fonction
M1 — État présent	
X
t
X 
t
​
 	Référence instantanée
M2 — État et fitness	
X
t
,
F
t
X 
t
​
 ,F 
t
​
 	Référence enrichie
M3 — Fenêtre historique	Fenêtre finie de 
X
X	Mémoire explicite non récurrente
M4 — Séquence historique	Séquence de 
X
X	Inférence récurrente par LSTM
Plan factoriel 
2
×
2
2×2 de la dynamique
Le plan expérimental prévu croise la règle de transport et la rétroaction de la fibre sur la fitness.

Règle de transport	Avec rétroaction 
(
λ
≠
0
)
(λ

=0)	Sans rétroaction 
(
λ
=
0
)
(λ=0)
Transport symétrique non commutatif	Modèle complet	Transport symétrique pur
Transport intégrable commutatif	Rétroaction sans non-commutativité du transport	Ligne de base
Ce plan doit permettre de distinguer les effets associés à la non-commutativité de ceux associés à la rétroaction sélective.

Son exécution complète et l'évaluation statistique des quatre conditions restent à réaliser.

5. Protocoles expérimentaux et résultats
Les validations sont organisées selon deux échelles principales :

N
=
3
N=3 : audit algébrique exact sur une face élémentaire ;

N
=
6
N=6 : comparaison numérique des ordres de mutation, balayage de 
η
η et audits de reproductibilité multi-graines.

Les simulations dynamiques de plus grande échelle, notamment 
N
=
8
N=8 et 
N
=
20
N=20, ne sont pas encore validées.

5.1 — Audit algébrique exact sur une face élémentaire, 
N
=
3
N=3
On considère :

X
0
=
(
s
1
,
s
2
,
s
3
)
∈
{
−
1
,
+
1
}
3
X 
0
​
 =(s 
1
​
 ,s 
2
​
 ,s 
3
​
 )∈{−1,+1} 
3
 
et la matrice initiale :

J
(
0
)
=
(
0
a
b
a
0
c
b
c
0
)
J 
(0)
 = 
​
  
0
a
b
​
  
a
0
c
​
  
b
c
0
​
  
​
 
avec :

a
=
J
12
(
0
)
,
b
=
J
13
(
0
)
,
c
=
J
23
(
0
)
a=J 
12
(0)
​
 ,b=J 
13
(0)
​
 ,c=J 
23
(0)
​
 
L'accessibilité du troisième nœud est :

A
3
(
J
)
=
J
13
2
+
J
23
2
A 
3
​
 (J)=J 
13
2
​
 +J 
23
2
​
 
Les deux chemins comparés sont :

A
:
1
→
2
et
B
:
2
→
1
A:1→2etB:2→1
La convention stricte post-flip donne les ordres :

J
A
=
U
2
(
U
1
(
J
(
0
)
)
)
J 
A
​
 =U 
2
​
 (U 
1
​
 (J 
(0)
 ))
J
B
=
U
1
(
U
2
(
J
(
0
)
)
)
J 
B
​
 =U 
1
​
 (U 
2
​
 (J 
(0)
 ))
En posant :

Δ
J
=
J
A
−
J
B
ΔJ=J 
A
​
 −J 
B
​
 
l'audit algébrique exact établit les ordres suivants :

Δ
J
12
=
O
(
η
2
)
ΔJ 
12
​
 =O(η 
2
 )
Δ
J
13
=
O
(
η
)
ΔJ 
13
​
 =O(η)
Δ
J
23
=
O
(
η
)
ΔJ 
23
​
 =O(η)
L'écart d'accessibilité s'écrit exactement :

Δ
A
3
=
(
J
13
(
A
)
)
2
−
(
J
13
(
B
)
)
2
+
(
J
23
(
A
)
)
2
−
(
J
23
(
B
)
)
2
ΔA 
3
​
 
​
  
=(J 
13
(A)
​
 ) 
2
 −(J 
13
(B)
​
 ) 
2
 
+(J 
23
(A)
​
 ) 
2
 −(J 
23
(B)
​
 ) 
2
 
​
 
soit :

Δ
A
3
=
Δ
J
13
(
J
13
(
A
)
+
J
13
(
B
)
)
+
Δ
J
23
(
J
23
(
A
)
+
J
23
(
B
)
)
ΔA 
3
​
 
​
  
=ΔJ 
13
​
 (J 
13
(A)
​
 +J 
13
(B)
​
 )
+ΔJ 
23
​
 (J 
23
(A)
​
 +J 
23
(B)
​
 )
​
 
Au premier ordre en 
η
η :

Δ
A
3
=
2
η
[
b
(
a
+
c
)
s
2
−
c
(
a
+
b
)
s
1
]
+
O
(
η
2
)
ΔA 
3
​
 =2η[b(a+c)s 
2
​
 −c(a+b)s 
1
​
 ]+O(η 
2
 )
​
 
Cet audit établit une expression analytique de l'écart d'accessibilité sur la face élémentaire étudiée.

Il ne valide pas, à lui seul, la dynamique jointe avec dissipation ni la persistance temporelle de l'accessibilité.

5.2 — Audit numérique de non-commutativité et balayage de 
η
η, 
N
=
6
N=6
Le protocole compare les deux ordres d'application de deux mutations distinctes :

k
→
m
et
m
→
k
k→metm→k
Paramètres de référence
Paramètre	Valeur
Nombre de sites 
N
N	6
Nombre de configurations initiales	10
Paires distinctes par configuration	15
Comparaisons par valeur de 
η
η	150
Échelle initiale de 
J
J	0.2
Graine initiale	7
Graine du paysage NK	7
Dépendances NK 
K
K	3
μ
μ	1.0
β
β	1.0
θ
A
θ 
A
​
 	0.1
λ
λ	0.5
Le même paysage NK et les mêmes configurations initiales sont conservés pour les comparaisons du balayage. Les six valeurs de 
η
η sont :

η
∈
{
0
,
  
0.05
,
  
0.1
,
  
0.25
,
  
0.5
,
  
1.0
}
η∈{0,0.05,0.1,0.25,0.5,1.0}
Chaque ligne correspond à 150 comparaisons.

Résultats du balayage
Les valeurs ci-dessous proviennent de l'exécution de audit_robustesse_eta.py du 25 novembre 2026.

η
η	Comparaisons	Gap moyen 
J
J	Gap max 
J
J	
Δ
A
ΔA moyen	
Δ
A
ΔA max	
Δ
w
Δw moyen	
Δ
w
Δw max	
Δ
w
≠
0
Δw

=0	Désaccords
0.00	150	0.00000	0.00000	0.00000	0.00000	0.00000	0.00000	0.0 %	0
0.05	150	0.037462	0.084396	0.008897	0.034977	0.11302	1.2393	68.0 %	20
0.10	150	0.075777	0.174236	0.018423	0.077010	0.20564	1.2900	70.0 %	34
0.25	150	0.197552	0.478854	0.053370	0.264180	0.48461	1.7072	82.7 %	91
0.50	150	0.431533	1.118733	0.148510	0.901980	0.83408	2.0004	93.3 %	178
1.00	150	1.060533	3.053254	0.615260	4.422100	0.97509	2.1480	96.7 %	213
Les erreurs maximales de symétrie et de diagonale sont nulles pour toutes les lignes. Les états finaux 
X
X sont identiques dans 100 % des comparaisons.

Définition exacte des indicateurs
Écart géométrique

L'écart moyen et l'écart maximal sont calculés sur les 150 comparaisons de matrices finales :

D
J
(
q
)
=
∥
J
A
(
q
)
−
J
B
(
q
)
∥
F
D 
J
(q)
​
 =∥J 
A
(q)
​
 −J 
B
(q)
​
 ∥ 
F
​
 
Différence d'accessibilité

Pour chaque comparaison :

D
T
(
q
)
=
card
⁡
(
T
(
J
A
(
q
)
)
△
T
(
J
B
(
q
)
)
)
D 
T
(q)
​
 =card(T(J 
A
(q)
​
 )△T(J 
B
(q)
​
 ))
Le compteur agrégé est :

C
T
=
∑
q
=
1
150
D
T
(
q
)
C 
T
​
 = 
q=1
∑
150
​
 D 
T
(q)
​
 
Le pourcentage de désaccords site par site est calculé sur les 
150
×
6
=
900
150×6=900 comparaisons possibles :

C
T
(
%
)
=
100
 
C
T
150
×
6
C 
T
(%)
​
 =100 
150×6
C 
T
​
 
​
 
Les compteurs observés correspondent respectivement à 20, 34, 91, 178 et 213 désaccords pour les valeurs positives de 
η
η.

Différence des taux

Une comparaison est comptée lorsque :

max
⁡
k
∣
w
A
,
k
−
w
B
,
k
∣
>
10
−
12
k
max
​
 ∣w 
A,k
​
 −w 
B,k
​
 ∣>10 
−12
 
La colonne 
Δ
w
≠
0
Δw

=0 représente le pourcentage de comparaisons satisfaisant ce critère. Il ne s'agit pas du pourcentage de composantes individuelles de taux différentes.

Correction terminologique

Le compteur précédemment désigné comme « franchissements de seuil » mesure en réalité les désaccords entre les masques d'accessibilité finaux des deux chemins. Il ne mesure pas des franchissements temporels du seuil le long d'une trajectoire.

Observations établies
Contrôle nul

À 
η
=
0
η=0, les deux ordres conduisent à la même fibre :

∥
J
A
−
J
B
∥
F
=
0
∥J 
A
​
 −J 
B
​
 ∥ 
F
​
 =0
Aucun désaccord de masque n'est observé.

Augmentation de l'écart géométrique

L'écart moyen et l'écart maximal augmentent pour chacune des valeurs de 
η
η testées.

L'écart moyen passe de :

0.037462
(
η
=
0.05
)
0.037462(η=0.05)
à :

1.060533
(
η
=
1.0
)
1.060533(η=1.0)
Cette observation indique une augmentation de la non-commutativité mesurée avec l'intensité du transport, dans le domaine testé. Elle ne démontre pas une loi générale de croissance ni une relation analytique universelle avec 
η
η.

Conservation de la base

La fraction d'états finaux 
X
X identiques est de 100 % pour toutes les valeurs de 
η
η.

Cela est cohérent avec la commutation de deux flips distincts sur la base combinatoire :

X
(
k
)
(
m
)
=
X
(
m
)
(
k
)
X 
(k)(m)
 =X 
(m)(k)
 
La divergence mesurée se situe donc dans la fibre 
J
J, et non dans l'état final des spins pour ce protocole.

Conservation des contraintes structurelles

Les erreurs maximales de symétrie et de diagonale sont nulles pour toutes les valeurs de 
η
η :

J
=
J
T
J=J 
T
 
J
i
i
=
0
J 
ii
​
 =0
Les contraintes structurelles testées sont donc respectées dans les 150 comparaisons de chaque ligne.

Transmission vers les observables de transition

La fraction de comparaisons présentant une différence non nulle entre les vecteurs de taux augmente globalement avec 
η
η, de 68.0 % à 
η
=
0.05
η=0.05 à 96.7 % à 
η
=
1.0
η=1.0.

La différence entre les fibres se répercute donc sur les taux calculés.

Cette variation peut provenir des différences d'accessibilité et des termes de fitness. Le protocole présenté ne sépare pas encore quantitativement ces contributions.

5.3 — Audit de reproductibilité multi-graines, 
N
=
6
N=6
Un deuxième audit a été exécuté afin de séparer l'effet des conditions initiales de celui du paysage NK.

Chaque exécution utilise :

N
=
6
N=6 ;

10 configurations initiales par graine ;

15 paires distinctes par configuration ;

150 comparaisons par ligne ;

η
=
0.25
η=0.25 ;

μ
=
1.0
μ=1.0 ;

β
=
1.0
β=1.0 ;

θ
A
=
0.1
θ 
A
​
 =0.1 ;

λ
=
0.5
λ=0.5 ;

K
=
3
K=3 ;

échelle initiale de 
J
=
0.2
J=0.2.

Les graines comparées sont :

{
7
,
13
,
21
,
42
,
99
}
{7,13,21,42,99}
L'audit comprend deux blocs indépendants.

Bloc A — Variation de la graine initiale
Le paysage NK est maintenu fixe :

n
k
_
s
e
e
d
=
7
nk_seed=7
La graine initiale des configurations 
X
0
,
J
0
X 
0
​
 ,J 
0
​
  varie.

Graine initiale	Graine NK	Écart moyen 
J
J	Écart maximal 
J
J	Err. sym.	Err. diag.	
X
X identique	
Δ
w
≠
0
Δw

=0	Désaccords 
D
T
D 
T
​
 
7	7	0.197552	0.478854	0	0	100.0 %	82.7 %	91 (10.1 %)
13	7	0.205107	0.489502	0	0	100.0 %	86.7 %	77 (8.6 %)
21	7	0.188030	0.426054	0	0	100.0 %	79.3 %	90 (10.0 %)
42	7	0.198132	0.460827	0	0	100.0 %	85.3 %	90 (10.0 %)
99	7	0.217580	0.443931	0	0	100.0 %	94.0 %	100 (11.1 %)
Synthèse sur les cinq graines :

Indicateur	Moyenne ± écart-type
Écart moyen 
J
J	
0.201280
±
0.009795
0.201280±0.009795
Comparaisons avec 
Δ
w
≠
0
Δw

=0	
85.6
%
±
4.9
%
85.6%±4.9%
Désaccords 
D
T
D 
T
​
 	
89.6
±
7.3
89.6±7.3
Ces résultats montrent une variabilité entre les configurations initiales dans le protocole testé.

Les contraintes suivantes restent satisfaites :

X
A
=
X
B
X 
A
​
 =X 
B
​
  dans 100 % des comparaisons ;

erreur de symétrie nulle ;

erreur diagonale nulle.

Bloc B — Variation de la graine du paysage NK
La configuration initiale est maintenue fixe :

i
n
i
t
_
s
e
e
d
=
7
init_seed=7
La graine du paysage NK varie.

Graine initiale	Graine NK	Écart moyen 
J
J	Écart maximal 
J
J	Err. sym.	Err. diag.	
X
X identique	
Δ
w
≠
0
Δw

=0	Désaccords 
D
T
D 
T
​
 
7	7	0.197552	0.478854	0	0	100.0 %	82.7 %	91 (10.1 %)
7	13	0.197552	0.478854	0	0	100.0 %	81.3 %	91 (10.1 %)
7	21	0.197552	0.478854	0	0	100.0 %	81.3 %	91 (10.1 %)
7	42	0.197552	0.478854	0	0	100.0 %	80.0 %	91 (10.1 %)
7	99	0.197552	0.478854	0	0	100.0 %	81.3 %	91 (10.1 %)
Synthèse sur les cinq paysages NK :

Indicateur	Moyenne ± écart-type
Écart moyen 
J
J	
0.197552
±
0.000000
0.197552±0.000000
Comparaisons avec 
Δ
w
≠
0
Δw

=0	
81.3
%
±
0.8
%
81.3%±0.8%
Désaccords 
D
T
D 
T
​
 	
91.0
±
0.0
91.0±0.0
Les indicateurs géométriques sont invariants entre les cinq réalisations NK.

Cette invariance est attendue de l'architecture actuelle : la fonction de transport mutate_postflip ne reçoit pas le paysage NK comme argument. Lorsque 
X
0
,
J
0
,
η
X 
0
​
 ,J 
0
​
 ,η et les paires de mutations sont identiques, changer le paysage NK ne modifie donc pas les fibres finales calculées.

Le résultat constitue un contrôle de cohérence et de séparation des modules du code. Il ne constitue pas une découverte indépendante sur les paysages NK.

Les taux de transition, en revanche, varient selon le paysage utilisé pour calculer la fitness. Le bloc B confirme ainsi une séparation entre :

la géométrie du transport, déterminée ici par les états initiaux et 
η
η ;

les taux calculés, qui dépendent également de la réalisation du paysage NK.

Synthèse statistique et limites
Les écarts-types indiqués sont des écarts-types descriptifs calculés sur les cinq résultats par graine, avec la convention de population (np.std, ddof=0).

Ils ne sont ni des intervalles de confiance ni des tests de significativité. Les cinq graines constituent un audit exploratoire de reproductibilité.

Mise à jour de la valeur de référence à 
η
=
0.25
η=0.25
Un ancien résultat de 80.0 % avait été rapporté pour la fraction de comparaisons avec 
Δ
w
≠
0
Δw

=0 à 
η
=
0.25
η=0.25.

Le nouvel audit de robustesse donne :

82.7
%
82.7%
Ce résultat concorde avec le bloc A de l'audit multi-graines pour init_seed=7 et nk_seed=7.

La valeur de 82.7 % est donc retenue comme résultat de référence dans cette version du README. L'ancienne valeur de 80.0 % n'est pas conservée comme résultat concurrent.

L'origine exacte de l'écart entre les anciennes exécutions n'est pas établie ici. Cette correction ne prétend donc pas identifier rétrospectivement la cause technique de la différence.

5.4 — Protocole de seuil 
θ
A
θ 
A
​
 
Le seuil 
θ
A
θ 
A
​
  est traité comme un paramètre explicite du modèle.

Un premier balayage exploratoire a été effectué sur 20 000 tirages :

J
∼
U
(
[
−
1
,
1
]
3
)
J∼U([−1,1] 
3
 )
pour la représentation à trois sites par les trois coefficients indépendants d'une matrice symétrique à diagonale nulle.

Cette première exploration utilisait une grille grossière et ne comportait pas de barres d'erreur.

Elle suggérait une dépendance des résultats de franchissement du seuil à l'intensité du transport sur une seule étape.

Ce résultat reste exploratoire. Il devra être reproduit avec :

une grille de seuils publiée intégralement ;

une grille raffinée ;

plusieurs graines aléatoires ;

des intervalles de confiance ;

une distinction explicite entre les franchissements au cours d'une trajectoire et les désaccords entre deux masques finaux.

Aucun seuil de référence ne doit être calibré a posteriori sur un maximum empirique sans justification indépendante.

6. Hypothèses de recherche
Les hypothèses sont formulées comme des propositions falsifiables. Leur état est distingué des observations numériques déjà obtenues.

H1 — Non-commutativité et géométrie discrète
Le transport symétrique de la fibre produit une non-commutativité sur certaines faces élémentaires de l'hypercube.

La structure spectrale linéarisée, notée 
M
g
e
l
e
ˊ
M 
gel 
e
ˊ
 
​
 , pourrait également dépendre de la classe de configuration.

État : non-commutativité mesurée dans les audits 
N
=
3
N=3 et 
N
=
6
N=6. La dépendance spectrale, la relation avec la parité et la portée générale de l'hypothèse restent à tester.

H2 — Mémoire métrique et accessibilité
La dissipation peut réduire la distance entre deux fibres issues de chemins distincts.

La règle d'accessibilité actuelle étant une fonction déterministe et seuillée de 
J
J, une divergence de masque ne peut cependant pas persister asymptotiquement lorsque 
D
J
→
0
D 
J
​
 →0 si les deux fibres restent bornées et si les marges au seuil sont uniformément positives.

L'hypothèse doit donc être séparée en deux propositions :

H2a — Accessibilité transitoire : une différence d'accessibilité peut persister temporairement lorsque les deux fibres sont distinctes, notamment lorsque certains scores 
A
k
(
J
)
A 
k
​
 (J) sont proches du seuil.

H2b — Accessibilité avec mémoire propre : une différence d'accessibilité peut subsister après convergence géométrique uniquement si une variable dépendante de l'histoire, une règle d'hystérésis ou un état interne supplémentaire est introduit.

État : non testé par les audits de non-commutativité présentés ici. La dynamique avec dissipation et le suivi temporel de 
D
J
(
t
)
D 
J
​
 (t), 
D
T
(
t
)
D 
T
​
 (t) et des marges restent à réaliser.

H3 — Organisation latente et reconstruction LSTM
Le LSTM (M4) pourrait reconstruire la fibre latente 
J
t
J 
t
​
  et prédire les changements d'accessibilité à partir de la séquence des configurations.

La comparaison doit être réalisée contre M1, M2 et M3, sur des trajectoires indépendantes de celles utilisées pour l'entraînement.

État : à tester. Aucune performance d'inférence ni supériorité du LSTM n'est revendiquée à ce stade.

H4 — Invariance transversale au paysage NK
Les propriétés géométriques du transport pourraient rester invariantes entre différentes réalisations du paysage natif 
F
N
K
F 
NK
​
 , lorsque les conditions initiales et les règles de transport sont identiques.

État : invariance observée dans l'audit statique multi-graines 
N
=
6
N=6, conformément à la séparation actuelle entre mutate_postflip et le calcul de la fitness NK. La généralisation aux propriétés spectrales et à la dynamique jointe reste à tester.

7. Contrôles et falsification
La validation du modèle repose sur cinq contrôles.

1. Contrôle de transport intégrable
Un contrôle candidat remplace la règle de transport symétrique par :

Ψ
k
j
i
n
t
(
X
′
)
=
s
k
′
s
j
′
Ψ 
kj
int
​
 (X 
′
 )=s 
k
′
​
 s 
j
′
​
 
où les spins sont évalués sur l'état post-flip.

La mise à jour correspondante est :

J
k
j
′
=
J
k
j
+
η
Ψ
k
j
i
n
t
(
X
′
)
J 
kj
′
​
 =J 
kj
​
 +ηΨ 
kj
int
​
 (X 
′
 )
avec symétrisation de l'arête.

Le contrôle vise à construire un transport dont les compositions associées à deux flips distincts commutent :

U
m
i
n
t
∘
U
k
i
n
t
=
U
k
i
n
t
∘
U
m
i
n
t
U 
m
int
​
 ∘U 
k
int
​
 =U 
k
int
​
 ∘U 
m
int
​
 
L'écart final attendu est :

∥
J
k
→
m
i
n
t
−
J
m
→
k
i
n
t
∥
F
=
0
​
 J 
k→m
int
​
 −J 
m→k
int
​
  
​
  
F
​
 =0
Cette égalité ne signifie pas que chaque mise à jour individuelle de 
J
J est nulle. Elle signifie que le résultat final ne dépend pas de l'ordre des deux transports.

L'intégrabilité de ce contrôle doit être vérifiée par des tests unitaires avant son utilisation comme référence expérimentale.

2. Mélange de l'ordre temporel
Les séquences historiques sont permutées afin de déterminer si les modèles d'inférence exploitent l'ordre temporel des observations plutôt qu'une simple distribution des états.

La comparaison doit conserver les mêmes ensembles d'entraînement et de test.

3. Contrôle de parité
Le transport symétrique est examiné sous la transformation :

X
→
−
X
X→−X
Le protocole compare les observables, les spectres et les commutateurs entre les deux configurations, en explicitant la transformation appliquée à 
J
J et les conditions initiales.

4. Contrôle de seuil
Le seuil dur est comparé à une sigmoïde lissée :

a
k
(
δ
)
(
J
)
=
1
1
+
exp
⁡
[
−
(
A
k
(
J
)
−
θ
A
)
/
δ
]
a 
k
(δ)
​
 (J)= 
1+exp[−(A 
k
​
 (J)−θ 
A
​
 )/δ]
1
​
 
Le paramètre 
δ
δ contrôle la largeur de la zone de transition.

L'analyse doit publier les résultats en fonction de 
θ
A
θ 
A
​
  et de 
δ
δ, avec intervalles d'incertitude et plusieurs graines.

5. Contrôle des fuites d'information
Pour l'inférence de 
J
t
J 
t
​
  :

les séquences de test doivent être indépendantes des séquences d'entraînement ;

les paysages NK de test doivent être séparés de ceux utilisés pour l'entraînement lorsque la généralisation aux paysages est étudiée ;

les fenêtres provenant d'une même trajectoire ne doivent pas être réparties aléatoirement entre entraînement et test ;

J
t
J 
t
​
 , lorsqu'il constitue la cible supervisée, ne doit pas être fourni comme variable d'entrée au modèle.

Les performances doivent être rapportées séparément pour la reconstruction de 
J
t
J 
t
​
 , la prédiction des taux et la prédiction des masques d'accessibilité.

8. Rapport conceptuel avec l'intégrale de chemin
Le projet établit un rapprochement structurel avec les formulations par chemins : une évolution peut être étudiée à partir de l'ensemble des trajectoires admissibles et des poids associés.

Une écriture formelle possible est :

Z
c
h
e
m
i
n
s
=
∑
chemins
e
−
S
e
f
f
(
chemin
)
Z 
chemins
​
 = 
chemins
∑
​
 e 
−S 
eff
​
 (chemin)
 
où 
S
e
f
f
S 
eff
​
  représenterait une action effective dépendant de la structure du réseau et de l'historique des transports sur la fibre.

Cette écriture est utilisée comme analogie de somme de poids sur des trajectoires. L'action 
S
e
f
f
S 
eff
​
  n'est pas encore dérivée à partir du processus stochastique défini dans ce projet.

Elle ne constitue pas une amplitude de Feynman en temps réel et ne postule aucune équivalence physique avec la mécanique quantique.

Le rapprochement est strictement structurel : le chemin parcouru peut modifier la fibre et, par conséquent, les possibilités ultérieures du système.

9. Organisation du dépôt
L'arborescence ci-dessous reprend les chemins de travail établis pour les modules principaux et les fichiers de validation.

text
NK-LSTM-PATH/
│
├── README.md
├── balayage_commutateur_nk_v3.py
├── audit_robustesse_eta.py
│
├── src/
│   ├── __init__.py
│   ├── fiber_bundle.py
│   ├── nk_landscape.py
│   └── trajectories.py
│
└── test/
    ├── __init__.py
    └── test_trajectories.py
Cette arborescence décrit les fichiers dont l'emplacement a été établi dans l'environnement de travail. Elle ne prétend pas représenter tous les fichiers éventuels du dépôt.

Fichiers dont l'emplacement reste à confirmer
Le nom audit_reproductibilite_graines.py est identifié dans les travaux précédents, mais son emplacement exact dans le dépôt n'est pas établi ici.

Il n'est donc pas ajouté artificiellement à l'arborescence.

Aucun dossier docs/, models/ ou experiments/ n'est présumé exister. L'arborescence sera complétée uniquement à partir de l'inventaire réel du dépôt.

10. État des validations
Élément	État
Audit algébrique exact sur la face 
N
=
3
N=3	Vérifié
Audit numérique de non-commutativité 
N
=
6
N=6	Exécuté
Balayage de six intensités 
η
η	Exécuté
Conservation de la symétrie de 
J
J dans l'audit 
N
=
6
N=6	Vérifiée
Conservation de la diagonale nulle de 
J
J	Vérifiée
Conservation de 
X
X entre les deux ordres de flips	Vérifiée
Désaccords entre masques finaux 
D
T
D 
T
​
 	Mesurés
Définition du compteur précédemment appelé « franchissements »	Clarifiée : désaccords de masques
Audit multi-graines sur les conditions initiales	Exécuté
Audit multi-graines sur les paysages NK	Exécuté
Invariance géométrique sous variation de nk_seed	Observée et conforme à l'architecture du code
Valeur de référence de 
Δ
w
≠
0
Δw

=0 à 
η
=
0.25
η=0.25	82.7 %, confirmée par le nouvel audit et le bloc multi-graines
Ancienne valeur de 80.0 %	Remplacée ; origine de l'écart historique non établie
Appariement des états initiaux entre les valeurs de 
η
η	Vérifié dans l'audit : mêmes configurations initiales, même paysage NK et mêmes conditions de comparaison pour chaque valeur de 
η
η.
Suite de huit tests dans test/test_trajectories.py	Exécutée : 8/8 réussis
Compatibilité des constructions NKLandscape dans le fichier de tests	Harmonisée dans la version exécutée
Compilation des modules principaux et du script d'audit	Vérifiée sans erreur
Importation conjointe des scripts d'audit et de balayage	Vérifiée
Test d'intégration des balayages à 
η
=
0
η=0 et 
η
=
0.2
η=0.2	Réussi
Contrôle de transport intégrable	À implémenter et tester
Dynamique jointe avec dissipation	Validation complète à poursuivre
Validation du simulateur d'Ogata	Tests correspondants réussis dans la suite exécutée
Validation du simulateur exact par morceaux à 
λ
=
0
λ=0	Tests correspondants réussis dans la suite exécutée
Conservation du risque intégré à travers un changement de masque	Test correspondant réussi dans la suite exécutée
Persistance temporelle de 
D
J
(
t
)
D 
J
​
 (t)	À tester
Persistance temporelle de 
D
T
(
t
)
D 
T
​
 (t)	À tester
Robustesse de l'accessibilité hors proximité du seuil	À tester
Comparaison M1–M4	À exécuter
Généralisation à 
N
=
8
N=8 et 
N
=
20
N=20	À exécuter
Précision sur la suite de tests
Le fichier test/test_trajectories.py contient huit méthodes de test couvrant notamment :

la borne supérieure des taux et l'invariance de 
Δ
F
k
ΔF 
k
​
  à 
λ
=
0
λ=0 ;

la dissipation analytique et sa propriété de semi-groupe ;

le transport post-flip et la dissipation des arêtes non incidentes ;

le bilan détaillé de Metropolis, le silence sous le seuil et les lois élémentaires des événements ;

la stabilité numérique du solveur quadratique ;

la conservation du risque intégré à travers les changements du masque d'accessibilité ;

la validation croisée en loi entre le simulateur d'Ogata et le simulateur exact par morceaux ;

la sélection du canal lorsque le tirage uniforme vaut exactement zéro.

Les huit tests ont été exécutés avec succès :

8
/
8
 
tests r
e
ˊ
ussis
8/8 tests r 
e
ˊ
 ussis
​
 
Cette réussite confirme les propriétés couvertes par la suite dans l'environnement d'exécution utilisé. Elle ne constitue pas une validation de l'ensemble des expériences périphériques ni de la dynamique complète dans toutes ses configurations.

11. Prochaines étapes expérimentales
Les étapes sont ordonnées selon leurs dépendances.

Étape 1 — Verrouillage de la reproductibilité
Pour chaque expérience, enregistrer :

les graines ;

les paramètres complets ;

les versions du code ;

les configurations 
X
0
X 
0
​
  ;

les matrices 
J
0
J 
0
​
  ;

les voisinages et les tables NK ;

les sorties numériques brutes.

L'appariement des réalisations initiales entre les valeurs de 
η
η doit être confirmé directement dans le code.

Étape 2 — Contrôle intégrable
Implémenter le transport candidat, vérifier sa commutation par tests unitaires et comparer ses résultats à ceux du transport symétrique.

Étape 3 — Validation approfondie de la dynamique jointe
Poursuivre la validation du processus comprenant les mutations, le transport post-flip et la relaxation continue.

Contrôler la borne d'intensité utilisée par l'amincissement, la gestion des événements et la conservation des contraintes de la fibre dans des simulations complètes.

Étape 4 — Persistance temporelle de la mémoire métrique et de l'accessibilité
Suivre simultanément :

D
J
(
t
)
,
D
T
(
t
)
,
m
k
,
A
(
t
)
,
m
k
,
B
(
t
)
D 
J
​
 (t),D 
T
​
 (t),m 
k,A
​
 (t),m 
k,B
​
 (t)
Distinguer les divergences transitoires produites par la proximité du seuil d'une éventuelle mémoire d'accessibilité dépendante de l'histoire.

Étape 5 — Reconstruction LSTM
Construire les jeux de données temporels, entraîner M1–M4 et comparer leurs capacités de reconstruction et de prédiction sur des trajectoires et des paysages indépendants.

Étape 6 — Extension de l'échelle
Après validation des contrôles précédents, étendre progressivement les expériences à :

N
=
8
puis
N
=
20
N=8puisN=20
La complexité numérique ne doit pas précéder la validation des invariants, de la reproductibilité et des métriques.

12. Critère de progression scientifique
Chaque extension du modèle doit conserver une distinction explicite entre :

Résultat algébrique : propriété démontrée à partir des règles mathématiques définies.

Observation numérique : résultat obtenu dans un protocole donné, avec paramètres, graines et limites documentés.

Hypothèse interprétative : propriété proposée qui exige une expérience discriminante.

Validation logicielle : conformité de l'implémentation aux spécifications et aux tests.

Une non-commutativité mesurée dans la fibre ne suffit pas à établir une mémoire topologique persistante.

Une différence entre deux masques d'accessibilité ne suffit pas à démontrer une mémoire indépendante du seuil.

Une reconstruction numérique de 
J
t
J 
t
​
  ne suffit pas, à elle seule, à établir une propriété cognitive du réseau LSTM.

Le programme de recherche progresse donc par validations successives :

G
e
ˊ
om
e
ˊ
trie discr
e
ˋ
te
⟶
Dynamique jointe
⟶
Persistance
⟶
Reconstruction temporelle
G 
e
ˊ
 om 
e
ˊ
 trie discr 
e
ˋ
 te⟶Dynamique jointe⟶Persistance⟶Reconstruction temporelle
​
 
L'objectif est de déterminer précisément quelles informations historiques sont conservées dans la structure adaptative, dans quelles conditions elles influencent les possibilités futures et jusqu'à quel point elles peuvent être reconstruites à partir des trajectoires observées.

Note de mise à jour — 25 novembre 2026
Audit de robustesse sur l'intensité de transport
L'exécution complète de audit_robustesse_eta.py a été réalisée avec les paramètres suivants :

Paramètre	Valeur
N
N	6
Configurations initiales	10
Graine initiale	7
Échelle initiale de 
J
J	0.2
Graine NK	7
K
K	3
μ
μ	1.0
β
β	1.0
θ
A
θ 
A
​
 	0.1
λ
λ	0.5
Grille 
η
η	0.0, 0.05, 0.1, 0.25, 0.5, 1.0
Les six valeurs de 
η
η ont été exécutées avec 150 comparaisons chacune.

Le contrôle 
η
=
0
η=0 a confirmé l'identité des fibres finales et l'absence de désaccords d'accessibilité. Pour les valeurs positives de 
η
η, les écarts géométriques, les différences d'accessibilité et les différences entre vecteurs de taux ont été mesurés.

Les résultats détaillés sont consignés dans la section 5.2.

Validation logicielle du noyau computationnel
Les fichiers suivants ont été récupérés depuis le dépôt et compilés sans erreur :

src/trajectories.py

src/fiber_bundle.py

src/nk_landscape.py

test/test_trajectories.py

La suite complète de validation a ensuite été exécutée avec succès :

8
/
8
 
tests r
e
ˊ
ussis
8/8 tests r 
e
ˊ
 ussis
​
 
Durée totale : 533,787 secondes.

Les huit tests couvrent notamment les bornes des taux de transition, la dissipation analytique, le transport post-flip, les propriétés de Metropolis, la stabilité du solveur quadratique, la conservation du risque intégré, la comparaison des simulateurs d'Ogata et exact par morceaux à 
λ
=
0
λ=0, ainsi que la sélection du canal à la frontière 
u
2
=
0
u 
2
​
 =0.

Une comparaison avec les anciens fichiers locaux a également confirmé que les modules fiber_bundle.py et nk_landscape.py sont identiques après normalisation. Les différences relevées dans trajectories.py et test_trajectories.py concernent uniquement la fin de fichier et l'adaptation des chemins d'importation à l'organisation du dépôt.

La compilation des scripts d'audit et de leurs dépendances a également été vérifiée. L'importation conjointe de balayage_commutateur_nk_v3.py et audit_robustesse_eta.py a réussi.

Un test d'intégration supplémentaire a vérifié les balayages à 
η
=
0
η=0 et 
η
=
0.2
η=0.2, avec validation des résumés et recalcul indépendant de la fraction de différences non nulles entre les taux. Le test a réussi dans les deux cas.

Portée de la validation
Ces validations confirment le bon fonctionnement des modules et des propriétés couvertes par les tests dans l'environnement d'exécution utilisé.

Elles ne constituent pas une validation de l'ensemble des expériences périphériques, de la persistance temporelle de la mémoire métrique ou de l'accessibilité, ni des performances de reconstruction LSTM.

Le programme expérimental décrit dans le présent README demeure donc ouvert sur ces questions.

## Repérage des compléments dans le README

Les précisions de cet addendum se rapportent aux passages suivants du document principal. Les renvois permettent de retrouver les résultats, les tableaux et les protocoles auxquels elles apportent un complément.

| Complément | Section d'origine | Élément concerné |
|---|---|---|
| A | 5.2 — Audit numérique de non-commutativité et balayage de η | Tableau des résultats du balayage : définition des écarts ΔA et Δw |
| B | 5.2 — Définition exacte des indicateurs | Compteur `total_threshold_crossings`, désaccords entre masques d'accessibilité et calcul de C_T |
| C | 5.2 — Paramètres de référence et protocole expérimental | Réutilisation des configurations initiales et du paysage NK pendant le balayage de η |
| D | 5.2 — Résultats du balayage ; 5.3 — Audit de reproductibilité multi-graines | Comparaison des fractions de 82,7 % et de 80,0 % |
| E | 5.3 — Bloc B : Variation de la graine du paysage NK | Invariance des écarts géométriques et dépendance des taux à la réalisation NK |
| F | 9 — Organisation des fichiers et arborescence du dépôt | Emplacement de `audit_reproductibilite_graines.py` |
| G | 10 — État de validation, limites et travaux restant à réaliser | Portée exacte des validations et limites des conclusions |

**Principe de lecture :** chaque complément doit être lu conjointement avec sa section d'origine. L'addendum précise ou rectifie l'interprétation de ces passages sans remplacer leur contenu initial.

## Description scientifique du protocole : dépendance au chemin et holonomie discrète dans NK-LSTM-PATH

1. Contexte et formulation de l'espace d'état

Le protocole étudie un système couplé associant une configuration génotypique discrète et une matrice d'interaction structurelle.

La configuration binaire de (N) sites est représentée par :

[
X=(s_1,\ldots,s_N)^\top\in{-1,+1}^N.
]

La structure interactionnelle est représentée par une matrice :

[
J\in\mathbb R^{N\times N},
]

symétrique et à diagonale nulle :

[
J=J^\top,\qquad \operatorname{diag}(J)=0.
]

L'espace des états complets est défini par :

[
\mathcal E=
{-1,+1}^N\times\mathcal S_{\mathrm{sym},0}(N),
]

où :

[
\mathcal S_{\mathrm{sym},0}(N)

\left{
J\in\mathbb R^{N\times N}
\mid
J=J^\top,\
\operatorname{diag}(J)=0
\right}.
]

Chaque état complet associe ainsi une configuration génotypique (X) à une structure interactionnelle (J).

2. Opérateurs de mutation et chemins sur l'hypercube

Soit (\sigma_k) l'opérateur d'inversion élémentaire du site (k) :

[
(\sigma_k(X))_i=
\begin{cases}
-s_k & \text{si }i=k,\
s_i & \text{si }i\neq k.
\end{cases}
]

Pour deux sites distincts (k\neq m), les opérateurs de mutation commutent :

[
\sigma_m\circ\sigma_k

\sigma_k\circ\sigma_m.
]

Cette propriété est exacte sur l'hypercube discret.

À partir d'un état initial ((X_0,J_0)), deux chemins mutationnels sont comparés :

[
\begin{aligned}
\text{Chemin A}:&
\quad
(X_0,J_0)
\xrightarrow{\sigma_k}
(X_1,J_1)
\xrightarrow{\sigma_m}
(X_A,J_A),\
\text{Chemin B}:&
\quad
(X_0,J_0)
\xrightarrow{\sigma_m}
(X'_1,J'_1)
\xrightarrow{\sigma_k}
(X_B,J_B).
\end{aligned}
]

Les deux chemins aboutissent nécessairement au même état génotypique :

[
X_A=X_B=\sigma_m(\sigma_k(X_0)).
]

En revanche, cette égalité ne garantit pas l'identité des matrices structurelles finales.

Le protocole examine précisément cette distinction entre commutation génotypique et dépendance au chemin du transport structurel.

3. Transport de la structure et défaut de fermeture

Chaque mutation entraîne une mise à jour de la matrice d'interaction par l'intermédiaire de l'opérateur "mutate_postflip", paramétré notamment par le coefficient de couplage (\eta).

On représente cette mise à jour par :

[
J'=T_k(X,J;\eta).
]

Les transports composés le long des deux chemins sont :

[
H_A=
T_m(X_1,J_1;\eta)
\circ
T_k(X_0,J_0;\eta),
]

[
H_B=
T_k(X'_1,J'_1;\eta)
\circ
T_m(X_0,J_0;\eta).
]

Le défaut de fermeture du transport est défini par :

[
\mathcal H_{km}(X_0,J_0)

J_A-J_B.
]

Sa mesure scalaire utilise la norme de Frobenius :

[
\Delta_J=
|J_A-J_B|_F

\sqrt{
\sum_{i,j=1}^{N}
(J_{A,ij}-J_{B,ij})^2
}.
]

Une valeur non nulle indique que les deux ordres de transport produisent des structures finales différentes pour l'état initial et la paire de mutations considérés.

Précision terminologique : le script conserve le nom informatique "commutator_fro" pour cette mesure. Ce nom ne doit pas être interprété comme un commutateur matriciel :

[
[J_A,J_B]=J_AJ_B-J_BJ_A.
]

La quantité calculée est bien l'écart de transport entre chemins, et non le commutateur d'algèbre des deux matrices finales.

3.1. Dépendance au chemin et interprétation géométrique

Lorsque :

[
\Delta_J>0,
]

le transport structurel dépend de l'ordre des mutations, bien que le génotype final soit identique.

Cette propriété constitue un défaut de fermeture sur le carré élémentaire de l'hypercube associé aux sites (k) et (m).

Elle peut être interprétée comme une manifestation d'holonomie discrète : le transport de la structure le long de deux chemins reliant les mêmes sommets ne restitue pas nécessairement le même élément de la fibre finale.

Une identification rigoureuse à une courbure géométrique exige toutefois de définir explicitement une connexion discrète sur le graphe de Hamming, ses opérateurs de transport et les propriétés de composition associées.

Le défaut mesuré ici constitue donc un indicateur de dépendance au chemin. Il ne représente pas, à lui seul, un tenseur de courbure différentielle.

4. Les deux règles de transport comparées

Le script compare deux règles de transport appliquées aux mêmes états initiaux et aux mêmes paires de mutations.

4.1. Contrôle intégrable

La règle de contrôle est définie par :

[
\Psi_{ij}(X')=
s'_i s'_j
\qquad (i\neq j),
]

avec :

[
\Psi_{ii}=0.
]

Dans le code, cette règle est implémentée par "psi_integrable(X, J)".

Elle construit une matrice à partir de l'état génotypique considéré, sans utiliser les valeurs de (J) pour construire directement (\Psi).

Comme les deux chemins aboutissent au même génotype, cette règle sert de contrôle de fermeture du transport.

Le résultat attendu est :

[
J_A=J_B,
\qquad
\Delta_J=0.
]

Cette propriété est également vérifiée numériquement dans le script : une erreur est déclenchée si le contrôle intégrable ne satisfait pas l'égalité des états génotypiques et l'égalité des états complets à la tolérance demandée.

4.2. Transport symétrique natif

Le second transport utilise l'opérateur natif :

[
\Psi=\psi_{\mathrm{sym}}(X',J),
]

implémenté dans "fiber_bundle.py".

Il est transmis à "mutate_postflip" comme fonction de transport.

Contrairement au contrôle précédent, les mises à jour successives peuvent conserver une dépendance à la structure intermédiaire et, par conséquent, à l'ordre des mutations.

Pour les configurations et valeurs de (\eta) effectivement testées, les simulations ont mis en évidence des écarts structurels non nuls.

Cette observation établit une dépendance au chemin dans les configurations concernées. Sa généralisation nécessite des balayages sur plusieurs graines, paramètres et configurations initiales.

5. Propagation aux observables fonctionnelles

Le protocole examine si les différences structurelles entre (J_A) et (J_B) se répercutent sur les observables fonctionnelles du système.

5.1. Écart d'accessibilité

L'accessibilité est représentée par :

[
A(J)\in\mathbb R^N.
]

L'écart entre les deux chemins est :

[
\Delta_A=
|A(J_A)-A(J_B)|_2.
]

Le script conserve également les différences par composante :

[
\delta A_i=A_i(J_A)-A_i(J_B).
]

5.2. Écart des taux de transition

Les taux de transition sont représentés par :

[
w(X,J)\in\mathbb R^N.
]

Comme les génotypes finaux sont identiques, la comparaison porte sur :

[
\Delta_w=
|w(X_A,J_A)-w(X_B,J_B)|_2.
]

L'écart maximal par site est :

[
\Delta w_{\max}

\max_{1\leq i\leq N}
|w_{A,i}-w_{B,i}|.
]

Les différences individuelles sont également conservées :

[
\delta w_i=w_{A,i}-w_{B,i}.
]

Une comparaison est considérée comme présentant un écart de taux non nul lorsque :

[
\Delta w_{\max}>\operatorname{tol}_w.
]

5.3. Désaccords entre les masques d'accessibilité

Le masque d'accessibilité est défini par le seuil (\theta_A) :

[
M_i(J)=
\mathbb I[A_i(J)>\theta_A].
]

Le nombre de désaccords entre les deux chemins est :

[
\Delta_{\mathrm{masque}}

\sum_{i=1}^{N}
\mathbb I
\left[
M_i(J_A)\neq M_i(J_B)
\right].
]

Ce compteur mesure le nombre de sites dont le statut d'accessibilité final diffère selon l'ordre des mutations.

Il identifie ainsi une divergence potentielle dans les possibilités de transition ultérieures.

Il ne mesure pas les franchissements temporels du seuil et ne démontre pas, à lui seul, l'ouverture ou la fermeture définitive de voies évolutives.

6. Construction et contrôle du protocole numérique

Le paysage NK canonique et les états initiaux sont générés une seule fois par exécution.

Les deux règles de transport utilisent les mêmes états initiaux, les mêmes matrices (J_0), les mêmes paramètres et les mêmes paires de sites.

Chaque paire non ordonnée est évaluée une seule fois :

[
(k,m),\qquad k<m.
]

Les deux ordres de mutation sont comparés au sein de chaque observation.

Pour (N) sites et (S) états initiaux, le nombre de comparaisons est :

[
N_{\mathrm{comparaisons}}

S\binom{N}{2}.
]

Les diagnostics de symétrie et de diagonale nulle sont calculés sur les matrices brutes avant leur validation structurelle par "_validate_J".

Cette distinction permet de mesurer les éventuels défauts structurels avant toute validation susceptible de rejeter les matrices.

Le protocole contrôle notamment :

- l'identité des génotypes finaux ;
- l'identité exacte ou approchée des états complets ;
- l'écart structurel entre les matrices finales ;
- les erreurs de symétrie ;
- les erreurs de diagonale ;
- les différences d'accessibilité ;
- les différences des taux de transition ;
- les désaccords entre les masques d'accessibilité finaux.

Les résultats détaillés, y compris les différences par site, sont conservés dans les enregistrements JSON.

7. Dépendances internes du code

Le script "balayage_commutateur_nk_v2.py" utilise les modules suivants :

"fiber_bundle.py"

Fournit les fonctions et opérateurs structurels :

- "_validate_J"
- "_validate_state"
- "accessibility_A"
- "mutate_postflip"
- "psi_sym"

"nk_landscape.py"

Fournit la classe :

- "NKLandscape"

Cette classe génère le paysage NK canonique utilisé dans les deux comparaisons.

"trajectories.py"

Fournit la fonction :

- "transition_rates"

Elle calcule les taux de transition à partir de l'état génotypique, de la matrice d'interaction et du paysage NK partagé.

Les imports effectifs du script sont :

from fiber_bundle import (
    _validate_J,
    _validate_state,
    accessibility_A,
    mutate_postflip,
    psi_sym,
)

from nk_landscape import NKLandscape

from trajectories import transition_rates

Cette organisation distingue la définition du paysage NK, les opérateurs de transport structurel et la dynamique des transitions.

8. Synthèse des niveaux conceptuels

Propriété| Formulation| Statut dans le protocole
Commutation génotypique| (\sigma_k\circ\sigma_m=\sigma_m\circ\sigma_k)| Exacte sur l'hypercube
Fermeture du transport| (J_A-J_B=0)| Vérifiée pour le contrôle intégrable
Défaut de fermeture| (\Delta_J=|J_A-J_B|_F)| Mesuré pour les deux transports
Dépendance au chemin| (\Delta_J>0)| Observée pour les configurations testées du transport symétrique
Propagation fonctionnelle| (\Delta_A,\Delta_w,\Delta_{\mathrm{masque}})| Mesurée numériquement
Courbure différentielle| Définition d'une connexion et de sa courbure| Formalisation ultérieure nécessaire

9. Conclusion

Le protocole met en évidence une distinction fondamentale entre la commutation des mutations génotypiques et la commutation des transports structurels.

Deux séquences mutationnelles peuvent conduire exactement au même génotype final tout en produisant des matrices d'interaction différentes.

Cette différence constitue une dépendance au chemin du transport dans l'espace des structures associées au génotype.

Le protocole permet ensuite d'examiner la propagation de cette différence vers l'accessibilité, les taux de transition et les masques fonctionnels du paysage adaptatif.

La relation centrale étudiée est donc :

[
\boxed{
\text{Commutation des mutations}
;\not\Rightarrow;
\text{Commutation des transports}
}
]

L'intérêt du modèle réside dans l'étude de cette dissociation : l'identité de l'état génotypique final n'implique pas nécessairement l'identité de son état structurel associé.

Le défaut de fermeture mesuré constitue ainsi un indicateur numérique de mémoire du chemin dans le transport structurel, dont l'interprétation en termes de courbure géométrique demeure une étape formelle distincte.
