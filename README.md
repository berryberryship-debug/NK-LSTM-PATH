NK-LSTM-PATH

Reconstruction de l'information historique dans les paysages adaptatifs de Kauffman

Modélisation multi-échelle des paysages NK, des trajectoires adaptatives et des représentations temporelles par réseaux LSTM.

Auteur : Xioranne Lessard
Projet : Axiomes Base Labs
Version : 0.1 — Protocole expérimental initial
Statut : Recherche exploratoire — hypothèses à tester.

---

1. Présentation

NK-LSTM-PATH est un projet de recherche computationnelle visant à étudier la structure informationnelle des trajectoires évolutives dans les paysages NK développés par Stuart Kauffman.

Le modèle NK décrit des systèmes combinatoires dans lesquels la contribution de chaque composant dépend de son propre état et de ses interactions avec d'autres composants.

La complexité du paysage dépend notamment de deux paramètres :

- N : nombre de composants du système.
- K : nombre d'interactions entre chaque composant et les autres composants.

L'augmentation de K modifie la structure du paysage de fitness et peut produire une fragmentation accrue des optima locaux.

Notre recherche introduit une autre dimension : la mémoire historique des trajectoires.

Nous cherchons à déterminer si l'évolution d'une configuration contient une information prédictive qui ne peut être reconstruite à partir de son seul état instantané.

2. Question fondamentale

Une configuration présente suffit-elle à caractériser les possibilités de son évolution future ?

Nous formulons l'hypothèse que l'histoire d'une trajectoire peut contenir une information supplémentaire sur la dynamique du système.

Soit :

[
x_t = \text{configuration présente}
]

[
F(x_t) = \text{fitness présente}
]

[
H_t=(x_0,x_1,\ldots,x_t)
]

où (H_t) représente l'histoire de la trajectoire.

Nous cherchons à examiner :

[
I(H_t;F_{t+1}\mid x_t,F_t)
]

Cette quantité représente l'information mutuelle conditionnelle entre l'histoire et la fitness future, compte tenu de l'état et de la fitness présents.

Une valeur positive indiquerait une information historique prédictive supplémentaire dans le protocole considéré.

Elle ne démontrerait pas, à elle seule, l'existence d'un invariant universel.

3. Architecture conceptuelle

Le projet repose sur quatre niveaux expérimentaux.

A — Génération du paysage NK

Construction de paysages combinatoires contrôlés.

[
F:X\rightarrow\mathbb{R}
]

Chaque configuration possède une valeur de fitness calculée à partir des contributions locales et de leurs interactions.

B — Génération des trajectoires

Exploration des paysages par mutations élémentaires.

[
x_0\rightarrow x_1\rightarrow\cdots\rightarrow x_t
]

Plusieurs trajectoires seront générées à partir de configurations initiales communes afin d'étudier les divergences, convergences et accessibilités.

C — Reconstruction temporelle par LSTM

Utilisation de réseaux Long Short-Term Memory pour apprendre les dépendances temporelles présentes dans les séquences.

[
h_t=\operatorname{LSTM}(z_t,h_{t-1})
]

où :

- (z_t) représente les observations disponibles à l'instant t ;
- (h_t) représente l'état latent mémoriel ;
- (h_{t-1}) représente la mémoire précédente.

L'objectif est d'étudier quelles propriétés historiques sont encodées dans les représentations latentes.

D — Recherche d'invariants

Comparaison des représentations et des propriétés dynamiques entre différents paysages et différentes valeurs de K.

Nous chercherons notamment :

- des structures récurrentes entre trajectoires ;
- des relations entre accessibilité et fitness ;
- des propriétés latentes reproductibles ;
- des régularités indépendantes de la réalisation particulière du paysage.

4. Modèles expérimentaux comparés

Quatre architectures seront confrontées.

Modèle| Entrées| Fonction
M1| État présent| Référence instantanée
M2| État + fitness| Référence enrichie
M3| Fenêtre historique| Mémoire explicite non récurrente
M4| Séquence historique| Réseau LSTM

Cette comparaison permettra de distinguer l'apport de l'information historique de celui de la seule architecture récurrente.

Les modèles devront utiliser des informations d'entrée comparables et respecter les mêmes partitions expérimentales.

5. Protocole initial

Paramètres envisagés pour la première expérience :

Paramètre| Valeur
N| 20
K| 0, 2, 4, 8
Paysages d'entraînement| 1 000
Paysages de validation| 200
Paysages de test| 200
Longueur maximale d'une trajectoire| 100
Mutation élémentaire| Un bit

Ces valeurs constituent une configuration initiale proposée. Elles ne représentent pas des résultats obtenus.

Les paysages de test seront indépendants des paysages d'entraînement.

Les graines aléatoires seront enregistrées afin de garantir la reproductibilité.

6. Hypothèses de recherche

H1 — Information historique

L'histoire d'une trajectoire peut contenir une information prédictive supplémentaire par rapport à l'état instantané.

H2 — Dépendance à la complexité épistatique

La quantité d'information historique utile pourrait varier en fonction de K.

H3 — Organisation latente

Les représentations internes du LSTM pourraient révéler des propriétés communes entre différentes trajectoires.

H4 — Invariance transversale

Certaines propriétés structurelles pourraient être conservées entre des paysages distincts, malgré leurs différences combinatoires.

Ces quatre hypothèses seront évaluées séparément.

7. Contrôles et falsification

La validité expérimentale repose sur des contrôles explicites.

1. Mélange de l'ordre temporel des séquences.
2. Réinitialisation de la mémoire récurrente.
3. Comparaison avec un modèle non récurrent.
4. Contrôle des fuites d'information entre entraînement et test.
5. Comparaison entre paysages connus et inconnus.
6. Vérification de la stabilité des résultats sur plusieurs graines.
7. Analyse des performances selon K.

Contrôle fondamental :

Dans un paysage NK statique, avec une dynamique de transition markovienne entièrement déterminée par l'état présent, l'histoire ne devrait pas apporter d'information prédictive supplémentaire lorsque l'état est parfaitement observé.

Ce cas constitue un contrôle négatif essentiel.

Un gain du LSTM devra donc être interprété en fonction des informations effectivement disponibles, de la politique de transition et de l'observabilité du paysage.

8. Rapport conceptuel avec Feynman

L'intégrale de chemin de Feynman constitue une référence conceptuelle pour interroger la relation entre état, trajectoire et résultat.

[
\mathcal A=\sum_{\text{paths}}e^{iS/\hbar}
]

Dans le formalisme quantique, les chemins contribuent à une amplitude complexe et peuvent interférer.

Dans NK-LSTM-PATH, les trajectoires sont des séquences combinatoires classiques.

Nous ne postulons aucune équivalence entre les deux formalismes.

Le rapprochement porte uniquement sur une question structurelle :

Quelle information appartient au chemin lui-même, au-delà de la description de ses états ?

9. Organisation du dépôt

NK-LSTM-PATH/
│
├── README.md
├── config.py
├── nk_landscape.py
├── trajectories.py
├── datasets.py
│
├── models/
│   ├── baseline_state.py
│   ├── baseline_history.py
│   └── lstm_model.py
│
├── experiments/
│   ├── experiment_01.py
│   └── controls.py
│
├── metrics.py
├── run_experiment.py
│
└── tests/
    ├── test_nk.py
    ├── test_trajectories.py
    └── test_leakage.py

10. Critères de validation

Le projet ne considérera pas une amélioration prédictive comme une découverte structurelle suffisante.

Une propriété candidate devra :

- être mesurable ;
- être reproductible ;
- résister aux contrôles ;
- généraliser à des paysages indépendants ;
- être distinguable d'un simple artefact d'apprentissage ;
- posséder des critères explicites de réfutation.

La stabilisation d'une représentation latente ne constitue pas, en elle-même, la démonstration d'un invariant du système.

11. Finalité scientifique

NK-LSTM-PATH cherche à explorer une question située à l'intersection de la théorie des systèmes complexes, de l'évolution, de l'apprentissage séquentiel et de la géométrie des espaces de possibilités.

Le déplacement recherché est le suivant :

[
\text{Configuration}
\rightarrow
\text{Trajectoire}
\rightarrow
\text{Mémoire}
\rightarrow
\text{Structure latente}
\rightarrow
\text{Invariant potentiel}
]

L'objectif n'est pas seulement de prédire le prochain état.

Il est de déterminer si la structure historique des transformations permet de révéler une organisation du paysage qui demeure invisible dans une description strictement instantanée.

---

NK-LSTM-PATH — Explorer la géométrie des possibles à travers la mémoire des trajectoires.
