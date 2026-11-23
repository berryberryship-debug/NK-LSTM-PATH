NK-LSTM-PATH

Reconstruction de l'information historique et géométrie endogène dans les paysages adaptatifs de Kauffman

Modélisation multi-échelle des paysages NK, des trajectoires adaptatives, des structures de fibrés discrets et des représentations temporelles par réseaux LSTM.

Auteur : Xioranne Lessard
Projet : Axiomes Base Labs
Version : 0.2 — Formalisation géométrique et protocole expérimental révisé
Statut : Recherche exploratoire — hypothèses à tester.

---

1. Présentation

NK-LSTM-PATH est un projet de recherche computationnelle visant à étudier la structure informationnelle des trajectoires évolutives dans les paysages adaptatifs développés par Stuart Kauffman.

Le modèle NK classique décrit un paysage statique où la contribution de chaque composant dépend de son état et de ses interactions avec K autres composants. Dans ce cadre standard, la dynamique de transition est markovienne : l'état présent détermine entièrement les possibilités futures.

Notre recherche introduit une rupture conceptuelle : la géométrie endogène des possibilités. Nous modélisons l'évolution non plus comme un déplacement sur une surface fixe, mais comme le transport d'une structure d'interactions (la fibre) qui modifie sa propre topologie au fur et à mesure de la trajectoire.

Le projet cherche à déterminer si le chemin parcouru contient une information prédictive irréductible, et si cette mémoire historique se manifeste de manière purement métrique (transitoire) ou topologique (persistante via la modification des transitions accessibles).

2. Question fondamentale

Une configuration présente suffit-elle à caractériser les possibilités de son évolution future ?

Nous distinguons deux formes de mémoire issues de l'histoire d'une trajectoire H_t :

1. La mémoire métrique : la distance entre deux états de la fibre adaptative J issue de deux chemins différents s'estompe sous l'effet de la dissipation.
2. La mémoire topologique : l'écart entre les fibres, même infime, modifie de façon persistante le franchissement des seuils d'accessibilité \mathcal{T}(X,J).

Nous cherchons à mesurer séparément :

D_J(t) = \| J_A(t) - J_B(t) \|_F \quad \text{et} \quad D_{\mathcal{T}}(t) = |\mathcal{T}(X,J_A(t)) \triangle \mathcal{T}(X,J_B(t))| 

La persistance de D_{\mathcal{T}}(t) alors que D_J(t) \to 0 constituerait la preuve d'une irréductibilité topologique du processus.

3. Architecture conceptuelle

Le projet repose sur un formalisme à deux niveaux (Base et Fibre) et quatre composantes dynamiques.

A — Base combinatoire (Paysage natif)
L'hypercube des configurations X \in \{0, 1\}^N. La fitness native F_{\mathrm{NK}}(X) est générée par le paysage NK.

B — Fibre adaptative et Connexion discrète
Au-dessus de chaque configuration, une fibre J \in \mathrm{Sym}_N(\mathbb{R}) représente la structure d'interactions adaptatives. 
Le passage d'un état à un autre (mutation d'un bit k) s'accompagne d'un transport le long d'une connexion discrète U_k :

J \mapsto J + \eta \Psi(X^k, k, J) 

La non-commutativité des transports ([U_k, U_m] \neq 0) définit une courbure discrète (holonomie) sur les faces de l'hypercube.

C — Dynamique jointe et Sélection
L'évolution est régie par un générateur de Markov couplé :

\mathcal{L} = \mathcal{L}_{\mathrm{mut}} + \mathcal{L}_{\mathrm{transp}} + \mathcal{L}_{\mathrm{diss}} 

La sélection stochastique (Boltzmann) dépend d'une fitness étendue F(X,J) qui couple le paysage natif et l'état de la fibre. L'accessibilité d'une mutation est conditionnée par la perméabilité de la fibre : A_k(J) = \sum_{j \neq k} J_{kj}^2 > \theta_A.

D — Reconstruction temporelle par LSTM
Les réseaux Long Short-Term Memory sont utilisés comme observateurs. Ils doivent inférer l'état latent de la fibre J_t (non observable directement) à partir de la seule séquence des états X_t. L'objectif est de tester si l'apprentissage séquentiel capture la mémoire topologique.

4. Modèles expérimentaux comparés

Pour isoler les effets, nous confrontons les architectures d'apprentissage et les contrôles dynamiques.

Pour les LSTM :

· M1 : État présent (Référence instantanée).
· M2 : État + fitness (Référence enrichie).
· M3 : Fenêtre historique (Mémoire explicite non récurrente).
· M4 : Séquence historique (Réseau LSTM).

Pour la dynamique (Design factoriel 2 \times 2) :

 Rétroaction sélective (\partial_J W \neq 0) Sans rétroaction (\partial_J W = 0)
Courbure réelle ([U_k, U_m] \neq 0) Modèle complet Courbure pure
Transport intégrable ([U_k, U_m] = 0) Feedback pur Ligne de base

5. Protocole initial

L'audit algébrique commence sur les petites dimensions avant de monter en échelle. Les paramètres sans dimension sont le taux de mutation normalisé \nu = \mu N \bar{W} / \lambda_0 et le rapport spectral r = \eta\mu / \lambda_0.

Paramètre Valeur
N (composants) 3 (audit), 8, 20
K (interactions) 2, 4, 8
Transitions Seuil dur A_k(J) > \theta_A
Métrique de distribution Distance de Wasserstein-1 (W_1)

Les seuils d'accessibilité \theta_A seront testés en version fixe (calibrée sur J^0) et en version dynamique (médiane de A_k) pour éviter le fine-tuning.

6. Hypothèses de recherche

H1 — Non-commutativité et courbure
Le transport symétrique de la fibre brise la parité du paysage et produit une courbure non nulle sur les faces, dont les seuils d'instabilité locale dépendent de la classe de configuration (spectres à r_c \in \{1, 2, 4\}).

H2 — Mémoire topologique vs métrique
Il existe un régime de paramètres où D_J(t) \to 0 (relaxation) mais où D_{\mathcal{T}}(t) persiste, créant une mémoire discrète de l'histoire dépendante du seuil \theta_A.

H3 — Organisation latente
Le LSTM (M4) parvient à reconstruire la variable latente J_t et à prédire les ruptures d'accessibilité \mathcal{T}, surpassant M1-M3 uniquement dans le régime de courbure active.

H4 — Invariance transversale
Les propriétés spectrales de l'opérateur de relaxation (matrice M_{\mathrm{gelé}}) et les transitions de phase associées sont invariantes entre différentes réalisations du paysage natif F_{\mathrm{NK}}.

7. Contrôles et falsification

La validité expérimentale repose sur des contrôles de falsification explicites :

1. Contrôle de transport intégrable : Remplacement de la règle de transport par une règle commutative (\Psi_{kj} = c s_k s_j). Toute différence observée avec le modèle réel est attribuable à la courbure.
2. Mélange de l'ordre temporel : Test de permutation des séquences.
3. Contrôle de parité : Le transport brise la parité s \to -s. Le modèle doit être testé sous les deux classes pour vérifier l'asymétrie des spectres.
4. Contrôle de seuil : Remplacement du seuil dur par une fonction sigmoïde lissée a_k^{(\delta)} pour vérifier la robustesse.
5. Contrôle des fuites : Isolation stricte des paysages de test (entraînement supervisé sans accès à J_t).

8. Rapport conceptuel avec la géométrie et Feynman

L'intégrale de chemin de Feynman stipule que l'amplitude d'un processus est la somme des contributions de tous les chemins possibles. Dans notre formalisme, la fonction de transition dépend de l'holonomie (la courbure accumulée) du chemin :

\mathcal{A} = \sum_{\text{chemins}} e^{-S(\text{chemin})} 

L'action S dépend de la topologie du réseau et de l'historique des transports. Nous ne postulons aucune équivalence avec la mécanique quantique. Le rapprochement est structurel : le passé n'est pas seulement une suite d'états, c'est un transport de structure.

9. Organisation du dépôt (Mise à jour v0.2)

```text
NK-LSTM-PATH/
│
├── README.md
├── config.py
├── nk_landscape.py       # Paysage natif F_NK
├── fiber_bundle.py       # (Nouveau) Matrices J, règle Psi, calcul des commutateurs
├── trajectories.py       # Dynamique jointe (mutations + transport)
├── datasets.py           # Génération des trajectoires et des contrôles
│
├── models/
│   ├── baseline_state.py
│   ├── baseline_history.py
│   └── lstm_model.py     # Modèle d'inférence de J_t
│
├── experiments/
│   ├── experiment_01_nk.py     # Ligne de base NK statique
│   ├── experiment_02_audit.py  # (Nouveau) Jalon 2b : spectres et courbure
│   └── controls.py             # Contrôles (intégrable, lissé, etc.)
│
├── metrics.py            # Wasserstein, Distance Topologique (D_J vs D_T)
├── run_experiment.py
│
└── tests/
    ├── test_nk.py
    ├── test_fiber_geometry.py  # Validation des spectres et de l'anholonomie
    └── test_leakage.py
```

10. Critères de validation

Une découverte structurelle ne peut être revendiquée que si :

· La courbure géométrique est mesurable sur la fibre.
· La mémoire topologique D_{\mathcal{T}}(t) survit à la relaxation métrique D_J(t).
· L'effet disparaît dans le modèle de contrôle à transport intégrable.
· Le LSTM démontre une capacité de prédiction statistiquement significative de \mathcal{T} par rapport aux baselines M1-M3.
· Les résultats sont stables sur au moins trois jeux de graines aléatoires et plusieurs valeurs de \theta_A.

11. Finalité scientifique

Le déplacement recherché est le suivant :

\text{Configuration} \rightarrow \text{Trajectoire} \rightarrow \text{Transport} \rightarrow \text{Courbure} \rightarrow \text{Mémoire topologique} 

L'objectif n'est pas seulement de prédire le prochain état. Il est de déterminer si la structure historique des transformations (la géométrie de l'espace des possibles) demeure invisible dans une description strictement instantanée, et si l'apprentissage profond peut révéler cette géométrie cachée.

---

NK-LSTM-PATH — Explorer la géométrie des possibles à travers la mémoire des trajectoires.
