V13.5 — Participation spatiale des modes singuliers du Jacobien F-I-B
Objectif
Le script v13_5_holographie_PR.py étudie la distribution spatiale des modes singuliers du Jacobien tangentiel conditionnel du système F-I-B.
L'objectif est de mesurer si les directions de perturbation associées aux plus grandes et aux plus petites valeurs singulières sont spatialement étendues ou concentrées sur une fraction du réseau.
L'analyse porte sur le Jacobien conditionnel :
[ J_{FI\mid B}
\left. \frac{\partial(F_{t+1},I_{t+1})} {\partial(F_t,I_t)} \right|_{B_t\ \mathrm{fixé}} ]
Le champ binaire (B_t) est maintenu constant pendant la différentiation. Le Jacobien décrit donc la dynamique tangentielle de la branche F-I correspondant à une configuration donnée de (B_t). Il ne constitue pas le Jacobien de l'application hybride complète F-I-B, qui comprend également la mise à jour discrète de (B).
Modèle dynamique
Le modèle utilise une mise à jour séquentielle :
[ F_{t+1}
(1-\alpha)F_t+ \alpha\tanh\left( I_t+G(2B_t-1)+D_F\Delta F_t \right) ]
[ I_{t+1}
(1-\beta)I_t+ \beta\tanh\left( F_{t+1}+D_I\Delta I_t-\Gamma I_t^3 \right) ]
puis actualise le champ binaire :
[ B_{t+1}
\mathbf{1} \left[ F_{t+1}+0.5I_{t+1}-\theta>0 \right]. ]
Le Laplacien (\Delta) est le Laplacien discret à cinq points avec conditions périodiques. Les paramètres par défaut sont définis dans le script.
Conditions comparées
Pour chaque graine et chaque instant sélectionné, trois conditions sont évaluées indépendamment.
1. Dynamique (dyn)
État directement issu de l'évolution du modèle F-I-B.
2. Contrôle Fourier (fourier)
Un même multiplicateur de phase est appliqué aux transformées de Fourier de (F) et de (I) :
[ \widehat F_s(k)=\widehat F(k)\phi(k), \qquad \widehat I_s(k)=\widehat I(k)\phi(k). ]
Le multiplicateur satisfait :
[ |\phi(k)|=1, \qquad \phi(-k)=\overline{\phi(k)}. ]
La composante continue (DC) est fixée à une phase égale à 1. La transformation conserve ainsi, analytiquement :
les modules spectraux individuels (|\widehat F(k)|) et (|\widehat I(k)|) ;
le spectre croisé complexe (\widehat F(k)\overline{\widehat I(k)}) ;
les normes spatiales de (F) et de (I), par Parseval ;
les moyennes spatiales, puisque la composante DC est inchangée.
Ces invariants sont soumis aux erreurs d'arrondi de la représentation flottante et des FFT.
Après transformation, le champ (B_s) est reconstruit à partir de (F_s+0.5I_s).
Deux modes de reconstruction sont disponibles :
quantile : le nombre de sites actifs est fixé exactement au nombre de sites actifs du champ (B) initial ;
theta : le seuil fixe SEUIL est appliqué au champ transformé ; la densité résultante peut donc varier.
Le contrôle Fourier ne conserve pas nécessairement la configuration spatiale de (B). Comme (B_s) intervient dans le Jacobien, le contraste dynamique-Fourier dépend aussi de cette reconstruction et du couplage de (B_s) au Jacobien.
3. Contrôle permutation (perm)
Une même permutation spatiale est appliquée à (F), (I) et (B).
Cette opération conserve :
les valeurs de chaque champ ;
les distributions marginales ;
les associations ponctuelles entre (F), (I) et (B).
Elle réarrange leur disposition sur le réseau et modifie leurs relations de voisinage avec le Laplacien périodique. Elle ne garantit pas la disparition de toute structure spatiale résiduelle.
Les trois conditions — dynamique, contrôle Fourier et contrôle permutation — sont des conditions comparées. Elles ne constituent pas une hiérarchie théorique prédéfinie.
Jacobien tangentiel conditionnel
En posant :
[ a=I_t+G(2B_t-1)+D_F\Delta F_t, ]
[ F_{t+1}=(1-\alpha)F_t+\alpha\tanh(a), ]
[ b=F_{t+1}+D_I\Delta I_t-\Gamma I_t^3, ]
et :
[ S_a=\operatorname{diag}(1-\tanh^2(a)), \qquad S_b=\operatorname{diag}(1-\tanh^2(b)), ]
les blocs du Jacobien conditionnel sont :
[ dFF=(1-\alpha)\mathbf I+\alpha D_F S_a\Delta, ]
[ dFI=\alpha S_a, ]
[ dIF=\beta S_b,dFF, ]
[ dII=(1-\beta)\mathbf I+ \beta S_b \left( dFI+D_I\Delta-3\Gamma\operatorname{diag}(I_t^2) \right). ]
Le Jacobien assemblé est :
[ J_{FI\mid B}
\begin{pmatrix} dFF & dFI\ dIF & dII \end{pmatrix}. ]
La dépendance séquentielle de (I_{t+1}) envers (F_{t+1}) est ainsi prise en compte dans le bloc (dIF).
Mesure de participation spatiale
Le script calcule une décomposition en valeurs singulières :
[ J=U\Sigma V^\mathsf{T}. ]
Pour chaque vecteur singulier droit (v), le poids spatial au site (i) est :
[ w_i=v_{F,i}^{,2}+v_{I,i}^{,2}. ]
Le rapport de participation est :
[ PR(w)= \frac{\left(\sum_iw_i\right)^2} {\sum_iw_i^2}. ]
La valeur normalisée est :
[ PR_{\mathrm{norm}}=\frac{PR(w)}{N}, \qquad N=L^2. ]
Pour un vecteur singulier droit normalisé, les poids sont non négatifs et leur somme vaut 1 à la précision numérique. Le PR effectif est compris entre 1 et (N), hors cas numérique dégénéré.
Une valeur normalisée proche de 1 indique une participation répartie sur une grande partie du réseau. Une valeur faible indique une concentration spatiale plus importante.
Le script rapporte :
PR_top1 : PR normalisé du vecteur singulier droit associé à la plus grande valeur singulière ;
PR_top5 : moyenne des PR normalisés des cinq vecteurs singuliers droits associés aux plus grandes valeurs singulières ;
PR_bot1 : PR normalisé du vecteur singulier droit associé à la plus petite valeur singulière ;
PR_bot5 : moyenne des PR normalisés des cinq vecteurs singuliers droits associés aux plus petites valeurs singulières ;
log10_kappa : logarithme décimal du conditionnement spectral estimé, si la plus petite valeur singulière dépasse le seuil numérique (10^{-12}).
Ces mesures décrivent l'étalement spatial des vecteurs singuliers droits. Elles ne constituent pas, à elles seules, une mesure de redondance informationnelle, de cohérence globale ou de conscience.
Analyse numérique
Deux modes de résolution sont disponibles :
SVD dense : méthode de référence, qui calcule la décomposition complète de la matrice ;
SVD creuse partielle : extraction exploratoire des modes associés aux plus grandes valeurs singulières (LM) et aux plus petites (SM).
Si une exception ARPACK interceptée survient pendant la résolution creuse, le script revient à la SVD dense. Ce repli ne signifie pas que toutes les erreurs possibles sont automatiquement récupérées.
L'extraction des plus petites valeurs singulières peut être numériquement délicate. La SVD creuse partielle ne garantit donc ni la convergence ni une meilleure scalabilité dans tous les régimes.
Le seuil (10^{-12}) utilisé pour le conditionnement est un seuil numérique choisi par l'implémentation ; il ne constitue pas une frontière universelle entre matrices singulières et non singulières.
Vérifications intégrées
La commande --verif exécute deux groupes de diagnostics.
Contrôle Fourier
Sur des champs réels de test générés indépendamment d'une trajectoire, le script mesure :
l'écart à l'unimodularité ;
l'écart à la symétrie hermitienne ;
les résidus imaginaires des transformations inverses ;
les écarts sur les modules spectraux individuels ;
l'écart sur le spectre croisé complexe ;
les écarts sur les normes spatiales.
Ces diagnostics vérifient numériquement les propriétés de la transformation sur des champs de test. Ils ne remplacent pas l'analyse des données réelles produites par une campagne expérimentale.
Jacobien conditionnel
Sur un petit réseau, le Jacobien analytique est comparé à une approximation par différences finies centrées, en maintenant (B) fixe. La comparaison est effectuée pour plusieurs valeurs de (D_F).
La vérification numérique ne constitue pas une preuve formelle, mais permet de détecter des erreurs d'implémentation dans les dérivées.
Comparaisons statistiques
Les contrastes sont calculés par différences appariées entre graines :
[ \Delta_{\mathrm{Fourier}}
M_{\mathrm{dyn}}-M_{\mathrm{fourier}}, ]
[ \Delta_{\mathrm{Permutation}}
M_{\mathrm{dyn}}-M_{\mathrm{perm}}, ]
où (M) désigne l'une des mesures rapportées.
Pour chaque contraste et chaque mesure, le script rapporte :
la moyenne des différences ;
l'intervalle bootstrap percentile à 95 % ;
la taille d'effet descriptive (d_z) ;
le nombre de différences positives et le nombre de graines valides.
La taille d'effet est calculée comme :
[ d_z=\frac{\overline{\Delta}}{s_\Delta}, ]
où (s_\Delta) est l'écart-type empirique des différences appariées.
Les unités statistiques sont les graines, et non les sites du réseau. Les intervalles bootstrap sont descriptifs et leur précision dépend du nombre de graines. Le script ne réalise pas de correction pour comparaisons multiples entre les mesures et les instants.
Périmètre d'interprétation
Le contraste dynamique-Fourier mesure une différence de participation spatiale entre la dynamique et un contrôle qui conserve les modules spectraux individuels et le spectre croisé complexe de (F) et (I).
Une différence observée indique que ces invariants spectraux ne suffisent pas, à eux seuls, à déterminer la mesure de participation calculée dans les conditions comparées.
Elle ne permet pas d'attribuer cette différence exclusivement à des corrélations d'ordre supérieur : la reconstruction de (B), sa densité ou sa configuration, ainsi que son couplage au Jacobien, contribuent également au contraste.
Le contraste dynamique-permutation répond à une autre question. Il mesure la sensibilité des résultats au réarrangement spatial commun des champs sur le réseau, qui modifie leurs voisinages relativement au Laplacien.
Les deux contrôles sont complémentaires, mais non interchangeables.
Le PR décrit une participation spatiale des vecteurs singuliers. Une différence de PR ne démontre pas, à elle seule, une causalité physique particulière ni une propriété informationnelle globale.
Sorties et reproductibilité
Le fichier CSV contient une ligne par combinaison de graine, d'instant et de condition. Les colonnes comprennent notamment :
seed, t, cas ;
densB, densité du champ binaire ;
lapF, norme du Laplacien de (F) ;
les cinq mesures spectrales.
Le script ne génère pas les graphiques de synthèse montrés dans une figure externe. Ces graphiques doivent être produits à partir du CSV, et leur provenance doit être conservée.
Pour rendre une figure reproductible, archiver avec elle :
le script exact utilisé ;
le CSV correspondant ;
les arguments de commande ;
la version de Python, NumPy et SciPy ;
le mode de SVD ;
le mode de reconstruction de (B) ;
les graines et les instants ;
le code de génération de la figure, s'il est distinct.
Aucun résultat numérique de campagne n'est présumé dans ce document. Les valeurs et graphiques doivent être considérés comme validés uniquement après exécution du script et vérification des sorties correspondantes.
Exécution
Vérifications intégrées :
python v13_5_holographie_PR.py --verif
Campagne par défaut ((L=21), 10 graines, instants 5, 100 et 500) :
python v13_5_holographie_PR.py --L 21 --seeds 10
SVD creuse partielle exploratoire :
python v13_5_holographie_PR.py --L 21 --seeds 10 --sparse
Contrôle Fourier avec seuil fixe pour reconstruire (B) :
python v13_5_holographie_PR.py --L 21 --seeds 10 --B-mode theta
Choix explicite des instants et du fichier de sortie :
python v13_5_holographie_PR.py --L 21 --seeds 10 --times 5,100,500 --out results/v13_5_pr.csv
Dépendances Python :
pip install numpy scipy
Fichier de sortie par défaut :
results/v13_5_pr.csv
Statut
Le document décrit le protocole et l'implémentation. La validation effective d'une campagne dépend de l'exécution du script, de l'examen du CSV et de la reproductibilité des figures associées. La présence des fonctions de vérification dans le code ne prouve pas, à elle seule, qu'elles ont été exécutées avec succès.
