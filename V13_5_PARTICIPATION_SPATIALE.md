---

## V13.5 — Participation spatiale des modes singuliers du Jacobien F-I-B

### Objectif

Le script `v13_5_holographie_PR.py` étudie la distribution spatiale
des modes singuliers du Jacobien tangentiel du système F-I-B.

Il cherche à déterminer dans quelle mesure les perturbations locales
du système sont associées à des modes singuliers spatialement étendus
ou, au contraire, concentrés sur une fraction du réseau.

L'analyse porte sur le Jacobien conditionnel :

\[
J_{FI|B}
=
\left.
\frac{\partial(F_{t+1},I_{t+1})}
{\partial(F_t,I_t)}
\right|_{B_t\ \mathrm{fixé}}
\]

Le champ binaire \(B_t\) est maintenu constant pendant la
différentiation. Le Jacobien obtenu décrit donc la dynamique
tangentielle de la branche F-I correspondant à une configuration
donnée de B. Il ne constitue pas le Jacobien classique de
l'application hybride complète F-I-B.

### Principe expérimental

Pour chaque graine et chaque instant sélectionné, le script construit
trois configurations comparables :

1. **Dynamique (`dyn`)**  
   État directement issu de l'évolution du modèle F-I-B.

2. **Substitut Fourier (`fourier`)**  
   Transformation des champs F et I par application d'un même
   multiplicateur de phase dans l'espace de Fourier.

   Ce multiplicateur satisfait les propriétés suivantes :

   - module unitaire : \(|\phi(k)|=1\) ;
   - symétrie hermitienne, à la précision flottante ;
   - conservation des modules spectraux individuels ;
   - conservation du spectre croisé complexe entre F et I ;
   - conservation des normes spatiales.

   Après transformation, le champ B est reconstruit à partir de
   \(F_s+0.5I_s\).

   En mode `quantile`, le nombre de sites actifs de B est conservé
   exactement. En mode `theta`, un seuil fixe est appliqué et la
   densité résultante peut varier.

3. **Permutation (`perm`)**  
   Une même permutation spatiale est appliquée à F, I et B.
   Les valeurs des champs et leurs associations entre eux sont
   conservées, mais leur organisation sur le réseau est modifiée.

Cette hiérarchie permet de comparer la dynamique à deux contrôles
distincts : un contrôle spectral par transformation de phase et
un contrôle spatial par permutation.

### Mesure de participation spatiale

Le script calcule la décomposition en valeurs singulières du Jacobien :

\[
J=U\Sigma V^\mathsf{T}
\]

Pour chaque vecteur singulier droit \(v\), le poids associé au site
spatial \(i\) est défini par :

\[
w_i=v_{F,i}^{\,2}+v_{I,i}^{\,2}
\]

Le rapport de participation est ensuite calculé :

\[
PR(w)=
\frac{\left(\sum_i w_i\right)^2}
{\sum_i w_i^2}
\]

La mesure est normalisée par le nombre de sites \(N=L^2\) :

\[
PR_{\mathrm{norm}}=\frac{PR(w)}{N}
\]

Une valeur proche de 1 indique une participation répartie sur une
grande partie du réseau. Une valeur plus faible indique une
concentration spatiale plus importante.

Le script rapporte :

- `PR_top1` : participation du mode associé à la plus grande
  valeur singulière ;
- `PR_top5` : participation moyenne des cinq modes dominants ;
- `PR_bot1` : participation du mode associé à la plus petite
  valeur singulière ;
- `PR_bot5` : participation moyenne des cinq modes contractés ;
- `log10_kappa` : logarithme décimal du conditionnement spectral,
  lorsque celui-ci peut être estimé.

Ces mesures caractérisent l'étalement spatial des vecteurs singuliers.
Elles ne constituent pas, à elles seules, une mesure de redondance
informationnelle ou de cohérence globale.

### Analyse numérique

Deux modes de résolution sont disponibles :

- **SVD dense** : méthode de référence pour les tailles de réseau
  où la matrice peut être traitée en mémoire ;
- **SVD creuse partielle** : extraction exploratoire des modes
  dominants et contractés, avec repli sur la SVD dense en cas
  d'échec de convergence ARPACK.

La SVD creuse ne garantit pas une meilleure scalabilité pour les
plus grandes tailles. En particulier, l'extraction des plus petites
valeurs singulières peut être numériquement délicate.

Le mode `--verif` permet de contrôler :

- l'unimodularité du multiplicateur Fourier ;
- sa symétrie hermitienne ;
- le résidu imaginaire des transformations inverses ;
- la conservation des spectres individuels ;
- la conservation du spectre croisé complexe ;
- la conservation des normes spatiales ;
- la cohérence du Jacobien analytique avec une approximation
  par différences finies centrées.

### Comparaisons statistiques

Les contrastes principaux sont calculés par différences appariées
entre graines :

\[
\Delta PR_{\mathrm{Fourier}}
=
PR_{\mathrm{dyn}}-PR_{\mathrm{fourier}}
\]

\[
\Delta PR_{\mathrm{Permutation}}
=
PR_{\mathrm{dyn}}-PR_{\mathrm{perm}}
\]

Le script rapporte, pour chaque mesure :

- la différence moyenne ;
- l'intervalle bootstrap percentile à 95 % ;
- la taille d'effet descriptive \(d_z\) ;
- le nombre de différences positives et le nombre de graines
  effectivement utilisées.

Les unités statistiques sont les graines indépendantes, et non
les sites du réseau. Les intervalles bootstrap sont donc descriptifs
et dépendent du nombre de graines disponibles.

### Périmètre d'interprétation

Le contraste dynamique-Fourier mesure une différence de participation
spatiale entre la dynamique et un contrôle conservant les invariants
spectraux de F et I.

Il permet d'examiner si la distribution spatiale des modes singuliers
dépend de caractéristiques qui ne sont pas déterminées par les
spectres individuels et croisés conservés.

Il ne permet toutefois pas d'attribuer cette différence exclusivement
à des corrélations d'ordre supérieur. La reconstruction de B, son
couplage au Jacobien et les propriétés locales des champs transformés
contribuent également au contraste.

Le contraste dynamique-permutation répond à une autre question :
il évalue la sensibilité des mesures à la réorganisation spatiale
des champs sur le réseau.

Les deux contrôles sont complémentaires, mais ne sont pas
interchangeables.

### Exécution

Validation numérique :

```bash
python v13_5_holographie_PR.py --verif
