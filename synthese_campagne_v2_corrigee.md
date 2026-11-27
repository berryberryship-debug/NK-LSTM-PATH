# Synthèse corrigée — Campagne V2 (NK-LSTM-GSA)

## 1. Observations mesurées

### A. Cohorte homogène en régime d'ensemble (J0 variable par état initial)
- **Périmètre retenu :** sur 23 exécutions symétriques uniques détectées (`runs/campagne_v2` et `runs/v2_*`, 3 doublons écartés), les 4 exécutions à `n_initial_states = 128` ont été exclues pour isoler une cohorte homogène de 19 exécutions (`n_initial_states = 512`, `eta = 0.25`, `transport_mode = symmetric`, `n_sites = 6`, `d_model = 64`, `gsa_layers = 2`, `beta_action = 1.0`, `lambda_geometry = 0.001`).
- **Perte de validation (`best_val_loss`, moyenne sur les graines `s7`, `s17`, `s27`) :**
  - `hybrid_concat (learned)` : 0.001746 (`s7` : 0.001803 ; `s17` : 0.001426 ; `s27` : 0.002008)
  - `hybrid_concat (none)` : 0.001783 (`s7` : 0.001922 ; `s17` : 0.001584 ; `s27` : 0.001844)
  - `hybrid_concat (fixed)` : 0.001822 (`s7` : 0.001858 ; `s17` : 0.001535 ; `s27` : 0.002074)
  - `gsa_only (learned)` : 0.002398 (`s7` : 0.002231 ; `s17` : 0.002417 ; `s27` : 0.002547)
  - `gsa_only (fixed)` : 0.002530 (`s7` : 0.002679 ; `s17` : 0.002438 ; `s27` : 0.002474)
  - `hybrid_gated (learned)` : 0.002783 (`s7` uniquement)
  - `lstm_only (none)` : 0.002847 (`s7` : 0.002594 ; `s17` : 0.003256 ; `s27` : 0.002690)
- **Extraction des 192 coûts d'arêtes non orientées (`c_e = softplus(gsa.edge_costs.edge_logits) + 1e-4`, valeur initiale uniforme `ln(2) + 1e-4 = 0.693247`) :**
  - `hybrid_concat (learned)` (`s17`, sélectionné par `best_val_loss = 0.001426` hors test) : min = 0.685554, moyenne = 0.693193, écart-type = 0.003428, max = 0.701439. Moyennes par site : S0 = 0.6930, S1 = 0.6933, S2 = 0.6935, S3 = 0.6933, S4 = 0.6928, S5 = 0.6932. Corrélations de Pearson inter-graines `(s7, s17, s27)` : `r(s7, s17) = 0.0667`, `r(s7, s27) = 0.0398`, `r(s17, s27) = 0.1302`.
  - `gsa_only (learned)` (`s7`, `s17`, `s27`) :
    - `s7` : min = 0.67711, moyenne = 0.69366, écart-type = 0.00604, max = 0.71019
    - `s17` : min = 0.67718, moyenne = 0.69353, écart-type = 0.00671, max = 0.70952
    - `s27` : min = 0.67869, moyenne = 0.69346, écart-type = 0.00682, max = 0.71820
    - Corrélations de Pearson inter-graines `(s7, s17, s27)` : `r(s7, s17) = 0.0244`, `r(s7, s27) = -0.1318`, `r(s17, s27) = 0.0569`.

### B. Expérience de contrôle en paysage J0 fixé (régime trempé, T = 2)
- **Protocole :** une seule matrice de couplage `J0` fixe gouverne l'ensemble des 512 états initiaux (`gsa_only`, `eta = 0.25`, graines `s7`, `s17`, `s27`).
- **Résultats comparés :**
  - `gsa_only_fixed` : `best_val_loss` moyen = 0.000141 | `Test Omega MSE` moyen = 0.000047 | cosinus moyen = 0.9956
  - `gsa_only_learned` (`lambda_geometry = 1e-3`) : `best_val_loss` moyen = 0.000141 | `Test Omega MSE` moyen = 0.000047 | cosinus moyen = 0.9956
    - `s7` : coûts dans [0.65578, 0.72850] (moyenne = 0.69335, écart-type = 0.01242)
    - `s17` : coûts dans [0.65794, 0.72043] (moyenne = 0.69307, écart-type = 0.01093)
    - `s27` : coûts dans [0.65766, 0.72200] (moyenne = 0.69325, écart-type = 0.00974)
    - Corrélations inter-graines `edge_logits` : `r(s7, s17) = 0.1300`, `r(s7, s27) = 0.1269`, `r(s17, s27) = 0.1131` (moyenne = 0.1233)
  - `gsa_only_learned` (`lambda_geometry = 1e-5`) : `best_val_loss` moyen = 0.000140 | `Test Omega MSE` moyen = 0.000046 | cosinus moyen = 0.9956
    - `s7` : coûts dans [0.56595, 0.78000] (moyenne = 0.69030, écart-type = 0.03620)
    - `s17` : coûts dans [0.58998, 0.79241] (moyenne = 0.68909, écart-type = 0.03605)
    - `s27` : coûts dans [0.59537, 0.79191] (moyenne = 0.68989, écart-type = 0.03364)
    - Corrélations inter-graines `edge_logits` : `r(s7, s17) = 0.2974`, `r(s7, s27) = 0.3469`, `r(s17, s27) = 0.3001` (moyenne = 0.3148)

### C. Extension aux trajectoires longues (T = 3, L = 4) sur paysage J0 fixé
- **Protocole :** même matrice `J0` fixe, 512 états initiaux (`25 600` paires : `10 240` permutations `(k, m, r)` vs `(r, m, k)` et `15 360` détours `(k, k, k)` vs `(m, k, m)`), avec contrôle d'échelle `gsa_only_fixed_scaled` (`beta_action = ln(2) + 1e-4 = 0.693247`).
- **Résultats comparés (`s7`, `s17`, `s27`) :**
  - `gsa_only_fixed` (`beta = 1.0`) : `best_val_loss` = 0.000176 | `Test MSE` global = 0.000093 (Perm : 0.000094 | Détours : 0.000092) | cosinus = 0.9986
  - `gsa_only_fixed_scaled` (`beta = 0.693247`) : `best_val_loss` = 0.000183 | `Test MSE` global = 0.000097 (Perm : 0.000095 | Détours : 0.000097) | cosinus = 0.9985
  - `gsa_only_learned` (`lambda_geometry = 1e-3`) : `best_val_loss` = 0.000182 | `Test MSE` global = 0.000095 (Perm : 0.000092 | Détours : 0.000097) | cosinus = 0.9985
    - Écart-type des coûts : `s7` = 0.01045, `s17` = 0.01011, `s27` = 0.00851
    - Corrélations inter-graines `edge_logits` : `r(s7, s17) = 0.2129`, `r(s7, s27) = 0.2459`, `r(s17, s27) = 0.0981` (moyenne = 0.1856)
  - `gsa_only_learned` (`lambda_geometry = 1e-5`) : `best_val_loss` = 0.000176 | `Test MSE` global = 0.000092 (Perm : 0.000092 | Détours : 0.000093) | cosinus = 0.9986
    - `s7` : `val` = 0.000180 | MSE = 0.000095 (Perm : 0.000093, Dét : 0.000096) | coûts dans [0.53148, 0.82249] (`0.68430 ± 0.04974`)
    - `s17` : `val` = 0.000167 | MSE = 0.000086 (Perm : 0.000084, Dét : 0.000087) | coûts dans [0.53631, 0.77432] (`0.68735 ± 0.04767`)
    - `s27` : `val` = 0.000180 | MSE = 0.000096 (Perm : 0.000099, Dét : 0.000094) | coûts dans [0.55238, 0.79020] (`0.68420 ± 0.04441`)
    - Corrélations inter-graines `edge_logits` : `r(s7, s17) = 0.5624`, `r(s7, s27) = 0.5893`, `r(s17, s27) = 0.4474` (moyenne = 0.5330)

## 2. Interprétation mécanistique
- **Rectification du diagnostic géométrique :** l'ancien diagnostic fondé sur le Gramien de `gsa.x_proj.weight` reflétait la colinéarité de la projection linéaire d'entrée et non les distances géodésiques sur l'hypercube `{0,1}^6`.
- **Rôle de l'invariance statistique :** lorsque chaque état initial tire une matrice `J0` aléatoire indépendante, l'ensemble d'entraînement est statistiquement invariant par permutation des 6 sites et retournement des spins ; une table de coûts statique `c_theta(x, a)` indépendante de `J0` converge donc vers la métrique uniforme de Hamming (`~ ln(2)`).
- **Amplification de la convergence métrique avec T :** en paysage `J0` fixé avec `lambda_geometry = 1e-5`, allonger les trajectoires de `T = 2` à `T = 3` fait passer la corrélation inter-graines des 192 coûts de `0.3148` à `0.5330` et réduit la MSE de test de `5.2 %` par rapport au témoin à échelle neutralisée `fixed_scaled` (`0.000097 -> 0.000092`), bien que `fixed` brut (`beta = 1.0`, `0.000093`) reste très proche grâce au couplage fort du biais temporel.

## 3. Hypothèses pour l'étape suivante
1. **Test du détour géodésique (Tolman) sur T = 3 :** vérifier si les 192 coûts appris (`r moyen = 0.5330`) sélectionnent effectivement le contournement `(m, k, m)` de plus faible résistance physique lors du blocage de l'arête directe `k`, ou tester `T = 4` pour observer si l'écart prédictif face à `fixed` brut se creuse.
2. **Coûts conditionnés `c_theta(x, a | J0)` en régime multi-`J0` :** remplacer la table statique `UndirectedPositiveEdgeCosts` par un champ de coûts dépendant explicitement de `J0` afin de transposer cette reproductibilité géométrique au régime d'ensemble.
