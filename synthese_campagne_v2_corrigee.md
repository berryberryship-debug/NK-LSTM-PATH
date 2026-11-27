# Synthèse corrigée — Campagne V2 (NK-LSTM-GSA)

## 1. Observations mesurées

### A. Cohorte homogène en régime d'ensemble (J0 variable par état initial, T = 2)
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
    - Corrélations de Pearson inter-graines `(s7, s17, s27)` : `r(s7, s17) = 0.0244`, `r(s7, s27) = -0.1318`, `r(s17, s27) = 0.0569` (moyenne = -0.0168).

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

### C. Extension aux trajectoires longues (T = 3, L = 4) et diagnostic de Tolman sur paysage J0 fixé
- **Protocole :** même matrice `J0` fixe, 512 états initiaux (`25 600` paires : `10 240` permutations `(k, m, r)` vs `(r, m, k)` et `15 360` détours `(k, k, k)` vs `(m, k, m)`), avec contrôle d'échelle `gsa_only_fixed_scaled` (`beta_action = ln(2) + 1e-4 = 0.693247`).
- **Résultats comparés (`s7`, `s17`, `s27`) :**
  - `gsa_only_fixed` (`beta = 1.0`) : `best_val_loss` = 0.000176 | `Test MSE` global = 0.000093 (Perm : 0.000094 | Détours : 0.000092) | cosinus = 0.9986
    - `s7` : `val` = 0.000177 | MSE (`all` = 0.000093, `perm` = 0.000093, `dét` = 0.000093)
    - `s17` : `val` = 0.000179 | MSE (`all` = 0.000092, `perm` = 0.000091, `dét` = 0.000092)
    - `s27` : `val` = 0.000173 | MSE (`all` = 0.000093, `perm` = 0.000098, `dét` = 0.000090)
  - `gsa_only_fixed_scaled` (`beta = 0.693247`) : `best_val_loss` = 0.000183 | `Test MSE` global = 0.000097 (Perm : 0.000095 | Détours : 0.000097) | cosinus = 0.9985
    - `s7` : `val` = 0.000180 | MSE (`all` = 0.000095, `perm` = 0.000091, `dét` = 0.000098)
    - `s17` : `val` = 0.000181 | MSE (`all` = 0.000096, `perm` = 0.000096, `dét` = 0.000096)
    - `s27` : `val` = 0.000186 | MSE (`all` = 0.000099, `perm` = 0.000099, `dét` = 0.000099)
  - `gsa_only_learned` (`lambda_geometry = 1e-3`) : `best_val_loss` = 0.000182 | `Test MSE` global = 0.000095 (Perm : 0.000092 | Détours : 0.000097) | cosinus = 0.9985
    - `s7` : `val` = 0.000188 | MSE (`all` = 0.000098, `perm` = 0.000092, `dét` = 0.000102) | coûts dans [0.66655, 0.72093] (`0.69334 ± 0.01045`)
    - `s17` : `val` = 0.000173 | MSE (`all` = 0.000088, `perm` = 0.000086, `dét` = 0.000090) | coûts dans [0.66432, 0.71914] (`0.69337 ± 0.01011`)
    - `s27` : `val` = 0.000186 | MSE (`all` = 0.000099, `perm` = 0.000099, `dét` = 0.000099) | coûts dans [0.67046, 0.71782] (`0.69317 ± 0.00851`)
    - Corrélations inter-graines `edge_logits` : `r(s7, s17) = 0.2129`, `r(s7, s27) = 0.2459`, `r(s17, s27) = 0.0981` (moyenne = 0.1856)
  - `gsa_only_learned` (`lambda_geometry = 1e-5`) : `best_val_loss` = 0.000176 | `Test MSE` global = 0.000092 (Perm : 0.000092 | Détours : 0.000093) | cosinus = 0.9986
    - `s7` : `val` = 0.000180 | MSE (`all` = 0.000095, `perm` = 0.000093, `dét` = 0.000096) | coûts dans [0.53148, 0.82249] (`0.68430 ± 0.04974`)
    - `s17` : `val` = 0.000167 | MSE (`all` = 0.000086, `perm` = 0.000084, `dét` = 0.000087) | coûts dans [0.53631, 0.77432] (`0.68735 ± 0.04767`)
    - `s27` : `val` = 0.000180 | MSE (`all` = 0.000096, `perm` = 0.000099, `dét` = 0.000094) | coûts dans [0.55238, 0.79020] (`0.68420 ± 0.04441`)
    - Corrélations inter-graines `edge_logits` : `r(s7, s17) = 0.5624`, `r(s7, s27) = 0.5893`, `r(s17, s27) = 0.4474` (moyenne = 0.5330)
- **Diagnostic du détour de Tolman (618 tâches de test et 384 tâches sur les 64 sommets, référence uniforme : Top-1 = 20.0 %, Top-2 = 40.0 %, Rang = 3.00 / 5, Regret = 1.000) :**
  - Sur l'holonomie d'entraînement `T = 3` (`(k, k, k)` vs `(m, k, m)`), la métrique apprise bat systématiquement le hasard uniforme :
    - `seed_7` : Top-1 = 28.5 %, Top-2 = 50.3 %, Rang = 2.64 / 5, Spearman = +0.118, Regret = 0.739
    - `seed_17` : Top-1 = 25.2 %, Top-2 = 45.8 %, Rang = 2.83 / 5, Spearman = +0.094, Regret = 0.865
    - `seed_27` : Top-1 = 26.4 %, Top-2 = 43.2 %, Rang = 2.82 / 5, Spearman = +0.079, Regret = 0.834
    - `ensemble_mean` : Top-1 = 26.9 %, Top-2 = 48.7 %, Rang = 2.71 / 5, Spearman = +0.115, Regret = 0.786 (-21.4 % de regret)
  - Face à l'arête directe non amortie `(k,)`, la métrique apprise ne bat pas le hasard (`ensemble_mean` : Top-1 = 18.0 %, Top-2 = 33.0 %, Rang = 3.25 / 5, Spearman = -0.048, Regret = 1.105 ; `seed_7` : Top-1 = 22.5 %, `seed_17` : Top-1 = 8.4 %, `seed_27` : Top-1 = 18.1 %) en raison de la contraction dissipative `(1 - eta)^3` propre aux triples retournements `(k, k, k)`.

### D. Coûts d'arêtes conditionnés c_theta(x, a | J0) en régime multi-J0 (paysages J0 variables)
- **Protocole :** `lambda_geometry = 1e-5`, graines `s7`, `s17`, `s27` sur paysages `J0` aléatoires indépendants (`512` `J0` à `T = 2` ; `128` `J0` à `T = 3`).
- **Régime multi-J0 à T = 2 (512 J0 distincts, 7 680 paires) :**
  - `gsa_only_fixed` : `best_val_loss` = 0.007999 | `Test Omega MSE` = 0.003560 | `Test Term MSE` = 0.003776 | cosinus = 0.7176
  - `gsa_only_fixed_scaled` : `best_val_loss` = 0.008026 | `Test Omega MSE` = 0.003559 | `Test Term MSE` = 0.003780 | cosinus = 0.7178
  - `gsa_only_static_learned` : `best_val_loss` = 0.007994 | `Test Omega MSE` = 0.003547 | `Test Term MSE` = 0.003756 | cosinus = 0.7189 (corrélation inter-graines moyenne = 0.1835)
  - `gsa_only_conditioned_j0` : `best_val_loss` = 0.007738 | `Test Omega MSE` = 0.003497 | `Test Term MSE` = 0.003560 | cosinus = 0.7267
    - `s7` : `val` = 0.008976 | `test_omega_mse` = 0.003932 | coûts dans [0.2209, 3.8740] (`moy = 1.5052`, `std_intra_J0 = 0.5276`, `std_inter_J0 = 0.5289`)
    - `s17` : `val` = 0.008345 | `test_omega_mse` = 0.003720 | coûts dans [0.2324, 3.1613] (`moy = 1.4113`, `std_intra_J0 = 0.4371`, `std_inter_J0 = 0.4307`)
    - `s27` : `val` = 0.005894 | `test_omega_mse` = 0.002840 | coûts dans [0.2158, 3.6813] (`moy = 1.3408`, `std_intra_J0 = 0.4822`, `std_inter_J0 = 0.4768`)
    - Corrélations inter-graines des coûts générés sur `J0` de test : `r(s7, s27) = +0.9033`, `r(s7, s17) = -0.8453`, `r(s17, s27) = -0.8227` (inversion de polarité Z2 sur `s17`).
- **Régime multi-J0 à T = 3 (128 J0 distincts, 6 400 paires) :**
  - `gsa_only_fixed` : `best_val_loss` = 0.063332 | `Test Omega MSE` = 0.037283 | `Test Term MSE` = 0.029332 | cosinus = 0.3912
  - `gsa_only_fixed_scaled` : `best_val_loss` = 0.065335 | `Test Omega MSE` = 0.037663 | `Test Term MSE` = 0.030086 | cosinus = 0.3863
  - `gsa_only_static_learned` : `best_val_loss` = 0.066070 | `Test Omega MSE` = 0.037407 | `Test Term MSE` = 0.029855 | cosinus = 0.3829
  - `gsa_only_conditioned_j0` : `best_val_loss` = **0.057119** | `Test Omega MSE` = **0.034243** (**-8.2 %** vs `fixed`, **-9.1 %** vs `fixed_scaled`, gain sur 3/3 graines) | `Test Term MSE` = **0.026603** (**-9.3 %** vs `fixed`) | cosinus = **0.4386** (**+12.1 %** relatif)
    - `s7` : `val` = 0.061007 | `test_omega_mse` = 0.035361 | coûts dans [0.0843, 2.1577] (`moy = 0.7468`, `std_intra_J0 = 0.2830`, `std_inter_J0 = 0.2875`)
    - `s17` : `val` = 0.052961 | `test_omega_mse` = 0.031674 | coûts dans [0.0948, 2.7120] (`moy = 0.9420`, `std_intra_J0 = 0.3559`, `std_inter_J0 = 0.3640`)
    - `s27` : `val` = 0.057390 | `test_omega_mse` = 0.035694 | coûts dans [0.1040, 2.2275] (`moy = 0.7023`, `std_intra_J0 = 0.2779`, `std_inter_J0 = 0.2470`)
    - Corrélations inter-graines des coûts générés sur `J0` inédits : `r(s7, s17) = +0.5765`, `r(s7, s27) = +0.4391`, `r(s17, s27) = +0.5775` (moyenne = **+0.5310**).

## 2. Interprétation mécanistique
- **Rectification du diagnostic géométrique :** l'ancien diagnostic fondé sur le Gramien de `gsa.x_proj.weight` mesurait la colinéarité de la projection linéaire d'entrée et non les distances géodésiques sur `{0,1}^6`.
- **Rôle de l'invariance statistique :** lorsque chaque état initial tire une matrice `J0` aléatoire indépendante, l'ensemble d'entraînement est statistiquement invariant par permutation des 6 sites et retournement des spins ; une table de coûts statique `c_theta(x, a)` indépendante de `J0` converge donc vers la métrique uniforme de Hamming (`~ ln(2)`).
- **Amplification de la convergence métrique avec T :** en paysage `J0` fixé avec `lambda_geometry = 1e-5`, allonger les trajectoires de `T = 2` à `T = 3` fait passer la corrélation inter-graines des 192 coûts de `0.3148` à `0.5330` et réduit la MSE de test de `5.2 %` par rapport au témoin à échelle neutralisée `fixed_scaled` (`0.000097 -> 0.000092`), bien que `fixed` brut (`beta = 1.0`, `0.000093`) reste très proche grâce au couplage fort du biais temporel.
- **Validation du conditionnement c_theta(x, a | J0) en régime multi-J0 :** lorsque la métrique est conditionnée par `J0`, le modèle généralise la topographie des coûts à des paysages `J0` inédits (`r moyen = +0.5310` entre graines à `T = 3`) et améliore simultanément la fermeture d'holonomie `Omega` (`-8.2 %` de MSE) et la prédiction finale `J_T` (`-9.3 %` de MSE). L'inversion de polarité observée sur la graine `s17` à `T = 2` (`r = -0.8453`) montre qu'une paramétrisation invariante sous `S_6 x Z_2` fondée sur les champs locaux non orientés `(|h_k(x, J0)|, sqrt(sum_{j != k} J0_{kj}^2))` permettra de verrouiller cette cohérence sans renversement de signe entre graines.
