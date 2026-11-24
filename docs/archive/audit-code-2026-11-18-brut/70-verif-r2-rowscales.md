# Vérification contradictoire de R2-API-01 et R2-API-02 (row scales)

Périmètre lu : `65-r2-api-python.md` (R2-API-01, R2-API-02 en entier), `BRIEF.md`, `BRIEF-R2.md`.
Rust : `llvq-artifact/src/format.rs:1-70, :468-497, :1572-1603` ; `llvq-quant/src/quantizer.rs:959-990` ;
`llvq-quant/src/rotation.rs` en entier ; `llvq-quant/src/gptq.rs:50-60, :288-300` ; `llvq-llm/src/bin/rowscale.rs`
en entier ; `llvq-llm/src/bin/export.rs:1-180` ; `llvq-llm/src/tetra_cpu.rs:15-44, :220-236` ;
`llvq-llm/src/fused.rs` (champs `tail`, `rotation`, refus sans rotation) ; `llvq-llm/src/sealed.rs:537` ;
`llvq-cuda/kernels/llvq_pers.cuh:1-30, :318-331` ; `llvq-cuda/kernels/pers_4et.cu:18-32` ;
`llvq-bench/examples/rhoapply.rs:1-110`.
Python : `ops/llvqtune/llvqtune/trainables/row_scales.py`, `row_norms.py`, `free_params.py`, `__init__.py` ;
`adapters/torch_model.py`, `adapters/torch_optimizer.py` ; `__main__.py` ; `train.sh` ; `README.md` ;
`tests/test_row_norms.py`, `tests/test_trainables.py` en entier.
Journaux et preregs : `dclm-rowscales-2026-09-20.txt`, `rowscale-controles-2026-09-19.txt`,
`tetranu-rowscales-2026-09-19.txt:110-120`, `dclm-8b-rowscales-2026-09-21.txt` et `dclm-14b-rowscales-2026-09-22.txt`
(survolés), `proofs/preregistration-tetranu-rowscales-2026-09-19-ECARTS.md:95-130`,
`proofs/preregistration-dclm-rowscales-2026-09-19.md:60-80`, `proofs/preregistration-mmluaux-4b-rowscales-2026-09-23.md:95-112`,
`docs/HISTORIQUE.md:392, :521-535`, `docs/ETAT.md:150-160`, `paper2/sections/model.tex:15-41`.
Aucun `cargo`, `uv`, `pytest`. Un seul calcul : un `python3` pur (sans numpy) qui réimplante la construction de
`rotation.rs` pour vérifier la formule de l'écart (section 4).

## Verdicts

- **R2-API-01 : NUANCÉ.** Le mécanisme est exact, ligne par ligne, et la formule de l'écart aussi. L'ordre de grandeur
  « 11 % de l'édition » tient en espace des poids. Mais le défaut est connu et assumé depuis le 2026-09-19 (déviation
  E8), aucun chiffre publié n'est faux, l'effet sur l'optimum de σ est du second ordre et estimé très sous le bruit
  MMLU, et une phrase du scénario est fausse (le `down_proj` du 8B n'a pas de queue). Gravité à ramener à mineure. Ce
  qui reste vrai et neuf : le chiffre de E8 (1,2e-4) est faux d'un facteur 18 et n'est pas une borne, E8 n'a pas été
  reporté aux bras 8B et 14B, et trois docstrings affirment le contraire de E8.
- **R2-API-02 : CONFIRMÉ.** σ est un paramètre f32 converti en bf16 avant le produit, sans autocast. C'est le schéma
  classique « maître f32, passage avant bf16 ». Une phrase est exagérée : le pas bf16 vaut un quart à un huitième de
  l'écart type mesuré de σ, pas « le même ordre ». La correction proposée sur le chemin `weight` ajouterait une copie
  f32 de tous les poids routés au graphe (14,1 Go au 4B) ; une forme sans surcoût existe.

## R2-API-01

### 1. Côté Rust : où vivent les row scales, la queue et la rotation

Le format le dit en deux lignes. `row_scales` : « One per output row, in the rotated basis » (`format.rs:479-480`).
`tail` : « Trailing columns kept at full precision, `d_out × (d_in % 24)` row-major, in the rotated basis »
(`format.rs:489-491`).

Le décodeur dense, `decode_matrix` (`format.rs:1582-1603`), fait dans l'ordre :
1. chaque bloc par `reconstruct_shape_gain(code, centroids, row_scales[i], block)` (`:1590`), qui multiplie
   `centroids[g] * row_scale` (`quantizer.rs:981`) : la row scale touche les blocs, et seulement eux ;
2. la queue copiée telle quelle derrière les blocs, sans row scale (`:1594-1597`) ;
3. `unrotate_weight_rows` sur la ligne entière de largeur `d_in` (`:1599-1601`), soit `apply_transpose`, `v ← Qᵀ v`
   (`rotation.rs:108-118, :156-162`).

C'est ce décodeur que lisent le MMLU et la ppl denses (`sealed.rs:537`) et `bin/export` (`export.rs:47-56`).

Les chemins fusés appliquent la même convention, en base tournée côté activation :
- CPU : `finish` rend `acc * rscale[r]` plus la queue fois `x[nblocks*24..]`, l'activation déjà tournée
  (`tetra_cpu.rs:226-235`, doc `:26-29` : « The row scale multiplies the sum and the exact tail is added last »).
- CUDA servi : `pers_epilogue` calcule `tv = Σ tail[i]·x[i]` puis `fma(acc, rscale, tv)` (`llvq_pers.cuh:325-331`,
  sites 2 et 3 de `:14-16`). 4E : la row scale est pliée dans les poids des blocs, la queue entre comme colonnes
  binary16 à part (`pers_4et.cu:23-25`).

Donc, sur les trois décodeurs : la row scale multiplie les blocs seulement, la queue n'est jamais mise à l'échelle,
et la queue vit en base tournée. La rotation porte sur tout `d_in`, queue comprise : `Rotation::new(m.d_in, seed)`
(`format.rs:1600`), `Q = (Q_odd ⊗ H_m) D` (`rotation.rs:26, :50-66`). Le chemin fusé refuse une matrice sans
rotation (`fused.rs:1730-1732`), donc les trois fichiers servis sont tournés.

### 2. Le binaire `rowscale`

Il lit `sigma.json` (`rowscale.rs:75-116`), accepte `row_scales` et `row_norms`, refuse `free_params` par nom. Il
multiplie `m.row_scales[i]` par σ, en sautant 1,0 (`:183-191`). Rien d'autre ne bouge : ni la queue, ni la rotation.
Aucun commentaire du fichier ne parle de la queue ni de la base. Sa doc dit « multiplying `row_scales[i]` by `s`
multiplies row `i` and only row `i` » (`:10-11`) : vrai pour l'indépendance des lignes, muet sur la queue. Ses tests
(`:405-461`) vérifient l'octet modifié, pas la fonction décodée. La fixture a pourtant une rotation et une queue de 8
(`:365-385`).

### 3. Côté Python

L'export est celui de `bin/export` : `decode_matrix`, donc **base naturelle**, puis f16 (`export.rs:54, :136`).
`train.sh:80-84` le charge en bf16 (`--dtype bf16`), `__main__.py:92-94`. `RoutedLinear` copie le poids en tampon gelé
(`torch_model.py:58-60`). Sans `--commute` (jamais passé par `train.sh`), le chemin est `weight`
(`torch_model.py:79-80`).

`RowScales.weight` multiplie les `lattice = d_in − d_in % 24` premières colonnes **naturelles** par σ et recolle les
`t` dernières colonnes naturelles telles quelles (`row_scales.py:78-81`, `lattice` de `torch_model.py:224-227`). Le
chemin `linear` calcule la même fonction (`:107-112`). Aucune rotation n'existe dans le modèle Python : `grep` ne
trouve ni `Rotation` ni signes dans `ops/llvqtune`.

### 4. Les deux fonctions, en formules

Notation, pour une ligne `i` d'une matrice de largeur `d = 24n + t`, `L = 24n` :

```text
ŵ ∈ R^d        la ligne décodée, base naturelle (ce qu'écrit l'export, avant f16)
Q              la rotation de rotation.rs ; r = Q ŵ = [b ; τ]
b ∈ R^L        les blocs décodés, proportionnels à row_scales[i]
τ ∈ R^t        la queue stockée (base tournée)
P = diag(0_L, I_t)   projecteur sur les t dernières coordonnées
décodeur       ŵ = Qᵀ [b ; τ]

fichier plié   w_file(σ)  = Qᵀ [σ b ; τ]               = σ ŵ + (1 − σ) Qᵀ P Q ŵ
entraînement   w_train(σ) = [σ ŵ_1..L ; ŵ_L+1..d]       = σ ŵ + (1 − σ) P ŵ

écart          w_file − w_train = (1 − σ) (QᵀPQ − P) ŵ
                                = (σ − 1) ([0 ‖ ŵ_queue_nat] − Qᵀ[0 ; τ])
```

C'est exactement la formule du constat, avec `T_i = Qᵀ[0 ; τ]`. Le terme dominant `σ ŵ` est commun aux deux.

Les deux coïncident si et seulement si `σ = 1`, ou `t = 0`, ou `Q = I` (fichier non tourné), ou plus généralement si
`Q` laisse invariant le sous-espace des `t` dernières coordonnées. Ce n'est jamais le cas ici. Sur toutes les
largeurs servies `t < 24 ≤ m` (`m` = 512, 4 096, 512, 1 024, 1 024 pour 2 560, 4 096, 9 728, 5 120, 17 408), donc la
queue tombe dans le dernier groupe Walsh-Hadamard, que `H_m` mélange en entier, et `Q_odd` mélange les `k` groupes
entre eux. Réponse aux deux questions : oui pour `t = 0` ; oui si la queue n'était pas tournée, mais elle l'est.

*Computed*, vérification de la formule : réimplantation en Python pur de la construction (signes, `wht` par groupe,
`Q_odd` par Gram-Schmidt), 20 lignes gaussiennes, σ = 1,03. La forme fermée colle au calcul direct à 2e-14 près.
Rapport `‖w_file − w_train‖ / ‖w_train − ŵ‖` moyen : 0,1122 à 2 560, 0,0863 à 4 096, 0,0365 à 9 728, contre
`√(2t/d)` = 0,1118, 0,0884, 0,0406.

### 5. Ordre de grandeur

```text
‖écart‖² = (1 − σ)² (‖τ‖² + ‖P ŵ‖² − 2 ⟨Qᵀ[0;τ], [0;ŵ_queue_nat]⟩)
```

Si l'énergie est isotrope dans les deux bases, `‖τ‖² ≈ ‖Pŵ‖² ≈ (t/d)‖ŵ‖²`, le terme croisé est négligeable, et
l'écart vaut `|σ − 1| √(2t/d) ‖ŵ‖` pour une édition voulue de `|σ − 1| ‖ŵ‖`.

| modèle | matrices | d | t | t/d | √(2t/d) |
|---|---|---|---|---|---|
| 4B | q, k, gate, up | 2 560 | 16 | 0,63 % | 11,2 % |
| 4B | o | 4 096 | 16 | 0,39 % | 8,8 % |
| 4B | down | 9 728 | 8 | 0,08 % | 4,1 % |
| 8B | q, k, o, gate, up | 4 096 | 16 | 0,39 % | 8,8 % |
| 8B | down | 12 288 | 0 | 0 | 0, exact |
| 14B | q, k, o, gate, up | 5 120 | 8 | 0,16 % | 5,6 % |
| 14B | down | 17 408 | 8 | 0,05 % | 3,0 % |

(`v_proj`, et au 4B les `o_proj` et `down_proj` int4, ne sont ni entraînés ni pliés.)

Ce que vaut « 11 % » : la norme de l'écart rapportée à la norme de l'édition voulue, sur une ligne, en espace des
poids. Autrement dit, la direction entraînée et la direction pliée font un angle d'environ `√(2t/d)` radian, 6,4° à
2 560, cosinus `1 − t/d` = 0,994. Rapporté à la ligne : `|σ − 1| × 0,112`, soit 0,34 % à l'écart type de σ (0,0303,
`dclm-rowscales-2026-09-20.txt:79`), et 7 % sur la ligne extrême σ = 0,385. **L'ordre de grandeur du relecteur tient.**
Une phrase est fausse : « toutes leurs matrices ont une queue ». Au 8B, les 36 `down_proj` (12 288 = 24 × 512) n'en
ont pas, et σ y est exact.

Deux raisons pour lesquelles l'estimation isotrope peut sous-estimer, toutes deux mesurables à 0 $ :
- la queue absorbe la compensation GPTQ de tous les blocs précédents (`gptq.rs:55-59`), et deux encodages de la même
  matrice ont des queues qui diffèrent de 42 à 56 % de leur amplitude (`HISTORIQUE.md:392`) : `‖τ‖²` peut dépasser
  `(t/d)‖ŵ‖²` ;
- un canal d'entrée aberrant parmi les `t` dernières colonnes naturelles gonflerait `‖Pŵ‖` et surtout l'écart vu
  en sortie.

**Ce qui compte pour la perte n'est pas cette norme.** Au σ* entraîné, la dérivée de la perte du fichier vaut
`∂F/∂σ_i = −G_i · (QᵀPQ − P) ŵ_i`, `G_i` le gradient de la ligne. Si `G_i` voit les coordonnées de la ligne de façon
isotrope, les parts systématiques `G_i·Pŵ_i` et `G_i·QᵀPQŵ_i` valent chacune environ `(t/d) G_i·ŵ_i` et s'annulent
au premier ordre. Le reste n'a pas de signe privilégié d'une ligne à l'autre. La perte d'optimalité est du second
ordre dans l'écart relatif ε, de `t/d` à `√(2t/d)` selon le cas : au plus `ε² ≈ 1,2 %` du gain au 4B, soit environ
0,04 point sur +3,15 (*estimé*, modèle quadratique). Même la lecture linéaire pessimiste, où l'écart retirerait au
gain une part égale à sa norme relative, donne 0,35 / 0,29 / 0,09 point au 4B / 8B / 14B : sous le bruit de 0,43
dans les trois cas (*estimé*).

Le chiffre de E8 est faux. E8 écrit « differ by at most 1.2e-4 in relative terms »
(`preregistration-tetranu-rowscales-2026-09-19-ECARTS.md:119`), soit `0,02 × t/d`. C'est le produit d'une amplitude
par une fraction d'énergie. En amplitude, l'écart relatif d'une ligne vaut `0,02 × √(2t/d)` = 2,2e-3 à 2 560 : 18 fois
plus. Et ce n'est pas une borne, puisque `|σ − 1|` monte à 0,62. Par chance, `t/d` est le bon ordre pour la part
systématique de l'effet sur le gradient, donc la conclusion de E8 (« the dominant term is identical ») reste
probablement juste, pour une autre raison que celle qu'il donne.

### 6. Ce qui réfute, ce qui était déjà su

- **Le défaut est connu, écrit et assumé.** Déviation E8, « What was trained is not exactly what is folded back »,
  trouvée par l'audit du 2026-09-19 (`preregistration-tetranu-rowscales-2026-09-19-ECARTS.md:102-122`). Reprise dans
  le prereg DCLM 4B, section « Known defects carried into this arm » (`preregistration-dclm-rowscales-2026-09-19.md:66-67`),
  dans le journal (`dclm-rowscales-2026-09-20.txt:118-120`, « stated, not repaired ») et dans le prereg mmlu-aux
  (`preregistration-mmluaux-4b-rowscales-2026-09-23.md:101-103`). `rhoapply.rs:37-54` avait déjà écrit, le 2026-09-07,
  la forme affine juste `r_i(ρ) = ρ·B_i + T_i`. R2-API-01 n'est donc pas une découverte. Il n'était pas non plus
  interdit par `BRIEF-R2.md`, qui ne liste que les fichiers d'audit.
- **Ce qu'E8 n'a pas fait.** Il n'est repris ni dans les preregs 8B et 14B ni dans leurs journaux (`grep` sur
  « natural », « E8 », « basis » : rien), alors que ces bras portent le même défaut. Les docstrings disent le
  contraire : `torch_model.py:7-13` (« exactly `centroids[g] * row_scale * u` on the coded columns, and the verbatim
  tail after them » ; « A per-row multiplier on the exported tensor's coded columns is a per-row multiplier on
  `row_scales` »), `README.md:73-74` (même phrase), et `row_scales.py:15-18`, qui justifie l'exclusion de la queue
  par l'argument même que la découpe actuelle viole dans l'autre base.
- **Le test cité ne peut pas voir l'écart.** `test_the_fold_is_the_trained_function` (`test_row_norms.py:86-112`)
  plie σ en Python avec la paramétrisation de l'entraînement, `weight[:, :cols].mul_(σ)` (`:107-111`), sur un modèle
  sans rotation (`:15-34`). Il ne passe ni par `rowscale` ni par `decode_matrix`. Pour σ il est tautologique ; pour τ
  il est juste, car les normes sont appliquées en base naturelle avant la rotation des deux côtés.
  `test_linear_agrees_with_building_the_weight` (`test_trainables.py:112-123`) compare deux formes Python entre elles.
- **Aucun journal ne compare l'entraîné au plié.** Les journaux donnent la KL d'entraînement (fonction entraînée,
  lots d'entraînement) et la ppl et le MMLU du fichier plié. Personne n'a évalué la fonction entraînée hors de la
  boucle. Rien ne confirme ni ne réfute l'effet ; le prereg dit d'ailleurs que la courbe n'est pas la mesure
  (`preregistration-dclm-rowscales-2026-09-19.md:69-70`).

### (a) Ce que cela change pour les fichiers publiés

Rien de mesuré. MMLU 61,11 / 68,16 / 74,20, gains +3,15 / +3,29 / +1,67, ppl 12,3268 / 9,5725 / 8,4622, GSM8K et
débits sont des mesures du fichier plié : reconstruction dense par `decode_matrix`, ou noyau, qui calculent tous deux
`w_file(σ)`. Ils restent vrais. Ce qui est inexact : σ n'est pas exactement l'optimum de la perte pour le fichier, et
la KL d'entraînement décrit `w_train`. `paper2/sections/model.tex:35-41` dit « We retrain only these scales… The
trained scales replace the old ones in place » : vrai tel quel.

### (b) La plus petite expérience qui chiffrerait le gain possible

Le MMLU ne peut pas le voir : effet attendu de 0,04 point, au plus 0,35 dans la lecture pessimiste, bruit 0,43. Il faut
une mesure plus fine d'abord, à 0 $, avec un go (règle 1, et « aucun calcul local sans go »).

1. **M0, 0 $, Mac, CPU, quelques minutes, aucun passage avant.** Sur `~/qwen3-4b-dclm.bin` et
   `~/qwen3-4b-dclm-ft.bin` avec `~/dclm-sigma.json` : par matrice, `‖τ‖²/‖ŵ‖²`, `‖Pŵ‖²/‖ŵ‖²`, et
   `‖w_file − w_train‖ / ‖w_train − ŵ‖` avec `w_file` décodé du fichier plié et `w_train` la paramétrisation Python
   appliquée au fichier de base. Mesure le « 11 % » réel, GPTQ compris.
2. **M1, 0 $, Mac, de l'ordre d'une heure.** KL contre le professeur dense sur des lots DCLM tenus à part (par
   exemple 32 lots de 2 × 1 024, même graine pour les trois), pour trois élèves : export de base ; export de base + σ
   par `RowScales` (la fonction entraînée) ; export du fichier plié. Écart = KL(plié) − KL(entraîné), gain = KL(base)
   − KL(entraîné). Critère : |écart| < 10 % du gain, on n'entraîne rien et on corrige le code pour l'avenir ;
   ≥ 10 %, on passe à M2.
3. **M2, seulement après M1, prereg tamponné.** Réentraîner le 4B avec la correction, même graine, même plan (9 507
   pas) : environ 4,00 $ d'entraînement (1,71 h sur l40sx1) et 0,72 $ de MMLU d'après `dclm-rowscales-2026-09-20.txt:3-4`,
   soit environ 4,72 $. Lecture primaire : la KL de M1 sur le nouveau fichier plié contre l'ancien. MMLU apparié
   ensuite, succès seulement à +0,43 point ou plus. Le prereg doit écrire d'avance que l'issue attendue est « non
   séparé ». Le 8B (chaîne 10,19 $) et le 14B (32,17 $) ne valent pas la dépense sur cette base.

### (c) La plus petite correction

**Côté Python, et seulement là.** Le côté Rust ne peut pas plier exactement : `w_train(σ)` n'est pas représentable
par le fichier, car `Q P ŵ` s'étale sur toutes les coordonnées de blocs et un scalaire par ligne ne l'absorbe pas. Un
pli « projeté » réécrirait les queues et laisserait un résidu d'environ `|1 − σ| √(t/d) ‖ŵ‖`.

La correction du relecteur est juste, et je la confirme terme à terme :
- `bin/export` écrit à côté de `llvq-int4.json` un `llvq-tails.safetensors` : pour chaque enregistrement treillis
  tourné, `<nom>.tail` `[d_out, t]` (la queue stockée), et par rotation `(d_in, seed)` un
  `rot.<d_in>_<seed>.tail_rows` `[t, d_in]`, les lignes `L..d` de `Q`, obtenues par `apply_transpose(e_j)` puisque
  `Qᵀ e_j` est la ligne `j` de `Q`. Plus une table nom → clé. Environ 68 Mo au 4B (16 957 440 valeurs de queue,
  `free_params.py:5`), quelques centaines de Ko de `tail_rows`.
- `RowScales` : `weight = σ ⊙ frozen + (1 − σ) ⊙ (tail @ tail_rows)`, sans découpe par colonnes ;
  `linear = σ ⊙ F.linear(x, frozen) + (1 − σ) ⊙ F.linear(F.linear(x, tail_rows), tail)`. Pour un fichier non tourné,
  `tail_rows = [0 | I_t]` redonne exactement la découpe actuelle.
- `main` refuse (sortie 2) un export de fichier tourné sans `llvq-tails.safetensors`, en nommant E8.
- Test Python en f64 : `Q` orthogonale de synthèse (QR d'une gaussienne 53 × 53, `t = 5`), `ŵ = Qᵀ[b ; τ]`,
  `weight(ŵ) == Qᵀ[σb ; τ]` à 1e-12. Mutant : remettre la découpe ; l'écart vaut `|1 − σ| √(2t/d)`. Plus le test Rust
  croisé du relecteur sur la fixture de `rowscale.rs` (rotation `0xABCD`, queue de 8).

Ni le format `.llvq`, ni l'index, ni un noyau, ni un token servi ne bougent. **Aucune des deux corrections ne change
un fichier publié** : seuls un réentraînement et un nouveau pli en produiraient de nouveaux, et le périmètre cité est
gelé. Décision d'opérateur.

Sans attendre, effort S, risque nul : corriger `torch_model.py:5-13`, `README.md:71-76`, `row_scales.py:15-18` pour
dire E8 et sa vraie taille ; écrire à la suite de E8 une note datée qui remplace 1,2e-4 par « environ 2e-3 par ligne
à |σ − 1| = 0,02, soit environ 11 % de l'édition, pas une borne » ; ajouter une ligne E8 aux journaux 8B et 14B.

`free_params` : là, l'erreur n'est pas de 11 %, elle est totale. Ses « queues » sont les colonnes naturelles
(`__main__.py:115-118`) et son export dit `tail := tail` (`free_params.py:79`), ce qui écrirait des valeurs naturelles
dans un champ tourné. Latent tant que `rowscale` refuse ce mode (`rowscale.rs:81-85`) ; le refus de R2-API-05 le
couvre.

### Gravité proposée

Mineure, pas majeure. Rien ne plante, aucun chiffre publié n'est faux, la mesure interne touchée (la KL
d'entraînement) est déjà déclarée « not the measurement », et l'effet estimé sur l'optimum est sous le bruit. Ce qui
justifie de le garder : le chiffre faux de E8, son absence aux bras 8B et 14B, trois docstrings fausses, un test qui
ne peut pas le voir.

## R2-API-02

### La chaîne, ligne par ligne

- `train.sh:84` passe `--dtype bf16` ; `__main__.py:86-94` charge l'élève en bf16 ; `torch_model.py:58-60` garde le
  poids en tampon bf16.
- `__main__.py:126` appelle `build(args.mode, shapes=…, device=…)` sans `dtype` : σ est une feuille f32
  (`row_scales.py:36, :49-52`). Même chose pour τ (`row_norms.py:56, :67`).
- `row_scales.py:78` : `sigma.unsqueeze(1).to(frozen.dtype)`. σ est arrondi en bf16 **avant** le produit, puis le
  produit est en bf16. Chemin `linear` : `.to(x.dtype)` (`:108`). τ : `tau.to(y.dtype)` (`row_norms.py:98`), et la
  sortie d'une RMSNorm de Qwen3 est au type de l'entrée.
- Aucun `autocast` dans `ops/llvqtune` (`grep`).
- L'optimiseur est un AdamW sur les paramètres f32 (`torch_optimizer.py:14-16`) : maître et état en f32.
- Le gradient : la rétropropagation du `.to()` rend un gradient f32 à σ. Sa valeur vient de la rétropropagation du
  produit bf16 (somme sur les colonnes de `grad · frozen`, résultat bf16). Il arrive donc en f32, avec la précision du
  bf16, environ trois chiffres. Sans conséquence pour Adam, qui normalise.
- L'export écrit le maître f32 (`row_scales.py:121-130`), lu sans perte en f64 par `rowscale`. Le fichier reçoit
  σ_f32 ; la perte a vu `bf16(σ_f32)`.

Le constat est exact. Pas bf16 : `2^-8 = 0,0039` sur [0,5 ; 1), `2^-7 = 0,0078` sur [1 ; 2). Au pic de `LR=3e-4`
(`train.sh:37`), il faut 13 et 26 pas de même signe pour franchir un pas : « une vingtaine » est juste. C'est le schéma
standard de l'entraînement en précision mixte, équivalent à un estimateur droit sur l'arrondi.

### Ce qui est exagéré

« La résolution est du même ordre que l'effet entraîné. » Mesuré : sd(σ) = 0,0303 au 4B
(`dclm-rowscales-2026-09-20.txt:79`), 0,0270 au 8B (`dclm-8b-rowscales-2026-09-21.txt:75`). Le pas vaut un huitième à
un quart de l'écart type. L'écart entre σ exporté et σ vu est uniforme sur ± un demi-pas : 0,11 % à 0,23 % en moyenne
quadratique, soit 4 à 7 % de l'écart type de σ − 1. Une erreur aléatoire de cette taille coûte au second ordre,
environ 0,2 à 0,6 % du gain (*estimé*). Gravité mineure : d'accord.

### La correction

Juste dans son principe : multiplier en précision haute puis arrondir une seule fois. Une réserve sur le chemin
`weight`, celui que `train.sh` utilise. `(frozen.float() * σ).to(bf16)` fait garder par autograd la copie f32 de
`frozen` pour le gradient de σ : 3,52 G valeurs (`row_scales.py:88-90`), 14,1 Go au 4B (*computed*), sur un 14B que
son job dit ne tenir que sur H200. Forme sans surcoût et équivalente à l'arrondi près :
`frozen + frozen * (sigma - 1).unsqueeze(1).to(frozen.dtype)`. `σ − 1` en bf16 garde une précision relative de
`2^-8` sur l'écart (environ 1e-4 à σ = 1,03), et l'arrondi final se fait poids par poids : la sortie d'une ligne
répond à σ en moyenne sous le pas. Chemin `linear` : `y + y * (sigma - 1).to(y.dtype)`. Même chose pour τ. Le test
proposé tient (σ = 1,001 s'arrondit à 1,0 en bf16 aujourd'hui, donc `torch.equal` est vrai ; après correction,
environ 13 à 26 % des poids de la ligne changent de valeur). Change ce que produirait un réentraînement : opérateur.

## Vu en passant, hors mission

- Troisième écart entre la fonction entraînée et le fichier, non relevé : l'export est en f16 (`export.rs:136`) et
  l'élève est chargé en bf16 (`train.sh:84`, `__main__.py:92-94`). Chaque poids est réarrondi à `2^-8` relatif près,
  environ 0,1 à 0,2 % en moyenne quadratique, indépendamment de σ. C'est du même ordre, en espace des poids, que
  l'écart de base à σ typique (0,34 %) et que l'écart d'export de R2-API-02. Aléatoire, il ne biaise pas σ. À écrire
  dans la docstring, pas à corriger.
- `row_norms.py:33-37` dit « Write-back: Not done… Folding `tau` into the carried f16 norms is a raw-tensor rewrite
  still to be written ». `rowscale.rs:31-39, :215-283` le fait. Cosmétique.
