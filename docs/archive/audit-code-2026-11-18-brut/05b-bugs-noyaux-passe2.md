# Bugs, noyaux GPU, seconde passe : constats
Périmètre réellement lu : en entier `llvq-cuda/kernels/pers_4et.cu`, `pers_4et_a12.cu`, `pers_4et_a12g.cu` (relu pour les appels), `pers_4et_a12g32.cu`, `pers_4et_a10.cu`, `pers_attn_full.cuh`, `llvq_rot.cuh`, `rotate_wide.cu`, `llvq_tetra48.cuh`, `llvq-llm/kernels/tv_batch_gemm.cu`, `tv_batch_mma.cu`, `tv_q4_wide.cu`, `llvq_rot.metal`, `emb_q4.metal`, `llvq-llm/src/batch.rs`, `batch_cuda.rs`. Le chemin servi de `pers_4et_a10rc.cu` (1-182 : `p4t_store_rc`, `p4t_ip_*`, `p4t_unit_rc`, `p4t_phase_rc`), de `pers_4et_a9.cu` (1-120 : `n_bar_a9`, `p4t_fold_a9`), de `pers_4et_stream.cu` (1-145 : `p4t_stream_cap`, `p4t_stream_stage`, `p4t_decode_stream`). `llvq_pers.cuh` 60-220, 212-1011 (sections 1 à 3, 3 à 7c, début de 8), 1490-1740 (constantes et `pers_attn_tile` de la section 10). `llvq_f1rank_v3.cuh` 85-220, `llvq_f1rank.cuh` 186-194 (`f1r_load`). Survolés : `persist4e_cuda.rs` 1609-2490 (`install_tree`, `install_tiled`, `tree_chain`, `chain_prefill`, `tree_launch`, `prepare`), `pers.rs` 2228-2545 et 2671-3098 (hôte des bras de l'arbre), `spec_loop.rs` 277-350, `served_decode.rs` 380-500, `served.rs` 170-230 et 400-415, `model.rs` 2360-2540 et 3180-3321, `engine.rs` 547-620 et 768-1215, `fused_metal.rs` 60-300, 980-1040, 1170-1370, 1620-1840, 2280-2305, `gpu.rs` 780-950, `attn_dec_batch.cu` (offsets), `llvq-metal/tests/rot_matches_host.rs` 425-600.
Non couvert : les corps non servis des fichiers d'arbre (`pers_4et_rc_body` après la ligne 182 de `pers_4et_a10rc.cu`, `pers_4et_a9_body`, `pers_4et_stream` au-delà de la ligne 145, `pers_4et_a11*.cu`, `a4` à `a8`) ; `rotate.cu` et la répartition des rotations dans `fused_cuda.rs` (première passe) ; `llvq_tetra48.metal` et `tv_q4_h.metal` ; la répartition des noyaux Metal par le paquet Python `llvq-tetra` (`mps.py`).
Bilan : 0 critique, 1 majeure, 0 mineure, 2 cosmétiques

Résumé. Sur les six parties demandées, un seul défaut réel : l'hôte Metal lance la rotation servie à 1 024 threads par groupe sans lire la limite du pipeline, alors que le banc de la même rotation la lit et explique pourquoi. Le reste est contrôlé sans constat, avec les valeurs calculées en fin de fichier : la passe d'arbre `a12ipg32ft` du 4B, le GEMM du lot, la rotation large, les décodeurs Tetra et l'embedding int4 Metal.

### BUG-70 La rotation Metal servie se lance à 1 024 threads par groupe sans lire la limite du pipeline
- Gravité : majeure
- Confiance : PLAUSIBLE (dépend de la puce et du compilateur Metal ; le M3 Max du dépôt accepte 1 024, le chemin y est servi et gardé)
- Où : `llvq-llm/src/fused_metal.rs:1208`, `:1305-1309`, `:1343-1347` ; à comparer à `fused_metal.rs:996` ; `llvq-llm/kernels/llvq_rot.metal:255-256`, `:326`, `:331`, `:344` ; `llvq-metal/tests/rot_matches_host.rs:432-438`
- Fait : `upload_rotation` fixe `threads = n.next_power_of_two().clamp(32, 1024)` (1208). Pour toutes les largeurs servies (2 560, 4 096, 9 728 au 4B ; 4 096, 12 288 au 8B ; 5 120, 17 408 au 14B) cela vaut 1 024. `RotOp::metal_fwd` lance `rot_apply_metal` par `dispatch_threads(threads, threads)` et `rot_apply_rows_metal` par `dispatch_threads(threads * rows, threads)` (1343-1347), sans comparer `threads` à `pipe.max_total_threads_per_threadgroup()`. Le noyau `_lutdt` du même fichier fait cette comparaison (996). Le test de la rotation la fait aussi, et son commentaire décrit le danger : « asking for more threads than the pipeline takes would silently launch a SECOND threadgroup on the same scratch » (`rot_matches_host.rs:433-437`, puis `r.nthreads.min(kern.max_threads_per_group())` à 438). Le noyau, lui, déclare que n'importe quel nombre de threads donne les mêmes bits (`llvq_rot.metal:255-256`), et `the_work_split_does_not_move_a_bit` le vérifie de 7 à 1 024 threads.
- Scénario : une puce Apple, ou une version du compilateur Metal, pour laquelle le pipeline de `rot_apply_metal` ou `rot_apply_rows_metal` annonce moins de 1 024 threads par groupe, par exemple sous la pression des registres de `rot_mix` (32 flottants de `col`, boucle déroulée). Un `dispatch_threads` dont le groupe dépasse la limite est hors contrat Metal : la couche de validation l'arrête ; sans elle le résultat dépend du pilote. Si le pilote ramène le groupe à la limite, `rot_apply_metal` tourne sur deux groupes qui écrivent le même `scratch` (course, rotation fausse, jetons faux sans erreur), et `rot_apply_rows_metal` reçoit `2 x rows` groupes : les groupes `r >= rows` lisent `xin` au-delà de la ligne et écrivent `scratch` et `xout` hors de leur allocation (`llvq_rot.metal:331`, `:344`). 144 lancements de rotation par jeton au 4B passent par là.
- Touche : chemin servi (Metal 4B, 8B, 14B)
- Correction proposée : dans `RotOp::metal_fwd`, prendre `threads = min(self.rot.threads, pipe.max_total_threads_per_threadgroup())` arrondi au multiple de 32 inférieur, refuser 0, et l'utiliser pour la grille et le groupe, comme à la ligne 996. Les bits ne bougent pas : le noyau ne dépend pas du nombre de threads.
- Effort : S
- Risque de la correction : faible. Hôte Metal seulement. Aucun format, aucun index, aucun fichier de noyau épinglé. Les jetons servis ne bougent pas (indépendance au nombre de threads, testée de 7 à 1 024).
- Vérification proposée : écrire dans le journal de chargement Metal la limite que le pipeline annonce pour `rot_apply_metal` et `rot_apply_rows_metal` sur le Mac. Puis un mutant qui force `threads = 2048` sous `MTL_DEBUG_LAYER=1`, qui doit être refusé par la validation aujourd'hui et ramené à la limite après la correction.

### BUG-71 Deux noyaux de la passe d'arbre se disent hors service alors qu'ils servent le préremplissage du 4B
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-cuda/kernels/pers_attn_full.cuh:2-6` ; `llvq-cuda/kernels/pers_4et.cu:56-60` ; à comparer à `llvq-llm/src/served.rs:408`, `llvq-llm/src/served_decode.rs:456`, `llvq-cuda/src/pers.rs:3076-3083`
- Fait : `pers_attn_full.cuh` dit « Never part of a served module », et `pers_4et.cu` dit « Not in any served module ». Le bras servi `CHAIN_ARM = "a12ipg32ft@8.10.8.20.6"` (`served.rs:408`) s'installe par `install_chain` (`served_decode.rs:456`). Son module est `pers_ft_apply(p4t_a12ipg32_unit_text)` (`pers.rs:3076-3083`) : il contient `pers_4et.cu` en entier, et `pers_attn_full.cuh` inséré dans `pers_attn_tile`. C'est le défaut du 4B sous `llvq serve` depuis le 2026-11-15.
- Scénario : un lecteur croit ces corps hors service et traite leurs chemins comme des bancs ; c'est l'erreur que la première passe relève pour `persist4e_cuda.rs` (BUG-52).
- Touche : doc
- Correction proposée : réécrire les deux en-têtes au prochain déplacement volontaire de l'empreinte du module de chaîne, pas avant.
- Effort : S
- Risque de la correction : fort. Les deux fichiers entrent dans le texte du module servi `a12ipg32ft`, dont le sha256 est épinglé (`pers.rs:4751`, `19b734a3…`) et sert de clé au cache de modules compilés. Un octet change l'empreinte, rouvre la porte du bras servi et invalide le cache.
- Vérification proposée : aucune, lecture suffisante.

### BUG-72 La rotation Metal servie répète une factorisation fausse de 25 600 et une limite qui n'existe pas
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/kernels/llvq_rot.metal:97-99` ; copie octet pour octet `llvq-tetra/llvq_tetra/data/llvq_rot.metal`, hachée dans `llvq-tetra/llvq_tetra/data/tetra-tables.json:79` ; même texte dans `llvq-cuda/kernels/llvq_rot.cuh:46-48`, corrigé par `rotate_wide.cu:29-32`
- Fait : le commentaire dit que 25 600 vaut 512 x 50 et « does NOT fit. The host refuses it ». 25 600 vaut 1 024 x 25, et k = 25 tient sous `LLVQ_ROT_KMAX = 32`. Sur Metal la rotation passe par la mémoire du périphérique : aucune limite de mémoire partagée ne bloque cette largeur. La liste « k = 1, 3, 5, 19 » oublie aussi le 14B (k = 5, 17), servi sur Metal. `rotate_wide.cu` documente l'erreur pour la copie CUDA, laissée en place parce qu'elle est dans l'unité servie ; la copie Metal n'a pas cette raison.
- Scénario : un lecteur qui prépare le 32B sur Metal croit la rotation de `down_proj` impossible, alors que seule la mémoire partagée du CUDA la bloque.
- Touche : doc
- Correction proposée : corriger le commentaire Metal (1 024 x 25, k = 25 tient, la limite est la mémoire partagée du CUDA) et régénérer la copie de la roue avec `bin/tetratables`. Laisser `llvq_rot.cuh`.
- Effort : S
- Risque de la correction : faible. Aucun bit calculé ne bouge. La copie de `llvq-tetra` et son sha256 doivent suivre, comme pour BUG-53.
- Vérification proposée : aucune, lecture suffisante.

## Contrôlé sans constat

Valeurs du 4B servi : hidden 2 560, inter 9 728, 32 têtes, 8 têtes KV, groups 4, 36 couches, vocab 151 936. 8B : 4 096, 12 288, groups 4. 14B : 5 120, 17 408, 40 têtes, groups 5. Tous les nombres ci-dessous sont *calculés*.

### 1. Passe d'arbre servie `a12ipg32ft` (chaîne du 4B)
- Chemin exécuté, établi depuis `install_chain` (`served_decode.rs:456`) et `tree_launch` (`persist4e_cuda.rs:2390-2393`) : `pers_4et_a12ipg32_f*_t384` = `pers_4et_a12g_body<FORM, 4, true>`. Il appelle `p4t_phase_rc<4, true>` (lecture en place `p4t_ip_load`), `p4t_fold_a9`, `p4t_fold_gu_a10`, les prologues `p4t_ln_rot_a10`, `p4t_dn_in_a10` et `p4t_ln_final_a10` sur l'aire v7, `p4g_attention`, puis `pers_attn_tile` et son entrée `pers_attn_tile_full`. Les arguments sont ceux de 4E (`&self.args`).
- Mémoire partagée : aire `p4g_area_floats(12, 32, 9 728, 4)` = 18 624 flottants (74 496 B), soit 93 568 B avec l'en-tête et les tables, sous 101 376. Le noyau refuse une aire plus petite (`pers_4et_a12g.cu:261-262`), avec la même formule que l'hôte (`persist4e_cuda.rs:1660`, `:1940`).
- Disposition de l'aire : l'attention groupée occupe [0, 18 560) flottants (74 240 B) ; l'aire v7 occupe [16 720, 18 560) (`p4t_v7_a10`, besoin max(2H, I) = 9 728 ≤ 16 720) ; les maxima `cmax` occupent [18 560, 18 624). L'attention ne touche jamais `cmax`. Elle recouvre l'aire v7, mais v7 est réécrite à chaque attente « gap » juste avant son seul lecteur : ln2 après le pli de o, l'entrée du down après le pli de gate/up, ln1 de la couche suivante après le pli du down, signes nuls avant la norme finale. Entre deux, seules `p4t_taps_a12` (globale, inactive à `pad` 0) et les plis (mémoire globale) s'exécutent. Les prologues utilisent [0, 2H) = [0, 5 120) et [0, I) = [0, 9 728).
- Plafond de tranche : `p4t_stream_cap(18 624, 12)` = 17 sous-pas. Le plan servi [8, 10, 8, 20, 6] donne au plus 7, 9, 7, 11 et 9 sous-pas. L'unité refuse un dépassement par le drapeau d'abandon, uniformément sur la CTA (`pers_4et_a10rc.cu:105-109`).
- Unité de flux : `XS = 24 ns + 4` (multiple de 4, rangées alignées sur 16 octets) ; lecture B maximale `24 ns - 1 < XS`. Les mots en place : Tetra `3 sa + h` et `+ 1`, int4 `6 sa + 3 h + k`, nuls au-delà de la rangée, à l'index tuilé v9 pour v, o et down, dans l'ordre du fichier pour la tête : les mêmes conventions qu'à l'installation (`persist4e_cuda.rs:2036-2047`), vérifiées mot à mot par `p4t_ip_check`. Bloc de queue : 2 560 = 106 x 24 + 16, sous-pas 54, blocs 0 à 107 : 106 queue, 107 nul. Colonnes d'activation au-delà de d_in mises à zéro, pièces de 8 demi-mots sans chevauchement (d_in multiple de 8). Lignes `ldx` = 9 728 demi-mots, alignées.
- Plis et sorties : `fpart` dimensionné par la phase la plus large à 32 rangées ; la tête domine avec 6 x 32 x 151 936 = 29 171 712 flottants, indices en 64 bits. Le pli de la tête compte 37 984 x 32 = 1 215 488 paires, en u32. Les offsets `p4t_place` sont multiples de 4 flottants pour tout R (d_out multiple de 4), donc le chemin vectoriel `float4` reste aligné. Le pli gate/up lit `sg.S` pour les deux projections, qui partagent `split_gu`.
- Attention groupée : un partage n'existe que pour `s < pos0 / 256`, et toute rangée y a `l - base > 256`, d'où nu ≥ 5, tpu 1, 8 tuiles, 8 pas, positions toutes dans le cache. L'énumération des tâches par rangée trouve toujours `nj ≥ n_kv`. Tickets `L.tickets + r n_kv` (32 x 8 alloués), états de split `part_stride` = 8 x 34 x 4 x 130 = 141 440 flottants par rangée. Barrières de la boucle de rangées et réutilisation de `cmd` (= `stage`) ordonnées par les `__syncthreads` des lignes 193-233. Slots de l'arbre `tk_l + slot n_kv 128 + h 128`, comme les écrit la phase RoPE.
- `pers_attn_tile_full` est bit pour bit `pers_attn_tile` à nv = 8. Mêmes chaînes fma par score, même papillon 8, 4, 2, 1, même max et même somme de 0 à 7, même chaîne PV. `pers_ex2_sel` égale `pers_ex2` sur NaN (rend x), sur x ≤ -126 et -inf (rend 0), et sur x > -126, +inf compris (mêmes opérations sur x).
- Hôte : la disposition de `P4tLaunch` (14 u32, puis les pointeurs à l'offset 56) est celle du noyau ; `n_bar_a9` = 4 + 13 x 36 = 472 attentes, compté dans le corps ; `split_head` 0 sur les lancements sans tête ; `pad` 0 hors taps ; l'anneau d'un lancement sans tête lit `u32::MAX`, comme documenté. `chain_launches` : `mv` = 0..n et `move_pos0 + mv.len() = pos0` à chaque lancement.

### 2. GEMM du lot (`tv_batch_gemm.cu`, servi) et MMA (`tv_batch_mma.cu`, compilé, non servi)
- Mémoire partagée des entrées servies (B, KC) : (8, 192) 47 360 B, (16, 192) 53 760 B, (32, 192) 66 560 B, (64, 96) 55 296 B, (128, 96) 81 920 B. Toutes sous 101 376, opt-in posé par `func_dynamic_shared` ; `bg_smem_bytes` reproduit `BG_SMEM_WORDS`.
- Alignements : `XW` ∈ {100, 52, 28} ≡ 0 mod 4 (copies de 16 octets), `WW` pair (copies de 8 octets de l'int4), `STAGE` ≡ 0 mod 4, `x` aligné sur 16 octets vérifié par `x_of`, d_in multiple de 8 (Tetra) ou de 32 (int4) vérifié.
- B non multiple de la tuile : n = 50 prend l'entrée de 64 rangées. Les rangées 50 à 63 sont mises en scène à zéro, les écritures gardées par `n < n_rows`, dans le chemin direct comme dans le pli.
- Bornes des mots : un chunk commence sur un bloc pair (KC/24 = 8, 4 ou 2) ; `f1r_load` lit les mots `⌊1,5 jl⌋` et `+ 1`, au plus `WC - 1`. Le mot int4 tuilé suit `pers_q4_tile_index` à l'identique (au 4B, down 1 216 mots : 1 152 en tuiles et 64 dans l'ordre du fichier).
- Split-K : `S = ⌈nk / slice⌉`, d'où `(S - 1) slice < nk` : aucune tranche vide. Partiels de `S x n x d_out` par projection, dont les offsets s'additionnent ; tickets `tk0 + tile` disjoints, remis à zéro par le dernier arrivé. Au 14B à B = 128 : q/k/v S = 6, 5,5 M flottants ; down S = 8, 5,2 M ; tête S = 1.
- d_out multiples de 128 au 4B, 8B et 14B, tête comprise (151 936 = 1 187 x 128). Les sorties `Launch` non initialisées sont entièrement écrites (128 rangées par CTA et toutes les colonnes `n < n_rows`).
- Lot : `max` borné à 1..=128 (`served.rs:185`), contrôle du lot dans `batch_logits_slots` (`model.rs:3290-3301`). Une séquence qui entre ou sort entre deux pas ne touche que `len[slot]` ; tickets d'attention indexés par rangée et remis à zéro à chaque lancement ; longueurs mêlées gérées par `pos[z]` et des offsets en 64 bits (`attn_dec_batch.cu:45-52`). Pool de 2 048 positions : cap arrondi à 256, `max_len ≤ cap` vérifié.
- MMA (non servi) : `smem(nt)` égal au calcul du noyau ; `PART_FLOATS / (n d_out)` borne S ; d_out / 128 ≤ 4 096 tickets ; `xsum` tient dans la place du `lo` de Tetra.

### 3. Rotation
- Servi (`rot_apply`, `rot_apply_rows`) au 14B : 17 408 x 4 = 69 632 B partagés, sous 101 376. k = 17 et 5 ≤ `LLVQ_ROT_KMAX` ; m ≥ 512, exigé par l'hôte de 4E (`pers.rs:440`).
- `rotate_wide.cu` et `tv_q4_wide.cu` ne sont compilés que pour une largeur que `rot_plan` refuse (25 600) ; aucune taille servie n'y passe. Leurs offsets de rangée sont en 64 bits ou petits (5 120 x 3 200 = 16,4 M en u32) ; le scratch `n_rows n + (KMAX - k) m` est vérifié par `launch_rot_dev`.

### 4. `llvq_pers.cuh`, sections 3 à 7b
- Index de table et d'en-tête : `PERS_HDR_SUM` = 32 à 63, 32 partiels au plus, sans chevauchement avec OK, KEY, TICKET et V7. `pers_small_load` et `pers_small_store` couvrent `k x 32` flottants à 384 threads (608 pour k = 19). `pers_load_1rt` et `pers_ln_1rt` lisent le mot de signes `q / 4`.
- `pers_mix_column_k4` lit au plus 8 flottants par rangée à k = 5 et 20 à k = 19, dans les `k x 32` copiés par `pers_v7_stage_rot`. L'aire v7 est alignée sur 16 octets : l'aire commence à 19 072 B, l'offset est multiple de 4 flottants.
- `pers_q4_row` et `pers_q4g64_row` : produits `row x nwords` en u32, au plus 151 936 x 640 = 97,2 M au 14B.

### 5. Décodeurs Tetra (`llvq_tetra48.cuh`, `llvq_f1rank_v3.cuh`, `f1r_load`)
- Tout mot de 48 bits, même corrompu, reste dans les tables : préfixes ≤ 127, branches ≤ 1 023, suffixes ≤ 127, rangées ≤ 4 095 (aussi pour `idx2` au-delà de `F1R_N0_MIXED`), `invnorm` masqué à 31, `gscale` à 0 ou 1. `f1r_load` du dernier bloc lit au plus le mot `⌈6 nblocks / 4⌉ - 1`, sous `stride`, vérifié par l'hôte.

### 6. Metal (`llvq_rot.metal`, `emb_q4.metal`)
- Hors BUG-70 : `rot_mix` borne l'index de lecture ; `scratch` de `rows x n` flottants ; `xout` distinct ; `small` factice d'un flottant à k = 1, jamais lu.
- `tv_emb_q4_metal` met en scène d x 4 octets, au plus 20 480 B au 14B, sous 32 768, refusé au-delà (`fused_metal.rs:1800-1803`) ; `y_off = r vocab` en u32, 28 269 rangées avant débordement, la tête n'en projette qu'une par pas. `emb_q4_gather_metal` : `tok d` au plus 40 960 x 5 120 = 209,7 M. `gpr = ⌈d / groupe⌉` calculé par le chargeur (`fused_metal.rs:2302`).

## Candidats au lot 1

BUG-70.
