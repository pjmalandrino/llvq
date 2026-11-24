# Bugs, noyaux GPU et leur hôte de lancement : constats
Périmètre réellement lu : en entier `llvq-cuda/kernels/pers_4e.cu`, `pers_segment.cu`, `pers_4et_a12g.cu`, `pers_4et_a12g32.cu`, `rotate.cu`, `attn_dec_batch.cu`, `llvq-llm/kernels/tv_q4_tile.cu`, `emb_q4.cu`, `tv_q4_h.metal`, `llvq-cuda/src/attn.rs`, `llvq-llm/src/attn.rs`. En grande partie : `llvq-cuda/kernels/llvq_pers.cuh` (sections 1, 7c à 11, lignes 73-207 et 814-2069), `llvq-llm/src/persist4e_cuda.rs` (1-1620, 1860-2709), `served.rs` (150-570), `served_decode.rs` (20-500), `batch_cuda.rs` (30-686), `batch.rs` (150-528), `fused_cuda.rs` (700-1480, 2370-2465, 2793-2911, 3059-3450), `fused_metal.rs` (1-1045, 1180-1840), `llvq-cuda/src/gpu.rs` (184-590), `modcache.rs` (1-200), `session.rs` (60-290), `llvq-runtime/src/engine.rs` (525-770, 895-1135, 1255-1300). Survolés : `pers.rs` (690-730, 1530-1710, 2280-2300, 2680-2690, 2940-3210), `tile.rs`, `spec_loop.rs` (chaîne), `model.rs` (KvCache 440-640, `batch_step_slots`), `persist_cuda.rs` (aides de 4E), `attn_cuda.rs`, `tv_tetra48_h.cu`, `llvq_tetra48.metal` (215-420, 780-875), `attn_dec.cu` (1-120).
Non couvert : les corps partagés de la passe d'arbre servie (`pers_4et.cu`, `pers_4et_stream.cu`, `pers_4et_a9.cu`, `pers_4et_a10.cu`, `pers_4et_a10rc.cu`, `pers_4et_a12.cu` : `p4t_phase_rc`, `p4t_io`, `p4t_fold_a9`, prologues) et `pers_attn_full.cuh` ; `tv_batch_gemm.cu` et `tv_batch_mma.cu` (le premier blanchi par l'audit du 2026-11-08) ; `llvq_rot.cuh`, `rotate_wide.cu`, `tv_q4_wide.cu` ; les décodeurs `llvq_tetra48.cuh` et `llvq_f1rank*.cuh` ; `llvq_rot.metal`, `emb_q4.metal` ; les sections 3 à 7b de `llvq_pers.cuh` ; la passe de vérification `pers_4ev.cu` (non servie).
Bilan : 0 critique, 1 majeure, 1 mineure, 2 cosmétiques

Audit du 2026-11-08 §1.1 (4E empoisonné après un watchdog ou un NaN) : corrigé à 925930d. `Runtime4e::settle` (`persist4e_cuda.rs:1309-1312`) appelle `pers::p4e_settle` (`pers.rs:1616-1635`) après chaque lancement de décodage, de vérification et d'arbre (`persist4e_cuda.rs:1385`, `:1584`, `:2409`), et remet `bar`, `abort`, `nan` et `bar_base` à 0 sur faute. Les tickets et les slots sont remis à zéro par chaque noyau à son départ (`pers_4e.cu:236-241`, `pers_4et_a12g.cu:304-307`).

## Noyaux réellement servis (référence pour les autres relecteurs)

Source : `llvq-runtime/library.json` (configs par backend), puis `served.rs`, `served_decode.rs`, `persist4e_cuda.rs`, `fused_cuda.rs`, `batch_cuda.rs`, `fused_metal.rs`.

| porte | config | module NVRTC ou bibliothèque MSL | symboles lancés |
|---|---|---|---|
| CUDA 4B (défaut de `llvq serve`) | `qwen3-4b-tetra-e4-4e-v9-chain.json` | les quatre modules ci-dessous | voir lignes suivantes |
| CUDA 8B, 14B ; 4B `LLVQ_PROFILE=eager` | `qwen3-{4b,8b,14b}-tetra-e4-4e-v9.json` | unité eager, attention, 4E | idem, sans la chaîne |
| CUDA batch (`LLVQ_BATCH`) | `qwen3-*-4e-v9-batch.json` | unité eager, attention, 4E, unités batch | idem + GEMM batch |
| Metal 4B, 8B, 14B | `qwen3-{4b,8b,14b}-tetra-e4.json`, `MetalTuning::default` | bibliothèques MSL | voir dernière ligne |

1. Unité eager (`fused_cuda.rs:713-780`, empreinte épinglée `e269ea5d…` sous v9) : défines de tuile et de préremplissage, `llvq_slot.cuh`, `matvec.cu`, `llvq_rot.cuh`, `rotate.cu`, `llvq_f1rank.cuh`, `llvq_f1rank_v3.cuh`, `llvq_tetra48.cuh`, `tv_tetra48_h.cu`, `emb_q4.cu`, `tv_q4_h.cu`, `tv_q4_tile.cu`. Lancés : `rot_apply`, `rot_apply_rows`, `tv_tetra48_h`, `tv_tetra48_rows_h`, `emb_q4_gather`, `tv_emb_q4_h`, `tv_q4_ht`, `tv_q4_hht` (et `tv_q4_hh` sur une copie, au contrôle de chargement `check_int4_tile_parity`). L'unité large (`rotate_wide.cu`, `tv_q4_wide.cu`) n'est compilée pour aucune taille servie.
2. Attention : `attn_dec.cu` seul (`llvq-cuda/src/attn.rs:60-65`), `attn_dec_g4_u64` (4B, 8B) ou `attn_dec_g5_u64` (14B), lignes de décodage du chemin eager (forme kernel dès la position 0).
3. 4E (empreinte `1b422b62…`) : `#define PERS_4E_ROWS 7u` puis `PERS_UNIT_PARTS` (`pers.rs:1674-1688`) : `llvq_slot.cuh`, `matvec.cu`, `llvq_f1rank.cuh`, `llvq_f1rank_v3.cuh`, `llvq_tetra48.cuh`, `llvq_rot.cuh`, `rotate.cu`, `llvq_pers.cuh`, `tetra48_pers.cu`, `tetra48_persall.cu`, `pers_ref.cu`, `pers_segment.cu`, `pers_4e.cu`. Lancés : `pers_rope_probe_f{0,1,2}` (chargement), `pers_4e_tokens_f{form}_t512` (servi, coopératif, une CTA par SM). `tetra48_pers.cu`, `tetra48_persall.cu`, `pers_ref.cu` sont compilés et jamais lancés.
4. Préremplissage en chaîne, 4B seulement : module de base `p4t_unit_text` (le texte 4E forme 7 et `pers_4et.cu`, symbole `pers_4et_rows_f*_t512` compilé et contrôlé en occupation, jamais lancé par la chaîne) et module du bras `a12ipg32ft` = `pers_ft_apply(p4t_a12ipg32_unit_text)` : texte 4E + `pers_4et.cu` + `pers_4et_stream.cu` + `pers_4et_a9.cu` + `pers_4et_a10.cu` + `pers_4et_a10rc.cu` (16) + `pers_4et_a12.cu` (`P4T_A12IP`) + `pers_4et_a12g.cu` + `pers_4et_a12g32.cu`, avec `pers_attn_full.cuh` inséré. Lancés : `pers_4et_a12ipg32_f*_t384`, et au chargement `p4t_tile_words`, `p4t_ip_check`.
5. Batch (`batch_cuda.rs:188-204`) : unité GEMM (`tv_batch_gemm.cu`, `tv_bg_n8`…`tv_bg_n128`) et attention (`attn_dec.cu` + `attn_dec_batch.cu`, `attn_slot_g{4,5}_u64`). L'unité MMA (`tv_batch_mma.cu`) est compilée, son chemin n'est pas celui de la config servie (`BatchProjs::Gemm`).
6. Metal : `llvq_tetra48.metal` (`tv_tetra48_metal_lut`), `llvq_rot.metal` (`rot_apply_metal`, `rot_apply_rows_metal`), `tv_q4_h.metal` (`tv_q4_metal` jusqu'à d_in 8 192, `tv_q4_metal_tiled` à 2 048 colonnes au-delà), `emb_q4.metal` (`emb_q4_gather_metal`, `tv_emb_q4_metal`). Attention candle (forme expanded). `llvq_tetra48_rows.metal`, `tv_q4_pf.metal` et `attn_dec.metal` ne servent qu'aux bras de `bin/metalrun`.

Non servis : `pers_4ev.cu` (vérification spéculative), les autres bras `pers_4et_*`, `pers_4e_k8.cuh` (v9k8, NOT SERVED), `pers_q4rows.cu`, `pers_trows.cu`, `pers_ctr.cu`, `planes*`, `golay70*`, `e1v`, `f1rank_v1/v2`.

Contrôles faits sans constat : mémoire partagée de 4E au 14B 96 064 B (calculé, `p4e_smem_bytes_form(7, 5120, 17408, 5) + 7 360`) sous les 101 376 B d'opt-in, l'aire v7 ne chevauchant ni le vecteur ni l'étage d'attention ; aucun produit d'indices u32 ne dépasse 2³² au 14B ni à B = 128 (le plus grand est la tête, 151 936 × 640 mots) ; comptes de barrières de 4E (`n_bar_4e = 2 + 6L`) et de `a12ipg32` (`n_bar_a9 = 4 + 13L + n_ot`, tête coupée ou non) égaux aux attentes des noyaux ; scratch d'attention dimensionné sur `MAX_CAP` et indexé par `ceil(cap / 256)` avec `cap ≤ MAX_CAP` ; alignement 16 octets des mots int4 tuilés garanti par `int4_tileable` ; tuile partielle de `tv_q4_metal_tiled` dans l'ordre de `tv_q4_metal`.

### BUG-50 Une erreur d'un lancement de la chaîne après le premier laisse les caches en avance sur la session, et chaque requête suivante échoue
- Gravité : majeure
- Confiance : CONFIRMÉ (le mécanisme se lit de bout en bout ; le déclencheur est une erreur de carte, voir Scénario)
- Où : `llvq-llm/src/persist4e_cuda.rs:2243-2256`, `:2257-2289`, `:2401-2421`, `:2196-2215`, `:2448-2450` ; `llvq-llm/src/session.rs:127-131`, `:143-147`, `:172-180` ; `llvq-runtime/src/engine.rs:539-541`, `:657`, `:668`
- Fait : `tree_launch`, pour un lancement qui déplace les lignes du précédent (`mv` non vide, donc tout lancement de chaîne après le premier et la passation), fait `c.reserve(pos0 + span)` puis `c.set_len(pos0)` sur chaque cache (2249-2255) AVANT ses contrôles (2257-2288), avant le lancement (2401) et avant `settle` (2409-2415). Toute sortie en erreur après ce bloc laisse donc la longueur des caches à `pos0`, alors que `chain_prefill` propage l'erreur par `?` sans rien remettre (2204, 2212) et que `Session::feed_with` n'étend `ids` qu'en cas de succès (`session.rs:177-178`). Côté session, `Session::truncate` sort tout de suite quand `n == self.ids.len()` (`session.rs:129-131`) sans toucher aux caches, et `reuse_prefix` l'appelle avec `keep ≤ ids.len()`. Le moteur renvoie un 500 sans réinitialiser la session (`engine.rs:539-541`).
- Scénario : 4B CUDA, config par défaut (chaîne), session vierge (`ids` vide). Un prompt de 100 jetons donne les lancements à `pos0` 0, 32, 64, 96 puis la passation à 100. Le lancement à 64 sort en erreur : watchdog d'une barrière (`P4eFault::Abort`), NaN au dernier lancement, ou OOM dans `reserve` (croissance du cache par pas de 256 positions, ancien et nouveau tampon coexistent). Les caches tiennent alors 64 positions (ou des longueurs mêlées si l'OOM frappe au milieu de la boucle), `ids` reste vide. Requête suivante, quel que soit son prompt : `keep = 0 = ids.len()`, `truncate` ne fait rien, `chain_prefill(pos0 = 0)` arrive à `prepare`, qui refuse : « 4E: a cache holds 64 positions, the launch starts at 0 » (2448-2450). Chaque requête suivante répond 500 jusqu'au rechargement du modèle (déchargement par keep-alive ou éviction) ; rien dans le moteur ne remet la session à neuf sur une erreur. Avec une session non vide, le blocage vaut pour toute requête qui reprend tout le préfixe tenu, c'est-à-dire toute suite de la même conversation. Hors runtime, un déclencheur déterministe existe : `ChainFault::PathOffByOne` fait échouer le contrôle de chemin (2260-2267) du deuxième lancement, après le `set_len`.
- Touche : chemin servi
- Correction proposée : dans `Runtime4e::chain_prefill`, sur toute erreur, remettre chaque cache à la longueur d'entrée (`c.set_len(pos0)`) avant de rendre l'erreur ; et dans `Session::truncate`, ne sortir tôt que si chaque cache tient déjà `n` positions (sinon `c.truncate(n)`). Déplacer les contrôles de `tree_launch` avant le bloc `mv` ferme aussi les cas déterministes.
- Effort : S
- Risque de la correction : faible. Chemin d'erreur seulement ; aucun format, aucun index, aucun fichier de noyau épinglé ; les jetons servis d'un chemin sans erreur ne bougent pas.
- Vérification proposée : test CPU dans `session.rs` : une session sur un petit modèle CPU, `feed` de 10 ids, `set_len(12)` sur chaque cache pour simuler l'échec, puis `reuse_prefix` d'un prompt qui prolonge les 10 ids ; affirmer que chaque cache tient 10 positions (échoue aujourd'hui). Sur carte : `chain_prefill` avec `ChainFault::PathOffByOne` puis une seconde requête, qui doit réussir après la correction.

### BUG-51 `prepare` agrandit les caches avant de refuser une capacité au-delà de 8 704, et les laisse inutilisables pour 4E
- Gravité : mineure
- Confiance : CONFIRMÉ (le comportement) ; non atteignable par `llvq serve` ni par le REPL, qui plafonnent le contexte (`engine.rs:1272-1276`, `repl.rs:211`, `:268`)
- Où : `llvq-llm/src/persist4e_cuda.rs:2451-2456` ; `:2249-2255` contre `:2279-2281` ; `llvq-llm/src/model.rs:589-604`, `:526-551`
- Fait : `prepare` appelle `c.reserve(pos0 + n)` (2451), qui réalloue les tampons Grow au multiple de 256 suivant, puis seulement refuse `cap > MAX_CAP` (2454-2456). `tree_launch` fait de même : `reserve(pos0 + span)` dans le bloc `mv` (2253) avant le contrôle `pos0 + span > MAX_CAP` (2279). `KvCache::truncate` ne réduit jamais la capacité (`model.rs:526-551`).
- Scénario : un harnais qui ne lit pas `Decoder::max_positions` lance 4E à `pos0 = 8 650` pour 64 jetons : `reserve(8 714)` porte la capacité à 8 960, puis le lancement est refusé. Les caches gardent 8 960 positions de capacité ; même après `truncate(100)`, chaque lancement 4E suivant sur ces caches est refusé (« a cache of 8960 positions, the attention scratch holds 8704 »), alors que la position tient. Seuls des caches neufs en sortent.
- Touche : runtime
- Correction proposée : refuser `pos0 + n > MAX_CAP` (et `pos0 + span > MAX_CAP` dans `tree_launch`) avant tout `reserve`, comme le fait déjà la ligne 2279 pour l'arbre mais trop tard.
- Effort : S
- Risque de la correction : nul. Aucun format, aucun fichier épinglé, aucun jeton servi.
- Vérification proposée : aucune, lecture suffisante ; sur carte, un appel `decode` à `pos0 = 8 650, n = 64` suivi d'un `truncate(100)` et d'un `decode` doit réussir après la correction.

### BUG-52 Commentaires de l'hôte 4E périmés sur ce qui est servi
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/persist4e_cuda.rs:13-27`, `:487-489`, `:191-194` ; `llvq-cuda/kernels/pers_4e.cu:23-25`
- Fait : la doc du module nomme `qwen3-4b-tetra-e4-4e-v4.json` et sa jumelle v6 comme config servie du 4B ; `install_rows` dit que la forme 2 (v4) est « what the served config runs » ; la doc de `Tree4e` dit « Not a served path: S1 measures it », alors que la passe d'arbre sert le préremplissage du 4B depuis le 2026-11-15 (`served_decode.rs:448-456`) ; `pers_4e.cu` dit qu'un o Tetra du 8B est refusé par l'hôte, alors que `Pers4eOT` le sert (`persist4e_cuda.rs:902-916`).
- Scénario : un lecteur qui croit la passe d'arbre hors service traite ses chemins d'erreur comme secondaires ; c'est le cas de BUG-50.
- Touche : doc
- Correction proposée : mettre à jour les trois commentaires de `persist4e_cuda.rs`. Laisser `pers_4e.cu` tel quel, ou le corriger au prochain déplacement de son empreinte.
- Effort : S
- Risque de la correction : nul pour `persist4e_cuda.rs`. `pers_4e.cu` est dans l'unité épinglée `1b422b62…` : un seul octet change l'empreinte et rouvre les portes servies.
- Vérification proposée : aucune, lecture suffisante.

### BUG-53 Le shader int4 Metal servi affirme qu'aucun hôte ne vérifie ses contrats
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/kernels/tv_q4_h.metal:45-63`, `:246-249` ; `llvq-llm/src/fused_metal.rs:1411-1421`, `:1426`, `:1600` ; copie `llvq-tetra/llvq_tetra/data/tv_q4_h.metal`, vérifiée par `llvq-tetra/llvq_tetra/metal.py:75-79`
- Fait : l'en-tête dit « None of the three is written. There is no Metal host for this kernel yet », et `tv_q4_metal` dit « NOBODY asserts that today » pour `d_in % 8`. `upload_int4_tiled` refuse pourtant `d_out % 8 != 0` et `d_in % 8 != 0`, `int4_tile_for` borne `d_in * 4` contre la limite, et `Int4Op` pose la mémoire de threadgroup.
- Scénario : un lecteur ajoute une garde au noyau servi ou croit le chemin non protégé.
- Touche : doc
- Correction proposée : réécrire les deux passages pour citer les gardes de `fused_metal.rs`.
- Effort : S
- Risque de la correction : faible. Le fichier est copié octet pour octet dans la roue `llvq-tetra`, avec son sha256 dans `tetra-tables.json` : il faut régénérer la copie et les tables (`bin/tetratables`). Aucun bit calculé ne bouge.
- Vérification proposée : aucune, lecture suffisante.

## Connus non corrigés

- CONNU-NON-CORRIGÉ (audit 2026-11-08 §4, « PLAUSIBLE, session.rs:96-106 ») : `Session::feed` (`session.rs:155-166`) laisse toujours les premières couches en avance si `hidden_cached` échoue en milieu de modèle, et l'early return de `truncate` (BUG-50) empêche toujours le rattrapage : la requête suivante part de caches dont la longueur diffère de `ids` (refus de 4E sur CUDA, positions décalées ou erreur de forme sur les portes eager selon le store).

## Candidats au lot 1

BUG-50, BUG-51, BUG-52 (partie `persist4e_cuda.rs` seulement).
