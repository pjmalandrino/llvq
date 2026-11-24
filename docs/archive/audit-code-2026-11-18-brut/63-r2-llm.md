# R2, moteur llvq-llm non lu au premier tour : constats
Périmètre réellement lu : en entier `llvq-llm/src/fused.rs` (1-3084), `fused_cuda.rs` (1-4189), `batch.rs`, `batch_cuda.rs`, `served_decode.rs`, `kvq.rs`, `kvladder.rs`, `calib.rs` (1-1662), `speculative.rs` (corps, 1-290), `dflash_q4.rs`, `dflash_spec.rs`, `artifact2.rs` (1-530). Par sections : `model.rs` 30-245, 325-705, 1010-1875, 2160-2320, 2366-2550, 3185-3345, 3470-3510, 3824-3880, 5340-5406 ; `dflash.rs` 160-660 ; `bin/smoke.rs` 960-1300 ; `session.rs` 96-185 ; `served.rs` 140-235, 255-290, 700-840, 990-1040 ; `rotplan.rs` 472-562 ; `device.rs` 60-150 ; `llvq-runtime/src/engine.rs` 50-200, 514-634, 790-1270 ; `llvq-runtime/src/store.rs` 420-457 ; `corpus.rs` 530-700 ; `llvq-artifact/src/tetra48.rs` 40-175, `format.rs` 660-760 et 1955-2010 ; `llvq-cuda/src/gpu.rs` 240-330 et 820-870, `attn.rs` 60-130. Noyaux, en lecture seule et pour les signatures seulement : `tv_tetra48_h.cu`, `emb_q4.cu`, `tv_q4_tile.cu`, `rotate.cu`.
Non couvert : `dflash.rs` 1-160 et 660-1117 (lectures de sortie, arbre, fixture) ; les tests de `speculative.rs` et de `model.rs` (3880-5340) ; `model.rs` 3510-3820 (mesures, lues par 04) ; `guided.rs` ; `persist_cuda.rs` (`tetra_of`, `int4_of`) ; `fused_metal.rs` et `fused_cpu.rs`, hors mission.
Bilan : 0 critiques, 0 majeures, 2 mineures, 3 cosmétiques

Résumé. Le transcodage du fichier scellé vers la carte, le dimensionnement des tampons, l'hôte du batch continu, la boucle de calibration et la reprise sur shard sont sains à la lecture. Deux défauts réels, tous deux mineurs : le pool du batch préremplit une requête par morceaux de 256 positions alors que le chemin batch-1 la préremplit par morceaux de 4 096, ce qui change l'arithmétique de l'attention sur certaines longueurs ; et trois lecteurs de fichiers préallouent sur un compte lu dans le fichier sans le borner. Le reste est de la documentation fausse ou du code mort.

### R2-LLM-01 Le pool du batch préremplit par morceaux de 256 positions, le chemin batch-1 par morceaux de 4 096 : sous la forme `kernel`, la même requête n'a pas la même attention selon qu'une autre est en cours
- Gravité : mineure
- Confiance : CONFIRMÉ pour la divergence des chemins (lus de bout en bout) ; l'écart de jeton qui en résulte est PLAUSIBLE, jamais observé
- Où : `llvq-llm/src/batch.rs:374-376` (doc : « the served prefill »), `:385` (`prompt.chunks(crate::model::MAX_ROWS)`) ; `llvq-llm/src/session.rs:166-176` (`ids.chunks(qwen.prefill_rows())`) ; `llvq-llm/src/model.rs:2723-2725` (`prefill_rows`, 4 096 sur CUDA), `:1124` (`MAX_ROWS = 256`), `:2222-2249` (branche `Kernel` d'`attention_ctx`) ; `llvq-llm/src/served.rs:271-276` et `:1019` (seuil 0 sous `decode: persistent`) ; `llvq-runtime/src/engine.rs:858` (`admit` vers `Pool::admit`) et `:948` (`solo` vers `Session::feed`) ; `configs/qwen3-{4b,8b,14b}-tetra-e4-4e-v9-batch.json` (`attention: kernel`)
- Fait : `Qwen3::prefill_slot` coupe le prompt par `MAX_ROWS` = 256. `Session::feed` le coupe par `prefill_rows()`, qui vaut 4 096 sur CUDA Tetra48 (`row_cap(4)`), et sa propre doc dit que la coupe doit être celle du modèle et non une constante. Sous `attention: kernel` sur un store `Grow`, un morceau d'au plus 8 lignes (`KMAX`) part à `kernel_rows`, un morceau plus long à `grouped_attention` ; `model.rs:2228-2230` écrit que les deux ne sont pas au bit. Sous les configs batch servies, `kernel_from` vaut 0, donc toutes les lignes d'un petit morceau vont au noyau.
- Scénario : `llvq serve` sous `LLVQ_BATCH`, config `qwen3-4b-tetra-e4-4e-v9-batch.json`, prompt de 260 jetons. Arrivé seul : `solo`, `Session::feed`, un morceau de 260 lignes, toutes par `grouped_attention`. Arrivé pendant qu'une autre requête décode : `admit`, `prefill_slot`, morceaux 256 puis 4, les lignes 256 à 259 par le noyau d'attention. Les sorties d'attention de ces 4 lignes changent de bits à chacune des 36 couches, donc leurs états cachés, les K/V des positions 256 à 259 dès la couche 1, et les logits du premier jeton. Longueurs touchées à coup sûr : plus de 256 et `len mod 256` dans 1..8, soit 257-264, 513-520, ..., 1 793-1 800 sous un contexte de 2 048. Pour les autres longueurs au-delà de 256, le découpage en 256 passe par la même `grouped_attention` avec d'autres formes de matmul : écart de bits PLAUSIBLE, non établi. La porte du batch ne l'a pas vu : `batchrun` passe aussi par `prefill_slot` (`bin/batchrun.rs:318`), et le plus long prompt GSM8K fait 215 jetons (*computed*, 04-bugs-moteur.md BUG-08).
- Touche : runtime (`llvq serve` sous `LLVQ_BATCH`, pas le défaut). Le décode groupé est déjà sous la licence T8 et pas au bit avec le batch 1 ; ce défaut ajoute une seconde divergence, non mesurée, qui dépend de la charge du serveur et non de la requête.
- Correction proposée : dans `Qwen3::prefill_slot` (`batch.rs:385`), remplacer `prompt.chunks(crate::model::MAX_ROWS)` par `prompt.chunks(self.prefill_rows())`, et réécrire la doc de `:374-376` pour dire « the chunking of `Session::feed` ». Test dans `mod tests` de `model.rs` (où `tiny_model_long` est visible) : `tiny_model_long(F16, 8, 2, 128, 320)`, `set_kv_store(Grow(256))`, `set_attn_heads(Kernel)`, noyau CPU de `crate::attn::kernel_for(&Device::Cpu, ...)`, `set_served_kernel_from(0)` ; un prompt de 260 ids par `Session::feed` d'un côté, par `prefill_slot` puis `slot_to_caches` de l'autre ; exiger les K et V de la dernière couche égaux au bit et le même premier jeton. Mutant : remettre `MAX_ROWS` à `:385`, le test doit rougir.
- Effort : S
- Risque de la correction : faible. Aucun format, aucun index, aucun fichier de noyau. Elle change les jetons servis sous `LLVQ_BATCH` pour les longueurs ci-dessus, et l'objet que mesure `batchrun` pour un prompt de plus de 256 jetons (aucun dans la porte GSM8K publiée) : à faire avec l'accord de l'opérateur.
- Vérification proposée : le test ci-dessus et son mutant, `cargo test -p llvq-llm --lib model::`.

### R2-LLM-02 Trois lecteurs préallouent sur un compte lu dans le fichier, sans la borne `PREALLOC_CAP` que le crate du format s'impose
- Gravité : mineure
- Confiance : CONFIRMÉ (lu ; le compte n'est borné nulle part en amont)
- Où : `llvq-llm/src/fused.rs:2423` (`Vec::with_capacity(head.matrices as usize)`), `:2545-2546` (`n_raw` puis `Vec::with_capacity(n_raw as usize)`) ; `llvq-llm/src/bin/seal.rs:115` ; à comparer à `llvq-artifact/src/format.rs:540` (`PREALLOC_CAP = 1 << 20`), `:587`, `:1099-1137` ; `read_header` lit `matrices` comme n'importe quel `u32` (`format.rs:1976`)
- Fait : `fused::load_with`, le chargeur commun aux trois backends (`fused_cuda.rs:3894`, `fused_metal.rs:2127`, `fused_cpu.rs:359`), et `bin/seal` réservent autant d'éléments que le fichier en annonce. Le crate du format borne chacune de ses réservations pour cette raison (« a torn file's last bytes can decode to an enormous `name_len` », `artifact2.rs:168-171` dit la même chose pour le shard).
- Scénario : un fichier dont l'en-tête annonce 0xFFFF_FFFF matrices. Un `FusedMatrix` pèse au moins 176 octets (nom, quatre `usize`, flux, deux `Vec`, rotation, octets), donc la réservation demande environ 756 Go (*computed*) ; l'allocation échoue et `handle_alloc_error` avorte le processus. Ce n'est pas une panique : le `catch_unwind` posé sur le chargement du runtime ne l'attrape pas. Même chose pour le compte de tenseurs portés (`RawTensor`, au moins 72 octets, environ 309 Go). Atteint par tout binaire qui charge un chemin (`fusedrun`, `mmlu`, `gsm8k`, `chat`, `batchrun`, `specbench`, `prefillbench`, `seal`). Pas par `llvq serve` : `pull` et `import` vérifient taille et sha256 contre les trois entrées épinglées (`store.rs:430-456`).
- Touche : format (lecteurs), outillage
- Correction proposée : rendre `PREALLOC_CAP` public dans `llvq-artifact` (ou le recopier en constante nommée) et écrire `Vec::with_capacity((head.matrices as usize).min(PREALLOC_CAP))` à `fused.rs:2423`, `(n_raw as usize).min(PREALLOC_CAP)` à `:2546`, et le même `min` à `seal.rs:115`. Test : écrire avec `llvq_artifact::write_header_kind(&mut f, 5, u32::MAX, CodeKind::Tetra)` un fichier réduit à son en-tête, puis `fused::load(path, FusedLayout::Tetra48)` doit rendre une `Err` (fin de fichier) ; mutant : retirer le `min`, le processus de test avorte.
- Effort : S
- Risque de la correction : nul. Aucun octet écrit ou lu ne change, seule la capacité initiale d'un `Vec`.
- Vérification proposée : le test ci-dessus, `cargo test -p llvq-llm --lib fused::`.

### R2-LLM-03 Quatre blocs de doc collés sur la mauvaise fonction dans `fused.rs` et `fused_cuda.rs`
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où et fait :
  - `llvq-llm/src/fused.rs:1362-1369` : la doc de `matrix_side_bytes` (« Device bytes a matrix costs **beside** its block stream ») est collée au-dessus de celle de `runtime_bytes_of` (`:1370-1382`) ; `matrix_side_bytes` (`:1392`) n'a plus de doc.
  - `llvq-llm/src/fused_cuda.rs:180-186` : la doc de `FusedInt4Proj` (« One `v_proj` served as affine int4 g128 ») est collée sur `type Loaded` (`:187-191`) ; `FusedInt4Proj` (`:193`) n'en a pas.
  - `llvq-llm/src/fused_cuda.rs:2779-2789` : la doc de `launch_tetra48_h` et son `#[allow(clippy::too_many_arguments)]` sont collés sur `upload_int4` (`:2790-2796`) ; `launch_tetra48_h` (`:3306`) a un autre commentaire.
  - `llvq-llm/src/fused_cuda.rs:3779-3797` : la doc de `load_with` commence par celle d'une fonction `load` supprimée, et renvoie à [`load`], qui n'existe pas dans ce module (`git grep "fused_cuda::load("` : rien).
- Touche : doc
- Correction proposée : déplacer chaque bloc sur sa fonction, retirer l'attribut de `upload_int4`, réécrire la première phrase de la doc de `load_with` sans le lien mort.
- Effort : S
- Risque de la correction : nul. Commentaires Rust seulement ; aucun fichier de noyau.
- Vérification proposée : `ops/check-linux-cuda.sh clippy -p llvq-llm --features cuda --all-targets`, puis `cargo doc` sous la même cible pour le lien.

### R2-LLM-04 `FusedRuntime::new` : un contrôle de registres fait deux fois, un accesseur mort au contrat faux, une table de classes construite pour une disposition qui n'en lit pas
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où et fait :
  - `llvq-llm/src/fused_cuda.rs:859-861` et `:876-878` : `rows_kernel_name(model.layout)` est poussé deux fois dans `spill_checked`, donc `tv_tetra48_rows_h` passe deux fois par `cuda.report`.
  - `fused_cuda.rs:1448-1455` : `max_d_in()` n'a aucun appelant (`git grep "max_d_in()"` : rien), et sa doc dit que `new` refuse au-delà de cette borne ; `new` ne la compare à rien, c'est `wide::rot_kernel` qui décide (`:1041`).
  - `fused_cuda.rs:955-971` : la table des 384 classes de la boule (`FastDecoder::new`, 512 x 6 mots) est construite et téléversée pour toute disposition, Tetra48 servie comprise, qui ne la lit jamais ; même chose pour le `FastDecoder` de `Transcoder::for_kind` (`fused.rs:2187`).
- Touche : chemin servi (chargement), sans effet sur les jetons
- Correction proposée : retirer le second `if let` de `:876-878` ; supprimer `max_d_in` (champ et accesseur) ou corriger sa doc ; garder `tab` tel quel ou le rendre `Option` comme `tetra_tabs` (le second point est facultatif).
- Effort : S
- Risque de la correction : nul pour les deux premiers points. Le troisième change la forme de `FusedRuntime` et passe par les quatre lancements de la boule. Aucun texte NVRTC ne bouge.
- Vérification proposée : vérification de type Linux (`ops/check-linux-cuda.sh clippy`).

### R2-LLM-05 En-têtes et messages qui décrivent un objet qui n'est plus servi
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où et fait :
  - `llvq-llm/src/fused.rs:13-17` : « `Planes14` by default, `Planes12x` or `Slot32` » ; Tetra48 et Golay70 manquent, et les trois configs servies sont en `tetra48`.
  - `fused.rs:33-35` : « Qwen3-4B has 252 projections and **144** rotations ». Le fichier servi porte 84 enregistrements int4 sans rotation (`fused_cuda.rs:1525`, `:2457`), donc 168 projections lattice et 96 rotations (*computed* : 36 `Attn`, 36 `Mlp`, 24 `down_proj` Tetra, aucun `o_proj`).
  - `llvq-llm/src/fused_cuda.rs:4-10` : même liste de dispositions, et le schéma « deux lancements, `rot_apply` puis `tv_*_h` » oublie la projection int4, un lancement sans rotation.
  - `fused_cuda.rs:1793`, `:1839`, `:1847`, `:1861`, `:1905`, `:1913`, `:1927` : noms d'op et messages d'erreur « q8 » (`llvq-emb-q8-gather`, `lm_head q8 expects d=`, `q8 gather: zero tokens`) sur le chemin q4 des trois configs servies.
- Touche : doc
- Correction proposée : réécrire les trois en-têtes sur Tetra48 et les enregistrements int4 ; remplacer « q8 » par le nom du mode (`self.rt.emode.name()`) dans les messages.
- Effort : S
- Risque de la correction : nul. Aucun script ne lit ces chaînes (`git grep "emb-q8-gather\|lm_head q8" ops/` : rien) ; aucun fichier de noyau.
- Vérification proposée : aucune, lecture suffisante.

## Contrôlé sans constat

Valeurs *calculées* sauf mention.

- **Transcodage Tetra48.** Mot `idx | gain << 47`, six octets petit-boutistes, ligne au pas `ceil(6 n / 8) x 2` mots (4B : `d_in` 2 560, 106 blocs, 640 octets ; 9 728, 405 blocs, 2 432 octets ; 14B : 17 408, 725 blocs, 4 352 octets). La fenêtre de deux mots du dernier bloc est vérifiée par assertion (`tetra48.rs:139-147`). `pack_plane_bytes` ajoute un mot nul, compté hors de `bytes`, et le pas, lui, est compté (le NUM-01 du premier tour corrige `rtbits`, pas ce chemin). L'ordre des 14 arguments de `launch_tetra48_h` et des 17 de `launch_tetra48_rows_h` est celui de `tv_tetra48_h.cu:46-59` et `:157-173`.
- **Queue `d_in mod 24`.** Narrowée une fois `f64 -> f32 -> f16`, `d_out x tail_w` valeurs, factice d'un élément quand `tail_w = 0`, jamais lue par le noyau. Le mode `boundary` décale la vue de queue de `r0 x tail_w`.
- **int4.** Groupe 128, `d_in % 8`, `d_out % 8`, `scales = d_out x ceil(d_in / 128)`, `packed = d_out x d_in / 2`, vue `u32` petit-boutiste. Toutes les largeurs servies (2 560, 4 096, 5 120, 9 728, 12 288, 17 408) sont multiples de 128 et de 32 (tuilé). Les GEMM du lot refusent un groupe partiel par `gpr x 128 == d_in`. `tv_q4_ht` et `tv_q4_hht` partagent `q4t_staged_dot` : le contrôle de parité au chargement sur la première largeur couvre le chemin à décalage du préremplissage.
- **Embedding q4.** `packed = vocab x d / 2`, `gpr = d / 64`, `vocab % 8` (151 936 = 8 x 18 992), `d x 4 <= 48 Kio` (20 480 o au 14B). Le contrôle des ids du tokenizer (lot 2) passe avant le retrait des tables.
- **Dépassements u32.** `HeadOp` calcule `y_off = r x vocab` en u32, qui déborderait à `r >= 28 269` ; `group_forward` refuse un appel de plus de 4 096 lignes, et `prefill_chunked`, `logits_last` ne projettent que la dernière. `EmbedOp` : `ntok x d <= 4 096 x 5 120`. Rotation : `start` et `row_stride` en éléments, sous 2^32 à 4 096 lignes de 17 408.
- **Batch, hôte.** `len[slot]` égal à `prompt + said - 1` sur les trois entrées (`admit`, `adopt`, `resume`), égal au nombre d'ids que `Session::from_caches` exige. `release` remet `len` à 0 et le slot est réécrit en entier, zéros de bourrage compris, par `caches_to_slot`. Contexte du pool `min(serve.context, rope)`, `room > 0` exigé par `prepare`, donc `len < cap` à chaque pas. `max` borné à 1..=128 (`GEMM_MAX_ROWS`), contexte à 256..=40 960. `batch_kv_bytes` égal à l'allocation de `BatchKv::new` (19,3 Go à 64 x 2 048 au 4B). Un pas raté libère tous les slots sans laisser de position fantôme : `len` n'avance qu'après le dernier bloc. `swap_remove` en ordre décroissant des requêtes finies : correct.
- **KV int8.** Entrée f16, donc `mn` exact en f16 ; la saturation se limite à l'arrondi de `s` ; groupe le long de `head_dim` seulement. Le batch refuse `kv: q8` (`served.rs:808-809`), seul cas où `KvCache::restore` requantifierait un historique déjà requantifié. `kvladder` est un mode de mesure, quantile au rang le plus proche correct.
- **Calibration.** Dénominateur `total_rows` = somme des lignes des fenêtres, toutes `(1, calib_len)` par `chunks_exact` ; masque de la première fenêtre valable pour toutes. Reprise : `shard_extent` tronque aux blocs entiers ; `resume_from_shard` vérifie ordre, dimensions, type par projection, coquille, centroïdes et graine effective ; en-tête de sortie `per_block x n_target`, égal aux enregistrements recopiés plus ceux du segment. Échelles de ligne prises sur les lignes tournées avant quantification ; queue arrondie en f32 avant la dérotation ; une seule application dans `decode_matrix`. `wikitext2-test` est une contamination volontaire, nommée.
- **Spéculatif, DFlash.** `speculative_greedy` : retour du cible à `pos + a + 1`, du brouillon au même point ou retard d'un jeton rattrapé, arithmétique juste. Atteignables seulement par `specrun`, `specbench` et `dflashcheck`, tous trois dans l'image ; jamais par `llvq` ni par une porte servie. Aucune fonction publique morte parmi celles comptées (`readout*`, `draft_tree_scored`, `lookup_*`, `continue_*`).

## Candidats au lot 1

R2-LLM-02, R2-LLM-03, R2-LLM-04 (deux premiers points), R2-LLM-05. R2-LLM-01 n'y va pas : il change des jetons servis sous `LLVQ_BATCH`.
