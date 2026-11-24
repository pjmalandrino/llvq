# Lot 1, commentaires Rust faux ou périmés : journal

Base : worktree `audit-code`, branche `audit/code-2026-11-18`, partie de `925930d`. Rien n'est commité.

## Corrigé

| ID | fichiers | ce qui change, en une ligne | vérification faite |
|---|---|---|---|
| QUA-11 | `llvq-search/src/lib.rs`, `llvq-search/src/pack.rs`, `llvq-quant/src/gptq.rs`, `llvq-artifact/src/error.rs`, `llvq-quant/src/quantizer.rs` | trois tranches de 8 coordonnées à 256 entrées (trois lectures), le moteur m ≤ 13 renvoyé à `generic` ; `push` dit qu'à `width == 0` une valeur non nulle est perdue ; `NotAnArtifact` nomme les cinq magies ; table de `LeechShapeGain` à 48 + k et 48 + 16, avec le plafond 13 dit ; l'échelle de ligne en f64 fichier, f32 carte | `CHUNK = 8`, `NCHUNK = 3`, boucles `for k in 0..NCHUNK` ; `MAX_SHELL = 13`, `block_bits` = `index_bits(cap) + gain_bits` ou `+ 16` ; `read_header` accepte `LVQ1`..`LVQ5` ; l'assertion de `push` n'est PAS déplacée (code), seul le commentaire change |
| QUA-10 | `llvq-artifact/src/format.rs` | la doc de `is_self_contained` et du champ `version` dit « sections présentes, peut-être vides », pas de renommage (22-verif) | `finish()` écrit deux compteurs à zéro ; méthode = `version >= 2` |
| NUM-07 | `llvq-quant/src/gptq.rs`, `llvq-quant/src/quantizer.rs` | « f64 dans le fichier, f32 sur la carte » ; `64/d_in` b/poids, 0,025 sur 2560 (*computed*) au lieu de `16/d_in` | `format.rs:18-22` (f64), `fused.rs:1304` `ROW_SCALE_BYTES = 4` |
| NUM-03 | `llvq-quant/src/linalg.rs` | accord à l'arrondi, pas au bit ; le test exige `1e-8 · max(max|U|, 1)` entrée par entrée, n ∈ {7, 24, 64, 129} ; les codes peuvent différer | lu `llvq-quant/tests/g5_gptq.rs:1020-1039` ; `CLAUDE.md` non touché |
| OPT-08 | `llvq-llm/src/fused.rs`, `llvq-llm/src/fused_cuda.rs` | 142 s et 145 s rattachés à Planes14 ; sous Tetra48 le transcodage recopie six octets par bloc | `llvq-artifact/src/tetra48.rs:149-174` (copie de `TETRA48_BYTES` par bloc), `fused.rs:2302` appelle `transcode_tetra48` |
| BUG-52 (partie `persist4e_cuda.rs`) | `llvq-llm/src/persist4e_cuda.rs` | en-tête : configs servies v9 aux trois tailles et chaîne 4B ; `install_rows` liste les formes 1 à 7, la 7 (v9) servie ; `Tree4e` dit chemin servi de la chaîne 4B | `configs/README.md` « At a glance » ; `served.rs::Rows::form` V9 → 7 ; `served_decode.rs:420-426` installe la chaîne via `install_chain` → `install_tree` |
| ARC-12 | `llvq-llm/src/persist4e_cuda.rs`, `llvq-llm/src/persist_cuda.rs`, `llvq-llm/src/device.rs` | `persist_cuda.rs` : le segment est une mesure, le module est sur le chemin servi par ses aides ; `as_any` (deux traits) : appelé par le décodage servi via `tetra_of`/`int4_of` | `persist4e_cuda.rs:50` importe `int4_of, mlp_args, seg_tetra, tensor_addr, tetra_of` ; `tetra_of`/`int4_of` appellent `as_any()` |
| QUA-59 (hors `served_decode.rs`) | `persist4e_cuda.rs`, `llvq-cuda/src/pers.rs`, `served.rs`, `fused_cuda.rs` | doc orpheline « Shared bytes of a 4E CTA » fondue dans celle de `p4e_smem_bytes_form` (qui la calcule) ; forme 1 n'est plus « servie », forme 7 ajoutée ; `P4E_BENCH_FORMS` : v4 servie du 2026-11-05 jusqu'à v9, v9 servie, v6 n'est plus dite servie ; `served.rs` : sept variables, liées à `BELOW_THE_DOOR` ; `THREADS` dimensionne aussi les lancements int4 | `BELOW_THE_DOOR: [&str; 7]` ; `launch_q4_h(.., THREADS)` aux lignes 1403, 1422, 2426 ; `configs/README.md` : v6 « measured, never served » |
| COH-09 | `llvq-runtime/src/main.rs`, `llvq-runtime/src/engine.rs` | `LLVQ_BATCH` → `qwen3-*-tetra-e4-4e-v9-batch.json` ; la chaîne est le défaut 4B sur CUDA, `LLVQ_PROFILE=eager` prend l'eager | `llvq-runtime/library.json` `cuda_batch` ; `library.rs::config_for` refuse `chain` par nom |
| COH-10 | `persist4e_cuda.rs`, `fused_cuda.rs`, `served.rs` | en-têtes : configs v9 + chaîne ; le prompt passe par `forward_rotated_rows` (noyau de lignes `tv_tetra48_rows_h`, jusqu'à `prefill_rows()` lignes) ; sept variables | `fused_cuda.rs:1584-1610`, `:630` ; `bin/ppl` à `925930d` n'a aucune référence au chemin fusé (la phrase « bin/ppl keeps the dense path » reste) |
| BUG-08 (partie `model.rs`) | `llvq-llm/src/model.rs` | le commentaire de la branche d'attention dit qu'un morceau ≤ `kern.max_rows()` va au noyau, préremplissage compris, qu'un morceau plus long va au noyau sous la forme de préremplissage noyau, sinon au groupé, et que la découpe décide donc des bits | `model.rs:2227-2245`, `kernel_rows` (`:937`), `kernel_prefill` (`:981`) ; `session.rs` relu : son commentaire (`feed` découpe par `prefill_rows()`) est juste, non modifié |
| SEC-05 (commentaire seul) | `llvq-cuda/src/modcache.rs` | la somme est celle du fichier lui-même : elle garde contre la corruption, pas contre une modification ; le cache vaut ce que vaut son répertoire | `modcache.rs:92-121` ; aucun contrôle de droits ajouté (20-verif) |
| TST-12 | `llvq-cuda/src/rowcand.rs`, `llvq-cuda/src/pers.rs` | `#[ignore = "a dump tool, not a test: writes target/... for ops/...; run by hand with --ignored"]` | scripts qui les appellent : `ops/rowcand-sass.sh`, `ops/rc-regprobe.sh`, `ops/tree-sass-probe.sh` ; `pers.rs` et `rowcand.rs` sont compilés sur le Mac (`cargo check -p llvq-cuda --all-targets` vert) |
| TST-11 | `llvq-runtime/src/engine.rs` | `"5"` sort de `bad` ; `assert_eq!(s("5"), KeepAlive::For(Duration::from_secs(5)))` ; l'assertion devient `assert!(r.is_err(), ...)` | mutant `secs(n)` → `secs(n * 60.0)` dans `parse_str` (1 site) : le test rougit à `engine.rs:1610` (la nouvelle ligne) ; défait par l'édition inverse, reverdit |

## Laissé de côté

| ID | raison |
|---|---|
| QUA-59 (parties `served_decode.rs`) | `llvq-llm/src/served_decode.rs` n'est pas dans ma liste de fichiers : la doc de `install_verify` collée sur `runtime_mut` (`:180-191`) et le renvoi à `persist4e_cuda::install` au lieu de `install_rows` (`:354-356`) restent à faire |
| QUA-11 (assertion de `push`) | le constat propose de déplacer l'assertion avant le retour anticipé : c'est du code, hors de mon domaine ; seul le commentaire dit maintenant le comportement réel |
| BUG-08 (partie `bin/gsm8k.rs`) | fichier de l'autre agent |
| BUG-52 (partie `pers_4e.cu`) | noyau épinglé, interdit |

## Constat faux ou déjà corrigé

| ID | preuve |
|---|---|
| (aucun) | tous les constats de ma liste se sont vérifiés dans le code à `925930d` |

## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)

- `llvq-llm/src/fused_cuda.rs` (commentaires seulement ; liens rustdoc `FusedRuntime::forward_rotated_rows` et `FusedRuntime::prefill_rows` vérifiés : `impl FusedRuntime` à la ligne 683)
- `llvq-llm/src/persist4e_cuda.rs` (commentaires seulement)
- `llvq-llm/src/persist_cuda.rs` (commentaires seulement)
- `llvq-cuda/src/pers.rs`, `llvq-cuda/src/rowcand.rs`, `llvq-cuda/src/modcache.rs` : déjà compilés sur le Mac par `cargo check -p llvq-cuda --all-targets` (vert), les deux `#[ignore = "..."]` compris

Relecture du diff : `git diff` sur mes 19 fichiers ne montre que des lignes `//`, `///`, `//!`, plus les deux `#[ignore = "..."]` et les quatre lignes du test `keep_alive_reads_ollamas_forms` (un commentaire, un `assert_eq!`, la liste `bad`, l'`assert!`).

## Commandes lancées et leur résultat (les lignes `test result:` telles quelles)

`git diff --stat` sur mes fichiers :

```
 llvq-artifact/src/error.rs     |  3 ++-
 llvq-artifact/src/format.rs    | 10 +++++++---
 llvq-cuda/src/modcache.rs      |  6 +++++-
 llvq-cuda/src/pers.rs          | 27 ++++++++++++++-------------
 llvq-cuda/src/rowcand.rs       |  2 +-
 llvq-llm/src/device.rs         | 10 ++++++----
 llvq-llm/src/fused.rs          |  8 +++++---
 llvq-llm/src/fused_cuda.rs     | 23 +++++++++++++----------
 llvq-llm/src/model.rs          | 14 +++++++++-----
 llvq-llm/src/persist4e_cuda.rs | 23 ++++++++++++-----------
 llvq-llm/src/persist_cuda.rs   |  7 +++++--
 llvq-llm/src/served.rs         | 10 ++++++----
 llvq-quant/src/gptq.rs         |  4 ++--
 llvq-quant/src/linalg.rs       |  7 +++++--
 llvq-quant/src/quantizer.rs    | 13 ++++++++-----
 llvq-runtime/src/engine.rs     |  9 ++++++---
 llvq-runtime/src/main.rs       |  2 +-
 llvq-search/src/lib.rs         | 12 ++++++------
 llvq-search/src/pack.rs        |  3 ++-
```

(chiffres pris avant le dernier reflow de `gptq.rs` et `quantizer.rs`, qui ne change que des retours à la ligne de commentaire)

`nice -n 10 cargo check -p llvq-search -p llvq-quant -p llvq-artifact -p llvq-llm -p llvq-runtime --all-targets -j 8` : ROUGE, deux fois, pour une cause hors de mes fichiers :

```
error: lifetime may not live long enough
   --> llvq-llm/src/sealed.rs:725:13
error: could not compile `llvq-llm` (lib test) due to 2 previous errors
```

`llvq-llm/src/sealed.rs` est le fichier de l'autre agent, en cours d'édition (56 lignes ajoutées). Contrôles de mes fichiers à côté :

- `nice -n 10 cargo check -p llvq-search -p llvq-quant -p llvq-artifact -p llvq-runtime -p llvq-cuda --all-targets -j 8` : `Finished`, aucun avertissement.
- `nice -n 10 cargo check -p llvq-llm --lib --bins -j 8` : `Finished`, aucun avertissement.

La commande complète est à relancer par la session principale une fois `sealed.rs` réparé.

`nice -n 10 cargo test -p llvq-runtime --lib keep_alive -j 8` :

```
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 17 filtered out; finished in 0.00s
```

Sous le mutant (`secs(n * 60.0)`) :

```
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 17 filtered out; finished in 0.00s
```
