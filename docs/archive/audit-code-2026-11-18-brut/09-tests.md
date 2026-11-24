# Tests et portes : constats
Périmètre réellement lu : `.github/workflows/ci.yml` (en entier) ; `llvq-llm/src/served.rs` (l. 505-1124), `llvq-llm/src/session.rs` (l. 70-297), `llvq-llm/src/served_decode.rs` (l. 120-230), `llvq-llm/src/sealed.rs` (l. 150-180, 690-710), `llvq-llm/src/persist4e_cuda.rs` (en-tête, l. 1300-1430, liste des `bail!`), `llvq-llm/src/fused_cuda.rs` (l. 689-800), `llvq-runtime/src/engine.rs` (l. 40-240, 300-395, 1540-1617), `llvq-runtime/src/api.rs` (l. 1020-1116) ; tests lus en entier : `llvq-llm/tests/served_unit.rs`, `session.rs`, `env_registry.rs`, `cpu_tetra.rs` (l. 1-235), `served_config.rs` (l. 150-380, 650-990), `hfpack.rs` (l. 160-210, 585-645), `llvq-cuda/tests/pers_matches_served.rs` (l. 90-315), `golay70_decoder_matches_rust.rs` (l. 380-450), `llvq-artifact/tests/common/mod.rs` ; `ops/jobs/paper-served.sh` (en entier), `served-v9.sh`, `chain-prefill-served.sh`, `verify-tree-n2.sh` (portes), `ops/run.py` (`cmd_selftest`, `cmd_oracle`, `cmd_bench`), `ops/mlx_tetra/test_served.py` (en-tête et `main`), `ops/check_image_lists.py` (fin) ; `llvq-tetra/tests/test_metal.py` et `test_fused.py` (en-têtes). Survolés par grep : tous les `#[ignore]`, `cfg_attr(debug_assertions, ignore)`, `#![cfg]` des dossiers `tests/`, les retours anticipés dans les tests, `set -euo pipefail` et `|| true` des 163 scripts de `ops/jobs/`, `unsafe` dans les cibles non-lib des six crates `forbid`.
Non couvert : les 1 300 tests en ligne des crates cœur (core, search, quant, artifact, bench) un par un ; `llvq-metal/tests` ; `tree_verify_host.rs` ; `kvq.rs` et `batch.rs` au-delà de la liste de leurs fonctions ; les tests Python de `ops/llvqtune` au-delà des `importorskip` ; la justesse des tolérances numériques des bancs (catégorie NUM).
Bilan : 0 critiques, 1 majeures, 8 mineures, 4 cosmétiques

Rappels `CONNU-NON-CORRIGÉ` (une ligne chacun) :
- CONNU-NON-CORRIGÉ, `docs/audit-architecture-2026-11.md` §0 point 6 et lot 8 : la CI ne construit ni `llvq-cuda` ni `llvq-metal`, et ne lance pas les 49 tests Python de `llvq-tetra` (le job `python` de `ci.yml:142-159` ne couvre que `llvqtune` sans torch, `test_served.py` et `check_image_lists.py`).
- CONNU-NON-CORRIGÉ, `docs/audit-architecture-2026-11.md` §3.5 : 43 tests de `llvq-llm` sous `#![cfg(any(target_os = "macos", feature = "cuda"))]` (attn_kernel, batch_gemm, batch_mma, dflash_*, proj_q4_wide, rotplan, served_unit, served_unit_copy, et `served_config.rs:868`) disparaissent sans bruit du job `test` Linux.
- CONNU-NON-CORRIGÉ, `docs/audit-architecture-2026-11.md` §3.5 et lot 13 : les comparaisons `same_bits` entre formes d'appel différentes sont toujours là, `llvq-llm/src/model.rs:4376-4377`, `4417`, `4514-4515`, `4964-4965`.
- CONNU-NON-CORRIGÉ, `docs/runtime-audit-2026-11.md` (« Five route tests depend on timing ») : `llvq-runtime/tests/server.rs:139` et ses voisins dorment toujours 50 à 1 500 ms (`server.rs:346`, `470`, `475`, `491`).

### TST-01 La comptabilité de session du décodage 4E servi n'a aucun test hors carte
- Gravité : majeure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/session.rs:210-214`, `225-271` ; `llvq-llm/src/served_decode.rs:122-131` ; `llvq-llm/tests/session.rs` (les 9 tests) ; `llvq-runtime/tests/server.rs:50`, `batch_server.rs:96`
- Fait : la branche `if dec.is_persistent()` de `Session::answer_fed` tient `ids` et la longueur des caches après chaque lancement 4E (`self.ids.push(last)`, `made[..made.len().saturating_sub(1)]`, `c.set_len(self.ids.len())`). `Decoder::is_persistent` rend `false` hors `cfg(all(target_os = "linux", feature = "cuda"))`. Tous les tests de `session.rs` et du runtime passent `Decoder::eager()`. Le refus « the persistent decode is greedy » (`session.rs:210-214`) n'est jamais atteint non plus. Les portes de job (`served-v9.sh:169-193`) comparent 256 jetons d'un seul tour au chemin eager, jamais un second tour d'une conversation.
- Scénario : un changement de la ligne 249 (par exemple `&made[..]`) met dans `ids` un jeton dont le cache ne tient pas la position. Le tour suivant de `llvq serve` ou de `bin/chat` sous la config v9 lit alors un cache décalé d'une position. Aucun test du Mac ni de la CI ne le voit ; seule une conversation à deux tours sur carte le montrerait.
- Touche : chemin servi
- Correction proposée : sortir la boucle 4E de `answer_fed` derrière une fermeture `(pos, tok, k) -> (made, stopped)`, comme `chunked_greedy` l'est déjà (`served_decode.rs:507`), et la tester sur le Mac avec un faux moteur qui rend des jetons fixés : `ids`, longueurs de cache, arrêt, annulation, puis un second tour comparé au chemin eager.
- Effort : M
- Risque de la correction : faible. Refactorisation hôte pure ; aucun format, aucun noyau épinglé ; les jetons servis ne bougent pas si la fermeture appelle le même `continue_greedy`.
- Vérification proposée : mutant `llvq-llm/src/session.rs:249`, remplacer `&made[..made.len().saturating_sub(1)]` par `&made[..]`, puis `cargo test -p llvq-llm --test session` : il survit (attendu).

### TST-02 La porte servie `Served::read` et celle du runtime ne sont exercées par aucun test
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/served.rs:706-711` ; `llvq-runtime/src/engine.rs:112-113` ; appelants `fusedrun.rs:2125`, `mmlu.rs:531`, `gsm8k.rs:117`, `chat.rs:80`, `cpurun.rs:42`, `batchrun.rs:79` et cinq autres binaires
- Fait : `check_env` et `Served::of` sont testés séparément, chacun sur des fermetures (`served_config.rs:276-341`, `656-669`). La seule jonction qui les enchaîne pour onze binaires, `Served::read`, n'est appelée par aucun test : le commentaire `served.rs:702-705` le dit voulu. `DeviceLoader::load` du runtime refait l'enchaînement à la main et aucun test ne l'appelle (les tests chargent `TinyLoader`, `CountingLoader` ou `BatchLoader`).
- Scénario : on supprime `check_env(...)` à `served.rs:708-709`, ou à `engine.rs:112-113`. Toute la suite passe. Un run servi accepte alors `LLVQ_TILE_BLOCKS` ou `LLVQ_PERS_4E` à côté de `LLVQ_CONFIG` en silence, et sa ligne de provenance ment.
- Touche : chemin servi
- Correction proposée : un test qui écrit une config dans un dossier temporaire et appelle `Served::read` avec un sous-processus (`CARGO_BIN_EXE_fusedrun` ou un petit binaire de test) sous `LLVQ_TILE_BLOCKS=64`, et attend le refus par nom. Pour le runtime, extraire de `DeviceLoader::load` une fonction pure `served_of(text, lookup)` et la tester.
- Effort : S
- Risque de la correction : nul. Test seulement.
- Vérification proposée : mutant `served.rs:708-709` supprimé, puis `cargo test -p llvq-llm --test served_config` : il survit.

### TST-03 Les fixtures Python de `hfpack` ne sont jamais comparées à ce que le packer écrit aujourd'hui
- Gravité : mineure
- Confiance : PLAUSIBLE (l'absence de comparaison est lue ; une dérive actuelle n'est pas mesurée)
- Où : `llvq-llm/tests/hfpack.rs:185-202` (`a_sealed_file_packs_into_a_described_directory`), `590-644` (`the_mini_fixture_describes_a_whole_qwen3_layer`) ; `llvq-tetra/tests/fixtures/tiny`, `fixtures/mini`
- Fait : les deux tests packent un fichier scellé synthétique et, seulement si `LLVQ_HF_FIXTURE` ou `LLVQ_HF_MINI_FIXTURE` est posé, copient la sortie dans le dossier de fixtures. Sans la variable, rien ne compare la sortie fraîche aux octets commités. Le commentaire `hfpack.rs:186-187` affirme pourtant « not a copy free to drift from it ». Le dernier commit des fixtures date du 2026-10-02 (`0708569`), `llvq-llm/src/hfpack.rs` a bougé le 2026-10-09 (`567f6fe`). À l'inverse, `rotation_matches_rust.rs:440-470`, `proj_q4_wide.rs:160-190` et `openapi.rs:56-70` comparent bien au fichier commité quand la variable d'écriture est absente.
- Scénario : `hfpack` change un champ de `quantization_config` que les tests Rust n'assertent pas. La suite Rust passe sur la nouvelle sortie, la suite `llvq-tetra` passe sur l'ancienne fixture, et le lecteur publié sur PyPI n'a jamais lu le nouveau format.
- Touche : format
- Correction proposée : sans variable, comparer chaque fichier de `out` à son homologue commité (octet pour octet, ou au digest de `llvq-digest.json`), comme le font les trois tests cités.
- Effort : S
- Risque de la correction : nul. Test seulement ; si la comparaison échoue aujourd'hui, c'est la fixture qu'il faut régénérer et la suite `llvq-tetra` qu'il faut relancer.
- Vérification proposée : `LLVQ_HF_FIXTURE=$TMPDIR/tiny cargo test -p llvq-llm --test hfpack a_sealed_file_packs_into_a_described_directory`, puis `diff -r $TMPDIR/tiny llvq-tetra/tests/fixtures/tiny` ; idem pour `mini`.

### TST-04 Un test d'archive scellée saute encore en silence, contre la règle 10
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-cuda/tests/golay70_decoder_matches_rust.rs:399-414` (`the_sealed_artifact_decodes_identically_through_golay70`)
- Fait : le test porte `cfg_attr(debug_assertions, ignore)` et non un `#[ignore]` inconditionnel. Sans `$HOME` ou sans `~/llvq-q4b.llvq`, il écrit `SKIP` et fait `return`, donc il rend `ok`. C'est la forme que `llvq-artifact/tests/common/mod.rs:43-50` a retirée des huit autres balayages, et que `docs/METHODE.md` §4 interdit. Il ignore aussi `LLVQ_SEALED_ARTIFACT`.
- Scénario : `cargo test --release` (sans `--include-ignored`) sur une machine sans l'archive rapporte ce test vert ; sur la machine qui l'a, il lance 150 M blocs dans la boucle rapide release.
- Touche : banc
- Correction proposée : `#[ignore = "..."]` inconditionnel et le chemin par `sealed_artifact_path()` (copier le module `common` ou le rendre partagé), qui échoue en nommant le fichier.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : `HOME=/nonexistent cargo test -p llvq-cuda --test golay70_decoder_matches_rust -- --ignored the_sealed_artifact_decodes_identically_through_golay70` : rapporte `ok` aujourd'hui.

### TST-05 Le test qui dit que les deux variables RESTORE se refusent ne teste pas le refus
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/sealed.rs:157-170` (`RestoreF16::from_env`), `693-704` (`the_two_restore_variables_select_a_precision_and_refuse_each_other`)
- Fait : le test n'appelle jamais `from_env`. Il vérifie `parse`, `at` et `describe_full`. Le bras `(false, false) => Err("... both set ...")` n'est atteint par aucun test, alors que `CLAUDE.md` liste ce refus.
- Scénario : un mutant qui fait gagner l'une des deux variables passe toute la suite ; un run `mmlu` avec les deux posées lirait alors un bras ambigu.
- Touche : banc
- Correction proposée : extraire `from_lookup(lookup: impl Fn(&str) -> Option<String>)`, comme `check_env`, et tester les quatre combinaisons, dont le refus par nom.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : mutant `sealed.rs:164`, remplacer le bras `(false, false) => Err(...)` par `(false, false) => Self::parse(&f16),`, puis `cargo test -p llvq-llm --lib sealed::` : il survit.

### TST-06 « L'unité servie compile » parse une liste recopiée qui n'est pas l'unité servie
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/tests/served_unit.rs:50-51` (`BASE`), `144-151` (`served_names`), `153-177` ; `llvq-llm/src/fused_cuda.rs:713-779`
- Fait : l'ordre réel des parties vit dans `fused_cuda.rs`, qui ne compile que sous Linux et `cuda`. Le test en garde une copie à la main. Cette copie omet la source d'embarquement que les configs servies chargent (`embed` vaut `q4` pour les trois configs v9, donc `emb_q4.cu` est ajouté à `fused_cuda.rs:721-725` et `750-755`, entre les plans et `tv_q4_h.cu`). Les tests de noms de `fused.rs:2817-2837` couvrent les collisions de symboles, pas la compilation de l'unité assemblée.
- Scénario : `fused_cuda.rs` change l'ordre ou ajoute une partie ; le test continue de parser son ancienne liste et reste vert. Une garde `#ifndef` cassée entre `emb_q4.cu` et une source voisine n'est vue que par NVRTC, sur carte.
- Touche : chemin servi
- Correction proposée : déplacer la liste dans une fonction pure de `fused.rs` (`served_unit_names(layout, emode, int4, tiled)`), l'appeler depuis `fused_cuda.rs` et depuis `served_unit.rs`, et parser l'unité v9 complète.
- Effort : S
- Risque de la correction : faible. Si la fonction rend la même liste dans le même ordre, le texte NVRTC et son sha256 ne bougent pas ; à vérifier par la ligne `NVRTC source` d'un run, sinon c'est une unité épinglée qui bouge.
- Vérification proposée : lecture suffisante pour l'écart ; pour la correction, comparer le sha256 imprimé à `fused_cuda.rs:790-793` avant et après.

### TST-07 Le trou `forbid(unsafe_code)` : 98 racines de crate non couvertes, et la CI ne vérifie que les `lib.rs`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `.github/workflows/ci.yml:127-135` ; cibles de `llvq-core`, `llvq-search`, `llvq-quant`, `llvq-artifact`, `llvq-bench`, `llvq-runtime` ; aucun `[lints]` dans leurs `Cargo.toml` ni `[workspace.lints]` dans `Cargo.toml`
- Fait : l'attribut `#![forbid(unsafe_code)]` d'un `lib.rs` ne couvre que la lib. Hors lib : 50 fichiers de `tests/` (1, 8, 9, 20, 7, 5), 28 exemples (26 de `llvq-bench`, 2 de `llvq-runtime`) et les 20 binaires de `llvq-bench/src/bin`. Trois seulement déclarent l'attribut eux-mêmes (`llvq-quant/tests/schur.rs:1`, `llvq-bench/examples/rowshrink.rs:85`, `hstats.rs:39`). Aucun ne contient `unsafe` aujourd'hui (grep). L'étape CI grep sept fichiers, `lib.rs` et `llvq-runtime/src/main.rs`.
- Scénario : un `unsafe` ajouté dans un binaire de `llvq-bench` ou un test de `llvq-artifact` passe clippy et la CI ; la carte de `CLAUDE.md` dit pourtant ces crates en `forbid`.
- Touche : outillage
- Correction proposée : dans les six `Cargo.toml`, `[lints.rust] unsafe_code = "forbid"`. Une table `[lints]` de paquet s'applique à toutes ses cibles (lib, bins, tests, exemples). `[workspace.lints]` seul ne change rien : chaque membre doit écrire `[lints] workspace = true`, et `llvq-cuda`, `llvq-metal`, `llvq-llm` ne peuvent pas l'adopter avec `forbid`, donc la table par paquet est plus simple et équivalente. L'étape grep de la CI peut alors être retirée.
- Effort : S
- Risque de la correction : faible. Aucune cible concernée ne contient `unsafe` ; une macro externe qui en émettrait ferait échouer la compilation, ce que la lib supporte déjà. Décision d'opérateur en attente selon `CLAUDE.md`.
- Vérification proposée : après l'ajout, `cargo clippy -p llvq-bench --all-targets` ; mutant : un `unsafe {}` dans `llvq-bench/src/bin/lcap.rs` doit casser la compilation.

### TST-08 Deux tests lisent des fichiers que `paths-ignore` exclut de la CI : ils peuvent rougir sur `main` sans que personne le voie
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `.github/workflows/ci.yml:59-63` ; `llvq-llm/tests/env_registry.rs:58` (`every_quoted_variable_is_in_the_registry` lit `docs/ENV.md`) ; `llvq-runtime/src/library.rs:341-352` (`latest_is_the_4b_and_the_digests_are_the_readme_ones` inclut `configs/README.md`)
- Fait : la CI ignore `'**.md'` et `'docs/**'`, au push sur `main` comme en PR. Ces deux tests dépendent pourtant de deux fichiers Markdown.
- Scénario : une PR qui ne touche que `docs/ENV.md` (une ligne supprimée) ou `configs/README.md` (un digest réécrit) ne lance aucun job. Le test casse sur `main` et l'échec n'apparaît qu'au prochain push de code, attribué à ce code.
- Touche : outillage
- Correction proposée : retirer `docs/ENV.md` et `configs/README.md` de l'exclusion par des motifs négatifs (`'!docs/ENV.md'`, `'!configs/README.md'`) dans les deux listes `paths-ignore`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### TST-09 Les 38 tests Metal de `llvq-llm` ne tournent dans aucune boucle documentée
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/tests/attn_metal.rs:16` (6 tests), `q4_metal_candle.rs:17` (6), `rot_metal_tg.rs:8` (1), `tetra48_metal_candle.rs:49` (12), `tetra48_metal_rows.rs:21` (10), `llvq-llm/src/fused_metal.rs:2380` (3, module sous `lib.rs:50-51`) ; `llvq-llm/Cargo.toml` (feature `metal` non par défaut) ; `llvq-runtime/Cargo.toml:32-34`
- Fait : ces fichiers portent `#![cfg(all(target_os = "macos", feature = "metal"))]`. Aucune crate n'active `llvq-llm/metal` par défaut. Les commandes de test de `CLAUDE.md` (`cargo test`, `cargo test --release -- --include-ignored`) ne passent pas `--features metal`, la CI non plus, aucun script de `ops/` non plus. Ils couvrent le backend Metal que `llvq serve` sert sur Mac (le noyau Tetra et ses jumeaux de boucle, l'int4, l'attention Metal). Un commit du 2026-11-06 (`d37da2b`) montre qu'on les lance parfois à la main.
- Scénario : une régression d'un noyau Metal servi laisse `cargo test` et `cargo test --release -- --include-ignored` verts sur le Mac.
- Touche : chemin servi
- Correction proposée : ajouter `cargo test -p llvq-llm --features metal` aux commandes de `CLAUDE.md` et à la liste « suite complète avant tout commit de format ».
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : `cargo test -p llvq-llm --features metal --test tetra48_metal_rows` sur le Mac, pour savoir s'ils sont verts aujourd'hui.

### TST-10 `test_metal.py` dit échouer sans MPS et saute
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-tetra/tests/test_metal.py:3-5` contre `21-24` ; `llvq-tetra/tests/test_fused.py:21-24`, `26-31`
- Fait : la docstring promet « they skip nothing: a machine with no MPS device fails here by name », et `pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), ...)` saute tout le module. Dans `test_fused.py`, le même `skipif` saute aussi `test_invnorm_is_one_over_the_root_of_sixteen_m`, qui ne touche pas le GPU.
- Scénario : sur Linux, `pytest` rapporte ces modules `skipped` ; le lecteur de la docstring croit à un échec par nom.
- Touche : outillage
- Correction proposée : corriger la docstring, et marquer `skipif` test par test pour que les tests purement CPU tournent partout.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### TST-11 Une assertion de `keep_alive_reads_ollamas_forms` est vide
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-runtime/src/engine.rs:1612-1615`
- Fait : `"5"` est dans la liste `bad`, mais `assert!(bad == "5" || r.is_err())` l'accepte quel que soit le résultat. `parse_str("5")` rend 5 secondes (`engine.rs:338-340`).
- Scénario : aucun défaut ; l'assertion ne dit rien pour ce cas.
- Touche : runtime
- Correction proposée : sortir `"5"` de `bad` et écrire `assert_eq!(s("5"), KeepAlive::For(Duration::from_secs(5)))`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### TST-12 Deux `#[ignore]` sans raison écrite, qui sont des outils et non des tests
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-cuda/src/rowcand.rs:525-527` (`dump_rowcand_module`), `llvq-cuda/src/pers.rs:4785-4787` (`dump_tree_modules`)
- Fait : `#[ignore]` nu ; la raison est dans le commentaire de doc. Les deux écrivent sous `target/` et n'assertent presque rien. Les scripts `ops/rowcand-sass.sh:6`, `ops/tree-sass-probe.sh:4`, `ops/rc-regprobe.sh:7` les appellent par `--ignored`.
- Scénario : `--include-ignored` les lance et écrit des fichiers sous `target/` comme effet de bord de la suite complète.
- Touche : outillage
- Correction proposée : `#[ignore = "writes target/...: a tool for ops/..."]`, ou en faire un binaire.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### TST-13 `the_shipped_config_is_the_served_object` épingle une config qui n'est plus l'objet servi
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/tests/served_config.rs:153-176`
- Fait : le test lit `configs/qwen3-4b-tetra-q5.json` (embed `q8`). L'objet servi est `qwen3-4b-tetra-e4-4e-v9.json` (embed `q4`, 4E v9), couvert par `the_v9_config_is_the_v6_config_with_the_v9_rows` (`served_config.rs:781-836`).
- Scénario : un lecteur croit que le nom garde l'objet servi.
- Touche : doc
- Correction proposée : renommer en `the_q5_config_is_the_paper_object` ou équivalent.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

## Ce qui ne tourne jamais automatiquement (état au `925930d`)

- CI Linux : clippy et tests de 7 crates, debug des 5 crates cœur, `fast-linalg`, Python sans torch. Pas `--include-ignored`, pas `llvq-cuda` (247 tests selon l'audit d'architecture, dont `pers_matches_served.rs`, 74 tests et leurs mutants intégrés), pas `llvq-metal`, pas les 43 tests `macos|cuda` de `llvq-llm`, pas `llvq-tetra`, pas les tests torch de `llvqtune`.
- Mac seulement, à la main : tout ce qui précède, plus les 38 tests `feature = "metal"` (TST-09), plus 63 tests `cfg_attr(debug_assertions, ignore)` de `llvq-cuda/tests` qui ne tournent qu'en `--release`.
- Carte seulement : `persist4e_cuda.rs` (2 709 lignes, aucun test, tous ses `bail!` de lancement et d'installation), la branche 4E de `Session` (TST-01), l'assemblage NVRTC réel (TST-06).
- Portes de job : les 163 scripts de `ops/jobs/` commencent par `set -euo pipefail`, et `cmd_bench` le préfixe au corps (`ops/run.py:1127`). Les `|| true` lus dans `paper-served.sh`, `served-v9.sh`, `chain-prefill-served.sh` et `verify-tree-n2.sh` sont sur des lignes de résumé ou des lectures déclarées non bloquantes (`paper-served.sh:106`), pas sur une porte. Le `grep -q MATCH` de l'oracle ne peut pas confondre un échec : `oracle.rs:82-90` n'imprime `MATCH` qu'au succès et sort en erreur sinon. Je n'ai pas trouvé de porte de job dont l'échec soit avalé.

## Candidats au lot 1

TST-02, TST-03, TST-04, TST-05, TST-07, TST-08, TST-09, TST-10, TST-11, TST-12, TST-13
