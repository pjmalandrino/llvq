# Lot 1, refus manquants des harnais et tests qui ne testent rien : journal

Racine : `.claude/worktrees/audit-code`, branche `audit/code-2026-11-18`, rien de commité.

## Corrigé

| ID | fichiers | ce qui change, en une ligne | vérification faite |
|---|---|---|---|
| BUG-06 | `llvq-llm/src/bin/mmlu.rs`, `llvq-llm/src/bin/ppl.rs` | `mmlu` refuse par nom un `limit` illisible, vide ou nul ; `ppl` refuse un `ctx` ou `max_windows` illisible, vide ou nul, et un corpus autre que `wikitext2` ou `c4` (un positionnel absent garde son défaut). | `git grep` dans `ops/jobs/` et `ops/run.py` : tous les appels sont `mmlu <f> cuda`, `mmlu <f> cuda 40`, `ppl 4096 12 cuda <f>`, aucun corpus passé, aucun `0`. Tests neufs `a_malformed_limit_is_refused_not_read_as_the_census` (bin mmlu) et `a_malformed_positional_is_refused_not_defaulted` (bin ppl). Mutants : `ensure!(n > 0)` retiré dans `mmlu` (rouge), `s.trim().parse()` dans `mmlu` (rouge), `Some("C4")` accepté dans `ppl` (rouge), `ensure!(n > 0)` retiré dans `ppl` (rouge) ; chacun défait, vert. |
| BUG-05 | `llvq-llm/src/bin/seal.rs` | `seal` refuse `LLVQ_MODEL` absent ou vide (plus de défaut 0.6B) et un `<in>.state` dont `model` diffère de `LLVQ_MODEL` ; sans `.state`, ou sans clé `model`, il scelle sur `LLVQ_MODEL` et l'imprime. Fonction pure `checkpoint_name`. | `git grep` : les quatre appels de `ops/jobs/` (`encode-14b.sh:412`, `guided-clip-encode-card-4b.sh:229`, `guided-encode-4b.sh:120`, `guided-encode-card-4b.sh:221`) posent `LLVQ_MODEL="$MODEL"`, la même variable que le `smoke` qui a écrit l'artefact ; `smoke.rs:853` écrit `model = <LLVQ_MODEL tel quel>`, donc la comparaison de chaînes ne refuse aucun appel existant. Test `the_checkpoint_is_required_and_must_be_the_encoded_one`. Mutants : `repo.starts_with(encoded)` accepté (rouge) ; défaut 0.6B rétabli (vert au premier essai, test renforcé pour exiger le message « is required » avec et sans `.state`, puis rouge). |
| BUG-03 | `llvq-llm/src/gsm8k.rs`, `llvq-llm/src/bin/gsm8kpair.rs` | `served_header` écrit `prefill_attention` sous la seule config qui le nomme ; `Dump::prefill()` (absent = `eager`) et `Dump::decode_cache()` (absent = `f16`) ; `gsm8kpair` imprime une ligne `config | prefill | decode cache` par bras. `pair` ne change pas. | Test `the_prefill_keys_are_written_only_when_they_differ_and_read_as_eager_when_absent` : un en-tête ancien (sans `config`, `prefill`, `decode_cache`) se lit `eager`, `f16`, `?`, et s'apparie comme avant, y compris avec un en-tête neuf. Les configs existantes n'écrivent aucune clé de plus. `ops/duel/analyze.py` lit `gsm8kpair` par regex sur `accuracy A =`, `A − B =`, `stopped at the cap` : intacts. Mutants : la clé `prefill_attention` non écrite (rouge), défaut `chain` au lieu de `eager` (rouge). |
| BUG-04 | `llvq-llm/src/sealed.rs` | `RestoreF16::describe` écrit la fenêtre (`down_proj@0-11`), et `describe_full` aussi. Libellé seulement. | `mmlupair` ne lit `# model=` que pour l'imprimer (`mmlupair.rs:149`, `:655`) : rien de ce qu'il accepte ou refuse ne change. Les portes `depth-restore-4b.sh:121` et autres lisent `($M matrices, $W weights)` et `restored` : intacts. Test `a_restored_window_is_named_in_the_label`. Mutant : fenêtre omise (rouge). |
| BUG-07 | `llvq-llm/src/bin/chat.rs` | `LLVQ_CHAT_SEED` lu en `u64`, `LLVQ_CHAT_MAX_NEW` en `usize` (`env_parse` générique) ; une valeur illisible, négative ou flottante est refusée par nom. | `git grep` : aucun script ne pose ces deux variables. Lecture suffisante (constat). |
| BUG-10 | `llvq-llm/src/bin/mmlu.rs` | La ligne `MMLU (micro, = paper) = … [kv …]` imprime `cfg.kv` sur le bras servi, `LLVQ_KV` sinon. Format inchangé. | `git grep -nE "MMLU \(micro"` : `errmap-4b-t-sweep.sh:28` et `q5-alloc-int4.sh:62` lisent la ligne par son préfixe ; seule la valeur peut changer, et seulement sous une config `kv: q8` (aucune livrée). |
| BUG-08 | `llvq-llm/src/bin/gsm8k.rs` | Commentaire seulement : la coupe par `MAX_ROWS` n'est pas neutre sous l'attention `kernel`, et `Session::feed` coupe à `prefill_rows()`. Le code ne bouge pas. | Relu contre `model.rs:2220-2245`, `session.rs:159`, `MAX_ROWS = 256`. |
| TST-05 | `llvq-llm/src/sealed.rs` | `RestoreF16::from_lookup(lookup)` extrait de `from_env` sans changer le comportement ; le test existant teste les quatre combinaisons, dont le refus des deux variables ensemble. Le commentaire de `parse` est rendu à `parse`. | Mutant du vérificateur : bras `(false, false) => Self::parse(&f16),` → `the_two_restore_variables_select_a_precision_and_refuse_each_other` rouge ; défait, vert. |
| TST-13 | `llvq-llm/tests/served_config.rs` | L'ancien test devient `the_q5_config_is_the_object_arbitrated_on_2026_09_08` (mêmes assertions, commentaire corrigé : le 56,95 n'est pas ce fichier selon `configs/README.md:40`) ; un nouveau `the_shipped_config_is_the_served_object` lit dans `llvq-runtime/library.json` la config CUDA du 4B (`qwen3-4b-tetra-e4-4e-v9-chain.json`, ETAT §2) et vérifie ses valeurs : tetra48, q4, rot_share 1, fuse 0 et pas de noyau segmenté, kv f16, attention kernel, decode persistent, rows v9, prefill chain, prefill_attention grouped, pas de batch. | Les configs et `library.json` sont interdits en écriture, donc le mutant porte sur le chemin lu : la v9 sans chaîne à la place (rouge), la config e4 étendue (rouge) ; défait, vert. |
| TST-02 | `llvq-llm/tests/served_read_env.rs` (neuf) | Un fichier, un test, seul dans son binaire : `Served::read` sur la config v9 servie, environnement nettoyé, puis `LLVQ_EMBED=q4` accepté, `LLVQ_EMBED=q8` refusé (« contradicts »), `LLVQ_TILE_BLOCKS=64` refusé, `LLVQ_PERS_4E=1` refusé (« measurement mode »). | Le mutant du vérificateur (supprimer `served.rs:708-709`) touche le fichier d'un autre agent : non appliqué. Émulé dans mon fichier : `Served::read` remplacé par `read_file` + `of` (c'est-à-dire `read` sans `check_env`) → rouge ; défait, vert. Partie runtime (`DeviceLoader::load`) laissée, voir plus bas. |
| QUA-58 | `llvq-llm/tests/softmax_whole.rs` (neuf) | Un test seul dans son binaire : `softmax_form()` vaut « in place », puis « whole tensor (LLVQ_SOFTMAX_WHOLE=1) » après `set_softmax_whole_for_measurement(true)`, puis revient. Aucun code de production. | Vert. Mutant NON lancé : il est dans `model.rs` (`usize::MAX` → `IN_PLACE_BYTES` à `:833`), fichier de l'autre agent. À lancer par la session principale : `cargo test -p llvq-llm --test softmax_whole` doit rougir. |
| QUA-53 | `llvq-llm/src/bin/fusedrun.rs` | La liste en ligne des modes refusés à côté de `LLVQ_CONFIG` devient une boucle sur `llvq_llm::served::MEASUREMENT_MODES` (déjà `pub`). Même message. Effet : `LLVQ_TIME_EVENTS` et `LLVQ_BENCH_ARMS` sont aussi refusés, ce que `CLAUDE.md` dit déjà. `ALONE` intact. | `git grep` : `fusedrun` ne lit pas ces deux noms ; dans `ops/jobs/`, `LLVQ_BENCH_ARMS` n'est posé qu'en ligne devant `planesbench`, jamais exporté. Linux seulement : relu deux fois, non compilé ici. |
| TST-04 | `llvq-cuda/tests/golay70_decoder_matches_rust.rs` | `#[ignore = "..."]` inconditionnel et copie locale de `sealed_artifact_path()` (`LLVQ_SEALED_ARTIFACT`, puis `$HOME/llvq-q4b.llvq`, échec qui nomme le fichier) à la place de `SKIP` puis `return`. | Avant : `ok` sans l'archive. Après : `LLVQ_SEALED_ARTIFACT=/nonexistent/q4b.llvq` → FAILED « the sealed archive is not on this machine: /nonexistent/q4b.llvq (from LLVQ_SEALED_ARTIFACT) » ; `HOME` vide (chaîne d'outils gardée) → FAILED, même message, chemin `$HOME/llvq-q4b.llvq`. Sans `--ignored` : `ignored`. Ce fichier compile et tourne sur le Mac (pas de `cfg` Linux, dev-dépendances portables). |
| NUM-02 | `llvq-bench/src/bin/rtbits.rs` | La ligne « f32 (published) » s'appelle « f32 » ; un commentaire dit que le b/param publié est la ligne f16, colonne embed q4 ; la ligne noyau f32 dit « the accounting of published kernel b/weight » (alignement des points gardé) ; les deux commentaires voisins disent laquelle des deux chaque chiffre publié emploie. Aucun calcul touché. | `git grep -nF "f32 (published)"` et `"f16 (served)"` dans `ops/`, `paper2/`, `docs/`, `.github/` : seulement des journaux de `docs/mesures/`. Aucun script ne lit ces libellés ; `encode-14b.sh` ne fait que `tee`. Les deux tests qui lancent `rtbits` (`int4_refusals`, `tetra_refusals`) cherchent « kernel b/weight » et « NO CLASS STATISTIC », toujours imprimés : verts. 2,7320 relu sur `sealed-4b-2026-09-23-brut/rtbits-sealed.txt:14`, ligne f16, colonne q4. |

## Laissé de côté

| ID | raison |
|---|---|
| BUG-01, BUG-02, BUG-09 | Hors de mon mandat : ils changent un instrument de mesure, décision de l'opérateur. |
| BUG-03, avertissement `max_new` | Le vérificateur montre que `max_new` est déjà imprimé sur les lignes A et B ; ajouter un avertissement dépasse « impression et clé optionnelle ». |
| BUG-08, découpe | Découper par `model.prefill_rows()` change le code du harnais : seul le commentaire est corrigé, comme demandé. |
| TST-02, partie runtime | `DeviceLoader::load` est dans `llvq-runtime/src/engine.rs`, fichier de l'autre agent ; en extraire `served_of` sort de mon périmètre. |
| TST-02 et QUA-58, mutants réels | Ils sont dans `served.rs` et `model.rs`, fichiers de l'autre agent. TST-02 est émulé dans mon test ; QUA-58 reste à faire (commande ci-dessus). |
| `docs/ENV.md` | La ligne `LLVQ_SEALED_ARTIFACT` dit « `llvq-artifact` tests » ; elle vaut aussi désormais pour le balayage Golay70 de `llvq-cuda`. Pas mon fichier. |
| `served.rs:702-705` | Le commentaire de `Served::read` dit qu'aucun test ne doit l'appeler et cite `the_shipped_config_is_the_served_object` ; `served_read_env.rs` l'appelle maintenant, dans un binaire isolé. Pas mon fichier : à aligner par l'agent des commentaires ou la session principale. |

## Constat faux ou déjà corrigé

| ID | preuve |
|---|---|
| (aucun) | Les douze constats corrigés ont été revérifiés dans le code avant correction. TST-04 : « ce fichier ne compile pas sur ce Mac » est faux, il compile et tourne (`cargo test -p llvq-cuda --test golay70_decoder_matches_rust`). |

## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)

- `llvq-llm/src/bin/fusedrun.rs` : la modification est dans le bloc `cfg(all(target_os = "linux", feature = "cuda"))`, jamais compilé ici. `CUDARC_CUDA_VERSION=12040 cargo clippy --target x86_64-unknown-linux-gnu -p llvq-llm --features cuda --bin fusedrun`.
- `llvq-cuda/tests/golay70_decoder_matches_rust.rs` : compilé, clippy et lancé sur le Mac ; la compilation croisée reste prudente.

## Incident

Mon script de mutants a mal défait deux mutants dont le remplacement était vide (BUG-06, `ensure!(n > 0)`) : la ligne a été réinsérée en tête de `llvq-llm/src/bin/mmlu.rs` et de `llvq-llm/src/bin/ppl.rs` au lieu de son site, pendant quelques minutes. Remis à la main, vérifié par `git diff` puis par les tests ci-dessous. Si un autre agent a compilé `llvq-llm` dans cette fenêtre, il a vu une erreur de compilation dans ces deux binaires ; elle n'existe plus. Le script refuse désormais un remplacement vide.

## Commandes lancées et leur résultat (les lignes `test result:` telles quelles)

Toutes en `nice -n 10`, `-j 8`, debug.

- `cargo check -p llvq-llm -p llvq-bench --all-targets` : `Finished`, zéro avertissement (après correction d'une erreur de durée de vie dans mon test de `sealed.rs`).
- `cargo clippy -p llvq-llm -p llvq-bench --all-targets` : `Finished`, zéro avertissement.
- `cargo clippy -p llvq-cuda --test golay70_decoder_matches_rust` : `Finished`, zéro avertissement.
- `cargo test -p llvq-llm --lib sealed::` : `test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 217 filtered out; finished in 0.01s`
- `cargo test -p llvq-llm --lib gsm8k::` : `test result: ok. 21 passed; 0 failed; 0 ignored; 0 measured; 215 filtered out; finished in 0.00s`
- `cargo test -p llvq-llm --test served_config` : `test result: ok. 44 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s`
- `cargo test -p llvq-llm --test served_read_env` : `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
- `cargo test -p llvq-llm --test softmax_whole` : `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
- `cargo test -p llvq-llm --test env_registry` : `test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s`
- `cargo test -p llvq-llm --bin mmlu` : `test result: ok. 21 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
- `cargo test -p llvq-llm --bin ppl` : `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
- `cargo test -p llvq-llm --bin seal` : `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
- `cargo test -p llvq-bench --test int4_refusals` : `test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.79s`
- `cargo test -p llvq-bench --test tetra_refusals` : `test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.90s`
- `cargo test -p llvq-cuda --test golay70_decoder_matches_rust` : `test result: ok. 0 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out; finished in 0.00s`
- `LLVQ_SEALED_ARTIFACT=/nonexistent/q4b.llvq cargo test -p llvq-cuda --test golay70_decoder_matches_rust -- --ignored the_sealed_artifact_decodes_identically_through_golay70` : `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.00s` (attendu : échec qui nomme le fichier)
- Même commande avec `HOME=<scratchpad>/nohome`, `CARGO_HOME` et `RUSTUP_HOME` gardés : `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.00s` (attendu)
- Mutants (chacun : édition, rouge, édition inverse vérifiée identique à l'octet, vert) :
  - `sealed.rs` bras `(false, false) => Self::parse(&f16),` : `test result: FAILED. 18 passed; 1 failed; 0 ignored; 0 measured; 217 filtered out; finished in 0.01s`
  - `sealed.rs` fenêtre omise de `describe` : `test result: FAILED. 18 passed; 1 failed; 0 ignored; 0 measured; 217 filtered out; finished in 0.01s`
  - `gsm8k.rs` clé `prefill_attention` non écrite : `test result: FAILED. 20 passed; 1 failed; 0 ignored; 0 measured; 215 filtered out; finished in 0.01s`
  - `gsm8k.rs` défaut `chain` : `test result: FAILED. 20 passed; 1 failed; 0 ignored; 0 measured; 215 filtered out; finished in 0.00s`
  - `mmlu.rs` sans refus de 0 : `test result: FAILED. 20 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
  - `mmlu.rs` `s.trim().parse()` : `test result: FAILED. 20 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
  - `ppl.rs` `C4` accepté : `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
  - `ppl.rs` sans refus de 0 : `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
  - `seal.rs` préfixe accepté : `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
  - `seal.rs` défaut 0.6B rétabli : `test result: ok. 1 passed; …` au premier essai (survivant), puis, test renforcé : `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
  - `served_config.rs` chemin lu = v9 sans chaîne : `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 43 filtered out; finished in 0.00s`
  - `served_config.rs` chemin lu = e4 étendue : `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 43 filtered out; finished in 0.00s`
  - `served_read_env.rs` `read` sans `check_env` : `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s`
