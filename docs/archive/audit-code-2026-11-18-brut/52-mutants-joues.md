# Mutants joués, file 51

Aucun vrai bug de production révélé. Aucun test rouge sans mutant. Aucun survivant inattendu. Aucun test renforcé.

Début : 2026-11-19 23:54 (CET). Fin : 2026-11-20 00:17. Durée : 23 minutes.

Arbre de départ : `git diff | md5` = `1d8b0d21920fb9f85c8773632d1a7dda`, 94 fichiers, 3 817 insertions, 523 suppressions.
Copie : `scratchpad/avant-mutants.patch`, même md5. Arbre de fin : `git diff | md5` = `1d8b0d21920fb9f85c8773632d1a7dda`.
Les fichiers non suivis n'ont pas été touchés (même liste avant et après).

Méthode : une édition par l'outil Edit, le test ciblé en debug derrière `when-idle.sh`, l'édition inverse, puis le md5 du
`git diff` du fichier comparé à sa valeur d'avant. Les journaux bruts sont dans le scratchpad de la session (`m<n>.log`,
`p<n>.log`).

Quand le mutant remet le texte de `HEAD`, la ligne n'apparaît pas dans `git diff` : l'application a été vérifiée par le
changement du md5 du fichier (n° 5, 12, 14, 25).

## Tableau

Commandes abrégées : `T` = `IDLE_LOAD_LIMIT=8 when-idle.sh nice -n 10 cargo test -j 8`, suivi de `-- <filtre> --test-threads=2`.

| n° | fichier:site | mutant | commande | résultat | verdict |
|---|---|---|---|---|---|
| 1 | `llvq-llm/src/session.rs:304` `answer_launches` | `&made[..made.len().saturating_sub(1)]` → `&made[..]` | `T -p llvq-llm --test session` | `test result: FAILED. 10 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out` (les deux tests `persistent_branch` annoncés) | tué |
| 2 | `session.rs:140` `truncate` | `if n == self.ids.len() { return Ok(()); }` après l'`ensure!` | idem | `test result: FAILED. 11 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (`a_failed_launch_leaves_caches_the_next_request_cuts_back`) | tué |
| 3 | `session.rs:83` `positions_held` | bras `Cat` → `c.len()` | idem | `test result: FAILED. 11 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (« store Cat: layer 0 kept 4 positions after a cut to zero ») | tué |
| 4 | `session.rs:142` `truncate` | `!= n` → `== n` | idem | `test result: FAILED. 10 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out` (les deux annoncés) | tué |
| 5 | `llvq-runtime/src/engine.rs:1319` chargement | `.unwrap_or_else(\|p\| Err(..))` → `.unwrap_or_else(\|p\| std::panic::resume_unwind(p))` (équivaut au retrait de `catch_unwind`, en une édition) | `T -p llvq-runtime --test engine_busy -- a_load_that_panics_...` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 4 filtered out` (« the request ended on None, not on the load's panic ») | tué |
| 6 | `engine.rs:688` après `feed_with` | `self.session.reset(&self.loaded.qwen);` retiré | `T -p llvq-runtime --test server -- a_failed_prefill_leaves_an_empty_cache_to_the_next_request` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 16 filtered out` | tué |
| 7 | `engine.rs:689` après `feed_with` | `self.memo.clear();` retiré seul | idem | `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 16 filtered out` | survivant annoncé |
| 8 | `engine.rs:1355` bras de panique du batch | boucle `for l in live.drain(..)` retirée | `T -p llvq-runtime --test batch_server -- a_panic_in_the_batched_step_ends_every_live_stream_with_an_error` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 8 filtered out` (« neither ended nor failed within 120 s », 120,09 s) | tué |
| 9 | `engine.rs:1358` bras de panique du batch | boucle `for job in pending.drain(..)` retirée | `T -p llvq-runtime --test batch_server` | `test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out` | survivant annoncé |
| 10 | `engine.rs:882` pas groupé raté | ligne `*inf.expires.lock()... = l.job.keep_alive.until(..)` retirée | `T -p llvq-runtime --test batch_server -- a_failed_batched_step_leaves_the_model_to_its_keep_alive` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 8 filtered out` (« outlived its keep-alive of one second ») | tué |
| 11 | `engine.rs:1439` `same_model` | `&& e.blob == blob` retiré | `T -p llvq-runtime --test engine_busy -- a_blob_written_again_under_the_same_manifest_loads_the_model_again` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 4 filtered out` (« the engine read from the removed file still serves ») | tué |
| 12 | `engine.rs:1439` `same_model` | `Manifest { modified_at: .., ..e.manifest.clone() } == *now` → `e.manifest == *now` | `T -p llvq-runtime --test engine_busy -- a_manifest_written_again_with_a_new_date_keeps_the_loaded_engine` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 4 filtered out` | tué |
| 13 | `llvq-runtime/src/store.rs:191` `blob_id` | `Some(BlobId { len: m.len(), modified: None, node: (0, 0, 0, 0) })` | `T -p llvq-runtime --test engine_busy -- a_blob_written_again_...` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 4 filtered out` (même message) | tué |
| 14 | `store.rs:295` `remove` | bloc `if let Err(e) = lock.try_lock() {..}` → `let _ = lock.try_lock();` | `T -p llvq-runtime --test pull -- rm_during_an_import_of_the_same_file_is_refused` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 7 filtered out` | tué |
| 15 | `store.rs:264` `manifests` | `Err(e) => unread.push(e)` → `Err(e) => return Err(e)` | `T -p llvq-runtime --test pull -- a_manifest_that_does_not_read_leaves_the_others_listed_and_removable` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 7 filtered out` (pull.rs:217, premier `list()`) | tué |
| 16 | `store.rs:311` `remove` | `if !held && unread.is_empty()` → `if !held` | idem | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 7 filtered out` (« a blob that a manifest may name was removed ») | tué |
| 17 | `llvq-runtime/src/chat.rs:98` `scan` | `shown.ends_with` / `shown.len() - k` → `text.…` | `T -p llvq-runtime --lib -- chat::` | `test result: FAILED. 6 passed; 1 failed; 0 ignored; 0 measured; 12 filtered out` (`Safe(4)` contre `Safe(3)`) | tué |
| 18 | `chat.rs:147` `TextStream::full`, UTF-8 coupé | `!tail.is_empty() && !tail.ends_with('\u{FFFD}')` → `!tail.is_empty()` | idem | `test result: FAILED. 6 passed; 1 failed; 0 ignored; 0 measured; 12 filtered out` (`the_stream_decodes_as_the_whole_list_at_every_id`) | tué |
| 19 | `llvq-runtime/src/api.rs:160` borne de `stop` | `> MAX_STOPS` → `> MAX_STOPS + 1` | `T -p llvq-runtime --test server -- a_stop_list_past_its_bounds_is_refused` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 16 filtered out` | tué |
| 20 | `api.rs:163` borne d'octets de `stop` | bloc `if let Some(s) = ... > MAX_STOP_BYTES` retiré | idem | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 16 filtered out` | tué |
| 21 | `api.rs:860` `generate` | avertissement `raw` + `system` retiré | `T -p llvq-runtime --test server -- one_object_when_stream_is_false_and_generate_takes_raw_and_templated_prompts` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 16 filtered out` | tué |
| 22 | `api.rs:689` `rm` occupé | `error(409, ..)` → `error(500, ..)` | `T -p llvq-runtime --test server --test pull` | `test result: ok. 8 passed; 0 failed; ...` et `test result: ok. 17 passed; 0 failed; ...` | survivant annoncé |
| 23 | `llvq-runtime/src/openai.rs:230` `HARMLESS` | `"logprobs"` remis dans la liste | `T -p llvq-runtime --test server -- the_openai_routes_answer_as_the_ollama_ones_in_openais_shapes` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 16 filtered out` (server.rs:429, `ask(true)`) | tué |
| 24 | `openai.rs:236` `warn_extra` | second `.filter(..)` retiré | idem | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 16 filtered out` (server.rs:431, `ask(false)`) | tué |
| 25 | `openai.rs:346` flux, `Event::Done` | bloc `if !ctx.warnings.is_empty() { fin["warnings"] = .. }` retiré | idem | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 16 filtered out` (server.rs:399) | tué |
| 26 | `llvq-artifact/src/sealed.rs:376` `read_raw` | `check_quant(&name, &dims, &q)?;` retiré | `T -p llvq-artifact --test hostile_files` | `test result: FAILED. 32 passed; 3 failed; 0 ignored; 0 measured; 0 filtered out` (groupe 0, charge courte, dims en débordement) | tué |
| 27 | `sealed.rs:353` largeur lue | `match` → `let bits = get_u32(r, "quant width")? as u8;` | idem | `test result: FAILED. 34 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (`a_raw_quant_width_is_checked_as_stored_not_narrowed`) | tué |
| 28 | `sealed.rs:237` `value_count` | `checked_mul` → `Some(acc.wrapping_mul(d))` | idem | `test result: FAILED. 34 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (`raw_dims_whose_product_overflows_are_refused_by_name`) | tué |
| 29 | `sealed.rs:255` `check_quant` | bloc `if q.group == 0` retiré | idem | `test result: FAILED. 34 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (« attempt to divide by zero ») | tué |
| 30 | `sealed.rs:342` compte f16 lu | `if data.len() != value_count(..)?` → `if false` | idem | `test result: FAILED. 34 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (`a_raw_f16_count_that_disagrees_with_its_dims_is_refused`) | tué |
| 31 | `sealed.rs:263` `check_quant` | `8 => n,` → `8 => n + 1,` | idem | `test result: FAILED. 34 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (`honest_raw_tensors_still_read_byte_for_byte`) | tué |
| 32 | `llvq-artifact/src/format.rs:1193` lecture, gain borné | bloc `if gain as usize >= n_cent` retiré | idem | `test result: FAILED. 34 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (`a_ball_gain_past_its_centroids_is_refused_on_read_not_a_panic`) | tué |
| 33 | `format.rs:1126` lecture, zéro centroïde | `if n_cent == 0 && ..` → `if false && n_cent == 0 && ..` | idem | `test result: FAILED. 34 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (`a_ball_record_with_blocks_and_no_centroid_is_refused_on_read`) | tué |
| 34 | `format.rs:556` `check_tail` | `let want = d_out * (d_in % DIM);` → `let want = d_in % DIM;` | idem | `test result: FAILED. 31 passed; 4 failed; 0 ignored; 0 measured; 0 filtered out` (dont `a_4b_shaped_record_and_its_tail_still_round_trip`) | tué |
| 35 | `format.rs:973` `code_stream` | `check_tail(..)?;` retiré | idem | `test result: FAILED. 34 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (`a_tail_count_off_the_dimensions_is_refused_by_both_writers`) | tué |
| 36 | `format.rs:793` `write_codes` | `check_tail(..)?;` retiré | idem | `test result: FAILED. 34 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (même test) | tué |
| 37 | `format.rs:993` `code_stream` | bloc `if gain as usize >= m.centroids.len()` retiré | idem | `test result: FAILED. 34 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (`a_gain_past_the_centroids_is_refused_by_both_writers`) | tué |
| 38 | `format.rs:798` `write_codes` | `find(\|c\| c.gain ..)` → `find(\|c\| false && c.gain ..)` | idem | `test result: FAILED. 34 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (« value 2 does not fit in 1 bits ») | tué |
| 39 | `format.rs:1182` lecture, index Ball | `if idx > max_idx` → `if false` | idem | `test result: FAILED. 33 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out` (les deux tests Ball annoncés) | tué |
| 40 | `format.rs:836` `write_codes`, index Ball | `if idx > max_idx` → `if false` | idem | `test result: FAILED. 34 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (« point writer: expected IndexOutOfRange, got None ») | tué |
| 41 | `format.rs:650` `max_index` | bras Ball → `u64::MAX` (fusionné au bras Tetra) | idem | `test result: FAILED. 33 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out` | tué |
| 42 | `llvq-quant/src/quantizer.rs:788` poids non fini | `!a.is_finite()` → `!a.is_finite() && false` (garde éteinte) | `T -p llvq-quant --lib -- quantizer::tests` | `test result: FAILED. 1 passed; 3 failed; 0 ignored; 0 measured; 0 filtered out` (les trois `should_panic`) | tué |
| 43 | `quantizer.rs:788` sur-refus | `!a.is_finite()` → `!(a.abs() < 1.0)` | idem | `test result: FAILED. 1 passed; 3 failed; 0 ignored; 0 measured; 0 filtered out` (dont `finite_tetra_blocks_still_code_as_the_encoder_says`) | tué |
| 44 | `llvq-llm/src/fused_cpu.rs:311` `check_token_ids` | `get_vocab(true)` → `get_vocab(false)` | `T -p llvq-llm --lib -- fused_cpu::` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 241 filtered out` | tué |
| 45 | `fused_cpu.rs:312` | `>= rows` → `> rows` | idem | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 241 filtered out` | tué |
| 46 | `llvq-llm/src/fused.rs:2435` `PREALLOC_CAP` | `.min(PREALLOC_CAP)` retiré | `T -p llvq-llm --lib -- fused::tests::a_huge_count` | `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 241 filtered out` | survivant annoncé (macOS) |
| 47 | `llvq-llm/src/repl.rs:138` borne | `&& k <= NUCLEUS_MAX_K` retiré | `T -p llvq-llm --lib -- repl::` | `test result: FAILED. 11 passed; 1 failed; 0 ignored; 0 measured; 230 filtered out` (`the_selection_gives_up_past_its_cap`) | tué |
| 48 | `repl.rs:156` garde d'égalité | `let strict = c[..=i + 1]..;` → `let strict = true;` | idem | `test result: FAILED. 11 passed; 1 failed; 0 ignored; 0 measured; 230 filtered out` (`a_tie_in_the_nucleus_falls_back_to_the_sort`) | tué |
| 49 | `repl.rs:152` division | `acc += x.1 / total;` → `acc += x.1 * (1.0 / total);` | idem | `test result: FAILED. 11 passed; 1 failed; 0 ignored; 0 measured; 230 filtered out` (`a_top_p_on_the_edge_of_a_share_cuts_where_the_sort_cuts`) | tué |
| 50 | `repl.rs:142` total réassocié | bloc `settled` → `let total: f32 = c.iter().rev().map(\|x\| x.1).sum(); let settled = true;` | idem | `test result: FAILED. 9 passed; 3 failed; 0 ignored; 0 measured; 230 filtered out` | tué |
| 51 | `repl.rs:92` `nucleus_sorted` | `keep = i + 1;` → `keep = i + 2;` | `T -p llvq-llm --test sampler_nucleus` | `test result: FAILED. 0 passed; 3 failed; 2 ignored; 0 measured; 0 filtered out` | tué |
| 52 | `llvq-quant/src/gptq.rs:383` propagation | `for k in 0..b {` → `for k in (0..b).rev() {` | `T -p llvq-quant --test gptq_row_slice` | `test result: FAILED. 1 passed; 2 failed; 1 ignored; 0 measured; 0 filtered out` (les deux tests de couche) | tué |
| 53 | `gptq.rs:393` propagation | `*cell -= x * ukj;` → `*cell = (-x).mul_add(ukj, *cell);` | idem | `test result: FAILED. 1 passed; 2 failed; 1 ignored; 0 measured; 0 filtered out` | tué |
| 54 | `llvq-quant/src/linalg.rs:260` `u_row` | `at(self.n, i, 0)..` → `at(self.n, i, 1)..` | idem | `test result: FAILED. 0 passed; 3 failed; 1 ignored; 0 measured; 0 filtered out` | tué |
| 55 | `linalg.rs:152` référence | `of_inverse: false` → `true` | `T -p llvq-quant --test g5_gptq -- a_dead_channel` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 18 filtered out` | tué |
| 56 | `linalg.rs:115` `fast::pivot` | `at: index` → `at: 0 * index` | `T -p llvq-quant --features fast-linalg --test g5_gptq -- a_dead_channel` | `test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 19 filtered out` | tué |
| 57 | `llvq-search/src/generic.rs:807` | `if let Some(h) = self.odd_flat.first() {..}` → `sections.push((self.odd_flat[0].bound, usize::MAX));` | `T -p llvq-search --test g2c_reference` | `test result: FAILED. 3 passed; 1 failed; 2 ignored; 0 measured; 0 filtered out` (« index out of bounds: the len is 0 but the index is 0 ») | tué |
| 58 | `llvq-bench/src/bin/rtbits.rs:412` | `stride_u32(d_in / DIM)` → `(6 * (d_in / DIM)).div_ceil(4)` | `T -p llvq-bench --bin rtbits` | `test result: FAILED. 11 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (`the_card_stream_adds_the_tetra48_row_pad_to_the_file_stream`) | tué |
| 59 | `rtbits.rs:442` ligne carte | `m.kind == CodeKind::Tetra` → `!=` | `T -p llvq-bench --test tetra_refusals` | `test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (`rtbits_prices_a_tetra_file_instead_of_refusing_it`) | tué |
| 60 | `llvq-llm/src/fused_metal.rs:532` `kernel_named` | `Some("ilp") => "tv_tetra48_metal_ilp",` remis | `T -p llvq-llm --features metal --lib -- fused_metal::tests` | `test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured; 243 filtered out` (`an_unknown_metal_kernel_is_refused`) | tué |
| 61 | `fused_metal.rs:237` `TETRA_REFERENCES` | `"tv_tetra48_metal_c3"` retiré, longueur `[&str; 3]` (le premier essai sans la longueur ne compilait pas, E0308) | idem | `test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured; 243 filtered out` (`every_kernel_the_variable_names_passes_validate`, sur `c3`) | tué |
| 62 | `fused_metal.rs:721` | `width != LANES` → `width != 64` | `T -p llvq-llm --features metal --test tetra48_metal_candle` | `test result: FAILED. 8 passed; 4 failed; 0 ignored; 0 measured; 0 filtered out` (« SIMD-groups of 32 threads ») | tué |
| 63 | `fused_metal.rs:302` | `offered < THREADGROUP_LIMIT` → `* 2` | idem | `test result: FAILED. 0 passed; 12 failed; 0 ignored; 0 measured; 0 filtered out` (`MetalRuntime::new` refusé par nom) | tué |
| 64 | `fused_metal.rs:180` `validate` | filtre d'avant `!k.starts_with("tv_tetra48_metal") \|\| k == "tv_tetra48_metal_ilp"` | `T -p llvq-llm --features metal --test tetra48_metal_rows` | `test result: FAILED. 9 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out` (`a_bad_tuning_is_refused_by_name`) | tué |
| 65 | `llvq-bench/src/bin/lcap.rs:78` lint `unsafe_code` | `let _ = unsafe { 0 };` en tête de `main` | `cargo clippy -p llvq-bench --bin lcap -j 8` | pas de `test result:` (compilation) : `error: usage of an \`unsafe\` block`, « requested on the command line with `-F unsafe-code` », sortie 101 | tué |
| 66 | `llvq-artifact/tests/codebook_fingerprint.rs` fin | `#[test] fn m() { unsafe {} }` | `cargo clippy -p llvq-artifact --tests -j 8` | pas de `test result:` : même erreur sur le test `codebook_fingerprint`, sortie 101 | tué |
| P1 | `llvq-tetra/llvq_tetra/fused.py:97` `_require_resident_device` | `if target != resident.device:` → `if False:` | `uv run --group dev pytest -q tests/test_fused_device.py` | `2 failed, 9 passed in 5.45s` (les deux cas `test_a_move_to_another_device_is_refused_by_name`) | tué |
| P2 | `llvq_tetra/reader.py:36` `require_layout_version` | `if version is not None and version != LAYOUT_VERSION:` → `if False:` | `pytest -q tests/test_quantizer.py -k "layout_version or reader_refuses"` | `2 failed, 30 deselected in 4.48s` | tué |
| P3 | `llvq_tetra/vocab.py:27` | `or desc.get("group") != 64` retiré | `pytest -q tests/test_quantizer.py -k vocabulary` | `2 failed, 4 passed, 26 deselected in 4.37s` (`change0`, `change1`) | tué |

Comptes : 69 mutants joués. 65 tués. 4 survivants, tous annoncés par la file (n° 7, 9, 22, 46). 0 survivant inattendu.
Metal : 5 mutants (60 à 64). Python : 3 (P1 à P3).

## Tests renforcés

Aucun. Aucun survivant inattendu ne l'exigeait.

## Survivants à traiter

Les quatre sont annoncés par la file ; ils restent des trous de test connus.

- n° 7, `engine.rs`, `self.memo.clear()` après un préremplissage raté. La remise à zéro de la session suffit à rendre
  `prompt_eval_cached_count` à 0. Un test qui lirait `memo` (ou un chemin qui s'en sert sans la session) le tuerait.
- n° 9, `engine.rs`, boucle `pending.drain(..)` du bras de panique du batch. Aucun test ne met une requête en attente
  au moment de la panique. Proposition : dans `batch_server.rs`, une troisième requête envoyée quand le pool est plein,
  puis la panique ; attendre une ligne d'erreur sur son flux.
- n° 22, `api.rs:689`, 409 de `rm` sur un blob occupé. Aucun test de route. Proposition : un test `server` qui tient
  le verrou du blob (comme `rm_during_an_import_of_the_same_file_is_refused`) et appelle `DELETE /api/delete`.
- n° 46, `fused.rs:2435`, `.min(PREALLOC_CAP)`. Sur macOS la réservation géante passe sans toucher la mémoire. À jouer
  sous Linux (CI), où le processus doit avorter.

## Non joués faute de temps ou hors périmètre

Hors périmètre (carte NVIDIA, `run.py`, `--release` ou `--include-ignored`) :
- `fused_cuda.rs`, `persist4e_cuda.rs` (refus « positions up to 8714 ») ; `rtbits` sur le 4B scellé ; la suite complète
  `--include-ignored` ; `golay70_decoder_matches_rust` ; les mesures `pick_times_against_the_sorted_reference` et
  `gptq_row_slice --ignored` ; le banc sans coût de `run.py` (`_is_runner_image`, `SPACE_IMAGE_PREFIXES`,
  `cmd_oracle --kernels-cap`) ; `hf_publish_prep.py`.

Plafond Metal atteint (5) :
- `fused_metal.rs:1308` et `:1352`, `rot_metal_tg` sous `MTL_DEBUG_LAYER=1` (`2 * self.rot.threads`).
- `group_within` (`let max = 128;`, attendu vert au bit) et `group_exactly` (exige le 4B scellé).

Plafond Python atteint (3) :
- `reader.py`, second mutant (`version is not None and` retiré) ; `PackedModel.__init__` sans `require_layout_version`.
- `fused.py` : `_require_input_device`, `_apply` sans `_require_resident_device`, `_require_rotation_dtype`,
  `rotation.k == 1` → `False`.
- `integration.py`, `_require_current_cuda` (deux mutants).
- `ops/llvqtune` : `train.sh` (`MAX_FIRST_KL`, le `!`), `__main__.py` (`outcome.complete`, `UNSAVED_MODES`, code de
  sortie), `loop.py` (`steps=`), `mmlu_corpus.py` / `dclm_corpus.py` (`revision=`), `mmlu_aux_overlap.py` (`REVISION`).

Non joués, redondants avec un mutant joué au même site :
- `gptq.rs` FMA contre `g5_spherical` (`spherical_feedback_matches_an_independent_reference_on_tetra`).
- `format.rs:1144` et `:1087` ensemble (chacun joué seul : n° 32 et 33).

## Écarts à la file

- n° 5 : la file demande de retirer `catch_unwind`. Le mutant joué relance la panique dans `unwrap_or_else`, en une
  seule édition. L'effet est le même : la panique du chargeur sort du fil.
- n° 61 : retirer `c3` du tableau exige aussi de passer sa longueur de 4 à 3. Le premier essai n'a pas compilé (E0308),
  puis une seule édition a changé les deux lignes.
- n° 66 : l'outil Edit a ôté le saut de ligne final du fichier en défaisant le mutant. Il a été rendu par
  `printf '\n' >>`, et `git diff -- llvq-artifact/tests/codebook_fingerprint.rs` est revenu vide (md5 `d41d8cd9…`).
- n° 43 : en plus du test attendu, les deux `should_panic` à NaN et à l'infini rougissent : le message du refus porte
  alors une valeur finie à une autre position.

## Ajout du 2026-11-20, après la suite release

| mutant | fichier | test | résultat |
|---|---|---|---|
| `silent && job.events.is_closed()` remplacé par `silent && false && …` (moteur simple) | `llvq-runtime/src/engine.rs` | `a_client_that_leaves_during_a_silence_stops_the_engine` | TUÉ sans limite de fils (attente 377 à 378 ms contre une borne de 127 à 129 ms) et sous `RAYON_NUM_THREADS=4` (197 à 201 ms contre 68 ms) ; site appliqué vérifié (1 occurrence), arbre restauré (0 occurrence) |

Non joués : les deux sites du batch (`tx.is_closed()` de la réponse solo, `l.job.events.is_closed()` d'une ligne),
sans test qui les vise.

