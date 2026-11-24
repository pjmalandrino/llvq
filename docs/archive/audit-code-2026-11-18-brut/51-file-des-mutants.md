# File des mutants, lots 2 à 7 et second tour

## 40-lot2-runtime


| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo check -p llvq-runtime --all-targets -j 8` | compile | sans objet |
| `nice -n 10 cargo clippy -p llvq-runtime --all-targets -j 8` | zéro avertissement | sans objet |
| `ops/check-linux-cuda.sh clippy -p llvq-runtime --features cuda --all-targets` | zéro avertissement sur la cible Linux | sans objet |
| `nice -n 10 cargo test -p llvq-runtime --test engine_busy -j 8 a_load_that_panics_fails_the_engine_and_the_next_request_loads_again` | `test result: ok. 1 passed` | `engine.rs:1301-1314` : remettre `let built = loader.load(&name, &manifest, &store).and_then(...)` sans `catch_unwind` ; attendu rouge (premier événement `None`, ou `submit` refusé « has stopped ») |
| `nice -n 10 cargo test -p llvq-runtime --test engine_busy -j 8 a_manifest_written_again_with_a_new_date_keeps_the_loaded_engine` | ok | `engine.rs:1503` : `same_model(&e.info.manifest, &manifest)` remplacé par `e.info.manifest == manifest` ; attendu rouge (`loads` à 2, ou 503 « loaded and busy ») |
| `nice -n 10 cargo test -p llvq-runtime --test server -j 8 a_failed_prefill_leaves_an_empty_cache_to_the_next_request` | ok | `engine.rs:684` : supprimer `self.session.reset(&self.loaded.qwen);` ; attendu rouge (`prompt_eval_cached_count` 10, le premier tour et l'ouverture assistant). Supprimer seulement `self.memo.clear()` (`:685`) SURVIT : la remise à zéro de la session suffit à rendre 0. |
| `nice -n 10 cargo test -p llvq-runtime --test server -j 8 a_stop_list_past_its_bounds_is_refused` | ok | `api.rs:160` : `> MAX_STOPS` en `> MAX_STOPS + 1` ; attendu rouge. `api.rs:163` : supprimer le bloc `if let Some(s) = ...` ; attendu rouge. |
| `nice -n 10 cargo test -p llvq-runtime --test batch_server -j 8 a_panic_in_the_batched_step_ends_every_live_stream_with_an_error` | ok | `engine.rs:1347-1349` : supprimer la boucle sur `live.drain(..)` ; attendu rouge sur les deux flux (« ended without an error line »). La boucle sur `pending` (`:1350-1353`) n'est exercée par aucun test : son mutant survit. |
| `nice -n 10 cargo test -p llvq-runtime --test batch_server -j 8 a_failed_batched_step_leaves_the_model_to_its_keep_alive` | ok, en moins de 5 s après les erreurs | `engine.rs:878` : supprimer la ligne `*inf.expires.lock()...` du bras du pas raté ; attendu rouge après 5 s (« outlived its keep-alive ») |
| `nice -n 10 cargo test -p llvq-runtime --test pull -j 8 rm_during_an_import_of_the_same_file_is_refused` | ok | `store.rs:252-257` : remplacer le bloc `if let Err(e) = lock.try_lock() {...}` par `let _ = lock.try_lock();` ; attendu rouge (`expect_err` sur un `Ok(true)`). `api.rs:689` (409) n'a pas de test de route : son mutant en 500 survit. |
| `nice -n 10 cargo test -p llvq-runtime --test openapi -j 8` | ok, 4 tests : `openapi.json` doit rester celui que les handlers génèrent | sans objet |
| `nice -n 10 cargo test -p llvq-runtime -j 8` | toute la crate verte, en comptant les lignes `test result:` (lib, `batch_server`, `engine_busy`, `openapi`, `pull`, `server`, doctests) | sans objet |

Notes pour qui lance :
- Les deux tests de `batch_server.rs` reposent sur le même schéma que `start_long` : la première requête doit encore
  décoder (400 jetons) quand la seconde arrive. Avec deux couches, le 5e appel d'`attend` tombe au 3e pas groupé, les
  deux requêtes vivantes.
- `a_failed_prefill_leaves_an_empty_cache_to_the_next_request` fait échouer le préremplissage par un mot dont l'id
  (100) dépasse les 64 lignes du petit modèle : candle refuse l'`index_select` (`InvalidIndex`, `candle-core` 0.9.2,
  `cpu_backend/mod.rs:686`). L'échec tombe avant toute couche : le test garde la remise à zéro du moteur, pas l'état
  incohérent des caches, qui ne se produit que sur carte (chaîne 4E).
- `a_load_that_panics_...` imprime la panique du chargeur sur stderr : c'est attendu.

## 41-lot2-session-hote


| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo clippy -p llvq-llm --all-targets -j 8` | zéro avertissement (session.rs, fused_cpu.rs, tests/session.rs) | aucun |
| `nice -n 10 cargo test -p llvq-llm --test session -j 8` | `test result: ok. 12 passed` (9 anciens, 3 neufs) | `llvq-llm/src/session.rs:304`, remplacer `&made[..made.len().saturating_sub(1)]` par `&made[..]` : rouges `the_persistent_branch_feeds_what_its_launches_wrote` (le deuxième lancement part de 10 au lieu de 9 : réponse `110..` au lieu de `109..`, journal décalé) et `the_persistent_branch_cuts_where_it_stops_or_is_cancelled` (annulation : 8 ids au lieu de 7) |
| idem | idem | `llvq-llm/src/session.rs:140`, insérer après le `ensure!` la ligne `if n == self.ids.len() { return Ok(()); }` : rouge `a_failed_launch_leaves_caches_the_next_request_cuts_back` (« after reuse_prefix: layer 0 holds 12 positions for 10 ids ») |
| idem | idem | `llvq-llm/src/session.rs:142`, `!= n` devient `== n` : rouges `a_failed_launch_leaves_caches_the_next_request_cuts_back` et `a_divergent_history_is_cut_back_to_the_common_prefix` |
| `nice -n 10 cargo test -p llvq-llm --lib fused_cpu:: -j 8` | `a_tokenizer_id_past_the_embedding_table_is_refused_by_name` vert | `llvq-llm/src/fused_cpu.rs` dans `check_token_ids`, `get_vocab(true)` devient `get_vocab(false)` : rouge (table de 3 lignes acceptée) ; puis `>= rows` devient `> rows` : rouge |
| `nice -n 10 cargo clippy -p llvq-llm --features metal --all-targets -j 8` | zéro avertissement (fused_metal.rs) | aucun |
| `nice -n 10 cargo test -p llvq-llm --features metal --test rot_metal_tg -j 8` | vert : les deux rotations rendent les mêmes bits qu'avant sur le M3 Max (la limite y vaut 1 024, `threads` ne bouge pas) | `llvq-llm/src/fused_metal.rs:1308` et `:1352`, `self.rot.threads.min(` devient `(2 * self.rot.threads).min(` : reste vert sous `MTL_DEBUG_LAYER=1` (le groupe est ramené à la limite) ; puis retirer `.min(pipe.max_total_threads_per_threadgroup())` en gardant `2 * self.rot.threads` : la couche de validation refuse le lancement (ce que faisait le code d'avant pour une limite sous 1 024) |
| `nice -n 10 cargo test -p llvq-llm --features metal --test tetra48_metal_candle -j 8` | vert, inchangé (passe aussi par `RotOp`) | aucun |
| `ops/check-linux-cuda.sh clippy -p llvq-llm --features cuda --all-targets` | zéro avertissement (persist4e_cuda.rs, fused_cuda.rs) | aucun sur Mac ; sur carte, `decode` à `pos0 = 8 650`, `n = 64`, puis `truncate(100)` et un `decode` : refus nommé « positions up to 8714 » au premier, succès au second |
| porte servie sur carte, au prochain run payé déjà prévu (aucun run à lancer pour ceci) | jetons et ligne `NVRTC source` identiques ; la ligne de chargement ne contient aucun refus | aucun |


## 42-lot3-format


`llvq-artifact` et `llvq-quant` sont touchées : la suite complète en release est due avant tout commit.

| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo check -p llvq-artifact --all-targets -j 8` | compile | — |
| `nice -n 10 cargo check -p llvq-quant --all-targets -j 8` | compile | — |
| `nice -n 10 cargo check -p llvq-bench --all-targets -j 8` | compile | — |
| `cargo clippy --all-targets` | zéro avertissement | — |
| `nice -n 10 cargo test -p llvq-artifact --test hostile_files -j 8` | tous verts, dont les 11 nouveaux | voir les lignes suivantes, une par test |
| idem, `a_raw_quant_width_is_checked_as_stored_not_narrowed` | vert | `sealed.rs:353-361` : remplacer le `match` par `let bits = get_u32(r, "quant width")? as u8;` → rouge sur 0x104 (lu 4, rend `Ok`) |
| idem, `a_raw_quant_group_of_zero_is_refused_not_a_division_by_zero` | vert | `sealed.rs:255-257` : supprimer le bloc `if q.group == 0` → rouge, panique « attempt to divide by zero » dans `div_ceil` |
| idem, `a_raw_quant_payload_short_of_its_dims_is_refused_not_a_panic` | vert | `sealed.rs:376` : supprimer `check_quant(&name, &dims, &q)?;` → rouge (`Ok`) ; le test de largeur reste vert, preuve que les deux contrôles sont distincts |
| idem, `raw_dims_whose_product_overflows_are_refused_by_name` | vert | `sealed.rs:237` : `.try_fold(1usize, \|acc, &d\| acc.checked_mul(d))` → `.try_fold(1usize, \|acc, &d\| Some(acc.wrapping_mul(d)))` → rouge (2⁸⁰ replié en 0, détail « packed bytes » au lieu de « usize ») |
| idem, `a_raw_f16_count_that_disagrees_with_its_dims_is_refused` | vert | `sealed.rs:342-347` : supprimer le `if` → rouge (`Ok`) |
| idem, `honest_raw_tensors_still_read_byte_for_byte` | vert | `sealed.rs:263` : `8 => n,` → `8 => n + 1,` → rouge (le tenseur int8 honnête est refusé à l'écriture) |
| idem, `a_ball_gain_past_its_centroids_is_refused_on_read_not_a_panic` | vert | `format.rs:1144-1146` : supprimer le `if gain as usize >= n_cent` → rouge (`Ok` avec gain 3 sur 3 centroïdes) |
| idem, `a_ball_record_with_blocks_and_no_centroid_is_refused_on_read` | vert | `format.rs:1087-1095` : supprimer le bloc `n_cent == 0` → rouge sur le détail (« gain rank 0 with 0 centroids » au lieu de « no gain centroid ») ; supprimer aussi `:1144-1146` → rouge (`Ok`) |
| idem, `a_gain_past_the_centroids_is_refused_by_both_writers` | vert | `format.rs:953-955` : supprimer le contrôle de `code_stream` → rouge (`Ok`) ; puis `format.rs:775-777` dans `write_codes` → rouge, panique « value 2 does not fit in 1 bits » de `BitWriter::push` |
| idem, `a_tail_count_off_the_dimensions_is_refused_by_both_writers` | vert | `format.rs:942` : supprimer `check_tail(..)?;` de `code_stream` → rouge (33 valeurs écrites) ; puis `:771` dans `write_codes` → rouge |
| idem, `a_4b_shaped_record_and_its_tail_still_round_trip` | vert | `format.rs:556` : `let want = d_out * (d_in % DIM);` → `let want = d_in % DIM;` → rouge (32 valeurs refusées, 16 attendues) |
| `nice -n 10 cargo test -p llvq-artifact -j 8` | les autres cibles restent vertes : `raw_tensors`, `g6_format`, `tetra_format` (dont `a_tetra_matrix_with_the_wrong_cap_or_gain_width_is_refused`, qui doit garder le refus de `gain_width` avant le mien), `mixed_kinds`, `int4_format`, `codebook_fingerprint`, `runtime_format` | — |
| `nice -n 10 cargo test -p llvq-quant --lib quantizer::tests -j 8` | 4 verts | `quantizer.rs:788-794` : supprimer le `if let Some(i) …` → rouge sur les trois `should_panic` (un NaN : message de l'assert de l'encodeur ; tout NaN : aucune panique ; −inf : autre message ou aucune). Mutant de sur-refus : `!a.is_finite()` → `!(a.abs() < 1.0)` → rouge sur `finite_tetra_blocks_still_code_as_the_encoder_says` (le bloc fini monte à 1,05) |
| `nice -n 10 cargo test -p llvq-bench --bin rtbits -j 8` | vert, dont `the_card_stream_adds_the_tetra48_row_pad_to_the_file_stream` | `rtbits.rs:412` : `stride_u32(d_in / DIM)` → `(6 * (d_in / DIM)).div_ceil(4)` (rangée arrondie à 4 octets) → rouge (rembourrage nul à 2 560) |
| `nice -n 10 cargo test -p llvq-bench --test tetra_refusals --test int4_refusals -j 8` | vert : `rtbits` tourne sur un fichier v5 avec les trois lignes nouvelles | — |
| `nice -n 10 cargo test -p llvq-llm -j 8` | vert : les lecteurs de tenseurs portés (`embedquant`, `embed_q4`, `embed_q8`, `hfpack`, `proj_q4`) lisent ce que `write_raw` écrit | — |
| `cargo run --release -p llvq-bench --bin rtbits -- <le 4B scellé>` | lignes existantes identiques au caractère à `rtbits-sealed.txt` ; nouvelles lignes : 29 294 592 bits de rembourrage, 2,5501 b/weight noyau carte, b/param « f16 + tetra48 stride » q4 près de 2,7393 | — |
| `cargo test --release -- --include-ignored` | suite complète verte, archives scellées comprises (`~/llvq-q4b.llvq`) : obligatoire avant commit, le lecteur et l'écrivain du format sont touchés | — |

## 43-lot4-outillage


| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo clippy -j 8 -p llvq-core -p llvq-search -p llvq-quant -p llvq-artifact -p llvq-bench --all-targets` | zéro avertissement, zéro erreur | ajouter la ligne `let _ = unsafe { 0 };` au début de `main` dans `llvq-bench/src/bin/lcap.rs` : la compilation doit échouer sur `unsafe_code` ; défaire l'édition, revert vert |
| même commande, mutant dans un test : ajouter `#[test] fn m() { unsafe {} }` à la fin de `llvq-artifact/tests/` (un fichier existant) | doit échouer sur `unsafe_code` | défaire, revert vert |
| `cd ops/llvqtune && PYTHONPATH=. uv run --group dev pytest -q tests/test_train_sh.py -k "ceiling"` (ou `python3 -m pytest` comme la CI) | `3 passed` : NaN refusé (4), 0.9 refusé (4), 0.3 lancé (0) | `ops/llvqtune/train.sh`, garde `MAX_FIRST_KL` : remplacer `math.isfinite(v) and v <= float(...)` par `not (v > float(...))` ; attendu : le test NaN rougit, les deux autres restent verts |
| `cd ops/llvqtune && PYTHONPATH=. uv run --extra torch pytest -q tests/test_mmlu_corpus.py tests/test_main_corpus.py -k revision` | `2 passed` (skip si torch ou huggingface_hub absents) | `mmlu_corpus.py`, retirer `revision=REVISION,` de `fetch_split` : attendu `KeyError: 'revision'`, 1 failed ; idem dans `dclm_corpus.py` |
| `uv run ops/run.py selftest` | les lignes `ok` comme avant (lit `config.json` du Hub, rien de facturé) | aucun |
| `uv run ops/run.py launch --help` | `--calib {c4,wikitext2,wikitext2-test,dclm-edu}` | aucun |
| `python3 -c "import math; [print(w, max(60, math.ceil(w*1.5*60))) for w in (0.5, 1.3, 2.6)]"` | `60`, `118`, `235` minutes | aucun |
| Le banc sans coût ci-dessous, depuis `ops/` : il remplace `run_job` et `run_uv_job` par une fonction qui imprime ce qu'elle reçoit et sort, donc RIEN n'est lancé. `HF_TOKEN=dummy` n'est envoyé nulle part | voir les attendus sous le bloc | `ops/run.py`, `_is_runner_image` : remplacer `startswith("llvq-runner")` par `startswith("")` ; attendu : le cas pytorch reçoit `['HF_TOKEN']` |

```bash
cd ops && HF_TOKEN=dummy uv run --with 'huggingface-hub>=1.26' python - <<'PY'
import sys, huggingface_hub as h
def stop(**k):
    s = k.get("secrets")
    print("   would launch: timeout", k.get("timeout"), "secrets", None if s is None else sorted(s))
    raise SystemExit(0)
h.run_job = stop
h.run_uv_job = stop
import run
cases = [
    ["bench", "true", "--image", "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime", "--timeout", "20m"],
    ["bench", "true", "--image", "hf.co/spaces/Pier-Jean/llvq-runner-cuda", "--timeout", "20m"],
    ["oracle", "--image", "x", "--flavor", "a100-large"],
    ["oracle", "--image", "x"],
    ["dequant", "--awq-repo", "Qwen/Qwen3-14B-AWQ", "--base-repo", "Qwen/Qwen3-14B",
     "--bucket", "b", "--name", "n", "--timeout", "2h30m"],
    ["dequant", "--awq-repo", "Qwen/Qwen3-14B-AWQ", "--base-repo", "Qwen/Qwen3-14B",
     "--bucket", "b", "--name", "n"],
]
for argv in cases:
    sys.argv = ["run.py", *argv]
    print("==", " ".join(argv[:4]))
    try:
        print("   returned", run.main())
    except SystemExit as e:
        print("   exit", e.code)
PY
```

Attendus du banc :
1. `bench` pytorch : la ligne de coût, puis `HF_TOKEN not passed: pytorch/... is not hf.co/spaces/<namespace>/llvq-runner*`, puis `secrets None`.
2. `bench` runner : la ligne de coût, puis `secrets ['HF_TOKEN']`.
3. `oracle` sur `a100-large` : `refused, a100-large: sm_80 < sm_89, ...`, `returned 2`, aucune ligne `would launch`.
4. `oracle` par défaut (`l4x1`) : `l4x1 at 0.80 $/h = 0.0133 $/min; ceiling 20m → at worst 0.27 $` AVANT `would launch: timeout 20m secrets None`.
5. `dequant --timeout 2h30m` : `exit --timeout '2h30m': write a number and ONE unit, ...`, aucune ligne `would launch`.
6. `dequant` par défaut : `cpu-upgrade at 0.03 $/h; ceiling 4h → at worst 0.12 $`, puis `HF_TOKEN not passed: the default uv image ...`, puis `would launch: timeout 4h secrets None`.

`hf_publish_prep.py`, sur une COPIE du fixture (le script écrit dans le répertoire quand il passe) :

| commande exacte | ce qu'elle doit montrer | mutant |
|---|---|---|
| `cp -r llvq-tetra/tests/fixtures/mini /tmp/pp && uv run --with huggingface_hub ops/hf_publish_prep.py /tmp/pp` | `REFUSED: --revision is missing ...`, code 2, rien téléchargé | aucun |
| `uv run --with huggingface_hub ops/hf_publish_prep.py /tmp/pp --revision 1cfa9a7208912126459214e8b04321603b3df60c` | télécharge les fichiers tokenizer et `config.json` du 4B, puis `REFUSED: hidden_size is 136 in /tmp/pp/config.json and 2560 in Qwen/Qwen3-4B@...`, code 2, `/tmp/pp` inchangé | aucun |
| `publish` | aucune commande sans coût ne l'exerce ; relecture seule. À voir au prochain `publish --compute-cap` réel : la ligne `refused: ... CUDA_COMPUTE_CAP` doit arriver avant `space ...` | aucun |

CI (TST-08), à voir sur GitHub au prochain lot de PR, pas localement :

| action | attendu |
|---|---|
| une PR qui ne touche que `docs/ENV.md` | la CI part |
| une PR qui ne touche que `configs/README.md` | la CI part |
| une PR qui ne touche que `docs/STYLE.md` ou `README.md` | la CI ne part pas |
| une PR qui ne touche que `.github/workflows/ci.yml` | la CI part ; si elle ne part pas, `**` n'attrape pas les chemins à point et il faut ajouter `'.github/**'` en tête de la liste |

## 44-lot3-python


| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `cd llvq-tetra && uv run --group dev pytest -q` | toute la suite verte, 64 tests du lot 1 plus 8 nouveaux | aucun ; c'est la base avant les mutants |
| `cd llvq-tetra && uv run --group dev pytest -q tests/test_quantizer.py -k layout_version` | `2 passed` | `llvq_tetra/reader.py`, `require_layout_version` : `if version is not None and version != LAYOUT_VERSION:` → `if False:` ; attendu `2 failed`. Second mutant : retirer `version is not None and ` ; attendu `1 failed` (`test_another_layout_version_is_refused_by_name`, sur l'absence), `1 passed` |
| `cd llvq-tetra && uv run --group dev pytest -q tests/test_quantizer.py -k reader_refuses` | `1 passed` | `llvq_tetra/reader.py`, `PackedModel.__init__` : supprimer la ligne `require_layout_version(self.qc.get("llvq_layout_version"))` ; attendu `1 failed` (le lecteur continue et échoue plus loin sur un autre message ou sur `model.safetensors` absent) |
| `cd llvq-tetra && uv run --group dev pytest -q tests/test_fused_device.py` | `6 passed`, sans MPS ni carte | `llvq_tetra/fused.py`, `_require_resident_device` : `if target != resident.device:` → `if False:` ; attendu `2 failed, 4 passed` (`test_a_move_to_another_device_is_refused_by_name`) |
| idem | idem | `llvq_tetra/fused.py`, `_require_input_device` : `if x.device != resident.device:` → `if False:` ; attendu `2 failed, 4 passed` (`test_an_input_on_another_device_is_refused_by_name`, autre exception levée plus loin) |
| idem | idem | `llvq_tetra/fused.py`, `FusedTetraLinear._apply` : supprimer la ligne `_require_resident_device(...)` ; attendu `1 failed, 5 passed` (cas `tetra` seulement) |
| `cd llvq-tetra && uv run --group dev pytest -q tests/test_mps.py tests/test_load.py` (sur le Mac, MPS présent) | verts comme avant : le `.to("mps")` de `finalize` passe la garde | aucun |
| `cd llvq-tetra && uv run --group dev python -m llvq_tetra.checkmps <dossier 4B>` (Mac, quand l'opérateur le décide) | sortie identique à la précédente : aucun module fusé refusé au chargement | aucun |

## 45-lot7-optimisation


| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo clippy -p llvq-llm --all-targets -j 8` | zéro avertissement | aucun |
| `nice -n 10 cargo clippy -p llvq-quant --all-targets -j 8` puis la même avec `--features fast-linalg` | zéro avertissement, `forbid(unsafe_code)` tenu | aucun |
| `nice -n 10 cargo test -p llvq-llm --lib repl -j 8` | les 3 tests neufs de `repl.rs` (`the_selected_nucleus_is_the_sorted_one_bit_for_bit`, `a_top_p_on_the_edge_of_a_share_cuts_where_the_sort_cuts`, `a_tie_in_the_nucleus_falls_back_to_the_sort`) et les 8 anciens verts | `repl.rs`, `nucleus_selected`, remplacer tout le bloc `let settled = c.iter().any(...)` par `let total: f32 = c.iter().rev().map(\|x\| x.1).sum(); let settled = true;` (total réassocié, sommé à l'envers) : rouge attendu dans `a_top_p_on_the_edge...`, et dans `the_selected_nucleus...` sur les lignes plates |
| idem | idem | `repl.rs`, `nucleus_selected`, `acc += x.1 / total;` devient `acc += x.1 * (1.0 / total);` : rouge attendu dans `a_top_p_on_the_edge...` |
| idem | idem | `repl.rs`, `nucleus_selected`, `let strict = c[..=i + 1]...;` devient `let strict = true;` : rouge attendu dans `a_tie_in_the_nucleus...` |
| `nice -n 10 cargo test -p llvq-llm --test sampler_nucleus -j 8` | `pick_draws_the_reference_token_on_every_row` et `..._at_degenerate_settings` verts : même jeton et même état du générateur que la copie de l'ancien `pick`, sur lignes piquées, plates, mantisse f16, NaN, infinis, constante, deux maxima égaux, températures et `top_p` dégénérés | mutant `let strict = true;` ci-dessus : rouge attendu sur « two equal maxima » ou les lignes en mantisse f16 (le jeton dépend de l'ordre des égalités) |
| `nice -n 10 cargo test --release -p llvq-llm --test sampler_nucleus -j 8` | en plus, `pick_draws_the_reference_token_at_vocabulary_size` vert à 151 936 jetons | idem |
| `nice -n 10 cargo test --release -p llvq-llm --test sampler_nucleus -j 8 -- --ignored --nocapture pick_times_against_the_sorted_reference` | MESURE avant/après sur le Mac, CPU : temps par tirage de `pick_reference` (ancien) et de `pick` (nouveau), médiane et plage sur 9 tours entrelacés de 64 tirages, sur une ligne piquée et une ligne plate de 151 936 logits synthétiques, température 0,7, `top_p` 0,9. Chronomètre tout `pick` (copie vers l'hôte comprise, sans modèle). À étiqueter *mesuré*, avec la machine | aucun |
| `nice -n 10 cargo test -p llvq-quant --test gptq_row_slice -j 8` puis la même avec `--features fast-linalg` | `a_row_of_u_is_the_entries_u_returns`, `..._on_a_scalar_grid`, `..._on_leech_and_in_parallel` verts : couche identique au bit à la boucle indexée copiée, rétraction on/off, avec et sans queue, une ligne nulle (tous les `k` sautés), quatre fils, sur les deux factorisations | `gptq.rs`, propagation, `*cell -= x * ukj;` devient `*cell = (-x).mul_add(ukj, *cell);` (FMA, un arrondi en moins) : rouge attendu dans les deux tests de couche |
| idem | idem | `gptq.rs`, propagation, `for k in 0..b {` (la boucle qui suit `let row = &mut weights.w[...]`) devient `for k in (0..b).rev() {` (somme par cellule réassociée) : rouge attendu dans les deux tests de couche |
| idem | idem | `linalg.rs`, `u_row`, `at(self.n, i, 0)..` devient `at(self.n, i, 1)..` : rouge attendu dans `a_row_of_u_is_the_entries_u_returns` et dans les tests de couche |
| `nice -n 10 cargo test -p llvq-quant --test g5_gptq --test g5_spherical -j 8`, avec et sans `--features fast-linalg` | `parallel_matches_serial_exactly`, `both_factorizations_agree` et `spherical_feedback_matches_an_independent_reference_on_tetra` toujours verts : la référence indexée sur Tetra tient | mutant `mul_add` ci-dessus : rouge attendu aussi dans `spherical_feedback_matches_an_independent_reference_on_tetra` |
| `nice -n 10 cargo test --release -p llvq-quant --features fast-linalg --test gptq_row_slice -j 8 -- --ignored --nocapture` | MESURE avant/après sur le Mac : temps d'une couche 512 × 4 096, grille scalaire (la propagation domine), un fil, boucle indexée copiée contre `quantize_layer`, médiane et plage sur 5 tours entrelacés, avec l'égalité au bit vérifiée à chaque tour. À étiqueter *mesuré*. Limite : la copie indexée vit dans la crate de test, compilée au même profil ; ce n'est pas l'ancien binaire | aucun |
| `nice -n 10 cargo test --release -p llvq-quant --features fast-linalg -j 8 -- --include-ignored` | la suite de la crate verte avant tout commit : la propagation sert l'encodeur qui produit les fichiers scellés | aucun |

Optionnel, go opérateur requis (calcul lourd local) : encoder trois blocs du 0.6B par `smoke` avec `LLVQ_ARTIFACT`, une
fois à `925930d` et une fois sur cette branche, avec les mêmes arguments, puis comparer le sha256 des deux artefacts
(attendu identique) et la phase de quantification imprimée.

## 80-r2-fix-runtime


| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo check -p llvq-runtime --all-targets -j 8` | compile | sans objet |
| `nice -n 10 cargo clippy -p llvq-runtime --all-targets -j 8` | zéro avertissement (surveiller `type_complexity` sur `Store::manifests`, `dead_code` sur les champs de `BlobId`, lus par le `PartialEq` dérivé) | sans objet |
| `ops/check-linux-cuda.sh clippy -p llvq-runtime --features cuda --all-targets` | zéro avertissement sur la cible Linux | sans objet |
| `nice -n 10 cargo test -p llvq-runtime --test engine_busy -j 8 a_blob_written_again_under_the_same_manifest_loads_the_model_again` | `test result: ok. 1 passed` | `engine.rs`, `same_model` : retirer `&& e.blob == blob` ; attendu rouge « the engine read from the removed file still serves » (1 chargement au lieu de 2). Second mutant, `store.rs`, `blob_id` : rendre `Some(BlobId { len: m.len(), modified: None, node: (0, 0, 0, 0) })` ; attendu rouge de même (même taille) |
| `nice -n 10 cargo test -p llvq-runtime --test engine_busy -j 8 a_manifest_written_again_with_a_new_date_keeps_the_loaded_engine` | ok, toujours 1 chargement | la ligne 23 de la file vise l'ancien site : le mutant devient, dans `same_model`, `Manifest { modified_at: now.modified_at, ..e.manifest.clone() } == *now` remplacé par `e.manifest == *now` ; attendu rouge (`loads` à 2) |
| `nice -n 10 cargo test -p llvq-runtime --test batch_server -j 8 a_panic_in_the_batched_step_ends_every_live_stream_with_an_error` | ok, en quelques secondes ; vert aussi avec `try_send` (le client lit, le canal a de la place) | `engine.rs`, bras de panique du batch : supprimer la boucle `for l in live.drain(..)` ; attendu ROUGE en au plus 120 s après le premier morceau, « the streams neither ended nor failed within 120 s », et non plus un test qui pend |
| `nice -n 10 cargo test -p llvq-runtime --test batch_server -j 8 a_failed_batched_step_leaves_the_model_to_its_keep_alive` | ok (même `two_in_the_pool`) | aucun nouveau |
| `nice -n 10 cargo test -p llvq-runtime --lib -j 8 chat::tests::a_stop_prefix_before_a_split_character_is_held` | ok | `chat.rs`, `scan` : remettre `text.ends_with(&s[..k])` et `text.len() - k` ; attendu rouge `Safe(4)` contre `Safe(3)` |
| `nice -n 10 cargo test -p llvq-runtime --lib -j 8 chat::` | les 7 tests de `chat` verts, dont les 4 anciens de `scan` | sans objet |
| `nice -n 10 cargo test -p llvq-runtime --test pull -j 8 a_manifest_that_does_not_read_leaves_the_others_listed_and_removable` | ok ; stderr porte deux lignes « llvq: a manifest that does not read is left out » et une « llvq: ... is kept: 1 manifest(s) do not read » | `store.rs`, `manifests` : `Err(e) => unread.push(e)` devient `Err(e) => return Err(e)` ; attendu rouge au premier `list().unwrap()`. `store.rs`, `remove` : `if !held && unread.is_empty()` devient `if !held` ; attendu rouge « a blob that a manifest may name was removed » |
| `nice -n 10 cargo test -p llvq-runtime --test pull -j 8` | les 8 tests verts, dont `import_checks_a_local_copy_and_rm_keeps_a_blob_another_name_holds` et `rm_during_an_import_of_the_same_file_is_refused` inchangés | sans objet |
| `nice -n 10 cargo test -p llvq-runtime --test server -j 8 the_openai_routes_answer_as_the_ollama_ones_in_openais_shapes` | ok | `openai.rs`, bras `Event::Done` du flux : supprimer le bloc `if !ctx.warnings.is_empty() { fin["warnings"] = ... }` ; attendu rouge sur `fin["warnings"][0]` (`unwrap` d'un `None`). Remettre `"logprobs"` dans `HARMLESS` ; attendu rouge sur `ask(true)`. Supprimer le second `.filter(...)` de `warn_extra` ; attendu rouge sur `ask(false)` |
| `nice -n 10 cargo test -p llvq-runtime --test server -j 8 one_object_when_stream_is_false_and_generate_takes_raw_and_templated_prompts` | ok | `api.rs`, `generate` : supprimer le `if req.raw == Some(true) && req.system...` ; attendu rouge sur `r["warnings"][0]` (`unwrap` d'un `None`) |
| `nice -n 10 cargo test -p llvq-runtime --test server -j 8 a_failed_prefill_leaves_an_empty_cache_to_the_next_request` | ok, inchangé (la remise à neuf reste) | sans objet nouveau |
| `nice -n 10 cargo test -p llvq-runtime --test openapi -j 8` | ok, 4 tests : `openapi.json` inchangé | sans objet |
| `nice -n 10 cargo test -p llvq-runtime -j 8` | toute la crate verte, en comptant les lignes `test result:` (lib, `batch_server`, `engine_busy`, `openapi`, `pull`, `server`, doctests) | sans objet |

Aucun test pour R2-API-15 (message seul, le relecteur jugeait la lecture suffisante) ni pour R2-DIF-06 (commentaire).

## 81-r2-fix-llm-metal


| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo check -p llvq-llm --all-targets -j 8` | compile, zéro avertissement | aucun |
| `nice -n 10 cargo clippy -p llvq-llm --all-targets -j 8` | zéro avertissement (vérifier en particulier `{NUCLEUS_MAX_K}` dans un `assert!`, et l'absence d'`allow` sur `upload_int4`) | aucun |
| `nice -n 10 cargo clippy -p llvq-llm --features metal --all-targets -j 8` | le chemin Metal compile : `pipe.as_ref().threadExecutionWidth()`, `dev.device().as_ref().maxThreadgroupMemoryLength()`, les `group` | aucun |
| `ops/check-linux-cuda.sh clippy -p llvq-llm --features cuda --all-targets` | `fused_cuda.rs` compile, zéro avertissement | aucun |
| `nice -n 10 cargo test -p llvq-llm --test session -j 8` | vert, 12 tests | `llvq-llm/src/session.rs:83`, remplacer `c.held()?.map_or(Ok(0), \|(k, _)\| k.dim(2))?` par `c.len()` : rouge sur `Cat`, « layer 0 kept 4 positions after a cut to zero » ; défaire, vert |
| `nice -n 10 cargo test -p llvq-llm --lib repl:: -j 8` | vert, dont `the_selection_gives_up_past_its_cap` | `llvq-llm/src/repl.rs:138`, retirer `&& k <= NUCLEUS_MAX_K` : rouge sur « a selection past 1024 weights ran » ; défaire, vert |
| `nice -n 10 cargo test -p llvq-llm --test sampler_nucleus -j 8` | vert, dont `pick_draws_the_reference_token_where_the_selection_gives_up` | `llvq-llm/src/repl.rs`, dans `nucleus_sorted`, `keep = i + 1` en `keep = i + 2` : rouge (le repli ne rend plus le vecteur de la référence) ; défaire, vert |
| `cargo test --release -p llvq-llm --test sampler_nucleus pick_times_against_the_sorted_reference -- --ignored --nocapture` | médianes de `pick` contre `pick_reference` sur trois bras (« six heads », « flat », « six heads, f16 mantissa ») ; à jouer avec la borne, puis sans (`repl.rs:138`, retirer `&& k <= NUCLEUS_MAX_K`), en interférant les deux binaires à la main | décision : si le bras le pire perd plus que le bras piqué ne gagne, désactiver le raccourci en remplaçant `repl.rs:66` `let p = nucleus_selected(&p, top_p).unwrap_or_else(\|\| nucleus_sorted(p, top_p));` par `let p = nucleus_sorted(p, top_p);`. Alors `nucleus_selected` et `NUCLEUS_MAX_K` ne servent plus qu'aux tests : les mettre sous `#[cfg(test)]` ou les retirer, sinon clippy signale du code mort |
| `nice -n 10 cargo test -p llvq-llm --lib fused::tests::a_huge_count -j 8` | vert : deux `Err` | `llvq-llm/src/fused.rs:2435`, retirer `.min(PREALLOC_CAP)` : sous Linux, le processus de test avorte sur une réservation de quelque 800 Go ; sous macOS la mémoire virtuelle peut accepter la réservation sans la toucher, et le mutant survivre. À constater ; s'il survit sur le Mac, le jouer sous Linux (CI) |
| `nice -n 10 cargo test -p llvq-llm --features metal --lib fused_metal::tests -j 8` | vert, dont `an_unknown_metal_kernel_is_refused` et `every_kernel_the_variable_names_passes_validate` | `fused_metal.rs`, remettre `Some("ilp") => "tv_tetra48_metal_ilp",` dans `kernel_named` : rouge sur `"ilp"` ; défaire, vert. Puis retirer `"tv_tetra48_metal_c3"` de `TETRA_REFERENCES` : `every_kernel_the_variable_names_passes_validate` rouge sur `c3` |
| `nice -n 10 cargo test -p llvq-llm --features metal --test tetra48_metal_rows -j 8` | vert, dont `a_bad_tuning_is_refused_by_name` | `fused_metal.rs:180`, remettre le filtre `!k.starts_with("tv_tetra48_metal") \|\| k == "tv_tetra48_metal_ilp"` : rouge sur `tv_tetra48_metal_arh` ; défaire, vert |
| `nice -n 10 cargo test -p llvq-llm --features metal --test tetra48_metal_candle -j 8` | vert : aucun lancement ne change sur le M3 Max | `fused_metal.rs:721`, `width != LANES` en `width != 64` : tout chargement refusé par nom (« SIMD-groups of 32 threads ») ; défaire. Puis `fused_metal.rs:302`, `offered < THREADGROUP_LIMIT` en `offered < THREADGROUP_LIMIT * 2` : `MetalRuntime::new` refusé par nom ; défaire. Puis dans `group_within`, `let max = pipe.max_total_threads_per_threadgroup();` en `let max = 128;` : les tests restent verts au bit (le groupe ne change aucun bit) ; défaire |
| `nice -n 10 cargo test -p llvq-llm --features metal --test q4_metal_candle -j 8` | vert : les lancements int4 gardent 256 | aucun test Metal sans fichier scellé ne passe par `GatherOp` (`git grep` de `upload_embed`, `MetalEmbed` dans `llvq-llm/tests` : rien sur Metal). Le mutant de `group_exactly` (`let max = 128;`) se constate sur le 4B scellé, par `bin/metalrun` ou `bin/chat` sur Metal : refus par nom « needs 256 threads » au premier jeton ; défaire |


## 82-r2-fix-coeur


| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo test -p llvq-search --test g2c_reference -j 8` | `a_one_level_codebook_is_searchable` ok, les autres inchangés | `llvq-search/src/generic.rs:807-809` : remplacer le `if let Some(h) = self.odd_flat.first() { … }` par `sections.push((self.odd_flat[0].bound, usize::MAX));` → le test panique « index out of bounds: the len is 0 » |
| `nice -n 10 cargo test -p llvq-artifact --test hostile_files -j 8` | `a_ball_index_past_its_cap_is_refused_on_read` et `a_ball_record_of_cap_1_holds_the_origin_only` ok, les autres inchangés | (1) `format.rs:1182`, `if idx > max_idx` → `if false` : les deux tests rougissent à la lecture. (2) `format.rs:986` (`code_stream`) idem : rouge sur « raw writer ». (3) `format.rs:836` (`write_codes`) idem : rouge sur « point writer ». (4) `format.rs:650-653`, bras Ball → `u64::MAX` : tout rouge |
| `nice -n 10 cargo test -p llvq-artifact -j 8` | aucun test existant ne rougit (fixtures revues ci-dessus) | |
| `nice -n 10 cargo test -p llvq-quant --test g5_gptq a_dead_channel -j 8` | ok sur le chemin de référence | `linalg.rs:152` `of_inverse: false` → `true` : rouge |
| `nice -n 10 cargo test -p llvq-quant --features fast-linalg --test g5_gptq a_dead_channel -j 8` | ok sur le chemin `faer`, `at == 5` | `linalg.rs:115` `at: index` → `at: 0` : rouge SOUS `fast-linalg` SEULEMENT (vert sans la feature, c'est attendu) |
| `nice -n 10 cargo test -p llvq-bench --test tetra_refusals -j 8` | `rtbits_prices_a_tetra_file_instead_of_refusing_it` ok | `llvq-bench/src/bin/rtbits.rs:442` `==` → `!=` : rouge, la sortie lit « 96 bits, 2.0000 b/weight, 0 pad bits » et « … 2.6667 » |
| `nice -n 10 cargo test -p llvq-llm --test env_registry -j 8` | vert | |
| `nice -n 10 cargo test -p llvq-llm --bin smoke an_empty_codebook -j 8` | vert (commentaire seul) | |
| `nice -n 10 cargo clippy --all-targets -j 8` puis `nice -n 10 cargo clippy -p llvq-quant --features fast-linalg --all-targets -j 8` | zéro avertissement ; la seconde compile `fast::pivot` et l'import de `LltError` | |
| `cargo test --release -- --include-ignored` avec `~/llvq-q4b.llvq` présent | **EXIGÉE** : `llvq-artifact` et `llvq-search` sont touchés, et le lecteur Ball refuse désormais un index au-delà de `N(cap)`. Les tests d'archive doivent relire le fichier Ball cap 12 sans `IndexOutOfRange`. Compter les lignes `test result:`, ne pas tronquer | |
| `cargo test --release -p llvq-quant --features fast-linalg` | `both_factorizations_agree` et le test neuf verts sur le chemin payé | |
| sur Linux : `LLVQ_SEALED_ARTIFACT=… cargo test --release -p llvq-cuda --test golay70_decoder_matches_rust -- --ignored` | la relecture Ball cap 12 de l'archive passe aussi par le lecteur modifié | |


## 83-r2-fix-python-ops


| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `cd llvq-tetra && uv run --group dev pytest -q tests/test_quantizer.py -k "card or set_device or caller"` | tous verts : 5 + 5 cas existants, `test_device_map_cuda_after_set_device_loads_on_the_current_card`, 4 cas de `test_a_card_only_the_caller_can_have_named_is_refused` | `integration.py`, `_require_current_cuda` : supprimer les deux lignes `if index == rewritten: continue` ; attendu : `test_device_map_cuda_after_set_device…` rouge. Second mutant : remettre `set_device({index})` en `set_device` sans index ; attendu : les 4 cas `…caller_can_have_named…` rouges sur le motif |
| `cd llvq-tetra && uv run --group dev pytest -q tests/test_fused_device.py` | 6 tests existants verts, plus 3 + 1 + 1 nouveaux verts | `fused.py`, `FusedTetraLinear._apply` : retirer la ligne `_require_rotation_dtype(…)` ; attendu : les 3 cas `test_a_dtype_change_that_reaches_the_mix…` rouges. Second mutant : remplacer `rotation.k == 1` par `False` ; attendu : `test_a_rotation_without_a_mix_takes_any_dtype` rouge |
| `cd llvq-tetra && uv run --group dev python -c "import numpy as np, torch; from llvq_tetra.fused import Rotation; r = Rotation(np.ones(24), np.eye(3), 'cpu'); r.to(torch.float16); r(torch.ones(1, 24))"` | `RuntimeError` de dtype dans `einsum` : la prémisse de R2-DIF-44. Si cela passe, la garde refuse un cas qui marchait sur CPU ; vérifier alors sur une carte avant de garder la garde | aucun |
| `cd llvq-tetra && uv run --group dev pytest -q tests/test_quantizer.py -k vocabulary` | 2 + 4 verts | `vocab.py` : retirer `or desc.get("group") != 64` ; attendu : cas `group` 128 et 32 rouges |
| `cd llvq-tetra && uv run --group dev pytest -q` puis `cd llvq-tetra && uv run --group dev python -m llvq_tetra.checkmps <out_dir>` sur le Mac | suite verte ; `checkmps` charge le 4B en `device_map="mps"` : le refus de `Int4Vocab` ne touche pas un dossier `hfpack` | aucun |
| `cd ops/llvqtune && uv run --extra torch --with pytest python -m pytest tests -q` | suite verte (les tests de `main` sautent sans torch) ; nouveaux : `test_json_sink.py` 3, `test_loop.py` 3, `test_main_exit.py` 5, `test_mmlu_corpus.py` 1, `test_train_sh.py` 1 | (a) `train.sh` : remettre `if python -c "… sys.exit(1 if … else 0)"` sans `!` ; attendu : `test_an_unreadable_ceiling_is_refused` rouge (code 0, deux appels). (b) `__main__.py` : retirer le bloc `if not outcome.complete:` ; attendu : `test_a_run_cut_short_exits_6…` rouge. (c) `loop.py` : retirer `steps=(done, plan.steps)` ; attendu : `test_every_checkpoint_says_how_far_the_run_got` rouge. (d) `__main__.py` : retirer le refus `UNSAVED_MODES` ; attendu : les deux cas `…saves_nothing…` rouges (`AssertionError` de `_NoWeights`). (e) `__main__.py` : remettre `return 0 if outcome.improved else 1` en fin ; attendu : `test_a_full_streaming_run_exits_0…` rouge. (f) `mmlu_aux_overlap.py` : changer un caractère de `REVISION` ; attendu : `test_the_overlap_check_reads…` rouge |
| Banc sans coût de `43-lot4-outillage.md` (bloc `cd ops && HF_TOKEN=dummy uv run --with 'huggingface-hub>=1.26' python - <<'PY' …`), avec ces `cases` : `["oracle", "--image", "hf.co/spaces/Pier-Jean/llvq-runner-cuda-sm80", "--flavor", "a100-large", "--kernels-cap", "80"]`, `["oracle", "--image", "x", "--flavor", "a100-large"]`, `["oracle", "--image", "x"]` | 1 : `compute cap: sm_80 against kernels declared for sm_80 (…)`, la ligne de coût `a100-large at 2.50 $/h …`, puis `would launch: timeout 20m secrets None`. 2 : `refused, a100-large: sm_80 < sm_89, …`, `returned 2`, aucun `would launch`. 3 : inchangé (`l4x1 at 0.80 $/h …` puis `would launch`) | `run.py`, `cmd_oracle` : supprimer le bloc `if args.kernels_cap is not None:` ; attendu : le cas 1 rend 2 |
| `cd ops && python3 -c "import run; f = run._is_runner_image; ok = ['hf.co/spaces/Pier-Jean/llvq-runner-cuda', 'https://huggingface.co/spaces/Pier-Jean/llvq-runner-cuda', 'https://hf.co/spaces/Pier-Jean/llvq-runner-cuda-sm80', 'huggingface.co/spaces/Pier-Jean/llvq-runner-cuda-qualite', 'hf.co/spaces/Pier-Jean/llvq-runner-cuda:abc']; no = [None, '', 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime', 'vllm/vllm-openai:v0.26.0', 'hf.co/spaces/Pier-Jean/other', 'hf.co/spaces//llvq-runner', 'hf.co/spaces/Pier-Jean/llvq-runner/x', 'ghcr.io/hf.co/spaces/a/llvq-runner']; print([f(i) for i in ok], [f(i) for i in no]); assert all(map(f, ok)) and not any(map(f, no))"` | deux listes, cinq `True` puis huit `False`, aucune `AssertionError` (`run.py` n'importe que la bibliothèque standard au chargement) | `run.py`, `SPACE_IMAGE_PREFIXES` : retirer `"https://huggingface.co/spaces/",` ; attendu : `AssertionError`, la deuxième valeur `ok` vaut `False` |
