# Patch proposé pour `CLAUDE.md` (lot 1, non appliqué)

Base : `CLAUDE.md` du worktree à 925930d. Chaque bloc donne un texte actuel exact (sous-chaîne unique du fichier) et
son remplacement, en anglais, avec la ligne de code qui le justifie. Rien n'est appliqué : l'opérateur décide, bloc
par bloc. Aucun chiffre de mesure ne change ; le « about 8 minutes » de la CI est gardé tel quel, et il n'a pas été
re-mesuré avec cinq jobs.

## 1. COH-01, ligne 123 : les deux configs Metal NOT-SERVED se chargent sur CUDA

Justification : `llvq-llm/src/served.rs:942`, `(AttnHeads::Expanded, _) | (_, false) => Ok(())` ; aucun code ne lit le
suffixe `NOT-SERVED` ; les champs des deux fichiers sont ceux de `-kernel-from1280.json` et `-grouped.json` ; la ligne
`served config:` imprime la note (`served.rs:854`, `:893-894`).

Texte actuel :
```
the `*-metal-*-NOT-SERVED.json` files load only in `bin/metalrun`.
```
Texte proposé :
```
on Metal the `*-metal-*-NOT-SERVED.json` files load only in `bin/metalrun`; on CUDA nothing refuses them, they resolve like `-kernel-from1280.json` and `-grouped.json`, and the `served config:` line prints their NOT SERVED note.
```

## 2. COH-04, ligne 123 : le drafter 0.6B (précision, facultative)

Le vérificateur (21-verif-moteur-runtime.md) juge l'oubli défendable : le drafter n'est pas un objet servi. Bloc
facultatif. Justification : `served.rs:540-545`, quatre formes dans `SEALED`, dont « 0.6B drafter ».

Texte actuel :
```
`decode` (`persistent` = 4E, only beside `kernel`, on the three sealed shapes)
```
Texte proposé :
```
`decode` (`persistent` = 4E, only beside `kernel`, on the three sealed shapes and the 0.6B drafter of `configs/qwen3-0p6b-tetra-e4-4e-v9.json`)
```

## 3. COH-08, lignes 69-70 : cinq jobs de CI, pas quatre

Justification : `.github/workflows/ci.yml:89` (`lint`, dont l'étape `forbid(unsafe_code)` l. 127), `:142` (`python` :
listes de l'image l. 151, refus de la porte MLX l. 153, `llvqtune` sans torch l. 155), `:169` (`test`), `:197`
(`core-debug`), `:225` (`faer`).

Texte actuel :
```
CI (`.github/workflows/ci.yml`) runs four jobs in parallel in about 8 minutes: clippy with the dependency-free check,
the tests under the `ci` profile, the core crates in debug, and `llvq-quant` with `faer`.
```
Texte proposé :
```
CI (`.github/workflows/ci.yml`) runs five jobs in parallel in about 8 minutes: clippy with the dependency-free and
`forbid(unsafe_code)` checks, the Python job (the CUDA image's three lists, the MLX door's refusals, `llvqtune`
without torch), the tests under the `ci` profile, the core crates in debug, and `llvq-quant` with `faer`.
```

## 4. COH-08, ligne 150 : défaut de `LLVQ_MODULE_CACHE`

Justification : `llvq-cuda/src/modcache.rs:174-176` et `:191-196`, `$XDG_CACHE_HOME/llvq/modules` d'abord, sinon
`$HOME/.cache/llvq/modules`. `docs/ENV.md:34` le dit déjà.

Texte actuel :
```
| `LLVQ_MODULE_CACHE` | absolute directory, `off`, unset = `~/.cache/llvq/modules` |
```
Texte proposé :
```
| `LLVQ_MODULE_CACHE` | absolute directory, `off`, unset = `$XDG_CACHE_HOME/llvq/modules`, else `~/.cache/llvq/modules` |
```

## 5. COH-12, ligne 80 : le dixième positionnel de `smoke`

Justification : `llvq-llm/src/bin/smoke.rs:665-667`, `let save_to = a.get(9).cloned();`, « Where to persist the
quantized projections ».

Texte actuel :
```
#   positional: n_calib · calib_len · n_eval · eval_ctx · device · nogs/gs/dc/sph · codebook (suffix f = free magnitude, L<n> = cap) · limit · rot
```
Texte proposé :
```
#   positional: n_calib · calib_len · n_eval · eval_ctx · device · nogs/gs/dc/sph · codebook (suffix f = free magnitude, L<n> = cap) · limit · rot · save_to (optional directory for the quantized projections)
```

## 6. COH-12, ligne 94 : `fusedrun` exige le fichier

Justification : `llvq-llm/src/bin/fusedrun.rs:2097-2100`, « usage: fusedrun <model.llvq> [n_tokens] », 32 jetons par
défaut.

Texte actuel :
```
cargo run --release -p llvq-llm --features cuda --bin fusedrun                                      # the kernel in the model
```
Texte proposé :
```
cargo run --release -p llvq-llm --features cuda --bin fusedrun -- <sealed> [n_tokens]               # the kernel in the model
```

## 7. COH-14, lignes 46-47 : dépendances externes de `llvq-llm` et `llvq-runtime`

Justification : `llvq-llm/Cargo.toml` (`[dependencies]` : `candle-core`, `candle-nn`, `candle-transformers`, `half`,
`tokenizers`, `hf-hub`, `serde`, `serde_json`, `anyhow`, `flate2`, `rayon`, `parquet` ; macOS, optionnels sous
`metal` : `candle-metal-kernels`, `objc2-metal`) ; `llvq-runtime/Cargo.toml` (`[dependencies]` : `candle-core`,
`tokenizers`, `anyhow`, `serde`, `serde_json`, `axum`, `tokio`, `futures-util`, `clap`, `utoipa`, `utoipa-axum`,
`sha2`, `ureq`, `native-tls`).

Texte actuel :
```
| `candle`, `tokenizers`, `hf-hub`, `parquet`, `rayon` | allowed |
```
Texte proposé :
```
| `candle`, `tokenizers`, `hf-hub`, `parquet`, `rayon`, `serde`, `serde_json`, `anyhow`, `flate2`, `half`; on macOS under `metal`, `candle-metal-kernels`, `objc2-metal` | allowed |
```

Texte actuel :
```
| `axum`, `tokio`, `clap`, `utoipa`, `sha2`, `ureq` | forbid |
```
Texte proposé :
```
| `axum`, `tokio`, `futures-util`, `clap`, `utoipa`, `utoipa-axum`, `sha2`, `ureq`, `native-tls`, `serde`, `serde_json`, `anyhow`, `candle-core`, `tokenizers` | forbid |
```

## 8. NUM-03, lignes 55-56 : `faer` et l'algèbre maison s'accordent à l'arrondi, pas au bit

Justification : `llvq-quant/tests/g5_gptq.rs:1036`, `worst <= 1e-8 * scale.max(1.0)` ; `llvq-quant/src/linalg.rs:15-22`
dit déjà « agree to rounding, not to the bit, and codes encoded on the two paths may differ » (22-verif-format-harnais.md,
NUM-03 CONFIRMÉ).

Texte actuel :
```
Without
`--features fast-linalg` the factorization is about 40 times slower for a bit-identical result. The in-house algebra
(`llvq-quant/src/linalg.rs`) stays the reference: `both_factorizations_agree` requires the same factor as `faer`. Do
not delete it.
```
Texte proposé :
```
Without
`--features fast-linalg` the factorization is about 40 times slower, and its factor agrees with `faer`'s to rounding,
not to the bit, so codes encoded on the two paths may differ. The in-house algebra (`llvq-quant/src/linalg.rs`) stays
the reference: `both_factorizations_agree` requires the two factors to agree within `1e-8 · max(max|U|, 1)`. Do not
delete it.
```

## 9. OPT-06, lignes 66 et 74 : la suite complète sous le profil `ci`

Justification : `Cargo.toml:24-37`, `[profile.ci]` hérite de `release` (les tests `cfg_attr(debug_assertions,
ignore)` tournent), sans LTO, 16 unités, `overflow-checks = true` ; le commentaire dit que les flottants n'en dépendent
pas. Le temps local des deux commandes n'est pas mesuré (vérification proposée par 07-optimisation.md : un
chronométrage par la session principale). À aligner aussi dans `docs/METHODE.md` s'il cite la commande (hors de mon
domaine).

Texte actuel (ligne 66) :
```
`cargo test --release -- --include-ignored` takes tens of
minutes.
```
Texte proposé :
```
`cargo test --profile ci -- --include-ignored` takes tens of
minutes.
```

Texte actuel (ligne 74) :
```
cargo test --release -- --include-ignored             # full suite, before any format commit
```
Texte proposé :
```
cargo test --profile ci -- --include-ignored          # full suite, before any format commit (release opt level, no LTO, overflow checks on)
```

## 10. TST-09, après la ligne 73 : la boucle des tests Metal

Aucun document vivant hors `CLAUDE.md` ne liste les boucles de test (`README.md:156` renvoie à `CLAUDE.md`), donc la
ligne va ici. Justification : `llvq-llm/tests/{attn_metal,q4_metal_candle,rot_metal_tg,tetra48_metal_candle,
tetra48_metal_rows}.rs` portent `#![cfg(all(target_os = "macos", feature = "metal"))]` (6 + 6 + 1 + 12 + 10 tests),
`llvq-llm/src/fused_metal.rs` 3 de plus sous `lib.rs:50` ; la feature `metal` n'est pas par défaut
(`llvq-llm/Cargo.toml:51`), ni la CI ni `ops/` ne la passent à `cargo test`. Ces tests n'ont pas été lancés dans ce lot
(la feature `metal` est interdite par le brief) : leur état vert est à constater.

Texte actuel :
```
cargo test                                            # fast loop
```
Texte proposé :
```
cargo test                                            # fast loop
cargo test -p llvq-llm --features metal               # macOS: the 38 Metal kernel tests, which no other loop and no CI job runs
```

## 11. CONNU-NON-CORRIGÉ (10-coherence.md), ligne 131 : `LLVQ_DTYPE`

Justification : `llvq-llm/src/eval.rs:82-84` accepte `f16`, `bf16`, `f32` et leurs alias ; `bin/ppl.rs:51` défaut
`F32`, `bin/mmlu.rs:499` défaut `F16`. `docs/ENV.md:56` le dit déjà.

Texte actuel :
```
| `LLVQ_DTYPE` | `f32` (`ppl` default), `f16` | evaluation dtype;
```
Texte proposé :
```
| `LLVQ_DTYPE` | `f32` (`ppl` default), `f16` (`mmlu` default), `bf16` | evaluation dtype;
```

## 12. CONNU-NON-CORRIGÉ (10-coherence.md), ligne 143 : `LLVQ_SEQ_BLOCK`

Justification : `llvq-llm/src/bin/smoke.rs:1166-1170`, unset, vide ou `0` donnent `false`, `1` donne `true`, toute
autre valeur est refusée par son nom.

Texte actuel :
```
any other value of `LLVQ_SEQ_BLOCK` is refused by name.
```
Texte proposé :
```
`LLVQ_SEQ_BLOCK` takes `0` (default) or `1`, and any other value is refused by name.
```
