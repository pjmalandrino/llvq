# Cohérence code / docs / configs : constats
Périmètre réellement lu : `CLAUDE.md` (worktree, 925930d) en entier ; `docs/ETAT.md` §1 à §4 ; `configs/README.md` en entier ; les 31 `configs/*.json` (champs extraits) ; `llvq-llm/src/served.rs` l. 1-1000 ; `llvq-llm/tests/served_config.rs` l. 140-330 ; `llvq-llm/tests/env_registry.rs` ; `llvq-llm/tests/hfpack.rs` l. 200-280 ; `ops/Dockerfile.cuda` l. 55-242 ; `ops/run.py` `UPLOAD_ALLOW` ; `.github/workflows/ci.yml` en entier ; `llvq-runtime/library.json`, `llvq-runtime/README.md`, `llvq-runtime/src/library.rs` l. 95-260 ; tous les `Cargo.toml` ; `docs/data/jobs.csv` (contrôle de forme par script) ; `docs/format-noyau.md` §8 ; `docs/ENV.md` l. 1-40.
Survolé : en-têtes `//!` de `persist4e_cuda.rs`, `served_decode.rs`, `fused_cuda.rs`, `kvq.rs`, `batch.rs`, `llvq-cuda/src/{lib,pers,tile,gpu}.rs`, `llvq-runtime/src/{lib,main,engine}.rs` ; `bin/smoke.rs` l. 625-690 et 1160-1170 ; `bin/fusedrun.rs` l. 2068-2160 ; `llvq-cuda/src/tile.rs` ; `eval.rs` ; `docs/plan-transformers.md` ; `docs/audit-architecture-2026-11.md` §1.5.
Non couvert : `docs/MODES.md`, `docs/CERVEAU.md`, `docs/how-it-works.md`, `docs/ROADMAP*.md` ; le reste de `docs/ENV.md` ligne par ligne contre chaque lecteur ; les `//!` des modules hors chemin servi ; `ops/` au-delà des scripts nommés dans `CLAUDE.md` ; `llvq-tetra/README.md` au-delà de la version.
Bilan : 0 critiques, 0 majeures, 6 mineures, 8 cosmétiques

Note de méthode : le `CLAUDE.md` injecté dans ma session (celui du dépôt parent) n'est pas celui du worktree à 925930d. Tous les constats ci-dessous visent le fichier du worktree. Plusieurs écarts du fichier injecté (huit crates, mur `rot_apply` « whatever the format », `grouped` « seul fichier ») sont déjà corrigés à 925930d et ne sont pas rapportés.

Contrôles faits sans constat :
- Les deux listes de `ops/Dockerfile.cuda` : 30 binaires construits (l. 55-57, 61, 88, 94), les mêmes 30 dans le `COPY` (l. 143-174). Les 5 `COPY --from=build /src/...` (l. 179, 196, 199, 201, 203) sont couverts par `UPLOAD_ALLOW` (`ops/run.py:747-763`). `ops/check_image_lists.py` le vérifie en CI.
- Les copies de `llvq-tetra/llvq_tetra/data/` : les 4 shaders Metal et les 6 sources CUDA sont identiques octet pour octet à leur source (`cmp`). `the_shipped_shader_is_the_repositorys` (`llvq-llm/tests/hfpack.rs:240`) compare octets et sha256, sans `#[ignore]`, et tourne dans le job `test` de la CI.
- `ServedFile` et `BatchFile` portent `#[serde(deny_unknown_fields)]` (`served.rs:94`, `:154`) : un champ mal orthographié est refusé. `every_shipped_config_resolves` (`served_config.rs:253`) parse tous les `configs/*.json`.
- `llvq-runtime/library.json` contre `docs/ETAT.md` §2 : configs CUDA, Metal, CPU, `cuda_batch`, `cuda_eager` concordent ; tailles et sha256 des trois fichiers scellés égaux à ceux de `configs/README.md`.
- Neuf crates, `forbid(unsafe_code)` présent dans les cinq crates cœur et dans `llvq-runtime` (lib et main), contrôlé en CI (`ci.yml` lint).
- Le paragraphe de `CLAUDE.md:157-161` sur `MEASUREMENT_MODES`, `ALONE` et `BELOW_THE_DOOR` est exact : 17 noms refusés par `fusedrun.rs:2133-2146`, 19 par `served.rs:516-521`, 7 dans `BELOW_THE_DOOR`.
- Tous les binaires, scripts `ops/` et tests nommés dans le bloc « Commands » existent, dans la crate dite.
- `docs/data/jobs.csv` : 367 lignes, 8 colonnes partout.

### COH-01 Les deux configs Metal NOT-SERVED se chargent sur CUDA, alors que deux documents disent « seulement dans metalrun »
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `CLAUDE.md:123` ; `configs/README.md:55`, `:168-169` ; `llvq-llm/src/served.rs:918-961`
- Fait : `CLAUDE.md:123` dit « the `*-metal-*-NOT-SERVED.json` files load only in `bin/metalrun` », `configs/README.md:55` « Loaded only by `bin/metalrun` ». `check_device` ne refuse `grouped` et `kernel` que si `metal` est vrai (`served.rs:942-943` : `(_, false) => Ok(())`). Aucun code ne lit le suffixe `NOT-SERVED` du nom (seul le fichier kv8 est refusé, par son champ `decode_cache`).
- Scénario : `LLVQ_CONFIG=configs/qwen3-4b-tetra-e4-metal-kernel-NOT-SERVED.json mmlu <sealed> cuda 40` résout et score par le noyau, exactement comme `qwen3-4b-tetra-e4-kernel-from1280.json` ; le fichier `metal-grouped` se comporte comme `qwen3-4b-tetra-e4-grouped.json`. La ligne `served config:` imprime la note « NOT SERVED », donc rien de silencieux, mais la phrase des deux documents est fausse.
- Touche : doc
- Correction proposée : remplacer dans `CLAUDE.md:123` par « on Metal they load only in `bin/metalrun`; on CUDA they resolve like `-kernel-from1280.json` and `-grouped.json`, their note saying NOT SERVED », et la même précision dans `configs/README.md:55` et `:168-169`.
- Effort : S
- Risque de la correction : nul (texte seul ; ne pas toucher les fichiers de config, dont le sha256 est épinglé par les journaux).
- Vérification proposée : aucune, lecture suffisante.

### COH-02 `llvq-runtime/README.md` décrit le runtime d'avant le préremplissage en chaîne et d'avant `LLVQ_BATCH`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-runtime/README.md:17-19`, `:42-43` ; `llvq-runtime/library.json` (entrée 4b, `"cuda": "qwen3-4b-tetra-e4-4e-v9-chain.json"`) ; `llvq-runtime/src/engine.rs:13-20` ; `llvq-runtime/src/library.rs:99-130`
- Fait : le README (dernier commit 2026-11-08) donne « 4E v9 (greedy only) » comme config CUDA des trois tailles et dit « The engine decodes one sequence at a time, so requests to one model queue ». Le code charge pour le 4B sur CUDA la config chaîne (préremplissage par la passe d'arbre) depuis le 2026-11-15, accepte `LLVQ_PROFILE=eager`, et sous `LLVQ_BATCH` sert en lots continus (`engine.rs:13-20`). Le README ne nomme ni `LLVQ_BATCH`, ni `LLVQ_PROFILE`, ni le cache de modules.
- Scénario : un lecteur du README conclut que `llvq serve` ne sert jamais deux requêtes à la fois et que le 4B préremplit par le chemin eager ; c'est la page que `README.md:91` donne comme porte d'entrée.
- Touche : doc
- Correction proposée : colonne CUDA du 4B : « 4E v9, chain prefill (`LLVQ_PROFILE=eager`: eager prefill), greedy only ». Remplacer l. 42 par « By default the engine decodes one sequence at a time, so requests to one model queue. Under `LLVQ_BATCH=<n>[:<positions>]` on CUDA it batches continuously through the `*-4e-v9-batch.json` configs, greedy only. » Ajouter une ligne sur `LLVQ_MODULE_CACHE` (premier démarrage ~8,5 min).
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### COH-03 `configs/README.md` se dit l'autorité sur chaque fichier et ne mentionne pas `qwen3-0p6b-tetra-e4-4e-v9.json`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `configs/README.md:12`, tables l. 20-31 et 37-61 ; `configs/qwen3-0p6b-tetra-e4-4e-v9.json` ; `ops/Dockerfile.cuda:227` ; `llvq-llm/src/served.rs:540-545`
- Fait : « This README is the authority on every file here » (l. 12). `grep 0p6b configs/README.md` ne renvoie rien. Le fichier existe (commit 4957972), l'image le vérifie par `test -f` (Dockerfile l. 227) et `check_persistent_object` accepte sa forme (« 0.6B drafter »).
- Scénario : un lecteur qui cherche l'état de ce fichier (servi ? mesuré ?) ne trouve rien dans l'autorité désignée ; `ETAT.md` ne le nomme pas non plus.
- Touche : doc
- Correction proposée : ajouter une ligne au tableau « At a glance » : « drafter of the speculative decode, never served: `qwen3-0p6b-tetra-e4-4e-v9.json` (`specrun`, `specbench`) », et une ligne au tableau « Every file » avec l'objet `qwen3-0p6b-sealed.bin` et la prereg `proofs/preregistration-draft-0p6b-2026-11-07.md`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### COH-04 « Refusé sur toute forme sauf le 4B » : faux depuis le 2026-11-07, et le drafter 0.6B est oublié partout
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `configs/README.md:43`, `:132` ; `CLAUDE.md:123` ; `docs/ETAT.md:128` ; `llvq-llm/src/served.rs:529-554`
- Fait : `configs/README.md:43` (fichier `-4e.json`) dit « Refused on Metal, on any shape but the 4B's », et l. 132 « refused on Metal, on any shape but the Qwen3-4B's », contredit par la phrase suivante (l. 134 « accepted on the three sealed shapes »). `CLAUDE.md:123` et `ETAT.md:128` disent « on the three sealed shapes ». Le code accepte quatre formes : 0.6B drafter, 4B, 8B, 14B (`served.rs:540-545`).
- Scénario : un lecteur qui veut servir le 8B en 4E lit en l. 132 que c'est refusé ; un lecteur du drafter lit dans `CLAUDE.md` que sa config serait refusée.
- Touche : doc
- Correction proposée : `configs/README.md:132` : « refused on Metal, on any shape but the sealed 4B, 8B and 14B and the 0.6B drafter ». l. 43 : même formule. `CLAUDE.md:123` et `ETAT.md:128` : « on the three sealed shapes and the 0.6B drafter ».
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### COH-05 `llvq-tetra/pyproject.toml` dit 0.2.0, aucun document vivant ne dit où en est 0.2.0
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-tetra/pyproject.toml:3` ; `docs/plan-transformers.md:38-39`, `:279`, `:385` ; `docs/ETAT.md:253` ; `docs/upstream/transformers-llvq-tetra/README.md:27`
- Fait : le paquet est en `version = "0.2.0"` depuis 567f6fe (« llvq-tetra 0.2.0: compressed on Metal with no compiler… »). `plan-transformers.md`, qui « carries its stages » d'après `CLAUDE.md`, et `ETAT.md:253` disent seulement que 0.1.0 est sur PyPI. `0.2.0` n'apparaît dans aucun document vivant ; seule la case non cochée de `docs/upstream/.../README.md:27` (« If 0.2.0 ships before the post ») l'évoque.
- Scénario : un lecteur ne peut pas savoir si `pip install llvq-tetra` (README.md) installe 0.1.0 ou 0.2.0, ni si le Metal compressé sans compilateur est publié.
- Touche : doc
- Correction proposée : ajouter une ligne d'étape à `plan-transformers.md` : « 0.2.0 (567f6fe): Metal compressed without a compiler, the vocabulary in 4 bits, rows a dispatch; on PyPI: <oui, date | non> », et aligner `ETAT.md:253`. L'état PyPI réel est à constater par l'opérateur (hors réseau ici).
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : `pip index versions llvq-tetra` par l'opérateur.

### COH-06 La CI ignore `docs/**` alors qu'un test garde `docs/ENV.md`
- Gravité : mineure
- Confiance : PLAUSIBLE
- Où : `.github/workflows/ci.yml` (`on.push.paths-ignore` et `on.pull_request.paths-ignore` : `'**.md', 'docs/**'`) ; `llvq-llm/tests/env_registry.rs:56-80`
- Fait : `every_quoted_variable_is_in_the_registry` lit `docs/ENV.md` et échoue si un `"LLVQ_*"` cité par le code n'y figure pas. Un PR qui ne touche que `docs/ENV.md` (ou que des `.md`) ne déclenche pas la CI.
- Scénario : un PR de rangement retire une ligne de `docs/ENV.md` encore lue par le code ; il passe sans CI et est fusionné ; le PR de code suivant, sans rapport, échoue sur `env_registry` et porte la faute.
- Touche : outillage
- Correction proposée : retirer `docs/ENV.md` de l'ignoré, par une exception (`'!docs/ENV.md'` n'existe pas en `paths-ignore` ; passer à `paths:` avec la liste positive, ou ajouter un job léger déclenché sur `docs/ENV.md` qui lance seulement `cargo test -p llvq-llm --test env_registry`).
- Effort : S
- Risque de la correction : faible (CI seule).
- Vérification proposée : un PR de test qui supprime la ligne `LLVQ_CONFIG` de `docs/ENV.md` doit déclencher et faire échouer la CI.

### COH-07 `configs/README.md` compte mal les champs d'une config
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `configs/README.md:63-65`, `:105-106`, `:181-183` ; `llvq-llm/src/served.rs:95-150`
- Fait : l. 64-65 « five required fields and four optional ones » ; l. 105 « Three are optional: attention, … decode and rows ». `ServedFile` a 5 champs requis et 7 optionnels qui changent ce qui tourne (`attention`, `decode`, `rows`, `prefill_attention`, `prefill`, `decode_cache`, `batch`), plus `note`. La liste des parseurs l. 181-183 omet `PrefillAttn::parse`, `Prefill::parse`, `DecodeCache::parse` et `BatchServe::of`.
- Scénario : un lecteur qui écrit une config d'après le README ignore trois champs refusés ou acceptés par le code.
- Touche : doc
- Correction proposée : l. 64-65 « five required fields and seven optional ones, plus `note` » ; l. 105 « Seven are optional: `attention` (2026-11-02), `decode` and `rows` (2026-11-05), `prefill_attention` and `batch` (2026-11-08), `prefill` and `decode_cache` (2026-11-15). » ; compléter la liste l. 182-183.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### COH-08 `CLAUDE.md` annonce quatre jobs de CI, `ci.yml` en lance cinq
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `CLAUDE.md:69-70` ; `.github/workflows/ci.yml:77` et jobs `lint`, `python`, `test`, `core-debug`, `faer`
- Fait : `CLAUDE.md` liste « clippy …, the tests under the `ci` profile, the core crates in debug, and `llvq-quant` with `faer` ». Le job `python` (listes de l'image par `ops/check_image_lists.py`, refus de la porte MLX, `ops/llvqtune`) manque ; le job `lint` vérifie aussi `forbid(unsafe_code)`. Dans la même table, `LLVQ_MODULE_CACHE` (l. 150) dit « unset = `~/.cache/llvq/modules` » alors que `llvq-cuda/src/modcache.rs:175-195` prend d'abord `$XDG_CACHE_HOME/llvq/modules`.
- Scénario : un lecteur croit que le piège des deux listes n'est vérifié qu'à la construction payante du Space.
- Touche : doc
- Correction proposée : `CLAUDE.md:69-70` : « runs five jobs in parallel: clippy with the dependency-free and `forbid(unsafe_code)` checks, the Python job (the image's three lists, the MLX door's refusals, llvqtune), the tests under the `ci` profile, the core crates in debug, and `llvq-quant` with `faer`. » ; l. 150 : « unset = `$XDG_CACHE_HOME/llvq/modules`, else `~/.cache/llvq/modules` ».
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### COH-09 Deux commentaires du runtime nomment une config et une valeur que le code n'a plus
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-runtime/src/main.rs:14-18` ; `llvq-runtime/src/engine.rs:666-667` ; `llvq-runtime/src/library.rs:99-110`
- Fait : `main.rs:16` dit que `LLVQ_BATCH` prend « the model's batch config (`configs/qwen3-*-tetra-e4-batch.json`) » ; `library.json` nomme `qwen3-*-tetra-e4-4e-v9-batch.json` en `cuda_batch`, et les `*-e4-batch.json` ne sont pas compilés dans le binaire (`library.rs:196-235`). `engine.rs:667` dit « the chain … under a chain config (`LLVQ_PROFILE=chain`) » ; `config_for` refuse `LLVQ_PROFILE=chain` par nom (`library.rs:107-112`).
- Scénario : `llvq --help` ou une lecture du source oriente vers une config non embarquée et une valeur refusée.
- Touche : runtime
- Correction proposée : `main.rs:16` : « (`configs/qwen3-*-tetra-e4-4e-v9-batch.json`) » ; `engine.rs:667` : « under a chain config (the 4B's default on CUDA; `LLVQ_PROFILE=eager` takes the eager prefill) ».
- Effort : S
- Risque de la correction : nul (commentaires).
- Vérification proposée : aucune, lecture suffisante.

### COH-10 Trois en-têtes de module décrivent un comportement servi périmé
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/persist4e_cuda.rs:4-5`, `:18-20` ; `llvq-llm/src/fused_cuda.rs:18-23` ; `llvq-llm/src/served.rs:44-47`
- Fait : `persist4e_cuda.rs:18-20` donne comme « served 4B config » `configs/qwen3-4b-tetra-e4-4e-v4.json` « and its v6 twin » ; le 4B est servi en v9 depuis le 2026-11-07 et par la chaîne depuis le 2026-11-15 (`configs/README.md:22-23`), et v6 n'a jamais été servi (`configs/README.md:27`). `fused_cuda.rs:18-23` dit « It does not handle more than one token per call … `bin/ppl` is untouched » ; le module a `forward_rotated_rows` (l. 1594) et lance `tv_tetra48_rows_h` pour le préremplissage (l. 3229-3295). `served.rs:44-47` dit « four variables that reach BELOW the served door » ; `BELOW_THE_DOOR` en a sept (`served.rs:570-578`, les trois `LLVQ_GUIDED_*` en plus).
- Scénario : un relecteur de noyau qui part du `//!` cherche la config v4 et croit le préremplissage fait token par token.
- Touche : doc
- Correction proposée : `persist4e_cuda.rs:18-20` : « The served configs, `configs/qwen3-{4b,8b,14b}-tetra-e4-4e-v9.json` (and the 4B's chain config), name the persistent decode… » ; `fused_cuda.rs:18-23` : remplacer la puce par « A prompt goes through `forward_rotated_rows` (the rows kernel, up to `MAX_PREFILL_ROWS` a chunk); a decode token through the one-row kernel. » ; `served.rs:44` : « seven variables … `LLVQ_KERNEL_DIR`, `LLVQ_TILE_BLOCKS`, `LLVQ_PREFILL`, `LLVQ_METAL_KERNEL` and the three `LLVQ_GUIDED_*` ».
- Effort : S
- Risque de la correction : nul si seuls les `.rs` sont touchés ; aucun de ces fichiers n'est un noyau épinglé.
- Vérification proposée : aucune, lecture suffisante.

### COH-11 `docs/data/jobs.csv` : six liens de journal ne mènent à rien sous `docs/mesures/`, et l'ordre des dates casse
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `docs/data/jobs.csv` lignes 13, 27, 80, 119-122, 238, 341-345
- Fait : la forme est saine (8 colonnes sur les 367 lignes). Mais la colonne `mesures_file` vaut, ligne 80 `m3-gptq-2026-08-30.txt`, lignes 119-121 `volume-2026-09-07.txt`, ligne 122 `q5-sur-v32-2026-09-07.txt` : aucun de ces fichiers n'existe dans le dépôt (existent `m3-gptq2-*-2026-08-30.txt` et des preregs `volume-*`, `q5-sur-v32-*`). Lignes 13, 27 et 238, le fichier est sous `docs/archive/`, pas sous `docs/mesures/`. Lignes 341-345 datées 2026-11-13 après une ligne du 2026-11-14.
- Scénario : la règle dure 9 (épuiser les traces avant de recoûter un run) mène à un journal introuvable pour ces cinq jobs.
- Touche : doc
- Correction proposée : corriger les six cellules vers le chemin réel (`archive/…` ou le journal qui porte le run, à retrouver par le `job_id`), ou écrire `absent` ; trier les lignes 341-345.
- Effort : S
- Risque de la correction : nul (ne pas toucher aux montants).
- Vérification proposée : relancer le petit script de contrôle de forme (8 colonnes, `docs/mesures/<f>` ou chemin explicite existant).

### COH-12 Bloc « Commands » de `CLAUDE.md` : deux lignes ne correspondent pas à ce que lit le binaire
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `CLAUDE.md:80`, `:94` ; `llvq-llm/src/bin/smoke.rs:629-667` ; `llvq-llm/src/bin/fusedrun.rs:2096-2100`
- Fait : l. 80 liste neuf positionnels de `smoke` ; `smoke.rs:667` en lit un dixième, `save_to = a.get(9)`, le répertoire où persister les projections. l. 94 lance `fusedrun` sans argument ; `fusedrun.rs:2097-2100` exige `<model.llvq>` et sort sur « usage: fusedrun <model.llvq> [n_tokens] ».
- Scénario : la ligne copiée telle quelle échoue à l'usage ; le dixième argument reste inconnu de qui ne lit que la carte.
- Touche : doc
- Correction proposée : l. 80 : ajouter « · save_to (optional directory) » ; l. 94 : `… --bin fusedrun -- <sealed> [n_tokens]`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### COH-13 Dates et formes servies périmées dans les documents vivants
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `docs/ETAT.md:1` ; `configs/README.md:47`, `:48`
- Fait : `ETAT.md:1` titre « Project state as of 2026-11-07 » alors que le document énonce des faits du 2026-11-15 (chaîne, l. 22-24) et du 2026-11-17 (l. 64-65). `configs/README.md:47-48` dit de la config v9 du 8B et du 14B « not the grouped form the 8B serves » / « the 14B serves » ; le 8B et le 14B servent v9 depuis le 2026-11-07 (même fichier l. 22).
- Scénario : un lecteur date l'état au 7 novembre et croit le 8B encore servi en `grouped`.
- Touche : doc
- Correction proposée : `ETAT.md:1` : « Project state as of 2026-11-17 » ; `configs/README.md:47` « not the grouped form the 8B served before it », idem l. 48 pour le 14B.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### COH-14 Table « Architecture » : dépendances externes incomplètes pour `llvq-llm` et `llvq-runtime`
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `CLAUDE.md:46-47` ; `llvq-llm/Cargo.toml` ; `llvq-runtime/Cargo.toml`
- Fait : `llvq-llm` dépend aussi de `serde`, `serde_json`, `anyhow`, `flate2`, `half`, et sous macOS de `candle-metal-kernels` et `objc2-metal`. `llvq-runtime` dépend aussi de `utoipa-axum`, `futures-util`, `native-tls`, `serde`, `serde_json`, `anyhow`, `candle-core`, `tokenizers`. La colonne des autres crates est exhaustive, ce qui laisse lire celles-ci comme exhaustives.
- Scénario : une revue de licences ou de surface d'attaque qui part de la carte manque `native-tls` et `ureq` côté réseau, et les crates Metal.
- Touche : doc
- Correction proposée : soit compléter les deux cellules, soit écrire « main ones: … » dans l'en-tête de colonne pour ces deux lignes.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### Anciens constats toujours présents
- CONNU-NON-CORRIGÉ : `CLAUDE.md:131`, `LLVQ_DTYPE` donne `f32`, `f16` ; `eval.rs:28` accepte aussi `bf16` et `mmlu` prend `f16` par défaut (`docs/audit-architecture-2026-11.md` §1.5 ; `docs/ENV.md:56` est juste).
- CONNU-NON-CORRIGÉ : `CLAUDE.md:143`, « any other value of `LLVQ_SEQ_BLOCK` is refused by name » ; `smoke.rs:1166-1170` accepte `1` (`docs/audit-architecture-2026-11.md` §1.5). Texte proposé : « values other than `0` and `1` are refused by name ».
- CONNU-NON-CORRIGÉ : `proofs/README.md:20` compte encore 22 préenregistrements ; `proofs/*.md` en compte 347 fichiers (`docs/audit-architecture-2026-11.md` §1.5).

## Candidats au lot 1
COH-01, COH-02, COH-03, COH-04, COH-05 (texte seul, l'état PyPI à fournir par l'opérateur), COH-07, COH-08, COH-09, COH-10, COH-11, COH-12, COH-13, COH-14, et les trois lignes CONNU-NON-CORRIGÉ. COH-06 touche la CI : effort S, risque faible, mais à valider par l'opérateur.
