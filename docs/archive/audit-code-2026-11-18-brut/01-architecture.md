# Architecture : constats
Périmètre réellement lu : en entier `docs/audit-architecture-2026-11.md`, les dix `Cargo.toml` (racine et neuf crates), `llvq-llm/src/lib.rs`, `llvq-llm/src/device.rs`, `llvq-runtime/library.json`, `.github/workflows/ci.yml` (jobs et cibles). Par sections : `llvq-llm/src/persist4e_cuda.rs` (1-160, 226-420, 1617-2085, 2300-2370), `llvq-cuda/src/pers.rs` (1666-1725, 2700-3260, 4700-4810), `llvq-llm/src/served.rs` (380-470, 516-665, 1055-1120), `llvq-llm/src/served_decode.rs` (plan des fonctions, 430-500), `llvq-llm/src/batch.rs` (1-60), `llvq-llm/src/batch_cuda.rs` (155-260, 375-440), `llvq-llm/src/kvq.rs` (20-170), `llvq-llm/src/persist_cuda.rs` (1-30, 100-135), les trois chargeurs (`fused_cuda.rs` 3826-4125, `fused_metal.rs` 2062-2360, `fused_cpu.rs` 300-485), `llvq-llm/src/model.rs` (800-870 et plan de `impl Qwen3`), `llvq-llm/src/bin/fusedrun.rs` (2060-2100), `llvq-llm/src/tetra_cpu.rs` (620-645), `llvq-runtime/src/{engine,library}.rs` (imports et `cfg`), `ops/Dockerfile.cuda` (listes de build), les gardes `#if` et symboles des huit fichiers de noyau de la chaîne de préremplissage, `ops/jobs/chain-prefill-served.sh` (28-41, 123-129). Survolés par `grep` : usages de `llvq_llm::` dans le runtime, `as_any`/`downcast_ref`, `env::var`, `cfg(...)`, `include_str!` inter-crates, copies de noyaux dans `llvq-tetra` (comparées par `cmp`).
Non couvert : l'intérieur des cinq crates du cœur et de `llvq-metal`, le corps de `fused_cuda.rs` et `fused.rs` (inchangés depuis 3e2d4f0), les corps des modes de mesure de `fusedrun.rs`, `rowcand`/`rowbench`, la porte MLX, l'intérieur des paquets Python `llvq-tetra` et `ops/llvqtune`, `ops/jobs/` au-delà d'un script.
Bilan : 0 critiques, 0 majeures, 9 mineures, 3 cosmétiques

## Suivi de l'audit d'architecture du 2026-11-07 (commit 3e2d4f0)

Une ligne par constat de l'ancien audit, état au commit `925930d`.

| ancien constat | état | preuve à 925930d |
|---|---|---|
| §Résumé 1, trois empreintes épinglées | toujours vrai, et étendu à deux nouveaux modules servis | `pers.rs:4704-4709` ; voir ARC-01 |
| §Résumé 2, le runtime ne sert pas v9 | corrigé (PR #15) | `llvq-runtime/library.json` : `cuda` = `qwen3-4b-tetra-e4-4e-v9-chain.json`, `-8b-...-v9.json`, `-14b-...-v9.json` |
| §Résumé 3, `LLVQ_PERS_SEGMENT` inatteignable | corrigé (f1d1872) | `fusedrun.rs:2062-2086`, une seule table `ALONE` |
| §Résumé 4, `LLVQ_PREFILL` et `LLVQ_METAL_KERNEL` non refusés | corrigé (def6cc8) | `served.rs:570-578`, `BELOW_THE_DOOR` |
| §Résumé 5 et §2.7, modes de mesure sans setter (`GRAPH_AB`, `GRAPH_DIAG`, `KV_AB`, `KV_PREALLOC`, `FUSE_AB`) | CONNU-NON-CORRIGÉ | toujours lus : `fusedrun.rs:2072-2076`, `served.rs:517` ; `fusedrun.rs` passe de 3 314 à 4 087 lignes, `main` fait 1 791 lignes (2097-3888) |
| §Résumé 6, CI sur 7 crates sur 9, aucun Python | partiel : job Python ajouté ; reste CONNU-NON-CORRIGÉ | `ci.yml:143-159` ; `--exclude llvq-cuda --exclude llvq-metal` toujours en `ci.yml:111` et `:191` ; les tests de `llvq-tetra` ne tournent pas en CI |
| §Résumé 7, `same_bits` entre formes d'appel dans `model.rs` | hors catégorie (TST), non requalifié | les sites existent toujours, `model.rs:4376-4377`, `:4514-4515`, `:4964-4969` |
| §Résumé 8 et §1.5, dérive de `CLAUDE.md` | corrigé pour l'essentiel | `docs/ENV.md` et son test (6bc7d75), `llvq-llm/src/lib.rs:7-11`, `AGENTS.md` réduit à un pointeur, `ci.yml:36,45` pointent `docs/archive/` |
| §1.5, `proofs/README.md` compte 22 préregs | CONNU-NON-CORRIGÉ | `proofs/README.md:20` |
| §1.5, `TIME_EVENTS`, `BENCH_ARMS` dans `MEASUREMENT_MODES` sans lecteur sous config | CONNU-NON-CORRIGÉ | `served.rs:517-518` ; seul `llvq-cuda/src/bin/planesbench.rs` les lit |
| §1.1, `llvq-cuda` dépend de `llvq-bench` en dépendance normale | CONNU-NON-CORRIGÉ, aggravé | `llvq-cuda/Cargo.toml`, `[dependencies] llvq-bench` ; voir ARC-05 |
| §1.1, surface publique de `llvq-llm` | CONNU-NON-CORRIGÉ, croissante | 1 133 déclarations `pub` contre 9 `pub(crate)` dans `llvq-llm/src/*.rs` (comptage `grep`, méthode plus large que l'ancien 762/8) |
| §1.1, noyaux partagés par `include_str!` entre crates | CONNU-NON-CORRIGÉ | `llvq-llm` vers `llvq-cuda` 12, `llvq-metal` vers `llvq-llm` 11, `llvq-cuda` vers `llvq-llm` 4 |
| §1.2, copies de noyaux dans `llvq-tetra` | sain, vérifié | les 10 copies sont identiques à leurs sources (`cmp`), gardées par `llvq-llm/tests/hfpack.rs` |
| §2.2, `rows` absent vaut v3 | CONNU-NON-CORRIGÉ | `served.rs:320-323` ; v5, v7, v8 ont quitté l'énumération (`served.rs:293-314`) |
| §2.3, hôte de la route P (`persist_cuda.rs`) | CONNU-NON-CORRIGÉ, et le chemin servi en dépend | `persist4e_cuda.rs:50` ; voir ARC-04 |
| §2.4, défaut `planes14` de `LLVQ_FUSED_LAYOUT` | CONNU-NON-CORRIGÉ | `fused.rs:97` |
| §2.4, palliatifs `planes14_host.rs`, `golay70_host.rs` | CONNU-NON-CORRIGÉ | les deux fichiers existent dans `llvq-cuda/src/` |
| §2.5, configs v4 et grouped 8B/14B gardées pour le runtime | CONNU-NON-CORRIGÉ, la raison est tombée | le runtime n'en charge plus aucune (`library.json`) ; les fichiers sont dans `configs/` |
| §2.6, 9 cibles mortes, pas de feature `archive` | CONNU-NON-CORRIGÉ, aggravé | les 9 fichiers existent (`llvq-bench/examples/f1linear.rs` … `llvq-metal/src/bin/hello.rs`) ; aucun `required-features` dans aucun `Cargo.toml` ; `llvq-llm` passe de 33 à 42 binaires |
| §2.7, trois parseurs de `LLVQ_THREADS` | CONNU-NON-CORRIGÉ, quatre aujourd'hui, dont un sur le chemin CPU servi | `tetra_cpu.rs:625-636` (vide refusé, défaut = cœurs), `smoke.rs:671-681` (vide = défaut, repli 4), `errmap.rs:314` (défaut 0), `f1recdump.rs:169` |
| §2.7, chargeur `LLVQ_KERNEL_DIR` recopié | CONNU-NON-CORRIGÉ | 24 lectures, dont `fused.rs:705`, `:761`, `:919`, `:1119` et 8 dans `fusedrun.rs` |
| §2.7, `NVRTC_ARCH` et `TILE_BLOCKS` paniquent | CONNU-NON-CORRIGÉ | `llvq-cuda/src/gpu.rs:38` (`assert!`), `llvq-cuda/src/tile.rs:405` (`panic!`) |
| §2.8, analyses ponctuelles dans `ops/` | CONNU-NON-CORRIGÉ, aggravé | 34 scripts Python ajoutés à `ops/` depuis 3e2d4f0, `ops/README.md` n'en indexe aucun |
| §3.1, fichiers géants | CONNU-NON-CORRIGÉ, aggravé | `model.rs` 4 303 → 5 402, `fusedrun.rs` 3 314 → 4 087, `pers.rs` 6 230, `persist4e_cuda.rs` 2 709 ; `fused_cuda.rs` (4 184) et `fused.rs` (3 082) inchangés |
| §3.1, interrupteurs globaux de mesure dans `fused_cuda.rs` | CONNU-NON-CORRIGÉ | `fused_cuda.rs:2918`, `:2930`, `:3148` ; un nouveau dans `model.rs:824` (`SOFTMAX_WHOLE`, documenté, sans effet sur les sorties) |
| §3.2, SHA-256 écrits à la main | CONNU-NON-CORRIGÉ | `llvq-cuda/src/sha256.rs`, `llvq-llm/src/digest.rs`, `llvq-metal/src/bin/rankbench.rs` |
| §3.2, helpers f16 en double | CONNU-NON-CORRIGÉ | `llvq-cuda/src/lib.rs:144`, `llvq-metal/src/lib.rs:351` |
| §3.2, modèle `tiny` recopié dans les tests | CONNU-NON-CORRIGÉ, aggravé | 9 fichiers de `llvq-llm/tests` définissent `fn tiny(` (7 avant) ; pas de `tests/common` |
| §3.4, `cfg` entremêlés | CONNU-NON-CORRIGÉ, aggravé | voir ARC-10 |
| §4 lot 6, régime de l'image CUDA | CONNU-NON-CORRIGÉ, aggravé | `ops/Dockerfile.cuda:55-57` : 30 `--bin` distincts (22 avant), dont six bancs ajoutés (`batchrun`, `specrun`, `vtreebench`, `specbench`, `dflashcheck`, `prefillbench`) |

## Constats

### ARC-01 Le module servi du préremplissage en chaîne compile sept bras de mesure pour un seul symbole lancé, et deux empreintes servies de plus gouvernent neuf fichiers de noyau
- Gravité : mineure
- Confiance : CONFIRMÉ pour la structure et les empreintes ; PLAUSIBLE pour la part du temps de compilation
- Où : `llvq-cuda/src/pers.rs:2552`, `:2672`, `:2747`, `:2883`, `:2958`, `:2976`, `:2996`, `:3015`, `:3076-3084`, `:3197-3199` ; `llvq-cuda/src/pers.rs:4704-4709`, `:4751` ; `llvq-cuda/kernels/pers_4et_stream.cu:47,478-480`, `pers_4et_a9.cu:32,462-464`, `pers_4et_a10.cu:39,444-446`, `pers_4et_a10rc.cu:19,438-441`, `pers_4et_a12.cu:326-337`, `pers_4et_a12g.cu:47,500-502`, `pers_4et_a12g32.cu:20,29-31` ; `llvq-llm/src/served.rs:408` ; `ops/jobs/chain-prefill-served.sh:32-34,129`
- Fait : le texte du bras servi `a12ipg32ft` (`served::CHAIN_ARM`) est construit par empilement : `#define P4T_A12IPG32`, puis le texte de `a12ipg`, de `a12ip`, de `a12`, de `a10r16`, de `a10`, de `a9`, du flux, du tuilé, puis le module 4E v9 entier et `pers_4et.cu`, chaque étage ajoutant son fichier `.cu`. Toutes les macros étant définies, chaque fichier émet ses `extern "C" __global__` : flux, a9, a10, a10r16, a12ip, a12ipg, a12ipg32, trois formes RoPE chacun, soit au moins 21 noyaux d'arbre, plus les noyaux du module 4E v9 qu'il embarque. Le chemin servi en lance un seul : `pers_4et_a12ipg32_f{form}_t384` pour la forme RoPE retenue. Le texte est épinglé par `19b734a3` (457 150 octets) et le module de base par `ce1c354d` ; le script de porte les vérifie (`chain-prefill-served.sh:33-34`). Les fichiers d'étage sont des copies modifiées de leur parent : environ 60 % de lignes de code communes entre étages successifs (a9/a10 : 166 sur 273 ; a10/a10rc : 173 sur 284 ; a12/a12g : 176 sur 329, comptage `difflib` hors commentaires et lignes courtes).
- Scénario : un premier démarrage du runtime sur CUDA pour le 4B compile ces modules en cubin : 509,2 s à froid contre 55,2 s depuis le cache (`library.rs:81-84`, mesure du journal `chain-prefill-served-2026-11-15`). Une partie de ce temps va à des noyaux jamais lancés (part non mesurée). Pour le nettoyage : la règle des « trois empreintes » de l'ancien audit est incomplète. Supprimer ou éditer un octet de `pers_4et.cu`, `pers_4et_stream.cu`, `pers_4et_a9.cu`, `pers_4et_a10.cu`, `pers_4et_a10rc.cu`, `pers_4et_a12.cu`, `pers_4et_a12g.cu`, `pers_4et_a12g32.cu`, `pers_attn_full.cuh`, ou l'ancre `PERS_FT_ANCHOR` (`pers.rs:3044`) dans `llvq_pers.cuh`, déplace l'empreinte du préremplissage servi. À l'inverse, `pers_4et_a4.cu`, `a5`, `a6`, `a8`, `a11`, `a11i`, `a10t`, `a10t2` (3 915 lignes) ne sont dans aucun texte servi.
- Touche : chemin servi
- Correction proposée : à court terme, inscrire les deux empreintes `ce1c354d` et `19b734a3` et la liste des neuf fichiers dans la règle des empreintes (`docs/format-noyau.md` ou l'en-tête de `pers.rs`), et classer les huit fichiers hors chaîne comme supprimables après étiquette. Au prochain déplacement d'empreinte accepté, écrire un fichier servi `pers_4et_chain.cu` aplati qui n'émet que les trois symboles `a12ipg32` sous un seul `#define`, et compiler le module de chaîne sans les étages.
- Effort : S pour la règle ; L pour l'aplatissement
- Risque de la correction : nul pour la règle. Fort pour l'aplatissement : il déplace `19b734a3` (et `ce1c354d` si le module de base change), donc une porte payante au 4B, et doit garder les jetons servis au bit.
- Vérification proposée : sur la carte, compiler le texte servi puis un texte qui ne garde que `P4T_A12IPG32` et ses dépendances, et comparer les temps NVRTC/ptxas et le SASS du symbole lancé. Sur le Mac, `nm` ou `cuobjdump -symbols` sur un cubin issu de `dump_tree_modules` (`pers.rs:4787`) compte les `__global__` émis.

### ARC-02 Le chargement servi de la chaîne compile le module d'arbre de base, qui ne sert qu'au bras de mesure `Rows`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/persist4e_cuda.rs:1624-1650`, `:1849-1852`, `:2365` (`(TreeForm::Rows, _) => (&v.f, &self.args)`) ; `llvq-llm/src/served_decode.rs:456` ; `llvq-runtime/src/library.rs:81-93` ; `ops/jobs/chain-prefill-served.sh:129`
- Fait : `Runtime4e::install_tree` construit sans condition le module `p4t_unit_text` (« 4E tree », 306 060 octets, `ce1c354d`), en tire le symbole `pers_4et_rows_*` et le garde dans `Tree4e::f`. Ce symbole n'est lancé que pour un bras de forme `TreeForm::Rows` (`:2365`). Le chemin servi installe un seul bras, `TreeForm::A12ipg32Ft` (`served_decode.rs:456`) ; `p4t_tile_words` et `p4t_ip_check` viennent des modules tuilés (`:1973`, `:1979-1982`), pas du module de base.
- Scénario : chaque `llvq run` d'un 4B en CUDA compile (à froid) ou charge (depuis le cache) ce module en plus du module servi. Le message du runtime le dit : « compiles its two tree modules » (`library.rs:91`).
- Touche : chemin servi, runtime
- Correction proposée : construire le module de base paresseusement, seulement si un bras `Rows` est demandé, et garder `f: Option<CudaFunction>` dans `Tree4e`. Mettre à jour la ligne de contrôle de `chain-prefill-served.sh:129` et le message de `library.rs:91`.
- Effort : S
- Risque de la correction : faible. Aucun texte de noyau ne change, aucune empreinte ne bouge, les jetons servis ne bougent pas (le symbole lancé et ses tampons sont les mêmes). Le journal de chargement perd la ligne « 4E tree: module 306060 bytes » que le script de porte attend : la porte servie doit être relancée ou son script corrigé.
- Vérification proposée : sur la carte, chargement servi avec le cache vide avant et après, temps de chargement et 256 jetons identiques ; sur le Mac, `cargo check` sous `CUDARC_CUDA_VERSION` pour le type `Option`.

### ARC-03 Le choix des modules d'arbre repose sur trois listes positionnelles de 28 cases tenues à la main, sur le chemin servi
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/persist4e_cuda.rs:1777-1807` (tableau `want`), `:1812-1842` (`t.fx[0..23]`), `:1863`, `:1870`, `:1877-1930` (match sur `k`), `:1933-1940`, `:1962`, `:1978`, `:2067-2074`
- Fait : `install_tree` transforme les bras demandés en `want: [bool; 28]` dans l'ordre des variantes ; `install_tiled` itère sur l'indice `k`, choisit le texte et le symbole par un `match k` à plages (`25..=27`, `18..=20`, `7..=9`, `_` pour 4 à 6), les fils par une autre plage (`16 | 20..=27`), l'aire groupée par `23 | 24 | 26 | 27`, le module de contrôle par `22..=27`, la conservation des copies par `want[..22]`, puis déstructure 28 options en `f, fs, f4, f5, fx[24]` ; un troisième `match` relit `fx[0]` à `fx[23]`. Le bras servi passe par ces trois tables (case 27, `fx[23]`).
- Scénario : ajouter un bras ailleurs qu'en fin de liste, ou décaler une variante de `TreeForm`, décale silencieusement les indices : le bras servi peut recevoir le symbole, l'aire de mémoire partagée ou le module de contrôle d'un autre bras. Rien ne le refuse à la compilation ; seules les empreintes vérifiées par le script de porte et un lancement sur carte le révèlent.
- Touche : chemin servi
- Correction proposée : une table `const TREE_ARMS: [TreeArmDef; 28]` indexée par `TreeForm as usize`, avec pour chaque bras son étiquette, sa fonction de texte, sa fonction de symbole, ses fils, sa forme d'aire, s'il lit en place ; `install_tiled` itère sur la table et rend une `[Option<CudaFunction>; 28]` indexée par la même variante. Fichier : `llvq-llm/src/persist4e_cuda.rs`, ou mieux un nouveau `persist4e_tree.rs` qui prend aussi `install_tree`, `install_tiled` et `tree_launch`.
- Effort : M
- Risque de la correction : faible. Aucun texte de noyau ne bouge, donc aucune empreinte ; les lignes de journal « 4E tree … » doivent garder leur texte, que les scripts de porte lisent.
- Vérification proposée : un test hôte qui, pour chaque `TreeForm`, vérifie que la table donne le même texte (sha256) et le même symbole que le code actuel ; mutant : permuter deux lignes de la table, le test doit tomber.

### ARC-04 Les chemins servis 4E et batch contournent les cinq traits de `device.rs` par `as_any`, et leurs accesseurs vivent dans un fichier marqué « mesure seulement »
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/device.rs:105-107`, `:150-152` ; `llvq-llm/src/persist_cuda.rs:8-9`, `:114-131` ; `llvq-llm/src/persist4e_cuda.rs:50`, `:803`, `:811` ; `llvq-llm/src/batch_cuda.rs:383`, `:419`, `:435`, `:493`, `:518`, `:531`, `:689` ; `llvq-llm/src/dflash_q4.rs:56`
- Fait : `device.rs` affirme que `as_any` sert à la route P et que « nothing on a served path calls it ». Or le 4E servi importe `tetra_of`, `int4_of`, `tensor_addr`, `mlp_args`, `seg_tetra` de `persist_cuda.rs` (`persist4e_cuda.rs:50`), fichier dont l'en-tête dit « Measurement only », et `tetra_of`/`int4_of` font `as_any()?.downcast_ref::<CudaLattice | CudaInt4>()`. Le batch servi (`llvq serve` sous `LLVQ_BATCH`) refait sept descentes vers `CudaLattice`, `CudaInt4`, `CudaEmbed` ; `persist4e_cuda.rs:803,811` et `dflash_q4.rs:56` en refont trois vers `CudaEmbed`. Onze sites au total, en trois fichiers, chacun avec son message d'erreur.
- Scénario : supprimer la route P (lot 4 de l'ancien audit) casse le décodage servi à la compilation ; un nouveau backend qui implémente les traits ne peut pas servir 4E ni le batch, et l'apprend au chargement par un des onze messages. La doc du seam ment au lecteur qui veut savoir ce qui est servi.
- Touche : chemin servi
- Correction proposée : créer `llvq-llm/src/cuda_views.rs` (sous le même `cfg` que `fused_cuda`) avec `tensor_addr`, `tetra_of`, `int4_of`, `embed_q4_of`, `seg_tetra`, `mlp_args`, et remplacer les onze descentes par ces fonctions ; `persist_cuda.rs` et `batch_cuda.rs` les importent de là. Corriger `device.rs:105-107` et `:150-152` pour nommer 4E et le batch comme utilisateurs servis.
- Effort : S
- Risque de la correction : nul. Déplacements de fonctions, aucun texte de noyau, aucune empreinte, aucun bit servi.
- Vérification proposée : `ops/check-linux-cuda.sh` (type-check Linux sur le Mac) ; `git grep -n 'downcast_ref::<Cuda' llvq-llm/src` ne doit plus montrer que `cuda_views.rs`.

### ARC-05 Sur macOS, `llvq-llm` dépend normalement de `llvq-cuda` pour un tireur de recherche, ce qui fait compiler `llvq-cuda` et `llvq-bench` au binaire `llvq`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/Cargo.toml`, table `[target.'cfg(target_os = "macos")'.dependencies]`, ligne `llvq-cuda = { path = "../llvq-cuda" }` (ajoutée par 9e00978, 2026-11-10) ; `llvq-cuda/Cargo.toml`, `[dependencies] llvq-bench` ; `llvq-llm/src/lib.rs:32-34` ; `llvq-llm/src/dflash_spec.rs:21`
- Fait : la seule utilisation de `llvq-cuda` par la bibliothèque `llvq-llm` sur Mac est `dflash_spec`, le tireur DFlash de la boucle spéculative (route fermée), qui importe `llvq_cuda::spec_loop`. `dflash_spec` n'est utilisé que par `bin/dflashcheck` et `bin/specbench`. Comme `llvq-cuda` dépend sans condition de `llvq-bench`, toute compilation Mac de `llvq-llm`, donc de `llvq-runtime`, compile la bibliothèque `llvq-cuda` (40 060 lignes dans le crate) et `llvq-bench`. CI ne le voit pas : ses runners sont tous `ubuntu-latest` (`ci.yml:91,144,171,199,227`).
- Scénario : `cargo build -p llvq-runtime --features metal` sur le Mac de développement compile deux crates de banc que le binaire servi n'appelle jamais ; le graphe annoncé par l'ancien audit (« `llvq-runtime` sits on `llvq-llm` ») est faux sur Mac.
- Touche : runtime, outillage
- Correction proposée : rendre la dépendance macOS optionnelle sous une feature `spec = ["dep:llvq-cuda"]`, mettre `dflash_spec` sous `cfg(any(feature = "cuda", feature = "spec"))`, et `required-features = ["spec"]` sur `dflashcheck` et `specbench` côté Mac. Séparément (lot 7 de l'ancien audit), passer `llvq-bench` de `llvq-cuda` derrière une feature des binaires `f1rankfloor`, `f1floorbench`, `planesbench`.
- Effort : S
- Risque de la correction : faible ; aucun format, aucune empreinte, aucun bit servi. Les commandes Mac de `dflashcheck` et `specbench` devront ajouter `--features spec`.
- Vérification proposée : `cargo tree -p llvq-runtime -e normal --target aarch64-apple-darwin | grep 'llvq-'` avant (montre `llvq-cuda` et `llvq-bench`) et après (ne les montre plus).

### ARC-06 La bibliothèque `llvq-llm` embarque dans tout binaire, runtime compris, les modules de recherche et les dépendances de jeux de données
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/Cargo.toml` (`parquet`, `hf-hub`, `flate2` sans condition) ; `llvq-llm/src/lib.rs:13-72` ; utilisateurs de `parquet` : `corpus.rs`, `tetra_diag.rs` ; de `hf_hub` : `corpus.rs`, `loader.rs` ; `llvq-runtime/Cargo.toml` (`llvq-llm = { path = "../llvq-llm" }`, features par défaut)
- Fait : tous les modules sont publics et compilés sans condition, dont ceux ajoutés depuis 3e2d4f0 pour la recherche : `dflash` (1 117 lignes, plus `dflash_profile.rs` 445 par `#[path]`), `speculative` (533), `guided` (614), `kvladder` (592), `kvcapture` (545), `spec_prompts` (126). Le runtime n'appelle que `served`, `served_decode`, `session`, `chatfmt`, `repl`, `batch`, `batch_cuda`, `fused_cpu`, `model` (comptage des `llvq_llm::` dans `llvq-runtime/src`), et fait son propre téléchargement par `ureq`. Il lie pourtant `parquet` (avec `snap`, `zstd`, `brotli`) et `hf-hub`. Les seuls liens du chemin servi vers ces modules sont trois constantes de noms de variables (`served.rs:575-577` vers `guided`) et le type `kvladder::LadderAux` dans `model.rs:397,1933`.
- Scénario : chaque build du runtime compile et lie le lecteur parquet et quatre mille lignes de code de recherche ; toute modification d'un module de recherche recompile le runtime.
- Touche : runtime, outillage
- Correction proposée : une feature `lab` de `llvq-llm`, active par défaut, qui porte `parquet`, `hf-hub` et les modules `corpus`, `loader`, `sealed`, `calib`, `gsm8k`, `eval`, `tetra_diag`, `errmodel`, `tempfit`, `kvcapture`, `kvladder`, `guided`, `dflash*`, `speculative`, `spec_prompts` ; déplacer les trois constantes `ENV_*` de `guided` dans `served.rs` ; `llvq-runtime` dépend de `llvq-llm` avec `default-features = false`. Les binaires gardent la feature par défaut, aucun chemin de fichier ne bouge.
- Effort : M
- Risque de la correction : faible ; pur découpage de compilation, aucun format, aucune empreinte. Le point délicat est `model.rs` (KV ladder, capture) : soit un `cfg` sur deux méthodes, soit ARC-09 d'abord.
- Vérification proposée : `cargo tree -p llvq-runtime -e normal | grep -c parquet` vaut 0 après ; `cargo build -p llvq-runtime` et `cargo test -p llvq-runtime` passent.

### ARC-07 Le type servi `served_decode::Decoder` porte les portes de mesure et la boucle spéculative, et expose son `Runtime4e` en mutable
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/served_decode.rs:187` (`runtime_mut`), `:191` (`install_verify`), `:210` (`continue_speculative`), `:270` (`continue_lookup`), `:376` (`decoder_measurement`), `:381` (`serve_measurement`) ; utilisateurs : `bin/vtreebench.rs:403`, `bin/prefillbench.rs:328,335,394,492,505,561,694,734`, `bin/specrun.rs:71,250`
- Fait : `served_decode.rs` passe de 291 à 663 lignes. Le runtime n'appelle sur `Decoder` que `is_persistent`, `launch_tokens`, `max_positions` et les chemins de génération ; les six méthodes ci-dessus ne servent qu'aux bancs. `runtime_mut` rend `&mut Runtime4e`, donc toute la surface de mesure (`install_tree` avec n'importe quel bras, `set_kernel_from_for_measurement`, `set_k8_release`) à quiconque tient un décodeur servi.
- Scénario : un appelant du runtime peut, sans passer par une config ni par `check_env`, installer un bras d'arbre de plancher (« values void », `persist4e_cuda.rs:246-254`) ou abaisser le seuil du noyau sur le décodeur qui sert. Rien ne le fait aujourd'hui ; rien ne l'empêche.
- Touche : chemin servi, runtime
- Correction proposée : garder dans `Decoder` les seuls chemins servis ; déplacer les six méthodes dans un trait d'extension `DecoderLab` d'un module `served_decode_lab.rs` (sous la feature `lab` d'ARC-06), que seuls les bancs importent.
- Effort : S
- Risque de la correction : nul ; déplacement de méthodes, aucun bit servi, aucune empreinte.
- Vérification proposée : `cargo build -p llvq-runtime` sans le trait en portée ; `ops/check-linux-cuda.sh` pour les bancs.

### ARC-08 Les trois chargeurs de backend recopient l'assemblage du modèle : config, tokenizer, table des sites, contrôle d'attache, total
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/fused_cuda.rs:3826-4184` (`:3937-3939`, `:3984`, `:4097-4120`, `:4082`) ; `llvq-llm/src/fused_metal.rs:2062-2360` (`:2123-2125`, `:2151`, `:2310`, `:2318-2330`, `:2333-2355`) ; `llvq-llm/src/fused_cpu.rs:304-485` (`:341-343`, `:353`, `:417-423`, `:442-465`) ; `llvq-llm/src/served.rs:1055-1095`
- Fait : chaque backend réécrit la lecture de `config.json` et du tokenizer, la table `by_site: HashMap<(usize, String), Proj>` et sa fermeture `take` avec le compte `claimed != total_sites`, le contrôle `Arc::ptr_eq(embed, head) != tie_word_embeddings`, et la ligne « total expected on the … ». Les signatures prennent 9 ou 10 arguments positionnels (`#[allow(clippy::too_many_arguments)]`), et Metal n'a pas `int4`. `device.rs:186-191` rappelle que chaque chargeur « owes the same check », règle que seule la recopie fait respecter.
- Scénario : un quatrième backend, ou une correction du contrôle d'attache, se fait en trois endroits ; un oubli rend un modèle qui tourne et ment (le commentaire `fused_cuda.rs:4107-4110` le dit pour CUDA).
- Touche : chemin servi
- Correction proposée : dans `fused.rs`, une fonction commune `assemble(model, by_site, embed_pair, knobs) -> FusedSealed` qui fait config, tokenizer, contrôle d'attache, comptage des sites et ligne de total ; chaque backend ne fournit que ses téléversements. Regrouper les réglages en `struct ServedKnobs { dtype, layout, emode, share, fuse, kv, int4, origin }`.
- Effort : M
- Risque de la correction : faible ; aucun format, aucune empreinte. Le texte « total expected on the card » est lu par les portes servies et doit rester au caractère près ; les poids chargés doivent rester les mêmes (bit des jetons inchangé).
- Vérification proposée : les tests de `served_config.rs` et de `cpu_tetra.rs` sur le Mac ; `oracle` sur Metal ; mutant : inverser l'attache dans la fonction commune, les trois backends doivent refuser.

### ARC-09 Les énumérations servies portent les bras de mesure
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/kvq.rs:43-62` (`K8V4`, `K4V4C`, `K4V4H`) ; `llvq-llm/src/batch.rs:43-53` (`GemmOcc`) ; `llvq-llm/src/persist4e_cuda.rs:226-315` (28 `TreeForm`, dont trois planchers « values void ») ; `llvq-llm/src/persist4e_cuda.rs:61-73,115-116` (champs `f_stamp`, `f_logits`, `e1_arm`, `local_bytes`, `k8` de `Runtime4e`)
- Fait : `KvMode` est le type que reçoivent les trois chargeurs servis (`fused_cpu.rs:312`, `fused_metal.rs:2070`) ; trois de ses cinq variantes ne viennent jamais d'une config (`KvMode::parse`, `kvq.rs:73-82`) mais doivent être traitées par chaque `match` (`kvq.rs:116-132`, `model.rs:407`). Même schéma pour `BatchProjs::GemmOcc` et pour `Runtime4e`.
- Scénario : chaque nouvel usage servi de ces types doit décider quoi faire d'un bras de mesure qu'il ne verra jamais ; ARC-06 ne peut pas sortir `kvladder` du build servi tant que `KvMode` le nomme.
- Touche : chemin servi
- Correction proposée : séparer `KvMode { F16, Q8 }` (servi) et `KvArm { Served(KvMode), Ladder(LadderArm) }` (mesure) dans `kvladder.rs` ; même découpage pour `BatchProjs`. Pour `TreeForm`, voir ARC-03.
- Effort : M
- Risque de la correction : faible ; aucun format, aucune empreinte, aucun bit servi.
- Vérification proposée : aucune, lecture suffisante ; ensuite `cargo test -p llvq-llm`.

### ARC-10 Le vocabulaire `cfg` du backend CUDA est recopié 146 fois et varie d'un module à l'autre
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src` et `llvq-runtime/src` : 97 `cfg(all(target_os = "linux", feature = "cuda"))`, 11 négations, 38 `cfg_attr(not(all(target_os = "linux", feature = "cuda")), allow(dead_code))` tous dans `llvq-llm/src/bin/fusedrun.rs` (à partir de `:51`) ; `llvq-llm/src/lib.rs:18` (`batch_cuda` sous `cfg(feature = "cuda")` seul, quand `fused_cuda`, `persist4e_cuda`, `attn_cuda` sont sous `all(target_os = "linux", feature = "cuda")`) ; `lib.rs:33` (`any(feature = "cuda", target_os = "macos")`)
- Fait : aucune `build.rs` dans `llvq-llm` ne nomme le prédicat une fois ; `fusedrun.rs`, binaire CUDA par nature, compile sur Mac avec 38 éléments morts déclarés tels, parce que ses tests portables (`alone_in`, `parse_gqa_ppl`, `paired_ppl_pct`, `fusedrun.rs:3892`) vivent dans le binaire.
- Scénario : une erreur de prédicat (comme `batch_cuda`) compile sur une cible et pas sur l'autre ; seul `ops/check-linux-cuda.sh` le voit, CI ne vérifie pas le chemin CUDA.
- Touche : outillage
- Correction proposée : `llvq-llm/build.rs` sans dépendance qui émet `cargo:rustc-cfg=cuda_backend` (Linux et `CARGO_FEATURE_CUDA`) et `metal_backend`, plus `cargo::rustc-check-cfg` ; remplacer les prédicats. Sortir les parseurs portables de `fusedrun.rs` dans un module de bibliothèque, puis `required-features = ["cuda"]` sur `fusedrun` et retrait des 38 `allow(dead_code)`.
- Effort : M
- Risque de la correction : nul ; aucun format, aucune empreinte, aucun bit servi.
- Vérification proposée : `cargo clippy --all-targets` sur le Mac et `ops/check-linux-cuda.sh` : zéro avertissement des deux côtés.

### ARC-11 `llvq-cuda/src/pers.rs` (6 230 lignes) mêle les textes des modules servis, ceux de 30 bras de mesure, des décodeurs de tampons horaires et des modèles de volume
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-cuda/src/pers.rs:1666-1725` (unités servies), `:1897-2092` (4E), `:2093-2219` (vérification), `:2220-3030` (arbre : 30 fonctions de texte et de symbole), `:2686-2742`, `:2802-2876` (décodeurs de tampons), `:3122-3162` (`p4t_prefetch_volume`, un modèle chiffré), `:4004-6230` (tests et empreintes)
- Fait : le fichier qui définit les trois textes épinglés servis (`SERVED_UNIT_PARTS`, `PERS_UNIT_PARTS`, et par ARC-01 la lignée de la chaîne) porte aussi les bras E1, rc, k8, l'arbre a0 à a12ipg32ft, et l'analyse des tampons. 288 éléments publics de premier niveau.
- Scénario : relire ce qui entre dans une empreinte servie demande de trier 4 000 lignes ; un ajout de bras se fait dans le même fichier que les textes servis, au risque d'éditer un `const` de partie partagé.
- Touche : chemin servi, banc
- Correction proposée : découper en `llvq-cuda/src/pers/{served.rs, e4.rs, tree.rs, stamps.rs, models.rs}` ; `served.rs` reçoit `SERVED_UNIT_PARTS`, `PERS_UNIT_PARTS`, `pers_unit_text`, `p4e_rows_define`, la lignée `p4t_unit_text` à `p4t_ft_unit_text` et les tests d'empreintes ; le reste va aux autres fichiers. Pur déplacement.
- Effort : M
- Risque de la correction : faible. Les empreintes hachent le texte généré, pas le `.rs` : elles ne bougent pas si les chemins `include_str!` sont corrigés (`../kernels` devient `../../kernels`). `pers.rs:5264` inclut `ops/jobs/pers-l2.sh` par chemin relatif, à corriger aussi.
- Vérification proposée : `cargo test -p llvq-cuda` sur le Mac : les tests `the_served_unit_is_the_published_one`, `the_4e_v9_module_is_the_one_the_doors_gate_on`, `the_d1_tree_modules_keep_their_bytes`, `the_ft_modules_keep_their_bytes` doivent passer sans changer une constante.

### ARC-12 Trois commentaires de frontière disent le contraire du code
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/persist4e_cuda.rs:18-24` ; `llvq-llm/src/persist_cuda.rs:8-9` ; `llvq-llm/src/device.rs:105-107`, `:150-152`
- Fait : `persist4e_cuda.rs` dit que la config 4B servie est `qwen3-4b-tetra-e4-4e-v4.json` ; le servi est v9 et la chaîne (`library.json`, `docs/ETAT.md` §2). `persist_cuda.rs` dit « Measurement only », alors que le décodage servi importe ses fonctions (`persist4e_cuda.rs:50`). `device.rs` dit que rien de servi n'appelle `as_any` (voir ARC-04).
- Scénario : un lecteur qui cherche la frontière servie dans ces en-têtes la trouve au mauvais endroit, et supprimerait `persist_cuda.rs` en le croyant de mesure.
- Touche : doc
- Correction proposée : réécrire les trois passages ; si ARC-04 est fait, seul celui de `persist4e_cuda.rs` reste à corriger.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

## Candidats au lot 1

ARC-04, ARC-05, ARC-07, ARC-12. ARC-02 est d'effort S et ne touche ni format, ni empreinte, ni bit servi, mais il change une ligne que le script de porte `chain-prefill-served.sh:129` vérifie : à prendre seulement avec la mise à jour de ce script.
