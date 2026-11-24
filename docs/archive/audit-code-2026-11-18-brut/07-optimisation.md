# Optimisation : constats
Périmètre réellement lu : en entier `llvq-llm/src/tetra_cpu.rs`, `llvq-llm/src/fused_cpu.rs`, `llvq-llm/src/session.rs`, `llvq-quant/src/rotation.rs` (partie applicative), `llvq-tetra/llvq_tetra/{modules,vocab,fused,dequant}.py` ; en partie `llvq-llm/src/sealed.rs` (chargement dense), `llvq-llm/src/fused.rs` (`load_with`, `Transcoder::stream`, empaquetage), `llvq-llm/src/fused_cuda.rs` (`load_resolved`, `upload_int4`), `llvq-llm/src/persist4e_cuda.rs` (installation, `decode`, modules d'arbre), `llvq-llm/src/served.rs` (aiguillage des backends), `llvq-llm/src/served_decode.rs` (`continue_greedy`, `install_chain`), `llvq-llm/src/repl.rs` (`Sampler`), `llvq-llm/src/bin/mmlu.rs` (boucle de score), `llvq-llm/src/bin/gsm8k.rs` (boucle), `llvq-llm/src/model.rs` (`window_nll`, `logits_last`, magasin KV), `llvq-llm/src/calib.rs` (appel GPTQ), `llvq-quant/src/gptq.rs` (boucle GPTQ, découpage parallèle), `llvq-quant/src/linalg.rs` (`GptqFactor`), `llvq-quant/src/quantizer.rs` (`TetraShapeGain`), `llvq-artifact/src/tetra48.rs`, `llvq-artifact/src/format.rs` (`decode_matrix`), `llvq-cuda/src/gpu.rs` (compilation, cache de modules), `llvq-cuda/src/pers.rs` (`q4_tile_words`), `llvq-cuda/src/attn.rs`, `llvq-runtime/src/engine.rs` (préparation, boucle batch-1, moteur batch), `llvq-runtime/src/store.rs`, `llvq-runtime/library.json`, les `Cargo.toml`, `.github/workflows/ci.yml`. Lus pour le cadre : `docs/archive/revue-debit-2026-11-05.md`, `docs/ETAT.md`, `docs/ROADMAP.md` §1-2, `docs/runtime-audit-2026-11.md`, `docs/archive/audit-bugs-2026-11-08.md` §4, les journaux `batch-l40s-2026-11-08`, `chain-prefill-served-2026-11-15`, `encode-14b-2026-09-22`, `ablation-qualite-4b-2026-11-12`, `recherche-debit-sondes-2026-11-14`.
Non couvert : `llvq-search` hors `tetra/encoder.rs` survolé (recherche du plus proche voisin non profilée), `fused_metal.rs`, `batch.rs` et `batch_cuda.rs`, `bin/smoke.rs` hors fils, `bin/ppl.rs`, `llvq-runtime/src/{api,openai,chat}.rs` lus par grep seulement, `ops/` sauf les fils `LLVQ_THREADS`, le reste de `llvq-tetra` (`mps.py`, `metal.py` survolés).
Bilan : 0 critiques, 0 majeures, 7 mineures, 1 cosmétique (plus 2 lignes CONNU-NON-CORRIGÉ en fin de fichier, non comptées)

Aucun constat ne propose de réécrire un noyau GPU ni ne rouvre une piste de `docs/ETAT.md` §6. Tous les gains sont *estimés* ou *calculés*, jamais mesurés ici.

### OPT-01 Le module 4E, l'unité fusée et l'attention passent par NVRTC à chaque démarrage, alors que le cache de modules existe
- Gravité : mineure
- Confiance : CONFIRMÉ pour le code ; la part du temps est *calculée* sur deux journaux
- Où : `llvq-llm/src/persist4e_cuda.rs:696` (module 4E, `Cuda::on_stream`), `llvq-llm/src/fused_cuda.rs:802` (unité fusée), `llvq-cuda/src/attn.rs:77` (`attn_dec`), `llvq-cuda/src/gpu.rs:320-322` (`on_stream` = `compile_ptx` puis chargement du PTX) ; le cache : `llvq-cuda/src/gpu.rs:345-395` (`on_stream_cached`), branché seulement pour les modules d'arbre (`persist4e_cuda.rs:1187-1192`, `served_decode.rs:447-455`).
- Fait : chaque chargement servi recompile en NVRTC le module 4E de 252 757 octets, l'unité fusée de 106 947 octets et l'unité d'attention. Le cache de cubins par clé (texte, arch, NVRTC, pilote, sm) n'est utilisé que par la préfill en chaîne. Le cache du pilote ne couvre que l'étape PTX vers SASS, jamais NVRTC.
- Scénario : chaque `llvq serve` ou `llvq run` sur carte, chaque rechargement après éviction LRU, chaque processus de harnais sous `LLVQ_CONFIG`. Dans le même job (`batch-l40s-2026-11-08`), le 8B charge en 19,9 s sous la config batch (sans module 4E, avec trois unités batch) et en 58,2 s sous la config v9 servie : 38,3 s d'écart (*calculé*, `door-batch-8b.txt` et `door-b1-8b.txt`). Un module E1 de 4E compile en 56,7 à 57,3 s de NVRTC seul (*mesuré*, `recherche-debit-sondes-2026-11-14-brut/check.txt`). Le 4B servi charge en 55,1 s (`chain-prefill-served-2026-11-15-brut/gsm8k-v9.txt`).
- Touche : chemin servi, runtime
- Correction proposée : charger le module 4E, l'unité fusée et l'attention par `on_stream_cached` avec le répertoire de `LLVQ_MODULE_CACHE`, comme les modules d'arbre. Gain *estimé* : 35 à 45 s par démarrage sur 55 à 58 s, la lecture d'un cubin en cache prenant quelques secondes selon le message de `gpu.rs:369-373`. En dollars de job, peu : environ 0,02 $ par processus à 1,80 $/h.
- Effort : S
- Risque de la correction : faible. Le texte source et son sha256 ne bougent pas, donc aucun fichier épinglé. Le chargement passe de `cuModuleLoadData(PTX)` au cubin de `driver_cubin` : même ptxas, mais l'identité des tokens doit être relue sur carte.
- Vérification proposée : sur carte, la porte servie 4B deux fois de suite avec le cache (froid puis chaud) : temps de chargement, lignes d'installation 4E (registres, octets locaux, résidence) et les 256 tokens de la porte identiques à ceux d'aujourd'hui. Environ 0,1 $ (*estimé*).

### OPT-02 L'échantillonneur trie les 151 936 logits à chaque jeton, et la température par défaut est 0,7 hors 4E
- Gravité : mineure
- Confiance : CONFIRMÉ pour le code ; le coût est *estimé*
- Où : `llvq-llm/src/repl.rs:58-90` (tri complet ligne 65) ; appelé à chaque jeton par `llvq-llm/src/session.rs:277` ; défaut `llvq-runtime/src/engine.rs:385-391` (0,7 et top_p 0,9 quand le moteur n'est ni 4E ni batch), appliqué `engine.rs:550` et `:674`.
- Fait : à température positive, `pick` copie la ligne de logits vers l'hôte, calcule 151 936 exponentielles, trie tout le vocabulaire par `sort_unstable_by`, puis coupe le noyau top_p. Seuls les premiers jetons du tri servent.
- Scénario : le runtime sur Metal (moteur Rust, forme expanded), sur CPU (`fused_cpu`) et sous `LLVQ_PROFILE=eager` sans température dans la requête. Un tri de 151 936 paires coûte de l'ordre de n log2 n ≈ 2,6 M comparaisons : 2 à 5 ms par jeton avec les exponentielles (*estimé*). Contre 19,2 ms par jeton pour la porte Metal Rust à 52,1 tok/s (ETAT §4), cela ferait 10 à 25 % du jeton ; 2 à 4 % des 116 ms du CPU à 8,6 tok/s.
- Touche : runtime
- Correction proposée : sélection partielle : `select_nth_unstable_by` sur les K plus grands (K = 64), tri de ces K seulement, et doublement de K tant que leur masse n'atteint pas `top_p` du total. Le total reste la somme sur tout le vocabulaire.
- Effort : S
- Risque de la correction : faible. La loi tirée est la même. Pour une graine donnée, le jeton tiré peut changer dans de rares cas : le total est sommé dans un autre ordre, et l'ordre entre poids égaux n'est pas défini. Aucun chiffre publié n'échantillonne (tous gloutons) ; les jetons servis à température 0 ne bougent pas.
- Vérification proposée : un test local qui chronomètre `Sampler::pick` sur 151 936 logits aléatoires à 0,7 / 0,9 contre l'argmax, avant et après, et qui vérifie sur 10 000 tirages que les deux versions donnent le même jeton hors égalités de poids.

### OPT-03 Le harnais MMLU recalcule le préfixe 5-shot à chaque question
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/bin/mmlu.rs:765-810`
- Fait : pour chaque question, le préfixe du sujet (en-tête et cinq exemples `dev`) est recollé, retokenisé avec la question, puis tout le prompt repasse dans `model.hidden` ou `model.logits`. Le préfixe est identique pour toutes les questions d'un sujet. Le commentaire `:792-794` donne environ 600 lignes par prompt ; la question et ses options en font de l'ordre de 100 à 150 (*estimé*).
- Scénario : un recensement complet. Dans `ablation-qualite-4b-2026-11-12`, chaque bras dense du 4B score 14 042 questions en 1 224 s, et le job paie 10 524 s pour 8 bras, soit 5,26 $ (*mesuré*). Avec le préfixe en cache, le préremplissage par question passerait d'environ 600 à 150 jetons ; gain *estimé* ×2,5 à ×4 sur le score, soit 2,9 à 3,7 $ sur ce job. Les « 20 $ » et « 70 $ » de recensement d'ETAT §5 baisseraient dans la même proportion (*estimé*).
- Touche : banc
- Correction proposée : une `Session` par sujet : `reuse_prefix` sur les ids de chaque prompt complet (la frontière de tokenisation est donc gérée par le préfixe commun des ids, pas par une découpe du texte), puis `feed` de la seule suite et la tête sur la dernière ligne. Garder l'ancien chemin comme défaut tant que la nouvelle forme n'est pas validée.
- Effort : M
- Risque de la correction : moyen. Ce n'est plus le même instrument : la suite calculée contre un cache n'a pas les mêmes formes de produits que le prompt entier, et ETAT §5 documente déjà qu'un découpage du préremplissage différent change un texte à une quasi-égalité. Les barres MMLU publiées ont été mesurées par le chemin actuel.
- Vérification proposée : `mmlu` en Metal à `limit=5` par sujet, ancien et nouveau chemin, puis `mmlupair` entre les deux dumps : temps total, et le nombre de choix discordants (attendu proche de zéro, à comparer au 0,43 pp du bruit apparié).

### OPT-04 La propagation GPTQ lit `U` par un accès borné élément par élément, dans la boucle la plus interne
- Gravité : mineure
- Confiance : CONFIRMÉ pour la boucle ; PLAUSIBLE pour l'absence de vectorisation
- Où : `llvq-quant/src/gptq.rs:381-392`, `llvq-quant/src/linalg.rs:221-224`
- Fait : pour chaque bloc de 24 colonnes, chaque ligne et chaque `k`, la boucle fait `*cell -= x * factor.u(s + k, r0 + j)` sur toute la queue de ligne. `u()` indexe un `Vec<f64>` par `i * n + j` avec contrôle de borne à chaque élément, ce qui empêche en général LLVM de vectoriser. Le travail total est `d_out · d_in² / 2` produits par matrice : 1 393 GFMA par couche du 14B et 226 par couche du 4B (*calculé*).
- Scénario : l'encodage du 14B, 405 s par couche dont 81 % de quantification, sur 23 threads (*mesuré*, `encode-14b-2026-09-22`). À 1 à 2,5 GFMA/s par thread en scalaire (*estimé*), la propagation pèse 24 à 60 s par couche, 6 à 15 % de l'encodage ; le reste de la quantification est surtout la recherche Tetra, environ 330 µs par bloc (`quantizer.rs:660`), soit 197 s par couche sur 23 threads (*calculé*).
- Touche : encodeur
- Correction proposée : ajouter à `GptqFactor` un accesseur de ligne et prendre la tranche `U[s + k][r0..]` une fois par `k` et la zipper avec `row[r0..]`. Chaque cellule reçoit les mêmes opérations dans le même ordre de `k` (multiplication puis soustraction, sans contraction), donc le résultat est identique au bit. En second temps, traiter plusieurs lignes par tuile de colonnes pour réutiliser la tuile de `U` en L1 : même ordre par cellule, même bits. Gain *estimé* : 3 à 10 % du temps d'encodage, soit 0,5 à 1,5 $ sur un encodage de 14B à 15,35 $.
- Effort : S (la tranche), M (la tuile)
- Risque de la correction : nul si l'ordre par cellule est gardé. Ne touche ni format, ni index, ni tokens servis.
- Vérification proposée : `parallel_matches_serial_exactly` et un test qui compare au bit, avant et après, la sortie de `quantize_layer` sur une matrice aléatoire 2 560 × 9 728 avec un quantifieur trivial, chronométrée ; et chercher `panic_bounds_check` dans l'assembleur de la boucle.

### OPT-05 GPTQ parallèle découpe les lignes en autant de paquets fixes que de threads
- Gravité : mineure
- Confiance : PLAUSIBLE
- Où : `llvq-quant/src/gptq.rs:629-651`
- Fait : `rows_per = d_out.div_ceil(threads)` ; chaque thread reçoit un seul paquet contigu, créé une fois par matrice. La matrice se termine quand le paquet le plus lent se termine.
- Scénario : sur le Mac, la règle du poste impose `LLVQ_THREADS ≈ ncpu−4` et `nice 10` ; un M3 Max a 4 cœurs E, et un thread placé sur un cœur E, ou retardé par l'activité de l'utilisateur, fixe la durée de chaque matrice. Les 4B et 8B s'encodent sur le Mac (ROADMAP §2.5). Sur les jobs x86, un voisin bruyant a le même effet. Gain *estimé* : nul sur cœurs homogènes au repos, jusqu'à ×1,5 à ×2 sur la quantification si un thread sur cœur E est deux fois plus lent.
- Touche : encodeur
- Correction proposée : des paquets de 8 à 16 lignes distribués par un `AtomicUsize` partagé entre les threads du `scope`. Les lignes ne se parlent pas, donc la sortie reste identique au bit.
- Effort : S
- Risque de la correction : nul ; `parallel_matches_serial_exactly` le garde.
- Vérification proposée : journaliser le temps de chaque thread dans `quantize_layer_parallel_capturing` sur trois blocs du 0,6B sur le Mac à 12 threads, et comparer `phases.quantize` avant et après.

### OPT-06 La suite complète documentée compile chaque binaire de test avec LTO et une seule unité, alors que le profil `ci` existe
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `CLAUDE.md:74` (`cargo test --release -- --include-ignored`), `Cargo.toml:20-37`
- Fait : le profil `release` a `lto = "thin"` et `codegen-units = 1`. Le dépôt compte 124 fichiers de test d'intégration et 81 binaires. Le profil `ci` garde l'optimisation de `release` (les tests `cfg_attr(debug_assertions, ignore)` tournent) sans LTO, à 16 unités ; le commentaire de `Cargo.toml:24-29` dit que les résultats flottants n'en dépendent pas.
- Scénario : la suite « avant tout commit de format ». En CI, les tests `--release` prenaient 1 394 s, ce qui a motivé le profil `ci` (*mesuré*, message du commit e66f24d). La commande locale n'a pas suivi.
- Touche : outillage, doc
- Correction proposée : remplacer la commande par `cargo test --profile ci -- --include-ignored` dans `CLAUDE.md` et dans `docs/METHODE.md` s'il la cite.
- Effort : S
- Risque de la correction : faible. `ci` ajoute `overflow-checks`, ce qui rend la suite plus stricte, pas plus lâche.
- Vérification proposée : chronométrer les deux commandes une fois, en série, par la session principale ; compter les lignes `test result:` des deux sorties.

### OPT-07 Le binaire `llvq` compile parquet, zstd-sys, brotli et thrift sans s'en servir
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/Cargo.toml:25` (`parquet` avec `snap`, `zstd`, `brotli`, `flate2`), `llvq-runtime/Cargo.toml` (dépend de `llvq-llm` en entier) ; lecteurs de parquet : `llvq-llm/src/corpus.rs`, `llvq-llm/src/tetra_diag.rs`, atteints par `calib`, `guided`, `kvcapture`, `gsm8k`.
- Fait : `parquet` n'est utilisé que par les corpus des harnais et de l'encodeur. Le runtime n'en lit aucun. Il tire `zstd-sys`, qui compile une bibliothèque C, `brotli`, `thrift` et `chrono`.
- Scénario : `cargo build --release -p llvq-runtime` à froid, ou `cargo install` par un utilisateur. Gain *estimé* : 20 à 60 s de compilation à froid et la taille du binaire.
- Touche : outillage, runtime
- Correction proposée : une feature `corpus` de `llvq-llm`, par défaut active, qui porte `parquet` et les modules qui lisent un corpus ; `llvq-runtime` dépend de `llvq-llm` avec `default-features = false`.
- Effort : M
- Risque de la correction : faible ; ne touche aucun chemin de calcul.
- Vérification proposée : `cargo build --release -p llvq-runtime --timings` avant et après, et `cargo tree -p llvq-runtime -i parquet` vide après.

### OPT-08 Deux commentaires donnent au transcodage un coût de 142 et 145 s qu'il n'a plus sous `tetra48`
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/fused.rs:2370-2372`, `llvq-llm/src/fused_cuda.rs:3886-3888`
- Fait : les deux commentaires citent le transcodage Planes14 du 2026-08-05. Sous `tetra48`, le layout servi, `transcode_tetra48` (`llvq-artifact/src/tetra48.rs:149-174`) ne fait que recopier six octets par bloc.
- Scénario : le 8B charge en 19,9 s en tout sous la config batch (*mesuré*, `batch-l40s-2026-11-08-brut/door-batch-8b.txt`) ; le transcodage ne peut pas y peser 145 s. Le lecteur qui cherche où passent les 55 s du chargement est envoyé vers le mauvais poste (voir OPT-01).
- Touche : doc
- Correction proposée : dater les deux chiffres et préciser qu'ils valent pour Planes14 ; sous `tetra48` le transcodage est linéaire et court.
- Effort : S
- Risque de la correction : nul
- Vérification proposée : aucune, lecture suffisante.

## Constats connus, toujours présents à 925930d

- CONNU-NON-CORRIGÉ : le CPU servi tourne en forme expanded avec le magasin `Cat` (`Tensor::cat` de tout l'historique et `repeat_kv` à chaque couche), car `llvq-runtime/library.json` nomme `qwen3-*-tetra-e4.json`, sans champ `attention`. Réf. `docs/archive/audit-bugs-2026-11-08.md` §4, « PLAUSIBLE, CPU servi ».
- CONNU-NON-CORRIGÉ : le moteur batch-1 du runtime garde une seule conversation en cache (`llvq-runtime/src/engine.rs:407`) ; deux clients qui alternent relisent tout leur historique. Réf. `docs/runtime-audit-2026-11.md`, « What remains », point 1.

## Candidats au lot 1

- OPT-06 : une commande de doc.
- OPT-08 : deux commentaires.
- OPT-04, partie « tranche » seulement : identique au bit par construction, encodeur seul, aucun format ni fichier épinglé.
