# Vérification contradictoire : moteur, runtime, tests, cohérence

Périmètre réellement lu : `llvq-runtime/src/engine.rs` (1-320, 480-790, 1240-1540), `llvq-runtime/src/api.rs` (165-190),
`llvq-runtime/src/library.rs` (78-130), `llvq-llm/src/persist4e_cuda.rs` (195-222, 680-705, 925-980, 1118-1160,
1545-1580, 1600-1860, 2161-2470), `llvq-llm/src/session.rs` (95-290), `llvq-llm/src/served_decode.rs` (30-180,
212-226, 340-365, 445-500), `llvq-llm/src/served.rs` (505-580, 640-730, 915-965, 1055-1100), `llvq-llm/src/model.rs`
(526-614, 815-870), `llvq-llm/src/sealed.rs` (140-180, 685-705), `llvq-llm/src/fused_cuda.rs` (700-815),
`llvq-cuda/src/gpu.rs` (20-60), `llvq-cuda/src/spec_loop.rs` (318-350), `llvq-llm/src/bin/fusedrun.rs` (2060-2145),
`llvq-llm/tests/{served_unit,served_unit_copy,served_config,session}.rs` (parties), `llvq-cuda/tests/golay70_decoder_matches_rust.rs`
(390-420), `llvq-artifact/tests/common/mod.rs` (35-55), les deux `Cargo.toml` de `llvq-llm` et `llvq-cuda`,
`configs/README.md`, `CLAUDE.md:123`, `docs/ETAT.md:128`, `ops/jobs/{chain-prefill-served,chain-runtime-door,served-v9}.sh` (parties).

| ID | Verdict | Gravité retenue | Raison en une ligne |
|---|---|---|---|
| QUA-50 | CONFIRMÉ | majeure | Panique de `load` hors `catch_unwind` : fil mort, état `Loading`, `queue` à 1 pour toujours, 503 pour tout autre modèle sous `LLVQ_MAX_LOADED=1`. |
| QUA-53 | NUANCÉ | mineure | Les copies existent, mais sous une config persistante `Served::read` refuse déjà les 19 noms ; `ALONE` est une autre liste, testée. |
| QUA-57 | NUANCÉ | mineure | Doublon d'ARC-02. Le symbole de base ne sert qu'à `Rows`, mais le module de base alloue aussi tous les tampons de l'arbre. |
| QUA-58 | NUANCÉ | cosmétique | Aucun test ne l'exerce, vrai ; « aucun test ne peut », faux : un test d'intégration lit l'atomique réel. |
| BUG-50 | CONFIRMÉ | majeure | Erreur d'un lancement après le premier : caches à `pos0`, `ids` inchangé, chaque requête qui garde tout le préfixe répond 500. Le déclencheur déterministe cité est faux. |
| BUG-51 | CONFIRMÉ | mineure | `reserve` avant le refus de capacité ; inatteignable par `llvq serve` et le REPL. |
| SEC-02 | CONFIRMÉ | mineure | `Runtime4e` garde des adresses brutes des tenseurs du modèle sans les tenir ; `Decoder` n'a pas de durée de vie liée. Aucun appelant ne le déclenche. |
| TST-01 | CONFIRMÉ | majeure | La branche 4E de `Session::answer_fed` n'est exercée par aucun test, ni Mac ni carte ; le trou est plus large que dit. |
| TST-02 | CONFIRMÉ | mineure | Aucun test n'appelle `Served::read` ni `DeviceLoader::load`. Le sous-processus `fusedrun` proposé ne marche pas sur Mac. |
| TST-04 | CONFIRMÉ | mineure | `SKIP` puis `return` : vert sans l'archive, et `LLVQ_SEALED_ARTIFACT` ignoré. |
| TST-05 | CONFIRMÉ | mineure | Le test ne touche jamais `from_env` ; le bras `(false, false)` n'est couvert par rien. |
| TST-06 | NUANCÉ | mineure | La liste du test omet `emb_q4.cu` et les defines ; `served_unit_copy.rs` couvre l'ordre (par une autre copie) mais ne parse rien. |
| ARC-02 | NUANCÉ | mineure | Fait juste ; effort S et risque faible sous-estimés : allocations, aire et découpes viennent du module de base, et trois scripts de porte lisent sa ligne. |
| ARC-05 | NUANCÉ | mineure | Fait juste ; la correction oublie deux tests Mac (`dflash_fixture.rs`, `dflash_prefill.rs`) qui sortiraient de la boucle par défaut. |
| COH-01 | NUANCÉ | mineure | `CLAUDE.md:123` et `configs/README.md:55` sont faux sur CUDA ; `configs/README.md:168` parle de Metal et reste juste. |
| COH-04 | NUANCÉ | cosmétique | `configs/README.md:43` et `:132` sont faux ; l'oubli du drafter dans `CLAUDE.md` et `ETAT.md` est défendable. |

## QUA-50 : CONFIRMÉ, majeure

Chaîne d'appel lue :
- `engine.rs:1493` : `Models::submit` appelle `Engine::spawn` sous le verrou, puis `:1502` soumet le job (`queue` passe à 1, `:1350`).
- `engine.rs:1267-1269` : le fil `llvq-engine-<name>` appelle `loader.load(...)` sans `catch_unwind`. Le `JoinHandle` rendu par `spawn(...)?` (`:1345`) est jeté : personne ne surveille le fil.
- Seuls `run_batch` (`:1300`) et `run` (`:1323`) sont protégés.
- Sur panique, le fil se déroule, la fermeture est détruite, donc `rx` aussi. Le canal `std::sync::mpsc` jette ses messages en attente : le `Job` meurt, son `Sender` d'événements aussi. Personne ne fait `queue.fetch_sub` : `queue` reste à 1. L'état reste `State::Loading` (`:1259`).
- Pas de `panic = "abort"` dans aucun `Cargo.toml`, pas de `set_hook` dans `llvq-runtime/src` : la panique ne tue que ce fil.

Ce qu'on observe :
- Première requête : `api.rs:175`, 500 « the engine stopped before answering ».
- Requêtes suivantes sur ce modèle : `:1456` n'oublie que `Failed`, le moteur `Loading` est retrouvé (`:1457`), `tx.send` échoue, 500 « the engine of X has stopped » (`:1353`).
- `idle()` (`:298-300`) est faux pour toujours (`queue` = 1). `reap` (`:1518`) ne le retire jamais. `unload` passe par `retire` (`:1377-1380`) et pose seulement `retiring`. `expires` est `None` (`:1501`).
- Sous `LLVQ_MAX_LOADED=1` (défaut, `:1424`), tout autre modèle reçoit 503 (`:1481-1489`) : aucun victime `idle`.
- `/api/ps` (`running`, `:1522-1534`) montre `Loading` sans fin.

Le déclencheur cité est réel :
- `llvq-cuda/src/gpu.rs:36-43` : `assert!` sur la forme `compute_NN`.
- Aucune mention de `LLVQ_NVRTC_ARCH` ni de `gpu::arch()` dans `llvq-runtime` (`git grep`). `check_env` ne le refuse pas, par choix (`served.rs:47`, test `served_config.rs:663`).
- `DeviceLoader::load` (`engine.rs:127`) appelle `served::load_resolved`, qui sur CUDA va à `fused_cuda::load_resolved` (`served.rs:1074`), qui imprime `llvq_cuda::gpu::arch()` (`fused_cuda.rs:800`) avant `Cuda::on_stream`. La panique part donc bien du fil du moteur.
- Cas limite : si le fil meurt avant le `send` de `:1502`, `queue` revient à 0 et le moteur est `idle`. Il reste listé (`expires` `None`) mais une éviction ou un `unload` le retire. Ce cas demande un chargement plus court que la prise du verrou, donc il est théorique.

Correction sûre pour un lot 1 ? oui. Locale à `engine.rs:1269`, calquée sur `:1300-1314`, aucun format, aucun noyau, aucun token. Test sur Mac avec un `Loader` qui panique : `cargo test -p llvq-runtime --lib engine::`.

## QUA-53 : NUANCÉ, mineure

Ce qui est vrai :
- `served.rs:516-521` : `MEASUREMENT_MODES`, 19 noms.
- `fusedrun.rs:2134-2137` : liste en ligne de 17 noms, les 19 moins `LLVQ_TIME_EVENTS` et `LLVQ_BENCH_ARMS`. Aucun test ne relie les deux.

Ce qui est faux ou exagéré :
- Sous une config persistante (les configs v9 servies), `Served::from_env` passe par `Served::read` (`served.rs:706-711`), qui appelle `check_env` (`:646-656`) et refuse déjà les 19 noms. Le refus en ligne de `fusedrun` ne compte que pour une config non persistante, et là l'acceptation est voulue et testée : `served_config.rs:667` accepte `LLVQ_TIME_EVENTS` sur une config eager.
- `ALONE` (`fusedrun.rs:2071-2075`) n'est pas une copie : c'est la liste des modes exclusifs entre eux, avec `LLVQ_CONTEXTS` et `LLVQ_PREFILL_TOKENS`. Elle a ses tests (`fusedrun.rs:3967-3994`).
- Les deux noms absents sont des modes de `planesbench` (`ops/jobs/*.sh` les posent tous en ligne devant `planesbench`).

Le scénario réel : un nouveau mode lu par `fusedrun`, ajouté à `MEASUREMENT_MODES` et pas à la liste en ligne, passe en silence sous une config eager. Les configs servies sont couvertes.

Correction sûre pour un lot 1 ? oui. `fusedrun.rs` n'entre dans aucune empreinte, aucun script ne grep le texte du refus, aucun job n'exporte `LLVQ_TIME_EVENTS` ou `LLVQ_BENCH_ARMS` devant `fusedrun`. Vérification : `CUDARC_CUDA_VERSION=12040 cargo clippy --target x86_64-unknown-linux-gnu -p llvq-llm --features cuda --bin fusedrun`.

## QUA-57 : NUANCÉ, mineure

Doublon d'ARC-02. Voir ARC-02 pour la preuve.

Ce qui est vrai : `install_tree` construit toujours le module de base (`persist4e_cuda.rs:1624-1628`) et son symbole n'est lancé que pour `TreeForm::Rows` (`:2365`). Le bras servi est `A12ipg32Ft` (`served_decode.rs:456`). Le partage du temps froid entre les deux modules n'est pas mesuré : le journal donne 315,9 s pour les deux ensemble (`docs/mesures/chain-prefill-served-2026-11-15.txt:49`).

Ce que le constat sous-estime moins qu'ARC-02 : l'effort M et le risque moyen qu'il annonce sont les bons ordres de grandeur, pour les raisons d'ARC-02.

Correction sûre pour un lot 1 ? non. Elle change des lignes que trois scripts de porte lisent par grep et demande une porte sur carte.

## QUA-58 : NUANCÉ, cosmétique

Ce qui est vrai :
- `model.rs:830-836` : `in_place_from()` lit l'atomique hors `cfg(test)`. `:859-862` : sous `cfg(test)`, une variable par fil.
- Aucun test n'appelle `set_softmax_whole_for_measurement` (`git grep` : seul `fusedrun.rs:2825`).

Ce qui est faux : « aucun test ne peut l'exercer ». `cfg(test)` ne vaut que pour les tests unitaires de la bibliothèque. Un test d'intégration dans `llvq-llm/tests/` lie la bibliothèque compilée sans `cfg(test)`, donc la version de `:830` qui lit l'atomique. `set_softmax_whole_for_measurement` et `softmax_form` sont `pub`. Un test de quatre lignes suffit, sans refactorisation.

Gravité : le commutateur ne change que la mémoire, et sa seule conséquence lisible est la ligne imprimée par `fusedrun:619`. Cosmétique.

Correction sûre pour un lot 1 ? oui. Un fichier de test neuf, seul dans son binaire pour ne pas croiser l'atomique global avec d'autres tests. Aucun code de production ne bouge. Commande : `cargo test -p llvq-llm --test <nouveau_fichier>`.

## BUG-50 : CONFIRMÉ, majeure (déclencheur rare)

Chaîne d'erreur lue de bout en bout :
- `engine.rs:657` : `reuse_prefix` tronque à `cached`. `:663` : `self.memo = segments`. `:668` : `feed_with(...)?`.
- `session.rs:172-180` : `feed_with` appelle `Decoder::chain_prefill` (`served_decode.rs:73-83`), puis étend `ids` seulement en cas de succès (`:178`).
- `persist4e_cuda.rs:2196-2215` : `chain_prefill` enchaîne les `tree_chain` par `?`, rien n'est remis en place sur erreur.
- `chain_launches` (`llvq-cuda/src/spec_loop.rs:318-341`) : le premier lancement a `mv` vide, chaque suivant porte le `mv` du précédent, la passation aussi (`:348-350`).
- `tree_launch` : pour `mv` non vide, `c.reserve(pos0 + span)` puis `c.set_len(pos0)` (`:2249-2255`) avant les contrôles (`:2257-2288`), avant `prepare` (`:2289`), avant le lancement (`:2401`), avant `settle` (`:2409-2415`).
- `engine.rs:539-541` : `run` envoie un 500 avec le message. Aucun `session.reset` sur ce chemin. Le seul `reset` du moteur batch 1 est à `:756`, sur l'échec de la fermeture du message.

Requête suivante :
- Session vierge avant l'échec : `ids` vide, `keep` = 0 = `ids.len()`, `truncate` sort tôt (`session.rs:129-131`), les caches gardent leur longueur. `chain_prefill(pos0 = 0)` : le premier lancement a `mv` vide, il passe les contrôles et arrive à `prepare`, qui refuse (`:2448-2450`). Donc toute requête, quel que soit son texte, répond 500.
- Session non vide : seule une requête qui garde tout le préfixe tenu reste bloquée. Une requête dont le préfixe commun est plus court passe par `c.truncate(keep)` (`model.rs:526-551`, `self.len = len` pour Grow) et répare les caches. Le constat le dit.
- Fin du blocage : le keep-alive (`engine.rs:1342`, 5 min par défaut) ou une éviction recharge le moteur. Tant que les requêtes arrivent, chacune remet `expires` à `None` puis à `now + 5 min` : le blocage dure.
- Le moteur batch (`LLVQ_BATCH`) n'est pas touché : il prend une config batch sans chaîne (`library.rs:115-123`) et une `Session` neuve par requête (`engine.rs:917`).

Ce qui est faux dans le constat : le déclencheur déterministe. `ChainFault::PathOffByOne` fausse le chemin de chaque morceau, le premier compris (`spec_loop.rs:325-334`), donc le premier lancement échoue au contrôle de chemin, avant tout `set_len`. Et `Runtime4e::chain_prefill` fixe `ChainFault::None` (`:2199`) : la panne n'est pas injectable par ce chemin. La vérification sur carte proposée ne peut pas s'écrire telle quelle.

Déclencheurs réels : watchdog d'une barrière (`P4eFault::Abort`, `:2410`), NaN (`:2413`), ligne sans token (`:2418-2421`), OOM dans `reserve`. Les deux premiers remettent les compteurs « for the next launch » : le code veut survivre à ces erreurs, et c'est la session qui ne survit pas. Une vraie faute de carte (adresse illégale) casse le contexte entier, et le blocage devient sans objet.

Correction sûre pour un lot 1 ? oui pour la partie hôte : dans `engine.rs`, sur une erreur de `feed_with`, `session.reset` et `memo.clear()` comme à `:756`, ou la garde de `Session::truncate`. Locale, chemin d'erreur seulement, testable sur CPU (`cargo test -p llvq-llm --test session`). Non pour le déplacement des contrôles dans `tree_launch` : vérifiable seulement sur carte. Remarque sur le test CPU proposé : `set_len` exige des tampons Grow (`model.rs:607-611`) ; le petit modèle du test doit tourner sous `KvStore::Grow`.

## BUG-51 : CONFIRMÉ, mineure

- `prepare` : `c.reserve(pos0 + n)` (`persist4e_cuda.rs:2451`) puis refus `cap > MAX_CAP` (`:2453-2456`).
- `tree_launch` : `reserve(pos0 + span)` (`:2253`) avant `pos0 + span > MAX_CAP` (`:2279`).
- `KvCache::reserve` arrondit au morceau (`model.rs:595-596`), `truncate` ne rend jamais la capacité (`:526-551`).
- `MAX_CAP` = `P4E_MAX_CAP` = 8 704 (`served_decode.rs:34`), multiple de 256 (`GROW_CHUNK`). Donc pour `pos0 + n <= 8 704` la capacité ne dépasse jamais 8 704 : déplacer le contrôle avant `reserve` ne change rien hors erreur.
- Inatteignable par `llvq serve` : le contexte est plafonné par `max_positions` (`engine.rs:1272`, `served_decode.rs:145-151`), et la chaîne ne dépasse pas la fin du prompt plus un.

Correction sûre pour un lot 1 ? oui. Refuser `pos0 + n > MAX_CAP` avant `reserve` dans `prepare`, et `pos0 + span > MAX_CAP` avant le bloc `mv`. Chemin d'erreur seulement, aucun token, aucun noyau. Vérification par clippy Linux ; l'effet ne se voit que sur carte.

## SEC-02 : CONFIRMÉ, mineure

- `persist_cuda.rs:114-120` : `tensor_addr` rend un `u64`.
- `persist4e_cuda.rs:699` : `pers_4e_parts` du modèle ; `:936`, `:955-956` : adresses de `ln1`, `q_norm`, `k_norm` ; `:974-976` : `norm_w`, `cos`, `sin`. Les mots Tetra et int4 passent aussi en adresses (`seg_tetra`, `vq.wq`, `oq.wq`), et les tables du runtime Tetra (`frt.table_view()`, `:682`) ne sont pas gardées : seul `tile_define` l'est (`:1157`).
- `:1121-1160` : `Runtime4e` ne garde que ses propres tampons (`_held16`, `_held32`, `_heldf`, `_held64`, `_held8`).
- `served_decode.rs:357` : `decoder(&self, model: &Qwen3, ..) -> Decoder`, sans durée de vie. `:158-173` : `continue_greedy` ignore `model` sur le chemin 4E.
- `engine.rs:36-39` : `qwen` est déclaré avant `decoder`, donc détruit avant. Aucun `Drop` de `Runtime4e` ne lance de noyau (aucun `impl Drop` dans `persist4e_cuda.rs`).

Atteignable seulement par un appelant Rust qui détruit ou remplace le modèle avant de décoder. Aucun appelant actuel ne le fait. C'est un trou de solidité d'API, pas une faille exploitable.

Correction sûre pour un lot 1 ? non. Effort M, CUDA seulement, invérifiable sur Mac au-delà de clippy, et elle touche l'installation du chemin servi en plusieurs endroits.

## TST-01 : CONFIRMÉ, majeure

- `session.rs:225-271` : branche `if dec.is_persistent()`. Ligne du mutant : `:249`, `false => self.ids.extend_from_slice(&made[..made.len().saturating_sub(1)])`.
- `served_decode.rs:122-131` : `is_persistent` rend `false` hors Linux et `cuda`, et `rt` n'est `Some` que sur carte.
- `llvq-llm/tests/session.rs` : tous les tests passent `Decoder::eager()` (lignes 129, 172, 189, 208, 232, 301, 320, 328, 330, 347). `llvq-runtime/tests` charge des chargeurs factices.

Le trou est plus large que dit :
- Les portes sur carte ne l'exercent pas non plus. `served-v9.sh:172-180` passe par `fusedrun`, qui appelle `continue_greedy` directement, pas `Session`. `gsm8k.rs:336`, `:352` aussi.
- `chain-runtime-door.sh` passe bien par `llvq serve`, donc par `answer_fed`, mais compare la chaîne au préremplissage eager, les deux sous 4E : un défaut de comptabilité commun aux deux bras ne se voit pas.
- Le mutant de `:249` casse déjà la réponse en cours, pas seulement le tour suivant : au deuxième lancement, `offset = self.ids.len()` (`:240`) est décalé d'une position.

Le mutant survit à coup sûr. Commande : `cargo test -p llvq-llm --test session`.

Correction sûre pour un lot 1 ? non. Refactorisation de la boucle 4E, effort M, sur le chemin servi par défaut.

## TST-02 : CONFIRMÉ, mineure

- `served.rs:706-711` : `Served::read`, avec `check_env` à `:708-709`. Ses seuls appelants passent par `Served::from_env` (`:682-690`), lui-même appelé seulement par des binaires (`git grep`, douze sites).
- Aucun test ne lance un binaire qui lit `LLVQ_CONFIG` (`CARGO_BIN_EXE_` vise `embedq`, `export`, `rtbits`, `classhist`, `lswap`, `radixstudy`).
- `engine.rs:112-113` : `DeviceLoader::load` refait l'enchaînement ; aucun test ne l'appelle (`main.rs:483` seul).
- Le test le dit voulu (`served.rs:702-705`).

Ce qui est faux dans la correction : un sous-processus `fusedrun` ne marche pas sur Mac. Hors Linux et `cuda`, `fusedrun` sort à `fusedrun.rs:2106-2110`, avant `Served::from_env`. `cpurun` lit la config avant tout chargement (`cpurun.rs:41-43`) et tourne partout : c'est le bon porteur.

Mutant : supprimer `served.rs:708-709`, puis `cargo test -p llvq-llm --test served_config`. Il survit à la lecture : aucun test n'atteint `Served::read`.

Correction sûre pour un lot 1 ? oui. Test seulement.

## TST-04 : CONFIRMÉ, mineure

- `golay70_decoder_matches_rust.rs:399-414` : `cfg_attr(debug_assertions, ignore)`, puis `eprintln!("SKIP ...")` et `return` sans `$HOME` ou sans le fichier. Lit `$HOME` seulement.
- `llvq-artifact/tests/common/mod.rs:43-55` documente et corrige ce motif ailleurs.

Commande la plus légère, en gardant la chaîne d'outils (changer `HOME` seul déplacerait `CARGO_HOME` et `RUSTUP_HOME`) :
`mkdir -p /tmp/llvq-nohome && CARGO_HOME="$HOME/.cargo" RUSTUP_HOME="$HOME/.rustup" HOME=/tmp/llvq-nohome cargo test -p llvq-cuda --test golay70_decoder_matches_rust -- --ignored the_sealed_artifact_decodes_identically_through_golay70`
Attendu aujourd'hui : `ok` avec `SKIP` sur stderr. Sans le changement de `HOME`, sur un poste qui a l'archive, ce test en debug balaie 150 M blocs : à éviter.

Correction sûre pour un lot 1 ? oui. Test seulement ; `llvq-cuda/tests` ne voit pas le module `common` de `llvq-artifact`, il faut une copie.

## TST-05 : CONFIRMÉ, mineure

- `sealed.rs:157-170` : `from_env`, bras `(false, false) => Err(...)` à `:164`.
- `sealed.rs:693-704` : le test appelle `parse`, `at`, `describe_full`, jamais `from_env`.
- Aucun test ne lance `mmlu` avec les deux variables (`git grep RESTORE_` hors `sealed.rs` : seulement `mmlu.rs`, `int4swap.rs`, commentaires).

Le mutant de `:164` survit à la lecture. Commande : `cargo test -p llvq-llm --lib sealed::`.

Au passage : le commentaire de `parse` (`:151-152`) est collé en tête de celui de `from_env`.

Correction sûre pour un lot 1 ? oui. Extraire `from_lookup` ne change rien au comportement.

## TST-06 : NUANCÉ, mineure

Ce qui est vrai :
- `fused_cuda.rs:740-779` : l'unité servie est `defines`, les quatre de base, les parties du layout, `emb_q4.cu` sous `EmbedMode::Q4` (`:721-725`, `:750-755`), `tv_q4_h.cu`, puis `tv_q4_tile.cu` sous v9.
- `served_unit.rs:144-151` : `served_names` omet `emb_q4.cu` et les defines. Aucun test ne parse l'unité avec `emb_q4.cu` sans chemin d'inclusion ; `host_embq4.cpp:50` l'inclut avec ses voisins.

Ce que le constat oublie : `llvq-llm/tests/served_unit_copy.rs` reconstruit déjà l'ordre de la porte avec les defines et `emb_q4.cu` (`:29-37`) et le compare à `llvq_cuda::pers::served_unit_text`. C'est aussi une copie à la main, elle ne lit pas `fused_cuda.rs`, et elle compare des chaînes sans parser. L'ordre est donc gardé contre le banc, pas contre la porte, et la compilation de l'unité assemblée n'est gardée par rien.

Pas de mutant utile sur Mac : `fused_cuda.rs` n'y compile pas. Lecture suffisante.

Correction sûre pour un lot 1 ? non pour la correction proposée : déplacer la liste dans `fused.rs` et l'appeler depuis `fused_cuda.rs` touche l'assemblage de l'unité servie, vérifiable seulement par la ligne `NVRTC source` d'un run sur carte. oui pour une correction de test seule : ajouter `emb_q4.cu` (par `load_emb_q4_sources`) et les defines à `served_names`, puis `cargo test -p llvq-llm --test served_unit`.

## ARC-02 : NUANCÉ, mineure

Ce qui est vrai :
- `persist4e_cuda.rs:1624-1628` : module `p4t_unit_text` construit sans condition. `:1641-1642` : son symbole. `:2365` : seul lancement, pour `TreeForm::Rows`. `Tree4e.f` n'est lu nulle part ailleurs (le `v.f` de `:1578` est celui de `Verify4e`).
- `served_decode.rs:456` : le bras servi est `A12ipg32Ft`. `library.rs:82`, `:92` : « two tree modules ».

Ce qui est sous-estimé :
- Le module de base n'est pas qu'un symbole. Son `cuda` alloue tous les tampons de l'arbre (`:1756-1774`, `cuda.zeros_*`, `cuda.up_*`) et reste dans `Tree4e._cuda`. Son `dev` borne la mémoire partagée (`:1638-1640`). Son `area` donne le plafond du flux (`:1743-1744`) et ses découpes `s1` servent aux bras de 16 warps (`:1666`).
- Les scripts de porte lisent sa ligne et comptent ses échecs de cache : `chain-prefill-served.sh:121`, `:129`, `:181` ; `large-prefill-ft.sh:156` ; `large-prefill-r32.sh:157`.
- Donc effort M et risque moyen, comme QUA-57 le dit, pas S et faible.

Correction sûre pour un lot 1 ? non. Elle change des lignes lues par grep et demande une porte sur carte.

## ARC-05 : NUANCÉ, mineure

Ce qui est vrai :
- `llvq-llm/Cargo.toml:44-46` : `llvq-cuda` en dépendance normale sur macOS. `llvq-cuda/Cargo.toml:17-18` : `llvq-bench` sans condition.
- Sur Mac, dans la bibliothèque, seul `dflash_spec` l'utilise hors test (`lib.rs:33-34`, `dflash_spec.rs:21`). Les autres usages de `fused.rs` et `kvq.rs` sont des tests sous `cfg(any(target_os = "macos", feature = "cuda"))`, couverts par la dev-dépendance (`Cargo.toml:76-77`). `served_decode.rs:224`, `:450` sont sous `cfg` CUDA.
- `llvq-runtime` dépend de `llvq-llm` (`llvq-runtime/Cargo.toml:14`) : il compile donc `llvq-cuda` et `llvq-bench` sur Mac.

Ce que la correction oublie : `llvq-llm/tests/dflash_fixture.rs:1` et `dflash_prefill.rs:20` utilisent `dflash_spec` sous `cfg(any(target_os = "macos", feature = "cuda"))`. Sous la feature proposée, il faut les passer sous `spec`, et ils sortent de la boucle `cargo test` par défaut du Mac. C'est une perte de couverture à décider.

Correction sûre pour un lot 1 ? non. Décision de structure (graphe de crates, features, couverture par défaut) : à l'opérateur.

## COH-01 : NUANCÉ, mineure

Ce qui est vrai :
- `served.rs:942` : `(AttnHeads::Expanded, _) | (_, false) => Ok(())`. Sur CUDA rien ne refuse `grouped` ni `kernel`, et rien ne lit le suffixe `NOT-SERVED` du nom (`git grep` : seul le fichier kv8, par son champ). `configs/qwen3-4b-tetra-e4-metal-kernel-NOT-SERVED.json` n'a ni `decode` ni `batch`.
- `CLAUDE.md:123` : « the `*-metal-*-NOT-SERVED.json` files load only in `bin/metalrun` », sans réserve. Faux sur CUDA.
- `configs/README.md:55` : « Loaded only by `bin/metalrun` ». Faux sur CUDA.

Ce qui est exagéré :
- `configs/README.md:168-169` est dans le paragraphe « `grouped` and `kernel` are refused on Metal » : « reach them only through `bin/metalrun` » parle des formes sur Metal. Juste.
- `configs/README.md:54` dit « `served.rs` refuses the form on Metal by name, so only `bin/metalrun` loads it » : la portée Metal est lisible.
- Rien de silencieux : la ligne `served config:` imprime la note « NOT SERVED ».

Correction sûre pour un lot 1 ? oui. Texte de `CLAUDE.md:123` et `configs/README.md:55` seulement ; ne pas toucher les fichiers de config.

## COH-04 : NUANCÉ, cosmétique

Ce qui est vrai :
- `served.rs:540-545` : quatre formes acceptées, drafter 0.6B compris.
- `configs/README.md:43` (« Refused on Metal, on any shape but the 4B's ») et `:132` (« on any shape but the Qwen3-4B's ») sont faux, et `:134` se contredit avec `:132` deux lignes plus bas.

Ce qui est exagéré :
- `CLAUDE.md:123` et `ETAT.md:128` disent « on the three sealed shapes » à propos de l'objet servi. Le drafter n'est pas un objet servi, il sert la route spéculative fermée. L'oubli est défendable ; ajouter le drafter est une précision, pas une correction.

Correction sûre pour un lot 1 ? oui pour `configs/README.md:43` et `:132`. Texte seul.
