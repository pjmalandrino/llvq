# Qualité de code, crates lourdes : constats
Périmètre réellement lu : en entier `llvq-llm/src/served.rs`, `llvq-llm/src/served_decode.rs` ; `llvq-llm/src/persist4e_cuda.rs` lignes 1-1180, 1617-2090, 2218-2440 ; `llvq-runtime/src/engine.rs` lignes 1-280 et 1245-1545 ; `llvq-runtime/src/library.rs` 1-300 ; `llvq-runtime/src/api.rs` 105-160 ; `llvq-runtime/src/main.rs` 295-335 ; `llvq-llm/src/bin/fusedrun.rs` 2050-2250 ; `llvq-cuda/src/pers.rs` 690-830 ; `llvq-cuda/src/gpu.rs` 1-90 ; `llvq-cuda/src/tile.rs` 370-450 ; `llvq-llm/src/fused_cuda.rs` 40-160 et 2900-3170 ; `llvq-llm/src/model.rs` 800-890 ; `llvq-llm/src/tetra_cpu.rs` 615-655 ; `llvq-llm/src/fused.rs` 95-130. Survolés par `grep` (unwrap, expect, panic, statics, allow, env, println, features, appels de `load_resolved`) : `fused_metal.rs`, `sealed.rs`, `kvq.rs`, `batch.rs`, `session.rs`, `fused_cpu.rs`, `hfpack.rs`, `openai.rs`, `store.rs`, les binaires `chat`, `mmlu`, `gsm8k`, `cpurun`, `specrun`, `specbench`, `prefillbench`, `vtreebench`, `batchrun`, les `Cargo.toml` des quatre crates.
Non couvert : le corps de `model.rs`, `fused.rs`, `FusedRuntime::new` de `fused_cuda.rs`, `fused_metal.rs`, `kvq.rs`, `batch.rs`, `session.rs`, `tetra_cpu.rs` ; le reste de `pers.rs`, `arms.rs`, `tile.rs` ; tout `llvq-metal` ; le corps de `openai.rs`, `store.rs`, `chat.rs` ; les binaires `seal`, `hfpack`, `smoke`, `mmlu`, `gsm8k` au-delà de leurs appels de chargement.
Bilan : 0 critiques, 1 majeures, 8 mineures, 2 cosmétiques

### QUA-50 Une panique pendant le chargement d'un modèle bloque le moteur du runtime jusqu'au redémarrage
- Gravité : majeure
- Confiance : CONFIRMÉ (mécanisme lu de bout en bout) ; le déclencheur cité est lu, d'autres `expect` du chargement ne sont pas tous vérifiés
- Où : `llvq-runtime/src/engine.rs:1267-1291` (chargement), `:1300` et `:1323` (les deux seuls `catch_unwind`), `:298-300` (`idle`), `:1349-1355` (`submit`), `:1456` (seul `Failed` oublié), `:1514-1520` (`reap`) ; déclencheur `llvq-cuda/src/gpu.rs:36-43`
- Fait : le fil du moteur appelle `loader.load(...)` hors de tout `catch_unwind`. Les chemins `run` et `run_batch` sont protégés, le chargement ne l'est pas. Si le chargement panique, le fil meurt, l'état reste `State::Loading`, et le récepteur du canal est détruit.
- Scénario : `LLVQ_NVRTC_ARCH=compute89` (faute de frappe) sur `llvq serve`. Cette variable n'est pas refusée à côté d'une config, par choix (`served.rs:47`). `gpu::arch()` fait un `assert!` à la première compilation NVRTC, donc dans `DeviceLoader::load`. Le job déjà soumis laisse `queue` à 1 : le moteur n'est jamais `idle`, `reap` ne le retire pas, `unload` le marque seulement `retiring`, et sous `LLVQ_MAX_LOADED=1` tout autre modèle reçoit un 503 ("loaded and busy"). Chaque nouvelle requête sur ce modèle reçoit un 500 "the engine of X has stopped" (`:1353`), et `/api/ps` affiche `Loading` sans fin. Seul un redémarrage du serveur débloque.
- Touche : runtime
- Correction proposée : envelopper le chargement dans `catch_unwind` comme `run`, passer l'état à `State::Failed(msg)` et vider la file avec le message, ce que fait déjà la branche `Err` de `:1282-1289`. `Models::submit` oublie alors le moteur échoué et réessaie au prochain appel.
- Effort : S
- Risque de la correction : nul. Ne touche ni format, ni noyau, ni tokens.
- Vérification proposée : un test dans `engine.rs` avec un `Loader` dont `load` panique : soumettre, attendre, vérifier `State::Failed`, puis qu'une deuxième soumission rappelle le chargeur. Mutant : retirer le `catch_unwind` ajouté, le test doit échouer.

### QUA-51 La correspondance entre les 28 formes de l'arbre et leurs emplacements est écrite à la main en huit endroits
- Gravité : mineure
- Confiance : CONFIRMÉ (lu ; les huit copies concordent aujourd'hui, vérifié emplacement par emplacement)
- Où : `llvq-llm/src/persist4e_cuda.rs:1777-1807` (liste `want`), `:1812-1841` (`built`, `fx[0..23]`), `:1877-1930` (texte et symbole par indice `k`), `:1933-1939` (threads par `k`, double de `TreeForm::threads` `:340-349`), `:1940` (aire groupée par `k`, double de `:1660` par forme), `:1962` et `:1978` (bornes `22..=27` et `..22`), `:2067-2074` (déstructuration de 28 variables, décalage de 4 vers `fx`), `:2364-2394` (choix du symbole), `:1697-1727` (étiquettes) ; prédicats de famille recopiés à `:2239`, `:2291`, `:2305`, `:2334`, `:2427`
- Fait : chaque propriété d'une forme (son emplacement, ses threads, son aire, sa famille a12, son attente a9, sa lecture en place) est un `match` ou un indice numérique local. `TreeForm` a déjà `parent`, `rmax` et `threads`, mais aucune autre propriété n'y vit.
- Scénario : ajouter une forme, ou en retirer une des 24 formes de mesure (a3f, a3p, a3t sont des planchers aux valeurs vides), demande de modifier huit sites en gardant les indices alignés. Un décalage d'un cran dans `fns` ou `fx` lance le symbole d'une autre forme sans erreur de compilation. Le chemin servi passe par là : le préremplissage en chaîne du 4B installe `A12ipg32Ft`, emplacement 27, `fx[23]` (`served_decode.rs:456`).
- Touche : chemin servi (installation du préremplissage en chaîne)
- Correction proposée : une table `TreeForm::ALL: [TreeForm; 28]` et des méthodes `slot()`, `label()`, `grouped_area()`, `has_taps()`, `reads_in_place()`, `waits_a9()` ; `TreeTiled` garde un seul tableau `[Option<CudaFunction>; 28]` indexé par `slot()`.
- Effort : M
- Risque de la correction : faible. Code hôte seulement, aucun texte de noyau, aucune empreinte. Les lignes d'installation doivent rester identiques au caractère : `ops/jobs/chain-prefill-served.sh:119-129` les lit.
- Vérification proposée : vérification de type Linux (`ops/check-linux-cuda.sh`) ; sur carte, la porte de la chaîne compare ses lignes d'installation à celles du journal `chain-prefill-served-2026-11-15`.

### QUA-52 `install_inner`, 507 lignes, mêle l'installation servie de 4E et les bras de mesure E1, avec sept copies du même sondage de symbole
- Gravité : mineure
- Confiance : CONFIRMÉ (lu)
- Où : `llvq-llm/src/persist4e_cuda.rs:658-1164` ; motif "symbole, rapport, occupation, ligne" à `:614-623`, `:718-724`, `:745-751`, `:767-773`, `:1468-1473`, `:1642-1647`, `:1947-1952`
- Fait : le chemin servi (`Served::decoder` → `install_rows` → `install_inner`) traverse les branches des bras E1 : symbole des logits (`:759-776`), remplissage v9k8 (`:778-799`), gabarit des corps `rc` (`:1009-1030`), rembourrage `smem_of` (`:707-711`). Ce sont des mesures closes ou gardées comme pistes (ETAT §6 : E1, v9ft, v9k8). La construction de l'image d'une couche fait 93 lignes en ligne (`:874-967`).
- Scénario : chaque modification du chemin servi se relit au milieu de quatre branches de mesure. L'ancien audit comptait `install_rows_as` à 421 lignes ; la fonction a grandi de 86 lignes depuis.
- Touche : chemin servi
- Correction proposée : extraire `probe_symbol(cuda, sym, threads, smem) -> (CudaFunction, rapport, occupation)` pour les sept copies, `layer_image(...)` pour la boucle `:874-967`, et regrouper les trois blocs E1 dans une fonction appelée seulement quand `e1_arm` est `Some`.
- Effort : M
- Risque de la correction : faible. Code hôte, aucun noyau ni empreinte. Les lignes doivent rester identiques : `ops/jobs/served-v9.sh:150` et `served-v9-big.sh:158` font un `grep -F` sur "4E: persistent module ... row form 7,".
- Vérification proposée : vérification de type Linux ; la porte v9 sur carte relit ses lignes et ses tokens.

### QUA-53 Trois listes des modes de mesure écrites à la main, et aucun test ne les relie
- Gravité : mineure
- Confiance : CONFIRMÉ (lu, comparé nom par nom)
- Où : `llvq-llm/src/served.rs:516-521` (`MEASUREMENT_MODES`, 19 noms) ; `llvq-llm/src/bin/fusedrun.rs:2134-2137` (refus à côté d'une config, 17 noms) ; `fusedrun.rs:2072-2076` (`ALONE`, 17 noms)
- Fait : la liste en ligne de `fusedrun` est `MEASUREMENT_MODES` moins `LLVQ_TIME_EVENTS` et `LLVQ_BENCH_ARMS`, recopiée au lieu d'être lue. Le seul test (`tests/served_config.rs:659-668`) lit `MEASUREMENT_MODES` et ignore les deux autres listes.
- Scénario : un nouveau mode ajouté à `MEASUREMENT_MODES` reste accepté en silence par `fusedrun` sous `LLVQ_CONFIG`, et l'inverse. C'est la classe de défaut que l'audit du 2026-09-11 a fermée une première fois.
- Touche : banc (porte servie de `fusedrun`)
- Correction proposée : boucler sur `llvq_llm::served::MEASUREMENT_MODES` à `fusedrun.rs:2134`. Le seul effet : `TIME_EVENTS` et `BENCH_ARMS`, que `fusedrun` ne lit pas, y seraient aussi refusés.
- Effort : S
- Risque de la correction : nul. `fusedrun.rs` n'entre dans aucune empreinte.
- Vérification proposée : aucune, lecture suffisante ; `cargo clippy` en Linux.

### QUA-54 La taille de l'étage d'attention de 4E est recopiée cinq fois dans `pers.rs`, et celle des états de split trois fois dans `persist4e_cuda.rs`
- Gravité : mineure
- Confiance : CONFIRMÉ (lu)
- Où : `llvq-cuda/src/pers.rs:1283`, `:1577`, `:2195`, `:2297`, `:3093` (`16 * 1024 * ring + 4 * 4 * groups * 130 + 4 * groups * 128`) ; `llvq-llm/src/persist4e_cuda.rs:847`, `:1490`, `:1760` (`n_kv * ceil(cap / 256) * groups * 130`, écrit une fois avec `n_heads`, deux fois avec `n_kv * groups`)
- Fait : ces formules reflètent `PERS_ATTN_SMEM` de `llvq_pers.cuh`. Chaque copie est littérale ; les constantes 130 (tête 128 plus deux flottants) et 256 (taille d'un split) n'ont pas de nom côté hôte.
- Scénario : une copie qui diverge du noyau sous-dimensionne la mémoire partagée d'un lancement 4E, de vérification ou d'arbre, et le noyau déborde sa zone. Aujourd'hui les cinq copies concordent.
- Touche : chemin servi (dimensionnement des lancements 4E et de l'arbre)
- Correction proposée : `pers::p4e_attn_stage_bytes(ring, groups)` et `pers::p4e_split_floats(n_kv, groups, cap)`, appelées partout.
- Effort : S
- Risque de la correction : faible. `pers.rs` est de l'hôte ; les empreintes portent sur le texte des noyaux, pas sur ce fichier.
- Vérification proposée : `cargo test -p llvq-cuda` sur le Mac (les tests de `pers.rs` recalculent ces tailles), puis un mutant sur une constante.

### QUA-55 Quatorze appels recopient les six champs d'un `Served` dans `load_resolved`
- Gravité : mineure
- Confiance : CONFIRMÉ (lu)
- Où : `llvq-runtime/src/engine.rs:127-138` ; `llvq-llm/src/bin/chat.rs:115`, `mmlu.rs:553`, `gsm8k.rs:136`, `cpurun.rs:46`, `batchrun.rs:85`, `specrun.rs:190`, `specbench.rs:206`, `prefillbench.rs:322`, `vtreebench.rs:386`, `fusedrun.rs:2839`, `:2967`, `:3143`
- Fait : chaque appel écrit `cfg.layout, cfg.embed, cfg.rot_share, cfg.fuse, cfg.kv, cfg.int4_layout()`. Sept passent directement par `fused_cuda::load_resolved` au lieu du répartiteur `served::load_resolved`.
- Scénario : un septième champ chargé (comme `int4_layout` l'a été le 2026-11-07) demande quatorze modifications. Un appel qui passe `Int4Layout::Served` à la main au lieu de `cfg.int4_layout()` sert v9 avec deux copies des mots int4 ou de mauvais mots.
- Touche : runtime, banc, outillage
- Correction proposée : `Served::load(&self, path, device, dtype, origin)` qui appelle `served::load_resolved` avec ses propres champs.
- Effort : S
- Risque de la correction : nul. Mêmes arguments, même fonction appelée.
- Vérification proposée : `cargo clippy --all-targets` et la vérification de type Linux.

### QUA-56 Les trois contrôles de parité int4 au chargement sont trois copies du même corps
- Gravité : mineure
- Confiance : CONFIRMÉ (lu)
- Où : `llvq-llm/src/fused_cuda.rs:3023-3052`, `:3059-3093`, `:3118-3143`
- Fait : chaque fonction tire `parity_activation`, alloue `ya` et `yb`, lance deux noyaux, cherche la première ligne qui diffère, formate l'erreur puis imprime une ligne. Seuls les deux lancements et les noms changent. `check_int4_tile_parity` tourne à chaque chargement v9 servi.
- Scénario : une correction du protocole (activation, valeur sentinelle `0xffff`) doit être faite trois fois.
- Touche : chemin servi (chargement, pas les tokens)
- Correction proposée : `compare_rows(name, a, b, what) -> Result<(), String>` pour la recherche et le message, les lancements restant dans chaque fonction.
- Effort : S
- Risque de la correction : nul. Les lignes imprimées doivent rester identiques.
- Vérification proposée : aucune, lecture suffisante ; la porte v9 relit la ligne "int4: tv_q4_hht ... bit for bit".

### QUA-57 Le préremplissage en chaîne servi compile un module d'arbre dont il ne lance jamais le symbole
- Gravité : mineure
- Confiance : CONFIRMÉ pour le code mort ; la part du temps de démarrage à froid est à mesurer
- Où : `llvq-llm/src/persist4e_cuda.rs:1625-1650` (module de base, 512 threads), `:2365` (seul usage de `Tree4e.f`, forme `Rows`) ; `llvq-llm/src/served_decode.rs:456` (le bras servi est `A12ipg32Ft`)
- Fait : `install_tree` compile et sonde toujours le module `p4t_unit_text`, quels que soient les bras demandés. Son symbole ne sert qu'au bras `Rows`. Le bras servi lit `fx[23]` de son propre module.
- Scénario : un premier démarrage du 4B sur une carte compile "two tree modules", 315,9 s de NVRTC et de JIT sur 509,2 s (*mesuré*, `docs/mesures/chain-prefill-served-2026-11-15.txt:49`) ; le module de base pèse 306 060 octets sur 763 210 (*mesuré*, `chain-prefill-served.sh:129`). À signaler au relecteur OPT.
- Touche : chemin servi (démarrage)
- Correction proposée : ne construire le module de base que si un bras `Rows` est demandé ; `Tree4e.f`, `smem` et `area` deviennent optionnels.
- Effort : M
- Risque de la correction : moyen. La porte de la chaîne compte deux échecs de cache et lit l'empreinte du module de base (`chain-prefill-served.sh:121`, `:129`) : il faut la réécrire et la refaire passer sur carte.
- Vérification proposée : un démarrage à froid sur carte, compter les modules et le temps de chargement.

### QUA-58 Le commutateur `SOFTMAX_WHOLE` n'est jamais lu sous `cfg(test)`, donc aucun test ne peut l'exercer
- Gravité : mineure
- Confiance : CONFIRMÉ (lu)
- Où : `llvq-llm/src/model.rs:824-867`
- Fait : en build de test, `in_place_from()` est remplacée par une variable par fil (`:853-862`) et ne lit plus l'atomique global. `set_softmax_whole_for_measurement` n'a donc aucun effet sous test, et `softmax_form()` y rend l'état du fil, pas celui du commutateur.
- Scénario : une régression qui casse le commutateur (par exemple `usize::MAX` remplacé par une autre valeur à `:833`) passe toute la suite. Seul un second processus sur carte (`LLVQ_SOFTMAX_WHOLE=1`, `ops/jobs/pic-vram-8k.sh:86`) la verrait.
- Touche : banc
- Correction proposée : un seul `in_place_from()` qui lit d'abord la valeur par fil si elle est posée, puis l'atomique ; ou une fonction pure `threshold(whole: bool) -> usize` testée directement.
- Effort : S
- Risque de la correction : nul. Hors build de test, le comportement reste celui d'aujourd'hui.
- Vérification proposée : un test qui pose le commutateur et lit `softmax_form()`.

### QUA-59 Docs de module et commentaires périmés sur le chemin servi (regroupés)
- Gravité : cosmétique
- Confiance : CONFIRMÉ (lu)
- Où et fait :
  - `llvq-llm/src/persist4e_cuda.rs:1-5` et `:13-27` : nomment comme config servie `qwen3-4b-tetra-e4-4e-v4.json` et son jumeau v6, "on the sealed 4B". Le servi est v9 aux trois tailles (ETAT §2).
  - `persist4e_cuda.rs:191-194` : `Tree4e` est dit "Not a served path: S1 measures it". Le 4B préremplit par cette passe depuis le 2026-11-15 (`served.rs:377-395`, `served_decode.rs:448-456`). C'est le plus trompeur : il invite à traiter l'arbre comme un banc.
  - `persist4e_cuda.rs:487-489` : "2 is v4, what the served config runs", et la liste s'arrête avant la forme 7.
  - `llvq-cuda/src/pers.rs:700-703` : la doc "Shared bytes of a 4E CTA" est orpheline, collée sur `P4E_ROWS_FORMS` ; `:705` dit la forme 1 "what the served config runs" ; l'énumération s'arrête à 6 ; `:718` dit v4 "the served form".
  - `llvq-llm/src/served.rs:44` : "four variables that reach BELOW the served door" ; `BELOW_THE_DOOR` en a sept (`:570-578`).
  - `llvq-llm/src/served_decode.rs:180-191` : la doc de `install_verify` est collée sur `runtime_mut`, et `install_verify` n'a pas de doc.
  - `served_decode.rs:354-356` : renvoie à `persist4e_cuda::install`, le code appelle `install_rows` (`:400`).
  - `llvq-llm/src/fused_cuda.rs:50` : `THREADS` est dit "for `tv_slot`" ; il dimensionne aussi les lancements int4 (`:3037`, `:3077`, `:3128`).
- Touche : doc
- Correction proposée : réécrire ces phrases sur l'objet servi v9 et la chaîne, déplacer la doc orpheline sur `p4e_smem_bytes`.
- Effort : S
- Risque de la correction : nul. Commentaires Rust seulement, aucun texte de noyau.
- Vérification proposée : aucune, lecture suffisante.

### QUA-60 Code mort sans effet dans les crates lourdes (regroupé)
- Gravité : cosmétique
- Confiance : CONFIRMÉ (`git grep` des appelants)
- Où et fait :
  - `llvq-llm/src/persist4e_cuda.rs:483-485` : `install` n'a aucun appelant. Tous les appelants passent `Some(512)` (`served_decode.rs:404`, `fusedrun.rs:2471`, `:2485`, `persist4e_cuda.rs:643`, `:655`), donc la branche `None => vec![768, 512]` (`:713-716`) et la doc des 768 threads (`:479-482`) ne servent plus.
  - `llvq-runtime/src/library.rs:290-297` : `Backend::config` rend `Result<Option<&str>, _>` sans jamais rendre `Ok(None)` ; le `ok_or_else` de `:125` répète le même message et ne peut pas se déclencher.
  - `llvq-runtime/src/engine.rs:1415` : `max_loaded.max(1)` double le filtre `> 0` de `from_env` (`:1421`).
  - `llvq-llm/src/fused_metal.rs:71-72` : `#[allow(dead_code)] const ROW_LANES` garde la trace d'une piste refusée ; une phrase de doc suffit.
  - `llvq-llm/Cargo.toml:56` : la feature `cudnn` n'est activée par aucun job, aucun Dockerfile, aucun script (`git grep cudnn`).
  - `llvq-llm/src/served.rs:444-463` et `:529-554`, `persist4e_cuda.rs:536` : la forme du 4B est écrite trois fois en tuple littéral.
- Touche : outillage
- Correction proposée : supprimer `install` et la branche 768 côté hôte (les symboles 768 restent dans le texte épinglé), aplatir `Backend::config` en `Result<&str, _>`, retirer le `max(1)`, `ROW_LANES` et la feature `cudnn`, et nommer `QWEN3_4B_SHAPE` une fois.
- Effort : S
- Risque de la correction : nul. Aucun noyau ni empreinte.
- Vérification proposée : `cargo clippy --all-targets` et la vérification de type Linux.

## CONNU-NON-CORRIGÉ

- `MEASUREMENT_MODES` contient encore les trois noms que seul `planesbench` lit (`served.rs:518` : `TIME_EVENTS`, `SEG_ARMS`, `BENCH_ARMS`), et `fusedrun` liste `SEG_ARMS` sans jamais le lire (`fusedrun.rs:2075`, `:2135`). Réf. `docs/audit-architecture-2026-11.md` §1.5 et lot 1.
- 4E servi importe cinq fonctions du module de la route P (`persist4e_cuda.rs:50` ← `persist_cuda.rs`). Réf. §2.3 et lot 4.
- `gpu::arch()` et `tile::request()` paniquent sur une valeur fausse (`gpu.rs:38`, `tile.rs:405`). `arch` n'a toujours pas de parseur pur, alors que `tile::parse_request` en a un ; c'est le déclencheur de QUA-50. Réf. §2.7.
- Le chargeur de `LLVQ_KERNEL_DIR` est recopié douze fois (`fused.rs:705`, `:761`, `:919`, `:1119`, `fused_cuda.rs:67`, `llvq-cuda/src/lib.rs:375`, `planesbench.rs` ×7) ; trois parseurs de `LLVQ_THREADS` (`tetra_cpu.rs:626`, `smoke.rs:671`, `f1recdump.rs:169`). Réf. §2.7.
- `LLVQ_FUSED_LAYOUT` vaut `planes14` par défaut, et `Golay70`, piste close, reste dans l'énumération servie (`fused.rs:97`, `:101`). Réf. §2.4.
- Quatre commutateurs globaux de mesure dans `fused_cuda.rs` (`:2918`, `:2930`, `:2969`, `:3148`). Réf. §3.1.

## Notes pour les autres relecteurs

- Concurrence : toutes les sections critiques du runtime font `lock().expect(...)` (`engine.rs:1446`, `:1477`, `:1501`, `:1510`, `:1515`, `:1523`, `:1532`). Si un fil panique en tenant un verrou, chaque requête HTTP suivante panique sur l'empoisonnement. Non vérifié qu'un tel chemin existe.
- Cohérence : `CLAUDE.md` dit que chaque mode de mesure est refusé à côté de `LLVQ_CONFIG`. `check_env` ne les refuse que sous `decode: persistent` (`served.rs:646`) ; sous une config eager (Metal et CPU du runtime), seul `fusedrun` les refuse.

## Candidats au lot 1

QUA-50, QUA-53, QUA-55, QUA-56, QUA-58, QUA-59, QUA-60
