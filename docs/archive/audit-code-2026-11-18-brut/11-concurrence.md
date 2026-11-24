# Concurrence et cycle de vie du runtime : constats
Périmètre réellement lu : en entier, `llvq-runtime/src/engine.rs`, `api.rs`, `openai.rs`, `chat.rs`, `client.rs`, `library.rs`, `main.rs`, `store.rs`, `lib.rs` ; `llvq-runtime/tests/common/mod.rs`, `tests/batch_server.rs`, `tests/engine_busy.rs` ; `llvq-llm/src/session.rs`, `batch.rs`, `served_decode.rs`. Survolés : `persist4e_cuda.rs` (`launch`, `settle`, `prepare`, `chain_prefill`, `tree_launch`), `served.rs` (`BatchServe`, `Decode`, `load_resolved`), `model.rs` (`KvCache`, `fresh_caches`, `batch_kv*`, `batch_step_slots`), `batch_cuda.rs` (flux CUDA), `llvq-cuda/src/pers.rs` (`p4e_settle`), `chatfmt.rs` (`turn`, `open_assistant`), `chat.html` (pull), la liste des tests de `tests/server.rs` et `tests/pull.rs`.
Non couvert : le corps de `tests/server.rs`, `tests/pull.rs`, `tests/openapi.rs` ; les noyaux CUDA du batch (`tv_batch_gemm.cu`, attention) et l'installation de 4E ; le reste de `chat.html`.
Bilan : 0 critiques, 3 majeures, 8 mineures, 0 cosmétiques

## Modèle de threads

1. Un fil OS par moteur chargé (`llvq-engine-<nom>`, engine.rs:1267). Il possède le modèle, le décodeur 4E, la session de conversation et le pool batch ; aucun autre fil n'y touche.
2. Les handlers HTTP sont des tâches tokio (runtime multi-thread, main.rs:485). Ils appellent `Models::submit`, synchrone, sous le verrou `engines`, puis lisent un canal tokio borné à 64 événements (engine.rs:1499).
3. Les jobs entrent dans le fil moteur par un `std::sync::mpsc` non borné (engine.rs:1265). Le fil moteur répond par `blocking_send`, qui bloque quand les 64 places sont prises.
4. Batch 1 : un job à la fois, dans l'ordre d'arrivée ; la session reste d'un job au suivant.
5. `LLVQ_BATCH` : le même fil unique admet, préremplit, fait le pas groupé et bascule entre 4E et le pool. Aucune course entre deux séquences du pool : tout est séquentiel sur ce fil et sur un seul flux CUDA (batch_cuda.rs:186, :342).
6. Le faucheur (tâche tokio, toutes les secondes, api.rs:1052) et les handlers lisent `queue` et `busy` (atomiques `SeqCst`), `state` et `expires` (`Mutex`) ; le fil moteur les écrit.
7. Ordre des verrous : `engines` puis `state` ou `expires`. Le fil moteur ne prend jamais `engines`. Aucun interblocage possible.
8. Les pulls tournent dans `spawn_blocking`, sérialisés entre eux par un verrou de fichier par blob (store.rs:159).
9. Les vraies courses sont donc : entre handlers sous `engines`, entre le fil moteur et ses clients (le canal), entre processus sur le store.
10. Conséquence directe : ce qui bloque le fil moteur bloque tout ce que ce moteur sert.

## Constats du 2026-11-08 et du 2026-11-07, vérifiés à 925930d

- §1.1 (4E empoisonné) : corrigé. `settle` remet `bar`, `abort`, `nan` et `bar_base` à 0 après une faute (persist4e_cuda.rs:1309-1312, :1385-1391 ; pers.rs:1616-1635).
- §1.2 (requête seule dans le pool) : corrigé par `back_to_solo` et `resume` (engine.rs:869-877, :951-987). Un job `Load` n'annule plus 4E (engine.rs:1050-1054). Reste : une arrivée refusée en 400, ou déjà abandonnée, annule encore 4E (engine.rs:1055-1058) ; le coût n'est plus que deux copies de KV aller-retour.
- §1.5 (store figé) : corrigé, `served_configs` par sha256 (library.rs:192-194).
- §1.7 (au-delà de 8 704) : corrigé par `solo_room` et le passage au pool au plafond (engine.rs:1020, :1087-1096).
- §1.8 (pool invisible) : corrigé, `check_free` (batch.rs:204-206, :290-307) et `size_vram` (api.rs:535). Reste : aucune marge pour le préremplissage, voir CON-04.
- CONNU-NON-CORRIGÉ : un client connecté qui ne lit plus bloque le moteur (runtime-audit-2026-11, « Found and not fixed »). Aggravé sous `LLVQ_BATCH` : CON-01.
- CONNU-NON-CORRIGÉ : un long préremplissage ne s'annule pas (runtime-audit-2026-11) ; `Session::feed` n'a toujours pas de rappel (session.rs:155-166).
- CONNU-NON-CORRIGÉ : un `feed` qui échoue entre deux couches laisse les caches plus longs que `ids` (runtime-audit-2026-11 ; audit-bugs-2026-11-08 §4) ; `truncate` sort tôt (session.rs:129-131). Même cause que CON-02, sur le chemin servi par chaîne.
- CONNU-NON-CORRIGÉ : `num_predict: 0` ou `max_tokens: 0` veut dire « jusqu'à la fin du contexte » au lieu d'un 400 (audit-bugs-2026-11-08 §4) ; engine.rs:620-623.

## Constats

### CON-01 Sous `LLVQ_BATCH`, un client qui cesse de lire gèle toutes les requêtes du pool
- Gravité : majeure
- Confiance : CONFIRMÉ
- Où : llvq-runtime/src/engine.rs:1201-1205, :1197, :1220-1234, :855-860, :1499 ; :1039 (décode seul)
- Fait : le fil moteur unique envoie chaque morceau de chaque requête vivante par `blocking_send` sur un canal de 64 places (engine.rs:1499). `advance` attend sans délai qu'une place se libère (engine.rs:1202). Le pas suivant n'a lieu qu'après la boucle sur toutes les requêtes vivantes (engine.rs:855-860).
- Scénario : fil client X : connecté, ne lit plus (`llvq run … | less` non défilé, onglet figé, proxy lent). Fil moteur : à chaque pas, `advance` pousse un `Delta` dans le canal de X. Après 64 morceaux et les tampons de socket, `blocking_send` (engine.rs:1202) ne rend plus la main. Les 63 autres requêtes vivantes ne reçoivent plus aucun jeton, en plein flux ; la file attend ; `llvq stop` ne fait que poser `retiring`, car le moteur reste occupé. Le constat du 2026-11-07 ne voyait que la file du batch 1 ; ici des réponses déjà commencées s'arrêtent net pour d'autres clients.
- Touche : runtime
- Correction proposée : envoyer par `try_send` dans le pool et, sur `Full`, traiter le client comme parti après un délai court (horodatage par `Live`), puis libérer le slot comme pour `After::Gone`. Même règle dans le rappel de `solo_decode` et dans `run`.
- Effort : M
- Risque de la correction : faible ; ni format, ni index, ni tokens servis.
- Vérification proposée : test CPU dans `tests/batch_server.rs`, au niveau de `Models` sans HTTP : soumettre B (`num_predict` 400) et lire ses événements ; au premier morceau, soumettre A (`num_predict` 400) et garder son `Receiver` sans le lire ; exiger le `Done` de B en moins de 10 s. Aujourd'hui B se fige environ 64 pas après l'arrivée de A.

### CON-02 Un préremplissage par chaîne qui échoue laisse les caches plus longs que `ids` : la conversation répond 500 à chaque nouvel essai
- Gravité : majeure
- Confiance : CONFIRMÉ (chemin lu de bout en bout ; le déclencheur est une faute de lancement 4E, rare)
- Où : llvq-llm/src/persist4e_cuda.rs:2196-2215, :2249-2255, :2289, :2448-2450 ; llvq-llm/src/session.rs:129-131, :143-147, :172-179 ; llvq-runtime/src/engine.rs:539-541, :657, :668 ; llvq-runtime/src/main.rs:472-475
- Fait : le 4B sur CUDA prefill par chaîne par défaut (`configs/qwen3-4b-tetra-e4-4e-v9-chain.json`). `chain_prefill` enchaîne des lancements de 32 lignes. Chaque lancement qui déplace des lignes pose `set_len(pos0)` sur tous les caches avant de lancer (persist4e_cuda.rs:2254). `feed_with` n'allonge `ids` qu'après le succès du tout (session.rs:177-178). Après une erreur au lancement k ≥ 2 (watchdog, NaN, « a row wrote no token », OOM de `reserve`), les caches tiennent `pos0_k` positions et `ids` moins. `run` envoie le 500 et garde la session telle quelle (engine.rs:539-541).
- Scénario : prompt de 2 000 jetons, environ 63 lancements ; le 10e sort sur le watchdog. Le client renvoie la même conversation. `reuse_prefix` trouve tout `ids` en préfixe, donc `truncate(ids.len())` sort tôt (session.rs:129-131) sans couper les caches. Le premier lancement du nouvel essai passe par `prepare`, qui refuse : « 4E: a cache holds X positions, the launch starts at Y » (persist4e_cuda.rs:2448-2450). Même 500 à chaque essai, et à chaque tour suivant : `llvq run` retire le message raté (main.rs:473) et renvoie un historique qui garde tout `ids` en préfixe. Seule une requête qui diverge plus tôt débloque (`/clear`, un autre client). Depuis la correction du §1.1, une faute coûtait une requête ; sur ce chemin elle coûte la conversation.
- Touche : chemin servi
- Correction proposée : dans `Session`, après toute erreur de `feed`, `feed_with` ou `answer_fed`, recaler chaque cache sur `ids.len()` (`KvCache::truncate` accepte une longueur plus courte sous Grow). Ou faire couper les caches par `Session::truncate` dès que leur longueur diffère de `ids.len()`. Couvre aussi le cas connu du `feed` éager.
- Effort : S
- Risque de la correction : faible ; aucun jeton servi ne bouge sur le chemin sans erreur ; aucun format, aucun fichier épinglé.
- Vérification proposée : sur carte, prompt de 100 jetons, faire échouer le 2e lancement de `chain_prefill` par un crochet de test, puis rejouer la même requête : 500 aujourd'hui. Sur CPU, une fois le recalage écrit : test de l'invariant « après toute `Err` d'une méthode de `Session`, chaque cache tient `ids.len()` positions ».

### CON-03 Un panic pendant le chargement laisse un moteur « loading » pour toujours, et bloque tout le serveur sous `LLVQ_MAX_LOADED=1`
- Gravité : majeure
- Confiance : PLAUSIBLE (la suite est lue ; il faut un panic dans `Loader::load`)
- Où : llvq-runtime/src/engine.rs:1267-1279, :1280-1290, :1349-1355, :1456, :1475, :1517-1518, :1380
- Fait : `loader.load` et `Marks::resolve` tournent hors de tout `catch_unwind` (engine.rs:1269-1279). Seul un `Err` passe l'état à `Failed`. Un panic tue le fil : l'état reste `Loading`, le `Receiver` part avec le job qu'il contenait, et `queue` reste à 1, car aucun `fetch_sub` ne vient.
- Scénario : handler H1 : `Models::submit` lance le moteur (engine.rs:1493), puis `Engine::submit` met `queue` à 1 et envoie le job (engine.rs:1350-1351). Fil moteur : `load` panique après quelques secondes (candle, cudarc ou tokenizers qui paniquent au lieu de rendre une erreur). Le client de H1 reçoit « the engine stopped before answering ». Ensuite, `retain` n'oublie que `Failed` (engine.rs:1456). Le moteur n'est jamais `idle` (queue à 1) : ni l'éviction (engine.rs:1475), ni le faucheur (engine.rs:1518), ni `llvq stop` (engine.rs:1377-1380) ne le retirent. Toute requête au même modèle reçoit 500 « the engine of X has stopped » (engine.rs:1351-1353) ; toute requête à un autre modèle reçoit 503 sous le plafond par défaut de 1. Il faut redémarrer le serveur.
- Touche : runtime
- Correction proposée : envelopper le chargement dans `catch_unwind`, comme `run` et `run_batch`, et passer à `Failed` avec le message du panic. En plus, faire oublier par `Models::submit` un moteur dont l'envoi échoue.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : test CPU dans `tests/engine_busy.rs` : un `Loader` qui dort 200 ms puis panique. `submit` A et lire `None` ; `submit` A de nouveau : exiger une deuxième tentative de chargement (compteur à 2) et non un 500 « has stopped » ; `submit` B sous plafond 1 : pas de 503.

### CON-04 Une erreur de carte qui persiste laisse le moteur « ready » : chaque requête répond 500
- Gravité : mineure
- Confiance : PLAUSIBLE
- Où : llvq-runtime/src/engine.rs:539-541, :843-851, :1147-1153, :1456 ; llvq-llm/src/batch.rs:199-207, :296
- Fait : une erreur du modèle (`run_inner`, un pas groupé, un `admit`) ne part qu'au client concerné. Le moteur reste `Ready` avec le même `Worker`. Seuls un panic et un chargement raté passent à `Failed`, l'état que `Models::submit` oublie pour recharger (engine.rs:1456).
- Scénario : 1) Une erreur CUDA collante (accès illégal dans un noyau, Xid) empoisonne le contexte. Chaque requête suivante répond 500 avec le même message, `/api/ps` dit `ready`, et le keep-alive ne décharge jamais le moteur sous trafic, car chaque job repousse `expires`. 2) `LLVQ_BATCH` dimensionné pour que le pool tienne tout juste : `check_free` accepte un pool égal à la mémoire libre (batch.rs:296), sans marge. Chaque préremplissage d'un prompt long échoue en OOM, chaque fois en 500, moteur `ready`. Le commentaire de `Pool::new` l'admet (batch.rs:201-203).
- Touche : runtime
- Correction proposée : passer en `Failed` sur les erreurs collantes du pilote (`ILLEGAL_ADDRESS`, `LAUNCH_FAILED`), avec « restart the server » ; exiger dans `check_free` une marge égale au pic d'un préremplissage de `context` jetons.
- Effort : M
- Risque de la correction : faible.
- Vérification proposée : test CPU dans `tests/batch_server.rs` : un `BatchDevice` dont `attend` rend toujours `Err`, deux requêtes de deux jetons au moins : deux 500, et `/api/ps` dit `ready` aujourd'hui.

### CON-05 La même requête de chat reçoit un prompt différent selon qu'un autre client a parlé entre deux tours
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-runtime/src/chat.rs:65-71 ; llvq-runtime/src/engine.rs:663, :760-763 ; llvq-llm/src/chatfmt.rs:72-93, :106-127
- Fait : après une réponse propre, le memo garde pour le message assistant les ids de la session depuis l'ouverture (engine.rs:760-763) : `<|im_start|>assistant\n`, le bloc de raisonnement (vide, 4 ids, quand think est off ; le raisonnement entier quand il est on), les ids générés, `<|im_end|>\n`. `render` réutilise ces ids tant que le message reçu a le même rôle et le même texte (chat.rs:66-68). Sinon il rend le texte par `turn` (chatfmt.rs:72-93), sans bloc de raisonnement et par un réencodage BPE.
- Scénario : client A, tour 1, puis tour 2 aussitôt : le prompt du tour 2 porte le bloc vide et les ids générés. Si un client B a envoyé une requête entre les deux tours, le memo est remplacé (engine.rs:663) et le même historique de A est rendu par texte : 4 ids de moins au moins, et d'autres ids pour la réponse si le BPE ne retombe pas sur les mêmes. La réponse gloutonne de A au tour 2 dépend donc de l'entrelacement avec B, et son `prompt_eval_count` aussi. Sous `LLVQ_BATCH` (pas de memo) c'est toujours le rendu texte. Avec think on, le chemin memo montre au modèle son raisonnement précédent, que le gabarit de Qwen3 retire.
- Touche : runtime
- Correction proposée : décision d'opérateur. Soit rendre toujours l'historique par texte et ne garder le cache que par préfixe d'ids (le dernier message assistant est alors relu à chaque tour) ; soit garder le memo et documenter qu'une conversation n'est déterministe que si personne ne parle entre deux tours.
- Effort : M
- Risque de la correction : moyen ; change les prompts multi-tours servis, pas l'identité mono-tour publiée.
- Vérification proposée : test CPU dans `tests/server.rs` avec `TinyLoader` : serveur 1 joue A1 puis A2 ; serveur 2 joue A1, B1, A2 ; comparer le `prompt_eval_count` de A2. Il diffère de 4 aujourd'hui.

### CON-06 Un panic du moteur batch coupe les flux en cours sans `done` ni erreur
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-runtime/src/engine.rs:792-793, :1300-1314, :1330-1332 ; llvq-runtime/src/api.rs:180-189 ; llvq-runtime/src/openai.rs:325-361 ; llvq-runtime/src/client.rs:115-126
- Fait : `live` et `pending` sont des locales de `run_batch` (engine.rs:792-793). Un panic les détruit pendant le déroulement : leurs `Sender` tombent sans dernier événement. Le gestionnaire (engine.rs:1300-1314) ne répond qu'aux jobs arrivés après. Le chemin batch 1, lui, envoie l'erreur au job en cours (engine.rs:1332).
- Scénario : un panic dans un pas (un `expect` atteint, une panique de candle) avec 10 flux NDJSON ouverts. `events` voit le canal fermé et termine le flux (api.rs:185) : réponse 200 tronquée, sans ligne `done: true` ni `error`. `llvq run` rend `Ok` à la fin du flux (client.rs:115-126) et affiche la réponse partielle comme complète. Un client OpenAI ne reçoit jamais `data: [DONE]`. Les jobs de `pending` reçoivent bien un 500 (`accepted` voit `None`), mais `queue` garde leur compte jusqu'à la requête suivante.
- Touche : runtime
- Correction proposée : sortir `live` et `pending` de `run_batch` (champs du `Worker`, ou `&mut` passés par l'appelant) ; après `catch_unwind`, leur envoyer `Event::Error` et décrémenter `queue` pour les jobs non pris. Côté client, traiter une fin de flux sans `done` comme une erreur.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : test CPU dans `tests/batch_server.rs` : un `BatchDevice` qui panique au 5e appel de `attend`, deux flux ouverts ; exiger que la dernière ligne de chacun porte `error`. Aujourd'hui elle porte un `message` sans `done`.

### CON-07 Après un pas groupé raté ou un moteur en échec, le modèle reste en mémoire hors keep-alive
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-runtime/src/engine.rs:843-851, :867, :780-790, :1300-1314, :1501, :1514-1519 ; llvq-runtime/src/api.rs:520-522, :535
- Fait : 1) Sur un pas raté, `run_batch` répond et libère chaque slot, mais ne pose pas `expires`, contrairement à la fin normale (engine.rs:867). Chaque `submit` l'a mis à `None` (engine.rs:1501), qui veut dire « jamais ». 2) Un moteur `Failed` après un pool refusé (engine.rs:780-790) ou un panic garde son `Worker`, poids et 4E sur la carte, tant que l'`Engine` existe. Le faucheur ne retire que `expired || retiring` (engine.rs:1517-1518), jamais `Failed`. `/api/ps` publie `size_vram` 0 pour un moteur `failed` (api.rs:520-522, :535).
- Scénario : `LLVQ_BATCH=128:2048` sur le 14B : `check_free` refuse le pool, l'état passe `failed`, et les 5 Go de poids restent sur la carte jusqu'à la requête suivante, comptés 0 par `/api/ps`. Autre cas : un pas qui échoue le soir laisse le modèle chargé toute la nuit malgré `keep_alive: 5m`.
- Touche : runtime
- Correction proposée : poser `expires` pour chaque requête vidée par un pas raté ; lâcher le `Worker` avant la boucle qui répond les erreurs d'un moteur `Failed` ; compter les poids dans `size_vram` tant que le fil les tient.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : test CPU dans `tests/batch_server.rs` : un `BatchDevice` qui rend `Err` à un appel donné, une requête avec `keep_alive: "1s"`, attendre 3 s : `/api/ps` doit être vide ; il liste encore le modèle aujourd'hui.

### CON-08 Un `pull` sans effet d'un modèle chargé le décharge, et fait répondre 503 tant qu'il sert
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-runtime/src/store.rs:48, :275-281, :300-306 ; llvq-runtime/src/engine.rs:1457, :1466-1490
- Fait : `register` réécrit le manifeste avec un nouveau `modified_at` (store.rs:48, :277), même quand le blob était déjà là et vérifié (store.rs:300-306). `Models::submit` compare le manifeste entier, `modified_at` compris (engine.rs:1457). Un manifeste « nouveau » retire le moteur chargé et en charge un autre.
- Scénario : 4B chargé, une réponse longue en cours, plafond 1. L'opérateur lance `llvq pull qwen3:4b` pour vérifier son store. Requête suivante : le moteur occupé passe `retiring`, la boucle du plafond ne trouve aucun moteur libre et répond 503 (engine.rs:1481-1489). Sous `LLVQ_BATCH`, les 503 durent jusqu'à ce que le pool se vide. Puis le modèle est rechargé de zéro : 55,2 s depuis le cache, 509,2 s à froid pour la config chaîne (library.rs:82-84).
- Touche : runtime
- Correction proposée : identifier le moteur par `sha256` et `configs`, pas par le manifeste entier ; ou garder l'ancien `modified_at` quand `register` réécrit un manifeste identique au reste près.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : test CPU dans `tests/server.rs` avec `CountingLoader` : une réponse, `store.write_manifest` du même manifeste avec un autre `modified_at`, une seconde réponse : `loads` vaut 2 aujourd'hui.

### CON-09 Ctrl-C n'arrête pas `llvq serve` tant qu'un flux ou un pull tourne, et un second Ctrl-C est ignoré
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-runtime/src/api.rs:1059-1063, :636-643 ; llvq-runtime/src/main.rs:485-499
- Fait : l'arrêt passe par `with_graceful_shutdown(ctrl_c)`, qui attend la fin de toutes les connexions ouvertes. `tokio::signal::ctrl_c` remplace le comportement par défaut de SIGINT pour toute la vie du processus. À la sortie de `block_on`, le `Drop` du runtime attend les tâches `spawn_blocking`, donc un pull en cours.
- Scénario : une réponse de 8 000 jetons sur le CPU (8,6 tok/s au 4B, library.rs:259-260, environ 15 min), un client figé de CON-01, ou un pull de 9 Go. Ctrl-C ne rend pas la main ; un second Ctrl-C non plus. Il faut `kill -9`.
- Touche : runtime
- Correction proposée : au second Ctrl-C, `std::process::exit` ; ou borner l'attente (par exemple 10 s) puis `rt.shutdown_timeout`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : à la main : `llvq serve`, une requête `num_predict` 4000 lue par un client figé, Ctrl-C deux fois : le processus reste.

### CON-10 `delete` pendant un `pull` ou un `import` du même blob enregistre un manifeste sans fichier
- Gravité : mineure
- Confiance : PLAUSIBLE
- Où : llvq-runtime/src/store.rs:228-243, :298-306, :409-416, :246-251
- Fait : `pull` et `import` tiennent le verrou de fichier du blob ; `remove` ne le prend pas. `remove` efface le manifeste, liste les autres, puis efface le blob si aucun ne le nomme.
- Scénario : fil P (`llvq pull qwen3:4b`, blob présent) entre dans `verify`, qui hache 9 Go en une dizaine de secondes (store.rs:301). Fil D (`llvq rm qwen3:4b`) efface manifeste et blob (store.rs:231, :236). P lit l'inode encore ouvert, la vérification passe, et `register` écrit le manifeste (store.rs:305). `llvq ls` montre le modèle ; son chargement échoue sur un fichier absent. Même course entre `copy` (store.rs:246-251) et `remove` de la source.
- Touche : runtime
- Correction proposée : prendre `lock_blob` dans `remove` autour de l'effacement du blob ; dans `pull` et `import`, revérifier que le blob existe juste avant `register`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : test CPU dans `tests/pull.rs` : blob de 200 Mo importé ; un second `import` du même fichier dans un fil, `remove` dans un autre pendant `verify` ; exiger qu'aucun manifeste ne nomme un blob absent. Dépend du temps : répéter 20 fois.

### CON-11 `prompt_eval_cached_count` donne à un client la longueur du préfixe commun avec la conversation d'un autre
- Gravité : mineure
- Confiance : PLAUSIBLE (seulement si plusieurs utilisateurs partagent un serveur)
- Où : llvq-runtime/src/engine.rs:657-658, :735 ; llvq-runtime/src/api.rs:444
- Fait : le moteur garde un seul cache, celui de la dernière requête, quel que soit le client. `reuse_prefix` rend la longueur exacte du préfixe commun en ids, publiée telle quelle dans `prompt_eval_cached_count`, et visible aussi dans `prompt_eval_duration`.
- Scénario : serveur lié au réseau local (`LLVQ_HOST=0.0.0.0`, que `check_origin` accepte pour une IP). A envoie une conversation avec un message système privé. B envoie `<|im_start|>system\n` suivi d'une supposition : `prompt_eval_cached_count` dit combien d'ids de la supposition sont justes. A renvoie tout son historique à chaque tour, donc B peut sonder une fois par tour de A. Ollama a la même propriété.
- Touche : runtime
- Correction proposée : décision d'opérateur. Au minimum documenter le serveur comme mono-utilisateur ; sinon ne publier `cached` que si le memo de la conversation est en accord avec la requête.
- Effort : S (doc) à M
- Risque de la correction : nul.
- Vérification proposée : test CPU dans `tests/server.rs` : A envoie « system: the cat sat on the mat », puis B « system: the cat dog » ; le `prompt_eval_cached_count` de B vaut le nombre d'ids communs.

## Vérifié sain

- Aucun mélange de KV entre requêtes : `caches_to_slot` réécrit tout le slot avec un remplissage de zéros (batch.rs:427-431), `release` remet `len` à 0 (batch.rs:266-271), `batch_logits_slots` refuse un slot nommé deux fois (model.rs:3293-3300), et aucun chemin ne rend un slot encore dans `live` (engine.rs:847-850, :861-863, :1157, :1176, :978).
- La bascule 4E vers pool et retour tient l'invariant « caches = prompt + dit - 1 » : 4E annulé ne nourrit pas son dernier jeton (session.rs:265-268), `resume` le vérifie par `from_caches` (session.rs:93-96).
- Aucun `Mutex` ne peut être empoisonné par le code lu : aucune section sous verrou ne panique, et tous les panics du fil moteur passent par `catch_unwind`, sauf le chargement (CON-03). Les atomiques sont `SeqCst` partout ; le seul `Relaxed` est le compteur d'ids d'OpenAI (openai.rs:66), sans effet.
- `take_job` pose `busy` avant de décrémenter `queue` (engine.rs:292-295), y compris pour un `Load` reçu pendant un décode seul (engine.rs:1051).

## Candidats au lot 1

CON-02, CON-03, CON-06, CON-07, CON-08, CON-09, CON-10
