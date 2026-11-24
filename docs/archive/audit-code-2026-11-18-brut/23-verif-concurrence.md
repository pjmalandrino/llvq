# Vérification contradictoire : concurrence et cycle de vie du runtime (11-concurrence.md)

Vérifié à `925930d`, en lecture seule. Lus en entier : `llvq-runtime/src/engine.rs`, `api.rs`, `chat.rs`, `store.rs`, `client.rs` (l. 90-127), `main.rs` (l. 150-300, 380-527), `tests/batch_server.rs`, `tests/engine_busy.rs`, `tests/common/mod.rs`, `tests/server.rs` (l. 100-556). Lus en partie : `openai.rs` (l. 300-370), `library.rs` (l. 70-100, 250-265), `library.json`, `chat.html` (l. 585-650), `llvq-llm/src/session.rs` (l. 80-190), `batch.rs` (l. 185-420), `chatfmt.rs` (`turn`, `open_assistant`), `docs/runtime-audit-2026-11.md` (« Found and not fixed »), et le code d'axum 0.8.9 dans le registre cargo (`src/serve/mod.rs`, arrêt gracieux).

| ID | Verdict | Gravité retenue | Raison en une ligne |
|---|---|---|---|
| CON-01 | CONFIRMÉ | majeure, sous `LLVQ_BATCH` seulement | `advance` fait `blocking_send` sans délai sur un canal de 64 ; rien ne détecte un client qui ne lit plus ; axum n'a pas de délai d'écriture. Le cas batch 1 est le CONNU de runtime-audit-2026-11. |
| CON-04 | NUANCÉ | mineure | Le fait est vrai, mais une erreur CUDA collante empoisonne le contexte du processus : passer à `Failed` et recharger ne guérit rien. Le test proposé ne déclenche rien sans deux requêtes qui se recouvrent. |
| CON-05 | NUANCÉ | mineure | Vrai : deux rendus différents du même historique. C'est le choix de conception écrit en tête de `chat.rs`, pas une course ; B ne lit rien chez A, il change seulement les ids du prompt de A. Le test ne donne pas toujours « 4 ». |
| CON-06 | CONFIRMÉ | mineure (`LLVQ_BATCH` et un panic) | `live` et `pending` sont des locales détruites au déroulement ; le flux NDJSON finit sans `done` ni `error`, le SSE sans `[DONE]`, et `Client::stream` rend `Ok`. |
| CON-07 | CONFIRMÉ | mineure | Le pas raté ne pose pas `expires` (reste `None` = jamais) ; un moteur `Failed` garde son `Worker` jusqu'à la requête suivante, compté 0 par `/api/ps`. Le test proposé ne déclenche rien avec une seule requête. |
| CON-08 | CONFIRMÉ | mineure | `register` réécrit `modified_at`, `Models::submit` compare le manifeste entier : un `pull` sans effet décharge et recharge, ou rend 503 si le moteur sert. |
| CON-09 | CONFIRMÉ | mineure | `with_graceful_shutdown` d'axum 0.8.9 attend toutes les connexions sans délai ; `ctrl_c` de tokio remplace SIGINT pour toute la vie du processus ; le `Drop` du runtime attend les `spawn_blocking`. « 9 Go » est faux : 5,09 Go au plus. |
| CON-10 | CONFIRMÉ (le relecteur disait PLAUSIBLE) | mineure | `remove` et `copy` ne prennent pas le verrou du blob ; l'entrelacement est possible et se rend déterministe par le rappel `report`. La correction proposée bloquerait un fil tokio pendant tout un pull. |
| CON-11 | NUANCÉ | mineure, hors configuration par défaut | Le fait est vrai, mais il faut `LLVQ_HOST=0.0.0.0` et plusieurs utilisateurs ; sur 127.0.0.1 à un seul utilisateur, personne ne sonde. Même propriété qu'Ollama. |

## Modèle de threads : vérifié

1. Un fil OS par moteur : `std::thread::Builder::new().name(format!("llvq-engine-{name}"))`, engine.rs:1267. Il possède le `Worker` (modèle, décodeur, session, memo, pool). Exact.
2. Handlers sur un runtime tokio multi-thread (main.rs:485). `Models::submit` est synchrone et prend `engines` (engine.rs:1446) ; le manifeste est lu sur disque avant le verrou (engine.rs:1440-1444). Canal d'événements `tokio::sync::mpsc::channel(64)` (engine.rs:1499). Exact. Le chemin « unload » utilise un canal de 4 rempli par `try_send` (engine.rs:1450-1452).
3. Jobs par `std::sync::mpsc::channel` non borné (engine.rs:1265). Le fil moteur répond par `blocking_send` partout (engine.rs:537, 787, 848, 901, 1012, 1104, 1112-1113, 1130, 1197, 1202, 1220, 1287, 1312, 1332, 1336). Aucun `try_send` ni délai côté moteur. Exact.
4. Batch 1 : `while let Ok(job) = rx.recv()` (engine.rs:1317), un job à la fois, session et memo gardés. Exact.
5. `LLVQ_BATCH` : le même fil fait admission, préremplissage, pas groupé, bascule solo et pool (engine.rs:778-879). Exact.
6. Le faucheur est une tâche tokio à 1 s (api.rs:1052-1058). `queue` et `busy` sont `SeqCst`. Précision : les handlers écrivent aussi (`queue.fetch_add` engine.rs:1350, `expires = None` engine.rs:1501, `retiring` engine.rs:1380 et 1460), pas seulement le fil moteur.
7. Ordre `engines` puis `state` ou `expires` ; le fil moteur ne prend jamais `engines` et ne tient jamais deux verrous à la fois. Aucun interblocage. Exact.
8. Pulls dans `spawn_blocking` (api.rs:636-643), sérialisés par `lock_blob` (store.rs:159-172). Précision : `remove` (store.rs:228-243) et `copy` (store.rs:246-251) ne prennent pas ce verrou, et le handler `delete` les appelle directement sur un fil tokio (api.rs:671-672). `llvq pull --from` importe dans le processus client, pas dans le serveur (main.rs:155-157).
9 et 10. Exact : ce qui bloque le fil moteur bloque tout ce que ce moteur sert.

Rien dans le modèle ne contredit un constat. Les deux précisions servent CON-10.

## CON-01 Sous `LLVQ_BATCH`, un client qui cesse de lire gèle toutes les requêtes du pool

Verdict : CONFIRMÉ. Gravité retenue : majeure, sous `LLVQ_BATCH` seulement.

Preuve, ligne à ligne :
- engine.rs:1499 : `tokio::sync::mpsc::channel(64)` par requête.
- engine.rs:855-860 : la boucle appelle `advance` pour chaque `Live`, dans l'ordre, sur le seul fil moteur. Le pas suivant (engine.rs:842) n'a lieu qu'après elle.
- engine.rs:1201-1202 : `l.job.events.blocking_send(Event::Delta {..})`. `blocking_send` attend une place sans délai. Il ne rend `Err` que si le `Receiver` est détruit.
- Côté HTTP : `events` (api.rs:180-189) ne lit le canal que lorsque hyper tire le corps. Un client qui ne lit plus remplit le tampon TCP, hyper cesse de tirer, le canal se remplit. axum 0.8.9 `serve` n'a aucun délai d'écriture ; la connexion reste ouverte tant que le client la garde.
- Aucune garde ailleurs : pas de `try_send`, pas de contrôle `is_closed` sinon à l'admission (engine.rs:898, 1127), qui ne voit pas un client vivant et lent.

Entrelacement : fil moteur dans `advance` pour X, bloqué dans `blocking_send` ; les autres `Live` du vecteur attendent ce retour ; `rx.try_recv` (engine.rs:807) n'est plus atteint, la file grossit. Possible, rien ne l'empêche.

Ce qui est déjà connu : runtime-audit-2026-11, « Found and not fixed », décrit le même blocage sur le chemin par défaut (file derrière le client figé). La part nouvelle est réelle : sous `LLVQ_BATCH`, des réponses déjà commencées pour d'autres clients s'arrêtent net. Le délai avant le gel inclut les tampons de socket : quelques milliers de jetons en HTTP, 63 événements exactement au niveau de `Models`.

Gravité : majeure tient pour le mode `LLVQ_BATCH`, qui est par construction multi-client. Ce mode n'est pas servi par défaut. Sur le chemin par défaut, le défaut est le CONNU, pas ce constat.

Test proposé : il montrerait bien le défaut, à deux conditions. B doit d'abord décoder seul (chemin solo), A arrive et B passe dans le pool : c'est le schéma de `start_long`. Il faut lire B avec un délai (`recv_timeout` ou `tokio::time::timeout`), sinon le test pend au lieu d'échouer, et détruire le `Receiver` de A à la fin pour libérer le fil moteur. Les jetons du modèle minuscule sont des mots, donc chaque pas produit un `Delta` non vide : A se remplit en 63 pas.

Test existant le plus proche :
`cargo test -p llvq-runtime --test batch_server a_lone_request_runs_on_the_batch_1_path_and_joins_the_pool_when_another_arrives`

Correction sûre pour un lot 1 ? non. Le délai au bout duquel un client lent est déclaré parti est un choix d'opérateur (il coupe aussi un client lent légitime), la règle doit aller à trois endroits (`advance`, `solo_decode`, `run`), et l'effort est M.

## CON-04 Une erreur de carte qui persiste laisse le moteur « ready »

Verdict : NUANCÉ. Gravité retenue : mineure.

Ce qui est vrai :
- engine.rs:539-541 : une `Err` de `run_inner` part au client, l'état reste `Ready`.
- engine.rs:843-851 : un pas groupé raté répond 500 à chaque `Live` et continue.
- engine.rs:1147-1153 : un `admit` raté répond 500.
- engine.rs:1456 : seul `Failed` est oublié par `Models::submit`.
- batch.rs:296 : `if bytes > free` accepte un pool égal à la mémoire libre ; le commentaire batch.rs:199-203 dit qu'un préremplissage échoue alors.

Ce qui est faux ou exagéré :
- Scénario 1 : une erreur collante du pilote (accès illégal) empoisonne le contexte primaire du processus. Passer à `Failed` ferait recharger le modèle dans le même processus, qui échouerait aussi. La correction ne gagne qu'un message plus clair (« restart the server »). « Le keep-alive ne décharge jamais le moteur sous trafic » est le comportement voulu, pas un défaut.
- Le test proposé ne déclenche rien tel qu'écrit. Une requête seule passe par le chemin solo (`Session::feed`, `Session::answer`, engine.rs:917-943), et le préremplissage d'un slot passe par `hidden_cached` (batch.rs:377-393) : ni l'un ni l'autre n'appelle `BatchDevice::attend`. Seul `Pool::step` l'appelle. Il faut deux requêtes qui se recouvrent (`start_long` puis une seconde). Et les deux « 500 » ne sont des statuts HTTP qu'avec `"stream": false` ; en flux, le statut est 200 et l'erreur arrive en ligne `error`.

Test existant le plus proche :
`cargo test -p llvq-runtime --test batch_server a_lone_request_runs_on_the_batch_1_path_and_joins_the_pool_when_another_arrives`

Correction sûre pour un lot 1 ? non. Reconnaître une erreur collante passe par le texte des erreurs du pilote, sans test CPU possible ; la marge de `check_free` demande de chiffrer le pic d'un préremplissage, et ne vaut que sous `LLVQ_BATCH`.

## CON-05 La même requête de chat reçoit un prompt différent selon qu'un autre client a parlé

Verdict : NUANCÉ. Gravité retenue : mineure.

Ce qui est vrai :
- engine.rs:760-763 : le memo garde pour le message assistant `session.ids()[messages_end..]`, donc l'ouverture d'`open_assistant` avec le bloc de raisonnement vide (4 ids quand think est off, chatfmt.rs `open_assistant`), les ids générés, `<|im_end|>\n`.
- chat.rs:66-68 : ces ids sont réutilisés tant que rôle et texte concordent.
- Sinon `turn` (chatfmt.rs) rend `<|im_start|>assistant\n{texte}<|im_end|>\n`, sans bloc de raisonnement, par réencodage.
- engine.rs:663 : toute requête acceptée remplace le memo. Une requête `Raw` le remplace aussi par un segment sans message.
- Sous `LLVQ_BATCH`, `prepare(&job.request, &[])` (engine.rs:906, 1135) : toujours le rendu texte.
- Avec think on, le memo montre au modèle son raisonnement précédent ; le client ne renvoie que `content` (le memo est indexé sur `router.content.sent`, engine.rs:761), donc la concordance tient.

Ce qui est nuancé :
- Ce n'est pas une course : un seul fil moteur, l'ordre des requêtes décide. La tête de `chat.rs` (l. 3-12) écrit déjà que le rendu texte ne donne pas les ids du cache. Le défaut est la conséquence connue d'un choix, pas un oubli.
- Ce que B change chez A : les ids du prompt du tour suivant de A (au moins 4 de moins), donc peut-être la réponse gloutonne de A, son `prompt_eval_count`, et le coût d'une relecture complète. Ce que B apprend de A : rien par ce mécanisme (voir CON-11 pour la longueur du préfixe).
- Réalisme sur 127.0.0.1 à un seul utilisateur : réel. Le même utilisateur peut avoir la page de chat et `llvq run` ouverts, et beaucoup de façades Ollama envoient une requête annexe entre deux tours (génération de titre). Une seule requête intercalée suffit.
- Le test proposé montrerait une différence, pas forcément « 4 ». Le tokenizer de test est à mots (`tests/common/mod.rs:33-51`), le décodage sans décodeur rejoint les jetons par des espaces, et les jetons spéciaux générés (`\n`, `<think>`) peuvent se réencoder autrement. Exiger l'inégalité, pas un écart de 4.

Test existant le plus proche :
`cargo test -p llvq-runtime --test server a_stateless_second_turn_reads_only_its_new_tokens_and_answers_as_the_stateful_run`

Correction sûre pour un lot 1 ? non. C'est une décision d'opérateur, et les deux options changent les prompts multi-tours servis.

## CON-06 Un panic du moteur batch coupe les flux en cours sans `done` ni erreur

Verdict : CONFIRMÉ. Gravité retenue : mineure (sous `LLVQ_BATCH`, et il faut un panic).

Preuve :
- engine.rs:792-793 : `pending` et `live` sont des locales de `run_batch`. Un panic les détruit au déroulement ; leurs `Sender` tombent sans dernier événement. Même chose pour le `Live` local de `solo` ou `solo_decode`.
- engine.rs:1300-1314 : après `catch_unwind`, seuls les jobs encore dans `rx` reçoivent l'erreur.
- engine.rs:1323-1332 : le chemin batch 1 garde le job par référence hors de la fermeture et lui envoie l'erreur. L'asymétrie est réelle.
- api.rs:185 : `rx.recv().await?` rend `None`, le flux NDJSON finit, statut 200 déjà envoyé, pas de ligne `done` ni `error`.
- openai.rs:325-361 : pas de `data: [DONE]`.
- client.rs:115-126 : la boucle finit, `Ok(())`. `llvq run` affiche la réponse partielle comme complète.
- Sans flux (`"stream": false`), `answer` rend bien 500 « the engine stopped before the end of the answer » (api.rs:769). Le constat ne vaut que pour les flux.
- Les jobs de `pending` ont été comptés dans `queue` (engine.rs:1350) sans `take_job` : `queue` reste haut jusqu'à ce que `Models::submit` oublie le moteur `Failed` (engine.rs:1456). Exact.

Test proposé : il montrerait le défaut si les deux flux se recouvrent (le pas groupé seul appelle `attend`). Avec deux couches, le 5e appel tombe au 3e pas. `chat_telling` rendrait le contenu partiel et un dernier `Value` sans `done` : exiger `error` sur la dernière ligne échoue aujourd'hui.

Test existant le plus proche :
`cargo test -p llvq-runtime --test batch_server a_lone_request_runs_on_the_batch_1_path_and_joins_the_pool_when_another_arrives`

Correction sûre pour un lot 1 ? oui pour le côté moteur : passer `live` et `pending` en `&mut` depuis la fermeture du fil, puis leur envoyer `Event::Error` et décrémenter `queue` après `catch_unwind`. Local à engine.rs, testable sur CPU par un `BatchDevice` qui panique, sans effet sur les jetons ni sur le protocole (une ligne `error` est déjà la forme d'erreur en flux). Le volet client (« fin sans `done` = erreur ») peut suivre à part.

## CON-07 Après un pas groupé raté ou un moteur en échec, le modèle reste en mémoire hors keep-alive

Verdict : CONFIRMÉ. Gravité retenue : mineure.

Preuve, partie 1 :
- engine.rs:1501 : chaque `submit` pose `expires = None`, qui veut dire « jamais ».
- engine.rs:843-851 : le pas raté répond et libère, puis `continue`, sans toucher `expires`. Comparer la fin normale (engine.rs:867), `solo` (engine.rs:824), `admit` (engine.rs:834), `resume` (engine.rs:874).
- engine.rs:796-804 : la boucle repasse `busy` à faux et attend. `reap` (engine.rs:1517-1518) ne retire que `expired || retiring`. `expires` reste `None` : le moteur reste chargé jusqu'à la requête suivante.

Preuve, partie 2 :
- engine.rs:780-790 : pool refusé, `Failed`, puis la boucle `rx.recv` tourne dans `run_batch(&mut self)` : le `Worker`, donc poids et 4E, reste vivant.
- engine.rs:1300-1314 et 1323-1338 : après un panic, `w` vit jusqu'à la fin de la fermeture du fil, pendant toute la boucle d'erreurs. Le cas batch 1 est sur le chemin par défaut.
- Le moteur `Failed` n'est retiré que par `Models::submit` (engine.rs:1456). Sans requête suivante, il reste.
- api.rs:520-522 : `Failed` donne `backend = None`, donc `on_device` faux et `size_vram` 0 (api.rs:535).

Nuance : avec un seul utilisateur, la requête suivante arrive vite et libère tout ; le scénario « toute la nuit » demande qu'il n'y en ait plus. Le test proposé ne déclenche rien avec une seule requête : elle passe par le chemin solo, qui n'appelle pas `attend` (voir CON-04). Il faut deux requêtes qui se recouvrent, toutes deux avec `keep_alive: "1s"`.

Tests existants les plus proches :
`cargo test -p llvq-runtime --test server the_store_routes_list_show_copy_delete_and_ps_follows_loads` (la partie keep-alive expiré) et
`cargo test -p llvq-runtime --test batch_server a_lone_request_runs_on_the_batch_1_path_and_joins_the_pool_when_another_arrives`

Correction sûre pour un lot 1 ? oui pour la partie 1 seulement : poser `expires` pour chaque `Live` vidé par un pas raté est une ligne dans engine.rs:847-850, testable sur CPU, sans effet sur les jetons. La partie 2 (lâcher le `Worker` avant la boucle d'erreurs) demande de restructurer la fermeture du fil, et le `size_vram` d'un moteur `Failed` change ce que `/api/ps` publie : à part.

## CON-08 Un `pull` sans effet d'un modèle chargé le décharge, et fait répondre 503 tant qu'il sert

Verdict : CONFIRMÉ. Gravité retenue : mineure.

Preuve :
- store.rs:48 : `Manifest::of` pose `modified_at: now_secs()`.
- store.rs:300-306 : blob présent et vérifié, puis `register` (store.rs:275-281) réécrit le manifeste avec la nouvelle date. Même chose dans `import` (store.rs:411-416).
- engine.rs:1457 : `e.info.manifest == manifest`, égalité du manifeste entier (`#[derive(PartialEq)]`, store.rs:19), `modified_at` compris.
- engine.rs:1466 : `retire` du moteur du même nom. Libre : il est retiré et un autre se charge. Occupé : `retiring`, il compte toujours, la boucle du plafond ne trouve aucun moteur `idle` et rend 503 (engine.rs:1481-1489).
- Le rechargement coûte 55,2 s depuis le cache sur la config chaîne (library.rs:80-84). Exact.

Réalisme : `llvq pull` d'un modèle déjà présent est un geste courant, et `llvq pull --from` passe par `import`, qui a le même effet. La page de chat ne propose « Get » que pour un modèle absent, donc elle ne le déclenche pas.

Test proposé : il montre le défaut. `common::store` écrit le manifeste ; réécrire le même avec un autre `modified_at`, puis une seconde requête : `loads` vaut 2 aujourd'hui.

Test existant le plus proche :
`cargo test -p llvq-runtime --test server stopping_a_busy_model_keeps_one_copy_until_its_answer_ends`

Correction sûre pour un lot 1 ? oui : comparer les manifestes en ignorant `modified_at` (ou garder l'ancienne date dans `register` quand le reste est identique). Local, testable sur CPU avec `CountingLoader`, sans effet sur les jetons. Un manifeste d'une autre révision recharge toujours.

## CON-09 Ctrl-C n'arrête pas `llvq serve` tant qu'un flux ou un pull tourne

Verdict : CONFIRMÉ. Gravité retenue : mineure.

Preuve :
- api.rs:1059-1063 : `with_graceful_shutdown(ctrl_c)`.
- axum 0.8.9, `src/serve/mod.rs` l. 278-303 : au signal, chaque connexion reçoit `graceful_shutdown` (l. 409), puis `close_tx.closed().await` (l. 303) attend toutes les connexions, sans délai. Une réponse HTTP/1 en cours va à son terme.
- `tokio::signal::ctrl_c` : la documentation de tokio dit que le comportement par défaut de SIGINT n'est jamais rétabli une fois le gestionnaire installé. Le second Ctrl-C est capté et ignoré.
- main.rs:485-499 : `rt` tombe à la fin de `serve()` ; le `Drop` d'un runtime tokio attend les tâches `spawn_blocking` en cours, donc un pull (api.rs:636-643), qui continue même quand son récepteur est parti (`let _ = tx.blocking_send`).
- Le fil moteur est un fil std non joint : il ne retient pas la sortie.

Correction de détail : « un pull de 9 Go » est faux. Le plus gros blob de `library.json` pèse 5 087 000 541 octets (14B), le 4B 1 418 224 685.

Test proposé : à la main, juste. Aucun test existant ne couvre l'arrêt ; pas de commande de test à donner.

Correction sûre pour un lot 1 ? non. Le comportement n'est pas testable dans le style de `llvq-runtime/tests` sans faire prendre à `api::serve` un futur d'arrêt en paramètre, et le choix entre « second Ctrl-C sort » et « attente bornée » est d'opérateur.

## CON-10 `delete` pendant un `pull` ou un `import` du même blob enregistre un manifeste sans fichier

Verdict : CONFIRMÉ (le relecteur disait PLAUSIBLE). Gravité retenue : mineure.

Preuve :
- store.rs:298 et 409 : `pull` et `import` prennent `lock_blob`.
- store.rs:228-243 : `remove` ne le prend pas. Il efface le manifeste (l. 231), liste les autres (l. 235), efface le blob si aucun ne le nomme (l. 236).
- store.rs:300-305 et 411-416 : blob présent, `verify` (hachage complet, store.rs:264), puis `register`.

Entrelacement possible, et déterministe : `register` appelle `report(Status("writing manifest"))` (store.rs:276) avant `write_manifest` (store.rs:278). Un test peut appeler `s.remove(&name)` dans ce rappel : le manifeste et le blob partent, puis `register` écrit un manifeste qui nomme un blob absent. Pas besoin de 200 Mo ni de 20 répétitions. Même course entre `copy` (lit la source l. 247, écrit l. 249) et `remove` de la source.

Conséquence : `llvq ls` montre le modèle ; son chargement échoue, le moteur passe `Failed`, la requête suivante réessaie et échoue encore. Un nouveau `pull` répare (blob absent, téléchargement). Mineure tient.

Correction de détail : « hache 9 Go en une dizaine de secondes » est faux, 5,09 Go au plus ; la fenêtre est de quelques secondes.

Défaut de la correction proposée : `lock_blob` appelle `f.lock()`, bloquant. Le handler `delete` appelle `store.remove` directement sur un fil tokio (api.rs:671-672). Pendant un pull de plusieurs minutes, ce fil resterait bloqué. Il faut `try_lock` et un refus (409 « a pull of this file is running »), ou `spawn_blocking`.

Test existant le plus proche :
`cargo test -p llvq-runtime --test pull import_checks_a_local_copy_and_rm_keeps_a_blob_another_name_holds`

Correction sûre pour un lot 1 ? non telle que proposée (elle bloque un fil du runtime async). Oui si `remove` prend le verrou par `try_lock` et refuse quand un pull le tient : local à store.rs et api.rs, testable par le rappel `report`, sans effet sur les jetons ; le refus est un nouveau statut d'erreur, dans la forme `{"error": ...}` qu'Ollama emploie déjà.

## CON-11 `prompt_eval_cached_count` donne à un client la longueur du préfixe commun avec la conversation d'un autre

Verdict : NUANCÉ. Gravité retenue : mineure, et seulement hors configuration par défaut.

Ce qui est vrai :
- session.rs:143-147 : `reuse_prefix` rend `common_prefix(&self.ids, prompt).min(prompt.len() - 1)`, la longueur exacte en ids.
- engine.rs:657 et 735, api.rs:444 : publiée telle quelle. `prompt_eval_duration` porte la même information.
- Le cache est unique par moteur, sans notion de client.

Ce que B apprend de A : le nombre d'ids de tête où sa supposition concorde avec le dernier prompt servi (par exemple le message système de A). Ce que B change chez A : la requête de B remplace le cache, donc le tour suivant de A relit tout (voir CON-05). D'où une seule sonde utile par tour de A, et une supposition à confirmer, pas un texte à extraire : deviner l'id suivant parmi environ 151 000 demande un tour de A par essai.

Réalisme : le serveur écoute sur `127.0.0.1:11435` par défaut (client.rs:10). `check_origin` (api.rs:1026-1047) refuse les pages d'autres sites. Sur un poste à un seul utilisateur, il n'y a personne pour sonder. Il faut `LLVQ_HOST=0.0.0.0` et plusieurs utilisateurs sur le réseau local. Ollama a la même propriété.

Test proposé : il montre le fait. Le tokenizer de test est à mots, donc « system: the cat sat on the mat » puis « the cat dog » donnent un préfixe commun calculable.

Test existant le plus proche :
`cargo test -p llvq-runtime --test server a_stateless_second_turn_reads_only_its_new_tokens_and_answers_as_the_stateful_run`

Correction sûre pour un lot 1 ? non. C'est une décision d'opérateur (documenter le serveur comme mono-utilisateur, ou masquer `cached`), et masquer le champ change ce que les clients Ollama lisent.

## Bilan pour le lot 1

Sûrs, au sens strict de la consigne : CON-06 (côté moteur), CON-07 (partie 1 seulement), CON-08. CON-10 seulement avec `try_lock` et un refus, pas avec la correction proposée. La liste du relecteur (CON-02, CON-03, CON-06, CON-07, CON-08, CON-09, CON-10) est trop large ici : CON-09 n'a pas de test CPU dans le style du dépôt, CON-10 bloquerait un fil tokio tel qu'écrit. CON-02 et CON-03 ne sont pas jugés dans ce fichier.

Trois tests proposés par le relecteur ne déclenchent rien tels qu'écrits (CON-04, CON-07, et le chiffre exact de CON-05) : sous `LLVQ_BATCH`, une requête seule ne passe jamais par `BatchDevice::attend`.
