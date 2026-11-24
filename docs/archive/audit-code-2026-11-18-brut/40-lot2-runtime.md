# Lot 2, le runtime (tenue du service en cas d'erreur) : journal

Partie de l'arbre du lot 1 appliqué, non commité. Aucune commande de compilation ni de test lancée : tout est relu, rien
n'est compilé. Fichiers touchés : `llvq-runtime/src/engine.rs`, `src/store.rs`, `src/api.rs` ; tests
`llvq-runtime/tests/engine_busy.rs`, `server.rs`, `batch_server.rs`, `pull.rs`. Aucun autre fichier.

## Modèle de threads, relu avant d'écrire

- Un fil OS par moteur (`llvq-engine-<nom>`). Il possède le `Worker` : modèle, décodeur, session, memo, pool.
- Les handlers sont des tâches tokio. `Models::submit` est synchrone et tient `engines` ; il prend ensuite `state` ou
  `expires`, jamais l'inverse. Le fil moteur ne prend jamais `engines` et ne tient jamais deux verrous à la fois.
- Jobs : `std::sync::mpsc` non borné, `queue` compté par `submit` (`fetch_add`) et rendu par `take_job` ou par les
  boucles d'erreur (`fetch_sub`). Événements : `tokio::sync::mpsc` de 64, `blocking_send` côté moteur.
- Le faucheur lit `queue`, `busy`, `expires`, `retiring` ; il ne retire qu'un moteur `idle`.
- Les pulls tournent dans `spawn_blocking` sous `lock_blob` (flock) ; `delete` appelle `Store::remove` sur un fil tokio.

## Corrigé

| ID | fichiers | ce qui change, en une ligne | vérification faite |
|---|---|---|---|
| QUA-50 / CON-03 | `engine.rs` (fil du moteur, chargement) | le chargement passe par `catch_unwind` ; une panique devient l'`Err` existant : état `Failed`, file répondue, `queue` remis à 0, et la requête suivante recharge | relu, NON COMPILÉ |
| BUG-50 / CON-02 (moteur) | `engine.rs` (`run_inner`) | sur une `Err` de `feed_with`, `session.reset` et `memo.clear()` avant de rendre l'erreur, comme la fermeture ratée d'un message | relu, NON COMPILÉ |
| CON-06 | `engine.rs` (`run_batch`, fil du moteur) | `pending` et `live` appartiennent à la fermeture du fil et survivent au `catch_unwind` ; chaque `Live` reçoit `Event::Error`, chaque job de `pending` aussi, et sort de `queue` | relu, NON COMPILÉ |
| CON-07 partie 1 | `engine.rs` (`run_batch`, pas raté) | chaque `Live` vidé par un pas raté pose `expires` comme une fin propre | relu, NON COMPILÉ |
| CON-08 | `engine.rs` (`Models::submit`, `same_model`) | un moteur chargé est retrouvé si son manifeste égale le nouveau à `modified_at` près ; une autre révision recharge toujours | relu, NON COMPILÉ |
| CON-10 | `store.rs` (`remove`, `BlobBusy`), `api.rs` (`delete`) | `remove` prend le verrou du blob par `try_lock`, jamais bloquant, et refuse par `BlobBusy` quand un pull ou un import le tient ; `delete` répond 409 | relu, NON COMPILÉ |
| SEC-03 | `api.rs` (`stop_list`, `MAX_STOPS`, `MAX_STOP_BYTES`) | 400 au-delà de 16 chaînes d'arrêt ou de 256 octets par chaîne, sur `/api/*` et `/v1/*` (même fonction) | relu, NON COMPILÉ |

Au passage, dans `engine.rs` : le calcul du message d'une panique, écrit deux fois, est devenu `panic_message`, et le
chargement l'emploie aussi. Même texte produit, aucune chaîne imprimée changée.

Détails qui comptent pour la relecture :
- QUA-50 : le message devient « loading <nom>: the load panicked: <message> » par la branche `Err` existante. Rien ne
  dépend plus de la mort du fil ; la seconde moitié proposée par CON-03 (oublier un moteur dont l'envoi échoue) n'est
  donc plus nécessaire et n'est pas faite.
- BUG-50 : correction du côté moteur seulement, comme demandé. Elle est indépendante de la garde de `Session::truncate`
  que l'autre agent écrit dans `llvq-llm`. Le chemin sans erreur ne bouge pas.
- CON-06 : `pending` contient des jobs comptés dans `queue` et jamais pris (`take_job` n'a lieu qu'au `pop_front`) ; un
  `Live` a déjà été pris. D'où `fetch_sub` pour les premiers seulement.
- CON-08 : `same_model` compare un clone dont `modified_at` est remplacé ; le moteur garde l'ancien manifeste dans
  `EngineInfo`. `/api/ps` ne publie pas `modified_at` (seul `/api/tags`, qui lit le store).
- CON-10 : `try_lock` de `std::fs::File` (flock sous Unix, comme `lock_blob`). Deux ouvertures du même fichier de verrou
  dans un même processus se bloquent l'une l'autre : `two_pulls_of_one_file_at_once_register_the_file_whole` repose
  déjà sur ce fait. `remove` crée `locks/` et le fichier de verrou s'ils manquent, comme `lock_blob`. Le match sur
  `TryLockError` a un bras générique (`other`, formaté en `{:?}`), pour ne pas dépendre de l'exhaustivité de l'enum.
- SEC-03 : `openapi.json` n'est pas touché. Le schéma généré ne décrit pas `stop` en détail (`options` est un objet libre,
  `stop` OpenAI est un `Value`), et aucune annotation `#[utoipa::path]` n'a changé. Aucun script de `ops/` n'envoie de
  liste d'arrêt au serveur (`git grep '"stop"' -- ops/`).

## Laissé de côté

| ID | raison |
|---|---|
| CON-06, chemin seul | Le `Live` local de `solo`, `solo_decode` et `resume`, et le job local d'`admit`, meurent encore sans dernier événement sur une panique. Plan : un `&mut Option<tokio::sync::mpsc::Sender<Event>>` passé à `run_batch`, posé avec un clone de `events` avant chaque appel à `solo`, `resume` et `admit`, remis à `None` après ; le bras de panique y envoie `Event::Error`. Environ 12 lignes, mais aucun test CPU ne l'atteint : le chemin seul n'appelle pas `BatchDevice::attend`, il faudrait un `Loader` dont le modèle panique dans `Session::answer`. À faire avec un compilateur. |
| CON-06, côté client | `Client::stream` rend encore `Ok` sur un flux fini sans `done`. Le vérificateur le met à part. Plan : dans `client.rs`, retenir si une ligne `done: true` est passée et rendre une erreur sinon ; vérifier d'abord que le flux de `/api/pull` (même fonction ?) n'en souffre pas : il n'a pas de `done`. |
| CON-07 partie 2 | Lâcher le `Worker` d'un moteur `Failed` avant la boucle d'erreurs, et compter les poids dans `size_vram` : restructure la fermeture du fil et change ce que `/api/ps` publie. Hors de la forme du vérificateur. |
| CON-10, `copy` | Même course entre `copy` et `remove` de la source (vérificateur). Non demandé. Plan : `copy` prend aussi le verrou par `try_lock` et refuse par `BlobBusy`. |
| CON-10, document OpenAPI | Le 409 n'est pas déclaré dans l'annotation de `delete`, qui ne déclare pas non plus le 500 déjà servi. L'ajouter change `openapi.json`, qui se régénère par `LLVQ_UPDATE_OPENAPI=1 cargo test -p llvq-runtime --test openapi` : à décider avec le reste du document, pas à la main. |
| CON-10, `delete` | Le handler décharge le modèle (`unload`) avant d'appeler `remove`. Un refus 409 laisse donc le modèle déchargé ; la requête suivante le recharge. Inoffensif, non corrigé. |

## Décision d'opérateur

| ID | option recommandée | valeur proposée | coût |
|---|---|---|---|
| CON-01 | `try_send` dans `advance` (pool seulement) avec un horodatage par `Live` : un client dont le canal reste plein au-delà du délai est traité comme parti (`After::Gone`), son slot libéré. Le batch 1 garde `blocking_send` : un client figé n'y bloque que la file, cas déjà connu. | 30 s sans place libre dans le canal de 64 | Effort M, environ 30 lignes dans `engine.rs`, un test CPU dans `batch_server.rs` (un `Receiver` gardé sans lecture au niveau de `Models`, le `Done` de l'autre requête exigé en moins de 40 s ; test long). Risque : un client lent légitime, derrière un proxy, coupé après 30 s sans lecture. |
| CON-04 | Ne pas recharger : une erreur collante du pilote empoisonne le contexte du processus. Passer à `Failed` avec « restart the server » quand le message contient `CUDA_ERROR_ILLEGAL_ADDRESS` ou `CUDA_ERROR_LAUNCH_FAILED`, et ne rien faire d'autre. La marge de `check_free` attend un chiffrage du pic d'un préremplissage. | marge = pic mesuré d'un préremplissage de `context` jetons, à mesurer sur carte | S pour le message (non testable sur CPU, la reconnaissance passe par le texte du pilote) ; M et un run sur carte pour la marge. |
| CON-05 | Garder le memo et documenter qu'une conversation n'est déterministe que si personne ne parle entre deux tours. L'autre option (toujours le rendu texte) change les prompts multi-tours servis. | aucune | 0 $ ; une phrase dans `llvq-runtime/README.md`. L'autre option : relecture du dernier message assistant à chaque tour, et une porte qualité multi-tours. |
| CON-09 | Second Ctrl-C : `std::process::exit(130)`. Premier Ctrl-C : arrêt gracieux borné, puis `rt.shutdown_timeout`. | 10 s d'attente, puis 1 s pour les tâches bloquantes | S, environ 15 lignes dans `api::serve` et `main.rs` ; test à la main seulement, sauf à faire prendre à `api::serve` un futur d'arrêt en paramètre. |
| CON-11 | Documenter le serveur comme mono-utilisateur, et l'avertissement au démarrage hors boucle locale (lié à SEC-04). Masquer `prompt_eval_cached_count` changerait ce que lisent les clients Ollama. | aucune | 0 $ ; deux phrases (README, `docs/ENV.md`) et une ligne `eprintln!` dans `serve`. |

## Constat faux ou déjà corrigé

| ID | preuve |
|---|---|
| aucun | Les sept constats se lisent dans le code à `925930d` plus le lot 1 : `loader.load` hors `catch_unwind` (engine.rs, fil du moteur) ; `feed_with(...)?` sans remise à zéro ; `pending` et `live` locales de `run_batch` ; pas de `expires` dans le bras du pas raté ; `e.info.manifest == manifest` sur le manifeste entier et `Manifest::of` qui pose `now_secs()` ; `remove` sans verrou ; `stop_list` sans borne. |

## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)

Aucun fichier sous `cfg(target_os = "linux")`. `engine.rs` contient `batch_engine` sous `cfg(all(target_os = "linux",
feature = "cuda"))`, que je n'ai pas touché ; `run_batch` et le fil du moteur se compilent sur toutes les cibles.
Une compilation croisée de `llvq-runtime --features cuda` reste prudente, puisque `run_batch` a changé de signature.

## Commandes lancées et leur résultat

Aucune. Consigne du lot : pas d'exécution locale. Lecture, `git grep`, `git diff` seulement.

## À lancer quand la machine est libre

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
