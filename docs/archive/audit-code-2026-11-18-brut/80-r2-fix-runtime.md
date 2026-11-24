# Second tour, le runtime : journal

Partie de l'arbre des lots 1 à 7 appliqués, non commités. Aucune commande de compilation ni de test lancée : tout est
relu deux fois, rien n'est compilé. Fichiers touchés : `llvq-runtime/src/store.rs`, `src/engine.rs`, `src/chat.rs`,
`src/openai.rs`, `src/api.rs` ; tests `llvq-runtime/tests/engine_busy.rs`, `batch_server.rs`, `pull.rs`, `server.rs` ;
hors périmètre, la seule ligne autorisée de `50-file-de-verification.md` (ligne 26, mutant du test de panique du batch).

## Corrigé

| ID | fichiers | ce qui change, en une ligne | vérification faite |
|---|---|---|---|
| R2-DIF-01 | `store.rs` (`BlobId`, `Store::blob_id`), `engine.rs` (`EngineInfo::blob`, `same_model`, `Models::submit`, `Engine::spawn`) | un moteur chargé n'est repris que si son manifeste égale le nouveau à `modified_at` près ET si le blob est le même fichier (périphérique, inode, ctime, mtime, taille) qu'au début du chargement | relu, NON COMPILÉ |
| R2-DIF-05 | `engine.rs` (bras de panique du batch) | `try_send` au lieu de `blocking_send` dans les deux boucles neuves (`live` et `pending`) ; la boucle `rx.recv()` d'avant et les autres `blocking_send` sont inchangés | relu, NON COMPILÉ |
| R2-DIF-02 | `tests/batch_server.rs` (`two_in_the_pool`), ligne 26 de `50-file-de-verification.md` | chaque flux tourne sur son fil et rend son résultat par `mpsc` ; le test attend au plus 120 s (`recv_timeout` sur une échéance commune), puis panique « the streams neither ended nor failed within 120 s » | relu, NON COMPILÉ |
| R2-DIF-06 | `engine.rs` (`run_inner`, bras `Err` de `feed_with`) | commentaire seul : `Session::truncate` coupe déjà les caches en avance ; la remise à neuf est une seconde garde, payée d'une relecture de l'historique | relu, NON COMPILÉ |
| R2-API-10 | `chat.rs` (`scan`) | le test des préfixes de chaîne d'arrêt se fait sur `printable(text)`, sans les U+FFFD finaux ; test neuf qui coupe « é » entre deux jetons d'octets | relu, NON COMPILÉ |
| R2-API-11 (listages, `rm`) | `store.rs` (`list`, `manifests` neuf, `remove`), `tests/pull.rs` | `list` saute un manifeste illisible et le nomme sur stderr (`llvq: ...`, la forme des autres journaux du serveur) ; `remove` lit la liste AVANT d'effacer, et garde le blob quand un manifeste illisible pourrait le nommer | relu, NON COMPILÉ |
| R2-API-09 (partie `/v1` et `raw`) | `openai.rs` (`complete`, `HARMLESS`, `warn_extra`), `api.rs` (`generate`), `tests/server.rs` | en flux, `warnings` part sur le morceau qui porte `finish_reason` ; `logprobs` sort de `HARMLESS` et avertit sauf à `false` ; `/api/generate` avertit quand `system` est donné sous `raw` | relu, NON COMPILÉ |
| R2-API-15 (message) | `openai.rs` (`content_text`) | une partie `{"type": "text"}` sans chaîne `text` reçoit « content part "text": no "text" field holding a string » | relu, NON COMPILÉ |

Détails qui comptent pour la relecture :

- R2-DIF-01. J'ai pris la forme du relecteur, élargie. `BlobId` tient la taille, `modified()` (mtime à la nanoseconde
  sous Unix) et, sous `cfg(unix)`, `dev`, `ino`, `ctime`, `ctime_nsec` de `std::os::unix::fs::MetadataExt`. Hors Unix,
  taille et mtime seulement (Windows est hors périmètre, `docs/plan-runtime.md`). Pourquoi pas l'inode seul : ext4
  réattribue volontiers l'inode d'un fichier effacé au suivant. Pourquoi pas la mtime seule : sur macOS,
  `std::fs::copy` (chemin d'`import`) peut cloner le fichier source avec ses dates. Les deux ensemble, plus la ctime
  que nul ne peut poser à la main, couvrent `pull` et `import` sur les deux systèmes. Un `pull` sans effet ne touche
  pas au blob (`verify` le lit, `register` n'écrit que le manifeste) : même `BlobId`, pas de rechargement. Risque
  résiduel accepté, dans le sens demandé : tout changement de ctime (un `chmod`, un attribut étendu posé par un outil
  du système) recharge le modèle une fois. Coût : un `stat` par requête dans `submit`, avant le verrou des moteurs.
  Le moteur garde le `BlobId` lu dans `spawn`, avant le chargement. Les tests existants n'ont pas de blob : `None`
  des deux côtés, donc `a_manifest_written_again_with_a_new_date_keeps_the_loaded_engine` reste à 1 chargement.
- R2-DIF-05. Le canal d'un job de `pending` est vide (64 places) : `try_send` y vaut `blocking_send`. Je l'ai passé
  aussi pour que le bras neuf n'ait qu'une forme. Un client dont le canal est plein ne reçoit pas la ligne d'erreur ;
  son flux finit sans `done` quand le `Sender` tombe, ce que CON-06 côté client (laissé de côté au lot 2) traitera.
- R2-DIF-06, choix : la remise à neuf reste, et seul le commentaire change. Raisons. (1) Les erreurs qui atteignent
  ce bras sont, en service, des erreurs du périphérique : `prepare` refuse avant tout dépassement de contexte, le
  chargeur refuse un tokenizer dont un id dépasse la table (SEC-07), et `ids` n'est jamais vide après
  `reuse_prefix`. L'erreur « bénigne » du test (`a_failed_prefill_leaves_an_empty_cache_to_the_next_request`, id 100
  sur une table de 64) n'est plus atteignable par un vrai modèle. (2) `truncate` répare la COMPTABILITÉ des caches
  (positions au-delà des ids) ; il fait confiance au contenu des positions gardées. Après une erreur du noyau, rien
  ne garantit ce qui a été écrit, et des caches neufs ne supposent rien. (3) Le coût n'est payé qu'après une erreur
  du périphérique, rare, et vaut une relecture de l'historique, pas une perte de texte : le client renvoie toute la
  conversation à chaque requête. Le test existant garde la remise à neuf et reste tel quel.
- R2-API-10. Le nouveau `scan` ne retient jamais moins que l'ancien : un préfixe qui finissait par U+FFFD dans
  l'ancien test se retrouve, sans ses U+FFFD, comme préfixe plus court dans `shown`. Les quatre tests existants de
  `scan` donnent les mêmes valeurs (recalculées à la main). Un U+FFFD réel écrit par le modèle fait retenir le
  préfixe un jeton de plus, puis `tail` ou le jeton suivant le rend.
- R2-API-11. `manifests` est la boucle de l'ancien `list`, qui range les erreurs de `get` au lieu de les propager ;
  les erreurs d'entrée-sortie sur les répertoires restent fatales, comme avant. Le type de retour passe par un alias
  (`Manifests`) pour rester sous le seuil de `clippy::type_complexity` (270 sans l'alias, compté à la main). Dans
  `remove`, la liste est lue avant d'effacer : une erreur de lecture laisse le magasin intact. Le cas « tous les
  manifestes lisibles » se comporte exactement comme avant (`o != n` remplace l'absence du manifeste effacé). Reste
  un état à moitié fait possible : le manifeste effacé, puis l'effacement du blob qui échoue sur une erreur
  d'entrée-sortie (droits) ; préexistant, non touché. Un `rm` du modèle dont le PROPRE manifeste est illisible
  échoue toujours avant d'effacer quoi que ce soit (pas d'état à moitié) ; il faut effacer le fichier à la main.
  Le message 500 de `/api/chat` sur ce modèle-là reste, et il nomme le fichier et le champ.
- R2-API-09. Le document OpenAPI ne change pas : les morceaux du flux n'y sont pas décrits, aucun schéma n'a bougé.
  Le morceau de `usage` ne porte pas `warnings`. `logprobs` à `false` est ce que llvq fait : aucun avertissement.
  Toute autre valeur non nulle (y compris l'entier du `/v1/completions` hérité) avertit.

## Laissé de côté

| ID | raison |
|---|---|
| R2-API-09, `tool_calls` dans l'historique | Choix d'opérateur (voir plus bas) : refuser ou avertir. Côté `/v1`, `OaiMessage` n'a pas de champ `tool_calls` et serde l'ignore ; l'ajouter change `openapi.json`, qui se régénère par `LLVQ_UPDATE_OPENAPI=1 cargo test -p llvq-runtime --test openapi`, jamais à la main. |
| R2-API-15, `param` | Renseigner `param` demande un argument de plus à `oai_error` (environ 25 appelants) ou une seconde fonction : hors de la plus petite correction. Plan : `fn oai_error_param(status, message, param: &str)` à côté, appelée par `content_text` (param `messages`), `options` (`n`, `stop`) et le refus de température de R2-API-13. |
| R2-API-11, temporaire de `write_manifest` | Constat juste (`<tag>.json.tmp` partagé par deux écrivains du même nom), non demandé par la mission et sans test possible hors course. Plan, 4 lignes : `static N: AtomicU64`, `p.with_extension(format!("json.{}.{}.tmp", std::process::id(), N.fetch_add(1, Relaxed)))` ; `manifests` ignore déjà tout nom qui ne finit pas par `.json`. |
| R2-API-12 | Plus de 30 lignes et une signature interne changée. Plan : `Store::pull` prend `report: &mut dyn FnMut(PullEvent) -> bool` (« le client écoute encore ») ; `lock_blob`, `verify` et `register` gardent leur forme par une fermeture qui jette le booléen ; dans la boucle de lecture, sur `false`, `out.flush()?` puis `bail!("the pull was cancelled; the partial file is kept for the next pull")`. Appelants : `api.rs` (`tx.blocking_send(..).is_ok()`), `main.rs` (`true`), `import` (fermeture adaptée), une demi-douzaine de fermetures `\|_\| {}` de `tests/pull.rs` à passer en `\|_\| true`. Test sur `slow_hub()` : un `report` qui rend `false` au premier progrès, erreur en moins d'une seconde, partiel présent. |
| R2-API-13 | Effort M, une méthode de trait et la lecture d'une config avant chargement. Plan du relecteur repris : `Loader::greedy_only(&Manifest) -> Option<bool>` par défaut `None` ; `DeviceLoader` lit le champ `decode` de la config que `config_for` choisirait (ou `LLVQ_BATCH`) ; `Models::submit` refuse en 400 avant `Engine::spawn` quand il rend `Some(true)` et que la température demandée est positive. Test : un `Loader` compté qui rend `Some(true)`, `submit` à 0,7 rend 400 et 0 chargement. |
| R2-DIF-05, cas du canal plein | Pas de test : il faut un client figé à la case près au moment d'une panique. Le test de panique existant garde la livraison quand il y a de la place. |

## Constat faux ou déjà corrigé

| ID | preuve |
|---|---|
| aucun | Les huit constats traités se lisent dans le code : `same_model` ignore `modified_at` et rien d'autre ne distingue le blob réécrit ; `Client::new` n'a qu'un délai de connexion et `first.join()` est sans borne ; `blocking_send` dans `live.drain(..)` ; le commentaire de `run_inner` ignore `Session::truncate` ; `text.ends_with` avant `printable` dans `scan` ; `self.get(&mn)?` dans `list` et `self.list()?` après `remove_file` dans `remove` ; `ctx.warnings` lu seulement hors flux et `logprobs` dans `HARMLESS` ; bras `(kind, _)` pour `{"type":"text"}` sans `text`. Le scénario de R2-DIF-02 suppose le mutant appliqué ; sans lui le test finit, ce qui ne change rien au défaut. |

## Décision d'opérateur

| ID | option recommandée | valeur proposée | coût |
|---|---|---|---|
| R2-API-11, `deny_unknown_fields` sur `Configs` | Lire enfin le champ `schema` du manifeste, et refuser par nom un `schema` plus grand que celui du binaire (« written by a newer llvq; update llvq »). Puis retirer `deny_unknown_fields` de `Configs` lu dans un manifeste, et garder la rigueur de `library.json` par un test qui compare ses clés à la liste connue. Un champ qui ajoute une config se lit alors par un binaire ancien, qui ne sait de toute façon pas la servir ; un champ qui change le sens d'un autre monte `schema` et se refuse. Rien n'est changé aujourd'hui : avec la correction des listages, un binaire ancien liste les autres modèles et nomme le manifeste qu'il ne lit pas. | `schema` = 1 aujourd'hui ; le monter à la prochaine rupture | S, environ 15 lignes dans `store.rs` et `library.rs` (le struct est partagé par `Entry` et `Manifest`, d'où le test de clés), un test dans `tests/pull.rs`. Touche la tolérance du format du magasin : décision d'opérateur. |
| R2-API-09, `tool_calls` dans un message | Refuser en 400 par nom sur `/api/chat`, comme `tools` (`refuse_unsupported`) : le plan dit « refused by name rather than half-implemented » pour les appels d'outils, et un historique dont on a retiré les appels n'est plus la conversation du client. Sur `/v1`, ajouter `tool_calls` à `OaiMessage` et refuser de même, puis régénérer `openapi.json` par la commande de test. L'autre option, un avertissement D3, ne casse aucun client mais laisse répondre sur un historique tronqué. | aucune | S : 5 lignes dans `api.rs`, 8 dans `openai.rs`, un cas dans `refusals_answer_with_their_status_and_a_reason` ; la régénération du document. Un client qui envoie aujourd'hui un historique avec `tool_calls` reçoit 400. |
| R2-API-14 | Ne rien changer tant que le comportement d'Ollama n'est pas lu dans son code. Le dépôt n'annonce rien sur les blancs de `thinking` (fixture OpenAPI, README, plan) ; le serveur envoie le texte des ids tel quel. Si l'opérateur veut l'alignement : retirer au routage les `\n` de tête de `content` et de tête et de queue de `thinking`, et garder dans le memo le texte ENVOYÉ, faute de quoi `render` ne retrouve plus le segment quand le client renvoie la réponse et relit tout l'historique. | aucune | S-M : `Router` et le memo dans `engine.rs`, un test multi-tours avec `think` qui exige `prompt_eval_cached_count` inchangé au second tour. Change ce que le client reçoit. |

## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)

Aucun fichier sous `cfg(target_os = "linux")`. `store.rs` a un bloc `cfg(unix)` (`MetadataExt`), compilé sur Mac et
sur Linux, et un bloc `cfg(not(unix))` que personne ne compile ici. Une compilation croisée de `llvq-runtime` reste
prudente pour ce bloc `cfg(unix)` et pour `EngineInfo`, qui a gagné un champ.

## Commandes lancées et leur résultat

Aucune. Consigne : pas d'exécution locale (une suite longue tourne sur la machine). Lecture, `git grep`, `git diff`
seulement.

## À lancer quand la machine est libre

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
