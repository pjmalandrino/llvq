# Revue du code neuf, runtime et moteur (second tour) : constats
Périmètre réellement lu : `git diff` entier de `llvq-runtime` (src, tests, README), `llvq-llm/src/session.rs` (en entier),
`llvq-llm/tests/session.rs` (en entier), `llvq-llm/src/repl.rs` (`Sampler`, `nucleus_sorted`, `nucleus_selected`, tests
neufs), `llvq-llm/tests/sampler_nucleus.rs`, `served_read_env.rs`, `softmax_whole.rs`, `served_config.rs` (diff),
`llvq-runtime/src/engine.rs` (lignes 1-1666, toutes les fonctions touchées et leurs appelants), `store.rs` (`lock_blob`,
`remove`, `pull`, `import`, `register`), `api.rs` (`stop_list`, `delete`, `pull`), `client.rs` (`stream`),
`persist4e_cuda.rs` (`chain_prefill`, `tree_launch`, `prepare`), `served_decode.rs` (`Decoder`, `make_grow_buffers`),
`model.rs` (`KvCache` en entier), `fused_metal.rs` (`RotOp`, `MetalRotation`), `fused_cpu.rs` (`check_token_ids`, gather),
`fused_cuda.rs` (chargeur, autour de l'appel neuf), `bin/{mmlu,ppl,seal,chat,fusedrun,gsm8k,gsm8kpair}.rs` (diffs),
`gsm8k.rs` (`served_header`, `Dump`, `pair`), `sealed.rs` (`RestoreF16`). Survolés : `llvq_rot.metal` (boucles et
attributs), `kvq.rs` (`store`), `served.rs` (`MEASUREMENT_MODES`, refus de `prefill_attention`), appels de `mmlu`, `ppl`,
`seal` dans `ops/jobs`.
Non couvert : `model.rs` hors `KvCache` (chemin `hidden_cached`, masques), `openai.rs` hors `stop`, le reste de
`fused_cuda.rs` et `fused_metal.rs`, `batch.rs` hors `Pool`. Rien n'a été compilé ni lancé.
Bilan : 0 critique, 0 majeure, 5 mineures, 1 cosmétique

### R2-DIF-01 `same_model` garde un moteur chargé depuis un blob qu'un pull de réparation vient de remplacer
- Gravité : mineure
- Confiance : CONFIRMÉ (chemin lu de bout en bout ; il faut un blob corrompu au moment du chargement)
- Où : llvq-runtime/src/engine.rs:1425-1430, :1503 ; llvq-runtime/src/store.rs:328-333, :439-444, :303-309, :48
- Fait : `Models::submit` reprend un moteur chargé dès que le manifeste ne diffère que par `modified_at`. Or `pull` et
  `import` sont aussi le geste de réparation : sur un blob présent dont `verify` échoue, ils le suppriment et rendent
  « pull again » (`store.rs:329-331`) ; le pull suivant retélécharge, renomme et écrit un manifeste neuf par
  `Manifest::of`, qui ne diffère que par la date. Le chargeur ne revérifie pas le sha256, et le handler `pull`
  (`api.rs:650-657`) ne décharge rien.
- Scénario : un blob s'abîme sur disque, puis le serveur charge `qwen3:4b` depuis lui et répond du bruit.
  L'utilisateur lance `llvq pull qwen3:4b` (refus, blob supprimé) puis encore `llvq pull qwen3:4b` (blob sain,
  manifeste neuf). Avant CON-08, la requête suivante chargeait un second moteur sur le blob sain. Maintenant
  `same_model` rend vrai et le moteur chargé depuis les octets abîmés sert jusqu'à son keep-alive ou un `llvq stop`.
  Aucune autre différence de manifeste n'existe dans ce cas : même sha256, même taille, même révision.
- Touche : runtime
- Correction proposée : garder `same_model` et lui ajouter l'identité du fichier. Dans `Engine::spawn`, lire
  `std::fs::metadata(store.blob_path(&manifest.sha256))` et ranger `(ino, mtime, mtime_nsec)`
  (`std::os::unix::fs::MetadataExt`, `None` si le blob manque) dans un champ neuf `EngineInfo::blob` ; dans `submit`
  à `:1503`, exiger aussi `e.info.blob == blob_id(&self.store, &manifest)`. Un blob retéléchargé est un fichier neuf,
  un pull sans effet garde le même. Test neuf dans `tests/engine_busy.rs` : écrire un blob, charger avec
  `CountingLoader`, supprimer puis réécrire le blob et le manifeste à `+3600`, exiger `loads == 2` ; le test
  `a_manifest_written_again_with_a_new_date_keeps_the_loaded_engine` doit rester à 1. Mutant : retirer la comparaison
  de `blob`, le test neuf rougit. Variante : ne réécrire le manifeste que si le blob a été écrit (`pull`/`import`
  gardent la date sur le chemin « blob présent et vérifié ») et revenir à l'égalité stricte.
- Effort : S
- Risque de la correction : faible. Aucun format, aucun noyau, aucun token servi.
- Vérification proposée : le test neuf ci-dessus et son mutant ; puis `cargo test -p llvq-runtime --test engine_busy`.

### R2-DIF-02 Le test du panic du batch pend sous le mutant que la file de vérification prévoit
- Gravité : mineure
- Confiance : CONFIRMÉ (lu ; le mutant n'a pas été lancé)
- Où : llvq-runtime/tests/batch_server.rs:344-366, :372-381 ; llvq-runtime/src/engine.rs:1335-1358 ;
  llvq-runtime/src/client.rs:69, :107-127 ; docs/archive/audit-code-2026-11-18-brut/50-file-de-verification.md:26
- Fait : `stream_ended` lit le flux par `Client::stream`, dont l'agent n'a qu'un délai de connexion (`client.rs:69`)
  et aucun délai de lecture. `two_in_the_pool` attend la fin des deux flux sans borne (`first.join()` et le second
  appel sur le fil du test). Avec la correction CON-06, `live` appartient à la fermeture du fil moteur
  (`engine.rs:1336`).
- Scénario : le mutant listé (« supprimer la boucle sur `live.drain(..)` », `engine.rs:1347-1349`). Après le panic,
  les deux `Live` restent dans `live`, donc leurs `Sender` vivent ; le fil passe à `rx.recv()` et y dort, puisque le
  moteur `Failed` n'est oublié qu'au prochain `submit` et que le faucheur ne retire pas un moteur dont `expires` vaut
  `None`. Les deux flux ne finissent jamais : le test pend au lieu de rougir sur « ended without an error line »,
  attendu par la file. Une régression réelle de la même forme pendrait la CI jusqu'à son délai global.
- Touche : outillage
- Correction proposée : dans `two_in_the_pool`, lancer les deux `stream_ended` sur des fils qui rendent leur résultat
  par un `mpsc` et les recevoir par `recv_timeout(Duration::from_secs(120))`, avec un `panic!` qui dit « the stream
  neither ended nor failed within 120 s ». Corriger l'attendu de la ligne 26 de la file : rouge par ce délai.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : appliquer le mutant `engine.rs:1347-1349` ; attendu rouge en au plus 120 s avec le
  message du délai ; défaire, vert.

### R2-DIF-03 La branche `Cat` de `positions_held` n'est tenue par aucun test, sur le chemin servi Metal et CPU
- Gravité : mineure
- Confiance : PLAUSIBLE (lu ; le mutant n'a pas été lancé)
- Où : llvq-llm/src/session.rs:81-86, :139-148 ; llvq-llm/tests/session.rs:262-290 ; llvq-runtime/tests/server.rs:203-212
- Fait : sous `Cat`, `KvCache::len` reste à 0 (`model.rs:412-424` ne le touche pas), d'où le recours à `held()`.
  La config Metal servie et la config CPU n'ont pas de champ `attention`, donc `Expanded`, donc `Cat`
  (`kvq.rs:680-686`). Le mutant `KvStore::Cat => c.len()` ne change rien pour `0 < n` (copie inutile) et saute la
  coupe à `n == 0` : `ids` est vidé, le cache garde son histoire.
- Scénario : sous ce mutant, `a_divergent_history_is_cut_back_to_the_common_prefix` appelle `reuse_prefix(&[])` puis
  ne vérifie que `session.is_empty()`, qui lit `ids` (`:287-288`) : vert. Le seul test du runtime qui passe par
  `keep = 0` après un tour (`generate` brut après un chat, `server.rs:203-212`) ne lit que `prompt_eval_count` et
  `eval_count`. Sur le serveur Metal, une requête dont le premier jeton diffère du cache (un `raw`) lirait des
  positions périmées.
- Touche : runtime
- Correction proposée : dans `a_divergent_history_is_cut_back_to_the_common_prefix`, après `:288`, nourrir
  `[1, 2, 9, 9]` dans la session vidée et comparer ses logits à ceux de `fresh` (même borne 1e-4), sur les deux
  magasins ; ou exiger que chaque cache rende `held()` vide ou 0 position.
- Effort : S
- Risque de la correction : nul (test seul).
- Vérification proposée : mutant `session.rs:83`, `KvStore::Cat => c.len()` ; attendu rouge sur `Cat` (erreur de
  forme du masque ou logits faux) ; défaire, vert.

### R2-DIF-04 Le repli de `nucleus_selected` coûte plus que l'ancien tri sur une ligne qui ne se stabilise pas
- Gravité : mineure
- Confiance : PLAUSIBLE (lu et compté ; rien de mesuré)
- Où : llvq-llm/src/repl.rs:119-155 (boucle `:128-153`), appel `:65-66` ; llvq-runtime/src/engine.rs:383-391, :694
- Fait : la justesse tient (voir « Relu sans constat »). Le coût, non : `p.to_vec()` copie 1,2 Mo, puis k double de
  64 tant que `k < n/4`. À 151 936 jetons, une ligne dont le total ne s'arrête pas fait 10 passes de
  `select_nth_unstable_by` sur toute la ligne et trie 65 + 129 + ... + 32 769 paires (65 482 au total), rend `None`,
  et `nucleus_sorted` refait le tri complet (*computed*). Une égalité dans le noyau rend `None` après au moins une
  sélection. Le journal 45 parle d'un gain « probablement nul » sur ces lignes : c'est une perte.
- Scénario : `llvq serve` sur Metal ou CPU, sans `temperature` dans la requête : 0,7 et `top_p` 0,9 par défaut
  (`engine.rs:385-391`), donc `pick` à chaque jeton. À 0,7, tout jeton à moins de 11,6 logits du maximum n'est pas
  absorbé par un total voisin de 1 (*computed* : 0,7 × ln(2^24)). Une position incertaine en a des milliers, et les
  logits du chemin Metal, calculés en f16, se touchent souvent dans le noyau. Le bras « flat » de l'instrument prévu
  (`sampler_nucleus.rs:181-182`) ne se stabilise jamais : il montrera le pire cas.
- Touche : runtime (échantillonnage Metal et CPU ; aucun jeton ne change)
- Correction proposée : borner la recherche, `while k < p.len() / 4 && k <= 1024` à `repl.rs:129`, soit au plus
  5 sélections avant le repli ; l'identité au bit ne dépend pas de la borne. Puis lancer
  `pick_times_against_the_sorted_reference` avec un troisième bras, la ligne « six heads, f16 mantissa » de `rows()`,
  et ne garder le raccourci que si le pire bras perd moins que le bras piqué ne gagne (décision d'opérateur, déjà
  ouverte au journal 45).
- Effort : S
- Risque de la correction : nul pour la justesse (repli exact). Instrument : non.
- Vérification proposée : l'instrument ignoré existant, en release, bras « flat » : attendu aujourd'hui une médiane de
  `pick` au-dessus de celle de `pick_reference` (*mesuré* à obtenir, avec la machine).

### R2-DIF-05 Le bras de panic du batch envoie l'erreur par `blocking_send` à un client qui ne lit plus
- Gravité : mineure
- Confiance : PLAUSIBLE (mécanisme lu ; la fenêtre est d'un pas)
- Où : llvq-runtime/src/engine.rs:1347-1349 (neuf) ; même forme, préexistante, à :1371
- Fait : la boucle neuve envoie `Event::Error` à chaque `Live` par `blocking_send` sur son canal de 64. Le fil tient
  `w`, poids et caches compris, jusqu'à la fin du bras. Pendant ce temps l'état est `Failed`, et `Models::submit`
  oublie le moteur (`:1502`) puis en charge un autre.
- Scénario : sous `LLVQ_BATCH`, un client cesse de lire ; un `advance` remplit la 64e case de son canal, et le pas
  groupé suivant panique. Le fil bloque dans le bras, le moteur suivant se charge à côté des poids encore tenus :
  sous `LLVQ_MAX_LOADED=1` sur la carte, deux copies du modèle, et l'échec de chargement probable du second tant que
  le client reste connecté. Les jobs de `pending` n'ont rien reçu, leur canal est vide : eux ne bloquent pas.
- Touche : runtime
- Correction proposée : `try_send` au lieu de `blocking_send` dans les trois boucles du bras batch
  (`:1347-1357`) et pour le job du bras non batch (`:1371`) : la ligne d'erreur est au mieux, un canal plein veut dire
  un client qui ne lit pas.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante (le cas demande un client figé à la case près).

### R2-DIF-06 Le commentaire de la remise à neuf sur un préremplissage raté décrit un défaut que `truncate` ferme déjà
- Gravité : cosmétique
- Confiance : CONFIRMÉ (lu)
- Où : llvq-runtime/src/engine.rs:680-683 ; llvq-llm/src/session.rs:139-148
- Fait : le commentaire dit que des caches plus longs que les ids seraient gardés par le préfixe de la requête
  suivante. Depuis le même lot, `Session::truncate` coupe chaque cache à `n`, même à `n == ids.len()`, donc
  `reuse_prefix` les ramène. La remise à neuf est une seconde garde, et elle jette la conversation sur toute erreur de
  `feed_with`, y compris une erreur levée avant toute écriture, ce que fixe le test
  `a_failed_prefill_leaves_an_empty_cache_to_the_next_request` (erreur à l'embedding, `server.rs:610-633`).
- Scénario : un lecteur qui retire la remise à neuf en se fiant au commentaire croit rouvrir BUG-50 ; un lecteur qui
  retire la garde de `truncate` croit la remise à neuf suffisante pour tous les appelants, ce qu'elle n'est pas hors du
  runtime (`examples/overhead.rs`, REPL).
- Touche : doc
- Correction proposée : réécrire `:680-683` : « `Session::truncate` already cuts caches that a failed prefill left
  ahead of the ids; the reset is a second guard, and costs one re-read of the history after any prefill error. »
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

## Relu sans constat

- `Session::truncate` et `positions_held` : `Cat` lit `held()` (deux clones d'Arc), `Grow` et `Prealloc` lisent `len` ;
  un cache vide, jamais rempli, ou libéré par `release` rend 0 ; à `n == ids.len()` aucun `c.truncate`, donc aucune
  copie sous `Cat` ; un cache plus court que `n` rend l'erreur nommée de `KvCache::truncate`, seulement atteignable
  par un `set_len` raté, impossible après le `reserve` de 4E ; le magasin int8 de v9k8 libère ses caches mais ne
  passe pas par `Session` (`bin/gsm8k`, `prefillbench`).
- `Session::answer_launches` contre `git show 925930d:llvq-llm/src/session.rs` : corps identique ligne pour ligne ;
  `launch_tokens` lu une fois (constante `LAUNCH_TOKENS`) ; `as_mut_slice` égale la coercion d'avant ; `stops` passé
  à la fermeture est celui de `Generate` ; `launch_tokens == 0` sort par `Stop` comme avant.
- Les trois tests 4E de `tests/session.rs` : valeurs recalculées à la main (journaux `(5,9,4),(9,108,4),(13,112,2)` et
  `(20,20,4),(24,123,3)`, capacités `Grow(16)` suffisantes) ; le test de lancement raté tue la sortie anticipée
  (positions 10 et 11 nulles dans l'attention).
- `catch_unwind` du chargement : aucun `Mutex` global sur le chemin de `DeviceLoader` (`git grep` de `static ...Mutex`
  dans `llvq-llm`, `llvq-cuda`, `llvq-runtime`) ; le déroulement libère ce que la mort du fil libérait ; `panic =
  "abort"` n'est posé dans aucun profil ; le test ne peut pas pendre (refus « has stopped » ou premier événement `None`).
- CON-06, comptes de `queue` : un job de `pending` n'est jamais pris (`take_job` au `pop_front` seulement, les arrivées
  de `solo_decode` vont à `pending` sans `take_job`, les `Load` y sont pris et répondus), un `Live` l'a été : ni double
  décompte ni oubli.
- CON-07 : `expires` posé par `Live` drainé suit la règle d'une fin propre (le dernier écrit gagne).
- CON-10, `Store::remove` : un seul verrou, rendu à chaque sortie par la fin de portée, `flock` par description de
  fichier (même processus ou non) ; le refus est nécessaire quand le pull concurrent est sur le chemin « blob présent ».
- SEC-03, `stop_list` : octets partout, liste vide acceptée, chaîne seule bornée, `/api/*` et `/v1/*` par la même
  fonction ; aucun script de `ops/` n'envoie de liste d'arrêt.
- BUG-51 : `MAX_CAP` = 8 704 = 34 × 256 et `GROW_CHUNK` = 256, donc `pos0 + n > MAX_CAP` refuse exactement ce que
  `cap > MAX_CAP` refusait après `reserve`, sans faire grossir le tampon ; le contrôle du bloc `mv` est celui de
  `:2283-2286`, avant `set_len`.
- BUG-70 : la mémoire de groupe vaut `n · 4` octets, indépendante du nombre de fils ; les noyaux bouclent par
  `nthreads` sans réduction entre fils (`llvq_rot.metal:136-437`) ; la grille garde `rows` groupes ; 0 refusé.
- SEC-07, `check_token_ids` : `>=` strict, jetons ajoutés par `get_vocab(true)`, maximum sur des ids épars ; appelé dans
  les trois chargeurs avant `take_embed_tables_at`, donc la table est encore dans `raw` ; Qwen3 : 151 668 < 151 936 ;
  le champ `context` de `/api/generate`, seule entrée d'ids bruts, est ignoré (`api.rs:424`, `:854`).
- `nucleus_selected`, justesse : valeurs (multiset du préfixe), total (`Sum` est un `fold` séquentiel ; le premier
  poids vaut 1,0, donc le départ `-0.0` ou `0.0` ne compte pas ; arrondi monotone), coupe (mêmes opérations, `i < k`),
  ordre (stricte décroissance jusqu'à `i + 1`, `-0.0` impossible pour `exp`) ; NaN, `+inf`, `-inf` masqué (poids
  `+0.0` admis et juste), ligne toute `-inf`, ligne constante, `top_p` NaN, négatif, 0, 1 ou plus, ligne de moins de
  260 jetons : même vecteur que le tri, ou repli.
- `sampler_nucleus.rs` : la référence est `pick` de `925930d` mot pour mot ; les cas de repli comparent l'ancien code à
  lui-même, les cas pris sont les seuls qui testent, et le ratio `taken` du test de `repl.rs` empêche un raccourci mort.
- `bin/mmlu` (`parse_limit`, `kv_ran`) : les appels de `ops/jobs` passent une limite absente, 1, 5 ou 40 ; `kv_ran` vaut
  `cfg.kv`, comme l'en-tête du dump.
- `bin/ppl` : tous les appels documentés sont `ppl 4096 12 <device> <fichier>` ; le positionnel de modèle vide reste
  accepté (`dclm-14b-ft-mmlu.sh:127`).
- `bin/seal` : les quatre appels de `ops/jobs` posent `LLVQ_MODEL="$MODEL"`, la chaîne que `smoke` écrit
  (`smoke.rs:803`, `:853`).
- `bin/chat` : une graine au-delà de 2^24 donne désormais un autre tirage qu'avant (l'arrondi f32 a disparu), voulu par
  BUG-07 ; aucun script ne pose ces variables.
- `bin/fusedrun` : `MEASUREMENT_MODES` contient les 17 noms d'avant, plus `LLVQ_TIME_EVENTS` et `LLVQ_BENCH_ARMS`, qui
  ne sont posés qu'en ligne devant `planesbench` dans `ops/jobs`.
- `gsm8k.rs` : `prefill_attention` est refusé à côté de la chaîne (`served.rs:762`), seules les configs `kprefill`
  NOT SERVED gagnent une clé ; `pair` ne lit pas l'en-tête.
- `sealed.rs` : libellé seulement ; `census-14b-ref.sh:213` ne lit que le libellé du checkpoint de référence.
- `served_read_env.rs`, `softmax_whole.rs` : seuls dans leur binaire, environnement nettoyé avant et après.
- Tests neufs du runtime : répertoires de magasin uniques (étiquette et pid, aucune étiquette en double), aucun état
  global partagé ; le test de `keep_alive` à `"5"` fixe le comportement existant de `parse_str`.
- Commentaires neufs de `persist4e_cuda.rs`, `fused_cuda.rs`, `main.rs`, `README.md` : noms vérifiés
  (`forward_rotated_rows`, `prefill_rows`, `tv_tetra48_rows_h`, `install_chain`, `cuda_batch`, `LLVQ_PROFILE=eager`,
  défaut de `LLVQ_MODULE_CACHE`).

## Candidats au lot 1

R2-DIF-02, R2-DIF-03, R2-DIF-05, R2-DIF-06.
