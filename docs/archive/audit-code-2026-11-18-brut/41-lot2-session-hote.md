# Lots 2 et 3, moteur hôte de llvq-llm (session, hôte 4E, chargeurs) : journal

Rien n'a été compilé ni lancé. Toutes les lignes ci-dessous sont relues, NON COMPILÉES.

## Corrigé

| ID | fichiers | ce qui change, en une ligne | vérification faite |
|---|---|---|---|
| TST-01 | `llvq-llm/src/session.rs`, `llvq-llm/tests/session.rs` | La boucle 4E de `answer_fed` passe telle quelle dans une méthode publique neuve `Session::answer_launches(first, max_new, stops, launch_tokens, on_tokens, launch)` ; `answer_fed` l'appelle avec `dec.continue_greedy` ; trois tests CPU la pilotent avec un lancement factice (`stand_in_launch`) | relu, NON COMPILÉ |
| BUG-50 / CON-02 (partie llvq-llm) | `llvq-llm/src/session.rs`, `llvq-llm/tests/session.rs` | `Session::truncate` n'a plus de sortie anticipée : chaque cache dont la longueur diffère de `n` est coupé à `n`, même à `n == ids.len()` ; nouvelle fonction privée `positions_held` (longueur par magasin) ; test `a_failed_launch_leaves_caches_the_next_request_cuts_back` | relu, NON COMPILÉ |
| CONNU « `feed` raté en milieu de modèle » | aucun fichier de plus | Couvert par la même garde, voir la note 3 ci-dessous | relu, NON COMPILÉ |
| BUG-51 | `llvq-llm/src/persist4e_cuda.rs` (Linux/CUDA) | `prepare` refuse `pos0 + n > MAX_CAP` avant la boucle des `reserve` ; `tree_launch` refuse `pos0 + span > MAX_CAP` dans le bloc `mv`, avant ses `reserve` et `set_len` | relu, NON COMPILÉ |
| BUG-70 | `llvq-llm/src/fused_metal.rs` | Les deux lancements de rotation (`rot_apply_tg_metal`, et `rot_apply_metal` / `rot_apply_rows_metal`) prennent `min(self.rot.threads, pipe.max_total_threads_per_threadgroup())` arrondi au multiple de `LANES` inférieur, et refusent 0 par nom, sur le modèle de `:996-999` | relu, NON COMPILÉ |
| SEC-07 (contrôle hôte) | `llvq-llm/src/fused_cpu.rs`, `llvq-llm/src/fused_metal.rs`, `llvq-llm/src/fused_cuda.rs` (Linux/CUDA) | Fonction `pub(crate) fused_cpu::check_token_ids` : le plus grand id du tokenizer (`get_vocab(true)`, jetons ajoutés compris) doit être `< dims[0]` de `model.embed_tokens.weight` ; appelée par les trois chargeurs juste après le parse du tokenizer ; test unitaire dans `fused_cpu.rs` | relu, NON COMPILÉ |

### Notes par constat

1. TST-01. `Decoder` est un type concret, sans trait : `is_persistent()` rend `false` hors Linux+CUDA (`served_decode.rs:122-131`), et `served_decode.rs` n'est pas dans mes fichiers. La seule voie sans toucher une signature publique est d'extraire la boucle derrière une fermeture, comme le proposait 09-tests.md. Production neuve : la signature et la doc de `answer_launches` (une quinzaine de lignes), la garde `max_new == 0` répétée (3 lignes, morte sur le chemin de `answer_fed` qui la fait déjà), l'appel dans `answer_fed` (4 lignes). Le corps de la boucle est déplacé ligne pour ligne. Deux écarts, tous deux sans effet : `dec.launch_tokens()` lu une fois avant la boucle (c'est la constante `LAUNCH_TOKENS`), et `&mut self.caches` devient `self.caches.as_mut_slice()`. La signature de `answer_fed` ne bouge pas. Le vérificateur (21-verif) disait « non » pour un lot 1 à cause du chemin servi ; c'est le lot 2, et la mission l'autorise sous trente lignes neuves.
   - La ligne du mutant n'est plus `session.rs:249` mais `session.rs:304`, dans `answer_launches`.
   - Ce qui reste non testé dans la branche 4E : le choix du premier jeton (`Fed::Hidden` ou `Fed::First`) et la fermeture qui appelle `continue_greedy`, trois lignes.
   - Les trois tests : `the_persistent_branch_feeds_what_its_launches_wrote` (trois lancements coupés par le plafond, flux éager entre deux tours, second tour arrêté par un jeton d'arrêt au deuxième lancement ; vérifie le journal des lancements `(offset, last, k)`, `ids`, et la longueur de chaque cache) ; `the_persistent_branch_cuts_where_it_stops_or_is_cancelled` (annulation, arrêt au premier jeton d'un lancement, premier jeton d'arrêt, plafond 1) ; `a_failed_launch_leaves_caches_the_next_request_cuts_back` (BUG-50).
   - Valeurs attendues calculées à la main : tour 1, prompt de 5, premier 9, plafond 11, lancements de 4 : journal `(5, 9, 4), (9, 108, 4), (13, 112, 2)`, réponse `9, 105..=114`, `ids` = prompt + les 10 premiers. Tour 2 : 20 positions tenues, premier 20, plafond 8, arrêt à la position 25 : journal `(20, 20, 4), (24, 123, 3)`, réponse `20, 120..=124`, 3 appels du rappel.
2. BUG-50 / CON-02. Le constat est juste dans le code : `truncate` sortait à `session.rs:129-131` sans toucher les caches. La nouvelle garde compare la longueur de chaque cache à `n`, par magasin : `Cat` lit la dimension 2 de `held()` (la forme de `from_caches:104`), `Grow` et `Prealloc` lisent `KvCache::len()`.
   - Chemin sans erreur inchangé : à `n == ids.len()` chaque cache tient déjà `ids.len()` positions, donc aucun `c.truncate` n'est appelé (pas de copie sous `Cat`, ce qu'aurait coûté la simple suppression de la sortie anticipée à chaque tour) ; à `n < ids.len()` chaque cache diffère de `n` et est coupé comme avant.
   - Un cache plus court que `n` (positions perdues) rend l'erreur nommée de `KvCache::truncate`, comme avant pour `n < ids.len()`.
   - Appelants de `Session::truncate` : `reuse_prefix` seulement (`git grep`). Appelants de `reuse_prefix` : `llvq-runtime/src/engine.rs:666` (`run`, avant `feed_with`) et `llvq-runtime/examples/overhead.rs:341`. Les deux gardent leur comportement sur le chemin sans erreur.
   - `persist4e_cuda.rs:2243-2256`, ordre de `set_len(pos0)` : laissé tel quel. Déplacer le bloc `mv` après les contrôles 2258-2289 serait une permutation sans effet sur le chemin de succès (ces contrôles ne lisent pas les longueurs des caches, seulement leur nombre), mais elle ne ferme que les sorties de contrôle, pas les vrais déclencheurs (watchdog, NaN, ligne sans jeton, OOM), qui arrivent après le lancement. Le vérificateur (21-verif, BUG-50) dit « non » pour ce déplacement, vérifiable seulement sur carte. La garde de `truncate` couvre tous les déclencheurs.
   - Je n'ai pas ajouté de remise en place des caches dans `Runtime4e::chain_prefill` sur erreur : la garde de `Session` suffit pour `Session`, et le correctif du runtime (autre agent) remet la session à neuf.
3. CONNU « `Session::feed` raté en milieu de modèle ». La même garde le couvre pour tout appelant qui repasse par `reuse_prefix` ou `truncate` avant de nourrir à nouveau. Revue des appelants qui continuent après une erreur de `feed` ou `feed_with` : `engine.rs:666-669` (`run`) passe par `reuse_prefix`, couvert ; `engine.rs:753-757` remet la session à neuf ; le moteur batch prend une `Session` neuve par requête (`engine.rs:918`, `:976`) ; le REPL (`repl.rs:241`, `:279`, `:310`) sort par `?` sur toute erreur. Aucun appelant ne nourrit de nouveau sans couper. `feed` lui-même n'est donc pas modifié. Un futur appelant qui le ferait ne serait pas couvert.
4. BUG-51. `MAX_CAP` = `P4E_MAX_CAP` = 8 704 (`persist4e_cuda.rs:56`, `served_decode.rs:34`). Le nouveau refus ne se déclenche que si `pos0 + n > MAX_CAP` ; dans ce cas l'ancien code refusait déjà (après `reserve`, la capacité est au moins `pos0 + n`, donc `cap > MAX_CAP` à `:2455`). Dans `tree_launch`, le `span` du bloc `mv` vaut celui de `:2279` dès que `depth` n'est pas vide ; `depth` vide est refusé à `:2258`. Aucun script ne lit ces messages (`git grep` de « scratch holds » et « positions up to » dans `ops/`, `paper2/`, `docs/METHODE.md`, `.github/` : rien).
5. BUG-70. Lu dans `llvq-llm/kernels/llvq_rot.metal` (lecture seule) : `rot_load`, `rot_wht_step`, `rot_scale_out`, `rot_mix`, `rot_mix_tg` et le corps de `rot_apply_tg_metal` bouclent tous par `for (i = tid; i < N; i += nthreads)` ; aucune hypothèse de puissance de deux ni de multiple de 32 ; chaque paire du papillon touche deux cases disjointes ; les barrières sont uniformes. Le commentaire du noyau le dit (`:255-256`, `:164-165`), et `llvq-metal/tests/rot_matches_host.rs:432-438` serre le groupe à `kern.max_threads_per_group()` pour la même raison. `rot_apply_rows_metal` prend `r = threadgroup_position_in_grid` : avec `dispatch_threads(threads * rows, threads)` et `threads` ramené à la limite, la grille garde exactement `rows` groupes. `self.rot.threads` vaut une puissance de deux dans [32, 1 024] (`fused_metal.rs:1208`) : sous une limite de 1 024 (M3 Max, selon le constat et la mission), `min` puis l'arrondi à 32 rendent la même valeur, donc rien ne change sur ce poste. Le refus par nom reprend le texte de `:998` ; `git grep` de « pipeline allows fewer » dans les dossiers lus par des scripts : rien.
6. SEC-07. Le contrôle compare le plus grand id que le tokenizer connaît à `dims[0]` de la table, inégalité stricte : un id `>= dims[0]` est refusé. J'ai pris le plus grand id plutôt que `get_vocab_size(true)` : la bibliothèque (tokenizers 0.21.4, `tokenizer/mod.rs:685-687`) signale elle-même que ce compte peut être faux, et un id est ce que le noyau indexe.
   - Qwen3, par la lecture : la table a 151 936 lignes (05b, « vocab 151 936 » ; 05-bugs « 151 936 x 640 mots »), le tokenizer a 151 669 entrées, ids 0 à 151 668 (20-verif, SEC-07). La ligne 151 669 est la première hors tokenizer : le préenregistrement DFlash l'appelle « mask row 151669 » (`docs/archive/debit-vague5-2026-11-02/prereg-speculation-1.md:73`). 151 668 < 151 936 : les trois fichiers publiés passent. Les trois partagent le tokenizer Qwen3 et `vocab_size` 151 936. Non mesuré sur les fichiers : aucune exécution.
   - Les jetons ajoutés reçoivent leur id de la bibliothèque, pas du JSON (`added_vocabulary.rs:284-296`) : le test le prend en compte, son jeton ajouté prend l'id 3.
   - Un fichier sans `model.embed_tokens.weight` passe ce contrôle et reste refusé là où la table est lue (`take_embed_tables_at` sous q4, `VarBuilder` sous f16) : aucun refus nouveau sur une entrée qui chargeait.
   - Aucun test ne charge un fichier scellé par `load_resolved` (`git grep` dans `llvq-llm/tests`, `llvq-runtime`) : aucun test existant ne voit le nouveau contrôle. Le chemin CPU bornait déjà l'id à la lecture (`fused_cpu.rs:248`) ; le contrôle y est ajouté pour que les trois chargeurs refusent le même fichier.
   - Coût : `get_vocab(true)` construit une table de 151 669 chaînes une fois par chargement.

## Laissé de côté

| ID | raison |
|---|---|
| BUG-50, déplacement des contrôles de `tree_launch` avant le bloc `mv` | Le vérificateur dit non (vérifiable sur carte seulement) ; la garde de `truncate` couvre aussi ces cas. Voir note 2. |
| BUG-50, remise des caches à `pos0` dans `Runtime4e::chain_prefill` sur erreur | Redondant avec la garde de `Session` et la remise à neuf du runtime ; Linux seulement, invérifiable ici. |

## Constat faux ou déjà corrigé

| ID | preuve |
|---|---|
| SEC-07, phrase de la mission « tokenizer et table ont la même taille pour Qwen3 » | Ils n'ont pas la même taille : 151 669 ids contre 151 936 lignes. Le contrôle est donc une inégalité (id max < lignes), pas une égalité ; une égalité refuserait les trois fichiers publiés. |
| BUG-50, déclencheur déterministe `ChainFault::PathOffByOne` | Déjà relevé faux par 21-verif : le premier lancement échoue avant tout `set_len`, et `chain_prefill` fixe `ChainFault::None`. |

## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)

- `llvq-llm/src/persist4e_cuda.rs` (BUG-51 : `prepare`, `tree_launch`)
- `llvq-llm/src/fused_cuda.rs` (SEC-07 : appel de `check_token_ids` dans `load_resolved`)

Fichier macOS + feature `metal`, hors de la boucle `cargo test` par défaut : `llvq-llm/src/fused_metal.rs` (BUG-70, SEC-07).

## Commandes lancées et leur résultat

Aucune commande de compilation ni de test. Seulement des lectures, `git grep`, `git diff`, `grep` et `sed -n` sur les sources.

## À lancer quand la machine est libre

| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo clippy -p llvq-llm --all-targets -j 8` | zéro avertissement (session.rs, fused_cpu.rs, tests/session.rs) | aucun |
| `nice -n 10 cargo test -p llvq-llm --test session -j 8` | `test result: ok. 12 passed` (9 anciens, 3 neufs) | `llvq-llm/src/session.rs:304`, remplacer `&made[..made.len().saturating_sub(1)]` par `&made[..]` : rouges `the_persistent_branch_feeds_what_its_launches_wrote` (le deuxième lancement part de 10 au lieu de 9 : réponse `110..` au lieu de `109..`, journal décalé) et `the_persistent_branch_cuts_where_it_stops_or_is_cancelled` (annulation : 8 ids au lieu de 7) |
| idem | idem | `llvq-llm/src/session.rs:140`, insérer après le `ensure!` la ligne `if n == self.ids.len() { return Ok(()); }` : rouge `a_failed_launch_leaves_caches_the_next_request_cuts_back` (« after reuse_prefix: layer 0 holds 12 positions for 10 ids ») |
| idem | idem | `llvq-llm/src/session.rs:142`, `!= n` devient `== n` : rouges `a_failed_launch_leaves_caches_the_next_request_cuts_back` et `a_divergent_history_is_cut_back_to_the_common_prefix` |
| `nice -n 10 cargo test -p llvq-llm --lib fused_cpu:: -j 8` | `a_tokenizer_id_past_the_embedding_table_is_refused_by_name` vert | `llvq-llm/src/fused_cpu.rs` dans `check_token_ids`, `get_vocab(true)` devient `get_vocab(false)` : rouge (table de 3 lignes acceptée) ; puis `>= rows` devient `> rows` : rouge |
| `nice -n 10 cargo clippy -p llvq-llm --features metal --all-targets -j 8` | zéro avertissement (fused_metal.rs) | aucun |
| `nice -n 10 cargo test -p llvq-llm --features metal --test rot_metal_tg -j 8` | vert : les deux rotations rendent les mêmes bits qu'avant sur le M3 Max (la limite y vaut 1 024, `threads` ne bouge pas) | `llvq-llm/src/fused_metal.rs:1308` et `:1352`, `self.rot.threads.min(` devient `(2 * self.rot.threads).min(` : reste vert sous `MTL_DEBUG_LAYER=1` (le groupe est ramené à la limite) ; puis retirer `.min(pipe.max_total_threads_per_threadgroup())` en gardant `2 * self.rot.threads` : la couche de validation refuse le lancement (ce que faisait le code d'avant pour une limite sous 1 024) |
| `nice -n 10 cargo test -p llvq-llm --features metal --test tetra48_metal_candle -j 8` | vert, inchangé (passe aussi par `RotOp`) | aucun |
| `ops/check-linux-cuda.sh clippy -p llvq-llm --features cuda --all-targets` | zéro avertissement (persist4e_cuda.rs, fused_cuda.rs) | aucun sur Mac ; sur carte, `decode` à `pos0 = 8 650`, `n = 64`, puis `truncate(100)` et un `decode` : refus nommé « positions up to 8714 » au premier, succès au second |
| porte servie sur carte, au prochain run payé déjà prévu (aucun run à lancer pour ceci) | jetons et ligne `NVRTC source` identiques ; la ligne de chargement ne contient aucun refus | aucun |

## Décision d'opérateur

- `Session::answer_launches` est une méthode publique neuve, exposée pour les tests. Option recommandée : la garder publique, `llvq-llm` expose déjà toute sa surface en `pub` ; l'alternative `#[doc(hidden)]` ne change rien au graphe.
- Les deux points de BUG-50 laissés de côté (ordre de `set_len` dans `tree_launch`, remise en place dans `chain_prefill`) : à reprendre seulement avec une porte sur carte, si l'opérateur veut fermer aussi les appelants de `Runtime4e` qui ne passent pas par `Session` (bancs `prefillbench`, `vtreebench`, `specrun`).
