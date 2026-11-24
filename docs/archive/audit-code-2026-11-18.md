# Audit du code, 2026-11-18

Audit de `main` à `925930d`, à 0 $. Rien n'a tourné sur une carte. Le rapport classe 131 constats en sept lots. Le
lot 1 est appliqué sur la branche `audit/code-2026-11-18` (section 6), les lots 2, 3, 4 et 7 aussi (section 7). Un
second tour d'audit a relu le tout et ses corrections sont appliquées (section 8). L'arbre entier est vérifié sur le
Mac : compilation, tests, clippy, contrôle croisé Linux, tests Metal, mutants, suite release. Rien n'a tourné sur une
carte NVIDIA. Rien n'est commité.

Résultat en une ligne : aucun constat ne change les tokens servis ni ne corrompt un fichier. Cinq défauts sont
majeurs. Quatre touchent la tenue du service en cas d'erreur, un touche un chiffre publié.

## 1. Méthode

- **Douze relecteurs** en parallèle, un par catégorie, en lecture seule : architecture, qualité (deux), bugs (moteur
  hôte, noyaux GPU), exactitude numérique, optimisation, sécurité et `unsafe`, tests et portes, cohérence code / docs /
  configs, concurrence du runtime, Python et outillage. Une seconde passe a repris les noyaux que la première n'avait
  pas lus.
- **Quatre vérificateurs contradictoires** ont ensuite essayé de réfuter 57 constats, dont toutes les majeures. Aucun
  n'a été réfuté. 21 ont été nuancés, le plus souvent sur la gravité.
- **Un mutant** a été lancé par la session principale (section 3, TST-01). Les autres preuves sont des lectures, des
  `git grep` et des recalculs.
- Les audits précédents ont été donnés aux relecteurs comme « déjà connu » : `docs/audit-architecture-2026-11.md`,
  `docs/runtime-audit-2026-11.md`, `docs/archive/audit-bugs-2026-11-08.md`, `docs/archive/revue-debit-2026-11-05.md`.

Les constats bruts, avec lignes, scénarios et corrections, sont dans
[`audit-code-2026-11-18-brut/`](audit-code-2026-11-18-brut/). Les fichiers `01-` à `12-` sont les constats, `20-` à
`23-` les vérifications, `30-` à `34-` les journaux du lot 1. Un ID renvoie à son fichier par son préfixe.

| préfixe | catégorie | fichier |
|---|---|---|
| ARC | architecture | `01-architecture.md` |
| QUA | qualité de code | `02-qualite-coeur.md`, `03-qualite-lourdes.md` |
| BUG | bugs | `04-bugs-moteur.md`, `05-bugs-noyaux.md`, `05b-bugs-noyaux-passe2.md` |
| NUM | exactitude numérique | `06-numerique.md` |
| OPT | optimisation | `07-optimisation.md` |
| SEC | sécurité et `unsafe` | `08-securite.md` |
| TST | tests et portes | `09-tests.md` |
| COH | cohérence code / docs / configs | `10-coherence.md` |
| CON | concurrence du runtime | `11-concurrence.md` |
| PY | Python et outillage | `12-python-ops.md` |

## 2. Bilan chiffré

| catégorie | majeures | mineures | cosmétiques |
|---|---|---|---|
| Architecture | 0 | 9 | 3 |
| Qualité de code | 1 | 15 | 8 |
| Bugs, moteur hôte | 0 | 6 | 4 |
| Bugs, noyaux GPU | 1, et 1 plausible | 1 | 4 |
| Exactitude numérique | 1 | 5 | 1 |
| Optimisation | 0 | 7 | 1 |
| Sécurité et `unsafe` | 0 | 8 | 3 |
| Tests et portes | 1 | 8 | 4 |
| Cohérence | 0 | 5 | 9 |
| Concurrence du runtime | 3 | 8 | 0 |
| Python et outillage | 0 | 18 | 4 |

Ce sont les gravités après vérification. Avant, les relecteurs annonçaient 1 critique et 12 majeures. Deux majeures
sont comptées deux fois, trouvées par deux relecteurs indépendants (QUA-50 et CON-03, BUG-50 et CON-02) : il y a cinq
majeures distinctes et confirmées. Une sixième, BUG-70, est plausible et n'a jamais été observée (lot 2).

Les cinq majeures Python et sécurité sont toutes redescendues en mineures. Les faits étaient exacts, les scénarios
exagérés : une variable non documentée qu'aucun job ne pose (PY-01), une version de layout qui n'a qu'une valeur
(PY-02), un chemin `launch` que les encodages n'empruntent plus (PY-03), 107 scripts sur 133 qui épinglent déjà le sha
du Space (PY-04), une compromission du registre amont sur des campagnes closes (SEC-01).

## 3. Les cinq majeures

### 3.1 Une panique au chargement bloque le runtime jusqu'au redémarrage (QUA-50, CON-03). CONFIRMÉ

- **Où.** `llvq-runtime/src/engine.rs:1269-1279`. Le chargement n'a pas de `catch_unwind`, alors que `run` et
  `run_batch` en ont un (`:1300-1314`).
- **Scénario.** `Loader::load` panique. Le fil meurt, l'état reste `Loading`, `queue` reste à 1. Le modèle répond 500
  pour toujours. Sous `LLVQ_MAX_LOADED=1`, tout autre modèle répond 503.
- **Correction.** Le même `catch_unwind` que `run`, et le passage à `Failed`. Effort S, local, testable sur Mac avec
  un `Loader` qui panique.

### 3.2 Un préremplissage par chaîne raté casse la conversation du 4B (BUG-50, CON-02). CONFIRMÉ

- **Où.** `llvq-llm/src/persist4e_cuda.rs:2243-2256` pose `set_len(pos0)` avant ses contrôles et avant le lancement.
  `Session::truncate` sort tôt quand `n == ids.len()` (`session.rs:129-131`).
- **Scénario.** Un lancement de chaîne après le premier échoue : watchdog, NaN ou mémoire de carte. Les caches restent
  plus longs que `ids`. Chaque requête qui garde le préfixe répond 500 jusqu'à `/clear` ou au rechargement. Le
  déclencheur déterministe cité par le relecteur est faux : il faut une vraie erreur de carte.
- **Correction.** Côté hôte : sur une erreur de `feed_with`, `session.reset` et `memo.clear()` comme à
  `engine.rs:756`. Effort S, chemin d'erreur seulement, testable sur CPU.

### 3.3 Sous `LLVQ_BATCH`, un client qui ne lit plus gèle tout le pool (CON-01). CONFIRMÉ

- **Où.** `llvq-runtime/src/engine.rs:1202` : `advance` fait `blocking_send` sans délai sur un canal de 64.
- **Scénario.** Un client reste connecté et cesse de lire. Son canal se remplit, le fil du moteur bloque, les autres
  flux du pool s'arrêtent. Rien ne détecte ce client, et axum n'a pas de délai d'écriture.
- **Portée.** Seulement sous `LLVQ_BATCH`, qui n'est pas servi par défaut. Hors batch, c'est le cas déjà connu de
  `docs/runtime-audit-2026-11.md`.
- **Correction.** Un délai au bout duquel un client lent est déclaré parti, à trois endroits (`advance`,
  `solo_decode`, `run`). Effort M. La valeur du délai est un choix d'opérateur : il coupe aussi un client lent
  légitime.

### 3.4 La branche 4E de `Session::answer_fed` n'a aucun test (TST-01). CONFIRMÉ par mutant

- **Où.** `llvq-llm/src/session.rs:225-271`. Cette branche tient `ids` et la longueur des caches entre deux tours.
- **Preuve.** Mutant `session.rs:249`, `&made[..]` à la place de `&made[..made.len().saturating_sub(1)]`. La suite
  `cargo test -p llvq-llm --test session` reste verte, 9 sur 9 (*mesuré*, session principale). Le vérificateur ajoute
  que la branche n'est exercée ni sur Mac ni sur carte.
- **Correction.** Un `Decode` factice sur CPU qui joue 4E dans les tests de session. Effort M.

### 3.5 Le b/param publié omet le pas de ligne que la carte tient (NUM-01). CONFIRMÉ, gravité ramenée de critique à majeure

- **Où.** `llvq-bench/src/bin/rtbits.rs:404-410` compte le flux Tetra à 48 bits par bloc. La carte tient chaque ligne
  arrondie à 8 octets (`llvq-artifact/src/tetra48.rs:61`). Le chargeur dit lui-même que ce pas se compte
  (`llvq-llm/src/fused.rs:2296-2300`). Le papier définit le b/param « at the widths the GPU holds »
  (`paper2/sections/model.tex:25-28`).
- **Vérifié.** Le noyau 4E v9 et la config chaîne lisent ce même tampon strié (`llvq_pers.cuh:1021`). Il n'y a pas de
  second agencement. La porte imprime déjà 2,550 / 2,377 / 2,524 b/weight, soit `rtbits` plus le pas, au millième.

| taille | publié | avec le pas (*computed*) |
|---|---|---|
| 4B | 2,7320 | 2,7393 |
| 8B | 2,6953 | 2,7001 |
| 14B | 2,7305 | 2,7324 |

- **Portée.** +0,0073 b/param au 4B, soit 0,27 %. Aucune comparaison ne bascule et b_max = 3,00 n'est pas approché.
  À deux décimales, le « 2,73 » du 4B devient 2,74 ; le 8B et le 14B ne bougent pas. L'audit du papier du 2026-09-28
  avait vu l'écart sans en trouver la cause et l'avait classé mineur.
- **Correction.** Dans `report_v5`, compter `d_out · stride_u32(d_in/24) · 32` bits par matrice Tetra, avec un test
  qui attend le 2,550 de la porte. Effort S pour le code. Les chiffres sont à confirmer en relançant `rtbits` sur les
  trois fichiers scellés. Republier (ETAT, README, papier 2, `llvq-runtime/README.md`) est une décision d'opérateur.

## 4. Ce qui sort sain

- **Tokens servis.** Aucun relecteur n'a trouvé de défaut qui les change.
- **Pool batch.** Aucun mélange de KV entre requêtes. Les slots et la bascule 4E / pool sont sains à la lecture.
- **4E.** Le bug « 4E empoisonné après un watchdog » du 2026-11-08 est corrigé : `settle` remet les compteurs à zéro
  après chaque lancement. Mémoire partagée au 14B : 96 064 B, sous les 101 376 B (*computed*). Aucun dépassement u32 au
  14B ni à B = 128.
- **Serveur.** Il écoute sur 127.0.0.1, refuse les origines étrangères, borne le corps à 2 Mo, `num_ctx` et
  `num_predict`. Les noms de modèle sont filtrés contre la traversée de répertoire. `pull` vérifie taille et sha256
  puis renomme de façon atomique. Aucun prompt n'est journalisé.
- **Python.** Aucun pickle, aucun `torch.load`, aucun `eval`. Les en-têtes PEP 723 sont épinglés. Aucun secret en
  clair dans l'arbre.
- **Outillage.** Les 163 scripts de job portent `set -euo pipefail`. Aucune porte de job n'avale son échec. Les deux
  listes de `Dockerfile.cuda` et `UPLOAD_ALLOW` concordent. Les copies de shaders de `llvq-tetra` sont identiques à
  l'octet, et le test qui le garantit tourne en CI.
- **Audit du 2026-11-08.** Ses constats de service 1.1, 1.2, 1.5, 1.7 et 1.8 sont corrigés.

## 5. Les sept lots

Le rang suit la simplicité d'abord, puis la criticité. Un lot ne dépend pas du suivant.

| lot | contenu | constats | effort | ce qu'il exige |
|---|---|---|---|---|
| 1 | Évident et peu critique : docs et commentaires faux, refus manquants des harnais, tests qui ne testent rien, petits refus Python | 42 visés | fait, section 6 | `cargo test` en debug, clippy |
| 2 | Tenue du service en cas d'erreur | 14 | S à M chacun | tests CPU du runtime, puis une porte sur carte pour 3.2 |
| 3 | Durcissement des lecteurs de fichiers | 16 | S chacun | suite complète en release (règle du dépôt sur le format) |
| 4 | Outillage payant et CI | 20 | S chacun | relecture par l'opérateur, un `publish` réel pour deux d'entre eux |
| 5 | Décisions d'opérateur | 12 | S pour le code | un choix, parfois une republication |
| 6 | Architecture et dette | 23 | M à L | par morceaux, hors des fichiers épinglés |
| 7 | Optimisation hors noyaux | 5 | S à M | une mesure avant et après |

### Lot 2. Tenue du service en cas d'erreur

Le plus utile après le lot 1. Tout est sur le chemin d'erreur, rien ne touche un token.

| ID | défaut | effort |
|---|---|---|
| QUA-50, CON-03 | panique au chargement, moteur en `Loading` à vie (3.1) | S |
| BUG-50, CON-02 | chaîne ratée, caches en avance sur `ids` (3.2) | S |
| TST-01 | branche 4E de `answer_fed` sans test (3.4) | M |
| CON-01 | client qui ne lit plus, pool gelé sous `LLVQ_BATCH` (3.3) | M |
| BUG-51 | `reserve` avant le refus de capacité au-delà de 8 704 ; un harnais seulement | S |
| CON-06 | un panic du batch coupe les flux sans `done` ni erreur | S |
| CON-07 | un pas raté ne pose pas `expires` ; un moteur `Failed` garde ses poids | S |
| CON-08 | un `pull` sans effet décharge et recharge le modèle | S |
| CON-10 | `delete` pendant un `pull` du même blob laisse un manifeste sans fichier | S, par `try_lock` |
| CON-04 | erreur de carte collante, moteur resté `ready` | M, sans test CPU possible |
| CON-09 | Ctrl-C n'arrête pas le serveur pendant un flux | M |
| SEC-03 | liste `stop` non bornée, une requête bloque le moteur | S |
| connu | un `Session::feed` raté en milieu de modèle laisse les caches désalignés | S |
| BUG-70 | Metal : la rotation servie se lance à 1 024 threads par groupe sans lire la limite du pipeline (`fused_metal.rs:1208`, `:1305-1309`), alors que le noyau voisin la lit (`:996`). Sur une puce qui annonce moins, erreur de validation ou rotation fausse. PLAUSIBLE : le M3 Max accepte 1 024, rien n'est observé. Prendre le minimum avec la limite ne bouge aucun bit | S |

### Lot 3. Durcissement des lecteurs de fichiers

Seul un fichier corrompu ou forgé déclenche ces défauts. Les fichiers publiés, épinglés par sha256, ne sont pas
touchés. Chaque correction est un refus de plus, sans changer un octet du format.

- **`read_raw` ne valide pas un tenseur quantifié** (QUA-01, NUM-04, SEC-06, trouvé trois fois). `bits` est lu par
  `as u8`, `group = 0` est accepté, les longueurs ne sont pas comparées à `dims`. `to_f32` panique. Le vérificateur
  corrige un point : les chemins servis ne sont pas protégés ailleurs, les normes passent par `to_f32` sans contrôle
  (`fused_cuda.rs:4040`, `fused_cpu.rs:400`, `fused_metal.rs:2263`).
- **Le gain n'est pas borné par le nombre de centroïdes** (QUA-02, NUM-05). Ball seulement.
- **Les écrivains ne vérifient pas la longueur de la queue** (QUA-03). Le 4B a des queues : 2 560 = 24 × 106 + 16.
- **Le gather d'embedding ne borne pas l'id de token** (SEC-07). Un contrôle hôte au chargement suffit, sans toucher
  au noyau.
- **Aucun lecteur Python ne lit `llvq_layout_version`** (PY-02). Préventif : une seule version existe.
- Aussi : QUA-04, QUA-05, QUA-12, NUM-06, TST-03, SEC-02, PY-10, PY-12.

### Lot 4. Outillage payant et CI

- **`run.py`** : `dequant` lance le job avant d'annoncer son coût (PY-06). `oracle` n'annonce aucun coût (PY-07).
  `launch` calcule un timeout en heures entières qui peut tomber sous l'estimation (PY-03). `launch --calib` ne propose
  pas `dclm-edu` (PY-08). Aucun chemin n'écrit la ligne de `jobs.csv` (PY-18).
- **Jeton HF** : `bench` passe `HF_TOKEN` à toute image, dont trois jobs sur une étiquette mobile tierce (SEC-01).
- **Provenance** : 26 scripts sur 133 lancent l'image flottante sans épingler le sha du Space (PY-04).
- **CI** : `forbid(unsafe_code)` ne couvre pas 98 cibles hors `lib.rs` ; aucune ne contient d'`unsafe` aujourd'hui, et
  un `[lints.rust]` par paquet suffit (TST-07). `paths-ignore` exclut `docs/ENV.md` et `configs/README.md` que deux
  tests lisent (TST-08, COH-06). Aucun test Python ne tourne en CI.
- Aussi : PY-05, PY-11, PY-13, PY-15, PY-16, PY-19, SEC-08, SEC-10, OPT-06, QUA-13.

### Lot 5. Décisions d'opérateur

Le code est simple. Le choix ne l'est pas, parce qu'il change un chiffre ou un instrument.

| ID | décision à prendre |
|---|---|
| NUM-01 | Compter le pas de ligne dans le b/param et republier 2,74 au 4B (3.5). |
| BUG-01 | Égalité de logits dans MMLU : le harnais retient la dernière lettre. Sur le 4B publié, 63,374 contre 63,381 avec la première (*computed*, 40 égalités). Aucun document ne dit suivre `lm-eval` sur ce point. |
| BUG-02 | `bin/mmlu` sous la config chaîne score le préremplissage groupé et imprime « prefill chain ». Refuser, ou corriger le libellé. Aucun chiffre publié n'en vient. |
| BUG-09 | `mmlu_split` lit une réponse texte vide comme « A ». |
| CON-05 | Le memo réutilise les ids générés, bloc `<think>` compris : le même historique donne deux prompts différents. C'est le choix écrit en tête de `chat.rs`. |
| CON-11, SEC-04 | Le serveur est mono-utilisateur de fait. Le dire dans la doc, ou masquer `prompt_eval_cached_count` et avertir hors boucle locale. |
| COH-05 | `pyproject.toml` de `llvq-tetra` est en 0.2.0, les docs disent 0.1.0 sur PyPI. L'état de PyPI est à dire par l'opérateur. |
| COH-11 | Six liens de journal cassés dans `docs/data/jobs.csv`. |
| `CLAUDE.md` | Six écarts avec le code. Le texte de remplacement est prêt dans `audit-code-2026-11-18-brut/31-claude-md-patch-propose.md`. Je n'ai pas modifié ce fichier. |
| OPT-03 | Mettre en cache le préfixe 5-shot de MMLU : 2,9 à 3,7 $ gagnés sur un job d'ablation à 5,26 $ (*estimé*), mais c'est un nouvel instrument à valider par `mmlupair`. |
| NUM-03 | `faer` et l'algèbre maison sont dits identiques au bit. Le test exige 1e-8 relatif. Rien n'établit qu'un fichier scellé se reproduit au bit sans `fast-linalg`. Mesurer, ou réécrire la phrase. |
| SEC-11 | La règle « `unsafe` aux seules frontières matérielles » ne nomme pas trois catégories que le code utilise. |

### Lot 6. Architecture et dette

L'audit du 2026-11-07 reste vrai et la dette a grossi : `fusedrun.rs` est passé de 3 314 à 4 087 lignes, `model.rs` de
4 303 à 5 402, l'image CUDA de 22 à 30 binaires. Le suivi ligne par ligne est en tête de `01-architecture.md`.

**La liste des fichiers épinglés est plus longue que les « trois empreintes » de l'ancien audit.** Le préremplissage
en chaîne a ajouté deux empreintes servies (`ce1c354d`, `19b734a3`). Elles couvrent neuf fichiers `pers_4et*` et une
ancre dans `llvq_pers.cuh`. À l'inverse, 3 915 lignes de bras hors chaîne sont libres (ARC-01).

- **Le chemin servi et les bras de mesure partagent les mêmes fonctions** : `install_inner` fait 507 lignes
  (QUA-52), la table des 28 formes d'arbre est écrite à la main en huit endroits (QUA-51, ARC-03), le `Decoder` servi
  porte les portes de mesure (ARC-07, ARC-09).
- **Le 4E et le batch servis contournent les traits de `device.rs`** par 11 `as_any` (ARC-04).
- **Dépendances** : sur macOS, `llvq-llm` dépend de `llvq-cuda` pour un tireur de recherche (ARC-05). Le binaire
  `llvq` compile parquet et zstd sans s'en servir (ARC-06, OPT-07).
- **Le chargement de la chaîne compile un module d'arbre dont il ne lance pas le symbole** (ARC-02, QUA-57). La
  correction n'est pas simple : ce module alloue aussi les tampons de l'arbre, et trois scripts de porte lisent sa
  ligne.
- **Duplication** : QUA-06 (1 324 lignes dans les exemples F1), QUA-54, QUA-55, QUA-56, ARC-08, ARC-10.
- **Code mort** : QUA-09, QUA-60, PY-22. **Ne pas supprimer `tetrascales`** : l'ancien audit le classe mort, mais il
  est la seule provenance de `Encoder::ALPHA` et `RATIO` du codeur Tetra servi (QUA-07).
- Aussi : ARC-11, TST-06, SEC-09.

### Lot 7. Optimisation hors noyaux

Rien n'est mesuré : tous les gains sont *estimés* par lecture. Aucune piste de noyau, aucune piste close.

| ID | piste | gain estimé |
|---|---|---|
| OPT-01 | Étendre le cache de cubins au module 4E, à l'unité fusée et à l'attention, recompilés par NVRTC à chaque démarrage | 35 à 45 s par démarrage de `llvq` |
| OPT-02 | `Sampler::pick` trie les 151 936 logits à chaque jeton à température positive | 2 à 5 ms par jeton sur Metal et CPU |
| OPT-04 | Accès borné élément par élément dans la boucle interne de la propagation GPTQ | 3 à 10 % de l'encodage |
| OPT-05 | Découpe statique des lignes dans GPTQ parallèle | non chiffré |
| OPT-07 | parquet, zstd, brotli compilés dans `llvq` sans utilité | temps de compilation |

## 6. Lot 1, appliqué

Le lot 1 est appliqué sur la branche `audit/code-2026-11-18`, dans le worktree `.claude/worktrees/audit-code`. **Rien
n'est commité ni poussé.** Quatre agents ont travaillé sur des fichiers disjoints ; chaque constat a été revérifié
avant correction. 43 constats sont corrigés, dont six en partie, dans 48 fichiers.

| domaine | corrigés | fichiers | journal |
|---|---|---|---|
| Documents vivants | 9 | `configs/README.md`, `llvq-runtime/README.md`, `docs/ENV.md`, `docs/ETAT.md` (date du titre), `docs/audit-architecture-2026-11.md` | `30-lot1-docs.md` |
| Commentaires Rust | 14 | 19 fichiers de `llvq-search`, `llvq-quant`, `llvq-artifact`, `llvq-llm`, `llvq-cuda`, `llvq-runtime` | `32-lot1-commentaires.md` |
| Harnais et tests | 13 | `bin/{mmlu,ppl,chat,seal,gsm8k,gsm8kpair,fusedrun}.rs`, `sealed.rs`, `gsm8k.rs`, `rtbits.rs`, trois fichiers de test, deux tests neufs | `33-lot1-harnais-tests.md` |
| Python | 7 | six fichiers de `llvq-tetra`, trois de ses tests, `ops/hf_cleanroom_check.py` | `34-lot1-python.md` |

### Ce qui change de comportement

Tout le reste est du texte, des commentaires ou des tests. Ces changements-ci sont des refus nouveaux. Une commande
valide fait la même chose qu'avant.

- **`bin/mmlu`, `bin/ppl`** refusent un positionnel illisible, vide ou nul au lieu de prendre le défaut (BUG-06).
  `limit=4O` lançait un recensement complet payé. `ppl` refuse aussi un corpus autre que `wikitext2` ou `c4`, qui
  était lu en silence comme `wikitext2`.
- **`bin/seal`** exige `LLVQ_MODEL` et refuse un `.state` qui nomme un autre modèle (BUG-05). Le défaut Qwen3-0.6B
  disparaît. Les quatre appels de `ops/jobs/` posent déjà la variable.
- **`bin/fusedrun`** lit la liste unique `served::MEASUREMENT_MODES` au lieu d'une copie de 17 noms (QUA-53). Effet :
  `LLVQ_TIME_EVENTS` et `LLVQ_BENCH_ARMS` sont refusés à côté d'une config, comme la doc le dit déjà.
  `LLVQ_CONTEXTS` et `LLVQ_PREFILL_TOKENS` ne sont pas dans la liste : les portes à prompt long ne changent pas.
- **`bin/chat`** lit la graine et `max_new` en entier, plus en f32 (BUG-07).
- **`bin/mmlu`** imprime le `kv` de la config sur le bras servi (BUG-10). Format de la ligne inchangé.
- **`gsm8kpair`** imprime `config` et `prefill` (BUG-03). **`RestoreF16::describe`** écrit la fenêtre de couches,
  `down_proj@0-11` (BUG-04). Libellés seulement ; un dump ancien se lit et s'apparie comme avant.
- **`rtbits`** : la ligne « f32 (published) » s'appelle « f32 » (NUM-02). Aucun calcul touché, le pas de ligne de
  NUM-01 n'est pas corrigé.
- **Un test d'archive** échoue en nommant le fichier au lieu de sauter (TST-04, règle 10).
- **`llvq-tetra`** refuse par nom : `LLVQ_HF_TILE` au-dessus de 64 (PY-01), un `device_map` qui nomme une autre carte
  que la carte courante (PY-09), un `quantization_config` sans `tetra_fingerprint` ou `block_dim` (PY-17). Quatre
  gardes passent de `assert` à `raise` (PY-20). Ces changements ne partent qu'à la prochaine version du paquet.
- **`ops/hf_cleanroom_check.py`** refuse deux arguments et dit quand la porte du 4B n'est pas lancée (PY-14).

Une correction de la session principale : l'agent Python refusait toute valeur de `LLVQ_HF_TILE` différente de 64, donc
aussi 32, qui donne une sortie juste. Le refus ne vise plus que les valeurs au-dessus de 64, celles qui débordent.

### Tests nouveaux ou renforcés

`a_malformed_limit_is_refused_not_read_as_the_census`, `a_malformed_positional_is_refused_not_defaulted`,
`the_checkpoint_is_required_and_must_be_the_encoded_one`, `a_restored_window_is_named_in_the_label`,
`served_read_env.rs` (TST-02, partie `llvq-llm`), `softmax_whole.rs` (QUA-58), le refus mutuel des deux variables
`RESTORE` par `from_lookup` (TST-05), `keep_alive_reads_ollamas_forms` (TST-11), `the_shipped_config...` aligné sur
l'objet servi (TST-13), et cinq tests pytest. Chaque test a son mutant : rouge sous le mutant, vert une fois défait.
20 mutants au total, tous tués.

### Contrôle

| contrôle | résultat |
|---|---|
| `cargo test --workspace`, debug | 227 lignes `test result`, toutes `ok` ; 1 416 tests passés, 0 échec, 143 ignorés (*mesuré*) |
| `cargo clippy --workspace --all-targets` | 0 avertissement |
| `uv run --group dev pytest`, `llvq-tetra` | 64 passés |
| Moitié Linux/CUDA, `ops/check-linux-cuda.sh clippy` | `-p llvq-llm --features cuda --all-targets` et `-p llvq-cuda --all-targets` : 0 erreur, 0 avertissement |
| Diff de l'agent « commentaires » | filtré : seulement des lignes de commentaire, plus deux raisons d'`#[ignore]` et une assertion de test |

La suite complète en release n'a pas tourné : le lot 1 ne touche ni format ni index. Aucun des tests Metal ou CUDA n'a
tourné.

### Laissé de côté dans le lot 1

- **`CLAUDE.md`** : non modifié. Le patch proposé, 12 blocs « texte actuel / texte proposé », est dans
  `31-claude-md-patch-propose.md`.
- **`docs/HISTORIQUE.md`** : non modifié. Onze faits remplacés dans les documents vivants sont listés dans
  `30-lot1-docs.md`, section « À verser dans HISTORIQUE.md ».
- **BUG-01, BUG-02, BUG-09, NUM-01** : instruments ou chiffres, lot 5.
- **Commentaires dans des fichiers épinglés** (`pers_4e.cu`, `pers_4et.cu`, `pers_attn_full.cuh`, `llvq_rot.metal`) :
  ils attendent le prochain déplacement d'empreinte.
- **Parties non faites** : TST-02 côté runtime, QUA-59 dans `served_decode.rs`, TST-10 dans `test_fused.py`, la doc de
  `served.rs:702-705` (elle dit qu'aucun test ne doit appeler `Served::read`, ce que `served_read_env.rs` fait
  maintenant, seul dans son binaire).
- **COH-05** : l'état de la 0.2.0 sur PyPI est à dire par l'opérateur.

## 7. Lots 2 à 7 : écrits sans compilateur, vérifiés ensuite

Le 2026-11-19, à la demande de l'opérateur et sans rien exécuter sur sa machine, six agents ont écrit les corrections
des lots 2, 3, 4 et 7. **Ces lots ont été écrits sans compilateur.** Ils ont été compilés et testés ensuite : voir la section 8.6. L'état vérifié
du lot 1 est gardé dans `lot1.patch` ; l'arbre entier est dans `lots-1-a-7.patch`. La file des commandes et des mutants
à lancer est dans `50-file-de-verification.md`.

| lot | fait | journal | ce qui reste avant d'y croire |
|---|---|---|---|
| 2, runtime | QUA-50/CON-03, BUG-50/CON-02 côté moteur, CON-06, CON-07 partie 1, CON-08, CON-10, SEC-03 ; 7 tests | `40-lot2-runtime.md` | compilation, tests CPU, contrôle croisé Linux |
| 2 et 3, session et hôte | TST-01 (boucle 4E déplacée telle quelle dans `Session::answer_launches`), BUG-50 par la garde de `truncate`, BUG-51, BUG-70, SEC-07 ; 4 tests | `41-lot2-session-hote.md` | idem, puis une porte sur carte : `session.rs` et `persist4e_cuda.rs` sont le chemin servi |
| 3, format | QUA-01/NUM-04/SEC-06, QUA-02/NUM-05, QUA-03, QUA-12, NUM-06, et NUM-01 côté code ; 16 tests | `42-lot3-format.md` | suite complète en release avec les fichiers scellés : les refus doivent laisser passer les trois fichiers publiés |
| 3, Python | PY-02 (version), PY-10 ; 8 tests | `44-lot3-python.md` | pytest |
| 4, outillage | PY-03, 05, 06, 07, 08, 11, 13, 16, SEC-01, TST-07, TST-08/COH-06 ; 3 tests | `43-lot4-outillage.md` | le banc sans coût de `run.py`, clippy, une PR pour la CI |
| 7, optimisation | OPT-02, OPT-04 ; 2 tests d'identité | `45-lot7-optimisation.md` | les tests d'identité, puis une mesure avant et après |

Relectures de la session principale, sur les diffs :
- `session.rs` : l'ancien et le nouveau corps de la boucle 4E sont identiques ligne par ligne ; le cas `max_new == 0`
  reste traité avant.
- `engine.rs` : le chargement sous `catch_unwind` retombe sur le chemin d'erreur existant ; le chemin de succès de
  `feed_with` est inchangé.
- `ops/run.py` : chaque chemin de succès est gardé ; le timeout de `launch` n'est jamais plus court qu'avant.

Points à trancher par l'opérateur avant de garder ces corrections :
- **SEC-01.** `bench` et `dequant` ne passent plus `HF_TOKEN` à une image qui n'est pas `llvq-runner*`. Si un job sur
  image tierce a besoin du jeton, il échouera après facturation. Non mesuré.
- **OPT-02.** Le tri actuel de `Sampler::pick` est instable : le raccourci ne s'applique que quand il démontre le même
  vecteur au bit, et retombe sur l'ancien code sinon. Gain probablement nul à haute température et sur des logits f16.
  37 lignes de preuve pour un gain non mesuré : à garder seulement si la mesure le justifie.
- **Ball à 0 centroïde.** Un enregistrement dont tous les blocs sont l'origine se décodait ; il est maintenant refusé.
- **PY-10.** `forward` d'un module fusé refuse une entrée posée sur un autre appareil.
- **NUM-01.** `rtbits` imprime maintenant les deux comptes. Quelle ligne les documents citent reste à décider.
- **Décisions préparées, non prises** : CON-01 (délai d'un client lent, 30 s recommandé), CON-04, CON-05, CON-09
  (10 s puis sortie au second Ctrl-C), CON-11, BUG-01, BUG-02, BUG-09, PY-04, PY-15, PY-18, PY-19, SEC-08, SEC-10.
  Chaque journal a sa section « Décision d'opérateur ».

Non fait :
- **Lot 6, architecture.** Découper `pers.rs` ou `install_inner` sans compilateur serait imprudent, et plusieurs points
  sont des choix de structure. QUA-04 et QUA-05 ont un plan dans `42-lot3-format.md`.
- **OPT-01** (cache de cubins pour 4E) : plan et mesure sur carte dans `45-lot7-optimisation.md`.
- **Lot 5** : aucune décision prise à la place de l'opérateur.

Une correction au constat PY sur la CI : un job Python existe déjà pour `llvqtune`. Seul `llvq-tetra` n'en a pas.

## 8. Second tour, 2026-11-19 et 2026-11-20

L'opérateur a demandé un second audit après les corrections, avec remédiation directe. Six relecteurs ont relu, deux
le code neuf des lots 1 à 7, quatre ce que le premier tour n'avait pas lu. Constats bruts : `60-` à `65-`.

### 8.1 Bilan

52 constats : 0 critique, 3 majeures annoncées, 49 mineures ou cosmétiques. Après vérification il reste deux majeures,
toutes deux sur Metal.

| relecteur | majeures | mineures | cosmétiques | fichier |
|---|---|---|---|---|
| Code neuf, runtime et moteur | 0 | 5 | 1 | `60-r2-diff-runtime.md` |
| Code neuf, format et outillage | 0 | 6 | 1 | `61-r2-diff-format-outillage.md` |
| Crates cœur non lues | 0 | 2 | 1 | `62-r2-coeur.md` |
| Moteur `llvq-llm` non lu | 0 | 2 | 3 | `63-r2-llm.md` |
| Metal | 2 | 10 | 2 | `64-r2-metal.md` |
| Protocole et Python | 0 après vérification | 13 | 4 | `65-r2-api-python.md` |

Ce qui sort sain à la seconde lecture :
- **Le chemin Tetra servi** : mot de 48 bits, table, treillis, encodeur, fenêtre de lecture du dernier bloc, empreinte.
- **Aucun refus neuf ne rejette un fichier valide.** `write_raw` refusait déjà ce que `read_raw` refuse maintenant. Vérifié
  par la lecture sur les formes du 4B, du 8B, du 14B et du 0.6B.
- **GPTQ par tranche** : identique au bit à l'ancienne boucle.
- **`llvq-llm`** : transcodage vers la carte, tailles des tampons, hôte du batch, calibration et reprise sur shard.
- **Metal** : la correction de la rotation (BUG-70) est juste et complète ; le mur int4 à `d_in` 8 192 n'a aucun chemin
  silencieux ; `grouped` et `kernel` ne sont atteints par aucune porte servie.

### 8.2 Deux régressions introduites par le premier tour, levées

Le premier tour avait écrit ses lots 2 à 7 sans compilateur. La seconde lecture y a trouvé deux régressions, toutes deux
corrigées :
- **R2-DIF-40.** `torch.cuda.set_device(k)` puis `device_map="cuda"` était refusé par `llvq-tetra`. `transformers`
  réécrit `"cuda"` en `cuda:0` avant que le paquet ne le voie. L'index ambigu est de nouveau accepté, comme à `925930d`.
- **R2-DIF-41.** `run.py oracle` refusait `a100-large` même avec l'image sm80. Il reçoit `--kernels-cap`, comme `bench`.

Une troisième était en germe : `same_model` ignorait `modified_at`, donc un moteur chargé depuis un blob abîmé restait
servi après réparation (R2-DIF-01). Le moteur n'est repris que si le blob sur disque est le même fichier.

### 8.3 Les deux majeures Metal

- **R2-MTL-02, corrigé.** Aucun hôte ne vérifiait que le SIMD-group fait 32 voies, que les shaders supposent partout.
  `MetalRuntime::pipeline` refuse par nom un pipeline d'une autre largeur. PLAUSIBLE seulement : sur Apple Silicon la
  largeur vaut 32.
- **R2-MTL-01, non corrigé, décision d'opérateur.** `candle` 0.9.2 ne relit pas le statut d'une commande GPU après
  l'attente (`candle-metal-kernels-0.9.2/src/metal/commands.rs:239-259`). Sur une erreur de carte, le chemin Metal servi
  lirait des logits périmés et rendrait un texte faux au lieu d'une erreur. Aucun contrôle côté hôte n'est possible :
  les tampons de commandes sont privés à `candle`. Options : vendoriser par `[patch.crates-io]`, patch amont.

### 8.4 Les échelles par ligne (R2-API-01), ramené de majeure à mineure

Le fait est exact. Les trois décodeurs appliquent l'échelle de ligne aux blocs seulement, et la queue est stockée en
base tournée. `llvqtune` multiplie toute la ligne en base naturelle. L'écart vaut √(2t/d) de l'édition voulue : 11,2 %
au 4B, 8,8 % au 8B, 3 à 6 % au 14B (*computed*, poids isotropes ; `70-verif-r2-rowscales.md`).

Ce n'est pas une découverte : la déviation E8 du prereg du 2026-09-19 l'écrit et l'assume. L'effet sur l'optimum de σ
est du second ordre, environ 0,04 point de MMLU au 4B, et 0,35 / 0,29 / 0,09 point en lecture pessimiste, sous le bruit
de 0,43. Aucun chiffre publié ne change : ce sont des mesures du fichier plié.

Ce qui est neuf et reste à écrire à côté des preregs, par l'opérateur :
- le « 1,2e-4, at most » de E8 multiplie une amplitude par une énergie ; il sous-estime l'écart d'un facteur 18 ;
- la déviation E8 manque aux bras 8B et 14B ;
- trois docstrings disent le contraire du code, et `test_the_fold_is_the_trained_function` ne peut pas voir l'écart.

Pour le chiffrer : la KL au professeur de trois élèves (base, entraîné, plié), à 0 $ et environ une heure sur le Mac,
avant tout réentraînement à 4,72 $. Non lancé.

### 8.5 Corrections du second tour

35 constats corrigés par quatre agents, journaux `80-` à `83-`. Aucun n'a été trouvé faux à la revérification.

| domaine | corrigé | journal |
|---|---|---|
| Runtime | R2-DIF-01, 02, 05, 06 ; R2-API-09 et 15 en partie, 10, 11 | `80-r2-fix-runtime.md` |
| `llvq-llm` et Metal | R2-DIF-03, 04 ; R2-LLM-02, 03, 04, 05 ; R2-MTL-02, 03, 04, 05, 06 | `81-r2-fix-llm-metal.md` |
| Crates cœur | R2-SRC-01, 02, 03 ; R2-DIF-42, 46 | `82-r2-fix-coeur.md` |
| Python et outillage | R2-DIF-40, 41, 43, 44, 45 ; R2-MTL-09 ; R2-API-03, 05, 07, 16, 17 | `83-r2-fix-python-ops.md` |

### 8.6 Vérification de tout l'arbre

Les lots 2 à 7, écrits sans compilateur, n'avaient qu'une erreur de compilation : un type entier ambigu dans un test de
chronométrage. Le second tour compile du premier coup. Trois tests Python neufs demandaient un échauffement plus long
que leur run ; corrigés.

| contrôle | résultat (*mesuré*, 2026-11-19 et 2026-11-20) |
|---|---|
| `cargo check --workspace --all-targets` | 0 erreur, 0 avertissement |
| `cargo test --workspace`, debug | 229 lignes `test result`, toutes `ok` ; 1 461 passés, 0 échec, 146 ignorés |
| `cargo clippy --workspace --all-targets` | 0 avertissement |
| `ops/check-linux-cuda.sh clippy`, `llvq-llm --features cuda`, `llvq-cuda`, `llvq-runtime` | 0 erreur, 0 avertissement |
| `cargo test -p llvq-llm --features metal`, sur le GPU du M3 Max | 91 lignes `test result`, toutes `ok` ; 647 passés, 0 échec |
| `cargo test -p llvq-quant --features fast-linalg` | 97 passés |
| pytest `llvq-tetra` | 88 passés |
| pytest `ops/llvqtune` | 139 passés |
| `ops/mlx_tetra/test_served.py`, `ops/check_image_lists.py` | 11 sur 11 ; listes concordantes |
| Mutants | 69 joués, 65 tués, 4 survivants tous annoncés (`52-mutants-joues.md`) ; arbre revenu à l'identique |
| Suite complète en release, tests ignorés compris, archive et trois fichiers scellés présents | 219 binaires de test, tous `ok` ; 1 603 passés, 0 échec, 0 ignoré. Trois binaires étaient rouges au premier passage, pour trois pannes antérieures à l'audit : voir 8.9 |
| `cargo test --release -p llvq-llm --features metal -- --include-ignored`, sur le GPU du M3 Max | 91 lignes `test result`, toutes `ok` ; 671 passés, 0 échec, 0 ignoré |
| `rtbits` sur les trois fichiers scellés, lecteur durci | les trois fichiers sont lus sans refus. Modèle entier, embedding q4, queue f16 : compte du fichier 2,7320 / 2,6953 / 2,7305 b/param, identique au publié ; compte de la carte, pas de ligne tetra48 compris, 2,7393 / 2,7000 / 2,7324. Le 8B *calculé* de la section 3.5 (2,7001) se lit 2,7000 *mesuré* |

Les quatre mutants survivants, connus : le seul `memo.clear()` après une erreur de préremplissage ; la boucle sur
`pending` du bras de panique du batch ; le 409 de `rm` pendant un `pull`, sans test de route ; la borne
`PREALLOC_CAP`, qui survit sur macOS et se rejoue sous Linux.

Rien n'a tourné sur une carte NVIDIA. `session.rs`, `persist4e_cuda.rs` et les chargeurs sont le chemin servi : une
porte sur carte reste due avant de servir ces corrections.

### 8.7 L'échantillonneur (OPT-02), gardé sur mesure

| distribution, 151 936 logits, temp 0,7, top_p 0,9 | ancien `pick` | nouveau | rapport |
|---|---|---|---|
| six têtes | 1 987,6 µs | 495,8 µs | ×4,0 |
| six têtes, mantisse f16 | 1 652,4 µs | 488,3 µs | ×3,4 |
| plate | 2 173,5 µs | 3 044,4 µs | ×0,71 |

Médianes de 9 tours de 64 tirages, *mesuré* sur le Mac par `pick_times_against_the_sorted_reference`. Logits
synthétiques. Le gain vaut pour le chemin Metal Rust et le CPU à température positive, pas pour CUDA où 4E est greedy.
Environ 1,5 ms de moins par jeton, soit de 52,1 à environ 56 tok/s sur Metal (*estimé*, non mesuré de bout en bout).
Sur une distribution plate le nouveau code perd 0,9 ms.

### 8.8 Décisions d'opérateur ouvertes après le second tour

| ID | décision |
|---|---|
| R2-MTL-01 | Vendoriser `candle` pour relire le statut d'une commande Metal, ou attendre un patch amont. |
| R2-LLM-01 | Sous `LLVQ_BATCH`, le pool préremplit par 256 et la session par 4 096 : un prompt de 257 à 264 jetons n'a pas la même attention selon qu'une autre requête est en cours. Correction d'une ligne, qui change des jetons servis sous `LLVQ_BATCH`. Aucun chiffre publié ne bouge : le plus long prompt GSM8K du batch fait 215 jetons. |
| R2-API-01, 02, 04, 06 | L'instrument d'entraînement des échelles : base de σ, arrondi bf16, gradients calculés pour rien, professeur lu à `main`. |
| R2-API-11 | `deny_unknown_fields` sur le store : un binaire ancien ne lit plus un store récent. Recommandation : un champ `schema` refusé par nom s'il est plus récent. |
| R2-API-09 | `tool_calls` dans l'historique : refuser en 400 ou avertir. |
| R2-API-14 | Sauts de ligne en tête et en fin de `thinking`. |
| SEC-01, R2-DIF-45 | Restreindre le jeton HF à l'espace de noms de l'opérateur. |
| `llvqtune` | Un run plus court que son plan sort maintenant en 6 ; à nommer dans le prochain prereg qui appelle `train.sh`. |
| neuf | Sur Metal, `model.float()` convertirait en f32 les tables f16 de `Int4Vocab`, que le shader lit en `half`. Non corrigé, plan dans `83-r2-fix-python-ops.md`. |

Non corrigés et sans décision à prendre : R2-MTL-07 (MLX n'expose pas la limite de threads), R2-MTL-08, R2-MTL-10 à
14, R2-API-08, 12, 13, les commentaires faux dans des shaders épinglés.

### 8.9 Trois pannes que la suite release a montrées, antérieures à l'audit

La commande de `CLAUDE.md`, `cargo test --release -- --include-ignored`, était rouge sur `main` à `925930d`. Les trois
causes sont dans du code que l'audit n'avait pas touché. Elles sont corrigées, et la suite est verte.

| panne | cause | correction | preuve |
|---|---|---|---|
| `llvq-llm/tests/dflash_prefill.rs` sans `--features metal` | la boucle sur f32, bf16, f16 date de SP2 et n'a tourné que sur Metal. Sur CPU, candle n'a pas de matmul bf16, et son f16 s'écarte de la fixture MLX sur des quasi-égalités (5/15 et 3/15 jetons en tête) | le passage CPU teste f32 seul, ce que l'en-tête du fichier annonçait déjà. Metal garde les trois types | vert sur CPU et sur Metal |
| `llvq-metal/tests/rot_matches_host.rs`, `the_metal_rotation_is_the_cuda_rotation_bit_for_bit` | le pilote `host_rotate.cpp` écrit 5 largeurs, 3 résidus, le vecteur de `rotate_wide.cu` et 5 drapeaux depuis `22ad8e9`. Le test attendait encore 4 largeurs et 3 résidus : 396 octets au lieu de 608 à n = 24. Il est ignoré en debug, donc personne ne l'a vu | la longueur attendue suit le contrat du pilote. Aucun fichier de noyau touché | vert, identité au bit gardée |
| `llvq-runtime`, client parti pendant un silence | le moteur ne remarquait un client parti qu'en échouant un envoi. Quand un jeton ne donne rien à envoyer (séquence UTF-8 inachevée, préfixe d'une chaîne d'arrêt retenu), rien n'échoue. Le modèle de test a un silence de 0,27 à 0,52 s ; la requête suivante attendait 200 à 380 ms (*mesuré*). Le test existant ne le voyait que si le départ tombait dans le silence, ce qui arrive sous `RAYON_NUM_THREADS=4` et pas sans | `engine.rs` interroge le canal (`is_closed`) quand un jeton n'envoie rien, aux trois sites : moteur simple, réponse solo du batch, ligne du batch. Un client qui reste ne voit aucun changement | attente ramenée à 0,4 à 1,3 ms (*mesuré*). Test neuf `a_client_that_leaves_during_a_silence_stops_the_engine`, qui mesure le silence puis y fait partir le client. Mutant sur le site du moteur simple : tué aux deux vitesses |

Limites. Les deux sites du batch ont la même correction sans test propre. Sur un vrai modèle les silences durent
quelques jetons : le gain en service est faible, c'est la tenue du test qui change. Cette correction est sur le chemin
servi du runtime, donc sous la même porte sur carte que le reste.

Après ces corrections : `cargo clippy --workspace --all-targets` 0 avertissement, contrôle croisé Linux de
`llvq-runtime` 0, `llvq-runtime` et `llvq-metal` verts en debug (127 passés) et en release (131 passés), `llvq-runtime`
vert aussi sous `RAYON_NUM_THREADS=4` (63 passés). Tout est *mesuré* le 2026-11-20.

## 9. Ce que l'audit n'a pas couvert

- **Rien n'a tourné sur une carte.** Les défauts qui dépendent d'un ordonnancement GPU restent à la lecture.
- **Noyaux** : les deux passes ont lu les noyaux servis (4E, chaîne `a12ipg32ft`, GEMM du batch, rotation, décodeurs
  Tetra, embedding int4). Restent les corps d'arbre non servis (`a4` à `a11`, `pers_4et_stream`), `llvq_tetra48.metal`
  et `tv_q4_h.metal`. Trois commentaires faux sont dans des fichiers épinglés (BUG-52 en partie, BUG-71, BUG-72) : à
  corriger au prochain déplacement d'empreinte, pas avant.
- **Crates cœur** : `llvq-search` (`fastdec.rs`, `cns.rs`, le chercheur Ball), `llvq-artifact` (`e1c.rs`, `e1v.rs`,
  `runtime.rs`), la plupart de `llvq-bench` et les tests d'intégration.
- **`llvq-llm`** : le corps de `fused.rs`, `fused_cuda.rs`, `fused_metal.rs`, `calib.rs`, `speculative.rs`,
  `dflash*.rs`, `kvladder.rs` et les bancs.
- **`llvq-metal`** en entier.
- **Python** : `ops/llvqtune` (adaptateurs, `trainables`), les scripts MLX, 153 des 163 scripts de job.
- **Les chiffres de mesure contre leurs journaux** : hors périmètre.
