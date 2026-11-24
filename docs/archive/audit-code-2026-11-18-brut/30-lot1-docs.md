# Lot 1, documents vivants : journal

Base 925930d, branche `audit/code-2026-11-18`. Markdown seulement, rien de commité. Chaque phrase écrite a été relue
contre le code cité dans la colonne « vérification ». Aucun chiffre de mesure touché, aucun « 2.73 » changé.

## Corrigé

| ID | fichiers | ce qui change, en une ligne | vérification faite |
|---|---|---|---|
| COH-01 (`configs/README.md:55`) | `configs/README.md` | la ligne `metal-kernel-NOT-SERVED` dit : sur Metal seul `metalrun` la charge, sur CUDA rien ne la refuse, elle se résout comme `-kernel-from1280.json` et sa note NOT SERVED s'imprime. `:168` laissé tel quel (juste selon 21-verif) | `served.rs:942` `(_, false) => Ok(())` ; champs des deux JSON identiques hors `note` (script) ; `served.rs:854`, `:893-894` impriment la note |
| COH-02 | `llvq-runtime/README.md` | colonne CUDA du 4B : 4E v9 après le préremplissage en chaîne, `LLVQ_PROFILE=eager` pour l'eager ; § Limits : `LLVQ_BATCH=<n>[:<positions>]`, configs `*-4e-v9-batch.json`, requête seule sur 4E, glouton, pas de cache de conversation, refus sur Metal et CPU ; `LLVQ_MODULE_CACHE` et son défaut | `library.json` (`cuda` 4B = chain, `cuda_eager`, `cuda_batch`) ; `library.rs:97-130` `config_for` ; `engine.rs:8-20`, `:58-73` ; `modcache.rs:174-196` |
| COH-03 | `configs/README.md` | `qwen3-0p6b-tetra-e4-4e-v9.json` ajouté au tableau « At a glance » et au tableau « Every file » (drafter, jamais servi, `specrun`, préreg, route close en ETAT §6) ; b/param « not counted here », aucun chiffre inventé | le JSON et sa note ; `specrun.rs:1-8` ; `proofs/preregistration-draft-0p6b-2026-11-07.md` existe ; `served.rs:540-545` ; `docs/plan-speculation.md:27-31` |
| COH-04 (`:43`, `:132`) | `configs/README.md` | « on any shape but the 4B's » devient « the sealed 4B, 8B and 14B and the 0.6B drafter » aux deux endroits ; la contradiction de `:132` / `:134` disparaît | `served.rs:529-554` (`SEALED`, 4 formes) |
| COH-07 | `configs/README.md` | « five required fields and four optional ones » devient « five required, seven optional, and a free-text `note` » ; « Three are optional » devient les sept avec leur date ; la liste des parseurs ajoute `PrefillAttn::parse`, `Prefill::parse`, `DecodeCache::parse` et `BatchServe::of` ; le titre « Why every value is a string » devient « How the values are parsed », car `batch` est un objet de deux nombres | `served.rs:94-150` (`ServedFile`), `:154-162` (`BatchFile`), `:184-192`, `:714-830` ; seuls les cinq requis passent par le parseur des variables d'environnement (`git grep` des sept parseurs hors `served.rs`) ; aucun lien vers l'ancien titre (`git grep`) |
| COH-13 | `docs/ETAT.md`, `configs/README.md` | titre « as of 2026-11-07 » devient « as of 2026-11-17 » ; « the grouped form the 8B serves » / « the 14B serves » deviennent « served before it » | ETAT l. 22 et 64-65 datent du 2026-11-15 et du 2026-11-17 ; `configs/README.md:22` (v9 servi depuis 2026-11-07) |
| QUA-07 (partie doc) | `docs/audit-architecture-2026-11.md` | `tetrascales` sort de la liste des morts, avec la raison (provenance de `Encoder::ALPHA` et `RATIO`, sortie brute non commitée), et passe au clôturage de B1 ; lot 3 : 8 cibles, 797 lignes ; lot 5 : 53 cibles, 21 547 lignes. Le tableau des catégories garde 9 / 1 027, car `tetrascales` remplit toujours la règle qui définit « dead » | `llvq-search/src/tetra/encoder.rs:49-52` ; `wc -l` des neuf cibles = 1 027, `tetrascales` = 230 ; lot 5 recompté : 62 cibles de B1-B3, B5-B7, moins 8 morts, moins `gbench` = 53 ; 21 317 + 230 = 21 547 |
| QUA-08 (partie `docs/ENV.md`) | `docs/ENV.md` | la ligne `LLVQ_Q4B` (« int4 bit study switch ») quitte « tool » pour le tableau « Tests » : chemin du 4B scellé lu par les tests unitaires de `radixstudy`, défaut `~/llvq-q4b.llvq`, déplace la recherche sans la satisfaire, ne lit pas `LLVQ_SEALED_ARTIFACT`. Le code n'est pas touché | `llvq-bench/src/bin/radixstudy.rs:1517-1524`, `:2471`, `:2526` ; `llvq-artifact/tests/common/mod.rs:18`, `:51-58` |
| CONNU-NON-CORRIGÉ `LLVQ_SEQ_BLOCK` (partie `docs/ENV.md`) | `docs/ENV.md` | « unset or `1` » devient « unset, empty or `0` (default), `1` » | `smoke.rs:1166-1170` |

## Laissé de côté

| ID | raison |
|---|---|
| COH-05 | Consigne : l'état PyPI de `llvq-tetra` 0.2.0 doit venir de l'opérateur. `pyproject.toml` dit 0.2.0, `ETAT.md:253` et `plan-transformers.md` disent seulement 0.1.0 sur PyPI. Rien corrigé. |
| COH-04 (`CLAUDE.md:123`, `docs/ETAT.md:128`) | Hors de ma liste : le vérificateur juge l'oubli du drafter défendable dans ces deux documents (le drafter n'est pas un objet servi). Bloc facultatif n° 2 dans le patch `CLAUDE.md`. |
| COH-01 (`CLAUDE.md:123`), COH-08, COH-12, COH-14, NUM-03, OPT-06, TST-09, CONNU `LLVQ_DTYPE` et `LLVQ_SEQ_BLOCK` dans `CLAUDE.md` | `CLAUDE.md` est interdit : blocs exacts dans `31-claude-md-patch-propose.md`. |
| TST-09 | Aucun document vivant hors `CLAUDE.md` ne liste les boucles de test (`README.md:156` renvoie à `CLAUDE.md`, `METHODE.md` ne cite pas `cargo test`) : la ligne est dans le patch, bloc 10. Les 38 tests Metal n'ont pas été lancés (feature `metal` interdite par le brief). |
| CONNU `LLVQ_DTYPE` | `docs/ENV.md:56` est déjà juste ; reste `CLAUDE.md:131`, dans le patch. |
| CONNU `proofs/README.md:20` (22 préenregistrements) | `proofs/*` est interdit. |
| QUA-07 (reste) | Verser la sortie de `tetrascales` dans `docs/mesures/` est interdit (`docs/mesures/*`), et la sortie brute n'existe plus de façon durable ; la mention du scratchpad est dans `encoder.rs`, du code, hors de mon domaine. |
| QUA-08 (code) | Lire `LLVQ_SEALED_ARTIFACT` dans `radixstudy` touche du code : hors de mon domaine. |

## Constat faux ou déjà corrigé

| ID | preuve |
|---|---|
| NUM-03 (partie `linalg.rs`) | Dans le worktree, `llvq-quant/src/linalg.rs:15-22` dit déjà « agree to rounding, not to the bit, and codes encoded on the two paths may differ » et la tolérance `1e-8 · max(max|U|, 1)` (modifié par un autre agent du lot, pas par moi). Reste `CLAUDE.md:55-56`, bloc 8 du patch. |

## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)

Aucun. Markdown seulement.

## À verser dans HISTORIQUE.md

Faits remplacés dans les documents vivants, avec l'ancienne formulation, pour décision de l'opérateur :

- `docs/ETAT.md` : le titre « Project state as of 2026-11-07 » devient « as of 2026-11-17 ».
- `llvq-runtime/README.md` : colonne CUDA du 4B « 4E v9 (greedy only) » ; la chaîne sert le 4B par défaut sous `llvq` depuis le 2026-11-15.
- `llvq-runtime/README.md` : « The engine decodes one sequence at a time, so requests to one model queue. » ; vrai seulement sans `LLVQ_BATCH` depuis le 2026-11-08.
- `configs/README.md` : `qwen3-4b-tetra-e4-4e.json` « Refused on Metal, on any shape but the 4B's » et § decode « refused on Metal, on any shape but the Qwen3-4B's » ; quatre formes sont acceptées depuis le 2026-11-07.
- `configs/README.md` : « five required fields and four optional ones » et « Three are optional: `attention` … then `decode` and `rows` » ; sept champs optionnels depuis le 2026-11-15 (`prefill_attention` et `batch` le 2026-11-08, `prefill` et `decode_cache` le 2026-11-15).
- `configs/README.md` : « not the grouped form the 8B serves » / « the 14B serves » ; le 8B et le 14B servent v9 depuis le 2026-11-07.
- `configs/README.md` : `metal-kernel-NOT-SERVED.json` « Loaded only by `bin/metalrun` » ; faux sur CUDA depuis l'écriture du fichier, le 2026-11-06.
- `configs/README.md` : titre « Why every value is a string » et liste de huit parseurs ; `batch` (2026-11-08) n'est pas une chaîne.
- `docs/ENV.md` : `LLVQ_Q4B` « int4 bit study switch », rangé en « tool » ; c'est un chemin de test.
- `docs/ENV.md` : `LLVQ_SEQ_BLOCK` « unset or `1` » ; `0` et vide sont aussi acceptés.
- `docs/audit-architecture-2026-11.md` : `tetrascales` dans la liste des neuf cibles mortes à supprimer (lot 3, 9 cibles, 1 027 lignes ; lot 5, 52 cibles, 21 317 lignes).

## Commandes lancées et leur résultat (les lignes `test result:` telles quelles)

`nice -n 10 cargo test -p llvq-llm --test env_registry -j 8` (lit `docs/ENV.md`) :
```
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
```

`nice -n 10 cargo test -p llvq-runtime --lib latest_is_the_4b -j 8` (lit `configs/README.md` par `include_str!`) :
```
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 17 filtered out; finished in 0.00s
```

Contrôle par script : chacun des 14 blocs « texte actuel » de `31-claude-md-patch-propose.md` apparaît exactement une
fois dans `CLAUDE.md`.
