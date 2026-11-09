# Audit des bugs et des mesures, 2026-10-25 au 2026-11-08

Audit du 2026-11-08, à 0 $. Périmètre : `main` (1997edc) et les PR ouvertes #23 à #28. Méthode : six relectures
parallèles (une par PR ouverte, une pour le code fusionné, une pour les mesures), puis chaque trouvaille revérifiée à
la main. Les preuves sont une lecture du code, un recalcul depuis les fichiers bruts, un mutant ou `hf jobs inspect`.
Aucun code n'est corrigé ici. Les trouvailles sont classées *CONFIRMED* (preuve reproduite) ou *PLAUSIBLE* (lecture
seule, ou estimation non mesurée).

Résultat en une ligne : aucun bug ne change les tokens servis ni le b/param. Il en reste trois qui touchent le service :
- 4E se bloque après une seule erreur ;
- une requête restée seule dans le pool batch finit à 0,47× ;
- le store du runtime ne charge plus les modèles tirés avant la v9.

Une colonne publiée est mal étiquetée : les Go sur carte sont *computed*, pas *measured*. Le registre vaut **304,05 $**,
et non 297,59 $.

## 1. Graves : service et chiffres publiés

### 1.1 4E reste empoisonné après un seul watchdog ou un seul NaN. CONFIRMED

- **Où.** `llvq-llm/src/persist4e_cuda.rs:333-334` téléverse `abort` et `nan` à 0 une seule fois, à l'installation.
  Le noyau ne fait que les passer à 1 (`llvq-cuda/kernels/llvq_pers.cuh:140` et `:980`, `pers_4ev.cu:647`). Le début
  de lancement remet à zéro `slots` et les tickets, pas ces deux drapeaux (`pers_4e.cu:228-240`). Aucun code hôte ne les
  remet à zéro. `launch` et `verify` sortent en erreur avant `p4e_after_launch` (`persist4e_cuda.rs:723`, `:884`).
- **Scénario.** Une barrière dépasse `PERS_SPIN_LIMIT` (environ 67 ms) sous contention GPU, ou un logit NaN apparaît.
  Toutes les requêtes suivantes répondent 500 (« the counters are poisoned »). Le moteur reste `Ready`, et chaque job
  repousse `expires`, donc le keep-alive ne le décharge jamais sous trafic.
- **Impact.** Disponibilité : le modèle servi est perdu jusqu'au redémarrage du processus. Rien sur tok/s ni qualité.
- **Correction.** Après toute sortie en erreur de `launch` ou `verify`, remettre `bar`, `abort`, `nan` et `bar_base` à
  0. Ou faire passer le moteur en `Failed` sur une erreur 4E, pour qu'il recharge.

### 1.2 PR #24 : une requête qui survit à ses voisines finit sur le GEMM à B = 1. CONFIRMED

- **Où.** `llvq-runtime/src/engine.rs:794` (branche `claude/batch-l40s`) n'entre dans `solo()` (4E) que si
  `live.is_empty() && pending.len() == 1`. Aucun chemin ne ramène une requête du pool vers 4E. De plus, `engine.rs:937`
  à `947` posent `arrived = true` pour tout job reçu, même un `Load` ou une requête refusée en 400.
- **Scénario.** A (2 000 jetons) et B (20 jetons) sont vivantes. Après la fin de B, A décode ses ~1 980 jetons restants
  dans le pool à B = 1. Autre cas : un simple `llvq run` sans prompt (Load) annule le 4E d'une requête seule.
- **Impact.** Débit servi : 106,2 tok/s au GEMM B = 1 contre 225,4 tok/s pour 4E seul au 4B, soit 0,47× sur la queue
  (*computed* depuis deux chiffres *measured*, batch-l40s-v4). Le banc ne le voit pas, car `ops/batch_serve_bench.py`
  lance des requêtes de même longueur, qui finissent au même pas. Le test
  `a_lone_request_runs_on_the_batch_1_path...` joue ce cas mais n'affirme que `widest == 2`.
- **Correction.** Si `live.len() == 1 && pending.is_empty()` sous un décodeur persistant, recopier le slot dans une
  `Session` (l'inverse de `caches_to_slot`) et reprendre `solo`. Ne poser `arrived` que pour un job admis qui n'est
  pas un `Load`.

### 1.3 La colonne « GB on card » est calculée, pas mesurée. CONFIRMED

- **Où.** `docs/ETAT.md:30,44` et `README.md:16,22` donnent 1,38 / 2,76 / 5,04 Go sous l'étiquette *measured* (le
  README dit « every cell is measured »).
- **Preuve.** Le banc imprime ces valeurs comme une attente : `total expected on the card: 1.38 GB (projections 1.16 +
  carried 0.22)` (`docs/mesures/served-v9-2026-11-07-brut/door.txt:15`). Le même job a mesuré, avec `cuMemGetInfo`
  juste après le chargement, 1,980 / 3,356 / 5,637 Go (`door.txt:27` et ses deux voisins), soit +43 / +22 / +12 %.
  Après un prompt de 8 192 jetons, le pic mesuré par le duel était de 11,48 / 12,66 / 17,09 Go (avant la PR #28).
- **Impact.** Go publiés sous-estimés, et une étiquette fausse (règle 8). Le b/param publié est aussi *computed*
  (rtbits), pas *measured*.
- **Correction.** Étiqueter la colonne « poids, *computed* » et ajouter les deux colonnes *measured* : après
  chargement, et au pic après un prompt de 8 192 jetons.

### 1.4 PR #28 : le pic prédit du bras `kprefill` est trop bas de 0,44 à 0,77 Go. CONFIRMED

- **Où.** `ops/pic_vram_8k.py:53` ne compte que les activations de l'attention (2·hidden + 3·q + 5·kv par ligne). Or
  `forward_cached` appelle `return self.finish(x, ctx, idx, cap)` (`model.rs` vers L2137 sur ebbd0da) dans sa propre
  portée. Ses liaisons masquées restent donc vivantes pendant tout le MLP : h, qkv, q et k normés, q et k après RoPE,
  plus ctx et sa copie transposée. `mlp_residual` garde aussi `gu` jusqu'à `down_proj`.
- **Calcul.** Au moment `gu[0].apply(act).mul(&gu[1])`, il y a par ligne 4·hidden + 5·q + 4·kv + 4·inter valeurs f16,
  plus le h du chunk précédent. Pour 4 096 lignes, l'écart avec le script vaut +0,42 / +0,53 / +0,73 Go. Le h précédent
  ajoute 0,021 / 0,034 / 0,042 Go. Le pic devient 3,91 / 5,47 / 8,14 Go au lieu de 3,47 / 4,91 / 7,37 (*computed*).
- **Impact.** Go. Avec la bande de ±0,5 Go du préreg, la porte G2 échoue au 8B et au 14B telle qu'écrite. La phrase
  « 3,47 contre 3,52 pour llama.cpp » devient 3,91 contre 3,52. Les pics servis (5,65 / 7,09 / 10,08) ne bougent pas :
  là, le MLP reste sous l'attention.
- **Correction.** Le préreg n'est pas tamponné. Calculer ctx dans un bloc interne, consommer ctx dans `finish`, lâcher
  `gu` avant `down_proj`. Faire prendre au script le max des deux phases, puis corriger la ligne `kprefill` avant
  tampon. La session pic-vram a été prévenue avec le décompte poste par poste. Elle comptait +0,20 / +0,29 / +0,44 ;
  l'écart restant vient des variables locales de l'attention, encore en portée.

### 1.5 Le store du runtime fige les noms de config : les stores CUDA tirés avant la v9 ne chargent plus. CONFIRMED

- **Où.** `llvq-runtime/src/store.rs:47` copie `configs` dans le manifeste au `pull`. Au chargement, seule la config
  `cpu` est rafraîchie depuis la bibliothèque (`library.rs:117`, `with_cpu_config`). Le commit 7be533e a retiré
  `qwen3-4b-tetra-e4-4e-v4.json` et les deux `*-grouped.json` de `CONFIGS`.
- **Scénario.** Un manifeste écrit le 2026-11-06 nomme v4 ou grouped, et son chargement échoue : « config … is not
  compiled into this binary ». Le prochain changement de config (v10) cassera tous les stores de la même façon. Une
  config renommée mais encore compilée serait servie périmée, sans message.
- **Correction.** Résoudre les trois configs par `sha256` dans la bibliothèque au chargement, comme pour `cpu`. Garder
  les noms du manifeste pour l'affichage.

### 1.6 Le bug des trois tenseurs de scores est encore sur `main`, et sa forme CPU est pire. CONFIRMED

- La correction (ebbd0da) n'existe que sur la branche de la PR #28. Le chemin CPU du runtime tourne en f32, par chunks
  de 4 096 lignes, avec un contexte de 40 960 (`max_positions()` vaut `None` pour l'eager). Au 4B, un prompt de 8k
  donne 4,29 Go par tenseur de scores et 12,9 Go à trois vivants. À 40k, un seul tenseur pèse encore 21,5 Go après le
  correctif (*computed*).
- **Correction.** Fusionner la #28. Sur CPU, plafonner aussi le chunk de préremplissage, par exemple à 512 lignes, ou
  le contexte servi.

### 1.7 PR #24 : `LLVQ_BATCH=<n>:<positions>` au-dessus de 8 704 fait échouer une requête seule en plein vol. CONFIRMED

- **Où.** À `engine.rs:1157-1158` (branche batch), la seconde ligne remplace le contexte par
  `serve.context.min(rope)` et perd `P4E_MAX_CAP` = 8 704. `served.rs` accepte `batch.context` jusqu'à 40 960 à côté
  de `decode: persistent`.
- **Scénario.** `LLVQ_BATCH=8:16384` avec un prompt seul de 10k jetons : le préremplissage réussit, puis 4E refuse
  (`persist4e_cuda.rs:915`) et le client reçoit un 500. Les configs livrées (2 048) n'y sont pas exposées.
- **Correction.** Envoyer au pool une requête dont prompt + `max_new` dépasse `P4E_MAX_CAP`, ou plafonner
  `batch.context` à côté d'un décodeur persistant.

### 1.8 PR #24 : le pool KV batch (19 à 21 Go) est alloué au chargement et n'apparaît nulle part. CONFIRMED

- À 64 × 2 048 positions, le pool pèse 19,3 / 19,3 / 21,5 Go pour 1,38 / 2,76 / 5,04 Go de poids (*computed* :
  147 456 B par position au 4B et 8B, 163 840 B au 14B). `/api/ps` publie la taille du fichier comme `size_vram`
  (`api.rs:530`). Au 14B, `LLVQ_BATCH=128:2048` demande 42,9 Go de pool plus 5,04 Go de poids, probablement au-delà de
  48 Go (*estimated*, non lancé). Aucun contrôle préalable contre la mémoire de la carte n'existe.
- **Correction.** Un refus au chargement si pool + poids dépassent la mémoire libre, et `size_vram` qui compte le pool.
  Tout chiffre en Go du batch doit inclure le pool (règle 6).

## 2. Moyennes : mesures qui se contredisent

### 2.1 La « dérive » de 4E v9 dans le duel est surtout une comparaison entre deux grandeurs. CONFIRMED

- Le journal du duel (`duel-iq2-2026-11-08.txt:111`, PR #26) compare son balayage décode seul (227,4 / 144,5 / 90,0) aux
  chiffres publiés de la porte (221,5 / 142,5 / 86,3), qui incluent le prompt. À grandeur égale, le balayage publié
  après 5 jetons vaut 226,2 [226,2-226,4] / 144,5 [144,4-144,8] / 87,3 [86,9-87,8] (served-v9). Le préreg du duel
  prédisait d'ailleurs 226 / 144 / 87.
- Il reste deux écarts réels.
  - Au 14B, +3,1 % avec des plages disjointes. Même module 4E (sha256 1b422b62), même unité NVRTC, mêmes tokens, même
    carte et même pilote. La variance d'instance de carte est la cause plausible. Rien ne l'explique dans le dépôt.
  - Au 8B, la porte du batch v1 (PR #24) lit 137,1 [136,2-138,3] contre 142,5 publiés, soit -3,8 % avec des plages
    disjointes, sans commentaire.
- **L'écart D1 du duel donne une fausse cause.** Il attribue la différence à 22338dd, qui « change la taille de groupe
  de la passe 4E ». Or 22338dd ne modifie que la closure d'une ligne de journal, à l'installation de la passe de
  vérification (`persist4e_cuda.rs:780`). Le module servi est identique.
- *PLAUSIBLE* : le Space a été republié à 10:11:45 UTC (2a58855), entre la création du job Tetra (10:06:37) et son
  départ (11:00:50). Aucune révision d'image n'est enregistrée, donc « tourné sur 0536a87c » n'est pas prouvé. Le
  diff ne touche que du code batch.
- **Correction.** Comparer balayage à balayage, corriger D1, et écrire que les instances de carte déplacent le 8B et le
  14B d'environ 4 %. Ce bruit est plus large que les plages publiées, qui sont intra-job.

### 2.2 PR #27 : les « portes » spéculatives à 395 et 469 tok/s ne comptent que le décode. CONFIRMED

- `docs/plan-speculation.md:67`, ETAT §5 et la PR les placent à côté des 221,5 de la porte gloutonne, mesurée avec un
  prompt de 5 jetons. Les prompts edit et rag sont longs : leur porte gloutonne mesurée vaut 98,5 et 45,6 tok/s
  (`spec-lookup-2026-11-08-brut/spec-4b.txt`). Le préremplissage y coûte 0,476 s et 1,36 s (*computed* : 85/98,5 −
  85/219,9 et 79/45,6 − 79/212,4).
- Avec un décode à 395 et 469, la porte vaudrait environ 123 et 52 tok/s, soit 1,24× et 1,13× et non 1,79× et 2,12×
  (*computed*).
- **Correction.** Les appeler tok/s de décode et donner la porte avec le préremplissage mesuré.

### 2.3 PR #26 : deux verdicts du duel ne tiennent pas tels qu'écrits. CONFIRMED

- **La vitesse et le pic d'IQ2_XXS sont mesurés et commités, mais le journal dit « not measured (D2) »** (ligne 127).
  `{4b,8b,14b}/bench-own-iq2_xxs-d5.json` donnent 308,7 [308,4-310,6] / 229,3 [229,3-230,6] / 137,1 [137,1-137,7]
  tok/s, recalculés depuis `samples_ts`. Deux prédictions du préreg ratent :
  - 8B : 229,3 contre 185 [165, 210] ;
  - 14B : 137,1 contre 115 [100, 130].

  Le verdict « tient sur la mémoire d'IQ2_XXS » n'est pas étayé : le pic à 8 192 n'a pas été mesuré, et au 4B après 5
  jetons, c'est l'égalité (2,00 contre 2,01 Go).
- **llama.cpp garde son embedding d'entrée en mémoire hôte** (`CPU_Mapped model buffer size` 255,02 / 255,02 / 318,77
  Mio dans les logs), hors du pic nvidia-smi. Tetra garde le sien sur la carte. En le comptant, le verdict « pic après
  5 jetons » s'inverse au 8B (3,62 contre 3,42 Go) et devient une égalité au 14B.
- **Correction.** Rapporter IQ2_XXS comme mesuré non retenu (D2), et écrire l'embedding hôte à côté du verdict mémoire.

### 2.4 Comptabilités b/param mélangées dans un même tableau (PR #26). CONFIRMED

- ETAT §3 sur la branche du duel ajoute les lignes IQ2_S en octets de fichier (2,82 / 2,80 / 2,69) à côté de Tetra en
  rtbits (2,73 / 2,70 / 2,73). Le tableau montre donc Tetra 0,09 b/param sous IQ2_S au 4B, alors que le duel mesure une
  égalité (2,8206 contre 2,8188). Le tok/s mêle le décode llama-bench (287,5) et notre porte (221,5). Les Go mêlent
  taille de fichier et poids sur carte.
- **Correction.** Une comptabilité par tableau : ajouter la ligne Tetra du duel (2,8206 b/param, 227,4 tok/s de décode,
  1,418 Go de fichier).

### 2.5 Ratios entre piles, contre la règle 5. CONFIRMED

- `docs/HISTORIQUE.md:1247` : « leaves Tetra four times below AWQ in aggregate », soit vLLM divisé par notre pile.
- `docs/mesures/batch-l40s-2026-11-08.txt:87` (PR #24) : « about two times below AWQ … three to five times below it
  in GB/s », alors que la ligne 33 du même journal dit « No x across the two stacks ».
- Règle de décision M14 de `metal-nuit-2026-11-06.txt:1165` : « 1.5 times our rate » (mlx-lm contre Rust).
- Le duel donne la perplexité en ratio (« x0.7500 », « 25.0 % lower »), contre les §5 et §6 de son propre préreg.
- **Correction.** Des niveaux, ou des GB/s côte à côte, sans quotient.

### 2.6 PR #27 : deux affirmations fausses sur le microbanc de vérification. CONFIRMED

- **« Every row within one binary16 ulp » est faux.** Sur l'activation 0, 1 105 920 − 1 013 203 − 78 525 = 14 192
  lignes (1,3 %) sont à 2 ulps ou plus (`spec-micro-2026-11-08-brut/r1/bench.txt:100`). `vrows_i16_check`
  (`llvq-cuda/src/pers.rs:1787`, `_ if e <= VROWS_I16_TOL => {}`) ne les compte nulle part.
- **La porte 4 n'est vérifiée que sur l'activation 0, et le bug est toujours dans le code.**
  `llvq-cuda/src/persbench.rs:829` itère `got.iter().zip(refs.iter().enumerate())`, et `got` a K entrées. L'écart 3
  le déclare, mais la tête de branche ne le corrige pas : relancer `ops/jobs/spec-micro.sh` échouerait de nouveau à la
  porte 4.
- **Correction.** Compter la classe « plus d'un ulp mais sous la tolérance » dans `I16Check`. Relancer chaque
  activation pour les bras K = 1, comme la branche I16 le fait déjà.

## 3. Registre des coûts

Le cumul de chaque branche ne compte que ses propres lignes. Le total reconstitué depuis `docs/data/jobs.csv` de toutes
les branches, locales comprises, sans doublon de `job_id` :

| source | lignes ajoutées | coût |
|---|---|---|
| `main` (1997edc) | 278 | 280,04 $ |
| PR #24 batch-l40s | 5 | 3,47 $ |
| PR #26 duel | 13 | 17,55 $ |
| PR #27 spec (dont `spec-lookup-r1` sans coût) | 4 | 0,98 $ |
| PR #28 pic-vram | 0 | 0 $ |
| `rows-grid` (6aef647f), seulement sur la branche locale non poussée `claude/route-300-toks-de97fe` (cf19769) | 1 | 0,25 $ |
| `spec-lookup-r1` (6af0b3e0), terminé à 18:38 UTC, 3 519 s de carte à 1,80 $/h | (ligne vide) | 1,76 $ |
| **Total** | **301** | **304,05 $** (*computed*) |

- **Les cumuls annoncés.** 280,04 $ est la base de `main`. 280,53 $ et 281,02 $ sont la branche spec à deux moments.
  297,59 $ est la branche du duel seule. 294,59 $ n'apparaît dans aucune branche. L'union des branches distantes
  donne 302,04 $ sur 300 lignes, vérifiée par la fusion d'essai.
- **ETAT est périmé partout.** `main` dit 277,11 $ sur 269 lignes chiffrées de 274 (`docs/ETAT.md:164`). Le vrai
  chiffre est 280,04 $ sur 273 de 278 : il manque les quatre lignes de #17 et #22. Les branches batch, duel et pic-vram
  gardent 277,11 $. La branche spec dit « of 281 », il y a 282 lignes.
- **Cinq lignes dépassent le temps de carte HF** (`started_at` → `finished_at` de `hf jobs inspect`) :

  | job | registre | carte | coût carte |
  |---|---|---|---|
  | served-4e | 0,66 $ | 1 234 s | 0,62 $ |
  | shaker-4e | 0,48 $ | 724 s | 0,36 $ |
  | served-v4 | 0,42 $ | 745 s | 0,37 $ |
  | pers-4e-v6 | 0,24 $ | 260 s | 0,13 $ |
  | spec-gate | 0,30 $ | 341 s | 0,17 $ |

  Ces journaux ont compté depuis la création du job, attente comprise : +0,45 $ au total. La base de facturation HF
  n'est pas vérifiée ici. Les autres lignes de la période tombent sur le temps de carte à l'arrondi près.
- **Correction.** Pousser `cf19769` ou reprendre sa ligne et son journal (sinon ce journal peut disparaître). Remplir
  la ligne `spec-lookup-r1`. Régénérer la ligne Spend d'ETAT depuis le CSV au moment de chaque fusion.

## 4. Faibles

- **PR #24, `engine.rs:535-538`.** `\\` en fin de ligne laisse un antislash, un saut de ligne et 25 espaces dans les
  deux messages de refus « cannot sample ». C'est une régression du message 4E de `main`, et le test ne vérifie
  qu'une sous-chaîne. Correction : `\` simple, et le test sur la chaîne entière.
- **PR #24, preuve hôte du GEMM.** Elle ne voit aucune synchronisation : sur l'hôte, `BG_WARP_SYNC` vaut
  `__syncthreads`, et `bg_wait` et `bg_commit` ne font rien. Supprimer `__syncthreads()` à `tv_batch_gemm.cu:351`
  passe 4 cas sur 4, alors que cette barrière porte la charge sur carte. Aucun script de mutation n'est commité, donc
  « 17/17 » ne se reproduit pas. Le noyau est correct tel qu'écrit.
- **Mutants survivants du runtime (`main`).** Retirer le filtre `e.idle()` de l'éviction LRU (`engine.rs:792`) et celui
  de `reap` (`:835`) laisse passer les 42 tests (refait ici). Une régression déchargerait un moteur occupé, et
  `LLVQ_MAX_LOADED=1` tiendrait deux modèles. Il manque un test avec un `Loader` lent.
- **PR #27, `lookup_draft`.** Le mutant qui retire `.rev()` (le n-gramme le plus long d'abord) survit aux 4 tests. Cela
  ne touche que l'acceptance : les tokens restent gloutons.
- **PLAUSIBLE, CPU servi.** Le décode ralentit avec le contexte. La config CPU sans champ `attention` prend expanded avec
  le store `Cat` : `Tensor::cat` de tout l'historique et `repeat_kv` à chaque couche. Cela fait environ 5,9 Go de copies
  par jeton à 4 096 positions (*estimated*, proxy numpy 0,65 s par jeton). `cpu-tetra-2026-11-07` n'a mesuré que des
  prompts d'environ 40 positions. À mesurer avec `cpurun` sur un prompt de 2k à 4k avant tout changement.
- **PLAUSIBLE, `session.rs:96-106`.** Un `feed` qui échoue en milieu de couche (OOM au préremplissage) laisse les caches
  en avance sur `ids`. Un nouvel essai du même prompt écrirait alors au mauvais indice.
- **PLAUSIBLE, PR #28.** `settle_growth` synchronise à chaque couche pendant le préremplissage, et rend chaque fois le
  tenseur S libéré au pilote : environ 0,03 à 0,3 s de TTFT sur 34,5 s (*estimated*). Le h du chunk précédent reste
  vivant pendant le chunk suivant (21 / 34 / 42 Mo). `shared_attention` (non servi) garde le motif à 4·S.
  `LLVQ_SOFTMAX_WHOLE` est ignoré sans message hors du mode `LLVQ_CONTEXTS`.
- **PLAUSIBLE, candle.** Le softmax CUDA indexe en `int` (`reduce.cu`). Le softmax d'un seul tenseur déborde quand
  têtes × l × L ≥ 2³¹, par exemple un chunk de 4 096 lignes au-delà de 16 384 clés à 32 têtes. Aucun chemin servi n'y
  arrive (4E plafonne à 8 704) ; les blocs de 1 024 lignes de la #28 l'écartent.
- **Documents vivants en retard.**
  - `docs/recherche-debit-maths.md` donne 454 / 288 / 171 tok/s (δ 0,30 ms), contre 363 / 223 / 136 dans
    `plan-speculation.md` (δ 0,64 ms).
  - Les corps des PR #25 et #27 sont périmés : #25 annonce encore 278 / 196 / 121, et #27 dit le prompt lookup « not
    run on a card ».
  - ETAT:147-149 et ROADMAP:26-33 disent le spéculatif « next, on a branch », alors que spec-gate est fusionné et
    fermé en §6, sans décision d'opérateur citée.
  - Metal : 52,1 tok/s dans ETAT:123, 65,5 dans ROADMAP:78. Ce dernier bras inclut grouped, que Metal refuse.
  - Les `note` des configs v9 8B et 14B disent encore « Not yet read at the served door ».
- **Contrôles plus faibles qu'ils n'en ont l'air.**
  - ETAT §4 Batch : « every sequence decodes the batch-1 door tokens », mais toutes les lignes reçoivent le même prompt
    de 5 jetons. La preuve est le pilote GSM8K à prompts différents.
  - Le modèle de pic de la #28 « tombe sur les trois pics » après un terme de pool *estimated* ajouté pour expliquer la
    mesure : c'est un ajustement.
  - `gqa-ppl` : max |dlogit| de 13,13 contre une prédiction [0,04 ; 0,5]. Ce n'est expliqué nulle part depuis, alors
    que le 8B (3,83) et le 14B (4,31 et 11,95) montrent les mêmes pics.
- **Divers.**
  - `cpurun gate` utilise 1,96 au lieu de t(3) = 3,182, d'où un intervalle 38 % trop étroit, que rien de publié
    n'utilise.
  - `max_tokens: 0` signifie « jusqu'à la fin du contexte » au lieu d'un 400.
  - La porte MLX fait confiance à `artifact_sha256` sans hacher les poids.
  - Spec-micro coûte 0,48 $ dans le journal et 0,49 $ dans le CSV.
  - HISTORIQUE:884 dit « three jobs, $1.55 » pour quatre lignes.
  - Les tableaux d'ETAT et du README donnent des médianes sans plage (règle 7).

Vérifié propre, en résumé :
- Tous les préregs de la période ont un `.ots` dont l'empreinte est celle du `.md` actuel. Chaque tampon précède la
  création de son job ; le plus serré est fi-decode-anchor, à 2 s.
- Les journaux du duel se recalculent depuis les fichiers bruts : vitesses, pics, ppl, MMLU et GSM8K.
- Les bandes ppl 8B et 14B, les deltas GSM8K grouped et leurs McNemar se recalculent exactement.
- Le noyau `tv_batch_gemm.cu` ne montre aucun défaut d'indice, de forme, de débordement u32 ni de course.
- Le correctif servi de la #28 est exact, et ses tests au bit sont de vraies comparaisons.
- Le spéculatif garde les tokens gloutons, contrôlés contre le décode 4E servi dans le même processus.
- Les trois échecs de fusion cités dans la commande de cet audit sont déjà corrigés sur `main` par 1997edc :
  - `Configs` sans `cpu` ;
  - `LLVQ_BATCH_REF` et `LLVQ_TOKENIZER` absents du registre ;
  - le 501 envoyé avant lecture du corps.

## 5. Carte des fusions

La #25 est contenue dans la #27 : ce sont ses trois premiers commits. Il suffit de la fermer avec la #27.

Conflits textuels, par `git merge-tree` :

| paire | fichiers en conflit | nature |
|---|---|---|
| #24 × #28 | `served.rs`, `model.rs`, `gsm8k.rs`, `tests/served_config.rs`, `configs/README.md` | chacune ajoute un champ à `ServedFile` (`batch`, `prefill_attention`) ; garder les deux champs, enchaîner les deux `match` de `provenance()`, garder les deux tests |
| #27 × #28 | `docs/ETAT.md` | texte |
| #24 × #27, #24 × #26, #27 × #26 | `docs/data/jobs.csv`, `docs/HISTORIQUE.md` | garder toutes les lignes |
| #26 × #28 | aucun | |
| #23 × #28 | `model.rs`, `fusedrun.rs`, `docs/ENV.md` | #23 découpe `model.rs` et `fusedrun` en modules ; les 539 lignes de la #28 sont à reporter dans les modules |
| #23 × #24 | `model.rs` | même cause |
| #23 × #27 | `planesbench.rs`, `persbench.rs`, `docs/ENV.md` | #23 découpe `planesbench` et retire `LLVQ_PERS_ARMS`, `LLVQ_PERS_MICRO` et `LLVQ_TIME_EVENTS` (liste `REMOVED`) |
| #23 × #26 | aucun conflit textuel, mais **la CI casse** | la #23 ajoute `ops/check_big_files.py` (limite 1 Mio) ; la #26 ajoute 12 fichiers bruts de 1 251 052 à 4 142 403 octets |

Essai réel de l'union #28 + #24 + #27 + #26, dans un worktree jetable, conflits résolus comme dans le tableau :
- `cargo check --workspace --all-targets` passe ;
- `cargo test --workspace` en debug donne 212 lignes `test result`, 0 échec, dont `env_registry` et `server.rs` ;
- `ops/check-linux-cuda.sh check` sur `llvq-llm` et `llvq-runtime` avec la feature `cuda`, et `cargo check` Linux de
  `llvq-cuda`, passent.

Aucune incompatibilité sémantique trouvée entre ces quatre-là. Le préremplissage du chemin batch (`prefill_slot`) passe
par `hidden_cached`, donc il hérite de la synchronisation de la #28.

Ordre de fusion sûr :
1. **#28** : petite, corrige le chemin servi.
2. **#24** : résoudre `ServedFile` en gardant les deux champs, puis corriger 1.2, 1.7 et le `\\` avant ou juste après.
3. **#27** : conflits de docs seulement ; fermer la #25 ; remplir la ligne `spec-lookup-r1`.
4. **#26** : conflits de docs seulement.
5. **#23 en dernier**, refaite sur le nouveau `main`. Ses découpages sont des déplacements purs, plus faciles à rejouer
   que les ajouts de #24, #27 et #28 à reporter dans les modules. Avant elle, mettre les 12 fichiers du duel au bucket
   ou dans `ops/big-files.allow`, et vérifier qu'aucun job encore utile ne pose une variable de `REMOVED`.

Après chaque fusion, régénérer la ligne Spend d'ETAT depuis le CSV.
