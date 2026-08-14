# Pré-enregistrement — P3 : ce qu'un cache KV int8 coûte en qualité et en débit, et les seuils qui l'autorisent ou le ferment

**Date : 2026-08-14.** Écrit **avant toute mesure** et **avant la première
ligne de code**. À cette heure :

- **aucune ligne de quantification de cache KV n'existe**, sur aucun backend :
  `grep -rniE 'LLVQ_KV|kv_q8|kvq|quantize.*cache|cache.*quant'` sur `*.rs` et
  `*.cu` ne rend **rien** ;
- **aucune mesure, nulle part, de la sensibilité de la ppl ou du MMLU à la
  précision du cache KV** ; le seul précédent q8 porte sur l'**embedding** ;
- **`bin/gbench` compile et n'a aucun journal** : `grep -rln gbench
  docs/mesures/` ne rend rien sur les 41 fichiers du répertoire ;
- **le fichier de référence est absent de la machine** :
  `ls ~/qwen3-4b-llvq.bin ~/q4b-e8.llvq` → *No such file or directory*, vérifié
  le jour même ; récupérable à l'octet près (§2.1) ;
- **aucun instrument de mémoire device n'existe côté Metal** : la seule colonne
  « Go carte » est dans `bin/fusedrun` (`fusedrun.rs:221`), chemin CUDA — d'où
  le §1.4, le gain mémoire est **compté**, pas mesuré.

**Emplacement de signature :** `proofs/preregistration-p3-2026-08-14.md`, liens
**relatifs au dépôt** ; toute copie hors de `proofs/` est un brouillon où ils ne
résolvent pas.

> ⚠️ Ni signé GPG ni horodaté tant que l'opérateur ne l'a pas fait
> (`ots stamp proofs/preregistration-p3-2026-08-14.md`). **Le tampon est
> porteur** : demandé avant la première ligne de `kvq.rs`.

### L'héritage du 08-10, garde par garde

« Hérite sans dérogation » serait **faux sur quatre gardes sur six**. Les six de
[`preregistration-2026-08-10.md`](preregistration-2026-08-10.md) §7 (`:236-252`) :

| garde du 08-10 | statut en P3 |
|---|---|
| rapport = médiane des rapports **round par round**, jamais un quotient de deux minima (`:242-244`) | ✅ **appliquée** (§1.6, §2.5) |
| vérification avant tout chronométrage (`:245-246`) | ✅ **appliquée sous une autre forme** : pas de référence f64, aucun noyau n'étant écrit — les six gates V0 du §3 |
| un seul processus, bras entrelacés, ordre fixe (`:240-241`) | ❌ **dérogée** — `gbench` charge un modèle par processus ; 3 paires alternées (§2.5) |
| 7 rounds, les 2 premiers jetés (`:242`) | ❌ **dérogée** — 3 paires + le warmup de `gbench.rs:109-124` |
| zéro octet de mémoire locale au rapport de registres (`:247-248`) | ❌ **sans objet** |
| coût GPU par job dans `docs/data/jobs.csv` (`:252`) | ❌ **sans objet** — Metal local, 0 $ |

Les quatre dérogations sont **inscrites au §7bis à la signature**. Héritage sans
réserve de la discipline du
[P1 du 2026-08-13](preregistration-p1-2026-08-13.md) — exactitude avant
chronométrage — ⚠️ mais ce P1 **n'est pas horodaté** (seuls `2026-08-10` et
`2026-08-11` portent un `.ots`) : cet héritage-là est **déclaratif**.

---

## 0. Ce que ce lot peut et ne peut pas conclure

P3 mesure la **qualité** et le **débit** d'un cache KV int8 sur **Metal**,
**M3 Max**, sur le **4B scellé décodé en dense**, à **batch 1**.

🚨 **`ppl` et `mmlu` n'établissent RIEN sur le gain mémoire.** Deux raisons
indépendantes : ils passent par `Block::forward`, qui construit un
`KvCache::default()` **local à la fonction** (`model.rs:726`) et ne tient donc
qu'**une couche** — 16 777 216 o, pas les 0,604 Go des 36 (seuls `generate`
`:1011` et `generate_phased` `:1077` les gardent vivants, via `fresh_caches`
`:919-921`) ; et l'objet dominant de la passe est le tenseur de scores de
`model.rs:787`, `[1, 32, 4096, 4096]` = **2,147 Go en f32** (calculé), `bin/ppl`
construisant en **F32** par défaut (`ppl.rs:51`). L'objet quantifié est **64 à
128× plus petit** que lui.

**Ce qu'un vert achète** : le droit d'être câblé et mesuré en mémoire sur carte
en P4, rien d'autre. **Ce qu'un rouge fait** : il ferme le levier KV q8, pour
**0 $** et un plafond de **4 h de Mac**.

⚠️ **Ne transporte pas vers le chemin servi** : sur Metal, `sealed::load`
reconstruit des tenseurs **denses** — pas de noyau fusé côté Apple. Le rapport
mesuré borne un **coût de plomberie**, pas le débit du produit.

⚠️ **Ni vers une autre échelle** : Qwen3-4B et Qwen3-8B ont **le même cache par
token**, 147 456 o (`docs/archive/plan-de-test-v2-cuda.md:324`). Seul un modèle
à géométrie différente en serait un (32B 262 144 · Llama-3.1-70B 327 680
o/token, même ligne).

## 0bis. Ce que P3 ne peut pas fermer, faute d'un nombre en face

**Trois verts ne rendent pas le barreau 32 Go passable**, pour deux raisons hors
de son périmètre.

1. **Le triplet (carte, contexte, marge) n'est pas arbitré** : les cases §A de
   `docs/note-produit-2026-08-13.md` sont **vides** (`:20-25`), et la note dit
   « Un seul triplet fera foi — coché en §A, **avant toute lecture de
   résultat** » (`:72`).
2. 🚨 **La tolérance « capacity-first » n'est chiffrée NULLE PART** : cinq
   occurrences dans `docs/` et `proofs/`, **aucune avec un nombre**
   (`docs/archive/etude-moe-memoire-extreme-2026-08-12.md:180`,
   `docs/archive/passation-exec-2026-08-13.md:106`,
   `proofs/preregistration-p1-2026-08-13.md:278`, `:402`, `:423`). Une issue
   « X passe la tolérance capacity-first » **ne peut rien fermer** : qui la veut
   ouverte dira qu'elle ne passe pas, qui la veut fermée dira l'inverse, et rien
   ne départage.

**Décidé ici :** ce document **ne pose aucune issue de cette forme** et interdit
de lire ses verts comme telle. Un vert rend la colonne « KV q8 » de la note
produit §B **chiffrable avec sa provenance** ; il ne rend aucun barreau passable
et **ne ferme pas P5**.

## 1. La comptabilité, figée ici

**1.1 — Le gain KV se dit en octets/token, jamais en b/param modèle entier.**
La règle de chiffres n°1 (`CLAUDE.md:1778`) vise la mémoire **de paramètres** ;
le cache n'en est pas un. Il se dit en **octets/token, avec le batch, le
contexte et la géométrie nommés sur la même ligne**
(`docs/archive/plan-de-test-v2-cuda.md:316` la formule, `:324` la table).

**Géométrie du 4B** — 36 couches, 8 têtes KV, `head_dim` **128**, batch 1
(`docs/fiche-4b.md:78`). ⚠️ **Aucun `config.json` Qwen3 n'est sur cette machine**
(vérifié le jour même) : la source est le dépôt, pas le checkpoint. ⚠️ `head_dim`
n'est **pas** `hidden_size / num_attention_heads` (2560/32 = 80 contre 128) ; le
code lit `cfg.head_dim` (`model.rs:615-619`) et toute arithmétique ici fait
pareil — le quotient sous-compterait le cache de ×1,6.

| grandeur | valeur | provenance |
|---|---|---|
| valeurs de cache par token | 73 728 (`2 · 36 · 8 · 128`) | calculé |
| **f16** | **147 456 o/token** | calculé, = `plan-de-test-v2-cuda.md:318` |
| **q8 g64** (échelle + biais f16 par groupe de 64) | **78 336 o/token** | calculé |
| q8 par tête (g128) | 76 032 o/token | calculé |
| f16 → q8 g64 | **÷1,882** — **pas ÷2** | calculé |
| f16 → q8 g128 | ÷1,939 | calculé |

À ctx 4096, batch 1, 36 couches : **0,604 → 0,321 Go**, soit **0,283 Go** ; à
ctx 32768 : 4,832 → 2,567, **2,265 Go** (calculé). Le levier est à contexte
long ; à 4k il pèse ~11 % du modèle servi (2,60 Go).

**1.2 — La facturation est celle du dépôt** : `bits + 32/group`
(`embedquant.rs:95-97`) — **8,5 b/valeur** en g64, **8,25** en g128, même
constante que `EMBED_Q8_BPP = 8.5` (`llvq-llm/tests/fused_tail.rs:205`). **Un
« KV q8 = moitié du f16 » est faux et ne sera écrit nulle part** ; la note
produit, qui le facture à ~8,0, n'est pas opposable ici (§8).

**1.3 — La granularité est figée, et elle ne traverse JAMAIS l'axe temps.**

- **Servie : 2 groupes de 64 le long de `head_dim`, par (token, tête),
  séparément pour K et pour V.** `head_dim = 128` est l'axe le plus interne du
  cache (`model.rs:760-765`, forme `[b, n_kv, t, head_dim]`) : un groupe tient
  **entièrement dans une position**.
- **La raison** : un préfill de `l` tokens et `l` pas de décodage doivent
  produire **les mêmes octets**. Toute granularité par bloc de temps casse cette
  égalité et fait diverger préfill et décodage **sans qu'aucun seuil de ce
  document ne l'attrape** — ni ppl ni MMLU ne font que du préfill.
- **Repli, facturation figée d'avance** : une échelle par (token, tête),
  8,25 b/valeur, mesurée **uniquement** dans le cas prévu au §6, jamais comme
  échappatoire à un rouge de qualité.
- **Schéma d'`embedquant::quantize_affine` sans amendement** (`:22`) : min/max
  sur le groupe, `s = f16((mx − mn)/255)`, `b = f16(mn)`,
  `q = round((w − b)/s)` **clampé sur [0, 255]**, calculé **contre les `s`/`b`
  déjà arrondis en f16** (`:50-71` ; `qmax` à `:39`). Stockage `DType::U8` —
  candle 0.9.2 n'a pas de `I8`.

**1.4 — Le gain mémoire est un COMPTE, pas un critère — transitoire compris.**

| poste, ctx 4096, batch 1 | f16 | q8 g64 | provenance |
|---|---|---|---|
| cache résident, 36 couches | 603 979 776 o | 320 864 256 o | calculé |
| transitoire de déquantification (1 couche vivante) | — | **+16 777 216 o** | calculé |
| net | — | **−266,3 Mo** | calculé |

⚠️ Sans cette ligne, un lecteur suppose le gain gratuit : la déquantification
matérialise l'historique en flottant avant `repeat_kv` (`model.rs:783-784`). À
batch > 1, ou si une implémentation matérialise plus d'une couche, **le signe
peut s'inverser** — entorse au §7bis.

**1.5 — Les unités de qualité, et pourquoi les deux seuils hérités sont
jetés.**

🚨 **« Kill si ppl > 0,7 % »** est du bruit de **graine de calibration**, sur
**Qwen3-0.6B, 3 blocs**, entre **fichiers différents**
(`docs/archive/verdicts-lot-b-2026-08-06.md:19-21`). Un A/B KV compare **le même
fichier à lui-même, à empreinte identique** : l'évaluation est **déterministe**
et le Δ n'a pas ce bruit-là.

**La barre juste pour la ppl est l'intervalle t APPARIÉ, fenêtre par fenêtre.**
`bin/ppl` imprime le NLL de **chaque** fenêtre à **9 décimales**
(`ppl.rs:116-131`). Protocole : `Δ_w = NLL_q8(w) − NLL_f16(w)` sur les 12
fenêtres, `moyenne ± t_{0,975 ; 11} · s/√12` (t = 2,201), relatif par
`exp(Δ̄) − 1`. **Validité au journal** : les 12 `n` imprimés doivent être
identiques ; sinon la statistique se pondère et le journal le dit.
⚠️ Deux réserves : le **« ~6 »** du commentaire (`ppl.rs:119`) n'est **sourcé
nulle part** et aucun seuil ne s'y appuie ; et ces lignes partent sur **stderr**
(`:125`) quand le résultat part sur **stdout** (`:136`) — **sans redirection
`2>` elles sont perdues**, et le calcul apparié n'a aucun consommateur écrit.

🚨 **« σ McNemar 0,4-0,6 pp » est jeté aussi** : ni mesuré ni relatif au chiffre
publié, c'est une estimation dont la source écrit « jamais calculé »
(`docs/archive/errata-rapport-lot-a-2026-08-06.md:55-56`), portant sur le taux
**poolé non pondéré**, pas sur le **micro stratifié**. **La SE appariée
réellement mesurée** (bootstrap apparié stratifié par matière, 10 000 tirages,
graine `0xb0075eed`, correction de population finie —
`docs/mesures/mmlupair-4b-8b-2026-08-13.txt`) :

| paire | SE micro stratifié | SE contrôle non pondéré | ligne |
|---|---|---|---|
| 4B f16 ↔ AWQ | **0,96 pp** | 0,54 pp | `:24-25` |
| 4B f16 ↔ LLVQ | **1,41 pp** | 0,91 pp | `:52-53` |
| 8B f16 ↔ AWQ | **0,79 pp** | 0,47 pp | `:82-83` |
| 8B f16 ↔ LLVQ | **1,02 pp** | 0,76 pp | `:110-111` |
| 4B AWQ ↔ LLVQ | **1,44 pp** | 0,90 pp | `:143-144` |
| 8B AWQ ↔ LLVQ | **1,12 pp** | 0,76 pp | `:173-174` |

**Poser 0,5 pp sur le micro se donnerait une puissance qu'on n'a pas, d'un
facteur ~2.** ⚠️ **La SE de la paire de P3 est une SORTIE du run, pas une
entrée** — ces six valeurs portent sur des **modèles différents**, celle de P3
sur le **même fichier à deux précisions de cache**. **Aucun seuil n'est fixé
contre ces six nombres** : d'où la **condition d'intervalle explicite** du §4.2
plutôt qu'un pari sur une SE inconnue.

**1.6 — L'unité de débit est le tok/s, et c'est une PLAGE** (règle de chiffres
n°2, `CLAUDE.md:1782`), là où ppl et MMLU sont **déterministes à empreinte
identique** et se citent en point. Un rapport se forme **par paire d'invocations
de même rang**, jamais comme quotient de deux meilleurs.

## 2. Le protocole, figé ici

**2.1 — Le fichier de référence, et son sha256 comme condition d'entrée.**
`~/qwen3-4b-llvq.bin`, **1 770 527 533 octets**, sha256
**`9db213ef9fa9d7d7000789a8a529ce9459ce9ba6002ef5a72fd5a1c05c1c84b0`**
(`docs/fiche-4b.md:38-44`, MESURE, identique au dépôt
`Pier-Jean/Qwen3-4B-LLVQ-2bit` au commit `f00daa7bc1dd12a720304a4483f2219d10f15c96`).
Récupération : `hf download Pier-Jean/Qwen3-4B-LLVQ-2bit qwen3-4b-llvq.bin --local-dir .`
(`docs/hf-model-card.md:150`).

- **Vérifié avant toute mesure**, imprimé en tête de journal. **Un sha qui ne
  tombe pas ⇒ aucune mesure n'a lieu.**
- 🚨 **La taille ne suffit pas, et le piège est sur cette machine** :
  `~/qwen3-4b-llvq-L4swap.bin` y est à **1 770 528 117 o** — **584 octets
  d'écart**, invisible sur une ligne de `ls` — et sa qualité diffère (17,7459 en
  ppl). Le sha256 est la **seule** identification qui vaille.
- ⚠️ **Ne jamais planifier `ppl … ~/llvq-q4b.llvq`** : magie **LVQ1**,
  `sealed::load` le **refuse** (`is_self_contained()` faux, `sealed.rs:71-83`).
  Ce garde-fou est une **déclaration de version, pas un contrôle de contenu** :
  `~/q8b-c12.llvq` porte LVQ3 et serait accepté sans contenir d'embedding.
- Un run sur un substitut est un **fumigène** : le fichier est **nommé sur la
  ligne de résultat** et **aucun Δ n'est annoncé contre 16,9415 / 55,59**.

**2.2 — Le point d'insertion : tranché ici, pas en codant.** `KvCache::append`
est **privée** (`model.rs:209`), ses champs aussi (`:198-199`), et **`model.rs`
ne lit aucune variable d'environnement** : `grep -c 'std::env'
llvq-llm/src/model.rs` = **0**. `fused.rs` centralise `LLVQ_FUSED_LAYOUT`
(`:102-105`) et `LLVQ_EMBED` (`:147-150`) sous le contrat « un nom inconnu est
une **erreur**, jamais un repli — *a typo silently falling back to a default
would make an A/B lie* » (`fused.rs:84-86`).

1. Un module neuf, `llvq-llm/src/kvq.rs`, porte `KvMode { F16, Q8 }` avec le
   **contrat exact** d'`EmbedMode::parse` (`fused.rs:135-144`) : `None |
   Some("")` ⇒ `F16`, `Some("f16")`, `Some("q8")`, **tout autre nom est une
   `Err`**. `from_env()` lit `LLVQ_KV`.
2. 🚨 **`#[derive(Default)]` est RETIRÉ de `KvCache` (`model.rs:196`).** Tant
   qu'un `Default` existe, un câblage incomplet compile et un bras non branché
   rend Δ = 0 sur les trois axes — **trois verts entièrement vides**. Sans lui,
   le compilateur force les **deux** sites de construction à porter le mode :
   `Block::forward` (`:726` — chemin de `ppl`, de `mmlu` et de
   `generate_uncached` via `logits`→`hidden`→`blk.forward`, `:941-955`) et
   `fresh_caches` (`:920` — chemin de `generate` `:1011` et de `generate_phased`
   `:1077`). **Les deux doivent le porter** : n'en câbler qu'un rendrait
   `LLVQ_VERIFY_CACHE` rouge pour une bonne raison, n'en câbler aucun le
   rendrait vert pour une mauvaise.
3. Mode **passé en valeur** jusqu'à `KvCache` (constructeur explicite) ;
   `model.rs` garde ses **zéro `std::env`**, revérifié par le même `grep -c`.
4. **`ppl`, `mmlu`, `gbench`, `run` résolvent le mode une fois et l'impriment
   sur la ligne de résultat**, à côté du dtype et de l'empreinte. ⚠️ **Cette
   impression n'est PAS une preuve d'activité** — c'est le §3.6 qui l'est.

**2.3 — Hors périmètre, acté ici.** **Quantifier K avant RoPE** : le cache est
rempli **après** la rotation (`rotary.apply` `model.rs:780`, `cache.append`
`:781`) ; la variante exigerait de déplacer `append` **et** de réappliquer RoPE
à la lecture sur tout l'historique — une réécriture du bloc. **Le token courant
est quantifié comme les autres, et c'est assumé** : `append` **retourne ce qu'il
stocke** (`:220`), consommé tel quel par `repeat_kv`, donc **aucune séparation
écriture/lecture**. La plupart des schémas KV-int8 gardent la position courante
en pleine précision ; **nous ne le faisons pas** — désavantage assumé, qui joue
**contre** nos chiffres.

**2.4 — Les commandes, figées.** Gabarit du précédent q8
(`~/llvq-nuit-b/queue.sh:30`, `:52`, `:64`).

```bash
cargo build --release -p llvq-llm --features metal \
    --bin ppl --bin mmlu --bin gbench --bin run --bin mmlupair

# perplexité — DEUX bras, dtype forcé des deux côtés, stderr CAPTURÉ
env LLVQ_DTYPE=f16 LLVQ_KV=f16 target/release/ppl 4096 12 metal ~/qwen3-4b-llvq.bin \
    > p3-ppl-f16.out 2> p3-ppl-f16.err
env LLVQ_DTYPE=f16 LLVQ_KV=q8  target/release/ppl 4096 12 metal ~/qwen3-4b-llvq.bin \
    > p3-ppl-q8.out  2> p3-ppl-q8.err

# MMLU — DEUX bras, dtype forcé, dump par question sur LES DEUX dès le premier run
env LLVQ_DTYPE=f16 LLVQ_KV=f16 LLVQ_MMLU_DUMP=p3-mmlu-f16.csv \
    target/release/mmlu ~/qwen3-4b-llvq.bin metal 40
env LLVQ_DTYPE=f16 LLVQ_KV=q8  LLVQ_MMLU_DUMP=p3-mmlu-q8.csv \
    target/release/mmlu ~/qwen3-4b-llvq.bin metal 40
target/release/mmlupair p3-mmlu-f16.csv p3-mmlu-q8.csv     # SANS --intersect

# débit — ordre des arguments : [device] [n_new] [model]
env LLVQ_DTYPE=f16 LLVQ_KV=f16 target/release/gbench metal 128 ~/qwen3-4b-llvq.bin
```

- ⚠️ **`LLVQ_DTYPE=f16` des deux côtés est obligatoire** : `ppl` construit en
  **F32** (`ppl.rs:51`), `mmlu` (`mmlu.rs:287`) et `gbench` (`gbench.rs:66`) en
  **F16**. Quantifier un cache f32 vers u8 et un cache f16 vers u8 sont **deux
  expériences différentes**. La contrainte est **dans la commande gelée**.
- ⚠️ **`--intersect` est INTERDIT** : ce drapeau accepte délibérément deux dumps
  d'empreintes différentes (`mmlupair.rs:953-967`) ; son seul cas légitime, un
  recensement contre un run échantillonné, n'est pas P3.
- ⚠️ **`LLVQ_MMLU_DUMP` sur les DEUX bras dès le premier run** : les campagnes du
  08-06 et du 08-08 ont été lancées sans lui, leurs réponses par question sont
  **perdues** et se repaieraient **55 min de Mac par bras** (`mmlu.rs:49-64`).
- **L'empreinte de tokens est un CONTRÔLE DE SAISIE, pas un gate** : elle ne peut
  échouer que sur une erreur d'argument — même corpus, même tokenizer, et
  l'échantillon MMLU « depends only on the subject name's length »
  (`mmlu.rs:52-53`) ; `bin/mmlupair` la refuse déjà de lui-même. **Le mot
  « gate » est réservé au contrôle positif du §3.6.** Les valeurs historiques
  (`3f1baca9033bf251` ppl, `65dcd53655e8bfa5` MMLU) sont un **repère** dont les
  seules occurrences en `.rs` sont des **fixtures de test** (`mmlu.rs:718-722`,
  `mmlupair.rs:931-940`) : un repère qui tombe **suspend la publication du Δ
  contre l'historique**, sans invalider le Δ entre les deux bras du jour.

**2.5 — Le débit, et la faiblesse de son protocole, nommée ici.** 🚨 **C'est le
maillon faible du document** : `bin/gbench` charge **un** modèle par processus,
donc le rapport est **forcément un quotient inter-processus** — inéliminable
avec l'instrument existant, seulement atténuable.

- **invocations alternées** `f16, q8, f16, q8, f16, q8` — **3 paires** par
  `n_new`, jamais deux runs du même bras à la suite ;
- **`n_new ∈ {128, 1024}`**, pour la raison que le binaire argumente lui-même :
  « one prompt length would measure one point of a curve and be quoted as the
  curve » (`gbench.rs:34-36`). Il balaie déjà **deux prompts** (`:37-42`) ; les
  deux `n_new` ajoutent l'axe contexte. Soit **12 invocations** et **quatre
  séries** (§4.3) ;
- garde câblée : `n_new ≥ 2`, sinon « the KV cache is never exercised and the
  number is a prefill in disguise » (`gbench.rs:61-63`) ;
- 🚨 **Le budget de la série 1024 se décide sur la PREMIÈRE invocation f16, et
  jamais après.** Une invocation fait **trois** générations de `n_new` tokens —
  un warmup (`gbench.rs:109-124`) plus les deux prompts, soit 3 072 tokens
  décodés à `n_new = 1024`. Si elle dépasse **10 min**, la série 1024 est
  **abandonnée en entier — pas réduite** : le verdict est rendu sur la seule
  série 128 et **étiqueté « contexte court seulement »**, ce qui **interdit de
  servir le q8 par défaut quelle que soit sa valeur**. **`n_new` n'est jamais
  changé après la première invocation** — réduire après avoir vu l'horloge,
  c'est choisir le point le plus favorable après coup.

⚠️ **Le débit ainsi mesuré est un COÛT sans son BÉNÉFICE, et c'est le sens
conservateur** : à `n_new = 1024` le contexte atteint le millier de tokens
[estimé — les deux prompts font quelques dizaines de tokens et **leur compte
exact n'est pas dans le dépôt** ; il sera imprimé au journal, la colonne existe
(`gbench.rs:141`, `ids.len()`)], soit un quart de 4096 : la déquantification est
payée en entier, l'allègement du trafic de cache à peine exercé. Un vert est
solide ; un rouge se relit à la lumière de cette réserve **sans qu'elle serve à
le renégocier**.

**2.6 — Le coût machine, par poste, et son plafond.**

| poste | temps | provenance |
|---|---|---|
| build + les six gates V0 (§3) | ~20 min | **estimé** |
| ppl f16 | 5 min | **mesuré, mais sur `q4b-e8.llvq` et à cache f16** (`~/llvq-nuit-b/journal.txt:19-20`) |
| MMLU f16 | 55 min | **mesuré, même réserve** (`journal.txt:29-30`) |
| ppl q8 · MMLU q8 | **inconnus**, *estimés* au même ordre | le surcoût de déquantification est l'inconnue du lot |
| gbench | 12 invocations, ≤ 10 min chacune | **≤ 2 h**, borné par le §2.5 |

**Plafond global : 4 h, 0 $.** Au-delà, la série 1024 est **abandonnée** selon le
§2.5 — jamais réduite.

## 3. V0 avant V1 — l'exactitude d'abord, sans exception

**Aucune milliseconde chronométrée, aucun ppl lancé, avant que les six points
suivants soient verts.**

1. **Aller-retour par groupe contre une référence écrite indépendamment** du
   chemin de production — pas contre lui-même.
2. 🚨 **La branche dégénérée DOIT être exercée** : quand `(mx − mn)/255`
   s'arrondit à zéro en f16, `quantize_affine` force `s = 1`, met **tous les `q`
   à 0** et laisse le biais porter la valeur (`embedquant.rs:56-59`, `:65-71`).
   **Un groupe de 64 quasi-constant dans une tête est atteignable**, et sa
   variation intra-groupe est alors **jetée en silence**.
3. 🚨 **Le `round()` est porteur, et sa suppression doit casser un test** :
   `to_dtype(U8)` depuis un flottant **tronque vers zéro** dans le noyau Metal de
   candle (`candle-metal-kernels-0.9.2/src/metal_src/cast.metal:51`,
   `output[i] = static_cast<U>(static_cast<IR>(input[i]));`). Un cast direct
   produirait un **biais d'un demi-quantum** que **ni ppl ni MMLU ne
   signaleraient comme une faute de code**. Mutation exigée : retirer `round()`
   (`embedquant.rs:67`) ⇒ au moins un test rouge.
4. **Identité préfill ↔ décodage, au bit près** : le même (token, tête) produit
   **les mêmes octets** qu'il vienne d'un préfill de `l` tokens ou d'un pas de
   décodage. Traduction testable du §1.3.
5. **`LLVQ_VERIFY_CACHE=1` sur `bin/run`, tokens identiques.** Les deux chemins
   comparés — `generate` (`fresh_caches`, `model.rs:920`) et `generate_uncached`
   (`Block::forward`, `:726`) — passent par des **sites de construction
   différents** : concluant **seulement** si les deux portent le mode (§2.2.2).
   Garde interne : `n_new ≥ 3` (`run.rs:97-100`).
6. 🚨 **Le contrôle positif — c'est LE gate de ce document, et il manquait.** Un
   test unitaire exige que `Block::forward` sous `KvMode::Q8` rende un tenseur
   **différent** de celui rendu sous `F16` sur la géométrie du 4B (36 × 8 × 128),
   et **qu'au moins un groupe de 64 voie sa valeur modifiée**. Sans lui, un bras
   non branché rend Δppl = 0,000 %, ΔMMLU = 0,00 pp, débit 1,00× — et le §5 ne le
   voit pas, puisqu'il ne se déclenche que sur une *amélioration*.

⚠️ **Ces tests échouent quand leur fichier manque, ils ne sautent pas**
(`CLAUDE.md:1758`). Forme juste : `llvq-artifact/tests/common/mod.rs:26-49`.
**Tout écart enterre le bras sans mesure.**

## 4. Les seuils, posés avant la première mesure

Trois axes, trois verdicts indépendants. **Le quatrième — la mémoire — n'a pas
de seuil** : le gain est un **compte** (÷1,882) et ne peut pas échouer. Aucune
règle de maison numérotée ne l'exempte — `CLAUDE.md:1776-1790` n'en porte que
trois et aucune ne dit ça : **ce pré-enregistrement décide** que l'axe mémoire
est **rapporté**, pas **jugé**.

### 4.1 Qualité — perplexité

| mesure | verdict |
|---|---|
| Δppl ≤ **+0,5 %** *et* borne haute de l'IC95 apparié ≤ **+1,0 %** | **vert** |
| Δppl ≤ +0,5 % *mais* borne haute IC95 > +1,0 % | **non résolu** — relance **unique à 24 fenêtres** (~10 min/bras) ; si l'IC reste au-dessus de +1,0 %, l'axe qualité est **publié sans verdict** et le q8 **n'est pas servi par défaut** |
| Δppl ∈ ]+0,5 % ; +2,0 %] | **point de courbe** — non servi par défaut, publié avec son coût |
| Δppl > **+2,0 %** | **mort** en qualité |
| Δppl < −0,5 % (amélioration) | **chercher l'erreur avant d'en faire un titre** (§5) |

### 4.2 Capacités — MMLU micro stratifié

| mesure | verdict |
|---|---|
| \|Δ\| ≤ **1,0 pp** *et* IC95 apparié **entièrement contenu dans [−2,0 ; +2,0] pp** | **vert** |
| \|Δ\| ≤ 1,0 pp *mais* IC95 débordant [−2,0 ; +2,0] | **non résolu** — **décidé ici : l'axe capacités est publié sans verdict**, q8 non servi par défaut. Passer `limit` de 40 à 100 coûterait ~2,3 h/bras, plus que le plafond global de 4 h (§2.6) : ce n'est pas une option de ce lot |
| Δ ∈ [−2,0 ; −1,0[ pp | **point de courbe**, non servi par défaut |
| Δ < **−2,0 pp** | **mort** |
| Δ > **+1,0 pp** (q8 *meilleur* d'un point) | **chercher l'erreur avant d'en faire un titre** (§5) — d'abord un suspect de non-branchement ou de dump croisé |

⚠️ **La condition d'IC de la ligne 1 est porteuse** : la SE appariée du micro
stratifié vaut **0,79 à 1,44 pp** sur les paires connues (§1.5) ; sans elle, un
Δ de −0,9 pp serait « vert » alors que son IC95 pourrait descendre dans la bande
« mort » du même tableau. La SE mesurée par `bin/mmlupair` **accompagne** le Δ et
l'**étiquette** ; elle n'est pas le gate, et **aucun seuil n'est posé à 0,5 pp**
— ce n'est pas de la rigueur, c'est le refus d'un chiffre jamais mesuré.

### 4.3 Débit — `gbench`, Metal, modèle dense

**Une série = un couple (`n_new`, prompt), soit QUATRE séries de 3 paires.** Le
rapport d'une série est la **médiane de ses 3 rapports de même rang** ; `E` est
l'**étendue** de ces 3 rapports.

**La règle opératoire, posée ici et autoportante.** Le 08-10 §4 (`:113-123`)
exige un `Δ_contrôle` **intra-processus** que `gbench` ne peut pas produire
(`:108-111`) : elle ne s'applique pas et P3 ne s'en réclame pas. Celle de P3
est : **un verdict de débit n'est rendu, série par série, que si `E` ne contient
ni 0,80 ni 0,90.** Sinon la série est **non résolue** (§6).

| médiane des rapports de même rang | verdict |
|---|---|
| ≥ **0,90×** le bras f16, **sur les quatre séries** | **vert** |
| ∈ [0,80 ; 0,90[ sur au moins une série | **arbitrage** — non servi par défaut ; option de contexte long, explicitement activée |
| < **0,80×** sur une série quelconque | **mort comme chemin servi** |

⚠️ **Les trois lignes ne sont pas exclusives — la plus sévère l'emporte** : 0,85×
sur une série et 0,75× sur une autre ⇒ **mort**. Et le « sur les quatre séries »
est porteur : un seuil qui ne vaudrait que sur la série la plus favorable serait
un quotient de deux minima déguisé en critère.

### 4.4 La composition des trois

**Le vert d'ensemble exige les trois verts.** Un rouge de qualité **ou** de
capacités ferme le chantier ; un rouge de débit seul le déplace en option de
contexte long (§6). **Aucun des trois ne se rachète par un autre** — 0,283 Go
**n'achètent pas** un point de MMLU.

## 5. La prédiction, et ce qui ne la fonde pas

**Aucune fourchette n'est prédite, ni en qualité ni en débit.**

**Le précédent q8 embedding** — ppl **16,9379** contre 16,9415 (**−0,02 %**) et
MMLU **55,44 ± 1,35** contre 56,09 ± 1,36 (**−0,65 pp**), Metal
(`docs/mesures/nuit-b-2026-08-06.txt:47`, `:49`) — 🚨 **ne fonde rien ici** :
l'embedding est une table **statique quantifiée une fois**, le cache KV est
**produit à chaque pas, dans chaque couche, et relu en entier à chaque pas**.
Trois différences jouent **contre** la transposition : K **post-RoPE** (§2.3),
**token courant quantifié** (§2.3), **branche dégénérée** bien plus atteignable
sur 64 valeurs d'une tête (§3.2). ⚠️ **Son appariement MMLU n'est pas
vérifiable** : cette ligne **n'imprime aucune empreinte** (`:49`) et le 56,09
vient du run du 08-02, **antérieur** à leur impression ; `3f1baca9033bf251` porte
sur la **seule ligne ppl** (`:47`). C'est le reproche exact que le §1.5 adresse
au « σ McNemar ».

**Le verdict A2 sur `repeat_kv`** — ~2,4 Go par token à ctx 4096, ~3,6 ms à
662 Go/s — 🚨 **ne fonde rien non plus**, et sa source le dit : « Aucun code
modifié, aucun job lancé — la question se tranche par lecture de candle »
(`docs/archive/verdict-a2-repeat-kv-2026-08-06.md:4-6`). Le volume est
**calculé**, les 662 Go/s **supposés**.

🚨 **Ce qui fonde réellement les cinq seuils : des jugements d'ingénierie posés
d'avance, et rien d'autre.** **0,90× / 0,80×** est ce qu'une fonctionnalité
mémoire a le droit de coûter en débit — pas dérivé d'un budget de bande
passante, ni d'un compte d'instructions, ni d'un profil, **le profileur n'ayant
toujours jamais servi** (`CLAUDE.md` §2c) : c'est **le nombre le plus faible du
document**, au même titre que le 0,45 ns du P1. **+0,5 %** est le coût de
perplexité qu'une telle fonctionnalité a le droit de prendre sans être signalée,
**+2,0 %** le point où elle coûte plus qu'un demi-point de bits, **1,0 pp** et
**−2,0 pp** leurs traductions en capacités. **Aucun n'est dérivé d'une SE, d'un
calcul de puissance ou d'un précédent.** Tous sont posés d'avance **pour ne pas
être négociés après coup**, pas parce qu'ils sont solides ; leur direction est
conservatrice — couplés à la réserve du §2.5, ils peuvent tuer un cache q8 qui
gagnerait à contexte long, et un faux négatif coûte une idée là où un faux
positif coûte un format servi.

**Le garde-fou, dans les deux sens : un seuil dont on connaît le verdict d'avance
n'est pas un critère.** C'est pourquoi l'empreinte est rétrogradée en contrôle de
saisie (§2.4) et l'axe mémoire rapporté sans être jugé (§4). **Si un bras rend
mieux que prévu — Δppl < −0,5 %, ΔMMLU > +1,0 pp, ou débit > 1,0× — chercher
l'erreur avant d'en faire un titre** : bras non branché (premier suspect, §3.6),
empreintes discordantes, dump écrit par un seul bras, redirection `2>` oubliée,
`--intersect` passé par mégarde.

## 6. Les issues, et ce que chacune fait au dossier

| issue mesurée | conséquence, décidée d'avance |
|---|---|
| les **trois verts** | **P3 est acquis** : `LLVQ_KV=q8` câblé, mesuré en mémoire sur carte en P4 (`fusedrun`, colonne « Go carte »), et la colonne « KV q8 » de la note produit §B devient **chiffrable** — sans rien fermer d'autre (§0bis) |
| qualité verte, capacités vertes, **débit ∈ [0,80 ; 0,90[** | **option de contexte long**, jamais le défaut ; **seul cas** où la variante g128 (8,25 b/valeur, §1.3) est mesurée |
| qualité verte, capacités vertes, **débit < 0,80×** | **mort comme chemin servi sur Metal dense** ; réouverture **uniquement** sur une évidence neuve côté CUDA fusé, jamais sur une relecture du même run |
| qualité verte, capacités vertes, **débit non résolu** (`E` recouvrant 0,80 ou 0,90 sur ≥ 1 série) | le bras se relance **UNE fois à 5 paires** ; si `E` recouvre encore un seuil, **P3 reste ouvert sans verdict de débit**, `LLVQ_KV=q8` **n'est PAS servi par défaut**, et le point se publie **non résolu à cette résolution** |
| débit vert, **qualité et/ou capacités non résolues** (§4.1, §4.2) | l'axe concerné est **publié sans verdict** après sa relance unique ; **q8 non servi par défaut** ; P3 reste ouvert sur cet axe |
| verdict de débit rendu **sur la seule série 128** (§2.5) | verdict **étiqueté « contexte court seulement »** : q8 **jamais servi par défaut**, quelle que soit sa valeur |
| **Δppl > +2,0 %** ou **ΔMMLU < −2,0 pp** | **P3 est mort** ; la colonne « KV q8 » de la note produit §B est **retirée** |
| l'un des deux axes qualité en **point de courbe**, l'autre vert | **non servi** ; publié comme point de la courbe qualité↔mémoire du KV, avec sa provenance |
| **V0 rouge**, contrôle positif du §3.6 compris | **aucun chiffre, aucun verdict** : correction d'abord, entorse au §7bis |

⚠️ **Ce qu'un vert de P3 demande à P4 est une MESURE DE MÉMOIRE, pas un chiffre
de débit : P3 ne pose aucun critère sur le noyau de P4.** *(Décision d'opérateur
du 2026-08-14, notée ici pour que les deux documents ne dérivent pas : le critère
**K2 de P4 se lit PAR COLONNE** — `T(k=8)/8 ≤ 0,60 × T(k=1)`, soit
`T(k=8) ≤ 4,80 × T(k=1)`. La forme « `T(k=8) ≤ 0,60 × T(k=1)` » est
arithmétiquement impassable : un noyau à k=8 fait 8× les FMA et 8× les stores de
k=1.)*

**Aucune de ces issues ne dépend de P1 ni ne le bloque** : P3 ne touche à aucun
décodeur de rang, à aucun layout VRAM, et se mesure sur un modèle dense.

## 7. Ce qui invaliderait ce pré-enregistrement

- **si le sha256 ne tombe pas** (§2.1), aucune mesure n'a lieu — la taille seule
  ne l'établit pas (584 o séparent le scellé de son jumeau L4swap présent) ;
- 🚨 **si le bras q8 n'était pas actif**, aucun verdict n'est rendu et le run est
  **rejeté**. Signatures : **les 12 NLL par fenêtre identiques entre les deux
  bras** à 9 décimales, ou **les deux dumps MMLU identiques ligne à ligne**.
  L'impression de `LLVQ_KV` ne vaut pas preuve (§2.2.4) ;
- **si les deux bras n'impriment pas la même empreinte**, **aucun Δ n'est
  formable** ;
- **si les lignes par fenêtre de `ppl` sont absentes** (redirection `2>`
  oubliée), **le bras se relance** — publier un Δ ppl sans intervalle apparié est
  **interdit** ; ça coûte 5 minutes (§2.6) ;
- **si un seul des deux bras MMLU a écrit son dump**, la statistique appariée est
  impossible et **les 55 min de l'autre bras sont à repayer** ;
- **si `LLVQ_VERIFY_CACHE=1` échoue** (§3.5), la granularité traverse l'axe temps
  ou un seul site porte le mode : le bras n'existe pas, et **aucun chiffre de
  qualité produit avant ce constat n'est publiable** ;
- **si `grep -c 'std::env' llvq-llm/src/model.rs` ne rend plus 0** après le
  patch, **aucune mesure n'a lieu tant que le compte n'est pas revenu à 0** —
  même forme binaire que la règle du sha256 ;
- **si le transitoire du §1.4 matérialise plus d'une couche à la fois**, la
  comptabilité mémoire se refait **avant** tout verdict ;
- **si `E` recouvre un seuil du §4.3** après la relance à 5 paires, le verdict de
  débit **n'est pas rendu** (§6) ;
- **si un lecteur tire de P3 un verdict sur le barreau 32 Go ou sur une tolérance
  « capacity-first »**, il sort de ce document : **aucune de ces deux choses n'a
  de nombre en face** (§0bis).

## 7bis. Écarts au protocole — journal, tenu à chaud

*(Chaque entorse s'écrit ici le jour où elle est commise, avec sa raison et son
coût — la règle du 08-10.)*

### É0 — 2026-08-14, à la signature : les quatre dérogations aux gardes du 08-10

Inscrites **avant** toute mesure, pas découvertes après (ancres dans le bloc
d'héritage en tête) : **(1) un seul processus, bras entrelacés** — impossible,
`gbench` charge un modèle par processus, remplacé par **3 paires alternées**,
rapport formé paire par paire ; **(2) 7 rounds dont 2 jetés** — remplacé par
**3 paires** plus le warmup de `gbench.rs:109-124` ; **(3) vérification f64 ligne
à ligne** — sans objet, aucun noyau écrit, remplacée par les **six gates V0** du
§3 dont le contrôle positif ; **(4) rapport de registres et coût GPU par job** —
sans objet, Metal local, 0 $, pas de `jobs.csv`.

**Ce que ça coûte, dit franchement** : la résolution du bras débit est plus
faible qu'aucun banc GPU publié par ce projet. D'où la règle de non-résolution
explicite du §4.3 et la conséquence que le §6 lui attache.

**Aucune autre entorse à la signature.** Aucune ligne de code écrite, aucune
mesure prise.

## 8. Ce qui est connu à la signature — divulgation datée

Les seuils du §4 sont posés aujourd'hui, **avant** la première ligne de
`kvq.rs`, et ils ne bougent pas.

- **Aucune ligne de quantification de cache KV n'existe** : aucun compte
  d'instructions, aucun profil, aucune milliseconde.
- **Le précédent q8 embedding est connu et écrit ici** (§5) : ppl 16,9379 contre
  16,9415, MMLU 55,44 contre 56,09 — ce dernier delta à **appariement non
  vérifiable**. Le §5 dit pourquoi il ne fonde aucun seuil.
- **Les six SE appariées du 2026-08-13 sont connues et tabulées** (§1.5) ; elles
  servent à **interdire** le 0,5 pp hérité, pas à fixer un seuil. **Les deux
  seuils hérités de la passation** — « ppl > 0,7 % » et « σ McNemar
  0,4-0,6 pp » — sont connus et **rejetés**, motifs au §1.5.
- **La géométrie du cache est connue et CALCULÉE** : 147 456 o/token en f16,
  78 336 en q8 g64, ÷1,882 — un **compte**, donc pas un critère (§4). ⚠️ Sa
  source est le dépôt (`docs/fiche-4b.md:78`), **pas un `config.json`** : aucun
  n'est présent sur cette machine.
- **Le fichier de référence est absent de la machine**, récupérable à l'octet
  près, et un jumeau à 584 octets près y est présent (§2.1).
- **`bin/gbench` n'a jamais produit de journal** : le premier chiffre de débit du
  projet sur cet instrument sera celui de P3, **sans antécédent contre lequel le
  lire**.
- **La note produit du 2026-08-13 n'est pas opposable ici** : cases §A vides
  (`:20-25`), et son passage KV f16→q8 vaut **÷2,03** — donc ~8,0 b/valeur sans
  échelles, *calculé depuis son propre tableau* (`:61-69`) — alors que la **même
  formule** facture l'embedding q8 à 8,5 (`:62`). Elle ne nomme ni le modèle de
  sa géométrie KV, ni le batch, ni la granularité, ni l'unité de son « 32 Go ».
  **Aucun seuil n'est fixé contre elle** ; si P3 est vert, il lui fournit le seul
  chiffre KV citable avec sa provenance.

**Ce qui reste OUVERT, nommé pour qu'on ne le découvre pas après** : la
résolution du bras débit (§7bis É0) ; l'absence de tout antécédent `gbench` ; la
SE appariée de la paire de P3, qui est une sortie et non une entrée (§1.5) ; le
compte exact de tokens des deux prompts de `gbench` (§2.5) ; et les deux choses
que P3 ne peut pas fermer faute d'un nombre en face (§0bis).
