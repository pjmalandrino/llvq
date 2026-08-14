# Pré-enregistrement — P2 : ce qu'un dump de routage temporel peut trancher, et le modèle MoE qu'on accepte de quantifier

**Date : 2026-08-14.** Écrit **avant toute mesure**. Emplacement canonique :
`proofs/preregistration-p2-2026-08-14.md` — **les liens relatifs résolvent
depuis `proofs/`**, et le fichier y est déposé avant le tampon. À cette heure :

- **aucun hit LRU n'existe**, sur aucun modèle. Le seul dump est
  [`docs/data/moe-routing-gptoss20b-2026-08-12.json`](../docs/data/moe-routing-gptoss20b-2026-08-12.json),
  **agrégé** : 5 675 octets, **8 clés**, aucune temporelle (mesuré) ;
- **l'information temporelle est détruite DANS LE HOOK**, pas à l'écriture —
  `ops/moe_routing.py:94` (`reshape(-1, …)`), `:97` (`topk(…).reshape(-1)`),
  `:98` (`bincount`). Le hook doit être **modifié** ;
- **le simulateur LRU n'a zéro ligne de code** (`ops/moe_lru.py` : absent) ;
- **aucun crate Rust n'a la moindre notion d'expert, de MoE ou de routeur**
  (`grep -rniE "expert|moe|router|gating|top_k|topk" --include='*.rs' .` → une
  seule ligne, faux positif sur « propagating », `llvq-llm/tests/rotplan.rs:66`) ;
- **aucun MoE n'a jamais été quantifié à 2 bits**, chez nous ni ailleurs à
  notre connaissance. P2 **ne remplit pas** cette case — c'est P6.

Dernier verdict, lot du 13 ([`preregistration-2026-08-13.md`](preregistration-2026-08-13.md)
§T4, journal [`moe-ciseau-2026-08-13.txt`](../docs/mesures/moe-ciseau-2026-08-13.txt)) :
la variante chaud/froid **en VRAM** est enterrée (mélange 3,268–3,405 contre
3,20, `:156-157`) et la branche **LRU** y est tuée par l'arithmétique du
backing puis par un hit d'**oracle statique** de 74,08 % là où il en fallait
99,8 % (`:176-182`). P2 juge ce que ce verdict ne pouvait pas juger : **le
LRU**, sur l'alternative survivante — le **tier froid en RAM hôte**.

> Il **hérite sans dérogation** des gardes du [2026-08-10](preregistration-2026-08-10.md)
> (§7), de sa règle de provenance et de sa comptabilité (§6).
>
> ⚠️ Ni signé GPG ni horodaté tant que l'opérateur ne l'a pas fait
> (`ots stamp proofs/preregistration-p2-2026-08-14.md`). Le tampon est
> **porteur** : demandé avant la première ligne du hook modifié. *(Seuls les
> 08-10 et 08-11 portent un `.ots` ; ni le 08-13 ni P1 — mesuré.)*

---

## 0. Ce que ce test peut et ne peut pas conclure

**Ce qu'il mesure** : sur un **gpt-oss-20b déquantifié en bf16**, 24 couches,
32 experts, top-4, 131 072 tokens de C4, la **fraction des visites de cellule
servies par un cache LRU** de taille α, dans l'ordre du **décode
mono-séquence**. Rien d'autre.

**Ce qu'il ne mesure pas** : aucune milliseconde — aucune mesure PCIe n'existe
dans ce dépôt (`ops/moe_ciseau.py:97-99`), les 32/50/63 Go/s sont des ordres de
grandeur constructeur ; aucune qualité ; le routage du 120b, jamais capturé ;
le routage du **MXFP4 natif** — le dump du 12 a été pris sur un modèle
déquantifié en bf16 pour MPS ([`…-2026-08-12.txt:12`](../docs/mesures/moe-routing-gptoss20b-2026-08-12.txt)).

🚨 **L'asymétrie est inscrite d'avance.** Le dump est un **20b, 32 experts
top-4** (96 visites/token) ; la cible un **~120b, 128 experts top-4** (144
visites, 36 couches — calculé, `ops/moe_ciseau.py:268`). **UN VERT NE SE
TRANSPORTE PAS ; UN ROUGE, SI.**

## 0bis. Prérequis PORTEURS, et les tolérances qui ne ferment rien

**(a) La fiche produit doit être arbitrée et commitée AVANT le tampon.**
[`docs/note-produit-2026-08-13.md`](../docs/note-produit-2026-08-13.md) est
**non suivie par git** (`??`, vérifié) et déclare n'être opposable tant que ses
cases ne sont pas cochées (`:3-4`). Trois portent ce document :

| case | ce qu'elle décide ici | état |
|---|---|---|
| **A1** — carte cible (`:25-30`) | **le point de verdict α** (§2.5) : 32 Go ⇒ 0,5868 · 24 Go ⇒ 0,4344 · Mac unifié ⇒ **pas de bus PCIe, le test ne s'applique pas** | ⬜ vide |
| **A6** — offload PCIe « référence » ou « servi » (`:54-57`, « servi aussi (package A en dépend) ») | **l'architecture même du test** : le tier froid en RAM hôte EST cet offload. Coché « référence seulement », P2 mesure une solution qu'on ne livre pas | ⬜ vide |
| **génération PCIe** | le seuil vert du §4.1 est calé sur **gen4**, alors que la seule carte recommandée est une **5090 PCIe gen5** (`:25-26`) | non cochée |

Sans ces trois arbitrages, α = 0,5868 et le lien gen4 sont **indéfendables** :
« le rouge était en gen4, notre carte est gen5 » requalifierait le verdict après
coup. ⚠️ **Et la fiche n'arbitre PAS le statut de `Golay70`** (servi /
référence) : son `:57` est la case **A6, l'offload PCIe**, pas un choix de
layout. Le layout servi du 120B n'est arbitré **nulle part** (§1.5, §2.5).

**(b) La tolérance de dégradation est une CONVENTION PRODUIT, figée ici.** Les
**10 %** du §4.0 ne sont ni mesurés, ni calculés, ni estimés : un choix, hérité
du ciseau où il est un littéral à deux sites (`ops/moe_ciseau.py:473`, `:514`),
pas même une constante nommée. Le budget lui est **strictement proportionnel**,
donc le seuil aussi. **Figée ici, elle ne se recalcule pas après la mesure** ;
la courbe seuil(tolérance) est publiée au §4.0 et **seule la ligne 10 % porte
verdict**. Elle se coche dans la note produit avant le tampon.

**(c) Une tolérance non chiffrée ne ferme aucune décision.** La tolérance dite
« **capacity-first** » n'est chiffrée **nulle part** :
`passation-exec-2026-08-13.md:106`,
`etude-moe-memoire-extreme-2026-08-12.md:179-181` et trois renvois dans
[`preregistration-p1-2026-08-13.md`](preregistration-p1-2026-08-13.md) —
**aucune occurrence ne porte de nombre** (vérifié le 2026-08-14). Tant qu'elle
n'est pas arbitrée dans un document, **aucune issue ne se ferme en
l'invoquant** — y compris « la cascade-archive passe la tolérance
capacity-first », qui **ne peut pas fermer P5**. P2 ne l'invoque nulle part ; la
règle vaut pour tout autre plafond non posé (cf. C5, §4.3).

## 1. La comptabilité, figée ici

**1.1 — L'unité de charge est la CELLULE `(couche, expert)`, jamais
l'expert.** Chaque couple a sa hessienne et son bloc de poids ; la calibration
est figée à **quatre hessiennes par bloc**, une par activation partagée
(`llvq-llm/src/model.rs:51`, `:65-72`) — aucune place pour une hessienne par
expert. Le dump en porte **768 = 24 × 32** (mesuré).

🚨 **Le piège d'agrégation est celui du macro/micro de MMLU**, et le script le
relève (`ops/moe_routing.py:126-129`) : Gini **agrégé** 0,169
([`…-2026-08-12.txt:65`](../docs/mesures/moe-routing-gptoss20b-2026-08-12.txt))
contre **0,351** (couche 0) à **0,748** (couche 23) par couche (`:40-63`,
mesurés). **Aucun chiffre de ce document ne se lit sur une distribution agrégée
par expert.**

**1.2 — L'unité de résultat est le hit sur les VISITES DE CELLULE**, modèle
entier : `miss/token = (1 − hit) × visites/token`, avec `visites/token = K × L`
= **96** au 20b (mesuré, K=4 L=24) et **144** au 120b (calculé, K=4 L=36,
`ops/moe_ciseau.py:268`). ⚠️ **Tous les seuils sont exprimés à 144 visites**, la
conversion stricte ; les 96 ne figurent qu'en regard et ne portent aucun verdict.

**1.3 — α est une fraction de CELLULES résidentes**, `k = ⌈α × 768⌉` (**ceil** ;
le ciseau utilise `round`, `ops/moe_ciseau.py:493`, et les deux coïncident sur
les huit α du §2.5 — vérifié). Cache **global**, l'allocation la **plus
favorable** à la piste (`ops/moe_ciseau.py:145-147`). La variante par couche à
budget uniforme est un encadrement, jamais un verdict.

**1.4 — La mémoire se dit en b/param MODÈLE ENTIER, embedding compris** (sur un
MoE il est un epsilon, 0,3–2 % au-delà de 80 Md, `etude…:34-38` — la convention
se réénonce quand même). La VRAM ne porte que le chaud :
**`VRAM(α) = 117 Md × α × 3,589/8 + 1,2 Go = 52,489·α + 1,2`** (calculé).

⚠️ **Le 1,2 Go est une convention, pas une mesure** : il ferme exactement
`117e9 × 3,20/8 = 46,8 ; +1,2 = 48,0`. Le commentaire qui l'affirme
(`ops/moe_ciseau.py:80-82`) n'est qu'un commentaire, `CARD_GB = 48.0` (`:83`)
est **défini et jamais référencé**, les classes de carte sont en dur aux lignes
`:499` et `:517`. **Ce 1,2 Go EST la marge** — il n'y en a pas d'autre
([`moe-ciseau-2026-08-13.txt:230-231`](../docs/mesures/moe-ciseau-2026-08-13.txt)).

**1.5 — `B_HOT = 3,589` est `Golay70`, refusé deux fois de servir** (écarté le
08-07 à 1,31× contre 1,6× posé d'avance ; v2 **non adoptée** le 08-11 à 1,77×
contre 2,0×, [`preregistration-2026-08-11.md`](preregistration-2026-08-11.md)) :
`LLVQ_FUSED_LAYOUT` l'**admet**, aucun chemin de production ne le **sert**. Ce
document conserve ce choix sur la foi du renversement de critère de l'étude MoE
(`etude…:9-16`) : **toute ligne portant un 3,589 hérite d'un layout hors
production.**

🚨 **Le tier froid est transféré en format SERVI (3,589 b/poids).** La variante
« archive sur le bus » (2,219, −38 % de trafic,
[`moe-ciseau-2026-08-13.txt:210-213`](../docs/mesures/moe-ciseau-2026-08-13.txt))
est une **AUTRE architecture** : elle exige un transcodage device-side dont le
coût ALU n'est mesuré nulle part, le dépôt ne sachant convertir un format qu'**au
chargement** (`LLVQ_EMBED`, `llvq-llm/src/fused.rs:132-148`), pas par miss.
**Elle ne peut pas être invoquée après coup pour requalifier un verdict** ; elle
se pré-enregistre séparément, et ses seuils sont publiés au §4.0.

**1.6 — Le régime est le DÉCODE MONO-SÉQUENCE, batch = 1.** Aucun résultat ne
se transporte à B > 1, **dans aucun sens** : jusqu'à B×K cellules par couche et
par pas, donc la masse froide croît pendant que la mutualisation des miss joue
en sens inverse — effet net ni mesuré ni borné (dette (iii) du §7 du ciseau).
**Le package A doit porter « une session à la fois », ou refaire ce test.**

**1.7 — Provenance sur chaque nombre** : *mesuré* / *calculé* / *estimé*, avec
sa comptabilité. Un nombre sans étiquette n'entre pas dans le journal.

## 2. Le protocole, figé ici

### 2.1 Ce que le hook doit stocker

Le hook modifié **écrit, sans les agréger** : pour chaque (fenêtre `w`, couche
`ℓ`), les indices d'experts du top-K **dans l'ordre de score décroissant** (la
variante croissante s'en dérive par tri, l'inverse est impossible).

**Format de la trace** : un en-tête d'une ligne JSON (modèle, n_experts, top_k,
hidden, moe_intermediate, tokens, ctx, device, dataset, layer_names, ordre des
K, dtype de l'index, **`transformers_version`**, **`tokens_sha256`**), puis :

```
uint8[ n_tokens ][ n_layers ][ top_k ]      si n_experts ≤ 256
uint16 little-endian, même forme            sinon   (règle posée d'avance)
```

**`tokens_sha256`** = SHA-256 du buffer des identifiants de tokens sérialisés en
**uint32 little-endian**, ordre de production, longueur `n_tokens` ; imprimé en
tête du journal **et** dans l'en-tête. ⚠️ **Elle ne peut PAS prouver l'identité
du flux avec celui du 12** (aucune empreinte n'existe pour lui — ABSENT) : elle
ne sert qu'aux runs postérieurs, le seul contrôle disponible ici restant V0.1.
**`transformers_version`** est exigé parce que l'en-tête PEP-723 ne pin aucune
version (`ops/moe_routing.py:4`) et que celle du run du 12 est **ABSENTE** du log.

**Coût** : `131 072 × 24 × 4 = 12 582 912` entrées, **12,58 Mo** en uint8
(calculé), **2 217×** le dump agrégé. **La trace vit hors du dépôt**, à
`~/llvq-moe/` ; entrent dans le dépôt le JSON ré-agrégé, le journal
`docs/mesures/` et le **SHA-256 de la trace** en tête du journal. ⚠️ **Un
fichier absent doit faire ÉCHOUER, jamais sauter** (forme correcte à
`ops/moe_ciseau.py:206-208`).

### 2.2 La ligne de commande — elle n'est écrite nulle part aujourd'hui

🚨 **Le run du 12 n'a PAS tourné avec les défauts du script** et sa commande est
**ABSENTE** : paramètres en prose dans le log (`…-2026-08-12.txt:3`, `:6`), et
le JSON n'enregistre ni `ctx`, ni `device`, ni `dataset`, ni graine. Les défauts
sont `--ctx 2048` (`ops/moe_routing.py:45`) et `--device cpu` (`:46`) ; le run a
utilisé **1024** et **mps**.

```bash
uv run ops/moe_routing.py \
  --model openai/gpt-oss-20b --tokens 131072 --ctx 1024 --device mps \
  --dataset allenai/c4 \
  --json  docs/data/moe-routing-gptoss20b-2026-08-14.json \
  --trace ~/llvq-moe/trace-gptoss20b-2026-08-14.u8

python3 ops/moe_lru.py --trace ~/llvq-moe/trace-gptoss20b-2026-08-14.u8 \
        > docs/mesures/moe-lru-2026-08-14.txt
```

`--trace` est **à ajouter** ; les six autres existent
(`ops/moe_routing.py:41-54` — **sept** arguments, `--from-json` compris).
`ops/moe_lru.py` est **à écrire**, **stdlib seule** (0 $, aucun GPU, aucun réseau).

### 2.3 Le budget de temps — pourquoi « ~438 s » serait faux

Le chronomètre est remis à zéro **après** le chargement
(`ops/moe_routing.py:219`, juste après `:210-213`).

| poste | valeur | provenance |
|---|---|---|
| boucle de passes avant | **438 s** | mesuré, [`…-2026-08-12.txt:32`](../docs/mesures/moe-routing-gptoss20b-2026-08-12.txt) |
| chargement du modèle | **14 s** | mesuré, `:13` |
| tokenisation du corpus | **ABSENT** | jamais chronométrée |
| surcoût de collecte temporelle | **ABSENT** | jamais chiffré |
| simulation LRU | **ABSENT** | le script n'existe pas |

**Le journal annoncera le temps total mesuré du re-run, pas les 438 s** — que la
passation (`docs/archive/passation-exec-2026-08-13.md:123`) cite sans dire qu'ils
excluent chargement et tokenisation.

### 2.4 L'ordre de simulation, et l'argument qui l'autorise

Ordre du **décode** : `pour t = 0..131 071 : pour ℓ = 0..23 : les K cellules`.
**Ce n'est pas l'ordre d'exécution du dump** — un prefill traite les 1024 tokens
d'une fenêtre à la couche 0, puis à la couche 1, ordre dans lequel un cache n'a
aucun sens. **L'argument qui autorise la transposition** : le routeur est
causal, donc la décision pour le token *i* ne dépend que de son préfixe.
⚠️ **C'est un argument, pas une mesure** — en bf16, prefill batché et décode
token-à-token peuvent différer sur un départage numérique ; aucune vérification
n'existe, P2 n'en produit pas, **la réserve est portée dans le journal**.

**Frontières de fenêtre** : les 128 fenêtres sont concaténées dans leur ordre de
production, le cache **traverse** — déclaré, pas corrigé. **Démarrage à froid** :
les miss obligatoires **sont comptés** (le choix strict) ; au point de verdict,
451 cellules sur 12 582 912 visites, **0,004 %** (calculé).

### 2.5 Les α mesurés, et la RÈGLE qui fixe le point de verdict

**Le point de verdict n'est pas un nombre choisi, c'est une règle** :
**α_verdict = le plus grand α que la carte du §A1 admet sous la comptabilité du
§1.4**, soit `(VRAM_carte − 1,2)/52,489`. **Les trois valeurs possibles sont
publiées ci-dessous** (colonne « rôle »), donc aucune ne peut être choisie après
la mesure ; le cas **Mac à mémoire unifiée** est **sans objet** — pas de bus
PCIe, donc pas de verdict (§6).

Huit α mesurés, `⌈α × 768⌉`, VRAM calculée par `52,489·α + 1,2` — donc avec un
`Golay70` hors production (§1.5) et un 1,2 Go de convention (§1.4) :

| α | cellules | VRAM | rôle |
|---|---|---|---|
| 0,2733 | 210 | 15,5 Go | limite arithmétique de la branche LRU en VRAM, tuée le 13 |
| 0,3500 | 269 | 19,6 Go | point de courbe |
| 0,4344 | 334 | 24,0 Go | α_verdict **si A1 = 24 Go** |
| 0,4384 | 337 | 24,2 Go | α_verdict **si le layout servi est `Planes14`** (32,0 Go dans cette comptabilité) |
| 0,4500 | 346 | 24,8 Go | point de courbe (l'α proposé par la passation) |
| 0,5000 | 384 | 27,4 Go | point de courbe |
| **0,5868** | **451** | **32,0 Go** | **α_verdict si A1 = 32 Go et layout `Golay70`** |
| 0,7161 | 550 | 38,8 Go | α_VRAM, la lame mémoire du ciseau |

⚠️ **Le verdict est posé à α_verdict et à lui seul** ; les sept autres sont des
points de courbe et **ne peuvent porter aucun verdict après coup**.

## 3. Exactitude avant lecture — V0 avant V1, sans exception

### V0.1 — Le contrôle qui MORD, en remplacement de celui qui ne mord pas

🚨 **Le contrôle de somme du ciseau est TAUTOLOGIQUE** : `ok_sums =
per_layer_sums == {ntok * K}` (`ops/moe_ciseau.py:233`) compare chaque couche à
`d['tokens'] × K`, or `d['tokens']` est **dérivé de la couche 0**
(`ops/moe_routing.py:234` ; vérifié : `d['tokens'] = 131072` et
`sum(counts[0]) // 4 = 131072`). Vrai par construction pour la couche 0 ; pour
les 23 autres il ne teste que l'égalité du nombre de passes avant. **Aucune
décision de routage.**

**Le contrôle qui le remplace** : ré-agréger la trace en `[24 × 32]` et exiger
l'**égalité cellule par cellule sur les 768 cellules** avec le dump du 12 —
égalité issue de 12 582 912 décisions, donc preuve conjointe que le flux s'est
reproduit, que le hook compte la même chose et que la trace n'a rien perdu.

⚠️ **Il peut échouer sans bug** : `load_calibration_tokens`
(`ops/moe_routing.py:57-74`) consomme `load_dataset(..., streaming=True)` **sans
`shuffle` ni `seed`**, la stabilité de l'ordre de streaming de `allenai/c4`
n'est ni contrôlée ni documentée, et la version de `transformers` du 12 est
ABSENTE. **Branche décidée d'avance, pour que P2 ne meure pas sur un événement
extérieur** — si l'égalité échoue, le nouveau dump devient la référence à
**trois conditions cumulatives vérifiées AVANT toute lecture** :

1. les **24 Gini par couche** sont à **±0,01** de ceux du 12 (`…:40-63`) ;
2. le classement des 768 cellules par charge a un **Kendall-τ ≥ 0,95** avec
   celui du 12 ;
3. la table d'oracle statique du §8.1 est **intégralement recalculée** sur le
   nouveau dump et republiée à côté de l'ancienne.

Si l'une échoue, **le run est nul**. Si les trois passent, la courbe est lue sur
le nouveau dump et le journal écrit que le flux C4 a dérivé.

### V0.2 — Lire la décision EXÉCUTÉE — tranché pour gpt-oss

🚨 Aujourd'hui le hook **re-dérive** le routage : `out[0]` (`:91`) puis son
propre `topk` (`:97`). **Tranché à 0 $ avant le tampon**, source lue le
2026-08-14 : `…/transformers/models/gpt_oss/modeling_gpt_oss.py` (transformers
**5.5.4**) — `GptOssTopKRouter.forward` (`:126-130`) rend un **3-uplet
`(router_logits, router_scores, router_indices)`** ; `router_indices` est produit
à `:128` par `torch.topk(router_logits, top_k, dim=-1)`, et c'est **ce membre**
que le MLP transmet au dispatch (`:143`, puis `GptOssExperts.forward(…,
router_indices, …)` à `:90`).

**Exigence** : la trace stocke **`out[2]`**, sans re-dérivation ; à défaut,
égalité stricte entre le `topk` du hook et `out[2]` sur **au moins une fenêtre
entière**. `torch.topk` rendant ses valeurs en ordre décroissant, `out[2]`
satisfait déjà l'ordre du §2.1. ⚠️ Confondant nommé : le hook applique
`.float()` avant son `topk` (`:94`) là où le module travaille en bf16 — l'upcast
étant injectif et monotone, seul un départage entre valeurs **exactement égales**
peut différer. La clause ABSENT ne vaut **que pour une architecture FUTURE** :
**un ABSENT sur gpt-oss-20b est un échec de V0, pas une réserve.**

### V0.3 — Aucune couche perdue, sans réintroduire la tautologie

Une couche dont le hook rend une forme inattendue est **ignorée en silence**
(`ops/moe_routing.py:92-96`, deux `return` muets) ; seul « zéro hook accroché »
lève (`:108-109`). **Exigence** : le compte attendu est
`windows.shape[0] × windows.shape[1] × K`, lu sur la **FORME DU CORPUS**
(`ops/moe_routing.py:206-207`), **jamais** sur `total[0].sum()` ; la ligne
`:234` est supprimée ou remplacée par une **assertion d'égalité entre les
deux**, ce qui fait du contrôle un test et non une définition. Toute couche dont
le compte diffère **fait échouer le run**.

### V0.4 — Le simulateur vérifié sur flux synthétiques AVANT la trace

**Politique** : LRU vrai — admission systématique au miss, **rafraîchissement de
la récence au hit**, éviction du moins récemment utilisé. Cinq cas :

1. ensemble de travail `≤ taille du cache` ⇒ **hit → 100 %** (miss obligatoires
   comptés) ;
2. tourniquet sur `taille + 1` ⇒ **hit = 0 %** ;
3. flux i.i.d. **plat** ⇒ hit ≈ taille / nombre de cellules ;
4. **le cas qui tue le FIFO** — taille 2, flux `A B A C A` : un LRU garde
   `{A,C}` et rend **2 hits sur 5**, un FIFO évince A à l'insertion de C et rend
   **1 sur 5**. Les cas 1-3 sont passés par un FIFO à l'identique, or le
   rafraîchissement au hit est **le seul mécanisme par lequel un LRU peut battre
   l'oracle statique** — donc le paramètre porteur de tout le §4.2 ;
5. **le cas qui exerce l'ordre des K** — taille 2, K = 2, pas 1 = `(X rang 1,
   Y rang 2)`, pas 2 = `(Z rang 1, X rang 2)` : insertion en score **décroissant**
   (règle du §2.1) ⇒ **0 hit sur 4** ; en ordre inverse ⇒ 1 sur 4.

Aucune ligne de la trace n'est lue avant que les cinq passent. **Tout échec de
V0 enterre la lecture, pas seulement le chiffre.**

## 4. Les seuils, posés avant la première mesure

### 4.0 Le budget de miss — recalculé, et l'écart de provenance nommé

🚨 **La borne « 0,75 ms le miss » de la passation
(`docs/archive/passation-exec-2026-08-13.md:129`) n'a AUCUN antécédent.** Le
ciseau donne **0,349 / 0,223 / 0,177 ms** en servi
([`moe-ciseau-2026-08-13.txt:201-208`](../docs/mesures/moe-ciseau-2026-08-13.txt)),
et le tableau de synthèse de la **même** passation (`:239`) donne « ~0,35 gen4 /
~0,22 gen5 », conforme. **Ce document retient le ciseau et retire le 0,75.**

| terme | valeur | provenance |
|---|---|---|
| temps/token 120B, tout chaud | **11,733 ms** | *estimé* — 5,1 Md × 3,589/8 = 2,288 Go ÷ 195 Go/s ; **plancher** (plafond de bande passante) ; 195 Go/s mesurés **CUDA/L40S** sur `Golay70` v1 |
| cellule d'expert, servi | **11,163 Mo** | *calculé* — 3 × 2880 × 2880 = 24 883 200 poids × 3,589/8 ; ⚠️ suppose `hidden = moe_inter = 2880` **au 120B** (`ops/moe_ciseau.py:284-287`) |
| miss = memcpy PCIe, servi | **0,3489 / 0,2233 / 0,1772 ms** | *estimé* — ~32 / ~50 / ~63 Go/s. **Aucune mesure PCIe ici** |
| budget à **+10 %** (convention §0bis b) | **3,3634 / 5,2553 / 6,6217** miss/token | *calculé* |

```
budget(BW)     = 0,10 × (5,1e9 × 3,589/8 ÷ 195e9) ÷ (24 883 200 × 3,589/8 ÷ BW)
hit_requis(BW) = 1 − budget(BW) / 144
```

**Les trois seuils, à quatre décimales — aucun arrondi n'intervient, dans aucun
sens** (la faute de l'É0 du 08-11) :

| lien | budget miss/tok | **hit requis, transporté 120B** | (en regard) natif 20b |
|---|---|---|---|
| gen4 x16 (~32 Go/s) | 3,3634 | **97,6643 %** | 96,4965 % |
| gen5 x16 (~50 Go/s) | 5,2553 | **96,3505 %** | 94,5257 % |
| gen5 x16 (~63 Go/s) | 6,6217 | **95,4016 %** | 93,1024 % |

**Les familles NON retenues, imprimées pour qu'on ne puisse pas les découvrir
après coup** (même formule) :

| variante non retenue | gen4 | gen5 ~50 | gen5 ~63 |
|---|---|---|---|
| **format archive sur le bus** (§1.5 l'interdit) | 96,2223 % | 94,0973 % | 92,5626 % |
| **miss recouverts** (§7 l'interdit) | 76,6430 % | 63,5049 % | 54,0161 % |
| tolérance **+5 %** au lieu de +10 % | 98,8322 % | — | — |
| tolérance **+20 %** au lieu de +10 % | 95,3286 % | — | — |

⚠️ **La ligne « +20 % » montre pourquoi la tolérance devait être figée au
§0bis** : son seuil VERT (95,3286 %) est **sous** la borne ROUGE à +10 %
(95,4016 %). Un hit à 95,35 % basculerait de rouge à vert en ne touchant qu'un
chiffre qui n'est pas un critère. **Seule la ligne 10 % porte verdict.**

⚠️ **Les 11,733 ms sont un PLANCHER**, donc un biais conservateur : tout débit
réel plus bas desserrerait le seuil. **Le seuil est figé sur ce plancher et NE
SE RECALCULE PAS sur un débit observé** — un débit mesuré ouvre une remesure
pré-enregistrée, pas une requalification.

### 4.1 Le seuil opposable — et **laquelle** des deux quantités il juge

🚨 **Le seuil hérité est à corriger, pas à recopier.** « hit (**statique ou
LRU**) < 96 % à α = 0,45 ⇒ le package A recule d'un cran »
(`passation-exec-2026-08-13.md:127-129`) confond deux quantités que seule la
mesure temporelle sépare, et l'une est **déjà décidée** : le hit d'oracle
statique se calcule aujourd'hui à 0 $ et vaut **89,8122 %** à α = 0,45,
**95,9833 %** à α = 0,5868 (§8.1). Un seuil dont on connaît le verdict avant la
mesure n'est pas un critère. **Seul le hit LRU peut porter un seuil** : inconnu,
il peut **battre comme rater** l'oracle statique (`ops/moe_ciseau.py:445-447`).

**Quantité jugée** : hit LRU, ordre décode, cache global, à **α_verdict** (§2.5),
transporté 120B (144 visites), **format servi sur le bus**, tolérance +10 %.

| hit LRU mesuré | verdict |
|---|---|
| ≥ **97,6643 %** | **vert** — tient au lien le plus pessimiste (gen4). *Miss sérialisés, aucun recouvrement.* |
| **[96,3505 ; 97,6643[** | **conditionné au lien** : tient en gen5 (~50 Go/s), pas en gen4. *Miss sérialisés, aucun recouvrement.* |
| **[95,4016 ; 96,3505[** | **conditionné au lien le plus rapide** (~63 Go/s) seulement. *Miss sérialisés, aucun recouvrement.* |
| < **95,4016 %** | **rouge** (§6). *Miss sérialisés, aucun recouvrement.* |

**Ce que ce seuil peut faire.** À α_verdict = 0,5868 l'oracle statique vaut
95,9833 % — dans la **troisième** bande : il faut que le LRU le batte de
**1,68 pp** pour être vert en gen4, et qu'il soit **pire de 0,58 pp** pour être
rouge. **Les deux directions sont atteignables**, ce qui n'était pas le cas à
α = 0,45 (7,85 pp à combler, verdict connu d'avance).

### 4.2 Ce dont le seuil dépend vraiment, et le fait SANS CHAÎNE

🕳️ **Le maillon qu'on croyait porteur est NEUTRE.** Le b/poids apparaît au
numérateur (masse active) **et** au dénominateur (taille de cellule sur le
bus) : il s'annule. Vérifié — recalculé avec `Planes14` (4,804 des deux côtés),
le seuil gen4 rend **97,664 %**, identique à trois décimales. `Golay70` déplace
la **CARTE**, donc α_verdict, **pas le SEUIL**.

**Le seuil ne dépend que de quatre quantités** : (a) le rapport 195 ÷ 32 Go/s,
deux estimés sur deux machines dont aucune n'est la cible ; (b) le rapport masse
active ÷ taille de cellule (5,1 Md ÷ 24 883 200), qui suppose
`hidden = moe_inter = 2880` au 120B ; (c) la **tolérance de 10 %** (§0bis b) ;
(d) le rapport format-froid ÷ format-chaud, figé à 1 par §1.5. **Aucune n'est
mesurée sur la cible.** S'y ajoute, pour le seul *transport*, le passage d'un
20b/32 experts à un 120b/128 experts, que la réserve de périmètre interdit
précisément pour un vert.

**Ce qui ne dépend d'aucune des quatre** — les deux quantités sortent du même
flux, dans le même run, sans aucune constante externe :

| quantité | définition | seuil |
|---|---|---|
| **gain de localité** | `hit_LRU(ordonné) − hit_oracle_statique`, à α identique | **< 0,5 pp aux HUIT α ⇒ la branche LRU est close définitivement** : pas de localité temporelle exploitable, le dump agrégé du 12 n'avait rien manqué. Entre 0,5 pp et le seuil du §4.1, la branche reste ouverte **au titre du point de courbe uniquement**, et sa réouverture exige une **idée neuve nommée**, pas une remesure |
| **part attribuable à l'ordre** | `hit_LRU(ordonné) − hit_LRU(mélangé)`, graine imprimée | reporté, non seuillé — il **sépare** ce qui vient du déséquilibre de fréquence (survit au mélange) de ce qui vient de l'ordre (détruit par le mélange) |

⚠️ **La barre de 0,5 pp est un CHOIX, posé ici et pas après** : la simulation
étant déterministe sur une trace fixe, ce ne peut pas être un σ
d'échantillonnage, c'est une barre de **pertinence**. **Ce bloc est le résultat
de P2 qui survivra même si les quatre quantités du §4.1 sont revues.**

### 4.3 Les critères de CHOIX DU MODÈLE — cinq, ordonnés, chacun peut échouer

**L'ordre d'exécution est C5 → C1 → C4 → C2 → C3**, et il est porteur : tant que
le chemin de chargement n'existe pas, C1 s'écrit **NON ÉVALUÉ** (jamais « passé »).

| | critère | forme opposable | ce qui l'échoue aujourd'hui |
|---|---|---|---|
| **C5** | architecture chargeable | le chemin de chargement et la géométrie de calibration existent, **ou** leur écriture est chiffrée **sous un plafond d'effort arbitré** | `llvq-llm` ne charge qu'**une** architecture, Qwen3 **dense** (`grep -rn "candle_transformers::models::" llvq-llm/` → `qwen3` seul : `loader.rs:71`, `model.rs:27`, `sealed.rs:31`, `fused_cuda.rs:1122`) ; `Block` a **sept matrices en dur** (`model.rs:559-566`, `:635-643`), calibration figée à **quatre hessiennes par bloc** (`model.rs:51`). 🚨 **Le plafond d'effort est ABSENT du dépôt** : tant qu'il n'est pas posé dans la note produit, **C5 s'écrit NON ÉVALUÉ et P6 ne s'ouvre pas** — même règle qu'au §0bis (c) |
| **C1** | référence non quantifiée | le checkpoint porte des poids bf16/f16 et le **contrôle identité** (`bin/oracle`, `max\|Δhidden\| = 0`) mesure encore ce qu'il mesurait sur les denses | **gpt-oss (MXFP4) et K2.6 (INT4) échouent** : *entraînés* quantifiés (`etude…:144-147`). Et le dump du 12 a été pris sur un gpt-oss **déquantifié en bf16** |
| **C4** | routage du modèle choisi | si le candidat ≠ gpt-oss-20b, un dump **du candidat** est un prérequis **chiffré** de P6 | aucun dump d'un autre modèle n'existe (`ls docs/data/`) |
| **C2** | devis à terme d'expert | le compte retombe à **≤ 2 %** sur le **checkpoint lui-même** : somme des formes lues dans les **en-têtes `.safetensors`** du dépôt HF, jamais une fiche web. Compte et référence imprimés **à l'unité** | 🚨 **`ops/run.py` échoue d'un ordre de grandeur** : `weight_counts` (`:233-262`) est **dense** — `grep -niE "moe\|expert" ops/run.py` → **aucune ligne**. Sur Qwen3-30B-A3B : **2,72 + 0,62 = 3,34 Md** contre 30,5 Md (`etude…:23`), **11 %** ; défaillance = devis **silencieusement faux**. ⚠️ Les fiches web sont à trois chiffres (`etude…:22-25`) : « 21 Md » couvre ±2,4 %, **plus que le seuil**. ⚠️ Sur un checkpoint natif quantifié, la conversion octets→paramètres s'écrit explicitement ou C2 est NON ÉVALUÉ |
| **C3** | politique « expert mort » | **IMPLÉMENTÉE derrière un drapeau**, **test létal existant**, **VERT** sur le code corrigé et **ROUGE par mutation** sur le code actuel — mutation journalisée (CLAUDE.md §5) | 🚨 Une politique est déjà **nommée** dans l'étude (`etude…:141-142`) : **la nommer ne suffit donc pas**, ce serait auto-satisfait. Elle est absente du **CODE** : `llvq-quant/src/linalg.rs:112-122` (`cholesky_lower` → `Err(NotPositiveDefinite)`) échoue franchement, **aucun repli**. Et **au moins une cellule est morte** (**L15/e20**, unique zéro des 768, mesuré) |

**Décision posée d'avance** : **aucun devis de P2 ni de P6 ne sort de
`ops/run.py` avant sa correction**, et le **~5,74 $ recalé ~10 $** du lot X
(`docs/archive/passation-lot-x-2026-08-12.md:177-192`, `docs/HISTORIQUE.md:240-243`)
est **retiré**, pas amendé : c'est un devis de 3,34 Md, pas de 30,5 Md. Jusqu'à
correction, le coût de P6 est **ABSENT**.

🚨 **C3 est un bloquant qui n'est pas un choix de modèle** : choisir un modèle
sans le trancher **déplacerait le blocage sans le lever**. La couverture
hessienne par expert est le « piège silencieux » de l'étude (`etude…:135-142`),
et **241 des 768 cellules (31,4 %)** sont sous le rang plein à 131 072 tokens.
⚠️ Majorant d'espoir : un expert peut voir 10 000 tokens colinéaires et rendre
une hessienne singulière quand même (`ops/moe_routing.py:23-27`).

## 5. La prédiction, et ce qui ne la fonde pas

**Prédiction, à α_verdict = 0,5868** : `hit_LRU ∈ [94 % ; 97 %]`, `gain de
localité ∈ ]0 ; +2 pp]`.

🚨 **La fourchette chevauche trois des quatre bandes, pas la première.** Si elle
se confirme, le §4.1 rend « conditionné au lien » et **n'apprend presque rien**
que le §8.1 ne disait déjà ; la seule connaissance neuve serait le §4.2. **Ce
qui rend le §4.1 utile, c'est le cas où la prédiction est FAUSSE** — un LRU
battant l'oracle statique de plus de 1,68 pp (vert gen4) ou passant dessous de
plus de 0,58 pp (rouge). **C'est un test de falsification de la prédiction, et
le journal le dira ainsi.**

**Ce qui la fonde — un seul élément, faible** : l'oracle statique vaut
**95,9833 %** à α_verdict (calculé sur le dump mesuré) ; il exploite **déjà**
tout le déséquilibre de fréquence, un LRU n'ajoutant que la composante
**temporelle** contre la capacité gaspillée sur les visites uniques — il peut
donc aussi **passer sous** l'oracle statique.

🚨 **Ce qui NE la fonde pas** : **aucune mesure de localité d'experts n'existe
dans ce dépôt**, sur aucun modèle, et ce document **ne cite aucune référence
externe** parce qu'il n'en a vérifié aucune. La prédiction ne s'appuie donc sur
**aucune mesure, ni interne ni externe** : c'est un jugement, et un jugement de
ce type a déjà été faux d'un facteur 2 ici (`Golay70` : estimé 1,9–2,4×, mesuré
1,77×).

⚠️ **Le mécanisme contre-intuitif, posé AVANT de lire le résultat** : **un Gini
ÉLEVÉ est FAVORABLE** (peu de cellules portent la masse). **Le poste de coût,
ce sont les couches PLATES** — à 0,29 miss/token la couche 0 (Gini 0,351) exige
**α = 1,000** quand la couche 23 (Gini 0,748) n'exige que **0,781** (mesuré,
[`moe-ciseau-2026-08-13.txt:86-90`](../docs/mesures/moe-ciseau-2026-08-13.txt)).
Un pré-enregistrement qui prédirait « plus de déséquilibre = pire » se
tromperait de signe.

**Et ce mécanisme porte un seuil, pas seulement une lecture** : le verdict
global n'est prononcé que si **aucune couche ne porte plus de 15 % de la masse
totale de miss** (24 couches ⇒ 4,17 % à l'uniforme ; 15 % = 3,6× l'uniforme).
Au-delà, **le verdict global est SUSPENDU** et le journal publie la variante par
couche à budget uniforme (§1.3) comme encadrement obligatoire. *(Le 15 % est un
choix posé ici ; valeur de l'oracle statique à α_verdict, §8.3 : 7,33 %.)*

**Si le hit LRU dépasse 99 %, chercher l'erreur avant d'en faire un titre** :
cache dimensionné en experts et non en cellules (32 au lieu de 768), flux
tronqué, miss obligatoires exclus, simulateur qui ne rafraîchit pas la récence
ou ignore l'ordre des K (V0.4, cas 4 et 5), trace ré-agrégée par mégarde.

## 6. Les issues, et ce que chacune fait au dossier

| issue mesurée | conséquence, décidée d'avance |
|---|---|
| hit LRU ≥ **97,6643 %** | **vert** : le barreau 32 Go tient au lien le plus pessimiste. **P6 s'ouvre** sur un modèle nommé — **sous réserve de C5 → C1 → C4 → C2 → C3**. ⚠️ **Le vert ne se transporte pas** : la capture du routage du **modèle cible** devient un prérequis chiffré de P6 |
| hit LRU ∈ **[96,3505 ; 97,6643[** | **P6 s'ouvre** (mêmes réserves), **MAIS** le package A est **reclassé « gen5 obligatoire »** dans `docs/note-produit-2026-08-13.md` **avant toute communication** — un package dont la carte dépend du bus n'est pas le même produit — et la capture du routage cible reste un prérequis chiffré |
| hit LRU ∈ **[95,4016 ; 96,3505[** | **P6 NE s'ouvre PAS sur ce dimensionnement** ; publié comme point de courbe. L'ouverture exige d'abord **soit** le dump du modèle cible, **soit** la classe de carte supérieure — ce second chemin **périme le « 32 Go »** de la fiche (`note-produit:128`) |
| hit LRU < **95,4016 %** | **rouge, et il se transporte** : la ligne « Chat/agent local sur MoE ~120B » de `docs/note-produit-2026-08-13.md:128` passe de « barreau A, 32 Go » à **hors barreau**, et le MoE ~120B **sort du périmètre produit** jusqu'à ce qu'une des deux pistes ait été **pré-enregistrée et mesurée** : (i) classe de carte supérieure — α_verdict **étant** le maximum de la carte (§2.5), tout α plus grand change de classe et périme le 32 Go ; (ii) **prefetch spéculatif** conçu et mesuré, pas une note de bas de page (dette (i) du §7 du ciseau : le préchargement suppose de connaître le routage AVANT la couche) |
| **gain de localité < 0,5 pp** aux huit α | branche LRU **close définitivement** : le dump agrégé suffisait, le verdict du 13 n'avait rien manqué. **C'est un résultat, pas un échec** |
| gain ≥ 0,5 pp mais §4.1 non vert | le LRU améliore l'oracle statique **sans suffire** : point de courbe publié avec le chiffre de l'amélioration — seule ligne produisant une connaissance neuve indépendante des quatre quantités du §4.2 |
| **une couche > 15 % de la masse de miss** | **verdict global SUSPENDU** (§5) ; encadrement par couche à budget uniforme publié avant toute conclusion |
| **A1 = Mac à mémoire unifiée** | **le test ne s'applique pas** (pas de bus PCIe, donc ni miss ni budget) : P2 se referme **sans verdict** sur l'axe LRU, seuls les §4.2 et §4.3 restant dus |
| **C5 non évaluable**, ou aucun candidat ne passe C5–C3 | **P6 ne s'ouvre pas**, quel que soit le hit ; le package A est bloqué sur un prérequis **d'ingénierie** (plafond d'effort absent, chemin de chargement MoE, politique expert mort) — le dire ainsi, jamais « en attente de mesure » |
| `ops/run.py` non corrigé au moment de conclure | **aucun devis de P6 n'est publiable** ; le coût de P6 s'écrit **ABSENT**, le ~10 $ du lot X reste retiré |

**Aucune de ces issues ne bloque P1, P3 ni P4.** **P6 dépend de P2** : c'est la
seule dépendance que ce document crée.

## 7. Ce qui invaliderait ce pré-enregistrement

- **fiche produit non commitée, ou cases A1 / A6 / génération PCIe non cochées
  avant le tampon** (§0bis a) : **le point de verdict n'existe pas**, aucune
  bande du §4.1 n'est opposable ;
- **échec de l'un des quatre contrôles V0** — égalité cellule par cellule sans
  les trois conditions de repli (V0.1), décision exécutée non lue sur gpt-oss
  (V0.2), couche incomplète (V0.3), l'un des cinq cas synthétiques (V0.4) — ou
  **trace absente** (le simulateur échoue, il ne saute pas) : **le run est nul,
  aucune courbe n'est publiable** ;
- **`moe_intermediate_size` repli silencieux sur `intermediate_size`**
  (`ops/moe_routing.py:199` — sur gpt-oss les deux valent 2880, donc
  **indiscernable**) : la géométrie de cellule (24 883 200 poids, 11,163 Mo) est
  fausse et **tout le budget du §4.0 avec**, à recalculer avant tout verdict ;
- **verdict lu sur une distribution agrégée par expert** (§1.1) : le chiffre
  décrit une distribution quatre fois trop plate ;
- **TOLÉRANCE NON CHIFFRÉE invoquée pour requalifier une issue** —
  « capacity-first », plafond d'effort non posé, ou toute autre : **la
  requalification est nulle** (§0bis c) ;
- **RECOUVREMENT, ou format archive sur le bus, invoqués après la mesure** : ni
  l'un ni l'autre ne requalifie **AUCUNE** issue. Ce sont des **architectures
  distinctes**, chacune exigeant son propre pré-enregistrement — le
  préchargement suppose de connaître le routage **AVANT** la couche, et cette
  connaissance n'existe pas (dette (i) du §7 du ciseau). Leurs seuils sont
  imprimés au §4.0 précisément pour qu'on ne puisse pas les découvrir après coup.

## 7bis. Écarts au protocole — journal, tenu à chaud

*(Chaque entorse s'écrit ici le jour où elle est commise, avec sa raison et son
coût — la règle du 08-10.)*

**Aucune entorse à ce jour.** Aucune ligne de hook modifiée, aucune trace
produite, aucun simulateur écrit.

## 8. Ce qui est connu à la signature — divulgation datée

Les seuils du §4 dérivent de la formule du §4.0, pas des tables ci-dessous.

**8.1 — Le hit d'ORACLE STATIQUE est déjà calculable, et le voici en entier.**
Calculé le 2026-08-14 sur le dump **mesuré** du 12, convention **ceil**, cache
global, 768 cellules classées par charge décroissante :

| α | cellules | hit statique | miss/tok 20b (96) | miss/tok 120b (144) |
|---|---|---|---|---|
| 0,2000 | 154/768 | 63,9145 % | 34,64 | 51,96 |
| 0,2733 | 210/768 | **74,0797 %** | 24,88 | 37,33 |
| 0,3500 | 269/768 | 82,3333 % | 16,96 | 25,44 |
| 0,4344 | 334/768 | 88,8342 % | 10,72 | 16,08 |
| 0,4384 | 337/768 | 89,0831 % | 10,48 | 15,72 |
| 0,4500 | 346/768 | 89,8122 % | 9,78 | 14,67 |
| 0,5000 | 384/768 | 92,5170 % | 7,18 | 10,78 |
| **0,5868** | **451/768** | **95,9833 %** | **3,86** | **5,78** |
| 0,7161 | 550/768 | 98,7085 % | 1,24 | 1,86 |
| 0,8500 | 653/768 | 99,8094 % | 0,18 | 0,27 |

Les lignes 0,2733 / 0,35 / 0,50 / 0,7161 / 0,85 **reproduisent au centième**
celles du 13 ([`moe-ciseau-2026-08-13.txt:218-226`](../docs/mesures/moe-ciseau-2026-08-13.txt)).
**Les cinq autres sont nouvelles**, dont celle du point de verdict — et c'est
elle qui rend le bras « statique » du seuil hérité déjà décidé (§4.1).

**8.2 — Le ciseau du 13 est REPRODUCTIBLE, vérifié** : `python3
ops/moe_ciseau.py` (stdlib nue) rend
[`moe-ciseau-2026-08-13.txt`](../docs/mesures/moe-ciseau-2026-08-13.txt) **au
caractère près** (seule la ligne `# date` diffère), donc α_VRAM 0,7161, les neuf
lignes de sweep, 74,08 %, 11,73 ms, 8,57 ms et les classes de carte sont
vérifiés **par exécution**.

**8.3 — Les autres faits connus à la signature** :

- **part de la masse de miss portée par la couche la plus coûteuse**, sous
  l'oracle statique à α_verdict = 0,5868 : **7,33 %** (couche 20 ; uniforme
  4,17 %) — calculé aujourd'hui, et c'est contre lui que la barre des 15 % du §5
  doit être lue ;
- **une cellule morte**, unique : **L15/e20** (mesuré,
  [`…-2026-08-12.txt:88-89`](../docs/mesures/moe-routing-gptoss20b-2026-08-12.txt)) ;
- **241 cellules sur 768 (31,4 %)** sous le rang plein à 131 072 tokens pour une
  hessienne de dimension 2880 (mesuré, `:93`) ; couvrir 90 % demanderait
  **1 572 209 tokens (×12)**, 99 % **21 769 744 (×166)** (`:94-96`) —
  ⚠️ extrapolations **linéaires** sur un quantile, jamais vérifiées ;
- **les budgets 0,70 et 0,29 miss/token n'ont AUCUNE source** ; le ciseau le
  déclare avant de s'en servir (`ops/moe_ciseau.py:90-93`). **Ce document ne
  s'en sert pas** : son budget est recalculé au §4.0 ;
- **le fichier de mesure du 12 contient DEUX verdicts contradictoires**, seul le
  **second** fait foi (`:78-83`) : le premier divisait par une cellule morte et
  annonçait « 2 880 000 000 000 000 tokens » ;
- **`--from-json` ne rejoue PAS le tableau de charge par couche**
  (`ops/moe_routing.py:179-186`) : les Gini par couche n'existent que sur le
  chemin *live*, donc les reproduire exige le re-run complet (§2.3) ;
- **`verdict()` n'examine que `sorted({hidden, moe_intermediate})`**
  (`ops/moe_routing.py:185`, `:258`) : sur gpt-oss les deux valent 2880, donc
  **une seule dimension** testée. Sur un modèle où `hidden ≠ moe_inter` — le cas
  général, celui des candidats de C1 — le dump rendra **deux colonnes**, `gate`
  et `up` (d_in = hidden) n'ayant pas la dimension de hessienne de `down`
  (d_in = moe_inter). **Le seuil de couverture devra dire LAQUELLE il juge** :
  infixable avant de connaître le modèle, donc inscrit comme **livrable de C1** ;
- **aucun hit LRU, aucune milliseconde PCIe, aucun chiffre de qualité 2 bits sur
  MoE n'existe** — ni ici, ni, à notre connaissance, ailleurs.

**8.4 — Ce qui n'est PAS rouvert** : le verdict VRAM du 13 (mélange > 3,20 ⇒
variante VRAM enterrée) reste acquis ; `Golay70` reste **non servi** malgré son
usage comme `B_HOT` (§1.5) ; la clause de profondeur ≤ 24 du spec X4 n'est pas
de ce chantier. ⚠️ Pour solde de tout compte, l'approximation du mélange —
`mix(α) = α·3,589 + (1−α)·2,219` appliqué aux 117 Md entiers alors que les
**2,34 Md hors experts** (2,0 %, `ops/moe_ciseau.py:277`) n'ont aucun layout
assigné — vaut une **amplitude de 0,027 b/poids** (recalculé au budget 0,70 :
3,2743 en chaud, 3,2469 en froid), soit **±0,014 autour du milieu**, 40 % de la
marge au seuil de 3,20 : elle ne retourne pas le verdict du 13, et **aucun
chiffre de ce document n'en dépend** (la VRAM de P2 ne porte que le tier chaud).
