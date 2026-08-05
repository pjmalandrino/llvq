# `ops/` — faire tourner la quantification sur Hugging Face Jobs

Le poste de dev est un Mac à **69 Go** de mémoire unifiée. Qwen3-32B pèse
**65,5 Go** en bf16 : le modèle seul remplit la machine, et le pic de
factorisation demande ~26 Go de plus. Le 32B ne tournera pas ici, quelle que
soit la qualité du code.

Ce dossier est la sortie déportée. Il est en Python parce que c'est l'API
native de HF Jobs, et **hors du workspace Rust** parce que les crates sont
volontairement auditables et sans dépendances — l'orchestration ne l'est pas.

## Ce que la mesure a changé

`CLAUDE.md` dit « Cholesky dominant ». **C'était vrai avant `faer`, ça ne l'est
plus.** Mesuré avec `bin/cholbench` sur M3 Max :

| n | G mult-add | s | G mult-add/s |
|---|---|---|---|
| 1024 | 0,72 | 0,02 | 33 |
| 2048 | 5,73 | 0,09 | 62 |
| 3072 | 19,33 | 0,22 | 87 |
| 4096 | 45,81 | 0,43 | 106 |

Décomposition du run qui a produit le 16,94 (Qwen3-4B, 14 447 s) :

| | temps | part |
|---|---|---|
| encodage Leech | ~8 600 s | **59 %** |
| passes avant, conversions f64, écriture | ~5 600 s | 39 % |
| **Cholesky** | **223 s** | **1,5 %** |

Le passage de 6,3 h à 3,45 h attribué à `faer` s'explique exactement par là.

**Conséquence : le run est massivement CPU, et le GPU ne fait que ~10 minutes
de travail réel** (2 passes avant sur 131 k tokens). Le seul chiffre qui décide
devient le **coût du cœur-heure**, et un H200 à 5 $/h est le pire de la liste.

## Estimer avant de lancer

```bash
uv run ops/run.py estimate Qwen/Qwen3-32B
uv run ops/run.py selftest        # confronte l'estimateur au run 4B réel
```

`selftest` n'est pas décoratif : il exige que le compte de poids retombe
**exactement** sur les 3 633 315 840 du run publié, et que l'encodage explique
40 à 80 % des 14 447 s mesurées. Un estimateur que personne n'a confronté à un
run réel est un tableur.

Sortie pour le 32B :

```
  poids quantifiés     31.21 Md
  poids portés 16 b     1.56 Md   (4.7 % des poids, 27 % de l'artefact)
  checkpoint bf16       65.5 Go
  artefact projeté      11.6 Go   ×5.7
  encodage Leech       245.9 cœur-h
  Cholesky               1.9 cœur-h   (0.8 %)

  cpu-performance          7.7    14.71
  rtx-pro-6000            10.8    29.62
  h200                    10.8    53.86
```

> ⚠️ **`tie_word_embeddings: false` sur le 32B.** Contrairement au 4B,
> `embed_tokens` et `lm_head` sont deux tenseurs. Ils font 4,7 % des poids mais
> **27 % de l'artefact**, et le ratio tombe à ×5,7 au lieu du ×7,4 nominal.
> C'est le piège du « x bits/poids » de `CLAUDE.md` §3, qu'on croyait réservé
> aux petits modèles.

## L'arbitrage non tranché

`cpu-performance` et `rtx-pro-6000` sortent à peu près au même prix, pour des
raisons opposées : le CPU est 2× moins cher au cœur-heure, mais doit faire les
passes avant à la main. Sur GPU c'est ~10 min ; sur CPU, plusieurs heures, et
mon incertitude sur le débit `gemm` f32 de candle sur 32 vCPU est d'un facteur
3. **L'estimateur ne modélise donc pas les passes avant** — habiller une
supposition en estimation serait pire que de l'omettre.

C'est l'étape 8B qui tranche, pour ~20 $ sur les deux flavors.

## Progression

| étape | flavor | coût | statut |
|---|---|---|---|
| 0. `oracle` sur CUDA | `l4x1` | 0,01 $ | ✅ `max \|Δhidden\| = 0.000e0` |
| 1. 0,6B, 3 blocs, CPU puis CUDA | `cpu-upgrade`, `l4x1` | 0,11 $ | ✅ chaîne validée, `verify_artifact` OK |
| 2. **8B complet** | `rtx-pro-6000` | **11,48 $** | ✅ **×1,267 à 2,0436 b/poids** |
| 3. 32B, 4 blocs (dé-risquage) | `rtx-pro-6000x2` | 5,43 $ | ✅ mémoire et bf16 OK, **621 s/bloc** |
| 4. **32B complet** | `rtx-pro-6000x2` | **~62 $ / ~11,4 h** | ⏸ en attente |

**Le dé-risquage a payé.** Il annonçait 9 h / 49 $ par extrapolation depuis le
8B ; la mesure à `d_in = 25600` donne **621 s/bloc contre ~500 prédits**, donc
11,4 h et ~62 $. 5,43 $ pour corriger une erreur de 13 $ avant engagement.

**Pourquoi l'extrapolation a raté** : le coût par poids n'est pas indépendant
de la largeur. La factorisation en `n³` passe de 1,6 % d'un run (0,6B) à 5,5 %
(8B) à **16,5 %** (32B). L'estimateur utilise désormais la plus grande des
constantes mesurées, pour se tromper par excès plutôt que par défaut.

Ne pas sauter à l'étape 3 : lancer un run sur un chemin CUDA jamais exécuté,
c'est le genre de job qui meurt à la 10ᵉ heure sur une divergence de backend
que `bin/oracle` aurait attrapée pour 7 $.

> 🔎 **L'étape 2 valide le pipeline, elle ne produit pas un chiffre publiable.**
> Qwen3-8B a lui aussi `tie_word_embeddings: false`, mais avec un `hidden` de
> 4096 seulement : l'embedding y pèse **15,2 % des poids et 57 % de l'artefact**,
> et le ratio tombe à **×3,7**. Le 8B est donc un mauvais vitrine — moins bon
> que le 4B (×4,63) *et* que le 32B (×5,7) — alors que la méthode est
> identique. Ne jamais le sortir comme résultat de compression.

**Prérequis compte** : les Jobs exigent un solde de crédits prépayés positif.
Sans ça, `launch` remonte un `402 Pre-paid credit balance is insufficient`
après avoir passé la garde de coût.

## Ce qui manque encore côté Rust

Le squelette est complet ; le binaire qu'il lance ne l'est pas.

| item | ce que ça bloque | statut |
|---|---|---|
| **C1** feature `cuda` | `--device cuda`. `llvq-llm/Cargo.toml:29` déclare `cuda` (et `cudnn` ligne 30), `eval.rs:52` route vers `Device::new_cuda`, `ops/Dockerfile.cuda` la construit | ✅ |
| **C2** `bin/oracle` sur CUDA | la preuve **a été refaite sur ce backend** : l'étape 0 ci-dessus (`l4x1`, 0,01 $) rend `max \|Δhidden\| = 0.000e0`, comme en Metal. Reste à rejouer à chaque changement de backend — c'est ce que `cmd_oracle` (`ops/run.py:505`) existe pour faire | ✅ |
| **C4** reprise sur checkpoint | les runs longs. Le timeout par défaut d'un Job est **30 min** ; la durée max n'est pas documentée | ❌ |
| **C5** chemin local pour `LLVQ_MODEL` | le montage `Volume(type="model")`. Sans lui le conteneur retélécharge 65 Go à chaque run | ❌ |
| **C6** mémoire du quantifieur | 12,4 Go de facteurs coexistent à 32B, dont 6,2 jamais lus quand `group_scales` est off | ❌ |
| C3 chargement bf16 | *devenu optionnel* : `cpu-performance` a 256 Go de RAM, le modèle tient en f32 | — |

**C4 est le plus structurant.** La calibration est séquentielle — le bloc *t*
est quantifié contre les activations qui ont traversé les blocs 0..*t*−1 **déjà
quantifiés** — donc reprendre veut dire : recharger le checkpoint de base,
ré-appliquer les matrices déjà écrites dans l'artefact (`decode_matrix` les
rend bit pour bit), et repartir au bloc *k*. C'est du design, pas un drapeau.

## Construire l'image

Depuis la **racine du dépôt**, pas depuis `ops/` :

```bash
docker build -f ops/Dockerfile -t <user>/llvq:cpu .
```

```bash
docker push <user>/llvq:cpu
```

La variante CUDA a sa propre recette, `ops/Dockerfile.cuda`. Un Space construit
son Dockerfile **tel quel** : `--build-arg` n'atteint jamais le builder du Hub,
donc le choix CPU/CUDA se fait par *quel fichier* on téléverse, et c'est
`publish --cuda` qui le fait (`cmd_publish` dans `ops/run.py`) :

```bash
uv run ops/run.py publish <user>/llvq-runner-cuda --cuda
```

En local, la même image se construit directement, toujours depuis la racine :

```bash
docker build -f ops/Dockerfile.cuda -t <user>/llvq:cuda .
```

La compute capability y est figée à `89` (Ada) : le builder d'un Space n'a pas
de GPU, donc la détection par `nvidia-smi` de `candle` échoue et le build meurt
sans elle. Changer la ligne et rebâtir pour viser un H200.

> `Cargo.lock` est suivi par git — `.gitignore` ne fait que deux lignes,
> `target/` et `__pycache__/` — et les deux Dockerfiles bâtissent avec
> `--locked`. L'image est donc construite depuis l'arbre commité, et un chiffre
> qu'elle produit est rattachable à un commit.

## Lancer

```bash
uv run ops/run.py launch --model Qwen/Qwen3-8B --flavor cpu-performance --image <user>/llvq:cpu --bucket <org>/llvq-runs --name qwen3-8b-c12L3
```

`launch` **refuse** au-dessus de `--max-usd` (60 $ par défaut) sans `--yes`, et
pose un `timeout` calculé à 1,5× l'estimation — le défaut de 30 minutes tuerait
tous les runs réels.

Deux volumes règlent le téléchargement et la sortie :

- `Volume(type="model", …, read_only=True)` monte le repo du checkpoint, donc
  65 Go ne sont pas retéléchargés à chaque relance (dépend de **C5**) ;
- `Volume(type="bucket", …, read_only=False)` reçoit l'artefact — et recevra
  les checkpoints de reprise de **C4**.

```bash
uv run ops/run.py watch <job_id>
```

## Sur le découpage entre machines

`cpu-upgrade` ressort à **0,93 $** pour les 246 cœur-heures du 32B, contre
14,71 $ sur `cpu-performance` — 16× moins cher au cœur-heure. C'est le plus gros
levier théorique de la liste, et il est probablement inexploitable.

Les **lignes** d'une matrice sont indépendantes (`parallel_matches_serial_exactly`
l'exige au bit près), donc une matrice peut s'éclater entre workers. Mais chaque
worker a besoin du facteur `U`, qui fait **5,24 Go** pour `down_proj` : le
distribuer 64 fois coûte plus que ce qu'il économise sur un run unique.

Les **blocs**, eux, ne sont pas parallélisables du tout — c'est la calibration
séquentielle. Les découpler voudrait dire calculer toutes les hessiennes sur le
modèle FP, ce que fait QuIP# (qui atteint quand même 17,04). C'est un arbitrage
de méthode, pas d'infra, et il ne se prend pas pour économiser 14 $.

**Le bon découpage est temporel, pas spatial** : C4, plus la facturation à la
minute, permet de tuer un run qui dérape sans rien perdre.
