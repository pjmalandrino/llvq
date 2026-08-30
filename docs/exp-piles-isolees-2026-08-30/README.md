# Piles isolées — un noyau par machine, deux backends (2026-08-30)

> **État : PRÉPARÉ, NON LANCÉ.** Rien dans ce dossier n'a consommé une minute de
> carte. Aucun job ne part sans go explicite de l'opérateur et sans le tampon du
> pré-enregistrement (§7 du `CLAUDE.md`).

## 1. La question

Le dossier compare LLVQ au FP16 et à l'AWQ 4 bits. Il n'a **jamais mesuré la
qualité d'un concurrent 2 bits** : le 17,04 de QTIP est une citation de la
Table 6 du papier, pas une mesure de notre harnais — et F2 le dit lui-même
(payload pseudo-aléatoire, « aucune phrase de qualité ne peut s'appuyer sur ce
bras »). Tous nos verdicts qualité à 2 bits se comparent à un chiffre emprunté.

Cette expérience y répond dans la forme voulue par l'opérateur : **une machine
par bras, chacune avec son moteur natif et son propre noyau, les mêmes données,
puis on éteint** — sur **deux backends**, **Metal** sur la machine de l'opérateur
et **CUDA** sur HF Jobs.

## 2. Ce que le design permet et ce qu'il interdit

Des machines séparées ne coexistent jamais. La règle du 08-17 s'applique
intégralement, et elle est *mesurée* : le témoin f16 de vLLM rend **83,09 tok/s**
là où le nôtre rend **43,6** sur la même carte — l'écart bout-en-bout est dominé
par le moteur, pas par le décodeur de poids.

🚨 **Et un backend est une pile au même titre qu'un moteur.** Le dossier a mesuré
que les × ne se divisent même pas entre **deux cartes CUDA** : sur A100 aucun
bras à décodage ne bat FP16 (`Planes14` 0,79× contre 2,14× sur L40S), et le ×1,78
**est** le rapport d'horloges. Entre CUDA et Metal, l'interdiction est
*a fortiori*.

| grandeur | comparable entre machines ? |
|---|---|
| **mémoire**, b/param modèle entier | ✅ **directement** — compte d'octets, aucun moteur dedans |
| **MMLU** micro | ✅ **directement** — ne dépend que du tokenizer et de 4 logprobs |
| **perplexité** | ⚠️ **en rapport au témoin f16 de sa propre machine** |
| **tok/s** | ⚠️ **en rapport au témoin f16 de sa propre machine** |

**Le correctif est structurel** : chaque machine fait tourner **aussi son propre
témoin f16**, sur les mêmes données. Chaque machine ne rend donc pas un nombre
mais un **rapport**. C'est déjà la forme d'`ops/awq_speed.py`.

## 3. Les fichiers

| fichier | contenu |
|---|---|
| [`PROTOCOLE.md`](PROTOCOLE.md) | ce qu'on mesure et **ce que chaque nombre a le droit de conclure** |
| [`MACHINES.md`](MACHINES.md) | une fiche par machine, par backend |

## 4. Quel bras tourne où

| bras | CUDA (HF, payant) | Metal (Mac, 0 $) |
|---|---|---|
| **LLVQ 2 bits** | ✅ **servi** — `fusedrun` | ⚠️ **micro-banc + qualité** — pas de `fused_metal.rs` |
| **AWQ 4 bits** | ✅ servi — vLLM | ❌ aucun moteur |
| **GPTQ 2 bits** | ⚠️ artefact à produire | ❌ aucun moteur |
| **`IQ2_XXS`** | ⚠️ servi — llama.cpp CUDA | ⚠️ **servi — llama.cpp Metal** |
| **MLX 2 bits** | ❌ n'existe pas hors Apple | ⚠️ **servi — natif**, `q_bits=2` vérifié |
| ~~QTIP~~ | ❌ pas d'artefact Qwen3 | ❌ |

🔎 **`IQ2_XXS` est le seul bras qui traverse les deux backends** — c'est le pont
de l'expérience.

🚨 **QTIP est écarté, sur un fait vérifié le 2026-08-30** : relaxml ne publie que
du Llama. Le porter coûterait *estimé* 10–20 $ plus un risque d'incompatibilité
d'architecture. Sa **vitesse** est déjà mesurée par F2, dans la forme la plus
forte qui soit (un seul processus, bras entrelacés, division licite : 2,27×
[2,27–2,28]). Ce qui lui manque reste sa **qualité**.

## 5. L'ordre, et il fait tomber le coût

**Tout ce qui peut être tranché sur Metal l'est avant de louer une carte.** C'est
ce que le projet a fait historiquement — « un banc gratuit vaut mieux qu'une
carte louée pour trouver les pièges de mesure ».

| phase | où | coût |
|---|---|---|
| **1** — C5 : confronter le témoin FP16 Metal à MPS / MLX / Accelerate | Mac | **0 $** |
| **2** — produire `IQ2_XXS` (imatrix + quantize) et **MLX 2 bits** | Mac | **0 $** |
| **3** — qualité LLVQ / MLX / `IQ2_XXS` sur Metal, + contrôle inter-backend C4 | Mac | **0 $** |
| **4** — produire l'artefact GPTQ 2 bits | HF | ~0,5–1,0 $ |
| **5** — machines CUDA : GPTQ (M3) et `IQ2_XXS` (M4) | HF | ~0,6 $ |
| — | LLVQ (M1) et AWQ (M2) : **déjà mesurés** | 0 $ |
| | **total** | **~1,1–1,6 $** |

Provenance : *estimé* par analogie avec le registre — `awq-vllm-4b` 0,11 $ pour
5 rounds, `campagne-8b-qualite` 1,01 $ pour ppl+MMLU à trois bras.

🚨 **La phase 1 est un préalable, pas une option.** Tant que C5 est rouge, aucun
× Metal ne se publie : `docs/fiche-4b.md:438` le dit depuis longtemps — *« le
2,07× est un rapport contre un noyau écrit par le même auteur. C'est l'angle
hostile restant, non adressé. »* Sur CUDA, F1 a réglé l'équivalent (1,024× et
1,015× de cuBLAS). Côté Metal, rien.

## 6. Ce qui reste à trancher avant de lancer

1. **Fixture MMLU pour llama.cpp** — son `--multiple-choice` attend un autre
   format. Soit on la bâtit aux 2 280 questions avec le même `qhash`, soit
   `IQ2_XXS` porte **ppl seule** au premier tour.
2. **Périmètre** — combien de bras, et MLX entre-t-il ou non.
3. **Le tampon** — le pré-enregistrement n'est pas écrit ; il ne le sera qu'une
   fois 1 et 2 tranchés, et sera horodaté **avant la première milliseconde**.

## 7. Deux dettes à solder pendant qu'on est sur le Mac — quasi gratuites

`docs/fiche-4b.md` marque **SUSPECT** le « 129,8 tok/s » et le « 2,39 Go » du q4
MLX : *aucune trace — ni log, ni script, ni historique shell*. Son §563 les
répare en **~2 min**. L'artefact q4 est déjà local (`~/qwen3-4b-mlx-q4`, 2,1 Go)
et `mlx_lm 0.24.0` est installé.
