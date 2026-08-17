# Pré-enregistrement — le bras AWQ 4 bits chronométré dans vLLM

> Écrit et commité **avant le lancement**, et avant qu'aucune milliseconde vLLM
> n'existe sur ce projet. Statut du tampon : comme celui de `fusedrun` 14B écrit
> le même jour, ce document **ne sera pas horodaté par OpenTimestamps**, et il
> faut dire pourquoi. Ce job **ne décide de rien** : il n'écarte aucun bras,
> n'adopte aucun format, ne franchit aucun seuil. Il remplit une case vide d'un
> tableau publié — et il la remplit dans **une autre pile logicielle** que
> toutes nos mesures. Ce qui doit être figé d'avance n'est donc pas un critère
> mais **la forme du rapport et la liste des phrases interdites**, qui sinon
> s'écriraient en voyant les nombres.
>
> ⚠️ L'antériorité repose sur la date de commit. Si l'opérateur veut le tampon,
> `ots stamp` avant le lancement, comme pour P1 et P1b.

---

## §1 — Divulgation datée : tout ce qui est connu à la signature

### 1.1 Ce que le projet a déjà mesuré, et qui pourrait « inspirer » la forme du rapport

| grandeur | valeur | provenance |
|---|---|---|
| 4B, bras dense f16 (le nôtre) | **43,5** tok/s · 8,04 Go | *mesuré*, [`mesures/campagne-finale-bras4-2026-08-07.txt`](../docs/mesures/campagne-finale-bras4-2026-08-07.txt) l. 112 |
| 4B, bras dense f16, autre invocation | **43,6** tok/s | *mesuré*, [`mesures/planes14-fusedrun-2026-08-06.txt`](../docs/mesures/planes14-fusedrun-2026-08-06.txt) |
| 4B, servi `Planes14` + `LLVQ_EMBED=q8` | **88,4** tok/s · 2,60 Go | *mesuré*, même journal l. 111 |
| 4B, fusé à **tête identique** (embedding f16) | **48,7** tok/s → **×1,12** | *mesuré*, `planes14-fusedrun` |
| 8B, dense f16 | **26,5** / **26,6** tok/s · 16,38 Go | *mesuré*, [`campagne-8b-vitesse`](../docs/mesures/campagne-8b-vitesse-2026-08-08.txt) / [`campagne-8b-q8`](../docs/mesures/campagne-8b-q8-2026-08-08.txt) |
| 8B, fusé à **tête identique** (embedding f16) | **34,4** tok/s → **×1,30** | *mesuré*, `campagne-8b-vitesse` |
| 8B, servi (têtes q8) | **69,3** tok/s → ×2,61 | *mesuré*, `campagne-8b-q8` |
| 14B, **tous bras** | **aucun tok/s** | un job est pré-enregistré le même jour ([`preregistration-fusedrun14b-2026-08-17.md`](preregistration-fusedrun14b-2026-08-17.md)) |

🚨 **Aucun tok/s vLLM n'existe sur ce projet, sur aucun matériel, pour aucun
modèle.** Vérifié le 2026-08-17 : `grep -rli vllm` sur l'arbre ne rend que six
fichiers de **prose** (`README.md`, `LAUNCH_ME.md`, `docs/hf-model-card.md`,
`docs/fiche-4b.md`, `docs/inference-cost-reduction-2026.md`,
`docs/archive/kernel-comparison-recon.md`) — pas une ligne de code, pas un
journal de mesure. La pile n'a jamais été installée, et la cellule
`speed_tokps` de la ligne `awq` de [`docs/data/campagne-finale.csv`](../docs/data/campagne-finale.csv)
est vide depuis le premier jour, tout comme le `---$^{\dagger}$` de
`paper/sections/evaluation.tex:29` et `:108`.

### 1.2 🚨 Le résultat déjà mesuré qui est CONTRE nous, et qui est publié

Banc à six bras du **2026-08-10**, un processus, rounds entrelacés, vérification
f64 ligne à ligne des six bras avant tout chronométrage
([`mesures/six-arm-awq-2026-08-10.txt`](../docs/mesures/six-arm-awq-2026-08-10.txt),
0,78 $) :

| bras | b/poids noyau | Go/s | borne d'octets | fraction atteinte |
|---|---|---|---|---|
| `Planes14` (notre production) | 4,804 | **428** | 3,33× | **65 %** |
| **AWQ w4g128** (`mit-han-lab/llm-awq`, porté chez nous) | 4,179 | **584** | 3,83× | **88 %** |

**Un noyau 4 bits déployé atteint 584 Go/s là où notre meilleur layout plafonne
à 428**, et il convertit 88 % de son avantage d'octets en vitesse contre 65 %
pour nous — alors même qu'il **relit l'activation quatre fois plus** (7,27 Go
contre 1,51). Ce n'est pas un artefact de comptabilité ; le journal l'établit.
Ce fait est connu, publié, et il rend **prévisible** que le bras vLLM sorte
devant nous. Il est écrit ici pour qu'on ne puisse pas dire ensuite que le
protocole a été taillé après l'avoir vu.

### 1.3 Le reste du contexte connu

- **Le plancher `nullk` (2026-08-16)** : une passe de projections qui ne lit
  aucun poids coûte **45,2 %** du bras servi, donc tout travail de **format**
  plafonne à **4,77× FP16**, `Planes14` en est à 2,16×, et son décodage ne pèse
  que **~7 %** du temps de trafic
  ([`mesures/nullk-plancher-2026-08-16.txt`](../docs/mesures/nullk-plancher-2026-08-16.txt)).
- **L'attribution du 2026-08-05** : les 2,04 ms/token se découpent en
  latence/occupation **39 %**, flux **33 %**, décodage **19 %**. ⚠️ Dénominateur
  différent de celui du plancher — ne pas les additionner.
- **Le biais connu de notre bras dense f16**, et sa direction : `Head::project`
  (`llvq-llm/src/model.rs:580-582`) appelle `Tensor::broadcast_matmul`, dont le
  bras à rhs de rang 2 **matérialise le poids transposé à chaque appel** — 778 Mo
  de vocabulaire recopiés par token au 4B, ~26 ms/token
  ([`mesures/phases-2026-08-07.txt`](../docs/mesures/phases-2026-08-07.txt)),
  remonté amont en [huggingface/candle#3871](https://github.com/huggingface/candle/issues/3871).
  **Le sens compte** : ce défaut est dans le **dénominateur** de nos rapports,
  donc il les **gonfle**. Les rapports cités ici (×1,12 / ×1,30) sont ceux à
  tête identique, où le défaut porte **des deux côtés** — c'est pourquoi ce sont
  eux qui se citent et pas ×2,03 / ×2,61.

### 1.4 Ce qui a été vérifié dans le dépôt et sur le Hub, le 2026-08-17, avant d'écrire

- **Le protocole maison, relu au source** (`llvq-llm/src/bin/fusedrun.rs`) :
  `PROMPT = "The capital of France is"` (l. 43), encodé **sans token spécial**
  (`encode(PROMPT, false)`, l. 119 et 177), **une** génération jetée puis **une
  seule** chronométrée par bras (l. 138-141 et 183-186),
  `rate = n_new / elapsed` — donc **prefill compris**, `dtype = F16` figé
  (l. 97), argmax glouton, VRAM **comptée** (`runtime_bytes + carried_bytes`)
  et non lue à la carte.
- **Le tokenizer**, vérifié indépendamment ce jour avec `tokenizers` :
  `"The capital of France is"` → **`[785, 6722, 315, 9625, 374]`, 5 tokens**,
  **identiquement** sur `Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-AWQ`,
  `Qwen/Qwen3-8B-AWQ` et `Qwen/Qwen3-14B-AWQ`, et le `tokenizer.json` des quatre
  a le **même sha256 `aeb13307a71acd8f…`** — qui est exactement la constante
  `tokenizer_sha256` épinglée dans `ops/awq_dequant.py:159` et scellée dans
  `~/qwen3-4b-llvq.bin`. L'empreinte de tokens est donc commune **par
  construction**, pas par chance.
- **Les trois dépôts AWQ**, `config.json` lu au Hub : `quant_method: awq`,
  `version: gemm`, `bits: 4`, `group_size: 128`, `zero_point: true`,
  `torch_dtype: float16` — les trois. ⚠️ Le 8B porte en plus
  `backend: autoawq`, absent des deux autres ; sans effet connu sur le routage
  vLLM, mentionné parce que c'est une différence réelle entre les dépôts.
- **Les bases sont en `bfloat16`** (`Qwen/Qwen3-4B`, `-8B`, `-14B`) — c'est ce
  qui rend `--dtype float16` obligatoire et non cosmétique (§2.3).
- **Les révisions** : les SHA `main` du 4B et du 14B relevés ce jour sont
  **identiques** aux révisions épinglées dans `ops/awq_dequant.py` — 4B AWQ
  `74d4bd2b…`, 4B base `1cfa9a72…`, 14B AWQ `31c69efc…`, 14B base `40c06982…`.
  🚨 **Le 8B n'a aucune entrée `EXPECTED`** : ses SHA (`4da05a8e…` /
  `b968826d…`) sont **relevés au Hub le 2026-08-17 et n'ont jamais été passés
  par `awq_dequant.py check`**. Le script les refuse par défaut (§2.5).
- **Le lanceur** : `ops/run.py::cmd_bench` (l. 943-1038) accepte `--image`
  arbitraire, exécute les lignes sous `set -euo pipefail`, monte un bucket en
  écriture sur `--out-mount`, et n'autorise que `l40sx1` sans `--any-flavor`
  (`BENCH_FLAVORS`, l. 940). 🕳️ Sa docstring affirme que le `--timeout` « est
  obligatoire et n'a pas de défaut silencieux » alors que l'argparse lui donne
  `default="30m"` (l. 1473). Le timeout sera donc **passé explicitement**, et
  cet écart docstring/code est signalé à l'opérateur.

### 1.5 Le budget

Plafond de lot accordé : **10 $**, dont **0,24 $** déjà dépensés (chiffre donné
par l'opérateur — ⚠️ [`docs/data/jobs.csv`](../docs/data/jobs.csv) s'arrête au
2026-08-16 et ne porte encore **aucune** ligne du 08-17 : le journal des coûts a
une dette d'une journée, à solder avant de croire un cumul lu dans ce fichier).
Ce job vise le **4B seul** : plafond réel **1,35 $** (§2.6).

---

## §2 — Le protocole, figé maintenant

### 2.1 Ce qui est répliqué du protocole maison, à l'identique

| élément | valeur | pourquoi c'est celle-là |
|---|---|---|
| prompt | `prompt_token_ids = [785, 6722, 315, 9625, 374]` | passés **directement**, jamais re-tokenisés : c'est la seule façon de garantir l'iso-prompt sans dépendre du chat template de vLLM |
| tokens générés | **128**, exactement | `fusedrun` en génère 128 dans tous les runs publiés |
| échantillonnage | greedy pur : `temperature=0.0, top_p=1.0, top_k=-1, n=1` | `fusedrun` fait un `argmax` sur logits castés en f32 |
| dtype | **`float16`** des deux bras | `fusedrun` fige `DType::F16` |
| chronométrage | mur autour du seul `generate`, **prefill compris**, `rate = 128 / elapsed` | identique à `fusedrun` |

### 2.2 Ce qui en diffère délibérément, et qui doit être déclaré

| élément | maison | ici | raison |
|---|---|---|---|
| rounds chronométrés | **1** (après 1 jeté) | **5** (après **2** jetés) | la carte rampe en horloge et vLLM capture ses graphes CUDA au premier appel ; deux jetés sont le minimum pour que le troisième soit en régime |
| forme du rapport | quotient de deux points | **médiane de 5 rapports formés round par round**, avec plage min–max | règle de maison n°2 — voir §3 |
| VRAM | comptée par nous | **rien** | §4 (iii) : aucun chiffre de VRAM ne sort de vLLM |

🚨 **Cette asymétrie de rounds est voulue et ne se rattrape pas après coup.**
Elle rend le bras vLLM **mieux** mesuré que le nôtre. C'est exactement pourquoi
le §3 interdit de former un quotient entre les deux piles : on comparerait une
médiane de cinq à un point unique, sur deux moteurs différents.

### 2.3 Les quatre pièges, neutralisés explicitement

1. **`dtype="float16"`.** Les checkpoints de base sont `bfloat16` (vérifié §1.4)
   et vLLM suit `torch_dtype` par défaut. Nos deux bras sont f16. Sans ce
   drapeau, le témoin f16 de vLLM serait en réalité un témoin **bf16**, et
   l'écart mesuré porterait un changement de dtype non déclaré. *(Les trois
   dépôts AWQ sont déjà `float16` : le drapeau ne les change pas, il les
   confirme.)*
2. **`enable_prefix_caching=False`.** Le cache de préfixe est **actif par
   défaut** en V1. Avec lui, les rounds 2..n **sautent le prefill** des 5 tokens
   de prompt : le round 1 mesure prefill + 128 décodes et les suivants 128
   décodes seuls. La plage min–max cesserait alors de mesurer la dispersion de
   la carte pour mesurer un **artefact de cache**, et la médiane porterait un
   protocole différent de celui de `fusedrun`, qui refait son prefill à chaque
   génération. Le script **relit** `cache_config.enable_prefix_caching` après
   l'initialisation et échoue s'il est vrai.
3. **`ignore_eos=True, min_tokens=128, max_tokens=128`.** Notre `generate` ne
   connaît pas l'EOS : il produit 128 tokens quoi qu'il arrive. Sans ces trois
   champs, vLLM s'arrête à l'EOS et le script diviserait **128** par le temps de
   **moins de 128 tokens** — une surestimation silencieuse et non bornée. Le
   script **compte les tokens rendus** et échoue si `len(token_ids) != 128`.
4. **`gpu_memory_utilization` fixé explicitement.** Le défaut de 0,9 préalloue
   ~43 Go sur une L40S de 48 : deux bras ne tiendraient pas dans le même
   processus, et **toute lecture de VRAM deviendrait ininterprétable** (on
   lirait la réservation, pas l'occupation). Valeurs posées d'avance :
   **0,30 pour le bras f16, 0,13 par bras quantifié**. Elles sont un **budget de
   place**, pas une mesure — et c'est une raison de plus pour laquelle §4 (iii)
   interdit d'en tirer un chiffre de mémoire.

### 2.4 🚨 Le piège de routage : `awq` contre `awq_marlin`

Sur Ada (sm_89), vLLM détecte qu'un checkpoint AWQ est **convertible en
Marlin**, le **repackage au chargement** et le journalise. Le noyau chronométré
n'est alors **pas** le GEMM d'AutoAWQ que notre banc du 08-10 a porté.

**Décidé d'avance : au 4B, les deux sont mesurés, dans le même processus, dans
les mêmes rounds.**

- bras `awq_marlin` : `quantization=None` — le **défaut**, donc ce qu'un
  utilisateur obtient ;
- bras `awq` : `quantization="awq"` — le GEMM forcé, comparable au bras porté du
  banc du 08-10.

Le script **imprime le journal de vLLM sur la décision de routage** et relit la
méthode résolue dans la configuration du moteur. ⚠️ Si le bras `awq` forcé ne
démarre pas (le moteur V1 a déjà retiré des méthodes de quantification par le
passé), c'est une **issue prévue** au §6, pas un échec de job.

> **Le chiffre publié portera le bras `awq_marlin`** — le défaut, donc le
> régime réel — et le bras `awq` forcé sera cité à côté comme le point qui relie
> ce job au banc du 08-10. Ce choix est fait **avant** de connaître lequel des
> deux est le plus rapide.

### 2.5 Les objets, épinglés

| | dépôt | révision | provenance de la révision |
|---|---|---|---|
| 4B AWQ | `Qwen/Qwen3-4B-AWQ` | `74d4bd2bd4bff9cafc9345221320bffb08b406a3` | `ops/awq_dequant.py:99`, **et** SHA `main` relevé le 08-17 : identiques |
| 4B base | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | `ops/awq_dequant.py:101`, idem |
| 14B AWQ / base | `Qwen/Qwen3-14B-AWQ` / `Qwen/Qwen3-14B` | `31c69efc…` / `40c06982…` | `ops/awq_dequant.py:184-186`, idem |
| 8B AWQ / base | `Qwen/Qwen3-8B-AWQ` / `Qwen/Qwen3-8B` | `4da05a8e…` / `b968826d…` | 🚨 **relevés au Hub le 08-17, jamais validés par `awq_dequant check`** — le script les **refuse** sans `--allow-unpinned-revision` |

🚨 **Ne jamais prendre `Pier-Jean/qwen3-4b-awq-deq` ni `-8b-awq-deq`** : ce sont
**nos reconstructions f16 denses**, elles ne contiennent aucun quartet. Les
mesurer produirait un « bras AWQ » qui est en fait un bras f16 aux poids
dégradés — un résultat plausible, tout faux, et indétectable au log. Le script
refuse tout dépôt dont le nom porte `-deq`.

**L'image est épinglée par tag ET par digest** :

```
vllm/vllm-openai:v0.26.0
  manifeste  sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52
  amd64      sha256:770fe65b2c73ee74a5c42165cf3433de4048cc2cd9c57a937ca4e35aba5aa87b
  poussé le 2026-07-25, 10,35 Go
```

Choix argumenté en fin de document (§9). **`:latest` est interdit** : il
change sous les pieds d'un run à l'autre et rendrait le journal non
reproductible.

### 2.6 La forme du job

Un seul processus dans le conteneur, trois bras vivants **simultanément**
(f16, awq_marlin, awq), rounds **entrelacés** dans un ordre de dispatch fixe.
C'est la condition qui rend un rapport légitime : la règle de maison n°2
interdit un quotient formé entre deux rounds n'ayant jamais coexisté.

> **Si les trois bras ne tiennent pas dans un seul processus**, le script bascule
> en mode séquentiel (un bras par processus, fusion des JSON), et le rapport est
> alors étiqueté **« rounds non entrelacés »** dans le journal **et** dans toute
> légende. Il ne se cite pas comme s'il était entrelacé.

Plafond : `--timeout 45m` → **1,35 $** au pire sur `l40sx1` (1,80 $/h).
Attendu ~15 min, soit **~0,45 $** (*estimé* : ~4 min de pull d'image 10,35 Go,
~3 min de téléchargement de 10,7 Go de checkpoints, ~3 min de chargement des
trois moteurs, ~2 min de générations, marge).

### 2.7 La ligne de commande, figée

Le script vit dans **notre** dépôt et le job tourne dans l'image **vLLM**, qui
ne le contient pas. Le convoyage se fait par le bucket que `cmd_bench` sait déjà
créer et monter en écriture (`--bucket auto`, qui appelle `sync_job_volume` sur
un dossier local et le monte sur `--out-mount`) — donc aucun mécanisme nouveau,
et le même volume sert d'aller pour le script et de retour pour le JSON.

```bash
mkdir -p /tmp/llvq-out/awq-speed-4b
cp ops/awq_speed.py /tmp/llvq-out/awq-speed-4b/

uv run ops/run.py bench \
  --image vllm/vllm-openai:v0.26.0 \
  --flavor l40sx1 \
  --timeout 45m \
  --bucket auto --root-out /tmp/llvq-out --name awq-speed-4b --out-mount /out \
  'export LLVQ_IMAGE_TAG=vllm/vllm-openai:v0.26.0' \
  'export LLVQ_IMAGE_DIGEST=sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52' \
  'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv' \
  'python3 -c "import vllm; print(\"vllm\", vllm.__version__)"' \
  'python3 /out/awq_speed.py --size 4b --arms f16,awq_marlin,awq --json /out/awq-speed-4b.json'
```

⚠️ `cmd_bench` n'a **pas** de drapeau `--env` : les variables se posent par des
lignes `export` dans le script, comme le fait déjà la campagne 14B
(`docs/archive/reprise-14b-2026-08-09.md`). Et le `--timeout` est passé
explicitement malgré son `default="30m"` — la docstring de `cmd_bench` le croit
obligatoire, l'argparse non (§1.4).

---

## §3 — La forme du rapport, décidée d'avance

> **Ce job publie DEUX rapports intra-pile — AWQ/FP16 mesuré dans vLLM, et
> LLVQ/FP16 à tête identique mesuré dans notre harnais — et JAMAIS un quotient
> traversant les deux piles.**
>
> Le rapport **vLLM** est la **médiane de 5 rapports formés round par round**,
> avec sa **plage min–max**.
>
> Le rapport **maison** est un **quotient de deux points uniques** (48,7 / 43,6
> au 4B, 34,4 / 26,5 au 8B) et sera **étiqueté comme tel dans la même
> légende** — pas dans une note lointaine, dans la même légende.

Autrement dit, la ligne publiable a cette forme, et pas une autre :

> Dans vLLM, l'AWQ 4 bits rend **×A [×A⁻ – ×A⁺]** son propre témoin f16
> (médiane de 5 rapports round par round). Dans notre harnais, notre noyau rend
> **×1,12** notre propre témoin f16 (quotient de deux points uniques, sans
> plage). **Les deux moteurs diffèrent ; A et 1,12 ne se soustraient ni ne se
> divisent.**

Un tok/s vLLM **absolu** peut figurer au journal de mesure — c'est une donnée
brute et elle a le droit d'exister. **Il n'entre dans aucune table du papier**,
parce qu'une colonne de tok/s dont les cellules viennent de deux moteurs est
exactement le genre de tableau hétérogène que ce dossier a déjà payé deux fois
(le « 5,51 contre 4,50 » de l'errata du lot A, le « 6,09 nu » aligné sur deux
appariés).

---

## §4 — Ce qu'on s'interdit de conclure

**(i) Aucune phrase de la forme « notre noyau est plus rapide / plus lent que
l'AWQ ».** L'écart bout-en-bout entre les deux piles est dominé par **vLLM
contre candle** — ordonnancement, graphes CUDA, attention paginée, noyaux de
norme et de RoPE, chemin du `lm_head` — et non par le décodeur de poids. Ce job
ne sépare pas ces deux causes et n'a aucun moyen de les séparer.

**(ii) La cellule vitesse AWQ des tables du papier RESTE VIDE.** Concrètement :
`docs/data/campagne-finale.csv` garde `speed_tokps` vide sur la ligne `awq`, et
`paper/sections/evaluation.tex:29` et `:108` gardent leur `---$^{\dagger}$`. Ce
qui change, c'est la **note de bas de tableau**, qui cessera de dire seulement
« sa vitesse n'est pas comparable » pour dire **combien** elle vaut dans son
moteur et **pourquoi** ce nombre ne peut pas remplir la case. Une case expliquée
vaut mieux qu'une case remplie par un nombre incomparable.

**(iii) Aucun chiffre de VRAM ne sort de vLLM.** `gpu_memory_utilization` est un
budget de préallocation : ce que l'outil rapporte est une **réservation**, pas
une occupation. Nos chiffres de mémoire sont *comptés* (`rtbits` sur les octets
réels, recoupés par l'affichage carte) et ceux de l'AWQ sont *calculés* par la
formule w4g128 sur ses octets de dépôt. Ce job n'ajoute rien à cette colonne et
n'a pas le droit d'y toucher.

**(iv) Le rapport maison cité est ×1,12 (4B) et ×1,30 (8B), jamais ×2,03 ni
×2,61.** Les seconds mesurent en grande partie la disparition d'une recopie de
vocabulaire, pas le noyau Leech.

**(v) Le biais de notre bras f16 est nommé AVEC SA DIRECTION.** `broadcast_matmul`
recopie le vocabulaire à chaque token ; ce défaut est dans le **dénominateur**
des rapports maison. À tête identique il porte des deux côtés, donc il **tire
nos rapports vers 1** : **nous sous-estimons notre propre avance**. Toute
citation du ×1,12 / ×1,30 qui omet ce sens est incomplète.

---

## §5 — La clause « M = 1 » du 2026-08-10, amendée explicitement

Le pré-enregistrement du 2026-08-10 écrit, sous « Ce qui est exclu d'avance »
(l. 146-149, texte exact vérifié) :

> **Le 4 bits ne sera pas cité sur un point unique à M = 1** : Marlin est une
> GEMM dont la plus petite tuile en M est 8, et à M = 1 tous les noyaux 4 bits
> convergent vers la même borne de bande passante. Le 4 bits n'existe que dans
> une table batchée.

**Cette clause visait un µ-banc de NOYAU**, celui de `planesbench` : un token,
252 matrices, un stream, des bras entrelacés qui ne mesurent qu'un décodeur de
poids. Dans ce cadre elle est juste et elle **reste en vigueur sans changement** :
le 08-10 lui a d'ailleurs obéi en publiant les **Go/s** du bras AWQ et en
avertissant que « la grandeur comparable est les Go/s, pas ce rapport ».

**Le présent job ne mesure pas un noyau.** Il mesure un **régime produit** :
batch 1, un utilisateur, un prompt, 128 tokens, sur du matériel local — le
régime qui définit la thèse de souveraineté du projet, et le seul dans lequel
`fusedrun` a jamais tourné. Y refuser le 4 bits parce que sa GEMM n'est pas dans
sa tuile optimale reviendrait à refuser de mesurer le seul régime qui nous
intéresse, sous prétexte qu'il désavantage le concurrent.

> **AMENDEMENT, posé avant la mesure : l'exclusion est MAINTENUE pour le noyau,
> et AMENDÉE pour le bout-en-bout.** Le 4 bits peut être cité à M = 1 dans une
> mesure de débit **bout-en-bout d'un moteur**, à condition (a) que le rapport
> reste **intra-pile** (§3), (b) que la médiane et la plage soient publiées, et
> (c) que la légende dise que M = 1 n'est pas le régime optimal d'une GEMM
> Marlin — donc que **ce chiffre ne majore pas** ce que l'AWQ sait faire.

**J'amende cette clause plutôt que de l'ignorer**, et l'amendement est daté du
même jour que ce document, avant tout lancement. Un pré-enregistrement qu'on
contourne en silence ne vaut rien ; un pré-enregistrement qu'on amende par écrit
avant de mesurer vaut ce que vaut l'argument de l'amendement — et celui-ci est
lisible et contestable.

---

## §6 — Table des issues, et ce que chacune fait au dossier

Le « rapport vLLM » désigne AWQ/f16 **dans vLLM** ; le « rapport maison »
désigne LLVQ/f16 **à tête identique** dans notre harnais (×1,12 au 4B). Les deux
ne se soustraient pas : cette table dit ce qu'on **écrit**, pas ce qu'on
conclut.

| issue mesurée | conséquence, décidée d'avance |
|---|---|
| **rapport vLLM > rapport maison**, plages disjointes | ⚠️ **L'issue contre nous, et elle est publiable telle quelle.** La note de bas de tableau écrit les deux rapports avec leurs moteurs, et le papier ajoute une phrase de Limitations : *dans son propre moteur, le 4 bits multiplie son témoin f16 davantage que notre noyau ne multiplie le nôtre ; les deux moteurs diffèrent, et nous ne savons pas séparer les deux causes.* **Aucun titre, aucun abstract ne change** — ils ne revendiquent déjà aucun avantage de vitesse contre l'AWQ. |
| **rapport vLLM < rapport maison**, plages disjointes | Même note, sens inversé, **et une phrase de plus** : ce résultat serait *surprenant* au vu du banc du 08-10 (§1.2), donc il se publie avec l'avertissement qu'il mesure deux moteurs et non deux décodeurs. **Interdit d'en faire un titre.** |
| **plages recouvrantes** | « **indiscernables à cette résolution** » — jamais « équivalents », jamais « parité ». C'est la règle de décision du 08-10 §4, transposée. |
| **vLLM route en `awq_marlin`** (attendu sur sm_89) | **Aucune surprise, et c'est le chiffre publié** : le défaut est le régime réel. Le bras `awq` forcé est cité à côté comme lien avec le banc du 08-10. Le journal porte la ligne de log de la décision. |
| **le bras `awq` forcé refuse de démarrer** | Le job **n'échoue pas** : il publie `awq_marlin` seul, et le journal écrit en toutes lettres que le GEMM AutoAWQ n'a pas pu être isolé dans cette version de vLLM — donc que le lien avec le banc du 08-10 n'est pas établi. |
| **`awq` et `awq_marlin` indiscernables** | Alors le repackage Marlin ne change rien à M = 1, ce qui **confirme empiriquement** le motif de la clause du 08-10 (§5) tout en justifiant son amendement. Se publie comme tel : c'est un petit résultat, mais c'en est un. |
| **le témoin f16 vLLM sort très loin de nos 43,5 tok/s** (dans un sens ou l'autre) | Ce n'est **pas** une anomalie : c'est la mesure du confondant de moteur lui-même, et c'est la donnée la plus utile du job pour justifier le §4 (i). Elle est publiée dans le journal, **hors table**. |

**Aucune de ces issues ne justifie de ne pas publier.** Elles changent une note
de bas de tableau, pas une décision.

---

## §7 — Ce qui invaliderait le job

Chacun de ces points fait **sortir le script en code non nul** (`cmd_bench`
tourne sous `set -euo pipefail`, donc le job passe en `ERROR` plutôt qu'en
`COMPLETED` sur un résultat vide — un job vert sans chiffre est le pire des cas).

1. **Divergence du contrôle f16 au token 1.** Le premier token généré par le
   bras f16 vLLM au 4B doit se détokeniser en `" Paris"`, comme dans
   [`mesures/planes14-fusedrun-2026-08-06.txt`](../docs/mesures/planes14-fusedrun-2026-08-06.txt).
   Une divergence **au premier token** signifie mauvais modèle, mauvais
   tokenizer ou échantillonnage non glouton : **le job est faux**, tous ses
   chiffres avec. ⚠️ Une divergence **tardive** est **attendue** (deux ordres
   d'accumulation, deux moteurs) : elle est signalée, elle n'échoue pas.
2. **Moins (ou plus) de 128 tokens produits** par une génération quelconque.
3. **Prefix caching resté actif** — relu dans la configuration du moteur après
   initialisation, pas supposé depuis le drapeau passé.
4. **Dispersion inter-round > 5 %** sur un bras, mesurée en
   `(max − min) / médiane` sur les 5 rounds gardés. Au-delà, la carte ou le
   moteur n'est pas en régime et la médiane ne décrit rien.
5. **Version d'image non épinglée dans le journal** : le script exige la
   variable `LLVQ_IMAGE_TAG` et refuse de démarrer sans elle, ou si elle
   contient `latest`. Il imprime en plus la version exacte de vLLM lue dans le
   paquet installé — les deux, parce qu'un tag peut mentir et qu'une version
   sans tag ne se re-lance pas.

Deux invalidations supplémentaires, spécifiques à ce job :

6. **Un dépôt dont le nom porte `-deq`** (nos reconstructions denses) : refus
   avant tout chargement.
7. **Une révision non épinglée** sans `--allow-unpinned-revision` explicite.

---

## §8 — Journal des écarts au protocole, tenu à chaud

*(vide à la signature)*

---

## §9 — Annexe : pourquoi ce tag d'image, et ce qui reste dû

**Retenu : `vllm/vllm-openai:v0.26.0`**, digest de manifeste
`sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52`
(amd64 `sha256:770fe65b…`), poussé le **2026-07-25**, 10,35 Go — existence,
date et digest **vérifiés le 2026-08-17** via l'API de Docker Hub.

Le raisonnement, pour qu'il soit contestable :

- **`:latest` est exclu** par principe : il se déplace, donc il rend le journal
  irreproductible.
- **La tête est `v0.27.1`** (2026-08-11, correctif de `v0.27.0` du 08-10) —
  **six jours** d'exposition. Un `.0` tout frais et son premier correctif sont
  précisément la fenêtre où l'on rencontre les régressions de moteur ; ce job
  n'a pas de budget pour en découvrir une.
- **`v0.26.0`** (2026-07-27 au dépôt, image poussée le 07-25) a **trois
  semaines** de terrain, un minor complet de retard sur la tête, et c'est la
  dernière version stable dont l'écosystème a réellement l'usage.
- **CUDA** : les roues vLLM ont **CUDA 12.9 par défaut** sur cette lignée, ce
  qui satisfait la contrainte « CUDA 12.x » — la compatibilité mineure de CUDA 12
  fait tourner un binaire 12.9 sur tout pilote 12.x. Les images `nightly` en
  **CUDA 13.0** exigeraient un pilote ≥ 580, inconnu sur les cartes HF, et sont
  écartées pour cette seule raison.
- **sm_89 (Ada / L40S)** est une cible de première classe de vLLM ; aucune
  version récente ne la manque.

🚨 **Ce qui reste dû, et qui ne peut pas être vérifié depuis cette machine** :

1. **Que HF Jobs remplace bien l'`ENTRYPOINT` de l'image.** L'image vLLM porte
   `ENTRYPOINT ["vllm", "serve"]` ; si le `command` de `run_job` était **ajouté**
   plutôt que **substitué**, le conteneur exécuterait `vllm serve bash -lc …` et
   le job mourrait sans mesurer. Deux indices vont dans le bon sens — `JobInfo`
   expose `command` **et** `arguments` comme deux champs distincts (sémantique
   Kubernetes, où `command` écrase l'entrypoint), et la doc HF donne
   `run_job(image="duckdb/duckdb", command=["duckdb", "-c", …])` sur une image à
   entrypoint — mais **aucun des deux n'est une preuve**.
   > **Levée à ~0,01 $, et elle est recommandée avant le job payant** :
   > ```bash
   > uv run ops/run.py bench --image vllm/vllm-openai:v0.26.0 \
   >   --flavor cpu-upgrade --any-flavor --timeout 10m --name vllm-entrypoint-probe \
   >   'echo ENTRYPOINT-OVERRIDE-OK' \
   >   'python3 -c "import importlib.metadata as m; print(\"vllm\", m.version(\"vllm\"))"'
   > ```
   > La version est lue dans les **métadonnées du paquet**, sans `import vllm` :
   > sur une machine sans carte, l'import peut échouer pour une raison qui n'a
   > rien à voir avec la question posée, et la sonde deviendrait ambiguë.
   > Si la sortie porte `ENTRYPOINT-OVERRIDE-OK` et une version, la question est
   > close. Sinon il faut passer par `hf jobs uv run --image vllm/vllm-openai:…`,
   > que la doc HF documente explicitement pour cette image — et `ops/run.py`
   > n'a pas de sous-commande pour ça (`cmd_dequant` utilise `run_uv_job` mais
   > code son script en dur).
2. **Que le moteur V1 accepte `quantization="awq"` forcé** dans cette version.
   Traité comme une issue (§6), pas comme une hypothèse.
3. **Que trois moteurs vLLM cohabitent dans un processus** avec les
   `gpu_memory_utilization` posés au §2.3. Traité par un repli séquentiel
   déclaré (§2.6).
