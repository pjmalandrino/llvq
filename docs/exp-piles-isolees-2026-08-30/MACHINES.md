# Les machines — deux backends, une fiche par bras (2026-08-30)

> **Rien n'est lancé.** Commandes *indicatives* : celles des machines ✅ viennent
> de jobs passés, celles des ⚠️ restent à valider au moment du lancement.
>
> Modèle unique : **Qwen3-4B**. Une seule taille — cf. `PROTOCOLE.md` §8.

## 0. La règle du backend

🚨 **Un backend est une pile.** Les × ne se divisent pas entre CUDA et Metal —
*a fortiori*, puisque le dossier a déjà mesuré qu'ils ne se divisent même pas
entre **deux cartes CUDA** : sur A100 aucun bras à décodage ne bat FP16
(`Planes14` 0,79×), et le lot G a tranché que le ×1,78 **est** le rapport
d'horloges 2 520/1 410 MHz.

Chaque backend a donc **son propre témoin f16**, et on compare des rapports.

## 1. Quel bras tourne où

| bras | CUDA (HF Jobs, payant) | Metal (ta machine, 0 $) |
|---|---|---|
| **LLVQ 2 bits** | ✅ **servi** — `fusedrun` | ⚠️ **micro-banc seulement** — pas de `fused_metal.rs` |
| **AWQ 4 bits** | ✅ servi — vLLM | ❌ aucun moteur |
| **GPTQ 2 bits** | ⚠️ servi — vLLM, artefact à produire | ❌ aucun moteur |
| **`IQ2_XXS`** | ⚠️ servi — llama.cpp CUDA | ⚠️ **servi — llama.cpp Metal** |
| **MLX 2 bits** | ❌ n'existe pas hors Apple | ⚠️ **servi — natif**, artefact à produire |
| ~~QTIP~~ | ❌ pas d'artefact Qwen3 | ❌ |

🔎 **`IQ2_XXS` est le seul bras qui traverse les deux backends.** C'est le pont
de toute l'expérience : le seul format dont on pourra dire « voici son rapport à
son témoin f16 ici, et là ».

🚨 **Le piège Metal, et il est structurel.** LLVQ n'a pas de chemin servi sur
Metal : `bin/thesis` mesure **252 projections sur un token**, pendant que
`mlx_lm.generate` fait tourner un **modèle entier sur 256 tokens**. Les comparer
serait exactement la faute « deux dénominateurs » du §7. Sur Metal, LLVQ ne joue
qu'en **qualité** et en **micro-banc**.

---

# Partie A — CUDA, sur HF Jobs

## M1 — LLVQ 2 bits, notre noyau ✅

- **Artefact** : le 4B scellé, `Planes14` servi, `LLVQ_EMBED=q8`.
- **Témoin f16** : notre bras dense, même binaire. ⚠️ **handicapé** —
  `Head::project` → `broadcast_matmul` recopie 778 Mo de vocabulaire par token.
  Asymétrie **contre nous**. D'où la règle des deux formulations : le × servi ne
  se publie jamais seul, son compagnon à **tête identique** va dans la même table.
- **Configuration publiée** : `LLVQ_ROT_SHARE=0 LLVQ_FUSE=0`. 🚨 Ne pas activer
  la fusion de D1 : les tables à trois tailles reposent sur une configuration
  identique partout.

```bash
LLVQ_FUSED_LAYOUT=planes14 LLVQ_EMBED=q8 \
  cargo run --release -p llvq-llm --features cuda --bin fusedrun
```

**Valeurs connues** : 87,0 tok/s [86,8–87,0] dans 2,56 Go · 5,162 b/param.

## M2 — AWQ 4 bits, dans vLLM ✅

- vLLM **0.26.0**, image épinglée. Vitesse par `ops/awq_speed.py` (⚠️ le script
  déclare lui-même : **rounds séquentiels, non entrelacés**). Qualité par
  `ops/awq_dequant.py`, qui le ramène en f16 dense dans notre harnais — donc
  **empreinte de tokens identique** à M1.
- **Valeurs connues** : 200,49 tok/s [200,39–200,61] contre un f16 vLLM à
  **83,09** · ppl 13,5207 · MMLU 70,04 · 5,302 b/param.

🔎 **C'est la machine de contrôle du design.** Seul bras mesuré des deux côtés —
chez lui *et* chez nous. Si ses deux rapports coïncident, l'hypothèse « le
rapport au témoin transfère entre piles » devient **mesurée**. Ce contrôle ne
coûte rien de plus et vaut mieux qu'un bras de plus.

## M3 — GPTQ 2 bits, dans vLLM ⚠️

- **Pourquoi** : le **plancher du marché** — ce qu'une pile standard donne
  réellement à 2 bits. L'étude arXiv:2505.02214 conclut qu'à 2 bits sur Qwen3,
  seules les méthodes à compensation par calibration tiennent.
- **Artefact à produire** : GPTQModel, calibré sur **C4 anglais shard 1** — le
  shard que `llvq-llm/src/corpus.rs:187` réserve à la calibration. 🔎 **Même
  corpus que LLVQ** : supprime le confondant de l'AWQ officiel, calibré ailleurs.
- ⚠️ **En notre faveur** : à 2 bits vLLM n'utilise pas Marlin (4 bits seulement)
  mais le chemin ExLlamaV2, moins optimisé.
- **Travail** : `ops/awq_speed.py` porte déjà `ARMS: dict[str, tuple[str, str|None]]`
  (l. 143) et grep déjà `"gptq"` (l. 296). C'est **une entrée**, pas du code neuf.

## M4 — `IQ2_XXS`, llama.cpp CUDA ⚠️

- **2,06 bpw** contre nos 2,0702 — la comparaison de débit la plus serrée du
  dossier. 🚨 **`Q2_K` ne convient pas** : ~2,6–3,0 bpw, aucun codebook.
- **Contrefactuel LUT** du `docs/BACKLOG.md` §4.4 : un codebook assez petit pour
  tenir en table, là où nos 1,1·10¹⁴ points imposent le dépliage à 4,80 b/poids.
- **Artefact produit sur Metal** (partie B), puis **le même GGUF** tourne ici.

---

# Partie B — Metal, sur ta machine — **0 $**

> **Principe d'ordre** : tout ce qui peut être tranché sur Metal l'est **avant**
> de louer une carte. C'est ce que le projet a fait historiquement — « un banc
> gratuit vaut mieux qu'une carte louée pour trouver les pièges de mesure ».

## N0 — L'équivalent Metal de F1 🚨 **le plus utile, et il n'existe pas**

`docs/fiche-4b.md:438` nomme l'angle et le déclare non adressé :

> *« Ce baseline n'a par ailleurs jamais été confronté à MPS, MLX ou Accelerate :
> le 2,07× est un rapport contre un noyau écrit par le même auteur. C'est l'angle
> hostile restant, non adressé. »*

Sur CUDA, **F1 a réglé exactement ça** : notre témoin FP16 maison est à
**1,024×** (banc 2 bras) et **1,015×** (banc 5 bras) de cuBLAS sur L40S, tous
deux ≤ 1,05 — c'est ce qui fait tenir *tous* les rapports « vs FP16 » publiés.

**Il n'y a aucun équivalent Metal.** Tant qu'il manque, le 2,03–2,09× de
`bin/thesis` est un rapport contre un noyau du même auteur. Confronter notre
matvec FP16 Metal à **MPS / MLX / Accelerate** coûte **0 $** et vaut plus que
n'importe quel bras supplémentaire de cette expérience.

## N1 — LLVQ sur Metal : qualité complète + micro-banc ⚠️

- **Qualité** : complète, via `sealed::load` — et c'est un **contrôle
  inter-backend**. Le même artefact scoré sur Metal et sur CUDA doit rendre la
  même chose. Historiquement vérifié : baseline **70,42 (Metal) → 70,32 (CUDA)**,
  soit 0,08 σ.
- **Noyau** : `bin/thesis` — un token, 252 matrices, 7 rounds dont 2 jetés,
  rapport formé round par round. **2,03–2,09× vs FP16**, 1 105 920 lignes
  vérifiées contre référence f64.
- ❌ **Pas de tok/s bout-en-bout** : pas de `fused_metal.rs`.

```bash
cargo run --release -p llvq-metal --bin mslcheck   # les 7 points d'entrée MSL, 3 s
cargo run --release -p llvq-metal --bin thesis     # LLVQ vs FP16, 252 projections
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal \
  --bin ppl -- 4096 12 metal ~/qwen3-4b-llvq.bin 2> ppl-nll-metal.txt
```

## N2 — `IQ2_XXS`, llama.cpp Metal ⚠️ **le pont**

- llama.cpp est **first-class sur Metal** : c'est le moteur natif macOS.
- **Production locale, 0 $** : `llama-imatrix` sur le même C4 shard 1, puis
  `llama-quantize` en `IQ2_XXS`. Le GGUF produit ici part ensuite sur M4.
- **Mémoire** : `octets du GGUF ÷ params` **est** le b/param modèle entier,
  mesuré. Meilleure provenance que notre bras, dont l'embedding est *modélisé*
  à 8,5 b/param — à étiqueter comme tel.
- ⚠️ **Trou** : pas de harnais MMLU. Son `--multiple-choice` attend un autre
  format ; rejouer nos 2 280 questions au même `qhash` demande la fixture.
  **Décision en souffrance** : bâtir la fixture, ou ppl seule au premier tour.

## N3 — MLX 2 bits, natif Apple ⚠️

- ✅ **Vérifié le 2026-08-30** : `mlx_lm.convert.convert(hf_path, mlx_path,
  quantize, q_group_size, q_bits, dtype, …, dequantize, quant_predicate)` —
  `q_bits` existe, **mlx_lm 0.24.0 est installé**, et l'artefact **q4 est déjà
  local** (`~/qwen3-4b-mlx-q4`, 2,1 Go).
- 🔎 **`dequantize` est un paramètre** : MLX sait rendre un f16 dense, donc la
  qualité d'un artefact MLX est scorable **dans notre harnais**, à empreinte
  identique. C'est le chemin décrit par `fiche-4b.md:556`.
- ⚠️ **Asymétrie déjà documentée et jamais dite** : **MLX quantifie AUSSI
  l'embedding** — 253 tenseurs = 252 projections + `model.embed_tokens`
  (`fiche-4b.md:339`). Le bon comparateur de notre artefact est « q4 sur les
  linéaires + embedding f16 », **pas** le fichier MLX tel quel.

🕳️ **Deux dettes à solder au passage, quasi gratuites.** `fiche-4b.md` marque
**SUSPECT** le « 129,8 tok/s » et le « 2,39 Go » du q4 MLX : *aucune trace — ni
log, ni script, ni historique shell*. Son §563 les répare en **~2 min** avec
`/usr/bin/time -l mlx_lm.generate … | tee`. À faire pendant qu'on est sur la
machine.

---

## Hors périmètre — QTIP ❌

Vérifié le 2026-08-30 : **relaxml ne publie QTIP que pour la famille Llama**.
Aucun checkpoint Qwen3 ; le porter coûterait *estimé* 10–20 $ plus un risque
d'incompatibilité d'architecture.

**Sa vitesse est déjà mesurée, et mieux que cette expérience ne le ferait** : F2
l'a chronométré dans **un seul processus**, bras entrelacés —
`t(Planes14) ÷ t(QTIP)` = **2,27× [2,27–2,28]**, division **licite**. Lui donner
sa propre machine serait un recul. Ce qui lui manque reste sa **qualité**.
