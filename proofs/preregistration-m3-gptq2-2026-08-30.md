# Pré-enregistrement — M3 : le bras GPTQ 2 bits, produit chez nous, chronométré et scoré dans vLLM

> Écrit et commité **avant le lancement**, et avant qu'aucun octet de GPTQ
> n'existe sur ce projet. Ce job **remplit une case vide** — le dossier n'a
> jamais mesuré la qualité d'un concurrent **2 bits** : le 17,04 de QTIP est une
> citation de la Table 6 du papier, pas une mesure de notre harnais, et F2 le dit
> lui-même (payload pseudo-aléatoire, « aucune phrase de qualité ne peut
> s'appuyer sur ce bras »).
>
> ⚠️ **Statut du tampon.** L'antériorité repose sur la date de commit tant que
> `ots stamp` n'est pas passé. 🚨 **L'opérateur doit tamponner avant le premier
> job**, comme pour P1, P1b, F2 et D1. Sans tampon, ce document reste une
> intention datée par git, pas une ancre.
>
> Contexte de dossier : [`docs/exp-piles-isolees-2026-08-30/`](../docs/exp-piles-isolees-2026-08-30/).

---

## §1 — Divulgation datée : tout ce qui est connu à la signature

### 1.1 Ce que le projet a déjà mesuré, et qui pourrait « inspirer » la forme du rapport

| grandeur | valeur | provenance |
|---|---|---|
| 4B, LLVQ servi `Planes14` + `q8` | **87,0** tok/s [86,8–87,0] · 2,56 Go | *mesuré*, B2 |
| 4B, LLVQ à **tête identique** | **48,3** [48,1–48,3] → **×1,11** [1,11–1,11] | *mesuré*, B2 |
| 4B, notre bras dense f16 | **43,6** [43,5–43,6] · 8,04 Go | *mesuré*, B2 |
| 4B, AWQ dans **vLLM** | **200,49** [200,39–200,61] tok/s | *mesuré*, `awq-vllm-4b-2026-08-17` |
| 4B, témoin **f16 de vLLM** | **83,09** tok/s | *mesuré*, même job |
| 4B, ppl : f16 · AWQ · LLVQ | 12,2369 · 13,5207 (×1,105) · 16,9422 (×1,385) | *mesuré*, campagne A4 |
| 4B, MMLU micro : f16 · AWQ · LLVQ | 70,32 · **70,04** · **55,59** | *mesuré*, campagne A4 |
| 4B, b/param modèle entier : LLVQ · AWQ | 5,162 · 5,302 | *mesuré*, `rtbits` |

### 1.2 🚨 Les résultats déjà mesurés qui sont CONTRE nous, et qui sont publiés

1. **Sur un 4B, le 4 bits nous domine sans discussion en qualité** : MMLU 70,04
   pour l'AWQ officiel contre **55,59** chez nous, soit **−14,45 pp** apparié
   [+11,60 ; +17,27]. La ppl dit la même chose : ×1,105 contre ×1,385.
2. **Sur l'axe 2 bits, le concurrent vivant nous bat en vitesse** : QTIP tourne
   **2,27× [2,27–2,28]** plus vite que `Planes14` en lisant 2,40× moins d'octets
   (F2, division licite — un seul processus, bras entrelacés).

### 1.3 🚨 Le biais propre à CE job, et il faut le nommer

**Nous attendons de gagner ce bras.** GPTQ à 2 bits est réputé mauvais, et
l'étude empirique de la quantification de Qwen3 (arXiv:2505.02214) ne lui accorde
qu'« un niveau de performance minimal » — le meilleur des mauvais.

C'est exactement la situation où l'on relâche les contrôles : un résultat
favorable **attendu** ne déclenche pas la relecture qu'un résultat défavorable
déclenche. D'où les §4 et §7, écrits maintenant.

### 1.4 Ce qui a été vérifié dans le dépôt et sur le Hub, avant d'écrire (2026-08-30)

| fait | valeur | conséquence |
|---|---|---|
| `ops/awq_speed.py:143` | `ARMS: dict[str, tuple[str, str \| None]]` — *nom → (dépôt, `quantization`)* | ajouter le bras est **une entrée**, pas du code neuf |
| `ops/awq_speed.py:296` | ses clés de routage de noyau contiennent déjà `"gptq"` | les logs de routage seront capturés sans modification |
| `ops/awq_speed.py` | rounds **séquentiels, non entrelacés** — le script le déclare | ⚠️ à répéter dans le rapport |
| `llvq-llm/src/corpus.rs:187` | C4 anglais, **shard 1** = calibration, shard 0 = évaluation | le bras GPTQ sera calibré **sur le même shard que LLVQ** |
| `ops/awq_dequant.py:29` | interdit formel de `gptqmodel.utils.model_dequant.convert_awq_file` | **aucune bibliothèque de dequant n'est de confiance ici** |
| relaxml sur le Hub | QTIP publié pour **Llama seulement** | QTIP hors périmètre, décidé et écrit |
| registre | **87,94 $ sur 75 jobs** | cumul à la signature |

### 1.5 Le budget

**Plafond posé par l'opérateur : 5 $ pour cette vague.** Décomposition *estimée* :
production de l'artefact ~0,5–1,0 $, machine M3 (vitesse + qualité) ~0,3 $, le
reste étant de la marge pour deux ou trois reprises. 🚨 **Au-delà de 5 $, le job
s'arrête et l'opérateur est resollicité.**

---

## §2 — Le protocole, figé maintenant

### 2.1 Les objets, épinglés

| objet | épingle |
|---|---|
| modèle de base | `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c` |
| corpus de calibration | C4 anglais **validation, shard 1** — le shard réservé à la calibration |
| volume de calibration | **131 k tokens**, le volume de tous nos artefacts publiés |
| image vLLM | épinglée par `LLVQ_IMAGE_TAG` / `LLVQ_IMAGE_DIGEST`, la même qu'au 08-17 |
| carte | **L40S** — 🚨 verdict L40S/Ada, cf. §4 |

### 2.2 Ce qui est répliqué à l'identique du protocole maison

- **Vitesse** : médiane sur **5 rounds**, 1 génération jetée, **avec plage**.
  Jamais un point unique.
- **ppl** : wikitext-2 test, ctx 4096, **12 fenêtres**, f16, **NLL par fenêtre
  conservées**.
- **MMLU** : micro (`Σright/Σtotal`), **2 280 questions**, **dump par question
  avec `qhash`** — jamais un taux agrégé seul.
- **Empreinte de tokens imprimée des deux côtés** : `3f1baca9033bf251` (ppl),
  `65dcd53655e8bfa5` (MMLU).

### 2.3 Ce qui en diffère délibérément, et qui doit être déclaré

1. **La qualité est scorée DANS vLLM** (`prompt_logprobs`), pas par déquantisation
   vers f16 dense. C'est un choix : on mesure le bras **tel qu'il est déployé**,
   et une seule brique sert M2 et M3 au lieu d'un pipeline de dequant par format.
2. **Conséquence assumée** : le *niveau* de ppl n'est comparable qu'à l'intérieur
   de vLLM. Le **niveau de MMLU**, lui, traverse — il ne dépend que du tokenizer
   et du logprob de 4 tokens de réponse.
3. **Le témoin f16 tourne dans le même processus**, sur les mêmes données.

### 2.4 🚨 LE GATE DE L'INSTRUMENT — le scorer vLLM doit reproduire une réponse connue

Le scorer `prompt_logprobs` est un **instrument neuf**. La leçon du §5 du
`CLAUDE.md` — et celle, plus récente, du `grep` sur les `.ots` — dit qu'on
n'accorde aucune confiance à un instrument avant d'avoir prouvé qu'il rend la
bonne valeur sur un cas dont on connaît la réponse.

**Le cas connu existe et il est gratuit : le bras AWQ.** Nous en avons déjà la
qualité, mesurée dans notre propre harnais par déquantisation :

| | valeur connue |
|---|---|
| AWQ 4 bits, MMLU micro | **70,04** |
| AWQ 4 bits, ppl | **13,5207** |
| f16, MMLU micro | **70,32** |

⇒ **Le scorer vLLM tourne d'abord sur `f16` et `awq_marlin`.** S'il ne reproduit
pas ces valeurs, **il est cassé et le bras GPTQ n'est pas lu.** Critère chiffré,
posé maintenant :

- **MMLU** : |Δ| ≤ **1,5 pp** sur chacun des deux bras connus (la SE appariée
  entre modèles vaut 0,79 à 1,44 pp — en deçà, on ne saurait pas distinguer un
  scorer juste d'un scorer légèrement faux).
- **ppl** : |Δ| ≤ **2 %** sur chacun des deux bras connus.

🔎 **Et ce gate a une valeur au-delà du contrôle** : s'il passe, l'hypothèse
« le rapport au témoin transfère entre piles » cesse d'être supposée et devient
**mesurée** sur le bras AWQ, qui est le seul à exister des deux côtés.

### 2.5 Les asymétries, déclarées d'avance et dans quel sens elles jouent

| asymétrie | sens |
|---|---|
| à 2 bits, vLLM n'utilise **pas Marlin** (4 bits seulement) mais le chemin ExLlamaV2, moins optimisé | **POUR NOUS** — le bras GPTQ est handicapé en vitesse |
| le bras GPTQ est calibré sur **notre** corpus (C4 shard 1), l'AWQ officiel sur un corpus inconnu | **POUR NOUS** vis-à-vis de l'AWQ — c'est le seul bras tiers dont la calibration est la nôtre |
| `M = 1` n'est pas le régime optimal d'une GEMM (plus petite tuile en `M = 8`) | **CONTRE le bras GPTQ** — ce chiffre ne majore pas ce qu'il sait faire |
| notre bras dense f16 recopie 778 Mo de vocabulaire par token | **CONTRE NOUS**, mais hors de ce job — M1 n'est pas rejoué ici |

### 2.6 🚨 Le piège de format : `g_idx` / act-order

GPTQ avec `desc_act` permute les canaux d'**entrée**. C'est le cousin exact du
piège que `ops/awq_dequant.py` existe pour prévenir — *« publishing plausible,
wrong weights »*, une permutation qui charge, tourne et produit des nombres.

**Deux faits qui décident du traitement** :

- ✅ La permutation GPTQ est **stockée dans le fichier** (`g_idx`), pas une
  convention implicite comme l'`AWQ_ORDER`. Elle se lit, elle ne se devine pas.
- ⚠️ Le contrôle « le résidu contre le modèle de base vaut le bruit de
  quantification » **s'affaiblit à 2 bits** : à 4 bits il séparait 0,105 (bon
  ordre) de 0,97 (mauvais) ; à 2 bits le plancher de bruit monte et l'écart se
  resserre. **Le contrôle se dégrade là où on en a besoin.**

⇒ **Décision, prise maintenant** : ce job ne déquantifie pas, donc il n'assume
aucune convention de dépaquetage — c'est vLLM qui lit le fichier avec le code de
son propre format. Le contrôle qui remplace L1 est le **gate §2.4** : un bras
dont les poids seraient permutés ne reproduirait aucune valeur connue.

---

## §3 — La forme du rapport, décidée d'avance

1. **Trois grandeurs, trois provenances étiquetées** : b/param (*calculé* sur
   octets *mesurés*), qualité (*mesuré*), débit (*mesuré*, médiane à plage).
2. **Le débit se donne en DEUX formulations** : le tok/s brut **et** son rapport
   au témoin f16 **de la même pile**. 🚨 Le brut ne se publie jamais seul.
3. **Une table, pas une narration.** Chaque cellule porte sa barre ou déclare
   qu'elle n'en a pas.
4. **La mention « rounds séquentiels, non entrelacés »** figure dans le rapport,
   parce que le script la déclare.
5. **Les NLL par fenêtre et les dumps par question sont commités**, pas résumés.
   Le §7 du `CLAUDE.md` a payé trois fois pour cette règle.

---

## §4 — Ce qu'on s'interdit de conclure

1. 🚨 **Aucune division entre piles.** Ni `t(vLLM) ÷ t(fusedrun)`, ni l'inverse,
   ni sous forme de phrase. Le témoin f16 de vLLM rend 83,09 là où le nôtre rend
   43,6 : l'écart est dominé par le moteur.
2. 🚨 **« Nous battons le 2 bits » ne se dit pas sur ce bras.** GPTQ 2 bits est
   le **plancher du marché**, pas l'état de l'art. L'état de l'art à 2 bits est
   QTIP, et sa qualité reste non mesurée par nous.
3. 🚨 **Aucun classement entre deux bras 2 bits sur un seul tirage chacun.** Les
   deux sont **calibrés**, donc chacun est un tirage, et F5 donne σ = **5,2 %**
   à cette taille. Pour un écart de 14 pp, sans effet ; **pour un écart de 3 pp,
   décisif** — et 3 pp est l'ordre de grandeur plausible.
4. **Verdict L40S/Ada.** F4 a mesuré que sur A100 aucun bras à décodage ne bat
   FP16 et que le ×1,78 **est** le rapport d'horloges. Rien ici ne s'étend à une
   autre carte.
5. **Aucune loi d'échelle** : une seule taille.
6. **Rien sur le raisonnement en usage réel** : MMLU est un QCM.

---

## §5 — Table des issues, et ce que chacune fait au dossier

| issue | ce qu'elle établit |
|---|---|
| **le gate §2.4 est rouge** | le scorer est cassé ; **le bras GPTQ n'est pas lu**, et on l'écrit. Le job a coûté ~0,3 $ et rendu un fait négatif utile |
| GPTQ 2 bits nettement sous LLVQ en MMLU | la case vide se remplit : **notre premier concurrent 2 bits mesuré**, et il perd. ⚠️ contre le *plancher du marché*, pas contre l'état de l'art |
| GPTQ 2 bits **au-dessus** de LLVQ | résultat majeur et défavorable — à publier tel quel, et à instruire (calibration ? format ? les deux ?) |
| écart < 3 pp dans un sens ou l'autre | **les données sont muettes** (§4.3). On l'écrit, on ne le narre pas |
| la quantification GPTQ échoue sur Qwen3 | blocage technique documenté, comme F2 l'autorisait pour QTIP. ~0,5 $ pour un fait |

---

## §6 — Ce qui invaliderait le job

1. Le **témoin f16 de vLLM** ne reproduit pas **83,09** tok/s à ±3 %.
2. Le **gate §2.4** est rouge sur l'un des deux bras connus.
3. Les **empreintes de tokens** diffèrent d'une machine à l'autre.
4. Le b/param du GPTQ n'est pas cohérent avec les octets de son fichier.
5. Le **plafond de 5 $** est atteint.

Dans chacun de ces cas : **on arrête, on écrit ce qui s'est passé, on ne lit
aucun bras quantifié.**

---

## §7 — Journal des écarts au protocole, tenu à chaud

> À remplir pendant le run, jamais après. Un écart non consigné à chaud est un
> écart qui se réécrit en voyant les nombres.

| date | écart | raison | effet sur les conclusions |
|---|---|---|---|
| — | — | — | — |
