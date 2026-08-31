# Protocole « piles isolées » v2 — la version réutilisable (2026-08-31)

> **Objet.** Comparer le bras LLVQ servi à des quantifieurs concurrents, chacun
> dans son propre moteur, sur les mêmes données — et pouvoir **ré-exécuter**
> cette comparaison après chaque amélioration du noyau, à protocole constant.
>
> **Statut.** Ce document est destiné au tampon OpenTimestamps. Il ne s'éditera
> **jamais** : tout écart d'une exécution future s'écrit dans un fichier
> `-ECARTS-<date>.md` à côté, nommé ici d'avance. Il ne contient aucune section
> à remplir — c'est la leçon n°9 de l'audit
> (`docs/exp-piles-isolees-2026-08-30/AUDIT-2026-08-31.md`), payée trois fois
> par ce dépôt (préregs du 08-10, du 08-11, et la table d'écarts du préreg M3).
>
> **Généalogie.** v1 = `docs/exp-piles-isolees-2026-08-30/{README,PROTOCOLE,
> MACHINES}.md` + `proofs/preregistration-m3-gptq2-2026-08-30.md`, exécutés les
> 30-31 août (12 jobs, 1,29 $, journaux `docs/mesures/m3-*` et `m4-*`). Chaque
> règle ci-dessous qui diffère de v1 cite la leçon d'audit qui l'a changée.

---

## §1 — Les constantes ancrées

Toute exécution future se vérifie contre ces valeurs avant de produire un
chiffre neuf. Elles sont *mesurées*, leurs journaux sont commités.

### 1.1 L'objet

| constante | valeur | provenance |
|---|---|---|
| modèle | `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c` | épinglé partout |
| **paramètres réels** (têtes LIÉES) | **4 022 468 096** | vérifié par **4 instruments** : notre comptabilité, le GGUF f16 (16,013 b/param), `llama-quantize`, `llama-bench` (« 4.02 B »). Le compte de gptqmodel (4 411 424 256) est FAUX de +9,67 % — leçon n°7 |
| empreinte ppl | `3f1baca9033bf251` (wikitext-2 test, ctx 4096, 12 fenêtres) | tous les journaux ppl |
| empreinte MMLU | `65dcd53655e8bfa5` (2 280 questions, 57 matières) | tous les dumps |
| empreinte calibration | `40300263e5d0afa2` (C4 en/validation **shard 1**, 305 docs, 131 072 tokens = 64×2048, préfixe contigu) | `m3-gptq2-production`, `m3-iq2-metal` |
| tokens de réponse MMLU | `' A'`=362 `' B'`=425 `' C'`=356 `' D'`=422 — un token chacun, à vérifier à chaque tokenizer | `m3-qhash-local` |

### 1.2 L'étalon de scorer — le f16 sur quatre moteurs

Tout scorer MMLU neuf doit reproduire le f16 dans **[70,3 ; 70,9]** (micro).

| moteur | MMLU micro |
|---|---|
| candle + notre noyau CUDA | 70,32 |
| vLLM 0.26.0 + Marlin | 70,34 |
| transformers | 70,84 |
| llama.cpp (Metal) | 70,36 |

Étendue mesurée : **0,52 pp**. Un scorer hors de cette bande est cassé — on ne
lit aucun bras quantifié derrière lui (leçon n°1 : ce gate a attrapé une erreur
réelle à son premier usage, pour 0,20 $).

### 1.3 Les témoins de débit, par pile (L40S sauf mention)

| témoin f16 | tok/s | pile |
|---|---|---|
| notre bras dense | 43,6 [43,5–43,6] | candle — ⚠️ **handicapé** (778 Mo de vocabulaire recopiés/token) |
| vLLM 0.26.0 | 83,09 | vLLM |
| llama.cpp CUDA | 84,83 ± 0,05 | llama.cpp |
| llama.cpp Metal (M3 Max) | 43,08 ± 0,10 | llama.cpp |

🔎 vLLM et llama.cpp s'accordent à 2,1 % — le handicap de notre témoin (1,95×)
est corroboré par deux moteurs extérieurs. **Seul le × à tête identique mesure
notre noyau.**

### 1.4 Les artefacts de référence, par hachage

| artefact | sha256 |
|---|---|
| `qwen3-4b-iq2xxs.gguf` (2,0625 bpw, imatrix C4 shard 1) | `19a8ed4946353b6fdc5d19ba9766ffe903923cb7aefe848dcbcd5dba1de27605` |
| `qwen3-4b-f16.gguf` | `259393f93c7a55515161c1db0609118d8c1e94f78c01ad75f5d8ee0471cc0c3f` |
| préreg M3 (v1) | `87575cd00a967973a93852df29aab5c7192a287f5b2f3a9382041f4b9ec304d4` |

### 1.5 La table de référence v1 — ce que la ré-exécution compare

| branche | pile | b/param | MMLU micro | ppl (× témoin) | débit (× témoin de SA pile) |
|---|---|---|---|---|---|
| LLVQ 2 bits `Planes14`+q8 | candle/CUDA | 5,162 | 55,59 | ×1,3845 | ×2,00 brut · **×1,11** tête identique |
| AWQ 4 bits | vLLM/CUDA | 5,302 | 69,82 (chez lui) · 70,04 (chez nous) | ×1,1049 | ×2,413 |
| IQ2_XXS | llama.cpp/Metal | 2,479 | 39,39 | ×2,6287 | ×2,647 |
| IQ2_XXS | llama.cpp/CUDA | 2,479 | 38,87 | — | ×3,688 |
| GPTQ 2 bits g128 | — | 3,489 | ∅ ne génère pas | ∅ | ∅ vLLM 0.26.0 refuse `bits=2` |

Paire maîtresse : **LLVQ − IQ2_XXS = +16,20 pp, IC95 [+12,64 ; +19,72]**, SE
1,81, bootstrap apparié stratifié par matière, 10 000 tirages, graine
`0xb0075eed`, sur les 2 280 mêmes questions.

---

## §2 — Ce qu'un changement de NOYAU invalide, et ce qu'il n'invalide pas

C'est la section qui rend le protocole réutilisable à bas coût.

| colonne | bouge avec un noyau ? | à re-mesurer |
|---|---|---|
| **débit** de notre bras | ✅ OUI — c'est l'objet de la mesure | `fusedrun`, médiane 5 rounds, les DEUX formulations (servi + tête identique), config publiée `ROT_SHARE=0 FUSE=0` sauf décision contraire de l'opérateur |
| **qualité** (ppl, MMLU) de notre bras | ❌ NON si l'artefact scellé est inchangé — les poids sont les mêmes octets | rien ; contrôle : mêmes tokens gloutons, divergence au token 89 |
| **b/param** de notre bras | ❌ NON si le layout VRAM est inchangé ; ✅ si le layout change | `rtbits` si layout modifié |
| les bras **tiers** | ❌ NON — rien chez eux ne dépend de notre noyau | rien ; leurs lignes de la table §1.5 se recopient |
| les **témoins** f16 | ❌ NON sauf changement de version de moteur ou de carte | vérifier que la version d'image est celle de §1.3, sinon re-mesurer le témoin |

⇒ **Une ré-exécution nominale après amélioration du noyau = UN job** (`fusedrun`
deux configs, ~0,3-0,6 $) **+ les contrôles du §3.** La table complète n'est à
refaire que si le format, l'artefact ou une version de moteur change.

🚨 Si le changement de noyau touche le **décodage** (pas seulement le
lancement) : ajouter le contrôle pick-level du §3.4 — l'agrégat peut rester
stable pendant que les réponses changent (leçon n°6 : 95,79 % d'accord seulement
entre les noyaux de déquantisation Metal/CUDA d'IQ2_XXS, pour un Δ agrégé de
0,52 pp).

---

## §3 — Les gates, dans l'ordre, avec leur « si rouge »

| gate | condition | si rouge |
|---|---|---|
| **G1 — vivacité** | tout processus local surveillé par **CPU%** (pas présence, pas existence de fichier) ; stdin `< /dev/null` ; sortie non tamponnée | tuer, diagnostiquer, relancer — leçon n°13 : 6 h 41 perdues sur un stdin |
| **G2 — reconnaissance** | toute image/API jamais utilisée reçoit un job de reco (~0,05 $) : binaires localisés, sha256 des artefacts après transfert | pas de job de mesure tant que la reco n'a pas rendu — leçon n°14 : les 3 jobs sans reco sont les 3 morts de la vague |
| **G3 — instrument** | tout scorer neuf reproduit le f16 dans [70,3 ; 70,9] micro, et rend **micro ET macro étiquetées** | l'instrument est cassé ; aucun bras quantifié n'est lu |
| **G4 — identité** | empreintes de tokens identiques (§1.1) ; artefacts par sha256 quand un fichier traverse des machines | la machine ne participe pas |
| **G5 — comptabilité** | b/param sur le dénominateur **4 022 468 096** ; tout autre compte se justifie contre les 4 instruments | le chiffre ne se publie pas — leçon n°7 |
| **G6 — génération** | tout bras quantifié NOUVEAU produit du texte fluide sur `"The capital of France is"` avant d'être scoré | le bras est « ne tourne pas » : sa qualité est ∅, seuls mémoire et refus datés survivent — leçon n°10, le cas GPTQ |
| **G7 — discrimination** | tout test de départage doit pouvoir rendre des verdicts DIFFÉRENTS selon l'hypothèse vraie, et sa lecture est posée avant | le test ne se lance pas — leçon n°11 : « charabia » ne distinguait rien |

## §4 — Les seuils de lecture et les phrases interdites

- **Différence MMLU entre deux bras calibrés séparément** : non résolue sous
  **~6 pp** (2·σ, σ = 2,92 pp mesuré sur 3 graines au 4B). En deçà : « les
  données sont muettes » — et rien d'autre.
- **Différence à fichier constant** : la barre est l'IC apparié (~±0,43 pp
  MMLU ; ±0,12 % ppl).
- 🚨 **Interdites, en toute circonstance** :
  1. toute division de × entre piles ou entre cartes — propriété du matériel,
     démontrée deux fois (Planes14 L40S/A100 ; IQ2_XXS ×2,647/×3,688 à sha256
     identiques) ;
  2. un tok/s brut de notre bras sans son compagnon à tête identique ;
  3. une perplexité seule comme preuve de qualité (§3ter : le 2 bits abîme le
     raisonnement, la ppl mesure la restitution) ;
  4. « les deux systèmes répondent la même chose » déduit d'un agrégat stable —
     l'agrégat peut tenir à 0,5 pp avec 4 % de réponses changées ;
  5. « nous battons le 2 bits » sur un bras plancher (GPTQ) — l'état de l'art
     est QTIP, toujours non mesuré en qualité.

## §5 — Procédure d'écart

Ce document ne s'édite jamais. Chaque exécution qui dévie écrit
`proofs/protocole-piles-isolees-v2-ECARTS-<date>.md` : l'écart, sa raison, son
effet sur les conclusions, **à chaud**. Un écart non consigné à chaud se
réécrit en voyant les nombres.

## §6 — Coûts de référence (mesurés, vague v1)

| opération | coût mesuré |
|---|---|
| scorer MMLU un bras dans vLLM (2 280×4 prompts, L40S) | ~0,20 $ |
| gate à deux bras (f16 + AWQ) | 0,39 $ |
| quantification GPTQ 2 bits 4B (gptqmodel, L40S) | 0,23 $ |
| chaîne IQ2 complète sur Mac (imatrix 3 min + quantize 1 min 42) | 0 $ |
| débit llama.cpp 2 bras sur L40S | 0,04 $ |
| MMLU IQ2 sur CUDA (prompts expédiés) | 0,06 $ |
| job de reconnaissance | 0,02-0,05 $ |
| **vague v1 complète** | **1,29 $** |

## §7 — Ce que la v2 laisse ouvert, en le nommant

1. **QTIP en qualité** — l'état de l'art 2 bits, pas d'artefact Qwen3 ;
   *estimé* 10-20 $ + risque d'architecture. La case reste vide et déclarée.
2. **ppl côté vLLM** — volet jamais câblé ; aucune ppl vLLM ne se publie.
3. **Variance de calibration des bras tiers** — un tirage chacun ; le σ de
   2,92 pp leur est appliqué par analogie déclarée.
4. **Le banc métier d'extraction documentaire** (§6 Phase 5, jamais fait) — la
   divergence pick-level mesurée le 31 est son meilleur argument à ce jour :
   4 % de réponses changées invisibles à l'agrégat.
