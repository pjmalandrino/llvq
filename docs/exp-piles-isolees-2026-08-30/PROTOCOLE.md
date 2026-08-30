# Protocole de test — piles isolées (2026-08-30)

> Écrit **avant** toute mesure. Chaque section dit ce qu'on relève, avec quel
> instrument, et surtout **ce que le nombre a le droit de conclure**.

## 0. Le choix de la métrique primaire, et il n'est pas le choix habituel

**MMLU est le primaire. La perplexité est le secondaire.** C'est l'inverse de
l'usage, et ça découle du design.

1. **La perplexité ne traverse pas les piles.** Elle dépend de conventions de
   fenêtrage propres à chaque moteur — nos 12 fenêtres de 4096 ne sont pas les
   chunks glissants de `llama-perplexity`. Son **niveau** n'a de sens qu'à
   l'intérieur d'une machine ; seul son rapport au témoin f16 en sort.
2. **MMLU traverse.** Il ne dépend que du tokenizer et du logprob de 4 tokens de
   réponse. Deux moteurs qui partagent le tokenizer rendent des MMLU comparables
   **en niveau**.
3. **Et c'est la métrique qui voit les dégâts.** Le §3ter l'établit depuis le
   2026-08-02 : le 2 bits abîme le **raisonnement** bien plus que la
   **restitution**, et c'est la restitution que mesure surtout un corpus de
   perplexité. Au 4B, la ppl bouge de ×1,384 pendant que MMLU perd **14,73 pp**.
   🚨 Règle du dossier, rappelée ici : *ne jamais présenter la perplexité seule
   comme preuve de qualité.*

## 0bis. L'axe backend — CUDA et Metal sont deux piles

🚨 **Un backend est une pile, au même titre qu'un moteur.** Le dossier a déjà
mesuré que les × ne se divisent même pas entre **deux cartes CUDA** : sur A100
aucun bras à décodage ne bat FP16 (`Planes14` 0,79× contre 2,14× sur L40S), et le
lot G a tranché que le ×1,78 **est** le rapport d'horloges 2 520/1 410 MHz.
Entre CUDA et Metal, l'interdiction est *a fortiori*.

⇒ **Chaque backend a son propre témoin f16**, et seuls les rapports se comparent.

### Ce que Metal peut porter, et ce qu'il ne peut pas

| | Metal |
|---|---|
| **qualité** (P1, P2) | ✅ complète — et c'est un **contrôle inter-backend** |
| **mémoire** (P3) | ✅ complète |
| **débit LLVQ** (P4) | ❌ **impossible** — il n'existe pas de `fused_metal.rs` |
| **débit des bras tiers** | ✅ llama.cpp et MLX y sont servis nativement |

🚨 **Le piège, et il est structurel.** `llvq-llm` ne porte que `fused.rs` et
`fused_cuda.rs`. Sur Metal, LLVQ n'existe qu'en micro-banc : `bin/thesis` mesure
**252 projections sur un token**, quand `mlx_lm.generate` fait tourner un
**modèle entier sur 256 tokens**. **Les comparer serait la faute « deux
dénominateurs » du §7.** Sur Metal, LLVQ ne joue qu'en qualité et en micro-banc.

### C4 — Le contrôle inter-backend, gratuit

Le **même artefact scellé** scoré sur Metal et sur CUDA doit rendre **la même
qualité** : ce sont les mêmes poids, décodés par deux implémentations
indépendantes. Historiquement vérifié sur la baseline — **70,42 (Metal) → 70,32
(CUDA)**, soit 0,08 σ, à travers un changement de backend, de carte et de dtype.

⇒ Si les deux backends divergent en qualité sur le même fichier, **c'est un bug
de portage**, pas un résultat. Ce contrôle ne coûte rien et il attrape la classe
d'erreur la plus coûteuse du projet (cf. §5 du `CLAUDE.md`, la quatrième prise :
*une transcription porte les gardes de son original sans porter les hypothèses
qui les rendaient suffisantes*).

### C5 — L'équivalent Metal de F1 🚨 **il n'existe pas, et c'est une dette nommée**

Sur CUDA, **F1** a établi que notre témoin FP16 maison vaut **1,024×** (banc
2 bras) et **1,015×** (banc 5 bras) de cuBLAS sur L40S — tous deux ≤ 1,05. C'est
ce qui fait tenir *tous* les rapports « vs FP16 » publiés.

Côté Metal, `docs/fiche-4b.md:438` nomme l'angle et le déclare non adressé :
*« ce baseline n'a jamais été confronté à MPS, MLX ou Accelerate : le 2,07× est
un rapport contre un noyau écrit par le même auteur. C'est l'angle hostile
restant, non adressé. »*

⇒ **Tant que C5 n'est pas fait, aucun × Metal ne se publie.** Le confronter à
MPS / MLX / Accelerate coûte **0 $** et vaut plus que n'importe quel bras
supplémentaire de cette expérience.

## 1. P1 — MMLU micro *(primaire)*

- **Fixture** : 2 280 questions, 57 matières, agrégation **micro**
  (`Σright/Σtotal`), jamais macro — cf. le 🚨 du §3ter : l'échange macro/micro
  vaut ~1 pp et frappe plus fort le bras quantifié que le témoin.
- **Sortie obligatoire** : **dump par question avec `qhash`**, pas un taux
  agrégé. C'est ce qui rend les paires formables après coup — `mmlupair` en a
  fait neuf pour 0 $ le 2026-08-17, sur des dumps datant du 2026-08-10.
- **Statistique** : bootstrap **apparié stratifié par matière**, 10 000 tirages,
  graine `0xb0075eed`, plus McNemar exact. Le ± d'échantillonnage d'un bras seul
  n'est **pas** la barre d'une différence.
- **Cut gratuit et le plus informatif** : MMLU **par groupe de matières**
  (raisonnement vs restitution). Le dump le porte déjà — c'est un `awk`, pas un
  run. C'est là que le mécanisme se voit : `abstract_algebra` et `accounting`
  tombent à **25 %, le hasard**, pendant qu'histoire et droit tiennent au-dessus
  de 80 %.

## 2. P2 — Perplexité *(secondaire)*

- wikitext-2 test, ctx 4096, 12 fenêtres, f16.
- **NLL par fenêtre conservées** — `bin/ppl` les imprime à 9 décimales **sur
  stderr**, donc perdues sans `2>`. Le §7 a payé cette leçon trois fois : un
  journal de synthèse est une perte irréversible.
- **Ne se publie qu'en rapport** au témoin f16 de sa propre machine.
- Sert de **pont vers la littérature** (tout le monde publie de la ppl), pas de
  preuve de qualité.

## 3. P3 — Mémoire

- **b/param modèle entier, embedding compris.** Jamais un b/poids de projections
  contre un b/param de modèle entier : c'est la faute grave de l'errata du lot A.
- Comparable **directement** entre machines : c'est un compte d'octets.
- 🔎 Note de provenance qui joue **contre** nous : pour un GGUF,
  `octets du fichier ÷ params` **est** la valeur, mesurée. Pour notre bras,
  l'embedding est *modélisé* à 8,5 b/param. Étiqueter les deux différemment.

## 4. P4 — Débit

- Médiane sur **5 rounds**, 1 génération jetée, **avec plage** — jamais un point
  unique. Le §7 : les millisecondes dérivent d'une invocation à l'autre là où les
  octets se reproduisent au chiffre.
- **Toujours en rapport au témoin f16 de la même machine.** Un tok/s nu ne se
  publie pas.

## 5. Les contrôles — sans eux rien de ce qui précède ne vaut

### C1 — Le témoin f16 de chaque machine doit reproduire les valeurs connues

C'est le contrôle le plus important et le moins cher, et c'est l'analogue
inter-moteurs du « contrôle identité rend ×1,000 exact » du §5.

Au 4B, le f16 doit rendre **MMLU 70,32 ± 1,28** et **ppl 12,2369**.

🚨 **Si le témoin f16 d'une machine ne reproduit pas ces valeurs, cette machine
est cassée et son bras quantifié ne veut rien dire.** On ne lit aucun bras
quantifié avant que son témoin soit vert. Un écart de témoin est une panne de
harnais, pas un résultat.

### C2 — L'empreinte de tokens, identique partout

`65dcd53655e8bfa5` (MMLU) et `3f1baca9033bf251` (ppl). C'est ce qui **prouve**
« mêmes données » au lieu de le déclarer. Deux machines qui affichent la même
empreinte ont lu le même texte, token pour token. Une machine qui n'imprime pas
son empreinte ne participe pas à la comparaison.

### C3 — Les asymétries, déclarées d'avance et dans quel sens elles jouent

| asymétrie | sens |
|---|---|
| notre bras dense recopie 778 Mo de vocabulaire par token (`Head::project` → `broadcast_matmul`) | **contre nous** — il est au dénominateur de nos rapports, donc nous sous-estimons notre avance |
| le noyau vLLM à 2 bits n'est **pas** Marlin (4 bits seulement) mais le chemin ExLlamaV2 | **pour nous** — le bras GPTQ est handicapé en vitesse |
| `IQ2_XXS` est à **2,06 bpw** contre nos 2,0702 | quasi neutre — la comparaison de débit la plus serrée du dossier |
| `Q2_K` est à ~2,6–3,0 bpw | 🚨 **autre classe de débit — ne pas l'utiliser**, ce serait la faute « deux dénominateurs » |

## 6. La barre de lecture, posée d'avance

- **SE appariée MMLU** : **0,43 pp** à fichier constant ; **0,79 à 1,44 pp**
  entre modèles différents (mesuré le 2026-08-15, contre les 0,4–0,6 pp
  *estimés* et jamais calculés qui circulaient avant). ⇒ **Un écart MMLU sous
  ~1,5 pp entre deux bras n'est pas résolu.** Ne pas le narrer.
- **σ de calibration** : F5 rend **5,2 %** de ppl à la taille publiée, pas les
  0,7 % du lot B (qui valaient pour 3 blocs de Qwen3-0.6B et sont faux d'un
  facteur ~7 ici).

  🚨 **Et il s'applique à cette expérience, contrairement aux A/B à fichier
  constant.** Les bras **GPTQ** et **IQ2_XXS** sont **calibrés** — GPTQ sur C4,
  IQ2 par matrice d'importance — donc chacun est **UN tirage** de fenêtres de
  calibration, exactement comme notre propre artefact. Conséquence :

  - le **niveau absolu** de chaque bras calibré est celui d'un tirage, non privilégié ;
  - une différence entre **deux bras calibrés séparément** mêle *méthode* et *tirage* ;
  - pour un écart de 14 pp c'est sans effet ; **pour un écart de 3 pp entre deux
    2 bits, c'est décisif** — et c'est précisément l'ordre de grandeur attendu.

  ⇒ **Aucun classement entre deux bras 2 bits ne se publie sur un seul tirage
  chacun.** Soit l'écart est large devant 5,2 %, soit il faut plusieurs graines,
  soit on écrit que les données sont muettes.

## 7. Les gates, dans l'ordre

| gate | condition | si rouge |
|---|---|---|
| **G-a** | le témoin f16 de chaque machine reproduit 70,32 / 12,2369 | la machine est cassée — on ne lit pas son bras quantifié |
| **G-b** | l'empreinte de tokens est identique sur toutes les machines | la machine ne participe pas à la comparaison |
| **G-c** | le bras quantifié rend un b/param cohérent avec ses octets | erreur de comptabilité — cf. l'errata du lot A |
| **G-d** | le même artefact scellé rend la **même qualité** sur Metal et sur CUDA | bug de portage, pas un résultat (C4) |
| **G-e** | le témoin FP16 **Metal** a été confronté à MPS / MLX / Accelerate | **aucun × Metal ne se publie** tant que c'est rouge (C5) |

## 8. Ce que cette expérience ne pourra pas dire

- **Aucun rapport de vitesse entre deux machines.** Jamais.
- **Aucun débit LLVQ bout-en-bout sur Metal.** Il n'existe pas de
  `fused_metal.rs` : sur ce backend, LLVQ ne joue qu'en qualité et en micro-banc
  (`bin/thesis`, 252 projections sur un token).
- **Rien sur QTIP**, hors périmètre faute d'artefact Qwen3.
- **Aucune loi d'échelle** : une seule taille (4B).
- **Rien sur le raisonnement en usage réel** : MMLU est un QCM. Le benchmark
  métier d'extraction documentaire réclamé au §6 de la Phase 5 n'a **toujours**
  ni verdict ni date, et cette expérience ne le remplace pas
  (cf. arXiv:2607.08734 : perplexité et exactitude restent stables pendant que
  les réponses individuelles changent).
