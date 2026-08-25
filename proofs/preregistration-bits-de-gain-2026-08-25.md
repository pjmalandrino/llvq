# Pré-enregistrement — le partage des 48 bits entre direction et gain

**Écrit, commité et tamponné AVANT le premier bras.** Item **§3.1** de
[`docs/BACKLOG.md`](../docs/BACKLOG.md) (« A/B des bits de gain, à débit
constant »). Go de l'opérateur donné le 2026-08-25 pour l'étage 1 seulement.
Coût annoncé : **0 $** — Mac M3 Max, aucune carte louée.

---

## §0 — Ce que ce gate peut et ne peut pas conclure

1. **Il ne peut RIEN adopter.** C'est un gate à profondeur sur Qwen3-0.6B :
   il sert à ne pas payer douze heures de machine sur un bras qui casse le
   modèle. Un bras vert n'est pas un bras meilleur, c'est un bras qui a le
   droit d'être mesuré au 4B.
2. **Il ne dit rien sur MMLU.** Il produit une perplexité. Le §3ter du dossier
   établit depuis le 2026-08-02 que le 2 bits abîme le **raisonnement** bien
   plus que la **restitution**, et que c'est la restitution qu'un corpus de
   perplexité mesure surtout. L'hypothèse testée par cette expérience vit
   précisément dans l'écart entre les deux — donc **exiger un gain de
   perplexité au gate tuerait l'hypothèse qu'on cherche à tester**. Le gate ne
   filtre que la destruction.
3. **Il ne dit rien sur la vitesse ni sur la VRAM.** L'expérience est
   iso-débit par construction (§2) : les trois bras écrivent 48 bits par bloc.
   Et deux des trois ne se serviraient pas — le noyau fusé gèle le champ de
   gain à un bit (`llvq-artifact/src/runtime.rs:416-418`, assertion
   `runtime.rs:437-441` ; idem `e1c.rs:346`).
4. **Il ne dit rien d'une autre taille de modèle.** Qwen3-0.6B a
   `hidden = 1024` et `intermediate = 3072` ; le 4B publié a 2560 et 9728.
   Aucun verdict de ce gate ne se transporte.
5. **Il ne dit rien de ce qu'un autre tirage de calibration aurait donné.**
   Un seul tirage, partagé par les trois bras (§8, C4).

---

## §1 — La question, et pourquoi le choix actuel n'est fondé par rien

La configuration servie est `leech1c12` : boule Λ₂₄(12), 47 bits d'index +
1 bit de gain = 48 bits par bloc de 24 poids, soit **2,000 b/poids de code
exactement**.

**Ce choix n'a jamais été fondé par une mesure au niveau modèle, à aucune
taille.** Sa justification écrite est un argument de **débit** —
`llvq-quant/src/quantizer.rs:298-302` : « `cap = 12` drops the index from 48 to
47 bits, which pays for a gain bit at the same total rate — the paper's Table 8
best configuration » — et la Table 8 du papier amont est une table de
**distorsion sur source gaussienne**, pas un résultat LLM.

Or le papier donne **trois réponses différentes selon le protocole** :

| protocole | optimum annoncé |
|---|---|
| Table 8 — source gaussienne | **1 bit** (notre choix) |
| Table 6 — perplexité LLM, Qwen3-4B, 2 bits, sans FT | **2 bits** : wiki 15,54 contre 17,05 |
| Table 6 — MMLU LLM, même ligne | **0 bit** : **60,7** contre 59,3 |
| Annexe I — sous Spherical GPTQ | **0 bit** |

Notre configuration **n'a aucun équivalent dans la Table 6** : elle n'existe
que dans la table de distorsion. Et le 60,7 de leur configuration 0 bit est le
meilleur MMLU 2 bits sans fine-tuning de toute leur table — devant QTIP (57,4)
— pendant que nous rendons **55,59**.

**La question de ce gate** : l'un des deux autres partages du même budget de
48 bits survit-il à profondeur sur un petit modèle, et mérite-t-il donc d'être
mesuré au 4B où la question de MMLU peut être posée ?

⚠️ **Ce que ce document ne fait pas** : il ne teste pas les bits de gain
*toutes choses égales par ailleurs*. C'est arithmétiquement impossible à débit
constant — voir §8, C1.

---

## §2 — La prémisse d'iso-débit, vérifiée avant d'écrire ce document

`Codebook::block_bits` (`llvq-llm/src/calib.rs:228-251`) vaut
`index_bits(max_shell) + gain_bits`, et `index_bits(cap)`
(`llvq-quant/src/quantizer.rs:572-576`) vaut `bit_length(N(cap))` avec
`N(cap) = Σ_{m=2..cap} |Shell(m)|`.

| bras | boule | N(cap) | index | gain | **bits/bloc** | classes |
|---|---|---|---|---|---|---|
| `leech0c13` | Λ₂₄(13) | 280 974 212 784 720 | 48 | 0 | **48** | 383 |
| `leech1c12` ← servi | Λ₂₄(12) | 111 043 117 458 000 | 47 | 1 | **48** | 301 |
| `leech2c11` | Λ₂₄(11) | 40 556 880 458 640 | 46 | 2 | **48** | 222 |

*Provenance* : les trois cardinalités sont **calculées** et recoupées de deux
façons indépendantes — par la série thêta de Λ₂₄ et contre la constante
`N_SHELL_13_CUMULATIVE = 280 974 212 784 720` (`llvq-core/src/leech.rs:54`),
elle-même verrouillée par le test `classes_reproduce_theta_series`. Les trois
largeurs sont **imprimées par le binaire lui-même** au parsing, avant tout
chargement de modèle.

🚨 **La coquille 11 n'était couverte par aucun test du workspace** avant ce
document : aucun `with_shell_cap(_, 11)`, et `index_width_follows_the_shell_cap`
n'épinglait que 12 et 13. Trois assertions ont été ajoutées le 2026-08-25 et
prouvées létales par mutation — largeur d'index au cap 11, round-trip
bit-pour-bit à cap 11 + 2 bits de gain sur le chemin GPTQ complet, et une ligne
cap 11 dans le balayage de format. **Si `index_bits(11) ≠ 46`, ce
pré-enregistrement est nul et non avenu** (§7).

---

## §3 — Le protocole, figé ici

Qwen3-0.6B, **28 blocs — pleine profondeur**, calibration wikitext-2 *train*
64 × 2048 = 131 072 tokens, rotation d'entrée activée, échelles de groupe
désactivées (`nogs`), évaluation wikitext-2 *test* 12 fenêtres de 2048, dtype
f32, backend Metal, 16 threads d'encodeur.

**Une seule chose change d'un bras à l'autre : le 7ᵉ argument positionnel.**

```
target/release/smoke 64 2048 12 2048 metal nogs leech0c13 999 rot > ~/llvq-nuit-b/gain-ab-0c13.log 2>&1
target/release/smoke 64 2048 12 2048 metal nogs leech1c12 999 rot > ~/llvq-nuit-b/gain-ab-1c12.log 2>&1
target/release/smoke 64 2048 12 2048 metal nogs leech2c11 999 rot > ~/llvq-nuit-b/gain-ab-2c11.log 2>&1
```

**Deux prérequis, posés ici parce qu'ils changent ce que le chronomètre
mesure.** (a) Qwen3-0.6B est **absent** de la machine et doit être rapatrié
avant le premier bras, sinon celui-ci paie le téléchargement. (b) Le binaire
doit être reconstruit **avec `fast-linalg`** : celui présent ne porte aucun
symbole `faer`, et le gate du 2026-08-07 a tourné sans la feature — sa
factorisation y pesait **43,3 % du temps**. Résultat bit-identique dans les
deux cas (`both_factorizations_agree`), mais ~14 min par bras.

🚨 **Le témoin est ré-ancré, et c'est un défaut du protocole hérité.** Le
script `m3-queue.sh` du 2026-08-07 passait `leech`, que le parseur résout en
boule 13 + 1 bit de gain = **49 bits/bloc, 2,2068 b/poids** — donc *pas* le
codebook servi. Le témoin de ce gate est `leech1c12`, à 48 bits.

---

## §4 — Les quatre contrôles à passer AVANT de lire la moindre perplexité

Si l'un échoue, **aucun verdict du §5 n'est rendu**.

1. **Iso-débit.** Les trois bras doivent imprimer le **même** débit effectif,
   et il doit valoir **2,1656 b/poids** — *calculé* d'avance par
   `Cost::bits_per_weight` sur les formes du 0.6B (196 matrices,
   440 401 920 poids, 18 149 376 blocs, 4 816 896 poids de queue,
   344 064 lignes) à 48 bits/bloc. Repère de validation de la formule : le bras
   `leech` du 08-07, à 49 bits, imprimait 2,2068. Un écart signifie qu'un bras
   n'est pas au débit annoncé, et l'expérience cesse d'être iso-débit.
2. **Configuration résolue.** Chaque log doit porter sa ligne de codebook avec
   le bon triplet : 0/13/**383 classes**, 1/12/**301**, 2/11/**222**, et
   « 48 bits/block » aux trois.
3. **Baseline.** La perplexité FP32 de référence ne dépend pas du codebook :
   elle doit valoir **19,5038** aux trois bras. Si elle bouge, c'est le harnais
   qui a bougé, pas le codebook — et le gate est nul.
4. **`fast-linalg` effectif.** L'avertissement « compilé SANS `fast-linalg` »
   doit être **absent** des trois logs, et la factorisation doit être retombée
   sous ~5 % du profil par phase. S'il reste à ~43 %, le build n'a pas pris la
   feature et les durées annoncées sont fausses.

---

## §5 — Les seuils, posés avant le premier bras

### 5.1 — Étage 1, ce gate : un filtre de destruction, pas un classement

Soit `a` la perplexité du témoin `leech1c12` et `b` celle d'un candidat, à
28 blocs.

| verdict | condition | conséquence |
|---|---|---|
| **VERT** | `b ≤ 1,02 · a` | le bras gagne le droit d'un run 4B — rien de plus |
| **ROUGE** | `b > 1,02 · a` | le bras est **mort** ; aucun run 4B, et on l'écrit |

Le facteur 1,02 est la règle de maison déjà utilisée
(`~/llvq-nuit-b/m3-queue.sh:40`), celle qui a tué le design C à ×1,99 et
économisé un run de quatre heures. Elle n'est **pas** réinventée pour cette
expérience.

⚠️ **Ce seuil ne fonde aucune adoption.** Un candidat à `b = 1,015·a` est vert
et reste, à ce stade, moins bon que le témoin en perplexité — c'est voulu :
voir §0.2.

### 5.2 — Étage 2, au 4B : la règle d'adoption, posée maintenant

Posée ici pour qu'aucune liberté ne reste après avoir vu l'étage 1. **L'étage 2
exige son propre go de l'opérateur** (~12 h de machine, 0 $) et sera couvert
par son propre document si le protocole doit bouger.

- **Grandeur qui JUGE** : MMLU micro, **apparié** par `bin/mmlupair`, empreinte
  de tokens identique des deux côtés (règle du §2 de `CLAUDE.md`).
- **Adoption** : le candidat doit rendre **≥ +2,0 pp** contre `leech1c12`
  **et** son IC95 apparié doit **exclure zéro**. Les deux, pas l'un ou l'autre.
- **Pourquoi +2,0** : la SE appariée mesurée entre modèles différents vaut
  0,79 à 1,44 pp, donc l'IC est le test ; le plancher de +2,0 est un critère de
  **matérialité**, parce qu'adopter exigerait de lever le gel du champ de gain
  dans le chemin servi — huit assertions et quatre shaders — un coût que
  personne n'a chiffré. Sous +2,0 : on publie, on ne change rien.
- **Grandeur qui DOCUMENTE** : la perplexité, appariée fenêtre par fenêtre.
  Elle ne juge pas.
- **Si les deux métriques se contredisent** — cas explicitement attendu, le
  papier lui-même les classe en sens inverse — **MMLU décide**, parce que le
  déficit ouvert est un déficit de capacités. Exception : si la perplexité se
  dégrade de plus de 5 % pendant que MMLU gagne ≥ 2 pp, on publie les deux et
  **on n'adopte rien** sans un run de plus.

### 5.3 — La branche du résultat trop bon

Si un candidat rend **≥ +5 pp** de MMLU, **chercher l'erreur avant d'en faire
un titre**. L'écart 0↔2 bits du papier ne vaut que 1,4 pp ; un basculement de
5 pp obtenu en déplaçant un bit de gain serait surprenant, pas heureux.
Suspects à vérifier dans cet ordre : empreinte de tokens différente entre les
deux bras · un bras scoré sur le mauvais fichier · un débit qui n'est pas
réellement 48 bits/bloc · la queue ou les échelles qui auraient changé de
largeur.

---

## §6 — Les issues, toutes énumérées d'avance

| issue mesurée | conséquence, décidée d'avance |
|---|---|
| Un contrôle du §4 échoue | aucun verdict n'est rendu ; on corrige et on relance les **trois** bras |
| `index_bits(11) ≠ 46` | ce document est nul ; l'expérience est reconçue (§7) |
| Les deux candidats ROUGES | **le choix de 1 bit est confirmé par la mesure** et cesse d'être hérité — résultat publiable, suspect fermé, coût total ~1 h |
| Un candidat VERT, l'autre ROUGE | run 4B du seul bras vert, sur go séparé |
| Les deux VERTS | run 4B des deux, sur go séparé — ~24 h de machine |
| Un candidat vert **et meilleur** en ppl à 28 blocs | ne change **rien** au §5.1 : c'est toujours un droit d'entrée au 4B, pas une adoption |
| Écart entre bras sous 0,5 % | **non résolu**, jamais « égaux » : la marge est sous ce qu'un tirage de calibration déplace (§8, C4). Les bras passent quand même s'ils sont sous 1,02 |
| Un bras échoue en cours (panique, OOM) | on ne substitue rien : le bras est **non rendu**, et l'écart est écrit au §7bis |
| Étage 2 : candidat ≥ +2,0 pp, IC excluant zéro | **le suspect est réel** ; le gel à 1 bit du chemin servi devient une dette à chiffrer |
| Étage 2 : candidat dans (0 ; +2,0) pp ou IC contenant zéro | le suspect est **borné, pas confirmé** ; on publie, la configuration servie ne bouge pas |
| Étage 2 : candidat ≤ 0 pp | le choix de 1 bit est confirmé au 4B — c'est un résultat, et c'est l'issue la plus probable a priori |
| Étage 2 : métriques contradictoires | on publie les deux, on n'adopte rien (§5.2) |
| Étage 2 : candidat ≥ +5 pp | §5.3 — on cherche l'erreur d'abord |

---

## §7 — Ce qui invaliderait ce pré-enregistrement

- `index_bits(11) ≠ 46`, ou `index_bits(13) ≠ 48`, ou `index_bits(12) ≠ 47` —
  la prémisse d'iso-débit du §2 tomberait, et avec elle tout le dessin de
  l'expérience.
- Les trois bras n'imprimant pas le même débit effectif (§4.1).
- Une baseline FP32 qui diffère d'un bras à l'autre (§4.3).
- La découverte que `nogs`, `rot` ou la calibration ne sont pas identiques
  entre bras.

---

## §7bis — Écarts au protocole, journal tenu à chaud

*Chaque entorse s'écrit ici le jour où elle est commise, avec sa raison et son
coût — règle du 2026-08-10.*

**Vide à la signature.**

---

## §8 — Réserves et confondants, déclarés avant la mesure

**C1 — Ce n'est PAS un A/B « bits de gain », et le titre du document le dit.**
Les trois bras portent trois boules différentes (13, 12, 11). Le codebook de
**direction** change en même temps que celui de **gain** — c'est
arithmétiquement inévitable à débit constant, puisque c'est précisément le
partage d'un budget fixe. La quantité produite est donc « quel partage des
48 bits », jamais « combien de bits de gain toutes choses égales ». Non
corrigé, non corrigeable.

**C2 — Les temps d'encodage ne seront pas comparables** : 383, 301 et 222
classes. Le sens est connu (moins de classes, plus rapide), l'amplitude non.
Sans effet sur la quantité produite ; à ne pas lire comme un résultat.

**C3 — Dépendance fraîche.** La coquille 11 n'avait aucune couverture avant le
2026-08-25 ; les assertions ajoutées le même jour sont la seule chose qui
garantit ce bras. Déclaré parce qu'un test écrit le jour du code qu'il garde
partage ses angles morts (§5 de `CLAUDE.md`).

**C4 — Un seul tirage de calibration.** F5 a mesuré σ = 5,2 % entre graines au
4B. Ici les trois bras partagent la graine, le corpus et l'ordre : la
comparaison est **contrôlée en interne**, et l'effet mesuré est déterministe.
Mais rien n'établit que son **signe** serait le même à un autre tirage, et
emprunter le σ de F5 pour le borner serait fabriquer un nombre.

**C5 — Le gate est un proxy, et ce dossier a deux précédents où un proxy local
meilleur prédisait une composition pire** (`group_scales`, design C). Les deux
**relâchaient** la rigidité de norme ; changer la résolution du gain la
conserve (`retract_to_level` reste vrai). Que le motif transfère d'un
changement de *rigidité* à un changement de *résolution* n'est écrit nulle
part. C'est la raison pour laquelle le gate ne peut que tuer.

**C6 — Le chemin servi ne saurait porter deux des trois bras.** Si l'étage 2
élit `leech0c13` ou `leech2c11`, la suite n'est pas « refaire le 4B » mais
« refaire le 4B **et** rouvrir le layout runtime ». Ce coût n'est chiffré nulle
part et il est hors périmètre ici.

**C7 — Aucune comptabilité mémoire ne sera disponible pour les deux nouveaux
bras** : `bin/rtbits` asserte `gain_bits == 1`
(`llvq-bench/src/bin/rtbits.rs:555-560`). Sans effet sur la qualité, bloquant
pour toute phrase en b/param modèle entier.

---

## §9 — Coût, sorties, et où elles survivent

- **Coût : 0 $.** Mac M3 Max, aucune carte louée. Aucun plafond budgétaire
  n'est engagé et aucun n'est demandé.
- **Durée estimée** : ~20 min par bras avec `fast-linalg`, soit ~1 h pour les
  trois. *Estimé*, pas mesuré : le seul chiffre mesuré est 34 min/bras sur un
  binaire **sans** la feature et sur la boule 13.
- **Sorties** : les trois logs **bruts** vont dans le dépôt, pas seulement une
  synthèse — règle du §7 de `CLAUDE.md` (« un journal de synthèse est une perte
  irréversible dès que le canal de rétention expire »). Journal éditorialisé
  dans `docs/mesures/`, sur la forme de
  `m3-gate-design-c-2026-08-07.txt`.

---

## §10 — Divulgation datée, à la signature

Connu au moment de signer, et rien d'autre :

- `leech1c12` rend **16,9415** de perplexité f16 et **55,59** de MMLU micro au
  4B ; la baseline FP32 du 0.6B vaut **19,5038**.
- Le gate du 2026-08-07 a rendu 35,9806 pour son bras A (`leech`, 49 bits) et
  71,4249 pour le design C, verdict rouge. **Ce 35,9806 n'est pas réutilisable
  ici** : autre codebook, autre débit, et le binaire a changé trois fois depuis.
- Aucune quantification n'a été lancée à `leech0c13` ni à `leech2c11`, à aucune
  taille, dans toute l'histoire du projet.
- Les tests de la coquille 11 ont été écrits **avant** ce document et prouvés
  létaux par mutation.
- Prédiction de l'auteur, écrite pour être opposable : **les deux candidats
  passent le gate** (le mécanisme ne relâche aucune rigidité), et au 4B **aucun
  ne dépasse +2,0 pp** — l'écart 0↔2 du papier ne vaut que 1,4 pp, et notre
  déficit face à leur configuration 0 bit est de 5,1 pp. Si cette prédiction se
  vérifie, le suspect n°3 du déficit MMLU se ferme et il reste la composition
  du corpus et la compensation bas-rang.
