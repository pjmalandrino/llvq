# Plan d'actions — trois phases

> Issu de l'audit externe du 2026-08-12 (voir la dernière entrée de
> [`HISTORIQUE.md`](HISTORIQUE.md)). Trois phases **ordonnées par ratio
> information/coût**, chacune avec ses tâches, ses critères écrits d'avance
> et son coût.
>
> 🧭 **2026-08-25 — LE RAF COURANT VIT DÉSORMAIS DANS
> [`BACKLOG.md`](BACKLOG.md).** Ce fichier-ci reste le document de **niveau
> projet** : les trois phases, ce qu'elles décident, leurs critères écrits
> d'avance et leurs coûts. Il ne porte plus la liste de ce qui reste à faire
> au jour le jour — une session qui reprend lit `BACKLOG.md` d'abord, puis ce
> plan pour savoir *pourquoi* un item y figure.
>
> ✅ **2026-08-24 — LE PAPIER EST SOUMIS : ACM TACO, `TACO-2026-428`, au
> commit `e21a8bb`.** La **Phase 1 est soldée** (détail en tête de sa
> section) : le bras **QTIP** est entré au corps (`tab:layouts`,
> `sec:qtip`, lot F2) et le récit d'échelle porte ses barres aux trois
> tailles. ⚠️ **Décision d'opérateur du 2026-08-25 : le dépôt GitHub RESTE
> PUBLIC** — le manuscrit est anonymisé et la page de titre donne les URL à
> l'**éditeur** seul, pour que l'artefact soit localisable sans que le
> manuscrit brise sa propre anonymité. 🕳️ **Toute phrase de ce dossier qui
> dit ou suppose « dépôt privé pendant la revue » est périmée**, y compris
> dans le plan TACO cité juste dessous.
>
> 🗓️ **2026-08-18 — Le chantier « papier » a son plan d'exécution :
> [`plan-taco-2026-08-18.md`](plan-taco-2026-08-18.md)**, issu d'un audit
> complet du dépôt (9 agents, vérifications sur pièces) cadré sur une
> soumission ACM TACO. Il porte le recadrage du claim, les trois décisions
> d'opérateur (option A/B sur la courbe d'échelle, sort du 8B, périmètre
> batch), le RAF ordonné (B1-B4 bloquants, F1-F7 attendus) et le chemin
> critique en 4 jalons (~30-100 $, 18-25 j-h). Ce plan-ci reste le document
> de niveau projet ; le plan TACO est son exécution côté publication.
>
> 🚨 **2026-08-16 — LE TRIPLET PRODUIT EST ARBITRÉ, ET IL FERME L'AXE FORMAT.**
> L'opérateur a tranché les trois cases qui manquaient à
> [`note-produit-2026-08-13.md`](note-produit-2026-08-13.md) : **A2 = 8k**,
> **A4 = marge 5 Go**, **A6 = offload en référence seulement**, unité **32 GiB**
> confirmée. Le §B de la note **fait foi** et donne
> **`b_max` = 3,00 b/poids noyau** pour le barreau 70B dense sur 32 GiB.
>
> **Ce que ce seuil décide, et c'est le résultat le plus structurant du mois** :
> *aucun layout du portefeuille ne le passe*. `Planes14` (le servi) est à
> **+60 %**, `Planes12x` +44,6 %, `e1c12` aligné +42,8 %, `Golay70` **+19,5 %**,
> et le seul qui franchit — E1v, −20,1 % — a son décodeur fermé à 0,25× FP16.
> Combiné au plancher du 2026-08-16 — le format ne se dispute qu'une part du
> temps, `Planes14` en capture déjà l'essentiel, et son décodage ne coûte que
> **~7 %** du temps de trafic —, **il n'y a plus de raison de chercher un
> format** : ni pour la vitesse, ni pour la mémoire.
>
> 🕳️ **Cette phrase disait « tout travail de format plafonné à 4,77×,
> `Planes14` en capturant déjà 2,16× ». LE PLAFOND EST FAUX depuis le
> 2026-08-21, et c'est l'erratum le plus utile du mois.** Le lot F2 a porté
> **QTIP** dans notre propre banc, **un seul processus**, 7 rounds dont 2
> jetés, bras entrelacés : **QTIP 2 bits = 2,246 ms [2,245–2,248]**, 0,91 Go
> lus, 2,0000 b/poids, 405 Go/s, **4,89× FP16**, contre `Planes14` 5,103 ms,
> 2,18 Go, 4,804 b/poids, **2,15×** — soit **r = 2,27× [2,27–2,28]** en faveur
> de QTIP *(mesuré ; comptabilité **b/poids noyau** des deux côtés ; rapport
> encadré par l'extérieur, le banc n'imprimant pas `r` par round)*.
> **`t(QTIP) = 2,246 ms < t(nullk) = 2,306 ms`** : un noyau réel finit les
> mêmes 252 projections **en lisant 0,91 Go** plus vite que notre passe qui
> n'en lit **aucun** (séparation 2,7 % contre une résolution de 0,72 %), et
> `f = 61,1 %` contre un plafond **pré-enregistré** de 59,6 % — erratum
> consigné au journal, le `.ots` interdisant l'édition du prereg
> ([`mesures/f2-p3-qtip-banc-2026-08-21.txt`](mesures/f2-p3-qtip-banc-2026-08-21.txt)).
> ⇒ **`nullk` n'est pas un plancher machine : c'est le plancher de NOTRE
> GÉOMÉTRIE DE LANCEMENT** — un warp par ligne de sortie, 252 lancements —
> et **un noyau autrement formé passe dessous**. Le papier porte déjà la
> correction (`paper/main.tex`, `paper/sections/layouts.tex`) ; les documents
> internes la reçoivent ici.
>
> ⚠️ **Et il faut dire de quoi chacune des deux moitiés est faite, parce
> qu'elles n'ont pas la même force.** *Mémoire* : un **compte** sur le
> portefeuille existant — aucun de ces layouts ne passe, ce qui ne borne pas un
> format futur. *Vitesse* : la fermeture ne repose plus sur un plafond, **il
> n'y en a pas**. Ce qui la tient est (i) le **partage du temps** — le format
> se dispute la part que `Planes14` capture déjà, son décodage pesant ~7 % du
> trafic — et (ii) une **induction sur trois points mesurés**, pas une borne :
> `Golay70` v1 (1,31×), v2 (1,77×) et E1v (0,25×) réduisent tous les octets et
> sortent tous **plus lents**, bornés en calcul.
>
> 🔎 **Une idée neuve sur le coût ALU rouvrirait la moitié vitesse — QTIP en
> est PRESQUE une, et ce « presque » est tout le sujet : ce n'est pas un
> défaut d'implémentation qui nous sépare de lui, c'est la TAILLE de notre
> codebook**, et le papier en nomme le mécanisme. Un état de treillis
> 16 bits tient dans une table de correspondance de 2 KiB ; un codebook de
> **1,1·10¹⁴ points n'y tient pas**. Notre index de réseau doit donc être
> **déplié** en flux de plans de bits à **4,80 b/poids**, et le noyau paie ces
> octets à la vitesse mémoire — l'écart de temps suit l'écart de trafic
> (2,40× d'octets pour 2,27× de temps, à fractions de borne quasi égales,
> 61 % contre 65 %). **Aucune idée connue ne replie ce flux à codebook
> constant** ; ce qui est ouvert, c'est le codebook lui-même, et ce n'est plus
> un travail de layout.
>
> **Trois branches se referment ici, dont deux qui pendaient :**
> 1. **L'argument mémoire de `Golay70`** — orphelin depuis le 2026-08-11, où la
>    ligne avait été fermée sur la seule *vitesse* (1,77× < 2,0×) alors que la
>    question était devenue mémoire. Verdict rendu : **+19,5 % au-dessus du
>    barreau, il ne suffisait pas non plus**. Fermé sur les deux axes.
> 2. **E3 ne se rouvre pas** malgré le seuil qui monte de 2,60 à 3,00 : son
>    3,0444 est une **borne basse de tassage**, pas un stride payé (dans la même
>    table, la géométrie `Planes` vaut 3,2060 là où le banc mesure 4,804 —
>    1,598 b/poids d'écart sur le même objet). Sa borne est déjà au-dessus du
>    seuil. L'enterrement en sort **renforcé**, pas fragilisé.
> 3. **Le package A (MoE) n'est plus un engagement produit** : A6 = référence
>    seulement. P2 et P6 sortent du chemin critique — leur pause a désormais une
>    raison, et non plus seulement une priorité.
>
> ⚠️ **Ce que ça ne ferme pas** : le MoE reste le seul axe connu qui change la
> *classe* de modèle chargeable, et l'étude du 2026-08-12 tient. Ce qui est
> décidé, c'est qu'on ne le **sert** pas.
>
> 🧭 **L'ordre de priorité qui en découle** — les deux fronts qui portent les
> critères fondamentaux sont ceux qui n'ont **aucune ligne de code** :
> ✅ **(1) Phase 1** — le point 14B au papier, 0 $, seul bloquant du papier :
> **LIVRÉE, papier soumis le 2026-08-24** ;
> **(2) Phase 2** — la qualité, le seul axe dont une découverte change le
> verdict produit à échelle fixée ;
> **(3) la famille `k` de P4 §2.6** — le poste des 45 % que le plancher désigne
> et qu'aucun format ne touche ; 🚨 **REQUALIFIÉ le 2026-08-25 : ce poste n'est
> pas un « sol machine » mais le coût de NOTRE GÉOMÉTRIE DE LANCEMENT** (F2 :
> un noyau réel passe sous `nullk`), et **D1 en a déjà repris une part sans
> toucher au format** — 252 → **144 matvec par token**, **×1,061
> [1,050–1,069]** à `ROT_SHARE` constant sur le chemin servi
> ([`mesures/d1-fusion-servie-2026-08-24.txt`](mesures/d1-fusion-servie-2026-08-24.txt)).
> Ce qu'on amortit est **attaquable**, et il vient d'être attaqué une fois ;
> **(4) Phase 3** — le 32B, après que 1 soit livrée et 2 tranchée.
> ⚠️ **(1) est livrée depuis le 2026-08-24** (papier soumis), donc cet ordre se
> lit aujourd'hui **2 → 3 → 4**.
> Tout le reste (e1c12, KV long, MoE) passe derrière et n'a plus de caractère
> bloquant.
>
> 🕳️ **Et ce chapeau disait le contraire de tout ça.** Il annonçait l'axe noyau
> « rouvert sur le décodage du rang » par le plan P1→P7. Cette ligne a rendu ses
> verdicts (P1, P1b, P1c, P5, E1v) : elle est **close**, et elle s'est fermée en
> montrant qu'elle attaquait un poste minoritaire. Les trois phases ci-dessous
> redeviennent la ligne principale, sans ligne parallèle.
>
> **Règles transverses, non négociables** (elles ont chacune été payées) :
> - Rien ne se lance sur GPU payant, rien ne se publie, sans **go explicite**.
>   Coût annoncé avant, cumul rapporté après.
> - Une variable par A/B. Tout ce qui touche aux **magnitudes** exige le gate
>   à profondeur sur 0.6B (28 blocs) — un A/B à 3 blocs ne suffit pas
>   (design C, `group_scales`).
> - Tout seuil de décision s'écrit et s'ancre (.ots) **avant** la mesure ;
>   un fichier ancré qui doit être édité est **ré-ancré**.
> - Chaque chiffre publié porte sa provenance (mesuré/calculé/estimé), son
>   bras (scellé/servi), et sa comptabilité (b/poids payload vs b/param
>   modèle entier).

> 🗓️ **Note du 2026-08-12 au soir — l'axe MÉMOIRE, à ne pas confondre avec
> l'axe noyau que ce plan arrête.** Une spec parallèle (le « lot X », née de
> la mesure AWQ) visait les b/poids et non la vitesse. Sa partie gratuite a
> été lancée le jour même :
> **`E1c` est prouvé exact** (3,7618 b/poids noyau contre 4,3424, sweep de
> 150 681 600 blocs) et **E3 est enterré sur papier** (3,0444 contre un
> critère de 2,60 posé d'avance). Verdicts :
> [`archive/passation-lot-x-2026-08-12.md`](archive/passation-lot-x-2026-08-12.md).
>
> **Ce plan ne s'en trouve pas modifié**, et c'est délibéré : il ne reste du
> lot X qu'un **banc de vitesse à ~0,2 $** (X3), qui est bien un item d'axe
> noyau — donc arrêté par la décision ci-dessus jusqu'à arbitrage explicite.
> Consigné ici pour qu'une session future ne le redécouvre pas comme un
> chantier ouvert : il est **fini côté gratuit, en attente côté carte**.

---

## 🆕 Ce que les lots F, G et D1 ont changé (2026-08-18 → 2026-08-24)

> Neuf campagnes en une semaine, **28,56 $ facturés depuis le 08-18** sur
> **27 jobs** (*mesuré*, [`data/jobs.csv`](data/jobs.csv), qui porte
> **73 lignes pour 87,36 $** depuis le 2026-08-02). **Aucune n'avait d'entrée
> dans les documents de reprise** au moment où cette section est écrite, et
> deux d'entre elles retirent une phrase que ce plan portait.
> Compteurs au 2026-08-25 (*mesuré*) : `docs/mesures/` **69 fichiers**,
> `docs/data/*.csv` **13 fichiers**, `proofs/` **22 documents et 16 ancrages
> `.ots`** — ⚠️ dont **aucun n'a jamais été upgradé**, cf. § 1.5.
>
> ⚠️ **La colonne « coût » ne somme pas au total, et c'est normal** : elle
> donne le job qui porte le résultat cité, là où un lot en compte plusieurs
> (F2 cumule 1,44 $ sur ses phases P0/P2/P3) et où le registre porte aussi les
> essais échoués à 0,00 $. Le total qui fait foi est celui de `jobs.csv`.

| lot | ce qu'il rend | coût *(mesuré)* | journal |
|---|---|---|---|
| **F2 / P3** | **QTIP dans notre banc** : 2,246 ms [2,245–2,248], 0,91 Go, 2,0000 b/poids, 405 Go/s, **4,89× FP16** contre `Planes14` 5,103 ms / 2,18 Go / 4,804 b/poids / 2,15× → **r = 2,27× [2,27–2,28]**. **`t(QTIP) < t(nullk)`** : le plafond de 4,77× est **FAUX** | 0,89 $ | [`f2-p3-qtip-banc`](mesures/f2-p3-qtip-banc-2026-08-21.txt) |
| **F5** | **σ de calibration = 5,2 %** au 4B (étendue 10,3 %), trois runs **complets**, les trois paires appariées **résolues** | 21,45 $ | [`f5-graines-4b`](mesures/f5-graines-4b-2026-08-19.txt) |
| **B2** | débits bout-en-bout **à plages**, trois tailles ; **14B à tête identique ×1,41 [1,40–1,41]**, la cellule que le dossier déclarait inexistante | 0,80 $ | [`b2-fusedrun-plages`](mesures/b2-fusedrun-plages-2026-08-18.txt) |
| **B3** | le 8B **re-scellé depuis le bucket** rejoue **5,322 b/param** au millième — contre **12,6 $** provisionnés pour une requantification | 0,24 $ | [`b3-8b-reseal`](mesures/b3-8b-reseal-2026-08-18.txt) |
| **F1** | le **dénominateur publiable** : `r = 1,024` (2 bras) et **1,015** (5 bras), ≤ 1,05 — notre témoin FP16 maison est **au niveau de cuBLAS sur L40S** | 0,08 $ | [`f1-cublasf16`](mesures/f1-cublasf16-2026-08-18.txt) |
| **F3** | chronométrage par **events CUDA** : écart hôte−device **0,1–0,2 %** (4–8 µs/round) ; `ncu` refusé par la **plateforme** (`ERR_NVGPUCTRPERM`), driver 580.159.03 capturé | 0,86 $ | [`f3-events`](mesures/f3-events-2026-08-19.txt) |
| **F4** | **seconde architecture** : sur **A100-SXM4-80GB, aucun bras à décodage ne bat FP16** — `Planes14` 0,79× · `Slot32` 0,73× · `Planes12x` 0,73× · `Golay70` v2 0,62× · v1 0,44× · AWQ 1,82× · cuBLAS 1,14× | ~1,00 $ *(estimé — premier `a100-large` du registre)* | [`f4-a100`](mesures/f4-a100-2026-08-18.txt) |
| **G1/G2** | **les horloges tranchent l'A100** : L40S **2 520 MHz**, A100 **1 410 MHz**, toutes deux **épinglées au boost max**, aucun bridage → 2 520/1 410 = **1,787** ∈ [1,60 ; 1,95] tamponné | 0,21 $ *(0,08 + 0,13)* | [`g-horloges-planes12x`](mesures/g-horloges-planes12x-2026-08-23.txt) |
| **G3** | **`Planes12x` SERVI bout-en-bout au 4B** : **85,0 tok/s [84,7–85,1] dans 2,36 Go**, ×1,96 sur le dense, ÷3,41 de mémoire carte, tokens gloutons identiques | 0,79 $ | *idem* |
| **D1** | **la fusion sur le chemin servi** : 252 → **144 matvec/token**, **×1,061 [1,050–1,069]** à `ROT_SHARE` constant | 0,24 $ | [`d1-fusion-servie`](mesures/d1-fusion-servie-2026-08-24.txt) |

**Cinq choses que ça change pour ce plan, dans l'ordre de ce qu'elles coûtent
à ignorer.**

1. 🚨 **Le plafond de format n'existe pas** (F2). L'axe format reste fermé —
   la fermeture ne tient pas à un plafond mais au partage du temps et à
   l'induction sur trois décodeurs bornés en calcul — mais **l'argument
   change**, et `nullk` doit se dire « plancher de notre géométrie de
   lancement », jamais « sol machine ». Détail et mécanisme du **dépliage** en
   tête de fichier.
2. 🚨 **Le σ de calibration vaut 5,2 %, pas 0,7 %** (F5). Tout gate de la
   Phase 2 qui **recalibre** se lit contre cette barre. Détail au § 2.1.
3. ❌ **Le claim de vitesse a un domaine de validité MESURÉ** (F4 + G).
   « À vitesse de matvec » est un résultat **L40S/Ada** : sur A100 tous les
   bras à décodage **ralentissent en absolu** et leurs Go/s effectifs
   **chutent** (`Planes14` 425 → 250, `Slot32` 428 → 266). Le plancher `nullk`
   y passe de 4,79× à 1,68×, et G établit que le ×1,78 inter-cartes **est le
   rapport d'horloges** (1,787), sur deux cartes épinglées sans bridage.
   ⚠️ **Les × inter-cartes ne se divisent pas** — règle du prereg F4 §3.
   ⚠️ Et ce que G prouve est une **horloge**, pas une occupation : les
   compteurs restent refusés par la plateforme.
4. ✅ **Le poste de géométrie est attaquable, et D1 l'a entamé** : la fusion
   `q/k/v` et `gate/up` par lignes rend **×1,061** sur le chemin servi, six
   critères pré-enregistrés verts — 128 tokens identiques entre bras fusés,
   divergence au dense **au même token 89**, **+3 686 400 octets exactement**
   (+0,008117 b/poids), même sha256 NVRTC des deux côtés. Décomposition
   mesurée : **87,0** (servi publié) → **94,9** (hissage de la rotation seul)
   → **100,6 tok/s [99,9–100,7]** (plus la fusion). Conséquence de
   priorisation au § « famille `k` ».
5. ✅ **Deux dettes de comparabilité sont soldées pour 1,04 $** : le
   dénominateur FP16 est **mesuré au niveau de cuBLAS** (F1, donc tous les
   « vs FP16 » publiés tiennent), et l'écart hôte−device est **0,1–0,2 %**
   (F3, donc la soumission hôte est **entièrement recouverte** — ce qui
   **affaiblit** l'hypothèse « le poste latence, c'est l'hôte » **sans la
   réfuter**).

---

## Phase 1 — Papier v2 et solde des dettes de cohérence

> ✅ **SOLDÉE le 2026-08-24 — LE PAPIER EST SOUMIS : ACM TACO,
> `TACO-2026-428`, au commit `e21a8bb`.** Les cases restées `[ ]` ci-dessous
> ne sont pas des travaux en cours : elles ont été traitées dans le manuscrit
> soumis ou requalifiées par une mesure postérieure. **Deux entrées que le
> plan n'avait pas prévues y figurent** : le bras **QTIP** est au corps
> (`tab:layouts`, `sec:qtip`, lot F2) — il fait du papier le seul document du
> dossier à comparer notre noyau à un 2 bits déployé **dans le même
> processus** — et la réserve d'`evaluation.tex` sur le **tirage de
> calibration** a reçu son chiffre (**σ = 5,2 %**, lot F5) ; le journal F5
> demande explicitement qu'elle reste **au corps**, pas en annexe.
>
> ⚠️ **Ce que la soumission ne solde pas, et qui est suivi dans
> [`BACKLOG.md`](BACKLOG.md)** : les surfaces publiées hors papier
> (`docs/hf-model-card.md`, README), les **16 ancrages `.ots` jamais
> upgradés** (§ 1.5), et la conséquence de la décision du 2026-08-25 — **le
> dépôt reste public**, donc toute phrase de plan supposant un dépôt privé
> pendant la revue est **périmée**.
>
> 🕳️ **Un titre de cette section est faux et reste en place** : le § 1.1
> s'intitule « le bloquant » ; il ne l'est plus, il est **livré**. Conservé
> tel quel parce que les verdicts de métrique qu'il porte — genou MMLU contre
> genou perplexité — sont la **raison d'être** des formulations du manuscrit
> soumis, et qu'une session qui les relit sans eux réécrirait le mauvais
> récit.

**Objectif** : que plus aucune surface publiée (papier, README, `CLAUDE.md`,
`proofs/`) ne porte une affirmation que le dépôt lui-même a démentie.
**Coût : 0 $ (option +~3 $). Durée : 2-3 jours. Aucun risque.**
⚠️ **Ce devis est celui du 2026-08-12 et il a été dépassé par ce que la phase
a réellement demandé** : les lots F et G qui l'ont accompagnée ont coûté
**28,56 $ sur 27 jobs** depuis le 08-18 (*mesuré*). Ce n'est pas un
dépassement de la Phase 1 telle qu'elle était écrite — c'est que la
cohérence a exigé des **mesures**, pas seulement des relectures.

### 1.1 Intégrer le point 14B au papier (le bloquant)

Le 14B est mesuré ([`mesures/campagne-14b-qualite-2026-08-10.txt`](mesures/campagne-14b-qualite-2026-08-10.txt)) :
×1,1894 de ppl, −6,85 pp apparié IC95 [+4,52 ; +9,12], écart AWQ **+6,09 pp,
IC95 [+3,62 ; +8,52], SE 1,25 pp, McNemar exact p = 1,143e-11**.

🚨 **Ce paragraphe disait « écart AWQ 6,09 pp ⚠️ non apparié (78,21 − 72,12 :
la paire `AWQ − LLVQ` n'existe pas au 14B, ni IC ni McNemar — ne jamais lui en
donner) ». C'était faux, et corrigé le 2026-08-17 pour 0 $** : les trois dumps
dormaient dans le **bucket monté**, la vérification qui les avait déclarés
perdus ayant cherché *sur la machine*
([`mesures/mmlupair-14b-2026-08-17.txt`](mesures/mmlupair-14b-2026-08-17.txt)).
Le point estimé ne bouge pas d'un centième ; il cesse d'être nu. **Les trois
écarts sont donc homogènes** : 4B +14,45 [+11,60 ; +17,27] · 8B +7,49
[+5,28 ; +9,70] · 14B **+6,09 [+3,62 ; +8,52]**.

- [ ] Ajouter les lignes 14B à `data/echelle-4b-8b.csv` (ou un CSV dédié) et
      régénérer la figure d'échelle en **3 points**. ✅ Le CSV a désormais
      `params_total` et les `vram_*` du 14B (5,106 vs 5,404, **−5,5 %**), et
      les neuf écarts appariés vivent dans
      [`data/mmlu-appariee.csv`](data/mmlu-appariee.csv).
- [ ] Réécrire le récit d'échelle — abstract, intro, évaluation, limitations.
      🚨 **La consigne de cette case a changé le 2026-08-17 et il faut la lire
      en entier.** Elle disait : « "the gap halves" → le **genou** (fonte de
      l'excès de ppl −43 % puis −14 % ; écart AWQ 14,45 → 7,49 → 6,09 pp) ;
      "two points, not a law" → "**three points, a knee, not a law**" ».
      🚨 **AMENDÉE LE SOIR MÊME, et c'est l'amendement qui compte : LE VERDICT
      DÉPEND DE LA MÉTRIQUE, donc la consigne aussi.** Le paragraphe ci-dessus
      est **vrai sur MMLU** et **faux sur la perplexité**, qui a reçu ses barres
      au 4B en fin de journée.
      **Sur l'écart MMLU au 4 bits, le genou ne survit pas au test** : la chute
      vaut −6,96 pp du 4B au 8B (SE 1,82, p = 0,0001, **résolue**) et −1,40 pp
      du 8B au 14B (SE 1,68, p = 0,40, **NON résolue**). Écrire « a knee » sur
      cette métrique publierait un ralentissement que les barres ne séparent
      pas. ⚠️ Et « no knee » y serait tout aussi faux : p = 0,40 ne prouve pas
      l'égalité, les données sont **muettes** sur ce palier.
      **Sur la perplexité, le genou est RÉSOLU** : pas 4B→8B ×0,881211
      [0,856 ; 0,907] · pas 8B→14B ×0,974855 [0,959 ; 0,991] · leur différence
      appariée **−0,100992 [−0,137670 ; −0,064313], t = −6,06**
      ([`mesures/ppl-appariee-4b-2026-08-17.txt`](mesures/ppl-appariee-4b-2026-08-17.txt)).
      **La formule à écrire est « three points, not a law »**, avec les trois IC
      et les tests de palier des **deux** métriques — la direction tient sur les
      deux (MMLU 4B→14B : −8,36 pp, p ≈ 1e-5), la *forme* est mesurée en
      perplexité et indéterminée sur les capacités, et **le 32B est ce qui
      trancherait** cette seconde question. 🚨 **Ne jamais écrire « the knee »
      ni « no knee » sans nommer la métrique** : chaque forme nue est fausse de
      moitié.
      ⚠️ Côté perplexité, « −43 % » contre « −14 % » **peut désormais s'écrire,
      barré des deux côtés** — 🕳️ la consigne précédente disait « le premier
      n'est pas barrable (journal 4B de synthèse, pas de NLL par fenêtre) », ce
      que la récupération des logs du job a démenti : **−42,8 % IC95
      [−51,8 ; −33,5]** contre **−13,9 % IC95 [−22,8 ; −4,9]** sur f16. Sur AWQ
      le second pas vaut **−1,58 % [−3,14 ; −0,004]** et exclut zéro **de
      0,005**, donc **ne jamais écrire « the gap closes significantly »**
      ([`mesures/ppl-appariee-8b-14b-2026-08-17.txt`](mesures/ppl-appariee-8b-14b-2026-08-17.txt)).
      ⚠️ **Ces deux nombres ne sont pas dans la même paramétrisation** et les
      juxtaposer tels quels suggère un rapport de 9 qui n'existe pas : −13,9 %
      est la fonte de l'**excès**, −1,58 % celle du **rapport** (la seule forme
      que le journal publie sur AWQ). En rapport contre rapport : **f16
      −2,51 % [−4,12 ; −0,88]** contre **AWQ −1,58 % [−3,14 ; −0,004]**.
- [ ] Remplacer « the 4-bit baseline starts paying » par le résultat apparié.
      **Les trois échelles sont désormais testées** (1.2 faite le 2026-08-13),
      et la phrase **tient**, à condition de citer sa statistique :

      | f16 − AWQ, micro stratifié | verdict |
      |---|---|
      | 4B : +0,27 pp, IC95 [−1,63 ; +2,13] | **non résolu** |
      | 8B : +3,07 pp, IC95 [+1,61 ; +4,69] | **résolu** |
      | 14B : +0,76 pp, IC95 [−0,65 ; +2,17] | non résolu |

      ⚠️ Deux réserves à porter dans le texte, pas à lisser. (i) À 4B, le
      contrôle **non pondéré** résout (+1,97 [+0,92 ; +3,02]) ce que le micro
      stratifié ne résout pas — désaccord porté par `professional law`
      (poids 10,9 %). (ii) Le 14B **non résolu** après un 8B **résolu** n'est
      pas monotone : « starts paying » décrit le 8B, pas une tendance.
- [ ] Mettre à jour « Cost of evidence » (le total intègre les jobs 14B) et
      réconcilier les deux totaux 14B qui circulent (30,20 $ vs 31,46 $).

### 1.2 ✅ FAIT le 2026-08-13 — rejeu apparié 4B/8B (1,30 $, pas 3)

`echelle-4b-8b` § 1bis posait ce rejeu comme condition pour toute phrase sur
l'écart AWQ à ces tailles : **la réserve est levée**. Six bras MMLU rejoués
avec dumps par question, **empreinte `65dcd53655e8bfa5` sur les six**, et les
six micros reproduisent les chiffres publiés **au centième** — le harnais
traverse trois mois sans dériver.
[`mesures/mmlupair-4b-8b-2026-08-13.txt`](mesures/mmlupair-4b-8b-2026-08-13.txt),
dumps dans [`data/mmlu-dumps/`](data/mmlu-dumps/).

| Δ = A − B, micro stratifié | 4B | 8B |
|---|---|---|
| f16 − AWQ | +0,27 [−1,63 ; +2,13] **non résolu** | +3,07 [+1,61 ; +4,69] résolu |
| f16 − LLVQ | +14,73 [+11,98 ; +17,47] | +10,57 [+8,58 ; +12,57] |
| **AWQ − LLVQ** | **+14,45 [+11,60 ; +17,27]** | **+7,49 [+5,28 ; +9,70]** |

✅ **« The gap halves » sort renforcé** : sur l'axe qui porte la thèse
(AWQ − LLVQ), les deux IC95 **ne se recouvrent pas**.
⚠️ **L'axe f16 ne suit pas** : les IC de f16 − LLVQ **se recouvrent**
([11,98 ; 12,57]). La fonte est solide face au 4 bits, non résolue face au
f16 — asymétrie à écrire, pas à lisser.
⚠️ **Aucun de ces intervalles ne teste la différence des différences entre
échelles.** `mmlupair` apparie deux bras sur les mêmes questions ; il
n'apparie pas deux tailles de modèle. Non-recouvrement d'IC ≠ test formel, et
le papier ne doit pas laisser croire le contraire.

### 1.3 Corrections de chiffres et d'étiquetage (papier)

- [ ] Étiqueter systématiquement bras **scellé** (−14,73 / −10,56) vs
      **servi q8** (−14,6 / −10,5) — aujourd'hui une seule phrase donne la clé.
- [x] ✅ **2026-08-16 — ARBITRÉ ET PROPAGÉ : le chiffre publié est 5,162**
      (le verdict de `rtbits`, qui écrit lui-même « LE CHIFFRE 4B q8 À PUBLIER
      EST 5,162 ») ; le 5,15 survit partout **étiqueté** « division de
      l'affichage carte arrondi », jamais comme LE chiffre. Porté sur
      `README.md`, `docs/campagne-finale-2026-08-07.md`,
      `docs/cheatsheet-defense.md`, et le couple `campagne-finale.csv` ↔
      `evaluation.tex`. ⚠️ **Deux extensions au-delà de la demande** :
      `tab:progression` portait 5,15 pour le **même objet** (deux tables du
      même PDF donnant deux b/param d'un seul objet), et `check_tables.py`
      comparait à 2 décimales — il aurait donc *imposé* 5,16 et laissé la
      cellule dériver ; il compare désormais la chaîne du CSV telle quelle.
      🚨 **Reste `docs/hf-model-card.md` (4 sites)** — surface **publiée** sur
      HF : l'éditer sans republier crée une divergence entre le dépôt et
      l'objet publié. **Décision opérateur.**
      🕳️ Cette case était `[ ]` pendant que le §1.6 portait la même décision
      `[x]` — deux entrées du même fichier, contradictoires, sur un arbitrage
      qui n'était pas encore rendu. Corrigé le 2026-08-16.
- [ ] Le 1,29× du 8B est un quotient inter-invocations (34,4/26,6, deux
      jobs) — le former en apparié ou l'étiqueter, comme la règle du papier
      l'exige des autres rapports.
- [ ] Résoudre les deux contradictions internes relevées : « 2.07 on-disk »
      (intro) vs « payload + 8 % » (évaluation) ; « records plus per-class
      tables » (§ Slot32) vs « per-class tables are excluded » (méthodologie).
- [ ] Restreindre « all ratios are medians with ranges » (limitations) aux
      ratios de banc — les ratios bout-en-bout ne le sont pas.

### 1.4 Étendre `check_tables.py` (l'angle mort identifié)

- [ ] Couvrir `tab:campaign` (`campagne-finale.csv`), `tab:campaign8b`
      (`tableau-8b.csv`), `tab:phases` (`phases.csv`), `tab:progression`
      (`progression.csv`) — les CSV existent déjà, le script n'en vérifie
      qu'une table sur six.

### 1.5 `proofs/` — remettre la vérifiabilité en règle

- [x] ✅ **2026-08-16** — `proofs/README.md` porte désormais l'état RÉEL des
      11 pré-enregistrements à deux colonnes indépendantes (*posé avant sa
      mesure ?* / *atteste le fichier courant ?*), avec la recette de
      vérification hors ligne (`ots info | head -1` contre `shasum -a 256`) et
      la cause de la détache (la refonte documentaire `b799c32`). La
      revendication « vérifiable sans nous faire confiance » ne survit que pour
      ceux qui la méritent.
- [ ] ⏳ **Décision opérateur** — les deux ancres détachées (08-10, 08-11)
      restent détachées. Ré-ancrer *après* la mesure produirait exactement
      l'objet sans valeur qu'un tampon existe pour éviter : les trois remèdes
      possibles sont écrits dans le README, **aucun n'est retenu**. Le plus
      proche de ce qui a marché ici serait un garde de build sur les empreintes
      (la forme du garde de `bin/rankbench`, le seul mécanisme qui ait tenu).
- [ ] 🚨 **2026-08-25 — AUCUN des 16 ancrages `.ots` n'a jamais été upgradé**,
      et c'est la dette qui touche le plus directement la revendication de
      vérifiabilité. Compté ce jour : **16 fichiers `.ots` pour 22 documents**
      dans `proofs/`, **tous** porteurs de **4 `PendingAttestation` et 0
      `BitcoinBlockHeaderAttestation`**. Un tampon en attente atteste une
      **promesse de calendrier**, pas une chaîne : tant que l'upgrade n'a pas
      tourné, « vérifiable sans nous faire confiance » revient à faire
      confiance aux calendriers. L'upgrade coûte **0 $ et une commande** —
      soit on le passe, soit la dette se déclare **à l'endroit où la
      vérifiabilité est revendiquée**, pas seulement ici.
      ⚠️ Sans effet sur l'**antériorité** des seuils : ce qui est en cause est
      la force de la preuve, pas la date qu'elle promet.
- [ ] ⏳ **Quatre preregs sans aucun tampon** : `2026-08-13`, `p2`, `p3`, `p4`.
      Pour `p2` et `p4`, la mesure n'a pas eu lieu — `ots stamp` reste possible
      et **doit précéder** le premier noyau `k` (exigence du prereg P4 lui-même).
      Pour `08-13` et `p3`, la mesure a eu lieu : seule une déclaration de dette
      dans le document répare quelque chose.

### 1.6 `CLAUDE.md` et `README.md` — purge des affirmations démenties

- [x] ✅ **2026-08-16** — « l'écart au 4 bits fond deux fois plus vite »
      retiré, le 14B et le genou ajoutés aux tables de `CLAUDE.md` (§3ter,
      §3bis). ⏳ **Reste `README.md`** : « two points » → trois.
- [x] 🚨 **2026-08-17 (matin) — le genou ajouté la veille est RETIRÉ**, et de
      toutes les surfaces vivantes (`CLAUDE.md` §3ter/§3bis/§6, `HISTORIQUE.md`,
      `echelle-4b-8b`, ce fichier, `README.md`, `cheatsheet-defense`). Motif :
      les trois écarts AWQ − LLVQ étant enfin appariés, la chute d'un palier au
      suivant se teste — **4B→8B résolue (p = 0,0001), 8B→14B NON résolue
      (p = 0,40)**. Le ralentissement n'est pas séparé par les barres. ⚠️ Et
      p = 0,40 ne prouve pas l'égalité : les données sont **muettes**.
      🕳️ Illustration exacte de la règle du §5 de `CLAUDE.md` — la correction
      du 08-16 était juste sur ce qu'elle retirait et a introduit, dans le même
      geste, une affirmation non testée.
- [x] 🚨 **2026-08-17 (soir) — la moitié PERPLEXITÉ du retrait ci-dessus est
      elle-même RETIRÉE, et les mêmes surfaces repassées pour NOMMER LA
      MÉTRIQUE.** Le retrait du matin portait sur le seul test alors disponible
      (MMLU), où il **tient**. La perplexité a reçu ses barres au 4B le soir —
      les NLL vivaient dans les logs du job, 0 $ — et son genou est **RÉSOLU**
      (pas1 − pas2 = −0,100992 [−0,137670 ; −0,064313], t = −6,06).
      **Deux métriques, deux verdicts** : ce n'est pas une contradiction, c'est
      une différence de puissance (49 140 tokens appariés entre tailles contre
      2 280 questions non appariées) et d'objet (raisonnement contre
      restitution). 🚨 Toute phrase sur le genou **nomme désormais sa métrique**
      — les deux formes nues sont fausses de moitié chacune.
- [x] ✅ **2026-08-17** — « la paire `AWQ − LLVQ` n'existe pas au 14B / ne
      jamais citer 6,09 avec un intervalle / la recalculer exige de refaire la
      campagne » retiré partout : elle existe, pour 0 $
      (+6,09 [+3,62 ; +8,52], McNemar 1,143e-11).
- [x] ✅ **2026-08-17** — ligne mémoire du 14B ajoutée (**5,106 vs 5,404,
      −5,5 %**), avec la **non-monotonie** de la marge et son mécanisme (part de
      l'embedding), et intervalles de perplexité 8B/14B, le 4B **visiblement
      sans**. 🚨 **Cette dernière réserve est levée le soir même** : les trois
      cellules 4B sont barrées (LLVQ/f16 **+38,45 %** [+33,62 ; +43,45] ·
      AWQ/f16 +10,49 % [+8,55 ; +12,47] · LLVQ/AWQ +25,31 % [+20,01 ; +30,84]),
      donc **la colonne perplexité porte ses barres aux trois tailles**.
      Coût : 0 $ — les NLL étaient dans les logs du job, désormais commitées.
- [x] ✅ **2026-08-17** — « `E1c14` est plus gros que `Planes14` une fois
      aligné » requalifié en **verdict 4B** : la pénalité d'alignement vaut
      +15,47 % au 4B mais +4,18 % au 14B, où `E1c14` aligné passe **sous**
      `Planes14` (4,6410 contre 4,7063). ⚠️ Aucun de ces nombres n'est une
      vitesse — cela ne le ressuscite pas.
- [x] ✅ **2026-08-16** — « 25 % de mémoire en moins » → **10,6 %** (5,322
      contre 5,956, `rtbits-planes-8b`), dans `echelle-4b-8b` **et**
      `CLAUDE.md`. 🕳️ Au passage : le rapport ne suivait pas ses propres
      nombres (ils donnaient 9,9 %) et le « 5,37 » n'était étayé par aucune
      mesure de b/param — c'était un payload `Slot32` en b/poids de projections.
- [x] ✅ **2026-08-16** — `LLVQ_FUSED_LAYOUT` documenté à ses **4** valeurs ;
      paragraphe « SKIP: sur stderr » supprimé ; couche fossile de fin de §6
      balisée ✅/🚨 sans suppression ; « quatre crates du cœur » → **cinq** ;
      ancres de code vérifiées dans le source et corrigées.
      ⚠️ **Ce plan portait lui-même une ancre fausse** : `fused.rs:106 → :120`
      pour `LLVQ_EMBED`, alors que la bonne est **`:137`** (`EmbedMode::parse`,
      vérifié).
- [x] ✅ **2026-08-16** — `5,15` → **5,162** b/param, avec son étiquette
      (le 5,15 est une citation d'affichage carte arrondi).
- [ ] En-tête : donner les deux formulations de débit (×2,03 / ×1,12), et
      « qualité identique au bit près » → « mêmes tokens jusqu'au tie-break ».
- [ ] Reste de « Divers » : cellule G6 « close depuis le 08-07 » → re-close le
      08-11 ; `Planes12x` « pas dans le modèle » → « câblé, pas servi »
      (⚠️ **trois** états, pas deux : câblé ≠ servi ≠ absent).

> 🔎 **Ce que la passe du 2026-08-16 a trouvé en plus, et qui n'était dans
> aucune case.** (i) La table des layouts de `CLAUDE.md` — la plus citée du
> dossier — intitulait « b/poids **payload** » une colonne de b/poids
> **noyau** (5,510 / 4,804 / 4,342 / 3,589 ; les payloads valent 5,3756 /
> 4,6667 / 4,2029). Défaut de dénominateur, exactement la règle n°1 du §7.
> (ii) `docs/format-noyau.md` ne chiffrait **jamais le dénominateur** du coût
> de décodage qu'il mesure : le plancher y est désormais une section.
> (iii) ⚠️ Une dette de provenance a été **annoncée à tort puis retirée** : le
> critère de 1,6× qui a écarté `Golay70` v1 **est** antérieur à sa mesure — il
> est dans le message du commit `caef2ac`, 52 minutes avant. `git log -S` ne
> cherche que dans le contenu des fichiers, jamais dans les messages de commit,
> et une absence avait été conclue d'un outil dont le périmètre n'était pas
> énoncé. **Il reste que l'antériorité tient par une date de commit et non par
> un tampon** — et `paper/sections/layouts.tex:129` écrit « fixed before
> measuring » sans référence : à sourcer, pas à retirer.

### Gate de sortie de phase

`make` (figures + check + pdf) vert avec les nouvelles tables couvertes ;
grep-liste des phrases condamnées = 0 occurrence hors `archive/` ; tag
**`paper-v2`**. Livrable : papier corrigé + dépôt cohérent.

✅ **Franchi le 2026-08-24 par la soumission** (`TACO-2026-428`, commit
`e21a8bb`). ⚠️ **La moitié « dépôt cohérent » du livrable, elle, n'est pas
close par la soumission** : le papier a été corrigé du plafond de 4,77×
(F2), les documents internes ne l'étaient pas — ce fichier reçoit la
correction aujourd'hui, les autres surfaces sont suivies dans
[`BACKLOG.md`](BACKLOG.md). 🕳️ **C'est exactement le motif que la Phase 1
existe pour empêcher** : une affirmation retirée d'une surface et laissée sur
les autres, ici pendant quatre jours.

---

## Phase 2 — Qualité : attaquer le mécanisme du déficit MMLU

**Objectif** : tester les deux seuls leviers qui visent le *mécanisme* du
déficit (le raisonnement s'effondre, la restitution tient — l'oracle de
calibration ne borne que la perplexité, donc la piste volume est morte mais
pas celles-ci). **Coût : ≤ 10 $ GPU + jours d'ingénierie. C'est le seul axe
dont une découverte change le verdict produit.**

🚨 **2026-08-19 — F5 TRAVERSE CETTE PHASE DE PART EN PART : σ de calibration
= 5,2 %, pas 0,7 %.** *(Il touche aussi le papier, dont il chiffre la réserve
la plus forte — mais c'est ici qu'il change un **gate**.)* Tout ce qui suit
**recalibre** par construction, donc tout gate de cette phase se lit contre
cette barre. Le détail et ce qu'il ne touche pas sont au § 2.1 ; la
conséquence de budget est immédiate et il faut la lire avant de deviser :
⚠️ **le « ≤ 10 $ » tient encore, mais seulement si les requantifications
restent LOCALES** (2,4 h sur M3 Max, 0 $, puis MMLU carte à 1-2 $ le bras).
Dès qu'un bras se requantifie sur carte, le repère mesuré est **~7,1 $ le run
4B complet** (F5 : trois runs, 21,45 $), et **répliquer un bras pour le
séparer du tirage multiplie d'autant**. **Le go de dépense se redemande sur
le protocole retenu, pas sur ce chiffre.**

### 2.1 Composition du corpus de calibration (quasi gratuit, en premier)

Hypothèse : calibrer sur un corpus pondéré raisonnement (maths, code,
chaînes de déduction) déplace ce que GPTQ préserve, là où C4 sur-représente
la restitution.

🚨 **LE GATE DE CETTE CASE SE LIT DÉSORMAIS CONTRE UN σ DE 5,2 %, ET C'EST LE
CHANGEMENT LE PLUS LOURD DE LA PHASE.** Le lot F5 a payé **trois runs
complets** du 4B — `LLVQ_CALIB_SEED ∈ {1,2,3}`, même corpus lu dans la même
heure, même codebook, même rotation, même protocole, empreinte de tokens
`3f1baca9033bf251` **partout**, donc appariables fenêtre par fenêtre :
**ppl scellé f16 = 16,7425 / 15,8836 / 15,1027** *(mesuré, bras **scellé**)*,
**étendue 1,6398 ppl = 10,3 %** de la médiane, **σ (n = 3) = 0,8202 ppl =
5,2 %**, et **les trois paires appariées sont RÉSOLUES** (t = +4,54 / +10,92
/ +7,68). Contrôle : les trois rendent **2,0702 b/poids effectifs et 1,771 Go
scellés**, identiques au fichier publié — **seule la qualité bouge, pas le
débit**
([`mesures/f5-graines-4b-2026-08-19.txt`](mesures/f5-graines-4b-2026-08-19.txt),
21,45 $ *mesuré*).

🕳️ **Le σ de 0,7 % hérité du lot B est donc faux d'un facteur ~7 à la taille
publiée, et le seuil « tout effet sous ~1,5 % est du bruit » avec lui** : il
était estimé sur **3 blocs de Qwen3-0.6B**, et le dossier l'a transporté à
l'objet publié pendant deux semaines. La réserve du § 2 de `CLAUDE.md` — « ce
σ n'est pas la barre d'erreur de l'objet publié » — était **juste et
insuffisante** : elle disait qu'on ne savait pas, sans dire de combien on
pouvait se tromper.

⚠️ **Ce que F5 ne touche PAS, et qu'il faut écrire avec.** (i) Les A/B à
**fichier constant** — KV q8, layouts runtime, embedding q8, tous les
verdicts de format — **ne recalibrent pas** : leur barre reste l'intervalle
apparié à **±0,12 %**, et cette variance ne les atteint pas. (ii) Les trois
artefacts **4B/8B/14B ont tous tourné sans graine**, donc sur le **même
préfixe contigu** : la courbe d'échelle compare des objets calibrés
identiquement et **ne porte pas cette variance**. (iii) Deux verdicts du lot
B — l'**oracle** (−1,6 %) et la **courbe de volume** (−1,2 % pour ×13 de
tokens) — tombent désormais **sous le plancher de bruit**, **sans que leur
conclusion soit renversée** : « le volume de calibration est plafonné » était
fondé sur des effets trop petits pour être distingués, ce qui est maintenant
**mesurable** et le reste. Le design C (×1,99) et le swap L ≤ 4 (+4,75 %)
restent **hors de portée** du bruit.

🔎 **Et une lecture mécaniste, étiquetée hypothèse** (celle du journal) : nous
calibrons sur ~131 k tokens contre ~100× plus au papier amont. Un jeu de
calibration petit ne déplace pas seulement la moyenne — l'oracle a montré que
non — il porte de la **variance**. C'est ce que la composition du corpus
prétend justement déplacer, donc l'hypothèse de 2.1 et celle de F5 visent le
même mécanisme par deux bouts.

- [ ] Construire 3 bras de calibration à 131 k tokens, même graine : C4 pur
      (contrôle = l'existant), C4+raisonnement 50/50, raisonnement pur.
- [ ] **Écrire le critère avant** : +2 pp de MMLU micro 4B vs contrôle, en
      apparié (`mmlupair`), hors σ McNemar. En dessous : piste close, comme
      l'oracle.
      🚨 **Amendement du 2026-08-19 — un contrôle UNIQUE ne suffit plus.**
      Trois bras de composition sont trois **recalibrations**, donc trois
      tirages d'une distribution dont l'étendue vaut **10,3 %** en
      perplexité ; un « +2 pp » lu contre un contrôle unique **confondrait
      l'effet de composition avec l'effet de tirage**. Deux formes tiennent :
      **n graines par bras** (coût ×n, à budgéter au repère de ~7,1 $ le run
      4B carte, ou 2,4 h de Mac par bras si tout reste local), ou un critère
      qui **exige de dépasser l'étendue inter-graines**. À défaut, le verdict
      honnête est **« non séparable »**, pas « piste close » — la nuance que
      F5 vient d'imposer rétroactivement aux deux verdicts du lot B.
      ⚠️ **Et la barre de cette case est en MMLU apparié, pas en perplexité** :
      F5 mesure la variance de tirage **sur la perplexité** et ne la transporte
      pas telle quelle sur les capacités. C'est un **ordre de grandeur** à
      respecter, pas un nombre à reporter — le mesurer sur MMLU demanderait
      de scorer les trois artefacts de F5, qui sont au bucket.
- [ ] Gate à profondeur 0.6B (28 blocs, gate ppl automatique — le corpus ne
      touche pas les magnitudes mais la règle s'applique quand même).
- [ ] Requantifier le 4B en local par bras (2,4 h M3 Max chacun, 0 $),
      MMLU sur L40S (~1-2 $ par bras, dumps conservés).

### 2.2 Compensation bas-rang post-hoc (EoRA/Recover-LoRA)

Le plus gros gain publié dans la littérature (+4-11 pp de MMLU), jamais
tenté ici. Coût en jours d'ingénierie, pas en dollars.

- [ ] Design : par couche, adapter `A·B` de rang r ajusté sur le résidu
      `W − Ŵ` dans la métrique hessienne (les hessiennes de calibration
      existent), servi comme correction additive f16 à côté du chemin fusé.
- [ ] **Comptabilité d'avance** : le rang se paie en b/param modèle entier —
      fixer le budget avant (proposition : ≤ 0,25 b/param, soit r ≈ 16 sur
      les projections du 4B) sinon on rachète la qualité en octets et toute
      la comparaison AWQ est à refaire.
- [ ] **Critère avant** : refermer ≥ 4 pp du déficit 4B (la moitié basse du
      publié), apparié, dans le budget d'octets déclaré.
      ✅ **Ce critère-ci échappe au σ de F5, et c'est ce qui le rend le plus
      lisible des deux leviers** : l'adaptateur se pose **sur l'artefact
      existant**, sans requantifier — l'A/B « avec / sans » est donc à
      **fichier de quantification constant**, du côté des A/B dont la barre
      appariée vaut ±0,12 %. ⚠️ Cela ne vaut que tant qu'aucune graine ne
      bouge : ajuster l'adaptateur **et** recalibrer serait deux variables à
      la fois, et le second effet est sept fois le premier.
- [ ] Gate à profondeur 0.6B, puis 4B. C'est un mécanisme post-hoc : il ne
      touche pas la boucle GPTQ, donc pas le motif design C — mais le gate
      reste obligatoire.

### 2.3 Si un levier rend : consolider

- [ ] Combiner les leviers gagnants, remesurer la ligne 4B complète de la
      campagne (mêmes empreintes `3f1baca9…`/`65dcd536…`), mettre à jour
      [`fiche-4b.md`](fiche-4b.md), [`campagne-finale-2026-08-07.md`](campagne-finale-2026-08-07.md)
      et le papier (v2.1), republier l'artefact HF si le fichier change.

**Si tout échoue** : le déficit sans fine-tuning est borné proprement — c'est
un négatif publiable, et la décision produit devient explicite : attendre le
32B/70B (phase 3) ou clore le volet produit.

---

## Phase 3 — Le point 32B : le dernier point de courbe

**Objectif** : trancher **si** la courbe d'échelle **des capacités** s'aplatit
— c'est le point qui décide de la thèse d'échelle.
🕳️ **Cet objectif disait « trancher si le genou du 14B est un palier ou une
pause », et il présupposait le genou.** Retiré le 2026-08-17 : sur l'**écart
MMLU au 4 bits**, la chute est **non résolue** entre 8B et 14B (1,40 pp,
SE 1,68, p = 0,40), donc il n'y a pas de genou établi à qualifier — et p = 0,40
ne dit pas non plus qu'il n'y en a pas. **Le 32B en devient plus décisif, pas
moins** : il est le seul point qui puisse séparer les deux lectures.
⚠️ **Et il ne faut pas croire la question réglée par la perplexité**, dont le
genou est **résolu** depuis le soir du 2026-08-17 (−0,100992
[−0,137670 ; −0,064313], t = −6,06). Les deux courbes sont distinctes, et c'est
la courbe de **capacités** que ce point sert : « ne plus jamais présenter la
perplexité seule comme preuve de qualité » (`CLAUDE.md` §3ter).
**Coût : ~62 $ estimé le 08-03
(621 s/bloc mesurés, bf16/C3 validé) ; budget avec marge : 80 $. Une nuit.**

> 🎯 **2026-08-16 — l'arbitrage donne à ce point une seconde raison d'être, plus
> forte que la première.** Ce paragraphe disait « la seule échelle proche du
> régime souveraineté (70B) ». C'est à corriger : au barreau arbitré, la carte
> laisse 27,93 Go pour les poids, soit **~43-46 Md de paramètres** à
> 5,162 b/param — **le 70B ne rentre pas, et aucun format connu ne l'y fait
> rentrer** (§B bis de la note produit).
>
> **Le 32B n'est donc pas le point le plus proche du produit : c'est la plus
> grande classe que le produit admette.** Les 4B, 8B et 14B sont des points de
> courbe ; le 32B serait **l'objet servi**. C'est le seul run de qualité qui
> mesurerait ce qu'on livre réellement, et ça change son rang : il cesse d'être
> un luxe de courbe pour devenir la mesure du produit.
>
> ⚠️ Sans effet sur ses **préconditions**, qui restent entières (Phase 1 livrée,
> Phase 2 tranchée, AWQ 32B scorable, go budget). On ne repaie pas une campagne
> pour un papier incohérent, et on ne la paie pas deux fois si un levier de
> qualité doit y entrer.

### Préconditions (go/no-go — dans l'ordre)

1. ✅ **SATISFAITE le 2026-08-24** — Phase 1 livrée, papier soumis
   (`TACO-2026-428`, commit `e21a8bb`). ⚠️ **Mais la condition change de
   nature plutôt que de disparaître** : la soumission ouvre une **fenêtre de
   revue**, donc un run de qualité qui déplacerait un chiffre du manuscrit se
   gère comme une **correction en revue**, pas comme une v2 libre. Ce qui
   était « on ne repaie pas une campagne pour un papier incohérent » devient
   « on ne rend pas incohérent un papier déjà soumis ».
2. Phase 2 **tranchée** : si un levier rend, le 32B se mesure **avec** le
   levier (sinon on paie deux fois) ; si rien ne rend, décision explicite de
   payer pour la courbe seule.
3. Vérifier qu'un **AWQ 32B officiel** existe et se score dans notre harnais
   (sans bras 4 bits, le point perd la moitié de sa valeur).
4. Go budget explicite — aucun plafond HF n'est en vigueur, le redemander.

### Contenu

- [ ] Requantification `leech1c12` 32B (~11,4 h, `rtx-pro-6000x2`, bf16),
      `verify_artifact` bit à bit.
- [ ] Qualité : ppl + MMLU **appariés** (dumps conservés), bras f16, AWQ,
      LLVQ, mêmes empreintes des deux côtés.
- [ ] Vitesse/VRAM : `fusedrun` `Planes14`+q8, et **`Planes12x`** — c'est
      l'échelle pour laquelle il a été câblé (part de queue ~5 %).
      ✅ **2026-08-23 — `Planes12x` n'est plus « câblé, pas servi » : il est
      MESURÉ SERVI au 4B** (G3, 0,79 $) — **85,0 tok/s [84,7–85,1] dans
      2,36 Go** (projections 1,94 + portés 0,41), **×1,96 [1,95–1,96]** sur le
      dense, **÷3,41** de mémoire carte, tokens gloutons identiques et
      divergence au **token 89/128**, le tie-break historique de `Planes14`
      ([`mesures/g-horloges-planes12x-2026-08-23.txt`](mesures/g-horloges-planes12x-2026-08-23.txt)).
      **C'est le point servi le plus compact mesuré**, et le bras 32B hérite
      donc d'un **chemin établi** au lieu d'un layout à valider en même temps
      que la taille.
- [ ] ⚠️ **Protocole de débit : la forme B2, pas le point unique.** 1
      génération jetée + **5 chronométrées**, médiane et **plage**, quotient
      des médianes **étiqueté** (les rounds des deux bras ne coexistent
      jamais, donc aucun rapport round par round n'existe sur ce banc-là).
      Le point unique du 08-17 n'est plus une forme publiable
      ([`mesures/b2-fusedrun-plages-2026-08-18.txt`](mesures/b2-fusedrun-plages-2026-08-18.txt)).
      Et **les deux bascules livrées depuis se posent explicitement** —
      `LLVQ_ROT_SHARE` / `LLVQ_FUSE_AB` / `LLVQ_FUSE` (D1, ×1,061 sur le
      servi) : les mesurer **au même réglage qu'au 4B**, sinon la ligne 32B ne
      se compare à rien.
- [ ] ⚠️ **Et la ligne à publier est celle À TÊTE IDENTIQUE.** B2 l'a établie
      aux trois tailles : **×1,11 → ×1,29 → ×1,41 [1,40–1,41]**, série
      **strictement croissante**, là où la série brute (×2,00 · ×2,57 ·
      ×2,55) **n'a aucun ordre** parce qu'elle est dominée par le handicap
      variable de *notre* bras dense. Un 32B qui ne mesure que le brut
      ajouterait un point à la mauvaise courbe.
- [ ] ⚠️ **Nommer la CARTE dans le verdict de vitesse, et la nommer d'avance.**
      F4 a mesuré une seconde architecture : sur **A100-SXM4-80GB, aucun bras
      à décodage ne bat FP16** (`Planes14` **0,79×**), et le lot G établit que
      le ×1,78 inter-cartes **est le rapport d'horloges** (2 520 contre
      1 410 MHz, les deux **épinglées au boost max**, aucun bridage). Un
      « ×N » de 32B publié sans sa carte referait exactement la faute qu'F4 a
      obligé à corriger. ⚠️ Et **les × inter-cartes ne se divisent pas** :
      deux cartes donnent deux verdicts, pas un rapport.
- [ ] **Critères pré-enregistrés et ancrés avant lancement.** Proposition à
      affiner en phase 1 : (i) chute MMLU appariée ≤ 6 pp **et** écart AWQ
      ≤ 4 pp → la thèse d'échelle tient, v2 du papier en « courbe à 4 points » ;
      (ii) sinon → acter le palier, conclure le volet produit, le papier reste
      ce qu'il est : un papier systèmes avec un négatif propre.
      🕳️ *(i) disait « survit au genou » — retiré le 2026-08-17, il n'y a pas de
      genou établi.* ⚠️ **Et le critère doit désormais porter sur un ÉCART
      TESTÉ, pas sur un point estimé** : la leçon du 08-17 est qu'un palier de
      1,40 pp entre deux campagnes n'est pas séparable avec des SE de ~1,2 pp
      chacune. Formuler le seuil du 32B sur la **chute d'écart 14B→32B avec son
      z**, pas sur la différence nue — sinon le run reproduira exactement le
      défaut qu'on vient de corriger.

### Livrables

Courbe d'échelle à 4 points, verdict produit final documenté dans
[`HISTORIQUE.md`](HISTORIQUE.md), et la décision de soumission (MLSys) prise
sur pièces.

---

## Ouvert par P3 (2026-08-15) — le KV q8 à contexte long

> ✅ **2026-08-16 — CE N'EST PLUS UN PRÉREQUIS PRODUIT.** A2 est arbitré à
> **8k**, c'est-à-dire **dans la région où P3 a mesuré**. Le verdict « contexte
> court seulement » suffit donc pour servir le q8 au contexte retenu, et
> l'instrument à contexte long cesse d'être bloquant. La question ci-dessous
> reste ouverte et juste ; elle n'a plus d'échéance, et elle ne se rouvre que si
> A2 bouge.
> ⚠️ **Ce qui reste vrai et non mesuré** : le **gain mémoire** du q8 n'est qu'un
> compte (147 456 → 78 336 o/token, ÷1,882, géométrie 36 couches × 8 têtes KV ×
> head_dim 128, batch 1). Il se cite en octets/token avec sa géométrie, jamais
> en b/param — un cache n'est pas un paramètre. Il tombera gratuitement dans la
> colonne « Go carte » du premier `fusedrun` de P4.

**Ce qui est acquis** : un cache KV à 8,5 bits (int8 + échelle et biais f16 par
groupe de 64, ÷1,882) ne coûte **rien en qualité** sur le 4B — +0,049 % de
perplexité, +0,33 pp de MMLU, les deux intervalles appariés contenant zéro — et
**5 à 7 % de débit à contexte court**. `LLVQ_KV=q8` est livré, testé, câblé par
constructeur, et **n'est pas le défaut**
([`mesures/kvq8-4b-2026-08-15.txt`](mesures/kvq8-4b-2026-08-15.txt)).

**Ce qui reste ouvert, et c'est la question produit** : le comportement à
contexte long. La série `n_new = 1024` a été abandonnée par la règle du §2.5 du
pré-enregistrement (première invocation 661 s > 600 s), donc le verdict est
étiqueté « contexte court seulement » — ce qui interdit le défaut quelle que
soit la valeur mesurée.

C'est précisément la région qui décide : à `n_new = 1024` le bras f16 tombe à
**5,6 tok/s contre 9,6** à 128, donc le coût du cache y domine, donc c'est là
que l'allègement devrait payer. Le lot a mesuré **le coût sans son bénéfice**.

**Ce qu'il faudrait, et ce que ça n'est pas.** Ce n'est pas une relance de
`gbench` à budget plus large : l'instrument charge un modèle par processus, ses
rapports sont inter-processus par construction, et sa série longue dépasse à
elle seule le plafond d'un lot. Il faut soit un banc qui garde le modèle
résident entre les deux bras, soit la mesure sur carte en P4 — où la colonne
« Go carte » de `fusedrun` donnerait le **gain mémoire**, que ce lot n'a pas
mesuré non plus (ppl et mmlu tiennent 16,8 Mo de cache, pas 0,604 Go).

⚠️ **Ne pas rouvrir en relisant le même run.** Les deux séries `n_new = 128`
sont rendues et vertes ; le manque n'est pas une imprécision, c'est une région
non visitée. Toute réouverture demande un instrument, pas une relecture.

---

## 🚨 FERMÉ le 2026-08-16 — E1v sur le chemin servi

`e1v` rend **0,25× FP16, 25 Go/s** sur L40S — contre un plancher de 1,60× posé
d'avance (critères d'X3 du 2026-08-12, repris sans amendement). Journal :
[`mesures/e1v-cuda-2026-08-16.txt`](mesures/e1v-cuda-2026-08-16.txt), job
`6a814ba31f5885ae605bcb55`, **0,85 $**.

**Le format a tenu, le décodeur non** : 1,09 Go lus contre 2,18 pour `Planes14`,
la moitié au chiffre annoncé, exactitude 2,4e-8·Σ|w·x| sur 1 105 920 lignes, 79
registres et zéro octet local. Ce qui plafonne est le calcul.

**Ce que ça clôt, au-delà d'E1v.** Quatre routes sous `Planes14` ont été
tentées — E3 (3,04 b/poids contre 2,60, sur papier), `Golay70` v2 (1,77× contre
2,0×), `e1c14` (plus gros une fois aligné, sur papier), E1v (0,25×). Toutes
bornées en **calcul**, aucune en octets. **Le plancher servi reste `Planes14` à
4,804 b/poids**, et c'est un résultat, pas une collection d'échecs.

🔎 **Et l'arithmétique qui aurait dû le prédire** : l'attribution du gisement
CUDA donne latence/occupation **39 %**, flux **33 %**, décodage **19 %**. Cette
ligne attaquait les 33 % en gonflant les 19 %. Le poste majoritaire n'a jamais
été touché.

**✅ `e1c12` a son verdict depuis le 2026-08-16, et il n'est pas celui qu'on
attendait** ([`mesures/e1c12-aligne-2026-08-16.txt`](mesures/e1c12-aligne-2026-08-16.txt),
0 $). Avec le terme d'exceptions des deux côtés : **E1c12 aligné = 4,2880 b/poids
noyau contre 4,3424 pour `Planes12x`, soit −1,3 % — il SURVIT.** Le modèle se
valide au passage sur le cas déjà tranché : `e1c14` aligné rend 5,2354, le
chiffre exact d'X3.

🔎 **Mais ce que ça change n'est pas ce que ça mesure.** L'argument d'E1c était
de supprimer le bourrage ; aligné, il en rend presque tout. Il reste 1,3 %
contre un layout qui n'est lui-même pas servi. **La question d'E1c12 cesse donc
d'être une question de bits** : elle devient « la transposition rend-elle le
décodage plus rapide que `Planes12x` ? », ce qui était son argument d'origine —
32 lanes lisant le même mot, une diffusion L1 au lieu de 32 lectures dispersées.

⚠️ Et cette question **n'hérite pas du verdict d'E1v**. E1v est mort d'être borné
en CALCUL. E1c décode le même contenu que `Planes12x` — des sélections sur des
plans de bits — et sa transposition est un problème de motif de lecture, pas
d'ALU. Le pronostic ne se transporte pas.

> 🚨 **2026-08-16 — mais la question perd son enjeu, et il faut le dire avant
> d'écrire un noyau.** Le barreau produit arbitré vaut **3,00 b/poids noyau**
> (§B de la note produit) ; `e1c12` aligné pèse **4,288**, soit **+42,8 %**.
> Même admis au banc, il ne rapproche du barreau ni lui ni personne — son gain
> se lit **−10,7 % de bits contre `Planes14`**, pas un changement de classe.
> Et le partage du temps borne son gain de vitesse : le format ne se dispute
> qu'une part de la passe, `Planes14` en capture déjà l'essentiel, et son
> décodage ne coûte que **~7 %** du temps de trafic.
> 🕳️ **Cette phrase disait « tout format plafonne à 4,77×, `Planes14` en
> capture déjà 2,16× » — retiré le 2026-08-21.** Le plafond n'existe pas :
> QTIP rend 2,246 ms là où `nullk` en rend 2,306, donc un noyau réel passe
> **sous** le prétendu sol (F2). **Ce qui ne change pas ici, c'est le verdict
> sur `e1c12`** : son argument n'a jamais été un plafond mais le peu qu'il
> reste à gagner sur le poste que le format touche — et ce peu-là est mesuré,
> pas déduit.
>
> **Conséquence de priorisation, pas de verdict** : la ligne `e1c12` reste
> ouverte et légitime, mais elle passe **derrière** Phase 1, Phase 2 et la
> famille `k`. Elle ne se lance pas seule — le job carte se mutualise avec
> celui de P4, sans quoi on paie deux fois 1 468-1 481 s de transcodage hôte.
> ⚠️ **Et avant tout banc, ses seuils doivent être amendés et ré-ancrés** : les
> trois seuils d'X3 (≥ 1,9× / ≥ 2,05× / < 1,6×) ont été posés en comptabilité
> **non alignée** (3,7618 b/poids), or le bras à mesurer lit le flux **aligné**
> (4,2880, +14 % d'octets). Dans la comptabilité alignée, la tranche « ≥ 1,9×
> remplace `Planes12x` » n'achète plus que −1,3 % : **le seul verdict qui garde
> un sens est ≥ 2,05× contre `Planes14`**.

## 🆕 MESURÉ le 2026-08-16 — le plancher, et ce qu'il plafonne

> 🚨 **CE TITRE EST FAUX DEPUIS LE 2026-08-21 : le plancher ne plafonne RIEN.**
> Conservé pour que la correction se lise à l'endroit où l'erreur a été
> commise. Ce que `nullk` mesure reste exact et utile — c'est son **statut**
> qui a changé : **plancher de NOTRE géométrie de lancement**, pas sol
> machine. Voir l'encadré sous la table.

`nullk` — même grille, même tuilage, même staging, même épilogue, **aucun poids
lu** — rend **2,305 ms** contre 5,102 pour `Planes14` :
[`mesures/nullk-plancher-2026-08-16.txt`](mesures/nullk-plancher-2026-08-16.txt),
job `6a81b2b71f5885ae605bdcc9`, **0,77 $**.

**Le plancher est 45,2 % du bras servi**, et il n'était jusqu'ici qu'un reste
obtenu par soustraction.

| | |
|---|---|
| ~~plafond absolu de tout travail de **format**~~ | 🕳️ **RETIRÉ le 2026-08-21** — la ligne disait **4,77× FP16 (= FP16 / plancher)** ; elle supposait que rien ne pouvait passer sous `nullk`. Un noyau réel y passe |
| où `Planes14` en est | **2,16×** |
| ce que le format achète, **net** du plancher | **3,11×** (8,691 ms de trafic contre 2,797) |
| coût du décodage de `Planes14` | **~7 %** du temps de trafic (779 Go/s net contre 836 pour du FP16 pur) |

> 🚨 **CE QUI A TUÉ LE PLAFOND, et il vaut la peine d'être lu en entier.** Le
> lot F2 a porté **QTIP** dans ce même banc — un seul processus, 7 rounds dont
> 2 jetés, bras entrelacés. **QTIP finit les mêmes 252 projections en 2,246 ms
> [2,245–2,248] en lisant 0,91 Go ; `nullk` en met 2,306 en n'en lisant
> aucun** (séparation 2,7 %, résolution 0,72 %). `f = 61,1 %` contre un
> plafond **pré-enregistré** de 59,6 % — **erratum consigné au journal**, le
> `.ots` interdisant l'édition du prereg tamponné
> ([`mesures/f2-p3-qtip-banc-2026-08-21.txt`](mesures/f2-p3-qtip-banc-2026-08-21.txt)).
>
> ⇒ **`nullk` borne notre géométrie de lancement — un warp par ligne de
> sortie, 252 lancements — et rien d'autre.** Un noyau autrement formé passe
> dessous. Le mécanisme de l'écart, lui, est nommé et il n'est pas un défaut
> d'implémentation : un état de treillis 16 bits tient dans une table de
> correspondance de 2 KiB, **un codebook de 1,1·10¹⁴ points n'y tient pas**,
> donc notre index de réseau est **déplié** en flux de plans de bits à
> **4,80 b/poids** et le noyau paie ces octets à la vitesse mémoire — 2,40×
> d'octets pour 2,27× de temps, à fractions de borne quasi égales
> (`paper/sections/layouts.tex`).
>
> ✅ **Ce que la table garde**, et qui n'a jamais dépendu du plafond : les
> **45,2 %**, les **~7 %** de coût de décodage, et le partage du temps entre
> format et reste. C'est là-dessus, et non sur un sol, que repose la fermeture
> de l'axe format.

🚨 **Ce que ça dit de quatre tentatives** : le format n'a que 55 % du temps à
disputer, `Planes14` en capture déjà l'essentiel, et 45 % ne sont touchés par
**aucun** format. Ce poste **était** le plus gros et le seul jamais attaqué —
c'est exactement ce que la famille **k** de P4 §2.6 existe pour amortir, et
elle n'est toujours pas écrite.
⚠️ **Mais depuis le 08-21 on sait ce qu'on y amortit** : le
coût de **notre** géométrie, pas une fatalité de la machine — donc un poste
**attaquable**, et **D1 en a repris une part le 08-24** (252 → 144 lancements,
×1,061) sans écrire un seul noyau `k`.

⚠️ Ce n'est **pas** les « 39 % » de l'attribution du 05-08 : celle-ci découpe
2,04 ms par **token**, normes et attention comprises. Deux dénominateurs.

---

## Priorité 3 — La famille `k` (P4 §2.6) : le seul poste jamais attaqué

**Ce que le plancher désigne, et que rien n'a jamais visé.** 45,2 % du temps
d'une passe n'est touché par **aucun** format. La famille `k` — le même noyau
servant `k` colonnes par lancement — est le seul levier écrit qui l'amortisse :
le plancher se paie **une fois pour `k` colonnes** au lieu d'une fois par
colonne. **Coût : code sur Mac (0 $), puis un job mutualisé 0,8-1,0 $.**

> 🚨 **REQUALIFIÉE le 2026-08-25, et la requalification change ce qu'on a le
> droit de promettre.** Ce titre et ce paragraphe supposaient que les 45,2 %
> étaient un **sol machine** ; F2 a montré que non — un noyau réel (QTIP,
> 2,246 ms) passe **sous** `nullk` (2,306 ms). **Ce que la famille `k`
> amortit, c'est le coût de NOTRE GÉOMÉTRIE DE LANCEMENT** : un warp par ligne
> de sortie, 252 lancements par token.
> 🕳️ **Et le titre est faux d'un mot depuis le 2026-08-24** : ce poste n'est
> plus « le seul **jamais** attaqué » — D1 l'a entamé, sans `k` et pour
> 0,24 $. Conservé pour que la correction se lise là où la promesse a été
> écrite.
>
> ✅ **Deux conséquences, et la seconde est une bonne nouvelle.** (i) Le
> discours change : on n'« amortit » plus une fatalité, on **corrige une forme
> qu'on a choisie** — ce qui rend le poste attaquable par d'autres voies que
> `k`. (ii) **D1 en a déjà repris une part, le 2026-08-24, sans écrire un seul
> noyau `k`** : concaténer `q/k/v` et `gate/up` **par lignes** fait passer le
> chemin servi de **252 à 144 matvec par token** et rend **×1,061
> [1,050–1,069]** à `ROT_SHARE` constant — six critères pré-enregistrés verts,
> 128 tokens identiques entre bras fusés, divergence au dense **au même token
> 89**, **+3 686 400 octets exactement** (+0,008117 b/poids), **même sha256
> NVRTC** des deux côtés. Décomposition mesurée : **87,0** (servi publié) →
> **94,9** (hissage de la rotation seul) → **100,6 tok/s [99,9–100,7]** (plus
> la fusion)
> ([`mesures/d1-fusion-servie-2026-08-24.txt`](mesures/d1-fusion-servie-2026-08-24.txt),
> 0,24 $).
>
> ⚠️ **Ce que D1 ne fait PAS, et qui laisse `k` entière** : il **réduit le
> nombre** de lancements à `k = 1`, il n'amortit pas le plancher **sur
> plusieurs colonnes**. Les deux leviers se composent en principe et **n'ont
> jamais été mesurés ensemble** ; un banc `k` qui ignorerait la fusion
> mesurerait un chemin qui n'est plus celui qu'on sert. ⚠️ Et le point de
> départ d'un futur verdict `k` **a bougé** : le servi de référence n'est plus
> 87,0 mais **100,6 tok/s** dès que `ROT_SHARE` et la fusion sont posés.

⚠️ **Le garde qui décide de la valeur produit, à poser avant d'écrire.** La
famille `k` n'amortit qu'à **`k` > 1**, donc en **prefill / lot** — c'est le
segment A5 « ≤ 60 s/document en lot de 8 ». À **`k` = 1**, le régime du chat
interactif (A5 « ≥ 20 tok/s »), **le plancher reste entier** et cette famille
n'y change rien. Un verdict de `k` ne se transporte donc pas au débit
interactif, et le dire d'avance évite de republier un gain de banc comme un
gain produit.

### Ce qui doit être écrit avant tout chronomètre (0 $, sur Mac)

- [ ] **`ots stamp` sur [`preregistration-p4-2026-08-14.md`](../proofs/preregistration-p4-2026-08-14.md)**
      — exigence du prereg lui-même : *avant le premier noyau `k`, et avant le
      go de dépense*. Il n'a aujourd'hui aucun tampon.
- [ ] **Consigner au §7bis** (resté vide) les deux dérogations déjà commises :
      le run `nullk` du 08-16 était une réduction du plan (un bras sur les
      **huit** que le §2.5 liste — le §4.4 dit « au moins huit bras » —,
      2 phases au lieu de 3, contrôle sans `golay70v2`), et `e1c14` — listé au
      §2.5 comme bras à écrire — a été enterré depuis.
      ⚠️ **Une contradiction à trancher rétroactivement** : le §2.13 pose que
      les rounds de fusion « ne sont ni lus ni publiés en P4 », et le journal
      `nullk` publie la table de fusion avec ses verdicts chiffrés. Si ce job
      n'était pas P4, le plancher non plus n'a pas été produit sous protocole
      P4 — les deux lectures ne tiennent pas ensemble.
- [x] ✅ **2026-08-19 — `LLVQ_TIME_EVENTS` est livré et a tourné** (F3,
      0,86 $) : chronométrage par events CUDA, en rounds séparés. **K1 et K2
      ont désormais leur instrument.** Ce qu'il a rendu au passage est un
      résultat en soi : l'écart **hôte − device vaut 0,1–0,2 %** (4–8 µs par
      round), **deux ordres de grandeur sous l'attente** — la soumission hôte
      est **entièrement recouverte**, ce qui **affaiblit** l'hypothèse « le
      poste latence, c'est l'hôte » **sans la réfuter**
      ([`mesures/f3-events-2026-08-19.txt`](mesures/f3-events-2026-08-19.txt)).
      ❌ **Et ce que la plateforme refuse** : `ncu` s'installe et s'attache,
      mais les compteurs sont interdits (`ERR_NVGPUCTRPERM`). C'est déclaré
      comme **fait de plateforme** et F3 est clos **sans retentative** — donc
      un profil d'occupation ne fait pas partie des instruments disponibles,
      et aucun verdict de `k` ne doit en dépendre. Driver **580.159.03**
      capturé (la dette du 08-05 sur le champ `driver_version` est soldée).
- [x] ✅ **2026-08-18 — `cublasf16` a tourné, et le dénominateur tient** (F1,
      0,08 $) : `r = médiane(t_tv_f16 / t_cublasf16)` formé **round par
      round** (les deux bras coexistent, le rapport par round est licite ici)
      vaut **1,024** en phase 2 bras et **1,015** en phase 5 bras, **≤ 1,05**
      — bande 1 du prereg. **Notre témoin FP16 maison est au niveau de cuBLAS
      sur L40S**, donc **tous les « vs FP16 » publiés tiennent** et P4 a le
      dénominateur que le papier attendait
      ([`mesures/f1-cublasf16-2026-08-18.txt`](mesures/f1-cublasf16-2026-08-18.txt)).
      ⚠️ **C'est un verdict L40S** : sur A100, F4 mesure le témoin maison à
      **1,14× de cuBLAS** — autre carte, autre bande, et ça ne contredit rien.
- [ ] **`mvkf16`** (contrôle §4.3) et extension nominative du gate zéro-spill
      aux six sites par bras (§2.12) — un site oublié panique après ~25 min
      déjà payées.
- [ ] **La famille elle-même** : `planes14k` à `k ∈ {1,2,4,8}`, accumulateurs à
      **grille inchangée** (§2.6), `TILE_BLOCKS_K = 32` partout (§2.8), une
      seule unité NVRTC et un seul sha256 (§2.7), **`nullk` dispatché à chaque
      `k` dans les mêmes rounds** (§2.9.3) — c'est lui qui rend le verdict
      lisible, puisque c'est le plancher qu'on prétend amortir.

### Les seuils, et les deux qui étaient faux

| | seuil |
|---|---|
| **K1** | se lit sur le **rapport vs FP16**, jamais sur le temps — ⚠️ il était écrit à l'envers dans le brouillon, et sur les médianes du 08-11 les deux formes concluent l'inverse l'une de l'autre |
| **K2** | **par colonne** : `T(k=8)/8 ≤ 0,60 × T(k=1)`, soit `T(k=8) ≤ 4,80 × T(k=1)` — ⚠️ la forme héritée (`T(k=8) ≤ 0,60 × T(k=1)`) est **arithmétiquement impassable** : un noyau à `k=8` fait 8× les FMA et 8× les stores |
| **K3** | zéro spill sur tous les sites |

### Le job, et son budget corrigé

- [ ] Job **mutualisé** (famille `k` + éventuellement le bras `e1c12`) :
      **0,8-1,0 $**, pire cas 2,70 $. ⚠️ Le « 0,3-0,5 $ » qui circule est faux
      d'un facteur 2 : tout job `planesbench` à 5 bras ou plus paie
      **1 468-1 481 s de transcodage hôte** avant le premier round, et il ne se
      réduit pas en désélectionnant (le transcode `Slot32` est inconditionnel).
      **`--timeout 90 m` à poser explicitement** — le défaut de 30 min tuerait
      le job pendant le transcodage. **Go de dépense requis.**
- [ ] **Expliquer d'abord la dérive inter-run de 1-1,5 %** (`Planes12x` 2,01
      contre [1,95–1,99], `Golay70` v1 1,34 contre [1,29–1,32]) — l'intra-run
      est de 0,13 %. Aucun verdict n'en dépend aujourd'hui ; un run publiant à
      1 % près devrait la comprendre.

### Ce qu'il faudra faire du résultat

- [ ] **Refaire l'attribution PAR TOKEN.** Les 45,2 % (252 projections) et les
      39 % (2,04 ms/token) ont **deux dénominateurs** ; les rapprocher demande
      de refaire la mesure, **pas de reporter un nombre**. Tant que ce n'est pas
      fait, aucun chiffre de plancher ne se cite au niveau produit — et le
      bout-en-bout reste par ailleurs plafonné ~1,28× par l'overhead de
      lancement.
      ⚠️ **Deux amendements du 08-24/25, et ils vont en sens contraire.**
      (i) Le mot « plancher » ne se cite plus **du tout** sans dire qu'il est
      celui de **notre géométrie** (F2). (ii) Le « ~1,28× d'overhead de
      lancement » est un plafond **hérité et jamais remesuré depuis** que le
      nombre de lancements a changé : **D1 en a supprimé 108 sur 252** et rend
      ×1,061 sur le servi. Le plafond n'est donc pas réfuté, il est
      **périmé dans son dénominateur** — le remesurer fait partie de cette
      case, et il est moins cher qu'avant : l'instrument (`LLVQ_TIME_EVENTS`)
      existe depuis F3.

## Ouvert par P1 (2026-08-15) — le bras CUDA de P4, et P5

**Ce qui est acquis**, sur pré-enregistrement horodaté avant la mesure
([`mesures/p1-rankbench-2026-08-15.txt`](mesures/p1-rankbench-2026-08-15.txt)) :
`marche-binomiale` **0,3101 ns/bloc** ✅ (seuil 1,50), `cascade-uniformisée`
**1,7809** ✅ (seuil 2,00), `cascade-archive` **10,8115** ❌ (seuil 2,00).
Ancres `sol` 0,0777 et `masques` 0,1486, reproduites à quelques pour cent du
run `decreal` du 08-01.

**Les deux portes que ça ouvre, et elles ne s'ouvrent pas ensemble par hasard :**

- 🚨 **Le bras cascade/marche de P4 N'EST PAS autorisé — retiré le soir même
  par P1b.** Le 0,3101 ≤ 0,45 décrivait **une marche de 24 créneaux, pas un
  bloc** ; un bloc rend **0,6735 ns** et le gate est à 0,45. Régime
  intermédiaire du §4.2 : *le bras survit comme point de la courbe et n'achète
  AUCUN bras CUDA — il faut une idée neuve, pas un job.* Ce qui suit reste vrai
  du jour où le gate serait franchi : ⚠️ *autorisé* ne serait pas *lancé*, le
  job carte restant soumis au go de dépense, et son budget réel est
  **0,8-1,0 $** et non les 0,3-0,5 annoncés
  (tout job `planesbench` à 5 bras ou plus paie 1 468-1 481 s de transcodage
  hôte avant le premier round). ⚠️ Et **trois bras de P4 n'ont toujours aucune
  ligne de code** : cuBLAS (le dénominateur publiable — `tv_f16` est maison et
  ne peut pas l'être), le noyau **E1c CUDA**, et le **support k colonnes** ;
  plus le chronométrage par forme via events CUDA, sans lequel K2 n'est
  attribuable à rien.
- **P5 s'ouvre** — la règle est « si et seulement si la **marche** passe
  0,45 », et c'est bien elle qui rend 0,3101. La cascade uniformisée, verte,
  n'y aurait pas suffi : c'est le seul cas où les deux règles divergent, et il
  s'est présenté à l'endroit prévu.

✅ **P5 est CLOS 4/4 depuis le 2026-08-15**, et ce paragraphe décrivait son RAF
d'avant la mesure. Ce qu'il demandait a été rendu :
[`mesures/p5-cns-2026-08-15.txt`](mesures/p5-cns-2026-08-15.txt), 0 $.

| jalon | rendu |
|---|---|
| **C1** largeur réalisée | 53,7370 bits/bloc → **2,3877 b/poids noyau** (2,3983 en coupe alignée ligne) |
| **C2** bijection | **150 681 600 blocs**, zéro écart, fixture origine comprise |
| **C3** forme du décodeur | **90 pas** au maximum contre 96 déclarés ; zéro division en source **et** en assembleur hôte |
| **C4** transcodage | **1,088× `Planes14`** [1,087–1,090] contre un seuil de 2,0 — le pronostic « côté 84 s plutôt que 404 s » est vérifié |

⚠️ **Et ce que P5 n'a jamais promis, c'est une vitesse de décodage** — le
document le dit lui-même. C'est l'extension CUDA qui l'a mesurée, et elle a
fermé la ligne (0,25×). Un format prouvé exact, borné en pas et transcodable
vite reste un format dont le décodeur peut être mort : **les quatre jalons de
P5 sont verts et la ligne est fermée**, sans contradiction.

**Ce que P1 ne dit pas, et qu'il ne faut pas lui faire dire.** Il mesure un
décodage **seul**, sur **Metal**, **un bloc par lane**, sans matvec, sans
réduction inter-lanes, sans tuilage. Le 0,45 ns du gate est une **inférence
inter-matériel** — un chiffre Metal qui autorise une dépense CUDA — et le §5 du
pré-enregistrement dit lui-même que c'est la partie la plus faible du
document : une marge de sécurité ×2 sur un jugement d'ingénierie, prise sur une
machine qui n'est pas la cible. Elle peut tuer un décodeur qui passerait sur
carte ; c'est le sens qu'on préfère.
