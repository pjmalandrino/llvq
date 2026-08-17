# Plan d'actions — trois phases

> Issu de l'audit externe du 2026-08-12 (voir la dernière entrée de
> [`HISTORIQUE.md`](HISTORIQUE.md)). Trois phases **ordonnées par ratio
> information/coût**, chacune avec ses tâches, ses critères écrits d'avance
> et son coût.
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
> Combiné au plancher du 2026-08-16 (tout travail de format plafonné à 4,77×,
> `Planes14` en capturant déjà 2,16×), **il n'y a plus de raison de chercher un
> format** : ni pour la vitesse, ni pour la mémoire.
>
> ⚠️ **Et il faut dire de quoi chacune des deux moitiés est faite, parce
> qu'elles n'ont pas la même force.** *Mémoire* : un **compte** sur le
> portefeuille existant — aucun de ces layouts ne passe, ce qui ne borne pas un
> format futur. *Vitesse* : le plancher ne dit **pas** que la vitesse est
> épuisée — il laisse 2,16× → 4,77× de marge nominale sur le terme de trafic.
> Ce qui la referme est une **induction sur trois points mesurés**, pas une
> borne : `Golay70` v1 (1,31×), v2 (1,77×) et E1v (0,25×) réduisent tous les
> octets et sortent tous **plus lents**, bornés en calcul. Une idée neuve sur
> le coût ALU rouvrirait la moitié vitesse ; aucune n'est connue.
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
> **(1) Phase 1** — le point 14B au papier, 0 $, seul bloquant du papier ;
> **(2) Phase 2** — la qualité, le seul axe dont une découverte change le
> verdict produit à échelle fixée ;
> **(3) la famille `k` de P4 §2.6** — le poste des 45 % que le plancher désigne
> et qu'aucun format ne touche ;
> **(4) Phase 3** — le 32B, après que 1 soit livrée et 2 tranchée.
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

## Phase 1 — Papier v2 et solde des dettes de cohérence

**Objectif** : que plus aucune surface publiée (papier, README, `CLAUDE.md`,
`proofs/`) ne porte une affirmation que le dépôt lui-même a démentie.
**Coût : 0 $ (option +~3 $). Durée : 2-3 jours. Aucun risque.**

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
      **Le genou ne survit pas au test** : la chute de l'écart vaut −6,96 pp du
      4B au 8B (SE 1,82, p = 0,0001, **résolue**) et −1,40 pp du 8B au 14B
      (SE 1,68, p = 0,40, **NON résolue**). Écrire « a knee » publierait comme
      résultat un ralentissement que les barres ne séparent pas.
      ⚠️ Et « no knee » serait tout aussi faux : p = 0,40 ne prouve pas
      l'égalité, les données sont **muettes** sur ce palier.
      **La formule à écrire est « three points, not a law »**, avec les trois
      IC et les deux tests de palier — la direction tient (4B→14B : −8,36 pp,
      p ≈ 1e-5), sa *forme* reste indéterminée, et **le 32B est ce qui
      trancherait**.
      ⚠️ Côté perplexité, ne pas opposer « −43 % » et « −14 % » : le premier
      **n'est pas barrable** (journal 4B de synthèse, pas de NLL par fenêtre),
      le second vaut −13,9 % IC95 [−22,8 ; −4,9] sur f16 et **−1,58 %
      [−3,14 ; −0,004]** sur AWQ — ce dernier exclut zéro **de 0,005**, donc
      **ne jamais écrire « the gap closes significantly »**
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
- [ ] ⏳ **Quatre preregs sans aucun tampon** : `2026-08-13`, `p2`, `p3`, `p4`.
      Pour `p2` et `p4`, la mesure n'a pas eu lieu — `ots stamp` reste possible
      et **doit précéder** le premier noyau `k` (exigence du prereg P4 lui-même).
      Pour `08-13` et `p3`, la mesure a eu lieu : seule une déclaration de dette
      dans le document répare quelque chose.

### 1.6 `CLAUDE.md` et `README.md` — purge des affirmations démenties

- [x] ✅ **2026-08-16** — « l'écart au 4 bits fond deux fois plus vite »
      retiré, le 14B et le genou ajoutés aux tables de `CLAUDE.md` (§3ter,
      §3bis). ⏳ **Reste `README.md`** : « two points » → trois.
- [x] 🚨 **2026-08-17 — le genou ajouté la veille est RETIRÉ**, et de toutes
      les surfaces vivantes (`CLAUDE.md` §3ter/§3bis/§6, `HISTORIQUE.md`,
      `echelle-4b-8b`, ce fichier, `README.md`, `cheatsheet-defense`). Motif :
      les trois écarts AWQ − LLVQ étant enfin appariés, la chute d'un palier au
      suivant se teste — **4B→8B résolue (p = 0,0001), 8B→14B NON résolue
      (p = 0,40)**. Le ralentissement n'est pas séparé par les barres. ⚠️ Et
      p = 0,40 ne prouve pas l'égalité : les données sont **muettes**.
      🕳️ Illustration exacte de la règle du §5 de `CLAUDE.md` — la correction
      du 08-16 était juste sur ce qu'elle retirait et a introduit, dans le même
      geste, une affirmation non testée.
- [x] ✅ **2026-08-17** — « la paire `AWQ − LLVQ` n'existe pas au 14B / ne
      jamais citer 6,09 avec un intervalle / la recalculer exige de refaire la
      campagne » retiré partout : elle existe, pour 0 $
      (+6,09 [+3,62 ; +8,52], McNemar 1,143e-11).
- [x] ✅ **2026-08-17** — ligne mémoire du 14B ajoutée (**5,106 vs 5,404,
      −5,5 %**), avec la **non-monotonie** de la marge et son mécanisme (part de
      l'embedding), et intervalles de perplexité 8B/14B, le 4B **visiblement
      sans**.
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

---

## Phase 2 — Qualité : attaquer le mécanisme du déficit MMLU

**Objectif** : tester les deux seuls leviers qui visent le *mécanisme* du
déficit (le raisonnement s'effondre, la restitution tient — l'oracle de
calibration ne borne que la perplexité, donc la piste volume est morte mais
pas celles-ci). **Coût : ≤ 10 $ GPU + jours d'ingénierie. C'est le seul axe
dont une découverte change le verdict produit.**

### 2.1 Composition du corpus de calibration (quasi gratuit, en premier)

Hypothèse : calibrer sur un corpus pondéré raisonnement (maths, code,
chaînes de déduction) déplace ce que GPTQ préserve, là où C4 sur-représente
la restitution.

- [ ] Construire 3 bras de calibration à 131 k tokens, même graine : C4 pur
      (contrôle = l'existant), C4+raisonnement 50/50, raisonnement pur.
- [ ] **Écrire le critère avant** : +2 pp de MMLU micro 4B vs contrôle, en
      apparié (`mmlupair`), hors σ McNemar. En dessous : piste close, comme
      l'oracle.
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

**Objectif** : trancher **si** la courbe d'échelle s'aplatit — c'est le point
qui décide de la thèse d'échelle.
🕳️ **Cet objectif disait « trancher si le genou du 14B est un palier ou une
pause », et il présupposait le genou.** Retiré le 2026-08-17 : la chute de
l'écart au 4 bits est **non résolue** entre 8B et 14B (1,40 pp, SE 1,68,
p = 0,40), donc il n'y a pas de genou établi à qualifier — et p = 0,40 ne dit
pas non plus qu'il n'y en a pas. **Le 32B en devient plus décisif, pas moins** :
il est le seul point qui puisse séparer les deux lectures.
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

1. Phase 1 livrée (on ne repaie pas une campagne pour un papier incohérent).
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
> Et le plancher borne son gain de vitesse : tout format plafonne à 4,77×,
> `Planes14` en capture déjà 2,16×, son décodage ne coûte que ~7 % du trafic.
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

`nullk` — même grille, même tuilage, même staging, même épilogue, **aucun poids
lu** — rend **2,305 ms** contre 5,102 pour `Planes14` :
[`mesures/nullk-plancher-2026-08-16.txt`](mesures/nullk-plancher-2026-08-16.txt),
job `6a81b2b71f5885ae605bdcc9`, **0,77 $**.

**Le plancher est 45,2 % du bras servi**, et il n'était jusqu'ici qu'un reste
obtenu par soustraction.

| | |
|---|---|
| plafond absolu de tout travail de **format** | **4,77×** FP16 (= FP16 / plancher) |
| où `Planes14` en est | **2,16×** |
| ce que le format achète, **net** du plancher | **3,11×** (8,691 ms de trafic contre 2,797) |
| coût du décodage de `Planes14` | **~7 %** du temps de trafic (779 Go/s net contre 836 pour du FP16 pur) |

🚨 **Ce que ça dit de quatre tentatives** : le format n'a que 55 % du temps à
disputer, `Planes14` en capture déjà l'essentiel, et 45 % ne sont touchés par
**aucun** format. Le plancher est le poste le plus gros et le seul jamais
attaqué — c'est exactement ce que la famille **k** de P4 §2.6 existe pour
amortir, et elle n'est pas écrite.

⚠️ Ce n'est **pas** les « 39 % » de l'attribution du 05-08 : celle-ci découpe
2,04 ms par **token**, normes et attention comprises. Deux dénominateurs.

---

## Priorité 3 — La famille `k` (P4 §2.6) : le seul poste jamais attaqué

**Ce que le plancher désigne, et que rien n'a jamais visé.** 45,2 % du temps
d'une passe n'est touché par **aucun** format. La famille `k` — le même noyau
servant `k` colonnes par lancement — est le seul levier écrit qui l'amortisse :
le plancher se paie **une fois pour `k` colonnes** au lieu d'une fois par
colonne. **Coût : code sur Mac (0 $), puis un job mutualisé 0,8-1,0 $.**

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
- [ ] **Chronométrage par forme via events CUDA**, en rounds séparés (§2.10) :
      **sans lui, K1 et K2 ne sont ni verts ni rouges**.
- [ ] **`cublasf16`** — le dénominateur publiable. `tv_f16` est maison et ne
      peut pas l'être : **sans `cublasf16`, aucun chiffre de P4 n'entre au
      papier**. Imprimer la version de cuBLAS (§2.15).
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
