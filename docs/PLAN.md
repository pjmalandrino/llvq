# Plan d'actions — trois phases

> Issu de l'audit externe du 2026-08-12 (voir la dernière entrée de
> [`HISTORIQUE.md`](HISTORIQUE.md)). Trois phases **ordonnées par ratio
> information/coût**, chacune avec ses tâches, ses critères écrits d'avance
> et son coût.
>
> 🚨 **L'axe noyau n'est plus arrêté, et ce chapeau disait le contraire.** Il
> l'était sur les **layouts** (Golay70 v2 : 1,77× < 2,0× pré-enregistré, plus
> de piste à format inchangé), et il ne figure toujours dans aucune des trois
> phases ci-dessous. Mais le plan d'exécution P1→P7, validé par l'opérateur le
> 2026-08-13, l'a rouvert sur un **autre axe — le décodage du rang** — et P1 y
> a rendu son verdict le 2026-08-15 (voir la section en fin de fichier). Les
> trois phases restent la ligne principale ; P1→P7 est une ligne parallèle,
> avec ses propres pré-enregistrements dans [`../proofs/`](../proofs/).
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
×1,1894 de ppl, −6,85 pp apparié IC95 [+4,52 ; +9,12], écart AWQ 6,09 pp.

- [ ] Ajouter les lignes 14B à `data/echelle-4b-8b.csv` (ou un CSV dédié) et
      régénérer la figure d'échelle en **3 points**.
- [ ] Réécrire le récit d'échelle — abstract, intro, évaluation,
      limitations : « the gap halves » → le **genou** (fonte de l'excès de
      ppl −43 % puis −14 % ; écart AWQ 14,45 → 7,49 → 6,09 pp) ; « two
      points, not a law » → « three points, a knee, not a law ». La
      direction tient toujours ; la forme de la courbe change.
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
- [ ] 5,15 → **5,162** b/param (le verdict de `rtbits`, 08-09), ou garder
      5,15 avec son étiquette « affichage carte arrondi ».
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

- [ ] Ré-ancrer la version courante du prereg du 08-10 (nouvel .ots) avec une
      note d'historique honnête (trois éditions post-ancrage, lesquelles).
- [ ] Corriger `proofs/README.md` : ne revendiquer « vérifiable sans nous
      faire confiance » que pour le prereg du 08-11, qui le mérite.

### 1.6 `CLAUDE.md` et `README.md` — purge des affirmations démenties

- [ ] Retirer « l'écart au 4 bits fond deux fois plus vite » (2 occurrences),
      ajouter le 14B et le genou ; README : « two points » → trois.
- [ ] « 25 % de mémoire en moins (5,37 contre 5,96) » → **~11 %** (5,32
      contre 5,96) — dans `echelle-4b-8b` et `CLAUDE.md`.
- [ ] Bloc de commandes : `LLVQ_FUSED_LAYOUT` admet **4** valeurs, pas 2.
- [ ] Supprimer le paragraphe « SKIP: sur stderr » (motif éradiqué le 08-08).
- [ ] Baliser ✅/🚨 la couche fossile de fin de § 6 (« smoke en cours »,
      « étapes restantes » toutes faites, « deux points ouverts » tranchés).
- [ ] En-tête : donner les deux formulations de débit (×2,03 / ×1,12), et
      « qualité identique au bit près » → « mêmes tokens jusqu'au tie-break ».
- [ ] Divers : « quatre crates du cœur » → cinq ; ancres `fused.rs:68/:106`
      → `:89/:120` ; cellule G6 « close depuis le 08-07 » → re-close le
      08-11 ; Planes12x « pas dans le modèle » → « câblé, pas mesuré ».

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

**Objectif** : trancher si le genou du 14B est un palier ou une pause —
c'est le point qui décide de la thèse d'échelle, la seule échelle proche du
régime souveraineté (70B). **Coût : ~62 $ estimé le 08-03 (621 s/bloc
mesurés, bf16/C3 validé) ; budget avec marge : 80 $. Une nuit.**

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
      ≤ 4 pp → la thèse d'échelle survit au genou, v2 du papier en « courbe
      à 4 points » ; (ii) sinon → acter le palier, conclure le volet produit,
      le papier reste ce qu'il est : un papier systèmes avec un négatif
      propre.

### Livrables

Courbe d'échelle à 4 points, verdict produit final documenté dans
[`HISTORIQUE.md`](HISTORIQUE.md), et la décision de soumission (MLSys) prise
sur pièces.

---

## Ouvert par P3 (2026-08-15) — le KV q8 à contexte long

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

**Ce que P5 demande maintenant, dans l'ordre** : d'abord la **décision de
passation** de rouvrir la clause « profondeur ≤ 24 » du spec X4 — le binaire
classe déjà e1v en réouverture, pas en résultat de banc ; puis la
**ré-bijection CNS** au transcodage (champs séparés par étage, 53,332
bits/bloc) ; puis le sweep intégral contre le décodeur d'archive, et le
chronométrage du transcodage (attendu côté 84 s de `Planes14` plutôt que 404 s
de `Planes12x`, E1v étant un re-rangement sans recherche réseau — **à
vérifier, pas à affirmer**).

**Ce que P1 ne dit pas, et qu'il ne faut pas lui faire dire.** Il mesure un
décodage **seul**, sur **Metal**, **un bloc par lane**, sans matvec, sans
réduction inter-lanes, sans tuilage. Le 0,45 ns du gate est une **inférence
inter-matériel** — un chiffre Metal qui autorise une dépense CUDA — et le §5 du
pré-enregistrement dit lui-même que c'est la partie la plus faible du
document : une marge de sécurité ×2 sur un jugement d'ingénierie, prise sur une
machine qui n'est pas la cible. Elle peut tuer un décodeur qui passerait sur
carte ; c'est le sens qu'on préfère.
