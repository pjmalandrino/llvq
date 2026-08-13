# Plan d'actions — trois phases

> Issu de l'audit externe du 2026-08-12 (voir la dernière entrée de
> [`HISTORIQUE.md`](HISTORIQUE.md)). Trois phases **ordonnées par ratio
> information/coût**, chacune avec ses tâches, ses critères écrits d'avance
> et son coût. L'axe noyau est **formellement arrêté** (Golay70 v2 :
> 1,77× < 2,0× pré-enregistré, plus de piste à format inchangé) — il ne
> figure dans aucune phase.
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
