# Historique — le fil chronologique du projet

> **Ce fichier est l'unique document d'historique.** Une entrée par période,
> le verdict, les chiffres qui font foi, et les liens vers les pièces.
>
> Le système documentaire tient en trois strates :
>
> 1. **Ce fichier + `PLAN.md`** — le fil (passé) et la suite (futur). Ce sont
>    les deux seuls documents de narration qu'on met à jour.
> 2. **[`archive/`](archive/)** — les documents d'époque : passations,
>    verdicts, specs, audits, plans de session. Conservés intégralement,
>    **plus jamais édités**. ⚠️ Un document d'archive peut contenir des
>    affirmations démenties depuis — c'est ici et dans `CLAUDE.md` que vit la
>    vérité courante, l'archive ne vaut que comme généalogie.
> 3. **[`mesures/`](mesures/) + [`data/`](data/)** — les faits bruts :
>    journaux de runs datés et CSV. Jamais modifiés, cités par le papier.
>
> Restent vivants dans `docs/` les documents de **référence** (mis à jour
> quand l'objet qu'ils décrivent change) : [`fiche-4b.md`](fiche-4b.md),
> [`campagne-finale-2026-08-07.md`](campagne-finale-2026-08-07.md),
> [`echelle-4b-8b-2026-08-08.md`](echelle-4b-8b-2026-08-08.md),
> [`format-noyau.md`](format-noyau.md),
> [`llvq-paper-notes.md`](llvq-paper-notes.md),
> [`llvq-rust-implementation-plan.md`](llvq-rust-implementation-plan.md),
> [`inference-cost-reduction-2026.md`](inference-cost-reduction-2026.md),
> [`cheatsheet-defense.md`](cheatsheet-defense.md),
> [`hf-model-card.md`](hf-model-card.md).
>
> **Règle d'entretien** : une entrée nouvelle s'ajoute en bas, datée, avec ses
> verdicts étiquetés (✅ acquis / ❌ réfuté / ⚠️ dette). On n'édite pas les
> entrées passées — si un verdict se retourne, on l'écrit dans une entrée
> nouvelle qui cite l'ancienne.

## État courant (au 2026-09-02)

🆕 **LA ROADMAP RECHERCHE EST ADOPTÉE (D0, 2026-09-02)** — trois axes (mesure,
format sans dépliage, qualité), gates et kills posés d'avance :
[`ROADMAP-RECHERCHE.md`](ROADMAP-RECHERCHE.md), fondée sur
[`audit-recherche-2026-09-01.md`](audit-recherche-2026-09-01.md) et
[`projection-gains-2026-09-01.md`](projection-gains-2026-09-01.md). **Vague 1,
plafond 5 $ : M2 d'abord** (attribution de la chute MMLU par type de projection
— un A/B à fichier constant, SE appariée 0,43 pp, indépendant du σ de
calibration), **M1 en parallèle sur le Mac** (shrinkage hors-diagonale de H,
0 $). Les deux boutons sont livrés (`LLVQ_RESTORE_F16`, `LLVQ_H_SHRINK`), les
deux préregs sont **en brouillon, non tamponnés**, rien n'a tourné. ⚠️ La
projection de F1 place F1b **sous son propre kill** (rétention 88,9-89,6 %
contre 90,3 %) : F1a doit compter avant de coder (journal de la roadmap).

❌ **LE PAPIER A ÉTÉ RENVOYÉ SANS REVUE PAR TACO LE 2026-08-27** (`TACO-2026-428`,
motif : le périmètre — aucune revue technique, aucun chiffre contesté ;
`paper/main.tex:4-7`, commit `209a695`) **et il est DÉPOSÉ EN PRÉPRINT le soir
même** : Zenodo, DOI de concept **`10.5281/zenodo.22133606`**, v1
`10.5281/zenodo.22133607`. La venue suivante n'est **pas choisie** — le
manuscrit est « entre deux venues » (`alignement-v1-restant-2026-08-31.md`).

✅ **LA CONFIG SERVIE v1 EST GELÉE AUX TROIS TAILLES depuis le 2026-08-31** :
`planes14 + LLVQ_EMBED=q8 + LLVQ_ROT_SHARE=1 + LLVQ_FUSE=1` — **100,6
[99,9–100,7] / 75,5 [75,5–75,6] / 46,8 [46,7–46,8] tok/s dans 2,57 / 5,41 /
9,40 Go** (*mesuré*, [`mesures/vague2-fusion-8b-14b-2026-08-31.txt`](mesures/vague2-fusion-8b-14b-2026-08-31.txt) ;
la cellule 4B vient de D1). Fusion dans la bande tamponnée aux trois tailles
(×1,061 / ×1,055 / ×1,028) **et sur A100** (×1,063). ⚠️ Ces tok/s se disent
toujours en deux formulations — le dense au dénominateur est *notre* bras
handicapé — et **la série à tête identique ×1,11 → ×1,29 → ×1,41 (B2,
ROT_SHARE=0/FUSE=0) reste LA série qui mesure le noyau : elle n'est PAS
re-mesurée sous v1.**

✅ **La vague 2 est COMPLÈTE : 2,17 $ sur un plafond de 5** (*mesuré*,
[`data/jobs.csv`](data/jobs.csv)). A1 : `r = t(144)÷t(252) = 0,8158
[0,8150–0,8162]` — bande **mixte**, ni latence pure ni occupation pure, ni A2
ni A3 éliminé. A4 : **aucun bras réseau ≥ 1,00× FP16 sur A100** (`Planes14`
0,79×, F4 reproduit au centième) — la réserve Ada du papier est un **mécanisme
mesuré à deux points** ; et `r` est **invariant entre cartes** (0,8198, temps
×1,809 ≈ horloges 1,787).

⚠️ **Le point dur reste la QUALITÉ, mais il a désormais un étalon concurrent** :
IQ2_XXS, premier 2 bits mesuré chez nous, rend **+16,20 pp [+12,64 ; +19,72]**
de MMLU en notre faveur à 0,37 % de bits d'écart — et **nous bat du double en
mémoire servie** (2,479 contre 5,162 b/param). Le bruit inter-graines du MMLU
vaut **σ = 2,92 pp** au 4B — l'échelle de volume de calibration **ne part pas**.

✅ **LA PHASE A (GÉOMÉTRIE) EST RENDUE ET CLOSE, 1,11 $ sur un plafond de 4,
SANS RIEN CHANGER AU CHEMIN SERVI** (entrée du 09-01 → 09-02). **A2, CUDA
Graphs en forme hybride : +13,45 % [13,36–13,58] au 4B, +10,1 % au 8B,
+6,1 % au 14B** — les trois résolus, deux au-dessus du seuil d'adoption.
**A3, huit variantes d'occupation : aucun bras portable ne passe le gate
banc** (le meilleur, `pers`, +1,56 % [+1,01 ; +1,86]) ; le bras de banc
`persall` **borne** la géométrie à +26,36 % [+25,31 ; +26,61] du matvec
fusé, ce que les graphs ont déjà encaissé. ⚠️ Le sous-remplissage de o/down
n'est **pas** le résidu (split-K −1,87 %). Kill de phase non déclenché.

🚨 **ET A2 N'EST PAS SERVI — décision d'opérateur du 2026-09-02, tranchée
par la MÉMOIRE et non par le débit** : la fenêtre KV que la capture exige
coûte **+47 % de VRAM au 4B à la cible produit de 8k pour +12,6 % de débit**
(*calculé*, jamais mesuré au-delà de `prealloc(256)`). Refusé sur l'axe même
où vit la thèse. **La config servie v1 est inchangée, le cœur intact**, et
les mesures restent publiées telles quelles. Ce qui est refusé est le couple
(graph, fenêtre 8k), pas le graph : à 2k l'arbitrage s'inverse. Conditions
de réouverture au §É7 des écarts.

🧭 **Prochain chantier : la QUALITÉ (phase B du plan d'après-dépôt)** — la
géométrie sous candle a rendu ce qu'elle pouvait, et ce qu'elle rendait
encore se paie en mémoire.

**Compteurs au 2026-09-02** (*mesurés*) : **102 lignes** dans
[`data/jobs.csv`](data/jobs.csv) pour **92,51 $ facturés** (`awk` sur le CSV) ·
**96** journaux `.txt` dans [`mesures/`](mesures/) · **13** CSV dans
[`data/`](data/) · **40** documents et **28** ancrages `.ots` dans
[`../proofs/`](../proofs/) — **20 ancrés Bitcoin, 8 en attente de calendrier**
(*mesuré* le 09-02 par `ops/otsaudit.py`,
[`mesures/ots-etat-2026-09-02.txt`](mesures/ots-etat-2026-09-02.txt) : les
huit sont m3, vague 2, protocole v2, et les cinq préregs A2/A3 du 09-01 ;
les deux du 08-10/08-11 ne collent toujours pas à leur fichier).
🕳️ **La fin de ce fichier (entrée du 08-24, §5) répète « aucun des 16 `.ots`
n'a jamais été upgradé … 0 Bitcoin » : MESURÉ FAUX le 2026-08-26** — le grep ne
pouvait pas voir les ancres (étiquette binaire de 8 octets), 16 des 20 tampons
en portaient déjà 3 ou 4 — cf. l'entrée du 08-25→27 et
[`mesures/ots-etat-2026-08-26.txt`](mesures/ots-etat-2026-08-26.txt). On
n'édite pas l'entrée passée ; c'est ici que vit le démenti.

## 2026-07-24 → 07-28 — Fondations : G1→G4, encodeur ×5,4

✅ Golay [24,12,8] + Λ₂₄ verrouillés par invariants exacts (kissing number,
série thêta, N(13) à 15 chiffres). Recherche NN exacte m ≤ 13, indexage
bijectif 48 bits, **G4 : 92,23 % de rétention** à 2 bits/dim. Encodeur
porté à 639 µs/bloc/cœur (Phase 2c).
✅ Piège résolu : l'extraction *texte* du PDF du papier est corrompue, le
rendu *image* est parfait — relecture intégrale transcrite dans
[`llvq-paper-notes.md`](llvq-paper-notes.md).
✅ Leçons de méthode fondatrices : tests létaux par mutation (l'étage Golay
neutralisé passait la suite), flip de parité — le balayage multi-types était
du code mort.
📄 [`archive/passation-2026-07-31.md`](archive/passation-2026-07-31.md)

## 2026-07-29 → 07-31 — G5 : premier 4B, et l'erreur de comptabilité

❌ Premier run 4B annoncé à 2,0653 bits/poids — **faux** : la magnitude libre
par bloc pesait 16 bits non facturés (réel : 2,7289). « Zéro bit de gain »
veut dire une constante par tenseur, pas un flottant par bloc.
✅ Fichier scellé **`leech1c12`** : **16,9617 de ppl à 2,1696 b/poids**
(QTIP : 17,04 à 2,000). G5 vert avec réserve (+8,5 % de bits).
📄 [`archive/retraction-et-gain.md`](archive/retraction-et-gain.md)

## 2026-08-01 — Audit A : le harnais validé, le déficit réel

✅ La chute MMLU publiée mélangeait macro et micro : en micro (= papier), la
baseline reproduit le papier à **0,22 pp** — harnais validé — et la chute
réelle est −14,33 pp (pas −15,26, pas les −9,5 du papier).
✅ Confondant dtype (ppl f32 / MMLU f16) mesuré **nul** (0,1 %).
✅ `bin/ppl` sait scorer le fichier scellé : 16,9415 en f16, ×1,3846.
📄 [`archive/audit-publication-2026-08-03.md`](archive/audit-publication-2026-08-03.md) (suites)

## 2026-08-02 → 08-04 — Premier 8B, dé-risquage 32B, renversement coquille

✅ 8B `leech1c12L3` sur GPU loué : ×1,267, 11,48 $ — premier signal d'échelle.
✅ 32B dé-risqué sur 4 blocs (5,43 $) : bf16 OK, 621 s/bloc → run complet
~62 $, pas 49.
❌ « La coquille unique bat l'union » — renversé le 08-04 : à débit empaqueté
égal, **l'union gagne** (0,0725 vs 0,0762 de MSE). La table fautive divisait
par un débit fractionnaire qu'aucun fichier ne paie.
📄 [`archive/passation-2026-08-04.md`](archive/passation-2026-08-04.md)

## 2026-08-05 — Lot K-1 : l'échelle bits↔vitesse, et le portage CUDA

✅ Banc Metal à 7 bras, un protocole, une comptabilité : `Slot32` 5,510
b/poids → 2,03× [2,03–2,10] ; la courbe est brutalement non linéaire.
✅ Les 4 pièges de mesure GPU documentés ([`format-noyau.md`](format-noyau.md)) ;
règle « une plage, pas un point » (2,029/2,050/2,080 sur le même binaire).
✅ Noyau de rotation CUDA écrit (15 mutants tués) ; attribution instrumentée
des 2,04 ms de Slot32 (latence 39 %, flux 33 %, décodage 19 %).
📄 [`archive/spec-lot-a-2026-08-05.md`](archive/spec-lot-a-2026-08-05.md) ·
[`archive/portage-noyau-cuda.md`](archive/portage-noyau-cuda.md) ·
[`archive/audit-perf-noyau-cuda-2026-08-05.md`](archive/audit-perf-noyau-cuda-2026-08-05.md)

## 2026-08-06 — Lot A : le noyau dans le modèle, et le verdict du 4 bits

✅ **`fusedrun` : le noyau tourne dans le modèle** — 47,0 tok/s / 3,28 Go
(dense : 43,5 / 8,04), 88 tokens gloutons identiques.
❌ **Sur un 4B, le 4 bits nous domine partout sauf le disque** : MMLU 70,04 %
(AWQ) contre 55,59 % (nous). C'est le verdict qui recadre tout le projet.
✅ **C1 gagné le même jour : `Planes14`** — plus petit ET plus rapide que
`Slot32` (4,804 b/poids, 1,14×), branché : 48,7 tok/s / 2,96 Go.
⚠️ Errata lot A : la comparaison mémoire se dit en **b/param modèle entier**
— jamais « 5,51 contre 4,50 » (deux dénominateurs, deux quatre-bits).
✅ Lot B (nuit) : σ inter-graines ≈ 0,7 % (première barre d'erreur), oracle de
calibration **plafonné** (−1,6 % : le run ×100 est enterré), damping nul,
embedding q8 validé sans perte, plafond L ≤ 4 **mort** (+4,75 % de ppl).
📄 [`archive/passation-lot-a-2026-08-06.md`](archive/passation-lot-a-2026-08-06.md) ·
[`archive/errata-rapport-lot-a-2026-08-06.md`](archive/errata-rapport-lot-a-2026-08-06.md) ·
[`archive/verdicts-lot-b-2026-08-06.md`](archive/verdicts-lot-b-2026-08-06.md)

## 2026-08-07 — La nuit des verdicts, et l'embedding q8 en production

❌ **Design C réfuté à pleine profondeur** : ×1,99 de ppl sur 28 blocs, gate
automatique, 0 $ de GPU. Confirme le motif : un proxy local meilleur prédit
une composition pire ; la rigidité de norme est porteuse.
✅ `Planes12x` (overlay épars) validé au banc : 4,342 b/poids, qualité exacte.
✅ **Embedding q8 livré : 88,4–88,5 tok/s / 2,60 Go.** Attribution du saut :
notre bras dense recopiait 778 Mo de vocabulaire par token
(`broadcast_matmul`, remonté amont : candle#3871). Donc **deux formulations
obligatoires** : ×2,03 brut, **×1,12 à tête identique**.
❌ E2 `Golay70` : 3,589 b/poids mais 1,31× < critère 1,6× posé d'avance —
écarté (v1). L'échelle des formats close (provisoirement).
📄 [`archive/verdicts-nuit-2026-08-07.md`](archive/verdicts-nuit-2026-08-07.md) ·
[`campagne-finale-2026-08-07.md`](campagne-finale-2026-08-07.md) ·
[`archive/rapport-etat-2026-08-07.md`](archive/rapport-etat-2026-08-07.md)

## 2026-08-08 — L'échelle 4B→8B, à une seule variable

✅ 8B requantifié en `leech1c12` (même config exacte que le 4B) : **×1,220 de
ppl, −10,56 pp de MMLU** (4B : ×1,385, −14,73). L'écart au 4 bits passe de
14,45 à 7,49 pp. Vitesse : 69,3 tok/s / 5,45 Go avec q8 (**sans q8 le 8B ne
renverse rien** — les tables déliées pèsent 2,49 Go en f16).
✅ Discipline de tests : `#[ignore]` inconditionnel pour les tests d'archive
(un skip silencieux est interdit), durées honnêtes documentées.
✅ Empreinte de codebook épinglée par test (`0x338f_420f_1186_6319`).
📄 [`echelle-4b-8b-2026-08-08.md`](echelle-4b-8b-2026-08-08.md)

## 2026-08-09 — Planes12x câblé, et le vrai chiffre à publier

✅ `Planes12x` câblé dans le modèle (transcodage ×4,8 vs `Planes14`) — mais
**pas payant à 8B** : la VRAM y est déjà gagnée. Câblé pour le 14B/32B.
⚠️ `rtbits` tranche : « le chiffre 4B q8 à publier est **5,162** b/param »
(le 5,15 en circulation est une citation d'affichage arrondi).
✅ L'étiquette « chemin candle » corrigée : le piège est dans la primitive,
la baseline handicapée est **la nôtre**.
📄 [`archive/reprise-14b-2026-08-09.md`](archive/reprise-14b-2026-08-09.md) ·
[`mesures/rtbits-planes-8b-2026-08-09.txt`](mesures/rtbits-planes-8b-2026-08-09.txt)

## 2026-08-10 — La mesure AWQ retourne le critère ; le 14B plie la courbe

✅ **AWQ porté dans notre banc** : 584 Go/s, 3,38×, **88 % de sa borne
d'octets — nous : 65 %**. Le critère de vitesse de E2 est périmé ; E2 rouvert
sur l'axe mémoire, nouveau seuil pré-enregistré : **2,0×**.
⚠️ Dette : le pré-enregistrement du 08-10 a été édité **après** son .ots sans
ré-ancrage (`ots verify` échoue sur le fichier courant).
✅ **Campagne 14B, appariée (`mmlupair`)** : ×1,189 de ppl, **−6,85 pp**
IC95 [+4,52 ; +9,12], écart AWQ 6,09 pp *(⚠️ différence nue — la paire
`AWQ − LLVQ` n'existe pas au 14B, donc ni IC ni McNemar, contrairement au
−6,85 qui précède)*. **La courbe a un genou** : fonte de
l'excès de ppl −43 % (4B→8B) puis **−14 %** (8B→14B). « L'écart au 4 bits
fond deux fois plus vite » est **retiré**. Et f16−AWQ au 14B est **non
résolu** (IC contient zéro) : « le 4 bits commence à payer » ne repose sur
rien de testé.
📄 [`archive/spec-apres-awq-2026-08-10.md`](archive/spec-apres-awq-2026-08-10.md) ·
[`archive/kernel-comparison-recon.md`](archive/kernel-comparison-recon.md) ·
[`mesures/campagne-14b-qualite-2026-08-10.txt`](mesures/campagne-14b-qualite-2026-08-10.txt) ·
[`mesures/six-arm-awq-2026-08-10.txt`](mesures/six-arm-awq-2026-08-10.txt)

## 2026-08-11 — Golay70 v2 : la réparation marche et ne suffit pas

❌ **v2 (coset hissé au niveau bloc, format inchangé) : 1,77× [1,76–1,78],
263 Go/s** — 1,32× sur la v1, mais **sous le seuil pré-enregistré de 2,0×**
(prereg du 08-11 : antériorité propre, commit 09:30, .ots 09:31, mesure
13:34). **Non adopté.** Plus aucune piste connue à format inchangé : l'axe
noyau est épuisé proprement.
✅ `golay70` câblé dans `fusedrun` (mesurable, pas servi) ; registre de
provenance ouvert ; lot D du papier fait.
📄 [`archive/projections-golay70-2026-08-11.md`](archive/projections-golay70-2026-08-11.md) ·
[`archive/passation-golay70-2026-08-11.md`](archive/passation-golay70-2026-08-11.md) ·
[`mesures/golay70-v2-sept-bras-2026-08-11.txt`](mesures/golay70-v2-sept-bras-2026-08-11.txt)

## 2026-08-12 — Papier dégraissé, audit externe, refonte documentaire

✅ Papier dégraissé (−16 % de mots), tag `paper-v1`.
🔎 **Audit externe complet** (relecture referee + traçabilité + croisement
docs/code, contre-expertise adversariale). Ce qui tient : ~40 chiffres du
papier retracés au chiffre près (le 22,83 $ recalculé exact), le code
confirme les docs partout, la chaîne Golay70 v2 est exemplaire, le papier est
jugé publiable (MLSys) et **pas verbeux**. Ce qui ne tient pas :
1. ❌ **Le point 14B est absent du papier, du README et de `CLAUDE.md`**
   alors qu'il plie la courbe que ces trois surfaces vendent (« the gap
   halves », « two points, not a law ») — et `echelle-4b-8b` § 1bis
   conditionnait « starts paying » à un rejeu apparié jamais fait.
2. ⚠️ Prereg 08-10 non ré-ancré ; « 25 % de mémoire en moins » faux (réel
   ~11 %) ; 5,15 vs 5,162 ; couche fossile non balisée en fin de `CLAUDE.md` ;
   bloc de commandes limité à 2 layouts sur 4.
3. Verdict stratégique : l'axe noyau s'arrête ; l'actif est le papier et la
   question qualité. → le plan en trois phases : [`PLAN.md`](PLAN.md).
✅ Refonte documentaire : 36 documents de session déplacés vers
[`archive/`](archive/), ce fichier créé comme unique historique, tous les
renvois (docs, README, `CLAUDE.md`, commentaires de code) mis à jour.

## 2026-08-12 (suite) — Lot X sur le Mac : E1c prouvé exact, E3 enterré sur papier

Trois verdicts, **0 $ et aucune carte** — le lot X livrait du code la veille
au soir ([`archive/spec-memoire-extreme-2026-08-12.md`](archive/spec-memoire-extreme-2026-08-12.md),
[`archive/runbook-lot-x-mac.md`](archive/runbook-lot-x-mac.md)), il fallait le
lancer.

✅ **X1/X2 — `E1c` est exact.** Les flux `Planes14`/`Planes12x` transposés sur
le groupe de 32 (le warp), ce qui fait disparaître le bourrage de stride :
**4,5551 et 3,7618 b/poids noyau** contre 4,8040 et 4,3424. Sweep intégral du
4B scellé, **150 681 600 blocs**, les deux variantes contre le décodeur
d'archive, 401 s
([`mesures/e1c-sweep-4b-2026-08-12.txt`](mesures/e1c-sweep-4b-2026-08-12.txt),
[`mesures/rtbits-e1c-4b-2026-08-12.txt`](mesures/rtbits-e1c-4b-2026-08-12.txt)).
Le terme d'exceptions d'`E1c12` est **identique** à celui de `Planes12x` :
l'écart entre les deux layouts *est* le bourrage, rien d'autre.
⚠️ **Aucune vitesse n'est mesurée** et la colonne doit rester vide jusqu'à la
carte — le noyau lirait 82 ou 106 mots par groupe là où `Planes14` en lit 4-5
par lane. Le banc X3 (~0,2 $) est désormais un **pur** test de vitesse.

❌ **X4 — E3 est enterré sur papier.** Le barreau qui visait le 24-32 Go en
décodant l'index dans le noyau : `bin/radixstudy` prix chaque décomposition
shift-only sur les blocs réels, le meilleur point vaut **3,0444 b/poids noyau
contre un critère de 2,60 posé d'avance** (+17 %), rouge dans les deux
comptabilités et sur les deux bras
([`mesures/radixstudy-x4-2026-08-12.txt`](mesures/radixstudy-x4-2026-08-12.txt)).
La raison ne se contourne pas : le point *dans* sa classe coûte déjà **41,50
des 47 bits** d'index, il ne reste ~5,5 bits pour le choix de classe, et toute
variante à champ explicite les repaie en 10 bits d'en-tête. **Le plafond
mémoire du projet est `E1c`**, et la ligne « K2.6 sur un poste » de l'étude
MoE tombe avec E3. Troisième barreau fermé par un critère écrit d'avance,
après E2 au banc et Golay70 v2 au pré-enregistrement.

⚠️ **Le devis MoE recalé, et l'estimateur avoue son biais.** Son selftest ne
rend compte que de **59 %** du run 4B réel ; `Qwen3-30B-A3B` est devisé à
5,74 $ sur `rtx-pro-6000`, donc ~10 $ au même ratio — un bras du gate X5-MoE
annoncé à 25-55 $, pas son total. Sur ce MoE, **63 % de l'artefact** est porté
en 16 bits : le piège d'embedding du 8B, en pire.

⚠️ **Le routage MoE mesuré, et il ouvre une question que le pipeline n'a
jamais posée.** `gpt-oss-20b`, 131 k tokens de C4, unité = la cellule
`(couche, expert)` puisque chaque expert a sa propre hessienne : **31,4 % des
768 cellules sont sous le rang plein**, et **une est morte** — zéro routage,
qu'aucun corpus ne ressuscite. Couvrir 90 % des cellules demanderait ×12 de
calibration (~+13 % de run seulement, l'encodeur étant proportionnel aux
poids), 99 % en demanderait ×166. **Le devis X5-MoE n'explose donc pas ; c'est
la politique « expert mort » qui manque**, notre pipeline supposant partout
une hessienne inversible. ⚠️ `gpt-oss` active 12,5 % de ses experts quand
`Qwen3-30B-A3B` en active 6,3 % et K2.6 2,1 % : ce tableau est un **plancher**
de difficulté ([`mesures/moe-routing-gptoss20b-2026-08-12.txt`](mesures/moe-routing-gptoss20b-2026-08-12.txt)).

⚠️ Dette de prose relevée : le runbook jugeait E3 sur la colonne *b/p moyen*
(2,73) quand le bin juge sur la colonne *groupé 32* (3,19) — même verdict des
deux côtés, mais deux nombres pour un seul critère.

Verdicts détaillés : [`archive/passation-lot-x-2026-08-12.md`](archive/passation-lot-x-2026-08-12.md).

## 2026-08-13 — Le rejeu apparié : la phrase du papier tient, mais pas dans les deux comptabilités

Phase 1.2 du plan, **1,30 $** (`oracle` 0,01 + 4B 0,49 + 8B 0,80, `l40sx1`).
Six bras MMLU rejoués avec dumps par question, **empreinte
`65dcd53655e8bfa5` sur les six** — et les six micros reproduisent les chiffres
publiés **au centième** (70,32 / 70,04 / 55,59 et 76,08 / 73,01 / 65,52). Le
harnais traverse trois mois sans dériver ; c'est le contrôle qui rend le reste
lisible ([`mesures/mmlupair-4b-8b-2026-08-13.txt`](mesures/mmlupair-4b-8b-2026-08-13.txt),
dumps dans [`data/mmlu-dumps/`](data/mmlu-dumps/)).

| paire (Δ = A − B) | 4B | 8B |
|---|---|---|
| f16 − AWQ | **+0,27 pp [−1,63 ; +2,13] NON RÉSOLU** | **+3,07 pp [+1,61 ; +4,69] résolu** |
| f16 − LLVQ | +14,73 pp [+11,98 ; +17,47] | +10,57 pp [+8,58 ; +12,57] |
| AWQ − LLVQ | +14,45 pp [+11,60 ; +17,27] | **+7,49 pp [+5,28 ; +9,70]** |

✅ **« The 4-bit baseline starts paying » est testé, et il tient.** À 4B
l'écart f16↔AWQ n'est pas résolu ; à 8B il l'est. C'est exactement la phrase,
et `echelle-4b-8b` §1bis peut lever sa réserve.

⚠️ **Mais elle dépend de la comptabilité, et il faut le dire.** À 4B, le
contrôle **non pondéré** résout ce que le micro stratifié ne résout pas
(+1,97 pp [+0,92 ; +3,02]). Le désaccord est porté par `professional law`
(poids 10,9 %, −10,0 pp) : dans la comptabilité publiée l'AWQ est
indiscernable du f16 à 4B, dans l'autre il perd déjà. Écrire la phrase **avec
son statistique**, pas comme un fait nu.

✅ **Le resserrement de l'écart au 4 bits est mieux fondé qu'avant** :
14,45 → 7,49 pp, et les deux IC95 **ne se recouvrent pas**. « The gap halves »
sort renforcé de son premier test.

⚠️ **L'axe f16, lui, ne suit pas** : les IC de f16 − LLVQ **se recouvrent**
(14,73 [11,98 ; 17,47] contre 10,57 [8,58 ; 12,57], recouvrement
[11,98 ; 12,57]). La fonte du déficit est donc solide **face au 4 bits** et
non résolue **face au f16** — asymétrie à porter dans le papier plutôt qu'à
lisser.

⚠️ Aucun de ces intervalles ne teste la *différence des différences* entre
échelles : `mmlupair` apparie deux bras sur les mêmes questions, il n'apparie
pas deux tailles de modèle. Non-recouvrement d'IC ≠ test formel.

## 2026-08-14/15 — Le cache KV à 8,5 bits ne coûte pas de qualité, et il n'est pas servi pour autant

P3 du plan d'exécution, **0 $, ~2 h 45 de Mac**. Cinq pré-enregistrements
posés dans la journée (P1 amendé trois fois, P2 à P5 écrits puis réécrits
contre une revue adversariale qui avait trouvé 18 bloquants), puis P3 mesuré
de bout en bout. Journal :
[`mesures/kvq8-4b-2026-08-15.txt`](mesures/kvq8-4b-2026-08-15.txt).

Le contrôle d'abord, parce qu'il rend le reste lisible : le bras f16 rend
**ppl 16,9415** sur l'empreinte `3f1baca9033bf251` et **MMLU 56,09 %** sur
`65dcd53655e8bfa5` — le premier au dix-millième du chiffre publié, le second
à l'identique de la valeur Metal du 08-02. Le fichier est vérifié par sha256
avant toute mesure, et le refactor de `KvCache` qu'a demandé ce lot ne déplace
rien sur le chemin f16.

| axe | Δ (q8 − f16) | intervalle apparié | verdict |
|---|---|---|---|
| perplexité | **+0,049 %** | [−0,071 ; +0,170] % | ✅ |
| MMLU micro | **+0,33 pp** | [−0,45 ; +1,22] pp | ✅ |
| débit, `n_new` = 128 | **0,927× et 0,945×** | E ne recouvre ni 0,80 ni 0,90 | ⚠️ 2 séries sur 4 |

**Les deux axes de qualité contiennent zéro.** Un cache à 8,5 bits — int8 plus
échelle et biais f16 par groupe de 64, soit ÷1,882 et non ÷2 — ne se distingue
pas du f16 sur 12 fenêtres de perplexité ni sur 2 280 questions de MMLU
(1,3 % de discordance, McNemar p = 1,0000).

🚨 **Et pourtant le q8 n'est pas servi par défaut**, parce que la série
`n_new = 1024` a été **abandonnée en entier** : sa première invocation f16 a
mis **661 s** contre un seuil de 600 posé d'avance. La règle du §2.5 interdit
de la réduire — « réduire après avoir vu l'horloge, c'est choisir le point le
plus favorable après coup » — et le §4.3 exige le vert **sur les quatre
séries**. Le verdict s'étiquette donc « contexte court seulement », ce qui
interdit le défaut quelle que soit la valeur mesurée. **661 s, c'est 10 %
au-dessus du seuil** : exactement le dépassement qu'on négocie après coup.

⚠️ **Ce que le lot n'a pas mesuré est la question produit.** À `n_new = 1024`
le bras f16 tombe à 5,6 tok/s contre 9,6 à 128 : le coût du cache domine, donc
c'est là que le q8 devrait payer — et c'est la seule région inaccessible. Le
pré-enregistrement l'avait nommé d'avance : *le débit ainsi mesuré est un coût
sans son bénéfice*. On a mesuré la facture, pas la recette.

🔎 **Deux seuils hérités étaient faux, et la mesure le montre.** Le « ppl
0,7 % » est du bruit de graine de calibration entre fichiers *différents* :
l'intervalle apparié réel vaut ±0,12 %, quatorze fois plus serré. Le « σ
McNemar 0,4-0,6 pp » n'avait jamais été calculé : la SE appariée mesurée ici
est **0,43 pp**, contre 0,79-1,44 pp au 08-13 — l'écart n'est pas du bruit, ce
sont deux objets différents (modèles différents, 7-28 % de discordance, contre
le même fichier à deux précisions de cache, 1,3 %).

🔎 **Un cadeau sur une dette ouverte** : le 56,09 % reproduit la valeur Metal
du 08-02 à trois mois d'écart. L'errata du lot A relève un écart non expliqué
entre ce 56,09 (Metal) et 55,59 (CUDA), « 5× le glissement de la baseline ».
Cette mesure ne le referme pas — il faudrait un rejeu CUDA — mais elle établit
que le harnais Metal ne dérive pas : l'écart est Metal ↔ CUDA, pas temporel.

Le chantier MoE (P2, P6) est mis en pause par l'opérateur le 2026-08-14, plan
conservé, **modèle tranché** : Qwen3-30B-A3B, gpt-oss écarté sur le critère de
référence f16. L'estimateur de `ops/run.py` est corrigé au passage — il rendait
3,34 Md pour un modèle de 30,5, sans lever d'exception.

## 2026-08-15 — P1 mesuré : uniformiser la boucle vaut un ordre de grandeur

P1 du plan d'exécution, **0 $, une journée de Mac**. Le pré-enregistrement
[`p1-2026-08-13`](../proofs/preregistration-p1-2026-08-13.md) est **horodaté
avant le run** (SHA256 `5109b35f…`, quatre calendriers OpenTimestamps) — la
première fois du projet qu'un seuil est ancré avant la mesure qu'il juge, et
non après. Journal :
[`mesures/p1-rankbench-2026-08-15.txt`](mesures/p1-rankbench-2026-08-15.txt).

16 777 216 blocs réels **tirés au réservoir** (algorithme R, graine imprimée)
dans le 4B scellé — et non des préfixes contigus, qui sont un début de fichier
et pas un tirage. 18 rounds dont 3 jetés, tous les bras à chaque round, surcoût
de soumission mesuré **par round** avec sa dispersion.

| bras | o/bloc | ns/bloc | × le sol, méd [min–max] |
|---|---|---|---|
| `sol` | 12 | 0,0777 | 1,00× |
| `masques` | 12 | 0,1486 | 1,92× [1,39–2,30] |
| `cascade-archive` | 8 | **10,8115** | 131,31× [100,29–139,32] |
| `cascade-uniformisée` | 8 | **1,7809** | 21,90× [16,79–23,32] |
| **`marche-binomiale`** | 12 | **0,3101** | **3,84× [2,99–4,20]** |
| `sol-rang` (É3a) | 8 | 0,0796 | 0,99× [0,78–1,19] |

✅ **`marche-binomiale` 0,3101 contre 1,50 — VERT.**
✅ **`cascade-uniformisée` 1,7809 contre 2,00 — VERT**, 0,22 ns de marge.
❌ **`cascade-archive` 10,8115 contre 2,00 — ROUGE.**

**Le résultat que rien ne prédisait : uniformiser la boucle vaut un ordre de
grandeur.** 10,81 → 1,78 ns sur les **mêmes bits, la même table et la même
recherche de classe** — le seul écart entre les deux bras est la forme du
travail (24 pas identiques, sélection branchless, réciproques magiques, zéro
indexation dynamique de registres). Et la marche binomiale, qui ne divise
jamais, tombe à **3,84× le plancher de la machine**, soit environ deux fois le
décodeur le plus rapide qu'elle ait jamais exécuté.

**Trois conséquences, toutes pré-enregistrées.** Le **gate CUDA de P4 est
franchi** (0,3101 ≤ 0,45) : le bras cascade/marche du job carte est autorisé,
sous réserve du go de dépense. **P5 s'ouvre** — la règle du §4.2 est « si et
seulement si la MARCHE passe 0,45 », et c'est bien elle. **E1v n'est pas
mort-né** : le §4.3 fermait la ligne si l'archive passait 2,00 ns ; elle en
rend 10,81.

> 🚨 **La première de ces trois conséquences a été RETIRÉE le jour même**, 57
> minutes après avoir été écrite, et l'entrée du **2026-08-15 (soir)** ci-dessous
> la retourne : le bras qui a franchi 0,45 décodait une **marche de 24
> créneaux**, pas un **bloc** ; un bloc rend **0,6735 ns**. Le seuil n'a pas
> bougé — il a été appliqué à la quantité qu'il nomme. Les deux autres
> conséquences (ouverture de P5, E1v non mort-né) **tiennent**, et les chiffres
> de cette entrée sont ceux du run. Ce renvoi est ajouté le 2026-08-16 ; le
> texte de l'entrée n'est pas modifié.

🔎 **Le 6ᵉ bras a servi le jour même.** L'É3(a), arbitré avant le tampon,
ajoutait `sol-rang` — le plancher du flux de rang, 8 o/bloc. Il rend 0,0796
contre 0,0777 pour `sol` : les deux planchers sont indiscernables, donc
l'adressage ne discrimine rien à cette échelle et les 0,3101 de la marche sont
du **décodage**. Sans ce bras, c'était une conjecture. Deux autres propositions
du même amendement — transposer la règle de suspension aux seuils absolus,
chiffrer l'acceptation du tirage — ont été **écartées** par l'opérateur : le
banc rend ses nombres, la conclusion reste chez lui.

⚠️ **V0 a coupé le premier run**, et c'est ce pour quoi il existe :
`cascade-archive` échouait sur **883 blocs sur 16 777 216** (5·10⁻⁵), le shader
retrouvant le rang de zéros par un scan de `counts_b` — faux quand ce rang a un
compte nul. Un échantillon l'aurait manqué ; le sweep intégral CPU ne pouvait
pas le voir, la composition Rust lisant `c.n_off` directement. Aucun
chronométrage n'a eu lieu sur ce run-là.

⚠️ **Les bras ne lisent pas le même nombre d'octets et aucun ns/bloc n'est
corrigé du trafic.** Et l'étendue de la colonne « × le sol » est dominée par
son dénominateur : le sol tourne en 1,3 ms là où une soumission coûte 0,14. Le
ns/bloc, formé sur le minimum propre à chaque bras, est la quantité stable.

🔎 **Les deux ancres se reproduisent** : `sol` 0,0777 contre 0,084 au run
`decreal` du 08-01, `masques` 0,1486 contre 0,152 — −7,5 % et −2,3 % après
quatre mois, un autre binaire et un autre tirage. Le §1.4 interdit d'y lire un
seuil ; le §7 demandait d'expliquer un écart notable, et il n'y en a pas.

**V0, dans la journée qui a précédé** : la fixture synthétique ferme les 98
entrées de table qu'aucun fichier cap 12 n'atteint (origine, 82 classes de
coquille 13, 15 classes inutilisées) ; le sweep intégral passe les deux
décodeurs sur les 150 681 600 blocs ; l'aller-retour rang → arrangement → rang
ferme **sur l'arrangement du GPU**, via un jumeau instrumenté épinglé au bras
chronométré par le produit scalaire ; et la bijection est établie par
énumération exhaustive des 49 entrées de cardinalité ≤ 65 536.

🕳️ **Un vert vide trouvé par mutation, et il vaut d'être retenu** : l'étalon du
bras marche lisait `rec.vals` — la table qu'il vérifiait. Une mutation donnant
aux classes de coquille 13 la norme de la coquille 12 a été tuée par la cascade
et **survécue** par la marche, à neuf décimales. C'est le « malentendu partagé »
que le §3.1 des pré-enregistrements existe pour écarter, et il était dans le
harnais.

## 2026-08-15 (soir) — Un compte d'opérations n'est pas une prédiction de temps : ×1,002 prédit, ×2,17 mesuré

P1b, P1c et P5, **0 $, même machine, même journée**. Journaux :
[`mesures/p1b-marche-bloc-2026-08-15.txt`](mesures/p1b-marche-bloc-2026-08-15.txt),
[`mesures/p1c-e1v-flux-2026-08-15.txt`](mesures/p1c-e1v-flux-2026-08-15.txt),
[`mesures/p5-cns-2026-08-15.txt`](mesures/p5-cns-2026-08-15.txt).

🚨 **L'AUTORISATION DU BRAS CUDA DE P4 EST RETIRÉE, et elle a vécu 57
minutes.** Le bras `marche-binomiale` qui avait franchi le gate de 0,45 ns
décodait **une marche de 24 créneaux, pas un bloc** — un bloc pair en demande
deux, plus le mot de Golay, la réparation de parité et trois règles de signe.
P1b l'a mesuré : **`marche-bloc` rend 0,6735 ns/bloc**, soit **×2,17** la
marche. Vert contre le kill de 1,50 (distance 0,8265), **dépassé** contre le
gate de 0,45. Ce n'est pas rouvrir le seuil de P1 — c'est l'appliquer à la
quantité qu'il nomme. Le **régime intermédiaire** du §4.2 de P1 s'applique mot
pour mot : *le bras survit comme point de la courbe et n'achète AUCUN bras
CUDA — il faut une idée neuve, pas un job.*
Traçable en git : autorisation écrite au commit `b18fe52` (13:42:02), retirée
au commit `c40641b` (14:39:33). **57 minutes**, pas une demi-journée.
Ce qui n'est **pas** touché : l'ouverture de P5, dont la condition porte sur la
marche ; les quatre critères de P5 ; le non-verdict d'E1v par l'archive
(10,4533 ns dans ce run contre un seuil de 2,00).

🕳️ **Et c'est la TROISIÈME fois sur ce projet qu'un compte niveau source se
trompe d'un facteur ~2 sur un noyau** : `Golay70` (1,9-2,4× promis, 1,77×
rendu), `E1c` (facteur 2 sur un compte de mots), et ici. Le compteur pondéré
par le fichier donnait 39,55 pas pour une marche et 39,64 pour un bloc — d'où
la prédiction **×1,002**, écrite avant la mesure. Il ne comptait que les
balayages de marche : ni le Golay, ni le second appel, ni la parité, ni les
trois règles de signe. **Un compte d'opérations n'est pas une prédiction de
temps, même quand il porte sur la boucle qu'on croit dominante.**

❌ **L'idée neuve a été tentée le soir même, et elle est réfutée.** Hypothèse
écrite avant la mesure : `marche-bloc` porte deux tableaux de 24 octets indexés
par le créneau, donc le ×2,17 pourrait être un **débordement** en mémoire
thread-local plutôt qu'un travail. Le bras plat, qui supprime ces 48 octets,
rend **0,8346 ns contre 0,6704** — **24 % PLUS LENT**. La règle de restitution,
écrite d'avance et symétrique de celle qui avait retiré, porte sur le meilleur
décodeur de **bloc** du run : **0,6704 > 0,45, elle ne tire pas.**
⚠️ Le ×2,17 reste donc **non attribué** : le compte de pas s'est trompé,
l'unique hypothèse chiffrable est fausse, et l'attribuer demanderait un
profileur — que ce projet n'a jamais utilisé (§2c de `CLAUDE.md`).

✅ **P1c : le vrai flux E1v décodé rend 0,6795 ns/bloc**, et son adressage à
largeur variable ne coûte que **+0,0083 ns, soit +1,2 %** sur `marche-bloc`.
Vert contre le kill de 1,50 ; la restitution ne tire pas là non plus (0,6711 >
0,45 sur le run à 9 bras). ⚠️ Ce bras mesure le **meilleur cas** de l'adressage
E1v : `gid` y est l'indice de bloc, donc l'alignement est vrai par
construction, alors que le matvec servi met un warp par **ligne**.

✅ **P5 est clos, 4/4** — la ré-bijection CNS d'E1v, 0 $ :
**C1** largeur réalisée **53,7370 bits/bloc → 2,3877 b/poids noyau** contre un
critère C0 de 2,60 · **C2** bijection sur les **150 681 600 blocs**, zéro
écart, l'origine couverte par la fixture et non par le sweep · **C3** forme du
décodeur : **90 pas au maximum** contre 96 déclarés, **zéro division** en
source **et** en assembleur hôte · **C4** transcodage **1,088× [1,087–1,090]**
`Planes14` contre un seuil de 2,0. ⚠️ La réouverture qu'ils achètent est
**nominative et provisoire** : le droit de porter E1v sur carte, pas une
adoption. E1v reste **plus gros** que l'archive (2,3877 contre 2,1912 b/poids
noyau).

⚠️ **Dette de provenance, déclarée dans les journaux eux-mêmes** : le `.ots` de
P1b et celui de P5 sont posés à **15:23, après leurs mesures** (commit
`6983a2e`) ; leur antériorité ne repose que sur leurs commits. Ce qui est
tamponné avant toute mesure, c'est **P1** (13:37 le 08-15), **donc le seuil de
0,45 lui-même** — et c'est lui qui décide ici. P1c, lui, est horodaté avant sa
mesure.

## 2026-08-16 — Le plancher plafonne tout travail de format, et désigne le seul poste jamais attaqué

Deux jobs L40S, **1,62 $** au total. Journaux :
[`mesures/e1v-cuda-2026-08-16.txt`](mesures/e1v-cuda-2026-08-16.txt),
[`mesures/nullk-plancher-2026-08-16.txt`](mesures/nullk-plancher-2026-08-16.txt),
[`mesures/e1c12-aligne-2026-08-16.txt`](mesures/e1c12-aligne-2026-08-16.txt).
Passation autonome :
[`archive/passation-2026-08-16.md`](archive/passation-2026-08-16.md).

❌ **E1v est FERMÉ pour le chemin servi : 0,25× FP16 [0,25–0,25], 25 Go/s**
(job `6a814ba31f5885ae605bcb55`, 0,85 $). **44,253 ms** de médiane contre
10,988 pour le FP16 et 5,100 pour `Planes14` — **8,7× plus lent que le layout
servi**, 25 Go/s contre 428. Les critères d'X3 étaient publiés le 2026-08-12 et
n'ont pas été amendés : ≥ 2,05× remplace `Planes14`, ≥ 1,90× remplace
`Planes12x`, **< 1,60× la ligne se referme**. Manqué d'un facteur **6,4** ;
aucune marge d'interprétation n'existe à cette distance.

✅ **Ce que la mesure ne retire pas : le format tient exactement sa promesse.**
E1v lit **1,09 Go là où `Planes14` en lit 2,18** — la moitié — soit **2,398
b/poids noyau** contre 4,804 ; exactitude **2,4e-8·Σ|w·x| sur 1 105 920
lignes** du premier coup, **79 registres, zéro spill, zéro octet local**. La
coupe alignée ligne est pesée sur les **octets écrits** : 53,7370 → **53,9941
bits/bloc**, soit **2,3877 → 2,3983 b/poids noyau** — le chiffre était calculé
(P5, 08-15), il est mesuré. ⚠️ Le « +0,48 % » que porte le journal de P5 est le
surcoût relatif en **bits/bloc** (0,2571 / 53,7370) ; le couple **b/poids**, lui,
bouge de +0,44 %. Deux bases, et il ne faut pas coller le premier au second.
**Ce qui est mort est le décodeur EN LIGNE**, borné
en calcul — deux marches binomiales, un mot de Golay, une réparation de parité
et trois règles de signe contre les sélections de `Planes14`. Le format reste
disponible **hors boucle** (disque, transport).

🆕 **LE PLANCHER EST MESURÉ, et c'est le résultat de fond de la quinzaine**
(job `6a81b2b71f5885ae605bdcc9`, 0,77 $, un noyau de trente lignes). `nullk`
garde la grille, le tuilage, le staging, `warp_sum` et l'épilogue, et **ne lit
aucun poids** : **2,305 ms contre 5,102 pour `Planes14`**, soit **45,2 %**.
Tout se dérive d'un seul run :

| | |
|---|---|
| ~~plafond absolu de tout travail de **format**~~ 🕳️ **plancher de NOTRE géométrie de lancement** — requalifié le 2026-08-21, QTIP passe dessous | **4,77× FP16** [4,74–4,77] |
| où `Planes14` en est | **2,16×** [2,15–2,16] |
| ce que le format achète **net** du plancher | **3,11×** (8,691 ms de trafic contre 2,797) |
| coût du décodage de `Planes14` | **~7 %** du temps de trafic (779 Go/s net contre 836) |
| part du temps qu'**aucun** format ne touche | **45,2 %** |

🚨 **Quatre routes sous `Planes14` ont été tentées — E3, `Golay70` v2, `e1c14`,
E1v — et TOUTES sont bornées en calcul, aucune en octets.** Le plancher dit
pourquoi c'était le mauvais front : le format se dispute **au plus 55 %** du
temps, `Planes14` en capture déjà l'essentiel, et **le poste majoritaire n'a
jamais été attaqué**. C'est ce que la **famille `k` de P4 §2.6** existe pour
amortir, **et elle n'est pas écrite**. Le chiffre qui l'aurait dit coûtait
0,77 $.
⚠️ Ce 45,2 % porte sur **252 projections d'un token** ; les « 39 % » de
l'attribution du 2026-08-05 découpent **2,04 ms par token**, normes, attention
et rotation comprises. **Deux dénominateurs** : le rapprochement demande de
refaire l'attribution, pas de reporter un nombre.
⚠️ Le plancher n'est pas du gaspillage — staging, réduction, épilogue et
écriture sont du travail qu'un noyau réel doit faire. C'est un plancher, pas
une perte ; mais c'est un **plafond** sur ce que le format peut gagner.
⚠️ L'AWQ est **hors** de ce tableau, et c'est une garde : son noyau a sa propre
grille, et lui soustraire ce plancher donnerait 2 006 Go/s — au-dessus de la
HBM d'une L40S. Un chiffre impossible qui dit où la soustraction cesse d'être
licite.

✅ **`e1c12` survit à l'alignement warp ; `e1c14` est enterré** (0 $, calculé
sur le 4B scellé, exceptions comprises). Bourrer chaque ligne à un multiple de
32 blocs porte 150 681 600 blocs à 173 998 080 (**+15,47 %**) sans ajouter un
seul poids : **`e1c12` aligné rend 4,2880 b/poids noyau contre 4,3424 pour
`Planes12x` (−1,3 %)** ✅, **`e1c14` aligné 5,2354 contre 4,8040 pour
`Planes14` (+9,0 %)** ❌. Le second est un **contrôle** : 5,2354 est le chiffre
exact d'X3, que le terme d'exceptions ne touche pas.
🔎 **Conséquence, et c'est le vrai résultat** : la question d'`e1c12` cesse
d'être une question de **bits** — 1,3 % n'achète rien contre un layout
lui-même non servi — et devient une question de **vitesse de transposition**.
⚠️ Et elle **n'hérite pas** du verdict d'E1v : E1v est mort d'être borné en
**calcul** ; `e1c12` décode le **même contenu** que `Planes12x` — des
sélections sur des plans de bits — et sa question est un **motif de lecture**.

🕳️ **Deux casses préexistantes du chemin CUDA, et personne ne pouvait le
savoir.** L'image était **incompilable depuis le 2026-08-15** : `fused_cuda.rs`
sans le `KvMode` livré par le KV q8, puis deux tables `[_; N_ARMS]` restées à 7
littéraux quand P4 a porté `N_ARMS` à 15. Une troisième s'est ajoutée le
2026-08-16, **à trois lignes de sa propre correction** — un alias `const` doit
annoncer un type, donc restater une longueur. Cause commune : **tout ce qui vit
sous `cfg(target_os = "linux")` n'a aucun filet**, et la seule chose qui
l'exerce est une construction d'image que personne ne lance par routine.
**Parade posée** : les tables de bras vivent désormais dans `arms.rs`, qui
compile et se teste **sur le Mac**, avec un test exigeant qu'aucun bras
exécutable n'ait de ligne manquante ni de ligne en double ; et `bin/cuhcheck`
fait **parser 14 unités CUDA sur le Mac**, une classe d'erreur que le dépôt
déclarait inattrapable. ⚠️ Reste ouvert : `planesbench.rs` (**2 555 lignes**,
`wc -l` au commit de cette entrée — la passation du jour écrit « 2 100 », qui
ne se retrouve dans aucune révision du fichier ce jour-là : il n'est jamais
descendu sous 2 495) et `fused_cuda.rs` ne sont typés par aucune machine de
développement.

🕳️ **Et la leçon de portage, payée par un défaut que le miroir Rust a
attrapé** : `e1v_peek` est transcrit du `peek` Metal scellé, dont le
commentaire garantit « `off` est toujours au moins 10 ». **Faux pour E1v**,
dont l'en-tête vit dans le préfixe du groupe : son premier champ est à
l'offset zéro, et `hi << (64 − 0)` est un décalage de 64 — **indéfini en
C++**, exécuté par NVIDIA comme un décalage de rien. L'index Golay de presque
tous les blocs serait revenu corrompu. **Une transcription porte les gardes de
son original sans porter les hypothèses qui les rendaient suffisantes.** Le
seul dispositif qui l'attrape est de faire **exécuter le texte lui-même**
contre une référence indépendante (`tests/host_e1v.cpp`).

⚠️ **Deux dettes de ce run, déclarées plutôt que lissées.** (1)
`proofs/preregistration-e1v-cuda-2026-08-15.md` **n'est pas horodaté**, par
décision explicite de l'opérateur : son antériorité ne repose que sur sa date
de commit — les **seuils**, eux, sont ceux d'X3 publiés le 2026-08-12, donc
antérieurs par un chemin indépendant. (2) Deux bras du contrôle sortent de leur
plage publiée **par le haut** (`Planes12x` 2,01 contre [1,95–1,99],
`Golay70` v1 1,34 contre [1,29–1,32]) : dérive **inter-run**, l'intra-run étant
de 0,13 % et la demi-étendue max de 0,46 %. Aucun verdict n'en dépend ici ; un
run publiant à 1 % près devrait l'expliquer.

### ⚠️ Une dette de provenance jamais relevée : le critère de 1,6× qui a écarté E2

**Le critère de 1,6× n'a pas de tampon, et le papier en fait pourtant un
argument de rigueur** — `paper/sections/layouts.tex` écrit « below the
$1.6\times$ criterion we had fixed before measuring ». Vérifié ici, le
2026-08-16, sur l'historique git :

- ✅ **Le critère EST antérieur à la mesure, et le dépôt le montre** — mais
  dans un **message de commit**, pas dans un fichier : `caef2ac`
  (**2026-08-07 10:36:27**), le commit qui livre le layout `Golay70`, se
  termine par « Reste le banc : ≥ 1,6× vs FP16 = point « bits » ; ≥ 1,85× =
  domine `Slot32` ». La mesure arrive **52 minutes plus tard**, au commit
  `4a09d8b` (**11:28:59**), celui qui crée
  [`mesures/e2-golay70-bench-2026-08-07.txt`](mesures/e2-golay70-bench-2026-08-07.txt).
- ⚠️ **Aucun FICHIER ne le portait avant la mesure.** `git log -S'critère de 1,6'`
  et `git log -S'1,6×' -- '*.md'` ne rendent aucun commit antérieur à
  `4a09d8b` ; la spec d'où E2 sortait,
  [`archive/pistes-format-vram-2026-08-05.md`](archive/pistes-format-vram-2026-08-05.md),
  ne porte aucun critère de vitesse dans l'état où elle était à `caef2ac`.
  *(L'unique occurrence plus ancienne de « 1,6× » dans le dépôt est le
  « ≈ 1,6× de plus que `shell_bests` » de l'encodeur — sans rapport.)*

🚨 **Correction du 2026-08-16, le jour même : la première rédaction de cette
section concluait « le dépôt ne porte aucune trace du critère avant les nombres
qu'il juge ». C'est FAUX**, et faux par la faute qu'elle dénonçait : `git log -S`
ne cherche que dans le **contenu** des fichiers, jamais dans les messages de
commit, et le message de `caef2ac` — le commit que la section citait elle-même —
porte le critère. Un outil dont on n'énonce pas le périmètre ne prouve pas une
absence.

**Ce que la vérification établit donc, et rien de plus** : l'antériorité tient
par une **date de commit**, exactement le degré de preuve que ce dépôt qualifie
ailleurs de « plus faible qu'un tampon » (journal de P5, et la dette déclarée
sur le pré-enregistrement E1v-CUDA). Ce n'est pas « rien » et ce n'est pas un
`.ots`. Le verdict d'E2 n'est de toute façon pas remis en cause : 1,31× est très
loin de 1,6×, et `Golay70` v2 l'a re-jugé depuis sur une chaîne propre. Ce qui
reste à l'opérateur : décider si le papier peut écrire « fixed before
measuring » sur cette base seule, ou doit citer le commit.

🔎 **À distinguer explicitement de `Golay70` v2, dont la chaîne est
exemplaire** : critère commité à **09:30:36** (`9402e4e`), `.ots` ancré hors
du dépôt à **09:31:06** (`f56ae30`), mesure à **13:34:31** (`759b562`), le
2026-08-11. C'est ce contraste qui rend la dette lisible : le projet **sait**
faire, et ne le faisait pas encore le 08-07.

## 2026-08-17 — Le bucket n'avait jamais été inventorié : trois trous comblés pour 0 $

> ⚠️ **Cette entrée retourne des verdicts d'entrées antérieures qu'elle ne
> modifie pas** (règle d'entretien) : celle du 2026-08-10 (« la courbe a un
> genou », « écart AWQ 6,09 pp — différence nue, la paire n'existe pas »),
> celle du 2026-08-15 (X3, « `e1c14` enterré ») et l'« État courant » du
> 2026-08-16. Les passages visés sont cités mot pour mot ci-dessous.

🕳️ **LA CAUSE COMMUNE, et elle vaut plus que les trois résultats.** Le dépôt
donnait pour perdus l'artefact 14B scellé **et** les trois dumps MMLU du 14B.
La vérification du 2026-08-16 qui l'a conclu est **exacte** — elle a cherché
**sur la machine**. Or le job de campagne n'écrivait pas sur une machine : il
écrivait dans le **bucket monté**, le dispositif qui existe précisément pour
que les sorties survivent au conteneur (`ops/run.py --bucket`). Tout y dormait
depuis le 2026-08-10. Ce que le dossier chiffrait à « une campagne MMLU à
refaire » a coûté **579 ko de bande passante** ; l'artefact, **~9 min** contre
les 27,67 $ et 302 min de sa quantification. **Le bucket contient 69 fichiers
et 46,7 Go, et personne ne l'avait inventorié depuis sa création le
2026-08-02.** Règle posée en conventions (`CLAUDE.md` §7) : *toute sortie
déclarée perdue mérite un `hf buckets ls` avant qu'on chiffre un re-run.*
⚠️ Ce n'est **pas** une garantie : le **8B scellé** a été cherché aux deux
endroits et il est bel et bien **perdu** — le bucket n'en héberge que la
version *projections seules*. `hf buckets ls` change ce qu'on sait, pas ce qui
existe.

✅ **1. LA PAIRE `AWQ − LLVQ` DU 14B EXISTE.** Elle vaut **+6,09 pp, IC95
[+3,62 ; +8,52], SE 1,25 pp, McNemar exact p = 1,143e-11** (A✓B✗ 230 /
A✗B✓ 106, 14,7 % de discordantes), bootstrap apparié stratifié par matière,
10 000 tirages, graine `0xb0075eed`, 2 280 questions, empreinte
`65dcd53655e8bfa5` des deux côtés
([`mesures/mmlupair-14b-2026-08-17.txt`](mesures/mmlupair-14b-2026-08-17.txt)).
🚨 **Ce que ça retire** : l'entrée du 2026-08-10 et `CLAUDE.md` écrivaient
« **différence nue** — la paire `AWQ − LLVQ` n'existe pas au 14B, ses dumps
sont perdus, la recalculer exige de refaire la campagne MMLU 14B, **ne jamais
citer 6,09 avec un intervalle** ». **Le point estimé ne bouge pas d'un
centième** : ce n'est pas un chiffre neuf, c'est le même qui cesse d'être nu,
et la prudence est **satisfaite** plutôt que contournée. Authenticité établie
**avant** usage — les trois micros rejouent 78,97 / 78,21 / 72,12 et
`f16 − LLVQ` rejoue ses **quatre** grandeurs publiées (+6,85 [+4,52 ; +9,12],
SE 1,16, McNemar 8,666e-16). Les trois dumps sont **commités** dans
[`data/mmlu-dumps/`](data/mmlu-dumps/) : la perte ne peut plus se reproduire.
**Le fil des trois écarts est homogène pour la première fois** : 4B +14,45
[+11,60 ; +17,27] · 8B +7,49 [+5,28 ; +9,70] · 14B **+6,09 [+3,62 ; +8,52]**.

❌ **2. ET LE « GENOU » NE TIENT PAS AU TEST.** Les trois écarts étant enfin de
la même espèce, la chute d'un palier au suivant se teste (SE composées en
quadrature — *calculé* ; campagnes distinctes, pas d'appariement inter-modèles) :

| palier | chute | SE | z | p | verdict |
|---|---|---|---|---|---|
| 4B → 8B | 6,96 pp | 1,82 | 3,82 | 0,0001 | ✅ **RÉSOLU** |
| **8B → 14B** | **1,40 pp** | **1,68** | **0,83** | **0,40** | ❌ **NON RÉSOLU** |
| 4B → 14B | 8,36 pp | 1,91 | 4,38 | ≈ 1e-5 | ✅ **RÉSOLU** |

🚨 **Sont donc RETIRÉES** les phrases qui faisaient du **ralentissement** un
résultat : « la courbe a un genou », « elle ne se referme pas au même rythme,
elle ralentit », « une extrapolation linéaire aurait sur-promis », « depuis le
14B on sait qu'elle n'est pas droite ». Elles reposaient sur des points estimés
que les barres **ne séparent pas**. ⚠️ **Et p = 0,40 ne prouve pas l'égalité
non plus** : sur ce palier les données sont **muettes**, pas concluantes — « ça
ralentit » et « ça continue » restent toutes deux compatibles.
⚠️ Du côté **perplexité** le genou n'est pas même testable, pour une raison de
conservation : le pas 8B→14B a désormais sa barre (−13,9 %, IC95
[−22,8 ; −4,9] sur référence f16 — réel mais **mal borné**), mais le pas 4B→8B
qui porte le « −43 % », donc toute la force de l'argument, **n'est pas
barrable**. **Ce qui sort renforcé est la conclusion opérationnelle déjà
écrite : on ne publie pas de loi d'échelle sur trois points, et le 32B reste ce
qui trancherait.**

✅ **3. LA LIGNE MÉMOIRE EXISTE AUX TROIS TAILLES**
([`mesures/rtbits-14b-2026-08-17.txt`](mesures/rtbits-14b-2026-08-17.txt)).
`params_total` du 14B = **14 768 307 200** (*mesuré* dans le fichier scellé,
recoupé par l'arithmétique de l'architecture : 13 212 057 600 linéaires +
1 555 824 640 embedding + 424 960 normes). Le 14B servi (`Planes14` +
embedding q8) pèse **5,106 b/param modèle entier** contre **5,404** pour l'AWQ
officiel — **sous l'AWQ de 5,5 %**. La ligne : 4B 5,162 vs 5,302 (−2,6 %) · 8B
5,322 vs 5,956 (−10,6 %) · 14B 5,106 vs 5,404 (−5,5 %). ⚠️ **La marge n'est PAS
monotone** — elle culmine au 8B — et le mécanisme n'est pas la méthode mais la
**part de l'embedding** (9,7 % · 15,2 % · 10,5 %), que l'AWQ laisse en f16 et
que nous passons en q8. *Trois points, un mécanisme, aucune loi.* ⚠️ Étiquettes :
nos cellules sont *calculées* sur octets **mesurés**, embedding **modélisé** à
8,5 b/param — même statut aux trois tailles.
🚨 **Cette entrée finissait sur « Ni la vitesse ni la VRAM carte n'ont jamais
été mesurées à 14B » : DÉMENTI le soir même — le 14B est SERVI, 42,9 tok/s dans
9,39 Go contre 17,0 dans 29,54 au bras dense, 128 tokens identiques**
([`mesures/fusedrun-14b-2026-08-17.txt`](mesures/fusedrun-14b-2026-08-17.txt),
job `6a83121be55292eada79b611`, 1,24 $). ⚠️ **Ce fil n'a pas encore son entrée
pour ce run** — le journal fait foi en attendant.
Le recoupement carte donne **5,0866 b/param** contre les
5,106 de cette ligne, **−0,38 %**, dans une bande de ±0,5 % posée avant le run :
c'est un **troisième instrument**, pas un remplacement (il divise un affichage
arrondi au centième de Go).

✅ **4. LES NEUF CELLULES DE PERPLEXITÉ REÇOIVENT LEUR BARRE — sauf trois**
([`mesures/ppl-appariee-8b-14b-2026-08-17.txt`](mesures/ppl-appariee-8b-14b-2026-08-17.txt),
intervalles t appariés fenêtre par fenêtre, n = 12, t = 2,200985160 ; *calculé*
sur des NLL déjà au dépôt, **aucun token rescoré**) :

| | AWQ / f16 | **LLVQ / f16** | LLVQ / AWQ |
|---|---|---|---|
| 8B | +4,80 % [+4,24 ; +5,35] | **+22,01 % [+19,37 ; +24,70]** | +16,42 % [+14,17 ; +18,72] |
| 14B | +3,81 % [+3,27 ; +4,34] | **+18,94 % [+17,22 ; +20,68]** | +14,58 % [+13,10 ; +16,08] |

Les six excluent zéro, et les **72** comparaisons fenêtre par fenêtre vont
toutes dans le même sens. Contrôle négatif validé : le même fichier à deux
embeddings rend +0,0004 %, IC contenant zéro — **l'instrument sait rendre un
nul**. ⚠️ Ces intervalles ne portent **que** l'échantillonnage du corpus ; ils
ne contiennent **pas** le tirage de calibration, non mesuré à ces échelles.
🚨 **LE 4B N'A PAS D'INTERVALLE ET N'EN AURA PAS sans rejeu** : son journal de
campagne est une **synthèse**, les NLL par fenêtre n'existent pas — et ce n'est
pas une des trois paires qui manque, ce sont les trois.

🔎 **5. LE VERDICT D'ENTERREMENT D'`E1c14` ÉTAIT UN VERDICT 4B, et il ne
transfère pas.** L'entrée du 2026-08-15 (X3) écrit « `e1c14` enterré : plus
gros une fois aligné au warp, +9,0 % » — vrai **au 4B**, faux au 14B. La
pénalité d'alignement warp vaut **+15,47 % de blocs sur les formes du 4B** mais
**+4,18 % sur celles du 14B**, dont les lignes sont plus longues. Sur blocs
réels du 14B : `E1c14` aligné **4,6410** contre 4,7063 pour `Planes14`
(−1,4 %) ; `E1c12` aligné 3,8021 contre 4,2420 (−10,4 %). ⚠️ **Cela ne le
ressuscite pas** — aucun de ces nombres n'est une vitesse, et `E1c` n'a jamais
été dispatché par un banc, à aucune largeur. Ce qui est établi est étroit : **la
pénalité d'alignement est une fonction des FORMES, pas une constante du
layout**. Une phrase « `E1c14` est plus gros que `Planes14` » sans « au 4B » est
fausse au 14B.

🔎 **LA SECONDE LEÇON, sur un objet neuf : un journal de SYNTHÈSE est une perte
irréversible.** Les campagnes 8B et 14B ont gardé leur **sortie brute** et sont
barrables aujourd'hui, gratuitement ; la campagne 4B a gardé un **tableau** et
ne l'est plus. `bin/ppl` imprime ses NLL à 9 décimales sur stderr exprès. Le
coût de la sortie brute est **quelques kilo-octets**. Consignée en conventions
(`CLAUDE.md` §7).

**Coût du lot : 0 $.** Aucun GPU, aucun job. Lecture seule sur le Hub, ~6,5 Go
de bande passante, quelques dizaines de secondes de CPU sur le Mac de dev.
📄 [`mesures/rtbits-14b-2026-08-17.txt`](mesures/rtbits-14b-2026-08-17.txt) ·
[`mesures/mmlupair-14b-2026-08-17.txt`](mesures/mmlupair-14b-2026-08-17.txt) ·
[`mesures/ppl-appariee-8b-14b-2026-08-17.txt`](mesures/ppl-appariee-8b-14b-2026-08-17.txt) ·
[`data/mmlu-appariee.csv`](data/mmlu-appariee.csv)

⚠️ **Une incohérence interne aux journaux, à ne pas lire comme une
contradiction** : `rtbits-14b` §4 et §5 écrivent encore « la paire MMLU
`AWQ − LLVQ` au 14B n'existe toujours pas ». Il a été **écrit avant**
`mmlupair-14b` (12:14 contre 12:19) et les journaux ne se modifient jamais.
C'est `mmlupair-14b` qui fait foi sur ce point.

## 2026-08-17 (soir) — Les logs de Jobs HF ne sont pas purgés : la perplexité du 4B retrouve sa barre, et le genou se dédouble

> ⚠️ **Cette entrée retourne deux verdicts de l'entrée du 2026-08-17 (matin),
> qu'elle ne modifie pas** (règle d'entretien) : son §2 (« le genou ne tient
> pas au test », « du côté perplexité le genou n'est pas même testable ») et
> son §4 (« LE 4B N'A PAS D'INTERVALLE ET N'EN AURA PAS sans rejeu »). Les
> passages visés sont cités mot pour mot ci-dessous. **Le §2 reste vrai sur
> MMLU** ; c'est sa portée qu'il faut corriger, pas son calcul.

🕳️ **LA CAUSE, ET C'EST DÉSORMAIS UN MOTIF, PLUS UN INCIDENT.** L'entrée du
matin conclut « les NLL par fenêtre du 4B n'existent pas, son journal de
campagne est une synthèse » et devise leur rétablissement à un rejeu carte de
~0,25 $. **Faux : `hf jobs logs 6a746d8f6b79c09949c23fb4` rend les 36 lignes
`window i/12 nll …` (3 bras × 12 fenêtres, 9 décimales) en deux secondes, pour
0 $.** C'est la **deuxième fois en deux jours** qu'une sortie déclarée perdue
vivait ailleurs — la veille, les dumps MMLU du 14B dans le **bucket monté** — et
dans les deux cas une **dépense avait été devisée contre cette absence**. La
cause est la même aux deux coups : la conclusion « perdu » venait d'avoir
cherché **au mauvais endroit**, jamais d'un canal interrogé et vide.
**Règle élargie en conventions ([`../CLAUDE.md`](../CLAUDE.md) §7)** : *avant de
budgéter un re-run, épuiser les canaux de rétention — `hf buckets ls`,
`hf jobs logs`, `hf jobs inspect`* (ce dernier rend la **ligne de commande
exacte** d'un job passé). ⚠️ **La rétention des logs HF n'est NI documentée NI
garantie** : elle couvre aujourd'hui les 62 jobs du projet depuis le 2026-08-02
(vérifié sur le plus ancien), elle peut cesser demain — d'où le commit du brut
([`mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt`](mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt),
sha256 `07bf4119…`), et non une citation par identifiant de job. Le corollaire
posé hier en sort **renforcé, pas démenti** : un journal de synthèse est une
perte irréversible **dès que le canal de rétention expire**.

✅ **1. LES TROIS CELLULES DE PERPLEXITÉ DU 4B ONT LEUR BARRE — la colonne est
complète aux TROIS tailles** ([`mesures/ppl-appariee-4b-2026-08-17.txt`](mesures/ppl-appariee-4b-2026-08-17.txt),
intervalles t appariés fenêtre par fenêtre, n = 12, ddl 11, t = 2,200985160 ;
*calculé* sur des NLL **mesurées le 2026-08-06**, aucun token rescoré) :

| Qwen3-4B | excès | IC95 | t | fenêtres |
|---|---|---|---|---|
| AWQ / f16 | **+10,49 %** | [+8,55 ; +12,47] | +12,38 | 12/12 |
| **LLVQ / f16** | **+38,45 %** | [+33,62 ; +43,45] | +20,18 | 12/12 |
| LLVQ / AWQ | **+25,31 %** | [+20,01 ; +30,84] | +11,50 | 12/12 |

🚨 **Ce que ça retire** : le §4 de l'entrée du matin écrit « **LE 4B N'A PAS
D'INTERVALLE ET N'EN AURA PAS sans rejeu** […] ce n'est pas une des trois
paires qui manque, ce sont les trois ». Authenticité établie **avant** usage,
comme pour les dumps de la veille : les moyennes de NLL rendent
12,2369 / 13,5207 / 16,9422 — les trois ppl publiées depuis le 2026-08-06, au
dix-millième — et le contrôle d'additivité des différences appariées tombe
**exactement à 0,0**, ce qu'un décalage d'une fenêtre entre deux bras
casserait. Les six intervalles 8B/14B du matin sont **reparsés par le même
code** et retombent au chiffre : deux implémentations indépendantes d'accord.

🚨 **2. LE GENOU SE DÉDOUBLE — DEUX MÉTRIQUES, DEUX VERDICTS, ET C'EST UNE
INFORMATION.** Le pas 4B→8B devient testable, donc le genou de perplexité
entier — c'est ce pas qui porte le « −43 % », c'est-à-dire toute la force de
l'argument de ralentissement.

| métrique | pas 4B→8B | pas 8B→14B | le ralentissement |
|---|---|---|---|
| **perplexité** *(apparié, 12 fenêtres, même texte aux trois tailles)* | ×0,881211 [0,856 ; 0,907], t = −9,62, 12/12 | ×0,974855 [0,959 ; 0,991], t = −3,38, 10/12 | ✅ **RÉSOLU** — pas1 − pas2 = **−0,100992** [−0,137670 ; −0,064313], t = −6,06, 11/12 |
| **écart MMLU au 4 bits** *(non apparié entre tailles, SE en quadrature)* | −6,96 pp, p = 0,0001 | −1,40 pp, p = 0,40 | ❌ **NON RÉSOLU** sur le second pas |

Le « −43 % » publié depuis le 2026-08-10 reçoit enfin sa barre : la fonte de
l'**excès** vaut **−42,8 %, IC95 [−51,8 ; −33,5]** du 4B au 8B, contre
**−13,9 %, IC95 [−22,8 ; −4,9]** du 8B au 14B. 🕳️ Le point est reproduit à
0,2 point près, et l'ancien « −42 % » de `CLAUDE.md` en était la **troncature**.

🚨 **Ce que ça retire, et ce que ça NE retire PAS.** L'entrée du matin écrit
« du côté **perplexité** le genou n'est pas même testable, pour une raison de
conservation » — **démenti**. Elle écrit aussi « **le genou ne tient pas au
test** » : **vrai, et toujours vrai, SUR MMLU** (8B→14B : −1,40 pp, SE 1,68,
z 0,83, **p = 0,40**). Les deux tiennent ensemble parce qu'elles ne parlent pas
de la même métrique. Trois mécanismes l'expliquent sans supposer d'erreur nulle
part : **(1) la puissance** — la perplexité est appariée *entre tailles* (la
fenêtre *i* est le même texte aux trois campagnes, empreinte
`3f1baca9033bf251`) et pèse **49 140 tokens scorés**, là où MMLU compose deux
campagnes indépendantes de **2 280 questions** sans appariement possible ;
**(2)** les deux ne mesurent pas la même chose — le 2 bits abîme le
**raisonnement** bien plus que la **restitution** (verdict du 2026-08-02), et
c'est la restitution qu'un corpus de perplexité mesure surtout ; **(3)** les
deux lignes ne comparent pas les mêmes bras, mais sur la référence AWQ la
perplexité **reste résolue** (−0,057562 [−0,101127 ; −0,013997], t = −2,91),
donc le changement de référence n'explique pas l'écart à lui seul.

🚨 **RÈGLE DE RÉDACTION, IMPÉRATIVE ET APPLIQUÉE CE SOIR À TOUTES LES SURFACES
VIVANTES** (`CLAUDE.md` §3ter/§3bis/§6, `echelle-4b-8b`, `PLAN.md`,
`note-produit`, `cheatsheet-defense`, `README.md`, `data/README.md`) : **toute
phrase sur le genou doit NOMMER SA MÉTRIQUE.** « Le genou tient » nu est faux
de moitié ; « le genou ne tient pas » nu l'est de l'autre moitié. La forme
juste : *le ralentissement est résolu en perplexité et ne l'est pas sur l'écart
MMLU au 4 bits.* ⚠️ Et p = 0,40 ne prouve toujours pas l'égalité sur MMLU : les
données y restent **muettes**, et une seconde métrique qui répond ne rend pas
la première bavarde.

⚠️ **LA RÉSERVE QUI VAUT POUR LES NEUF INTERVALLES ET MORD PLUS FORT SUR LE
GENOU.** Ils portent **une seule** source de variabilité, l'échantillonnage des
12 fenêtres de wikitext-2. Le tirage de **calibration** en est absent aux trois
échelles, et l'y ajouter en empruntant le σ de 0,7 % (mesuré sur 3 blocs de
Qwen3-0.6B) serait **fabriquer un nombre**. Le genou compare **trois artefacts
produits chacun une fois** : un t = −6,06 sur la variabilité de corpus ne dit
rien de ce que trois autres graines auraient donné. **Trois points ne font
toujours pas une loi, et le genou résolu ne dit pas où la courbe s'aplatit.**

✅ **3. LE DISQUE DU 14B EST ACQUIS, et il l'était sans qu'aucune surface le
dise.** `qwen3-14b-llvq.bin` pèse **6 506 354 741 o = 6,506 Go** — *mesuré*,
confirmé à l'octet par `hf buckets ls` **et** par le log de scellement, donc par
deux routes indépendantes. 🕳️ **Ce point disait « le triptyque du 14B est donc
disque acquis, vitesse manquante, VRAM carte manquante : deux cellules vides,
pas trois — aucune des deux n'est comblée par cette entrée » : c'était exact au
moment où il a été écrit, et les DEUX cellules ont été comblées le soir même**
(42,9 tok/s, 9,39 Go —
[`mesures/fusedrun-14b-2026-08-17.txt`](mesures/fusedrun-14b-2026-08-17.txt)).
La prudence de la dernière phrase était la bonne : elle n'anticipait pas la
mesure, elle l'attendait.

🕳️ **4. DETTE DE REGISTRE SOLDÉE : `jobs.csv` s'arrêtait au 2026-08-13** et
manquait les deux jobs du 08-16 — `6a814ba31f5885ae605bcb55` (llvq-e1v, l40sx1,
1 697 s, **0,85 $**) et `6a81b2b71f5885ae605bdcc9` (llvq-nullk, l40sx1, 1 544 s,
**0,77 $**). **Le total facturé passe de 55,59 à 57,21 $.** ⚠️ Le 55,59 n'a pas
bougé pour autant : il devient un **sous-total** — aucune cellule du papier ne
repose sur ces deux jobs, donc les y fondre aurait gonflé « le coût de cette
évidence » sans rien ajouter à ce qu'il paie. Détail et provenance des deux
montants dans [`data/README.md`](data/README.md).

## 2026-08-17 (second lot du soir) — Le bras AWQ a enfin un tok/s, dans SON moteur : ×2,41 chez vLLM contre ×1,12 chez nous, et le résultat est contre nous

> ⚠️ **Cette entrée ne retourne aucun verdict antérieur.** Elle remplit une
> cellule **vide depuis le premier jour** — la vitesse du bras AWQ — et laisse
> intact l'interdit qui l'entourait : les deux rapports ne se divisent pas.

✅ **1. LA PREMIÈRE MESURE DE VITESSE DE L'AWQ DANS SON PROPRE MOTEUR.** Aucun
tok/s vLLM n'existait sur ce projet, sur aucun matériel, pour aucun modèle :
`fusedrun` ne sait pas charger un checkpoint AWQ, et l'AWQ que le banc du
2026-08-10 chronomètre est un **portage** de la GEMV de mit-han-lab dans notre
harnais — une mesure de **noyau**, pas de **produit**. Job
`6a830d53e55292eada79b600`, l40sx1, **226 s de running = 0,11 $** contre un
plafond annoncé de 1,35 $ ; image **épinglée** `vllm/vllm-openai:v0.26.0`
(digest `sha256:ffb2d59b1c059a5b…`) ; pré-enregistrement
[`../proofs/preregistration-awq-vllm-2026-08-17.md`](../proofs/preregistration-awq-vllm-2026-08-17.md)
**commité avant le lancement**. Trois bras **vivants simultanément dans un seul
processus**, `prompt_token_ids` passé en dur (aucune re-tokenisation possible),
128 tokens `ignore_eos`, `--dtype float16` forcé, `enable_prefix_caching`
**relu à False** dans la config résolue et non supposé depuis le drapeau,
2 générations jetées puis **5 chronométrées**, prefill compris :

| bras (Qwen3-4B) | méd. tok/s | plage | étendue |
|---|---|---|---|
| f16 | **83,09** | [83,08 ; 83,11] | 0,03 % |
| `awq_marlin` | **200,49** | [200,39 ; 200,61] | 0,11 % |
| `awq` forcé | **200,69** | [200,54 ; 200,80] | 0,13 % |

**Rapports intra-pile, formés round par round** (jamais un quotient de deux
minima) : `awq_marlin`/f16 = **×2,413** [2,412 ; 2,414] · `awq`/f16 =
**×2,415** [2,414 ; 2,417]. Contrôle de chargement : premier token
`id 12095 ' Paris'`, identique à
[`mesures/planes14-fusedrun-2026-08-06.txt`](mesures/planes14-fusedrun-2026-08-06.txt) ;
la divergence **après** le premier token est attendue (deux moteurs, deux
ordres d'accumulation) et n'invalide rien. Tous les gardes verts,
`violations` vide.

🚨 **2. LA LECTURE PUBLIABLE, ET C'EST LA SEULE : ce que la quantification
achète DANS SA PROPRE PILE.** **×2,413** [2,412 ; 2,414] pour le 4 bits chez
vLLM (médiane de 5 rapports) · **×1,12** pour nous chez nous (4B, à tête
identique — 48,7 / 43,6, *quotient de deux points uniques, sans plage*). **Les
deux moteurs diffèrent : 2,413 et 1,12 ne se soustraient ni ne se divisent.**
🚨 **Le résultat est contre nous, et le §6 du pré-enregistrement l'a déclaré
publiable tel quel avant de le connaître.** Il est cohérent avec ce que le
dossier savait par un autre chemin : notre noyau atteint **65 %** de sa borne
d'octets là où l'AWQ porté en atteint **88 %**
([`mesures/six-arm-awq-2026-08-10.txt`](mesures/six-arm-awq-2026-08-10.txt)).
Aucun titre, aucun abstract ne change — ils ne revendiquent aucun avantage de
vitesse contre l'AWQ. ⚠️ Et la légende doit dire que **M = 1 n'est pas le
régime optimal d'une GEMM Marlin** (plus petite tuile en M = 8) : ce ×2,413
**ne majore pas** ce que l'AWQ sait faire — condition (c) de l'amendement du
§5, posée d'avance.

🚨 **3. LE ROUTAGE TRANCHE — ET IL TRANCHE CONTRE LA LECTURE ÉVIDENTE.** Le log
du job porte deux fois, depuis deux processus moteur distincts (pids 501 et
813), `[auto_awq.py:473] Using MarlinLinearKernel for AutoAWQMarlinLinearMethod`,
et la config résolue des **deux** bras AWQ lit `quantization=auto_awq` — alors
que l'un a demandé `None` et l'autre `"awq"`. Dans vLLM 0.26.0, `"awq"` est
normalisé en `auto_awq`, qui choisit Marlin sur Ada. **Donc le bras « awq
forcé » n'a PAS isolé la GEMM AutoAWQ : les deux bras ont exécuté le même
noyau.** Trois conséquences, toutes dans le sens du moins qu'on peut dire :

- ❌ **Le lien avec le banc du 2026-08-10 n'est pas établi.** C'est l'issue
  prévue au §6 (« le bras `awq` forcé refuse de démarrer → publier
  `awq_marlin` seul, et écrire en toutes lettres que le GEMM AutoAWQ n'a pas
  pu être isolé »), sous une forme que le §6 n'avait pas anticipée : le bras a
  démarré, il n'était simplement **pas le bras demandé**.
- 🕳️ **LES 0,10 % ENTRE LES DEUX BRAS NE CONFIRMENT PAS LA CLAUSE « M = 1 » DU
  2026-08-10** (« à M = 1 tous les noyaux 4 bits convergent vers la même borne
  de bande passante »). **La vérifier demande DEUX noyaux ; ici il n'y en a
  qu'un, chargé deux fois.** 200,69 contre 200,49 mesure la **reproductibilité
  d'un seul noyau** entre deux instanciations du moteur — un chiffre utile,
  mais un **autre** chiffre. La clause reste **NON TESTÉE** par ce job, dans un
  sens comme dans l'autre. ⚠️ Le journal signale avoir failli publier
  l'inverse : « forcer `awq` ne change rien, donc les deux noyaux convergent »
  est exactement ce que la table des débits suggère **quand on ne lit pas la
  ligne de routage du log**. C'est le motif habituel du dossier — une
  conclusion vraie de forme, tirée d'un instrument qui n'a pas mesuré ce qu'on
  croit.
- ✅ **Ce que les 0,10 % bornent, en revanche : la résolution de ce banc.**
  Aucun effet sous ~0,1 % ne s'y tranche.

🕳️ **4. UN DÉFAUT DE L'INSTRUMENT, RELEVÉ PLUTÔT QUE LAISSÉ.** Le JSON commité
(`data/awq-speed-4b-2026-08-17.json`) porte un champ `kernel_log` par bras et
**ne contient pas** les deux lignes Marlin : vLLM V1 exécute son EngineCore
dans un **processus séparé** (`spawn`), donc le `LogTap` du parent ne voit pas
les enregistrements des pids 501 et 813. **La preuve du routage n'existe que
dans la sortie standard du job**, d'où sa recopie mot pour mot dans le journal.
Le garde « aucune ligne de log sur le routage » **n'a pas tiré**, puisqu'une
ligne — la mauvaise — avait bien été captée : **un garde vert sur une capture
incomplète**.

⚠️ **5. CE QUE LA MESURE AJOUTE, ET QU'IL NE FAUT PAS SUR-LIRE.** Le bras f16
est le **même modèle dense, même dtype, même prompt, même carte** des deux
côtés : vLLM **83,09** tok/s (*mesuré*) contre **43,6** chez nous (*mesuré le
2026-08-06*), soit **×1,91** (*calculé*). C'est la première mesure du
**confondant de moteur** sur un travail identique. 🚨 **Mais elle n'est pas
décomposable** : notre bras dense porte le défaut connu de `broadcast_matmul`
(778 Mo de vocabulaire recopiés par token), donc ce ×1,91 mélange « qualité du
moteur » et « notre propre défaut », et **ce job ne les sépare pas**. Écrire
« le moteur vaut ×1,9 » serait une **inférence**, pas une mesure.

🚨 **6. LES CINQ INTERDITS DU §4, ÉCRITS AVANT LA MESURE ET TOUJOURS EN
VIGUEUR.** (i) **aucune** phrase « notre noyau est plus rapide / plus lent que
l'AWQ » — les deux rapports vivent dans des piles différentes et **ne se
divisent pas** ; l'écart bout-en-bout est dominé par vLLM contre candle.
(ii) **la cellule vitesse AWQ des tables du papier RESTE VIDE** — elle est
désormais *expliquée*, pas *remplie* ; ce qui change est la note de bas de
tableau. (iii) **aucun chiffre de VRAM ne sort de vLLM** (il préalloue : ce
qu'il rapporte est une réservation, pas une occupation). (iv) le rapport maison
cité en regard est **×1,12** (4B, à tête identique), jamais ×2,03 — ni ×2,61 au
8B. (v) le biais de notre bras f16 est nommé **avec sa direction** : il porte
des deux côtés à tête identique, donc il **tire nos rapports vers 1** — **nous
sous-estimons notre propre avance**.

⚠️ **7. DETTE CONCRÈTE ET SOLVABLE : LE 8B AWQ EST BLOQUÉ.** `ops/awq_speed.py`
le porte en `pinned=False` et le **refuse** sans `--allow-unpinned-revision`,
parce que ses deux révisions (`4da05a8e…` pour `Qwen3-8B-AWQ`, `b968826d…` pour
`Qwen3-8B`) ont été **relevées au Hub le 2026-08-17 et n'ont aucune entrée
`EXPECTED` nulle part dans le dépôt** — elles n'ont donc jamais passé les
contrôles structurels d'`ops/awq_dequant.py`. **Une révision que personne n'a
validée n'est pas un épinglage, c'est un instantané.** Ce n'est pas un mystère :
la lever demande de faire passer `ops/awq_dequant.py check` sur ces deux
révisions et d'écrire l'entrée dans `EXPECTED` — après quoi le bras 8B se
mesure au même tarif que le 4B. ⏳ **Le 14B, lui, attend son vis-à-vis maison**
(`fusedrun` 14B) ; **aucun de ses résultats n'existe au moment où ceci
s'écrit, et rien ici n'en anticipe un.**

**Coût du lot : 0,11 $** — le total du registre est à **57,56 $**
([`data/README.md`](data/README.md), qui porte les quatre jobs du 08-17).
📄 [`mesures/awq-vllm-4b-2026-08-17.txt`](mesures/awq-vllm-4b-2026-08-17.txt) ·
[`data/awq-speed-4b-2026-08-17.json`](data/awq-speed-4b-2026-08-17.json) ·
[`../proofs/preregistration-awq-vllm-2026-08-17.md`](../proofs/preregistration-awq-vllm-2026-08-17.md) ·
[`../ops/awq_speed.py`](../ops/awq_speed.py)

## 2026-08-17 (troisième lot du soir) — Le 14B est SERVI : l'entrée qui manquait à ce fil

> 🕳️ **RATTRAPAGE, écrit le 2026-08-25.** L'entrée du matin note elle-même
> « ⚠️ **ce fil n'a pas encore son entrée pour ce run** — le journal fait foi en
> attendant », et l'entrée du second lot du soir écrit « le 14B attend son
> vis-à-vis maison ; **aucun de ses résultats n'existe au moment où ceci
> s'écrit** ». Les deux étaient exactes à la minute où elles ont été écrites, et
> le run avait déjà eu lieu quand la seconde l'a été. **Un fil chronologique qui
> renvoie à un journal « en attendant » est un fil qui a un trou** : le voici
> comblé, sans un chiffre ajouté à ce que le journal portait.

✅ **Le 14B tourne sur le chemin fusé** (job `6a83121be55292eada79b611`, l40sx1,
2 472 s = **1,24 $**, pré-enregistrement commité avant le lancement ;
[`mesures/fusedrun-14b-2026-08-17.txt`](mesures/fusedrun-14b-2026-08-17.txt)).
`Planes14` + `LLVQ_EMBED=q8`, 128 tokens **identiques** au bras dense :
**42,9 tok/s dans 9,39 Go** contre **17,0 tok/s dans 29,54 Go** (*mesuré* ; les
Go sont un **compte d'octets hôte** imprimé par `fusedrun`, jamais une lecture
de `nvidia-smi`).

- ✅ **Mémoire ÷3,14** — direct, deux lectures de la même carte par le même
  instrument. ⚠️ Le binaire forme ses rapports sur ses valeurs internes ; diviser
  les cellules **arrondies** rend ÷3,15 et ×2,52. **On publie ceux du binaire :
  ÷3,14 et ×2,53.**
- 🚨 **Le brut ×2,53 ne se publie JAMAIS seul** : son dénominateur est *notre*
  bras dense, handicapé, et le handicap est **maximal à cette taille** — têtes
  déliées, donc 1 555,8 Mo de vocabulaire recopiés par token (`Head::project` →
  `broadcast_matmul`), que le profil fencé price à 53,9 ms contre 1,2.
- ❌ **Le rapport « à tête identique » N'EXISTE PAS au 14B**, et il n'est pas
  dérivable du profil fencé : les deux sens du remplacement rendent ×1,78 et
  ×1,24 (facteur 1,44), et les totaux fencés ratent de **71 %** le rapport
  mesuré de leur propre invocation. Il se produirait par un **run**
  (`LLVQ_EMBED=f16`), pas par un calcul. ✅ **Ce trou est comblé le 2026-08-18
  par B2 — ×1,41 [1,40–1,41], mesuré** ; voir l'entrée suivante.
- ✅ **Recoupement posé d'avance** : 9,39 Go × 8 ÷ 14 768 307 200 =
  **5,0866 b/param** contre **5,106** par `rtbits`, soit **−0,38 %** dans une
  bande de ±0,5 %. Le 14B a son **troisième instrument**.
- 🕳️ **Et « ×2,53 est le plus élevé des trois tailles » est FAUX** : 4B ×2,03 ·
  **8B ×2,61** · 14B ×2,53. La suite n'est **pas monotone**, la part du
  `lm_head` décroît monotonement (9,67 % · 7,60 % · 5,27 %, *calculé*) et ne
  rend donc pas l'ordre observé — **aucun mécanisme n'est revendiqué**.

⚠️ Ni la qualité ni aucune **plage** n'ont été mesurées à cette taille par ce
run : c'est un point unique, et c'est exactement ce que B2 corrige le lendemain.

## 2026-08-18 — Les points deviennent des plages, le 8B ressuscite pour 0,24 $, et le dépôt cesse de se contredire

Quatre lots dans la journée, **2,57 $** facturés au total (*calculé* sur
[`data/jobs.csv`](data/jobs.csv) ; B3 tourne sur `cpu-xl`, les trois autres sur
l40sx1). Le fil commun est
qu'aucun ne cherche un layout : ils **solidifient ce qui existe** — remplacer
des points par des médianes, retrouver un objet perdu, mesurer un dénominateur,
et faire taire les contradictions entre surfaces d'entrée.

✅ **1. B2 — `fusedrun` PASSE DU POINT UNIQUE À LA MÉDIANE SUR 5 ROUNDS, AUX
TROIS TAILLES** (jobs `6a84b268` / `6a84b26b` / `6a84b269`, l40sx1, ~2,25 $ ;
pré-enregistrement tamponné avant lancement ;
[`mesures/b2-fusedrun-plages-2026-08-18.txt`](mesures/b2-fusedrun-plages-2026-08-18.txt)).
1 génération jetée + 5 chronométrées par bras, rapport = **quotient des
médianes avec enveloppe** — les rounds des deux bras ne coexistent jamais, donc
aucun rapport round par round n'est licite ici et le journal le déclare.

| taille | config | fusé tok/s [plage] | dense tok/s | × médianes [env.] | Go fusé/dense (÷) |
|---|---|---|---|---|---|
| 4B | q8 (servi) | **87,0** [86,8–87,0] | 43,6 | ×2,00 [1,99–2,00] | 2,56 / 8,04 (÷3,14) |
| 4B | f16 (tête id.) | 48,3 [48,1–48,3] | 43,5 | **×1,11** [1,11–1,11] | 2,93 / 8,04 (÷2,75) |
| 8B | f16 (tête id.) | 34,1 [34,0–34,1] | 26,4 | **×1,29** [1,29–1,29] | 6,58 / 16,38 (÷2,49) |
| 8B | q8 (servi) | 68,2 [68,2–68,3] | 26,5 | ×2,57 [2,57–2,58] | 5,41 / 16,38 (÷3,03) |
| 14B | q8 (servi) | 43,3 [43,2–43,4] | 17,0 | ×2,55 [2,54–2,56] | 9,39 / 29,54 (÷3,14) |
| **14B** | **f16 (tête id.)** | **23,9** [23,8–24,0] | 17,0 | **×1,41** [1,40–1,41] | 10,85 / 29,54 (÷2,72) |

🆕 **LE FAIT NEUF QUE SEULES LES PLAGES RÉVÈLENT : à tête identique la série est
STRICTEMENT CROISSANTE — ×1,11 → ×1,29 → ×1,41 — là où la série brute
(×2,00 · ×2,57 · ×2,55) n'a AUCUN ordre.** C'est la série à tête identique qui
mesure le noyau, et elle dit : *plus le modèle grossit, plus le noyau et le
format paient*. La brute est dominée par le handicap variable du bras dense.
✅ **Et la cellule 14B à tête identique — que le dossier déclarait inexistante
et non dérivable — est MESURÉE** : les deux reconstructions du profil fencé
rendaient ×1,78 et ×1,24, la mesure tombe **entre les deux**, ce qui règle la
question de savoir laquelle croire — **aucune**.
🕳️ **Les points historiques ne sont pas démentis, ils sont ENCADRÉS.** Écarts à
l'ancrage : −1,6 % (87,0 vs 88,4), −0,8 % (48,3 vs 48,7), −0,9 % (34,1 vs 34,4),
−1,6 % (68,2 vs 69,3), +0,9 % (43,3 vs 42,9), 0 % (17,0). Aucun ne sort de la
bande de ±10 % posée d'avance, et l'écart 88,4/86,9 des journaux d'époque
**s'explique** : c'était la dispersion inter-invocations qu'un point unique ne
pouvait pas voir. ⚠️ Le « 2,60 Go » d'époque était l'**affichage carte arrondi** ;
2,56 est le **compte d'octets hôte**, le même instrument que le ÷3,14 du 14B.
🕳️ **Erratum sans effet sur les mesures** : le job 14B a dépassé son propre cap
(42,5 min contre 40) et HF l'a tué en statut « ERROR — Job timeout », **après**
la dernière ligne de mesure. Les deux invocations sont complètes dans les tee du
bucket et dans les logs. Le cap était calibré sur le 08-17 à **une** invocation ;
deux invocations 14B en demandent ~43.

✅ **2. B3 — LE 8B EST RE-SCELLÉ DEPUIS LE BUCKET POUR 0,24 $, ET IL REJOUE AU
MILLIÈME** (job `6a84ab2b`, cpu-xl, 856 s ;
[`mesures/b3-8b-reseal-2026-08-18.txt`](mesures/b3-8b-reseal-2026-08-18.txt)).
Les trois critères pré-enregistrés sont verts : taille **4,324 Go** dans
[4,25 ; 4,40] · `rtbits` local reproduit **`params_total` = 8 190 735 360 exact**
et **`Planes14` + q8 = 5,322 b/param modèle entier au millième** · le fichier
s'ouvre sous l'empreinte de codebook `0x338f_420f_1186_6319`, qui refuse
mécaniquement toute autre. **Coût total de la « perte » du 8B : 0,24 $ et
14 minutes**, contre les **12,61 $** de requantification que le plan
provisionnait — la règle « épuiser les canaux de rétention » du §7, appliquée
une troisième fois et payante une troisième fois.
⚠️ **Ce que ça n'établit pas, et le journal le dit avant qu'on le lui demande** :
l'égalité d'octets avec l'original perdu est **invérifiable par construction**
et n'est pas revendiquée. Le re-scellé est l'objet des campagnes **au sens des
grandeurs dérivées**. Le filet aval est B2-8B, qui compare ses tokens gloutons
au bras dense — **128 identiques**.
🕳️ **Et une correction de fait, datée du lendemain, sur un pré-enregistrement
TAMPONNÉ.** Le préreg B3 affirme que le run 8B d'origine portait « graine
1000000 » ; **faux** — l'argument 7 de `smoke` est `blocks`, une sentinelle
« tous les blocs », et la graine vit dans `LLVQ_CALIB_SEED`, absent de
l'environnement du job. Le run d'origine a donc tourné **graine non posée**,
c'est-à-dire préfixe contigu. La conclusion « aucun re-tirage » reste vraie, et
plus fortement qu'écrit : *il n'y avait aucune graine à récupérer.*
🚨 **Le pré-enregistrement n'est PAS édité** — une ancre atteste des octets à
une date, la corriger pour rendre le texte juste détruirait ce qu'elle prouve
(le défaut réalisé sur les préregs des 08-10 et 08-11). Le document reste faux
sur ce point, le journal est l'endroit où ça se lit, et les deux se citent.

✅ **3. F1 — LE DÉNOMINATEUR EXISTE : NOTRE TÉMOIN FP16 EST AU NIVEAU DE cuBLAS**
(job `6a84b267`, l40sx1, 163 s, **~0,08 $** ;
[`mesures/f1-cublasf16-2026-08-18.txt`](mesures/f1-cublasf16-2026-08-18.txt)).
`r = médiane(t_témoin ÷ t_cuBLAS)`, formé **round par round** (les deux bras
coexistent dans chaque round, donc la forme est licite ici) : **1,024** en
phase à 2 bras, **1,015** en phase à 5 bras, contre un seuil de 1,05 posé
d'avance. **Tous les rapports « vs FP16 » publiés depuis le 2026-08-05
tiennent** — c'est la bande 1 du pré-enregistrement, et ni la bande 2
(requalification obligatoire) ni la bande 3 ne s'appliquent.
🔎 **Ce que le job établit sans l'avoir cherché** : l'échelle publiée **rejoue
sous le nouveau dénominateur et dans le même processus** — `Planes14` 2,16×
[2,16–2,16], AWQ 3,37×, `nullk` 4,79× — cohérents au centième avec les journaux
des 08-11 et 08-16. En Go/s : témoin 661, cuBLAS 671-676. **L'écart est du flux,
pas de l'ALU.** ⚠️ Réserves inchangées : `CUBLAS_GEMM_DEFAULT` (pas le meilleur
`GemmEx` possible) et sortie binary16 côté cuBLAS contre f32 côté témoin —
**déclarées au bénéfice de cuBLAS**.

✅ **4. LE LOT « DÉPÔT SANS CONTRADICTION », 0 $.** Un sweep adversarial à trois
rapporteurs après la passe d'hygiène rend **huit prises**, et il faut les lire
comme une classe et non comme huit bugs : *une surface d'entrée qui contredit le
code envoie une session entière au mauvais endroit.* `LAUNCH_ME` niait **trois
fois** le cache KV que `bin/run` porte depuis `9c24d26` ; le cheatsheet
recommandait en réponse **quatre valeurs rétractées** ; le papier se
contredisait sur la cellule mémoire du 4B ; la carte HF **ouvrait sur le chiffre
Metal qu'elle proscrit** ; la CI annonçait 3 tests ignorés là où il y en a
**16** ; et le garde de budget de bits fraîchement écrit pouvait **déborder
u64**. Chaque correction **se cite**.
🕳️ **Et une prise de code qui n'était pas cosmétique** : `BitWriter::push` à
`width = 64` était un **shift de 64 bits** — panique en debug, correct par
accident en release parce que l'accumulateur valait 0 à ce point. Corrigé,
`llvq-search --test g6_pack` **passe désormais dans les deux profils**, et la CI
gagne le **leg debug** qui l'aurait attrapé. La note de `CLAUDE.md` qui disait
« échoue en debug, ne pas le lire comme une régression » était **exacte du 08-15
au 08-18, et décrivait un bug réel**. Dans la foulée, chaque longueur lue par le
lecteur d'archive devient **une revendication à vérifier** (allocations bornées
par les octets qui arrivent, budget de bits validé avant `BitReader`, cap de
coquille validé avant l'assert de `llvq-search`) — **quatorze tests hostiles**
couvrent des variantes d'erreur qui n'avaient aucune assertion.

## 2026-08-19 — Trois lots, trois verdicts : la soumission hôte est négligeable, l'A100 renverse l'échelle, et le σ de calibration est faux d'un facteur 7

**23,31 $** de carte dans la journée (*calculé* sur
[`data/jobs.csv`](data/jobs.csv)), dont **21,45 pour le seul F5** — la deuxième
journée la plus chère du projet, derrière les 27,67 $ de la quantification 14B
du 2026-08-09. Les trois lots sont indépendants et pré-enregistrés séparément.

✅ **1. F3 — LES EVENTS CUDA : L'ÉCART HÔTE−DEVICE VAUT 0,1-0,2 %, DEUX ORDRES
DE GRANDEUR SOUS L'ATTENTE** (job `6a8567b4`, l40sx1, 1 729 s, **~0,86 $** ;
[`mesures/f3-events-2026-08-19.txt`](mesures/f3-events-2026-08-19.txt)).
`LLVQ_TIME_EVENTS` chronomètre chaque bras par events device en plus du
wall-clock hôte : l'écart vaut **4 à 8 µs par round ENTIER** sur les neuf bras,
là où le pré-enregistrement annonçait « 0,5-2 ms ». **Dans ce banc, la
soumission hôte est entièrement recouverte** — pendant que le noyau *N* tourne,
l'hôte enfile *N+1*.
⚠️ **Ce que ça élimine est UNE hypothèse, pas le poste.** Les écarts
**inter-noyaux** sur le stream sont **dans** le span device : l'écart hôte−device
est un **minorant** du poste latence. Ce qui tombe, c'est « le poste latence,
c'est l'hôte qui n'arrive pas à suivre » ; le poste reste **device** (bulles
inter-noyaux, montée en occupation). 🚨 **Et deux dénominateurs à ne pas
confondre** : F3 mesure le **banc** (252 matrices en boucle serrée), là où
l'attribution 39/33/19 du 2026-08-05 mesure le **chemin modèle** (candle,
2,04 ms/token, un autre dispatcher). F3 rend l'hypothèse moins probable
ailleurs **sans la réfuter**.
✅ **N1 — l'issue la plus tranchante des trois se réalise, et elle est
déclarée** : `ncu` **s'installe et s'attache** au processus, puis la plateforme
répond `ERR_NVGPUCTRPERM`. Ce n'est donc pas « outillage indisponible » mais
**la plateforme refuse les compteurs** — ça ne se contourne pas par un meilleur
script, ça demande un droit que HF Jobs ne donne pas. **F3 est CLOS sans
retentative non pré-enregistrée**, et le papier documente l'indisponibilité
comme un **fait de plateforme**, pas comme une négligence.
✅ **D1 — dette du 2026-08-05 soldée** : le premier contact carte interrogeait
`driver.version`, champ invalide jamais corrigé ; le champ est `driver_version`,
et tous les runs de cette image portent désormais leur driver — **580.159.03**.
⚠️ Les runs antérieurs gardent un driver inconnu : **déclaré, pas comblé.**

🚨 **2. F4 — SECONDE ARCHITECTURE : SUR A100, AUCUN BRAS À DÉCODAGE NE BAT
FP16** (job `6a8559fc`, a100-large, 1 034 s, **~1,00 $ estimé** — premier job
`a100-large` du registre, tarif jamais observé ;
[`mesures/f4-a100-2026-08-18.txt`](mesures/f4-a100-2026-08-18.txt)). C'est la
branche « résultat à publier » du pré-enregistrement qui se réalise.

| bras | méd ms A100 | vs FP16 A100 | Go/s(min) | repère L40S |
|---|---|---|---|---|
| `nullk` (plancher) | 4,107 | **1,68×** | 18 | 4,79× |
| FP16 (témoin) | 6,915 | 1,00× | 1 052 | 661 Go/s |
| FP16 cuBLAS | 6,041 | 1,14× | 1 204 | 1,02× |
| AWQ w4g128 | 3,793 | **1,82×** | 501 | 3,37× · 584 |
| LLVQ `Planes14` | 8,742 | **0,79×** | 250 | 2,16× · 425-427 |
| LLVQ `Slot32` | 9,413 | 0,73× | 266 | 1,87× · 428 |
| LLVQ `Planes12x` | 9,423 | 0,73× | 209 | 1,98× · 356 |
| LLVQ `Golay70` v2 | 11,121 | 0,62× | 147 | 1,77× · 263 |
| LLVQ `Golay70` v1 | 15,705 | 0,44× | 104 | 1,31× · 195 |

**Le mécanisme est mesuré, pas supposé.** FP16 et cuBLAS **convertissent** la
HBM (661 → 1 052 et 676 → 1 204 Go/s) pendant que **tous** les bras à décodage
**ralentissent en absolu** et voient leurs Go/s effectifs **chuter** (425 → 250 ·
428 → 266 · 356 → 209 · 263 → 147). *Une borne mémoire ne produit pas ça* : sur
A100 ces noyaux sont bornés par le **calcul par SM**. Le plancher `nullk` passe
de 2,305 à 4,107 ms et ne vaut plus que 1,68× — le sol latence/lancement mange
**59 %** du temps FP16 sur A100 contre **21 %** sur L40S.
✅ **Ce que ça achète au papier, et c'est plus fort que ce que ça coûte** : le
claim *« decode at matvec speed »* cesse d'être un point unique avec une
limitation « une seule carte » et devient un résultat **L40S/Ada à domaine de
validité MESURÉ**. La lecture roofline en sort renforcée : c'est elle qui
explique l'inversion, et la figure gagne un second point d'ancrage.
✅ V1 (les 11 noyaux compilent et se chargent en `sm_80`, `LLVQ_NVRTC_ARCH` fait
ce qu'il dit) et V2 (vérification f64 ligne à ligne : **pires erreurs identiques
à celles du L40S** — l'arithmétique des noyaux ne dépend pas de la carte) sont
verts. ⚠️ **Les × inter-cartes NE SE DIVISENT PAS**, règle du §3 du préreg.
⚠️ L'hypothèse « c'est le rapport des fréquences SM » est posée ici **sans être
tranchée**, aucun preflight n'imprimant l'horloge — elle le sera par le lot G le
2026-08-23.

🚨 **3. F5 — LE TIRAGE DE CALIBRATION VAUT 10,3 % D'ÉTENDUE, SOIT SEPT FOIS LE σ
QUE LE DOSSIER SUPPOSAIT** (trois jobs `rtx-pro-6000`, f32, **21,45 $** contre
≤ 22,50 annoncés ; pré-enregistrement **amendé et RÉ-ANCRÉ** avant les runs de
mesure, le tampon de la version morte conservé à part ;
[`mesures/f5-graines-4b-2026-08-19.txt`](mesures/f5-graines-4b-2026-08-19.txt)).
Trois runs **complets** du 4B identiques à un caractère près —
`LLVQ_CALIB_SEED ∈ {1,2,3}` —, même corpus lu dans la même heure, même codebook,
même rotation, et la **même empreinte de tokens d'évaluation que les neuf bras
publiés** (`3f1baca9033bf251`), donc appariables fenêtre par fenêtre :

| graine | ppl scellé f16 | ppl smoke f32 |
|---|---|---|
| 1 | **16,7425** | 16,7725 |
| 2 | **15,8836** | 15,9054 |
| 3 | **15,1027** | 15,1125 |

**Étendue 1,6398 ppl = 10,3 % de la médiane · σ (n = 3) = 0,8202 ppl = 5,2 %.**
🚨 **Les trois paires appariées sont RÉSOLUES** : g1−g2 t = +4,54, g1−g3
t = +10,92, g2−g3 t = +7,68. **Ce n'est pas du bruit de mesure** — le choix des
fenêtres de calibration déplace la perplexité d'un effet que l'instrument sépare
largement de zéro.
🕳️ **Ce que ça retire** : la règle de travail du lot B, « σ ≈ 0,7 %, tout effet
sous ~1,5 % est du bruit », **mesurée sur 3 graines de Qwen3-0.6B à 3 blocs**.
À la taille publiée, sur le modèle publié, avec le protocole publié, σ vaut
**5,2 %**. L'avertissement du §2 de `CLAUDE.md` (« ce σ n'est pas la barre
d'erreur de l'objet publié ») était **juste, et sa prudence insuffisante**.
✅ **Ce que ça N'invalide PAS, et c'est la moitié qui sauve le dossier.** (a) Les
**A/B à fichier constant** — KV q8, layouts runtime, embedding q8, **tous les
verdicts de format** — **ne recalibrent pas** : cette variance ne les touche pas
et leur barre reste l'intervalle apparié à **±0,12 %**. (b) Les artefacts
publiés 4B/8B/14B ont **tous** tourné **sans graine**, donc sur le **même
préfixe contigu** : la courbe d'échelle compare des objets calibrés
identiquement et **ne porte pas cette variance**. (c) Deux verdicts du lot B
tombent en revanche **sous le nouveau plancher de bruit** — l'oracle (−1,6 %) et
la courbe de volume (−1,2 % pour ×13 de tokens) sont tous deux **dans l'étendue
inter-graines** ; leur conclusion « le volume de calibration est plafonné »
**n'est pas renversée**, elle était fondée sur des effets trop petits pour être
distingués, ce qui est désormais **mesurable et le reste**. Le design C (×1,99)
et le swap L ≤ 4 (+4,75 %) restent **hors de portée du bruit**.
🚨 **CE QUI NE SE COMPARE PAS : le point publié n'est pas un quatrième tirage.**
L'artefact publié rend 16,9415, et il est tentant d'écrire que les trois graines
le battent toutes. **Ce serait deux variables à la fois**, et les réserves du
préreg le disaient avant la mesure : le publié est un **préfixe contigu**, pas un
tirage aléatoire — il n'appartient pas à la distribution échantillonnée — et le
shard C4 a changé après lui. **Ce qui est propre est l'étendue ENTRE les trois**,
à corpus et régime constants.
✅ **Contrôle qui rend le reste lisible** : les trois rendent **2,0702 b/poids
effectifs et 1,771 Go scellés**, identiques au fichier publié, et
`verify_artifact` est vert sur les trois. **Seule la qualité bouge ; le débit du
code n'a pas bougé d'un tirage à l'autre.**
🔎 **Une lecture mécaniste, étiquetée HYPOTHÈSE et personne ne l'avait
formulée** : nous calibrons sur ~131 k tokens contre ~100× plus au papier amont.
Un jeu de calibration petit ne déplace pas seulement la moyenne — l'oracle a
montré que non — **il porte de la variance**.

## 2026-08-20/21 — F2 : QTIP entre dans notre banc, et il passe sous notre propre plancher

Deux jours de portage puis **1,44 $** de carte au total sur neuf jobs (contre
≤ 1,80 annoncés ; dont 0,06 **estimé** — le job NVRTC n'a ni runtime ni log),
pré-enregistrement **tamponné avant le premier job**. Le noyau QTIP
amont est **épinglé au commit** et vérifié par sha256, GPL v3, **mesuré et non
redistribué**. Journaux :
[`mesures/f2-p0-recon-2026-08-20.txt`](mesures/f2-p0-recon-2026-08-20.txt),
[`mesures/f2-p2-qtip-exactitude-2026-08-21.txt`](mesures/f2-p2-qtip-exactitude-2026-08-21.txt),
[`mesures/f2-p3-qtip-banc-2026-08-21.txt`](mesures/f2-p3-qtip-banc-2026-08-21.txt).

✅ **1. LE VERDICT PRÉ-ÉCRIT TOMBE, ET IL EST CONTRE NOUS.**
`r = t(Planes14) ÷ t(QTIP)`, phase 2, **mêmes formes, même processus, bras
entrelacés** : **r = 2,27× [2,27–2,28]**, plage **entièrement au-dessus de 1**.
La formulation était pré-enregistrée et **aucune reformulation n'était
permise** : *« le noyau QTIP porté est r× plus rapide que notre meilleur layout
sur ces formes ; la motivation héritée de l'introduction est CONFIRMÉE dans
notre harnais »*. La ligne de table :

| bras | méd ms | Go lus | b/poids | Go/s | vs FP16 |
|---|---|---|---|---|---|
| **QTIP 2 bits** | **2,246** [2,245–2,248] | **0,91** | **2,0000** | 405 | **4,89×** [4,89–4,90] |
| `Planes14` (servi) | 5,103 | 2,18 | 4,804 (noyau) | 425 | 2,15× |

⚠️ **La grandeur comparable est les Go/s, pas le ×** — étiquette « CONCURRENT »
posée d'avance. Le banc n'imprime pas `r` par round ; les deux dérivations du
journal **encadrent la plage vraie par l'extérieur** (conservateur), et le
contrôle par les rapports round-par-round rend 2,274 [2,26 ; 2,28].

🚨 **2. ET L'ERRATUM, QUI EST LE RÉSULTAT LE PLUS UTILE DU RUN : `nullk` N'EST
PAS UN PLANCHER MACHINE.** Le pré-enregistrement **tamponné** écrit au §5
« aucun bras ne peut aller plus vite [que `nullk`] », structurellement, quel que
soit le noyau. **C'est MESURÉ FAUX** : `t(QTIP) = 2,246 ms < t(nullk) =
2,306 ms` — le bras QTIP est **plus rapide que notre passe qui ne lit AUCUN
octet de poids**, séparation **2,7 %** contre une résolution `2R = 0,72 %`. Et
`f = Go/s(QTIP) ÷ Go/s(FP16) = 61,1 %` contre un plafond même-processus de
**59,6 %** : **au-dessus, de 1,5 point pour un δ de 0,2**.
**Le .ots interdit d'éditer le pré-enregistrement ; l'erratum se consigne donc
dans le journal**, et il porte loin :

- 🕳️ **Le « plafond absolu de tout travail de format = 4,77× FP16 » de l'entrée
  du 2026-08-16 est FAUX.** Il faut lire **« plancher de NOTRE géométrie de
  lancement »** — un warp par ligne de sortie, 252 lancements de cette
  géométrie. Le noyau QTIP fait le **même travail de projection**, dans **sa**
  géométrie (`<<<128, 1024, 64 Kio>>>`, 252 lancements aussi), **en lisant
  0,91 Go et en moins de temps que notre passe à zéro octet**.
- 🕳️ **Les « 39 % latence/occupation » de l'attribution du 2026-08-05 tombent
  dans la même classe** : ce sont des propriétés de notre géométrie, qu'un autre
  noyau traverse — et non un poste incompressible de la machine.
- ✅ Lecture du §5ter, case *« f proche du plafond »*, **atteinte par le haut** :
  le noyau QTIP convertit **tout ce que notre plancher permettait, et
  davantage**. Son décodage en treillis est **gratuit au sens du roofline sur
  cette carte**.

🔎 **3. LE MÉCANISME EST NOMMÉ, ET C'EST CE QUI EMPÊCHE DE LIRE LE RÉSULTAT
COMME UN DÉFAUT D'IMPLÉMENTATION.** Les deux formats stockent **2,000 bits de
code par poids sur disque**. Sur les mêmes matrices, le noyau treillis lit
**0,91 Go** là où `Planes14` en lit **2,18** : un **rapport d'octets de 2,40×
pour un rapport de temps de 2,27×**, les deux noyaux convertissant **61 % et
65 %** de leurs bornes d'octets. **À efficacité quasi égale, l'écart de temps
suit l'écart de trafic** — et ce qui fixe le trafic est le **dépliage au
chargement** : un codebook de **1,1·10¹⁴ points ne peut pas tenir dans une table
de correspondance**, là où un **état de treillis 16 bits** y tient (LUT de
2 Kio ; les codebooks tabulés comme l'E8P de QuIP# portent 2¹⁶ entrées).
**L'index de réseau doit donc être déplié en un flux de plans de bits à
4,80 b/poids, et le noyau paie ces octets à la vitesse mémoire. C'est la taille
du codebook qui impose le dépliage, pas notre code.**

✅ **4. CONTRÔLES.** Exactitude : **1 105 920 lignes** contre la référence f64,
pire erreur QTIP **5,4e-8·Σ|w·x|** contre un seuil de **1e-5 — le NÔTRE**, pas
le 1e-3 d'AWQ. Spill : **0 octet local** sur les cinq shims (48-56 registres),
l'opt-in 64 Kio passé sans bruit. Dispersion QTIP **0,13 %** de la médiane.
Dérive inter-phases `R = 0,36 %`, donc **tout écart sous 0,72 % est non
résolu** — et `r` est séparé de 1 par **127 %**.
⚠️ **Asymétries déclarées d'avance et NON corrigées** : QTIP ne porte ni queue
f32 ni échelle de ligne f32 (**en sa faveur**) ; payload pseudo-aléatoire
(licite — code à débit fixe, noyau sans branche dépendante des données) ;
**AUCUNE phrase de qualité ne peut s'appuyer sur ce bras**. Réserve permanente :
QTIP **tel que livré**, réglage figé, **aucun tuning dans aucun sens**.
🕳️ **Le portage a coûté deux runs à une inversion d'opérandes dans
`__byte_perm`** — le mutant a été tué par le test qui manquait, écrit à cette
occasion. C'est la cinquième prise du motif du §5 de `CLAUDE.md` : *faire
exécuter le texte du noyau contre une référence indépendante, pas le relire.*

## 2026-08-23 — Lot G : les horloges tranchent l'A100, et `Planes12x` devient un point SERVI

Trois jobs facturés — plus deux morts avant tout octet lu — pour **1,00 $** au
total (*calculé* sur [`data/jobs.csv`](data/jobs.csv)), pré-enregistrement
tamponné (sha256 `dfd65c42…`). Journal :
[`mesures/g-horloges-planes12x-2026-08-23.txt`](mesures/g-horloges-planes12x-2026-08-23.txt).

✅ **1. G1/G2 — L'HYPOTHÈSE LAISSÉE OUVERTE PAR F4 EST TRANCHÉE, ET PAR UNE
LECTURE PLUTÔT QUE PAR UN ARGUMENT.** Échantillonnage `nvidia-smi` à 1 Hz
pendant toute la durée du banc, sur les deux cartes :

| carte | driver | SM médiane | SM max (active) | throttle |
|---|---|---|---|---|
| L40S | 580.178.04 | **2 520 MHz** | 2 520 MHz | `0x1` (GpuIdle seul) |
| A100 | 580.159.03 | **1 410 MHz** | 1 410 MHz | `0x1` (GpuIdle seul) |

**Les deux cartes tournent ÉPINGLÉES à leur boost max** (datasheet 2 520 et
1 410), **aucun bridage thermique ni puissance** — la seule raison d'événement
d'horloge active est `GpuIdle`, le creux entre noyaux. **Rapport 2 520/1 410 =
1,787**, dans le critère tamponné **[1,60 ; 1,95]** ✅, et il **colle au
ralentissement mesuré du témoin sans lecture** : `nullk` ×1,772 au banc G,
×1,781 au banc F4 publié. **Le ×1,78 de la table A100 EST le rapport
d'horloges** — le §3.5 du papier passe d'« horloges constructeur » à « horloges
**mesurées, épinglées, sans bridage** ».
⚠️ **Réserve inchangée, et elle est celle de F3** : c'est une preuve
d'**horloge**, pas un **profil d'occupation** — les compteurs restent refusés
par la plateforme (`ERR_NVGPUCTRPERM`).
✅ **Contrôle** : le banc **rejoue le papier** sur les bras communs — L40S
`nullk` 4,75× [4,74–4,75] · `Planes14` 2,14× [2,13–2,14] · FP16 669 Go/s (papier
F4 : 4,77× · 2,15× · 661) ; A100 `nullk` 1,70× · `Planes14` 0,79× · FP16
1 055 Go/s (papier : 1,68× · 0,79× · 1 052). Dispersion inter-run, même ordre.

🆕 **2. G3 — `Planes12x` EST SERVI BOUT-EN-BOUT AU 4B, ET C'EST LE POINT SERVI
LE PLUS COMPACT MESURÉ** (job `6a8c2355`, l40sx1, **0,79 $**). Protocole
`fusedrun` de B2 : 1 génération jetée + 5 chronométrées, médiane [plage], tokens
comparés au bras dense.

| bras | tok/s [plage] | Go carte (compte hôte) | b/poids inférence |
|---|---|---|---|
| fusé `Planes12x` + q8 | **85,0** [84,7–85,1] | **2,36** (proj 1,94 + portés 0,41) | 4,277 |
| dense f16 | 43,4 [43,4–43,4] | 8,04 | 16,000 |

**Vitesse ×1,96 [1,95–1,96]** (quotient des médianes — les rounds ne sont jamais
entrelacés), **mémoire ÷3,41**, divergence gloutonne au **token 89/128**, le
tie-break historique de `Planes14` **reproduit**. Les deux prédictions du préreg
tiennent : débit ~84 ∈ [76 ; 90] ✅, VRAM ~2,39 ∈ [2,30 ; 2,48] ✅.
🕳️ **Ce que ça retire** : `Planes12x` traînait depuis le 2026-08-09 l'étiquette
« **câblé, non servi** », et le dossier répétait que « câblé n'est pas mesuré ».
**Il est désormais MESURÉ SERVI au 4B.** Contre `Planes14` servi (B2, 87,0 tok/s
dans 2,56 Go) : **−2,3 % de débit pour −0,20 Go de carte** — l'arbitrage produit
« vitesse contre bits » a enfin ses deux points sur la même carte, dans le même
protocole.
⚠️ **Le prix hors ligne reste celui qu'on connaissait** : **1 340 s de
chargement**, le transcodage `Planes12x` refaisant une recherche réseau à cinq
niveaux par bloc. **Coût payé une fois**, mais payé.
🕳️ **Note de plomberie, déclarée** : l'échantillonneur `nvidia-smi` a été rendu
robuste au nom de champ du driver au lieu du `-l 1` verbatim du préreg ; **le
critère scientifique tamponné est inchangé** — pur détail d'invocation. Et le
premier essai de G3 a échoué sur un « Volume mount failed » (infra transitoire,
**0 octet lu**), relancé.

## 2026-08-24 — Le papier est soumis à ACM TACO, et le premier levier hors format est tiré

Deux choses le même jour, et elles se tiennent : **D1 était l'objection n°1 des
relecteurs externes** — *« le levier nommé, jamais tiré »* — et il fallait la
lever avant d'envoyer.

✅ **1. D1 — LA FUSION DES LANCEMENTS ENTRE SUR LE CHEMIN SERVI** (job
`6a8c6fbc`, l40sx1, 488 s, **0,24 $** contre ≈ 0,60 annoncés et un plafond de
1,35 ; pré-enregistrement tamponné puis commité **avant** le lancement ;
[`mesures/d1-fusion-servie-2026-08-24.txt`](mesures/d1-fusion-servie-2026-08-24.txt)).
`q/k/v` consomment la même activation, `gate/up` aussi : concaténées **par
lignes**, elles font un matvec au lieu de trois et de deux — **252 → 144
lancements par token**.

| bras | tok/s (médiane) | plage | Go carte | matvec/token |
|---|---|---|---|---|
| dense | 43,5 | [43,4–43,5] | 8,04 | — |
| **fusé F1** | **100,6** | [99,9–100,7] | 2,57 | **144** |
| fusé F0 | 94,9 | [94,1–95,2] | 2,56 | 252 |

**Fusion : ×1,061 [1,050–1,069]**, dans la bande **[1,00 ; 1,12]** posée
d'avance, pour **+3 686 400 octets EXACTEMENT** — la valeur prédite par
arithmétique avant le run (`gs_off`, un u32 par ligne fusée, 36 couches ×
25 600 lignes = 921 600 × 4 o), soit **+0,008117 b/poids**.
✅ **Les six critères sont verts, et deux d'entre eux portent la preuve** : F1 et
F0 rendent **les mêmes 128 tokens entre eux** (à poids fixés la fusion ne
réassocie rien, l'égalité devait être exacte — elle l'est) et **divergent du
dense au même token 89**, le tie-break historique du 08-06 reproduit ; **144
contre 252 `matvec_lancements/token` imprimés** — un « 128 tokens identiques »
vert pendant que les deux bras émettent 252 matvecs aurait prouvé les tokens et
**rien du lot**.
✅ **R5 fermé par une mesure plutôt que par un argument** : le préreg craignait
que l'ajout d'une unité de traduction NVRTC ralentisse **aussi** le bras de
référence et gonfle le gain. Les deux bras impriment la **même source NVRTC,
64 776 octets, même sha256, en 10 parties**. **Une seule unité, deux modes de
lancement** — le ×1,061 n'est pas un artefact d'allocation de registres
déplacée.
🚨 **L'écart à expliquer, et il est expliqué** : F0 rend 94,9 tok/s là où le
chemin servi **publié** rend 87,0 — **+9,1 %**, hors de la bande de ±3 % que le
préreg imposait de rapporter avant tout verdict. Le mécanisme est **vérifié, pas
supposé** : ce job impose `LLVQ_ROT_SHARE=1` (obligatoire pour le bras fusé,
`check_fuse` **refuse** la paire `FUSE=1 + ROT_SHARE=0` — l'accepter ferait
bouger deux mécanismes à la fois), tandis que les 87,0 publiés ont tourné à
`ROT_SHARE=0`. **Décomposition, les trois points mesurés sur cette carte** :

| configuration | tok/s | ce qui bouge |
|---|---|---|
| `ROT_SHARE=0, FUSE=0` | **87,0** | la configuration servie publiée |
| `ROT_SHARE=1, FUSE=0` | **94,9** | le hissage de la rotation seul (×1,091) |
| `ROT_SHARE=1, FUSE=1` | **100,6** | plus la fusion (**×1,061**) |

⚠️ **Le ×1,091 du hissage est une lecture INTER-JOBS** (87,0 vient de B2, sur
une autre unité de traduction) : **il se rapporte, il ne se publie pas** comme
une mesure de ce lot. Le **×1,061 de la fusion**, lui, est **intra-job**, à unité
de traduction identique et `ROT_SHARE` constant — c'est le seul des deux qui
soit un résultat.
⚠️ **Ce que le lot NE mesure pas** : le 8B et le 14B n'ont **pas** été rejoués
sous fusion, donc la table à trois tailles du papier reste sur **une seule**
configuration (`ROT_SHARE=0, FUSE=0`) — propriété qu'elle utilise, et qu'un 4B
fusé isolé casserait. 🚨 Et le **11,7 % du banc** (5,096 → 4,504 ms) **ne se
transporte pas** : il est mesuré en f32, hors modèle, sur le temps **matvec
seul**. *11,7 % du temps matvec et 6,1 % du temps par token sont deux quantités
différentes, cohérentes entre elles.*
🔎 **Et une chose est prouvée pour la première fois** : aucune ligne de
`fused_cuda.rs`, de `SegPlan` ni du corps du noyau n'est vérifiable hors carte.
Deux dispositifs les prouvent ici, dans cet ordre — **le build de l'image**
(première compilation réelle de ces fichiers) puis **le gate C1/C2**. Un
`gs_off` faux aurait multiplié **certaines** lignes par le mauvais centroïde en
laissant les autres intactes ; un pad resté au milieu du flux épissé aurait
décalé tout un segment de 4 octets et rendu des nombres **finis, plausibles et
faux sans jamais planter**. Les deux auraient déplacé la divergence **bien avant
le token 89**.

✅ **2. LE PAPIER EST SOUMIS.** *ACM Transactions on Architecture and Code
Optimization*, **`TACO-2026-428`**, type « Original Work / New Paper », source au
commit **`e21a8bb`** sur `main`. Deux fichiers : `paper.pdf` (21 pages, **19,55
comptées sur 20**) et `titlepage.pdf`, **seul porteur d'identité**. Proof vérifié
avant envoi : **zéro fuite d'identité**, mode review actif.
✅ **Le bras QTIP est au corps** (`tab:layouts`, `sec:qtip`) — donc le résultat
qui est contre nous est **publié dans la table principale**, avec son mécanisme,
et non renvoyé en annexe. C'est ce que F2 avait été lancé pour permettre.
✅ **Divulgation de l'usage d'IA générative**, obligatoire et dont l'omission est
un **motif de desk-reject** : la politique ACM sur la paternité l'autorise mais
exige qu'elle soit divulguée **dans le travail**, outil nommé et tâches
décrites. La formulation est **spécifique** parce qu'une ligne du type
« assistance à la relecture » serait fausse. ⚠️ La règle plus dure de FAccT, qui
**interdit** le texte généré, est propre à cette conférence et **ne s'applique
pas à TACO**.

⚠️ **3. DEUX DETTES OUVERTES À LA SOUMISSION, et une a été tranchée depuis.**

- 🕳️ **Le dépôt GitHub.** La note de soumission d'origine disait à l'éditeur que
  l'accès serait donné « sur demande », ce qui **suppose le privé**, et la page
  de titre écrivait que le dépôt « est mis en privé pour la durée de la revue ».
  **DÉCISION OPÉRATEUR DU 2026-08-25 : le dépôt RESTE PUBLIC.** La page de titre
  est réécrite pour l'expliquer — *le manuscrit est intégralement anonymisé et
  ne porte aucun lien qui identifie l'auteur ; les URL sont données ici, pour
  l'éditeur, afin que l'artefact soit localisable sans que le manuscrit brise sa
  propre anonymité.* **Toute phrase du dossier qui dit ou suppose « dépôt privé
  pendant la revue » est périmée.**
- ⚠️ **Le nom d'auteur est faux dans ScholarOne** — `MALANDRINO, MALANDRINO`, le
  prénom n'ayant jamais été saisi, plus une seconde affiliation parasite.
  **Invisible des relecteurs** (double aveugle), mais **c'est ce qui serait
  publié**. `main.tex` et `titlepage.tex` portent le bon nom. Se corrige par un
  mail au bureau éditorial ou au tour de révision.

⚠️ **4. LE RAF PENDANT L'ATTENTE, AUCUN ENGAGÉ — décision opérateur.** **D3**,
requantification calibrée sur DCLM-edu (~15 $), gardée **exprès** comme
l'expérience du tour de *Major revision* : c'est **le seul suspect non testé du
déficit MMLU**. Puis batch > 1, prefill GEMM, point 70B, profil Nsight — ce
dernier bloqué par la plateforme (F3). arXiv après le premier tour, une fois
l'endossement réglé. 🚨 **Ne pas re-décider ce qui est soumis : toute retouche du
papier attend la lettre de décision.**

⚠️ **5. LA DETTE QUI TRAVERSE TOUT LE DOSSIER ET QUE PERSONNE N'AVAIT COMPTÉE :
aucun des 16 `.ots` n'a jamais été upgradé.** Balayage des 16 fichiers le
2026-08-25 : **4 `PendingAttestation` et 0 `BitcoinBlockHeaderAttestation`
chacun** — **64 pending, 0 Bitcoin**. Ce sont des **reçus de calendrier**, pas
encore des ancrages sur la chaîne : l'antériorité qu'ils prouvent tient **tant
que les calendriers répondent**, et le papier comme ce fil s'appuient sur eux
pour l'argument de rigueur. `ots upgrade` rendrait la preuve indépendante des
calendriers ; **il n'a jamais été lancé.** 🕳️ Le dossier a relevé, ligne à ligne,
quels préregs étaient tamponnés **avant** leur mesure et lesquels ne l'étaient
pas — sans jamais vérifier **ce que le tampon vaut**. *La méta-couche dérive
aussi.*

**Compteurs au 2026-08-25** (*mesurés*) : **69** journaux dans
[`mesures/`](mesures/), **13** CSV dans [`data/`](data/), **73 lignes** dans
[`data/jobs.csv`](data/jobs.csv) pour **87,36 $ facturés au total** — dont
**28,56 $ sur 27 jobs depuis le 2026-08-18**, la quinzaine la plus dense du
projet —, **22** pré-enregistrements et **16** ancrages `.ots`.

## 2026-08-25 → 08-27 — Le tour de qualité à 0 $, l'instrument des tampons pris en défaut, le desk reject de TACO et le dépôt Zenodo

Trois journées qui se referment sur un renversement d'instrument, un
renvoi éditorial et un DOI. Dans l'ordre.

✅ **1. LE 08-25 SUR LE MAC (0 $) : LE PARTAGE DES 48 BITS EST GATÉ, PUIS SON
ÉCHELLE EST RÉFUTÉE PAR SA PROPRE RÉPLICATION.** Étage 1 du gate (86 min de
Mac, [`mesures/gain-ab-gate-0.6b-2026-08-25.txt`](mesures/gain-ab-gate-0.6b-2026-08-25.txt),
préreg tamponné avant le premier bras) : à **iso-débit vérifié 2,1656 b/poids**
sur Qwen3-0.6B/28 blocs, `leech0c13` (0 bit de gain) rend **39,3309 (×2,017)**
et `leech2c11` (2 bits) **39,5350 (×2,027)** — tous deux **−9 %** sous le
témoin `leech1c12`, **la configuration servie, pire des trois** (43,4865,
×2,230). Puis le quatrième barreau `leech4c10` rend **47,1537 (×2,418)** — pire
des quatre, la prédiction signée fausse par le haut, et la « lecture en U » du
matin morte en six heures (commit `fb06beb`). ❌ **Et la réplication à
`LLVQ_CALIB_SEED=1` inverse le classement** : le premier au tirage 1 finit
dernier au tirage 2, un bras bouge de 13,9 % là où l'écart entre les quatre
n'était que de 10,6 % — **le bruit dépasse le signal, l'échelle des bits de
gain est réfutée** (commit `ebd40e6` ; la réserve R1 du préreg, écrite avant la
première mesure, avait raison).
🔎 **Ce qui survit et ne dépend d'aucun tirage** : le **biais radial** est
mesuré ([`mesures/cosdiag-biais-radial-0.6b-2026-08-25.txt`](mesures/cosdiag-biais-radial-0.6b-2026-08-25.txt)) —
le code de gain quantifie ‖w‖ quand l'optimum à direction fixée est
‖w‖·cos θ ; surcoût **+3,69 %** à la boule 12 servie, monotone en taille de
boule (+3,39 → +4,49 % des boules 13 à 10). Pure géométrie, piste gratuite.
🔎 Et une lecture du PDF amont (commit `8300145`, tables re-rendues en image) :
**notre cap 12 + 1 bit n'apparaît dans aucune table LLM du papier** — leurs
résultats LLM n'emploient que Λ₂₄(13)+0 et Λ₂₄(11)+2, et le 0 bit gagne les
trois métriques sur quatre modèles sur cinq. La configuration que nous servons
n'a de résultat LLM nulle part, pas même chez les auteurs.

✅ **2. LE BRUIT DE MMLU ENTRE GRAINES EST MESURÉ, ET IL FERME L'ÉCHELLE DE
VOLUME** (préregs tamponnés `548ad9b` et `b1a0be1` avant le premier job ;
jobs du 08-25 au soir, 0,58 $, analyse des 25-27/08, entrée au dépôt par
`e8f9e07` ; [`mesures/bruit-mmlu-graines-4b-2026-08-25.txt`](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)).
Les trois artefacts 4B complets de F5 (graines 1/2/3, survivants du bucket :
0,58 $ au lieu de ~21) rendent **58,02 / 52,19 / 55,17 %** de MMLU micro —
**étendue 5,83 pp, s (n−1) = 2,9154 pp**, deux paires appariées sur trois
résolues (s1−s2 : +5,83 [+2,70 ; +9,11], McNemar p = 2,4e-06). Les deux
préregs posaient la même branche : **s > 2,0 pp → on ne lance pas**.
❌ **L'échelle de volume de calibration NE PART PAS** — ni ×8, ni ×32, ni ×96 ;
~19 $ économisés, et la prédiction signée (0,5-1,5 pp) est **réfutée**.
🔎 Détail qui vaut d'être vu : les **rangs s'inversent entre métriques** — la
graine 1, pire en perplexité (16,7425), est la **meilleure** en MMLU (58,02).
⚠️ Le premier job est mort en 15 s sur `exit 127` : la commande était
`['bash', script]` au lieu de `['bash', '-lc', script]` — bash a lu le script
comme un **nom de fichier**. Relancé 37 min plus tard, vert.
🕳️ **Et la passation du 08-27 a lu cet échec comme « un job EN VOL, jamais
récolté », en budgétant 0,49-0,55 $ de relance** ([`plan-cloture-2026-08-27.md`](plan-cloture-2026-08-27.md)
§L1) — le successeur avait réussi le soir même, deux lignes plus haut dans
`hf jobs ps -a`. **Cinquième prise de la règle du §7 de CLAUDE.md** (épuiser
les canaux de rétention avant de chiffrer un rejeu), et cette fois le canal
n'était ni un bucket ni des logs : la liste des jobs elle-même.

🕳️ **3. LE 08-26 : « AUCUN DES 16 `.OTS` N'A JAMAIS ÉTÉ UPGRADÉ … 0
`BitcoinBlockHeaderAttestation` » — la ligne que ce fichier porte encore en §5
de l'entrée du 08-24 — EST MESURÉ FAUX, ET LE DÉFAUT EST DANS L'INSTRUMENT**
([`mesures/ots-etat-2026-08-26.txt`](mesures/ots-etat-2026-08-26.txt),
reproductible par `ops/otsaudit.py`, commit `9105ef8`). Le format `.ots`
stocke le type d'une attestation dans une **étiquette binaire de 8 octets**,
jamais en texte : `grep BitcoinBlockHeaderAttestation` rend **0 sur un fichier
qui en porte quatre** — et `grep PendingAttestation` rend **0 aussi**, donc le
« 4 pending » publié à côté n'avait **jamais été mesuré**, il avait été inféré
des quatre calendriers puis présenté comme mesuré. *Un instrument qui rend la
même valeur dans les deux cas ne mesure rien* — la leçon du §5 de CLAUDE.md,
sur un format au lieu d'un test. État réel : **20 tampons** (pas 16), dont
**16 portent 3 ou 4 ancres Bitcoin** et 4 (ceux du 08-25) attendent encore.
⚠️ Le journal établit ce que **les fichiers portent**, pas que la chaîne le
confirme (403 du proxy ce jour-là) ; les racines de Merkle engagées sont
imprimées par bloc pour qu'un tiers fasse le dernier pas.
🚨 **Et une dette réelle apparaît là où celle-ci se referme** : les préregs du
**08-10 et du 08-11 n'attestent plus de leurs propres octets** — la passe
d'anonymisation TACO (`01fdbe6`, 2026-08-19) les a réécrits, et **la version
attestée n'est récupérable sous aucune révision** (les 128 blobs `.md` de
toute l'histoire git ont été hachés, aucun ne rend les condensats engagés).
Un tampon dont on a réécrit le document ne se répare pas ; il se déclare.
✅ Le 08-26 porte aussi le **protocole de variance de calibration** et son plan
de travaux (`b3362ab` → `1bb7f93`) : le bras scalaire recoupé et réécrit
(faux sur trois points), et le gate A jugé **inobservable** avant le premier
run (`ec14526`).

✅ **4. LE 08-27 AU MATIN : LE PROGRAMME DE CLÔTURE** (commit `190d0b7`,
[`plan-cloture-2026-08-27.md`](plan-cloture-2026-08-27.md)) — **neuf lots,
9 à 13 $ hors 4B, ~30 h de Mac, onze pistes écartées avec leur raison**, sous
deux règles : aucun lot ne se justifie par « ça pourrait renverser le
verdict », et chaque lot porte un gate tamponné avant la mesure. Le pari
produit reste clos par arithmétique (plafond `Planes14` 3,331× < AWQ mesuré
3,38×) ; ce que le programme renforce, c'est la **force de l'énoncé de
fermeture** du manuscrit. ✅ Et l'`ots upgrade` que L1 réclamait est **passé le
jour même** (réseau ouvert dans cette session) :
**20 tampons sur 20 portent désormais au moins une ancre Bitcoin**
([`mesures/ots-etat-2026-08-27.txt`](mesures/ots-etat-2026-08-27.txt) —
pour le préreg bruit-MMLU : blocs 964047, 964057, 964090, 964094).

❌ **5. LE 08-27 : DESK REJECT DE TACO.** Le manuscrit `TACO-2026-428` est
**renvoyé sans revue par l'Editor-in-Chief, sur le périmètre** — « returned …
on 2026-08-27 without review, on scope grounds » (`paper/main.tex:4-7` ;
commit `209a695`). **Aucun chiffre n'est contesté, aucune revue technique n'a
eu lieu.** La venue suivante n'est pas choisie.

✅ **6. ET LE PAPIER EST DÉPOSÉ LE SOIR MÊME** (commit `209a695`, 22:14) :
**Zenodo, DOI de concept `10.5281/zenodo.22133606`, version v1
`10.5281/zenodo.22133607`** — préprint d'auteur, non relu par les pairs, et le
PDF le dit. 🔎 **Le piège de la désanonymisation, et il vaut d'être connu** :
flipper `anonymous`/`review` fait imprimer par `acmart` un bloc « ACM
Reference Format », l'ISSN 1544-3973, un DOI bidon `10.1145/…` et un pied de
page « ACM Trans. Arch. Code Optim., Vol. 0 » sur les 21 pages — une fausse
revendication de publication, **sur un DOI permanent et non supprimable,
depuis la revue qui vient de renvoyer le papier**. `nonacm=true` retire le
visible mais **pas le XMP** (`dc:publisher = ACM`), d'où le `\hypersetup`
explicite. Vérifié **sur le PDF, pas sur la source** : toutes les occurrences
rendent 0, et le corps est identique au soumis (comparaison des
multi-ensembles de mots, 0 mot ajouté).

## 2026-08-28 → 08-30 — Le plan d'après-dépôt, puis la campagne « piles isolées » : le premier concurrent 2 bits mesuré chez nous, et un bras concurrent qui ne rend rien de publiable

✅ **1. LE PLAN D'APRÈS-DÉPÔT** (2026-08-29, commit `3bd1d05`,
[`plan-apres-depot-2026-08-29.md`](plan-apres-depot-2026-08-29.md)) : quatre
phases chiffrées sur coûts mesurés — **geler la config servie** (~0,25 $),
**la géométrie de lancement** (~2-4 $, orientée par F1 : le par-noyau est à
2 % de cuBLAS, le gisement est *entre* les noyaux), **la qualité**
(~12-25 $ : MMLU de la graine 3, EoRA à budget de bits posé d'avance,
corpus), puis **les familles** (~17 $ dense, MoE ~65 $ sur go explicite).
❌ Le mini-papier « calibration de la hessienne » est **enterré** : l'oracle du
lot B borne la calibration à −1,6 % et F5 montre une variance de graine qui
écrase damping et volume — raffiner un terme sous le bruit d'un autre. Le
même jour, des brouillons de diffusion sont écrits sans être publiés (billet
HF `3b8a144`/`21f0366`, post r/LocalLLaMA `7c550f2`/`a220052` — la
publication est une décision d'opérateur listée au plan, en attente).
🕳️ Le soir du 08-29, la « Phase P » fait passer **le port vLLM devant la
géométrie** (`783aeb0`) — un ordre **renversé le 08-31 par décision
d'opérateur** (`deaa449`, cf. l'entrée suivante).

✅ **2. LA CAMPAGNE M3/M4 (30-31/08) : 1,29 $ au registre** (*mesuré*,
[`data/jobs.csv`](data/jobs.csv) — 11 lignes `m3-*`/`m4-*` ; ⚠️ le commit
`1aad002` et le protocole v2 en comptent **12**, écart d'un job non élucidé
ici). Préreg tamponné avant le premier job (`250c726`).

**2a. Le gate de l'instrument a rougi une fois — sur NOUS, pas sur lui.**
Premier job : rouge parce que notre script agrégait en **macro** (72,85) au
lieu du micro — le défaut du §3ter, refait 25 jours après sa première
correction ; le dump rapatrié rend **70,36 contre 70,32** et 2 272/2 280
mêmes picks ([`mesures/m3-gate-mmlu-vllm-2026-08-30.txt`](mesures/m3-gate-mmlu-vllm-2026-08-30.txt)).
Seconde passe **verte sur les deux bras**
([`mesures/m3-gate2-mmlu-vllm-2026-08-30.txt`](mesures/m3-gate2-mmlu-vllm-2026-08-30.txt)) :
**f16 dans vLLM 70,34 % contre 70,32 connu** (0,02 pp), AWQ/Marlin **69,82
contre 70,04** (0,22 pp), picks 99,61/99,56 %. Et le bras GPTQ ajoute un
**troisième moteur** : f16 dans transformers **70,84 %**
([`mesures/m3-gptq2-mmlu-2026-08-30.txt`](mesures/m3-gptq2-mmlu-2026-08-30.txt)),
puis llama.cpp/Metal un quatrième (70,36) — **le même modèle f16 scoré par
QUATRE implémentations indépendantes, étendue 0,52 pp.**

**2b. Le bras GPTQ 2 bits : la quantification réussit, et rien de publiable
n'en sort** ([`mesures/m3-gptq2-production-2026-08-30.txt`](mesures/m3-gptq2-production-2026-08-30.txt)).
Premier bras tiers du dossier **calibré sur NOTRE corpus** (empreinte
`40300263e5d0afa2`, 131 072 tokens C4 — le confondant de calibration tombe) ;
artefact de 1 754 463 312 octets = **3,489 b/param** au bon dénominateur — 🚨
le job imprimait 3,182 sur le dénominateur de gptqmodel (4 411 424 256, faux
de +9,67 % = la part de l'embedding, têtes liées comptées deux fois) : la
règle n°1 du §7 enfreinte dans le fichier qui la cite, corrigée. ❌ **Mais
vLLM 0.26.0 REFUSE `bits=2`** — le bras ne se sert pas dans le moteur prévu ;
scoré via transformers : **MMLU 24,74 % = le hasard à quatre choix**.
🚨 **CE CHIFFRE N'EST PAS PUBLIABLE, et le préreg l'avait annoncé** (§1.3 :
« nous attendons de gagner ce bras » — un résultat favorable attendu exige
*plus* de vérification) : la sortie est **dégénérée, pas aléatoire** (pick A
68,9 %, écart médian des logits divisé par 3,4) — **indistinguable entre
effondrement réel et chargement cassé**. La génération libre rend un
**charabia intégral** (le 24,74 ne mesure aucune capacité), et le wrapper et
`.model` rendent **le même charabia au caractère près** — suspect du
déballage réfuté, mais les deux chemins ne sont pas indépendants (le wrapper
délègue). **La colonne qualité du bras GPTQ reste vide.**

**2c. IQ2_XXS : le premier concurrent 2 bits dont ce dossier mesure la
qualité — et le verdict est double** (Metal, 0 $,
[`mesures/m3-iq2-metal-2026-08-30.txt`](mesures/m3-iq2-metal-2026-08-30.txt) ;
même corpus de calibration que le bras GPTQ, empreinte machine-vérifiée).

| | b/poids ou bpw | ppl (× son témoin) | MMLU micro | chute |
|---|---|---|---|---|
| LLVQ 2 bits | 2,0702 | ×1,3845 | 55,59 % | −14,73 pp |
| IQ2_XXS | 2,0625 | ×2,6287 | 39,39 % | −30,97 pp |

Paire appariée sur les 2 280 mêmes questions (qhash identiques) : **+16,20 pp,
IC95 [+12,64 ; +19,72]**, SE 1,81 — l'intervalle entier au-dessus du seuil de
lecture posé d'avance (~6 pp, soit 2× le σ inter-graines de 2,92). **À
0,37 % d'écart de bits, un codebook de Leech à 1,1·10¹⁴ points retient
16,20 pp de MMLU de plus qu'un codebook qui tient en LUT** — et les deux
métriques concordent, celle du raisonnement comprise. ⚠️ **Et le prix est
toujours là, c'est la même thèse retournée** : IQ2_XXS sert **2,479 b/param
contre nos 5,162** — pas besoin d'être déplié. Le contrefactuel LUT du
BACKLOG §4.4 est mesuré, et il tranche **dans les deux sens**.

**2d. M4, le bras pont CUDA : l'interdit inter-cartes reproduit sur un bras
TIERS** ([`mesures/m4-iq2-cuda-2026-08-30.txt`](mesures/m4-iq2-cuda-2026-08-30.txt)).
Le **même GGUF, sha256 prouvé identique des deux côtés** (`19a8ed49…`), rend
**×2,647 sur Metal et ×3,688 sur CUDA** contre le f16 de sa propre pile —
+39 % à fichier, moteur et modèle identiques. « Les × inter-cartes ne se
divisent pas » cesse d'être une prudence maison : c'est une propriété du
matériel, mesurée hors de nos bras. 🔎 Et un recoupement non cherché : vLLM
(83,09) et llama.cpp (84,83) s'accordent à 2,1 % sur le f16 L40S — **1,95×
notre bras dense** : le handicap `broadcast_matmul` est corroboré du dehors,
et il joue contre nos × bruts, pas pour. ✅ Le MMLU d'IQ2_XXS sur CUDA
(08-31) : **38,87 contre 39,39 Metal — agrégat invariant à 0,52 pp**, mais
**même pick à 95,79 % seulement** (contre 99,6 % pour le f16 entre moteurs) —
les noyaux de déquantisation 2 bits divergent réellement, et 96 désaccords
s'annulent presque (commit `4834ab3`).

✅ **3. LE PROTOCOLE « PILES ISOLÉES » v2 EST AUDITÉ PUIS TAMPONNÉ** (commits
`6b053f3` et `e8f9e07`, 08-31 au matin ;
[`../proofs/protocole-piles-isolees-v2-2026-08-31.md`](../proofs/protocole-piles-isolees-v2-2026-08-31.md),
sha256 `987a07f4…`). Quatorze éléments du v1 confrontés à son exécution
réelle ; la v2 **ancre les constantes** que quatre instruments ont convergé à
établir — dénominateur **4 022 468 096** (têtes liées), étalon f16 à quatre
moteurs **[70,3 ; 70,9]**, les trois empreintes, les sha256 des GGUF — plus le
**seuil de lecture σ = 2,92 pp**, et sept gates ordonnés. Elle est construite
pour la **ré-exécution après amélioration du noyau** : un changement de noyau
ne bouge que notre débit (~0,3-0,6 $ le rejeu, contre 1,29 $ la table
entière), et elle ne contient **aucune section à remplir** — les écarts vont
dans un fichier nommé d'avance, jamais dedans.

## 2026-08-31 — Vague 2 : la fusion transfère aux trois tailles, et la config servie v1 est GELÉE

Pré-enregistrement [`../proofs/preregistration-vague2-gel-geometrie-2026-08-31.md`](../proofs/preregistration-vague2-gel-geometrie-2026-08-31.md)
(sha256 `e23e9895…`, sous le protocole v2, gates G1-G7 tels quels, nominal
~3,0 $, plafond 5 $), **tamponné et commité 77 s avant la création du premier
job** (*calculé* : commit `f37cfa8` à 09:14:15, job `6a952a14` créé à
09:15:32 — horodatage ObjectId). Le biais y est nommé d'avance : le gel
FUSE=1 est le résultat *attendu*, donc c'est un 8B ou 14B raté qui aurait
exigé la lecture soigneuse.

✅ **1. LA FUSION TRANSFÈRE AUX DEUX TAILLES RESTANTES, forme D1 exacte**
(deux jobs, 0,45 + 0,86 $ ;
[`mesures/vague2-fusion-8b-14b-2026-08-31.txt`](mesures/vague2-fusion-8b-14b-2026-08-31.txt)) :

| taille | gain de fusion (intra-job) | bande [1,00 ; 1,12] | lancements/token |
|---|---|---|---|
| 4B (D1, 08-24) | ×1,061 [1,050–1,069] | ✅ | 252 → 144 |
| **8B** | **×1,055 [1,054–1,058]** | ✅ | 252 → 144 |
| **14B** | **×1,028 [1,027–1,029]** | ✅ | **280 → 160** (40 couches) |

Les six critères de D1 tiennent partout : 128 tokens **identiques** F1/F0/
dense, même sha256 NVRTC des deux bras (`347f50a4…`), deltas d'octets
**exacts** (+4 423 680 au 8B, +6 717 440 au 14B). 🔎 **Fait transversal, non
exigé par le préreg** : le gain **décroît** avec la taille (×1,061 → ×1,055 →
×1,028) — cohérent avec une latence de lancement qui s'amortit sur des matvec
plus gros ; observation à trois points, pas une loi, et un **prior pour A1**.

✅ **2. LE GEL, PAR UNE RÈGLE ÉCRITE AVANT LES CHIFFRES** :

  **config servie v1 = `planes14` + `LLVQ_EMBED=q8` + `LLVQ_ROT_SHARE=1` + `LLVQ_FUSE=1`**

| taille | tok/s servi v1 [plage] | Go carte | (ancien servi B2) |
|---|---|---|---|
| 4B ⚠️ *cellule importée de D1 (08-24), pas re-mesurée dans cette vague* | **100,6** [99,9–100,7] | 2,57 | 87,0 · 2,56 |
| 8B | **75,5** [75,5–75,6] | 5,41 | 68,2 · 5,41 |
| 14B | **46,8** [46,7–46,8] | 9,40 | 43,3 · 9,39 |

⚠️ **Règle des deux formulations, comme toujours** : les ×2,85 (8B) et ×2,75
(14B) contre le dense sont des quotients de médianes de bras jamais
entrelacés, et le dense est *notre* bras handicapé — seul le gain de fusion
est un rapport intra-job. **La série à tête identique (×1,11 → ×1,29 →
×1,41, ROT_SHARE=0/FUSE=0) reste LA série qui mesure le noyau, et elle n'est
PAS re-mesurée sous v1** — un rejeu tête-identique serait une autre campagne.
🕳️ **Ce que le gel retire** : l'interdit « les tables publiées sont à
ROT_SHARE=0/FUSE=0, un 4B fusé isolé casserait la propriété » — la propriété
« une seule configuration partout » est désormais préservée **par le gel**,
plus par l'interdit. Aucune qualité ne bouge : mêmes artefacts scellés, pas
un octet de poids ne change.

✅ **3. DÉCISION D'OPÉRATEUR (`deaa449`) : A2 ET A3 SE FONT QUOI QU'IL
ARRIVE.** Le plan du 29 mettait Graphs et occupation derrière le port vLLM ;
l'opérateur le renverse — le papier revendique un noyau, candle est le seul
moteur auditable de bout en bout, et l'argument souverain exige que ce chemin
soit rapide **par lui-même**, pas par procuration. Ce que la décision ne
change pas : les critères d'adoption (un résultat nul ne s'adopte pas), et le
rôle d'A1 — ordonner A2 contre A3.

✅ **4. LES SURFACES CITENT UNE SEULE CONFIG SERVIE, ET ELLES LA NOMMENT**
(`5ff5c55` : README — bloc « served configuration v1, frozen 2026-08-31 » —,
model card locale, CLAUDE.md ; puis la passe du soir `46164a9`). Une
vérification systématique (25 agents, adversariale) a trouvé **sept surfaces
de plus** donnant encore l'ancienne config comme courante : deux corrigées le
soir (`format-noyau.md`, compteurs et note ots de CLAUDE.md), le reste
inventorié dans [`alignement-v1-restant-2026-08-31.md`](alignement-v1-restant-2026-08-31.md),
séparé en mécanique et en ⚖️ à arbitrer — **dont ce fichier-ci, muet depuis le
08-24, le plus gros morceau**. La table B2 n'est pas remplacée : elle est la
mesure ROT_SHARE=0/FUSE=0.

## 2026-08-31 (soir) — Les quatre morts d'A1, puis A1 et A4 verts : la vague 2 est complète pour 2,17 $

Le bras A1 (`nullk` à 252 contre 144 lancements) est mort **quatre fois avant
de rendre un chiffre** — trois morts d'infrastructure, une de lanceur — pour
0,02 $ au total, chacune consignée au fichier d'écarts que le préreg nommait
d'avance ([`../proofs/preregistration-vague2-gel-geometrie-2026-08-31-ECARTS.md`](../proofs/preregistration-vague2-gel-geometrie-2026-08-31-ECARTS.md)) :

1. **É1 — le binaire n'était pas dans l'image** (probe `which`, 15 s, 0,01 $) :
   `nullkbench` absent des **deux** listes explicites du Dockerfile
   (`cargo build --bin` et `COPY`) — compiler le crate ne suffit pas
   (`c6642e4`).
2. **É2 — E0599 à la PREMIÈRE compile Linux du bin** : le trait
   `PushKernelArg` qui porte `.arg()` n'était pas importé (`nullkbench.rs:139`).
   Le « compile » du registre était un constat **macOS, donc vide** — le corps
   du bin est `cfg(linux)`, le Mac ne compile qu'un stub. Une ligne
   (`970d27d`) ; le Space est resté en `BUILD_ERROR` ~9 h 40.
3. **É3 — la seconde mort aurait eu lieu SUR CARTE, et elle est tuée à 0 $
   AVANT relance** : seul bin du crate à ne pas prépendre `llvq_slot.cuh`,
   dont l'`#include` gardé par `#ifndef` aurait fait refuser la source à
   NVRTC (sans système de fichiers) — **après** le probe et **après** le début
   de la facturation. Reproduit par `clang -E -nostdinc` sur l'unité exacte,
   corrigé par l'assemblage de `planesbench` (`3815eda`).
4. **É4 — le LANCEUR** : la relance par CLI a fait parser `-lc` comme
   `--label c` par click, et bash a reçu le script en **nom de fichier**
   (exit 127, ~0,01 $). Relance par l'**API** avec un assert d'identité du
   tableau de commande contre `hf jobs inspect` du job d'origine.

🔎 **L'instrument qui manquait existe désormais** :
`CUDARC_CUDA_VERSION=12040 cargo check --target x86_64-unknown-linux-gnu -p
llvq-cuda` type-checke le crate CUDA **depuis le Mac** — il aurait vu É2 avant
tout build. É3, lui, ne se voit qu'en **exécutant le texte du noyau** (leçon
§5 de CLAUDE.md) ; le `clang -E` en est la forme à 0 $.

✅ **A1 EST VERT — 15 s de GPU, 0,01 $**
([`mesures/a1-nullk-252-144-2026-08-31.txt`](mesures/a1-nullk-252-144-2026-08-31.txt)) :
`nullk` 252 lancements **2,200 ms [2,198–2,201]**, 144 lancements **1,794
[1,793–1,796]**, **r = 0,8158 [0,8150–0,8162]** — rapport formé round par
round, 7 rounds dont 2 jetés, un seul processus. Lecture pré-enregistrée
(préreg `e23e9895` §A1, avant le chiffre) : r ≤ 0,65 → latence, A2 ; r ≥ 0,90
→ occupation, A3 ; **0,8158 → BANDE MIXTE** — ni A2 ni A3 éliminé, les parts
se publient : **Δ = 0,406 ms pour 108 lancements = 3,76 µs/lancement**
(*calculé*, cohérent avec les 3,63 µs du lot A du 08-06). Le prior déclaré de
0,83 est confirmé à 1,7 %. ⚠️ Le 2,200 ms de ce processus ne se soustrait à
aucun temps d'un autre processus — pas au 2,306 de F2 non plus.

✅ **A4 EST VERT, ET LA RÉSERVE ADA DEVIENT UN MÉCANISME** (deux jobs
a100-large, **0,83 $** sous le devis ;
[`mesures/a4-a100-2026-08-31.txt`](mesures/a4-a100-2026-08-31.txt)). Dans les
termes posés d'avance : **AUCUN bras réseau n'atteint 1,00× FP16 sur A100** —
`Planes14` **0,79×** [0,79–0,79], `Slot32` 0,73× — et la table **reproduit F4
au centième** (cuBLAS 1,14×, `nullk` 1,69×). La branche « sinon » du préreg
s'applique : la réserve Ada du papier **ne saute pas, elle devient un
mécanisme mesuré à deux points, étendu à la géométrie fusée** — une issue de
publication, pas un échec. Les faits neufs :

- **La fusion TRANSFÈRE à la seconde architecture** : ×1,063 [1,062–1,067] de
  F1 sur F0, contre ×1,061 [1,050–1,069] sur L40S — mêmes **+3 686 400 octets
  exactement**, 128 tokens identiques F1/F0/dense, bande tenue sur une carte
  que le préreg n'avait pas promise. F1 rend **63,4 tok/s [63,4–63,5] dans
  2,57 Go** contre **51,4 [51,4–51,5]** à son propre dense (×1,23 intra-job).
  🚨 **Aucun × A100 ne se divise par un × L40S**, et les 63,4 ne se comparent
  pas aux 100,6 autrement que par le mécanisme d'horloges du lot G ; le dense
  au dénominateur reste *notre* bras handicapé — deux formulations, toujours.
- **Le rapport de géométrie est INVARIANT entre architectures** : `nullkbench`
  en prime (É5, déclaré au fichier d'écarts) rend **r = 0,8198
  [0,8196–0,8202]** sur A100 contre 0,8158 sur L40S le même soir — **0,5 %
  d'écart quand les temps absolus s'étirent de ×1,809** (*calculé*), à 1,2 %
  du rapport d'horloges 1,787 du lot G. Le poste par-lancement suit l'horloge
  comme le reste : **propriété de la géométrie, pas de la carte.**
- **`fusedrun` a tourné sur A100 pour la première fois**, sur une image sm80
  née le soir même : `publish --compute-cap 80` réécrit la seule ligne `ENV`
  de la recette canonique au téléversement, sans recette jumelle à tenir
  synchrone (`7e67f28`) — plus `LLVQ_NVRTC_ARCH=compute_80` pour nos noyaux
  NVRTC, les deux pièges documentés **avant** le job.

✅ **La vague 2 est COMPLÈTE : 0.1 ✓ gel ✓ A1 ✓ A4 ✓ — 2,17 $ sur un plafond
de 5** (*mesuré*, somme des 7 lignes du 08-31 dans
[`data/jobs.csv`](data/jobs.csv) ; commit `1fc429e`).

## 2026-09-01 — Le préreg A2/A3 est arbitré et tamponné, et le plan pointe la phase A

✅ **L'ARBITRAGE D'OPÉRATEUR : A2 D'ABORD** (commit `833d630` ;
[`../proofs/preregistration-a2-a3-geometrie-2026-08-31.md`](../proofs/preregistration-a2-a3-geometrie-2026-08-31.md),
sha256 **`802006c5…`**, tamponné sur quatre calendriers **avant la première
milliseconde de tout job A2/A3**). A1 ayant rendu la bande mixte, l'ordre
revenait à l'opérateur : A2 (CUDA Graphs) passe devant — son pool
par-lancement est **mesuré directement** (0,406 ms/108 = 3,76 µs ; extrapolé
linéairement aux 252 lancements : **0,947 ms ≈ 43 %** du plancher,
*calculé*, hypothèse de linéarité **déclarée**) et son critère tranche vite.
🚨 **Les priors, TOUS défavorables à A2, sont déclarés dans le préreg** — la
ligne « CUDA Graphs » avait été fermée par mesure au lot A (0,167 ms = 0,8 %
d'un token, **en plafond**) et ce document la rouvre **sur décision
(`deaa449`), pas sur fait neuf** ; F3 (soumission hôte recouverte) et F1
(témoin ≤ 1,05× de cuBLAS) bornent l'attente.

Les règles gelées, telles qu'ancrées :

- **A2** : blocant nommé d'abord — **préallocation du cache KV à formes
  fixes** (un graph statique ne capture pas un `Tensor::cat` qui grandit ;
  2-4 j de dev). Critères : gain bout-en-bout **≥ 8 % → adopté**, **< 3 % →
  clos**, entre les deux : point de courbe. 🚨 Règle `check_fuse` transposée :
  les deux bras de l'A/B portent la **même** préallocation — le graph est la
  seule variable ; la prealloc se mesure dans **son propre A/B avant**.
- **A3** (occupation) : gate d'entrée au port — un bras doit battre
  **`planes14` en géométrie FUSÉE (144 lancements, la géométrie servie v1)**
  de **≥ 10 %** au banc (« battre les 252 ne compte pas : ce serait re-mesurer
  la fusion ») ; puis mêmes seuils bout-en-bout qu'A2 (**≥ 8 % / < 3 %**).
- **Le kill de phase, ancré — il ne vivait qu'en prose depuis le plan du
  08-29** : *si A1 + A2 + A3 rendent < 8 % cumulés bout-en-bout, l'axe
  géométrie SOUS CANDLE est clos par mesure* — le cumul se mesure sur le
  chemin servi v1, intra-job, jamais en additionnant des pourcentages de jobs
  différents. Plafond de phase : 4 $, distinct du plafond 5 $ de la vague 2.

✅ **Et le plan pointe la phase A** (`6fc74e4`) ; `fiche-4b.md`,
`cheatsheet-defense.md` et la model card locale reçoivent leurs médianes B2
et la ligne v1 en annotations datées (`2ad11b7`, `cd82f19`) — deux d'entre
elles portaient encore les points uniques d'avant B2 (48,7 / 88,4-88,5 /
×1,12), **deux générations en arrière**.

## 2026-09-01 → 09-02 — A2 rendu, A3 rendu : les graphs adoptent au 4B et au 8B, aucune variante d'occupation ne passe le gate, et la borne de la géométrie est mesurée

> 🕳️ L'entrée A2 est écrite ici **après** A3, qui l'a précédée d'un jour :
> c'était la dette documentaire déclarée à la passation du 09-01 au soir.
> Les chiffres viennent des journaux `a2-*-2026-09-01.txt` et du registre.

**A2 — CUDA Graphs sur la boucle token (2026-09-01, sept jobs, 0,87 $).**
Sous le préreg de phase `802006c5…` et le préreg d'étapes `af6c12d2…`, une
étape par tampon :
- ❌ **Étape 1, préallocation KV : STOP** — `r = prealloc/cat = 0,8919
  [0,8884–0,8953] < 0,97`, la prealloc **régresse de ~11 %** à fenêtre 256,
  prior `r = 1,00` réfuté ; retour opérateur, mécanisme instruit (vues
  narrow striées).
- ✅ **Étape 1b, store étendu à fenêtre fixe : VERT** — `r = 0,9917
  [0,9883–0,9935] ≥ 0,97`, coût net de la base fixe **−0,83 %, résolu**.
- ⚠️ **Étape 2 : capture certifiée, gate 2 rouge** — le pas entier
  s'enregistre (noyaux candle, cuBLAS et les nôtres), mais éager et graph
  divergent au token 2 ; aucun chrono rendu, comme le préreg l'exigeait.
  Diagnostic instrumenté : **le premier replay qui suit un préfill est
  inexact (max|Δlogits| = 1,12e1), les douze suivants sont exacts au bit
  près** — forme **hybride** (premier token de décodage éager, replay
  ensuite).
- ✅ **Étape 3, 4B : ADOPTÉ** — `r = graph/éager = 1,1345 [1,1336–1,1358]`,
  **+13,45 %** bout-en-bout, +12,6 % net contre v1 ; 128 tokens identiques
  sur 5 rounds ; éager 99,2 [99,1–99,2], graph **112,5 [112,4–112,6] tok/s**.
- **Transfert** (préreg `a2-transfert-8b-14b`, seuils par taille, règle de
  gel « les trois ou aucune ») : ✅ **8B adopté, +10,1 % [+10,08 ; +10,14]**
  (76,2 → 83,9 tok/s ; le replay **pur** y passe) ; ⚠️ **14B point de
  courbe, +6,1 % [+6,02 ; +6,11]** (46,9 → 49,8 ; replay pur rouge au token
  7, hybride vert) → **pas de gel v2**, la config servie reste v1 partout.

**A3 — variantes d'occupation au banc (2026-09-01 → 09-02, deux jobs,
0,24 $).** Huit bras dans la section Fusion de `planesbench`
(`kernels/planes_occ.cu`, sélecteur `LLVQ_SEG_ARMS`), chacun ne bougeant
qu'un mécanisme, justesse établie avant tout chrono — **six bras identiques
au bit près à `tv_planes_seg` sur 1 105 920 lignes**, les deux split-K à
≤ 2,0e-8 sur les sites scindés. Gate gelé aux ÉCARTS avant le job :
`gain = (t_ref − t)/t_ref` round par round, `t_ref` = Planes14 fusé
re-mesuré (4,565 ms), plage entière ≥ 10 % = passe
([`mesures/a3-occupation-banc-2026-09-01.txt`](mesures/a3-occupation-banc-2026-09-01.txt)).
- ❌ **Aucun bras portable ne passe** : pad −0,14 % [−0,52 ; −0,08] ·
  mr2 −10,45 · mr4 −24,62 · mr2p −9,85 · **pers +1,56 % [+1,01 ; +1,86]**
  (le seul positif, résolu, six fois sous le seuil) · sk1 −1,87 [−2,50 ;
  −1,77] · sk2 −9,87. **Point de courbe, pas de port.**
- ❌ **Le sous-remplissage de o/down n'est pas le résidu** : sk1 porte
  leurs grilles de 320 à 640/1 280 CTAs sans changer un octet et rend
  −1,87 %. L'hypothèse centrale de la note de design est réfutée dans son
  signe ; multi-lignes par warp perd davantage avec R.
- ✅ **La borne de la géométrie est mesurée** : `persall` (les 144 sites
  d'un round en UN lancement, bras de banc non portable par construction)
  rend **+26,36 % [+25,31 ; +26,61]**, 1,203 ms sur 4,565 — ≈ 0,54 ms de
  par-lancement (A1) plus ≈ 0,66 ms de bulles inter-noyaux (*calculé*).
  Transporté 1:1 au token servi ce serait ≈ +12 % (*calculé*, indicatif) ;
  **A2 a mesuré +13,45 %** — les graphs ont déjà encaissé, à l'ordre de
  grandeur près, ce que la géométrie pouvait rendre. Ce qui reste à A3
  après A2 est ce que `pers` mesure.
- 🕳️ **Une prise, instruite** : le premier job (`6a97394c…`, 0,12 $) est
  mort à la justesse de sk1 — huit warps écrivaient leurs partiels, chacun
  fençait le sien, le thread 0 tirait le ticket sans attendre les sept
  autres (le sample CUDA a UN écrivain par bloc). Un `__syncthreads()`
  entre fences et ticket ; et un bras faux **invalide** désormais sans tuer
  le job.
- ✅ **Kill de phase non déclenché** (A2 seul rend > 8 %) ; **la phase A se
  ferme avec A2 pour seule adoption AU SENS DU CRITÈRE DE DÉBIT** — 1,11 $
  dépensés sur un plafond de 4. ⚠️ « Adopté » y veut dire « passe le seuil
  préenregistré », pas « servi » : voir la décision juste dessous.

🚨 **DÉCISION D'OPÉRATEUR DU 2026-09-02 : A2 N'EST PAS SERVI, ET CE N'EST PAS
LE DÉBIT QUI TRANCHE, C'EST LA MÉMOIRE.** Un graph statique ne capture pas un
cache qui grandit : il exige une fenêtre KV **préallouée et payée en entier
quelle que soit la longueur du prompt**, là où le chemin servi alloue au fil
de l'eau. À la cible produit de 8k, cette fenêtre coûte **+1,21 Go au 4B sur
2,57, soit +47 % de VRAM pour +12,6 % de débit** (*calculé*, géométrie GQA du
dossier ; +22 % au 8B, +14 % au 14B). Sur l'axe même où vit la thèse du
projet, le marché est refusé : **les résultats se publient, le cœur ne bouge
pas.** `KvStore::Cat` reste le défaut, `LLVQ_KV_PREALLOC` et `LLVQ_GRAPH_AB`
restent des modes de mesure de `bin/fusedrun`, la config servie v1 est
inchangée.
🚨 **Ce +47 % est *calculé*, JAMAIS MESURÉ** : la seule fenêtre qui ait tourné
est `prealloc(256)`, dont le coût mémoire est 0,038 Go — négligeable, ce qui
explique qu'aucun job A2 n'ait vu passer ce poste. Le −0,83 % de é1b est un
coût en **temps**, pas en mémoire.
⚠️ **Et ce qui est refusé est le couple (graph, fenêtre 8k), pas le graph** :
à 2k la même arithmétique donne +12 % de mémoire pour +12,6 % de débit, et
l'arbitrage s'inverse. La ligne rouvrirait sur une cible de contexte plus
basse, sur le cache KV en q8 (livré, non défaut), ou sur une capture
compatible d'un cache qui grandit. Détail et conditions de réouverture :
[`proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md`](../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md) §É7.

## 2026-09-02 — D0 : la roadmap recherche est adoptée, M2 passe devant M1, et les deux boutons sont livrés sans qu'un run ait tourné

Décision d'opérateur en trois OK, après lecture des trois documents de la
branche `claude/repo-audit-optimization-20r2pz` (roadmap, audit, projection
des gains) : **adoption**, **plafond de 5 $ pour la vague 1**, **M1 en
parallèle sur le Mac**. Branche fusionnée dans `main` (`1e8583c`, docs
seulement, branche de secours `secours/main-avant-roadmap-2026-09-02`), D0
consigné au journal de la roadmap (`91b5921`).

- ✅ **Le premier sujet est M2, pas M1 — et c'est un changement d'ordre par
  rapport à la roadmap, motivé.** M2 (restaurer un type de projection en f16
  depuis le checkpoint dans le 4B scellé, sept bras + deux groupes + deux
  contrôles, MMLU apparié) est un **A/B à fichier constant** : aucun bras ne
  recalibre, donc la barre est la SE appariée de **0,43 pp** et non les
  2,92 pp de bruit inter-graines. Il ne dépend ni du verdict de M1 ni du σ de
  calibration, il coûte **≈ 2,3 $** (*estimé* depuis 0,19 $/bras mesuré le
  08-25), et c'est le seul sujet de la roadmap qui **ne peut pas être tué** :
  c'est une mesure. Ce qu'il décide : l'ouverture de Q5 (précision mixte, +0,05
  b/poids si la cible est `k`) contre Q6/F3 — et le coût décide autant que le
  signal (une cible `down` coûterait ≥ +0,49 b/poids, soit l'AWQ).
- ✅ **Faisabilité vérifiée dans le code, pas supposée** : `decode_matrix`
  dé-rotationne à la sortie (`llvq-artifact/src/format.rs:435`), le chargeur
  scellé construit le modèle depuis une table nom → tenseur en base naturelle,
  et `Proj::Dense` ne porte aucune rotation — restaurer un type, c'est écraser
  des entrées de cette table par les tenseurs du checkpoint, ramenés au dtype
  exactement comme le `VarBuilder` du bras f16. **Avec les sept restaurés, le
  modèle EST le checkpoint** (test
  `the_mmap_source_narrows_like_the_var_builder_does`), ce qui donne un
  contrôle haut exact (70,32) en plus du contrôle bas (55,59).
- ✅ **Livré** : `LLVQ_RESTORE_F16=<types>|all` dans `sealed::load_with_restored`,
  `bin/mmlu` et `bin/ppl` — refus d'un nom inconnu, d'un type absent du
  fichier, d'une forme qui diffère, d'une restauration sans `LLVQ_MODEL`, et
  de la variable sur un checkpoint ; la restauration est imprimée dans le
  label, l'en-tête du dump et la ligne de résultat (9 tests). Et
  `LLVQ_H_SHRINK=ρ` (M1) dans `calib::RunConfig::h_shrink` et `bin/smoke` :
  `H ← ρ·H + (1−ρ)·diag H` en **base naturelle, avant la rotation**, appel
  sauté à ρ = 1, clé `.state` écrite seulement si ρ < 1 pour que les shards
  d'avant reprennent inchangés (5 tests, dont les deux mutants qui comptent :
  diagonale touchée, ρ² au lieu de ρ).
- ⚠️ **Préregs en BROUILLON, non tamponnés, rien n'a tourné** :
  [`proofs/preregistration-m2-attribution-4b-2026-09-02.md`](../proofs/preregistration-m2-attribution-4b-2026-09-02.md)
  (11 bras, contrôles à rejouer au centième, règle de lecture cible/plat/diffus,
  réplique sur la graine 3 de F5 si une cible est désignée) et
  [`proofs/preregistration-m1-hessienne-shrink-2026-09-02.md`](../proofs/preregistration-m1-hessienne-shrink-2026-09-02.md)
  (12 runs 0,6B/28 blocs, ρ ∈ {1 ; 0,9 ; 0,7 ; 0,5} × 3 graines, contrôle
  (ρ = 1, s = 1) → 38,4507, adoption = étendue ÷ 2 sans déplacer la médiane de
  plus d'une étendue, kill ρ\* = 1). `ops/m1_shrink_queue.sh` **refuse de
  démarrer sans le `.ots`**. Les deux prédictions signées sont **opposées au
  prior de l'audit** : `q`/`k` sous 1,5 pp à cause de `q_norm`/`k_norm`, et
  kill pour M1 (le bruit croît avec la profondeur : composition séquentielle,
  pas estimateur).
- ✅ **Plomberie de M2 vérifiée à 0 $ sur le fichier publié, avant la carte**
  ([`mesures/m2-plomberie-mac-2026-09-02.txt`](mesures/m2-plomberie-mac-2026-09-02.txt)) :
  comptes de poids exacts (94 371 840 pour `k_proj`, 3 633 315 840 pour
  `all`), **« all restauré » = le checkpoint à 114/114 picks, Δ = +0,00 pp**,
  refus qui refusent. `limit 2`, donc aucun chiffre de qualité n'en sort.
- ⚠️ **Une note posée sur F1 avant qu'on y touche** : l'estimation de perte de
  forme de la projection (+7-9 % de MSE) donne une rétention gaussienne de
  88,9-89,6 %, **sous le kill de F1b** (90,3 %). F1a doit compter la rétention
  exacte de la région de mise en forme avant toute ligne de code.
