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

## État courant (au 2026-08-12)

Le noyau fusé sert un Qwen3-4B 2 bits à **88,4–88,5 tok/s dans 2,60 Go**
(×2,03 brut contre notre bras dense, **×1,12 à tête identique** — la seule
formulation qui mesure le noyau). Layout de production **`Planes14`**
(4,804 b/poids, 2,15× vs FP16), embedding **q8**. Qualité : le point dur —
−14,7 pp de MMLU au 4B, −10,6 au 8B, **−6,85 au 14B (apparié)** ; la courbe
d'échelle a un **genou**. L'axe noyau est épuisé proprement (Golay70 v2 :
1,77× < seuil pré-enregistré 2,0×, non adopté). La suite : [`PLAN.md`](PLAN.md).

---

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
IC95 [+4,52 ; +9,12], écart AWQ 6,09 pp. **La courbe a un genou** : fonte de
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
