# Revue du papier TACO en mode rapporteur — 2026-08-22

> Objet : `paper/main.pdf` au commit `1ae5a35` (identique à `v4.pdf`), 21 pages
> dont ~19,8 comptées sur 20 strictes (références exclues, annexe comprise).
> Méthode : 8 lectures indépendantes (adéquation TACO, noyau GPU, quantification
> PTQ, statistiques, rédaction, figures, cohérence des chiffres, stratégie de
> publication), 41 vérifications contradictoires des critiques lourdes contre
> le papier ET le dépôt, une critique de complétude, puis quatre vérifications
> manuelles des points qui changent le verdict (journal `f2-p3`, `fused_cuda.rs`,
> recalcul du genou sur les trois graines, issue candle publique).
> Tout ce qui suit est étiqueté *vérifié* (relu sur pièce) ou *avis*.

## 0. Le verdict en une phrase

**Crédible, oui — au-dessus du standard TACO sur la mesure. Lisible, non. Et
le papier se présente comme une défaite là où il porte une contribution.** En
l'état, vote le plus probable : Major revision chez un rapporteur bienveillant,
Reject chez un rapporteur de noyaux. TACO n'accorde **qu'une** Major revision ;
tout ce qui peut être corrigé avant soumission doit l'être avant.

## 1. Ce qui tient (et qu'il ne faut pas perdre en coupant)

*Vérifié.* Les chiffres sont cohérents avec leurs CSV et journaux : les 10
lignes de la Table 1, la Table 3 A100, les Tables 4–8, les 9+6 intervalles
appariés, les transcriptions de la Table 9, les 17 chemins `docs/mesures`
cités existent, `check_tables.py` passe, le commit QTIP est épinglé et
vérifié par sha256, et toute l'arithmétique dérivée recalculée (« of bound »,
2,27×, 2,7 %, 59,6 %, 3,35 b/w, ⌈log₂N⌉ = 47, N(13), 102 400 − 101 376) est
juste. Le protocole (dix bras dans un processus, médiane de rapports par
round, vérification f64 ligne à ligne avant tout chrono, concurrents portés
à commit épinglé, deux hiérarchies mémoire) est **meilleur que ce que TACO
publie d'ordinaire**. Le résultat Planes14 « plus petit ET plus rapide à Go/s
constant » est un vrai résultat systèmes. Aucune lentille n'a trouvé de
chiffre qui renverse un verdict.

## 2. Les risques de rejet, par ordre de gravité (tous vérifiés sur pièce)

### R1 — Le papier se lit comme un résultat négatif sans leçon réutilisable
Titre « at Matvec Speed » (faux à 65 % de borne, faux sur A100), abstract qui
énonce une borne puis la rétracte dans le même paragraphe, intro p. 2 « *on
that axis the lattice loses* », §6 « *We are slower than the 2-bit kernel we set
out to beat, and the margin is large* ». Un rapporteur fatigué retient « Leech
perd contre QTIP » et vote « insufficient contribution ». Or la contribution
existe : premier décodeur 301 classes sans divergence ; l'in-VRAM rate comme
axe de conception distinct du taux disque, avec son genou mesuré vers 4,3 b/w ;
le prix de l'index combinatoire chiffré (2,18 Go lus contre 0,91 : **2,40× les
octets pour 2,27× le temps** — c'est ça le mécanisme, pas « l'efficacité
noyau ») ; le basculement de régime L40S/A100. Elle n'est jamais présentée
comme LA contribution.

### R2 — Le noyau, contribution annoncée, n'est pas décrit
§2 = 1,3 page, zéro figure, zéro pseudocode, zéro géométrie de lancement. Le
« one warp per output row » n'apparaît qu'en incise §3.1. Tout existe dans le
dépôt et n'est pas dans le papier : 256 threads/CTA = 8 lignes, tuile de 128
blocs = 12 288 o de shared, 40 registres, 0 spill, 4 LDG.32 par bloc à stride
14 o, grille d_out/8 (`llvq_planes.cuh`, `f2-p3-qtip-banc:132-147`). Les
mécanismes d'exceptions de Planes12x et Golay70 ne sont décrits nulle part.
Pour un journal de *code optimization*, c'est le cœur manquant.

### R3 — Une erreur factuelle contre le journal cité par la Table 1
§3.2 : « *the fused kernel is Slot32-only* » et « *roughly a third of the gap
has never been attacked* ». **Faux** : `f2-p3-qtip-banc-2026-08-21.txt`
l. 265-276 (le job même de la Table 1) mesure la fusion q/k/v + gate/up **sur
Planes14** : 5,096 → 4,504 ms, −0,594 ms [0,586–0,596], −11,7 %, bit-identique
sur 921 600 lignes. C'est précisément l'expérience que tout rapporteur de noyau
demandera (« pourquoi ne pas appliquer le 39 % au layout servi ? ») — et elle
est faite. Le papier la nie.

### R4 — « Resolved on perplexity » ne survit pas aux graines du papier
*Vérifié par recalcul* (NLL par fenêtre de `docs/data/f5-nll/`, 8B et 14B de
leurs campagnes, même test apparié que `ppl-genou.csv`). Le genou, en
remplaçant le 4B publié par chaque graine de calibration :

| bras 4B | ppl 4B | pas 4B→8B (t) | genou (pas1 − pas2) | IC95 | t |
|---|---|---|---|---|---|
| publié | 16,9422 | −0,1265 (−9,62) | **−0,1010** | [−0,1377 ; −0,0643] | −6,06 |
| seed1 | 16,7425 | −0,1146 (−9,22) | −0,0891 | [−0,1274 ; −0,0509] | −5,13 |
| seed2 | 15,8836 | −0,0619 (−5,47) | −0,0365 | [−0,0744 ; **+0,0014**] | −2,12 |
| seed3 | 15,1027 | −0,0115 (**−0,93**) | **+0,0139** | [−0,0268 ; +0,0547] | +0,75 |

Avec seed3 le pas 4B→8B n'est même plus significatif et le genou **change de
signe**. Le papier porte « resolved on perplexity » dans l'abstract, l'intro,
la Table 8, §5, l'Annexe A et la conclusion, tout en publiant l'étude de
graines qui le réfute (§5 « What one calibration draw is worth »). Un
rapporteur statisticien qui fait ce calcul (cinq minutes, données commitées)
conclut à une sur-affirmation, et c'est la pire espèce : celle que le papier
se fait à lui-même. **Tout le §5-échelle et l'Annexe A reposent sur un tirage
par taille.** Formulation survivable : « conditional on the three artifacts
as built; a calibration re-draw at 4B alone moves the knee from −0.10 to
+0.01 ».

### R5 — Le chemin servi n'a pas de prefill, et le papier ne le dit nulle part
*Vérifié* : `fused_cuda.rs:17-22` — « *It does not handle more than one token
per call … A prompt of l tokens loops l times … useless for scoring a
2048-token window. Perplexity therefore keeps the dense path.* » Conséquences :
le 87,0 tok/s de l'abstract est mesuré sur un prompt de **5 tokens**
(`fusedrun.rs:43`) ; un prompt de 2 048 tokens coûterait ~10 s de projections
sur le 4B ; et **les colonnes ppl/MMLU « LLVQ fused (ours) » ne passent jamais
par le noyau** — l'équivalence est prouvée (bijection + f64), pas mesurée à
travers lui. `grep -ci prefill paper.txt` = 0. Découvert par le rapporteur,
c'est un Reject ; déclaré (« decode-phase GEMV, batch 1, prompt ℓ en ℓ GEMV,
TTFT à 512 tokens = x s »), c'est une limite.

### R6 — Le diagnostic A100 « compute-bound / ridge point » n'est pas étayé
Le noyau-plancher (qui ne lit rien) passe de 2,306 à 4,107 ms entre L40S et
A100 : ×1,78, soit le rapport des horloges SM (2 520/1 410 = 1,79). Une chaîne
de FMA dépendants ne « croise » pas un ridge point ; c'est de la latence
d'émission. Et « *every decoding arm falls below the FP16 baseline* »
(abstract) est contredit par la Table 3 elle-même : AWQ à 1,82×. Dire « every
lattice arm », et remplacer le mécanisme par « latency/issue-bound at this
launch geometry », ou relever les horloges (`nvidia-smi --query-gpu=clocks.sm`,
0 $) et l'étayer.

### R7 — Anonymat : l'issue candle désanonymise en une requête
*Vérifié* : `huggingface/candle#3871` est publique sous le handle de l'auteur
et contient « 778 MB », « broadcast_matmul », « Qwen3-4B », « lm_head ». Le
papier cite les quatre et « reported upstream with a reproducer and a patch ».
Les guidelines TACO : « *subject to immediate rejection* ». Arrondir (« a
transposed copy of the vocabulary table, ∼0.8 GB per token »), retirer « we
reported ».

### R8 — Densité et registre
Abstract : 364 mots, ~30 nombres. Une seule figure en 20 pages, jamais citée
dans la section qu'elle illustre (seule référence : Limitations). Phrases à
28 mots de moyenne, ~20 au-delà de 60 ; ~45 % des phrases du corps portent une
négation ou une qualification. Registre de cahier de laboratoire : dates,
dollars, commits, horodatages, 31 chemins `docs/…`, « ten days later »,
« previously circulated », « each passed adversarial review » (Table 5 — à
supprimer : en double-aveugle la question « adversarial review par qui ? » n'a
pas de bonne réponse). Résultats répétés 4 à 5 fois (genou ×7, mur 32B ×5,
« never divide » ×4, « median of per-round ratios » ×4). Les cinq phrases
qu'un rapporteur citera contre le papier : intro p. 2 « *it is the
contribution, and on that axis the lattice loses* » ; §3.1 « *our motivation
was right about the world, wrong about us* » ; §6 « *We are slower than the
2-bit kernel we set out to beat* » ; abstract « *so that cap was a property of
our launch geometry, not of the card* » ; §3 « *the layout missed a
pre-registered bar twice* ».

## 3. Corrections Major (ne renversent pas un vote, mais pèsent)

- **Table 7 ↔ Table 8** : 65,63 (bras servi q8) vs l'écart 7,49 qui suppose
  65,52 (scellé) — le nombre qui réconcilie n'est imprimé nulle part.
- **Taux disque sous trois étiquettes incompatibles** : « exactly 2 bits/weight
  of payload » (§2.1), « 2.07 bits per weight of payload; §5 » (intro, §3.1 —
  renvoi pendant : §5 ne le contient pas), 2,17 écrit. Une phrase qui pose
  2,000 (code) / 2,07 (effectif) / 2,17 (écrit), une fois.
- **Coûts périmés** : « $84.68 across 58 jobs » vs `jobs.csv` à 67 jobs,
  86,12 $ ; « $2.33 for the whole kernel campaign of §3 » n'inclut pas le run
  à dix bras de la Table 1 (0,89 $).
- **Petites fautes visibles sans le dépôt** : note § Table 6 « 5.15 » → 5,17
  (2,60 × 8 / 4,022) ; §3.3 « 427 » → 428 (Table 3, même page) ; Table 5
  « −1.6% » → −1,7 % ; ×1,384 → ×1,385 (16,9422/12,2369).
- **Graines** : le 16,9422 publié est **hors** de l'intervalle des trois
  graines [15,10 ; 16,74] — pire que les trois — et le papier ne situe pas le
  point publié (préfixe contigu, shard C4 différent : journal F5). À dire en
  une phrase, sinon un rapporteur le découvre.
- **« within noise on perplexity »** (§3.1, QTIP vs nous) : aucune barre
  n'existe pour QTIP. Écrire « comparable in their respective harnesses at
  matched code rate (Table 9) ».
- **Mémoire** : 2,60 / 5,45 / 9,39 Go sont des poids à 133 tokens de contexte.
  Le « 70B dans ∼20 GB » de l'intro est un taux *disque* ; servi à 5,1 b/param
  c'est ~45 Go, KV non compris. Une phrase de budget VRAM complet (poids + KV
  à une longueur nommée), ou retirer le 20 Go.
- **Statistiques** : « no cross-size pairing exists » (MMLU) est faux — les
  neuf bras partagent les 2 280 mêmes questions (`qhash` par question dans
  `docs/data/mmlu-dumps/`) ; l'argument de puissance « 49 140 paired tokens
  against 2 280 unpaired questions » compare deux unités ; tests multiples
  sans correction avec une narration sur p ≈ 0,0497 (« clears zero by five
  thousandths »). Ramener l'appareil à **une table** (9 paires ppl, 9 paires
  MMLU, 4 pas, 2 genoux : estimation, IC95, n, verdict) et ≤ 150 mots.
- **Related work** : llama.cpp IQ2_XXS/XS/S et le noyau E8P de QuIP# sont les
  « codes de réseau qui tiennent en LUT » — exactement le contrefactuel de
  la thèse « unfolds worse » (10¹⁴ points ne tiennent pas en LUT, 2¹⁶ oui).
  Absents. Une phrase suffit, un point mesuré serait mieux.
- **Availability** : « 8B and 14B … reproducible at the costs » — le 8B
  re-scellé existe (B3, 08-18) mais un rejeu de quantification ne rend pas le
  même objet (drift C4 déjà constaté) ; le bras QTIP dépend de sources GPL
  fetchées au job, non redistribuées. Une demi-page « artifact appendix »
  (fichier additionnel, hors pages) : matériel, durée, coût, ce qui est GPL,
  ce qui est publié.

## 4. Le plan de figures (à payer par les coupes du §5)

Une figure en 21 pages, pour un papier de noyau et de formats mémoire, est le
défaut le plus visible et le moins cher à corriger. Par impact :

| # | Figure | Remplace | Coût | Source / outil |
|---|---|---|---|---|
| A | **Cartes d'octets des quatre records** à l'échelle du bit : Slot32 [class 9][gain 1][smask 24][m1..m4@24] = 106-130 b, fenêtre 5 mots ; Planes14 […][3 plans][pad 6] = 112 b, stride 14 o ; Planes12x = 96 b + table d'exceptions (3,38 %) ; Golay70 = 72 b + codewords 16 Kio. À droite : b/poids et vs FP16. | 10 lignes de prose p. 3-5 | 0,3 p. | `llvq_*.cuh` (largeurs) + `echelle-formats.csv` ; `make_figures.py` |
| B | **Flot de données du noyau Planes14** : x → rotation (shared 4n o, barrière = mur §6) → un warp/ligne, lane = un bloc, fenêtre 4 mots → ClassRec (512×24 o, L1) → 24 fma sur x en shared → warp_sum → ×scale + queue f32. Flèches annotées des ms de la Table 2. **Second panneau : la fenêtre glissante QTIP** (16 bits, 12 de recouvrement, LUT 2 Kio) — la thèse du papier, jamais dessinée. | Table 2 (si annotée) | 0,3 p. | TikZ ; `attribution-cuda-2026-08-05.txt` |
| C | **Courbe d'échelle à trois panneaux** avec IC95 : excès ppl (3 paires), écart MMLU (3 paires), b/param vs AWQ avec part d'embedding. Caption : un verdict par métrique **conditionné au tirage** (R4). | Table 8 + la liste des neuf intervalles de l'Annexe A | ≈ 0 net | `ppl-appariee.csv`, `mmlu-appariee.csv` ; `make_figures.py` |
| D | **Croisement L40S / A100** en dumbbell : Go/s atteints par bras sur les deux cartes, règles FP16 661/1052 et cuBLAS. Dit en un regard ce que la Table 3 interdit de diviser. | Table 3 | ≈ 0 net | `echelle-formats*.csv` |
| E | **Profil MMLU par matière**, 57 points, x = FP16, y = 2 bits, ligne 25 % ; la seule preuve de « damage reasoning far more than recall », affirmé quatre fois. | rien | 0,3 p. | `mmlu-dumps/` |
| F | Barres empilées de l'attribution Slot32 (3,78 plancher + 0,681 + 0,041 + 0,118 + 0,396 + 0,803 = 5,82 ms) avec **le terme récupéré par la fusion Planes14**. | Table 2 | 0,1 p. net | si B n'absorbe pas Table 2 |

Corrections de la **Fig. 1** dans tous les cas (30 min dans `make_figures.py`) :
deux collisions d'étiquettes (« ceiling » sous QTIP, Planes12x/Golay70 hoisted) ;
renommer l'hyperbole « byte bound » et la règle « no-weights kernel (our
launch geometry) » — le vocabulaire floor/ceiling est inversé par rapport à ce
qui est tracé ; **citer la figure dans §3**.

## 5. La liste de coupes (≈ 4,5–5,5 pages libérées)

(A) Abstract 364 → ≤ 220 mots, ≤ 8 nombres, sans la borne 4,77 % et sa
rétractation. (B) Intro 1 005 → 600 : garder 87,0 / 2,60 Go / ×1,384 /
−14,73 ; retirer 32B, 14B, « 14.6 (served arm; 14.73 …) », tout le genou.
(C) §3 Methodology 330 → 120 mots, asymétries et « two phases » en Annexe B.
(D) Golay70 + hoisted + « The bar had moved » 520 → 200. (E) §3.1 QTIP 750 →
350 : retirer « a citation, not a measurement », « three possible verdicts
were written and timestamped », « true and useless », la prédiction 59,6 % à
deux phrases. (F) §3.2 : retirer l'auto-critique de l'estimation antérieure.
(G) §4 1 238 → 750 : **supprimer Table 5** (journal de bord), « brought to the
runner on 2026-08-19 », les deux reconstructions 1,78/1,24, Marlin repack.
(H) §5 1 737 → 1 000 : « two denominators » en deux phrases, genou en une
phrase + renvoi, « Does the 4-bit baseline start paying » en deux phrases,
coûts → une ligne en Availability, zéro chemin `docs/`. (I) §6 1 162 → 600 :
trois doublons (échelle, « over FP16 », « carry ranges »), Hadamard 230 → 80
mots avec table numérotée. (J) Annexe A 1 410 → 800 : contrôles
d'authenticité → README, « the bar the −43% lacked », t = 2,2063 vs 2,2010,
une seule paramétrisation ; **Table 9 remonte dans §5** (c'est la comparaison
que tout rapporteur PTQ cherche). (K) Tout le processus (dates, dollars,
commits, horodatages, chemins) → Annexe B en **deux tables** : critères
pré-enregistrés avec dates ; Figure/Table → CSV → journal.

Structure cible : §1 Intro (1 p., quatre contributions en liste) · §2 Le
décodeur fusé (1,5 p., Fig. A + B) · §3 Layouts sur une carte (3 p.) · §4
Contre les noyaux déployés (1,5 p. : AWQ, QTIP, ce que la fraction de borne
peut dire) · §5 Intégration (1,5 p.) · §6 Évaluation (2,5 p., Fig. C + Table 9)
· §7 Limitations (1 p., avec l'enveloppe : decode batch 1, prefill en ℓ GEMV,
NVIDIA seul, dépliage N s au chargement, VRAM hors KV) · §8 Related · §9
Conclusion · Annexes A (tests, 1 p.) et B (protocole et provenance, 0,7 p.).

## 6. Un papier ou plusieurs ?

**Un seul, recentré.** Les morceaux séparables :

| bloc | contenu | seul, c'est publiable ? |
|---|---|---|
| A | décodeur multi-coquilles + famille de layouts + échelle à dix bras | **oui, c'est le papier TACO** |
| B | plancher, attribution, A100, QTIP/AWQ dans le même processus | non : c'est l'appareil de preuve de A ; publié d'abord (workshop/IISWC), il ferait du TACO une « conference extension » à 30 % de neuf à justifier |
| C | intégration + défaut `broadcast_matmul` | issue GitHub + billet ; dans le papier, un résultat (la série à tête identique) et une note |
| D | courbe qualité/échelle 4B/8B/14B, statistiques appariées, graines | **non en l'état** : une graine par taille, étendue inter-graines de 10,3 % qui englobe le second pas, genou qui change de signe (R4), qualité 4B à 5 points MMLU sous le papier source. Une note arXiv compagnon au mieux ; à TMLR/EMNLP il faudrait 3 graines × 3 tailles (~80 $) et le test set entier |

D passe en supplément (Fig. C + une table + un paragraphe dans le corps).
Les ~4,5 pages ainsi libérées sont exactement celles que le noyau n'a pas.

## 7. Trois options, avec les probabilités

*Avis.* P = passer le premier tour (Accept / Minor / Major), entre parenthèses
l'acceptation finale. Base : TACO rejette ~40-50 % au premier tour et n'accorde
qu'une Major. Les huit lentilles vont de 38 à 78 % (médiane ~50) ; je suis en
dessous parce que trois risques (R4 prefill, R5 genou, R7 anonymat) ont été
trouvés par des vérificateurs, pas par les lecteurs, et qu'un rapporteur en
trouve au moins un.

**Option 1 — Polish seul** (0 $, ~1 semaine : corriger R3, R6, R7, les
chiffres du §3, requalifier « resolved », déclarer le prefill en une phrase ;
pas de restructuration, pas de figure) : **35 % (25 %)**. On retire les causes
de rejet immédiat ; on garde un papier de noyau sans description de noyau,
une figure, et une défaite en titre. Un rapporteur hostile suffit.

**Option 2 — Restructurer, 0 $, 3-4 semaines** : recadrage contribution
d'abord (titre sans « at Matvec Speed », abstract ≤ 220 mots, quatre
contributions), D en supplément, figures A-D (+E si la place), liste de coupes
du §5, enveloppe déclarée (prefill, batch 1, NVIDIA, dépliage, KV), corrections
R3/R4/R6/R7, la fusion Planes14 **déjà mesurée** publiée dans §3.2 (−11,7 %) :
**60 % (45 %)**. C'est le rapport effort/gain le plus élevé : tout le matériel
existe dans le dépôt, rien ne se lance.

**Option 3 — Option 2 + ≤ 60 $ et 6-8 semaines** : fusion Planes14 et témoin
FP16 fusé **dans le chemin servi** (Table 1 remesurée à onze bras), WikiText
entier + MMLU complet sur les neuf bras (~40 $ : ferme « want of power » et
aligne sur le papier source), MMLU des trois graines (~3 $ : dit si −14,7 pp
est la méthode ou le tirage), `Planes12x` bout-en-bout (quelques cents : la
seule ligne où le produit bat le 4 bits en VRAM de > 10 %), horloges SM sur
les deux cartes (0 $), un TTFT à 512 tokens : **68 % (52 %)**. Ce que l'option
3 **n'achète pas** : un noyau qui rattrape QTIP. Même à 100 % de sa borne
d'octets, Planes14 plafonne à 16/4,804 = **3,33× FP16**, sous l'AWQ (3,38×) et
à 0,68× QTIP. Une nouvelle géométrie (split-K, multi-lignes par warp) déplace
le « vs FP16 » de 2,15 vers ~2,5 et ne change aucun verdict ; la garder comme
réponse aux rapporteurs pendant la fenêtre de Major revision, pas avant.

**Recommandation : option 2 maintenant, en y greffant les trois lignes de
l'option 3 qui coûtent moins de 5 $ (fusion Planes14 servie, `Planes12x`
bout-en-bout, horloges SM).** Le reste de l'option 3 est du matériel de
réponse aux rapporteurs. Le risque résiduel qui ne s'achète pas : un
rapporteur qui exige le régime batché ou un prefill GEMM — d'où l'importance
de scoper « decode-phase GEMV » dès le titre.

## 8. Dettes hors papier relevées en passant

- `CLAUDE.md` (en-tête) dit encore « le rapport à tête identique N'EXISTE PAS
  au 14B » et 88,4–88,5 tok/s : périmé depuis B2 (08-18) — ×1,41 [1,40–1,41]
  et 87,0 [86,8–87,0]. Le papier est à jour, pas le fichier de reprise.
- `campagne-finale.csv` dit « 2,60 = affichage carte » ; la note § de la
  Table 6 dit « host-side byte count, rounded » ; B2 rend 2,56 en compte hôte.
  Une provenance, pas deux.

## Annexe — le recalcul du genou (reproductible)

```python
import re, math
def nlls(p):
    b, c = [], []
    for l in open(p):
        m = re.search(r'window\s+(\d+)/12\s+nll\s+([\d.]+)', l)
        if m:
            c.append(float(m.group(2)))
            if int(m.group(1)) == 12: b.append(c); c = []
    return b
f16_4, awq_4, llvq_4 = nlls('docs/mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt')
f16_8, awq_8, llvq_8 = nlls('docs/mesures/campagne-8b-qualite-2026-08-08.txt')
awq_14, f16_14, llvq_14 = nlls('docs/mesures/campagne-14b-qualite-2026-08-10.txt')
seeds = [nlls(f'docs/data/f5-nll/seed{i}-nll.txt')[0] for i in (1, 2, 3)]
def t(d):
    n = len(d); m = sum(d)/n; s = math.sqrt(sum((x-m)**2 for x in d)/(n-1))/math.sqrt(n)
    return m, m/s, (m-2.200985*s, m+2.200985*s)
for name, l4 in [('publié', llvq_4)] + [(f'seed{i+1}', s) for i, s in enumerate(seeds)]:
    r4 = [l-f for l, f in zip(l4, f16_4)]; r8 = [l-f for l, f in zip(llvq_8, f16_8)]
    r14 = [l-f for l, f in zip(llvq_14, f16_14)]
    s1 = [b-a for a, b in zip(r4, r8)]; s2 = [c-b for b, c in zip(r8, r14)]
    print(name, t(s1), t([a-b for a, b in zip(s1, s2)]))
```
