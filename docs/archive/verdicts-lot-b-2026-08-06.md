# Verdicts du lot B — nuit du 2026-08-05 au 06

> File séquentielle sur M3 Max (15 étapes, 23:22 → 02:36, tout vert), journal
> brut : [`mesures/nuit-b-2026-08-06.txt`](../mesures/nuit-b-2026-08-06.txt),
> logs complets : `~/llvq-nuit-b/`. Protocole des runs 3 blocs : Qwen3-0.6B,
> 16 fenêtres de 2048 (33k tokens) sauf B3, éval 12 fenêtres de 2048,
> `leech1c12`, rotation d'entrée, group_scales off. Corpus vérifié bras par
> bras dans les logs. Les mesures e4/e8 portent l'empreinte de tokens
> `3f1baca9033bf251`, identique aux références publiées — comparables.

## B1 — La barre d'erreur existe enfin

| run | ppl quantifiée |
|---|---|
| ancre (préfixe contigu, λ=1e-2) | 20,6643 |
| graine 1 / 2 / 3 | 20,6239 / 20,4709 / 20,7687 |
| λ=3e-3 / λ=3e-2 | 20,6740 / 20,6014 |

**σ(graines) ≈ 0,15 ppl ≈ 0,7 %** (3 points — estimation grossière, mais
c'est la première du projet). Règle pratique : sur un run unique de 3 blocs,
**tout effet < ~1,5 % (2σ) est du bruit**. Le damping est **nul** sur la
plage 3e-3..3e-2 (écart 0,35 %, sous 1σ) — exactement la prédiction inscrite
dans le code depuis des semaines. ✅

## B2 + B3 — La famille calibration est PLAFONNÉE

- **Oracle** (calibrer sur wikitext-2 *test* lui-même — la triche maximale,
  plafond de tout ce que volume/corpus/longueur peuvent rendre) : 20,3289
  contre 20,6643 à l'ancre = **−1,6 %**, soit 2,3σ — au bord de la
  significativité, et ça ne referme que **29 % de l'écart de quantification**
  des 3 blocs.
- **Courbe de volume** (c4, une variable) : 131k → 500k → 1,73M tokens
  (le corpus chargé plafonne à 847 fenêtres, pas 977) :
  20,9818 → 20,9564 → 20,7254 = **−1,2 % pour ×13 de volume**, ~1,7σ.

**Conséquence directe : le run calibration ×100 à 20-27 $ (D2 du plan) est
enterré.** Même la contamination délibérée ne rend que 1,6 % de perplexité ;
×100 de volume honnête en rendra moins. Le critère écrit à l'avance dans
`pistes-battre-q4.md` (« si l'oracle ne rend que 2-3 %, le suspect est
plafonné ») est atteint par le bas. Par élimination — la rotation de sortie
est morte hier (Table 9), la calibration est bornée aujourd'hui — **le design
C (chemin des magnitudes) reste seul suspect chiffré du déficit qualité.**

⚠️ Réserves : 3 blocs, 0.6B, perplexité seulement — le précédent group_scales
(signe inversé à pleine profondeur) interdit d'en faire un théorème ; et la
piste *composition* du corpus (P18, raisonnement) vise le mécanisme MMLU, que
l'oracle ppl ne borne pas.

## B4 — L'embedding int8 est gratuit ; l'int4 coûte 1,5 % de ppl

| artefact | ppl (réf 16,9415) | MMLU micro (réf 56,09 ± 1,36) | froid |
|---|---|---|---|
| `q4b-e8.llvq` (int8) | **16,9379** — identique (−0,02 %) | 55,44 ± 1,35 (−0,65 pp, sous le σ) | **1,406 Go (−365 Mo, −21 %)** |
| `q4b-e4.llvq` (int4) | 17,1986 (**+1,52 %**) | 55,74 ± 1,35 (−0,35 pp, bruit) | 1,211 Go (−559 Mo, −32 %) |

**Verdict : int8 validé, sans aucune perte mesurable** — le « interdit de
publication, qualité ABSENT » de `fiche-4b.md` est levé pour e8. L'int4 est
défendable (le MMLU ne le distingue pas, e4 sort même 0,3 pp au-dessus d'e8
— du bruit) mais paie 1,5 % de perplexité, réelle car l'éval est
déterministe à empreinte identique. **Feu vert qualité pour le chemin
d'exécution int8 (E1 du plan)** : les −365 Mo à froid sont acquis, les
−0,39 Go de VRAM attendent le noyau de gather int8.

## B5 — L'histogramme par classe (matin du 06) : la fourchette E2 penche vers le haut

Nouveau bin `llvq-bench/src/bin/classhist.rs` (la sortie de `rtbits`, citée
par les docs, reste intacte), passé en revue adversariale : chaque chiffre
reproduit par une dérivation indépendante, aucun bug.

- **8,7234 % des blocs** (13 144 531) appartiennent à des classes violant le
  co-design « ≤ 2 valeurs par résidu mod 4 » — la branche **haute** de la
  fourchette E2 : compter **~3,3 b/poids** (avec overlay d'exceptions), pas
  2,92. Fait structurel (pigeonhole, vérifié sur les comptes) : **tout bloc
  L=5 est violant**, donc le plafond L≤4 élimine mécaniquement 38,8 % des
  violants ; il en reste **5,5279 %** parmi les blocs L≤4, concentrés dans
  une poignée de classes (motif type : {0,4,8} ou {1,5,9} coprésents).
- **L'entropie de l'index est 46,6536 bits/bloc contre 47 payés** (0,35 bit
  de marge, 0,74 %) : le format v1 est quasi optimal pour sa propre
  distribution — le codage entropique du rang (P9) est **définitivement
  clos**, cette fois par la structure et plus seulement par zstd.
- Recoupements exacts : 150 681 600 blocs, 286 classes observées, L=5 =
  3,3824 %, 58/301 classes violantes (46 paires + 12 impaires), jamais 4
  valeurs par résidu, jamais 3 dans les deux.

## B6 — Le swap L≤4 (en cours de mesure)

`llvq-bench/src/bin/lswap.rs` : revue adversariale favorable sur les six
invariants (blocs L≤4 bit-identiques via `read/write_matrix_raw` épinglé par
`raw_passthrough_is_byte_identical` ; gain/échelles/queue/embedding
intouchés ; boule 12 + cap 4 sans débordement du format v1 ; stride archive
FIXE 48 bits/bloc donc aucun offset déplacé). Deux leçons de la revue :
(1) le premier run est mort en route et laissait un fichier tronqué
*plausible* — discipline tmp+rename ajoutée avant relance ; (2) ⚠️ le gain
copié n'est plus optimal pour la nouvelle direction : le Δppl mesuré est un
**majorant** du coût réel d'un encodeur L≤4 qui rechoisirait son gain.
**Résultats du swap** : 5 096 688 blocs échangés exactement, hors-swap
bit-identique (taille au octet près), gains intacts, rechargement de
contrôle passé. Prix angulaire homogène : cos(ancien, nouveau) moyen
**0,931**, min 0,917, max 0,944 — pas de blocs pathologiques.

**Mesures** (fichier scellé `~/qwen3-4b-llvq-L4swap.bin`, protocole et
empreinte de tokens `3f1baca9033bf251` identiques à la référence) :

| | référence (scellé publié) | swappé L≤4 | Δ |
|---|---|---|---|
| ppl wikitext f16 | 16,9415 (×1,3846) | **17,7459** (×1,4503) | **+4,75 %** |
| MMLU micro | 56,09 ± 1,36 | **55,43 ± 1,34** | −0,66 pp (sous le σ) |

**Verdict : le plafond L≤4 par swap post-hoc coûte +4,75 % de perplexité —
il détruit le titre.** Notre marge sur QTIP était de 0,6 % (16,94 contre
17,04) : le fichier plafonné repasse AU-DESSUS (17,75 > 17,04). L'éval étant
déterministe à empreinte identique, l'effet est réel, pas du bruit. Le
« −0,26 pt gaussien » sous-estimait d'un ordre de grandeur ce que la vraie
distribution paie. Deux issues restent ouvertes, dans l'ordre :
1. **L'overlay épars (E1b′) prend la tête** : qualité exacte pour
   +0,22-0,24 b/poids — à ~4,23 b/poids il passe sous les 4,5 sans toucher
   à la perplexité. C'est désormais la voie recommandée pour le point ~4,2.
2. La **requantification avec plafond dans la boucle** (GPTQ compense en
   ligne, et le gain serait re-choisi — le +4,75 % est un majorant à double
   titre) peut être bien meilleure que le swap ; à chiffrer par un vrai run
   (~3,5 h Mac) seulement si l'overlay ne suffit pas.
3. **E1a (sans plafond, 4,67 b/poids)** n'est pas concerné : zéro perte par
   construction, sa valeur relative monte encore.

Le MMLU, lui, ne bouge pas (−0,66 pp, sous l'erreur d'échantillonnage) —
cohérent avec la leçon du projet : perplexité et MMLU mesurent des choses
différentes, et le plafond abîme la restitution fine avant le raisonnement.

## Ce que ça change au plan d'action

- **D2 (calibration ×100) : mort.** ~25 $ économisés, et le GPU reste pour
  la mesure et le format.
- **D1 (design C) : promu seul suspect qualité chiffré** — prochain
  chantier qualité, avec son budget MMLU (~4 h/point).
- **E1 (runtime int8 embedding/lm_head) : dérisqué**, int8 d'abord.
- **B5** (histogramme par classe) et **B6** (swap L≤4 → Δppl) restent à
  faire — du code, pas des runs.
- Le patch `LLVQ_CALIB=wikitext2-test` de `smoke.rs` (7 lignes, l'oracle) est
  à commiter avec ce doc.
