# Pistes : faire tenir le format VRAM sous 4,5 b/poids — 2026-08-05

> 🗓️ **BANDEAU D'ÉTAT — dernière revue le 2026-08-08. Cette note a été
> exécutée : elle n'est plus une carte de pistes, c'est la généalogie d'une
> échelle qui a été mesurée jusqu'au bout.** Ce qui reste ouvert tient en deux
> lignes, tout le reste est tranché.
>
> | barreau de la note | statut au 2026-08-08 | source |
> |---|---|---|
> | **E1a (plans binaires AoS-14)** = `Planes14` | ✅ **gagné, mesuré ET branché** le 06. 4,804 b/poids (la note prédisait 4,80), **2,14–2,16× vs FP16**, **1,14× plus rapide que `Slot32` à contenu décodé identique**. Layout **par défaut** du chemin fusé | [`mesures/c1-planesbench-2026-08-06.txt`](../mesures/c1-planesbench-2026-08-06.txt), [`mesures/planes14-fusedrun-2026-08-06.txt`](../mesures/planes14-fusedrun-2026-08-06.txt) |
> | **E1b′ (overlay épars)** = `Planes12x` | ⚠️ **mesuré au banc, PAS branché.** 4,342 b/poids, 1,98× vs FP16, overlay **exact** sur les 1 105 920 lignes. C'est un point de fonctionnement prêt, pas une production | [`mesures/e2-golay70-bench-2026-08-07.txt`](../mesures/e2-golay70-bench-2026-08-07.txt), [`verdicts-nuit-2026-08-07.md`](verdicts-nuit-2026-08-07.md) §M2 |
> | **E1b sec (plafond L≤4 sans overlay)** | ❌ **MORT en qualité** : swap mesuré sur le fichier scellé, **+4,75 % de perplexité** (16,9415 → 17,7459), repasse au-dessus de QTIP. Le « −0,26 pt gaussien » sous-estimait d'un ordre de grandeur | [`verdicts-lot-b-2026-08-06.md`](verdicts-lot-b-2026-08-06.md) §B6 |
> | **E2 (étage Golay/XOR)** = `Golay70` | ❌ **mesuré et écarté le 07.** Le format tient — reconstruction exacte prouvée sur les 150,7 M blocs — mais **3,589 b/poids réels** (pas 2,92–3,05 : la fourchette de la note était trop optimiste) et **1,31× vs FP16**, sous le critère de 1,6× posé d'avance. Le double coset borne le noyau en ALU (195 Go/s effectifs) | [`mesures/e2-golay70-bench-2026-08-07.txt`](../mesures/e2-golay70-bench-2026-08-07.txt) |
> | **E1c (SoA bit-slicé)**, **E3 (unranking en noyau)** | ⬜ jamais tentés. Après le verdict E2, personne n'a rouvert l'échelle | — |
> | « **% de blocs violants inconnu — LA mesure manquante** » | ✅ **faite** : **8,7234 %** (13 144 531 blocs), branche **haute** de la fourchette. Et l'entropie de l'index vaut 46,6536 bits contre 47 payés, ce qui **clôt définitivement** le codage entropique du rang | [`verdicts-lot-b-2026-08-06.md`](verdicts-lot-b-2026-08-06.md) §B5 |
> | « le dispatch plafonne le bout-en-bout à ~1,28× » | 🔎 **mesuré, et c'est plus bas** : à tête identique le bout-en-bout rend **×1,12** (48,7 contre 43,6 tok/s). Le ×2,03 souvent cité inclut le remplacement du `lm_head` de candle, pas le noyau Leech | [`mesures/phases-2026-08-07.txt`](../mesures/phases-2026-08-07.txt) |
>
> **Et le mur que la note voulait renverser est tombé** : à embedding int8, le
> modèle entier pèse **5,15 b/param contre 5,30 pour l'AWQ 4 bits réel** —
> voir [`campagne-finale-2026-08-07.md`](../campagne-finale-2026-08-07.md).
> ⚠️ Ce renversement **ne tient pas à 8B** tel quel : les têtes n'y sont pas
> liées, l'embedding pèse 15,2 % du modèle, et il a fallu étendre le q8 aux
> têtes déliées pour repasser devant (5,323 contre 5,956) —
> [`tableau-8b-2026-08-07.md`](tableau-8b-2026-08-07.md).

> Note d'exploration issue d'une session de recherche (8 agents de lecture +
> 3 vérificateurs adversariaux sur l'arbre du 2026-08-05). **Rien n'avait été
> modifié dans le code, rien n'avait été mesuré sur carte** au moment de sa
> rédaction — tout ce qui suit était de la lecture vérifiée et de
> l'arithmétique recoupée. Le texte d'origine est conservé sous le bandeau,
> avec ses prédictions, parce que c'est ce qui permet de juger la méthode :
> E1a est tombé à 0,004 b/poids près, E2 s'est trompé de 0,6 b/poids.

## L'idée en clair

Le fichier à froid fait 2,07 b/poids et **on n'y touche pas**. Le problème est
au chargement : on « déplie » le fichier en VRAM dans le format `Slot32` pour
que le GPU lise vite, et ce dépliage gaspille — **5,51 b/poids** pour une
information qui en vaut 2. Résultat connu (`docs/archive/face-au-4-bits.md`) : le
format qui va vite ne rentre pas mieux que du 4 bits.

**La trouvaille : le gaspillage est dans la représentation, pas dans la
physique.** `Slot32` stocke l'appartenance de chaque coordonnée à son niveau
en *one-hot* — jusqu'à 4 masques de 24 bits ([llvq_slot.cuh:5]) — là où 2 à 3
bits par coordonnée suffisent (codage binaire de l'indice de niveau, en
« plans de bits »). Vérifié : **rien dans le décodeur ne consomme le
one-hot** (pas de popcount, signe dans un champ séparé). C'était la
dé-sérialisation la plus simple de l'archive, pas un choix d'optimalité.
Recoder en binaire est une bijection exacte : même codebook, même fichier,
même modèle au bit près.

### Lexique éclair

- **payload** : bits du format VRAM seuls ; **thesis** : la comptabilité du
  banc (payload + bases + queue f32 + échelles de ligne). Conversion dérivée
  et bouclée sur le 4B : `thesis = payload × 0,99534 + 0,1589`.
- **plafond L≤4** : interdire les blocs à 5 niveaux (3,38 % des blocs) en les
  remplaçant par leur meilleur représentant à 4 niveaux. Coût estimé
  −0,26 pt de rétention (gaussien, **jamais converti en ppl**).
- **overlay épars** : garder le plafond dans le flux principal + une petite
  table d'exceptions corrigée par une passe batchée → qualité **exacte**.

## L'échelle des barreaux

| barreau | mécanisme | payload | thesis (4B) | VRAM 70B¹ | qualité |
|---|---|---|---|---|---|
| Slot32 aujourd'hui | one-hot, strides 11/14/17 | 5,376 | 5,510 | ~50 Go | référence |
| plafond L≤4 (levier connu) | one-hot groupé | 4,708 | 4,843 | — | −0,26 pt est. |
| **E1a — AoS 14 o, 3 plans** | binaire, stride uniforme, **sans plafond** | **4,667** | **4,80** | 44,4 | **aucune perte** |
| E1b — AoS 12 o, 2 plans | binaire + plafond L≤4 | 4,000 | 4,14 | 38,7 | −0,26 pt |
| E1b′ — AoS 12 o + overlay | binaire + exceptions | ~4,23 | ~4,36 | 40,7 | **exacte** |
| E1c — SoA bit-slicé ×32 | plans transposés, zéro arrondi | 3,417 | 3,56 | 33,7 | −0,26 pt (ou +overlay) |
| E2 — étage GF(2)/Golay | codeword recalculé par XOR | 2,92–3,05 | ~3,06 | 29,4 | co-design à mesurer |
| E3 — unranking en noyau | l'index disque exécuté | ~2,3–2,5 | — | 24,3 | ⚠️ abandonne l'invariant |
| plancher | le fichier scellé | 2,07 | — | — | — |

> 📏 **Ce que la mesure a rendu, barreau par barreau** (ajouté le 2026-08-08 ;
> les colonnes ci-dessus sont les *prédictions* du 05, gardées telles quelles).
> Bancs L40S, rapports formés round par round :
>
> | barreau | payload prédit | **payload mesuré** | **vs FP16 mesuré** |
> |---|---|---|---|
> | Slot32 (référence de l'époque) | 5,376 | **5,510** (compta thesis) | 1,87× |
> | E1a → `Planes14` | 4,667 | **4,804** (thesis) | **2,14×** |
> | E1b′ → `Planes12x` | ~4,23 | **4,342** (thesis) | 1,98× |
> | E2 → `Golay70` | 2,92–3,05 | **3,589** | **1,31× — écarté** |
>
> Les payloads prédits et mesurés ne sont pas dans la même comptabilité (la
> note prédisait en « payload », le banc compte en « thesis » : + queue f32
> + échelles de ligne) : la conversion de la note, `thesis = payload × 0,99534
> + 0,1589`, rend `4,667 → 4,8042` pour E1a contre **4,804** mesurés, et
> `5,376 → 5,5098` pour Slot32 contre **5,510**. La conversion est exacte au
> millième. La prédiction **E2**, elle, était fausse de **~0,5 b/poids** (≈3,06
> annoncés en thesis contre 3,589 mesurés) : c'est la fourchette « ~2,9 » qui
> n'a pas survécu au compte réel des exceptions — **7,4357 %** des blocs sont
> violants ou L=5, et il faut les corriger dans le même lancement.
> Source : [`mesures/e2-golay70-bench-2026-08-07.txt`](../mesures/e2-golay70-bench-2026-08-07.txt).

¹ Convention canonique de `fiche-4b.md` : Llama-3.1-70B, 68,45 Md quantifiés
+ 2,10 Md d'embedding **f16**. Repères : **q4 = 39,7 Go** (lui quantifie
l'embedding) ; embedding int8 ≈ −2,1 Go partout ; **cache KV 8k = +2,7 Go** ;
seuil 24 Go honnête = **2,31 b/poids** (pas 2,74 — l'embedding f16 mange
l'écart). ⚠️ Les seuils « passe sous le 4 bits » et « sous 32 Go » avaient
été annoncés dans la convention naïve `70e9 × b/8` et **basculent** dans la
canonique : toujours chiffrer dans celle-ci.

## Les barreaux, en trois lignes chacun

**E1a (le barreau gratuit).** 3 plans binaires couvrent jusqu'à 8 niveaux, le
codebook plafonne à 5 → **aucun plafond nécessaire, zéro perte**. Égale octet
pour octet l'option (c′) de l'audit (4,667) mais sans son coût qualité, et
pèse moins que le plafond L≤4 « inconditionnel » (4,7083). Stride uniforme →
plus de tableau `bases`, fenêtre 4 mots, sh ∈ {0,16}. Décode : 2-3 AND +
3-4 SEL par coordonnée contre 4 AND + 4 SEL aujourd'hui — *moins*
d'instructions. Aucun doc du dépôt n'y avait pensé (grep vérifié).

**E1b / E1b′.** En plafonnant à 4 niveaux, 2 plans suffisent : 96 bits = 12
octets **alignés** (3 mots, shift nul — le terme « fenêtre désalignée » de
l'attribution disparaît). L'overlay rend la qualité exacte pour +0,22–0,24
b/poids (chiffre corrigé en vérification : l'adresse de bloc, 28 bits, avait
été oubliée). La passe de correction doit être **batchée** — pas 252
lancements de plus dans un système où le dispatch pèse 48 % du token.

**E1c (le risqué).** Empaqueter par groupe de 32 blocs sans arrondi à
l'octet : 3,417 exact. Mais ~+30-40 % d'ALU d'extraction, et l'ALU est le
**premier poste mesuré** — à tester après l'AoS, pas avant.

**E2 (la vraie recherche).** Λ₂₄ est bâti sur Golay, qui est *linéaire* : un
codeword se re-calcule par **XOR de 12 constantes** `Kᵢ = (0xC75≪i)|(1≪23)`
(vérifié exhaustivement contre `golay.rs` sur les 4096 messages). Coset
pair : le plan {|xᵢ| ≡ 2 mod 4} EST un codeword → 12 bits au lieu de 24.
Coset impair : les signes sont entièrement forcés (règle inversible,
`generic.rs:470`, `index.rs:33`) → le smask se recalcule. Compte honnête :
**~70 bits/bloc les deux cosets = 2,92** (une variante à 46 bits côté impair
a été proposée puis **réfutée** : le bit de codeword donne le résidu signé,
pas celui de |x|). Preuve que le gisement est réel : 2,917 à stride fixe
passe *sous* la borne rtbits « variable, adressage gratuit » (2,9243) —
l'information algébrique n'est pas comptée par nos comptabilités.
⚠️ Exige « ≤ 2 valeurs par résidu mod 4 » : 88/383 classes violent, **% de
blocs inconnu** — c'est LA mesure manquante.

**E3 (à étiqueter honnêtement).** Décoder l'index disque en noyau réintroduit
du séquentiel dépendant des données — exactement ce qui fait les 0,69× de
`Grouped32` (procès instruit : sa lenteur vient du décodage sériel, PAS du
stride variable — `Slot32` a le même adressage et fait 2,03× ; `Flat32`,
plus gros et plus rapide que Grouped32, le contre-prouve). Admissible
seulement avec un unranking branchless à profondeur bornée — ou en
**concevant l'index pour le décodeur** (l'espace entre E2 et E3 : rang Golay
stocké + rang intra-classe à radices choisies pour le shift, façon QTIP).

## Ce qui est vérifié vs ce qui reste une hypothèse

- **Vérifié** : tous les comptes de bits (deux fois, adversarialement), la
  bijection one-hot↔binaire, l'algèbre Golay (exhaustif), le procès
  Grouped32, les conversions de comptabilité.
- **Hypothèse** : TOUT ce qui parle de vitesse. L'attribution mesurée du
  2026-08-05 a renversé « l'ALU est hors de cause » (c'est le premier poste,
  59 % du gisement, compte niveau source faux d'un facteur 2). Règle du
  dépôt : pas de SASS, pas de conclusion. L'argument favorable d'E1 est que
  *tous* les termes baissent (octets, sélections, en-tête, bases,
  désalignement) — mais ça se tranche sur carte, pas sur papier.
- **Piège d'implémentation noté** : jamais `vals[idx]` avec index calculé
  (indexation dynamique → mémoire locale, [llvq_slot.cuh:139], et l'hôte
  rejette tout spill) ; arbre de sélections prédiquées obligatoire.

## Ordre d'expériences (coûts, et chacune tranche quelque chose)

| # | expérience | coût | tranche |
|---|---|---|---|
| 0 | histogramme **par classe** dans `rtbits` (une passe sur l'artefact 4B) | 0 $, Mac | la fourchette E2 (2,92–3,05 ou 3,2–3,4) |
| 1 | **Δppl du plafond L≤4** par swap transcodé sur le fichier scellé (synergie Job B de l'audit) | ~0 $, Mac+Metal | le −0,26 pt gaussien en vraie perplexité ; conditionne E1b et l'overlay |
| 2 | **E1a en 6ᵉ layout** au banc 7 bras (`LLVQ_KERNEL_DIR`, pas de rebuild ; le transcodeur et la vérification ligne à ligne survivent par construction) | ~0,1–0,2 $ | LA question : stride uniforme + plans binaires ≥ 0,95× Slot32 ? À qualité strictement identique |
| 3 | selon verdicts : E1b, overlay batché | ~0,3 $ | le point 4,0–4,2 |
| 4 | prototype E2 (transcodeur + noyau XOR) | chantier ~25-mutants, 5-10× E1 | le point ~2,9 |

**Métrique de succès** (le dispatch plafonne le bout-en-bout à ~1,28×, donc
les × du matvec ne sont pas la valeur) : *(Go VRAM à 70B, ≥ parité de
vitesse au banc, Δppl)*. Exemple : E1a accepté si ≥ 0,95× Slot32 à
−0,71 b/poids thesis.

## Repères de littérature (de mémoire, à vérifier avant de citer)

QuIP#/E8P (signes E8 reconstruits par calcul — l'analogue du XOR Golay
d'E2) · QTIP (décodage par calcul, trellis « bitshift » — le précédent d'E3
et l'étalon du coût de décode) · SpQR/SqueezeLLM (dense + épars — le
précédent mesuré de l'overlay) · Marlin (co-design layout/pipeline).
Le noyau Leech multi-coquilles fusé reste sans précédent — ça ne change pas.

## En une phrase

Le mur « 5,51 contre 4,5 » n'est pas un mur de physique mais de
représentation : un étage combinatoire stocké en one-hot (E1, gratuit,
−15 %), un étage linéaire stocké au lieu d'être calculé (E2, −40 %, à
co-designer), et un étage de rang dont le décodage reste la question ouverte
(E3) — les deux premiers suffisent à renverser le verdict produit face au
4 bits, le troisième est ce que « 70B sur 24 Go » exigeait depuis le début.
