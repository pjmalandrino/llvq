# Spec — Lot X : les pistes mémoire extrême (2026-08-12)

> 🗓️ **ADDENDUM du même jour — la cible MoE change les critères.** L'étude
> [`etude-moe-memoire-extreme-2026-08-12.md`](etude-moe-memoire-extreme-2026-08-12.md)
> établit que sur un MoE (actifs/totaux = 2-5 %), le trafic par token est
> proportionnel aux *actifs* pendant que la VRAM se paie sur les *totaux* :
> le critère 1,6× de ce lot est un critère **dense** et ne s'applique pas à
> cette cible. Conséquences : `Golay70` (3,589 ᵐ, exact) est admis de fait
> pour le MoE ; X2 reste le finaliste *dense* 70B/40 Go ; X4 (E3) devient
> l'enjeu unique de l'extrême ; et le gate le moins cher du lot devient
> **X5-MoE** (Qwen3-30B-A3B contre 32B dense, ~25-55 $ — le déficit 2 bits
> suit-il les totaux ou les actifs ?). Les critères X3 restent inchangés
> pour la cible dense.

> Spec d'exécution autonome : une session neuve doit pouvoir dérouler ce lot
> sans autre contexte. Objectif stratégique : **ouvrir la classe de machine
> que le q4 ne peut pas atteindre** — un 70B sur 40 Go (X1/X2), puis la cible
> 24-32 Go (X4). Garde-fous, identiques au lot A : **annoncer chaque coût GPU
> avant lancement et le cumul après ; go explicite de l'utilisateur avant
> tout job facturé ; une variable à la fois ; les critères d'admission sont
> posés dans cette spec, AVANT toute mesure, et ne se renégocient pas après.**
>
> Ce lot ne touche pas aux poids : tous les layouts X1/X2 sont des bijections
> du fichier scellé, la qualité est identique au bit près par construction.
> Le seul gate qualité est l'égalité des tokens de `fusedrun` (X3).

## Pourquoi ce lot (l'arithmétique qui le justifie)

Convention canonique 70B (`fiche-4b.md`) : 68,45 Md quantifiés + 2,10 Md
d'embedding ; `VRAM = (68,45·b + E)/8`, E = 16,8 en q8 ; KV 8k = +2,7 Go.
Formule contrôlée sur deux points mesurés : 4B → 2,57 calculé / **2,60
mesuré** ; 8B → 5,42 / **5,45**.

| point | b/poids (thesis) | 70B, q8 + KV 8k | classe de machine |
|---|---|---|---|
| `Planes14` (prod) | 4,804 | 45,9 Go | 48 Go — derrière le q4 |
| repère q4 (AWQ) | — | 42,4 Go | 48 Go |
| `Planes12x` (câblé) | 4,342 | 42,0 Go | 48 Go — parité |
| **X1 = E1c-14** | **4,56** ᵖ | 43,8 Go | 48 Go |
| **X2 = E1c-12** | **3,76** ᵖ | **37,0 Go** | **40 Go — hors de portée du q4** |
| X4 = E3 (~2,3-2,5) | ~2,4 ᵉ | 25,3 Go | 32 Go |
| plancher (le fichier) | 2,219 | 23,8 Go | 24 Go — zéro marge |

ᵖ prédit (comptes de bits exacts, conversion thesis vérifiée au millième sur
E1a) · ᵉ estimé (aucun design arrêté). ⚠️ Le seuil 24 Go exige 2,24 b/poids
avec le KV 8k : même E3 n'a **aucune marge** — la cible honnête du lot est
40 Go, puis 32 Go, pas 24.

## État de départ (vérifié le 2026-08-12)

- Cinq layouts runtime existent avec transcodeur + round-trips bit-exacts +
  banc multi-bras (`bin/planesbench`, rapports round par round, vérification
  des 1 105 920 lignes contre f64). Un 6ᵉ bras s'ajoute sans rebuild d'image
  (`LLVQ_KERNEL_DIR` pour les `.cu` ; le code hôte, lui, exige un rebuild).
- L'échelle des formats est **close** (`Planes14`/`Planes12x` gagnants,
  `Golay70` écarté à 1,31× sous le critère de 1,6×). Ce lot ne la rouvre pas :
  il teste le barreau **jamais mesuré** (E1c, `pistes-format-vram-2026-08-05.md`),
  celui qui n'a ni le double coset de Golay70 ni l'arrondi d'octet des AoS.
- Le contenu L ≤ 4 + exceptions est **déjà en production de banc** : la table
  d'exceptions de `Planes12x` (5 096 688 blocs, 3,3824 % — recensement
  `rtbits`) et sa passe de correction batchée sont prouvées exactes. X2 les
  réutilise telles quelles.
- Interdits hérités : réglages noyau gelés (bancs/padding, gather `tab`,
  table en shared, f16 table) ; pas de conclusion de vitesse sans mesure sur
  carte (un compte niveau source a déjà été faux d'un facteur 2) ; jamais
  `vals[idx]` à index calculé (mémoire locale — détecteur
  `local_size_bytes() == 0`).

## X0 — La spec de layout, sur papier (0 $)

Deux variantes, **une seule idée** : transposer les bits par-slot sur le
groupe de 32 blocs, qui coïncide avec le warp. Le mot k du groupe porte le
bit k des 32 blocs ; la lane i extrait `(mot >> i) & 1`.

**Le découpage qui évite le pire.** Réassembler la *classe* (9 bits) bit à
bit serait le poste le plus cher — donc l'en-tête reste en AoS empaqueté et
seuls les bits par-slot sont transposés :

```
groupe de 32 blocs = [ en-têtes : 32 × 10 bits = 10 mots, AoS packé ]
                     [ par-slot transposé : (smask, p0, p1[, p2]) × 24 slots ]

X1 (E1c-14)  3 plans, sans plafond   : 10 + 96 mots = 106 mots  → 4,4167 b/poids payload
X2 (E1c-12)  2 plans + overlay 12x   : 10 + 72 mots =  82 mots  → 3,4167 b/poids payload
```

Les deux tombent sur un **nombre entier de mots par groupe — zéro bit
d'arrondi, zéro table de bases** (adresse du groupe = `106·g` ou `82·g`
mots). C'est exactement le gain sur `Planes14`/`Planes12x` : mêmes contenus,
moins les bits de padding (6 par bloc pour 14, 14 pour 12x).

En comptabilité thesis (conversion `×0,99534 + 0,1589`, exacte au millième
sur E1a et Slot32) : X1 = **4,556** ; X2 = 3,560 + **0,202 d'overlay**
(dérivé : 4,342 − 4,140 sur Planes12x, même table, même passe) = **3,76**.

**Deux noyaux à bencher, pas un** — c'est la seule inconnue réelle :

- **K-U (chargements uniformes)** : la boucle de slots lit les mots
  transposés à adresse identique pour tout le warp (broadcast L1), 3-4
  extractions `shift+and` par slot, même arbre de sélection à 4/8 feuilles
  que `planes12_dot`/`planes_dot`. Coût : ~82-106 instructions de load par
  groupe là où Planes14 en fait 4-5 par lane — c'est l'origine du
  « +30-40 % d'ALU » estimé.
- **K-S (distribution par shuffle)** : chaque lane charge ~3 mots du groupe
  puis les redistribue par `__shfl_sync` — moins d'instructions de load,
  latence de shuffle en plus.

L'en-tête de la lane i est à l'offset `10·i` de la région AoS : deux mots au
plus (funnel shift), une fois par bloc. Queue de ligne : `nblocks` n'est pas
multiple de 32 — le transcodeur **padde le dernier groupe avec des blocs
origine** (classe 0, tout à zéro) et le noyau garde le même code.

**Livrable X0** : les deux structs, les offsets, le pseudo-code des deux
noyaux, et le compte d'octets — relu adversarialement avant d'écrire une
ligne. ⚠️ Le compte est un *majorant d'espoir*, pas une prédiction de
vitesse : règle du dépôt.

## X1/X2 — Transcodeur, référence CPU, tests (0 $, Mac)

1. **Transcodeur** : X2 se transcode **depuis les structures hôte de
   `Planes12x`** — la recherche `lswap` (le poste des 404 s) est déjà faite,
   le repack est trivial. X1 idem depuis `Planes14`. Ne pas refaire la
   recherche réseau.
2. **Référence CPU** écrite indépendamment (`decode_block` par groupe
   transposé), épinglée bit pour bit sur `Indexer::decode`.
3. **Tests, la liste minimale** — mêmes formes que les 5 layouts existants :
   round-trip bit-exact par bloc ; canonicité des octets (bits de signe des
   slots zéro nuls, padding du groupe de queue nul — un mutant doit
   l'exiger) ; largeur unifiée table↔assert ; sweep intégral du 4B scellé
   sous `#[ignore]` inconditionnel (échec franc si l'artefact manque, jamais
   `SKIP` vert). Mutants à tuer d'office : transposition ligne/colonne
   inversée (le classique), offset d'en-tête, funnel shift dégénéré, padding
   non nul, exception résolue sur le mauvais flux.
4. `cargo clippy` zéro warning ; `unsafe` nulle part (c'est du code de
   format, pas une frontière matérielle).

## X3 — Le banc, et les critères déjà posés (~0,2 $)

Bras 6 et 7 au `planesbench` (K-U et K-S pour X2 ; X1 seulement si X2 est
rouge, voir décision). Protocole inchangé : un seul processus, 7 rounds dont
2 jetés, tous les bras à chaque round dans le même ordre, médiane du rapport
round par round avec plage, comptabilité thesis identique, 1 105 920 lignes
vérifiées contre f64, préflight zéro spill.

**Critères d'admission — posés maintenant, avant toute mesure :**

| verdict | critère |
|---|---|
| X2 **admis** comme point de fonctionnement 70B | ≥ **1,6×** vs FP16 (le critère qui a écarté Golay70, réutilisé tel quel) |
| X2 **remplace `Planes12x`** dans l'échelle | ≥ **1,9×** vs FP16 (≤ 5 % de retrait sur les 1,98× de 12x, pour −0,58 b/poids) |
| X1 **remplace `Planes14`** en production | ≥ **2,05×** vs FP16 (il n'économise que 0,25 b/poids : il doit être quasi gratuit) |
| sous 1,6× | l'échelle se referme **définitivement** côté transposition ; la seule voie restante est X4 |

Câblage `LLVQ_FUSED_LAYOUT` + `fusedrun` A/B (tokens gloutons identiques au
bras dense) **seulement pour un layout admis** (~0,3 $). Millisecondes
brutes dans `docs/mesures/`, comme toujours.

## X4 — E3, l'étude sur papier d'abord (0 $)

Le 24-32 Go exige ~2,4 b/poids : décoder (presque) l'index du fichier dans
le noyau. Le rang v1 est disqualifié par la mesure (106× le sol, chaîne
sérielle data-dépendante, ~509 ops/bloc). La voie est le **co-design** :
concevoir un index *pour* le décodeur — rang Golay stocké (12 bits, table
16 Kio en L1, déjà prouvée par Golay70) + rang intra-classe à **radices
alignées sur des puissances de 2** pour que l'unranking soit des shifts à
profondeur bornée, façon QTIP — en payant l'arrondi de chaque radix en bits.

**Livrable X4 : une note de design, pas du code.** Elle doit contenir : le
compte de bits exact du sur-coût d'arrondi des radices (par classe, sur la
table des 384) ; le compte d'opérations du décodeur proposé, borné et
branchless ; et la comparaison aux 82 mots de X2. **Critère d'ouverture du
chantier code** : ≤ **2,6 b/poids thesis** projeté ET un décodeur à
profondeur fixe ≤ 24 étapes sans état sériel inter-slot. Sinon, E3 est
enterré avec le même soin que E2, et on le dit.

## X5 — Le gate stratégique : le silo se prouve à iso-VRAM

Le silo n'est pas « on rentre où le q4 ne rentre pas » — llama.cpp fait
tourner des 70B à ~2,3 bpw sur 24 Go depuis 2024, qualité dégradée. Le silo
est : **à budget VRAM égal, le gros modèle LLVQ bat-il le modèle moyen en
q4 ?** État mesuré aujourd'hui, et il est *contre nous* :
AWQ-4B = 70,04 % de MMLU dans ~2,7 Go, contre LLVQ-8B = 65,52 % dans
5,45 Go — le q4 deux fois plus petit gagne sur les deux axes. Ce qui porte
la thèse, c'est la **tendance** : notre déficit fond (−14,73 → −10,56 pp) et
celui du q4 croît (−0,28 → −3,07 pp) quand la taille double.

| étape | paire | coût | tranche |
|---|---|---|---|
| proxy | **LLVQ-32B contre AWQ-14B** (~20,5 contre ~9,8 Go) | ~62 $ (quant) + ~10 $ (éval) | la tendance tient-elle un doublement de plus ? |
| le vrai | LLVQ-70B contre q4-32B (les deux ≈ 21-25 Go) | ~230-280 $ | le silo lui-même |

**Ordre imposé : X5-proxy avant tout engagement 70B.** Prérequis 70B connus :
C3 (bf16) acquis, dérisquage 4 blocs (~10 $, pic `faer` à n = 28672), coût
par poids surlinéaire (4,77→6,36·10⁻⁵ cœur-s du 8B au 32B — la dernière
extrapolation était 25 % basse, prendre la marge).

## Pièges connus (payés cher — sources dans le dépôt)

- **La transposition est le terrain idéal du mutant équivalent** : un swap
  ligne/colonne peut être une involution sur certains motifs de test. Les
  sondes du round-trip doivent inclure des blocs asymétriques par
  construction (classe impaire, L = 5 via exception, signes non palindromes).
- Bit-order : LSB-first dans l'octet, octets croissants — le seul endroit du
  port où une erreur passe tous les tests CPU et ne casse que sur GPU.
- La dispersion inter-processus interdit de trancher quelques % entre deux
  invocations : K-U contre K-S se départagent **dans le même run**.
- Les b/poids se publient étiquetés (payload / thesis / rtbits) ; les
  comparaisons mémoire en **b/param modèle entier, embedding compris**.
- L'overlay se corrige **dans le même lancement** (CTAs en plus) — jamais
  252 lancements supplémentaires.
- `ppl` f32 / `fusedrun` f16 : tout A/B au même dtype.

## Fin de lot — passation obligatoire

Verdicts dans `docs/passation-lot-x-<date>.md` ; mise à jour de l'échelle
des formats dans `CLAUDE.md` §3 (elle passe de « close » à « close + X »
quel que soit le verdict — un rouge se consigne avec le même soin qu'un
vert) ; mesures brutes dans `docs/mesures/` ; et si X2 est admis, la
projection 70B de ce fichier se refait avec les Go/s **mesurés** à la place
des hypothèses.

## Hors périmètre (ne pas faire dans ce lot)

Toute requantification (le lot est à poids constants) · rouvrir Golay70 sans
idée neuve sur l'ALU · le chantier qualité (1 bit de gain, corpus, EoRA) —
c'est l'autre front, il a ses propres specs · tout réglage noyau gelé ·
lancer le 70B avant le verdict X5-proxy.
