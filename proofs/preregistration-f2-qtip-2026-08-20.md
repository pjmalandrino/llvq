# Pré-enregistrement F2 — le bras QTIP (2026-08-20)

Écrit, commité et **tamponné avant le premier job**. Item F2 du plan TACO
([`docs/plan-taco-2026-08-18.md`](../docs/plan-taco-2026-08-18.md)), plan
d'exécution [`docs/plan-f2-qtip-2026-08-20.md`](../docs/plan-f2-qtip-2026-08-20.md).
Go de dépense opérateur : à demander avant P2, coût annoncé au §6.

## 1. La question, et pourquoi elle n'est pas rhétorique

Ce dossier motive son propre travail par une phrase qu'il **n'a jamais
vérifiée** : le noyau LLVQ amont « est rapporté plus lent que celui de QTIP ».
C'est une citation, pas une mesure. Le papier l'admet et va plus loin — il
qualifie cette comparaison de *« the single most valuable missing
measurement »* — tout en établissant ailleurs qu'ajouter un bras à un run
entrelacé coûte des centimes. Un relecteur a fait de cette asymétrie le moteur
de son verdict, et il a raison de le faire.

F2 mesure donc QTIP **dans notre harnais, dans le même processus et les mêmes
rounds que nos bras**, exactement comme le bras AWQ l'a été.

⚠️ **Ce que F2 ne fait pas** : il ne mesure aucune qualité. Le payload est
pseudo-aléatoire (§3), donc les poids décodés ne sont les poids de personne.
Aucune phrase de qualité ne pourra s'appuyer sur ce bras, quel que soit son
résultat.

## 2. Ce que P0 a établi, et qui n'est donc plus en jeu

P0 (2026-08-20, 0 $, journal
[`f2-p0-recon-2026-08-20.txt`](../docs/mesures/f2-p0-recon-2026-08-20.txt)) a
résolu le format et écrit le codec. Ce qui suit est **acquis** et ne fait pas
partie des questions ouvertes de P2/P3 :

- le format (tuiles 16×16, registre à décalage circulaire de 512 bits,
  `h = ((s+1)·s)`, index `(h>>6)&0x1FF`, signe au bit 15 du premier élément),
  vérifié par **quatre implémentations indépendantes concordantes** ;
- **2,0000 b/poids** exactement, sans queue ni échelle de ligne ;
- les cinq formes du 4B passent tous les `static_assert` amont ;
- le typecheck Rust de la cible Linux (0 erreur, 0 warning clippy).

## 3. La comptabilité, posée d'avance

| | |
|---|---|
| octets facturés | `d_out · d_in / 4`, soit **2,0000 b/poids** |
| exclu | le codebook (512 half2 = 4 Kio), **constante résidente** — même règle que nos tables de classes, déjà appliquée à `Slot32` |
| **asymétrie déclarée** | QTIP ne porte **ni queue f32 ni échelle de ligne f32**, que tous nos bras facturent. Elle joue **en faveur de QTIP** et n'est pas corrigée — même traitement que l'asymétrie d'AWQ, déjà publiée |
| payload | **pseudo-aléatoire, graine dérivée de la forme** (déterministe : un run de contrôle doit restituer le même objet). Légitime parce que le code est à débit fixe — tout motif de bits est un mot de code valide — et suffisant pour le temps, le noyau n'ayant aucune branche dépendante des données |
| seuil de vérification | **`TOL` = 1e-5·Σ\|w·x\|, le nôtre**, et non le 1e-3 concédé à AWQ/cuBLAS : QTIP écrit du f32, pas du binary16, donc rien ne justifie de relâcher |
| géométrie | **la leur**, recopiée : `<<<128, 1024, 65536>>>`. La changer ferait de ce bras une mesure de notre réglage |

## 4. Les deux jobs, verbatim, et ce que chacun a le droit de conclure

Le plan du matin en prévoyait trois, dont un premier à `HAS_KERNEL` inchangé.
🕳️ **Cette découpe reposait sur une supposition fausse** : que la première
étape n'exigeait pas de reconstruction d'image. Elle en exige une — le script
de récupération est un *script*, et l'étage runtime ne copiait que des
binaires. Puisqu'il faut reconstruire de toute façon, la compilation et
l'exactitude tiennent dans un seul job.

**Prérequis, hors mesure** : image reconstruite depuis un commit ≥ celui qui
porte ce pré-enregistrement (elle embarque `fetch-qtip.sh` et les shims). La
construction est gratuite et n'est pas un job GPU.

### P2 — NVRTC accepte-t-il le texte, et le noyau calcule-t-il juste ? (≤ 0,30 $)

```
hf jobs run --flavor l40sx1 --timeout 25m -d \
  -v hf://Pier-Jean/Qwen3-4B-LLVQ-2bit:/model \
  -v hf://buckets/Pier-Jean/jobs-artifacts:/out \
  hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  -- bash -lc 'set -euo pipefail
mkdir -p /out/f2-qtip-2026-08-20
fetch-qtip.sh /scratch/qtip 2>&1 | tee /out/f2-qtip-2026-08-20/p2-fetch.txt
export LLVQ_QTIP_DIR=/scratch/qtip
LLVQ_BENCH_ARMS="fp16,qtip" \
  planesbench /model/qwen3-4b-llvq.bin 2>&1 | tee /out/f2-qtip-2026-08-20/p2-exactitude.txt'
```

Deux bras seulement : le témoin, obligatoire, et QTIP. **Ce job ne mesure
rien** — il établit trois choses et pas une de plus : que
`ops/fetch-qtip.sh` retrouve l'amont aux sha256 attendus, que NVRTC compile le
texte (le rapport de registres nomme les cinq shims et donne leur spill), et
que les 1 105 920 lignes passent le seuil `TOL`.

🚨 **Conclusion autorisée : « compile / ne compile pas », « exact / inexact »,
plus registres et spill. AUCUN temps ne se publie de ce job, y compris s'il en
imprime** — deux bras ne sont pas le protocole du banc, et une ligne chronométrée
là serait une mesure hors protocole.

### P3 — le banc (≤ 0,40 $)

Conditionné à P2 vert.

```
hf jobs run --flavor l40sx1 --timeout 40m -d \
  -v hf://Pier-Jean/Qwen3-4B-LLVQ-2bit:/model \
  -v hf://buckets/Pier-Jean/jobs-artifacts:/out \
  hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  -- bash -lc 'set -euo pipefail
mkdir -p /out/f2-qtip-2026-08-20
fetch-qtip.sh /scratch/qtip 2>&1 | tee /out/f2-qtip-2026-08-20/p3-fetch.txt
export LLVQ_QTIP_DIR=/scratch/qtip
LLVQ_BENCH_ARMS="slot32,planes14,planes12x,golay70v1,fp16,awq,golay70v2,cublasf16,nullk;slot32,planes14,planes12x,golay70v1,fp16,awq,golay70v2,cublasf16,nullk,qtip" \
  planesbench /model/qwen3-4b-llvq.bin 2>&1 | tee /out/f2-qtip-2026-08-20/p3-banc.txt'
```

**Deux phases dans un seul processus** : les bras publiés, puis les mêmes plus
`qtip`. C'est le protocole de l'ajout d'AWQ puis de `golay70v2`, à l'identique.
7 rounds, 2 jetés, ordre de dispatch = ordre d'enregistrement, rapports formés
**round par round** puis médiane + plage.

**Conclusion autorisée** : la ligne QTIP de la table, et la **fraction de sa
borne d'octets** — la seule grandeur comparable entre formats de tailles
différentes, comme la table publiée le fait déjà pour AWQ (88 %) et
`Planes14` (65 %).

### Ce que le mécanisme garantit, et qui a été corrigé pour ce pré-enregistrement

- `LLVQ_QTIP_DIR` est **une variable à part**, jamais `LLVQ_KERNEL_DIR`. Cette
  dernière signifie « surcharge TOUS les noyaux depuis ce répertoire » et
  échoue franchement sur tout fichier absent : la pointer sur la sortie de
  `fetch-qtip.sh` aurait cassé le banc entier. 🕳️ Le câblage la lisait, et
  c'est une revue d'exécutabilité — pas une mesure — qui l'a attrapé.
- `qtip` n'est **jamais** dans `ArmSet::runnable()`, donc un `planesbench` nu,
  sur n'importe quelle machine, dispatche exactement ce qu'il dispatchait avant
  que ce bras existe. Il ne devient sélectionnable **que nommé explicitement**
  (`FETCHED_AT_RUNTIME`), et le banc échoue alors bruyamment si le noyau
  n'est pas là.

## 5. La grille d'interprétation, écrite AVANT la mesure

Elle existe pour qu'aucune issue ne puisse être tournée après coup. Le nombre
qui décide est **la fraction de la borne d'octets**, pas le × contre FP16 : un
× récompense mécaniquement qui lit le moins, et QTIP lit **2,0 b/poids** contre
nos 4,80, donc un × brut le flatterait par construction.

**La grandeur qui décide est `Q` = la fraction de sa borne d'octets que QTIP
convertit en temps**, formée exactement comme celles déjà publiées : les Go/s
du bras divisés par les Go/s du témoin FP16 sur les mêmes formes, sur les
mêmes rounds gardés, dans le même processus. Ce n'est **pas** le × contre FP16,
qui récompense mécaniquement qui lit le moins — QTIP lit 2,0 b/poids contre nos
4,80, donc un × brut le flatterait par construction.

Les deux repères sont **mesurés et publiés** : nous à **65 %**, AWQ à **88 %**.
La grille est une **partition sans trou** de l'axe, bornée par eux — et non
trois seuils ronds séparés par des zones où l'on choisirait sa case après coup.
🕳️ La première version de ce paragraphe avait précisément ces trous (« ≳ 85 »,
« ≈ 60-70 », « ≲ 50 » : rien ne couvrait 78 % ni 55 %).

| cas, exhaustif et disjoint | ce qui se publie, et c'était écrit d'avance |
|---|---|
| **Q ≥ 88 %** (au niveau d'AWQ ou au-dessus) | La motivation héritée devient un fait mesuré **chez nous**. Notre 65 % est encadré par **deux** compétiteurs indépendants, ce qui déplace la charge : l'écart n'est plus « nous contre un noyau GEMM 4 bits » mais « nous contre l'état de l'art, toutes familles ». Le « ce qu'un successeur devrait attaquer » de la conclusion gagne son second point d'appui. |
| **65 % + δ < Q < 88 %** | QTIP est **entre** nous et AWQ. Se publie tel quel, sans récit : la position est le résultat, et l'écart QTIP↔AWQ devient une seconde quantité à expliquer, que nous n'expliquons pas. |
| **\|Q − 65 %\| ≤ δ** (indiscernable de nous) | Résultat **neuf** : deux décodeurs de familles différentes — treillis et réseau de Leech — convertissent la même fraction, et l'écart avec AWQ se relit comme *format-GEMM contre format-codebook* plutôt que comme un défaut du nôtre. ⚠️ « Indiscernable » se dit avec δ, jamais comme une égalité. |
| **Q < 65 % − δ** | La motivation héritée est **fausse dans notre harnais** : leur noyau y convertit moins que le nôtre. Se publie aussi, et sans triomphalisme — avec la réserve « tel que livré, réglage d'origine » du bloc ci-dessous. |
| **P2 rouge** (compilation ou exactitude) | Une phrase précise en limitations **nommant le blocage**, et la tentative journalisée. C'est la sortie que le relecteur déclare lui-même acceptable, et elle est meilleure que l'aveu actuel. |

**δ est défini avant la mesure** : la demi-plage min–max du bras QTIP sur ses
5 rounds gardés, additionnée à celle de `Planes14` dans le même run. Deux
fractions dont les plages se recouvrent ne se départagent pas — c'est la règle
n°2 du §7 de `CLAUDE.md` (une plage, pas un point), appliquée ici avant d'avoir
un nombre plutôt qu'après.

🚨 **Aucune de ces quatre issues n'annule F2.** Le bras est un point de mesure,
pas un candidat à l'admission : il n'a **pas** de seuil de rejet, et il ne
remplace rien. C'est la différence avec `Golay70`, dont le 1,6× puis le 2,0×
étaient des critères d'adoption.

⚠️ **Réserve permanente, valable pour les quatre issues** : nous mesurons QTIP
**tel que livré**. `BLOCK_COUNT = 128` et `BLOCK_SIZE = 1024` sont leur réglage,
figé pour le matériel de leur papier. Un tuning pourrait déplacer leur chiffre ;
nous n'en faisons aucun, dans aucun sens, et cette phrase se publie avec le
résultat quel qu'il soit.

## 6. Coûts, caps, et ce qui arrête tout

**Coût annoncé, total : ≤ 0,70 $** (P2 ≤ 0,30 + P3 ≤ 0,40), sur L40S au
tarif du registre — les caps durs sont les `--timeout` des commandes du §4. Cumul rapporté après chaque job.
Option A100 (le point QTIP dans `tab:a100`) : **+0,80 $**, décision **après**
P3, go séparé, hors de ce pré-enregistrement.

Arrêts durs, posés d'avance :
- P2 rouge à la compilation → **stop**, on écrit la phrase de limitations
  nommant le blocage, on ne relance pas ;
- P2 rouge à l'exactitude → **stop**. Une erreur d'exactitude signifie que
  notre lecture du format est fausse quelque part ; on ne chronomètre pas un
  noyau dont on ne sait pas ce qu'il calcule, et on ne « relâche pas le seuil
  pour voir » ;
- P3 : si les bras titulaires bougent de plus que leur demi-plage intra-run
  entre les deux phases, la phase 2 ne se publie pas — critère repris tel quel
  de l'ajout d'AWQ.

## 7. Anomalies déclarées avant la mesure

- **A1 — spill.** `local_bytes != 0` sur un shim QTIP se rapporte et
  s'interprète : leur noyau demande 1024 threads et 64 Kio de partagée, un
  spill serait un fait sur *leur* réglage à *notre* occupation, pas un défaut
  du nôtre.
- **A2 — opt-in 64 Kio refusé.** Se rapporte comme un fait de plateforme, au
  même titre que le `ERR_NVGPUCTRPERM` déjà publié.
- **A3 — dispersion.** Si la plage d'un round à l'autre dépasse 10 % de la
  médiane sur le bras QTIP, la ligne se publie **avec sa plage** et sans
  deuxième décimale.
- **A4 — l'amont a bougé.** `ops/fetch-qtip.sh` échoue si un sha256 diffère.
  Si cela arrive, **le job ne tourne pas** : on ne mesure pas un fichier qu'on
  n'a pas pré-enregistré.

## 8. Provenance et licence

Le noyau est **GPL v3** et n'est pas dans ce dépôt ; il est récupéré au commit
épinglé `e90c6688c8dfae326a3a81b5eb032db7c6680ec0`, sha256 vérifiés, patché de
quatre lignes mortes, et un `qtip_device.cuh` device-only en est extrait
([`docs/qtip-provenance.md`](../docs/qtip-provenance.md)).

⚠️ **Limite déclarée d'avance** : ce bras, seul de tous, **ne se rejoue pas hors
ligne** — il faut le réseau et l'amont vivant. Cette réserve se publie avec le
chiffre.
