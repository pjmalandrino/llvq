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
  `h = ((s+1)·s)`, index `(h>>6)&0x1FF`, signe au bit 15 du premier élément).
  ⚠️ **La force de cette vérification n'est pas uniforme et il ne faut pas la
  surdire** : la *fenêtre circulaire* est confrontée 200/200 à une
  transcription exécutée de leur `unpack_trellis`, et le *décodage d'un état*
  à une seconde lecture indépendante (leur Python et leur CUDA s'accordent) ;
  le *placement des tuiles* n'est établi que par lecture.

  🚨 **Et c'est LA question ouverte de F2, contestée par une relecture
  indépendante.** Le noyau lit son flux par un motif entrelacé par lane —
  `weight_idx = tileIdM·pas + warpId·256 + laneId·4`, puis `__shfl_sync` vers
  la lane suivante pour la fenêtre à cheval — et rien ne garantit par la
  lecture seule que les 32 u16 d'une tuile soient contigus, ni que l'état `s`
  porte les poids plats `2s` et `2s+1`. Un relecteur adverse dérive le
  contraire. Ce que nous savons vraiment : le déroulement du treillis DANS une
  tuile est vérifié 200/200, et le décodage d'un état l'est par deux lectures
  indépendantes ; **l'adressage, lui, ne l'est pas**. P2 le tranche — c'est sa
  raison d'être principale, et un échec de vérification y sera une réponse,
  pas un accident ;
- **2,0000 b/poids** exactement, sans queue ni échelle de ligne ;
- les cinq formes du 4B passent tous les `static_assert` amont ;
- le typecheck Rust de la cible Linux (0 erreur, 0 warning clippy) — ⚠️ obtenu
  au commit `34a2d58` via `ops/devtools/nvcc`, **postérieur** au journal P0
  cité ci-dessus, qui ne le contient pas. Et un typecheck n'est pas une
  compilation device : trois bugs d'exécution y ont survécu, corrigés depuis
  (activation binary16 non construite, tlut non arrondi, branche de dispatch
  manquante).

## 3. La comptabilité, posée d'avance

| | |
|---|---|
| octets facturés | `d_out · d_in / 4`, soit **2,0000 b/poids** |
| exclu | le codebook (512 half2 = **2 Kio** ; les 64 Kio de partagée sont cette même LUT répliquée 32 fois pour étaler les bancs), **constante résidente** — même règle que nos tables de classes, déjà appliquée à `Slot32` |
| **asymétrie déclarée** | QTIP ne porte **ni queue f32 ni échelle de ligne f32**, que tous nos bras facturent. Elle joue **en faveur de QTIP** et n'est pas corrigée — même traitement que l'asymétrie d'AWQ, déjà publiée |
| payload | **pseudo-aléatoire, graine dérivée de la forme** (déterministe : un run de contrôle doit restituer le même objet). Légitime parce que le code est à débit fixe — tout motif de bits est un mot de code valide — et suffisant pour le temps, le noyau n'ayant aucune branche dépendante des données |
| seuil de vérification | **`TOL` = 1e-5·Σ\|w·x\|, le nôtre**, et non le 1e-3 concédé à AWQ/cuBLAS : QTIP écrit du f32, pas du binary16, donc rien ne justifie de relâcher |
| géométrie | **la leur**, recopiée : `<<<128, 1024, 65536>>>`. La changer ferait de ce bras une mesure de notre réglage |
| **`f`, la fraction de la borne d'octets** | `f = Go/s(bras) ÷ Go/s(témoin FP16 du MÊME run)`, les deux formés sur la **médiane** des 5 rounds gardés, publiée à **0,1 point**. 🚨 Le binaire imprime `Go/s(min)`, qui est un **quotient de deux minima** — la forme que la règle n°2 du §7 de `CLAUDE.md` interdit. La valeur sur minima se journalise **à côté**, comme contrôle ; **si les deux versions classent le résultat dans deux cases différentes du §5, rien ne se publie** avant investigation. Le dénominateur est le témoin **FP16**, jamais `nullk`. 🕳️ La parenthèse « Go/s du bras / Go/s du plancher nullk » de `preregistration-f4-a100-2026-08-18.md:53` est **erronée** — `nullk` rend 18 Go/s pour 0,159 b/poids, ce qui donnerait des fractions de 2 300 % — et cet erratum est consigné ici plutôt que laissé à circuler |
| borne de QTIP | `16 / 2,0000` = **8,00× FP16**, la plus haute de la table (nous : 3,33×, AWQ : 3,83×) |

## 4. Les deux jobs, verbatim, et ce que chacun a le droit de conclure

Le plan du matin en prévoyait trois, dont un premier à `HAS_KERNEL` inchangé.
🕳️ **Cette découpe reposait sur une supposition fausse** : que la première
étape n'exigeait pas de reconstruction d'image. Elle en exige une — le script
de récupération est un *script*, et l'étage runtime ne copiait que des
binaires. Puisqu'il faut reconstruire de toute façon, la compilation et
l'exactitude tiennent dans un seul job.

**Prérequis, hors mesure — et il est plus contraignant qu'il n'en a l'air.**
L'image doit être reconstruite depuis **le commit qui porte ce
pré-enregistrement ou un descendant**, et rien de moins : la dernière image
citée par un journal est bâtie sur `1a585d0` (2026-08-18), alors que TOUT le
câblage QTIP lui est postérieur — le codec, les shims, la branche de dispatch,
`curl`/`python3` dans l'étage runtime, et `ops/fetch-qtip.sh` dans
`UPLOAD_ALLOW`. Une image antérieure ne contient **aucune** de ces pièces et
échouerait avant la première mesure. La construction est gratuite et n'est pas
un job GPU ; le commit exact sera consigné au journal de P2.

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

**Conclusions autorisées** : la ligne QTIP de la table ; **`r`**, la comparaison
directe de temps du §5bis, qui répond à la question du §1 ; et **`f`**, la
fraction de sa borne d'octets du §5ter. Les deux se publient ensemble, jamais
l'une sans l'autre — `f` ne porte aucune phrase de vitesse relative.

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

## 5. La fraction SATURE à 2 bits, et c'est ce qui décide de la métrique

🚨 **Deux versions de ce document ont bâti leur grille sur `f`, la fraction de
la borne d'octets, et les deux étaient fausses.** Le défaut est arithmétique,
il se démontre sans mesurer, et c'est exactement ce qu'un pré-enregistrement
doit attraper.

**Défaut 1 — `f` basse ≠ noyau lent.** `f` est relative à la borne de *chaque*
bras, et les bornes diffèrent : la nôtre vaut `16/4,804` = 3,33× FP16, celle de
QTIP `16/2,0000` = **8,00×**. La conversion est exacte, `× FP16 = 8·f`, donc :

| `f` | × FP16 | contre `Planes14` (2,15×) | contre AWQ (3,37×) |
|---|---|---|---|
| 50 % | 4,00× | ×1,86 | ×1,19 |
| **42,1 %** | 3,37× | ×1,57 | **parité** |
| **26,9 %** | 2,15× | **parité** | ×0,64 |

Une case « `f` basse → la motivation héritée est fausse » aurait donc publié
une réfutation pendant que la mesure disait le contraire.

**Défaut 2, et il est plus profond — `f` ne peut PAS atteindre 100 % à ce
taux.** Le plancher `nullk`, une passe de projections qui ne lit aucun poids,
est mesuré à **4,79× FP16**
([`f1-cublasf16-2026-08-18.txt`](../docs/mesures/f1-cublasf16-2026-08-18.txt)) :
aucun bras ne peut aller plus vite. Or la borne d'octets de QTIP vaut 8,00×.
Donc

> **`f(QTIP)` est plafonnée à 4,79 / 8,00 = 59,9 %**, structurellement, quel
> que soit le noyau — c'est-à-dire **sous notre propre 64,6 %**.

Deux cases de la grille précédente (`f ≥ 88 %` et `64,6 % < f < 88 %`) étaient
donc **inatteignables**, et la case « QTIP convertit moins que nous » était
**garantie d'avance**. Comparer `f(QTIP)` à notre 64,6 % revenait à truquer la
comparaison par construction.

🆕 **Le fait général, posé avant la mesure** : la fraction de la borne d'octets
n'est interprétable que tant que la borne reste sous le plancher, soit
`b > 16/4,79 = 3,34 b/poids`. Tous les bras du dossier sont au-dessus
(`Planes14` 4,804 · AWQ 4,179 · `Golay70` 3,589) ; **QTIP, à 2,0000, est le
premier à passer dessous.** C'est une limite de la métrique que le papier n'a
jamais eu à énoncer parce qu'aucun de ses bras ne l'atteignait, et elle se
publie avec ce bras — quel que soit son résultat.

**Conséquence sur la métrique : `r` (§5bis) décide, `f` (§5ter) documente.**

## 5bis. `r`, la comparaison directe — la grandeur qui décide

**`r = t(Planes14) ÷ t(QTIP)`**, formé **round par round** sur les rounds
gardés, médiane + plage. Licite parce que les deux bras tournent dans la même
phase du même processus, sur les mêmes formes, entrelacés.

C'est la seule grandeur qui répond à la phrase héritée du §1, et les trois
verdicts sont écrits d'avance. **Aucun n'est facultatif** :

- **plage entièrement au-dessus de 1** → *« le noyau QTIP porté est r× plus
  rapide que notre meilleur layout sur ces formes : la motivation héritée de
  l'introduction est CONFIRMÉE dans notre harnais »* ;
- **plage entièrement sous 1** → *« notre noyau est 1/r× plus rapide que le
  noyau QTIP tel que livré : la motivation héritée est RÉFUTÉE dans notre
  harnais »* ;
- **plage recouvrant 1** → *« les deux noyaux sont indiscernables sur ces
  formes ; la motivation héritée n'est ni confirmée ni réfutée »*.

🚨 **Aucune phrase de vitesse relative ne peut s'appuyer sur `f`.**

## 5ter. `f`, et ce qu'elle documente une fois `r` publié

`f` se publie **toujours avec son plafond** — « `f` = X %, sur un maximum
atteignable de 59,9 % à ce taux » — et jamais comme une comparaison nue avec
nos 64,6 %.

| cas | ce qui se publie |
|---|---|
| **`f` proche de 59,9 %** (à δ près du plafond) | QTIP convertit **tout ce que le plancher permet** : son noyau n'est pas ce qui le borne, c'est la latence de lancement que le §3.1 du papier attribue déjà. Le décodage en treillis est alors gratuit au sens du roofline. |
| **`f` nettement sous 59,9 %** | Il reste de la marge dans le décodage de QTIP, comme il en reste dans le nôtre. L'écart des deux marges est une seconde quantité, que nous ne prétendons pas expliquer. |
| **P2 rouge** (compilation ou exactitude) | Une phrase précise en limitations **nommant le blocage**, et la tentative journalisée. C'est la sortie que le relecteur déclare lui-même acceptable, et elle est meilleure que l'aveu actuel. |

**δ est défini avant la mesure** : la demi-plage min–max du bras QTIP sur ses
5 rounds gardés, additionnée à celle de `Planes14` dans le même run. Deux
grandeurs dont les plages se recouvrent ne se départagent pas — règle n°2 du §7
de `CLAUDE.md`, appliquée avant d'avoir un nombre plutôt qu'après.

🚨 **Aucune issue n'annule F2.** Le bras est un point de mesure, pas un candidat
à l'admission : il n'a pas de seuil de rejet et ne remplace rien. C'est la
différence avec `Golay70`, dont le 1,6× puis le 2,0× étaient des critères
d'adoption.

⚠️ **Réserve permanente** : nous mesurons QTIP **tel que livré**.
`BLOCK_COUNT = 128` et `BLOCK_SIZE = 1024` sont leur réglage, figé pour le
matériel de leur papier. Un tuning pourrait déplacer leur chiffre ; nous n'en
faisons aucun, dans aucun sens, et cette phrase se publie avec le résultat.

## 6. Coûts, caps, et ce qui arrête tout

🕳️ **Les chiffres annoncés par les deux versions précédentes (0,20 / 0,30 $)
n'étaient ancrés sur rien et sont ~3× sous le registre.** `docs/data/jobs.csv`
donne cinq runs de *ce binaire sur cet artefact* : `baseline-head` 26 min /
0,78 $ · `six-arm-awq` 26 min / 0,78 $ · `golay70-v2-sept-bras` 26 min /
0,77 $ · `llvq-nullk` 26 min / 0,77 $ · `f3-events-ncu` 29 min / 0,86 $. Le
poste dominant n'est pas les rounds mais **la construction des tampons**
(~1 460 s), qui ne dépend guère du nombre de bras.

| job | durée attendue | **coût annoncé** | ancrage |
|---|---|---|---|
| P2 (2 bras, construction complète) | ~25 min | **≤ 0,85 $** | les cinq runs ci-dessus |
| P3 (10 bras, deux phases) | ~30 min | **≤ 0,95 $** | `f3-events-ncu`, 9 bras, 0,86 $ |
| **total F2** | | **≤ 1,80 $** | |

Option A100 (le point QTIP dans `tab:a100`) : **≤ 1,20 $**, et non 0,80 —
⚠️ le tarif `a100-large` n'est **toujours pas mesuré**, sa seule entrée au
registre étant elle-même étiquetée *estimée* (1,00 $). Décision **après** P3,
go séparé, hors de ce pré-enregistrement.

Cumul rapporté après chaque job. Les caps durs sont les `--timeout` du §4.

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

## 6bis. Sorties, et où elles survivent

Sans bucket monté, rien de ce qu'un job écrit ne survit au conteneur, et la
rétention des logs HF n'est ni documentée ni garantie (§7 de `CLAUDE.md`,
payé deux fois). Donc, pour les deux jobs :

- montage `-v hf://buckets/Pier-Jean/jobs-artifacts:/out`, et **chaque sortie
  passe par `tee`** vers `/out/f2-qtip-2026-08-20/` ;
- journal final commité dans `docs/mesures/f2-qtip-<étape>-2026-08-20.txt`,
  **sortie brute comprise** — pas une synthèse ;
- une ligne par job dans `docs/data/jobs.csv` (identifiant, durée, coût
  facturé, étiquette *mesuré* ou *estimé*) ;
- le `PROVENANCE.txt` écrit par `fetch-qtip.sh` est recopié dans le journal :
  c'est lui qui atteste **quel** fichier amont a été mesuré.

## 7. Anomalies déclarées avant la mesure

- **A1 — spill.** `local_bytes != 0` sur un shim QTIP est **rapporté et le job
  continue** ; sur l'un de NOS noyaux il reste fatal. 🕳️ Le code faisait
  échouer les deux jusqu'au 2026-08-20 : un spill dans le noyau d'un
  concurrent porté tel que livré est un **fait** — leur réglage à notre
  occupation — et refuser de mesurer serait choisir de ne pas savoir.
- **A2 — opt-in 64 Kio.** ⚠️ **Ce n'est plus une inconnue** : le plafond
  d'opt-in de la L40S est **mesuré à 101 376 o** depuis le 2026-08-17
  (`fusedrun-14b-2026-08-17.txt`), et 65 536 < 101 376. `func_dynamic_shared`
  prendra donc la branche `OptIn`. Un refus resterait un fait de plateforme à
  rapporter, mais il serait une **surprise**, pas un risque anticipé.
- **A3 — dispersion.** Si la plage d'un round à l'autre dépasse 10 % de la
  médiane sur le bras QTIP, la ligne se publie **avec sa plage** et sans
  deuxième décimale.
- **A4 — l'amont a bougé.** `fetch-qtip.sh` échoue si un sha256 diffère.
  🕳️ La version précédente disait « le job ne tourne pas » : c'est faux, le
  script s'exécute **dans** le conteneur, donc le job démarre, échoue à la
  première minute et **facture cette minute** (~0,03 $). Ce qui est vrai est
  qu'aucune mesure n'en sort. La minute se porte quand même au registre.
- **A5 — répertoire déjà peuplé.** `fetch-qtip.sh` refuse d'écrire dans un
  répertoire non vide. Une reprise dans le même conteneur, ou un `/scratch/qtip`
  laissé par une étape antérieure, fait donc échouer le script — ni le réseau
  ni le sha256 en cause. Chaque job écrit dans un chemin neuf.
- **A6 — le réseau depuis un job.** ⚠️ **Hypothèse non vérifiée** : aucun job
  de ce dossier n'a jamais eu besoin d'atteindre l'Internet public. Si
  `raw.githubusercontent.com` est injoignable, P2 échoue à la première minute
  et le repli est de préparer le répertoire hors ligne et de le monter.

## 8. Provenance et licence

Le noyau est **GPL v3** et n'est pas dans ce dépôt ; il est récupéré au commit
épinglé `e90c6688c8dfae326a3a81b5eb032db7c6680ec0`, sha256 vérifiés, patché de
quatre lignes mortes, et un `qtip_device.cuh` device-only en est extrait
([`docs/qtip-provenance.md`](../docs/qtip-provenance.md)).

⚠️ **Limite déclarée d'avance** : ce bras, seul de tous, **ne se rejoue pas hors
ligne** — il faut le réseau et l'amont vivant. Cette réserve se publie avec le
chiffre.
