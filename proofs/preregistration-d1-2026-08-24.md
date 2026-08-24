# Pré-enregistrement D1 — la fusion des lancements sur le chemin servi (2026-08-24)

Écrit et tamponné **avant** le lancement. Item D1 du plan de révision, issu de
[`docs/revue-taco-2026-08-22.md`](../docs/revue-taco-2026-08-22.md) et des trois
relectures externes : c'est l'objection n°1 — le papier nomme le plus gros
levier (39 % de l'écart au plancher DRAM) et ne le tire pas sur le chemin
servi. Go de dépense de l'opérateur du 2026-08-24.

## 1. Ce que le lot change, et ce qu'il ne décide pas

q/k/v consomment la même activation, gate/up aussi. Concaténées **par lignes**,
elles font un matvec au lieu de trois et de deux : **252 → 144 lancements par
token**. Le noyau `tv_planes_seg_h` et toute la logique hôte sont écrits et
verts sur la machine de développement ; ce job mesure ce qu'ils valent
bout-en-bout, et **prouve leur correction** — aucune ligne de
`fused_cuda.rs`, de `SegPlan` ni du corps du noyau n'est vérifiable hors carte
(`llvq-llm --features cuda` ne compile sur aucune cible depuis ce Mac).

Ce lot **ne décide pas** de l'adoption : il produit une ligne de plus, mesurée,
que le papier publiera dans un sens ou dans l'autre.

## 2. Le job, verbatim

Image `hf.co/spaces/Pier-Jean/llvq-runner-cuda`, `l40sx1`, `--timeout 45m`,
modèle `Pier-Jean/Qwen3-4B-LLVQ-2bit` monté sur `/model`, bucket
`Pier-Jean/jobs-artifacts` sur `/out`.

```
LLVQ_ROT_SHARE=1 LLVQ_FUSE_AB=1 LLVQ_EMBED=q8 \
  fusedrun /model/qwen3-4b-llvq.bin 128 2>&1 | tee /out/d1-2026-08-24/d1-4b-fuse-ab.txt
```

`LLVQ_FUSE_AB=1` joue **trois bras dans un seul processus**, chacun chargé puis
**droppé** avant le suivant : fusé `LLVQ_FUSE=1`, fusé `LLVQ_FUSE=0`, puis le
bras dense. La carte ne porte jamais deux bras à la fois.

⚠️ `LLVQ_ROT_SHARE=1` est **obligatoire** et n'est pas un réglage libre : un
groupe fusé est **un** site, il tourne une fois par ligne quoi qu'en dise le
mode. `check_fuse` refuse la paire `FUSE=1` + `ROT_SHARE=0`, parce que
l'accepter ferait bouger le hissage de rotation **en même temps** que la fusion
et que le delta serait alors deux mécanismes additionnés. Les deux bras
`LLVQ_FUSE` de ce job partagent donc `ROT_SHARE=1` : **la seule chose qui bouge
est le nombre de lancements matvec.**

## 3. Ce qui se publie, posé d'avance

### 3.1 Le gate de correction — il prime sur toute mesure de vitesse

Ces trois conditions sont vérifiées **avant** de lire une milliseconde. Une
seule qui tombe suspend la publication du chiffre de vitesse et ouvre une
investigation ; rien ne se « corrige » en silence.

- **C1** — les bras `LLVQ_FUSE=1` et `LLVQ_FUSE=0` rendent **les mêmes 128
  tokens**. C'est la condition forte : à poids fixés, la fusion ne réassocie
  rien (mêmes blocs, même ordre, mêmes centroïdes), donc l'égalité doit être
  exacte, pas approchée.
- **C2** — les deux bras divergent du bras **dense** à la **même position**.
  Le journal du 2026-08-06 établit que `Planes14` et `slot32` divergent au
  même token 89 sur 128 ; c'est ce contrôle qui autorise à lire 89 comme un
  tie-break et non comme un défaut. Le critère de ce lot est le même,
  transposé.
- **C3** — au sein d'un bras, les tokens sont identiques entre rounds (le
  binaire l'imprime déjà).

### 3.2 Le compte de lancements — le lot a-t-il fait ce qu'il dit

- **L1** — la ligne de bras imprime **144** `matvec_lancements/token` sous
  `LLVQ_FUSE=1` et **252** sous `LLVQ_FUSE=0`, au 4B.
  Un « 128 tokens identiques » vert pendant que les deux bras émettent 252
  matvecs prouverait les tokens et rien du lot.

### 3.3 La mémoire — une prédiction arithmétique, pas une attente

- **M1** — `runtime_bytes(FUSE=1) − runtime_bytes(FUSE=0) = +3 686 400 octets
  EXACTEMENT` au 4B. Dérivation : `gs_off` est un `u32` par ligne fusée ;
  36 couches × (q 4096 + k 1024 + v 1024 + gate 9728 + up 9728 = 25 600
  lignes) = 921 600 lignes × 4 o. Rien d'autre ne bouge : la charge utile vaut
  `14·d_out·nblocks` quel que soit le groupage (stride uniforme, pas de bases)
  et `matrix_side_bytes` est additive en `d_out` à `tail_w` partagé.
  Soit **+0,008117 b/poids**. Une valeur différente est une anomalie, pas un
  arrondi.

### 3.4 La vitesse — la bande est posée AVANT, dans la bonne comptabilité

🚨 **Le 11,7 % du banc ne se transporte pas, et ce lot ne le republiera pas.**
Ce 11,7 % est mesuré en f32, sur `tv_planes_seg`, hors modèle, sur le temps
matvec seul (5,096 → 4,504 ms). Ce job mesure des **tok/s bout-en-bout en
f16**, où les matvecs ne sont qu'une part du temps par token. Publier le
premier à la place du second serait le motif que le §X3 du dossier a déjà payé
sur `E1c`.

La comptabilité de ce job, et le calcul de la bande :

- bras servi actuel au 4B : **87,0 tok/s = 11,49 ms/token** ;
- économie matvec mesurée au banc : **0,594 ms/token** ;
- si elle se transportait intégralement : 10,90 ms → **91,7 tok/s**, soit
  **×1,055**.

**Bande pré-enregistrée sur le rapport `FUSE=1 / FUSE=0` : [1,00 ; 1,12].**

- **sous 1,00** — la fusion **coûte** sur le chemin servi. Résultat publié tel
  quel, comme mesure négative : le papier porte déjà le vocabulaire pour ça.
- **au-dessus de 1,12** — dépasse ce que l'économie matvec du banc peut
  expliquer (0,594 ms sur 11,49 = 5,2 %). Investigation **avant** publication :
  un bras `FUSE=0` anormalement lent est le mécanisme le plus probable (cf. R5
  ci-dessous), et il gonflerait le gain.

**Forme du rapport** : les deux bras chargent leur modèle exclusivement et
leurs rounds ne coexistent jamais, donc c'est un **quotient de deux médianes**
avec son enveloppe conservatrice, étiqueté comme tel — la forme de B2, pas
celle des bancs à bras entrelacés. Aucune troisième décimale.

### 3.5 La référence, nommée d'avance

🚨 **L'ajout de `tv_planes_seg_h.cu` change l'unité de traduction NVRTC pour
TOUS les bras**, y compris `FUSE=0`. Un changement d'allocation de registres
qui ne *spille* pas est invisible. **La référence de ce lot est donc le bras
`LLVQ_FUSE=0` de CE job**, pas les 87,0 tok/s publiés — ceux-ci sont un
ancrage de comparaison. Si l'écart `FUSE=0` ↔ 87,0 dépasse ±3 %, il est
rapporté et expliqué avant tout verdict sur la fusion.

## 4. Coût annoncé avant lancement

`l40sx1` à 1,80 $/h. Trois chargements de modèle (~130 s par bras fusé, ~190 s
le dense) plus 3 × 6 générations de 128 tokens. **Attendu ≈ 0,60 $**, plafond
dur au timeout de 45 min : **1,35 $**. Cumul rapporté après, à
`docs/data/jobs.csv`.

## 5. Sorties

Journal : `docs/mesures/d1-fusion-servie-2026-08-24.txt` (sortie brute).
Registre : une ligne à `docs/data/jobs.csv`. Papier : §3.4 (la phrase « not yet
on the served path » disparaît ou se précise), Table 3 si C1-C2 passent et si
le rapport sort de la bande basse, et §6.
