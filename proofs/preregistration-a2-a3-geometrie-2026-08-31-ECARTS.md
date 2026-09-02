# Écarts au préreg A2/A3 géométrie (802006c5…) — volet A3, 2026-09-01

> Le préreg tamponné ne s'édite pas ; ce fichier, nommé par lui d'avance,
> reçoit ce qui s'en écarte. Les écarts d'A2 vivent dans
> `preregistration-a2-etapes2-3-graph-2026-09-01-ECARTS.md` (préreg de
> phase af6c12d2…) ; ceux-ci ne concernent que le §5 (A3). Tout est écrit
> **avant le premier job A3**, comme la note de design §4 l'exige pour É1.

## É1 — La lecture du « ≥ 10 % » du §5, fixée avant le job

Le préreg dit « battre `planes14` en géométrie FUSÉE de ≥ 10 % » sans
donner la formule. Lecture retenue — la conservatrice de la note de design
§4 :

- `gain = (t_ref − t_bras) / t_ref`, formé **round par round** (7 rounds
  dont 2 jetés, médiane et plage), avec `t_ref` = « Planes14, q+k+v et
  gate+up fusés » **re-mesuré dans le même processus** — jamais le 4,504 ms
  de F2 (autre processus, autre unité de traduction ; il cadre, il ne porte
  pas).
- Un bras **passe** le gate banc si **toute sa plage** est ≥ 10 % ; il est
  **point de courbe** si toute sa plage est < 10 % ; il est **non résolu** si
  la plage est à cheval — auquel cas une relance est permise **une fois**,
  dans le plafond de phase, sans changer le bras.
- Pourquoi Δ/t_ref et non un ratio ≥ 1,10 : sur 4,504 ms le premier exige
  ≤ 4,054 ms, le second ≤ 4,095 — Δ/t_ref est la lecture la plus exigeante,
  donc celle qu'un bras ne peut pas gagner par le choix de la formule.

## É2 — Les bras construits, contre ceux de la note de design

La note (§2) décrit trois bras : (a) multi-lignes par warp, (b) persistant
par site et global, (c) grille fixe + split-K intra-bloc. Ce qui est
construit (`llvq-cuda/kernels/planes_occ.cu`, sélecteur `LLVQ_SEG_ARMS`) :

| bras | statut | ce qui diffère |
|---|---|---|
| `mr2`, `mr4` | (a), tel quel | — |
| `mr2p` | **ajouté** | `mr2` sur le staging paddé : les deux mécanismes s'additionnent-ils ? coût nul, mêmes bits |
| `pad` | **ajouté, hors note** | même grille, même ordre, mêmes bits — seul le stride du staging passe de 24 à 28 flottants. Mécanisme *calculé, jamais mesuré* : à 96 o de stride, les chargements 128 bits d'un quart de warp partent des bancs 0, 24, 16, 8, 0… et les lanes 0 et 4 se heurtent (conflit 2-way) ; à 112 o ils partent de 0, 28, 24, …, 4 — 32 bancs distincts. K-1(b) avait mesuré « pas de conflit » sur **Metal**, dont le modèle de bancs n'est pas celui de NVIDIA (32 bancs × 4 o) ; la question n'a jamais été posée à la carte. |
| `pers` | (b1), tel quel | grille = résidence **lue** (registres du noyau chargé, limites de la carte) × SM |
| `persall` | (b2), tel quel | **banc seulement**, un lancement par round, sortie propre par site ; ne se porte pas (rot/attention/normes entre les sites) |
| `sk1`, `sk2` | (c) **transformé** | split-K **entre CTAs** — la grille est l'ensemble des paires (groupe de lignes, tranche de K) — avec un **fixup déterministe** par le dernier CTA (un ticket atomique par CTA, somme des partiels en ordre fixe, aucun `atomicAdd` flottant), au lieu du split intra-CTA à `g` lignes et tout K par CTA. Motif *calculé* : à `g` lignes par CTA avec tout K, le staging L2 → partagée de l'activation est multiplié par 8/g — **14,5 Go par token à g = 1** contre ~1,8 aujourd'hui, plus que les 2,18 Go de poids — quand le split entre CTAs le laisse **invariant** (chaque CTA stage sa seule tranche). `sk1` scinde aux frontières de tuile (o : 2, down : 4 → 640 et 1 280 CTAs), `sk2` aux demi-tuiles. Conséquence, prévue par la note : pas d'égalité bit-à-bit sur les sites scindés (l'association change), référence f64 au seuil 1e-5 du banc ; égalité bit-à-bit exigée sur les sites non scindés. |
| jumeaux `nullk` | **non construits** | recommandation (ii) de la note. Le plancher 1,794 ms borne `pad`/`mr`/`pers` approximativement et ne borne ni `sk` ni `persall` (F2). À construire en itération si un bras est à cheval sur le gate. |
| `LLVQ_TIME_EVENTS` par site | **non branché** sur la section Fusion | recommandation (iii) ; l'attribution par site attendra un bras vert |

Invariants tenus par les huit, vérifiés par le banc **avant tout chrono** :
`planes_dot` inchangé, mêmes octets lus, une ligne écrite une fois par un
thread ; **1 105 920 lignes identiques au bit près** à `tv_planes_seg` pour
tout bras à ordre d'accumulation inchangé, référence f64 sur les sites
scindés de `sk`.

## É3 — Les priors, déclarés avant le premier chiffre

| bras | prior (gain banc, Δ/t_ref) | d'où il vient |
|---|---|---|
| `pad` | 0 à +7 % | ≤ ~0,3 ms de cycles de partagée gaspillés sur 4,504 (*calculé* ; 0 si le compilateur émet déjà des chargements sans conflit) |
| `mr2` | −5 à +8 % | note §2(a) : ILP et staging amorti contre une grille ÷2 |
| `mr4` | −10 à +8 %, **spill possible** | idem, quatre `planes_dot` en vol par lane ; un spill arrête le job avant transcodage (≤ 0,01 $), on relance sans lui |
| `mr2p` | ≈ `pad` + `mr2` | si les mécanismes sont indépendants |
| `pers` | +5 à +12 % | note §2(b1) : rampe et vagues, pas le sous-remplissage |
| `sk1` | +5 à +15 % | note §2(c) : o/down passent de 37,6 % de remplissage à des vagues pleines ; borne des octets 28 % |
| `sk2` | `sk1` ± 3 % | vagues plus fines contre un fixup doublé |
| `persall` | +15 à +25 % | note §2(b2) : le pool par-lancement (~0,54 ms) en plus |

Prior global : **au plus un bras portable passe le gate au premier job**, et
l'adoption bout-en-bout (≥ 8 %) est improbable pour un bras seul —
arithmétique de la note §4 : 10 % au banc ≈ +4,5 % bout-en-bout, entre 3 et
8, point de courbe ; l'adoption demande ~18 % au banc ou la combinaison avec
A2 (adopté au 4B, +13,45 %). Le kill de phase (§6) se mesure de toute façon
en cumul intra-job.

## É5 — Premier job ROUGE INSTRUIT (2026-09-01, job `6a97394c…`, 236 s, 0,12 $) : cinq bras bit-exacts, une course dans le fixup de `sk`

Aucun chrono rendu — le job est mort à la justesse, avant la boucle de
mesure, exactement comme le préreg A2 l'exigeait de son gate rouge. Ce qu'il
établit quand même :

- **NVRTC accepte l'unité** (142 629 octets, sha256 `ef950895…`) et **les
  sept noyaux A3 chargent sans un octet de spill** — `mr4` à 64 registres,
  `mr2`/`mr2p`/`persall` à 48, `pad`/`pers`/`sk` à 40 (contre 40 pour
  `tv_planes_seg`). Le prior « spill possible » d'É3 sur `mr4` est réfuté.
- **`pad`, `mr2`, `mr4`, `mr2p`, `pers` sont IDENTIQUES AU BIT PRÈS à
  `tv_planes_seg` sur les 1 105 920 lignes** — pas une association déplacée
  par le compilateur, contrairement à la crainte qui avait motivé le repli
  f64 du commit `bb16d11`.
- **`sk1` est faux sur `layers.0.down_proj`** (pire erreur 2,25e-2·Σ|w·x|),
  après un `o_proj` juste (deux tranches). Mécanisme, instruit à la
  lecture : chaque warp stocke son partiel puis fait sa fence, mais le
  thread 0 tire le ticket **sans attendre les sept autres warps** — le
  motif CUDA « threadFenceReduction » a un seul écrivain par bloc, ce noyau
  en a huit. À quatre tranches et 1 280 CTAs, un dernier CTA a lu un partiel
  que son voisin n'avait pas encore écrit ; à deux tranches la fenêtre n'a
  pas mordu — chance de timing, pas correction.

Correctif, avant le second job : un `__syncthreads()` entre les fences et le
ticket (le kernel le consigne), et **un bras faux à la justesse est
INVALIDÉ sans tuer le job** — il n'est jamais chronométré, il sort en ROUGE
dans le bloc du gate, les autres bras gardent leur mesure. Ce qui reste
fatal : une erreur de lancement, ou une référence qui ne se calcule pas.
`sk2` et `persall` n'ont pas été atteints ; leurs priors d'É3 tiennent.

## É4 — Le job, tel qu'il sera lancé

- Image reconstruite (le binaire `planesbench` change ; les noyaux sont du
  texte, mais pas le banc). `LLVQ_BENCH_ARMS="slot32,planes14,fp16"` — la
  section Fusion exige `slot32` et `planes14`, `fp16` n'est pas
  désélectionnable ; la table à trois bras qui précède **n'est pas
  publiable** (unité NVRTC changée, sha256 imprimé) et ne l'est pas. Seule
  la section Fusion porte le verdict.
- `LLVQ_SEG_ARMS="pad,mr2,mr4,mr2p,pers,sk1,sk2,persall"` — les huit,
  appendés après les quatre bras historiques, dans les mêmes rounds, sur les
  mêmes tampons (aucun octet de flux résident en plus).
- L40S, comme le gate ; A100 en annexe seulement si un bras passe (~0,25 $).
- Coût : **~0,4–0,7 $** (*estimé* : F2 à 10 bras et 5 layouts = 0,89 $ pour
  30 min ; ici 2 layouts transcodés, ~15–20 min) sur un **plafond de phase de
  4 $**, dont **0,87 $ dépensés** avant ce job (A2). Cumul rendu après.
