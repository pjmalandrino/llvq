# Audit de performance du noyau CUDA fusé — 2026-08-05

> Audit statique mené sur l'arbre au commit `affe6ac` (branche `noyau-cuda`), par
> lecture du code et arithmétique sur les mesures existantes — **aucun job
> lancé, aucun fichier de code modifié**. Méthode : six dimensions instruites en
> parallèle (mémoire, occupation, lancement, ALU, format, protocole), chaque
> constat soumis à un vérificateur adversarial indépendant qui a refait
> l'arithmétique. Les corrections des vérificateurs sont intégrées au texte.
>
> ⚠️ **Toutes les attributions en millisecondes de ce document sont des
> MODÈLES** (arithmétique posée sur les agrégats mesurés du
> 2026-08-05), pas des mesures. C'est précisément le déficit que §5 corrige :
> l'instrumentation qui transformerait ces bornes en chiffres n'existait pas
> encore *à la date de rédaction*. Un modèle contredit par le premier job
> instrumenté doit céder.
>
> 🗓️ **BANDEAU D'ÉTAT — dernière revue le 2026-08-08. L'instrumentation existe
> depuis le 06-07, et deux modèles de ce document ont dû céder.**
> 1. **ε, le coût de lancement, est mesuré : 3,63 µs** — 252 × 3,63 = 0,915 ms,
>    soit 15,8 % du bras LLVQ. Trois mesures indépendantes concordent
>    (`bin/matvec` 1,85 µs de soumission CPU, `rotbench` 3,3 µs, `graphbench`
>    3,63 µs). Et **le CUDA Graph n'en récupère que 18 %** : le coût de
>    lancement n'est pas de la soumission, c'est la mise en route des blocs sur
>    les SM ([`mesures/a3-graph-2026-08-06.txt`](../mesures/a3-graph-2026-08-06.txt)).
> 2. **Le premier poste du token n'était dans aucune des six dimensions
>    auditées** : le `lm_head` de candle, **25,9 ms/token**, une copie
>    transposée de 778 Mo du vocabulaire à chaque pas. Les blocs transformer
>    entiers en pèsent 10,4. L'audit modélisait finement un poste qui n'était
>    pas le goulot ([`mesures/phases-2026-08-07.txt`](../mesures/phases-2026-08-07.txt)).
> 3. Et le gisement principal n'a pas été pris par des coupes ALU mais par un
>    **changement de représentation** : one-hot → plans de bits, `Planes14`,
>    −0,706 b/poids et 1,14× à contenu décodé identique
>    ([`mesures/c1-planesbench-2026-08-06.txt`](../mesures/c1-planesbench-2026-08-06.txt)).
>
> ⚠️ Pendant l'audit, un chantier parallèle a écrit `rotate.cu`,
> `llvq_rot.cuh`, `rotbench.rs` et modifié `gpu.rs`/`lib.rs` (fichiers non
> commités, 17h54–18h05). Les références de ligne de ce document valent pour
> `affe6ac` ; celles de `matvec.rs` et `gpu.rs` peuvent avoir bougé depuis.

## 1. Le fait central, et ce qu'il borne

Mesuré le 2026-08-05 sur le modèle publié
([`cuda-matvec-modele-2026-08-05.txt`](../mesures/cuda-matvec-modele-2026-08-05.txt)) :

| bras | min ms | Go lus | Go/s |
|---|---|---|---|
| FP16 (`tv_f16`, loads 128 bits) | 10,975 | 7,27 | **662** |
| LLVQ fusé (`tv_slot`) | 5,805 | 2,50 | **431** |

Le rapport publié est 1,89×. Mais à octets/poids égaux il vaudrait
16,000/5,510 = **2,90×** : le bras LLVQ lit ses 2,50 Go à 431 Go/s là où la
même carte, le même stream et le même protocole en servent 662 au bras FP16.
Le plancher DRAM du bras LLVQ est 2,50/0,662 = **3,78 ms** ; l'écart aux
5,805 ms mesurées — **~2,03 ms** — est le gisement de tout ce qui suit.

Deux corrections de plafond avant de rêver au 2,90× :

- **Les gaps de lancement le rendent inatteignable en l'état.** 252 noyaux
  séquentiels portent chacun un gap fixe g ; à g ∈ [1 ; 4] µs (la fourchette
  que le dossier refuse lui-même de trancher, `portage-noyau-cuda.md` §6.3),
  ε = 252g ∈ [0,25 ; 1,0] ms **sur chaque bras**. Même un `tv_slot` au débit
  du FP16 rendrait ~2,68×, pas 2,90×. Corollaire : le 1,89× publié est un
  **minorant mécanique** — corrigé des gaps il vaut 1,93× (g=1) à 2,14× (g=5).
- **Ne pas double-compter ε.** Le « 662 Go/s » du dénominateur contient déjà
  les gaps (wall-clock). En compte propre à g=5 µs : bande passante vraie
  ≈ 748 Go/s, et la part des gaps dans l'écart est ~0,82 ms (~40 %), pas 60 %.

## 2. L'attribution du gisement — des bornes, pas des parts

Les six dimensions rendent des **bornes qui se recouvrent** (le même temps de
stall peut être compté comme « conflit de bancs », « latence non masquée » ou
« sous-remplissage ») : elles ne s'additionnent pas, et c'est pour ça que
l'instrumentation de §5 passe avant tout réglage.

| candidat | borne modélisée | mécanisme | référence code |
|---|---|---|---|
| **Conflits de bancs 8 voies sur `xs`** | **0,3 – 2,9 ms** | lane L lit `xs[24L+j]` ; 24L mod 32 ∈ {0,8,16,24} → 4 bancs pour 32 lanes, conflit 8 voies sur chacune des 24 LDS. La borne dépend de ce que NVRTC a émis (LDS.32 → haut de fourchette ; LDS.128 → ~0,6 ms) — indécidable sans mesure | `llvq_slot.cuh:157-169`, `matvec.cu:98-100` |
| **Latence non masquée (MLP)** | recouvre le même temps | chaîne sérielle `bases[g]` → adresse → 5 mots → `tab[id]` : ~20 o en vol par warp (loi de Little : 431 Go/s × ~240 ns ≈ 0,7 Ko/SM, contre 1,1 Ko/SM au bras FP16), et la boucle `j` n'est pas déroulée — rien ne recouvre le bloc j+32 pendant le décodage du bloc j | `llvq_slot.cuh:70-76, 96-97, 110` ; `matvec.cu:98-100` |
| **Sous-remplissage de la grille** | **0 – 1,7 ms** | capacité 852 blocs résidents (142 SM × 6) ; k/v lancent 128 blocs (15 % d'occupation), o/down 320 (38 %), q 512 (60 %) — 50,6 % des octets passent par des grilles sous-remplies. L'agrégat 431 Go/s est compatible avec « uniforme ~471 partout » comme avec « gate/up à 694 et le reste à ~320 » : **indiscernable sans temps par forme** | grille : `gpu.rs:400-407` |
| **Gaps de lancement** | **0,25 – 1,0 ms** | 252 × g, partagé par les deux bras, 2× plus lourd en relatif sur le bras rapide (noyau moyen 23 µs contre 43,6) | `matvec.rs` boucle de rounds |
| Fenêtre payload (5 × u32 désalignés) | ~0,3 ms | 16,5 secteurs/instruction contre 4 idéal (§3.7 confirmé sur les secteurs) — mais le L1 absorbe les relectures : la DRAM ne voit que les octets utiles, le coût résiduel est l'émission | `llvq_slot.cuh:96-97` |
| Gather `tab[f.id]` | ~0,5 ms, dont 0,1–0,3 récupérable | 6 LDG dispersés par bloc ; la table **reste résidente** même avec le carve-out réel (72 Ko de shared occupés → ~28 Ko de L1, re-référence ~100× plus rapide que l'éviction) | `llvq_slot.cuh:110, 151-153` |
| `bases`, queue, barrières, remplissage | < 0,25 ms cumulés | clos par l'arithmétique, ne pas y dépenser un job | `llvq_slot.cuh:70-76`, `matvec.cu:104-108` |

**ALU : hors de cause aujourd'hui.** À 431 Go/s les pipes sont à ~1,7 % (FP32),
~32 % (INT/sélections, plancher 1,9 ms), ~19 % (émission, plancher 1,1 ms) —
compte niveau source à ±30-50 %, aucun SASS n'existant dans le dépôt. Aucun
pipe saturé : les micro-optimisations d'instructions (arbre de sélection,
`funnelshift`, LUT) valent **~0 ms observable** et ne redeviendront
pertinentes qu'une fois le noyau proche de 3,78 ms.

## 3. Ce que l'audit renverse dans le plan K7

L'ordre de balayage K7 (`portage-noyau-cuda.md` §5 : float4 → maxrregcount →
staging coopératif → CUDA Graph → tuile → padding) a été écrit **avant** le
fait 431 vs 662. Cinq révisions :

1. **Le staging coopératif §3.7(a) — l'option préférée du doc — a un gain
   prédit ≈ 0.** Relire les fenêtres désalignées depuis le shared paie ~4
   voies de conflits (5 LDS×4 + 5 LDG + 5 STS ≈ 30 cycles contre 22 en
   direct), pour +4,6 Ko de shared/bloc et l'adressage à cheval sur deux
   groupes à gérer. La « conclusion de conception » de §3.7 date d'avant le
   fait central. **Ne pas écrire (a) avant le verdict du bras sol** (§5).
2. **Le balayage `maxrregcount` est un no-op.** C'est un *plafond* : {48, 64,
   ∞} compilent le binaire actuel à l'identique (40 registres, déjà sans
   contrainte — `gpu.rs` ne pose que `arch`), et 32 force le spill que le
   contrat `local_bytes == 0` rejette au démarrage. À retirer du plan ; le
   levier réel est l'ILP **à la source** (point 4).
3. **Le padding de tuile remonte de dernier à premier — avec float4.**
   La réfutation Metal (K−1(b), 0,4 % plus lent) ne se transporte pas :
   l'arithmétique 32 bancs × 4 o est celle de NVIDIA, et c'est ici qu'elle
   prédit 8 voies. Précision du vérificateur : **pad 28 n'est propre qu'en
   accès float4** (en scalaire il laisse un conflit 4 voies résiduel ; le pad
   scalaire sans conflit serait 25). Occupancy : 128×28×4 = 14 336 o × 6
   blocs = 86 Ko ≤ 100 Ko/SM — tient.
4. **Le déroulage ×2 est absent de K7 et devient une expérience de tête.**
   Précharger la fenêtre (bases + 5 mots) du bloc j+32 pendant le décodage du
   bloc j double le trafic en vol par warp — exactement le déficit identifié.
   Coût : ~+7 registres (→ ~47, soit 5 blocs/SM sur les grilles pleines,
   −17 % de warps ; ×2 plein sur les grilles sous-remplies où rien ne limite).
   Si le noyau est borné latence, c'est le geste qui rapproche de 662 Go/s
   sans toucher ni au format ni au protocole.
5. **CUDA Graphs passe de « après le staging » à « condition de
   publication »**, parce que ε contamine chaque A/B futur : tout gain de
   noyau mesuré à travers 252 lancements est dilué par le terme constant
   252g, qui doit être connu **avant** le balayage. Piège concret trouvé dans
   le code : `Cuda::new` prend `ctx.default_stream()` — le **NULL stream
   legacy**, dont la capture est refusée par le driver
   (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). Une ligne à changer
   (`ctx.new_stream()`), mais elle change l'objet mesuré et doit être dite ;
   contrôle à trois bras : legacy / new_stream / new_stream+graph. cudarc
   0.19.8 expose l'API complète (`begin_capture`/`end_capture`/`launch`),
   ~40 lignes. À noter pour l'honnêteté du protocole : le 2,03× Metal vient
   d'**un** command buffer par passe — la transposition actuelle en 252
   `cuLaunchKernel` est structurellement défavorable au bras CUDA, et
   asymétriquement au bras rapide.

Et un levier **absent** de K7 :

6. **Fusionner q+k+v et gate+up.** Ils consomment la même activation et
   partagent d_in (donc nblocks et tail_w identiques) : concaténer par lignes
   donne 4 lancements/couche au lieu de 7 (252 → 144, −0,27 ms de gaps à
   g=2,5) **et** résorbe le sous-remplissage k/v (128 blocs → 768 fusionnés,
   15 % → 90 % d'occupation). C'est aussi la structure que `bin/run` exigera
   de toute façon (q/k/v indépendants entre eux ; o et down restent seuls).
   Coût : ~100 lignes hôte + `gscale` indexé par segment, **sur les deux
   bras** (règle « les deux bras ou aucun »).

Confirmés au passage, à **ne pas** re-dépenser : table pas en `__constant__`
(§3.1), pas de tensor cores en décode (§3.6), ne pas élargir la tuile (§3.5),
ne pas copier la table en shared (24,6 Ko/bloc → 4 blocs/SM, régression), pas
de split-K tant que la fusion n'a pas résorbé k/v, 512 threads/bloc rejeté
(k/v tomberaient à 64 blocs), hoisting de `bases` par `__shfl` rejeté
(≤ 1 secteur/warp déjà), réordre de `smask` clos (l'offset fixe est le point
du layout), table en f16 close (l'arrondi f16 vaut 24–49× le seuil 1e-5 de la
vérification ligne à ligne).

## 4. Le levier format — après le travail noyau, pas avant

- **Le plafond L ≤ 4 vaut au plus ~0,70 ms des ~2,03 ms** (−12,4 % d'octets :
  2,50 → 2,20 Go ; à 431 Go/s constants → 5,10 ms, rapport ~2,15×). Combiné à
  un noyau qui atteint 662 Go/s : 3,32 ms, soit **3,30×** (= 16,000/4,843,
  cohérent). La réserve est capitale : la réduction d'octets ne paie que si
  le limiteur est bien le pipeline mémoire — d'où l'ordre : noyau d'abord.
  Coût qualité (ordre de grandeur gaussien) : −0,26 pt de rétention
  (92,72 → 92,46 %), à vérifier en ppl sur le vrai modèle.
- **L'expérience qui tranche sans requantifier : échanger les 3,38 % de
  blocs à 5 niveaux.** Les 5,1 M de blocs L=5 fixent à eux seuls 66,7 % des
  groupes au stride 17 — l'excès pèse 301,6 Mo, exactement les 0,667 b/poids
  mesurés (recoupement clos). Or **tout l'outillage existe** :
  `BallSearcher::with_level_cap(4)` (`generic.rs:403`), et `transcode()`
  prend des indices arbitraires — le swap est une passe amont sur le
  `Vec<u64>`, le gain (qui code la norme) ne bouge pas, et la référence f64
  du banc étant construite sur les blocs *transcodés*, **la vérification
  ligne à ligne survit par construction**. Trois bras : publié / swappé /
  swappé + noyau sans m4 — sépare l'effet octets de l'effet ALU
  (~−50 instructions/bloc sans m4). Coût : ~5 min de swap sur le Mac
  (précalculé et monté), job ~0,3-0,5 $ s'il tourne dans le chargeur du banc
  (8 vCPU). Si le total mesuré < ~0,3 ms, le levier format recule.
- **L'option (c) uniforme 16 o est dominée.** Elle ne gagne que 0,8 %
  d'octets (−0,05 ms) et paie sa coalescence 0,625 b/poids que d'autres
  gestes obtiennent à 0 bit. Si L ≤ 4 est payé un jour, la forme gagnante est
  **(c′) : stride uniforme 14 o** — `byte = 14b`, sh ∈ {0,16}, fenêtre de
  **4 mots** (122 ≤ 128), plus de `bases` ni de chevauchement de groupes,
  **4,667 b/poids** (moins que le L ≤ 4 groupé) : 3,34× au plafond mémoire.
  À n'arbitrer qu'après le verdict noyau.

## 5. Les instruments manquants, et le plan de mesure recommandé

Le déficit n'est pas de validité mais d'**attribution** : l'agrégat de 252
matrices ne peut pas départager les candidats de §2. Trois instruments, tous
du code **hôte** — donc un **rebuild d'image obligatoire** (40-70 min, non
facturé ; `LLVQ_KERNEL_DIR` ne couvre que les `.cu`) — un seul rebuild couvre
tout :

1. **Le temps par forme** (~30 lignes : 2 `CudaEvent` par lancement, agrégés
   par forme, min/méd/max + Go/s par forme et par bras, dans des rounds
   *supplémentaires* pour ne pas toucher au protocole publié). Une lecture
   tranche : gate/up à ~620 Go/s et k/v à ~100 → occupation ; tout à ~430 →
   motif d'accès.
2. **Le noyau « sol »** (~40 lignes : mêmes `slot_byte` + 5 loads + `bases`,
   accumulation bidon anti-élision, même grille — `floor_probe` de
   `preflight.cu:90-100` en est l'embryon). Sol ≈ 3,8-4,1 ms → le motif de
   lecture est disculpé, le gisement est le décodage/la latence ; sol ≈
   5,6 ms → le flux Slot32 est le plafond et le format devient prioritaire.
   Donne au passage le pic mesuré sous ECC que le dossier réclame (§6.8
   rétracté).
3. **t_submit + horloges** (quelques lignes : temps CPU de la boucle de
   soumission seule ; `nvidia-smi -q -d PERFORMANCE,CLOCK`). Ferme
   l'hypothèse « GPU affamé » et fixe l'horloge que tous les modèles de §2
   supposent (2,2–2,5 GHz, jamais relevée).

Plus deux bras de validité dans le même rebuild : **cuBLAS/cuBLASLt** (libs
vérifiées présentes dans l'image runtime, inventaire du 2026-08-04) — si un
GEMM n=1 tire ~730 Go/s le titre s'érode à ~1,72×, et il vaut mieux le
mesurer soi-même qu'attendre un relecteur ; et le **A/B stream/graph** du §3.5.

**Séquence recommandée** (dans l'ordre, ~1-1,5 $ de jobs au total) :

| étape | contenu | coût | ce qu'elle décide |
|---|---|---|---|
| rebuild image | events par forme, sol, t_submit, horloges, new_stream+graph, bras cuBLAS, bras fusion qkv/gate-up | 0 $ (40-70 min, 2 tentatives à prévoir) | — |
| **Job A — attribution** | bras entrelacés : témoin · float4@24 · pad28+float4 · déroulage×2 · déroulage+float4 ; + sol + events par forme + legacy/new_stream/graph | ~0,5 $ | quel candidat de §2 porte les ~2 ms ; g mesuré ; l'ordre K7 réécrit sur mesure |
| **Job B — format** | Slot32 publié · swappé L≤4 · swappé+noyau sans m4 (artefact swappé précalculé sur le Mac) | ~0,3-0,5 $ | les −0,70 ms du plafond sont-ils réels ; part octets vs part ALU |
| ensuite | selon verdicts : fusion qkv/gate-up en titre, requantification L≤4 complète (3,45 h + ppl/MMLU), (c′), staging (a) seulement si le sol l'exige | — | — |

Grille de lecture du Job A : float4@24 ≥ +10 % → le modèle de bancs NVIDIA
tient, lire le padding dans le même job ; ~+3 % comme Metal → bancs réfutés
ici aussi, c'est la latence — le déroulage devient le chemin ; déroulage ×2
sans effet → chercher côté sectorisation DRAM.

## 6. Protocole : le 1,89× est solide, dans les limites qu'il se donne

Bras entrelacés dans chaque round, warmup réel (la vérification paie le JIT),
rapport formé round par round, dispersion intra-processus 0,05–0,19 % : rien
à retirer. À publier avec deux phrases obligatoires : « médiane des rapports
round par round » et « dispersion inter-jobs non encore mesurée — elle valait
2,5 % sur Metal ». Jamais de troisième décimale. Et dire que c'est un
minorant : corrigé des gaps, 1,93–2,14× selon g.

## 7. Bout-en-bout — là où l'audit débouche

Le banc mesure le noyau ; le produit est `bin/run`. Côté Metal, ~48 % du
temps par token est hors matmuls (~1000 lancements/token), ce qui plafonne le
fusé à ~1,28× bout en bout : **graph + fusions** (qkv, gate-up, rotation
fusionnée dans les producteurs d'activations plutôt qu'en +144
lancements/token) ne sont donc pas de la métrologie mais le chemin du
produit. Le noyau de rotation CUDA, lui, n'est plus à écrire : `rotate.cu` /
`llvq_rot.cuh` / `rotbench.rs` sont apparus dans l'arbre pendant cet audit
(chantier parallèle, non commité au moment où ces lignes sont écrites) — il
reste son premier run sur carte, puis sa fusion.
