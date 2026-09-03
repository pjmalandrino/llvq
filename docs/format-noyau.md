# Format en VRAM et noyau fusé

Le format que le noyau lit en VRAM, le matvec fusé qui le lit, et les
pièges de mesure, dans leur état du 2026-09-02. L'état du projet est dans
[ETAT.md](ETAT.md), le fil des décisions dans [HISTORIQUE.md](HISTORIQUE.md),
les règles du labo dans [METHODE.md](METHODE.md). Tout « vs FP16 » est un
résultat L40S, sauf mention A100 (§7).

## 1. Le problème

Le fichier scellé code chaque bloc de 24 poids sur 48 bits : un index de
47 bits dans la boule Λ₂₄(12) et un bit de gain, soit 2,000 b/poids de code
(*calculé*, [fiche-4b.md](fiche-4b.md)). La boule compte N(12) = 111 043 117 458 000 points, soit 1,1·10¹⁴
(*calculé*, [fiche-4b.md](fiche-4b.md), verrouillé par
`classes_reproduce_theta_series` dans `llvq-core`).
Trois voies existent pour lire cet index dans un matvec.

| voie | ce qu'elle exige | verdict |
|---|---|---|
| table de correspondance | 1,1·10¹⁴ entrées ; un état de treillis 16 bits (QTIP) tient en 2 Kio, un codebook E8P (QuIP#) en 2¹⁶ entrées | impossible pour le réseau de Leech |
| dépliage au chargement | transcoder l'index en plans de bits lus par décalage et masque, 4,804 b/poids noyau | servi (`Planes14`, §3) |
| décodage arithmétique en ligne | rang, Golay, cosets (E3, `Golay70`, E1v) | fermé, chaque bras borné en calcul (§2) |

Le format servi déplie donc 2,000 bits d'information en 4,804 b/poids
(*mesuré*, [e2-golay70-bench-2026-08-07.txt](mesures/e2-golay70-bench-2026-08-07.txt)),
payés à la vitesse de la mémoire. Le dépliage est le prix d'entrée de la
famille, imposé par la taille du codebook (`paper/sections/layouts.tex`).

## 2. Les layouts

Trois comptabilités. *Payload* : bits du record par poids quantifié.
*Noyau* : payload, bases, queue f32 et échelles de ligne f32, telle que le
banc CUDA la facture. *Aligné warp* : noyau après bourrage de chaque ligne à
un multiple de 32 blocs, la seule forme que le matvec servi peut lire. Les
« vs FP16 » et Go/s sont ceux du banc dix bras du 2026-08-21, un seul
processus (*mesuré*, f2-p3-qtip-banc). Les verdicts de `Golay70` ont été
rendus sur les runs du 08-07 et du 08-11, à 1,31× [1,29–1,32] et 1,77×
[1,76–1,78] ; le banc du 08-21 les rejoue à 1,34× et 1,78×.

| layout | payload | noyau | aligné warp | vs FP16 L40S [plage] | Go/s | état | journal |
|---|---|---|---|---|---|---|---|
| `Slot32` | 5,3756 | 5,510 | s.o. | 1,89× [1,89–1,89] | 431 | câblé, non servi, repli `LLVQ_FUSED_LAYOUT=slot32` | [k1c-rtbits](mesures/k1c-rtbits-2026-08-05.txt), F2 |
| `Planes14` | 4,6667 | 4,804 | s.o. | 2,15× [2,15–2,16] | 428 | servi, défaut | [c1-planesbench](mesures/c1-planesbench-2026-08-06.txt), F2 |
| `Planes12x` | 4,2029 | 4,342 | s.o. | 2,00× [2,00–2,00] | 359 | câblé, mesuré servi une fois, non défaut | F2, [g-horloges](mesures/g-horloges-planes12x-2026-08-23.txt) |
| `Golay70` v1 | non publié | 3,589 | s.o. | 1,34× [1,34–1,34] | 199 | écarté le 08-07, critère 1,6× | [e2-golay70](mesures/e2-golay70-bench-2026-08-07.txt) |
| `Golay70` v2 | non publié | 3,589 | s.o. | 1,78× [1,77–1,78] | 264 | câblé `golay70`, non adopté, seuil 2,0× tamponné | [golay70-v2](mesures/golay70-v2-sept-bras-2026-08-11.txt), [préreg](../proofs/preregistration-2026-08-11.md) |
| `E1c14` | 4,4167 | 4,5551 (non aligné) | 5,2354 au 4B (+9,0 %) ; 4,6410 au 14B (−1,4 %) | jamais mesuré | s.o. | absent ; enterré au 4B | [x3-alignement](mesures/x3-alignement-warp-2026-08-15.txt), [rtbits-14b](mesures/rtbits-14b-2026-08-17.txt) |
| `E1c12` | 3,6196 | 3,7618 (non aligné) | 4,2880 au 4B (−1,3 %) ; 3,8021 au 14B (−10,4 %) | jamais mesuré | s.o. | absent ; ouvert, question de vitesse | [e1c12-aligne](mesures/e1c12-aligne-2026-08-16.txt), rtbits-14b |
| E1v | non publié | 2,3877 | 2,3983 (coupe alignée ligne) | 0,25× [0,25–0,25] | 25 | fermé pour le chemin servi, critère 1,60× | [e1v-cuda](mesures/e1v-cuda-2026-08-16.txt) |

Les payloads de `Slot32`, `Planes14` et `Planes12x` sortent du même balayage
du 4B scellé (*mesuré*, e1c12-aligne). Les bits d'`E1c` sont *calculés* sur
les blocs réels. La pénalité d'alignement warp vaut +15,47 % de blocs sur
les formes du 4B et +4,18 % sur celles du 14B (*calculé*, x3-alignement,
rtbits-14b). Les lignes du 14B sont plus longues, 213 et 725 blocs contre
106, 170 et 405. Un verdict d'alignement porte donc sa taille.

Quatre états : *absent* (aucun code ne le sélectionne), *câblé*
(`LLVQ_FUSED_LAYOUT` l'accepte, donc mesurable), *mesuré servi, non défaut*
(`Planes12x` depuis G3), *servi* (le défaut, d'où sortent les chiffres
publiés). `LLVQ_FUSED_LAYOUT` refuse toute autre valeur (`llvq-llm/src/fused.rs`).

E3, qui décodait l'index du fichier dans le noyau, est enterré sur papier :
3,0444 b/poids noyau contre un critère de 2,60 (*calculé*,
[radixstudy-x4-2026-08-12.txt](mesures/radixstudy-x4-2026-08-12.txt)). Les
seuils X3 d'`E1c` (2,05×, 1,9×, 1,6×) sont posés en comptabilité non alignée
et doivent être ré-ancrés avant tout banc, sinon le banc mesure un désalignement.

## 3. La disposition de Planes14

Un record `Planes14` fait 112 bits, soit exactement 14 octets, à l'offset
14·b du bloc b, LSB-first (`llvq-cuda/src/planes14_host.rs`) :

```
[classe : 9][gain : 1][smask : 24][plan0 : 24][plan1 : 24][plan2 : 24][0 : 6]
```

Le niveau du slot j vaut `plan0[j] | plan1[j] << 1 | plan2[j] << 2`. Il
indexe les valeurs canoniques de la classe, lues dans une table constante de
384 entrées. `smask` porte le signe du slot j, bit nul sur les slots à zéro.
Trois plans couvrent jusqu'à huit niveaux. Les blocs réels en portent trois
à cinq, 65,9 % en portent quatre, et moins de 0,1 % en portent un ou deux
(*mesuré*, [k1c-rtbits-2026-08-05.txt](mesures/k1c-rtbits-2026-08-05.txt)). Le stride
est uniforme, sans table de bases. Le payload vaut 112/24 = 4,6667 b/poids.

Une lane décode un bloc en 24 tours fixes : trois tests de bit pour le
niveau, un bit pour le signe, aucune divergence. Elle lit quatre à cinq mots
de 32 bits par bloc. `Slot32` lit un record de 9 + 1 + 24·L bits à stride
11, 14 ou 17 octets, dans une fenêtre de cinq mots. À contenu décodé
identique, `Planes14` va 1,14× [1,14–1,15] plus vite que `Slot32`, à Go/s
constants : le temps tombe comme les octets (*mesuré*,
[c1-planesbench-2026-08-06.txt](mesures/c1-planesbench-2026-08-06.txt)).

`Planes12x` ajoute un overlay épars. Les blocs à quatre niveaux ou moins
prennent un record de 12 octets à deux plans. Les 5 096 688 blocs à cinq
niveaux (3,3824 % des 150 681 600, *mesuré*, [HISTORIQUE.md](HISTORIQUE.md)
2026-08-09) gardent leur record `Planes14` de 14 octets dans une table
d'exceptions par matrice. Le même lancement ajoute la correction
(exact − approché)·x, memset compris. La reconstruction est bit-exacte. Le
plafond sec à quatre niveaux, sans overlay, coûte +4,75 % de perplexité
(*mesuré*, [verdicts-lot-b-2026-08-06.md](archive/verdicts-lot-b-2026-08-06.md)).
Le prix de l'overlay est un second flux irrégulier : la fraction de la borne
d'octets tombe de 65 % à 54 % (*calculé*,
[six-arm-awq-2026-08-10.txt](mesures/six-arm-awq-2026-08-10.txt)).

Chaque layout est vérifié contre une référence f64 sur les 1 105 920 lignes
du 4B, seuil 1e-5. Les pires erreurs valent 2,2e-8·Σ|w·x| pour `Slot32` et
`Planes14`, 2,9e-8 pour `Planes12x` (*mesuré*, e2-golay70-bench). La
bijection est prouvée sur les 150 681 600 blocs.

## 4. Le matvec fusé

Le noyau met un warp par ligne de sortie : staging de l'activation en
partagée, une lane par bloc, `warp_sum`, épilogue de la queue f32
(`TailPolicy::KeepExact`) et écriture de y. Chaque projection est précédée
d'une rotation d'incohérence `rot_apply`, noyau à un bloc, qui met toute
l'activation en partagée f32 : 8,05 µs à n = 2560 en isolation (*mesuré*,
[rotation-cuda-2026-08-05.txt](mesures/rotation-cuda-2026-08-05.txt)).
`fused_cuda` remplace les 252 `Linear` du 4B par ces deux lancements ;
`bin/fusedrun` l'appelle dans le modèle.

La configuration servie v1 fusionne `q+k+v` et `gate+up` par lignes
(`LLVQ_FUSE=1`) et hisse la rotation au groupe (`LLVQ_ROT_SHARE=1`). Les 252
matvec par token deviennent 144 au 4B, les 280 deviennent 160 au 14B.
`check_fuse` refuse `FUSE=1` avec `ROT_SHARE=0` : un groupe fusé est un seul
site de rotation, et un delta ne porte qu'un mécanisme.

| taille | tok/s v1 [plage] | Go carte | gain de la fusion à `ROT_SHARE` constant | surcoût exact | journal |
|---|---|---|---|---|---|
| 4B | 100,6 [99,9–100,7] | 2,57 | ×1,061 [1,050–1,069] | +3 686 400 o (+0,008117 b/poids) | [d1-fusion](mesures/d1-fusion-servie-2026-08-24.txt) |
| 8B | 75,5 [75,5–75,6] | 5,41 | ×1,055 [1,054–1,058] | +4 423 680 o | [vague2](mesures/vague2-fusion-8b-14b-2026-08-31.txt) |
| 14B | 46,8 [46,7–46,8] | 9,40 | ×1,028 [1,027–1,029] | +6 717 440 o | vague2 |

Tout est *mesuré*, bande [1,00 ; 1,12] tamponnée avant les jobs
([préreg D1](../proofs/preregistration-d1-2026-08-24.md),
[préreg vague 2](../proofs/preregistration-vague2-gel-geometrie-2026-08-31.md)).
Au 4B, la décomposition vaut 87,0 tok/s à `ROT_SHARE=0/FUSE=0` (*mesuré*,
[b2-fusedrun-plages](mesures/b2-fusedrun-plages-2026-08-18.txt)), puis 94,9
[94,1–95,2] avec le hissage seul (*mesuré*, d1-fusion), puis 100,6 ; le
×1,091 du hissage est une lecture inter-jobs et ne se publie pas. Les six critères de D1 sont verts :
128 tokens identiques entre bras fusé et non fusé, divergence au dense au
même token 89, même sha256 de source NVRTC. Le surcoût est l'index de
centroïde de gain (`gs_off`), un u32 par ligne fusée : 921 600 lignes,
+3 686 400 octets (*mesuré*, d1-fusion).

Au banc, la fusion rend 11,7 % du temps matvec seul sur `Planes14`, 5,096
contre 4,504 ms en f32 (`tv_planes_seg`, *mesuré*, d1-fusion). Ce chiffre ne
se transporte pas au 6,1 % par token du chemin servi : deux quantités. La
série à tête identique, la seule qui mesure le noyau, vaut ×1,11, ×1,29,
×1,41 du 4B au 14B (*mesuré*, b2-fusedrun-plages), à `ROT_SHARE=0/FUSE=0`.
Elle n'est pas rejouée sous v1.

La phase A borne le reste de la géométrie. Le bras de banc `persall`, non
portable, rend +26,36 % [+25,31 ; +26,61] sur le matvec fusé. Aucun bras
portable ne passe le gate de 10 %, et un split-K sur `o` et `down` rend
−1,87 % (*mesuré*, [a3-occupation-banc-2026-09-01.txt](mesures/a3-occupation-banc-2026-09-01.txt)).
Les CUDA Graphs (A2) rendent +13,45 % au 4B et ne sont pas servis, pour une
raison de mémoire ([ETAT.md](ETAT.md) §7).

## 5. Le transcodage au chargement

Le fichier scellé ne bouge pas. Le chargement déplie l'index vers le layout
demandé, une fois par processus.

| transcodage | durée | machine | journal |
|---|---|---|---|
| `Planes14`, 4B, 16 threads | 84 s | M3 Max | *mesuré*, [HISTORIQUE.md](HISTORIQUE.md) 2026-08-09 |
| `Planes12x`, 4B, 16 threads | 404 s (×4,8) | M3 Max | idem |
| `Planes14`, 4B, chargement `fusedrun` | 130,9 s | L40S | *mesuré*, [b2-fusedrun-plages](mesures/b2-fusedrun-plages-2026-08-18.txt) |
| `Planes12x`, 4B, chargement `fusedrun` | 1 340 s | L40S | *mesuré*, [g-horloges](mesures/g-horloges-planes12x-2026-08-23.txt), G3 |
| `Slot32` + `Planes14` avec preuve de bijection, banc | 150 s | L40S | *mesuré*, [c1-planesbench](mesures/c1-planesbench-2026-08-06.txt) |
| sept bras avec preuves bloc par bloc, banc | 1 464 s | L40S | *mesuré*, [golay70-v2](mesures/golay70-v2-sept-bras-2026-08-11.txt) |

`Planes12x` reste hors défaut : servi une fois, il rend 85,0 tok/s
[84,7–85,1] dans 2,36 Go, contre 87,0 dans 2,56 pour `Planes14` (*mesuré*,
g-horloges et b2-fusedrun-plages : −2,3 %, −0,20 Go, deux jobs). Il refait
une recherche réseau à cinq niveaux par bloc, d'où le facteur 4,8 puis les
1 340 s sur carte. Ce coût est payé à chaque chargement, sur carte louée :
c'est l'arbitrage. Au banc dix bras, `Planes12x` fait 0,93× [0,93–0,93] de
`Planes14` sur les seules projections ; le reste du modèle amortit l'écart.

## 6. Le plancher nullk

Une passe des 252 projections qui ne lit aucun octet de poids coûte 2,306 ms,
soit 45,2 % du bras servi et 4,77× FP16 [4,76–4,77] (*mesuré*, F2 ; 2,305 ms au
premier run, [nullk-plancher-2026-08-16.txt](mesures/nullk-plancher-2026-08-16.txt)).
`tv_nullk` garde la grille, le tuilage, les deux barrières, le staging de
l'activation, `warp_sum`, l'épilogue et l'écriture de y. Il retire la
lecture et le décodage du bloc (31 registres, 0 octet local).

| bras (même processus) | méd ms | Go lus | b/poids noyau | Go/s | vs FP16 [plage] |
|---|---|---|---|---|---|
| `nullk` | 2,306 | 0,07 | 0,159 | 31 | 4,77× [4,76–4,77] |
| QTIP 2 bits, concurrent | 2,246 [2,245–2,248] | 0,91 | 2,000 | 405 | 4,89× [4,89–4,90] |
| AWQ w4g128, concurrent | 3,252 | 1,90 | 4,179 | 584 | 3,38× [3,37–3,38] |
| `Planes14` | 5,103 | 2,18 | 4,804 | 428 | 2,15× [2,15–2,16] |
| FP16 cuBLAS | 10,830 | 7,27 | 16,000 | 672 | 1,02× [1,02–1,02] |
| FP16 témoin maison | 10,994 | 7,27 | 16,000 | 661 | 1,00× |

`nullk` mesure notre géométrie de lancement : un warp par ligne de sortie,
252 lancements. Dans cette géométrie, `Planes14` achète 3,11× net du plancher
(8,691 ms de trafic FP16 contre 2,797). Son décodage coûte environ 7 % du
temps de trafic, 779 Go/s nets contre 836 (*calculé*, nullk-plancher). Les
nets valent 836, 779, 710, 617 et 275 Go/s pour FP16, `Planes14`, `Slot32`,
`Planes12x` et `Golay70` v1. Le format se dispute au plus 55 % de ce temps.

`nullk` ne mesure pas la carte. QTIP, lancé dans sa propre géométrie
(`<<<128, 1024, 64 Kio>>>`, 252 lancements aussi), finit les mêmes
projections en 2,246 ms en lisant 0,91 Go. La séparation vaut 2,7 % contre
une résolution 2R = 0,72 % (*calculé*, F2). Sa fraction de borne d'octets vaut
61,1 % contre un plafond tamponné de 59,6 %
([préreg F2](../proofs/preregistration-f2-qtip-2026-08-20.md)) ; l'erratum
est dans le journal, le tampon n'est pas réédité. Le rapport
r = t(`Planes14`) ÷ t(QTIP) vaut 2,27× [2,27–2,28] pour 2,40× de trafic : à
efficacité proche (61 % et 65 % de la borne), le temps suit les octets. Entre
formats à bits différents, la grandeur comparable est les Go/s : AWQ 584,
`Planes14` 428, QTIP 405. Aucune phrase de qualité ne s'appuie sur le bras
QTIP, dont le payload est pseudo-aléatoire.

Soustraire `nullk` à un bras d'une autre grille est illicite : AWQ rendrait
2 006 Go/s nets, au-dessus de la HBM, et QTIP un net négatif (*calculé*,
f2-p3-qtip-banc). À 144
lancements le plancher tombe à 1,794 ms contre 2,200 à 252 dans le même
processus, r = 0,8158 [0,8150–0,8162], soit 3,76 µs par lancement
(*mesuré*, [a1-nullk-252-144-2026-08-31.txt](mesures/a1-nullk-252-144-2026-08-31.txt)).

L'attribution du 2026-08-05 a un autre dénominateur : les 2,04 ms de
gisement de `Slot32` par token au-dessus de son plancher DRAM (5,82 ms
contre 3,78). Le flux par la fenêtre de cinq mots pèse 0,681 ms (33 %), le
reste 1,199 ms (59 %) (*mesuré*,
[attribution-cuda-2026-08-05.txt](mesures/attribution-cuda-2026-08-05.txt)) ;
le partage de ce reste en latence-occupation 39 % (0,803 ms) et décodage
19 % (~0,396 ms) vient des trois noyaux « sol » du lot A
([rapport-lot-a-2026-08-06.md](archive/rapport-lot-a-2026-08-06.md)).
Le 45,2 % porte sur 252 projections, le 39 % sur un token : les rapprocher
exige de refaire l'attribution. Les 39 % sont une propriété de notre
géométrie de lancement.

## 7. Domaine de validité

Sur A100-SXM4-80GB, aucun bras à décodage ne bat FP16 (*mesuré*,
[f4-a100-2026-08-18.txt](mesures/f4-a100-2026-08-18.txt), même code, même
protocole, `LLVQ_NVRTC_ARCH=compute_80`).

| bras | méd ms A100 | vs FP16 A100 | Go/s A100 | Go/s L40S |
|---|---|---|---|---|
| `nullk` | 4,107 | 1,68× [1,68–1,68] | 18 | 31 |
| FP16 témoin | 6,915 | 1,00× | 1 052 | 661 |
| FP16 cuBLAS | 6,041 | 1,14× [1,14–1,15] | 1 204 | 672 |
| AWQ w4g128 | 3,793 | 1,82× [1,82–1,82] | 501 | 584 |
| `Planes14` | 8,742 | 0,79× [0,79–0,79] | 250 | 428 |
| `Slot32` | 9,413 | 0,73× [0,73–0,73] | 266 | 431 |
| `Planes12x` | 9,423 | 0,73× [0,73–0,73] | 209 | 359 |
| `Golay70` v2 | 11,121 | 0,62× [0,62–0,62] | 147 | 264 |
| `Golay70` v1 | 15,705 | 0,44× [0,44–0,44] | 104 | 199 |

Le FP16 convertit la HBM (661 → 1 052 Go/s) et nos bras chutent (428 → 250,
431 → 266) : sur A100 ils sont bornés par le calcul par SM. Le plancher
mange 59 % du temps FP16 sur A100 contre 21 % sur L40S (*calculé*, F4). Les
pires erreurs sont identiques sur les deux cartes. L'ordre interne de nos
layouts tient ; l'échelle contre FP16 s'inverse en bloc. Les × inter-cartes
ne se divisent pas : deux processus, deux témoins.

Le mécanisme est l'horloge. Les deux cartes tournent épinglées à leur boost
max, 2 520 MHz sur L40S et 1 410 sur A100, seul événement `GpuIdle`
(*mesuré* à 1 Hz, [g-horloges-planes12x-2026-08-23.txt](mesures/g-horloges-planes12x-2026-08-23.txt)).
Le rapport 1,787 tombe dans le critère tamponné [1,60 ; 1,95] et colle au
ralentissement de `nullk` : ×1,772 au banc G, ×1,781 au banc F4, ×1,809 sur
les temps d'A4 (*mesuré*, [a4-a100-2026-08-31.txt](mesures/a4-a100-2026-08-31.txt)).
Les compteurs d'occupation sont refusés par la
plateforme (`ERR_NVGPUCTRPERM`, [f3-events-2026-08-19.txt](mesures/f3-events-2026-08-19.txt)).

Le dénominateur FP16 est vérifié. Le témoin maison vaut 1,024 (deux bras)
et 1,015 (cinq bras) de cuBLAS sur L40S, critère ≤ 1,05 (*mesuré*,
[f1-cublasf16-2026-08-18.txt](mesures/f1-cublasf16-2026-08-18.txt)), et 1,02×
[1,02–1,02] au banc dix bras. Il est à 1,5 à 2,4 % de cuBLAS ; les rapports
« vs FP16 » ne flattent pas le numérateur. Sur A100 le même témoin est à
1,14× de cuBLAS. « Décode à la vitesse du matvec » est un résultat L40S/Ada
à domaine de validité mesuré sur deux cartes ; le point A100 le borne.

## 8. Le mur de la mémoire partagée

Le chemin fusé s'arrête au 32B, sur la rotation `rot_apply` (*mesuré*, [rot-partagee-14b-2026-08-17.txt](mesures/rot-partagee-14b-2026-08-17.txt),
[fusedrun-14b-2026-08-17.txt](mesures/fusedrun-14b-2026-08-17.txt)). Une
transformée de Walsh-Hadamard enchaîne log₂ m étages séparés par des
barrières, et CUDA n'a pas de barrière entre blocs. `rot_apply` est donc un
noyau à un bloc, qui met l'activation entière en partagée f32, quel que soit
le dtype d'entrée. La plus large activation est l'entrée de `down_proj`.

| modèle | `intermediate_size` | partagée demandée | défaut 49 152 o | opt-in 101 376 o |
|---|---|---|---|---|
| 4B | 9 728 | 38 912 o | passe | passe |
| 8B | 12 288 | 49 152 o, à l'octet | passe | passe |
| 14B | 17 408 | 69 632 o | échoue | passe |
| 32B | 25 600 | 102 400 o | échoue | échoue, de 1 024 o |

Les trois attributs de la L40S sont *mesurés* au préflight (fusedrun-14b) :
`MAX_SHARED_MEMORY_PER_BLOCK` 49 152 o, `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`
101 376 o, `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` 102 400 o (budget par SM,
pas par bloc). L'opt-in se pose par
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` sur la fonction avant tout
lancement. Le garde compare aux deux bornes et nomme celle qui est franchie
(`llvq_cuda::shared`, portable, testé sur Mac). Le premier job 14B a échoué
proprement sur l'ancien garde : exit 1 après 488 s, 0,24 $, aucun token. Le
32B reste refusé de 1 024 o, la réserve du driver ; un garde qui le
laisserait passer produirait une corruption silencieuse. Deux issues sont
nommées, aucune conçue : un staging en f16 (51 200 o, ~14,6 étages
d'accumulation en demi-précision) ou un découpage en deux noyaux.

Ce mur est identique sous tous les layouts et hors du plancher `nullk`,
qui chronomètre 252 projections rotation exclue.

## 9. Les pièges de mesure

- Un seul processus, tous les bras entrelacés, même ordre à chaque round,
  7 rounds dont 2 jetés. Deux `Planes14` de deux processus ne se comparent
  pas : autre unité de traduction NVRTC.
- Un rapport se forme round par round, jamais comme quotient de deux minima.
  Sur le run K-1, 21,728 / 10,126 rend 2,19 là où la médiane round par round
  rend 2,14× [2,10–2,16] (*mesuré*, [k1-metal-2026-08-05.txt](mesures/k1-metal-2026-08-05.txt)).
- Trois invocations du même binaire non modifié rendent 2,029×, 2,050×,
  2,080× à octets et erreurs identiques (*mesuré*,
  [thesis-temoin-2026-08-04.txt](mesures/thesis-temoin-2026-08-04.txt)). Un
  effet de quelques pour cent ne se tranche pas entre deux invocations. On
  publie une plage.
- `fusedrun` charge chaque bras seul : aucun round des deux bras ne coexiste.
  Le rapport est un quotient de médianes avec enveloppe, et les tokens
  gloutons se comparent au dense (divergence au token 89 sur les trois
  layouts servis).
- `float4` rend 3,5 % sur LLVQ et 5,1 % sur FP16, donc le rapport ne bouge
  pas (2,04× contre 2,09×). Le bras FP16 `float4` n'est pas bit-exact
  (3,1e-8, somme sans `fma` explicites), confondant déclaré (*mesuré*, K-1).
- Le conflit de bancs prédit par
  [portage-noyau-cuda.md](archive/portage-noyau-cuda.md) §3.2 n'existe pas
  sur Apple : le pas de 28 flottants rend 0,4 % plus lent que 24 (*mesuré*,
  K-1).
- Sur Metal, les buffers de 11 à 17 Mo tiennent dans le SLC de 48 Mo : le
  régime DRAM se force à froid, 4 copies de chaque flux en rotation. Un banc
  de décodage n'écrit jamais les poids décodés.
- Le transcodage hôte coûte 1 464 s avant le premier round d'un banc à sept
  bras (*mesuré*, golay70-v2), 1 468 à 1 481 s selon les jobs : lancer avec
  `--timeout 90m`. Le job 14B de B2 a été tué à 42,5 min pour 40 demandées,
  après sa dernière mesure (*mesuré*, b2-fusedrun-plages).
- `LLVQ_NVRTC_ARCH=compute_89` par défaut, `compute_80` pour l'A100 ; autre forme refusée.
- `LLVQ_TIME_EVENTS=1` chronomètre le span device par events CUDA, hors
  protocole publié. L'écart hôte−device vaut 0,1 à 0,2 %, 4 à 8 µs par round
  entier (*mesuré*, f3-events) : le poste latence est device, bulles
  inter-noyaux comprises.
- La ligne V0 du journal `nullk` imprime « pires erreurs 0.0e0 » : ce bras
  n'a aucun étalon, lire « non comparé ».
- Les « Go carte » sont un compte d'octets hôte imprimé par `fusedrun`,
  jamais `nvidia-smi`. Les ÷ sont licites ; les valeurs absolues ne se
  comparent pas à un affichage carte (2,60 Go affichés contre 2,56 comptés).

## 10. Note de provenance

Un même objet porte plusieurs chiffres, un par comptabilité. Ils ne se
comparent jamais entre eux.

| comptabilité | ce qu'elle compte | exemple |
|---|---|---|
| fichier, `bin/seal` | bits de payload / tous les poids, queue comprise | 4B : 2,1595 b/poids |
| fichier, `bin/smoke` | mêmes bits / poids quantifiés seuls | 4B : 2,1696 |
| taux idéal, `smoke` | 48 bits par bloc, queue à 16 bits, un fichier jamais écrit | 4B : 2,0702, ne se cite pas pour ce fichier |
| payload | bits du record / poids quantifiés | `Slot32` 5,3756 · `Planes14` 4,6667 · `Planes12x` 4,2029 |
| `rtbits`, `bin/matvec` | payload + une base u32 par groupe, stride à l'octet | `Slot32` 5,3756 (modèle entier) et 5,375 (gate_proj seule) |
| noyau, `bin/thesis` et banc CUDA | idem + queue f32 + échelles de ligne f32 | `Slot32` 5,510 · `Planes14` 4,804 · `Planes12x` 4,342 |
| inférence, `fusedrun` | idem, queue portée en binaire16 | `Planes14` 4,729 · `Planes12x` 4,277 |
| modèle entier | octets carte / tous les paramètres, embedding compris | `Planes14` + q8 : 5,162 (4B), 5,322 (8B), 5,106 (14B) |

Sources : fiche-4b pour le fichier, k1c-rtbits et e1c12-aligne pour payload
et `rtbits`, F2 pour le noyau, b2-fusedrun-plages et g-horloges pour
l'inférence, rtbits-14b pour le modèle entier.

Un « 4,804 » et un « 4,729 » côte à côte sont deux numérateurs. Une échelle
bits↔vitesse aligne les bits et les vitesses d'une seule comptabilité et
d'un seul run. Toute comparaison mémoire avec un concurrent se dit en
b/param modèle entier, embedding compris ([METHODE.md](METHODE.md) §2).
