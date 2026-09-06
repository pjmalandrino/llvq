# Écarts au pré-enregistrement du 4B en Tetra — écrits après le tampon, jamais dedans

> Le pré-enregistrement
> [`preregistration-tetra-4b-2026-09-06.md`](preregistration-tetra-4b-2026-09-06.md)
> (sha256 `a5b065b6…`) est tamponné. Ce qui le corrige s'écrit ici.

## É1 — La qualité se mesure sur L40S, pas sur Metal, et Tetra entre dans la forme d'A4

Le §3 du préreg fait tourner `bin/ppl` et `bin/mmlu` sur `metal`. **C'est faux comme protocole de
comparaison**, et l'opérateur l'a relevé dans l'heure qui a suivi le tampon.

Les repères servis viennent de la campagne A4 du 2026-08-06 : ppl et MMLU mesurés **sur une seule
carte L40S, un seul harnais, la même empreinte de tokens**, trois bras dans la même campagne (f16
12,2369 · AWQ 13,5207 · LLVQ 16,9422 ; MMLU 70,32 · 70,04 · 55,59), pour 0,71 $ (*mesuré*,
`docs/mesures/a4-campagne-2026-08-06.txt`). Mesurer Tetra sur Metal et le lire contre ces
nombres-là mêlerait le format et le harnais — la faute même que le contrôle 0 existe pour éviter
côté encodeur.

**Ce qui est fait à la place** : Tetra est un **quatrième bras de la forme A4**. Un job L40S, un
processus, une empreinte de tokens, avec le bras LLVQ `Planes14` rejoué à côté comme témoin. Le
§4.2 du préreg (« le fichier publié rejoue 16,9415 ») devient « le fichier publié rejoue 16,9422 »,
le nombre de la carte, et il reste ce qui sépare une dérive du harnais d'une différence de format.

Conséquences chiffrées, contre le §7 du préreg :

- l'évaluation quitte le Mac : plus de 2 à 4 h de MMLU par bras, ~0,71 $ et ~40 min de carte ;
- l'image HF doit être reconstruite : celle en service date du commit `7f2fb1f`, antérieur aux
  étapes 2, 3 et 4, donc elle ne sait pas lire un fichier v5 Tetra ;
- le fichier scellé, ~1,4 Go, monte dans le bucket avant le job.

**Ce qui ne change pas** : l'encodage reste sur le Mac, comme celui du fichier publié
(4,01 h, M3 Max, *mesuré*, `docs/fiche-4b.md` §3.4) ; l'encodage est déterministe et l'objet mesuré
est l'artefact, pas la machine qui l'a écrit. Le disque, les b/param et le coût d'encodage se lisent
sur le Mac : ce sont des propriétés du fichier et de la course, pas du harnais d'évaluation.

## É2 — L'oracle a tourné sur les deux backends du Mac, pas sur la carte

Règle dure 10, `bin/oracle` avant tout chiffre : fait sur Metal et sur CPU, `max |Δhidden| = 0,000e0`
des deux côtés sur le 4B (*mesuré*, 2026-09-06). C'est l'oracle du chemin d'**encodage**. L'oracle du
chemin d'évaluation est celui de la carte, et il fait partie du job.

## É3 — Le contrôle 0 échoue : l'encodeur ne reproduit plus le fichier publié

Le §3 bis fait dépendre la lecture du préreg d'un contrôle : un bloc transformer réencodé en
`leech1c12` aujourd'hui doit être identique octet pour octet au bloc 0 du fichier publié. **Il ne
l'est pas** (*mesuré*, 2026-09-06, `llvq-bench/examples/driftcheck.rs`) :

```
  matrice                 queue                        gains          indices
  q_proj    65 536 / 65 536 diffèrent   37 794 / 434 176   358 091 / 434 176
  k_proj    16 384 / 16 384             9 455 / 108 544     89 574 / 108 544
  v_proj    16 384 / 16 384             9 421 / 108 544     89 477 / 108 544
  o_proj    40 960 / 40 960            52 171 / 435 200    380 580 / 435 200
  gate      155 648 / 155 648         120 005 / 1 031 168   900 716 / 1 031 168
  up        155 648 / 155 648         122 200 / 1 031 168   899 517 / 1 031 168
  down       20 480 / 20 480          128 845 / 1 036 800   941 216 / 1 036 800
```

Ce qui est **identique** : dimensions, genre de code, cap de coquille, graine de rotation, centroïdes
de gain, échelles de ligne. Ce qui diffère : la queue, les gains, 87 % des indices — 3 659 171 sur 4 185 600, de 82,4 % sur `v_proj` à 90,8 % sur `down_proj`.

La forme de l'écart le situe. La queue est faite de poids conservés exacts **après** la compensation
GPTQ ; elle diffère de 42 à 56 % de la magnitude des poids en moyenne — ni du bruit numérique
(1e-15), ni deux tirages indépendants (le rapport vaudrait 1,41). Les échelles de ligne et les
centroïdes, invariants par rotation et calculés sur les poids seuls, sont intacts. **Ce sont donc les
hessiennes qui diffèrent**, pas la rotation ni le quantificateur.

Cause la plus probable, *non prouvée* : le commit `4a3e5f0` du 2026-08-26 a changé le volume de
calibration demandé, de `c4_calibration(8_000_000)` à `c4_calibration(n_calib × calib_len × 6)`, soit
786 432 caractères au lieu de 8 millions, et a remplacé un clamp silencieux par un `ensure!`. La
ligne de journal d'aujourd'hui porte « 81 available » là où les 8 M caractères en offraient 847.

**Portée du constat, au-delà de ce préreg** : le dépôt ne reproduit plus son propre artefact publié.
La perplexité 16,9422 et le MMLU 55,59 de la campagne A4 ne sont comparables à aucun fichier encodé
après le 2026-08-26, quel que soit son codebook.

**Décision d'opérateur, 2026-09-06** : on continue sans témoin réencodé. Le préreg §3 bis
prévoyait, si ce contrôle échouait, un arbitrage entre rejouer le témoin (4 h de Mac) et
caractériser la dérive ; l'opérateur a tranché pour ni l'un ni l'autre. Conséquence, à porter partout
où le chiffre de Tetra sera cité : **l'écart mesuré entre Tetra et le fichier publié contient le
format ET la dérive du dépôt sur un mois, et rien ne les sépare.** Le contrôle 2 du §4 reste en
place — le fichier publié est réévalué sur la carte dans le même job — mais il ne borne que la
dérive du harnais d'évaluation, pas celle de l'encodeur.


## É4 — `rtbits` ne lit pas un fichier v5, et le b/param est calculé à la main

Le §5 du préreg annonce « b/param modèle entier | `bin/rtbits` sur le scellé ». `rtbits` **refuse** un
fichier Tetra, et c'est le refus que l'étape 3 a posé exprès : il lit les index comme des classes v1,
et aucun layout runtime ne sert Tetra avant F1d.

Le chiffre est donc *calculé*, avec la comptabilité qui rend 5,162 pour le format servi
(`docs/mesures/rtbits-planes-8b-2026-08-09.txt` l. 350-352), et l'arithmétique est écrite ici pour
qu'elle soit refaisable :

```
  Planes14 : 3 633 315 840 × 4,803977  +  388 956 160 × 8,5  +  196 096 × 16
           = 20 763 630 625 bits  ÷ 4 022 468 096  =  5,1619 b/param   (2,595 Go)
  Tetra    : le même, moins 64 bits par bloc sur 150 681 600 blocs (112 → 48)
           = 11 120 008 225 bits  ÷ 4 022 468 096  =  2,7645 b/param   (1,390 Go)
```

⚠️ Ces gigaoctets sont ceux de l'arithmétique en b/param, pas le compte d'octets hôte du moteur que
`docs/ETAT.md` §2 donne à 2,57 Go pour le format servi. Les deux comptabilités diffèrent (queue en
f16, `gs_off`) et il ne faut pas les mélanger : la ligne Tetra se lit contre le 2,595 de la même
formule, jamais contre le 2,57 mesuré à l'hôte.

## É5 — Le contrôle 3 est vérifié à la quatrième décimale, pas à la septième

Le §4.3 demande le débit « au dix-millionième ». `smoke` n'imprime que quatre décimales, et sur un
autre dénominateur (2,1696 sur les poids quantifiés seuls, soit 2,159507 une fois ramené aux
3 633 315 840 poids de projection de `docs/fiche-4b.md`). Les deux artefacts diffèrent d'ailleurs de
1 040 octets. Ce qui est vérifié, et que le journal aurait dû écrire ainsi : **2,1595 b/poids des
deux côtés, à la quatrième décimale**.
