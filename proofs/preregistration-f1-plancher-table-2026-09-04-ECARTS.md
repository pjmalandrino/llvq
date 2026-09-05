# Écarts au pré-enregistrement du plancher de table F1 — écrits après le job, jamais dans le tampon

> Le pré-enregistrement
> [`preregistration-f1-plancher-table-2026-09-04.md`](preregistration-f1-plancher-table-2026-09-04.md)
> est tamponné, et le journal
> [`docs/mesures/f1-plancher-table-2026-09-05.txt`](../docs/mesures/f1-plancher-table-2026-09-05.txt)
> est un fait brut : ni l'un ni l'autre ne s'édite. Ce qui les corrige s'écrit ici.

## É1 — Le point DRAM est à 4 Gio, pas 1 Gio, parce que la L2 fait 96 Mio

Le §4 fixe l'étalonnage DRAM à 1 Gio « au moins 20× la L2 », sur une L2 supposée
de 48 Mo. La carte en porte **96 Mio** (*mesuré* à l'attribut, deuxième
tentative, `6a9b4793`) ; 1 Gio n'était que 10,7× et la garde du banc a refusé de
démarrer, 0,00 $. Le point est à 4 Gio (42,7×), alloué à zéro sur la carte au
lieu d'être téléversé. Le contenu n'est jamais lu pour sa valeur.

## É2 — La géométrie est à 6 blocs par SM, pas 8, et la L1 fait 28 Ko

Le §4 écrit « contre 8 blocs/SM aujourd'hui ». `sm_89` plafonne à 1 536 threads
par SM : **6 blocs** de 256 (`llvq-cuda/src/occ.rs:435`, test
`residency_reproduces_the_l40s_figure`). Le banc imprimait `102 400 / 12 288 = 8`
sans la limite de threads ni la réserve de 1 Kio par bloc. Conséquence : 6 ×
(12 288 + 1 024) = 79 872 o de partagée → carveout 100 Ko → **L1 = 28 Ko**, dont
12 Kio pris par la tuile d'activation (*calculé*, constantes Ada hors dépôt,
politique du pilote *estimée*). C'est ce qui explique D(8 Kio) = 0,344 contre
D(16 Kio) = 0,663 : le point 16 Kio rate déjà 7,6 % de ses accès. **Le bas de
l'encadrement du §5 n'est pas un coût de hit L1 pur.** Une tuile de 16 blocs
donnerait ~112 Ko de L1 ; jamais mesuré, et le plancher nullk y changerait.

## É3 — Les bras mémoire partagée ne mesurent pas le placement, et la prédiction sur Dsm est fausse

Le §7 prédisait Dsm(48 Kio) entre 0,5 et 2 ms, « possiblement sous le budget ».
Mesuré : **5,097 ms**. La prédiction est fausse. Mais le bras ne mesure pas ce
qu'elle visait :

- l'ancre `smem-fill` recharge le jeu chaud **par bloc** : 138 240 blocs par
  passe × 24 / 48 Kio = **3,4 / 6,8 Go** de trafic de table (*calculé*), 3,5 à
  7× le flux de poids de F1 ;
- l'occupation tombe de 48 warps par SM à 16 puis 8 ; l'ancre absorbe la perte
  (4,930 et 7,474 ms contre 2,191 pour nullk) ;
- Dsm compare donc des consultations L2 à 8 ou 16 warps contre des consultations
  L2 à 48 warps : c'est la comparaison entre occupations que
  `docs/format-noyau.md` §6 interdit, déplacée d'un cran.

Ce que la mesure tue est **cette implémentation** (rechargement par bloc de 256
threads), pas le placement. Contre-exemple dans le dépôt : le bras QTIP du banc
F2 fait 1,82 milliard de lectures en mémoire partagée par passe (LUT de 64 Kio,
1 bloc de 1 024 threads par SM) en 2,246 ms
([f2-p3-qtip-banc](../docs/mesures/f2-p3-qtip-banc-2026-08-21.txt)). La phrase
« placer le jeu chaud aggrave », écrite le matin du 05, est **retirée**.

## É4 — Les comptes du §2 sont surestimés, sans effet sur le plateau

- 151 388 160 blocs et 454 164 480 consultations comptent la queue comme des
  blocs ; le banc fait **150 681 600 blocs → 452 044 800 consultations**
  (`nblocks = d_in / 24` par ligne), −0,47 %.
- 6 octets par entrée comptait le signe deux fois (`f1table.rs`,
  `(2·max+1).next_power_of_two().trailing_zeros() + 1`). Toute coordonnée d'une
  section a la parité `p` : stocker `(y − p)/2 ∈ [−5, 4]` tient en 4 bits, donc
  **4 octets par entrée pour toute orbite**, table **2 224 Kio** au lieu de
  3 336 (*calculé*, `examples/f1shrink.rs`). Le banc charge d'ailleurs un u32.
- Le rapport 15 pour 1 devient 14,5 Go / 0,98 Go = 14,8 pour 1. Le plateau est
  plat de 128 Kio à 16 Mio : ces corrections ne le bougent pas.

## É5 — L'encadrement du §5 est fermé par le calcul de la distribution réelle

Le §5 encadre D(réel) entre D(16 Kio) = 0,663 et D(4 Mio) = 4,517 ms faute de
connaître l'accès. Calculé le 05 à 0 $ (`llvq-bench/examples/f1accesscv.rs`,
27 376 blocs ; trois autres implémentations sans code commun, 800 à 8 000
blocs, d'accord à ±4 points) : les 16 Kio les plus chauds servent **17–21 %**
des consultations, 48 Kio 34 %, 128 Kio 44–48 %. Tenir H = 1,538 ms exigerait
71–77 % sous 16 Kio. Pour la table 67/9 : **D_réel ≈ 3,6–3,9 ms = 2,4–2,5 × H**
(*estimé*, mélange linéaire des points mesurés, pessimiste au-delà de 16 Kio).
Le bas de l'encadrement supposait 100 % de couverture ; il n'est pas atteignable
par cette table.

Ce qui l'atteint : une **table universelle de rangs**, 16 Kio pour les 528
régions, motifs calculés sur F₂ — mesurée à −0,6 pp de rétention contre F1
exact sur 2 000 puis 4 000 blocs (`examples/f1rankbench.rs`). Son coût de table
tombe entre D(8 Kio) et D(16 Kio) dans la géométrie d'aujourd'hui. L'inconnue
devient le décodage arithmétique, hors du périmètre de ce banc.

## É6 — « F1 est mort » a été dit sur ce plancher, et c'est retiré

Le §1 écrit que ce banc ne décide rien. Le matin du 05, le résultat a pourtant
été rapporté comme « F1 est mort sur le décodage ». C'est retiré. Règle
d'opérateur du 2026-09-05, inscrite à `docs/METHODE.md` §1 : **un kill
s'arbitre sur un critère fondamental, et par l'opérateur seul** ; un plancher,
un encadrement ou une projection l'informent et ne le prononcent pas. H est la
porte F1d (« pas plus lent que Planes14 au 4B »), un critère de débit à un point
produit où la VRAM ne contraint rien ; sur la classe de modèle, F1 est le seul
format sous le `b_max` du triplet.

## É7 — Didx est négatif et n'est pas expliqué

hash3 fait strictement plus que nullk et le bat de 0,103 ms [0,102 ; 0,104].
Candidats : génération de code (`__fmaf_rn(1.0f, x, acc)` replié en FADD, autre
ordonnancement de la boucle), ou effet de position (nullk est le premier bras de
chaque tour, juste après `smem-fill 48` à 1 bloc/SM). Indécidable sans le SASS ;
le banc n'imprime ni PTX ni SASS. Conséquence : D(S) est formé contre hash3, la
plus rapide des deux ancres, donc **pessimiste d'au plus 0,103 ms** ; ±0,1 ms
est la résolution de ce banc, et D(8 Kio) = 0,344 n'a que 3,3× cette marge.

## É8 — Le contrôle d'élision n'a été fait qu'au point 4 Mio

Le §6.1 exige que la sortie d'un bras de table diffère de celle de nullk. Le
banc ne le vérifie qu'au point 4 Mio (contenu aléatoire). Suffisant : une
charge dont le résultat nourrit `y` ne s'élide pas, et la même chaîne sert tous
les points ; mais le préreg disait « un bras de table », et un seul l'a été.
