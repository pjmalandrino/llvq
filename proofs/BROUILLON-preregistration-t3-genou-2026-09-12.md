# BROUILLON — palier 0 de T3 : où est le genou de l'attention en int4 ?

**BROUILLON, NON TAMPONNÉ. Il n'autorise rien et ne gate rien.** Il attend une
décision d'opérateur que la mesure du 2026-09-12 a rendue nécessaire : le crible
gratuit ne peut pas résoudre son propre effet (§0), et la variante qui le résout
coûte 2,80 $. Le go du 2026-09-12 (« Go test ») portait sur un palier à 0,00 $ ;
il ne couvre pas cette dépense.

Une fois la variante choisie, ce fichier est figé, tamponné (`ots stamp`) et
renommé `preregistration-t3-genou-<date>.md`. **Aucun bras de traitement avant
le tampon** (règle dure 2).

Code mesuré : arbre de travail à `7d62cff`, aucune modification de `llvq-llm`
ni de `llvq-artifact` depuis le scellement du fichier Tetra du 2026-09-06.

---

## §0 — Ce que ce palier peut être, et ce qu'il ne peut pas

**Mesuré le 2026-09-12, à 0,00 $, avant toute écriture de ce brouillon**
([journal](../docs/mesures/t3-genou-2026-09-12.txt)) : l'incrément que T3
apporte au-dessus de l'objet servi vaut **+1,52 pp, IC95 [−1,12 ; +4,22], SE
1,37 — non résolu** sur la statistique publiée. Sur le compte non pondéré il
vaut +2,19 pp [+0,66 ; +3,72] et McNemar le résout (p = 0,0164). Les deux
statistiques se contredisent : le gain de T3 est réel sur le compte brut et
disparaît sous la pondération par population.

Conséquence arithmétique, et elle décide du montage : les sous-ensembles que ce
palier veut cribler portent des incréments attendus de 0,5 à 1,5 pp, **tous plus
petits que celui qui vient d'échouer à se résoudre**. À 2 280 questions, aucun
bras ne résoudra son effet, et le plan proportionnel (ligne A, barre ÷1,32 à
1,65) n'y suffit pas non plus : il amènerait l'IC de +1,52 à [−0,51 ; +3,55] au
pire, [−0,11 ; +3,15] au mieux (*calculé* sur la SE mesurée).

D'où deux variantes, et une seule question pour l'opérateur.

| | variante **M** — le crible | variante **C** — la porte |
|---|---|---|
| où | Mac, Metal, `limit=40` | L40S, split complet 14 042 questions |
| bras | 5 (C0, C2, A1, A2, A3) | 4 (C2, A1, A2, A3) |
| coût | **0,00 $**, ~3,2 h de M3 Max | **2,80 $** *calculé* (4 × 0,70 $ *mesuré*, 24 min/bras), ~1,6 h |
| ce qu'elle rend | un classement et trois estimations ponctuelles | des écarts exacts : plus d'échantillon, donc plus de barre (*mesuré*, recensement du 2026-09-11, « ± 0,00 ») |
| ce qu'elle ne rend pas | aucun IC excluant zéro (§0) | rien sur le transfert Metal ↔ carte |
| METHODE §1 | porte une **mesure**, pas un gate | porte un **gate** |

L'auteur recommande **C** : 2,80 $ — 2 % du total projet à ce jour, quatre bras
du même prix que celui du recensement du 09-11 — est ce que coûte le passage
d'un classement à une décision. Elle porte le plan T3 de ~4,25 $ à ~7,05 $. La
variante M reste défendable si la contrainte est « zéro dollar » plutôt que
« décider », et elle ne ferme pas C : un crible M peut être suivi d'un seul bras
C sur le survivant, à 1,40 $ (le survivant plus sa base).

**Second résultat gratuit du même jour, qui légitime le crible quelle que soit
la variante** : le chemin `LLVQ_RESTORE_Q4` — poids du checkpoint, sans
compensation GPTQ — ne se distingue pas d'un encodage réel. T2 lit 56,95 contre
55,52 pour le fichier mixte réellement ré-encodé, Δ = +1,43 pp
[−1,42 ; +4,38], McNemar p = 0,2727, **non résolu**. Le crible n'est pas prouvé
optimiste.

## §1 — La question, et pourquoi elle décide

T3 — l'attention entière en int4 g128 — est le plus gros gain MMLU mesuré du
dossier : **+4,99 pp** [+2,18 ; +7,90] contre Tetra nu (*mesuré*, chantier 1).
Mais `v_proj` est **déjà servi en int4** depuis le 2026-09-08 et vaut +3,47 pp à
lui seul. Ce que T3 ajoute réellement à l'objet servi est **+1,52 pp pour
+0,4435 b/param — 3,43 points par b/param**, contre 70,4 pour `v_proj` seul.
Vingt fois pire, pour 62 % de la marge qui reste avant le b_max du triplet
produit.

**T3 n'est pas un bloc, et c'est là qu'est la question.** En GQA, Qwen3-4B a 8
têtes KV : `k` et `v` font 1024 × 2560 quand `q` et `o` font 4096 × 2560. Le
coût en bits d'un type passé de Tetra (2,1498 b/poids) à int4 g128 (4,250) est
donc dans un rapport de quatre (*calculé*, dimensions de `docs/fiche-4b.md` §2 ;
la méthode reproduit exactement le +0,0493 de `v` et le +0,4927 de l'attention
d'ETAT §5 octies) :

| ajouté à l'objet servi (`v` déjà int4) | Δ b/param | b/param | GB carte *calculé* | % de la marge restante |
|---|---|---|---|---|
| rien — l'objet servi | — | 2,8138 | 1,390 *mesuré* | — |
| **`k`** → A1 = `v+k` | **+0,0493** | 2,8631 | 1,414 | 6,9 % |
| `o` → A2 = `v+o` | +0,1971 | 3,0109 | 1,487 | 27,4 % |
| `k+o` → A3 = `v+k+o` | +0,2464 | 3,0602 | 1,512 | 34,3 % |
| `k+o+q` → T3 | +0,4435 | 3,2573 | 1,609 | 61,7 % |

Croisé avec l'attribution M2 (bras f16 sur le fichier publié, ramenés à l'int4
par le taux de survie stratifié 0,751 mesuré sur Tetra), l'efficacité prédite
s'effondre d'un facteur quatre à l'intérieur de T3 : `k` ~32 points par b/param,
`o` ~9,0, `q` ~7,0. **Hypothèse à trancher : `v+k` capture l'essentiel de T3
pour 11 % de ses bits.** Si elle tient, T3 tel qu'il est écrit dans la roadmap
est le mauvais objet et le bon coûte quatre fois moins de mémoire.

## §2 — Les bras, verbatim

Fichier **Tetra pur** (celui du chantier 1, pas l'objet servi mixte), f16, un
seul processus, `LLVQ_RESTORE_Q4` sur le checkpoint.

```
F=<...>/qwen3-4b-tetra.bin                     # 1 770 529 149 octets
export LLVQ_MODEL=Qwen/Qwen3-4B
# variante M : DEV=metal, LIM=40      · variante C : DEV=cuda, LIM=census
# C0 — contrôle de harnais, variante M seulement, doit rejouer 53,49
LLVQ_MMLU_DUMP=$S/g-t0.csv                                mmlu $F $DEV $LIM
# C2 — base des incréments (l'objet servi, en RESTORE)
LLVQ_RESTORE_Q4=v_proj        LLVQ_MMLU_DUMP=$S/g-v.csv   mmlu $F $DEV $LIM
# A1 — v+k
LLVQ_RESTORE_Q4=k_proj,v_proj LLVQ_MMLU_DUMP=$S/g-vk.csv  mmlu $F $DEV $LIM
# A2 — v+o
LLVQ_RESTORE_Q4=v_proj,o_proj LLVQ_MMLU_DUMP=$S/g-vo.csv  mmlu $F $DEV $LIM
# A3 — v+k+o
LLVQ_RESTORE_Q4=k_proj,v_proj,o_proj LLVQ_MMLU_DUMP=$S/g-vko.csv mmlu $F $DEV $LIM
```

Appariement : `mmlupair $S/g-v.csv $S/g-vk.csv $S/g-vo.csv $S/g-vko.csv`, la
base étant **C2 et non C0** — la question est l'incrément au-dessus de ce qui
est servi.

**Pourquoi C0 est rejoué en variante M et pas en variante C.** Les trois dumps
du chantier 1 ont été écrits **sur carte** ; apparier des bras Metal contre eux
mélangerait le device au traitement, donc la variante M refait ses propres
bases. La variante C tourne sur la même carte que le chantier 1 et n'en a pas
besoin — son contrôle est ailleurs (§3.6).

**Pourquoi pas de bras `q` seul.** `q` est le type le plus cher (+0,1971
b/param) et le dernier des deux tirages d'attribution (+1,85 puis +0,49 pp en
f16). Sa valeur s'obtient par différence, T3 − A3, sur des chiffres déjà
mesurés. Un bras de plus paierait la question la moins utile.

## §3 — Les contrôles, et si l'un tombe aucun chiffre n'est publié

1. Empreinte de tokens identique sur tous les bras d'une variante
   (`65dcd53655e8bfa5` en M, celle du split complet en C).
2. Artefact **1 770 529 149 octets**, en-tête v5, kind Tetra.
3. Matrices et poids restaurés imprimés par bras (*calculé*) : C2 **36** et
   **94 371 840** · A1 **72** et **188 743 680** · A2 **72** et **471 859 200** ·
   A3 **108** et **566 231 040**.
4. Tous les dumps sont écrits avec leur trailer `# end`.
5. Aucun bras n'est égal à un autre au centième. Une égalité dirait que la
   restauration n'a pas eu lieu.
6. **Le contrôle de reproduction, par variante.**
   - **M** : C0 rend **53,49** et C2 **56,95**, chacun à **±0,5 pp** — le
     contrôle 6 du préreg du chantier 1, qui est passé (treize matières sur
     quinze identiques au bit). Au-delà, tout repart sur carte.
   - **C** : le sous-ensemble de 2 280 questions du bras C2, extrait par
     `mmlupair … --intersect` contre `mmlu-4b-tetra-q5.csv`, rend **56,95 à
     ±0,5 pp**. Les échantillons s'emboîtent — `select` mélange par graine puis
     tronque ([ROADMAP-QUALITY](../docs/ROADMAP-QUALITY.md) ligne A). Le
     mécanisme est **lu dans le code, pas mesuré** (`mmlupair.rs:332-375` : le
     refus de plan compare `alloc_name`, « flat » des deux côtés, et
     `--intersect` ne lève que le refus d'empreinte, sa documentation nommant
     exactement ce cas). Aucun dump census n'est commité, donc rien ne l'a
     essayé ; **si ce contrôle refuse au lancement, la variante C perd son
     contrôle de reproduction** et le premier bras est C0 au split complet, à
     0,70 $ de plus.
7. A1, A2 et A3 tombent dans **[53,49 − 0,43 ; 58,48 + 1,35]** en variante M.
   Un sous-ensemble hors bornes n'est pas impossible — la sous-additivité
   n'impose pas la monotonie — mais il déclenche une inspection écrite dans les
   écarts avant toute publication.

## §4 — Ce qui se publie, et ce qui NE se compare PAS

Se publie : les MMLU des bras, les incréments **E(S) = MMLU(S) − MMLU(C2)**
appariés (bootstrap stratifié 10 000 tirages, graine `0xb0075eed`, correction de
population finie, McNemar exact), et le rapport **R(S) = E(S) / Δb(S)** en
points par b/param, Δb(S) *calculé* du §1.

Ne se compare pas :
- **Aucun bras n'est un noyau.** `LLVQ_RESTORE_Q4` quantifie puis **déquantifie
  avant le matvec** : aucun octet de 4 bits n'est lu par un noyau, aucune
  vitesse, aucune empreinte réelle. C'est un coût d'information.
- **Ces chiffres ne se comparent pas à l'objet servi** (55,52 à 2 280 questions,
  56,37 au split complet) : fichier différent, et l'écart entre les deux vaut
  +1,43 pp non résolu (§0).
- **E(A3) n'est pas E(A1) + E(A2) − E(C2).** Sous-additivité mesurée 0,618 à
  0,792 ; aucun chiffre d'addition ne sera publié.
- Les deux variantes ne se comparent pas entre elles : plans d'échantillonnage
  différents, `mmlupair` refuse l'appariement et il a raison.
- Un seul tirage de calibration, une seule taille, un seul device.

## §5 — La règle de décision, posée AVANT de voir le chiffre

Soit S* le sous-ensemble de plus grand R(S) parmi A1, A2, A3.

| | conséquence |
|---|---|
| **R(S*) ≥ 20 pts/b-param**, et en variante C l'écart exclut zéro | S* part au palier 1 : ré-encodage réel `LLVQ_INT4_TYPES=S*` sur le Mac, 2 h 27, 0 $ |
| **R(S*) < 20 mais E(S*) ≥ +1,5 pp** (IC excluant zéro en variante C) | Le gain existe et il est cher. Décision opérateur entre S*, T3 entier et le statu quo, sur la marge avant b_max. Aucun palier ne s'enchaîne sans elle |
| **R(S*) < 20 et max E(S) < +1,5 pp** | La précision mixte au-delà de `v_proj` ne rend pas ce qu'elle coûte. **T3 sort du chemin court** ; la suite est les lignes 5, 6 et C de la roadmap qualité, à 0 $ et sans dépenser un bit |
| **autrement** — E(S) < 0 partout, un S dépasse T3 de plus de 1,35 pp, ou un contrôle du §3 tombe | Rien n'est décidé, rien n'est publié, tout va dans les écarts |

En variante M, la règle se lit **sur les estimations ponctuelles**, et le
palier 1 qu'elle déclenche est lui aussi à 0 $ : un crible qui choisit mal ne
coûte que du Mac. En variante C elle se lit sur des écarts exacts et elle
décide.

La barre porte sur **R**, l'efficacité, parce que c'est la question du go : la
marge avant b_max est de 0,7186 b/param et T3 en dépense 62 % pour 1,52 pp. Le
seuil de 20 points par b/param est **une proposition de l'auteur** : il se place
au-dessus de la ligne 20 de la roadmap qualité (Q6b bas-rang, 18 à 35
pts/b-param *estimé*) et loin au-dessus des 3,43 de T3 entier. **Un kill est la
décision de l'opérateur** (METHODE §1) ; ce brouillon n'en écrit aucun.

## §6 — Divulgation datée, et la prédiction signée

Connu à la signature : T0 53,49 · T2 56,95 (+3,47 [+1,42 ; +5,57]) · T3 58,48
(+4,99 [+2,18 ; +7,90]) · **T2 → T3 +1,52 [−1,12 ; +4,22], non résolu, et +2,19
[+0,66 ; +3,72] non pondéré, résolu** (§0) · **T2 → fichier mixte ré-encodé
+1,43 [−1,42 ; +4,38], non résolu** (§0) · survie f16 → int4 sous Tetra 0,751
stratifiée · M2 sur le fichier publié en f16 : `v` +4,48, `o` +2,35, `k` +2,09,
`q` +1,85 · M2 sur la graine 3 : `o` +2,08, `k` +1,11, `q` +0,49, ces deux
derniers non résolus · sous-additivité 0,618 à 0,792 · objet servi 55,52
(2 280) et 56,37 (split complet).

**Prédiction de l'auteur, opposable :**

1. **E(A1) entre +0,5 et +1,5 pp** — 60 %.
2. **E(A2) entre +0,2 et +1,2 pp** — 55 %.
3. **R(A1) > R(A2) et R(A1) > R(A3)** : `k` est le meilleur rapport des trois — 70 %.
4. **En variante C, au plus un des trois écarts exclut zéro** — 60 %.
5. Donc **la troisième ligne du §5**, sauf si R(A1) passe 20, ce qui demande
   E(A1) ≥ +0,99 pp — au milieu de l'intervalle prédit en 1, et c'est le point
   le plus incertain de la prédiction.

Motif : `k` et `v` sont deux matrices de même taille, adjacentes dans le calcul
de l'attention, et `v` a rendu +3,47 pp ; `k` est troisième ou cinquième des
sept types selon le tirage, donc son gain isolé en f16 se lit +1,11 à +2,09,
ramené à +0,83 à +1,57 par la survie int4, puis raboté par la sous-additivité
au-dessus d'un `v` déjà en int4.

**Le défaut connu de ce motif :** la sous-additivité de 0,618 à 0,792 a été
mesurée sur des bras **f16 isolés sous Planes14**, jamais sur des bras int4
empilés sous Tetra. Si `k` et `v` réparent le même défaut — les deux entrées de
même largeur du bloc d'attention — elle peut mordre bien plus dur et E(A1)
tomber sous +0,3 pp. S'ils réparent des choses disjointes, elle peut ne pas
mordre du tout.

**Ce qui rendrait la prédiction fausse de façon instructive : R(A2) > R(A1).**
Cela voudrait dire que le gain de l'attention est porté par `o_proj`, la seule
matrice de l'attention dont l'entrée n'est pas une sortie de RMSNorm — donc que
le déficit de Tetra est un défaut d'**échelle d'activation** et non de matrice,
et la ligne B de la roadmap qualité (température de sortie, pente 0,4660 contre
0,9599 pour AWQ) passerait devant toute précision mixte.

## §7 — Ce que ce palier ne peut pas être

Un chemin servi. Il ne crée aucun fichier, n'écrit aucun octet de 4 bits, ne
touche ni au mot de 48 bits, ni au treillis, ni au bit de gain, et ne mesure ni
vitesse ni empreinte. Les 1,414 à 1,609 GB du §1 sont *calculés* ; ce qui les
mesurera est le palier 4, et le risque qu'il porte est déjà écrit : `tv_q4_h`
n'a pas de noyau par lots, donc une projection int4 s'éclate ligne par ligne au
préremplissage (`llvq-llm/src/model.rs:1010`), et passer de 36 à 144 records
int4 alourdirait la pente de 3,929 ms par jeton de prompt.
