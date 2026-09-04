# Pré-enregistrement — le plancher de la table du décodeur F1

**Écrit, commité et TAMPONNÉ le 2026-09-04, AVANT la première milliseconde.**
Vague 2, plafond **2,00 $** ; ce job coûte **~0,09 $**, plafond propre **0,30 $**.

🚨 **Ce fichier ne s'édite plus.** Ce qui le corrige va dans un `-ECARTS.md`.

## §1 — Ce n'est pas une porte

Règle d'opérateur du 2026-09-04 (`docs/METHODE.md` §1) : une porte se pose sur
un critère fondamental — disque, VRAM, débit, qualité, classe de modèle, coût
d'encodage, plancher de bruit — jamais sur un intermédiaire.

Ce banc chronomètre un noyau synthétique qui ne calcule aucun produit du modèle.
Ses millisecondes ne sont le débit de rien. **Il ne décide donc rien.**

Ce qu'il informe : faut-il écrire `tv_l3e8`, qui est une semaine de travail. Un
choix d'effort d'ingénierie, pas un verdict de projet.

## §2 — La question, et pourquoi elle vaut 0,09 $

F1 arrête de déplier : 48 bits écrits, 48 bits lus, contre 2,16 écrits et 4,804
servis. Sa qualité est mesurée et elle tient — 89,55 % de rétention contre un
témoin boule-12 à 92,00 dans le même processus (*mesuré*,
`docs/mesures/f1b-*`, 16 000 blocs).

Le risque est entièrement dans le décodeur. Décoder devient `étiquette → point`
à travers une table qui **ne tient pas en mémoire partagée** : 3 336 Kio contre
les 101 376 o d'opt-in de la carte, 34× de trop (*calculé*,
`llvq-bench/examples/f1table.rs`). Une passe modèle sur les 3 633 315 840 poids
de projection du 4B fait 151,4 M blocs, donc **454 M consultations**, soit
14,5 Go de lectures de table contre 0,98 Go de lectures de poids — **15 pour 1**.

Trois pistes de format sont mortes sur le coût de décodage et aucune sur la
qualité. E1v avait de **meilleurs octets que F1**, 2,3877 b/poids, et mesurait
0,25× FP16 (`docs/mesures/e1v-cuda-2026-08-16.txt`).

## §3 — L'échelle, et le budget que F1 doit tenir

Tout en 252 lancements, la géométrie de tous les planchers publiés.

```
  B = t(Planes14) − t(nullk) = 5,103 − 2,306 = 2,797 ms
      ce que le noyau servi dépense en flux ET décodage
  S = flux de poids de F1 = 0,981 Go à 779 Go/s net = 1,259 ms
  ────────────────────────────────────────────────────────
  H = B − S = 1,538 ms
      ce que F1 a pour sa table ET tout son décodage
```

`B` et le débit net de 779 Go/s sont *mesurés* (`docs/format-noyau.md` §6).

## §4 — Le protocole

Vingt bras, entrelacés, tous à chaque tour, ordre fixe, 9 tours dont 2 de
chauffe écartés. Toute différence se forme **tour par tour**, jamais comme un
quotient de minima.

L'échelle de lecture est un barreau, jamais une différence brute :

```
  Didx  = t(hash3) − t(nullk)      fabriquer les indices, que le vrai F1
                                   obtient GRATUITEMENT de son mot de poids
  D(S)  = t(tab3, S) − t(hash3)    LA TABLE SEULE, à l'empreinte S
  Dsm   = t(smem) − t(smem_fill)   les consultations quand le jeu chaud est
                                   PLACÉ en mémoire partagée, à cette occupation
```

⚠️ `Dsm` ne se forme pas contre `nullk`. Un bras à 60 Kio/bloc et un à 12 Kio/bloc
n'ont pas la même occupation, et différencier à travers ça est ce que
`format-noyau.md` §6 interdit — l'interdiction qui a fait sauter le premier
seuil de F1d ce matin. D'où l'ancre appariée.

**Empreintes** : 8 Kio, 16, 128 Kio, 1 Mio, 4 Mio, 16 Mio, 64 Mio, 1 Gio.
Puissances de deux pour que l'adresse soit un ET et que seul le mélangeur décide
où une consultation tombe. Le dernier point est l'étalonnage DRAM : il doit
valoir au moins 20× la L2, sinon un « échec de cache » est un tiers de succès.

**Jeux chauds en mémoire partagée** : 24 Kio (2 blocs/SM, 17 % des consultations)
et 48 Kio (1 bloc/SM, 33 %), contre 8 blocs/SM aujourd'hui.

## §5 — Pourquoi l'empreinte est balayée et l'accès n'est pas modélisé

🚨 Un tirage uniforme sur toute la table est **pessimiste**, pas optimiste, et
l'avoir d'abord présenté comme un plancher optimiste était une erreur.

Sur les 67 orbites des sections extrêmes, **deux portent 256 des 512 régions** —
la moitié des consultations extrêmes, un tiers de toutes, dans 48 Kio
(*calculé*, f1table.rs, tailles `[128, 128, 4, 4, …]`). Un tirage uniforme
détruit un jeu chaud que le matériel garderait gratuitement, donc un kill lu
dessus serait un kill sur le mélangeur.

L'accès uniforme sur S octets est monotone en S. Le balayage encadre donc la
vérité des deux côtés sans inventer aucune distribution :

```
  D(16 Kio)  ≤  D(l'accès réel, biaisé)  ≤  D(4 Mio)
```

## §6 — Les contrôles, et si l'un tombe aucun chiffre n'est publié

1. **Rien n'est élidé.** Les mots chargés se replient dans un flottant qui
   multiplie les 24 fentes, donc la chaîne global → registre → `acc` → `y` est
   complète. Le banc exige en plus que la sortie d'un bras de table **diffère**
   de celle de `nullk` : un compilateur qui aurait supprimé les trois lectures
   laisserait le repli constant et les deux sorties identiques au bit.
2. **Tout est observable** : chaque ligne de sortie écrite, finie, pas toute nulle.
3. **Le point DRAM est ≥ 20× la L2**, vérifié contre l'attribut de la carte, et
   le banc refuse de démarrer sinon.
4. **Une seule géométrie, un seul processus.** Aucun temps ici ne se compare à
   un temps d'un autre processus.

## §7 — Prédiction signée, opposable

**D(4 Mio) entre 4 et 10 ms, soit 3 à 6 fois le budget H de 1,538 ms.**
**D(16 Kio) sous 1 ms.** **Didx sous 0,3 ms.**

Motif : 14,5 Go de trafic de table ; à une bande passante L2 de 1 500 à
3 000 Go/s cela fait 9,7 à 4,8 ms.

🚨 **Cette prédiction repose sur une bande passante L2 que ce dépôt n'a jamais
mesurée.** C'est précisément pourquoi ce banc existe, et c'est la partie la plus
faible de la prédiction — pas le raisonnement, la constante.

**Ce qui la rendrait fausse de façon instructive** : D(4 Mio) sous 2 ms. Cela
voudrait dire que les 32 octets par consultation sont une majoration grossière —
que les voies d'un warp se regroupent bien plus que je ne crois — et il faudrait
comprendre pourquoi avant d'écrire quoi que ce soit.

**Sur les bras mémoire partagée** : Dsm(48 Kio) entre 0,5 et 2 ms, donc
possiblement **sous** le budget. Si ça se vérifie, la sortie de F1 n'est pas la
table complète mais le placement du jeu chaud — et le périmètre de F1 change.

⚠️ Historique des prédictions signées de ce dossier : deux fausses le 08-25 ;
une juste sur le nombre et fausse sur la conclusion le 09-02 ; une réfutée sur
un tirage et confirmée sur l'autre le 09-04 matin ; deux justes le 09-04
(M2b graine 3, et F1b à cinq centièmes de point). Celle-ci est opposable, pas
crédible d'avance.

## §8 — Ce que ce banc ne peut pas être

Un coût. C'est un **plancher**. Aucun flux de poids ne dispute la L1 ni la LSU
ici, et les étiquettes n'arrivent pas de ce flux — donc la chaîne d'adresses
DRAM → étiquette → L2 → coordonnées n'est pas reproduite. Ce régime-là est celui
de F1d. Ce banc dit seulement si F1d vaut la peine d'être écrit.

Et il n'a aucune référence f64 contre laquelle être vérifié : comme `nullk` et
comme `sol` de `bin/rankbench`, il ne calcule aucun produit du modèle. Ce qu'on
lui demande est d'être OBSERVABLE, pas juste.
