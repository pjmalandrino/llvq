# Pré-enregistrement — F1b : rétention gaussienne du codebook à trois sections

**Écrit, commité et TAMPONNÉ le 2026-09-04, AVANT la première mesure.**
Vague 2 ouverte le même jour, plafond **2,00 $** ; F1b coûte **0 $**, Mac.

🚨 **Ce fichier ne s'édite plus.** Ce qui le corrige va dans un `-ECARTS.md`.

## §1 — F1b N'EST PAS UNE PORTE, et c'est le point de départ

Règle d'opérateur du 2026-09-04, inscrite à `docs/METHODE.md` §1 : **une porte se
pose sur un critère fondamental, jamais sur un intermédiaire.** Les critères
fondamentaux sont les quatre axes — disque, VRAM, débit, qualité — plus la
classe de modèle chargeable, le coût d'encodage et le plancher de bruit.

F1b mesure une rétention sur une source gaussienne synthétique. Aucun modèle
n'y intervient. Ce n'est aucun des sept.

Donc F1b **ne décide rien**. Elle produit un nombre qui alimente la prédiction
signée contre laquelle F1c sera lue, et F1c porte la première porte de l'axe,
sur la perplexité. Les anciens seuils — adoption ≥ 91,0 %, kill < 90,3 % — sont
retirés, pas remplacés (`docs/ROADMAP.md` §2.2 bis).

Conséquence de forme : ce préreg n'a pas de tableau de décision. Il fixe un
protocole et une prédiction, et c'est tout ce qu'un tampon peut protéger ici.

## §2 — Ce qui est mesuré

Un codebook lisant Λ₂₄ comme un code en cosets de E₈ à trois sections,
mot `[état 8][s₁ w₁][s₂ w₂][s₃ w₃][gain 1]` de 48 bits, w₁+w₂+w₃ = 39.

Acquis, à ne pas re-dériver (`docs/mesures/f1a-comptes-2026-09-04.txt` avec sa
correction, `docs/archive/f1-regle-de-troncature-2026-09-04.md`) :

- ordre en trio `0x00149f | 0x0f6840 | 0xf08320`, 256 états Λ₂₄, 16 branches
  par état de Golay ;
- sections extrêmes sur un coset de 4·E₈ (covolume 2¹⁶), section du milieu sur
  un coset de 2√2·E₈ (covolume 2¹²) ;
- bijection **prouvée**, deux constructions indépendantes, fermeture des
  covolumes 2¹⁶ · 2¹² · 2¹⁶ / 256 = 2³⁶ = det(√8·Λ₂₄).

**Périmètre.** Banc uniquement. Rien dans `llvq-core`, `llvq-search`,
`llvq-quant`, `llvq-artifact` ni `llvq-llm` ne bouge ; aucun chemin servi,
aucun format, aucun test existant, et `codebook_fingerprint` ne bouge pas. La
permutation en trio vit dans une copie privée de la table de mots de Golay,
à l'intérieur du nouveau module.

## §3 — Le protocole, et la seule chose qui rend le nombre comparable

20 000 blocs gaussiens i.i.d., graine figée, dimension 24, et **un témoin
boule-12 dans LE MÊME PROCESSUS**, sur les mêmes blocs, avec la même règle de
score et le même balayage d'échelle.

⚠️ **Le 92,14 % n'est pas à nous.** C'est la Table 8 du papier, sur sa MSE non
arrondie 0,077718 ; ce dépôt ne l'a jamais mesuré. Le témoin de ce job est donc
la première mesure interne de cette configuration, et le journal doit imprimer
sa valeur À CÔTÉ du 92,14 pour que l'écart de banc soit visible avant qu'on
lise le nombre de F1.

**Le débit se compte sur 48 bits pour 24 dimensions, soit 2,000 b/dim, des deux
côtés.** Diviser par les 47 bits du champ point flatterait F1 de 1,9 point ;
diviser par un débit fractionnaire est l'erreur du 2026-08-04, qui a coûté une
rétention de 92,24 % retirée.

**Découpes rapportées** : 12/15/12 en tête, plus 12/16/11 et **13/13/13** —
cette dernière parce que c'est la formulation littérale de la roadmap et qu'un
prototype la donne cinq points plus bas, pour une raison d'anisotropie qui n'a
rien à voir avec l'idée de F1.

**Troncature** : les 2^w points de plus petite norme du coset, égalités
départagées par ordre lexicographique croissant des 8 coordonnées. Un rayon
fixe est refusé : le nombre de points d'un coset dans une boule dépend du
décalage du coset, donc le mot cesserait d'être une bijection et le débit
annoncé serait faux.

## §4 — Les contrôles, et si l'un tombe aucun chiffre n'est publié

1. **Permutation, assertion directe.** `perm(0x00149f) == 0x0000ff`,
   `perm(0x0f6840) == 0x00ff00`, `perm(0xf08320) == 0xff0000`, et les trois
   dans la table permutée. ⚠️ Ce contrôle remplace celui que la spec proposait :
   reproduire 196 560 par la construction à trois sections rend le même nombre
   sous l'ordre trio, sous l'identité et sous une permutation aléatoire — il ne
   voit pas ce qu'il est censé attraper (mesuré par la relecture adverse).
2. **Appartenance.** Tout point gagnant, permutation défaite, passe
   `llvq_core::Leech::contains`. Zéro exception tolérée.
3. **Rangs par MOTIF, pas par coset.** Le DP de comptage et les tables de
   norme exacte sont indexés par motif. Un comptage groupé par coset compte des
   vecteurs prenant une coordonnée d'un motif et la suivante d'un autre : mesuré
   à 609 553 contre 2 401 sur un coset réel, un facteur 254.
4. **Bijection.** `unrank(i)` pour i dans 0..2^w rend 2^w points distincts,
   tous dans un seul motif.
5. **Suboptimalité de l'encodeur, bornée et non supposée.** Le même encodeur à
   balayage tourne sur la boule-12, où `nearest_angular` donne la réponse
   exacte ; l'écart mesuré est publié. Sans ce contrôle, un rouge ne se
   distingue pas d'un encodeur faible.

## §5 — Prédiction signée, opposable

**Rétention 89,6 % pour la meilleure découpe, plage [88,8 ; 90,3].**
Le témoin boule-12 interne tombe entre 91,9 et 92,2.

Motif : la borne fermée pour un produit de trois boules de dimension 8 est
89,10 % à 2,000 b/dim, et c'est un **plafond** pour toute région par sections,
pas un plancher. Trois effets poussent au-dessus — le format est shape-gain
donc c'est la résolution angulaire qui compte et non la forme radiale ; F1
sature l'espace de code (2⁴⁷ exactement contre N(12) = 2⁴⁶·⁶⁶, +21 % de
points) ; les 256 états font une union de produits décalés. Un pousse en
dessous : l'anisotropie résiduelle de la découpe entière.

**Ce qui rendrait cette prédiction fausse de façon instructive** : un nombre
au-dessus de 90,3. Cela voudrait dire que le plafond du produit ne s'applique
pas comme je le crois, et il faudrait comprendre pourquoi avant F1c.

⚠️ Historique des prédictions signées de ce dossier : deux fausses le
2026-08-25 ; une juste sur le nombre et fausse sur la conclusion le 09-02 ;
une réfutée sur un tirage et confirmée sur l'autre le 09-04 matin ; une juste
sur les deux le 09-04 après-midi. Celle-ci est opposable, pas crédible d'avance.
