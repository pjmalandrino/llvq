# Pré-enregistrement — P1b : ce qu'un BLOC coûte, et non une marche

**Date : 2026-08-15, au soir.** Écrit **après P1, avant la première ligne du
bras neuf**, et **avant toute milliseconde** le concernant.

> ⚠️ Ni signé ni horodaté tant que l'opérateur ne l'a pas fait
> (`ots stamp proofs/preregistration-p1b-2026-08-15.md`). P1 l'a été **avant**
> sa mesure ; P5 ne l'a pas été et son journal porte la dette. Celui-ci demande
> le tampon avant la première milliseconde, comme P1.

---

## 0. Pourquoi ce document existe

[P1](preregistration-p1-2026-08-13.md) a mesuré `marche-binomiale` à
**0,3101 ns/bloc** et le §4.2 en a tiré que le bras CUDA de P4 est autorisé
(seuil 0,45) et que [P5](preregistration-p5-2026-08-14.md) s'ouvre.

**Mais ce bras décode UNE MARCHE de 24 créneaux, pas un BLOC.** Un bloc de
classe **paire** en demande **deux** — le support de `w` créneaux et son
complément de `24 − w` — plus le mot de Golay et la réparation de parité.
`binomial_walk.metal` §9 le déclarait ABSENT dès son écriture et exigeait que
le journal dise lequel des deux il mesure, « et la lecture conservatrice est
celle qui coûte le plus au bras ». **La première émission du journal de P1 ne le
disait pas** ; la réserve y a été ajoutée le même soir, avec ses chiffres.

P1 est **horodaté, donc en lecture seule pour toujours** : sa table ne peut pas
accueillir un sixième bras, et il ne faut pas essayer. P5, lui, exclut
explicitement toute vitesse de décodage (§0). Ce bras n'appartient donc à aucun
des deux, et c'est pour ça qu'il a son document.

## 1. Ce qui est connu à la signature — divulgation datée

- **0,3101 ns/bloc** pour `marche-binomiale`, **1,7809** pour la cascade
  uniformisée, **10,8115** pour l'archive, ancres `sol` **0,0777** et
  `sol-rang` **0,0796** [mesuré, `docs/mesures/p1-rankbench-2026-08-15.txt`].
- **Le compte de pas, pondéré par les 150 681 600 blocs du 4B** [mesuré, via le
  compteur de la CNS, journal de P1 §réserve] :

  | | moyenne | max |
  |---|---|---|
  | une marche de 24 créneaux (ce que P1 a mesuré) | 39,55 | 48 |
  | un bloc complet (ce que ce bras mesure) | 39,64 | **90** |

  **En moyenne les deux coûtent la même chose** — le travail total se conserve,
  `w + (24 − w) = 24`. **En queue non** : le maximum double. Sur GPU c'est le
  max dans le warp qui décide du temps, pas la moyenne, et c'est toute la raison
  d'être de ce bras.
- **La largeur CNS maximale est 56 bits** sur 8 champs [mesuré, classe 323], donc
  le record tient dans les **96 bits** du bras marche : les deux bras liront un
  stride de 12 octets, et leur écart ne pourra pas venir de l'adressage.
- **Aucune milliseconde n'existe pour un décodage de bloc E1v**, sur aucun
  matériel. Aucun compte de registres, aucun profil.

🚨 **Ces nombres sont écrits ici pour qu'on ne puisse pas dire après coup qu'ils
ont inspiré le seuil.** Le seuil du §3 est celui de P1, inchangé, et il datait du
2026-08-13.

## 2. Le bras, figé ici

Un seul bras neuf, **`marche-bloc`**, ajouté **en dernière position** de la table
de `bin/rankbench` pour ne réordonner aucun bras existant (P1 §1.3, qui reste
opposable même si son document est clos).

| | |
|---|---|
| kernel | `binomial_block.metal::decode_block`, **fichier neuf** |
| flux | record CNS empaqueté, **stride 12 o, le même que `marche-binomiale`** |
| étalon | `llvq_search::cns::cns_decode`, sur les mêmes records |
| ce qu'il décode | le bloc entier : mot de Golay, les deux marches, les trois règles de signe, la réparation de parité |

🚨 **`decode_walk` n'est pas modifié.** C'est le bras que P1 a mesuré et son
document est scellé ; toucher un registre le rendrait incomparable à son propre
journal. Le bras neuf est un **fichier séparé**, et les deux tournent dans le
même run pour que leur écart soit lisible round par round.

⚠️ **Ce que le bras neuf paie en plus, et qu'il faut nommer avant de lire son
chiffre** : une recherche de mot de Golay (une lecture indexée par la classe et
le rang), deux marches au lieu d'une sur le coset pair, la réparation de parité,
et la règle de signe à trois branches sélectionnées par arithmétique. **Ce n'est
pas la même quantité de travail, et le journal ne dira jamais « la marche coûte
X » à partir de ce bras** : il dira ce qu'un bloc coûte.

## 3. Les seuils — ceux de P1, sans amendement

Ils ne sont pas rouverts et aucun n'est inventé ici.

| | seuil | conséquence |
|---|---|---|
| `marche-bloc` | ≤ **1,5 ns/bloc** | > 1,5 ⇒ le décodeur d'E1v est **mort**, comme P1 §4.1 le dit du bras marche |
| gate CUDA de P4 | ≤ **0,45 ns/bloc** | > 0,45 ⇒ 🚨 **l'autorisation du bras CUDA de P4 est RETIRÉE** |

**La seconde ligne est la seule décision neuve de ce document, et elle est
écrite avant la mesure.** P1 §4.2 pose son gate en **ns/bloc**. Le bras qui l'a
franchi décodait une marche ; si un bloc coûte davantage, alors le gate a été
franchi par un nombre qui ne décrivait pas ce qu'il prétendait décrire, et
l'autorisation tombe. **Ce n'est pas rouvrir le seuil de P1 — c'est appliquer le
seuil de P1 à la bonne quantité.**

⚠️ **Et P5 n'est PAS remis en cause par ce bras.** Son ouverture est acquise sur
`marche-binomiale ≤ 0,45` (P1 §6, condition énoncée sur ce bras-là), ses quatre
critères sont verts, et rien ici ne les touche. Ce qui se joue est le **bras
CUDA de P4**, pas la ré-bijection.

## 4. V0 avant V1 — sans exception

1. Le shader **compile** (`bin/mslcheck`, qui liste les points d'entrée un par
   un, jamais les fichiers).
2. **Chaque bloc du tirage** de `rankbench` — 2^24 — vérifié contre
   `cns_decode`, tolérance `1e-5 · Σ|wᵢxᵢ|`, la forme de `thesis` et non le
   plancher absolu de `decreal`.
3. **La fixture synthétique** : l'origine, les deux bornes de chaque classe, les
   82 classes de coquille 13 — les 98 entrées qu'aucun fichier cap 12 n'atteint.
4. **Tout écart enterre le bras sans banc.**

⚠️ L'étalon `cns_decode` est **déjà balayé** sur les 150 681 600 blocs du 4B
contre `FastDecoder::decode` (P5 C2, zéro écart). Ce n'est pas une référence
neuve : c'est celle que P5 a prouvée.

## 5. La comptabilité — celle de P1 §1, héritée sans dérogation

`N = 2^24`, 18 rounds dont 3 jetés, tous les bras à chaque round dans l'ordre
figé, surcoût mesuré **par round** avec sa dispersion imprimée, rapports formés
**round par round**. Aucun seuil, aucun rapport, aucune conclusion ne se lit
contre un chiffre d'un autre run — **y compris contre les 0,3101 ns de P1**, qui
sont un autre run et un autre bras. `sol`, `masques` et `sol-rang` se
remesurent dans le même processus et servent de contrôle.

## 6. Ce qui invaliderait ce pré-enregistrement

- si les ancres ne se reproduisent pas dans le run, aucun chiffre n'est
  publiable avant d'en connaître la cause ;
- si le bras échoue V0, il n'existe pas — pas de chronométrage, pas de verdict ;
- si le bras se révèle éliminé par le compilateur, sa milliseconde ne mesure
  rien : le journal doit montrer que sa sortie a été relue ;
- si le bras rend **mieux que 0,20 ns/bloc**, chercher l'erreur avant d'en faire
  un titre (P1 §5, hérité).

## 7. Écarts au protocole — journal, tenu à chaud

*(Chaque entorse s'écrit ici le jour où elle est commise.)*

### É1 — 2026-08-15, après la mesure de `marche-bloc` : le bras mesurait peut-être un débordement

> ❌ **MESURÉ ET RÉFUTÉ le même soir.** `marche-bloc-plat` rend **0,8346 ns/bloc**
> contre **0,6704** pour le bras qu'il devait battre : **24 % plus lent**. Le
> débordement n'explique pas le ×2,17, et ce qui l'a remplacé coûte davantage.
> La règle de restitution ci-dessous **ne tire pas** : le meilleur décodeur de
> bloc reste à 0,6704 > 0,45, et **l'autorisation du bras CUDA de P4 reste
> retirée**. Journal :
> [`p1b-marche-bloc-2026-08-15.txt`](../docs/mesures/p1b-marche-bloc-2026-08-15.txt).

**Ce qui s'est passé.** `marche-bloc` rend **0,6735 ns/bloc**, soit **×2,17** le
bras marche là où le compte de pas prédisait ×1,002. L'écart a été attribué en
prose — Golay, second appel, parité, règles de signe — sans être décomposé.
En relisant le shader neuf contre `cascade_uniform.metal` §1(d), une cause bien
plus précise apparaît, et elle était **écrite dans le dépôt avant que j'écrive
le bras** :

> « **No dynamic indexing of registers.** […] On a GPU a dynamically indexed
> register array is spilled to thread-local memory; this arm would be measuring
> a spill. »

`binomial_block.metal` porte **deux** tableaux de 24 octets indexés
dynamiquement — `kinds_on[s]` et `kinds_off[s]`, écrits par les marches et relus
par la boucle de dépôt. `decode_walk`, lui, n'en a **aucun** : il dépose
directement depuis le masque `taken` (`dep_slots`) et ne matérialise jamais un
genre par créneau. **La différence structurelle entre les deux bras n'est donc
pas seulement le travail supplémentaire du bloc — c'est aussi un débordement en
mémoire thread-local que le bras marche ne paie pas.**

**Ce qui est proposé : un bras de plus, `marche-bloc-plat`**, même sortie, même
record, même stride, dont l'état par créneau tombe de **48 octets indexés par le
créneau** à **8 mots indexés par un compteur de boucle**. Les rangs de signe s'y
calculent par popcount plutôt que par lecture indexée :

- rang d'un créneau **dans le support** : `popcount(cw & ((1<<s) − 1))` ;
- rang d'un créneau **parmi les non-nuls hors support** : même popcount sur le
  masque des genres non nuls, réunion de leurs masques `taken`.

🚨 **Et une promesse est retirée avant d'être mesurée.** La première rédaction
de cet É1 disait « sans **aucun** tableau indexé dynamiquement ». C'est
**impossible** et il vaut mieux l'écrire que le découvrir après : le rang de
signe hors support est le nombre de créneaux non nuls **précédents**, donc il
exige le masque complet des genres non nuls, donc il exige d'avoir parcouru
**tous** les genres avant de déposer le premier. Un dépôt genre par genre au fil
de la marche ne peut pas le connaître. Il reste donc **un** tableau — les
masques `taken`, `MAX_KINDS = 8` mots — mais il est indexé par le **compteur de
boucle**, pas par une donnée, là où le bras mesuré indexe 48 octets par le
**créneau**. C'est une réduction, pas une suppression, et le journal dira
« réduction ».

⚠️ **Un second point d'implémentation, décidé ici plutôt que découvert** : les
deux champs de signes vivent **à la fin** du record, après les deux
arrangements, donc un dépôt au fil de la marche ne les a pas encore lus. Ils
sont extraits d'avance à un décalage que le record connaît
(`10 + wb_golay + Σ wb_on + Σ wb_off`), sans changer le format d'un bit — la
largeur ne bouge pas, donc C1 de P5 n'est pas touché, et le flux empaqueté non
plus.

**Les seuils ne bougent pas**, et la règle de restitution est écrite **avant** la
mesure du bras neuf, symétrique de celle qui a retiré :

> **Si le meilleur décodeur de BLOC du run rend ≤ 0,45 ns, l'autorisation du
> bras CUDA de P4 est RÉTABLIE.**

⚠️ **Pourquoi cette symétrie est nécessaire et non une complaisance.** Le §3 de
ce document a retiré une autorisation parce que le gate porte sur un **bloc** et
qu'aucun décodeur de bloc ne le franchissait. Si un décodeur de bloc le
franchit, le gate est franchi — c'est la règle du §4.2 de P1, inchangée,
appliquée à la même quantité. Une règle qui ne saurait que **retirer** serait
aussi arbitraire qu'une règle qui ne saurait que **donner** ; ce qui la rend
opposable est qu'elle est écrite avant de voir le chiffre, dans les deux sens.

⚠️ **Et ce qui ne change pas** : `marche-bloc` garde son verdict, 0,6735 ns,
publié. Un bras plus rapide ne le rétracte pas — il ajoute un point à la courbe
et dit ce que coûtait le débordement. Les deux tournent dans le même run.

**Antériorité.** Écrit avant la première ligne de `binomial_block_flat.metal`,
et avant toute milliseconde le concernant. Ce document n'est toujours pas
horodaté ; la dette est déclarée en tête du journal de P1b.
