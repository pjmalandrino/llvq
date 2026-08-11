# Pré-enregistrement — réouverture E2 : le critère qui jugera Golay70 v2

**Date : 2026-08-11.** Écrit **avant toute mesure de vitesse de la v2** — il
n'existe à cette heure aucune milliseconde, aucun compte de registres SASS,
aucun profil du décodeur v2 : le seul objet chronométré à ce jour est la v1,
au banc du 2026-08-10. Le dernier banc en date est
[`docs/mesures/six-arm-awq-2026-08-10.txt`](../docs/mesures/six-arm-awq-2026-08-10.txt).

> Ce document est le **lot A** de
> [`docs/spec-apres-awq-2026-08-10.md`](../docs/spec-apres-awq-2026-08-10.md) §6 :
> le critère de remplacement du 1,6×, posé avant le job qui s'y mesurera.
> Il complète le pré-enregistrement du
> [2026-08-10](preregistration-2026-08-10.md), dont il **hérite sans
> dérogation** les gardes (§7), la comptabilité d'octets (§6) et la règle de
> décision sur la résolution du banc (§4) — rien de tout cela n'est recopié
> ici, c'est le même engagement.
>
> ⚠️ Comme le précédent : ni signé GPG ni horodaté tant que l'opérateur ne
> l'a pas fait (`ots stamp`, tag signé — §6.5 du plan de test). D'ici là,
> l'antériorité repose sur la date de commit.

---

## 1. Pourquoi le critère de 1,6× ne peut plus trancher

`Golay70` a été écarté par un seuil de **1,6× contre FP16, posé d'avance**
(`docs/mesures/e2-golay70-bench-2026-08-07.txt`). Il a rendu 1,31× à
l'époque, 1,34× au banc à six bras. **Ce rejet était défendable et le
reste** — rien ici ne le réécrit.

Ce qui a changé est une évidence neuve, pas un chiffre qui déplaît : le banc
du 2026-08-10 a montré qu'un noyau 4 bits déployé (AWQ w4g128, 584 Go/s,
3,38×) est **plus rapide et plus petit** que notre layout de production
`Planes14`. La thèse « notre noyau est rapide » est indéfendable ; l'argument
résiduel du projet est **la mémoire** — et un critère de vitesse ne peut pas
trancher une question de mémoire. Le critère de 1,6× n'est pas effacé, il est
**périmé par cette évidence**, et son remplaçant se pose **avant** de
remesurer. (Texte long : spec §3, écrit le 2026-08-10, avant la v2.)

## 2. Le critère de remplacement

Deux conditions, une par axe. Les seuils sont ceux du spec §3 (2026-08-10),
**avec deux corrections de comptabilité** issues de
[`docs/projections-golay70-2026-08-11.md`](../docs/projections-golay70-2026-08-11.md)
§1–2, consignées ici parce qu'elles changent les nombres, pas le critère :

1. **L'embedding q8 se facture à 8,5 b/param, pas 8,0** (int8 + échelle et
   biais f16 par groupe de 64 — la formule de `rtbits`, validée au millième
   sur les quatre cellules 4B/8B connues). Le b/param `Golay70` du 4B est
   donc **4,065** (calculé), pas 4,016.
2. **L'invariant est la marge relative, pas le 4,1 absolu.** Le « ≤ 4,1 »
   du spec encode « ≥ 20 % de marge vs l'AWQ déployé » *au 4B* (AWQ 4B =
   5,302 b/param, mesuré). Le 8B projeté (4,290) viole le 4,1 absolu tout en
   portant la meilleure marge du tableau (−28,0 %) : c'est la marge qui est
   transposable, le 4,1 n'en est que l'instance 4B.

| condition | verdict |
|---|---|
| v2 **≥ 2,0× FP16** au banc *et* **marge mémoire ≥ 20 %** vs l'AWQ déployé du même modèle (b/param modèle entier, embedding q8 à 8,5) | **adopté** pour le chemin servi : câblage `fusedrun`, arbitrage produit rouvert |
| entre **1,6× et 2,0×** | **non adopté**, publié comme point de la courbe débit↔taux — ce qu'il est déjà |
| **< 1,6×** | le décodage à double coset est irréductible ; **E2 se referme définitivement**, le papier garde son résultat négatif |

**Pourquoi 2,0×** : la vitesse de `Planes12x` au même banc — un seuil déjà
atteint par un layout servi, pas un chiffre inventé. **Pourquoi 20 %** : la
marge qui survit à l'objection MMLU, celle que les 3,5 % de `Planes14` ne
survivaient pas (spec §3).

### Ce qui est déjà tranché, et ce qui reste ouvert

**La condition mémoire est déjà connue satisfaite au 4B** — c'est un compte,
pas un chronométrage : 4,065 contre une borne d'adoption de
0,8 × 5,302 = **4,241** b/param, soit une marge de **23,3 %**. La v2 ne
change pas un octet du format, donc ce compte ne bougera pas.

**Tout le verdict pend donc à la seule condition de vitesse.** Le job du
lot C ne mesure qu'elle.

## 3. La prédiction, et ce qui la fonde

**Fourchette estimée pour la v2 : 1,9–2,4× FP16** (projections §3.3). C'est
un **compte d'instructions** (~14 → ~7 ops entières par slot, prologue par
bloc amorti sur 24 slots), pas un profil ni une mesure — le profileur n'a
jamais été utilisé sur ce projet, et l'émission INT/FP32 partage les ports
sur Ada. Le seuil de 2,0× est **dans** la fourchette : **le critère peut
échouer, c'est prévu pour.** Repères v1 (mesurés, six bras, 2026-08-10) :
8,223 ms / 198 Go/s / 1,34×, contre `Planes14` 5,111 ms / 428 Go/s / 2,16×
et FP16 11,016 ms.

Si la v2 rend plus que 2,4×, c'est **mieux que ce que le compte
d'instructions autorise** : avant d'en faire un titre, chercher l'erreur
(octets non facturés, bras dégradé, contrôle non reproduit).

## 4. Le protocole du job (lot C), figé maintenant

Même carte (L40S), même protocole que `six-arm-awq` : un processus, 7 rounds
dont 2 jetés, bras entrelacés dans chaque round, ordre de dispatch fixe,
rapports formés **round par round** — et toutes les gardes du §7 du
pré-enregistrement du 08-10, sans dérogation.

**Sept bras, deux phases dans le même processus** (le sélecteur
`LLVQ_BENCH_ARMS` du lot B, qui solde l'entorse É1) :

- **phase 1 — le contrôle** : les six bras du run publié du 08-10
  (`fp16, slot32, planes14, planes12x, golay70v1, awq`), seuls leurs tampons
  résidents ;
- **phase 2 — la table** : les mêmes **plus `golay70v2`**, ajouté en dernier
  dans l'ordre de dispatch (la règle : un bras ajouté ne réordonne jamais les
  bras existants). v1 et v2 lisent **les mêmes tampons** (zéro octet de
  format changé), donc la phase 2 n'ajoute aucune résidence VRAM.

Le bras `golay70v1` est le décodeur **publié**, gelé sous des symboles
renommés (`tv_golay70_v1`) : le rapport v2/v1 se forme dans les mêmes rounds
— jamais contre les millisecondes d'un autre job, la règle du §4 du 08-10
(« aucun rapport n'est cité contre un jeu de bras qui ne l'a pas produit »).

`Δ_contrôle` (§4 du 08-10) se lit entre les deux phases, sur les six bras
communs ; la règle de décision `R = max(Δ_contrôle, demi-étendue intra-run)`
s'applique telle quelle au rapport v2/v1 comme au v2/FP16.

**Gardes propres à ce job, à lire avant les rounds :**

- registres et `local_size_bytes == 0` pour `tv_golay70` (v2) **et**
  `tv_golay70_v1` — la v1 tenait en 40 registres, 0 octet local ; un spill
  de la v2 invalide l'estimation du §3 **avant** toute mesure : pas de
  verdict, on corrige d'abord et on le consigne ;
- vérification f64 ligne à ligne des deux bras Golay70 contre la même
  référence exacte, avant tout chronométrage (seuil 1e-5) ;
- coût du job dans `docs/data/jobs.csv`, et première entrée de
  `ops/manifest.jsonl` (dette du spec §5).

## 5. Ce qui est connu à la signature — divulgation datée

Pour qu'on ne puisse pas dire après coup que ces comptes ont « inspiré » le
critère : le critère est celui du spec du **2026-08-10** ; les comptes
suivants datent du **2026-08-11** et ne portent que sur la **correction**
(format, exceptions), jamais sur la vitesse.

- **Le sweep de l'artefact 4B scellé est passé sur le Mac de dev** :
  150 681 600 blocs décodés identiques Golay70 (v2) ↔ Slot32,
  11 204 181 exceptions (7,4357 %), payload 3,4461 b/poids —
  `the_sealed_artifact_decodes_identically_through_golay70`, exécution
  positive (pas un skip : compte de blocs imprimé).
- **Le recensement E2 du 8B est fait** (0 $, `classhist` sur
  `q8b-c12.llvq`, journal
  [`docs/mesures/classhist-e2-8b-2026-08-11.txt`](../docs/mesures/classhist-e2-8b-2026-08-11.txt)) :
  **7,4116 %** d'exceptions totales contre 7,4357 % au 4B, et la composante
  « pair violant » — la seule jamais comptée hors 4B — transfère à
  **4,0394 %** contre 4,05 %. L'hypothèse de transfert du §2.4 des
  projections est close **par un compte** : la sensibilité 6→9 % envisagée
  était hors sujet, le taux réel bouge de 0,3 % relatif.
- **Aucune milliseconde v2 n'existe.** Ni banc, ni registre SASS, ni profil.

## 6. Les issues, et ce que chacune fait au dossier

| issue mesurée | conséquence, décidée d'avance |
|---|---|
| v2 ≥ 2,0× : **adopté** | câblage `LLVQ_FUSED_LAYOUT=golay70` dans `fusedrun`, tokens gloutons contre le bras dense, mesure tok/s qui tranche l'additivité (les ~55 et ~69 tok/s concurrents restent tous deux non mesurés d'ici là — en citer deux ou aucun) |
| 1,6–2,0× : **non adopté** | point de la courbe débit↔taux du papier, avec sa provenance ; E2 reste clos pour le chemin servi |
| < 1,6× | E2 clos **définitivement** ; le résultat négatif du papier est confirmé par une seconde attaque |
| v2 > 2,4× | vérification d'erreur d'abord (§3), publication ensuite seulement |

**Aucune de ces issues ne bloque le lot D du spec** (la conséquence papier de
la mesure AWQ), qui est dû quel que soit le verdict.

## 7. Ce qui invaliderait ce pré-enregistrement

- si le **contrôle de phase 1 ne reproduit pas** le run publié du 08-10 dans
  ses plages (au sens de la règle §4 du 08-10), le banc a dérivé et **aucun
  chiffre de la campagne n'est publiable** avant d'en connaître la cause ;
- si la v2 (ou la v1 gelée) **échoue la vérification f64**, le bras n'existe
  pas — on ne chronomètre pas un décodeur dont la reconstruction n'est pas
  prouvée ;
- si `tv_golay70` **spille**, l'estimation du §3 est invalide : pas de
  verdict sur ce run, correction d'abord, entorse consignée au §7bis du
  pré-enregistrement du 08-10 ;
- si le b/param mesuré par `rtbits` sur l'artefact servi s'écartait du 4,065
  calculé, la comptabilité mémoire du §2 est fausse et la condition mémoire
  se remesure avant tout verdict.

## 7bis. Écarts au protocole — journal, tenu à chaud

*(Vide à la signature. Chaque entorse s'écrit ici le jour où elle est
commise, avec sa raison et son coût — la règle du 08-10.)*
