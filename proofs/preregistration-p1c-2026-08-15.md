# Pré-enregistrement — P1c : ce que le FLUX E1v coûte, adressage compris

**Date : 2026-08-15, tard.** Écrit **avant la première ligne du shader neuf** et
**avant toute milliseconde** le concernant.

> ⚠️ Ni signé ni horodaté tant que l'opérateur ne l'a pas fait
> (`ots stamp proofs/preregistration-p1c-2026-08-15.md`). P1 l'a été **avant**
> sa mesure ; P5 et P1b **après**, et leurs journaux portent la dette. Celui-ci
> demande le tampon avant la première milliseconde, comme P1.

---

## 0. Pourquoi ce document existe, et ce qu'il refuse de laisser passer

[P1b](preregistration-p1b-2026-08-15.md) a mesuré `marche-bloc` à **0,6704
ns/bloc** et en a tiré que l'autorisation du bras CUDA de P4 tombe. Ce chiffre
est celui d'**un décodage de bloc sur un record à stride fixe de 12 octets**.

**Le flux E1v n'est pas ça.** Ses records ont des **largeurs variables** — 56
bits au pire, 10 au minimum pour l'origine — et une lane ne trouve le sien
qu'après une **somme préfixe sur les largeurs de son groupe** (le warp-scan),
elle-même précédée d'une lecture du mot de base du groupe. Le stride fixe est
un **proxy** : il décode le même contenu, il ne paie pas la même facture
d'adresses.

🚨 **Donc 0,6704 est une BORNE INFÉRIEURE du coût d'un bloc E1v, et la
passation du soir l'écrit ainsi.** Ce document mesure la vraie facture. Il ne
peut que **creuser** l'écart au gate, jamais le combler — et c'est précisément
pour ça qu'il faut le mesurer plutôt que de laisser le proxy tenir lieu de
chiffre.

P1 et P1b sont **horodatés, donc scellés** : ni l'un ni l'autre ne peut
accueillir ce bras. D'où ce document.

## 1. Ce qui est connu à la signature — divulgation datée

- **0,6704 ns/bloc** pour `marche-bloc` (stride fixe), **0,8346** pour sa
  variante plate, **0,3097** pour `marche-binomiale` (une marche), ancres `sol`
  **0,0812** et `masques` **0,1466**
  [mesuré, `docs/mesures/p1b-marche-bloc-2026-08-15.txt`].
- **E1v pèse 2,3877 b/poids noyau**, 53,7370 bits/bloc adressés, **mesurés sur
  les octets écrits** [P5 C1].
- **La largeur maximale d'un record est 56 bits** sur 8 champs (classe 323) ;
  la moyenne pondérée par le fichier est **53,74 − 32/32 = 52,74** bits de
  charge utile par bloc [calculé].
- **L'alignement warp coûte +0,48 % en bits** à E1v et +15,47 % à `E1c14`
  [calculé, journal de P5 et `x3-alignement-warp-2026-08-15.txt`].
- **Aucune milliseconde n'existe pour un décodage du flux E1v**, sur aucun
  matériel. Aucun compte de registres, aucun profil.

🚨 Écrits ici pour qu'on ne puisse pas dire après coup qu'ils ont inspiré le
seuil. Les seuils du §3 sont ceux de P1, datés du 2026-08-13, et ils ne bougent
pas.

## 2. Le bras, figé ici

Un seul bras neuf, **`e1v-flux`**, ajouté **en dernière position** de la table
de `bin/rankbench` — un bras ajouté ne réordonne jamais le dispatch des bras
qui ont produit un journal (P1 §1.3, opposable même si son document est clos).

| | |
|---|---|
| kernel | `e1v_flux.metal::decode_e1v`, **fichier neuf** |
| flux | le **vrai** flux : `llvq_artifact::e1v::transcode_e1v` sur les blocs tirés, mots de base compris |
| adressage | mot de base du groupe, puis **somme préfixe SIMD** sur les 32 largeurs de charge utile |
| étalon | `llvq_search::cns::cns_decode`, déjà balayé sur les 150 681 600 blocs contre `FastDecoder::decode` (P5 C2) |

🚨 **`decode_block` n'est pas modifié.** C'est le bras que P1b a mesuré et son
document est scellé. Le bras neuf est un fichier séparé, et les deux tournent
dans le même run pour que **leur écart soit exactement le prix de l'adressage**
— même contenu décodé, même table, même étalon, seule la façon de trouver le
record change.

## 3. Les seuils — ceux de P1, sans amendement

| | seuil | conséquence |
|---|---|---|
| `e1v-flux` | ≤ **1,5 ns/bloc** | > 1,5 ⇒ le décodeur d'E1v est **mort** |
| gate CUDA de P4 | ≤ **0,45 ns/bloc** | ≤ 0,45 ⇒ l'autorisation retirée par P1b est **RÉTABLIE** |

**La seconde ligne est symétrique de celle qui a retiré**, et elle est écrite
avant la mesure pour la même raison : une règle qui ne saurait que retirer
serait aussi arbitraire qu'une règle qui ne saurait que donner. Elle est
formulée sur **le meilleur décodeur de bloc du run**, pas sur ce bras — si
`marche-bloc` reste le meilleur, c'est lui qui compte.

⚠️ **Et il ne faut pas s'attendre à ce qu'elle tire.** Ce bras paie tout ce que
`marche-bloc` paie, **plus** l'adressage. Sa raison d'être n'est pas d'espérer
un meilleur chiffre : c'est que le chiffre publié décrive l'objet qu'on
prétend servir.

## 4. Ce que ce banc mesure, et les deux choses qu'il ne mesure pas

**Il mesure** : un décodage seul, sur Metal, sur un M3 Max, **un bloc par
lane**, sans matvec, sans réduction inter-lanes, sans tuilage — la comptabilité
de P1 §0, héritée sans dérogation.

🚨 **Il ne mesure pas le cas servi, et c'est une réserve à écrire dans le
journal, pas une note de bas de page.** Dans ce banc, `gid` **est** l'indice de
bloc, donc un SIMD group de 32 lanes consécutives **est exactement un groupe
E1v** : l'alignement est vrai **par construction**. Le matvec servi met un warp
par **ligne**, et `nblocks mod 32` vaut 10 ou 21 sur les cinq formes du 4B —
**aucun** warp n'y lit un seul groupe
([`x3-alignement-warp-2026-08-15.txt`](../docs/mesures/x3-alignement-warp-2026-08-15.txt)).
Ce banc mesure donc le **meilleur cas** de l'adressage E1v ; le cas servi lira
deux mots de base et scannera deux régions d'en-têtes.

**Il ne mesure pas non plus le coût en bits de cet alignement**, qui est acquis
ailleurs et vaut +0,48 %.

## 5. V0 avant V1 — sans exception

1. Le shader **compile** (`bin/mslcheck`, un point d'entrée à la fois).
2. **Chaque bloc du tirage** — 2^24 — vérifié contre `cns_decode`, tolérance
   `1e-5 · Σ|wᵢxᵢ|`, la forme de `thesis` et non le plancher absolu de
   `decreal`.
3. **Le flux lui-même est déjà prouvé** : `transcode_e1v` a fait son
   aller-retour sur les 150 681 600 blocs du 4B, plus un test de carte des mots
   qui relit les en-têtes depuis les octets bruts sans passer par le lecteur du
   module. Ce n'est pas une référence neuve.
4. **Tout écart enterre le bras sans banc.**

## 6. Ce qui invaliderait ce pré-enregistrement

- si les ancres `sol`, `masques` et `sol-rang` ne se reproduisent pas dans le
  run, aucun chiffre n'est publiable avant d'en connaître la cause ;
- si le bras échoue V0, il n'existe pas ;
- si la sortie du bras n'est pas relue, sa milliseconde ne mesure rien ;
- si le bras rend **mieux que 0,20 ns/bloc**, chercher l'erreur avant d'en
  faire un titre (P1 §5, hérité) — et ici la suspicion est plus forte
  qu'ailleurs, ce bras devant coûter **plus** que `marche-bloc` et non moins ;
- 🚨 **si `simd_prefix_exclusive_sum` n'est pas disponible** sur la version de
  MSL du runtime, le bras est écrit autrement — et alors **il mesure autre
  chose** : une somme préfixe sérielle n'est pas un warp-scan, et le journal
  doit dire laquelle a tourné.

## 7. Écarts au protocole — journal, tenu à chaud

**Aucune entorse.** Rien de ce qui suit ne déplace un seuil, n'ajoute un bras ni
ne retire une vérification ; les six entrées sont des **précisions** et des
**ajouts**, et elles sont écrites ici parce que le §7 existe pour ça.

🚨 **Elles sont écrites AVANT le tampon, et c'est la seule fenêtre où elles
pouvaient l'être.** Un pré-enregistrement horodaté est en lecture seule **pour
toujours** — l'éditer change son SHA256 et invalide le `.ots`, et le ré-ancrer
produirait un horodatage postérieur à la mesure qu'il juge, c'est-à-dire
exactement l'objet sans valeur que le tampon existe pour éviter
([`proofs/README.md`](README.md), qui donne la règle et sa seule exception
jamais consentie).

⚠️ **Donc ce §7 cesse d'être « tenu à chaud » à la seconde du `ots stamp`.** Tout
écart constaté après le tampon va dans le **journal de mesure**
(`docs/mesures/p1c-…`) et dans `proofs/README.md`, jamais ici. Une session
future qui viendrait « compléter le §7 » d'un document scellé détruirait la
seule chose qui fait sa valeur.

**(a) L'inconnue nommée au §6 est levée, et dans le bon sens.**
`simd_prefix_exclusive_sum` **existe** dans le MSL de ce runtime : le shader
compile sur Apple M3 Max, simd 32, max group 1024
(`cargo run --release -p llvq-metal --bin mslcheck`, 2026-08-15). Le bras mesure
donc bien un **warp-scan matériel** et non une somme préfixe sérielle. Le
journal du run doit le dire ; il le peut maintenant.

**(b) Précision de vocabulaire sur un nombre divulgué au §1 — 56 est un
RECORD, pas une charge utile.** Le §1 divulgue « la largeur maximale d'un
record est 56 bits sur 8 champs (classe 323) », et c'est exact :
`CnsLayout::bits()` compte l'en-tête. Mais ce que la somme préfixe additionne
est la **charge utile**, en-tête exclu, donc **46 bits** sur cette même classe
323. Dix bits séparent les deux quantités et seul leur nom les distingue ; une
somme préfixe qui additionnerait des records mettrait dix bits de dérive dans
chaque lane de chaque groupe. Le §1 n'est pas corrigé — il n'est pas faux —
mais le shader, le banc et le test nomment désormais les deux séparément.
*Trouvé par un test qui vérifiait le nombre divulgué au lieu de lui faire
confiance.*

**(c) Ajout au §5 : une passe de FIXTURE, en plus du tirage.** Le §5.2 demande
les 2^24 blocs du tirage, et ils seront faits. Mais le 4B publié ne porte
**aucun bloc origine** (le banc l'assertait déjà avant P1c) : l'id 511 — dont la
charge utile est vide, dont l'entrée de table est 0 et non `1+ci`, et qui ne
contribue rien à la somme préfixe — n'est donc atteignable par **aucun** tirage.
S'y ajoutent les 97 entrées de table que le fichier n'habite pas. La fixture
couvre les 383 classes aux deux bouts, l'origine, et **un groupe entier de la
classe la plus large** — la plus grande somme préfixe que l'adressage puisse
subir, qu'un mélange n'atteint jamais. Un ajout ne dispense de rien : le tirage
reste exigé.

**(d) V0 est établi AVANT que le tampon soit dépensé.** Un tampon est une porte
à sens unique, et le §5.4 enterre un bras dont V0 échoue. La passe de fixture
tourne donc dans **`bin/p1v0`** — le binaire dont c'est le rôle, sans garde de
tampon et sans chronomètre — et elle est **verte le 2026-08-15** : 1 280 blocs,
pire écart **2,194e-7** relatif à Σ|w·x|, sur les 383 classes, l'origine et le
groupe le plus large. Le tirage de 2^24 blocs reste dans le banc, comme le §5.2
le demande.

**(e) Ajout au §2 : les DEUX étalons, et leur égalité exigée.** Le §2 nomme
`cns_decode`. Le banc le calcule, calcule aussi `FastDecoder::decode`, et
**exige l'égalité au bit près** sur les 2^24 blocs avant de chronométrer quoi
que ce soit. Pour le prix d'une passe CPU, **C2 de P5 est refait sur les blocs
mêmes qu'on s'apprête à chronométrer** au lieu d'être cité.

**(f) Le garde de tampon du banc couvre désormais trois documents** — P1, P1b et
P1c — au lieu du seul P1. Ajouter un bras ajoute son document au garde, sans
quoi le §0 de ce document ne serait tenu que par la prose.

**(g) Le tampon est posé par délégation, et il ne dit pas qui.** L'opérateur a
délégué le `ots stamp` le 2026-08-15, le §0 le lui attribuant par défaut. Ça ne
change rien à ce que le tampon établit — **l'antériorité du document, pas son
auteur** — et il faut le dire ici plutôt que le laisser supposer : aucune
signature GPG n'existe sur ce document, comme sur aucun autre de ce répertoire
(`proofs/README.md`, point 1). Qui a écrit quoi repose sur git et sur la parole
de l'opérateur ; ce que le `.ots` ajoute, c'est qu'aucune des deux ne peut
antidater le fichier.

### Ce que les mutants ont dit (§5 du dossier, appliqué aux gardes neufs)

| mutant | effet attendu | observé |
|---|---|---|
| une dérive de **texte** dans la région partagée des deux shaders, sémantique inchangée | seul un test de texte peut la voir | `the_two_arms_share_their_helpers_byte_for_byte` **ROUGE** |
| une largeur de charge utile fausse d'**un seul bit** sur une classe | V0 rouge, et la casse doit se propager au **reste du groupe** | **ROUGE, 30 blocs sur 1 280**, pire écart 9,36e-1 relatif — trois occurrences de la classe, chacune empoisonnant les lanes qui la suivent |

Le second est le garde qui compte : il montre que la somme préfixe est
réellement vérifiée par V0, et il en montre la **forme** — un bit de largeur
fausse ne corrompt pas un champ, il désynchronise un groupe.
