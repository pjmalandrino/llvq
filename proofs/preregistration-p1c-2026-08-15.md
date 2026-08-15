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

**Aucune entorse à ce jour.** Ce document est écrit avant la première ligne du
shader neuf.
