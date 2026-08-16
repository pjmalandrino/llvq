# Pré-enregistrement — E1v sur CUDA : le layout servable, mesuré dans la comptabilité du banc servi

**Date : 2026-08-15, nuit.** Écrit **avant la première milliseconde CUDA** d'E1v
et avant l'écriture de son hôte. Le noyau (`llvq-cuda/kernels/llvq_e1v.cuh`)
existe déjà et n'a jamais tourné ; ce document est écrit en le sachant, et le §1
le divulgue.

> ⚠️ Ni signé ni horodaté tant que l'opérateur ne l'a pas fait
> (`ots stamp proofs/preregistration-e1v-cuda-2026-08-15.md`). Le tampon prouve
> l'antériorité, pas l'auteur. Une fois posé, **ce document est en lecture seule
> pour toujours**, §7 compris — c'est la règle de `proofs/README.md`, et les
> écarts postérieurs vont au journal de mesure.

---

## 0. Pourquoi ce document existe, et ce qu'il refuse

P1c a mesuré le décodeur d'E1v **sur Metal, sans matvec, un bloc par lane**, et
il a rendu **0,6795 ns/bloc**, vert contre 1,50. Ça établit que le décodeur
n'est pas mort. Ça n'établit rien sur le chemin servi : aucun tok/s, aucune
bande passante, aucun rapport à FP16, et pas la même machine.

🚨 **Et le §4.2 de P1 est explicite : un bras vert dans ce régime intermédiaire
« n'achète AUCUN bras CUDA — il faut une idée neuve, pas un job ».** Aller sur
carte est donc une **extension délibérée**, décidée par l'opérateur le
2026-08-15, et non une conséquence du verdict de P1c. Ce document existe pour
que cette extension ait des critères écrits avant sa mesure plutôt qu'après.

## 1. Ce qui est connu à la signature — divulgation datée

- **E1v aligné ligne pèse 2,3983 b/poids noyau**, contre 2,3877 en ordre de
  fichier : **+0,478 %**. Les deux sont **mesurés sur les octets écrits** du 4B
  scellé, dans un seul run et une seule comptabilité
  [`llvq-artifact/tests/e1v_format.rs`, 2026-08-15].
- **Le layout servi, `Planes14`, pèse 4,8040 b/poids noyau et rend 2,14×
  [2,11–2,15] contre FP16 à 425 Go/s** ; `Planes12x` 4,342 et 1,98×
  [1,95–1,99] ; `Golay70` v2 3,589 et 1,77×, écarté [publiés].
- **Le décodeur E1v sur Metal rend 0,6795 ns/bloc**, et l'adressage n'y coûte
  que **+1,2 %** sur le proxy à stride fixe [mesuré, P1c].
- 🚨 **Le noyau CUDA ne porte PAS ce corps-là.** Le contrat du crate — *no
  dynamic indexing*, `local_size_bytes() == 0` — interdit les `kinds[24]`
  indexés par slot, donc le `.cuh` transcrit le corps **plat**, que Metal mesure
  **24 % plus lent** (0,8342 contre 0,6711 ns/bloc). **Les deux chiffres ne se
  soustraient pas.**
- **Le noyau parse** (`bin/cuhcheck`) et **son miroir est vert** : 3 008 blocs,
  les 383 classes, 6 origines, créneau par créneau en égalité f32 exacte contre
  `E1vBlocks::decode_block` [`llvq-artifact/tests/e1v_cuda_mirror.rs`].
- **Aucune milliseconde CUDA n'existe pour E1v**, sur aucune carte. Aucun compte
  de registres, aucun profil, aucun hôte.

🚨 Écrits ici pour qu'on ne puisse pas dire après coup qu'ils ont inspiré les
seuils. Les seuils du §3 sont ceux d'X3, datés du 2026-08-12, et ils ne bougent
pas.

## 2. Le bras, figé ici

Un seul bras neuf, **`e1v`**, déjà **enregistré en dernière position** de
`llvq-cuda/src/arms.rs` (index 15) et refusé par nom tant que son drapeau
`HAS_KERNEL` est faux.

| | |
|---|---|
| noyau | `kernels/llvq_e1v.cuh`, corps **plat**, warp-scan `__shfl_up_sync` |
| flux | `transcode_e1v_rows` — la coupe **alignée ligne**, la seule qu'un warp par ligne puisse lire |
| comptabilité | celle de `planesbench` : b/poids **noyau**, rapports formés **round par round** contre `fp16` |
| étalon | la référence f64 partagée du banc, ligne à ligne, comme tout autre layout |

**Plan de phases**, et il n'est pas négociable : `fp16, planes14, planes12x,
slot32, golay70v1, awq` d'abord — la phase publiée, qui reproduit la table de
référence — puis la même **plus `e1v`**. Une seule phase ne produit aucun
Δ_contrôle, donc aucune règle de décision, et rien dans la sortie ne le dirait.

## 3. Les seuils — ceux d'X3, sans amendement

| | seuil | conséquence |
|---|---|---|
| remplacer `Planes14` | ≥ **2,05×** vs FP16 | E1v devient le layout servi |
| remplacer `Planes12x` | ≥ **1,9×** | E1v prend la place du point bits-bas |
| plancher | < **1,6×** | **la ligne se referme** côté CNS, comme elle s'est refermée sur E2 |

Entre 1,6× et 1,9× : **point de courbe débit↔taux, et rien d'autre** — le
statut exact de `Golay70` v2, qui vit dans le dépôt sans être servi.

⚠️ **Il ne faut pas s'attendre à ce que le haut du tableau tire.** E1v décode
deux marches binomiales, un mot de Golay, une réparation de parité et trois
règles de signe là où `Planes14` fait des sélections. Sa raison d'être est la
**mémoire** — 2,3983 contre 4,8040, soit **la moitié** — et le seuil de 1,6×
existe pour dire à partir de quand cette moitié ne s'achète plus.

## 4. Ce que ce banc mesure, et les trois choses qu'il ne mesure pas

**Il mesure** : un matvec fusé sur les formes réelles du 4B, contre `fp16` dans
le même processus et le même round, dans la comptabilité b/poids noyau.

**Il ne mesure pas la qualité** : elle est **inchangée par preuve** — E1v est
une re-bijection de l'archive, balayée sur les 150 681 600 blocs (P5 C2). Aucun
ppl, aucun MMLU n'est à refaire, et aucun ne sera cité comme résultat de ce run.

**Il ne mesure pas le modèle de bout en bout** : ni tok/s, ni Go carte. E1v
n'est câblé dans aucun modèle, et `fusedrun` ne le connaît pas.

**Il ne mesure pas le coût du transcodage** : `transcode_e1v_rows` tourne sur
l'hôte du job, et son temps est un coût de préparation, pas un résultat.

## 5. V0 avant V1 — sans exception

1. Le noyau **parse** (`bin/cuhcheck`) — fait.
2. Le **miroir** est vert sur les 383 classes et l'origine — fait.
3. **Sur la carte**, avant tout chronométrage : le décodeur est comparé à la
   référence Rust sur chaque ligne des matrices du modèle, comme les quatre
   `*_decoder_matches_rust.rs` existants. **C'est la seule des trois qui exerce
   le vrai `__shfl_up_sync`**, et les deux premières ne la remplacent pas.
4. **`local_size_bytes() == 0`**, lu par `preflight`. Ce n'est pas une
   formalité : le corps plat a été choisi **contre** un corps 24 % plus rapide
   sur Metal précisément pour ça. S'il déborde quand même, le choix était faux
   et le chiffre mesure autre chose que ce que ce document croit mesurer.
5. **Tout écart enterre le bras sans banc.**

## 6. Ce qui invaliderait ce pré-enregistrement

- si les six bras de la phase de contrôle ne reproduisent pas la table publiée,
  aucun chiffre n'est publiable avant d'en connaître la cause ;
- si le flux transcodé sur la carte ne pèse pas **2,3983 b/poids noyau** aux
  5e-4 près, l'hôte n'écrit pas le flux que ce document nomme ;
- si `local_size_bytes()` n'est pas nul (§5.4) ;
- si le bras rend **mieux que 2,2×**, chercher l'erreur avant d'en faire un
  titre : ce serait battre `Planes14` en décodant strictement plus, ce qu'aucune
  lecture de source ne prédit ;
- 🚨 si le job tourne **en une seule phase**, il ne produit aucun Δ_contrôle et
  aucune de ces règles ne s'applique.

## 7. Écarts au protocole — journal, tenu à chaud jusqu'au tampon

**Aucun.** Ce document est écrit avant l'hôte, avant le bras, et avant toute
milliseconde CUDA.

⚠️ Ce §7 cesse d'être modifiable à la seconde du `ots stamp`. Tout écart
constaté ensuite va dans le journal de mesure et dans `proofs/README.md`, jamais
ici.
