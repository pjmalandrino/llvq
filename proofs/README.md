# `proofs/` — ce qui est opposable à un lecteur

Ce répertoire porte le **petit, le diffable, le porteur** : ce qu'un auditeur
peut lire, hasher et confronter sans accès à notre compte Hugging Face.
Format et raison d'être : [`docs/plan-de-test-v2-cuda.md`](../docs/plan-de-test-v2-cuda.md) §6.

> **Un identifiant de job n'est pas une preuve opposable.**
> `https://huggingface.co/jobs/<user>/<id>` redirige un visiteur anonyme vers
> un formulaire de login. C'est un pointeur pour un auditeur à qui on a donné
> accès — pas une pièce.

## Ce qui vit ici

| fichier | rôle |
|---|---|
| `preregistration-<date>.md` | ce qu'on s'engage à conclure **avant** de mesurer : prédiction, règle de décision chiffrée, table des issues, et ce qui invaliderait l'ensemble |

## Ce qui n'y vit pas

- les **logs bruts**, les dumps MMLU, les `nvidia-smi` : volumineux, ils vont
  au dataset public, cité **par révision figée** et jamais par `/main/` ;
- les **checkpoints déquantifiés** : reproductibles depuis le checkpoint et
  `ops/awq_dequant.py`. On dépose le script et les sha256, pas 8 Go ;
- les **chiffres eux-mêmes** : ils sont dans `docs/data/*.csv`, et c'est de là
  que le papier se régénère.

## L'état honnête de ce répertoire

⚠️ Il vient d'être créé et il est **incomplet par rapport à ce que §6 décrit**.
Manquent, dans l'ordre où ça compte :

1. **La signature et l'horodatage.** §6.5 exige un tag annoté signé GPG *et* un
   `ots stamp` (OpenTimestamps) sur chaque pré-enregistrement. Un commit signé
   prouve *qui*, pas *quand* ; seul OpenTimestamps rend l'antériorité
   vérifiable sans nous faire confiance. **Actions de l'opérateur** — elles
   demandent une clé privée. Sans elles, l'antériorité repose sur une date de
   commit, qui est ré-éditable.
2. 🕳️ **Le manifeste et son `verify` — CORRECTION du 2026-08-10.** Une version
   antérieure de ce fichier disait qu'ils « restent à écrire ». **C'est faux, et
   l'erreur aurait envoyé quelqu'un réécrire un outil qui existe** :
   [`ops/manifest.py`](../ops/manifest.py) implémente déjà `record`, `verify`,
   `report` et `selftest`, avec exactement la règle porteuse — hachage du log et
   de l'objet mesuré, commit et propreté de l'arbre, identifiant
   `[[claim:ID]]`, et surtout `value_evidence()`, qui exige que **la valeur
   déclarée se retrouve littéralement dans son log**. Son message d'échec
   nomme la falsification que le hachage seul ne voit pas : « une valeur a
   quitté son log alors que le log, lui, est intact — l'entrée a été éditée à
   la main après l'enregistrement, et c'est celle qui atteint directement le
   papier ».

   **Ce qui manque n'est donc pas l'outil, c'est son usage :
   `ops/manifest.jsonl` n'existe pas.** Zéro entrée, alors que le papier porte
   une centaine de nombres. L'instrument est bâti, le registre est vide — le
   motif que ce dossier connaît par cœur : un garde-fou écrit et jamais armé.

   ⚠️ Et il vit dans `ops/`, pas ici, alors que §6.3 le range dans `proofs/`.
   Décider **une fois** lequel des deux emplacements fait foi, plutôt que de
   laisser deux conventions coexister.

Tant que le registre est vide, ce répertoire tient pour la partie
pré-enregistrement et documente une intention pour la partie provenance. Le
dire est moins coûteux que de laisser croire l'inverse.
