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
2. **`manifest.jsonl`** — une ligne par run, *y compris les runs ratés*.
3. **`verify.py`** — la règle porteuse : tout nombre du papier marqué
   `[[claim:ID]]`, et un échec de build si l'ID n'a pas de ligne, si un hash ne
   retombe pas, si `git_dirty` est vrai, ou si la valeur du papier diffère du
   JSON. **Un nombre sans trace doit devenir une erreur de build, pas un
   oubli.**

Tant que 2 et 3 n'existent pas, ce répertoire documente une intention pour la
partie manifeste, et tient réellement pour la partie pré-enregistrement. Le
dire est moins coûteux que de laisser croire l'inverse.
