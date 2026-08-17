# `proofs/` — ce qui est opposable à un lecteur

Ce répertoire porte le **petit, le diffable, le porteur** : ce qu'un auditeur
peut lire, hasher et confronter sans accès à notre compte Hugging Face.
Format et raison d'être : [`docs/archive/plan-de-test-v2-cuda.md`](../docs/archive/plan-de-test-v2-cuda.md) §6.

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

1. **L'horodatage est fait ; la signature reste due.** §6.5 exige un tag
   annoté signé GPG *et* un `ots stamp` (OpenTimestamps) sur chaque
   pré-enregistrement. **Les deux pré-enregistrements portent leur `.ots`**
   (`preregistration-2026-08-10.md.ots`, `preregistration-2026-08-11.md.ots`,
   ce dernier ré-ancré après sa correction É0 et avant le job qu'il jugeait),
   donc l'antériorité ne repose plus sur une date de commit ré-éditable et se
   vérifie sans nous faire confiance :
   ```bash
   ots verify proofs/preregistration-2026-08-11.md.ots
   ```
   > 🚨 **CETTE COMMANDE ÉCHOUE AUJOURD'HUI, et sur les DEUX fichiers.**
   > Vérifié le 2026-08-16 : les deux `.md` ont été **édités après leur
   > ancrage**, donc leur SHA256 courant n'est plus celui que le `.ots`
   > atteste. Le paragraphe ci-dessus était vrai le jour où il a été écrit et
   > il ne l'est plus — c'est exactement le motif que ce dépôt documente
   > partout, cette fois dans le fichier qui promet le contraire. **Le détail,
   > les empreintes et ce qui reste opposable : section « L'inventaire des
   > tampons » ci-dessous.** Le répertoire n'a par ailleurs plus deux
   > pré-enregistrements mais **onze**.

   ⚠️ **La signature GPG, elle, n'est pas faite** — elle prouverait *qui* et
   demande une clé privée : action de l'opérateur.

   🚨 **Les deux pré-enregistrements sont désormais GELÉS, et leur en-tête
   ment sciemment.** Chacun s'ouvre sur « ni signé GPG ni horodaté … » :
   c'était vrai à la seconde où il a été écrit, et ça ne l'est plus. Cette
   phrase **ne sera pas corrigée**, parce que corriger le fichier
   changerait son empreinte et invaliderait le `.ots` — et le ré-ancrer
   maintenant produirait un horodatage *postérieur* à la mesure qu'il
   juge, c'est-à-dire exactement l'objet sans valeur qu'on cherchait à
   éviter. **Un pré-enregistrement horodaté est en lecture seule pour
   toujours.** La seule exception jamais consentie est celle du 2026-08-11
   (correction É0), faite le jour même, **avant** le job, et suivie d'un
   ré-ancrage — le §7bis du fichier la consigne, et l'ancien `.ots` reste
   opposable dans l'historique git (`f56ae30`).

   > 🕳️ **Le gel a été enfreint, et il l'a été APRÈS avoir été écrit ici.**
   > Les deux fichiers ont été édités depuis — le 08-11 par la refonte
   > documentaire du 2026-08-12 (`b799c32`, trois réécritures de liens
   > `docs/` → `docs/archive/`), le 08-10 quatre fois. La règle « lecture
   > seule pour toujours » ne vivait que dans cette prose, et une règle qui
   > ne vit que dans la prose se saute — c'est la phrase que le garde de
   > `bin/rankbench` a fini par écrire dans un binaire. Détail et empreintes :
   > section « L'inventaire des tampons ».

   Corollaire pour toute session future : les mises à jour d'état vont
   **ici**, jamais dans un fichier scellé.
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

> ⚠️ **Et « tient pour la partie pré-enregistrement » est trop généreux depuis
> le 2026-08-16.** Sur les onze documents du répertoire, **deux** tiennent la
> promesse entière — tampon posé avant la mesure *et* attestant les octets
> qu'on lit aujourd'hui. L'inventaire ci-dessous dit lesquels, et de quelle
> façon chacun des neuf autres décroche.

## L'inventaire des tampons — vérifié le 2026-08-16

Ce répertoire promet qu'un lecteur peut « lire, hasher et confronter sans
accès à notre compte ». **La promesse ne vaut que pour les lignes ✅
ci-dessous**, et il faut deux colonnes pour la dire, parce qu'un tampon peut
échouer de deux façons indépendantes : être posé **trop tard**, ou attester une
**version qui n'est plus là**.

| pré-enregistrement | `.ots` | posé **avant** sa mesure ? | atteste le **fichier courant** ? |
|---|---|---|---|
| `preregistration-2026-08-10.md` | oui | ✅ oui (`.ots` commité le 08-10 09:55) | ❌ **non** — édité **4 fois** depuis |
| `preregistration-2026-08-11.md` | oui | ✅ oui (ré-ancré après É0, avant le job) | ❌ **non** — édité **1 fois** depuis |
| `preregistration-2026-08-13.md` | **aucun** | ❔ rien ne l'établit — mtime et date de commit seuls | — |
| **`preregistration-p1-2026-08-13.md`** | oui | ✅ **oui** — tampon avant la première milliseconde | ✅ `5109b35f…` |
| `preregistration-p1b-2026-08-15.md` | oui | ❌ **non** — tamponné **après** ses mesures | ✅ `d027c9d2…` |
| **`preregistration-p1c-2026-08-15.md`** | oui | ✅ **oui** — le garde de `rankbench` refuse de démarrer sans lui | ✅ `5b2ccc3b…` |
| `preregistration-p2-2026-08-14.md` | **aucun** | — | — |
| `preregistration-p3-2026-08-14.md` | **aucun** | — | — |
| `preregistration-p4-2026-08-14.md` | **aucun** | — | — |
| `preregistration-p5-2026-08-14.md` | oui | ❌ **non** — tamponné **après** ses mesures | ✅ `3b45b450…` |
| `preregistration-e1v-cuda-2026-08-15.md` | **aucun** | ❌ non — **décision explicite de l'opérateur** | — |

**Ce que ça donne, en clair.** **Deux** documents seulement — `p1` et `p1c` —
tiennent la promesse entière : tampon posé avant la mesure *et* attestant les
octets qu'on lit aujourd'hui. `p1b` et `p5` ont un tampon intègre mais **posé
après** leurs mesures : il prouve que le document n'a pas bougé *depuis*, pas
qu'il précédait le chiffre — leurs journaux portent la dette et la déclarent
(`p5-cns-2026-08-15.txt` : « ce document n'est PAS horodaté … P1, lui, a été
tamponné AVANT sa mesure »). Les deux ancêtres du 08-10 et du 08-11 sont dans
le cas inverse : posés à temps, mais **l'objet attesté n'est plus dans l'arbre
de travail**.

### Le défaut du 08-10 et du 08-11, et comment le vérifier hors ligne

Un `.ots` atteste un SHA256, pas un nom de fichier. Éditer le `.md` — même pour
réécrire un lien — détache l'ancre. La vérification ne demande **aucun réseau** :

```bash
ots info proofs/preregistration-2026-08-11.md.ots | head -1   # empreinte ancrée
shasum -a 256 proofs/preregistration-2026-08-11.md            # empreinte courante
```

| document | empreinte **ancrée** | empreinte **courante** | la version ancrée est |
|---|---|---|---|
| 2026-08-10 | `21aa2f97a3fd7814…` | `fa3140b444296220…` | `git show 8a25792:proofs/preregistration-2026-08-10.md` |
| 2026-08-11 | `1903c32cedea94e3…` | `01a5c4d246964fcd…` | `git show 7678348:proofs/preregistration-2026-08-11.md` |

Les deux blobs git ci-dessus **rendent exactement l'empreinte ancrée** —
vérifié. Donc rien n'est perdu : ce qui est opposable est la **version
historique**, pas le fichier courant, et il faut le dire au lecteur plutôt que
de lui donner une commande qui échoue.

🕳️ **La cause n'est pas une négligence isolée, et c'est ce qui la rend
intéressante.** Le 08-11 a été détaché par la **refonte documentaire du
2026-08-12** (`b799c32`), qui réécrivait `docs/…` en `docs/archive/…` partout
— trois lignes de lien dans un fichier scellé. Le commit contrôlait « 0 lien
cassé, cargo check vert, make check du papier vert » ; **rien ne contrôlait les
ancres**, parce qu'un `.ots` n'est ni un lien ni du code. Le 08-10 avait déjà
été édité trois fois avant ça, dont une par un commit dont le titre est
« Un pré-enregistrement qui ne consigne pas ses entorses ne vaut rien » : la
règle du **gel** existait, le §1 ci-dessus l'énonce, et elle a été enfreinte
avant d'être écrite.

⏳ **Ce qui reste à décider, et ça revient à l'opérateur.** Ré-ancrer serait la
mauvaise réponse — le §1 le dit : un ré-ancrage postérieur à la mesure produit
exactement l'objet sans valeur qu'un tampon existe pour éviter. Les voies
possibles, non tranchées ici : déposer à côté du `.ots` un fichier
d'empreinte + révision git de la version ancrée ; ou geler la version ancrée
sous un nom propre et renvoyer le fichier vivant vers elle ; ou automatiser un
garde qui casse le build quand une empreinte ancrée cesse de correspondre — la
forme qu'a prise le garde de `rankbench`, qui est le seul mécanisme du dépôt à
avoir tenu. **Aucune de ces trois n'est retenue à ce jour.**
