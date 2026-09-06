# Pré-enregistrement — chantier 3 : les 5,1 points d'écart au papier viennent-ils du codebook ?

**Écrit, commité et TAMPONNÉ le 2026-09-06, AVANT la première seconde d'encodage.**
Go de l'opérateur du 2026-09-06. Ligne 3 de
[`docs/ROADMAP-QUALITY.md`](../docs/ROADMAP-QUALITY.md).

Ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md`.

**Coût : 0,00 $ pour l'encodage** (~4 h 30 sur le Mac), puis **~0,45 $** pour
l'évaluation sur `l40sx1`. Aucun plafond en vigueur (opérateur, 2026-09-06).
Total projet avant ce travail : 99,88 $.

Code mesuré : arbre à `a88f126`, sur un `git worktree` isolé — le dépôt
principal est en cours de modification par le chantier d'intégration du kind
int4, et l'encodage ne doit pas lire un fichier à moitié écrit.

---

## §1 — La question, et pourquoi aucune piste de la roadmap ne la pose

Nous lisons **16,94 de perplexité pour 55,59 de MMLU** ; le papier lit **17,05
pour 60,7** sur le même Qwen3-4B. **Nous sommes meilleurs en perplexité et
5,1 points moins bons en MMLU.** Le harnais est innocenté : notre f16 rend
70,32 contre 70,2 au papier, et il rejoue à l'identique sur quatre moteurs.

Le fait que la roadmap ne citait pas, et qui rouvre la question
(`docs/mesures/gain-ab-gate-0.6b-2026-08-25.txt`, ADDENDUM 2, tables du papier
relues) : **les deux configurations LLM du papier sont Λ₂₄(13) + 0 bit de gain
et Λ₂₄(11) + 2 bits. La nôtre, `cap12 + 1 bit`, n'apparaît dans AUCUNE table LLM
du papier.** Le 60,7 est la ligne `leech0c13`, et ce codebook n'a jamais été
encodé au 4B.

Les pistes Q1 à Q7 cherchent toutes de la qualité absolue en partant de notre
pipeline. Aucune ne teste une différence **documentée** entre notre objet et le
sien. Celle-ci en teste une, et elle coûte une nuit de Mac.

## §2 — Le run, verbatim

```
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_ARTIFACT=<...>/q4b-leech0c13.llvq \
LLVQ_THREADS=12 nice -n 10 \
  smoke 64 2048 12 4096 metal nogs leech0c13 999 rot
```

Une seule variable contre le Tetra du 2026-09-06 : le codebook. Même modèle,
même corpus C4, même volume de 64 × 2048 = 131 072 tokens, même mode `nogs`,
même rotation, même device, même limite. Puis `seal`, puis un job `l40sx1` :
perplexité et MMLU avec `LLVQ_MMLU_DUMP`, empreinte `65dcd53655e8bfa5`.

## §3 — Le témoin n'est pas nécessaire ici, et voici pourquoi

La dérive d'encodeur du 2026-08-26 ne touche que les comparaisons contre le
**fichier publié**, qui lui est antérieur. `leech0c13` encodé aujourd'hui et
`Tetra` encodé le 2026-09-06 partagent le même état du code. **Leur écart est
sans dérive par construction.**

Le témoin `leech1c12` reste dû pour corriger rétrospectivement les chiffres
publiés contre `Planes14`. Il n'est pas un prérequis de ce travail.

## §4 — Ce qui peut rendre ce banc illisible, et il faut le lire au démarrage

🚨 **`leech1c12` et `tetra` écrivent 48 bits par bloc tous les deux, exprès,
pour que les deux bras se comparent à débit constant. `leech0c13` n'a aucune
raison de le faire** : zéro bit de gain, boule m ≤ 13 au lieu de m ≤ 12.

`smoke` imprime `effective rate = X bits/weight` dans les premières minutes.
`leech1c12` donne 2,0702 en débit idéal et 2,1595 sur le fichier scellé.

**Si le débit de `leech0c13` s'écarte de plus de 0,05 b/poids de 2,1595, la
comparaison n'est PAS à bits constants**, et aucune ligne du §5 ne s'applique
telle quelle : le résultat se publie en b/param, et un codebook qui gagne en
dépensant plus de bits ne prouve rien sur le codebook.

## §5 — La règle de décision, posée AVANT de voir le chiffre

Ce bras **réencode**, donc la barre n'est pas 0,43 pp mais **2,92 pp de MMLU**
(σ de tirage de calibration au 4B) et 5,2 % de perplexité. Un seul tirage. Les
seuils ci-dessous sont donc espacés en conséquence, et l'écart cherché (5,1 pp)
est au-dessus du bruit.

Soit **M** le MMLU micro de `leech0c13`, apparié contre Tetra (53,49) sur les
mêmes 2 280 questions. Repères : papier 60,7 · `Planes14` publié 55,59 ·
f16 70,32.

| | conséquence |
|---|---|
| **M ≥ 58,5** | **Le codebook porte l'essentiel de l'écart.** Le choix servi est rouvert comme décision de format, et le bit de gain fixe de Tetra devient une question de conception et non un acquis. La ligne 3 de la roadmap passe en tête, devant les finetunings. |
| **55,5 ≤ M < 58,5** | **Partiel.** Le codebook porte une part. Rien ne bouge sur un seul tirage : un second tirage précède toute décision de format. |
| **M < 55,5** | **Le codebook est innocenté.** L'écart au papier est notre pipeline — volume de calibration, composition du corpus, rotation. Les chantiers de calibration et de finetuning prennent la tête de la roadmap, et le débat sur le codebook servi se ferme. |
| **autrement** — débit hors de 2,1595 ± 0,05, ou M > 60,7 | **Rien n'est décidé sur le codebook.** Le premier cas se republie en b/param avant toute lecture. Le second dirait que nous battons le papier avec son propre codebook, donc que l'écart est ailleurs et que ce banc a changé deux choses sans le savoir : il partirait aux écarts. |

## §6 — Divulgation datée, et la prédiction signée

Connu à la signature : Tetra 53,49 / 16,1569 · `Planes14` publié 55,59 / 16,9422
· f16 70,32 / 12,2369 · papier `leech0c13` 60,7 / 17,05 · papier `leech1c12`
absent de ses tables LLM · au 0,6B, `leech0c13` contre `leech1c12` est un pile
ou face, −9,56 % de perplexité au tirage 1 et **+16,5 % au tirage 2**
(`gain-ab-gate-0.6b-2026-08-25.txt`).

**Prédiction de l'auteur, opposable : M entre 53 et 57, centre 55. Donc la
troisième ligne du §5, ou la deuxième par le bas.**

Motif : le papier calibre sur 6 100 séquences là où nous en utilisons 131 072
tokens, un facteur d'environ 95, et ce dépôt a mesuré qu'un bras se déplace de
**13,9 % en ne changeant que le texte de calibration**, à pleine profondeur. Le
volume et le corpus me paraissent porter plus que le codebook.

Ce que vaut cette prédiction : sur ce dossier l'auteur en a signé onze et
**sept étaient fausses**, dont deux le 2026-09-06 même.

**Ce qui la rendrait fausse de façon instructive : M ≥ 58,5.** Cela voudrait
dire que nous servons depuis le début un codebook que le papier ne recommande
pas pour les LLM, que les cinq points se rachètent sans changer une ligne du
pipeline, et que Tetra — qui fixe un bit de gain par construction — hérite du
mauvais côté de ce choix. Le format entier serait à rediscuter, et ce serait la
découverte la plus chère de la semaine.

## §7 — Ce que ce travail ne peut pas être

Une comparaison au papier toutes choses égales. Le corpus, le volume, la
rotation et l'implémentation diffèrent, et ce banc n'en change **qu'une seule**,
le codebook, contre notre propre Tetra. Il dit ce que le codebook vaut **chez
nous**. Il ne reproduit pas le papier et ne prétend pas l'expliquer entièrement.

Et il ne dit rien de Tetra comme format servi : `leech0c13` est un codebook
Ball, il ne déplie pas moins et n'a pas de mot de 48 bits à champs fixes.
