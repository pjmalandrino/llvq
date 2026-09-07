# Pré-enregistrement — le corpus de calibration du papier explique-t-il l'écart ?

**Écrit, commité et TAMPONNÉ le 2026-09-07, AVANT la première seconde d'encodage.**
Go de l'opérateur du 2026-09-07.

Ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md`.

**Coût : ~14 $** (un encodage 4B sur `rtx-pro-6000x2`, plafond de timeout 200 min
soit 18,33 $ au pire) plus ~0,25 $ d'évaluation. Aucun plafond en vigueur.
Total projet avant : 120,64 $.

---

## §1 — Le fait, lu dans le papier le 2026-09-07 et jamais transcrit avant

Page 7, section 5, première phrase :

> *« For empirical results, we compute (GPTQ-style) layer-wise Hessians on
> **6,100 sequences from DCLM-edu** (Li et al., 2024; Allal et al., 2025),
> matching the calibration set size used in prior work (Tseng et al., 2024a). »*

**Nous calibrons sur `wikitext2` ou `c4`.** Le mot DCLM n'apparaît nulle part
dans les 324 lignes de `docs/llvq-paper-notes.md` : ce dépôt n'a jamais su sur
quoi le papier calibre. Nos notes ne transcrivent aucune ligne de la section 5
avant la table des résultats.

## §2 — Pourquoi ce corpus-là, et pas seulement plus de tokens

Notre écart au papier est une **dissociation**, pas un déficit : nous lisons
**16,94 de perplexité pour 55,59 de MMLU** là où il lit **17,05 pour 60,7**.
Meilleurs sur l'une, 5,1 points moins bons sur l'autre.

Une hessienne est une statistique du texte de calibration : elle dit quelles
directions d'activation comptent. `c4` et `wikitext2` sont du web générique et
de l'encyclopédie — ils alignent le proxy local sur la vraisemblance de ce
texte-là. **DCLM-edu est du web filtré sur du contenu éducatif**, c'est-à-dire
le domaine de MMLU.

Une calibration alignée sur le domaine d'évaluation produirait exactement notre
signature en négatif. C'est la seule hypothèse restante qui explique **le signe**
de l'écart et pas seulement sa taille.

Ce qui a déjà été éliminé : le **codebook** (`leech0c13` encodé chez nous rend
+1,19 pp avec un IC contenant zéro — sous réserve du régime, voir §7) et le
**volume seul** (×32 rend +2,98 pp, soit 58 % de l'écart au mieux).

## §3 — Le bras, et son témoin déjà payé

Un seul bras de traitement, et **le témoin existe** : `V32`, encodé le
2026-09-07, 4 194 304 tokens de C4, `rtx-pro-6000x2`, **MMLU 55,29**, ppl
15,8620, dump `mmlu-v32.csv` dans le bucket.

```
  bras : smoke 2048 2048 12 4096 cuda nogs tetra 999 rot
         LLVQ_CALIB=dclm-edu, 2048 × 2048 = 4 194 304 tokens
  puis seal, ppl, et mmlu avec LLVQ_MMLU_DUMP
```

**Une seule variable contre V32 : le corpus.** Même modèle, même volume, même
codebook, même mode, même rotation, même flavor, même image, même graine.

Le shard `data/000_00000.parquet` de `HuggingFaceTB/dclm-edu` porte 776 000
lignes et environ 1 169 M tokens (*mesuré*, 2026-09-07) : 280 fois le besoin.
Sa révision est épinglée comme `corpus.rs` l'exige.

## §4 — Les contrôles, et si l'un tombe aucun chiffre n'est publié

1. Empreinte de tokens `65dcd53655e8bfa5`, 2 280 questions.
2. Débit effectif **2,0702 b/poids**, comme Tetra et comme V32. Le corpus ne
   touche pas le format ; un écart ici invaliderait le banc.
3. `verify_artifact` relit les 3 633 315 840 poids bit pour bit.
4. Le journal du run imprime **le corpus, le nombre de fenêtres et le nombre de
   tokens** : `dclm-edu, 2048 × 2048 = 4194304 tokens`. Une ligne qui dirait
   `c4` invaliderait tout.
5. La lecture bornée du parquet imprime **combien de groupes de lignes elle a
   lus sur le total** et **combien de caractères contre le budget**. Une lecture
   qui atteindrait la fin du fichier dirait que le budget est mal dérivé.
6. La perplexité mesurée en cours d'encodage doit prédire celle de la carte à
   moins de 0,5 % — elle l'a fait quatre fois.

## §5 — La règle de décision, posée AVANT de voir le chiffre

Ce bras **réencode**, donc la barre est **2,92 pp de MMLU** et 5,2 % de
perplexité. Un seul tirage.

Soit **ΔC = MMLU(dclm-edu) − 55,29**, apparié contre `mmlu-v32.csv`.

| | conséquence |
|---|---|
| **ΔC ≥ +2,0 pp et IC95 > 0** | **Le corpus porte une part réelle de l'écart au papier.** `dclm-edu` devient le corpus de calibration servi, la ligne 6 de la roadmap qualité passe en tête, et la question du corpus est close en notre faveur pour 14 $. |
| **+0,5 ≤ ΔC < +2,0** | **Effet réel mais petit.** On adopte le corpus, parce qu'il ne coûte rien à servir, mais l'écart au papier reste majoritairement ailleurs et la roadmap ne bouge pas. |
| **−0,5 < ΔC < +0,5** | **Le corpus n'explique rien.** Après le codebook et le volume, la troisième cause candidate tombe. Ce qui reste est la **mécanique du pipeline** : GPTQ sphérique, politique de gain, métrique de recherche. La roadmap change de tête. |
| **ΔC ≤ −0,5** | **Le corpus nuit**, et c'est une information forte : la calibration alignée sur le domaine d'évaluation serait une mauvaise idée chez nous, contre l'intuition et contre le papier. À comprendre avant tout autre bras de calibration. |

Lecture secondaire, sans porte : **si ΔC est positif ET que la perplexité se
dégrade**, c'est la signature de la dissociation, et elle confirmerait le
mécanisme du §2 plutôt que le seul résultat. Si les deux s'améliorent, le gain
est générique et le §2 n'est pas démontré.

## §6 — Divulgation datée, et la prédiction signée

Connu : Tetra 53,49 / 16,1569 · V1c 52,31 / 17,6681 · **V32 55,29 / 15,8620** ·
`Planes14` publié 55,59 / 16,9422 · `leech0c13` 54,67 / 19,6093 · f16 70,32 /
12,2369 · le papier 60,7 / 17,05 · sous-additivité calibration × précision
0,562 · un bras se déplace de 13,9 % en ne changeant que le texte de calibration
(*mesuré*, 0,6B, `gain-ab-gate-0.6b-2026-08-25.txt`).

**Prédiction de l'auteur, opposable : ΔC entre 0 et +3 pp, centre +1,2, donc la
DEUXIÈME ligne du §5. Perplexité entre +2 et +10 %, donc dégradée.**

Motif : l'argument d'alignement de domaine est le seul qui explique le signe de
l'écart, et le déplacement de 13,9 % mesuré au 0,6B montre que le texte de
calibration a un levier réel. Mais ce dépôt a mesuré quatre fois que la
perplexité et le MMLU ne bougent pas ensemble, et trois causes candidates sur
trois ont déjà rendu moins que leur promesse.

Dix-huit prédictions signées sur ce dossier, huit fausses.

**Ce qui la rendrait fausse de façon instructive : ΔC sous −0,5.** Cela dirait
que calibrer sur le domaine d'évaluation nuit, ce qui contredirait le papier sur
son propre protocole, et il faudrait comprendre avant d'ouvrir un autre bras.

## §7 — Ce que ce banc ne peut pas être

Une reproduction du papier. Il change **une** variable contre notre propre V32.
Le papier diffère encore de nous par le GPTQ sphérique (leur défaut, notre
absence), la politique de gain, et le volume — 6 100 séquences chez eux, dont la
longueur n'est pas donnée page 7.

Et il ne rejuge pas `leech0c13` : ce codebook a été mesuré sous GPTQ euclidien,
c'est-à-dire dans le régime où le papier lui-même le donne perdant. Ce verdict
est en suspens, pas acquis.
