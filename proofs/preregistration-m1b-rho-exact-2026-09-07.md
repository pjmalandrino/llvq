# Pré-enregistrement — M1b : le bras par ligne, avec la formule exacte

**Écrit, commité et TAMPONNÉ le 2026-09-07, AVANT la première mesure.**
Go de l'opérateur du 2026-09-07, après l'effondrement du bras B de M1.

Ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md`.

**Coût : ~0,25 $** (un bras MMLU plus une ppl sur `l40sx1`, plafond 25 min soit
0,75 $ au pire). Aucun réencodage, l'artefact et l'outil existent.
Total projet : 120,46 $.

---

## §1 — Ce que M1 a mesuré, et pourquoi son bras B ne répond pas

M1 rend (*mesuré*, `docs/mesures/m1-rho-2026-09-07.txt`) :

```
  A, toutes les échelles × ρ_global = 0,929234   MMLU 55,14   ppl 17,6981
  B, chaque échelle × son ρ_i                    MMLU 28,11   ppl 67,53
  Tetra, référence                               MMLU 53,49   ppl 16,1569
```

Le hasard est à 25,0 : **B a détruit le modèle**, alors qu'il est meilleur que A
sur l'objectif (‖ΔW‖² 0,11959 contre 0,11981). La quatrième ligne du §5 de M1
s'est appliquée et rien n'a été décidé.

**La cause est une faute de formule, la mienne.** Le §2 de M1 pose
`ρ_i = ⟨w_i, r_i⟩ / ⟨r_i, r_i⟩`. Or la ligne décodée est **affine** en ρ et non
linéaire : `r_i(ρ) = ρ·B_i + T_i`, où `T_i` est la queue `KeepExact`, que
`row_scales` ne met pas à l'échelle. La formule de M1 range donc `T_i` des deux
côtés du quotient.

Ce n'est pas un détail d'arrondi. `llvq-quant/src/gptq.rs:56-59` dit que la
queue **reçoit** le retour d'erreur de tous les blocs antérieurs et l'absorbe
exactement, donc `⟨w_i, T_i⟩ ≠ ‖T_i‖²`, et la formule replie cette compensation
dans le numérateur. Une revue adverse l'a chiffré avant le lancement :
**écart-type des `ρ_i` 0,035584 avec la formule du préreg, 0,015761 avec la
formule exacte.** Plus de la moitié de l'étalement publié par M1 est un artefact,
et le minimum à 0,261 en fait probablement partie.

## §2 — La formule, dérivée

Minimiser `‖w_i − (ρ·B_i + T_i)‖²` en ρ donne

```
  ρ̃_i = ⟨w_i − T_i , B_i⟩ / ⟨B_i , B_i⟩
```

C'est l'optimum exact des moindres carrés à codes gelés, cas `H = I`, pour la
seule quantité que `row_scales` peut multiplier.

Elle se recoupe : `ρ̃` doit rendre `‖ΔW‖²(ρ̃) ≤ ‖ΔW‖²(ρ de M1)` ligne par ligne,
puisqu'elle minimise exactement ce que l'autre approxime.

## §3 — Le protocole

Le même outil, `llvq-bench/examples/rhoapply.rs`, avec la formule du §2. Un
artefact neuf où seul `row_scales` diffère, puis `seal`, puis un job `l40sx1` :
MMLU avec `LLVQ_MMLU_DUMP`, empreinte `65dcd53655e8bfa5`, et une perplexité.

Un seul bras de traitement. Les témoins existent : `t0.csv` pour Tetra nu et
`m1-a.csv` pour le bras A, tous deux sur `l40sx1`, mêmes questions.

## §4 — Les contrôles, et si l'un tombe aucun chiffre n'est publié

1. Empreinte `65dcd53655e8bfa5`, 2 280 questions.
2. L'artefact fait **980 791 242 octets**, comme la source, et ne diffère d'elle
   que dans les zones `row_scales`, vérifié octet par octet.
3. Idempotence : `ρ̃ = 1` partout rend un fichier identique au sha256 près.
4. **`sd(ρ̃) ≈ 0,0158` et `min(ρ̃)` sont publiés.** Si `sd` dépasse 0,020, la
   formule n'est pas celle du §2 et rien n'est publié.
5. `‖ΔW‖²(ρ̃) ≤ ‖ΔW‖²(B de M1) ≤ ‖ΔW‖²(A) ≤ ‖ΔW‖²(original)` : quatre
   inégalités que l'arithmétique impose.

## §5 — La règle de décision, posée AVANT de voir le chiffre

Soit **M' = MMLU(B′)**, apparié. Repères : Tetra 53,49 · A 55,14 · barre 0,43 pp.

| | conséquence |
|---|---|
| **M' ≥ 56,14 et IC95(M' − A) > 0** | **Il existe une structure par ligne réelle.** L'axe ligne du chantier 14 survit, et la forme close par ligne entre dans l'encodeur servi. |
| **\|M' − 55,14\| < 1,0** | **Aucune structure par ligne.** L'axe ligne se réduit au constant global de A, et la moitié ligne du chantier 14 se ferme sur son +1,65 pp. C'est ce que M0 prédisait sur blocs simulés. |
| **M' ≤ 54,14** | **L'objectif `‖ΔW‖²` reste un mauvais proxy par ligne, même avec la formule exacte.** Découverte sur l'objectif et non sur l'axe ligne : toute optimisation par ligne non pondérée par la hessienne est disqualifiée, ce qui touche aussi les pistes 13 et 18 de la roadmap. |
| **autrement** — M' sous 40, ou un contrôle du §4 qui tombe | **L'outil est suspect**, pas le résultat. Un second effondrement avec la formule exacte dirait que le défaut n'était pas la formule, et il faudrait le trouver avant toute autre mesure. |

## §6 — Divulgation datée, et la prédiction signée

Connu : tout le §1, plus M0 qui mesure `ρ*` identique à la sixième décimale de 6
à 1 600 blocs par ligne sur blocs simulés, et dont le témoin sur blocs mélangés
ne trouve aucune structure.

**Prédiction de l'auteur, opposable : M' entre 54,4 et 55,9, donc la DEUXIÈME
ligne du §5. `sd(ρ̃)` entre 0,015 et 0,017, `min(ρ̃)` au-dessus de 0,70.**

Motif : M0 n'a trouvé aucune structure par ligne, et la formule exacte retire
plus de la moitié de l'étalement observé par M1. Ce qui reste devrait ressembler
au constant global.

Dix-sept prédictions signées sur ce dossier, huit fausses.

**Ce qui la rendrait fausse de façon instructive : M' ≤ 54,14.** Cela dirait que
même l'optimum exact par ligne nuit, donc que `‖ΔW‖²` non pondéré est
disqualifié comme critère par ligne — et cela condamnerait d'avance les pistes
qui l'utilisent, à commencer par la requalification de la queue par saillance.

## §7 — Ce que ce banc ne peut pas être

Une mesure de la forme pondérée par la hessienne. Il minimise `‖ΔW‖²`, pas
`Tr(ΔW H ΔWᵀ)`, et M1 vient de montrer que la différence n'est pas académique.
