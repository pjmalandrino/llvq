# Écarts au pré-enregistrement M1

> Le pré-enregistrement
> [`preregistration-m1-rho-2026-09-07.md`](preregistration-m1-rho-2026-09-07.md)
> (sha256 `ac83cdc2…`) est tamponné. Il ne s'édite pas.

## É1 — La formule du §2 est fausse, et son bras a détruit le modèle

**Ce que le préreg pose** : `ρ_i = ⟨w_i , r_i⟩ / ⟨r_i , r_i⟩`.

**Ce qui est vrai** : la ligne décodée est **affine** en ρ et non linéaire.
`r_i(ρ) = ρ·B_i + T_i`, où `T_i` est la queue `KeepExact`, que `row_scales` ne
met pas à l'échelle — `reconstruct_shape_gain` est le seul consommateur de
`row_scales` et ne voit pas la queue. La formule range donc `T_i` des deux
côtés du quotient.

Ce n'est pas un arrondi. `llvq-quant/src/gptq.rs:56-59` dit que la queue
**reçoit** le retour d'erreur de tous les blocs antérieurs et l'absorbe
exactement, donc `⟨w_i, T_i⟩ ≠ ‖T_i‖²`, et la formule replie cette compensation
dans le numérateur.

**Le coût** : le bras B rend **28,11 de MMLU quand le hasard est à 25,0**, et
67,5 de perplexité. Un job de 0,38 $ dont la moitié est inexploitable.

**Ce qui l'a rattrapé** : une revue adverse avait chiffré l'écart **avant** le
lancement — sd 0,035584 contre 0,015761 pour la formule exacte — et l'avait
classée « déviation qui appartient au `-ECARTS.md` ». Elle a été lue et n'a pas
arrêté le lancement, parce que le préreg était déjà tamponné et que le §2 fixe
la formule. Le processus a fonctionné pour l'enregistrer, pas pour l'empêcher.

**La correction** : M1b, préreg
[`preregistration-m1b-rho-exact-2026-09-07.md`](preregistration-m1b-rho-exact-2026-09-07.md),
avec `ρ̃_i = ⟨w_i − T_i , B_i⟩ / ⟨B_i , B_i⟩`. Le bras B' rend **55,09**, et
toute l'anomalie `gate_proj` du bras B — écart-type cinq fois les autres types,
minimum à 0,261 — disparaît : sd 0,017211, minimum 0,766655.

## É2 — La prédiction signée est fausse sur ses deux bornes

Le §6 prédit `ΔM` entre 0 et +1,0 avec un centre à +0,3, et `ρ_global` réel
entre 0,955 et 0,968. Mesuré : **+1,65 pp** et **0,929234**.

Le motif de la seconde borne était que le biais radial vaut `1 − cos θ`, mesuré
à 0,039 par M0. Le biais réel est de **7,1 %**, parce que le désaccord d'échelle
de la production — les échelles fixées sur la ligne originale avant la boucle,
puis des résidus dont les normes ont dérivé — s'ajoute au `1/cos θ`. M0 nommait
cette limite dans son propre journal ; le §6 de M1 ne l'a pas reportée dans son
motif.

Le contrôle 4 du §4, qui exigeait de publier `ρ_global` à côté du 0,960745 de M0
et déclarait qu'un écart de plus de 0,02 irait aux écarts, a fonctionné.

## É3 — La quatrième ligne du §5 s'est appliquée, et elle était bien écrite

Le §5 prévoit « B pire que A de plus de 0,43 pp → rien n'est décidé », en
ajoutant que ce cas dirait que `‖ΔW‖²` est un mauvais proxy du MMLU par ligne.

C'est arrivé, à 27 points d'écart. Mais la cause n'est pas celle que la ligne
nommait : ce n'est pas l'objectif qui est un mauvais proxy par ligne, c'est la
formule qui n'optimisait pas l'objectif annoncé. M1b le tranche : avec la
formule exacte, B' et A rendent le même modèle à **0,04 pp**, McNemar p = 0,922,
et 4,6 % de questions discordantes seulement.

**La conclusion du chantier reste celle de M0** : l'axe ligne est un seul
nombre. Elle est maintenant établie sur les vrais poids et non plus seulement
sur des blocs simulés.

## É4 — Le correctif dégrade la perplexité, ce qu'aucune ligne du §5 n'anticipait

`A` : MMLU +1,65 pp et perplexité **+9,54 %**. `B'` : +1,61 pp et **+10,63 %**.

Le préreg ne pose aucun seuil sur la perplexité et n'en demande pas. Elle a été
mesurée parce que le job la produisait. C'est la quatrième dissociation du
dossier, et la première où une correction qui rapproche les poids de la vérité
au sens de `‖ΔW‖²` améliore le MMLU en dégradant la perplexité.

Un préreg qui adopterait ce correctif devra poser une règle sur les deux
métriques, pas sur une seule.
