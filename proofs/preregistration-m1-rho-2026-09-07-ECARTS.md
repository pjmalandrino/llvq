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

## É5 — Le préreg décrit le fichier nu ; le bras du 2026-09-13 tourne sur le mixte servi

**Ce que le préreg pose** : son §4.6 fixe le témoin à `t0.csv` du chantier 1 —
le **Tetra nu**, 53,49, sur carte, 2 280 questions, empreinte
`65dcd53655e8bfa5` — et son §5 mesure `ΔM = MMLU(meilleur bras) − 53,49`.

**Ce qui change le 2026-09-13**, et c'est l'opérateur qui a ouvert l'axe L01
sur l'objet servi :

1. **Le fichier n'est plus le même.** ρ est appliqué à
   `q4b-tetra-q5.llvq`, l'objet **mixte servi** — 216 enregistrements `Tetra`
   et 36 `v_proj` en int4 g128 — et non au Tetra nu de 252 enregistrements.
   Les 36 int4 n'ont pas de `row_scales` : ils traversent l'outil intacts, ce
   qui est vérifié et compté (*mesuré*,
   [l01-rhoapply-mixte-2026-09-13](../docs/mesures/l01-rhoapply-mixte-2026-09-13.txt)).
   L'outil ne savait pas lire ce fichier avant ce jour : il mourait au
   troisième enregistrement.

2. **Le témoin de 53,49 ne s'applique plus.** Le fichier servi lit **55,52** en
   bras dense sur carte (recensement du 2026-09-11, job `6aa414dd`). Le §5 se
   lit donc contre 55,52 et non 53,49. **La règle de décision du §5 est
   conservée dans sa forme** — les trois seuils +1,0 / +0,43 / sous la barre —
   parce que c'est une règle sur un **écart apparié**, pas sur un niveau.

3. **Le device change : Metal, pas L40S.** Le dépôt a mesuré le device à
   −1,18 pp de MMLU à fichier constant, donc apparier un bras Metal contre le
   dump carte de 55,52 confondrait le correctif et la carte. **Le témoin est
   donc re-mesuré sur Metal dans les mêmes conditions que les bras**, et c'est
   un troisième bras que le préreg ne prévoit pas. Les trois bras partagent le
   plan d'échantillonnage et l'empreinte du §4.1.

4. **ρ_global vaut 0,929433 et non 0,929234.** Le second est mesuré sur les 252
   enregistrements du fichier nu, le premier sur les 216 du servi
   (*mesuré*, 1 069 056 lignes). L'écart à ρ\* de M0 reste au-dessus du seuil
   de 0,020 du §4.4 : |Δ| = 0,031312, et il est publié comme le §4.4 l'exige.

**Ce que ça ne change pas** : les contrôles du §4 sont tous tenus sur le
nouveau fichier — même taille à l'octet, idempotence à ρ = 1 **identique octet
pour octet**, ρ_global publié à côté de celui de M0. La barre reste **0,43 pp**,
puisque les trois bras vivent sur le même fichier aux échelles de ligne près.

**Ce qui reste faux si on l'ignore** : rien de ce bras ne peut être comparé aux
chiffres publiés du bras M1 de 2026-09-07, qui vivent sur un autre fichier et
une autre carte.

## É6 — Le bras B a été relancé alors que É1 disait qu'il mourrait

**Le fait** : le bras B du 2026-09-13 rend **29,79 de MMLU quand le hasard est à
25,0**, contre 55,70 pour le témoin. Δ = −25,91 pp, IC95 [−29,10 ; −22,62],
McNemar p = 2,3e−86, 44,6 % de discordantes.

**C'est É1 rejoué à l'identique sur un autre fichier.** É1 établit que la
formule du §2 est fausse — la ligne décodée est affine en ρ, pas linéaire — et
que le bras B de 2026-09-07 a rendu 28,11 pour cette raison. `rhoapply` calcule
la forme corrigée `ρ̃_i` et la publie comme diagnostic, mais **écrit `ρ_i`**,
parce que le §2 tamponné fixe la formule. Le bras était mort avant de partir.

Coût : ~70 min de Mac, 0,00 $. Aucune carte. Ce qui est perdu est du temps, pas
de l'argent, et la faute est de conduite, pas d'outil : É1 était dans ce fichier
et a été lu le jour même.

**Ce que ça n'ouvre pas.** É3 a déjà tranché la question que ce bras aurait
posée : avec la formule exacte, `B'` et `A` rendent le même modèle à **0,04 pp**
(IC95 [−1,17 ; +1,05], McNemar p = 0,922). Relancer un bras B sur `ρ̃_i`
re-mesurerait M1b. **L'axe ligne reste un seul nombre**, et cette conclusion ne
dépend pas du fichier.

## É7 — Le §5 n'a pas de ligne pour « au-dessus du seuil, mais non résolu »

Le bras A du 2026-09-13 rend **ΔM = +1,23 pp**, IC95 [−0,50 ; +3,06].

La ligne 1 du §5 exige `ΔM ≥ +1,0 pp` **et** `IC95 > 0` : la première condition
est tenue, la seconde non. La ligne 2 couvre `+0,43 ≤ ΔM < +1,0` : ΔM est
au-dessus. La ligne 3 couvre `ΔM < +0,43`. **Aucune ligne ne décrit ce cas.**

Le §5 a été écrit en supposant qu'un effet au-dessus du seuil s'accompagnerait
d'un intervalle qui exclut zéro. À 2 280 questions il ne s'accompagne de rien :
la demi-largeur appariée à fichier constant vaut 1,7 pp ici, pour un effet de
1,23. C'est le même défaut de dimensionnement que
[t3-genou-2026-09-12](../docs/mesures/t3-genou-2026-09-12.txt) a établi sur un
autre chantier, et il n'est pas propre à ce préreg.

**Ce qui s'applique en fait, c'est la ligne 4** — « B pire que A de plus de
0,43 pp → rien n'est décidé » — et elle s'applique pour la raison de É6, pas
pour celle qu'elle nomme.

**Un préreg qui rouvrirait cet axe doit poser son seuil et sa puissance
ensemble.** Sur le split complet la SE appariée est ~0,17 pp et un effet de
1,2 pp se résout ; sur 2 280 questions aucun seuil de 0,43 ou de 1,0 n'est
testable, quel que soit le chiffre obtenu.
