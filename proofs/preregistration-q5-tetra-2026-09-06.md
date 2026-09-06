# Pré-enregistrement — chantier 1 : le gain de `v_proj` transpose-t-il à Tetra ?

**Écrit, commité et TAMPONNÉ le 2026-09-06, AVANT la première mesure des bras
de traitement.** Go de l'opérateur du 2026-09-06 (« go chantier 1 »), ligne 1 de
[`docs/ROADMAP-QUALITY.md`](../docs/ROADMAP-QUALITY.md), sanctuarisée le même jour.

Ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md` à côté.

**Coût : 0,00 $.** Tout tourne sur le Mac. Trois bras de ~38 min, soit ~2 h de
M3 Max. Vague 2 : 1,72 $ dépensés sur 2,00 ; ce travail n'y touche pas.

Code mesuré : arbre de travail à `f01bc9d` plus les documents du 09-06, aucune
modification de `llvq-llm` ni de `llvq-artifact` depuis le scellement du fichier
Tetra.

---

## §1 — La question, et pourquoi elle décide

Q5 est **mesuré et adopté**, mais sur une base qui n'est plus l'objet servi.
`v_proj` rendu en int4 g128 vaut **+3,60 pp** de MMLU [+1,47 ; +5,79] sur le
fichier publié et **+2,71 pp** [+0,59 ; +4,93] sur la graine 3 (*mesuré*,
`docs/mesures/m2b-v4bits-2026-09-02.txt` et `m2b-graine3-*`). Les deux sont sur
**Planes14**, base 55,59.

Tetra sert 53,49. Trois choses ont changé et aucune n'est neutre :

1. **Le signe du coût mémoire s'inverse.** Sous Planes14, `v` en int4
   **rendait** −0,013 b/param. Sous Tetra il **coûte +0,0493** (*calculé*,
   94 371 840 poids × (4,250 − 2,1498) ÷ 4 022 468 096). La conclusion
   « adopter » tenait sur un gain gratuit ; elle ne tient plus pour la même
   raison.
2. **La base est 2,10 pp plus bas**, donc il y a plus de marge jusqu'au plafond
   f16 de 70,32. Un même mécanisme peut rendre davantage.
3. **Le 8B dit que le déficit de Tetra grandit avec la taille** (−3,91 pp
   contre −2,10 au 4B, *mesuré*, `docs/mesures/tetra-8b-2026-09-06.txt`). Si ce
   déficit est concentré dans l'attention, la précision mixte le rachète ; s'il
   est diffus, elle ne le rachète pas. Personne ne sait lequel.

**Ce que ce travail mesure, et c'est la seule inconnue : combien de l'attribution
mesurée sous Planes14 transpose à Tetra.**

## §2 — Les bras, verbatim

Tout sur le Mac, Metal, f16, `bin/mmlu` à 40 questions par matière.

```
F=<scratchpad>/tetra4b/qwen3-4b-tetra.bin      # 1 770 529 149 octets
export LLVQ_MODEL=Qwen/Qwen3-4B
# T0  — témoin, déjà mesuré, voir §3 contrôle 6
LLVQ_MMLU_DUMP=$S/mmlu-4b-tetra-metal.csv            mmlu $F metal 40
# T1  — v_proj rendu en f16      → Gf
LLVQ_RESTORE_F16=v_proj LLVQ_MMLU_DUMP=$S/t-vf16.csv mmlu $F metal 40
# T2  — v_proj rendu en int4 g128 → G4
LLVQ_RESTORE_Q4=v_proj  LLVQ_MMLU_DUMP=$S/t-vq4.csv  mmlu $F metal 40
# T3  — attention entière en int4 g128 → Ga
LLVQ_RESTORE_Q4=q_proj,k_proj,v_proj,o_proj \
                        LLVQ_MMLU_DUMP=$S/t-attnq4.csv mmlu $F metal 40
```

T1 est nécessaire parce que le taux de survie `s = G4 / Gf` demande **les deux
mesures sur la même base**. Le +4,48 de M2 est un chiffre Planes14 ; le
réutiliser comme dénominateur sous Tetra serait exactement l'erreur que ce
préreg existe pour empêcher.

## §3 — Les contrôles, et si l'un tombe aucun chiffre n'est publié

1. Empreinte de tokens **`65dcd53655e8bfa5`**, 2 280 questions, sur les quatre bras.
2. Artefact **1 770 529 149 octets**, en-tête v5, kind Tetra.
3. T1 et T2 impriment **36 matrices et 94 371 840 poids** restaurés ; T3 en
   imprime **4 × 36 = 144** et **943 718 400** poids (*calculé* :
   2 × 377 487 360 + 2 × 94 371 840).
4. Les quatre dumps par question sont écrits, avec leur trailer.
5. T1 et T2 ne sont égaux **ni entre eux ni à T0** au centième. Une égalité
   dirait que la restauration n'a rien fait, ou que le passage à 4 bits n'a rien
   coûté.
6. **Le transfert de harnais CUDA → Metal.** Deux bras ont été lancés le
   2026-09-06 à 19:26, AVANT ce préreg, pour mesurer si Metal rejoue la carte :
   `Planes14` publié doit rendre 55,59 et Tetra 53,49. Sur les quinze premières
   matières, treize sont identiques au bit et deux diffèrent d'une question en
   se compensant. **Ces deux bras sont des contrôles, pas des traitements**, et
   leurs valeurs attendues étaient publiées avant d'être mesurées : il n'y a
   aucun degré de liberté à exploiter. Si l'écart final dépasse **0,5 pp** sur
   l'un des deux, tout ce travail repart sur carte et rien n'est publié depuis
   Metal.

## §4 — Ce qui se publie, et ce qui NE se compare PAS

Se publie : **Gf, G4, Ga** contre T0, appariés sur les dumps (bootstrap
stratifié 10 000 tirages, graine `0xb0075eed`, McNemar exact), le taux de survie
**s = G4 / Gf**, et le coût mémoire *calculé* en b/param modèle entier :
2,7645 → **2,8138** pour T2, → **3,2572** pour T3.

Ne se compare pas :
- **Aucun bras n'est un noyau.** Ils quantifient puis déquantifient. Aucun octet
  de 4 bits n'est lu par un noyau, aucune vitesse n'est mesurée, aucune empreinte
  réelle n'est produite. L'écrivain `kind = 2` n'existe pas et `tv_q4_h` n'a
  jamais tourné sur carte.
- **Gf et G4 de ce travail ne se comparent pas au +4,48 et au +3,60 de M2/M2b.**
  Base différente, format différent, device différent. Seul le rapport `s` est
  comparable d'une base à l'autre, et encore, avec deux points de chaque côté.
- **Ga n'est pas la somme des quatre bras isolés.** La sous-additivité mesurée
  vaut 0,641 et 0,776 dans l'attention (*calculé*, `m2-attribution` et
  `m2rep-graine3`). Aucun chiffre d'addition ne sera publié.
- Un seul tirage de calibration, une seule taille, un seul device.

## §5 — La règle de décision, posée AVANT de voir le chiffre

Barre de bruit à fichier constant : **0,43 pp**. Elle s'applique : les quatre
bras lisent le même fichier et les mêmes questions.

| | conséquence |
|---|---|
| **G4 ≥ +2,5 pp, IC95 entièrement > +1,0** | L'attribution transpose. Q5 devient le chantier servi : écrivain `kind = 2`, `tv_q4_h` sur carte, scellé Tetra mixte. La ligne 1 de la roadmap qualité passe de « mesure » à « ingénierie », et la queue f32 → f16 (ligne 7) la finance : −0,0675 b/param contre +0,0493. |
| **+1,0 ≤ G4 < +2,5 pp** | L'attribution transpose en partie. Q5 reste en réserve, et Ga décide : si Ga ≥ +4,0, la cible devient l'attention entière à 3,2572 b/param ; sinon la précision mixte sort de la roadmap comme levier principal et les lignes 14, 15 et 17 (les finetunings) passent devant. |
| **G4 < +1,0 pp** | L'attribution NE transpose PAS. Le déficit de Tetra n'est pas concentré dans `v_proj`. Q5 se ferme sur cette cible, M2 est à refaire entièrement sur Tetra avant toute autre décision de précision mixte, et le budget de bits libéré par la ligne 7 va ailleurs. |
| **autrement** — G4 ≥ +2,5 mais IC95 débordant sous +1,0, ou Gf < G4 | Rien n'est décidé. L'IC trop large appelle un second tirage de calibration ; Gf < G4 est une contradiction interne (4 bits ne peut pas battre f16 sur la même matrice) qui invalide le protocole et envoie tout dans les écarts. |

La barre porte sur **G4**, pas sur `s`. Un taux de survie flatteur sur un gain
devenu petit ne décide rien.

## §6 — Divulgation datée, et la prédiction signée

Connu à la signature : Tetra 53,49 · Planes14 55,59 · f16 70,32 · `v_proj` f16
sous Planes14 60,07 (+4,48) · `v_proj` int4 sous Planes14 +3,60 et +2,71 ·
survie 80,4 % et 94,4 % · attention entière f16 sous Planes14 +6,90 et +5,08.

**Prédiction de l'auteur, opposable : Gf entre +3,5 et +6,5 pp ; G4 entre +2,8
et +5,5 pp ; s entre 0,78 et 0,95 ; Ga entre +5,0 et +8,5 pp. Donc la première
ligne du §5.**

Motif : la base est 2,10 pp plus bas et le plafond est le même, donc le même
mécanisme dispose de plus de marge ; et la survie int4 a tenu au-dessus de 0,80
sur les deux tirages Planes14.

Ce que vaut cette prédiction : sur ce dossier l'auteur en a signé neuf et **six
étaient fausses**, dont les deux du 2026-09-06 (le signe de la perplexité au 4B,
et le sens de l'écart au 8B). Elle est là pour être opposable.

**Ce qui la rendrait fausse de façon instructive : G4 sous +1,0 pp.** Cela
voudrait dire que le déficit de Tetra est diffus et non concentré, ce qui
expliquerait aussi pourquoi il grandit avec la taille — un défaut réparti sur
toutes les matrices se paie proportionnellement au nombre de matrices. Dans ce
cas la roadmap qualité change de tête : les finetunings passent devant la
précision mixte, et il faudrait comprendre avant d'écrire quoi que ce soit.

## §7 — Ce que ce travail ne peut pas être

Un chemin servi. Il mesure un **coût d'information**, pas une vitesse ni une
empreinte. Il ne crée aucun format mixte, n'écrit aucun octet de 4 bits, et ne
touche ni au mot de 48 bits, ni au treillis, ni au bit de gain.
