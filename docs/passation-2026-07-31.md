# Passation — état au 2026-07-31, 23h50

> Ce que la session sait et que le code ne dit pas. Lire `CLAUDE.md` pour le
> projet, ce fichier pour savoir où reprendre.

## En une phrase

Le modèle quantifié **existe, est publié, et démarre seul** ; le noyau fusé
n'existe pas mais **son verrou principal a été levé ce soir, mesuré sur GPU**.

## Ce qui est acquis, et ce qui ne l'est pas

| | état |
|---|---|
| Taille sur disque | ✅ **1,771 Go contre 8,045 Go, ×4,54** — fichier pesé, décodé bit pour bit |
| Modèle publié | ✅ [Pier-Jean/Qwen3-4B-LLVQ-2bit](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit), démarre sans cache HF ni réseau |
| Qualité | ✅ **16,9617** de perplexité à **2,1595 b/poids** (QTIP : 17,04 à 2,000) |
| Place en mémoire | ❌ le modèle chargé occupe **~8 Go de RAM**, autant qu'en FP16 |
| Vitesse | ❌ **aucun gain**, aucun noyau |

Les deux derniers tombent ensemble : c'est le même travail, le noyau fusé.

## L'histoire de la journée, en trois retournements

1. **Trois défauts de comptabilité trouvés** en essayant d'écrire le fichier —
   la rétraction sphérique annulait le code de gain, `row_scale` dérivait, et
   `Direction`/`ShapeGain` étaient le même quantifieur. Le chiffre publié
   passe de « 14,91 à 2,11 bits » (faux) à **16,96 à 2,16 bits** (vrai, pesé).
2. **Le noyau semblait condamné** : remplacer le rang de permutation coûtait
   +0,34 b/poids, ce qui mettait 24 % au-dessus de QTIP pour la même qualité.
3. **Il ne l'était pas.** Le fichier garde le rang ; seul le layout **en RAM**
   change. Et le décodage à masques a été mesuré sur GPU à **0,11 ns/bloc**,
   soit 1,43× le coût de ne rien décoder du tout.

## Reprendre ici

**Étape suivante décidée** : fixer le format runtime réel et le mesurer sur de
vrais blocs. Trois choses en un seul travail :

1. **Le format** — comment encoder 3, 4 **ou 5** niveaux de magnitude. Le banc
   actuel suppose 4 partout. Deux variantes à départager par la mesure : champ
   fixe à 5 niveaux (simple, un peu gros) contre variable (compact, mais les
   lanes divergent).
2. **Le transcodeur** `.llvq` → layout runtime. Pont manquant entre l'artefact
   et le noyau, nécessaire dans tous les cas. `decfull` donne le décodeur rang
   rapide (243 ns/bloc) qui le rend viable : **~3,1 s** pour un 4B sur 12 cœurs.
3. **La mesure sur ces blocs-là**, qui met enfin la divergence dans la balance.

Puis : matvec fusé sur une couche contre le FP16 de la même machine, puis
intégration modèle.

⚠️ **Ne pas attaquer le matvec avant d'avoir fixé le format.** On écrirait le
noyau autour d'un layout provisoire et le premier `×` porterait sur des blocs
synthétiques — l'erreur de ce soir (trois bancs faux avant le bon), mais à
l'échelle d'un noyau.

## Les pièges, chèrement acquis

**Un banc GPU se trompe silencieusement.** Trois fois ce soir, chaque fois en
changeant le verdict :

- stocker les poids décodés → on mesure ses propres écritures non coalescées
  (195 Go/s), pas le décodage. Un noyau fusé n'écrit jamais ; le banc non plus.
- relire l'activation depuis la mémoire globale par itération → sol plafonné à
  80 Go/s. La charger une fois par threadgroup.
- travail du même ordre que le surcoût de soumission (~0,18 ms) → soustraire
  laisse du bruit. Dimensionner en millisecondes.

**Toujours relire et vérifier la sortie du kernel.** Un noyau que personne ne
regarde est un noyau que le compilateur peut supprimer, et un noyau mort se
mesure très bien.

**Le débit de 2,52 T op/s est une latence**, pas un débit crête — c'est une
chaîne dépendante de multiply-add. Tous les budgets « opérations par bloc »
qui en dérivent sont ~2× pessimistes. À re-mesurer avec du parallélisme
d'instructions.

**Le piège u64** : les *valeurs* des multinomials tiennent en u64, mais 21!..24!
débordent. Un décodeur u64 ne doit jamais passer par les factorielles — table
par classe, ou récurrence `M' = M·c_j/n`.

## Ce que l'audit adversarial a corrigé (à ne pas re-croire)

- Le plafond « ~340-400 tok/s » **oubliait le lm_head lié** (778 Mo f16, lu en
  entier à chaque token). Plafond honnête : **~190-200 tok/s**, ~3,8× le FP16.
  Levier suivant identifié : quantifier aussi le lm_head, 36-40 % du trafic.
- Le décodage GPU « coopératif » (ballot + prefix-sum) comptait des
  instructions simdgroup contre un budget en lane-ops : **une instruction
  occupe 32 lanes**. La forme viable est un bloc par lane, ce qui est celle
  mesurée.

## Cartographie du code ajouté aujourd'hui

```
llvq-artifact/       le format .llvq — ZÉRO dépendance, 3 crates dans l'arbre
                     (contre 690 pour llvq-llm)
llvq-metal/          micro-bancs GPU, macOS only
  bin/hello          plomberie + capacités machine
  bin/decode         sol / masques / rang, 16,7 M blocs
llvq-bench/
  bin/decbench       coût du décodage v1 vs plancher trivial (207×)
  bin/decprofile     ⚠️ biaisé, annoté, superseded par decfull
  bin/decfull        décodeur v1 rapide, 243 ns/bloc, bit-identique
  bin/decfast        la récurrence seule, 4,7×
  bin/arrbits        coût en bits des schémas d'arrangement, par coset
  bin/classprofile   structure des blocs réels (3-5 magnitudes)
llvq-llm/
  bin/seal           .llvq projections → modèle autonome
  bin/run            charge un modèle scellé et génère
```

## Branches

- `main` → poussée, contient tout jusqu'au modèle scellé et `LAUNCH_ME.md`.
- `g6-format-noyau` → 10 commits, le travail de format noyau. **À merger dans
  `main` quand la session reprend**, ou à continuer telle quelle.

## Points ouverts non tranchés

- **Coquille unique** : toujours pas implémentée (`set_shell_cap` fait une
  boule, pas une coquille). Norme constante, 79 classes au lieu de 383. Elle
  n'attaque pas le poste dominant, donc moins urgente qu'avant — mais elle
  trancherait la question ouverte au papier dans le `README`.
- **Contrôle de variance** : jamais lancé. Deux configs identiques ont donné
  7 % d'écart de perplexité ; tant qu'on ignore cette dispersion, tout écart
  de quelques pourcents est ininterprétable.
- **Décodeur portable** (Python/C++) pour que le `.llvq` s'ouvre sans compiler
  le Rust. `llvq-artifact` est sans dépendance exactement pour ça.
- **Contact aux auteurs du papier** : brouillon `docs/mail-qualcomm-draft.md`
  **périmé** (chiffres faux). Crédible sur le mode « reproduction indépendante
  + deux questions », pas sur « on fait mieux ».
