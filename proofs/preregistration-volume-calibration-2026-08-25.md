# Pré-enregistrement — le volume de calibration au 4B, en échelle

**Écrit, commité et tamponné AVANT le premier barreau.** Go de l'opérateur du
2026-08-25 (« go façon B »). Ce document couvre **les trois barreaux** ; chacun
part sur go séparé.

Dépend du pré-enregistrement
[`preregistration-bruit-mmlu-graines-2026-08-25.md`](preregistration-bruit-mmlu-graines-2026-08-25.md)
(sha256 `c97e1abf…`), dont la sortie **s** est le plancher de bruit de MMLU au
4B. Ce document ne l'amende pas.

---

## §1 — La question

Notre 4B perd **14,73 pp de MMLU** contre f16. Le papier amont, **même méthode**,
en perd **~9,5**. Deux différences connues expliquent au plus cet écart de
~5 pp : ils calibrent sur **~100× plus de texte** que nous, et ils tournent la
rotation en entrée **et** en sortie là où nous n'avons que l'entrée.

Ce document teste **la première seule**.

Le mécanisme attendu est une question d'estimation, et il se chiffre. Sur la
couche la plus large du 4B, `down_proj`, la hessienne fait **9 728 × 9 728**, et
on l'estime sur **131 072 tokens** — soit **13,5 exemples par dimension**. C'est
mince : l'estimation est bruitée, et le bruit change à chaque tirage de texte.

| barreau | tokens | exemples/dim (couche large) | bruit d'estimation |
|---|---|---|---|
| ×1 *(publié — témoin)* | 131 072 | 13,5 | référence |
| ×8 | 1 048 576 | 107,8 | ÷2,83 |
| ×32 | 4 194 304 | 431,2 | ÷5,66 |
| ×96 *(protocole du papier)* | 12 582 912 | 1 293,5 | ÷9,80 |

---

## §2 — Une seule variable, et le témoin est déjà payé

Tout est identique à l'artefact publié — `leech1c12`, corpus **C4**, rotation
d'entrée, `nogs`, damping 1e-2, dtype f32, 36 blocs, `rtx-pro-6000` — **sauf le
nombre de fenêtres de calibration**.

⚠️ **Le tirage reste le PRÉFIXE CONTIGU** (pas de `LLVQ_CALIB_SEED`), comme
l'artefact publié. Conséquence voulue : les quatre volumes sont des préfixes
**emboîtés** du même texte (×1 ⊂ ×8 ⊂ ×32 ⊂ ×96). Ils ne diffèrent que par « on
lit plus loin », jamais par « on lit ailleurs ».

**Témoin : l'artefact 4B publié.** ppl **16,9422**, MMLU micro **55,59 ± 1,35**,
empreintes `3f1baca9033bf251` (ppl) et `65dcd53655e8bfa5` (MMLU). Déjà mesuré,
coût zéro.

---

## §3 — Les trois barreaux, verbatim

Image `hf.co/spaces/Pier-Jean/llvq-runner-cuda`, flavor `rtx-pro-6000`, bucket
monté sur `/out`, `--timeout` posé à 2× la durée estimée du barreau.

```
export LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_THREADS=23
export LLVQ_ARTIFACT=$D/q4b-vN.llvq
smoke <W> 2048 12 4096 cuda nogs leech1c12 999 rot 2>&1 | tee $D/smoke.txt
seal $D/q4b-vN.llvq $D/q4b-vN-sealed.llvq 2>&1 | tee $D/seal.txt
LLVQ_DTYPE=f16 ppl 4096 12 cuda $D/q4b-vN-sealed.llvq \
  > $D/ppl-stdout.txt 2> $D/ppl-nll-par-fenetre.txt
LLVQ_MMLU_DUMP=$D/mmlu.csv mmlu $D/q4b-vN-sealed.llvq cuda 40 2>&1 | tee $D/mmlu.txt
sha256sum $D/q4b-vN-sealed.llvq
```

`<W>` = **512** (×8) · **2048** (×32) · **6144** (×96). Le témoin publié est
`<W> = 64`.

Durées et coûts *estimés* par un modèle calibré sur le profil mesuré du 4B
(`f5-graines-2026-08-19/seed1/smoke.txt` : 222,5 s fixes par bloc + 4,06 s par
bloc et par unité de volume) — modèle qui **reproduit à ×1 les 155 min
facturés**, ce qui est sa seule validation :

| barreau | durée | coût |
|---|---|---|
| ×8 | 2,9 h | ~8 $ |
| ×32 | 3,8 h | ~11 $ |
| ×96 | 6,4 h | ~18 $ |

---

## §4 — Par quel barreau on commence, décidé par **s** et non par l'envie

**s** = écart-type MMLU des trois graines F5, mesuré par le job
`6a8de89b984507d9db4e4664` **avant** que ce document ne serve.

| | barreau de départ |
|---|---|
| **s ≤ 1,0 pp** | **×8** — un effet de ~2 pp y serait lisible |
| **1,0 < s ≤ 2,0 pp** | **×32** — ×8 serait trop faible pour être lu |
| **s > 2,0 pp** | **aucun** — on ne lance pas, le design est à repenser |

⚠️ **Le document précédent posait cette règle pour un bras UNIQUE et ne
nommait pas ×8**, la façon B ayant été arbitrée après lui. La transposition est
écrite ici, avant toute mesure de volume, et elle ne relâche rien : ×8 n'est
admis que dans la branche la plus favorable, et la branche « on ne lance pas »
est inchangée.

**s majore le bruit réel de ce test** : les trois graines sont des tirages
indépendants, les barreaux sont des préfixes emboîtés, donc corrélés.

---

## §5 — Règle d'arrêt, posée d'avance

Δ = MMLU(barreau) − 55,59, apparié question par question contre le témoin
publié (McNemar + bootstrap stratifié par matière sur les dumps).

- **|Δ| > 2s** → on monte au barreau suivant, pour confirmer et chiffrer.
- **|Δ| ≤ 2s** → on monte **d'un seul** barreau de plus (l'effet peut être
  non linéaire en volume), puis **on s'arrête** s'il est encore plat.
- Dans tous les cas on s'arrête à ×96.

Conséquence chiffrée : le pire cas est **deux barreaux sur un résultat nul**,
soit ~6,7 h et ~19 $.

---

## §6 — Contrôles, et si l'un tombe le barreau n'est pas publié

1. **Volume réellement obtenu** — `smoke` imprime les fenêtres qu'il a pu lire.
   Si le shard C4 plafonne sous la demande (c'est arrivé sur wikitext : 847
   fenêtres pour 977 demandées), **le barreau se publie à son volume réel**, pas
   au volume demandé, et l'étiquette du tableau change.
2. **Débit constant** : `2,0702 b/poids effectifs` et un artefact scellé de
   **1 770 528 125 octets**. Le volume de calibration ne doit rien changer à la
   taille du fichier — s'il la change, ce n'est pas la variable qu'on croit.
3. **Empreintes** `3f1baca9033bf251` (ppl) et `65dcd53655e8bfa5` (MMLU).
4. **Factorisation sous 15 %** du run et pas d'avertissement `fast-linalg`.
5. **Vérification bit à bit** des 3 633 315 840 poids par `verify_artifact`.

---

## §7 — Ce que ce document NE décide pas

- **Aucune adoption.** Un résultat positif ne change **rétroactivement aucun
  chiffre publié** : les tables aux trois tailles restent celles des artefacts
  calibrés à ×1. Il ouvrirait une décision sur les artefacts **futurs**, et
  cette décision est celle de l'opérateur, pas de ce document.
- **Rien sur la rotation** entrée+sortie, l'autre différence connue au papier.
- **Rien sur la composition** du corpus. Plus de C4 rend H plus fidèle à C4 ;
  si notre déficit vient de ce que C4 est le mauvais *genre* de texte, ce test
  ne peut pas le voir et ne le prétendra pas.
- **Rien sur une autre taille.** Ce qui est mesuré ici est un fait 4B.

---

## §8 — Divulgation datée, à la signature

- Aucun run de quantification n'a jamais tourné à plus de 131 072 tokens de
  calibration, à aucune taille, depuis le début du projet.
- La famille calibration avait été déclarée **plafonnée** le 2026-08-06 sur la
  base d'une mesure « oracle » à **−1,6 %**. Cette mesure portait sur **3 blocs
  de Qwen3-0.6B** et sur la **perplexité**. Ce document la considère comme
  n'ayant jamais été testée à la taille publiée ni sur la métrique qui porte le
  déficit, et c'est la raison d'être du présent barreau.
- **Prédiction de l'auteur, écrite pour être opposable** : ×8 rend
  **Δ entre +0,5 et +2,5 pp** de MMLU, et l'essentiel de ce que le volume peut
  rendre est déjà là — ×96 ne dépassant pas **+4 pp**, donc ne refermant pas
  seul l'écart de 5 pp au papier.
- **Prédiction secondaire, que ce job NE TESTE PAS** (aucun instrument ne
  rapporte l'erreur par couche) : l'effet devrait être le plus fort sur
  `down_proj` (9 728 entrées, 13,5 exemples/dim) et le plus faible sur les
  projections d'attention (2 560 entrées, 51 exemples/dim).
- ⚠️ **Ce que vaut la prédiction principale** : l'auteur en a signé **deux** le
  2026-08-25 et **les deux étaient fausses**. Elle est là pour être opposable,
  pas pour être crue.
