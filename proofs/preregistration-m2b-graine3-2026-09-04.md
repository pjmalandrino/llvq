# Pré-enregistrement — M2b sur la graine 3 : remplacer une extrapolation par une mesure

**Écrit, commité et TAMPONNÉ le 2026-09-04, AVANT la première milliseconde du
job.** Go d'opérateur du 2026-09-04. Vague 1 : 4,60 $ dépensés sur 5, ce job
en coûte ~0,29, la vague finirait à **4,89 $ sur 5**.

🚨 **Ce fichier ne s'édite plus.** Ce qui le corrige va dans un `-ECARTS.md` à
côté.

## §1 — Pourquoi ce job, en une phrase

`docs/mesures/m2rep-graine3-4b-2026-09-04.txt` conclut sur un nombre **calculé,
pas mesuré** : le gain de `v_proj` en int4 g128 sur la graine 3 y vaut
0,804 × 2,87 = **+2,31 pp**, obtenu en appliquant au plafond f16 de cette
graine le taux de survie mesuré une seule fois, sur un autre fichier. Un noyau
de précision mixte ne se construit pas sur une extrapolation à un point. Ce job
la remplace par une mesure, pour 0,29 $ et ~10 min.

Il ne rouvre aucune décision. L'arbitrage du 2026-09-04 (M2b encaissable, Q5
ouvre) tient et n'est pas conditionné à ce résultat ; le chantier de noyau
démarre en parallèle de ce job.

## §2 — Le job, verbatim

Image `hf.co/spaces/Pier-Jean/llvq-runner-cuda`, **non reconstruite** (même
raison qu'au §É5 de `…-m2-attribution-4b-2026-09-02-ECARTS.md` : le seul écart
voulu avec M2b est le fichier quantifié). Flavor `l40sx1`.

```
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda --flavor l40sx1 \
  --bucket Pier-Jean/jobs-artifacts --name m2b-graine3-4b --timeout 25m \
  -- "$(cat <<'SH'
OUT=/out/m2b-graine3-4b-2026-09-04
mkdir -p $OUT
nvidia-smi --query-gpu=name,driver_version --format=csv | tee $OUT/gpu.txt
F=/out/f5-graines-2026-08-19/seed3/q4b-s3-sealed.llvq
ls -l $F | tee $OUT/artefact.txt
export LLVQ_MODEL=Qwen/Qwen3-4B
LLVQ_MMLU_DUMP=$OUT/mmlu-shipped.csv mmlu $F cuda 40 2>&1 | tee $OUT/mmlu-shipped.txt
LLVQ_RESTORE_Q4=v_proj LLVQ_MMLU_DUMP=$OUT/mmlu-v4.csv \
  mmlu $F cuda 40 2>&1 | tee $OUT/mmlu-v4.txt
SH
)"
```

`--timeout 25m` : M2b a pris 10 min sur deux bras identiques. Au pire
0,75 $, ce qui laisse la vague à 5,35 $ **au-dessus du plafond de 5** — donc le
plafond réel est le temps mesuré, et un dépassement se déclare. Le timeout est
un garde-fou machine, pas une autorisation de dépense.

## §3 — Contrôles, et si l'un tombe aucun chiffre n'est publié

1. **Empreinte de tokens `65dcd53655e8bfa5`** sur les deux bras.
2. **Fichier de 1 770 528 125 octets** (l'artefact graine 3 ; pas 1 770 527 533,
   qui est le fichier publié).
3. **Contrôle bas** : le bras livré rend **55,17 %** et son dump reproduit
   `docs/data/m2rep-graine3/mmlu-shipped.csv`, sha256
   `5f14bd3493ed809bd8d74a5b76b639ecc038e51489c7696867eafc1172e3dca9` —
   **identique octet pour octet**, comme le 2026-09-04. Deux jobs indépendants
   ont déjà rendu cet octet ; un troisième qui ne le rendrait pas invalide le
   job, pas le fichier.
4. **Le bras int4 imprime son compte de poids** : 36 matrices, 94 371 840 poids.

## §4 — Ce qui se publie

`Δ = int4 − livré` en micro stratifié, IC95 apparié (bootstrap stratifié par
matière, 10 000 tirages, graine `0xb0075eed`), McNemar exact, par `mmlupair`.
En regard : le taux de survie `s = G4 / Gf` avec **Gf = +2,87 pp**, le plafond
f16 mesuré sur CETTE graine le 2026-09-04.

Ce qui ne se compare pas : le b/param. `v_proj` en int4 g128 vaut 5,149 contre
5,162 (*calculé*, `m2b-v4bits-2026-09-02.txt`) et ce calcul ne dépend pas de la
graine — les formes sont les mêmes. Il n'est pas re-mesuré et pas re-publié.

## §5 — La règle de lecture, posée AVANT de voir le chiffre

Une seule grandeur par frontière, la partition est complète, et la dernière
ligne existe parce que le §5 de M2b n'en avait pas (correctif n° 3 de
`…-m2b-v4bits-2026-09-02-ECARTS.md` §É2).

| condition sur l'IC95 apparié de G4 | conséquence |
|---|---|
| **entièrement > 0** | le gain est **confirmé sur un second tirage**. Le noyau de précision mixte se construit, et le chiffre servi se publie comme une **plage sur deux tirages**, jamais comme le seul +3,60. |
| **contient 0** | le gain **n'est pas résolu sur ce tirage**. Le noyau se construit quand même — l'arbitrage du 2026-09-04 ne dépend pas de ce job — mais tout document qui cite +3,60 porte désormais « mesuré sur le fichier publié, non résolu sur la graine 3 ». |
| **entièrement < 0** | le passage en int4 **dégrade** sur ce tirage. Le chantier de noyau s'arrête et l'écart entre les deux tirages devient la question. |

⚠️ **Le taux de survie `s` ne décide rien.** Il est rapporté parce qu'il a servi
à extrapoler ; un `s` flatteur sur un gain devenu petit ne vaut rien. C'est
l'avertissement du §5 de M2b, repris tel quel.

## §6 — Prédiction signée, opposable

**G4 entre +1,8 et +2,8 pp, IC95 entièrement au-dessus de 0.** Motif : le
plafond f16 de cette graine est +2,87 et le seul taux de survie mesuré est
0,804 ; l'extrapolation donne +2,31, et la fourchette est élargie de ±0,5 pp
parce qu'un taux de survie mesuré une fois n'a pas d'erreur connue.

**Ce qui rendrait cette prédiction fausse de façon instructive** : un G4 proche
de +3,60, c'est-à-dire un gain int4 qui ne suit PAS le plafond f16. Cela
voudrait dire que le 4 bits ne récupère pas une fraction du gain de pleine
précision mais atteint un niveau propre — et alors le plafond f16 mesuré par
M2 n'est pas la bonne grandeur de référence pour Q5.

⚠️ Les prédictions signées de ce dossier : deux fausses le 2026-08-25, une
juste sur le nombre et fausse sur la conclusion le 2026-09-02, une réfutée sur
un tirage et confirmée sur l'autre le 2026-09-04. Celle-ci est opposable, pas
crédible d'avance.
