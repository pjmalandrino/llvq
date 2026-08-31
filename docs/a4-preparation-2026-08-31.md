# A4 — préparation du job A100 (2026-08-31, avant lancement)

> A4 est le second bras inconditionnel du préreg vague 2 (`e23e9895…`, §A4),
> ordre imposé : **après A1**. Ce document prépare la commande et consigne
> deux contraintes d'exécution découvertes à la lecture du code — AVANT le
> job, pour qu'elles ne deviennent pas des écarts découverts sur carte.

## Contrainte 1 — le vocabulaire du banc n'est pas celui du préreg

Le préreg §A4 écrit « banc (`planes14`, `planes14_seg`, `nullk`, `f16`,
`cublasf16`) ». Le vocabulaire réel de `LLVQ_BENCH_ARMS`
(`llvq-cuda/src/arms.rs:47-74`) : `fp16` (pas `f16`), et **`planes14_seg`
n'est pas un bras nommable** — la section seg (`tv_slot_seg` +
`tv_planes_seg`) tourne **automatiquement après la table** dès que
`planesbench` reçoit un chemin de modèle (en-tête de `planesbench.rs`,
« The A4 section, after the table »). La commande ci-dessous couvre donc
exactement ce que le préreg demande, sous les noms que le binaire accepte.
Ce n'est pas un écart : c'est la traduction préreg → vocabulaire.

## Contrainte 2 — `fusedrun` NE PEUT PAS tourner sur A100 avec l'image actuelle

`ops/Dockerfile.cuda:23` fige `CUDA_COMPUTE_CAP=89` : les noyaux candle
(famille `fusedrun`) sont du PTX sm_89, compatible **vers le haut seulement**
— sur sm_80 le job démarre, facture, télécharge, et ne charge aucun noyau
(`ops/run.py:60-89`, qui refuse `a100-large` pour cette raison exacte ;
`MIN_COMPUTE_CAP=89`). La famille `llvq-cuda` (NVRTC au démarrage +
`LLVQ_NVRTC_ARCH=compute_80`) tourne, elle — précédent F4, neuf bras verts.

**Décision d'opérateur requise, deux issues :**
- **(a) recommandée** : publier une image jumelle `CUDA_COMPUTE_CAP=80` —
  **une commande depuis le 2026-08-31 au soir** :
  `uv run ops/run.py publish Pier-Jean/llvq-runner-cuda-sm80 --cuda
  --compute-cap 80` (build HF 0 $, ~15-20 min). Pas de recette jumelle dans
  le dépôt : `publish --compute-cap` réécrit la seule ligne `ENV` au
  téléversement et le déclare dans la recette téléversée ET dans `COMMIT` —
  un fichier canonique, pas deux qui dérivent. → A4 complet, banc + fusedrun.
- **(b)** : A4 = banc seul sur l'image actuelle ; la moitié `fusedrun` se
  déclare bloquée par l'image dans le fichier d'écarts de la vague 2, datée.

## La commande (moitié banc — prête ; lancement direct `hf jobs run`, comme F4)

- flavor `a100-large` · image `hf.co/spaces/Pier-Jean/llvq-runner-cuda`
- volumes : modèle `Pier-Jean/Qwen3-4B-LLVQ-2bit` → `/model` (lecture seule),
  bucket `Pier-Jean/jobs-artifacts` → `/out`
- ⚠️ `ops/run.py bench` refusera le flavor (garde sm_89) : lancement direct,
  l'override nommé ici, comme le run.py le prescrit (:84-89).

```
set -euo pipefail
mkdir -p /out/vague2-2026-08-31
which planesbench nullkbench
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv | tee /out/vague2-2026-08-31/a4-gpu.txt
LLVQ_NVRTC_ARCH=compute_80 preflight 2>&1 | tee /out/vague2-2026-08-31/a4-preflight.txt
LLVQ_NVRTC_ARCH=compute_80 LLVQ_BENCH_ARMS="planes14,fp16,cublasf16,nullk" \
  planesbench /model/qwen3-4b-llvq.bin 2>&1 | tee /out/vague2-2026-08-31/a4-a100-planesbench.txt
LLVQ_NVRTC_ARCH=compute_80 nullkbench 2>&1 | tee /out/vague2-2026-08-31/a4-a100-nullkbench.txt
```

(le bras `nullkbench` en prime : le même r=t(144)/t(252) sur la seconde
architecture, gratuit dans le même job — les deux points d'horloge du lot G
prédisent son échelle, pas son rapport)

Moitié `fusedrun` (si issue (a)) — même job ou job séparé sur l'image sm80 :

```
LLVQ_FUSED_LAYOUT=planes14 LLVQ_EMBED=q8 LLVQ_ROT_SHARE=1 LLVQ_FUSE=1 \
  fusedrun /model/qwen3-4b-llvq.bin 128 2>&1 | tee /out/vague2-2026-08-31/a4-a100-fusedrun.txt
```

## Lecture (rappel du préreg, inchangée)

Un bras réseau **≥ 1,00× FP16** sur A100 → la réserve « résultat Ada » saute ;
sinon l'attribution horloge du lot G s'étend à la géométrie fusée et la
réserve devient un mécanisme mesuré à deux points. Les deux issues se
publient. 🚨 Aucun × A100 ne se divise par un × L40S.

Budget vague 2 : dépensé 1,32 $ + A1 ~0,2 $ ; A4 ~0,9-1,0 $ tient sous le
plafond de 5 $.
