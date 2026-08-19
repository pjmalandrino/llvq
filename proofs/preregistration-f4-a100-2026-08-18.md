# Pré-enregistrement F4 — seconde architecture : A100/HBM (2026-08-18)

Écrit et commité **avant** le lancement. Item F4 du plan TACO ; go de dépense
opérateur du 2026-08-18 (session groupée).

## 1. La question

Toute l'échelle bits↔vitesse publiée est mesurée sur UNE hiérarchie mémoire
(L40S : GDDR6, sm_89, 48 Mo de L2). La question de F4 n'est pas « les ×
sont-ils les mêmes ailleurs » — ils ne le seront pas, et c'est attendu —
mais : **l'ORDRE de l'échelle et la lecture roofline tiennent-ils sur une
seconde hiérarchie** (A100 : HBM2e, sm_80, 40 Mo de L2) ? Un décodage sans
divergence dont le verdict dépendrait de la carte ne serait pas un résultat
de format.

Mécanisme livré au commit `1a585d0` : `LLVQ_NVRTC_ARCH` (défaut `compute_89`
inchangé) — la passe NVRTC et l'assert `binary_version` suivent l'override.
Ce job est aussi la première exécution de ce mécanisme.

## 2. Le job, verbatim (image ≥ commit `1a585d0`)

`a100-large`, `--timeout 40m` :

```
hf jobs run --flavor a100-large --timeout 40m -d \
  -v hf://Pier-Jean/Qwen3-4B-LLVQ-2bit:/model \
  -v hf://buckets/Pier-Jean/jobs-artifacts:/out \
  hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  -- bash -lc 'set -euo pipefail
mkdir -p /out/b2-plages-2026-08-18
nvidia-smi --query-gpu=name,memory.total --format=csv
LLVQ_NVRTC_ARCH=compute_80 preflight 2>&1 | tee /out/b2-plages-2026-08-18/f4-a100-preflight.txt
LLVQ_NVRTC_ARCH=compute_80 LLVQ_BENCH_ARMS="slot32,planes14,planes12x,golay70v1,fp16,awq,golay70v2,cublasf16,nullk" \
  planesbench /model/qwen3-4b-llvq.bin 2>&1 | tee /out/b2-plages-2026-08-18/f4-a100-planesbench.txt'
```

Une seule phase, neuf bras (les sept de l'échelle publiée + `cublasf16` +
`nullk`). `e1v` est exclu délibérément : jamais compilé par un compilateur
device, son risque ne se mélange pas à la question de F4.

**Coût annoncé avant lancement : ≤ 2,50 $** — le tarif a100-large n'est pas
dans le registre (première utilisation) ; l'estimation vient du banc 7 bras
L40S (25 min, 0,74 $) et d'un tarif A100 supposé ~2-4 $/h. Le cap dur est le
timeout 40 min. Si le tarif observé rend le pire cas > 2,50 $, le registre
le dira et le prochain préreg A100 partira du tarif mesuré. Cumul rapporté
après.

## 3. Ce qui se publie, posé d'avance

- La table du banc sur A100 : médianes + plages par bras, rapports formés
  round par round — même protocole, même comptabilité d'octets.
- La comparaison inter-cartes se fait sur **l'ordre de l'échelle** et sur
  les **fractions de la borne d'octets** (Go/s du bras / Go/s du plancher
  nullk), jamais en divisant un × L40S par un × A100.

## 4. Anomalies et verdicts, définis avant la mesure

- **V1 (mécanisme)** — le rapport device doit dire sm_80 et l'assert
  `binary_version` doit tenir avec l'override : c'est F4 qui se prouve. Un
  échec ici est un échec du mécanisme, pas de l'échelle.
- **V2 (correction)** — la vérification f64 ligne à ligne doit passer aux
  mêmes seuils (1e-5 ; binary16 1e-3 pour AWQ/cublasf16). Un bras qui ne
  reproduit pas sa référence sur A100 invalide SON chiffre et constitue un
  résultat en soi (divergence d'arithmétique inter-architectures).
- **V3 (l'échelle)** — si l'ORDRE des bras LLVQ (Planes14 > Planes12x >
  Slot32 > Golay70) change sur A100 au-delà du recouvrement des plages,
  c'est un **résultat à publier**, pas une anomalie à expliquer avant
  publication : le papier dirait alors « l'échelle est une propriété du
  couple format×carte », ce qui est une conclusion différente et honnête.
- **A1** — dispersion : plage > 10 % de la médiane sur un bras → investigué
  avant publication (même seuil que B2).

## 5. Réserves déclarées d'avance

- Un point A100 ne fait pas une étude de portabilité : il transforme
  « mesuré sur L40S » en « mesuré sur deux hiérarchies mémoire », rien de
  plus, et le papier le dira dans ces termes.
- Le driver JIT n'intervient pas (NVRTC compile compute_80 nativement) mais
  `driver.version` sera capturé si la requête passe — la dette du premier
  contact carte (jamais corrigée) se solde ici si nvidia-smi le rend.

## 6. Sorties

Journal : `docs/mesures/f4-a100-2026-08-18.txt`. Registre : une ligne (avec
le tarif a100-large observé, première entrée de ce flavor).
