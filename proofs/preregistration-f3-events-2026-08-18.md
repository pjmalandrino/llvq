# Pré-enregistrement F3 — attribution instrumentée : events, Nsight, driver (2026-08-18)

Écrit et commité **avant** le lancement. Item F3 du plan TACO ; go de dépense
opérateur du 2026-08-18 (« F3 en parallèle si tu peux »).

## 1. Ce que F3 attaque

L'attribution 39/33/19 (latence/flux/décodage) est un **reste par
soustraction** de noyaux-sols, sans compteur matériel — déjà réattribuée une
fois, et sa moitié « occupation » tient sur un job unique à 0,07 $. F3 ne la
remplace pas d'un coup : il livre (a) la **composante mesurable** — le span
device par bras encadré d'events CUDA, dont l'écart au wall-clock hôte isole
soumission non recouverte + sync ; (b) une **tentative** Nsight Compute dont
les trois issues sont décidées ici ; (c) la capture du **driver**, jamais
faite (la requête du 05-08 utilisait le champ invalide `driver.version` —
le champ correct est `driver_version`).

Mécanisme livré au commit `b7698fe` : `LLVQ_TIME_EVENTS=1` — deux events par
(bras, round), variable absente = protocole publié byte-identique (le motif
`LLVQ_TIME_PHASES`).

## 2. Le job, verbatim (image ≥ commit `b7698fe`)

`l40sx1`, `--timeout 35m` :

```
hf jobs run --flavor l40sx1 --timeout 35m -d \
  -v hf://Pier-Jean/Qwen3-4B-LLVQ-2bit:/model \
  -v hf://buckets/Pier-Jean/jobs-artifacts:/out \
  hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  -- bash -lc 'set -euo pipefail
mkdir -p /out/b2-plages-2026-08-18
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv
echo "##### ETAPE 1 — events par bras, echelle complete #####"
LLVQ_TIME_EVENTS=1 LLVQ_BENCH_ARMS="slot32,planes14,planes12x,golay70v1,fp16,awq,golay70v2,cublasf16,nullk" \
  planesbench /model/qwen3-4b-llvq.bin 2>&1 | tee /out/b2-plages-2026-08-18/f3-events.txt
echo "##### ETAPE 2 — tentative Nsight Compute (issues pre-decidees au prereg) #####"
(apt-get update -qq && apt-get install -y -qq --no-install-recommends cuda-nsight-compute-12-4) \
  || echo "NCU-INSTALL-IMPOSSIBLE"
NCU=$(command -v ncu || ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1 || true)
if [ -n "$NCU" ]; then
  LLVQ_BENCH_ARMS="fp16,planes14;fp16,planes14" timeout 600 "$NCU" \
    --metrics dram__bytes.sum.per_second,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum \
    --launch-count 24 --target-processes all \
    planesbench /model/qwen3-4b-llvq.bin 2>&1 | tee /out/b2-plages-2026-08-18/f3-ncu.txt \
    || echo "NCU-RUN-REFUSE (voir f3-ncu.txt)"
else
  echo "NCU-ABSENT"
fi'
```

**Coût annoncé avant lancement : ≤ 1,20 $** (étape 1 ≈ le banc F1 en 9 bras
~10-15 min ; étape 2 bornée par `timeout 600` ; cap dur 35 min ≈ 1,05 $ au
tarif L40S observé). Cumul rapporté après.

## 3. Verdicts et issues, posés avant la mesure

- **E1 (events)** — la quantité publiable est l'**écart hôte − device par
  bras** (médiane sur les 5 rounds gardés). Attendu de l'ordre de 0,5-2 ms
  par round de 252 matrices (~2-8 µs/lancement × 252-504 lancements) ;
  aucune bande de kill. **Anomalies** : device > hôte (impossible — bug
  d'events, invalide l'étape) ; écart négatif ; dispersion de l'écart
  > 50 % de sa médiane. Le chiffre requalifie l'attribution du papier en
  « décomposition par soustraction, DONT la composante hôte est désormais
  bornée par events », quelle que soit l'issue de l'étape 2.
- **N1 (Nsight)** — trois issues, toutes acceptables, aucune improvisée :
  (a) compteurs accessibles → `dram__bytes.sum.per_second` et l'occupation
  de `planes14`/`fp16` corroborent ou corrigent les postes flux/occupation ;
  le papier gagne son profil. (b) installation OK mais compteurs refusés
  (`ERR_NVGPUCTRPERM`, attendu sur infra partagée) → le refus est LE
  résultat : le papier documente que le profil matériel est inaccessible
  sur cette infra et garde la requalification (a-minima). (c) installation
  impossible → idem (b), en le disant. Dans les trois cas, F3 est CLOS —
  pas de retentative non pré-enregistrée.
- **D1 (driver)** — `driver_version` capturé en tête de job : la dette du
  premier contact carte (08-05) est soldée pour tous les runs de cette
  image. S'ajoute au journal et à `fiche-4b`-niveau provenance des runs
  futurs, sans réécrire les anciens (leur driver reste inconnu, et le
  restera — c'est déclaré, pas réparé rétroactivement).

## 4. Réserves déclarées d'avance

- Les events mesurent le span du stream, pas l'activité SM : les écarts
  inter-noyaux sur le stream sont DANS le span device. L'écart hôte−device
  est donc un MINORANT du poste latence côté hôte, pas le poste entier.
- L'étape 2 profile `--launch-count 24` : un échantillon des premiers
  lancements, pas la distribution — assez pour un débit DRAM par noyau,
  pas pour une variance.

## 5. Sorties

Journal : `docs/mesures/f3-events-2026-08-18.txt`. Registre : une ligne.
Conséquence papier (B4) : la figure d'attribution se requalifie selon E1/N1.
