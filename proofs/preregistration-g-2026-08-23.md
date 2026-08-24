# Pré-enregistrement G — trois mesures à moins de 5 $ pour la v5 du papier (2026-08-23)

Écrit et tamponné **avant** tout lancement. Issu de la revue
[`docs/revue-taco-2026-08-22.md`](../docs/revue-taco-2026-08-22.md) §7
(option 2 + les trois lignes de l'option 3 qui coûtent moins de 5 $) ; go de
principe de l'opérateur le 2026-08-22 (« ok avec toi, let's go »), go de
dépense à confirmer sur le coût annoncé au §4.

## 1. Ce que ce lot mesure, et ce qu'il ne décide pas

Trois cases que la v5 du papier déclare aujourd'hui comme non mesurées :

- **G1/G2 — horloges SM pendant le banc, L40S puis A100.** Le §3.5 de la v5
  attribue le ralentissement ×1,78 du témoin sans lecture (2,306 → 4,107 ms)
  au rapport des horloges SM *constructeur* (2 520 / 1 410 MHz) et le dit
  « hypothèse cohérente, non mesurée ». Ce lot lit les horloges réelles
  pendant les noyaux, par `nvidia-smi` échantillonné à 1 s.
- **G3 — `Planes12x` servi bout-en-bout sur le 4B.** Le layout le plus
  économe en VRAM est câblé dans `fusedrun` (`LLVQ_FUSED_LAYOUT=planes12x`)
  et n'a jamais été chronométré dans le modèle. C'est la seule ligne où le
  produit passerait sous l'AWQ de plus de 10 % en b/param (4,745 contre
  5,302, `rtbits`).

Aucun seuil de kill : ce lot **mesure**. Ce qui est posé d'avance est la
lecture de chaque issue.

## 2. Les jobs, verbatim (image `Pier-Jean/llvq-runner-cuda` courante)

Sorties tee vers `/out/g-2026-08-23/` (bucket `Pier-Jean/jobs-artifacts`).

**G1 — L40S** (`l40sx1`, `--timeout 25m`) :
```
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv -l 1 > /out/g-2026-08-23/g1-l40s-clocks.csv &
LLVQ_BENCH_ARMS="fp16,planes14,nullk" planesbench /model/qwen3-4b-llvq.bin 2>&1 | tee /out/g-2026-08-23/g1-l40s-planesbench.txt
```

**G2 — A100** (`a100-large`, `--timeout 25m`, lancé par `hf jobs run`
direct comme F4, `LLVQ_NVRTC_ARCH=compute_80`) : même script, fichiers
`g2-a100-*`.

**G3 — `Planes12x` servi** (`l40sx1`, `--timeout 40m`, artefact HF monté) :
```
LLVQ_FUSED_LAYOUT=planes12x LLVQ_EMBED=q8 fusedrun /model/qwen3-4b-llvq.bin 128 2>&1 | tee /out/g-2026-08-23/g3-4b-planes12x-q8.txt
```
Protocole de `fusedrun` inchangé : 1 génération jetée + 5 chronométrées,
médiane [min–max], tokens comparés au bras dense de la même invocation.

## 3. Ce qui se publie, posé d'avance

**G1/G2.** Par carte : horloge SM médiane sur les échantillons pris pendant
les rounds du banc (fenêtre = entre le premier et le dernier « round » du
journal). La quantité publiée est le **rapport L40S/A100 des médianes**.
- Si le rapport est dans **[1,60 ; 1,95]** : le §3.5 passe de « hypothèse
  cohérente avec les horloges constructeur » à « cohérent avec les horloges
  *mesurées* pendant le banc (x contre y MHz) ». Le mécanisme reste une
  hypothèse (pas de profil), mais ancrée sur une lecture.
- Hors de cet intervalle : la phrase sur les horloges est **retirée** du
  §3.5 et remplacée par « le ralentissement du témoin sans lecture n'est pas
  expliqué ». Pas de troisième option.
- Les raisons de throttling actives et la puissance sont rapportées telles
  quelles, sans interprétation.
- Contrôle : les rapports vs FP16 des deux bras doivent rejouer Table 1 /
  Fig. 5 à la dispersion près (±3 %) ; sinon la lecture d'horloge est
  rapportée mais pas rattachée aux chiffres publiés.

**G3.** Trois cases de la Table 3 (tab:e2e) — tok/s médiane [plage], Go
carte (compte hôte), et le rapport aux tokens du bras dense.
- **Prédictions (calculées, à confronter)** : VRAM ≈ **2,39 Go** (2,60 −
  (4,804 − 4,342) × 3 616,4 M / 8) ; débit ≈ **84 tok/s** (87,0 ralenti des
  0,39 ms/token que le banc mesure entre `Planes14` et `Planes12x` sur les
  252 projections). Une médiane hors **[76 ; 90]** tok/s ou une VRAM hors
  **[2,30 ; 2,48]** Go est une anomalie à expliquer avant publication.
- Tokens : identiques au bras dense jusqu'au tie-break attendu (token 89).
  Une divergence **avant le token 80** est l'anomalie A1 : suspend la
  publication et ouvre une vérification du chemin `planes12x` (le flux
  hôte est prouvé bit-exact, le noyau f16 ne l'est que par les tokens).
- Quelle que soit l'issue, la ligne entre dans tab:e2e **et** dans
  limitations : si `Planes12x` est plus lent que ce que le banc prédit,
  c'est le coût du canal d'exceptions sur le chemin servi, et il se publie.

## 4. Coût annoncé avant lancement

| job | flavor | $/h | durée attendue | plafond (timeout) |
|---|---|---|---|---|
| G1 | l40sx1 | 1,80 | ~8 min (transcodage Slot32 + Planes14 seuls) | 25 min → 0,75 $ |
| G2 | a100-large | 2,50 | ~10 min | 25 min → 1,04 $ |
| G3 | l40sx1 | 1,80 | ~25 min (transcodage `Planes12x` ~7 min sur 8 vCPU + 6 générations) | 40 min → 1,20 $ |

**Attendu ≈ 1,3 $, au pire 3,0 $.** Cumul rapporté après, au registre
`docs/data/jobs.csv`.

## 5. Sorties

Journal : `docs/mesures/g-horloges-planes12x-2026-08-23.txt` (sorties
brutes des trois jobs, CSV d'horloges compris). Registre : trois lignes à
`docs/data/jobs.csv`. Papier : §3.5 (phrase d'horloges), Table 3 et §6
(ligne `Planes12x`), Appendix B (provenance + ligne tab:prereg).
