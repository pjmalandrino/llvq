# LLVQ sur Qwen3-4B — le dossier de publication (2026-08-07)

> Le document de référence : tous les résultats de la campagne finale, leur
> provenance job par job, et les données propres pour figures dans
> [`data/`](data/). Chaque chiffre est mesuré sur NVIDIA L40S, dans un seul
> harnais, avec empreintes de tokens imprimées et identiques entre bras
> (**ppl `3f1baca9033bf251` · MMLU `65dcd53655e8bfa5`**). Vulgarisation :
> [`comparatif-simple-2026-08-07.md`](comparatif-simple-2026-08-07.md) ;
> état complet du projet : [`rapport-etat-2026-08-07.md`](rapport-etat-2026-08-07.md).

## 1. Le résultat principal

**Un Qwen3-4B quantifié à 2 bits (fichier de 1,77 Go) tourne à 88,4 tok/s
dans 2,60 Go de VRAM — ×2,03 le débit et ÷3,09 la mémoire du moteur de
référence f16 — à qualité strictement identique au fichier publié** (ppl
16,9358 contre 16,9422 ; MMLU 55,70 ± 1,35 contre 55,59 ± 1,35 — mêmes
questions, mêmes fenêtres, mêmes empreintes). En empreinte mémoire totale
(b/param, modèle entier), il passe **sous l'AWQ 4 bits officiel : 5,15
contre 5,30**.

| | FP16 | AWQ 4 bits | LLVQ sans noyau | **LLVQ planes14+noyau+q8** |
|---|---|---|---|---|
| disque | 8,04 Go | 2,67 Go | **1,77 Go** | **1,41 Go**² |
| VRAM | 8,04 Go | 5,30 b/param¹ | 8,04 Go | **2,60 Go · 5,15 b/param** |
| vitesse | 43,5 tok/s | —¹ | 43,5 tok/s | **88,4 tok/s** |
| ppl wikitext | 12,2369 | 13,5207 (×1,105) | 16,9422 (×1,384) | **16,9358 (×1,384)** |
| MMLU micro | 70,32 ± 1,28 | 70,04 ± 1,25 | 55,59 ± 1,35 | **55,70 ± 1,35** |

¹ Dans son propre moteur ; jamais exécuté dans le nôtre — vitesse non
comparable, VRAM = sa propre comptabilité (5,302 b/param).
² Disque de la colonne 4 = `q4b-e8.llvq` (1,406 Go, embedding int8
pré-cuit), le fichier sur lequel sa qualité est mesurée ; sa vitesse et sa
VRAM viennent de `qwen3-4b-llvq.bin` + `LLVQ_EMBED=q8` (quantification au
chargement, contenu bit-identique). Deux fichiers, un contenu.
Données : [`data/campagne-finale.csv`](data/campagne-finale.csv).

**La lecture honnête en trois axes** : le noyau+format+embedding valent
×2,03/÷3,09 gratuits en qualité (colonnes 3→4, même fichier) ; face au
FP16 le prix du 2 bits est ×1,384 de ppl et −14,6 pp de MMLU ; face au
4 bits nous gagnons le disque et la VRAM, lui la qualité — le pari restant
est l'échelle (Qwen3-8B : ×1,267 contre ×1,384 au même protocole).

## 2. L'échelle des formats VRAM (le banc à 5 bras)

| layout | b/poids payload | méd. ms | vs FP16 | verdict |
|---|---|---|---|---|
| FP16 (témoin) | 16,000 | 11,025 | 1,00× | — |
| Slot32 | 5,510 | 5,883 | 1,87× [1,86–1,88] | l'ancien défaut |
| **Planes14** | 4,804 | 5,156 | **2,14× [2,11–2,15]** | **en production** |
| **Planes12x** | 4,342 | 5,563 | 1,98× [1,95–1,99] | le point « bits », qualité exacte |
| Golay70 | 3,589 | 8,410 | 1,31× [1,29–1,32] | écarté (borné calcul, critère 1,6×) |

Un seul job, cinq bras entrelacés à chaque round, vérification ligne à
ligne contre f64 (1 105 920 lignes, pires erreurs 2,2-3,0e-8) avant tout
chronométrage, zéro octet de mémoire locale sur les cinq noyaux.
Données : [`data/echelle-formats.csv`](data/echelle-formats.csv).

## 3. L'attribution du ×2,03 (temps par phase d'un token)

| phase | dense f16 | fusé tête f16 | fusé tête q8 |
|---|---|---|---|
| embedding | 0,026 | 0,025 | 0,017 |
| blocs transformer | 13,291 | 10,439 | 10,432 |
| **lm_head** | **26,672** | **25,886** | **0,598** |
| argmax + divers | 0,101 | 0,097 | 0,078 |

Le moteur de référence (candle 0.9.2) matérialise une copie transposée de
778 Mo du vocabulaire **à chaque token** (`broadcast_matmul` →
`contiguous()` → gather `ucopy_f16` ; le `TODO: Avoid concretising` est
dans son source, `tensor.rs:1550`). Notre noyau q8 lit 413 Mo une fois,
sans copie. D'où la formulation double : **×2,03 contre le moteur tel que
tout le monde l'utilise ; ~×1,4 contre ce moteur corrigé de sa copie**
(recomposition des phases). Phases bornées par synchronisation — elles
s'attribuent, leur somme n'est pas un tok/s.

> ⚠️ **Il existe un troisième chiffre, et c'est le seul qui soit à la fois
> mesuré bout-en-bout et attribuable au noyau Leech : ×1,12.** C'est le rapport
> **à tête identique** — f16 des deux côtés, la colonne « fusé tête f16 » du
> tableau ci-dessus contre la colonne « dense f16 » — soit 48,6 contre
> 43,5 tok/s dans le même job. Le ~×1,4 est une *recomposition* de phases
> fencées ; le ×1,12 est un tok/s relevé. **Ne jamais publier le ×2,03 sans
> lui.** Source : [`mesures/phases-2026-08-07.txt`](mesures/phases-2026-08-07.txt),
> recoupé par [`mesures/planes14-fusedrun-2026-08-06.txt`](mesures/planes14-fusedrun-2026-08-06.txt)
> (48,7 contre 43,6 dans un autre job).
Données : [`data/phases.csv`](data/phases.csv).

## 4. La progression de la semaine

| date | étape | VRAM | tok/s | b/param |
|---|---|---|---|---|
| 05-08 | sans noyau (état initial) | 8,04 Go | 43,5 | 16,00 |
| 06-08 | noyau fusé branché (Slot32) | 3,28 Go | 47,0 | 6,52 |
| 06-08 | layout Planes14 (C1) | 2,96 Go | 48,7 | 5,89 |
| 07-08 | embedding q8 (M1) | **2,60 Go** | **88,5** | **5,15** |

Chaque étape : implémentation + revue adversariale (mutants rejoués,
sondes indépendantes) + mesure gated. Le fichier disque et la qualité
n'ont jamais changé. Données : [`data/progression.csv`](data/progression.csv).

## 5. Ce qui a été écarté, avec ses mesures

- **Plafond L≤4 sec** : +4,75 % de ppl (swap mesuré) — remplacé par
  l'overlay exact.
- **Golay70 (E2)** : 3,589 b/poids réels, reconstruction exacte, mais
  1,31× — le décodage à double coset borne le noyau en calcul.
- **Design C** (qualité) : ×1,99 de ppl à pleine profondeur — deuxième
  occurrence du motif « proxy local meilleur, composition désastreuse ».
- **Calibration** ×volume/corpus : plafonnée par l'oracle à −1,6 %.
- **Rotation de sortie, codage entropique du froid, spéculatif 4B** :
  morts sur mesure ou sur le papier lui-même.

## 6. Les jobs — provenance complète

Chaque cellule du dossier remonte à un job daté avec son coût
([`data/jobs.csv`](data/jobs.csv), mesures brutes dans
[`mesures/`](mesures/)). Les jobs porteurs :

| date | job | coût | ce qu'il établit |
|---|---|---|---|
| 06-08 | `6a746d8f` / `6a746f9e` | 0,71 $ | qualité des bras f16/AWQ/LLVQ (campagne A4) |
| 06-08 | `6a748463` | 0,08 $ | C1 : Planes14 1,14× vs Slot32, contenu identique |
| 06-08 | `6a7492bc` | 0,33 $ | branchement : 48,7 tok/s / 2,96 Go + contrôle |
| 07-08 | `6a751120` | 0,90 $ | overlay validé + q8 : 88,5 tok/s / 2,60 Go |
| 07-08 | `6a7586d2` | 0,33 $ | l'attribution par phases |
| 07-08 | `6a759c00` | 0,74 $ | E2 mesuré et écarté |
| 07-08 | `6a75eb76` | 0,47 $ | qualité du bras 4 sur le même silicium |

**Coût GPU total** : lot A (branchement + campagne A4) **2,19 $** ;
séquence C1 → campagne finale **2,85 $** ; **grand total 5,04 $**.
*(Correction au passage : deux totaux intermédiaires erronés dans
`rapport-etat` (« 2,05 $ ») et `campagne-finale` (« 3,33 $ ») sont
rectifiés — le détail par job ci-dessus fait foi.)*

## 7. Réserves et périmètre (à reproduire dans toute communication)

1. La vitesse AWQ n'est pas comparable (moteurs différents) ; sa qualité
   l'est (même harnais, mêmes empreintes).
2. Le ×2,03 porte la formulation double du §3.
3. Un seul modèle (4B), un seul silicium (L40S), décodage glouton ;
   l'axe d'échelle (8B/70B) est le pari, pas un résultat.
4. Le déficit MMLU du 2 bits (−14,6 pp) est réel, reproduit trois fois,
   et non résolu — les suspects survivants sont documentés au rapport
   d'état.
5. Les ratios vitesse sont des médianes de rapports round par round avec
   plage ; jamais de troisième décimale.
