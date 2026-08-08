# La campagne finale — quatre bras, seize cellules, un seul harnais (2026-08-07)

> Le tableau que le projet visait : FP16, le 4 bits officiel, notre LLVQ
> sans noyau, et notre LLVQ de dernière génération — tous mesurés sur la
> même L40S, le même harnais, les mêmes questions et les mêmes fenêtres,
> empreintes de tokens imprimées et identiques sur chaque ligne de qualité
> (**ppl `3f1baca9033bf251`, MMLU `65dcd53655e8bfa5`**). Sources :
> [`mesures/a4-campagne-2026-08-06.txt`](mesures/a4-campagne-2026-08-06.txt)
> (bras 1-3 qualité),
> [`mesures/campagne-finale-bras4-2026-08-07.txt`](mesures/campagne-finale-bras4-2026-08-07.txt)
> (bras 4 qualité + contrôle),
> [`mesures/nuit-planes12x-q8-2026-08-07.txt`](mesures/nuit-planes12x-q8-2026-08-07.txt)
> et [`mesures/phases-2026-08-07.txt`](mesures/phases-2026-08-07.txt)
> (vitesse/VRAM/attribution).

## Le tableau

| | FP16 | Q4 (AWQ officiel) | LLVQ sans noyau | **LLVQ planes14 + noyau fusé + embed q8** |
|---|---|---|---|---|
| **disque** | 8,04 Go | 2,67 Go | **1,77 Go** | **1,41 Go**⁴ |
| **VRAM** | 8,04 Go | 5,30 b/param¹ | 8,04 Go² | **2,60 Go (5,15 b/param)** |
| **vitesse** | 43,5 tok/s³ | non comparable¹ | 43,5 tok/s | **88,4-88,5 tok/s** |
| **perplexité** (wikitext) | 12,2369 | 13,5207 (×1,105) | 16,9422 (×1,384) | **16,9358 (×1,384)** |
| **MMLU** (micro) | 70,32 ± 1,28 | 70,04 ± 1,25 | 55,59 ± 1,35 | **55,70 ± 1,35** |

¹ L'AWQ n'a jamais tourné dans notre moteur : sa VRAM est celle de son
propre moteur (b/param modèle entier), sa vitesse ne se compare pas ici —
la réserve établie par la campagne A4, maintenue.
² Sans le noyau, le modèle se décode en f16 au chargement et tourne comme
du FP16 — c'était l'état du projet lundi.
³ Le bras FP16 et le bras LLVQ-dense partagent le même moteur f16 (mêmes
formes, mêmes noyaux) : vitesse et VRAM identiques par construction,
recoupé par le protocole miniature (42,8 tok/s sur le checkpoint).

## Les trois lectures du tableau

**1. Le noyau et le format valent ×2,03 et ÷3,09 — gratuits en qualité.**
Entre les colonnes 3 et 4, le même modèle au bit près⁴ : le passage
au noyau fusé + Planes14 + embedding q8 double la vitesse et divise la
mémoire par trois, pendant que la perplexité et le MMLU restent dans le
bruit d'échantillonnage (16,9358 contre 16,9422 ; 55,70 contre 55,59 —
mêmes empreintes, mêmes questions). C'est la contribution d'ingénierie du
projet, mesurée de bout en bout.

**2. Face au FP16 : dominance nette, un prix de qualité connu.** ×2,03 en
vitesse, ÷3,09 en mémoire, ÷4,5 sur disque ; le prix est ×1,384 de
perplexité et −14,6 pp de MMLU — le coût du 2 bits sur un 4B, inchangé
depuis le fichier publié.

**3. Face au 4 bits : chaque axe a son vainqueur, et il faut les dire
tous.** Nous gagnons le disque (1,77 contre 2,67) et désormais la **VRAM**
(5,15 contre 5,30 b/param — c'était l'axe perdu il y a trois jours) ; la
vitesse ne se compare pas honnêtement (moteurs différents) ; l'AWQ gagne
la qualité, largement (70,0 contre 55,7 de MMLU). Sur un 4B, le verdict
d'A4 reste vrai sur l'axe capacités ; le pari du 2 bits reste l'échelle
(le 8B se dégrade déjà moins : ×1,267 contre ×1,384).

## La provenance du ×2,03, pour les relecteurs

Attribution par phases ([`mesures/phases-2026-08-07.txt`](mesures/phases-2026-08-07.txt)) :
~2,9 ms/token viennent du noyau Leech sur les projections, ~25 ms du
remplacement du chemin lm_head de candle (qui recopie 778 Mo par token —
le `TODO` est dans son code). Formulation double : **×2,03 contre le
moteur de référence tel que tout le monde l'utilise ; ~×1,4 contre ce
moteur si on lui corrigeait sa copie** (estimé des phases mesurées).

## Coût de l'ensemble

Campagne A4 (bras 1-3) : 0,71 $. Bras 4 : 0,47 $. Toute la séquence
C1 → campagne finale (layouts, branchement, embedding, phases, E2, cette
campagne) : **2,85 $** de GPU (détail job par job :
[`data/jobs.csv`](data/jobs.csv)).
