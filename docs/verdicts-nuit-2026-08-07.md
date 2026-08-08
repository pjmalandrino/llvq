# Verdicts de la nuit du 06 au 07 — trois marches, deux verdicts verts, un rouge

> Trois chantiers menés dans la nuit (implémentation + revue adversariale
> chacun, puis mesure), 0,90 $ de GPU, le Mac pour le reste. Mesures brutes :
> [`mesures/nuit-planes12x-q8-2026-08-07.txt`](mesures/nuit-planes12x-q8-2026-08-07.txt)
> et [`mesures/m3-gate-design-c-2026-08-07.txt`](mesures/m3-gate-design-c-2026-08-07.txt).

## M1 ✅ — L'embedding int8 en production : ×2,03 bout-en-bout, 2,60 Go

`LLVQ_EMBED=q8` : quantification au chargement par la fonction même de
`bin/embedq` (la validation qualité du lot B se transfère structurellement —
bit-identité vérifiée contre les octets du fichier scoré), un seul buffer
int8 pour l'embedding et le lm_head lié, deux noyaux neufs, zéro spill.

| | dense f16 | fusé planes14 + q8 |
|---|---|---|
| tok/s | 43,5 | **88,5 (×2,03)** |
| Go carte | 8,04 | **2,60 (÷3,09)** |
| tokens | référence | identiques jusqu'au tie-break token 89 |

⚠️ **Le saut de 48,7 à 88,5 tok/s dépasse de loin le trafic du lm_head**
(~0,6 ms attendus). Mécanisme attribué : le noyau q8 remplace le chemin
candle `broadcast_matmul` sur vue transposée du head — qui payait
manifestement bien plus que les 1,18 ms du banc. **À instrumenter avant d'en
faire un titre** ; le chiffre, lui, est mesuré (même job, bras dense en
contrôle). En b/param modèle entier : **~5,15 — sous les 5,30 de l'AWQ réel**.

> ✅ **Instrumenté le jour même (2026-08-07), et l'attribution est confirmée.**
> Phases par token, fences device : `lm_head` **25,886 ms** sur le bras fusé à
> tête f16 et 26,672 ms sur le bras dense, contre 10,439 ms pour tous les
> blocs + normes. Candle recopie les 778 Mo du vocabulaire à chaque token — le
> `TODO` est dans son propre code. Le noyau q8 ramène ce poste à ~0,6 ms.
> **Conséquence de publication, et elle est contraignante : le ×2,03 mesure
> une correction du moteur de référence, pas le noyau Leech.** À tête
> identique — f16 des deux côtés — le même job rend **×1,12** (48,6 contre
> 43,5 tok/s). **Ne jamais publier le ×2,03 sans le ×1,12 à côté.**
> Source : [`mesures/phases-2026-08-07.txt`](mesures/phases-2026-08-07.txt).

## M2 ✅ — L'overlay épars tient : 4,342 b/poids à qualité exacte, 2,01×

Banc à 4 bras, overlay prouvé exact sur les 1 105 920 lignes (pire erreur
2,9e-8, identique aux autres bras au bruit près) :

| bras | méd ms | b/poids | vs FP16 |
|---|---|---|---|
| Slot32 | 5,818 | 5,510 | 1,89× |
| **Planes14** | **5,094** | 4,804 | **2,16×** |
| **Planes12x** | 5,476 | **4,342** | 2,01× |

Planes12x paie sa correction atomique 7 % contre Planes14 (0,93×) mais reste
devant l'ancien Slot32 **sur les deux axes**. L'échelle a deux points de
fonctionnement : **Planes14 pour la vitesse, Planes12x pour les bits** —
qualité identique par construction dans les deux cas. (Le memset large et la
correction sont facturés au bras 12x ; comptabilité des pads unifiée.)

⚠️ **Statut au 2026-08-08 : `Planes14` est le layout par défaut et il est
branché ; `Planes12x` est validé au banc et n'est PAS branché.** Le choix
produit entre les deux reste à prendre. Ne pas écrire « Planes12x est en
production » — aucune mesure bout-en-bout ne l'a jamais exercé.

## M3 ❌ — Le design C est RÉFUTÉ à pleine profondeur : ×1,99 de dégradation

Gate 0.6B, 28 blocs, 131k tokens, une variable, protocole du run publié :

| bras | ppl (baseline 19,5038) | b/poids |
|---|---|---|
| chemin publié | 35,9806 | 2,2068 |
| **design C** | **71,4249 (×1,99)** | 2,2068 |

Le 4B n'est **pas parti** — le gate automatique l'a bloqué, comme prévu.

**Ce que ce rouge établit, et il est précieux :**

1. **Le suspect n°1 du déficit MMLU est mort tel qu'implémenté.** La
   fidélité au doc (`retraction-et-gain.md`, design C) a été vérifiée en
   revue adversariale point par point, le solve est l'algèbre verrouillée du
   crate, la re-projection est scellable, le défaut de signe trouvé en revue
   était corrigé avant le run — et le proxy local décroît *strictement*
   (test vert). Le mécanisme promis par la Table 9 ne se transfère pas par
   ce chemin dans notre chaîne. Réserve honnête : c'est *notre lecture* du
   design C qui est réfutée ; le papier n'en donne pas le pseudo-code.
2. **C'est la deuxième occurrence du même motif** : proxy local meilleur,
   composition à 28 couches désastreuse (group_scales : 21,24 → 21,17 sur
   3 blocs, 44,66 → 53,60 sur 28 ; design C : proxy strictement meilleur
   par couche, ×1,99 au total). **La rigidité de norme de la rétraction
   sphérique est porteuse à profondeur** — ce n'est plus une anecdote, c'est
   un fait de méthode du pipeline, et il mérite une place dans le README.
3. **Les suspects restants du −14,7 pp**, par ordre : la config 1 bit de
   gain (écart 0↔2 bits : 1,4 pp, run 4B + MMLU à payer), la composition du
   corpus (P18, mécanisme raisonnement), la compensation post-hoc
   (EoRA/Recover-LoRA, gains publiés +4-11 pp), le FT échelles (P17,
   +2,1 pp). Et l'issue stratégique inchangée : **l'axe d'échelle** (le 8B
   se dégrade déjà moins ; jamais mesuré en MMLU).

## L'état de l'échelle format après cette nuit

| point | b/poids payload | b/param modèle entier (embed q8) | vs FP16 (banc) |
|---|---|---|---|
| Slot32 (il y a 48 h) | 5,510 | 6,52 (embed f16) | 1,89× |
| Planes14 | 4,804 | **5,15** | 2,16× |
| Planes12x | 4,342 | **4,69** | 2,01× |
| AWQ réel (repère) | — | 5,30 | jamais mesuré chez nous |
| MLX q4 (repère) | — | 4,50 | — |

**Planes12x + embedding q8 = 4,69 b/param : sous l'AWQ réel, à 4 % du MLX
q4 — à qualité LLVQ strictement identique au fichier publié.**

> 🕳️ **Correction du 2026-08-08 — cette section se terminait par « la marche
> suivante de l'échelle reste E2 (Golay, ~3,3 payload), non commencée ».
> C'était faux au moment même où c'était écrit** : E2 a été implémenté et
> mesuré **le jour même**, et il est **écarté**. Le format tient — la
> reconstruction est exacte sur les 150,7 M blocs, l'information Golay est bien
> recomputable — mais le payload réel est **3,589 b/poids** (pas ~3,3 : la
> fourchette venait de la branche haute de l'histogramme B5, et même elle était
> optimiste) et le noyau ne rend que **1,31× vs FP16**, sous le critère de 1,6×
> posé d'avance. Le décodage à double coset borne le noyau en ALU : 195 Go/s
> effectifs, contre 425 pour `Planes14`.
> **L'échelle s'arrête donc proprement à `Planes14` / `Planes12x`.** Pistes
> notées et non poursuivies : spécialiser les warps par coset, ou ne payer le
> XOR que côté pair.
> Sources : [`mesures/e2-golay70-bench-2026-08-07.txt`](mesures/e2-golay70-bench-2026-08-07.txt),
> [`rapport-etat-2026-08-07.md`](rapport-etat-2026-08-07.md) §3 et §6.
