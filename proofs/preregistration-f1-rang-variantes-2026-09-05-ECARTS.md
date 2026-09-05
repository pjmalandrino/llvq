# Écarts au pré-enregistrement des trois arithmétiques — écrits après le job, jamais dans le tampon

> Le pré-enregistrement
> [`preregistration-f1-rang-variantes-2026-09-05.md`](preregistration-f1-rang-variantes-2026-09-05.md)
> (sha256 `a3765c0a…`) est tamponné, et le journal
> [`docs/mesures/f1-rang-variantes-2026-09-05.txt`](../docs/mesures/f1-rang-variantes-2026-09-05.txt)
> est un fait brut : ni l'un ni l'autre ne s'édite. Ce qui les corrige s'écrit ici.

## É1 — La prédiction signée est fausse sur les quatre temps, dans le sens instructif

Prédit Du_v1 1,8–2,3, Du_v2 2,0–2,5, Du_v3 1,6–2,2, T_v* 2,6–3,1 ms ; mesuré **1,074 / 2,802 / 1,051 /
1,694**. V1 et V3 sont bien en dessous de leur fourchette, V2 au-dessus, T_v* à 1,1 ms sous le bas de la
sienne. La clause « fausse de façon instructive » (Du_v3 < 1,3 ms) a tiré : l'arithmétique — les 24
conversions entier → flottant par bloc — était presque tout le surcoût mesuré le matin (~1,6 des ~2,0 ms), et
la chaîne de petites lectures dépendantes, l'autre suspect nommé, n'en était aucun (V2 seul : +0,112 ms).
Le §7 attribuait ~0,6 ms aux I2F non masquées et le reste à la chaîne ; c'était l'inverse.

## É2 — v* se lit à 0,025 ms près, sous la résolution déclarée

T_v3 = 1,694 [1,691 ; 1,701] et T_v1 = 1,719 [1,709 ; 1,723] : plages disjointes, mais l'écart est sous
les ±0,1 ms que le §6 donne comme résolution de ces bancs, et les deux sont à 40 registres. Le §6 dit « la
variante de T_v minimal » : v3. Pour F1d, v1 et v3 valent ; le choix entre les deux se fera dans le noyau
fusé, où le compte d'instructions ALU (V3 en a plus, sur le pipe lent) et la chaîne FMA (V1 = celle de
`tv_f1r`, V3 somme par bloc) pèsent autrement.

## É3 — Le job a coûté 4 s de carte après 50 min d'attente

`running_secs = 4` (l'attente, 3 004 s, ne se facture pas) : 0,00 $ au centime. Le plafond machine de
10 min affichait « au pire 0,30 $ » ; c'est un garde-fou, pas une dépense (précédents M2b, plancher du 05).

## É4 — Ce qui n'est pas un écart

Le contrôle 7 par ligne, la mise hors jeu d'une variante fautive et la correction de V3 (`__byte_perm` ne
sait pas répliquer le signe) ont été portés dans le texte AVANT `ots stamp` ; le tampon les couvre. Le
contrôle 7 n'a mis personne hors jeu.
