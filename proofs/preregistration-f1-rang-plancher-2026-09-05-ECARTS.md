# Écarts au pré-enregistrement du plancher compilé F1 — écrits après le job, jamais dans le tampon

> Le pré-enregistrement
> [`preregistration-f1-rang-plancher-2026-09-05.md`](preregistration-f1-rang-plancher-2026-09-05.md)
> (sha256 `119b02d5…`) est tamponné, et le journal
> [`docs/mesures/f1-rang-plancher-2026-09-05.txt`](../docs/mesures/f1-rang-plancher-2026-09-05.txt)
> est un fait brut : ni l'un ni l'autre ne s'édite. Ce qui les corrige s'écrit ici.

## É1 — S est sous la fourchette, et ce n'est pas un débit

Prédit 0,9 à 1,3 ms ; mesuré **0,645 ms** [0,639 ; 0,650]. 0,908 Go en 0,645 ms
feraient 1,41 To/s, au-dessus de la HBM. Le §3 lisait `S` comme « le flux F1
dans notre géométrie » ; c'est ce que la lecture du mot ajoute à `nullk`, un
plancher de latence et d'occupation que le flux recouvre en partie. `S` est
une différence, pas une bande passante, et la fourchette « 700–980 Go/s »
n'avait pas de sens pour elle. `T` reste la seule lecture ; `Du` hérite de
l'ambiguïté (ce que le flux cachait sous `nullk` retombe dans `Du`).

## É2 — Du est 1,7 fois au-dessus du haut de la fourchette

Prédit 0,9 à 1,6 ms ; mesuré **2,701 ms** [2,695 ; 2,709]. La table seule à
16 Kio valait 0,663 au plancher du 05 ; le reste, ~2,0 ms, est l'arithmétique
et les trois petites lectures. Le §8 extrapolait « ~250 opérations, masquées à
~0,2 ms comme les ~30 de hash3 » : le masquage de 30 opérations entières ne
dit rien de 24 conversions entier → flottant (pipe I2F à 1/8 du débit FMA sur
sm_89) ni d'une chaîne de trois lectures dépendantes. Ce sont les deux
suspects, non séparés par ce banc (*estimé*, journal).

## É3 — T est au-dessus de la fourchette, dans la deuxième ligne du §7

Prédit 2,0 à 3,0 ms, centre 2,5 ; mesuré **3,346 ms** [3,342 ; 3,354] =
1,20 × B. La ligne « 2,797 < T ≤ 3,50, registres ≤ 64, locaux = 0 » s'applique :
F1d s'écrit avec ce décodeur, projection publiée (+0,55 ms de passe, ~95 tok/s
au 4B, *estimé*), et l'arbitrage contre la VRAM et la classe est à l'opérateur.
La clause « fausse de façon instructive » visait T > 3,5 avec registres < 64 —
manquée de 0,15 ms ; le mécanisme qu'elle nommait (la chaîne de petites
lectures non masquée) reste le premier suspect.

## É4 — Le plafond machine du job affichait 0,30 $

`ops/run.py bench --timeout 10m` annonce « at worst 0.30 $ » ; le plafond
propre du §0 est 0,10 $. Le job a coûté 0,01 $ (17 s). Comme au M2b du 04, le
timeout est un garde-fou machine, pas une autorisation de dépense.

## É5 — Ce qui n'est pas un écart

Les deux corrections signalées par la relecture avant le tampon — 0,908 Go et
non 0,98, contrôle 1 sur 256 blocs en ordre ligne-majeure par copie — ont été
portées dans le texte AVANT `ots stamp` ; le tampon les couvre.
