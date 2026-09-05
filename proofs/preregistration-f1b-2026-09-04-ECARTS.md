# Écarts au pré-enregistrement F1b — écrits après la mesure, jamais dans le tampon

> Le pré-enregistrement
> [`preregistration-f1b-2026-09-04.md`](preregistration-f1b-2026-09-04.md)
> (sha256 `fe93b67d…`) est tamponné, et le journal
> [`docs/mesures/f1b-retention-2026-09-04.txt`](../docs/mesures/f1b-retention-2026-09-04.txt)
> est un fait brut : ni l'un ni l'autre ne s'édite. Ce qui les corrige s'écrit ici.

## É1 — Le contrôle §4.5 n'a pas été fait comme écrit

**Ce que le préreg demande** : « le même encodeur à balayage tourne sur la
boule-12, où `nearest_angular` donne la réponse exacte ; l'écart mesuré est
publié. »

**Ce qui a été fait** : le témoin boule-12 est calculé par la réduction en
coquilles exacte (`precompute13` puis `t_ball12`, m ≤ 12), pas par l'encodeur à
balayage. La suboptimalité de l'encodeur F1 est bornée autrement, en deux
morceaux :

- `llvq-bench/examples/f1enc.rs` compare l'encodeur de section à la recherche
  exhaustive sur la région : **100 % d'optima exacts** sur 400 cibles par
  section, pour les trois sections (*mesuré*, rejoué le 2026-09-05) ;
- le balayage d'échelle ne produit que des points réalisables, donc il ne peut
  que sous-estimer F1 (*raisonnement*, `f1.rs`, doc de `Codebook`).

**Pourquoi c'est un écart et pas une équivalence** : le contrôle du préreg
aurait mesuré l'écart du balayage lui-même sur un codebook où la vérité est
connue. Le remplacement borne la section et argumente le balayage ; il ne
mesure pas le balayage. Le biais est à sens unique et contre F1, donc le
89,55 % est une borne basse et non un nombre flatté — mais de combien, ce
dépôt ne le sait pas. Le pilote à 64,65 % (grille d'échelle du mauvais côté du
pic) montre que ce terme peut être énorme quand la grille est fausse.

## É2 — Deux découpes sur 2 000 blocs au lieu de 20 000

Le §3 fixe 20 000 blocs. Seule 12/15/12 les a (4 000 + 16 000). 12/16/11 et
13/13/13 sont sur 2 000 blocs d'évaluation, par coût (7 min chacune contre
64 pour la découpe principale). Leurs nombres, 89,38 et 85,96, se citent avec
cette mention. L'ordre des trois découpes ne dépend pas de l'échantillon.

## É3 — Un pilote faux avant le run publié, sans nouveau tampon

Le premier run a rendu 64,65 %. Cause : grille d'échelle 0,600 à 3,706, tout
entière au-delà du pic de `t` (s ≈ 0,344, `examples/f1scale.rs`). Grille
corrigée à 0,10 à 0,93 ; 90,27 % sur 2 000 blocs, puis 89,55 % sur 16 000.
La grille n'est pas dans le préreg (qui fixe la règle de score, le débit, le
témoin et la troncature), donc aucun tampon n'a été refait. Ce serait à
refaire si la grille était un jour considérée comme faisant partie du
protocole : elle décide d'un quart du résultat.

## É4 — Le journal a été écrit le lendemain

Le préreg cite `docs/mesures/f1b-*` ; le journal n'existait pas le soir du 04.
Il est écrit le 05 à partir de la sortie brute conservée dans la transcription
de la session (`f1b-retention-2026-09-04.txt`, section « la sortie brute »).
