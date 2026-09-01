# Pré-enregistrement — A2 étapes 2+3 : la capture, et l'A/B graph (2026-09-01)

> Sous le préreg de phase `preregistration-a2-a3-geometrie-2026-08-31.md`
> (sha256 `802006c5…`, tamponné), dont les seuils sont GELÉS et repris ici
> tels quels. Ce document ne s'édite jamais ; écarts →
> `proofs/preregistration-a2-etapes2-3-graph-2026-09-01-ECARTS.md`, nommé
> d'avance. Tampon avant la première milliseconde mesurée sur carte.

## La question

Le rejeu par CUDA Graph du pas de décodage rend-il, sur le chemin servi v1 à
store étendu (é1b), le gain que les seuils de phase exigent ?

## Les gates, DANS L'ORDRE — un rouge arrête tout

1. **La capture existe** : stream frais (`new_cuda_with_stream`), event
   tracking coupé avant toute allocation, `end_capture` rend un graph
   non-nul. Un `None` = stream legacy = défaut de câblage, pas une mesure.
2. **Justesse AVANT tout chrono** : les 128 tokens du chemin graph sont
   IDENTIQUES à ceux du chemin éager, même processus, même StepState — une
   divergence signifie qu'une variation par-token a échappé à l'inventaire,
   et on ne chronomètre pas un chemin faux. (Certifié en amont à 0 $ : test
   CPU à modèle aléatoire, témoin Metal sur le 4B scellé — mais la carte est
   le seul juge du graph lui-même.)
3. **La mesure** : 5 paires de rounds entrelacées éager/graph, UN processus,
   UN modèle, UN StepState, UN buffer de logits — la seule variable est
   exécuter ou rejouer. `r = tok/s(graph) ÷ tok/s(éager)`, formé round par
   round, médiane et plage. Chaque round re-gate les tokens.

## La lecture — les seuils de PHASE, gelés le 2026-09-01, appliqués tels quels

- gain bout-en-bout **≥ 8 % → adopté** ; **< 3 % → clos** ; entre les deux :
  point de courbe, non adopté.
- Le **net contre la config servie v1** = ce gain **− 0,83 %** (le coût
  mesuré de la base fixe, é1b, r = 0,9917) — c'est ce net qui entre dans le
  kill de phase (A1+A2+A3 < 8 % cumulés → axe clos).
- ⚠️ Les deux bras incluent le préfill (protocole `fusedrun` inchangé) :
  symétrique, il dilue r vers 1 d'au plus ~4 % de part de temps — déclaré.

## Le prior, déclaré — défavorable à l'adoption

Le pool par-lancement mesuré (A1) : 3,76 µs/lancement ; ~1 050
lancements/token (*estimé*, cadrage §1) dont ~290 nôtres ; le plafond si le
graph récupérait 0,66 µs/lancement (la seule récupération jamais MESURÉE,
lot A 08-06, sur noyaux quasi nuls) ≈ **0,7 ms ≈ 7 % d'un token** — un
plafond optimiste, pas une attente. F3 : la soumission hôte est déjà
recouverte à 0,1-0,2 %. **Attendu : entre « clos » (< 3 %) et « point de
courbe »** ; l'adoption exigerait que le rejeu gagne aussi ce que le banc nu
ne portait pas. On mesure parce que `deaa449` l'a décidé et parce que le
kill de phase a besoin du chiffre — pas parce qu'on attend un vert.

## Budget

~0,15 $ le job vert ; jusqu'à ~0,5 $ d'itérations si la capture révèle une
variation manquée (chaque mort est un fait de fermeture-token à consigner).
Phase A dépensée : 0,18 $ sur 4.
