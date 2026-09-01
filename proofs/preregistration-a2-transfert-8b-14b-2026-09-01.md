# Pré-enregistrement — A2, transfert du graph aux 8B et 14B (2026-09-01)

> Suite de la branche ADOPTÉ du 4B (+13,45 % [13,36–13,58], job `6a96d5e1…`,
> journal `a2-verdict-2026-09-01.txt`) et de la discipline du gel v1 : « les
> trois tailles ou aucune ». Ce document ne s'édite jamais ; écarts →
> `proofs/preregistration-a2-transfert-8b-14b-2026-09-01-ECARTS.md`, nommé
> d'avance. Tampon avant la première milliseconde mesurée.

## La question

Le rejeu hybride (1er token de décodage éager, replay ensuite, store étendu
fenêtre 256) transfère-t-il aux deux autres tailles servies — et à quel
niveau ?

## Le protocole, figé — IDENTIQUE au 4B

`fusedrun` sous `LLVQ_GRAPH_AB=1 LLVQ_KV_PREALLOC=256`, config servie v1
de chaque taille, un processus par taille, gates AVANT tout chrono (capture
non-nulle ; tokens identiques éager/graph, re-vérifiés à chaque round),
5 paires de rounds entrelacées, r = graph/éager round par round. Artefacts :
les scellés du bucket déjà servis par la vague 2 (8B `77e76284`, 14B
`3f21abde`). l40sx1, ~0,3 $ + ~0,45 $ (*estimés* sur les durées vague 2).

## La lecture, posée d'avance — PAR TAILLE, seuils de phase inchangés

À chaque taille : **≥ 8 % → adopté** · **< 3 % → clos** · entre : point de
courbe. ET la règle de gel : **un gel v2 n'est possible que si LES TROIS
tailles atteignent la branche adopté** — un verdict mélangé laisse la
config servie à v1 et publie les trois points tels quels, sans gel.

## Le prior, déclaré — et il prédit un verdict MÉLANGÉ

Le gain de fusion décroissait avec la taille (×1,061 / ×1,055 / ×1,028) :
le pool par-lancement s'amortit sur des tokens plus longs. Transposé
(*calculé*, hypothèses déclarées : pool par-lancement ≈ constant en ms,
token 4B ≈ 10 ms → 8B ≈ 14,7 ms → 14B ≈ 23 ms aux débits v1) :
  8B ≈ +9 % (au-dessus du seuil, de peu) · 14B ≈ +6 % (POINT DE COURBE
attendu, sous le seuil d'adoption). Si le 14B rend ≥ 8 %, le prior est
réfuté dans le bon sens ; s'il rend < 3 %, dans le mauvais. Les deux bornes
du gate hybride (justesse d'abord) valent inchangées — un rouge de justesse
à une taille est un fait de fermeture-token propre à cette taille, à
consigner, pas à moyenner.

## Budget

~0,75 $ nominal, plafond 2 $ pour la vague (itérations comprises).
Phase A dépensée avant cette vague : 0,45 $ sur 4.
