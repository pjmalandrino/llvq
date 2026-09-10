# Pré-enregistrement — F1e §0 : le chemin servi tourne-t-il sur une carte ?

**Écrit, commité et TAMPONNÉ le 2026-09-10, AVANT le lancement.** Go permanent
de l'opérateur du 2026-09-09 : « enchaîne sur les 6.8 et 6.9 avec 20 $ de
plafond ».

Ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md`.

**Coût : ~0,40 $** (*estimé*, un job `l40sx1` d'une dizaine de minutes de
calcul : chargement du 4B scellé, `oracle`, puis 64 jetons). Plafond de timeout
**30 min**, soit **0,90 $ au pire**. Cumul projet **137,74 $** ; wave 3 :
**3,48 $ dépensés sur 20,00 $**.

---

## §1 — Pourquoi ceci et pas F1e tout de suite

F1e coûte ~8 $, un tiers du plafond, et sa première ligne est un noyau qui
**n'a jamais tourné sur une carte** : `tv_q4_h` est écrit, vérifié comme C++
hôte par `llvq-llm/tests/proj_q4.rs`, et n'a jamais été lancé
(ROADMAP §2.2 quinquies). `tv_tetra48_h` non plus — F1d a mesuré le bras de
banc `tv_f1r_v3g`, pas le noyau servi dans le modèle.

Dépenser 8 $ pour découvrir qu'un des deux ne charge pas serait la dépense que
les règles de ce dépôt existent pour éviter. Cette étape est un prérequis de
F1e **quelle que soit** l'arbitration de l'opérateur sur F1d.

## §2 — Ce qui est mesuré

Un seul job `l40sx1`, deux commandes :

1. **`oracle`** sur le backend cuda — règle dure 10, la passe avant contre
   `candle-transformers`. Elle ne dit rien du noyau ; elle dit que le harnais
   est celui des autres runs.
2. **`fusedrun`** sur `tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin`
   (1 794 564 765 o, l'objet servi arbitré le 2026-09-08 : 216 Tetra + 36 int4
   g128), avec `LLVQ_FUSED_LAYOUT=tetra48`, 64 jetons.

Le fichier a été téléversé dans le seau le 2026-09-10, taille vérifiée
identique à l'octet.

## §3 — Ce qui compte comme un succès

| ce qu'on lit | ce que ça établit |
|---|---|
| l'unité NVRTC compile et les deux noyaux se chargent | `tv_tetra48_h` et `tv_q4_h` existent sur carte |
| les registres et `local_bytes` des deux | le contrat de non-débordement |
| les jetons produits sont du texte, pas du bruit | le décodage servi est juste dans le modèle |
| la VRAM tenue | le premier chiffre de mémoire *mesuré* et non calculé |
| tok/s sur 64 jetons | un ORDRE DE GRANDEUR, pas la mesure de F1e |

**Aucune porte ne se lit ici.** La barre de 100,6 tok/s est celle de F1e, sur
son protocole ; 64 jetons ne la mesurent pas — le chargement domine.

## §4 — Ce qui serait un échec, et ce qu'il coûterait

Un refus de `fused.rs` par `kind`, un noyau absent de la liste de sources, un
`illegal memory access` sur le premier lancement de `tv_q4_h`. Chacun se
corrige sur le Mac pour 0 $ et se relance pour 0,40 $ — contre 8 $ découverts
au milieu d'un recensement MMLU.

## §5 — La décision que je NE prends pas

F1d a rendu un résultat **renversé d'une carte à l'autre** : Tetra 1,17× plus
rapide que Planes14 sur Ada à la tuile servie, 0,82× sur Blackwell. La clause
de tuerie de la ROADMAP — « plus lent pour moins d'octets » — est donc vraie
sur une carte et fausse sur l'autre.

**C'est un arbitrage d'opérateur et je ne le prends pas.** Le recensement MMLU
complet de F1e (~8 $) attend qu'il ait vu le tableau. Ce job-ci ne l'engage
pas : il coûte 0,40 $ et il est nécessaire dans tous les cas de figure.

## §6 — Les prédictions signées

| # | prédiction | intervalle |
|---|---|---|
| Q1 | l'unité NVRTC compile du premier coup | oui |
| Q2 | `tv_tetra48_h` : registres | **40** [36 ; 48] |
| Q3 | `tv_q4_h` : registres, et 0 octet local | ≤ 40, 0 local |
| Q4 | VRAM tenue par le modèle servi | **1,5 Go** [1,2 ; 2,0] |
| Q5 | quelque chose refuse au premier essai | **oui, 60 %** |

Q5 est écrite parce que les quatre étapes précédentes de ce chantier ont chacune
buté sur une plomberie de lancement (É1 à É5 de la tuerie), et parce qu'un
noyau qui n'a jamais tourné a une probabilité qui n'est pas petite de ne pas
tourner.
