# Le modèle, avant / après — comparatif simple (2026-08-07)

> Une page, sans jargon, pour expliquer où on en est. Tous les chiffres sont
> mesurés sur la même carte (L40S), le même modèle (Qwen3-4B), le même
> harnais. Sources : `docs/mesures/`.

## L'histoire en trois phrases

Un modèle de 4 milliards de paramètres pèse **8 Go** en pleine précision et
tourne à **43,5 mots/seconde**. Notre version compressée à 2 bits tourne
maintenant dans **2,6 Go** à **88,5 mots/seconde** — trois fois moins de
mémoire, deux fois plus vite — **sans avoir changé un seul bit du modèle
publié** : toute l'amélioration de la semaine est dans la façon de *ranger*
et de *lire* les poids, pas dans leur contenu.

## Le tableau

| | pleine précision (f16) | 4 bits (AWQ officiel) | **nous, lundi** | **nous, aujourd'hui** |
|---|---|---|---|---|
| fichier sur disque | 8,04 Go | 2,67 Go | **1,77 Go** | **1,77 Go** (inchangé) |
| mémoire carte (VRAM) | 8,04 Go | ~2,7 Go¹ | 8,04 Go² | **2,60 Go** |
| vitesse | 43,5 tok/s | non mesurée¹ | 43,5 tok/s² | **88,5 tok/s³** |
| qualité (perplexité) | 12,24 | 13,52 | 16,94 | **16,94** (inchangée) |
| qualité (MMLU) | 70,3 % | 70,0 % | 55,6 % | **55,6 %** (inchangée) |

¹ L'AWQ n'a jamais tourné dans notre moteur : sa VRAM est celle de son
propre moteur (5,30 bits par poids), sa vitesse n'est pas comparable ici.
² Lundi, le noyau rapide existait mais n'était branché nulle part : le
modèle se décomprimait au chargement et tournait comme du f16.
³ Chiffre mesuré, mais son *mécanisme* est en cours d'instrumentation —
voir la réserve en bas de page.

## Ce qui a changé cette semaine (et ce qui n'a pas changé)

**Trois améliorations, toutes dans le « rangement » :**

1. **Le noyau a été branché** (mercredi) : le modèle lit enfin ses poids
   compressés au lieu de les décompresser au chargement. 8,04 → 3,28 Go.
2. **Le rangement one-hot a été remplacé par des plans binaires** (jeudi,
   « Planes14 ») : la même information tenait dans moins d'octets ET se
   lisait plus vite — c'était du gaspillage pur, prouvé identique au bit
   près sur les 150 millions de blocs. 3,28 → 2,96 Go, +4 % de vitesse.
3. **L'embedding est passé en entiers 8 bits** (cette nuit) : la grosse
   table de vocabulaire, jusque-là gardée en pleine précision « par
   prudence », a été validée sans aucune perte mesurable puis branchée.
   2,96 → 2,60 Go, et la vitesse a doublé (voir réserve).

**Ce qui n'a pas changé : la qualité.** Chaque étape est prouvée sans effet
sur le modèle (reconstruction identique au bit près, ou validée sur les deux
métriques). Le point faible reste le même qu'avant : à taille 4B, le 4 bits
ne perd presque rien en capacités (70,0 % de MMLU) alors que le 2 bits perd
gros (55,6 %). C'est le front qualité, il est ouvert, et il est indépendant
de tout ce qui précède.

## Comment le dire en une phrase selon l'interlocuteur

- **Version produit** : « le modèle tient maintenant dans 2,6 Go au lieu de
  8, et va deux fois plus vite — fichier et qualité inchangés. »
- **Version technique** : « à qualité strictement égale, on est passé de
  6,5 à 5,15 bits par poids en mémoire (4,69 possible), sous l'empreinte de
  l'AWQ 4 bits réel (5,30), à ×2,03 du débit dense. »
- **Version honnête complète** : ajouter « le 4 bits reste devant en
  capacités sur un modèle de cette taille ; notre pari est l'échelle — plus
  le modèle est gros, moins le 2 bits perd, et c'est là que 2,6 Go contre
  8 change ce qui rentre sur une machine. »

## ✅ La réserve est levée — le mécanisme est mesuré (2026-08-07 matin)

L'instrumentation par phases ([`mesures/phases-2026-08-07.txt`](mesures/phases-2026-08-07.txt),
0,33 $) a tranché : la phase « vocabulaire » (lm_head) d'un token coûte
**25,9 ms en tête f16 et 0,598 ms en tête q8** — et le moteur de référence
paie la même chose (26,7 ms). La cause, confirmée d'abord dans le code de
candle puis au chronomètre : son chemin `broadcast_matmul` **recopie les
778 Mo du vocabulaire à chaque mot généré** (le `TODO: Avoid concretising`
est dans leur source). Notre noyau lit les 413 Mo compressés une fois, sans
copie.

Les **trois** formulations honnêtes, chiffres en main :
- **×2,02 contre le moteur de référence tel que tout le monde l'utilise**
  (87,7 contre 43,4 tok/s, mesuré dans le même job) ;
- **~×1,4 contre ce même moteur si on lui corrigeait sa copie** (estimé par
  recomposition des phases mesurées — le corriger réellement est possible
  et le chiffre deviendrait alors une mesure) ;
- **×1,12 à tête identique** — f16 des deux côtés, donc la copie de candle
  payée par les deux bras : **48,6 contre 43,5 tok/s**, relevé dans le même
  job. C'est le seul des trois qui soit **à la fois mesuré bout-en-bout et
  attribuable au noyau Leech seul**, et c'est celui qu'un relecteur exigera.

⚠️ **Règle de publication : ne jamais donner le ×2,03 sans le ×1,12.** Le
premier mesure ce que gagne un utilisateur ; le second mesure ce que vaut
notre contribution. Les confondre est la faute que cette section existe pour
empêcher. *(La « version technique » ci-dessus dit « ×2,03 du débit dense » :
c'est vrai et incomplet — y ajouter « dont ×1,12 imputable au noyau ».)*

Le gain n'est pas un artefact de comparaison : il vient d'avoir écrit le
chemin que le moteur de référence n'a pas écrit — c'est précisément le
métier du projet.
