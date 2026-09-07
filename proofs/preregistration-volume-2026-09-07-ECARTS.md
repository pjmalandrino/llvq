# Écarts au pré-enregistrement du volume de calibration

> Le pré-enregistrement
> [`preregistration-volume-2026-09-07.md`](preregistration-volume-2026-09-07.md)
> (sha256 `8a765e9d…`) est tamponné. Il ne s'édite pas.

## É1 — Le bras V96 est impossible sur cette carte, et le préreg ne pouvait pas le savoir

**Ce que le préreg demande** : `smoke 6144 2048 12 4096 cuda nogs tetra 999 rot`,
soit 12 582 912 tokens de calibration.

**Ce qui s'est passé** : `Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY)` après le
tokenisage, avant le premier bloc. Job `6a9e49b1259f8e97255e9a24`,
`rtx-pro-6000x2`, 2,9 min, **0,26 $**.

**La cause, et c'est un plafond du dépôt que personne n'avait écrit** :
`quantize_model(model, hidden: &mut [Tensor], …)`
(`llvq-llm/src/calib.rs:573-579`) garde **toutes** les fenêtres de calibration
résidentes sur l'accélérateur et les fait avancer bloc par bloc. Le coût est
donc linéaire en volume, et il est connu d'avance :

```
  une fenêtre = 2048 tokens × 2560 hidden × 4 o = 21,0 Mo
  modèle f32 16,1 Go · hessiennes (2·2560² + 9728²) f32 0,43 Go

  ×96  6144 fenêtres = 128,8 Go de hidden → 145,4 Go   sur 96 disponibles
  ×48  3072          =  64,4             →  80,9       tenu, sans marge
  ×32  2048          =  42,9             →  59,5       confortable
  ×16  1024          =  21,5             →  38,0
```

**Le volume de calibration de ce dépôt est plafonné par la VRAM, pas par le
corpus.** Le shard C4 en offrait assez — le journal du job l'imprime :
« 6144 windows of 2048 = 12582912 tokens (7960 available) ». La contrainte est
ailleurs, et elle explique après coup pourquoi le volume du papier n'a jamais
été essayé : sur cette architecture il ne tient sur aucune carte du catalogue.
Notre 131 072 tokens n'était pas un choix mesuré, c'était un point jamais poussé.

**Ce qui suit** : le bras de traitement doit être relancé à un volume qui tient,
ou le code doit apprendre à découper la capture. Le choix est une décision
d'opérateur, et le §4 du préreg s'appliquera au volume effectivement mesuré, en
le nommant. `ΔV` ne se lira donc pas comme « le volume du papier » mais comme
« le volume atteint », ce qui est une borne inférieure de l'effet cherché si
l'effet est monotone — et rien ne garantit qu'il l'est.

## É2 — Le témoin V1c n'est pas touché

Le bras `6a9e49b1259f8e97255e9a26`, 131 072 tokens, tient dans 38 Go et tourne.
Il reste le témoin de device du §2, il garde son sens quel que soit le volume
retenu pour le bras de traitement, et son `ΔD` se lit comme prévu.

## É3 — La flavor n'est pas `l40sx1`, et `ops/run.py` l'a refusée avant de céder

Le préreg nomme `rtx-pro-6000x2`. `ops/run.py` refuse toute flavor hors
`l40sx1` pour les bancs, afin qu'aucun ratio de vitesse ne se compare entre
cartes ; l'option `--any-flavor` a été posée, et la consigne du refus est
tenue ici : **les chiffres de ce banc sont de la qualité, pas de la vitesse, et
la flavor est nommée dans le journal.**
