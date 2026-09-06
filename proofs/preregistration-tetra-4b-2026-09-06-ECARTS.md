# Écarts au pré-enregistrement du 4B en Tetra — écrits après le tampon, jamais dedans

> Le pré-enregistrement
> [`preregistration-tetra-4b-2026-09-06.md`](preregistration-tetra-4b-2026-09-06.md)
> (sha256 `a5b065b6…`) est tamponné. Ce qui le corrige s'écrit ici.

## É1 — La qualité se mesure sur L40S, pas sur Metal, et Tetra entre dans la forme d'A4

Le §3 du préreg fait tourner `bin/ppl` et `bin/mmlu` sur `metal`. **C'est faux comme protocole de
comparaison**, et l'opérateur l'a relevé dans l'heure qui a suivi le tampon.

Les repères servis viennent de la campagne A4 du 2026-08-06 : ppl et MMLU mesurés **sur une seule
carte L40S, un seul harnais, la même empreinte de tokens**, trois bras dans la même campagne (f16
12,2369 · AWQ 13,5207 · LLVQ 16,9422 ; MMLU 70,32 · 70,04 · 55,59), pour 0,71 $ (*mesuré*,
`docs/mesures/a4-campagne-2026-08-06.txt`). Mesurer Tetra sur Metal et le lire contre ces
nombres-là mêlerait le format et le harnais — la faute même que le contrôle 0 existe pour éviter
côté encodeur.

**Ce qui est fait à la place** : Tetra est un **quatrième bras de la forme A4**. Un job L40S, un
processus, une empreinte de tokens, avec le bras LLVQ `Planes14` rejoué à côté comme témoin. Le
§4.2 du préreg (« le fichier publié rejoue 16,9415 ») devient « le fichier publié rejoue 16,9422 »,
le nombre de la carte, et il reste ce qui sépare une dérive du harnais d'une différence de format.

Conséquences chiffrées, contre le §7 du préreg :

- l'évaluation quitte le Mac : plus de 2 à 4 h de MMLU par bras, ~0,71 $ et ~40 min de carte ;
- l'image HF doit être reconstruite : celle en service date du commit `7f2fb1f`, antérieur aux
  étapes 2, 3 et 4, donc elle ne sait pas lire un fichier v5 Tetra ;
- le fichier scellé, ~1,4 Go, monte dans le bucket avant le job.

**Ce qui ne change pas** : l'encodage reste sur le Mac, comme celui du fichier publié
(4,01 h, M3 Max, *mesuré*, `docs/fiche-4b.md` §3.4) ; l'encodage est déterministe et l'objet mesuré
est l'artefact, pas la machine qui l'a écrit. Le disque, les b/param et le coût d'encodage se lisent
sur le Mac : ce sont des propriétés du fichier et de la course, pas du harnais d'évaluation.

## É2 — L'oracle a tourné sur les deux backends du Mac, pas sur la carte

Règle dure 10, `bin/oracle` avant tout chiffre : fait sur Metal et sur CPU, `max |Δhidden| = 0,000e0`
des deux côtés sur le 4B (*mesuré*, 2026-09-06). C'est l'oracle du chemin d'**encodage**. L'oracle du
chemin d'évaluation est celui de la carte, et il fait partie du job.
