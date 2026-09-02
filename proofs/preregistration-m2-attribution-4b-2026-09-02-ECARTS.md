# Écarts et corrections au pré-enregistrement M2 — écrits après le job, jamais dans le tampon

> Le pré-enregistrement
> [`preregistration-m2-attribution-4b-2026-09-02.md`](preregistration-m2-attribution-4b-2026-09-02.md)
> (sha256 `71712e60…`) est tamponné, et le journal
> [`docs/mesures/m2-attribution-4b-2026-09-02.txt`](../docs/mesures/m2-attribution-4b-2026-09-02.txt)
> est un fait brut : ni l'un ni l'autre ne s'édite. Ce qui les corrige s'écrit ici.

## É1 — Le journal ne chiffre le coût de la cible QU'EN f16, et en conclut trop

**Ce que le journal dit** (section « le coût mémoire dit que le gain n'est pas
encaissable sous v1 ») : servir une cible en f16 coûte au mieux **+0,263
b/param**, ce qui porte le 4B à 5,425 contre 5,302 pour l'AWQ, donc au-dessus.
C'est **exact**, et la table qui l'accompagne est juste.

**Ce qu'il omet, et l'omission change la conclusion.** Il ne calcule le coût
que pour f16, alors que f16 n'est pas le seul niveau concevable — c'est
seulement le seul que le chemin servi sait produire *aujourd'hui*. Or notre
format 2 bits **coûte 4,804 b/poids en VRAM** parce qu'il déplie l'index. Un
vrai 4 bits en coûte **moins**. En b/param modèle entier (*calculé*, servi
5,162 ; part des projections 90,3 % ; v_proj = 2,6 % des projections) :

| `v_proj` servi en… | b/poids | Δ b/param | total | contre AWQ 5,302 |
|---|---|---|---|---|
| Leech déplié `Planes14` (aujourd'hui) | 4,804 | +0,000 | 5,162 | ✅ |
| f16 | 16,000 | **+0,263** | 5,425 | ❌ |
| 4 bits type AWQ w4 g128 | 4,156 | **−0,015** | **5,147** | ✅ |
| 4 bits, hypothèse prudente | 4,500 | −0,007 | 5,155 | ✅ |

**Servir `v_proj` en vrai 4 bits coûterait MOINS de VRAM qu'aujourd'hui.** Ce
n'est pas un paradoxe, c'est la thèse du papier retournée contre nous : le
dépliage fait payer 4,80 bits par poids pour 2,00 bits d'information, donc un
format honnête à 4 bits est plus compact que notre 2 bits en mémoire.

**La bonne conclusion, qui remplace celle du journal** : Q5 n'est pas « non
rentable ». Q5 est **potentiellement gratuit sur l'axe mémoire, et positif sur
l'axe qualité** — et ce qui manque n'est ni un budget ni une cible, c'est un
**chemin de précision mixte** dans le noyau fusé et dans le format d'archive.
La conclusion « chantier de format » du journal tient donc ; ce qui tombe est
le mot « non rentable », qui la faisait lire comme un abandon.

⚠️ **Trois choses que ce calcul ne dit pas, et elles sont lourdes.**

1. **Le +4,48 pp est mesuré à f16, pas à 4 bits.** À 4 bits le gain serait
   moindre. De combien : **non mesuré**. Le seul repère du dossier est que
   l'AWQ 4 bits perd 0,28 pp sur le f16 *sur le modèle entier* au 4B, ce qui
   suggère qu'une matrice seule en garderait l'essentiel — mais c'est une
   analogie, pas une mesure, et elle porte sur un autre objet.
2. **Le coût DISQUE augmente** : 2,0702 → 2,1333 b/poids effectifs (+0,063).
   Sur l'axe où nous gagnons contre tout le monde, c'est une concession.
3. **Le chemin n'existe pas.** Il faut un second format dans le noyau fusé, et
   l'archive doit le porter. Ce n'est pas de la recherche, c'est du travail.

## É2 — L'expérience qui tranche É1 est bon marché, et elle n'est pas lancée

`LLVQ_RESTORE_F16` restaure depuis le checkpoint **en f16**. Le même mécanisme
avec une quantification scalaire 4 bits au chargement — exactement ce que
`LLVQ_EMBED=q8` fait déjà pour l'embedding — donnerait le bras manquant :
**`v_proj` à 4 bits, tout le reste tel que livré, MMLU apparié**. Un bras, ~7
min de L40S, **≈ 0,20 $** (*estimé* au tarif mesuré de M2), sur les 2,83 $ qui
restent de la vague.

Il tranche la réserve n° 1 ci-dessus, qui est la seule qui décide. Il demande
son propre pré-enregistrement — le tampon du 2026-09-02 ne couvre pas un bras
qui n'y figure pas — et un go d'opérateur.

## É3 — Ce que le §4 avait posé d'avance et qui s'est vérifié

La somme des sept gains marginaux vaut **23,85 pp** pour un déficit de
**14,73 pp**. Le §4 interdisait d'avance de lire ces sept nombres comme une
décomposition du déficit ; la mesure montre qu'il avait raison, et de 62 %.
