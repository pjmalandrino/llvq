# Passation — lot A refermé (2026-08-06)

> Le lot A de [`spec-lot-a-2026-08-05.md`](spec-lot-a-2026-08-05.md) est
> terminé, ses quatre étapes rendues. **Coût total : 2,19 $** sur douze jobs,
> contre 5,5 $ estimés. Branche `noyau-cuda`, mergée sur `main`.

## Ce que le lot a produit, en une table

| | avant le lot | après |
|---|---|---|
| le noyau fusé | jamais appelé en inférence | **tourne dans le modèle** |
| mémoire carte | 8,04 Go | **3,28 Go** (÷2,45) |
| débit | 42,7 tok/s | **47,0** |
| chargement | 209 s | **128 s** |
| texte produit | référence | identique jusqu'au token 89/128 |

## Les quatre étapes

**A1 — premier passage sur carte. Vert sur les cinq critères.**
L'oracle rend `max |Δhidden| = 0.000e0`, la rotation rend sur carte
exactement les chiffres du harnais Mac, la divergence de tokens est tardive
(89 sur 128, un tie-break), les octets device sont rapportés, et le débit fusé
dépasse le dense. Journal :
[`a1-lot-a-2026-08-06.txt`](mesures/a1-lot-a-2026-08-06.txt).

**A2 — la recopie du cache KV. Fermée par candle, tranchée sans job.**
`broadcast_matmul` matérialise aussi (le `TODO: Avoid concretising` est de
candle), `repeat_kv` est déjà la forme que ses auteurs ont mesurée comme la
plus rapide, et boucler sur les têtes ajouterait 288 lancements par token.
Chiffré : ~0,06 ms à 70 tokens de contexte (0,3 %, invisible), ~3,6 ms à 4096.
**À rouvrir dès qu'un chiffre à contexte long est visé.**
[`verdict-a2-repeat-kv-2026-08-06.md`](verdict-a2-repeat-kv-2026-08-06.md).

**A3 — CUDA Graph. Mesuré, négatif, A3(b) annulé.**
`g = 3,63 µs`, confirmé par trois instruments indépendants ; ε = 0,915 ms,
15,8 % du bras LLVQ. Le graph n'en récupère que 18 %, soit **0,8 % d'un
token**, et c'est un plafond. La spec conditionnait A3(b) à un (a) concluant :
il ne l'est pas.
[`a3-graph-2026-08-06.txt`](mesures/a3-graph-2026-08-06.txt).

**A4 — la campagne à quatre bras. Le verdict du projet.**
[`a4-campagne-2026-08-06.txt`](mesures/a4-campagne-2026-08-06.txt).

| bras | b/poids | ppl | MMLU micro |
|---|---|---|---|
| f16 | 16,000 | 12,2369 | 70,32 % ± 1,28 |
| AWQ 4 bits (Qwen) | 4,156 | 13,5207 | 70,04 % ± 1,25 |
| LLVQ 2 bits | 2,170 | 16,9422 | **55,59 % ± 1,35** |

## Le résultat, sans détour

**Le harnais est certifié** : la baseline rend 70,32 contre 70,42 exigés,
0,08 σ d'écart, et la perplexité retombe sur le chiffre publié. Les deux
autres bras sont donc lisibles.

**Le 4 bits ne perd rien.** −0,28 pp de MMLU, sous l'erreur
d'échantillonnage. Il est indiscernable du f16 sur cet axe.

**Nous perdons 14,73 points.** Mesuré côte à côte, même empreinte de tokens,
et cohérent avec le −14,33 du 2026-08-02 : la dégradation se reproduit.

**Sur un Qwen3-4B, le 4 bits nous domine partout sauf le disque** — et le
disque n'a jamais été ce qui limite. `face-au-4-bits.md` posait cette
conclusion sur des chiffres hétérogènes ; elle est maintenant mesurée sur un
seul silicium.

## Ce qui survit, et ce qui ne survit pas

Survit : **le noyau**. Un décodeur de Leech fusé qui bat le f16 de 1,89× sur
les projections, tourne dans un vrai modèle, rend les mêmes tokens et divise
la mémoire par 2,45. Ça n'existe nulle part ailleurs, papier compris.

Ne survit pas : **le produit sur un 4B**.

## Deux corrections à porter dans le dossier

1. **Le build d'image prend 12-14 min, pas 40-70.** Sept builds, aucun échec.
   La spec et `ops/README.md` annoncent un chiffre faux d'un facteur 4 — et
   c'est lui qui décourageait d'écrire du code hôte instrumenté, donc lui qui
   a laissé le noyau sans instrumentation jusqu'ici.
2. **`ops/run.py estimate` ne s'applique pas aux jobs de scoring** (il modélise
   une quantification, facteur ~8). La spec du lot A demandait de l'utiliser ;
   `experience-mesure.md` l'interdit explicitement. C'est le second qui a
   raison — le plafond utile est le `timeout`.

## Le point de décision suivant

**C1 — les plans binaires** ([`prep-c1-planes-2026-08-06.md`](prep-c1-planes-2026-08-06.md),
[`pistes-format-vram-2026-08-05.md`](pistes-format-vram-2026-08-05.md)). C'est
la seule piste identifiée qui attaque le chiffre qui décide : les 5,51 b/poids
en VRAM, contre 4,50 pour le 4 bits. Tant qu'on est au-dessus, le gain de
2,45× mesuré aujourd'hui ne suffit pas à faire tenir un modèle là où le 4 bits
ne tient pas — ce qui est la seule raison d'aller à 2 bits.

Même infra de banc, ~0,1-0,2 $.
