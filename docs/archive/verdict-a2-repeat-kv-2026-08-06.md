# A2 — la recopie du cache KV : bloquée par candle, et chiffrée

> Verdict du lot A, étape A2 ([`spec-lot-a-2026-08-05.md`](spec-lot-a-2026-08-05.md)).
> **Aucun code modifié, aucun job lancé** — la question se tranche par lecture
> de candle. La spec prévoyait ce cas : « si candle l'impose structurellement,
> le documenter et passer ».

## Le constat de départ

`model.rs::forward_cached` fait, une fois par bloc et par token :

```rust
let k = candle_transformers::utils::repeat_kv(k, groups)?.contiguous()?;
let v = candle_transformers::utils::repeat_kv(v, groups)?.contiguous()?;
```

Qwen3-4B a 8 têtes KV pour 32 têtes de requête, donc `groups = 4` : le cache
est **matérialisé quatre fois**, pour K et pour V, dans chacun des 36 blocs.

## Pourquoi on ne peut pas l'éviter dans candle 0.9

Deux chemins, tous deux fermés :

**1. `broadcast_matmul` matérialise aussi.** L'idée naturelle est de voir `q`
comme `[b, n_kv, groups, t, d]` et `k` comme `[b, n_kv, 1, s, d]`, puis de
laisser le produit diffuser sur l'axe `groups`. Mais `Tensor::broadcast_matmul`
(`candle-core/src/tensor.rs:1545`) fait exactement ce qu'on cherchait à
éviter :

```rust
// TODO: Avoid concretising the broadcasted matrixes via contiguous.
(false, true) => lhs.matmul(&rhs.broadcast_as(&r_shape)?.contiguous()?),
```

Le `TODO` est de candle, pas de nous. Tant qu'il est là, diffuser coûte la
même copie que répéter.

**2. `repeat_kv` est déjà la forme rapide.** Son commentaire cite une mesure
des auteurs (PR huggingface/candle#2043) : *« Using cat is faster than a
broadcast as it avoids going through a potentially strided copy »*. La forme
actuelle n'est donc pas une maladresse — c'est celle que candle recommande
après l'avoir mesurée.

**3. Boucler sur les têtes KV serait pire.** Faire 8 produits séparés
éviterait la copie mais ajouterait 8 lancements par bloc, soit **288 par
token**, sur un décode dont l'attribution du 2026-08-05 montre que la moitié
du budget est déjà de la latence de lancement.

## Ce que ça coûte réellement — et pourquoi ça ne se voit pas ici

Le volume recopié par token vaut `36 × 2 × groups × n_kv × t × d × 2` octets,
c'est-à-dire qu'il croît **linéairement avec le contexte** :

| contexte | recopié par token | à 662 Go/s |
|---|---|---|
| 70 tokens (le banc `fusedrun`) | ~41 Mo | **~0,06 ms** |
| 1 024 | ~600 Mo | ~0,9 ms |
| 4 096 | ~2,4 Go | **~3,6 ms** |

Sur le banc actuel — prompt de 5 tokens, 32 générés — le poste est de l'ordre
de 0,06 ms sur 21,9, soit **0,3 %**. Il est invisible, et c'est pourquoi la
plomberie mesurée aujourd'hui ne l'a jamais fait apparaître.

À 4 096 de contexte il vaudrait ~3,6 ms par token, c'est-à-dire **plus que
tout ce que la plomberie a récupéré aujourd'hui (2,05 ms)**.

## Verdict

Fermé pour candle 0.9, documenté, **et à rouvrir dès qu'un chiffre à contexte
long est visé** — ce que ni `fusedrun` ni le protocole miniature ne mesurent
aujourd'hui. Deux façons de le rouvrir :

* que candle lève son `TODO` (rien à faire de notre côté que suivre) ;
* écrire notre propre noyau d'attention, ce qui sort très largement du
  périmètre du lot A et rejoint le chantier « une opération au lieu de mille »
  de l'attribution.

⚠️ Le tableau ci-dessus est **arithmétique, pas mesuré** : il multiplie des
tailles de tenseurs par une bande passante mesurée ailleurs. Il donne un ordre
de grandeur et sert à décider s'il vaut la peine d'instrumenter, pas à publier.
