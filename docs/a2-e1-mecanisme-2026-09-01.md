# A2 étape 1 — le mécanisme de la régression, et les trois issues (2026-09-01)

> Suite de la BRANCHE STOP du préreg `ad77df46…` : r = prealloc/cat =
> **0,8919 [0,8884–0,8953]** sur CUDA (job `6a968b14…`, 0,09 $,
> [`mesures/a2-e1-prealloc-ab-2026-09-01.txt`](mesures/a2-e1-prealloc-ab-2026-09-01.txt)),
> sous le seuil pré-posé de 0,97. Le prior (r ≈ 1,00) est réfuté. Ce document
> instruit le POURQUOI — deux faits à 0 $ — puis les issues. La décision est
> à l'opérateur, comme la branche l'exige.

## Le mécanisme, établi par deux faits à 0 $

**1. Fait de layout (mesuré, sonde candle CPU)** : la vue
`narrow(2, 0, len)` d'un buffer `[b, 8, W, hd]` est **non contiguë à TOUT
`len < W`** — les strides de tête restent ceux de la fenêtre pleine — et ne
redevient contiguë qu'à `len == W`. Le bras prealloc de l'étape 1 a donc
nourri l'attention en tenseurs striés à **chaque** pas de décodage, là où le
bras `cat` rend des tenseurs contigus par construction.

**2. Fait inter-backend (mesuré, sonde grossière Metal, 0 $)** : `bin/run`
sur le 4B scellé, M3 Max, 64 tokens par prompt, 2 invocations par bras — en
régime stationnaire **cat ~9,4–9,6 tok/s contre prealloc ~6,8 (−29 %)**, et
le meilleur prompt prealloc (6,8) reste sous le pire prompt cat (7,4).
⚠️ Sonde par-prompt, PAS le protocole fusedrun (pas de rounds appariés, et
la première invocation prealloc porte un échauffement de pipeline visible :
3,1→4,1 avant 6,8 stable). Ce qu'elle établit est la DIRECTION et
l'inter-backend, pas une grandeur publiable.

**Le mécanisme** : `repeat_kv` copie l'histoire ×4 à chaque pas via
`Tensor::cat` — son commentaire amont dit lui-même qu'il existe pour éviter
« a potentially strided copy ». En entrée non contiguë, cette copie retombe
précisément dans les chemins striés, sur CUDA (−11 %) comme sur Metal
(−29 %, qui les paie plus cher). Le poste retiré (la copie ×1 du `cat` de
stockage) était plus petit que le poste dégradé.

## Les trois issues

| | ce que c'est | ce que ça achète | coût | verdict attendu |
|---|---|---|---|---|
| **1. Contrôle** : `narrow().contiguous()` | re-matérialiser la vue à chaque pas | la parité probable (r ≈ 1) — mais une ALLOCATION par pas et des formes qui grandissent : **ne sert pas la capture**, l'objet même d'A2 | ~0,5 j + 0,1 $ | contre-preuve seulement ; le mécanisme est déjà établi par les deux faits ci-dessus |
| **2. Forme fixe complète** | attention sur toute la fenêtre W, masque −inf au-delà de `len` ; à `len == W` permanent, la vue est CONTIGUË par le fait de layout n°1 — le problème disparaît par construction | des formes constantes = l'étape 2 (capture) devient possible ; c'est l'objet que les DEUX bras de l'A/B graph porteront de toute façon | ~0,5-1 j + 0,1 $ | le travail d'attention devient constant (~2× la moyenne du protocole 128 tokens) : la base coûte contre `cat`, et ce coût entre dans l'arithmétique du kill |
| **3. Stocker DÉJÀ-ÉTENDU + forme fixe** ⭐ | fenêtre à 32 têtes au lieu de 8 (×4 = **151 Mo** à W=256, trivial) — `repeat_kv` DISPARAÎT : plus AUCUNE copie d'histoire par pas, dans un chemin qui la paie ×4 aujourd'hui, `cat` compris | tout ce que 2 achète, PLUS la suppression du plus gros poste par-pas des deux bras ; la base fixe peut battre `cat` | ~1-1,5 j + 0,1 $ | c'est la seule issue qui peut rendre r ≥ 1 ; ⚠️ la réserve du doc de `KvCache` (« cacher l'étendu = ×4 octets pour rien ») valait pour un cache NON borné — à fenêtre bornée, 151 Mo contre la mort de la copie ×4 |

**Recommandation : l'issue 3**, mesurée par le même gabarit `LLVQ_KV_AB`
(mêmes rounds appariés, même gate de tokens), lecture identique à celle du
préreg d'étape 1 (r ≥ 0,97 porte l'étape 2 ; sinon retour). Si sa base bat
`cat`, l'étape 2 part d'un chemin déjà meilleur — et le gain du graph s'y
ajoutera au lieu de compenser une régression.

## Ce que ça change à l'arithmétique d'A2 (à garder sous les yeux)

Le net d'A2 au sens du kill de phase = **gain du graph − coût de sa base**.
Avec l'issue 3, le second terme peut être négatif (une base plus rapide) ;
avec l'issue 2 seule, il est positif et le graph doit d'abord le rembourser.
Le plafond estimé du graph (~7 %, cadrage §5) ne survivrait pas à une base
à −11 % — c'est exactement pourquoi la branche STOP existait.
