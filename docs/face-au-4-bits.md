# Face au 4 bits — la comparaison qui manquait (2026-08-01)

> Tout ce qui précède comparait LLVQ au **FP16**. C'est la mauvaise référence :
> personne ne déploie du FP16 en local. La vraie question est *contre du 4 bits*,
> et elle n'avait jamais été posée. Voici la réponse, mesurée sur la même
> machine, le même modèle, le même jour.

## Le protocole

Qwen3-4B, M3 Max. Le 4 bits est produit **localement** depuis le checkpoint
FP16 déjà en cache — aucun téléchargement, aucun modèle tiers :

```bash
mlx_lm.convert --hf-path Qwen/Qwen3-4B -q --q-bits 4 --q-group-size 64 \
  --mlx-path qwen3-4b-mlx-q4
mlx_lm.generate --model qwen3-4b-mlx-q4 --prompt "…" --max-tokens 256 --temp 0
```

MLX plutôt que llama.cpp pour deux raisons : c'est le chemin natif Metal
d'Apple, donc l'adversaire le plus juste pour un noyau Metal ; et la conversion
GGUF aurait exigé un script Python absent de l'installation Homebrew.

## Le verdict

| | MLX 4 bits | LLVQ 2 bits | |
|---|---|---|---|
| **disque** | 2,263 Go | **1,771 Go** | **×1,28** pour nous |
| **RAM** (mesuré / calculé) | **2,39 Go** | 3,28 Go | ×1,37 **contre** nous |
| **débit** (bout en bout, mesuré) | **129,8 tok/s** | ~78,5 tok/s *(projeté)* | ×1,65 **contre** nous |
| **qualité** | ~1-2 % de dégradation | **×1,386** | franchement contre nous |
| **bits/poids effectifs** | 4,50 | 3,52 disque / **5,51 RAM** | |

**Sur un 4B, le 4 bits nous domine sur tous les axes sauf le disque, et de peu.**
Les 129,8 tok/s sont stables à 0,5 % près sur trois runs, mesurés de bout en
bout — attention, normes et cache KV compris. Nos 78,5 sont une *projection*
qui exclut tout ça : l'écart réel est au moins ×1,65, probablement pire.

## La leçon, et elle est structurelle

Le gain de place de LLVQ est **sur le disque** (3,52 b/poids). Mais le format
que le noyau rapide lit en RAM coûte **5,51 b/poids** — *plus* que les 4,50 du
4 bits. La vitesse a été achetée avec les bits mêmes qui justifiaient le 2 bits.

C'est visible en extrapolant à 70B, là où la thèse est censée vivre :

| 70B | taille | tient sur… |
|---|---|---|
| FP16 | 140 Go | rien de local |
| **MLX 4 bits** | **39,4 Go** | Mac 48 Go |
| LLVQ `Slot32` (rapide) | **48,2 Go** | ❌ *pire que le 4 bits* |
| LLVQ `Grouped32` (lent) | 29,3 Go | Mac 32 Go ✅ |
| LLVQ sur disque | 19,0 Go | — |

**Le format qui va vite ne rentre pas mieux que du 4 bits ; le format qui rentre
mieux ne va pas vite.** On n'a pas encore les deux à la fois, et c'est *le*
problème à résoudre — pas un détail d'optimisation.

## Ce que ça ne dit pas

- **Le noyau reste une contribution réelle.** Un décodeur Leech multi-coquilles
  fusé qui bat le FP16 de 2,07× n'existe nulle part ailleurs, le papier compris
  (mono-coquille, plus lent que QTIP). Ce qui est réfuté, c'est le *produit* sur
  un 4B, pas l'ingénierie.
- **La qualité n'est pas mesurée sur des tâches.** ×1,386 de perplexité contre
  ~1-2 % pour le 4 bits est un écart massif, mais aucun MMLU n'a été passé, ni
  chez nous ni sur le 4 bits de cette machine.
- **Le régime batché n'est pas testé** — et c'est celui d'un cloud.

## Les trois sorties possibles

1. **Fermer l'écart de RAM.** Quantifier le `lm_head` (389 M poids encore en
   f16, 0,778 Go) descend `Slot32` à 2,77 Go et `Grouped32` à 1,68 Go — ce
   dernier passe *sous* MLX. Le levier était identifié depuis juillet ; il
   devient prioritaire.
2. **Rendre `Grouped32` rapide.** C'est le vrai sujet : 3,35 b/poids à vitesse
   utile battrait le 4 bits sur la place *et* tiendrait la route en débit. Le
   passage `Flat32` → `Slot32` a montré qu'un changement de format bien choisi
   vaut 2,4× ; il reste peut-être une forme intermédiaire.
3. **Assumer le créneau.** Le 2 bits ne sert que là où le 4 bits **ne rentre
   pas** : 70B sur 32 Go, 405B sur 128 Go. Sur ces points-là, `Grouped32` gagne
   même lent, parce que l'alternative est *ne pas charger le modèle*.

Aucune de ces trois n'est acquise. La comparaison a coûté une heure et vaut
plus que la journée de noyau qui l'a précédée.

## La sortie n° 2, mesurée : plafonner les niveaux (`bin/lcap`)

La largeur du payload runtime est `34 + 24(L−1)` bits, où L est le nombre de
magnitudes distinctes du bloc — **zéro compris**. Donc L *est* le coût mémoire,
et le plafonner est le seul bouton qui échange directement distorsion contre
RAM. Mesuré sur source gaussienne, 20 000 blocs d'entraînement + 20 000
d'évaluation, shape–gain 1 bit, centroïdes ajustés sur le train :

| L max | codebook | index | b/dim | MSE | rétention | **RAM b/poids** |
|---|---|---|---|---|---|---|
| 5 *(actuel)* | 2,81·10¹⁴ | 48 b | 2,0417 | 0,0714 | 93,26 % | 5,667 |
| **4** | 2,61·10¹⁴ *(93 %)* | 48 b | 2,0417 | 0,0719 | **93,01 %** | **4,667** |
| **3** | 6,23·10¹³ *(22 %)* | 46 b | 1,9583 | 0,0854 | **90,65 %** | **3,667** |
| 2 | 7,53·10¹⁰ | 37 b | 1,5833 | 0,1524 | 85,71 % | 2,667 |

**`L ≤ 4` est quasi gratuit** : −0,25 point de rétention pour **−1 bit/poids**
de RAM. On jette 7 % du codebook et on ne perd presque rien — la prédiction
haute dimension se vérifie. À prendre sans discuter.

**`L ≤ 3` est un vrai arbitrage** : −2,61 points de rétention pour −2 bits/poids.
Mais il fait passer le format **sous le 4 bits** (3,667 contre 4,50), et il
baisse aussi le débit disque (1,958 b/dim contre 2,042).

Bénéfice de bord : sous plafond, **tous les blocs ont la même largeur**, donc
le stride devient fixe — plus de table de bases, plus de padding par groupe, et
un décodage plus court (2 masques au lieu de 4 à `L ≤ 3`). Le noyau devrait
être *plus* rapide, pas moins.

### Ce que ça change à 70B

| en RAM | taille | tient sur |
|---|---|---|
| FP16 | 140 Go | rien |
| `Slot32` actuel | 48,2 Go | Mac 64 Go |
| `L ≤ 4` | 40,8 Go | Mac 64 Go |
| MLX 4 bits | 39,4 Go | Mac 48 Go |
| **`L ≤ 3`** | **32,1 Go** | **Mac 48 Go** |

`L ≤ 3` reprend l'avantage mémoire perdu — 18 % sous le 4 bits — tout en
gardant la vitesse. **La question « petit ET rapide » a une réponse
affirmative.**

⚠️ Source gaussienne, une seule graine, pas de GPTQ. Les vrais poids ne sont
pas gaussiens et la boucle GPTQ déforme leur distribution : ceci **indique**,
ça ne tranche pas. L'A/B sur 3 blocs du vrai modèle tranche — et vu ce que
MMLU vient de révéler (on perd déjà plus de capacités que le papier), la
mesure à surveiller n'est pas la perplexité mais MMLU.
