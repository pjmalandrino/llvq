# La rétraction annulait le code de gain — 2026-07-31

> 🗓️ **BANDEAU D'ÉTAT — dernière revue le 2026-08-08. Le diagnostic de ce
> document tient intégralement ; sa question de conception ouverte a été
> tranchée, et par la négative.**
>
> Le **design C** — rétraction libre + résolution close des échelles en fin de
> couche, la sortie que ce document recommandait et que
> [`pistes-facteurs-cles-2026-08-05.md`](pistes-facteurs-cles-2026-08-05.md)
> avait promu **suspect n°1 du déficit MMLU** — a été implémenté, revu
> adversarialement point par point, puis **mesuré à pleine profondeur : ×1,99
> de perplexité** (0,6B, 28 blocs, une seule variable, 71,4249 contre 35,9806
> au chemin publié). Le gate automatique a bloqué le run 4B de 4 h qui devait
> suivre. Source :
> [`verdicts-nuit-2026-08-07.md`](verdicts-nuit-2026-08-07.md) §M3, journal
> [`mesures/m3-gate-design-c-2026-08-07.txt`](../mesures/m3-gate-design-c-2026-08-07.txt).
>
> **Réserve honnête, et elle compte** : c'est *notre lecture* du design C qui
> est réfutée. Le papier n'en donne pas le pseudo-code, et la fidélité au
> présent document a été vérifiée en revue avant le run.
>
> **Ce que ce rouge apprend, et qui dépasse le design C** : c'est la **deuxième
> occurrence** du motif « proxy local strictement meilleur, composition à
> 28 couches désastreuse » — après `group_scales` (21,24 → 21,17 sur 3 blocs,
> 44,66 → 53,60 sur 28). **La rigidité de norme de la rétraction sphérique est
> porteuse à profondeur.** Ce n'est plus une anecdote, c'est un fait de méthode
> du pipeline, et il condamne d'avance toute variante qui rend la magnitude
> libre couche par couche.

> Trouvé en préparant l'écriture de l'artefact 2 bits. Deux défauts de
> comptabilité, corrigés ; une question de conception ouverte, **non tranchée**
> *(à la date de rédaction — tranchée le 2026-08-07, cf. bandeau)*.
> Branche `g6-artefact`, rien n'est mergé.

## Résumé

Le chiffre publié — **14,9104 de perplexité à 2,1117 bits/poids** sur Qwen3-4B
— a une perplexité juste et un **débit faux**. Le modèle évalué stocke une
magnitude flottante libre par bloc de 24 poids, soit 16 bits que la
comptabilité ne facture pas. Le débit réel est **~2,75 bits/poids**, et la
compression réelle **×3,96** au lieu de ×4,63.

Les deux défauts sont corrigés et testés. Mais le correctif a une conséquence
sur la sémantique du Spherical GPTQ qui demande une décision et un A/B.

## Défaut 1 — la rétraction annule le gain

`LeechShapeGain::quantize` choisit un niveau de gain et rend un bloc de norme
`picked`. Puis `quantize_layer` applique la rétraction sphérique (Eq. 17) :

```rust
let n = qbuf[..b].iter().map(|a| a * a).sum::<f64>().sqrt();  // = picked
let k = norm_before / n;
for q in qbuf[..b].iter_mut() { *q *= k; }                     // norme = norm_before
```

La norme finale est `norm_before` — la norme du bloc avant quantification, un
flottant libre. **Le niveau choisi disparaît intégralement.**

Mesuré (`g5_retraction.rs`) : deux codebooks de gain disjoints, `[0,25 / 0,75]`
et `[6 / 19]`, produisent des poids **bit-identiques** — `max |Δ| = 0e0`. Et
après une passe de couche, 24 blocs prennent 24 magnitudes distinctes alors que
le code n'en a que 2.

Le contrôle sans rétraction passe : ce n'est pas le harnais, c'est la
composition.

**Pourquoi aucun test ne l'attrapait.** `the_gain_is_actually_quantized`
appelle `quantize()` **directement**. Il n'exerce jamais la rétraction que
`quantize_layer` applique juste après, et tous les runs utilisent
`retract: true`. C'est le motif documenté quatre fois au §5 du `CLAUDE.md` —
une assertion qui n'exerce pas le paramètre qu'elle est censée couvrir — et
l'ironie est que ce test-là avait été écrit pour attraper la quatrième
occurrence.

## Défaut 2 — le `row_scale` dérivait

`row_scale`, la référence à laquelle le code de gain rapporte chaque bloc,
était recalculé **à chaque bloc** sur l'état courant de la ligne, qui bouge à
mesure que l'erreur se propage. Mesuré sur une même ligne : 0,04737 au bloc 0,
0,04837 au bloc 1.

Or la comptabilité facture **une f16 par ligne** (`Report::rows * 16`), et le
commentaire du trait énonce l'intention : *« That scale is one float per
row »*. Une ligne de 2560 aurait eu besoin de 106 échelles distinctes.

## Ce que ça coûte

Recalculé sur les vraies dimensions de Qwen3-4B (36 blocs, 2560 × 9728, 7
projections par bloc, embedding lié de 151 936 × 2560 en f16) :

| configuration | bits/bloc | bits/poids | artefact | vs FP16 8,04 Go |
|---|---|---|---|---|
| cap 13 + 1 bit de gain, **annoncé** | 49 | 2,1117 | 1,737 Go | ×4,63 |
| cap 13, **réel** (48 index + 16 f16) | 64 | **2,7338** | **2,019 Go** | **×3,98** |
| cap 12 + 1 bit, annoncé (run en cours) | 48 | 2,0702 | 1,718 Go | ×4,68 |
| cap 12, **réel** | 63 | **2,6923** | **2,001 Go** | **×4,02** |

> Le modèle de comptabilité ci-dessus reproduit **exactement** les chiffres
> publiés — 2,1117 b/w, 1,737 Go, ×4,63 — et retrouve aussi le 2,7289 de la
> ligne « magnitude libre » du `CLAUDE.md` au millième près, l'écart de 0,0049
> étant précisément les échelles de ligne que `Codebook::Direction` ne stocke
> pas. Les chiffres « réel » ne sont donc pas une estimation.

**La perplexité reste valide.** 14,9104 est une mesure, pas un calcul. C'est
son étiquette de débit qui est fausse.

Et le point qui fait mal : **à 2,73 bits/poids, la comparaison au papier
change de nature.** Leur meilleure configuration sans fine-tuning tient
15,54 à 2,000 bits. On tient 14,91, mais à 37 % de bits en plus, pas 5,6 %.

## Le run `leech1c12` de la nuit, et ce qu'il dit

Terminé le 2026-07-31 à 04h26, 12 715 s pour 252 matrices et 3 633 315 840
poids. Baseline **12,2336** — identique au run publié, donc le harnais n'a pas
bougé et la comparaison est valide.

| | bits annoncés | bits **réels** | wiki | × |
|---|---|---|---|---|
| publié (`leech1`, cap 13) | 2,1117 | 2,7338 | 14,9104 | ×1,219 |
| cette nuit (`leech1c12`, cap 12) | 2,0702 | **2,6923** | **15,3272** | ×1,253 |

**Le run était fondé sur une prémisse fausse.** Son raisonnement : plafonner à
`Λ₂₄(12)` fait tomber l'index de 48 à 47 bits, *« ce qui paie le bit de
gain »*. Mais le bit de gain ne servait à rien — il était annulé par la
rétraction. Le cap n'a donc rien payé : il a réduit le codebook de direction
(moins de points, donc plus de distorsion angulaire) en échange d'un bit sur
**64** au lieu d'un bit sur 48.

Le résultat est cohérent avec ça : **0,04 bit/poids d'économie pour 2,8 % de
perplexité en plus**, et une dégradation dans le sens attendu quand on rétrécit
un codebook.

⚠️ « Cohérent avec » n'est pas « démontré par ». 2,8 % est du même ordre que
l'écart inexpliqué de la section suivante, et tant que la variance du pipeline
est inconnue, cet écart pourrait aussi être du bruit.

L'artefact écrit fait **6,8 Go** — des reconstructions f16, comme prévu. Ce
n'est pas le fichier compressé.

## ⚠️ Une conséquence sur la lecture de l'historique

Si la rétraction annulait le gain, alors `Codebook::Direction` et
`Codebook::ShapeGain` choisissaient le même point du réseau **et** finissaient
sur la même norme : un seul quantifieur sous deux noms.

Mesuré (`under_the_old_retraction_shape_gain_was_direction_only`) : sur une
couche entière, écart relatif maximal **7,1 × 10⁻¹⁵**, mêmes zéros, centroïdes
sains. Ce sont bien les mêmes poids.

Donc les deux premières lignes du tableau 4B comparent **deux configurations
identiques** :

| | bits/poids | wiki |
|---|---|---|
| wikitext, magnitude libre | 2,7289 | 14,2684 |
| wikitext, 1 bit de gain | 2,1117 | 15,2909 |

L'écart de **7 %** ne peut pas venir du codebook. Deux suspects :

1. **Une divergence numérique amplifiée.** La calibration est séquentielle —
   36 blocs, chacun quantifié contre les activations sorties des blocs déjà
   quantifiés. `nearest_angular` est un argmax : deux candidats à 10⁻¹⁵ l'un
   de l'autre suffisent à faire basculer le point choisi, et l'écart n'est
   plus microscopique. Sur 3 blocs l'écart mesuré était de 0,04 % ; sur 36, de
   7 %.
2. Une différence de configuration non consignée entre les deux runs.

**Ce que ça invalide.** La conclusion « quantifier le gain ne coûte presque
rien — 0,04 % de perplexité pour 0,52 bit/poids » comparait deux
configurations identiques. Elle n'a jamais été mesurée. (Elle reste peut-être
vraie ; simplement, rien ne l'établit.)

**Ce que ça suggère de faire, et c'est bon marché.** Un **run de contrôle de
variance** : lancer deux fois la même configuration sur 36 blocs, ou lancer
`direction` contre `leech1f` (qui sont le même quantifieur), et regarder
l'écart de perplexité. Tant qu'on ne connaît pas cette dispersion, on ne sait
pas si un écart de 5 % entre deux configurations est un signal ou du bruit —
et plusieurs conclusions du projet reposent sur des écarts de cet ordre.

## Les correctifs

| # | correctif | où |
|---|---|---|
| 1 | `BlockQuantizer::retraction_target()` — le quantifieur nomme la sphère ; par défaut la norme d'entrée, pour `LeechShapeGain` le **niveau de code le plus proche** | `quantizer.rs`, `gptq.rs` |
| 2 | `row_scale` calculé une fois par ligne, avant la boucle des blocs | `gptq.rs` |

État : **70 tests verts, zéro warning clippy.** Les 16 tests GPTQ existants
passent inchangés, dont `parallel_matches_serial_exactly` et
`correction_is_the_analytic_minimizer`.

Cinq tests nouveaux dans `llvq-quant/tests/g5_retraction.rs`, dont celui qui
manquait vraiment :

> `the_claimed_rate_matches_what_the_reconstruction_needs` — le nombre de
> magnitudes distinctes que la reconstruction produit doit tenir dans les bits
> que le quantifieur facture. Une assertion de **coût**, pas de qualité.

## ⚠️ La question ouverte, à trancher

Le correctif 1 a une conséquence : pour `LeechShapeGain`, la rétraction
devient un **no-op** — elle vise le niveau que `quantize` a déjà produit. Le
« Spherical » du Spherical GPTQ ne fait donc plus rien de spécifique pour ce
quantifieur.

Ça peut être sans conséquence, ou coûter cher : la Table 9 du papier montre
que le GPTQ euclidien sans rotation s'effondre à 191,90 de perplexité là où le
Spherical GPTQ tient à 6,90. **On ne sait pas** quelle part de ce gain vient
de la rétraction elle-même et quelle part de la rétroaction d'erreur.

Trois designs cohérents, dont un seul est mesuré :

| | rétraction | bits/bloc | code de gain | mesuré ? |
|---|---|---|---|---|
| **A** — `retract_to_level` (défaut) | vers le niveau du code | 47 + k | porteur | ❌ |
| **B** — `with_free_magnitude()` | vers la norme exacte | 47 + 16 | inerte | ✅ c'est ce que tous les runs ont fait |
| **C** — Algorithme 3 à la lettre | norme exacte, puis résolution close en fin de couche | ? | délégué au solve | ❌ `group_scales` dégradait |

Le papier dit que les magnitudes sont tenues « par la contrainte de norme
pendant GPTQ **puis par une résolution close en fin de couche** » — c'est
`refine_group_scales`, qu'on désactive parce qu'il dégradait la perplexité sur
28 blocs. On a donc la rétraction **sans** le mécanisme censé re-quantifier les
magnitudes derrière. C est peut-être la vraie lecture du papier, et mériterait
qu'on relise l'annexe I avant de choisir.

**Le choix est une mesure, pas une préférence.** Les deux comportements sont
accessibles pour être A/B-és.

## L'A/B à lancer, et pourquoi il ne choisit pas

A/B sur 3 blocs de Qwen3-0.6B, une seule variable, ~8 min chacun :

```bash
LLVQ_MODEL=Qwen/Qwen3-0.6B LLVQ_CALIB=c4 cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- 64 2048 12 2048 metal nogs leech1c12 3 rot
```

```bash
LLVQ_MODEL=Qwen/Qwen3-0.6B LLVQ_CALIB=c4 cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- 64 2048 12 2048 metal nogs leech1c12f 3 rot
```

Le suffixe `f` de `leech1c12f` restaure le comportement B, et **le facture
correctement** (16 bits au lieu de 1) — les deux lignes de débit sont donc
directement comparables.

⚠️ Cet A/B ne sert **pas** à choisir entre A et B. B n'est pas une option : il
coûte 0,64 bit/poids de plus que ce qu'il annonce. L'A/B sert à mesurer **ce
que coûte l'honnêteté** en perplexité. Si A dégrade peu, on publie A. Si A
dégrade beaucoup, la vraie réponse est C, et il faut relire l'annexe I.

## Ce qu'il faut corriger dans la communication

Trois endroits portent le chiffre faux, et le brouillon de mail demande
lui-même qu'aucun chiffre ne diverge du README :

- `README.md` — le tableau de résultat et la section « Read this before
  quoting the number »
- `CLAUDE.md` — § Qwen3-4B, et la ligne « Compression réelle sur le 4B »
- `docs/archive/mail-qualcomm-draft.md`

À ne pas faire avant l'A/B : le chiffre juste dépend de quelle option on
retient.

## La leçon, qui est la même que les quatre précédentes

> Tant qu'on simule, on peut se raconter ce qu'on veut sur ce que ça coûterait ;
> dès qu'il faut écrire des octets, il faut les compter.

C'est déjà écrit dans le `CLAUDE.md`, à propos de l'erreur de comptabilité
précédente (2,0653 annoncé pour 2,7289 réel). Le même paragraphe s'applique
mot pour mot à celle-ci — et elle a été trouvée exactement de la même façon,
en préparant l'écriture de l'artefact.

La nuance neuve : le test écrit pour empêcher la récidive **testait le
quantifieur nu**, pas le chemin composé. Une assertion sur un composant ne dit
rien du pipeline qui l'utilise.
