# Pré-enregistrement — M0 : l'axe ligne du chantier 14 est-il une constante déguisée ?

**Écrit, commité et TAMPONNÉ le 2026-09-07, AVANT la première exécution.**
Go de l'opérateur du 2026-09-07. Première porte du chantier 14
([`docs/ROADMAP-QUALITY.md`](../docs/ROADMAP-QUALITY.md) ligne 14).

Ce fichier ne s'édite plus. Ce qui le corrige va dans un `-ECARTS.md`.

**Coût : 0,00 $.** CPU du Mac, aucune carte, aucun téléchargement, aucun
checkpoint, aucune hessienne. Une demi-journée. Total projet : 119,76 $.

---

## §1 — Le fait qui ouvre le chantier, vérifié dans le code

L'encodeur choisit le niveau de gain sur la **norme** du bloc :

```rust
// llvq-quant/src/quantizer.rs:558-559, et :777 pour Tetra
let g = norm / self.row_scale;
let level = nearest_level_index(&self.centroids, g);
```

Le papier le choisit par moindres carrés contre la direction retenue
(`docs/llvq-paper-notes.md:195`) : `β*_i = q(w_i)ᵀ w_i / (q(w_i)ᵀ q(w_i))`,
c'est-à-dire `‖x‖·cos θ` et non `‖x‖`. **Chaque bloc servi est trop long du
facteur `1/cos θ`, systématiquement, sur les 151 388 160 blocs du 4B.**

Et les échelles de ligne ne corrigent pas ce biais : `llvq-quant/src/gptq.rs:245`
les fixe **avant** la boucle de blocs, à la RMS de la ligne originale — « fixed
**before** the block loop », dit le commentaire. Elles ne sont ajustées contre
rien.

## §2 — La question, et pourquoi elle vaut une demi-journée

Le chantier 14 propose d'apprendre 1 105 920 échelles de ligne par descente de
gradient, en deux à trois semaines. **Si la correction optimale est la même
constante pour toutes les lignes, ces 1 105 920 paramètres sont un seul nombre**,
et le chantier se réduit à une multiplication.

Ce banc mesure trois choses sur des blocs simulés, et rien d'autre.

## §3 — Le protocole

Un exemple `llvq-bench`, CPU seul, sur **200 000 blocs** tirés par
`llvq_bench::gauss_block` à la graine **`0x0f1b_2026_0907`**, encodés par
l'encodeur Tetra **de production** (`llvq_search::tetra::encoder`).

Les blocs sont groupés en lignes simulées de **106** et de **405** blocs, les
deux tailles du 4B (`d_in = 2560` et `9728`, sur des blocs de 24). L'échelle de
ligne est calculée comme en production (`llvq_quant::quantizer::row_scale`), et
les centroïdes de gain sont ajustés **une fois pour tous les blocs**, comme
`fit_gain_centroids` le fait par matrice et non par ligne.

Notations, pour le bloc `p` d'une ligne d'échelle `s`, de point retenu `y_p` et
de direction unitaire `û_p = y_p/‖y_p‖` :

```
  a_p = ‖x_p‖ / s                          ce que l'encodeur quantifie aujourd'hui
  τ_p = ⟨x_p, û_p⟩ / s                     ce que le papier quantifierait
  c_p = centroïde retenu pour le bloc p
  reconstruction servie = c_p · s · û_p
```

**Ce qui est publié, quatre nombres :**

1. `ρ* = Σ c_p τ_p / Σ c_p²` — le rétrécissement global optimal des échelles.
2. `R = (J(1) − J(ρ*)) / J(1)` — la réduction relative de l'objectif, avec
   `J(t) = Σ ‖x_p − t·c_p·s·û_p‖²`.
3. `σ(ρ*)` — l'écart-type de `ρ*` d'une ligne à l'autre, aux deux tailles.
4. **La variante du papier** : les mêmes trois nombres quand le niveau est
   choisi sur `τ_p` au lieu de `a_p`, centroïdes réajustés sur `τ/s`.

## §4 — Ce que ce banc ne peut pas être

Une mesure de qualité. Il ne produit aucun artefact, ne touche aucun modèle et
ne rend aucun point de MMLU. **Il ne décide donc rien sur un critère
fondamental** (règle d'opérateur du 2026-09-05) : il dit s'il faut écrire M1,
qui lui mesurera.

Sa limite : les blocs sont gaussiens, donc les lignes simulées sont homogènes,
et `σ(ρ*)` y est plus petit que sur des poids réels. Ce que le dossier oppose à
cette limite est une mesure : sur 20 000 blocs réels du Qwen3-0.6B capturés dans
la boucle GPTQ servie, l'encodeur donne un ratio de **1,00** au cas gaussien et
les blocs tournés ont une **kurtosis de 3,01**
(`docs/mesures/f1-encodeur-blocs-reels-2026-09-05.txt`). La rotation rend les
blocs gaussiens ; c'est ce qui rend la simulation admissible, et c'est aussi ce
qui la borne.

## §5 — La règle de décision, posée AVANT de voir les chiffres

| | conséquence |
|---|---|
| **`ρ* ∈ [0,95 ; 0,97]` ET `R ≤ 3 %` ET `σ(ρ*) ≤ 0,15·(1−ρ*)`** | **L'axe ligne est une constante déguisée en 1 105 920 paramètres.** M2 et la boucle de gradient sont abandonnés. Il reste M1, un seul bras, et l'axe colonne. |
| **`σ(ρ*) > 0,4·(1−ρ*)`** | **Il y a une structure par ligne réelle.** M2, la forme close par ligne, vaut son prix et passe avant l'axe colonne. |
| **`0,15·(1−ρ*) < σ(ρ*) ≤ 0,4·(1−ρ*)`** | Zone grise. M1 se fait quand même, et son résultat décide de M2 : sans effet mesurable de la constante, la structure ne rattrapera rien. |
| **autrement** — `ρ* > 0,99` ou `ρ* < 0,90`, ou `R > 8 %` | **Rien n'est décidé et le banc est suspect.** `ρ*` près de 1 dirait qu'il n'y a pas de biais radial, ce que le §1 rend improbable ; `ρ*` sous 0,90 ou `R` au-delà de 8 % dirait que l'objectif servi est bien pire que la rétention mesurée de 89,55 % ne le laisse croire, et il faudrait comprendre avant de publier. |

Le quatrième nombre, la variante du papier, **ne porte aucune porte**. Il est
publié pour ce qu'il est : le chiffrage d'un correctif gratuit, sans bit et sans
champ de format, que personne n'a jamais évalué.

## §6 — Divulgation datée, et la prédiction signée

Connu : rétention Tetra 88,89 % contre le témoin boule-12 à 92,00 · excès de
perplexité ×1,320 · 0,2779 nats · barre 0,43 pp à fichier constant et 2,92 pp
au réencodage · les échelles de ligne pèsent 1 105 920 scalaires, les centroïdes
504, la queue 16 957 440, les normes 196 096.

**Prédiction de l'auteur, opposable : `ρ*` entre 0,95 et 0,97, `R` entre 1 et
3 %, `σ(ρ*)` sous 0,15·(1−ρ*) aux deux tailles. Donc la PREMIÈRE ligne du §5 :
l'axe ligne est mort.**

Motif : `ρ*/1 ≈ 1 − δ̄` avec `δ̄ = 1 − cos θ` moyen, et `δ̄ ≤ 0,0418` se dérive de
la rétention de 89,55 % à 2,000 b/dim. L'écart-type inter-lignes vaut
`sd(δ)/√n_blocs`, donc 0,4 % à 106 blocs et 0,2 % à 405 : deux ordres de
grandeur sous la constante elle-même.

Quinze prédictions signées sur ce dossier, sept fausses.

**Ce qui la rendrait fausse de façon instructive : `σ(ρ*) > 0,4·(1−ρ*)`.** Cela
dirait que le biais radial dépend de la ligne — donc de sa distribution de
normes — et non seulement de la géométrie du réseau. L'axe ligne redeviendrait
un vrai chantier, et la moitié de la roadmap qualité changerait de tête.
