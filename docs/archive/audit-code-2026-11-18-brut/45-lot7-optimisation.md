# Lot 7, optimisation hors noyaux : journal

Deux constats écrits (OPT-02, OPT-04), aucun compilé, aucun test lancé. OPT-01 reçoit un plan. Aucun gain n'est
mesuré ici : chaque gain cité est celui de `07-optimisation.md`, *estimé*.

## Corrigé

| ID | fichiers | ce qui change, en une ligne | vérification faite |
|---|---|---|---|
| OPT-02 | `llvq-llm/src/repl.rs`, `llvq-llm/tests/sampler_nucleus.rs` (neuf) | `Sampler::pick` prend le noyau par sélection partielle quand il peut prouver le même vecteur au bit, sinon par l'ancien tri complet, recopié tel quel dans `nucleus_sorted` | relu deux fois, NON COMPILÉ |
| OPT-04 | `llvq-quant/src/linalg.rs`, `llvq-quant/src/gptq.rs`, `llvq-quant/tests/gptq_row_slice.rs` (neuf) | `GptqFactor::u_row(i)` rend la ligne `i` de `U` ; la propagation lit `U[s + k][r0..]` en tranche, zippée avec `row[r0..]` | relu deux fois, NON COMPILÉ |

### OPT-02 : pourquoi le jeton tiré est le même, au bit

Constat revérifié dans le code : le tri est `sort_unstable_by` sur la seule clé du poids (`total_cmp`, décroissant). Il
n'est pas stable. Deux poids égaux sont donc rangés par l'algorithme de tri, pas par l'indice. Aucune sélection
partielle ne reproduit cet ordre-là. L'optimisation ne force donc pas l'identité dans ce cas : elle s'abstient.

`pick` construit les poids comme avant, puis appelle `nucleus_selected(&p, top_p)`, et sinon `nucleus_sorted(p, top_p)`.
`nucleus_sorted` est l'ancien code ligne pour ligne, sur le vecteur `p` dans l'ordre des indices, donc le même tri sur
la même entrée. Le tirage (masse, `next_u64`, somme cumulée) n'a pas bougé et consomme un seul pas du générateur, comme
avant.

`nucleus_selected` rend `Some` seulement si trois conditions tiennent. Chacune se démontre.

1. Les valeurs. `select_nth_unstable_by(k, desc)` puis le tri des `k + 1` premiers donnent les `k + 1` plus grandes
   valeurs, triées. Pour `total_cmp`, « égal » veut dire les mêmes bits, donc la suite des valeurs triées est unique,
   quel que soit l'ordre des égalités : elle coïncide avec le préfixe du tri complet.
2. Le total. L'ancien code somme tout le vecteur trié, de gauche à droite (`Sum` de `f32` est un `fold` par `+`, départ
   `-0.0` ou `0.0` selon la version de Rust ; dès le premier poids `>= +0` les deux départs donnent le même résultat). Si, à l'indice `j`,
   `total + w_j == total`, alors pour tout `0 <= y <= w_j` on a `total <= fl(total + y) <= fl(total + w_j) = total`,
   car l'arrondi au plus proche est monotone. Tous les poids après `j`, dans `c` comme hors de `c`, sont dans
   `[+0, w_j]`. La somme complète vaut donc la somme du préfixe. La précondition exclut NaN, infini et signe négatif
   (`-0.0` compris).
3. La coupe et l'ordre. La boucle de coupe fait les mêmes opérations (`acc += w / total`, `acc >= top_p`) sur les
   mêmes valeurs avec le même total, donc s'arrête au même `i`, à condition que `i < k`. Les poids `c[0..=i+1]` doivent
   être strictement décroissants : alors chaque poids gardé n'existe qu'une fois dans toute la ligne (les égaux sont
   contigus dans le tri), et l'élément à la position `i'` du tri complet est forcément `c[i']`, indice compris. Une
   égalité rend `None` tout de suite.

`top_p` NaN ou `>= 1`, une ligne de moins de 260 jetons, un total jamais stabilisé ou une coupe hors de `c` jusqu'à
`k < n / 4` rendent `None`. Le garde `top_p >= 1` est un garde de temps, pas de justesse : à `top_p = 1`, le noyau est
en général toute la ligne, et la sélection répétée coûterait plus que le tri.

Ce que cela veut dire en pratique, *estimé* sans mesure. Le raccourci sert sur une ligne piquée (quelques jetons loin
au-dessus du reste) à température usuelle. Il retombe sur le tri complet à haute température (le reste ne s'absorbe
plus dans le total), sur un noyau large, et sur toute égalité dans le noyau. Des logits passés en f16 donnent souvent
des égalités : sur ce chemin le gain peut être nul. Le chemin servi CUDA est glouton et n'est pas touché.

Taille : `nucleus_selected` fait 37 lignes physiques (22 instructions), au-dessus du seuil de 30 lignes du brief. Je
l'ai écrite parce que la mission la demande et que la preuve est complète. Si l'opérateur préfère l'écarter, il suffit
de remplacer l'appel par `nucleus_sorted(p, top_p)` : le comportement redevient l'ancien sans autre changement.

### OPT-04 : pourquoi l'encodage est le même, au bit

Les deux chemins de `GptqFactor::new` (`faer` sous `fast-linalg`, maison sinon) et `new_reference` remplissent le même
champ `u: Vec<f64>`, `n × n` ligne par ligne, zéro sous la diagonale (`linalg.rs`, `fast::factor` et
`reference_factor`). `u_row` est donc commune aux deux, et `u_row(i)[j]` est l'élément même que lit `u(i, j)`.

Par cellule `row[r0 + j]`, l'ancienne et la nouvelle boucle font, pour chaque `k` croissant non sauté, une
multiplication `x * U[s+k][r0+j]` puis une soustraction, chacune arrondie. Rust ne contracte pas en FMA sans
`mul_add` explicite. Les cellules sont indépendantes : l'ordre en `j`, et une vectorisation par voies, ne changent
aucun arrondi. Les longueurs sont égales (`factor.dim() == d_in` est vérifié en tête de la fonction). Aucun `unsafe`,
aucun `get_unchecked`, aucune dépendance ajoutée.

Le test existant `spherical_feedback_matches_an_independent_reference_on_tetra` (`llvq-quant/tests/g5_spherical.rs`)
garde sa propagation indexée par `u(i, j)` : il devient une seconde référence indépendante, sur Tetra.

## Laissé de côté

| ID | raison |
|---|---|
| OPT-01 | Hors mission : chemin de chargement servi, invérifiable sans carte. Plan et mesure ci-dessous. |
| OPT-03 | Hors mission : nouvel instrument MMLU. |
| OPT-05 | Hors mission. |
| OPT-07 | Hors mission : structure du graphe de crates. |
| OPT-04, seconde phase (tuile de lignes) | Changement de structure de boucle, plus de 30 lignes, à faire avec un compilateur et une mesure. |
| OPT-04, `llvq-quant/src/schur.rs:36` | Même accès `factor.u(start + k, end + j)` dans une boucle interne, mais hors de mes fichiers. Même réécriture possible avec `u_row`, même argument d'identité. |
| OPT-04, `GptqFactor::solve_block` | Lit `U` par colonne (`u(s + k, s + j)`, pas de `n`), donc pas de tranche contiguë. Coût en `b²` par ligne, faible. |

### OPT-01 : le plan précis

Trois chargements passent par `Cuda::on_stream` (NVRTC puis `cuModuleLoadData` du PTX) à chaque démarrage :
`llvq-llm/src/persist4e_cuda.rs:696` (module 4E), `llvq-llm/src/fused_cuda.rs:802` (unité fusée),
`llvq-cuda/src/attn.rs:77` (`AttnModule::on_stream_for`). Le cache existe : `Cuda::on_stream_cached`
(`llvq-cuda/src/gpu.rs:345`), clé (sha256 du texte, arch, NVRTC, pilote, sm), branché seulement pour les modules
d'arbre (`persist4e_cuda.rs:1187`, via `TreeModules::Cubin`, et `served_decode.rs:install_chain`).

1. Lire le répertoire une fois au chargement servi : `llvq_cuda::modcache::dir_from(|v| std::env::var(v).ok())`,
   comme `install_chain`.
2. 4E : remplacer l'appel de `persist4e_cuda.rs:696` par `on_stream_cached(stream.clone(), &src, dir.as_ref())` et
   pousser sa ligne dans `lines` (« 4E: module cache: hit/miss ... »). Le répertoire arrive par un paramètre de
   l'installation, pas par une lecture d'environnement enfouie.
3. Unité fusée : même remplacement à `fused_cuda.rs:802`, la ligne imprimée à côté de « NVRTC source: ».
4. Attention : ajouter `AttnModule::on_stream_cached_for(stream, groups, dir)` à côté de `on_stream_for`, sans changer
   la signature existante, et l'appeler depuis le chargement servi seulement.
5. Documenter le nouveau périmètre de `LLVQ_MODULE_CACHE` dans `docs/ENV.md`, `configs/README.md`,
   `llvq-runtime/README.md` ; la ligne de `CLAUDE.md` est à l'opérateur.

Risque à lire sur carte : le chemin caché charge le cubin de `persbench::driver_cubin` (`cuLinkComplete`) au lieu de
laisser `cuModuleLoadData` compiler le PTX. Même texte PTX, même ptxas du pilote, mais pas forcément les mêmes options
de JIT. Les registres, les octets locaux, la résidence et les jetons doivent donc être relus, pas supposés.

Mesure qui le validerait (sur carte L40S, environ 0,1 $ *estimé* par l'audit, go opérateur requis) : la porte servie
4B deux fois de suite sur un cache vide, comme l'étape (1) de `proofs/preregistration-chain-prefill-served-2026-11-15.md`,
`LLVQ_CONFIG=configs/qwen3-4b-tetra-e4-4e-v9-chain.json LLVQ_MODULE_CACHE=$O/modcache fusedrun <sealed 4B> 8`, puis la
même chose sous `configs/qwen3-4b-tetra-e4-4e-v9.json`. Ce qu'elle doit montrer : au premier passage des lignes
« miss » pour 4E, l'unité fusée et l'attention ; au second des lignes « hit » et un temps de chargement inférieur
aux 55,2 s du journal `chain-prefill-served-2026-11-15` ; les lignes d'installation 4E (registres, octets locaux,
résidence) identiques à celles du journal ; les 256 jetons de la porte identiques à ceux publiés ; `oracle` vert
d'abord, règle 10.

## Constat faux ou déjà corrigé

| ID | preuve |
|---|---|
| aucun | OPT-02 et OPT-04 revérifiés dans le code : `repl.rs:65` trie bien tout le vocabulaire à chaque jeton à température positive ; `gptq.rs:389` lit bien `factor.u(s + k, r0 + j)` cellule par cellule. Aucun vérificateur (`20-` à `23-`) ne les nuance. |

## Décision d'opérateur

| sujet | option recommandée | pourquoi |
|---|---|---|
| OPT-01 : étendre `LLVQ_MODULE_CACHE` au module 4E, à l'unité fusée et à l'attention, sous toute config servie | oui, après la mesure ci-dessus | le texte et son sha256 ne bougent pas ; seule la voie de chargement change, et elle se vérifie sur carte en une porte |
| OPT-02 : garder la sélection malgré 37 lignes | la garder si les tests d'identité et les mutants passent | repli exact sur l'ancien code ; sinon, retour à `nucleus_sorted` seul en une ligne |
| OPT-04 : `GptqFactor::u_row` est une méthode publique nouvelle | l'accepter | ajout pur, aucune signature existante ne change, et `schur.rs` peut s'en servir ensuite |

## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)

Aucun. `repl.rs`, `gptq.rs`, `linalg.rs` et les deux tests compilent sur toutes les cibles.

## Commandes lancées et leur résultat (les lignes `test result:` telles quelles)

Aucune compilation, aucun test : règle du brief des lots 2 à 7. Lancé seulement `git diff`, `git grep`, `grep`, `sed`,
`awk` (longueur de ligne).

## À lancer quand la machine est libre

| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo clippy -p llvq-llm --all-targets -j 8` | zéro avertissement | aucun |
| `nice -n 10 cargo clippy -p llvq-quant --all-targets -j 8` puis la même avec `--features fast-linalg` | zéro avertissement, `forbid(unsafe_code)` tenu | aucun |
| `nice -n 10 cargo test -p llvq-llm --lib repl -j 8` | les 3 tests neufs de `repl.rs` (`the_selected_nucleus_is_the_sorted_one_bit_for_bit`, `a_top_p_on_the_edge_of_a_share_cuts_where_the_sort_cuts`, `a_tie_in_the_nucleus_falls_back_to_the_sort`) et les 8 anciens verts | `repl.rs`, `nucleus_selected`, remplacer tout le bloc `let settled = c.iter().any(...)` par `let total: f32 = c.iter().rev().map(\|x\| x.1).sum(); let settled = true;` (total réassocié, sommé à l'envers) : rouge attendu dans `a_top_p_on_the_edge...`, et dans `the_selected_nucleus...` sur les lignes plates |
| idem | idem | `repl.rs`, `nucleus_selected`, `acc += x.1 / total;` devient `acc += x.1 * (1.0 / total);` : rouge attendu dans `a_top_p_on_the_edge...` |
| idem | idem | `repl.rs`, `nucleus_selected`, `let strict = c[..=i + 1]...;` devient `let strict = true;` : rouge attendu dans `a_tie_in_the_nucleus...` |
| `nice -n 10 cargo test -p llvq-llm --test sampler_nucleus -j 8` | `pick_draws_the_reference_token_on_every_row` et `..._at_degenerate_settings` verts : même jeton et même état du générateur que la copie de l'ancien `pick`, sur lignes piquées, plates, mantisse f16, NaN, infinis, constante, deux maxima égaux, températures et `top_p` dégénérés | mutant `let strict = true;` ci-dessus : rouge attendu sur « two equal maxima » ou les lignes en mantisse f16 (le jeton dépend de l'ordre des égalités) |
| `nice -n 10 cargo test --release -p llvq-llm --test sampler_nucleus -j 8` | en plus, `pick_draws_the_reference_token_at_vocabulary_size` vert à 151 936 jetons | idem |
| `nice -n 10 cargo test --release -p llvq-llm --test sampler_nucleus -j 8 -- --ignored --nocapture pick_times_against_the_sorted_reference` | MESURE avant/après sur le Mac, CPU : temps par tirage de `pick_reference` (ancien) et de `pick` (nouveau), médiane et plage sur 9 tours entrelacés de 64 tirages, sur une ligne piquée et une ligne plate de 151 936 logits synthétiques, température 0,7, `top_p` 0,9. Chronomètre tout `pick` (copie vers l'hôte comprise, sans modèle). À étiqueter *mesuré*, avec la machine | aucun |
| `nice -n 10 cargo test -p llvq-quant --test gptq_row_slice -j 8` puis la même avec `--features fast-linalg` | `a_row_of_u_is_the_entries_u_returns`, `..._on_a_scalar_grid`, `..._on_leech_and_in_parallel` verts : couche identique au bit à la boucle indexée copiée, rétraction on/off, avec et sans queue, une ligne nulle (tous les `k` sautés), quatre fils, sur les deux factorisations | `gptq.rs`, propagation, `*cell -= x * ukj;` devient `*cell = (-x).mul_add(ukj, *cell);` (FMA, un arrondi en moins) : rouge attendu dans les deux tests de couche |
| idem | idem | `gptq.rs`, propagation, `for k in 0..b {` (la boucle qui suit `let row = &mut weights.w[...]`) devient `for k in (0..b).rev() {` (somme par cellule réassociée) : rouge attendu dans les deux tests de couche |
| idem | idem | `linalg.rs`, `u_row`, `at(self.n, i, 0)..` devient `at(self.n, i, 1)..` : rouge attendu dans `a_row_of_u_is_the_entries_u_returns` et dans les tests de couche |
| `nice -n 10 cargo test -p llvq-quant --test g5_gptq --test g5_spherical -j 8`, avec et sans `--features fast-linalg` | `parallel_matches_serial_exactly`, `both_factorizations_agree` et `spherical_feedback_matches_an_independent_reference_on_tetra` toujours verts : la référence indexée sur Tetra tient | mutant `mul_add` ci-dessus : rouge attendu aussi dans `spherical_feedback_matches_an_independent_reference_on_tetra` |
| `nice -n 10 cargo test --release -p llvq-quant --features fast-linalg --test gptq_row_slice -j 8 -- --ignored --nocapture` | MESURE avant/après sur le Mac : temps d'une couche 512 × 4 096, grille scalaire (la propagation domine), un fil, boucle indexée copiée contre `quantize_layer`, médiane et plage sur 5 tours entrelacés, avec l'égalité au bit vérifiée à chaque tour. À étiqueter *mesuré*. Limite : la copie indexée vit dans la crate de test, compilée au même profil ; ce n'est pas l'ancien binaire | aucun |
| `nice -n 10 cargo test --release -p llvq-quant --features fast-linalg -j 8 -- --include-ignored` | la suite de la crate verte avant tout commit : la propagation sert l'encodeur qui produit les fichiers scellés | aucun |

Optionnel, go opérateur requis (calcul lourd local) : encoder trois blocs du 0.6B par `smoke` avec `LLVQ_ARTIFACT`, une
fois à `925930d` et une fois sur cette branche, avec les mêmes arguments, puis comparer le sha256 des deux artefacts
(attendu identique) et la phase de quantification imprimée.
