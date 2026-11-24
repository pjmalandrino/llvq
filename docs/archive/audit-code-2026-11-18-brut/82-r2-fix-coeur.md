# Second tour, crates cœur (R2-SRC-01 à 03, R2-DIF-42, R2-DIF-46) : journal

Écrit sans aucune exécution : ni cargo, ni test, ni clippy. Seuls lecture, `git grep`, `git diff` et `python3 -c` pour
l'arithmétique. Tout le code ci-dessous est relu deux fois, NON COMPILÉ.

Les cinq constats ont été revérifiés dans le code avant correction. Les cinq sont exacts.

## Corrigé

| ID | fichiers | ce qui change, en une ligne | vérification faite |
|---|---|---|---|
| R2-SRC-01 | `llvq-search/src/generic.rs`, `llvq-search/tests/g2c_reference.rs`, `llvq-llm/src/bin/smoke.rs` (commentaire du parseur) | `nearest_by` ne pousse la section du coset impair que si `odd_flat` a une entrée (`.first()`), option (a) ; test neuf `a_one_level_codebook_is_searchable` contre la forme close | relu, NON COMPILÉ |
| R2-SRC-02 | `llvq-artifact/src/format.rs`, `llvq-artifact/tests/hostile_files.rs` | `max_index(kind, cap)` = `N(cap)` pour Ball, `u64::MAX` sinon ; le lecteur (`read_lattice_body`) et les deux écrivains (`write_codes`, `code_stream`) refusent tout index Ball `> N(cap)` en `IndexOutOfRange` ; deux tests neufs | relu, NON COMPILÉ |
| R2-SRC-03 | `llvq-quant/src/linalg.rs`, `llvq-quant/tests/g5_gptq.rs` | le chemin `faer` recopie le pivot que `LltError::NonPositivePivot { index }` porte ; champ public neuf `of_inverse` sur `NotPositiveDefinite`, et un texte distinct quand c'est la factorisation de `H⁻¹` qui cède ; test neuf `a_dead_channel_is_reported_at_its_own_pivot` | relu, NON COMPILÉ |
| R2-DIF-42 | `llvq-bench/tests/tetra_refusals.rs` | deux assertions sur la sortie réelle de `rtbits` : la ligne carte (128 bits, 2,6667 b/weight, 32 bits de rembourrage) et le b/weight noyau carte (3,3333) | relu, NON COMPILÉ |
| R2-DIF-46 | `docs/ENV.md` | la ligne `LLVQ_SEALED_ARTIFACT` nomme aussi `llvq-cuda/tests/golay70_decoder_matches_rust.rs` ; format de table inchangé, le nom reste entre backquotes | relu |

### R2-SRC-01 : pourquoi (a) et pas (b)

Sous `with_level_cap(1)`, il reste une classe : la classe paire `(±2)²⁴`, coquille 6, mot de Golay de poids 24, un
seul niveau. Aucune classe impaire, aucune classe paire de poids 0. `odd_flat` est vide, `even_by_w` a un groupe.

Avec `.first()` : la liste `sections` ne contient que les groupes pairs. La branche `sec == usize::MAX` n'est jamais
prise, donc la lecture `self.odd_flat[0]` de la ligne 818 n'est jamais atteinte. Sauter le coset impair est exact,
puisqu'aucun de ses points n'est dans le code-livre. Le reste de la recherche est inchangé : l'élagage, l'ordre et la
matérialisation sont ceux du chemin L ≥ 2. Le chercheur rend donc le plus proche voisin du code-livre plafonné.

Pour tout code-livre qui marchait, `odd_flat` est non vide : `first()` rend `odd_flat[0]`, la section poussée est la
même, les voisins sont les mêmes au bit.

Le test le prouve sans passer par `shell_bests` (qui panique sous L1 dans `materialize` pour les 11 coquilles vides,
chemin sans appelant de production, laissé tel quel) : la réponse a une forme close. Tous les points ont la même
norme, donc le gagnant angulaire est le gagnant en produit scalaire : `2·sign(xᵢ)` partout, et si le nombre de
signes moins est impair, le signe de la plus petite `|xᵢ|` est retourné (condition d'appartenance : `Σx ≡ 0 mod 8`,
donc un nombre pair de `−2`). Le chemin euclidien est aussi vérifié contre l'origine à clé 0, aux β 0,05 et 0,5, qui
font gagner tantôt le point, tantôt l'origine.

Le test `an_empty_codebook_is_refused` de `smoke.rs` dit déjà « degenerate but searchable » pour `leech1c6L1` : c'est
vrai maintenant, il ne change pas. Le commentaire du parseur disait que `L1` vide le code-livre ; il dit maintenant
« jusqu'à la coquille 5 ».

### R2-SRC-02 : la borne, sa source, son coût

- Source du nombre : `llvq_search::classes::enumerate_classes(cap)` puis `ClassSet::shell_cardinality(m)` sommé sur
  `2..=cap`, la somme exacte que `llvq_quant::quantizer::index_bits` fait déjà. `quantizer.rs` n'est pas dans mes
  fichiers : je n'ai pas pu y sortir une `ball_size` partagée. La somme est donc écrite une seconde fois dans
  `format.rs::max_index`, sur la même fonction de `llvq-search`. Les deux peuvent être fusionnées plus tard en
  faisant appeler `max_index` par `index_bits` ou l'inverse.
- Ordre de l'index : `index.rs` range les coquilles par ordre croissant, l'origine en 0. Donc `idx ≤ N(cap)` équivaut
  à « point de `Λ₂₄(cap) ∪ {0}` ».
- `N(12)` = 111 043 117 458 000 et `N(13)` = 280 974 212 784 720 (*computed*, série thêta de Λ₂₄ par
  `65520/691·(σ₁₁(m) − τ(m))`, `python3 -c`). `N(13)` retrouve `N_SHELL_13_CUMULATIVE`. Le test épingle de plus `N(12)`
  par l'`Indexer` : `decode(N12)` est sur la coquille 12, `decode(N12 + 1)` sur la 13.
- Cap 0 et 1 : `N = 0`, seul l'index 0 (l'origine) passe. Pas de refus par nom de `cap < 2` dans `index_width` : il
  toucherait les écrivains et `code_widths` (utilisé par `hfpack`) pour une entrée que le contrôle par index borne
  déjà. Aucun écrivain du dépôt n'écrit un cap < 2 (`set_shell_cap` l'interdit, le parseur de `smoke` aussi).
- Écrivains revus (aucun ne peut écrire un index au-delà de son cap) : `calib.rs:1556` et `smoke` écrivent
  `max_shell` du code-livre `ShapeGain`, dont le chercheur a `set_shell_cap(max_shell)` et saute `h.shell > cap` dans
  les trois boucles de `nearest_by` ; `reproject` ne fait que changer le signe du point, même coquille ; un bloc nul
  donne l'index 0. `lswap` re-cherche avec `set_shell_cap(12)`. `rhoapply`, `artifact2` recopient des index lus.
  L'archive `~/llvq-q4b.llvq` (Ball cap 12) vient de cet encodeur : tous ses index sont `≤ N(12)`, le refus ne la
  touche pas. À confirmer par la suite complète (file ci-dessous).
- Fixtures revues : `hostile_files` (index 3, 5, 7, 9, 0..212), `int4_format`, `mixed_kinds`, `g6_format`,
  `codebook_fingerprint` (tirages filtrés `shell ≤ 12`), `int4_refusals`, `int4_refusal`, `int4_wiring` (index 0),
  `hfpack` (index < 1000), `tetra_format`, `runtime_format` (cap 13, `N13` borne). Les `RawMatrix` de `radixstudy`
  avec des classes de coquille 13 sous cap 12 ne sont jamais écrits ni relus : ils vont en mémoire au balayage. Les
  tests qui lisent un enregistrement décalé (`a_v5_record_read_as_v4…`, `g6_format` tag retourné, `int4_wiring`
  étiquette fausse) acceptent tout `Err`.
- Coût de lecture (*estimated*) : `enumerate_classes(cap)` une fois de plus par enregistrement Ball (déjà appelée une
  fois par `index_width`), moins d'une milliseconde, 252 enregistrements au 4B. Puis une comparaison `u64` par bloc,
  hors de toute dépendance de données, à côté des deux lectures de bits qui existent déjà : environ 150 M blocs au 4B,
  de l'ordre de 0,1 s au plus. Le contrôle reste dans la boucle de lecture ; le placer ailleurs coûterait une seconde
  passe.
- Tetra : `max_index` rend `u64::MAX`, aucun comportement ne change sur les fichiers servis.

### R2-SRC-03 : ce que faer donne

Lu dans `~/.cargo/registry/src/…/faer-0.24.4` (la version du `Cargo.lock`) : `LltError` est
`enum { NonPositivePivot { index: usize } }`, sans `#[non_exhaustive]`, réexporté en `faer::linalg::solvers::LltError`.
`index` est la colonne globale du pivot (`Err(j + fail_idx)` dans `ldlt/factor.rs:418` et `:633`). La régularisation
par défaut est nulle (`LltRegularization::default`). Le test est `!(diag > 0)`, donc un pivot NaN est refusé aussi :
le doute du relecteur sur NaN sous `fast-linalg` est levé pour `llt`.

`git grep NotPositiveDefinite` : aucun autre crate ne construit ni ne filtre le type (`calib.rs` l'affiche,
`e8gptq` et `g5_calibration` font `expect` ou `is_err`). Le champ public neuf `of_inverse` ne casse donc aucun appelant.
Le chemin sans la feature garde ses `Ok`/`Err` et ses pivots ; seul le texte change quand c'est la seconde
factorisation (celle de `H⁻¹`) qui cède. Le texte de la première est mot pour mot l'ancien. `git grep -nF` de
« not positive definite » et « at pivot » dans `ops/ paper2/ docs/METHODE.md .github/` : rien.

## Laissé de côté

| ID | raison |
|---|---|
| R2-SRC-01, `shell_bests` sous L1 | `materialize` panique (`unreachable!`) pour les coquilles sans classe ; seul `nearest_ball13`, sans appelant de production, y mène. Hors du constat à corriger |
| R2-SRC-02, refus par nom de `cap < 2` | le contrôle par index suffit (seule l'origine passe) ; un refus dans `index_width` toucherait aussi `code_widths` et `hfpack` |
| R2-SRC-02, `ball_size` partagée | `llvq-quant/src/quantizer.rs` hors de mes fichiers ; la somme est écrite deux fois sur la même fonction de `llvq-search` |
| R2-SRC-03, branche `of_inverse: true` | aucune entrée connue ne factorise `H` et casse `H⁻¹` de façon déterministe sur les deux chemins ; le test vérifie le texte de cette branche, pas son déclenchement |

## Constat faux ou déjà corrigé

Aucun.

## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)

Aucun. `docs/ENV.md` nomme un test CUDA, sans le toucher.

## À lancer quand la machine est libre

| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo test -p llvq-search --test g2c_reference -j 8` | `a_one_level_codebook_is_searchable` ok, les autres inchangés | `llvq-search/src/generic.rs:807-809` : remplacer le `if let Some(h) = self.odd_flat.first() { … }` par `sections.push((self.odd_flat[0].bound, usize::MAX));` → le test panique « index out of bounds: the len is 0 » |
| `nice -n 10 cargo test -p llvq-artifact --test hostile_files -j 8` | `a_ball_index_past_its_cap_is_refused_on_read` et `a_ball_record_of_cap_1_holds_the_origin_only` ok, les autres inchangés | (1) `format.rs:1182`, `if idx > max_idx` → `if false` : les deux tests rougissent à la lecture. (2) `format.rs:986` (`code_stream`) idem : rouge sur « raw writer ». (3) `format.rs:836` (`write_codes`) idem : rouge sur « point writer ». (4) `format.rs:650-653`, bras Ball → `u64::MAX` : tout rouge |
| `nice -n 10 cargo test -p llvq-artifact -j 8` | aucun test existant ne rougit (fixtures revues ci-dessus) | |
| `nice -n 10 cargo test -p llvq-quant --test g5_gptq a_dead_channel -j 8` | ok sur le chemin de référence | `linalg.rs:152` `of_inverse: false` → `true` : rouge |
| `nice -n 10 cargo test -p llvq-quant --features fast-linalg --test g5_gptq a_dead_channel -j 8` | ok sur le chemin `faer`, `at == 5` | `linalg.rs:115` `at: index` → `at: 0` : rouge SOUS `fast-linalg` SEULEMENT (vert sans la feature, c'est attendu) |
| `nice -n 10 cargo test -p llvq-bench --test tetra_refusals -j 8` | `rtbits_prices_a_tetra_file_instead_of_refusing_it` ok | `llvq-bench/src/bin/rtbits.rs:442` `==` → `!=` : rouge, la sortie lit « 96 bits, 2.0000 b/weight, 0 pad bits » et « … 2.6667 » |
| `nice -n 10 cargo test -p llvq-llm --test env_registry -j 8` | vert | |
| `nice -n 10 cargo test -p llvq-llm --bin smoke an_empty_codebook -j 8` | vert (commentaire seul) | |
| `nice -n 10 cargo clippy --all-targets -j 8` puis `nice -n 10 cargo clippy -p llvq-quant --features fast-linalg --all-targets -j 8` | zéro avertissement ; la seconde compile `fast::pivot` et l'import de `LltError` | |
| `cargo test --release -- --include-ignored` avec `~/llvq-q4b.llvq` présent | **EXIGÉE** : `llvq-artifact` et `llvq-search` sont touchés, et le lecteur Ball refuse désormais un index au-delà de `N(cap)`. Les tests d'archive doivent relire le fichier Ball cap 12 sans `IndexOutOfRange`. Compter les lignes `test result:`, ne pas tronquer | |
| `cargo test --release -p llvq-quant --features fast-linalg` | `both_factorizations_agree` et le test neuf verts sur le chemin payé | |
| sur Linux : `LLVQ_SEALED_ARTIFACT=… cargo test --release -p llvq-cuda --test golay70_decoder_matches_rust -- --ignored` | la relecture Ball cap 12 de l'archive passe aussi par le lecteur modifié | |

## Commandes lancées et leur résultat

Aucune commande de compilation ni de test. Lancé : `git grep`, `git diff`, `sed`/`grep` en lecture, et
`python3 -c` pour `N(12)`, `N(13)` et les b/weight de `rtbits` (128/48 = 2,6667 ; 160/48 = 3,3333).
