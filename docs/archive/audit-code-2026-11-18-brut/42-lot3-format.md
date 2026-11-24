# Lot 3, durcissement des lecteurs de format, et NUM-01 (code) : journal

Écrit sans aucune exécution : ni cargo, ni test, ni clippy. Seuls `git grep`, `git diff`, `git log` et `python3 -c`
pour l'arithmétique. Tout le code ci-dessous est relu, NON COMPILÉ.

`read_raw` est dans `llvq-artifact/src/sealed.rs` (lignes 288-329 à `925930d`), pas dans `llvq-llm/src/sealed.rs`.
Ce dernier n'en est qu'un appelant (`:554`, puis `to_f32` sur un tenseur quantifié) ; il n'avait rien à changer, la
validation à la lecture le couvre. Les autres appelants (`fused.rs:2549`, `hfpack.rs:334`, `export`, `embedq`,
`rowscale`, `dflash`, `rtbits`) sont couverts de même : ils passent tous par ce lecteur.

## Corrigé

| ID | fichiers | ce qui change, en une ligne | vérification faite |
|---|---|---|---|
| QUA-01 / NUM-04 / SEC-06 | `llvq-artifact/src/sealed.rs` | `read_raw` lit la largeur en u32 et refuse tout autre que 4 ou 8 avant de la rétrécir ; puis appelle `check_quant`, le contrôle que `write_raw` fait (groupe non nul, produit des dims sans débordement, longueurs `packed`/`scales`/`biases`) ; une f16 dont le compte n'égale pas le produit des dims est refusée après lecture des valeurs, si bien qu'un compte menteur reste `Truncated`. Tout en `Error::Inconsistent` nommé par le tenseur. `write_raw` refuse aussi `group = 0` au lieu de diviser par zéro | relu deux fois, NON COMPILÉ |
| QUA-02 / NUM-05 | `llvq-artifact/src/format.rs` | lecture : `read_lattice_body` refuse 0 centroïde sur un enregistrement qui a des blocs, et tout gain `>= n_centroids` dans la boucle ; écriture : `write_codes` (avant l'en-tête) et `code_stream` refusent un gain `>= n_centroids`. Tetra : `gain_width` impose déjà 2 centroïdes et 1 bit de gain, les gains valent 0 ou 1, le contrôle ne peut pas s'y déclencher | relu, NON COMPILÉ |
| QUA-03 | `llvq-artifact/src/format.rs` | `check_tail` exige `tail.len() == d_out · (d_in mod 24)` dans `write_codes` et `code_stream`, avant toute largeur et avant tout octet. Le cas du 4B (2 560 = 24 × 106 + 16, 16 par ligne) est testé valide | relu, NON COMPILÉ |
| QUA-12 | `llvq-artifact/src/format.rs`, `sealed.rs` | `u32_field` (`pub(crate)`, `u32::try_from` vers `Error::Inconsistent`) remplace `as u32` pour la longueur de nom, `d_out`, `d_in`, le nombre de centroïdes (`put_record_head`, converti avant le premier octet), les compteurs de `seal`, le rang et le groupe de `write_raw` ; `put_str` fait de même, le nom tronqué à 64 caractères dans l'erreur | relu, NON COMPILÉ |
| NUM-06 | `llvq-quant/src/quantizer.rs` | `TetraShapeGain::quantize` panique par nom (« non-finite weight … of a Tetra block ») sur tout poids non fini, en release, avant le calcul de la norme. Le test porte sur chaque élément, pas sur la norme : un bloc fini ne prend aucune branche nouvelle, et un bloc fini mais immense reste codé comme avant | relu, NON COMPILÉ |
| NUM-01 (code) | `llvq-bench/src/bin/rtbits.rs` | `report_v5` tient un second compte, `lattice_card_stream_bits` (`d_out · stride_u32(d_in/24) · 32` par matrice Tetra, compte fichier pour une matrice Ball) et imprime trois lignes nouvelles : le flux carte et ses bits de rembourrage, le b/weight noyau « tail at f16, Tetra rows at the card's tetra48 stride », et une ligne « f16 + tetra48 stride » au tableau modèle entier. Aucune ligne existante ne change d'un caractère | relu, NON COMPILÉ |

Vérifications faites à la main pour NUM-01 (*computed*, `python3 -c`) : `llvq-bench` dépend déjà de `llvq-artifact`
(`llvq-bench/Cargo.toml`) ; `tetra48` est `pub mod` (`lib.rs:59`) et `stride_u32` est `pub fn`. Sur les formes du 4B
scellé (168 matrices Tetra : q, k, gate, up à 2 560 sur 36 couches, down à 9 728 sur 24), le compte fichier redonne
exactement les 2 862 612 480 poids, 14 647 296 de queue, 946 176 lignes et 1,9898 b/weight de flux de
`docs/mesures/sealed-4b-2026-09-23-brut/rtbits-sealed.txt`. Le rembourrage vaut 884 736 × 32 + 61 440 × 16 =
29 294 592 bits. Le b/weight noyau f16 passe de 2,5420 (imprimé) à 2,5501, le 2,550 de la porte
(`served-v9-2026-11-07-brut/door.txt:7`). Le test le vérifie.

Avant de toucher une ligne imprimée : `git grep -nF` de « b/weight of stream », « tail at f16 », « f16 (served) »,
« WHOLE MODEL », « carried MEASURED », « kernel b/weight, over », « tail at f32 » dans `ops/`, `llvq-bench/tests/`,
`paper2/`, `.github/`, `docs/METHODE.md`. Seules prises : `ops/e8_stage1.py:216` (un `print` de texte libre, ne lit
rien) et `ops/floor.py`, `ops/gptq_quantize.py`, `ops/iq2_metal.sh` (« WHOLE MODEL » dans leurs propres messages).
Les tests `tetra_refusals.rs` et `int4_refusals.rs` lisent « lattice matrices », « int4 matrices », « kernel b/weight »
et « NO CLASS STATISTIC IS PRINTED », que je ne touche pas. Aucune ligne existante n'a changé ; trois lignes s'ajoutent.

## Laissé de côté

| ID | raison |
|---|---|
| QUA-04 | Refactorisation, à faire avec un compilateur. Plan : faire de `write_codes` un encodeur qui construit un `RawMatrix` (indices par `encode`, gains copiés, mêmes `row_scales`, `centroids`, `tail`, `rotation_seed`, `shell_cap`, `kind`) puis appelle `write_matrix_raw` ; `QuantizedMatrix::bits` et le `Ok(..)` de `write_matrix_raw` deviennent une seule fonction de comptage. Après ce lot, les deux chemins portent les mêmes contrôles (QUA-02, QUA-03 posés dans les deux), ce qui réduit le risque de dérive sans l'éliminer. Portes : `raw_passthrough_is_byte_identical`, `the_default_writer_still_writes_v4`, `code_stream_is_the_record_bytes`, puis la suite complète en release avec les archives scellées. Coût de l'encodeur : une copie des indices, 150 M `u64` au 4B, à mesurer |
| QUA-05 | Refactorisation, à faire avec un compilateur. Plan : un module privé `llvq-artifact/src/io.rs` qui porte `put_u32`, `put_u64`, `get_u32`, `get_u64`, `get_bytes`, `PREALLOC_CAP`, une seule `get_u16s` (signature `n: u64`, celle de `format.rs`, les appels de `sealed.rs` passent `groups as u64`) et `u32_field`, que ce lot a déjà posé `pub(crate)` dans `format.rs` et que `sealed.rs` appelle. `format.rs` et `sealed.rs` l'importent. Octets écrits inchangés par construction ; même porte que QUA-04 |
| QUA-12, sites restants | `format.rs:1482` (`m.group as u32`) : le groupe est vérifié égal à 128 trois contrôles plus haut, la conversion ne peut pas tronquer. `sealed.rs` `q.bits as u32` : élargissement d'un `u8`. Les `as u64` d'un `usize` : sans perte sur 64 bits. Aucun test : forcer le refus demande un nom de 4 Gio ou un `Vec` de 2³² centroïdes |
| QUA-01, `to_f32` | `RawTensor::to_f32` panique toujours sur un tenseur construit à la main qui saute `write_raw` ; tout tenseur lu par `read_raw` passe désormais les contrôles. Rendre `Result` change une signature publique appelée hors du crate (`llvq-llm`, `rtbits`) : à faire avec un compilateur. Documenté dans sa doc |
| NUM-06, Ball | `LeechShapeGain::quantize` n'a pas le refus ; le chercheur Ball n'est pas servi et la mission vise `TetraShapeGain` |
| NUM-06, `set_row_scale(NaN)` | Le quantificateur retombe sur 1,0 quand `calib.rs` stocke la valeur NaN. Avec le refus, la première ligne NaN fait tomber le run avant l'écriture, donc la divergence ne peut plus produire de fichier. `calib.rs` n'est pas dans mes fichiers |
| hors constat | `write_raw` écrit encore l'étiquette, le nom et les dims avant de refuser une charge incohérente, et `write_codes` l'en-tête avant `PointOutsideCodebook` ou `IndexTooWide`. C'était déjà ainsi ; les contrôles nouveaux de `write_codes` passent avant l'en-tête. Remonter les autres est une restructuration |

## Décision d'opérateur

- **NUM-01, quel chiffre publier.** `rtbits` imprime maintenant les deux comptes côte à côte, sans rien retirer. Les
  documents citent la ligne « f16 (served) », compte fichier du flux. La ligne « f16 + tetra48 stride » est ce que
  la carte tient, et c'est la définition écrite du papier (« at the widths the GPU holds »). Je recommande de
  republier depuis la ligne carte (4B 2,7393, 8B 2,7001, 14B 2,7324, chiffres du vérificateur, *computed*) après un
  `rtbits` sur les trois fichiers scellés. Je n'ai touché aucun document.
- **Un enregistrement Ball à 0 centroïde.** Il est refusé à la lecture et à l'écriture dès qu'il a un bloc. Un tel
  enregistrement dont tous les blocs sont l'origine se décodait sans panique avant ce lot (l'origine ne lit pas de
  centroïde) : c'est le seul cas où un refus nouveau touche une entrée qui ne paniquait pas. Aucun écrivain du dépôt ne
  le produit (`fit_gain_centroids` rend au moins un niveau) et la mission demande ce refus. Je recommande de le garder.

## Constat faux ou déjà corrigé

| ID | preuve |
|---|---|
| NUM-04, SEC-06 (partie) | « Les chemins servis sont protégés ailleurs » est faux, comme le vérificateur l'établit : les normes passent par `to_f32` sans contrôle dans les trois chargeurs fusés. La correction de ce lot est dans le lecteur, donc les couvre |
| QUA-02 (partie) | « refuser `n_cent == 0` pour Tetra » était déjà fait par `gain_width` (`gb == 1` exige deux centroïdes) |

## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)

Aucun.

## Commandes lancées et leur résultat

Aucune compilation, aucun test (règle des lots 2 à 7). Lancé : `git grep`, `git diff`, `git log -S`, et
`python3 -c` pour les comptes du 4B (884 736, 61 440, 14 647 296, 29 294 592, 2,542045, 2,550108).

## À lancer quand la machine est libre

`llvq-artifact` et `llvq-quant` sont touchées : la suite complète en release est due avant tout commit.

| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo check -p llvq-artifact --all-targets -j 8` | compile | — |
| `nice -n 10 cargo check -p llvq-quant --all-targets -j 8` | compile | — |
| `nice -n 10 cargo check -p llvq-bench --all-targets -j 8` | compile | — |
| `cargo clippy --all-targets` | zéro avertissement | — |
| `nice -n 10 cargo test -p llvq-artifact --test hostile_files -j 8` | tous verts, dont les 11 nouveaux | voir les lignes suivantes, une par test |
| idem, `a_raw_quant_width_is_checked_as_stored_not_narrowed` | vert | `sealed.rs:353-361` : remplacer le `match` par `let bits = get_u32(r, "quant width")? as u8;` → rouge sur 0x104 (lu 4, rend `Ok`) |
| idem, `a_raw_quant_group_of_zero_is_refused_not_a_division_by_zero` | vert | `sealed.rs:255-257` : supprimer le bloc `if q.group == 0` → rouge, panique « attempt to divide by zero » dans `div_ceil` |
| idem, `a_raw_quant_payload_short_of_its_dims_is_refused_not_a_panic` | vert | `sealed.rs:376` : supprimer `check_quant(&name, &dims, &q)?;` → rouge (`Ok`) ; le test de largeur reste vert, preuve que les deux contrôles sont distincts |
| idem, `raw_dims_whose_product_overflows_are_refused_by_name` | vert | `sealed.rs:237` : `.try_fold(1usize, \|acc, &d\| acc.checked_mul(d))` → `.try_fold(1usize, \|acc, &d\| Some(acc.wrapping_mul(d)))` → rouge (2⁸⁰ replié en 0, détail « packed bytes » au lieu de « usize ») |
| idem, `a_raw_f16_count_that_disagrees_with_its_dims_is_refused` | vert | `sealed.rs:342-347` : supprimer le `if` → rouge (`Ok`) |
| idem, `honest_raw_tensors_still_read_byte_for_byte` | vert | `sealed.rs:263` : `8 => n,` → `8 => n + 1,` → rouge (le tenseur int8 honnête est refusé à l'écriture) |
| idem, `a_ball_gain_past_its_centroids_is_refused_on_read_not_a_panic` | vert | `format.rs:1144-1146` : supprimer le `if gain as usize >= n_cent` → rouge (`Ok` avec gain 3 sur 3 centroïdes) |
| idem, `a_ball_record_with_blocks_and_no_centroid_is_refused_on_read` | vert | `format.rs:1087-1095` : supprimer le bloc `n_cent == 0` → rouge sur le détail (« gain rank 0 with 0 centroids » au lieu de « no gain centroid ») ; supprimer aussi `:1144-1146` → rouge (`Ok`) |
| idem, `a_gain_past_the_centroids_is_refused_by_both_writers` | vert | `format.rs:953-955` : supprimer le contrôle de `code_stream` → rouge (`Ok`) ; puis `format.rs:775-777` dans `write_codes` → rouge, panique « value 2 does not fit in 1 bits » de `BitWriter::push` |
| idem, `a_tail_count_off_the_dimensions_is_refused_by_both_writers` | vert | `format.rs:942` : supprimer `check_tail(..)?;` de `code_stream` → rouge (33 valeurs écrites) ; puis `:771` dans `write_codes` → rouge |
| idem, `a_4b_shaped_record_and_its_tail_still_round_trip` | vert | `format.rs:556` : `let want = d_out * (d_in % DIM);` → `let want = d_in % DIM;` → rouge (32 valeurs refusées, 16 attendues) |
| `nice -n 10 cargo test -p llvq-artifact -j 8` | les autres cibles restent vertes : `raw_tensors`, `g6_format`, `tetra_format` (dont `a_tetra_matrix_with_the_wrong_cap_or_gain_width_is_refused`, qui doit garder le refus de `gain_width` avant le mien), `mixed_kinds`, `int4_format`, `codebook_fingerprint`, `runtime_format` | — |
| `nice -n 10 cargo test -p llvq-quant --lib quantizer::tests -j 8` | 4 verts | `quantizer.rs:788-794` : supprimer le `if let Some(i) …` → rouge sur les trois `should_panic` (un NaN : message de l'assert de l'encodeur ; tout NaN : aucune panique ; −inf : autre message ou aucune). Mutant de sur-refus : `!a.is_finite()` → `!(a.abs() < 1.0)` → rouge sur `finite_tetra_blocks_still_code_as_the_encoder_says` (le bloc fini monte à 1,05) |
| `nice -n 10 cargo test -p llvq-bench --bin rtbits -j 8` | vert, dont `the_card_stream_adds_the_tetra48_row_pad_to_the_file_stream` | `rtbits.rs:412` : `stride_u32(d_in / DIM)` → `(6 * (d_in / DIM)).div_ceil(4)` (rangée arrondie à 4 octets) → rouge (rembourrage nul à 2 560) |
| `nice -n 10 cargo test -p llvq-bench --test tetra_refusals --test int4_refusals -j 8` | vert : `rtbits` tourne sur un fichier v5 avec les trois lignes nouvelles | — |
| `nice -n 10 cargo test -p llvq-llm -j 8` | vert : les lecteurs de tenseurs portés (`embedquant`, `embed_q4`, `embed_q8`, `hfpack`, `proj_q4`) lisent ce que `write_raw` écrit | — |
| `cargo run --release -p llvq-bench --bin rtbits -- <le 4B scellé>` | lignes existantes identiques au caractère à `rtbits-sealed.txt` ; nouvelles lignes : 29 294 592 bits de rembourrage, 2,5501 b/weight noyau carte, b/param « f16 + tetra48 stride » q4 près de 2,7393 | — |
| `cargo test --release -- --include-ignored` | suite complète verte, archives scellées comprises (`~/llvq-q4b.llvq`) : obligatoire avant commit, le lecteur et l'écrivain du format sont touchés | — |
