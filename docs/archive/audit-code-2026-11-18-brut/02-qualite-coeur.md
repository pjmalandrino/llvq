# Qualité de code, crates cœur : constats
Périmètre réellement lu : en entier `llvq-core/src/{lib,rng,shells}.rs`, `llvq-search/src/lib.rs`, `llvq-search/src/pack.rs` (lignes 1-80), `llvq-quant/src/quantizer.rs`, `llvq-quant/src/gptq.rs` (lignes 1-420), `llvq-artifact/src/format.rs`, `llvq-artifact/src/sealed.rs`, `llvq-artifact/src/tetra48.rs`, `llvq-artifact/tests/common/mod.rs`. Survolés : `llvq-search/src/{generic,generic_ref,cns}.rs`, `llvq-search/src/tetra/encoder.rs` (constantes et provenance), `llvq-quant/src/schur.rs`, `llvq-artifact/src/{runtime,e1c,error}.rs` (plan et signatures), `llvq-bench/src/bin/{rtbits,radixstudy}.rs`, `llvq-bench/examples/rhoapply.rs`, `llvq-bench/src/f1/rankbook.rs`. Balayages outillés sur les cinq crates : `pub fn` référencées une seule fois dans le dépôt, fonctions identiques octet pour octet (hors commentaires et blancs), longueur et arité des fonctions, `#[allow]`, `let _ =`, lectures d'environnement et de fichiers.
Non couvert : `llvq-search/src/{index,classes,fastdec,rankdec}.rs` et `tetra/{rank,trellis,word}.rs` en détail, `llvq-quant/src/{linalg,rotation}.rs`, `llvq-artifact/src/{runtime,e1c,e1v,blockrec,codebook}.rs` en détail, les tests d'intégration des cinq crates, et la plupart des bins et exemples de `llvq-bench`.
Bilan : 0 critiques, 0 majeures, 8 mineures, 5 cosmétiques

### QUA-01 `read_raw` ne valide pas un tenseur quantifié, et `to_f32` panique sur ce qu'il a laissé passer
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-artifact/src/sealed.rs:305-325 (lecture), sealed.rs:126-137 (`to_f32`), sealed.rs:246-273 (l'écriture, qui valide)
- Fait : `write_raw` refuse une largeur autre que 4 ou 8 et vérifie `packed`, `scales` et `biases` contre `dims`. `read_raw` ne vérifie rien de cela. Il lit `bits` par `get_u32(..) as u8`, troncature silencieuse. Il accepte `group = 0`. Il ne compare pas la longueur F16 ni la longueur `packed` au produit de `dims`. `to_f32` traite toute largeur autre que 8 comme 4 (`_ =>`, ligne 135).
- Scénario : un fichier dont le champ `quant width` vaut `0x104` est lu comme int4 et passe le garde `q.bits == 4` du chemin servi (`llvq-llm/src/fused.rs:222`). Un `group = 0` fait paniquer `to_f32` sur `row_len.div_ceil(0)` (ligne 128). Un `packed` plus court que `dims` fait paniquer `to_f32` par un index hors bornes. Le dépôt promet une `Err` nommée sur un fichier hostile (`format.rs:536-554`, `tests/hostile_files.rs`), et ce tronçon n'en donne pas. Le chargeur servi passe par ce lecteur (`llvq-llm/src/fused.rs:2547`).
- Touche : format
- Correction proposée : dans `read_raw`, refuser `bits` hors de {4, 8} sans `as u8`, refuser `group == 0`, et vérifier `packed.len()`, `groups` et la longueur F16 contre `dims` avec les mêmes formules que `write_raw`, en `Error::Inconsistent`.
- Effort : S
- Risque de la correction : faible. Lecture seule de champs déjà présents ; aucun octet écrit ne change ; les fichiers scellés valides passent.
- Vérification proposée : trois tests dans `hostile_files.rs` : `quant width = 0x104`, `group = 0`, `packed` amputé d'un octet ; chacun doit rendre `Err`, pas paniquer.

### QUA-02 Le rang de gain n'est jamais comparé au nombre de centroïdes, et `decode_matrix` panique
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-artifact/src/format.rs:1029-1030 et 1073-1076 (lecture), format.rs:894-904 et 752-765 (écriture), llvq-quant/src/quantizer.rs:966 (l'index)
- Fait : la largeur du gain est `ceil(log2(n_centroids))`. Le lecteur prend `gb` bits par bloc sans vérifier `gain < n_centroids`, et accepte `n_centroids = 0` pour un enregistrement Ball (`gain_width` rend 0). `reconstruct_shape_gain` indexe `centroids[code.gain as usize]` sans garde.
- Scénario : un enregistrement Ball à 3 centroïdes porte 2 bits de gain ; un gain 3 est lu, décodé, puis `decode_matrix` panique sur `centroids[3]`. Même chose avec 0 centroïde et n'importe quel bloc hors origine. Les deux écrivains (`write_codes`, `code_stream`) laissent aussi passer ce gain, donc le fichier est produit par ce crate et refusé seulement par une panique. `decode_matrix` sert le chemin dense de `llvq-llm/src/artifact2.rs:109,485`.
- Touche : format
- Correction proposée : dans `read_lattice_body`, refuser `n_cent == 0` pour Ball et Tetra, et refuser tout `gain >= n_cent` pendant la boucle de lecture ; ajouter le même contrôle dans le chemin d'écriture commun (voir QUA-04).
- Effort : S
- Risque de la correction : faible. Aucun fichier publié n'a de gain hors bornes ; le format ne change pas.
- Vérification proposée : test hostile : un `RawMatrix` à 3 centroïdes et un gain 3, écrit puis relu, doit rendre `Err` à l'écriture ou à la lecture.

### QUA-03 Les écrivains de matrices ne vérifient pas la longueur de la queue
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-artifact/src/format.rs:711-726 (`write_codes`), format.rs:874-891 (`code_stream`), format.rs:688-690 (écriture de la queue), format.rs:1042-1046 (lecture)
- Fait : le lecteur lit exactement `d_out · (d_in % 24)` valeurs de queue. Les deux écrivains vérifient le nombre de codes et de `row_scales`, pas celui de `tail`, et `put_record_head` écrit toutes les valeurs reçues.
- Scénario : un `QuantizedMatrix` dont `tail` a une valeur de trop s'écrit sans erreur ; à la relecture, la valeur en trop est prise pour l'octet bas de la longueur du flux de codes, et l'erreur sort plus loin sous un autre nom (`Truncated`). Le 4B a des largeurs non multiples de 24 (2560 = 24·106 + 16), donc la queue existe sur l'objet servi. `write_matrix_int4` affirme « every invariant is checked before a byte is written, for the same reason write_matrix_raw's are » (ligne 1340), ce qui est faux pour la queue.
- Touche : format
- Correction proposée : vérifier `tail.len() == d_out * (d_in % DIM)` dans le contrôle commun des deux écrivains.
- Effort : S
- Risque de la correction : nul sur les fichiers existants ; ne touche que des entrées fausses.
- Vérification proposée : test : écrire un `RawMatrix` avec une queue trop longue doit rendre `Err::Inconsistent`.

### QUA-04 La validation et l'empaquetage d'un enregistrement treillis sont écrits deux fois
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-artifact/src/format.rs:696-770 (`write_codes`) et format.rs:864-950 (`code_stream`, `write_matrix_raw`) ; format.rs:508-514 et 946-949 (la formule des bits)
- Fait : les deux chemins refont le refus int4, le contrôle `d_out · nblocks`, le contrôle `row_scales`, le calcul des largeurs, la boucle `IndexTooWide` puis `push(idx); push(gain)`, et la somme des bits de charge utile (`QuantizedMatrix::bits` et le `Ok(..)` de `write_matrix_raw`). Ils ont déjà divergé : `write_codes` passe par `index_width` et `gain_width`, `code_stream` par `code_widths` ; QUA-02 et QUA-03 manquent dans les deux.
- Scénario : chaque nouvel invariant d'enregistrement doit être ajouté à deux endroits ; l'oubli de l'un donne un fichier que l'autre chemin aurait refusé.
- Touche : format
- Correction proposée : faire de `write_codes` un encodage des points en `RawMatrix` (indices, gains) puis un appel à `write_matrix_raw`, de sorte qu'un seul chemin valide et empaquette.
- Effort : M
- Risque de la correction : moyen. Touche l'écrivain du format ; les tests `raw_passthrough_is_byte_identical` et `the_default_writer_still_writes_v4` doivent rester verts, et la suite complète est due avant commit.
- Vérification proposée : suite `llvq-artifact` complète en release, dont les tests d'archive scellée.

### QUA-05 Les primitives d'entrée-sortie du format sont copiées entre `format.rs` et `sealed.rs`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-artifact/src/format.rs:517-554 et 1329-1335 ; llvq-artifact/src/sealed.rs:151-221
- Fait : `put_u32`, `put_u64`, `get_u32`, `get_u64`, `get_bytes` et `PREALLOC_CAP` sont écrits deux fois, à l'identique (le commentaire de `sealed.rs:182` parle de « twin »). `get_u16s` existe deux fois avec deux signatures (`n: u64` contre `n: usize`).
- Scénario : un correctif de robustesse sur `get_bytes` (le plafond d'allocation) doit être porté deux fois ; QUA-01 montre que `sealed.rs` a déjà pris du retard sur la rigueur de `format.rs`.
- Touche : format
- Correction proposée : un module privé `io.rs` dans `llvq-artifact` qui porte ces sept fonctions, importé par les deux fichiers.
- Effort : S
- Risque de la correction : faible. Déplacement de code privé, octets écrits inchangés.
- Vérification proposée : suite `llvq-artifact` ; `raw_passthrough_is_byte_identical`.

### QUA-06 1 324 lignes de fonctions identiques dans les exemples F1 de `llvq-bench`, alors que la bibliothèque les porte déjà
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `rho_of` ×7 (llvq-bench/src/f1/rankbook.rs:101, examples/f1enclazy.rs:163, f1encprof.rs:170, f1encrankscales.rs:180, f1encsolve.rs:153, f1rankbench.rs:133, f1rankenc.rs:108) ; `candidates_with` ×6, `nearest_constrained` ×6, `nearest` ×5, `encode_at_scale` ×4 (dont rankbook.rs:293) ; `solve` (87 lignes), `base_pair`, `build`, `end_entry` en double entre f1enclazy.rs et f1encreal.rs
- Fait : un balayage des corps de fonction (commentaires et blancs retirés, 12 lignes au moins) trouve 21 groupes de copies exactes, 1 324 lignes de trop. `llvq_bench::f1::rankbook` est un module public de la bibliothèque et contient déjà cinq de ces fonctions.
- Scénario : une correction du codeur F1 dans `rankbook.rs` ne touche pas les exemples qui en portent une copie, et deux exemples peuvent mesurer deux règles différentes sous le même nom.
- Touche : banc
- Correction proposée : si l'ancien audit clôt ces exemples derrière une feature `archive` (lot B1), rien d'autre ; sinon, remplacer les copies par `use llvq_bench::f1::rankbook::...`. Complète `docs/audit-architecture-2026-11.md` §2.6, qui les classe sans relever la copie.
- Effort : M
- Risque de la correction : faible. Exemples hors chemin servi ; `f1encreal` est cité par paper 2 (B7) et doit garder son chemin.
- Vérification proposée : aucune, lecture suffisante ; `cargo build --examples -p llvq-bench` après coup.

### QUA-07 `tetrascales`, classé mort, est la seule provenance des deux constantes du codeur Tetra servi
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-search/src/tetra/encoder.rs:51 et 361-385 ; llvq-bench/examples/tetrascales.rs ; docs/audit-architecture-2026-11.md:241
- Fait : `Encoder::ALPHA = 0.3218` et `RATIO = 1.14` sont « pinned [...] by `llvq-bench/examples/tetrascales.rs` on 2026-09-05 », et la sortie brute est dite dans « the session's scratchpad », donc hors dépôt. L'ancien audit liste `tetrascales` parmi les exemples morts à supprimer.
- Scénario : le lot de nettoyage supprime `tetrascales`, et les constantes du codeur de l'objet servi perdent leur programme de reproduction ; la sortie brute n'existe déjà plus de façon durable.
- Touche : doc
- Correction proposée : retirer `tetrascales` de la liste des morts et le mettre derrière la clôture `archive` comme B7 ; remplacer la mention du scratchpad par un journal dans `docs/mesures/` (le bloc de sortie est déjà recopié dans le commentaire).
- Effort : S
- Risque de la correction : nul. Doc seulement ; ne touche ni la constante ni le codeur.
- Vérification proposée : aucune, lecture suffisante.

### QUA-08 Deux variables pour trouver la même archive scellée, et `docs/ENV.md` décrit mal la seconde
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-bench/src/bin/radixstudy.rs:1519-1523 ; llvq-artifact/tests/common/mod.rs:18 et 51-58 ; docs/ENV.md:69
- Fait : les tests d'archive de `llvq-artifact` cherchent `$LLVQ_SEALED_ARTIFACT` puis `~/llvq-q4b.llvq` par un module commun écrit « so three copies of the message [...] cannot drift ». Les tests scellés de `radixstudy` réécrivent la recherche avec une autre variable, `LLVQ_Q4B`. `docs/ENV.md` décrit `LLVQ_Q4B` comme « int4 bit study switch » ; c'est un chemin.
- Scénario : un opérateur qui pose `LLVQ_SEALED_ARTIFACT` comme le dit CLAUDE.md voit les tests de `radixstudy` échouer sur `~/llvq-q4b.llvq` absent.
- Touche : outillage
- Correction proposée : lire `LLVQ_SEALED_ARTIFACT` dans `radixstudy` (en gardant `LLVQ_Q4B` comme alias refusé par nom, ou en le supprimant), et corriger la ligne de `docs/ENV.md`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### QUA-09 Quatre fonctions publiques sans appelant, dont une à la doc fausse
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : llvq-artifact/src/format.rs:1575 (`ArtifactWriter::with_version`) ; llvq-search/src/cns.rs:190 (`cns_layouts`) ; llvq-search/src/generic.rs:659 (`BallSearcher::nearest_ball13`) ; llvq-bench/src/f1.rs:106 (`TrioCode::natural_index`) ; llvq-quant/src/quantizer.rs:870 (`nearest_level`, privée, appelée par un seul `debug_assert`)
- Fait : `git grep -w` sur tout le dépôt ne trouve que la ligne de définition pour les quatre premières. La doc de `nearest_ball13` dit « ball Λ₂₄(13) ∪ {0} », mais la fonction ne compare que les sorties de `shell_bests`, qui portent les couches 2 à 13 (`shell: si as u32 + 2`) : l'origine n'est jamais candidate. `nearest_level` ne sert qu'à vérifier en debug qu'un centroïde est fini (quantizer.rs:530-536).
- Scénario : un lecteur prend `nearest_ball13` pour une recherche sur la boule avec origine.
- Touche : encodeur
- Correction proposée : supprimer les quatre fonctions ; remplacer le `debug_assert` par une vérification de finitude des centroïdes à la construction.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : `cargo clippy --all-targets` reste à zéro avertissement.

### QUA-10 `Header::is_self_contained` répond oui pour un fichier de projections seules
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : llvq-artifact/src/format.rs:1737-1739 et 1767-1769 ; format.rs:1725-1730 (`finish`) ; llvq-llm/src/dflash.rs:215
- Fait : la méthode rend `version >= 2`. `ArtifactWriter::new` écrit en v4 et `finish()` ferme avec zéro tenseur et zéro blob, ce que fait `smoke` (`llvq-llm/src/bin/smoke.rs:606`). Le fichier produit est un fichier de projections seules, et `is_self_contained()` rend vrai. La doc du champ dit « 1 for projections only; 2+ for a self-contained file ».
- Scénario : `dflash::sealed_q4_table` sur une sortie brute de `smoke` passe le garde « is not a sealed file » et échoue plus loin sur un tenseur absent, avec un autre message.
- Touche : format
- Correction proposée : renommer en `has_sealed_sections()` et corriger la doc, ou faire vérifier aux appelants que les sections ne sont pas vides.
- Effort : S
- Risque de la correction : faible ; renommage d'une API utilisée par cinq appelants.
- Vérification proposée : aucune, lecture suffisante.

### QUA-11 Commentaires qui contredisent le code
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : llvq-search/src/lib.rs:20-27 et 34-36 ; llvq-quant/src/gptq.rs:283-288 ; llvq-artifact/src/error.rs:13 ; llvq-search/src/pack.rs:44-55 ; llvq-quant/src/quantizer.rs:407-411
- Fait :
  - `llvq-search/src/lib.rs` décrit des tables sur « two 12-coordinate halves (4096 entries each) » ; le code utilise trois tranches de 8 coordonnées à 256 entrées (`CHUNK = 8`, `NCHUNK = 3`, lignes 89-91). Il annonce le moteur générique m ≤ 13 comme « Phase 2b » à venir ; il existe (`generic.rs`).
  - `gptq.rs:285` dit l'échelle de ligne « stored as a single f16 per row » ; le format la stocke en f64 (`format.rs:18-22`, `put_u64(s.to_bits())`).
  - `error.rs:13` : « does not begin with `crate::MAGIC` », qui vaut `LVQ4` seul ; le lecteur accepte cinq magies.
  - `BitWriter::push` dit paniquer si la valeur ne tient pas en `width` bits ; à `width == 0` il rend la main avant l'assertion et une valeur non nulle est perdue sans bruit.
  - La table de `LeechShapeGain` donne « 47 + k » bits par bloc pour la variante par défaut ; `LeechShapeGain::new` prend le plafond 13, soit 48 + k (`block_bits` calcule juste).
- Scénario : un lecteur qui se fie au commentaire se trompe sur la structure de la recherche, sur le coût des échelles ou sur un garde.
- Touche : doc
- Correction proposée : corriger les cinq textes ; pour `push`, déplacer l'assertion avant le retour anticipé (`assert!(width > 0 || value == 0)`).
- Effort : S
- Risque de la correction : nul pour les textes ; l'assertion de `push` ne change aucun octet d'un fichier valide.
- Vérification proposée : aucune, lecture suffisante.

### QUA-12 Conversions `as` qui tronquent sans bruit à l'écriture du format
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : llvq-artifact/src/format.rs:671, 673, 674, 679, 1410, 1708, 1713 ; llvq-artifact/src/sealed.rs:172, 231, 274, 275 (écriture) ; sealed.rs:311 (lecture, traité en QUA-01)
- Fait : longueurs de nom, `d_out`, `d_in`, nombre de centroïdes, groupe, nombre de tenseurs et de blobs passent de `usize` à `u32` par `as`. Aucune valeur réelle n'approche 2³², donc rien ne se déclenche aujourd'hui.
- Scénario : une valeur au-delà de 2³² serait écrite tronquée et relue comme une autre.
- Touche : format
- Correction proposée : `u32::try_from(..)` mappé en `Error::Inconsistent` dans `put_record_head`, `seal`, `put_str` et `write_raw`.
- Effort : S
- Risque de la correction : nul ; octets inchangés pour toute valeur légale.
- Vérification proposée : aucune, lecture suffisante. À transmettre au relecteur d'exactitude numérique.

### QUA-13 `rhoapply` jette l'en-tête de la source et lit la source à la version du fichier réécrit
- Gravité : cosmétique
- Confiance : PLAUSIBLE
- Où : llvq-bench/examples/rhoapply.rs:2056-2066 et 2202-2213 (aussi 1385)
- Fait : `let _ = read_header(&mut rs)` sur l'artefact source, puis `next_raw(&mut rs, h.version)` où `h` est l'en-tête du fichier réécrit.
- Scénario : si la réécriture change la version (v4 vers v5), chaque enregistrement source est lu avec le mauvais cadre, ce que le lecteur refuse ; si la version est la même, rien. Non confirmé que la réécriture puisse changer la version.
- Touche : banc
- Correction proposée : garder l'en-tête source et passer sa propre version à `next_raw`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### Constats d'anciens audits toujours présents à `925930d`
- CONNU-NON-CORRIGÉ : les six exemples morts `f1linear`, `f1mids`, `f1radii`, `f1rankenc`, `f1time`, `tetrascales` sont toujours dans `llvq-bench/examples/` (docs/audit-architecture-2026-11.md §2.6 ; nuance en QUA-07).
- CONNU-NON-CORRIGÉ : `llvq-cuda/Cargo.toml:18` dépend toujours de `llvq-bench` sans condition (docs/audit-architecture-2026-11.md, lot 7).
- CONNU-NON-CORRIGÉ : le code hôte Golay70 et Planes12x est toujours dans `llvq-artifact/src/runtime.rs:830-1645` (docs/audit-architecture-2026-11.md §2.4).
- CONNU-NON-CORRIGÉ : `rhoapply.rs` garde son SHA-256 écrit à la main (ligne 177) et sa fonction `run` de 1 020 lignes (docs/audit-architecture-2026-11.md §3.1, §3.2).

## Candidats au lot 1
QUA-07, QUA-08, QUA-09, QUA-11, QUA-13.

QUA-01, QUA-02, QUA-03, QUA-05 et QUA-12 sont d'effort S et sans risque pour les fichiers valides, mais ils touchent le lecteur ou l'écrivain du format : à passer par la suite complète, donc hors lot 1 selon la règle du brief.
