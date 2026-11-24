# Exactitude numérique : constats
Périmètre réellement lu : en entier `llvq-core/src/{leech,shells,golay}.rs`, `llvq-search/src/{index,classes,pack}.rs`, `llvq-search/src/tetra/{mod,word,rank,encoder}.rs`, `llvq-quant/src/{quantizer,gptq,linalg,rotation}.rs`, `llvq-artifact/src/{sealed,tetra48,codebook}.rs`, `llvq-artifact/src/format.rs` (lignes 76 à 1969), `llvq-llm/src/{tetra_cpu,embedquant}.rs`, `llvq-llm/src/bin/rowscale.rs`, `llvq-tetra/llvq_tetra/{dequant,tetra,reader,modules,fused,conversions}.py`. En partie : `llvq-search/src/tetra/trellis.rs` (1-200), `llvq-search/src/rankdec.rs` (1-240), `llvq-search/src/lib.rs` (100-300), `llvq-search/src/generic.rs` (860-953), `llvq-quant/src/schur.rs` (en-tête), `llvq-artifact/src/runtime.rs` (555-640 et balayage des conversions), `llvq-artifact/src/blockrec.rs` (en-tête), `llvq-bench/src/bin/rtbits.rs` (1-120, 284-530), `llvq-llm/src/sealed.rs` (325-622), `llvq-llm/src/kvq.rs` (1-340), `llvq-llm/src/hfpack.rs` (90-590), `llvq-llm/src/bin/{int4swap,embedq}.rs` (en-têtes), `llvq-llm/src/fused.rs` (195-255, 1285-1470, 1999-2010, 2290-2345, 2470-2530), `llvq-llm/src/calib.rs` (1380-1580), `llvq-llm/src/fused_cpu.rs` (250-340), `llvq-llm/src/artifact2.rs` (`shard_extent`), `llvq-tetra/llvq_tetra/{metal,integration}.py` (extraits), `llvq-quant/tests/g5_gptq.rs` (`both_factorizations_agree`), les sorties `rtbits` des trois fichiers scellés dans `docs/mesures/`, `paper2/sections/model.tex` (1-100).
Non couvert : `llvq-search/src/{fastdec,cns,generic_ref}.rs`, l'essentiel de `generic.rs` et `lib.rs` (chercheur Ball, non servi), `llvq-artifact/src/{e1c,e1v}.rs`, le reste de `runtime.rs` (couches Ball, couvertes par `published_4b_kernel_rates_are_reproduced`), `llvq-llm/src/kvladder.rs`, `llvq-tetra/llvq_tetra/{mps,checkq4,checkq4guards,vocab}.py`, les tests d'intégration (sauf un), les noyaux CUDA et Metal (hors périmètre), le code de `faer` (hors racine).
Bilan : 1 critiques, 0 majeures, 5 mineures, 1 cosmétiques

Vérifié propre, pour ne pas le relire : la bijection Tetra (champs, bornes `CLASS_BOUNDS`/`MIXED_BOUND`, `i2` aux deux côtés de `N0_MIXED`, mot 0 et `2⁴⁷ − 1`), `BitWriter`/`BitReader` à cheval sur un mot, `stride_u32` et la fenêtre de lecture du dernier bloc, `word_at` du CPU, les multinomiales en `u128` (24! < 2⁸⁰), les soustractions non signées de l'encodeur (`cost − w`), l'arrondi du tail en f32 avant la dérotation (`calib.rs:1505-1513`), et la parité Python/Rust de `dequantize_lattice` (mêmes opérations, même ordre de sommation dans `mix`, pas de FMA de part et d'autre, `np.sqrt` correctement arrondi).

### NUM-01 Le b/param publié omet le pas de ligne `tetra48` que la carte tient : 4B 2,7320 au lieu de 2,7393
- Gravité : critique
- Confiance : CONFIRMÉ
- Où : `llvq-bench/src/bin/rtbits.rs:404-410` ; `llvq-artifact/src/tetra48.rs:55-63` et `:76-84` ; `llvq-llm/src/fused.rs:2294-2306` ; `paper2/sections/model.tex:25-29` et `:88`
- Fait : `report_v5` compte chaque matrice Tetra à `d_out · (d_in/24) · 48` bits, le flux dense du disque. La carte tient le flux `tetra48`, dont chaque ligne est arrondie à 8 octets (`stride_u32`). Le chargeur l'écrit lui-même : « The pad IS counted in `bytes` here ... a rate that dropped it would be the 2026-07-31 accounting error in a new place » (`fused.rs:2296-2300`), et `tetra48.rs:76-81` dit la même chose. Le papier définit le b/param comme « all the weight bits the served model keeps ... at the widths the GPU holds » (`model.tex:25-28`). La ligne publiée, « f16 (served) », prend le tail et les échelles à la largeur de la carte, mais le flux à la largeur du disque.
- Scénario : pas par ligne à `d_in` = 2560 ou 4096, 4 octets ; à 9728, 5120, 17408, 2 octets ; à 12288, 0. Depuis les comptes de `rtbits-sealed.txt` (tail 14 647 296, 946 176 lignes), le 4B a 884 736 lignes à 4 octets et 61 440 à 2 octets, soit 29 294 592 bits (*computed*). Sur 4 022 468 096 paramètres : +0,00728 b/param, 2,7320 devient 2,7393, et le « 2,73 » d'ETAT, du README et du papier devient 2,74. Le b/weight noyau passe de 2,5420 à 2,5501, ce que la porte imprime déjà (« projections 1.16 GB on the card, 2.550 b/weight », cité dans `paper2-audit-2026-09-28-brut/audit-findings.json:1857`). 8B : 1 216 512 lignes à 4 octets, 2,6953 devient 2,7001. 14B : 1 745 920 lignes à 2 octets, 2,7305 devient 2,7324. Les deux arrondis à deux décimales du 8B et du 14B ne bougent pas, les quatre décimales du tableau du papier bougent toutes les trois. Ce n'est pas non plus la comptabilité fichier : le fichier 4B fait 2,8206 b/param (1 418 224 685 octets). Termes voisins non comptés, plus petits : le mot de garde de `pack_plane_bytes` (4 octets par matrice) et les deux centroïdes f32 par matrice, négligeables ; les tables de rotation (`signbits` et `small` 32×32 f32 par clé, au plus 144 clés au 4B, ≤ 0,0015 b/param, *estimated*, non vérifié sur le chemin 4E).
- Touche : doc (chiffre publié), banc
- Correction proposée : dans `report_v5`, compter `d_out · stride_u32(d_in/24) · 32` bits par matrice Tetra, ou imprimer les deux lignes (flux disque, flux carte) et dire laquelle le papier cite. Ajouter un test qui pose les comptes du 4B et attend 2,5501 b/weight, le chiffre de la porte.
- Effort : S pour le code ; la republication des chiffres est une décision d'opérateur.
- Risque de la correction : faible pour le code. Elle change un chiffre publié : `paper2/sections/model.tex:88`, `paper2/sections/experiments.tex:530`, `docs/ETAT.md` (4B 2,73 vers 2,74), `README.md:25`, `llvq-runtime/README.md:17`, `docs/CERVEAU.md`, et les constantes `2.7320` de `ops/dflash_brouillon_int4_report.py:15` et `ops/spec_sans_poids_report.py:14`. Aucun format, index, noyau épinglé ni token servi.
- Vérification proposée : relancer `rtbits` sur `qwen3-4b-sealed.bin` après la correction et comparer au « 2.550 b/weight » de la porte ; `python3 -c` sur les comptes ci-dessus suffit à reproduire 2,7393 / 2,7001 / 2,7324.

### NUM-02 `rtbits` étiquette « published » la ligne que le papier ne publie pas
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-bench/src/bin/rtbits.rs:356-359`, `:469-476`, `:494`
- Fait : la sortie appelle la ligne au tail f32 « f32 (published) » et « the accounting every published figure used ». Les b/param publiés des trois fichiers scellés sont ceux de la ligne « f16 (served) » : 2,7320 / 2,6953 / 2,7305 (`paper2/sections/model.tex:88`, audit `paper2-audit-2026-09-28-brut/audit-findings.json:1731`). La ligne étiquetée « published » donne 2,7903 / 2,7333 / 2,7456.
- Scénario : un lecteur qui reprend « le chiffre publié » d'une sortie `rtbits` cite 2,7903 pour le 4B, soit +0,058 b/param au-dessus du papier.
- Touche : banc, doc
- Correction proposée : renommer les deux lignes par ce qu'elles comptent (« tail f32, fichier » et « tail f16, carte ») et dire dans le commentaire laquelle chaque document cite.
- Effort : S
- Risque de la correction : nul. Aucun chiffre ne bouge, seulement des libellés ; les journaux bruts déjà commités gardent l'ancien libellé.
- Vérification proposée : aucune, lecture suffisante.

### NUM-03 « Bit-identique » entre `faer` et l'algèbre maison : le test n'exige que 1e-8
- Gravité : mineure
- Confiance : CONFIRMÉ pour la contradiction ; PLAUSIBLE pour l'écart effectif des codes
- Où : `CLAUDE.md:55` ; `llvq-quant/src/linalg.rs:15-19` ; `llvq-quant/tests/g5_gptq.rs:1020-1039`
- Fait : `CLAUDE.md` dit que sans `fast-linalg` la factorisation est 40 fois plus lente « for a bit-identical result », et `linalg.rs` dit « same mathematics to the last rounding » et que `both_factorizations_agree` « requires the two to produce the same factor ». Le test accepte `|Δ| ≤ 1e-8 · max|U|`. `faer` est bloqué, vectorisé et parallèle (`Cargo.toml`, feature `rayon`) ; l'ordre de ses sommes n'est pas celui des boucles de `cholesky_lower`.
- Scénario : un `smoke` lancé sans `--features fast-linalg` pour reproduire un fichier scellé obtient un facteur `U` différent au dernier bit ; GPTQ propage cette différence et les codes peuvent diverger. Rien ne garantit non plus que `faer` rende le même `U` sur deux CPU (largeur SIMD, tailles de cache), donc la reproductibilité au bit d'un fichier scellé d'une machine à l'autre n'est pas établie.
- Touche : encodeur, doc
- Correction proposée : corriger les deux phrases (« accord à 1e-8 relatif, pas au bit ; les codes peuvent différer ») ; si la reproductibilité au bit est voulue, l'écrire comme exigence et la tester sur un petit `smoke` dans les deux configurations.
- Effort : S (doc)
- Risque de la correction : nul pour la doc.
- Vérification proposée : un test local qui compare `GptqFactor::new` et `new_reference` avec `assert_eq!` sur les bits, sous `--features fast-linalg`, à n = 129 ; il devrait échouer.

### NUM-04 `read_raw` n'oppose aucune borne à un tenseur porté quantifié ; `to_f32` panique ou lit faux
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-artifact/src/sealed.rs:310-325` (lecture), `:122-141` (`to_f32`), `:248-273` (le contrôle n'existe qu'à l'écriture) ; appelants `llvq-llm/src/sealed.rs:548-549`, `llvq-llm/src/hfpack.rs:346-348`
- Fait : `read_raw` lit `bits` comme `u32` puis le tronque en `u8` (`as u8`), lit `group`, `packed`, `scales`, `biases` sans les comparer à `dims`. `write_raw` fait ces contrôles, pas la lecture. `to_f32` divise par `q.group` (`div_ceil`), indexe `packed`, `scales` et `biases` sans borne, et traite toute largeur autre que 8 comme 4 bits (`_ =>`). `len()` est un `product()` non vérifié.
- Scénario : un fichier corrompu ou fabriqué avec `group = 0` fait paniquer `sealed::load` (« attempt to divide by zero ») et `hfpack`. Un `bits` stocké à 260 se lit 4. Un `bits = 2` avec un `packed` de taille int4 se décode en silence comme int4. Les chemins servis sont protégés ailleurs : `fused_cuda.rs:1743` contrôle les longueurs, `fused_cpu.rs` passe par `Q4Matrix::check`. Ce sont donc le chemin dense (MMLU, ppl) et `hfpack` qui paniquent au lieu de rendre `Err`, contre la règle de `tests/hostile_files.rs`.
- Touche : format (lecteur)
- Correction proposée : dans `read_raw`, refuser `bits ∉ {4, 8}` avant la troncature, `group = 0`, un `product()` qui déborde, et des longueurs `packed`/`scales`/`biases` différentes de ce que `write_raw` exige.
- Effort : S
- Risque de la correction : faible. Aucun octet du format ne change ; un fichier valide est toujours lu.
- Vérification proposée : un test dans `hostile_files.rs` qui écrit un tenseur porté `TAG_QUANT` avec `group = 0` puis appelle `read_raw(..).to_f32()` ; il panique aujourd'hui.

### NUM-05 Un enregistrement Ball peut porter un gain hors des centroïdes ; `decode_matrix` panique
- Gravité : mineure
- Confiance : CONFIRMÉ (fichier corrompu seulement)
- Où : `llvq-artifact/src/format.rs:600-618` (`gain_width`), `:1426-1437` (`decode_codes`) ; `llvq-quant/src/quantizer.rs:966`
- Fait : pour Ball, `gain_width` accepte tout nombre de centroïdes, 0 compris (`0usize.next_power_of_two()` vaut 1, largeur 0). Le champ de gain fait `⌈log₂ n⌉` bits et rien ne vérifie `gain < n`. `reconstruct_shape_gain` indexe `centroids[code.gain]`.
- Scénario : n = 3 donne un champ de 2 bits, un gain 3 lit `centroids[3]` et panique. n = 0 donne un champ vide, et le premier bloc non nul panique sur `centroids[0]`. Aucun écrivain du dépôt ne produit ces fichiers (`fit_gain_centroids` rend toujours 2^k niveaux) ; Tetra est protégé (`gb == 1` exige n = 2).
- Touche : format (lecteur), Ball seulement
- Correction proposée : refuser à la lecture un nombre de centroïdes nul ou qui n'est pas une puissance de deux, ou refuser `gain >= n` dans `decode_codes` avec une erreur nommée.
- Effort : S
- Risque de la correction : faible. Aucun fichier écrit par ce dépôt n'est refusé.
- Vérification proposée : un test `hostile_files.rs` : enregistrement Ball à 3 centroïdes et un gain 3, attendre `Err`, pas une panique.

### NUM-06 NaN dans un bloc : un NaN fait paniquer l'encodeur Tetra, un bloc tout NaN devient l'origine en silence
- Gravité : mineure
- Confiance : PLAUSIBLE (il faut un poids NaN dans la matrice tournée)
- Où : `llvq-search/src/tetra/encoder.rs:876-899` ; `llvq-quant/src/quantizer.rs:748-755`, `:763-765`, `:778-790` ; `llvq-llm/src/calib.rs:1454-1461`
- Fait : `lower_scale` prend `m` par `f64::max`, qui ignore NaN. Un bloc avec un seul NaN donne `s0 = NaN`, puis `encode_at_scale` panique sur son `assert!` (`:852`). Un bloc entièrement NaN donne `m = 0` et `encode` rend l'origine, mot 0. `TetraShapeGain::quantize` ne l'arrête pas (`norm == 0.0` est faux pour NaN) ; le `debug_assert!` de `direction` ne vit qu'en debug. En release le bloc est écrit à zéro, l'erreur NaN est propagée aux colonnes suivantes. Par ailleurs `set_row_scale(NaN)` retombe sur 1,0 dans le quantificateur, alors que `calib.rs` stocke la valeur NaN calculée par `row_scale` : le modèle évalué et le fichier divergent sur cette ligne.
- Scénario : un point de contrôle avec un bloc de 24 poids NaN (ou une ligne NaN) se scelle sans erreur en release ; un seul NaN isolé fait tomber le run. Aucun fichier publié n'est concerné à notre connaissance.
- Touche : encodeur
- Correction proposée : refuser par nom tout poids non fini au début de `TetraShapeGain::quantize` (ou de `quantize_layer_capturing`), en release.
- Effort : S
- Risque de la correction : faible. Pour une entrée finie aucun code ne change ; ni format, ni index, ni tokens.
- Vérification proposée : un test qui passe `[f64::NAN; 24]` à `TetraShapeGain::quantize` en release et attend une panique nommée ; il rend aujourd'hui un bloc nul.

### NUM-07 Les commentaires disent l'échelle de ligne stockée en f16
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-quant/src/gptq.rs:283-288` ; `llvq-quant/src/quantizer.rs:40-47`
- Fait : `gptq.rs` écrit « It is stored as a single f16 per row », et `set_row_scale` compte « `16/d_in` bits per weight ». Le fichier stocke l'échelle en f64 (`format.rs:18-22`, `:685-687`) et la carte en f32 (`fused.rs:1298-1304`, `ROW_SCALE_BYTES = 4`). `rtbits` la compte à 32 bits.
- Scénario : un lecteur qui refait la comptabilité depuis ces commentaires compte 16 bits par ligne au lieu de 32 (carte) ou 64 (fichier).
- Touche : doc
- Correction proposée : écrire « f64 dans le fichier, f32 sur la carte ».
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

## Candidats au lot 1
NUM-02, NUM-03, NUM-04, NUM-05, NUM-06, NUM-07. NUM-04 et NUM-05 ajoutent des refus au lecteur sans changer un octet du format. NUM-01 n'y est pas : sa correction change un chiffre publié.
