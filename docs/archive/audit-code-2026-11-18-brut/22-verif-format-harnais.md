# Vérification contradictoire : format, numérique, harnais

Vérificateur : lecture seule à `925930d`, aucun cargo, aucun uv, aucun réseau. Arithmétique refaite par `python3 -c`
sur les journaux `rtbits` et les dumps MMLU commités.

| ID | verdict | gravité retenue | raison |
|---|---|---|---|
| NUM-01 | NUANCÉ | majeure | Le pas de ligne manque bien au b/param publié, et la porte le compte déjà (2,550 / 2,377 / 2,524 b/weight imprimés = rtbits + pas, au millième). Mais +0,0073 b/param au 4B, aucune comparaison ne bascule : pas critique. Défaut déjà vu sans cause par l'audit papier du 2026-09-28, classé « nit », jamais corrigé. |
| NUM-02 | CONFIRMÉ | mineure | `rtbits` étiquette « f32 (published) » la ligne 2,7903 ; le papier publie la ligne « f16 (served) ». |
| NUM-03 | CONFIRMÉ | mineure | `CLAUDE.md:55` et `linalg.rs:15-19` disent « bit-identical » ; le test tolère 1e-8 relatif, et `faer` prend un autre chemin (inverse par LLT). |
| QUA-01 | CONFIRMÉ | mineure | Le plus exact des trois textes sur `read_raw` : aucune affirmation fausse, et il dit bien que le chargeur servi passe par ce lecteur. |
| NUM-04 | NUANCÉ | mineure | Défaut vrai, mais « les chemins servis sont protégés ailleurs » est faux : les normes passent par `to_f32` sans contrôle dans les trois chargeurs fusés. |
| SEC-06 | NUANCÉ | mineure | Le plus complet (allocation dictée par `dims`, épinglage sha256), mais il porte la même affirmation fausse sur les chemins servis. |
| QUA-02 | CONFIRMÉ | mineure | Gain non borné par le nombre de centroïdes ; Ball seulement, Tetra déjà refusé par `gain_width`. |
| NUM-05 | CONFIRMÉ | mineure | Même défaut que QUA-02, périmètre le plus exact (Ball seulement, aucun écrivain du dépôt ne le produit). |
| QUA-03 | CONFIRMÉ | mineure | Aucun des deux écrivains ne vérifie `tail.len()` ; le commentaire de `write_matrix_int4` le promet. |
| QUA-07 | CONFIRMÉ | mineure | Les constantes du codeur servi renvoient à `tetrascales`, que l'audit du 2026-11-07 range parmi les morts à supprimer ; la sortie brute n'est pas commitée. |
| QUA-10 | CONFIRMÉ | cosmétique | `is_self_contained` rend `version >= 2` ; un fichier `finish()` à sections vides répond oui. Il y a 11 appelants et 2 tests, pas 5. |
| BUG-01 | CONFIRMÉ | mineure | `max_by` rend le dernier maximum ; 4B publié 63,374 en « dernier », 63,381 en « premier », 40 égalités (recalculé). |
| BUG-02 | CONFIRMÉ | mineure | Sous la config chain, `bin/mmlu` score le préremplissage groupé et imprime « prefill chain ». Aucun chiffre publié. |
| BUG-03 | NUANCÉ | mineure | `prefill`, `config`, `prefill_attention` absents de `describe` : vrai. « max_new sans un mot » est faux : `describe` l'imprime. |
| BUG-05 | CONFIRMÉ | mineure | `bin/seal` prend le 0.6B par défaut et ne lit jamais le `model` du `.state`. |
| BUG-06 | CONFIRMÉ | mineure | `parse().ok().unwrap_or(défaut)` dans `mmlu` et `ppl` ; `limit = 0` imprime 0,00 % et des `NaN %`. |
| NUM-06 | NUANCÉ | mineure | Chemin exact, mais il faut un poids non fini dans le checkpoint, ce qu'aucun checkpoint Qwen n'a. |

## NUM-01 Le b/param publié omet le pas de ligne `tetra48`

Verdict : NUANCÉ. Le fait et les trois chiffres corrigés sont justes. La gravité « critique » est exagérée.

### Ce que `rtbits` compte

- `llvq-bench/src/bin/rtbits.rs:404-410` : chaque matrice Tetra compte `d_out · (d_in/24) · WORD_BITS` bits. C'est le
  flux dense du disque.
- `rtbits.rs:377-382` (`V5Rate::kernel_bpw`) : flux + tail (f32 ou f16) + `rows · 32` + bits int4. Aucun autre terme.
- `rtbits.rs:487-500` : les deux lignes du tableau « WHOLE MODEL » réutilisent `kernel_bpw(true)` et `kernel_bpw(false)`,
  plus l'embedding et les normes. Le pas n'apparaît dans aucune ligne ni aucun terme de la sortie v5. Les seules
  mentions de rembourrage dans le fichier (`:306-318`, `:1078`, `:1134`) concernent E1c et Planes12x, pas Tetra.

### Ce que la carte tient sous la config servie v9

- `llvq-artifact/src/tetra48.rs:55-63` : `stride_u32 = ceil(6·nblocks/8)·2`, rangée arrondie à 8 octets.
  `tetra48.rs:76-84` : `bits_per_weight` compte le pas et dit qu'un taux qui l'omet refait l'erreur du 2026-07-31.
- `llvq-llm/src/fused.rs:2294-2311` : le transcodeur produit `HostStream::Tetra48 { words, stride_u32 }` et compte
  `tb.data.len()`, pas compris, dans `bytes`.
- `llvq-llm/src/fused_cuda.rs:3613-3617` : ces mots strié sont téléversés tels quels (`DeviceStream::Tetra48`).
- `llvq-llm/src/persist_cuda.rs:122-144` : `tetra_of` et `seg_tetra` passent à 4E les mêmes `words` et `stride_u32`.
- `llvq-cuda/kernels/llvq_pers.cuh:1016-1021`, `:1373-1374` : le noyau 4E lit `words + r * stride_u32`. Il lit donc le
  même tampon strié, pas un autre agencement.
- `persist4e_cuda.rs:1054-1069` : sous v9, seuls les mots **int4** sont tuilés, en place, à taille égale
  (`fused_cuda.rs:2875-2895`, `q4_tile_words` est une permutation dans la ligne). Rien ne change pour les mots Tetra.
- Config chain (4B servi depuis le 2026-11-15) : son `note` dit l'arm `a12ipg32ft` « on the words 4E and the eager
  path hold, read in place ». Les copies tuilées de `install_tiled` (`persist4e_cuda.rs:1863-2060`) servent les arms
  de mesure ; sous l'arm `ip` elles sont construites pour un contrôle puis libérées (`:2059`). Pas de second
  agencement tenu.

Conclusion : la carte tient un seul agencement Tetra, le flux strié de `tetra48`. Le chiffre corrigé du relecteur
est le bon.

### Contre-épreuve décisive : la porte compte déjà le pas

La porte imprime ses octets de projections depuis `runtime_bytes_of` (`fused.rs:1382-1390`), qui somme les `bytes`
du transcodeur, pas compris. Ses lignes commitées :

- 4B : « projections 1.16 GB on the card, 2.550 b/weight » (`docs/mesures/served-v9-2026-11-07-brut/door.txt:7`).
- 8B : 2.377 ; 14B : 2.524 (`served-v9-8b-…-brut`, `served-v9-14b-…-brut`).

Recalcul (*computed*, `python3 -c`) depuis les comptes de `rtbits` et les formes Qwen3 :

| taille | lignes Tetra et pas | bits de pas | rtbits f16 b/weight | + pas | porte | b/param publié | + pas |
|---|---|---|---|---|---|---|---|
| 4B | 884 736 à d_in 2560 (4 o), 61 440 à 9728 (2 o) | 29 294 592 | 2,5420 | 2,5501 | 2,550 | 2,7320 | 2,7393 |
| 8B | 1 216 512 à 4096 (4 o), 77 824 à 12288 (0 o) | 38 928 384 | 2,3713 | 2,3769 | 2,377 | 2,6953 | 2,7001 |
| 14B | 1 638 400 à 5120, 107 520 à 17408 (2 o) | 27 934 720 | 2,5217 | 2,5238 | 2,524 | 2,7305 | 2,7324 |

Les décompositions sont uniques : elles retrouvent exactement les `tail` et `row scales` imprimés par `rtbits`
(`sealed-4b-2026-09-23-brut/rtbits-sealed.txt`, `sealed-8b-27-2026-09-24-brut/rtbits-8b-B.txt`,
`sealed-8b-14b-2026-09-23-brut/rtbits-14b.txt`). Les trois b/weight de la porte égalent rtbits + pas au millième. Le
relecteur a raison sur les six chiffres.

### La définition du dépôt

- `paper2/sections/model.tex:25-29` : « all the weight bits the served model keeps … at the widths the GPU holds ».
- `docs/format-noyau.md` §10 : « whole model | card bytes / all parameters ».
- `docs/ETAT.md` §2 : « b/param: *computed* by `rtbits`, whole model, int4 tables included ». Ne tranche pas la largeur.
- `docs/METHODE.md` ne contient pas de règle sur le pas. Son entrée du 2026-07-31 porte sur un banc de décodage.
  L'« erreur du 2026-07-31 » citée par `tetra48.rs:79` et `fused.rs:2300` est celle de `HISTORIQUE.md:19` : une
  magnitude non comptée.
- Aucun endroit n'écrit et n'assume le choix d'exclure le pas. Le seul texte contraire est la proposition de l'audit
  papier du 2026-09-28 (`docs/mesures/paper2-audit-2026-09-28-brut/audit-findings.json`, `results[4]/verification/
  missed[0]`) : il a vu l'écart 2,5420 / 2,550 et 2,739 b/param, sans en trouver la cause, l'a classé « nit », et a
  proposé d'écrire « Bits per parameter count the file ». Ce texte n'est pas dans le papier, qui garde « at the widths
  the GPU holds ». Le défaut est donc connu depuis le 2026-09-28 et non résolu. Le relecteur NUM-01 en donne la cause.

### Autres termes tenus par la carte et omis

Tous *estimated* sur le code, aucun ne change l'arrondi à quatre décimales sauf le dernier :
- mot de garde de `pack_plane_bytes` (`fused.rs:1999-2009`) : 4 o par matrice, 672 o au 4B ;
- deux centroïdes f32 par matrice : environ 1,3 Ko ;
- tables Tetra partagées (`fused_cuda.rs:151-157`, `:977-988`) : quelques Ko ;
- tables de rotation (`fused.rs:1401-1433`, `small` toujours 32×32 f32, `signbits` n/8 o) : 96 clés au 4B (la porte
  imprime 96 `rot_launches`), environ 445 Ko, soit 0,0009 b/param. La porte ne les compte pas non plus (2,550 et non
  2,551).

### Gravité

Le chiffre publié est faux à sa précision imprimée : quatre décimales dans `model.tex:88` pour les trois tailles, et
« 2,73 » devient « 2,74 » au 4B dans ETAT, README et `llvq-runtime`. L'écart est de 0,27 % au 4B. Aucune comparaison
publiée ne bascule : AWQ est à 5,30 / 5,96 / 5,40, IQ2_S à 2,82 / 2,80 / 2,69 (`docs/ETAT.md` §3), et le duel IQ2_S
se fait déjà en octets de fichier. Je retiens **majeure** : une contradiction entre la définition écrite du papier et
son chiffre, sans effet sur une conclusion.

### La correction

Celle du relecteur est juste : compter `d_out · stride_u32(d_in/24) · 32` bits par matrice Tetra, avec
`llvq_artifact::tetra48::stride_u32` (`rtbits` dépend déjà de `llvq-artifact`). Sa liste de sites oublie
`configs/qwen3-4b-tetra-e4-4e-v9.json`, `…-v9-chain.json`, `…-kv8-NOT-SERVED.json`, `configs/README.md`,
`llvq-runtime/library.json` et `docs/data/paper2-chain.csv` (que `make check` compare au tableau). Les preregs qui
citent 2,7320 ne se modifient pas.

Correction sûre pour un lot 1 ? non : elle change un chiffre publié dans le papier, ETAT, README et les configs servies.
Décision d'opérateur.

## NUM-02 « f32 (published) » n'est pas la ligne publiée

Verdict : CONFIRMÉ, mineure.
- `rtbits.rs:494` étiquette « f32 (published) » ; `:356-357` et `:470` disent « the accounting every published figure
  used ». Pour le b/param modèle entier, c'est faux : 2,7320 / 2,6953 / 2,7305 (`model.tex:88`) sont les lignes
  « f16 (served) ». 2,7903 n'est cité nulle part hors `docs/mesures/` (`git grep`).
- L'en-tête de `V5Rate` (`:357-359`) parle des b/weight noyau publiés avant Tetra (Planes14 4,804), où l'étiquette
  était vraie. Elle a été recopiée sur la ligne modèle entier.

Correction sûre pour un lot 1 ? oui : libellés d'un banc seulement, aucun chiffre ne bouge.

## NUM-03 « Bit-identique » entre `faer` et l'algèbre maison

Verdict : CONFIRMÉ, mineure.
- `CLAUDE.md:55` : « for a bit-identical result ». `linalg.rs:15-19` : « same mathematics to the last rounding » et
  « requires the two to produce the same factor ».
- `llvq-quant/tests/g5_gptq.rs:1015-1039` : tolérance `1e-8 · max|U|`.
- `linalg.rs:91-106` : `faer` calcule `llt`, puis `inverse()`, puis un second `llt`. Ce n'est pas l'ordre des boucles
  de `cholesky_lower` et du produit triangulaire maison. `Cargo.toml` de `llvq-quant` active `rayon`. L'égalité au bit
  n'a aucune raison de tenir. Que les codes divergent ensuite reste non montré.

Correction sûre pour un lot 1 ? oui pour les deux phrases de doc, sans effet sur le code.

## QUA-01, NUM-04, SEC-06 : `read_raw` ne valide pas un tenseur quantifié

Le défaut commun est vrai.
- `llvq-artifact/src/sealed.rs:305-324` : `bits` lu par `get_u32(..) as u8` (troncature), `group` accepté à 0,
  longueurs `packed`, `scales`, `biases` et F16 jamais comparées à `dims`.
- `sealed.rs:246-273` : `write_raw` fait ces contrôles.
- `sealed.rs:122-141` : `to_f32` fait `Vec::with_capacity(n)` avec `n = dims.product()` non vérifié, `div_ceil(q.group)`,
  indexe sans borne, et traite toute largeur autre que 8 comme 4 (`_ =>`, `:135`).

Atteinte, remontée jusqu'aux entrées :
- `llvq-llm/src/fused.rs:2547` : le chargeur fusé lit chaque tenseur porté par `read_raw`.
- L'embedding est retiré de `model.raw` (`fused_cuda.rs:3948-3951`) et validé par `upload_embed`
  (`fused_cuda.rs:1743-1751`). `embed_at` (`fused.rs:222`) laisse passer une largeur 0x104 lue 4.
- Tout le reste, c'est-à-dire les normes, passe par `to_f32` sans contrôle : `fused_cuda.rs:4035-4041`,
  `fused_cpu.rs:397-401`, `fused_metal.rs:2263`. Une norme `TAG_QUANT` à `group = 0` fait paniquer le chargeur servi.
- `llvq-llm/src/sealed.rs:548-549` (chemin dense, `mmlu`, `ppl`) et `hfpack.rs:346-348` paniquent de même.
- Atténuation réelle : le runtime vérifie le sha256 épinglé au téléchargement (`llvq-runtime/src/store.rs:263-267`,
  `:380-385`). Un fichier forgé exige donc un accès local ou un chemin passé à la main. Rust sûr : panique ou abandon,
  pas de corruption mémoire.

Lequel est le plus exact :
- **QUA-01** est le plus exact. Toutes ses affirmations tiennent, et il est le seul à dire que le chargeur servi passe
  par ce lecteur (`fused.rs:2547`). Il ne cite pas l'allocation par `dims`.
- **SEC-06** est le plus complet (allocation `dims = [2^40]`, `checked_mul`, sha256 épinglé). Mais « Les chemins servis
  valident déjà ces longueurs » est faux pour les normes.
- **NUM-04** a la même erreur (« Les chemins servis sont protégés ailleurs ») et cite `Q4Matrix::check` pour
  `fused_cpu.rs`, qui ne couvre que les enregistrements int4, pas les tenseurs portés (`fused_cpu.rs:400`).

Gravité retenue pour les trois : mineure. Correction sûre pour un lot 1 ? oui : refus à la lecture, aucun octet du
format, aucun fichier publié refusé. Elle touche `llvq-artifact` : suite complète en release obligatoire.

## QUA-02 et NUM-05 : gain non borné par les centroïdes

Verdict : CONFIRMÉ tous deux, mineure.
- `format.rs:1029-1030` : `n_cent` lu, `gain_width` (`:600-618`) donne `ceil(log2 n)`, 0 pour n = 0 sur Ball.
- `format.rs:1073-1076` : `gb` bits lus sans comparer à `n_cent`.
- `llvq-quant/src/quantizer.rs:966` : `centroids[code.gain as usize]` sans garde. Atteint par `decode_matrix`
  (`format.rs:1518`), appelé par `artifact2.rs:109` et `:485`.
- Tetra est protégé : `gain_width` exige `gb == 1`, donc n = 2 exactement (`:607-616`), et un gain d'un bit vaut 0 ou 1.
- Écrivains (`format.rs:752-765`, `:894-904`) : `BitWriter::push` (`llvq-search/src/pack.rs:48-58`) refuse une valeur
  plus large que `gb` bits, pas une valeur `>= n`. Avec n = 0, `push(.., 0)` sort sans rien écrire. QUA-02 a raison :
  le crate peut écrire ce fichier. NUM-05 a raison aussi : aucun appelant ne le fait.
- NUM-05 est le plus exact sur le périmètre. QUA-02 est plus complet sur la correction (côté écriture), mais refuser
  « n_cent == 0 pour Tetra » est déjà fait.

Correction sûre pour un lot 1 ? oui : refus `gain >= n_cent` à la lecture et à l'écriture, aucun fichier existant
touché. Dans `llvq-artifact` : suite complète en release.

## QUA-03 Les écrivains ne vérifient pas la longueur de la queue

Verdict : CONFIRMÉ, mineure.
- `format.rs:711-726` (`write_codes`) et `:874-891` (`code_stream`) : codes et `row_scales` contrôlés, `tail` non.
- `put_record_head` (`:688-690`) écrit toute la queue reçue ; `read_lattice_body` (`:1042-1046`) en lit exactement
  `d_out · (d_in % 24)`.
- `write_matrix_int4` (`:1338-1340`) promet que les invariants de `write_matrix_raw` sont tous contrôlés.
- Déclenchement : erreur de programmation seulement ; le quantificateur produit la bonne longueur.

Correction sûre pour un lot 1 ? oui, un contrôle dans les deux écrivains, aucun fichier valide refusé. Dans
`llvq-artifact` : suite complète en release.

## QUA-07 `tetrascales`, classé mort, est la provenance des constantes du codeur

Verdict : CONFIRMÉ, mineure.
- `llvq-search/src/tetra/encoder.rs:49-52` et `:361-385` : `ALPHA = 0.3218`, `RATIO = 1.14` « pinned … by
  `llvq-bench/examples/tetrascales.rs` », sortie brute « in the session's scratchpad ».
- `git ls-files | grep tetrascales` ne rend que l'exemple : la sortie n'est pas commitée.
- `docs/audit-architecture-2026-11.md:241` le range parmi les 9 morts, et `:425` prévoit leur suppression (étape 3).
- Atténuation : le bloc de sortie est recopié dans le commentaire, et `llvq-bench/tests/tetra_encoder.rs` garde la
  rétention 88,89 % en non-régression. Les constantes restent vérifiables, pas leur dérivation.

Correction sûre pour un lot 1 ? oui : doc seulement (retirer `tetrascales` de la liste des morts, verser la sortie
dans `docs/mesures/`). Ne pas toucher aux constantes.

## QUA-10 `is_self_contained` répond oui pour des sections vides

Verdict : CONFIRMÉ, cosmétique.
- `format.rs:1765-1769` : `version >= 2`. `finish()` (`:1724-1730`) écrit deux compteurs à zéro.
- Le commentaire (« Whether the file carries embeddings, norms and tokenizer ») est faux pour une sortie de `smoke`.
- La sémantique « version » est voulue et figée par deux tests : `llvq-artifact/tests/codebook_fingerprint.rs:796`
  et `tetra_format.rs:192`. 11 sites d'appel (`rtbits` ×2, `embedq`, `export`, `int4swap`, `seal`, `dflash`, `fused`,
  `hfpack` ×2, `sealed`), pas 5.

Correction sûre pour un lot 1 ? oui pour le commentaire. Non pour le renommage : API publique, 11 sites et deux
tests qui figent la sémantique.

## BUG-01 MMLU départage une égalité vers la dernière lettre

Verdict : CONFIRMÉ, mineure.
- `llvq-llm/src/bin/mmlu.rs:818-820` : `(0..4).max_by(|&x, &y| options[x].total_cmp(&options[y]))`.
- Sur une égalité stricte de f32, `total_cmp` rend `Equal` (mêmes bits), et `Iterator::max_by` garde l'élément le plus
  tardif : il rend le dernier des maxima, donc D, puis C, puis B, puis A. Seule nuance : `total_cmp` sépare `-0.0` de
  `+0.0`, sans effet ici.
- Recalcul (*computed*) depuis les logits du dump, que `dump_row` écrit par `{}` (représentation exacte) :
  `mmlu-4b-abl-control-FULL.csv`, 14 042 questions, 40 égalités, 63,374 en « dernier », 63,381 en « premier »,
  63,378 en crédit fractionnaire. FP16 : 4B 19 égalités, 70,140 contre 70,168 ; 8B 16, 75,053 contre 75,039 ;
  14B 17, 78,878 contre 78,899. Le `pick` du dump égale le dernier maximum sur toutes les lignes.
- Ni `docs/` ni `paper2/sections` ne disent suivre lm-eval-harness sur le départage. `mmlu.rs:35-37` ne s'y réfère que
  pour la moyenne micro.
- Écarts sous la barre (SE McNemar environ 0,4 pp), mais la règle pèse deux fois plus sur le bras quantifié.

Correction sûre pour un lot 1 ? non : le 4B publié passe de 63,37 à 63,38 sous la règle « premier » ou fractionnaire.
Décision d'opérateur, rescorable sans relance.

## BUG-02 `bin/mmlu` sous une config chain score le préremplissage groupé

Verdict : CONFIRMÉ, mineure.
- `mmlu.rs:548` imprime `cfg.provenance()`, qui ajoute « prefill chain (tree pass …) » (`served.rs:888-891`).
- `mmlu.rs:580` : `cfg.apply` ne fait que des contrôles d'objet sous chain (`served.rs:979-981`) puis `apply_form`,
  qui n'installe aucun décodeur ni passe d'arbre.
- `mmlu.rs:799-804` : `model.hidden` ; `model.rs:2963-2970` appelle `Block::forward`, qui bâtit `KvCache::new`
  (`model.rs:2125-2126`), store `Cat` (`model.rs:356-357`). Dans la branche `Kernel` (`model.rs:2220-2245`),
  `grow` est faux : tout passe par `grouped_attention`.
- La config chain sert le 4B (`docs/ETAT.md` §2). Aucun chiffre MMLU publié ne vient de ce chemin.

Correction sûre pour un lot 1 ? oui pour le refus par nom dans `bin/mmlu` : local, aucun token servi touché.

## BUG-03 `gsm8kpair` ne distingue pas deux préremplissages

Verdict : NUANCÉ, mineure.
- Vrai : `served_header` (`gsm8k.rs:343-365`) n'écrit pas `prefill_attention`. `describe` (`gsm8kpair.rs:38-63`)
  n'imprime ni `config`, ni `prefill`, ni `decode_cache`. Le `model` du dump ne porte pas la config
  (`bin/gsm8k.rs:157`).
- Faux : « un écart et un p de McNemar sans un mot » sur `max_new`. `describe` imprime `max_new` sur les lignes A et B
  (`gsm8kpair.rs:42-49`). Il n'y a pas d'avertissement, mais la valeur est visible.

Correction sûre pour un lot 1 ? oui : impression et clé d'en-tête optionnelle, les dumps anciens se lisent pareil.

## BUG-05 `bin/seal` prend le 0.6B par défaut

Verdict : CONFIRMÉ, mineure.
- `llvq-llm/src/bin/seal.rs:53` : `unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into())`.
- `seal.rs:88-122` : copie tout tenseur absent de l'artefact, sans comparer noms ni formes. `.state` jamais lu.
- `bin/smoke.rs:853` écrit bien `model` dans `.state`.
- Un checkpoint de mêmes formes (variante Base ou 2507) produit un fichier qui charge sans message.

Correction sûre pour un lot 1 ? oui : refus de `LLVQ_MODEL` absent et d'un `.state` qui nomme un autre modèle.
Aucun fichier scellé ne bouge.

## BUG-06 Positionnels lus avec `.parse().ok().unwrap_or(défaut)`

Verdict : CONFIRMÉ, mineure.
- `mmlu.rs:495` (`limit`), `ppl.rs:40-41` (`ctx`, `max_windows`), `ppl.rs:48` et `:120-124` (tout corpus autre que
  `c4` devient WikiText-2, ce que le commentaire `:46-47` assume).
- `limit = 0` : `micro` rend 0 (`mmlu.rs:369-379`, `rate` à 0 quand `scored == 0`), la ligne par sujet imprime
  `NaN %` (`mmlu.rs:849`).

Correction sûre pour un lot 1 ? oui : binaires de banc, une commande valide fait la même chose.

## NUM-06 NaN dans un bloc

Verdict : NUANCÉ, mineure.
- Vrai : `encoder.rs:892-899`, `f64::max` ignore NaN. Un NaN isolé donne `s0 = NaN`, `is_normal()` faux, branche
  `xs`, puis l'`assert!` de `encode_at_scale` (`:852`) panique. Un bloc tout NaN donne `m = 0` et le mot 0.
- `quantizer.rs:781` : `norm == 0.0` faux pour NaN ; le `debug_assert!` de `direction` (`:750-754`) est muet en
  release. `set_row_scale(NaN)` retombe sur 1,0 (`:764`), alors que `calib.rs:1454-1461` stocke `row_scale` brut.
- Aucune garde de finitude des poids en amont (`git grep is_finite` dans `calib.rs`, `gptq.rs`, `smoke.rs`). La
  Cholesky refuse un NaN dans la hessienne (`linalg.rs:121`), pas dans les poids.
- Nuance : il faut un checkpoint à poids non finis. Les poids Qwen publiés sont finis, et la rotation d'un fini reste
  finie. La cascade GPTQ décrite est plausible, non vérifiée.

Correction sûre pour un lot 1 ? oui si le refus est posé dans `llvq-quant` (`TetraShapeGain::quantize` ou
`quantize_layer_capturing`) : aucun code ne change pour une entrée finie. Posé dans `llvq-search`, il exige la suite
complète en release.
