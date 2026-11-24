# Revue du code neuf, format, encodeur, Python et outillage (R2-DIF) : constats
Périmètre réellement lu : diffs entiers et fonctions entières de `llvq-artifact/src/sealed.rs`, `format.rs` (`u32_field`, `check_tail`, `put_record_head`, `write_codes`, `code_stream`, `write_matrix_raw`, `read_lattice_body`, `seal`), `error.rs`, `llvq-artifact/tests/hostile_files.rs` (partie neuve) ; `llvq-quant/src/gptq.rs` (propagation), `linalg.rs` (`reference_factor`, `fast::factor`, `u_row`), `quantizer.rs` (`TetraShapeGain`, `LeechShapeGain`, `fit_gain_centroids`, `nearest_level_index`) ; `llvq-quant/tests/gptq_row_slice.rs` ; `llvq-bench/src/bin/rtbits.rs` (diff et test) ; les cinq `Cargo.toml` ; `.github/workflows/ci.yml` ; `ops/run.py` (`cap_ok`, `device_ok`, `cmd_launch`, `cmd_publish`, `cmd_oracle`, `cmd_bench`, `cmd_dequant`, `_is_runner_image`, `_hf_token_secret`, `_timeout_minutes`) ; `ops/hf_publish_prep.py`, `ops/hf_cleanroom_check.py`, `ops/llvqtune` (diff, `train.sh` l. 95-140, `loop.py`, `jsonl_recorder.py`) ; `llvq-tetra` (diff entier, `fused.py`, `integration.py`, `quantizer.py`, `reader.py`, `metal.py` l. 100-160, `chat.py`, tests neufs `test_fused_device.py`, `test_checks.py`, partie neuve de `test_quantizer.py`) ; les écrivains `bin/seal.rs` (raws), `bin/embedq.rs`, `bin/int4swap.rs`, `bin/rowscale.rs`, `embedquant.rs::quantize_affine`, `calib.rs` (queue), `hfpack.rs` (config), l'historique `git log -p` de `sealed.rs` (905513c, 587a479) et la première version de `format.rs` (1076264) ; dans transformers 5.18.0 du `.venv` de `llvq-tetra` : `integrations/accelerate.py:96-140`, `modeling_utils.py:248-260, 4185-4365`, `quantizers/auto.py:360-390` ; `huggingface_hub` 2.4.0 en cache (`_jobs_api.py`).
Survolé : `docs/ENV.md`, `docs/ETAT.md`, `docs/audit-architecture-2026-11.md`, `llvq-runtime/README.md`, `configs/README.md`, `llvq-search/src/lib.rs`, `pack.rs`, `llvq-cuda/src/{modcache,pers,rowcand}.rs`, `llvq-cuda/tests/golay70_decoder_matches_rust.rs` (diffs de commentaires, contrôlés contre le code cité).
Non couvert : `llvq-llm` (autre relecteur), la table de `configs/README.md` ligne par ligne contre les journaux, `checkmps.py` au-delà des trois `assert` convertis, les 163 scripts de job au-delà de la forme de leur `--image`.
Bilan : 0 critiques, 0 majeures, 6 mineures, 1 cosmétique

## Constats

### R2-DIF-40 `device_map="cuda"` après `torch.cuda.set_device(k)` est refusé, et le message de refus conseille exactement ce geste
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-tetra/llvq_tetra/integration.py:64-65, 69-92 ; llvq-tetra/tests/test_quantizer.py:101-113 ; transformers 5.18.0 `integrations/accelerate.py:115-119`, appelé à `modeling_utils.py:4193`, avant `get_hf_quantizer`, donc avant `validate_environment` et `update_device_map` (`quantizers/auto.py:378-382`).
- Fait : transformers réécrit `device_map="cuda"` en `{"": torch.device(f"cuda:{LOCAL_RANK}")}`, `LOCAL_RANK` valant 0 s'il n'est pas posé, avant que le quantificateur ne le voie. `target_device` reçoit donc `cuda:0`, jamais `"cuda"` nu. `_require_current_cuda` compare cet index 0 à `torch.cuda.current_device()`.
- Scénario : machine à deux cartes, pas de `LOCAL_RANK`. L'utilisateur fait `torch.cuda.set_device(1)` puis `from_pretrained(name, device_map="cuda")`. Le quantificateur voit `cuda:0`, la carte courante est 1 : refus « names device(type='cuda', index=0). Select that card with torch.cuda.set_device or CUDA_VISIBLE_DEVICES and pass device_map="cuda" ». Suivre la moitié `set_device` du conseil redonne le même refus. À `925930d` ce geste chargeait sur la carte 1 (`finalize` fait `model.to("cuda")`, l'appareil courant, et les tampons fusés aussi). Le cas `"cuda"` de `test_the_current_card_is_accepted` appelle `target_device("cuda")` directement, une forme que transformers ne transmet jamais : le test ne peut pas voir ce chemin. `CUDA_VISIBLE_DEVICES=1` marche (la carte visible 0 est courante).
- Touche : outillage (paquet `llvq-tetra`, part à la prochaine version)
- Correction proposée : minimale, dans `_require_current_cuda` : retirer « torch.cuda.set_device or » du message et dire que transformers lit `"cuda"` comme `cuda:$LOCAL_RANK` ; ajouter un test qui passe `{"": torch.device("cuda", 0)}` avec `torch.cuda.is_available` et `torch.cuda.current_device` remplacés par `monkeypatch` (True, 1) et attend le refus sans `set_device` dans le message. Complète, effort M : porter l'index jusqu'au bout (`finalize` et `_compress` sous `with torch.cuda.device(index)`, `model.to(torch.device("cuda", index))`, caches `metal.shared_tables` et `invnorm_on` indexés par `str(x.device)` au lieu du genre), ce qui rend le geste `set_device` et `cuda:1` tous deux justes au lieu de refusés.
- Effort : S (minimale), M (complète)
- Risque de la correction : nul pour la minimale. Aucun noyau, aucun format, aucun token servi ; le bras transformers seulement.
- Vérification proposée : le test monkeypatché ci-dessus, rouge sur le message actuel (il contient `set_device`), vert après ; mutant : remettre « torch.cuda.set_device or » dans le message, rouge.

### R2-DIF-41 `run.py oracle` refuse l'image sm80 sur `a100-large`, que `bench` accepte par `--kernels-cap`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/run.py:1016-1021 (`cmd_oracle` appelle `cap_ok`), :1628-1635 (arguments d'`oracle`, sans `--kernels-cap`) ; à comparer à `cmd_bench` :1082-1096 et :1641-1645.
- Fait : `cap_ok` juge contre `MIN_COMPUTE_CAP = 89`, la cible de l'image standard. `bench` a l'échappatoire `--kernels-cap`, dont l'aide nomme « the sm80 image ». `oracle` n'en a aucune.
- Scénario : `uv run ops/run.py oracle --image hf.co/spaces/Pier-Jean/llvq-runner-cuda-sm80 --flavor a100-large` rend 2 avec « refused, a100-large: sm_80 < sm_89, the image can load no kernel there ». Le message est faux pour cette image : elle existe (`publish --compute-cap 80`, `docs/mesures/a100-2026-10-06.txt:139`) et `ops/jobs/a100-runner.sh:10-15, 88-90` y fait tourner `oracle` sur A100. À `925930d` la même commande partait. La règle dure 10 demande `oracle` d'abord sur chaque backend.
- Touche : outillage
- Correction proposée : `o.add_argument("--kernels-cap", type=int, default=None, metavar="N")` et, dans `cmd_oracle`, la même branche que `cmd_bench` :1084-1093 (`card = FLAVORS[...]["cap"]`, `usable = card is None or card >= args.kernels_cap`, ligne imprimée). Le refus par défaut reste.
- Effort : S
- Risque de la correction : nul. Outillage seulement.
- Vérification proposée : le banc sans coût de `43-lot4-outillage.md` (`run_job` remplacé), cas `["oracle", "--image", "x", "--flavor", "a100-large", "--kernels-cap", "80"]` : ligne de coût puis `would launch` ; sans `--kernels-cap` : refus, code 2. Mutant : supprimer la branche `kernels_cap` dans `cmd_oracle`, le premier cas rend 2.

### R2-DIF-42 La ligne carte de `rtbits` n'a aucun test qui passe par `report_v5`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-bench/src/bin/rtbits.rs:440-446 (la branche `if m.kind == CodeKind::Tetra`), :488-494 (la ligne imprimée), :1675 (`the_card_stream_adds_the_tetra48_row_pad_to_the_file_stream`) ; llvq-bench/tests/tetra_refusals.rs:63-93 ; llvq-bench/tests/int4_refusals.rs:96-125.
- Fait : le test unitaire construit `V5Rate` à la main et calcule lui-même `lattice_card_stream_bits`, une copie de l'accumulation de `report_v5`. Il vérifie `tetra_card_stream_bits` et `card_kernel_bpw`, pas la branche qui choisit entre compte carte et compte fichier. Les deux tests d'intégration lancent `rtbits` sur un fichier v5 et ne lisent que « lattice matrices », « kernel b/weight » et « NO CLASS STATISTIC IS PRINTED ».
- Scénario : mutant `m.kind == ...Tetra` en `m.kind != ...Tetra`, ou `+= file_bits` sans condition. La ligne carte imprime « 0 pad bits » et le b/weight carte égale le b/weight fichier. Toute la suite reste verte. C'est la ligne que la décision NUM-01 propose de republier (2,7393 au 4B).
- Touche : banc (instrument d'un chiffre publiable)
- Correction proposée : dans `rtbits_prices_a_tetra_file_instead_of_refusing_it`, ajouter `assert!(text.contains("32 pad bits the file does not store"), ...)`. Le fichier du test a `d_out = 1`, `d_in = 48` : `stride_u32(2) = 4` u32, soit 128 bits contre 96 dans le fichier (*computed*).
- Effort : S
- Risque de la correction : nul. Test seulement, aucune ligne imprimée ne change.
- Vérification proposée : le mutant `!=` ci-dessus, rouge sur la nouvelle assertion, vert une fois défait.

### R2-DIF-43 La garde `MAX_FIRST_KL` de `train.sh` reste ouverte sur une exception Python
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/llvqtune/train.sh:131-137
- Fait : la correction PY-16 code « passe » par la sortie 1 et « refuse » par la sortie 0. Une exception non rattrapée dans `python -c` sort aussi en 1, donc se lit « passe ». `float('$MAX_FIRST_KL')` lève sur une valeur mal écrite.
- Scénario : `MAX_FIRST_KL=0,5` (virgule) posé à la main sur un job : `float('0,5')` lève `ValueError`, sortie 1, le `if` est faux, l'entraînement de deux heures part sans plafond. Même chose si la première perte arrivait sous une forme que `float` ne lit pas. Le cas NaN, lui, est bien fermé.
- Touche : outillage
- Correction proposée : inverser le sens pour que seul 0 laisse passer : `if ! python -c "...; sys.exit(0 if math.isfinite(v) and v <= float('$MAX_FIRST_KL') else 3)"; then ... exit 4; fi`. Toute sortie non nulle, exception comprise, refuse.
- Effort : S
- Risque de la correction : nul ; le sha256 de `train.sh` n'est pas nommé par un préreg à ma connaissance, à vérifier par `git grep` avant.
- Vérification proposée : un cas de `tests/test_train_sh.py` avec `SHIM_KL="0.3"` et `MAX_FIRST_KL="0,5"`, attendu code 4 et un seul appel ; rouge sur le code actuel (code 0, deux appels). Les trois cas existants restent verts.

### R2-DIF-44 Sur CUDA, `.to(dtype)` passe la garde neuve et casse le module de rotation partagé
- Gravité : mineure
- Confiance : PLAUSIBLE
- Où : llvq-tetra/llvq_tetra/fused.py:86-100 (docstring de `_require_resident_device` : « a dtype change ... passes »), :121-122 (`Rotation` enregistre `signs` et `small` comme tampons), :133 (`torch.einsum` sur `self.small`), :223-225 (le bras CUDA tourne `flat` en float32).
- Fait : `Rotation` est un `nn.Module` enfant des `FusedTetraLinear`, ses deux tables sont des tampons, donc `nn.Module.to(torch.float16)` les convertit. Le bras CUDA de `forward` passe `flat` en float32 puis appelle `einsum("gt,...tm->...gm", small, v)` : `small` en f16 contre `v` en f32. `k > 1` sur toutes les largeurs de Qwen3 (2 560 = 512 × 5, 9 728 = 512 × 19), donc l'`einsum` est toujours pris. Ce n'est pas introduit par le diff, mais la garde neuve et son docstring promettent qu'un changement de dtype est sûr, et le test `test_a_conversion_on_the_same_device_is_accepted` arme `rotation=None`.
- Scénario : `model = from_pretrained(name, device_map="cuda")` puis `model.to(torch.float16)` (transformers ne bloque `.to(dtype)` que pour bnb, GPTQ et HQQ). Le premier `generate` échoue dans `einsum` sur un dtype mêlé, attendu sans l'avoir exécuté : erreur obscure au lieu d'un refus nommé. Sur Metal, `RotationKernel` garde ses tables en attributs simples (`mps.py:412-413`) et n'est pas touché.
- Touche : outillage (paquet `llvq-tetra`, bras CUDA)
- Correction proposée : dans `Rotation.__init__`, garder `signs` et `small` en attributs simples float32 comme `RotationKernel`, au lieu de `register_buffer` ; le déplacement d'appareil est déjà refusé par la garde des modules fusés, et ces tables ne sont pas dans le fichier. Ne pas recaster en float32 dans `forward` : `small` en f16 aurait déjà perdu des bits.
- Effort : S
- Risque de la correction : faible. Le `state_dict` du modèle chargé perd deux clés par rotation, qu'aucun lecteur ne lit (`is_serializable` rend False). Aucun noyau, aucun token servi.
- Vérification proposée : sur CPU, sans carte : `r = Rotation(signs, small, "cpu")` avec `n = 24` (`m = 8`, `k = 3`), `r.to(torch.float16)`, puis `r(torch.ones(1, 24))` ; attendu aujourd'hui une `RuntimeError` de dtype, après correction la même sortie qu'avant le `.to`.

### R2-DIF-45 `_is_runner_image` ne reconnaît qu'une des quatre écritures d'image Space et ne borne pas l'espace de noms
- Gravité : mineure
- Confiance : PLAUSIBLE
- Où : ops/run.py:1287-1291 ; `huggingface_hub` 2.4.0 `_jobs_api.py:99-104` (`_SPACE_IMAGE_PREFIXES`).
- Fait : la bibliothèque accepte `https://huggingface.co/spaces/`, `https://hf.co/spaces/`, `huggingface.co/spaces/` et `hf.co/spaces/`. `_is_runner_image` exige exactement quatre segments commençant par `hf.co/spaces`. Dans l'autre sens, tout espace de noms passe : `hf.co/spaces/<n'importe qui>/llvq-runner-x` reçoit `HF_TOKEN`, alors que le docstring de `_hf_token_secret` promet « only to the image this repository builds ».
- Scénario : les 125 + 8 lancements du dépôt écrivent `hf.co/spaces/Pier-Jean/llvq-runner-cuda[...]` et passent (*computed*, `git grep` sur `ops/jobs`). Un job écrit `--image huggingface.co/spaces/Pier-Jean/llvq-runner-cuda` part sans jeton : facturé, puis en échec sur un dépôt privé. Aucun script actuel ne le fait.
- Touche : outillage
- Correction proposée : retirer l'un des quatre préfixes (copie locale de la liste), couper `:tag` et `@digest` du nom, puis exiger l'espace de noms de l'opérateur (`args.namespace` ou une constante `RUNNER_NAMESPACE = "Pier-Jean"`) et un nom qui commence par `llvq-runner`. Fonction pure, testable sans réseau.
- Effort : S
- Risque de la correction : nul ; à faire avec la décision d'opérateur SEC-01 déjà ouverte.
- Vérification proposée : un test stdlib (`run.py` n'importe que la bibliothèque standard au chargement) sur les formes `hf.co/...`, `https://huggingface.co/...`, `.../llvq-runner-cuda:sha`, `hf.co/spaces/autre/llvq-runner` et `pytorch/pytorch:2.5.1-...`.

### R2-DIF-46 `docs/ENV.md` attribue `LLVQ_SEALED_ARTIFACT` aux seuls tests de `llvq-artifact`
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : docs/ENV.md:96 ; llvq-cuda/tests/golay70_decoder_matches_rust.rs:402
- Fait : le lot 1 fait lire `LLVQ_SEALED_ARTIFACT` au balayage Golay70 de `llvq-cuda`. La colonne « read by » dit encore « `llvq-artifact` tests ». `env_registry.rs` ne vérifie que les noms, il reste vert.
- Scénario : un lecteur qui cherche qui lit la variable ne trouve pas le balayage CUDA.
- Touche : doc
- Correction proposée : « `llvq-artifact` tests, `llvq-cuda/tests/golay70_decoder_matches_rust.rs` ».
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

## Relu sans constat

- `read_raw`, refus neufs (largeur 4 ou 8 lue en u32, groupe nul, produit des dims, longueurs `packed`, `scales`, `biases`, compte f16) : aucun fichier écrit par le dépôt ne peut les déclencher. `write_raw` porte depuis 905513c (f16, `t.data.len() != t.len()`) et 587a479 (quant, même formule `rows · ceil(row_len / group)`, largeurs 4 et 8 seules) le même contrôle avant d'écrire ; un groupe nul y paniquait dans `div_ceil`. Seuls `seal` (via `ArtifactWriter::seal`), `embedq` et `rowscale` écrivent des raws, tous par `write_raw` ; `int4swap` et `embedq` recopient le reste en octets. Largeurs écrites : 4 et 8 (`embedq`, groupe 64). Arithmétique Qwen3 (embedding 151 936 × 1 024 / 2 560 / 4 096 / 5 120, lignes multiples de 64) : groupes par ligne entiers, aucune queue de groupe. Rang 0 (`dims = []`) : 1 valeur des deux côtés ; dimension nulle : 0 groupe des deux côtés.
- `check_tail` : la queue vient de `calib.rs:1506-1523`, `d_out · (d_in mod cfg.block)`, `cfg.block = 24` pour Ball et Tetra ; le lecteur lit `d_out · (d_in mod 24)` depuis 1076264. 4B 16, 8B 16 et 0 (12 288), 14B 8 et 8, 0.6B 16 et 0 (3 072) : queue vide acceptée. Les autres constructeurs (`artifact2::to_quantized`, `rhoapply`, tests d'`int4swap` et `rowscale`) recopient la queue lue ou la calculent de même.
- Borne `gain < n_centroids` (lecture, `write_codes`, `code_stream`) : `fit_gain_centroids` rend `1 << k_bits` niveaux ou `vec![1.0]`, donc un champ de `log2 n` bits exact ; `nearest_level_index` rend un rang `< n`. Tetra : deux centroïdes, gain 0 ou 1. Ball à un centroïde : champ de 0 bit, gain 0. Le refus « 0 centroïde avec des blocs » est déjà une décision d'opérateur.
- `u32::try_from` (nom, rang, groupe, `d_out`, `d_in`, centroïdes, compteurs de `seal`) : que des longueurs et des comptes, aucun masque ni mot bas tronqué exprès.
- `TetraShapeGain::quantize` : le test porte sur chaque élément par `is_finite` ; sous-normal, zéro négatif et valeur finie dont le carré déborde ne prennent aucune branche nouvelle et suivent l'ancien chemin à l'identique.
- `GptqFactor::u_row` et propagation : `at(n, i, j) = i·n + j` ; `faer` et la référence remplissent `u` ligne par ligne, triangle haut, zéros dessous ; `u_row(i)` est `u[i·n .. (i+1)·n]`, fin valide pour `i = n - 1`. Tranches `row[r0..]` et `u_row(s+k)[r0..]` de même longueur `d_in - r0` (`factor.dim() == d_in` asserté l. 235) ; même `x * ukj` puis `-=`, même ordre de `k`, saut `x == 0` inchangé : identique au bit. `schur.rs:36` et `solve_block` non touchés.
- `rtbits`, trois lignes ajoutées : `stride_u32(nb) = ceil(6 nb / 8) · 2` ; pad par ligne 32 bits à 2 560 et 4 096, 16 à 9 728, 5 120 et 17 408, 0 à 12 288 et 3 072 ; 4B : 884 736 × 32 + 61 440 × 16 = 29 294 592 bits ; 2,5420 et 2,5501 refaits à la main (*computed*). Les modifications de libellés existants viennent du lot 1 (NUM-02).
- `[lints.rust] unsafe_code = "forbid"` : aucun `unsafe`, `allow(unsafe_code)`, `no_mangle`, `export_name`, `link_section`, `build.rs` ni dev-dependency dans les cinq paquets ; seule macro de std concernée, `thread_local!` (`examples/f1encprof.rs`), est `allow_internal_unsafe` ; chaîne CI 1.95.
- `ci.yml` : `paths` avec `**` puis négations puis deux réinclusions donne exactement l'ancien `paths-ignore` plus `docs/ENV.md` et `configs/README.md` ; `**` prend les fichiers racine et `.github/**`, `**.md` les `.md` racine. Aucun autre fichier exclu n'est lu par un test (`git grep` des lectures de `docs/`, `paper*/`, `proofs/`, `.md`).
- `run.py launch` : `max(60, ceil(1,5 · wall · 60))m`, unité `m` lue par la bibliothèque (`int(float(n) · 60)`), jamais plus court qu'avant. `publish` : recette lue et vérifiée avant `git_state` et `create_repo`, mêmes octets. `dequant` : coût et refus de format avant `run_uv_job` ; la bibliothèque 2.4.0 transporte le script sans le jeton de l'utilisateur. `bench` : les deux Spaces du dépôt (`llvq-runner-cuda`, `llvq-runner-cuda-qualite`) reçoivent le jeton.
- `hf_publish_prep.py` : la géométrie comparée vient du `config.json` du checkpoint recopié tel quel par `seal` puis `hfpack` (`hfpack.rs:397-428`) ; seul appelant, `CLAUDE.md:89`, passe `--revision`. `hf_cleanroom_check.py` : l'appel de `CLAUDE.md` passe trois arguments.
- `require_layout_version` : `hfpack` écrit l'entier 1 depuis 0c4197f, comme `tetra_fingerprint` et `block_dim` ; les deux fixtures les portent.
- `_apply` de `FusedTetraLinear` et `Int4FusedLinear` : la sonde vide rend `cuda:0` contre `cuda:0` pour `finalize` (`model.to("cuda")`), `mps:0` contre `mps:0` ; `.half()`, `.float()`, `.to(dtype)`, `.cuda()` sans index sur la carte courante, `model.to(model.device)` passent ; transformers 5.18 ne fait plus de `.to` après `postprocess_model`, et `accelerate_dispatch` ne part pas sur `{"": "cpu"}` (`modeling_utils.py:4358-4365`). `forward` : entrée et codes sur le même appareil dans `generate`. `chat.py` : `setdefault` ne change que le cas `--device` différent de l'accélérateur. Le README (snippet CPU, `device_map="mps"`, `"cuda"` sur une carte) n'est refusé nulle part.
- `LLVQ_HF_TILE` : 64 (seul job, `hf-cuda-4b.sh:63`) et 32 passent ; refus avant toute compilation.
- `llvqtune` : `REVISION` de DCLM égale au chemin `SHARD` ; celle de `cais/mmlu` est le `refs/main` du cache local. Les deux tests sautent sans torch, comme annoncé.
- Tests neufs `hostile_files.rs`, `gptq_row_slice.rs`, `test_fused_device.py`, `test_checks.py` : chacun peut rougir sous le mutant de son journal ; `gptq_row_slice.rs` compare à une copie de l'ancienne boucle, c'est son objet et non un défaut.

## Candidats au lot 1

R2-DIF-41, R2-DIF-42, R2-DIF-43, R2-DIF-46, et la correction minimale de R2-DIF-40 (message et test). R2-DIF-44 et R2-DIF-45 demandent un choix (forme des tables de rotation, espace de noms autorisé) et passent après.
