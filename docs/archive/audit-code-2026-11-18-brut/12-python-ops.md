# Python et outillage : constats
Périmètre réellement lu : en entier `llvq-tetra/llvq_tetra/` (`__init__`, `integration`, `quantizer`, `reader`, `tetra`, `modules`, `conversions`, `vocab`, `dequant`, `fused`, `metal`, `mps`, `checkdense`, `checkdecode`, `checkq4`, `checkq4guards`, `chat`, `gentokens`), `llvq-tetra/pyproject.toml`, `llvq-tetra/README.md` (début), `csrc/tetra_cuda.cu` (binding) ; `ops/run.py` en entier ; `ops/Dockerfile.cuda` (listes build et COPY) ; `ops/hf_publish_prep.py`, `ops/hf_tripwire_probe.py`, `ops/hf_cleanroom_check.py` en entier ; `ops/llvq_hf_check.py` (main) ; `ops/llvqtune/train.sh`, `llvqtune/__main__.py`, `domain/loop.py` (fin), `adapters/dclm_corpus.py`, `adapters/mmlu_corpus.py` (fetch) ; `ops/llvq-mlx`, `ops/mlx_tetra/served.py`, `serve.py`, `llvq_mlx.py` (`build`, `TetraLinear`, gardes) ; `ops/jobs/sensitivity-map-4b.sh` en entier et les neuf autres scripts les plus récents par grep ciblé.
Survolés : `checkmps.py`, `comparetokens.py`, `awq_speed.py`, `awq_dequant.py` (révisions, exceptions), `otsaudit.py`, les en-têtes PEP 723 de `ops/`.
Non couvert : `llvq-tetra/tests/` (relecteur tests), `awq_dequant.py` au-delà des révisions et des exceptions, `awq_batch.py`, `hf_cuda_check.py`, `mlx_tetra/ppl.py`, `check.py`, `mutate.py`, `strip.py`, les adaptateurs `torch_model.py` et `trainables/` de llvqtune, les 153 autres scripts de `ops/jobs/`.
Bilan : 0 critiques, 4 majeures, 15 mineures, 3 cosmétiques

### PY-01 `LLVQ_HF_TILE` autre que 64 fait déborder la mémoire partagée du noyau CUDA
- Gravité : majeure
- Confiance : CONFIRMÉ
- Où : llvq-tetra/llvq_tetra/metal.py:129,137 ; llvq-tetra/llvq_tetra/integration.py:27,209 ; llvq-tetra/llvq_tetra/fused.py:211-214 ; llvq-tetra/llvq_tetra/csrc/tetra_cuda.cu:76 ; data/kernels/llvq-llm/kernels/tv_tetra48_h.cu:61-71
- Fait : l'extension CUDA est compilée avec `-DTILE_BLOCKS={LLVQ_HF_TILE}` (défaut 64, aucune borne). Le module passe toujours `self._tile = integration.TILE_BLOCKS = 64`, et le binding alloue `tile · 24 · 4` octets de mémoire partagée dynamique. Le noyau indexe `extern __shared__ float xs[]` jusqu'à `TILE_BLOCKS · 24`.
- Scénario : `LLVQ_HF_FUSED=1 LLVQ_HF_TILE=128` sur une carte. Le noyau stage 128 blocs dans 6 144 octets alloués pour 64 : écriture hors de la mémoire partagée, comportement indéfini, sorties fausses ou erreur de lancement. `docs/ENV.md:107` documente la variable « no range check ».
- Touche : outillage (paquet PyPI, bras CUDA)
- Correction proposée : une seule source pour la tuile. Lire `LLVQ_HF_TILE` une fois dans `integration` et la passer au module, ou refuser toute valeur différente de `TILE_BLOCKS` par nom dans `_extension_cuda`.
- Effort : S
- Risque de la correction : nul. Paquet Python seul, noyau épinglé non touché.
- Vérification proposée : sur carte, `LLVQ_HF_TILE=128` avec la garde ajoutée doit refuser ; sans garde, `compute-sanitizer --tool memcheck` sur `gentokens` signalerait l'accès partagé hors borne.

### PY-02 Aucun lecteur ne vérifie `llvq_layout_version`, et un tenseur absent se charge non initialisé
- Gravité : majeure
- Confiance : CONFIRMÉ (absence de contrôle) ; PLAUSIBLE (conséquence sur un fichier d'une autre version)
- Où : llvq-tetra/llvq_tetra/quantizer.py:54,65-70 ; llvq-tetra/llvq_tetra/reader.py:38-53 ; llvq-tetra/llvq_tetra/modules.py:66-72 ; llvq-llm/src/hfpack.rs:64-67 ; ops/mlx_tetra/llvq_mlx.py:688
- Fait : `hfpack.rs` dit que `LAYOUT_VERSION` est « bumped when a tensor name or a byte convention changes ». `LlvqConfig` stocke le champ sans le lire, `PackedModel` et la porte MLX l'ignorent. Les buffers de `TetraLinear` et `Int4Linear` naissent `torch.empty`. Si une clé manque au fichier, `transformers` 5 avertit seulement, et `_init_weights` de Qwen3 ne touche pas ces modules. Le contrôle des clés manquantes n'existe que dans `gentokens.py:139` et les tests, jamais dans le quantizer.
- Scénario : un `hfpack` futur en layout 2 renomme `row_scales` ou change l'ordre des octets. `llvq-tetra` 0.2.0 accepte le fichier (empreinte et `code_order` inchangés), charge des poids faux ou de la mémoire non initialisée, et le seul signal est un warning. C'est le piège « modèle aléatoire » que `__init__.py` combat, rouvert par une version.
- Touche : format (lecteur Python)
- Correction proposée : refuser par nom `llvq_layout_version != 1` dans `LlvqConfig.__init__` et `PackedModel.__init__`. Dans `_process_model_after_weight_loading`, refuser si un buffer de module LLVQ figure dans les clés manquantes, ou initialiser les buffers flottants à NaN et vérifier `isfinite` dans `materialize` et `from_loaded`.
- Effort : S
- Risque de la correction : faible. Aucun format ni noyau touché ; un fichier layout 1 se charge comme avant.
- Vérification proposée : test sur `fixtures/mini` avec `llvq_layout_version` mis à 2 dans une copie du config, qui doit refuser ; et une copie privée de `row_scales`, qui doit refuser au lieu d'avertir.

### PY-03 Le `timeout` de `run.py launch` tronque à l'heure entière et peut tomber sous l'estimation
- Gravité : majeure
- Confiance : CONFIRMÉ
- Où : ops/run.py:682
- Fait : `timeout=f"{max(1, int(wall * 1.5))}h"`. `int` tronque. Le docstring de `device_ok` (lignes 138-140) rappelle que ce timeout est la marge à 1,5 fois.
- Scénario : estimation `wall = 1,3 h`. `1,3 × 1,5 = 1,95`, tronqué à 1, timeout 1 h, inférieur à l'estimation elle-même. Pour `wall = 2,6 h` le timeout est 3 h, soit 1,15 fois. Le job est tué avant la fin, facturé en entier, sans artefact, et l'estimation exclut déjà les passes avant.
- Touche : outillage
- Correction proposée : arrondir au-dessus en minutes, `timeout=f"{math.ceil(wall * 1.5 * 60)}m"`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante ; un test pur de la chaîne pour `wall` dans {0,5 ; 1,3 ; 2,6}.

### PY-04 L'image des jobs flotte, et rien dans le conteneur ne dit de quel commit elle sort
- Gravité : majeure
- Confiance : CONFIRMÉ
- Où : ops/jobs/*.sh (133 scripts lancent `--image hf.co/spaces/Pier-Jean/llvq-runner-cuda`, par ex. chain-runtime-door.sh:217, kv8-memory.sh:212) ; ops/run.py:921-924,930-949 ; ops/Dockerfile.cuda:143-171 ; llvq-runtime/build.rs:5-14 ; ops/run.py:744-758 (UPLOAD_ALLOW)
- Fait : l'image est la dernière construction du Space, sans étiquette de commit. `publish` écrit `/src/COMMIT` dans l'étage de build seulement ; le commentaire de `run.py` reconnaît que la copie dans l'étage d'exécution n'est pas faite. `.git` n'est pas téléversé, donc `LLVQ_BUILD_COMMIT` vaut `unknown` dans l'image. Trois scripts sur 163 mentionnent la provenance.
- Scénario : un préreg est tamponné, puis un `publish` reconstruit le Space depuis un autre commit avant le lancement. Le job mesure un binaire différent de celui du préreg, et aucun fichier de `$O` ne permet de le voir.
- Touche : outillage (provenance de toute mesure carte)
- Correction proposée : `COPY --from=build /src/COMMIT /usr/local/share/llvq/COMMIT` dans `Dockerfile.cuda`, et une ligne `cat /usr/local/share/llvq/COMMIT | tee "$O/image-commit.txt"` dans le corps type des jobs. Optionnel : exporter le sha dans `LLVQ_BUILD_COMMIT` au build depuis ce fichier.
- Effort : S (recette), M (gabarit des jobs)
- Risque de la correction : faible. Reconstruction d'image ; aucun noyau ni token servi touché.
- Vérification proposée : un job `bench` de 2 min qui affiche le fichier, après publication.

### PY-05 `publish` valide la réécriture de `CUDA_COMPUTE_CAP` après avoir téléversé les sources et `COMMIT`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/run.py:908-915, 943-949, 957-963, 971-976
- Fait : quatre commits successifs (`upload_folder`, `COMMIT`, `Dockerfile`, `README.md`). Le refus « exactly one CUDA_COMPUTE_CAP line » arrive après les deux premiers.
- Scénario : `publish --cuda --compute-cap 80` sur une recette à deux lignes `ENV CUDA_COMPUTE_CAP=`. Le script refuse avec code 1, mais le Space a déjà les nouvelles sources, l'ancien `Dockerfile`, et un `COMMIT` qui annonce `compute_cap rewritten to 80`. Le Space reconstruit cet état mixte, et un job lancé ensuite s'exécute dessus. Même risque si un téléversement réseau échoue entre deux commits.
- Touche : outillage
- Correction proposée : faire toutes les validations avant `create_repo`, puis un seul `create_commit` avec les quatre opérations.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### PY-06 `run.py dequant` lance le job payant avant d'annoncer son coût et de lire son timeout
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/run.py:1197-1231, 1267-1283
- Fait : `run_uv_job` part ligne 1197 ; le coût et `_timeout_minutes` viennent ligne 1227-1231. `_timeout_minutes` lève `SystemExit` sur `2h30m`.
- Scénario : `dequant --timeout 2h30m`. Le job est créé (ou refusé par l'API), puis le script meurt sur `SystemExit` sans afficher l'id ni le coût : un job peut tourner sans que l'opérateur ait son identifiant. Dans tous les cas, la règle dure 1 (coût annoncé avant) n'est pas tenue.
- Touche : outillage
- Correction proposée : déplacer le calcul et l'affichage du coût avant `run_uv_job`, comme dans `cmd_bench`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### PY-07 `oracle` ne vérifie pas la capacité de calcul et n'annonce aucun coût ; `bench --any-flavor` non plus pour une saveur inconnue
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/run.py:999-1030 ; ops/run.py:113-119, 1065-1087, 1128-1134
- Fait : `cmd_oracle` appelle `device_ok` mais pas `cap_ok`, et n'imprime pas de coût. `oracle` est un binaire candle, donc lié au PTX sm_89. Dans `cmd_bench`, `cap_ok` répond `True, "CPU"` pour une saveur absente de `FLAVORS`, et le coût n'est imprimé que si `usd_h` est connu.
- Scénario : `run.py oracle --flavor a100-large --device cuda`. Le job démarre, tire l'image, télécharge le 0.6B et échoue au premier noyau, ce que `launch` refuse en amont. `bench --any-flavor --flavor a10g-large` part sans contrôle de capacité ni coût affiché.
- Touche : outillage
- Correction proposée : appeler `cap_ok` et imprimer `usd_h × 20 min` dans `cmd_oracle` ; dans `cmd_bench`, refuser une saveur absente de `FLAVORS` même sous `--any-flavor`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### PY-08 `run.py launch` ne peut pas calibrer sur `dclm-edu`, et n'épingle ni le modèle ni le corpus
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/run.py:1538-1544, 598-602, 247-251
- Fait : `--calib` accepte `c4`, `wikitext2`, `wikitext2-test`, défaut `c4`. `smoke.rs` accepte `dclm-edu`, le corpus des fichiers scellés (CLAUDE.md). Le commentaire ligne 1541 dit qu'un corpus absent ici est « unreachable from launch ». `LLVQ_MODEL` part sans révision, `LLVQ_DATASET_REV` n'est jamais passé, et `fetch_config` lit `raw/main`.
- Scénario : un réencodage du protocole publié via `launch` est impossible sans contourner le script ; un `launch` par défaut calibre sur c4, hors protocole, et sur la révision courante du modèle et du corpus.
- Touche : outillage (encodeur)
- Correction proposée : ajouter `dclm-edu` aux choix ; ajouter `--model-revision` et `--dataset-rev` transmis au conteneur. Changer le défaut est une décision d'opérateur.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### PY-09 `device_map="cuda:1"` place le modèle sur `cuda:0`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-tetra/llvq_tetra/integration.py:50-64, 169-170 ; llvq-tetra/llvq_tetra/fused.py:152-157 ; llvq-tetra/llvq_tetra/metal.py:176-210
- Fait : `target_device` réduit chaque valeur à son genre, `"cuda"`. `finalize` fait `model.to("cuda")` et les buffers fusés vont sur `"cuda"`, c'est-à-dire l'appareil courant. Les caches `invnorm_on` et `_tables_on_device` sont indexés par `"cuda"`.
- Scénario : machine à deux cartes, `device_map={"": 1}` ou `"cuda:1"`. Le modèle atterrit sur la carte 0, sans message. L'utilisateur qui envoie ses entrées sur `cuda:1` obtient une erreur d'appareil ; celui qui visait la carte 1 pour sa mémoire sature la carte 0.
- Touche : outillage (paquet PyPI)
- Correction proposée : garder l'appareil complet (`torch.device`) dans `target_device`, refuser plusieurs index distincts, et l'utiliser partout où `device` est passé.
- Effort : S
- Risque de la correction : faible.
- Vérification proposée : test unitaire pur de `target_device` sur `{"": 1}`, `"cuda:1"`, `{"a": 0, "b": 1}`.

### PY-10 Les tenseurs résidents des modules fusés ne suivent pas `model.to()`, et la branche CPU appelle une op Metal
- Gravité : mineure
- Confiance : CONFIRMÉ (lecture) ; PLAUSIBLE (scénario)
- Où : llvq-tetra/llvq_tetra/fused.py:119-122, 166-171, 182-226, 246-247, 290-303 ; llvq-tetra/llvq_tetra/chat.py:63-69
- Fait : `_words`, `_gscale`, `_rscale`, `_tail`, `_wq`, `_scales`, `_biases` sont des attributs simples, pas des buffers, alors que `bias` est un `Parameter` qui, lui, bouge. Quand l'entrée n'est ni sur `mps` ni sur `cuda`, `forward` tombe dans la branche qui appelle `torch.ops.llvq.tv_tetra48` (op de l'extension Metal 0.1.0, jamais chargée sur ce chemin) avec des tenseurs CPU ; idem `tv_q4` pour `Int4FusedLinear`.
- Scénario : Mac, `LLVQ_HF_FUSED=1 python -m llvq_tetra.chat --device cpu`. `chat.py` ne pose pas `LLVQ_HF_DEVICE`, le quantizer compresse sur `mps`, puis `model.to("cpu")` déplace biais et normes mais pas les codes. Le premier `forward` échoue sur un attribut d'op absent, ou mélange appareils, au lieu d'un refus nommé.
- Touche : outillage (paquet PyPI)
- Correction proposée : dans `forward`, refuser par nom une entrée dont l'appareil n'est pas celui de `_words` ; supprimer la branche Metal-sur-CPU. Dans `chat.py`, poser `LLVQ_HF_DEVICE` comme le fait `gentokens.py:125`.
- Effort : S
- Risque de la correction : faible. Les chemins `mps` et `cuda` ne changent pas.
- Vérification proposée : test CPU avec un `FusedTetraLinear` armé de tenseurs factices sur `meta` et une entrée CPU, qui doit lever le refus nommé.

### PY-11 llvqtune télécharge DCLM-edu et MMLU auxiliaire sans révision
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/llvqtune/llvqtune/adapters/dclm_corpus.py:24-28, 41-46, 71-76 ; ops/llvqtune/llvqtune/adapters/mmlu_corpus.py:93-101
- Fait : `SHARD` nomme le snapshot `dbad8ad7…` dans `~/.cache`, mais si ce chemin manque, `fetch_shard` appelle `hf_hub_download` sans `revision`, donc `main`. Sur une carte (HF_HOME ailleurs, cache vide) c'est toujours ce cas. Même chose pour MMLU `auxiliary_train`.
- Scénario : le dépôt `HuggingFaceTB/dclm-edu` change son shard 0. Un réentraînement des échelles par ligne lit un autre texte que le premier, avec la même graine, et le journal ne le dit pas.
- Touche : outillage
- Correction proposée : passer `revision="dbad8ad71224482740cd9c9d353591adbf62fe04"` (et une révision épinglée pour MMLU) à `hf_hub_download`, et écrire la révision dans le journal.
- Effort : S
- Risque de la correction : nul si la révision est celle du cache local actuel.
- Vérification proposée : aucune, lecture suffisante.

### PY-12 La porte MLX charge les poids f16 avec `strict=False`
- Gravité : mineure
- Confiance : PLAUSIBLE
- Où : ops/mlx_tetra/llvq_mlx.py:745-755
- Fait : `model.load_weights(weights, strict=False)`. Un nom de tenseur f16 du fichier qui ne correspond à aucun paramètre est ignoré, et un paramètre non couvert garde son initialisation (normes à 1).
- Scénario : un fichier dont une norme porte un autre nom, ou un modèle non lié dont `lm_head` est int4 (non traité par la boucle f16 ni par la branche embedding) : le modèle se construit et génère avec un paramètre par défaut, sans erreur. La liste `OBJECTS` limite la porte au 4B lié, mais `check.py` appelle `build` sur n'importe quel répertoire.
- Touche : outillage (porte MLX)
- Correction proposée : comparer l'ensemble des noms chargés à l'ensemble attendu (paramètres f16 du modèle moins ceux des modules Tetra et int4) et refuser tout écart par nom.
- Effort : S
- Risque de la correction : faible.
- Vérification proposée : `check-mini` avec une copie du fixture où une norme est renommée.

### PY-13 `hf_publish_prep.py` accepte une révision absente et prend le tokenizer du 4B par défaut
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/hf_publish_prep.py:101-105, 118
- Fait : `--revision` vaut `None` par défaut, alors que l'aide dit « pin it ». `--checkpoint` vaut `Qwen/Qwen3-4B` quel que soit le répertoire.
- Scénario : préparation du 8B sans `--checkpoint` ni `--revision` : le dépôt publié reçoit le `generation_config.json` et le `tokenizer_config.json` du 4B à la révision courante, sans avertissement.
- Touche : outillage (publication)
- Correction proposée : rendre `--revision` et `--checkpoint` obligatoires, ou déduire le checkpoint du `config.json` empaqueté et refuser une révision absente.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### PY-14 `hf_cleanroom_check.py` à trois arguments saute la porte du 4B et sort 0
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/hf_cleanroom_check.py:90-92
- Fait : `if rc or len(argv) < 4: return rc`. Avec le fixture et le répertoire empaqueté mais sans le dump `bin/run`, la deuxième porte ne tourne pas et le script rend 0 sans le dire.
- Scénario : un oubli du troisième argument lit « vert » alors que seule la porte du fixture a tourné.
- Touche : outillage
- Correction proposée : accepter exactement 2 ou 4 arguments, refuser 3 ; imprimer « 4B gate not run » quand il y en a 2.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### PY-15 Cinq scripts de job calculent le plafond en supposant un `TIMEOUT` en minutes
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/jobs/sensitivity-map-4b.sh:60,143-144 ; ops/jobs/patch-map-4b.sh:115 ; ops/jobs/gsm8k-depth-restore-4b.sh:126 ; ops/jobs/lens-write-depth.sh:90 ; ops/jobs/contrast-map-4b.sh:91
- Fait : `TIMEOUT=${TIMEOUT:-45m}` est surchargeable, puis `MIN=${TIMEOUT%m}` et `awk` divise par 60.
- Scénario : `TIMEOUT=2h`. `MIN="2h"`, `awk` lit 2, le script annonce un plafond de 0,06 $ au lieu de 3,60 $. `run.py bench` réimprime le bon plafond juste après, ce qui limite le dégât, mais la ligne `cost:` du script est fausse.
- Touche : outillage
- Correction proposée : refuser un `TIMEOUT` qui ne finit pas par `m` (`case "$TIMEOUT" in *[0-9]m) ;; *) refuse ...`), ou convertir les heures.
- Effort : S
- Risque de la correction : nul. Le sha256 du script change, ce que les préregs déjà tamponnés nomment : ne corriger qu'au prochain job de chaque script.
- Vérification proposée : `DRY_RUN=1 TIMEOUT=2h` doit refuser.

### PY-16 `train.sh` laisse passer une première perte NaN sous `MAX_FIRST_KL`
- Gravité : mineure
- Confiance : PLAUSIBLE
- Où : ops/llvqtune/train.sh (bloc `MAX_FIRST_KL`, `float('$FIRST_KL') > float('$MAX_FIRST_KL')`) ; ops/llvqtune/llvqtune/domain/loop.py:209-212,234
- Fait : `first_loss` peut être NaN (export cassé, débordement f16). `nan > x` est faux, donc la garde conclut « sous le seuil » et l'entraînement de deux heures part. La garde `FIRST_LOSS_BAND` refuse bien NaN, elle.
- Scénario : un export corrompu dont la première KL est NaN, lancé avec `MAX_FIRST_KL` seul.
- Touche : outillage
- Correction proposée : écrire la condition à l'envers, `sys.exit(1 if v <= max else 0)` avec `math.isfinite(v)` exigé.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : un cas dans `tests/test_train_sh.py` avec `first_loss` NaN.

### PY-17 Un `quantization_config` sans `tetra_fingerprint` ou `block_dim` lève une erreur Python opaque
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-tetra/llvq_tetra/quantizer.py:59-60, 86-88 ; llvq-tetra/llvq_tetra/tetra.py:123
- Fait : `LlvqConfig` vérifie `code_order` et `records`, pas ces deux champs. Absents, `require_fingerprint(None)` lève `AttributeError: 'NoneType' object has no attribute 'lower'` et `int(None)` un `TypeError`.
- Scénario : un modèle mal empaqueté, ou un `quant_method: "llvq"` d'un autre outil : l'utilisateur reçoit une trace interne au lieu d'un refus qui nomme le champ.
- Touche : outillage (paquet PyPI)
- Correction proposée : refuser par nom les champs requis absents dans `LlvqConfig.__init__`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : test unitaire `LlvqConfig(code_order="msb_first_dense", records={"a": {}})`.

### PY-18 Aucun chemin de `run.py` n'écrit la ligne de `docs/data/jobs.csv`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : ops/run.py:686-687, 1029, 1149-1150, 1233 ; docs/data/jobs.csv
- Fait : les quatre commandes qui facturent impriment l'id et sortent. Le registre est tenu à la main à la clôture (`docs/METHODE.md:148`), et les lignes anciennes portent des montants vides (« montant non journalise »).
- Scénario : un job qui meurt, ou dont l'opérateur ne clôt pas la session, n'a aucune ligne ; le cumul annoncé par la règle 1 est alors faux sans signal.
- Touche : outillage
- Correction proposée : à chaque lancement, ajouter une ligne `date,job_id,name,flavor,,,,lancé` que la clôture complète ; la commande `monitor` peut remplir `billed_min` et `cost_usd`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### PY-19 Les planchers de llvqtune autorisent la paire torch 2.5 et transformers 5 connue cassée
- Gravité : mineure
- Confiance : PLAUSIBLE
- Où : ops/llvqtune/pyproject.toml (extra `torch`) ; ops/llvqtune/llvqtune/__main__.py:93,139 ; llvq-tetra/pyproject.toml (commentaire des planchers)
- Fait : `torch>=2.5`, `transformers>=4.44`, sans borne haute. `llvq-tetra/pyproject.toml` note que torch 2.5.1 casse contre transformers 5.x. `__main__.py` passe `torch_dtype=`, renommé `dtype` dans transformers 5. Les jobs installent dans l'image `pytorch:2.5.1`.
- Scénario : `uv sync` sans le verrou, ou un job qui `pip install` les dépendances de l'extra, résout transformers 5 sur torch 2.5 et échoue au chargement, après facturation.
- Touche : outillage
- Correction proposée : borner `transformers<5` (ou monter torch au plancher de llvq-tetra) et passer `dtype=`.
- Effort : S
- Risque de la correction : faible.
- Vérification proposée : lire `uv.lock` et le `pip-freeze.txt` du dernier job llvqtune pour voir la paire réellement résolue.

### PY-20 Des gardes de refus reposent sur `assert`
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : llvq-tetra/llvq_tetra/checkq4guards.py:43-48 ; llvq-tetra/llvq_tetra/checkmps.py:48,53,70 ; ops/mlx_tetra/llvq_mlx.py:374,546,604,656,741
- Fait : le décompte des sites d'un mutant, la garde « la mutation ne touche pas la référence », l'alignement `d_out` du noyau MLX et la taille des mots Tetra sont des `assert`.
- Scénario : sous `python -O` ou `PYTHONOPTIMIZE=1`, ces gardes disparaissent : un mutant appliqué aux deux noyaux se lit « SURVIVED » ou « caught » sans rien tester, et la porte MLX lance un noyau sur une forme non alignée.
- Touche : outillage
- Correction proposée : remplacer par `if ...: raise ValueError(...)`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### PY-21 Le README de llvq-tetra exige ninja et Xcode sur Mac, ce que le code n'exige plus
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : llvq-tetra/README.md:37-46 ; llvq-tetra/llvq_tetra/mps.py:1-6 ; llvq-tetra/llvq_tetra/quantizer.py:92-101
- Fait : le README dit que le chemin fusé « needs `ninja` and a compiler: Xcode command line tools on a Mac ». Le chemin Metal passe par `torch.mps.compile_shader`, « no ninja, no command line tools », et `validate_environment` n'exige ninja que pour CUDA. Le README ne mentionne pas `device_map`, la voie de 0.2.0.
- Scénario : un utilisateur Mac installe un compilateur inutile, ou renonce.
- Touche : doc
- Correction proposée : réécrire la section « Two ways to run it » sur `device_map="mps"` et `"cuda"`, ninja pour CUDA seulement.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### PY-22 Code mort et fichiers sans appelant
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : llvq-tetra/llvq_tetra/checkmps.py (cité par aucun fichier du dépôt, `git grep checkmps`) ; llvq-tetra/llvq_tetra/checkdense.py:41,60-62 (compteur `bad` inutile : retour au premier écart) ; ops/otsaudit.py:95,128 (`open(doc,"rb").read()` sans fermeture) ; ops/hf_cleanroom_check.py:84 (`site-packages` seul, `dist-packages` refusé à tort sous Debian)
- Fait : voir les lignes.
- Scénario : sans effet fonctionnel ; `checkmps` part dans la roue PyPI sans que rien ne le documente.
- Touche : outillage
- Correction proposée : citer `checkmps` dans CLAUDE.md ou l'exclure de la roue ; retirer `bad` ; `Path(doc).read_bytes()`.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune.

## Connus non corrigés

- CONNU-NON-CORRIGÉ : `ops/hf_speed_mac.py` n'est cité nulle part (dernier commit 2026-10-09). Réf. `docs/audit-architecture-2026-11.md` §1.2.
- CONNU-NON-CORRIGÉ : aucun test Python ne tourne en CI. Réf. `docs/audit-architecture-2026-11.md` constat 6 et lot 8.

## Ce qui tient

- Garde-fou d'enregistrement : `__init__.py:49-57` ne tolère l'échec d'import que si torch ou transformers sont absents. Un transformers trop vieux lève. Les chemins `from_pretrained` testés (`device_map` absent, `"auto"`, `"cpu"`, `"mps"`, `"cuda"`, dictionnaire, `AutoModel` au lieu de `AutoModelForCausalLM`) refusent ou chargent correctement : `replace_modules` lève si le nombre de modules remplacés diffère du nombre d'enregistrements. Le piège « transformers seul, sans import » reste ouvert et documenté (`hf_publish_prep.py:17-30`).
- Dépendances du paquet : tout ce qu'importe `llvq_tetra` est déclaré ; aucun import n'exige Metal ou CUDA au chargement ; `data/` et `csrc/` partent dans la roue (hatch, aucune exclusion).
- `ops/jobs/` : les 163 scripts portent `set -euo pipefail`. Les dix plus récents refusent un dossier de sortie déjà présent dans le bucket, gardent le code de retour par `PIPESTATUS[0]` derrière `set +e`, et n'utilisent `|| true` que sur des `grep` et `hf buckets ls` d'affichage. Les listes `cargo build` et `COPY` de `Dockerfile.cuda` concordent (30 binaires).

## En-têtes PEP 723 (liste)

- `ops/run.py` : `huggingface-hub>=1.26`, plancher sans épingle.
- `ops/llvq_hf_check.py` : `numpy>=2.0`, `safetensors>=0.4`, planchers.
- `ops/awq_dequant.py` : `numpy>=2.0`, `safetensors>=0.4`, `huggingface-hub>=1.26`, planchers.
- `ops/mlx_tetra/{check,llvq_mlx,mutate,prof,serve,strip}.py` : épinglés exactement, mais `requires-python = ">=3.10"` alors que `llvq-tetra` exige `>=3.11` ; `floor.py` épingle `mlx==0.29.3` contre 0.32.4 ailleurs (dit dans son commentaire).
- Sans en-tête : `hf_publish_prep.py` (importe `huggingface_hub`, lancé par `uv run --with`), `otsaudit.py` (idem `opentimestamps`), `hf_tripwire_probe.py`, `hf_cleanroom_check.py`, `awq_speed.py` (délibéré), `awq_batch.py`, `hf_cuda_check.py`, `hf_speed_mac.py`, `mlx_tetra/ppl.py` et `served.py` (modules importés).

## Candidats au lot 1

PY-01, PY-02, PY-03, PY-05, PY-06, PY-07, PY-08 (ajout du choix `dclm-edu` seulement), PY-09, PY-11, PY-13, PY-14, PY-16, PY-17, PY-20, PY-21.
