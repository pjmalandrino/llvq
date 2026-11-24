# Sécurité et unsafe : constats
Périmètre réellement lu : en entier `llvq-runtime/src/{store,library,main,api,client}.rs`, `llvq-artifact/src/{sealed,tetra48}.rs`, `llvq-llm/src/sealed.rs`, `llvq-tetra/llvq_tetra/reader.py`, `llvq-cuda/src/modcache.rs`, `ops/hf_publish_prep.py` (en-tête et shim), `.github/workflows/ci.yml` (jobs et étapes). Lus par extraits ciblés : `llvq-runtime/src/{engine,openai,chat}.rs` (bornes, KeepAlive, prepare, Router, scan), `llvq-artifact/src/format.rs:540-560, 983-1330` (têtes, corps lattice et int4), `llvq-llm/src/fused.rs:200-240, 2266-2330, 2379-2560`, `llvq-llm/src/fused_cuda.rs:1450-1510, 1695-1900, 3850-3880`, `llvq-llm/src/persist4e_cuda.rs:683-1160`, `llvq-llm/src/persist_cuda.rs:90-140`, `llvq-llm/src/served_decode.rs:37-190, 345-380`, `llvq-llm/src/tetra_cpu.rs:300-620`, `llvq-llm/src/fused_metal.rs:1639-1676, 2275-2310`, `llvq-llm/src/fused_cpu.rs:270-300`, `llvq-llm/src/loader.rs:300-320`, `llvq-llm/src/dflash.rs:265-280`, `llvq-llm/src/batch_cuda.rs:60-110`, `llvq-cuda/src/{gpu.rs:250-300, e1v_host.rs:60-90}`, `llvq-artifact/src/blockrec.rs:90-135`, `llvq-metal/src/lib.rs:130-150`, `llvq-llm/kernels/emb_q4.{cu,metal}` (gather), `llvq-tetra/llvq_tetra/{metal.py:100-175, csrc/tetra_cuda.cu:30-110, csrc/tetra_decode.mm:75-230}`, `ops/run.py:556-700, 740-920, 1060-1230`, `ops/Dockerfile.cuda`, `ops/jobs/{iq2-s-m,iq2m,paper-iq2-census,duel-iq2}.sh` (lancement). Balayés par recherche : tous les `unsafe` du workspace hors `docs/archive`, les secrets (motifs de jetons), `curl | sh`, les en-têtes PEP 723 de `ops/` (51 fichiers), `torch.load`, `pickle`, `eval`, `trust_remote_code`, `auto_map`, les `hf_hub_download`/`from_pretrained`, la source de `ureq` 2.12.1 (redirections), `axum-core` 0.5.6 (limite de corps), `candle-core` 0.9.2 et `cudarc` 0.19.8 (contexte, flux).
Non couvert : le contenu des noyaux CUDA/Metal au-delà du gather d'embedding (indices de lecture en fonction des dimensions du fichier, surtout `llvq_pers.cuh` et `pers_4et.cu`), les binaires de banc de `llvq-cuda/src/bin` et `llvq-metal/src/bin` (leurs `unsafe` sont des lancements et lectures de tampons sur des données construites par le banc), `ops/llvqtune`, `ops/mlx_tetra`, `llvq-tetra/llvq_tetra/{integration,quantizer,mps,fused}.py` au-delà des appels de noyau, `hfpack.rs` côté écriture, le contenu de `chat.html` (déjà relu par `docs/runtime-audit-2026-11.md`).
Bilan : 0 critiques, 1 majeure, 7 mineures, 3 cosmétiques

Jugement d'exposition. Le serveur `llvq` écoute sur `127.0.0.1:11435` par défaut (`llvq-runtime/src/client.rs:10`), refuse un `Host` qui n'est ni un nom de boucle locale ni une adresse IP, et refuse un `Origin` étranger (`api.rs:994-1047`). Le corps d'une requête est borné à 2 Mo par axum (`axum-core-0.5.6/src/ext_traits/request.rs:319`, extracteur `Bytes`). `llvq pull` ne télécharge que les trois entrées compilées dans `library.json`, à une révision de 40 caractères, et vérifie taille et sha256 avant le `rename` atomique (`store.rs:290-393`). Le jeton `HF_TOKEN` ne suit pas une redirection (`ureq` 2.12.1, `RedirectAuthHeaders::Never` par défaut, `unit.rs:206-224`). Les lecteurs du format bornent leurs préallocations (`PREALLOC_CAP`) et vérifient les longueurs des enregistrements lattice et int4 avant tout transfert vers la carte. Aucun secret en clair dans l'arbre suivi. Aucun `pickle`, `torch.load` ni `eval` sur une donnée chargée ; `llvq-tetra` lit `safetensors` et vérifie l'empreinte de la carte Tetra. Les constats ci-dessous sont donc des défauts de second rang, sauf SEC-01.

### SEC-01 Le jeton HF et le bucket entier en écriture sont confiés à une image tierce à étiquette flottante
- Gravité : majeure
- Confiance : CONFIRMÉ pour le code ; la présence du jeton dépend du shell de l'opérateur au lancement
- Où : ops/run.py:1121-1122, 1135-1148 ; ops/jobs/iq2-s-m.sh:18-21 ; ops/jobs/iq2m.sh:20-23 ; ops/jobs/paper-iq2-census.sh:85-88
- Fait : `run.py bench` passe `secrets={"HF_TOKEN": ...}` à tout job dès que la variable existe localement, quelle que soit l'image (`--image`). Trois scripts lancent `ghcr.io/ggml-org/llama.cpp:full-cuda`, une étiquette mobile, avec `--bucket <bucket des artefacts de jobs> --out-mount /out` : le bucket entier est monté en écriture.
- Scénario : l'image `:full-cuda` est remplacée en amont (compromission du registre ou du dépôt llama.cpp) entre deux lancements. Le code de l'image tourne avec `HF_TOKEN` dans son environnement et `/out` en écriture. Il obtient le jeton de l'opérateur (écriture sur les dépôts Hub, dont `-served`, lancement de jobs facturés) et peut réécrire les journaux bruts des autres jobs dans le bucket, sur lesquels reposent les chiffres publiés. Une image épinglée par commit (`duel-iq2.sh:112-114`) n'a pas ce défaut, `vllm/vllm-openai:v0.26.0` l'a à moitié (étiquette de version).
- Touche : outillage
- Correction proposée : ne passer `HF_TOKEN` que sur option explicite (`--hf-token`), jamais par défaut, et refuser l'option quand `--image` n'est pas l'image de ce dépôt. Épingler les images tierces par digest `@sha256:`. Monter un sous-chemin du bucket, ou un bucket dédié, pour les images tierces.
- Effort : S
- Risque de la correction : faible. Ne touche ni format, ni noyau, ni tokens. Un job qui lit un dépôt privé devra demander le jeton par nom.
- Vérification proposée : un test Python de `run.py` qui construit les arguments de `run_job` pour `bench --image ghcr.io/...` avec `HF_TOKEN` posé et vérifie que `secrets` est `None`.

### SEC-02 Le décodeur 4E garde des adresses brutes dans les tenseurs du modèle, et une API sûre permet de lui survivre
- Gravité : mineure
- Confiance : CONFIRMÉ pour le trou d'API ; aucun appelant actuel ne le déclenche
- Où : llvq-llm/src/persist_cuda.rs:114-120 ; llvq-llm/src/persist4e_cuda.rs:699, 954-955, 974-976, 1121-1160 ; llvq-llm/src/served_decode.rs:37-39, 158-173, 357 ; llvq-runtime/src/engine.rs:36-39
- Fait : `tensor_addr` lit l'adresse de carte d'un `Tensor` candle en `u64`. `Runtime4e` stocke ainsi les poids de norme, `cos`, `sin`, et les mots des projections du modèle, sans garder de clone des `Tensor` (seuls ses propres tampons vont dans `_held*`). `Served::decoder(&Qwen3, ..)` rend un `Decoder` sans durée de vie liée au modèle, et `continue_greedy(model, ..)` ignore `model` sur le chemin persistant (`rt.decode(..)`, ligne 170). Dans `Loaded`, `qwen` est déclaré avant `decoder`, donc libéré avant lui.
- Scénario : un appelant en Rust sûr construit le décodeur sur le modèle A, libère A, puis appelle `continue_greedy(&B, ..)`. Le noyau 4E lit de la mémoire de carte libérée, peut-être réallouée : tokens faux ou adresse illégale. Aujourd'hui `Runtime4e` n'a pas de `Drop` qui lance un noyau et le moteur tient les deux ensemble, donc rien ne se déclenche.
- Touche : chemin servi
- Correction proposée : faire garder par `Runtime4e` un clone (`Tensor::clone`, un `Arc`) de chaque tenseur du modèle dont il lit l'adresse, à côté des `_held*`. Ou lier `Decoder<'m>` à `&'m Qwen3`.
- Effort : M
- Risque de la correction : faible. Aucun octet de noyau ne bouge, les tokens restent identiques ; quelques références en plus, pas de mémoire en plus.
- Vérification proposée : aucune locale (CUDA). Sur carte, un test qui construit le décodeur, libère le modèle et décode doit échouer à la compilation après la correction par durée de vie, ou rester correct après la correction par clones.

### SEC-03 Une seule requête peut bloquer le moteur avec une liste de chaînes d'arrêt
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-runtime/src/api.rs:135, 143-153 ; llvq-runtime/src/openai.rs:253 ; llvq-runtime/src/chat.rs:89-101 ; llvq-runtime/src/engine.rs:458-459, 689, 1033, 1195
- Fait : `stop` accepte un nombre quelconque de chaînes, de taille quelconque, dans la limite des 2 Mo du corps. À chaque pas (un token, ou un lancement 4E de 64 tokens), `Router::step` décode tout le contenu et appelle `chat::scan`, qui fait un `text.find(s)` par chaîne puis, pour chaque chaîne, un `ends_with` par préfixe.
- Scénario : un client local envoie `/api/chat` avec 200 000 chaînes d'arrêt de 8 caractères. Chaque pas coûte environ 200 000 recherches sur un texte qui grandit jusqu'à des dizaines de Ko, soit des milliards d'octets comparés par pas. Le moteur traite une requête à la fois : toutes les suivantes attendent, et un client qui part n'est vu qu'entre deux pas.
- Touche : runtime
- Correction proposée : refuser au-delà de 16 chaînes d'arrêt et de 256 octets par chaîne, avec un 400 qui le dit (OpenAI en accepte 4).
- Effort : S
- Risque de la correction : nul pour les tokens servis ; un client qui envoie plus de 16 chaînes reçoit un 400.
- Vérification proposée : un test de route qui envoie 17 chaînes et attend un 400 ; un mutant qui retire la borne doit le faire échouer.

### SEC-04 Lié hors de la boucle locale, le serveur n'a aucune authentification et ne prévient pas
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-runtime/src/client.rs:10-21 ; llvq-runtime/src/main.rs:480-499 ; llvq-runtime/src/api.rs:1026-1031, 1038 ; llvq-runtime/src/api.rs:660-696
- Fait : `LLVQ_HOST=0.0.0.0:11435` lie toutes les interfaces. `check_origin` accepte tout `Host` qui est une adresse IP littérale, et un client sans `Origin` passe. Aucune route n'a d'authentification. `serve` n'écrit aucun avertissement quand l'adresse n'est pas une adresse de boucle locale. Par ailleurs tout `Origin` de boucle locale passe, quel que soit le port.
- Scénario : un opérateur lie `0.0.0.0` pour atteindre la page de chat depuis un portable. Tout hôte du réseau local peut alors `DELETE /api/delete` les modèles, déclencher `/api/pull` du 14B (9 Go), charger un modèle `keep_alive: -1`, ou monopoliser le moteur. Une page servie par n'importe quel serveur local (un `python -m http.server` sur un HTML téléchargé) peut aussi piloter le serveur lié en boucle locale. C'est le comportement d'Ollama ; l'exposition par défaut est la boucle locale, d'où la gravité.
- Touche : runtime
- Correction proposée : imprimer un avertissement explicite au démarrage quand l'adresse liée n'est pas de boucle locale, et le documenter dans `docs/ENV.md`. Option plus forte : un jeton `LLVQ_API_KEY` exigé hors boucle locale.
- Effort : S (avertissement), M (jeton)
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### SEC-05 Le cache de modules exécutables n'est protégé que par une somme qu'il porte lui-même
- Gravité : mineure
- Confiance : CONFIRMÉ pour le mécanisme ; PLAUSIBLE pour l'exposition (exige un répertoire partagé)
- Où : llvq-cuda/src/modcache.rs:22-25, 92-121, 159-199 ; llvq-cuda/src/gpu.rs:357-393
- Fait : le cache stocke le cubin que la carte exécute, avec un en-tête qui répète la clé, la longueur et le sha256 du cubin. La lecture vérifie que le sha256 du corps égale celui de l'en-tête du même fichier. Le commentaire affirme qu'un « file edited by hand reads as a miss ». C'est faux pour une modification volontaire : la clé est déterministe et publique, et l'auteur recalcule la somme. `LLVQ_MODULE_CACHE` accepte tout répertoire absolu, sans contrôle de propriétaire ni de droits.
- Scénario : `LLVQ_MODULE_CACHE=/tmp/llvq-modules` sur une machine partagée, ou un cache posé sur un volume que d'autres jobs écrivent. Un autre utilisateur y dépose un cubin de son choix au nom attendu. Le prochain démarrage de la config `chain` le charge sans compiler : code GPU arbitraire dans le processus servi, qui peut fausser les tokens. Le défaut `~/.cache/llvq/modules` appartient à l'utilisateur et n'a pas ce défaut.
- Touche : chemin servi
- Correction proposée : corriger le commentaire (intégrité contre la corruption, pas contre la modification). Refuser un répertoire dont le propriétaire n'est pas l'utilisateur courant ou qui est accessible en écriture au groupe ou aux autres, comme le fait ssh.
- Effort : S
- Risque de la correction : nul pour les tokens ; aucun fichier de noyau ne bouge.
- Vérification proposée : un test de `dir_from` sur un répertoire en mode 0777 qui attend un refus ; un mutant qui retire le contrôle doit le faire échouer.

### SEC-06 Un tenseur brut quantifié n'est pas validé à la lecture, et son décodage panique ou avorte sur un fichier forgé
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : llvq-artifact/src/sealed.rs:101-103, 122-142, 305-325 ; llvq-llm/src/sealed.rs:537-552
- Fait : `read_raw` lit `bits`, `group`, `packed`, `scales`, `biases` et `dims` sans vérifier qu'ils s'accordent, alors que `write_raw` (lignes 246-273) fait ces contrôles. `RawTensor::to_f32` calcule `n = dims.product()` sans contrôle de dépassement, fait `Vec::with_capacity(n)`, divise par `q.group` et indexe `packed[i]`, `scales[g]` sans borne. `sealed::load` appelle `to_f32` avant le contrôle de forme de candle.
- Scénario : un `.bin` modifié passé à `mmlu`, `ppl` ou `run` : `group = 0` donne une division par zéro (panique), un `packed` court donne un indice hors borne (panique), `dims = [2^40]` donne une allocation dictée par le fichier (abandon du processus). Rust sûr, donc pas de corruption mémoire. Les chemins servis valident déjà ces longueurs (`fused_cuda.rs:1743-1751`, `fused_cpu.rs:278-295`, `fused_metal.rs:1650-1664`) et le runtime ne charge qu'un fichier dont le sha256 est épinglé.
- Touche : format (lecteur seulement)
- Correction proposée : dans `read_raw`, appliquer à la lecture les contrôles de `write_raw` (largeur 4 ou 8, `group > 0`, produit des dims vérifié par `checked_mul`, longueurs de `packed`, `scales`, `biases`), et rendre `Error::Inconsistent`.
- Effort : S
- Risque de la correction : faible. Ne change aucun octet du format ni l'index ; refuse seulement des fichiers que l'écrivain n'a pas pu produire. Suite complète requise, le lecteur du format est touché.
- Vérification proposée : un test dans `llvq-artifact` qui forge un enregistrement `TAG_QUANT` avec `group = 0`, puis un avec `packed` court, et attend `Err` de `read_raw`.

### SEC-07 Les noyaux de gather d'embedding ne bornent pas l'identifiant de token, et rien ne compare le tokenizer à la table
- Gravité : mineure
- Confiance : CONFIRMÉ pour l'absence de garde ; le déclenchement exige un fichier forgé
- Où : llvq-llm/kernels/emb_q4.cu:53-68 ; llvq-llm/kernels/emb_q4.metal:46-60 ; llvq-llm/kernels/emb_q8.cu:53 ; llvq-llm/src/fused_cuda.rs:1705-1770, 1772-1782
- Fait : `emb_q4_gather` lit `row = ids[..]` puis `wq[row * (d >> 3) + ..]` sans comparer `row` à `vocab`, qui n'est pas passé au noyau. `upload_embed` vérifie la cohérence interne de la table, mais aucun chargeur ne vérifie que le tokenizer du fichier (`tokenizer.json`, même fichier) produit des identifiants inférieurs au nombre de lignes de la table.
- Scénario : un fichier scellé dont le `tokenizer.json` a plus d'entrées que la table d'embedding a de lignes, chargé par `fusedrun` ou `mmlu`. Un prompt qui produit un identifiant au-delà de la table fait lire la carte hors du tampon : lecture de mémoire voisine injectée dans le modèle, ou erreur d'adresse illégale. Sur le Qwen3 publié le tokenizer (151 669) est sous la table (151 936), et le runtime ne charge que des fichiers épinglés, donc rien ne se déclenche au service.
- Touche : chemin servi (garde manquante), sans effet sur les fichiers publiés
- Correction proposée : au chargement, refuser un fichier où `tokenizer.get_vocab_size(true)` dépasse `dims[0]` de la table d'embedding. Contrôle côté hôte seulement.
- Effort : S
- Risque de la correction : nul. Aucun noyau ne change, donc aucune empreinte épinglée ne bouge ; les tokens servis restent identiques.
- Vérification proposée : un test sur la fixture mini de `llvq-tetra/tests/fixtures/mini` (ou une table synthétique) avec un tokenizer plus grand que la table, qui attend le refus.

### SEC-08 `run.py publish` téléverse les fichiers ignorés par git du périmètre
- Gravité : mineure
- Confiance : PLAUSIBLE (aucun fichier sensible n'est dans le périmètre de ce worktree aujourd'hui)
- Où : ops/run.py:747-771, 887-889, 908-914 ; .gitignore (ligne `.env`)
- Fait : le périmètre est construit par `args.root.rglob("*")` filtré par `UPLOAD_ALLOW` (`llvq-*/**`, `configs/**`) et `UPLOAD_IGNORE`. Le contrôle de propreté lit `git status --porcelain`, qui ne montre pas les fichiers ignorés. Un fichier ignoré sous `llvq-*/` part donc dans le Space sans être vu. Le `.gitignore` dit que les jetons PyPI et TestPyPI vivent dans un `.env` local ; seul `llvq-tetra/**` est exclu nommément.
- Scénario : un `.env`, un fichier de jeton, un `.llvq` local ou un journal posé sous `llvq-llm/` ou `llvq-runtime/` est téléversé dans le Space de build, public si `--public`.
- Touche : outillage
- Correction proposée : téléverser `git ls-files` intersecté avec le périmètre, ou au moins ajouter `**/.env`, `**/*.env`, `**/*.llvq`, `**/*.bin` à `UPLOAD_IGNORE`.
- Effort : S
- Risque de la correction : faible ; vérifier que les fichiers générés au build ne sont pas attendus.
- Vérification proposée : un test Python qui crée `llvq-llm/.env` dans un arbre temporaire et vérifie qu'il n'est pas dans le périmètre calculé.

### SEC-09 `Checkpoint::var_builder` est une fonction sûre qui porte un contrat `# Safety`
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : llvq-llm/src/loader.rs:311-318
- Fait : la doc dit « # Safety: the checkpoint files must not be modified while the mapping is alive », mais la fonction n'est pas `unsafe fn` : l'appelant n'a aucune obligation à tenir, et le `unsafe` du mmap est absorbé sans condition. 14 appels.
- Scénario : aucun défaut atteignable en pratique (les fichiers du cache Hub sont adressés par contenu). Le contrat affiché ne protège rien.
- Touche : outillage
- Correction proposée : remplacer la section `# Safety` par un commentaire `// SAFETY:` interne qui dit pourquoi le mmap est acceptable ici (fichiers du cache Hub, jamais réécrits en place).
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### SEC-10 Images de base et actions GitHub épinglées par étiquette, pas par digest
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : ops/Dockerfile.cuda:21, 28, 112 ; ops/Dockerfile.check:26 ; .github/workflows/ci.yml (actions/checkout@v4, actions/cache@v4, actions/setup-python@v5)
- Fait : `nvidia/cuda:12.4.1-devel-ubuntu22.04` et `-runtime-` sont des étiquettes ; `apt-get install` n'épingle pas les versions ; rustup s'installe par `curl | sh` en HTTPS (la chaîne d'outils est ensuite épinglée par `rust-toolchain.toml`). Les actions sont épinglées par étiquette majeure. Le workflow a `permissions: contents: read`, aucun secret et pas de `pull_request_target`, ce qui borne l'impact côté CI.
- Scénario : une étiquette déplacée en amont change l'image qui produit les chiffres payés, sans trace dans le journal.
- Touche : outillage
- Correction proposée : épingler les deux images CUDA par `@sha256:` et les actions par SHA de commit.
- Effort : S
- Risque de la correction : nul ; l'image reconstruite doit repasser `oracle`.
- Vérification proposée : aucune, lecture suffisante.

### SEC-11 La règle « unsafe aux seules frontières matérielles » ne nomme pas trois catégories présentes
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : CLAUDE.md (section Architecture) ; llvq-llm/src/lib.rs:7-11 ; llvq-llm/src/tetra_cpu.rs:356-607 ; llvq-llm/src/batch_cuda.rs:104 ; llvq-llm/src/fused_cuda.rs:1860
- Fait : la règle liste mmap, lancement de noyau, lecture d'un tampon d'appareil. Le code a aussi 17 sites d'intrinsèques NEON/AVX2 sur le chemin CPU servi par le runtime, des `unsafe impl DeviceRepr` et des allocations de carte non initialisées (`stream.alloc`). Les sites SIMD sont sains : chaque bloc est précédé d'un `assert!` de longueur (`tetra_cpu.rs:389, 417, 441, 501, 526, 546`) et l'AVX2 n'est atteint qu'après détection (`Isa::detect`).
- Scénario : un relecteur qui applique la règle à la lettre refuserait du code sain, ou laisserait passer un nouveau site hors catégorie.
- Touche : doc
- Correction proposée : ajouter à la règle les intrinsèques SIMD derrière détection de capacité, `DeviceRepr` sur une structure `repr(C)` dont la taille est testée, et l'allocation non initialisée entièrement écrite par le noyau qui suit.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : aucune, lecture suffisante.

### Déjà signalés, toujours présents
- CONNU-NON-CORRIGÉ : `[workspace.lints]` absent, donc `forbid(unsafe_code)` ne couvre ni les tests d'intégration des cinq crates cœur, ni les 20 binaires et 26 exemples de `llvq-bench` ; l'étape CI ne vérifie que `lib.rs` et `main.rs`. Aucun `unsafe` n'y figure aujourd'hui (`docs/audit-architecture-2026-11.md` §1.5 et lot 15).
- CONNU-NON-CORRIGÉ : commentaires `SAFETY` toujours partiels, 59 occurrences dans `llvq-llm/src` et `llvq-cuda/src` pour plus de 140 sites (`docs/audit-architecture-2026-11.md` §1.4 et lot 15).
- CONNU-NON-CORRIGÉ : un client connecté qui ne lit plus bloque le moteur, `blocking_send` sans délai à `llvq-runtime/src/engine.rs:537` (`docs/runtime-audit-2026-11.md`, « Found and not fixed »).

## Vérifié sans constat
- `from_raw_parts` de `e1v_host.rs:88` et des deux tests e1v : `GpuBlockRec` est `repr(C)`, 108 octets, alignement 4, sans remplissage implicite (8 f32 + 8 f32 + 32 u8 + 4 u8 + u32 + 4 u8), tailles assertées à la compilation (`blockrec.rs:132-133`).
- `disable_event_tracking` du chemin servi (`fused_cuda.rs:3870-3878`) n'est appelé que si le contexte n'a qu'un flux ; candle `Device::new_cuda` prend le flux par défaut (`candle-core-0.9.2/src/cuda_backend/device.rs:278-279`).
- Les dimensions des enregistrements sont confrontées à la configuration avant tout lancement 4E (`persist4e_cuda.rs:888-919`), les longueurs lattice et int4 à la lecture (`format.rs:1048-1069, 1260-1310`), les mots Tetra masqués à 47 bits et la table des normes masquée à 32 entrées (`tetra48.rs:47-53, 154-165`).
- Les opérateurs torch de `llvq-tetra` vérifient type, contiguïté, appareil et longueurs avant le lancement (`tetra_cuda.cu:52-71`, `tetra_decode.mm:78-163, 213-230`) ; le tenseur `tail` n'est contrôlé qu'en type côté CUDA, mais sa forme est imposée par le tampon enregistré (`modules.py:69-71`).
- Aucun en-tête PEP 723 de `ops/` n'a de dépendance non épinglée hors `numpy<2.4` (`ops/hybrid_checkpoint.py`).
- `ModelName` refuse `/`, `..` et les noms qui commencent par un point avant toute formation de chemin (`library.rs:17-38`) ; `pull` ne forme son URL qu'à partir de la bibliothèque compilée.

## Candidats au lot 1
SEC-01, SEC-03, SEC-04 (avertissement seul), SEC-05, SEC-07 (contrôle hôte seul), SEC-08, SEC-09, SEC-11. SEC-06 est d'effort S et ne change pas le format, mais touche le lecteur de `llvq-artifact` : à passer seulement avec la suite complète.
