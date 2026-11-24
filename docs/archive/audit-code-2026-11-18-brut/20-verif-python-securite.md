# Vérification contradictoire : Python, outillage et sécurité

Constats vérifiés : PY-01 à PY-09, PY-12, PY-14 de `12-python-ops.md` ; SEC-01, SEC-03, SEC-05, SEC-07 de `08-securite.md`.
Base : worktree `audit-code` à `925930d`. Lecture seule, aucun cargo, uv ni pytest. Arithmétique refaite par `python3 -c`.

| ID | Verdict | Gravité retenue | Raison |
|---|---|---|---|
| PY-01 | NUANCÉ | mineure | Débordement réel, mais seulement pour `LLVQ_HF_TILE` > 64, variable documentée « no range check » dans `docs/ENV.md` seul, et le seul job qui la pose met 64 |
| PY-02 | NUANCÉ | mineure | Aucun lecteur ne lit `llvq_layout_version`, mais `LAYOUT_VERSION = 1` est le seul qui existe : défaut préventif, aucun fichier ne le déclenche |
| PY-03 | NUANCÉ | mineure | Arithmétique exacte (estimation 1,0 à 1,33 h donne un timeout ≤ l'estimation), mais les encodages réels passent par `bench` depuis septembre |
| PY-04 | NUANCÉ | mineure | 107 des 133 scripts qui lancent l'image épinglent le sha du Space et refusent s'il a bougé ; « trois scripts sur 163 » est faux |
| PY-05 | NUANCÉ | mineure | L'ordre des commits est vrai, mais le refus cité est inatteignable : les deux recettes ont exactement une ligne `CUDA_COMPUTE_CAP` |
| PY-06 | CONFIRMÉ | mineure | `run_uv_job` part ligne 1197, le coût s'imprime ligne 1229 |
| PY-07 | CONFIRMÉ | mineure | `cmd_oracle` n'appelle pas `cap_ok` et `--flavor` accepte `a100-large` ; perte bornée à 20 min |
| PY-08 | CONFIRMÉ | mineure | `--calib` refuse `dclm-edu` ; le dégât est faible car les encodages publiés passent par `bench` |
| PY-09 | CONFIRMÉ | mineure | `target_device` réduit `cuda:1` et `{"": 1}` à `"cuda"`, puis `model.to("cuda")` |
| PY-12 | NUANCÉ | cosmétique | `strict=False` est vrai, mais la porte servie épingle un seul fichier par sha256 et `check.py` compare le fixture de bout en bout |
| PY-14 | CONFIRMÉ | mineure | Avec deux arguments sur trois, la porte du 4B est sautée et le script rend 0 |
| SEC-01 | NUANCÉ | mineure | Fait exact, mais exige une compromission du registre amont ; les trois scripts sont des campagnes closes ; l'écriture du bucket ne dépend pas du jeton |
| SEC-03 | CONFIRMÉ | mineure | `stop` non borné, `chat::scan` sur tout le texte à chaque pas ; client local seulement |
| SEC-05 | NUANCÉ | mineure | Le commentaire est faux pour une modification volontaire ; l'exposition exige que l'opérateur choisisse un répertoire partagé |
| SEC-07 | CONFIRMÉ | mineure | Aucune borne sur l'id dans `emb_q4_gather`, aucun contrôle tokenizer contre table ; exige un fichier forgé, et le runtime ignore `context` |

## PY-01

Ce qui est vrai.
- `llvq-tetra/llvq_tetra/metal.py:129` lit `LLVQ_HF_TILE` (défaut 64, aucune borne) et `:137` compile `-DTILE_BLOCKS={tile}u`.
- `integration.py:27` fixe `TILE_BLOCKS = 64`, passé à `FusedTetraLinear.from_loaded` à `:209`, gardé dans `self._tile` (`fused.py:171`).
- `fused.py:211-214` passe `self._tile` à `torch.ops.llvq_cuda.tv_tetra48`.
- `csrc/tetra_cuda.cu:76` alloue `tile · 24 · 4` octets de mémoire partagée dynamique, donc 6 144 octets.
- `data/kernels/llvq-llm/kernels/tv_tetra48_h.cu:61,67-74` indexe `xs[i]` pour `i < (jhi - jlo) · 24`, avec `jhi - jlo` jusqu'à `TILE_BLOCKS`.

Ce qui nuance.
- Le débordement n'existe que pour une valeur > 64. Avec 32 le noyau stage 3 072 octets dans 6 144 alloués : sortie juste, mémoire perdue.
- Le chemin Metal n'est pas touché : `fused.py:220` passe `metal.tiled_source(self._tile)`, une seule source.
- La variable n'est citée que par `docs/ENV.md:107` (« no range check ») et `ops/jobs/hf-cuda-4b.sh:63`, qui pose 64. Le README de `llvq-tetra` ne la nomme pas. Un utilisateur PyPI ne la connaît pas.
- Déclenchement : un opérateur qui pose une variable interne au-delà du défaut, sur le bras CUDA du paquet, hors chemin servi. Mineure, pas majeure.

Correction sûre pour un lot 1 ? oui. Refuser par nom dans `_extension_cuda` toute valeur différente de `integration.TILE_BLOCKS` est local au paquet Python, ne touche aucun fichier de noyau, aucun format ni token servi.

## PY-02

Ce qui est vrai.
- `quantizer.py:54` stocke `llvq_layout_version` sans le lire ; `:65-70` ne vérifie que `code_order` et `records`.
- `reader.py:38-53` (`PackedModel`) vérifie `quant_method`, `code_order`, l'empreinte et `block_dim`, pas la version.
- `modules.py:66-72` crée les buffers en `torch.empty`.
- `llvq-llm/src/hfpack.rs:64-67` : `LAYOUT_VERSION: u32 = 1`, « bumped when a tensor name or a byte convention changes ».
- Le contrôle des clés manquantes n'est fait que par `gentokens.py:137-143`, `ops/hf_cleanroom_check.py:53-56` et les tests (`tests/test_load.py:58-66`).

Ce qui nuance.
- `PackedModel` refuse un tenseur absent (`reader.py:68`), donc la moitié « non initialisé » ne vaut que pour le chemin `transformers`.
- Aucune version 2 n'existe. Le scénario suppose un `hfpack` futur. Les fixtures et les trois fichiers publiés portent 1.
- Les versions 0.1.0 et 0.2.0 sont sur PyPI et ne changeront pas : la garde ne protège que les versions suivantes.
- Je n'ai pas pu relire `transformers` 5 (pas de source locale) pour confirmer qu'une clé manquante n'est qu'un avertissement ; le commentaire de `hf_cleanroom_check.py:50-52` le dit pour le cas « méthode non enregistrée ».

Correction sûre pour un lot 1 ? oui pour le refus par nom de `llvq_layout_version != 1` dans `LlvqConfig.__init__` et `PackedModel.__init__`. Non pour le refus des clés manquantes dans `_process_model_after_weight_loading` : il faut d'abord établir que ce crochet voit les clés manquantes dans `transformers` 5, ce que personne n'a lu.

## PY-03

Ce qui est vrai.
- `ops/run.py:682` : `timeout=f"{max(1, int(wall * 1.5))}h"`. `int` tronque.
- Recalcul (`python3 -c`), estimation en heures, timeout, rapport timeout/estimation :
  0,5 → 1 h (2,00) ; 0,7 → 1 h (1,43) ; 1,0 → 1 h (1,00) ; 1,3 → 1 h (0,77) ; 1,33 → 1 h (0,75) ; 2,0 → 3 h (1,50) ; 2,6 → 3 h (1,15) ; 3,9 → 5 h (1,28).
- Sur [1,0 ; 1,33[ h le timeout est inférieur ou égal à l'estimation, qui exclut déjà les passes avant (`run.py:575`, docstring `:138-140`).
- L'API prend une seule unité, et les minutes sont acceptées (`_timeout_minutes`, `run.py:1267-1283`). La correction en minutes proposée est valide.

Ce qui nuance.
- Les encodages récents ne passent pas par `launch`. `ops/jobs/encode-14b.sh` et les encodages 4B de novembre (`docs/data/jobs.csv`, 2026-11-13 et 14) passent par `bench`, qui exige un `--timeout` explicite (`run.py:1606`). `launch` reste documenté (`ops/README.md:144`).
- Dégât quand il se déclenche : un encodage tué et facturé sans artefact. Majeur pour ce run, rare aujourd'hui. Mineure retenue, majeure si `launch` redevient le chemin d'encodage.

Correction sûre pour un lot 1 ? oui. `math.ceil(wall * 1.5 * 60)` en minutes est local à `run.py`, qui n'entre pas dans l'image.

## PY-04

Ce qui est vrai.
- `ops/Dockerfile.cuda` n'a aucun `COPY --from=build /src/COMMIT` ; le commentaire de `run.py:921-924` le reconnaît.
- `llvq-runtime/build.rs` rend `unknown` sans `.git`, et `.git` n'est pas dans `UPLOAD_ALLOW` (`run.py:747-763`).

Ce qui réfute le scénario.
- 133 scripts de `ops/jobs/` lancent `llvq-runner-cuda`. 107 d’entre eux lisent le sha du Space au lancement et refusent s'il diffère de `IMAGE_SHA`, par exemple `ops/jobs/kv8-memory.sh:70-74` :
  `[ "$NOW" = "$IMAGE_SHA" ] || { echo "refused: the Space is at $NOW, expected $IMAGE_SHA" ...`
  et écrivent `image $NOW` dans le journal. `served-14b.sh:64,119` exige aussi un `IMAGE_SHA` de 40 caractères.
- Le sha du Space renvoie au fichier `COMMIT` du dépôt Space, qui nomme le commit LLVQ (`run.py:930-949`). La chaîne de provenance existe, hors conteneur.
- « Trois scripts sur 163 mentionnent la provenance » est faux. Restent 26 scripts sans garde, et une fenêtre entre la lecture du sha et le démarrage du job.

Gravité : mineure. Correction sûre pour un lot 1 ? non. Le `COPY` change la recette, donc une reconstruction d'image et un job payant pour la vérifier ; le gabarit des jobs change le sha256 de scripts que des préregs nomment.

## PY-05

Ce qui est vrai.
- `run.py:908-915` `upload_folder`, `:944-949` `COMMIT`, `:957-963` le refus « exactly one CUDA_COMPUTE_CAP line », `:971-976` le `Dockerfile`. Le refus arrive après deux commits.

Ce qui nuance.
- `grep -c "ENV CUDA_COMPUTE_CAP="` donne 1 pour `ops/Dockerfile.cuda` et 1 pour `ops/Dockerfile.check`. Le refus cité est inatteignable avec les recettes du dépôt.
- Le risque qui reste est un échec réseau entre deux commits. Un état mixte change le sha du Space, et les 107 scripts qui épinglent ce sha refuseraient de partir (voir PY-04).

Gravité : mineure, basse. Correction sûre pour un lot 1 ? oui pour déplacer la validation avant `create_repo`. Le commit unique est aussi local à `run.py`, mais demande un `publish` réel pour être vu.

## PY-06

Confirmé.
- `run.py:1197` appelle `run_uv_job(...)`. Le coût et `_timeout_minutes` viennent à `:1227-1231`. `_timeout_minutes` lève `SystemExit` sur `2h30m` (`:1279-1282`).
- Chaîne : `run.py dequant --timeout 2h30m` → `cmd_dequant` → `run_uv_job(timeout="2h30m")` → puis `SystemExit` avant l'impression de l'id.
- `--timeout` a un défaut `4h` (`:1635`), donc le cas courant imprime son coût, mais après le lancement. La règle dure 1 n'est pas tenue.

Correction sûre pour un lot 1 ? oui. Déplacer le bloc du coût avant `run_uv_job`, comme `cmd_bench` (`:1127-1134`).

## PY-07

Confirmé.
- `cmd_oracle` (`run.py:999-1030`) appelle `device_ok` seulement. `--flavor` a `choices=sorted(FLAVORS)` (`:1584`), qui contient `a100-large` (cap 80) et `t4-medium` (cap 75).
- `oracle` est de la famille candle, liée au PTX sm_89 (`run.py:73-76`). `run.py oracle --flavor a100-large --device cuda` démarre, facture, puis échoue au premier noyau.
- `cmd_bench` : `--flavor` sans `choices` (`:1593`) ; `cap_ok` rend `True, "CPU"` pour une saveur absente de `FLAVORS` (`:113-115`) ; `--any-flavor` lève la liste blanche (`:1080`) ; le coût n'est imprimé que si `usd_h` existe (`:1128-1134`).
- Perte bornée : `oracle` a un timeout de 20 min (`:1025`), 0,83 $ au plus sur `a100-large`.
- Aucun script de `ops/jobs/` n'utilise une saveur hors `FLAVORS` (l40sx1, a100-large, h200, rtx-pro-6000 seulement), donc refuser une saveur inconnue ne casse rien.

Correction sûre pour un lot 1 ? oui.

## PY-08

Confirmé.
- `run.py:1543-1544` : `--calib`, défaut `c4`, choix `c4`, `wikitext2`, `wikitext2-test`. Le commentaire `:1538-1542` demande de garder la liste au pas de `CalibCorpus`, qui accepte `dclm-edu` (`llvq-llm/src/bin/smoke.rs:22,1948`).
- `fetch_config` lit `raw/main` (`run.py:247-251`). `LLVQ_DATASET_REV` existe côté Rust (`llvq-llm/src/corpus.rs:44`) et `launch` ne le transmet pas (`run.py:598-602`).
- Nuance : les encodages réels passent par `bench` avec l'environnement écrit par le script, donc le dégât est une dérive de liste, pas une mesure faussée.

Correction sûre pour un lot 1 ? oui pour ajouter `dclm-edu` aux choix. Non pour le défaut et les révisions : changer le défaut est une décision d'opérateur, et la révision du modèle demande de vérifier ce que `smoke` sait épingler.

## PY-09

Confirmé.
- `integration.py:49-64` : pour `"cuda:1"`, `1` ou `{"a": 0, "b": 1}`, la boucle ajoute `"cuda"` à `kinds` et rend `"cuda"`.
- `quantizer.py:109-112` charge sur le CPU, puis `integration.py:169-170` fait `model.to("cuda")`, l'appareil courant.
- `fused.py:152-157` envoie les buffers sur `device = "cuda"`. Les caches `metal.invnorm_on(dev)` et `shared_tables(dev)` sont indexés par le genre (`fused.py:203-204`).
- Aucun test ne couvre un index non nul. Un dictionnaire sur deux cartes passe la garde « single device » (`:62-63`) parce que les deux valeurs donnent le même genre.

Correction sûre pour un lot 1 ? oui dans sa forme minimale : refuser par nom un index autre que 0 ou plusieurs index distincts. Non pour la correction complète (porter un `torch.device` dans tous les caches), qui touche quatre fichiers et demande une machine à deux cartes pour être vérifiée.

## PY-12

Ce qui est vrai.
- `ops/mlx_tetra/llvq_mlx.py:754` : `model.load_weights(weights, strict=False)`. Les noms f16 sont pris de `pm.raw` (`:733-736`).

Ce qui nuance.
- La porte servie n'accepte qu'un objet, par sha256 d'artefact : `ops/mlx_tetra/served.py:30-32,98-100` (`886391a8…`, le 4B scellé). Ce fichier est celui de toutes les mesures M16.
- `check.py:200-210` construit le fixture mini et, pour deux couches au plus, compare le passage entier à `transformers` (`section_d`). Une norme ignorée sur le fixture y serait vue.
- `strict=True` ne marcherait pas tel quel : les poids int4 et l'embedding sont posés à la main et absents de `weights`. La correction demande d'établir l'ensemble attendu, ce que le constat ne fait pas.
- Reste un trou sur les outils (`prof.py`, `strip.py`, `check.py` hors fixture) pour un fichier non épinglé. Cosmétique.

Correction sûre pour un lot 1 ? non. L'ensemble attendu reste à définir, et la porte MLX a ses mutants (`mutate.py`) à rejouer sur Mac.

## PY-14

Confirmé.
- `ops/hf_cleanroom_check.py:90-92` : `rc = fixture(...)` puis `if rc or len(argv) < 4: return rc`.
- Avec `script fixture packed` (argv de longueur 3), le fixture passe, la porte du 4B est sautée, le script rend 0 sans le dire.
- La forme à un argument est légitime (docstring `:18-21`, le relecteur sans téléchargement). La forme à deux ne l'est pas.

Correction sûre pour un lot 1 ? oui. Refuser une longueur d'argv de 3 et imprimer « 4B gate not run » pour 2.

## SEC-01

Ce qui est vrai.
- `ops/run.py:1147` (`bench`) et `:1224` (`dequant`, image `uv` tierce par défaut) passent `secrets={"HF_TOKEN": ...}` dès que la variable existe dans le shell, quelle que soit l'image.
- `ops/jobs/iq2-s-m.sh:18-21`, `iq2m.sh:20-23`, `paper-iq2-census.sh:85-88` lancent `ghcr.io/ggml-org/llama.cpp:full-cuda` avec `--bucket` sur la racine du bucket des jobs et `--out-mount /out`. Le volume est monté `read_only=False` (`run.py:1117-1119`).
- `duel-iq2.sh:39` utilise `nvidia/cuda:12.8.1-devel-ubuntu24.04`, une étiquette de version ; les jobs vLLM `vllm/vllm-openai:v0.26.0`.

Ce qui nuance.
- Le scénario exige une compromission du registre ou du dépôt llama.cpp. Probabilité basse, impact fort.
- Les trois scripts sont des campagnes closes (dernier commit 2026-09-25). Aucun script récent ne lance `:full-cuda`.
- La présence du jeton dépend du shell. Le droit d'écrire le bucket, lui, ne dépend pas du jeton : le volume suffit. La correction « pas de jeton par défaut » ne ferme donc pas la réécriture des journaux.
- Retourner le défaut du jeton pour tous les `bench` peut faire échouer après facturation un job de notre image qui lit un dépôt privé (`dclm-14b-rowscales.sh`, `dclm-8b-rowscales.sh` nomment `HF_TOKEN`).

Gravité : mineure. Correction sûre pour un lot 1 ? oui pour une partie : ne pas passer `HF_TOKEN` quand `--image` n'est pas une image `hf.co/spaces/<namespace>/llvq-runner*`, dans `bench` et `dequant`. Non pour le retournement du défaut, ni pour le sous-chemin du bucket, qui change où des scripts écrivent.

## SEC-03

Confirmé.
- `llvq-runtime/src/api.rs:135` et `openai.rs:253` appellent `stop_list`, qui accepte un tableau de toute taille et des chaînes de toute longueur (`api.rs:143-153`).
- `engine.rs:458-459` : à chaque pas, `self.content.full(tok)` puis `chat::scan(&ftext, stop)` ; appelé par `:689`, `:1033`, `:1195`.
- `chat.rs:89-101` : un `text.find(s)` par chaîne sur tout le texte, puis un `ends_with` par préfixe de chaque chaîne.
- Aucun `DefaultBodyLimit` dans `llvq-runtime/src`, donc la limite d'axum de 2 Mo tient. Recalcul : 2 Mo / 11 octets par chaîne de 8 caractères donne environ 190 000 chaînes ; sur un texte de 20 Ko, de l'ordre de 3,8 Go comparés par pas.
- Le serveur écoute en boucle locale par défaut ; seul un client local, ou un hôte du réseau si l'opérateur lie `0.0.0.0`, peut le faire. Un client local peut déjà occuper le moteur avec une requête longue légitime ; ici il le fait des heures avec une seule requête. Le champ `context` d'Ollama est ignoré (`api.rs:839-841`), donc ce n'est pas un autre point d'entrée.

Correction sûre pour un lot 1 ? oui. Un 400 au-delà de 16 chaînes ou 256 octets ne touche pas les tokens. Vérifier que `tests/openapi.rs` ne fige pas le schéma de `stop` ; s'il le fait, régénérer le document par `LLVQ_UPDATE_OPENAPI`.

## SEC-05

Ce qui est vrai.
- `llvq-cuda/src/modcache.rs:92-96` écrit l'en-tête avec le sha256 du cubin ; `:100-121` vérifie le corps contre l'en-tête du même fichier.
- Le commentaire `:22-25` dit « a file edited by hand reads as a miss ». Faux pour une modification volontaire : la clé est déterministe (`:60-70`) et la somme se recalcule.
- `dir_from` (`:178-200`) n'exige qu'un chemin absolu, sans contrôle de propriétaire ni de mode.
- `llvq-cuda/src/gpu.rs:357-392` charge le cubin trouvé sans recompiler. Chemin servi : la config chain du 4B (`configs/qwen3-4b-tetra-e4-4e-v9-chain.json`).

Ce qui nuance.
- Le défaut `~/.cache/llvq/modules` appartient à l'utilisateur. L'attaque exige que l'opérateur pose un répertoire partagé en écriture.
- Le seul usage hors défaut dans le dépôt est `ops/jobs/chain-prefill-served.sh:89` : `MC=$O/modcache`, sur le montage du bucket. Un contrôle de mode « comme ssh » risque de refuser ce montage, dont le propriétaire et les droits ne sont pas ceux d'un disque local.

Gravité : mineure (le commentaire faux seul serait cosmétique). Correction sûre pour un lot 1 ? oui pour corriger le commentaire. Non pour le contrôle de droits : il touche le chargement du chemin servi et peut refuser le cache d'un job de mesure sur bucket.

## SEC-07

Confirmé.
- `llvq-llm/kernels/emb_q4.cu:53-68` : `row = ids[ids_off + blockIdx.x]`, puis `wq[row * (d >> 3) + ...]` et `scales[s0 + ...]`, sans `vocab` en argument.
- `fused_cuda.rs:1705-1770` (`upload_embed`) vérifie la cohérence interne de la table (bits, groupe, `d`, `vocab % 8`, longueurs), pas l'accord avec le tokenizer.
- Aucun `get_vocab_size` dans `llvq-llm/src` ni `llvq-runtime/src` (`git grep`).
- Les ids générés viennent des logits de la tête, donc d'une ligne de la table. Seuls les ids du prompt, venus du tokenizer, peuvent dépasser.
- Le runtime ignore le champ `context` d'Ollama (`api.rs:839-841`) : aucun client ne pousse d'ids bruts. Il ne charge que des fichiers épinglés par sha256. Sur Qwen3, 151 669 entrées de tokenizer contre 151 936 lignes.
- Déclenchement : un fichier forgé chargé par `fusedrun`, `mmlu` ou un banc. Mineure.

Correction sûre pour un lot 1 ? oui. Un contrôle hôte au chargement, tokenizer contre `dims[0]`, ne change aucun noyau, aucune empreinte épinglée ni aucun token sur les fichiers publiés.
