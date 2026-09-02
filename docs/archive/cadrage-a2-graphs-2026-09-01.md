# Note de cadrage — A2 : CUDA Graphs sur la boucle token de `fusedrun`

Préreg : `proofs/preregistration-a2-a3-geometrie-2026-08-31.md` (sha256 `802006c5…`, vérifié sur le fichier), §4. Critères gelés : **≥ 8 % bout-en-bout → adopté ; < 3 % → clos** ; entre les deux : point de courbe. Règle transposée de `check_fuse` (`llvq-llm/src/fused.rs:563-589`) : les deux bras de l'A/B graph portent la **même** préallocation KV — le graph est la seule variable, et la prealloc se mesure dans son propre A/B **avant**. Décision d'opérateur `deaa449` : A2 se fait quoi qu'il arrive ; les priors sont défavorables et déclarés (§5 ci-dessous).

---

## 1. La séquence de lancements d'un token, telle qu'elle est

Config servie v1 (`planes14` + `LLVQ_EMBED=q8` + `LLVQ_ROT_SHARE=1` + `LLVQ_FUSE=1`), Qwen3-4B, décode `l = 1`, `LLVQ_KV` au défaut f16.

**Nos lancements (famille cudarc, via `llvq_cuda::gpu::Cuda`)** — *mesuré* pour les 288 (imprimés par `fusedrun` sur la ligne de bras, calculés par `rotplan::rot_launches` / `matvec_launches_per_token`, `llvq-llm/src/fused_cuda.rs:1688-1690`), *dénombré à la lecture du code* pour les 2 restants :

| noyau | par token | site |
|---|---|---|
| `rot_apply` | **144** (36 couches × 4 sites : groupe q+k+v, `o_proj`, groupe gate+up, `down_proj`) | `RotOp`/`RotSegOp`, `fused_cuda.rs:841-882, 1083-1123` |
| `tv_planes_seg_h` | 72 (2 groupes × 36) | `FusedSegOp`, `fused_cuda.rs:1153-1218` |
| `tv_planes_h` | 72 (`o_proj`, `down_proj` × 36) | `FusedOp`, `fused_cuda.rs:913-1055` |
| `emb_q8_gather` | 1 | `EmbedOp`, `fused_cuda.rs:701-743` |
| `tv_q8_h` (lm_head) | 1 (une ligne au décode) | `HeadOp`, `fused_cuda.rs:766-814` |
| **total nôtres** | **290** | |

**Lancements candle (noyaux candle-kernels + cuBLAS)** — *dénombré à la lecture de `Block::forward_cached` (`llvq-llm/src/model.rs:951-1024`), estimé pour les chemins internes de candle (cat, `repeat_kv`)* : par couche, 3 RMSNorm (`ln1`, `q_norm`, `k_norm`), 2 RoPE, ~4 copies pour les deux `Tensor::cat` du cache KV, 2 copies `repeat_kv`, 2 GEMM cuBLAS (q·kᵀ et ·v), 1 échelle, 1 `broadcast_add` du masque, 1 softmax, 2 additions résiduelles, 1 silu, 1 mul ≈ **21/couche** → ~750 sur 36 couches. Hors couches : norme finale, `to_dtype(F32)`, argmax. Un token ≈ **1 050 lancements, dont 290 nôtres** (*estimé*) — une capture qui ne prendrait que notre famille laisserait ~70 % des lancements dehors.

**Le point décisif — un seul stream, mais le mauvais.** `FusedRuntime::new` prend `dev.cuda_stream()` (`fused_cuda.rs:240`) et compile par `Cuda::on_stream` (`fused_cuda.rs:287` ; le commentaire `fused_cuda.rs:28-34` explique que le partage achète l'*ordre*). Le handle cuBLAS de candle est créé sur ce même stream (`candle-core-0.9.2/src/cuda_backend/device.rs`). **Tout le token — les deux familles — passe donc par UN stream : une `cudaStreamBeginCapture` sur ce stream embrasse structurellement les deux.** C'est l'atout d'A2, et il est vérifié dans le code, pas supposé.

Trois obstacles, tous à mécanisme nommé :

1. **Le stream est le NULL legacy.** `fusedrun` crée le device par `Device::new_cuda(0)` (`llvq-llm/src/bin/fusedrun.rs:169`) → `BackendDevice::new` → `context.default_stream()` (`candle-core-0.9.2/src/cuda_backend/device.rs:279`). Le driver refuse de le capturer — c'est exactement le cas que `Cuda::capture` surface (« le driver n'a rien capturé — stream NULL legacy ? », `llvq-cuda/src/gpu.rs:284-309`). Candle expose le remède : `Device::new_cuda_with_stream(0)` (`candle-core-0.9.2/src/device.rs:254` → `cuda_backend/device.rs:255`, `context.new_stream()`).
2. **Le stream frais bascule cudarc en multi-stream mode.** `is_in_multi_stream_mode` passe à vrai dès **un** `new_stream()` (`cudarc-0.19.8/src/driver/safe/core.rs:450-452`), et l'event tracking est actif par défaut (`core.rs:92`) : chaque `arg()` pousse `cuEventRecord`/`cuStreamWaitEvent` autour du lancement (`launch.rs:218-265`) — ce qui coûte (3,67 contre 3,63 µs/lancement, *mesuré* `docs/mesures/a3-graph-2026-08-06.txt`) et **invalide la capture** (c'est ce qui a tué le premier job du 08-06). Remède connu du dépôt : `disable_event_tracking()` avant toute allocation (`gpu.rs:269`), accessible depuis candle via `dev.cuda_stream().context()`. Sûr parce qu'un seul stream est piloté — l'argument de `gpu.rs:241-274` tient tel quel pour `fusedrun`.
3. **Ce qui reste hors stream/hors capture, par token** : l'argmax → `to_scalar::<u32>` (D2H + synchronisation, `model.rs:1249`) ; `Tensor::from_slice(&[next])` (H2D depuis un Vec host temporaire, `model.rs:1254` — un nœud memcpy H2D dans un graph relirait un pointeur mort) ; le masque causal **reconstruit côté host à chaque token** (`causal_mask_offset`, `model.rs:1135-1143`, appelé à `model.rs:1161`). Côté allocations : cudarc alloue en `cuMemAllocAsync` et libère en `cuMemFreeAsync` (`core.rs:1530-1538`, `core.rs:811`) — capturables en nœuds, mais **tout tenseur candle créé pendant la capture et droppé après** émettrait un free illégal sur une allocation possédée par le graph. Il faut une « fermeture token » : les tenseurs du token capturé restent vivants tant que le graph vit, et l'entrée (id de token) comme la sortie (hidden/logits) sont des buffers **stables** écrits/lus hors graph. Le flag `AUTO_FREE_ON_LAUNCH` que pose `Cuda::capture` (`gpu.rs:296`) aide, il ne dispense pas de gérer les drops Rust.

**Frontière de capture proposée** : capturer `embed → 36 blocs → norm → lm_head` ; laisser dehors argmax/`to_scalar`/`from_slice`. Règle de conception : *tout paramètre qui varie par token devient un CONTENU de buffer stable — jamais une forme, jamais un pointeur.* Concerne : l'offset RoPE (`cos.narrow(0, offset, l)`, `model.rs:181-183` — le pointeur serait figé au capture ; remède : rafraîchir hors graph un buffer cos/sin `[1, hd/2]` par deux petits D2D), le masque (contenu rafraîchi dans un buffer `[1,1,1,W]` fixe), l'index d'écriture KV. L'alternative — mise à jour des nœuds — n'existe dans cudarc 0.19.8 **qu'en brut unsafe** (`sys::cuGraphExecKernelNodeSetParams`, `sys/mod.rs:10345` ; aucun wrapper sûr dans `graph.rs`), avec ~1 050 nœuds à retrouver : à écarter en première itération.

## 2. Le site du `Tensor::cat` qui grandit

`KvCache::append`, `llvq-llm/src/model.rs:231-250` : k et v `[b, n_kv=8, seq, head_dim=128]` f16, concaténés **dim 2** (`model.rs:241` et `:245`), appelé de `forward_cached` à `model.rs:999` — stockés *avant* `repeat_kv`, donc 8 têtes KV, pas 32. La doc du site assume le choix (`model.rs:219-222`) : « Concatenation, not a preallocated ring: a ring needs a maximum length decided in advance ». Coût du cat : chaque token recopie **toute** l'histoire — au token 128 du protocole `fusedrun`, ~36 couches × 2 × 128×8×128×2 o ≈ 19 Mo recopiés pour ce seul token (*calculé*) ; le poste est O(contexte) par pas, cumulé O(contexte²).

**Ce que la préallocation à formes fixes exige** — les trois éléments de la mission, chacun avec son mécanisme :
- **fenêtre bornée W** : buffers fixes `[1, 8, W, 128]` par couche (k et v) — W devient une constante de config, ce que le dépôt refusait par principe et que le graph impose ;
- **écriture par index** : `Tensor::slice_set` existe (`candle-core-0.9.2/src/tensor_cat.rs:246`, écriture en place) — c'est le mécanisme exact de `candle_nn::kv_cache::Cache::append` (`candle-nn-0.9.2/src/kv_cache.rs:58-79`) ;
- **masque porteur de validité** : les positions non écrites reçoivent −inf ; formes d'attention constantes (`scores [1,32,1,W]`), donc GEMM, softmax et `repeat_kv` deviennent capturables.

Compatibilité `KvMode::Q8` : `quantize_dequantize` s'applique au **nouveau** k/v avant stockage (`model.rs:234-237`) — inchangé sous slice_set ; l'A/B se fait de toute façon à mode constant (f16, le défaut servi).

**Lien avec le « préfill minimal » de `deaa449`** (plan `docs/plan-apres-depot-2026-08-29.md:132-135`) : lever le refus `MAX_ROWS = 256` (`model.rs:352`, gardes `model.rs:573` et `SegPlan::run`) par chunking de la boucle de lignes. Ce n'est **pas** un prérequis d'A2 (le prompt de `fusedrun` fait ~5 tokens et le préfill reste hors graph, `l > 1`) — mais le préfill remplit la même fenêtre préallouée : le construire sur elle évite de faire le chantier deux fois, et la prealloc est ce que le port P réutilise.

## 3. Ce que le dépôt possède déjà — et ce qui manque

**Possédé :**
- `Cuda::capture` (`gpu.rs:284-309`) : mode `RELAXED`, fermeture de capture sur les deux chemins d'erreur, erreur du body prioritaire sur `CAPTURE_INVALIDATED`, `None` → message nommant le stream legacy, `upload()` avant retour. Prêt à l'emploi sur le stream de candle.
- `Cuda::new_on_fresh_stream` + la leçon `disable_event_tracking` (`gpu.rs:241-274`), payée le 08-06 (premier job graphbench échoué).
- `graphbench.rs` (`llvq-cuda/src/bin/graphbench.rs`) : les trois bras legacy/frais/frais+graph — 3,63 / 3,67 / 2,97 µs/lancement (*mesuré*, `a3-graph-2026-08-06.txt`) — et la règle : *changer de stream change l'objet mesuré ; le bras témoin de l'A/B doit porter le même stream que le bras graph.*
- `fusedrun` : protocole médiane 5 rounds + plage, `LLVQ_FUSE_AB` (deux bras dans un processus), gates de discrimination, impression des comptes rot/matvec par bras — le gabarit exact de l'A/B prealloc puis de l'A/B graph.
- Le témoin anti-bug-de-cache : `generate_uncached` (`model.rs:1266`) + `LLVQ_VERIFY_CACHE=1` (`bin/run.rs:99-110`) — l'unique réponse indépendante à « la prealloc a-t-elle déplacé RoPE ou le masque », car un cache faux produit du texte fluide et différent.
- Dans le lock, pas dans notre code : `candle_nn::kv_cache` (Cache prealloc + `RotatingKvCache` avec `attn_mask`/`positions`) — preuve d'écosystème que slice_set suffit ; notre `KvCache` reste le nôtre (mode q8, stockage pré-`repeat_kv`).
- Le pool à disputer est mesuré : Δ = 0,406 ms / 108 lancements = **3,76 µs/lancement** L40S (*mesuré/calculé*, `a1-nullk-252-144-2026-08-31.txt`), r invariant 0,8158 / 0,8198 entre L40S et A100 (`a4-a100-2026-08-31.txt`).

**Manquant :** un mode prealloc dans `KvCache` (derrière une variable qui refuse les valeurs inconnues — le motif `FusedLayout::parse`, `fused.rs:88-100`) ; un masque à contenu rafraîchi sur buffer fixe (aujourd'hui Vec host + H2D par token) ; un RoPE à position indirecte ; le passage de `fusedrun` à `new_cuda_with_stream` + event tracking off **sur les deux bras** ; la fermeture token (rétention des tenseurs capturés, buffers d'E/S stables). Rien pour la mise à jour de nœuds — non nécessaire si l'indirection par contenu est retenue.

## 4. Découpage en étapes — critère de sortie et coût

| # | étape | critère de sortie | coût |
|---|---|---|---|
| 0 | **Prealloc KV + masque, pré-vol Mac** (0 $) : fenêtre bornée, slice_set, masque par contenu ; testable dense sur CPU/Metal | tokens gloutons **identiques** cat vs prealloc (`LLVQ_VERIFY_CACHE` + tests) ; débordement de fenêtre = erreur nommée, jamais silence | ~1 j |
| 1 | **A/B prealloc-contre-cat, SANS graph** — exigé par le préreg §4, avant toute capture. Config v1, intra-job, seul le mode cache bouge | Δ publié avec plage, tokens identiques ; si la prealloc rend ≥ 8 % seule, c'est déjà un résultat de la ligne, et le graph se mesure ensuite **contre le bras prealloc** | ~0,25 $ (*estimé*, gabarit D1 : 0,24 $ mesuré) |
| 2 | **Capture dans `fusedrun`** : stream frais + event tracking off, capture d'un token de décode, replay N | critère de **correction**, pas de vitesse : `end_capture` ≠ `None`, pas d'`INVALIDATED`, et tokens du chemin graph identiques au chemin sans graph du même processus | 1-2 jobs d'itération, ~0,3-0,5 $ (*estimé* ; précédent : le premier job du 08-06 a échoué) |
| 3 | **A/B graph-contre-non-graph**, même prealloc, même stream, même event tracking des deux côtés | ≥ 8 % adopté · < 3 % clos · entre : point de courbe ; médiane 5 rounds + plage, tokens identiques | ~0,25 $ |

**Les 2-4 j du plan (`plan-apres-depot-2026-08-29.md:149`) tiennent-ils ?** Bas de fourchette seulement. Le chemin critique n'est pas le graph, c'est l'inventaire des variations par token (offset RoPE, masque, index KV, drops de tenseurs) — et **rien de « capture » ne se pré-vole sur le Mac** (le crate ne compile que sur Linux+CUDA) : chaque oubli se découvre en job facturé. 2-4 j = 1 j étapes 0-1, 1-2 j étape 2, 1 j étape 3 + journal — tenable si aucune classe de variation n'exige de toucher un noyau candle ; le RoPE indirect est le dépassement le plus probable (+1-2 j s'il faut dupliquer un rope à nous). Annoncer **3-6 j** est plus honnête. Budget : étapes 1-3 ≈ 0,8-1 $ (*estimé*), sous le plafond de phase de 4 $ (préreg §7) ; chaque itération de debug capture ajoute ~0,2 $.

## 5. Les risques, nommés

1. **La frontière candle/cudarc — le premier.** (a) Le switch de stream change l'objet mesuré : si le bras témoin reste sur le legacy, l'A/B crédite le graph de ce que vaut le stream (leçon des trois bras, *mesurée*) ; (b) l'event tracking multi-stream invalide la capture silencieusement (*mesuré* 08-06) ; (c) les drops de tenseurs candle après capture émettent des `free_async` illégaux — le seul des trois sans précédent mesuré, donc celui qui coûtera un job.
2. **cuBLAS dans la capture** : les 72 GEMM d'attention passent par le handle candle (même stream) ; un workspace alloué paresseusement *pendant* la capture l'invaliderait — la génération de chauffe (déjà au protocole) doit précéder la capture.
3. **Un paramètre resté porté par une forme ou un pointeur** → recapture par token → gain évaporé. La revue de l'étape 2 vérifie les cinq classes (RoPE, masque, index KV, entrée, sortie) *avant* le job.
4. **Le prior honnête, et il est défavorable.** Plafond mesuré 08-06 : le graph récupère 0,66 µs/lancement sur un noyau quasi nul, soit 0,167 ms = 0,8 % d'un token sous la géométrie 252 (*mesuré*) ; F3 : la soumission hôte est déjà recouverte à 0,1-0,2 % (*mesuré*). Transposé à v1 (token ≈ 9,94 ms à 100,6 tok/s, ~1 050 lancements) : 1 050 × 0,66 µs ≈ 0,7 ms ≈ **7 % — un plafond, sous hypothèses déclarées** (économie uniforme sur des lancements plus gros que le noyau nul — elle sera moindre). La fourchette [< 3 % ; ~7 %] chevauche exactement les seuils gelés : **le verdict attendu est « clos » ou « point de courbe », l'adoption exigerait que le graph gagne aussi ce que le banc nu ne portait pas** (pilotage des ~750 lancements candle, allocations async). C'est pour ça qu'on mesure au lieu de fermer sur dossier.
5. **Ce qu'on construit même si c'est un rouge** — et c'est le contrat de `deaa449` : (i) la **prealloc KV**, prérequis de toute capture future, du préfill long et du port P, et qui supprime seule un poste O(contexte²) — elle survit au verdict du graph ; (ii) le chemin stream frais + event tracking off documenté dans le modèle servi ; (iii) le chiffre qui manque au kill de phase (préreg §6) : si A1+A2+A3 < 8 % cumulés, l'axe géométrie **sous candle** est clos *par mesure*, tamponné — pas par lassitude.
6. **A100** : hors périmètre du préreg A2 (critères L40S), mais si un bras A100 s'ajoute, `LLVQ_NVRTC_ARCH=compute_80` **et** l'image sm80 sont tous deux requis (les deux pièges sont documentés, `a4-a100-2026-08-31.txt`, garde-fous).
