# Second tour, chemin Metal et crate llvq-metal : constats
Périmètre réellement lu : en entier `llvq-llm/src/fused_metal.rs` (2 473 lignes, correction BUG-70 comprise), `llvq-llm/src/attn_metal.rs`, `llvq-llm/kernels/llvq_tetra48.metal`, `llvq_tetra48_rows.metal`, `tv_q4_h.metal`, `emb_q4.metal`, `llvq_rot.metal`, `attn_dec.metal`, la fin de `tv_q4_pf.metal` (`tv_q4_metal_rows`) ; `llvq-tetra/llvq_tetra/metal.py`, `mps.py`, `fused.py`, `vocab.py`, `csrc/tetra_decode.mm`, `integration.py` lignes 110-240 ; `ops/llvq-mlx`, `ops/mlx_tetra/served.py`, `serve.py`, `llvq_mlx.py` ; `llvq-metal/src/lib.rs` lignes 1-360, chaque bloc `unsafe` des 15 fichiers qui en portent, les tailles des tampons qu'ils lisent, les miroirs `#[repr(C)]` de `p1host.rs` contre `cascade_uniform.metal` et `binomial_walk.metal`. Dans les dépendances, en lecture : `candle-metal-kernels-0.9.2/src/metal/{commands,command_buffer,encoder,compute_pipeline}.rs`, `candle-core-0.9.2/src/metal_backend/device.rs` (allocation, options de ressource) et `mod.rs:1910-1921` (`to_cpu`). `served.rs:915-1036` et `served_decode.rs:340-400` pour les formes d'attention.
Survolés : `llvq-metal/src/bin/rankbench.rs` (V0, boucle de temps), `p1v0.rs`, `thesis.rs`, `matvec.rs`, `decreal.rs`, `decode.rs`, `hello.rs` aux sites `unsafe` et de lancement ; `llvq-metal/shaders/*.metal` aux déclarations `threadgroup` ; `ops/mlx_tetra/strip.py`.
Non couvert : le corps des shaders de `llvq-metal/shaders/` (bancs P1, archivés), `tv_q4_pf.metal` hors `tv_q4_metal_rows`, `emb_q8.metal`, `ops/mlx_tetra/{check,ppl,mutate,prof,floor,test_served}.py`, `llvq-tetra/llvq_tetra/check*.py`, le reste de `rankbench.rs`, `thesis.rs`, `matvec.rs`, `metalsplit.rs`. Rien n'a été exécuté.
Bilan : 0 critique, 2 majeures, 10 mineures, 2 cosmétiques

## Constats

### R2-MTL-01 Le chemin Metal servi lit un résultat faux en silence quand la commande GPU échoue : candle 0.9.2 ne relit pas le statut après l'attente
- Gravité : majeure
- Confiance : CONFIRMÉ pour le défaut (lu dans la dépendance, chemin atteignable sur toute erreur de commande) ; le déclencheur, une erreur GPU, n'a jamais été observé ici
- Où : `~/.cargo/registry/src/*/candle-metal-kernels-0.9.2/src/metal/commands.rs:239-259` (`ensure_completed`), `:176-204` (`flush_and_wait`) ; `candle-core-0.9.2/src/metal_backend/mod.rs:1910-1921` (`to_cpu`) ; côté LLVQ, tout `metal_fwd` de `llvq-llm/src/fused_metal.rs` et la lecture des logits par l'échantillonneur
- Fait : `ensure_completed` lit `cb.status()` UNE fois, avant d'attendre. Sur `Committed` ou `Scheduled` il appelle `wait_until_completed()` et rend `Ok(())` sans relire le statut. Sur `NotEnqueued` ou `Enqueued` il fait `commit()` puis `wait_until_completed()`, même chose. Le bras `MTLCommandBufferStatus::Error` n'est atteint que si la commande avait déjà fini en erreur avant l'appel. `to_cpu` copie ensuite le tampon et le lit. Aucun gestionnaire de complétion ne lit `error()` ailleurs (`grep CompletedHandler` vide dans les deux crates).
- Scénario : le 14B servi (5,04 Go sur la carte) sur un Mac de 16 Go sous pression mémoire, ou un préremplissage long qui touche le chien de garde GPU de macOS. La commande se termine en `Error` (`kIOGPUCommandBufferCallbackErrorOutOfMemory` ou `Timeout`). candle rend `Ok`, `to_vec1` lit le tampon des logits, qui vient du pool et n'est pas remis à zéro (`fused_metal.rs:29-31` le dit : les octets d'un tenseur précédent). L'échantillonneur choisit un jeton sur ces octets. Le serveur répond un texte faux au lieu d'une erreur. Même effet sur un lancement refusé par le pilote (R2-MTL-03).
- Touche : chemin servi (moteur Rust sur Metal, `llvq` sans MLX), runtime
- Correction proposée : vendoriser `candle-metal-kernels` 0.9.2 par `[patch.crates-io]` et, dans `Commands::ensure_completed`, relire `cb.status()` après chaque `cb.wait_until_completed()` et rendre `MetalKernelError::CommandBufferError(cb.error())` sur `Error`. Signaler en amont. Côté LLVQ rien d'autre : le moteur traite déjà une `Err` de génération comme une erreur de requête.
- Effort : M
- Risque de la correction : faible. Aucun noyau, aucun format, aucun bit servi : seul un chemin d'erreur change. Elle touche une dépendance, donc le `Cargo.lock` et la construction de l'image Mac.
- Vérification proposée : un test `#[cfg(feature = "metal")]` qui lance par candle un `CustomOp1` dont le noyau boucle assez longtemps pour le chien de garde, ou qui dépasse `maxTotalThreadsPerThreadgroup` sans couche de validation, puis `to_vec1` : attendu `Err` après le correctif, `Ok` avant. Mutant : retirer la relecture du statut, le test repasse au vert à tort.

### R2-MTL-02 Aucun hôte Metal ne vérifie que le SIMD-group fait 32 voies, alors que tous les noyaux servis le supposent
- Gravité : majeure
- Confiance : PLAUSIBLE (32 sur toute puce Apple ; la valeur qui tranche est `MTLComputePipelineState.threadExecutionWidth` sur un Mac Intel à GPU AMD ou Intel, qu'Apple documente comme dépendant du GPU)
- Où : `llvq-llm/src/fused_metal.rs:49-50` (`LANES = 32`, constante), `:624-657` (`pipeline`, aucune lecture de la largeur), `:1962-1965` (le chargeur dit viser « Apple silicon or x86 ») ; `llvq_tetra48.metal:336-349` (`warp_sum`, xor 16..1), `:842` (`row = gid >> 5`), `:866` (`lane == 0u` écrit la ligne) ; même hypothèse dans `tv_q4_h.metal:98-104, 238`, `emb_q4.metal:70-76, 98`, `llvq_tetra48_rows.metal` ; `attn_dec.metal:186-189` (groupes de 16) ; `llvq-tetra/llvq_tetra/mps.py:114, 152` ; `llvq-metal/src/lib.rs:117-121` lit `simd_width` mais aucun appelant ne l'affirme
- Fait : la ligne d'un thread est `gid >> 5` et seule la voie 0 du SIMD-group écrit `y[row]`. Le butterfly lit les voies `l ^ 16 .. l ^ 1`. Rien ne compare `threadExecutionWidth` à 32.
- Scénario : `llvq` construit avec `--features metal` sur un Mac Intel à Radeon. Si le pipeline rapporte 64, les voies 32 à 63 d'un SIMD-group portent la ligne impaire mais `thread_index_in_simdgroup` vaut 32..63 : aucune n'est la voie 0, la ligne impaire n'est jamais écrite et garde les octets du pool ; la ligne paire mélange 64 voies. Si le pipeline rapporte 16 (GPU Intel), `simd_shuffle_xor(v, 16)` sort du groupe. Les jetons sont faux, sans erreur.
- Touche : chemin servi (moteur Rust sur un Mac non Apple silicon) ; paquet `llvq-tetra` sur MPS
- Correction proposée : dans `MetalRuntime::pipeline`, juste après `new_compute_pipeline_state_with_function`, lire `pipe.as_ref().threadExecutionWidth()` (trait `objc2_metal::MTLComputePipelineState`) et refuser par nom tout autre que `LANES`, avant l'insertion dans le cache. Même garde dans `attn_metal::compile` (au moins 16, et 32 attendu). Dans `mps.py`, refuser hors `platform.machine() == "arm64"` si l'objet noyau de torch n'expose pas la largeur.
- Effort : S
- Risque de la correction : nul sur le M3 Max (32, aucun bit ne bouge). Hôte seulement, aucun fichier `.metal` touché.
- Vérification proposée : imprimer la largeur dans la ligne `metal:` du chargeur sur le Mac ; mutant `!= LANES` changé en `!= 64` : tout chargement Metal est refusé par nom.

### R2-MTL-03 Sept lancements sur neuf de `fused_metal.rs` ne lisent pas la limite de threads du pipeline
- Gravité : mineure
- Confiance : PLAUSIBLE (sur le M3 Max ces pipelines acceptent 256 puisque le chemin tourne ; la valeur qui tranche est `pipe.max_total_threads_per_threadgroup()` de chaque pipeline sur une autre puce ou un autre compilateur)
- Où : `llvq-llm/src/fused_metal.rs`, liste complète des lancements du fichier :
  - `:762-765` `TetraRows`, `dispatch_threads(d_out·32, 256)` : limite non lue (bras de mesure, `prefill_rows > 1`) ;
  - `:1002-1005` `TetraMatvec`, noyaux `_lutdt`, `dispatch_thread_groups` : limite lue (`:996`) ;
  - `:1034` `TetraMatvec`, matvec Tetra servie `_lut`, groupe `GROUP = 256` : limite non lue ;
  - `:1312-1315` `RotOp`, `rot_apply_tg_metal` : limite lue (`:1308`), correction BUG-70 juste ;
  - `:1356-1359` `RotOp`, rotation servie : limite lue (`:1352`), correction BUG-70 juste ;
  - `:1540-1543` `Int4Rows`, `tv_q4_metal_rows`, 256 : non lue (bras de mesure) ;
  - `:1617-1620` `Int4Op`, int4 servi, 256 : non lue ;
  - `:1777-1780` `GatherOp`, embedding q4 servi, 256 : non lue ;
  - `:1843-1846` `HeadOp`, tête q4 servie, 256 : non lue.
  Hors du fichier : `attn_metal.rs:258-261` (64) et `:271-274` (16), non lues, forme non servie.
- Fait : la grille de chaque lancement est exacte (`d_out` multiple de 8, `vocab·32`, `n_ids·256`), donc aucune dimension partielle n'adresse hors bornes. Seul le groupe de 256 n'est jamais confronté au pipeline.
- Scénario : une puce ou un pilote où le pipeline de `tv_tetra48_metal_lut` annonce moins de 256 threads. Le lancement viole la limite ; sans couche de validation le résultat n'est pas défini, et R2-MTL-01 fait qu'aucune erreur ne remonte.
- Touche : chemin servi (`_lut`, int4, gather, tête) ; bancs pour `TetraRows` et `Int4Rows`
- Correction proposée : une fonction `fn group_of(pipe: &ComputePipeline, name: &str) -> Result<usize>` qui refuse par nom si `pipe.max_total_threads_per_threadgroup() < GROUP`, appelée aux six sites non gardés avant `dispatch_threads`. Refuser plutôt que réduire : les noyaux sont justes à tout groupe multiple de 32 (ils bouclent sur `tgs`), mais le contrat `d_out % 8` de l'hôte est écrit pour 256.
- Effort : S
- Risque de la correction : nul sur le M3 Max. Hôte seulement.
- Vérification proposée : un test `feature = "metal"` qui compile les cinq pipelines servis et affirme `max_total_threads_per_threadgroup() >= 256` en imprimant la valeur ; mutant `GROUP` remplacé par 2048 dans la garde : le chargement est refusé par nom.

### R2-MTL-04 `LLVQ_METAL_KERNEL=ilp` lance un noyau à 8 voies sur une grille de 32 : écritures hors de `y`
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/fused_metal.rs:467-480` (`kernel_named`, `Some("ilp")` à `:474`), `:460-462`, `:532`, `:1996` (le noyau vient de la variable sans `validate`), `:1014` et `:1024-1034` (`tetra_rows_per_simd("…_ilp") = 1`, grille `d_out·32`) ; `llvq-llm/kernels/llvq_tetra48.metal:1030` (`row = gid / LLVQ_ROW_LANES`, 8) et `:1063` (`y[row] = …`) ; `docs/ENV.md:26`
- Fait : `MetalTuning::validate` refuse `tv_tetra48_metal_ilp` (`:177`), mais `kernel_named` l'accepte et `MetalRuntime::new` et `from_tables` le prennent sans passer par `validate`. La grille reste `d_out·32` threads, donc `row` va jusqu'à `4·d_out − 1` : lectures de `words` et de `rscale` hors bornes, écriture de `y[row]` jusqu'à trois fois au-delà du tampon de `d_out` flottants.
- Scénario : `LLVQ_METAL_KERNEL=ilp cargo test -p llvq-llm --features metal --test tetra48_metal_candle`. Les douze tests construisent leur runtime par `MetalRuntime::new` sans `set_tuning` (`tetra48_metal_candle.rs:94, 129, 252, 296, 435, 443`). Le noyau écrit dans d'autres tampons du pool de candle. À noter : la variable n'atteint plus aucun binaire. La porte servie la refuse (`served.rs:576`, `check_env`), et `bin/metalrun` écrase le noyau par `set_tuning` (`metalrun.rs:258`). `docs/ENV.md:26` la décrit pourtant comme un mode.
- Touche : banc et tests Metal
- Correction proposée : retirer `Some("ilp")` de `kernel_named` et le mot `ilp` du message, pour que la variable et `validate` acceptent le même ensemble. Mieux : faire passer `kernel_from_env` par `MetalTuning { tetra_kernel, ..Default::default() }.validate()`. Mettre `docs/ENV.md:26` à jour, ou supprimer la variable si l'opérateur confirme qu'aucun banc ne la lit plus.
- Effort : S
- Risque de la correction : nul. Hôte seulement, chemin servi inchangé (la variable y est refusée).
- Vérification proposée : dans `an_unknown_metal_kernel_is_refused` (`fused_metal.rs:2402`), ajouter `"ilp"` aux valeurs refusées ; mutant : remettre la ligne `Some("ilp")`, le test rougit.

### R2-MTL-05 `MetalTuning::validate` accepte quatre noyaux à sortie f16 et deux à arguments différents, qui rendent des nombres faux sans erreur
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-llm/src/fused_metal.rs:176-182` (le seul filtre est `starts_with("tv_tetra48_metal")` et `!= …_ilp`), `:304-318` (`source_of`), `:925-936` (sortie allouée en F32 quand `half_io` est faux), `:966-969` (seul l'index 0 de mémoire de groupe est dimensionné) ; `llvq_tetra48.metal:718` (`_arh`, `device half* y`), `:933-934` et `:940` (`_lutg`, `[[threadgroup(1)]]` écrit) ; `llvq_tetra48_rows.metal:356` (`_luth`), `:401` (`_lutdh`), `:453-454` (`_lutdr`, tampons 15 et 16), `:653` (`_lutdth`)
- Fait : avec `half_io = false`, `validate` laisse passer `tv_tetra48_metal_arh`, `_luth`, `_lutdh`, `_lutdth` (ils écrivent des `half` dans un tampon que l'hôte lit en `f32`), `_lutg` (sa table de 256 octets à l'index 1 n'est jamais dimensionnée) et `_lutdr` (noyau de préremplissage : `nrows` et `d_out` non liés).
- Scénario : un banc qui écrit `MetalTuning { tetra_kernel: "tv_tetra48_metal_luth", ..Default::default() }` passe `validate` et `set_tuning`, puis chaque projection rend `d_out/2` paires de `half` lues comme `f32`. `bin/metalrun` n'est pas touché, sa liste fermée (`metalrun.rs:247-256`) protège. Le test `a_bad_tuning_is_refused_by_name` (`tetra48_metal_rows.rs:181-192`) ne couvre que `_ilp`, `tv_q4_metal` et un nom inconnu.
- Touche : banc (API publique de mesure)
- Correction proposée : remplacer le filtre de `:177` par une liste fermée : `SERVED_TETRA`, les six de `TETRA_ROW_LOOPS`, et `tv_tetra48_metal`, `_ar`, `_u2`, `_c3`. Tout autre nom est refusé par nom. Les jumeaux f16 ne s'atteignent que par `half_io`, `_lutdr` que par `prefill_rows`.
- Effort : S
- Risque de la correction : nul. Hôte seulement ; la configuration servie (`SERVED_TETRA`) reste acceptée.
- Vérification proposée : ajouter `_luth`, `_lutg`, `_lutdr`, `_arh` à la boucle `bad` de `a_bad_tuning_is_refused_by_name` ; mutant : rétablir `starts_with`, le test rougit.

### R2-MTL-06 `THREADGROUP_LIMIT` n'est jamais comparé à la mémoire de groupe de la machine, contrairement à ce que dit son commentaire
- Gravité : mineure
- Confiance : PLAUSIBLE (32 768 sur toutes les puces Apple silicon ; la valeur qui tranche est `MTLDevice.maxThreadgroupMemoryLength`)
- Où : `llvq-llm/src/fused_metal.rs:73-80`, utilisée à `:170`, `:279`, `:1287`, `:1813` ; aucun appel à `maxThreadgroupMemoryLength` dans `llvq-llm` (`git grep` vide)
- Fait : le commentaire dit « a machine that offered less would have to be refused rather than quietly served a kernel that stages past its limit ». Rien ne refuse : la constante sert de borne aux demandes de l'hôte, jamais confrontée au device.
- Scénario : un GPU qui offre moins de 32 768 octets. L'int4 non tuilé jusqu'à `d_in = 8 192` demande 32 768 octets, la tête q4 du 14B 20 480. `setThreadgroupMemoryLength` dépasse la limite, l'effet n'est pas défini, et R2-MTL-01 masque l'erreur.
- Touche : chemin servi (refus au chargement)
- Correction proposée : dans `MetalRuntime::from_tables` et `MetalRuntime::new`, lire `maxThreadgroupMemoryLength()` du `MTLDevice` (trait `objc2_metal::MTLDevice`, via `dev.device().as_ref()`) et refuser par nom s'il est sous `THREADGROUP_LIMIT`.
- Effort : S
- Risque de la correction : nul sur le M3 Max (égalité). Hôte seulement.
- Vérification proposée : imprimer la valeur dans la ligne `metal:` ; mutant : comparer à `THREADGROUP_LIMIT * 2`, tout chargement Metal est refusé par nom.

### R2-MTL-07 La porte MLX servie lance la rotation à 1 024 threads par groupe, valeur fixe, sans lecture possible de la limite
- Gravité : mineure
- Confiance : PLAUSIBLE (lancé et mesuré sur M3 Max seulement ; la valeur qui tranche est `maxTotalThreadsPerThreadgroup` du pipeline `llvq_rot_tg`, 28 Kio de mémoire de groupe statique et `float col[K]` à K = 19 au 4B)
- Où : `ops/mlx_tetra/llvq_mlx.py:205-260` (`ROT_BODY`, `threadgroup float s[GB * M]`, `ROT_TG_FLOATS = 7168`), `:523-528` (`grid=(1024, rows, 1), threadgroup=(1024, 1, 1)`) ; `ops/mlx_tetra/served.py:132-139` (la porte ne lit que les versions et le mot `gpu`)
- Fait : le noyau boucle partout par `nt = threads_per_threadgroup.x`, avec des barrières uniformes. Sa sortie ne dépend donc pas du nombre de threads. Le 1 024 est un choix de débit, qui est aussi la limite matérielle maximale. C'est la classe de BUG-70, sur la porte MLX.
- Scénario : un Mac M1 ou M2 où ce pipeline annonce moins de 1 024 threads. Selon MLX, refus par exception au premier jeton, ou lancement hors limite. Je ne peux pas trancher sans la source C++ de MLX, absente de l'arbre. La porte imprime pourtant la ligne de la config servie avant.
- Touche : porte MLX servie (outillage Mac)
- Correction proposée : passer le groupe de `ROT` à une constante `ROT_TG = 512` (ou 256) dans le même fichier, sans changer le corps : même arithmétique, mêmes bits. Puis rejouer `ops/llvq-mlx bench` pour le débit. Ou garder 1 024 et le borner par `mx.metal.device_info()` si cette clé existe dans mlx 0.32.4.
- Effort : S pour le code, un banc local pour le débit
- Risque de la correction : faible pour les bits (aucun) ; elle peut changer un débit mesuré (M16), donc décision d'opérateur.
- Vérification proposée : `ops/llvq-mlx check` (les portes de `check.py` comparent `ROT` à `Rotation.reference`) au nouveau groupe, puis `bench` avant et après.

### R2-MTL-08 La porte MLX identifie l'objet par un champ que le packer écrit, et compile les noyaux de l'arbre de travail sans empreinte
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `ops/mlx_tetra/served.py:123-129` (`check_object` compare `quantization_config.artifact_sha256` de `config.json`), `ops/mlx_tetra/serve.py:52-55` ; `ops/mlx_tetra/llvq_mlx.py:68-69, 100-105` (`HEADER` lu dans `REPO/llvq-llm/kernels/`, jamais par `llvq_tetra.metal.shader_source`, qui vérifie le sha256 des copies) ; `configs/mlx/qwen3-4b-tetra-e4-mlx.json` (épingle `mlx`, `mlx_lm` et l'objet, pas les octets lus)
- Fait : la porte ne hache ni `model.safetensors` ni les deux fichiers `.metal`. Un répertoire packé dont le `config.json` porte `886391a8…` passe, quel que soit le contenu des tenseurs. Un `llvq_tetra48.metal` ou `llvq_tetra48_rows.metal` modifié dans l'arbre passe aussi. La porte imprime ensuite « served config: … object qwen3-4b-sealed.bin 886391a8 ».
- Scénario : un `model.safetensors` réécrit par un `hfpack` d'une autre branche, avec le `config.json` gardé ou recopié. Ou une mesure lancée depuis un worktree où un noyau Metal est en cours de modification. La ligne imprimée atteste un objet qui n'a pas tourné.
- Touche : porte MLX servie (outillage)
- Correction proposée : ajouter à la config MLX une clé `packed_sha256`, le sha256 de `model.safetensors`, vérifiée dans `check_object`. Le coût est de hacher 1,4 Go, une à deux secondes. Ajouter aussi une clé `kernels_sha256` sur la concaténation des deux `.metal`, vérifiée avant `import llvq_mlx`. Les deux refus vont dans `test_served.py`.
- Effort : S
- Risque de la correction : faible ; change le schéma de `configs/mlx/*.json`, donc décision d'opérateur.
- Vérification proposée : `test_served.py` avec un répertoire factice dont `model.safetensors` diffère d'un octet : refus attendu ; mutant : sauter la comparaison, le test rougit.

### R2-MTL-09 `Int4Vocab` de `llvq-tetra` ne vérifie ni `bits == 4`, ni `group == 64`, ni `d % 8`, que le shader code en dur
- Gravité : mineure
- Confiance : PLAUSIBLE (aucun fichier publié n'a un autre groupe ; le défaut attend un fichier qui en a un)
- Où : `llvq-tetra/llvq_tetra/vocab.py:23-32` ; `llvq-tetra/llvq_tetra/integration.py:126-140` (seul `encoding == "quant"` est lu) ; `llvq-llm/kernels/emb_q4.metal:59, 62-63, 104, 112` (`d >> 3`, `c >> 6`, nibble) ; à comparer à `llvq-llm/src/fused.rs:222` (le chemin Rust exige `EMBED_GROUP`) et `fused_metal.rs:1662-1669`
- Fait : le noyau lit un groupe de 64 et des mots de 8 nibbles. Le lecteur Python transmet `groups_per_row` du fichier sans contrôler le groupe ni les bits. Le chemin dense du même paquet, lui, utilise `d["group"]` (`conversions.py:44`).
- Scénario : un fichier packé dont l'embedding est int4 g128 (ou g32). `Int4Vocab` charge des tables de forme cohérente, puis le noyau applique l'échelle du groupe `c >> 6` : les rangées gathered et les logits sont faux, sans erreur. Le chemin CPU, qui déquantifie avec le bon groupe, donnerait d'autres jetons.
- Touche : paquet Python (`llvq-tetra` sur MPS)
- Correction proposée : dans `Int4Vocab.__init__`, refuser par nom `desc["bits"] != 4`, `desc["group"] != 64` ou `d % 8 != 0`, avant `register_buffer`.
- Effort : S
- Risque de la correction : nul. Python seulement, aucun shader.
- Vérification proposée : un test pytest sans MPS qui construit `Int4Vocab` avec `group = 128` et attend `ValueError` ; mutant : retirer la garde.

### R2-MTL-10 Les bancs `llvq-metal` ne lisent jamais le statut de la commande, et `time` garde le minimum
- Gravité : mineure
- Confiance : CONFIRMÉ pour le code ; le déclencheur est une erreur GPU
- Où : `llvq-metal/src/lib.rs:152-174` (`dispatch`), `:198-223` (`dispatch_many`), `:237-262` (`dispatch_batch`) : `wait_until_completed` puis rien ; `:164`, `:212`, `:250` (`group.min(max)` réduit le groupe en silence) ; `:316-335` et `:266-286` (`time`, `time_many` : meilleur de `reps`) ; `:145-147` (`read` ne compare pas `len` à `buf.length()`) ; utilisateurs : `rankbench.rs:1208`, `thesis.rs:878`, `matvec.rs:631-645`, `decreal.rs:231-262`
- Fait : aucune lecture de `cmd.status()` ni de `cmd.error()` dans la crate, hors un `Command::status` de processus (`rot_matches_host.rs:245`). Une commande en erreur finit vite. Comme `time` et `time_many` retiennent le minimum, c'est elle qui fournit le chiffre. Les sorties sont vérifiées une fois avant de chronométrer, pas pendant.
- Scénario : un passage du banc `rankbench` (instrument préenregistré P1) où une répétition tombe sur le chien de garde ou sur une faute mémoire. Sa durée, proche du coût de soumission, devient le « best » ou une médiane basse. Autre cas : `dispatch` réduit le groupe à la limite du pipeline sans le dire, et la ligne imprimée garde « 256 ».
- Touche : banc (instruments de mesure P1, `thesis`, `matvec`)
- Correction proposée : dans les trois fonctions de lancement, après `wait_until_completed()`, `assert_eq!(cmd.status(), metal::MTLCommandBufferStatus::Completed, …)` avec le texte de l'erreur. Dans `read`, `assert!(len * size_of::<T>() <= buf.length() as usize)`. Dans `dispatch*`, refuser au lieu de réduire quand `group > max_threads_per_group()`, ou imprimer la valeur réduite.
- Effort : S
- Risque de la correction : faible ; touche un instrument de mesure, donc décision d'opérateur, mais aucune valeur ne change sans erreur.
- Vérification proposée : `cargo run -p llvq-metal --bin hello` vert ; mutant : `assert_ne!` à la place de `assert_eq!`, `hello` doit paniquer au premier lancement.

### R2-MTL-11 Le pilote CUDA compilé par `rot_matches_host.rs` a un chemin fixe dans `$TMPDIR`, partagé entre worktrees
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-metal/tests/rot_matches_host.rs:234-249` (`std::env::temp_dir().join("llvq_host_rotate_msl")`) ; même classe dans `llvq-cuda/tests/decoder_matches_rust.rs:41, 75`, `e1v_decoder_matches_rust.rs:73`, `golay70_decoder_matches_rust.rs:119`, `f1rank*_matches_rust.rs` (nom par `tag`, sans PID)
- Fait : le commentaire de `:234-235` règle la collision avec la crate CUDA, pas celle entre deux checkouts. Le dépôt tourne en plusieurs worktrees (`.claude/worktrees/*`) et deux sessions lancent parfois les suites en même temps.
- Scénario : deux worktrees exécutent `rot_matches_host` à la même seconde, avec deux `host_rotate.cpp` différents. L'un exécute le binaire que l'autre vient d'écrire, ou un binaire à moitié écrit (`Text file busy`, ou une référence d'une autre version) : faux rouge, ou faux vert si les deux versions s'accordent par hasard.
- Touche : tests
- Correction proposée : `std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("llvq_host_rotate_msl")`, propre au répertoire cible de chaque worktree, comme le font déjà `attn_dec_matches_rust.rs:27` et `attn_batch_matches_one_row.rs:11` avec le PID.
- Effort : S
- Risque de la correction : nul.
- Vérification proposée : lecture suffisante ; `cargo test -p llvq-metal --test rot_matches_host` vert.

### R2-MTL-12 La liaison Objective-C++ de `llvq-tetra` garde moins que l'hôte Rust : tuile jusqu'à 4 096, queue non contrôlée sur MPS, cache clé sur un hachage
- Gravité : mineure
- Confiance : CONFIRMÉ
- Où : `llvq-tetra/llvq_tetra/csrc/tetra_decode.mm:144` (`tile <= 4096`) contre `:191` (`setThreadgroupMemoryLength(tile·96)`) ; `:153-154` (`tail` : dtype et contiguïté, pas `is_mps()`, alors que `:146-152` l'exigent des autres) ; `:50-54` (clé de cache `[src hash]/nom`, sans comparaison du texte) ; `llvq-tetra/llvq_tetra/metal.py:176-184` (`tiled_source` n'accepte qu'un `tile > 0`) ; référence côté Rust : `fused_metal.rs:509-515` (32..=256, puissance de deux)
- Fait : une tuile de 342 à 4 096 demande de 32 832 à 393 216 octets de mémoire de groupe, au-dessus des 32 768 d'Apple. `mtl(tail)` convertit la mémoire d'un tenseur CPU en `id<MTLBuffer>` si `tail` n'est pas sur MPS. `NSString.hash` ne hache que les 96 premiers, médians et derniers caractères d'une longue chaîne (documentation de CFString), et une collision rendrait un autre pipeline.
- Scénario : `checkmps.py` (`:128`) ou un appel direct `torch.ops.llvq.tv_tetra48(..., tile=512, ...)` : lancement hors limite, résultat non défini, et MPS ne remonte pas l'erreur à Python. La collision de hachage n'a pas de scénario réaliste aujourd'hui : chaque source utilisée diffère dans ses 32 premiers caractères.
- Touche : outillage (contrôles `checkmps`, `checkq4`), paquet Python
- Correction proposée : `TORCH_CHECK(tile >= 32 && tile <= 256 && (tile & (tile - 1)) == 0, …)`, la règle de `fused_metal.rs:509`. Ajouter `tail.device().is_mps()` à la garde de `:153`. Clé de cache sur le texte complet, `[src isEqual:]` via un `NSDictionary` indexé par la chaîne elle-même.
- Effort : S
- Risque de la correction : nul. Pas de shader touché, mais une nouvelle version du paquet.
- Vérification proposée : un test pytest MPS qui attend `RuntimeError` pour `tile = 512` ; mutant : remettre `4096`.

### R2-MTL-13 `emb_q4.metal` dit que l'hôte affirme `vocab % 8 == 0` ; aucun hôte ne le fait
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-llm/kernels/emb_q4.metal:31-32` ; `llvq-llm/src/fused_metal.rs:1651-1687` (`upload_embed` contrôle `d` et les longueurs, pas `vocab`) ; `llvq-tetra/llvq_tetra/mps.py:346-355`
- Fait : l'affirmation est fausse, et inutile : `dispatch_threads` lance exactement `vocab·32` threads, le dernier groupe partiel est légal et `tgs` donne sa taille réelle. Ce commentaire est de la même classe que BUG-53.
- Scénario : un lecteur qui porte ce noyau vers `dispatch_thread_groups` croit la garde présente et lance des lignes au-delà de `vocab`.
- Touche : doc (shader livré dans la roue avec son sha256)
- Correction proposée : réécrire la phrase (« `dispatch_threads` launches exactly `vocab · 32` threads; no host asserts `vocab % 8`, and none needs to ») au prochain déplacement d'empreinte, puis régénérer la copie par `bin/tetratables`.
- Effort : S
- Risque de la correction : faible, mais elle touche un fichier `.metal` livré et haché : décision d'opérateur.
- Vérification proposée : aucune, lecture suffisante.

### R2-MTL-14 Le chemin Metal de `llvq-tetra` lance le décodeur fidèle `tv_tetra48_metal` et dit lancer le noyau servi
- Gravité : cosmétique
- Confiance : CONFIRMÉ
- Où : `llvq-tetra/llvq_tetra/csrc/tetra_decode.mm:125-126` (« The served kernel, bound »), `:167` (`tv_tetra48_metal`) ; `llvq-tetra/llvq_tetra/mps.py:8-14` (« a variant of a served one »), `:185-187` (variante de `tv_tetra48_metal`), `:173` (`tetra48_dot`, décodage PRMT) ; servi sur Metal : `fused_metal.rs:155-156` (`tv_tetra48_metal_lut`)
- Fait : le paquet Python lance le portage fidèle, avec émulation de PRMT, pas `_lut`, que sert le moteur Rust. Les deux sont égaux au bit par leurs portes (`tetra48_metal_candle.rs`), donc aucun nombre ne change, mais le débit diffère (`fused_metal.rs:896-909` : 19,85 contre 12,66 ms par jeton).
- Scénario : un lecteur compare le débit de `llvq-tetra` sur MPS à celui de `llvq` et attribue l'écart au moteur, alors qu'il vient du décodeur choisi.
- Touche : doc (paquet Python)
- Correction proposée : écrire « the faithful decoder `tv_tetra48_metal`, gated equal to the served `_lut` » aux deux endroits.
- Effort : S
- Risque de la correction : nul (commentaires et docstring).
- Vérification proposée : aucune, lecture suffisante.

## Contrôles faits sans constat

- **Correction BUG-70.** `fused_metal.rs:1305-1315` et `:1349-1359` prennent `min(self.rot.threads, pipe.max_total_threads_per_threadgroup())`, arrondi au multiple de 32 inférieur, et refusent 0 par nom. La grille vaut `threads · rows` pour un groupe de `threads`, donc toujours un groupe par rangée. Les quatre boucles de `llvq_rot.metal` (`rot_load`, `rot_wht_step`, `rot_scale_out`, `rot_mix`, `:138-245`) et `rot_apply_tg_metal` (`:416-438`) avancent par `nthreads`. Aucun bit ne dépend du compte. La correction est juste et complète.
- **Mur int4 à `d_in ≥ 8 192`.** Aucun chemin silencieux. Rust : `int4_tile_for` (`fused_metal.rs:270-284`) envoie au-delà de 32 768 octets vers `tv_q4_metal_tiled`, la tuile est validée à chaque fiche (`:1437`), la longueur de mémoire de groupe est celle de la tuile (`:1612`), et les jumeaux `_h` et `_p` mettent aussi en scène des f32. `tv_q4_metal_rows` lit le device sans mise en scène. Python : `tetra_decode.mm:218-222` refuse par nom au-delà de 32 768, et `mps.py` passe toujours par la variante tuilée ou par les rangées à 1 024 colonnes. MLX : `nn.QuantizedLinear`. La tête q4 refuse `d · 4 > 32 768` par nom (`:1812-1816`).
- **Formes d'attention sur Metal.** Toutes les portes servies passent par `Served::apply`, qui appelle `check_device` (`served.rs:977`) : `engine.rs` par `cfg.serve`, `chat`, `gsm8k`, `mmlu`, `fusedrun`, `batchrun`. Seul `bin/metalrun` atteint `grouped` et `kernel`, par `apply_measurement` (`served.rs:992-1004`, `metalrun.rs:369-372`). `prefill_attention` exige `attention: kernel` (`served.rs:724`), qui est refusé sur Metal. Les trois configs Metal de `library.json` sont `qwen3-*-tetra-e4.json`, en forme `expanded`.
- **Mémoire de groupe déclarée contre dimensionnée.** Chaque `[[threadgroup(0)]]` des noyaux servis reçoit sa longueur : Tetra `tile · 96` (`:966-969`), int4 (`:1612`), tête (`:1838`), rotation tg (`:1300`). Les cinq tables de `_lutdt` sont dimensionnées à `:993-995`. Toutes les longueurs sont multiples de 16 octets. `_lutd*`, `_lutdr`, `tv_q4_metal_rows` et `emb_q4_gather_metal` ne déclarent pas de mémoire de groupe.
- **Structures hôte contre shader.** `attn_metal::Args` fait 40 octets, comme `AttnArgs`, dans le même ordre. `p1host` : `CascadeRec` 88, `DivTab` 600, `WalkRec` 52, tous gardés par des assertions `const`. Les arguments scalaires des noyaux servis sont des `u32` ou `f32` liés par `set_param`, un par index, dans l'ordre du shader. Ordre vérifié index par index pour `tv_tetra48_metal_lut`, `_lutdr`, `tv_q4_metal`, `tv_q4_metal_tiled`, `tv_q4_metal_rows`, `emb_q4_gather_metal`, `tv_emb_q4_metal`, `rot_apply_metal`, `rot_apply_rows_metal`, `rot_apply_tg_metal`, et pour les appels de `mps.py` contre `TETRA_ROWS_KERNEL` et `Q4_ROWS_KERNEL`.
- **Durée de vie et mode de stockage.** Tables et flux sont tenus en `Arc<Buffer>` pour la vie du modèle (`MetalTables`, `MetalTetraProj`, `MetalInt4Proj`, `MetalEmbedTable`). Les temporaires, comme le `scratch` de `RotOp` à `:1322`, retournent au pool dès la fin de l'encodage. Il n'y a pas de course pour autant : candle 0.9.2 alloue en `StorageModeShared` avec suivi des aléas (`device.rs:69-71`, `Untracked` commenté), un fil unique encode dans l'entrée 0 du pool, et un command buffer retient ses ressources. La lecture CPU par `to_cpu` attend la fin (`mod.rs:1919`), mais sans relire le statut (R2-MTL-01).
- **Bornes des shaders servis.** Lectures de table masquées (`s16 & 63`, `i* & 0x7ff`, `m & 31`). `f1r_load` du dernier bloc couvert par le pas arrondi à 8 octets et par le contrôle de longueur de l'hôte (`:585-596`). Pas d'écriture au-delà de `d_out`, puisque les grilles sont exactes. `y_off = r · vocab` en u32 jusqu'à 28 269 rangées (déjà noté au premier tour). Barrières de `_lut`, `_tiled` et `_lutg` uniformes ; `attn_split_metal` mélange par 16 voies alignées dans des SIMD-groups de 32.
- **`unsafe` de `llvq-metal`.** 29 blocs dans 15 fichiers, plus la déclaration `pub unsafe fn read` (`lib.rs:145`), tous des `Kernel::read` ou `from_raw_parts` sur un tampon `StorageModeShared` lu après `wait_until_completed`. Pour chacun, la longueur lue égale la taille allouée (`hello.rs:54`, `decode.rs:224`, `decreal.rs:205`, `p1v0.rs:169, 240, 257, 320`, `rankbench.rs:913, 1057`, `thesis.rs:774`, `matvec.rs:512`, et dans les tests). Restent deux défauts : le statut non lu et `len` non borné par `buf.length()` (R2-MTL-10). `rankbench` et `p1v0` affirment que le nombre de blocs est un multiple du groupe (`rankbench.rs:1040-1043`, `p1v0.rs:227-233`), ce qui protège la mise en scène `if (tid < DIM)` des shaders P1.
- **Python, appareil et synchronisation.** `FusedTetraLinear`, `Int4FusedLinear` et `Int4Vocab` lancent sur le flux MPS de torch. La lecture côté CPU passe par `.cpu()`, qui synchronise. `RotationKernel` garde son résultat par identité d'objet et par version (`mps.py:418-436`), ce qui reste correct.

## Toujours présents d'un ancien audit

- CONNU-NON-CORRIGÉ : BUG-53, `tv_q4_h.metal:45-63` et `:246-249` disent toujours qu'aucun hôte ne vérifie les contrats du noyau (fichier livré et haché).
- CONNU-NON-CORRIGÉ : BUG-72, `llvq_rot.metal:97-99` dit toujours « 25600 = 512 · 50 and does NOT fit ».
- CONNU-NON-CORRIGÉ : PY-10, en partie. `fused.py:223-251` garde la branche qui appelle `torch.ops.llvq.tv_tetra48`, op Metal, sur un appareil qui n'est ni `mps` ni `cuda`. Le lot 3 n'a ajouté que le refus d'une entrée sur un autre appareil.
- CONNU-NON-CORRIGÉ : TST-09, les tests `feature = "metal"` hors de toute boucle documentée. C'est ce qui laisse R2-MTL-04 et R2-MTL-05 sans garde automatique.

## Candidats au lot 1

R2-MTL-02, R2-MTL-03, R2-MTL-04, R2-MTL-05, R2-MTL-06, R2-MTL-09, R2-MTL-11, R2-MTL-12, R2-MTL-14. Ce sont des refus ou des gardes côté hôte, ou des commentaires hors des shaders. Aucun ne touche un `.metal`, un format, l'index, ni un jeton servi sur le M3 Max. Restent hors du lot 1 : R2-MTL-01 (dépendance vendorisée), R2-MTL-07 (débit mesuré de la porte MLX), R2-MTL-08 (schéma de config MLX), R2-MTL-10 (instrument de mesure), R2-MTL-13 (shader livré et haché).
