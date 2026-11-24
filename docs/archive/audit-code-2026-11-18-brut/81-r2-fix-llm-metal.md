# Second tour, llvq-llm (session, échantillonneur, chargeurs, chemin Metal) : journal

Rien n'a été compilé ni lancé. Chaque constat a été relu dans le code avant la correction.

## Corrigé

| ID | fichiers | ce qui change, en une ligne | vérification faite |
|---|---|---|---|
| R2-DIF-03 | `llvq-llm/tests/session.rs` | `a_divergent_history_is_cut_back_to_the_common_prefix` exige, après `reuse_prefix(&[])`, que chaque cache rende 0 position par `held()`, puis nourrit `[1, 2, 9, 9]` et compare les logits à `fresh` (borne 1e-4), sur `Cat` et `Grow` | relu, NON COMPILÉ. Constat vérifié : sous `Cat`, `KvCache::len` reste 0 (`model.rs` `append` ne le touche pas), donc le mutant `Cat => c.len()` saute la coupe à `n == 0` et l'ancien test ne lisait que `ids` |
| R2-DIF-04 | `llvq-llm/src/repl.rs`, `llvq-llm/tests/sampler_nucleus.rs` | `nucleus_selected` s'arrête à `k <= NUCLEUS_MAX_K = 1024` (au plus 5 sélections : 64 à 1 024), puis rend `None` et `pick` trie toute la ligne comme avant ; doc de la borne ; test unitaire `the_selection_gives_up_past_its_cap` (une ligne qui se stabilise à 513 est prise et égale au tri, une qui ne se stabilise qu'à 2 049 est refusée) ; test d'identité neuf `pick_draws_the_reference_token_where_the_selection_gives_up` (lignes de 16 384, où la borne mord avant `n/4`, plus 1 500 têtes sur un fond `-inf`) ; troisième bras « six heads, f16 mantissa » dans l'instrument de chronométrage | relu, NON COMPILÉ. Arithmétique du test unitaire refaite à la main : poids `1 - i·2^-20` distincts et exacts en f32, total stable au premier zéro, coupe à `top_p = 0.5` vers i = 250 et 750, sous k. `nucleus_sorted` reste une fonction séparée, appelable seule |
| R2-LLM-02 | `llvq-llm/src/fused.rs`, `llvq-llm/src/bin/seal.rs` | constante publique `fused::PREALLOC_CAP = 1 << 20` (celle de `llvq-artifact` est privée et le crate n'est pas à moi) ; `min(PREALLOC_CAP)` sur les deux `with_capacity` de `load_with` et celui de `seal` ; test `a_huge_count_in_a_short_file_is_an_error_not_an_abort` (en-tête v5 Tetra à `u32::MAX` matrices, puis à 0 matrice et `u32::MAX` tenseurs portés : `Err` attendu) | relu, NON COMPILÉ. Un fichier valide se lit pareil : seule la capacité initiale change |
| R2-LLM-03 | `llvq-llm/src/fused.rs`, `llvq-llm/src/fused_cuda.rs` | doc de `matrix_side_bytes` remise sur sa fonction ; doc de `FusedInt4Proj` remise sur la struct (« One projection », plus « One `v_proj` ») ; doc de `launch_tetra48_h` et son `#[allow(clippy::too_many_arguments)]` remis sur `launch_tetra48_h`, retirés de `upload_int4` (5 arguments) ; première phrase de `load_with` réécrite sans le lien mort vers `[load]` | relu, NON COMPILÉ. Commentaires seulement |
| R2-LLM-05 | `llvq-llm/src/fused.rs`, `llvq-llm/src/fused_cuda.rs` | en-têtes : les cinq dispositions, Tetra48 pour les trois fichiers servis, la projection int4 en un lancement sans rotation ; « 252 projections and 144 rotations » remplacé par 84 int4, 168 lattice, 96 rotations (36 q/k, 36 gate/up, 24 `down_proj` Tetra), 144 pour un 4B tout lattice ; doc de `CudaEmbed` et `lm_head` ; noms d'op `llvq-emb-q8-gather` et `llvq-lmhead-q8` devenus `llvq-emb-gather` et `llvq-lmhead` ; les quatre messages « q8 » disent `self.emode.name()` | relu, NON COMPILÉ. `git grep -nF` des six chaînes dans `ops/ paper2/ docs/ .github/ llvq-runtime/ llvq-llm/ llvq-tetra/` : aucun lecteur hors `fused_cuda.rs`. Le 84 / 96 vient des commentaires du code (`fused_cuda.rs` : « 84 a token on the sealed 4B », « 12 int4 down_proj », « 24 Tetra ones ») et du constat ; non recompté sur le fichier |
| R2-LLM-04 (points 1 et 2) | `llvq-llm/src/fused_cuda.rs` | second `if let Some(n) = rows_kernel_name(model.layout) { spill_checked.push(n) }` retiré, il était identique au premier ; doc de `max_d_in()` et du champ corrigées (informative, `new` ne la compare à rien, la rotation est décidée par largeur par `wide::rot_kernel`), fonction gardée | relu, NON COMPILÉ. `Cuda::report` n'imprime rien (`llvq-cuda/src/gpu.rs:473`), donc aucune ligne de log ne disparaît |
| R2-MTL-02 | `llvq-llm/src/fused_metal.rs` | `MetalRuntime::pipeline` lit `threadExecutionWidth()` (trait `objc2_metal::MTLComputePipelineState`, par `pipe.as_ref()`) et refuse par nom tout autre que `LANES`, avant la mise en cache | relu, NON COMPILÉ. Signature lue dans `objc2-metal-0.3.2/src/generated/MTLComputePipeline.rs:413` (méthode sûre, `NSUInteger`) ; `ComputePipeline: AsRef<ProtocolObject<dyn MTLComputePipelineState>>` lu dans `candle-metal-kernels-0.9.2/src/metal/compute_pipeline.rs` |
| R2-MTL-03 | `llvq-llm/src/fused_metal.rs` | deux aides : `group_within` (garde `GROUP` si le pipeline le prend, sinon le divise par deux jusqu'à 32, sinon refus par nom) pour `TetraRows`, `TetraMatvec` (noyaux à un lancement), `Int4Rows`, `Int4Op`, `HeadOp` ; `group_exactly` (refus par nom si la limite est sous 256, nombre inchangé) pour `GatherOp` | relu, NON COMPILÉ. Shaders lus : `_lut`, `_lutdr`, twins de `llvq_tetra48_rows.metal`, `tv_q4_metal*`, `tv_q4_metal_rows`, `tv_emb_q4_metal`, `tv_q8_metal` trouvent leur ligne par `gid >> 5` et mettent en scène par `i += tgs` : le groupe ne change aucun bit. Les deux gathers prennent leur jeton de `threadgroup_position_in_grid` sur une grille de `n_ids · 256` : refus seulement. Diviser par deux garde des groupes entiers là où 256 l'était. Sur le M3 Max (limite 1 024) les six lancements gardent 256 |
| R2-MTL-06 | `llvq-llm/src/fused_metal.rs` | `check_threadgroup_memory` lit `maxThreadgroupMemoryLength()` (trait `objc2_metal::MTLDevice`, par `dev.device().as_ref()`) et refuse par nom sous `THREADGROUP_LIMIT`, dans `MetalRuntime::new` et `from_tables` ; commentaire de la constante mis à jour | relu, NON COMPILÉ. Méthode lue dans `objc2-metal-0.3.2/src/generated/MTLDevice.rs:1440` ; `Device: AsRef<ProtocolObject<dyn MTLDevice>>` dans `candle-metal-kernels-0.9.2/src/metal/device.rs:39`, motif `self.as_ref().newCommandQueue()` déjà compilé par candle |
| R2-MTL-04 | `llvq-llm/src/fused_metal.rs` | `kernel_named` n'accepte plus `ilp` (message sans `ilp`) ; test `an_unknown_metal_kernel_is_refused` étendu à `"ilp"` ; test neuf `every_kernel_the_variable_names_passes_validate` | relu, NON COMPILÉ. `git grep` : aucun test ni binaire ne pose `LLVQ_METAL_KERNEL=ilp` ; `llvq-metal` appelle `_ilp` par metal-rs, hors de ce chemin |
| R2-MTL-05 | `llvq-llm/src/fused_metal.rs`, `llvq-llm/tests/tetra48_metal_rows.rs` | `validate` remplace le filtre `starts_with` par une liste fermée : `SERVED_TETRA`, `TETRA_ROW_LOOPS`, et la constante publique neuve `TETRA_REFERENCES` (`tv_tetra48_metal`, `_ar`, `_u2`, `_c3`) ; `a_bad_tuning_is_refused_by_name` étendu à `_arh`, `_luth`, `_lutdh`, `_lutdth`, `_lutg`, `_lutdr`, `tetra48_probe` | relu, NON COMPILÉ. Signatures des quatre références lues (sortie `float`, seule `threadgroup(0)`). Tests existants passés en revue : `gate` (served + six row loops), `gate_half` (bases `lut`, `lutd`, `lutdt` avec half-io), `half_io_refuses_what_it_cannot_serve` (`lutp`, `lutd4` restent dans la liste et tombent sur le refus half-io, même message), `bin/metalrun` (liste fermée incluse dans la nouvelle) : aucun ne change d'issue |

## Laissé de côté

| ID | raison |
|---|---|
| R2-LLM-01 | décision d'opérateur (change des jetons sous `LLVQ_BATCH`), voir plus bas |
| R2-MTL-01 | aucun contrôle hôte possible avec l'API en main, voir « Décision d'opérateur » |
| R2-LLM-04 point 3 | hors mission (change la forme de `FusedRuntime`) |
| R2-MTL-02 et R2-MTL-03 dans `attn_metal.rs` | forme d'attention non servie sur Metal (refusée par nom dans les configs) ; hors des sites que la mission nomme. Même garde à poser dans `attn_metal::compile` (largeur 32, limite au moins 64 pour `attn_split_metal` et `canon::LANES` pour `attn_merge_metal`) si l'opérateur le veut |
| `docs/ENV.md:26` | décrit encore `ilp` comme valeur de `LLVQ_METAL_KERNEL` ; pas dans mes fichiers. À retirer par l'agent des docs |

## Constat faux ou déjà corrigé

| ID | preuve |
|---|---|
| (aucun) | Les dix constats revérifiés tiennent. Nuance sur R2-MTL-03 : le titre dit « sept sur neuf », la liste en compte six non gardés ; les six sont traités |

## Fichiers Linux/CUDA touchés (à compiler en croisé par la session principale)

- `llvq-llm/src/fused_cuda.rs` (R2-LLM-03, R2-LLM-04, R2-LLM-05 : commentaires, un `if let` retiré, quatre messages et deux noms d'op).

## Commandes lancées et leur résultat

Aucune : consigne de non-exécution.

## À lancer quand la machine est libre

| commande exacte | ce qu'elle doit montrer | mutant à essayer ensuite (fichier, ligne, édition, résultat attendu) |
|---|---|---|
| `nice -n 10 cargo check -p llvq-llm --all-targets -j 8` | compile, zéro avertissement | aucun |
| `nice -n 10 cargo clippy -p llvq-llm --all-targets -j 8` | zéro avertissement (vérifier en particulier `{NUCLEUS_MAX_K}` dans un `assert!`, et l'absence d'`allow` sur `upload_int4`) | aucun |
| `nice -n 10 cargo clippy -p llvq-llm --features metal --all-targets -j 8` | le chemin Metal compile : `pipe.as_ref().threadExecutionWidth()`, `dev.device().as_ref().maxThreadgroupMemoryLength()`, les `group` | aucun |
| `ops/check-linux-cuda.sh clippy -p llvq-llm --features cuda --all-targets` | `fused_cuda.rs` compile, zéro avertissement | aucun |
| `nice -n 10 cargo test -p llvq-llm --test session -j 8` | vert, 12 tests | `llvq-llm/src/session.rs:83`, remplacer `c.held()?.map_or(Ok(0), \|(k, _)\| k.dim(2))?` par `c.len()` : rouge sur `Cat`, « layer 0 kept 4 positions after a cut to zero » ; défaire, vert |
| `nice -n 10 cargo test -p llvq-llm --lib repl:: -j 8` | vert, dont `the_selection_gives_up_past_its_cap` | `llvq-llm/src/repl.rs:138`, retirer `&& k <= NUCLEUS_MAX_K` : rouge sur « a selection past 1024 weights ran » ; défaire, vert |
| `nice -n 10 cargo test -p llvq-llm --test sampler_nucleus -j 8` | vert, dont `pick_draws_the_reference_token_where_the_selection_gives_up` | `llvq-llm/src/repl.rs`, dans `nucleus_sorted`, `keep = i + 1` en `keep = i + 2` : rouge (le repli ne rend plus le vecteur de la référence) ; défaire, vert |
| `cargo test --release -p llvq-llm --test sampler_nucleus pick_times_against_the_sorted_reference -- --ignored --nocapture` | médianes de `pick` contre `pick_reference` sur trois bras (« six heads », « flat », « six heads, f16 mantissa ») ; à jouer avec la borne, puis sans (`repl.rs:138`, retirer `&& k <= NUCLEUS_MAX_K`), en interférant les deux binaires à la main | décision : si le bras le pire perd plus que le bras piqué ne gagne, désactiver le raccourci en remplaçant `repl.rs:66` `let p = nucleus_selected(&p, top_p).unwrap_or_else(\|\| nucleus_sorted(p, top_p));` par `let p = nucleus_sorted(p, top_p);`. Alors `nucleus_selected` et `NUCLEUS_MAX_K` ne servent plus qu'aux tests : les mettre sous `#[cfg(test)]` ou les retirer, sinon clippy signale du code mort |
| `nice -n 10 cargo test -p llvq-llm --lib fused::tests::a_huge_count -j 8` | vert : deux `Err` | `llvq-llm/src/fused.rs:2435`, retirer `.min(PREALLOC_CAP)` : sous Linux, le processus de test avorte sur une réservation de quelque 800 Go ; sous macOS la mémoire virtuelle peut accepter la réservation sans la toucher, et le mutant survivre. À constater ; s'il survit sur le Mac, le jouer sous Linux (CI) |
| `nice -n 10 cargo test -p llvq-llm --features metal --lib fused_metal::tests -j 8` | vert, dont `an_unknown_metal_kernel_is_refused` et `every_kernel_the_variable_names_passes_validate` | `fused_metal.rs`, remettre `Some("ilp") => "tv_tetra48_metal_ilp",` dans `kernel_named` : rouge sur `"ilp"` ; défaire, vert. Puis retirer `"tv_tetra48_metal_c3"` de `TETRA_REFERENCES` : `every_kernel_the_variable_names_passes_validate` rouge sur `c3` |
| `nice -n 10 cargo test -p llvq-llm --features metal --test tetra48_metal_rows -j 8` | vert, dont `a_bad_tuning_is_refused_by_name` | `fused_metal.rs:180`, remettre le filtre `!k.starts_with("tv_tetra48_metal") \|\| k == "tv_tetra48_metal_ilp"` : rouge sur `tv_tetra48_metal_arh` ; défaire, vert |
| `nice -n 10 cargo test -p llvq-llm --features metal --test tetra48_metal_candle -j 8` | vert : aucun lancement ne change sur le M3 Max | `fused_metal.rs:721`, `width != LANES` en `width != 64` : tout chargement refusé par nom (« SIMD-groups of 32 threads ») ; défaire. Puis `fused_metal.rs:302`, `offered < THREADGROUP_LIMIT` en `offered < THREADGROUP_LIMIT * 2` : `MetalRuntime::new` refusé par nom ; défaire. Puis dans `group_within`, `let max = pipe.max_total_threads_per_threadgroup();` en `let max = 128;` : les tests restent verts au bit (le groupe ne change aucun bit) ; défaire |
| `nice -n 10 cargo test -p llvq-llm --features metal --test q4_metal_candle -j 8` | vert : les lancements int4 gardent 256 | aucun test Metal sans fichier scellé ne passe par `GatherOp` (`git grep` de `upload_embed`, `MetalEmbed` dans `llvq-llm/tests` : rien sur Metal). Le mutant de `group_exactly` (`let max = 128;`) se constate sur le 4B scellé, par `bin/metalrun` ou `bin/chat` sur Metal : refus par nom « needs 256 threads » au premier jeton ; défaire |

## Décision d'opérateur

### R2-LLM-01, découpe du préremplissage du pool

Constat vérifié : `Qwen3::prefill_slot` coupe par `MAX_ROWS` (256), `Session::feed` par `prefill_rows()` (4 096 sur CUDA Tetra48). Je recommande la correction du relecteur, `prompt.chunks(self.prefill_rows())` à `batch.rs:385`, avec son test dans `model.rs` et son mutant. Raisons : la même requête doit avoir la même attention qu'elle arrive seule ou pendant un autre décodage, et aucun chiffre publié ne bouge (le plus long prompt de la porte GSM8K du batch fait 215 jetons, sous la coupe). Coût : des jetons servis changent sous `LLVQ_BATCH` pour les prompts de plus de 256 jetons. À faire avec un compilateur et un go.

### R2-MTL-01, statut de la commande après l'attente

Constat vérifié dans `candle-metal-kernels-0.9.2/src/metal/commands.rs:239-259` : `ensure_completed` lit le statut avant `wait_until_completed()` et ne le relit pas après. Un contrôle hôte local n'est pas possible avec ce que `fused_metal.rs` a en main : les tampons de commandes sont privés à `Commands` (champ `commands` privé de `MetalDevice`), `flush_and_wait` les vide de `in_flight` et les lâche, `ComputeCommandEncoder` n'expose pas son tampon, et `MTLCommandEncoder` n'a pas d'accesseur vers lui dans `objc2-metal` 0.3.2. Options :

1. Vendoriser `candle-metal-kernels` 0.9.2 par `[patch.crates-io]` et relire `cb.status()` après chaque `wait_until_completed()`, `Err(CommandBufferError(cb.error()))` sur `Error`. Une dizaine de lignes, mais `Cargo.toml`, `Cargo.lock` et une copie de dépendance dans l'arbre.
2. Patch amont à candle, puis montée de version. Propre, délai inconnu, et une montée de candle touche tout le chemin servi.
3. Contrôle hôte partiel : un tampon témoin partagé, écrit par un noyau minuscule en fin de passe et relu après la synchronisation. Il détecte une commande perdue en fin de file, pas une erreur au milieu, et ajoute un lancement par passe : partiel et coûteux.

Mon choix : 1, avec le signalement amont (2) en parallèle, et retour à la version amont dès qu'elle relit le statut.

### R2-DIF-04, garder ou retirer le raccourci

La borne est posée ; la mesure tranche. Ligne à changer pour retirer le raccourci : `llvq-llm/src/repl.rs:66` (voir la file).
