# Roadmap 27B: development backlog

Draft of 2026-09-28, not adopted, companion of [ROADMAP-27B](ROADMAP-27B.md). Each package lists its items in
dependency order. Sizes: S under half a day, M one to two days, L three to five. Every test that needs a checkpoint,
a corpus or a dump is `#[ignore]` in CI and fails naming the missing file (rule 10). Every item that touches the file
format runs the full suite before its commit.

Summed from these sizes (*computed*): about 26 to 41 days before the operator's 27B decision (P0, P1, the critical
part of P2, P3, the engine gate of P6), and about 43 to 69 days for everything but the conditional items (block
streaming, the V split, the bf16 kernel path, Metal).

## P0. Guard rails

| item | where | size | test |
|---|---|---|---|
| `tetra_lock`: re-encode a fixed Qwen3-0.6B slice with the `tetra1` recipe, compare to a pinned sha256. Run on the Mac before each merge of a Qwen3.5 change, output kept in the merge's journal | new `llvq-llm/tests/tetra_lock.rs` (`driftcheck` refuses Tetra files) | S | mutant: calibration volume or rotation seed changed flips the hash |
| Fixtures: the five `config.json` and the 27B `model.safetensors.index.json`, with their revisions | new `llvq-llm/tests/fixtures/qwen35/` | S | read by the P1 tests |
| Diagnostic binaries refuse a `qwen3_5` config by name: `tetra_schur`, `errmap`, `hcapture`, `probe`, `kltemp`, `gbench`, `hratio`, `cosdiag`, `gaindiag` | those bins; `model.rs:1337` panics today | S | one refusal test |

## P1. Forward pass on the Mac

| item | where | size | test |
|---|---|---|---|
| `ModelCfg` with `enum Arch {Qwen3, Qwen35}`: nested `text_config`, `layer_types`, rotary dim and theta from `rope_parameters`, tie flag at top level, `output_gate_type` accepted only as `swish` and ignored. A `qwen3::Config` view keeps the Qwen3 path and its 12 literals in 9 files compiling | new `config.rs`; constructors `smoke.rs:991`, `ppl.rs:106`, `mmlu.rs:625`, `gsm8k.rs:164`, `oracle.rs:65`; `sealed.rs:562`, `fused*.rs` | M | fixtures parse; a synthetic config with top-level tie and no `text_config` key loads tied; two disagreeing levels are refused |
| One name map: `model.language_model.*` to `model.*`, `model.visual.*` and `mtp.*` dropped, counted and printed. Used by the loader, `seal.rs:95`, export and `int4swap`. Shard filter for truncated depth | `loader.rs:174-318` | S | 8 layers of the 27B select shards 1, 2, 3, 16, 18 (16.9 GB) |
| Reference dumper, PEP-723 under `uv run`: transformers `==5.17.0` and torch pinned, sha256 of `modeling_qwen3_5.py` in every dump. Eager attention and torch fallbacks. Passes: f32, bf16, and an f64 pass that patches the six hard f32 casts (both norms, rotary, softmax, both delta rules). A 300-token window from the calibration corpus. Per layer: hidden state, the inputs of `in_proj_qkv`, `out_proj`, `o_proj` after the gate and `down_proj`, `g` and `beta`, absmax. Logits, NLL, 16 decode steps, chat ids. `--tiny`: seeded normal init of every tensor, norms included, `A_log` set so `exp(g)` spans 0.5 to 0.999, vocab 248,320, DeltaNet value width 768 against attention width 128. `--layers N`; optional `--stream` keeps one layer resident for a full-depth f32 dump on the Mac | new `ops/oracle_qwen35.py`, prompts in `docs/data/` | L | chunked against recurrent inside torch; a forward hook asserts no tensor below f64 in the f64 pass |
| `oracle --ref <dump>`: per-layer relative error, first failing layer, top-1 and KL on logits, decode steps. Refuses a dump under 130 tokens or from another transformers version | `bin/oracle.rs`, `ops/run.py:971-1001` | M | every mutant below turns it red before its green counts (rule 10) |
| Norms: zero-centred `x̂·(1+w)` in f32; gated norm with plain `w` then `silu(z)` | `model.rs:1296-1303` | S | mutants: `w` for `1+w`; `q_norm` and `k_norm` swapped; input and post-attention norms swapped |
| Partial RoPE on 64 dims, theta 1e7; vision token ids 248053 to 248057 refused | `model.rs:157-215` | S | factor 1.0 is bit-identical to today's RoPE |
| Gated attention: per-head query and gate halves of `q_proj`, `sigmoid(gate)` before `o_proj` | `model.rs:1385-1476` | M | mutants: halves swapped; gate after `o_proj`; silu for sigmoid |
| DeltaNet decode: conv as 4 multiply-adds (never grouped `conv1d`, 10,240 launches a layer), SiLU, split, l2norm, q scaled by 1/√128, `repeat_interleave` (V head `h` reads K head `h / ratio`), `g` in f32, `beta` in the model dtype then widened, f32 state, gated norm | new `gdn.rs` | M | f64 scalar reference at ratios 1, 2, 3; mutants: tile mapping, SiLU dropped, q scale dropped, no l2norm |
| DeltaNet chunked prefill with initial state: chunk 64, decay masked before the exp, `(I + A)⁻¹` by the nilpotent product (six batched matmuls, candle has no triangular solve), softplus as `max(x, 0) + log1p(exp(−|x|))` | `gdn.rs` | L | `chunked_matches_recurrent` at lengths 1, 63, 64, 65, 200, with and without state; mutant: tril mask dropped |
| Hybrid block and caches: `enum Mixer`, `enum LayerCache`, mask to attention layers only, KV cache on 16 layers. `step_state` and `KvStore::Prealloc` refused on hybrid (A2 ground, rule 3) | `model.rs:1208-1980`, callers in `calib.rs:807-1385` | L | Qwen3 oracle stays at max\|Δ\| = 0; tiny `generate` equals `generate_uncached` |
| dtype policy from the dumps' absmax: f16 while absmax stays under about 3.2e4, DeltaNet always f32 | `gdn.rs`, `eval.rs:68-87` | S | f16 and bf16 arms against the f32 dump |

## P2. Encode chain

Rows 1 to 7 and 11 are on the path to the operator's decision; rows 8 to 10 and 12 are not.

| item | where | size | test |
|---|---|---|---|
| 1. Roles `MixerIn`, `MixerOut`, `MlpIn`, `MlpOut` keep `Act::index` 0..3, so rotation seeds do not move | `model.rs:38-81, 1327-1366`, `calib.rs:591` | M | Qwen3 plan unchanged; seeds unique over a 64-layer hybrid plan |
| 2. Capture: `MixerIn` after the input norm (shared by `in_proj_qkv`, `in_proj_z`); `MixerOut` after the gated norm, or after the attention gate | `model.rs` capture sites | M | captured tensors against the dump; mutant: `o_proj` input taken before the gate |
| 3. Per-block matrix plan; `shard_extent` walks it; the streamed header counts the plan's sum | `calib.rs:615-632`, `artifact2.rs:156-470`, `smoke.rs:1085-1103`, `tests/int4_wiring.rs` | M | tiny hybrid: two segments byte-identical to one; a torn shard inside a DeltaNet block resumes there; full suite |
| 4. `PROJ_TYPES` per architecture: `in_proj_qkv`, `in_proj_z`, `out_proj` added to `LLVQ_INT4_TYPES`, `LLVQ_RESTORE_*`, `int4swap`; `in_proj_a/b` refused as targets | `sealed.rs:41`, `smoke.rs:717`, `bin/int4swap.rs` | S | refusal tests; mutant: `out_proj` dropped |
| 5. Text-only seal that streams records (VmHWM 62.4 GB at 14B, *measured*, [encode-14b](mesures/encode-14b-2026-09-22.txt); about 115 GB at 27B, *estimated*), text-only `config.json`; `sealed::load` builds the hybrid | `bin/seal.rs:47-181`, `sealed.rs:486-600` | M | carried tensors of the sealed 0.8B are by name the complement of its 150 records; sealed ppl within 1% of the encoding; full suite |
| 6. Carry of the 3.5 beds' F32 `A_log` and DeltaNet norm (decision 13): a raw F32 tag, or the f16 narrowing measured by the 0.8B oracle | `llvq-artifact/src/sealed.rs:39-87` | S | F32 tag: round trip, an old reader refuses it by name, full suite; f16 narrowing: the 0.8B oracle stays under its stamped bound |
| 7. `verify_artifact` resolves hybrid names | `artifact2.rs:632-700` | S | mutant: block index shifted by one fails on the 0.8B |
| 8. Streaming export in parts of at most 1 GB, loadable as `Qwen3_5ForCausalLM` (VmHWM 86.5 GB at 14B, *measured*, `jobs.csv` export-14b; about 158 GB at 27B, *estimated*) | `bin/export.rs:78-240`, `ops/jobs/export-14b.sh:200` | M | transformers logits match our forward on the 0.8B export |
| 9. `rtbits` over the text denominator; `ops/run.py` estimate for the nested config and 400 matrices. 6.36e-5 core-s/weight stays the billing guard, the 14B's 2.82e-5 (*computed*, 1.226 µs a weight × 23 vCPU) is printed beside it | `llvq-bench`, `ops/run.py:205-352` | M | selftest lands on 24,326,963,200, 6,912,212,992 and 497,025,024 weights |
| 10. 27B memory path of decision 3: none (h200 f32 or rtx-pro-6000 bf16), or block-streaming `smoke` | `smoke.rs:960-1300`, `calib.rs:940-1395` | 0 or L | streaming: 0.8B artifact byte-identical to the resident run |
| 11. Full-depth gate script for the 0.8B, modelled on the design C gate | new `ops/` script | S | a broken arm goes red |
| 12. Optional: `in_proj_qkv` split into q, k, v row records, concatenated at load, if attribution asks for V in int4 | `artifact.rs`, `sealed.rs`, served loader | M | a split file decodes to the unsplit matrix bit for bit |

## P3. Evaluation

| item | where | size | test |
|---|---|---|---|
| GGUF letter ids read from the model's tokenizer (357, 417, 351, 414). `gguf_mmlu_thin.py:34` hard-codes Qwen3's 362, 425, 356, 422; `gguf_mmlu.py:97` loads the Qwen3-4B tokenizer. Needed by K | `ops/gguf_mmlu.py:97`, `ops/gguf_mmlu_thin.py:34` | S | refuses an id that does not decode to ' A' to ' D'; the Qwen3-4B IQ2 run still reads 39.78 |
| Re-census `MAX_PREFILL_ROWS` on the new tokenizer; narrow to the last row before the head (full logits about 1.5 GB) | `model.rs:528-545`, `bin/mmlu.rs:795-799` | S | identical picks with and without the narrowing on 57 questions |
| Chat templates `Qwen35OptIn` (0.8B, 2B), `Qwen35OptOut` (4B, 9B), `Qwen38` | `chatfmt.rs:67-127` | S | ids equal `apply_chat_template` from the dump, think on and off |
| Chat-wrapped MMLU control beside raw 5-shot | `bin/mmlu.rs` | S | paired through `mmlupair` on 4B and 4B-Base |
| `mmlu --split validation`: 1,531 questions, 5-shot examples from `dev`, its own fingerprint | `bin/mmlu.rs:659-660`, `corpus.rs` | S | a validation run never prints `a74a6d62` |
| Gate check: per-layer `exp(g)` and `beta` against BF16 | `bin/oracle.rs` | S | mutant: `in_proj_a` scaled 2.75× goes red, whatever perplexity does |
| Subject-range option for `mmlu`, offset for `gsm8k`, so long censuses split in two jobs | `bin/mmlu.rs`, `bin/gsm8k.rs` | S | two ranges concatenated equal one dump |

## P4. Served path

| item | where | size | test |
|---|---|---|---|
| `rotplan.rs` reads the P2 roles; `split_name` unchanged | `rotplan.rs:143-189` | S | rotation sites on a sealed tiny hybrid |
| Served loader generic over `Qwen3` and `Qwen35`; `in_proj_a/b` concatenated and carried f16 | `fused.rs:2408`, `fused_cuda.rs:2694`, `fused_metal.rs:1448` | M | claimed = total sites; first divergence from dense at token 32 or later on the 0.8B |
| Launch accounting per token in `fusedrun` | `bin/fusedrun.rs` | S | hand count on the tiny model |
| DeltaNet step kernel, NVRTC, 2 launches a layer, prefill looped per token | new `kernels/gdn_step_h.cu`, `fused_cuda.rs` | L | host build against the Rust reference; mutants; tokens identical to the candle-op arm |
| GQA without the `repeat_kv` copy, hybrid only unless the Qwen3 oracle stays at max\|Δ\| = 0 | `model.rs:1436-1456` | S | tokens identical |
| `planesbench`: activation sized to the file's widest `d_in`, shapes read from the file | `planesbench.rs:131-139, 1793, 1917, 3535` | S | the 14B file refused on 2026-09-23 passes |
| Image: new bins in `cargo build --bin` and the runtime `COPY`; configs under `RUN test -f`; `space-build-log.py` markers rewritten per rebuild; rtx-pro-6000 in `BENCH_FLAVORS` for C20; dumps reach jobs through the bucket with their sha256 | `ops/Dockerfile.cuda`, `ops/run.py:747-759, 1014`, `ops/jobs/` | S | a bin missing from one list fails the probe in 15 s; a wrong dump hash refuses the job |
| bf16 activations through `tv_tetra48_h`, `tv_q4_h`, `rot_apply`, only if the full-depth absmax passes about 3.2e4 | `llvq-cuda`, `fused_cuda.rs:2713-2720` | L | same kernel tests at bf16 |
| Metal, optional: tiled `tv_q4_metal` past `d_in` 8192; DeltaNet step in MSL | `fused_metal.rs`, `kernels/` | M + L | bit-identical at `d_in` ≤ 8192; tokens identical |

## P5. Row-scale training

| item | where | size | test |
|---|---|---|---|
| `llvqtune` routes `linear_attn`, reads the nested config and prefix; fold `(1+w)·τ − 1` for zero-centred norms | `ops/llvqtune/.../torch_model.py:41-230`, `bin/rowscale.rs:31-37` | L | routed count = records − int4; 20-step probe under `MAX_FIRST_KL` |
| `--grad-checkpoint` made effective: student in `train()` (no dropout in Qwen3.5), teacher in `eval()` | `llvqtune/__main__.py:92-96` | S | 9B peak with and without it; identical first KL |
| Gate on a fixed held-out batch instead of `improved` on the stream | `domain/loop.py:81` | S | a frozen optimizer fails the gate |
| `h200x2` in `FLAVORS` ($10.00/h, `hf jobs hardware` of 2026-09-28), only if the 9B peak rules out one h200 | `ops/run.py:92-110` | S | `bench` prints a ceiling |

## P6. Baselines

| item | where | size | test |
|---|---|---|---|
| llama.cpp engine gate on `qwen35`: a BF16 GGUF of the 4B through `gguf_mmlu` against our dense path, same 2,280 questions | job script | S | \|d\| ≤ 1.5 pp and p ≥ 0.05; Qwen3-4B read 70.36 against 70.32, \|Δ\| 0.04 pp, no p computed ([m3-iq2-metal](mesures/m3-iq2-metal-2026-08-30.txt)) |
| llama.cpp image pinned by digest at a build with MTP support (PR #24025); the job prints `block_count` and `nextn_predict_layers` | `ops/jobs/*iq2*.sh` | S | a 65-block file loads |
| vLLM pinned by digest at v0.27.1, the first line with the text-only `Qwen3_5ForCausalLM`; `SIZES['27b']`; a flavor parameter; one `--kv-cache-dtype` for every arm | `ops/awq_speed.py`, `ops/jobs/gsm8k-vllm.sh`, `paper-awq-speed.sh` | M | engine gate at one small size, \|d\| ≤ 1.5 pp and p ≥ 0.05 |
| 4-bit reader of decision 5: compressed-tensors for `RedHatAI@91bd022d`, or an AutoAWQ entry for `mattbucci@2cc4cffb` | `ops/awq_dequant.py:98-240, 650-674` | M to L | L1/L2/L4 controls; argmax against vLLM |
| b/param per arm over text tensors, numerator and denominator, MTP and vision removed | `ops/` (the GGUF reader of [qwen35-recon](mesures/qwen35-recon-2026-09-28.txt)) | S | reproduces 5.40 for Qwen3-14B-AWQ |
