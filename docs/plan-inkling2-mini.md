# Plan: Inkling 2 Mini, from the quality gate to a served object

Goal: a Tetra file of `thinkingmachines/Inkling-2-Mini` (26.39 B parameters, natively NVFP4) at about 2.2 bits per
parameter, measured against the native model and the public 2-bit files, then served.
Two tracks share J1 and J2. The short track (J1, J2, J3) ends on a file that `transformers` loads compressed. The long
track (J4, J5) builds our own forward pass and a routed kernel.
If tonight's gate reads PASS, lot 1 (§4) starts at once: code only, on the Mac, $0, no commit.
If it reads FAIL or NOT SETTLED, the coordinator analyses the cause and fixes what lies inside the method, off the
prereg and written as deviations; a heavy question waits for the operator (his word of 2026-11-21, evening).
Everything beyond lot 1 waits for the operator.
Never without him: a paid run, a format change, kernel work on a card, a publication, a served Qwen3 object, a kill.
Cost: 43 to 91 agent-days in all, 11 to 19 for the short track. Dollars: $1 to $3 for the short track, $40 to $145 with
J5 and J6 (*estimated*, §3).
The encode is the long pole on the Mac: 12 to 17 hours for all experts (*computed*, §1).

## 1. The object and its budget

The tensor table fixes the budget (*computed* from the checkpoint header, revision `ed40e7da`). Experts are 92.7 % of
the parameters, so their rate sets the file.

| part | parameters | default type | GB |
|---|---|---|---|
| experts, 24 layers x 128 x (gate, up, down) | 24,461,180,928 | Tetra, 2.0185 b/weight | 6.172 |
| attention `wq_du`, `wk_dv` | 226,492,416 | Tetra | 0.057 |
| attention `wv_dv`, `wo_ud`, `gate_attn`, `wr_du` | 417,595,392 | int4 g128 | 0.222 |
| `embed`, `unembed` (untied) | 926,318,592 | int4 g64 | 0.521 |
| router, controllers, convolutions, norms, queries, sinks | 38,797,824 | as shipped, 16 bits | 0.078 |
| text model | 26,070,385,152 | | 7.049, 2.163 b/param |
| vision tower | 318,256,128 | as shipped, BF16 | 0.637 |
| with vision | 26,388,641,280 | | 7.686, 2.330 b/param |

Expert rate: 48-bit words plus one f32 row scale a row, as `rtbits` counts (`llvq-bench/src/bin/rtbits.rs:361`). In
the `Tetra48` VRAM layout (2.150 kernel b/weight) the experts take 6.574 GB. The MTP head (317.9 M parameters, separate
file) is outside the table. One token at batch 1 reads 8 experts in each of 48 layer visits, 0.77 GB of codes.

Encode: 1.02e9 expert blocks at 429 to 556 µs a block and thread, on 10 to 12 threads, take 12 to 17 hours on the Mac
(*computed*, anchors of `docs/mesures/innovation-cout-encodage-2026-11-12.txt`). The same journal's 14B encode scales
to $20 to $30 on `rtx-pro-6000` (*estimated*). Tonight's measured rate replaces both (§8).

## 2. Milestones and tracks

| milestone | content | short track | long track |
|---|---|---|---|
| J0 | tonight's gate on the looped core | needed | needed |
| J1 | the whole model by the offline route, measured | needed | needed |
| J2 | the artifact: records, seal, `rtbits`, decode equality | needed | needed |
| J3 | a `transformers` door with an experts module | end | |
| J4 | the Rust forward pass, oracle, `ppl`, `gsm8k` | | needed |
| J5 | routed kernel, persistent schedule, prefill, Metal door | | end |
| J6 | quality levers, publication | after J3 | after J5 |
| J7 | served through the OpenAI API, tool calls included, for coding agents | | after the model generates in our engine |

The operator chose the long track on 2026-11-22: our own engine and kernel serve the object, and the `transformers`
door (J3) is deferred. Order: J1 and J2 (the file), J4 (the Rust forward pass), then the fused kernel in the model
launched eagerly from the host as a first resident run on a card, then J5 (gathered launch, persistent schedule).
The eager step reuses `tv_tetra48` unchanged; at the 16 µs a launch of the 4B's eager host (4.11 ms for 252
projections) the 1,440 launches a token of this model cap it near 40 tok/s (*estimated*, host overhead alone).

The object is for use, not for a paper: its weights stay compressed on the device (operator, 2026-11-22). A dense
load is a check, never a deliverable, so J3 is the resident module and nothing less. On CUDA the package binds the
lattice kernel alone today: attention's int4 matrices and the tables fall back to dense there, 9.0 GB resident for
the text model against 7.05 GB with the int4 kernel bound (*computed* from §1, KV cache and activations not
counted).

Each milestone closes on a short review against the native model: quality, whole-model b/param, GB on the card,
served tok/s, encode cost, noise floor. No proxy closes one.

## 3. Milestones against the code

Lines are new or changed, by analogy with the comparable. Every figure in this table is *estimated*, except J1's and
J6's encode hours, *computed* in §1.

| | lines | comparable | agent-days | Mac h | card $ |
|---|---|---|---|---|---|
| J1 | Python 1,200 to 1,800, Rust 300 to 500 | `bin/hcapture.rs` 281, `bin/extq.rs` 508, `bin/gsm8k.rs` 519 | 4 to 7 | 15 to 25 | 0 |
| J2 | Rust 600 to 1,000 | `bin/seal.rs` 189, `hfpack.rs` 861, `artifact2.rs` 763 | 3 to 5 | 1 to 3 | 0 |
| J3 | Python 500 to 900 | `fused.py` 307, `integration.py` 217 | 4 to 7 | 2 to 5 | 1 to 3 |
| J4 | Rust 3,000 to 4,500 | Qwen3 forward in `model.rs` (5,402 with the served paths) | 10 to 20 | 5 to 15 | 0 |
| J5 | CUDA and host 4,000 to 7,000 | `pers.rs` 6,230, `persist4e_cuda.rs` 2,709, `llvq_pers.cuh` 2,069 | 20 to 40 | | 20 to 60 |
| J6 | 300 to 800 | `ops/llvqtune` adapters | 2 to 5 | 12 to 17 | 20 to 80 |

J5 assumes 10 to 25 card jobs at the $1 to $3 of recent bench jobs (`docs/data/jobs.csv`). J6 holds row scales on an
`h200`, $17 to $35 (0.05 s a step per B parameters, 9,507 steps, cost journal), a second draw, and a census.

### J0, tonight

The prereg `proofs/preregistration-inkling2-mini-quality-gate-2026-11-21.md` reads KL against the native model on the
looped layers, against the `unsloth` GGUF rungs on the same layers. Tonight's offline route: Python capture on the
reference code (`ops/inkling/`), `bin/extq` on each (W, H) pair, reconstructions back into the teacher.

### J1, the whole model by the offline route

Reused as is: `extq::encode_matrix` and `factor_hessian` (`llvq-llm/src/extq.rs:88-221`), which take any width that is
a multiple of 24; `Rotation::new` accepts any width (`llvq-quant/src/rotation.rs:50-58`); `quantize_affine` for the
tables (`llvq-llm/src/embedquant.rs:22`); `int4_record` for the g128 records (`llvq-llm/src/sealed.rs:371`).

Extended: the capture. In passes 2 and 3 the query, key and value inputs differ, each through its own message gate
(`modeling_inkling2_loop.py:961-967`). Attention thus needs four Hessians a layer, summed over passes: the query stream
(read by `wq_du`, `gate_attn`, `wr_du`, the sinks), the key stream, the value stream, the gated output (width 3,072).
`calib.rs` assumes four activations and seven matrices a block (`model.rs:38-82`, `calib.rs:668-685`), so attention
goes through a generic (W, H) driver beside `extq`.

New: the whole-model evaluator (KL, top-1, perplexity, all 24 layers, the rival files in full) and generation for
GSM8K. The reference code carries per-pass caches (`modeling_inkling2_loop.py:1354-1357`), so `generate` works on it.
The student's experts in f16 weigh 49 GB, which the 64 GiB Mac holds on the CPU path only. GSM8K-200 comes first, the
size of the rivals' self-reported scores.

Gate J1: whole-model KL and GSM8K-200 of the student against the native model, read against the rivals' full files at
their bits, under a prereg stamped before the first measurement.

### J2, the artifact

9,216 expert records (24 x 128 x 3) fit format v5 unchanged: names are `u32`-prefixed strings, shapes two `u32`
(`llvq-artifact/src/format.rs:670-675`), the record count a `u32` (`format.rs:1569`). The index map and
`codebook_fingerprint` do not move.

Two facts constrain it. `split_name` expects `model.layers.{b}.{proj}.weight` (`format.rs:1534-1545`), and the native
names start `model.llm.layers`, so the seal renames to the repository's convention. The sealed stream carries raw
tensors as f16 or affine integers only (`llvq-artifact/src/sealed.rs:83-87`), so BF16 and F32 tensors either become f16
(lossy on the F32 router bias and queries) or travel in a sidecar safetensors (decision 3).

Tonight's `bin/extq` keeps its codes. Each expert leaves `codes/eNNN.llvqrec`: three records, gate, up and down, in
format v5 (`write_matrix_with`, Tetra kind), named `model.layers.{L}.mlp.experts.{e}.{gate|up|down}_proj.weight`. Each
file is read back and decoded against the reconstruction bit for bit before it is renamed into place, and
`extq verify` repeats the check on a whole layer. No layer encoded tonight is encoded again for J2.

New: an Inkling seal that gathers these records with the attention records and the tables, `rtbits` over file and
sidecar, and `hfpack` for these names (`hfpack.rs:458` writes a Qwen2 tokenizer config).

Gate J2: every record decodes through `decode_matrix` (`format.rs:1510`) to tonight's reconstruction bit for bit, a
flipped code fails the check, and `rtbits` gives the whole-model b/param. The full suite runs before any commit.

### J3, a first door through `transformers`

`replace_modules` refuses anything but `nn.Linear` (`llvq-tetra/llvq_tetra/integration.py:76-77`), and the experts
are a 3-D `nn.Parameter` module (`modeling_inkling2_loop.py:775-807`). `TetraLinear.materialize` decodes to a dense
weight (`modules.py:74-95`), which would rebuild the 49 GB. The new module keeps the codes resident and decodes the
routed experts with the existing Metal op (`mps.py:191`, `fused.py:113`) and the CUDA extension of stage 4 of
`docs/plan-transformers.md`.

Gate J3: same logits as the dense reconstruction, within a tolerance stated in its prereg, on CUDA and on Metal.
Speed is measured, not promised.

### J4, the Rust forward pass

No model trait exists. `ppl` and `gsm8k` build `Qwen3::new` directly (`bin/ppl.rs:106`, `bin/gsm8k.rs:173`), and
`Qwen3` is a struct over candle's Qwen3 `Config` (`model.rs:27`, `model.rs:2546`). The minimal abstraction is a
`CausalLm` trait in a new `llvq-llm/src/lm.rs`: forward over a prompt, one decode step on a cache, the vocabulary.
`Qwen3` implements it by delegation, so its paths do not move. The Inkling model lives in `llvq-llm/src/inkling/`.

The loader reuses `Source::var_builder` (`llvq-llm/src/loader.rs:315`); candle 0.9.2 reads `F8_E4M3` and `U8`. New:
the e2m1 nibble decode and the checkpoint's gate/up row interleave (both documented in tonight's `nvfp4.py`), relative
bias, sinks, L2 clips, short convolutions with separate prefill and decode paths (`modeling_inkling2_loop.py:373-385`),
dynamic convolutions in layers 5, 11, 17 and 23, the gated output, block residuals, message gates, 48 caches, routing,
the softcapped SwiGLU, and a chat template (`chatfmt` is Qwen3's). `bin/oracle.rs` compares against candle's Qwen3,
which has no Inkling counterpart, so the oracle reads f32 dumps of the reference code.

Lot 1 wrote the first slice on 2026-11-22, in the worktree `inkling-fwd`: the embedding and layers 0 to 5 in
prefill match the reference dumps on 64 tokens on CPU, relative error 2e-7 to 7e-7 a layer, routing identical
(*measured*, `~/llvq-inkling-mini/lot1-c/oracle_cpu.log`; a mutant without the convolution's output scale reads
1.2 to 1.3). It corrects this paragraph on five points. Full layers (5, 11, 17, 23) take three different inputs
for query, key and value in every pass, outside the loop too. The block residual is an attention over the block
states, and the final state goes through `final_input` before the norm. Every short convolution adds a residual
and a 1/sqrt(K+1) scale, and the dynamic one has no separate decode path. The vocabulary of the head is the
unpadded 200,058 and the logits are softcapped at 30. `Source::var_builder` does not fit: names carry
`model.llm.`, the queries are offset by one layer, and the loader maps the safetensors directly.

Gate J4: oracle on every backend first, hidden states and logits against the dumps within its stated tolerance, in
prefill and in decode; then `ppl` on the reconstruction equal to J1's perplexity within the paired band.

### J5, kernels and serving

4E knows three shapes (`llvq-cuda/src/pers.rs:611-613`) and a fixed site list (`pers.rs:178`). A routed token picks
its sites on the device: 8 experts x 3 records x 48 visits, 1,152 matrices. New: a gathered decode and matvec launch,
a persistent schedule over 48 visits, a grouped prefill, the Metal or MLX door. Widths stay at 3,072 or less, under
the `rot_apply` wall (`docs/format-noyau.md` §8).

Gate J5: tokens equal to J4's eager path, or inside the ±0.12 % perplexity band, then served tok/s, GB after load and
the 8k peak on one card, each with its price in memory.

The throughput items that cost nothing on the other criteria, or near nothing, are ordered in
`docs/plan-debit-inkling2-mini.md`: decode, prompt reading, their memory, and the items with a price that are not
scheduled.

### J6, levers and publication

Loop-aware calibration on the quantized prefix, row scales (`ops/llvqtune` needs adapters for 3-D experts, and a
card), a starved-expert policy, a second draw. Publication and the paper angle are the operator's.

### J7, serving through the OpenAI API (asked by the operator on 2026-11-22)

Target: a coding agent such as opencode plugs into the served model through `/v1/chat/completions`.
`llvq-runtime` already serves the three Qwen3 files that way: `llvq pull`, `run` and `serve`, Ollama's routes, and
`/v1/chat/completions`, `/v1/completions` and `/v1/models` streamed as server-sent events with usage and a reasoning
field (`llvq-runtime/src/openai.rs`, `README.md`). Three things are missing for this model and this use.

- Tool calls. The runtime refuses them by name (`openai.rs:436`, `api.rs:163`, `chat.rs:29`, and "out of scope" in
  `docs/plan-runtime.md`). A coding agent lives on them. Inkling's own template carries the format: a
  `<|message_tool|>` role and a `<|content_invoke_tool_json|>` content type (`chat_template.jinja` of the checkpoint,
  6,868 bytes). New: tools rendered through the template, the call block parsed out of the stream, `tool_calls`
  returned whole and as streamed deltas, `role: tool` messages accepted.
- The engine. It owns a `Qwen3` and Qwen3's hand-written template (`engine.rs:26`, `:37`). New: the engine over
  `CausalLm` (`lm.rs`, lot 1), and Inkling's template with its reasoning effort.
- Context. The 4B under 4E holds 8,704 positions; an agent's system prompt and tool schemas need tens of thousands.
  The KV cache of this model costs 0.44 GB at 8,192 positions and 1.24 GB at 32,768 in f16 (*computed*), so 32,768
  is the target and the time to read a long first prompt is the figure a user will feel.

opencode's side, read on its documentation on 2026-11-22: a custom provider is declared with the package
`@opencode/ai/providers/openai-compatible`, `settings.baseURL` ending in `/v1`, and a model entry with
`capabilities.tools` and `limit.context`; for a model it does not know it assumes JSON tool support and a 200,000
token context, so the entry must state the real limits.

A first real session ran on 2026-11-22: opencode 1.18.20 against `llvq serve` on the sealed Qwen3 4B on Metal,
inside a sandbox (no write outside a scratch folder, no network but the local machine). The client took our
stream: six tool-call round trips were parsed, executed and answered (*measured*). Its first request carried
9,116 prompt tokens and took 178 s to answer on the shared Mac; each later step added about 2,170 tokens and
took 52 to 63 s, the history being read again only from what changed. The session did not complete its task:
the 4B asked six times for `/src/main.rs`, a path outside the repository that the client refused, until the
520 s cap. That is the model, not the route. Two facts for this plan follow. A coding agent's first prompt is
about 9,000 tokens, so the time to read a long prompt decides whether the served model is usable: at the 10
tok/s of today's row-by-row prefill of Inkling it would be 15 minutes (*computed*), which makes the grouped
launch a condition of J7 and not only of speed. And this client's version takes the configuration shape
`provider` / `npm: @ai-sdk/openai-compatible` / `options.baseURL`, not the `providers` / `package` shape its
online documentation shows.

Gate J7: a scripted client drives a tool-call round trip (call emitted, result returned, answer) on the served
model, streamed and not, then opencode itself edits a file in a scratch repository through the endpoint. Order:
after the model generates in our engine. The tool-call layer is model-independent and can be proven first on the
served Qwen3 4B, whose template has its own tool format.

### The week to a publication (operator, 2026-11-22: "enough to publish the model at the end of the week")

The horizon is a publication of the model around 2026-11-27. The act of publishing stays the operator's, asked at
the time. What a publication needs, each with its fundamental criterion:

| piece | criterion it must state | state on 2026-11-22, 10:00 |
|---|---|---|
| the sealed file: experts, attention, tables | whole-model b/param and GB, text and with vision | experts of 15 layers of 24 encoded, tables done, attention capture running |
| quality | KL of the whole quantized model against the native one, and GSM8K against the native model and the best rival, same harness | KL on the loop's experts read (PASS); GSM8K not run |
| the engine | same tokens as its dense reference, GB resident on a card and on the Mac | prefill and decode green on CPU and Metal with native experts; resident experts green on one layer |
| speed | served tok/s on a card and on the Mac, with its memory price, and the time to read a long prompt | not measured; the eager path is expected slow, the grouped launch is the lever |
| serving | `llvq pull`, `run`, `serve`; the OpenAI route with tool calls; a real coding-agent session | tool calls served on Qwen3; Inkling's template in work; engine not yet generic |
| the card and the licence | every figure labelled, the rivals' figures beside ours, Apache 2.0 and the upstream acceptable-use policy passed on | the policy page is not read yet |

The bar for the whole quantized model, read on 2026-11-22 in our own harness (WikiText-2 test, 8 windows of 2,048
tokens, every tensor of the rival file transplanted into the reference forward, *measured*): unsloth
`UD-Q2_K_XL`, 9.139 GB and 2.801 b/param, reads KL 0.0651 [0.0584, 0.0728], top-1 89.99 %, perplexity x 1.047;
`UD-IQ2_XXS`, 8.486 GB and 2.600 b/param, reads 0.0837 [0.0758, 0.0930], 88.58 %, x 1.066. Our file aims at
7.05 GB and 2.163 b/param (*computed*). First costs outside the experts, each alone on one window: the embedding
table in int4 g64 0.0028, the output table 0.0030, two int4 attention matrices of layer 0 0.0021. Attention is
2.4 % of the weights and every token crosses all of it, so its type rule is decided on the three arms of the
whole model (experts; plus attention; plus tables), not on §1's default.

Attention and the tables alone, experts native, on the 8 windows (*measured*, 2026-11-22, the 144 attention
matrices encoded from the full capture with no fallback):

| what is quantized | weights | GB, against GB in f16 | KL, 95 % interval | top-1 |
|---|---|---|---|---|
| the six attention matrices, §1's rule | 644.1 M | 0.279 against 1.288 | 0.0234 [0.0207, 0.0259] | 93.35 % |
| `wq_du` and `wk_dv` alone, lattice | 226.5 M | 0.057 against 0.453 | 0.0118 [0.0101, 0.0135] | 95.39 % |
| `wv_dv`, `wo_ud`, `gate_attn`, `wr_du` alone, int4 g128 | 417.6 M | 0.222 against 0.835 | 0.0170 [0.0153, 0.0186] | 94.50 % |
| the two tables, int4 g64 | 926.3 M | 0.521 against 1.853 | 0.0078 [0.0071, 0.0084] | 96.28 % |

Attention costs about 0.023 to 0.030 KL for each GB it saves, the tables 0.006, and the experts of the loop 0.010
against their native NVFP4 (*computed*). Even at 4 bits attention is the dearest place to save bytes. Keeping it
in 16 bits puts the text model at 8.06 GB instead of 7.05 (*computed*) and removes its 0.023.
The operator settled it on 2026-11-22: attention stays in 16 bits, as shipped, for the first file, and the
finer rules come later. The object is therefore the experts of 24 layers in the lattice code, the two tables in
int4 g64, everything else as shipped: 8.059 GB and 2.473 b/param for the text model, 8.696 GB and 2.636 b/param
with vision (*computed* in the kernel accounting). Not read yet: the six matrices one by one, and `wq_du` and
`wk_dv` in int4.

That object was read whole on 2026-11-22, once the 24 layers were encoded (*measured*,
[whole-model](mesures/inkling2-mini-whole-model-2026-11-22.txt)): KL 0.0880 [0.0783, 0.0969], top-1 88.66 %,
perplexity x 1.068. Paired by window it is 0.0042 above `UD-IQ2_XXS` [+0.0003, +0.0087], a file 5 % larger, and
0.0229 above `UD-Q2_K_XL` [+0.0169, +0.0294], a file 13 % larger. The file holds the rivals' ladder; it does not
beat it. The experts alone read 0.0834, not separated from `UD-IQ2_XXS`.

The loop's result does not carry to the whole model, because the layers outside the loop cost more:

| lattice experts of layers | KL | a layer |
|---|---|---|
| 0 to 5 | 0.0144 | 0.0024 |
| 6 to 17, the loop (the gate) | 0.0378 | 0.0032 |
| 18 to 23 | 0.0365 | 0.0061 |
| of which 20 to 23 | 0.0219 | 0.0055 |

Layers 18 to 23 are a quarter of the layers and 44 % of the experts' KL. A layer of experts moved to int4 g128 adds
0.282 GB (*computed*), and at best buys 0.019 to 0.026 KL a GB, where the rivals' ladder moves by 0.0285: bytes
spent that way follow the ladder and do not beat it. No expert has been encoded in int4. What could beat the
ladder is a better code of those six layers at the same bits: a calibration on the quantized prefix, a floor for
the starved experts (25.0 to 41.4 % of their cells), the row scales. None is measured on this model.

The file is sealed and verified (2026-11-22, same journal): `inkling2-mini.llvq`, 6.777 GB, and its sidecar,
1.366 GB; 2.473 b/param in the kernel accounting and 2.490 as stored; `inkseal verify` PASS, every native tensor
in exactly one place and no text tensor omitted. On the experts alone, over the 24 layers, the lattice code at
2.037 bits reads the KL of `IQ2_XXS` at 2.372 bits and is below `IQ1_S` and `IQ1_M` on 8 windows of 8
(*measured*). The file loses that as a file because it keeps attention in 16 bits.

The engine ran on a card for the first time on 2026-11-22 (*measured*,
[first-card](mesures/inkling-first-card-2026-11-22.txt), prereg timestamped before it): on one L4, 24 layers of
experts resident in lattice codes, the 32 tokens of the Mac's resident run, 32 of 32, 10.11 GB after the experts,
0.166 s a token on the eager path, verdict PASSES, $0.04. On the Mac the same commit plus the first wave of the
throughput plan reads 0.192 s a token with 24 resident layers, against 0.526 with 15 that morning
(`docs/plan-debit-inkling2-mini.md`).

Order of the week: the file and the whole-model KL first; then the first run on a card (resident GB, tokens, first
tok/s) and GSM8K on the same card; then the grouped launch for speed, and the row scales if GSM8K leaves a gap worth
their $17 to $35; then the runtime on Inkling with tools and a real coding-agent session; the card of the model
last. Card budget in force: $25 (2026-11-22), nothing spent.

## 4. Lot 1, started without the operator if and only if the gate reads PASS

Code only, on the Mac, $0, in the worktree, no commit and no push unless he asks. Builds use their own
`CARGO_TARGET_DIR`, `-j 2`, `nice 15`, because the encode calls `bin/extq` by path (METHODE §6). No lot 1 process loads
the model while the encode holds the guard's lock: one heavy job at a time.
A PASS reached only after an off-prereg fix starts lot 1 too, and the morning report says which reading it was.

| item | content | acceptance gate |
|---|---|---|
| 1a | attention capture per visit: query, key, value, gated output, summed over passes | on two sequences, the three input Hessians of a sliding pre-loop layer (0 to 4) are equal bit for bit and those of pass 2 differ; layer 5 is a full-attention layer with three input norms, so its three differ before the loop too (found by lot 1, `modeling_inkling2_loop.py:1006`, `:1037-1044`); token counts equal tokens x passes; dropping pass 3 is caught |
| 1b | generic (W, H) driver for attention, Tetra and int4 g128 | bit equality with the block loop on the tiny model, as `tests/extq_equiv.rs` does for experts |
| 1c | int4 g64 tables on the native `embed` and `unembed` | `quantize_affine` on a Qwen3 table reproduces `embedq`'s bytes |
| 1d | whole-model evaluator and generation | with no tensor substituted, KL is exactly 0; one perturbed expert makes it positive; 32 greedy tokens by cache equal 32 by full forward |
| 2 | Inkling seal over tonight's `codes/eNNN.llvqrec`, sidecar, `rtbits`, decode equality | §3 J2's gate on one layer of tonight's codes, read from the sealed file |
| 3 | `inkling/` config, loader, layers 0 to 5, dump script, oracle | layer 5's output against the reference dump within the tolerance its test states; removing the convolution's output scale fails it |

Lot 1 closed on 2026-11-22 at 08:50, every gate green and checked by the coordinator on the raw logs (*measured*).
1a and 1d: with no tensor substituted KL is exactly 0 on 2,048 tokens, one perturbed expert gives 1.3e-3, 32 cached
greedy tokens equal the full forward's; the generation by this Python route costs 4.6 s a token. 2: layer 6 sealed
from the night's codes decodes to its reconstructions, 384 records and 1,019,215,872 weights with 0 differing,
experts at 2.037085 archive bits, three flipped bits each caught. 1b: the generic driver equals the block loop
byte for byte, Tetra and int4 g128. 1c: `inktable` reproduces `embedq q4` on a Qwen3 table, 87,515,213 bytes
identical, and the two native tables exist in int4 g64, 260.5 MB each. 3: see J4. Not done: the full attention
capture (1 to 1.5 h of GPU, being made resumable in slices), a seal of all 24 layers, any card run.
The recipe's output does not move when H is scaled by 0.5 or 0.7 (lot 1, 1b), so the division by the weight sum
is immaterial to the bytes.

The cumulative encode keeps running on the Mac under the guard and the sentinel. The operator set no hour
(2026-11-21, evening: the next day is a Sunday).

Never started without him: any paid run (no spending cap is in force, one is owed first), a format change or any
change to the index map, CUDA kernel work on a card, a publication, a change to the served Qwen3 objects, a kill.

## 5. Dependencies and parallel work

```
J0 ──> J1 ──> J2 ──> J3            short track
        │      └───> J5            needs J2's records
        └──> J4 ──> J5 ──> J6      long track; J4 needs only the native file
```

| agent | owns | files |
|---|---|---|
| A, Python | 1a, 1d | new files under `ops/inkling/` |
| B, Rust | 1b, 1c, 2 | a new seal module in `llvq-llm/src/`, new bins, `llvq-bench/src/bin/rtbits.rs`; target dir `target-lot1` |
| C, Rust and Python | 3 | `llvq-llm/src/lm.rs`, `llvq-llm/src/inkling/`, `ops/inkling/dump_ref.py`, in its own worktree `inkling-fwd` |

1c went from A to B: the tables reuse `quantize_affine` in Rust, which is correct by construction, where a Python
copy would have to be proved equal. B and C both build `llvq-llm`, so C works in a worktree of its own, cut from the
same base; its files are new and join this branch by copy. Each adds its own `pub mod` line to `lib.rs`. Nobody
edits tonight's tools: the encode calls them by path all day.

## 6. Risks and open questions, ranked

1. A rival holds the slot. `LookAtMyQuant` reads 8.87 GB at 2.69 b/weight after QAT, KL 0.0259 before training,
   GSM8K 143/200 against 147 native (*self-reported*). Our default is 7.05 GB without vision; quality against QAT is
   unknown.
2. No f16 reference. Two ports of the same weights differ by KL about 0.002 (*self-reported*), a floor under which
   nothing resolves.
3. Starved experts. About 8,192 tokens an expert a pass, 3.56 a dimension (*computed*); gpt-oss had 31.4 % of cells
   under full rank (*measured*, `docs/HISTORIQUE.md`). `extq` falls back to `H = I` on zero tokens.
4. Tonight's codes are per-expert files, not a sealed artifact. The seal of J2 reads them; a change of recipe
   (row scales, another calibration) costs the encode again, about 30 Mac minutes a layer.
5. Loop correctness in J4: fp32 islands, two convolution paths, 48 caches, messages. The oracle covers prefill then
   decode.
6. Random access in J5: 0.77 GB of codes a token, sites known only after the router.
7. Generation on the Mac: the f16 student exceeds the Metal working set, and GSM8K-200's duration is unknown.
8. `transformers`: the architecture is on `main` at `046bdb8cc2`, `llvq-tetra` was cleanroomed on 5.18 and 5.19; no
   release carrying both is checked.
9. One rotation seed per expert (`bin/extq.rs:65`): a kernel rotates the input once per routed expert, under 1 % of
   that expert's matvec (*computed*).

## 7. Decisions that are the operator's

| # | decision | default if silent | cost |
|---|---|---|---|
| 1 | spending cap | nothing paid | $0 |
| 2 | where the remaining encode runs | on the Mac under the guard, no fixed hour; he stops it with one word | 12 to 17 Mac h, or $20 to $30 on a card |
| 3 | BF16 and F32 tensors | sidecar safetensors, no format change | a raw tag is a format change and a full suite |
| 4 | type rule (settled 2026-11-22 for the first file: experts all Tetra, tables int4 g64, attention in 16 bits as shipped) | experts all Tetra; attention per §1 | the 4B's int4 `down_proj` band (a third of its layers) on the experts adds about 0.75 GB; attention all Tetra saves 0.12 GB |
| 5 | b/param perimeter | text model and with vision, both stated | none |
| 6 | GSM8K route | GSM8K-200 in the Python harness on the CPU | a card run, or a GGUF export: llama.cpp's main branch reads the model since `29b52d398b` (PR 32694, 2026-11-20; read on GitHub, not run here) |
| 7 | starved experts | tonight's rule | J6 |
| 8 | rotation seeds | per expert, as tonight | a shared seed re-encodes everything |
| 9 | the phase | a development phase on the existing method, opened by name (`docs/MODES.md`) | none |
| 10 | row scales, second draw | after J1's reading | J6 |
| 11 | publication, paper, licence (Apache 2.0 with a usage policy) | nothing published | none |
| 12 | a kill | none | none |

## 8. The night's numbers

| quantity | value |
|---|---|
| gate reading, layer set it was read on | PASS, on {6..17}, the 12 layers of the loop, at 06:25 on 2026-11-22 |
| KL_T and top-1 of the lattice arm | 0.0378 [0.0328, 0.0425] and 92.04 % (*measured*, 8 windows of 2,048 tokens of WikiText-2 test) |
| KL of `UD-IQ1_S`, `UD-IQ1_M`, `UD-IQ2_XXS`, `UD-Q2_K_XL` on the same set | 0.0539, 0.0454, 0.0430, 0.0290 (*measured*) |
| expert bits: lattice, and each rung | 2.037; 2.229, 2.368, 2.444, 2.764 (*computed*). The lattice's KL falls at 2.55 bits on the ladder (*computed*, linear in log KL) |
| encode seconds a layer, threads | about 1,720 s (29 minutes) at 10 threads, 13.3 to 13.5 s an expert, *measured* on layers 6 to 15 |
| teacher tokens a second, calibration length used | 122 (131,072 tokens captured in 1,073 s on MPS, *measured*); 64 x 2,048 as preregistered |
| census: cells under 2,304 tokens, one visit and after three | 27.96 % in one visit; 11.2 % of the cells of layers 6 to 17 after the sum over three passes (*measured*) |
| layers encoded when the report is written | 6 to 17, 1,536 experts, codes read back bit for bit; 0 to 5 and 18 to 23 are being encoded on 2026-11-22 |

## 9. What this plan does not establish

It ran nothing. Tonight's `capture.py`, `prep_layer.py` and `evalkl.py` did not exist when it was written. Unchecked:
llama.cpp running the model (its support is read on GitHub only), the teacher's speed, whether `faer`'s factor is
identical across thread counts, any served speed.
