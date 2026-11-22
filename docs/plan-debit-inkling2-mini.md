# Plan: Inkling 2 Mini throughput

Decode runs at 1.9 tokens a second on the Mac today (*measured*, 0.526 s a token, 15 of 24 layers resident). The
items below reach 5 to 7 tokens a second with no new glue kernel and 10 to 30 with them (*estimated*). None of them
changes the file, its bits per parameter or its quality criteria, and most of them lower the memory in use.

This is a plan. Nothing in it has been launched, compiled or timed. Every gain is *estimated* until its row is
timed. It does not move the week's order in `docs/plan-inkling2-mini.md`: the encode, the whole-model reading and
the seal come first.

The items have two sources, named row by row. "ours" comes from reading our own engine, the library under it and
our earlier journals. The others were read in the MIT-licensed runtime of LookAtMyQuant
(`github.com/ShinMegamiBoson/lookatmyquant`, commit `ae0ca83`), and the row names the file. No line of it was
copied or run. Every item is rewritten in our engine, in f32, on our kernels.

## 1. What is measured today

Mac, Metal, 15 resident layers, 32-token prompt. Journal: `STEP3.md`, `gate3_decode_breakdown.json` and
`gate3_rows.json` in `docs/mesures/inkling2-mini-quality-gate-2026-11-21-brut/engine/`.

| part of one decoded token | s a token | note |
|---|---|---|
| total | 0.526 | 1.9 tokens a second |
| resident MoE, 39 calls | 0.072 | 1.85 ms a call, 40 lattice launches a call, 1,560 a token |
| native MoE, 9 calls | 0.228 | 25 ms a call, decoded on the CPU and uploaded |
| the rest: attention, convolutions, norms, routing, head | 0.225 | not timed part by part |

| prompt reading | time | note |
|---|---|---|
| 32 tokens | 3.3 s | 34,330 lattice launches |
| 2,048 tokens, one row a launch | 56.1 s | 36 tokens a second, 1,926,000 launches |
| 2,048 tokens, 8 rows a launch | 28.0 s | 73 tokens a second, 255,000 launches, output rows narrowed to f16, argmax 31 of 32 |

One anchor from the same engine on the same Mac: the sealed Qwen3 4B decodes at 65.5 tokens a second, 15.3 ms a
token, and reads 1.38 GB of quantized weights a token (*measured*, `docs/mesures/metal-nuit-2026-11-06.txt`).
Inkling reads 0.78 GB of codes a token, as the next table shows. The gap between 15 ms and 526 ms is therefore not
in the weights.

### What one token reads

*Computed* from shapes: hidden 2,304, expert width 1,152, 24 query heads, 8 key-value heads of 128, window 1,024,
200,058 output rows, 48 visits a token, 8 of them full attention.

| read by one decoded token | weights | bytes today | bytes after this plan |
|---|---|---|---|
| experts: 48 visits, 8 experts, 3 matrices | 3.06 B | 0.78 GB as codes | 0.78 GB |
| attention: 48 visits, 6 matrices | 1.29 B | 5.15 GB, held in f32 | 2.58 GB as stored |
| head | 0.46 B | 4.6 GB of traffic | 0.25 GB as int4 |
| cache, at 9,000 tokens of context | | 19.5 GB of copies | 0.93 GB |
| total at a short context | 4.81 B, 18 % of the model | 10.5 GB | 3.6 GB |

- One prompt token costs 8.7 G floating-point operations: 6.1 G in the experts, 2.6 G in the attention matrices. A
  9,000-token prompt costs about 85 T with its attention scores.
- After this plan the 16-bit attention is about 70 % of the bytes a token reads.

### What our code does around those reads

- The head widens the whole output table from BF16 to f32 at every token, in 13 chunks
  (`llvq-llm/src/inkling/mod.rs:398-411`): 0.92 GB read, 1.84 GB written, 1.84 GB read by the product.
- The cache grows by concatenation. Each visit copies its whole key and value history, then copies them three more
  times for the 24 query heads, then once more for the transpose (`layers.rs:341-348`, `308-314`, `413`). That is
  2.16 MB a cached position a token: 0.07 GB at 32 tokens, 19.5 GB at 9,000.
- The sliding layers keep and scan their whole history under a mask. They are 40 of the 48 visits and need 1,024
  positions each. The cache holds 393 KB a position in f32: 3.54 GB at 9,000 tokens, 12.9 GB at 32,768.
- The router reads its scores back to the processor at every visit, and its constant bias too
  (`layers.rs:222-223`): 96 device reads a token.
- A resident MoE call loops over 8 experts on the processor (`resident.rs:294-317`). Each expert costs 2 uploads,
  2 rotations, 3 products and about 10 separate tensor operations.
- The relative bias, its mask and its keep flags are built on the processor for every visit and uploaded
  (`layers.rs:285-305`): three vectors of rows times cached positions.
- The prompt is read in one call (`mod.rs:336`). At 9,000 tokens one visit would hold 7.8 GB of attention scores.
  A long agent prompt cannot be read today. This was not tried.
- The attention matrices are widened to f32 at load (`loader.rs:88-90`): 2.58 GB on the device for 1.29 GB in the
  file.
- Every resident record is its own buffer in the library's pool: up to 16 an expert, 30,720 at 15 layers and
  49,152 at 24 (`fused_metal.rs:610-613`, `1202-1203`). The 4B has about 1,300. candle 0.9.2 walks that pool when
  it allocates an output (`find_available_buffer`) and rebuilds it at every command-buffer flush, every 50
  operations (`drop_unused_buffers`, `metal_backend/device.rs`). What this costs a token is not measured.

## 2. Three classes of compromise

Every item is judged against the engine as it stands, on the same file and the same backend.

| class | meaning | gate |
|---|---|---|
| A | Same operations in the same order. Only where a value is computed, or how often it is copied, changes. | Logits bit for bit equal to the path before the change, on the gate prompts, CPU and Metal. |
| B | Same file, same mathematics in f32, another order of additions. | Greedy tokens equal on the fixed prompts, routing flips counted, and the ±0.12 % perplexity band of `docs/ETAT.md` §4 against the path before the change. |
| C | A price in quality, memory, format or identity. | Listed in §8, not scheduled. |

Class B is near zero and not zero for one reason. The router has near ties, with margins down to 3e-6 (*measured*).
Any change of rounding ahead of a router can flip one expert at such a tie. The gate counts those flips and reads
the perplexity; it does not promise identical text on every prompt.

A gate is green only after a mutant turned it red, as in step 3 of the resident path.

One debt stands before any class B item is called green. Gate 3c of the resident path reads -0.158 % against its
±0.12 % band on one window and 15 layers (*measured*). It is read again on 24 layers and several windows first.

## 3. Decode: items with no price, or near none

| id | what | in our code | source | class | gain a token (*estimated*) | memory |
|---|---|---|---|---|---|---|
| D0 | All 24 layers resident. No new code: it is tonight's file. | `inkling/resident.rs` | ours | A | 0.228 s becomes 0.017 s; decode 0.31 s | experts 6.27 GB resident, no dense copy |
| K1 | Resident data out of the library's pool: allocated on the device directly, or in a few large buffers. | `fused_metal.rs:610-613`, `1202-1203` | ours | A | no figure; removes a cost that grows with the resident layers | none |
| K2 | Command-buffer cadence set for this model. It is a setting of the library, 50 operations a buffer by default. | startup of `bin/inkrun` and of the runtime | ours; on the 4B the setting alone moved a step between 33.9 and 27.7 tokens a second (header of `bin/metalrun.rs`) | A | no figure; one sweep | none |
| D1 | Read the router bias once. At one row, skip the gather, the scatter and the 16 uploads of an MoE call. | `layers.rs:223`, `resident.rs:300-316` | ours | A | 0.01 to 0.03 s | none |
| D10 | A visit written for one row: no row gather, no transposed copies, convolution states in a ring instead of a concatenation. | `layers.rs:59-73`, `366-424` | ours | A | about 100 operations a visit become about 60 | none |
| D2 | Routing on the device: scores, bias, top 8 and weights in one f32 kernel, with the reference tie rule (higher score, then lower index). The expert ids stay in a device buffer. | `layers.rs:221-241` | `fast_kernels.py:2918-3055` | B | 48 reads removed, 0.005 to 0.015 s | none |
| D3 | One launch a side of the MoE for the 8 experts. The 128 experts of a layer sit in one buffer with an offset table, and the kernel reads the ids D2 left. The weighted sum stays in f32, in ascending expert order. | `resident.rs`, `fused_metal.rs`, `fused_cuda.rs` | `fast_kernels.py:2726-2831`, `1877-2025`; `model.py:637-665` | A aimed, B accepted | 40 launches a call become 4; 0.089 s becomes 0.010 to 0.020 s | offset tables, kilobytes |
| D4 | Head through the int4 kernel, reading the table as the chosen file stores it. | `mod.rs:398-411`, `fused_metal.rs:270` | ours; `fast_kernels.py:1380-1456` | the file's own values | 4.6 GB of traffic becomes 0.25 GB; 0.03 to 0.05 s | -0.68 GB |
| D5 | Cache written in place in preallocated buffers. A ring of 1,024 for the 40 sliding visits. Query heads read their key-value head without a copy, the grouped form `docs/ETAT.md` already serves on the 4B. | `layers.rs:308-348`, `SlotCache` | ours for in place and grouped; `model.py:272-504` for the ring | A in place, B ring and grouped | nil at 32 tokens; 19.5 GB of copies a token removed at 9,000 | 3.54 GB becomes 0.93 GB at 9,000; 12.9 becomes 2.48 at 32,768 |
| D6 | Relative bias and mask computed on the device. | `layers.rs:285-305` | `fast_kernels.py:4284-4534` | A | small in decode, required by P2 | none |
| D7 | Fused glue kernels in f32: block residual, message gates and norm in one launch; the five products that read the attention input in one launch; convolutions with their norm; weighted sum, convolution and residual. | `layers.rs:366-424`, `576-656` | `fast_kernels.py:3194-4789`, `model.py:1426-1572` | B | about 100 operations a visit become about 12; 0.225 s becomes 0.012 to 0.08 s | -1.29 GB if the kernel reads attention as stored |
| D8 | Pipelined loop: the argmax and the next embedding stay on the device, and step k+1 is encoded before token k is read. | decode loop, `lm.rs` | `generate.py:388-410` | A | the GPU no longer waits for the processor; pays after D7 | +0.25 GB, embed table int4 resident |
| D9 | Rotation fused into its neighbour, same signs, no re-encode. | rotation of `fused_metal.rs` | `fast_kernels.py:482-721`, `3398-3629` | A aimed | 4 launches a call become 3 | none |

Four notes.

- D0's estimate assumes a call costs the same with 24 resident layers as with 15. The pool of K1 grows by 60 %
  between the two, so K1 goes with D0.
- D2 compares `sigmoid(score) + bias` in f32, as the reference does. The 16-bit scores of the source are not taken.
- D3 keeps the body of the served lattice product, so each output row is the same number. Only the launch changes.
  It also takes the pool from 49,152 buffers to a few hundred, which settles K1 for good.
- D7 is the long item: one to two weeks a backend. Its two readers disagree by a factor of 6 on its gain. The first
  hour of D1 times the rest in eight parts and replaces both guesses.

## 4. Prompt reading: items with no price, or near none

| id | what | in our code | source | class | gain (*estimated*) | memory |
|---|---|---|---|---|---|---|
| P1 | The rows kernel writes f32 rows. | rows tuning of `fused_metal.rs` | ours | A against the one-row kernel | 2,048 tokens in 28 s instead of 56 s, without the 31 of 32 | none |
| P2 | Prompt read in chunks of query rows over a cache already filled, so the scores stay under a set budget. | `mod.rs:336`, `layers.rs:366-424` | ours; `model.py:1745-1757` | A | a 9,000-token prompt becomes readable: about 125 s at P1's rate | scores bounded, 0.44 GB at 512 rows |
| P3 | Each expert a visit uses is decoded once into a transient dense tile and applied to all its rows by a plain product. One decode an expert a visit, instead of one a row. At one row the fused kernel stays. | `resident.rs:254-277`, a decode-only kernel | ours; the source decodes once a tile of 32 rows (`fast_kernels.py:2372-2681`, `2860-2918`) | B | 255,000 launches become about 18,000 decodes at 2,048 tokens; 28 s becomes 3 to 6 s | transient, 32 MB an expert in f32 |
| P4 | Attention of the prompt in one kernel, bias and mask inside, no score tensor. | `layers.rs:366-424` | `fast_kernels.py:4284-4534` | B | with P3, a 9,000-token prompt in 10 to 30 s | removes the score tensor |
| P5 | Prefix reuse across requests: a snapshot of the 4 convolution states at the end of each turn (3.8 MB), the cache cut back to it when the next prompt starts with the same tokens. | `InklingCache`, `llvq-runtime/src/engine.rs:672` | ours; the source rereads the whole prompt | A | an agent's long prompt is read once a session, not once a request | 3.8 MB a snapshot |
| P7 | The cache after a static system prompt saved to disk, and loaded when a new session starts with the same tokens. | `llvq-runtime` store | ours | A | the first request of a session skips the read | disk only: 0.9 GB for 9,000 tokens with D5's ring |
| P6 | Draft by n-gram lookup in the prompt, accepted only where it equals the greedy token. | `llvq-runtime` | `generate.py:46-247` | A once P1 holds | pays only when the answer copies the prompt, which code agents do | none; needs S3 |

Three notes.

- The 85 T operations of a 9,000-token prompt are a floor that no kernel removes. At 3 to 8 T effective a second
  on this Mac that is 10 to 26 s (*computed*; the rate is an assumption). P3 and P4 reach that floor. P5 and P7
  are what remove the wait, by not reading the same prompt twice.
- P2, P5 and P7 decide whether a coding agent can use the model at all. opencode's first prompt was 9,116 tokens
  (*measured* on the 4B trial).
- D5's ring and P5's rollback pull against each other: a ring forgets what a deep rollback needs. A snapshot
  therefore carries its own 1,024 sliding positions, 0.34 GB in f32, and the runtime keeps the last two turns.

## 5. Served time that is not kernel time

| id | what | in our code | source | class | gain | price |
|---|---|---|---|---|---|---|
| S1 | The reasoning effort of the chat template exposed through the OpenAI field of the same name. | `inkchat.rs`, `llvq-runtime/src/openai.rs` | ours | the client's own choice | fewer thinking tokens before the answer, when the client asks | none |
| S3 | A verified step of several rows: feed tokens that are known or guessed in one pass, keep the longest prefix the model agrees with, roll the cache back for the rest. | `CausalLm`, `InklingCache` | ours; the tree pass of `docs/ETAT.md` §5 is its CUDA relative | A once P1 holds | the base of P6 | none; needs D5's rollback |
| S2 | Several streams in one pass: the rows of different requests share the launches of D3. | `llvq-runtime` batch pool | ours; the Qwen3 batch configs | A a stream once P1 holds | aggregate tokens a second, not one stream's | one cache a stream |

## 6. What these items do to memory

| on the device | today | after | by |
|---|---|---|---|
| experts, 24 layers | 6.27 GB (*computed* from 3.92 GB for 15 layers, *measured*) | 6.27 GB | D0 |
| everything else | 5.09 GB (*measured*), of which attention in f32 2.58 and head in BF16 0.92 (*computed*) | 3.12 GB (*computed*) | D4, D7 |
| cache at 9,000 tokens, f32 | 3.54 GB (*computed*) | 0.93 GB (*computed*) | D5 |
| transient copies a token at 9,000 | 19.5 GB of traffic (*computed*) | 0.93 GB read (*computed*) | D5 |

The file stores 1.83 GB for everything else (*computed*). The 1.3 GB between that and 3.12 GB is not attributed yet.
One item raises the memory after load: D8 puts the int4 embedding table on the device, 0.25 GB. Every speed figure
is published with the memory after load and the peak at a long prompt beside it.

## 7. Order and ceilings

Mac only, $0. No item starts before tonight's queue has finished, and no two compiling agents run during a
measurement.

| wave | items | work | decode after it (*estimated*) | prompt after it (*estimated*) |
|---|---|---|---|---|
| 0 | D0, K1 | none for D0, hours for K1 | 0.31 s, 3.2 a second | unchanged |
| 1 | K2, D1, D10, D4, D5 in place, P1, P2, P5, S1 | hours each, 2 to 3 days | 0.24 to 0.28 s, 3.6 to 4.2 a second | 73 tokens a second; 9,000 tokens readable, once a session |
| 2 | D2, D3, D9, D5 ring and grouped, D6, P3 | 3 to 6 days, Metal | 0.14 to 0.20 s, 5 to 7 a second | 2,048 tokens in 3 to 6 s |
| 3 | D7, D8, P4, P7, S3, P6, S2, then the CUDA twins | 1 to 2 weeks a backend | 0.03 to 0.10 s, 10 to 30 a second | 9,000 tokens in 10 to 30 s, once |

K1, K2 and D10 are not counted in the decode estimates: they have no figure yet.

Waves 0 and 1 fit before the publication window of `docs/plan-inkling2-mini.md`. Wave 2 fits if the seal and the
card run leave three days. Wave 3 is after it.

Two ceilings on this Mac, both *computed* and neither a forecast.

- Decode: after the plan a token reads 0.78 GB of codes and 2.83 GB of 16-bit and int4 weights. The lattice kernel
  reached 129 GB/s of codes on this Mac through the MLX door (1.377 GB a token times 93.9 tokens a second, same
  journal): 6 ms. At an assumed 130 to 290 GB/s the rest takes 10 to 22 ms. That is 35 to 60 tokens a second.
- Prompt: 10 to 26 s for 9,000 tokens, from the operation count above.

On CUDA, D2 and D3 are milestone J5's gathered launch. No card figure is computed here. The card budget is not
touched by this plan: the first compressed run and GSM8K keep it.

## 8. Items with a price, listed and not scheduled

| item | price | what it would buy |
|---|---|---|
| Attention in the lattice code or in int4 | 0.0234 KL (*measured*); the operator kept 16 bits on 2026-11-22 | the bytes a token reads go from 3.6 GB to 1.6 GB (*computed*): the largest lever left on decode after this plan |
| One rotation shared by a layer's experts (`fast_kernels.py:3398-3629`) | a re-encode and a KL gate | 8 input rotations a call become 1 |
| Router scores or partial sums in 16 bits | expert flips at near ties | nothing we need |
| Cache in f16 or int8 | rounding ahead of every router; never measured on this model | halves the cache or better |
| Prompt read in f16 products | rounding ahead of every router, kept in the cache for the whole session | a faster product on this GPU, not measured |
| Draft head `mtp.safetensors` of the model, verified 6 to 8 positions at a time (`inkling/drafter.py`) | 0.64 GB more on the device, a download, identity at near ties not guaranteed | 1.5 times by its author's figure, not run here |
| A draft from one loop pass instead of three, verified by the full model (ours, untested) | a second cache for the draft; acceptance unknown and possibly near zero | a draft that adds no weights; one offline count of agreement says whether it is worth a line |
| Our own tree pass (`docs/ETAT.md` §5) | memory and identity, the reasons it was set aside on the 4B | 2.30 times on GSM8K on the 4B |
| Tool-call scaffolding forced by a grammar | the output is constrained: a behaviour choice, the operator's | valid calls, and their fixed tokens in one pass through S3 |
| The 8-rows kernel with f16 rows, as it is | argmax 31 of 32 (*measured*); replaced by P1 | nothing P1 does not give |

## 9. How a wave is read

- Served tokens a second after a 32-token prompt and after a 9,000-token prompt, median of rounds with its range.
- Time to the first token of a 9,000-token prompt, first request and second request of a session.
- Memory after load and peak during the long prompt.
- The class gate of each item, with its mutant.
- A chat and a tool round trip through `llvq-runtime` beside the bench, not the bench alone.

## 10. What this plan does not establish

- The 0.225 s of the rest has no measured split. Every gain taken from it is a reader's estimate.
- What the library's buffer pool costs a token is read in its code, not timed.
- The Mac is shared, and the figures of §1 are one session's. The 4B anchor is from another day.
- Nothing is timed on CUDA. The first card run of `docs/plan-inkling2-mini.md` gives that baseline.
- The source's own speeds are self-declared and were not run here.
- It does not reopen the choice of engine. An MLX port of the forward pass is another route and is not costed. On
  the 4B it took the Mac from 65.5 to 93.9 tokens a second (*measured*, same journal).
