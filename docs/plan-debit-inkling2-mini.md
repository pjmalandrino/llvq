# Plan: Inkling 2 Mini throughput

Decode runs at 1.9 tokens a second on the Mac today (*measured*, 0.526 s a token, 15 of 24 layers resident). The
items below reach 5 to 7 tokens a second with no new glue kernel and 10 to 30 with them (*estimated*). None of them
changes the file, its bits per parameter or its quality criteria, and most of them lower the memory in use.

This is a plan. Nothing in it has been launched, compiled or timed. Every gain is *estimated* until its row is
timed. It does not move the week's order in `docs/plan-inkling2-mini.md`: the encode, the whole-model reading and
the seal come first.

Several mechanisms were read in the MIT-licensed runtime of LookAtMyQuant
(`github.com/ShinMegamiBoson/lookatmyquant`, commit `ae0ca83`). Each row names the file it was read in. No line of
it was copied or run. Every item is rewritten in our engine, in f32, on our kernels.

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

What reading our own code adds, all *computed* from shapes (hidden 2,304, 24 query heads, 8 key-value heads of 128,
window 1,024, 200,058 output rows, 48 visits a token, 8 of them full attention):

- The head widens the whole output table from BF16 to f32 at every token, in 13 chunks
  (`llvq-llm/src/inkling/mod.rs:398-411`). That is 0.92 GB read and 1.84 GB written, then 1.84 GB read by the
  product: 4.6 GB of traffic a token.
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

## 2. Three classes of compromise

Every item is judged against the engine as it stands, on the same file and the same backend.

| class | meaning | gate |
|---|---|---|
| A | Same operations in the same order. Only where a value is computed, or how often it is copied, changes. | Logits bit for bit equal to the path before the change, on the gate prompts, CPU and Metal. |
| B | Same file, same mathematics in f32, another order of additions. | Greedy tokens equal on the fixed prompts, routing flips counted, and the ±0.12 % perplexity band of `docs/ETAT.md` §4 against the path before the change. |
| C | A price in quality, memory, format or identity. | Listed in §7, not scheduled. |

Class B is near zero and not zero for one reason. The router has near ties, with margins down to 3e-6 (*measured*).
Any change of rounding ahead of a router can flip one expert at such a tie. The gate counts those flips and reads
the perplexity; it does not promise identical text on every prompt.

A gate is green only after a mutant turned it red, as in step 3 of the resident path.

One debt stands before any class B item is called green. Gate 3c of the resident path reads -0.158 % against its
±0.12 % band on one window and 15 layers (*measured*). It is read again on 24 layers and several windows first.

## 3. Decode: items with no price, or near none

| id | what | in our code | read in | class | gain a token (*estimated*) | memory |
|---|---|---|---|---|---|---|
| D0 | All 24 layers resident. No new code: it is tonight's file. | `inkling/resident.rs` | ours | A | 0.228 s becomes 0.017 s; decode 0.31 s | experts 6.27 GB resident, no dense copy |
| D1 | Read the router bias once. At one row, skip the gather, the scatter and the 16 uploads of an MoE call. | `layers.rs:223`, `resident.rs:300-316` | ours | A | 0.01 to 0.03 s | none |
| D2 | Routing on the device: scores, bias, top 8 and weights in one f32 kernel, with the reference tie rule (higher score, then lower index). The expert ids stay in a device buffer. | `layers.rs:221-241` | `fast_kernels.py:2918-3055` | B | 48 reads removed, 0.005 to 0.015 s | none |
| D3 | One launch a side of the MoE for the 8 experts. The 128 experts of a layer sit in one buffer with an offset table, and the kernel reads the ids D2 left. The weighted sum stays in f32, in ascending expert order. | `resident.rs`, `fused_metal.rs`, `fused_cuda.rs` | `fast_kernels.py:2726-2831`, `1877-2025`; `model.py:637-665` | A aimed, B accepted | 40 launches a call become 4; 0.089 s becomes 0.010 to 0.020 s | offset tables, kilobytes |
| D4 | Head through the int4 kernel, reading the table as the chosen file stores it. | `mod.rs:398-411`, `fused_metal.rs:270` | `fast_kernels.py:1380-1456` | the file's own values | 4.6 GB of traffic becomes 0.25 GB; 0.03 to 0.05 s | -0.68 GB |
| D5 | Cache written in place in preallocated buffers. A ring of 1,024 for the 40 sliding visits. Query heads read their key-value head without a copy, the grouped form `docs/ETAT.md` already serves on the 4B. | `layers.rs:308-348`, `SlotCache` | `model.py:272-504` | A in place, B ring and grouped | nil at 32 tokens; 19.5 GB of copies a token removed at 9,000 | 3.54 GB becomes 0.93 GB at 9,000; 12.9 becomes 2.48 at 32,768 |
| D6 | Relative bias and mask computed on the device. | `layers.rs:285-305` | `fast_kernels.py:4284-4534` | A | small in decode, required by P2 | none |
| D7 | Fused glue kernels in f32: block residual, message gates and norm in one launch; the five products that read the attention input in one launch; convolutions with their norm; weighted sum, convolution and residual. | `layers.rs:366-424`, `576-656` | `fast_kernels.py:3194-4789`, `model.py:1426-1572` | B | about 100 operations a visit become about 12; 0.225 s becomes 0.012 to 0.08 s | -1.29 GB if the kernel reads attention as stored |
| D8 | Pipelined loop: the argmax and the next embedding stay on the device, and step k+1 is encoded before token k is read. | decode loop, `lm.rs` | `generate.py:388-410` | A | the GPU no longer waits for the processor; pays after D7 | embed table int4 resident, 0.25 GB |
| D9 | Rotation fused into its neighbour, same signs, no re-encode. | rotation of `fused_metal.rs` | `fast_kernels.py:482-721`, `3398-3629` | A aimed | 4 launches a call become 3 | none |

Three notes.

- D2 compares `sigmoid(score) + bias` in f32, as the reference does. The 16-bit scores of the source are not taken.
- D3 keeps the body of the served lattice product, so each output row is the same number. Only the launch changes.
- D7 is the long item: one to two weeks a backend. Its two readers disagree by a factor of 6 on its gain. The first
  hour of D1 times the rest in eight parts and replaces both guesses.

## 4. Prompt reading: items with no price, or near none

| id | what | in our code | read in | class | gain (*estimated*) | memory |
|---|---|---|---|---|---|---|
| P1 | The rows kernel writes f32 rows. | rows tuning of `fused_metal.rs` | ours | A against the one-row kernel | 2,048 tokens in 28 s instead of 56 s, without the 31 of 32 | none |
| P2 | Prompt read in chunks of query rows over a cache already filled, so the scores stay under a set budget. | `mod.rs:336`, `layers.rs:366-424` | `model.py:1745-1757` | A | a 9,000-token prompt becomes readable: about 125 s at P1's rate | scores bounded, 0.44 GB at 512 rows |
| P3 | Tokens sorted by expert and read in tiles of 32: the codes of an expert are decoded once a tile. | `resident.rs:254-277` | `fast_kernels.py:2138-2182`, `2372-2681`, `2860-2918` | A aimed, B accepted | launches 255,000 become 10,000 to 20,000 at 2,048 tokens | none |
| P4 | Attention of the prompt in one kernel, bias and mask inside, no score tensor. | `layers.rs:366-424` | `fast_kernels.py:4284-4534` | B | with P3, a 9,000-token prompt in 10 to 30 s | removes the score tensor |
| P5 | Prefix reuse across requests: a snapshot of the 4 convolution states at the end of each turn (3.8 MB), the cache cut back to it when the next prompt starts with the same tokens. | `InklingCache`, `llvq-runtime/src/engine.rs:672` | ours; the source rereads the whole prompt | A | an agent's long system prompt is read once a session, not once a request | 3.8 MB a snapshot |
| P6 | Draft by n-gram lookup in the prompt, accepted only where it equals the greedy token. | `llvq-runtime` | `generate.py:46-247` | A once P1 holds | pays only when the answer copies the prompt, which code agents do | none; needs D5's rollback |

P2 and P5 decide whether a coding agent can use the model at all. opencode's first prompt was 9,116 tokens
(*measured* on the 4B trial).

## 5. What these items do to memory

| on the device | today | after | by |
|---|---|---|---|
| experts, 24 layers | 6.27 GB (*computed* from 3.92 GB for 15 layers, *measured*) | 6.27 GB | D0 |
| everything else | 5.09 GB (*measured*), of which attention in f32 2.58 and head in BF16 0.92 (*computed*) | 3.12 GB (*computed*) | D4, D7 |
| cache at 9,000 tokens, f32 | 3.54 GB (*computed*) | 0.93 GB (*computed*) | D5 |
| transient copies a token at 9,000 | 19.5 GB of traffic (*computed*) | 0.93 GB read (*computed*) | D5 |

The file stores 1.83 GB for everything else (*computed*). The 1.3 GB between that and 3.12 GB is not attributed yet.
One item raises the memory after load: D8 puts the int4 embedding table on the device, 0.25 GB. Every speed figure
is published with the memory after load and the peak at a long prompt beside it.

## 6. Order

Mac only, $0. No item starts before tonight's queue has finished, and no two compiling agents run during a
measurement.

| wave | items | work | decode after it (*estimated*) | prompt after it (*estimated*) |
|---|---|---|---|---|
| 0 | D0 | none | 0.31 s, 3.2 a second | unchanged |
| 1 | D1, D4, D5 in place, P1, P2, P5 | hours each, 2 to 3 days | 0.24 to 0.28 s, 3.6 to 4.2 a second | 73 tokens a second; 9,000 tokens readable, once a session |
| 2 | D2, D3, D9, D5 ring and grouped, D6 | 3 to 6 days, Metal | 0.14 to 0.20 s, 5 to 7 a second | unchanged |
| 3 | D7, D8, P3, P4, P6, then the CUDA twins | 1 to 2 weeks a backend | 0.03 to 0.10 s, 10 to 30 a second | 9,000 tokens in 10 to 30 s |

Waves 0 and 1 fit before the publication window of `docs/plan-inkling2-mini.md`. Wave 2 fits if the seal and the
card run leave three days. Wave 3 is after it.

On CUDA, D2 and D3 are milestone J5's gathered launch. The card budget is not touched by this plan: the first
compressed run and GSM8K keep it.

## 7. Items with a price, listed and not scheduled

| item | price |
|---|---|
| One rotation shared by a layer's experts (`fast_kernels.py:3398-3629`) | a re-encode and a KL gate |
| Router scores or partial sums in 16 bits | expert flips at near ties |
| Cache in f16 or int8 | rounding ahead of every router; halves the cache; never measured on this model |
| Draft head `mtp.safetensors` of the model, verified 6 to 8 positions at a time (`inkling/drafter.py`) | 0.64 GB more on the device, a download, identity at near ties not guaranteed |
| Our own tree pass (`docs/ETAT.md` §5) | memory and identity, the reasons it was set aside on the 4B |
| Attention in the lattice code or in int4 | 0.0234 KL (*measured*); the operator kept 16 bits on 2026-11-22 |
| The 8-rows kernel with f16 rows, as it is | argmax 31 of 32 (*measured*); replaced by P1 |

## 8. How a wave is read

- Served tokens a second after a 32-token prompt and after a 9,000-token prompt, median of rounds with its range.
- Time to the first token of a 9,000-token prompt, first request and second request of a session.
- Memory after load and peak during the long prompt.
- The class gate of each item, with its mutant.
- A chat and a tool round trip through `llvq-runtime` beside the bench, not the bench alone.

## 9. What this plan does not establish

- The 0.225 s of the rest has no measured split. Every gain taken from it is a reader's estimate.
- The Mac is shared, and the figures of §1 are one session's.
- Nothing is timed on CUDA. The first card run of `docs/plan-inkling2-mini.md` gives that baseline.
- The source's own speeds are self-declared and were not run here.
- It does not reopen the choice of engine. An MLX port of the forward pass is another route and is not costed.
