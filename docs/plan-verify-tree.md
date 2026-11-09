# Tree verify on 4E: plan

The tree verify fails its first card gate and the route stops (S1, 2026-11-09, $0.12). On the L40S at the chat
context, a one-row pass costs 2.80 4E tokens and a 16-row pass 3.09, against a planned 1.09 and 1.27 (*measured*,
[verify-tree-s1](mesures/verify-tree-s1-2026-11-09.txt)). An extra row costs 0.0196 of a token, as planned; the
pass's fixed cost is the gap. Where that cost goes is not measured: the bench stopped before its phase table
(deviation 1 of the prereg). Identity holds on the card, and the one-row pass makes 4E's token at 64 of 64 positions.
At the measured c_16 the 4B with EAGLE-3 would decode about 155 tok/s on chat, below the 221 of greedy 4E
(*computed*).

The plan below is kept as written before S1. Step A held at $0: with the published EAGLE-3 head, a tree of 15 drafts
emits 2.28 tokens a pass on chat and 3.15 on GSM8K at 3 head steps, against 1.69 and 2.42 for the chain of 3
(*measured*, [eagle-tree](mesures/eagle-tree-2026-11-09.txt)). The kernel passes its host gates: rows keep their bits
at R = 1 to 32, R = 1 sits within 3.5e-3 relative RMS of 4E's logits, 10 mutants of 10 die (S0, *measured*,
[tree-verify-s0-host](mesures/tree-verify-s0-host-2026-11-09.txt)). The plan estimated the pass at 1.18 greedy tokens
at 8 rows and 1.27 at 16 (*estimated*, section 2), hence about 352 tok/s on chat and 484 on GSM8K (*computed*,
section 4); S1 measured 2.92 and 3.09.

## 1. Why a tree on tensor cores

At batch 1 a token reads the whole model once: 1.59 / 2.79 / 5.32 ms of weight bytes at 864 GB/s for the 4B / 8B /
14B (*computed*, `recherche-debit-maths.md` §11). Verifying several drafted tokens in one pass reads them once for all.
Today's verify pass (`llvq-cuda/kernels/pers_4ev.cu`) cannot use that: an extra row costs 0.74 / 0.89 / 0.98 of a 4E
token (*measured*, [spec-final](mesures/spec-final-2026-11-08.txt)). Two parts of the kernel make it so.

- **The prologues run once a row a CTA.** 4E has every CTA compute ln1, ln2 and the down's input over the whole vector
  (`llvq_pers.cuh` §7b), to save a grid wait. The verify repeats them row after row. At one row they take about
  1.47 ms of 4E v4's 3.71 ms of row phases (*measured*, [shaker-4e](mesures/shaker-4e-2026-11-05.txt) inner stamps):
  about 45 % of today's extra row (*estimated*).
- **The row loop is an f32 FMA chain a row.** Each block is decoded once, then 24 FMAs and 24 shared loads a row. A
  second row costs 0.38 of a row pass (*measured*, shaker-4e), and the verify loop sits 2.4 times above that floor.

The plan removes both. Each block is decoded once into f16 (its two scales folded, as `tv_batch_gemm.cu` does) and
multiplied for up to 8 rows by one `mma.m16n8k16`, 16 rows by two. Each row's prologue runs once on the grid, on one
CTA, behind one more grid wait. An extra row then costs its activation bytes and its attention, about 0.01 of a token
(*estimated*, section 2). That pays for a tree of drafts instead of a chain of 3.

## 2. What a verify pass of R rows would cost

c_R is the pass in one-token 4E v9 launches (4.42 / 6.95 / 11.00 ms, spec-final). All *estimated*, at 4B:

| part | in a 4E token | fixed part of the R-row pass | each extra row |
|---|---|---|---|
| Tetra, int4 and head rows | 2.72 ms (1.81 + 0.55 + 0.36) | times β = 0.9 to 1.3 | 0 to 0.005: one more n8 tile every 8 rows |
| prologues: ln1, ln2, down input | every CTA, whole vector | once a row on one CTA, plus 3 grid waits a layer: +0.08 ms | 0 while R stays under the grid's 142 CTAs |
| activation into shared memory | one f32 vector a phase | | 0.004 to 0.013: 106 MB a row a pass (R × d_in × 2 bytes a CTA a phase, the down split over d_in) from L2 at 5.7 to 2.3 TB/s (fitted, `recherche-debit-maths.md` §5), plus margin |
| attention, under 500 positions | 0.66 ms | the same jobs | 0.002 to 0.004 |
| head argmax, residual, EAGLE states | | | about 0.001 |
| sum, in 4E tokens | 1 | Δ = -0.05 to +0.25, central +0.09 | δ = 0.005 to 0.023, central 0.012 |

c_R = 1 + Δ + (R - 1) δ, with δ 0.006 to 0.025 past 16 rows:

| R | low | central | high |
|---|---|---|---|
| 4 | 0.97 | 1.13 | 1.32 |
| 8 | 0.99 | 1.18 | 1.41 |
| 16 | 1.03 | 1.27 | 1.60 |
| 32 | 1.14 | 1.56 | 2.03 |

- **β is the main unknown.** v2 has about 970 instruction slots a block (*computed*: 118.7 M blocks in 2.51 ms on 568
  sub-partitions at 2.52 GHz) and issues about 160 instructions a block (*estimated*, from the code). It waits on
  latency, which the MMA form shortens: no FMA chain, an f16 pair out of a quad in two instructions. B1's shorter
  chain bought x1.143 (*measured*, [spec-micro](mesures/spec-micro-2026-11-08.txt)). Against that, the GEMM's single
  row runs at half 4E's speed.
- **The batch GEMM's 1.055 is not shared compute.** At B = 1 and B = 8 it runs the same `tv_bg_n8` entry
  (`batch_cuda.rs`, first entry with at least n rows), the 7 empty rows zero-filled by `cp.async`. Its 0.52 ms over 7
  rows is the work done a sequence outside the projections: attention, norms, argmax (*computed*,
  [batch-l40s-v2](mesures/batch-l40s-v2-2026-11-08.txt)).
  From 1 to 16 rows its step grows x1.107 / x1.000 / x0.968 at 4B / 8B / 14B: 0.015 / 0 / under 0 of a 4E token a row
  (*computed*). From 16 to 32 rows, 0.035 to 0.039 a row, each row on its own cache.
- The 8B and 14B take the same range: the staging bytes and the token both grow with the widths.

What the code demands, and how the plan meets it:

- **Registers.** 4E v9 holds 128 registers and 0 B of local memory at 512 threads; the f32 verify spills 144 to 272 B
  (spec-gate). An MMA row phase keeps 8 accumulators per 16 x 16 tile, 4 A and 4 B fragment registers and the
  decode's working set, under v2's 24 decoded floats. The tree verify is its own symbol: the served module does not
  move.
- **Shared memory.** 99 KB a block. 4E stages the whole f32 vector of a phase: 39 / 49 / 70 KB for the down at 4B /
  8B / 14B, which is why two rows of the down do not fit at 8B and 14B. Sixteen f16 rows of the 4B's hidden already
  take 80 KB. The tree verify stages K chunks through a `cp.async` ring, as the GEMM does: 6 KB a stage at R = 16 and
  192 columns. That frees the whole-vector area at every size.
- **Activation re-read.** Every CTA reads all R rows of a phase's input. With 18 output rows a CTA, the down would
  read 5 times more activation than weights at R = 16. Splitting the down over d_in in 8 slices, with an ordered fold,
  brings it to about 8 MB a layer with the partials, against 5.5 MB for one f32 row today (*computed*).
- **Attention with a tree mask.** Row r attends the prefix and its ancestors. Its keys are visited in logical order
  (prefix, then ancestors as if contiguous), so its result does not depend on the tree's shape. The accepted rows' K
  and V move to their positions in the next pass's first phase.
- **The head.** The int4 g64 table (218.8 MB at 4B) becomes a GEMM of R columns: 12 GFLOP at R = 16 under a 0.36 ms
  stream. Each row keeps its own argmax key.

Nothing in the code forces an extra row above 0.05 of a token once each prologue runs once a row: 0.005 to 0.023
above. Left as they are, the prologues alone cost about 0.33 of a token a row (*estimated*: 1.47 of 4.42 ms).

## 3. Tokens a pass, measured

Same sealed 4B through the MLX door, same head at fd331e59, same five prompts and 50 GSM8K problems as eagle-alpha;
the chain of 3 replays its numbers exactly (*measured*, [eagle-tree](mesures/eagle-tree-2026-11-09.txt)).

| tree | verify rows | head steps | chat | code | summary | edit | rag | GSM8K |
|---|---|---|---|---|---|---|---|---|
| chain of 3 | 4 | 3 | 1.69 | 2.04 | 1.74 | 2.44 | 3.00 | 2.42 |
| dynamic, 7 nodes | 8 | 3 | 2.12 | 2.36 | 2.25 | 2.70 | 3.55 | 2.90 |
| dynamic, 15 nodes | 16 | 3 | 2.28 | 2.77 | 2.48 | 2.70 | 3.55 | 3.15 |
| dynamic, 15 nodes | 16 | 4 | 2.34 | 2.77 | 2.41 | 2.94 | 3.90 | 3.28 |
| dynamic, 31 nodes | 32 | 4 | 2.45 | 3.04 | 2.83 | 3.23 | 3.71 | 3.55 |
| static, 15 nodes | 16 | 4-5 | 2.18 | 2.55 | 2.25 | 2.94 | 3.39 | 3.09 |

- Dynamic is EAGLE-2's tree: each level expands the 8 best candidates by cumulative probability. Static is a fixed
  set of rank paths chosen on the other five workloads.
- Depth past 4 buys little: the head's acceptance falls after its fourth draft.
- The five prompts are one completion each, 78 to 255 positions: standard errors 0.10 to 0.21 at 16 rows. GSM8K's is
  0.014 (*computed*).
- The head ran as published, BF16 in f32. Its int4 g64 form is not measured: the run was interrupted when the Mac
  froze on 2026-11-09 and was dropped.

## 4. Projection

Decode tok/s = τ / (c_R + d D) times greedy decode, with τ from section 3, c_R from section 2, d head steps and D =
0.05 [0.04-0.07] of a token a step (*computed* from *estimated* costs; greedy decode from spec-final). The 8B and 14B
reuse the 4B's τ: no head was run at those sizes. D is costed for an int4 head and τ is the BF16 head's.

| size | tree | chat | code | GSM8K |
|---|---|---|---|---|
| 4B | 8 rows, 3 steps | 353 [290-423] | 391 [321-469] | 473 [388-567] |
| 4B | 16 rows, 4 steps | 352 [275-435] | 416 [325-514] | 484 [379-598] |
| 8B | 8 rows, 3 steps | 226 [185-271] | 250 [205-300] | 301 [247-361] |
| 14B | 8 rows, 3 steps | 142 [117-171] | 157 [129-188] | 188 [155-226] |

- Greedy 4E decodes 221.0 / 141.3 / 89.1 on chat (*measured*, spec-final). The central chat gain is x1.60 at every
  size by construction: τ and c_R are the 4B's.
- On chat, 8 and 16 rows read the same; 16 gain on code and GSM8K; 32 rows lose at every workload.
- Kill rule of the task: at c_16 = 1.25 and D = 0.04 the 4B chat reads x1.66, over x1.3. The code holds no extra row
  above 0.05 of a token (section 2). Step A passes.
- On a tensor-core verify even the chain of 3 would reach x1.32 on chat; today's verify gives it x0.72.

## 5. Stages

| stage | what | gate | cost |
|---|---|---|---|
| S0 | the tree verify `pers_4et.cu`: MMA Tetra, int4 and head rows over K chunks, prologues once a row, tree-mask attention, KV move; host-checked through `host_grid.h` with the MMA modelled from the PTX fragments (`tests/host_mma.h`, `tests/tree_verify_host.rs`) | row r bit for bit equal at R = 1 to 32 and under slot permutations; R = 1 within 1e-2 relative RMS of 4E's logits on the host fixture; mutants killed (row order, scale association, mask, fold order) | $0; passes on the host, 2026-11-09 |
| S0b | the head's int4 g64 rounding on the Mac, `ops/eagle_tree.py ... int4` | tokens a pass within 0.1 of the BF16 head on GSM8K | $0, not measured: interrupted on 2026-11-09, needs the operator's go to run alone on the Mac |
| S1 | card microbench, `bin/vtreebench` (`ops/jobs/verify-tree-s1.sh`): R = 1, 2, 4, 8, 16, 32 tree passes beside the 4E token in one process, identity across R and slots, `vt1` against 4E on 64 positions, the phases of `vt1` and `vt16` | c_16 ≤ 1.25 passes; c_16 ≤ 1.60 or c_8 ≤ 1.48 (4B chat still at x1.3) goes to the operator; prereg `proofs/preregistration-verify-tree-s1-2026-11-09.md` | $0.12; FAIL on 2026-11-09: c_16 3.092, c_8 2.922 at the chat context; rag context and phases not produced (deviation 1) |
| S2 | the verify arithmetic as a greedy decoder at 4B, 8B, 14B (R = 1 rounds) | paired perplexity within ±0.12 % of 4E at the three sizes, 50-problem GSM8K pilot (section 6) | about $1.5 |
| S3 | the 4B door: int4 head and tree on the card, the round loop on the host | speculative tokens equal to the R = 1 verify decode's on 5 prompts and 50 GSM8K; 4B chat decode ≥ x1.3 of greedy 4E | about $1.5 |
| S4 | the 8B and 14B heads (RedHat, pinned) on the card | tokens a pass measured; doors ≥ x1.2 of greedy on chat | about $2 |
| S5 | a head retrained on Tetra features, optional | a $5-10 fine-tune pilot first: +0.15 tokens a pass on chat | $40 to 90 a size |

Every card job needs the operator's go, a spend cap, a timestamped prereg and a row in `docs/data/jobs.csv`. A failed
gate stops the route and reports. S1 reads the reopening condition of `ETAT.md` §6 (an extra row under 0.2 of a token,
a 2-row pass under 1.4); reopening stays the operator's call. The round loop is driven from the host: A2 (CUDA
Graphs) stays out of the core.

## 6. The quality gate

The served tokens become the argmax of the verify pass, whose arithmetic changes: f16 weights with their scales
folded, f16 activations, f32 accumulation in the tensor cores. Every round runs that arithmetic, a round without a
draft included, so the served object is one function of the prompt.

- **Identity.** The speculative decode's tokens equal the greedy decode of the R = 1 verify pass, bit for bit, at
  every prompt. That holds when row r's result depends only on row r: the K chunks, the d_in split and the fold order
  must not depend on R or on the row's slot. It holds on the host (S0); S1 and S3 check it on the card.
- **The band.** Paired perplexity of the R = 1 verify decode against 4E within ±0.12 % at 4B, 8B and 14B (16 × 2,048
  tokens, as batch-l40s-v2), the tolerance `ETAT.md` §4 gives the attention forms.
- **GSM8K.** The 50-problem pilot through the verify decode: correct counts against 40 / 46 / 46 for 4E, discordant
  pairs reported. The batch GEMM, the same arithmetic, read +0.0026 % [-0.0077 ; +0.0128] and 40 of 50 on both arms at
  4B (*measured*, batch-l40s-v2).

## 7. The head on the card

- **Form.** The EAGLE-3 layer and its 32k head in int4 g64: 123 / 225 / 343 MB, +0.24 / +0.22 / +0.19 b/param on the
  whole model (*computed*, `plan-speculation.md`). A step reads them once for the K = 8 rows of a tree level: 0.05 of
  a token at 550-700 GB/s, plus a step's fixed cost (*estimated*, D = 0.04 to 0.07). What int4 rounding does to the
  tokens a pass is not measured (S0b).
- **Inputs.** 4E writes each verify row's residual at three layer inputs, [2, L/2, L - 3] in EAGLE-3's convention: 2,
  18 and 33 at 4B and 8B, 2, 20 and 37 at 14B; the RedHat heads' configs decide. The residual is in the natural basis:
  the rotations act inside each projection.
- **The tree.** One head step a level: top-K of each expanded node's 32k logits, the K best cumulative scores kept, the
  N best of all levels sent to the verify with a parent index each. The draft vocabulary maps to the target's by d2t.
- **Retraining (S5).** The published head reads Tetra at the chain α of eagle-alpha. A head trained on Tetra's own
  features needs the target's states over about 50 M tokens: the dense reconstruction through `llvq-tetra` in
  SpecForge, one to two GPU-days a size, $40 to 90 (*estimated*). Never without the operator's go.

## 8. What this plan does not establish

- β: no MMA row loop has run inside 4E. The GEMM's single row is half 4E's speed.
- The card's arithmetic. S0 models the tensor core's sum as a sequential f32 sum, and NVRTC has not compiled the
  kernel's device branches. Identity on the card rests on each output element reading only its own column of B.
- The tokens a pass on the card. They are measured through the MLX door, whose greedy text parts from the CUDA door's
  at token 243 of the house prompt; the verify arithmetic will part again. S3 measures them on the card.
- Five of six workloads are one prompt each, 78 to 255 rounds. GSM8K, 50 problems, carries the weight.
- The 8B and 14B tokens a pass: no head was run at those sizes. Their projections reuse the 4B's.
- The tokens a pass with an int4 head: the projection pays an int4 head's step and assumes the BF16 head's acceptance.
- Long contexts: at 8,192 positions attention adds about 0.01 of a token a row (*estimated*), and the 4E token itself
  grows to 6.2 ms.
