# DRAFT, not stamped, not committed

# Preregistration. The prefill attention with a lower score peak: identity on a grid of shapes, the attention's share of the first token, the score peak, and three card rates, from 2,048 to 8,192 tokens of prompt

**Draft revision 3, written on 2026-11-02 for the operator's review. To be committed and TIMESTAMPED BEFORE the
run.** Operator go: **not given**. It cannot run at HEAD 06e1602: none of the code of §2 exists there, so §2 and its
host results come first (about 2 to 4 days of code, *estimated*, wave roadmap §5). Cost: about **$0.77** on l40sx1 as
a job of its own, *estimated* below (§8), timeout 60 min, so $1.80 at worst. A reduced form, steps 3a, 3c, 3d and 3e
only, can ride the operator's next card job for about **$0.09** (§3). Cap asked: **$2**, which covers one retry. No cap
is in force (ETAT §5), and one is owed before the next paid job. Spent before it: $248.84 over 237 rows of the registry
(*measured*, `docs/data/jobs.csv`). Five billed jobs of 2026-09-19 to 09-21 are missing from it, $3.19 (*computed* on
their measured durations, `wave5-deep/downloads/persistent-jobs/jobs-ps-all.json`), so at least $252.03 was billed
(*computed*). Marks ⟨…⟩ are filled before the stamp.

## 1. Why this run exists

The prefill keeps three f16 score tensors alive in every layer: the scaled scores, the masked ones and the softmax
output (`llvq-llm/src/model.rs:582-590` grouped, `:1682-1688` expanded). At the second chunk of an 8,192-token prompt
each is 32 × 4,096 × 8,192 × 2 B = 2.15 GB on the 4B, so 6.44 GB of peak, 8.05 GB at 14B and 12.9 GB at 32B
(*computed*). The product triplet's margin is 5 GB (ETAT §4).

Five things are not known, and the next prefill decisions rest on them:

1. **Which lower-peak variants keep the bits.** Three variants keep both cuBLAS calls of today and are exact by
   construction (`wave5-deep/prefill.md` §3.1). Two change the shapes cuBLAS sees. On this card only 2 of the 6
   prefill reshapes at 2,048 tokens and more kept their bits, 4 of 8 with the 5-token prompt (*measured*,
   `docs/mesures/gqa-ab-2026-11-01.txt:62-72`), so those two are T8 candidates, and this run counts where they hold.
2. **The attention's share of the first token.** It is *estimated* at 2.21-2.55 s of the 35.02 s at 8,192, from three
   dense windows of one job (`docs/mesures/ppl-scelles-2026-10-04-brut/ppl-{4b,8b,14b}.txt`). It decides whether a
   prefill attention kernel is worth its weeks once the GEMM route lands.
3. **The f16 tensor rate the card delivers with f32 accumulation.** NVIDIA's page gives FP16 and BF16 tensor at
   362.05 TFLOPS dense; a repository archive reads 181 (`docs/archive/portage-noyau-cuda.md:410`). The 8B-4B window
   difference suggests 187-271 achieved (*estimated*).
4. **The FP64 instruction rate.** The exact dequant C-dense of the GEMM route costs 81 G FP64 instructions a 4B pass
   (*computed*), 0.11 s at the 0.716 T/s the spec arithmetic gives (*computed*, 142 SMs × 2 units × 2.52 GHz).
5. **Whether the 4-row kernel `tv_tetra48_rows_h` returns the one-row kernel's bits on the card.** It is proved on the
   host only (`llvq-llm/kernels/tv_tetra48_h.cu:137-141`).

## 2. What the code will hold before the stamp

To be written, and its host results filled in here before stamping:

- `llvq-llm/src/model.rs`:
  - an `AttnRows` setting on every block, `Off` (default, every published path) or one of:
    - `Lifetimes`: the first score binding dropped before the softmax, the masked one before the PV product;
    - `InPlace(r)`: both GEMMs as today; scale, mask and softmax per block of `r` rows, written back into the score
      buffer with `slice_set`;
    - ⟨`InPlaceAll`⟩: both GEMMs as today; candle's own affine, binary-add and softmax kernels in place on the whole
      buffer through `inplace_op1`; written only if the operator wants it before the stamp, else its rows below go;
    - `Rows(r)`: query rows in blocks of `r`, each through the existing function, joined by `Tensor::cat`;
    - `RowsTrim(r)`: the same, with keys narrowed to the causal limit of the block, and lean blocks;
  - with `l ≤ r`, the existing function runs verbatim. Set through `Qwen3::set_attn_rows` only.
- `llvq-llm/src/fused_cuda.rs`: a public check of `tv_tetra48_rows_h` against `tv_tetra48_h` on one record, for step
  3d. Both launchers are private today (`launch_tetra48_rows_h` at `:2306`, `launch_tetra48_h` at `:2374`), and the
  load-time `check_int4_half_parity` (`:2224-2260`) is the pattern.
- Host tests on the Mac CPU, *results to be filled*:
  - both forms, f32 and f16, l not a multiple of r, r ∈ {1, 7, 64}, Cat- and Grow-shaped k and v: every variant equal
    to `Off` bit for bit on the CPU;
  - mutants killed: wrong mask rows, wrong trim offset, reversed concatenation, a block skipped, a block written back
    at the wrong offset, the scale applied after the mask;
  - every matmul operand the variants build passes `cuda_gemm_layout_check`, a replica of candle 0.9.2's CUDA layout
    rules (`cuda_backend/mod.rs:1341-1432`); its mutant, the layout of gqa-ppl deviation É1, is refused;
  - the pool calls and the FP64 probe's launcher live in `llvq-cuda` and pass, under `nice`,
    `CUDARC_CUDA_VERSION=12040 cargo clippy --target x86_64-unknown-linux-gnu -p llvq-cuda --all-targets`. The
    `cfg(linux)` side of `llvq-llm` does not build on the Mac, so the image is its first compiler, and gate 1 is its
    check.
- `llvq-llm/src/bin/fusedrun.rs`: a measurement mode `LLVQ_PREFILL_AB=<lengths>`, or `LLVQ_PREFILL_AB=micro` for
  steps 3a, 3c, 3d and 3e alone, on the bench door. It is refused beside `LLVQ_CONFIG` (its name joins the list at
  `fusedrun.rs:678-680`) and beside every other measurement mode. It sets the grouped form on `Grow(256)` itself, as
  `LLVQ_GQA_AB` does, and turns cudarc's events off on the single stream. Its bitwise comparators cover the f16
  attention outputs and the f32 logits of 3b.
- A kernel `fp64_probe`, compiled by NVRTC beside the others (`wave5-deep/code/prefill/fp64_probe.cu`).
- Peak memory: the default memory pool's `CU_MEMPOOL_ATTR_USED_MEM_HIGH` (cudarc 0.19.8 `sys/mod.rs:4397`), reset to 0
  before each measured section and read after it, from `llvq-cuda`. The pool's `CU_MEMPOOL_ATTR_USED_MEM_CURRENT`
  (`sys/mod.rs:4396`) is read just before the section, so that a rise is defined (gate 6). This is one `unsafe` call at
  a hardware boundary.
- No new binary. `ops/jobs/prefill-ab.sh` also samples `nvidia-smi --query-gpu=memory.used` every 100 ms, as a
  second channel.

## 3. What runs

**Base commit, the operator's choice, named here before the stamp: ⟨a or b⟩.**
- (a) debit-trims 06e1602 plus this prereg's code. The first-token predictions of §6 stand as signed.
- (b) the tip of the operator's branch for audit steps 2-5 plus this code. The step-2 fusions change the prefill's
  time, so `Off`'s first-token predictions are re-signed from that branch's own measured sweep before the stamp.
  Fusions that are not bit for bit against candle also move the decoded texts, so under (b) gates 4 and 4b read the
  hashes that branch's own replay printed, named here before the stamp.

**Form of the run, the operator's choice: ⟨full or riding⟩.**
- Full: one image built from the commit that carries this file, one job, l40sx1, all steps below.
- Riding: steps 3a, 3c, 3d and 3e (`LLVQ_PREFILL_AB=micro`) appended to the operator's next scheduled card job, about
  3 min. That job's own gates apply, and gates 3, 5, 6 and 7 below are added (3d loads the file, so gate 3 holds there
  too). Step 2b and gate 4b are added too, 1.5 min
  more, unless that job already replays the served door. Step 3b does not run, and the rules that need it (R1's
  end-to-end clause, the first-token parts of §6) wait for a full run.

The object: the sealed 4B (`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`), the served flags spelled
out from `configs/qwen3-4b-tetra-e4-grouped.json` (sha256 `2455fe85…`).

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The bench door, replayed.** `fusedrun <file> 256` with the dense arm, as in step 2 of the run of 2026-11-02. It
   checks that the default path did not move.
   - **2b. The served door, replayed.** `LLVQ_CONFIG=configs/qwen3-4b-tetra-e4-grouped.json fusedrun <file> 256`. The
     dispatch on `AttnRows` sits in `forward_cached`, whose unexpanded branch (`model.rs:1648-1657`) is the served 4B
     path, grouped on `Grow(256)`. The bench door runs the expanded form on `Cat` and does not reach it.
3. **`LLVQ_PREFILL_AB=2048,4096,8192`**, in five sub-steps.
   - **3a. Attention alone, on a grid.**
     - The grid: prompts P every 256 from 256 to 8,192, plus 16 pseudo-random P in [2, 8,192] from a seed printed
       first, each mapped to the (l, L) chunk pairs of the prompt driver (`tokens.chunks(4096)`, `model.rs:2190-2203`),
       deduplicated: about 48 pairs (`identity_grid`, `wave5-deep/code/prefill/attn_rows.rs`).
     - Seeded random f16 q, k and v laid out as the stores lay them out (`Cat` for the expanded form, `Grow(256)` views
       for the grouped form), and the causal mask of `build_causal_mask`. k and v are built by appending to a
       `KvCache` on that store, one append per chunk, so the batch strides cuBLAS sees are the served ones: a
       `Grow(256)` buffer holds whole chunks of 256 positions (`llvq-llm/src/kvq.rs:420-426`). The expanded call includes the
       `repeat_kv(..).contiguous()` of k and v, as `forward_cached` does, so that it times what the dense window timed.
     - Both forms; the variants `Off`, `Lifetimes`, `InPlace(512)`, `InPlace(1024)`, ⟨`InPlaceAll`⟩, `Rows(512)`,
       `Rows(1024)`, `RowsTrim(512)` and `RowsTrim(1024)`.
     - At each pair: `Off` once, then each variant against it, bitwise (the count of differing elements and max |Δ|).
     - Times at (2,048, 2,048), (4,096, 4,096) and (4,096, 8,192) only: 1 discarded and 5 timed rounds, each round a
       burst of 10 calls between two fences, every variant in every round in the same order.
     - The pool high-water of each variant at (4,096, 8,192), grouped form.
   - **3b. End to end** (full run only).
     - Round 0, untimed: `Off`, `Lifetimes`, `InPlace(512)`, `InPlace(1024)`, ⟨`InPlaceAll`⟩, `Rows(1024)` and
       `RowsTrim(1024)`, each prompt of 2,048, 4,096 and 8,192 in chunks of 4,096, built as `LLVQ_CONTEXTS` builds
       them (`fusedrun.rs:438-494`), so that `Off` meets the served sweep's prompts. The prompt's last-position logits
       against `Off` bitwise, then 256 greedy tokens against `Off` (first divergence). The pool high-water of each
       prefill.
     - Rounds 1 to 3, timed at 8,192: `Off`, `InPlace(512)` and `RowsTrim(1024)`, interleaved in a fixed order. The
       ratios `t_Off / t_arm` are formed round by round.
   - **3c. f16 GEMM probe.** Random f16 operands through candle's matmul at the 4B's prefill shapes (4,096 × 2,560 by
     2,560 × 9,728; 4,096 × 9,728 by 9,728 × 2,560; 4,096 × 2,560 by 2,560 × 4,096) and at the grouped score shape
     (batch 8, 16,384 × 128 by 128 × 8,192). 1 discarded and 5 timed rounds, each a burst of 20 calls between two
     fences; achieved TFLOPS = 2mnk / t.
   - **3d. Rows against one row.** For one real record at each Tetra width (2,560 and 9,728): `tv_tetra48_rows_h` on 4
     pseudo-random rows against 4 launches of `tv_tetra48_h`, bitwise.
   - **3e. FP64 probe.** `fp64_probe`, grid 142 × 8 blocks of 256 threads, 4 chains, at 4,096 iterations and at 0. 1
     discarded and 5 timed bursts of 10 launches between two fences; rate = instructions / time.

## 4. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The sealed file and the config at their bytes and sha256 on the mount.
3. Every load prints `int4: tv_q4_hh equals tv_q4_h bit for bit`.
4. Step 2 prints `256 tokens identical to the dense arm`, and its decoded line hashes to `8784ffc4…`.
   - **4b.** Step 2b's provenance line names `attention grouped on grow(256)`, it prints `No dense arm on this path`, and
     its whole decoded text hashes to `0597cee8…`, the bytes both served configs produced on 2026-11-02
     (`docs/mesures/served-grouped-2026-11-02-brut/lines.sha256`).
   - Under base (b), the two hashes of gates 4 and 4b are those the branch's own replay printed (§3).
5. Step 3 prints `cudarc event tracking: off (one stream, ordered by issue)`, the grid's seed, one line per (pair, form,
   variant), and its other tables in full.
6. **The instruments discriminate.**
   - Each bitwise comparator, f16 for the attention outputs and f32 for the logits, flags a planted one-ulp change in
     one element of a copy of its reference.
   - The pool high-water read after the `Off` attention at (4,096, 8,192), grouped form, exceeds the pool's current use
     read just before it by at least 2.1 GB (3S = 6.44 GB expected, *computed*).
   - The FP64 probe at 4,096 iterations takes at least 20 times its time at 0 iterations: a loop the compiler removed
     would read the same.
   - A comparator or a probe that cannot fail proves nothing.
7. The header of every step reads `tile 64`, and no `LLVQ_*` variable is in the job environment, each step passing
   its own on its command line.

## 5. Outcomes reported, never gated

- 3a: for each variant and form, the number of grid pairs identical out of the total, and for the others the count of
  differing elements and max |Δ| per pair; the times; the high-water.
- 3b:
  - for each arm and length, whether the logits are identical (else max |Δlogit|), whether the tokens are identical
    (else the first divergence), and the high-water;
  - at 8,192, the time to the first token of each timed arm, median and range, and the ratio, round by round, median
    and range.
- 3c: TFLOPS per shape. 3d: identical, or the first differing row. 3e: T instructions/s, median and range.
- The nvidia-smi peak over the job.

## 6. Signed predictions

| quantity | point | interval |
|---|---|---|
| `Lifetimes`, `InPlace(512)`, `InPlace(1024)`, ⟨`InPlaceAll`⟩: every grid pair, both forms, and 3b | **identical** | certain, by construction |
| `Rows(1024)` identical at the three served shapes in both forms | **yes** | probability 0.25 |
| `Rows(512)`, `RowsTrim(512)`, `RowsTrim(1024)`, the same | **yes** | probability 0.15-0.2 |
| `Rows(1024)` identical on the whole grid in both forms | **no** | probability of yes 0.1-0.15 |
| `Rows(512)`, `RowsTrim(r)` identical on the whole grid in both forms | **no** | probability of yes 0.1 |
| `Off`, grouped, one call at (2,048, 2,048) / (4,096, 4,096) / (4,096, 8,192), ms | **5.5 / 22 / 44** | [4, 8] / [17, 28] / [35, 56] |
| `Off` expanded over `Off` grouped, time, at each of the three shapes | **1.00** | [0.85, 1.15] |
| attention of an 8,192 prefill, 36 × (t(4,096, 4,096) + t(4,096, 8,192)), grouped `Off`, s | **2.4** | [2.0, 2.8] |
| the same, expanded `Off`, s | **2.35** | [2.0, 2.7] |
| `InPlace(512)` over `Off` time, (4,096, 8,192) | **1.13** | [1.04, 1.25] |
| `RowsTrim(1024)` over `Off` time, (4,096, 8,192) | **0.82** | [0.72, 0.95] |
| `Rows(1024)` over `Off` time, (4,096, 8,192) | **1.00** | [0.93, 1.10] |
| `Off` time to the first token at 8,192, base (a), s | **35.02** | ±3 % |
| `t_Off / t_InPlace(512)` at 8,192 | **0.991** | [0.980, 1.000] |
| `t_Off / t_RowsTrim(1024)` at 8,192 | **1.017** | [1.005, 1.030] |
| pool high-water, `Off`, 8,192 prefill, GB | **9.5** | [8.5, 10.8] |
| pool high-water, `InPlace(512)`, 8,192 prefill, GB | **5.8** | [5.0, 6.8] |
| pool high-water, `RowsTrim(1024)`, 8,192 prefill, GB | **4.3** | [3.6, 5.2] |
| 3c, gate/up shape, TFLOPS | **225** | [170, 330] |
| 3d, rows against one row | **identical** | probability 0.8 |
| 3e, FP64, T instructions/s | **0.70** | [0.45, 0.75] |

Reasoning, *computed* or *estimated* as marked:
- **Identity of the exact variants.** Both GEMMs keep today's operands, shapes and strides; scale and mask add are
  elementwise, and the mask add normalizes the sign of zero; the softmax is per row; `slice_set` is a copy
  (`prefill.md` §3.1).
- **Identity of the row variants.** gqa-ab measured 8 prefill reshapes on this card, and at 2,048 tokens and more 2 of
  6 kept their bits. The shared
  form, which changes only the batch count and strides, moved at 4,096 and 8,192. The grouped form failed at the second
  chunk of 8,192 (L = 8,192), the case a row block meets. Identity flips with L at a fixed form, so a whole grid
  passing is less likely than three shapes passing.
- **Times.** Three dense windows of one job put the attention of one 4,096² pass at 0.74-0.85 s, so 20.5-23.6 ms a
  layer, and twice that at L = 8,192 (*estimated*, `prefill.md` §3.2). The windows ran the expanded form; the grouped
  form's prefill reads +0.5 % at 8,192 on the rows path (*measured*, served-grouped).
- **`InPlace(512)`.** One extra read and write of S per layer, 4.3 GB at the second chunk and 2.15 GB at the first,
  about 0.3 s over an 8,192 prefill (*estimated*).
- **`RowsTrim`.** Its staircase covers 0.81 of the (4,096, 8,192) rectangle at r = 1,024 and 0.75 of both chunks
  (*computed*), so about 0.55-0.65 s less at 8,192 (*estimated*).
- **Memory.** 1.38 GB of weights, 1.21 GB of grouped cache at 8,192 (*computed*), the score tensors (3S, S + 2 S_b, or
  2 S_b for a lean block), and about 0.6 GB of activations and workspace (*estimated*).
- **3c.** 27.1 TFLOP of extra 8B projections ran in about 0.10-0.145 s inside the dense windows (*estimated*).
- **3e.** AD10x carries 2 FP64 units per SM; the probe has 4 independent chains a thread to hide latency.

## 7. Decision rules

- **R1, an exact variant is eligible.** `Lifetimes`, `InPlace(r)` or ⟨`InPlaceAll`⟩ is eligible if all hold:
  - it is bit for bit at every grid pair in both forms;
  - it is bit for bit end to end, logits and 256 tokens at 2,048, 4,096 and 8,192 (a full run);
  - it lowers the measured high-water;
  - for an arm timed end to end, its ratio range at 8,192 lies at or above 0.98; for the others, the ratio of its
    micro time to `Off`'s at (4,096, 8,192), formed round by round, has its whole range at or below 1.25.

  The eligible variant with the smallest high-water is proposed for the served path behind a config field (absent
  means `Off`), with no T8. A riding run can make a variant eligible on the grid only; its end-to-end clause waits for
  a full run.
- **R2, the row variants are T8 candidates.** `Rows(r)` and `RowsTrim(r)` change cuBLAS shapes, and the served prefill
  meets every (len, L), so no grid makes them exact. The report gives, per variant and form, the identical pairs out
  of the grid, max |Δ|, max |Δlogit| and the first token divergence end to end, the high-water and the time ratio.
  - If one is identical on the whole grid and end to end, the operator may rule that the grid suffices. The default is
    T8.
  - `RowsTrim(1024)` becomes the third arm of `prereg-prefill-2.md` step 5 if its round-by-round ratio range at 8,192
    lies at or above 1.01, or, in a riding run, if the ratio of its micro time to `Off`'s at (4,096, 8,192), formed
    round by round, has its whole range at or below 0.90.
  - Otherwise it waits for the prefill attention kernel d2. No quality control runs on this job's strength.
- **R3, the harness.** If `Lifetimes`, `InPlace(r)` or `InPlaceAll` differs from `Off` at any grid pair or end to end,
  the harness or the exactness argument is wrong. The identity results are void, nothing is proposed, and the code is
  fixed first; for `InPlace`, the strict form with contiguous blocks is tried first.
- **R4, the attention's share.** A(8k) = 36 × (t(4,096, 4,096) + t(4,096, 8,192)), from 3a's `Off` medians.
  - The priority reads the grouped form, the served one:
    - below 0.8 s, a prefill attention kernel waits behind decode work;
    - from 0.8 s to under 1.5 s, it stays at mid priority;
    - at 1.5 s or more, it moves ahead of the other prefill work once the route lands.
  - The bound reads the expanded form, the dense window's form. Outside [2.0, 2.7] s, the decomposition of
    `prefill.md` §3.2 is broken and is redone before the GEMM-route prereg is stamped. Otherwise it stands.
    - 2.7 s is the most the dense window allows, with L4 at its physical floor.
    - With its 0.1 s print resolution, the 8B window bounds L4 at 0.135-0.280 s, so the expanded A(8k) at 2.12-2.61 s
      (*computed*, `calc/ttft_model.py` §3). A reading under 2.0 s breaks the same algebra from below, and with it
      the 0.45-1.2 s of a prefill attention kernel, which rests on L4 directly.
- **R5, the GEMM rate.** If the best of 3c's prefill shapes reads above 181 TFLOPS, the half-rate peak for f32
  accumulation is refuted on this card, since a measured rate cannot exceed its peak. At 181 or less the question stays
  open: a low probe cannot separate a low peak from an inefficient shape. The number feeds later predictions only.
- **R6, rows against one row.** If identical, the rows path design of `prefill.md` §3.3 stands. If not, the operator
  is told that today's prefill already differs from decode in its projections. No served change follows either way.
- **R7, the FP64 rate.**
  - At 0.5 T instructions/s or more, C-dense's estimated pass of 0.15-0.25 s stands, and the branch builds C-dense
    first.
  - Below 0.3, a C-dense pass costs at least 0.27 s at peak and 0.35-0.54 s in practice, about 20 % of the route's
    first token for two passes, so C-rot16 is weighed first.
  - From 0.3 to under 0.5, C-dense goes first, and the timed pass of `prereg-prefill-2.md` decides.
- **R8, the ROADMAP §4 hold.** ROADMAP §4 holds that serving the prefill "reopens the layout". No proposal of R1 or R2
  is served before the operator's reading of that hold (wave report, decision 9). Serving remains his decision in
  every case.
- **Otherwise**, for example if `Off` misses its first-token prediction by more than 3 %: the timing tables are
  reported as another protocol, the identity results stand, and no proposal rests on a time.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h (`ops/run.py:105`):

| item | minutes |
|---|---|
| container and image pull | 3 |
| oracle | 2 |
| step 2, with the dense arm's load | 3 |
| step 2b | 1.5 |
| 3a: about 48 pairs × 2 forms × 9 variants, calls of about 12 ms on average, plus the timing bursts at three shapes | 2 |
| 3b, round 0: 7 arms × (8.2 + 17.0 + 35.0 s of prefill + 3 decodes of 256 tokens) | 8 |
| 3b, rounds 1 to 3: 3 arms × 3 rounds × 35 s | 5.3 |
| 3c, 3d and 3e | 1 |

In all about 26 min, **$0.77**. The anchor is served-grouped, 18 min for $0.54 for two sweeps of the same lengths
(`docs/data/jobs.csv`). Timeout 60 min, **$1.80** at worst. Riding: 3a, 3c, 3d and 3e, about 3 min, **$0.09**, and
$0.05 more with step 2b.

## 9. Deviations

Any departure goes in `proofs/preregistration-prefill-rows-<date>-ECARTS.md`, beside this file, never inside it.

## 10. What this run does not establish

- Identity beyond the grid: a variant that changes a cuBLAS shape is certified on the grid's pairs only.
- Quality: an exact variant needs none. A row variant needs its own T8 control.
- The GEMM route, its dequant and its first token: a separate prereg (`prereg-prefill-2.md`).
- A prefill attention kernel's time: 3c bounds it, nothing more.
- Anything at 8B or 14B, on another card, or on Metal.
- Quality at 8,192 of context: no harness scores past 4,096 positions.
