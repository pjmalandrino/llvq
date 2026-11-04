# Lead "bigmodel": the 32B walls and the big-model choice

Written on 2026-11-02, read-only on the repository: no file there changed, no run, no build, $0. Revised the same day
after an adversarial review: 3 high and 9 medium findings, all fixed, and six cheaper experiments added to the ladder
(§10 lists each finding and what changed). Labels: *measured* (named journal or primary source), *computed*
(arithmetic on measured or code inputs), *estimated* (judgment, or arithmetic with one estimated input). [M], [C], [E]
in tables. Code sketches are in `wave5-deep/code/bigmodel/`, all drafts never compiled or run. Draft preregs:
`prereg-bigmodel-1.md` (gate K32, Mac) and `prereg-bigmodel-2.md` (rotation and int4 identity, card). Arithmetic:
`wave5-deep/calc/bigmodel/k32_numbers.py`.

## 1. Verdict

Both 32B walls lift bit for bit, and the lift is small. The rotation can stage in device memory instead of shared
memory with the same helpers in the same order (R0, about 25 lines of kernel), which is what the Metal port already
does and gates bit for bit against the CUDA text. The int4 `down_proj` can stage its activation as f16 bits, 51,200 B
instead of 102,400 at d_in 25,600 [C], with the same operand in every FMA; that kernel needs its own opt-in, and the
load path must size each int4 entry point separately (§3.6). Both are selected only where today's kernels are
refused, and both live in translation units of their own, so no kernel and no printed sha256 of the served 4B, 8B or
14B moves. The walls exist only on 99 KiB cards (L40S, RTX 5090, RTX PRO 6000); the A100 (166,912 B [M]) and the H200
(232,448 B, CUDA guide) pass the 32B as written. Code, host tests and one card identity job: about 3-4 days and
$0.30-0.39 [E].

A faster rotation (R1: two stages in registers, radix-4 passes without division, the mix on every thread with its
chain cut at 8, 24 or 32 terms) is the larger prize, and its size is the main unknown. The 4B's 96 rotations hold
about 0.72 ms of GPU work above the launch floor in isolation [C], and R1 would leave about 0.16 ms [E]. How much of
the 0.56 ms saved shortens the step is not known: the only in-situ datum removed whole launches, host side included,
in two processes, on an older object (§8.6). So R1 is worth ×1.02-1.08 at 4B, ×1.02-1.06 at 14B and ×1.02-1.16 for a
32B against R0 [E]. Two cheap steps bound it before 2-3 days of R1 code: the step-0 profile of the throughput audit
(V0b, $0 here if the operator's branch runs it) and a null-rotation arm, same launch with no work (V0c, half a day of
code and about $0.3).

The big-model choice needs no card to start. Gate K32 scores published GGUFs on the Mac at $0 against dumps the
repository already holds. Stage A (a BF16 engine check and the UD-IQ2_M 14B) settles D_ours, our sealed 14B against
the best published 2-bit rung, to about ±1.8 pp. Stage B (the UD-IQ2_M 32B) runs only if D_ours is above zero. Its
headline, a 2-bit 32B against a 4-bit 14B plus 1 pp, resolves only to about ±3.9 pp on the plan against a predicted
margin of 1.7 [E]: most likely unresolved, and the full split is the way out, at about six times the Mac time. The
order: the pilot and stage A on the Mac (operator go), V0b and V0c before any R1 code, and the wall lift only once
K32 points to a 32B on a 99 KiB card.

## 2. What exists today

| area | where | state |
|---|---|---|
| rotation reference | `llvq-quant/src/rotation.rs:50-66, 97-145, 192-212` | `Q = (Q_odd ⊗ H_m) D`, m the largest power of two in n, f64. At 25,600: m = 1,024, k = 25 [C] |
| CUDA rotation | `llvq-cuda/kernels/rotate.cu:38-67, 103-140` | one CTA, 1,024 threads (`fused_cuda.rs:2060`), the vector in f32 shared memory (`gpu.rs:813, 861`, `n * 4`), log2(m) barriers |
| its helpers | `llvq-cuda/kernels/llvq_rot.cuh:114-126, 159-169, 212-233` | phases take `(tid, nthreads)`; the WHT stage divides by a runtime `len` (`:163`); the mix uses m threads of 1,024 and pads its chain to KMAX = 32 |
| a wrong comment, left in place | `llvq_rot.cuh:46-48`, copied at `llvq-llm/kernels/llvq_rot.metal:97-98` | calls 25,600 = 512 · 50 too wide for KMAX. It is 1,024 · 25 [C]; the wall is shared memory. The correction belongs in docs: `llvq_rot.cuh` is part of the served unit whose sha256 every served log prints (`fused_cuda.rs:553`), and `llvq_rot.metal` ships byte-identical in the llvq-tetra wheel, digest `4d08e4cf…` in `tetra-tables.json`, checked at run time (`llvq_tetra/metal.py:62-79`) and by the fast-loop test `the_shipped_shader_is_the_repositorys` (`llvq-llm/tests/hfpack.rs:240`) |
| shared guard | `llvq-cuda/src/shared.rs:42-121`, tests `:142-186` | `rot_plan` refuses 25,600 by 1,024 B on L40S bounds; pinned by a test |
| rotation load path | `llvq-llm/src/fused_cuda.rs:756-783` | `rot_plan` per rotation, one opt-in per model for its widest rotation: none for the served 4B (38,912 B, and `func_dynamic_shared` then returns `func` untouched, `gpu.rs:423-447`), 69,632 B for the 14B; refuses a 32B on sm_89 |
| Metal precedent | `llvq_rot.metal:28-43`, `fused_metal.rs:880-920` | the rotation staged in a device scratch, bit-identical to the CUDA text; threadgroup staging measured +7.5 % slower at 2,560 (commit `31a5acb`, message only) |
| int4 kernels | `llvq-llm/kernels/tv_q4_h.cu:82-149` | `tv_q4_hh` widens f16 into f32 shared, whole activation |
| int4 load path | `fused_cuda.rs:641-660, 2116-2200`, `fused.rs:734-736` | ONE size for both entry points, `int4_shared_bytes(all records)` = max d_in × 4; `upload_int4` checks each record's d_in × 4 against the ceiling and stores it as that record's launch size. A 32B with one int4 `down_proj` asks 102,400 B and fails at load |
| int4 prefill | `fused_cuda.rs:941-951` | a row at a nonzero offset, or any non-f16 activation, is cast to f32 and goes to `tv_q4_h`: the f32 path stays walled at 25,600 |
| int4 card check | `fused_cuda.rs:2224-2261` | at load, `tv_q4_hh` against `tv_q4_h` bit for bit on the first int4 record |
| host tests | `llvq-cuda/tests/rotation_matches_rust.rs:42-51`, `tests/host_rotate.cpp:45, 104`; `llvq-llm/tests/host_tv_q4_h.cpp:15-19` | rotation widths up to 12,288, the shared stand-in holds 16,384 floats and refuses more, so neither 17,408 (served 14B) nor 25,600 is host-tested. The int4 kernel is compile-checked, never executed: only `q4_deq` and a hand-written mirror of the accumulation run |
| card bench | `llvq-cuda/src/bin/rotbench.rs:67-80, 114-128` | one kernel, nine widths incl. 17,408, the opt-in posed for the widest case on the whole function; in both Docker lists (`ops/Dockerfile.cuda:87, 147`) |
| latencies, isolated | `rotation-cuda-2026-08-05.txt`; `docs/data/campagne-14b-vitesse-2/rotbench.txt` | no opt-in, 2026-08-05: 8.05 µs (2,560), 6.72 (4,096), 19.30 (9,728), 26.23 (12,288). Opt-in at 69,632 B, 2026-08-17: 7.90, 6.70, 18.92, 26.15 and 29.76 (17,408) [M]. Chained on an empty stream |
| the only in-situ datum | `f1e0-2026-09-10.txt` §0 bis | sharing the rotation removed 72 launches at 2,560 (216 to 144): 94.2 to 100.8 tok/s, two processes, the tetra-q5 object of 2026-09-09 [M] |
| sites per token | wave-1 U1; the int4 sets of `configs/*-e4.json` | 96 (4B), 127 (8B), 101 (14B) [M]: two shared rotations a layer plus each Tetra `down_proj`, and each Tetra `o_proj` at 8B |
| GGUF harness | `ops/gguf_mmlu.py`, `ops/vllm_score.py:375-412`, `llvq-llm/src/bin/mmlupair.rs:185-196, 326-331` | scores a GGUF with `bin/mmlu`'s rule; its dumps carry no `# end` trailer, so `mmlupair` refuses them, and a census dump against a plan dump needs `--intersect` |
| 27B plan | `docs/ROADMAP-27B.md`, `ROADMAP-27B-DEV.md` | draft, not adopted; no wall (widest 17,408); DeltaNet step kernel in P4, 2 launches a layer |

What works: every served width, on every card. What is missing: any kernel that runs 25,600 on a 99 KiB card, a host
test past 16,384, any execution of an int4 kernel off a card, a card test that compares bytes rather than errors, a
measurement of what the rotation costs in situ, and any rotation faster than one SM walking log2(m) barriers.

## 3. Design

### 3.1 The exactness contract

Every output coordinate of the rotation is fixed by five steps: (1) f16 to f32 widening, exact; (2) the sign flip,
exact; (3) butterfly stages len = 1, 2, ..., m/2 in that order, each mapping (a, b) to (a + b, a − b) with a at the
lower index, f32 round-to-nearest; (4) `col = y * inv`, one multiply by the host-narrowed `inv`; (5) the chain `acc =
fma(small[g][t], col[t], acc)` for t ascending from acc = +0.0f. There is no reduction across threads, and no `a*b+c`
other than the explicit `__fmaf_rn`, since the col product is a multiplicand. NVRTC runs with its defaults
(`gpu.rs:14-16, 270-276`: fmad on, no ftz), and f16-derived values are normal in f32, so nothing can contract or
flush. Any thread mapping, any stage grouping that keeps the stage order, and any memory space give the same bits. The
2026-08-05 job already saw the card reproduce the Mac's error figures to the printed decimal for this reason; the
gates below compare bytes instead.

**Cutting the chain at KB ≥ k.** The terms t in [k, KMAX) are `fma(0.0f, 0.0f, acc)`: the host pads `small` with zeros
and `col[t] = 0.0f` past k. `0 × 0 = +0`, and `acc + (+0) = acc` for every acc except −0. acc starts at +0. An fma
returns −0 only for (−0) + (−0), impossible from a +0 accumulator, or by underflow of a nonzero exact result below
2^−150. Here every nonzero WHT value is a multiple of 2^−24 (f16 inputs), `inv` ≥ 2^−7, and the Gram–Schmidt entries
sit far above 2^−30 (checkable once per table at load), so every nonzero product exceeds about 2^−61 and any nonzero
exact sum is a multiple of about 2^−107: no underflow occurs and acc is never −0 [C]. Non-finite values pass unchanged
(`fma(0, 0, ±Inf) = ±Inf`; a NaN accumulator is already the canonical NaN an FFMA emits). The cut changes no bit. It
matters: at k = 5, the most frequent sites, the padded chain does 6.4 times the useful FMAs.

### 3.2 R0: the minimal wall lift

`rot_apply_dev` and `rot_apply_rows_dev` (sketch: `code/bigmodel/rotate_wide.cu`) call `rot_load`, `rot_wht_step` and
`rot_mix` on a device `scratch` instead of `extern __shared__`. `__syncthreads()` orders global accesses within a
block, so the same barriers suffice. Host contract: a scratch distinct from `xout`, sized `PREFILL_ROWS * n + (KMAX −
k) * m` floats, so a speculated read of the unrolled col loop stays allocated (the Metal port clamps the index for the
same reason); zero dynamic shared memory; same grid, block and arguments otherwise. Bit identity holds by construction
against `rot_apply`. It is selected only where `rot_plan` refuses. Expected cost at 25,600: 25-90 µs a site, one SM
walking 10 stages through L1 and L2 [E]. With no dynamic shared memory requested, the carveout can leave most of the
SM's 128 KB to L1, enough for the 100 KB working set if little else competes [E]. Metal saw device staging cost no
more than threadgroup staging, since the working set sat in cache.

### 3.3 R1: the fast rotation for today's widths

`rot_apply_r` keeps one CTA of 1,024 threads and n * 4 bytes of shared memory, so it drops into today's call sites.
Its phases (sketch, same file):

1. **Load and two stages in registers.** Thread c owns coordinates 4c..4c+3: widen, flip (the four sign bits share
   one word), stage 1 on pairs (0,1), (2,3), stage 2 on (0,2), (1,3), one float4 store. Coalesced and conflict-free.
2. **Radix-4 passes.** For lg = 2, 4, ... while 4·2^lg ≤ m, a thread owns the quad `j0, j0+len, j0+2len, j0+3len`
   with `j0 = ((q >> lg) << (lg+2)) | (q & (len−1))` and applies stage len then 2len to it: the four pairs those two
   stages touch there. One radix-2 pass finishes an odd stage count. No division (today's stage computes
   `p / len` and `p % len` on a runtime divisor).
3. **Mix on every thread.** P = 1,024 / m threads share a column (2 at m = 512), each taking a contiguous slice of the
   k outputs, so the slice is warp-uniform and `small` reads are broadcasts. The chain is cut at KB = 8, 24 or 32 by a
   uniform branch on k (template instances), per §3.1.

Barriers fall from log2(m) + 1 to 1 + ⌈(log2 m − 2)/2⌉: 10 → 5 at m = 512, 11 → 5 at 1,024, 13 → 6 at 4,096 [C].
Registers: about 25 in phases 1-2 and 40-48 in the mix [E] (today's kernel: 47 [M]), under the 64 that 1,024 threads
allow. No shuffle and no `__constant__` table, so the phases replay on the host at any thread count, as today's do. R1
inherits the load path's one opt-in per model, so it is timed on the same two objects as `rot_apply`: no opt-in for
the 4B, 69,632 B for the 14B (prereg 2, the two handles).

Estimated work [E]: about 1.0-1.5 µs at 2,560 and 2.5-3.5 µs at 9,728, against 4.7 and 16.0 µs today (isolated minus
the 3.33 µs launch floor [C]). The 2026-08-05 bench rule applies: a source-level count has failed by 2× on this
codebase, so only the card decides.

### 3.4 R1w for the 32B, and R2 as the fallback

`rot_apply_rw` is R1 for widths past the ceiling. Groups pass through shared memory gb at a time (gb · m · 4 B; at m =
1,024, gb = 12 is 48 KiB, on the default bound, so 3 batches at 25,600), each finished group goes to a device scratch,
and the mix reads the scratch after one barrier. One launch, one CTA, no inter-block protocol; the per-element
operations are those of §3.1, so the bits are `rot_apply`'s. Expected 6-15 µs at 25,600 [E].

If R1w measures slow, R2 spreads the work: k CTAs, one per group, and the last CTA to finish does the mix
(`__threadfence`, an `atomicAdd` ticket per row, `__ldcg` reads, the ticket re-armed by the last CTA). One launch, no
spin-wait, same bits whichever CTA arrives last. It needs a device counter per row and a one-stream contract, which is
why it is not first. The two-kernel split that `rotate.cu:14-18` names costs one extra launch a site and gains nothing
over R1w.

### 3.5 Geometry and budgets

| kernel | grid × block | dynamic shared | other memory | barriers (m = 512 / 1,024) | launches a site |
|---|---|---|---|---|---|
| `rot_apply` today | 1 × 1,024 | n · 4 (102,400 at 25,600) | none | 10 / 11 | 1 |
| R0 `rot_apply_dev` | 1 × 1,024 | 0 | scratch n_rows·n + (32−k)·m floats | 10 / 11 | 1 |
| R1 `rot_apply_r` | 1 × 1,024 | n · 4, unchanged | none | 5 / 5 | 1 |
| R1w `rot_apply_rw` | 1 × 1,024 | gb · m · 4 (48 KiB at gb = 12) | scratch as R0 | 6 per batch, 3 batches at 25,600 | 1 |
| R2 (option) | k × 1,024 per row | m · 4 | scratch, 1 u32 ticket a row | 5 + 2 | 1 |

### 3.6 The int4 wall

`tv_q4_hs` (sketch: `code/bigmodel/tv_q4_wide.cu`) stages the activation as raw f16 bits and widens at use:
`h2f(xs16[c + k])` is exactly the float `tv_q4_hh` stored, so every FMA sees the same operands and the warp butterfly
is unchanged. 51,200 B at 25,600 [C]. That is past the 49,152 B default, so `tv_q4_hs` needs its own opt-in; the
journal note that an f16 staging fits "sous le DÉFAUT" (`rot-partagee-14b-2026-08-17.txt:92`) is wrong by 2,048 B.
One CTA of 8 warps per SM, like the 14B's int4 down at 17,408 today. It takes `x_off` so prefill rows at an offset
stop going through the f32 cast and `tv_q4_h` (`fused_cuda.rs:941-951`), which would stay walled. The body is
duplicated rather than templated, in a unit of its own, so neither the served kernels' text nor the served unit's
sha256 moves.

The load path has to change in three places, or a 32B fails at load even with the new kernel:

1. **One size per entry point.** Today both entry points get `int4_shared_bytes(all records)`. With
   `int4_staging(d_in)` deciding per record (§3.7), `tv_q4_h` and `tv_q4_hh` get max d_in × 4 over the F32-staged
   records only, and `tv_q4_hs` gets max d_in × 2 over the F16-staged ones, each through `func_dynamic_shared`, which
   poses the opt-in when the size passes the default. For a 32B on the L40S: 32,768 B (o_proj at 8,192, default) and
   51,200 B (opt-in) [C].
2. **Per-record launch size.** `upload_int4` stores `shared = d_in * 4` and checks it against the ceiling
   (`fused_cuda.rs:2162-2173`); it must store and check the staged size, d_in × 2 for an F16 record.
3. **Activations that are not f16.** An F16-staged record can only take an f16 activation as is. The f32 activation
   path (a non-f16 dtype, the `+cast` arm of `LLVQ_TRIM_AB`) must either go to an f32-input tiled kernel with the same
   chain or be refused by name at that record. Narrowing f32 to f16 would change the arithmetic.

**How the int4 kernel gets checked.** The host pattern does not execute it: `host_tv_q4_h.cpp` runs `q4_deq` and a
hand-written mirror of the lane order, nothing else (`:15-19`). So the checks are on cards. At every load that has an
F16-staged record: `tv_q4_hs` against `tv_q4_hh` on the first int4 record, where both fit, and `tv_q4_hs` against
`tv_q4_ht` on the F16-staged record, beside `check_int4_half_parity`. Before any 32B exists, V2 runs the same checks on
synthetic records under a measurement mode: at 25,344, the widest d_in whose f32 staging fits 101,376 B, all three
kernels pairwise; at 25,600, `tv_q4_hs` against `tv_q4_ht`. The direct check at the 32B width, `tv_q4_hs` against
`tv_q4_hh` at 25,600, runs only where the f32 staging fits, the A100 (V5).

`tv_q4_ht` is the faster option: f32 tiles of TQ columns, TQ a multiple of 256, each lane walking its residue class of
words in increasing order, so its FMA chain is the same chain. TQ = 6,144 gives 24 KB and four CTAs per SM. It costs
two barriers a tile. It is worth measuring at 14B, where U1 suspects the one-CTA int4 down in the 2.2 ms gap between
the 14B's measured 17.5 ms step and its 15.3 ms fit [E].

f16 staging of the rotation itself, the first way out named in `format-noyau.md` §8, is not exact: it rounds every WHT
intermediate. It would be a T8 event, and nothing here needs it.

### 3.7 Selection and integration

`code/bigmodel/selection.rs` adds pure functions beside `shared::plan`: `rot_kernel(n, default, optin)` returns
`Shared(fit)` or `Device`; `int4_staging(d_in, ...)` returns F32, F16 or None; `int4_entry_bytes(d_ins, ...)` returns
the dynamic shared size of each int4 entry point. Rule: a width keeps today's kernel whenever today's kernel can run
it. Tests pin 2,560, 9,728, 12,288 (on the default bound) and 17,408 (opt-in) on L40S bounds as unchanged, 25,600 and
31,744 as moved, and 25,600 as unchanged on A100 and H200 bounds; the int4 test follows
`the_int4_staging_fits_the_card_at_every_sealed_size` (`fused.rs:2484`) with the three sealed shapes unchanged and a
32B shape added. Integration points: `RotBuffers` (`fused_cuda.rs:78`) gains the kernel choice, set in
`upload_rotation` (`:2047-2061`); the load path loads the wide unit only when used, allocates one scratch, and adds
the entry points to `spill_checked` (`:571`, `:756-783`); the three launch sites branch (`:1423, :1493, :1940`);
`gpu.rs` gains `launch_rot_dev` and its rows twin; the int4 path sizes per entry point at `:641-660` and per record at
`:2116-2200`; `llvq-llm/tests/served_unit.rs` gains a parse of the wide units. After its gates, R1 replaces
`rot_apply` inside `Shared` and R1w replaces R0 inside `Device`, behind the same function. Launch count per token does
not change.

### 3.8 What the branch on the audit's steps 2-5 needs

The audit's step 2 fuses norm + rotation and SiLU·mul + rotation, and the audit already files it under T8.

1. **Bit-exactness against today's candle ops is a hypothesis, not a design fact.** A load-time byte check decides
   it: the fused kernel's f16 rotation input against candle's ops on fixed inputs. If it fails, the T8 tolerance gate is
   the fallback. What exactness would take, read in candle-kernels 0.9.2 and candle-nn 0.9.2:
   - **The norm** is built by nvcc with `-O3` and the default fmad (`build.rs:14-26`), so `tmp += xi * xi`
     (`reduce.cu:196-197`) is a contracted FMA; the fused kernel must write `__fmaf_rn(xi, xi, tmp)`. The column
     stride per thread and the two-level `warp_reduce_sum` follow candle's launch, 1,024 threads when the row has at
     least 1,024 columns and 32 otherwise (`ops.rs:545`). `mean = tmp / ncols` is a correctly rounded division. The
     scale is `rsqrtf(mean + eps)`, an approximate instruction that the fused kernel must call too, never `1 / sqrtf`.
     The output is `(scale * x) * alpha`, two multiplies, rounded to f16.
   - **SiLU on `__half`** is `x / (1 + hexp(-x))` (`unary.cu:59-61`, `cuda_utils.cuh:177`). `hexp` and the `__half`
     division are cuda_fp16 intrinsics. The header-less NVRTC units would have to reproduce their instruction
     sequences for the CUDA version candle was built with (12.4 in the image). That is unproven.
2. R1's phase 1 is the natural seam, since it starts from 4 f16 values a thread.
3. Keep the rotation's `inv` as the host-narrowed f32, never `rsqrtf`: unlike the norm's scale, it has no candle
   counterpart to match.
4. Any split of the vector across CTAs must keep the stage order; nothing else is needed, since no reduction exists.
5. A fused kernel that stages n f32 hits the same 102,400 B at the 32B down site, so the R1w structure applies to it.
6. The int4 v side stream of the gestures lead fills the window the one-CTA rotation leaves, so R1 shrinks that gain:
   the two do not add, and one A/B measures both.
7. For V0b: the step-0 nsys profile can record, at no extra cost, the GPU time of each `rot_apply` at 2,560 and
   9,728 and the idle gap before each one. That is the number R1 needs most.

### 3.9 The big model from the throughput side

| quantity | dense Qwen3-32B | hybrid Qwen3.8-27B |
|---|---|---|
| parameters | 32,762,123,264 [C, config `9216db57`] | 26,895,998,464 text, + 0.425 G MTP, + 0.461 G vision [M, qwen35-recon] |
| our memory | about 10.72 GB, 2.62 b/param [E, wave-2 plan D] | about 9.2 GB [C, ROADMAP-27B §6] |
| rival at our rate | UD-IQ2_M 11.55 GB, 2.821 b/param [C from the Hub listing] | ISTA IQ2_S 9.26 GB, 2.751 [M, qwen35-recon] |
| bytes a token | 10.28 GB [E]; cap 54-72 tok/s at B = 560-740 GB/s [C] | about 8.8 GB incl. 0.30 GB of state read and written [E] |
| KV at 8k | 2.15 GB a token [C]; the expanded form would read 17.2 GB, so the grouped form, with its own T8, is a precondition | 0.54 GB, 16 of 64 layers [C] |
| prefill score peak at 8k | 12.88 GB, 64 heads [C, the ML audit's formula]; the prefill lead's chunking takes it to 1.1-3.2 GB | not computed |
| walls | rotation and int4 on 99 KiB cards | none: widest 17,408 [C] |
| launches | about 1,650 a token [E, U1 per-layer count × 64] | about 3,800 with candle-op DeltaNet, about 1,700 after the step kernel [E] |
| speed today-equivalent | 37-41 tok/s short, 27-29 at 8k on L40S [E, plan D] | 23-26, at most 31 by bytes [E, ROADMAP-27B] |
| drafter | EAGLE-3 heads for Qwen3-32B on the Hub (AngelSlim, RedHatAI) [M, listing] | native MTP head; ROADMAP-27B decision 11 drops it |

Which lever pays where [E]:
- **32B:** a weight-bound step, so verification flat in k (the brick lead) and a drafter pay most; the grouped
  attention and a real KV q8 pay at 8k; the fast rotation saves 1.4-3.3 ms of GPU time a token against R0, ×1.02-1.16
  depending on how much of it shortens the step, likely the upper half since a weight-bound step leaves the GPU busy.
- **27B:** launch-bound with candle-op DeltaNet. The DeltaNet step kernel (ROADMAP-27B-DEV P4) comes first, at
  −7 to −11 ms. Then the fast rotation, about 4 sites a layer, roughly 256 a token, worth 1.5-2 ms of GPU time. The MTP
  head is the cheapest drafter, but verification must checkpoint about 151 MB of state per drafted position or
  recompute it. KV levers weigh little there.
- **Shared ground.** Neither model is unique on the 32 GiB triplet card: a 4-bit 32B (Q4_K_M 19.76 GB) and a 4-bit
  27B (17.68 GB) both fit its 29.4 GB. Our format's distinct claims are speed (half the bytes) and the 16 GiB class.
  There the 27B fits comfortably (9.9 GB of 12.2), while our 32B fits only with a KV q8 (11.8 GB) [E].

The cheapest decisive validation is on the Mac and costs $0: K32 (prereg-bigmodel-1) for the 32B, gate K of
ROADMAP-27B for the 27B. A dense 32B quality check needs no wall lift anywhere, since the dense path never runs
`rot_apply`; it needs an 80 GB card (65.5 GB of f16 weights [C]). The kernel path would run on an A100 with the sm80
image or on an H200 with the standard image, both without the lift [C from the limits; never run].

## 4. Deciding unknowns

| unknown | current best | would confirm | would kill (proposal; the kill is the operator's) |
|---|---|---|---|
| how much rotation GPU time shortens the step | unknown; f1e0 saw 9.65 µs per removed launch, host side included [C]; the graph A/B puts the host's share of the eager step near 12 % [C] | V0c: null/real ≥ ×1.04, range above 1.00 | V0c median < ×1.02: R1 can gain under 2 %, proposed dropped from the served path |
| R1's isolated latency | 2,560: 4.3-4.8 µs, 9,728: 5.8-6.8 µs [E]; today 8.05 / 19.30 [M, handle A] | R1/K0 ≤ 0.80 at 2,560 and ≤ 0.70 at 9,728, round by round, same handle (prereg 2) | R1/K0 ≥ 0.95 at both: proposed, R1 not pursued on the served path |
| R1 end to end | ×1.02-1.08 at 4B, ×1.02-1.06 at 14B [E] | V3: the same tokens, 4B median ≥ ×1.03 with its range above 1.00 | range touching 1.00: not served; a timeline comes first |
| R0's cost at 25,600 | 25-90 µs [E]; V0d bounds R0/K0 from above at 9,728 before any code | ≤ 90 µs: R0 serves a 32B's first light at ≤ about 3 ms a token [E, about 34 down sites] | > 90 µs: R1w before any 32B is served |
| R1w at 25,600 | 6-15 µs [E] | ≤ 15 µs: R1w proposed as the 32B path | > 15 µs: write R2 |
| card = host, bytes | holds by construction; never compared as bytes on a card | every sha256 equal to the host fixture, every width, every kernel; every int4 pair equal | any mismatch: that kernel is dropped; no tolerance path |
| our standing at 14B, D_ours = ours − UD-IQ2_M | +2.9 [−0.7, +7.4] [E]; resolution ±1.8 pp | lower 95 % bound > 0 | upper bound < 0: the rival rung beats our sealed 14B; quality first |
| the 2-bit 32B class, Δ = P32 − (AWQ 14B + 1.0) | +1.7 [E]; resolution ±3.8-4.0 pp on the plan, so most likely unresolved | lower 95 % bound of Δ > 0 | upper bound < 0: no 32B encode on this evidence; unresolved: full split or stop (operator) |
| the 32B chain cost | $70-135 with dense censuses, $120-190 with kernel censuses at today's prefill [E] | a 4-block quote job puts the encode peak under 96 GB and the training under 141 GB | training past one H200: multi-GPU llvqtune work, re-priced |
| the 27B | everything; dev 26-41 days before its decision [E, ROADMAP-27B] | gate K, then G5 and G6 | per ROADMAP-27B §4 |

## 5. Validation ladder

Two tracks that do not wait on each other: the big-model track (P0, V4, V4b) on the Mac at $0, and the kernel track
(V0b to V5) on cards. Within each, cheapest and earliest first. Every rule below is a proposal to the operator;
benches inform, end-to-end gates decide.

| step | settles | method | kind and cost | prereg | decision rule | needs |
|---|---|---|---|---|---|---|
| V0 | what is true today | code, journals, dumps, primary sources (§8) | $0, done | no | n/a | nothing |
| P0 | the GGUF instrument on today's llama.cpp build, and the Mac time per question | 100 plan questions (every 23rd row) on `Qwen3-4B-BF16.gguf`, picks against the 2026-08-30 llama.cpp dump | $0, local run, 10-20 min of Mac GPU, 8.05 GB download [E] | yes, inside prereg-bigmodel-1 | four letters 100/100 and ≤ 3 picks different: the instrument holds, stage A goes ahead; otherwise stop and write the cause | go (local run) |
| V4 stage A | D_ours and the engine at 14B | E14 `Qwen3-14B-BF16.gguf` and R14 UD-IQ2_M on the plan, paired against the repository's census dumps | $0, local run, 3.5-6.5 h of Mac GPU, 34.97 GB [E] | yes, prereg-bigmodel-1 | its §7: V, C, D stop here; lo > 0 runs stage B | go; the ruling reopening rival GGUF arms; the `emit_dump` trailer committed |
| V4 stage B | Δ, the 2-bit 32B against AWQ 14B + 1 | R32 UD-IQ2_M on the plan | $0, local run, 3-5 h, 11.55 GB [E] | same | A, B or U; U: full split (≈ 6.2× the Mac time, $0) or stop | stage A's condition, and the operator's word |
| V4b | the big-model value, 27B | gate K of ROADMAP-27B | $0, local run, 3-6 h an arm [E, ROADMAP-27B] | yes (planned there) | ROADMAP-27B §4 | go; the Qwen3.5 letter ids for the GGUF scorer first |
| V0b | in-situ GPU time of each rotation and the idle gap before it | read the audit's step-0 nsys profile of the served 4B | $0 here if the operator's branch runs it; $0.3-0.6 alone [E, the audit] | the branch's | 96 rotations under 0.25 ms of GPU time a token: R1 proposed dropped without V0c; 0.25 ms or more: V0c; otherwise (no profile): V0c | the branch's job |
| V0c | the ceiling of any rotation speedup that keeps the launch | measurement mode `LLVQ_ROT_NULL_AB=1` at the bench door: `rot_apply` against `rot_null` (same launch, arguments and dynamic shared, empty body; sketch `code/bigmodel/rot_null.cu`, `rot_ab_modes.rs`), 2 + 5 interleaved rounds, 256 tokens, 4B (14B optional) | paid: half a day of code; 8-10 min, $0.24-0.30 at 4B; 12-15 min, $0.36-0.45 with the 14B [E; anchor `gqa-ab-4b` 10 min $0.29] | yes | null/real median ≥ ×1.04 with range above 1.00: write R1; median < ×1.02: R1 proposed dropped from the served path; otherwise the operator decides (14B only, or after the fusions) | cap and go |
| V0d | R0's slowdown against K0, before R0 exists (optional) | today's image: `rotbench` with `LLVQ_KERNEL_DIR` holding an unchanged `llvq_rot.cuh` and a `rotate.cu` whose `rot_apply` stages in place in `xout` (sketch `code/bigmodel/override-r0proxy/`); three pairs of processes, stock then proxy, in one job; the launch still requests n · 4 B of shared memory, so the proxy reads fairly only up to 12,288 and overstates R0 at 17,408 | paid, informative only: 5-7 min, $0.15-0.21, timeout 20 min = $0.60 [E] | stamp if a number is published | proxy/stock ≤ 1.5 at 9,728: R0 stays the first wall-lift step; > 2.0: R1w goes first for the 32B; otherwise no change, V2 decides | cap and go |
| V1 | arithmetic identity of R0, R1, R1w; the int4 wide unit parses | host tests: SMAX to 32,768; widths 5,120, 6,144, 8,192, 17,408, 25,600, 31,744; replays at 1, 7, 32, 256, 1,024 threads; byte equality with `rot_apply`'s text; fixture with input and output hashes; mutants of §6. Int4: a parse in the `served_unit.rs` pattern and an executed mirror in the `host_tv_q4_h.cpp` pattern, a reproduction and not the kernel | $0, local dev tests, minutes, light | no | all equal and every mutant killed: proceed to V2; any difference: fix before any card; otherwise no card | normal development |
| V1b | R1's idea on another GPU, informative only | R1's phases in MSL in a new file outside the wheel's copy list (under `llvq-metal/`), never in `llvq_rot.metal`; identity against the host text; local latency at 2,560 and 9,728 | $0, local run, minutes of GPU | stamp if a number is published | identity holds: V2 goes ahead whatever the Metal speed; Metal speed is never carried to sm_89 (rule 5); otherwise fix | go (local run) |
| V2 | card identity at every width, spill, isolated latency on the right objects; the int4 staging identity | extended `rotbench` (fixture compiled in, two handles, 4 kernels × 12 widths, 5 interleaved rounds) and `fusedrun LLVQ_INT4_WIDE_CHECK=1` on the 4B | paid: 10-13 min, $0.30-0.39, timeout 30 min = $0.90 [E; anchor `gqa-ab-4b` 10 min $0.29] | yes, prereg-bigmodel-2 | its §7 | cap and go |
| V3 | R1 end to end, bit for bit | bench door, expanded attention, `LLVQ_ROT_AB=1` (refused beside `LLVQ_CONFIG`), env of the e4 configs; sealed 4B then sealed 14B, one model a process; K0 and R1 in every round, 2 + 5 rounds, 256 tokens after a 5-token prompt; no 8,192 arm, since rotation work does not depend on context | paid: 14-25 min, $0.42-0.75, timeout 45 min = $1.35 [E; anchors `served-grouped-4b` 18 min $0.54 with a four-length sweep under two configs, `paper-served-14b-r2` 16 min $0.47 with an f16-head arm] | yes, written after V2 | tokens identical (at 4B the line hashes to `8784ffc4`), 4B median ≥ ×1.03 with range above 1.00 and 14B range above 1.00: propose serving R1; 4B median in [×1.00, ×1.03): optional, no change; range below 1.00 at either size: not served; otherwise no change | cap, go, then a served-path ruling |
| V5 | the lifted kernels equal the unlifted ones at the 32B width (optional) | a100-large, sm80 image: `rotbench` K0, R0, R1w at 25,600, and `fusedrun LLVQ_INT4_WIDE_CHECK=1`, where `tv_q4_hh` fits at 25,600 (102,400 ≤ 166,912 B) and is compared to `tv_q4_hs` directly | paid: 6-12 min, $0.25-0.50 [E; anchors `g2-a100-horloges` 4 min $0.13, `a4-a100-banc` 6 min $0.25, `a100-vllm` 10 min $0.41] | yes | every hash and every int4 pair equal: the identity claim holds on a second card at the 32B width; otherwise a card-specific defect, stop | go, `--kernels-cap 80 --any-flavor` |
| V6 | the 32B itself | the chain: encode, row scales, int4 swap, censuses, served speed, GSM8K | paid, $70-190 [E] | several | only after V4 case A, the 14B→32B gap gate written, V1-V3 green, and a grouped-form T8 plan | the operator's budget decision |

**V3 on the served grouped config** would need an operator ruling that admits `LLVQ_ROT_AB` beside `LLVQ_CONFIG`, as
`LLVQ_CONTEXTS` is admitted: it changes no output bit. The grouped form is reachable only through a config's
`attention` field or through `LLVQ_GQA_AB`, itself a measurement mode. Without that ruling, the expanded-form ratio is
reported as such; on the grouped form the step is shorter, so the ratio would be slightly larger.

## 6. Build plan after validation

1. **Docs and harness, 0.5 day.** Correct `format-noyau.md` §8, ETAT §4:120, CLAUDE.md:15-16 and ROADMAP §4: 25,600 =
   1,024 · 25; the wall holds on 99 KiB cards only; a bit-exact lift exists in design; the f16 staging of the rotation
   is not exact and would itself need the opt-in. Leave `llvq_rot.cuh` and `llvq_rot.metal` byte-identical: their
   comments get fixed at the next deliberate `tetratables` re-run, on the operator's ruling about the publication
   perimeter. Move the dependency-free `sha256_hex` out of the Linux-only `gpu.rs` (`lib.rs:34-35`) into a portable
   module, so host and card hash with the same code. Raise SMAX, add the widths, write
   `tests/fixtures/rot-hashes.txt` with input and output hashes.
2. **K32's commit, 0.5 day.** The `emit_dump` trailer and the start-of-run fingerprint check
   (`code/bigmodel/emit_dump_trailer.py`), `ops/k32_project.py` and its check against `mmlupair`, the driver and the
   pilot helper.
3. **V0c, 0.5 day**, unless V0b settles it. `rot_null` and `rot_null_rows` in a unit of their own, the measurement mode,
   its refusal beside `LLVQ_CONFIG` and beside the other modes.
4. **R0 and selection, 1-1.5 days.** Kernels, launchers, `rot_kernel`, load path, scratch, spill list. Tests: host
   byte equality; selection fixtures (L40S, A100, H200). Mutants: `Device` returned for `OptIn`; `<` for `<=` at
   12,288; row scratch at `r * row_stride`; `x_off` dropped.
5. **tv_q4_hs and tv_q4_ht, 1-1.5 days.** Kernels with `x_off`; `int4_staging` and `int4_entry_bytes`; the per-entry
   opt-in and the per-record size; the f32 activation path decided; the load-time checks and `LLVQ_INT4_WIDE_CHECK`.
   Mutants: a truncating f16 conversion; `x_off` ignored; F16 chosen before F32; the F16 record's size left at
   d_in × 4.
6. **R1 and R1w, 2-3 days**, only after V0c says write it. Kernels and replays. Mutants M1-M7 of
   `host_rotate_wide.cpp`: reversed subtraction in a pass, stage order swapped, KB below k, wrong sign word, batch
   offset, `inv` moved after the chain, rows output stride. A missing barrier is invisible on the host; V2's hash gate
   is its net.
7. **rotbench, 0.5 day.** Fixture compiled in, two handles, four kernels, twelve widths, input and output hash gates,
   interleaved rounds (`code/bigmodel/rotbench_wide.rs`). No recipe change, because the fixture is compiled in.
8. **V2**, then **`LLVQ_ROT_AB` and V3, 0.5-1 day**, at the bench door.
9. **Served switch**, on the operator's word, only after V3. The served unit's printed sha256 moves then, whichever
   unit carries R1.

Total [E]: the wall alone (steps 1, 4, 5, 7 and V2) is about 3-4 days and $0.30-0.39; R1 to a served switch adds
about 3-4.5 days and $0.7-1.2 (V0c at 4B and V3); K32 is 0.5 day of code and $0. The ML audit put the group staging at 1-3
days.

## 7. Rules map

- **T8.** None for R0, R1, R1w, `tv_q4_hs` or `tv_q4_ht`: each is bit-exact by construction and gated by byte
  identity on host and card. The f16 staging of the rotation (`format-noyau.md` §8) would be a T8 event and is
  rejected. The audit's step-2 fusions sit under T8 already, and their exactness against candle is a hypothesis
  (§3.8). A served 32B at 8k needs the grouped attention, whose T8 the operator granted at 4B only; that is a separate
  event per size.
- **A2.** Untouched. R2's ticket is not a persistent kernel and has no spin-wait.
- **Rule 6.** The 32B at 2.62 b/param counts the whole model with its int4 tables [E]. The GGUF arms use file bytes
  over 4,022,468,096, 14,768,307,200 and 32,762,123,264 parameters [C]. The 27B's MTP head is ROADMAP-27B decision 4.
- **Rules 4, 5, 7.** No × across cards or engines: K32 compares quality only, and any llama.cpp speed is quoted in
  GB/s. A served 32B speed claim carries its same-head ratio. V0c, V2, V3 form ratios round by round in one process;
  V0d is two processes and is labelled informative.
- **Closed leads (rule 3).** No ETAT §6 row is touched. The frozen micro-settings of 2026-08-05
  (`docs/archive/spec-lot-a-2026-08-05.md:129-132`) concern the matvec; R1 uses none of them in any case (no shuffle, no
  `__constant__` table). The IQ2 rival arms were dropped by the operator on 2026-09-19 after two card failures
  (`iq2m-2026-09-19.txt`); that is a journal ruling, not an ETAT §6 row, and K32 reopening them on the Mac needs a go
  (rule 1).
- **Rule 1 and rulings owed.** (1) Go for K32's pilot and stage A, a local run, and the ruling reopening rival GGUF
  arms. (2) Go for stage B on its condition, and after U the choice between the full split and a stop. (3) The change
  to `ops/vllm_score.py` that writes the trailer, or the fallback where `k32_project.py` is authoritative. (4) Cap and
  go for V0c and V2, then V3; V0d and V5 are optional. (5) V3's door: the bench door on the expanded form, or a ruling
  admitting `LLVQ_ROT_AB` beside `LLVQ_CONFIG`. (6) A served-path ruling if V3 is green: bit-exact, but the served
  object's kernels and its printed unit sha256 change. (7) The doc corrections of §6 step 1, with the kernel and shader
  files left byte-identical. (8) The big model: 32B, 27B or neither, after V4 and V4b. (9) The text of the 14B→32B gap
  gate. (10) For a 32B at 8k, the grouped form's T8.
- **METHODE.** Benches inform, the end-to-end V3 decides, kills are the operator's. Every rule above ends in an
  otherwise. Preregs are stamped before the first measurement.

## 8. Checked now at $0

1. 25,600 = 1,024 × 25, so m = 1,024 and k = 25 ≤ 32 [C, `rotation.rs:52-53`]. Qwen3-32B is 5,120 / 25,600 / 64 layers
   / 64 query and 8 KV heads / head_dim 128 / vocab 151,936 / untied (Hub `config.json`; local cache revision
   `9216db57`). That gives 32,762,123,264 parameters, KV 262,144 B a position (2.15 GB at 8k), and rotation widths 5,120
   (k 5), 8,192 (k 1) and 25,600 (k 25) [C].
2. Shared memory per block after opt-in: 99 KB on compute capability 8.6, 8.9 and 12.x, 163 KB on 8.0, 227 KB on 9.0
   (CUDA programming guide, compute-capabilities appendix). Matches the measured 101,376 B on the L40S
   (`fusedrun-14b-2026-08-17.txt`) and 166,912 B on the A100 (`f4-a100-2026-08-18.txt:95`). The h200 flavor (cap 90) runs
   the standard image (`ops/run.py:109`), so it could serve a 32B with today's kernels; never run.
3. The Metal rotation stages in device memory and is gated bit-identical to the CUDA text (`llvq_rot.metal:28-43`,
   `fused_metal.rs:880-920`). On Metal, threadgroup staging measured +7.5 % slower at 2,560, "limited by its eleven
   serial barriers and its one threadgroup" (commit `31a5acb`, 2026-09-21, message only, no journal).
4. The rotation host harness stops at 16,384 floats (`host_rotate.cpp:45, 104`), and the CASES stop at 12,288
   (`rotation_matches_rust.rs:42-51`). The served 14B width is checked only by the 2026-08-17 card run. The int4 host
   harness executes `q4_deq` and a mirror, never `tv_q4_h` (`host_tv_q4_h.cpp:15-19`).
5. `rot_apply` runs 1,024 threads for every n ≥ 513 (`fused_cuda.rs:2060`); its mix uses m of them
   (`llvq_rot.cuh:221`); its stage divides by a runtime divisor (`:163`); its chain is padded to 32 (`:225-229`).
6. **What is known about the rotation in situ, and what is not.** Sharing the rotation removed 72 launches at 2,560
   and moved the served object of 2026-09-09 from 94.2 to 100.8 tok/s, in two separate processes (`f1e0-2026-09-10.txt`
   §0 bis): 9.65 µs per removed launch [C], against 8.05 µs isolated. That is the cost of removing a whole launch: host
   dispatch, allocation and events, the launch itself and the GPU work. R1 removes only the GPU work and keeps every
   launch. In-situ rotation time has never been measured ("Normes, activations, attention et rotation ne sont mesurées
   ni ici ni là", `f3-events-2026-08-19.txt:159`). The CUDA-graph A/B returned +13.45 % (`a2-e3-graph-ab`, 2026-09-01,
   `jobs.csv`): the graph removed about 12 % of that eager step [C], which the audit reads as the host's share of
   submission. The isolated sum of the 4B's 96 sites is
   1.04 ms, 0.72 ms above the launch floor [C]; how much of it shortens the step is the transfer factor that V0b and
   V0c measure.
7. `tv_q4_hh` widens f16 while staging (`tv_q4_h.cu:145-146`). `int4_shared_bytes` takes max d_in × 4
   (`fused.rs:734-736`) and is posed on both entry points (`fused_cuda.rs:645-652`); `upload_int4` checks each record's
   d_in × 4 against the ceiling and keeps it as the launch size (`:2162-2173, 2198`). Offset rows and non-f16
   activations take the f32 cast and `tv_q4_h` (`:941-951`); a load-time parity pattern exists (`:2224-2261`). 51,200 B
   exceeds the 49,152 B default, so the journal note at `rot-partagee-14b-2026-08-17.txt:92` is wrong.
8. NVRTC compiles with default options (`gpu.rs:270-276`); `gpu.rs` is Linux-only (`lib.rs:34-35`).
9. The 2,280-question plan is a subset of the full-split dumps: all 2,280 keys are present with the same qhash in the
   sealed 14B and AWQ census dumps; the 4B and 14B plan dumps share keys and qhash and both end on fingerprint
   `65dcd53655e8bfa5`; the census dumps end on `a74a6d6213602979`; the 14B f16 plan dump and the f16 census agree on
   2,280 of 2,280 picks [C, `docs/data/mmlu-dumps/`, `docs/mesures/*-brut/`].
10. On the plan, micro with the house SE (`mmlu.rs:397-415`, finite population correction) [C]: 14B f16 78.97 (1.19),
   AWQ 78.21 (1.20), sealed 75.85 (1.23). Paired sealed − AWQ −2.36 (0.92), f16 − sealed +3.12 (0.81). The full split
   reads 78.88, 78.12 and 75.66 [M]. The first version of this note used p(1−p)/n without the correction (1.22, 1.23,
   1.27, 0.96).
11. Engine gate at 4B: llama.cpp f16 70.36 against our 70.32 (`m3-iq2-metal-2026-08-30.txt`), recomputed as paired
   +0.04 pp, SE 0.08, with 9 of 2,280 picks different [C].
12. Hub API, read 2026-11-02 [M]: `unsloth/Qwen3-14B-GGUF` at `a04a82c4`: BF16 29,543,424,160 B (`1d12adb9…`),
   UD-IQ2_M 5,425,956,224 B (2.939 b/param, `3c4aa2f8…`), UD-IQ3_XXS 3.258, Q4_K_M 4.876, Q8_0 8.504.
   `unsloth/Qwen3-32B-GGUF` at `931c8406`: UD-IQ2_M 11,553,366,688 B (2.821, `6d78bba9…`), UD-IQ2_XXS 2.265, Q4_K_M
   4.826, Q8_0 8.502. `unsloth/Qwen3-4B-GGUF` at `22c9fc8a`: BF16 8,051,285,536 B (`f59dbdf0…`). Full listings in
   `wave5-deep/downloads/bigmodel-hub-2026-11-02/`. EAGLE-3 heads for Qwen3-32B exist (AngelSlim, RedHatAI, others).
   The Qwen3.8-27B card names MTP and gives no speculative recipe or acceptance figure.
13. The Mac: M3 Max, 64 GiB, 378 GiB free, llama.cpp 10050 (`b15ca938a`) at `/opt/homebrew/bin`, no
   `convert_hf_to_gguf.py` in that install. The local HF cache holds `Qwen/Qwen3-4B@1cfa9a72` in full (the revision
   the 2026-08-30 conversion used), Qwen3-14B as a LICENSE only, Qwen3-32B as a config only. The 2026-08-30 F16 GGUF is
   no longer on disk. The ROADMAP-27B M0 blocker (12 GiB free on 2026-09-28) is gone.
14. The runtime image holds binaries, `fetch-qtip.sh` and `configs/` only (`ops/Dockerfile.cuda:131-153, 158, 175`);
   the build stage copies the whole uploaded tree (`:31`); `llvq-*/**` is in `UPLOAD_ALLOW` (`ops/run.py:751`). So a
   card binary can compile a fixture in with `include_str!` and cannot read one at run time. `rotbench` and `fusedrun`
   sit in both lists (`:87, 147` and `:53, 137`); BENCH_FLAVORS is `("l40sx1",)` (`ops/run.py:1037`).
15. Cost anchors in `jobs.csv` (total $248.84 over 237 rows [C]): `gqa-ab-4b` 10 min $0.29, `served-grouped-4b` 18 min
   $0.54 (its script estimated 29 min), `paper-served-14b-r2` 16 min $0.47, `a100-vllm` 10 min $0.41, `a4-a100-banc`
   6 min $0.25, `g2-a100-horloges` 4 min $0.13, the 14B encode $15.03 plus export $0.32, its row scales $9.78 on an
   h200, `census-14b-ref` $3.14, `gsm8k-14b-served` $4.24.
16. The Qwen3 report gives base-model MMLU 81.05 at 14B and 83.61 at 32B (arXiv 2505.09388, Tables 4-5): the size
   gain behind the K32 predictions, on another harness and other checkpoints.
17. `mmlupair` refuses a dump with no `# end fingerprint=… questions=…` trailer (`mmlupair.rs:185-196`); `bin/mmlu`
   writes FNV over the concatenated prompt ids (`mmlu.rs:759, 785, 854`); `emit_dump` writes no trailer
   (`vllm_score.py:375-412`) and neither does `mmlu-4b-gguf-f16.csv`. `--intersect` waives the run fingerprint for a
   census against a plan run and keeps the per-question qhash (`mmlupair.rs:326-331`). `gguf_mmlu.py` sends prompts as
   text (`:54-62`).
18. candle-kernels 0.9.2: `rmsnorm` sums `tmp += xi * xi` and scales by `rsqrtf(mean + eps)` (`reduce.cu:186-227`),
   built with `-O3` (`build.rs:14-26`); `silu_fwd` is `x / (1 + expg(-x))` (`unary.cu:59-61`) with `expg(__half)` =
   `hexp` (`cuda_utils.cuh:177`). candle-nn 0.9.2 launches the norm with 1,024 threads at 1,024 columns or more, 32
   below (`ops.rs:545`).
19. `llvq_rot.metal` is byte-identical in `llvq-llm/kernels/` and `llvq-tetra/llvq_tetra/data/` (sha256 `4d08e4cf…`),
   recorded in `tetra-tables.json` among 10 shipped sources; `llvq_rot.cuh` is not shipped but is part of the served
   unit (`llvq-llm/tests/served_unit.rs`), whose sha256 the runtime prints (`fused_cuda.rs:553`).
20. `LLVQ_KERNEL_DIR` replaces the sources a binary loads, all or nothing (`llvq-cuda/src/lib.rs:230-335`): two files
   for `rotbench` (`llvq_rot.cuh`, `rotate.cu`), the whole served unit for `fusedrun`. It is refused beside
   `LLVQ_CONFIG` (`served.rs:111, 160-167`). `launch_rot` always requests n · 4 B of dynamic shared memory
   (`gpu.rs:812-814`).
21. `generate` decodes exactly `max_new` tokens with no stop token (`model.rs:2119-2150`), so an arm with garbage tokens
   still times the same work; `LLVQ_TRIM_AB` is the house pattern for a one-process A/B with a token gate
   (`fusedrun.rs:178-247`).
22. The e4 configs give the int4 sets: 4B v, o and down@12-23; 8B v and down@10-26; 14B v, o and down@10-28
   (`configs/*-e4.json`). Two shared rotations a layer plus each Tetra down (and each Tetra o_proj at 8B) give 96, 127
   and 101 sites [C], matching U1.

## 9. Not established

- Any card number for R0, R1 or R1w, R1's end-to-end gain, R0's cost at 25,600, and the in-situ GPU time of a rotation.
- Whether `tv_q4_hs`, at one CTA per SM, streams a 25,600-wide int4 down well, and what `tv_q4_ht` buys at 14B.
- Whether a fused norm or SiLU·mul can match candle's bits.
- Any 32B number: quality, speed, encode peak, training memory. Whether D_ours at 14B transfers to 32B.
- K32's outcomes. The plan resolves D_ours to about ±1.8 pp and Δ to about ±3.9 pp at 95 %; the full split closes the
  sampling part under the house convention, at about six times the Mac time.
- The Mac time per GGUF arm: no journal records one; P0 anchors it.
- Everything about the 27B, including MTP acceptance against our sealed object.
- The grouped attention at group 8 (32B) under T8. Any sm_120 behavior beyond the shared limit.
- The −0 argument of §3.1 covers finite f16-derived inputs; NaN payloads already differ between host and card today.

## 10. Review of 2026-11-02 and what changed

| # | severity | finding | what changed |
|---|---|---|---|
| 1 | high | `rotbench` read the host fixture from a repository path that the runtime image does not have | the fixture is compiled in with `include_str!`; prereg 2 §2 says why, and why a runtime `COPY` is rejected; `rotbench_wide.rs` |
| 2 | high | `mmlupair` refuses every GGUF dump (no `# end` trailer), and census against plan needs `--intersect` | the commit carrying prereg 1 must make `emit_dump` write the trailer and check it before scoring (`emit_dump_trailer.py`); commands named with `--intersect`; gate 4 on the plan fingerprint; a fallback where `k32_project.py` is authoritative after a check against `mmlupair` |
| 3 | high | V3 asked for the grouped config with an A/B variable that the house refuses beside `LLVQ_CONFIG`; cost understated | V3 runs at the bench door on the expanded form, 4B and 14B, 5-token prompt only; re-priced at $0.42-0.75; the grouped option is named as a ruling |
| 4 | medium | "the rotation sits on the critical path" was an inference labelled *computed* | relabelled; f1e0 described as the removal of a whole launch; R1's gains rescaled with a transfer factor (×1.02-1.08 at 4B); V0b and V0c placed before any R1 code |
| 5 | medium | prereg 2 rule II pronounced a kill on a bench; rule III ignored rule I | II says "proposed to the operator"; II and III apply only to kernels that passed I, and III is a design proposal |
| 6 | medium | Q8_0 confounded the engine check with quantization | E14 is the BF16 file, size, sha256 and revision read from the Hub API; gate 3 adds at most 46 different picks |
| 7 | medium | A against B was decided on a point estimate with a ±3 pp error and a 1.7 pp margin | interval rule on Δ with SEs in quadrature (±3.8-4.0 pp under METHODE §3); case U names the full-split options; stage B runs only when D_ours is above zero |
| 8 | medium | K0 was timed on the 14B's opt-in object for the 4B's widths | two handles per shared-staged kernel; rule IV per handle against the journal of the same object (handle B against 2026-08-17 as well, a small extension of the reviewer's fix) |
| 9 | medium | the int4 integration missed per-entry-point sizes and `tv_q4_hs`'s own opt-in | §3.6 lists the three load-path changes; `selection.rs` gains `int4_entry_bytes` and a 32B-shape test; the wrong journal note is named |
| 10 | medium | the claimed host test at 25,600 rests on a pattern that never executes the kernel | reworded; card checks added: synthetic records in V2 (L40S) and `tv_q4_hs` against `tv_q4_hh` at 25,600 in V5 (A100) |
| 11 | medium | the "comment fix" would move a shipped, digest-checked shader and the served unit's printed hash | corrections go to docs; the files stay byte-identical; V1b's MSL lives in an unshipped file |
| 12 | medium | the bit-exact norm and SiLU fusions were stated as established | §3.8 states them as a hypothesis decided by a load-time byte check, T8 as fallback, with the exact candle details |

The six cheaper experiments the review proposed are in the ladder: P0 (pilot), V0b (profile), V0c (null bound), the
staged K32, V0d (in-place proxy, with its carveout caveat), and the int4 check at 25,600 in V5. No finding was judged
wrong.
