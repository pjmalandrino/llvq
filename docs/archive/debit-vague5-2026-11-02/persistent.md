# Persistent kernel: one launch per token on sm_89 (route P, then 4E, route D as fallback)

Written read-only on 2026-11-02 and revised the same day after an adversarial review; §10 maps each finding to its
fix. No repository file changed, no run, $0 spent. The only network calls were read-only listings of HF jobs and
their logs.

The object is the sealed Qwen3-4B (`qwen3-4b-sealed.bin`: 168 Tetra48 records, 84 int4 g128 records, a tied int4 g64
table), with the grouped config, on an L40S, batch 1, greedy. Each number carries its label at first use: *measured*
(named journal), *computed* (arithmetic on measured or code inputs), *estimated* (judgment, or any estimated input).

Three routes:
- route P: one cooperative segment kernel per layer, with attention left to candle;
- 4E: one persistent kernel looping over N tokens;
- route D: fusions in normal grids.

## 1. Verdict

**Route P can take the 4B from 133.3 tok/s to about 221-298 tok/s at short context** (133.3 *measured*,
`served-grouped-2026-11-02.txt`; the range *estimated*, §4). It keeps the published bits by design and checks them at
load. The model uses B_T = 490-670 GB/s, capped at the best read rate measured on this card (cuBLAS, 671 GB/s,
*measured*, `banc-252-2026-10-06`), and h = 1.5-3.5 µs. The first step, the MLP segment alone, is worth +10 to +25 %
(*estimated*).

4E reaches 339-394 tok/s short and 156-207 at 8k with the same cap (*computed* from the wave-2 judge's model with
estimated inputs; his 426 and 216 assumed 740 GB/s). It needs our attention kernel under T8 and the operator's A2
reading.

Five unknowns decide the lead, in this order:
- U1: the stream rate of Tetra rows in a 1-CTA-per-SM geometry, with 24 warps per SM against 48 today;
- U2: the exposed cost of one dependent all-CTA wait on sm_89, never published;
- U3: the bit identity of re-gridded rows against the served kernel, never observed on a card;
- U5: the cooperative launch overhead on sm_89;
- U4: the equality of the absorbed candle operations.

**Three findings at $0 since the first draft:**

1. **The one on-card identity test of a re-shaped Tetra kernel compared two different matrices.**
   - `tv_tetra48_seg` failed on 2026-09-20 at row 0: −0.3080684 fused against −0.57255346 separate
     (`llvq-cuda/tests/tetra48_segment_matches_unfused.rs:10-12`), a 46.2 % difference (*computed*).
   - The bench transcodes the fused Tetra stream from the ball file's indices and launches it with the ball file's
     scales. The unfused arm it is compared with decodes the Tetra file (§2, §3.12). No kernel could have passed.
   - The kernel is neither cleared nor convicted. The operator's branch should fix the stream's source before wiring
     it (`code/persistent/seg_source_fix.rs`).
2. **The served NVRTC unit is reproducible offline.** Its 11 parts, joined as `gpu.rs:177-181` joins them, give
   101,625 bytes and sha256 `6193094310bc…`, the digest the served door prints (*computed* from the working tree).
   Its PTX can be read at $0 with NVRTC 12.4 and no card (step L1a).
3. **Padding is a question of the persistent geometry only.** It was measured twice in normal grids and bought
   nothing: 24 x reads cost 0.118 ms of a Slot32 pass, and a stride-28 staging moved fused Planes14 by −0.14 %
   (`attribution-cuda-2026-08-05.txt:59-66`, `a3-occupation-banc-2026-09-01.txt:33`). At the rates route P needs, a
   stride-24 scalar read would cap the stream near 358 GB/s (*computed*). The PTX and the card's cubin say whether the
   loads are scalar (§3.8).

NVRTC 12.4 ships no `cuda_fp16.h`. So candle's SiLU is tabulated over its 65,536 inputs at load, which makes it exact
by exhaustion. The f16 add and multiply are exact through f32. Only RMSNorm needs an emulation checked at load.

**The first experiments cost $0**: host replays on the Mac, a PTX reading in a CPU container, and the bench fix of
finding 1.

**The first paid step is now the cheap one** (L1, `prereg-persistent-0.md`).
- What it measures: a one-launch Tetra pass at today's occupancy, the fused Tetra geometry, barrier and launch
  microbenchmarks, the boundary A/B in the model, and the card's cubins.
- Cost: about $0.45 (*estimated*), under a 90-minute timeout, so $2.70 at worst. It touches no frozen setting.

**The persistent arm set comes second** (L2, `prereg-persistent-1.md`, about $0.36). It runs only if L1 recommends it
and the operator lifts the frozen settings of decision 8.

Both steps inform the choice between routes P and D. The operator decides.

## 2. What exists today

| piece | where | state |
|---|---|---|
| served Tetra matvec | `llvq-llm/kernels/tv_tetra48_h.cu:46-89` | lane l adds blocks j = l, l+32, ... in increasing j across tiles (`:68-82`), `warp_sum` then `f2h(acc*rscale + tail_dot_h)` (`:84-88`); grid d_out/8 CTAs of 256 threads (`fused_cuda.rs:2391-2395`) |
| served NVRTC unit | `fused_cuda.rs:493-560`, `fused.rs:548-553`, `gpu.rs:177-181` | 11 parts: defines, `llvq_slot.cuh`, `matvec.cu`, `llvq_rot.cuh`, `rotate.cu`, `llvq_f1rank.cuh`, `llvq_f1rank_v3.cuh`, `llvq_tetra48.cuh`, `tv_tetra48_h.cu`, `emb_q4.cu`, `tv_q4_h.cu`, joined by `\n`; only `--gpu-architecture=compute_89` is passed (`gpu.rs:270-277`), and the driver JIT-compiles the PTX |
| R-row prefill kernel | `tv_tetra48_h.cu:157-229` | bit identity to R one-row calls tested on the host only (`llvq_tetra48.cuh:166-168`) |
| Tetra block dot | `llvq-cuda/kernels/llvq_tetra48.cuh:107-133` | 24 explicit `__fmaf_rn` reading `xb[TETRA48_ORDER[k]]`, returns `acc*gscale[g]*invnorm[m]`; tables `F1rTables` (`llvq_f1rank.cuh:87-92`), fetch (`llvq_f1rank_v3.cuh:181-221`) |
| int4 rows, head, gather | `tv_q4_h.cu:82-110`, `emb_q4.cu:53-104` | explicit `__fmaf_rn`, `__fmul_rn`, `__fadd_rn` everywhere; `row` read from `blockIdx` (`tv_q4_h.cu:91`) |
| rotation | `llvq_rot.cuh:114-233`, `rotate.cu:38-67` | three phases with `(tid, nthreads)`, no cross-thread reduction; one CTA per call, 96 calls a token |
| launch sites | `fused_cuda.rs:1592-1615` (Tetra), `:1770-1776` (int4) | one launch per projection on candle's stream |
| load-time card checks | `check_int4_half_parity`, `fused_cuda.rs:2224-2262`; spill contract `:615-620` | the pattern a persistent path must follow |
| one-stream trim | `fused_cuda.rs:2821-2844` | cudarc events off only on the served door (`source.is_some()`, `:2832`) and only while the context holds one stream; `load_with` passes `None` (`:2783`) |
| decode loop | `model.rs:2119-2152` | logits cast to f32 (`:2141`), argmax and blocking `to_scalar` (`:2142`), token re-uploaded (`:2147`) |
| KV store `Grow(256)` | `model.rs:394-414`, `:421-446`; `kvq.rs:470` | grows by 256-position chunks, `truncate` moves a length only |
| cooperative launch | cudarc 0.19.8 `launch.rs:243`, occupancy `core.rs:2278`, attributes `core.rs:2442`; `cuLink*` in `sys/mod.rs:12030-12094` | available; never called in the repository |
| bench arms | `llvq-cuda/src/arms.rs:47-81`, `planesbench.rs:2854`, A3 section `planesbench.rs:4119-4470` with `occ.rs` | the Tetra arm runs `tv_f1r_v3g` (f32 tail and output), not the served kernel; A3 is the template for one-launch arms checked bit for bit, and it invalidates a red arm without killing the job (`:4377-4432`) |
| A3's `persall` | `planes_occ.cu:376-442` | 144 fused Planes14 sites in one launch, 710 CTAs (5 a SM at 48 registers), 3.366 ms, 649 GB/s (*computed* from 2.185 GB, *measured*, a3) |
| measurement modes | `fusedrun.rs:677-679` (refused beside `LLVQ_CONFIG`), `run_trim_ab` `:187-243`, `GqaArm` `:263-305`, `LLVQ_GQA_AB` block `:767-806` | the template for in-situ boundary arms, events off included |
| fused Tetra pairs | `llvq-cuda/kernels/tetra48_seg.cu:107-163` | bench kernel only, behind `LLVQ_SEG_TETRA` with a fatal check (`planesbench.rs:1392-1406`); `seg_kernel_name(Tetra48)` is `None` (`fused.rs:800-805`) |
| the 2026-09-20 failure | `tests/tetra48_segment_matches_unfused.rs:10-12`; job `6aafac8152d0dbd7f1d73830` | row 0 at −0.3081 fused against −0.5726 separate. The fused stream comes from the first (ball) file's indices and scales (`planesbench.rs:3876-3893`, `:3954-4004`), the separate arm from the Tetra file (`:2910-2919`, `:1854-1903`). The host tests of `8493f4b` use one source for both. The job's logs and its bucket file are gone (§8) |

Still missing: any persistent kernel, any device barrier or counter, any cooperative launch, a device argmax, the
served Tetra kernel inside `planesbench`, an f16 twin of the segmented kernel, in-situ boundary arms, any PTX or SASS
reading of the served kernels, and a Tetra fusion check that compares a stream with its own matrix.

## 3. Design

### 3.1 Order

**Step A** is one cooperative kernel per layer, in this order:
1. ln2, then the MLP rotation;
2. the gate and up rows, with the SiLU-mul epilogue;
3. W3 and the down rotation for the 24 Tetra-down layers, or chunk counters for the 12 int4-down layers;
4. the down rows, with the residual epilogue.

**Step B** extends the segment from the attention output to the next layer's q, k and v: the o rows plus the
residual, W2, step A's body, W1, ln1, the q/k rotation, the q and k rows, and the v rows on the unrotated input.

Candle keeps q/k-norm, RoPE, the KV append, both cuBLAS products, scale and softmax: 10 launches a layer (*computed*,
U1 count).

**4E** absorbs those launches under T8 (Floor 3's kernel), plus the final norm, the head and the argmax, and loops
over N tokens.

### 3.2 Geometry and budgets

One CTA per SM, so the grid is the SM count (142 on the L40S), with `__launch_bounds__(T, 1)`. The kernel is
launched with `launch_cooperative`, so an over-sized grid is refused by the driver instead of hanging. The grid is
checked first with `occupancy_max_active_blocks_per_multiprocessor`, which must return at least 1.

| T | warps | register cap | note |
|---|---|---|---|
| 512 | 16 | 128 | fallback if 768 spills |
| 768 | 24 | 80 | default: a Tetra row takes 40 registers (`tv_tetra48_h.cu` comment); the mix needs about 45-50 (*estimated*) |
| 1,024 | 32 | 64 | probe: more latency hiding if the fat kernel fits in 64 |

Shared memory per CTA (*computed*) has three parts:
- the Tetra tables, 18,816 B (16,384 + 128 + 2,048 + 128 + 128, the sizes `banc-252-brut/sweep-64.txt` prints);
- one staging buffer sized to the widest input, in f32;
- 512 B for counters, norm partials and claim cursors.

The trio-28 layout (§3.8) needs more staging than the 24-float one:

| model | widest input | total at stride 24 | total at stride 28 | carveout | L1 left |
|---|---|---|---|---|---|
| 4B | down, 9,728 | 58,240 B | 64,720 B | 64 KiB | 64 KiB |
| 8B | down, 12,288 | 68,480 B | 76,672 B | 100 KiB | 28 KiB |
| 14B | down, 17,408 | 88,960 B | 100,560 B | 100 KiB | 28 KiB |

Everything fits the 101,376 B opt-in (*measured*, `format-noyau.md` §8), though the 14B at stride 28 has only 816 B
to spare.

int4 inputs are staged in f16 and widened on read, which is exact. Tetra inputs must stay f32: the rotated activation
is not representable in f16. Two CTAs per SM do not fit at 4B (2 × 58 KB), so the design accepts 24-32 warps per SM.
That trade is unknown U1.

### 3.3 Work decomposition

Rows per CTA at 142 CTAs (*computed*):

| phase | rows | per CTA |
|---|---|---|
| q+k+v | 6,144 | 43.3 |
| o | 2,560 | 18.0 |
| gate+up | 9,728 row pairs | 68.5 |
| down | 2,560 | 18.0 |
| head | 151,936 | 1,070 |

Each phase gets static contiguous row slices per CTA, from a host table balanced by rows (`pers.rs::slices`). Inside
a CTA, warps claim the next row through a shared-memory `atomicAdd` and a broadcast shuffle.

A row's bits do not depend on which warp or CTA computes it, because the lane-to-block map and `warp_sum` are fixed.
Every site ends with `__syncthreads()` before thread 0 resets the claim cursor, so no warp still in site s can draw an
index of site s + 1 (review finding 4).

The down phase keeps 18 of 24 warps busy, the same 18 warps per SM that the eager 320-CTA grid gives today. Splitting
a row across warps would change the reduction order, so it is excluded: A3's split-K arms were not bit-exact.

Row functions take the row as a parameter, with bodies copied from the served kernels
(`code/persistent/tetra48_pers.cu`). The Tetra row is `pers_row_acc<STRIDE, TRIO, PREFETCH>`: the served lane order,
`tetra48_dot` or its trio-order twin `tetra48_dot_trio4`, then `warp_sum`. Its epilogue is the served expression,
`f2h(acc * rscale[row] + tail_dot_h(...))`. The int4 and head rows reuse `q4_staged_dot`'s body. Gate row i and up
row i run in the same warp, one after the other, so the epilogue has both values in lane 0.

### 3.4 Synchronization

One monotonic arrival counter serves every all-CTA wait (code in `tetra48_pers.cu` §1):

```c
__syncthreads();                         // the CTA's stores precede the arrival
if (threadIdx.x == 0) {
    __threadfence(); atomicAdd(bar, 1u); // grid-sync pattern; red.release.gpu is a later optimization
    while ((int)(ld_acquire_gpu(bar) - target) < 0) __nanosleep(ns), ns = min(2*ns, 512);
    __threadfence();
}
__syncthreads();
```

- **Targets** are `base + k × grid`. The host keeps `base` per runtime and adds `n_bar × grid` after each launch, with
  `n_bar` computed by `pers.rs::n_bar`, so no memset is issued per launch. The comparison is wrap-safe while fewer
  than 2³¹ arrivals separate value and target.
- **Fine-grained dependencies** use one monotonic counter per site, with the target `(launch_idx + 1) × contributors`
  computed on the device. These are the int4-down chunks in route P, and q/k/v to attention to o in 4E. Only the
  sites someone waits on are signalled.
- **Every CTA arrives exactly once per barrier**, with rows or without (the Cohere miscount pitfall plan A cites).
- **Coherence.** Activations written during the launch are read with `__ldcg`. The read-only path `ld.global.nc` is
  not coherent with stores of the same launch, and the served kernels' `const __restrict__` pointers may compile to
  it. Weights and tables keep the default path.
- **Residual buffers alternate** per half-layer, so a fast CTA writing x' never overwrites the x that a slow CTA
  still reads. W1 and W2 separate every reuse; the schedule validator checks it.
- **Deadlock freedom:**
  - co-residency is the driver's guarantee under cooperative launch;
  - every CTA runs the same phase list;
  - every wait names producers in earlier phases only, and every producer signals (`pers.rs::validate`);
  - only thread 0 spins, and nothing in the launch waits on the host;
  - a bounded spin (about 67 ms at the 512-ns back-off cap) raises an abort flag in device memory and every CTA exits.
    The host then reads the flag, clears the counters and falls back to the eager path for the process.
- **sm_89 has no `setmaxnreg`, no PDL and no clusters.** One register allocation serves every phase, and the barrier
  is a global-memory round trip.

### 3.5 Prologues: norm and rotation in every CTA

Each CTA reads the residual (5,120 B at 4B, `__ldcg`) and computes candle's RMSNorm (§3.6). It then rotates the
result in place in its staging buffer. This redundant work avoids one more all-CTA wait per site.

- **Signs**, as `rot_load` applies them.
- **Walsh-Hadamard in registers.** One warp per 512-group holds 16 values per lane. The stages len = 1, 2, 4, 8 run
  inside the lane, then 16 to 256 as `shfl_xor` 1 to 16, in increasing order. The lower element keeps a + b and the
  upper a − b, with a the lower value: the operations of `llvq_rot.cuh:164-167`. The 8B's 4,096-groups add three
  shared-memory stages.
- **Mix without padding.** Each output's chain runs over the k real terms, then one `__fadd_rn(acc, 0.0f)`. The
  padded terms are `fma(+0, +0, acc)`, which equal one `acc + (+0)` under round to nearest, since `small` is padded
  with +0.0f (`fused.rs:1259`). Predicated full unrolling keeps `col` in registers.
- **Cost per CTA** (*estimated*): 1.0-2.0 µs at n = 2,560 and 1.5-2.5 µs at n = 9,728, against 8.05 and 19.30 µs on
  one SM today (*measured*, `rotation-cuda-2026-08-05.txt`).
- **Prefetch.** During a prologue the CTA prefetches the first rows of its next phase into L2, capped at window ×
  864/142 GB/s. Plan A's rule keeps prefetch out of streaming phases.

### 3.6 Candle operations, exact by construction where possible

NVRTC 12.4 has no `cuda_fp16.h`. NVIDIA bundles it only from CUDA 13.3, and then behind an extraction and an include
path. The runtime image ships no header (`llvq_slot.cuh:18-20`). So candle's `hexp` and `__hdiv` cannot be
recompiled.

| candle op (0.9.2) | how the segment reproduces it | evidence |
|---|---|---|
| `usilu_f16`, `x / (__half(1) + hexp(-x))` (`unary.cu:58-61`) | a 65,536-entry table built at load by running candle's own kernel on every f16 bit pattern; 128 KiB, read through L2 | equal by exhaustion |
| `bmul_f16` (`__hmul`) | `f2h(h2f(a) × h2f(b))`: the f32 product is exact (22 significant bits), one rounding | equal to `mul.f16` RN by construction |
| `badd_f16` residual (`__hadd`) | `f2h(h2f(a) + h2f(b))` | double rounding 24 to 11 bits is innocuous since 24 ≥ 2·11 + 2 |
| `rmsnorm_f16`, n ≥ 1,024 (`reduce.cu:187-228`, block 1,024 by `candle-nn ops.rs:545`) | 1,024 virtual threads' partials, the same xor butterflies and pairings, `__fdiv_rn(tmp, ncols)`, `rsqrtf(mean + eps)`, `(scale·x)·alpha` | two variants of `tmp += xi*xi` (contracted or not), selected at load; refused if neither matches |
| `fast_argmax` (`reduce.cu:476-523`, one block of 1,024) | 64-bit key, `ord16(v) << 32 \| (1023 - bitrev10(i & 1023)) << 22 \| (0x3FFFFF - (i >> 10))`, NaN refused | tie rule derived from the code; 0 mismatches on 300 adversarial cases in the wave-2 simulation |

Candle compiles these kernels with `nvcc -O3` and no fast-math (`candle-kernels build.rs:13-17`), so contraction is
on. Every emulated op is compared with candle's kernel on the card at load, on real activations and adversarial
vectors, in the style of `check_int4_half_parity`. Any op that differs keeps its candle launch, which costs about 72
launches a token for the norms (plan A's fallback).

### 3.7 Exactness, op by op

| op | argument | check |
|---|---|---|
| Tetra rows | same lane order, same `tetra48_dot` (or its trio twin: same FMAs on the same values), same `warp_sum`, same epilogue expression | row by row against `tv_tetra48_h` on every record at load |
| int4 and head rows | all arithmetic in explicit intrinsics: contraction cannot move | row by row against `tv_q4_hh`, `tv_emb_q4_h` |
| rotation | same stages and per-output chains, padding by the +0 identity | per-CTA output against `rot_apply` at every width |
| norm, SiLU, mul, add | §3.6 | load-time parity, candle fallback |
| argmax | the key reproduces candle's order | adversarial vectors at load |
| attention (4E only) | own split-KV kernel; cuBLAS's order cannot be reproduced | T8: paired perplexity, prefill and teacher-forced decode |

Three Tetra expressions are contractible under the default `--fmad=true`:
- `acc += t·iv`, where `t = acc_b·gscale` (`llvq_tetra48.cuh:132` and its caller);
- the tail loop (`matvec.cu:337`);
- the epilogue (`tv_tetra48_h.cu:87`).

A contraction can happen at two places:
- NVVM either fuses a pair into `fma.rn.f32` in the PTX, which ptxas keeps, or leaves a bare `mul.f32` and `add.f32`;
- the card JIT-compiles NVRTC's PTX with driver 580.178.04's ptxas (`a3-gpu.txt`), not with the 12.4 devel ptxas,
  and that ptxas may fuse a bare pair at load.

Two readings follow:
- step L1a reads the PTX at $0;
- step L1 dumps the card's cubins through `cuLinkComplete` (`code/persistent/culink_dump.rs`) for an offline SASS
  reading. A devel-stage `cuobjdump` is indicative only.

Two cases for the persistent kernel:
- where the PTX already holds `fma.rn.f32`, the persistent kernel spells `__fmaf_rn`, and identity holds by
  construction;
- where it leaves a bare pair, identity is the card's to decide (L2). A moved row is then fixed by spelling what the
  served SASS does.

This is a risk stated, not assumed away. The one on-card test of a re-shaped Tetra kernel, on 2026-09-20, could not
inform it (§2).

### 3.8 The staged activation layout, a question of the persistent geometry only

**In normal grids, two measurements settle it:**
- the 24 x reads at stride 24 cost 0.118 ms of a 5.82 ms Slot32 pass, "réfuté comme priorité"
  (`attribution-cuda-2026-08-05.txt:59-66, 120`);
- `tv_planes_pad`, a staging at stride 28, gained −0.14 % [−0.52 ; −0.08] on fused Planes14
  (`a3-occupation-banc-2026-09-01.txt:33`). The same journal says the conflicts "ne se paient pas en temps sur ce
  noyau" (`:141-142`).

Bank padding has been frozen since 2026-08-05 (`spec-lot-a-2026-08-05.md:131`). This lead does not propose it for the
eager kernels. The first draft said the layout "matters more than the plans assumed" and predicted 0.85 for an eager
stride-28 arm. Both are withdrawn (review finding 1).

**The persistent geometry differs because it aims at 550-670 GB/s instead of 278.** Lane l reads block j = l + 32n at
`xs + 24j`. Two cases:
- **32-bit loads.** Lanes equal mod 4 share a bank, an 8-way conflict: 192 wavefronts per warp and block, against 24
  without conflict (*computed*, `pers.rs::scalar_conflict_ways`). At 864 GB/s an SM would need 2.41 wavefronts a
  cycle for those reads alone, which caps the stream near 358 GB/s (*computed* at 2.52 GHz).
- **Six 128-bit loads.** Each quarter-warp conflicts 2-way: 48 wavefronts, a demand of 0.60 a cycle, and no cap. The
  compiler can emit them only if it proves 16-byte alignment. The load-store vectorizer groups loads by address, not
  by use order, so Tetra's permuted order does not prevent it.

The first draft assumed vectorization by use order and a 613 GB/s cap; that is withdrawn (review finding 10). Today's
278 GB/s uses at most 0.78 of the pipe even in the scalar case, which agrees with both measurements above.

**So three readings come in order:**
1. L1a reads the PTX: `ld.shared.f32` against `ld.shared.v4.f32`.
2. L1 dumps the card's cubin.
3. L2 carries two layouts that are conflict-free whatever the compiler does:
   - stride 25 with scalar loads: gcd(25, 32) = 1;
   - trio order at stride 28 with explicit `float4` loads: each quarter-warp's 16-byte words fall in 8 distinct bank
     groups (`pers.rs::quad_conflict_ways`). Scalar loads at stride 28 would still conflict 4-way, a cap at 716 GB/s
     (*computed*).

Both layouts multiply the same values in the same order, so they are bit-identical. Lifting padding for the
persistent kernel is the operator's decision 8.

### 3.9 4E specifics

- **Token loop.** It is bounded to N tokens a launch (64 proposed). Every CTA reads EOS from the same argmax word, so
  the exit is uniform. Tokens stream to a pinned host-mapped ring that the device never waits on.
- **`Grow` re-pointing.** Before each launch the host grows every K/V buffer to `cap ≥ pos + N`, under the 256-chunk
  rule (one copy at most per 256 tokens). It passes the 72 base pointers, `cap` and `pos`. After the launch it sets
  `len = pos + n_done`. Three `KvCache` methods: `reserve`, `raw_ptrs`, `set_len`. The extra memory is N × 147,456 B,
  9.4 MB at N = 64 (*computed*), against the 1.21 GB window that É7 refused.
- **Argmax.** Each CTA reduces its 1,070 head rows to one key with one 64-bit `atomicMax`, then W4. Slots alternate by
  token parity, and CTA 0 clears the other slot after W4.
- **Embedding.** Every CTA dequantizes the token row with `e4_deq`: the same f16 values as `emb_q4_gather`.
- **Waits per token** (*computed*; configs `qwen3-8b-tetra-e4.json`, `qwen3-14b-tetra-e4.json`):
  - 97 at 4B: 35 W1, the final W1, 36 W2, 24 W3 and W4;
  - 128 at 8B, where o is Tetra (36 W_o) and there are 19 Tetra downs;
  - 102 at 14B.
- **Teacher-forced mode.** It reads tokens from a buffer and emits target log-probabilities, with one extra reduction
  a token. T8 needs it.

### 3.10 Integration points

- **A separate NVRTC module** on candle's stream (`Cuda::on_stream`, `gpu.rs:359`), so the served unit keeps its bytes
  and its sha256. Never `new_on_fresh_stream`: a second stream breaks the event trim.
- **`llvq-llm/src/persist_cuda.rs`** (Linux, cuda): a `PersistentRuntime` with the module, the scratch buffers
  (counters, h, partials, the SiLU table), the schedule tables, the load-time checks, and `local_bytes == 0` on every
  new symbol.
- **Dispatch.** `Block::finish` and `forward_cached` dispatch to the segment for one-row decode steps only; prefill
  keeps its path.
- **Selection.** First a measurement mode of `fusedrun` for the A/B, then an optional `decode` field in served
  configs. `served.rs:67-111` keeps `deny_unknown_fields`; an absent field means the eager path, and Metal refuses the
  field by name.
- **Shared code.** The emulated candle ops and the argmax key live in one header shared with route D's fusions.

### 3.11 8B and 14B

The widths are 4,096/12,288 at 8B and 5,120/17,408 at 14B. The 8B adds a Tetra o site (rotation at n = 4,096, k = 1)
and its W_o. The 14B has 40 layers and 5 query heads a group.

Per-layer DRAM time at 864 GB/s is 37.2, 66.3 and 120.6 µs (*computed*, gap-2), so the same fixed cost weighs about
three times less at 14B. The trio-28 staging leaves the 14B 816 B of shared memory to spare (§3.2).

Route P at 14B reaches about 100-125 tok/s short (*estimated*, judge, at B up to 740 GB/s, not recomputed). Today the
14B runs 57.2 tok/s, *measured* untrimmed and expanded (ETAT §2).

### 3.12 Route D, for the operator's branch

The branch implementing steps 2-3 of the 2026-10-29 audit can keep every bit with these designs.

**Before wiring `tv_tetra48_seg` or an f16 twin, fix and rerun its check.**
- Its one on-card check, on 2026-09-20, failed at row 0 of layer 0's gate+up: −0.3080684 fused against −0.57255346
  separate (`tests/tetra48_segment_matches_unfused.rs:10-12`). That is a 46.2 % difference (*computed*): a different
  matrix, not a moved association.
- The bench builds the fused Tetra stream from the ball file's indices and launches it with the ball file's
  centroids, `gs_off`, `rscale` and tail (`planesbench.rs:3876-3893`, `:3954-4004`, `:3060-3071`). The unfused arm
  decodes the Tetra file (`:2910-2919`, `:1854-1903`). With two encodes of one projection, no kernel could pass.
- The fix (`code/persistent/seg_source_fix.rs`):
  - build the stream from the Tetra file's records;
  - add a host test with two different sources that fails on HEAD, and kill its mutant;
  - make the Tetra check an invalidation on A3's rule;
  - rerun the check on the card.
- Until then the kernel is unproven, neither cleared nor convicted.

**Then `tv_tetra48_seg_h`.**
- It takes `tv_tetra48_seg`'s grid and `gs_off`, with `tail_dot_h` and `f2h` (the served epilogue), and sets
  `seg_kernel_name(Tetra48) = Some("tv_tetra48_seg_h")`.
- In the bench on the served file, section A4 skips the whole q+k+v group because v is int4
  (`planesbench.rs:3949-3956`), so only gate+up is checked there. In the model q+k can fuse as a pair:
  `tetra48_seg.cu`'s header counts 216 to 144 Tetra launches.
- Time it on the bare 252 file, where the bench fuses q+k+v.

**Other singles:**
- Residual epilogues through `exact_hadd`, and SiLU-mul through the candle table and `exact_hmul`: equal by
  construction.
- q/k-norm is candle's 32-thread tree at n = 128. RoPE's `__hmul` and `__hsub` can be contracted by ptxas in candle's
  PTX, so carry the candidate forms and select at load against `rope_f16`, or keep candle's RoPE.
- A faster single-CTA rotation with the register WHT.
- No trio staging in the eager kernel: padding bought nothing in normal grids (§3.8). If L1a shows scalar loads, an
  eager stride-25 arm is a cheap bit-exact candidate, with a prior near zero gain: 0.99 [0.96, 1.01] of
  `tv_tetra48_h`'s time (*estimated*).

**Validation.** One `fusedrun` A/B with every single and the bundle, identical tokens required. A fusion is kept when
its paired gain exceeds 1 % (the judge's rule). The boundary A/B of step L1 can ride on the branch's next `fusedrun`
job: about 3 min, $0.09 (*estimated*), with its own prereg section.

**Pitfalls:**
- no second stream;
- no `ld.global.nc` on data written in the same launch;
- `rsqrtf`, never `1/sqrtf`;
- the `__half(1)` constant;
- candle's block-size switch at 1,024 columns;
- on the bench door, `load_with` leaves cudarc's events on: a mode that wants the served trim turns them off itself,
  as `LLVQ_GQA_AB` does.

## 4. Deciding unknowns

The model used for route P, 4B short context (*estimated*; `calc/persistent/route_p_model.py`):

t = 0.7487 GB/B_T + 0.4094 GB/B_Q + 0.2188 GB/B_H + 360 × a + 36 × c + 96 × h + 36 × drain + prologues + 0.09 ms

- a is a candle launch (2.2-2.6 µs), c a cooperative launch (3.0-4.5 µs);
- B_T = 670, 590 and 490 GB/s give 298, 260 and 221 tok/s, with the other inputs moved together (*computed* from
  estimated inputs);
- the first draft's top end, 324 tok/s at 820 GB/s, sat above every read rate measured on this card: cuBLAS 671 GB/s,
  our FP16 kernel 660, `persall` 649 (review finding 8).

A persistent bench pass over the bare file's f16 family (0.9467 GB, *computed*) maps to B_T = 0.9467 GB / t:

| pass time | B_T (*computed*) |
|---|---|
| 1.42 ms | 667 GB/s |
| 1.65 ms | 574 GB/s |
| 2.00 ms | 473 GB/s |

Physics floors it at 1.096 ms (864 GB/s).

| unknown | best value today | confirms the lead | proposal to the operator if it fails |
|---|---|---|---|
| U1 persistent Tetra stream rate B_T at 24-32 warps per SM | nothing on Tetra. `persall` 649 GB/s on Planes14 at 40 warps per SM (*computed*, a3). Best read rate on the card 671 GB/s (cuBLAS, *measured*). Stride-24 scalar cap 358 GB/s if the loads are scalar (*computed*, §3.8). Latency: an SM at 24 warps holds at most about 4.6 KB in flight (one 6-byte word per lane), against about 3.7 KB that 6.1 GB/s needs over about 600 ns (*estimated*), so the stream is likely latency-bound; the bit-preserving remedy, a register prefetch of block j + 32 (`audit-perf-noyau-cuda-2026-08-05.md:116-122`, never measured), is arm `psmem_t28_u2` | L1: `tpersall` ≤ 1.80 ms (B_pa ≥ 545 GB/s). L2: best decode arm ≤ 1.65 ms (B_T ≥ 574) | L1: `tpersall` above 2.60 ms: route D first. L2: rule C3 |
| U2 exposed cost h of one dependent all-CTA wait, skew included | not published for sm_89. RTX 5090: 2.2 µs atomic barrier, 3 µs `grid.sync` (*measured* by AlpinDale, one author). H100: 1.29 µs against 0.96 µs for a graph boundary (one author) | L1: 142-CTA barrier ≤ 2.5 µs. L2: h_exp = Δt/145 ≤ 2.5 µs | barrier ≥ 4 µs, no better than a launch: propose parking 4E, and route P only if U1 is strong |
| U3 bit identity of re-gridded rows on the card | never observed: the 2026-09-20 attempt compared two different matrices (§2). Expected identical where the PTX shows `fma.rn.f32` at the three contractible sites; undetermined where it leaves bare pairs to the driver's ptxas (§3.7) | 0 differing rows: f32 family in L1, f16 family against the served unit in L2 | rows moved within 1e-5 (f32 value against f64): spell the served SASS, a code fix. Outside 1e-5: a defect, the arm is out |
| U4 candle-op equality | SiLU, mul and add exact by construction (§3.6); RMSNorm emulated | every load-time parity check passes | a failing op keeps its candle launch (about 72 launches a token for norms) |
| U5 cooperative launch overhead | V100/P100: a cooperative null kernel 15 % slower end to end (*computed* from Zhang et al. Table I, R1 fact-check) | L1: ≤ normal launch + 1.5 µs | above + 3 µs: one segment a layer, never two |
| U6 registers without spill at 768 threads | static, unknown | `local_bytes == 0` at 768 | 512 threads (128 registers) |
| U7 value of a removed boundary in situ | 5.5 µs (D1, Planes14) and 8.4 µs (`persall`), both *computed* and non-transferable (digest J11) | split@wave − empty ≥ 3 µs: the dataflow design pays | ≤ 1.5 µs: price boundaries flat (gap-2 barrier row, ×1.10-1.23 for 4E) |

## 5. Validation ladder

Cheapest first. Each paid step needs a go, a cap and a stamped prereg (rules 1 and 2). A kill is always the operator's.

| step | settles | method | cost, load | prereg | decision rule | go needed |
|---|---|---|---|---|---|---|
| L0 host replays | rotation register-WHT order and +0 padding at every width; RMSNorm tree emulation against a host replay of candle's tree; argmax key against a host `fast_argmax` replay; schedule validator, counts (97, 145, 242), wrap-safe targets, staging bijections, bank maps, the byte ledger | Rust host tests and `tests/*.cpp` in the `(tid, nthreads)` style; mutants on stage order, padding, key bits, arrival counts, signals | $0, light local `cargo test` | no | all equal and every mutant killed: go on. Any mismatch: fix the design, the lead waits | development go |
| L1a PTX reading | load widths of the staged reads at NVVM level; `fma.rn.f32` against bare `mul`/`add` at the three sites, in `tv_tetra48_h`, `tv_tetra48_rows_h`, `tv_f1r_v3g`, `tv_tetra48_seg` | `code/persistent/ptx_probe.py`: NVRTC 12.4.127 in a linux/amd64 container on the Mac. The served unit is assembled and refused unless it hashes to `6193094310bc…` | $0, a few minutes of local CPU, no card, no image build | no | scalar loads: the stride arms become L2's main question, and route P's model carries the 358 GB/s cap until L2. Six `v4` loads: the stride arms stay as controls with a null prior. Otherwise re-derive the cap from the counted wavefronts. `fma.rn` at a site: spell `__fmaf_rn`. A bare pair: the card decides | go for a light local run |
| L1b seg-source fix | whether the Tetra fusion can be checked at all; route D's precondition | build the fused Tetra stream from the Tetra file (`seg_source_fix.rs`); host test with two different sources; mutant = HEAD's source | $0, local `cargo test` | no | fails on HEAD and passes after: the diagnosis holds. Passes on HEAD: the diagnosis is wrong, and the 2026-09-20 failure stays open | development go, in the operator's branch or here |
| L1 first paid step | U1 anchor (`tpersall`), U3 in the f32 family, route D's Tetra fusion delta, U2 without skew, U5, U7, the card's cubins | one job, l40sx1, two processes (`prereg-persistent-0.md`, `pers-combined.sh`) | about $0.45; timeout 90 min (five table arms, METHODE §6), $2.70 worst | yes | prereg-persistent-0 §7, rules A-G | go, cap ($3 asked with L2), stamp; no frozen setting touched |
| L1-SASS | the card's SASS: load widths, contraction sites, served and bench kernels | `nvdisasm` or `cuobjdump -sass` from a CUDA 13.0 devel image on L1's cubins, off the card | $0, local | no | rule F of prereg-persistent-0 | go for a light local run |
| L2 persistent arms | U1 in route P's geometry, U2 with skew (h_exp, h_marg), U3 in the f16 family against the served unit, U6, layout, prefetch, occupancy | one job, l40sx1, one process (`prereg-persistent-1.md`, `pers-arms.sh`) | about $0.36; timeout 90 min, $2.70 worst | yes | prereg-persistent-1 §7, rules A-E | go, cap, stamp; decision 8's lift (512+ threads, tables in shared, padding, decode before SASS); L1 not in its lower branch |
| L3 route P step A A/B | the end-to-end gain, same text | `fusedrun` mode: grouped served against grouped plus the persistent MLP segment, one process, 2 + 5 rounds | about $0.35-0.6 | yes | ≥ 8 % with identical text and gate 4's line `8784ffc4`: adopt. 3-8 %: a curve point for the operator. Below 3 %, or a token change the fallback cannot remove: kill proposal | go, decision 8 |
| L4 step B A/B | the gain over step A | same, three arms | about $0.4-0.6 | yes | ≥ 8 % over step A: adopt. 3-8 %: curve point. Otherwise keep step A | go |
| L5 8B and 14B | transfer | same A/B per size, each with its reference line | about $0.4-0.8 per size | yes | ≥ 8 %: serve at that size. Otherwise curve point | go |
| L6 4E | the whole-token kernel | after Floor 3's attention under T8; teacher-forced perplexity plus tok/s | $3-6 (*estimated*, judge) | yes | ≥ 8 % over Floor 3 with T8 passed: adopt. Below 3 %: kill proposal. Otherwise curve point | A2 reading, large-model choice |

An sm_120 run of L1 or L2 under `--any-flavor` (about $1.7-2.75, *estimated*) is optional. No ratio crosses cards.

The R-row identity series of the first draft is no longer on this ladder. It serves batch above 1, prefill and
speculative verification, which ROADMAP §4 holds (decision 9). If the operator lifts that hold, the series is one
`planesbench` section of about $0.1 marginal on the L2 job, with its own rule.

## 6. Build plan after validation

All durations are *estimated*.

1. **L0 code, 4-6 days.**
   - `llvq-cuda/src/pers.rs` with its tests (`code/persistent/pers.rs` sketch);
   - host replays of the WHT, the norm tree and the argmax key;
   - the `CUDARC_CUDA_VERSION=12040` clippy type-check of the Linux half;
   - mutants: reversed WHT stage, dropped `__fadd_rn`, padding with −0, key without bit reversal, an arrival skipped
     by a CTA with no rows, a counter wait on a later site, a waited site that does not signal.
2. **L1a and L1b, 1-2 days.** The probe, the seg-source fix and its host test. The fix belongs to the operator's
   branch if it gets there first.
3. **L1 code, 1.5-2 weeks.**
   - `tv_tetra48_persall` and its `occ.rs` site words (`code/persistent/tetra48_persall.cu`);
   - the non-fatal Tetra check;
   - the microbenchmark module and its cooperative launches;
   - `LLVQ_DUMP_CUBIN` (`culink_dump.rs`);
   - `LLVQ_BOUNDARY_AB` in `fusedrun` (`fusedrun_boundary_ab.rs`);
   - CLAUDE.md lines for the new variables.
   No new binary, so the two-lists trap of `ops/Dockerfile.cuda` and `UPLOAD_ALLOW` does not apply.
4. **L2 code, 2-3 weeks**, only if L1 recommends it.
   - the persistent module (`code/persistent/tetra48_pers.cu`);
   - the served unit loaded into the bench as `tetra48h`;
   - the identity machinery (poison, debug f32 buffer, A3's moved-association rule);
   - the prologue checks.
   Registering the arms in `arms.rs` instead would put one-launch arms inside the per-matrix dispatch loop
   (`planesbench.rs:3682-3720`) and grow `N_ARMS`, `HAS_KERNEL` and `DISPLAY_NAMES`.
5. **Step A, 2-3 weeks.** `persist_cuda.rs`, the MLP segment, the load-time checks (rows, rotation, norm, SiLU table,
   add, mul), the per-op fallback and the abort path. Gate L3.
6. **Step B, 1.5-2 weeks.** The segment extended to o and q/k/v. Gate L4.
7. **Ports, 1 week a size.** Widths, the 8B's W_o, the shared budgets (the 14B at stride 28 is tight). Gate L5.
8. **4E, 10-16 weeks** (judge), after Floor 3, the A2 reading and the large-model choice.

The durations keep the factor of 2 of optimism the verifiers found in every plan. `LLVQ_KERNEL_DIR` removes image
builds for kernel-body edits on the bench door only.

## 7. Rules map

- **T8.** Route P targets bit identity and spends none. A failing parity check keeps candle's op, unless the operator
  licenses that op under T8 on a stamped prereg. 4E spends one T8, for attention, inherited from Floor 3.
- **A2 (ETAT §6, rule 3).** Route P captures nothing, holds no fixed window, and launches normally on `Grow`. É7
  refused the pair (graph, 8k window), not persistence. Whether "A2 stays out of the core" covers a device token loop
  (4E), and possibly per-token segment launches (route P), is the operator's reading.
- **The A3 grammar** (`preregistration-a2-a3-geometrie-2026-08-31.md` §5, `:117-122`).
  - A bench arm must beat the FUSED geometry: beating the 252-launch geometry would only re-measure the fusion.
  - Both preregs here therefore time `tv_tetra48_seg` in the same rounds: L1 rule C and L2's Q ratio.
  - A bench arm needs ≥ 10 % over the in-process reference to merit a port. The end-to-end gate is ≥ 8 % to adopt
    and < 3 % to close.
  - The A3 journal reads `persall` as a bound, not portable, because rotation, attention and norms sat between sites.
    Route P absorbs the rotation and the norms, the new fact that journal named.
- **Archive settings of 2026-08-05.** None of them is an ETAT §6 row, so a go under rule 1 lifts them and rule 3 does
  not apply. The report's decision 8 keeps them frozen by default ("la route P n'est pas ouverte"). L2 needs four of
  them lifted:
  - CTAs of 512 threads and more: "512 threads/bloc rejeté", `audit-perf-noyau-cuda-2026-08-05.md:152`;
  - tables in shared memory: `:151`, and "table en shared (régression)", `spec-lot-a-2026-08-05.md:131`;
  - bank padding: "bancs/padding (réfuté)", `spec-lot-a-2026-08-05.md:131`, with its two measurements (§3.8);
  - decode frozen until the SASS: `spec-lot-a-2026-08-05.md:132`, a condition L1's cubin dump meets.
  L1 touches none of them.
- **Decision 9** (ROADMAP §4 on batch and prefill) is not needed by L1 or L2 since the R-row series left them.
- **Mixed Tetra plus int4 bench arm** (ETAT §6, closed 2026-10-06). The bench uses the bare 252 file only; int4 rows
  are timed end to end. A synthetic pure-int4 persistent arm would need the operator's reading of that row.
- **Rule 6.** No weight changes, so b/param is unchanged at 2.7138 for the 4B sealed object (*computed*,
  `embed-q4-swap`). Runtime scratch stays under 0.02 GB (*estimated*); 4E adds 9.4 MB of KV at N = 64.
- **Rules 4, 5, 7.** One card, one engine; ratios round by round; the raw and same-head ratios are equal here, since
  no arm changes the head.
- **METHODE §6.** Both `planesbench` jobs time five table arms, so both carry `--timeout 90m`.
- **Decision `deaa449`** keeps candle as the auditable engine. Absorbed candle ops stay checked against candle's
  kernels at every load, and the operator rules on the replacement (report decision 12).

## 8. Checked now at $0

1. cudarc 0.19.8 exposes `LaunchArgs::launch_cooperative` (`src/driver/safe/launch.rs:243`), a wrapper of
   `cuLaunchCooperativeKernel` loaded dynamically (`result.rs:1359-1381`, `sys/mod.rs:11730-11744`). It also exposes
   `occupancy_max_active_blocks_per_multiprocessor` (`core.rs:2278`), `set_attribute` (`core.rs:2442`) and
   `cuLinkCreate_v2`, `cuLinkAddData_v2`, `cuLinkComplete`, `cuLinkDestroy` (`sys/mod.rs:12030-12094`). No file in
   `llvq-llm/src` or `llvq-cuda/src` calls any of them.
2. Candle's `fast_argmax` takes each thread's first element unconditionally, then a strict `>`, and its tree keeps the
   left operand on ties (`candle-kernels-0.9.2/src/reduce.cu:476-523`). The launch is one block of
   `min(1024, n).next_power_of_two()` threads and uploads dims and strides on every call
   (`candle-core-0.9.2/src/cuda_backend/mod.rs:324-333`). Tied threads resolve by the lowest differing index bit: the
   smallest bit-reversed index wins.
3. Candle's RMSNorm uses 1,024 threads at 1,024 columns or more, 32 below (`candle-nn-0.9.2/src/ops.rs:545`): strided
   `tmp += xi*xi`, xor butterfly 16 to 1, 32 warp partials, a second butterfly, `tmp / ncols`, `rsqrtf(mean + eps)`,
   `T(scale*x*alpha)` (`reduce.cu:187-228`).
4. `candle-kernels` builds its PTX with `nvcc -O3 --expt-relaxed-constexpr -std=c++17`, no fast-math
   (`build.rs:13-17`).
5. Candle's f16 SiLU is `x / (T(1) + expg(-x))` with `hexp` (`unary.cu:58-61`, `cuda_utils.cuh:177`). Its f16 RoPE is
   `src[i1]*c - src[i2]*s` and `src[i1]*s + src[i2]*c` in `__half` operators (`reduce.cu:291-312`).
6. NVRTC bundles `cuda_fp16.h` only from CUDA 13.3, behind `nvrtcInstallBundledHeaders` and `-I` (NVIDIA NVRTC
   guide). The image's runtime stage has no CUDA header (`llvq_slot.cuh:18-20`). `ops/fetch-qtip.sh:102` says NVRTC
   carries `cuda_fp16.h`, which `qtip_prelude.cuh:8-13` contradicts by declaring `half2` itself.
7. The served Tetra row adds blocks in increasing j per lane whatever the tile (`tv_tetra48_h.cu:68-82`). `warp_sum` is
   a fixed xor butterfly (`matvec.cu:85-90`). Three contractible expressions remain (`llvq_tetra48.cuh:132`,
   `matvec.cu:337`, `tv_tetra48_h.cu:87`).
8. int4 and head rows use explicit intrinsics only (`tv_q4_h.cu:74-78, 104-106`; `emb_q4.cu:45-49, 98-100`).
9. `small` is padded with +0.0f (`fused.rs:1259`), and `rot_mix` runs all `LLVQ_ROT_KMAX = 32` terms
   (`llvq_rot.cuh:49, 220-231`), so the single `__fadd_rn(acc, 0.0f)` replacement of the padded chain holds.
10. The F₂ trellis is checked on all 4,096 inputs at `Tetra::new` (`llvq-search/src/tetra/trellis.rs:211-219`). Its
    only card reading, `tv_f1r_v2`, ran 5.625 ms against 3.875 for v3 (*measured*, `f1-rang-variantes-2026-09-05.txt`).
    The F₂ arms left this design.
11. Bank maps (*computed*, `pers.rs` tests): 32-bit loads conflict 8-way at stride 24, not at all at 25, 4-way at 28.
    128-bit quarter-warp loads conflict 2-way at 24 and not at all at 28.
12. Served 4B bytes (*computed* from row formulas): Tetra 748.7 MB, int4 409.4 MB, head 218.8 MB, total 1.377 GB. The
    bare file's bench pass: words 0.9083 GB, f16 family 0.9467 GB, f32 family 0.9806 GB (the bench prints 0.98).
13. The shared budgets of §3.2 and the rows per CTA of §3.3 (*computed*).
14. The served door turns cudarc's events off only on one stream (`fused_cuda.rs:2821-2844`). `load_with` passes
    `source: None` (`:2783`), so the bench door keeps them on unless a mode turns them off, as `LLVQ_GQA_AB` and
    `LLVQ_GQA_PPL` do (`fusedrun.rs:747-756, 795-806`). A module can join candle's stream through `Cuda::on_stream`
    (`gpu.rs:359`).
15. Measurement modes are refused beside `LLVQ_CONFIG` by a list (`fusedrun.rs:677-679`). `run_trim_ab` voids a run on
    any divergent token (`:187-243`). The grouped form is set by `GqaArm::Grouped.apply` (`:301-305`), whose names
    print `grow(256)+grouped`. The served form's line `attention grouped on grow(256)` comes from the getters
    (`fusedrun.rs:865`, `served.rs:250`).
16. `planesbench` requires `fp16` in every phase and dispatches in registration order (`arms.rs:13-41`). Its Tetra arm
    is `tv_f1r_v3g` (`planesbench.rs:2854`). The A3 section checks one-launch arms bit for bit and invalidates a red
    arm without killing the job (`planesbench.rs:4330-4470`). Section A4 runs only with slot32 and planes14 selected
    (`:3861`).
17. `planesbench` and `fusedrun` are already in both lists of `ops/Dockerfile.cuda` (`:52-54`, `:87`, `:131-153`).
    `BENCH_FLAVORS = ("l40sx1",)` (`ops/run.py:1037`).
18. Builds: 148 s for planes14, fp16, nullk and tetra48; 1,495 s for ten arms (*measured*,
    `banc-252-2026-10-06-brut`). The A4 fusion transcode took 77 s (*measured*, `banc-tetra-fuse-2026-09-20-brut`).
19. `docs/data/jobs.csv`: 237 rows, $248.84 over the 232 priced ones. The median `l40sx1` rate is $1.80/h over 107
    jobs of 5 min or more (*computed*).
20. Served configs: the 8B serves v_proj and down_proj@10-26 int4, so its o is Tetra. The 14B serves v, o and
    down@10-28 int4.
21. The three bench inputs sit in the bucket: `ball-ref-2026-09-20/qwen3-4b-llvq.bin`,
    `tetra-4b-2026-09-06/qwen3-4b-tetra.bin`, `sealed-4b-2026-09-23/qwen3-4b-sealed.bin` (listing taken today by
    another wave-5 agent, `wave5-deep/bucket-listing.txt`).
22. **The 2026-09-20 Tetra fusion failure.**
    - The recorded line is in `llvq-cuda/tests/tetra48_segment_matches_unfused.rs:10-12`, and its cause is in the code
      (§2, §3.12).
    - The job is `6aafac8152d0dbd7f1d73830`: ERROR, 1,602 s on l40sx1 (*measured*, `hf jobs ps -a`).
    - `HfApi.fetch_job_logs` returns 0 lines for it and for its sibling `6aafc951…`. A 2026-10-06 job returns 98 lines
      and a 2026-09-26 job none, so retention is near 30 days.
    - The bucket's `banc-tetra-fuse-2026-09-20/banc.txt` was overwritten at 12:20 UTC by `6aafc951…` (12,116 B, the
      size of `docs/mesures/banc-t64-2026-09-20-brut/banc.txt`).
    - The verifier's "$0 log recovery" is therefore not possible. The recorded line was enough.
    - Raw listings: `wave5-deep/downloads/persistent-jobs/`.
23. **Three jobs missing from `docs/data/jobs.csv`.** `6aaf9e00…` (COMPLETED), `6aafac81…` and `6aafc951…` (ERROR), all
    of 2026-09-20, ran 1,599, 1,602 and 1,556 s on l40sx1 (*measured*). At $1.80/h that is about $2.38 (*computed*),
    for the operator to register.
24. **The served NVRTC unit reproduces offline.** Defines `#define TILE_BLOCKS 64u`, `#define TETRA48_ROWS 4u` and
    `#define TETRA48_TILE 128u`, then the ten files of §2, joined by `\n`: 101,625 bytes, sha256
    `6193094310bc18acdb0a0c554807ff45f6df09c1f6ab49e0577032874f107812`. That is the digest `served-grouped-2026-11-02`
    prints (*computed*, `python3 -I` on the working tree, whose kernels match HEAD).
25. **Padding measurements.** `attribution-cuda-2026-08-05.txt:44, 59-66, 120` and
    `a3-occupation-banc-2026-09-01.txt:33, 112-117, 141-142` (§3.8).
26. **4E under the cap.** The judge's model (`wave2/judge.md:261-262`) gives 394 tok/s short at B = 671 GB/s and 207
    at 8k with η = 0.60, against his 426 and 216 at 740 GB/s (*computed* from estimated inputs).
27. **METHODE §6** asks `--timeout 90m` of any `planesbench` with five arms or more (`docs/METHODE.md:223-226`).

## 9. Not established

- Any persistent stream rate, wait cost or cooperative launch cost on sm_89.
- The load widths of Tetra's staged reads (L1a, L1-SASS), hence whether the 358 GB/s scalar cap exists.
- Whether `tv_tetra48_seg` is bit-exact once its stream comes from the right file.
- Bit identity of any re-gridded, persistent or R-row Tetra kernel on a card; RMSNorm emulation parity on a card.
- Whether 768 threads fit the fat kernel without spills; the instruction-cache pressure of a many-phase kernel.
- The int4 rows' persistent rate (46 % of bytes with the head at 4E, *computed* by wave 1).
- KV residency in L2 under the weight stream, for short-context attention in 4E.
- The sustained SM clock under a kernel that loads DRAM and ALU together. Every cap of §3.8 assumes 2.52 GHz; a lower
  clock tightens them.
- Every 8B, 14B, sm_120 and 5090 figure of this note.

## 10. Response to the review of 2026-11-02

| # | finding (severity) | what changed |
|---|---|---|
| 1 | the "$0 finding" on layout ignored two padding measurements and the frozen setting (high) | both cited (§3.8, §8.25); the eager arm and its 0.85 dropped; the layout kept as a persistent-geometry question; padding added to the settings to lift (§7, prereg-1 header); "changes the plan" removed from the verdict |
| 2 | the failed on-card test of `tv_tetra48_seg` was not cited, and U3 was "identical by construction" (high) | cited, and its cause found in the code: the fused stream came from the other file (§2, §3.12, `seg_source_fix.rs`). The logs are gone (§8.22), but the recorded values suffice. U3 lowered to a stated risk (§3.7, §4). The branch is told to fix and rerun the check first |
| 3 | the f64 fallback at 1e-5 voids binary16 arms (high) | the f32 value before `f2h` goes to a debug buffer and is judged at 1e-5; binary16 outputs at 1e-3; an arm with a moved association is still timed (prereg-1 gate 5) |
| 4 | work-claiming race in the waitless arms (high) | `__syncthreads()` at the end of every site in every mode; `y_all` poisoned before every identity launch, and a row left at 0xFFFF invalidates the arm (`tetra48_pers.cu` §6, prereg-1 gate 6) |
| 5 | process 2's gate lines were never printed on the bench door (high) | the mode turns events off and prints the line itself, sets and prints the grouped form from the getters, and the job compares its digest with the served door's in the same job (`fusedrun_boundary_ab.rs`, prereg-0 gate 4) |
| 6 | h_exp divided by 97 overstated h by 1.49 (medium) | the divisor is the 145 grid-wide syncs, computed in `pers.rs`; every arm pays the same per-site barrier; only waited sites signal; the trailing barriers exist; `pchain_w2` gives the skew-free slope; rule C re-derived |
| 7 | the reference was the 252-launch geometry, against the A3 grammar (medium) | `tv_tetra48_seg` timed in the same rounds in L1 and L2; L1 rule C and L2's Q ratio state the A3 condition |
| 8 | prediction intervals below the physics, B_T up to 820 GB/s (medium) | intervals floored (1.051 ms `pword`, 1.096 ms decode arms); the verdict caps B_T at 671 GB/s: 221-298 tok/s, and 4E 339-394 |
| 9 | 45-minute timeout against METHODE §6 (medium) | 90 minutes, $2.70 worst, cap $3 for the two jobs |
| 10 | a devel `cuobjdump` cannot show the card's SASS; use-order vectorization assumed (medium) | PTX first at $0 (L1a, with the served unit's digest reproduced); card cubins through `cuLinkComplete` (L1); stride-25 scalar and explicit-`float4` trio arms; the 613 GB/s cap withdrawn |
| 11 | frozen settings not listed as a precondition (medium) | listed in prereg-1's header and in the L2 row; L1 touches none |
| 12 | the rows series serves leads on hold (medium) | dropped from this job; decision 9 named (§5) |
| 13 | the three files disagreed on `pchain_w`'s base; macros; missing modes (medium) | one template body, twelve symbols, a table in `pers.rs` that names each; `pchain_w` = `psmem_t28` + waits by a test; F₂ arms dropped; 512/1024 and the trailing barriers implemented |
| 14 | no arm tested the latency mechanism of U1 (medium) | `psmem_t28_u2`, a register prefetch of block j + 32, bit-identical (rule B') |
| - | the six cheaper experiments proposed | the log recovery is not possible (§8.22); the PTX reading is L1a; the padding journals are read (§3.8); the persall port, the microbenchmarks, the boundary A/B and the non-fatal seg check form L1, the first paid step |
