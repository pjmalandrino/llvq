# Lead "brick": decode once, multiply many (Floor 4K), deep dive

Written 2026-11-02 and revised the same day after an adversarial review (§10). Read-only on the repository, $0 spent,
no build, no kernel run. Two arithmetic scripts ran once under `python3 -I` (`calc/brick/synthetic_actq.py` and
`geometry_budget.py`, outputs beside them). Labels: *measured* (named journal or primary source), *computed*
(arithmetic on measured or code inputs, shown or reproducible), *estimated* (judgment, or arithmetic with one estimated
input). Card: L40S (sm_89, 142 SMs, 864 GB/s, 2.52 GHz boost) unless named. Object: the sealed 4B, grouped config.
"Row" is a weight row; "token" is an activation row (k tokens per pass). Paths under `wave5-deep/` are in this
scratchpad; every other path is in the repository.

## 1. Verdict

The brick keeps one property by construction: every output row is bit-identical across k = 1 (dp4a), k ≥ 2 (int8
`mma.m16n8k32`), prefill rows and a host C++ replay. Block dots are exact integers and the slice partials are summed as
exact int64, so no order, split or launch geometry can move a bit. Its MMA and decode work stay flat in k up to 8
tokens. Its launch geometry does not. The tile-major stream gives each warp 16 rows, so keeping today's warp count
needs a K split, and the L40S measured the two nearest moves as losses at k = 1 on `Planes14`: split-K −1.87 % to
−9.87 %, several rows a warp −10.45 % to −24.62 % (*measured*, `a3-occupation-banc-2026-09-01.txt:34-40`).

The kernel estimates are therefore wide, all *estimated* (§3.5):
- k = 1 against the served kernel: 0.88-1.30 of its time for dp4a, 0.90-1.40 for IMMA;
- r(8) = 1.03-1.35 and r(16) = 1.08-1.70 for IMMA;
- the whole verify step, with int4 and the head on SIMT rows: r(8) = 1.10-1.30 at today's 7.44 ms step and 1.30-1.7 at a
  lean 2.6 ms step. The lean upper end crosses the 1.45 kill proposal.

The price in quality is one T8 event: the 168 Tetra inputs become int16 per 384-coordinate slice with a power-of-two
exponent. On synthetic activations this adds 3 % of the noise power that f16 rounding already puts on the activation
(Gaussian, median; 5 % at p99) and 13-15 % for Student t with ν = 4 (median; 30-45 % at p99). A massive channel rotated
by the repository's construction stays at 1-4 % (*computed*, 2,000 vectors a case,
`wave5-deep/calc/brick/synthetic_actq.out`). The paired perplexity should move by less than 0.03 % against the
±0.12 % band (*estimated*).

Four unknowns decide the lead, and three are about speed. The first is the geometry's cost at k = 1 and its growth in
k, through staging, int64 atomics and occupancy, none of them measured. The second is the kernel r(8) with the real
decoder. The third is the whole-step r(8) once int4, the head and attention have exact k-row paths. The fourth is the
T8 cost on real activations, and no activation dump exists.

The cheapest order runs three $0 steps on the Mac first, plus an optional fourth (§5): `artstat` on the three sealed
files (seconds), a Metal f32 perplexity paired with the card's f16 NLLs (minutes, `prereg-brick-1a.md`), then the
int16 injection control (15-60 min, `prereg-brick-1.md`). The first paid step is β0, a geometry bench with a trivial
decoder: about $0.5-1.1 a try after about a week of code (`prereg-brick-0.md`). β0 can stop the lead before the 5-8
weeks of reference, emulator and real decoder that bench β still needs after it. Bench β thus comes 6-9.5 weeks of
code from the start, 9-19 weeks at the verifiers' 1.5-2× factor (*estimated*, `prereg-brick-2.md`).

## 2. What exists today

| where | what it does | status for the brick |
|---|---|---|
| `llvq-cuda/kernels/llvq_f1rank_v3.cuh:89-96` | value tables as biased bytes `val + 128` | XOR `0x80808080` gives signed int8 quads with the decoder untouched |
| `llvq_f1rank_v3.cuh:146-152, 157-161` | quad = 2 PRMT + byte mask + LOP3 select, bytewise | works unchanged |
| `llvq_f1rank_v3.cuh:181-209` | fields and 6 gathers (3 rows, prefix, branch, then suffix, dependent) | kept; the F₂ trellis is a separate arm (§3.4) |
| `llvq_tetra48.cuh:26-30` | bounds: max\|y\| = 10, n2 ≤ 416, m ≤ 26 | gives \|P\| < 2^21.643 at int16 (§3.1) |
| `llvq_tetra48.cuh:86-95` | shell norm by `__dp4a` on `q ^ 0x80808080` | dp4a already runs in the served kernel |
| `llvq_tetra48.cuh:107-133` | served block dot, `(acc·g)·inv` | replaced by the integer semantics (T8) |
| `llvq_tetra48.cuh:172-205` | `tetra48_dot_rows`: decode once, R f32 FMA chains | bit-identical to R single calls on the host only |
| `llvq-llm/kernels/tv_tetra48_h.cu:46-89` | served kernel: warp per row, 8 rows per CTA, f32 tile staging, tables read through L1 from global pointers, lane-0 `acc * rscale[row] + tv` | that last line is contractible under `--fmad=true` |
| `tv_tetra48_h.cu:157-229` | rows kernel, `TETRA48_ROWS × TILE × 96 B ≤ 49,152`; 40 registers for the one-row kernel (`:180-182`) | the SIMT incumbent for k rows; 6 CTAs and 48 warps per SM |
| `llvq-cuda/kernels/nullk.cu` | the served geometry's floor: same grid, staging, `warp_sum`, epilogue, write, no weight read | `docs/format-noyau.md` §6 forbids subtracting it from another grid: the brick needs its own null arm |
| `llvq-search/src/tetra/mod.rs:105-110`, `trellis.rs:126-128, 210-220` | `LINEAR_COLUMNS`, `linear_input`, all 4,096 inputs asserted | TA[s8] ^ TB[b1 \| b2<<1 \| b3<<5], 512 B |
| `llvq-artifact/src/tetra48.rs:61-63, 113-175` | row-strided little-endian transcode at load | template for the tile-major transcode |
| `llvq-artifact/src/codebook.rs:216-262` | `tetra_fingerprint` digests the map | a runtime layout cannot move it |
| `llvq-artifact/src/format.rs:484, 1510-1530` | `rotation_seed` stored per record; `decode_matrix` un-rotates with it | the Mac injection takes Q from the record |
| `llvq-llm/src/bin/artstat.rs:92-106` | prints c1/c0 min, p50, max, records with c0 = 0 or a negative level | settles the S grid on the served files; no journal has run it |
| `llvq-cuda/kernels/rotate.cu:38-63, 103+` | `rot_apply` (one CTA), `rot_apply_rows` (one CTA per token) | gains a quantizing epilogue |
| `llvq-cuda/kernels/llvq_rot.cuh` | phases take `(tid, nthreads)`, `rot_mix` is a 32-term padded `__fmaf_rn` chain | card equals host (*measured*, `a1-lot-a-2026-08-06.txt:58`) |
| `llvq-cuda/src/bin/rotbench.rs` | times `rot_apply` on the card, reports spills | home of the quantizing-epilogue arms; already in both Dockerfile lists |
| `llvq-llm/kernels/tv_q4_h.cu:74-110` | int4 g128, `__fmul_rn`/`__fadd_rn` dequant, warp per row | no rows kernel; `int4_rows` loops (`model.rs:1213`) |
| `llvq-llm/src/fused_cuda.rs:114-124, 1600-1624, 1840-1870, 2216-2262` | `DeviceStream::Tetra48`, decode and rows dispatch, `check_int4_half_parity` | integration points and the load-time check pattern |
| `llvq-cuda/tests/host_shim.h:31, 70, 92, 103-110` | `__fmaf_rn` = `std::fma`, `__byte_perm`, signed `__dp4a`; barriers and shuffles vacuous | no ldmatrix, no mma, no mixed-sign dp4a |
| `llvq-cuda/src/arms.rs:1-40`, `planesbench.rs:106, 109, 116, 2854` | arm registry; the NVRTC unit never changes with the selection; `fp16` in every phase; synthetic `GSCALE` [0.625, 1.375]; Tetra arm = `tv_f1r_v3g` | a new kernel changes the unit of every arm, so ratios hold within one process |
| `llvq-cuda/src/gpu.rs:14-16, 262-282, 445-465` | NVRTC with `--fmad=true`, compute_89 PTX that the driver JITs, `Cuda::report` reading `num_regs` and local bytes | every brick float op is an `_rn` intrinsic; register and spill gates read this, since no SASS tool exists |

What works: decode once in SIMT, bit-identical on the host; a decoder whose bytes are int8 operands after one XOR; a
verified F₂ trellis; a partition-invariant rotation measured equal to the host on the card. What is missing: the integer
semantics with its Rust reference and host replay; any ldmatrix, mma or mixed-sign dp4a code; a host emulation of warp
collectives; the tile-major layout; a launch geometry for 16-row tiles that keeps today's warp count and has been timed;
the quantizing rotation epilogue; k-row int4 and head kernels; a row-invariant attention; a card check of any rows
kernel under `--fmad=true`; and the served files' centroid span.

## 3. Design

### 3.1 The canonical semantics

Every path reproduces this reference: dp4a k = 1, IMMA k ≥ 2, prefill rows, the persistent kernel, the host replay.

**Q, quantize one token.** Slices are 16 blocks (384 coordinates) from block 0; the f16 tail columns are excluded and
stay f32.
- `amax_σ = max |x_i|` over the slice, exact and order-free. `E_σ` is its binary exponent, read from the f32 bits.
- `e_σ = E_σ − 14`, then the cap `e_σ = max(e_σ, e_hi − 3)` with `e_hi = max_σ e_σ`. Put `e_lo = e_hi − 3` and the
  shift `s_σ = e_σ − e_lo ∈ [0, 3]`. A zero slice takes `e_lo`; a non-finite amax makes the token's outputs NaN.
- `x̃_i = clamp(rint_half_even(x_i · 2^−e_σ), −32767, 32767)`. Multiplying by a power of two is exact; only the max
  element can reach the clamp, when its mantissa is ≥ 1.99997.
- Planes: `hi = x̃ >> 8` (arithmetic, s8), `lo = x̃ & 0xFF` (u8), so `x̃ = 256·hi + lo` exactly.

**D, block dot.** `P_b = Σ_j y_j·x̃_j` over the 24 decoded values. |P| ≤ 99.92 × 32,767 < 2^21.643 (*computed*:
Σ|y| ≤ √(24·416) by Cauchy-Schwarz on the bounds of `llvq_tetra48.cuh:26-30`). Exact in int32.

**A, accumulate.** `S_f32[g][m] = round_f32(gscale[g] / √(16m))`, evaluated in f64, for g ∈ {0, 1} and m ∈ 1..26, and
`S[g][0] = 0` (the origin): 52 nonzero scales. `E_S` is the exponent of the smallest nonzero |S| minus 23, and
`S_int = S_f32 · 2^−E_S` is an exact signed integer. |S_int| < 2^28 holds when the nonzero |S| span at most 4 binades,
that is when max(|c0|, |c1|) / min(|c0|, |c1|) ≤ 3.14, since √26 alone spans 2.35 binades (*computed*). The bench
file's levels are synthetic (`planesbench.rs:106`, 1.375/0.625 = 2.20, 3.49 binades). The served files' levels, which
artscale and row-scale training can move, have never been read; step S0 reads them. A matrix past the bound is refused
at load by name. Its fallback is one int64 accumulator per gain level, each on its own E_S, joined in the epilogue by
two conversions and one fixed add; it is designed only if S0 finds such a matrix. A zero level (c0 = 0, counted by
`artstat`) leaves 26 nonzero scales. A negative level is carried by the sign of S_int. Then
`I = Σ_b (P_b << s_σ(b)) · S_int[g_b][m_b]` in int64: |I| < 2^61.3 at 405 blocks (the 4B down width) and < 2^62.7
at 1,067 blocks (the 32B down width) (*computed*). Integer addition is associative, so any order and any split give
the same I.

**E, epilogue, one thread per output, fixed order.** `u = cvt.rn.f32.s64(I)`; `v = __fmul_rn(u, 2^(e_lo + E_S))`, exact
while v stays normal. If `e_lo + E_S < −126`, v is formed by two power-of-two products, each correctly rounded and the
same on host and card (NVRTC runs `--ftz=false`). Then `tail = __fmaf_rn` chain from +0 over `i = 0..tail_w−1` of
`(h2f(tail[r][i]), x_tail[i])`, and `y = f2h(__fmaf_rn(v, rscale[r], tail))`.

**Why every path returns the same bits.** The quantizer depends on one token only. P and I are exact integers. The
epilogue is a fixed sequence of correctly rounded operations written as `_rn` intrinsics, which NVRTC never contracts,
and `std::fma` with `-ffp-contract=off` on the host. The tail chain has one order. `rint`, `max`, the clamp and the
power-of-two products are exact. Prefill keeps the identity only if it runs this semantics; a cuBLAS route (1c) is its
own T8, and exact speculation does not need it (§3.9).

Two variants are kept as fallbacks, never as defaults. Plan C's float chain per slice gives the same accuracy class but
forces the work split onto slice boundaries and a fixed-order combine. An int24 three-plane activation keeps
|P| < 2^29.7, exact in int32 through the C-operand chain, and is the fallback if int16 fails T8. Its products reach
2^57.7, so it needs an int64 accumulator per slice and a 128-bit row accumulator, one carry add per slice; the order
stays free. It is a new three-plane kernel, weeks of code, not a rerun.

### 3.2 Layouts

**Weight stream, tile-major, transcoded at load (0 bits).** Per matrix, per 16-row tile T, per round ρ of two blocks,
lane L holds the word of row `16T + (L mod 16)`, block `2ρ + L/16`. Two planes: `lo32[T][ρ][L]` (bits 0-31) and
`hi16[T][ρ][L]` (bits 32-47). A warp reads 128 B + 64 B per round, coalesced, with no `f1r_load` phase arithmetic.
An odd block count (405) pads the last round with word 0, which decodes to the origin and contributes exactly 0. All
row counts of the 4B, 8B and 14B are multiples of 16. The map is a bijection on 48-bit words, so the sealed file,
`tetra_fingerprint` and `codebook_fingerprint` do not move. The VRAM stream shrinks by 3.3 MB at 4B (*computed*: 0 B
of padding a row instead of 4 B at 106 blocks, 6 B instead of 2 B at 405 blocks).

**A record, one per (row, block), in a 1 KB warp tile.** `[q0 q2 q4 S | q1 q3 q5 0]`, 32 bytes, q signed quads. MMA
column k maps to trio position `8s + j` for `k = 4s + j` and to `8s + 4 + j` for `k = 16 + 4s + j`, s < 3; trio position
p maps to natural coordinate `TETRA48_ORDER[p]`. Bytes 12-15 carry `S_int` because the activation is zero at k = 12..15;
that zero is an invariant with its own mutant. Half h of record l sits at 16-byte chunk `2l + (h ^ ((l >> 2) & 1))`.

**B record, the activation.** `[slice][block][plane][token][32 B]`, each 32 B as four 8-byte thread slots: slot t holds
k = 4t..4t+3 then 16+4t..19+4t; slot 3 is zero. Read as eight u32, the same record gives the dp4a operands in quad order
`[q0' q1' q2' q3' q4' q5' 0 0]`, so one buffer feeds both back ends. The quantizer writes it, natural-to-trio order
included.

**Bank conflicts, enumerated** (*computed*, quarter-warp phases for 128-bit accesses, half-warp for 64-bit). Without
the swizzle the decode stores and `ldmatrix.x4` take 16 wavefronts per round against 8 ideal (2-way). With it both take
8. The B slots take 2 wavefronts per warp `LDS.64`, the minimum for 256 B; a token-major layout would take 8.

### 3.3 Fragments

Re-derived from CUTLASS `SM80_16x8x32_S32S8S8S32_TN` (*measured*, primary source, §8 item 6). With g = lane/4 and
t = lane%4: A holds rows g and g+8 at k = 4t..4t+3 (a0, a1) and 16+4t..19+4t (a2, a3); B holds token g at the same k; C
holds (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1). The S8U8 variant used for the lo plane inherits the same layouts.
Lane t < 3 therefore receives section t of rows g and g+8, and lane t = 3 receives S for rows g and g+8 in a0 and a1.
`ldmatrix.x4` takes row addresses from lanes 0-7 (rows 0-7, half 0), 8-15 (rows 8-15, half 0), 16-23 (rows 0-7,
half 1) and 24-31 (rows 8-15, half 1). The β0 job checks this map on the card before any emulator relies on it (§5).

### 3.4 Kernels and launch geometry

- **Decoder.** The served v3 decoder, unchanged bit for bit. Its biased bytes become signed int8 quads by one XOR with
  `0x80808080` at use. This leaves the 2026-08-05 setting "décodage gelé jusqu'au SASS"
  (`docs/archive/spec-lot-a-2026-08-05.md:130-132`) untouched. The F₂ trellis touches the decoder: it is a separate
  identity-preserving arm and waits for decision 8's reading.
- **Tables through L1**, as in `tv_tetra48_h`. Staging the 19.7 KB of decoder and S tables in shared memory per CTA would
  move 1.36 GB a pass over 69,120 CTAs (*computed*, `calc/brick/geometry_budget.out`). Shared tables belong to a
  persistent CTA (4E), under decision 8.
- `rot_quant_epilogue` (device function): from the f32 rotation output, per-slice amax (warp reductions), the
  exponents, the B records and the f32 tail. Partition-invariant. Used by `rot_apply_q` and `rot_apply_rows_q`.
- `tv_t48i_dp4a<R>`, R = 1..16 tokens: the k = 1 decode kernel and the SIMT rows kernel for prefill. The lane decodes
  its word into six signed quads, then per token runs `dp4a.s32.s32` on the hi words, a shift by 8, `dp4a.s32.u32` on
  the lo words, a shift by s_σ, and `mad.wide.s32` with S into an int64 accumulator.
- `tv_t48i_mma<NT>`, NT = 1 (k ≤ 8) or 2 (k ≤ 16) n8 tiles, the flat k-row kernel. Per k-step:

```
ldmatrix_x4(a, tile + swz_addr(lane));                  // rows g, g+8 of one block
s_g  = __shfl_sync(~0u, a[0], (lane & ~3) | 3);         // S_int of row g (pad bytes)
s_g8 = __shfl_sync(~0u, a[1], (lane & ~3) | 3);         // S_int of row g+8
for n in 0..NT:
    uint2 bh = lds64(Bhi[blk][n] + 8*lane), bl = lds64(Blo[blk][n] + 8*lane);
    int d[4] = {0,0,0,0};
    mma_s8s8(d, a, bh, d);                               // P_hi
    for e in 0..4: d[e] <<= 8;                           // C operand of the lo MMA
    mma_s8u8(d, a, bl, d);                               // P exact, |P| < 2^21.643
    for e in 0..4: acc[n][e] = mad_wide(d[e] << sh[n][e&1], e < 2 ? s_g : s_g8, acc[n][e]);
```

**The geometry is a new choice, and A3 is its prior.** The tile-major stream hands a warp 16 rows. Keeping today's
1,105,920 warps a pass with 16-warp CTAs needs a 16-way K split per 16-row tile: inside the CTA (K-groups, each with
its own activation buffer) or across CTAs (int64 atomics). Four points of that family on the 252 bench matrices
(*computed*, `calc/brick/geometry_budget.out`; staging is the L2-to-shared activation bytes a pass, atomics are int64
`red.global.add` a pass, smem is warp tiles, buffers and int64 scratch per CTA, no tables):

| geometry | CTA | k = 1 | k = 8 | k = 16 |
|---|---|---|---|---|
| served | 8 rows, whole-row f32 staging | 1.82 GB, 0 | 14.5 GB, 0 | 29.1 GB, 0 |
| G1 | 16 rows × 16 K-groups, whole row staged, no split | 0.45 GB, 0, 23-37 KB | 3.62 GB, 0, down over 99 KiB | over 99 KiB |
| G2 | 64 rows × 4 K-groups, split 4 across CTAs | 0.115 GB, 4.4 M, 22.5 KB | 0.92 GB, 35 M, 68 KB | 120 KB, over 99 KiB |
| G3 | 128 rows × 2 K-groups, split 8 | 0.059 GB, 8.8 M, 20 KB | 0.47 GB, 71 M, 48 KB | 0.94 GB, 142 M, 80 KB |
| G4 | 256 rows × 1 K-group, split 16 | 0.030 GB, 17.7 M, 19.5 KB | 0.24 GB, 142 M, 44 KB | 0.47 GB, 283 M, 72 KB |

All keep the warp count and halve the CTA count (69,120 against 138,240). In exchange the brick takes on what A3
measured as losses at k = 1 on the same card and the same warp-per-row family. The first is a cross-CTA split with
partials and a counter: `sk1` −1.87 % [−2.50 ; −1.77], `sk2` −9.87 % (*measured*). The journal's reading: filling the
grid costs (partials, tickets, fixup, shorter CTAs) and returns nothing (`a3-occupation-banc-2026-09-01.txt:96-103`).
The second is lower occupancy: 16-warp CTAs at 64-84 registers give 16-32 resident warps per SM against 48 for
`tv_tetra48_h` at 40 registers (*computed* on estimated registers). A3's `mr4` lost 24.62 % with 64 registers, 4 CTAs
per SM and the grid divided by 4 (`:105-109`). The brick does not divide the warp count; its occupancy cost is
unmeasured. The staging cut (4-61× at k = 1) is the one term in its favour, and A3's `pers`, which staged x once per
CTA, gained only +1.56 % (`:37`, `:119`). β0 times G2 and G4, each with its own null arm, before any decoder is written.

**Cross-CTA combine, self-cleaning.** Each CTA sums its warps' partials in shared memory in any order. It then issues
one `red.global.add.u64` per (row, token) into a scratch that persists across launches, and, after `__threadfence()`,
one `atomicAdd` on its row group's counter. The CTA that reads Cx − 1 is last. It reads the sums through L2
(`ld.global.cg`), runs the epilogue, then stores 0 to every slot it read and to the counter. The next launch, ordered
by the stream, starts from zeros, so no memset joins the 1,410 launches a token. The scratch is zeroed once at
allocation. The load-time check runs the same matrix twice and requires equal bits, which a missed reset breaks; the
host emulator carries the mutant "last CTA skips the clean". Arrival order cannot move a bit.

### 3.5 Budgets

**Floor plus work, at k = 1.** Write t = F + W. F is the geometry's null arm: launch, staging, combine, epilogue and
write, with no weight read. W is the weight stream, decode and multiply beyond it. For the served geometry
F = `nullk` = 2.294 ms and W = 3.734 − 2.294 = 1.440 ms. The DRAM floor of the pass's 0.98 GB is 1.134 ms, so W
already runs at 79 % of the DRAM rate (*computed* on banc-252, *measured*). The pass costs 8.87 clk per row-block per
SM, of which DRAM is 2.69 and the floor's share 5.45 (*computed*). The regime is latency- and launch-bound. Two
consequences follow.
- A kernel that keeps nullk's floor cannot go below 2.294 + 1.134 = 3.43 ms, 0.89 of the 3.85 ms estimated for
  `tv_tetra48_h` (*computed* on an *estimated* denominator). The brick can beat that only with a lower floor of its own,
  which its smaller staging allows and its occupancy and combine oppose.
- `docs/format-noyau.md` §6 forbids subtracting `nullk` from an arm on another grid, and QTIP's own geometry already
  finished under it, 2.246 ms against 2.306, while reading 0.91 GB (*measured*, F2). So the brick's floor is its own
  null arm, timed in β0 and bench β. W(k) = t(k) − F(k) is legitimate on the same grid.

**Growth with k.** For IMMA, the MMA count and the B loads are constant up to k = 8 (one n8 tile holds 8 tokens) and
double at k = 16. What grows with k is F: staging (×k), int64 atomics (×k) and the epilogue (×k outputs). So
r(k) = 1 + [F(k) − F(1) + W(k) − W(1)] / t(1), and the unknowns sit in two L2 rates that nobody has measured on the
L40S. Under assumed rates of 3-7 TB/s and 0.1-0.5 T int64 atomics/s, the best point of the family adds 0.18-0.58 ms at
k = 8 and 0.39-1.62 ms at k = 16. That gives r(8) = 1.04-1.17 and r(16) = 1.09-1.48 from the floor alone (*estimated*,
`calc/brick/geometry_budget.out`). With the work growth and latency that DRAM may not hide, the estimates are
r(8) = 1.03-1.35 and r(16) = 1.08-1.70 (*estimated*).

**Per row-block on sm_89, k ≤ 8, tables through L1**, the pipe view (*estimated* from source counts, ±30 % until a
card measures it). It bounds the work term only and says nothing of the floor:

| item | k = 1 dp4a | IMMA, one n8 tile | each extra n8 tile |
|---|---|---|---|
| L1 table gathers: 3 rows and 3 trellis per row-block | as served | as served | 0 |
| decode stores / ldmatrix / B loads / S shuffles | none / none / ~0.4 / none | 0.25 / 0.25 / 0.25 / 0.13 | +0.25 (B) |
| ALU pipe (PRMT, LOP3, SHF, SEL), ~55 lane-ops | 0.86 | 0.86 | +0.13 (shifts) |
| FMA-heavy pipe (IMAD, dp4a, IMAD.WIDE) | ~0.4 | ~0.5 | +0.13-0.25 |
| tensor, 2 MMA per 16 row-blocks at 1,024 int8 MAC/clk/SM | 0 | 0.50 | +0.50 |
| DRAM at 100 % of 864 GB/s, 6.5 B per row-block | 2.69 | 2.69 | 0 |

An operation count does not predict a time. METHODE §7 records it from P1b on Metal (×1.002 *estimated*, ×2.17
*measured*, `p1b-marche-bloc-2026-08-15.txt:76`). On CUDA, a source-level instruction count missed by a factor of 2 on
the August kernel (`docs/archive/spec-lot-a-2026-08-05.md:133-134`). β0 and bench β exist for that reason.

**Shared memory per CTA** (*computed*, tables through L1): warp tiles 1 KB per warp, the activation buffers and the
int64 scratch of the table above. G3 and G4 reach k = 16 under the 101,376 B opt-in; G2 stops at k = 8. In a
persistent CTA, where the tables would sit in shared memory (18,816 B), a SIMT rows kernel that streams the same
two-slice ring in f32 needs 2 × 384 × 4 B a token, 48 KB at k = 16, 67 KB in all, and fits the opt-in as the brick does
(*computed*). So fit does not separate the two designs. The brick's case is the cost of an extra token: two MMAs per 16
rows per 8 tokens against 24 f32 FMA and 24 shared loads per block per token. Bench β measures it, `tetra48rows`
against `t48imma`.

**Registers** (*estimated*): A 4, B 4, D 4, int64 accumulators 8 per n8 tile, decode about 25, weight ring 8,
addressing about 10: 64-72 at k ≤ 8 and 76-84 at k = 16. The gate is 0 local bytes, read back by `Cuda::report`; the
count is reported against an expectation of ≤ 96 at NT = 2.

### 3.6 Whole step and rotation

The brick needs only the quantizing epilogue from the rotation. At k = 1 it rides on `rot_apply_q`. For k tokens,
`rot_apply_rows_q` runs one CTA per token, so it stays flat up to 142 tokens (*computed*: one 1,024-thread CTA per SM).
The epilogue adds slice amax reductions, exponents and B-record writes to 96 rotations a token. At an assumed
0.3-1.5 µs each that is 0.03-0.14 ms a token (*estimated*). Bench β times it in `rotbench` and adds it to every k = 1
ratio. A persistent kernel that recomputes the rotation in every CTA pays k × 0.44-1.5 µs per site, so for k ≥ 2 it
must use the two-phase form of floor 1b; that is 4E's design.

**Whole-step r(8)** (*estimated*, `calc/brick/geometry_budget.out`). The Tetra share of today's step is 2.5-2.8 ms
(audit §2: 3.3-3.7 ms for the 168 Tetra and 84 int4 projections, *estimated*). Per extra token, the brick adds
(r_kernel(8) − 1) × share / 7 = 0-0.14 ms, SIMT int4 rows 0.07-0.13 ms and SIMT head rows 0.04-0.05 ms. Attention is
small at short context. So r(8) = 1 + 7 × (0.11 to 0.32) / 7.44 = 1.10-1.30 today. At the lean 2.6 ms step of 4E,
with a Tetra share of 1.13-1.5 ms, it is 1.30-1.69. The lean upper end crosses the 1.45 kill proposal of U6. Below 1.3
at a lean step needs the int4 and head kernels on the integer template too.

### 3.7 int4 records and the head

Keep today's numerics first: SIMT rows kernels for `tv_q4_hh` and `tv_emb_q4_h` decode each word once and run one
`__fmaf_rn` chain per token, so they are bit-identical to k = 1 by construction and need no T8. The IMMA form (u4 codes
as A, the natural-basis activation as int24 in three planes per 128-group, zero point folded through X_g = Σ x̃ computed
once per token) is a later, separate T8. Natural-basis inputs need int24, because nothing spreads their outliers: on a
synthetic massive channel int16 per group costs about 1 % per element for the other members of its group (*computed*,
step amax/32,768 against a 1,000× outlier). Three planes overflow int32 (128 × 15 × 2^23 > 2^31), so the combine is
int64 there too.

### 3.8 Integration in the Rust code

- `llvq-artifact`: a `tetra48_tiles` runtime layout beside `tetra48.rs`, pure Rust, `forbid(unsafe)` kept, with the
  Rust reference of §3.1 (quantizer, S grid with its refusal by name, int64 accumulation, epilogue).
- `llvq-cuda/kernels/llvq_t48i.cuh` (host-compilable device functions) and `t48i.cu` (entry points); arms registered
  last in `arms.rs`; quantizing-epilogue arms in `rotbench`. No new binary, so the two-lists trap of `Dockerfile.cuda`
  and `UPLOAD_ALLOW` is not touched.
- The served kernel as a bench arm: `tv_tetra48_h.cu` is embedded by `llvq-llm/src/fused.rs:479`, not by
  `llvq-cuda`. The bench needs it in its own unit, through a relative `include_str!` of that file and its prerequisites.
  That is code, not a recipe change.
- `llvq-llm/src/fused_cuda.rs`: a `DeviceStream::Tetra48Tiles` variant, launchers, and a load-time card check after
  `check_int4_half_parity`: dp4a k = 1, IMMA k = 1, 8, 16, the host-replayed rows on a real record, and two launches of
  the same matrix, bit for bit, before any token.
- `llvq-llm/src/rotplan.rs` and `model.rs`: the quantizing rotation, the k-row path without `regroup` copies.
- A new config file with a `kernel` field, as `attention` was added, built on `qwen3-4b-tetra-e4-grouped.json`;
  existing configs unchanged.

### 3.9 What the in-flight branch needs from this lead

The operator is implementing the audit's steps 2-5 now (`docs/archive/audit-debit-2026-10-29.md` §5). Nothing here
re-plans them. Four points would keep the brick possible later at no cost now.
- **Step 2, norm + rotation and SiLU·mul + rotation fused.** Keep the rotated output a separable last phase: the f32
  vector complete in shared memory, then written. A quantizing epilogue can then follow without reopening the fused
  arithmetic. Keep every reduction tree fixed by the vector, not by the launch: the same blockDim and order at one row
  and at k rows, one CTA a token. The k-row version then returns the bits of k one-row calls.
- **Step 3, `tv_tetra48_seg`.** Keep segment boundaries at multiples of 16 rows. They are today: q 4,096 | k 1,024 and
  gate 9,728 | up 9,728 at 4B, 4,096 | 1,024 and 12,288 at 8B, 5,120 | 1,024 and 17,408 at 14B. Keep the activation a
  typed argument of the launcher in `fused_cuda.rs` rather than a bare f32 pointer threaded through the dispatch, so a
  B-record activation can be passed later. Its bench job could carry the incumbent rows arms at four (R, tile) pairs
  for little extra cost (§5, S4).
- **Step 5, the GEMM prefill route.** Exact speculation needs verify = decode: the k-row verify pass must return the
  bits of k one-token decode steps. It does not need prefill = decode. A cuBLAS prefill is 1c's own T8 and does not
  block the brick.
- **Step 4, the Metal int4 wall.** No interaction: the brick has no Metal path in this plan.

## 4. Deciding unknowns

| unknown | best value now | confirms | kills (proposal; the kill is the operator's) |
|---|---|---|---|
| U0 served gain-level span | unknown; bench file 2.20 (3.49 binades), synthetic (*computed*) | every record with two nonzero levels at ratio ≤ 3.14, three files (S0) | none: a refusal by name and the two-accumulator form for the named matrices |
| U1 T8 cost of int16 per slice | \|Δppl\| < 0.03 % (*estimated*). Anchors: grouped attention +0.0095 % [−0.0048, +0.0238] (*measured*, ETAT §4); Metal against CUDA at f16 on the 8B, [−0.009 %, +0.010 %] (*measured*, `ppl-bases-carte-2026-10-05.txt`); synthetic noise-power share 3 % to 13-15 % at the median (*computed*) | both paired intervals inside ±0.12 %, S3 then S8 | int16 and int24 both out: stop |
| U2 real activation shape | unknown, no dump. Synthetic slice amax/rms, p50 / p99: 3.1-3.2 / 4.5-5.4 Gaussian, 5.4 / 12.3-12.5 for t4, 2.2-2.9 / 3.9-5.1 for a rotated massive channel; the cap binds in 0 %, 0.1-0.9 % and 0 % of vectors (*computed*) | reported, never gated | none: p99 > 32 at a site is journaled as a risk for 8B and 14B |
| U3 kernel r(8), r(16) on real words | 1.03-1.35 and 1.08-1.70 (*estimated*, §3.5) | r(8) ≤ 1.30, the audit's entry gate (audit §5, step 6) | r(8) > 1.60: no flat verify on sm_89 |
| U4 k = 1 speed against the served kernel | dp4a 0.88-1.30, IMMA 0.90-1.40 of its time (*estimated*: the floor-plus-work bound at the low end, A3's losses at the high end) | the faster brick arm ≤ 1.00, the report's own condition (rapport-debit-vague2.md, gate 4K) | above 1.00: the brick at k ≥ 2 only makes verify ≠ decode, so exact speculation is off unless the operator takes the k = 1 tax |
| U5 card equals host under NVRTC | by construction, never run | 0 differing rows, 252 matrices, every k, every round | a mismatch that survives one fix |
| U6 whole-step r(8) | 1.10-1.30 today, 1.30-1.7 lean (*estimated*, §3.6) | ≤ 1.30 in situ, S9 | > 1.45 with int4 and head on the template; the lean upper end crosses it with SIMT int4 and head |
| U7 L2 rates behind F(k) | staging 0.24-0.92 GB and 35-142 M atomics a pass at k = 8; 0.47-0.94 GB and 142-283 M at k = 16 (*computed*); both rates unmeasured | r(16)/r(8) ≤ 1.15 | > 1.40: verify capped at k = 8 |

## 5. Validation ladder

Every step that produces a number needs an operator go; every measurement with noise needs a stamped prereg (rules 1
and 2). Benches inform; end-to-end gates decide; a kill is the operator's (METHODE §1). The steps were renumbered in
the revision: S5 was L0, S3 was L1 and L2, S6 was L3, S7 was L4, S8 was L5, S9 was L6.

| step | settles | method | cost | prereg | decision rule (with otherwise) | go or ruling |
|---|---|---|---|---|---|---|
| S0 | U0 | `artstat` on the 4B, 8B and 14B sealed files: level ratio min, p50 and max, c0 = 0, negative levels (`artstat.rs:92-106`) | $0, existing binary, seconds | none: a deterministic read of stored fields; the journal keeps the binary's and the files' sha256 | every two-level record at ratio in [1/3.14, 3.14]: the S grid stands; otherwise the named matrices get the two-accumulator form, designed before S5 | go |
| S1 | U1, informs | `ppl 4096 12 metal <sealed 4B>` in f32 with today's binary, paired window by window with the card's f16 NLLs of `ppl-scelles-2026-10-04-brut/ppl-4b.txt` | $0, no new code, 5-10 min of Metal | `prereg-brick-1a.md` | see that file; it informs U1 and never replaces S3 | go |
| S2 | U2, optional | one per-token 384-slice amax line in `hcapture`, at its 2 × 128-token wiring shape. With no code, its existing diagonals give the slice-rms balance per site | $0, about a day of code, minutes of Mac | owed if run; numbers reported, never gated | informs the cap only; superseded by S3's statistics | go |
| S3 | U1 on the dense path, U2 | the int16 and int24 injection on the Mac, 12 WikiText-2 and 12 C4 windows, with the transparency, discrimination, seed and inertness gates first | $0, 2-4 days of code, 15-60 min of Mac | `prereg-brick-1.md` | see that file | go |
| S4 | U3, U4, U7 for the geometry; the incumbent's rows yardstick; the fragment map | β0: G2 and G4 with a trivial decoder, null and word arms, the served kernel and its rows entry point, the F₂ arm if ruled, the preflight fragment probe | about $0.5-1.1 a try, cap $3, about a week of code | `prereg-brick-0.md` | see that file | go, cap, decisions 7 and 9; decision 8 for the F₂ arm |
| S5 | the semantics is exact and replayable | Rust reference, host C++ replay, warp emulator, mutants | $0, 2-3 weeks | no | all mutants killed and replay = Rust: go on; a surviving mutant: fix the test first; otherwise stop | go to code a new numerics reference (rule 1), after S3 rule A or B and S4 rule A, or the operator's choice in S4 rule B |
| S6 | U3, U4, U5 on sm_89, with the real decoder | bench β, `prereg-brick-2.md` | about $0.8-1.5 a try, cap $3 | yes | see that file | go, cap |
| S7 | U3, U5 on sm_120 and sm_80 | bench β on `rtx-pro-6000` (`--any-flavor`) and `a100-large` (compute_80) | about $1.7 + $1.2 (f1d-rtx $1.67, a100-bench $1.13) | yes | identity on both and r(8) ≤ 1.3 on sm_120: keep the card list; otherwise a per-card fallback | go per card, decision 11 |
| S8 | U1 through the kernel, k = 1 speed at the door | the brick config against the grouped config: paired ppl by prefill and by teacher-forced decode on WikiText-2 and C4; served-door tok/s paired on the same image; GSM8K paired; MMLU at 40 a subject, both arms, reported | about $4.3, or $9.3 with the grouped GSM8K baseline and MMLU (below) | yes | below | decision 3 before serving |
| S9 | U6, exactness of speculation | a k-row verify step with SIMT int4 and head rows and Floor 3's attention; an oracle drafter replaying the greedy continuation | about $0.5-1 (*estimated*) | yes | a measurement: identity is a void gate (speculative = plain bit for bit); r(8) in situ is reported; the decision moves to Floor 5's end-to-end gate | Floor 3's T8, decisions 5 and 6 |

**S8 in detail.** The brick config is the grouped config plus a `kernel` field, because the 4B is served grouped since
2026-11-02 (ETAT §2). No task score has run under the grouped config (ETAT §5). If that debt is still open, S8 pays the
grouped GSM8K baseline, $2.13 on the gsm8k-4b-served anchor, and pairs against it. Prices: ppl about $1.6 (gqa-ppl-4b-r2,
$0.82 a corpus); the served door $0.54 (served-grouped-4b); GSM8K through the brick $2.13; MMLU at 40 a subject, two
arms, about $2.85 (f1e-recensement-2-bras); in all $4.3-9.3 (*estimated*). Rules, applied in order:
- **V.** Any gate fails: void, journaled, no reading.
- **A.** Both ppl intervals inside ±0.12 % on both corpora and both paths, GSM8K paired inside |d| ≤ 1.5 pp with McNemar
  p ≥ 0.05 (the engine-gate criterion of `gsm8k-wave1-2026-09-26.txt`), and the served-door tok/s ratio's round range
  not wholly below 1.00: the T8 is proposed to the operator and the brick config to serve k = 1.
- **B.** As A, but the tok/s range wholly below 1.00. The operator chooses: serve with the tax, which keeps exact
  speculation possible, or keep the incumbent at k = 1. In that case the brick runs at k ≥ 2 only, verify ≠ decode, and
  exact speculation is off.
- **C.** A ppl interval wider than 0.24 pp that contains 0, with none wholly outside the band: the control lacks power;
  the operator chooses between more windows and a stop.
- **D.** A ppl interval not inside the band: the int24 kernel is built and re-controlled, weeks of code, or the
  operator stops.
- **Otherwise** (ppl inside and GSM8K out, or any other split): not served, the injection and the kernel are audited,
  and the operator decides.

## 6. Build plan after validation

0. **β0, about 1-1.5 weeks.** The tile-major transcode (bijection tests, odd block counts, the word-zero pad), G2 and G4
   with trivial decoders, null and word arms, the self-cleaning combine, the epilogue, a Rust integer replay of the
   trivial product, the preflight fragment probe, and the served kernel embedded in the bench unit.
1. **Spec and references, 2-3 weeks.** Rust reference; `llvq_t48i.cuh`; `tests/host_t48i.cpp` and
   `tests/t48i_matches_rust.rs`; a host warp emulator (`tests/warp_emu.h`: 32 lanes as threads, two barriers per
   collective, with `ldmatrix`, `mma.m16n8k32` s8/u8 and `shfl` modelled on the CUTLASS layouts β0 checked). Drafts of
   the reference, the device k-step, the emulator and the Mac injection are in `wave5-deep/code/brick/`, never compiled
   or run. Mutants to kill: exponent off by one; ties away from even; clamp removed (mantissa 1.99997); cap Δ = 3
   removed; slice boundary off by one block; tail quantized; `hi = x/256` (truncation); lo read signed; nonzero B pad;
   trio order dropped; section and k swapped; int32 accumulator; E_S off by one; E_S taken over positive S only; S at
   m = 0 not zero; `cvt.rz`; rscale before 2^e; tail order reversed; F₂ packing b2 at bit 6; N0 split at 1239;
   emulator a1/a2 swapped; last CTA skips the clean.
2. **Kernels and bench arms, 3-5 weeks.** The real decoder in the geometry β0 chose, dp4a, IMMA, section-per-thread,
   null arm, quantizing-epilogue arms in `rotbench`; `LLVQ_KERNEL_DIR` iterates kernel bodies without a rebuild.
3. **Served integration, 1.5-2 weeks.** Layout variant, quantizing rotation, the load-time check, the config field.
   Serve dp4a first, for decode and the prefill rows, only if S6 and S8 find it not slower at k = 1. Otherwise the
   operator rules on the tax before any serving. IMMA joins later for k ≥ 2 with no new T8, since its rows are the same
   bits.
4. **int4 and head SIMT rows, 1-2 weeks**, bit-identical, no T8; the int4 and head IMMA form later (+2-3 weeks, T8).
5. **k-row verify path, 2-4 weeks** without the attention kernel.

Bench β comes after steps 0-2: 6-9.5 weeks of code (*estimated*), and 9-19 weeks at the verifiers' 1.5-2× factor. The
whole plan is 12-20 weeks, 18-40 at that factor: one person, queues of 30 min to 9.5 h, no nsys or ncu, warp code the
Mac does not run without the emulator. Per card: sm_89 first. sm_120 runs by PTX JIT from compute_89, since NVRTC 12.4
cannot target compute_120. There full-rate INT32 makes the pass MIO-bound at about 1.4 clk against 1.80 of DRAM
(*estimated*, 188 SMs, 1,597 GB/s, an assumed 2.4 GHz). sm_80 is a third exactness witness only: its DRAM time is
0.48 clk per row-block and every k is SM-bound (*computed*). Inside 4E the brick is the projection task type; its
register count sets the persistent kernel's allocation, and shared tables become possible there (decision 8).

## 7. Rules map

- **T8 events.** (1) int16 semantics for the Tetra projections: one event for dp4a and IMMA together, since they are
  identical by construction. (2) int4 and head on IMMA with int24 planes: a separate later event. (3) A row-invariant
  attention for exact speculation: Floor 3's event, not this one. (4) A cuBLAS prefill route: 1c's event.
- **A2.** Untouched: no graph capture, ordinary launches. Using the brick inside 4E waits for 4E's own A2 reading.
- **Rule 6.** 0 bits added to the format; the tile-major stream drops 3.3 MB of row padding, about −0.007 b/param over
  the whole 4B (*computed*). The F₂ arm would shrink the trellis tables from 2.3 KB to 512 B.
- **Rules 4, 5, 7, 8.** Same-head ratios at any publication; no × across cards; round-by-round medians; labels.
- **ETAT §6 (rule 3).** No row reopened. "A mixed Tetra plus int4 bench arm" binds β0 and bench β: they time the 252
  Tetra matrices of the bare file only, and the int4 kernels are timed end to end. A3 is not an ETAT §6 row; its
  journal is the measured prior of §3.4, not a lead this one reopens.
- **docs/archive settings, a go each (rule 1).** "Tensor cores, ldmatrix, `__dp4a` : sans objet" of 2026-08-05 carries
  the reopen clause "si le prefill entre au périmètre". k-row verify and prefill are both new facts, and dp4a already
  runs in the served kernel. "Table en shared": the non-persistent brick reads tables through L1, as the served kernel
  does, so the setting is not touched; only a persistent use raises it (decision 8). "Décodage gelé jusqu'au SASS":
  no SASS tool exists in `ops/` or the crates. `ops/Dockerfile.cuda:84-86` ships the kernels as text, and NVRTC emits
  compute_89 PTX that the driver JITs (`gpu.rs:262-282`). The brick keeps the v3 decoder bit for bit, so the setting
  holds; the F₂ arm touches it and waits for decision 8. A pre-job dump would need a recipe change, an operator
  decision like nsys (decision 13); it would read toolkit 12.4 ptxas, not the driver's JIT. The 2026-08-05 speculative
  verdict: β0 and bench β supply new facts, and lifting it is decision 5.
- **Operator rulings.** Decision 2, a cap per paid step (default: no paid job). Decision 7, tensor cores, dp4a and
  ldmatrix in scope (default: the brick does not start). Decision 9, the ROADMAP §4 hold, "Batch above 1, and prefill
  ... this reopens the layout if prefill is ever served" (`docs/ROADMAP.md:102`). The brick is a k-row path with a
  new VRAM layout. The default literal reading holds the multi-row path, so S4 and every later step wait for it. Decision
  8, its decode-freeze reading, for the F₂ arm. Decision 3, the T8, after S3 and S8. Decisions 5 and 6 before any
  speculation code (S9). Decision 11, the sm_120 venue (S7). The config field name. S0-S3 run on the Mac at $0, time
  no multi-row kernel, and need a go only.

## 8. Checked now at $0

1. The v3 tables are biased bytes `val + 128` in [112, 144], and the select is bytewise, so one XOR gives int8 quads
   (`llvq_f1rank_v3.cuh:44-50, 89-96, 146-152`).
2. The served kernel already calls `__dp4a` for the shell norm (`llvq_tetra48.cuh:86-95`).
3. The bounds max|y| = 10, n2 ≤ 416, m ≤ 26 (`llvq_tetra48.cuh:26-30`) give Σ|y| ≤ 99.92, |P| < 2^21.643, 2^24.643
   after the slice shift, |I| < 2^61.3 at 405 blocks and 2^62.7 at 1,067 with |S_int| < 2^28 (*computed*,
   `calc/brick/geometry_budget.out`). There are 2 × 26 = 52 nonzero scales.
4. `linear_input` packs s8 at bits 0-5, b1 at 6, b2 at 7-10, b3 at 11 (`trellis.rs:126-128`); `derive_columns` asserts
   all 4,096 inputs and the pinned columns (`trellis.rs:210-220`); two 64-entry tables are 512 B.
5. The transcode only moves the word between two conventions (`tetra48.rs:1-36, 113-175`); `tetra_fingerprint` digests
   the map, not the runtime layout (`codebook.rs:216-262`).
6. CUTLASS `mma_traits_sm80.hpp` (main): A ((4,8),(4,2,2)):((64,1),(16,8,256)), B ((4,8),(4,2)):((32,1),(8,128)), C =
   `SM80_16x8_Row`; S8U8, U8S8 and U8U8 inherit them (*measured*, primary source). The verifier re-derived the map.
7. The XOR swizzle on bit 2 of the record index makes decode stores and `ldmatrix.x4` conflict-free (8 wavefronts per
   round against 16), and the B layout reaches the 2-wavefront minimum (*computed*, enumeration, re-derived by the
   verifier).
8. ExLlamaV3 (master, primary source): plain int8 is the CUDA default, "~0.9% output RMS deviation"; its residual mode
   gives "~15-16 bit effective activation precision". The scale is amax/127, not a power of two, on Hadamard-transformed
   activations, for the mul1 codebook only, K ≤ 5 on Ada and Ampere. These are author comments, not journals. The
   kernel issues mixed-sign `dp4a.u32.s32` through inline PTX (`exl3_gemv_int8_kernel.cuh:66`, local copy), the form
   the lo plane needs.
9. Synthetic activations, 2,000 vectors a case, median and p99 (*computed*, `calc/brick/synthetic_actq.out`): f16
   rounding 2.08e-4 relative rms; int16 per slice 3.5-3.6e-5 (Gaussian), 7.6-7.9e-5 and p99 1.1-1.5e-4 (t4), 2.3-3.7e-5
   (rotated massive channel); noise-power share against f16 3 % (p99 5 %), 13-15 % (p99 30-45 %), 1-4 % (p99 7-18 %).
   The cap binds in 0 % (Gaussian, massive channel), 0.1-0.9 % (t4) and 2-10 % (t3) of vectors and costs at most 8 %
   (t4) and 17 % (t3) of rms in the worst vector. int8 per vector 0.8-0.9e-2 Gaussian, the order of ExLlamaV3's 0.9 %;
   int24 1.4e-7. The verifier's 6-vector rerun (t4 up to 57 % of the f16 noise power) falls inside the t4 tail here.
10. CodeGEMM (arXiv 2512.17970) Table 9 is the aggregate linear-layer latency of one Llama-3-8B decoder block against
    batch 1-16 on the paper's A100. QuIP# 163 → 1,554 µs (×9.53), QTIP 190 → 1,991 (×10.48), CodeGEMM m1v4 153 →
    1,416 (×9.25), cuBLAS 332 → 340 (*computed* on measured values). No published codebook or trellis kernel shows a flat
    t(k).
11. magnitude (local snapshot, untrusted): codes enter `mma.sync.m16n8k16` as exact integers, activations are rounded to
    bf16 or f16, and rows 1..16 "match M = 1 bit for bit under one launch configuration"; a comment, not a measurement,
    tied to the launch configuration, which the int64 semantics does not need.
12. NVRTC compiles with `--fmad=true` (`gpu.rs:14-16`); the served epilogue `acc * rscale[row] + tv` and `tail_dot_h`
    are contractible, so today's card bits are not the host test's by construction. N rows in one call already differ
    from N one-row calls by 5.6e-2 to 6.8e-2 on a logit, never attributed (`f1e0-2026-09-10.txt:346-350`).
13. No activation dump exists: the recursive listing of `hf://buckets/Pier-Jean/jobs-artifacts` (920 files) holds no
    hessian, activation or `.npy` file (`wave5-deep/bucket-listing.txt`). `hcapture` writes diagonals, means and
    per-token norms in the encoder's basis, keyed by `effective_rotation_seed(0x110FEED, block, act)`
    (`hcapture.rs:40-46`, `l36-capture-2026-09-13.txt:39-45`). Its wiring run wrote to `/tmp/hcap`, which no longer
    exists.
14. Bench inputs exist: the ball file and the bare Tetra file of banc-252 (`ops/jobs/banc-252.sh:22-24`).
15. A3 on the L40S, k = 1, `Planes14`, one process (*measured*, `a3-occupation-banc-2026-09-01.txt:34-40`): `mr2`
    −10.45 %, `mr4` −24.62 %, `sk1` −1.87 %, `sk2` −9.87 %, `pers` +1.56 %, `persall` +26.36 % (one launch a round,
    declared not portable).
16. banc-252 (*measured*): `nullk` 2.294 ms, Tetra 3.734 ms, 0.98 GB a pass. `docs/format-noyau.md` §6: subtracting
    `nullk` from another grid is illegitimate; QTIP finished under it in its own geometry, 2.246 ms against 2.306.
17. No SASS tool in `ops/` or the crates (grep for cuobjdump, nvdisasm, ptxas: none). `Cuda::report` reads back
    `num_regs`, local and shared bytes (`gpu.rs:445-465`).
18. `ppl` reads no `LLVQ_THREADS`; only `smoke.rs:669` and `errmap.rs:314` do. This M3 Max has `hw.ncpu` = 16. `ppl`
    takes `c4` as its fifth argument (`ppl.rs:46-48`) and prints per-window NLLs.
19. Time anchors for Metal: `kl-temperature-4b-2026-09-16-brut/wikitext.log:4-10` ran 4B f32 windows of 1,024 positions,
    8 in 11 s dense and 9 s quantized, loads 35.4 s. The 8B in f16 at 4,096 positions took 21.6-30.6 s a window
    (`dclm-8b-2026-09-21-brut/run.log:161-172`, `dclm-8b-rowscales-2026-09-21-brut/ppl-ft-f16.txt`).
20. `rotation_seed` is stored per record (`format.rs:484`), and `decode_matrix` un-rotates with it (`format.rs:1527`).
    The dense path's `Proj::Dense` carries no rotation key (`sealed.rs:116-117`), so the injection must keep the seed at load.
21. The report's line "À k = 1, la brique tournerait 1,16 à 1,84 fois plus vite que le noyau servi" (rapport §6) is
    not supported: the floor-plus-work bound gives at most 1/0.88 = 1.14 times faster with nullk's floor, and A3's prior
    points to a loss.

## 9. Not established

- Any kernel of this design running; every speed and ratio here is *estimated*.
- The brick's own floor, its occupancy, the L40S L2 bandwidth and atomic rate, and so F(k).
- The T8 cost on the sealed model: S3 and S8 measure it.
- The shape of real rotated activations and the gain-level span of the sealed matrices.
- dp4a, PRMT and ldmatrix rates on sm_89, all sm_120 rates and clocks.
- Whether 128-bit shared loads with broadcast addresses cost fewer than four wavefronts.
- Card identity under NVRTC of the int64 path and of `cvt.rn.f32.s64` against the host conversion.
- The cost of the quantizing epilogue on the card.
- The whole-step r(8), which needs Floor 3's attention and the int4 and head rows.
- The schedule, which carries no margin for a failed bet.

## 10. Verifier findings and what changed

Every finding of the 2026-11-02 review was checked against its source. Twenty were right as stated and are fixed. Two
were right with a nuance, written beside them.

| finding | where it is fixed |
|---|---|
| A3 prior ignored at k = 1; occupancy 16 against 48 warps per SM | §1, §3.4 (geometry table, A3), §4 U4, β0 null arms; serving dp4a now conditional with its otherwise (§5 S8, §6 step 3) |
| a statistic overrode the paired criterion | `prereg-brick-1.md` §7: statistics reported, never gated |
| rules did not partition | `prereg-brick-1.md` §7: inside defined, rules V, A, D, B, otherwise C, in order |
| SASS precondition cannot run | dropped; gates read `Cuda::report`; §7 "Décodage gelé jusqu'au SASS" corrected |
| 3-5 weeks before bench β | §1, §6: 6-9.5 weeks (9-19 with the factor), and β0 after about one |
| time anchor misread | `prereg-brick-1.md` §8. Nuance: the misread log itself is the closest anchor (4B, f32, Metal, 1,024 positions); it and the 8B f16 log bracket the time, 15-60 min |
| k = 1 intervals below the floor | §3.5 floor plus work; U4 lower ends 0.88 and 0.90. Nuance: that bound holds for nullk's floor; the brick's own null arm can lower it, as the reviewer allowed |
| U6 used the optimistic kernel end | §3.6, U6: 1.10-1.30 today, 1.30-1.7 lean, past 1.45 |
| nobody zeroes the accumulators | §3.4 self-cleaning last CTA, atomics in the geometry table, combine timed in the null arms |
| "only k ≥ 3 design that fits" | §3.5 reworded: both fit; the case is the cost per extra token |
| incumbent identity as a void gate | `prereg-brick-2.md` §5, reported |
| quantizing epilogue outside the timings | §3.6; `prereg-brick-2.md` adds `rotbench` arms at k = 1, 8, 16 |
| nothing for the in-flight branch | §3.9 |
| `ppl` does not read `LLVQ_THREADS` | `prereg-brick-1.md`: the injection pool reads it and prints it; gate on that line |
| transparency gate blind to a wrong basis | `prereg-brick-1.md`: Q from each record's seed, a seed gate and a basis gate |
| decision 9 missing | §7 |
| L5 decided serving without throughput, L6 gated on a component | §5 S8 adds the served-door tok/s; S9 is a measurement; "int24 kernel built and re-controlled" |
| GSM8K baseline under the grouped config | §5 S8 names the base config and prices the baseline and MMLU |
| op-count citation misattributed | §3.5 and `prereg-brick-2.md` §1 |
| 6-vector synthetic figures | `calc/brick/synthetic_actq.py`, 2,000 vectors a case; §1, §8 item 9 |
| S grid on synthetic centroids; positive-only E_S; 54 scales | §3.1, S0, `code/brick/t48i_reference.hpp` |
| wiring checks after the control | `prereg-brick-1.md` §3: wiring and inertness first |

The revision also corrects one error the review did not raise. The first draft put the decoder tables in shared
memory per CTA. In a 69,120-CTA grid that would move 1.36 GB a pass, so the non-persistent brick reads them through
L1 as the served kernel does (§3.4).

The seven cheaper experiments the review proposed are all in the ladder: `artstat` (S0), the f32 against f16 pairing
(S1), the `hcapture` slice line (S2), and, inside β0 (S4), the incumbent rows yardstick, the geometry floor with its
null arm, the fragment probe and the F₂ arm. One reading differs. S1 informs U1 and does not demote S3 to a
confirmation: dominance in noise power is not a bound, because int16 per slice puts a uniform absolute error on the
small elements of a slice and f16 does not (`prereg-brick-1a.md` §1).
