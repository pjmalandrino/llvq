# VRAM format and fused kernel

The format the kernel reads in VRAM, the fused matvec that reads it, and the
measurement pitfalls, as of 2026-09-02. Project state is in
[ETAT.md](ETAT.md), the decision thread in [HISTORIQUE.md](HISTORIQUE.md),
the lab rules in [METHODE.md](METHODE.md). Every "vs FP16" is an L40S
result, unless the A100 is named (§7).

## 1. The problem

The sealed file codes each block of 24 weights in 48 bits: a 47-bit index
into the Λ₂₄(12) ball and one gain bit, which is 2.000 b/weight of code
(*computed*, [fiche-4b.md](fiche-4b.md)). The ball holds N(12) = 111,043,117,458,000 points, 1.1·10¹⁴
(*computed*, [fiche-4b.md](fiche-4b.md), locked by
`classes_reproduce_theta_series` in `llvq-core`).
Three paths exist for reading that index inside a matvec.

| path | what it requires | verdict |
|---|---|---|
| lookup table | 1.1·10¹⁴ entries; a 16-bit trellis state (QTIP) fits in 2 KiB, an E8P codebook (QuIP#) in 2¹⁶ entries | impossible for the Leech lattice |
| unfolding at load | transcode the index into bit planes read by shift and mask, 4.804 b/weight kernel | served (`Planes14`, §3) |
| online arithmetic decoding | rank, Golay, cosets (E3, `Golay70`, E1v) | closed, every arm compute-bound (§2) |

The served format therefore unfolds 2.000 bits of information into 4.804 b/weight
(*measured*, [e2-golay70-bench-2026-08-07.txt](mesures/e2-golay70-bench-2026-08-07.txt)),
paid at memory speed. Unfolding is the family's entry price, forced by the
size of the codebook (`paper/sections/layouts.tex`).

## 2. The layouts

Three accountings. *Payload*: record bits per quantized weight.
*Kernel*: payload, bases, f32 tail and f32 row scales, as the CUDA benchmark
bills it. *Warp-aligned*: kernel after padding each row to a multiple of 32
blocks, the only shape the served matvec can read. The "vs FP16" and GB/s
come from the ten-arm benchmark of 2026-08-21, a single process (*measured*,
f2-p3-qtip-banc). The `Golay70` verdicts were returned on the 08-07 and
08-11 runs, at 1.31× [1.29–1.32] and 1.77× [1.76–1.78]; the 08-21 benchmark
replays them at 1.34× and 1.78×.

| layout | payload | kernel | warp-aligned | vs FP16 L40S [range] | GB/s | status | journal |
|---|---|---|---|---|---|---|---|
| `Slot32` | 5.3756 | 5.510 | n/a | 1.89× [1.89–1.89] | 431 | wired, not served, fallback `LLVQ_FUSED_LAYOUT=slot32` | [k1c-rtbits](mesures/k1c-rtbits-2026-08-05.txt), F2 |
| `Planes14` | 4.6667 | 4.804 | n/a | 2.15× [2.15–2.16] | 428 | served, default | [c1-planesbench](mesures/c1-planesbench-2026-08-06.txt), F2 |
| `Planes12x` | 4.2029 | 4.342 | n/a | 2.00× [2.00–2.00] | 359 | wired, measured as served once, not default | F2, [g-horloges](mesures/g-horloges-planes12x-2026-08-23.txt) |
| `Golay70` v1 | not published | 3.589 | n/a | 1.34× [1.34–1.34] | 199 | dropped on 08-07, criterion 1.6× | [e2-golay70](mesures/e2-golay70-bench-2026-08-07.txt) |
| `Golay70` v2 | not published | 3.589 | n/a | 1.78× [1.77–1.78] | 264 | wired as `golay70`, not adopted, 2.0× threshold timestamped | [golay70-v2](mesures/golay70-v2-sept-bras-2026-08-11.txt), [prereg](../proofs/preregistration-2026-08-11.md) |
| `E1c14` | 4.4167 | 4.5551 (unaligned) | 5.2354 at 4B (+9.0%); 4.6410 at 14B (−1.4%) | never measured | n/a | absent; buried at 4B | [x3-alignement](mesures/x3-alignement-warp-2026-08-15.txt), [rtbits-14b](mesures/rtbits-14b-2026-08-17.txt) |
| `E1c12` | 3.6196 | 3.7618 (unaligned) | 4.2880 at 4B (−1.3%); 3.8021 at 14B (−10.4%) | never measured | n/a | absent; open, a question of speed | [e1c12-aligne](mesures/e1c12-aligne-2026-08-16.txt), rtbits-14b |
| E1v | not published | 2.3877 | 2.3983 (row-aligned cut) | 0.25× [0.25–0.25] | 25 | closed for the served path, criterion 1.60× | [e1v-cuda](mesures/e1v-cuda-2026-08-16.txt) |

The payloads of `Slot32`, `Planes14` and `Planes12x` come from the same
sweep of the sealed 4B (*measured*, e1c12-aligne). The `E1c` bits are
*computed* on the real blocks. The warp-alignment penalty is +15.47% of
blocks on the 4B shapes and +4.18% on the 14B ones (*computed*,
x3-alignement, rtbits-14b). The 14B rows are longer, 213 and 725 blocks
against 106, 170 and 405. An alignment verdict is therefore tied to its size.

Four states: *absent* (no code selects it), *wired*
(`LLVQ_FUSED_LAYOUT` accepts it, so it is measurable), *measured as served,
not default* (`Planes12x` since G3), *served* (the default, where the
published numbers come from). `LLVQ_FUSED_LAYOUT` refuses any other value
(`llvq-llm/src/fused.rs`).

E3, which decoded the file index inside the kernel, is buried on paper:
3.0444 b/weight kernel against a criterion of 2.60 (*computed*,
[radixstudy-x4-2026-08-12.txt](mesures/radixstudy-x4-2026-08-12.txt)). The
X3 thresholds for `E1c` (2.05×, 1.9×, 1.6×) are set in unaligned accounting
and must be re-anchored before any benchmark, otherwise the benchmark
measures a misalignment.

## 3. The Planes14 layout

A `Planes14` record is 112 bits, exactly 14 bytes, at offset 14·b for block
b, LSB-first (`llvq-cuda/src/planes14_host.rs`):

```
[class : 9][gain : 1][smask : 24][plane0 : 24][plane1 : 24][plane2 : 24][0 : 6]
```

The level of slot j is `plane0[j] | plane1[j] << 1 | plane2[j] << 2`. It
indexes the class's canonical values, read from a constant table of 384
entries. `smask` carries the sign of slot j, with a zero bit on zero slots.
Three planes cover up to eight levels. Real blocks carry three to five,
65.9% carry four, and fewer than 0.1% carry one or two
(*measured*, [k1c-rtbits-2026-08-05.txt](mesures/k1c-rtbits-2026-08-05.txt)). The stride
is uniform, with no base table. The payload is 112/24 = 4.6667 b/weight.

A lane decodes a block in 24 fixed steps: three bit tests for the level,
one bit for the sign, no divergence. It reads four to five 32-bit words per
block. `Slot32` reads a record of 9 + 1 + 24·L bits at a stride of 11, 14
or 17 bytes, in a five-word window. At identical decoded content,
`Planes14` runs 1.14× [1.14–1.15] faster than `Slot32` at constant GB/s:
time falls with the bytes (*measured*,
[c1-planesbench-2026-08-06.txt](mesures/c1-planesbench-2026-08-06.txt)).

`Planes12x` adds a sparse overlay. Blocks with four levels or fewer take a
12-byte record with two planes. The 5,096,688 five-level blocks
(3.3824% of 150,681,600, *measured*, [HISTORIQUE.md](HISTORIQUE.md)
2026-08-09) keep their 14-byte `Planes14` record in a per-matrix exception
table. The same launch adds the (exact − approximate)·x correction, memset
included. The reconstruction is bit-exact. The hard four-level cap, with no
overlay, costs +4.75% perplexity
(*measured*, [verdicts-lot-b-2026-08-06.md](archive/verdicts-lot-b-2026-08-06.md)).
The overlay's price is a second irregular stream: the fraction of the byte
bound falls from 65% to 54% (*computed*,
[six-arm-awq-2026-08-10.txt](mesures/six-arm-awq-2026-08-10.txt)).

Every layout is checked against an f64 reference over the 4B's 1,105,920
rows, threshold 1e-5. The worst errors are 2.2e-8·Σ|w·x| for `Slot32` and
`Planes14`, 2.9e-8 for `Planes12x` (*measured*, e2-golay70-bench). The
bijection is proved over the 150,681,600 blocks.

## 4. The fused matvec

The kernel puts one warp per output row: the activation staged in shared
memory, one lane per block, `warp_sum`, the f32 tail epilogue
(`TailPolicy::KeepExact`) and the write of y. Every projection is preceded
by an incoherence rotation `rot_apply`, a single-block kernel that puts the
whole activation in f32 shared memory: 8.05 µs at n = 2560 in isolation
(*measured*,
[rotation-cuda-2026-08-05.txt](mesures/rotation-cuda-2026-08-05.txt)).
`fused_cuda` replaces the 4B's 252 `Linear` layers with these two launches;
`bin/fusedrun` calls it inside the model.

The served v1 configuration fuses `q+k+v` and `gate+up` by rows
(`LLVQ_FUSE=1`) and hoists the rotation to the group (`LLVQ_ROT_SHARE=1`).
The 252 matvecs per token become 144 at 4B, the 280 become 160 at 14B.
`check_fuse` refuses `FUSE=1` with `ROT_SHARE=0`: a fused group is a single
rotation site, and a delta carries only one mechanism.

| size | tok/s v1 [range] | GB on card | fusion gain at constant `ROT_SHARE` | exact overhead | journal |
|---|---|---|---|---|---|
| 4B | 100.6 [99.9–100.7] | 2.57 | ×1.061 [1.050–1.069] | +3,686,400 B (+0.008117 b/weight) | [d1-fusion](mesures/d1-fusion-servie-2026-08-24.txt) |
| 8B | 75.5 [75.5–75.6] | 5.41 | ×1.055 [1.054–1.058] | +4,423,680 B | [vague2](mesures/vague2-fusion-8b-14b-2026-08-31.txt) |
| 14B | 46.8 [46.7–46.8] | 9.40 | ×1.028 [1.027–1.029] | +6,717,440 B | vague2 |

Everything is *measured*, band [1.00; 1.12] timestamped before the jobs
([D1 prereg](../proofs/preregistration-d1-2026-08-24.md),
[wave 2 prereg](../proofs/preregistration-vague2-gel-geometrie-2026-08-31.md)).
At 4B, the decomposition is 87.0 tok/s at `ROT_SHARE=0/FUSE=0` (*measured*,
[b2-fusedrun-plages](mesures/b2-fusedrun-plages-2026-08-18.txt)), then 94.9
[94.1–95.2] with the hoist alone (*measured*, d1-fusion), then 100.6; the
×1.091 of the hoist is a cross-job reading and does not get published. D1's six criteria are green:
128 identical tokens between the fused and unfused arms, divergence from
the dense arm at the same token 89, same sha256 of the NVRTC source. The
overhead is the gain centroid index (`gs_off`), one u32 per fused row:
921,600 rows, +3,686,400 bytes (*measured*, d1-fusion).

On the benchmark, fusion returns 11.7% of the matvec-only time on `Planes14`,
5.096 against 4.504 ms in f32 (`tv_planes_seg`, *measured*, d1-fusion).
That number does not carry over to the 6.1% per token of the served path:
two different quantities. The same-head series, the only one that measures
the kernel, is ×1.11, ×1.29, ×1.41 from 4B to 14B (*measured*,
b2-fusedrun-plages), at `ROT_SHARE=0/FUSE=0`. It has not been replayed
under v1.

Phase A bounds the rest of the geometry. The `persall` benchmark arm, not
portable, returns +26.36% [+25.31; +26.61] on the fused matvec. No portable
arm passes the 10% gate, and a split-K on `o` and `down` returns
−1.87% (*measured*, [a3-occupation-banc-2026-09-01.txt](mesures/a3-occupation-banc-2026-09-01.txt)).
CUDA Graphs (A2) return +13.45% at 4B and are not served, for a memory
reason ([ETAT.md](ETAT.md) §7).

## 5. Transcoding at load

The sealed file does not move. Loading unfolds the index into the requested
layout, once per process.

| transcode | duration | machine | journal |
|---|---|---|---|
| `Planes14`, 4B, 16 threads | 84 s | M3 Max | *measured*, [HISTORIQUE.md](HISTORIQUE.md) 2026-08-09 |
| `Planes12x`, 4B, 16 threads | 404 s (×4.8) | M3 Max | same |
| `Planes14`, 4B, `fusedrun` load | 130.9 s | L40S | *measured*, [b2-fusedrun-plages](mesures/b2-fusedrun-plages-2026-08-18.txt) |
| `Planes12x`, 4B, `fusedrun` load | 1,340 s | L40S | *measured*, [g-horloges](mesures/g-horloges-planes12x-2026-08-23.txt), G3 |
| `Slot32` + `Planes14` with bijection proof, benchmark | 150 s | L40S | *measured*, [c1-planesbench](mesures/c1-planesbench-2026-08-06.txt) |
| seven arms with block-by-block proofs, benchmark | 1,464 s | L40S | *measured*, [golay70-v2](mesures/golay70-v2-sept-bras-2026-08-11.txt) |

`Planes12x` remains non-default: served once, it returns 85.0 tok/s
[84.7–85.1] in 2.36 GB, against 87.0 in 2.56 for `Planes14` (*measured*,
g-horloges and b2-fusedrun-plages: −2.3%, −0.20 GB, two jobs). It redoes a
five-level lattice search per block, hence the 4.8 factor and then the
1,340 s on card. That cost is paid at every load, on a rented card: that is
the trade-off. On the ten-arm benchmark, `Planes12x` runs at 0.93× [0.93–0.93]
of `Planes14` on the projections alone; the rest of the model absorbs the gap.

## 6. The nullk floor

A pass over the 252 projections that reads no weight byte costs 2.306 ms,
45.2% of the served arm and 4.77× FP16 [4.76–4.77] (*measured*, F2; 2.305 ms on the
first run, [nullk-plancher-2026-08-16.txt](mesures/nullk-plancher-2026-08-16.txt)).
`tv_nullk` keeps the grid, the tiling, the two barriers, the activation
staging, `warp_sum`, the epilogue and the write of y. It removes the block
read and decode (31 registers, 0 local bytes).

| arm (same process) | med ms | GB read | b/weight kernel | GB/s | vs FP16 [range] |
|---|---|---|---|---|---|
| `nullk` | 2.306 | 0.07 | 0.159 | 31 | 4.77× [4.76–4.77] |
| QTIP 2 bits, competitor | 2.246 [2.245–2.248] | 0.91 | 2.000 | 405 | 4.89× [4.89–4.90] |
| AWQ w4g128, competitor | 3.252 | 1.90 | 4.179 | 584 | 3.38× [3.37–3.38] |
| `Planes14` | 5.103 | 2.18 | 4.804 | 428 | 2.15× [2.15–2.16] |
| FP16 cuBLAS | 10.830 | 7.27 | 16.000 | 672 | 1.02× [1.02–1.02] |
| FP16 in-house control | 10.994 | 7.27 | 16.000 | 661 | 1.00× |

`nullk` measures our launch geometry: one warp per output row,
252 launches. In that geometry, `Planes14` buys 3.11× net of the floor
(8.691 ms of FP16 traffic against 2.797). Its decode costs about 7% of the
traffic time, 779 GB/s net against 836 (*computed*, nullk-plancher). The
net figures are 836, 779, 710, 617 and 275 GB/s for FP16, `Planes14`,
`Slot32`, `Planes12x` and `Golay70` v1. The format is in play for at most 55%
of that time.

`nullk` does not measure the card. QTIP, launched in its own geometry
(`<<<128, 1024, 64 KiB>>>`, 252 launches as well), finishes the same
projections in 2.246 ms while reading 0.91 GB. The separation is 2.7%
against a resolution 2R = 0.72% (*computed*, F2). Its byte-bound fraction is
61.1% against a timestamped ceiling of 59.6%
([F2 prereg](../proofs/preregistration-f2-qtip-2026-08-20.md)); the erratum
is in the journal, the timestamp is not reissued. The ratio
r = t(`Planes14`) ÷ t(QTIP) is 2.27× [2.27–2.28] for 2.40× the traffic: at
similar efficiency (61% and 65% of the bound), time follows the bytes. Between
formats at different bit rates, the comparable quantity is GB/s: AWQ 584,
`Planes14` 428, QTIP 405. No quality claim rests on the QTIP arm, whose
payload is pseudo-random.

Subtracting `nullk` from an arm on another grid is illegitimate: AWQ would
give 2,006 GB/s net, above HBM, and QTIP a negative net (*computed*,
f2-p3-qtip-banc). At 144
launches the floor falls to 1.794 ms against 2.200 at 252 in the same
process, r = 0.8158 [0.8150–0.8162], which is 3.76 µs per launch
(*measured*, [a1-nullk-252-144-2026-08-31.txt](mesures/a1-nullk-252-144-2026-08-31.txt)).

The 2026-08-05 attribution has a different denominator: the 2.04 ms of
`Slot32` headroom per token above its DRAM floor (5.82 ms against 3.78).
The stream through the five-word window weighs 0.681 ms (33%), the
remainder 1.199 ms (59%) (*measured*,
[attribution-cuda-2026-08-05.txt](mesures/attribution-cuda-2026-08-05.txt));
splitting that remainder into latency-occupancy 39% (0.803 ms) and decode
19% (~0.396 ms) comes from the three "ground" kernels of batch A
([rapport-lot-a-2026-08-06.md](archive/rapport-lot-a-2026-08-06.md)).
The 45.2% is over 252 projections, the 39% over one token: bringing them
together requires redoing the attribution. The 39% is a property of our
launch geometry.

## 7. Domain of validity

On the A100-SXM4-80GB, no decoding arm beats FP16 (*measured*,
[f4-a100-2026-08-18.txt](mesures/f4-a100-2026-08-18.txt), same code, same
protocol, `LLVQ_NVRTC_ARCH=compute_80`).

| arm | med ms A100 | vs FP16 A100 | GB/s A100 | GB/s L40S |
|---|---|---|---|---|
| `nullk` | 4.107 | 1.68× [1.68–1.68] | 18 | 31 |
| FP16 control | 6.915 | 1.00× | 1,052 | 661 |
| FP16 cuBLAS | 6.041 | 1.14× [1.14–1.15] | 1,204 | 672 |
| AWQ w4g128 | 3.793 | 1.82× [1.82–1.82] | 501 | 584 |
| `Planes14` | 8.742 | 0.79× [0.79–0.79] | 250 | 428 |
| `Slot32` | 9.413 | 0.73× [0.73–0.73] | 266 | 431 |
| `Planes12x` | 9.423 | 0.73× [0.73–0.73] | 209 | 359 |
| `Golay70` v2 | 11.121 | 0.62× [0.62–0.62] | 147 | 264 |
| `Golay70` v1 | 15.705 | 0.44× [0.44–0.44] | 104 | 199 |

FP16 converts the HBM (661 → 1,052 GB/s) and our arms drop (428 → 250,
431 → 266): on the A100 they are bound by per-SM compute. The floor eats
59% of the FP16 time on the A100 against 21% on the L40S (*computed*, F4).
The worst errors are identical on both cards. The internal ordering of our
layouts holds; the scale against FP16 flips wholesale. Cross-card ×
values do not divide: two processes, two controls.

The mechanism is the clock. Both cards run pinned at their max boost,
2,520 MHz on the L40S and 1,410 on the A100, with `GpuIdle` the only event
(*measured* at 1 Hz, [g-horloges-planes12x-2026-08-23.txt](mesures/g-horloges-planes12x-2026-08-23.txt)).
The 1.787 ratio falls inside the timestamped criterion [1.60; 1.95] and
matches `nullk`'s slowdown: ×1.772 on benchmark G, ×1.781 on benchmark F4,
×1.809 on the A4 times (*measured*, [a4-a100-2026-08-31.txt](mesures/a4-a100-2026-08-31.txt)).
The occupancy counters are refused by the
platform (`ERR_NVGPUCTRPERM`, [f3-events-2026-08-19.txt](mesures/f3-events-2026-08-19.txt)).

The FP16 denominator is checked. The in-house control is 1.024 (two arms)
and 1.015 (five arms) of cuBLAS on the L40S, criterion ≤ 1.05 (*measured*,
[f1-cublasf16-2026-08-18.txt](mesures/f1-cublasf16-2026-08-18.txt)), and 1.02×
[1.02–1.02] on the ten-arm benchmark. It sits within 1.5 to 2.4% of cuBLAS; the
"vs FP16" ratios do not flatter the numerator. On the A100 the same control
is at 1.14× of cuBLAS. "Decodes at matvec speed" is an L40S/Ada result
whose domain of validity is measured on two cards; the A100 point bounds it.

## 8. The shared-memory wall

The fused path stops at 32B, on the `rot_apply` rotation (*measured*, [rot-partagee-14b-2026-08-17.txt](mesures/rot-partagee-14b-2026-08-17.txt),
[fusedrun-14b-2026-08-17.txt](mesures/fusedrun-14b-2026-08-17.txt)). A
Walsh-Hadamard transform chains log₂ m stages separated by barriers, and
CUDA has no barrier between blocks. `rot_apply` is therefore a single-block
kernel that puts the whole activation in f32 shared memory, whatever the
input dtype. The widest activation is the input to `down_proj`.

| model | `intermediate_size` | shared requested | default 49,152 B | opt-in 101,376 B |
|---|---|---|---|---|
| 4B | 9,728 | 38,912 B | passes | passes |
| 8B | 12,288 | 49,152 B, to the byte | passes | passes |
| 14B | 17,408 | 69,632 B | fails | passes |
| 32B | 25,600 | 102,400 B | fails | fails, by 1,024 B |

The L40S's three attributes are *measured* at preflight (fusedrun-14b):
`MAX_SHARED_MEMORY_PER_BLOCK` 49,152 bytes, `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`
101,376 bytes, `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` 102,400 bytes (a
per-SM budget, not per block). The opt-in is set through
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` on the function before any
launch. The guard compares against both bounds and names the one that is
crossed (`llvq_cuda::shared`, portable, tested on Mac). The first 14B job
failed cleanly on the earlier guard: exit 1 after 488 s, $0.24, no tokens.
The 32B stays refused by 1,024 bytes, the driver's reserve; a guard that
let it through would produce silent corruption. Two ways out are named,
neither designed: an f16 staging (51,200 bytes, ~14.6 stages of
half-precision accumulation) or a split into two kernels.

This wall is the same under every layout and outside the `nullk` floor,
which times 252 projections with the rotation excluded.

## 9. Measurement pitfalls

- One process, all arms interleaved, same order every round, 7 rounds with
  2 discarded. Two `Planes14` from two processes do not compare: a
  different NVRTC translation unit.
- A ratio is formed round by round, never as a quotient of two minima. On
  run K-1, 21.728 / 10.126 gives 2.19 where the round-by-round median
  gives 2.14× [2.10–2.16] (*measured*, [k1-metal-2026-08-05.txt](mesures/k1-metal-2026-08-05.txt)).
- Three invocations of the same unmodified binary give 2.029×, 2.050×,
  2.080× at identical bytes and errors (*measured*,
  [thesis-temoin-2026-08-04.txt](mesures/thesis-temoin-2026-08-04.txt)). An
  effect of a few percent cannot be settled between two invocations. We
  publish a range.
- `fusedrun` loads each arm alone: no round of the two arms coexists. The
  ratio is a quotient of medians with an envelope, and the greedy tokens
  are compared against the dense arm (divergence at token 89 on the three
  served layouts).
- `float4` returns 3.5% on LLVQ and 5.1% on FP16, so the ratio does not
  move (2.04× against 2.09×). The `float4` FP16 arm is not bit-exact
  (3.1e-8, a sum without explicit `fma`), a declared confound (*measured*, K-1).
- The bank conflict predicted by
  [portage-noyau-cuda.md](archive/portage-noyau-cuda.md) §3.2 does not
  exist on Apple: a stride of 28 floats runs 0.4% slower than 24 (*measured*,
  K-1).
- On Metal, buffers of 11 to 17 MB fit in the 48 MB SLC: the DRAM regime is
  forced cold, 4 copies of each stream rotated. A decode benchmark never writes
  the decoded weights.
- Host transcoding costs 1,464 s before the first round of a seven-arm
  benchmark (*measured*, golay70-v2), 1,468 to 1,481 s depending on the job:
  launch with `--timeout 90m`. The B2 14B job was killed at 42.5 min for 40
  requested, after its last measurement (*measured*, b2-fusedrun-plages).
- `LLVQ_NVRTC_ARCH=compute_89` by default, `compute_80` for the A100; any other form is refused.
- `LLVQ_TIME_EVENTS=1` times the device span with CUDA events, outside the
  published protocol. The host−device gap is 0.1 to 0.2%, 4 to 8 µs per
  whole round (*measured*, f3-events): the latency item is device-side,
  inter-kernel bubbles included.
- The V0 line of the `nullk` journal prints "pires erreurs 0.0e0": that arm
  has no reference, read it as "not compared".
- The "GB on card" figures are a host byte count printed by `fusedrun`,
  never `nvidia-smi`. The ÷ are legitimate; the absolute values do not
  compare with a card readout (2.60 GB displayed against 2.56 counted).

## 10. Provenance note

One object carries several numbers, one per accounting. They never compare
with each other.

| accounting | what it counts | example |
|---|---|---|
| file, `bin/seal` | payload bits / all weights, tail included | 4B: 2.1595 b/weight |
| file, `bin/smoke` | same bits / quantized weights alone | 4B: 2.1696 |
| ideal rate, `smoke` | 48 bits per block, 16-bit tail, a file never written | 4B: 2.0702, not cited for this file |
| payload | record bits / quantized weights | `Slot32` 5.3756 · `Planes14` 4.6667 · `Planes12x` 4.2029 |
| `rtbits`, `bin/matvec` | payload + one u32 base per group, byte stride | `Slot32` 5.3756 (whole model) and 5.375 (gate_proj alone) |
| kernel, `bin/thesis` and CUDA benchmark | same + f32 tail + f32 row scales | `Slot32` 5.510 · `Planes14` 4.804 · `Planes12x` 4.342 |
| inference, `fusedrun` | same, tail carried in binary16 | `Planes14` 4.729 · `Planes12x` 4.277 |
| whole model | card bytes / all parameters, embedding included | `Planes14` + q8: 5.162 (4B), 5.322 (8B), 5.106 (14B) |

Sources: fiche-4b for the file, k1c-rtbits and e1c12-aligne for payload and
`rtbits`, F2 for the kernel, b2-fusedrun-plages and g-horloges for
inference, rtbits-14b for the whole model.

A "4.804" and a "4.729" side by side are two numerators. A bits↔speed scale
aligns the bits and the speeds of one accounting and one run. Any memory
comparison with a competitor is stated in b/param over the whole model,
embedding included ([METHODE.md](METHODE.md) §2).
