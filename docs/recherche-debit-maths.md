# Throughput from the structure of Tetra

**Decoding math alone cannot take the served doors past 278, 196 and 121 tok/s at 4B, 8B and 14B (*estimated*): that is
the token with Tetra rows at DRAM speed and everything else unchanged.** Today the doors read 221.5, 142.5 and 86.3. One
lever reaches about half of that headroom. The lattice points are exact int8 vectors, so with the activation rounded to
int8 a block of 24, the block dot becomes six `dp4a`, and the served block falls from 237 to 152 SASS instructions
(*computed*). The doors then project to 248, 168 and 103 tok/s on the arithmetic, 232, 152 and 92 at the 41 % that 4E's
versions realised (*estimated*). It changes the bits, and every estimate leaves quality in place: the rounding adds
2.3e-4 of the weight error's noise power (*computed*). It would be served under the attention form's ±0.12 % perplexity
gate. At the bit, two exact levers give 234, 151 and 90 at most. In batch the step's slope is activation traffic set by
the tile shape, and the lattice does not enter it.

State as of 2026-11-08. A study at $0: SASS counted on the Mac, a numpy simulation, arithmetic. No card ran. Every
derived number comes from `ops/debit_maths.py`; the instruction counts and the time model are in the journal
[sass-tetra-int8](mesures/sass-tetra-int8-2026-11-08.txt). The route at the bit for the 4B is
[`plan-300.md`](plan-300.md); this study looks only at what the format's maths changes.

## 1. Ranked levers

Doors in tok/s, batch 1, 5-token prompt, L40S. High is the arithmetic; low keeps 41 % of the saved milliseconds, the
share 4E's versions realised ([`plan-300.md`](plan-300.md) §3). All gains and costs *estimated*.

| rank | lever | 4B | 8B | 14B | quality | memory | cost | gate |
|---|---|---|---|---|---|---|---|---|
| 1 | B1, int8 activation and `dp4a` in the Tetra rows | 248 (232) | 168 (152) | 103 (92) | not at the bit; ppl < 0.01 % expected | 0 | 1-2 weeks, about $2.5 | microbench ×1.15 over v2 alone, then paired ppl ±0.12 % |
| 2 | B2b, lane rotation at the row tail | 230 (225) | 148 (145) | 88 (87) | at the bit | 0 | 1-2 weeks, about $1 | microbench ×1.05, door identity |
| 3 | B2a, trio staging, raw `prmt`, SoA words | 226 (223) | 147 (144) | 89 (87) | at the bit | 0 | 2-4 days, about $1 | microbench ×1.03, door identity |
| | B1 and B2b together | 254 (234) | 171 (153) | 104 (93) | not at the bit | 0 | | |
| | B2a and B2b together, the exact ceiling | 234 (227) | 151 (146) | 90 (88) | at the bit | 0 | | |

In batch, aggregate tok/s on the 4B at B = 128:

| rank | lever | today | projected | quality | gate |
|---|---|---|---|---|---|
| 1 | N1, tile height scaled to the compression | 3,997 | 6,400, or 8,400 with a 5 ms fixed step | the batch kernel's own gate | slope ≤ 0.09 ms a sequence |
| 2 | N2, int8 tensor cores on exact lattice weights | | +7 % at most, only above B ≈ 72 | as B1 | |

Probability. B2a and B2b are exact by construction, and their instruction or item counts are *computed*; only the time
they buy is estimated. B1's count is *computed*, its time model is fitted on four eager arms and never checked inside
4E, and two earlier static models of 4E's rows were refuted. N1 explains the measured excess slope to the second digit
with one fitted bandwidth (§5). The batch kernel belongs to another session, which is fixing that slope now; this study
only gives its bound.

## 2. Hard ceilings

| size | door today | Tetra rows at DRAM, rest unchanged | weight floor |
|---|---|---|---|
| 4B | 221.5 | 278 | 628 |
| 8B | 142.5 | 196 | 359 |
| 14B | 86.3 | 121 | 188 |

The middle column removes the whole instruction cost of the Tetra rows: in-situ row time minus its bytes at 864 GB/s,
0.92, 1.91 and 3.30 ms (*estimated*; the 4B's 1.81 ms of Tetra rows is *computed* in [`plan-300.md`](plan-300.md) §4
from v6's stamps and the solo rates, and scaled by Tetra blocks for the larger sizes). The weight floors are *computed*
in [`plan-300.md`](plan-300.md). Everything between the two columns is attention, norms, rotations, barriers and the
head, which no property of the lattice touches.

**Issue balance.** To stream Tetra at DRAM speed an SM must finish 0.374 blocks a clock on the L40S and 0.679 on an RTX
5090 (*computed*: 864 and 1,792 GB/s, 142 SMs at 2.52 GHz and 170 at 2.41, 6.45 bytes a block with tail and scales). At
237 instructions a block that is 69 % of the L40S's issue slots and 126 % of the 5090's: on the 5090, today's block
cannot reach DRAM speed even at full issue. At 152 it is 44 % and 81 %. The eager loop issues at 53 % (§3), so on the
L40S the served block caps the stream near 77 % of DRAM and the int8 block does not cap it. Tetra's block costs 9.9
instructions a weight, where an int4 nibble costs a few (not counted here). That gap is the price of the lattice.

**Barriers.** A Qwen3 layer has four grid-wide dependencies: h before the norm, every head before o, o before the second
norm, the whole inner vector before down. 4E holds six a layer, 218 a token. Cutting to four saves 72 × 0.771 µs, 0.056
ms (*computed*, barrier cost *measured* in [step1-micro](mesures/step1-micro-2026-11-07.txt)). That is a floor.

## 3. The time model

The eager Tetra loop's time is linear in its instruction count a block (*computed*, journal sass-tetra-int8). On the
four F1 floor decoders ([f1-rang-variantes](mesures/f1-rang-variantes-2026-09-05.txt), one process), a line through the
word-only arm (50 instructions, 2.825 ms) and v3 (219, 3.875 ms) predicts the other three within +2.5 %: f1r at 499
instructions with 24 I2F, v1 at 238, v2 at 523. The slope is 6.21 µs an instruction a block over the 4B's 151.3M blocks,
which is 2.13 warp instructions a clock an SM, 53 % of issue. The loop is issue-latency bound. The count predicts the
time whatever pipe an instruction uses: f1r's 24 I2F run on a quarter-rate pipe and still sit on the line.

At the served 237 instructions, 64.3 % of the above-floor time is instruction-driven and 35.7 % is the word stream. The
projections apply that split to the in-situ Tetra rows, 1.81 ms at 4B, and scale the instruction part with the count.
This is the step that is not established: 4E runs one CTA an SM and its rows stream faster in situ than alone
([pers-rowloop](mesures/pers-rowloop-2026-11-05.txt): 379 GB/s alone).

## 4. The levers at batch 1

### B1. The dot in the code's own integers

The decoder already builds each quad of four coordinates as biased bytes, and XOR `0x80808080` turns them into the
signed int8 values; the norm reads them that way with six `dp4a`. The coordinates are bounded, |y| ≤ 10, so with the
activation rounded to int8 the block dot is six more `dp4a` into an int32 that cannot overflow (24 × 10 × 127 = 30,480).
One conversion and two multiplies fold the activation scale, the gain and the inverse norm. This removes 24 `PRMT`, 24
`FADD` and 24 `FFMA` a block: 237 to 163 instructions, 152 with B2a's raw `prmt` and SoA words (*computed*). The
activation is rounded once a site by each CTA after its rotation, in trio order, with one f32 scale a block of 24: 2,560
values, a few hundred cycles, about 0.04 ms a token over the 96 Tetra sites of the 4B (*estimated*, netted in the
table). Shared memory for the activation falls from 96 to 24 bytes a block.

Savings, high (low): 0.49 (0.20) ms at 4B, 1.05 (0.43) at 8B, 1.86 (0.76) at 14B. The gain grows with size because the
Tetra rows grow and the non-row work does not.

Quality. Rounding to int8 a block of 24 leaves a relative error of 4.4e-3 to 5.1e-3 on a matvec output, against 2.1e-4
for f16 and 5.7e-3 to 8.4e-3 for int8 a token (*computed*, Gaussian and outlier activations after a randomized
Hadamard). Tetra's own weight error is 0.333 relative, so the rounding adds 2.3e-4 of the noise power the lattice
already puts there. If perplexity moved in proportion, the 4B's 2.8 % gap to FP16 would grow by 0.0006 % (*estimated*).
The tokens change, so eager and 4E change together and the published runs are not reproduced at the bit. The gate is the
one the operator took for attention: paired WikiText-2 perplexity within ±0.12 %, then the 50-problem GSM8K pilot. The
CPU backend (`tetra_cpu`) can carry the same arithmetic and read the paired perplexity on the Mac at $0 before any card.

Cost. The eager kernel, the 4E row form and the CPU twin, with host probes and mutants: 1 to 2 weeks. Card: a microbench
arm, about $0.25, then three doors and three paired perplexities, about $2.3 (*estimated* from the step jobs in
`docs/data/jobs.csv`). Risk: the in-situ transfer of §3.

### B2a. Exact savings in the served block

Three changes, each with the same 24 values multiplied by the same activations in the same order:

- Stage the activation in trio order at stride 28. 4E's stream reads the natural staging through `TETRA48_ORDER`; the
  trio staging (`PERS_INV_ORDER`, already in `llvq_pers.cuh`) gives six aligned `LDS.128` and removes the two-way bank
  conflict [pers-sass](mesures/pers-sass-2026-11-05.txt) counted at stride 24. 237 to 230.
- Build the byte mask with `prmt.b32` in sign mode through inline PTX. `__byte_perm` masks the selector to three bits,
  which is why `f1r_v3_bytemask` spends three instructions; the raw instruction spends two. 230 to 224.
- Transcode the words at load into a u32 plane and a u16 plane. Every load is aligned and `f1r_load`'s odd/even select
  goes. The bytes are the same up to row padding. 230 to 228; all three, 222.

222 is 6.3 % under 237. Savings, high (low): 0.09 (0.04) ms at 4B, 0.19 (0.08) at 8B, 0.34 (0.14) at 14B.

### B2b. Lane rotation at the row tail

A row at d_in 2,560 holds 106 blocks. Lane l adds blocks l, l + 32 and so on, so each row takes four items a lane and
the fourth runs 10 lanes of 32: 17 % of the items are idle (*computed*). The fix keeps every logical lane's blocks and
order but changes which physical lane plays it: physical lane P runs logical lane (P - o_k) mod 32 of row k, with o_k
the row's start offset in the warp's stream. Every lane is then busy until the last row. Each logical chain is the same
FMA chain, and one `__shfl_sync` puts the 32 partials back in `warp_sum`'s tree positions. Same bits by construction.
Items fall to 0.855, 0.902 and 0.957 of today's at 4B, 8B and 14B (*computed* on each size's Tetra shapes); the 4B gains
most because 2,560 is the worst fit to 32 lanes.

Savings, high (low): 0.17 (0.07) ms at 4B, 0.24 (0.10) at 8B, 0.18 (0.07) at 14B. Risk: a lane holds two rows'
accumulators across a boundary, and register pressure already refused v7 at 128 registers
([step2-4e](mesures/step2-4e-2026-11-07.txt)).

## 5. Batch

**The slope is activation traffic.** The batch GEMM of PR #24 reloads the f32 activation for every 64-row tile. Per
sequence and step that is 4 bytes × 4.02e9 weights ÷ 64 = 0.251 GB of L2 reads at 4B (*computed*). Its slope is 0.175 ms
a sequence against AWQ's 0.066 (*computed* from the steps at B = 32 and 128 *measured* in PR pjmalandrino/llvq#24, not
merged); the 0.109 ms excess is that traffic if L2 delivers 2.3 TB/s. The general rule: a CTA tile of M rows and N
sequences reads M × w bytes of weight and N × a bytes of activation a column, so the weights dominate only when M ≥ N ×
a ÷ w. At Tetra's w = 0.27 bytes a weight that is M ≥ 7.4 N for an f16 activation and 3.7 N for int8, against 3.8 N and
1.9 N for int4 g128. A better compression demands taller tiles. At M = 256 with an f16 activation the reload falls to 31
MB a sequence, and the slope to about 0.08 ms. With today's fixed step, 9.6 ms by the same fit, that is 6,400 tok/s at B
= 128 and 2,600 at B = 32 (113 GB/s whole model a step); with a 5 ms fixed step, 8,400 and 4,200 (183 GB/s). AWQ in vLLM
reads 428 GB/s at B = 32 (*computed* from its tok/s in the same PR). All projections *estimated*. Split-K restores the
grid that tall tiles shrink and does not change the activation traffic.

**Where Tetra can lead an int4 engine.** Per step, Tetra saves (2.67 - 1.38) GB ÷ 864 GB/s = 1.49 ms of weight reads at
4B (*computed*, whole-model bytes on card), whatever B. That is the format's whole batch advantage. At B = 32, AWQ's
step is 6.23 ms; at B = 128 with 256 positions the per-sequence KV reads (0.044 ms) and MMA (0.022 ms) already cost 8.4
ms. The edge is real below B = 32 and small above 128.

**When weights stop mattering.** At the L40S's dense f16 tensor rate (362 TFLOPS, datasheet) a sequence costs 22.2 µs of
MMA a step. The MMA time passes the weight time at B ≈ 72 for Tetra and B ≈ 139 for int4 (*computed*). Above 139 both
formats are compute-bound and Tetra's decode is pure overhead.

**The decode fits under the stream.** Decoded once a step, at 152 instructions and 53 % issue, the 4B's Tetra blocks
take 0.75 ms against 1.60 ms of weight reads (*computed*). The decode is not a batch ceiling, provided each weight is
decoded once a step: one CTA covers all B sequences of its rows, which holds to B = 128.

**N2, int8 tensor cores.** The lattice values are exact int8, so `mma.m16n8k32.s8` can take them without the f16
rounding the batch kernel applies to its folded weights. A block's scale cuts the k-loop at 24, so each block pads to 32
and the int32 tile is rescaled a block: 75 % of the 733 TOPS (datasheet), 1.5 times the f16 rate. That moves B* from 72
to 109 and saves under 1 ms of a step that costs at least 15 ms at B = 128: +7 % at most. The activation rounds to int8
as in B1. Last in rank.

**B1 at small B.** The extra sequence on CUDA cores costs a second FMA chain a block: 24 `FFMA` and 96 bytes of shared
memory today, 6 `dp4a` and 24 bytes under B1. A 4E with a few rows would pay about a quarter of today's extra-row
arithmetic. Past a few rows a tensor-core kernel still wins (*estimated*).

## 6. Leads closed by computation or measurement

| lead | why it does not pay | label |
|---|---|---|
| dot in code space by lookup tables (T-MAC style) | with p known a coordinate takes 10 values; a table over 2 coordinates holds 100 entries, 50 a coordinate: 512 KB at d_in 2,560 in f32, 128 KB in int8, against 100 KB of shared memory, and half a random `LDS` a weight against a quarter `dp4a` | *computed* |
| factor the row by scale class | 2 gains × 27 shells = 54 scale values a matrix; folding them saves 2 instructions of 237 and costs 54 accumulators or a reordered sum | *computed* |
| Conway or M₂₄ symmetry, tables shared by class | in batch 1 each block meets its own activation slice, so no precomputation is shared across blocks; in batch the decode is already shared across B | by reading |
| Golay pattern by F₂ algebra | linear on all 4,096 paths, but the algebraic decode was 0.11 ms slower and the chain of three small loads cost nothing measurable | *measured*, f1-rang-variantes |
| decoder tables in shared memory | ×0.948 on the one-launch pass | *measured*, [pers-4e-v2](mesures/pers-4e-v2-2026-11-05.txt) |
| f16 `HFMA2` dot | 1.5 instructions a coordinate against 0.5 for B1, with a coarser rounding | *computed* |
| rotation folded into the residual stream | needs γ folded and a re-encode of the three files, moves quality; the Hadamard arithmetic is 2,560 × 10 adds a site and 4E already runs it in registers (`pers_wht512`), so at most part of ln1 and ln2 is saved | by reading |
| fewer grid barriers | 0.056 ms at most (§2) | *computed* |

**Speculative decoding.** Its closure binds on the verify pass, 1.95 greedy tokens for 2 rows at 4B against the 1.4 of
its reopening condition ([spec-gate](mesures/spec-gate-2026-11-07.txt)). The cost is per-row prologues, attention and
spills, not the decode, so no lattice argument moves it, and a cheaper drafter does not meet the condition either. **KV
cache.** The cache holds activations; nothing in the lattice applies, and an int4 cache moves quality. **Host
overhead.** 4E removed it; nothing to add.

## 7. Next step

One microbench job decides B1 and B2 before any port, about $0.30 (*estimated*). It times, in `pers_trows.cu`'s geometry
and bit-checked where the arm owes the bits:

- `pword`, the words read with no decode (`tetra48_pers.cu`, host-proven for the L2 prereg, never run): it measures the
  stream part of §3 in a one-launch geometry, which fixes the low bound of every lever above.
- `i8`, B1's row: the gate is ×1.15 over v2 alone, the bar plan-300 used for row levers.
- `exact`, B2a's three changes, and `rot`, B2b's rotation: gates ×1.03 and ×1.05 over v2, with v2's bits.

B1's quality gate runs first on the Mac through `tetra_cpu`, at $0. A prereg is stamped before the job (rule 2), and the
spend cap owed by `ETAT.md` §5 comes before it.

## 8. What this study does not establish

- Any time on a card for any variant: none ran.
- That the eager slope of §3 holds inside 4E.
- The card's own instruction counts: the devel ptxas counts 237 for a block the driver compiles to 243.
- B1's perplexity: the error is simulated on synthetic activations.
- The batch L2 bandwidth: 2.3 TB/s is fitted, not measured.
- Any card but the L40S, and the 5090 figures beyond the issue arithmetic.
