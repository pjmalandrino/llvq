# Throughput from the structure of Tetra

**The best lever the lattice's maths offers takes the served doors from 221.5, 142.5 and 86.3 tok/s to about 239, 157
and 95 at 4B, 8B and 14B (*estimated*, range 229-248, 148-164, 90-99).** It has three parts. The 48-bit word is
relabeled at load so the decoder indexes contiguous fields and reads the Golay patterns from two tables by
linearity. The activation is rounded to int16 a block of 24, and the dot runs on the lattice's exact int8 values
through `dp2a`. Lane rotation removes the idle lanes at the end of each row. The block falls from 237 to 145 SASS
instructions (*computed*). The rounding error is 2.0e-5 relative, ten times under the f16 rounding the batch kernel
already passed (*computed*), so it is served under the attention form's ±0.12 % perplexity gate and not at the bit.
At the bit, the relabeled decoder with lane rotation gives 231, 149 and 89. No decode can take the doors past 260,
174 and 105: that is the token with Tetra rows streaming at 671 GB/s, the best rate any of our kernels has read.

State as of 2026-11-08, after a second pass that corrected the first (§9). A study at $0: SASS counted on the Mac,
numpy simulations, exact enumeration, arithmetic. No card ran. Every derived number comes from `ops/debit_maths.py`;
counts and models are in the journals [sass-tetra-int8](mesures/sass-tetra-int8-2026-11-08.txt) and
[sass-tetra-lean](mesures/sass-tetra-lean-2026-11-08.txt). The route at the bit for the 4B is
[`plan-300.md`](plan-300.md); this study looks only at what the format's maths changes.

## 1. Ranked levers

Doors in tok/s, batch 1, 5-token prompt, L40S. The central figure assumes the Tetra words stream at 671 GB/s in
situ. The range runs from 41 % of the central saving, the share 4E's versions realised
([`plan-300.md`](plan-300.md) §3), to a stream at 864 GB/s. All gains and costs *estimated*.

| rank | lever | 4B | 8B | 14B | quality | memory | cost | gate |
|---|---|---|---|---|---|---|---|---|
| 1 | B1: relabeled decoder, int16 activation and `dp2a`, lane rotation | 239 [229-248] | 157 [148-164] | 95 [90-99] | not at the bit; rounding 2.0e-5, under f16's | 0 | 2-3 weeks, about $2.5 | microbench ×1.15 over v2 alone, then paired ppl ±0.12 % |
| 2 | B2: relabeled decoder and lane rotation | 231 [225-235] | 149 [145-152] | 89 [88-91] | at the bit | 0 | 2 weeks, about $1 | microbench ×1.08, door identity |
| | B1 with int8 in place of int16 | 238 [228-245] | 156 [148-163] | 95 [90-99] | rounding 5.0e-3 | 0 | | |
| | ceiling: Tetra rows at 671 GB/s | 260 | 174 | 105 | | | | |

In batch, aggregate tok/s on the 4B at B = 128:

| rank | lever | today | projected | quality | gate |
|---|---|---|---|---|---|
| 1 | N1, tile height scaled to the compression | 3,997 | 6,400, or 8,400 with a 5 ms fixed step | the batch kernel's own gate | slope ≤ 0.09 ms a sequence |
| 2 | N2, int8 tensor cores on exact lattice weights | | +7 % at most, only above B ≈ 72 | rounding as B1's int8 | |

Probability. B2 is exact by construction and its counts are *computed*; only the time it buys is estimated. Each of
its two halves alone saves about 0.1 ms at 4B, the size of the time model's residual (§3), so neither is resolved on
its own. B1's count is *computed* and its time rests on the same model, never checked inside 4E. N1 is consistent
with the measured slope, but one fitted bandwidth always fits one number, so its attribution is not demonstrated.
The batch kernel belongs to another session; this study only gives its bound.

## 2. Hard ceilings

| size | door today | Tetra rows at 671 GB/s | at 864 GB/s | weight floor |
|---|---|---|---|---|
| 4B | 221.5 | 260 | 278 | 628 |
| 8B | 142.5 | 174 | 192 | 359 |
| 14B | 86.3 | 105 | 116 | 188 |

The two middle columns remove the whole instruction cost of the Tetra rows and keep everything else. In-situ Tetra
rows take 1.81, 3.65 and 6.18 ms (*estimated*: the 4B's is *computed* in [`plan-300.md`](plan-300.md) §4 from v6's
stamps and solo rates; the 8B's and 14B's close each token from its door, int4 rows, head and non-row work). Their
bytes take 1.15, 2.38 and 4.11 ms at 671 GB/s. The weight floors are *computed* in [`plan-300.md`](plan-300.md).
Everything between the ceiling and the floor is attention, norms, rotations, barriers, int4 rows and the head.

**Issue balance.** To stream Tetra at DRAM speed an SM must finish 0.374 blocks a clock on the L40S and 0.679 on an
RTX 5090 (*computed*: 864 and 1,792 GB/s, 142 SMs at 2.52 GHz and 170 at 2.41, 6.45 bytes a block with tail and
scales). At 237 instructions a block that is 69 % of the L40S's issue slots and 126 % of the 5090's: on the 5090,
today's block cannot reach DRAM speed even at full issue. At 145 it is 42 % and 77 %. These are counts for sm_89;
CUDA 12.4 does not compile sm_120. Tetra's block costs 9.9 instructions a weight, where an int4 nibble costs a few
(not counted here). That gap is the price of the lattice.

**Barriers.** A Qwen3 layer has five grid-wide dependencies: h before the norm, q, k and v before attention, every
head before o, o before the second norm, the inner vector before down. 4E holds six a layer, 218 a token. Cutting to
five saves 0.029 ms (*computed*, barrier cost 0.771 µs *measured* in
[step1-micro](mesures/step1-micro-2026-11-07.txt)). That is a floor.

## 3. The time model

The eager Tetra loop's time is linear in its instruction count a block (*computed*, journal sass-tetra-int8). On the
four F1 floor decoders ([f1-rang-variantes](mesures/f1-rang-variantes-2026-09-05.txt), one process), a line through
the word-only arm (50 instructions, 2.825 ms) and v3 (219, 3.875 ms) predicts the other three within +2.5 %: f1r at
499 instructions with 24 I2F, v1 at 238, v2 at 523. The slope is 6.21 µs an instruction a block over the 4B's 151.3M
blocks, 2.13 warp instructions a clock an SM, 53 % of issue. The count predicts the time whatever pipe an instruction
uses: f1r's 24 I2F run on a quarter-rate pipe and still sit on the line.

The line is good on large differences and blind to small ones. v1 and v3 sit 19 instructions apart, the line puts
them 0.118 ms apart, and they read 0.025 ms apart. A saving under about 0.09 ms at 4B is inside that residual.

In situ, the rows cannot stream faster than their bytes. The instruction part of in-situ Tetra rows is therefore
row time minus bytes over bandwidth: 0.66 ms at 671 GB/s and 0.92 at 864 for the 4B, 36 to 51 % of row time
(*computed*, journal sass-tetra-lean). Per instruction that is ×0.72 to ×1.00 of the eager slope, which fits 4E's
higher issue rate. The projections scale that part with N - 50, N the block's count, and with the share of items
left after lane rotation.

## 4. The levers at batch 1

### The relabeled decoder (in B1 and B2)

The card may hold any permutation of a block's 48 bits: the bytes and the point do not change. The relabeled word
puts r above i1, so the first rank row's index is one 12-bit field, and gathers the fields the decoder shifts out.
The Golay map from (s8, b1, b2, b3) to the three pattern bytes is linear (*computed*, exhaustive: 4,096 distinct
patterns, weights 0, 8, 12, 16, 24), so two tables of 64 and 128 entries replace the chain prefix, branch, suffix.
The matrix's two gains sit in registers. With the trio staging and the raw `prmt` of the first pass, the exact block
falls from 237 to 207 instructions, the int8 block from 152 to 140 (*computed*).

The F1 measurement of 2026-09-05 does not close this: its algebraic Golay decoder cost twelve masked XORs and 24 more
instructions, and the time followed the count. Two table reads and one XOR cost fewer instructions than the chain.

### The dot in the lattice's integers (B1)

The decoder builds each quad of four coordinates as biased bytes, and XOR `0x80808080` turns them into the signed int8
values; the norm already reads them that way with `dp4a`. With |y| ≤ 10 the dot is exact integer arithmetic once the
activation is an integer.

- int16 activation, `dp2a`: twelve instructions a block, an int32 that cannot overflow (24 × 10 × 32,767 = 7.9M),
  rounding error 2.0e-5 relative on a matvec output, 3.6e-9 of the noise power the lattice already puts there
  (*computed*). The block counts 145.
- int8 activation, `dp4a`: six instructions, 5.0e-3 relative, 2.3e-4 of the noise power. The block counts 140.

int16 costs five instructions and about one tok/s at 4B, and buys a rounding ten times finer than the f16 activation
the batch kernel already serves under its gate. It is the form to build. Either way the tokens change, eager and 4E
change together, and the published runs are not reproduced at the bit. The gate is the attention form's: paired
WikiText-2 perplexity within ±0.12 %, then the 50-problem GSM8K pilot. The CPU backend (`tetra_cpu`) can carry the
same arithmetic and read the paired perplexity on the Mac at $0 first. Rounding the activation costs about 0.4 µs a
site, 0.04 ms a token at 4B (*estimated*, netted in the table).

### Lane rotation (in B1 and B2)

A row at d_in 2,560 holds 106 blocks. Lane l adds blocks l, l + 32 and so on, so each row takes four items a lane and
the fourth runs 10 lanes of 32: 17 % of the items are idle (*computed*). Physical lane P can play logical lane
(P - o_k) mod 32 of row k, with o_k the row's start offset in the warp's stream. Each logical chain keeps its blocks
and its order, and one `__shfl_sync` puts the 32 partials back in `warp_sum`'s tree positions: same bits by
construction. Items fall to 0.855, 0.902 and 0.957 of today's at 4B, 8B and 14B (*computed* on each size's Tetra
shapes). Risk: a lane holds two rows' accumulators across a boundary, and register pressure already refused v7 at
128 registers ([step2-4e](mesures/step2-4e-2026-11-07.txt)).

### B3, the decoder's floor on paper

Enumerating what a block must do (two word loads, three row gathers, two Golay reads, about ten field operations,
four instructions a quad, twelve `dp2a`, six `dp4a` for the norm, the fold) gives about 90 instructions
(*estimated*), against 145 compiled. Closing that gap would save another 0.18 to 0.25 ms at 4B and run into the 671
GB/s ceiling. It is hand scheduling with no structure behind it, and the model cannot resolve its steps one by one.
Not ranked.

## 5. Batch

**The slope is plausibly activation traffic.** The batch GEMM of PR pjmalandrino/llvq#24 (not merged) reloads the
f32 activation for every 64-row tile: 4 bytes × 4.02e9 weights ÷ 64 = 0.251 GB of L2 reads a sequence and step at 4B
(*computed*). Its slope is 0.175 ms a sequence against AWQ's 0.066 (*computed* from steps *measured* at B = 32 and
128 in that PR). The 0.109 ms excess equals that traffic if L2 delivers 2.3 TB/s, a fitted value, not a measured
one. The general rule holds without it: a tile of M rows and N sequences reads M × w bytes of weight and N × a bytes
of activation a column, so the weights dominate only when M ≥ N × a ÷ w. At Tetra's w = 0.27 bytes a weight that is
M ≥ 7.4 N for a 16-bit activation and 3.7 N for int8, against 3.8 N and 1.9 N for int4 g128. At M = 256 with a 16-bit
activation the reload falls to 31 MB a sequence and the slope to about 0.08 ms. With today's fixed step, 9.6 ms by
the same fit, that is 6,400 tok/s at B = 128 and 2,600 at B = 32 (113 GB/s whole model a step); with a 5 ms fixed
step, 8,400 and 4,200 (183 GB/s). AWQ in vLLM reads 428 GB/s at B = 32 (*computed* from its tok/s in the same PR).
Split-K restores the grid that tall tiles shrink and does not change the activation traffic.

**Where Tetra can lead an int4 engine.** Per step, Tetra saves (2.67 - 1.38) GB ÷ 864 GB/s = 1.49 ms of weight reads
at 4B (*computed*, whole-model bytes on card), whatever B. At B = 32, AWQ's step is 6.23 ms; at B = 128 with 256
positions the per-sequence KV reads (0.044 ms) and MMA (0.022 ms) already cost 8.4 ms. The edge is real below B = 32
and small above 128.

**When weights stop mattering.** At the L40S's dense f16 tensor rate (362 TFLOPS, datasheet) a sequence costs 22.2 µs
of MMA a step. The MMA time passes the weight time at B ≈ 72 for Tetra and B ≈ 139 for int4 (*computed*). Above 139
both formats are compute-bound and Tetra's decode is pure overhead.

**The decode fits under the stream.** Decoded once a step at 152 instructions and 53 % issue, the 4B's Tetra blocks
take 0.75 ms against 1.60 ms of weight reads (*computed*), less with the relabeled decoder. The decode is not a batch
ceiling, provided each weight is decoded once a step: one CTA covers all B sequences of its rows, which holds to B =
128.

**N2, int8 tensor cores.** The lattice values are exact int8, so `mma.m16n8k32.s8` takes them with no rounding of the
weights. A block's scale cuts the k-loop at 24, so each block pads to 32 and the int32 tile is rescaled a block: 75 %
of the 733 TOPS (datasheet), 1.5 times the f16 rate. That moves B* from 72 to 109 and saves under 1 ms of a step that
costs at least 15 ms at B = 128: +7 % at most.

**Small B on CUDA cores.** A second sequence costs a second dot a block: today 24 `FFMA` and 96 bytes of shared
memory, under B1 twelve `dp2a` and 48 bytes. Past a few rows a tensor-core kernel still wins (*estimated*).

## 6. What the lattice's constructions offer the decode

The question for each construction: does it give a path index → vector → dot shorter than the trio's?

- **Turyn and the MOG.** Λ₂₄ contains the points (a + x, b + x, a + b + x) over E8. The trio's three octads are the
  MOG's three bricks, so the Tetra decoder already works in that split. Rewritten in it, the dot is ⟨a, X1 + X3⟩ +
  ⟨b, X2 + X3⟩ + ⟨x, X1 + X2 + X3⟩: the sums of activation sections are shared by every row, but it is still 24
  products. It saves work only through tables over the E8 components (next item), and Tetra's labels are not those
  components: a re-encode.
- **Tables of dots, AQLM and T-MAC style.** A table of T entries over an 8-coordinate section costs 8 T products an
  activation and saves 8 a row. It pays when T < d_out. A Tetra section is fixed by its parity, its Golay byte (128
  values each, *computed*) and its rank row (4,096), up to 1,048,576 entries, 60 times the widest d_out of 17,408.
  E8 components of a 2-bit-a-dimension code need about 2^16 each, still above d_out. AQLM's 2^8 codebooks pay because
  they are small and shared.
- **A Walsh-Hadamard transform over the 12 information bits.** It gives ⟨c, x⟩ for all 4,096 codewords in 49,152
  additions a block of activation. It does not help: y = p·1 + 2c + 4k, and k depends on c and the rank in each
  coordinate (odd blocks y = (-1)^c ⊙ v₁(ρ); even blocks y = 4κ(ρ) + c ⊙ (2 - 8κ(ρ))). No term is Golay-only, so
  the per-coordinate work stays, and the byte tables already fold c into the values at no cost.
- **Sublattices.** Splitting y = 2z + t over Λ₂₄/2Λ₂₄ (2^24 classes), or over D24 or E8³, leaves 24 products, and no
  block shares its coset or its activation slice with another.
- **Hexacode, Conway-Sloane, Vardy-Be'ery, Amrani-Be'ery, Forney.** These are nearest-point decoders, hundreds to
  thousands of operations a block: the encoder's problem. Index to point is already 24 table-driven values here.
  They could speed the encoder, which this study does not cover.
- **QuIP#'s E8P.** A 256-entry table of absolute patterns, sign bits and a parity: the same shape as Tetra's rank rows,
  Golay byte and p. Storing packed values in place of ranks needs 16 bytes a row and parity, 128 KiB, and saves no
  instruction over `prmt`.
- **QTIP.** Its trellis code computes a weight from a bit window in two or three instructions with no table reads.
  That is a different code. Taking it means re-encoding all three files: the 8B's and 14B's sealing chains cost
  $10.19 and $32.17 ([`HISTORIQUE.md`](HISTORIQUE.md)), and quality moves.
- **MMA int4.** Tetra values need five signed bits (|y| ≤ 10) and Ampere and Ada have no s4 × s8 MMA. Closed.
- **Tensor cores at batch 1.** The dot is already twelve `dp2a`; the decode is what costs. Closed.
- **Bit-slice.** The values come from per-block table reads, which would have to be transposed into bit planes, and
  the activation is not binary. Closed.

## 7. Leads closed

| lead | why it does not pay | label |
|---|---|---|
| tables of dots (T-MAC, AQLM) | a section's codebook is up to 1,048,576 entries, 60× the widest d_out | *computed* |
| Walsh-Hadamard over the Golay information bits | no Golay-only term in y·x | by algebra |
| Turyn, E8³, D24, Λ₂₄/2Λ₂₄ components | 24 products remain; tables need components ≤ d_out | *computed* |
| hexacode and nearest-point decoders | encoder side | by reading |
| factor the row by scale class | 54 scale values a matrix; 2 instructions of 237 for 54 accumulators | *computed* |
| decoder tables in shared memory | ×0.948 on the one-launch pass | *measured*, [pers-4e-v2](mesures/pers-4e-v2-2026-11-05.txt) |
| f16 `HFMA2` dot | 1.5 instructions a coordinate against 0.5 for `dp2a` | *computed* |
| MMA int4, bit-slice, tensor cores at batch 1 | §6 | *computed* |
| QTIP's computed code | re-encode of the three files, quality to measure | by reading |
| rotation folded into the residual stream | needs γ folded and a re-encode; 4E already runs the Hadamard in registers | by reading |
| fewer grid barriers | 0.029 ms at most | *computed* |

**Speculative decoding.** Its closure binds on the verify pass, 1.95 greedy tokens for 2 rows at 4B against the 1.4 of
its reopening condition ([spec-gate](mesures/spec-gate-2026-11-07.txt)). The cost is per-row prologues, attention and
spills, not the decode, so no lattice argument moves it. **KV cache.** It holds activations; nothing in the lattice
applies. **Host overhead.** 4E removed it.

## 8. Next step

One microbench job decides B1 and B2 before any port, about $0.30 (*estimated*). In `pers_trows.cu`'s geometry,
bit-checked where an arm owes the bits:

- `pword`, the words read with no decode (`tetra48_pers.cu`, host-proven for the L2 prereg, never run): it measures
  the in-situ stream rate that sets every central figure above.
- `lean`, B2's relabeled decoder; `rot`, its lane rotation; `lean_i16`, B1's row. Gates ×1.05, ×1.05 and ×1.15 over
  v2 alone.

B1's quality gate runs first on the Mac through `tetra_cpu`, at $0. A prereg is stamped before the job (rule 2), and
the spend cap owed by `ETAT.md` §5 comes before it.

## 9. Second pass

The second pass reread every first-pass figure against its own inputs, counted three new variants and enumerated the
constructions of Λ₂₄ (§6).

What fell:

- **The in-situ share.** The first pass gave 64.3 % of in-situ row time to instructions, the eager share. That implied
  Tetra words streaming at about 1,200 GB/s at 14B, above DRAM. Bounded by the stream, the share is 36 to 51 % at
  4B. B1 with the first pass's 152-instruction block drops from 248 / 168 / 103 to 235 / 154 / 94 at the central
  figure.
- **The ceilings.** 278 / 196 / 121 assumed the DRAM peak and the scaled row times. At 671 GB/s and closed tokens they
  are 260 / 174 / 105.
- **Small exact savings.** The first pass's B2a (237 to 222) and lane rotation alone each save about 0.1 ms at 4B,
  inside the model's 0.09 ms residual. They are not resolved alone.
- **The barrier floor.** Four dependencies a layer was wrong; there are five, and the saving is 0.029 ms.
- **N1's proof.** "Explains the slope to the second digit" was one fitted bandwidth for one number: consistent, not
  shown.

What holds: every instruction count, the linear model on large differences, the int8 error, the batch arithmetic
(B*, the 1.49 ms edge, M ≥ 7.4 N), the issue balance as arithmetic, and the 8B and 14B row times within 5 %.

What is new: the relabeled decoder (207 exact, 140 int8), the int16 dot through `dp2a` (rounding 2.0e-5, 145), and
the lattice constructions read on the decode side, all closed (§6, §7).

The ranking moves. B1 becomes the relabeled decoder with the int16 dot and lane rotation, 239 / 157 / 95 central,
and int16 replaces int8 because its rounding sits under f16's for five instructions. B2 becomes the relabeled decoder
with lane rotation, 231 / 149 / 89. Batch is unchanged in rank, with N1's attribution downgraded.

## 10. What this study does not establish

- Any time on a card for any variant: none ran.
- That Tetra words stream at 671 GB/s in situ; `pword` measures it.
- The card's own instruction counts: the devel ptxas counts 237 for a block the driver compiles to 243.
- B1's perplexity: the errors are simulated on synthetic activations.
- The batch L2 bandwidth: 2.3 TB/s is fitted, not measured.
- Any card but the L40S, and anything on the 5090 beyond the issue arithmetic.
