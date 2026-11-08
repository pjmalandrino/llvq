# How LLVQ works: the maths, the lattice, the kernel

This is the one explanation of the project. It goes from the idea to the served kernel, and it is up to date on
2026-11-07. Numbers carry their label and journal once. The served state and every other headline number live in
[`ETAT.md`](ETAT.md).

## 1. Why bits per weight

Generating one token at batch 1 reads every weight of the model once. The arithmetic is cheap and the memory is not,
so the speed of a token is set by the bytes the card has to stream. Fewer bits per weight means a bigger model fits
on the same card, and each token reads less.

FP16 stores 16 bits a weight. Good 4-bit methods such as AWQ store about 4.5 to 6 bits a parameter once their scales
and tables are counted. LLVQ aims at 2 bits for the matrices, and the sealed Qwen3 files land at about 2.7 bits a
parameter for the whole model, embedding included.

At 2 bits a weight, a scalar grid has four levels. A codebook over 24 weights at once spends the same 48 bits far
better, because it can put its points where blocks of weights actually fall.

## 2. Vector quantization, and why a lattice

Scalar quantization rounds each number on its own grid. Vector quantization rounds a whole block of numbers to the
nearest point of a codebook. A block of 24 weights at 2 bits a weight gets 48 bits, so the codebook may hold 2⁴⁸
points. The question is where to put them.

Two things make a codebook good. Its points must cover space evenly, so that the rounding error is small. And it must
be usable: you need to find the nearest point fast when encoding, and you need to rebuild a point fast from its index
when decoding. A learned codebook of 2⁴⁸ entries fails the second test, since it cannot even be stored.

A lattice solves both. A lattice is a regular grid of points: every integer combination of a few basis vectors. Its
points are defined by rules, so nothing is stored. Some lattices pack space far better than the plain integer grid.
In 24 dimensions the best one known, and proven optimal, is the Leech lattice Λ₂₄. That is the LLVQ paper's choice
([arXiv:2603.11021](https://arxiv.org/abs/2603.11021), transcribed in [`llvq-paper-notes.md`](llvq-paper-notes.md)).

LLVQ also splits each block into a direction and a length, the shape-gain split. The lattice point gives the
direction. One extra bit picks the length from two values fitted per matrix.

## 3. The Golay code

Λ₂₄ is built from the extended binary Golay code [24, 12, 8]. It is a set of 4,096 binary words of length 24,
defined by 12 information bits. Any two words differ in at least 8 positions. Its weight distribution is
1 + 759·x⁸ + 2576·x¹² + 759·x¹⁶ + x²⁴. The 759 words of weight 8 are called octads, and they come back in Tetra.

The code lives in [`llvq-core/src/golay.rs`](../llvq-core/src/golay.rs). Its weight distribution, like every
constant in this section and the next, is checked by exhaustive enumeration in
[`llvq-core/tests/g1_invariants.rs`](../llvq-core/tests/g1_invariants.rs).

## 4. The Leech lattice in integer coordinates

We work in the scaled copy √8·Λ₂₄, which sits inside the integer grid Z²⁴. A vector x of 24 integers is a lattice
point when three conditions hold, and the code tests exactly these three
([`Leech::contains`](../llvq-core/src/leech.rs)):

1. All coordinates have the same parity p, all even or all odd.
2. The bits of (xᵢ − p)/2 mod 2 form a Golay codeword.
3. The sum of the coordinates is 0 mod 8 when p = 0, and 4 mod 8 when p = 1.

So every lattice point can be written

```text
x[i] = p + 2·c[i] + 4·k[i]      p in {0,1}, c a Golay codeword, k integers with Σk ≡ p (mod 2)
```

Conditions 1 and 2 are built into that form, and condition 3 becomes one parity bit on the sum of the k.

**Shells.** The points with the same norm form a shell. Shell m holds the points with ‖x‖² = 16m. The first shell,
m = 2, has 196,560 points, the kissing number of Λ₂₄. The next ones hold 16,773,120, then 398,034,000, then
4,629,381,120 points. All shells up to m = 12 together hold 111,043,117,458,000 points, which fits in 47 bits. With
one gain bit, that is the 48-bit code of a block.

**Indexing.** The first LLVQ format (v1) stores the rank of the point in that ball of 1.1·10¹⁴ points. That is
optimal in bits. It is also impossible to decode inside a kernel: you would need a table of 10¹⁴ entries, or redo a
long rank computation for every block.

## 5. From a weight matrix to blocks

Three steps prepare the weights. Tetra and v1 share all three.

**Rotation.** Real weight rows have outlier channels, and a lattice quantizer wants inputs that look isotropic. Each
matrix is multiplied by an orthogonal matrix Q built from random signs and a Walsh-Hadamard transform. Since
W·x = (W·Qᵀ)(Q·x), the file stores W·Qᵀ and the runtime rotates the activation x on the fly. Rotating the wrong way
gives finite, plausible and wrong numbers, with no error raised.

**GPTQ.** The rows are quantized one block of 24 columns at a time. After each block, the error is pushed onto the
columns not yet quantized, weighted by a Hessian collected from calibration text
([`llvq-quant/src/gptq.rs`](../llvq-quant/src/gptq.rs)). The sealed files are calibrated on DCLM-edu, the paper's set.

**Shape and gain.** A block is rebuilt from its lattice point y by one formula, shared by the encoder and the
decoder:

```text
w[j] = y[j] · centroids[g] · rscale / √(16·m)      with m = ‖y‖² / 16
```

`g` is the gain bit, `centroids` holds the two lengths of the matrix, and `rscale` is a per-row scale. The point
y = 0 rebuilds as zeros. When a row length is not a multiple of 24, its last few weights stay outside the lattice (the
"tail", f32 on disk, f16 on the card): 2,560 is 106 blocks plus 16.

**Row scales.** After encoding, the per-row scales are trained against the model's loss with the codes frozen
(`ops/llvqtune`). This costs zero bits and gained 3.15, 3.29 and 1.67 MMLU points at 4B, 8B and 14B (*measured*,
[dclm-rowscales](mesures/dclm-rowscales-2026-09-20.txt), [-8b](mesures/dclm-8b-rowscales-2026-09-21.txt),
[-14b](mesures/dclm-14b-rowscales-2026-09-22.txt)).

## 6. Planes14: decoding by unfolding

Our first served layout dodged the decoding problem. At load time each 47-bit rank is turned into a record the GPU can
read with shifts and masks: a 9-bit class, the gain bit, 24 sign bits and three 24-bit planes that pick each
coordinate's level from a 384-entry constant table. That is 112 bits, 14 bytes a block.

Decoding is then almost free, and that is the trouble: 2 bits of information cost 4.804 bits a weight in VRAM,
paid at memory speed (*computed*, [`format-noyau.md`](format-noyau.md) §3). Planes14 is still the layout of the paper 1
file ([`fiche-4b.md`](fiche-4b.md)).

## 7. Tetra: a code you can read where it is stored

Tetra keeps the lattice and the 48 bits and changes which 2⁴⁷ points get a name, so that the name can be read
directly. It came in on 2026-09-06. The code is in [`llvq-search/src/tetra/`](../llvq-search/src/tetra/).

### The trio and the 64-state trellis

Start from x = p + 2c + 4k. The Golay codeword c can be read as a path through a trellis: cut the 24 positions into
sections and track which partial words are still possible at each cut. In the natural order a cut carries 1,024
states, too many. Reorder the positions as three disjoint octads, a "trio" found among the 759, and each cut carries
only 64 states. The codeword becomes three bytes, one per section of 8 positions, and the path through the trellis is
exactly the 12 information bits of the code.

### One table of rank vectors

Inside a section, each coordinate is x[j] = o[j] + 4·k[j], where o[j] = p + 2c[j] is in {0, 1, 2, 3}. For each residue
o the possible values are listed outward from zero:

```text
o = 0 :   0,  +4,  -4,  +8,  -8, +12, -12, +16
o = 2 :  +2,  -2,  +6,  -6, +10, -10, +14, -14
o = 1 :  +1,  -3,  +5,  -7,  +9, -11, +13, -15
o = 3 :  -1,  +3,  -5,  +7,  -9, +11, -13, +15
```

A section is then stored as 8 ranks into those lists, one nibble each. The ranks are sorted by cost Σ(2ρ+1)², and the
table keeps the best 2,048 rank vectors in each of two parity classes. That is 4,096 rows of u32, 16 KiB, shared by
every block of every matrix. No rank exceeds 4, so |x[j]| ≤ 10 everywhere.

The parity of Σk is a function of the ranks alone, and a Golay codeword meets each octad of the trio in an even number
of positions. That is what lets one table serve every section and state. The Σk constraint is threaded through the
word as one bit: section 1 declares its class r, section 2 lands in class δ, and section 3 is forced into class
p ⊕ r ⊕ δ.

### The 48-bit word

| field | p | r | s8 | b1 | i1 | b2 | i2 | b3 | i3 | g |
|---|---|---|---|---|---|---|---|---|---|---|
| bits | 1 | 1 | 6 | 1 | 11 | 4 | 11 | 1 | 11 | 1 |

`p` is the block parity and `r` section 1's class. `s8`, `b1`, `b2`, `b3` are the 12 bits of the Golay path. `i1`,
`i2`, `i3` are rows of the rank table. `g` is the gain bit, which the format never decodes. The fields tile bits 0 to
47 with no gap.

Every 48-bit word is a valid name. Decoding is a bijection from 2⁴⁷ labels onto 2⁴⁷ lattice points, so a loader never
has to check an index for being out of range.

### What Tetra gives up

Tetra names a different set of points than the ball: 1.4·10¹⁴ points, reaching shells up to m = 26. Its encoder is
no longer an exact nearest-neighbour search. It solves each section for a few candidate roundings, then takes the best
path through the trellis over both parities, in about 300 µs a block on one core. On Gaussian test blocks it keeps
88.89 % of the signal against 92.00 % for the ball, at 2 bits a dimension on both sides. On the real 4B the result went
the other way: perplexity 4.64 % lower than Planes14 (*measured*, [tetra-4b](mesures/tetra-4b-2026-09-06.txt)).

### What the card holds

The word on disk is the word in VRAM. The loader only flips the bit order to little-endian and pads each row to a u32
boundary, because a 6-byte record is not 4-aligned. Beside the stream, five constant tables hold the whole decoder in
18,816 bytes: the 4,096 rank rows, 1,024 middle edges, 128 prefix bytes, 128 suffix bytes and 32 inverse norms.

Tetra reads **2.150 kernel bits a weight** in VRAM against 4.804 for Planes14, for the same 48 bits a block
(*computed*, [`data/echelle-formats.csv`](data/echelle-formats.csv)). The 0.150 above 2.000 is the f32 tail and the
row scales.

## 8. What a sealed file holds

A sealed file is not pure Tetra. On the 4B:

- 168 projections in Tetra.
- 84 projections in int4 with groups of 128: every `v_proj`, every `o_proj`, and `down_proj` of layers 12 to 23. These
  are the matrices that lost the most in Tetra.
- The embedding and head tables in int4 with groups of 64, which pays for the int4 projections.
- Norms in f16, `config.json` and `tokenizer.json`.

That reads 2.73 bits a parameter for the whole model, every table counted (*computed* by `rtbits`). The 8B and 14B
place their int4 matrices a little differently, and read 2.70 and 2.73. Always compare memory on the whole model;
the kernel figure of 2.150 counts only the Tetra matrices.

## 9. The fused kernel: decode and multiply in one launch

The engineering idea of the project is to never build a dense matrix. The kernel reads the 6-byte codes, rebuilds the
24 values in registers, multiplies them by the activation and throws them away, all in one pass.

The served Tetra matvec is `tv_tetra48_h`
([`llvq-cuda/kernels/llvq_tetra48.cuh`](../llvq-cuda/kernels/llvq_tetra48.cuh)):

- One warp computes one output row. Lane l takes blocks l, l+32, l+64, and so on, in increasing order.
- The rotated activation is staged in shared memory, a tile of blocks at a time. `LLVQ_TILE_BLOCKS` picks the tile:
  64 on the L40S, which gave Tetra 17.6 % for free because a smaller tile leaves L1 to the decoder tables
  (*measured*, [banc-252](mesures/banc-252-2026-10-06.txt)).
- A block decodes in a fixed order: the Golay bytes from the edge tables, the three rank rows, then the values. Values
  are bytes pulled from tables held in registers by the `prmt` instruction, four lookups at a time. The norm comes
  from the same bytes through six `dp4a`. The conversion to float avoids the int-to-float unit with a bit trick. About
  141 instructions a block, 40 registers, no local memory.
- The lanes are summed with `warp_sum`, then the epilogue applies the gain centroid, the inverse norm, the row scale
  and the f32 tail.

The int4 matrices have their own matvec in the same unit. The rotation of the activation is its own kernel,
`rot_apply`, shared by the projections that read the same input (`LLVQ_ROT_SHARE=1`: q, k, v share one, gate and up
another).

On the 252 projections of the 4B, one pass takes 3.734 ms against 3.263 for AWQ and 11.022 for FP16 on an L40S
(*measured*, banc-252). Half of AWQ's bytes is not half its time: our decode costs arithmetic that AWQ does not pay.
On an A100 the advantage over FP16 shrinks to 1.11× against our FP16 kernel, and cuBLAS is faster still
(*measured*, [a100](mesures/a100-2026-10-06.txt)).

## 10. 4E: one launch a token

With the fused kernel in place, the eager path still launches a kernel per matrix, plus norms, rotations and attention.
Over the 4B's 252 matrices alone, one launch takes 2.394 ms against 3.009 for the 144 launches of the fused path
(*measured*, [pers-l1](mesures/pers-l1-2026-11-04.txt)). 4E removes the launches. It runs the whole token, every
layer and the head, in one cooperative CUDA launch ([`llvq-cuda/kernels/llvq_pers.cuh`](../llvq-cuda/kernels/llvq_pers.cuh)).

**How it runs.** The grid has one block (CTA) per SM, 142 on the L40S, each with 512 threads. A token is a fixed
sequence of phases: norm and rotation, q/k/v rows, RoPE and KV append, attention, o rows, norm and rotation, gate/up
rows, the down input, down rows, then the head and its argmax. Between phases every CTA waits at a grid barrier, 218
a token. CTAs claim rows from a shared counter.

**Bit identity.** 4E must return exactly the eager path's tokens. A Tetra row in 4E is the served row: same lanes, same
block order, same 24 products in the same order, same epilogue. The three places where a compiler could fuse a
multiply-add are written out by hand, so no compiler can move them. Every 4E version is checked against the eager path
on the card before it is timed, and every served door checks it again.

**Limits.** 4E decodes batch 1 and greedy only. It reads the three sealed shapes and has no Metal version.

### The versions

| version | what changed | tok/s | journal |
|---|---|---|---|
| v1 | the first one-launch token | ×0.993 of eager | [pers-4e-r2](mesures/pers-4e-r2-2026-11-05.txt) |
| v2 | attention on the split-KV kernel's mapping, prefetch in the row loop | 178.7 bench | [pers-4e-v2](mesures/pers-4e-v2-2026-11-05.txt) |
| v3 | 512 threads a CTA, rows restructured | 179.2 bench | [pers-4e-v3](mesures/pers-4e-v3-2026-11-05.txt) |
| v4 | v2's row loop inside v3's structure | 178.0 door, served from 2026-11-05 | [served-v4](mesures/served-v4-2026-11-05.txt) |
| v5 | last rows spread over the warps | ×1.001, refused | [shaker-4e](mesures/shaker-4e-2026-11-05.txt) |
| v6 | both norms and the down input in one round trip a thread | 197.2 door | [served-v6](mesures/served-v6-2026-11-07.txt) |
| v7 | constants copied into shared memory during barrier waits | does not install, spills | [step2-4e](mesures/step2-4e-2026-11-07.txt) |
| v8 | v7's staging, plus a 3-stage `cp.async` ring in attention | 227.9 bench | step2-4e |
| v9 | v8, with the int4 rows read from tiled words | 221.5 door, served since 2026-11-07 | [served-v9](mesures/served-v9-2026-11-07.txt) |

All *measured* on the 4B, one L40S, batch 1, after a 5-token prompt. "Bench" is the measurement form inside `fusedrun`;
"door" is the served config end to end, prompt included, which reads about 7 % lower.

**v9 in one paragraph.** An int4 row reads 4 words per lane per tile of 128 words. v9 lays the words out at load so
that those 4 words sit next to each other, at 128t + 4l + s, and one 128-bit load fetches them. The products and their
order do not change, only the addresses. The eager path reads the same tiled words through `tv_q4_ht` and `tv_q4_hht`
([`llvq-llm/kernels/tv_q4_tile.cu`](../llvq-llm/kernels/tv_q4_tile.cu)), so the card holds the int4 bytes once. Each
load checks the tiled reader against the file-order one on the card before serving.

At the door, v9 decodes 221.5, 142.5 and 86.3 tok/s at 4B, 8B and 14B, 1.72×, 1.38× and 1.38× over the eager path
with the same tokens (*measured*, served-v9, [-8b](mesures/served-v9-8b-2026-11-07.txt),
[-14b](mesures/served-v9-14b-2026-11-07.txt)).

**Where a token goes.** The weights alone set a floor of 1.59 ms a token on the 4B, 628 tok/s (*computed*, 1.377 GB
at 864 GB/s). v9's token takes 4.19 ms in the bench. In v6 the rows streamed at about 500 GB/s; the rest of the token was the
norms and rotations, attention's fixed cost at short context, and CTAs waiting at barriers. The study in
[`plan-300.md`](plan-300.md) puts the ceiling at about 235 to 258 tok/s at the door while staying bit for bit
(*estimated*). Going further means speculative decoding or batching.

## 11. Attention

Three attention forms exist, picked by the `attention` field of a config. `expanded` copies each key and value head to
its query heads, as paper 2 ran it. `grouped` reads each cached head once. `kernel` sends decode rows to our split-KV
kernel `attn_dec.cu`, which 4E uses. The served configs use `kernel`. It is not bit for bit with `expanded`, because
the sums run in a different order, so it is served under a tolerance: paired WikiText-2 perplexity within ±0.12 % of
the reference form. Every size passed it ([`ETAT.md`](ETAT.md) §4).

## 12. On the Mac

Metal has its own Tetra kernels in MSL ([`llvq-metal`](../llvq-metal/) and `llvq-llm`). The sealed 4B runs fused
through them with the same 256 tokens as the dense reconstruction (*measured*,
[metal-nuit](mesures/metal-nuit-2026-11-06.txt)). The served Mac path is different: the MLX door (`ops/llvq-mlx`) runs the
sealed 4B inside mlx-lm's Qwen3 with our Tetra Metal kernels. It is not bit for bit with the Rust engine, and it is
served under the same ±0.12 % perplexity tolerance (*measured*, [mlx-pooled](mesures/mlx-pooled-2026-11-07.txt)).

## 13. Where to read the code

| what | where |
|---|---|
| Golay code, Λ₂₄, shells, theta series | `llvq-core/src/{golay,leech}.rs` |
| exact search and the v1 index | `llvq-search/src/` |
| Tetra: trio, word, rank table, trellis, encoder | `llvq-search/src/tetra/` |
| GPTQ, block loop, reconstruction | `llvq-quant/src/{gptq,quantizer}.rs` |
| the file format and the VRAM transcoding | `llvq-artifact/src/`, `tetra48.rs` |
| the served Tetra decode | `llvq-cuda/kernels/{llvq_f1rank_v3,llvq_tetra48}.cuh` |
| 4E | `llvq-cuda/kernels/llvq_pers.cuh`, `llvq-cuda/src/pers.rs`, `llvq-llm/src/persist4e_cuda.rs` |
| served configs and their checks | `configs/`, `llvq-llm/src/served.rs` |
| the layouts, accounting and measurement traps | [`format-noyau.md`](format-noyau.md) |
