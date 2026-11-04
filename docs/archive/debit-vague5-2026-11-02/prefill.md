# Prefill and time to first token: deep dive of lead "prefill"

Wave 5, 2026-11-02, revision 2 after the adversarial review (§10 lists the changes). Read-only on the repository: no
file changed, no build, no run, $0. Object, unless named: sealed 4B, L40S, batch 1, grouped config. Labels: *measured*
(named journal or primary source), *computed* (arithmetic on measured or code inputs), *estimated* (judgment, or
arithmetic with one estimated input). The arithmetic is in `wave5-deep/calc/ttft_model.py` and
`calc/paired_dense_vs_fused.py`.

## 1. Verdict

The GEMM route takes the first token of an 8,192-token prompt from 35.02 s (*measured*, served-grouped-2026-11-02) to
about 2.85-3.4 s (*estimated*). The exact arithmetic C-dense lands at 3.05-3.40 s, the rotated-basis C-rot16 at
2.85-3.10 s. A prefill attention kernel on top takes it to 0.45-1.2 s (*estimated*). An 8,192 + 512 request then falls
from 41.7 s to 9.5-10.1 s, then 7.1-7.9 s, where its 6.7 s of decode become the wall (*computed* for the decode from
served-grouped, *estimated* for the rest).

Attention is most of what the route leaves: 2.2-2.55 s at 8k today (*estimated*). Three dense windows timed in one
job pin it (*measured*, ppl-scelles-2026-10-04). The 8B has the 4B's attention geometry and runs 0.155 s longer a
window, which bounds the 4B's linear work at 0.155-0.25 s per 4,096 rows. The 14B window agrees.

Three variants cut the score peak with no T8, because both cuBLAS calls keep today's operands. `Lifetimes` takes the
4B's 6.44 GB to 4.29, `InPlace(512)` to 2.68 and `InPlaceAll` to 2.15 (*computed*). `Rows` and `RowsTrim` go lower,
0.54-1.07 GB, but change the shapes cuBLAS sees. On this card only 2 of 6 measured prefill reshapes kept their bits
(*measured*, gqa-ab-2026-11-01), so these two are T8 candidates. `RowsTrim(1024)` also removes a quarter of the
attention's work, about 0.5-0.65 s of the route's first token (*estimated*). At 64 query heads (32B) even
`InPlaceAll` takes 4.29 of the triplet's 5 GB margin, which must also absorb the CUDA context and every other
activation. Only a T8 variant or the d2 kernel fits there.

The route is a T8. The dense reconstruction and the fused prefill pair at +0.0095 % [-0.027 %, +0.046 %] of
perplexity over 12 old windows (*computed*). That pairing joins two jobs a month apart, so it is reported and not
published (METHODE §2). It indicates a small cost and licenses nothing; the fresh-window control of
`prereg-prefill-2.md` is what a T8 decision would rest on. The bit-exact rows path 1a reaches 5.9-9.1 s at best
(*estimated*), bounded by shared-memory reads of the f32 activation. It keeps its value for verify and as the
T8-free fallback.

Four unknowns decide:
- cuBLAS identity under row chunking, with a prior of 0.1-0.25 (*estimated*);
- the attention time at 8k, 2.2-2.55 s (*estimated*), which a micro timing settles;
- the FP64 cost of an exact dequant, 0.15-0.23 s a pass (*estimated*);
- the achieved f16 tensor rate, which the window difference puts at 187-271 TFLOPS, above the 181 half-rate reading
  (*estimated*).

All four, plus the rows kernel's identity with the one-row kernel, take about 3 min of card time, $0.09
(*computed*), riding the operator's next job for audit steps 2-4. The full version adds timed first tokens and the
served-door replay, about $0.77 on a job of its own (`prereg-prefill-1.md`). Neither changes a served path, and
neither is a T8.

## 2. What exists today

| piece | where | state |
|---|---|---|
| prompt driver | `model.rs:2190-2203`; `fusedrun.rs:457` | chunks of 4,096 (`tokens.chunks(chunk)`, `MAX_PREFILL_ROWS`, `model.rs:738`), each through `hidden_cached` |
| causal mask | `model.rs:2454-2467` | one host `Vec<f32>` of l × L per chunk, holding +0 and -inf, uploaded then cast: 134 MB at the second chunk of 8k |
| attention, expanded | `model.rs:1682-1688` | three live f16 score tensors |
| attention, grouped (served 4B) | `model.rs:571-593` | same; the first `scores` binding (`:582`) is shadowed, not dropped |
| Tetra rows kernel | `tv_tetra48_h.cu:157-230`, `llvq_tetra48.cuh:180-205` | 4 rows × tile 128 = 49,152 B (`tile.rs:100-118`); per-row FMA order kept; identity with the one-row kernel proved on the host only |
| rotation of rows | `rotate.cu:103`, `fused_cuda.rs:893` | one launch per chunk of 4 rows |
| int4 projections | `model.rs:1213-1228`, `fused_cuda.rs:930-959`, `tv_q4_h.cu:117-149` | one launch per row; a row at an offset pays an f32 cast; activation staged whole, d_in × 4 B |
| `regroup` | `model.rs:1102-1106`, `:1206`, `:1227` | 252 copies per prompt row |
| head | `model.rs:2113-2117`, `:2140` | last row only, so it does not enter TTFT |
| dense reconstruction | `llvq-artifact/src/format.rs:1510-1532`, `sealed.rs:519-531` | host f64 decode and un-rotation, narrowed to f32, cast to f16 on the card |
| exact reconstruction outside Rust | `llvq-tetra/llvq_tetra/dequant.py` | numpy f64, operation for operation, gated bit for bit against the `hfdense` digests |
| measurement modes | `fusedrun.rs:438-494` (`LLVQ_CONTEXTS`), `:526-610` and `:700-766` (`LLVQ_GQA_PPL`) | TTFT sweep and paired perplexity exist; no attention timing, no memory peak, windows capped at 4,096 |

What works: the prefill is deterministic, and the prefill pairs 4:128, 8:64 and 16:32 leave the logit guard unchanged to
every printed digit on the card (*measured*, `f1e-prefill-sweep-2026-09-11.txt`). What is missing:
- the GEMM route;
- an int4 rows kernel and a head rows kernel;
- a lower score peak in the prefill attention, and a flash kernel;
- a memory-peak reading;
- an 8k quality harness.

Where the 35.02 s goes:
- 4.00 to 4.27 ms per prompt token from 2,048 to 8,192 (*computed* on `served-grouped-2026-11-02.txt`);
- about 486 row-granular operations per token at about 8 µs each (*computed*, wave 1 U1, and 4.80 µs per removed
  operation in `f1e0-2026-09-10.txt:360`);
- attention, 2.2-2.55 s at 8k (*estimated*, §3.2), 6-7 % of the total.

## 3. Design

### 3.1 Prefill attention with a lower score peak (a), zero format bits

Each layer keeps three f16 score tensors alive at once, of S = n_heads × l × L × 2 bytes each. The scaled scores stay
bound to their first `let`, the masked tensor is a second, and the softmax output is a third (`model.rs:582-590`
grouped, `:1682-1688` expanded). Five variants lower the peak. The first three keep both cuBLAS calls of today,
operand for operand. The last two change the shapes cuBLAS sees.

Exact variants:
- **A0, `Lifetimes`.** The first binding moves into the mask arm and dies before the softmax; the masked tensor dies
  before the PV product. Same calls, same shapes. Peak 2S.
- **A3, `InPlace(r)`.** QK^T runs in full into the raw score buffer, as today. Each block of r query rows is narrowed,
  scaled, masked and softmaxed, then written back into that buffer with `Tensor::slice_set` (candle-core 0.9.2
  `tensor_cat.rs:246-300`, a strided copy). PV then runs in full on the buffer. Peak S + 2 S_b, with S_b the block's
  share. One extra pass over S costs about 0.2-0.3 s at 8k (*estimated*). It uses public candle calls only, so the
  same code runs on the Mac CPU.
- **A4, `InPlaceAll`.** The same three candle kernels run in place on the whole buffer through
  `Tensor::inplace_op1`. `affine_f16` takes a null input, which the kernel reads as in place (`affine.cu`).
  `badd_f16` writes onto its contiguous left operand. `softmax_f16` takes the buffer as source and destination. None
  of the three declares `__restrict__`, and each thread reads then writes its own index only. Peak S, no extra pass.
  It is CUDA code against candle's kernel names (`get_or_load_func`, `cuda_backend/device.rs:227`), so it costs more
  care than A3.

T8 candidates:
- **A1, `Rows(r)`.** Query rows in blocks of r, each block through the existing function on `q.narrow(2, r0, len)`
  and `mask.narrow(2, r0, len)`, joined by `Tensor::cat`. For l ≤ r the old call runs verbatim, so decode and short
  prompts cannot move. Peak 3 S_b, or 2 S_b with lean blocks.
- **A2, `RowsTrim(r)`.** The same, with k, v and the mask columns narrowed to the block's causal limit, keys
  [0, L − l + r0 + len). At 8k the work falls to 0.75 (r = 1,024) or 0.71 (r = 512) of today's rectangles
  (*computed*).

Peak score memory at the second chunk of an 8k prompt (L = 8,192), in GB (*computed*):

| model | today, 3S | A0, 2S | A3, r = 512 | A3, r = 256 | A4, S | A1/A2 lean, r = 512 |
|---|---|---|---|---|---|---|
| 4B, 8B (32 heads) | 6.44 | 4.29 | 2.68 | 2.42 | 2.15 | 0.54 |
| 14B (40 heads) | 8.05 | 5.37 | 3.36 | 3.02 | 2.68 | 0.67 |
| 32B (64 heads) | 12.88 | 8.59 | 5.37 | 4.83 | 4.29 | 1.07 |

The triplet (ETAT §4) subtracts the 8k cache from the card on its own line. Its 5 GB margin must absorb the
activations, this transient and the CUDA context, which nothing has measured on this card
(`docs/archive/note-produit-2026-08-13.md`, §B bis). Up to 40 heads, A3 at r = 256 or A4 leaves at least 2.0 GB of
the margin for the rest (*computed*). At 64 heads A4 alone takes 4.29 GB. The other activations of an 8k prefill at 32B, about 1 GB, and
the CUDA context then push past 5 GB (*estimated*). There only A1, A2 or d2 fit, and each is a T8 event.

**Why A0, A3 and A4 keep every bit**, operation by operation:
- QK^T and PV are today's calls, with the same operands, shapes and strides. cuBLAS picks today's algorithm and
  returns today's values.
- The scale is candle's affine kernel, `x * mul + add` with add = +0 (`affine.cu`). Under A3 it reads a narrowed view
  through the kernel's strided branch, with the same expression. A contraction that differed between the two
  branches could change only the sign of a zero result.
- The mask add is x + m, with m either +0 or -inf. It turns a zero of either sign into +0, keeps every other x, and
  gives -inf where m is -inf. After it, every element equals today's, sign of zero included.
- The softmax runs one warp per row (grid `n_rows`, block `(1, 32, 1)`, `candle-nn-0.9.2/src/ops.rs:340-365`). Column
  j goes to lane j mod 32, each lane sums its columns in order, and a fixed xor tree reduces the lanes
  (`candle-kernels-0.9.2/src/reduce.cu:233-273`). A row's result depends on that row only.
- `slice_set` is a copy.

The strict form of A3 calls `.contiguous()` on each block before the scale, so the contiguous branch runs as today,
for one more copy. The card check decides whether it is ever needed.

**A1 and A2 change what cuBLAS sees.** QK^T and PV go through `cublasGemmStridedBatchedEx` with `CUBLAS_COMPUTE_32F`
and `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (`candle-core-0.9.2/src/cuda_backend/mod.rs:2566-2620`). The algorithm depends on
m, n, k and the batch count, and a split-K or sliced-K choice changes the summation order. A2's softmax keeps its
bits: the removed columns hold -inf, their exp is exactly 0, and the remaining columns keep their lanes. Only the two
products are in question.

**The card has already answered a close question, mostly with no.** The prompt's last-position logits of two
reshaped forms against the expanded one, bitwise (*measured*, `docs/mesures/gqa-ab-2026-11-01.txt:62-72`):

| prompt | chunks (l, L) | shared form: batch 8, M = l | grouped form: batch 8, M = 4l |
|---|---|---|---|
| 2,048 | (2,048, 2,048) | bit for bit | 3.711e-2 |
| 4,096 | (4,096, 4,096) | 3.516e-2 | bit for bit |
| 8,192 | (4,096, 4,096), (4,096, 8,192) | 3.906e-2 | 3.418e-2 |

- Only 2 of 6 reshapes kept their bits.
- The shared form keeps the expanded per-head M, N and K and changes the batch count and batch strides only. It still
  moved at 4,096 and 8,192, so the batch count alone changes cuBLAS's rounding on this card. ETAT §4 records the same.
- The grouped form matched at 4,096 only. At 8,192 its first chunk is that matching case, so the difference comes
  from the second chunk, L = 8,192, which is the case a row block meets.
- gqa-ppl's nine-decimal agreement between the grouped and expanded prefill paths is that same 4,096 case
  (*measured*, `gqa-ppl-2026-11-01.txt`). It does not extend to other shapes.

Prior that a variant is bit for bit at the three served shapes in both forms (*estimated*): `Rows(1024)` about 0.25,
`Rows(512)` and `RowsTrim(r)` about 0.15-0.2. Across the whole grid below, about 0.1-0.15 for `Rows(1024)` and about
0.1 for the others. The expected outcome is exact variants only, with A1 and A2 as T8 candidates.

**Identity checks on the card:**
1. A grid of (l, L) pairs (`identity_grid`, `code/prefill/attn_rows.rs`). Prompts P every 256 from 256 to 8,192, plus
   16 pseudo-random P from a printed seed, each mapped to the chunk pairs the prompt driver makes. About 48 pairs,
   both forms, every variant, on seeded random f16 q, k and v laid out as the stores lay them out. Each output is
   compared bitwise with `Off` at the same pair. The pairs include remainder blocks: a 6,000-token prompt gives
   (1,904, 6,000), split into 1,024 + 880 or 3 × 512 + 368 rows. The grid costs seconds of card time (*estimated*).
2. End to end: last-position logits and 256 greedy tokens after prompts of 2,048, 4,096 and 8,192, bitwise against
   `Off`.
3. Once an exact variant is adopted, a load-time check at one shape, as `check_int4_half_parity` does
   (`fused_cuda.rs:2224`). For an exact variant it guards the harness, not a heuristic.

A grid certifies identity on its pairs only. The served prefill meets every (len, L), and the gqa-ab table shows
identity flipping with L at a fixed form. A variant that changes a cuBLAS shape therefore stays a T8 candidate even
if the whole grid passes, unless the operator rules that a grid suffices.

**What ships under each outcome.** A3 or A4 identical everywhere is expected with certainty. It becomes eligible
without T8, behind a config field, after the operator's go and his reading of the ROADMAP §4 hold. If A0, A3 or A4
differs anywhere, the harness or the argument above is wrong, and the code is fixed first. A1 and A2 go to the
operator as T8 candidates with their grid counts, max |Δlogit| and first token divergence. Their control can ride
`prereg-prefill-2.md` step 5 as a third arm, `RowsTrim(1024)` on top of the route, for about +9 min and +$0.27
(*computed*). Otherwise they wait for d2, which makes chunking exact relative to itself (§3.4).

### 3.2 GEMM route (c), for the branch doing audit step 5

**Integration.** The branch goes in `group_forward` (`model.rs:1132`), after the dense branch, for rows ≥ R*. Each
projection materializes an f16 matrix and calls `candle_nn::Linear::new(w, None).forward(x)`. That is the dense arm's
own call: for a contiguous rank-3 input, candle reshapes to 2D and runs one matmul (`candle-nn-0.9.2/src/linear.rs`).
R* ≥ 16, printed on the provenance line, keeps gate 4 (a 5-token prompt) and any k ≤ 16 verify on the rows path. The
rows path keeps the "N tokens in one call against N calls" contract that exact speculation needs.

**Two arithmetics.** Both are T8 against today's prefill, and the operator chooses between them.

- **C-dense: the dense arm, exactly.**
  - The chain is `decode_matrix`'s (`format.rs:1510-1532`): each f64 point times centroid · row_scale / sqrt(16m),
    the f32 tail widened, then `apply_transpose` (`rotation.rs:109-160`). That means the mix with an accumulator from
    zero over t ascending, the WHT with len doubling then ×1/sqrt(m), then the signs. The result is narrowed
    f64 → f32 → f16; candle's cast rounds to nearest (`cast.cu`, `out[i] = inp[i]`).
  - Two references exist: Rust `decode_matrix`, and numpy `llvq_tetra/dequant.py`.
  - What it gives: under the same attention form and chunking, the route's hidden states equal the dense arm's bit for
    bit. The served prefill becomes the arithmetic on which MMLU and perplexity are published. Every record checks
    against its dense digest.
  - What it costs: AD10x runs FP64 at 1/64 of FP32 (*estimated*: NVIDIA's L40S page lists no FP64). That is 81 G FP64
    instructions a full 4B pass, 0.11 s at peak and 0.15-0.23 s in practice, and 0.5-0.9 s at 14B (*estimated*).
  - It also needs two tables the card does not hold: f64 row scales and the f32 tail (the card holds an f16 tail and
    f32 scales, `fused.rs:1056-1065`). They weigh 66.2 MB at 4B, 0.132 b/param if resident (*computed*). Keep them in
    pinned host memory and upload them per pass.
- **C-rot16: rotated basis, f16.**
  - The weights are W_rot = point × gscale × invnorm × rscale, computed in f32 then rounded to f16, with the card's f16
    tail. The activation is the f32 output of `rot_apply_rows`, cast to f16.
  - It is cheap, about 10 ms a pass (*estimated*), adds no resident memory, and works at every size.
  - It is a third arithmetic. The only references are the served per-weight values (`tetra48_decode_f`,
    `llvq_tetra48.cuh:214-233`) rounded to f16, and after that a tolerance.

Recommendation: C-dense first. The kernel can be checked bit for bit against the digests and the dense arm, and its
T8 imports a known quality, the published one. C-rot16 replaces it at the size where the measured FP64 cost passes 20 %
of TTFT.

**C-dense kernel, `deq_tetra48_f64_h`:**
- one CTA of 256 threads per output row;
- the row's words decoded by the served decoder (`f1r_v3_quads`, `tetra48_n2`) into exact integers;
- the f64 row in shared memory when d_in × 8 ≤ 101,376 B (2,560: 20 KB; 9,728: 78 KB; 12,288: 98 KB), otherwise in an
  L2 scratch (the 14B down site, 17,408: 139 KB);
- the mix done by one thread per position j, with k accumulators, and barriers between WHT stages;
- every f64 operation through `__dmul_rn`, `__dadd_rn`, `__dsub_rn`, `__dsqrt_rn`, `__ddiv_rn`, because NVRTC defaults
  to `--fmad=true`;
- the store as `f2h(__double2float_rn(x))` (`f2h` rounds to nearest even, `matvec.cu:239`), never a direct f64 → f16
  conversion, whose single rounding differs;
- int4 records through `q4_deq` (`tv_q4_h.cu:74-78`): an f32 multiply then an add, then f16.

A sketch is in `code/prefill/deq_tetra48_f64.cu`.

**Transient VRAM** (*computed*):
- the f16 scratch of the largest matrix: 49.8, 100.7, 178.3 and 262.1 MB at 4B, 8B, 14B and 32B;
- for C-rot16, also the rotated activation in f32 and f16: up to 0.48 GB at an 8,192-row down site at 4B, half that
  at 4,096 rows.

The memory returns through the stream-ordered pool: cudarc 0.19.8, which candle 0.9.2 uses, allocates with
`cuMemAllocAsync` (`core.rs:1535-1536`).

**First token.** The route lands at 2.85-3.4 s at 8k (*estimated*). Three dense windows of 4,096 tokens were timed in
one job: 1.036 s on the 4B, 1.191 s on the 8B and 1.764 s on the 14B (*measured* timestamps,
`ppl-scelles-2026-10-04-brut/ppl-{4b,8b,14b}.txt`). A window is A, the attention over 4,096², plus L4, the projections
and elementwise work of 4,096 rows, plus h, an all-position head and log-softmax of 0.03-0.05 s on the 4B
(*estimated*).
- The 8B has the 4B's attention geometry: 32 query heads, 8 key heads, head 128, 36 layers. A cancels in
  W8 − W4 = 0.155 s.
- With L4 on the 8B at 1.6-1.9 times the 4B's (projections ×1.91, elementwise ×1.26-1.6) and 0.005-0.015 s more head,
  L4 on the 4B is 0.155-0.25 s (*estimated*). The 0.1 s print widens it to 0.135-0.28.
- The 14B window agrees. With attention ×1.389 and L4 ×2.9-3.4, it implies 0.152-0.237 s.
- So A(4,096²) = 0.74-0.85 s and A(8k) = 3A = 2.21-2.55 s (*estimated*), or 20.5-23.6 ms a layer at (4,096, 4,096) and
  40.9-47.3 ms at (4,096, 8,192). A low end of 1.85 s would need an 8B window of 1.27-1.38 s.
- Physical floor: the 4B's projections in one window are 29.8 TFLOP, 0.082 s at the 362 TFLOPS spec. So L4 is at least
  about 0.12 s and A(8k) at most 2.7 s (*computed*), which is R4's upper threshold in `prereg-prefill-1.md`.

The algebra ties A to L4: TTFT(8k) = 2·L4 + 3·A + 0.04 + 2·dq = 3W − L4 − 3h + 0.04 + 2·dq, with dq one dequant pass
and 0.04 s for the second chunk's mask (*estimated*). Both ends of a range come from one L4.

| route, 4B, chunks of 4,096 | TTFT(2,048) | TTFT(4,096) | TTFT(8,192) |
|---|---|---|---|
| the dense arm's own prefill | | | 2.75-2.90 s |
| C-rot16, today's attention | 0.34-0.42 s | 1.05-1.13 s | 2.85-3.10 s |
| C-dense, today's attention | 0.44-0.57 s | 1.15-1.28 s | 3.05-3.40 s |
| plus `RowsTrim(1024)` (T8) | | | minus 0.5-0.65 s |
| plus a flash attention, two chunks | | | 0.49-1.22 s |
| plus a flash attention, one chunk | | | 0.44-0.97 s |

All *estimated*. The dense window ran the expanded form. Under the grouped form the rows path reads 35.02 s against
34.84 at 8k (*measured*, served-grouped), so the served config may add up to 0.2 s. At 14B the route gives 4.6-5.6 s
(C-rot16) and 5.3-6.7 s (C-dense) against 74-81 s today (*estimated*, from the 14B's own window).

**The dequant is paid per prefill call.** For short prompts it dominates. R* breaks even with today's rows path near
40-60 rows under C-dense and near 10-15 under C-rot16 (*computed*: dequant time over 4 ms a row). A census through the
served object, 14,042 questions averaging 606 prompt tokens (*measured*, `f1e0-2026-09-10.txt:200-210`), would pay
35-54 min of dequant alone under C-dense. That is about $1.5-2.5 in all, against $20-70 today (ROADMAP §2.1), and
about $0.5-0.8 under C-rot16 (*estimated*).

The census inherits the dense arm's score under the expanded attention only. Under C-dense, the route's hidden states
equal the dense arm's when both run the expanded form, and the published 63.37 is the expanded form's (ETAT §2 and
§5). Then only the head separates the two scores. The served 4B runs the grouped form, whose prefill differs from the
expanded one at a 2,048-token prompt (3.711e-2, gqa-ab), and MMLU prompts reach 3,096 tokens (f1e0). Under the grouped
config the head and the attention form both separate the scores. So the C-dense census that inherits the dense one
runs under the expanded config. The grouped config's census is a separate reading.

### 3.3 Rows path 1a, bit for bit

- **`tv_q4_rows_hh<R>`.**
  - Each lane keeps its word order: words wi ≡ lane (mod 32), ascending, with R accumulators and one `q4_deq` per
    weight applied to R rows.
  - The activation is staged in tiles of T columns, T a multiple of 256, so a lane's words stay in order across tiles.
  - The f16 rows are read at their offsets and widened while staged. Widening is exact, and it removes the per-row
    cast.
  - Budgets (*computed*): untiled, 8 rows need 311,296 B at d_in 9,728 and 557,056 B at 17,408. Tiled at the 49,152 B
    default, R = 4 gives T = 3,072 (4 and 6 tiles), R = 8 gives T = 1,536 (7 and 12), R = 16 gives T = 768 (13 and 23).
    At the 101,376 B opt-in, R = 8 gives T = 3,072.
  - One launch per projection per chunk, with `gridDim.y = rows / R`.
  - Sketch: `code/prefill/tv_q4_rows_hh.cu`.
- **Tetra rows.**
  - The same 2D grid replaces one launch per 4 rows.
  - One `rot_apply_rows` per site per chunk.
  - The prefill pair can move to 16:32 at zero bits (*measured* invariance, f1e-prefill-sweep).
- **`regroup` removal.** Allocate y [rows, d_out] once per site; each kernel writes at y + row0 · d_out.
  `drive_rows_batched` (`rotplan.rs:472-552`) then carries one tensor per site.
- **Head rows kernel.**
  - Same pattern on `tv_emb_q4_h` (`emb_q4.cu:75`): the table is read once for k rows.
  - It serves verify, and all-position scoring, where gqa-ppl rereads 218.8 MB per row. It does not serve TTFT.
- **Load-time identity.**
  - Per kernel class, one real record and R pseudo-random rows, compared with R one-row launches, bitwise; the load
    refuses on a mismatch.
  - This matters because the card identity of today's rows kernel with the one-row kernel has never been checked.
  - The f1e0 guard measured N tokens in one call against N calls at 6.8e-2 (200 tokens) and 5.6e-2 (800 tokens) of
    max |Δlogit| (*measured*, `f1e0-2026-09-10.txt:349-350`), without separating attention from projections.
- **TTFT model.**
  - Each weight costs one 4-byte shared read per row: 14.53 GB per prompt token. That is 0.32 ms at 45.8 TB/s
    without conflicts, and 0.57 ms with the 2-way conflict of the 96-byte lane stride (*computed*).
  - Add 0.1-0.2 ms of other work, so 0.45-0.8 ms a token: 5.9-9.1 s at 8k with 2.21-2.55 s of attention
    (*estimated*).
  - Scalar shared loads instead of LDS.128 would put it near 2.2 ms a token. The SASS read decides.

### 3.4 Prefill attention kernel (d)

- **d1: candle-flash-attn**, FlashAttention 2 as candle vendors it (f16 and bf16, GQA, causal).
  - 1-3 days of wiring, plus a heavy nvcc build in an image whose builder has already been SIGKILLed.
  - A T8, with no promise that chunk sizes or a decode row share its bits.
  - An external dependency, so an operator decision.
- **d2: our kernel `tv_attn_prefill_h`, row-invariant by construction.**
  - KV tiles of 64 keys, aligned to absolute positions. A row at position p reads tiles 0 to p/64 whatever the chunk.
  - Per-tile partials (m_t, l_t, O_t), folded in ascending tile order, sequentially.
  - Equal order alone does not give equal bits. NVRTC compiles with `--fmad=true`, and two separately compiled kernels
    may contract the fold's rescale-and-add (O·α + O_t·β, l·α + l_t·β) differently, or pick another exp. The fold
    therefore lives in one header shared by the prefill kernel and the stage-3 decode kernel
    (`code/prefill/attn_fold.cuh`). Every multiply and add in it is `__fmul_rn` or `__fadd_rn`, its one exp is `exp2f`
    of an `__fmul_rn` prescale, and its one division is `__fdiv_rn`.
  - The per-tile scores and PV products come from the same `mma.sync.m16n8k16` sequence in both kernels; the decode
    kernel pads its row into a 16-row tile. That the tensor core accumulates an element the same way whatever its row
    in the tile is an assumption, which the card test "prefill row equals decode row" checks.
  - The split-KV decode kernel writes per-tile partials and folds them sequentially. A tree of per-split folds would
    be another order.
  - Host replay with clang++ `-ffp-contract=off` against the device fold, given the per-tile partials the device
    prints. Mutants: a contractible fold expression, `expf` in one kernel and `attn_exp` in the other, tiles aligned to
    the chunk start, a per-split combine, a reciprocal-multiply division.
  - Consequences: (a) becomes exact by construction, and attention in a prefill of N tokens equals N decode steps. One
    T8 then covers both.

d2 geometry for sm_89:
- a CTA of 4 warps, 64 query rows of one head (16 per warp), Bc = 64, hd = 128;
- `mma.sync.m16n8k16` with f16 inputs and f32 accumulation;
- Q fragments in 32 registers, a 16 × 64 f32 S tile in 32, a 16 × 128 f32 O in 64, about 160 registers per thread;
- K and V tiles of 16 KB each plus 16 KB of Q staging, so 48 KB and 2 CTAs per SM (40,960 of 65,536 registers);
- a grid of (l/64) × n_heads, 2,048 CTAs at l = 4,096, with whole tiles past the diagonal skipped.

Its expected time at 8k (1.98e13 FLOP, causal triangle, *computed*):

| at | time |
|---|---|
| 362 TFLOPS | 0.055 s |
| 181 TFLOPS | 0.109 s |
| 91.6 TFLOPS (CUDA cores) | 0.216 s |
| 50-70 % of the tensor peak (*estimated*) | 0.08-0.22 s |

The host mask and the score tensors disappear.

This is a T8 against today's path, which rounds scores to f16 before the scale and the mask; the kernel keeps them in
f32. d2's design and identity tests are at $0; its paid steps come after (c). Alone, on today's rows path, it buys
35.0 s → 32.6-33.0 s (*estimated*).

## 4. Deciding unknowns

| unknown | current best value | confirms | kills (a proposal; the operator decides) |
|---|---|---|---|
| U1. attention time at 8k, today's path | 2.21-2.55 s (*estimated*, three windows of one job) | measured 1.5-2.7 s: (d) is worth 1.3-2.6 s after (c) | below 0.8 s: (d) drops behind decode work; 0.8-1.5 s: mid priority; above 2.7 s: the window decomposition is wrong, redo §3.2 |
| U2. cuBLAS identity under A1/A2 | 2 of 6 measured prefill reshapes identical (*measured*, gqa-ab); prior 0.1-0.25 (*estimated*) | bitwise on the grid and end to end: the operator may rule the grid suffices | any difference, the expected case: T8 candidates; A0, A3, A4 unaffected |
| U3. FP64 dequant cost (C-dense) | 0.15-0.23 s a 4B pass (*estimated*, 1/64 rate) | probe ≥ 0.5 T instructions/s and a measured pass ≤ 20 % of TTFT: keep C-dense | probe below 0.3, or a pass above 20 %: build C-rot16 |
| U4. GEMM route T8 quality | +0.0095 % [-0.027 %, +0.046 %], cross-job, reported, not published (*computed*) | fresh windows, both paths, inside ±0.12 % | an interval leaving the band: route not served, 1a becomes the TTFT route |
| U5. VRAM peak at 8k | 9.6 GB today at 4B (*estimated*: 1.38 + 1.21 + 6.44 + about 0.6) | measured pool high-water within 10 % of the model | far above: unknown allocations, profile first |
| U6. rows kernel against one-row on the card | never run | bitwise: 1a is the verify substrate as designed | mismatch: today's prefill already differs from decode in projections, and verify needs a fix first |
| U7. achieved f16 rate with f32 accumulation | spec 362.05 dense; archive 181; window difference 187-271 achieved (*estimated*) | probe above 181: the half-rate peak is refuted | probe at or below 181: open, since a low probe cannot separate a low peak from an inefficient shape |

## 5. Validation ladder

Every step needs the operator's go, a cost announced before and the running total after (rule 1). Paid steps need a
stamped prereg (rule 2). Cost anchors from `docs/data/jobs.csv`, l40sx1 at $1.80/h (`ops/run.py:93-110`):
served-grouped 18 min for $0.54; gqa-ppl-r2 28 min for $0.82, at 123.5 s a window (*computed* from its printed clock);
trims-contexts 12 min for $0.35. Spent so far: $248.84 over 237 rows (*measured*).

**V0, $0, done here.**
- The derivations of §3 and the facts of §8.
- The U2 prior rebuilt from the gqa-ab identity table: 2 of 6 prefill reshapes identical, and the second-chunk case
  failed under both reshapes.
- U1 narrowed from three windows of one job: 2.21-2.55 s.
- U7 estimated from the same windows: 27.1 TFLOP of extra 8B projections run in about 0.10-0.145 s, so 187-271 TFLOPS
  achieved. That suggests f32 accumulation runs above 181, without settling it.

**V1, $0, on the Mac.**
- Host tests of A0 to A4 on the CPU (A4 through its CPU twin): both forms, f32 and f16, l not a multiple of r,
  r ∈ {1, 7, 64}, Cat- and Grow-shaped k and v, outputs equal to `Off` bit for bit. Mutants: wrong mask rows, wrong trim
  offset, reversed concatenation, a block written back at the wrong offset, the scale applied after the mask.
- A host replica of candle's CUDA stride rules (`cuda_gemm_layout_check`, from `cuda_backend/mod.rs:1341-1432`),
  asserted on every matmul operand the variants build. The CPU gemm accepts layouts cuBLAS refuses: gqa-ppl attempt 1
  died on exactly that for $0.09 (deviation É1). Mutant: the É1 layout must be refused. Without this check, a green
  CPU run proves nothing about the card.
- The Linux half type-checked from the Mac where the Mac can, under `nice` (METHODE §6). Host code placed in
  `llvq-cuda` passes `CUDARC_CUDA_VERSION=12040 cargo clippy --target x86_64-unknown-linux-gnu -p llvq-cuda
  --all-targets` (CLAUDE.md), so the pool high-water calls and the FP64 probe's launcher go there. The `cfg(linux)`
  code of `llvq-llm` does not build on the Mac with the cuda feature, and the image is its first compiler
  (*measured*, served-grouped-2026-11-02 header). That part stays a thin measurement mode.
- For the branch: a clang++ `-ffp-contract=off` replay of the C-dense device functions against `decode_matrix` on
  fixture records, bitwise (the `tests/host_tetra48.cpp` pattern). Mutants: contracted FMA, single-step f64 → f16,
  reversed mix order, WHT scale before the stages.
- Rule: all green with every mutant killed lets the code go to a card. Otherwise it is fixed first.
- Needs: a go for `cargo test` and `cargo clippy` under `nice` and `-j 4`, under 10 min of machine load.

**V2, $0 on the builder.** The SASS read of `tv_tetra48_rows_h` and of the int4 rows sketch: shared load width and
conflicts. It settles 1a's floor: LDS.128 means 0.45-0.8 ms a token, scalar loads about 2.2 ms. Needed only if 1a is
pursued. Rule: LDS.128 with 2-way conflicts or better keeps 1a's 12 s gate reachable. Otherwise the XOR swizzle
comes first.

**V3-lite, about $0.09, riding the operator's next card job.** The measurement mode's micro steps only: 3a (the
identity grid with per-layer timings at the three served shapes), 3c (the f16 GEMM probe), 3d (rows against one row)
and 3e (the FP64 probe). About 3 min of card time (*estimated*); riding saves the container and oracle overhead of a
job of its own.
- It settles U2 on the grid, U1 through the per-layer times, U3's peak, U6 and U7.
- It needs the measurement mode merged into the branch that builds audit steps 2-4, which is the operator's call. Its
  base is that branch's tip; nothing in the micro steps rests on a first-token prediction.

**V3, about $0.77, `prereg-prefill-1.md`, a job of its own.** V3-lite plus the served-door replay under the grouped
config, end-to-end identity at 2k, 4k and 8k, timed first tokens at 8,192 for `Off`, `InPlace(512)` and
`RowsTrim(1024)` with rounds interleaved, and the real prefill's pool high-water.
- About 26 min, timeout 60 min ($1.80), cap asked $2 (*computed* from the jobs.csv anchors).
- Base commit, the operator's choice: debit-trims 06e1602, before the step-2 fusions, with the first-token
  predictions as signed; or the branch tip, with them re-signed before the stamp.
- It touches no served path and is not a T8.

**V4, about $1.40-1.67, `prereg-prefill-2.md`, the branch's step 5 gate.**
- Card checks: the dequant against the dense arm on all 252 records, and against five digests read from the bucket
  copy `hf-cuda-4b-2026-09-30/model/llvq-dense-digest.json`, whose source sha256 the job checks. `hfdense` is in
  neither list of `ops/Dockerfile.cuda`, and no 4B dense digest is committed.
- TTFT of the rows path against the route at 2k, 4k and 8k, interleaved; the pool peak.
- The T8 control under the served grouped config: paired prefill-path perplexity on 24 fresh windows (WikiText-2 test,
  windows 13 to 36), and a prompt-then-decode path. Its harness gate compares the two paths on the same positions,
  2,048 to 4,095.
- A third arm, `RowsTrim(1024)` on the route, is A2's T8 control on the same windows, if V3 leaves A2 worth one.
- About 47 min and $1.40 with two arms, 56 min and $1.67 with three (*estimated*, decode at the measured 8.78-10.20 ms
  a token). Timeout 90 min ($2.70), cap asked $3. The gqa-ppl-r2 anchor, 123.5 s a window, matches the per-step costs
  used.
- It needs the operator's T8 decision for the route before any serving, and his reading of the ROADMAP §4 hold.

**V5, paid, the task debt.** MMLU through the served object, about $1.5-2.5 under C-dense and $0.5-0.8 under C-rot16
once the route serves (§3.2). Under the expanded config it inherits the dense census; the grouped config's census is a
separate run. GSM8K through the kernel at about $2 (the expanded arm ran at $2.13, *measured*, jobs.csv). It settles
the task cost of the T8s carried (grouped form, route). Rule: inside the paired bars keeps the route. Otherwise the
operator decides.

**V6, d2.** A host replay of the shared fold header; card identity across chunk sizes and against a decode row. Both
are exact by construction, so any difference is a bug or a failed MMA-row assumption. Then a T8 control on both paths,
and TTFT. About $1-2 (*estimated*).

**V7, 1a**, if the route's T8 is refused or for verify. Gate: TTFT(8k) ≤ 12 s, decode not slower, gate 4 unchanged,
the load checks green. About $1 (*estimated*).

**Map to audit step 5**, which the branch is building now:
- V1's replay and V4 are its validation, and its target is 2.85-3.4 s at 8k, not 2 s.
- (a) is not in the audit. It is the ML audit's zero-bit fix, and V3-lite validates it without touching step 5.
- V3-lite's GEMM probe gives step 5 its throughput number for free.

**Pitfalls the branch should avoid:**
1. FMA contraction in the f64 chain: NVRTC compiles with `--fmad=true`, and the Rust reference never contracts.
2. A direct f64 → f16 conversion: the dense arm rounds twice, f64 → f32 → f16.
3. Un-rotating in f32: `format.rs:1505-1506` says it moves the last bits.
4. The int4 dequant in f64 or contracted: it must be f32 multiply, then add (`dequant.py` header, `tv_q4_h.cu:74-78`).
5. A non-contiguous input to `Linear`: candle switches to a broadcast matmul with another cuBLAS call.
6. R* below 16: at 5 or less it moves gate 4, and anything below 16 puts a k-row verify on the route, which breaks
   exact speculation.
7. Rotating the activation under C-dense: the dense weights are in the natural basis.
8. Raising the outer chunk past 4,096 before (a): the 3S peak grows with l × L, 12.9 GB at l = L = 8,192 at 4B.
9. A TTFT gate without a memory reading: report the pool high-water.
10. A new binary: keep everything as `fusedrun` measurement modes. Otherwise both lists of `ops/Dockerfile.cuda` and
    `UPLOAD_ALLOW` apply.
11. Metal has no f64: C-dense is CUDA-only, and a Metal route is a separate T8.
12. A matmul layout that only the CPU accepts (gqa-ppl É1): run `cuda_gemm_layout_check` on every new operand.
13. `cfg(linux)` code left unchecked on the Mac: put the unsafe calls in `llvq-cuda`, which the Mac type-checks
    (METHODE §6), and keep the `llvq-llm` side, which only the image compiles, thin.
14. A harness gate over unmatched positions: compare the prompt-then-decode path with the prefill path on the same
    positions (`window_nll_split`, `code/prefill/gemm_route.rs`).
15. A reference that is not on the card: `hfdense` is not in the image; read the bucket's digest and check its source
    first (rule 9).

## 6. Build plan after validation

| order | step | effort (*estimated*) | tests and gates |
|---|---|---|---|
| 1 | A0 everywhere, then A3 (or A4) behind a config field (`"attention_rows"`; absent means `Off`) | 1-2 days, A4 2-3 | CPU identity tests and mutants; stride check; load-time check; V3 rules |
| 2 | GEMM route, C-dense, behind a config field (`"prefill": "gemm"`; absent means `"rows"`) and R* | 4-7 days | host replay; card digest check per record; hidden state equal to the dense arm at 2k; V4 |
| 3 | C-rot16, if U3 says so or for 14B/32B | 2-3 days | per-weight check against `tetra48_decode_f`; its own T8 |
| 4 | A2 `RowsTrim`, if its T8 control passes (V4 third arm) | 1 day | grid report; its T8 control; R* untouched |
| 5 | d2 flash prefill kernel with the shared fold header | 3-6 weeks | host fold replay; card identity across r and against a decode row; mutants (contraction, fold order, tile alignment, per-split combine); T8 control |
| 6 | 1a pieces, then verify | 2-3 weeks | load-time identity per kernel class; "N in one call = N calls" for projections; V7 |

Each served change follows the `attention` field precedent: an absent field reproduces every published number.

## 7. Rules map

- **T8 events.**
  - The GEMM route, whichever arithmetic: a new T8 decision with its own control. The grant of 2026-11-01 covers the
    grouped form at 4B only.
  - A1 and A2, whatever a grid says, unless the operator rules that a grid suffices. A0, A3 and A4 never.
  - d1 or d2. If d2 shares its fold with the stage-3 decode kernel, the operator rules whether one decision covers
    both.
  - 1a: no T8 by design.
- **ROADMAP §4 hold** ("this reopens the layout if prefill is ever served"). Both preregs carry it in their decision
  rules: serving any prefill change waits on the operator's reading (wave report decision 9). The branch at work
  suggests a reading; none is written.
- **Cross-job readings** (METHODE §2): the dense-against-fused pairing is reported and not published. Only the fresh
  windows of prereg-prefill-2 feed a T8.
- **A2 (graphs):** untouched. Nothing here captures or replays.
- **Rule 6:** C-dense's resident side tables would cost +0.132 b/param at 4B (*computed*); host-resident, 0. Transient
  VRAM is stated in MB beside the unchanged b/param.
- **Rule 4:** both arms of every TTFT A/B share the head (last row only), so the raw ratio is the same-head ratio. This
  is stated, not assumed.
- **Rule 5:** a llama.cpp or vLLM prefill anchor is read as prompt tok/s or achieved TFLOPS on the same L40S, never as
  a ratio across cards or stacks.
- **Closed leads.** No ETAT §6 row covers prefill. The 2026-08-05 archive clause ("tensor cores sans objet pour un
  matvec ... si le prefill entre au périmètre") lives only in `docs/archive`, so lifting it needs a go under rule 1;
  its own condition is arguably met by the 8k triplet.
- **Operator rulings owed:**
  - the ROADMAP §4 reading;
  - T8 for the route;
  - C-dense or C-rot16;
  - whether a grid suffices for A1 and A2 (the default is no);
  - an external dependency for d1;
  - V3's base commit, and whether V3-lite rides his branch's next job;
  - the spend cap.

## 8. Checked now at $0

1. The 3S peak is in the code: the first `scores` binding survives its shadowing in both forms (`model.rs:582-590`,
   `:1682-1688`). The sizes follow from 32, 32, 40 and 64 query heads (`rtbits-14b-2026-08-17.txt:226-227`; Qwen3-32B
   `config.json` on the Hub).
2. Softmax is per-row and per-lane fixed (`reduce.cu:233-273`, launch at `ops.rs:340-365`), so row blocks and -inf
   trimming leave it bit for bit.
3. candle's f16 GEMM is `cublasGemmStridedBatchedEx` with `COMPUTE_32F` and `DEFAULT_TENSOR_OP`
   (`cuda_backend/mod.rs:2566-2620`). Its layout rules (`:1341-1432`) accept a row-narrowed q, a key-narrowed k or v,
   and Grow views; they refuse the É1 layout of gqa-ppl attempt 1.
4. On this card, 2 of 6 prefill reshapes kept their bits (*measured*, `gqa-ab-2026-11-01.txt:62-72`). gqa-ppl's
   agreement between the two prefill paths is the 4,096 case of that table.
5. Dense against fused prefill, paired over the same 12 windows: +0.0095 % [-0.0268 %, +0.0458 %], 5 of 12 windows
   worse, largest window gap 0.178 % (*computed* from `ppl-scelles-2026-10-04-brut/ppl-4b.txt` and
   `gqa-ppl-2026-11-01-brut/gqa-ppl.txt`). The two jobs ran a month apart on two images, so this reading is reported
   and not published. The windows are not fresh, and the head differs between the two arms.
6. The three dense windows take 1.036, 1.191 and 1.764 s (*computed* from measured timestamps). They put L4 on the 4B
   at 0.155-0.25 s and A(8k) at 2.21-2.55 s (*estimated*).
7. The dense arm's weights are `decode_matrix` (f64 chain) then `to_dtype(F16)` on the card (`sealed.rs:519-531`). The
   candle cast rounds to nearest (`cast.cu`, `out[i] = inp[i]`).
8. The un-rotation order is mix, WHT, signs, with no `mul_add` (`rotation.rs:109-160`, `:192-212`). `dequant.py`
   mirrors it and is gated bit for bit against the digests.
9. The card holds an f16 tail, f32 row scales and f32 centroids (`fused.rs:1056-1065`, `:1122`, `:1130`, `:2334`).
   C-dense needs 66.2 MB more at 4B (*computed*).
10. `Linear::forward` on a contiguous rank-3 input is reshape plus one 2D matmul (candle-nn 0.9.2 `linear.rs`).
11. The L40S has 142 SMs, 49,152 B of shared memory per block by default and 101,376 B at opt-in, and 96 MiB of L2
    (*measured*, `f1-plancher-table-2026-09-05.txt:23`).
12. NVIDIA's L40S page lists FP16 and BF16 tensor at 362.05 dense and 733 with sparsity, FP32 at 91.6 and TF32 at 183.
    The archive reads 181 for f32 accumulation (`docs/archive/portage-noyau-cuda.md:410`). The 8B-4B window difference
    suggests 187-271 TFLOPS achieved (*estimated*).
13. cudarc 0.19.8 (candle's) allocates stream-ordered (`core.rs:1535-1536`) and exposes
    `CU_MEMPOOL_ATTR_USED_MEM_HIGH` (`sys/mod.rs:4397`). An in-process peak reading is possible without nvidia-smi.
14. The guard "N tokens in one call against N calls" measured 6.8e-2 and 5.6e-2 of max |Δlogit| at 200 and 800 tokens
    (`f1e0-2026-09-10.txt:349-350`). It mixes attention and projections.
15. The f16 scratch of the largest matrix is 49.8, 100.7, 178.3 and 262.1 MB at 4B, 8B, 14B and 32B (*computed*).
16. `slice_set` requires contiguous tensors and writes through a strided `copy2d` (`tensor_cat.rs:246-300`,
    `cuda_backend/mod.rs:2244`). candle's affine, binary and softmax kernels declare no `__restrict__`, and the affine
    kernel has an in-place mode, a null input.
17. The served door's whole decoded text hashes to `0597cee8` under both configs
    (`served-grouped-2026-11-02-brut/lines.sha256`); the bench door's reference line hashes to `8784ffc4`.
18. `hfdense` writes one sha256 per record over the f32 values and records its source file's sha256 (`hfpack.rs`,
    `dense_digest`). It is not built into the CUDA image (`ops/Dockerfile.cuda:52-54`). The bucket holds a 4B copy
    (`bucket-listing.txt:473`, 28,987 B).
19. MMLU 5-shot prompts average 606 tokens, with subject maxima up to 3,096 (*measured*, `f1e0-2026-09-10.txt:200-210`).
    The published 63.37 is the expanded form's (ETAT §5).
20. gqa-ppl-r2 ran 123.5 s a window over 12 windows (*computed* from its printed clock); jobs.csv bills it 28 min.

## 9. Not established

- Any time on the card for the route, the score-peak variants or d2; the FP64 rate of the L40S.
- The split between attention and linear work inside the dense window: U1 rests on three windows and estimated scale
  factors.
- cuBLAS identity for A1 and A2 on this card and CUDA 12.4, beyond the six reshapes gqa-ab measured.
- The quality of the route on fresh windows, on prompt-then-decode, and at 8k context. No 8k harness exists.
- Whether the tensor core accumulates an element the same way whatever its row in an MMA tile (d2's assumption).
- Whether candle-flash-attn builds in the image, and its causal alignment for seqlen_q < seqlen_k.
- The SASS load width of the rows kernels.
- Anything at 8B or 14B beyond scaling arithmetic, on sm_120, or on Metal.

## 10. Changes in revision 2 (review trail, to drop before any commit)

- The identity prior now comes from the gqa-ab table (2 of 6 reshapes), not from gqa-ppl's single 4,096 case. Priors
  fall from 0.5-0.7 to 0.1-0.25, and A1/A2 are T8 candidates whatever a grid says.
- Two exact variants added, A3 `InPlace(r)` (the reviewer's) and A4 `InPlaceAll` (in place on the whole buffer with
  candle's own kernels). Either one now carries the memory goal up to 40 heads with no T8.
- U1 narrowed to 2.21-2.55 s with the 8B and 14B windows; the TTFT algebra ties A to L4, so the route reads 2.85-3.4 s.
- The dense-against-fused pairing is labelled cross-job, reported, not published, and it indicates only.
- d2's fold moves into one header with explicit rounding intrinsics, one exp and one division.
- V1 gains the Linux type-check and the CUDA stride check; V3-lite rides an existing job; V4 is re-priced with its form
  named, its digest source fixed and an optional third arm for A2.
- The census caveat now names the attention form beside the head.
- Found in this pass, beyond the review: the triplet subtracts the 8k cache on its own line, and its 5 GB margin
  absorbs activations and the CUDA context (`note-produit-2026-08-13.md` §B bis); the `llvq-llm` cuda feature does
  not build on the Mac (served-grouped header), so the reviewer's `cargo check` becomes a `-p llvq-cuda` check with
  the unsafe calls moved there; the job script read the object from `/mnt/bucket` where jobs mount it at `/out`.
