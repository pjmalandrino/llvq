# Attention: our split-KV decode kernel and a real int8 KV store

Wave 5, lead `attention`, revision 2 of 2026-11-02, after an adversarial review (§10). Read-only: no repository file
changed, no run, $0 spent. Object: the sealed Qwen3-4B on one L40S, batch 1, greedy, grouped form served. Labels:
*measured* (named journal or primary source), *computed* (arithmetic on measured or code inputs), *estimated*
(judgment, or one estimated input). Beside this file: `prereg-attention-1.md`, `-1b`, `-2a`, `-2`, `-3`; drafts in
`code/attention/` (kernel, Rust replay, host probe harness, FlashInfer bench and job scripts, llama.cpp job script),
never compiled or run; arithmetic in `calc/attention/attention_v2.py`, run once with `python3 -I`.

## 1. Verdict

One split-KV kernel per layer would take the 4B's decode at 8,192 positions from 77.5 to about 106-114 tok/s with the
f16 cache, and to about 120-125 with a real int8 cache (*estimated*, η 0.60-0.85, before any other floor). At a
5-token prompt the lead's own share is ×1.02-1.04, 134.5 to about 137-140 tok/s. The q and k norms, RoPE and the KV
write are worth about 0.3 ms more; that fusion is the operator's branch (audit step 2), credited to whichever lands
first, and this lead gives it a bit-keeping design (§3.4).

The gain comes from two places. Today's context-dependent attention reads the cache at 0.25-0.26 of 864 GB/s at all
three lengths (*computed* from `served-grouped-2026-11-02.txt`). The kernel removes three to five launches a layer.

The kernel's arithmetic can be fully specified: explicit round-to-nearest intrinsics, a polynomial exp2, absolute
256-position splits, left folds. Decode, verify and optionally prefill rows then get the same attention bits on every
NVIDIA card. A CPU replay checks the kernel text on the Mac at $0, through the repository's host-probe pattern, and the
card at load. The kernel costs one T8 event at the 4B (f32 online softmax replacing cuBLAS and candle's f16 softmax),
about 5-8 weeks of code and about $6.4-7.3 of jobs for the 4B (*estimated*).

The int8 store adds no second T8 for storage: it returns today's `LLVQ_KV=q8` values bit for bit. The pair the lead
would serve, the kernel with a q8 cache, still needs one paired quality check, the third arm of the T8 control (V6).
It also needs the operator's band for q8: the only q8 measurement, +0.049 % [−0.071 ; +0.170]
(`kvq8-4b-2026-08-15.txt`), already crosses ±0.12 %.

Two speed unknowns decide: the kernel's read rate on Ada, and its verify cost t(k)/t(1). A recount puts the verify
cost at 1.0-1.7 at 8k and between 1.0 and 7 at 2k, depending on a mapping choice that changes no bit (*estimated*,
§3.7). The kernel-only bench measures both for about $0.2, right after the CUDA kernel exists and before any model
integration (V4a).

**First experiment:** FlashInfer's single-decode kernel on the same L40S and our geometry (`prereg-attention-1`, about
$0.15-0.42, no code of ours). It shares our 16-lane × 8-dim partition and our 256-position split floor, and runs with
a cold L2 and no host work. It sets the η band, the mapping set of build step 2, an input to the 14B's choice of cores
and a check on the int8 design. In parallel at $0: the CPU replay and the host probe of the kernel text.

## 2. What exists today

| where | what it does |
|---|---|
| `llvq-llm/src/kvq.rs:43-49`, `98-165` | `KvMode::Q8` is a quantize-dequantize round trip: group of 64 along `head_dim`, f16 scale and minimum, the result widened back to f16. The cache keeps its dtype. About 30 candle launches per tensor. |
| `kvq.rs:172-175`, test `330-344` | 147,456 B a token in f16, 78,336 in q8, ratio 1.882 (36 layers, 8 KV heads, 128). |
| `kvq.rs:414-427`, `470`, `488-558` | `KvStore::{Cat, Prealloc, Grow}`, `GROW_CHUNK = 256`, `AttnHeads::{Expanded, Shared, Grouped}`; the store rides the form. |
| `llvq-llm/src/model.rs:297-304` | `append` quantizes new rows under `Q8` and keeps their f16 reconstruction. |
| `model.rs:394-414`, `444` | `append_grow`: `slice_set` of the new rows, `narrow` views whose batch stride is the capacity; `truncate` is O(1) on `Grow`. |
| `model.rs:571-593` | `grouped_attention`: q as `[b, n_kv, G·l, 128]`, one cuBLAS matmul per KV head, scale, mask, candle softmax, second matmul. |
| `model.rs:1632-1647` | q_norm and k_norm (candle `rmsnorm`), then RoPE (candle `rope`), then the scale, as separate launches. |
| `model.rs:738`, `2191-2203` | `MAX_PREFILL_ROWS = 4096`; `prefill_chunked` returns the last position's logits only. |
| `model.rs:2243-2273` | `decode_nll_paired`: teacher-forced decode of one window under two (store, form) arms in lockstep. |
| `llvq-llm/src/bin/fusedrun.rs:534-620`, `679` | `LLVQ_GQA_PPL`: windows taken from token 0 (`:561`); measurement modes refused beside `LLVQ_CONFIG`. |
| `llvq-llm/src/fused_cuda.rs:61-71`, `2224-2262` | kernel sources honour `LLVQ_KERNEL_DIR`; load-time card parity check (`check_int4_half_parity`). |
| `llvq-llm/src/served.rs:110-111` | `LLVQ_KERNEL_DIR` and `LLVQ_TILE_BLOCKS` are refused beside `LLVQ_CONFIG` only. |
| `llvq-cuda/src/gpu.rs:14-16`, `270-277`, `448-461` | NVRTC at `compute_89`, `--fmad=true` by default; the driver's `num_regs`, `local_bytes` and shared bytes read at load (`report`). |
| `llvq-cuda/kernels/planes_occ.cu`, `occ_sk` | the split-K ticket fixup this lead reuses: fence and barrier before the atomic, `__syncthreads_or`, acquire fence, volatile reads, self-reset; an empty slice still draws its ticket. |
| `llvq-cuda/tests/host_shim.h`, `llvq-llm/tests/host_tv_q4_h.cpp` | the host probe: the kernel text compiled by clang++ (`-std=c++17 -O2 -ffp-contract=off`) and executed against Rust. |
| `ops/Dockerfile.cuda` | no nvcc step: kernels are `include_str!` sources handed to NVRTC at load, and the runtime image has no `cuda_fp16.h` (`matvec.cu`, `f2h`). |
| candle 0.9.2 | the f16 scale, softmax, RoPE and rmsnorm kernels that today's attention runs (§8, items 4, 5 and 26). |

What works: the grouped form reads each cached byte once, with the same 256 tokens as the expanded form at all four
lengths, and passed its tolerance (prefill +0.0000 %, decode +0.0095 % [−0.0048 ; +0.0238], top-1 99.788 %,
`gqa-ppl-2026-11-01.txt`). `Grow` gives O(1) rollback. A teacher-forced lockstep harness exists.

What is missing: a kernel that streams the cache (today's context-dependent work runs at about 220 GB/s at every
length); an int8 store; an 8k quality harness (one prefill call takes 4,096 rows, and windows start at token 0, where
every earlier control scored); a row-invariant attention for verify. Today's scores, scale and softmax also run in f16.

## 3. Design

### 3.1 The canonical arithmetic, frozen after the kernel-only bench

One per-row arithmetic, executed by every mapping. It is frozen after the kernel-only bench (V4a) chooses its unit
size, and before the T8 control (V6): no quality control runs on any variant before the freeze, so choosing among them
by speed biases nothing.

- **Lanes.** 16 lanes own one (row, query head); lane l owns dims 8l to 8l+7. Each partial dot is an `__fmaf_rn`
  chain over its 8 dims from +0, on f16 inputs widened exactly (`dot8`).
- **Score.** The 16 partials go through the tree of xor offsets 8, 4, 2, 1, each level a pairwise `__fadd_rn`
  (`tree16`). RN addition commutes, so every lane of a butterfly ends with the same bits. The score is multiplied once
  by c = scale × log2(e), rounded to f32 on the host.
- **Tile.** 8 positions per online-softmax update, aligned to absolute positions: tile maximum, one carry
  `ex2(m_old − m_new)`, eight `p = ex2(s − m_new)`, ts as a left fold in position order, d ← d·carry + ts, then each
  accumulator ← acc·carry followed by 8 `__fmaf_rn(p, v, acc)` in position order. Positions at or past L_r do no
  arithmetic and are never multiplied: a stale V row may hold inf.
- **Unit.** 64 positions, 8 tiles, aligned to multiples of 64 (V4a may choose 32). **Split.** 256 positions, aligned
  to multiples of 256. **Folds.** Left to right: units within a split, splits within a row. The merge is two
  `__fmul_rn` and one `__fadd_rn` per term; an empty operand (m = −inf) is skipped or copied.
- **Output.** f16 round-to-nearest of `__fdiv_rn(acc, d)`.
- **exp2.** A fixed polynomial: NaN propagates, x ≤ −126 or −inf gives +0, then floor, exact fraction, six
  `__fmaf_rn`, and an exponent add in unsigned arithmetic (a signed left shift of a negative int is undefined in
  C++17, and the host probe compiles this text). `ex2(+0)` and `ex2(−0)` both give 1, which makes a ±0 tie in a
  maximum harmless. The sketch's coefficients are Taylor placeholders, about 1e-5 relative error at f → 1; the frozen
  ones come from a minimax fit. MUFU `ex2.approx` is faster and gives card-specific bits.

**Exactness argument.** Every float operation is an explicit RN intrinsic, so no contraction choice of NVRTC or the
driver moves a bit. Which units, tiles and splits a row visits depends on its length L_r alone. The combine order
depends on split indices alone. Hence a row's output is a function of (q_r, K[0..L_r), V[0..L_r)) only, whatever the
number of rows in the call, the CTA shape or which CTA finishes last. With the polynomial exp2 the same holds across
cards, and a Rust replay (`code/attention/attn_ref_sketch.rs`: `f32::mul_add` is a fused multiply-add, `*` and `+`
are RN, `half::f16` rounds to nearest even) returns the card's bits.

**Bit-neutral, tuned on the bench:** which threads hold which heads (all G heads on one 16-lane group, or one head per
group as FlashInfer's `bdy = G`); the tree's schedule (allreduce, every lane ends with every score, or reduce-scatter,
each lane ends with 8G/16 scores of the same tree and computes their exp2 once); a CTA per split or per unit; stage
size and depth; CTAs per SM; widening tiles to f32 once per CTA when k > 1. **Bit-relevant:** TT = 8, the unit and
split sizes and alignments, the tree, the tile order, the polynomial. At 14B, if G = 5 spills at TT = 8, the 14B may
freeze TT = 4; each size has its own T8 anyway.

Against today's path the kernel is not bit for bit, and cannot be: cuBLAS's reduction order is not ours, and today's
scores, scale and softmax run in f16. That is the T8 event.

### 3.2 The decode and verify kernel

Signature (sketch, `attn_dec_f16_g4` and `_g5`, `code/attention/attn_dec_sketch.cu`): f16 is stored as `unsigned
short` and converted by `h2f` and `f2h` from `matvec.cu`, since the runtime image has no `cuda_fp16.h`. Arguments:
`q [k][n_heads][128]` after norm and RoPE; `k_new`, `v_new [k][n_kv][128]`, or null when the branch's fused prologue
has already written the rows; the cache `kc`, `vc [n_kv][cap][128]` (the `Grow` buffers); `pos_dev`, a u32 on the
card holding the cached length before the call; `k`, `cap`, `n_kv`, `c`; a partials scratch `[n_kv][s_max][8][G][130]`
f32; tickets `[n_kv]` u32 at zero; the output `[k][n_heads][128]`.

Geometry. Grid `(gridDim.x, n_kv)`: gridDim.x is the host's `ceil((pos0 + k) / 256)` in eager mode, 128 threads,
8 groups of 16 lanes.
- k = 1: groups compute units of the split in parallel and group 0 folds them in order. At 8k: 256 CTAs.
- 2 ≤ k ≤ 8 (verify): with a CTA per split, group r computes row r over its 4 units in sequence, with the same calls
  and hence the same bits; with a CTA per unit, the 8 groups take the 8 rows of one unit and the combine folds units,
  then splits.
- Fresh rows, when `k_new` is not null: the CTA owning the split of position pos0 + r writes row r's K and V to the
  cache and stages them from `k_new`, `v_new`, so no position is written by one CTA and read by another in the same
  launch. When the branch's prologue kernel has written them, every position comes from the cache, written by an
  earlier launch on the same stream.

Budgets (G = 4; *estimated*; the driver reports registers and spills at load, and the bench prints them):

| resource | k = 1 | k > 1 | G = 5 |
|---|---|---|---|
| registers a thread | ~130 | ~170 | ~160 / ~200 |
| shared, f16 store | 2 stages × 16 KB + 8.3 KB unit states ≈ 41 KB | 3 × 4 KB ≈ 12 KB | +2 KB |
| CTAs per SM | 2 | 2-3 | 2 |
| bytes in flight per CTA | ≥ 16 KB | ≥ 4 KB | same |

`__launch_bounds__(128, 2)` caps registers at 255. With 2 CTAs per SM and one 16 KB stage in flight each, 256 CTAs at
8k keep about 4 MB in flight and 64 CTAs at 2k about 1 MB, against about 0.5 MB that 864 GB/s at 0.6 µs of latency
needs (*computed*, Little's law, latency *estimated*). L40S limits: 102,400 B of shared memory per SM, 101,376 per block
on opt-in (*measured*, `docs/format-noyau.md` §8).

### 3.3 Combine

Every launched CTA draws one ticket on its KV head's counter, early exits included, so the last arrival is ticket
gridDim.x − 1 whatever the device length. A CTA with work first writes, per row and head, its split state (m, d,
acc[128]). Then every CTA fences, waits at a barrier, and thread 0 draws the ticket; `__syncthreads_or` tells the CTA
whether it drew the last one. The last CTA fences again, reads the partials through volatile loads, folds for each row
r the splits 0 to S_r − 1 in ascending order with S_r = ceil(L_r / 256) computed from `pos_dev`, writes the f16
output, and resets the ticket. That is `planes_occ.cu`'s `occ_sk`, including the barrier whose absence cost a job on
2026-09-01. One launch a layer, no second kernel. With a CTA per unit, two levels of tickets keep the nesting: the
last unit of a split folds its four units, the last split folder folds the splits.

At 8k the scratch is 8 × 32 × 4 × 520 B = 0.53 MB a layer written and read from L2, against 33.5 MB of cache
(*computed*). A separate merge kernel, as in FlashDecoding and FlashInfer (§8, items 11 and 14), would return the same
bits: the order is fixed either way.

A grid frozen at s_max = cap / 256 stays correct with this protocol. A graph-captured or persistent design would
also need tickets per layer or an epoch counter; nothing here builds either, and A2 stays out of the core (§7).

### 3.4 What the branch needs for the q and k norms, RoPE and the KV write

The operator's branch is fusing "QK-norm + RoPE + écriture KV" (audit step 2). The audit expected a T8 tolerance gate
for it. A design that keeps candle's bits needs none, and this is what it takes.

- **rmsnorm** (`reduce.cu:189-228`, 32-thread blocks below 1,024 columns, `ops.rs:545`): lane l sums x² over columns
  l, l+32, l+64, l+96 in that order, from 0; xor butterfly 16 to 1; `mean = tmp / 128`, exact; `rsqrtf(mean + eps)`
  with the same f32 eps; output `f16_rn((scale × x) × w)`, two RN products. One ambiguity: `tmp += xi * xi` may compile
  to an fma or to a multiply and an add. Two forms.
- **RoPE in f16** (`reduce.cu:292-312`): `dst[i1] = x1·c − x2·s` and `dst[i2] = x1·s + x2·c` in `__half`. Those
  products and sums carry no rounding modifier, so the driver may fuse either product of each output into an fma,
  independently for the two outputs. Three forms per output: no fma, fma on the first product, fma on the second.
- **Where the choice is made.** candle ships PTX compiled when the image is built, and the driver JITs it at load
  (driver 580.178.04 in `served-grouped`). Its contraction choices are the driver's, and a driver update can change
  them. The parity check therefore runs at every load, like `check_int4_half_parity`.
- **Traps.** `rsqrtf` is the MUFU approximation: replacing it by `1 / sqrtf` changes bits; keeping it gives the same
  bits on the same card. The cos and sin tables are read from the same f16 tensors, never recomputed with `sincosf`.
  NVRTC's default `--ftz=false` matches candle's build, which has no fast-math.
- **The check.** At load, every candidate form against candle's own `rope` and `rms_norm` on the card, on random rows
  (normal, subnormal and large values) and on rows of the first prefill. The first form that matches bit for bit on
  every row is used; none matches, candle's ops stay. Mutants the check must catch: the butterfly reordered, the wrong
  form forced, cos and sin recomputed, `1 / sqrtf`.
- **The KV write** is a copy: exact. If the fused prologue writes K and V into the `Grow` buffers, this lead's kernel
  drops its append duty.

The kernel-only bench carries the same probe at no extra cost (`prereg-attention-2a`, step 5), in case the branch has
not settled the question by then.

### 3.5 The int8 store

Layout, four `Grow(256)` buffers per layer: K codes `[n_kv][cap][128]` u8, K coefficients `[n_kv][cap][4]` f16
(s0, mn0, s1, mn1), and the same for V. 2 × (128 + 8) × 8 = 2,176 B a position and a layer, 78,336 B a token over 36
layers, as `kvq::bytes_per_token` counts (*computed*). Rows of 128 B and 8 B keep 16-byte and 8-byte alignment.

Read semantics R1 (recommended): the tile loader copies codes and coefficients into an int8 stage, then a warp pass
writes `f16_rn(rn(code × s) + mn)` into the same f16 tile the f16 kernel reads. These are, bit for bit, the values of
`kvq::quantize_dequantize`. candle computes the scale as `(mx − mn) × f32(1/255)` (`Tensor / f64` is `affine(1/v, 0)`,
`candle-core/src/tensor.rs:3044`), divides in IEEE, rounds half away from zero, clamps, and multiplies then adds in two
kernels. Its last step adds `graded_v · graded + mn · ungraded`, a signed zero for a graded group, which changes no
value. An ungraded group (its range rounds to 0 in f16) dequantizes to its minimum, except a minimum of −0, which
gives +0. Everything after the loader is the f16 kernel, unchanged. The fresh-row quantizer follows the same steps.
The attention of the call uses the dequantized values of what it just stored, never the raw f16, as `kvq.rs`
requires.

A ±0 tie at a group's minimum may store the other zero than candle's reduction does. No dequantized value changes, so
identity checks compare dequantized values, never raw coefficient bytes.

Consequences. The int8 store is a storage change with zero bits against "the kernel plus `LLVQ_KV=q8`": identity, not
T8, gates it. The prefill, while it stays on cuBLAS, dequantizes the history into a reused f16 scratch (33.5 MB a layer
at 8k) with the same formula, so the prefill path is also bit for bit today's q8 path. Cost on Ada: 4 instructions an
element, once per CTA, about 32 warp instructions per (position, KV head), 1.5-3 µs a layer at 8k against 26.5 µs of
int8 DRAM time (*estimated*). Semantics R2 (scores on codes, `s·(q·code) + mn·Σq`, magnitude's k8v4 form) is cheaper
and more precise, and would be a new T8; it waits for evidence that R1's instructions show.

Quality. q8 quality is the round trip's quality, measurable now with today's grouped attention and an 8k harness
(`prereg-attention-3`). The pair kernel plus q8 is scored by V6's third arm. The band is the operator's (§7).

### 3.6 Integration points in the Rust code

- `llvq-cuda/kernels/attn_dec.cu`: the kernel, assembled after `llvq_slot.cuh` and `matvec.cu`, with an entry in
  `embedded_source` and in `bin/cuhcheck`'s units, loaded through NVRTC honouring `LLVQ_KERNEL_DIR`. Its launcher sits
  in `llvq-cuda` too, which `CUDARC_CUDA_VERSION=12040 cargo clippy --target x86_64-unknown-linux-gnu -p llvq-cuda`
  type-checks on the Mac, so a type error does not cost a 17-minute image build; `fused_cuda.rs` reaches both as it
  reaches `rotate.cu`. The CPU replay stays in `llvq-llm`, where `llvq-cuda` is a Linux-only dependency. The host probe
  sits in `llvq-llm/tests` (`host_attn_dec.cpp` with `attn_dec_host.rs`, `code/attention/host_attn_dec_sketch.cpp`)
  and includes the kernel text from `llvq-cuda/kernels`, a dev-dependency there.
- `device.rs`: a fifth trait, `AttnKernel: Send + Sync`, with `max_rows()` (8) and `attend(...)`. Its CPU fake is the
  canonical replay, so host tests run the same arithmetic as the card.
- `kvq.rs`: `AttnHeads::Kernel`, parsed from `"kernel"` in the config's `attention` field, `store() =
  Grow(GROW_CHUNK)`. Every exhaustive `match` on the enum must be updated: fusedrun's A2 modes broke with E0004 the last
  time a variant was added (passation of 2026-11-01).
- `model.rs`, `Block::forward_cached`: for `l ≤ max_rows()`, reserve `Grow` capacity (split `append_grow` into a
  reserve and a write), call the kernel, which writes the rows, then `len += l`; for longer calls, today's grouped path,
  which writes the same bytes.
- `fused_cuda.rs`: scratch, tickets and the position word; the load-time checks (card against the CPU
  replay at L ∈ {1, 255, 256, 257, 1000, 4096} and k ∈ {1, 3, 8}; row invariance; a grid of s_max CTAs against the
  host's count) stop the load on any differing bit; the driver report of every entry point.
- `served.rs`: accept `"kernel"`, refuse it on Metal by name, as `"grouped"` is refused.
- `fusedrun.rs`: modes `LLVQ_ATTN_BENCH` (build step 2: synthetic sets, no artifact needed), `LLVQ_ATTN_AB` (step 3),
  `LLVQ_KVQ_PPL` (step 4: first-window index, 8k prefill by chunks of 4,096, a decode path that prefills 4,096 then
  teacher-forces 4,096, arms carrying a `KvMode`); all added to the refusal list at `:679` and to CLAUDE.md's list.
- No new binary, so the Dockerfile's two lists and `UPLOAD_ALLOW` do not change; `llvq-*/**` already uploads the kernel
  source.

### 3.7 CUDA cores or tensor cores, issue cost, verify cost, and what to expect

At k = 1 the kernel does G FLOP per byte of f16 cache: 4 at 4B and 8B, 5 at 14B, against an FP32 ridge of 106 FLOP/B
(91.6 TFLOPS over 864 GB/s, *computed*). The ridge says nothing about latency, which decides here.

**Issue cost, recounted** (*estimated*, `calc/attention/attention_v2.py` §4). Per lane and per tile of 8 positions,
the sketch's allreduce schedule issues about 144 + 343G instructions: 144 to load and convert the K and V slices, 64G
FFMA for the dots, 64G for the butterfly (a shuffle and an add in every lane), 117G for nine exp2 computed by all 16
lanes, 72G for the accumulators, 26G for the maxima, the scale, ts and d. That is about 95 warp instructions per (row,
position, KV head) at G = 4, 116 at G = 5. The reduce-scatter schedule keeps the same tree, computes each exp2 once and
broadcasts p through shared memory: about 57 (71 at G = 5). Widening tiles to f32 once per CTA brings k > 1 to about
50.

**k = 1.** Every unit runs in parallel, so a layer's compute time is one unit's chain: 11-19 µs with the allreduce
schedule, 6-12 µs with reduce-scatter, at 0.25-0.45 instructions a cycle per warp. DRAM time is 46-65 µs at 8,192 and
11-16 µs at 2,048 (η 0.85-0.60). The kernel is memory-bound at 8k and near balance at 2k.

**Verify.** At k = 8 with a CTA per split, four units run in sequence on 2-7 warps per SM: 43-77 µs (allreduce) or
23-41 µs (reduce-scatter) at every length. t(8)/t(1) is then 1.0-1.7 at 8,192 and 2-7 at 2,048. A CTA per unit spreads
the same work over 7-29 warps per SM and keeps t(8)/t(1) at 1.0-1.7 at every length (*estimated*). The mapping is
bit-neutral, so the bench chooses it. For the speculation lead, each extra verify row at 8k costs (t(8)/t(1) − 1)/7 of
about 2.0 ms: 0.03 to 0.43 ms for a ratio of 1.1 to 2.5, 1 to 16 % of the 2.75 ms per row its P3 profile charges for
the rest of the step (*estimated*). That profile assumes the attention adds nothing per row; V4a measures it.

**Tensor cores** at k = 1 would run 4 valid rows of 16 (5 of 16 at 14B) and break even with CUDA cores on raw FMA
rate; they win from G·k ≥ 16 and for prefill chunks. llama.cpp agrees on Ada: master `cd8427d` sends batch-1 decode
with an f16 cache to its CUDA-core vector kernel when the GQA ratio is at most 4, at any length, and when it is 5 below
8,192 positions; above, it uses its tensor-core kernel (`fattn.cu:656-665`). Its vector kernel runs one CTA per query
head (`fattn-vec.cuh:104-111`). FlashInfer 0.6.14's CUDA-core decode dispatches group sizes 1, 2, 3, 4 and 8 only, so
the FlashInfer anchor brackets G = 5 by 4 and 8 and tries its tensor-core path at 5 (§8, item 24). FlashInfer's
"40%+" for CUDA-core GQA came from the weak FP32 of A100 and H100 (§8, item 12). Choice: CUDA cores, one arithmetic
for k ≤ 8. Tensor cores only in a separate prefill variant (§3.8), or at 14B if the bench says so (D9).

Expected times, attention only, per token over 36 layers, with 36 × 3 µs of fixed cost (*estimated*):

| context | today, grouped (growth + 6 launches a layer) | kernel f16, η 0.60 / 0.75 / 0.85 | kernel q8, same η |
|---|---|---|---|
| 2,048 | 1.83 ms | 0.69 / 0.57 / 0.52 ms | 0.42 / 0.36 / 0.33 ms |
| 4,096 | 3.25 ms | 1.27 / 1.04 / 0.93 ms | 0.73 / 0.60 / 0.55 ms |
| 8,192 | 5.95 ms | 2.44 / 1.97 / 1.75 ms | 1.35 / 1.10 / 0.98 ms |

End to end, decode tok/s, with the 5-token step at 7.12 ms: the lead's own five launches a layer removed, prologue on
candle (*estimated*, `calc/attention/attention_v2.py` §3):

| context | today (*measured*) | kernel f16, η 0.60 / 0.75 / 0.85 | kernel q8, same η |
|---|---|---|---|
| 5 | 134.5 | 140 | 140 |
| 2,048 | 113.8 | 130 / 132 / 133 | 135 / 136 / 136 |
| 4,096 | 98.0 | 121 / 124 / 126 | 129 / 131 / 132 |
| 8,192 | 77.5 | 106 / 111 / 114 | 120 / 123 / 125 |

If the branch's fused prologue lands first and absorbs the KV write, this lead removes three launches a layer instead
of five, and its 5-token share is ×1.02 instead of ×1.04. Both gains together lower the step by about 0.6-0.7 ms in
either order; that sum is measured in one job, never added from two (METHODE §3).

The η here is the end-to-end slope, η_marg(L) = 147,456 × L / ((t(L) − t(5)) × 864 GB/s); today it is 0.259, 0.252
and 0.256 at 2,048, 4,096 and 8,192 (*computed*). The bench's η_k is the same quantity for one pass of 36 kernel calls
with a cold L2. At 14B (G = 5, 40 layers, 163,840 B a position, 1.342 GB at 8k) the kernel term is 2.2-2.8 ms at η
0.6-0.75 (*computed* from estimates); the 14B has not been re-timed since the trims, so no tok/s is projected.

### 3.8 Verify rows and prefill chunks

**Verify.** The same kernel serves up to 8 rows: row r has absolute position pos0 + r and sees L_r = pos0 + r + 1
positions. Its output equals the decode call at that position bit for bit, by §3.1, which is what exact greedy
speculation needs. Rejected rows are dropped by `truncate`, O(1) on `Grow`; stale rows past the length are never read.
The program's exact speculation also needs row-invariant projections, which this lead does not provide.

**Prefill, option P1: the same kernel.** Rows of a 4,096-row chunk go through the canonical arithmetic, 8 per CTA, so
that a tile is fetched once per 8 rows. Every row of every path then gets the same attention bits, the prefill stops
materializing the score tensors (6.44 GB at 8k today, audit ML of 2026-11-02), and "N rows in one call equal N calls"
holds for attention. Cost: the causal 8k prefill does 19.8 TFLOP of attention at the 4B (*computed*: 33.6 M position
pairs × 32 heads × 512 FLOP × 36 layers), 0.22 s at the FFMA peak. With 1.7-2.8 instructions per FFMA (§3.7) and 1.5-3
instructions a cycle per SM, about 0.5-1.6 s (*estimated*).

**Prefill, option P2: a tiled tensor-core variant.** FlashAttention-2-style tiles, 0.055 s at the 362 TFLOPS tensor
peak (*computed*; FlashInfer's table gives 366 TFLOPS with f32 accumulation for the RTX 6000 Ada), about 0.1-0.2 s in
practice (*estimated*). It breaks prefill-decode identity, as today, carries its own T8, and needs a go to lift the
2026-08-05 archive closure that calls tensor cores "sans objet" except "si le prefill entre au périmètre".

With the GEMM route, the prefill lead puts the time to the first token at 8k at 2.4-3.4 s with today's attention, of
which attention is 1.85-2.42 s (*estimated*, `wave5-deep/prefill.md`). P1 would land near 1.0-2.6 s and P2 near
0.6-1.2 s (*estimated*).

## 4. Deciding unknowns

| unknown | current best value | confirms (proposal) | kills (proposal; the operator decides) |
|---|---|---|---|
| D1. Our kernel's effect on the decode step | η 0.75 [0.60, 0.85] *estimated*; today 0.25-0.26 *computed*; cuBLAS FP16 streams at 0.76-0.78 on this card (`format-noyau.md` §7, *measured*) | in-situ 8k range low end ≥ 1.15 and 5-token median ≥ 0.99 (V4b) | in-situ 8k range entirely below 1.00 or 5-token median < 0.97, after one revision (V4b); η_k is a diagnostic only |
| D2. Short-context cost | ×1.02-1.04 at 5 tokens *estimated* | in V4b's pass line | in V4b's stop line |
| D3. Row invariance on the card | true by construction, never run | load-time check and bench, every variant, both grids | none: a mismatch is a bug, fixed before any figure is used |
| D4. Kernel text and card equal to the CPU replay | true with RN intrinsics and the polynomial exp2, never run | host probe bit for bit at $0 (V1), then the card at load | a mismatch with a found cause may switch to MUFU and a tolerance check; otherwise a bug |
| D5. Prologue parity with candle (the branch's) | unknown: the driver may fuse candle's f16 RoPE products, and possibly the f32 sum of squares | one form per output bit for bit at load (V4a step 5, or the branch's own job) | none matches: candle's ops stay, or the branch's fusion takes a T8 |
| D6. T8 quality, kernel against grouped | decode-path Δppl 0.00 % ± 0.03 % *estimated*; the prefill path is unchanged while it stays off the kernel | both decode intervals inside ±0.12 % at 4,096 and 8,192 (V6); GSM8K within the operator's bar (V7) | an interval outside the band on one side |
| D7. q8 at 8k on the sealed object | +0.049 % [−0.071 ; +0.170] at 4k on the paper-1 file, Metal dense (*measured*) | accept line of the ruled band's rule set (V5) | reject line of that rule set |
| D8. What a 16-lane CUDA-core GQA decode reaches on this card | unknown; FlashInfer E8 predicted 0.75 [0.50, 0.92] | E8 ≥ 0.70 keeps the band [0.60, 0.85] (V2) | E8 < 0.45 lowers it to [0.30, 0.50] and puts the build order to the operator; no kill |
| D9. G = 5 on CUDA cores | memory-bound by §3.7 *estimated* | η_k(8k) at G = 5 over G = 4 ≥ 0.85 (V4a); FlashInfer H ≥ 0.85 (V2) | below: the 14B's choice goes to V9, with the tensor-core arms as input |
| D10. Verify cost t(k)/t(1) | 1.0-1.7 at 8k; 1.0-7 at 2k by mapping *estimated* | r_8(8k) ≤ 1.5 for the chosen mapping (V4a) | above: the speculation lead's slope is recomputed; no kill |
| D11. The served pair, kernel plus q8 | never scored | V6's third arm on the accept line of the ruled band at 4k and 8k, and V5 accepted | V6's third arm on the reject line: no q8 default |

## 5. Validation ladder

From cheapest to dearest. Every paid step needs the operator's go, a cap, a stamped prereg, and the cost announced
before and the running total after (rules 1 and 2). l40sx1 costs $1.80/h. Prices are *estimated*
(`calc/attention/attention_v2.py` §9).

| step | settles | method | cost | prereg | decision rule |
|---|---|---|---|---|---|
| V0 | code facts, budgets | this note | $0 | no | done |
| V1 | D3, D4 for the text, the R1 identity, the edges | CPU replay and host tests; the host probe compiles the kernel text with clang++ and diffs `ex2_fixed`, `dot8`, `tree16`, `tile_*`, `st_merge`, `deq_r1`, `quant_codes` against the replay bit for bit; one probe build under `-fsanitize=undefined`; `cuhcheck` and the cross-target clippy of the launcher; mutants (attn_ref_sketch) | $0, local, `nice` and `cargo -j 4` | no | all pass and every listed mutant dies → step 2; otherwise fix |
| V2 | D8, D9 bracket, the mapping set, the int8 proxy | FlashInfer single decode in the pinned vLLM image, arms at G 4, 5, 8, f16 and e4m3, CUDA and tensor cores | $0.15-0.42, timeout $0.75 | `prereg-attention-1` | four readings change the build plan; no kill |
| V2b | an engine-level slope and time to first token | llama.cpp depth sweep, optional, image pinned by digest | $0.12-0.15, timeout $0.60 | `prereg-attention-1b` | informs decision 9 and the engine table; no action of this lead |
| V3 | dropped | registers and spills come exact from the driver at load (V4a); the prologue forms are chosen by the on-card parity probe (V4a step 5 or the branch) | | | |
| V4a | η_k, r_k for k ∈ {1, 2, 4, 8}, the mapping, the unit size, registers, D5, D9 input, D10 | kernel-only bench in `fusedrun` (`LLVQ_ATTN_BENCH`), after build step 2, iterated through `LLVQ_KERNEL_DIR` | $0.18-0.24 a run, timeout $0.75 | `prereg-attention-2a` | readings choose the mapping and freeze the arithmetic; no gate |
| V4b | D1, D2, D3 in the model | in-situ A/B at 5/2k/4k/8k after build step 3 | $0.39-0.48, timeout $1.35 | `prereg-attention-2` | identity fail → bug; 8k range entirely below 1.00 or 5 tokens < 0.97 → stop proposal; 8k low end ≥ 1.15 and 5 tokens ≥ 0.99 → V6; any other result → one revision, diagnosed by η_k; otherwise not settled |
| V5-pre | D7, screen only | q8 against f16 at 8k on fresh windows, Mac Metal dense reconstruction, after build step 4's harness reaches the Metal path | $0, 1-2 h of the Mac under `nice`, `LLVQ_THREADS` ≈ ncpu−4; needs the operator's go for a local run | no | the ruled band's reject line → the operator is told before V5 is paid; any other result → V5 as planned |
| V5 | D7 | q8 round trip against f16 at 8k, grouped form, 24 fresh windows | $1.54, timeout $2.70 | `prereg-attention-3`, band ruled first | one rule set per band |
| V6 | D6, D11 | T8 control, three arms in lockstep: grouped + f16 (served), kernel + f16, kernel + q8 round trip; decode path at 4k (12 fresh windows, 61-72) and 8k (6 windows, 7-12); the kernel arms' prefill path skipped, since it is the grouped arm's while rows above 8 stay off the kernel, with one window checking that bit for bit; the harness gate on the served arm | $1.69, timeout $3.60 | yes, after V4b | kernel against served: both decode intervals inside ±0.12 % → V7; one entirely outside the band → not served (proposal); any other result, an interval crossing a bound → power, operator. Pair against served: the ruled band's rule set. A gate failed → not settled |
| V7 | D6, tasks | GSM8K through the kernel, 1,319 problems, paired with the expanded dump (and the grouped one if paid) | $1.9-2.1; anchor `gsm8k-4b-served` $2.13 for 71 min | yes | inside the operator's bar → T8 decision to the operator; outside → not served; otherwise not settled |
| V8 | int8 speed, identity | int8 store against the kernel's round trip: dequantized values and logits bit for bit, then in situ at 2k/4k/8k | $0.55 | yes | identity fails → bug; V5 or V6's pair not accepted → memory-only option at most; 8k ratio over the f16 kernel ≥ 1.05 → q8 default to the operator; below 1.05 → memory-only option; a gate failed → not settled |
| V9 | 8B, 14B, D9 | per size: in situ, T8 control, GSM8K; the 14B's cores chosen from V2, V4a and its own in-situ figure | $4-7 a size; anchors GSM8K `8b` $2.37, `14b` $4.24 | yes, per size | as V4b, V6, V7 |

Cheaper experiments that hold up, and where they sit: the FlashInfer anchor replaces the llama.cpp sweep as the first
experiment (V2, V2b now optional). The host probe moves the first execution of the kernel text from a paid card to the
Mac (V1). The kernel-only bench measures η_k and the verify cost right after the CUDA kernel exists, with registers and
spills from the driver and the prologue probe in the same job (V4a, replacing V3). V6 drops the kernel arms' prefill
path, which is bit for bit the grouped arm's, and spends part of the saving on the pair's arm: $1.69 against $2.13 for
the first design. V5-pre may spare V5.

V1 and V5-pre cost no money; V2, V2b and V5 need no kernel of ours and can run while it is written. The 4B part of the
ladder costs about $6.4-7.3 (V2, V4a once or twice, V4b to V8), plus $0.12-0.15 if V2b runs; the 8B and 14B about
$8-14 more.

## 6. Build plan after validation

1. `AttnKernel` trait and its CPU replay; host tests: row invariance at L ∈ {1, 7, 8, 9, 63, 64, 65, 255, 256, 257,
   1000, 4096} and k = 1 to 8, f64 agreement within 2e-6, R1 identity with `quantize_dequantize` on the CPU backend
   (edge groups included). The host probe of the kernel text, `host_attn_dec.cpp`, from the first line of kernel
   written. Mutants listed in `attn_ref_sketch.rs`. 3-5 days, $0.
2. CUDA kernel, f16, G = 4 and 5, variants V-A, V-B, V-C and V-B32 (`prereg-attention-2a`), the launcher, the
   load-time checks and `LLVQ_ATTN_BENCH`. First compilers: clang++ through the host probe and `cuhcheck` for the kernel
   text, the cross-target clippy for the launcher, both on the Mac at $0; NVRTC on the card at load. The image build
   compiles the Rust side only, about 17 min a build. 1.5-2.5 weeks. Then V4a,
   one iteration through `LLVQ_KERNEL_DIR` if its readings ask for it, and the freeze.
3. Form `"kernel"`, served-config field, Metal refusal, `LLVQ_ATTN_AB`, E0004 sweep. 3-5 days. Then V4b.
4. Harness: first-window index, 8k prefill by chunks, decode path, `KvMode` arms (`LLVQ_KVQ_PPL`). 2-3 days. V5-pre and
   V5 can run after it.
5. V6, then V7. Gate: the operator's T8 decision; then `configs/qwen3-4b-tetra-e4-kernel.json`, a new file, with a
   reference decoded line checked the way `served-grouped` checked its own.
6. int8 store, R1: four buffers, quantizer, tile dequant, prefill dequant scratch; gate: identity of dequantized values
   with the round trip on the card. 1-1.5 weeks. Then V8.
7. 8B and 14B configs (G = 5 template), V9.
8. Prefill: P1 or P2, chosen by the operator from §3.8, each with its control.

The q and k norms, RoPE and the KV write are the branch's (§3.4). If its fused prologue writes K and V, step 2's
append is skipped through a null `k_new`. Code in all: about 5-8 weeks for one person (*estimated*), consistent with
the program's floor 3 (4-6 weeks, plus 2-3 for the int8 option).

## 7. Rules map

| event | exactness | check | ruling |
|---|---|---|---|
| kernel on decode and verify rows, 4B | T8: f32 online softmax against cuBLAS and f16 softmax | V6 and V7 | a T8 decision of its own; the grouped form's licence does not extend (D12 of the wave-3 digest) |
| kernel plus q8 cache, the served pair | quality change | V5 (prefill path) and V6's third arm (decode path), each in one job | the band (B1 ±0.12 % or B2 P3's) and the default are the operator's; `kvq8-4b`'s own verdict kept q8 off by default for want of long-context evidence |
| int8 store R1 | bit for bit with the q8 round trip, on dequantized values | V8 identity | none beyond the q8 band |
| kernel on prefill chunks (P1) | T8 of the prefill path | in V6 if built before it | as the first row |
| tensor-core prefill (P2) | T8 | own control | go to lift the 2026-08-05 archive closure (rule 1; not an ETAT §6 row, so rule 3 does not bind) |
| fused q and k norms, RoPE, KV write (the branch) | bit for bit with candle when one form matches at load, candle fallback | load-time parity, every load | the branch's; replacing candle ops is decision 12 of the program report |
| 8B and 14B | T8 per size; the kernel can replace the expanded form directly, one event a size instead of two | V9 | per size |

- **A2.** Nothing here captures a graph. The kernel is launched normally on `Grow`, and A2 stays out of the core
  (rule 3, ETAT §6 and line 212). Every CTA draws a ticket, so a grid frozen at s_max stays correct; that is one
  precondition of a graph or a persistent kernel, not a sufficient one, and nothing here builds either. A served q8
  default would meet the letter of one written reopen condition of A2 (ECARTS §É7, "si le cache KV passait en q8");
  even then ETAT keeps A2 out of the core unless the operator rules otherwise.
- **Rule 6.** A cache is not a parameter (`kvq.rs:84-89`): its size is stated in bytes a token with the geometry,
  78,336 against 147,456 B at the 4B. Its effect on the triplet is stated in b_max, kernel b/weight: 3.00 with a q8
  cache, 2.86 with f16.
- **Rules 5 and 7.** FlashInfer and llama.cpp are compared in GB/s only; our ratios are formed round by round, medians
  with ranges.
- **ROADMAP §4** holds "q8 KV cache at long context" and names this very remedy; starting it needs a go. The prefill
  options touch "Batch above 1, and prefill", also on hold: decision 9 of the program report.
- **METHODE.** Benches inform, the end-to-end gates decide, every stop is a proposal, every rule above ends in an
  otherwise, one mechanism per A/B (kernel against grouped at equal cache mode; q8 against f16 at equal form), and the
  served pair is scored against the served object inside one job (§3).

## 8. Checked now at $0

1. `KvMode::Q8` keeps the reconstructed f16 in the cache; no byte is saved (`kvq.rs:98-110`; `model.rs:297-304`).
2. 147,456 and 78,336 B a token at the 4B, ratio 1.882, pinned by a test (`kvq.rs:330-344`).
3. `Grow` appends by `slice_set` and hands `narrow` views whose batch stride is the capacity (`model.rs:394-414`); its
   `truncate` moves a length only (`model.rs:444`).
4. Today's grouped path: one cuBLAS matmul per KV head over `[b, n_kv, G·l, 128]` (`model.rs:571-593`); the scale is an
   `affine_f16` in `__half` (`affine.cu:42`); candle's softmax takes max and exp in `__half` with `hexp`, sums in f32,
   one warp per row (`reduce.cu:233-273`, `cuda_utils.cuh:177`, `ops.rs:357-360`).
5. q_norm, k_norm and RoPE are four candle launches a layer (`model.rs:1632-1646`); candle's rmsnorm uses 32-thread
   blocks below 1,024 columns (`ops.rs:545`) and a xor butterfly 16 to 1 (`reduce.cu:115-120`); RoPE runs in the
   tensor's type (`reduce.cu:292-312`).
6. candle's `Tensor / f64` is `affine(1/v, 0)` (`tensor.rs:3044`), and its float `round` is `roundf`, half away from
   zero (`unary.cu:208`, `cuda_utils.cuh:138`): the q8 quantizer is reproducible with RN intrinsics and on the host.
7. NVRTC compiles at `compute_89` with `--fmad=true` by default (`gpu.rs:14-16`, `270-277`); explicit intrinsics are the
   only way to pin contraction.
8. `MAX_PREFILL_ROWS = 4096` (`model.rs:738`); `run_gqa_ppl` takes windows from token 0 (`fusedrun.rs:541-561`);
   WikiText-2 test holds 299,078 tokens (`ppl-scelles-2026-10-04.txt:27`), so 36 windows of 8,192, of which windows 1-6
   hold the 12 windows of 4,096 every control so far scored.
9. Today's context slope: 223.7, 218.0 and 220.8 GB/s at 2,048, 4,096 and 8,192, η 0.259, 0.252, 0.256 (*computed* from
   `served-grouped-2026-11-02.txt`: 7.43, 8.78, 10.20, 12.90 ms a token). The judge's "221-233 GB/s" upper value does not
   reproduce, as the wave-3 digest found.
10. `gqa-ppl-2026-11-01`: prefill bit for bit at 4,096 rows, decode +0.0095 % [−0.0048 ; +0.0238], 49,036 of 49,140
    top-1, max |dlogit| 13.13; about 123 s a window of 4,096 for two arms and both paths (job log). The q8 numbers of
    `kvq8-4b-2026-08-15`: f16 16.9415, q8 16.9499, +0.0494 % [−0.0713 ; +0.1703]; MMLU 56.09 against 55.76; q8 5-7 %
    slower at short context; the n_new 1,024 series abandoned at 661 s; all on `~/qwen3-4b-llvq.bin` (sha256
    `9db213ef…`), Metal on an M3 Max, dense path, about 2 h 45 of the Mac. Its interval half-width, 0.12 %, is the bar
    ETAT §4 cites.
11. FlashInfer `main` (`f00ee26`, 2026-11-02), single decode at group 4, f16, d = 128: vec 8, bdx 16, bdy 4, 128 threads,
    bdz 2, 2 shared stages on sm_80+, about 8.3 KB of shared memory (`decode.cuh:635-698`, `utils.cuh:333-340`); chunk =
    max(ceil(L / max_chunks), 256), merged by LSE in chunk order (`decode.cuh:704-746`, `state.cuh`).
12. FlashInfer blog: bandwidth utilization is bytes read over latency over peak, cold L2; MHA decode "close to 100%" on
    the RTX 6000 Ada and RTX 4090; split-KV gives nothing on Ada for MHA; the CUDA-core GQA kernel "40%+" on A100 and
    H100; RTX 6000 Ada spec 142 SMs, 960 GB/s, 90 TFLOPS FP32, 366 TFLOPS tensor at equal f16 and f32 accumulation
    (local copy `wave3-verify/w7-flashinfer/blog.txt`).
13. llama.cpp master (`cd8427d`, 2026-11-02): `FATTN_KQ_STRIDE` 256 (`fattn-common.cuh:9`); the Ada vector-kernel rule of
    §3.7 (`fattn.cu:656-680`); 128 threads (`fattn-vec.cuh:4-11`); f32 accumulation on NVIDIA (`common.cuh:771-773`);
    parallel blocks chosen by occupancy and wave efficiency (`fattn-common.cuh:1136-1215`). `llama-bench` takes
    `-d`, `-fa 1` and writes `samples_ns` in JSONL (`tools/llama-bench/llama-bench.cpp:614`, `865`, `1843`).
14. FlashDecoding: split, per-split attention writing one LSE per row, a separate reduction kernel (crfm.stanford.edu,
    2023-10-12).
15. magnitude (local copy, 2026-10-29 snapshot): split-KV decode, one block per (KV head, partition, row); k8v4 history
    scored on codes and values accumulated with the bias folded at the end; partitions are equal shares of each row's
    total, so their boundaries move with the length; merge in partition order; PDL, which needs sm_90
    (`attention_decode_k8v4.cu`, `decode_merge.cuh`).
16. Both GGUFs of the llama.cpp sweep are in the bucket (`m4-iq2-cuda-1b57b7d3/`); the earlier job ran `llama-bench -p 0
    -n 128 -r 5 -ngl 99`, no `-fa`, no `-d` (`hf jobs inspect 6a94a1d4…`); no depth sweep exists in the bucket.
17. Registry: 237 rows, $248.84 (*computed* from `docs/data/jobs.csv`, 5 rows unpriced).
18. `ops/Dockerfile.cuda` has no nvcc, ptxas or cuobjdump step (`FROM nvidia/cuda:12.4.1-devel` at line 21 builds the
    Rust side only); kernels are `include_str!` sources handed to NVRTC at load (`llvq-cuda/src/lib.rs:230-334`,
    `fused_cuda.rs:61-71`). The runtime image carries no `cuda_fp16.h` (`matvec.cu`, comment on `f2h`).
19. The host-probe pattern: `llvq-cuda/tests/host_shim.h` turns CUDA keywords into C++; probes are compiled by
    `clang++ -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Werror` (`tetra48_matches_rust.rs:85-111`); llvq-llm has
    its own (`host_tv_q4_h.cpp` with `proj_q4.rs`, f16 through `_Float16`). The shim's `__shfl_xor_sync` takes three
    arguments and returns its input; `bin/cuhcheck` supplies a 4-argument stub for its parse.
20. `planes_occ.cu`'s `occ_sk`: an empty slice still draws its ticket; fence, barrier, `atomicAdd`, `__syncthreads_or`,
    acquire fence, volatile reads, self-reset; a missing barrier cost the job of 2026-09-01 ($0.15), documented there.
21. The driver's `num_regs`, `local_bytes`, shared bytes and binary version are read at load (`gpu.rs:448-461`).
22. `LLVQ_KERNEL_DIR` is refused beside `LLVQ_CONFIG` only (`served.rs:110-111`, `BELOW_THE_DOOR`).
23. vLLM v0.26.0's `docker/Dockerfile` installs `flashinfer-jit-cache==0.6.14` and `cuda-nvcc` "for runtime JIT
    compilation" on a CUDA base image; its `requirements/cuda.txt` pins `flashinfer-python` and `flashinfer-cubin`
    0.6.14 and `torch==2.11.0` (GitHub, tag v0.26.0, read 2026-11-02). The repository's vLLM jobs pass that tag, with
    manifest `sha256:ffb2d59b…` recorded (`awq-vllm-4b-2026-08-17.txt`); the pinned digest's own content is checked by
    the job, not here.
24. FlashInfer v0.6.14 (read 2026-11-02): `vec_size = max(16 / sizeof(KV), HEAD_DIM / 32)`, `bdx = HEAD_DIM / vec_size`;
    128 threads except at group 8; partitioned path above 256 positions with `kv_chunk_size = max(ceil_div(seq_len,
    max_num_kv_chunks), 256)`, then `MergeStates`; `DISPATCH_GQA_GROUP_SIZE` accepts 1, 2, 3, 4 and 8 (main also 6
    and 16); `single_decode_with_kv_cache(q, k, v, kv_layout, pos_encoding_mode, use_tensor_cores, …)`, where
    `use_tensor_cores=True` routes to the FA2 single-prefill kernel (`decode.cuh`, `utils.cuh`, `decode.py`).
25. The llama.cpp `full-cuda` image failed twice at container start on 2026-09-25 (`MOUNT_ATTR_IDMAP`), $0 each, and
    passed an hour later (`docs/data/jobs.csv`); the repository's llama.cpp jobs passed the floating tag.
26. candle 0.9.2's rmsnorm sums `tmp += xi * xi` over columns l, l+32, l+64, l+96, then a xor butterfly 16 to 1,
    `mean = tmp / ncols`, `rsqrtf(mean + eps)`, output `static_cast<T>(scale * x * a)` (`reduce.cu:189-228`); its RoPE
    computes `src[i1] * c - src[i2] * s` and `src[i1] * s + src[i2] * c` in `T` (`reduce.cu:292-312`).
27. The q8 round trip runs about 30 candle launches per tensor and ends with `graded_v * graded + mn * ungraded`
    (`kvq.rs:110-165`): an ungraded group whose minimum is −0 dequantizes to +0 (*computed* from the code).
28. Metal and CUDA give the same perplexity on a quantized file: paired over 12 windows, [−0.009 %, +0.010 %] (*measured*,
    8B base, `ppl-bases-carte-2026-10-05.txt`).
29. `llvq-llm` depends on `llvq-cuda` only on Linux under the `cuda` feature, and as a dev-dependency for its tests
    (`llvq-llm/Cargo.toml:25-29`, `55-68`); `llvq-cuda`'s Linux half type-checks on the Mac through the cross-target
    clippy of CLAUDE.md's command list. NVRTC receives the architecture and no other option (`gpu.rs:266-282`), so
    `--ftz=false`, `--prec-div=true` and `--fmad=true` are its defaults.

## 9. Not established

- The kernel's η on any card, its register count and spills, its fixed cost per layer, and its verify cost.
- Whether the bit-neutral variants are as far apart as §3.7's model says; the model's IPC per warp is a judgment.
- The polynomial exp2's cost at k = 8, and its accuracy once the coefficients are fitted.
- Candle's contraction choices under the driver's JIT, hence the prologue's parity (the branch's question).
- The FlashInfer version inside the pinned image digest: the tag's files say 0.6.14, and the job checks.
- q8 quality at 8,192 on the sealed object, and the pair kernel plus q8.
- GSM8K under the kernel; whether the 13.13-logit spikes recur and how often.
- The A100: at 19.5 TFLOPS FP32 its ridge is near 10-12 FLOP/B, and the recount makes an issue-bound kernel there more
  likely (*estimated*); sm_120 and the 5090; Metal, which has no such kernel.
- The prefill options' costs (P1 0.5-1.6 s, P2 0.1-0.2 s at 8k) and the attention share of today's 35 s, a fit.
- Any 8B or 14B speed after the trims.

## 10. Review response

The adversarial review of revision 1 found one high and nine medium problems; each is fixed in the text above.

| finding | change |
|---|---|
| high: `prereg-attention-2` gated on the bench's η_k | the speed rules read in-situ ratios only (V4b); η_k diagnoses the revision; D9 moved to V9 with η_k as input |
| graph and persistent claims broke the ticket combine | every CTA draws a ticket, `occ_sk` pattern (§3.3, sketch); the claim reduced to a precondition (§7) |
| verify cost never measured, issue count low | recount (§3.7, about 95 and 57 warp instructions); V4a times k ∈ {1, 2, 4, 8}; D10 |
| the first compiler was said to be the image build | the host probe runs the kernel text on the Mac (V1, harness draft); NVRTC on the card second |
| V3 could not choose the RoPE form | V3 dropped; driver report at load; on-card parity probe in V4a or the branch (§3.4) |
| D1 was first measured after 3-5 weeks of code | V4 split: a kernel-only bench (V4a, `prereg-attention-2a`) after build step 2, iterated through `LLVQ_KERNEL_DIR`, then the in-situ A/B (V4b) |
| `prereg-attention-3`'s rules did not fit B2 | one rule set per band, each partitioning its outcomes; rejections phrased as proposals |
| the served pair, kernel plus q8, never scored | V6's third arm; D11; verdict reworded |
| the first experiment changed no action, on a floating image | FlashInfer anchor with four action-bearing readings (`-1`); llama.cpp optional, pinned by digest, infra failure noted (`-1b`) |
| the prologue gain overlapped the branch | credited to the branch; the lead's own 5-token share ×1.02-1.04; prefill path named and equal in both arms of V4b, V5, V6 |

Smaller points fixed: P1 recosted (0.5-1.6 s); V6 recosted ($1.69 for three arms, the first design was $2.13); the
±0 edges of `deq_r1` and NaN in `ex2_fixed`; the signed shift in `ex2_fixed`; the Taylor placeholders labelled.
