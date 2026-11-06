# Metal serving v2: plan

Written on 2026-11-06 after M7b to M13. Proposal, not adopted: every step below waits for the operator's go. Numbers
are labeled; the journal is `docs/mesures/metal-nuit-2026-11-06.txt`.

## 1. Where the sealed 4B stands on the M3 Max

- Decode after a 5-token prompt: 65.9 tok/s on a cool machine (M12), 54.4 in a loaded window (M8). *Measured.*
- Decode at 4,096 tokens of context: 40.2 tok/s (M10). *Measured.*
- Reading a prompt: 130 tokens/s at 4,096 tokens (M10). *Measured.*
- The token is GPU-bound: the host spends 5.4 ms of a 15.2 ms token (M12). *Computed.*
- Floor: 1.38 GB read a token. At 400 GB/s nominal that is 3.5 ms, at a realistic 300 GB/s 4.6 ms, about 220 tok/s.
  *Computed.* We run at about 90 GB/s of weights.

Where a short-context token goes, from the per-class costs (M7b, M13), *computed*:

| class | ms a token | share | what reads it |
|---|---|---|---|
| Tetra matvec | 9.0 | 58 % | 749 MB at 85 to 90 GB/s |
| int4 matvec | 2.4 | 16 % | 409 MB at 82 to 232 GB/s |
| rotations | 1.95 | 13 % | 108 launches, 18 µs each |
| attention half | 1.25 | 8 % | 8.6 ms at 4,096 positions |
| norms, activation, residuals | 0.87 | 6 % | |

These are costs measured class by class, and a class's 36 layers are independent there, so they may overlap: the split
is a guide, not an exact sum.

## 2. Why it is slow, cause by cause

1. **The Tetra matvec is decode-bound.** Each 24-weight block costs three dependent rank-table gathers, three code
   lookups and 24 value lookups before its 24 multiply-adds. Prefetching, several rows a simdgroup and staging the
   tables in threadgroup memory all failed to move it (+3 % at best, x0.84 for the tables). The memory is idle.
2. **The rotation runs on one core.** `rot_apply_metal` gives a whole vector to one threadgroup and runs every
   Walsh-Hadamard stage through device memory with a device barrier. `rot_apply_tg_metal` stages in threadgroup memory
   for widths up to 8,192, which is 72 of the 108 rotations, and is not wired.
3. **Long-context attention uses a GEMM for a GEMV.** In decode a key head sees 4 query rows. candle's gemm takes its
   smallest tile, 32 rows, and its split-K path is disabled (`should_use_split_k` is dead code), so the product with V
   runs as 32 threadgroups for 40 GPU cores, each walking 4,096 positions alone: 76 GB/s of KV. The existing split-KV
   kernel (`attn_dec.metal`, `+ker`) read only x1.04 in M6 and was written for exact bits, not speed.
4. **candle's eager path issues about 1,500 operations a token**, each with an allocation from the pool and hazard
   tracking. The host's 5.4 ms is hidden today; it becomes the wall once the GPU work drops below it.
5. **Independent work may be serialized.** q, k and v read one rotated vector, gate and up another. Nothing forces them
   to run one after the other, and whether candle's encoders let them overlap in the model is not measured.

## 3. The design

Three layers, from the cheapest to the most uncertain.

### A. Quick wins in the current path (hours, bit for bit or under the M6 tolerance)

- Wire `rot_apply_tg_metal` for widths up to 8,192. Same arithmetic, so the same bits; gated by the existing
  rotation test.
- A decode attention kernel split along positions (flash-decoding): each threadgroup takes a slice of the cache for
  one key head and its 4 query heads, reads K and V once, and the slices merge by their max and sum. It plugs into
  `AttnKernel` in `attn_metal.rs`, with no change to the model. Not bit for bit (another summation order), so it is
  measured against the M6 band (+-0.12 % ppl). First, profile `attn_dec.metal` through `attnprof` to learn why it read
  x1.04.

### B. An engine of our own: one command buffer a token

Leave candle out of the decode loop on Metal:

- Every pipeline compiled and every buffer allocated at load: activations, scratch, the KV cache at its capacity.
- One command buffer and one concurrent compute encoder a token, with explicit memory barriers between dependent
  stages, and untracked resources. Independent dispatches (q, k, v; gate and up) overlap.
- Fused kernels, about 10 dispatches a layer instead of about 40:
  - RMSNorm and rotation in one kernel;
  - q, k and v in one dispatch over the rotated vector;
  - q/k RMSNorm, RoPE and the cache write in one kernel;
  - flash-decoding attention;
  - o_proj with the residual add in its epilogue;
  - gate and up with SiLU and the product in the epilogue;
  - rotation, then down_proj with the residual add;
  - the int4 head with the argmax, so greedy decode reads back one token instead of 600 KB of logits.
- The engine sits behind a `metalrun` arm and the same gates as today: bit for bit where the arithmetic is the same,
  the M6 tolerance where it is not. The model, CUDA and the served configs do not move.

### C. The Tetra matvec, as a research track

The largest single cost and the least certain lever. Leads not yet tried:
- decode two blocks a lane interleaved, so the dependent gathers of one hide behind the arithmetic of the other;
- replace the per-coordinate lookups by a table of decoded byte quads per rank row, read once a section;
- a load-time re-encoding into a layout cheaper to decode, at a measured cost in bytes (b/param stated for the whole
  model, rule 6).

### D. Later: speculative decoding

Batching rows is cheap in this format (the decode of a block is paid once for 8 rows), so verifying several draft
tokens costs less than decoding them. Prompt lookup first (no draft model), a small draft model after.

## 4. Budget a token, estimated

| | now | after A | after A+B | after A+B+C (Tetra at 150 GB/s) |
|---|---|---|---|---|
| short context | 15.5 ms (65 tok/s) | about 14 ms | about 11 ms (90 tok/s) | about 7.5 ms (130 tok/s) |
| 4,096 positions | 24.9 ms (40 tok/s) | about 18 ms | about 14 ms (70 tok/s) | about 10.5 ms (95 tok/s) |

*Estimated*, from section 1's split and the floors of each kernel. Each row is a hypothesis for the step's gate, not a
promise.

## 5. Order, gates, cost

All local, $0.

0. **The bar.** Measure mlx-lm's Qwen3-4B 4-bit on this Mac, short and at 4,096 positions: the rival a Mac user
   already has. It needs a download of about 2.3 GB (operator's go).
1. **A** with one prereg: rotation, then attention. Gate: same tokens (rotation), M6 band (attention).
2. **B, one layer first.** The engine on one decoder layer, standalone, against the same layer through candle. Gate:
   at least x1.3 a layer, else B stops and the reason is journaled.
3. **B, the whole model** behind an arm; gates as above; then the bundle.
4. **C** in parallel with 2 and 3, by microbench before any arm.

## 6. Decisions for the operator

1. Fused kernels change roundings: served under the M6 tolerance, as the grouped attention is on CUDA, or only what
   stays bit for bit?
2. Our own engine (B) or MLX as the engine with Tetra as a custom kernel: MLX brings attention, norms and RoPE near
   their peak at once, at the price of a new dependency and its Rust bindings.
3. The rival download of step 0.
