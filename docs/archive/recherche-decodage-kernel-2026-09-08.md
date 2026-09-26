# Quantized kernel execution and LLVQ performance

Fused weight decoding is standard, and LLVQ's code does not materialize a dense weight matrix during each projection.
The evidence supports a narrower concern: compression can save memory traffic while exposing instruction and scheduling costs.
Those costs differ between the served Planes14 path and the experimental Tetra decoder.

This assessment covers batch-one CUDA generation, with separate discussion of batched GEMM and alternative hardware.
Source inspection refers to commit `c878c2e7cbe7ea86768f6a6db3febd7f5d36c6b8`, on September 8, 2026.
Experimental values below come from existing journals. No new benchmark supports this assessment.

## 1. What our kernel executes

`planes_dot` reconstructs one coordinate in a register and immediately consumes it in a floating-point multiply-add.
It never writes reconstructed projection weights to global memory.

The actual dataflow is:

```text
compressed weight stream in device memory
    -> fields and small codebook values in registers
    -> coordinate selection and sign
    -> FMA with an activation staged in shared memory
    -> register accumulators
    -> warp reduction, row scale, exact tail, output store
```

The relevant code is [llvq_planes.cuh](../llvq-cuda/kernels/llvq_planes.cuh), lines 58–127.
`planes_fields` extracts the class, gain, signs and level planes.
`planes_dot` reads the class values and uses four independent accumulation chains.
The gain multiplies the completed block contribution, avoiding a gain multiplication for every coordinate.

[tv_planes_seg_h.cu](../llvq-llm/kernels/tv_planes_seg_h.cu), lines 90–127, supplies the served projection loop.
Each warp owns an output row. Its lanes process different compressed blocks.
The loop stages activations, synchronizes, computes the tile, then advances.
It contains no explicit asynchronous weight-prefetch pipeline and no Tensor Core MMA operation.
Hardware scheduling and compiler instruction scheduling can still overlap work across instructions and warps.

The source therefore establishes a fused SIMT matvec, not a sequence of separate dequantization and matrix-multiplication kernels.
The number of arrows in a conceptual diagram cannot predict its latency.
Dependencies within one lane also do not imply that the entire GPU executes those stages serially.

### Tetra has a different critical path

[llvq_f1rank_v3.cuh](../llvq-cuda/kernels/llvq_f1rank_v3.cuh) fetches rank and pattern information and builds packed coordinate bytes.
Its floating-point conversion constructs IEEE representations through byte permutations and a bias subtraction.
[llvq_tetra48.cuh](../llvq-cuda/kernels/llvq_tetra48.cuh), lines 77–123, adds the block norm and gain.
Packed integer dot products compute the squared norm; a small inverse-norm table supplies normalization.
The point then contributes through one floating-point FMA chain in natural coordinate order.

The source uses six packed quads and 24 FMAs per block, with six `__dp4a` operations for the norm
(*computed*, unrolled loops in the cited header). These are source-level counts, not measured SASS throughput.
`__dp4a` computes the weight norm here; it does not multiply the floating-point activation by the weights.

[tetra48_v3g.cu](../llvq-cuda/kernels/tetra48_v3g.cu) embeds that routine in the experimental floor geometry.
Its existence does not establish performance in `fusedrun` on a sealed mixed model.
The production reconstruction helper also includes a dump function for correctness checks.
That diagnostic write of decoded coordinates must not be confused with the matvec's dataflow.

## 2. What the academic literature establishes

### Fast decoding inside the multiply

**MARLIN (2024)** retains on-the-fly dequantization in registers.
It combines asynchronous memory loading, instruction scheduling, suitable layouts and Tensor Core computation.
Its conversion uses bit manipulations instead of naive integer-to-float casts.
The transferable lesson is to hide a short decode behind other work.
Its batching results do not establish that a Tensor Core rewrite wins for LLVQ's batch-one matvec. [1]

**FLUTE (EMNLP Findings 2024)** implements fused LUT dequantization and matrix multiplication.
It reorders static weights offline, vectorizes table entries, duplicates tables to reduce bank conflicts, and partitions work with Stream-K.
This directly identifies layout and on-chip table bandwidth as performance variables.
Offline rearrangement need not enlarge the number of stored bits as Planes14 does. [2]

**QuIP# (2024) and QTIP (NeurIPS 2024)** show that structured quantization can support fast decoding.
QuIP# exploits lattice symmetries to compress its codebook.
QTIP designs a bitshift trellis that permits parallel decoding from local bit windows.
Its computed and hybrid codes trade arithmetic against table size.
Neither paper treats the presence of a decoder as intrinsically disqualifying. [3, 4]

**LiquidGEMM (September 2025 preprint)** directly studies a dequantization bottleneck in W4A8 GEMM.
CUDA-side conversion can fail to feed the faster Tensor Cores.
It proposes a cheaper representation and a pipeline overlapping loads, conversion and MMA.
This supports the concern that nominally memory-bound quantization can become instruction-bound.
Its activation precision and GEMM execution differ from our floating-point-activation SIMT matvec. [5]

### Computing from codes through activation-dependent tables

**CodeGEMM (December 2025 preprint)** replaces repeated centroid reconstruction with a table of centroid–activation inner products.
Codes gather those partial sums, which are reused across output rows.
This is particularly relevant to the question of eliminating coordinate reconstruction.
The method requires small reusable codebooks; its AQLM-derived representation is not our Leech format.
Its telemetry uses memory-utilization proxies, which should not be interpreted as hardware-counter DRAM bandwidth. [6]

**LUT-GEMM (ICLR 2024) and T-MAC (EuroSys 2025)** use tables of activation combinations for low-bit multiplication.
T-MAC targets CPUs and exploits their table-lookup instructions.
Its no-dequantization formulation is an algorithmic alternative, but CPU speedups do not predict an L40S result. [7, 8]

**FluxBin (August 16, 2026 preprint)** couples binary-basis quantization with a CUDA LUT kernel.
It folds column scales into activation-table construction and applies row scales after lookup accumulation.
Its handling of salient columns preserves regular execution through a mapping to dense storage.
This recent work reinforces designing scales, representation and execution together.
Adopting its representation would require new quantization and quality evaluation; its reported speedups are not LLVQ forecasts. [9]

### Proposed hardware is a separate category

**LUT Tensor Core (ISCA 2025)** proposes specialized lookup hardware and new MMA-like instructions.
The paper also finds that a software LUT kernel can underperform dequantization-based CUTLASS on an existing A100.
Its custom-hardware results use simulation and hardware modeling.
They support the architectural interest of lookup computation, without providing an available replacement for our CUDA kernel. [10]

## 3. What our measurements support

### Planes14 pays for an expanded representation

At 4B, the repository reports **5.162 b/param for Planes14 plus q8 embedding**, against **2.764 b/param for Tetra plus q8**
(*computed*, whole-model accounting in [ETAT](ETAT.md) §3 and [Tetra journal](mesures/tetra-4b-2026-09-06.txt)).
Tetra's figure is a format projection, not measured allocation by a served Tetra engine.
The mixed Tetra plus int4 `v_proj` object has separate accounting and is not the bare-Tetra comparison here.

Planes14 stores 14 bytes per lattice block, while Tetra uses six
(*computed*, `LLVQ_PLANES_STRIDE` and Tetra's packed word map).
The simple Planes14 decoder is purchased partly by expanding the representation before inference.
Calling it a two-bit runtime obscures that tradeoff.

The F2 journal compares projection kernels in one process on L40S:

| Arm | Median milliseconds [range] | Journal's effective GB/s |
|---|---:|---:|
| Planes14 | 5.103 [5.101–5.115] | 428 |
| QTIP | 2.246 [2.245–2.248] | 405 |
| Our no-weight floor | 2.306 | 31 |

Times are *measured*; effective GB/s are *computed* from nominal stream bytes and benchmark timing.
The journal labels the GB/s column from minimum-time accounting; it is not a median hardware-counter bandwidth.
Source: [F2/P3](mesures/f2-p3-qtip-banc-2026-08-21.txt).
The comparable cross-format quantity is effective GB/s, not divided speedups against different baselines.

QTIP reads a smaller stream at roughly comparable effective bandwidth.
This supports traffic reduction as a major explanation of its shorter time.
It does not isolate every instruction cost: QTIP has its own grid and omits our row-scale and tail work.
Its synthetic payload supplies no quality comparison.

### The no-weight floor is not a causal pie chart

The floor retains staging, barriers, reductions, tail handling and output work while removing weight reads and decode.
QTIP finishes below that floor in its own geometry.
Therefore the floor is neither a hardware limit nor a universal unavoidable cost.

The often-cited **approximately 7% decode overhead** is *computed* by comparing bandwidth after floor subtraction
([original journal](mesures/nullk-plancher-2026-08-16.txt)).
It assumes the remaining work admits that decomposition.
Removing a decoder changes register demand, instruction scheduling and overlap.
The figure is useful evidence against an enormous Planes14 decoding penalty on L40S, but not a direct causal measurement.

The A100 result bounds that interpretation.
Planes14 takes 8.742 ms while the in-house FP16 control takes 6.915 ms
(*measured*, [F4](mesures/f4-a100-2026-08-18.txt)).
The clock investigation supports substantial dependence on per-SM execution rather than memory bandwidth alone
([clock journal](mesures/g-horloges-planes12x-2026-08-23.txt)).
It does not separate decode instructions from the rest of that execution cost.

### Tetra's precursor provides stronger arithmetic evidence

The same-process arithmetic comparison keeps the compressed words and rank table fixed:

| Arm | Median milliseconds [range] |
|---|---:|
| Initial rank decoder | 5.515 [5.509–5.518] |
| Variant v1 | 3.900 [3.890–3.904] |
| Variant v3 | 3.875 [3.871–3.882] |

Values are *measured*, [rank variants journal](mesures/f1-rang-variantes-2026-09-05.txt).
The paired v3 difference is −1.639 ms [−1.644; −1.627].
Register demand also falls from 48 to 40, with no local-memory allocation reported.

This establishes a large benefit from changing the arithmetic implementation.
It cannot charge the entire gain to conversion-unit latency independently of the changed register pressure and scheduling.
The variant removing the small dependent pattern lookups was slower, so those lookups were not the useful optimization there.
The benchmark omitted Tetra's full normalization and real labels; it does not establish served token throughput.

### Several plausible fixes have already failed

The A3 campaign found no useful gain from padding the shared activation stride.
Its reported change was −0.14% [−0.52; −0.08], within the control noise.
Split-K and multiple rows per warp also regressed
(*measured*, [A3](mesures/a3-occupation-banc-2026-09-01.txt)).

Source-level bank arithmetic alone should therefore not reopen padding as an established opportunity.
It predicts possible conflicts, whose timing depends on emitted load width and overlap. [11]
Those results constrain the tested Planes14 geometry; they do not prove every Tetra layout has the same optimum.

## 4. What CodeGEMM actually buys

CodeGEMM does not make the activation-dependent table free.
For a weight tile of width `t_w`, vector length `v`, codebook count `m` and code width `b`, it builds
`m · 2^b · (t_w/v)` scalar partial sums, then reads one scalar per code.
The paper's complexity is `O(m·MNK·(2^b/M + 1/v))`, under the assumption that the output dimension is much larger than `2^b`.
Its measurements attribute roughly 20–46% of cycles to building the table, depending on tile and matrix size. [6]

The reuse condition matters more than the slogan "no dequantization".
For one activation sub-vector, the table is useful only when many output rows reuse its code entries.
With a large effective code alphabet, most rows request different entries and the build cost approaches the work it replaces.
CodeGEMM's reported configurations use 8-bit codes and tile 2,048 output rows, which gives strong reuse.

### The direct Tetra translation is too large

A small explicit vector codebook permits the following exact real-arithmetic rearrangement:

```text
For each activation block x_b:
    P_b[k] = dot(C[k], x_b) for every codebook entry k
For each output row r:
    accumulate scale[r,b] * P_b[code[r,b]]
```

The table depends on both the current activation and its input position.
It must be rebuilt as activations change, and that build cost must be included.
Reuse across output rows can amortize construction. Floating-point reassociation may change rounding.

LLVQ has an implicit structured codebook rather than a small flat centroid table.
A table indexed by every possible Tetra shape field would have `2^47` slots
(*computed*, 48-bit word minus the gain bit in [llvq_tetra48.cuh](../llvq-cuda/kernels/llvq_tetra48.cuh)).
This counts candidate code addresses, not distinct lattice points. Direct enumeration is infeasible either way at that scale.

Factoring over Tetra's three 8-dimensional sections is possible in principle.
For an end section, the raw key is `(p, c, row)`: 2 parities, 128 possible prefix or suffix patterns, and 2,048 rank rows.
That is up to 524,288 partial sums, or 2 MiB in FP32, for one activation section.
For the middle section, 2 parities, 1,024 branch patterns and 2,048 mixed rows give up to 4,194,304 sums, or 16 MiB.
These are per activation tile, not one permanent model table, so they cannot live in shared memory.
(*computed* from `GOLAY_STATES`, `BRANCHES`, `CLASS_ROWS` and `N0_MIXED` in [Tetra](../llvq-search/src/tetra/mod.rs)).

The whole-block shell and gain do not mathematically forbid partial sums.
They are common scalars after the three section dots:
`y = (d1 + d2 + d3) · gscale[g] · invnorm[m]`.
They can remain in the epilogue, as [tetra48_dot](../llvq-cuda/kernels/llvq_tetra48.cuh) already does.
The shell index `m` depends on all 24 coordinates, so it cannot be folded into an independent section table without carrying cross-section state.
That is an extra lookup or reduction, not a proof that section partial sums are impossible.

The main obstacle is the alphabet and the available reuse.
Tetra's section key contains an 11-bit rank row plus pattern and parity bits, while CodeGEMM's strongest case uses 8-bit codes.
For one block, even a two-coordinate micro-table needs 1 parity bit, 2 pattern bits and two 3-bit ranks, hence 512 slots.
Twelve such tables for a 24-coordinate block occupy 24 KiB in FP32, or 12 KiB in FP16.
That is a plausible shared-memory experiment, but it requires a new tiling with many output rows sharing one activation block.
The current kernel stages 128 activation blocks and assigns one warp to each output row, so it does not provide CodeGEMM's 2,048-row reuse.

With two-coordinate tables, construction costs up to 512 entries × 2 FMAs × 12 groups, about 12,288 FMAs per input block before any row reads.
This can pay when a CTA reuses the table for hundreds or thousands of output rows.
It is a loss if a CTA reuses it for only the current eight or thirty-two rows.
The exact threshold depends on table precision, lookup latency and the register footprint of the new row tile.

An idealized arithmetic break-even makes the reuse requirement concrete.
The current path spends 24 FMAs per output row and input block.
The pair-table path spends up to 12,288 FMAs to build one block's tables, then 12 table reads and accumulations per row.
Ignoring decode and lookup latency, the table wins only above roughly 1,024 output rows per shared activation block.
At 2,048 rows the idealized count is 36,864 operations versus 49,152 FMAs; at 8 rows it is 12,384 versus 192.
These are *computed* operation counts, not timings, and they explain why a kernel-grid change is mandatory.

The existing rank table stores coordinate recipes, not completed activation products.
A product-table design must account for section patterns, parity, rank extraction, activation position and reuse.
It must preserve the whole-block normalization and gain, although those two factors can stay after the section sum.

The original LLVQ paper already identifies constant norm as a hardware advantage of a single shell.
Its reported CUDA demonstration uses fused dequantization and matvec on a single shell.
Our multi-shell path has additional scaling requirements
([local paper transcription](llvq-paper-notes.md), Key finding 2 and Appendix C).
This makes a factorized activation-product table a research hypothesis, not a drop-in implementation decision.

## 5. Which fundamental criterion can improve

The primary target is decode-phase latency, hence tokens per second at fixed quantized model quality and fixed stored bits.
The Psumbook changes the arithmetic and on-chip reuse; it does not change the `.llvq` payload or the quantizer's distortion.

The memory-capacity criterion is already won by Tetra before a Psumbook exists.
Tetra is 2.1498 kernel b/weight against Planes14's 4.8040, and 2.764 versus 5.162 b/param in the 4B whole-model accounting
(*computed*, [ETAT](ETAT.md) §§3 and 5 quinquies).
The Psumbook is temporary shared memory, so it adds no persistent model bytes.
It can still reduce occupancy if its table consumes too much shared memory.

An optimistic throughput bound can be computed from the existing Planes14 bench, with clear limits.
Planes14 reads 2.18 GB and spends 5.103 ms; the Tetra stream estimate is about 0.98 GB
(*measured* for Planes14, *computed* for the Tetra stream, [ETAT](ETAT.md) §5 bis and [F2/P3](mesures/f2-p3-qtip-banc-2026-08-21.txt)).
If Tetra sustained Planes14's 779 GB/s effective rate and paid no extra decode cost, its current-grid lower bound would be about
`2.306 + 0.98/2.18 × 2.797 = 3.56 ms`, or 1.43× faster than Planes14.
This is an *estimated upper bound*, not a Tetra result: it assumes the same launch floor, bandwidth and occupancy.
The Psumbook's role would be to approach this bound by removing decode arithmetic, not to lower the bound itself.

The energy criterion should move in the same direction if the table reduces CUDA-core instructions without lowering occupancy.
CodeGEMM reports improved compute and energy efficiency from precomputed inner products, but its A100 setup, codebook and tiling differ from LLVQ.
Energy per token therefore requires a direct measurement.

The quality criterion should remain unchanged with FP32 partial sums and exact table construction.
FP16 partial sums may reduce shared memory from 24 KiB to 12 KiB, but they introduce a new rounding path and need a perplexity and MMLU check.

The gain is likely to grow with output rows sharing an activation tile and with prefill or batching.
It may disappear on small matrices or on the A100, where this repository already observes a shift toward per-SM compute limits
(*measured*, [F4](mesures/f4-a100-2026-08-18.txt)).

## 6. Implications for the next decision

The strongest supported diagnosis is a combination of expanded runtime traffic and architecture-sensitive instruction costs.
There is insufficient evidence to attribute the served slowdown chiefly to coordinate reconstruction.

The next Tetra evaluation should retain the existing full-reconstruction reference and compare real-format outputs before timing.
The floor decoder, full gain-and-norm decoder, Planes14 and competitors should coexist in the benchmark process.
Each competitor keeps its own geometry. Differences within one geometry should be formed round by round.

Useful diagnostics are compiled instruction mix, register and spill counts, and hardware counters when available.
Instruction-issue stalls, shared-memory traffic and actual DRAM bytes would distinguish mechanisms that effective GB/s cannot.
The earlier counter refusal remains documented in [F3](mesures/f3-events-2026-08-19.txt).

A separate research proposal should price factorized activation-product tables before implementing a new kernel.
The first candidate is not a full Tetra table.
It is a two-coordinate Psumbook with 512 entries per coordinate pair, FP16 and FP32 variants, and a CTA tile large enough to reuse one activation block across many output rows.
The calculation must record table bytes, construction work, reuse, shell-index work, register count and numerical error.
It should compare those costs with the current packed-byte decoder in the same process.

Prefill and batching require separate conclusions because they reuse weights across tokens.
Tensor Core feeding and decode amortization then become different optimization problems from batch-one matvec.

End-to-end claims also need an identical vocabulary head.
The historical B2 4B comparison gives raw ×2.00 [1.99–2.00] and same-head ×1.11 [1.11–1.11]
(*measured*, [B2](mesures/b2-fusedrun-plages-2026-08-18.txt), quotients of medians with envelopes).
Those values precede the served rotation sharing and projection fusion; they are not current-v1 attribution.

## Sources

The following are primary sources, accessed September 8, 2026. Preprint status refers to the version consulted.

1. Frantar et al. [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/html/2408.11743v1). August 21, 2024, §§3.3–3.4.
2. Guo et al. [Fast Matrix Multiplications for Lookup Table-Quantized LLMs](https://arxiv.org/html/2407.10960v1). July 2024; EMNLP Findings 2024, §3.
3. Tseng et al. [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/html/2402.04396v2). 2024, codebook design and inference evaluation.
4. Tseng et al. [QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/html/2406.11235v2). October 28, 2024 version; NeurIPS 2024, §§3.1 and 4.3.
5. [LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving](https://arxiv.org/html/2509.01229v1). September 2025 preprint, §§3–4.
6. Park et al. [CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs](https://arxiv.org/html/2512.17970v1). December 19, 2025 preprint, §§3–4.
7. [LUT-GEMM: Quantized Matrix Multiplication Based on LUTs for Efficient Inference in Large-Scale Generative Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/file/a4f98ce85f440ee269b0df57b4368719-Paper-Conference.pdf). ICLR 2024.
8. [T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge](https://arxiv.org/html/2407.00088v2). 2024 preprint; EuroSys 2025.
9. Yang et al. [FluxBin: Flexible LUT-based Ultra-low-bit LLM Inference by Algorithm-Kernel Synergy](https://arxiv.org/html/2608.15602v1). August 16, 2026 preprint, §3.2.
10. [LUT Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference](https://arxiv.org/html/2408.06003v2). May 2025 version; ISCA 2025, §§2 and 4.
11. NVIDIA. [CUDA Best Practices Guide: Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory). Banking and broadcast rules.
