# Prereg. M1: the fused Tetra matvec under PyTorch, weights resident compressed

Status: written and committed on 2026-09-30, BEFORE the first matvec.
Operator go given 2026-09-30, for Metal work, solo, 0 $. No paid job is part of this.
Plan: `docs/plan-transformers.md`, "What Metal needs". Stage 2:
`proofs/preregistration-hf-metal-decode-2026-09-30.md`.

A timestamped prereg is no longer edited. Any departure goes in
`proofs/preregistration-hf-metal-m1-2026-09-30-ECARTS.md`.

## 1. Question

Can a `transformers` model hold our weights **compressed in memory** and still say the same thing?

Stage 1 loads the packed 4B and gives the right tokens, with the weights dense: 8.05 GB at f16
against a 1.42 GB file. The compression exists on disk only. M1 puts the served Tetra matvec in the
forward pass so the 168 Tetra records are never materialized.

## 2. What is built

`torch.ops.llvq.tv_tetra48`, a binding over `tv_tetra48_metal`, the served shader. Per projection,
resident on the GPU: the `tetra48` words, the two gain centroids in f32, the row scales in f32, the
tail in f16. Shared: the four decode tables and `invnorm`, 32 entries, `1/sqrt(16m)` in f32 with
`invnorm[0] = 0`, exactly as `llvq_llm::fused` builds them.

Four constraints the code must respect, read in `llvq-llm/src/fused_metal.rs` and the shader:

- the source is compiled with `#define LLVQ_TILE_BLOCKS n` prepended, and the threadgroup buffer is
  `n · 24` floats;
- `d_out` must be a multiple of 8, since 32 threads take a row and the threadgroup is 256, and the
  kernel carries no row guard: a partial group would store past the output. Every `d_out` of the 4B
  is a multiple of 8;
- the stream is `d_out · row_stride_u32` words, plus at most one guard word, because `f1r_load`
  reads one word past the last block;
- `x` is the whole `d_in` activation. The kernel's epilogue reads the tail columns from
  `x + nblocks · 24` itself.

## 3. Three decisions taken before the code

| decision | what is done | why not the other |
|---|---|---|
| the rotation | applied in torch, f32, on the device: the sign flip, the butterfly, then the small mix. Per module, so a shared rotation is computed once per projection that uses it | `rot_apply_metal` exists and is the served path's own, but binding it is a second op to gate and it reads f16 where this path is f32. Swapping it in later is a drop-in, and the redundant work is named in the journal rather than hidden |
| the prefill | T dispatches per projection, one per token, no dense fallback | a dense fallback would need the materialized weight resident, which is the thing M1 removes |
| the int4 records | left dense, as stage 1 materializes them | the int4 matvec is blocked by the staging wall, which is M2. Mixing the two would put one gate on two changes |

## 4. The gate

**The same 64 greedy token ids as stage 1, on the four prompts of `bin/run`.** Stage 1's dump is the
reference (`docs/mesures/hf-quantizer-4b-2026-09-30.txt`, 256 of 256 against `bin/run`), so this
compares a compressed forward pass against a dense one that is itself pinned to our engine.

The kernel is not bit-identical to the dense path and no tolerance is widened to pretend otherwise:
it reads the tail in f16 and the row scales in f32 where the file holds f32 and f64. What is demanded
is the tokens.

Beside the gate, and not a gate: the per-row maximum absolute difference against the dense
reconstruction on one matrix of each shape, reported so the arithmetic gap has a number.

## 5. Controls

1. Resident memory measured, not computed, on the loaded model: the sum of the device buffers.
2. `d_out` not a multiple of 8, a stream of the wrong length, a missing tail: refused by name.
3. One mutant at least: the tail dropped, the row scale dropped, the rotation skipped. Each must
   change the tokens.
4. The op refuses a projection whose `row_stride_u32` is not `stride_u32(nblocks)`.
5. `pytest` green, `cargo clippy` silent on what this lot touches. The served shader is unchanged by
   M1, and `the_shipped_shader_is_the_repositorys` still passes.
6. Stage 2's gate still passes: the decode op is untouched.

## 6. Signed predictions

**The tokens match on all four prompts.** At the 4B our own kernel and dense paths agree over 256
tokens, and this is the same kernel over the same words.

**Resident weight memory lands between 2.8 and 3.4 GB**, against 8.05 dense at f16: 0.71 GB of Tetra
words, 33 MB of side data, 1.54 GB of int4 left dense at f16 and 0.78 GB of embedding.

**The per-row maximum difference against the dense reconstruction lands under 1e-2 relative**, and
the tail in f16 dominates it rather than the f32 row scales.

**The prefill is slow and no number from it is published.** One dispatch a token a projection, so a
prompt of 8 tokens costs 8 times 168 launches, and this plan claims nothing about speed at any stage.

I have been wrong on signed predictions five times in this repository, the last on 2026-09-30 by
naming the wrong of two costs. These are scored the same way.

## 7. What M1 cannot establish

- Nothing about the int4 records, which stay dense. That is M2, and it needs a served shader changed.
- Nothing about the embedding, which stays dense. That is M3.
- Nothing about speed, here or anywhere in this plan.
- Nothing about CUDA, which is planned separately and needs a paid job and its own go.
- Nothing about the 8B and the 14B.
