# Prereg. M2: the int4 matvec on Metal, tiled beside the served kernel

Status: written and committed on 2026-09-30, BEFORE the first dispatch of the new kernel.
Operator go given 2026-09-30. Cost: 0 $, Mac.
Plan: `docs/plan-transformers.md`, "What Metal needs". M1:
`proofs/preregistration-hf-metal-m1-2026-09-30.md`, journal
`docs/mesures/hf-metal-m1-4b-2026-09-30.txt`.

A timestamped prereg is no longer edited. Any departure goes in
`proofs/preregistration-hf-metal-m2-2026-09-30-ECARTS.md`.

## 1. Question

Can the 84 int4 records of the 4B run on Metal, so the model holds none of its projections dense?

M1 left them materialized, 3.1 GB of the 5.44 the loaded model allocates. `tv_q4_metal` stages the
whole activation in threadgroup memory, `d_in · 4` bytes against the 32,768 an M-series offers, so it
refuses `d_in` above 8,192 and every sealed file's `down_proj` is above it: 9,728 at the 4B, 12,288 at
the 8B, 17,408 at the 14B.

## 2. The decision: a second entry point, not an edit

`tv_q4_metal` is a served kernel and is not touched. `tv_q4_metal_tiled` is added beside it in the
same file, which is that file's own established practice: `llvq_tetra48.metal` already carries nine
variants of one matvec, and the served path picks one by name.

So the served Rust path keeps calling `tv_q4_metal`, byte for byte what it was, and only the torch
path calls the new one. `MetalRuntime::upload_int4`'s refusal above 8,192 is also untouched: it guards
the Rust runtime, which this does not go through.

## 3. Why the tiling is bit-identical, by construction and not by luck

In `tv_q4_metal`, lane `l` of a simdgroup accumulates the words `l, l + 32, l + 64, …` of its row, in
that order, and floating-point addition is not associative, so that order is part of the answer.

The tiled kernel stages `tile_cols` columns at a time, which is `tile_cols / 8` words. **If
`tile_cols` is a multiple of 256** then `tile_cols / 8` is a multiple of 32, so within each tile lane
`l` takes the words `c₀/8 + l, + 32, …` and the concatenation over tiles is exactly the sequence
`l, l + 32, l + 64, …` the untiled kernel walks. Same terms, same order, same roundings.

A tile that is not a multiple of 256 breaks that and is refused by name rather than rounded up. The
last tile may be short, which changes nothing: its first word index is still a multiple of 32.

## 4. The gate

**Bit-identical to the served kernel, where the served kernel runs.** Both entry points are bound as
torch ops and dispatched on the same inputs for every `d_in` that fits under 8,192, at tiles 256,
2,048 and 8,192. Equality of every output value, not a tolerance: the two kernels are the same
arithmetic in the same order or the claim of §3 is false.

Then, and only then: **the 84 int4 records of the 4B run**, and the model gives the same 256 greedy
ids as M1 and `bin/run`.

## 5. Controls

1. A tile that is not a multiple of 256 is refused by name, and so is a tile whose staging exceeds
   32,768 bytes.
2. `d_in` not a multiple of 8 is refused: eight nibbles a word is what makes a row start on a word,
   and the served kernel's own comment records that nobody asserts it today.
3. Per-row agreement with the dense reconstruction, one matrix of each int4 shape, reported as a
   number and gated at 1e-2 relative. After M1 this is a gate and not a remark.
4. Mutants, at least: the group index shifted by one, the nibble order reversed, a tile of 128. Each
   must be caught, and by which gate is recorded.
5. The served shader's own bytes are unchanged outside the added function, which `git diff --stat`
   shows and the journal quotes.
6. M1's gate still passes, and stage 2's.

## 6. Signed predictions

**The two kernels agree exactly, at all three tiles.** The argument of §3 is a proof about indices,
not an approximation, so the prediction is equality. If it fails it fails on the short last tile.

**Resident memory drops from 5.436 GB to about 2.9 GB.** The 84 int4 records are 3.1 GB dense at f32
and 0.41 GB as the kernel reads them, so 5.436 − 3.1 + 0.41 = 2.75, plus the embedding's 1.56 which
does not move. Call it [2.6 ; 3.1].

**The 256 ids are unchanged.** The arithmetic is bit-identical to a kernel whose own dense arm matched
over 256 tokens.

**The per-row residue lands under 1e-3 relative**, the int4 path having no lattice decode and no
rotation, only an affine dequantization in f32.

I have been wrong on signed predictions six times here, twice on 2026-09-30. Scored the same way.

## 7. What M2 cannot establish

- Nothing about the embedding, which stays dense. That is M3.
- Nothing about the served Rust path, which is untouched and still refuses `d_in` above 8,192.
- Nothing about speed, at any stage.
- Nothing about the 8B and the 14B, whose `down_proj` are wider still and would exercise more tiles.
