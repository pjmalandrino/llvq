# Annex to the prereg of 2026-11-21: encoder settings and bit accounting

Written before the first encode, as the prereg requires. The prereg
`preregistration-inkling2-mini-quality-gate-2026-11-21.md` (sha256 `e947b7c8…cfec49`, stamped at 22:01) does not move.

## The sealed recipe this gate reuses

The served Tetra 4B was encoded by
`LLVQ_CALIB=dclm-edu LLVQ_INT4_TYPES=v_proj smoke 64 2048 12 4096 <dev> nogs tetra1 999 rot`
(`ops/jobs/guided-encode-4b.sh:44-45`, `docs/data/jobs.csv` rows 189 and 343). The codebook `tetra1` is chosen at
encode time; it is not a layout applied to other codes.

Per-matrix settings, used here unchanged through `llvq-llm/src/extq.rs`, which calls the existing `llvq_quant`
functions only:

- damping 1e-2 relative to mean(diag H);
- `h_shrink` 1 and `gain_scale` 1;
- `GptqConfig` with blocks of 24, retract, `nogs`, no design C, no spherical feedback, lambda 1e-2, `KeepExact`;
- the input rotation, the factorization, the rotated rows, the 2-level Lloyd fit of 40 iterations, frozen row scales,
  the row-parallel quantization, the un-rotation.

Checked before the first encode (`llvq-llm/tests/extq_equiv.rs`, 5 passed of 5, rerun by the coordinator at 22:39): on
the seven matrices of block 0 of a tiny Qwen3 the external chain equals the block loop bit for bit; a damping of 2e-2
and a changed seed break the equality; Gaussian weights under H = I give a relative MSE of 0.0854 against the 0.0851
of `docs/mesures/innovation-tetra-v2-forme-2026-11-11.txt`.

## What differs from the recipe, by construction

- The fused matrix `w13[e]` is split into its gate half and its up half. They share one Hessian, one rotation and one
  factor, as the `Act::Mlp` group does for Qwen3. `w2[e]` has its own Hessian, of width 1,152.
- The rotation seed is `0x110FEED ^ layer<<32 ^ expert<<20 ^ group<<16`, group 0 for the input of `w13` and 1 for the
  input of `w2`. The Qwen3 seed space is keyed by (block, activation) and has no expert.
- The Hessian arrives in f64 as an unnormalised weighted sum and is divided by its weight sum. The recipe accumulates
  in f32 and divides by the token count. The scale is neutral for the result up to rounding.
- An expert with no routed token gets H = I and the flag `identity_h`. A factorization that still fails after damping
  gets H = I and the flag `factor_failed`, where the recipe would abort. Both flags are counted and reported.
- No int4 type is used: every expert matrix is in the lattice code.

## Bit accounting for the criterion

b_T, the lattice arm's expert bits, is the **archive** figure: the payload bits of the saved records, 2.0371 bits a
weight (48-bit words, f64 row scales, two f64 centroids a matrix). The driver asserts on every write that this figure
equals the bits it saved. The kernel figure of `rtbits`, 2.0185, is reported beside it and is not the one the criterion
reads. A GGUF rung reads the stored bytes of its expert tensors, its own scales included, as the prereg says.

The rungs' expert bits are already read from the files' headers, before any KL (`eval/gguf_bits.json`): on {6..17}
2.229, 2.368, 2.444 and 2.764 for `UD-IQ1_S`, `UD-IQ1_M`, `UD-IQ2_XXS` and `UD-Q2_K_XL`; on {6, 7, 8} 2.014, 2.014,
2.264 and 2.570. So on {6..17} b_T sits under the lowest rung, and the prereg's rule for that case applies: PASS when
KL_T is at or under that rung's KL, and no rung can read FAIL.

## Codes are kept

Each expert's three records are written with the repository's own `write_matrix_with` (format v5, Tetra kind) to
`codes/eNNN.llvqrec`, read back, and required to decode to the reconstruction bit for bit before the file is renamed
into place. A flipped bit fails `extq verify`.

## Measured before production

14.4 s an expert at 10 threads on a synthetic layer of the real shapes, 383 µs a block and thread; about 31 minutes a
layer (*estimated* from that probe).
