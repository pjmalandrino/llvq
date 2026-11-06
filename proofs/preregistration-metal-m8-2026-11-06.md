# Preregistration. M8: the half-io twins on the sealed 4B, Metal decode speed

Written on 2026-11-06 in the morning on branch `claude/metal-nuit-3c4257`, once the code it describes existed. To be
timestamped before its first measurement, then never edited; departures go to
`preregistration-metal-nuit-2026-11-06-ECARTS.md` under an M8 heading. Cost **$0**, local, under the operator's go of
2026-11-06; run from 12:00, the operator's window for heavy runs.

## 1. Why

The Metal matvecs store f32 (Tetra) or read and store f32 (int4), and the adapter casts around them: 168 narrowing
casts for the Tetra records and 84 widening plus 84 narrowing casts for the int4 records, 336 candle launches a token
that carry no arithmetic (`fused_metal.rs`, `MetalLattice::matvec`, `matvec_int4`). The CUDA twins store f16 in the
kernel. `+h` loads the half-io twins (`_luth`, `_lutdh`, `tv_q4_metal_h`, `tv_q4_metal_tiled_h`), which do the same
two roundings inside the kernel. Gated bit for bit: `tests/tetra48_metal_rows.rs` and `tests/q4_metal_candle.rs`
against the served kernels narrowed by candle's cast (every f16 equal), `metalrun gate` 253 of 253 on the sealed
file, and a mutation run on cfd2abb that killed 13 of 13 mutants (eight in the four twins, five in the host
wiring; journal).

## 2. What runs

Binary `metalrun` sha256 `c09b08c70df5cb7db8cd8c0c8287cd59e79dc614d05a2d2a2c7585921bff51ab`, built from cfd2abb,
the parent of the commit that carries this file, which adds only this file. Quiet machine (no build from 15 minutes
before the first run to the last), nice 10. M8 runs right after M7b, in the same quiet window. Sealed file
`qwen3-4b-sealed.bin`, sha256 `886391a8…`.

1. G: `metalrun gate <sealed> configs/qwen3-4b-tetra-e4.json lutd,lutd+h,lut+h,lutd+h+grp`.
2. S1: `metalrun time <sealed> configs/qwen3-4b-tetra-e4.json lutd,lutd+h,lut 256 7` (5-token prompt).
3. S2: `metalrun time <sealed> configs/qwen3-4b-tetra-e4.json lutd+grp,lutd+h+grp 64 5 4096` (4,096 tokens of
   WikiText-2).

## 3. Gates

1. G: 253 of 253 for every arm against `lutd`.
2. Every arm returns the same tokens every round, and the same tokens as the first arm.

## 4. Rules

- Per context, `lutd+h / lutd` (S1) and `lutd+h+grp / lutd+grp` (S2), decode, round by round: wholly above 1.000,
  `+h` is retained and becomes the bundle's `new` arm; straddling, a wash, retained anyway only if the median is above
  1.000 (it removes launches and moves no bit); wholly below, not retained, and the journal says why.
- `lut` in S1 places the night's reference (`lutd / lut`, M2 read x1.025).
- Off this rule: nothing is served. `served.rs` and the configs do not move.

## 5. Predictions, signed

*Estimated*.

| quantity | prediction | why |
|---|---|---|
| S1 `lutd+h / lutd` decode | x1.03 to x1.10 | 336 launches at 1.66 to 5 µs against a 19 ms token |
| S2 `lutd+h+grp / lutd+grp` decode | x1.01 to x1.06 | the same launches against a longer token |
| S1 `lutd / lut` | x1.00 to x1.05 | M2 |
