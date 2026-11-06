# Preregistration. M9: the Metal prefill kernel on the sealed 4B, time to the first token

Written on 2026-11-06 at 12:45 on branch `claude/metal-nuit-3c4257`, once the code it describes existed. To be
timestamped before its first measurement, then never edited; departures go to
`preregistration-metal-nuit-2026-11-06-ECARTS.md` under an M9 heading. Cost **$0**, local, under the operator's go of
2026-11-06 ("je veux pouvoir tester un 4B Metal qui bombarde").

## 1. Why

M8 read the 4,096-token prompt of the WikiText-2 window in 60 s on Metal, about 68 tokens/s, near the decode rate:
the Metal adapter declares no rows kernel, so the prefill runs the matvec once a prompt row and decodes the 749 MB of
Tetra codes once a row. `+p` sets `MetalTuning::prefill_rows` to 8: the Tetra projections take chunks of 8 rows
through `tv_tetra48_metal_lutdr`, which decodes each block once and applies it to the 8 rows. Per row the arithmetic
is the decode kernel's, in its order, so each row is the served matvec narrowed to f16. Gated:
`tests/tetra48_metal_rows.rs` (`the_prefill_kernel_returns_each_rows_served_bits_narrowed`, every chunk length 1 to 8,
the 4B's two widths and the loop edges, every f16 equal). The int4 records keep their row-by-row loop. A smoke run at
512 tokens, one timed round, off protocol and not quoted, read the same 16 tokens and a first token 1.7 times sooner.

## 2. What runs

Binary `metalrun` sha256 `a68aa617ebd684d6603191af8f7d7daa00cc55bc02df8856699d2132508405cb`, built from 4e4891b, the
parent of the commit that carries this file, which adds only this file. Quiet machine (no build from 15 minutes before
the first run to the last), nice 10. Sealed file `qwen3-4b-sealed.bin`, sha256 `886391a8…`.

1. G: `metalrun gate <sealed> configs/qwen3-4b-tetra-e4.json lutd+h+grp,lutd+h+grp+p`.
2. S3: `metalrun time <sealed> configs/qwen3-4b-tetra-e4.json lutd+h+grp,lutd+h+grp+p 64 5 4096` (4,096 tokens of
   WikiText-2, the M8 S2 prompt).

## 3. Gates

1. G: 253 of 253 for the `+p` arm.
2. S3: the same 64 tokens on both arms every round. Different tokens stop the reading: the prefill is not bit for bit
   and the kernel is withdrawn.

## 4. Rules

- Primary: `+p / base` on the time to the first token, round by round. Wholly above 1.000, `+p` is retained and
  joins the bundle's `new` arm. Straddling or below, not retained.
- Secondary: the decode ratio. `+p` touches no decode launch; a decode ratio wholly below 0.98 is journaled as a
  departure to explain.
- Off this rule: nothing is served. `served.rs` and the configs do not move.

## 5. Predictions, signed

*Estimated*.

| quantity | prediction | why |
|---|---|---|
| S3 first token, `base / +p` time ratio | x1.5 to x2.5 | Tetra was about 9 of 14.6 ms a prompt row; int4, rotations and attention stay |
| S3 decode `+p / base` | x0.98 to x1.02 | no decode launch changes |
