# Preregistration. M10: the int4 prefill kernel beside the Tetra one, time to the first token

Written on 2026-11-06 at 12:48 on branch `claude/metal-nuit-3c4257`, once the code it describes existed. To be
timestamped before its first measurement, then never edited; departures go to
`preregistration-metal-nuit-2026-11-06-ECARTS.md` under an M10 heading. Cost **$0**, local, under the same go as M9.

## 1. Why

M9 times `+p`, the Tetra prefill kernel; the int4 records (v, o, and down in 12 layers) keep their row-by-row loop
under it. `+pq` also sends their chunks through `tv_q4_metal_rows`: each word dequantized once for up to 8 rows, the
activation read from device memory, so no tile and no threadgroup wall. Per row the arithmetic is the served kernels',
and the f16 is the adapter's narrowing done in the kernel. Gated: `tests/q4_metal_candle.rs`
(`the_int4_prefill_kernel_returns_each_rows_served_bits_narrowed`, chunks 1 to 8 at six widths including the three
past the wall, bound at an offset, every f16 equal). `Int4Proj` gains `rows_per_launch` and `matvec_rows` with a
default of 1, so the CUDA adapter does not move. A smoke run at 512 tokens, one round, off protocol and not quoted,
read the same tokens on base, `+p` and `+pq`.

## 2. What runs

Binary `metalrun` sha256 `2c778780d4f130b7485443ac701e24674e2cffcb325f2bbdf0a2ae6164edc0d5`, built from
af523c5, the parent of the commit that carries this file, which adds only this file. Right after
M9, in the same quiet window, nice 10.

1. G: `metalrun gate <sealed> configs/qwen3-4b-tetra-e4.json lutd+h+grp,lutd+h+grp+pq`.
2. S4: `metalrun time <sealed> configs/qwen3-4b-tetra-e4.json lutd+h+grp+p,lutd+h+grp+pq 64 5 4096`.

## 3. Gates

1. G: 253 of 253.
2. S4: the same 64 tokens on both arms every round; otherwise the int4 prefill kernel is withdrawn.

## 4. Rules

- Primary: `+pq / +p` on the time to the first token, round by round. Wholly above 1.000, `+pq` is retained and
  replaces `+p` in the bundle's `new` arm. Straddling or below, `+p` stays (if M9 retains it).
- Secondary: the decode ratio, expected flat; wholly below 0.98 is a departure to explain.
- Nothing is served. `served.rs` and the configs do not move.

## 5. Predictions, signed

*Estimated*.

| quantity | prediction | why |
|---|---|---|
| S4 first token, `+p / +pq` time ratio | x1.05 to x1.20 | int4 about 2.4 of the remaining ~6.7 ms a prompt row, halved or better |
| S4 decode `+pq / +p` | x0.98 to x1.02 | no decode launch changes |
