# Preregistration. M12: does a Metal decode token wait on the host?

Written on 2026-11-06 at 14:45 on branch `claude/metal-nuit-3c4257`. To be timestamped before its first measurement,
then never edited; departures go to `preregistration-metal-nuit-2026-11-06-ECARTS.md` under an M12 heading. Cost
**$0**, local, under the operator's go of 2026-11-06 ("Oui, découper").

## 1. Why

M7b left 6.3 ms of a 15.9 ms `blocks` phase outside the projection replay (attention, RoPE, q/k norms, cache,
reshapes), and its class-removal split failed twice (ECARTS E6, E7). M8 read 336 removed cast launches as about
0.1 ms, so a dependent launch is cheap on the GPU side. Two causes are left for the 6.3 ms, and they call for
different levers: the GPU running candle's small attention kernels, or the GPU idle while the host encodes the
token's ~1,500 operations. The process's CPU time tells them apart with no new code.

## 2. What runs

Binary `metalrun` sha256 `00dd5d40cf41bf7707eb5eacef8f05feabbba4e5b5b345d977ecc12226c0c54e`, built from b9ba4a9, the
parent of the commit that carries this file, which adds only this file. Quiet machine (no build from 15 minutes before
the first run to the last), nice 10. Both under `/usr/bin/time -l`:

1. D1a: `metalrun time <sealed> configs/qwen3-4b-tetra-e4.json lutd+h+grp+pq 64 1`
2. D1b: `metalrun time <sealed> configs/qwen3-4b-tetra-e4.json lutd+h+grp+pq 320 1`

Each makes 3 generations (2 discarded, 1 timed) of N tokens after the 5-token prompt, after the same load and bit
gate. So D1b runs 768 more decode tokens than D1a and nothing else more.

## 3. Quantities

- host ms a token = ((user + sys) of D1b - (user + sys) of D1a) / 768, *computed*.
- wall ms a token = D1b's printed decode ms a token, *measured*.
- ratio = host / wall.

## 4. Rules

- ratio at or above 0.80: the token is host-bound. The next lever is the host cost of an operation (candle op count,
  our custom ops' encoder setup), not a kernel; D2 (the attention split by op on the GPU) is not built.
- ratio at or below 0.50: GPU-bound. D2 is built next, with its own prereg.
- between: mixed; both are named, D2 first.
- One reading, no rounds: this chooses work and is never quoted as a rate.

## 5. Predictions, signed

*Estimated*.

| quantity | prediction | why |
|---|---|---|
| host ms a token | 6 to 12 | ~1,500 encodes at 4 to 8 µs |
| ratio | 0.4 to 0.7 | M8's cheap casts argue against a fully host-bound token |
