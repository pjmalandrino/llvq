# Preregistration. M13: the attention half of a Metal decode token, by op, at 6 and 4,096 positions

Written on 2026-11-06 at 15:00 on branch `claude/metal-nuit-3c4257`. To be timestamped before its first measurement,
then never edited; departures go to `preregistration-metal-nuit-2026-11-06-ECARTS.md` under an M13 heading. Cost
**$0**, local, under the operator's go of 2026-11-06 ("Oui, découper"). D2 of M12's rule.

## 1. Why

M12 read the decode token as GPU-bound (host 5.4 of 15.2 ms). M7b's remainder (blocks minus the projection replay,
6.3 ms) was read as attention, RoPE, norms and cache. `metalrun attnprof` replays those ops alone. A 3-round check
of it, run before this file and not quoted, read about 0.9 ms for all of them at 6 positions, and M7b's per-class
alone costs plus that sum to about 15.5 ms against a 15.9 ms blocks phase. The working reading is that the replay
overlaps independent kernels, which a dependent chain in the model cannot, so the remainder was an artefact and the
alone costs are the split. The predictions below are informed by that check.

## 2. What runs

Binary `metalrun` sha256 `7b5bbee25848fd4ed52113b3de1a8b12a972e573746b03e19d07d09bcdc25767`, built from 1cd57f5, the parent of the commit that carries this file, which adds
only this file. Fifteen minutes after the last build, nice 10.

1. A1: `metalrun attnprof <sealed> configs/qwen3-4b-tetra-e4.json lutd+h+grp+pq 11` (6 positions held).
2. A2: `metalrun attnprof <sealed> configs/qwen3-4b-tetra-e4.json lutd+h+grp+pq 11 4096` (4,096 positions held).

## 3. Quantities and rules

- Each class alone (qknorm, rope, append, attn, ctxglue) and their sum, median and range over 11 rounds, off protocol
  (closed by a synchronize), never quoted as a rate.
- The split of a token is then M7b's alone costs plus A1's sum, *computed*; set against M7b's blocks phase.
- If A1's sum is under 2.0 ms, the 6.3 ms remainder is called an artefact of the replay's overlap and the next lever is
  chosen on the alone costs: Tetra first.
- A2 against A1 gives what a 4,096-position cache adds to a token's attention, to set against the decode gap between
  5 and 4,096 tokens of prompt (M8: 18.4 against 25.6 ms a token).

## 4. Predictions, signed

*Estimated*, informed by the check.

| quantity | prediction |
|---|---|
| A1 sum of the classes alone | 0.8 to 1.6 ms |
| A1 attn | 0.4 to 0.9 ms |
| A2 attn | 4 to 9 ms |
| A2 every other class | within 0.1 ms of A1's |
