# Preregistration. M7b: M7 rerun with the fixed instrument

Written on 2026-11-06 at 10:20 on branch `claude/metal-nuit-3c4257`. To be timestamped before its first measurement,
then never edited; departures go to `preregistration-metal-nuit-2026-11-06-ECARTS.md` under an M7b heading. Cost
**$0**, local, under the operator's go of 2026-11-06.

## 1. Why

M7 (stamped 10:07) ran at 10:17 and its marginal costs came out negative, an artefact of the instrument (ECARTS É6).
This rerun keeps M7's question, arms, gates, rules and predictions word for word, with the instrument fixed: every
config frees its outputs at the end of every layer, each reading is preceded by an untimed full pass and averages
three passes closed by one synchronize, and the config order turns by one each round.

## 2. What runs

Binary `metalrun` sha256 `ccb7c07c3c3069465264aad650d063a044ad636e43446ebeb1b70a58221d5b6f`, built from the commit
that carries this file. Quiet machine (no build from 15 minutes before the first run to the last), nice 10.

1. P1: `metalrun profile <sealed> configs/qwen3-4b-tetra-e4.json lutd 11`
2. P2: `metalrun profile <sealed> configs/qwen3-4b-tetra-e4.json lut 11`

## 3. Gates

M7's three, plus:

4. Every class's marginal median at or above -0.10 ms, and the sum of marginals at or below the full sequence plus
   10 %. Otherwise the marginal column is unusable again; the journal says so, and the next kernel is chosen on the
   alone GB/s and the remainder, as M7 left them.

## 4. Rules and 5. Predictions

M7's §4 and §5, unchanged.
