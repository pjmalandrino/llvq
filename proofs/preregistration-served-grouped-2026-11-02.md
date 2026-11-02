# Preregistration. The served door under the grouped attention, and the reference line it owes

**Written on 2026-11-02, to be committed and TIMESTAMPED BEFORE the run.** Operator go to launch given 2026-11-02 ("Go, plafond 2 $").
Cost: about **$0.87** on l40sx1, *estimated* below (§8), timeout 60 min, so $1.80 at worst. Cap asked for this wave:
**$2**, which covers one retry. Spent before it: $248.30 over 236 rows of the registry, 206 priced (*measured*,
`docs/data/jobs.csv`).

## 1. Why this run exists

The grouped attention passed its quality control on 2026-11-01
(`docs/mesures/gqa-ppl-2026-11-01.txt`) and became servable on 2026-11-02 through the `attention` field of a served
config (commit `e68cd51`, `configs/qwen3-4b-tetra-e4-grouped.json`). Nothing has run through that door.

Two holes follow from that, and one job closes both.

**The reference line.** Gate 4 of every job on this object checks that a decoded line hashes to `8784ffc4…`, the line
of run 1 of the paper table. That gate is what makes a run's provenance checkable, and it is defined for the expanded
form only. The A/B of 2026-11-01 returned the same 256 tokens for both forms after a 5-token prompt, so the hash
probably carries, and probably is not a gate.

**Every number on the grouped form comes from the bench door.** `LLVQ_GQA_AB` builds three arms in one process,
truncates each cache back to the prompt between decodes, and reports `255 / t`. The served door loads one arm from a
config file and reports `256 / t` over a whole `generate`. The two protocols are close and they are not the same, and
no figure of the served object should rest on the first one.

## 2. What runs

One image built from the commit that carries this file. One job, l40sx1, the sealed 4B
(`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`), and the two configs, written by the job from its
launch command and checked by sha256: `configs/qwen3-4b-tetra-e4.json` (`098fe49c…`, expanded, unchanged since
2026-09-23) and `configs/qwen3-4b-tetra-e4-grouped.json` (`2455fe85…`, the same object with
`"attention": "grouped"`). `ops/jobs/served-grouped.sh`.

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The bench door, replayed.** `fusedrun <file> 256` with the served flags spelled out, against the dense arm of the
   same process, as in run 1 of the paper table. It checks that the default path did not move when the field landed:
   `served.rs`, `kvq.rs` and three binaries changed.
3. **The served door, expanded.** `LLVQ_CONFIG=<e4> fusedrun <file> 256`, one arm, no dense reference. Its decoded
   line is hashed.
4. **The served door, grouped.** `LLVQ_CONFIG=<e4-grouped> fusedrun <file> 256`. Its decoded line is hashed and
   compared with step 3's.
5. **The served object against the context, both forms.** `LLVQ_CONTEXTS=5,2048,4096,8192` beside each config, the one
   measurement mode the served door accepts, which changes only the prompt it is given. Prompts in chunks of 4,096,
   256 new tokens, 1 discarded round and 5 timed, the sweep refusing any round whose tokens diverge from the first.

## 3. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The sealed file at its bytes and sha256 on the mount, and both configs at theirs.
3. Every load prints `int4: tv_q4_hh equals tv_q4_h bit for bit`.
4. Step 2 prints `256 tokens identical to the dense arm`, and its decoded line hashes to `8784ffc4…`.
5. Every served step prints a provenance line naming the form and the store: `attention expanded on cat` in steps 3
   and 5a, `attention grouped on grow(256)` in steps 4 and 5b. This is the gate that proves the field reached the
   kernel rather than only the parser: a config read and then dropped would print the line and run the other form,
   and the model's own getters are what the line is built from.
6. The served steps print `No dense arm on this path`, and no ratio.
7. The header of every step reads `tile 64`. No `LLVQ_*` variable is in the job environment; each step passes its own
   on its own command line.

## 4. Outcomes reported, never gated

- The two served decoded lines, their sha256, and whether they are equal.
- Served tok/s at 256 tokens after a 5-token prompt, both forms, median of 5 rounds with its range.
- For each form and each of the four prompt lengths: time to the first token, decode tok/s, ms a decode token.
- Whether either form's sweep refuses a round for divergence.

## 5. Signed predictions

| quantity | point | interval |
|---|---|---|
| step 2, fused tok/s (bench door) | **120** | [113, 126] |
| step 3, served tok/s, expanded | **122** | [116, 128] |
| step 4, served tok/s, grouped | **134** | [126, 142] |
| the two decoded lines | **equal** | |
| sweep, expanded, decode tok/s at 5 / 2,048 / 4,096 / 8,192 | **122.5 / 82.7 / 61.4 / 33.2** | [118, 126] / [79, 86] / [59, 64] / [32, 35] |
| sweep, grouped, decode tok/s at the same four | **134 / 113 / 97 / 77** | [126, 142] / [104, 121] / [89, 105] / [70, 84] |
| sweep, expanded, first token at 8,192, s | **34.9** | [33, 37] |
| sweep, grouped, first token at 8,192, s | **34.9** | [30, 40] |

Reasoning. Steps 3 and 5a should reproduce the trimmed arm of 2026-10-30 (122.4) and its sweep (122.5 / 82.7 / 61.4 /
33.2): same code, same object, one arm instead of five. The grouped points are the A/B's ratios applied to those
medians (1.1028, 1.3823, 1.5995, 2.3392), widened because the protocol differs and because the served door reports
`256 / t` where the A/B reports `255 / t`. The first token is the prefill, which holds no cache history, so the form
should not change it; the interval is wider for the grouped form because its prefill reshapes a 4,096-row `q` and has
never been timed, only scored (`gqa-ppl-2026-11-01`).

## 6. Decision rules

- **The reference line.** If the two decoded lines are equal, the grouped config inherits `8784ffc4…` and later jobs
  gate it on the same hash. If they differ, the grouped config gets its own reference line, printed in the journal and
  named in `ops/jobs/`, and no number of the published table moves on this run's strength.
- **Which file is the 4B's served object** stays the operator's, with this run as its input. The expanded config is
  the object paper 2 and the Hub cards describe; a run that is faster at every length does not by itself change what
  is published.
- **If the served door does not reproduce the bench door** within 3 % at 5 tokens for the expanded form, the bench
  door's figures are the ones to doubt, this run's numbers are reported and nothing is served differently, and a
  kernel timeline comes before any further step on this path.
- **If the grouped sweep loses to the expanded one at any length**, the form is not served at that length and the
  reading of 2026-11-01 is wrong about the served door; the run says so and stops the path.

## 7. Deviations

Any departure from this file goes in `proofs/preregistration-served-grouped-2026-11-02-ECARTS.md`, beside it, never
inside it.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Container and image pull 3 min, oracle 2 min, step 2 about 3 min, steps 3 and 4
about 2 min each with their loads. Step 5, per form: six rounds at four lengths, prefills of 0.03, 8.15, 16.91 and
34.86 s (*measured*, 2026-10-30) make 60 s a round, and the decodes about 17 s a round, so about 8 min a form and
16 min for both. In all about 29 min, **$0.87**. Timeout 60 min, **$1.80** at worst.

## 9. What this run does not establish

- Quality: nothing is scored here. MMLU and GSM8K through the served kernel under the grouped form are a separate run.
- Anything at 8B or 14B, on another card, or on Metal, which refuses the form by name.
- The prefill, beyond its wall-clock: 34.9 s to the first token at 8,192 is the hole the GEMM route would close, and
  this run does not touch it.
- The int8 cache, the step after this one.
