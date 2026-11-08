# Preregistration. Qwen3-0.6B in Tetra, the drafter of the speculative decode

**Written on 2026-11-07.** Stamped with OpenTimestamps before the first measurement, then never edited; a deviation
goes beside it in `proofs/preregistration-draft-0p6b-2026-11-07-ECARTS.md`.

Operator go: "ça vaut le coup pour notre famille de tester avec le 0.6B" (2026-11-07), relayed by the session that
started branch `claude/spec-0p6b-96ba73`. Cost: **$0**, the Mac only. Registry before this file: $277.11
(*computed*, `docs/data/jobs.csv` on main at 529100a), plus the batching session's jobs.

## 1. Why this object exists

Speculative decoding needs a drafter that shares the target's tokenizer. Qwen3-0.6B shares the tokenizer of every
Qwen3 the repository serves (4B, 8B, 14B). Its quality decides only the acceptance rate: under greedy acceptance the
target's tokens do not depend on the drafter. No 0.6B is sealed today.

## 2. What runs

On the Mac, `nice -n 10`, `LLVQ_THREADS=12` (16 cores, 4 left), Metal:

1. `smoke 64 2048 12 4096 metal nogs tetra1 999 rot` on `Qwen/Qwen3-0.6B`, `LLVQ_CALIB=dclm-edu`,
   `LLVQ_INT4_TYPES=v_proj,o_proj`, `LLVQ_ARTIFACT=<dir>/qwen3-0p6b-dclm.llvq`. This is the 14B encode's recipe
   (`ops/jobs/encode-14b.sh`: tetra1, dclm-edu 64 x 2048, rotation, nogs, damping 1e-2, f32) with `o_proj` in int4
   as well, which 4E reads at the 4B and the 14B.
2. `seal` to `qwen3-0p6b-sealed-f16.bin`, then `embedq ... q4` to `qwen3-0p6b-sealed.bin` (the embedding int4 g64,
   `LLVQ_EMBED=q4`, the served tables).
3. `rtbits` on the sealed file: b/param for the whole model, tables included. The file size in bytes.
4. `ppl` on the sealed file, f16, Metal, 12 windows of 4,096 (the `ppl-scelles` protocol).

Declared departures from the served files' chain: no trained row scales (a GPU training job, which buys quality the
drafter only spends on acceptance), and `o_proj` encoded int4 by the encoder rather than swapped in from the
checkpoint by `int4swap`.

## 3. Gates

1. `oracle Qwen/Qwen3-0.6B 64 metal` prints MATCH before the encode (rule 10).
2. `smoke` verifies every record against the evaluated model (its own check) and writes 196 records, 56 of them int4.
3. `seal` and `embedq` exit 0; the sealed file loads in `ppl`.
4. No `LLVQ_*` variable inherited by the shell beyond the ones listed.

## 4. Signed predictions

*Computed* from the shapes (28 layers, hidden 1,024, inter 3,072, 16 query heads, 8 KV heads of 128, vocabulary
151,936, tied embedding): 352.3 M Tetra weights at 2.166 b/weight, 88.1 M int4 g128 weights at 4.25, the 155.6 M
embedding at 4.5, norms in f16.

- Sealed file: 0.23 GB [0.21, 0.26].
- b/param, whole model: 3.08 [2.95, 3.25]. The embedding is a third of the parameters at this size, which is why the
  figure sits above the 2.7 of the served files.
- Perplexity f16, WikiText-2, 12 x 4,096: 26 [21, 34] (*estimated*: the 0.6B's own f16 perplexity is not in the
  repository; Tetra cost the 4B 12.24 to 12.58, and a smaller model pays more).

## 5. What this does not establish

The acceptance rate against the sealed targets, the drafter's speed on the card, and anything about the targets'
tokens. Those are the card gate's (its own prereg).
