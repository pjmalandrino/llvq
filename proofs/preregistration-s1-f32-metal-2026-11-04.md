# Preregistration. The sealed 4B in f32 against f16, both on Metal: how far f16 alone moves this object

**Written 2026-11-04, timestamped with `ots stamp` before the first measurement, not committed.** Cost: **$0**, a
local run on the Mac (M3 Max, 64 GB) with the binary of commit `06e1602` and no code change. Spent before it: $248.84
over 237 rows of the registry (*measured*, `docs/data/jobs.csv`); this run adds nothing. It is step S1 of the brick
design (`wave5-deep/brick.md` §5, a scratchpad draft) and needs nothing before it. It replaces the draft
`prereg-brick-1a.md` of 2026-11-02, which was never stamped; §10 lists what changed and why.

## 1. Why this run exists

The brick (Floor 4K) replaces the f32 activation of the 168 Tetra projections by an int16 per 384-coordinate slice with
a power-of-two exponent. That is a T8 event. On synthetic activations the int16 step adds 3 % of the noise power that
f16 rounding already puts on an activation (Gaussian, median), 13-15 % (Student t, ν = 4, median) and 1-4 % for a
massive channel rotated by the repository's construction (*computed*, 2,000 vectors a case,
`wave5-deep/calc/brick/synthetic_actq.out`).

The card serves the 4B in f16. What f16 alone costs this object has never been measured: the one cross-dtype pair on
record put f16 on a card against f32 on the Mac, so it mixed the dtype with the device. This run holds the device, the
binary, the file and the token stream fixed and changes only `LLVQ_DTYPE`. On the 8B the device alone, at f16 on both
sides, gave [−0.009 %, +0.010 %] (*measured*, `docs/mesures/ppl-bases-carte-2026-10-05.txt`).

If f16 everywhere moves this object by less than ±0.02 %, a perturbation with a fraction of f16's noise power, at a
subset of the inputs, is expected to move it by less. That expectation is *estimated* by noise-power dominance and is
not a bound. int16 per slice puts a uniform absolute error on the small elements of a slice, which f16 does not, so the
two noises do not have the same distribution. This run therefore informs U1 and never replaces the injection control
(`prereg-brick-1.md`, step S3).

## 2. What runs

The sealed 4B, `~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin` (sha256 `886391a8…`, checked in full at gate 2). Binary
`ppl` built from commit `06e1602` with `--features metal` into a separate target directory, its sha256 in the journal.
Both arms run under `nice -n 10`, one after the other, never at the same time.

1. **Oracle** on Qwen3-4B, Metal, f32 (rule 10). Qwen3-4B, not Qwen3-0.6B: it is the checkpoint already in the local
   cache, and the oracle's question is the backend, which this answers at the size under test.
2. Arm F32: `ppl 4096 12 metal ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin` with `LLVQ_DTYPE` unset (f32), stderr kept.
3. Arm F16: the same command with `LLVQ_DTYPE=f16`, stderr kept.
4. After both: the twelve F16 window NLLs paired with the twelve F32 window NLLs, window by window.
5. Secondary, after both: the twelve F16 Metal NLLs paired with the twelve card f16 NLLs of
   `docs/mesures/ppl-scelles-2026-10-04-brut/ppl-4b.txt` (12.5834, fingerprint `3f1baca9033bf251`). That pair holds
   the dtype and changes the device only. It is reported, never gated, and it is not the measurement of this file.

## 3. Gates, each one voids the run

1. `oracle` prints MATCH on Metal.
2. The sealed file's full sha256 equals `886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8`, the one in
   `ppl-scelles-2026-10-04-brut/files.sha256`.
3. Both arms: token fingerprint `3f1baca9033bf251`; 12 window lines with `n 4095` each; the header reads `dtype f32`
   on arm F32 and `dtype f16` on arm F16; both on a Metal device.
4. No `LLVQ_*` variable is set except `LLVQ_DTYPE=f16` on arm F16 and `LLVQ_THREADS=12`, which `ppl` does not read
   (only `smoke.rs` reads it). `LLVQ_CONFIG` in particular is unset, so both arms are the dense reconstruction.
5. The same `ppl` binary, by sha256, runs both arms.
6. The card file's sha256 is recorded in the journal before the secondary pairing is computed.

## 4. Outcomes reported, never gated

Both Metal perplexities; the paired mean ΔNLL, F16 minus F32, with its paired t interval over 12 windows
(t₀.₉₇₅,₁₁ = 2.201); the same as a perplexity ratio, exp(ΔNLL) − 1; the windows worse under F16; the largest
per-window difference. The same four for the secondary pair (card f16 minus Metal f16).

## 5. Signed predictions

| quantity | point | interval |
|---|---|---|
| Metal f32 ppl, WikiText-2 | **12.58** | [12.55, 12.62] |
| Metal f16 ppl, WikiText-2 | **12.58** | [12.55, 12.62] |
| Metal f16 over Metal f32 − 1, paired | **+0.005 %** | [−0.02 %, +0.03 %] |
| windows worse under f16 | **7 of 12** | [4, 10] |
| paired interval inside [−0.02 %, +0.02 %] | **yes** | |
| secondary: card f16 over Metal f16 − 1, paired | **+0.000 %** | [−0.02 %, +0.02 %] |

Reasoning: the 8B's device effect at f16 on both sides was ±0.01 % (*measured*). The one earlier cross-dtype pair on
record, 8.9893 in f32 on the Mac against 8.9899 in f16 on the card, differs by +0.007 % unpaired (*computed* on
*measured* values quoted in `ppl-bases-carte-2026-10-05.txt`, which does not name that model). f16 should cost
slightly, hence the positive sign.

## 6. Decision rules

They read the primary pair only (F16 against F32, both Metal). A paired interval is inside a band when it lies within
it. Apply in order; the first that matches decides. None of them decides the T8.

- **V.** A gate fails: void, the defect is journaled, no reading.
- **A.** The interval lies inside [−0.02 %, +0.02 %]. f16 everywhere moves this object by less than a sixth of the
  ±0.12 % band. The injection control keeps its rules and is read as a confirmation; the journal writes the dominance
  argument as an estimate.
- **B.** The interval lies inside [−0.12 %, +0.12 %] but not inside ±0.02 %. The injection control decides as
  written, and the f16 effect is journaled beside U1.
- **Otherwise.** f16 alone moves this object past the band, or the interval is too wide to say. The dominance argument
  gives nothing. The injection control decides alone. The result goes to the operator as a fact about the served
  dtype, since every published 4B perplexity is f16 on a card.

## 7. Cost

$0. Twelve windows at 5-31 s each per arm (*estimated* between two *measured* anchors). The 4B in f32 on Metal ran
windows of 1,024 positions at 1.1-1.4 s each (`kl-temperature-4b-2026-09-16-brut/wikitext.log:4-10`), so ×4-5 at
4,096 positions. The 8B in f16 on Metal ran 4,096-position windows at 21.6-30.6 s (`dclm-8b-2026-09-21-brut/run.log`),
with the same weight bytes as the 4B in f32. Two arms, one load each, and the oracle: 5-15 min.

## 8. Deviations

Any departure from this file goes in `<this file>-ECARTS.md`, beside it, never inside it.

## 9. What this run does not establish

- The int16 effect itself: `prereg-brick-1.md` measures it.
- Anything through the kernel, at another size, or on another file.
- Which of f16 storage, f16 arithmetic or the f16 KV cache carries the difference: `LLVQ_DTYPE` moves all three. The
  result line prints `kv f16` on both arms, but that names the mode, and the mode keeps the cache at the model's dtype
  (`kvq.rs:44-46`), so it is f32 on arm F32.

## 10. What changed from the draft `prereg-brick-1a.md` (2026-11-02, never stamped)

- The pair. The draft paired a Mac f32 run with the card's f16 NLLs, which changes the dtype and the device at once.
  A review named that confound. Both arms are now on Metal with the same binary; the card file moves to a secondary
  pair that changes the device only.
- The oracle runs on Qwen3-4B, cached locally, instead of Qwen3-0.6B, which is not.
- Gate 4 names `LLVQ_THREADS=12`, set by the operator's habit for local jobs and read by no binary run here.
- One prediction added per new quantity (the f16 Metal perplexity, the secondary pair). The draft's predictions are
  kept as they were.
