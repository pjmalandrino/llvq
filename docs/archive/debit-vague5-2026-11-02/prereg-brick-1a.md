DRAFT, not stamped, not committed

# Preregistration. The sealed 4B in f32 on Metal against its own f16 card NLLs: how far f16 alone moves this object

**Draft of 2026-11-02, to be committed and TIMESTAMPED BEFORE the run.** Operator go to launch: **not given**. Cost:
**$0**, a local run on the Mac with today's binary and no new code, about 5-10 min of Metal (*estimated*, §7). `ppl`
reads no `LLVQ_THREADS`, and this run is GPU-bound; it starts under `nice 10` (METHODE §6). Spent before it: $248.84
over 237 rows of the registry (*measured*, `docs/data/jobs.csv`); this run adds nothing. It is step S1 of
`wave5-deep/brick.md` §5 and needs nothing before it.

## 1. Why this run exists

The brick (Floor 4K) replaces the f32 activation of the 168 Tetra projections by an int16 per 384-coordinate slice with
a power-of-two exponent. That is a T8 event. On synthetic activations the int16 step adds 3 % of the noise power that
f16 rounding already puts on an activation (Gaussian, median), 13-15 % (Student t, ν = 4, median) and 1-4 % for a
massive channel rotated by the repository's construction (*computed*, 2,000 vectors a case,
`wave5-deep/calc/brick/synthetic_actq.out`).

The card serves the 4B in f16. Its twelve per-window NLLs of the sealed file on WikiText-2 are on file
(`docs/mesures/ppl-scelles-2026-10-04-brut/ppl-4b.txt`, 12.5834, fingerprint `3f1baca9033bf251`). The same file has
no per-window f32 run on the Mac. Pairing the two measures, on this object, f16 everywhere (weights, activations and the
card's dense arithmetic) plus the device. On the 8B the device alone, at f16 on both sides, gave
[−0.009 %, +0.010 %] (*measured*, `ppl-bases-carte-2026-10-05.txt`).

If f16 everywhere moves this object by less than ±0.02 %, a perturbation with a fraction of f16's noise power, at a
subset of the inputs, is expected to move it by less. That expectation is *estimated* by noise-power dominance and is
not a bound. int16 per slice puts a uniform absolute error on the small elements of a slice, which f16 does not, so the
two noises do not have the same distribution. This run therefore informs U1 and never replaces the injection control
(`prereg-brick-1.md`).

## 2. What runs

The sealed 4B, `~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin` (sha256 `886391a8…`, checked in full at gate 2), today's
binary at a commit named in the journal, no code change.

1. **Oracle** on Qwen3-0.6B, Metal (rule 10).
2. `ppl 4096 12 metal ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin` with `LLVQ_DTYPE` unset (f32) and stderr kept
   (`2>`), since `ppl` prints its per-window NLLs there (METHODE §5).
3. On the Mac, after the run: the twelve card NLLs of `ppl-4b.txt` and the twelve Mac NLLs, paired window by window.

## 3. Gates, each one voids the run

1. `oracle` prints MATCH on Metal.
2. The sealed file's full sha256 equals the published one and the one in `ppl-scelles-2026-10-04-brut/files.sha256`.
3. The token fingerprint reads `3f1baca9033bf251`; 12 window lines with `n 4095` each; the header reads `dtype f32`.
4. No `LLVQ_*` variable is set; `LLVQ_CONFIG` in particular is unset, so both arms are the dense reconstruction.
5. The card file's sha256 is recorded in the journal before the pairing is computed.

## 4. Outcomes reported, never gated

The Mac perplexity; the paired mean ΔNLL, card f16 minus Mac f32, with its paired t interval over 12 windows; the same
as a perplexity ratio; the windows worse; the largest per-window difference.

## 5. Signed predictions

| quantity | point | interval |
|---|---|---|
| Mac f32 ppl, WikiText-2 | **12.58** | [12.55, 12.62] |
| card f16 over Mac f32 − 1, paired | **+0.005 %** | [−0.02 %, +0.03 %] |
| windows worse on the card | **7 of 12** | [4, 10] |
| paired interval inside [−0.02 %, +0.02 %] | **yes** | |

Reasoning: the 8B's device effect at f16 on both sides was ±0.01 % (*measured*). The one earlier cross-dtype pair on
record, 8.9893 in f32 on the Mac against 8.9899 in f16 on the card, differs by +0.007 % unpaired (*computed* on
*measured* values quoted in `ppl-bases-carte-2026-10-05.txt`, which does not name that model). f16 should cost
slightly, hence the positive sign.

## 6. Decision rules

A paired interval is inside a band when it lies within it. Apply in order; the first that matches decides. None of them
decides the T8.

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

$0. Twelve windows at 5-31 s each (*estimated* between two *measured* anchors). The 4B in f32 on Metal ran windows of
1,024 positions at 1.1-1.4 s each (`kl-temperature-4b-2026-09-16-brut/wikitext.log:4-10`), so ×4-5 at 4,096 positions.
The 8B in f16 on Metal ran 4,096-position windows at 21.6-30.6 s (`dclm-8b-2026-09-21-brut/run.log:161-172`,
`dclm-8b-rowscales-2026-09-21-brut/ppl-ft-f16.txt`), with the same weight bytes as the 4B in f32. Add one 35 s load
(*measured*, kl-temperature) and the oracle (42 s, METHODE §6): 2-7 min, about 10 min with the build.

## 8. Deviations

Any departure from this file goes in `<this file>-ECARTS.md`, beside it, never inside it.

## 9. What this run does not establish

- The int16 effect itself: `prereg-brick-1.md` measures it.
- Anything through the kernel, at another size, or on another file.
- Which of f16 storage, f16 arithmetic or the device carries the difference.
