# Preregistration. The grouped attention's perplexity against the published one, by prefill and by decode

**Written on 2026-11-01, to be committed and TIMESTAMPED BEFORE the run.** Operator go to launch given 2026-11-01 ("Go, plafond 2 $"). Cost: about
**$1.05** on l40sx1, *estimated* below (§8), timeout 60 min, so $1.80 at worst. Cap: **$2**. Spent before
it: $247.39 over 229 priced jobs (*measured*, `docs/data/jobs.csv`).

## 1. Why this run exists

The GQA A/B of 2026-11-01 (`docs/mesures/gqa-ab-2026-11-01.txt`) measured the grouped attention form, on the `Grow`
store, at 1.1028, 1.3823, 1.5995 and 2.3392 times the published decode speed after prompts of 5, 2,048, 4,096 and
8,192 tokens. It returned the published 256 tokens at every length. It is not bit for bit: the logits move by up to
6.6e-2. By that prereg's rules it falls under decision T8.

On 2026-11-01 the operator took T8 for this change: a bounded tolerance replaces bit identity, provided a quality
control passes first ("Tolérance + contrôle"). This run is that control, and §2 is the tolerance it applies.

## 2. The tolerance

The grouped form may replace the expanded one on the served path if, on the sealed 4B through the served kernel, over
the 12 windows of 4,096 tokens of WikiText-2 test that `ppl-scelles-2026-10-04` scored (token fingerprint
`3f1baca9033bf251`), **both** of these paired 95 % intervals lie inside **[−0.12 %, +0.12 %]**:

- the prefill path, `ppl_grouped / ppl_published − 1`, each window in one forward;
- the decode path, the same ratio, each window fed one true token at a time through the cache.

The margin is the paired interval of an A/B at constant file that `docs/ETAT.md` §4 gives, 0.12 % in perplexity
(*measured*, `kvq8-4b-2026-08-15`). A change that stays inside it is no larger than the noise of comparing one file to
itself under a runtime change.

## 3. What the code now holds, and what the Mac showed

The commit that carries this file adds, on branch `debit-trims`:

- `Qwen3::decode_nll_paired`: one window scored token by token through the cache under two arms in lockstep, one
  cache set per arm, the same true token fed to both, the next true token scored. Each step also compares the two
  arms' logits: the top-1 token, and the largest absolute difference.
- `fusedrun`'s measurement mode `LLVQ_GQA_PPL=<ctx>,<windows>`: for each window, `window_nll` under each arm (the
  prefill path), then `decode_nll_paired` (the decode path). Arms: `cat+expanded` (published) and `grow(256)+grouped`.
  The windows are the sampling unit; each interval is a paired t interval over them.

*Measured* on the Mac CPU, tiny random model with groups of 4 (`paired_decode_scores_what_the_window_scores`): one arm
against itself is bit for bit; the decode path agrees with `window_nll` within 1e-5 relative; the grouped form moves
the summed NLL by less than 1e-6 relative and the logits by less than 6e-7; the shared form moves nothing. The
interval arithmetic is checked by hand values (`paired_ppl_interval`). 7 mutants of the new code, each killed.

## 4. What runs

One image built from the commit that carries this file. One job, l40sx1, the sealed 4B
(`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`), config `configs/qwen3-4b-tetra-e4.json`
(`ops/jobs/gqa-ppl.sh`).

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The published arm, replayed**, as in step 2 of the GQA A/B: it checks that the default path did not move.
3. **The control.** The served flags spelled out, plus `LLVQ_GQA_PPL=4096,12`, cudarc's events off as on the served
   door.

## 5. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The sealed file and the config at their bytes and sha256 on the mount.
3. Every load prints `int4: tv_q4_hh equals tv_q4_h bit for bit`.
4. Step 2 prints `256 tokens identical to the dense arm`, and its decoded line hashes to `8784ffc4…`.
5. Step 3 prints `cudarc event tracking: off (one stream, ordered by issue)`, the token fingerprint
   `3f1baca9033bf251`, 12 window lines and both summary lines.
6. **The harness.** On the published arm, the decode path's perplexity over the prefill path's lies in
   [0.99, 1.01]. Both paths score the same tokens; a scorer that read the wrong target or position would miss by far
   more.
7. The header of every step reads `tile 64`, and no `LLVQ_*` variable is in the job environment.

## 6. Signed predictions

| quantity | point | interval |
|---|---|---|
| published, prefill ppl through the kernel | **12.58** | [12.45, 12.72] |
| published, decode over prefill ppl | **1.000** | [0.998, 1.002] |
| grouped / published − 1, prefill, point | **0.000 %** | [−0.02 %, +0.02 %] |
| grouped / published − 1, decode, point | **0.000 %** | [−0.03 %, +0.03 %] |
| both paired intervals inside [−0.12 %, +0.12 %] | **yes** | |
| decode top-1 agreement over 49,140 steps | **99.7 %** | [99.0 %, 99.95 %] |
| decode max \|dlogit\| over every step | **0.12** | [0.04, 0.5] |

Reasoning, *computed*. The dense reconstruction of the same file scored 12.5834 on the card (`ppl-scelles`); the
kernel computes the same products in another order. The two arms differ by rounding only, symmetric in sign, so their
per-window NLL differences should average to zero with a spread far under the band. The A/B's first decode steps moved
the logits by 2.5e-2 to 6.6e-2; the largest over 49,140 steps should be a few times that. A top-1 token flips only
where the two leading logits sit within that distance.

## 7. Decision rules

- **Both intervals inside the band:** the tolerance is met. The grouped form on the `Grow` store becomes the served
  attention in a follow-up commit, with a new reference decoded line for gate 4 of later runs. How it is switched on,
  an engine default of the served door or a field of the served config, is put to the operator with that commit.
- **An interval leaving the band on one side of zero:** the grouped form changes quality beyond the noise. It is not
  served, and the expanded form stays.
- **An interval wider than the band but containing zero:** the control lacks power. Nothing is served; the operator
  chooses between more windows and stopping.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h. Container and image pull 3 min, oracle 2 min, step 2 about 3 min. Step 3: load and
corpus 1 min; per window, two prefills of 4,096 tokens at about 4.1 ms a token plus the head row by row, about 37 s,
then 4,095 lockstep decode steps at about 22 ms (12.2 and 8.8 ms for the two arms at a mean of 2,048 positions, plus
the comparisons), about 90 s. 12 windows, about 25 min. In all about 35 min, **$1.05**. Timeout 60 min, **$1.80** at
worst.

## 9. What this run does not establish

- Task accuracy. GSM8K through the served kernel would add about $2; the dump of the expanded arm exists
  (`gsm8k-wave1-2026-09-26-brut/run1-served/`), so only the grouped arm would run. Not in this run.
- The product's 8,192 context: one prefill call takes at most 4,096 rows, so windows stop there.
- Anything at 8B or 14B, on another card, or on Metal.
