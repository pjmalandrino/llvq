DRAFT, not stamped, not committed

# Preregistration. The int16 activation of the brick, measured on the Mac before any kernel exists

**Draft of 2026-11-02, revised the same day after review; to be committed and TIMESTAMPED BEFORE the run.** Operator
go to launch: **not given**. Cost: **$0**, a local run on the Mac, about 15-60 min of Metal and part of the CPU
(*estimated*, §8), after 2-4 days of code. Machine load: the injection's CPU pool is sized by `LLVQ_THREADS`, which this
commit teaches `ppl` to read (today only `smoke.rs:669` and `errmap.rs:314` read it). Its default is 12, ncpu − 4 on
this 16-core M3 Max, and the run header prints it. The run starts under `nice 10` (METHODE §6). Spent before it:
$248.84 over 237 rows of the registry (*measured*, `docs/data/jobs.csv`); this run adds nothing. It is step S3 of
`wave5-deep/brick.md` §5. Step S1 (`prereg-brick-1a.md`) is cheaper and may run first; nothing here depends on it.

## 1. Why this run exists

The brick (Floor 4K of the throughput program) replaces the f32 activation of the 168 Tetra projections by an int16 per
384-coordinate slice with a power-of-two exponent. Every block dot then becomes an exact integer, which is what makes
the k-row, prefill and host paths return the same bits. It is a T8 event: a numerical change served under the ±0.12 %
tolerance only after a control passes.

The estimate so far is synthetic, over 2,000 vectors a case (*computed*, `wave5-deep/calc/brick/synthetic_actq.out`).
f16 rounding costs 2.08e-4 relative rms. int16 per slice costs 3.5-3.6e-5 on Gaussian vectors, 3 % of f16's noise
power at the median and 5 % at p99. On Student t vectors (ν = 4) it costs 7.6-7.9e-5, 13-15 % at the median and
30-45 % at p99. A massive channel rotated by the repository's construction costs 1-4 % at the median. No activation of
the real model was ever dumped: the bucket holds none and the Mac holds none (`wave5-deep/bucket-listing.txt`). This
run measures what decides whether the int16 semantics is worth coding: the paired perplexity of the model with that
quantization applied. It also records the shape of the real activations at the projection inputs, which informs the
design and gates nothing.

## 2. What the code would hold

The commit that carries this file would add, on a branch, two measurement modes, both refused beside `LLVQ_CONFIG`:

- `LLVQ_ACTQ=<arms>` in `ppl` (dense path, Metal, f32). Before each Tetra projection of the sealed file, its input rows
  x go through `x̃ = Qᵀ · deq(quant(Q · x))`. Q is that record's own rotation: `Rotation::new(d_in, rotation_seed)`
  with the seed stored in the record (`llvq-artifact/src/format.rs:484`), the one `decode_matrix` un-rotated it with
  (`format.rs:1527`). The loader keeps the seed beside each dense Tetra projection, since `Proj::Dense` carries none
  (`llvq-llm/src/sealed.rs:116-117`). `quant` is the brick quantizer of `brick.md` §3.1, applied to the f32-rounded
  rotated vector, as the card would see it. Arms: `off` (published), `idrot` (Q then Qᵀ, no quantization), `i16s`
  (int16 per slice, exponent spread capped at 3), `i24s` (int24 per slice), `i8v` (int8 per vector, amax/127, the
  ExLlamaV3 plain mode, a positive control). The int4 inputs and the head are untouched in every arm. The windows are
  the sampling unit, all arms of a window run back to back, and each interval is a paired t interval.
- `LLVQ_ACTQ_STATS=<dir>` in the same run, for the `off` arm's prefill only. Per layer, site and token: the slice and
  vector amax/rms in the rotated basis (q/k, gate/up and Tetra down inputs) and per 128-group in the natural basis (v, o
  and int4 down inputs); whether the cap binds; the int16, int24 and f16 relative errors of the token. Flat
  little-endian files, as `hcapture` writes.
- At load, printed in the run header: (a) for each of the 168 Tetra records, the seed the hook holds, and whether it
  equals `effective_rotation_seed(ROTATION_SEED, block, act)`, the encoder's keying that `hcapture` reproduces
  (`l36-capture-2026-09-13.txt:39-45`); (b) for one block of one record per projection type (q, k, gate, up, down),
  `W_dense · Qᵀ` rebuilt in f64 against the rotated-basis block that `reconstruct_shape_gain` decodes. The identity
  `QᵀQ = I` cannot see a wrong basis; check (b) can, because it ties the hook's Q to the Q that rotated the weights.
- The line `actq threads N` in the run header.

On the Mac CPU, before the run (to be shown in the commit): the quantizer equals a hand-computed reference on edge cases
(exact powers of two, ties, mantissa 1.99997, zero slice, non-finite); `idrot` returns the input within 1e-6 relative; a
mutant per edge case is killed; the mutant "seed + 1" fails check (b).

## 3. What runs, in this order

The sealed 4B, `~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin` (sha256 `886391a8…`, checked in full at gate 2), on the
Mac, Metal, f32, one binary for every step, its sha256 in the journal.

1. **Oracle** on Qwen3-0.6B, Metal (rule 10).
2. **The hook-free reference.** `ppl 4096 12 metal <sealed> wikitext2` with `LLVQ_ACTQ` unset.
3. **The wiring checks, before any control window.** `LLVQ_ACTQ=idrot` on the first 2 WikiText-2 windows and
   `LLVQ_ACTQ=i8v` on the first 4, with checks (a) and (b) printed at load. The control starts only if gates 1-6 pass.
4. **The control.** `LLVQ_ACTQ=off,i16s,i24s ppl 4096 12 metal <sealed> wikitext2`, then the same on `c4`. The C4
   windows served no tolerance control so far (the grouped form's T8 used WikiText-2 only): they are the fresh windows.
5. **The statistics**, written during step 4's `off` arm on WikiText-2.

Every step keeps stderr (`2>`), where `ppl` prints the per-window NLLs (METHODE §5).

## 4. Gates, each one voids the run

1. `oracle` prints MATCH on Metal.
2. The sealed file's full sha256 matches the published one.
3. The WikiText-2 token fingerprint reads `3f1baca9033bf251`; the C4 fingerprint is the same in every arm; 12 window
   lines per arm and per corpus, and the summary lines.
4. **The instrument is transparent:** `idrot` over the hook-free reference lies within ±0.0005 % on its 2 windows.
5. **The instrument discriminates:** `i8v` moves the perplexity by more than 0.05 % on its 4 windows. A quantization
   256 times coarser than the one under test that moved nothing would mean the hook is not applied.
6. **The basis is the served one:** every one of the 168 Tetra records carries a seed and the hook holds it, and check
   (b) agrees within 1e-6 relative for the five projection types.
7. **The hook is inert when off:** step 4's `off` arm on WikiText-2 reproduces step 2's twelve NLLs bit for bit.
8. Step 2 reads 12.40-12.80, the window of the dense reconstruction on the card (12.5834 in f16, *measured*,
   `ppl-bases-carte-2026-10-05.txt`), widened for f32 on Metal.
9. The header prints `dtype f32` and `actq threads N`, and no `LLVQ_*` variable is set beyond `LLVQ_ACTQ`,
   `LLVQ_ACTQ_STATS` and `LLVQ_THREADS`.

## 5. Outcomes reported, never gated

Per corpus and arm: ppl, the paired ratio and its interval, the windows worse, top-1 agreement over all positions, the
largest |Δlogit|. Per site: percentiles 50 and 99 and the maximum of slice and vector amax/rms; the share of tokens
where the cap binds; the median and p99 of the int16 and int24 relative errors and their ratio to the f16 rounding
error of the same token. The seed cross-check (a), record by record.

## 6. Signed predictions

| quantity | point | interval |
|---|---|---|
| step 2 ppl, WikiText-2, Metal f32 | **12.58** | [12.45, 12.72] |
| `i16s / off − 1`, WikiText-2 | **0.000 %** | [−0.02 %, +0.02 %] |
| `i16s / off − 1`, C4 | **0.000 %** | [−0.02 %, +0.02 %] |
| both `i16s` paired intervals inside [−0.12 %, +0.12 %] | **yes** | |
| `i24s / off − 1`, both corpora | **0.000 %** | [−0.005 %, +0.005 %] |
| `i8v / off − 1`, 4 windows | **+0.3 %** | [+0.05 %, +2 %] |
| top-1 agreement `i16s` against `off` | **99.95 %** | [99.7 %, 99.99 %] |
| slice amax/rms at the Tetra inputs, p50 / p99 / max | **3.2 / 5 / 8** | [2.0, 6] / [3.5, 13] / [4.5, 40] |
| tokens where the cap binds, Tetra inputs | **0.1 %** | [0 %, 5 %] |
| int16 per-slice error over f16 rounding error, median | **0.17** | [0.10, 0.50] |
| group amax/rms at the int4 inputs, p99 | **6** | [3.5, 40] |
| check (a): stored seeds equal the encoder's keying | **168 of 168** | |

Reasoning, *computed* on synthetic vectors (`calc/brick/synthetic_actq.out`): slice amax/rms reads 3.1-3.2 at p50 and
4.5-5.4 at p99 for Gaussian vectors, 2.2-2.9 and 3.9-5.1 for a rotated massive channel, 5.4 and 12.3-12.5 for t4. The
error ratio reads 0.17 (Gaussian), 0.11-0.20 (massive channel) and 0.37-0.38 (t4). The rotation spreads outliers over
every coordinate of a site, so the Tetra inputs should look near-Gaussian, and the int16 step should then sit below
the device change of `ppl-bases-carte` (±0.01 % paired, *measured*). The natural basis keeps its outliers, hence the
wide int4 band.

## 7. Decision rules

A paired interval is **inside** when it lies within [−0.12 %, +0.12 %], and **wholly outside** when it does not meet
that band. Apply in order; the first rule that matches decides.

- **V.** Any gate fails, or an `i24s` interval is not inside while the `i16s` interval of the same corpus is inside.
  int24 is strictly finer than int16, so the second case points at the instrument. Void: the defect is journaled and
  nothing is read.
- **A.** Both `i16s` intervals inside: int16 per slice is the brick's default semantics. The next steps (S4, then S5)
  still wait for the operator's go and decisions 7 and 9.
- **D.** No `i16s` interval wholly outside, and an `i16s` interval wider than 0.24 pp that contains 0: the control
  lacks power. The operator chooses between more windows and a stop.
- **B.** Both `i24s` intervals inside: int24 per slice becomes the default, with its cost written down (a three-plane
  kernel and a 128-bit row accumulator, weeks of code).
- **Otherwise, C.** The result contradicts the analysis. Nothing is coded; the injection is audited first (basis,
  seeds, the hook's sites), per METHODE §7 on unexpected results.

The statistics are reported and never gated: a gate rests on a fundamental criterion, never on a proxy (METHODE §1).
If p99 slice amax/rms exceeds 32 at a site while rule A holds, the journal writes it as a risk for the 8B and the 14B,
and the choice is the operator's.

## 8. Cost

$0. Ninety window-forwards: 12 in step 2, 6 in step 3, 72 in step 4; 54 of them carry the injection.

- A window-forward takes 5-31 s (*estimated* between two *measured* anchors). The 4B in f32 on Metal ran windows of
  1,024 positions at 1.1-1.4 s each (`kl-temperature-4b-2026-09-16-brut/wikitext.log:4-10`), so ×4-5 at 4,096. The
  8B in f16 on Metal ran 4,096-position windows at 21.6-30.6 s (`dclm-8b-2026-09-21-brut/run.log:161-172`,
  `dclm-8b-rowscales-2026-09-21-brut/ppl-ft-f16.txt`), with the same weight bytes as the 4B in f32.
- The injection adds 3-10 s a window (*estimated*). The f64 rotations come to about 96 G flops a window: 168
  injections a token, 72 K flops each at d_in 2,560 and 544 K at 9,728 (*computed* from the Kronecker construction of
  `llvq-quant/src/rotation.rs`), on 12 threads at an assumed 1-3 GFLOP/s each. Host copies add about 24 GB a window at
  an assumed 30 GB/s.
- Four process starts at 35 s each (*measured*, kl-temperature) and the oracle.

In all 15-60 min, niced, with the GPU busy throughout. The run stops at 2 h whatever its state; that threshold is set
here and is not moved after seeing the clock.

## 9. Deviations

Any departure from this file goes in `<this file>-ECARTS.md`, beside it, never inside it.

## 10. What this run does not establish

- The kernel's numerics. The dense path keeps its f32 matmul and its own rounding; only the activation quantization,
  the dominant change, is reproduced. The rotation runs in f64 on the CPU here, in f32 on the card (`rot_mix`, equal to
  the host, *measured*, `a1-lot-a-2026-08-06.txt:58`). The control through the kernel (S8) remains owed before serving.
- Task scores: GSM8K and MMLU through the kernel are S8's.
- Anything at 8B or 14B, or with the int4 records on the integer template.
- Any speed.
