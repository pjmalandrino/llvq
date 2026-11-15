# Preregistration. Decode speed at long context (O13), R3: which quantized KV cache holds the tolerance

Written on 2026-11-14 on branch `recherche/debit-kv-2026-11-14`, once the four arms existed and their Mac gates had
run (§3). Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-recherche-debit-kv-ladder-2026-11-14-ECARTS.md`.

Four jobs, one an arm, launched by the operator's go: `ARM=<arm> ops/jobs/recherche-debit-kv-ladder.sh`. Cost: $4.0
to $5.3 for the four (*estimated*, §6). Cap: $7.20 (four 60 min timeouts, *computed* at $1.80/h). Spent before it:
$0 of the phase's $10.

R2, the KV capture (`preregistration-recherche-debit-kv-capture-2026-11-14.md`, stamped), is not run: the operator
judged its chance too low on 2026-11-14 and the phase moved to a smaller cache. Its code stays on the branch.

## 1. The question

At 8,192 tokens of context the decode attention of the sealed 4B is bound by the bytes of its KV cache: 1,208 MB in
f16, 2,646 µs of a 6,211 µs token (*measured*, large-4e-hors-poids-etude). A cache in fewer bits reads fewer bytes.

The question: which of four cache formats keeps the logits of the sealed 4B inside the served tolerance, ±0.12 % of
paired perplexity against the f16 cache, at 8,192?

It is a quality measurement. Each arm is a round trip: quantize, widen back, the cache keeps its dtype. No byte is
saved, no kernel reads a new format, and no speed is read. No arm becomes servable by these jobs.

## 2. The arms (`llvq-llm/src/kvladder.rs`, commits a4d81dc, e294ea5, e53afd9)

| arm | keys | values | B a position at 4B (*computed*) | share of f16 |
|---|---|---|---|---|
| f16 (reference) | f16 | f16 | 147,456 | 1 |
| `k8v8` | int8 affine g64 (the shipped q8) | the same | 78,336 | 0.531 |
| `k8v4` | int8 affine g64 | int4 affine g64 | 59,904 | 0.406 |
| `k4v4c` | int4 symmetric, one static step a RoPE pair | int4 affine g64 | 39,168 | 0.266 |
| `k4v4h` | Hadamard a head, then int4 affine g64 | the same | 41,472 | 0.281 |

`k4v4c`'s steps are the 0.999 quantile of each pair's norm over 7 levels, read on the first 4,096 tokens of
`dclm-edu` by one prefill on the f16 cache, frozen before WikiText-2 is read. The clip is the default; no other value
is run. Every arm codes a position from that position alone.

## 3. Stage 1 on the Mac at $0 (*measured*, 2026-11-14, CPU unit tests through the guard)

- Token independence at the quantizer and through the cache on the `Cat` and `Grow` stores, bit for bit, every arm.
- The f16 arm's NLL is bit-identical with and without the ladder state installed; every arm changes the logits.
- Six mutants applied on one matched site each and killed: q8 at 127 levels, `k8v4` values at 8 bits, `k4v4c` paired
  (i, i+1), `k4v4c` without its round, `k4v4h` with the identity, the arms passed uncoded into the cache.
- SQNR on synthetic sources, keys / values: `k8v8` 40.19 / 45.52 dB, `k8v4` 40.19 / 20.94, `k4v4c` 16.25 / 20.94,
  `k4v4h` 23.19 / 20.91.
- `cargo clippy -p llvq-llm --all-targets` and `ops/check-linux-on-mac.sh` clean; `env_registry` passes at 19 modes.

Not verified on the Mac: anything on a model. The arms' ops (`round`, the U8 cast, the f32 matmul) have never run on
the CUDA backend. The calibration has never read real keys.

## 4. What runs, an arm

`ops/jobs/recherche-debit-kv-ladder.sh` (l40sx1, output `/out/recherche-debit-kv-ladder-2026-11-14/<arm>`):
1. the oracle; the sealed 4B's size and sha256;
2. the load check, which stops the job: `LLVQ_KV_LADDER=<arm> LLVQ_GQA_PPL=256,2,0,ladder fusedrun <sealed>`;
3. the measurement: `LLVQ_KV_LADDER=<arm> LLVQ_GQA_PPL=8192,12,0,ladder fusedrun <sealed>`.

WikiText-2 test at revision 740312ad, windows 0 to 11 of 8,192 tokens, each fed one true token at a time through the
two caches in lockstep, both under the served 4E attention form (grow(256), the kernel on every decode row from
position 0). The flags are the served ones: tetra48, embed q4, rot share 1, fuse 0, `LLVQ_KV=f16`.

The four jobs run at the same time. They share nothing but the image and the sealed file.

## 5. The reading, fixed before the run

The score of an arm is the last table's line: arm / f16 − 1 in percent of perplexity, with its paired 95 % t
interval over the 12 windows (`paired_ppl_pct`).

| what the arm reads | verdict |
|---|---|
| the whole interval inside [−0.12, +0.12] | PASSES |
| the whole interval above +0.12 | FAILS |
| anything else | UNDECIDED |

- The format kept is the passing arm with the fewest bytes a position. `k4v4c` and `k4v4h` are within 6 % of each
  other in bytes: if both pass, the one with the lower point estimate is kept.
- If no arm passes and `k8v8` fails, the lead is dead: no kernel is written for a quantized cache.
- An UNDECIDED arm may be extended to windows 12 to 35 under its own deviation note and go, once. The 36 windows are
  then read together. No arm is extended twice and no window is dropped.
- A passing arm is a candidate, not a served format. Before any kernel: the 47 near-tie texts and GSM8K-200 not
  separated, under their own prereg.
- A job stopped by its load check, or cut before 12 windows, is void for its arm and says nothing of the arm.

Not read as a result: top-1 agreement and max |dlogit|. They are printed and reported.

## 6. Cost

*Estimated*: `LLVQ_GQA_PPL=8192,6,6,kernel` fed 6 windows through two arms in 940 s, 157 s a window
(attn-t8-r2-2026-11-04-brut/ppl-8192.txt). The round-trip ops were never timed: 160 to 220 s a window assumed, 32 to
44 min for 12 windows, $1.0 to $1.3 an arm. A measurement cut by its 52 min timeout is void: that happens above
260 s a window.

## 7. Signed predictions (main session)

1. `k8v8`: +0.03 % [−0.02, +0.09] as a point; it PASSES at 55 %, UNDECIDED at 40 %, FAILS at 5 %.
2. `k8v4`: +0.12 % [+0.03, +0.40]; PASSES at 15 %, UNDECIDED at 45 %, FAILS at 40 %.
3. `k4v4h`: +0.6 % [+0.15, +2.5]; PASSES at 5 %, FAILS at 80 %.
4. `k4v4c`: +2 % [+0.4, +10]; PASSES at 2 %, FAILS at 93 %.
5. `k4v4h` reads below `k4v4c`: 85 %.
6. The half width of a paired interval at 12 windows of 8,192 is 0.06 % [0.03, 0.12] for `k8v8`.
7. At least one job is void (an op refused on CUDA, or the timeout): 30 %.
8. A window takes 185 s [150, 240].

## 8. Not established by these jobs

- Any tok/s, any VRAM: the arms store f16.
- Any size but the 4B, any text but WikiText-2, any context but 8,192.
- That a kernel can read the kept format at the byte rate of the f16 cache.
- The prefill path: the decode path alone is scored.
