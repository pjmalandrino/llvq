# Method

The lab rules, one per paragraph: the rule, its reason, the date it was paid
for. The templates are in `docs/templates/`.

## 1. Before any measurement

- A written and timestamped prereg (`ots stamp`) comes before the first
  millisecond. Paid on 2026-08-07: the 1.6× criterion of `Golay70` has no
  earlier record than commit `caef2ac`, 52 min before the measurement
  (*measured*, git).
- The prereg carries the adoption and kill criteria, in numbers, on the exact
  quantity they name. Paid on 2026-08-15: the walk passes at 0.3101 ns/block
  under the 0.45 gate, the block yields 0.6735 (*measured*,
  [P1b](mesures/p1b-marche-bloc-2026-08-15.txt)). The CUDA arm was authorized
  for 57 min.
- The decision rule partitions the space of results and ends with an
  "otherwise". Paid on 2026-09-02: M2b yields +3.60 pp [1.47, 5.79] and no line
  of §5 applies (*measured*, [M2b](mesures/m2b-v4bits-2026-09-02.txt),
  [deviations](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md)).
- A timestamped prereg is never edited. The anchor attests to the bytes; a
  deviation is written beside it, in `<prereg>-ECARTS.md`, and cited. Paid on
  2026-08-26: the anonymization pass (`01fdbe6`) rewrote the preregs of 08-10
  and 08-11. None of the 128 `.md` blobs in git history yields the attested
  digest (*measured*, [ots](mesures/ots-etat-2026-08-26.txt)).
- The prereg carries a signed prediction, on the record, with its sign and its
  order of magnitude. Paid on 2026-09-02: the M1 kill predicted ρ* = 1, the
  measurement yields ρ = 0.7 (*measured*,
  [M1](mesures/m1-hessienne-shrink-2026-09-02.txt)). The "k_proj" prior falls:
  v_proj yields +4.48 pp against +2.09 (*measured*,
  [M2](mesures/m2-attribution-4b-2026-09-02.txt)).
- A threshold is not lowered after seeing the clock, and a series is not cut
  short after seeing its points. Paid on 2026-08-15: the n_new = 1024 series is
  abandoned at 661 s against a threshold of 600 (*measured*, [KV
  q8](mesures/kvq8-4b-2026-08-15.txt)).
- An A/B moves one mechanism only; `check_fuse` refuses `FUSE=1` with
  `ROT_SHARE=0` for that reason. Paid on 2026-07-28: two variables changed
  between two runs of the 0.6B, 45 min (*measured*, `bin/smoke`) with no verdict.

## 2. The numbers

- Every number carries *measured*, *computed* or *estimated*, and its accounting
  (`rtbits`, `matvec`, `thesis` or inference). The published 4B file weighs
  2.1595 b/weight over 3,633,315,840 projection weights (*computed* on
  measured counts, `bin/seal`, HF card, [4B datasheet](fiche-4b.md)) and 2.1696
  with the tail excluded from the denominator. The 2.0702 printed by `smoke` is
  an ideal rate: it is not cited for this file. QTIP on the benchmark reads 2.0000
  (*measured*, [F2](mesures/f2-p3-qtip-banc-2026-08-21.txt)), a third
  accounting. Paid on 2026-09-02: a "+47% VRAM" figure was circulating as
  measured when it is computed ([A2
  deviations](../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md)).
- A range, never a point. Milliseconds drift between invocations of the same
  binary: 2.029× / 2.050× / 2.080× (*measured*,
  [K1](mesures/k1-metal-2026-08-05.txt)). Paid on 2026-08-18: the single points
  88.4 tok/s and ×1.12 become 87.0 [86.8, 87.0] and ×1.11 (*measured*,
  [B2](mesures/b2-fusedrun-plages-2026-08-18.txt)).
- A decode benchmark never writes the decoded weights, loads the activation once
  per threadgroup and amortizes submission below ~12%. The DRAM regime is forced
  cold, 4 copies in rotation: the 48 MB SLC of the M3 Max made the buffers
  cache-resident. Paid on 2026-07-31: the verdict "25 tok/s, it's dead" was 5×
  too pessimistic (*computed* before the three benchmark defects were fixed, no
  journal; those defects are described in [kernel format](format-noyau.md)).
- A ratio is formed round by round, in a single process, all arms interleaved in
  the same order at each round. When the arms do not coexist, we publish the
  quotient of the medians with an envelope. A cross-job reading is reported and
  not published: the ×1.091 of rotation hoisting (*computed*,
  [D1](mesures/d1-fusion-servie-2026-08-24.txt)).
- Throughput is stated in two formulations, always together: raw against our
  dense arm, ×2.00 [1.99, 2.00], and same-head, ×1.11 [1.11, 1.11]
  (*measured*, B2). Our dense arm copies 778 MB of vocabulary per token
  (*measured*, [phases](mesures/phases-2026-08-07.txt)).
- Every memory comparison is stated in b/param over the whole model, embedding
  included: 5.162 against 5.302 for AWQ at 4B (*computed* on measured bytes,
  [rtbits](mesures/rtbits-planes-8b-2026-08-09.txt)). Paid on 2026-08-06: "5.51
  against 4.50" mixed two denominators and two four-bit formats
  ([errata](archive/errata-rapport-lot-a-2026-08-06.md)). The b/weight of the
  linear layers is never the compression rate of the model. At 0.6B the tied
  embedding accounts for 26% and the real ratio is ×2.77 against ×7.4 nominal
  (*computed* from the weight counts of the smoke of 2026-07-28, no journal). At
  8B the untied heads make up 57% of the sealed file, ×3.7 (*computed* from the
  weight counts of the run of 2026-08-02; sealed sizes 4.32 GB f16 and 2.49 GB
  of tables in [HISTORIQUE](HISTORIQUE.md), 2026-08-08).
- Cross-card × factors are not divided. The ×1.78 between L40S and A100 is the
  clock ratio 2,520 / 1,410 MHz (*measured*, nvidia-smi at 1 Hz,
  [G](mesures/g-horloges-planes12x-2026-08-23.txt)), which comes to 1.787
  (*computed*, same journal). A "vs FP16" carries its card: `Planes14` yields
  2.14× on L40S (*measured*, G) and 0.79× on A100 (*measured*,
  [F4](mesures/f4-a100-2026-08-18.txt)).
- Within-stack ratios are not divided across stacks. ×2.413 for AWQ under vLLM
  (*measured*, [vLLM](mesures/awq-vllm-4b-2026-08-17.txt)) and ×1.11 for us
  under candle are cited side by side. vLLM preallocates its VRAM: no memory
  number comes out of it. M = 1 is not Marlin's optimal regime (minimum tile
  M = 8): the ×2.413 is not an upper bound on AWQ ([vLLM
  prereg](../proofs/preregistration-awq-vllm-2026-08-17.md)).
- Between formats at different bit rates, the comparable quantity is GB/s: QTIP
  reads 2.000 b/weight and `Planes14` 4.804 (*measured*, F2).

## 3. The noise

- Anything that recalibrates is read against the calibration σ at 4B. In
  perplexity: 5.2%, range 10.3% over three full runs at $21.45 (*measured*,
  [F5](mesures/f5-graines-4b-2026-08-19.txt)). On MMLU: 2.92 pp (*measured*,
  [MMLU noise](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)). A smaller effect
  requires a replication over two draws. Paid on 2026-08-19: the 0.7% σ
  inherited from 3 blocks of the 0.6B (*measured*, [batch
  B](archive/verdicts-lot-b-2026-08-06.md)) was wrong by a factor of 7
  (*computed*) at the published size. On 2026-08-25, the ranking of the gain
  bits reverses on the second draw (*measured*,
  [gain](mesures/gain-ab-gate-0.6b-2026-08-25.txt)).
- An A/B at constant file does not carry that σ. Its error bar is the paired
  interval: SE 0.43 pp on MMLU and ±0.12% of perplexity (*measured*, KV q8).
  Between different models, the paired SE runs from 0.79 to 1.44 pp (*measured*,
  [mmlupair](mesures/mmlupair-4b-8b-2026-08-13.txt)).
- Two within-size paired CIs that do not overlap test nothing across sizes:
  `mmlupair` does not pair two models. Whether a gap falls across sizes is
  tested with the SEs in quadrature: 4B→8B 6.96 pp, z = 3.82; 8B→14B 1.40 pp,
  z = 0.83 (*computed*, [mmlupair 14B](mesures/mmlupair-14b-2026-08-17.txt)).
- The SE of an MMLU difference is the paired one (McNemar), never the ± of one
  arm. The count is done in micro. Paid on 2026-08-30: a gate reported in macro
  at 72.85% falls back to 70.36% in micro (*measured*, [M3
  gate](mesures/m3-gate-mmlu-vllm-2026-08-30.txt)).
- A 3-block A/B does not validate a mechanism that touches magnitudes. The
  full-depth gate (28 blocks of the 0.6B) is mandatory and automated before
  paying for a card. A better local proxy has twice predicted a worse
  composition. Paid on 2026-08-07: design C yields ×1.99 at full depth and the
  gate blocks a 4 h 4B run (*measured*,
  [verdicts](archive/verdicts-nuit-2026-08-07.md)).
- The kill of a phase is measured within one job on the served path. Adding up
  gains from separate jobs manufactures a number (phase A prereg, 2026-08-31).

## 4. The tests

- A test is declared green after mutation: you break the code and the suite must
  fail. A mutant that survives says weak test or dead code. Paid on 2026-07-28:
  a neutralized accumulator made no test fail, and the suffix sweep of the
  parity repair was dead code.
- A test that exercises a parameter only at its neutral value tests nothing:
  Golay stage neutralized, non-strict monotonicity, ridge tested at λ = 0.
- A test that skips when its file is missing must fail. `#[ignore]` declares the
  absence in the fast loop; when invoked, the test names the missing file
  (`llvq-artifact/tests/common/mod.rs`). Paid on 2026-08-08: eight `SKIP` sites
  went green on any machine without the archive.
- The text of a ported kernel is run against an independent reference; it is not
  proofread. Paid on 2026-08-16: a 64-bit shift in the `peek` of E1v corrupted
  the Golay index, caught by `llvq-cuda/tests/host_e1v.cpp`.
- `unsafe` is allowed at hardware boundaries (mmap, kernel launch, reading a
  device buffer) and forbidden elsewhere. Five crates carry
  `#![forbid(unsafe_code)]`; `llvq-metal`, `llvq-cuda`, `llvq-llm` have 12,
  13 and 11 mentions (*measured*, grep, 2026-08-08). The attribute does not
  cover integration tests: every auditability sentence carries that caveat.
- A selector refuses any unknown value (`LLVQ_FUSED_LAYOUT`).
- An instrument proves it discriminates before returning 0. Paid on 2026-08-26:
  `grep` on a `.ots` returns 0 on an anchored file just as on a pending file
  (8-byte binary tag).

## 5. The retention channels

- Before pricing a replay, exhaust `hf buckets ls`, `hf jobs logs`, `hf jobs
  inspect` and `hf jobs ps -a`. Five "lost" outputs lived elsewhere, with a
  quote already placed against each (*measured*,
  [HISTORIQUE](HISTORIQUE.md)). Paid on 2026-08-17: 14B MMLU dumps in the
  bucket, 579 kB against a campaign (*measured*, [mmlupair
  14B](mesures/mmlupair-14b-2026-08-17.txt)). 4B NLLs in the job logs, $0
  against $0.25 quoted (*measured* against *estimated*, [4B
  raw](mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt)).
- The bucket holds only what a job wrote to it: the original sealed 8B was not
  there. It contains 69 files, 46.7 GB (*measured* on 2026-08-17, mmlupair 14B),
  never inventoried.
- Keep the raw output and commit it. A summary journal is a loss as soon as the
  channel expires. `bin/ppl` prints the per-window NLLs on stderr: without `2>`,
  they are lost.
- An `ots upgrade` follows the anchoring of each timestamp, on operator go.
  State as of 2026-09-02: 28 timestamps, 20 anchored (*measured*,
  [ots](mesures/ots-etat-2026-09-02.txt)). The anchors are read from the files;
  checking them against the chain requires a node and has never been done.

## 6. The machines

- No run started or stopped without an explicit go. The cost is announced
  before, the running total after. Each wave carries a cap in its prereg.
  Phase A: $4 for $1.11 spent. Research wave 1: $5 for $2.46 (*measured*,
  `docs/data/jobs.csv`, 104 jobs for $94.97 as of the evening of 2026-09-02).
- A quote is checked against the register after the job. Paid on 2026-08-03: the
  32B predicted at ~500 s/block (*estimated*) yields 621 (*measured*,
  [HISTORIQUE](HISTORIQUE.md)), a 25% underestimate (*computed*). The cost per
  weight is not linear (n³ of the factorization).
- `oracle` first, on every backend: 42 s (*measured*, [batch A
  report](archive/rapport-lot-a-2026-08-06.md)). `--features fast-linalg`
  everywhere we pay: without it, 40× slower (*measured*,
  `llvq-llm/src/bin/smoke.rs:1095`) for a bit-identical result.
- A local job caps `LLVQ_THREADS` at about `ncpu − 4` and starts under `nice`
  from launch. Paid on 2026-09-02: the M1 queue moved to `nice 10` at the fifth
  measurement, at 1,470% CPU (*measured*, [M1
  deviations](../proofs/preregistration-m1-hessienne-shrink-2026-09-02-ECARTS.md)),
  perplexities bit-exact.
- No `cargo build` while a queue is calling a binary by path. The M1 queue
  identifies its binary by sha256 in the journal.
- A `planesbench` with five arms or more is launched with `--timeout 90m`: host
  transcoding costs ~1,470 s (*measured*,
  [E2](mesures/e2-golay70-bench-2026-08-07.txt)). Paid on 2026-08-18: the 14B
  job killed by timeout at 42.5 min for 40 requested (*measured*, B2).
- An HF job is launched through the API with `['bash', '-lc', script]` and an
  identity assert against `hf jobs inspect`; the CLI parses `-lc` as `--label
  c`. Paid on 2026-08-31: A1 died four times before a number, three from
  infrastructure and one from the launcher, for $0.02 (*measured*, [wave 2
  deviations](../proofs/preregistration-vague2-gel-geometrie-2026-08-31-ECARTS.md)).
- Every edit under `cfg(linux)` is type-checked from the Mac
  (`CUDARC_CUDA_VERSION=12040 cargo check --target x86_64-unknown-linux-gnu`).
  Paid from 2026-08-15 to 08-16: a CUDA image that would not compile, undetected.

## 7. The words

- 2026-07-31: a free magnitude left off the bill costs 0.66 b/weight (*computed*:
  2.0653 announced, 2.7289 actual, [retraction and
  gain](archive/retraction-et-gain.md)).
- 2026-08-15: an operation count does not predict a time (×1.002 *estimated*,
  ×2.17 *measured*, P1b).
- 2026-08-16: `git log -S` does not read commit messages; a tool whose scope is
  not stated proves no absence.
- 2026-08-17: a sentence about a curve names its metric. The scaling knee is
  resolved in perplexity (t = −6.06, *measured*, [paired
  ppl](mesures/ppl-appariee-4b-2026-08-17.txt)) and silent on MMLU (p = 0.40,
  mmlupair 14B).
- 2026-08-21: `nullk` is the floor of our launch geometry; a kernel shaped
  otherwise goes below it (2.246 ms against 2.306, *measured*, F2).
- 2026-08-30: an expected favorable result demands more verification (degenerate
  GPTQ arm, 24.74%, *measured*, [M3
  gptq2](mesures/m3-gptq2-mmlu-2026-08-30.txt)).
- 2026-09-01: a hypothesis refuted in its sign is recorded with its number
  (split-K −1.87%, *measured*, [A3](mesures/a3-occupation-banc-2026-09-01.txt)).
