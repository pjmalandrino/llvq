# Method

The lab rules: the rule, then the incident that taught it. Templates are in `docs/templates/`.

## 1. Before any measurement

- **A timestamped prereg comes before the first millisecond** (`ots stamp`). Paid on 2026-08-07: the 1.6× criterion of
  `Golay70` had no record older than 52 minutes before the measurement (*measured*, git).
- **A gate is written on a fundamental criterion**: disk, VRAM, throughput, quality, plus loadable model class,
  encoding cost and noise floor ([projection-gains](archive/projection-gains-2026-09-01.md) §0). A stage that cannot
  measure one carries a measurement, not a gate. Paid on 2026-09-04: the three F1 gates sat on proxies (Gaussian
  retention, table size, a competitor's milliseconds) and all three were unsound.
- **A kill is the operator's decision**, on a fundamental criterion. A floor or a projection informs it. Paid on
  2026-09-05: a synthetic floor was reported as "F1 is dead on decode cost"; the bench starved L1, and the format
  lived ([journal](mesures/f1-plancher-table-2026-09-05.txt),
  [deviations](../proofs/preregistration-f1-plancher-table-2026-09-04-ECARTS.md)).
- **The prereg gives adoption and kill criteria in numbers, on the exact quantity they name.** Paid on 2026-08-15: the
  walk passed at 0.3101 ns/block, the block read 0.6735 ([P1b](mesures/p1b-marche-bloc-2026-08-15.txt)).
- **The decision rule covers every outcome and ends with an "otherwise".** Paid on 2026-09-02: M2b read +3.60 pp
  [1.47, 5.79] and no line applied ([M2b](mesures/m2b-v4bits-2026-09-02.txt),
  [deviations](../proofs/preregistration-m2b-v4bits-2026-09-02-ECARTS.md)).
- **A timestamped prereg is never edited.** Deviations go beside it, in `<prereg>-ECARTS.md`. Paid on 2026-08-26: the
  anonymization pass `01fdbe6` rewrote the preregs of 08-10 and 08-11, and no blob in history matches their stamps
  ([ots](mesures/ots-etat-2026-08-26.txt)).
- **The prereg signs a prediction**, with its sign and order of magnitude. Paid on 2026-09-02: M1 predicted ρ* = 1 and
  read 0.7 ([M1](mesures/m1-hessienne-shrink-2026-09-02.txt)); the k_proj prior lost to v_proj, +4.48 pp against
  +2.09 ([M2](mesures/m2-attribution-4b-2026-09-02.txt)).
- **A threshold is not moved after seeing the clock, and a series is not cut after seeing its points.** Paid on
  2026-08-15: the n_new = 1024 series was abandoned at 661 s against 600 ([KV q8](mesures/kvq8-4b-2026-08-15.txt)).
- **An A/B moves one mechanism.** `check_fuse` refuses `FUSE=1` with `ROT_SHARE=0` for that reason. Paid on 2026-07-28:
  two variables changed between two 0.6B runs, 45 minutes, no verdict.

## 2. The numbers

- **Every number carries *measured*, *computed* or *estimated*, and its accounting** (`rtbits`, `matvec`, `thesis`
  or inference). One file can read 2.1595 or 2.1696 b/weight depending on whether the tail counts
  ([fiche-4b](fiche-4b.md)); QTIP's 2.0000 is a third accounting
  ([F2](mesures/f2-p3-qtip-banc-2026-08-21.txt)). Paid on 2026-09-02: a computed "+47 % VRAM" circulated as measured
  ([A2 deviations](../proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md)).
- **A range, never a point.** The same binary drifts: 2.029×, 2.050×, 2.080× ([K1](mesures/k1-metal-2026-08-05.txt)).
  Paid on 2026-08-18: 88.4 tok/s and ×1.12 became 87.0 [86.8, 87.0] and ×1.11
  ([B2](mesures/b2-fusedrun-plages-2026-08-18.txt)).
- **A decode bench never writes the decoded weights**, loads the activation once per threadgroup, amortizes
  submission, and forces the DRAM regime cold with buffers in rotation. Paid on 2026-07-31: "25 tok/s, it's dead" was
  5× too pessimistic ([format-noyau](format-noyau.md) §9).
- **A ratio is formed round by round, in one process**, the arms interleaved in the same order each round. Arms that
  cannot coexist get a quotient of medians with an envelope. A cross-job ratio is reported and not published
  ([D1](mesures/d1-fusion-servie-2026-08-24.txt)).
- **Throughput comes in two formulations, together**: raw against our dense arm, and same head. Our dense arm copies
  778 MB of vocabulary per token ([phases](mesures/phases-2026-08-07.txt)), so the raw ratio flatters us: ×2.00 raw,
  ×1.11 same head (B2).
- **Memory is compared in b/param over the whole model, embedding included.** Paid on 2026-08-06: "5.51 against 4.50"
  mixed two denominators and two 4-bit formats ([errata](archive/errata-rapport-lot-a-2026-08-06.md)). The b/weight of
  the linear layers is never the compression rate of the model.
- **No × is divided across cards or stacks.** The ×1.78 between L40S and A100 is the clock ratio
  ([G](mesures/g-horloges-planes12x-2026-08-23.txt)). `Planes14` reads 2.14× FP16 on the L40S and 0.79× on the A100
  ([F4](mesures/f4-a100-2026-08-18.txt)). AWQ's ×2.413 under vLLM and our ×1.11 under candle are cited side by side
  ([vLLM](mesures/awq-vllm-4b-2026-08-17.txt)). vLLM preallocates VRAM, so no memory number comes out of it.
- **Between formats at different bit rates, compare GB/s.**

## 3. The noise

- **Anything that recalibrates is read against the draw's σ at 4B**: 5.2 % of perplexity
  ([F5](mesures/f5-graines-4b-2026-08-19.txt)) and 2.92 pp of MMLU
  ([MMLU noise](mesures/bruit-mmlu-graines-4b-2026-08-25.txt)). A smaller effect needs two draws. Paid on 2026-08-19:
  the 0.7 % σ inherited from 3 blocks of the 0.6B was 7 times too small.
- **An A/B on one file uses the paired interval**: 0.43 pp of MMLU, ±0.12 % of perplexity (KV q8). Between models the
  paired SE runs from 0.79 to 1.44 pp ([mmlupair](mesures/mmlupair-4b-8b-2026-08-13.txt)).
- **A gap across sizes is tested with the SEs in quadrature**, since `mmlupair` does not pair two models
  ([mmlupair 14B](mesures/mmlupair-14b-2026-08-17.txt)).
- **The SE of an MMLU difference is McNemar's, counted in micro.** Paid on 2026-08-30: a gate read 72.85 % in macro and
  70.36 % in micro ([M3 gate](mesures/m3-gate-mmlu-vllm-2026-08-30.txt)).
- **A 3-block A/B does not validate a mechanism that touches magnitudes.** The full-depth gate (28 blocks of the 0.6B)
  runs before paying for a card. Paid on 2026-08-07: design C read ×1.99 at full depth and blocked a 4 h run
  ([verdicts](archive/verdicts-nuit-2026-08-07.md)).
- **The kill of a phase is measured in one job on the served path.** Adding gains from separate jobs manufactures a
  number.
- **An attention form that is not bit for bit is served under a tolerance**: paired WikiText-2 perplexity within
  ±0.12 % of the reference form (operator, 2026-11-01; on Metal, 2026-11-07).

## 4. The tests

- **A test is green after mutation**: break the code and the suite must fail. Paid on 2026-07-28: a neutralized
  accumulator broke no test, and a parity-repair sweep was dead code. A mutation that matches no site reads like a
  survivor, so count the sites.
- **A test that only exercises a parameter at its neutral value tests nothing.**
- **A test that skips for want of a file fails and names the file** (`llvq-artifact/tests/common/mod.rs`). Paid on
  2026-08-08: eight `SKIP` sites passed on any machine without the archive.
- **A ported kernel runs against an independent reference.** Paid on 2026-08-16: a 64-bit shift in E1v corrupted the
  Golay index, caught by `llvq-cuda/tests/host_e1v.cpp`.
- **`unsafe` lives only at hardware boundaries.** The core crates and `llvq-runtime` carry `#![forbid(unsafe_code)]`,
  which does not cover their integration tests.
- **A selector refuses any unknown value** (`LLVQ_FUSED_LAYOUT`, config fields).
- **An instrument proves it discriminates before it returns 0.** Paid on 2026-08-26: `grep` on a `.ots` returns 0 on
  an anchored file and on a pending one.
- **Count `test result:` lines; never cut a test log with `head`.**

## 5. Keeping the outputs

- **Before pricing a rerun, exhaust `hf buckets ls`, `hf jobs logs`, `hf jobs inspect` and `hf jobs ps -a`.** Five
  "lost" outputs lived elsewhere. Paid on 2026-08-17: the 14B MMLU dumps were in the bucket.
- **Commit the raw output.** A summary journal is a loss once the channel expires. `bin/ppl` prints per-window NLLs on
  stderr: without `2>`, they are gone.
- **`ots upgrade` follows each stamp**, on the operator's go. The anchors have never been checked against the chain.

## 6. The machines

- **No run starts or stops without an explicit go.** The cost is announced before and the running total after.
- **A quote is checked against the registry after the job.** Paid on 2026-08-03: the 32B predicted at ~500 s a block
  read 621, 25 % over. The cost per weight is not linear.
- **`oracle` first, on every backend. `--features fast-linalg` wherever we pay.**
- **A local job caps `LLVQ_THREADS` near `ncpu − 4` and starts under `nice`.**
- **No `cargo build` while a queue calls a binary by path.**
- **A `planesbench` with five arms or more gets `--timeout 90m`**: host transcoding takes about 1,470 s
  ([E2](mesures/e2-golay70-bench-2026-08-07.txt)).
- **An HF job is launched through the API with `['bash', '-lc', script]`** and checked against `hf jobs inspect`; the
  CLI parses `-lc` as `--label c`.
- **Every edit under `cfg(linux)` is type-checked from the Mac**:
  `CUDARC_CUDA_VERSION=12040 cargo check --target x86_64-unknown-linux-gnu`, and for the cuda half of `llvq-llm`,
  `ops/check-linux-cuda.sh check -p llvq-llm --features cuda --all-targets`.
- **Every binary of the CUDA image is in both `cargo build --bin` and the runtime `COPY` of `ops/Dockerfile.cuda`**,
  and every `COPY --from=build` path is in `UPLOAD_ALLOW` of `ops/run.py`.

## 7. The words

- A free magnitude left off the bill costs 0.66 b/weight: 2.0653 announced, 2.7289 actual
  ([retraction](archive/retraction-et-gain.md)).
- An operation count does not predict a time (×1.002 estimated, ×2.17 measured, P1b).
- A tool whose scope is not stated proves no absence: `git log -S` does not read commit messages.
- A sentence about a curve names its metric: the scaling knee shows in perplexity and not in MMLU
  ([paired ppl](mesures/ppl-appariee-4b-2026-08-17.txt)).
- `nullk` is the floor of our launch geometry, not of the card: QTIP goes below it (F2).
- An expected favourable result needs more checking: a degenerate GPTQ arm read 24.74 %
  ([M3 gptq2](mesures/m3-gptq2-mmlu-2026-08-30.txt)).
- A hypothesis refuted in its sign is recorded with its number ([A3](mesures/a3-occupation-banc-2026-09-01.txt)).
- A check true by construction is not a measurement.
