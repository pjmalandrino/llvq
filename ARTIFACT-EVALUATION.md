# Artifact Evaluation

This document is written for an ACM artifact evaluator, not for a user. It
says what can be reproduced, what cannot, what each badge would cost in time
and money, and where every claim in the paper is anchored. Where a badge is
out of reach, it says so and why rather than leaving the evaluator to
discover it.

`LAUNCH_ME.md` is the user-facing path (three commands, a running model).
This file is the auditor-facing one.

---

## 1. What this artifact is

An independent Rust implementation of Leech-lattice vector quantization
(LLVQ) for LLM weights, plus the fused CUDA kernel and the VRAM layouts the
paper is about. Eight crates; the mathematical core (lattice, exact
nearest-neighbour search, bijective indexing, GPTQ, artifact format) has
**no external dependencies** and can be read end to end.

| | |
|---|---|
| Code | <https://github.com/pjmalandrino/llvq>, MIT OR Apache-2.0 |
| Published model | <https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit> (Apache-2.0) |
| Paper sources | `paper/`, builds with `make`; submitted to ACM TACO on 2026-08-24 (TACO-2026-428) at commit `e21a8bb` |
| Measurement journals | `docs/mesures/` (69 files) |
| Figure/table data | `docs/data/*.csv` (13 files) |
| Cost ledger | `docs/data/jobs.csv` (73 GPU jobs, \$87.36 billed) |
| Pre-registrations | `proofs/` (22 documents, 16 OpenTimestamps anchors — **none upgraded yet**, §5) |

🕳️ **This table read 63 / 10 / 55 / 19+13 until 2026-08-25, and every one of
those numbers was low.** They were correct on the day they were typed and were
never recounted while nine measurement campaigns landed between 2026-08-18 and
2026-08-24 (B2, B3, F1, F2, F3, F4, F5, G, D1 — \$28.56 across 27 jobs). The
counts above were re-derived by command on 2026-08-25 (`ls docs/mesures | wc -l`,
`ls docs/data/*.csv | wc -l`, `wc -l docs/data/jobs.csv` minus the header,
`ls proofs/*.md | wc -l` minus this file's README, `ls proofs/*.ots | wc -l`).
An audit surface that under-declares its own evidence is the same defect as one
that over-declares it: the reader cannot tell which number was checked.

---

## 2. Badges: what is realistic

### Available — attainable

The repository and the model are public and permanently addressable. The one
gap for a strict reading of this badge: **neither GitHub nor Hugging Face is
an archival host**, and no DOI has been minted. Depositing the reviewed
revision on Zenodo is on the to-do list of `docs/plan-taco-2026-08-18.md`
(item F6) and is not done at the time of writing.

⚠️ **The repository stays public for the duration of the review — operator
decision of 2026-08-25, and it is deliberate rather than an oversight.** The
manuscript itself is anonymised; the URLs above are carried on the title page,
which only the editor sees, so that the artifact is locatable without the
manuscript breaking its own anonymity. An evaluator reading this file has
therefore already left the anonymous path, and nothing here should be taken as
a claim that the code was withheld during review.

### Functional — attainable, ~30 minutes, no GPU

Everything in §3 below runs on a laptop with a Rust toolchain and no
accelerator. It exercises the mathematical core, the artifact format and the
CUDA decoders' host mirrors — the last of which is the paper's main
correctness argument and needs **no GPU by design**.

### Reproduced — partially attainable, and the limits are structural

Three separate obstacles, in decreasing order of how much they matter:

1. **The kernel numbers need an NVIDIA L40S** (and the second-architecture
   result an A100). They were produced on rented cards through Hugging Face
   Jobs, which requires a pre-paid credit balance. The exact commands and
   costs are in `docs/data/jobs.csv` and in each journal's header.
2. **The published bytes are not byte-reproducible, and we say so in the
   paper.** The C4 calibration shard moved from `00000` to `00001` after the
   run, and the container format has since gained a magic number. A re-run
   today produces a different, equally valid file. The *method* reproduces;
   the *bytes* do not.

   🚨 **And since 2026-08-19 we know that the quality number does not
   reproduce to a point either — it reproduces to a band about 5 % wide.**
   Three *full* quantizations of the 4B, identical to one character apart
   (`LLVQ_CALIB_SEED` ∈ {1,2,3}, same corpus read within the same hour, same
   codebook, same rotation, same evaluation token fingerprint
   `3f1baca9033bf251`), give a sealed-f16 perplexity of **16.7425 / 15.8836 /
   15.1027** — a range of 1.6398 ppl, **10.3 % of the median**, σ (n = 3) =
   0.8202 ppl = **5.2 %** (*measured*, \$21.45 of GPU,
   `docs/mesures/f5-graines-4b-2026-08-19.txt`). All three paired differences
   are resolved (t = +4.54 / +10.92 / +7.68): this is calibration-window
   choice, not measurement noise. An evaluator who re-quantizes and lands
   several percent away from a published degradation ratio has **not** failed
   to reproduce us. ✅ The control says the same run is otherwise identical:
   all three yield 2.0702 effective b/weight and a 1.771 GB sealed file, the
   published values to the digit — only quality moves, never the rate.

   ⚠️ **What this does *not* loosen, and the distinction is the whole point.**
   (a) A/B comparisons **at constant file** — KV int8, runtime layouts, the
   int8 embedding, every format verdict — do not recalibrate, and keep their
   own measured paired bar of ±0.12 %. (b) The three published artifacts
   (4B/8B/14B) all ran **without a seed**, hence on the same contiguous
   prefix: the scale curve compares identically-calibrated objects and does
   not carry this variance. (c) 🕳️ The repository's inherited working rule —
   "σ ≈ 0.7 %, anything under ~1.5 % is noise" — was measured on 3 blocks of
   Qwen3-0.6B and is **wrong by a factor of ~7 at the published size**. Two
   lot-B verdicts (the calibration oracle at −1.6 %, the volume curve at
   −1.2 % for ×13 the tokens) now fall *below* the noise floor; their
   conclusion is not overturned, but it rested on effects too small to
   separate, which is a weaker statement than the one that was published.
3. **Two of the three model sizes are not published.** The 8B and 14B sealed
   artifacts exceed what we host; they are reproducible at the costs recorded
   in the ledger (\$12.61 and \$27.67 of GPU time for the quantization).

   ✅ **The 8B, however, no longer needs re-quantizing — it needs re-sealing,
   and that costs \$0.24.** The projections-only file survived in the mounted
   bucket, so the sealed artifact was rebuilt from it on 2026-08-18 and
   checked against three criteria fixed beforehand: size 4.324 GB in the
   pre-registered [4.25 ; 4.40] band, `params_total` = 8,190,735,360 exact,
   and **5.322 b/param** (whole-model accounting, `Planes14` + int8 embedding)
   reproducing the 2026-08-09 journal **to the thousandth**
   (`docs/mesures/b3-8b-reseal-2026-08-18.txt`). ⚠️ What that does *not*
   establish is byte-equality with the lost original, which is unverifiable by
   construction and is not claimed: the re-sealed file is the campaigns'
   object *in the sense of the derived quantities*, and the downstream net is
   B2-8B, which checks its greedy tokens against its own dense arm.

What an evaluator *can* reproduce without a card: every figure and every
table of the paper, from the committed CSVs, with the build refusing to
proceed if a table has drifted from its data (§4).

---

## 3. Functional: the four checks, in order

No GPU, no network after the clone, no Python.

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
cargo test --release
```

Expect a few minutes and zero failures. Tests reported as `ignored` are the
sweeps of a sealed multi-hundred-megabyte artifact absent from the
repository; they are declared, not hidden (§5 of `CLAUDE.md`), and they fail
loudly by name rather than skipping silently if invoked without the file.

**(a) The mathematics.** `llvq-core` and `llvq-search` check the Λ₂₄
invariants against known values: the kissing number 196{,}560, the theta
series, and the cumulative shell count
$N(13) = 280{,}974{,}212{,}784{,}720$ from the source paper's Table 1 — a
15-digit lock no incorrect constraint passes. Exact nearest-neighbour search
is verified against brute force, and the 48-bit index against a bijectivity
sweep.

**(b) The CUDA decoders, without a CUDA device.** `llvq-cuda/tests/` compiles
the **text of the kernels** with `clang++ -Werror -ffp-contract=off` and runs
it against the independent Rust decoder, all classes covered with a coverage
assertion. This is the dispositif that caught a shift-by-64 undefined
behaviour no review had seen. It is the paper's correctness argument and it
runs on the evaluator's laptop.

**(c) The artifact format against hostile input.**
`llvq-artifact/tests/hostile_files.rs` feeds the reader files that lie about
their own lengths, tags and dimensions, and requires the named error rather
than a panic or an out-of-memory abort.

**(d) The codebook fingerprint.** `codebook_fingerprint.rs` pins the index
map to `0x338f_420f_1186_6319`, re-derives it independently of the module
under test, and perturbs thirteen ingredients one at a time. A file written
against a different codebook is refused before a single weight is decoded.

Optional, with the published model (~1.8 GB download):

```bash
hf download Pier-Jean/Qwen3-4B-LLVQ-2bit qwen3-4b-llvq.bin --local-dir .
shasum -a 256 qwen3-4b-llvq.bin
# expect 9db213ef9fa9d7d7000789a8a529ce9459ce9ba6002ef5a72fd5a1c05c1c84b0
LLVQ_SEALED_ARTIFACT=$PWD/qwen3-4b-llvq.bin cargo test --release -- --include-ignored
```

That runs the full-artifact sweeps: 150{,}681{,}600 blocks, bijection and
overlay proved block by block. Budget **tens of minutes**, not minutes.

---

## 4. The paper rebuilds from the data, and refuses to lie about it

```bash
cd paper && make          # figures from ../docs/data/*.csv, table check, then latexmk
```

`scripts/make_figures.py` regenerates every figure from the committed CSVs —
no number in a figure is typed by hand. `scripts/check_tables.py` then
compares six tables cell by cell against those CSVs, recomputes the derived
columns, pins the paired statistics to their journals, and **fails the build**
on any drift. Two tables are not yet covered by that check and are named at
the end of the script rather than left to look verified.

Requires a TeX distribution with `acmart`, and Python 3 with matplotlib.

---

## 5. Where each claim is anchored

Every number in the paper traces to a dated, costed job. The chain is:

```
paper table cell  →  docs/data/<x>.csv  →  docs/mesures/<journal>.txt  →  docs/data/jobs.csv
                     (checked by                (raw output of the        (job id, flavor,
                      check_tables.py)           job, kept verbatim)       billed minutes, $)
```

Two conventions an evaluator should know, because they are unusual:

- **Raw output is committed, not summarized.** Per-window log-likelihoods and
  per-question MMLU dumps are in the repository (`docs/data/mmlu-dumps/`,
  `docs/mesures/*BRUT*.txt`) rather than cited by job identifier, because an
  aggregated log cannot be given error bars afterwards, and because our
  compute vendor's log retention is neither documented nor guaranteed.
- **Decision thresholds are pre-registered.** `proofs/` holds the documents
  that fixed each acceptance criterion *before* its measurement: **22
  documents and 16 `.ots` anchor files**. ⚠️ Those two counts do not pair off
  — **15 documents carry an anchor**, and the sixteenth `.ots` anchors a
  *frozen earlier version* of one of them
  (`preregistration-f5-graines-4b-2026-08-19.v1-l4x4.md.ots`, kept beside the
  re-anchor rather than discarded). `proofs/README.md` inventories all of them
  adversarially — which were stamped before their measurement, which attest
  the bytes you can read today, the two whose anchors were detached by later
  edits, and the seven that carry no anchor at all. **The claim of anteriority
  holds document by document, not in general** (11 of the 22 hold the whole
  promise), and that inventory is the place to check it.

  🚨 **The anchors are all still pending, and that is a real hole in the
  "verifiable without trusting us" claim.** Checked on 2026-08-25 by
  `ots info` on each of the 16 files: every one carries **4
  `PendingAttestation` and 0 `BitcoinBlockHeaderAttestation`**. A pending
  attestation is a *promise* by four third-party calendar servers — the same
  four on all sixteen files: `alice.btc.calendar.opentimestamps.org`,
  `bob.btc.calendar.opentimestamps.org`, `btc.calendar.catallaxy.com`,
  `finney.calendar.eternitywall.com` — that they hold the commitment; it is
  not yet a Bitcoin block header. Until they are upgraded, the anteriority of
  every pre-registration here **depends on the survival and honesty of those
  four servers**, which is exactly the trust the exercise was meant to remove.
  The fix is one command — `ots upgrade proofs/*.ots`, then commit the
  rewritten files — and it has never been run. Saying so is cheaper than
  letting an evaluator assume a Bitcoin anchor is there.

---

## 6. Budget for a full reproduction

| what | hardware | wall clock | cost |
|---|---|---|---|
| §3 functional checks | any laptop | ~30 min | 0 |
| §3 full-artifact sweeps | laptop + 1.8 GB download | tens of min | 0 |
| §4 paper rebuild | laptop + TeX | ~2 min | 0 |
| Kernel/layout tables | 1× L40S | ~30 min | ~\$1 |
| Second-architecture point | 1× A100 | ~20 min | ~\$1 |
| End-to-end throughput, 3 sizes | 1× L40S | ~1.5 h | ~\$2.5 |
| Re-quantize the 4B | 1× RTX PRO 6000 | ~2.6 h | ~\$7 |
| Re-quantize the 8B / 14B | rented GPU | 4.6 h / 5 h | \$12.61 / \$27.67 |
| Calibration-seed band, 3× the 4B | 3× RTX PRO 6000 | ~7.8 h | \$21.45 |

The ledger totals **\$87.36 across 73 jobs**, and that is a **floor**: five
jobs from the first day of CUDA porting are journaled without an amount.
🕳️ *This line read "\$63.36 across 55 jobs" until 2026-08-25; the ledger had
gained 18 rows and \$24.00 since, and nobody re-summed the column.* The
re-derivation is one command, and an evaluator should run it rather than trust
the sentence:

```bash
awk -F, 'NR>1 {n++; if ($6 != "") s += $6} END {printf "%d jobs, $%.2f\n", n, s}' \
  docs/data/jobs.csv
```

⚠️ **The last row is the one to read before budgeting a reproduction.** It buys
no new capability — it is the same 4B, quantized three times — and it exists
only to put an error bar on the quality numbers (§2). An evaluator reproducing
a *single* re-quantization for ~\$7 gets one draw from that band, not the
published value.

---

## 7. Known limitations of this artifact

- **No CI on GPU.** The repository's CI covers 6 of 8 crates on a
  GPU-less runner; the two accelerator crates cannot even be *compiled*
  there. The header of `.github/workflows/ci.yml` states at length what a
  green badge does not mean.
- **One reader.** `llvq-artifact` is the only implementation of the format.
  It is dependency-free and meant to be read, but there is no second
  implementation to cross-check it.
- **NVIDIA-only serving.** The weights live in an incoherence-rotated basis
  and the rotation kernel exists on CUDA only. The Metal side is a bench,
  not a runner.
- **The speed results have a *measured* validity domain, and it is one
  architecture wide.** Every "vs FP16" ratio in the paper is L40S (Ada). On an
  A100-SXM4-80GB, **no decoding arm beats FP16** — `Planes14` 0.79×, `Slot32`
  0.73×, `Planes12x` 0.73×, `Golay70` v2 0.62×, v1 0.44×, against AWQ 1.82×
  and cuBLAS 1.14× (*measured*, `docs/mesures/f4-a100-2026-08-18.txt`). Lot G
  then settled the mechanism rather than leaving it as a hypothesis: both
  cards are pinned at their boost maximum with no throttling, and the SM clock
  ratio 2520/1410 = **1.787** matches the slowdown of the no-read witness
  (×1.772 and ×1.781 on two independent benches). An evaluator who reproduces
  §6's kernel table on a different card should expect different numbers, and
  that is a result of ours, not a failure of theirs.
- **Our no-read witness (`nullk`) is not a machine floor, and one sentence in
  a stamped pre-registration says otherwise.** Porting QTIP into our own bench
  — one process, 7 rounds with 2 discarded, ratios formed round by round —
  gives QTIP 2 bits at **2.246 ms [2.245–2.248]**, 0.91 GB read, 2.0000
  b/weight, against `Planes14` at 5.103 ms, 2.18 GB, 4.804 b/weight (*kernel*
  accounting on both sides): QTIP runs **2.27× [2.27–2.28]** faster than our
  served layout, and — the erratum — **faster than our own pass that reads no
  weight bytes at all** (2.246 < 2.306 ms). So `nullk` measures the floor of
  *our launch geometry* (one warp per output row, 252 launches), not of the
  hardware, and any "ceiling on what a format can buy" derived from it is a
  property of our kernel structure. The paper names the mechanism behind the
  gap rather than pleading implementation quality (`paper/sections/layouts.tex`):
  a codebook of 1.1e14 points does not fit in a lookup table where a 16-bit
  trellis state does, so the lattice index has to be *unfolded* into a
  bit-plane stream at 4.80 b/weight — the unfolding is imposed by the size of
  the codebook. Journal: `docs/mesures/f2-p3-qtip-banc-2026-08-21.txt`; the
  stamped document could not be edited, so the erratum is recorded in the
  journal instead — which is the rule, not an evasion (§5).
- **`LLVQ_DATASET_REV` cannot pin a commit.** One variable covers three
  dataset repositories, and a SHA is valid in only one of them; corpus
  pinning therefore works at the granularity of a branch name. This is a
  known weakness of the reproducibility tooling, not an oversight.
