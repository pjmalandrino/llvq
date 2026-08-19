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
| Paper sources | `paper/`, builds with `make` |
| Measurement journals | `docs/mesures/` (63 files) |
| Figure/table data | `docs/data/*.csv` (10 files) |
| Cost ledger | `docs/data/jobs.csv` (55 GPU jobs) |
| Pre-registrations | `proofs/` (19 documents, 13 OpenTimestamps anchors) |

---

## 2. Badges: what is realistic

### Available — attainable

The repository and the model are public and permanently addressable. The one
gap for a strict reading of this badge: **neither GitHub nor Hugging Face is
an archival host**, and no DOI has been minted. Depositing the reviewed
revision on Zenodo is on the to-do list of `docs/plan-taco-2026-08-18.md`
(item F6) and is not done at the time of writing.

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
3. **Two of the three model sizes are not published.** The 8B and 14B sealed
   artifacts exceed what we host; they are reproducible at the costs recorded
   in the ledger (\$12.61 and \$27.67 of GPU time for the quantization).

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
  that fixed each acceptance criterion *before* its measurement; nine carry an
  OpenTimestamps anchor. `proofs/README.md` inventories them adversarially —
  including the two whose anchors were detached by later edits, and the ones
  stamped after their measurement rather than before. **The claim of
  anteriority holds document by document, not in general**, and that
  inventory is the place to check it.

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

The ledger totals **\$63.36 across 55 jobs**, and that is a **floor**: five
jobs from the first day of CUDA porting are journaled without an amount.

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
- **`LLVQ_DATASET_REV` cannot pin a commit.** One variable covers three
  dataset repositories, and a SHA is valid in only one of them; corpus
  pinning therefore works at the granularity of a branch name. This is a
  known weakness of the reproducibility tooling, not an oversight.
