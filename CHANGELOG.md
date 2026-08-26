# Changelog

All notable changes to this repository are recorded here. Versions are git
tags; nothing in this workspace is published to crates.io.

Every number below carries its provenance — *measured*, *computed* or
*estimated*, and in which accounting. That is the repository's own convention
(`CLAUDE.md` §7) and it is applied to these notes too. Ratios over time are
given as a median with its min–max range across the rounds that produced it,
never as a quotient of two minima from rounds that never coexisted.

---

## [0.0.1] — 2026-08-26

**The revision the paper describes.** This tag names the state of the code,
the measurement journals and the manuscript at the point where
`paper/main.tex` — *Unfolding the Leech Lattice: Fused Multi-Shell Decoding
and VRAM Layouts for 2-Bit LLM Weights* — was finished for ACM TACO. It is cut
so that "the reviewed revision" names one commit instead of a moving branch.
It is not a stability promise: the artifact format is versioned separately and
independently (see *Compatibility* below).

It is the repository's first **version** tag, not its first tag. Two tags
predate it and neither is a release: `preregistration-2026-08-10`, which
stamps a pre-registration, and `paper-v1` ("Papier compressé", 2026-08-12),
a manuscript snapshot that sits on a line that is **not an ancestor of
`main`**. Read them as what they are — timestamps on documents — and do not
read a version history into them.

### What is in it

An independent, from-scratch Rust implementation of Leech-lattice vector
quantization ([arXiv:2603.11021](https://arxiv.org/abs/2603.11021)), plus the
serving path the method needs and did not have. Eight crates:

| crate | what it is |
|---|---|
| `llvq-core` | Golay [24,12,8] and Λ₂₄ in integer coordinates |
| `llvq-search` | exact nearest-neighbour search over unions of shells, m ≤ 13 |
| `llvq-quant` | Spherical GPTQ (Algorithms 1 and 3) |
| `llvq-artifact` | the `.llvq` container: writer, reader, decoder |
| `llvq-metal` | Apple GPU micro-benchmarks |
| `llvq-cuda` | the fused kernel on NVIDIA, compiled by NVRTC at start-up |
| `llvq-llm` | model side: forward pass, calibration, perplexity, MMLU, serving |
| `llvq-bench` | rate–distortion and encoder throughput |

Five of them — `llvq-core`, `llvq-search`, `llvq-quant`, `llvq-artifact`,
`llvq-bench` — have **zero external dependencies** and carry
`#![forbid(unsafe_code)]` in their `lib.rs`, so reading a quantized model
never requires a tensor runtime. CI enforces the dependency-free property
mechanically rather than asserting it in prose. Two reservations the
repository insists on, and this note repeats rather than rounds off:
`llvq-quant` can pull in `faer` behind the optional `fast-linalg` feature,
which is off by default and is what the published models were factorized with;
and `forbid(unsafe_code)` in a `lib.rs` does **not** reach that crate's
integration tests, which are separate crates. Closing that hole needs a
workspace-level lint, which is a workspace decision and is not taken here.

The three contributions the paper is about:

1. **A fused dequantize-plus-matvec kernel for the full 301-class Λ₂₄(12)
   codebook** — to our knowledge the first; the reference kernel decodes one
   shell. Divergence-free, verified row by row against an f64 reference over
   the 1 105 920 output rows of the published model.
2. **The in-VRAM rate as a design axis distinct from the on-disk rate.** Four
   bit-exact layouts, timed in one process, one protocol, one byte accounting.
3. **Deployed 4-bit (AWQ) and 2-bit (QTIP) GEMV kernels in that same
   process**, so the comparison is not across papers.

### Headline measurements

All kernel rows below are one L40S, one process, one byte accounting; the
ratio is the median of the per-round ratio with its min–max range. Source:
[`docs/data/echelle-formats.csv`](docs/data/echelle-formats.csv). `b/weight
kernel` counts payload plus the f32 tail and f32 row scales.

| arm | b/weight kernel | GB read | GB/s | % of byte bound | vs FP16 |
|---|---|---|---|---|---|
| `nullk` (reads no weights — the floor) | 0.159 | 0.07 | 31 | 5 | 4.77× [4.76–4.77] |
| FP16 (control) | 16.000 | 7.27 | 661 | 100 | 1.00× |
| `Slot32` (one-hot, superseded) | 5.510 | 2.50 | 431 | 65 | 1.89× [1.89–1.89] |
| **`Planes14` (bit planes — served default)** | **4.804** | 2.18 | 428 | 65 | **2.15× [2.15–2.16]** |
| `Planes12x` (sparse overlay) | 4.342 | 1.97 | 359 | 54 | 2.00× [2.00–2.00] |
| `Golay70` v2 | 3.589 | 1.63 | 264 | 40 | 1.78× [1.77–1.78] |
| AWQ 4-bit kernel (competitor) | 4.179 | 1.90 | 584 | 88 | 3.38× [3.37–3.38] |
| QTIP 2-bit kernel (competitor) | 2.000 | 0.91 | 405 | 61 | 4.89× [4.89–4.90] |

Read that table three ways. Bit planes beat one-hot masks on **both** size and
speed at constant bandwidth. The curve turns between 4.8 and 4.3 b/weight,
where a second irregular stream enters, and collapses at 3.6, where decoding
stops being shifts and masks. And the trellis kernel reads 2.40× fewer bytes
and runs 2.27× faster than our served layout (*computed* from the columns
above) at near-equal fractions of their byte bounds — the time gap tracks the
traffic gap. That is the price of unfolding a codebook too large for a lookup
table, and we publish it rather than omit it.

**End to end, Qwen3-4B on an L40S**, 128 tokens, greedy tokens identical to
the dense arm; median of five timed generations after one discard. Source:
[`docs/data/campagne-finale.csv`](docs/data/campagne-finale.csv).

| arm | disk | VRAM | b/param, whole model | tok/s | ppl | MMLU micro |
|---|---|---|---|---|---|---|
| FP16 | 8.04 GB | 8.04 GB | 16.00 | 43.5 [43.4–43.6] | 12.2369 | 70.32 ± 1.28 |
| AWQ 4-bit (official Qwen) | 2.67 GB | 2.67 GB | 5.302 | — | 13.5207 | 70.04 ± 1.25 |
| **LLVQ, `Planes14` + int8 head** | **1.41 GB** | **2.60 GB** | **5.162** | **87.0 [86.8–87.0]** | 16.9358 | 55.70 ± 1.35 |

⚠️ **The 2.00× speed figure against our own dense arm does not measure the
kernel, and must never be published alone.** That denominator is handicapped:
our dense path calls `broadcast_matmul` and recopies the vocabulary every
token (reported upstream as
[huggingface/candle#3871](https://github.com/huggingface/candle/issues/3871)).
**With the output head held identical across arms — the only comparison that
isolates the kernel-and-format path — the gain is 1.11× at 4B** (48.3 against
43.5 tok/s), 1.29× at 8B and 1.41× at 14B. That series grows with model size.

**Quality across the three sizes served**, WikiText-2 at 4096 context and MMLU
micro-averaged, both arms printing the same token fingerprint. Source:
[`docs/data/echelle-4b-8b.csv`](docs/data/echelle-4b-8b.csv).

| model | ppl vs f16 | MMLU drop | b/param served | AWQ b/param | margin |
|---|---|---|---|---|---|
| Qwen3-4B | ×1.3845 | −14.73 pp | 5.162 | 5.302 | −2.6 % |
| Qwen3-8B | ×1.2201 | −10.56 pp | 5.322 | 5.956 | −10.6 % |
| Qwen3-14B | ×1.1894 | −6.85 pp | 5.106 | 5.404 | −5.5 % |

The deficit shrinks across the three sizes measured. **Three points are not a
scaling law**, the margin against AWQ is not monotone — its mechanism is the
embedding share, not the method — and this repository does not extrapolate to
70B.

The 4B row here reads 55.59 where the campaign table above reads 55.70: the
two score different files — this table the f16-embedding artifact scored at
all three sizes, the campaign table the pre-baked int8-embedding one. The
difference is inside the noise of the measurement, and both are stated rather
than one silently reused for the other.

### What this release does *not* claim

Stated here because a release note that lists only wins is the failure mode
this repository is built against.

- **The 2-bit competitor's kernel is faster than ours** — 2.27× in one
  process. No tuning pass was attempted on either side.
- **QTIP also outruns our no-weights control**, so that control measures *our*
  launch geometry, not a machine floor. Launch geometry holds 39 % of the gap
  to the DRAM floor.
- **On a second memory hierarchy the result does not survive**: on an A100,
  every lattice arm falls below FP16
  ([`docs/data/echelle-formats-a100.csv`](docs/data/echelle-formats-a100.csv)).
- **The MMLU deficit is unexplained.** Three suspects are bounded, none
  explains it; the leading untested one is calibration *composition* (the
  source paper uses DCLM-edu, we use C4), not volume.
- **The measured domain is decode-phase GEMV at batch 1**, on NVIDIA only,
  with VRAM counting weights only. No prefill GEMM on the served path; 32B is
  outside the envelope (rotation exceeds the shared-memory bound). The full
  validity envelope is a table in the paper's limitations section.
- **The published bytes are not byte-reproducible**, and the paper says so:
  the C4 calibration shard moved after the run.
- **The artifact is not GGUF, AWQ or safetensors.** `transformers`,
  `llama.cpp`, vLLM and TGI do not read it; the only reader that exists is
  `llvq-artifact`, here.

### Reproducing

`LAUNCH_ME.md` is the three-command user path.
[`ARTIFACT-EVALUATION.md`](ARTIFACT-EVALUATION.md) is the auditor path: what
each ACM badge would cost in time and money, and where every claim is
anchored. The CPU half — the mathematical core, the artifact format and the
CUDA decoders' host mirrors, which are the main correctness argument and need
no GPU by design — runs on a laptop in roughly half an hour.

The served configuration and both evaluations replay from the artifact alone
on an NVIDIA GPU:

```bash
LLVQ_FUSED_LAYOUT=planes14 LLVQ_EMBED=q8 cargo run --release \
  -p llvq-llm --features cuda --bin fusedrun -- <artifact>

LLVQ_DTYPE=f16 cargo run --release \
  -p llvq-llm --features cuda --bin ppl -- 4096 12 cuda <artifact>

cargo run --release -p llvq-llm --features cuda --bin mmlu -- <artifact> cuda 40
```

Each binary prints a fingerprint of the tokens it scored; **numbers compare
only when fingerprints match**. `LLVQ_FUSED_LAYOUT` rejects an unrecognised
value rather than falling back to the default in silence.

The published Qwen3-4B artifact is at
[Pier-Jean/Qwen3-4B-LLVQ-2bit](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit):
`qwen3-4b-llvq.bin`, 1 770 527 533 bytes, sha256
`9db213ef9fa9d7d7000789a8a529ce9459ce9ba6002ef5a72fd5a1c05c1c84b0`. The 8B and
14B artifacts are unpublished — requantizing would be a new calibration draw,
not the same object.

### Method and provenance

- **22 pre-registrations in `proofs/`, 16 with an OpenTimestamps anchor**
  posted before the run they govern. Criteria were written before the
  measurement in every case where a measurement decided something, including
  the ones that came back red and closed a line of work.
- **69 measurement journals** in `docs/mesures/`, **13 CSVs** in `docs/data/`
  behind every table and figure. `make` in `paper/` fails on a table that
  drifts from its CSV.
- **A cost ledger**: `docs/data/jobs.csv`, 73 rows, 68 with a billed amount,
  **$87.36** of rented GPU in total.
- **Retractions are traced, not overwritten.** `CLAUDE.md` is a lab notebook
  that keeps superseded figures next to what replaced them and why. Where it
  and the published documents disagree, `README.md` and `docs/fiche-4b.md`
  win — `CLAUDE.md` says so itself.
- Generative AI use is disclosed in the paper under ACM's authorship policy.

### Housekeeping in this release

- The eight crates now inherit `version` from `[workspace.package]`, set to
  `0.0.1`; they previously each carried the `cargo new` default of `0.1.0`,
  which matched no tag. One place to bump next time.
- `ARTIFACT-EVALUATION.md`'s inventory counts were refreshed against the tree
  (journals 63 → 69, CSVs 10 → 13, pre-registrations 19 → 22, anchors 13 → 16,
  jobs 55 → 73 with the $87.36 total). The paper quotes none of these numbers,
  so nothing in the manuscript moves.

### Compatibility

Licence MIT OR Apache-2.0. Toolchain pinned to Rust 1.95.0
(`rust-toolchain.toml`) because the bench publishes time ratios and the
compiler is a measurement parameter. CI covers 6 of the 8 crates — a hosted
runner has no GPU and no CUDA toolkit, so the two GPU crates cannot be
compiled there at all, which is a structural limit and is documented as one in
the workflow.

**The tag does not version the file format.** The `.llvq` container has its own
contract, pinned by `llvq_artifact::codebook_fingerprint()` at
`0x338f_420f_1186_6319` and enforced by a test that fails before any file is
written or read. A future `0.0.2` may change code freely; changing that
fingerprint breaks every artifact ever produced and is a separate decision.

[0.0.1]: https://github.com/pjmalandrino/llvq/releases/tag/v0.0.1
