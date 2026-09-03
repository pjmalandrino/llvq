# LLVQ, repository map

Loaded at the start of every session: where to resume and what we do not do. The result numbers are in
`docs/ETAT.md`, the history in `docs/HISTORIQUE.md`.

## Goal

Cut the inference cost of LLMs for sovereignty: fit bigger models on local hardware. The lever is the number of
bits per weight; at 2 bits a 70B goes from 140 GB to 18 GB on disk (*computed*). In VRAM the served format unfolds
the index to 4.804 b/weight (*measured*, `docs/mesures/e2-golay70-bench-2026-08-07.txt`). Under the product triplet
in force, the largest admissible class is 43.3 billion parameters at 5.162 b/param; the 32B is the served object,
the 70B does not fit (*computed*, `docs/ETAT.md` §6). We implement the LLVQ paper in Rust, vector quantization on
the Leech lattice Λ₂₄ ([arXiv:2603.11021](https://arxiv.org/abs/2603.11021)). The engineering contribution is the
multi-shell fused kernel: dequantization and matvec in a single CUDA kernel.

## Where to resume

Read in this order; each document stands on its own at its level.

1. `docs/ETAT.md`: served config, headline numbers, open decisions.
2. `docs/ROADMAP.md`: what comes next, with its gates and its costs.
3. `docs/HISTORIQUE.md`: the chronological thread, one entry per period.
4. `docs/METHODE.md`: the lab rules, preregs, labels, retention.
5. `docs/STYLE.md`: the writing rules for every living document.

The paper is transcribed in full in `docs/llvq-paper-notes.md`; do not reopen the PDF, and never run `pdftotext`
on it (corrupted extraction). The journals are in `docs/mesures/`, the preregs in `proofs/`, the job registry in
`docs/data/jobs.csv`. `docs/archive/` is not edited. Data: `docs/data/mmlu-appariee.csv` (nine pairs, 3 sizes ×
3 arms), `docs/data/mmlu-dumps/`, `docs/data/ppl-genou.csv`, `docs/data/echelle-formats.csv`,
`docs/data/awq-speed-4b-2026-08-17.json`; `docs/data/README.md` gives the provenance of the amounts.
`docs/fiche-4b.md` describes the published object, `docs/format-noyau.md` the kernel and its measurement traps,
`docs/echelle-4b-8b-2026-08-08.md` the scaling.

## Architecture

Eight crates, members of `Cargo.toml` (*measured*, `Cargo.toml:3-12`).

| crate | role | external dependencies | `unsafe` |
|---|---|---|---|
| `llvq-core` | Golay [24,12,8], Λ₂₄, shells | none | forbid |
| `llvq-search` | exact NN search, classes m ≤ 13, indexing, packing, `rankdec` | none | forbid |
| `llvq-quant` | Spherical GPTQ, dense algebra, block loop | `faer` 0.24, optional, feature `fast-linalg` | forbid |
| `llvq-artifact` | `.llvq` format: writer, reader, decoder | none | forbid |
| `llvq-bench` | rate-distortion, encoder throughput, decode cost | none | forbid |
| `llvq-metal` | macOS GPU micro-benchmarks, MSL shaders, `rankbench` | `metal` | allowed |
| `llvq-cuda` | NVIDIA fused kernel compiled by NVRTC, benchmarks | `cudarc`, `cfg(target_os = "linux")` | allowed |
| `llvq-llm` | forward pass, corpora, perplexity, MMLU, fused path in the model | `candle`, `tokenizers`, `hf-hub`, `parquet` | allowed |

The full tree of `llvq-artifact` is 3 crates; that of `llvq-llm` is 261 packages, 291 with `metal,fast-linalg`
(*measured*, `docs/archive/audit-publication-2026-08-03.md`). `unsafe` is allowed only at hardware boundaries:
mmap, kernel launch, reading a device buffer. Caveat: `#![forbid(unsafe_code)]` in a `lib.rs` does not cover
integration tests, which are separate crates; closing that hole needs `[workspace.lints]`, and that operator
decision is pending. Without `--features fast-linalg`, the factorization is ~40× slower for a bit-identical result
(*measured*, `llvq-llm/src/bin/smoke.rs:1253-1265`). The in-house algebra path (`llvq-quant/src/linalg.rs`,
246 lines) stays the verification reference: `both_factorizations_agree` (`llvq-quant/tests/g5_gptq.rs:825`)
requires the same factor as `faer`. Do not delete it. The encoder (nearest neighbor) runs offline once per model;
the decoder (index → vector) runs at every GEMM, in shifts and masks. Never optimize one while thinking about the
other. The derivations are locked by `classes_reproduce_theta_series` and `even_repair_matches_dp_reference`:
Λ₂₄ cosets, single parity flip, telescoping sums, global objective of `nearest_scaled` faster than `shell_bests`.
Their reason is in `docs/HISTORIQUE.md`, entry "Foundations, G1 to G4". `bin/fusedrun` is the kernel in the model;
`bin/run` is the dense demo, not ported.

## Commands

Two test loops, two orders of magnitude. `cargo test` in debug skips the heavy tests
(`cfg_attr(debug_assertions, ignore)`) and runs in minutes. `cargo test --release -- --include-ignored` takes tens
of minutes (*measured* on 2026-08-08: 17 min without finishing `llvq-artifact`, no journal). The sealed-archive
tests carry an unconditional `#[ignore]`, require `~/llvq-q4b.llvq`, and fail with an error naming that file if it
is missing.

```bash
cargo test                                            # fast loop
cargo test --release -- --include-ignored             # full suite, before any format commit
cargo clippy --all-targets                            # zero warnings
cargo run --release -p llvq-bench --bin llvq-bench    # also: encbench, betasweep, decbench, classhist
cargo run --release -p llvq-metal --bin thesis        # macOS; also: matvec, decreal, mslcheck, p1v0, rankbench
cargo run --release -p llvq-cuda --bin planesbench -- <model.llvq>   # Linux + CUDA; also: preflight
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_ARTIFACT=q4b.llvq cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- 64 2048 12 4096 metal nogs leech1c12 999 rot   # requantize the 4B (4.01 h = 14,447 s on M3 Max, measured, docs/fiche-4b.md §3.4)
#   positional: n_calib · calib_len · n_eval · eval_ctx · device · gs/nogs · codebook (suffix f = free magnitude, L<n> = cap) · limit · rot
LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --bin seal -- q4b.llvq qwen3-4b-llvq.bin   # sealed file expected 1.771 GB (measured, docs/fiche-4b.md)
cargo run --release -p llvq-llm --features metal --bin mmlu -- <checkpoint|sealed> metal 40
cargo run --release -p llvq-cuda --bin nullkbench                     # the floor; every bin of the image goes in `cargo build --bin` AND the COPY of ops/Dockerfile.cuda
uv run ops/awq_speed.py … | uv run ops/awq_dequant.py check           # AWQ: unpinned revision refused; locks L1/L2/L4
cargo run --release -p llvq-llm --features metal --bin oracle              # forward pass vs candle, on every backend
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal <sealed>
cargo run --release -p llvq-llm --features cuda --bin fusedrun             # the kernel in the model
uv run ops/run.py estimate|selftest|publish|oracle|launch|monitor         # HF Jobs
uv run --with opentimestamps ops/otsaudit.py                              # state of the .ots anchors
```

Other `llvq-llm` binaries: `mmlu`, `mmlupair`, `embedq`, `seal`. Those of `llvq-bench` also include `rtbits`
(b/param accounting) and `radixstudy` (E3). `rankbench` refuses to start without
`proofs/preregistration-p1-2026-08-13.md.ots`.

### Environment variables

| variable | values | effect |
|---|---|---|
| `LLVQ_FUSED_LAYOUT` | `planes14` (default), `planes12x`, `slot32`, `golay70` | VRAM layout of the fused kernel; any other value is refused |
| `LLVQ_EMBED` | `f16` (default), `q8` | embedding quantized at load; `q8` is the served config |
| `LLVQ_KV` | `f16` (default), `q8` | int8 KV cache, shipped, not the default (short context only) |
| `LLVQ_ROT_SHARE` | `0`, `1` | one rotation per group of projections; served = `1` |
| `LLVQ_FUSE` | `0`, `1` | q+k+v and gate+up fusion; served = `1`; `FUSE=1` with `ROT_SHARE=0` refused |
| `LLVQ_FUSE_AB` | `1` | `fusedrun`: both arms of the fusion in a single process, the shape of D1 |
| `LLVQ_TIME_PHASES` | `1` | `fusedrun`: per-phase profile, outside the published protocol |
| `LLVQ_DTYPE` | `f32` (`ppl` default), `f16` | evaluation dtype; comparing ppl and MMLU requires the same on both sides |
| `LLVQ_CALIB` | `wikitext2` (default), `c4`, `wikitext2-test` | `smoke`: calibration corpus; `c4` is the paper's protocol |
| `LLVQ_ARTIFACT` | path | `smoke`: writes the compressed artifact, packed indices; absent, nothing is written |
| `LLVQ_RESUME` | path of a shard | `smoke`: resume from that shard; requires `LLVQ_ARTIFACT` |
| `LLVQ_SEALED_ARTIFACT` | path | `llvq-artifact` archive tests: moves the search for the sealed file (`tests/common/mod.rs`) |
| `LLVQ_CALIB_SEED` | integer | calibration windows drawn at random instead of the prefix (`smoke`) |
| `LLVQ_DAMPING` | float | relative damping of the Hessian (`smoke`) |
| `LLVQ_H_SHRINK` | ρ in [0, 1], default `1` | `H ← ρ·H + (1−ρ)·diag(H)` before rotation (`smoke`, M1 knob) |
| `LLVQ_RESTORE_F16` | projection types separated by commas, or `all` | `mmlu`, `ppl`: those types taken from the checkpoint in f16, the rest as served |
| `LLVQ_RESTORE_Q4` | same list | same in int4 g128; setting both is refused |
| `LLVQ_MODEL` | HF repo or local directory | checkpoint; required by `RESTORE_*`, never a default in `mmlu` |
| `LLVQ_THREADS` | integer | cap of the encoding pool (`smoke`); ncpu−4 and `nice` on a shared machine |
| `LLVQ_NVRTC_ARCH` | `compute_NN`, default `compute_89` | NVRTC target; `compute_80` for A100; any other form refused |
| `LLVQ_TIME_EVENTS` | `1` | device span by CUDA events (`planesbench`), outside the published protocol |
| `LLVQ_BENCH_ARMS` | phases separated by `;` | arms of `planesbench`; unknown name refused |
| `LLVQ_QTIP_DIR` | directory | upstream QTIP kernel, GPL v3, not redistributed (`docs/qtip-provenance.md`) |

`LLVQ_KV_PREALLOC`, `LLVQ_GRAPH_AB` and `LLVQ_SEG_ARMS` are measurement modes, never a served config.

## Hard rules

The detail and the reasons are in `docs/METHODE.md`.

1. Do not start or stop a run, and make no structural decision, without an explicit go; announce the cost before and the running total after.
2. Timestamp the prereg before the first measurement; never edit a timestamped prereg, write the deviation beside it.
3. Do not implement A2 (CUDA Graphs) in the core; do not reopen A2, E1v or Golay70 outside the conditions written in `docs/ETAT.md` §7 (A2: `proofs/preregistration-a2-a3-geometrie-2026-08-31-ECARTS.md` §É7).
4. Never publish the raw ratio alone; always give the same-head ratio.
5. Never divide a × across cards (L40S, A100) or across stacks (vLLM, us); compare AWQ and QTIP in GB/s.
6. State every memory comparison in b/param for the whole model, embedding included.
7. Publish medians with ranges formed round by round, never a quotient of two minima.
8. Label every number *measured*, *computed* or *estimated*, with its accounting and its journal.
9. Before costing a rerun, exhaust `hf buckets ls`, `hf jobs logs`, `hf jobs inspect`; keep the raw output.
10. A test that skips for want of an archive fails and names that archive; mutate the code before declaring a gate green; `oracle` first on every backend.

## Conventions

- Code comments and documentation in English; conversation in French.
- Zero warnings from `cargo clippy --all-targets`.
- `docs/STYLE.md` for every living document; a fact that changes is replaced, the old one goes to `HISTORIQUE.md` with its date.
- The five core crates stay free of external dependencies and in `forbid(unsafe_code)`.
- Two tiers of `ignore`: `cfg_attr(debug_assertions, ignore)` for compute, unconditional `#[ignore]` for sealed archives.
- Full suite before any commit that touches a format or the indexing; any change to the index map breaks format v1 (`codebook_fingerprint` pins it).
- When documents disagree, the object wins: the sealed file, the journal. `docs/fiche-4b.md` is authoritative on the published file, `paper/` on what is submitted; README and the datasheet take precedence over this map.
- Every paid run goes through `ops/run.py` with `--features fast-linalg`, and the job goes into `docs/data/jobs.csv`.
