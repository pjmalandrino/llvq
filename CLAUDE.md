# LLVQ, repository map

Loaded at the start of every session: where to resume, and what we do not do. The numbers are in `docs/ETAT.md`, the
history in `docs/HISTORIQUE.md`.

## Goal

Cut the inference cost of LLMs for sovereignty: fit bigger models on local hardware. The lever is the number of bits
per weight. We implement the LLVQ paper in Rust, vector quantization on the Leech lattice Λ₂₄
([arXiv:2603.11021](https://arxiv.org/abs/2603.11021)). The engineering contribution is the multi-shell fused kernel:
dequantization and matvec in a single CUDA kernel.

Three Qwen3 files are sealed and served at about 2.7 b/param, 4B, 8B and 14B, and paper 2 is written on them
(`paper2/`). The `Tetra` format reads 2.148 kernel b/weight against 4.804 for `Planes14`, which is what put it under
the product triplet's b_max of 3.00. **No model above 14B is served**: the `rot_apply` wall of `docs/format-noyau.md`
§8 closes that path whatever the format, and the 32B has never been encoded.

## Where to resume

Read in this order; each document stands on its own at its level.

1. `docs/ETAT.md`: served configuration, headline numbers, open decisions.
2. `docs/ROADMAP.md`: what comes next, with gates and costs. `docs/ROADMAP-QUALITY.md` is the quality axis.
3. `docs/HISTORIQUE.md`: the chronological thread, one entry per period.
4. `docs/METHODE.md`: the lab rules. `docs/STYLE.md`: the writing rules for every living document.

The source paper is transcribed in full in `docs/llvq-paper-notes.md`; do not reopen the PDF, and never run `pdftotext`
on it (corrupted extraction). Journals in `docs/mesures/`, preregs in `proofs/`, job registry in `docs/data/jobs.csv`,
served configs in `configs/`. `docs/archive/` is not edited. `docs/fiche-4b.md` is authoritative on the published
`Planes14` file, `paper2/` on what paper 2 claims, `docs/format-noyau.md` on the kernel and its measurement traps.

## Architecture

Eight crates, members of `Cargo.toml`.

| crate | role | external dependencies | `unsafe` |
|---|---|---|---|
| `llvq-core` | Golay [24,12,8], Λ₂₄, shells | none | forbid |
| `llvq-search` | exact NN search, classes m ≤ 13, indexing, packing, `rankdec` | none | forbid |
| `llvq-quant` | GPTQ, dense algebra, block loop | `faer` 0.24, optional, feature `fast-linalg` | forbid |
| `llvq-artifact` | `.llvq` format: writer, reader, decoder | none | forbid |
| `llvq-bench` | rate-distortion, encoder throughput, decode cost | none | forbid |
| `llvq-metal` | macOS GPU micro-benchmarks, MSL shaders, `rankbench` | `metal` | allowed |
| `llvq-cuda` | NVIDIA fused kernel compiled by NVRTC, benchmarks | `cudarc`, `cfg(target_os = "linux")` | allowed |
| `llvq-llm` | forward pass, corpora, perplexity, MMLU, fused path in the model | `candle`, `tokenizers`, `hf-hub`, `parquet` | allowed |

`unsafe` is allowed only at hardware boundaries: mmap, kernel launch, reading a device buffer. Caveat:
`#![forbid(unsafe_code)]` in a `lib.rs` does not cover integration tests, which are separate crates; closing that hole
needs `[workspace.lints]`, and that operator decision is pending. Without `--features fast-linalg` the factorization is
about 40 times slower for a bit-identical result. The in-house algebra path (`llvq-quant/src/linalg.rs`) stays the
verification reference: `both_factorizations_agree` requires the same factor as `faer`. Do not delete it.

The encoder (nearest neighbour) runs offline once per model; the decoder (index to vector) runs at every GEMM, in
shifts and masks. Never optimize one while thinking about the other. The derivations are locked by
`classes_reproduce_theta_series` and `even_repair_matches_dp_reference`. `bin/fusedrun` is the kernel in the model,
`bin/run` the dense demo.

## Commands

Two test loops, two orders of magnitude. `cargo test` in debug skips the heavy tests
(`cfg_attr(debug_assertions, ignore)`) and runs in minutes. `cargo test --release -- --include-ignored` takes tens of
minutes. The sealed-archive tests carry an unconditional `#[ignore]`, need `~/llvq-q4b.llvq`, and fail naming that file
if it is missing.

```bash
cargo test                                            # fast loop
cargo test --release -- --include-ignored             # full suite, before any format commit
cargo clippy --all-targets                            # zero warnings
cargo run --release -p llvq-bench --bin llvq-bench    # also: encbench, betasweep, decbench, classhist, rtbits, radixstudy, gaindisagree
cargo run --release -p llvq-metal --bin thesis        # macOS; also: matvec, decreal, mslcheck, p1v0, rankbench
cargo run --release -p llvq-cuda --bin planesbench -- <model.llvq>   # Linux + CUDA; also: preflight, nullkbench
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=dclm-edu LLVQ_ARTIFACT=q4b.llvq cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- 64 2048 12 4096 metal nogs leech1c12 999 rot
#   positional: n_calib · calib_len · n_eval · eval_ctx · device · nogs/gs/dc/sph · codebook (suffix f = free magnitude, L<n> = cap) · limit · rot
LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --bin seal -- q4b.llvq qwen3-4b-llvq.bin   # also: export, rowscale, embedq, int4swap
cargo run --release -p llvq-llm --features metal --bin mmlu -- <checkpoint|sealed> metal 40         # the dense reconstruction
LLVQ_CONFIG=configs/qwen3-4b-tetra-e4.json cargo run --release -p llvq-llm --features cuda --bin mmlu -- <sealed> cuda 40   # THROUGH the served kernel
cargo run --release -p llvq-llm --features cuda --bin fusedrun                                      # the kernel in the model
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal <sealed>
cargo run --release -p llvq-llm --features metal --bin oracle                                       # forward pass vs candle, on every backend
CUDARC_CUDA_VERSION=12040 cargo clippy --target x86_64-unknown-linux-gnu -p llvq-cuda --all-targets  # type-checks the Linux half on a Mac
uv run ops/run.py estimate|selftest|publish|oracle|launch|monitor                                   # HF Jobs
uv run --with opentimestamps ops/otsaudit.py                                                        # state of the .ots anchors
uv run ops/awq_speed.py … | uv run ops/awq_dequant.py check                                         # AWQ, unpinned revision refused
cd paper2 && make && RELEASE=1 make check                                                           # the paper, and its tables against their CSVs
```

Two traps. Every binary of the CUDA image goes in `cargo build --bin` **and** in the runtime `COPY` of
`ops/Dockerfile.cuda`; a bin in one list only does not exist on the target. Every `COPY --from=build /src/x` must also
be in `UPLOAD_ALLOW` of `ops/run.py`, or the Space dies after 12 minutes. `rankbench` refuses to start without
`proofs/preregistration-p1-2026-08-13.md.ots`.

### Environment variables

| variable | values | effect |
|---|---|---|
| `LLVQ_CONFIG` | path to a served config (`configs/*.json`) | **the served object, as a file.** Puts `fusedrun` on a one-arm path with no dense reference and `mmlu` on the **kernel** instead of a dense reconstruction. No built-in served default: unset, both binaries are the benches they have always been. A variable that contradicts the file is refused, not outvoted |
| `LLVQ_FUSED_LAYOUT` | `planes14` (default), `planes12x`, `slot32`, `golay70`, `tetra48` | VRAM layout of the fused kernel; any other value refused |
| `LLVQ_EMBED` | `f16` (default), `q8`, `q4` | embedding quantized at load, or read as the file stores it; `q4` in the three served configs, whose files carry it int4 g64 |
| `LLVQ_KV` | `f16` (default), `q8` | int8 KV cache, shipped, not the default (short context only) |
| `LLVQ_ROT_SHARE` | `0`, `1` | one rotation per group of projections; served = `1` |
| `LLVQ_FUSE` | `0`, `1` | q+k+v and gate+up fusion; served = `0` under `Tetra48`, which carries no segmented kernel; `FUSE=1` with `ROT_SHARE=0` refused |
| `LLVQ_TILE_BLOCKS` | unset (default), `auto`, power of two in 32..=512 | blocks of the activation one CTA stages in shared memory. Unset reads the measured row for the card, 64 on sm_89 and 32 on sm_120, and falls back to 128 where no row exists. Zero bits, bit-identical output, worth +16.1% to Tetra. Every figure published before 2026-09-20 was measured at 128 |
| `LLVQ_NVRTC_ARCH` | `compute_NN`, default `compute_89` | NVRTC target; `compute_80` for A100; any other form refused |
| `LLVQ_DTYPE` | `f32` (`ppl` default), `f16` | evaluation dtype; comparing ppl or MMLU requires the same on both sides |
| `LLVQ_CALIB` | `wikitext2` (default), `c4`, `dclm-edu`, `wikitext2-test` | `smoke`: calibration corpus; `dclm-edu` is the paper's own set and the one the sealed files use |
| `LLVQ_ARTIFACT` | path | `smoke`: writes the compressed artifact; absent, nothing is written |
| `LLVQ_RESUME` | path of a shard | `smoke`: resume from that shard; requires `LLVQ_ARTIFACT` |
| `LLVQ_INT4_TYPES` | projection types, comma-separated | `smoke`: those types written as int4 g128 records instead of lattice codes; empty by default |
| `LLVQ_RESTORE_F16` / `LLVQ_RESTORE_Q4` | projection types, or `all` | `mmlu`, `ppl`: those types taken from the checkpoint in f16 or int4 g128, the rest as served; setting both refused |
| `LLVQ_MODEL` | HF repo or local directory | checkpoint; required by `RESTORE_*`, never a default in `mmlu` |
| `LLVQ_MMLU_ALLOC` | `flat` (default), `proportional`, `proportional=<total>` | `mmlu`: how the budget spreads over the 57 subjects. `mmlupair` refuses two dumps of different plans, so re-barring a published arm costs a full run |
| `LLVQ_CALIB_SEED`, `LLVQ_DAMPING`, `LLVQ_H_SHRINK`, `LLVQ_GAIN_SCALE`, `LLVQ_SEQ_BLOCK`, `LLVQ_THREADS` | see `docs/METHODE.md` | `smoke` knobs. Their default is the published path and is bit-identical; any other value of `LLVQ_SEQ_BLOCK` is refused by name. `LLVQ_THREADS` ≈ ncpu−4 on a shared machine |
| `LLVQ_SEALED_ARTIFACT` | path | `llvq-artifact` archive tests: moves the search for the sealed file |
| `LLVQ_QTIP_DIR` | directory | upstream QTIP kernel, GPL v3, not redistributed (`docs/qtip-provenance.md`) |

`LLVQ_KV_PREALLOC`, `LLVQ_GRAPH_AB`, `LLVQ_SEG_ARMS`, `LLVQ_TIME_PHASES`, `LLVQ_TIME_EVENTS`, `LLVQ_BENCH_ARMS` and
`LLVQ_PREFILL_TOKENS` are measurement modes, never a served config. Every one is refused by name beside `LLVQ_CONFIG`,
except `LLVQ_PREFILL_TOKENS`, which is the served path's own gate.

## Hard rules

The detail and the reasons are in `docs/METHODE.md`.

1. Do not start or stop a run, and make no structural decision, without an explicit go; announce the cost before and the running total after.
2. Timestamp the prereg before the first measurement; never edit a timestamped prereg, write the deviation beside it.
3. Do not implement A2 (CUDA Graphs) in the core; do not reopen a closed lead outside the condition written in its row of `docs/ETAT.md` §6.
4. Never publish the raw ratio alone; always give the same-head ratio.
5. Never divide a × across cards (L40S, A100) or across stacks (vLLM, us); compare AWQ and QTIP in GB/s.
6. State every memory comparison in b/param for the whole model, tables included.
7. Publish medians with ranges formed round by round, never a quotient of two minima.
8. Label every number *measured*, *computed* or *estimated*, with its accounting and its journal.
9. Before costing a rerun, exhaust `hf buckets ls`, `hf jobs logs`, `hf jobs inspect`; keep the raw output.
10. A test that skips for want of an archive fails and names that archive; mutate the code before declaring a gate green; `oracle` first on every backend.

## Conventions

- Code comments and documentation in English; conversation in French.
- Zero warnings from `cargo clippy --all-targets`.
- `docs/STYLE.md` for every living document: fact first, one idea a sentence, no em dash, no banners. A fact that
  changes is replaced, and the old one goes to `HISTORIQUE.md` with its date.
- The five core crates stay free of external dependencies and in `forbid(unsafe_code)`.
- Full suite before any commit that touches a format or the indexing; any change to the index map breaks format v1
  (`codebook_fingerprint` pins it).
- When documents disagree, the object wins: the sealed file, the journal. README and `docs/fiche-4b.md` take precedence
  over this map on the published object.
- Every paid run goes through `ops/run.py` with `--features fast-linalg`, and the job goes into `docs/data/jobs.csv`.
