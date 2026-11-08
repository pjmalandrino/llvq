# Architecture and code-quality audit, 2026-11

Audit of `main` at `3e2d4f0`, 2026-11-07, read only: nothing built, nothing run, nothing deleted, $0. Every number is
*measured* (counted with `wc`, `git grep`, `git log` or a small scanner over `git ls-files`) unless it says
*estimated*. The working notes behind each section were kept outside the repository; the counts can be redone with
the commands named in each section.

## Summary

The repository holds 157,737 lines of Rust in 9 crates, 18,676 lines of CUDA and Metal in 72 kernel files, and about
36,000 lines of Python and shell (*measured*). The served product is a small part of it. Of 101 binaries and examples
(51,506 lines), 19 serve or build the product and 66 can leave the default build (*measured*, 24,377 lines).

The architecture is sound at the crate level. The five core crates are dependency-free and `forbid(unsafe_code)`, CI
checks it, and the GPU backends sit behind five traits in `llvq-llm/src/device.rs`. The debt is inside `llvq-llm` and
`llvq-cuda`: four files over 3,000 lines, measurement modes woven into the served binaries, and parsing and refusal
logic copied by hand.

Main findings:

1. **Three pinned digests govern what can be deleted.** The served NVRTC units (`6193…`, `e269…`) and the v9 4E module
   (`1b422b62`) hash text that also carries Slot32 and the v3/v5 forms. Removing one byte of `llvq_slot.cuh`,
   `matvec.cu`, `llvq_pers.cuh`, `pers_4e.cu` or `pers_segment.cu` moves a pin, and a moved pin means a new paid door
   at each size. Most cleanup can avoid those files.
2. **The runtime did not serve what `docs/ETAT.md` says is served.** At `3e2d4f0`, `llvq-runtime/library.json` named the
   v4 config for the 4B on CUDA and the grouped configs for 8B and 14B. PR #15 (`7be533e`) moved it to v9 at all three
   sizes the same evening.
3. **One measurement mode cannot run.** `LLVQ_PERS_SEGMENT` lists itself among the variables it refuses
   (`llvq-llm/src/bin/fusedrun.rs:1971`), so `=1` always fails. No job, journal or prereg uses it.
4. **Two variables change the served load without being refused beside a config**: `LLVQ_PREFILL`
   (`llvq-cuda/src/tile.rs:154`) and `LLVQ_METAL_KERNEL` (`llvq-llm/src/fused_metal.rs:462`, unknown values warn and
   fall back).
5. **About 2,150 of the 3,314 lines of `fusedrun.rs` exist only for measurement modes** (*estimated*). Five modes have
   no setter left and no journal for two of them (`GRAPH_AB`, `GRAPH_DIAG`, `KV_AB`, `KV_PREALLOC`, `FUSE_AB`, about
   470 lines, *estimated*).
6. **CI covers 7 of 9 crates and no Python.** The 315 tests of `llvq-cuda` and `llvq-metal`, the `cuda` and `metal`
   features of `llvq-llm`, and 173 Python tests never run there. CI was red on every push to `main` from 2026-09-28
   to 2026-11-07.
7. **Five tests in `model.rs` still assume candle's CPU matmul rounds the same across shapes.** Three tests of that
   class fell on the Linux runner on 2026-11-07 and were relaxed.
8. **`CLAUDE.md` drifts from the code**: 49 variables read by the code are missing from it, and the `unsafe` claim in
   `llvq-llm/src/lib.rs` is wrong.

The plan in §4 has 16 batches. The first six remove or fence about 25,500 lines, fix the two defects above, and touch
none of the pinned files. Splitting the giant files comes late, because three open branches edit `model.rs` and
`fusedrun.rs`. Everything paper 2 cites stays reproducible from tag `v0.0.2`, plus a new tag taken before the first
deletion.

## 1. The architecture as it is

### 1.1 Crates

| crate | Rust lines | files | bins / examples / test files | `#[test]` | path deps | `unsafe` sites | lint |
|---|---:|---:|---|---:|---|---:|---|
| `llvq-core` | 1,141 | 6 | 0 / 0 / 1 | 17 | none | 0 | forbid |
| `llvq-search` | 8,090 | 22 | 0 / 0 / 8 | 73 | core | 0 | forbid |
| `llvq-quant` | 6,740 | 15 | 0 / 0 / 9 | 89 | core, search; `faer` optional | 0 | forbid |
| `llvq-artifact` | 14,394 | 31 | 0 / 0 / 21 | 153 | core, search, quant | 0 | forbid |
| `llvq-bench` | 24,526 | 59 | 20 / 26 / 7 | 97 | core, search, quant, artifact | 0 | forbid |
| `llvq-metal` | 10,368 | 20 | 8 / 1 / 9 | 68 | core, search, artifact; `metal` 0.29 (macOS); dev: bench | 30 | none |
| `llvq-cuda` | 28,890 | 45 | 10 / 0 / 43 | 247 | **bench** (normal); core, search, artifact, quant and `cudarc` (Linux) | 63 | none |
| `llvq-llm` | 58,663 | 104 | 33 / 1 / 44 | 551 | core, search, artifact, quant; candle, tokenizers, hf-hub, parquet; `llvq-cuda` (Linux, feature `cuda`); dev: bench, `llvq-cuda` (macOS) | 48 | none |
| `llvq-runtime` | 4,925 | 15 | 1 / 1 / 5 | 30 | llvq-llm; axum, tokio, clap, utoipa, sha2, ureq | 0 | forbid |

Total: 157,737 Rust lines, 1,325 `#[test]` functions (*measured*). The binary `llvq` lives in
`llvq-runtime/src/main.rs` through a `[[bin]]` entry.

Dependency direction, from the Cargo files:

- core ← search ← quant ← artifact ← bench: a clean chain, no external dependency, enforced by the CI `lint` job.
- `llvq-llm` sits on the four core crates and pulls `llvq-cuda` on Linux under `--features cuda`.
- `llvq-runtime` sits on `llvq-llm` and uses 8 of its modules.
- `llvq-cuda` depends on `llvq-bench` as a normal, untargeted dependency, for two binaries (`f1rankfloor`,
  `f1floorbench`) and four tests. So the served build, `llvq-llm --features cuda`, compiles the benchmark crate.
- 28 `include_str!` cross crate boundaries by relative path: `llvq-llm` → `llvq-cuda` 11, `llvq-metal` → `llvq-llm`
  11, `llvq-cuda` → `llvq-llm` 4, two more. Kernel text is shared across crates by path, not by crate API.

Inside `llvq-llm`, every module is public (762 `pub` items against 8 `pub(crate)`). Fan-in is highest for `model` (16
importers) and `fused` (8). Twelve module pairs import each other; the real knot is `model` ↔ `fused` ↔ `device` ↔
`attn`. `served` ↔ `served_decode` and `served_decode` ↔ `persist4e_cuda` also import each other.

### 1.2 Python packages and scripts

| package | lines | tests | how they run | in CI |
|---|---:|---:|---|---|
| `llvq-tetra/` (PyPI 0.1.0) | 3,117 package | 49 | `uv run --group dev pytest`, `uv.lock` committed | no |
| `ops/llvqtune` | 2,243 package | 113 | `--extra torch` plus pytest; 7 of 15 test files are pure Python | no |
| `ops/mlx_tetra/` (served MLX door) | 2,305 in 10 scripts | 11 (`test_served.py`, stdlib only) + `mutate.py` | PEP 723 headers, no pyproject | no |
| `ops/metal-package/` (prototype) | 72-line launcher | 0 | none | no |

Lines and tests *measured*. The kernel copies in `llvq-tetra/llvq_tetra/data/` are byte-identical to their sources
today, and `llvq-llm/tests/hfpack.rs` (`the_shipped_shader_is_the_repositorys`, `the_shipped_cuda_closure_is_complete`)
keeps them so in the CI `test` job. The MLX door imports `llvq_tetra` and reads the Metal shaders from
`llvq-llm/kernels`, so no Tetra decoder is duplicated by accident.

`ops/` holds 61 top-level scripts (15,946 lines) and `ops/jobs/` 102 files (12,292 lines). Of the 61: 26 are live tools,
8 are helpers called by jobs, 27 are one-shot analyses cited only by journals (6,047 lines), and `ops/hf_speed_mac.py`
is cited nowhere (*measured*). `ops/README.md` indexes 2 of the 61.

### 1.3 Kernels

72 kernel files in three places: `llvq-cuda/kernels` (45), `llvq-llm/kernels` (12 CUDA, 8 Metal), `llvq-metal/shaders`
(7). The largest is `llvq_pers.cuh` (2,069 lines, 78 device functions, 27 `#if`). The longest kernel functions are
`attn_dec_body` (297 lines), `pers_segment_body` (257) and `pers_4e_body` (252).

### 1.4 `unsafe` boundaries

141 `unsafe` sites: `llvq-cuda` 63, `llvq-llm` 48, `llvq-metal` 30. By category: kernel launches 80, Metal buffer reads
30, event tracking 10, device allocation 7, mmap 6, `DeviceRepr` 3, cuLink and cuBLAS FFI 2. Three are not at a hardware
boundary: `from_raw_parts` reinterpretations of host data at `llvq-cuda/src/e1v_host.rs:88` and in two e1v tests. They
are sound given `repr(C)` and the build-time asserts. No `transmute`, no unchecked indexing. 33 of the 141 carry a
`SAFETY` comment.

### 1.5 Where `CLAUDE.md` no longer matches the code

| claim | code |
|---|---|
| `llvq-llm/src/lib.rs:7-10`: "`unsafe` is allowed here, and only here in the workspace", for mmap | `llvq-cuda` and `llvq-metal` also use it; mmap is 6 of `llvq-llm`'s 48 sites |
| The table names the GPU crates "allowed" | none of `llvq-cuda`, `llvq-metal`, `llvq-llm` carries any lint attribute; `[workspace.lints]` is absent, as `CLAUDE.md` says |
| Environment table names 59 variables | the code reads 113 project variables; 49 read by Rust or Python are undocumented, among them `LLVQ_MMLU_DUMP` (set by 35 jobs), `LLVQ_DATASET_REV` (10 jobs), `LLVQ_PREFILL`, `LLVQ_METAL_KERNEL`, `LLVQ_SEG_TETRA` |
| `LLVQ_DTYPE`: `f32` default, `f16` | also accepts `bf16` and aliases; `mmlu` defaults to `f16` |
| `LLVQ_SEQ_BLOCK`: any other value refused | `1` is accepted |
| `LLVQ_CALIB_SEED` … `LLVQ_SEQ_BLOCK`: "see `docs/METHODE.md`" | `METHODE.md` names none of them |
| `TIME_EVENTS`, `BENCH_ARMS`, `SEG_ARMS` refused beside a config | only `planesbench` reads them, and it never reads `LLVQ_CONFIG`; the refusal guards nothing |
| ETAT §2: v9 served at 3 sizes | `llvq-runtime/library.json` and `library.rs:115-131` serve v4 (4B) and grouped (8B, 14B) |
| `.github/workflows/ci.yml` comments cite `docs/portage-noyau-cuda.md` and "CLAUDE.md §5, §7" | the file is in `docs/archive/`; `CLAUDE.md` has no numbered sections |
| `AGENTS.md` | an old fork of `CLAUDE.md`: 82 of 131 non-empty lines differ, and it still describes `Planes14` as served |
| `proofs/README.md` counts 22 preregs | 167 exist |

All 59 documented variables are still read, and every documented default matches the code (*measured*). Nine binaries
default `LLVQ_MODEL` to Qwen3-0.6B silently, `seal` and `smoke` among them; `CLAUDE.md` does not say so.

## 2. Dead and measurement-only code

### 2.1 The constraint: three pinned digests

Three tests in `llvq-cuda/src/pers.rs` pin the sha256 and length of the text the served doors compile. The door jobs
refuse a log that does not print the same digests.

| pin | test (`pers.rs`) | parts | cited by |
|---|---|---|---|
| `61930943…`, 101,625 B | `the_served_unit_is_the_published_one` (2516) | `llvq_slot.cuh`, `matvec.cu`, rotation, f1rank ×2, tetra48, `tv_tetra48_h.cu`, `emb_q4.cu`, `tv_q4_h.cu` | 160 files (128 journals) |
| `e269ea5d…`, 106,947 B | `the_tiled_served_unit_is_…` (2559) | the above + `tv_q4_tile.cu` | 17 files, `ops/jobs/served-v9-big.sh:57` |
| `1b422b62…`, 252,757 B | `the_4e_v9_module_is_the_one_the_doors_gate_on` (2546) | `llvq_slot.cuh`, `matvec.cu`, f1rank ×2, tetra48, rotation, `llvq_pers.cuh`, `tetra48_pers.cu`, `tetra48_persall.cu`, `pers_ref.cu`, `pers_segment.cu`, `pers_4e.cu` | 20 files, `served-v9-big.sh:58,152,158` |

What moves them:

- Any Slot32 text removed from `llvq_slot.cuh` or `matvec.cu` moves all three. The served code uses only their shared
  helpers (`h2f`, `f2h`, `warp_sum`, `tail_dot_h`, `LLVQ_DIM`) and never `ClassRec`, `slot_*` or `tv_slot*`. The same
  two files ship with their sha256 in the frozen safetensors package (`llvq-llm/src/hfpack.rs:791-799`) and in the
  `llvq-tetra` 0.1.0 wheel.
- Any byte removed from `llvq_pers.cuh`, `pers_4e.cu` or `pers_segment.cu` moves `1b422b62`. So does removing the L1/L2
  measurement kernels (`tetra48_pers.cu`, `tetra48_persall.cu`, `pers_ref.cu`, 603 lines), which 4E never calls but
  which sit in `PERS_UNIT_PARTS`.
- `pers_trows.cu`, `pers_q4rows.cu` and `pers_ctr.cu` compile in their own module and move no pin.

A moved pin costs three constants in `pers.rs`, two in `ops/jobs/served-v9*.sh`, a re-copy into `llvq-tetra`, and a
new served door per size or an operator-accepted equivalence argument ($2.65 for the three v9 doors, *computed*
from their journals). Removing only `#if` branches not taken under form 7 leaves the preprocessed text identical except
line numbers; a `clang -E` comparison on the Mac could show that, but accepting it is the operator's call.

### 2.2 4E row forms

The host selection is a clean enum (`served::Rows4e`, `llvq-llm/src/served.rs:198-283`): absent means v3, and v5, v7, v8
are refused by name. The forms are cumulative: v9 = v8 + tiled int4, v8 = v7 + attention ring, v7 = v6 + section 7c,
v6 = v4 + section 7b. So the code of v4, v6, v7 and v8 is v9's code. Only v3 (`pers_tetra_stream`, about 80 lines) and
v5 (about 110 kernel lines) own code v9 never runs (*estimated* spans).

| form | owns | live dependants | verdict |
|---|---|---|---|
| v3 | ~80 kernel lines, the default of an absent `rows` | `qwen3-4b-tetra-e4-4e.json`, 12 journals | keep until `rows` is mandatory, then remove after tag |
| v4 | selector only | **the runtime's 4B CUDA config** (`library.json:16`), Dockerfile `test -f` line 197 | keep until the runtime moves to v9 |
| v5 | ~110 kernel lines, 56 test lines (`pers_matches_served.rs:1527-1582`) | none; closed in ETAT §6 | remove after tag; kernel part moves `1b422b62` |
| v6 | selector, a few `== 4` variants | v9's door proves its tokens against the v6 config (`served-v9.sh:43`) | keep while that door is the reference |
| v7, v8 | selector only; 114 and 80 test lines | 2 journals each | remove the selectors and their bench entries after tag; the code stays as v9's |

Keeping v9 alone would remove about 300 kernel lines, 420 test lines and 60 host lines (*estimated*). That is a small
gain against a pin move, so §4 removes only the host selectors first.

### 2.3 Other persistent and decode code

| item | lines | in a pinned unit | verdict |
|---|---|---|---|
| route P segment host, `llvq-llm/src/persist_cuda.rs` + `fusedrun` mode `LLVQ_PERS_SEGMENT` | 373 + ~310 (*estimated*) | the kernel body `pers_segment_body` is 4E's MLP phase, so the `.cu` stays | host side removable; five helpers are reused by `persist4e_cuda.rs:50` and must move first |
| L1/L2 persistent arms (`LLVQ_PERS_ARMS`, `LLVQ_TETRA_OCC`, `LLVQ_PERS_MICRO`) | 603 kernel + ~660 host in `pers.rs` | yes | drop host plumbing after tag; leave the kernels until the next pin move |
| row-loop and step-1 microbenches (`pers_trows.cu`, `pers_q4rows.cu`, `pers_ctr.cu`, `LLVQ_TROWS`) | 1,001 kernel + 265 test + host | no | remove after tag |
| `llvq-cuda/src/persbench.rs` | 1,664 | n/a | goes with the items above |
| gestures (`gesture_cuda.rs`, `gesture_spin.cu`, `LLVQ_GESTURE_AB`) | 534 + 14 + ~390 in `fusedrun` | no | remove after tag; keep `argmax_key.rs`, whose key order 4E's head shares |

No "4D" exists; "etage0" appears only in `docs/archive/` and `proofs/`.

### 2.4 Fused-kernel layouts

`LLVQ_FUSED_LAYOUT` selects a closed enum (`fused::FusedLayout`, `llvq-llm/src/fused.rs:78-125`) with plain match arms.
The Ball layouts form a chain (Planes14 is transcoded from Slot32, Planes12x reuses Planes14's records, Golay70 reuses
Planes12x's), so they come out from the end. Paper 2's §5.2 table and Appendix A come from `planesbench` with all ten
arms (`ops/jobs/banc-252.sh:57`). The image commit of those runs, `5333ac8`, is an ancestor of `v0.0.2`, so the tag keeps
them reproducible.

| layout | kernel lines | test lines | paper 2 | verdict |
|---|---:|---:|---|---|
| Golay70 | 712 | 1,195 | Appendix A rows | remove after tag (closed, ETAT §6) |
| Planes12x | 473 | 1,180 | Appendix A row | remove after tag, after Golay70 |
| Planes14 | 1,014 | 1,459 | the main comparison of §5.2, tile sweeps, A100 | keep; removal is an operator decision |
| Slot32 | inside the pinned units | 249 (`host_matvec.cpp`, `decoder_matches_rust.rs`) | Appendix A | keep the kernel text; host code goes only with Planes14 |

Host code also goes with Golay70 and Planes12x: about 830 lines of `llvq-artifact/src/runtime.rs`, `golay70_host.rs`
(520), and their arms in `fused.rs`, `fused_cuda.rs` and `planesbench.rs` (*estimated* several hundred). Two stale
stopgaps, `llvq-cuda/src/planes14_host.rs` and `golay70_host.rs`, say in their headers "delete when the artifact
transcoder exists"; it exists (`llvq-artifact/src/runtime.rs:485`, `:1540`).

The default of `LLVQ_FUSED_LAYOUT` is `planes14`, an unserved layout. A run with no config measures Planes14 silently.
If Planes14 ever goes, the variable should become mandatory rather than default to `tetra48`, so an old journal command
fails instead of measuring another layout.

### 2.5 Configs

| config | dependants | verdict |
|---|---|---|
| `qwen3-{4b,8b,14b}-tetra-e4-4e-v9.json` | served | keep |
| `qwen3-{4b,8b,14b}-tetra-e4.json` (expanded) | paper 2 `README.md:63`; runtime Metal entries | keep |
| `qwen3-4b-tetra-e4-4e-v4.json`, `qwen3-{8b,14b}-tetra-e4-grouped.json` | compiled into the runtime (`library.rs:115-131`) | keep until the runtime moves |
| `qwen3-4b-tetra-e4-grouped.json` | 47 journals, 10 jobs, the moving Hub repository | keep |
| `qwen3-4b-tetra-e4-4e.json` (v3), `-4e-v6`, `-kernel-from1280`, `qwen3-*-tetra-q5.json` ×3 | journals, jobs, Dockerfile `test -f` lines 183-201, `served_config.rs` | remove after tag, with those edits; low value, each file is about 1 KB |
| `*-metal-*-NOT-SERVED.json` ×2 | `bin/metalrun`, `ops/metal-package.sh:20-21`, one journal | remove with the Metal package prototype |

### 2.6 Binaries and examples

72 bins and 29 examples, 51,506 lines (*measured*). Categories, from each target's references in the Dockerfile, ops,
living docs, journals, preregs and paper 2:

| category | targets | lines |
|---|---:|---:|
| served or product (`llvq`, `seal`, `hfpack`, `mmlu`, `gsm8k`, `ppl`, `fusedrun`, …) | 19 | 12,343 |
| reproduction of paper 2, not already served | 14 | 12,212 |
| tool used by ops or a living doc | 11 | 7,002 |
| weak tool (one command-list line, a closed lead's script, or only the image) | 15 | 5,227 |
| measurement only (journals and preregs only) | 33 | 13,695 |
| dead (no reference outside code comments, `HISTORIQUE.md`, `docs/archive/`) | 9 | 1,027 |

Dead: `f1linear`, `f1mids`, `f1radii`, `f1rankenc`, `f1time`, `tetrascales` (examples, `llvq-bench`), `gaindiag`,
`probe` (`llvq-llm`), `hello` (`llvq-metal`).

Grouped into batches (lines *measured*):

| batch | targets | lines | cited by |
|---|---:|---:|---|
| B1 F1 encoder and table study (`f1*` examples, `tetrascales`, bin `f1bench`) | 18 | 6,779 | journals of 2026-09-04/05 |
| B2 format-v1 decoder era (`arrbits classprofile decfast decfull decprofile decbench lcap lswap radixstudy classhist encbench betasweep cholbench tetraencbench`) | 14 | 4,521 | journals 2026-07-31 to 08-13; `CLAUDE.md` command list; tests run `classhist`, `lswap`, `radixstudy` |
| B3 Metal P1 era (`hello decode decreal p1v0 matvec mslcheck rankbench`, example `metalsplit`) | 8 | 4,382 | P1 journals and preregs; `thesis` stays for `LAUNCH_ME.md` |
| B4 CUDA closed leads in the image (`graphbench nullkbench f1floorbench matvec`) | 4 | 1,876 | Dockerfile and `ops/jobs/image-smoke-14b.sh:123` |
| B5 `llvq-llm` quality diagnostics (`gaindiag probe cosdiag kltemp tetraalt artscale hcapture gbench`, example `hstats`) | 9 | 2,125 | journals 2026-08-25 to 09-16; row 18 of `ROADMAP-27B-DEV.md` plans refusals in six of them |
| B6 E8, gain, spherical and row studies | 6 | 1,872 | refuted or closed leads |
| B7 paper-2-only reproduction (`tetrashell f1rankbench driftcheck f1encreal f1count f1recdump hratio`) | 7 | 2,822 | `paper2/PROVENANCE.md` cites two by path: fence, never delete |

What paper 2 needs to rerun: the six bins its README names (`fusedrun`, `mmlu`, `mmlupair`, `gsm8k`, `gsm8kpair`,
`rtbits`), plus the producers behind `PROVENANCE.md` and the `source` columns of `docs/data/paper2-*.csv`: `planesbench`,
`ppl`, the build chain (`smoke`, `seal`, `rowscale`, `export`, `embedq`, `int4swap`), `preflight`, `rotbench`, and the
B7 targets. `make` and `check_tables.py` read only the CSVs. `ARTIFACT-EVALUATION.md` and `LAUNCH_ME.md` describe the
paper 1 object and promise `cargo test`, `run`, `fusedrun`, `thesis`, `ppl`, `smoke`, `seal`.

The recommended mechanism is a per-crate `archive = []` feature with `required-features = ["archive"]` on each fenced
target. Paths do not move, so the frozen journals and `PROVENANCE.md` keep valid links. A new crate would break them.

### 2.7 Environment variables and measurement modes

113 project variables are read (*measured*): 101 `LLVQ_*` by Rust at run time, one at build time
(`LLVQ_BUILD_COMMIT`), 6 by Python only, 5 non-`LLVQ` ones. There is no central module: 157 direct read sites plus 19
through helpers, 129 of them in `llvq-llm`. `served.rs` centralises only the five variables a config carries, and
shares their parsers with the config, which works.

The 17 `MEASUREMENT_MODES`:

| mode | est. lines | set by a job since 2026-11-01 | verdict |
|---|---:|---|---|
| `GRAPH_AB`, `GRAPH_DIAG`, `KV_AB`, `KV_PREALLOC`, `FUSE_AB` | ~470 | no; `GRAPH_AB` and `KV_AB` have no journal | remove |
| `PERS_SEGMENT` | ~310 | no; unreachable | remove (or fix the one line and run it) |
| `TIME_PHASES`, `TIME_EVENTS`, `SEG_ARMS` | ~485 | no; last journals August–September | remove after tag |
| `TRIM_AB` | ~150 | last job 2026-10-30 | remove once the trims are recorded as served |
| `GESTURE_AB`, `BOUNDARY_AB` | ~590 | one job each, one-shot verdicts | remove after tag |
| `GQA_AB`, `GQA_PPL`, `PERS_4E`, `INT4_WIDE_CHECK`, `BENCH_ARMS` | not counted | yes (3, 7, 7, 1, 8 jobs) | keep |

Line counts *estimated* by brace matching on the gating blocks and their helpers. Parsing is duplicated: six 0/1 flag
parsers with three behaviours (9 flags read any value other than `1` as off, `VERIFY_CACHE` turns on when set to `0`),
three parsers for `LLVQ_THREADS`, the "runs alone" exclusion lists written out 10 times and not symmetric (`GQA_PPL`
silently ignores `GRAPH_AB`, `KV_AB`, `FUSE_AB`), and the `LLVQ_KERNEL_DIR` loader copied 12 times. `NVRTC_ARCH` and
`TILE_BLOCKS` panic on a bad value where the others return an error.

### 2.8 Scripts, jobs and raw data

- 27 one-shot analyses in `ops/` (6,047 lines) are cited only by journals; `ops/f1a_shaping.py` is cited by
  `paper2/PROVENANCE.md` and stays.
- `ops/jobs/`: no job names a missing binary, variable or config (*measured* over the 46 variables used). 27 jobs are
  cited nowhere by file name. `dclm-rownorms.sh` and `dclm-rownorms-mmlu.sh` name a prereg that only exists as a
  `BROUILLON-` draft. `llvq-cuda/src/pers.rs` does `include_str!` on `ops/jobs/pers-l2.sh` in a test, so that job cannot
  move alone.
- The Dockerfile's build list and `COPY` list match today (22 and 22), and every `COPY --from=build` source is in
  `UPLOAD_ALLOW`. A third list must stay in step: `ops/jobs/image-smoke-14b.sh:123`.
- The tracked tree weighs 208.6 MB, of which 145.7 MB (70 %) is the 118 `-brut/` raw directories under `docs/mesures/`.
  The largest files are `sigma.json` (22.5 MB, cited), a `server.log` (12.7 MB, never cited) and six `.ptx`/`.cubin`
  files of 4.7 to 8.6 MB. The object store is 61.0 MB over 1,258 commits. Git LFS is installed but unused. Rewriting
  history is out of scope: the journals are frozen and anchored.

## 3. Quality

### 3.1 Giant files and long functions

4,588 Rust functions; 117 over 100 lines, 24 over 300 (*measured* with a brace-depth scanner that strips comments and
strings). In library code: 2,430 functions, 38 over 100 lines.

| lines | function | file |
|---:|---|---|
| 3,813 | `run` | `llvq-cuda/src/bin/planesbench.rs` (75 % of the file) |
| 1,658 | `main` | `llvq-llm/src/bin/fusedrun.rs` |
| 1,020 | `run` | `llvq-bench/examples/rhoapply.rs` |
| 969 | `run` | `llvq-metal/src/bin/rankbench.rs` |
| 856 | `run` | `llvq-cuda/src/bin/matvec.rs` |
| 808 | `main` | `llvq-llm/src/bin/smoke.rs` |
| 561 | `FusedRuntime::new` | `llvq-llm/src/fused_cuda.rs` (longest in library code) |
| 454 | `quantize_model_capturing` | `llvq-llm/src/calib.rs` |
| 421 | `install_rows_as` | `llvq-llm/src/persist4e_cuda.rs` |
| 405 | `pack` | `llvq-llm/src/hfpack.rs` |

What the four largest library files mix:

- `model.rs` (4,303 lines, 194 functions): activation capture, phase timing, RoPE, a KV cache with three storage modes,
  attention variants, projection dispatch over dense, lattice, int4 and segment, `Block`, a `Qwen3` impl of about 890
  lines with nine generation loops, and sampling. Tests from line 3,034. A natural split: `kv.rs`, `proj.rs`,
  `block.rs`, `generate.rs`, `capture.rs`.
- `fused_cuda.rs` (4,184 lines, 111 functions, no test module): NVRTC unit assembly, device trait impls, read-only views
  for 4E, nine candle custom ops, uploads, 12 layout launchers (including Planes12x and Golay70), and four process-global
  measurement switches.
- `fused.rs` (3,082 lines): layout enum, NVRTC part lists, host streams and transcoding for every layout.
- `fusedrun.rs` (3,314 lines): the served door, 4E and seven measurement modes in one binary; it reads 24 variables, has
  29 `allow(dead_code)`, and pastes the same event-tracking `unsafe` block six times in `main`.

`served.rs` (764 lines) and `served_decode.rs` (291) are cohesive.

### 3.2 Duplication

| what | copies |
|---|---|
| SHA-256 | 4 hand-written (`llvq-cuda/src/sha256.rs`, `llvq-llm/src/digest.rs`, `rankbench`, `rhoapply`) + the `sha2` crate in the runtime |
| f16 helpers `f16_bits`, `f16_to_f64` | `llvq-cuda/src/lib.rs` and `llvq-metal/src/lib.rs`; `llvq-llm` uses `half` |
| stats helpers | `spread` ×5, `median_range` ×5, `worst_error` ×3, `le_u32` ×6 |
| the `tiny` test model | ×7 identical in `llvq-llm/tests`, which has no `common` module |
| host files pulled in by `include!` | 8 sites, one under `allow(clippy::all)` |
| Metal host plumbing | `llvq-metal` (metal-rs 0.29) and `llvq-llm/src/fused_metal.rs` (objc2-metal), 200-300 lines (*estimated*) |
| argument parsing | 65 of 100 bins parse `env::args` by hand |

The kernel copies in `llvq-tetra` are generated and gated (§1.2), so they are not debt.

### 3.3 Error handling

Library code is clean on panics: 16 `unwrap()`, 158 `expect(` and 11 `panic!` in about 54,000 lines (*measured*, split by
path). Four error styles coexist: typed enums (`llvq-artifact`, the runtime), `Result<_, String>` (all of `llvq-cuda`,
19 files of `llvq-llm`), `anyhow` (17 files of `llvq-llm` and every bin), and candle's `Result` (15 files). The glue
between them is 253 `map_err` into anyhow, 171 `candle::Error::msg` and 311 `to_string`/`format!`. Thirteen files mix
styles; `fused_cuda.rs` has 14 String and 183 candle uses.

### 3.4 Platform `cfg`

288 `cfg` sites; `llvq-llm` has 118 (54 `target_os = "linux"`, 62 `feature = "cuda"`, 23 macOS). The backend split is
clean at module level through the five traits of `device.rs`, and `model.rs` has 4 `cfg` sites. The interleaving sits in
`fusedrun.rs` (38), `served_decode.rs` (8), `served.rs`, `attn.rs`, `fused.rs` and `llvq-runtime/src/library.rs`.

### 3.5 Tests

| crate | `#[test]` | run by CI |
|---|---:|---|
| core, search, quant, artifact, bench | 429 | yes, in release-like and debug profiles |
| `llvq-llm` | 551 | the CPU half; tests gated `cfg(any(target_os = "macos", feature = "cuda"))` drop out silently on Linux |
| `llvq-runtime` | 30 | yes |
| `llvq-cuda` | 247 | no: the crate cannot build on a Linux runner without `nvcc`; runs on the Mac (host C++) |
| `llvq-metal` | 68 | no: macOS only |
| Python (3 packages) | 173 | no |

The sealed-archive tests carry an unconditional `#[ignore]` and fail by name without their file; CI does not pass
`--include-ignored`, so they run only on the Mac.

Fragile tests. On 2026-11-07 three tests fell on the Linux runner, the first time it reached `llvq-llm`:
`attention_forms_agree` and `paired_decode_scores` (`model.rs`, commit `f43a774`) and
`a_divergent_history_is_cut_back_to_the_common_prefix` (`tests/session.rs`, commit `abdbaae`). All three assumed that
candle's CPU matmul gives the same bits for two shapes or layouts of the same product. That holds on the Mac and not on
x86-64. Five assertions of the same class remain in `model.rs` (lines 3472-3473, 3613, 3869-3874, 4086-4089: `same_bits`
over logits from different call shapes). They pass on the runner today and rest on an implementation detail of candle's
gemm. Bit equality is safe where the same kernel runs on the same shape (kernel against its host reference, cache
against itself); across shapes, a tolerance with the strict check kept on macOS is the pattern `f43a774` already uses.

### 3.6 Docs against code

| document | lines | target in `STYLE.md` |
|---|---:|---:|
| `docs/HISTORIQUE.md` | 1,186 | 400 |
| `docs/ETAT.md` | 273 | 100 |
| `docs/METHODE.md` | 256 | 150 |
| `CLAUDE.md` | 194 | 120 |
| `README.md` | 176 | 120 |

Stale or parked: `AGENTS.md` (§1.5), `docs/plan-300.md` (opens by saying 300 tok/s is out of reach),
`docs/plan-debit-noyau.md`, `docs/plan-metal-debit.md`, `docs/runbook-rownorms-2026-09-22.md`, `docs/ROADMAP-27B*.md`
(drafts, not adopted), `docs/architecture-c4.html` and `docs/cours-layouts-runtime.html` (predate Tetra),
`proofs/README.md`, `ops/README.md`. The living Markdown docs carry no em dash; `docs/modele-erreur.md` has 9 and the old
HTML files 71 and 58. `ops/Dockerfile.cuda` comments keep French paragraphs and emoji banners.

## 4. Cleanup plan

### 4.1 Rules for every batch

- **Tag first.** One tag on `main` before the first deletion (for example `pre-clean-2026-11`). Tag `v0.0.2` already
  freezes everything paper 2 ran; the new tag covers what came after, including all of 4E, which no tag holds today.
- **Never touch the frozen perimeter**: the three sealed repositories and their digests (`886391a8`, `7bdb9a55`,
  `61db37fe`), the safetensors repository, `docs/hub/qwen3-*-sealed.md` and `qwen3-4b-safetensors.md`, the journals in
  `docs/mesures/`, the timestamped preregs, `docs/archive/`, and the `paper2/` text.
- **Keep the three pinned digests still** unless a batch says otherwise and the operator accepts the door cost.
- One batch, one commit, one PR. `cargo test`, `cargo clippy --all-targets`, the Linux type-check
  (`ops/check-linux-cuda.sh`) and, where the image changes, the three image lists in step.

### 4.2 Batches, ordered by gain over risk

| # | batch | gain | risk | prerequisite |
|---:|---|---|---|---|
| 1 | **Defects and doc truth.** Fix or remove `LLVQ_PERS_SEGMENT`; put `LLVQ_PREFILL` and `LLVQ_METAL_KERNEL` under the served refusal; make `METAL_KERNEL` refuse unknown values; drop the three `planesbench`-only names from `MEASUREMENT_MODES`; rewrite the env table of `CLAUDE.md` (49 missing, 4 mismatches); fix `llvq-llm/src/lib.rs:7-10`, the `ci.yml` pointers, `proofs/README.md`; replace `AGENTS.md` by a pointer to `CLAUDE.md` | two served-path holes closed, one dead mode settled, the map true again | very low; no kernel text | none |
| 2 | **Runtime on the served configs.** Done by PR #15 (`7be533e`) | the `llvq` binary serves what ETAT says; unblocks removing v4 and grouped 8B/14B later | done | none |
| 3 | **Delete the 9 dead targets** (§2.6) | 1,027 lines, 9 targets | none | tag |
| 4 | **Delete the no-setter modes**: `GRAPH_AB`, `GRAPH_DIAG`, `KV_AB`, `KV_PREALLOC`, `FUSE_AB`, and `PERS_SEGMENT` with the host of `persist_cuda.rs` (move its five shared helpers into `persist4e_cuda.rs` first) | ~1,150 lines (*estimated*), 6 names out of `MEASUREMENT_MODES`, `KvStore::Prealloc` gone from `model.rs` | low; `fusedrun` and `model.rs` are not in a pinned unit; refusal-list tests to update | tag; coordinate with the batching branch, which edits `model.rs` |
| 5 | **Fence measurement-only targets** behind `archive` features: B1, B2, B3, B5, B6, B7, less the dead targets of batch 3 and the image's `gbench` (fence, never delete B7) | 52 targets, 21,317 lines out of the default build; about 30 % fewer `llvq-llm` link steps in CI's test job (*estimated*) | low; archived code stops being checked unless a non-blocking `--features archive` check job is added; gate the `CARGO_BIN_EXE_*` cases of `classhist`, `lswap`, `radixstudy` | tag; decide B5 with row 18 of `ROADMAP-27B-DEV.md`; `rankbench` and `mslcheck` in B3 are the operator's call |
| 6 | **CUDA image diet**: B4 plus `gbench` out of the build, `COPY` and `image-smoke-14b.sh` lists, then fenced | 5 of 22 image link steps; 5 targets, 2,033 lines | low; the three-lists trap; the Space must be republished from the branch (shared resource, operator go) | batch 5's feature |
| 7 | **`llvq-bench` off the served build**: make it an optional or bin-only dependency of `llvq-cuda` (move `f1rankfloor`, `f1floorbench` behind a feature) | the served build stops compiling the bench crate | low | batch 6 for `f1floorbench` |
| 8 | **CI coverage**: a Python job (`test_served.py` and the pure `llvqtune` tests, about a minute, *estimated*), a grep check of the three image lists, a non-blocking archive check; add `paper2/**` and `proofs/**` to `paths-ignore` | 173 Python tests partly covered; the two-lists trap checked mechanically; fewer useless Rust runs | very low; a few CI minutes | batch 5 for the archive check |
| 9 | **Measurement modes with stale verdicts**: `TIME_PHASES`, `TIME_EVENTS`, `SEG_ARMS`, `TRIM_AB`, `GESTURE_AB` with `gesture_cuda.rs`, `BOUNDARY_AB`; the row-loop and step-1 microbenches and `persbench.rs` | ~2,900 lines (*estimated*); `fusedrun.rs` near half its size | low to medium; the microbench kernels are outside the pinned units, the L1/L2 kernels are not and stay | tag; operator agrees the verdicts are final |
| 10 | **Shared helpers**: one stats module, one SHA-256 (hand-written, in a dependency-free crate), one f16 helper, `tests/common` for `tiny`, one flag parser, one exclusion table for the `fusedrun` modes, one `KERNEL_DIR` loader | 500-800 lines (*estimated*); makes the `PERS_SEGMENT` class of bug impossible | low; touches many files, pure refactor | batches 4 and 9 first, so less is refactored |
| 11 | **Golay70 then Planes12x**, kernels, tests, host, bench arms; delete the stale `planes14_host.rs`/`golay70_host.rs` stopgaps where the artifact transcoder replaces them | ~1,185 kernel + 2,375 test + ~1,400 host lines (*estimated*) | medium; `planesbench` loses two arms of paper 2's Appendix A at HEAD (still at `v0.0.2`); no pinned file touched | tag; operator accepts Appendix A reproducible from the tag only |
| 12 | **4E selectors v5, v7, v8**, host side only: enum refusals keep their text, `P4E_BENCH_FORMS` and their test sections go | ~250 test lines, simpler `install_rows_as` | low; kernel text untouched, no pin moves | tag |
| 13 | **Fragile tests**: replace the five cross-shape `same_bits` assertions by a tolerance, strict on macOS | no more red runner from a candle update | low | none |
| 14 | **Split the giant files** by pure moves: `model.rs` into five modules, `fused_cuda.rs` into runtime, ops, upload, launch, `fusedrun.rs` modes into modules, `planesbench run` into per-arm functions | files under ~1,500 lines; reviewable diffs later | medium only by conflicts: the batching, route-300 and plan-noyau branches edit these files | open branches merged or parked; batches 4, 9, 11 first |
| 15 | **Error and lint hygiene**: a typed error for `llvq-cuda`, `SAFETY` comments on the 108 uncommented `unsafe` sites, `[workspace.lints]` | uniform errors; the lint hole in integration tests closed | low; `[workspace.lints]` is a pending operator decision | operator decision |
| 16 | **Docs to their targets**: cut `HISTORIQUE.md`, `ETAT.md`, `METHODE.md`; move finished plans and the pre-Tetra HTML to `docs/archive/`; index `ops/README.md`; set a rule that raw files over 1 MB go to the bucket with a sha256 in the journal; add `.venv/` to `.gitignore` | readable entry points; the tree stops growing by raw logs | very low | none |

Batches 3, 5 and 6 delete or fence 66 targets and 24,377 lines (*measured*); batch 4 removes about 1,150 more
(*estimated*). None of batches 1 to 6 touches a pinned file. Batches 9 to 12 add about 8,700 lines (*estimated*).

### 4.3 What stays, whatever the batch

- The served path at three sizes: `served.rs`, `served_decode.rs`, `persist4e_cuda.rs`, `fused_cuda.rs` (Tetra48 and
  int4 parts), `llvq_pers.cuh` and the rest of the v9 module, `attn_dec.cu`, the Metal Tetra kernels and
  `ops/mlx_tetra/`.
- The build chain of the sealed files: `smoke`, `seal`, `rowscale`, `export`, `embedq`, `int4swap`, `hfpack`, `hfdense`,
  `tetratables`, and the dependency-free algebra reference `llvq-quant/src/linalg.rs`.
- Paper 2's producers: the six README bins, `planesbench` with its Planes14, Slot32, FP16, AWQ and Tetra arms, `ppl`,
  `rtbits`, `preflight`, `rotbench`, and the B7 targets behind the `archive` feature.
- `LAUNCH_ME.md`'s promises (`run`, `fusedrun`, `thesis`, `ppl`, `smoke`, `seal`) and the Planes14 path it walks, until
  the operator retires that document.
- The configs the runtime compiles in, the expanded configs paper 2 names, and the grouped 4B config the moving Hub
  repository serves.
- Every journal, prereg, `.ots` anchor and `-brut/` directory.

### 4.4 Decisions for the operator

1. Batch 1 is PR #19 (wave 0), and batch 2 is done by PR #15.
2. Batch 6: republish the CUDA Space after the image diet.
3. Batch 11: accept that Golay70 and Planes12x are reproducible from `v0.0.2` only.
4. Whether `LAUNCH_ME.md` and `ARTIFACT-EVALUATION.md` belong to the frozen paper 1 perimeter, which decides the fate
   of Planes14 and `thesis`.
5. `[workspace.lints]`, already pending.
6. When to take the pinned-digest move: the next kernel change that moves `1b422b62` anyway is the cheap moment to drop
   v3, v5 and the L1/L2 kernels from the v9 module.
