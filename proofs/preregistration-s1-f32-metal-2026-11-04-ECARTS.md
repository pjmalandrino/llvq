# Deviations from preregistration-s1-f32-metal-2026-11-04.md

## É1. The binaries were not built at a clean `06e1602`

§2 says `ppl` is built from commit `06e1602`. The worktree is shared with other sessions, which committed during the
build: `fa5e739` (11:59:51 local) landed before the `ppl` binary was written (12:00:19 local), and `env.txt` of the
run records HEAD `fa5e739`. Uncommitted edits to `llvq-cuda` (`planesbench.rs`, `seg_host.rs`, a test) were also in
the tree, later committed as `c68ba42`.

Effect on the measurement: none expected. `git diff --stat 06e1602 HEAD` over `*.rs`, `*.cu`, `*.cuh`, `*.toml` and
`*.metal` touches only `llvq-cuda` (one bin, one module, one test), and `llvq-cuda` is linked into `llvq-llm` only by
the `cuda` feature (`llvq-llm/Cargo.toml:29, :51`). This build used `--features metal`, so every crate the `ppl`,
`oracle` and `artstat` binaries link has the source of `06e1602`. Both arms ran the same binary, sha256 `5f3ff087…`
(gate 5). The run is not voided.

## É2. The pairing was computed by a scratch script, not repository tooling

The repository has no ppl pairing binary. The paired t interval was computed by `pair.py`, kept beside the raw logs.
It was checked first on the published 8B device pair of `ppl-bases-carte-2026-10-05.txt` and reproduced it:
[−0.0091 %, +0.0097 %], 7 of 12 windows worse, against [−0.009 %, +0.010 %] and 7 of 12 published.
