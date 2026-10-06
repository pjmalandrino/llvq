# Deviations from the A100 prereg of 2026-10-06

The prereg is `preregistration-a100-2026-10-06.md`, sha256 `add482d7`, stamped before any run. It
is not edited. Written before the first A100 job launched.

## É1. The launcher refused sm_80, and the override is a new flag

`ops/run.py bench` refuses every card under sm_89 (`MIN_COMPUTE_CAP`), because the standard
image's candle kernels target 89. Its own comment names the fix it lacks, a per-family guard, and
says an `llvq-cuda` job on sm_80 goes through `hf jobs run` directly, as F4 did, with the override
named in its prereg. This prereg did not name it.

The three jobs instead pass `--kernels-cap 80`, a flag added for this run that states the compute
capability the job's kernels target and refuses a card below it. Each declaration is true of its
job: `planesbench` compiles every kernel through NVRTC under `LLVQ_NVRTC_ARCH=compute_80`; the
sm80 image compiles candle for 80; vLLM's kernels support sm_80. Nothing measured changes.

## É2. The sm80 image comes from commit 44a18d3, and a first publish went wrong

The prereg says the sm80 Space is rebuilt from the commit that carries it. It is rebuilt from
`44a18d3`, two commits later, whose only change is `ops/run.py`, outside the uploaded perimeter,
so the build context is the same.

A first publish from `5f4d5fb` uploaded `llvq-tetra/`, a local Python package with a 750 MB
virtualenv that the allow pattern `llvq-*/**` matched: 19,618 files in 28 commits before the Hub's
20,000-file limit refused the last one. The Hub's secret scanner then flagged numpy's own test
names as Lob keys; no key of ours was in it. The folder was deleted from the Space (commit
`c9b72043`), and `44a18d3` excludes it and refuses any perimeter over 1,000 files. The republished
Space holds 363 files, as the canonical one does. Its sha, after the build, is written in the
journal before the served job launches.

Written before the served job launched: the sm80 Space finished its build at sha
`7903f89dea63dc10fec1fd430ba5445d1be71363`. Its `COMMIT` file names `44a18d3`, a clean perimeter,
and `CUDA_COMPUTE_CAP` rewritten to 80; it holds 363 files.
