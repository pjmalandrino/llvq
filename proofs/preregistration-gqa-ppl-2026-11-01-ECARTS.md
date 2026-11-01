# Deviations from the GQA perplexity prereg of 2026-11-01

The prereg is `preregistration-gqa-ppl-2026-11-01.md`, sha256 `23f72985`, stamped before any run. It is not edited.

## É1. The first attempt stopped at the first grouped prefill, and the run restarts from a fixed image

Job `6ae71e5e0e0338d1bb7d705d`, image `0884bd31` (commit `1c54288`), ran 176 s, about $0.09. Gates 1 to 4 passed:
oracle MATCH, object and config at their sha256, the int4 parity line, and step 2's decoded line hashed to
`8784ffc4`. Step 3 printed its header with the token fingerprint `3f1baca9033bf251`, then stopped before its first
window line:

    matmul is only supported for contiguous tensors ... rstride: Layout { shape: [1, 8, 4096, 128],
    stride: [4194304, 128, 1024, 1] } mnk: (16384, 128, 4096)

The grouped arm's prefill path (`window_nll`, the uncached forward) goes through a fresh `Cat` cache. Its first append
returns the value projection as a transposed view, whose rows interleave the 8 key heads. The expanded form never
meets it, since `repeat_kv(..).contiguous()` lays it out again. candle's CUDA matmul refuses that layout; the CPU gemm
accepts it, which is why the Mac tests passed. The GQA A/B is not affected: its unexpanded arms ran on the `Grow`
store, which copies every append into its buffer.

The fix, `rows_contiguous` in `model.rs`, lays out again any `k` or `v` whose last two axes are not row-major, before
the unexpanded forms read it, and passes a `Grow` view untouched (`rows_contiguous_copies_only_what_a_matmul_refuses`,
two mutants killed). It changes no value, only a layout, and does not touch the expanded path. The image is rebuilt
from the commit that carries this file, and the job is relaunched as `RETRY=2`, which writes to
`gqa-ppl-4b-2026-11-01-r2/`. Attempt 1's raw outputs are kept in `docs/mesures/gqa-ppl-2026-11-01-brut/attempt1/`.
Attempt 1 plus the relaunch's estimate, $0.09 + $1.05, stay under the $2 cap.
