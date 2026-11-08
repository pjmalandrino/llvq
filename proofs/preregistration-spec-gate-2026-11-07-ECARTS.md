# Deviations from `preregistration-spec-gate-2026-11-07.md`

Written beside the stamped prereg, which is not edited. Journal: `docs/mesures/spec-gate-2026-11-07.txt`.

1. **First run void at the drafter's load (job 6aefa831, about 10 min, $0.30).** Image 8d00a878 from 88fded4. Gates
   1 and 2 held at the three sizes and each target loaded with its 4E v9 (card memory 1.980, 3.356 and 5.637 GB, the
   served doors' figures). The drafter's load then stopped before any token: the eager path's attention kernel
   (`attn_dec.cu`, the `kernel` form the drafter's config names) was compiled for 4 and 5 query heads per KV head,
   and the 0.6B has 2. No speculative arm ran; no number of it is reported. Raw output in
   `docs/mesures/spec-gate-2026-11-07-brut/` (run 1 at the top level).
2. **Fixes before the second run.** `attn_dec_g2.cu` adds the kernel's 2-group entries as a part of its own, compiled
   only for a model of 2 groups (`AttnModule::on_stream_for`): the served module's text and digest do not move. On the
   host the 2-group entries return the replay's bits (`attn_dec_matches_rust`), and the card checks them against the
   replay at every load, as it does the others.
3. **The drafter re-sealed.** The first run's install line also showed the drafter carrying its duplicate head table
   in f16 on the card (0.31 GB, never read). `seal` now leaves out `lm_head.weight` when the model ties its ends.
   The new file `qwen3-0p6b-sealed.bin` is 251,746,661 B, sha256 `bf924246…`, 3.099 b/param over the 596.0 M
   parameters the model reads, perplexity 40.3017 with the same scored tokens (`3f1baca9`): the same weights
   (*measured*, `docs/mesures/draft-0p6b-2026-11-07-brut/v2/`). The job reads it from `draft-0p6b-2026-11-07-v2/`
   and writes to `spec-gate-2026-11-07-r2/`.
4. **The Mac lock.** `flock` is not installed on the Mac; the drafter's perplexity ran under the same lock file
   through `fcntl.flock` (`lock.py`, the same system call).
