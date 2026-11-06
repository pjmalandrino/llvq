# Deviations of proofs/preregistration-attn-bench-r2-2026-11-04.md

## E1, written before the launch: the image is the integration image 142216fc, built from 02ec809

The run uses Space sha `142216fc72326a98b94316a7f6cd2e989f0efc50`, built from commit `02ec809` of branch
`integration/lot1`. That commit merges three session branches: this one (`claude/quizzical-kowalevski-38c5fc`, tip
`5255470`), the gestures branch (`claude/gifted-grothendieck-59977f`, tip `ccbb3c3`) and the 32B wall branch
(`claude/priceless-bouman-73205e`, tip `78285a5`). All three fork from `030747b`. The prereg asks for "one image built
from the commit that carries this file": `02ec809` carries it unchanged, and it carries more.

What the image holds beyond the stamp commit `d8fb74e`, read with `git diff d8fb74e 02ec809`:

- **This branch, after the stamp (`5255470`).** `LLVQ_GQA_PPL=<ctx>,<windows>,<first>,kernel`, a measurement mode in
  `bin/fusedrun.rs`; edits stay inside the `LLVQ_GQA_PPL` code. The 27 lines of `model.rs` are a test.
- **The gestures branch.** `Qwen3::logits_last_native`, `Qwen3::generate_loop` and `TokenLoop` in `model.rs`,
  `gesture_cuda.rs`, `argmax_key.rs`, the kernels `tok_argmax.cu` and `gesture_spin.cu`, and the mode
  `LLVQ_GESTURE_AB` in `bin/fusedrun.rs`. All of it is off by default: `generate` and every published path run as
  before (`preregistration-gestures-2026-11-04.md` §2).
- **The 32B wall branch.** Two new kernel units, `rotate_wide.cu` and `tv_q4_wide.cu`; `llvq_cuda::wide` and
  `llvq_cuda::sha256`; `rotbench`; and in `fused_cuda.rs` and `fused.rs` the size of each int4 entry point, the opt-in
  of `tv_q4_hs` and the refusal of an f16-staged record fed f32. Its `model.rs` change is a test. `llvq_rot.cuh`,
  `rotate.cu` and `tv_q4_h.cu` are byte-identical, so the served unit's text does not move
  (`preregistration-bigmodel-2-2026-11-04.md` §2), and a test pins the served 4B, 8B and 14B on today's kernels.
- **Integration commits.** `e965ca6` lengthens two arrays of `bin/cuhcheck.rs` for the two new units. `81ff209`
  changes `configs/README.md` only. `configs/qwen3-14b-tetra-e4-grouped.json` is new and unread by this run.

What this run measures, and whether the merge touches it:

- **Step 2, `attnbench`.** `llvq-cuda/kernels/attn_dec.cu`, `llvq-cuda/src/attn.rs`, `attn_ref.rs` and
  `bin/attnbench.rs` are byte-identical between `d8fb74e` and `02ec809`. `attn_dec.cu` reads sha256
  `c7b0d8d3f2e3066ba1a2b4331c0bb6c92f22182c78a31b7fb74e839330a1838d` at both commits (*computed*), the sha256 of the
  fixed-cost revision `c8a01d3`. Gate 2 compares the image's printed sha256 with the file at launch, so it reads the
  same text the stamp named.
- **Step 3, the served sweep.** `fusedrun` under `LLVQ_CONFIG` goes through `model.rs` and `fused_cuda.rs`, which the
  merge changed. The attention path is this branch's own: `llvq-llm/src/attn.rs`, `attn_cuda.rs`, `kvq.rs`,
  `served.rs` and `device.rs` are byte-identical to `d8fb74e`. The int4 load path is the 32B branch's, held to today's
  kernels by its test. The gestures code is off by default. So no merged change is meant to move a token or a time of
  step 3, and two lines of the same job check it before this run: the gestures part's gates 4 and 5 (bench door and
  served door, decoded text byte for byte against the published digests) run first on the same card.

## E2, written before the launch: the run is one part of the lot job `lot1-4b`

The operator ordered every card measurement of 2026-11-04 launched together, at one cost (`ops/jobs/lot1.sh`). This
run is the third of four parts of one job on one l40sx1, after the gestures A/B and the 32B wall check and before the
attention T8 control. Its job script is the one `ops/jobs/attn-bench-r2.sh` writes, unchanged (`EMIT=<file>`), run as
its own bash process; its output goes to `/out/attn-bench-r2-2026-11-04/` as the prereg's script names it.

- Its ceiling is an in-job `timeout 31m` around that process, in place of the job timeout of 33 min, which the
  container's start no longer shares. The lot job's own timeout is 180 min.
- Its cost is its share of the lot job, about 11 min of card time beyond the start (*estimated*), so about $0.33; the
  container start and the image pull are paid once for the four parts.
- The first two parts ran on the same card before it: the card is warm, which the first run's was not. The grouped
  arm's comparison with the first run (prediction "same, each within ±2 %") reads that difference too.
- The part's gates are unchanged and checked by its own script. A gate that fails voids this run only. The T8 control
  that follows runs only if this part exits 0.

Rule 3 of §6 is read as written, and the lead's stop line is reported to the operator. The T8 control's own prereg
(`preregistration-attn-t8-2026-11-04.md`) reads the same two sweep logs for its condition; that reading starts no
served change and decides nothing of this run.
