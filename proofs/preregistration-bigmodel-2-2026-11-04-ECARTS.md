# Deviations of proofs/preregistration-bigmodel-2-2026-11-04.md

## E1, written before the launch: the image is the integration image 142216fc, built from 02ec809

The run uses Space sha `142216fc72326a98b94316a7f6cd2e989f0efc50`, built from commit `02ec809` of branch
`integration/lot1`. That commit merges three session branches: this one (`claude/priceless-bouman-73205e`, tip
`78285a5`, which carries the stamp), the gestures branch (`claude/gifted-grothendieck-59977f`, tip `ccbb3c3`) and the
attention branch (`claude/quizzical-kowalevski-38c5fc`, tip `5255470`). All three fork from `030747b`. The prereg names
"one image built from the commit that carries this file": `02ec809` carries it unchanged, and it carries more.

What the image holds beyond `78285a5`, read with `git diff 78285a5 02ec809`:

- **The attention branch.** A third attention form, `AttnHeads::Kernel`, with its kernel `attn_dec.cu`, the launcher
  `llvq-cuda/src/attn.rs`, the replay `attn_ref.rs`, the bench `attnbench` (a new binary, in both lists of
  `ops/Dockerfile.cuda`), the trait `AttnKernel` in `device.rs`, `llvq-llm/src/attn.rs` and `attn_cuda.rs`, and in
  `model.rs` a `Block` branch taken only when the form is `Kernel`. The `Expanded` and `Grouped` dispatch is unchanged.
  `bin/fusedrun.rs` gains the context sweep's kernel arm and the `LLVQ_GQA_PPL` kernel mode.
- **The gestures branch.** `Qwen3::logits_last_native`, `Qwen3::generate_loop` and `TokenLoop` in `model.rs`,
  `gesture_cuda.rs`, `argmax_key.rs`, the kernels `tok_argmax.cu` and `gesture_spin.cu`, and the mode
  `LLVQ_GESTURE_AB` in `bin/fusedrun.rs`, all off by default.
- **Integration commits.** `e965ca6` lengthens two arrays of `bin/cuhcheck.rs` for `rotate_wide.cu` and `attn_dec.cu`.
  `81ff209` changes `configs/README.md` only.

What this run measures, and whether the merge touches it: no merged change does. Byte-identical between `78285a5` and
`02ec809` (*computed*, `git diff --quiet`): `llvq-cuda/kernels/rotate_wide.cu`, `llvq_rot.cuh`, `rotate.cu`,
`llvq-llm/kernels/tv_q4_wide.cu`, `tv_q4_h.cu`, `llvq-cuda/src/bin/rotbench.rs`, `wide.rs`, `gpu.rs`, `sha256.rs`,
`llvq-llm/src/fused_cuda.rs`, `fused.rs`, and both fixtures, `rot-hashes.txt` and `q4-wide-hashes.txt`. In
`bin/fusedrun.rs` the `LLVQ_INT4_WIDE_CHECK` branch is this branch's; its refusal list now names `LLVQ_GQA_PPL` and
`LLVQ_GESTURE_AB`, and it runs before both. `llvq-cuda/src/lib.rs` gains the attention modules beside `wide` and `sha256`, which it already
held. Step 4 loads the 4B at the bench door and decodes no token, so the attention changes in `model.rs` are not on its
path.

## E2, written before the launch: the run is one part of the lot job `lot1-4b`

The operator ordered every card measurement of 2026-11-04 launched together, at one cost (`ops/jobs/lot1.sh`). This
run is the second of four parts of one job on one l40sx1, after the gestures A/B, before the attention bench r2 and
its T8 control. Its job script is the one `ops/jobs/bigmodel-identity.sh` writes, unchanged (`EMIT=<file>`), run as its
own bash process; its output goes to `/out/bigmodel-identity-2026-11-04/` as the prereg's script names it.

- The prereg says the image is rebuilt "after the two runs that go first". It is rebuilt once, before all of them, and
  the three runs share the image and the job.
- Its ceiling is an in-job `timeout 18m` around that process, in place of the job timeout of 20 min, which the
  container's start no longer shares. The lot job's own timeout is 180 min.
- Its cost is its share of the lot job, about 6.5 min of card time beyond the start (*estimated*), so about $0.20.
- Gate 6 holds as written: the lot job sets no `LLVQ_*` variable in its environment, and the part's own check refuses
  one.
- Rule III reads K0 against its values of 2026-08-05 and 2026-08-17 per handle. The gestures A/B runs on the same card
  before it, so the card is warm; a K0 move beyond 15 % is read as rule III says, against the object, and this run's
  within-run ratios stand.
- The part's gates are unchanged and checked by its own script. A gate that fails voids this run only.
- Registry before the lot: $261.35 over 251 rows (*measured*, `docs/data/jobs.csv`), where the prereg read $260.89
  over 250 before the two runs it expected to go first.
