# Upstream report: `broadcast_matmul` copies its rhs (candle)

Opened 2026-08-09:
[huggingface/candle#3871](https://github.com/huggingface/candle/issues/3871).

| file | what |
|---|---|
| [`ISSUE.md`](ISSUE.md) | **verbatim archive** of the posted body; do not edit it, its point is that it matches the issue |
| [`candle-broadcast-matmul.patch`](candle-broadcast-matmul.patch) | the fix plus its test, `git diff` against `main` @ `6f74e7c`. Editable copy |
| [`repro/`](repro) | the standalone reproducer, CPU, no GPU and no model. Editable copy |
| [`../../figures/broadcast-matmul-bug.svg`](../../figures/broadcast-matmul-bug.svg) | the mechanism diagram, two lanes |

The reproducer and the patch are **inlined in the issue body**: relative links do not
resolve from candle's tracker, and a report that depends on a link to a third-party
repository ages badly. They stay here as separate files because they are the ones we
edit; `ISSUE.md` is only a fingerprint.

## What is established, and how

1. **The defect exists** and it is on `main` (`6f74e7c`, 0.11.0-dev) as well as on the
   0.9.2 we use: `Tensor::broadcast_matmul`, arm `(false, true)`, materializes
   `rhs.broadcast_as(...).contiguous()`. For an output head
   `(1,1,2560) × (151936,2560)ᵀ` in f16, that is **778 MB copied per call**, and from a
   transposed view, so a strided *gather*, not a memcpy.
2. **Measured on both sides.** CPU (4 vCPU, `main`, release): 8,104 ms against 76.6 ms
   for the manual fold in f16; 23,663 against 151 in f32. GPU (L40S, our 08-07 job,
   [`mesures/phases-2026-08-07.txt`](../../mesures/phases-2026-08-07.txt)): head phase
   **26.7 ms/token** against 13.3 ms for the 36 blocks combined.
3. **The fix holds.** It folds the leading dims of the lhs into the rows, the trick
   `candle_nn::Linear::forward` already does, moved up into the primitive. On `main`:
   `cargo test -p candle-core --release` passes in full (`grad_tests` included),
   `cargo fmt --check` is clean, and the added test **dies** if the fold is mutated
   (`reshape((batch*m, k))` → `reshape((m, batch*k))`). After the fix,
   `broadcast_matmul` and the manual fold are the same code: 81.0 ms in f16.
4. **Unsought bonus: the fold is also more accurate in f16.** Against the f32 product of
   the same inputs, relative error **1.37e-2** for the broadcast path against
   **3.41e-4** for the fold, ×40. The batched path looks like it accumulates in a
   narrower type; mechanism not chased further.

## The attribution was fixed across the repository (2026-08-09)

When we went back to the source to write the issue, one claim in our publications did not hold:
**`candle_nn::Linear::forward` already avoids this path**, deliberately, with the comment "we avoid
using a broadcasted matmul as it is much slower". It folds the leading dims exactly as our fix does,
and `candle_transformers::models::qwen3::ModelForCausalLM::forward` applies its head through
`Linear`. So candle's qwen3 does not pay this copy.

The one that pays it is `Head::project`, **our** code,
`llvq-llm/src/model.rs:553`, `h.broadcast_matmul(&t.t()?)`. The dense arm of
`bin/fusedrun` loads through `sealed::load`, so that is indeed the path producing the
26.7 ms.

The number was right; the label was not. **The five formulations recorded here were
fixed** in the commit *"La copie de 778 Mo était la nôtre, et seize documents disaient
candle"*, plus eleven others found in a sweep: `README.md`, `docs/hf-model-card.md`,
`paper/main.tex`, `paper/sections/{integration,related,conclusion}.tex`, `CLAUDE.md` (two
places) and nine docs. The dated journal `verdicts-nuit-2026-08-07.md` got an erratum at
its head rather than a rewrite.

**What did not move**: no number. Not the ×2.03, not the ×1.12, not the 26.7 ms, not the
778 MB. The argument comes out **stronger**. The handicapped baseline is ours, so the ×2.03
no longer carries anything on its own, and the paper now says so explicitly instead of
leaving it implied.

**Still right and untouched**: `verdict-a2-repeat-kv-2026-08-06.md`,
`passation-lot-a-2026-08-06.md`, `rapport-lot-a-2026-08-06.md` and
`tableau-8b-2026-08-07.md`. They describe the behaviour of `broadcast_matmul` without
attributing it to candle's models.

## Still to be decided

1. **Fix `Head::project`** (`llvq-llm/src/model.rs:553`) with the same fold. One line,
   quality unchanged. It moves the tok/s of the **dense arm**, and with it the reference point of
   every published table. To be done, and said out loud: it is a new control, not a silent fix.
2. **The upstream PR.** The patch is ready and checked on `main` @ `6f74e7c`, the issue
   says "happy to send this as a PR"; all that is missing is a fork to push to.
