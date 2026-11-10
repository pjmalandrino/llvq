# Deviations from prereg N10 (`preregistration-verify-tree-n10-2026-11-10.md`)

The prereg is not edited. Each deviation is written here before the measurement it affects.

## Deviation 1: the measured code moves from afd23b7 to cb626c2

Written on 2026-11-10 at about 15:10 CET, before any measurement. The image built from the stamped commit 43f54fd
(sources afd23b7) failed to compile on Linux: Space `Pier-Jean/llvq-runner-cuda` at sha b67eeb5e, stage BUILD_ERROR at
15:00:51 CET, `cargo build --release --locked -p llvq-llm --features cuda ...` exit 101 on

    error[E0425]: cannot find value `cap` in this scope
        --> llvq-llm/src/persist4e_cuda.rs:1268:86

(`cap` had moved inside the per-arm loop; the install's print after the loop still named it). The Linux half of
`persist4e_cuda.rs` cannot be type-checked on the Mac. Nothing was launched, nothing billed.

The measured code moves to cb626c2 ("Tree verify: the 512-thread slice cap computed where the install prints it"),
which adds two lines to `llvq-llm/src/persist4e_cuda.rs`: a comment and `let cap = pers::p4t_stream_cap(area, nw);`
before that print. It changes only a host-side print; no kernel file, no module text, no arithmetic. The image is
published from a clean detached checkout of cb626c2. Predictions and decision rule unchanged.
