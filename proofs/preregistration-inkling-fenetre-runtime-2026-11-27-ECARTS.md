# Deviations from the prereg inkling-fenetre-runtime-2026-11-27

The prereg beside this file is timestamped and not edited. What the lot did differently, in order.

R1. The requests. The prereg names the long prompt of `inkling-mac-serve-2026-11-26` (8,791 tokens). The lot sends
the same static block through request `u1` of `ops/inkling/prefix_checks.py` on `/api/chat`, as the prefix journals
did, and not request `e1` of `serve_checks.py` on `/v1/chat/completions`. The reason: `u1` sets temperature 0 (`e1`
sets none) and `/api/chat` reports the cached count and the server's read time, which H-iv needs. The short prompt is
request `chat` of the same file (temperature 0, 48 tokens). Both prompts are rendered at 8,791 and 36 tokens by the
server's own count.

R2. The order of the commit. The code was committed (82ec459) before the live gates, so that the binary measured
names a commit of its own. The rule of the prereg was kept by a stop: had a gate failed before 06:15, the commit would
have been undone and the diff saved as a patch.

R3. The refusals. `narrow` on the CPU backend and on a Qwen3 of the store were run live (cheap: both refuse before a
weight is read). `narrow` on CUDA and on the candle stack on Metal are covered by a unit test only
(`narrow_turns_the_window_on_on_metal_fused_and_is_refused_elsewhere`); no CUDA run and no candle-stack local file
were made.

R4. A doc comment left stale. `llvq-llm/src/inkling/layers.rs`, `set_window_scores`, says the runtime never turns the
switch on. It now does, under `--attn-window narrow`. The lot's order forbids touching `llvq-llm`; the comment is left
for the main session.

R5. The mutant "the switch parsed and not applied" is placed on `window_scores_for`, the function that maps the parsed
arm and the load to the action. The two lines that carry its result (the call in `load_local`, the call to
`set_attn_window` in `main.rs`) have no unit test: a server loads a model to reach them. The live narrow servers
cover them, by their added load line and by their read time.
