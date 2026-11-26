# Prereg: the decode of Inkling 2 Mini on the Mac, bit-for-bit steps (2026-11-26)

Written before the first timed decode run of this lot. Not edited after its timestamp; deviations go in `-ECARTS.md`
beside it.

## Question

On this Mac (M3 Max, 64 GB, Metal) the served stack decodes 26.66 tok/s after a 32-token prompt, median of 3 runs
[24.50, 27.99], and 22.73 tok/s after a 9,000-token prompt, median of 4 runs [22.45, 23.56] (*measured*,
`docs/mesures/inkling-mac-serve-2026-11-26.txt` and `docs/mesures/inkling-mac-prompt-2026-11-26.txt`). Three changes
of class A, which keep every logit bit, may shorten a step: the command-buffer cadence of candle (plan item K2), the
two rotations of the fused MoE call staged in threadgroup memory, and `down` of the fused MoE call spread over one
simdgroup an expert. How much of a step does each remove?

## Object and method

Branch `debit/inkling-mac-2026-11-26`. Release build, `--features metal`, of `inkrun` (crate `llvq-llm`). Every run
goes through `~/.llvq-guard/heavy.sh`, one at a time. Files: the text checkpoint, the 24 verified layers, the int4
table `unembed.q4g64.llvqraw`, the 32-token prompt `lot4-coord/tokens.json`.

Stack of every timed arm: `--glue fused --attn fused+scores --head int4 --head-table <table> --require-layers 24
--device metal --n 129`, so 128 timed decode steps a run. The figure of a run is the median of its 128 step times.

Machine. Load at one minute and GPU utilisation are written before each run (`<label>.machine`). A run starts only
with load under 6, GPU under 10 % and the guard's lock free; a run that started outside those bounds is redone, not
kept.

Timing. The spread of one run is about +-7 %, so one run proves nothing. A step is timed in at least 5 rounds, each a
reference run (the current default) then a variant run, alternated. The gain of a round is the reference's median step
over the variant's, minus one. The journal gives every round, the median of the five gains and their range.

1. K2, no code. `CANDLE_METAL_COMPUTE_PER_BUFFER` at the default (50), then 4 to 6 values on each side, one screening
   run each. The best value, and any value within 2 % of it, then goes through the 5 rounds against the default.
2. Rotations of the fused MoE call in threadgroup memory: a `moe_rot_metal` twin whose Walsh-Hadamard stages run in
   threadgroup memory (n = 2,304 and 1,152, 9.2 KB at most). First a kernel-only bench on synthetic experts at the
   model's shapes against the served kernel. The wiring and the 5 rounds follow only if the kernel bench gains.
3. `down` of the fused MoE call with one simdgroup an expert: one threadgroup an output row, the selected experts'
   products in threadgroup memory, summed by one lane in ascending slot order from +0 with the two roundings of the
   served kernel. Same order as step 2: kernel bench first.

Each new kernel has a twin test against the kernel it replaces, every f32 equal, and at least one mutant seen red with
its sites counted. `cargo clippy --all-targets` with `metal`: zero warnings.

## Decision rule

A step is class A when its 128 tokens equal the reference run's and its logits file has the reference run's sha256.
Otherwise it is withdrawn, whatever its time.

A class-A step is kept when the median of its 5 round gains exceeds 1 % and at least 4 of the 5 rounds go the same
way. A kept step becomes the default on Metal. If a default changes, the server reading `ops/inkling/serve-mac.sh`
(13 checks) is run at the end of the lot, and `ops/check-linux-cuda.sh clippy` if code shared with CUDA is touched.

Stop rules. The sentinel's critical alert, a load or memory alert, or free disk under 60 GiB stops everything.

## Signed predictions (the coordinating session, Claude Fable 5.2, 2026-11-26; copied as given)

"Mes prédictions signées (elles ont été trop hautes deux fois, je les signe plus bas) :
- D1, K2 : gain de 0 à 1,5 ms par jeton ; probabilité 0,4 qu'il dépasse 0,5 ms.
- D2, rotations : 0 à 1,0 ms ; probabilité 0,5 qu'il dépasse 0,3 ms.
- D3, `down` : 0,3 à 1,5 ms ; probabilité 0,5 qu'il dépasse 0,5 ms.
- D4, fin de lot : 27 à 29,5 tok/s après 32 jetons ; probabilité 0,5 d'atteindre 28.
- D5 : aucun changement du pic mémoire (8,03 Go après chargement, à ±0,05)."

D1 to D3 are per decode step. D5 reads `memory.after_resident` of the final default run.

## Already refuted, not rerun

Wider reads, per-section indexing, tables in threadgroup memory (`lutdt`), `lutd2/4/p`, f16 outputs, removing small
launches, two rows a simdgroup, one kernel a visit, a persistent kernel.

## Cost

$0. Local only, on the operator's Mac, which stays in use.
