# Deviations from the M1 prereg of 2026-09-30

The prereg is `proofs/preregistration-hf-metal-m1-2026-09-30.md`, sha256 `37d69af37538d7b9`,
timestamped before the first matvec and never edited. The journal is
`docs/mesures/hf-metal-m1-4b-2026-09-30.txt`.

Three departures, and one finding that is larger than the lot.

## 1. The device reference arm was not run, because it killed the machine

The plan for the gate was three arms: dense on the CPU, dense on the device, fused on the device,
so that the kernel's effect could be told from the device's. The dense arm on MPS at f32 needs
16 GB of weights plus the transient copy of a `.to("mps")`; the host fell to 66 MB of free pages and
the process was killed. The operator had asked the machine not to be saturated and it was, by me.

It turned out not to be needed: the fused arm runs on the GPU, the reference on the CPU, and the
256 ids are identical. So the device did not change the tokens and there is nothing left for a third
arm to separate. Had they differed, that arm would have been owed, at f16 to fit.

## 2. One prediction missed, and one half of another not measured

§6 predicts resident memory in [2.8 ; 3.4] GB. Measured: 5.436 GB. The interval was computed with
the int4 records and the embedding at **f16** and the arm was run at **f32**, where those two weigh
3.1 and 1.56 GB. The component the lot is actually about landed where it was predicted: 0.749 GB
measured against 0.74 computed.

§6 also predicts that the per-row residue stays under 1e-2 relative "and the tail in f16 dominates
it rather than the f32 row scales". The bound held, 2.61e-5. The attribution was not measured, and
the journal says so rather than repeating it.

## 3. Control 3 is satisfied, and it should not be trusted again

§5 control 3 asks for one mutant at least, "the tail dropped, the row scale dropped, the rotation
skipped. Each must change the tokens." All three were run and all three change the tokens at the full
gate, so the control passes as written.

It passes badly. **The tail mutant survives eight ids on all four prompts, and survives all 64 on two
of them.** It is not a void mutant: with the tail zeroed the per-row error against the dense
reconstruction goes from 2.61e-5 to 8.79e-2 on `k_proj` and from 8.79e-6 to 3.53e-2 on `down_proj`,
so the kernel reads the tail and dropping it costs 8.79 % and 3.53 % of the output. The gate caught
it at token 13 on the best of the four prompts.

The consequence reaches back over stages 1 and 2 as well, which both took token identity as their
gate: **four easy prompts and 64 greedy ids cannot see a 3 to 9 % per-row error.** The per-row check
finds the same defect in 20 seconds with a 3,400-fold margin, and it was ranked in this prereg as
"beside the gate, and not a gate".

Proposed, not decided, since it changes what a gate is in this plan: the per-row check against the
dense reconstruction becomes a gate of its own, on one matrix of each shape, and token identity stays
beside it as the end-to-end check it is good at. That is an operator decision and it is written in
`docs/plan-transformers.md` as an open one.
