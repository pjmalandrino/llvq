# Prereg: the audits carried onto main change no byte the kernels compute, and the slim file set serves (2026-11-24)

## Object

Main after three audit sets are carried onto it: the code audit of 2026-11-18, pull request 23 (audit batches of
2026-11-08) and the two llvq-tetra audits. One image built from that tree, two jobs on one card.

## What is read

1. `ops/jobs/inkling-cuda-ttft.sh`, arms `chunk d2000 c2000`, a 2,000-token prompt. The sha256 of the logits of each
   arm against the value the same card gave before the audits (journal inkling-cuda-ttft-2026-11-24):
   - L40S: `chunk` and `d2000` 9c1624adfddbc2cab212b3a8ee4793ba88abd4d71994f2e390a92b7a3a9b014e, `c2000` bf90416e5ec021195177da93fbd3a8ad2df9ecd43edefa945ab43b5a182fa955
   - L4: `chunk` and `d2000` ad2e86472796351ff0643bbdee3b2b54a05998906eeb869429438c830621a709,
     `c2000` 60974683586440b110340a3b77f17818563449e123c209233425f891d5a5f05f
2. `INSTALL=image ops/jobs/inkling-install-card.sh`: the image's `llvq` pulls `inkling2:mini` at Hub revision
   697661e3c825a282a1d7ad4be758ef674d003e0b, whose `model.safetensors` no longer carries the two BF16 tables
   (8,145,313,108 bytes to download), and passes the server checks (a) to (f).

## Criterion

PASS of 1: the three digests equal the card's references. Any other digest is a FAIL: the audits moved a computed
value, and the measurements of the model are to be redone or the change found and removed.
PASS of 2: the pull ends, and every correctness check passes.

## Signed predictions

- 1 passes: probability 0.9. Read in the diffs: no kernel text file changes, the launch files gain guards and names.
- 2 passes: probability 0.75. Read in the code: the fused stack takes the embedding from the int4 table and the
  head from the int4 table, and never asks the checkpoint for `embed.weight` or `unembed.weight`. Not run.
- nvidia-smi after the load, if 2 is reached: within 100 MiB of 7,786 MiB (L4) or of the L40S figure of this
  morning. The tables were never on the card.

## Cost

Two jobs, about 15 minutes of card in all, $0.25 to $0.50, *estimated*. Card spend on this model before: $8.50 of $25.
