# Throughput leads nobody owns, 2026-11-05

Short list from [revue-debit-2026-11-05](revue-debit-2026-11-05.md) §3.2, where each lead has its effort, bits and
card cost. Gains are *estimated* for the 4B against 134.7 tok/s short and 104.0 at 8,192. None is started.

| lead | short | 8k | first step |
|---|---|---|---|
| `nsys` timeline of the served door | none | none | add `nsys` to the image, one job, about $0.15 |
| prefill GEMM route | none | first token from 35 s to about 3 s | T8, 4-7 days of code |
| attention kernel at FlashInfer's η (0.545 to about 0.8) | none | +10 | attnbench in situ, $0.3-0.5 |
| kernel attention form at 8B and 14B | none | 8B to about 88, 14B to about 58 | T8 per size, $1.5-2.5 |
| `tv_q4_hh` timed alone, tiled `tv_q4_ht` at 14B | unknown | unknown | bench arm, $0.1-0.2 |
| progressive head read, certified argmax | +3 | +2 | bench arm, $0.2; worth more once 4E lands |
| serve the gestures (measured ×1.012) | +1.6 | +1.0 | serving decision only |
| `attend_writing` from position 1,280 | none | +1.6 | 1 day, rides any job |
| `tv_tetra48_seg` served: f16 twin, q+k and gate+up | +10 | +6 | 3-5 days, identity proof $0.3-0.5 |
| fast single-CTA rotation R1 | +11 | +6 | V0c, $0.24-0.30 |
| port 4E v2's prefetch row form to the eager kernels | +12 | +7 | after 4E v2, 2-3 days |
| per-shape `LLVQ_TILE_BLOCKS` | 0-5 % | same | sweep, $0.2 |

The eager items stop helping near 244 tok/s, where the 4.11 ms host enqueue binds
([lot1](../mesures/lot1-4b-2026-11-04.txt)).
