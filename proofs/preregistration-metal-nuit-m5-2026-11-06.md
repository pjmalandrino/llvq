# Preregistration. M5: the grouped attention form against the expanded one on Metal, the 12-window design

Written on 2026-11-06 at about 01:30 on branch `claude/metal-nuit-3c4257`, after M1 to M4 of the night
(`docs/mesures/metal-nuit-2026-11-06.txt`). To be timestamped before its first measurement, then never edited.
Cost **$0**, local, under the operator's go for this night ("la machine est libre"). No one can answer during the run:
every choice below is fixed here.

## 1. Why

`served.rs` refuses the grouped form on Metal by name and names the measurement that would lift the refusal: a
perplexity on Metal. M3 read bit identity on 4 windows of 4,096 (16,380 decode steps, max |dlogit| 0). The L40S
design that licensed the form on the card is 12 windows of 4,096 (`gqa-ppl-2026-11-01`). This run is that design on
Metal, by the decode path. M4 read the form at x1.75 at 2,048 tokens and x2.38 at 4,096.

## 2. What runs

Binary `metalrun` sha256 `a05c8ba7c16a3603fc564fce8438508b63339d9a92a0c8b4ccffcb978c0600fc` (the M4 binary), by path,
nice 10, quiet machine (no build from 15 minutes before the start to the end):

`metalrun ppl <sealed> configs/qwen3-4b-tetra-e4.json lutd 4096 12`

The first 12 windows of 4,096 of the WikiText-2 test split, each fed one true token at a time through two caches of
one model in lockstep, expanded (Cat) and grouped (Grow 256), `Qwen3::decode_nll_paired`. The first 4 windows are M3's
and must read again as M3 read them.

## 3. Reading and rule

Reported: each window's two perplexities, the pooled ratio, the per-window range, top-1 agreement, the largest logit
difference.

- **Bit identity**: max |dlogit| 0 over every step of the 12 windows. The plan records the form as bit for bit on
  Metal at 12 windows, and a NOT SERVED Metal config naming it is prepared for the operator.
- **Inside the band**: not bit for bit, pooled ratio and every window within ±0.12 %. The plan records it as a
  tolerance event inside the L40S band; the config is prepared, labelled so.
- **Outside the band**: the plan records the form as refused on Metal; no config.
- **Otherwise** (windows 1 to 4 differ from M3, a crash): journaled, nothing prepared.

In every case the refusal in `served.rs` stays: lifting it is the operator's decision.

## 4. Prediction

Bit identity on all 12 windows (M3: 4 of 4). Wall time 50 to 60 minutes (M3: 255 s a window).
