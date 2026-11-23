# Lot 2, agent D, step 3: the resident compressed experts in the model (interim, 11:05)

Gate 3a is RED as written, and gate 3c is RED by 0.04 point. Gate 3b reports the generation: 32 of 32 greedy tokens
equal the reconstruction arm. Every one of the 44 routing flips sits at a margin smaller than the score change the
f16 path makes. Code in F, nothing committed. Every number is *measured* unless labelled.

## What exists

- The seam: `layers::MoeSource`, the one call at the MoE site (`experts.moe(x, routing, cfg)`). Every dense
  `ExpertSource` implements it through `experts_forward`, unchanged; `resident::ResidentExperts` implements it
  through `resident::experts_forward`. The layer code is otherwise unchanged.
- The plan: `Inkling` holds one `LayerExperts` per layer, `Native` (decoded lazily, the default) or
  `Source(Box<dyn MoeSource>)`. `Inkling::make_resident(layers_dir, rt)` makes resident every layer whose
  `Lxx/VERIFIED` exists. `Inkling::use_reconstructions(layers_dir, layers)` puts the same layers on `HatExperts`.
- `resident.rs`: G's file, plus `impl MoeSource` and `verified_layers`. The tests are in `tests/inkling_resident.rs`:
  G's two tests, then `gate3_*`.
- Gates 1 and 2 on CPU through the new seam: still green. Gate 1: logits rel 1.9e-6, argmax 32/32, routing
  identical (`gate1_cpu_step3.log`). Gate 2: 32/32 tokens, 1.04 s/token (`gate2_cpu_step3.log`).

## Resident layers

0, 1, 2 and 6 to 17: 15 layers, every one whose `VERIFIED` file existed at 10:10. The others (3, 4, 5, 18 to 23) run
their native experts, decoded on the CPU one expert at a time and uploaded for the call. Of the 48 visits, 39 are
resident.

## Gate 3a: resident (Metal) against reconstruction (CPU), 32-token prompt, RED

Command: `inkling_resident_step3 --ignored --exact gate3_resident_model_matches_reconstruction` under `heavy.sh`.
Log `gate3a.log`, journal `gate3a.json`. Tolerance, fixed before the run: 2e-2 relative (max abs over max |ref|) on
every visit's hidden state and on the logits.

- Logits: max abs 0.743 against max |ref| 28.8, rel 2.58e-2 > 2e-2. RED.
- Hidden states: 11 visits over the bound (12, 22, 24, 25, 26, 28, 30, 35, 38, 39, 40). The worst is visit 30
  (layer 6, pass 3) at 4.04e-2.
- Routing: 44 flips over the 39 resident visits.
- Argmax equal at 32 of 32 positions.
- The bound was an estimate that summed G's per-call error over 39 visits and ignored routing flips, which each move
  one token's whole MoE output.

## Gate 3b: 32 greedy tokens of the resident arm through `CausalLm`

Command: `inkling_resident_step3 --ignored --exact gate3_resident_model_generates`. Log `gate3b.log`, journal
`gate3b.json`.

- Resident against reconstruction arm: 32 of 32 equal.
- Resident against native (`generate_native.json`): 13 of 32 equal, first difference at position 13. Expected for a
  quantized student.
- Text: " role in the 1997 film Sense and Sensibility .\n\nHe was born in 1956 in Birmingham , England . He was
  educated at"
- Memory, Metal `current_allocated_size`:

  | point | allocated | difference |
  |---|---|---|
  | before the load | 0.5 MB | |
  | after the dense weights | 5.09 GB | attention, embedding norm, head in BF16, routers |
  | after the experts | 9.02 GB | +3.92 GB; 3.87 GB uploaded as codes, rotations and side data, 15 layers |
  | at the end of the generation | 9.43 GB | |

  The native experts of the 9 other layers are never resident: one expert at a time is uploaded per call, then
  freed.
- Time:
  - Experts load: 10.3 s, 15 layers.
  - Prefill, 32 tokens: 3.30 s, 34,330 lattice launches.
  - Decode: 0.518 s/token, 1,560 lattice launches a token.
  - The reconstruction arm on CPU, for scale: prefill 16.9 s, 2.05 s/token.

## Flip analysis (gate 3a rerun with the scores, `gate3a.json`; the first run is kept as `gate3a_v1.json`)

The rerun gives the same numbers as the first run: logits rel 2.58e-2, 44 flips, argmax 32/32.

For each flip, `margin` is the 8th minus the 9th `score + bias` in the reconstruction arm. `change` is the largest
change of that token's `score + bias` over the 128 experts between the two arms at that visit.

| quantity | min | median | max |
|---|---|---|---|
| margin | 2.9e-6 | 2.8e-3 | 1.9e-2 |
| change | 4.3e-5 | 2.7e-2 | 5.9e-2 |
| margin / change | 0.012 | 0.149 | 0.886 |

- All 44 flips have margin under change: every one is explained as noise. None is at a wide margin.
- Five sit above a ratio of 0.5: visits 12, 21, 22, 24 and 41.
- The flips concentrate on the prompt's last tokens: 28 to 31 account for 39 of the 44. The other five are at tokens
  14 and 15. Errors compound along the sequence through attention.
- The first flip is at visit 12 (layer 12, pass 1, token 28, margin 3.5e-5); it is where the hidden error jumps from
  about 1e-4 to 2.4e-2.

Hidden-state error per visit, on the tokens whose routing is identical in both arms at every visit so far:

| visits | rel, all tokens | rel, never-rerouted tokens |
|---|---|---|
| 0 to 11 (no flip yet) | 4.7e-5 to 1.2e-4 | the same, 32 tokens |
| 12 to 17 | 9.3e-3 to 2.4e-2 | 2.0e-3 to 4.3e-3 (29 to 31 tokens) |
| 18 to 41 (passes 2, 3) | 1.2e-2 to 4.0e-2 | 4.0e-4 to 1.3e-3 (26 to 27 tokens) |
| 42 to 47 (layers 18 to 23, native) | 3.5e-3 to 8.4e-3 | 3.5e-4 to 8.2e-4 |

- The structural error of the kernel path stays at about 1e-4 per visit while no token has flipped. That matches G's
  per-call measurement.
- From visit 12, a token that has never flipped still carries 2e-3 to 4e-3 in pass 1. My suspicion, not yet shown:
  these are tokens 29 to 31, which read the flipped token 28 through attention. A per-token breakdown is in the next
  run.

## Decode breakdown (`gate3_decode_breakdown.json`, 8 steps after the 32-token prompt, every MoE call timed)

| part | median s/token | per call |
|---|---|---|
| total (timed run) | 0.526 | |
| resident MoE, 39 calls | 0.072 | 1.85 ms |
| native MoE, 9 calls (layers 3, 4, 5, 18 to 23, decoded on the CPU and uploaded) | 0.228 | 25 ms |
| the rest (attention, convolutions, norms, routing, head) | 0.225 | |

- 1,560 lattice launches a token.
- Tonight's 24 resident layers (*estimated*): the 9 native calls become resident at 1.85 ms each. That gives
  0.072 + 0.017 + 0.225, about 0.31 s/token, with the rest as the next target. The rest likely includes the head,
  which widens the 200,058 BF16 rows to f32 chunk by chunk at every step (not measured separately).

## Gate 3c (stated before its run, coordinator decision of 10:25)

The first 2,048 tokens of WikiText-2 test window 0 (`dump_ref.py window`, ids in `wikitext_w0_2048.json`, the
windows of `data.eval_windows`), same 15 layers resident. Paired next-token NLL, resident (Metal) against
reconstruction (CPU). PASS when the perplexity ratio is within +-0.12 % (`docs/ETAT.md` §4, "Attention tolerance").

Reported beside it:
- the 95 % interval of the paired mean over positions (one window, so the positions are the replicates; indicative,
  since they are correlated);
- mean and max KL(reconstruction || resident);
- top-1 agreement.

Choice of 2,048 rather than 512 (*estimated*): the 32-token reconstruction prefill took 15.4 s, mostly reading the
reconstructions (about 46 GB over 39 visits), which does not grow with the prompt. The expert matmuls for 2,048
tokens come to about 13 TFLOP in f32, a few minutes on this CPU. So the reconstruction arm should take well under
10 minutes. Rerun command for tonight, all 24 layers resident once they are verified:
`INKLING_GATE3C_TOKENS=2048 F/target/release/inkling_resident_final --ignored --nocapture --exact gate3c_resident_perplexity_matches_reconstruction`
under `heavy.sh`.

## Gate 3c, measured: RED

Command: `inkling_resident_step3b --ignored --nocapture --exact gate3c_resident_perplexity_matches_reconstruction`
under `heavy.sh`. Log `gate3c.log`, journal `gate3c.json`. 2,048 tokens, 2,047 paired positions.

- Perplexity: resident 5.1479, reconstruction 5.1561.
- Ratio -0.158 %, 95 % interval over positions [-0.310, -0.006]. The band is +-0.12 %: FAIL.
- The resident arm is the better of the two, but the band is two-sided.
- KL(reconstruction || resident): mean 4.58e-4, max 1.49e-2 per position. That is about 1 % of the 0.04 the student
  shows against the native teacher on the loop layers (*computed*, different evaluations: an indication).
- Top-1 agreement: 2,028 of 2,047.
- Time: resident arm 64.0 s on Metal, reconstruction arm 166.4 s on CPU.

## The neighbour-codes mutant (layer 7 resident on layer 8's codes)

Binary `inkling_resident_mut_neighbour2`; tag `mut_neighbour` on the journals. Logs `gate3a_mut_neighbour.log` and
`gate3c_mut_neighbour.log`. Both gates RED, far above the noise of the clean path:

| quantity | clean | mutant | ratio |
|---|---|---|---|
| gate 3a logits rel | 2.58e-2 | 1.42e-1 | 5.5x |
| gate 3a visit 7 rel | 7.2e-5 | 2.78e-1 | 3,800x |
| gate 3a worst visit | 4.0e-2 | 2.8e-1 | 6.9x |
| gate 3a median visit | 1.35e-2 | 7.9e-2 | 5.8x |
| gate 3a routing flips | 44 | 828 | 19x |
| gate 3a argmax equal | 32 / 32 | 28 / 32 | |
| gate 3c ppl ratio | -0.158 % | +2.270 % [+1.08, +3.48] | 14x |
| gate 3c KL mean | 4.6e-4 | 2.5e-2 | 55x |
| gate 3c top-1 agreement | 2,028 / 2,047 | 1,923 / 2,047 | |

The cleanest separation is the hidden state at the mutated visit itself: 7.2e-5 against 0.278.

## Rows kernel in prefill (`gate3_rows.json`, `prefill_rows = 8`, `tv_tetra48_metal_lutdr`)

Same 15 resident layers; a fresh runtime put on the rows tuning by `MetalRuntime::set_tuning`. No kernel or served
default changed.

| prompt | served (1 row a launch) | rows kernel (8 rows a launch) |
|---|---|---|
| 32 tokens, median of 3 | 2.87 s, 34,330 lattice launches | 2.80 s, 12,285 launches |
| 2,048 tokens, 1 round | 56.1 s, 1,926,000 launches | 28.0 s, 255,000 launches |

- The rows kernel halves the 2,048-token prefill and cuts its launches by 7.6x.
- At 32 tokens it changes almost nothing: the prefill there is not launch-bound.
- It is not bit for bit with the served matvec: it narrows each output row to f16. Logits against the served
  tuning: max abs 0.64 of 28.8, argmax 31 of 32.
- It was not run through gates 3a and 3c, so it is not proposed as served. Whether to adopt it is the operator's
  decision.
- Decode is untouched in the served tuning. Under the rows tuning a one-row step would also take the f16 rows
  kernel; it was not measured.

## Final batch (final source, 11:15+, `batch6.log`): layer 3 became resident in between

Layer 3's `VERIFIED` appeared after the 15-layer runs. The final binary therefore ran 16 resident layers: 0 to 3 and
6 to 17, so 42 of the 48 visits.

- Unit tests (inkling, Metal tiny included): 8 passed.
- Gate 1, CPU: logits rel 1.9e-6, argmax 32/32.
- Gate 2, CPU: 32/32 tokens, 1.03 s/token.
- Gate 3a, 16 layers: GREEN.
  - Logits rel 9.6e-5, argmax 32/32, 0 routing flips.
  - Worst visit 2.3e-4: the per-call error of the kernel path, as G measured it.
  - Log `gate3a_v3.log`, journal `gate3a.json`. The 15-layer journal is kept as `gate3a_v2.json`.

The red of gate 3a on 15 layers stands as recorded. With one more layer resident, the trajectory no longer meets the
near-tie of visit 12, token 28 (margin 3.5e-5), and no flip occurs. My reading, a strong suggestion but not a proof:
the 44 flips of the 15-layer run were one near-tie cascading through attention and the loop. The kernel path's own
error is about 1e-4 per visit. Gate 3a's verdict therefore depends on whether the prompt meets a near-tie. Gate 3c,
on 2,048 tokens, was only run on 15 layers; tonight's 24-layer rerun is its next reading.

## Pending

- Gate 3c on 16 or more layers.
