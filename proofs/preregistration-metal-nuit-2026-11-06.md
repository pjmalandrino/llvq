# Preregistration. The sealed 4B fused on Metal: the wall lifted, then the CUDA row loops ported, timed on the Mac

Written on 2026-11-05 at night on branch `claude/metal-nuit-3c4257`, once the code it describes existed (commit
`aebe662`, binaries `metalrun` sha256 `bc2f9c80…` and `oracle` sha256 `e036eabf…` built from it). To be timestamped before the first measurement, then never edited; a
departure goes to `preregistration-metal-nuit-2026-11-06-ECARTS.md`.

Cost: **$0**. Local runs on the operator's M3 Max, covered by the operator's go for local measurements this night.
No HF job, no download: the sealed 4B is on disk (`~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin`, 1,418,224,685 B,
sha256 `886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8`, checked 2026-11-05).

## 1. Why this run exists

No sealed file had run fused on Metal: the loader refused the 12 int4 `down_proj` records of the 4B, whose d_in of
9,728 stages 38,912 B against the 32,768 B of threadgroup memory (`docs/archive/debit-vague5-2026-11-02/mac.md` §1).
The only way the sealed 4B decoded on the Mac was the dense reconstruction through candle, 7.5-9.6 tok/s on the
paper-1 file, low-grade *measured* (commit messages).

The throughput work of 2026-10-29 to 2026-11-05 took the served 4B on the L40S from 114 to 178 tok/s through the
door. Metal had none of it. This run measures, on the Mac:
- what the wall lift gives against the dense arm (the only way the file ran here before);
- what the port of the CUDA row loops gives against the served Metal kernels, each one mechanism, each bit for bit.

## 2. What the code holds

- `fused_metal.rs`: an int4 record whose `d_in * 4` exceeds 32,768 routes to `tv_q4_metal_tiled` at 2,048 columns a
  tile (`Q4_TILE_COLS`); the others keep `tv_q4_metal`. `tv_q4_h.metal` is not edited.
- `kernels/llvq_tetra48_rows.metal`, compiled after the served unit's text as a library of its own:
  - `_lutp`: the served `_lut` loop with block j + 32's words loaded before block j decodes (4E v4's `pers_tetra_row`
    PREFETCH);
  - `_lutd`: the activation read from device memory, no staging, no tile, no barrier;
  - `_lutdp`: both;
  - `_lutd2`, `_lutd4`: `_lutd` with 2 or 4 rows a simdgroup, each row keeping 32 lanes and the served lane order, so
    a lane carries 2 or 4 independent chains and reads the activation once for them.
- `kernels/tv_q4_pf.metal`, on the same rule: `tv_q4_metal_p` and `tv_q4_metal_tiled_p`, word wi + 32 and its scale
  and bias loaded while word wi is used (4E's `pers_q4_row<PREFETCH>`).
- `MetalTuning` names them; the served door passes `None` and is unchanged.
- `bin/metalrun`: arms of one file in one process (§3).

Gates already green at $0 before this file (`cargo test --release`, fast math on as shipped):
- `llvq-metal/tests/q4_tiled_matches_served.rs`, 8 tests: tiled = served where both run (tiles 256, 2,048, 8,192);
  tiled = host lane-order reference at 9,728, 12,288 and 17,408; fast math on = off; 16 repetitions at 320
  threadgroups; both prefetching twins = served.
- `llvq-llm/tests/q4_metal_candle.rs`, 2 tests: the binding at six widths, routing, offset activation, refusals by
  name, and the four int4 pipelines compiled.
- `llvq-llm/tests/tetra48_metal_rows.rs`: each Tetra twin = served = CPU oracle at d_in 2,560 and 9,728, the loop
  edges, tiles 32 and 256, refusals, and a multi-row twin's refusal of a d_out it cannot tile. The multi-row twins
  were added after the first gate run; they enter this run only if this file passes on them and `metalrun gate`
  gives them 253 of 253 before the stamp (§4 gate 3 then applies to them like the others).
- `metalrun gate` on the sealed 4B: every twin arm, 253 of 253 records (252 projections and the head) equal bit for bit
  to the served arm on the served arm's own decode-step inputs.
- Mutants: 27 planted, 27 killed (6 in `tv_q4_metal_tiled`, 5 in the int4 twins, 10 in the Tetra twins, 6 in the
  host), each by a site count of 1 in its kernel's region; the barrier removals among them. Journaled in
  `docs/mesures/metal-nuit-2026-11-06.txt`.
- `metalrun gate` with every arm of §3 step 2 but D: 253 of 253 for each, after the multi-row twins were added.

## 3. What runs

All built once by `nice -n 10 cargo build --release -j 4 -p llvq-llm --features metal,fast-linalg --bin metalrun
--bin oracle`, then run by path under `nice -n 10`, never through `cargo run`. Each binary's sha256 is printed by the
binary or recorded in the journal.

0. **Quiet machine.** No build, no other agent session's build, at least 15 minutes after the last build. Load
   samples (other processes' %CPU, GPU utilization, `pmset -g therm`) printed by `metalrun` at the start, after every
   arm and at the end.
1. **Oracle on Metal** (rule 10): `oracle Qwen/Qwen3-4B 64 metal` on the cached checkpoint. Must print its match line.
2. **M1, the arms**: `metalrun time <sealed> configs/qwen3-4b-tetra-e4.json
   lut,lutp,lutd,lutdp,lutd2,lutd4,lut+q4p,lutdp+q4p,lut+t8192,lut+grp,D 256 7`.
   - Config: the expanded e4 config, the only one the Metal door accepts.
   - First arm `lut` is the reference: the served Metal kernels with the wall lifted. Call it **F0**.
   - The bit gate first: every fused arm against F0 on 253 records.
   - Then 2 discarded + 7 timed rounds, the arms in the order above every round, each a `generate_timed` of 256 tokens
     after the 5-token prompt "The capital of France is", prefill in chunks of 256.
   - Per arm: door rate (256 over prompt plus decode) and decode rate (255 over decode), median and range over rounds;
     each arm over F0, round by round; first token divergence against F0; the token digest of F0.
   - `lut+grp` is the grouped attention form, a tolerance event refused on the served Metal door: a speed reading
     only. `D` is the dense reconstruction through candle, f16.
3. **Phased, off protocol**: `metalrun phased <sealed> <config> lut 64`, then the same for the best twin. Fenced
   phases (embed, blocks, head, argmax). Never a rate.
4. **Dispatch, off protocol**: `metalrun dispatch`: dependent one-element candle ops, then 1,024 dependent
   `[1, 2560]` f16 adds, per-op time. It prices a launch through candle on this Mac, which decides whether a Metal
   analog of 4E (fewer launches) is worth building.
5. **M2, confirmation**: `metalrun time <sealed> configs/qwen3-4b-tetra-e4.json D,lut,<after> 256 7`, where `<after>`
   is the arm §5 retains. This is the command the morning's test repeats.
6. **M3, the grouped form's tolerance on Metal, a reading only, last and optional**: `metalrun ppl <sealed> <config>
   <after> 4096 4`. Four WikiText-2 test windows of 4,096 tokens, each fed one true token at a time through two caches
   of the same model in lockstep, expanded and grouped (`Qwen3::decode_nll_paired`, the decode path of the L40S's
   `gqa-ppl-2026-11-01`). Prints each arm's perplexity, the grouped over expanded perplexity ratio per window, top-1
   agreement and the largest logit difference. The prefill path is not scored: Metal prefills 256 rows a call. Runs
   only if the night leaves an hour of quiet machine after M2.

## 4. Gates

Each voids the run, or the arm it names:

1. The binaries build; `oracle` prints its match line on Metal.
2. `metalrun` prints the sealed file's sha256 as above.
3. The bit gate: 253 of 253 for every fused arm but `lut+grp`'s attention, which the gate does not cover (it compares
   projections and the head, which the form does not touch). A fused arm under 253 is dropped and journaled.
4. Every arm returns the same tokens every round (the binary refuses otherwise).
5. Every fused arm but `lut+grp` returns F0's 256 tokens. An arm that does not is dropped, whatever its speed.

## 5. Decision rule

On M1's decode rate, each arm over F0, round by round, 7 ratios:

- **Retained**: gates 3 to 5 hold and the range of the ratio lies wholly above 1.000.
- **Wash**: gates hold and the range straddles 1.000. Not retained, journaled with its median.
- **Refused**: the range lies wholly below 1.000.
- `lut+t8192` moves no bit and is not a mechanism of the after arm: it is retained as the int4 tile only if its range
  lies wholly above 1.000, otherwise 2,048 stays.
- M3 decides nothing: its reading goes to the operator with U5 of `mac.md`, whose band is ±0.12 % on a design of
  12 windows. Four windows are a first reading, not that design.
- `lut+grp` is never retained here: the form is a tolerance event, refused on the Metal door. Its reading informs the
  operator, who alone may lift that refusal after a paired perplexity on Metal (U5 of `mac.md`).
- **The after arm** is the Tetra twin with the highest median among the retained ones, with `+q4p` if `lut+q4p` is
  retained, with the int4 tile if it is retained. If `lutdp+q4p` is retained and its median is the highest of all,
  it is the after arm. If nothing is retained, the after arm is F0 itself and M2 shows the wall lift alone.
- M2 confirms: the after arm's 256 tokens equal F0's and D's comparison is printed. If the after arm's ratio range in
  M2 does not lie wholly above 1.000, the journal says so and the morning command still shows it, labelled.
- **Otherwise** (a gate fails on F0 itself, the oracle fails, the machine is not quiet by §3 step 0, a hang): stopped,
  journaled, and the operator decides.

## 6. Predictions, signed, before any measurement

All *estimated*. Decode rates at 256 tokens after 5 on this M3 Max, expanded attention.

| arm | prediction | why |
|---|---|---|
| F0 (`lut`) | 45-61 tok/s | `mac.md` §3.2: Tetra at 74-77 GB/s, int4 at 75-250, the q4 head at 150-250, rotations, a 2.4-3.6 ms rest |
| D | 6-10 tok/s; F0/D x4-9 | the dense arm read 7.5-9.6 on the paper-1 file, and copies its f16 head table every token |
| `lutp` / F0 | x1.00 to x1.10, sign + | the words are one of three dependent loads a block; the collapsed suffix chain read x1.004 on this Mac (`5f439df`), so latency inside a block may not bind |
| `lutd` / F0 | x0.95 to x1.15, sign unknown | no barrier and no staging, but 8 rows of a threadgroup now read the activation through the cache rather than from threadgroup memory |
| `lutdp` / F0 | x1.00 to x1.20, sign + | both |
| `lutd2` / F0 | x1.00 to x1.25, sign + | two chains a thread at half the simdgroups; `_ilp`'s four rows a simdgroup gave +5 % in situ |
| `lutd4` / F0 | x0.95 to x1.25, sign unknown | four chains a thread, a quarter of the simdgroups: occupancy may fall at d_out 1,024 |
| `lut+q4p` / F0 | x1.00 to x1.05, sign + | int4 is 30 % of the 4B's weight bytes a token |
| `lut+t8192` / F0 | x0.97 to x1.03 | residency, `mac.md` §3.1 |
| `lut+grp` / F0 | x0.95 to x1.05 | under 5 % or negative at a short context, `mac.md` §3.3 (b1); tokens may diverge |
| dispatch | 3 to 15 us an op | not measured on this Mac through candle before; `31a5acb` quotes "about 2 us of launch" |
| M3 grouped / expanded perplexity | within ±0.05 % on each window | the L40S read +0.0095 % [-0.0048 ; +0.0238] on 12 windows |

A prediction outside its interval is reported as missed, beside the reading.

## 7. Labels and rules

- Every rate is *measured* on this Mac; GB/s of weights is *computed* (weights a token times tok/s, weights a token
  = projections plus the head table, 1.377 GB for the fused arms).
- Rule 4: F0/D is the raw ratio. The same-head ratio is not formed this night: no f16-head fused arm exists on Metal
  yet, and the dense arm's head copies its table every token (`mac.md` §3.2). The journal names the head copy beside
  every F0/D.
- Rule 5: no ratio against the L40S, vLLM, llama.cpp or torch. A Mac number never stands beside a card number
  without both machines and both attention forms named.
- Rule 6: memory in b/param for the whole model, tables included: 2.74 by the loader's host count (1.38 GB over
  4,022,468,096 parameters, *computed*).
- Rule 7: medians with ranges formed round by round.
- No served config is changed. A Metal measurement config, if one is written, says NOT SERVED in its note.

## 8. Journal

`docs/mesures/metal-nuit-2026-11-06.txt`: the raw output of every step above, the binaries' sha256, the mutation run,
and the verdict per arm under §5.
