# Preregistration. 4E v2 on the card: its tokens and decode rate at 5 and 8,192 prompt tokens with the phase table, and the GB/s of the one-launch pass's variants

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-05 on branch `claude/serene-bohr-1edf69`, once the code it describes
existed. To be timestamped before the first measurement, then never edited; a departure goes to
`preregistration-pers-4e-v2-2026-11-05-ECARTS.md`.

Operator go: phase 1 of "4E v2" was granted at $0 (changes with negligible impact on the fundamental criteria). This
run is a card run: its launch is the operator's, through the integration session. Cost: about **$0.39** on l40sx1
(*estimated*, §8). Timeout 40 min: **$1.20** at worst (*computed* at $1.80/h). Spent before it: see
`docs/data/jobs.csv` at launch.

## 1. Why this run exists

The target is the sealed 4B at 300 tok/s or more after a short prompt on the L40S; llama.cpp IQ2_XXS reaches 312.85 on
this card (*measured*, 2026-08-30). No ratio is formed across stacks (rule 5). The third 4E reading (R3,
`docs/mesures/pers-4e-r3-2026-11-05.txt`) gave the same tokens as the eager path and no gain: x0.997 at 5 tokens,
x0.516 at 8,192. Its phase table named three costs:
- 4E's attention phase: 2,355 us a token at 5 tokens, 13,273 at 8,192 (*measured*);
- norm/RoPE/append serial on three CTAs: 220 us (*measured*);
- the rows' stream: 1.377 GB at 290 GB/s on the mean CTA's work (*measured*).

4E v2 changes all three (§2), each keeping every bit of the path it is compared with. This run asks the card, in
order: identity; 4E v2's rate and phase table in both geometries; and, on the one-launch pass L1 measured at 410 GB/s,
how many GB/s each of the stream changes is worth on its own.

It is a measurement. No rule of §7 adopts or serves anything.

## 2. What the code holds at the commit that carries this file

Off by default; no served path moves (the served unit's digest `6193094310bc…` is a gate).

- `llvq-cuda/kernels/llvq_pers.cuh` section 10, the attention phase, on `attn_dec.cu`'s mapping:
  - a job is one (split, KV head) on a whole CTA, jobs spread over CTAs;
  - each step stages 32 positions, K and V, once for the KV head's 4 query heads;
  - at 512 threads the next step's loads are in flight;
  - the task states live in shared memory, and the units fold there;
  - a split row's fold is done by the last of its KV head's jobs (a ticket), 8 splits a round trip;
  - no grid barrier inside the phase: a token passes 2 + 6 × 36 = 218 barriers, against 254 before.
- Section 11, norm/RoPE/append: head w on CTA w mod grid (48 heads on 48 SMs), and each lane's four values read once.
- The rows:
  - the 512-thread symbols keep the next Tetra block or int4 word in flight (q/k/v, o, gate/up, down, head; the same
    order of accumulation); the 768-thread symbols run without it, 24 warps a SM instead;
  - the decoder tables are staged once a launch, not every layer;
  - CTAs ask L2 for their rows of a later phase:
    - the q/k/v slice before ln1;
    - the o and gate/up slices during attention;
    - the down slice before the MLP's barrier.
- The threshold: the runtime refuses a launch below position 1,280 unless a measurement lowers it. Below 1,280 the
  served config runs candle's grouped arithmetic (cuBLAS), which 4E does not compute.
- `bin/fusedrun` `LLVQ_PERS_4E=64`:
  - at 8,192 tokens, the served config itself: the kernel from 1,280, every decode row past it, inside the T8;
  - at 5 tokens, the measurement form of the earlier readings: the kernel on every row, eager and 4E, no T8;
  - both geometries (512 and 768 threads), each where its token symbol loads without local memory, interleaved
    with eager in every round;
  - one stamped launch a geometry, outside the timed rounds.
- `llvq-cuda/kernels/tetra48_persall.cu`: L1's `tv_tetra48_persall` byte for byte, and four variants: `_pf` (rows'
  prefetch), `_ts` (tables in shared memory), `_ts_pf`, `_ts_pf_l2` (plus the L2 prefetch of the CTA's next group).
  `planesbench` `LLVQ_TETRA_OCC=1` checks each against A4 bit for bit and times it in the same rounds.

Host checks at $0 (`docs/mesures/pers-4e-v2-host-2026-11-05.txt`):
- 27 tests of the kernel text against the served text;
- every 4E variant's token loop equal to the served chain, positions past 256 (the ticket fold) included;
- attention equal to `attn_ref.rs` at 8 lengths up to 2,049 and on grids of 1 to 5 CTAs of 2 to 16 warps;
- the one-launch variants equal to `tv_tetra48_seg`;
- 12 new mutants caught.

Under the devel ptxas 12.4 (indicative):
- 512 threads: 120 registers, 0 B local;
- 768 threads: 80 registers, 0 B local;
- the 768 stamp symbol: 24 B stack.

Never run on a card.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/pers-4e-v2.sh`:
- the sealed 4B, `sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`;
- the two banc-252 files, `ball-ref-2026-09-20`, and `tetra-4b-2026-09-06` with sha256 `0adb7cfd…`;
- each file checked by size, then sha256.

1. **Oracle** on Qwen3-0.6B (rule 10).
2. `LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E=64 fusedrun <sealed> 64`:
   - 64 tokens a run, at the 5-token prompt and at 8,192 tokens;
   - 2 discarded and 5 timed rounds of eager, 4E 512 and 4E 768;
   - then one stamped launch a geometry.
3. `LLVQ_CONFIG=<kernel-from1280 config> fusedrun <sealed> 16`: the served door's NVRTC digest.
4. `LLVQ_TILE_BLOCKS=64 LLVQ_BENCH_ARMS=slot32,planes14,fp16,tetra48 LLVQ_SEG_TETRA=1 LLVQ_TETRA_OCC=1
   LLVQ_DUMP_CUBIN=… planesbench <ball> <tetra>`: L1's bench, without the microbenchmarks, with the four variants.

## 4. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. The three files are at their bytes and sha256; the config is the committed one.
3. `cudarc event tracking: off (one stream, ordered by issue)`, and `4E: RoPE forms equal to candle's rope on this
   card` with at least one form.
4. Identity. `4E at 512 threads: tokens equal the eager path's` is printed at both contexts, and no round moves a
   token. The 8,192 context prints `the served config: … inside the T8`.
5. The served door prints `attention kernel from position 1280 on grow(256)` and the NVRTC digest
   `6193094310bc…`.
6. `planesbench`: `tetra48: 252 of 252 matrices matched by name`, tile 64.
7. No `LLVQ_*` variable in the job environment.

Per geometry and per variant, voiding only its own numbers:
- a 768-thread 4E whose tokens differ, or that does not load without local memory (reported, not timed);
- a one-launch variant that moves a row or spills (`INVALIDATED` or `SPILLS`, not timed);
- stamps whose launch moved its tokens.

## 5. Outcomes reported, never gated

- Each 4E symbol's registers and local bytes, from the loaded module (the driver's ptxas), and the cubins of the
  persistent module (`cubin/`).
- Per context: eager, 4E 512 and 4E 768 decode tok/s, median and range; 4E / eager round by round (*measured*, same
  stack).
- The phase table of each geometry at each context: per phase, the wall, the slowest and mean CTA's work, and the CTA
  most often slowest.
- The rows' stream rate: 1.377 GB over the mean CTA's work summed over q/k/v, o, MLP and head (*computed* from the
  table).
- The one-launch pass: each variant's ms and B_pa (0.9806 GB / t), and its speed over `tpersall` round by round.

## 6. Signed predictions

*Estimated* by `ops/pers_4e_v2_estimate.py` from R3's measured table: the attention, norm and rows terms of §2, with
ranges, nothing fitted.

| quantity | point | interval |
|---|---|---|
| 4E 512 tokens against eager, both contexts | **identical** | |
| 4E 768 loads without local memory (driver ptxas) | **yes** | |
| attention wall, 5 tokens, us a token | **250** | [150, 600] |
| attention wall, 8,192, us a token | **2,100** | [1,500, 3,500] |
| norm/RoPE/append wall, us a token | **70** | [40, 120] |
| rows' stream on the mean CTA's work, 4E 512, GB/s | **380** | [290, 480] |
| 4E 512 decode, 5 tokens, tok/s | **240** | [180, 290] |
| 4E 512 decode, 8,192, tok/s | **165** | [125, 200] |
| eager, served config, 8,192, tok/s | **104** | [100, 108] |
| 4E 768 against 4E 512, 5 tokens | **1.0** | [0.85, 1.15] |
| tpersall, GB/s | **410** | [400, 420] |
| tpersall_pf over tpersall | **1.05** | [0.95, 1.20] |
| tpersall_ts over tpersall | **1.20** | [1.00, 1.45] |
| tpersall_ts_pf over tpersall | **1.30** | [1.00, 1.60] |
| tpersall_ts_pf_l2 over tpersall_ts_pf | **1.00** | [0.95, 1.10] |

Reasoning:
- The attention phase loses R3's per-position serial walk (about 1.5 us a position on 32 warps). What remains at a
  short context is up to 8 steps of one round trip a step. At 8,192 it is the KV cache stream, 33.8 MB a layer.
- The rows' rate is the widest term: the prefetch doubles each lane's bytes in flight, but whether the rows are bound
  by latency alone is what the table tells (cause 2 of E2 held at 290 GB/s).
- 300 tok/s needs the rows at about 490 GB/s on the mean CTA's work, with the middle attention and norm (*computed*
  by the script). It is predicted not to be met by this version.

## 7. Decision rules

- **A, identity.** If 4E 512's tokens are identical at both contexts, v2 keeps the bits on this card. Otherwise the
  run is void, and the first divergence is found on the host with the card's dumps.
- **B, attention.** Attention is fixed if its wall is 600 us or less at 5 tokens and 3,000 us or less at 8,192.
  Otherwise the phase's own stamps (slowest against mean CTA, the CTA most often slowest) name the next host test.
  Possible causes: the steps, the ticket fold, the staging rounds.
- **C, norm.** norm/RoPE/append at 120 us or less is fixed.
- **D, rows.** Read the rows' stream rate of the faster geometry:
  - at 490 GB/s or more: the rows reach the target's need, and the token's remaining gap to 300 is the fixed walls;
  - from 350 to 490: the next lever is named by the one-launch variants (rule E);
  - under 350: the prefetch and the 24 warps did not move the stream, and the next step is the card's SASS of the 4E
    row loops (`cubin/`), at $0.
- **E, the one-launch variants.**
  - Tables in shared at 1.15 or more over `tpersall` (whole range) confirm the L1/shared pipe model
    (`pers-sass-2026-11-05.txt`): 4E already holds its tables in shared, so its remaining limit is latency or
    issue.
  - The prefetch's speed over its base (`_pf` over `tpersall`, `_ts_pf` over `_ts`) at 1.10 or more says latency
    binds there.
  - `_l2` at 1.05 or more over `_ts_pf` says L2 prefetching pays on a pass with no fixed walls.
  - Each reading is a GB/s on the same rounds as A4, never a ratio across stacks.
- **F, the target.** 4E at 300 tok/s or more over its whole range at 5 tokens is reported as the target reached in
  the measurement form only. Serving it would need the operator's license of the kernel form below 1,280 (a T8
  extension, read by the paired perplexity at those positions): 4E does not compute the grouped arithmetic the served
  config runs there.
- **G.** A miss is reported as a miss, with the term of §6 it refutes. No threshold moves after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h:

| step | time |
|---|---|
| container and image | 3 min |
| oracle | 1 to 2 min |
| fusedrun, R3's 170 s with one more geometry in every round and two stamped launches a context | about 3.5 min |
| the served door | 1 min |
| planesbench, L1's 270 s, four more one-launch variants, no microbenchmarks | about 4.5 min |

About 13 min, **$0.39**. Timeout 40 min: **$1.20** at worst.

## 9. What this run does not establish

- Serving 4E: below 1,280 it needs a license 4E's arithmetic does not have; past it, the A2 reading
  (persistent.md §7) and the operator.
- 4E at 8B (a Tetra o, not in this version) or 14B (its config differs; the code accepts the shape, untested here).
- The 768-thread geometry's rate where the driver's ptxas spills it.
- The int4 rows' rate on the one-launch pass, which carries Tetra rows only.
