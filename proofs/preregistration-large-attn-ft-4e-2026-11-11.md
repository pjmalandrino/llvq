# Prereg: 4E v9ft, the attention tile's full-tile path in the decode, sealed 4B

Written on 2026-11-11 on branch `claude/arbre-prefill-2026-11-11`, from commit e66e2e0, once the code it describes
existed. Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-large-attn-ft-4e-2026-11-11-ECARTS.md`.

Go: the operator, 2026-11-11 evening ("ok pour ton job": the two jobs of the full attention tile), card spending
inside a **$25** cap; **$12.66** spent before these two jobs (relayed by the orchestrating session). Cost: about
**$0.55** on l40sx1 (*estimated*, "Cost"), timeout 25 min, **$0.75** at worst. With the prefill job
(`preregistration-large-prefill-ft-2026-11-11.md`, $0.90 at worst): **$14.31** at worst after both.

Measured code: the commit that holds this file. The image is published from a clean detached checkout of the commit
that holds this file's stamp and the prefill prereg's; one image serves both jobs. The build and its Mac gates:
`docs/mesures/large-attn-tuile-pleine-mac-2026-11-11.txt`. The mechanism and the reference: E1,
`docs/mesures/large-4e-e1-2026-11-11.txt` (job 6af46f1d).

## The arm

`v9ft` (`pers.rs::p4e_e1_edits`, through E1's mechanism): E1's v9 reference (the served v9 row loop with the logits
symbols) and the full-tile edit (`pers.rs::pers_ft_edit`). `llvq-cuda/kernels/pers_attn_full.cuh` is inserted before
`pers_attn_tile`, and at its head `if (nv == PERS_TT) { pers_attn_tile_full(...); return; }`. On a full tile of 8
positions the body has no guard between positions; partial tiles keep the guarded loop. Each score, the tile max,
each ex2, ts and each PV sum keep their operands and their order. Nothing else moves: 512 threads, the row loop, the
KV store. b/param of the file (2.73) does not move.

Module texts at the card's tile define: E1's v9 reference d55a097b, v9ft 258,955 bytes 22ff8320. The served v9
module (1b422b62) keeps its bytes (pinned by `pers::tests::the_d1_tree_modules_keep_their_bytes` and
`the_ft_modules_keep_their_bytes`).

## Mac gates (before the stamp)

- SASS, devel ptxas 12.4 sm_89: one warp's in-order issue of a tile call drops from 2,936 to 1,548 cycles on 4E
  (*estimated*, assumed latencies); v9ft adds about 30 local operations a thread in the row phases (spills).
- Host twin (`pers_matches_served.rs`): `the_4e_e1_arms_are_the_served_chain` runs v9ft; its token loop equals the
  served chain on both row variants, each RoPE form, from 5 (tiles of 6, 7 and 8 positions) and from 270 (33 full
  tiles, then 7, 8 and 1, across the split at 256), on two consecutive launches and on the stamp symbol.
  `the_4e_v9ft_mutants_are_killed`: two mutants, each killed.
- fusedrun's `LLVQ_PERS_4E` takes the prompt lengths as a third field (commit e66e2e0); its parser test passes and a
  mutant on the ascending check is killed. Clippy at zero new warnings on the Mac. fusedrun with `cuda` is compiled
  by the image build.

## Setup

One image, one job, `ops/jobs/large-attn-ft-4e.sh` in its default form, on l40sx1 only. Output
`large-attn-ft-4e-2026-11-11/` in `hf://buckets/Pier-Jean/jobs-artifacts/`. Inputs: the sealed 4B (sha256 886391a8),
`configs/qwen3-4b-tetra-e4-4e-v9.json`. Each call is its own process:

0. The served v9 config at the door, `LLVQ_CONFIG=<v9> LLVQ_CONTEXTS=5 fusedrun <sealed> 64`, under the 100 ms
   nvidia-smi sampler: control 2 and the served short-context peak.
1. `LLVQ_PERS_4E=64:v9,v9ft:1024,4096,8192 fusedrun <sealed> 64` (the served flags on the command line, as E1): both
   modules in one process, at a 5-token prompt and at 1,024, 4,096 and 8,192. At each context: the 64 tokens of each
   arm against the eager path; 2 discarded + 5 timed interleaved rounds (eager, v9, v9ft each round), medians and
   ratios round by round; one stamped launch an arm (phase walls); one logits launch an arm (64 positions x 151,936
   binary16 logits of v9ft against v9's, bit for bit). Below position 1,280 (5 and 1,024) both sides run the kernel
   attention on every row (measurement form, no T8); at 4,096 and 8,192 the served config's T8.
2. `LLVQ_PERS_4E=64:<arm>:1024,4096,8192 fusedrun <sealed> 64`, one process an arm (v9, v9ft), each under the
   sampler: the peak VRAM of each arm, and its logits fingerprints for a cross-process comparison.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file and the config at their sha256; no `LLVQ_*` in the job environment.
2. The served door ran the served 4E v9 module (252,757 bytes, sha256 1b422b62).
3. The process of step 1 loaded E1's v9 reference (d55a097b) and v9ft (22ff8320).
4. If the image fails to build, nothing is launched. If v9ft fails to build or to load, the arm is void and the
   journal says so.

## Signed prediction

The operator's session's signed predictions, verbatim (sealed 4B, l40sx1, measurement form, v9ft against v9, 64 tokens):

1. v9ft logits equal v9's bit for bit at every context; tokens 64 of 64.
2. v9 within 1.5 % of job large-4e-e1's 236.6 tok/s at 5 and 161.0 at 8,192.
3. v9ft over v9 at 8,192, round by round: x1.055 [1.01, 1.10], that is 170 tok/s [163, 177]. The ceiling at full memory bandwidth is about 178 tok/s (computed in large-prefill-attn-mma-etude-2026-11-11).
4. At 5: x0.995 [0.97, 1.005] (the new spills outside attention).
5. At 4,096: x1.04 [1.00, 1.08]. At 1,024: x1.01 [0.98, 1.03].
6. Peak VRAM of v9ft within 0.02 GB of v9's.

Rule: PASSES if the logits are bit for bit at every context AND v9ft is at least x1.05 at 8,192 AND at least x0.99 at 5. If it passes, v9ft is a candidate only: nothing is served by this job.

Operational reading, fixed here: "every context" is 5, 1,024, 4,096 and 8,192 in the process of step 1. "v9ft over
v9, round by round" is the median over the 5 timed rounds of v9ft's tok/s over v9's in the same round, the ratio
line fusedrun prints for v9ft against v9. v9's tok/s for prediction 2 is its median in the process of step 1; against
E1's figures it is a cross-job comparison and is labelled so. "Peak VRAM" is the nvidia-smi peak of each one-arm
process of step 2.

## Decision rule

| result | outcome |
|---|---|
| logits bit for bit at the four contexts, v9ft / v9 at least x1.05 at 8,192 and at least x0.99 at 5 | PASSES: v9ft is a candidate, nothing served |
| logits bit for bit, either bound missed | FAILS for speed; the phase stamps say where the time went |
| logits differ at any context, or tokens move | VOID, the arm is out |

## Known limits

- The ptxas of the card's driver can differ from the devel image's; the loader prints the card's registers and
  spills, and v9ft's spills may grow there.
- The latencies behind the estimates (2,936 to 1,548 cycles a tile call) are assumed, not measured.
- v9ft has never been timed on a card.
- 1,024 is below the T8 threshold (1,280): it runs the measurement form, not the served config, as the 5-token
  prompt does.
- Not established: the served door under v9ft (no config names it); 8B, 14B; any card but the L40S.

## Cost

E1 (job 6af46f1d) ran 1,096 s, 19 min billed, for 7 module loads in 5 processes at two contexts (*measured*). This job
loads 5 modules in 4 processes: about 14 min at two contexts. The two added contexts, from E1's sampler walls (a
one-arm process 163 s, 70 s of it the load; the three-arm one 457 s, 351 s of it the load), cost about 55 s a
process: about 3 min for the three processes. About 17 to 18 min, **$0.55** at $1.80 an hour (*estimated*). Timeout
25 min: **$0.75** at worst. Spent before both jobs $12.66; this job $0.75 and the prefill job $0.90 at worst:
**$14.31** of $25 at worst after both.
