# Preregistration. tv_tetra48_seg on the card, once its fused stream comes from the Tetra file (2026-11-04)

**Written on 2026-11-04, committed and TIMESTAMPED BEFORE the run.** Operator go: given in chat on 2026-11-04 for the
first wave of the throughput validation roadmap, under a $10 HF cap for that wave. Cost: about **$0.60** on l40sx1
(*estimated*, §6); timeout 45 min, so **$1.35** at worst (*computed*). Spent before the wave: $248.84 over 237 rows of
`docs/data/jobs.csv` (*measured*), plus $3.19 of five billed jobs the registry misses (*computed*), so $252.03.

## 1. Why this run exists

`tv_tetra48_seg` computes several Tetra matrices of one layer (q+k+v, gate+up) in one launch. It has never passed a card
check. Its only one, on 2026-09-20, failed at row 0 of the first gate+up group: −0.3080684 fused against −0.57255346
separate (*measured*, quoted in `llvq-cuda/tests/tetra48_segment_matches_unfused.rs`). The cause is in the bench, not
proven in the kernel. `planesbench` built the fused Tetra stream from the ball file, its first argument
(`planesbench.rs`, `srcs`), and the separate Tetra arm read the Tetra file, its second. The check compared two
different matrices. Commit `c68ba42` builds the fused Tetra group from the Tetra file's members: words, centroids,
`gs_off`, row scales and tails. Two host tests build a group from two different sources and fail on the old
construction (5 mutants of 5 killed, *measured* on the Mac). The kernel itself is untested.

## 2. What runs

One job, l40sx1, the image rebuilt from the commit that carries this file (Space sha recorded by the launcher), script
`ops/jobs/seg-tetra.sh`. The two files of `banc-252-2026-10-06`: the ball reference
`ball-ref-2026-09-20/qwen3-4b-llvq.bin` (1,770,527,533 B) and the all-Tetra 4B `tetra-4b-2026-09-06/qwen3-4b-tetra.bin`
(1,770,529,149 B, sha256 `0adb7cfd…`). That run matched 252 of 252 matrices by name and fused 72 groups (*measured*,
`banc-252-2026-10-06-brut/banc.txt:153-158`). The all-Tetra file holds no int4 record, so every group has a Tetra
stream. One `planesbench` process, `LLVQ_SEG_TETRA=1`, `LLVQ_BENCH_ARMS=slot32,planes14,fp16,tetra48`.

## 3. Gates, each one voids the run

1. Both files at their bytes on the mount, the Tetra file at its sha256 prefix.
2. `tetra48: 252 of 252 matrices matched by name`.
3. The fusion section runs: `fusion: 72 groups` is printed.
4. The Slot32 and Planes14 groups are bit-exact, as on 2026-10-06. If a ball arm fails, the bench is broken, not the
   Tetra kernel.

## 4. Signed prediction

| quantity | point |
|---|---|
| Tetra groups checked | **72** |
| Tetra groups bit-exact against the unfused launches | **72 of 72** |

Reasoning, *estimated*. The fusion changes no arithmetic: the same blocks in the same order with the same centroids,
and the host tests show the stream is now the same stream block for block.

## 5. Decision rules

- **S1.** The bench prints `bit-exact against the unfused arms: 72 groups on Slot32 and Planes14, 72 on Tetra48`.
  `tv_tetra48_seg` is proven on sm_89 for the all-Tetra 4B. The q+k and gate+up fusion of the served object may be
  wired and measured under its own prereg, with the int4 `v_proj` group skipped.
- **S2.** The bench stops on `... / Tetra48: row r is a fused against b separate`. The kernel or the remaining
  bookkeeping is wrong. Nothing wires `tv_tetra48_seg`; the first differing group and row are reported, and the cause
  is sought on the host before any further card run.
- Otherwise, including any failed gate or a Tetra count other than 72, the run is void, and a retry goes under this
  prereg with its deviation beside it.

## 6. Cost

*Estimated* on l40sx1 at $1.80/h. `banc-252` spent 1,495 s proving ten arms block by block and 79 s transcoding the
fusion (*measured*, its log), 35 billed minutes in all (*measured*, `docs/data/jobs.csv`). Four arms here, so about
20 min, **$0.60**. Timeout 45 min, **$1.35**.

## 7. What this run does not establish

- Speed: the fused Tetra pair is timed only as the bench prints it, and no ratio is read here.
- The served object: its int4 `v_proj` changes which groups exist. A check on the served file is its own run.
- Any card but the L40S.

## 8. Deviations

Any departure goes in `proofs/preregistration-seg-tetra-2026-11-04.md-ECARTS.md`, beside this file.
