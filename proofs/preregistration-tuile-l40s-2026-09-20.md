# Preregistration. The tile on the served path, on an L40S

**Written, committed and TIMESTAMPED on 2026-09-20, BEFORE the run.**
Operator go given 2026-09-20. Cost: about **$0.30** on l40sx1, timeout 1 h.

## 1. Why this run exists

`llvq-cuda/src/tile.rs` ships the mechanism and refuses the policy, in its own words: with
`LLVQ_TILE_BLOCKS` unset the answer is 128, "the value every published number was measured at",
because the two rows of `TILE_BY_SM` come from `bin/f1rankfloor`, **a synthetic bench with its
own shapes**. It then names the rule: "Promoting a synthetic optimum to a served default
without measuring it on the served path is the class of error this repository keeps catching.
F1d measures all three columns on the real path; the operator flips the default afterwards."

F1d did that sweep **on sm_120 only** (`f1d-2026-09-10`, 0.82x / 0.86x / 1.03x at 128 / 64 / 32).
The L40S has never been swept on the served path. Its synthetic row says 64 is best:

  R = (v3g - nullk)/(planes14 - nullk)   128      64       32
  sm_89, synthetic                      0.5784  0.4668   0.4704

So this run measures the three columns on the served kernel, on an L40S, with the real model's
`nblocks` per projection. It is the missing half of the decision, not the decision.

## 2. What runs

`planesbench` with the ball object and the fine-tuned Tetra object, arms
`planes14,nullk,tetra48`, once per tile in `128, 64, 32`, one process a tile, everything else
held. Same two files as `banc-tetra-2026-09-20`.

## 3. Signed prediction

Today's bench at tile 128 reads Tetra 4.107 ms, Planes14 5.135, nullk 2.340, so R = 0.6322.
Scaling by the synthetic ratio 0.4668/0.5784 = 0.807:

| quantity | point | interval |
|---|---|---|
| Tetra ms at tile 64 | **3.77** | [3.5, 4.1] |
| R at 64, against 0.6322 at 128 | **0.51** | [0.45, 0.58] |
| gain on the total | **+8.3 %** | [+0, +15 %] |
| best tile on sm_89 | **64** | 32 is the alternative |

The interval's floor admits **no gain at all**, because the synthetic bench and the served path
differ in exactly the way the module warns about: `f1rankfloor` has its own shapes, and the
served kernel walks a real `nblocks` per projection.

## 4. What refutes what

- 64 wins by 5 % or more: the synthetic row transports, and flipping the default is supported by
  a served-path measurement, which is what `resolve_with` asks for.
- The three tiles land within 2 % of each other: the sm_89 row does not transport, the served
  default stays 128, and the synthetic table keeps its warning label.
- 128 wins: the synthetic optimum is an artefact of `f1rankfloor`'s shapes and the module's own
  caution was right.
- Tetra moves and Planes14 does not: the effect is the decoder table's residency and not a
  global occupancy change, which is the mechanism tile.rs claims.
