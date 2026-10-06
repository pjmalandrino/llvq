# Preregistration. The ten-arm bench with Tetra on all 252 matrices

**Written, committed and TIMESTAMPED on 2026-10-06, BEFORE the run.**
Operator go given 2026-10-06 ("252 matrices everywhere"). Cost: about **$1.05** on l40sx1,
*estimated* from the two runs it repeats (27 and 8 billed minutes on 2026-09-20), timeout 1 h 30,
so $2.70 at worst. Campaign cap $9, of which $0.50 is spent. Measured code: the pinned image
`97a2b62a6d0c8911dcd8e4d26a40461f0196ccc9`, built from commit `5333ac8`. Between that commit and
this one the kernel sources differ by comments only.

## 1. Why this run exists

Every arm of the published kernel table times 252 matrices except `Tetra`, which times 216. The
`Tetra` file the table used, `dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin`, carries its 36 `v_proj`
as int4 g128, and the `tetra48` arm skips them. So the `Tetra` row is not comparable to the
others, and the paper carries the gap as a caveat in two captions and one sentence.

A bare `Tetra` file of the same model exists, with 252 lattice records and no int4:
`tetra-4b-2026-09-06/qwen3-4b-tetra.bin`, 1,770,529,149 B, sha256 `0adb7cfd02ed7402...`,
2.1498 kernel b/weight (*measured*, `references-comptabilite-2026-09-18`). Timing the existing
`tetra48` arm on it gives every arm the same 252 matrices. No code changes.

Rule 9 first. F1d timed this exact file on an L40S on 2026-09-10: 4.355 ms at tile 128 and
3.717 ms at tile 64 (*measured*, `f1d-2026-09-10`). It cannot replace this run: it is another
process, an older image, and it has no AWQ arm, so its rows cannot sit in the published table.
It is used below as the prediction.

## 2. What runs

Two parts, each one process per configuration, as published.

**The ten-arm table**, at the served tile (unset `LLVQ_TILE_BLOCKS`, 64 on sm_89):

  phase 1   slot32,planes14,planes12x,golay70v1,fp16,awq,golay70v2,cublasf16,nullk
  phase 2   the same, plus **tetra48**

This is `ops/jobs/banc-tetra.sh` with one change: the second file.

**The tile sweep**, `fp16,planes14,nullk,tetra48`, one process at each of 128, 64 and 32. This is
`ops/jobs/tuile-l40s.sh` with the same one change.

The first file is unchanged: `ball-ref-2026-09-20/qwen3-4b-llvq.bin`, 1,770,527,533 B.

QTIP stays out, for the reason `preregistration-banc-tetra-2026-09-20.md` section 2 gives. Its
row keeps coming from the 2026-08-21 process, as the paper already says.

## 3. Gates, each one voids the run

1. Both files at their byte counts on the mount, and the `Tetra` file's sha256 prefix `0adb7cfd`.
2. `tetra48: 252 of 252 matrices matched by name`, in all four processes.
3. The ten-arm header reads `tile 64`.
4. The worst error of every arm under its threshold, as `planesbench` checks before timing.
5. No `LLVQ_*` variable in the job environment. The script sets `LLVQ_BENCH_ARMS` and
   `LLVQ_TILE_BLOCKS` on each command line, never in the environment.

## 4. Signed prediction

| quantity | point | interval |
|---|---|---|
| `Tetra` median ms, ten-arm table, tile 64 | **3.72** | [3.60, 3.85] |
| `Tetra` b/weight | **2.150** | [2.149, 2.151] |
| `Tetra` GB read a pass | **0.98** | [0.97, 0.99] |
| `Tetra` time over AWQ's, medians | **1.14** | [1.10, 1.18] |
| sweep, `Tetra` at 128 / 64 / 32, ms | **4.36 / 3.72 / 3.86** | [4.20, 4.50] / [3.60, 3.85] / [3.70, 4.00] |
| `Tetra` range over the sweep, largest over smallest | **17 %** | [12, 22] |
| FP16, AWQ, `Planes14`, nullk medians | the published ones | within 2 % each |

The points at 128 and 64 are F1d's. The point at 32 adds the 3.8 % that tile 32 cost over 64 on
the 216-matrix sweep. The AWQ ratio uses the published 3.261 ms.

## 5. What is published, whatever the numbers

The ten-arm table of this process replaces `docs/data/echelle-formats.csv` whole, and the sweep
replaces the numbers of the paper's tile paragraph. No row from the 2026-09-20 runs survives
next to a row from this one. The `Tetra` row then reads 252 matrices, and the paper loses its
216 caveats.

## 6. What refutes what

- `Tetra` lands inside [3.60, 3.85]: F1d reproduces in this process. With AWQ near 3.26 ms,
  `Tetra` is slower than AWQ by about 14 %, where the 216-matrix row read 5 %. The paper says so.
- `Tetra` lands under 3.60: the 36 `v_proj` cost less than F1d implies. Read against F1d's image,
  which is older, before any claim.
- `Tetra` lands above 3.85, or a published arm moves by more than 2 %: something changed on the
  card or in the image. The table is still published whole, and the move is written beside it.
- b/weight departs from 2.150: `rtbits` and the bench disagree on the stream, a defect in one.
