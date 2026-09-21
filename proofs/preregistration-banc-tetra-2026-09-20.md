# Preregistration. Tetra in the ten-arm bench, one process

**Written, committed and TIMESTAMPED on 2026-09-20, BEFORE the run.**
Operator go given 2026-09-20. Cost: a few minutes on l40sx1, about **$0.30**, timeout 1 h.

## 1. Why the whole table is re-measured and not one row

`docs/data/README.md` records the rule, applied when QTIP was added on 2026-08-21: adding a row
to a table measured in another process "would have put rounds from two processes side by side,
which the paper's methodology forbids, so the **whole** table was re-measured with every arm
present".

Tetra is absent from `docs/data/echelle-formats.csv` because the bench ran on 2026-08-21 and
the format was named on 2026-09-06. So every arm runs again, in one process, on an L40S, the
same card family as the August run.

## 2. What runs

  phase 1   slot32,planes14,planes12x,golay70v1,fp16,awq,golay70v2,cublasf16,nullk
  phase 2   the same, plus **tetra48**

That mirrors the August structure, where phase 2 was phase 1 plus the arm under test.

**QTIP is not in this run.** `arms.rs` carries `HAS_KERNEL[qtip] = false`, so naming it is
refused by name, and the flag is documented to flip only "in the same commit that shows a
device compile". The August journal does show one, so the flag is stale rather than protective,
but flipping it is a structural decision and is not taken here.

Consequence, stated before the measurement: **the QTIP row of the final table will come from a
different process than every other row.** That is the thing this protocol exists to avoid, and
it is accepted here only because the alternative is no Tetra row at all.

## 3. The subject

`/out/dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin`, sha256 `f8c1c903b753fe34...`, the fine-tuned
object of 2026-09-20: 216 Tetra records, 36 int4, 2.1309 kernel b/weight, 2.7475 b/param.

## 4. Signed prediction

| quantity | point | interval |
|---|---|---|
| Tetra median ms | **4.36** | [3.9, 4.9] |
| Tetra GB read a pass | **0.98** | [0.96, 1.00] |
| ratio against FP16 | **2.52** | [2.24, 2.82] |
| ratio against Planes14 | **1.17** | [1.04, 1.31] |

The point comes from F1d, which read Tetra at 1.17x Planes14 on an L40S at the served tile,
against this bench's Planes14 median of 5.103 ms. The GB figure is the one ETAT already carries.

**The tile is the served 128 and is not swept here.** `tile-sweep-2026-09-09` measured that 32
returns 42 % on Blackwell by leaving the decode table its L1; whether the served tile is also
wrong on sm_89 is a separate question and a separate run.

## 5. What refutes what

- Tetra lands near 4.36 ms: the extrapolation through Planes14 held, and QTIP at 2.246 ms is
  roughly twice as fast as us. The speed claim of any paper has to say so.
- Tetra lands under 2.5 ms: the extrapolation was wrong and we are at parity with QTIP.
- Tetra lands above 5.1 ms: it is slower than Planes14, which contradicts F1d on the same card,
  and the discrepancy is the result rather than the time.
- GB read departs from 0.98: the stream accounting of `rtbits` and the bench disagree, which
  would be a defect in one of them.
