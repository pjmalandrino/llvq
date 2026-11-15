# Second deviation from the prereg of R3 (KV ladder): `k4v4c` alone

Beside `preregistration-recherche-debit-kv-ladder-2026-11-14.md` (sha256 8913d843) and its first deviation note
(stamped, not edited). Written on 2026-11-14 before the third attempt of `k4v4c`, and stamped before it.

## 1. What happened

In the second attempt `k8v8`, `k8v4` and `k4v4h` passed their load check and are running. `k4v4c` (6af8a0c3) died
in its load check: `LLVQ_DATASET_REV` names one revision for every dataset a process reads, and the calibration
asked `HuggingFaceTB/dclm-edu` for WikiText's commit, a 404. No window was scored.

## 2. What changes, for `k4v4c` only

| second attempt | third attempt |
|---|---|
| `LLVQ_DATASET_REV=b08601e0` | unset: WikiText-2 and `dclm-edu` are read at main |
| no check of the scored text | the load check requires the token fingerprint of the other arms on its 2 windows of 256, `8f55a59b1ba97f07`; the fingerprint of the 12 windows of 8,192 must equal the other arms' or the arm is void |
| output `<arm>-r2` | `k4v4c-r3` |

The `dclm-edu` commit read is whatever main holds; the run prints it, and the journal records it.

## 3. What does not change

The other three jobs, the image, the reading of §5 and the predictions of §7.
