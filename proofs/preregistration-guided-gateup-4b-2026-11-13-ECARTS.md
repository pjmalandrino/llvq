# Departures from the prereg of the guided gate/up re-encode on the 4B

The prereg (`preregistration-guided-gateup-4b-2026-11-13.md`, sha256 `dac1bca14e763bf7…`) is stamped and not edited.
Departures, each written and stamped before the step it concerns.

## E1, the launch guard of two job scripts (2026-11-13, before the statistic job)

The first launch of `ops/jobs/guided-weights-4b.sh` refused itself: "still says NOT YET STAMPED". The stamped prereg
keeps that banner, as the stamped preregs of this repository do, so the guard could never pass. No job was submitted
and nothing was measured. The scripts and the token file had been uploaded to the bucket; they are inputs, not results.

The guard is replaced in the two scripts that carried it. It now refuses unless `ots info` on the `.ots` names the
sha256 of the prereg file. That is one changed check in each script; nothing else moves: the arms, the inputs, the
timeouts, the ceilings, the predictions and the decision rules are the prereg's.

| script | sha256 the prereg names | sha256 after this departure |
|---|---|---|
| `ops/jobs/guided-weights-4b.sh` | `69b809deb11c33c664f980842f089e16495cba4be1533b6c756b7fa76ced241c` | `a694493e762eb0bfeb7d277939742ec2f339a598e589f04a5f424e94890de664` |
| `ops/jobs/guided-mmlu-4b.sh` | `2708e4f1dd1b491577b94e1670ada0a2a59bef13fdace5b9ff32694d439688fb` | `17ff7d6a7f105b4d65d95bf1c4ddcb3f10a7106bd2ad21d509e93e08e8c52fb1` |

`ops/guided_weights.py`, `ops/layer_vs_output.py` and `ops/jobs/guided-encode-4b.sh` are unchanged.

## E2, both encodes on the card (2026-11-13, before either encode)

E1 above was stamped alone and committed at `d542885`; this section is added after it and the file is stamped again.

By the operator's decision of 2026-11-13 (CERVEAU O12, commit `49b714d`), the two encodes run on a card, not on the
Mac, and the phase's budget is $15 ($0.53 spent before this departure). His reason: a card encode of today's recipe
has never been timed, and its price and time are wanted for the larger sizes.

Route. One job per arm, `ops/jobs/guided-encode-card-4b.sh baseline|candidate`, on `rtx-pro-6000` (23 vCPU, $2.75/h),
image `hf.co/spaces/Pier-Jean/llvq-runner-cuda-qualite` at Space sha `37a8e9336c77fceb0b013bce6f52d0349fdd06a4`, built
from commit `4bdb838` of this branch. It is a private Space of its own, so the shared Space that the speed phase pins
does not move. The recipe is the prereg's, with two changes taken from `ops/jobs/encode-14b.sh`: device `cuda` instead
of `metal`, and 23 encoder threads instead of 12. The candidate's weights are copied from the bucket and checked
against their `files.sha256` before smoke reads them. Each arm is sealed in its own job. The MMLU job reads both sealed
files from the bucket and runs on the same private image, whose scoring sources equal those of the image the ablation
of 2026-11-12 ran (`git diff --stat 36230ff 4bdb838` empty on them).

Timeout. The script's default is 100 min. One arm is estimated at 85 to 95 min, so both arms are launched with
`TIMEOUT=120m`: a cut encode would be paid and lost.

Controls. The informational comparison of the baseline with the 2026-09-18 Mac base (prereg §4, and the EQUAL or
DIFFERENT line) no longer applies: a card encode is a different object from a Mac encode (`docs/data/jobs.csv`, row
`volume-v1c`), so the baseline's MMLU is set beside 57.95 for information only, and nothing is read from it. Both arms
run on the same card flavor and the same image. Their GPU lines are recorded and compared, not required equal: two
jobs can land on two physical cards, and bit-for-bit reproducibility of a card encode has never been measured. The
other controls hold, unchanged: the calibration token fingerprint `45fb781f65574d4d` in both arms; the baseline's
state without, and the candidate's state with, the guided line; the two states equal but for that line, checked by
`compare` before any MMLU is paid; a bit-for-bit verify of each artifact; equal sealed sizes; the dump checks and the
oracle.

Scripts. `ops/jobs/guided-encode-card-4b.sh` `e935f81a8bb1ec8eb1233303435e9f44a1e20a891aafc4707897c26e515f0893` (new).
`ops/jobs/guided-mmlu-4b.sh` `966b368e954ccc250eb52bc40f6b67651f917011e13df1ca62eda7539b174b20` (was `17ff7d6a…` after
E1). `ops/Dockerfile.cuda` `090e86c82113cb984d63ef9d88fc66ee8e47b3ab349a1985ede53f411f195018`, with `CARGO_BUILD_JOBS`
at 3 as on the speed branch (`e2ae88e`). `ops/jobs/guided-encode-4b.sh`, the Mac route, is no longer used.

Cost. Each encode: 85 to 95 min, $3.90 to $4.35 (*estimated*), ceiling $5.50 at 120 min. MMLU pair: $1.38 to $1.44
(*estimated*), ceiling $1.92. Estimated total $9.2 to $10.1. Worst case $12.92, on top of the $0.53 spent: $13.45,
inside $15.

Unchanged: the question, the arms' only difference (`LLVQ_GUIDED_WEIGHTS`, `LLVQ_GUIDED_GROUPS=4`), the statistic, the
decision rules and the signed predictions.
