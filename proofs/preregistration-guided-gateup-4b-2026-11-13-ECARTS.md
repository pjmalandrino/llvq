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
