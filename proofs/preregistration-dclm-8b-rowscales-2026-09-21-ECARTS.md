# Deviations. Preregistration dclm-8b-rowscales-2026-09-21

The prereg is stamped (sha256 `d81cbfcc0cd7fb77...`) and not edited.

## E1. The loop rate fell 0.0006 s/step under its interval

Predicted 0.52 s/step [0.40, 0.70]; measured 0.3994 (*measured*, `journal.jsonl` in the brut dir).
The h200 did better than the vendor factor of 2.3 over the L40S: 0.6479 / 0.3994 = 1.62 per step
at 1.846× the FLOPs, so ×3.0 per FLOP (*computed*; 0.6479 from `dclm-rowscales-2026-09-20.txt`).
The probe read 0.596, 49 % slower than the loop (17 % at the 4B, 0.7573 against 0.6479; both
*computed* from the `probe.jsonl` files). `MAX_TRAIN_SECONDS=7000` held with room: 9,507 × 0.596
= 5,666 s at the probe's rate (*computed*).

## E2. The timeout was written `135m`, not `2h15m`

`run.py` accepts one unit; `135m` is the same ceiling ($11.25).

## E3. The FT census ran under `CENSUS_DATE=2026-09-22`

Its bucket directory is `census-8b-ft-2026-09-22/`; the launcher's default. The prereg names no
directory.

## E4. The gain did not follow the prior

The prediction (+2.0) leaned on the paper's Table 6, where the gain falls as the base rises. The
8B gave +3.29 on a base 6.9 points higher than the 4B's (64.87 against 57.95, *measured*), where
the 4B gave +3.15. The interval held; the reasoning behind the point did not.

## E5. The KL anchor was a quarter figure, the prediction an eighth

The prereg's KL rationale cites the 4B's −3.8 %. That figure is last quarter against first
(`dclm-rowscales-2026-09-20.txt`). The prediction is last eighth against first. On eighths the 4B
read −6.6 %. The 8B read −2.0 % on eighths and −0.3 % on quarters. All four *computed* from the
`closed.losses` of the two `journal.jsonl` (4B: `dclm-rowscales-2026-09-19-brut/`), in equal
1,188-step eighths and 2,376-step quarters. The prediction held. The "less to repair" argument was
set against the wrong 4B figure: on the predicted measure the 8B falls by a third of the 4B's.

## E6. The MMLU was scored with the f16 embedding

The prereg publishes "b/param unchanged at 3.0683" beside the FT census. 3.0683 assumes a q8
embedding. The census ran `config=none`, dense reconstruction, so the embedding and the untied
head stayed f16 (`pairs.txt` in the brut dir). That width is 4.2080 b/param in the same rtbits
table (*computed*, `dclm-8b-2026-09-21-brut/rtbits.txt`). The file's width is unchanged by the
fold; the census MMLU at 3.0683 is not measured.
