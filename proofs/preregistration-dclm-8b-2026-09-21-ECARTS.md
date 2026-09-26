# Deviations. Preregistration dclm-8b-2026-09-21

The prereg is stamped (sha256 `33a270aec8ddb50b...`) and not edited. What it got wrong or did
not foresee is written here.

## E1. The memory prediction missed by a factor of 2.9

Predicted: `smoke` peak memory footprint 40 GB [33, 46]. Measured: **114.5 GB** footprint for a
23.2 GB resident set (`/usr/bin/time -l`, `smoke.txt`). The model of the prediction counted the
f32 model and the largest evaluation transient. What else `smoke` holds is not identified. At
18:23 `top` showed `smoke` at 91G, and the system-wide compressor held 82.8 GB in 6.3 GB, 13 to 1
(operator-session readings, not recorded in the brut dir). The compressor figure covers the whole
machine and is not assigned to `smoke`. Control 8 is a record, so the run stands.

The s/block column of `smoke` is a running average, elapsed over blocks done
(`smoke.rs:1122-1124`); it went from 373 to 379 s. Worked back from its rounding, blocks 1-6, the
swap window, ran at 375.0-375.5 s each against 371.5-373.0 s for blocks 7-24 (*computed*): 0.5 to
1.1 % slower. Blocks 25-36 ran at 388.5-392.5 s, 4.2 to 5.7 % slower, cause not identified. The
allocation behind the footprint is not diagnosed.

## E2. The swap was reported during the run, not only from `swap.txt`

The prereg said the operator is told if `swap.txt` shows swapping. `swap.txt` only samples before
and after `smoke`, and both samples are low: 4.0 and 4.4 GB (3,777 and 4,229 MiB). In between,
`sysctl` read 47,417 MiB used (49.7 GB) at 18:23, and it was reported then. The operator closed
apps; at 21:09 swap read 37,618 MiB (39.4 GB) and `top` showed `smoke` at 60G. These two readings
sit in the session log, not in the brut dir; the peak between samples is unknown.

## E3. The paid chain received its cap before the result

The prereg's row 2 reads "propose the paid chain with its cost; the operator sets a cap or
declines". The operator gave the go for the whole 8B chain at 21:13 ("tu peux enchainer sur la
suite du programme pour avoir le 8B au complet ... la totale"), then the cap at about 21:50,
verbatim: "Budget CAP pour le total sur le 8B 20$", and "enchaine solo jusqu'a avoir terminé
le 8B total, sauf gros problème". Both came before R was known. R fell in row 2, so the go and
the rule agree.
