# Deviations from `preregistration-paper-table-2026-09-25.md`

The prereg is stamped and not edited; what departed from it is written here.

## É1. Four jobs died at container start, and three were relaunched

The first IQ2 job and both vLLM jobs (09:03 UTC+2), then the IQ2 relaunch, failed before
any command ran: `OCI runtime create failed ... failed to set MOUNT_ATTR_IDMAP on
/usr/bin/nvidia-cuda-mps-control`, 4 s of running each. Two different third-party images,
one of them the pinned `vllm/vllm-openai:v0.26.0` that ran on 2026-08-17, so the platform
and not the scripts; the Hugging Face status page listed Jobs as operational. The same
three runs, unchanged, were relaunched an hour later and completed. No measurement was
taken twice.

## É2. The three served runs found two defects of the served path, fixed before any number

- **Lone int4 projection, several rows.** `model::group_forward` took its dense branch for
  a group of one int4 projection (no rotation key) and handed `tv_q4_h`, a single-vector
  kernel, every row at once. The prefill gate (203 rows) failed at 4B on `o_proj`
  (831,488 values for `d_in` 4096) and at 8B on `down_proj@10`. An int4 `v_proj` never
  reached that branch because q and k rotate. Fixed by fanning the rows out (`int4_rows`);
  one row is the old call, so a decode step is unchanged.
- **14B staging.** `tv_q4_h` stages its whole activation and was bounded by the 48 KiB
  default; a 14B int4 `down_proj` stages 69,632 B. Refused at load. Fixed by posing the
  opt-in on the function, as `rot_apply` already did (the L40S offers 101,376 B).

Commit `648a51b` on `tetra/4b-sealed` (`493df9c` on `claude/jolly-clarke-vprf0a`): two
tests, six mutants killed, the `llvq-llm` suite 53/53, the CUDA half type-checked in
`llvq-check` with a canary. The image was republished from it: Space `0fb47462`, where the
prereg's runs 1–3 were to run on `89110c04`. Runs 1–3 are relaunched on `0fb47462`; run 4
(AWQ 4B census) stays on `89110c04`, which differs only by these two fixes, neither on the
dense path it used.

The relaunches write to `paper-served-<size>-2026-09-25-r2/` (`RETRY=2` in
`ops/jobs/paper-served.sh`), because the failed attempts left their directories behind
and the launcher refuses an existing one.

What this means for the paper: the census MMLU of the three sealed objects are the dense
reconstruction's and never went through either defective path. What the defects did stop
is any claim that the objects **served** before this fix: none had.

## É3. Cost

Announced to the operator 3.25 $; the prereg re-costed 3.47 $ before the first job. The
failed attempts cost 0.10 $ of running (É1 ~0, É2 0.03 + 0.03 + 0.04).
