# Deviations from the GSM8K wave 1 preregistration (2026-09-26)

Prereg: `preregistration-gsm8k-campaign-2026-09-26.md`, sha256 `fe8d1a51`, stamped at 18:29 UTC
before the smoke.

## E1. Run 5 died at start on a module the pinned vLLM image does not carry

Attempt 1 (job `6ab81d8052d0dbd7f1d97607`) ran 67 s and stopped on `ModuleNotFoundError: No
module named 'pyarrow'`, when `ops/gsm8k_vllm.py` opened the `openai/gsm8k` parquet, before any
model was loaded. Cost 0.03 $. The assumption that the image carried pyarrow came from
`ops/vllm_score.py`, and it was not checked.

Fix: `ops/jobs/gsm8k-vllm.sh` installs `pyarrow==25.0.1` before the script, the pin of the
row-scale jobs, and writes the installed version into the job's directory. The script, its
sha256 `907ee931` and the protocol are unchanged. Attempt 2 runs under `RETRY=2`, in its own
directory.
