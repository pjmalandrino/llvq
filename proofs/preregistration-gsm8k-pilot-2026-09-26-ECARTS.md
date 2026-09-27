# Deviations from the GSM8K pilot preregistration (2026-09-26)

Prereg: `preregistration-gsm8k-pilot-2026-09-26.md`, sha256 `bfd7d75e`, stamped at 14:56 UTC
before the first generated token.

## E1. The two arms ran concurrently, from problem 9 of arm A

The prereg lists the two `gsm8k` commands one after the other and the first launch chained
them. Arm A slowed from 111 to 782 ms a token over its first seven problems, then came back
to 370, with the process steady at 11 GB and 65 % of the Mac's memory free (*measured*,
dump rows 1 to 8). The user's browser held the GPU at the same time. At 111 ms a token the
4B's 8 GB of f16 weights stream at about 72 GB/s, far under the M3 Max's 400 GB/s, so the
card had room for a second process.

At 15:05 UTC the chaining shell was stopped, arm A left running (pid 78759), and arm B
started with the command of §2, unchanged.

What it changes: the Metal timings of both arms are shared-GPU timings from then on. §4
already excluded them from any pricing. What it does not change: the tokens. Each process
decodes greedily with its own arithmetic, and a second process on the card moves no
logit. The lengths, the cap stops and the extraction, the three quantities the pilot
exists for, are untouched.

## E2. `ops/gsm8k_vllm.py` edited after control 4, in its generation part only

Control 4 ran on the stamped script (sha256 `6196fd2b`) at 16:29 UTC: 50 of 50 prompts
identical to arm A's dump, run fingerprint `94c7b98876288901` identical. The script was then
aligned on the vLLM arguments of `ops/awq_speed.py`, which ran in the pinned image:
`tokenizer_revision`, `enable_prefix_caching=False`, `top_p=1.0`, the `TokensPrompt` import
with its dict fallback, `use_tqdm=False`. New sha256 `907ee931`. The prompt builder is
untouched, and the same check passes again on the new file.

The pilot runs no vLLM generation, so no pilot number depends on the edit. The campaign
prereg cites `907ee931`.
