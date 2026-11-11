# Prereg M3: deviations

Beside `preregistration-spec-mem-m3-2026-11-11.md` (stamped, commit cf903db), which is not edited. Written on
2026-11-11 before the job was submitted, and stamped before submission.

## E1. Step (0): base Tetra, the served 4E object alone

Asked by the operator before submission, relayed by the orchestrating session.

Added to `ops/jobs/spec-mem-m3.sh`, before step (a), each in its own process beside its own 100 ms nvidia-smi sampler:
- `LLVQ_CONFIG=<v9> LLVQ_CONTEXTS=5 fusedrun <sealed> 256` (tag `base-short`);
- `LLVQ_CONFIG=<v9> LLVQ_CONTEXTS=8192 fusedrun <sealed> 256` (tag `base-8k`), the form of pic-vram-8k-2026-11-08.

No drafter and no tree pass are loaded. Both give the base peak VRAM and the base decode tok/s on the same card and
image. `fusedrun` is already in the image; the image is unchanged (Space built from cf903db). Only the job script and
this file moved after cf903db.

The job's summary (`pairs.txt`) prints, for each speculative arm, its pair (peak VRAM, tok/s) beside the base pair, with
the difference in GB and the ratio in tok/s:
- house a12, house a12ip and GSM8K a12ip against `base-short`;
- the 8k prompt against `base-8k`.
Which base each ratio uses:
- The base ratio divides specbench's tree tok/s by fusedrun's decode tok/s after the prompt. They come from two
  harnesses and two texts: specbench counts total tokens over total wall on its prompts, and fusedrun times the decode
  after a repeated seed prompt. This ratio is reported, not gated.
- The GSM8K and house ratios against greedy keep using specbench's own 4E greedy on the same text, as in SP8. The
  summary prints both, and names the base of each.

Signed predictions, added by the orchestrating session, verbatim:
- base peak at a short context 2.01 GB [1.9, 2.2] (measured before: pic-vram-8k);
- base peak at 8k 5.95 GB [5.7, 6.2];
- a12ip minus base at a short context +1.6 GB [1.2, 2.0];
- a12 minus base +2.9 GB [2.5, 3.4].

## E2. Cost and timeout

The two base processes add about 10 min (a load each, and at 8k six prefills of about 36.5 s each,
pic-vram-8k-2026-11-08-brut/speed.txt). New estimate: about 58 min, **$1.74** (*estimated*). Timeout 85 min: **$2.55**
at worst, under the $2.60 asked for the whole job. $10.88 of $15 at worst after it.
