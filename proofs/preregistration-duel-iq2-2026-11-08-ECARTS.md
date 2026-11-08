# Deviations from preregistration-duel-iq2-2026-11-08.md

The stamped prereg is not edited. Each deviation is written here, dated, before the measurement it touches.

## D1, 2026-11-08: our image is the Space at `0536a87c`, not `c16c4c73`

The prereg names Space `Pier-Jean/llvq-runner-cuda` at `c16c4c73` (built from `f63b590`). At launch the Space was at
`0536a87c`, rebuilt the same morning from `1929db1` (branch `claude/batch-l40s`) by another session that runs jobs on
it. Rebuilding the Space back would break that work, so our side runs on `0536a87c`.

What differs on our path, read from `git diff f63b590 1929db1`:

* `1929db1` contains main at `1997edc`, so the image carries `22338dd`, which changes the group size of the 4E pass in
  `llvq-llm/src/persist4e_cuda.rs` (the staged `small` and the head maxima are subtracted from the activation budget).
  This is the 4E of main today. The job's gate still requires 4E to decode the eager path's 256 tokens at each length.
* `served.rs` gains an optional `batch` field; the three v9 configs carry none, so the parsed config is unchanged.
* `fusedrun.rs` groups its mode refusals in one table; `LLVQ_CONTEXTS` alone passes as before.
* No file under `llvq-cuda/src` changes; `llvq-cuda/kernels` gains `attn_dec_batch.cu`, which the single-sequence path
  does not load.

Consequence: our speed and peak memory are those of main's served 4E, not of the build that produced 221.5 / 142.5 /
86.3 tok/s. The published figures stay where they are; the duel table uses the same-day reading.

## D2, 2026-11-08: run 1 stopped at the first bench on all three sizes; retries on two arms

Run 1 of each llama.cpp job stopped at its first `llama-bench`: `vram()` tee'd its peak line to stdout, which the caller
redirects into the bench JSON, so the parser read invalid JSON (fixed in `8c19ba9`, the line goes to stderr). Each run
1 got through the f16 gate, the imatrix, the rung choice, the quantization and the perplexity of IQ2_XXS before
stopping; those files are in the bucket.

The retries (`RETRY=1 REUSE=1`) take the first attempt's gate, rung and quantized files by sha256 and measure two arms,
on the operator's choice: IQ2_S, the rung at our bits, and unsloth's UD-IQ2_XXS. Our own IQ2_XXS keeps only its
perplexity from run 1; it gets no speed, VRAM, MMLU or GSM8K. Ceilings 2h / 2h / 3h30.

## D3, 2026-11-08: retry 1 stopped before any measurement; retry 2

Retries 1 of the 8B and the 14B stopped 12 minutes in, at the reuse block: it read `$DIR`, which the job did not
receive (`set -u`). Nothing was measured. The 4B retry 1 was cancelled in the queue, before it started. `DIR` is now
passed, a static check finds no other unassigned variable in the job body, and the reuse block was run on the Mac
against stub files before relaunching. Retry 2 measures the same two arms; ceilings 2h / 2h / 200m.

## D4, 2026-11-08: retry 2 starved a server slot; retry 3 with a larger server batch

Retry 2 of the 14B stopped in the MMLU census: one request went 900 s without an answer. The server log shows why:
`llama-server` with 16 slots and its default batch of 2,048 tokens fills its low slots first, and slot 15 was never
served while 6,750 other requests went through. Reproduced on the Mac with the 0.6B and 400 MMLU prompts: at the
default batch, slot 0 serves 62 requests and slot 15 one; at `-b 16384`, every slot serves 25. Retry 2 measured, before
stopping, the 14B IQ2_S perplexity (9.4696) and its decode speed; retry 3 measures them again. The 4B and 8B retries 2
were cancelled before they reached the census (the 4B in its build, the 8B in the queue).

The same reproduction shows that llama.cpp's MMLU picks depend on the server batch: 2 of 400 picks differ between the
two batch sizes (47.20 % against 46.99 %). Retry 3 runs every arm at `-b 16384`. Ceilings 100m / 110m / 190m, so the
duel stays under its $19.80 cap even if all three run to their ceiling.

## Correction to D1, 2026-11-08, after the run (audit `docs/archive/audit-bugs-2026-11-08.md` §2.1)

D1 says `22338dd` "changes the group size of the 4E pass". It does not: its only change to
`llvq-llm/src/persist4e_cuda.rs` is the closure that prints the verify pass's group sizes in its install line
(`git show 22338dd -- llvq-llm/src/persist4e_cuda.rs`). The 4E module the duel ran is the published one (sha256
`1b422b62`). D1 is kept as written; this note replaces its second bullet as a cause. The speed differences against
the published figures are read sweep to sweep in the journal.
