# Deviations from the Metal perplexity prereg of 2026-10-04

The prereg is `preregistration-ppl-metal-2026-10-04.md`, sha256 `b7a9312c`, stamped before the
run. It is not edited. **Nothing ran.** The route it describes is abandoned, and this file records
why and what replaces it.

## É1. The download does not complete, and the tool cannot resume

The prereg costs the run at $0 "after a 9.3 GB download from the bucket". That download is the
whole route, and it failed.

`hf buckets cp` has no timeout and no resume. When its socket dies it sleeps indefinitely at 0% of
CPU, and killing it restarts that file from byte zero. Four attempts on two files, 2026-10-04 and
2026-10-05:

| attempt | stalled at | of |
|---|---|---|
| 4B, 1 | 448 MB | 1,418 MB |
| 4B, 2 | 430 MB | 1,418 MB |
| 4B, 3 | completed, about 8 MB/s | 1,418 MB |
| 8B, 1 | 851 MB | 2,815 MB |

Three stalls in four attempts, between 430 and 851 MB. The 14B is 5,087 MB and no attempt has
held more than 851 MB unbroken, so that size is a lottery and not a delay.

What survives: `~/scelles-2026-10-04/qwen3-4b-sealed.bin`, 1,418,224,685 bytes, sha256
`886391a8`, which satisfies control 1 of the prereg for the 4B. That size could still be read on
Metal. The 8B and the 14B are not on the machine.

Operator decision of 2026-10-05: cut the download. The route is not retried.

## É2. What replaces it, and why it measures more

Read the two **trained bases** on a card instead of bringing the sealed files to the Mac. Both are
already in the bucket, on the right side of the network. This is preregistered separately, in
`preregistration-ppl-bases-carte-2026-10-05.md`.

The replacement gives strictly more than the abandoned route at 8B, which is the only size where
the Metal route had a paired base:

| quantity | Metal route | card route |
|---|---|---|
| sealing cost at 4B | device-clean, unpaired | device-clean, **paired** |
| sealing cost at 8B | device-clean, paired | device-clean, **paired** |
| device effect on a quantized file | the sealed file, three sizes | the base, 8B, **paired** |

The device effect moves from the sealed file to the base, which is the pairing that exists: the
8B base's twelve per-window NLLs are on disk in Metal
(`dclm-8b-rowscales-2026-09-21-brut/ppl-ft-f16.txt`), where no sealed file has a Metal reading at
all.

It also costs about $0.25 and transfers nothing.
