# Preregistration. The two trained bases on a card, and the device effect

**Written, committed and TIMESTAMPED on 2026-10-05, BEFORE the run.**
Cost: about **$0.25** on l40sx1, timeout 30 m, so $0.90 at worst. Campaign cap $9, of which $0.33
is spent. Measured code: the pinned image `97a2b62a6d0c8911dcd8e4d26a40461f0196ccc9`, built from
commit `5333ac8`. Repository at `5c6f5e2`.

It replaces `preregistration-ppl-metal-2026-10-04.md`, whose download route is abandoned
(É1 there), and closes É2 of `preregistration-ppl-scelles-2026-10-04.md`.

## 1. Why this run exists

The cost of sealing a file is measured at one size out of three, and it is not resolved there. The
reason is a device cross: the trained bases of the 4B and the 8B were read on Metal and their
sealed files on a card, so their difference mixes sealing with a change of arithmetic.

Reading the two bases on a card removes the cross. It buys, in this order:

1. **The cost of sealing, device-clean and paired, at 4B and 8B.** Both sides then come from a
   card, and the sealed side's twelve per-window NLLs are already on disk
   (`mesures/ppl-scelles-2026-10-04-brut/`).
2. **The device effect on a quantized file at 8B**, paired: that base's twelve per-window NLLs
   exist in Metal. No such control exists anywhere in the repository. The only device comparison
   on record, 8.9893 on the Mac in f32 against 8.9899 on the card in f16, is an f16 model and
   conflates the device with the dtype.

The 14B needs nothing: both of its readings are already on a card.

## 2. What runs

One job, l40sx1, two calls, nothing else.

```
LLVQ_DTYPE=f16 ppl 4096 12 cuda /out/dclm-ft-2026-09-19/qwen3-4b-dclm-ft.bin
LLVQ_DTYPE=f16 ppl 4096 12 cuda /out/dclm-8b-ft-2026-09-21/qwen3-8b-dclm-ft.bin
```

`LLVQ_DTYPE` inline, not in the job environment, which the job script refuses by name.
`LLVQ_CONFIG` unset, so this is the dense reconstruction, the arithmetic of every perplexity
cited here.

## 3. Controls

If one fails, no number gets published.

1. `oracle Qwen/Qwen3-0.6B 64 cuda` prints MATCH (hard rule 10).
2. Bytes on the mount: 1,794,564,765 and 4,364,205,777.
3. **sha256, and this one is load-bearing.** Folding row scales does not change a file's size, so
   the untrained 4B base `dclm-4b-2026-09-18/qwen3-4b-dclm.bin` has the **same 1,794,564,765
   bytes** as the trained one. Only the digest tells them apart: the trained files begin
   `f8c1c903` (4B, dclm-rowscales-2026-09-20) and `783cef67` (8B,
   dclm-8b-rowscales-2026-09-21). A mismatch means the bucket directory does not hold what its
   name says.
4. Each run prints dtype f16, context 4096, 12 windows.
5. Both print token fingerprint `3f1baca9033bf251`.

## 4. What gets published, and what does not get compared

Published: the two base perplexities on a card; the paired cost of sealing at 4B and 8B with its
interval; the paired device effect at 8B with its interval.

Not compared: the 4B device effect, which has no Metal window log. The device effect measured on a
**base** is not transferred to a **sealed** file without saying so: the sealed files carry an int4
embedding and int4 records the bases do not, and nothing here measures those on two backends.

## 5. Decision rule

| result | reading |
|---|---|
| the 8B device effect is inside ±0.12% | the repository's constant-file perplexity bar covers it, and the +1.396% already read across devices at 8B may be quoted as the cost of sealing |
| it exceeds ±0.12% | every cross-device perplexity in the file carries that figure from now on, including the 4B's ratios to f16 and AWQ in `ppl-scelles-2026-10-04` |
| the paired sealing cost excludes zero at both sizes | the int4 embedding has a measured perplexity cost on a served object, at two sizes |
| it contains zero at either size | the +1.52% carried from the 4B `Planes14` measurement does not transport to that size, and the 14B's unresolved reading is not an outlier |
| a sha256 fails control 3 | that size is dropped by name, the other runs, and the bucket directory is corrected before any re-run |
| otherwise | not settled, operator decision |

## 6. Signed prediction

| quantity | point | interval |
|---|---|---|
| 4B base on a card | **12.33** | [12.26, 12.40] |
| 8B base on a card | **9.57** | [9.52, 9.62] |
| device effect at 8B, paired, card minus Metal | **+0.10%** | [−0.50%, +0.50%] |
| sealing cost at 4B, paired | **+2.08%** | [+1.4%, +2.8%] |
| sealing cost at 8B, paired | **+1.40%** | [+0.8%, +2.0%] |

The reasoning: the device effect is predicted small because the reconstruction is decoded on the
host in both runs and only the forward pass changes, so each base should read near its Metal
value, 12.3267 and 9.5725. The two sealing costs then fall out of the sealed readings already in
hand, 12.5834 and 9.7062.

The known flaw: those two sealing costs are **predicted from the very cross-device differences
this run exists to validate**, so they agree with it by construction unless the device effect is
large. The quantity that can surprise is the device effect, and it is the one with the widest
interval relative to its point.
