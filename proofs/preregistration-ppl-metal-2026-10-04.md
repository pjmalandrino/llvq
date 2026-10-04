# Preregistration. The three sealed files on Metal, and the device effect

**Written, committed and TIMESTAMPED on 2026-10-04, BEFORE the run.**
Operator go given 2026-10-04. Cost: **$0**, the Mac, after a 9.3 GB download from the bucket.
Measured code: repository at `c7392f1`. It follows
`preregistration-ppl-scelles-2026-10-04.md` and closes its deviation É2.

## 1. Why this run exists

The cost of sealing is measured at one size out of three, and it is not resolved there. É2 of the
CUDA run's deviations says why: the trained bases of the 4B and the 8B were read on Metal, the
sealed files on a card, so their difference mixes sealing with a change of arithmetic.

This run reads the same three sealed files on Metal. It buys three things, in this order of
value:

1. **The device effect on a sealed file**, Metal against Cuda on identical bytes, at three sizes.
   No such control exists. Every cross-device perplexity in this repository rests on one f16
   anchor at 8B, 8.9893 on the Mac in f32 against 8.9899 on the card in f16, which conflates the
   device with the dtype and says nothing about a file holding lattice codes and int4 records.
2. **The paired cost of sealing at 8B**, device-clean, since
   `dclm-8b-rowscales-2026-09-21-brut/ppl-ft-f16.txt` holds that base's twelve per-window NLLs
   on Metal.
3. **The cost of sealing at 4B**, device-clean but unpaired: that base, 12.3267, is derived from
   a published ratio and no window log survives.

## 2. What runs

```
LLVQ_DTYPE=f16 ppl 4096 12 metal ~/scelles-2026-10-04/qwen3-4b-sealed.bin
LLVQ_DTYPE=f16 ppl 4096 12 metal ~/scelles-2026-10-04/qwen3-8b-sealed-B.bin
LLVQ_DTYPE=f16 ppl 4096 12 metal ~/scelles-2026-10-04/qwen3-14b-sealed.bin
```

Nothing else. `LLVQ_CONFIG` unset, dense reconstruction, the arithmetic the published MMLU uses.
`nice 10` and nothing else running on the machine.

## 3. Controls

If one fails, no number gets published.

1. Each file's sha256 equals the one the Cuda job read: `886391a8c03f66dc...`,
   `7bdb9a5503518081...`, `61db37fe7ce8e6a5...`.
2. Each run prints dtype f16, context 4096 and 12 windows.
3. All three print token fingerprint `3f1baca9033bf251`.
4. The 14B reconstructs in 68.7 GB of RAM without swapping. Its dense f16 is 29.5 GB; if the
   machine swaps, the run is killed and the size is dropped, not reported slower.

## 4. What gets published, and what does not get compared

Published: three perplexities on Metal, their difference from the Cuda run of the same bytes, the
paired interval of that difference at each size, and the paired cost of sealing at 8B.

Not compared: the 4B's sealing cost to the 8B's, since one is paired and the other is not. The
Metal numbers to any perplexity read at another context or window count.

## 5. Decision rule

| result | reading |
|---|---|
| the device effect is inside ±0.12% at all three sizes | the repository's constant-file perplexity bar covers it, and the cross-device comparisons already published stay as they are |
| it exceeds ±0.12% at any size | every cross-device perplexity in the file carries that figure from now on, including the 4B's ratios to f16 and AWQ in `ppl-scelles-2026-10-04` |
| the 8B paired sealing cost excludes zero | the int4 embedding has a measured perplexity cost on a served object, at one size |
| it contains zero | sealing costs nothing resolvable at 8B either, and the +1.52% carried from the 4B `Planes14` measurement does not transport |
| the 14B swaps or the decode refuses on Metal | the size is dropped with its reason, and the other two are published |

## 6. Signed prediction

| quantity | point | interval |
|---|---|---|
| device effect, Metal minus Cuda, each size | **+0.10%** | [−0.50%, +0.50%] |
| 8B paired sealing cost | **+1.40%** | [+0.80%, +2.00%] |
| 4B sealed on Metal | **12.51** | [12.35, 12.65] |
| 8B sealed on Metal | **9.70** | [9.55, 9.85] |
| 14B sealed on Metal | **8.44** | [8.30, 8.58] |

The reasoning: the device effect is predicted small because the reconstruction is decoded on the
host in both runs and only the forward pass changes, and the 8B sealing cost is predicted at the
+1.396% the cross-device difference already read, because the embedding's +1.52% predicts almost
the same number by a different route. The known flaw: those two agree by coincidence if the
device effect is not small, and this run is the first thing that can tell them apart.
