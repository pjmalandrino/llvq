# Preregistration. Perplexity of the three sealed files

**Written, committed and TIMESTAMPED on 2026-10-04, BEFORE the run.**
Operator go given 2026-10-04, with a cap of **$9** on the campaign. Cost of this run: about
**$1.35** on l40sx1, timeout 60 m, so $1.80 at worst. Measured code: the pinned image
`97a2b62a6d0c8911dcd8e4d26a40461f0196ccc9`, built from commit `5333ac8`, which already carries
`ppl` in both of `ops/Dockerfile.cuda`'s lists. Repository at `b41821b`.

## 1. Why this run exists

No sealed file has a perplexity. `paper2/sections/limitations.tex` says so in four words, and a
quantization paper is read for that number. The review of 2026-10-04 asks for it twice.

The three numbers that exist are the **trained bases before sealing**, all f16, wikitext-2 test,
context 4096, 12 windows, token fingerprint `3f1baca9033bf251`:

| size | f16 checkpoint | trained base | ratio | journal |
|---|---|---|---|---|
| 4B | 12.2361 | 12.3267 | 1.0074 | dclm-rowscales-2026-09-20 |
| 8B | 8.9899 | 9.5725 | 1.065 | dclm-8b-rowscales-2026-09-21 |
| 14B | 7.9832 | 8.4622 | 1.060 | dclm-14b-rowscales-2026-09-22 |

Sealing then moved the object twice. It wrote the embedding as int4 g64, measured at +1.52% of
perplexity at 4B on a `Planes14` base (ROADMAP-QUALITY row 4). And at 8B it moved the int4
budget from `o_proj` plus `down_proj@15-20` to `down_proj@10-26`, which gained 0.77 MMLU points
and has no perplexity reading at all (sealed-8b-27-2026-09-24).

So the published objects have no perplexity, and the gap between them and the bases above is
unmeasured.

## 2. What runs

One job, l40sx1, three calls, nothing else. The files are the bucket copies of the three objects
`ETAT.md` §2 serves.

```
LLVQ_DTYPE=f16 ppl 4096 12 cuda /out/sealed-4b-2026-09-23/qwen3-4b-sealed.bin
LLVQ_DTYPE=f16 ppl 4096 12 cuda /out/sealed-8b27-2026-09-24/qwen3-8b-sealed-B.bin
LLVQ_DTYPE=f16 ppl 4096 12 cuda /out/sealed-14b-2026-09-23/qwen3-14b-sealed.bin
```

`LLVQ_DTYPE` is inline on each call and not in the job environment, which the job script refuses
by name. `LLVQ_CONFIG` is not set, so this is the dense reconstruction, the same arithmetic the
published MMLU was read with. The corpus argument is left at its default, wikitext-2 test.

## 3. Controls

If one fails, no number gets published.

1. `oracle Qwen/Qwen3-0.6B 64 cuda` prints MATCH (hard rule 10).
2. Each file's byte count on the mount equals the bucket listing: 1,418,224,685, 2,815,098,745
   and 5,087,000,541.
3. Each file's sha256 begins with the digest the paper publishes: `886391a8`, `7bdb9a55`,
   `61db37fe` (`paper2/sections/availability.tex`).
4. Each run prints dtype f16 and context 4096.
5. All three print token fingerprint `3f1baca9033bf251`. The three files share one tokenizer, so
   a difference means the scored token stream is not the published one.

## 4. What gets published, and what does not get compared

Published: the three perplexities to four decimals, their ratio to the f16 checkpoint of the
same size, and the gap at 14B to AWQ's 8.2858. Both the perplexity and the MMLU of each file are
f16 on the same bytes, so they describe one object and the paper may state them side by side.

Not compared: these numbers to a perplexity from another corpus, another context length or
another window count. The AWQ figure at 14B was measured in August on an L40S and carries its
date when cited.

## 5. Decision rule

| result | reading |
|---|---|
| all three inside the intervals of section 6 | the numbers enter the experiments table and limitation L6 is struck |
| one or more above its interval's top | sealing costs more perplexity than the embedding's +1.52%, which is a finding about the int4 tables; published with a deviation file |
| any file below its trained base | a control is wrong, since no step of sealing improves perplexity; nothing is published until it is found |
| any token fingerprint other than `3f1baca9033bf251` | nothing is comparable, nothing is published |
| otherwise | not settled, operator decision |

## 6. Signed prediction

Each sealed file reads its trained base times 1.015, the measured cost of the int4 embedding.

| size | point | interval |
|---|---|---|
| 4B | **12.51** | [12.2, 13.0] |
| 8B | **9.72** | [9.4, 10.2] |
| 14B | **8.59** | [8.4, 9.0] |
| 14B against AWQ's 8.2858 | **+3.7%** | [+1.4%, +8.6%] |

The known flaw: the +1.52% was measured at 4B on a `Planes14` base whose embedding was q8, not
on any of these three files, and the 8B's int4 budget moved after its base was read. The 8B
interval is therefore the weakest of the three, and its top admits a cost four times the one
predicted.
