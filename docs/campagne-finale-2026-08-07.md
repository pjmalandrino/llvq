# The final campaign: four arms, sixteen cells, one harness (2026-08-07)

> The table the project was aiming for: FP16, the official 4-bit, our LLVQ
> without the kernel, and our latest-generation LLVQ, all measured on the
> same L40S, the same harness, the same questions and the same windows,
> with token fingerprints printed and identical on every quality row
> (**ppl `3f1baca9033bf251`, MMLU `65dcd53655e8bfa5`**). Sources:
> [`mesures/a4-campagne-2026-08-06.txt`](mesures/a4-campagne-2026-08-06.txt)
> (arms 1-3, quality),
> [`mesures/campagne-finale-bras4-2026-08-07.txt`](mesures/campagne-finale-bras4-2026-08-07.txt)
> (arm 4, quality + control),
> [`mesures/nuit-planes12x-q8-2026-08-07.txt`](mesures/nuit-planes12x-q8-2026-08-07.txt)
> and [`mesures/phases-2026-08-07.txt`](mesures/phases-2026-08-07.txt)
> (speed/VRAM/attribution).

## The table

| | FP16 | Q4 (official AWQ) | LLVQ without kernel | **LLVQ planes14 + fused kernel + embed q8** |
|---|---|---|---|---|
| **disk** | 8.04 GB | 2.67 GB | **1.77 GB** | **1.41 GB**⁴ |
| **VRAM** | 8.04 GB | 5.30 b/param¹ | 8.04 GB² | **2.60 GB · 5.162 b/param**⁵ |
| **speed** | 43.5 tok/s³ | not comparable¹ | 43.5 tok/s | **88.4-88.5 tok/s** |
| **perplexity** (wikitext) | 12.2369 | 13.5207 (×1.105) | 16.9422 (×1.384) | **16.9358 (×1.384)** |
| **MMLU** (micro) | 70.32 ± 1.28 | 70.04 ± 1.25 | 55.59 ± 1.35 | **55.70 ± 1.35** |

¹ AWQ has never run in our engine: its VRAM comes from its own
engine (b/param, whole model), and its speed does not compare here. That
is the caveat established by campaign A4, still standing.
² Without the kernel, the model decodes to f16 at load time and runs like
FP16. That was the state of the project on Monday.
³ The FP16 arm and the LLVQ-dense arm share the same f16 engine (same
shapes, same kernels): speed and VRAM identical by construction,
cross-checked by the miniature protocol (42.8 tok/s on the checkpoint).

⁴ **Two files, the same content. Do not confuse them.** Column 4's quality is
measured on `q4b-e8.llvq` (**1.406 GB**), which carries the pre-baked
int8 embedding; its speed and its VRAM are measured on
`qwen3-4b-llvq.bin` (1.770 GB) with `LLVQ_EMBED=q8`, which quantizes the
same embedding at load time. Both produce **bit-identical** content
(checked against the bytes from `embedq` on real rows), but they are not
the same bytes on disk. An earlier version of this document wrote "one and
the same 1.77 GB file", which was wrong.

⁵ **The two numbers in this cell do not follow from one another. Do not
do the division, it does not come out.** The **2.60 GB** is the card
display (`nvidia-smi`, rounded to the hundredth); the **5.162 b/param**
is recomputed on the exact bytes and the exact parameter count, whole
model including the embedding, by `rtbits`
([`mesures/rtbits-planes-8b-2026-08-09.txt`](mesures/rtbits-planes-8b-2026-08-09.txt),
which settles the point itself: "LE CHIFFRE 4B q8 À PUBLIER EST 5,162"). The
exact footprint is 2.595 GB. Dividing the displayed 2.60 gives **5.15**,
which is what this document published until 2026-08-17. The 5.15 is not
deleted, it is labelled: a quote of a rounded card display, not a
measurement of the object.

## The three readings of the table

**1. The kernel and the format are worth ×2.03 and ÷3.09, at no cost
in quality.** Between columns 3 and 4, the same model to the bit⁴: moving to
the fused kernel + Planes14 + q8 embedding doubles the speed and divides
memory by three, while perplexity and MMLU stay inside the sampling noise
(16.9358 against 16.9422; 55.70 against 55.59, same fingerprints, same
questions). That is the project's engineering contribution, measured end
to end.

**2. Against FP16: clear dominance, a known price in quality.** ×2.03 on
speed, ÷3.09 on memory, ÷4.5 on disk; the price is ×1.384 on perplexity
and −14.6 pp of MMLU. That is the cost of 2 bits on a 4B, unchanged since
the published file.

**3. Against 4-bit: each axis has its winner, and all of them have to be
stated.** We win on disk (1.77 against 2.67) and now on **VRAM** (5.162
against 5.30 b/param, the axis we were losing three days ago); speed does
not compare honestly (different engines); AWQ wins on quality, by a wide
margin (70.0 against 55.7 MMLU). On a 4B, the A4 verdict still holds on
the capabilities axis; the 2-bit bet is still scale (the 8B already
degrades less: ×1.267 against ×1.384).

## Where the ×2.03 comes from, for reviewers

Phase attribution ([`mesures/phases-2026-08-07.txt`](mesures/phases-2026-08-07.txt)):
~2.9 ms/token come from the Leech kernel on the projections, ~25 ms from
replacing **our** dense lm_head path, which copies 778 MB per token
(`Head::project` → `broadcast_matmul`; the `TODO` is in candle's code,
but the call is ours, and the `candle_transformers` models go through
`Linear` and do not pay this copy,
[huggingface/candle#3871](https://github.com/huggingface/candle/issues/3871)).
Two formulations: **×2.03 against our dense arm as measured; ~×1.4
against that same arm corrected for its copy** (estimated from the
measured phases). The kernel figure is still the same-head ×1.12.

## Total cost

Campaign A4 (arms 1-3): $0.71. Arm 4: $0.47. The whole sequence
C1 → final campaign (layouts, wiring, embedding, phases, E2, this
campaign): **$2.85** of GPU (job-by-job detail:
[`data/jobs.csv`](data/jobs.csv)).
