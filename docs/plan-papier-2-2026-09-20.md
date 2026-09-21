# Macro plan, paper 2: a 24-dimensional lattice read at its storage rate

Written 2026-09-20. Every number below is measured and has a journal. The
figures that do not exist yet are in section 9, with their cost.

## The arc, and why it is not a duplicate of paper 1

Paper 1 (TACO-2026-428, desk-rejected on **scope**, not on any contested
number) published `Planes14`: a 2-bit disk format whose decoder is simple
because the representation is **expanded before inference**. It reads 4.804
b/weight in VRAM. The repository's own note says it: "Calling it a two-bit
runtime obscures that tradeoff."

Paper 2 publishes the geometry that does not have to expand. Paper 1 becomes
the motivation, cited, rather than an overlapping claim.

## 1. The problem

A 2-bit code on disk is not a 2-bit runtime. Can a vector quantizer in
**dimension 24** be decoded at its storage rate? A naive table at 2 bits a
dimension has **2^48 entries**.

## 2. The geometry, which is the contribution

The Tetra word is `[state][s1][s2][s3]`, 48 bits. Decoding costs **three 11-bit
table reads** — three tables of 2,048 entries, a few kilobytes — instead of one
2^48 lookup.

Everything after this section measures that one idea.

## 3. The mechanism, and the central figure

On NVIDIA, shared memory and L1 are the **same** 102,400 bytes of SRAM per SM.
The activation tile takes L1 from the decoder table.

    amplitude across tiles 128 / 64 / 32
    Planes14    1.5 %
    Tetra      19.1 %

Planes14 is flat and Tetra swings 19 %, in one process, on one card, with the
other arm as the control. That contrast is what turns "the table has to stay
resident" from a story into a measurement.

Journal: `mesures/tuile-l40s-2026-09-20.txt`. The served default moved to the
measured row the same day, and `llvq-cuda/src/tile.rs` carries the policy.

## 4. The results table

Qwen3-4B, one process, one L40S, seven rounds with two discarded, ratios formed
round by round. `mesures/banc-t64-2026-09-20-brut/`, `data/echelle-formats.csv`.

| | Tetra | AWQ w4 | Planes14 | FP16 |
|---|---|---|---|---|
| GB read a pass | **0.95** | 1.90 | 2.18 | 7.27 |
| b/weight in VRAM | **2.148** | 4.179 | 4.804 | 16.000 |
| median ms | 3.424 | **3.261** | 4.997 | 10.973 |
| GB/s | 278 | 583 | 437 | 662 |
| wikitext ppl | **12.3268** | 13.5207 | — | 12.2361 |
| MMLU micro | 61.11 | **70.04** | — | 70.14 |

In the model, served flags, against its own dense arm in the same process:
**98.3 tok/s in 1.39 GB** against 43.4 in 8.04, x2.27 and /5.78, `oracle` MATCH
(`mesures/dclm-ft-fusedrun-2026-09-20.txt`).

**The line to lead with: our 2-bit object has a lower perplexity than the 4-bit
AWQ baseline while reading half the bytes.**

## 5. Quality for zero bits

Training the 1,069,056 row scales the file already holds: **+4.30 pp** on bare
Tetra and **+3.15 pp** on the DCLM base, at an unchanged rate, with the decoder
byte-identical. `mesures/tetranu-rowscales-2026-09-19.txt`,
`mesures/dclm-rowscales-2026-09-20.txt`.

The paper's own Table 6 makes this gain a decreasing function of the base
(r = -0.956 over five pairs), so a single number for the lever is not a
quantity — which is itself worth a paragraph.

## 6. What perplexity does not measure

Perplexity recovered to **1.0074x** f16 while MMLU stays **nine points** short.
This repository holds five dissociations running the other way, and Table 6 of
the LLVQ paper carries a sixth: its 0-gain-bit row reads a worse perplexity and
a better MMLU than its 2-gain-bit row.

Gating on perplexity would declare this model solved. It is not.

## 7. Limitations, written by us

- One model, one card for the kernel table.
- The QTIP row is from another process (2026-08-21) and its 2.246 ms sits
  **below the bench floor**, `nullk` at 2.289 ms. The August journal's
  permanent reservation is kept rather than dropped.
- The segmented Tetra kernel is written, compiled on the card, proved host-side
  by four tests, and **fails the bit-exact comparison**. It is behind
  `LLVQ_SEG_TETRA`, off by default, and is not claimed.
- The card-side identity check on the final object covered **32 tokens**, where
  F1e section 0 covered 256 on an earlier object.
- Every quality figure is a dense reconstruction, the protocol behind every
  published bar here; the served kernel reads the same records and was measured
  separately.

## 8. Method, which is a differentiator rather than a contribution

Twenty-odd preregistrations timestamped with OpenTimestamps before the first
measurement, signed predictions scored whether they flatter or not, deviations
written beside stamped files and never into them, mutation runs before a gate is
called green. Three of this session's own predictions were wrong and are
recorded as wrong, one by a factor of ten.

## 9. What does not exist yet, and what it costs

| # | | cost | why |
|---|---|---|---|
| 1 | **the venue** | $0 | TACO rejected on scope. The work was not judged |
| 2 | a second model for quality | ~$3 | the single strongest reviewer objection |
| 3 | IQ2_XXS on the census | $0.72 | three empty cells on the most deployed 2-bit format |
| 4 | 256 tokens on the final object | $0.30 | the identity claim, at full strength |
| 5 | the fusion, or a stated abandonment | 1 job | worth 2.7 % against AWQ; the smallest lever left |

Item 1 costs nothing and matters most. Items 2 to 5 together are under $5.

## 10. What NOT to claim

- Not "faster than the state of the art". AWQ is 5 % faster at four bits, and
  QTIP's number cannot carry a verdict while it beats the floor.
- Not "quality preserved". It is preserved on the metric the training optimized.
- Not a parity with the paper's fine-tuned column: ours is fine-tuned too, so
  the comparable row is their 62.8 and we are 1.69 pp under it.

The defensible sentence is a point on a curve: **quality at the level of the
2-bit state of the art, with the perplexity of the dense model, for half the
traffic of a 4-bit baseline — paid for with nine MMLU points against that
4-bit baseline.**
