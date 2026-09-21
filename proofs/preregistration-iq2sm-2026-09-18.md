# Preregistration. IQ2_S and IQ2_M, the honest rungs of llama.cpp's 2-bit ladder

**Written, committed and TIMESTAMPED on 2026-09-18, BEFORE the run.**
Operator go given 2026-09-18. Cost announced: about 20 min on l40sx1, **$0.60**, timeout 1 h.

## 1. Why the record's IQ2 number is the wrong one

The dossier carries IQ2_XXS at **38.87** micro on CUDA (2,280 questions, 2026-08-31). XXS is
the bottom rung of the ladder, around 2.06 bits a weight. Our bare `Tetra` is 2.1498 kernel
b/weight and reads 53.49 on the same questions, so the comparison as it stands flatters us: it
sets our format against the most aggressive rung, not the comparable one.

IQ2_S sits around 2.50 and IQ2_M around 2.70, which brackets our 2.1498 from above and sits
under our 2.9408. Those are the rungs a reader would ask about.

## 2. The configuration

One job, the `ghcr.io/ggml-org/llama.cpp:full-cuda` image. `llama-quantize` builds IQ2_S and
IQ2_M from `qwen3-4b-f16.gguf` already in the bucket, `llama-server` serves each, and
`ops/gguf_mmlu_thin.py` scores them against **`mmlu-prompts.jsonl` as it is**, the 6.19 MB file
that passed 2,280 of 2,280 qhash against the reference dumps in August.

Nothing is rebuilt. The prompts, the source GGUF and the scorer all come from the bucket
unchanged, so these arms are byte-comparable with the IQ2_XXS arm and with every 2,280-plan
dump in `docs/data/mmlu-dumps/`.

The rate is read from the **file size**, bytes times 8 over 4,022,468,096 parameters, the same
route the record uses for AWQ. The bpw llama.cpp advertises is not used.

## 3. Signed prediction

| arm | micro | b/param from the file |
|---|---|---|
| IQ2_S | **46** [40, 52] | 2.7 [2.4, 3.0] |
| IQ2_M | **49** [43, 55] | 2.9 [2.6, 3.2] |

The points interpolate between the measured IQ2_XXS at 38.87 and the f16 at 70.32 on a curve
that is steep at the bottom, which is what llama.cpp's own perplexity tables show across this
ladder. The intervals are wide because no rung between XXS and f16 has ever been measured here.

Named against me, and it is the outcome that would hurt: **IQ2_S at or above 53.49** would mean
llama.cpp matches our bare `Tetra` at a comparable rate, and the project's central claim would
need restating. That possibility is why this runs.

## 4. The readings

| IQ2_S | reading |
|---|---|
| <= 48 | Our format leads at comparable rate, and the ladder's shape is as expected |
| 48 to 53 | The lead is narrow and rate-dependent; every published comparison must carry both rates |
| **>= 53.5** | **llama.cpp matches bare `Tetra`.** The claim to restate is ours, and the diagnostic's priorities change |

## 5. Controls

1. The prompts file is the committed one, unmodified, 2,280 lines carrying their own qhash.
2. Both dumps carry the same question set as `mmlu-4b-gguf-iq2xxs-cuda.csv`, so the three
   llama.cpp arms are mutually paired.
3. The file sizes are recorded from the job's own `ls`, not from llama.cpp's advertised bpw.
4. The sha256 of each produced GGUF is recorded.

## 6. What it will not establish

- Nothing on the 14,042-question split: these are 2,280-question arms, like every foreign-stack
  arm in the record.
- Nothing about speed. The scorer measures answers, not tokens a second.
- Nothing about IQ3 or the K-quants, which are another ladder.
