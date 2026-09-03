#!/usr/bin/env bash
# The IQ2_XXS arm, produced and measured on Metal: 0 USD, on the dev machine.
#
# WHY THIS ARM
# ------------
# On 2026-08-30 the 2-bit GPTQ arm turned out to be unusable as a rival:
# vLLM 0.26.0 refuses to serve it (bits=2 sym=True), and where it does load it
# only generates gibberish. The quality column of the case file for a 2-bit
# competitor therefore stays EMPTY, as it has been from the start. The 17.04 of
# QTIP is a quotation from Table 6, never a measurement from our harness.
#
# The dividing line is the FAMILY rather than the bit count: per-group affine
# scalar (GPTQ, AWQ, MLX q2, Q2_K) against codebook / lattice / trellis (LLVQ,
# QTIP, QuIP#, AQLM, VPTQ, IQ2_XXS). If naive scalar quantization held up at
# 2 bits, neither QuIP#, nor QTIP, nor LLVQ would exist.
#
# IQ2_XXS is the ONLY candidate that holds up and is within our reach:
#   * 2.06 bpw against our 2.0702, the tightest comparison in the case file,
#     tighter than QTIP (2.000);
#   * a codebook SMALL ENOUGH TO FIT IN A LUT, so the counterfactual of
#     docs/BACKLOG.md §4.4. That is the DIRECT test of the paper's thesis:
#     our 1.1·10¹⁴ points do not fit, hence the unfolding to 4.80 b/weight;
#   * the most widely deployed 2-bit format in the world;
#   * produced here for 0 USD, and the same GGUF then goes to CUDA. It is the
#     only arm that crosses both backends.
#
# THE CALIBRATION IS OURS, AND IT IS VERIFIED
# -------------------------------------------
# calib-c4-shard1.txt comes from en/c4-validation.00001-of-00008.json.gz, the
# shard that llvq-llm/src/corpus.rs:187 reserves for calibration. It holds 305
# documents and 131,072 tokens, the n_calib × calib_len of every published LLVQ
# artifact. Its FNV-1a fingerprint is 40300263e5d0afa2, IDENTICAL to the GPTQ
# arm's. Two third-party arms will therefore see the same text as we do, and
# that is machine-verified rather than declared.
set -euo pipefail

WORK="${WORK:-/tmp/iq2}"
F16="$WORK/qwen3-4b-f16.gguf"
CALIB="$WORK/calib-c4-shard1.txt"
IMAT="$WORK/imatrix-c4-shard1.dat"
OUT="$WORK/qwen3-4b-iq2xxs.gguf"

for f in "$F16" "$CALIB"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 2; }
done

echo "== 1. importance matrix, on OUR calibration corpus =="
# -c 2048: the window length of our artifacts (calib_len), not a default.
[ -f "$IMAT" ] || llama-imatrix -m "$F16" -f "$CALIB" -o "$IMAT" -c 2048 -ngl 99

echo "== 2. IQ2_XXS quantization =="
llama-quantize --imatrix "$IMAT" "$F16" "$OUT" IQ2_XXS

echo "== 3. memory accounting =="
# WARNING: b/param over the WHOLE MODEL, embedding included, is the only legal
# accounting (§7, rule no. 1). For a GGUF it is direct: the bytes OF THE FILE
# divided by the parameters. Better provenance than our own arm, whose
# embedding is *modeled* at 8.5 b/param.
#
# WARNING: the denominator is 4,022,468,096, the REAL parameters of Qwen3-4B,
# TIED heads. Do not reuse a count that adds embed_tokens and lm_head: the GPTQ
# arm left a b/param wrong by +9.67% there on 2026-08-30.
python3 - "$OUT" "$F16" <<'PY'
import os, sys
PARAMS = 4_022_468_096
for p in sys.argv[1:]:
    b = os.path.getsize(p)
    print(f"  {os.path.basename(p):28s} {b:>14,} B   {b*8/PARAMS:6.3f} b/param")
print("  reference                     LLVQ Planes14+q8 5.162 · AWQ 5.302 · GPTQ2 3.489")
PY

echo "== 4. perplexity, wikitext-2, ctx 4096 =="
echo "WARNING: the llama.cpp PROTOCOL is not ours (sliding windows against"
echo "   12 non-overlapping windows). The LEVEL does not compare to ours;"
echo "   only the ratio to the f16 control OF THIS STACK gets published."

echo "== 5. throughput =="
llama-bench -m "$OUT" -m "$F16" -p 0 -n 128 -r 3
