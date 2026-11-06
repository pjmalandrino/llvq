#!/usr/bin/env bash
# The sealed Qwen3-4B on this Mac's GPU, before and after the Metal lot of 2026-11-06, with the same tokens.
#
#   ops/metal-bench.sh [sealed-file]          # short context, about 9 minutes
#   ops/metal-bench.sh --long [sealed-file]   # 4,096 tokens of context, about 25 minutes
#
# Short (2 discarded + 7 timed rounds of 256 tokens after a 5-token prompt):
#   lut   the fused path with the served Metal kernels (needs this branch: the int4 wall is lifted)
#   lutd  the same with tv_tetra48_metal_lutd, the Tetra matvec reading its activation unstaged (bit for bit)
#   D     the dense reconstruction through candle, the only way the sealed file decoded on Metal before
# Long (2 discarded + 5 timed rounds of 64 tokens after the first 4,096 tokens of WikiText-2's test split):
#   lutd      expanded attention, the only form the served Metal door accepts
#   lutd+grp  grouped attention, bit for bit with the expanded form on Metal (measurement arm, NOT SERVED)
#   lutd+ker  the split-KV attention kernel, attn_dec.metal, from position 0 (measurement arm, NOT SERVED)
# Each run prints a bit gate (253 of 253 records), each arm's median tok/s with its range, each arm over the first
# round by round, and whether each arm's tokens equal the first arm's. Quiet machine for a reading worth quoting: no
# build, no other GPU application. Nothing here is a served config.
set -euo pipefail
cd "$(dirname "$0")/.."
LONG=0
if [ "${1:-}" = "--long" ]; then LONG=1; shift; fi
SEALED="${1:-$HOME/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin}"
WANT_SIZE=1418224685
WANT_SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
[ -f "$SEALED" ] || { echo "no sealed file at $SEALED; fetch it: hf buckets cp hf://buckets/Pier-Jean/jobs-artifacts/sealed-4b-2026-09-23/qwen3-4b-sealed.bin <dest>"; exit 1; }
size=$(stat -f %z "$SEALED")
[ "$size" = "$WANT_SIZE" ] || { echo "$SEALED: $size bytes, want $WANT_SIZE"; exit 1; }
sha=$(shasum -a 256 "$SEALED" | cut -d' ' -f1)
[ "$sha" = "$WANT_SHA" ] || { echo "$SEALED: sha256 $sha, want $WANT_SHA"; exit 1; }
echo "sealed file checked: $SEALED, $size bytes, sha256 ${sha:0:8}"
nice -n 10 cargo build --release -p llvq-llm --features metal,fast-linalg --bin metalrun
if [ "$LONG" = 1 ]; then
  exec nice -n 10 target/release/metalrun time "$SEALED" configs/qwen3-4b-tetra-e4.json lutd,lutd+grp,lutd+ker 64 5 4096
fi
exec nice -n 10 target/release/metalrun time "$SEALED" configs/qwen3-4b-tetra-e4.json lut,lutd,D 256 7
