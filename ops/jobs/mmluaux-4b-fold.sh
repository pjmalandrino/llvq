#!/usr/bin/env bash
# DRAFT, not launched. Folds the MMLU-format arm's trained scales into the
# sealed 4B and puts the object in the bucket. Local, no card, $0.
#
# Runs after ops/jobs/mmluaux-4b-rowscales.sh. The base is the same sealed file
# the 61.11 object came from — ~/qwen3-4b-dclm.bin, 1,794,564,765 bytes, the
# DCLM arm that reads 57.95 — so the two fine-tuned objects differ by their
# sigma and by nothing else.
#
# The size check is the whole point: row scales cost zero bits, so an output
# that is not byte-for-byte the input's size means the fold did something it
# must not.
set -euo pipefail
REPO=$HOME/Documents/Pro/workspace/poc/llvq
cd "$REPO"

D=${D:-2026-09-23}
BK=Pier-Jean/jobs-artifacts
RUN=mmluaux-4b-rowscales-$D
OBJ=mmluaux-4b-ft-$D
BASE=${BASE:-$HOME/qwen3-4b-dclm.bin}
SIGMA=$HOME/mmluaux-sigma-$D.json
OUT=$HOME/qwen3-4b-mmluaux-ft.bin

test -f "$BASE" || { echo "refused: $BASE missing" >&2; exit 1; }
BYTES=$(stat -f %z "$BASE")
[ "$BYTES" = 1794564765 ] || { echo "refused: $BASE is $BYTES B, not the sealed 1,794,564,765" >&2; exit 1; }

echo "== pulling sigma and the journal =="
hf buckets cp "hf://buckets/$BK/$RUN/sigma.json" "$SIGMA"
hf buckets cp "hf://buckets/$BK/$RUN/journal.jsonl" "$HOME/mmluaux-journal-$D.jsonl"
# The arm is only this arm if its journal says so.
python3 -c "
import json, sys
h = json.loads(open('$HOME/mmluaux-journal-$D.jsonl').readline())
assert h['corpus'] == 'mmlu-aux', h
assert h['steps'] == 9507, h
assert h['tokens_total'] == 19470336, h
print('journal:', {k: h[k] for k in ('corpus', 'steps', 'seed', 'tokens_total')})
"

echo "== folding =="
cargo run --release -p llvq-llm --bin rowscale -- "$BASE" "$OUT" "$SIGMA"
NEW=$(stat -f %z "$OUT")
[ "$NEW" = "$BYTES" ] || { echo "refused: the fold wrote $NEW B against $BYTES; row scales cost zero bits" >&2; exit 1; }
shasum -a 256 "$BASE" "$OUT" | tee "$HOME/mmluaux-$D.sha256"

echo "== uploading =="
hf buckets cp "$OUT" "hf://buckets/$BK/$OBJ/qwen3-4b-mmluaux-ft.bin"
hf buckets ls "hf://buckets/$BK/$OBJ/"
