#!/usr/bin/env bash
# Stage the inputs of ops/jobs/inkling-first-card.sh in the bucket, and print the four pins the job takes.
#
#   bash ops/inkling/stage-first-card.sh <tokens.json>            checks, writes the manifest, prints what it would upload
#   GO=1 bash ops/inkling/stage-first-card.sh <tokens.json>       the same, then uploads (operator's token)
#
# <tokens.json>: {"prompt": [...], "reference": [...]}, the reference being the tokens of the Mac's resident run
# (inkrun --device metal, same prompt, same layers). What goes to hf://buckets/$BUCKET/$DEST/:
#   config.json         D/textonly/config.json, 1,408 B
#   model.safetensors   D/textonly/model.safetensors, 3,218,968,280 B: the 646 non-expert text tensors of the native
#                       file, same names, dtypes, shapes (checked against the native header by agent U, 2026-11-22)
#   tokens.json         the argument, a few KB
#   codes.sha256        sha256 of every codes file and VERIFIED, paths relative to layers/ (24 x 129 lines)
#   layers/Lxx.tar      Lxx/VERIFIED and Lxx/codes/e000..e127.llvqrec, about 259.6 MB each, 6.23 GB for 24
# 9.45 GB in all (computed from the files of 2026-11-22). Each archive is written to a scratch directory, uploaded and
# deleted before the next, so the Mac never holds more than one (260 MB).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
D=${INKLING_DATA:-$HOME/llvq-inkling-mini}
BUCKET=${BUCKET:-Pier-Jean/jobs-artifacts}
DEST=${DEST:-inkling2-mini-inputs-2026-11-22}
TOK=${1:?usage: stage-first-card.sh <tokens.json>}
GO=${GO:-0}
WORK=${WORK:-$D/lot2-u/stage}
mkdir -p "$WORK"

python3 -I - "$TOK" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))
assert isinstance(t.get("prompt"), list) and len(t["prompt"]) == 32, "prompt: 32 token ids expected"
assert isinstance(t.get("reference"), list) and len(t["reference"]) >= 32, "reference: the Mac's 32 tokens expected"
print("tokens file: prompt", len(t["prompt"]), "reference", len(t["reference"]))
PY
test "$(stat -f %z "$D/textonly/model.safetensors")" = 3218968280
for i in $(seq 0 23); do
  l=$(printf L%02d "$i")
  test -f "$D/layers/$l/VERIFIED" || { echo "refused: $l has no VERIFIED file" >&2; exit 1; }
  n=$(find "$D/layers/$l/codes" -name 'e*.llvqrec' | wc -l | tr -d ' ')
  test "$n" = 128 || { echo "refused: $l has $n codes files" >&2; exit 1; }
done
# the manifest: one line a file, `<sha256>  Lxx/...`, the format sha256sum -c reads on the card
( cd "$D/layers" && for i in $(seq 0 23); do l=$(printf L%02d "$i"); shasum -a 256 "$l/VERIFIED" "$l"/codes/e*.llvqrec; done ) \
  > "$WORK/codes.sha256"
test "$(grep -c . "$WORK/codes.sha256")" = $((24 * 129))
cp "$TOK" "$WORK/tokens.json"
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
echo "TEXTONLY_SHA=$(sha "$D/textonly/model.safetensors")"
echo "CONFIG_SHA=$(sha "$D/textonly/config.json")"
echo "TOKENS_SHA=$(sha "$WORK/tokens.json")"
echo "MANIFEST_SHA=$(sha "$WORK/codes.sha256")"
up() {
  if [ "$GO" = 1 ]; then hf buckets cp "$1" "hf://buckets/$BUCKET/$DEST/$2"; else echo "would upload $1 -> $DEST/$2 ($(stat -f %z "$1") B)"; fi
}
up "$D/textonly/config.json" config.json
up "$D/textonly/model.safetensors" model.safetensors
up "$WORK/tokens.json" tokens.json
up "$WORK/codes.sha256" codes.sha256
for i in $(seq 0 23); do
  l=$(printf L%02d "$i")
  if [ "$GO" = 1 ]; then
    COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata --format ustar -cf "$WORK/$l.tar" -C "$D/layers" "$l/VERIFIED" "$l/codes"
    up "$WORK/$l.tar" "layers/$l.tar"
    rm -f "$WORK/$l.tar"
  else
    echo "would archive $l (VERIFIED + 128 codes files) and upload it -> $DEST/layers/$l.tar"
  fi
done
echo "done; pass the four pins above to ops/jobs/inkling-first-card.sh"
