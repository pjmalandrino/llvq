#!/usr/bin/env bash
# Stage what ops/jobs/inkling-gsm8k-card.sh needs beyond the first-card inputs: the checkpoint's tokenizer.json, in the
# same bucket directory. Prints the pin the job takes.
#
#   bash ops/inkling/stage-gsm8k-card.sh          checks and prints what it would upload
#   GO=1 bash ops/inkling/stage-gsm8k-card.sh     the same, then uploads (operator's token)
#
# tokenizer.json is D/native/tokenizer.json (about 26 MB), the file `llvq_llm::inkchat` was gated on
# (tests/inkling_chat.rs, tests/inkgsm.rs). The other inputs (config.json, model.safetensors, codes.sha256, layers/)
# come from ops/inkling/stage-first-card.sh, whose pins the job also takes.
set -euo pipefail
D=${INKLING_DATA:-$HOME/llvq-inkling-mini}
BUCKET=${BUCKET:-Pier-Jean/jobs-artifacts}
DEST=${DEST:-inkling2-mini-inputs-2026-11-22}
GO=${GO:-0}
TOK="$D/native/tokenizer.json"
test -f "$TOK" || { echo "refused: no $TOK" >&2; exit 1; }
# The template the renderer reproduces must sit beside the tokenizer it was gated with.
TPL_SHA=$(shasum -a 256 "$D/native/chat_template.jinja" | cut -d' ' -f1)
test "$TPL_SHA" = 53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc \
  || { echo "refused: chat_template.jinja is $TPL_SHA, not the one llvq_llm::inkchat reproduces" >&2; exit 1; }
echo "TOKENIZER_SHA=$(shasum -a 256 "$TOK" | cut -d' ' -f1)"
# The two int4 g64 tables of the sealed file (bin/inktable), which `inkseal verify` checked equal to the sealed copies.
TE="$D/lot1-b/tables/embed.q4g64.llvqraw"; TU="$D/lot1-b/tables/unembed.q4g64.llvqraw"
test -f "$TE" && test -f "$TU" || { echo "refused: the sealed tables are not under $D/lot1-b/tables" >&2; exit 1; }
echo "EMBED_TABLE_SHA=$(shasum -a 256 "$TE" | cut -d' ' -f1)"
echo "UNEMBED_TABLE_SHA=$(shasum -a 256 "$TU" | cut -d' ' -f1)"
if [ "$GO" = 1 ]; then
  hf buckets cp "$TOK" "hf://buckets/$BUCKET/$DEST/tokenizer.json"
  hf buckets cp "$TE" "hf://buckets/$BUCKET/$DEST/tables/embed.q4g64.llvqraw"
  hf buckets cp "$TU" "hf://buckets/$BUCKET/$DEST/tables/unembed.q4g64.llvqraw"
else
  echo "would upload $TE and $TU -> $DEST/tables/ ($(stat -f %z "$TE") and $(stat -f %z "$TU") B)"
  echo "would upload $TOK -> $DEST/tokenizer.json ($(stat -f %z "$TOK") B)"
fi
