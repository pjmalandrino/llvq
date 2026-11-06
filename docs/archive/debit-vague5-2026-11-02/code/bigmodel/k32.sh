#!/usr/bin/env bash
# DRAFT, never run. Written read-only on 2026-11-02 for lead "bigmodel"
# (prereg-bigmodel-1.md, gate K32), revised the same day after review. Not
# part of the repository. A LOCAL MEASUREMENT RUN on the operator's Mac: it
# needs the operator's explicit go (CLAUDE.md rule 1), the ruling that
# reopens rival GGUF arms, and a stamped prereg before the first question.
# It assumes the commit carrying the prereg: ops/vllm_score.py writes the
# `# end` trailer and gguf_mmlu.py checks the run fingerprint before scoring
# (draft emit_dump_trailer.py), ops/k32_project.py and ops/k32_pilot.py exist.
#
#   STAGE=P0 bash k32.sh    pilot: 4B BF16, 100 questions; gate P0
#   STAGE=A  bash k32.sh    E14 (BF16) then R14 (UD-IQ2_M) on the plan; prints the case
#   STAGE=B  bash k32.sh    R32, refused unless k32_project.py says stage B runs
#
# Per arm, one at a time, machine otherwise idle, under nice:
#   1. download into the arm's own new directory AT THE PINNED REVISION, then
#      check bytes and LFS sha256 against the prereg's table before anything
#      starts; a file whose sha256 differs is not scored;
#   2. llama-server with the 2026-08-30 settings (-np 4 needs -c 8192 so each
#      slot holds a few-shot prompt, m3-iq2-metal-2026-08-30.txt);
#   3. ready = /health AND one completion that returns top_logprobs (the
#      2026-09-19 card attempt scored 9 minutes against an unloaded server);
#   4. score with ops/gguf_mmlu.py; record the wall time;
#   5. stop the server; keep dump, server log, provenance, wall time.
set -euo pipefail

REPO=/Users/pjmalandrino/Documents/Pro/workspace/poc/llvq
OUT=${OUT:-$HOME/llvq-k32}            # outside the repository; the journal copies from here
PORT=8080
PLAN=$REPO/docs/data/mmlu-dumps/mmlu-4b-f16.csv
STAGE=${STAGE:?set STAGE to P0, A or B}
mkdir -p "$OUT"

# arm  repo  revision  file  bytes  lfs-sha256      (Hub API, read 2026-11-02)
P0="P0bf16 unsloth/Qwen3-4B-GGUF 22c9fc8a8c7700b76a1789366280a6a5a1ad1120 Qwen3-4B-BF16.gguf 8051285536 f59dbdf0a0131f4bafe257e5eaf25137a88c89c7a009e9418b0e447accda58a8"
E14="E14bf16 unsloth/Qwen3-14B-GGUF a04a82c4739b3ef5fa6da7d10261db2c67dd1985 Qwen3-14B-BF16.gguf 29543424160 1d12adb9b56bea81a1b3b8b2eea0fcf5700307ba4ed5d56d2f64f222ec2d7b72"
R14="R14iq2m unsloth/Qwen3-14B-GGUF a04a82c4739b3ef5fa6da7d10261db2c67dd1985 Qwen3-14B-UD-IQ2_M.gguf 5425956224 3c4aa2f82f9f9c9b47ebbda4dc384d7a0fa3031f4d0ae4241e0fa6420aa0d9b1"
R32="R32iq2m unsloth/Qwen3-32B-GGUF 931c84066f88693a02ab8de820cfcd066d913241 Qwen3-32B-UD-IQ2_M.gguf 11553366688 6d78bba94fa70f54d8e2c85ee1648ca41e49bffbdcda2abd63f0a16714578377"

REFS=(
  "$REPO/docs/mesures/sealed-8b-14b-2026-09-23-brut/mmlu-14b/mmlu-14b-sealed-FULL.csv"
  "$REPO/docs/mesures/census-14b-ref-2026-09-22-brut/mmlu-14b-awq-FULL.csv"
  "$REPO/docs/data/mmlu-dumps/mmlu-14b-f16.csv"
)

score_arm() {   # $1 = arm line, $2 = reference dump (the plan, or the pilot's 100 rows)
  read -r ARM HFREPO REV FILE BYTES SHA <<<"$1"
  local DIR="$OUT/$ARM"
  [ -e "$DIR" ] && { echo "refused: $DIR exists; a rerun gets a new name in -ECARTS.md" >&2; exit 2; }
  mkdir -p "$DIR"
  echo "$ARM $HFREPO@$REV $FILE $BYTES $SHA" | tee "$DIR/provenance.txt"
  hf download "$HFREPO" "$FILE" --revision "$REV" --local-dir "$DIR"
  local GOT; GOT=$(stat -f %z "$DIR/$FILE")
  [ "$GOT" = "$BYTES" ] || { echo "$ARM: $GOT bytes, expected $BYTES (gate 2)" >&2; exit 2; }
  local H; H=$(shasum -a 256 "$DIR/$FILE" | awk '{print $1}')
  echo "sha256 $H" | tee -a "$DIR/provenance.txt"
  [ "$H" = "$SHA" ] || { echo "$ARM: sha256 $H, expected $SHA (gate 2)" >&2; exit 2; }

  nice -n 10 llama-server -m "$DIR/$FILE" -ngl 99 -np 4 -c 8192 --port "$PORT" \
      >"$DIR/server.log" 2>&1 &
  local SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  local ok=0
  for _ in $(seq 1 900); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null &&
       curl -sf "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
         -d '{"prompt":"Answer:","n_predict":1,"n_probs":4}' | grep -q top_logprobs; then
      ok=1; break
    fi
    sleep 2
  done
  [ "$ok" = 1 ] || { echo "$ARM: server never answered a real completion" >&2; exit 3; }

  local T0; T0=$(date +%s)
  ( cd "$REPO" && nice -n 10 uv run --with transformers --with datasets \
      ops/gguf_mmlu.py --arm "$ARM" --label "$HFREPO@$REV/$FILE" \
      --reference-dump "$2" --out "$DIR/mmlu-$ARM.csv" ) 2>&1 | tee "$DIR/score.log"
  echo "wall_s $(( $(date +%s) - T0 ))" | tee -a "$DIR/provenance.txt"
  kill "$SRV"; wait "$SRV" 2>/dev/null || true
  trap - EXIT
}

case "$STAGE" in
  P0)
    python3 -I "$REPO/ops/k32_pilot.py" write "$PLAN" "$OUT/pilot-plan.csv"
    score_arm "$P0" "$OUT/pilot-plan.csv"
    python3 -I "$REPO/ops/k32_pilot.py" compare "$OUT/P0bf16/mmlu-P0bf16.csv" \
        "$REPO/docs/data/mmlu-dumps/mmlu-4b-gguf-f16.csv" "$PLAN"
    ;;
  A)
    [ -f "$OUT/P0bf16/mmlu-P0bf16.csv" ] || { echo "refused: P0 has not run" >&2; exit 2; }
    score_arm "$E14" "$PLAN"
    score_arm "$R14" "$PLAN"
    python3 -I "$REPO/ops/k32_project.py" stageA "$PLAN" "${REFS[@]}" \
        "$OUT/E14bf16/mmlu-E14bf16.csv" "$OUT/R14iq2m/mmlu-R14iq2m.csv" | tee "$OUT/stageA.txt"
    echo "authoritative paired numbers (mmlupair):"
    ( cd "$REPO" && cargo run --release -p llvq-llm --bin mmlupair -- \
        docs/data/mmlu-dumps/mmlu-14b-f16.csv "$OUT/E14bf16/mmlu-E14bf16.csv" ) | tee -a "$OUT/stageA.txt"
    ( cd "$REPO" && cargo run --release -p llvq-llm --bin mmlupair -- --intersect \
        "${REFS[0]}" "$OUT/R14iq2m/mmlu-R14iq2m.csv" ) | tee -a "$OUT/stageA.txt"
    ;;
  B)
    grep -q "stage B (R32) runs" "$OUT/stageA.txt" ||
      { echo "refused: stage A did not end with lo > 0 and gate 3 passed" >&2; exit 2; }
    score_arm "$R32" "$PLAN"
    python3 -I "$REPO/ops/k32_project.py" stageB "$PLAN" "${REFS[@]}" \
        "$OUT/E14bf16/mmlu-E14bf16.csv" "$OUT/R14iq2m/mmlu-R14iq2m.csv" \
        "$OUT/R32iq2m/mmlu-R32iq2m.csv" | tee "$OUT/stageB.txt"
    ;;
  *) echo "STAGE must be P0, A or B" >&2; exit 2 ;;
esac
# The `cargo run` of mmlupair builds on the Mac: run it with the house's
# `nice -n 15` and `-j 4` if the build is not already there (passation
# 2026-11-01, "ne pas saturer la machine").
