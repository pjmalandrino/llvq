set -euo pipefail
O=/out/gesture-ab-4b-2026-11-04
F=/out/sealed-4b-2026-09-23/qwen3-4b-sealed.bin
CG=/out/gesture-ab-4b-2026-11-04/qwen3-4b-tetra-e4-grouped.json
SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
BYTES=1418224685
CONF_GRP=$'{\n  "layout": "tetra48",\n  "embed": "q4",\n  "rot_share": "1",\n  "fuse": "0",\n  "kv": "f16",\n  "attention": "grouped",\n  "note": "The qwen3-4b-tetra-e4 object (qwen3-4b-sealed.bin, sha256 886391a8), served with the grouped attention: the cached key and value heads are read where they lie, on a Grow(256) store, with no repeat_kv copy. Decodes the 4B at 134.4 tok/s after a 5-token prompt and 77.5 at 8,192 of context against 121.9 and 33.1 for the expanded form (measured, gqa-ab-2026-11-01). Not bit for bit on the card past 5 tokens: licensed by the tolerance of ETAT section 4, paired WikiText-2 perplexity +0.0000 % by prefill and +0.0095 % [-0.0048 ; +0.0238] by decode against a band of +-0.12 % (measured, gqa-ppl-2026-11-01). Measured on the L40S and on this object only: the 8B and the 14B have never run this form, and neither has Metal, which refuses it by name."\n}'
CONF_GRP_SHA=2455fe858076217664f5e57083db0c6f5eb6ad3b16cdffa210265e41a0d42a42
TABLES=1\ table\ \(model.embed_tokens.weight\,\ lm_head\ tied\ on\ it\)
REF_LINE_SHA=8784ffc423bfbeeaeb9a92abd426573cc45f9a083ba56e61e52d9ff40e3b018f
SERVED_TEXT_SHA=0597cee8b20befecdf92c2f3f2f1938d8216bd244542faab42d2637920a3c12a
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== the object on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo '== the served config, written from the launch command =='
printf '%s\n' "$CONF_GRP" > "$CG"
sha256sum "$CG" | tee -a "$O/files.sha256"
test "$(sha256sum "$CG" | cut -d' ' -f1)" = "$CONF_GRP_SHA"
cat "$CG"
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
TILE='tile 64 (served: measured optimum for sm_89)'
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_EMBED=q4'
echo '== STEP 2: the bench door replayed, 256 tokens against the dense arm, same process =='
date -u
env $FLAGS fusedrun "$F" 256 2>&1 | tee "$O/bench-q4-256.txt" | tail -30
date -u
grep -qF "$TILE" "$O/bench-q4-256.txt"
grep -qF "$PARITY" "$O/bench-q4-256.txt"
grep -F 'embedding: q4 g64 (LLVQ_EMBED)' "$O/bench-q4-256.txt" | grep -qF "$TABLES"
grep -qF 'fused F0: 256 tokens identical to the dense arm' "$O/bench-q4-256.txt"
test "$(grep -cF '  fused F0:  ' "$O/bench-q4-256.txt")" = 1
test "$(grep -F '  fused F0:  ' "$O/bench-q4-256.txt" | sha256sum | cut -d' ' -f1)" = "$REF_LINE_SHA"
echo 'gate 4: decoded line identical to paper-table run 1'
echo '== STEP 3: the served door, grouped, unchanged =='
date -u
env LLVQ_CONFIG="$CG" fusedrun "$F" 256 2>&1 | tee "$O/served-grouped-256.txt" | tail -20
date -u
grep -qF "$TILE" "$O/served-grouped-256.txt"
grep -qF "$PARITY" "$O/served-grouped-256.txt"
grep -E '^served config: ' "$O/served-grouped-256.txt" | grep -qF 'attention grouped on grow(256)'
grep -qF 'No dense arm on this path' "$O/served-grouped-256.txt"
test "$(grep -cE '^served : ' "$O/served-grouped-256.txt")" = 1
# The WHOLE decoded text, extracted as served-grouped.sh extracted the reference.
awk '/^served : /{f=1;next} /No dense arm on this path/{f=0} f' \
  "$O/served-grouped-256.txt" > "$O/line-grouped.txt"
test -s "$O/line-grouped.txt"
sha256sum "$O/line-grouped.txt" | tee "$O/lines.sha256"
test "$(sha256sum < "$O/line-grouped.txt" | cut -d' ' -f1)" = "$SERVED_TEXT_SHA"
echo 'gate 5: served text identical to the one of 2026-11-02'
echo '== STEP 4: LLVQ_GESTURE_AB=1, the served flags, 256 tokens =='
date -u
env $FLAGS LLVQ_GESTURE_AB=1 fusedrun "$F" 256 2>&1 | tee "$O/gesture-ab-256.txt" | tail -60
date -u
A=$O/gesture-ab-256.txt
grep -qF "$TILE" "$A"
grep -qF "$PARITY" "$A"
grep -qF 'cudarc event tracking: off (one stream, ordered by issue), before the load' "$A"
grep -qF 'attention grouped on grow(256), layout tetra48' "$A"
grep -qF "P1 tok_argmax: candle's fast_argmax_f16 index on 40 cases in sequence, NaN flag raised" "$A"
grep -qF 'P2 9 of 9 card mutants killed on the card, each edit found exactly once' "$A"
test "$(grep -cE '^  m[1-9] .*: [0-9] edit\(s\), each found once; killed, ' "$A")" = 9
grep -qE '^P3 kernel time, us a launch ' "$A"
grep -qE '^P4 host time behind a 5 ms spin, ms ' "$A"
grep -qF '5 arms × 5 interleaved rounds after 2 discarded, 256 identical tokens in every arm and round' "$A"
test "$(grep -cE '^ +[a-z-]+ / [a-z0-9]+ +[0-9.]+ \[' "$A")" = 5
grep -qE '^clock E, ms a decode step ' "$A"
grep -qE '^clock H/D: [0-9]+ of 315 spun steps valid' "$A"
echo 'gate 6: the A/B printed every line its prereg names'
echo '== raw kept ==' ; ls -la "$O" ; wc -l "$O"/*.txt

