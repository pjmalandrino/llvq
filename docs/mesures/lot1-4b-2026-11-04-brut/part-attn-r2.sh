set -euo pipefail
O=/out/attn-bench-r2-2026-11-04
F=/out/sealed-4b-2026-09-23/qwen3-4b-sealed.bin
CG=/out/attn-bench-r2-2026-11-04/qwen3-4b-tetra-e4-grouped.json
CK=/out/attn-bench-r2-2026-11-04/qwen3-4b-tetra-e4-kernel.json
SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
BYTES=1418224685
CONF_GRP=$'{\n  "layout": "tetra48",\n  "embed": "q4",\n  "rot_share": "1",\n  "fuse": "0",\n  "kv": "f16",\n  "attention": "grouped",\n  "note": "The qwen3-4b-tetra-e4 object (qwen3-4b-sealed.bin, sha256 886391a8), served with the grouped attention: the cached key and value heads are read where they lie, on a Grow(256) store, with no repeat_kv copy. Decodes the 4B at 134.4 tok/s after a 5-token prompt and 77.5 at 8,192 of context against 121.9 and 33.1 for the expanded form (measured, gqa-ab-2026-11-01). Not bit for bit on the card past 5 tokens: licensed by the tolerance of ETAT section 4, paired WikiText-2 perplexity +0.0000 % by prefill and +0.0095 % [-0.0048 ; +0.0238] by decode against a band of +-0.12 % (measured, gqa-ppl-2026-11-01). Measured on the L40S and on this object only: the 8B and the 14B have never run this form, and neither has Metal, which refuses it by name."\n}'
CONF_KER=$'{\n  "layout": "tetra48",\n  "embed": "q4",\n  "rot_share": "1",\n  "fuse": "0",\n  "kv": "f16",\n  "attention": "kernel",\n  "note": "MEASUREMENT ONLY, written by ops/jobs/attn-bench.sh: the qwen3-4b-tetra-e4 object with decode rows sent to the split-KV attention kernel. Not licensed: the kernel is not bit for bit with the grouped form, and its T8 decision is the operator\'s."\n}'
CONF_GRP_SHA=2455fe858076217664f5e57083db0c6f5eb6ad3b16cdffa210265e41a0d42a42
CONF_KER_SHA=d74ea29abee6b585348f56d849aa09f1f27db1e09eab4bbd23bf2fe38aeef3fa
LENGTHS=256\,2048\,4096\,8192
CONTEXTS=5\,256\,2048\,4096\,8192
SRC_SHA=c7b0d8d3f2e3066ba1a2b4331c0bb6c92f22182c78a31b7fb74e839330a1838d
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env (gate 4)' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee "$O/gpu.txt"
echo '== STEP 1: oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo "== STEP 2: attnbench $LENGTHS =="
date -u
attnbench "$LENGTHS" 2>&1 | tee "$O/attnbench.txt"
date -u
grep -qF "attn_dec.cu: embedded, sha256 $SRC_SHA" "$O/attnbench.txt"
test "$(grep -cE '^report attn_dec_g[45]_u(64|32): ' "$O/attnbench.txt")" = 4
test "$(grep -cE '^identity ' "$O/attnbench.txt")" -ge 4
test "$(grep -cE '^attn_dec_g[45]_u(64|32) +[0-9]+ +[0-9]+ ' "$O/attnbench.txt")" = 64
echo '== the object on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
echo '== the two configs, written from the launch command =='
printf '%s\n' "$CONF_KER" > "$CK"
printf '%s\n' "$CONF_GRP" > "$CG"
sha256sum "$CK" "$CG" | tee -a "$O/files.sha256"
test "$(sha256sum "$CK" | cut -d' ' -f1)" = "$CONF_KER_SHA"
test "$(sha256sum "$CG" | cut -d' ' -f1)" = "$CONF_GRP_SHA"
cat "$CK" "$CG"
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
TILE='tile 64 (served: measured optimum for sm_89)'
echo "== STEP 3: LLVQ_CONTEXTS=$CONTEXTS beside each config, 256 new tokens, kernel arm first =="
for arm in kernel grouped; do
  case $arm in
    kernel)  C=$CK; WANT='attention kernel on grow(256)' ;;
    grouped) C=$CG; WANT='attention grouped on grow(256)' ;;
  esac
  date -u
  env LLVQ_CONFIG="$C" LLVQ_CONTEXTS="$CONTEXTS" fusedrun "$F" 256 2>&1 \
    | tee "$O/sweep-$arm.txt" | tail -16
  date -u
  grep -qF "$TILE" "$O/sweep-$arm.txt"
  grep -qF "$PARITY" "$O/sweep-$arm.txt"
  grep -E '^served config: ' "$O/sweep-$arm.txt" | grep -qF "$WANT"
  test "$(grep -cE '^ +[0-9]+ +[0-9.]+ \[' "$O/sweep-$arm.txt")" = 5
  if [ "$arm" = kernel ]; then
    grep -qF 'the card equals the CPU replay bit for bit' "$O/sweep-$arm.txt"
  fi
done
echo '== raw kept ==' ; ls -la "$O" ; wc -l "$O"/*.txt

