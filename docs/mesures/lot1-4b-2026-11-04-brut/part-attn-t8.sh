set -euo pipefail
O=/out/attn-t8-4b-2026-11-04
R2=/out/attn-bench-r2-2026-11-04
F=/out/sealed-4b-2026-09-23/qwen3-4b-sealed.bin
CG=/out/attn-t8-4b-2026-11-04/qwen3-4b-tetra-e4-grouped.json
CK=/out/attn-t8-4b-2026-11-04/qwen3-4b-tetra-e4-kernel.json
SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
BYTES=1418224685
CONF_GRP=$'{\n  "layout": "tetra48",\n  "embed": "q4",\n  "rot_share": "1",\n  "fuse": "0",\n  "kv": "f16",\n  "attention": "grouped",\n  "note": "The qwen3-4b-tetra-e4 object (qwen3-4b-sealed.bin, sha256 886391a8), served with the grouped attention: the cached key and value heads are read where they lie, on a Grow(256) store, with no repeat_kv copy. Decodes the 4B at 134.4 tok/s after a 5-token prompt and 77.5 at 8,192 of context against 121.9 and 33.1 for the expanded form (measured, gqa-ab-2026-11-01). Not bit for bit on the card past 5 tokens: licensed by the tolerance of ETAT section 4, paired WikiText-2 perplexity +0.0000 % by prefill and +0.0095 % [-0.0048 ; +0.0238] by decode against a band of +-0.12 % (measured, gqa-ppl-2026-11-01). Measured on the L40S and on this object only: the 8B and the 14B have never run this form, and neither has Metal, which refuses it by name."\n}'
CONF_KER=$'{\n  "layout": "tetra48",\n  "embed": "q4",\n  "rot_share": "1",\n  "fuse": "0",\n  "kv": "f16",\n  "attention": "kernel",\n  "note": "MEASUREMENT ONLY, written by ops/jobs/attn-bench.sh: the qwen3-4b-tetra-e4 object with decode rows sent to the split-KV attention kernel. Not licensed: the kernel is not bit for bit with the grouped form, and its T8 decision is the operator\'s."\n}'
CONF_GRP_SHA=2455fe858076217664f5e57083db0c6f5eb6ad3b16cdffa210265e41a0d42a42
CONF_KER_SHA=d74ea29abee6b585348f56d849aa09f1f27db1e09eab4bbd23bf2fe38aeef3fa
SRC_SHA=c7b0d8d3f2e3066ba1a2b4331c0bb6c92f22182c78a31b7fb74e839330a1838d
FLAGS=LLVQ_FUSED_LAYOUT=tetra48\ LLVQ_ROT_SHARE=1\ LLVQ_FUSE=0\ LLVQ_KV=f16\ LLVQ_EMBED=q4
DATASET_REV=740312add88f781978c0658806c59bc2815b9866
REPLAY_ROWS_SHA=f11a0b2386062f6a1b306784d6ddd7247e0da635fc0d77e921a759610cde2f01
REPLAY_FP=94c7b98876288901
CENSUS_FP=bfa9135ce4b55c75
ROWS_PY=$'import hashlib,json,sys\ndef rows(p):\n    ls=[json.loads(l) for l in open(p,encoding="utf-8").read().split("\\n") if l][1:]\n    return [d for d in ls if not d.get("end")]\nr=rows(sys.argv[1])\nif len(sys.argv)>2:\n    keep={d["index"] for d in rows(sys.argv[2])}\n    r=[d for d in r if d["index"] in keep]\nout=[]\nfor d in r:\n    d.pop("prefill_s",None); d.pop("decode_s",None)\n    out.append(json.dumps(d,sort_keys=True,ensure_ascii=False))\nprint(len(out), hashlib.sha256("\\n".join(out).encode("utf-8")).hexdigest())'
COND_PY=$'import os,re,sys\nd=sys.argv[1]\ndef sweep(arm,want):\n    p=os.path.join(d,"sweep-%s.txt"%arm)\n    t=open(p,encoding="utf-8").read().splitlines()\n    sc=[l for l in t if l.startswith("served config: ")]\n    if len(sc)!=1 or want not in sc[0]: raise ValueError("%s: no served config line naming %s"%(p,want))\n    r={}\n    for l in t:\n        m=re.match(r"^ +(\\d+) +[\\d.]+ \\[[^\\]]*\\] +([\\d.]+) \\[([\\d.]+)\\D+([\\d.]+)\\] +[\\d.]+$",l)\n        if m: r[int(m.group(1))]=tuple(float(m.group(i)) for i in (2,3,4))\n    if sorted(r)!=[5,256,2048,4096,8192]: raise ValueError("%s: lengths %s"%(p,sorted(r)))\n    return t,r\ntry:\n    tk,k=sweep("kernel","attention kernel on grow(256)")\n    tg,g=sweep("grouped","attention grouped on grow(256)")\n    if not any("the card equals the CPU replay bit for bit" in l for l in tk):\n        raise ValueError("the kernel arm printed no load-time identity line")\nexcept (OSError,ValueError) as e:\n    print("r2 unreadable: %s"%e); print("T8 condition: not met (r2 void or absent)"); sys.exit(0)\nprint("  prompt   kernel tok/s [range]     grouped tok/s [range]    medians k/g   in situ [min k/max g, max k/min g]")\nfor L in (5,256,2048,4096,8192):\n    (km,kl,kh),(gm,gl,gh)=k[L],g[L]\n    print("  %6d   %6.1f [%.1f-%.1f]   %6.1f [%.1f-%.1f]   %.4f        [%.4f, %.4f]"%(L,km,kl,kh,gm,gl,gh,km/gm,kl/gh,kh/gl))\nr5=k[5][0]/g[5][0]; u8=k[8192][2]/g[8192][1]\nc5=r5>=0.97; c8=u8>=1.0\nprint("stop line, 5 tokens: medians kernel/grouped %.4f, %s 0.97"%(r5,">=" if c5 else "<"))\nprint("stop line, 8,192: upper end of the in-situ range %.4f, %s 1.00"%(u8,">=" if c8 else "<"))\nprint("T8 condition: met" if c5 and c8 else "T8 condition: not met (the lead stop line)")'
BAND_PY=$'import re,sys\ninside=True\nfor p in sys.argv[1:]:\n    ls=[l for l in open(p,encoding="utf-8").read().splitlines() if re.match(r"^  decode +[\\d.]+ ",l)]\n    m=re.search(r"([+-][\\d.]+) % \\[([+-][\\d.]+) ; ([+-][\\d.]+)\\]",ls[0]) if len(ls)==1 else None\n    if not m: print("%s: no single decode summary line"%p); sys.exit(1)\n    mid,lo,hi=(float(x) for x in m.groups())\n    ok=lo>=-0.12 and hi<=0.12\n    inside=inside and ok\n    print("%s: kernel / grouped - 1 = %+.4f %% [%+.4f ; %+.4f], %s the band"%(p,mid,lo,hi,"inside" if ok else "NOT inside"))\nprint("band: both inside" if inside else "band: not both inside")'
export HF_HOME=/tmp/hf
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
echo '== STEP 0: the condition, read from the r2 sweep logs =='
python3 -c "$COND_PY" "$R2" | tee "$O/condition.txt"
if ! grep -qx 'T8 condition: met' "$O/condition.txt"; then
  echo 'T8 control NOT RUN: the condition of its prereg section 3 does not hold (outcome C0, not a failure)'
  exit 0
fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== gate 2: the object and both configs on the mount =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"
printf '%s\n' "$CONF_KER" > "$CK"
printf '%s\n' "$CONF_GRP" > "$CG"
sha256sum "$CK" "$CG" | tee -a "$O/files.sha256"
test "$(sha256sum "$CK" | cut -d' ' -f1)" = "$CONF_KER_SHA"
test "$(sha256sum "$CG" | cut -d' ' -f1)" = "$CONF_GRP_SHA"
TILE='tile 64 (served: measured optimum for sm_89)'
PARITY='int4: tv_q4_hh equals tv_q4_h bit for bit'
kernel_loaded() {  # log: the kernel text, built and checked against the CPU replay on the card
  grep -qF "attn_dec.cu: embedded, sha256 $SRC_SHA" "$1"
  grep -qE '^attn_dec_g4_u64: the card equals the CPU replay bit for bit on [0-9]+ calls' "$1"
}
for spec in 4096:12:60:30m 8192:6:6:35m; do
  IFS=: read -r C W FIRST CEIL <<< "$spec"
  P=$O/ppl-$C.txt
  echo "== STEP: decode-path perplexity, $W windows of $C from window index $FIRST, kernel against grouped =="
  date -u
  env $FLAGS LLVQ_GQA_PPL="$C,$W,$FIRST,kernel" timeout "$CEIL" fusedrun "$F" 2>&1 | tee "$P" | tail -8
  date -u
  grep -qF "$TILE" "$P"
  grep -qF "$PARITY" "$P"
  grep -qF 'cudarc event tracking: off (one stream, ordered by issue)' "$P"
  kernel_loaded "$P"
  grep -E '^GQA perplexity: ' "$P" | grep -qF "wikitext2, $W windows of $C from window $FIRST (tokens "
  grep -E '^GQA perplexity: ' "$P" | grep -qF 'against grow(256)+grouped, decode path in lockstep'
  test "$(grep -cE "^  window +[0-9]+/$W  decode nll " "$P")" = "$W"
  if [ "$C" = 4096 ]; then
    grep -E '^GQA perplexity: ' "$P" | grep -qF 'prefill path on 1 window(s), checked bit for bit'
    test "$(grep -cxF '  window   1: prefill path bit for bit between the two arms' "$P")" = 1
  else
    grep -E '^GQA perplexity: ' "$P" | grep -qF 'prefill path on 0 window(s)'
  fi
  grep -qF "kernel / grouped − 1, paired 95 % t interval over $W windows" "$P"
  grep -qE '^  decode top-1 agreement [0-9]+/[0-9]+ steps' "$P"
done
echo 'gates 3 and 4: both perplexity runs printed every line'
python3 -c "$BAND_PY" "$O/ppl-4096.txt" "$O/ppl-8192.txt" | tee "$O/band.txt"
if ! grep -qx 'band: both inside' "$O/band.txt"; then
  echo 'GSM8K NOT RUN: a perplexity interval leaves the band (outcome K2, read in the prereg)'
  echo '== raw kept ==' ; ls -la "$O"
  exit 0
fi
provenance() {  # log, wanted attention label
  grep -E '^served config: ' "$1" | grep -qF "$2"
  grep -qF "$TILE" "$1"
  grep -qF "$PARITY" "$1"
}
echo '== STEP: the grouped replay, the pilot'"'"'s 50 problems on this image =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$CG" LLVQ_GSM8K_DUMP="$O/gsm8k-4b-grouped-n50.jsonl" \
  timeout 10m gsm8k "$F" cuda 50 2>&1 | tee "$O/gsm8k-4b-grouped-n50.txt" | grep -vE '^  \[ *[0-9]+/' | tail -12
date -u
provenance "$O/gsm8k-4b-grouped-n50.txt" 'attention grouped on grow(256)'
tail -1 "$O/gsm8k-4b-grouped-n50.jsonl" | grep -q '"end":true'
tail -1 "$O/gsm8k-4b-grouped-n50.jsonl" | grep -qF "\"fingerprint\":\"$REPLAY_FP\""
GOT=$(python3 -c "$ROWS_PY" "$O/gsm8k-4b-grouped-n50.jsonl")
echo "replay rows: $GOT; the grouped census at the same indices: 50 $REPLAY_ROWS_SHA"
if [ "$GOT" != "50 $REPLAY_ROWS_SHA" ]; then
  echo 'GATE 5 FAILED: the grouped replay differs from the grouped census; the kernel census is not run' >&2
  exit 3
fi
echo 'gate 5: this image writes the grouped census rows'
echo '== STEP: GSM8K under the kernel config, 1,319 problems =='
date -u
LLVQ_DATASET_REV="$DATASET_REV" LLVQ_CONFIG="$CK" LLVQ_GSM8K_DUMP="$O/gsm8k-4b-kernel.jsonl" \
  timeout 80m gsm8k "$F" cuda 2>&1 | tee "$O/gsm8k-4b-kernel.txt" | grep -vE '^  \[ *[0-9]+/' | tail -16
date -u
provenance "$O/gsm8k-4b-kernel.txt" 'attention kernel on grow(256)'
kernel_loaded "$O/gsm8k-4b-kernel.txt"
tail -1 "$O/gsm8k-4b-kernel.jsonl" | grep -q '"end":true'
tail -1 "$O/gsm8k-4b-kernel.jsonl" | grep -qF "\"fingerprint\":\"$CENSUS_FP\""
echo 'gate 6: the kernel census ends on the census fingerprint'
echo '== raw kept ==' ; ls -la "$O"

