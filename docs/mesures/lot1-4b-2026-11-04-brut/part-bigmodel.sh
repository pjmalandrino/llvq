set -euo pipefail
O=/out/bigmodel-identity-2026-11-04
F=/out/sealed-4b-2026-09-23/qwen3-4b-sealed.bin
SHA=886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8
BYTES=1418224685
mkdir -p "$O"
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* set in the job env' >&2; exit 1; fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tee "$O/gpu.txt"
echo '== gate 1: oracle (hard rule 10) =='
oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tee "$O/oracle-cuda.txt" | tail -2
grep -q MATCH "$O/oracle-cuda.txt"
echo '== the object on the mount: bytes and sha256 against the Mac =='
test "$(stat -c %s "$F")" = "$BYTES"
sha256sum "$F" | tee "$O/files.sha256"
test "$(cut -d' ' -f1 "$O/files.sha256")" = "$SHA"

echo '== STEP 2: rotbench, K0 and R0 against the host fixture =='
date -u
RB=0
rotbench 2>&1 | tee "$O/rotbench.txt" || RB=$?
date -u
echo "rotbench exit $RB" | tee -a "$O/rotbench.txt"
# gate 2: the card and both bounds
grep -qF '49152 B of shared memory per block by default, 101376 after opt-in' "$O/rotbench.txt"
# gate 3: four rotation entry points, none spilling (rotbench stops on a spill)
test "$(grep -cE '^  rot_apply(_rows)?(_dev)? +[0-9]+ registers, 0 local B' "$O/rotbench.txt")" = 4
# gate 4: no K0 arm at 25,600 or 31,744 (an explicit `if`: `! grep` never trips `set -e`)
if grep -E '^  .* (25600|31744) .*rot_apply/' "$O/rotbench.txt"; then echo 'gate 4 FAILED' >&2; exit 1; fi
# gate 5: the card drew the host's inputs (rotbench stops on the first that differs)
if grep -F 'INPUT sha256' "$O/rotbench.txt"; then echo 'gate 5 FAILED: input drift' >&2; exit 1; fi
echo 'gates 2-5 hold'

echo '== STEP 4: LLVQ_INT4_WIDE_CHECK on the sealed 4B =='
FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_EMBED=q4'
date -u
FR=0
env $FLAGS LLVQ_INT4_WIDE_CHECK=1 fusedrun "$F" 2>&1 | tee "$O/int4-wide.txt" | tail -20 || FR=$?
date -u
echo "fusedrun exit $FR" | tee -a "$O/int4-wide.txt"
# gate 7: the served load's own parity check, as always
grep -qF 'int4: tv_q4_hh equals tv_q4_h bit for bit' "$O/int4-wide.txt"
# gate 3, int4 half: tv_q4_hs does not spill
grep -qE '^  tv_q4_hs: [0-9]+ registers, 0 local B' "$O/int4-wide.txt"
echo 'gates 3 and 7 hold'
echo "== outcomes: rotbench exit $RB, int4 check exit $FR (0 = every comparison equal) =="
echo '== raw kept ==' ; ls -la "$O" ; wc -l "$O"/*.txt

