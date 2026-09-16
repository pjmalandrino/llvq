#!/usr/bin/env bash
# Is a lack of precision, somewhere specific, what caps Tetra?
#
# Seven arms of the SAME served Q5 file: the shipped object, then one arm per
# projection type taken back from the checkpoint at int4 g128. `LLVQ_RESTORE_Q4`
# does this on the sealed file WITHOUT re-encoding, which matters: two
# quantizations of identical settings differ by 6.6 % of perplexity and about
# 3 pp of MMLU (docs/mesures/errmap-4b-2026-09-15.txt), so any arm that
# re-encoded would drown a 1 to 2 pp effect in its own noise. At constant file
# that variance does not apply and the bar is the paired interval.
#
# The f16 version of this table exists (m2-attribution-4b-2026-09-02.txt) and
# gives each type's upper bound: gate +5.18, up +4.94, v +4.48, down +2.96,
# o +2.35, k +2.09, q +1.85 pp. int4 returns a fraction of it — on v_proj, the
# fraction that shipped as Q5 was about half.
#
# CONSTANT FILE IS NOT CONSTANT MEMORY. Each arm reconstructs densely, so the
# engine's VRAM says nothing about what the arm would cost as a served format.
# The surcharge of writing a type at int4 g128 instead of Tetra, computed on the
# 4B's own shapes (3,633,315,840 projection weights of 4,022,468,096 total,
# Tetra payload 2.1696 b/weight against int4 g128's 4.250), in b/param of the
# WHOLE model as hard rule 6 requires:
#
#   k_proj, v_proj   2.6 %  of weights   +0.0489 b/param   fits
#   q_proj, o_proj  10.4 %               +0.1954           fits
#   gate, up, down  24.7 %               +0.4641           DOES NOT FIT
#
# against a margin of 0.2355 (served 2.7645, b_max 3.00). Note k_proj + q_proj
# is +0.2443 and already over. The three big ones are measured anyway: the
# question is whether precision is the wall, not only which bits are affordable.
#
# RESTORE_Q4 reads the CHECKPOINT and round-trips it through the affine
# quantizer (`sealed.rs:545-551`), so it measures the information cost of four
# bits. It does not re-quantize the already-degraded Tetra weights, which would
# measure nothing.
#
# ⚠️ limit=40, so these dumps pair among themselves and with no census dump.
#
# ⚠️ FLAVOR: l4x1 with --any-flavor, because the whitelist holds l40sx1 alone.
# The guard exists for SPEED ratios, which are not comparable across cards; an
# MMLU score is a count of correct answers and is not a ratio. The flavor is
# named here and must be named in any published figure from this job.
set -euo pipefail
O=/out/q5-alloc-2026-09-16
F=/out/tetra-q5-2026-09-09/qwen3-4b-tetra-q5.bin
uv run ops/run.py bench \
  --image hf.co/spaces/Pier-Jean/llvq-runner-cuda \
  --flavor l4x1 --any-flavor --timeout 3h \
  --bucket Pier-Jean/jobs-artifacts --out-mount /out \
  --name q5-alloc-int4 \
  "mkdir -p $O" \
  'echo "== oracle (hard rule 10) ==" ; oracle Qwen/Qwen3-0.6B 64 cuda 2>&1 | tail -2' \
  "echo '== arm: shipped Q5, no restoration ==' ; date ; \
   LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-shipped.csv \
   mmlu $F cuda 40 2>&1 | tee $O/out-shipped.txt | tail -5 ; date" \
  "for T in q_proj k_proj o_proj gate_proj up_proj down_proj ; do \
     echo \"== arm: \$T restored at int4 g128 ==\" ; date ; \
     LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_RESTORE_Q4=\$T \
     LLVQ_MMLU_ALLOC=flat LLVQ_MMLU_DUMP=$O/mmlu-\$T.csv \
     mmlu $F cuda 40 2>&1 | tee $O/out-\$T.txt | tail -5 ; date ; \
   done" \
  "echo '== recap ==' ; grep -h 'MMLU (micro' $O/out-*.txt ; wc -l $O/mmlu-*.csv"
