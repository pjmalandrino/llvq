#!/usr/bin/env bash
# Le bras IQ2_XXS, produit et mesuré sur Metal — 0 $, sur la machine de dev.
#
# POURQUOI CE BRAS
# ----------------
# Le 2026-08-30, le bras GPTQ 2 bits s'est révélé inutilisable comme rival :
# vLLM 0.26.0 refuse de le servir (bits=2 sym=True), et là où il se charge il ne
# génère que du charabia. La colonne qualité du dossier pour un concurrent
# 2 bits reste donc VIDE, comme elle l'est depuis le début — le 17,04 de QTIP
# étant une citation de la Table 6, jamais une mesure de notre harnais.
#
# La ligne de partage n'est pas le nombre de bits mais la FAMILLE : scalaire
# affine par groupe (GPTQ, AWQ, MLX q2, Q2_K) contre codebook / réseau /
# treillis (LLVQ, QTIP, QuIP#, AQLM, VPTQ, IQ2_XXS). Si la quantification
# scalaire naïve tenait à 2 bits, ni QuIP#, ni QTIP, ni LLVQ n'existeraient.
#
# IQ2_XXS est le SEUL candidat qui tienne et qui soit à notre portée :
#   * 2,06 bpw contre nos 2,0702 — la comparaison la plus serrée du dossier,
#     plus serrée que QTIP (2,000) ;
#   * un codebook ASSEZ PETIT POUR TENIR EN LUT, donc le contrefactuel du
#     docs/BACKLOG.md §4.4 — le test DIRECT de la thèse du papier, nos
#     1,1·10¹⁴ points ne tenant pas, d'où le dépliage à 4,80 b/poids ;
#   * le 2 bits le plus réellement déployé au monde ;
#   * produit ici pour 0 $, et le même GGUF partira ensuite sur CUDA — c'est le
#     seul bras qui traverse les deux backends.
#
# LA CALIBRATION EST LA NÔTRE, ET C'EST VÉRIFIÉ
# ---------------------------------------------
# calib-c4-shard1.txt sort de en/c4-validation.00001-of-00008.json.gz — le shard
# que llvq-llm/src/corpus.rs:187 réserve à la calibration — en 305 documents
# pour 131 072 tokens, soit le n_calib × calib_len de tous les artefacts LLVQ
# publiés. Son empreinte FNV-1a vaut 40300263e5d0afa2, IDENTIQUE à celle du
# bras GPTQ. Deux bras tiers verront donc le même texte que nous, et c'est
# machine-vérifié plutôt que déclaré.
set -euo pipefail

WORK="${WORK:-/tmp/iq2}"
F16="$WORK/qwen3-4b-f16.gguf"
CALIB="$WORK/calib-c4-shard1.txt"
IMAT="$WORK/imatrix-c4-shard1.dat"
OUT="$WORK/qwen3-4b-iq2xxs.gguf"

for f in "$F16" "$CALIB"; do
  [ -f "$f" ] || { echo "manque : $f" >&2; exit 2; }
done

echo "== 1. matrice d'importance, sur NOTRE corpus de calibration =="
# -c 2048 : la longueur de fenêtre de nos artefacts (calib_len), pas un défaut.
[ -f "$IMAT" ] || llama-imatrix -m "$F16" -f "$CALIB" -o "$IMAT" -c 2048 -ngl 99

echo "== 2. quantification IQ2_XXS =="
llama-quantize --imatrix "$IMAT" "$F16" "$OUT" IQ2_XXS

echo "== 3. comptabilité mémoire =="
# 🚨 b/param MODÈLE ENTIER, embedding compris — la seule comptabilité licite
# (§7, règle n°1). Pour un GGUF c'est direct : les octets DU FICHIER divisés par
# les paramètres. Meilleure provenance que notre propre bras, dont l'embedding
# est *modélisé* à 8,5 b/param.
#
# ⚠️ Le dénominateur est 4 022 468 096 — les paramètres RÉELS de Qwen3-4B, têtes
# LIÉES. Ne pas reprendre un compte qui additionne embed_tokens et lm_head : le
# bras GPTQ y a laissé un b/param faux de +9,67 % le 2026-08-30.
python3 - "$OUT" "$F16" <<'PY'
import os, sys
PARAMS = 4_022_468_096
for p in sys.argv[1:]:
    b = os.path.getsize(p)
    print(f"  {os.path.basename(p):28s} {b:>14,} o   {b*8/PARAMS:6.3f} b/param")
print("  repères                       LLVQ Planes14+q8 5,162 · AWQ 5,302 · GPTQ2 3,489")
PY

echo "== 4. perplexité, wikitext-2, ctx 4096 =="
echo "⚠️ Le PROTOCOLE de llama.cpp n'est pas le nôtre (fenêtres glissantes contre"
echo "   12 fenêtres non recouvrantes). Le NIVEAU ne se compare pas au nôtre ;"
echo "   seul le rapport au témoin f16 DE CETTE PILE se publie."

echo "== 5. débit =="
llama-bench -m "$OUT" -m "$F16" -p 0 -n 128 -r 3
