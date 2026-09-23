# Preregistration. A second training on the sealed 4B: row scales and RMSNorm weights

**Written on 2026-09-23, BEFORE the run, timestamped before the training job starts.**
Operator go, 2026-09-23: "Finetuning avec RMSNorm, MMLU full dense", and the served job
dropped. Announced cost: training ~$4.00 on l40sx1 (timeout 4 h, ceiling $7.20), scoring
~$0.72 (timeout 1 h, ceiling $1.80): **~$4.72**, priced on the 61.11 arm's two jobs
(`6aaeeddc`, 130 min, $4.00; `6aaf099f`, 24 min, $0.72). Project total before this arm:
$216.85.

This is not the draft `BROUILLON-preregistration-dclm-rownorms-2026-09-22.md`: that arm trains
from the untrained DCLM base. This one starts from the sealed object.

## 1. The claim on trial

The base is `qwen3-4b-sealed.bin` (1,418,224,685 B, sha256 `886391a8…`, 2.7320 b/param
served): the 61.11 object with `o_proj` and `down_proj@12-23` as int4 g128 records and the
embedding int4 g64. It reads **63.37** micro on the full split: the dense reconstruction is
the restore arm of `embed-q4-swap-2026-09-23` by construction, and identical to it on 57
questions, picks and logits (`preregistration-sealed-4b-2026-09-23.md` §5).

Its 168 `Tetra` row scales were trained on 2026-09-19 with `Tetra` `o_proj` and `down_proj`
in place. Those neighbours are now int4, and the restore dropped the trained scales of the
48 replaced matrices. The q4 embedding also moved the tied head. So:

  base   sealed file                                       63.37   2.7320 b/param
  arm    + second training, 168 row scales + 73 norms      **?**   **2.7320** (0 bits)

The arm trains, with `--mode row_norms` of `ops/llvqtune`: a multiplier per row of the 168
lattice matrices (946,176 values, on top of the trained scales), and a multiplier per value of
the 73 RMSNorm weights (`input_layernorm`, `post_attention_layernorm`, `model.norm`; 186,880
values). The 84 int4 records are never routed: `bin/export` lists them in `llvq-int4.json`
and the trainer excludes them by name. Everything else is the 61.11 recipe: KL at T = 1
against `Qwen/Qwen3-4B`, DCLM-edu streaming at seed 0 (the same stream the first training
read), 7,200 s of training on l40sx1, lr 3e-4, seq 1024, batch 2, bf16.

## 2. Signed prediction

| quantity | point | interval |
|---|---|---|
| paired gain over the sealed base (63.37) | **+0.8 pp** | [−0.4, +2.0] |
| micro, full split | 64.2 | [63.0, 65.4] |

Reasoning. The first training gave +3.15 from an untrained base; the row axis then sat on a
KL plateau (0.218 to 0.228). This run starts at a KL of **0.2607** (the Mac probe, below),
above that plateau, because the swap changed 48 matrices and the head after the scales were
fitted: there is KL to take back. The norms add 17 % more diagonal degrees of freedom, and
`model.norm` can absorb part of the q4 head's error. Against me: the KL-to-MMLU link is loose
on this object (the 61.11 file reads perplexity ×1.0074 of f16 and nine MMLU points under
it), and a second pass over the same 19 M tokens can fit them rather than the task. A loss
beyond −0.4 would mean the norms or the re-fit cost what the KL gained.

## 3. Decision rule

Census on both sides, constant codes, the 0.43 pp bar of `embed-q4-swap`. Δ is the paired
micro difference, arm minus the sealed base's dump
(`docs/mesures/embed-q4-swap-2026-09-23-brut/mmlu-4b-embed-q4-swap-FULL.csv`), p the exact
McNemar.

| outcome | reading | action |
|---|---|---|
| Δ ≥ +0.43 and p < 0.05 | the second training carries MMLU | the trained file becomes the 4B reference object, same bytes count, 0 bits |
| \|Δ\| < 0.43 or p ≥ 0.05 | nothing measurable | the sealed file stays the reference |
| Δ ≤ −0.43 and p < 0.05 | the training costs MMLU | not kept |

## 4. Controls, each a stop condition

1. **Wiring, on the Mac, before stamping** (*measured*, 2026-09-23): the trainer on the
   export of the sealed file prints `84 int4 records named by the export, never routed`,
   `mode row_norms: free: 0 added parameters`, `wiring accepted: 4096 tokens, 168
   matrices`; its export carries 168 `sigma` and 73 `tau`; `rowscale` folds it into the
   sealed file with every multiplier found, and the same export at all ones folds to a
   byte-identical file.
2. **Routing, on the card, automatic.** `train.sh` stops before training when the probe's
   `first_loss` leaves the band **[0.25, 0.27]** (`FIRST_LOSS_BAND`). The Mac probe read
   0.2607 on MPS in bf16, seed 0, the same first batch; the untrained base reads 0.35267.
3. **Inputs on the mount**: sha256 of the export's five files and of the trainer archive,
   equal to the Mac's, and 84 names in `llvq-int4.json`.
4. **Fold.** The all-ones export of the real run folds to a byte-identical file; the real fold
   prints 168 scaled, 84 int4 passed through, 73 norms of 73 named; the folded file is
   1,418,224,685 B.
5. **Scoring.** `oracle` MATCH; 14,042 questions, fingerprint `a74a6d6213602979`, dump headers
   dtype f16, limit census, alloc flat, config none, dense reconstruction, kv f16.

## 5. What it will not establish

- Which half did it: row scales against norms. One arm, both at once.
- The served path. The kernel reads the same bytes (0 bits moved), but no card run is in this
  arm; the served job was dropped by the operator.
- The spread of the training: one seed, one draw.
