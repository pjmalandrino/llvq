DRAFT, not stamped, not committed

# Preregistration. The sealed 8B and 14B on the fused Metal path: oracles, logits, tokens and decode speed on the M3 Max

**Draft written on 2026-11-02 by the wave-5 deep dive (lead "mac"), revised the same day against an adversarial
review. It runs only after `prereg-mac-1.md` reached its rule R2, and after the operator allows the download below.
Its predictions are re-derived from `prereg-mac-1`'s journal by the procedure of §5, written into this file, and then
the file is committed and TIMESTAMPED BEFORE the run.** Operator go: **not given**. Cost: **$0** of compute. The
download is about 7.9 GB from the Hub. Machine time is about 25 min, or 45-55 min with the optional arms (*estimated*,
§8).

## 1. Why this run exists

The product is a model of 14B or more, and no 8B or 14B timing exists on any Mac, fused or dense. The 14B's dense
reconstruction needs 29.54 GB (*measured* on the card, `paper-table-2026-09-25.txt`). The sealed file holds 5.04 GB by
the loader's host count, 2.73 b/param over 14,768,307,200 parameters (*computed*), so the Mac is where the format's
memory saving is the whole point. The tiled records carry 18.8 % and 19.5 % of the weights a token at 8B and 14B
(*computed*, wave-5 `mac.md` §3.2), against 11.5 % at 4B. If the int4 kernels or the tile are slow, this is where it
shows.

## 2. Preconditions

1. `prereg-mac-1.md` passed its gates and reached R2. Its R3 fixed the Apple tile T; the other tile is T'.
2. The operator allowed two downloads, each checked by sha256 before use, never by name:
   - `Pier-Jean/Qwen3-8B-LLVQ-Tetra-sealed`, the `down_proj@10-26` object, 2,815,098,745 B, sha256
     `7bdb9a5503518081b1f652a93215250898caae5cc1b6a2f33ee9fb6094565c13`;
   - `Pier-Jean/Qwen3-14B-LLVQ-Tetra-sealed`, 5,087,000,541 B, sha256
     `61db37fe7ce8e6a57c5c59f257319c796574981ecb5e450a02c73dd3c17e2ca0`.
   Both digests were written on this Mac on 2026-09-23 and 24 (`~/q8b-14b-sealed-2026-09-23/sha256-*.txt`). `hf
   buckets cp` stalled 3 times in 4 in October (`5c6f5e2`), so the route is the Hub client with resume.
3. Configs `configs/qwen3-8b-tetra-e4.json` (sha256 `6f0d489f24605712…`) and `configs/qwen3-14b-tetra-e4.json`
   (`833aec4c3a213c81…`), both expanded, the only form at those sizes.
4. The same binaries as `prereg-mac-1`, unchanged, at the same sha256, or rebuilt under a deviation that names why.

## 3. What runs

The conditions of `prereg-mac-1.md` §3 hold unchanged: a quiet machine, the load samples and their thresholds, the
binaries by path under `nice -n 10`, the environment refused by prefix, 30 minutes after the last build. Prompt `"The
capital of France is"`, 128 new tokens, greedy, two discarded and 7 timed rounds, the arms in the same order at every
round, tok/s = 128 / t over a whole `generate`, ratios round by round. Every arm goes through `Served::apply`.

1. `oracle` on Metal on the cached Qwen3-4B, as in `prereg-mac-1` step 1. Rule 10 certifies the backend, and two f32
   copies of the 8B would not fit.
2. **Check, 8B.** `metalrun check <8B file> <8B config>`: F with host copies at tile T, the record gate on 252
   projections and the head (the untied `lm_head` table), then D loaded for the logits gate G-L on the WikiText-2
   window, then released. About 22 GB of host count at the peak (*computed*: 2.76 + 16.38, plus host copies).
3. **Check, 14B.** The same on 280 projections and the head. About 35 GB at the peak (*computed*: 5.04 + 29.54, plus
   host copies), under the working set the process prints first. If the printed working set is below 38 GB, G-L at 14B
   is skipped and the deviation file says so.
4. **8B, timed.** F (tile T), F' (tile T'). On the operator's choice, also Fh (f16 tables) and D (dense), which bring the
   process to about 26 GB of host count (*computed*: 2.76 + 2.76 + 4.55 + 16.38). Without both Fh and D, no ratio is
   published (rule 4), only tok/s and GB/s.
5. **14B, timed.** F (tile T), F' (tile T'), 10.1 GB. This is the tile's main test: the tiled records are 19.5 % of the
   weights.
6. **14B, the ratios, optional.** On the operator's choice, a second process with F, Fh and D, about 41.9 GB of host
   count (*computed*: 5.04 + 7.28 + 29.54), run only if the printed working set exceeds 45 GB. Its true cost: D loads in
   about 4-6 min (*estimated* from the card's 263.9 s), and nine rounds take about 11 min at the predicted rates
   (*computed*: 128 tokens at 21, 11 and 2.3 tok/s). Its costs are about 17 min and 42 GB of unified memory. Without
   it, no 14B ratio exists (rule 4).
7. After each timed process, untimed: F's 256-token line, printed as `fusedrun` prints its F0 line and compared by `grep
   -F '  fused F0:  ' | shasum -a 256` with the line of the same file on the card
   (`paper-table-2026-09-25-brut/paper-served-{8b,14b}-2026-09-25-r2/fused-q4-256.txt`). Then `generate_phased` on F,
   32 tokens, off protocol.

## 4. Gates, each one voids the run

1. `oracle` prints MATCH.
2. Files, configs and binaries at their bytes and sha256.
3. Every `metalrun` process prints the environment check: no `LLVQ_*`, `CANDLE_*`, `MTL_*` or `METAL_*` variable.
4. Routing lines: 8B `int4: 53 records, 17 through tv_q4_metal_tiled at <tile> columns a tile, 36 through tv_q4_metal`;
   14B `int4: 99 records, 19 through tv_q4_metal_tiled at <tile> columns a tile, 80 through tv_q4_metal`.
5. Record gates: 252 of 252 projections and the head at 8B, 280 of 280 and the head at 14B, bit for bit against their
   host oracles.
6. G-L at each size where it runs: no fewer than 248 of 256 positions with the same top token.
7. F and F' give the same window logits, every f32 at every position, and the same 128 tokens in every round.
8. Each arm returns the same tokens in all its rounds.
9. Every arm prints `attention expanded on cat`.
10. The host byte counts read 2.76 and 5.04 GB, the card's own accounting, a check of the load.

## 5. Signed predictions

The values below are placeholders computed today from the bracket of wave-5 `mac.md` §3.2. **Before stamping**, two
inputs are refitted on `prereg-mac-1`'s journal. The in-situ int4 rate r4 is solved from F's measured step with the
other terms of `code/mac/mac_bounds.py` held. The non-weight rest comes from its phased pass. `mac_bounds.py` then
recomputes every point and interval of this table, and the recomputed values replace these before the stamp. If
`prereg-mac-1` ended in R2b, the refit uses its ratios and phased split only, and the absolute points keep the bracket.
**After the stamp nothing moves**, and no deviation file revises a prediction.

| quantity | placeholder point | placeholder interval |
|---|---|---|
| 8B F, tok/s | **37** | [26, 50] |
| 14B F, tok/s | **21** | [14, 30] |
| F'/F, T' over T, 8B and 14B | **0.97** | [0.92, 1.01] |
| 8B F and 14B F, GB/s of weights (2.414 and 4.606 GB a token) | **89 / 97** | [63, 121] / [64, 138] |
| 8B D, tok/s, if run | **4.2** | [3.0, 5.5] |
| 8B Fh, tok/s, if run | **16** | [8, 23] |
| 8B F/D raw and Fh/D same head, if both run | **×8.8 / ×3.8** | [×5.5, ×13.7] / [×1.7, ×6.3] |
| 14B D and Fh, tok/s, if step 6 runs | **2.3 / 11** | [1.8, 2.9] / [6, 16] |
| 14B F/D raw and Fh/D same head, if step 6 runs | **×9.1 / ×4.7** | [×5.4, ×14.8] / [×2.3, ×7.9] |
| G-L agreement, 8B and 14B | **254 / 253 of 256** | [249, 256] / [248, 256] |
| 8B F against D, greedy tokens, if D is timed | **identical through 128** | |
| 14B F against D, greedy tokens, if step 6 runs | **may diverge** | the card diverged at token 78 (`paper-table-2026-09-25.txt`) |
| 8B and 14B, F's 256-token line against the card's | **8B equal, 14B equal** | a divergence, if any, after token 64 |

Reasoning: `mac.md` §3.2 and `mac_bounds.py`. Tetra at 75-120 GB/s at these shapes (more blocks a lane than at 4B),
int4 at 75-250, the q4 head at 150-250, 127 and 101 rotations a token at the fitted cost (*measured* counts on the card,
`fused-q4-256.txt:3` of the 8B and 14B paper-table runs), the non-weight rest scaled by layers. The f16-head arms pay the
copy of `mac.md` §3.2 on untied tables of 1.245 and 1.556 GB. The tile ratio leans lower than at 4B because the tiled
records weigh more here. The 14B's divergence on the card is why its token row predicts nothing firm.

## 6. Decision rules

- **R1.** A gate fails: void, journaled, fixed under a deviation, rerun.
- **R2.** Every gate passes and every load sample is within its threshold: the journal carries the first 8B and 14B Mac
  speeds, labelled "niced, on a quiet machine", and their GB/s beside llama.cpp's anchor with no ratio across stacks.
  Raw and same-head ratios appear only where D and Fh ran in the same process, and always together.
- **R2b.** Every gate passes and a load sample breaks a threshold: ratios only, absolute rates void outside the journal,
  one repeat allowed under a deviation.
- **R3, the tile.** F'/F wholly above 1.000 at both sizes: the operator may move the Apple row that `prereg-mac-1` set.
  Any other outcome: the row stays.
- **R4.** The 14B F median is under 12 tok/s: the number is journaled, no sentence beyond it leaves the journal, and
  the attribution pass (`mac.md` L6) runs before any further Metal work at 14B.
- **R5.** G-L passes and the 14B leaves the dense line before token 32: reported and diagnosed per record. Not by itself
  a defect, since the card diverged at token 78.
- **R6, framing.** As `prereg-mac-1` R7: no ratio against card figures, both attention forms and both machines named
  beside any card figure.
- **Otherwise**, not settled: the operator decides. No outcome kills a lead.

## 7. Deviations

Any departure goes in `proofs/preregistration-mac-2-<date>-ECARTS.md`, beside this file once committed. A deviation
records a protocol change, never a revised prediction.

## 8. Cost

*Estimated*. $0 of compute, and about 7.9 GB of download before the run. Oracle 2-4 min. Check 8B 4-5 min (D loads in
60-150 s). Check 14B 7-9 min (D loads in 4-6 min, scaled from the card's 263.9 s). 8B timed with F and F' about 2.5 min
with loads, or 9-11 min with Fh and D. 14B timed with F and F' about 3.5 min. Step 6, about 17 min. In all about 25 min,
45-55 with the optional arms. Unified memory peaks at about 35 GB in the 14B check and 42 GB in step 6, under the working
set printed first.

## 9. What this run does not establish

Quality beyond one window at each size, long context, prefill, the grouped attention, and any card other than this M3
Max. Whether Mac readings answer the local-hardware item of ETAT §5 stays an operator ruling.
