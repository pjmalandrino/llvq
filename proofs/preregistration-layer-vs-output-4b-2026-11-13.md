# Preregistration. The layer judge against the final-output judge, by projection group, on the sealed 4B

**Written on 2026-11-13, BEFORE the job. NOT YET STAMPED.** The signed predictions (§9) are filled by the main
session, then the file is stamped (`ots stamp`) and committed before `ops/jobs/layer-vs-output-4b.sh` launches.
Once stamped, this file is not edited; a deviation goes into `-ECARTS.md` beside it.

Measurement script: `ops/layer_vs_output.py`, sha256
`c031c52f1294f3e044621bf7ed3e7f5089da0fb593f0ced5ca826fbb67596f12` at writing. The job script refuses to launch
unless this file names the sha256 of the script it uploads.
Job script: `ops/jobs/layer-vs-output-4b.sh`, sha256
`93cc6cc64794ca7b77c2bf3f0bd33f75ca848b470b55da3e2627f92df63cd6ab` at writing. If either changes before the stamp,
these lines are updated before the stamp.

Cost: **~$0.90 to $1.05** (*estimated*, §7), ceiling **$1.35** (timeout 45 min at $1.80/h on `l40sx1`), inside the
operator's $5 ceiling for this measurement and the $10 of O12 ($0 spent before it).

## 1. The question

The sealed 4B reads 63.37 MMLU against 70.14 for FP16. Restoring one projection group to f16 recovers, in MMLU points
(*measured*, `docs/mesures/ablation-qualite-4b-2026-11-12.txt` lines 85-92): gateup +3.38 [+2.75, +4.02], down +2.01
[+1.46, +2.54], qkv +0.55 [+0.01, +1.11], o −0.13 [−0.46, +0.19], attn +0.73, mlp +4.86, all +6.72. That is the truth
this measurement is read against.

The encoder minimises, per matrix, the error at the layer's output. The question: does a judge at the model's final
output (note 2) rank the four disjoint groups as the truth does, where the layer judge (note 1) does not? It is asked
at group grain only, because the truth exists only there. Cost note:
`docs/mesures/recherche-qualite-r2-chiffrage-2026-11-13.txt`.

## 2. The objects and the path

- W: `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`, the snapshot `int4swap` read and every ablation
  arm restored from (`ops/jobs/ablation-qualite-4b.sh` line 53). Three shards, each checked in the job against the
  Hub's LFS sha256: `328a91d3…` (3,957,900,840 B), `6cd087b3…` (3,987,450,520 B), `e4bf4369…` (99,630,640 B). Stored in
  bf16, cast to f16 at load, as `sealed.rs` does for the ablation's restored matrices.
- W_hat: `hf://buckets/Pier-Jean/jobs-artifacts/sealed-4b-export-2026-09-23/model.safetensors`, 8,044,981,648 B,
  sha256 `2a018cbdea37849c6fde29576e75aefb9125d9964e20e78bd3afb2a0948bba59`, f16. It is `llvq-llm --bin export` of
  `qwen3-4b-sealed.bin` (1,418,224,685 B, sha256 `886391a8…`, the file the ablation scored), 398 tensors checked bit
  for bit against the decode when it was written (`docs/mesures/sealed-rownorms-2026-09-23-brut/export.txt` lines
  14-15). It carries the int4 g64 embedding dequantized to f16. The job copies it off the mount, hashes the copy and
  refuses on mismatch.
- Tokenizer: the checkpoint's `tokenizer.json`, sha256 `aeb13307…`, identical to the export's.
- Text: WikiText-2 test, `Salesforce/wikitext` at `b08601e04326c79dfdd32d625aee71d232d685c3`,
  `wikitext-2-raw-v1/test-00000-of-00001.parquet`, sha256 `5f1bea06…`, rows joined by a blank line and tokenized
  without special tokens, as `llvq-llm/src/corpus.rs` lines 410-416 and `bin/ppl` do. Non-overlapping windows of 4,096
  from token 0. The job refuses unless the stream holds 299,078 tokens and the first 12 windows carry the fingerprint
  `3f1baca9033bf251` that `bin/ppl` printed (`ppl-scelles-2026-10-04-brut/ppl-4b.txt` line 4). Checked on the Mac on
  2026-11-13 from the cached files: both equal (`docs/mesures/layer-vs-output-4b-2026-11-13-brut/pre-launch/check-tokens.txt`).
- Inputs listed on 2026-11-13 with `hf buckets ls` and the Hub API: `pre-launch/buckets.txt`, `pre-launch/hub.txt`.
- Arithmetic: PyTorch 2.5.1 on the image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, f16 weights and
  activations, RMSNorm in f32, SDPA causal attention, log-softmax in f32. The forward is the script's own, not
  transformers': on a tiny random Qwen3 it equals `Qwen3ForCausalLM` (transformers 5.18.0) to max |dlogit| 0 in f32
  (`pre-launch/parity.txt`). It is not the candle forward the MMLU truth ran; rule D (§4) ties it to that path.
- Number of windows: fixed in the job from the first window's time, `min(73, max(12, floor(1200 s / t_1)))`, the first
  N windows, printed before the second window runs. It is not chosen after seeing any score.

## 3. The arms, per window

Groups as the ablation defined them: qkv (q, k, v; 108 matrices), o (36), gateup (gate, up; 72), down (36), attn
(q, k, v, o; 144), mlp (gate, up, down; 108), all (252). Every arm resolves each tensor from one of the two resident
sets; none writes a weight.

| arm | projections | embedding and norms | what it is |
|---|---|---|---|
| `T` | W | W | the teacher; its log-probabilities are kept for the window |
| `S` | W_hat | W_hat | the fully sealed export |
| `M_g`, 7 | W_hat, group g from W | W_hat | mirror form; `M_all` is the ablation's `all` arm |
| `D_g`, 7 | W, group g from W_hat | W | direct form; `D_all` is the 252 sealed projections on an FP16 body |
| `T2` | W | W | `T` again, after every other arm of the window |

Per arm and window: `KL(T || arm)` summed over the 4,095 next-token positions, full vocabulary (151,936), and the arm's
NLL. Note 1 is hooked in two passes: `T` (x from the FP16 model, variant a) and `S` (x from the sealed model, variant b).

## 4. Controls

If C0, C1, C4 or C5 fails, no number is published.

- C0. Inputs: the shard, export, tokenizer and parquet sha256s, the token count and the fingerprint of §2; the
  installed versions equal the script header's pins. The script refuses before any score otherwise.
- C1. `KL(T || T2)` is exactly 0 on every window.
- C2, C3, **rule D**. Perplexity on windows 1 to 12, `exp(Σ nll / Σ n)`, against the Rust dense f16 path on the same
  tokens: `T` against **12.2369** (`docs/mesures/a4-campagne-4b-ppl-BRUT-2026-08-06.txt` lines 39-57), `S` against
  **12.5834** (`docs/mesures/ppl-scelles-2026-10-04-brut/ppl-4b.txt` lines 4-19). Rule D: each within **±0.5 %**.
  Outside it, the run decides nothing: its numbers are reported as diagnostics and §6 is not read.
  Why ±0.5 %: it is the tolerance this repository already wrote for an f16 perplexity read by another stack on the
  same checkpoint and windows (`proofs/preregistration-duel-iq2-2026-11-08.md` line 100), where llama.cpp read −0.042 %
  (`docs/mesures/duel-iq2-2026-11-08.txt` line 29). The ±0.12 % of CERVEAU Q3 is a paired half-width within one stack
  and does not bound a change of stack. The ratio `S / T` is printed beside the Rust ratio 1.02832 and is not a gate.
- C4. The sha256 of every tensor of both sets is the same before the first window and after the last.
- C5. The selftest runs green in the job before the measurement (`pre-launch/selftest.txt` is the Mac run).
- Reported, not gates: how many non-projection tensors (embedding, 145 norms) are identical in W and W_hat; that no
  projection is identical; `KL(T || D_all)` beside `KL(T || S)`, which shows what the embedding and the norms carry.

## 5. The statistics

Notation: windows w = 1..N, `kl_a(w)` and `nll_a(w)` the per-token means of arm a on window w (all windows hold 4,095
positions, so window means and token means coincide).

### 5.1 Note 2

- Mirror, recovered KL of group g: `r_g(w) = kl_S(w) − kl_{M_g}(w)`; reported as its mean over windows. Share:
  `Σ_w r_g(w) / Σ_w kl_S(w)`.
- Direct, KL of group g: `kl_{D_g}(w)`, mean over windows.
- NLL forms, secondary: `nll_S − nll_{M_g}` and `nll_{D_g} − nll_T`.
- Intervals, over windows: normal, `mean ± 1.95996 · sd / sqrt(N)` with the n−1 sd; bootstrap, 10,000 resamples of
  the N windows with replacement, seed 20261113, percentile 2.5 and 97.5; the share's bootstrap resamples numerator
  and denominator together.
- Pairs for the rule of §6: `r_down(w) − r_b(w)`, where b is whichever of qkv and o has the larger mean on that note;
  the same for the direct form and the NLL forms.

### 5.2 Note 1

Per projection i: `N1_i = Σ_tokens ||(W_i − Ŵ_i) x||² / Σ_tokens ||W_i x||²`, over every token of the N windows (4,096
each), f32 arithmetic, x the input of projection i in pass `T` (variant a) or `S` (variant b). Per group g: pooled
`Σ_{i∈g} num_i / Σ_{i∈g} den_i`, and plain mean of `N1_i` over i ∈ g. Note 1 has no sampling interval here.

### 5.3 What is written

`windows.tsv` (window, arm, sums and means), `matrices.tsv` (252 rows, both variants), `groups.tsv` (seven rows, every
quantity of §5), `report.txt`, `inputs.txt` (pins, window count, base hashes before and after), `run.txt`,
`selftest.txt`, `pip-freeze.txt`, `gpu.txt`, in `hf://buckets/Pier-Jean/jobs-artifacts/layer-vs-output-4b-2026-11-13/`.

## 6. Decision rules, written before the data

Truth on the four disjoint groups (ablation journal): gateup +3.38 [+2.75, +4.02], down +2.01 [+1.46, +2.54], qkv +0.55
[+0.01, +1.11], o −0.13 [−0.46, +0.19]. The truth does not resolve qkv against o, so their order is not asked of a note.

**A note AGREES with the truth** if, among gateup, down, qkv and o, it puts gateup first and down second, both above
qkv and o (larger value = carries more), and the difference between down and the better of qkv and o is resolved:
- note 2: the normal paired 95 % interval of §5.1 on `down − b` excludes 0 (lower bound above 0); the bootstrap
  interval is reported beside it and does not decide;
- note 1, which has no interval over questions or windows: the ordering holds in both x variants.
By chance two of the 24 orders of four groups satisfy the ordering (checked by the selftest).

**Primary readings.** Note 2 = mirror form in KL (`r_g`). Note 1 = pooled ratio, read in both x variants. Every other
form and aggregate (direct KL, the two NLL forms, the plain mean of note 1, each x variant alone) is reported and
labelled secondary; none decides.

| outcome | condition | reading |
|---|---|---|
| **R-a** | note 2 agrees, note 1 does not | the final-output judge sees at group grain what the layer judge misses |
| **R-b** | both agree | no evidence at this grain that the layer judge is wrong |
| **R-c** | note 2 does not agree | the final-output judge measured on text does not follow MMLU at this grain; the phase does not build an encoder on it without another check |
| **R-d** | note 1 agrees, note 2 does not | reported as such; same consequence as R-c |

Rule D red: no outcome is read. The script prints the outcome mechanically; this table is what it means.

## 7. Cost

- Setup, *estimated*: pip install of four pinned packages, about 1 min; the checkpoint from the Hub, 8.0 GB, about 2
  min (the ablation's first restore arm spent 94 s more than the others on that fetch, *computed* from
  `ablation-qualite-4b-2026-11-12-brut/summary.tsv` and the last elapsed line of `out-qkv.txt` against `out-gateup.txt`);
  shard and export hashing and the export copy off the mount, 2 to 4 min; the two base hashes, about 1 min each.
  8 to 11 min in all; the row-scale job of 2026-09-23 spent 165 s outside its step log (*computed*,
  `sealed-rownorms-2026-09-23.txt` line 3 against its `journal.jsonl`).
- Window loop: budgeted at 1,200 s. One window is 17 forwards of 4,096 tokens plus two f32 note-1 passes and 16 full
  vocabulary KLs. At the Rust path's 12.5 s for 12 windows (`ppl-4b.txt` line 16), 1.04 s a forward, a window costs
  about 21 to 26 s, so N ≈ 46 to 57 (*estimated*); a faster torch forward gives up to 73.
- Total about 30 to 35 min billed, **$0.90 to $1.05** at $1.80/h (`ops/run.py` line 105). Ceiling: timeout 45 min,
  **$1.35**. `windows.tsv` is flushed after every window, so a timeout keeps the windows done.
- Queue time is not billed. The job goes into `docs/data/jobs.csv` with its billed minutes.

## 8. What does not compare

- KL and NLL on running text are not MMLU points; only orders are compared, never magnitudes.
- Four disjoint groups are four points: an indication, not a law. attn, mlp and all are reported, not ranked.
- One calibration draw: this is the sealed file's ranking, not the encoder's.
- Nothing here measures an encoder or a change to one. Restoring int4 records (all of o, part of qkv and down) measures
  their cost too, as in the ablation.
- Note 1's x is not the encoder's own x: the encoder saw a sealed prefix and its own block at f16 on its DCLM draw
  (`llvq-llm/src/calib.rs` lines 555-568). Here x is the FP16 or the sealed model on WikiText-2. W_hat is also no
  longer the encoder's minimiser: row scales were trained on KL after it, and the 84 int4 records are RTN of W.
- The arithmetic is torch, not the candle path of the truth; rule D bounds that, it does not remove it.

## 9. Dated disclosure, and the signed predictions

Run before the stamp, on the Mac, with no model of real size loaded:
- `ops/layer_vs_output.py --selftest` on a tiny random 36-layer Qwen3 on CPU: green, 23 checks
  (`pre-launch/selftest.txt`). Three mutants (the swap ignored in the resolver, note 1's denominator on W_hat, v_proj
  dropped from qkv), each at one counted site, each caught, the file restored to its sha256 (`pre-launch/mutants.txt`).
- `--parity` against transformers 5.18.0: MATCH (`pre-launch/parity.txt`).
- `--check-tokens` on the cached tokenizer and parquet: 299,078 tokens, fingerprint `3f1baca9033bf251`
  (`pre-launch/check-tokens.txt`). No score of any model was computed.

Known before the job, bearing on the outcome:
- The truth of §6 and its intervals; mlp +4.86, attn +0.73, all +6.72; groups restore super-additively across
  attention and MLP (+1.14 [+0.43, +1.85], ablation line 118).
- The answer-logit SNR of the restore arms, a final-output amplitude on the MMLU answer logits, fitted on the same
  questions that give the truth (`docs/mesures/recherche-qualite-n1-2026-11-13-brut/run.tsv` rows `gauss`): sealed
  1.5216, o 1.5662, qkv 1.7803, down 1.7839, attn 1.8263, gateup 2.2110, mlp 2.8914, all 11.1479. It already puts
  gateup above down and qkv, with down and qkv equal at 1.78. It is not this measurement: it reads four logits on exam
  questions, this reads the whole vocabulary on running text.
- The sealed 4B's KL to FP16 on DCLM-edu, 2,048-token batches, read 0.2622 at the first step of its second training
  (`docs/mesures/sealed-rownorms-2026-09-23.txt` line 50); the per-batch sd over the logged steps was about 0.054
  (*computed*, cost note section 4). Bare Tetra, an older object, read 0.628 on WikiText-2 (`kl-temperature-4b` line 16).
- Perplexity of the sealed 4B 12.5834 against 12.2369 for FP16 on these 12 windows.
- The layer objective tr(dW H dW') ranked six types against MMLU at Spearman +0.54, p 0.297, on an older object
  (`docs/mesures/hratio-4b-2026-09-18.txt` lines 34-36).

### Signed predictions (main session)

Signed by the main session on 2026-11-13, before the stamp and before any score. Orders are among gateup, down, qkv,
o, largest first. Confidence is low on every order between down and qkv, and it is said so here: the answer-logit
SNR of the restore arms leaves them equal, the truth puts down far above qkv, and nothing known says which way running
text reads. The shares are scaled from that SNR, as 1 / SNR², which is not this measurement. The note 1 orders rest on
one argument only: down's input, the gated hidden state, is less concentrated than the normalised residual stream the
other lattice records read, so its relative error is expected higher.

| quantity | prediction | interval or alternatives |
|---|---|---|
| order by note 2, mirror KL (primary) | gateup > down > qkv > o, down against qkv resolved | 55 %; else gateup > qkv > down > o |
| order by note 2, direct KL | gateup > down > qkv > o | 50 %; else gateup > qkv > down > o |
| order by note 2, mirror NLL | gateup > down > qkv > o, down against qkv not resolved | same alternative |
| order by note 2, direct NLL | gateup > down > qkv > o, down against qkv not resolved | same alternative |
| order by note 1, pooled, FP16 x (primary, with the next) | down > gateup > qkv > o | 35 %; else gateup > qkv > down > o |
| order by note 1, pooled, sealed x | down > gateup > qkv > o | same as the FP16 x variant |
| order by note 1, mean, FP16 x | gateup > down > qkv > o | low confidence |
| order by note 1, mean, sealed x | gateup > down > qkv > o | low confidence |
| outcome among R-a, R-b, R-c, R-d | R-a | about 40 %; R-c about 30 %, R-b about 15 %, R-d about 15 % |
| KL(T ‖ S), nats per token, fully sealed | 0.21 | [0.15, 0.30] |
| mirror KL share recovered, qkv | 24 % | [15, 33] |
| mirror KL share recovered, o | 5 % | [1, 10] |
| mirror KL share recovered, gateup | 50 % | [40, 60] |
| mirror KL share recovered, down | 25 % | [17, 33] |
| mirror KL share recovered, attn | 30 % | [20, 40] |
| mirror KL share recovered, mlp | 70 % | [60, 80] |
| mirror KL share recovered, all | 96 % | [90, 99.5] |

An order prediction is "in" when the computed order equals it; a share or KL prediction is "in" when the computed
value falls inside its interval. The count goes into the journal.
