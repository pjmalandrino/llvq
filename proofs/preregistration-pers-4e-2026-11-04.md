# Preregistration. 4E on the card: the whole decode token in one cooperative launch on the sealed 4B, its tokens against the eager path, and its decode rate at 5 and 8,192 prompt tokens

**NOT STAMPED. NOT LAUNCHED.** Written on 2026-11-04 on branch `claude/serene-bohr-1edf69`, once the code it describes
existed. To be timestamped before the first measurement, then never edited; a departure goes to
`preregistration-pers-4e-2026-11-04-ECARTS.md`.

Operator go: given for the night of 2026-11-04 by the audit session's relay ("termine tout tu as mon go pour la nuit");
the launch and the cap are the audit session's to set when it runs its lot. Cost: about **$0.30** on l40sx1
(*estimated*, §8). Timeout 40 min: **$1.20** at worst (*computed* at $1.80/h). Spent before it: see
`docs/data/jobs.csv` at launch.

## 1. Why this run exists

The operator's target is the sealed 4B at 300 tok/s or more after a short prompt on the L40S; llama.cpp IQ2_XXS reaches
312.85 on this card (*measured*, 2026-08-30). No ratio is formed across stacks (rule 5). 4E, one launch a token for N
tokens, is the route the persistent lead estimates at 339-394 tok/s short and 156-207 at 8k (persistent.md §1,
*estimated*). Every phase of 4E equals the served chain bit for bit on the host, and the whole token loop does too
(`docs/mesures/pers-l0-host-2026-11-04.txt`). This run asks two things of the card, in order:

- **identity**: 4E's tokens equal the eager path's, under the attention 4E owes its bits to (the "kernel" form, every
  decode row on the kernel); a divergence voids the run;
- **rate**: 4E's decode tok/s at 5 and 8,192 prompt tokens, beside the eager path's in the same rounds.

It is a measurement. No rule of §7 adopts or kills anything; serving 4E needs the attention T8 the operator has not
granted, and the A2 reading (persistent.md §7).

## 2. What the code holds at the commit that carries this file

Off by default; no served path moves.

- `llvq-cuda/kernels/pers_4e.cu`: the token loop, six symbols (RoPE form 0, 1, 2 × 768 and 512 threads), one grid
  barrier per phase boundary (2 + 7 × 36 = 254 a token, `pers.rs::n_bar_4e`), the argmax slot alternated by position,
  an end-of-sequence read by every CTA. Probes `pers_rope_probe_f0..2`.
- `llvq-llm/src/persist4e_cuda.rs`: builds 4E from the served runtime's own buffers, runs candle's rope on the card at
  install and keeps the first form that matches (refuses if none), takes 768 threads only where the loaded module
  reports no local memory (else 512), and decodes on the Grow caches in place (`KvCache::reserve`, `set_len`).
- `bin/fusedrun` `LLVQ_PERS_4E=<tokens>`: one model on Grow(256), the "kernel" attention form with every decode row on
  the kernel (`Qwen3::set_kernel_from_for_measurement(0)`, measurement only). Per context: the served grouped door's
  tokens (a T8 reading, printed), one eager prefill, the eager decode (the reference), 4E's decode after the same
  prefill, then 2 + 5 interleaved rounds of both decodes, the caches truncated back to the prompt before each.

Host checks at $0: 24 tests of the kernel text against the served text (the token loop among them), the struct images
offset by offset, the launch bookkeeping and the Grow cache API with their mutants; clippy clean on the Mac and on both
Linux halves (except two warnings in `attn_cuda.rs` inherited from `integration/lot1`). Never run on a card.

## 3. What runs

One image built from the commit that carries this file. One job, l40sx1, `ops/jobs/pers-4e.sh`, the sealed 4B
(`sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`) checked by size then sha256.

1. **Oracle** on Qwen3-0.6B (rule 10).
2. `LLVQ_FUSED_LAYOUT=tetra48 LLVQ_EMBED=q4 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_PERS_4E=64 fusedrun <sealed> 64`: 64
   tokens a run (the first from the prefill, 63 decoded), at the 5-token prompt and at 8,192 tokens (the prompt
   repeated), 2 discarded and 5 timed rounds.

## 4. Gates

Each voids the run:

1. The image builds; `oracle` prints MATCH.
2. The sealed file is at its bytes and sha256.
3. The process prints `cudarc event tracking: off (one stream, ordered by issue)` and `4E: RoPE forms equal to candle's
   rope on this card: [...]` with at least one form (otherwise 4E refuses to install and the job fails: rule A).
4. `4E tokens equal the eager path's` is printed for both contexts, and no round moves a token.
5. No `LLVQ_*` variable in the job environment.

## 5. Outcomes reported, never gated

- The RoPE forms that match candle on this card, and the one run.
- The block size taken (768 or 512) and every symbol's registers and local bytes from the loaded module.
- Per context: eager and 4E decode tok/s, median and range, and 4E / eager round by round (*measured*, same stack).
- The served grouped door's first divergence from the reference tokens, or none (a T8 reading, not this run's gate).
- Rule H of the L1 prereg read on the measured 4E rate directly: against 300 tok/s at the short prompt.

## 6. Signed predictions

*Estimated*. The design's 339-394 tok/s assumed 97 waits a token; this kernel passes 254 grid barriers a token, and
L1's barrier microbenchmark is not read yet.

| quantity | point | interval |
|---|---|---|
| RoPE forms matching candle | **one or more** | |
| 4E tokens against the eager path, both contexts | **identical** | |
| eager decode, kernel form, 5 tokens, tok/s | **125** | [100, 150] |
| 4E decode, 5 tokens, tok/s | **230** | [140, 330] |
| 4E / eager at 5 tokens | **1.8** | [1.1, 2.6] |
| eager decode at 8,192, tok/s | **85** | [60, 110] |
| 4E decode at 8,192, tok/s | **140** | [80, 210] |
| block size taken | **512** | (768 spills 36 B under the devel ptxas) |

Reasoning: 1.377 GB a token at 490-650 GB/s is 2.1-2.8 ms of stream (*computed*); 254 barriers at 1.3-2.5 µs add
0.33-0.64 ms; attention at 8k adds the kernel's measured cost at that length. The intervals are wide because neither
the persistent stream rate nor the barrier cost is measured on this card yet (L1 reads both).

## 7. Decision rules

- **A, RoPE.** If no form matches, 4E stays off, the job fails at install, and the next step is reading candle's rope
  SASS from the cubin to spell its form. Otherwise the matching form is the one 4E carries.
- **B, identity.** If the tokens are identical in both contexts, 4E keeps the eager path's bits on this card. Otherwise
  the run is void, and the first divergence's layer is found on the host with the card's dumped tensors (a later job).
- **C, rate.** If 4E at the short prompt is at or above 300 tok/s over its whole range, 4E reaches the target as built,
  and finer waits are a gain on top. If its whole range is under 250, the barrier count (254 a token) is read against
  L1's barrier cost first: if 254 × h explains the gap to 300, the next step is the design's 97 waits; otherwise the
  stream rate is. Otherwise the reading is a curve point for the operator.
- **G.** A miss is reported as a miss, with the model term it refutes. No threshold moves after the clock.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h: container and image 3 min, oracle 1-2 min, the load and install 1-2 min, the short
context under a minute, the 8,192 context two prefills (about 30 s each, *estimated*) and 14 decodes of 63 tokens:
about 10 min, **$0.30**. Timeout 40 min: **$1.20** at worst.

## 9. What this run does not establish

- Serving 4E: the attention T8 and the A2 reading are the operator's.
- 4E at 8B (a Tetra o, not in this version) or 14B (its config differs; the code accepts the shape, untested here).
- Any rate with finer waits than one barrier per phase boundary.
