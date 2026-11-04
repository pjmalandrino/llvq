# Lead "speculation": exact speculative decoding on the sealed targets, and raising τ on chat

Deep dive of 2026-11-02, read-only, revised the same day after an adversarial review (§10 lists each finding and what
changed). No repository file changed, no run, $0. Object: sealed Qwen3-4B (`886391a8`), served under the grouped
config (`configs/qwen3-4b-tetra-e4-grouped.json`, sha256 `2455fe85`), L40S for the engine, the operator's M3 Max
(64 GB, 16 cores, *measured* by `sysctl`) for G0. K is the number of verify rows (anchor plus proposals); τ is tokens
emitted per verify round, bonus included; t(k) is the time of a k-row exact verify; β = (t(k)/t(1) − 1)/(k − 1), the
cost of one extra row in plain steps; d = draft time/t(1); S = τ/(r + d) with r = t(k)/t(1). q_τ = τ against the
sealed target over τ against Qwen3-4B (BF16 checkpoint run in f16), same drafter, same corpus, greedy.

## 1. Verdict

Exact speculation on the sealed 4B is sound by code reading, and today's engine can run it with no new kernel. It
does not pay yet. Through today's rows path an extra verify row costs β ≈ 0.50 of a step at 5 positions, 0.58 at
2,048 and 0.71 at 8,192 (t(4)/t(1) ≈ 2.5, 2.75 and 3.15, *estimated*, §3.3). At those costs prompt lookup gives
×1.05 on extraction decode at 2k context, ×0.96-1.34 over every input range. That is ×1.01 on a 2,048 + 300 request
today, where 8.20 s go to the first token (*measured*), and ×1.04 once the GEMM prefill route lands (*computed* on
estimated inputs). Chat stays at ×1.0 with gating, and 8k at ×0.94-1.14. So build step B1 alone buys an identity
proof and plumbing, not speed.

The gain needs a cheaper verify row. With B2 (int4 rows kernel, head rows kernel, no `regroup`, bits kept), β at 2k
falls to about 0.39 and prompt lookup gives about ×1.15. With B3 (one KV read for K rows), β falls to about 0.16 and
prompt lookup gives about ×1.42 (both *estimated*). Neural drafters (DeepSpec DSpark or DFlash, block 7) pay on chat
only with a flat verify, β ≈ 0.03: chat ×1.7-2.9 and code ×2.5-4.3 over that engine's own step, if q_τ holds
(*estimated*).

Four unknowns decide:
- q_τ, the drafters' acceptance against the sealed target: 0.80 [0.65, 0.95] on chat, *estimated*. G0 measures it.
- The rows path's t(k): measurable today with no code, through the `LLVQ_PREFILL_TOKENS` sweep (L1a, about $0.1).
- Row-invariance on the card. Two of the four risks are gone by design: the batched attention and the `Grow`
  stride. Three checks remain: the trailing multiply-add of the Tetra rows kernel, `rot_apply_rows`, and int4
  parity on all 84 records.
- The drafter's rule-6 bill: its KV is 168 MB at 8k in f16 (*computed*), on top of its weights.

First steps, both cheap and both before any engine code: L1a and the G0 pilot. B1 also waits on two rulings: R1 (the
speculative verdict of 2026-08-05) and R8 (the ROADMAP §4 hold on the multi-row path).

## 2. What exists today

Engine, branch `debit-trims` at `06e1602` (`llvq-llm/src/model.rs` unless named):

| piece | where | state |
|---|---|---|
| greedy loop | `generate` :2119-2151, `prefill_chunked` :2191-2203, `step_logits` :2207-2211, `decode_greedy` :2216-2232 | one token a step, host argmax (`argmax_last` :2390); `decode_greedy` has no stop token, fixed length |
| rollback | `KvCache::truncate` :421-447 | O(1) on `Grow`/`Prealloc` (moves `len`, keeps the capacity), O(L) copy on `Cat` |
| store per form | `served.rs:221-229`, `kvq.rs:414-427` | `grouped` gives `Grow(256)` (4B grouped config only); `expanded` gives `Cat` (8B, 14B, older configs) |
| attention | `forward_cached` :1647-1689, `grouped_attention` :571-593, `rows_contiguous` :490-499 | q viewed `[b, n_kv, groups·l, hd]`; a `Grow` view goes to cuBLAS uncopied, batch stride = capacity |
| masks | `step_mask` :1859-1866 | none at l = 1 on `Grow`, causal offset mask at l > 1 |
| K-row projections | `group_forward` :1132-1208, `prepare_rows` :855-872, `forward_rows` :915-964 | Tetra rows kernel, 4 rows a launch; a 1-row chunk takes the one-row path (:857, :923) |
| int4 rows | o_proj and down_proj@12-23 via `int4_rows` :1213-1228; v_proj via `forward_rows`'s `_` arm :957-962 | both reach `forward_int4` (`fused_cuda.rs:930-959`): an f16 row at offset 0 takes `tv_q4_hh`, any other row an f32 cast plus `tv_q4_h`; 84(2k−1) launches a k-row call |
| int4 parity | `check_int4_half_parity`, `fused_cuda.rs:2224-2260`, called at :817 | at load, on the first record only; refuses the load on a mismatch |
| copies | `regroup` :1102-1106 | `Tensor::cat`, 252k copies a k-row pass, 0 at decode |
| head | `Head::project` :1393, `HeadOp` `fused_cuda.rs:1282-1351` | one launch per row with `x_off` (:1330-1343), each reading the 218.8 MB int4 table |
| taps | `Capture::on_activation` :87, `Act` :38-49 | `Attn`, `AttnOut`, `Mlp`, `MlpOut`; no block-output tap, which is what the drafters read |
| measurement modes | `bin/fusedrun.rs:676-688` (refusal list), `LLVQ_PREFILL_TOKENS` :947-1118 | the one mode beside `LLVQ_CONFIG` that times a k-row forward of the served object |

**Multi-row evidence on the card** (*measured*, `docs/data/jobs.csv` lines 144-146 and
`f1e-prefill-sweep-2026-09-11.txt`). The guard of `LLVQ_PREFILL_TOKENS` compares N tokens in one call with N
one-token calls. That path includes the batched attention, which the exact verify removes.
- Ten lengths from 199 to 800 tokens: argmax identical every time, max |Δlogit| from 0.14 % to 2.15 % of the logit
  scale.
- The 1.8 % at 203 tokens (`f1e-smoke`) was explained the same day. Two-row tails at 201, 202 and 204 gave 0.14,
  0.19 and 0.14 % (`f1e-tail`). Three-row tails at 199 and 207 gave 0.31 and 0.15 %, while a full packet at 208 gave
  2.15 % (`f1e-tail3`). The journal's reading: the residual depends on the final token, not on the packetization.
- Packet size moves no bit of that path: at 606 tokens, 4:128, 8:64 and 16:32 give the same argmax 6722 and the same
  max |Δlogit|, 4.688e-2, to every printed digit. The journal reads it as "the cutting does not touch a bit".

**Row-invariance by code reading.**
- candle-nn 0.9.2 launches `rms_norm` and `softmax` with one block per row, the block sized by the row width only
  (`ops.rs:354-359`, `:543-548`). RoPE is elementwise.
- The head runs one identical launch per row, with only `x_off` differing.
- `rot_apply_rows` runs one CTA per row through the same helpers as `rot_apply` (`rotate.cu:38-66`, `:103-140`), at
  the same block size `rot.threads` (`fused_cuda.rs:1422-1439`, `:1491-1494`).
- The Tetra rows kernel keeps the one-row kernel's 24 explicit `__fmaf_rn` and its lane order (block j on lane
  j mod 32 for any tile that is a multiple of 32: decode tile 64, rows tile 128 at the served 4:128 pair).
- Two things are not explicit FMAs. The one-row kernel accumulates `acc += tetra48_dot(...)`, which returns
  `acc * gscale[g] * invnorm[m]` (`tv_tetra48_h.cu:80`, `llvq_tetra48.cuh:132`). The rows kernel accumulates
  `acc[r] += a * gs * iv` (`llvq_tetra48.cuh:203`). The epilogue `x * rscale[row] + tv` is the same in both.
  NVRTC compiles with its default `--fmad=true` (`llvq-cuda/src/gpu.rs:14-16`), so it may contract these
  multiply-adds, and it must contract them the same way in both kernels.
- The host test proves the two routes identical with clang++ `-ffp-contract=off` (`tests/host_tetra48.cpp`). It
  therefore cannot see a contraction difference.

llvq-tetra 0.2.0 (lock: torch 2.14.1, transformers 5.18.0; floors torch ≥ 2.14, transformers ≥ 5.17):
- Without `device_map`, every record is rebuilt dense and the int4 g64 embedding is dequantized into a stock
  `nn.Embedding`, head tied (`integration.py:128-171`, `conversions.py`). The model is a stock `Qwen3ForCausalLM`.
  `output_hidden_states` works, and the shared int4 head is `get_input_embeddings().weight`.
- On `mps` with `device_map`, decoder layers stay stock (hidden states work); the head is `Int4TiedHead` with no
  `weight`, callable as a function on any hidden state (`vocab.py`).
- Gate B: the same 256 greedy ids as `bin/run` over four prompts on CPU, Metal and an L4 (README).
- Hub `Pier-Jean/Qwen3-4B-LLVQ-Tetra` at `acc5f663` declares `artifact_sha256` `886391a8…`, the served object.

DeepSpec at `005e03b` (code MIT; local copies in `wave3-verify/deepspec/` and `wave5-deep/downloads/`):
- `eval.py:61-65` spawns one worker per CUDA device: zero workers on a Mac. The loop itself,
  `generate_decoding_sample` (`base_evaluator.py:307-441`), is device-agnostic.
- Table 1's inputs are `eval_datasets/{mt-bench,humaneval,gsm8k}.jsonl` at that commit, first turn only
  (`base_evaluator.py:31-55`), GSM8K cut to 500 by `random.Random(980406).shuffle` (`:513-525`), thinking off.
- Block 7 means 7 input slots (anchor plus 6 copies of the mask id 151669, `evaluator.py:109-115`), 7 proposals and
  8 verify rows (`draft_ops.py:115`, `base_evaluator.py:215`). Round τ = accepted + 1 (`:423`), accepted only on a
  stop-token round (`:416`). At T = 0 acceptance is exact argmax matching (`sampling.py:6-11`).
- The mask slots are embedded by the drafter's `embed_tokens` (`draft_ops.py:37`). That table is a frozen copy of
  the target's, so the mask input is the target's row 151669.
- Taps `[1, 9, 17, 25, 33]` = `hidden_states[i+1]` concatenated, 12,800 wide (`common.py:52-56`); the last layer is
  refused (`base_evaluator.py:100-112`). Each drafter layer projects the fused feature into its context K/V
  (`modeling.py:103-106`); the drafter cache is cropped to the anchor after each block (`draft_ops.py:44`).
- Markov stage: rank 256, logits_k = base_k + W2·W1[prev], 7 sequential argmax steps (`markov_head.py:55-90`).
- Embedding and head are frozen copies of the target's (`modeling.py:270-283`, trainer `freeze=True`); EAGLE-3 too.
- Training answers were regenerated by the target at T = 0.7, top-p 0.8, top-k 20, non-thinking
  (`scripts/data/README.md`), 1.3 M samples, 10 epochs (paper §4).

Served-object greedy text exists at $0. The served kernel's GSM8K run holds 1,319 completions; the FP16 dense arm's
run holds the same 1,319 problems (`docs/mesures/gsm8k-wave1-2026-09-26-brut/run1-served/`, `run4-f16dense/`).
Each row stores the text, `n_prompt`, `n_gen` (stop token excluded) and the stop reason. The served run sums 416,989
generated tokens, median 299, 8 at the 1,024 cap (*measured*). No token ids are stored.

Missing: a K-row loop, a drafter interface, a block-output tap, an exact verify, multi-row int4 and head kernels,
`regroup` removal, a G0 harness, and any τ against the sealed file.

## 3. Design

### 3.1 G0, the τ harness on the Mac

One PEP-723 script, `code/speculation/g0_tau.py` (draft, never executed). It vendors a pinned subset of DeepSpec
with a sha256 per file, the three eval datasets included. It never imports `eval.py` or `deepspec.utils`, whose
`__init__` pulls the optimizer and the distributed stack.

**Offline simulator.** Under greedy acceptance a speculative run is a deterministic function of the target's greedy
continuation and the drafter. So, per prompt and target:
1. Plain greedy continuation, single-row decode with a cache, stop on 151643 or 151645, max 512 new tokens.
2. One teacher-forced forward over prompt plus continuation with `output_hidden_states=True`; keep the 5 taps.
3. Replay the loop. A round anchored at s feeds the drafter the taps of positions < s and the anchor t_s. Its 7
   proposals are compared with t_{s+1..s+7}; a = leading matches; the next anchor is s + a + 1.
   - DSpark and DFlash call `forward_dspark_draft_block` unchanged, with the context rows taps[prev_s : s]. They
     mirror `build_dspark_proposal` at threshold 0 to also log slot logits; V2 checks the mirror against the
     original.
   - EAGLE-3 mirrors `_init_context`, `_propose` and `_update`. Its single layer makes each cached entry a function
     of (tap_i, t_{i+1}) alone (`eagle3/evaluator.py:28-37`).

This equals the online loop in exact arithmetic. The only difference is where the taps were computed (one prefill
against verify passes). Validation V2 bounds it. Cost: one target forward per prompt instead of one per round.

**Arms.**
- Targets: sealed 4B (llvq-tetra dense, f16 on `mps`) and Qwen3-4B at `1cfa9a7` (f16).
- Drafters at their pinned revisions: DSpark `3457dff`, DFlash `02d530b`, EAGLE-3 `b0b90fd`.
- Heads, three arms:
  - `own`: the published BF16 frozen copy;
  - `shared`: the sealed table in both `lm_head` and `embed_tokens`, mask row included;
  - `shared-bf16mask`: the sealed table with row 151669 restored to the drafter's own BF16 row. This 5,120-byte row
    weighs 1.0e-5 b/param (*computed*).

  In both shared arms the head is computed as f32 logits against the f16 table, not from a bf16 copy. L1 prints
  max |Δ| and the cosine of row 151669, sealed against BF16. Δτ(shared − shared-bf16mask) isolates the quantized
  mask embedding; Δτ(shared-bf16mask − own) isolates the head. EAGLE-3 drafts autoregressively and reads no mask
  row, so it runs `own` and `shared` only.
- Zero-byte: prompt lookup over 4/3/2-grams of the history, gated by match length, computed on loop-truncated text.
- Probes: DSpark body RTN int8 per channel and RTN int4 g128 (4.25 b/weight); a drafter context window of 2,048
  positions. No probe measures Tetra's 2.15 b/weight (§3.5).
- Optional: z-lab DFlash-b16 (MIT), after its slot convention is checked in its own code.
- Optional math arm at $0: the served kernel's and the FP16 arm's own GSM8K completions as references, teacher-forced
  for taps, after a re-tokenization round trip (L2c).

**Logged per round:** anchor, K, proposals, accepted count, each slot's top probability, DSpark's confidence logit.
Per prompt: tokens, EOS or cap, loop flag and onset.

**S over β, computed offline from the logged rounds.** S is reported over a β grid {0.03, 0.10, 0.20, …, 0.80},
fixed k ∈ {2, 3, 4, 8} and the best k, plus K scheduled per round from DSpark's confidence. Routing reads S at the β
that L1a measures for the corpus's mean context (§3.3). The reference points, all *estimated*:
- today's exact path: 0.50 at 5 positions, 0.58 at 2,048, 0.71 at 8,192;
- after B2: 0.28 / 0.39 / 0.59;
- after B2 + B3: 0.19 / 0.16 / 0.11;
- flat verify: 0.03.

Every choice made on data is made on even prompt ids and scored on odd ones: the gate, the best k, the best drafter,
the confidence temperature.

**Agreement.** Teacher-forced top-1 and full-vocabulary KL(dense ‖ sealed), on both continuations and on 12
WikiText-2 windows (continuity with the 74.45 % of the bare artifact), plus the per-layer cosine of sealed against
dense taps. At 4B both targets fit in one process, 2 × 8 GB.

**Corpora.**
- Chat: MT-Bench, 80 first turns. Code: HumanEval, 164. Math: GSM8K, 80 of the seeded 500. All three come from the
  vendored DeepSpec files.
- Extraction: 40 English and 40 French JSON-extraction prompts on openly licensed documents of 1-4k tokens, 10 of
  them at 6-8k. They are pinned in `corpora.jsonl` by sha256 before stamping.
- Product slice: at least 10 operator documents, local only, pinned by hash and never by text.
- Thinking: 40 prompts at T = 0.6, online loop, informational.

**Validation.**
- V1: BF16 target, bf16, T = 1, online loop with the vendored callbacks, on the vendored datasets, against Table 1
  (DSpark 3.64 MT-Bench, 5.38 HumanEval, 6.11 GSM8K, *measured* by the authors).
- V2: online T = 0 loop against the offline replay on 20 prompts.
- V3: taps from `output_hidden_states` equal forward hooks on decoder layers, bitwise in f32.
- V4: the drafter's frozen head equals the Qwen3-4B BF16 checkpoint table, read with `safetensors` in its own dtype,
  max |Δ| = 0. The f16 cast is not used for this check.

**Pitfalls.**
- Loops inflate τ, and prompt lookup most of all: it accepts a loop almost whole.
  - The sealed 4B loops on the reference prompt: the line every job checks is "The capital of France is Paris."
    repeated (`served-grouped-2026-11-02-brut/job.log:75`, `:93`, *measured*).
  - 8 of 1,319 GSM8K answers hit the cap (*measured*).
  - Prompt lookup's c and τ_d are computed on loop-truncated text. Neural τ is reported both ways (rule D5).
- T = 0 against T = 1: Table 1 is T = 1. For DFlash-b16, greedy τ is 1.07-1.10 times T = 1 τ (*computed* from the
  authors' 4.35/4.07 and 6.64/6.04). q_τ is formed at T = 0 on both targets.
- Load on the CPU, then `.to("mps")`: a direct `device_map="mps"` fp16 load segfaulted (*measured*,
  `preregistration-hf-metal-bloc-2026-10-09-ECARTS.md` É4).
- DeepSpec's modeling imports nine names from `transformers.models.qwen3.modeling_qwen3`; all resolve under 5.18.0
  (read now). Any shim a later pin needs is printed and recorded as a deviation.
  - If the shims fail, the offline design allows a two-environment handoff. llvq-tetra's environment writes
    references and taps, about 11 GB a target in f16 (*computed* on ~435 k positions). An environment at DeepSpec's
    pins then replays the drafters from disk.
- `crop` changed meaning. DeepSpec calls `cache.crop(start)` as "keep `start` positions". In transformers 5.18.0,
  `Cache.crop(tokens_to_remove)` keeps that reading only through a legacy path for positive values. It warns that
  this "will be removed in version 5.18" (`cache_utils.py:175-190`, `:1631`). G0 pins `transformers==5.18.0` and
  wraps the cache in a subclass whose `crop(n)` keeps n positions; a silent switch would crop the wrong end.
- transformers 5.18 ships a native DFlash path (`speculation_type="dflash"`, `DFlashTokenCandidateGenerator`,
  `DFlashCache`). It drafts from the main model's own input and output embeddings, the shared head by construction
  (`generation/candidate_generator.py:1505-1650`). Its block is the anchor plus block_size − 1 masks, z-lab's
  convention. It is an independent online cross-check for the z-lab arm, not for DeepSpec's block-7 checkpoints.
- Triton is not needed: it serves training-time flex attention; evaluation runs `sdpa`.
- `LLVQ_THREADS` does not reach torch: `torch.set_num_threads(12)` and `nice -n 10`. Machine load changes the wall
  clock only, never τ, agreement or KL, so G0 voids nothing on load and logs a `ps` snapshot as information.

### 3.2 An exact K-row verify in today's engine

**Mechanism.** Every committed token equals plain greedy if each verify row repeats the decode step's arithmetic for
its position. The induction: identical cache content below position p, identical computation at p, identical argmax.
Today's rows path gets there with three changes and no new kernel.
1. **Attention row by row.** After `cache.append` of the K positions, call the decode-shaped attention once for each
   i in 0..K: q row i (made contiguous, as `group_view` does at decode), K/V views narrowed to offset + i + 1, mask
   `None`, as `step_mask` gives at l = 1. Same shapes as decode, hence the same cuBLAS call. The gqa-ab journal shows
   why the shape matters: cuBLAS rounds a batch of 8 key heads differently from 32 expanded heads (*measured*).
2. **Round clipping at chunk boundaries.** A round anchored at `off` takes K' = min(K, ⌈(off + 1)/256⌉·256 − off)
   rows. Every row then reads the capacity, hence the batch stride, that plain decode reads at its position. The
   buffer grows exactly when plain decode would grow it. The cost is one shorter round per 256 tokens. This removes
   the stride question by construction, so no pre-reserved capacity is needed; pre-reserving would change the plain
   arm's strides and its reference line.
3. **The head per row**: the existing per-row launch is the decode launch. The argmax of row i goes through
   `argmax_last`'s own call on a `[1, 1, vocab]` narrow.

Norms, RoPE, residuals, SiLU and the embedding work row by row. The int4 projections need a choice:
- **(a), taken.** Rows ≥ 1 keep today's f32 cast plus `tv_q4_h`; decode's single row takes `tv_q4_hh`. The two
  return the same bits whenever `check_int4_half_parity` holds for that record, by construction: the same f32 values
  in shared memory, the same dot. B1 therefore extends the parity to all 84 records. In the measurement mode the
  check prints each record (identical, or max |Δ| and the first row) and stops nothing.
- **(b), only if a record fails (a).** Rows ≥ 1 take `tv_q4_hh` from offset 0. The change belongs inside
  `forward_int4` (`fused_cuda.rs:930-959`), the one function both call sites reach: `int4_rows` for o_proj and
  down_proj@12-23, and `forward_rows`'s `_` arm for v_proj (`model.rs:957-962`). It needs a real copy (zeros plus
  `slice_set`) or an `x_off` argument, as `HeadOp` has, on a new entry point beside the untouched `tv_q4_hh`.
  `Tensor::copy` keeps the offset, as `forward_int4`'s own comment says.

What remains for the card (U3): the trailing multiply-add contraction in the Tetra rows kernel against the one-row
kernel, `rot_apply_rows` against `rot_apply`, and int4 parity on every record. The batched attention and the `Grow`
stride are gone by design. Identity tests use K = 2..8, which covers:
- one packet of 2, 3 or 4 rows (prompt lookup's K);
- a 1-row tail through the one-row kernel (K = 5, `model.rs:857`, `:923`);
- partial packets of 2 and 3 rows (K = 6, 7; rows past `n_rows` fold onto row 0 and are not stored,
  `tv_tetra48_h.cu:199-205`);
- two full packets (K = 8).

```rust
// model.rs, beside step_logits (:2207). Sketch, never compiled.
pub trait Drafter {
    /// Up to `max` tokens after `history`, whose last token is the anchor.
    fn propose(&mut self, history: &[u32], max: usize) -> Vec<u32>;
    /// `accepted` proposals were kept, then `bonus`.
    fn commit(&mut self, accepted: usize, bonus: u32);
}
// Loop body, after one prefill. CHUNK = 256, the Grow chunk.
let room = (off + 1).div_ceil(CHUNK) * CHUNK - off;       // rows left before the next growth
let p = drafter.propose(&hist, (k_max - 1).min(room - 1));
if p.is_empty() {                                          // the served step, verbatim
    anchor = argmax_last(&self.step_logits(anchor, off, &mut caches)?)?;
    off += 1;
} else {
    let rows: Vec<u32> = std::iter::once(anchor).chain(p.iter().copied()).collect();
    let g = self.verify_rows(&rows, off, &mut caches)?;   // exact per-row argmax
    let a = p.iter().zip(&g).take_while(|(x, y)| x == y).count();
    for c in caches.iter_mut() { c.truncate(off + a + 1)?; } // anchor + a kept
    commit(&p[..a]); anchor = g[a]; off += a + 1;          // g[a] is the bonus
    drafter.commit(a, anchor);
}
```

`verify_rows` is `hidden_cached` (:1913-1931) with an `exact_rows` flag passed to `Block::forward_cached`. The flag
changes the unexpanded-attention branch (:1653-1658) into the per-row loop. Under (a) nothing else changes; under (b)
it also reaches `forward_int4`. A drafter with no proposal takes the served step verbatim, so gating costs nothing.
The loop stops at the first of 151643 and 151645, like `bin/gsm8k` (`gsm8k.rs:304-312`).

**Drafter state.** Zero-byte drafters hold only the host token history. A DSpark or DFlash drafter holds its own KV
for the context positions below the anchor. After a round it appends the taps of verify rows 0..a (anchor and
accepted) and drops its block positions, as `draft_ops.py:44` crops. EAGLE-3 crops to its pre-proposal length and
re-extends with (tap_i, t_{i+1}) pairs (`eagle3/evaluator.py:148-170`). Taps come from a block-output capture in
`hidden_cached`, kept for the accepted rows only.

**Stride evidence, and why the clip is still adopted.** In gqa-ab, rounds 2-6 of the `Grow` arms reused buffers that
round 1 had grown. At the 5-token prompt, positions 5-255 were read at capacity 512 in rounds 2-6 against 256 in
round 1, and every arm repeated its own 256-token text in all six decodes (gate 5 *measured*, capacities *computed*
from `append_grow`). At 2,048, 4,096 and 8,192 the first decode step already grew the buffer to the later rounds'
capacity, so only the 5-token prompt exercised the stride. That evidence is text-level only. The clip makes it
unnecessary. One straddle test with the clip switched off stays, as information. The oracle's plain reference comes
from fresh caches (`fresh_caches()`, :1869), or from a fork of a fresh prefill at the same capacity, never from a
truncated cache, whose capacity is larger.

**8B and 14B.** They run `Cat`, where `truncate` copies the prefix. `Expanded` on `Grow` gives O(1) rollback with the
published bits by construction: `repeat_kv` copies the narrowed view into the same contiguous expanded tensor that
`Cat` produces (`repeat_kv` is a `Tensor::cat` then a reshape, candle-transformers 0.9.2 `utils.rs:28-37`), so the
matmuls see identical inputs. The doc comment of `append_grow` (:383-393) already calls the pairing correct, never
served. It needs a store choice decoupled from the form and the gate-4 line per size, no T8. Its limit is 8k: a
per-row expanded attention re-copies the history per row, about 17 ms a row at 8k (*computed* from the 4B's
expanded and grouped decode at 8,192, 33.1 and 77.5 tok/s, `gqa-ab-2026-11-01.txt`). Long-context speculation at 8B
and 14B therefore needs the grouped form per size (floor 1d, T8 each).

**Oracle-drafter test.** The drafter replays the plain greedy continuation, so every proposal is accepted and every
verify row's logits are compared bitwise with the plain step at that position. Two more drafters exercise rollback:
adversarial (the plain token with probability 0.7, a random one otherwise) and always-wrong (τ = 1). On the host
with `tiny_model` this catches logic. Only the card catches rounding: the CPU gemm was bit for bit where cuBLAS was
not (passation 2026-11-01).

**What the branch implementing audit steps 2-5 should keep in mind.**
- The GEMM prefill route must never serve verify rows: choose the route by caller, not by row count.
- A fused op used at decode must be used row by row in verify, or identity breaks. Write fusions as row-indexed
  device functions with a load-time check against the candle op.
- `tv_tetra48_seg` is a one-row kernel: verify would fan it out per row (exact, K weight streams). Keep its per-lane
  reduction order independent of the row count, so a rows variant can inherit its bits. Spell its final multiply-adds
  as explicit `__fmaf_rn` (or explicit non-contracted forms) so that no kernel pair depends on `--fmad` contraction.
- Three print-only load checks would answer U3's kernel part at about $0 marginal on that branch's next job: Tetra
  rows against one-row at n_rows 2, 3 and 4; `rot_apply_rows` against `rot_apply`; `tv_q4_hh` against `tv_q4_h` on
  all 84 records (L1b).
- On Metal `rows_per_launch` is 1 (`fused_metal.rs:1346`): no Mac speculation before a Metal rows kernel.

### 3.3 Zero-byte drafting, and what a verify row costs

Prompt lookup takes the longest 4/3/2-gram suffix of the history found earlier (most recent occurrence) and proposes
what followed, from the history only. Session suffix does the same over the session's previous outputs, where
extraction repeats its JSON keys. Gate: draft only from a match of length ≥ m, and choose k ∈ {2, 3, 4} by match
length. Host-side, no device state, rollback free. K ≤ 4 keeps a round in one 4-row packet; K = 5 adds a full
one-row pass over the weights.

**Cost of one extra verify row, today's exact path** (ms, *estimated* unless labelled):

| item | central [range] | source |
|---|---|---|
| Tetra rows kernel | 0.93 [0.50, 1.50] | plan B's per-row table |
| int4 fan-out, 84 records | 0.8 [0.6, 1.0] | 0.41 GB of int4 re-read a row plus 84 casts |
| head, one launch | 0.32 | *computed*: 218.8 MB at ~690 GB/s |
| `regroup` copies | 0.875 [0.55, 1.2] | plan B |
| per-row attention launches | 0.76 [0.48, 1.04] | 216 launches at 3.5 [2.2, 4.8] µs; 4.80 µs is f1e0's *measured* cost per removed operation |
| per-row KV read | 0.668 µs × position | *computed* slope of the grouped door, `served-grouped-2026-11-02.txt` |

Against the grouped step (7.43 / 8.78 / 12.90 ms at 5 / 2,048 / 8,192, *measured*), β = 0.50 [0.33, 0.69],
0.58 [0.44, 0.74] and 0.71 [0.62, 0.82]. Hence t(4)/t(1) = 2.5 [2.0, 3.1], 2.75 [2.3, 3.2] and 3.15 [2.85, 3.45].
These figures were 2.1 in the first version of this dive. That number was the prefill path's, with batched
attention, one head launch per pass and no per-row KV term.

**What prompt lookup delivers.** Let c be the share of rounds with a match and τ_d the τ of drafted rounds; per-token
acceptance follows a geometric model fitted to τ_d(4). Then S = (c·τ_d(k) + 1 − c)/(c·r(k) + 1 − c) with
r(k) = 1 + β(k − 1), and k is the best of {2, 3, 4}. Central values: c 0.45 and τ_d(4) 2.6 (per-token acceptance
0.72). Ranges: c 0.25-0.65 and τ_d(4) 2.0-3.3. All *computed* on estimated inputs.

| regime | β | decode S, central | over c, τ_d | full box with β | request |
|---|---|---|---|---|---|
| extraction, 2,048 + 300 | 0.58 | ×1.05 (k = 2) | ×0.99-1.18 | ×0.96-1.34 | ×1.01 today; ×1.04 after the GEMM route |
| extraction, 4,096 + 300 | 0.64 | ×1.03 | | | ×1.00 today; ×1.02 after |
| extraction-like text at 5 positions | 0.50 | ×1.08 | ×1.01-1.26 | ×0.97-1.52 | n/a |
| 8,192 + 512 | 0.71 | ×1.00 | ×0.97-1.07 | ×0.94-1.14 | ×1.00 |
| chat (c 0.15, τ_d(4) 1.6) | 0.50 | ×0.98 at k = 2; ×1.0 with a stricter gate that drafts less | | | ×1.0 |
| extraction 2k after B2 | 0.39 | ×1.15 (k = 3) | | | ×1.12 after the GEMM route |
| extraction 2k after B2 + B3 | 0.16 | ×1.42 (k = 4) | | | ×1.33 after the GEMM route; ×1.08 today |

The request column uses 8.20 s to the first token at 2,048 and 17.03 s at 4,096 (*measured*, grouped door). A decode
of 300 tokens takes 2.64 s at 2,048 (*computed* on the fitted step). The GEMM route's time to the first token is
0.51-0.57 s at 2,048 (*estimated*, `wave3-report/requests.py`). Until that route lands, a short extraction request is
four-fifths prefill. B1's 1-2 weeks therefore buy about 1 % of such a request, and about 4 % after the route.

Token Recycling (~×2 by its authors) and Lookahead (up to ×1.8 on MT-Bench) need tens of verify rows; they wait for
a flat verify. The product regime is 8k, so B3 (one KV read for K rows) matters more than anything else here.

### 3.4 Raising τ on chat

| lever | effect on chat τ | verify rows | cost | when |
|---|---|---|---|---|
| self-distillation on sealed outputs | +10-30 % if q_τ < 0.85 (*estimated*); authors' anchors: OSD +0.10 to +0.65 acceptance rate on shifted queries, DistillSpec +10-45 % speed over standard SD | 8 | $18-30 | after G0 |
| DFlash-b16 (z-lab, MIT) | 4.35 (T = 0) and 4.07 (T = 1) on MT-Bench against DSpark-b7 3.64 (T = 1), all BF16, *measured* by the authors | 16 | 0 | flat verify, r(16)/r(8) < 1.12 |
| tree drafts | +10-15 % (*estimated*: a second branch at 2 positions lifts 0.72 toward 0.8) | ~16 | tree mask in an invariant attention | after the brick |
| confidence-scheduled K | no τ gain; picks k per round, the chat lever on a non-flat verify | ≤ 8 | 0 | E0 onward |
| Lookahead | up to ×1.8 MT-Bench (authors, full precision) | tens | 0 | not at batch 1 |
| suffix decoding (session tree) | low coverage on open chat; up to ×5.3 on agentic SWE-Bench and Text-to-SQL (authors, arXiv 2411.04975) | 2-8 | 0 | E0, on extraction |

Self-distillation recipe (*estimated* from `docs/data/jobs.csv` anchors):
- Data: 60 k prompts weighted toward chat and French extraction. The sealed dense reconstruction is exported to f16
  safetensors (`bin/export` exists; whether it reads the sealed v5 file is to check) and served in vLLM at
  DeepSpec's T = 0.7 / 0.8 / 20. 27 M tokens at 5,643 tok/s (*measured* batched vLLM FP16 4B, `gsm8k-vllm-4b-r2`)
  take 1.3 h, about $3 on l40sx1.
- Training: fine-tune DSpark from its release, with target features computed online. DeepSpec's cache would take
  30.7 KB a token (*computed*), 38 TB at full scale. 39 M tokens × 2 epochs against the row-scale anchor (19.8 M
  tokens in 130 min, $3.90, `tetranu-rowscales`) gives about $15.
- With a retry: $18-30. Engineering: 2-4 days for the online-feature loop. Training from scratch is out: about 12 B
  token-passes, over $2,000 (*estimated*).

### 3.5 Rule 6 and the license gap

Added b/param over the 4B's 4,022,468,096 parameters (*computed*; int4 g128 at 4.25 b/weight; the Tetra column at
2.15 b/weight is **unprobed**: no G0 probe measures what a Tetra drafter body accepts):

| item | BF16 | int8 | int4 (probed) | Tetra (unprobed) |
|---|---|---|---|---|
| DSpark body, 537.4 M + Markov 77.8 M | +2.447 | +1.224 | +0.650 | +0.369 (Markov int4) |
| DFlash body, 537.4 M | +2.138 | +1.069 | +0.568 | +0.287 |
| EAGLE-3 body, 149.4 M | +0.594 | +0.297 | +0.158 | +0.080 |
| own head, 389.0 M | +1.547 | +0.774 | shared table: 0 | n/a |
| drafter KV at 8k, 5 layers (1 for EAGLE-3) | f16 168 MB, +0.334 (EAGLE-3 +0.067) | q8 +0.167 (+0.033) | | |

The triplet margin at 4B is 0.414 b/param with q8 KV and 0.287 with f16 (*computed* in the wave-2 report). At the
probed rate, with the shared head:
- EAGLE-3 fits: int4 body 0.158 plus q8 KV 0.033 gives 0.191 < 0.414; with f16 KV, 0.158 + 0.067 = 0.225 < 0.287.
- DFlash does not fit: 0.568 for the body alone. DSpark does not either: 0.650.

At the unprobed Tetra rate DFlash would fill the f16 margin with its body alone and exceed the q8 margin with its KV
(0.287 + 0.167 = 0.454). Quantizing a drafter costs acceptance: GPTQ on an EAGLE-2 drafter gave "substantial
degradation" where QAT gave none (MiniCPM4, confirmed in the wave-1 fact-check, no numbers). G0's RTN probes size
that risk at int8 and int4 only.

License: DeepSpec's nine weight repositories declare no license (*measured*, Hub tags). MIT covers the code only.
z-lab's Qwen3-4B and Qwen3-8B DFlash-b16 are MIT and store no head or embedding (537,427,200 parameters, same taps,
same mask id; *measured*, safetensors header and config). There is none for the 14B. Asking DeepSeek for a license is
a message on the operator's behalf, his to send.

## 4. Deciding unknowns

Each row ends with an "otherwise", or points to the prereg rule that partitions it.

| unknown | best value now | confirms the lead | kills it (proposal to the operator) | otherwise |
|---|---|---|---|---|
| U1 q_τ, best drafter (chosen on even ids), shared head | chat 0.80 [0.65, 0.95], code 0.85 [0.70, 0.98] (*estimated*: T0/T1 uplift 1.07; bare-artifact top-1 74.45 %, *measured*) | ≥ 0.85 on both: no fine-tune | < 0.65 on either: neural drafting paused at 4B, pending a fine-tune pilot | fine-tune pilot before integration (prereg-1 D1) |
| U2 head-sharing Δτ (shared-bf16mask − own) | 0.00 [−0.10, +0.15] (*estimated*: frozen copies, so the int4 table is common to drafter and target) | ≥ −0.10: head costs 0 bytes | < −0.30: own head (+0.77 int8), only EAGLE-3 stays in the margin | both accountings to the operator (D2) |
| U2b mask-row Δτ (shared − shared-bf16mask) | 0.00 [−0.15, +0.05] (*estimated*) | ≥ −0.05: the sealed table serves the mask too | n/a: keeping the BF16 row costs 5 KB | report (D2b) |
| U3 card row-invariance of the exact path | host identity; batched attention and stride removed by design; three residual risks (§3.2) | E0 oracle test bitwise at K = 2..8, every parity line identical | an op not invariant and not fixable without T8: exact verify waits for the brick | long context only fails: short-context use (prereg-2 rule D) |
| U4 rows-path t(k), today | t_pf(4)/t_pf(1) ≈ 2.1 at context 0 (*estimated*), exact t(4)/t(1) 2.75 at 2,048 | L1a gives β_ex(2,048) ≤ 0.40 | β_ex > 0.60: B2's per-row estimate is revisited before any build | 0.40-0.60: B1 waits for B2 (prereg-0 rule) |
| U5 prompt-lookup c and τ_d, extraction | c 0.45 [0.25, 0.65], τ_d(4) 2.6 [2.0, 3.3] (*estimated*; prompt lookup about ×2.4 on summarization, authors) | S at L1a's measured β ≥ 1.10: B1 now | S below 1.10 at β_B2 and below 1.20 at β_B3 on the 6-8k slice: zero-byte waits for the brick | B1 with B2, or zero-byte waits for B3; without L1a, no route (prereg-1 D3) |
| U6 drafter quantization | int8 −1 % [−4, +1], int4 −8 % [−20, −2] (*estimated*) | int4 ≥ −5 %: int4 body licensed | int8 < −5 %: only BF16 or QAT drafters | int8 is the floor (D4) |
| U7 flat verify r(8) | 1.14-1.29 (*estimated*, wave-2 report §6) | ≤ 1.3 | > 1.6: neural chat stays ≤ ×1.2 | 1.3-1.6: neural chat ×1.2-1.6, the operator weighs it against the brick's cost |
| U8 DeepSpec weights license | none declared (*measured*) | a grant | refusal: z-lab MIT (4B, 8B) or own training | no answer: G0 measures, nothing ships with DeepSpec weights |

## 5. Validation ladder

| step | settles | method | cost | prereg | decision rule (proposal) | needs |
|---|---|---|---|---|---|---|
| L0 | facts of §8 | code, journals, primary sources | $0, done | no | n/a | none |
| L0b | the tail "residual" and packet invariance | `jobs.csv` lines 144-146, `f1e-prefill-sweep-2026-09-11.txt`, candle-nn launch geometry | $0, done (§2) | no | n/a | none |
| L1a | U4: rows-path t(k) for k = 1..8 on the served object | `LLVQ_CONFIG=<grouped> LLVQ_PREFILL_TOKENS=k fusedrun`, the one mode allowed beside the config, existing image | about $0.1 (*estimated*: `f1e-tail` and `f1e-tail3` billed 2 min each for 3-4 lengths), cap $1 | `prereg-speculation-0.md`, stamped | that prereg's rule on β_ex | go, R1 |
| L1b | U3's kernel part | three print-only load checks (§3.2), riding on the audit-steps branch's next job | about $0 marginal | a line in that job's prereg | all identical: U3 reduces to the card test of E0; any differs: design (b), or B1 waits | the branch owner's choice |
| L1 | harness runs; V3, V4; vendored imports under 5.18; mask-row print | 2 prompts × 32 tokens, tap, head and mask-row checks, ~1 h plus ~15 GB of downloads | local, niced | no | all pass: L2a; any fails: fix, record a deviation, rerun; otherwise stop | go |
| L2a | U1 early, DSpark only | pilot: `shared-bf16mask` head, both targets, 40 MT-Bench + 40 HumanEval prompts, 1-2 h, reused by L2 | local, $0 | prereg-1 stage 2, rule D0 | q_τ ≥ 0.90 on both: D1 "high" now; ≤ 0.60 on both: D1 "low" now; otherwise the full run | go, R1 |
| L2b | U5, routed the same night | prompt-lookup c and τ_d on token lists right after the references stage, minutes | local, $0 | prereg-1 stage 4b, rule D3 | D3 with L1a's β; without L1a, a report over the β grid and no route | L1a done for a route |
| L2c | how well G0's object stands for the served kernel; math τ against served text | teacher-force the 1,319 served and FP16 GSM8K completions after a re-tokenization round trip, ~20-30 min | local, $0 | prereg-1 stage 4c, rule D7 | top-1 agreement ≥ 0.98: q_τ is read as is; otherwise q_τ carries the measured disagreement and math τ on the served text becomes primary | go |
| L2 | U1, U2, U2b, U5, U6, loops, agreement | `prereg-speculation-1.md`, 11-14 h, two nights at `nice -n 10`, 12 threads | local, $0 | stamped | that prereg's rules D0-D7 | go; R1 |
| L3 | transfer to 8B and 14B | G0 on dense f16; at 14B one target per process, per-position argmax and top-64 logits cached to disk for agreement, exact KL on a 5 % sample of positions | local, $0 | stamped | q_τ within ±0.10 of the 4B: assume transfer; otherwise per-size routing. The 14B kernel and its reconstruction diverge at token 78 (ETAT §2), so a 14B q_τ on the reconstruction is a weaker proxy than at 4B | go |
| L4 | U3, exact speculation end to end | E0 job, `prereg-speculation-2.md`, after B1 | l40sx1, ~33 min, ~$1.00, $1.80 at the timeout (*estimated*; anchors `gqa-ab` $0.29, `served-grouped-4b` $0.54, `trims-contexts` $0.35) | stamped | prereg-2 rules C, D, A, B and otherwise; the predicted outcome today is rule B | go, cap, R1, R8 |
| L5 | fine-tune gain | data plus training, G0 rerun on the Mac | $18-30 | stamped | chat τ +10 % at unchanged code τ: keep the fine-tuned drafter; otherwise the release | go, cap; only if D1 routes there |
| L6 | U7 | bench β, flat-verify t(k) for the brick | $1-2 after 2-4 weeks of code | stamped | r(8) ≤ 1.3: neural integration; > 1.6: neural chat stays below ×1.2; otherwise the operator weighs it | go, rulings 3 and 7 of the wave-2 report |
| L7 | the fundamental gate | in-engine neural drafter, end to end | ~$1 | stamped | identical output and median ≥ ×1.25 on code and extraction, ≥ ×1.0 on chat: serve; otherwise zero-byte on extraction only | go |

G0 is a measurement, not a gate (METHODE §1): τ is a proxy for throughput. Its rule routes work. The gates are L4's
and L7's end-to-end tok/s with identical output.

## 6. Build plan after validation

1. **B1, exact verify and prompt lookup** (1-2 weeks; waits on R1 and R8).
   - Code: the `Drafter` trait, `verify_rows` with the per-row attention and the round clip, design (a) for int4, a
     prompt-lookup drafter, `KvCache::fork` (a buffer copy at the same capacity, so each identity run starts from
     one prefill), and the `fusedrun` mode `LLVQ_SPEC_AB` added to the refusal list at `fusedrun.rs:677-679`.
   - The mode spells out no flags itself. The job puts the five served flags on every command line and sets the
     grouped form on `Grow(256)` as `LLVQ_GQA_AB` does.
   - No new binary, so the binary half of the two-lists trap does not apply. The corpus still needs a route into the
     container. It goes through the bucket mount, like the sealed file: `hf buckets cp` before the job, then bytes and
     sha256 checked on the mount against the prereg.
     - Not through the launch command: `ops/run.py` passes the whole script as one `bash -lc` argument
       (`run.py:1132`), and Linux caps one argument at 128 KiB, while the corpus is about 230 KB (*estimated*).
     - Not under `docs/` or `ops/jobs/` either: those are outside `UPLOAD_ALLOW` (`run.py:747-758`), and the runtime
       stage copies only binaries, `fetch-qtip.sh` and `configs/` (`Dockerfile.cuda:131-175`).
   - Load-time parity, printed per record in the mode: Tetra rows against one-row at n_rows 2, 3 and 4 on every
     Tetra record, `tv_q4_hh` against the f32 path on all 84 int4 records, `rot_apply_rows` against `rot_apply`.
   - Host tests: oracle, adversarial and always-wrong drafters on `tiny_model`; prompt-lookup units; a capacity
     assertion that no round grows the buffer beyond what plain decode holds at that position.
   - Mutants to kill:
     - truncate to off + a instead of off + a + 1;
     - the bonus taken from row a + 1;
     - a mask on the per-row path;
     - the batched attention left in the exact path;
     - the round clip removed (killed by the capacity assertion, not by bits, since the CPU gemm ignores the stride);
     - prompt lookup matching its own suffix;
     - argmax taken over the K-row tensor instead of per-row `argmax_last`;
     - a v_proj coverage mutant: rows ≥ 1 of v_proj sent through a perturbed scale, which the card's oracle-drafter
       test must catch.
   - Gate: L4.
2. **B2, speed with bits kept** (1 week). `regroup` removal, an int4 rows kernel and a multi-row head kernel, each
   row in the decode reduction order, with load-time card checks. Expected β at 2k: 0.39 (*estimated*).
3. **B3, batched invariant attention**, with the attention floor (T8 for decode): one KV read for K rows, each row
   in the decode order, fixed split size. Expected β at 2k: 0.16, at 8k: 0.11 (*estimated*).
4. **B4, neural drafter** (after L6).
   - Block-output tap in `hidden_cached` (:1928); drafter body on existing kernels; the shared int4 head through
     B2's head kernel; Markov W2 in int4 (19.4 MB, L2-resident on a 96 MB L2); drafter KV on `Grow`, cropped per
     round.
   - In-engine τ must land within ±10 % of G0's on the same prompts; otherwise it is a bug.
5. **B5, 8B and 14B.** `Expanded` on `Grow` with the gate-4 line per size; the grouped form per size for 8k.

## 7. Rules map

- **T8:** E0 adds none if the identity holds; speculation then changes no output. The batched attention kernel and
  the brick are T8 events of their own floors. Verifying through today's batched prefill path would be a T8 event; it
  is not proposed.
- **A2:** the speculative loop launches normally, nothing is captured. 4E is decision 4 of the wave-2 report.
- **Rule 6:** §3.5, plus drafter KV. The operator fixes the unit, the KV assumption, head sharing, and whether the
  drafter and the down_proj int4 move share one margin.
- **Closed leads:** the speculative verdict of 2026-08-05 (`docs/archive/pistes-facteurs-cles-2026-08-05.md:75`,
  `pistes-battre-q4.md:142`, reopen condition "70B") has no ETAT §6 row: lifting it is a go under rule 1. Its premise
  was a sequential 0.6B draft model paying launches per draft token; none of the drafters here is one. ETAT §6 rows
  that bind: A2 (no graphs), and the bench ruling of 2026-10-06 (speed claims end to end on the served file).
- **ROADMAP §4, "Batch above 1, and prefill"**: the wave-2 report reads it literally by default (decision 9). Under
  that reading the multi-row path waits, and B1 serves a K-row forward at decode.
- **Rules 4, 5, 7:** raw and same-head ratios (both arms run the same head kernel); one card, one engine; medians with
  ranges formed round by round.
- **Rulings asked:**
  - R1: authorize G0 and L1a on the archived lead, or lift the verdict.
  - R2: the product documents, or synthetic French ones.
  - R3: rule 6 for a drafter.
  - R4: the license route.
  - R5: a go and a cap for E0.
  - R6: whether thinking mode is in scope.
  - R7: whether speculation becomes a served config field.
  - R8: does an exact K-row verify, which changes no format and no output, fall outside the ROADMAP §4 hold? By
    default (literal reading), B1 and E0 wait. G0 does not, since it costs $0 and touches no engine code; nor does
    L1a, an existing measurement mode on the served object.

## 8. Checked now at $0

1. Block 7 gives 7 proposals and 8 verify rows (`draft_ops.py:115`, `base_evaluator.py:215`); wave-2-spec's K = 7
   (anchor plus 6) is wrong.
2. At T = 0 DeepSpec's acceptance is exact argmax matching and the bonus is the argmax (`sampling.py:6-11`,
   `base_evaluator.py:252-285`).
3. All three DeepSpec drafters copy and freeze the target's embedding and head (`modeling.py:270-283`; trainer
   `freeze=True`; EAGLE-3 `modeling.py:248-262`). The shared int4 head replaces a frozen input, never a trained one.
   The mask slots read the same frozen table at row 151669 (`evaluator.py:109-115`, `draft_ops.py:37`).
4. Safetensors headers, read by range request: DSpark 4B has 64 tensors and 1,393,133,569 parameters, including
   `lm_head`, `embed_tokens` and `markov_w1`/`markov_w2` of 151,936 × 256; EAGLE-3 has a 4096 × 5120 `q_proj`; z-lab
   DFlash-b16 has 537,427,200 and no head.
5. Taps `[1, 9, 17, 25, 33]`, mask id 151669, for DeepSpec and z-lab alike (configs).
6. `eval.py` spawns `torch.cuda.device_count()` workers (`:61-65`); `deepspec/utils/__init__.py` imports the optimizer
   and the distributed stack; requirements pin torch 2.9.1, transformers 5.10.2, triton 3.5.1. Table 1's inputs are
   `eval_datasets/{mt-bench,humaneval,gsm8k}.jsonl`, first turn, GSM8K shuffled with seed 980406 and cut to 500.
7. DeepSpec regenerated answers at T = 0.7, top-p 0.8, top-k 20, non-thinking; its cache is ~38 TB for the 4B
   (`scripts/data/README.md` at `005e03b`).
8. llvq-tetra's dense path yields a stock Qwen3 with a dequantized int4 embedding and a tied head
   (`integration.py:128-171`, `conversions.py`); lock torch 2.14.1, transformers 5.18.0 (`uv.lock`).
9. The packed Hub model at `acc5f663` declares `artifact_sha256` `886391a8c03f…`, `Tetra+Int4G128`, embedding g64.
10. `KvCache::truncate` is O(1) on `Grow`, a prefix copy on `Cat` (:421-447); the store follows the form
    (`served.rs:221-229`); a `Grow` view reaches cuBLAS uncopied (`rows_contiguous`, :490-499 and its test :2832).
11. The CUDA head launches once per row (`fused_cuda.rs:1330`). Every int4 row at a nonzero offset takes an f32 cast
    and `tv_q4_h` (`:946-951`), on both call sites: `int4_rows` and `forward_rows`'s `_` arm for v_proj. RoPE is
    `candle_nn::rotary_emb::rope` (:192-197).
12. The card's multi-row residual is explained: the 3-row tail at 203 tokens gave 1.8 %, two other 3-row tails gave
    0.31 and 0.15 %, and a full packet at 208 gave 2.15 %. "The residual depends on the final token, not the
    cutting" (`jobs.csv` lines 144-146). Packet size moves no printed digit (`f1e-prefill-sweep-2026-09-11.txt`).
13. candle-nn 0.9.2 `rms_norm` and `softmax` launch `grid = n_rows` with a block sized by the row width only
    (`ops.rs:354-359`, `:543-548`).
14. `rot_apply_rows` runs the helpers of `rot_apply`, one CTA per row, at the same `rot.threads`
    (`rotate.cu:38-140`, `fused_cuda.rs:1422-1439`, `:1491-1494`).
15. The Tetra one-row and rows kernels share 24 explicit `__fmaf_rn` and the lane order. Their trailing
    multiply-adds are left to NVRTC's `--fmad=true`, the default the repository relies on (`gpu.rs:14-16`,
    `fused_metal.rs:535`).
16. `decode_greedy` has no stop token (:2216-2232); `bin/gsm8k` stops on two stop ids and excludes them from `n_gen`
    (`gsm8k.rs:304-312`). The reference line of every job is a loop (`served-grouped-2026-11-02-brut/job.log:75`).
17. The grouped config hashes to `2455fe85…` (repository file and the served-grouped job's `files.sha256`). The
    served-grouped image is Space sha `f77f803e`, from commit `f808856`.
18. `ops/run.py bench` runs the job as `bash -lc <script>`, one argument (`run.py:1132`). `UPLOAD_ALLOW` covers the
    Cargo files, `llvq-*/**`, `ops/fetch-qtip.sh` and `configs/**` (`run.py:747-758`).
19. The served GSM8K dump has text, `n_prompt`, `n_gen` and stop per problem, no ids; 416,989 generated tokens over
    1,319 problems (*measured*, read now).
20. DSpark Table 1, Qwen3-4B, T = 1: chat 3.64/3.54/3.29, code 5.13/5.38/4.86, math 6.11/5.70/4.89; no greedy table;
    training mix 17.6 % chat (arXiv 2607.05147).
21. DFlash-b16, Qwen3-4B: MT-Bench τ 4.35 at T = 0 and 4.07 at T = 1; trained 16, tested 8 generalizes, not the
    reverse (arXiv 2602.06036, Tables 1 and 8).
22. W4A16 and W8A8 targets keep EAGLE-2 τ "nearly" unchanged, while verification of a 60-token tree costs 1.8 decode
    steps under W4A16 against < 1.2 for FP16 (arXiv 2505.22179, from figures and text).
23. Qwen3-4B has 4,022,468,096 parameters (Hub safetensors metadata).
24. vLLM FP16 4B generated 392,381 GSM8K tokens in 69.5 s on l40sx1 (`docs/data/jobs.csv`, `gsm8k-vllm-4b-r2`).
25. z-lab lists MIT DFlash-b16 for Qwen3-4B and 8B and none for 14B (Hub API).
26. In llvq-tetra's environment (transformers 5.18.0), read in the `.venv` sources: `modeling_qwen3` still imports
    every name DeepSpec takes from it; `Cache.crop` takes `tokens_to_remove`, with absolute sizes on a deprecated
    path; and a native DFlash candidate generator exists that reads the main model's embeddings.

## 9. Not established

- Any τ against the sealed file, with any head; q_τ at 8B and 14B.
- Row-invariance on the card of the Tetra rows kernel's trailing multiply-adds, of `rot_apply_rows`, and of int4
  parity beyond the first record.
- t(k) of an exact verify, at any context; the rows path's t(k) (L1a measures it); the cost of the Markov stage in
  our engine.
- That DeepSpec's modeling code behaves the same under transformers 5.18: its imports resolve, but `crop` and the
  rotary constructor changed.
- The gain of a fine-tune on sealed outputs; every number in §3.4 is the authors' or *estimated*.
- Prompt-lookup coverage on the operator's documents, which no corpus in the repository represents.
- That a drafter context window of 2,048 keeps τ, which would cut the drafter's KV bill.
- What a Tetra-encoded drafter accepts.
- The legal status of DeepSpec's weights.

## 10. Revision after the adversarial review (2026-11-02)

| finding | severity | what changed |
|---|---|---|
| The 3-row tail was called unexplained | high | §2 and §8 item 12 carry `f1e-tail`, `f1e-tail3` and the sweep. U3 is restated as three residual risks. K = 2..8 is justified by packet coverage. |
| E0 timed fixed 256-token decodes past EOS and through loops | high | Both arms stop at 151643/151645 with equal token counts. The loop flag is printed per prompt, and the primary statistic uses loop-free prompts. G0 computes prompt lookup on loop-truncated text. |
| P0 used the prefill path's t(4) ≈ 2.1 | high | §3.3 itemizes the exact path's per-row cost by context: β 0.50 / 0.58 / 0.71. The headline is now ×1.05 decode and ×1.01 request at 2k. L1a measures the rows path before B1. |
| E0's rules and §4 did not partition | high | Prereg-2 has rules A, B (median < 1.10 or a round ≤ 1.00), C, D and a residual otherwise. §4 has an "otherwise" column. |
| Served flags not spelled out | medium | Prereg-2 copies gqa-ab: five flags on every command, the config checked as an object at sha256 `2455fe85`, arm-line gates. |
| Gates voided the outcomes rule C reads | medium | Gates are infrastructure only. Parity lines print, the identity tables print, and the t(k) table always prints. |
| No corpus route into the container | medium | The corpus goes through the bucket mount with bytes and sha256 checked (§6). The reasons against the other routes are given. |
| R8 missing | medium | Added to §7 and to prereg-2's preconditions. |
| D4 licensed a Tetra drafter from int4 probes | medium | D4 licenses int4 only, the Tetra column is marked unprobed, and the fits are restated at int4. |
| D1 and D3 chose on scored data and routed on an estimated β | medium | Choices are made on even ids and scored on odd ones. D3 routes on L1a's β and reports S over a β grid. |
| Mask row 151669 confounds head sharing | medium | A `shared-bf16mask` arm is added, the head is computed in f32 from the f16 table, and L1 prints the mask row. |
| The int4 flag missed v_proj | medium | Design (a) rests on all-record parity. Design (b) sits in `forward_int4`, the one function both call sites reach. A v_proj mutant is added. |
| Grow strides | medium | The round clip is adopted with a capacity-assertion mutant. The gqa-ab evidence is cited. The oracle reference comes from fresh caches. |
| Decode ratio shown without the request ratio | medium | The request column is in §3.3, and request ratios print beside decode ratios in prereg-2. |
| G0's core-count gate | medium | Removed: `ps` is logged as information, and wall-clock figures are informational. |
| L3 held two 14B targets in one process | medium | One target per process with cached top-k, and a reconstruction caveat (token 78). |
| V1 datasets not pinned | medium | Three `eval_datasets` files are vendored with sha256, and `corpora.jsonl` is pinned before stamping. |
