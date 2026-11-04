DRAFT, not stamped, not committed.

# Preregistration. E0: exact speculative greedy decoding through today's rows path, and what prompt lookup buys

**Written on 2026-11-02 as a draft, ahead of the code it measures; revised the same day after an adversarial review.**
To be committed and TIMESTAMPED BEFORE the run. It runs only after five conditions:
- build step B1 of `speculation.md` §6 is committed with its host tests and mutants;
- ruling R8 says that an exact K-row verify, which changes no format and no output, falls outside the ROADMAP §4 hold
  (by default, under the literal reading, it does not, and this run waits);
- ruling R1 authorizes work on the speculative lead;
- L1a's rule (`prereg-speculation-0.md`) or G0's rule D3 routes here, or the operator runs E0 for the identity proof
  alone;
- the operator gives a go with a cap (R5).

Cost: about **$1.00** on l40sx1, timeout 60 min, so **$1.80** at worst (*estimated*, §8). Spent before it: $248.84
over 237 lines of `docs/data/jobs.csv` (*measured*), plus L1a if it has run.

## 1. Why this run exists

G0 measures what a drafter proposes. Two things only the card can say:
- whether today's rows path, with one decode-shaped attention call per row and rounds clipped at the 256-position
  chunks, returns plain decode's logits bit for bit;
- what such a verify costs, t(k), at short and long context.

The evidence so far, all *measured* unless labelled:
- N tokens in one call against N one-token calls: argmax identical at ten lengths, max |Δlogit| 0.14 % to 2.15 % of
  the logit scale. The residual "depends on the final token, not on the cutting" (`docs/data/jobs.csv` lines
  144-146).
- Packet size moves no printed digit of that residual: 4:128, 8:64 and 16:32 at 606 tokens
  (`f1e-prefill-sweep-2026-09-11.txt`). That path includes the batched attention, which this design removes.
- By code reading: candle's `rms_norm` and `softmax` are row-invariant by launch geometry; RoPE is elementwise; the
  head runs one identical launch per row; `rot_apply_rows` runs `rot_apply`'s helpers at the same block size.
- What remains for the card: the Tetra rows kernel's trailing multiply-adds, which NVRTC's `--fmad=true` may contract
  (`speculation.md` §2); `rot_apply_rows`; and int4 parity beyond the first record.

The run also measures what prompt lookup buys per decode and per request, with identical output.

## 2. What the code will hold

- **`Qwen3::verify_rows`:**
  - K rows through the rows path, with attention per row: q row i made contiguous, K/V views narrowed to
    offset + i + 1, mask `None`.
  - The round clip: K' = min(K, ⌈(off + 1)/256⌉·256 − off).
  - int4 rows ≥ 1 on today's f32 path (design (a), `speculation.md` §3.2).
  - The head per row as today, and each row's argmax through `argmax_last`'s own call on a `[1, 1, vocab]` narrow.
- **`KvCache::fork`**: a device copy of the buffers at their capacity, so every run starts from a cache identical in
  content and capacity to a fresh prefill's.
- **`Drafter` and four drafters**: oracle (replays plain greedy), adversarial (plain token with probability 0.7, else
  a seeded random one), always-wrong, and prompt lookup (4/3/2-gram, gate and k from G0's D3, or m = 3 and k = 3 if
  G0 has not run).
- **The `fusedrun` mode** `LLVQ_SPEC_AB=<corpus path>,<k_max>`. It is a bench mode, refused beside `LLVQ_CONFIG`
  (added to the list at `fusedrun.rs:677-679`).
  - It sets the grouped form on `Grow(256)` for every arm, as `LLVQ_GQA_AB` does.
  - Every arm line prints the layout, embedding, attention and store labels.
  - Both arms stop at the first of 151643 and 151645, or at 512 tokens. No timed decode continues past a stop id.
- **Load-time parity, printed per record in this mode, never refusing the load:**
  - the Tetra rows kernel against the one-row kernel at n_rows 2, 3 and 4 on every Tetra record;
  - `tv_q4_hh` against the f32 path on all 84 int4 records;
  - `rot_apply_rows` against `rot_apply` on every rotation.

  Each line reads `identical`, or gives max |Δ| and the first differing row. The served first-record check
  (`check_int4_half_parity`) keeps its refusal, unchanged.

## 3. What runs

One image from the commit carrying this file; its Space sha is checked by the launch script (`IMAGE_SHA`). One job,
l40sx1, with the objects named by bytes and sha256 and checked on the mount:
- the sealed 4B, `sealed-4b-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8…`;
- the grouped config, `configs/qwen3-4b-tetra-e4-grouped.json`, sha256 `2455fe85…`, written by the job from the
  launch command and checked by sha256. It is the object the arms reproduce. The mode cannot read it, since a bench
  mode is refused beside `LLVQ_CONFIG`;
- the corpus, `spec-corpus-<date>/corpus.jsonl`, sha256 and bytes copied here before stamping. It is uploaded to the
  bucket with `hf buckets cp` before the job, as the sealed file was. It cannot ride in the launch command:
  `ops/run.py` passes the script as one `bash -lc` argument (`run.py:1132`), and Linux caps one argument at 128 KiB.

Every `fusedrun` command line carries
`FLAGS='LLVQ_FUSED_LAYOUT=tetra48 LLVQ_ROT_SHARE=1 LLVQ_FUSE=0 LLVQ_KV=f16 LLVQ_EMBED=q4'`, as
`ops/jobs/gqa-ab.sh:88` does.

1. **Oracle**: `oracle Qwen/Qwen3-0.6B 64 cuda` (rule 10).
2. **The default path, replayed.**
   - `env $FLAGS fusedrun "$F" 256`: 256 tokens identical to the dense arm, and the decoded line hashes to
     `8784ffc4`, paper-table run 1.
   - `env LLVQ_CONFIG="$CG" fusedrun "$F" 256`: the served door under the grouped config. The whole decoded text,
     cut between `served : ` and `No dense arm on this path` as `ops/jobs/served-grouped.sh:133-134` cuts it, hashes
     to `0597cee8`, as in served-grouped-2026-11-02.
3. **Identity**, clip on. Six prompts: two of 5 tokens, one of 250, two of 2,048 and one of 8,190. Each prompt is
   prefilled once into a master cache, and every run starts from a fork of it.
   - Plain reference: 256 tokens, fixed length, logits kept. Identity is per position, so loops and EOS do not
     matter here.
   - Oracle drafter at K = 2, 3, …, 8: every verify row's logits against the plain step's at the same position,
     bitwise.
   - Adversarial at K = 4 and 8, and always-wrong at K = 8: token streams against plain.
   - 250 and 8,190 put verify rounds next to a multiple of 256. Information only: the oracle at K = 8 with the clip
     off, on those two prompts.
4. **t(k).** `verify_rows` of k = 1 to 8 rows, and the plain step, at 5, 2,048 and 8,192 positions, each from a fork
   of the master cache. 1 discarded and 5 timed calls each, interleaved; medians with ranges.
5. **The A/B.** Arms `plain` (the served decode with a host-side stop test) and `spec-pl` (prompt lookup).
   - Prompts: 12 extraction prompts of 1-4k tokens, 12 chat prompts (MT-Bench) and 3 extraction prompts at 6-8k,
     pinned by id and sha256 in the corpus. The extraction prompts come from G0's references with at least 64
     tokens before EOS.
   - One prefill per prompt, timed once: that is the request's time to the first token.
   - Then 1 discarded and 5 timed decodes per arm, each from a fork. Both arms run every round, alternating which
     goes first. Each decode runs to the first stop id or 512 tokens.
   - The loop flag (a 16-gram seen 3 times) is computed per prompt on the plain text.
   - Per round and per corpus: the token-weighted decode ratio Σ t_plain / Σ t_spec over loop-free prompts, and over
     all prompts. The request ratio (Σ TTFT + Σ t_plain)/(Σ TTFT + Σ t_spec) prints beside them.

## 4. Gates, each one voids the run

Infrastructure only.
1. The image is the Space sha named, and `oracle` prints MATCH.
2. The sealed file, the grouped config and the corpus at their bytes and sha256 on the mount.
3. Every load prints `int4: tv_q4_hh equals tv_q4_h bit for bit`, the served first-record check.
4. Step 2 prints `fused F0: 256 tokens identical to the dense arm` with the line hash `8784ffc4`. The served door's
   line reads `attention grouped on grow(256)`, and its whole text hashes to `0597cee8`.
5. Every arm line of the mode prints `layout tetra48`, `embed q4` and `grouped on grow(256)`. Every header reads
   `tile 64`. No `LLVQ_*` variable is in the job environment, and the five flags are on every command line.
6. The mode prints its four tables in full: parity, identity, t(k) and the A/B.

The parity lines, the identity results and the A/B texts are outcomes. Rules C and D read them; they void nothing.
The t(k) table does not depend on identity and is always printed.

## 5. Outcomes reported, never gated

- Per parity record: identical, or max |Δ| and the first row.
- Per identity check: bitwise or not. If not: the largest |Δlogit|, the first position, and the first layer whose
  hidden state differs (a per-layer dump on the first failing row). The straddle runs, clip off.
- The t(k)/t(1) table at 5, 2,048 and 8,192 positions, and β = (t(k)/t(1) − 1)/(k − 1).
- Per corpus:
  - τ, c, τ_d, and tok/s per arm;
  - the decode ratio (median and range, round by round) and the request ratio;
  - tokens per arm and the loop flags.

## 6. Signed predictions

| quantity | point | interval | label |
|---|---|---|---|
| every parity line identical | yes | | *estimated* |
| oracle identity at 5 and 2,048, clip on | bitwise | | *estimated* |
| oracle identity at 8,190, clip on | bitwise | | *estimated* |
| straddle runs, clip off | bitwise | | *estimated*, lower confidence: stride |
| t(2)/t(1) at 5 positions | 1.50 | [1.33, 1.69] | *estimated* |
| t(4)/t(1) at 5 positions | 2.5 | [2.0, 3.1] | *estimated* |
| t(8)/t(1) at 5 positions | 4.8 | [3.6, 6.1] | *estimated*; a second packet adds a weight pass |
| t(4)/t(1) at 2,048 | 2.75 | [2.3, 3.2] | *estimated* |
| t(4)/t(1) at 8,192 | 3.15 | [2.85, 3.45] | *estimated* |
| **spec-pl / plain, extraction 1-4k, decode, loop-free** | **1.04** | [0.96, 1.20] | *estimated* |
| spec-pl / plain, extraction 1-4k, request | 1.01 | [0.99, 1.04] | *estimated* |
| spec-pl / plain, chat, decode | 1.00 | [0.96, 1.02] | *estimated* |
| spec-pl / plain, extraction 6-8k, decode | 1.00 | [0.94, 1.14] | *estimated* |
| prompts flagged as looping, of 27 | 1 | [0, 4] | *estimated* |

Reasoning, *estimated*, itemized in `speculation.md` §3.3. An extra verify row costs:
- the Tetra rows kernel's per-row work, 0.93 ms;
- the int4 fan-out, 0.8 ms;
- one head launch, 0.32 ms (*computed*);
- `regroup`'s copies, 0.875 ms;
- 216 attention launches, 0.76 ms;
- the KV read, 0.668 µs a position.

Against the grouped step (7.43, 8.78 and 12.90 ms, *measured*), that is β = 0.50, 0.58 and 0.71. Prompt lookup at
c 0.45 and τ_d(4) 2.6 then gives ×1.05 at 2,048 with k = 2, the best k there. The first version of this prereg
predicted 2.15 for t(4)/t(1) at 5 positions and 1.15 on extraction. Both came from the prefill path, which batches
attention, launches one head per pass and has no per-row KV term.

## 7. Decision rules

Primary criterion, on the throughput axis: the median of the 5 round-by-round token-weighted decode ratios
spec-pl/plain on loop-free extraction prompts of 1-4k tokens, with their range, same text. The raw ratio is also the
same-head ratio here: both arms run the same head kernel. The request ratio is printed beside it in every rule's
report. The primary is "formed" when at least 6 of the 12 extraction prompts are loop-free.

The rules are read in order, and the first that matches applies.
- **C.** Any identity check fails at 5 or 2,048 positions, or any parity line differs.
  - No speed claim. The A/B table is printed and labelled, "not the same text" where the texts differ.
  - The per-layer diff, or the parity line, names the op.
  - A fix that keeps bits gets its own run, for instance design (b) for an int4 record. Otherwise the exact verify
    waits for the brick, and any tolerance route is the operator's T8 decision.
- **D.** Identity and parity hold at 5 and 2,048 positions, identity fails at 8,190 with the clip on, and the primary
  is formed. Rules A and B apply to the 1-4k rows only, and the long-context path waits for its fix.
- **A.** Identity holds everywhere, every parity line is identical, the primary is formed, its median is ≥ 1.10 and
  every round is above 1.00. Propose a served config field for prompt-lookup speculation (ruling R7), gated by the
  reference line; the proposal states the request ratio.
- **B.** Identity holds everywhere, every parity line is identical, the primary is formed, and its median is below
  1.10 or some round is at or below 1.00.
  - The code stays as plumbing behind the measurement mode.
  - Zero-byte drafting waits for B2 (and B3).
  - The measured t(k) replaces the estimated β in every estimate.
- **Otherwise** (no rule above matches, for instance fewer than 6 loop-free extraction prompts): report every table;
  no proposal.

The straddle runs with the clip off never enter these rules.

Any stop is a proposal; the operator decides.

## 8. Cost

*Estimated* on l40sx1 at $1.80/h:
- container and pull 2 min; oracle 1.5; step 2 2.5 (the bench loads its 8 GB dense arm); two loads with every parity
  check, under 1 min;
- identity 4 min: the 8,190 prompt's prefill is 35 s, and each run decodes 256 tokens from a fork;
- t(k) 1.5 min;
- the A/B about 22 min. The prefills take 3.6 min, at about 4 ms a prompt token. The decodes take about 18 min:
  2 arms × 6 rounds per prompt, extraction answers about 300 tokens at about 9 ms, chat answers up to 512 at about
  7.5 ms, and the 8k answers at 13 ms.

In all about 33 min, **$1.00**. Timeout 60 min, **$1.80** at worst. Anchors in `docs/data/jobs.csv`: `gqa-ab`
(575 s, $0.29), `served-grouped-4b` (1,082 s, $0.54), `trims-contexts` ($0.35).

## 9. What this run does not establish

- Neural drafters in the engine, and any flat verify.
- The 8B and the 14B (`Cat` store; the `Expanded` on `Grow` pairing is its own run).
- Quality: speculation changes no output when identity holds, so there is nothing to score.
- Metal: `rows_per_launch` is 1 there.
- Long-context drafting value beyond 3 prompts.
