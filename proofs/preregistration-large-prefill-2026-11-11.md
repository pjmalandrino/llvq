# Prereg: the prompt prefilled through the tree pass read in place, chains of 16, sealed 4B

Written on 2026-11-11 on branch `claude/arbre-prefill-2026-11-11` (from `claude/arbre-nuit-2026-11-10` at 9797c7e),
once the code it describes existed. Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-large-prefill-2026-11-11-ECARTS.md`.

Go: the operator, 2026-11-11, card spending inside a **$15** cap; **$10.31** spent before this job, and another job of
at most $0.90 in preparation beside it; relayed by the orchestrating session. Registry (`docs/data/jobs.csv` on this
branch): **$317.31**. Cost: about **$0.72** on l40sx1 (*estimated*, "Cost"), timeout 32 min, **$0.96** at worst.

Measured code: the commit that holds this file. The image is published from a clean detached checkout of the commit
that holds this file's stamp. The first experiment of `docs/mesures/large-prefill-contexte-2026-11-11.txt` section 5,
with three departures from that note, decided before any run:

1. The head on the last pass only. The note found no switch for it; the a12 body (`pers_4et_a12.cu`, shared by a12 and
   a12ip) now skips the final norm and the head when a launch's `split_head` is 0, every wait kept. A launch with a
   non-zero `split_head` runs the same instructions as before. Host gate below; the card compiles it for the first time.
2. The hand-over is one a12ip launch of one row, the first answer token at the prompt's end, head off: it moves the
   last chunk into the cache, and 4E then decodes from that position (it computes that row again). No move-only launch
   exists, and none is written.
3. Perplexity on 12 windows of 2,048, not 16, to keep the job under 25 minutes.

## Question

How much sooner does the first token come when the prompt goes through the tree pass as chains of 16 known tokens
instead of the served eager prefill, at what peak card memory, and does it change the text 4E then writes, or the
perplexity?

## Setup

One image, one job, `ops/jobs/large-prefill.sh`, on l40sx1 only (no twin). Output `large-prefill-2026-11-11/` in
`hf://buckets/Pier-Jean/jobs-artifacts/`. Inputs: the sealed 4B (sha256 886391a8), `configs/qwen3-4b-tetra-e4-4e-v9.json`,
the house set `ops/data/spec-prompts-v1.jsonl` (sha256 f749eb35, 44 prompts), the long prompt
`ops/data/spec-long-8k-v1.jsonl` (sha256 da4edbd6, 8,191 tokens with the template, thinking off). The 1,024 and 4,096
prompts are cut from it by `prefillbench`: the template kept, the first content tokens (a cut may split a word). Every
length is exact. WikiText-2 raw test from the Hub (`llvq_llm::corpus::wikitext2_test`), windows 0 to 11 of 2,048.

Two `prefillbench` processes, `LLVQ_CONFIG` the v9 config, each beside two samplers (`nvidia-smi` memory.used every
100 ms for the process peak, and one stamped with the epoch every ~0.1 s for the peak inside each phase the binary
marks). GB = 1e9 B, MiB x 1,048,576 / 1e9.

- (A) `--form eager`: the served eager prefill (`Qwen3::prefill_next_in`, chunks of 4,096) then 4E, which is what
  `Decoder::generate_timed` and fusedrun's `LLVQ_CONTEXTS` time.
- (B) `--form chain --arm a12ip@8.10.8.20.6 --r 16`: the empty caches' Grow buffers made, the prompt as chains of 16
  (`spec_loop::chain_launches`: row r at depth r, each launch moving the previous one's rows first), the head on the
  last pass only, the hand-over launch, then 4E. No drafter, no taps, no weight copy.

Each process, in order:

1. Timing at 1,024, 4,096 and 8,191 tokens: 1 discarded + 3 timed rounds of prefill then 256 tokens of 4E greedy with no
   stop, tokens identical across rounds or refused. Time to the first token (chain: until the last pass's argmax is on
   the host), prefill until 4E can take over (chain: the hand-over included), decode tok/s after the first token
   (255 / decode wall). Median [range]. A `card memory:` line (cuMemGetInfo) after the load, after the tree install
   (B), after each length's rounds.
2. (B only) Identity: at 1,024, R = 16 against R = 1 (the pass's one-row path, a pass a token, each moving the
   previous); at 4,096, R = 8 against R = 16. Equal argmax, last row's logits bit for bit, and every layer's K and V over
   the prompt bit for bit (an FNV-1a hash a layer of the cache bits).
3. (B only) Phase split: the 8,191 prompt prefilled once more, the last three passes stamped (`pers::p4t_phases_a9`,
   wall per phase summed over the layers). Attention share of a pass = phase `attn` over the sum of the phases, median
   of the three.
4. Reading (i): the 44 house prompts, each prefilled by the form then 256 tokens of 4E greedy; with the three long
   prompts (their first round), the text to the first `<|im_end|>` included, dumped. (B) pairs with (A)'s dump: the
   count of identical texts of 47, and for each divergence its position and the pass's logit margin between the two
   tokens there (the chain prefill of the prompt and the shared tokens, last row's logits). A near-tie is a margin of
   at most 0.25 in logit, absolute. This definition is fixed here.
5. Reading (ii): the 12 windows, (A) by `Qwen3::window_nll` (the published prefill path, one forward), (B) by chain
   passes of 16 with the head and binary16 logits on every row, NLL in f64. Paired per window, the relative perplexity
   difference and its 95 % t interval (fusedrun's `paired_ppl_pct`, as `LLVQ_GQA_PPL`).

## Mac gates (before the stamp)

- Host twin (`llvq-cuda/tests/tree_verify_host.rs`, a12ip on the 4B's widths, two layers, from an empty cache):
  `chain_prefill_equals_the_one_row_path` (37 tokens, R = 16 and R = 5 against R = 1 on every row's argmax, logits,
  residual, K and V, and the hand-over row; head on the last pass only against head on every pass; the R = 1 pass at
  position 0 against 4E) and `chain_prefill_mutants_are_killed` (path off by one; a head guard that never skips). The
  journal carries their output.
- The earlier a12, a12ip and loop twin gates, after the kernel edit.
- clippy for the Linux target, zero new warnings.

## Controls

Each voids the run.

1. `oracle` prints MATCH; the file, the config and both prompt sets at their sha256; no `LLVQ_*` in the job environment.
2. 4E's module is the served v9 module (252,757 bytes, sha256 1b422b62) in both processes.
3. If the image fails to build, nothing is launched.

## Signed prediction

Signed by the orchestrating session, verbatim:

- host identity holds on the card: chain prefill equals the pass's one-row path on the last row's logits at 1,024.
- time to the first token, eager: 4.2 s [3.8, 4.8] at 1,024; 17.0 s [16, 18.5] at 4,096; 35.5 s [33, 38] at 8,191.
- time to the first token, chain R = 16: 0.95 s [0.6, 1.5] at 1,024; 5.4 s [3.5, 7.5] at 4,096; 16 s [10, 22] at 8,191.
- peak VRAM: eager 6.0 GB [5.7, 6.3] at 8k; chain 3.6 GB [3.0, 4.3] at 8k; chain no more than eager +0.25 GB at 1,024.
- at 8k, attention is at least 70 % of a chain pass's time.
- decode tok/s after the hand-over equals the eager arm's within 3 %.
- text after chain prefill identical to the text after eager prefill on 36 of 47 prompts [28, 44]; where it differs, the first divergence sits on a near-tie.
- Decision rule: PASSES for speed if the first token comes at least 3 times sooner at 1,024 and 2 times sooner at 8,191 with a peak no higher than eager's; quality is NOT settled by this job unless reading (ii) ran and landed inside +-0.12 %.

Operational reading, fixed here: "sooner" is the ratio of the two medians of the time to the first token. "Peak" is the
phase peak of each form's 8,191 timing rounds (the epoch sampler); the process peaks are published beside it. "Peak VRAM
at 8k" in the predictions is that phase peak. "Inside +-0.12 %" is the whole 95 % interval inside the band, as
`docs/ETAT.md` section 4 reads it.

## Decision rule

| result | outcome |
|---|---|
| first token at least 3x sooner at 1,024 and 2x sooner at 8,191, chain's 8,191 peak at or under eager's | PASSES for speed |
| otherwise | FAILS for speed, the ratios reported |
| identity at 1,024 fails on the card | the speed reading stands, the chain is not the pass's one-row path: the journal leads with it |
| reading (ii) ran and its interval sits inside +-0.12 % | quality inside the served tolerance on this reading; a GSM8K-200 pairing remains before serving |
| reading (ii) did not run, or its interval leaves the band | quality NOT settled |

## What gets published, and what does not get compared

Published: the loader's lines and `card memory:` lines, both samplers' files, the phase peaks, the timing table, the
identity lines, the phase split, both text dumps and the pairing, both NLL dumps and the pairing.

Not established: a served path (no engine or chat door calls the chain prefill); R above 16; one-read attention (D1,
D2); GSM8K after a chain prefill; 8B, 14B; any card but the L40S.

## Cost

M3 loaded a specbench in 426 s cold and 244 s warm (*measured*). Here: oracle about 1 min; (A) a cold load about 7 min,
timing about 4 min (the eager 8,191 prompt alone about 36 s a round, four rounds), house texts about 1 min, 12 windows
about 1.7 min; (B) a warm load with the tree install about 4 min, timing about 2 min, identity and phases about 1 min,
house texts and margins about 1.5 min, windows about 1 min. About 24 min, **$0.72** at $1.80 an hour (*estimated*).
Timeout 32 min: **$0.96** at worst; $11.27 of $15 at worst after it, $12.17 with the other job's $0.90.
