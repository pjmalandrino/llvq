# Preregistration. Decode-speed research phase at long context, R2: the KV capture of the sealed 4B and the mass score that can kill the phase

Written on 2026-11-14 on branch `recherche/debit-kv-2026-11-14`, once the code it describes existed and its Mac gates
had run (§3). Timestamped before the first measurement, then never edited; a departure goes to
`preregistration-recherche-debit-kv-capture-2026-11-14-ECARTS.md`.

One job, launched by the operator: `ops/jobs/recherche-debit-kv-capture.sh`. Cost: about $0.36 (*estimated*, §6).
Cap: $0.90 (the 30 min timeout, *computed* at $1.80/h). Spent before it: $0 of the phase's $10 (CERVEAU O13).

## 1. The question

A decode token of the sealed 4B after 8,192 tokens reads the whole KV cache: 1,208 MB (*computed*), beside 1,378 MB of
weights. The attention phase is 2,646 of the 6,211 µs of that token (*measured*,
`docs/mesures/large-4e-hors-poids-etude-2026-11-12.txt`). Reading part of the cache has never been measured here.

This run asks one thing. With the skipped softmax mass held under 1e-3 in every query head, what fraction of the KV
bytes must a decode row read? The run answers it for three selectors that bound the skipped mass and one oracle. If
the exact page maximum, the best case of any per-page bound, still reads more than half, no summary can save the
idea for a certified mass bound.

## 2. What the code holds

Commits: the capture at `ce9c286`, its registration at `b338de3`, the scorer and the job script at `158aea5`; this
prereg in the commit after them. No served path moves (§2.3).

### 2.1 The capture (`llvq-llm/src/kvcapture.rs`, `LLVQ_KV_CAPTURE` in `bin/fusedrun`)

`LLVQ_KV_CAPTURE=<ctx>,<tokens>,<corpus>,<window> fusedrun <sealed> <ignored> <out_dir>`.
- The window is tokens `w x (ctx + tokens)` to `(w + 1) x (ctx + tokens)` of the tokenized corpus. A corpus too short
  is refused by name, never shortened.
- The corpora: `wikitext2-test` (`corpus::wikitext2_test`, LLVQ_GQA_PPL's text), `c4` (the evaluation shard,
  `corpus::c4_validation`), `dclm-edu` (`corpus::dclm_edu_calibration`).
- The attention is the kernel0 arm of `LLVQ_GQA_PPL`: store grow(256), the kernel form on every decode row from
  position 0 (`set_kernel_from_for_measurement(0)`), the kernel built and checked against its CPU replay at load.
  4E equals this kernel bit for bit on the attention (R0 journal, item 3).
- Every token of the window goes through `Qwen3::hidden_cached` one at a time, teacher-forced. The first `ctx` only
  fill the cache.
- For each of the last `tokens` tokens and every layer, the query after q_norm and RoPE is kept, `[32][128]`
  binary16.
- After the last token, for every layer: K as stored, `[8][L][128]` binary16, and the L2 norm of every V row,
  `[8][L]` f32, with `L = ctx + tokens`.
- The self-check pair: the attention output (the input of `o_proj`) of the last token at layer 18 as the engine
  computed it, and that layer's V as stored.
- `meta.json`: shapes, the window's token ids, the sealed file's sha256 computed in the process, c = 1/sqrt(128), and
  the next-token NLL of the captured tokens as a sanity line.

### 2.2 Which positions a row attends

`Block::attention_ctx` appends the new key and value to the cache before any form attends. So the row of the token at
absolute position p attends positions 0..=p, its own key included. Captured token t (0-based) sits at p = ctx + t and
attends L(t) = ctx + t + 1 positions, the first L(t) of the final dump. The cache is append-only, so those positions
hold the bytes that row read. With ctx = 8,192 and 256 tokens, L runs from 8,193 to 8,448.

### 2.3 The hook and the served path

The hook is one default method of the `Capture` trait, `on_attention_query`, called once a block after RoPE. Its
default body is `Ok(())`. The seven other implementers of `Capture`, `NoCapture` included, keep that body; only
the capture's private `QueryTap` overrides it. Every served path passes `NoCapture` or runs 4E, which does not enter
`attention_ctx`. The attention output is read through the existing `Act::AttnOut` activation hook. No tensor operation
was added to any path that does not name the capture.

### 2.4 The scorer (`ops/kv_mass_score.py`, numpy, run on the Mac at $0)

For every captured token t, layer and KV head g with its 4 query heads, the scores s_j = c <q_h, k_j> over the L(t)
positions are recomputed in float64 from the binary16 dump.

The layout of a row:
- dense: positions [0, 64) (sinks) and [64 + nP, L), where n = (L - 256 - 64) div P. That is the recent 256 positions
  and the fewer than P left over between the last full page and them;
- pages: n pages of P positions, aligned to absolute positions from 64, P in {16, 32, 64}.

A page is skipped for a KV head only if it is skipped for its 4 query heads. For each delta in {1e-2, 1e-3, 1e-4}
and each P, four selectors:
1. `true-mass`, the oracle: pages in increasing order of their true mass (max over the 4 heads); the longest prefix
   whose summed mass stays at most delta in every head is skipped.
2. `page-max`: the bound on a page's mass for head h is P exp(s_max - m) / Z_lo, with the page's exact maximum score,
   Z_lo the partition sum over what is read, m its shift. Pages are read in decreasing max_h P exp(s_max - m) / Z_dense
   until the bound on the rest is at most delta in every head.
3. `box`: per page and channel the min and max of K; U = c sum_i max(q_i min_i, q_i max_i); the rule of 2 with U for
   s_max. Charged 512 B a page a KV head.
4. `ball`: centroid mu and radius rho of the page's keys, in float64; U = c (<q, mu> + |q| rho). Charged 258 B a page a
   KV head.

Bytes read are 512 B a position a KV head (K and V) over the dense positions and the read pages, plus the summaries
of every page, over 512 B x L(t). They are pooled over layers and KV heads by bytes, then summarized as median, p10
and p90 over tokens.

Every skip is asserted: the actual skipped mass of each head is at most the bound the selector used, times
(1 + 1e-9), and the bound is at most delta. Before scoring, the scorer recomputes the self-check pair and refuses a
capture whose relative error passes 1e-2.

## 3. The Mac gates (*measured*, 2026-11-14, one job at a time through the guard, 3 test threads)

| gate | command | result |
|---|---|---|
| capture, parser, layout, self-check | `cargo test -p llvq-llm --lib kvcapture` | `test result: ok. 3 passed; 0 failed` |
| modes run alone, the new one included | `cargo test -p llvq-llm --bin fusedrun refuse every_pair extras` | `test result: ok. 4 passed; 0 failed` |
| the kernel form through the replay, unchanged | `cargo test -p llvq-llm --lib model::tests::the_kernel_form` | `test result: ok. 1 passed; 0 failed` |
| the variable registry | `cargo test -p llvq-llm --test env_registry` | `test result: ok. 2 passed; 0 failed` |
| measurement modes refused beside 4E | `cargo test -p llvq-llm --test served_config measurement_modes` | `test result: ok. 1 passed; 0 failed` |
| lints | `cargo clippy -p llvq-llm --all-targets` | no warning |
| the Linux half | `ops/check-linux-on-mac.sh` | exit 0, no warning |
| the scorer | `uv run ops/kv_mass_score.py --selftest` | `SELFTEST PASS` |
| the job | `DRY_RUN=1 bash ops/jobs/recherche-debit-kv-capture.sh` | `DRY_RUN: job script parses; nothing launched` |

What the capture test shows, on a two-layer model with head dim 128 under the kernel's CPU replay:
- the logits and both caches are bit for bit those of the same feed with `NoCapture`;
- every file has its shape, and a second write to the same directory is refused;
- the self-check output, recomputed in float64 from the dumped query, K and V over positions 0..=p, matches the
  engine to 3.0e-4 of its largest value at 14 positions and 2.0e-4 at 3;
- over 0..p at 3 positions the error is 5.9e-4, three times larger. The tiny model's attention is near uniform, so
  this test separates the own key by a factor 3 only, and it cannot show that the query matters (1.98e-4 with the
  previous token's query). The scorer's self-check on the card's data is the gate that can.

The scorer's selftest, on synthetic captures in the capture's file layout (L = 1,501 to 1,508):
- one hot key (a sink) in every group: every selector skips every page at every delta and P; true-mass reads the
  dense share exactly (0.2146 at P = 16);
- one key repeated everywhere, so a page holds P / L of the mass: nothing is skipped at 1e-3; fractions 1.0000,
  1.0000, 1.0491, 1.0247 at P = 16 for true-mass, page-max, box, ball;
- concentrated attention on six runs of keys: 0.3022, 0.3641, 1.0491, 1.0247 at P = 16 and 1e-3;
- true-mass <= page-max <= box and page-max <= ball in pages read, on every row of the three cases.

| mutant | verdict |
|---|---|
| the factor P dropped from the page bound | killed by the bound assertion on all three cases |
| min where max is due in the box bound | killed on the one-hot and the concentrated cases |

A full-shape synthetic capture (36 layers scaled from 2, 32 heads, 8 KV heads, L up to 8,448) scores in about 1.7 s a
layer with a peak resident set of 0.83 GB on this Mac.

Not verified on the Mac: the capture on the card, the corpora's downloads, the write speed to the bucket mount.

## 4. What runs

`ops/jobs/recherche-debit-kv-capture.sh` (l40sx1, output `/out/recherche-debit-kv-capture-2026-11-14/`):
1. The oracle; the sealed 4B's size and sha256 `886391a8...`.
2. Three captures, `LLVQ_KV_CAPTURE=8192,256,<corpus>,0`, with the served flags (tetra48, embed q4, ROT_SHARE 1,
   FUSE 0, KV f16), in this order: `wikitext2-test` (dataset revision 740312ad, LLVQ_GQA_PPL's), `c4`, `dclm-edu`
   (revision main). Each has an 8 min timeout and writes `<corpus>/`. Each is checked and listed before the next
   starts: the sha256 line, the kernel0 attention line, 72 K and V-norm files, the four other files.
3. The scoring is done afterwards on the Mac: `uv run ops/kv_mass_score.py <dir>` for each text, then
   `--rule` over the three JSON files.

## 5. The reading, fixed before the run

- The score is the byte fraction read at delta = 1e-3, median over tokens, pooled by bytes, on each of the three
  texts.
- KILL: if selector `page-max` at its best P reads more than 0.50 of the KV bytes on two texts of three, the idea is
  dead for certified bounds of the max type: no summary of that type beats the exact page maximum. The scope of the
  kill is that type, page size times the exponential of a bound on the page's largest score, under the scorer's
  greedy order. It does not cover a summary that bounds a page's sum of exponentials directly, nor a selector that
  certifies nothing. The `true-mass` rows say how much room such a summary would have.
- ALIVE: if selector `box` at its best P reads less than 0.35 on two texts of three, a real summary is enough and the
  phase goes on to the perplexity arm.
- BETWEEN: the summary is the problem; the phase's search (R3, R4) is over summaries, scored by this same scorer at $0.
- What the run does not say: how a mass budget a layer becomes perplexity over 36 layers; any speed.

The order of the tests is KILL, then ALIVE, then BETWEEN; `ops/kv_mass_score.py --rule` applies it. A capture whose
self-check fails, or whose bound assertion fails, is void and is not scored.

## 6. Cost

*Estimated* from measured times:
- The feed. `LLVQ_GQA_PPL=8192,6,6,kernel` fed 6 windows of 8,192 tokens one at a time through two arms in lockstep
  in 940 s (`docs/mesures/attn-t8-r2-2026-11-04-brut/ppl-8192.txt`, *measured*). That is 156.7 s a window, 78 s an
  arm, 9.6 ms a token. One arm over 8,448 tokens: about 81 s.
- The load 4.3 s and the kernel check 1.1 s (same log). The sha256 of the 1.42 GB file: a few seconds.
- The dclm-edu shard is 2.91 GB downloaded whatever is read (`corpus.rs`): 30 to 60 s assumed. The other two corpora
  are tens of MB.
- The writes: about 725 MB a capture, 15 to 60 s assumed. No journal measures the mount's write speed.
- The oracle: about 1 min.

| what | estimate | cap |
|---|---|---|
| 60 + 3 x (81 + 10 + 60 + 60) s at worst | about 12 min, $0.36 | 30 min, $0.90 |

Bytes to the bucket (*computed*, L = 8,448): q 75.5 MB, K 622.9 MB, V norms 9.7 MB, the check layer's V 17.3 MB:
about 725 MB a capture, 2.18 GB for the three.

## 7. Not established by this run

- Perplexity. A mass budget per layer and head has no measured link here to the paired perplexity over 36 layers.
- Speed. No kernel reads part of the cache; nothing is timed.
- The 8B and the 14B, and any context but 8,192.
- Texts beyond window 0 of three corpora. A C4 window of 8,448 tokens is many short documents joined by blank lines.
  WikiText-2's window 0 spans several articles. The dclm-edu window 0 is text the sealed files were calibrated on.
- The prefill path. The cache here is filled by decode rows, as in LLVQ_GQA_PPL's decode arm; the served door fills
  it by a chunked prefill, whose K and V differ in rounding.
- A stored summary. The box's min and max are exact binary16 values of K. The ball's centroid and radius are float64
  here; stored in binary16 they would need outward rounding.

## 8. Signed predictions (main session)

Written before the stamp. Scores are the byte fraction read at delta = 1e-3, median over tokens, pooled by bytes.

1. The verdict: KILL 50 %, BETWEEN 35 %, ALIVE 15 %.
2. `page-max` at its best P reads 0.60 [0.35, 0.85] on WikiText-2. `true-mass` reads 0.40 [0.20, 0.65]. `box` reads
   0.90 [0.60, 1.05].
3. At delta = 1e-2, `page-max` reads 0.40 [0.20, 0.65] on WikiText-2.
4. With the box at P = 16, layers 0 and 1 each read more than 0.90. 80 %.
5. The three texts agree on the verdict. 65 %.
6. The self-check passes on the three captures. 85 %.
7. The job runs 720 s [500, 1,100]. It is void, in whole or in part, at 20 %.

Why the priors are low: the published statistics are poor (the top 1 % of keys holds 89 % of the mass at 100K on
Llama-3-8B), a page bound of the max type charges a page up to P times its real mass, and a page is skipped only if
its four query heads all allow it.
