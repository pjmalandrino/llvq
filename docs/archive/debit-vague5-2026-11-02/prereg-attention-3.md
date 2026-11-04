DRAFT, not stamped, not committed

# Preregistration. The q8 cache at 8,192 positions on the sealed 4B: perplexity by prefill and by decode

**Draft written on 2026-11-02, to be committed and TIMESTAMPED BEFORE the run.** Operator go to launch: **not given**.
Operator ruling needed first: the band of §2, which selects the rule set of §7. Cost: about **$1.54** on l40sx1,
*estimated* below (§9), timeout 90 min, so $2.70 at worst. Cap asked: **$3**. Spent before it: $248.84 over 237 rows
of the registry, plus what earlier waves of this lead spend.

## 1. Why this run exists

A real int8 cache halves the cache bytes: 78,336 against 147,456 B a token at the 4B, 0.64 against 1.21 GB at 8,192
(*computed*, `llvq-llm/src/kvq.rs`, test `the_geometry_is_the_published_one`). The product triplet already counts a q8
cache in b_max = 3.00 (ROADMAP §4); the engine stores f16. The design of attention.md §3.5 makes the int8 store return,
bit for bit, the values of today's `LLVQ_KV=q8` round trip (`kvq::quantize_dequantize`). Its quality on today's
attention is therefore the round trip's quality, which this run measures before any int8 code exists. The pair the
lead would serve, the kernel with a q8 cache, is scored on its decode path by the third arm of the T8 control (V6).

The only q8 measurement is short-context and on another object: +0.0494 % [−0.0713 ; +0.1703] of perplexity and
−0.33 pp of MMLU [−1.22 ; +0.45], on the paper-1 file, Metal, dense path, 12 windows of 4,096 (*measured*,
`docs/mesures/kvq8-4b-2026-08-15.txt`). Its verdict kept q8 off by default for want of a long-context measurement. That
interval already crosses +0.12 %, the band the operator used for the grouped form. An optional screen on the Mac may
precede this run (attention.md §5, V5-pre); it screens and licenses nothing.

## 2. The band, an operator ruling before the stamp

The q8 cache changes what attention reads, so it is a quality change, not a rounding one. Two bands exist in the
repository; the operator picks one before this file is stamped, and only that band's rule set of §7 applies:

- **B1, the T8 band:** the paired 95 % interval of `ppl_q8 / ppl_f16 − 1` inside [−0.12 %, +0.12 %] (the noise bar of
  an A/B at constant file, ETAT §4, used by `gqa-ppl-2026-11-01`).
- **B2, the P3 band:** point ≤ +0.5 % and upper bound ≤ +1.0 % (`proofs/preregistration-p3-2026-08-14.md`, as reported
  in `kvq8-4b-2026-08-15`).

## 3. What the code holds

To be filled at commit time. The commit adds to `bin/fusedrun` a measurement mode
`LLVQ_KVQ_PPL=<ctx>,<windows>,<first>` (refused beside `LLVQ_CONFIG`, like every measurement mode): windows of `ctx`
tokens taken from window index `first` on, so that fresh windows are scored; the prefill path by chunks of 4,096
(`MAX_PREFILL_ROWS`), each chunk scoring all its positions; the decode path as a prefill of the first 4,096 tokens
followed by 4,096 teacher-forced steps in lockstep, scoring the second half; arms that name a `KvMode` as well as a
store and a form. Host tests and mutants listed then.

## 4. What runs

One job, l40sx1, the sealed 4B (`886391a8…`), config `qwen3-4b-tetra-e4-grouped.json` (`2455fe85…`), flags spelled
out. `ops/jobs/kvq-8k.sh`.

**The prefill path.** Both arms prefill through the path served at stamp time, named here before the stamp: today
`prefill_chunked` with the rows kernel, 4,096 rows a call, and cuBLAS grouped attention; the GEMM route instead if the
operator's branch has served it by then. The path is the same in both arms, or the run is void (gate 5).

1. **Oracle** on Qwen3-0.6B, rule 10.
2. **The bench door replayed**: gate 4.
3. **The control.** `LLVQ_KVQ_PPL=8192,24,7`: WikiText-2 test, windows 7 to 30 of 8,192 tokens (tokens 49,152 to
   245,759 of 299,078; windows 1 to 6 hold the 12 windows of 4,096 every earlier control scored). Arms
   `grow(256)+grouped, KV f16` and `grow(256)+grouped, KV q8`. Prefill path on all 24 windows; decode path on windows
   7 to 12.

## 5. Gates, each one voids the run

1. The image builds, and `oracle` prints MATCH.
2. The sealed file and the config at their sha256 on the mount.
3. Every load prints `int4: tv_q4_hh equals tv_q4_h bit for bit`.
4. Step 2: 256 tokens identical to the dense arm, decoded line `8784ffc4…`.
5. Step 3 prints the token fingerprint of windows 7 to 30, the prefill path of each arm (the same), 24 prefill lines,
   6 decode lines, and both summaries.
6. **The harness.** On the f16 arm, over windows 7 to 12, the decode path's perplexity on positions 4,096 to 8,191
   over the prefill path's on the same positions lies in [0.99, 1.01].
7. `tile 64` in every header, no `LLVQ_*` variable in the job environment.

## 6. Signed predictions

| quantity | point | interval |
|---|---|---|
| f16 arm, prefill ppl at 8,192, windows 7 to 30 | **11.8** | [10.5, 13.5] |
| q8 / f16 − 1, prefill path, point | **+0.05 %** | [−0.03 %, +0.15 %] |
| q8 / f16 − 1, prefill path, half-width of the paired interval | **0.06 %** | [0.03 %, 0.12 %] |
| q8 / f16 − 1, decode path (6 windows), point | **+0.05 %** | [−0.10 %, +0.20 %] |
| decode top-1 agreement, positions 4,096 to 8,191 | **99.5 %** | [98.5 %, 99.9 %] |

Reasoning, *estimated*. The quantizer works per token and per group of 64 along `head_dim`, so its error does not
grow with the context; attention over more positions averages it. The 4k point on another object was +0.049 %. Windows
of 8,192 carry twice the tokens of the 4k windows and 24 of them are scored, so the paired half-width should fall to
about half of the 0.12 % seen with 12 windows of 4,096. The known flaw: long-range retrieval inside an 8k window may be
more sensitive to key noise than the 4k average shows, and WikiText-2 holds little of it.

## 7. Decision rules

The prefill path decides; the decode path is the consistency check. P is the prefill point, [Lo, Hi] its paired 95 %
interval, D the decode point. Only the rule set of the band ruled in §2 applies; each partitions its outcomes.
Rejections are proposals: the kill is the operator's.

**Rule set B1** (band [−0.12 %, +0.12 %]), evaluated in this order:

1. **Accept:** −0.12 % ≤ Lo and Hi ≤ +0.12 %, and |D| ≤ 0.12 %. The q8 cache's quality at 8,192 is accepted for this
   object. The int8 store (attention.md §6, step 6) may be built and gated by identity with this round trip; its speed
   job needs its own prereg and go. Serving q8 by default stays the operator's decision, after V6's third arm.
2. **Reject (proposal):** Lo > +0.12 %. Proposed to the operator: no q8 default and no int8 store for the 4B, and the
   triplet's b_max read with an f16 cache, 2.86.
3. **Paths disagree:** the interval inside the band and |D| > 0.12 %. Nothing changes; the operator chooses between more
   decode windows from the fresh pool and stopping.
4. **Lack of power:** Lo ≤ +0.12 % < Hi, or Lo < −0.12 % ≤ Hi. Nothing changes; the operator chooses between more
   windows from another fresh pool and stopping.
5. **Otherwise** (Hi < −0.12 %, q8 better than f16 beyond the band, which points at the harness, or a gate failed):
   not settled; raw output kept; operator.

**Rule set B2** (point ≤ +0.5 % and upper bound ≤ +1.0 %), evaluated in this order:

1. **Accept:** P ≤ +0.5 % and Hi ≤ +1.0 %, and D ≤ +0.5 %. As B1's accept line.
2. **Reject (proposal):** P > +0.5 % and Lo > +0.5 %, which includes every Lo > +1.0 %. As B1's reject line.
3. **Paths disagree:** P ≤ +0.5 %, Hi ≤ +1.0 % and D > +0.5 %. As B1's line 3.
4. **Lack of power:** P ≤ +0.5 % and Hi > +1.0 %. As B1's line 4.
5. **Otherwise** (P > +0.5 % with Lo ≤ +0.5 %, or a gate failed): not settled; raw output kept; operator.

## 8. Deviations

Any departure from this file goes in `proofs/preregistration-kvq-8k-<date>-ECARTS.md`, beside it.

## 9. Cost

*Estimated* on l40sx1 at $1.80/h. Container, pull, oracle and step 2 about 8 min. Prefill path: per window two arms ×
two chunks of 4,096 rows at about 4.25 ms a row (*measured* rate, served-grouped), 70 s, so 28 min for 24 windows.
Decode path: per window two prefills of 4,096 (17 s each) and 4,096 lockstep steps at about 29 ms (the f16 arm near
12 ms; the q8 arm slower by its round trip, about 30 candle launches per tensor, 2,160 a step, near 4.8 ms), about
154 s, so 15 min for 6 windows. In all about 51 min, **$1.54** (`calc/attention/attention_v2.py`, section 9). If the
GEMM route is served first, the prefill path costs a few minutes instead of 28 (its time to the first token at 8,192
is 2.4 to 3.4 s, *estimated* in `wave5-deep/prefill.md`). Anchor: `gqa-ppl` scored 12 windows of 4,096 by both paths in
1,643 s for $0.82 (`docs/data/jobs.csv`). Timeout 90 min, **$2.70** at worst.

## 10. What this run does not establish

- The int8 store itself: its speed and its identity with this round trip are a later job (V8).
- The q8 cache under the kernel: V6's third arm.
- MMLU or GSM8K under q8 at 8,192.
- Anything at 8B or 14B, on another card, or on Metal.
