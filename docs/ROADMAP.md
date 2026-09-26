# Roadmap

What comes next, with its gate and its cost. State as of 2026-09-26. Where things stand is in
[`ETAT.md`](ETAT.md), the past in [`HISTORIQUE.md`](HISTORIQUE.md), the rules in [`METHODE.md`](METHODE.md). The
quality axis has its own document, [`ROADMAP-QUALITY.md`](ROADMAP-QUALITY.md), sanctioned 2026-09-06 and ordered by
feasibility.

## 1. Starting point

Three Qwen3 files are sealed and served at about 2.7 b/param, and paper 2 is written on them
([`ETAT.md`](ETAT.md) §2). They score 4.76, 4.21 and 2.46 MMLU points below 4-bit AWQ for 45 to 52% of its bits per
parameter, and the gap shrinks with size. What comes next is not another format. It is closing the gaps the paper
itself lists, in the order below.

Every A/B at 0.6B follows the Design C gate: 28 blocks, same seed, then three seeds. Any experiment that recalibrates
is read against 5.2% of perplexity and 2.92 pp of MMLU. An A/B at constant file is read against 0.43 pp and 0.12%
([`ETAT.md`](ETAT.md) §4).

## 2. Next, in order

### 2.1 Score the served object, not its reconstruction

**Gate.** The census through the kernel lands within the paired bar of the dense reconstruction, or the paper's tables
move. **Cost** about $20 by the mixed route, about $70 for the full census through the kernel at three sizes
(*estimated*).

Every MMLU we report is read on the dense reconstruction with f16 tables, while the b/param count int4 tables. The one
measurement that ties the two arms cost $2.85 and ran on 2,280 questions at 4B on the 2026-09-09 object: the kernel
scored 55.66 against the dense path's 55.52, three discordant questions out of 2,280, McNemar p = 0.25 (*measured*,
[f1e-census](mesures/f1e-census-2026-09-11.txt)). That bounds the risk at one size, on an older object, with q8 tables.

Code comes before the run: `ppl` does not read `LLVQ_CONFIG`, so no perplexity can be scored through the kernel today.

### 2.2 Time equal work in the kernel bench

**Gate.** `planesbench` times the same 252 matrices for every arm. **Cost** $0 for the code, about $1 for the run.

Today it times 216 matrices for `Tetra` and 252 for every other arm: the int4 `v_proj` are counted and not timed. Every
× formed on those passes compares unequal work, which is why paper 2 carries it as a limitation instead of a ratio.
`planesbench` also refuses `d_in` 17408, the 14B `down_proj` (*measured*, census-14b-base, `planesbench.rs:1917`).

### 2.3 `down_proj` instead of `o_proj` in int4, at 4B and 14B

**Gate.** The paired gain against the shipped file clears 0.43 pp. **Cost** about $2 a size, no encoding.

At 8B, `down_proj` of layers 10 to 26 alone scored 0.77 points better [0.26, 1.28] than `o_proj` plus `down_proj`, with
fewer bytes, against our own signed prediction (*measured*, [sealed-8b-27](mesures/sealed-8b-27-2026-09-24.txt)). The
4B and 14B files still spend their int4 budget the old way, and nothing says 8B's answer is theirs.

### 2.4 Perplexity for the three sealed files

**Gate.** None: this is a number the paper does not have. **Cost** about $1 a size.

The 14B chain measured 8.4622 against AWQ's 8.2858 on the trained base before sealing (*measured*,
[dclm-14b-rowscales](mesures/dclm-14b-rowscales-2026-09-22.txt)). No sealed file has a perplexity of its own, and a
quantization paper is read for that number.

### 2.5 A second calibration draw per size

**Gate.** The three absolute levels move by less than the 2.92 pp the draw carries at 4B. **Cost** a full encode a
size, plus the row-scale training on top of it. The 14B encode cost $15.35 on `rtx-pro-6000`, the re-export included,
and its training $9.78 on an h200; the 8B training cost $5.66 (*measured*, `docs/data/jobs.csv`). The 8B and 4B encodes
ran on the Mac and are not billed.

Each level is one draw. The chain gains of [`ETAT.md`](ETAT.md) §4 compare fixed files on the same questions and do not
depend on it, so this is about the levels, not the gains.

### 2.6 A second kind of GPU

**Gate.** Formulate it before the run. On an A100 none of our earlier lattice kernels beat FP16, and the best tile
already depends on the card (64 on sm_89, 32 on sm_120). **Cost** about $2 for a served decode at three sizes.

## 3. Debt and hygiene

- `[workspace.lints.rust] unsafe_code = "forbid"` and `[lints] workspace = true` on the five core crates.
  `#![forbid]` in a `lib.rs` does not cover integration tests, which are separate crates.
- Host compilation of the `.cuh` files by `clang++` in CI, on the model of `llvq-cuda/tests/host_e1v.cpp`. `ci.yml`
  does not carry it.
- Two timestamps no longer attest their file, 2026-08-10 and 08-11, rewritten by the anonymization pass `01fdbe6`. A
  third, `f5-graines-4b-2026-08-19.v1-l4x4.md.ots`, has no `.md` beside it at all. The attested bytes are
  unrecoverable. Nothing repairs this; it is recorded so no reader trusts those three.
- `docs/hf-model-card.md` carries 5.162 b/param and the card online has not been republished since 2026-08-17. Both
  describe the `Planes14` object, not the sealed files. Republishing is an operator decision.
- The HF bucket has never been inventoried: 69 files, 46.7 GB as of 2026-08-17. An inventory comes before any re-run
  quote (rule 9).
- `ops/status.py`, which would generate [`ETAT.md`](ETAT.md) from `mesures/`, `jobs.csv` and `otsaudit`, is not
  written. Until it is, that document is maintained by hand and can go stale.
- No tag points at the deposited commit `e21a8bb`. `v0.0.1` points at its child `16c9c8b` and contains it.

## 4. On hold

- **MoE.** Model settled: Qwen3-30B-A3B, gpt-oss ruled out. A policy for experts below full rank is missing: 31.4% of
  (layer, expert) cells, one dead expert, measured on gpt-oss-20b as a floor (*measured*,
  [moe-routing](mesures/moe-routing-gptoss20b-2026-08-12.txt)). About $1.4 to open, about $69 to serve (*estimated*).
- **q8 KV cache at long context.** Quality green at short context, interval containing zero (*measured*,
  [kvq8-4b](mesures/kvq8-4b-2026-08-15.txt)). Long-context throughput is unmeasured. Reopening needs a benchmark with
  a resident model.
- **Batch above 1, and prefill.** Batch 1 accepted since 2026-08-18, edge regime. The optimal format depends on the
  batch, so this reopens the layout if prefill is ever served.
- **The 32B point.** About $62 and 11.4 h on `rtx-pro-6000x2` (*estimated*). The served path is walled there by the
  `rot_apply` limit on `down_proj`, so an encode would produce a file nothing serves. A gate on the drop in the
  14B-to-32B gap comes first.
- **A model above 14B in general**, for the same wall ([format-noyau](format-noyau.md) §8).

## 5. Decisions awaited

| decision | default if silent |
|---|---|
| which route for the served census, $20 or $70 | neither, the tables stay unscored |
| a spend cap for the next campaign | no paid job |
| publishing the three sealed files, and where | nobody outside can replay an MMLU |
| next venue for paper 2 | preprint only |
| republishing the Hugging Face model card on the sealed object | the card keeps describing `Planes14` |
| `ots upgrade` after each new stamp | stamps sit un-upgraded |
| the 32B budget, once a gate exists | not launched |
| document-extraction domain benchmark ([arXiv:2607.08734](https://arxiv.org/abs/2607.08734)) | not done |

Rule 1 applies to every row: no run starts or stops, and no structural decision is taken, without an explicit go, with
the cost announced before and the running total after.
