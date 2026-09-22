# Preregistration. The paper-2 recipe at Qwen3-14B, step 1: the encoding, on a card (2026-09-22)

**DRAFT, NOT STAMPED.** To be TIMESTAMPED (`ots stamp`) before `seg1` is launched. The `.ots`
will attest these bytes, and the launcher refuses both jobs until it exists. The commit that
carries both follows on the operator's go. Operator go, 2026-09-22, verbatim: "allé lance moi le
14B". Card mode (x1, two segments) and the $52 cap for the whole 14B chain are the card plan's
proposals (tasks `wghg5aptb`), not in those words. The operator's verbatim words on both go here
before stamping.

Once stamped, not edited again. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

Two fields are filled at stamping time, and the stamp waits for them:

- measured code: the commit the rebuilt image is built from, `<commit>`;
- image: Space `Pier-Jean/llvq-runner-cuda` at `<IMAGE_SHA>`, the rebuild that adds `export`.

**Scope: this step only.** Two paid jobs, `seg1` and `seg2`, on `rtx-pro-6000` x1, produce the 14B
base file, seal it, score its perplexity and export it. **Cost $21.36 central, hard cap $24.75**
(*estimated*, below). The census, the row-scale training and the served checks each need their
own prereg. 14B chain spent before this step: $0 at writing. If the census B+C runs first under
its own prereg, its amount is added in the journal, and the arithmetic holds: $24.75 plus its
$4.05 ceiling is $28.80 of $52. Project total before: **at least $186.99** (*computed*: $183.80
over the 176 of 181 rows of `docs/data/jobs.csv` that carry an amount, plus $3.19 per `hf jobs
inspect` for five jobs the registry does not hold, as counted in
`preregistration-dclm-8b-2026-09-21.md`). After: at most $211.74, or $214.31 if both jobs overrun
their timeout by the 28 min the platform has billed before.

## The card, declared first

The 4B and 8B paper-2 bases were encoded on Metal. This one is encoded on an NVIDIA RTX PRO 6000
(Blackwell, sm_120 by PTX JIT from the image's sm_89). The device is therefore a confound on
every comparison across sizes.

One pair measures it, at 4B. The same bare `Tetra` recipe, C4, the same 131,072 tokens, encoded on
`rtx-pro-6000x2` (`volume-v1c`) and on Metal (2026-09-06), scored on the card: sealed f16
perplexity 17.6681 against 16.1569, **+9.35 %**; MMLU 52.31 against 53.49, **−1.18 pp** on 2,280
questions (*measured*, `docs/data/jobs.csv:120`; the journal `volume-2026-09-07.txt` it cites is
not in the repository, the smoke log is in the bucket under `volume-2026-09-07/v1c/`). There is
no paired interval on the file. Against the 4B calibration draw, σ 5.2 % in perplexity and
2.92 pp in MMLU, the card term is 1.8 σ and 0.4 σ. Its cause is not established. TF32 is off:
candle 0.9.2 leaves it off by default and the repository never enables it. `calib.rs`
accumulates AᵀA in f32 on the accelerator, which is the hypothesis `jobs.csv:120` names, not a
proof.

What it touches: R, and later the MMLU, of the 14B set beside the 4B and 8B. What it does not
touch: any comparison inside the 14B. Base, trained arm, f16, AWQ and the served kernel are all
scored on cards, from one encoded object. A 4B bridge on the card costs about $5.10 and buys one
pair; resolving −1.18 pp at 2 σ needs about 50 pairs, about $250 (*computed* in the card plan).
The confound is declared, not bridged.

Three smaller changes, declared with it:

- **Two segments.** The run is cut at block 20 and resumed (`LLVQ_RESUME`). The hidden states are
  recomputed on the card, not restored, so the two-segment file is not guaranteed to equal a
  one-piece run to the bit (`smoke.rs:62-69`). On CPU the two are byte-identical
  (`tests/resume.rs`, and a full 0.6B run, `ops/README.md` C4). The caveat is printed on the
  result line (`smoke.rs:1369-1380`).
- **23 encoder threads instead of 12.** Not a confound: `parallel_matches_serial_exactly` pins
  the loop bit-exact whatever the count (`smoke.rs:811-813`).
- **The memory record.** A 30 s sampler of `nvidia-smi` and `/proc` replaces `/usr/bin/time -l`,
  which the runtime image does not carry.

## Question

Does the paper-2 recipe (`Tetra`, `v_proj` in int4 g128, calibration on `dclm-edu`), encoded on
a card, give at 14B a base file whose encoding perplexity stays within the 8B precedent once the
card term is allowed for?

This step carries a measurement, not a gate on the recipe. At 4B the base perplexity did not
predict MMLU, and the row-scale training moved it from ×1.328 to ×1.007 of f16 (*measured*,
`dclm-rowscales-2026-09-20.txt`). What it informs is whether the rest of the 14B chain is paid.

## Setup

Launcher: `ops/jobs/encode-14b.sh` (sha256 in the journal and in `provenance-seg{1,2}.txt`). Both
jobs go through `ops/run.py bench --any-flavor`: the `bench` whitelist holds `l40sx1` only, and
the override is declared here, as `run.py` asks. `run.py launch` cannot pass `LLVQ_INT4_TYPES`.

The recipe, in both segments:

```bash
env LLVQ_MODEL=Qwen/Qwen3-14B LLVQ_CALIB=dclm-edu LLVQ_INT4_TYPES=v_proj LLVQ_THREADS=23 \
    LLVQ_ARTIFACT=<out> [LLVQ_RESUME=<shard>] \
  smoke 64 2048 12 4096 cuda nogs tetra1 <20 | 40> rot
```

It is the 8B recipe (`preregistration-dclm-8b-2026-09-21.md` §Setup) with `cuda` for `metal` and
23 threads for 12: `tetra1`, `dclm-edu` 64 × 2048 from the prefix, rotation seed `0x110feed`,
`nogs`, damping 1e-2, f32, wikitext-2 test 12 × 4096. Unset and at default: `LLVQ_DTYPE`,
`LLVQ_DAMPING`, `LLVQ_H_SHRINK`, `LLVQ_GAIN_SCALE`, `LLVQ_SEQ_BLOCK`, `LLVQ_CALIB_SEED`. The job
refuses any inherited `LLVQ_*`, and refuses a log that prints "spherical feedback on" or a build
without `fast-linalg`.

Revisions: `Qwen/Qwen3-14B` at `40c069824f4251a91eefaf281ebe4c544efd3e18`, `dclm-edu` at
`dbad8ad71224482740cd9c9d353591adbf62fe04`. The Mac checks both before each launch. The job
reads the dataset revision off `smoke`'s log, and the checkpoint revision off `refs/main` of the
hf-hub cache. That cache is `$HOME/.cache/huggingface/hub`, not `HF_HOME`: `Api::new()` builds
`Cache::default()` (hf-hub 0.4.3, `api/sync.rs:229-231`). Each job refuses a revision other than
`40c06982`: `seg1` before its `DONE` marker, `seg2` before the seal. If no ref is found, the job
records that and the Mac's check is the only guard.

**seg1** (timeout 255 min):

1. `oracle Qwen/Qwen3-0.6B 64 cuda`, `MATCH` (hard rule 10).
2. The gate. The exact recipe on `Qwen/Qwen3-0.6B`: block 0 into a shard, then resumed to block 1.
   `tetra1`, int4 records and a resume have never run on a card. A failure here costs about
   8 min, $0.37.
3. `Qwen/Qwen3-14B`, bound 20: blocks 0..19, the shard written straight to
   `/out/dclm-14b-seg1-2026-09-22/`, as every card encode has written its `.llvq`. The writer is
   closed normally at the bound, `verify_artifact` reads the 140 records back, and the `.state`
   says `blocks_done = 20`. Then sha256 of both files and a `DONE` marker.

**seg2** (timeout 285 min), launched only after the Mac has checked the shard's byte count in the
bucket, the `DONE` marker, the image and the partial ratio:

1. `oracle`, as above.
2. The shard and its `.state` copied to `/scratch`, checked against `seg1`'s sha256.
3. Bound 40, `LLVQ_RESUME` on the copy: blocks 20..39, the whole `.llvq` straight to
   `/out/dclm-14b-2026-09-22/`. `smoke` verifies all 280 records against the evaluated model.
4. `LLVQ_MODEL=Qwen/Qwen3-14B seal` to `/scratch`, copied to `/out`, size and sha256 read back.
5. `LLVQ_DTYPE=f16 ppl 4096 12 cuda` on the sealed file. Outside 1 % of the encoding, the job
   stops before the export.
6. `export` to `/scratch`, the four files copied to `/out/dclm-14b-export-2026-09-22/` with
   sizes and sha256. The export is last on purpose: while `seg2` stays inside its estimated
   range, whose top equals its timeout, a cut can only land there. A loop more than 16 min over
   its range cuts the seal or the sealed perplexity instead; the `.llvq` is on `/out` by then.
7. `rtbits` is not in the image. `encode-14b.sh fetch` runs it on the Mac, on the fetched sealed
   file, after checking its sha256 against the job's.

## Controls

If one of 1 to 7 fails, no number from this step is published and no further paid 14B job is
proposed. 8 to 10 are records: a missing one is reported and the rest stands.

1. `oracle`, CUDA, f32, in both jobs: `MATCH`.
2. The 0.6B gate: both invocations verify bit for bit, kinds `Tetra+Int4G128`, 1 int4 record of
   7; the resume copies 7 records.
3. Header kinds `{Tetra, Int4G128}`; **40 int4 records out of 280** (36 of 252 at 8B); 20 of 140
   in the shard.
4. `verify_artifact` inside `smoke` reads back **13,212,057,600** weights bit for bit at f32
   (6,606,028,800 for the shard). Zero points outside Λ₂₄: the guard of `Tetra::encode` refuses at
   write time.
5. The resume: `seg1`'s `DONE`, `blocks_done = 20`; the copy matches `seg1`'s sha256; 140 records,
   6,606,028,800 weights and 20 blocks reloaded; `blocks_done = 40` at the end; the result line
   carries `segments = resumed at block 20`; both jobs resolved checkpoint revision `40c06982`
   wherever the cache holds a ref.
6. The sealed file opens: `format v5, 280 quantized matrices`, `carrying 163 tensors, 1556249600
   weights`. Its copy on `/out` equals the original in size and sha256. Its f16 perplexity on the
   card, same 12 windows, fingerprint `3f1baca9033bf251`, is finite and within 1 % of the
   encoding perplexity. Precedents: −0.18 % to +0.06 % on the seven encodings the 8B prereg lists,
   −0.027 % at 8B, −0.27 % on v1c (17.7163 to 17.6681).
7. `encode-14b.sh check`: every byte count in the bucket within 0.01 % of the prediction below.
   `hf buckets ls` is the authority: the mount has truncated a 4 GB write with no error
   (`ops/README.md`).
8. The export: `443 tensors identical bit for bit`, `40 from Int4G128`, four files with sizes and
   sha256. A failed export does not void 1 to 7: it is rerun alone.
9. `rtbits` on the Mac: the `b/param WHOLE MODEL` row `f16 (served)`, both columns; both kernel
   lines.
10. Memory: GPU memory used, sampled every 30 s; host `VmHWM` of each binary (`mem.csv`,
    `peaks.txt`).

## Retention

Both jobs' output directories and the Mac's `$HOME/q14b-dclm-2026-09-22/` (provenance, fetched
logs, `rtbits.txt`) go to `docs/mesures/encode-14b-2026-09-22-brut/`. The journal is
`docs/mesures/encode-14b-2026-09-22.txt`, and both jobs get a row in `docs/data/jobs.csv`. All of
it is committed on the operator's go. The objects stay in the bucket: `dclm-14b-seg1-2026-09-22/`,
`dclm-14b-2026-09-22/`, `dclm-14b-export-2026-09-22/`.

## What gets published, and what does not get compared

Published: R, the partial ratio of `seg1` as a record, the sealed f16 perplexity, both jobs'
durations and phase profiles, the byte counts, `rtbits`, the sampled memory peaks, the cost.

Not compared: the 14B R against the 4B or 8B R as a size effect without the card term beside it;
`Planes14` 14B as a format verdict (`leech1c12`, encoded in bf16 on a card, another corpus);
the segmentation as a result; any MMLU; the training; the kernel.

## Decision rule

`R` is quantized over baseline perplexity, both from the `exact-ppl` line of `seg2`'s `smoke`
(`smoke.rs:1402`), same process, f32, on the card, to four decimals.

The reference is the 8B paper-2 base, ×1.1983 (*measured*, Metal f32, `dclm-8b-2026-09-21.txt`),
times the card term ×1.0935 (*measured*, one 4B pair): **×1.3104** (*computed*). The width is the
4B calibration σ of 5.2 % on perplexity (*measured*, three seeds of `leech1c12` on C4,
`f5-graines-4b-2026-08-19.txt`; never measured on `Tetra`, at 14B or on a card). One draw above
the reference is ×1.3785, two draws ×1.4502. The floor is the 8B reference without the card term,
one draw down: ×1.1391 (*computed*). No row kills: the kill stays with the operator, on the
trained arm's MMLU.

| result | reading | what follows |
|---|---|---|
| controls 1-7 pass, R < 1.1391 | better than the 8B Metal base by more than a draw, before any card term | check before reading (METHODE §7), then report; operator decides |
| controls 1-7 pass, 1.1391 ≤ R ≤ 1.3785 | within one draw of the 8B precedent widened by the card term | report; the 14B chain continues under its own preregs, inside the $52 cap |
| controls 1-7 pass, 1.3785 < R ≤ 1.4502 | worse than one draw, within two | report; operator decides whether the h200 training ($10.33 central, $15.00 ceiling) is worth paying |
| controls 1-7 pass, R > 1.4502 | worse than two draws | no further paid 14B job; report with the phase profile and a proposed diagnosis; operator chooses |
| a control of 1-7 fails, R not finite, or a job dies | no usable file | nothing published; diagnose; a relaunch needs a new go |
| otherwise | not settled | operator decision |

×1.28 falls in row 2, ×1.40 in row 3, ×1.50 in row 4.

A stop signal before `seg2`: if `seg1`'s partial ratio (blocks 0..19 quantized, 20..39 dense)
exceeds ×1.4502, `seg2` is not launched without the operator. The launcher refuses it unless
`FORCE_SEG2=1`. The expectation behind it: quantizing twenty more blocks does not lower the
perplexity. It is *estimated*, not measured.

## Signed prediction

| quantity | point | interval |
|---|---|---|
| R | **×1.28** | [×1.17, ×1.38] |
| `seg1` partial ratio | **×1.13** | [×1.06, ×1.22] |
| `seg1`, billed | **216 min** | [179, 251] |
| `seg2`, billed | **250 min** | [213, 285] |
| `smoke`'s s/block column, both segments | **573 s** | [462, 678] |
| int4 records | **40 of 280** | exactly |
| shard | **1,719,924,170 B** | ± 0.01 % |
| `.llvq` | **3,439,848,370 B** | ± 0.01 % |
| sealed file | **6,563,782,117 B** | ± 0.01 % |
| export `model.safetensors` | **29,536,665,800 B** | ± 0.01 % |
| kernel b/weight, tail f16 (tail f32) | **2.0580** (2.0779) | ± 0.0005 |
| b/param, tail f16, embed q8 (tail f32) | **2.7371** (2.7548) | ± 0.0005 |
| b/param, tail f16, embed f16 | **3.5272** | ± 0.0005 |
| GPU memory, highest 30 s sample | **64 GB** | [60, 80] GB of 96 |
| GPU memory, true peak (not observed: the OOM test) | **77 GB** | [70, 88] GB of 96 |
| sealed f16 perplexity against encoding | **−0.1 %** | within ± 1 % |
| `seal` host `VmHWM` | **77 GB** | [60, 95] GB |

**R.** Three steps, each labelled. The 8B base reads ×1.1983 on Metal. From 8B to 14B,
`Planes14`, both encoded on cards and scored on the same L40S, went from ×1.2201 to ×1.1894
(*measured*, `echelle-4b-8b-2026-08-08.md`): the log-excess shrinks by a factor 0.872. Applied to the 8B base,
that gives ×1.1708 in Metal terms, and ×1.2804 with the card term (*computed*). The low end is the
size step without the card term, ×1.17. The high end is no size step, the full card term and one
draw, ×1.3785. Flaws: the size step is `Planes14`'s, and `Tetra` lost ground to `Planes14`
between 4B and 8B; the 14B `Planes14` was encoded in bf16 (`jobs.csv:27`) and the 8B in f32, so
the size step also carries a dtype change; the card term is one pair, at 4B, on bare `Tetra` and
C4, not on this recipe.

**Partial ratio.** Half of R's log-excess, ×1.13 (*computed*). The interval is R's interval
halved the same way, [×1.08, ×1.17], widened by a quarter of the log-excess on each side for the
spread over blocks, which nothing has measured (*estimated*). Flaw: it assumes the excess adds in
log terms and spreads evenly over the blocks. Neither is measured.

**Durations.** From the card plan and its first check (tasks `wghg5aptb`): 381.7 min of loop for 40
blocks on x1, the x2 loop's 181.7 CPU minutes doubled plus about 1,100 s of capture, transfer and
write (*estimated*). Half is 190.9 min, 573 s a block. The range, [154, 226] min a segment, is the
check's [353, 497]-min job less its 45 fixed minutes. Fixed parts of `seg1`: start 3, oracle 1,
gate 4, download of 29.5 GB, f32 load and baseline 12, verify, partial perplexity and sha256 5.
Of `seg2`: start and oracle 4, download, load and baseline 12, shard copy 2, reload 4, replay of
the first 20 blocks 3, verify and perplexity 7, seal and copy 8, sealed perplexity 3, export 6,
export copy 10. Flaw: no Tetra 14B has run anywhere; the only 14B encode was `leech1c12`, 302
billed min on x2.

**Bytes and rates.** *Computed* from the writer (`format.rs` `put_record_head`, `write_codes`;
`sealed.rs` `write_raw`, `write_blob`) and from `rtbits`' own formula. The same arithmetic
returns, to the byte, the 4B `.llvq` (1,004,826,858 B), the 8B `.llvq` (1,862,836,458 B) and the
8B sealed file (4,364,205,777 B), and all six 8B `rtbits` rates. The export size comes from
the card plan's emulation of `safetensors` 0.7.0 (`export-predict.py`, scratchpad, not in the
repository). It was exact on both 4B exports and on the 8B one, 16,381,516,776 B
(`dclm-8b-rowscales-2026-09-21.txt`). At 14B: 443 tensors, a 51,392-byte header. The 14B carries 541,081,600 Tetra words, 16,384,000 tail weights, 2,048,000
row scales and 209,715,200 int4 weights over 13,212,057,600 projection weights; the whole model is
14,768,307,200 parameters, of which 1,555,824,640 embedding and 424,960 norms.

**GPU memory.** The f32 model is 59.07 GB. A perplexity pass at context 4,096 holds attention
scores of 2.68 GB, three live, and logits of 2.49 GB, two live: 72.1 GB with the model. CUDA
context and workspaces bring about 74 GB, the baseline. The final perplexity of each segment
adds the 64 calibration windows, 2.68 GB: `hidden` is still in scope when `smoke` scores the
quantized model (`smoke.rs:1337`), so the peak is about 77 GB, at the end of the segment
(*computed*). The interval is wide upwards because the allocator may keep freed blocks.

The 30 s sampler cannot see that peak. The scores and logits of a perplexity pass live for
milliseconds to a fraction of a second, and NVML keeps no high-water mark, so a sample lands on
one only by chance. What the sampler sees is what stays resident: the model, the 64 windows and
the CUDA context, about 64 GB, plus whatever the loop's capture or the allocator holds at that
instant (*computed*). That is the signed sampled figure. The 77 GB is tested only by the job
surviving its final perplexity: a death above 96 GB is the miss.

The same model of memory missed by ×2.9 at 8B on Metal: 114.5 GB footprint against 40 GB
predicted, cause not diagnosed (`preregistration-dclm-8b-2026-09-21-ECARTS.md` E1). One
candidate does not carry over to CUDA: candle's Metal pool rounds every buffer up to the next
power of two and keeps freed buffers until a command-buffer flush (candle-core 0.9.2,
`metal_backend/device.rs:126-137, 228-270`); the CUDA backend allocates exact sizes through
cudarc (`cuda_backend/device.rs:53-57`). That is an explanation, not a measurement: no card
encode has recorded its VRAM.

What a death above 96 GB costs depends on where it falls. The baseline is the first test of the
14B footprint, about 20 min into `seg1`, after the gate: about $0.92. The final perplexity comes
after the loop, 2.7 GB higher. A death there costs the segment, up to its $11.69 or $13.06
ceiling. The shard or the `.llvq` is already written and verified by then, since `smoke` verifies
before it scores. `seg1` would then leave no `DONE` marker, and resuming from it needs a new go.

**Seal memory.** 8B: 40.3 GB resident for 282,304,512 decoded blocks, 142.7 B a block
(*measured*, `dclm-8b-2026-09-21-brut/seal.txt`). At 541,081,600 blocks: 77.2 GB (*computed*),
under the card's 256 GB of host memory.

**Cost.** `rtx-pro-6000` at $2.75/h (`hf jobs hardware`, 2026-09-22). `seg1` $9.90 [$8.20,
$11.50], ceiling $11.69 at 255 min. `seg2` $11.46 [$9.76, $13.06], ceiling $13.06 at 285 min. The
pair: $21.36 [$17.97, $24.57], ceiling $24.75 (*estimated*). Queue before each job, 1 h 12 to
3 h 06 on the 8B chain (*measured*); unbilled.

One calibration draw: at the 4B that moves MMLU by 2.92 pp and perplexity by 5.2 %.
