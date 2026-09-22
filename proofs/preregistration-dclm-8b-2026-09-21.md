# Preregistration. The paper-2 recipe at Qwen3-8B, step 1: the encoding (2026-09-21)

**Written on 2026-09-21 and TIMESTAMPED (`ots stamp`) BEFORE the oracle and the first block.**
The `.ots` attests these bytes. The commit that carries both follows on the operator's go.
Operator go, 2026-09-21, verbatim: "Mouai on va garder sur mac. On va ce faire le 8B déjà pour
voir ce que ça donne, et on verra après pour le 14B". A card encoding had been costed at about
$10 and was declined.

Not edited again. A fact it gets wrong goes beside it, in a `-ECARTS.md`.

**Scope: this step only**, producing the 8B base file and checking it. **Cost $0, hard cap $0**:
no HF job runs under this prereg. Mac time about 3 h 35 for the encoding segment (*estimated*,
below), plus the oracle, the seal and the sealed perplexity. No timeout: past 4 h 45 the operator
is told, and the run is not stopped without his go. The paid chain that follows (about $14
central, $30 worst, *estimated*) has no cap yet, and each of its jobs needs its own prereg.
Project total before and after this step: **at least $176.80** (*computed*: $173.61 over the 171
of 176 rows of `docs/data/jobs.csv` that carry an amount, plus $3.19 per `hf jobs inspect` for
five jobs the registry does not hold, `6aae7666`, `6aaf9e00`, `6aafac81`, `6aafc951`, `6ab087f2`;
five rows of 2026-08-05/06 carry no amount). Measured code: `5d36d52`.

## Question

Does the recipe of the paper-2 object (`Tetra`, `v_proj` in int4 g128, calibration on `dclm-edu`)
give at the 8B a base file whose encoding perplexity stays within one draw of the bare 8B `Tetra`?

No 8B file carries `v_proj` in int4 or was calibrated on `dclm-edu`. This stage carries a
measurement, not a gate on the recipe: at the 4B the base perplexity did not predict MMLU, and
the row-scale training moved it from ×1.328 to ×1.007 of f16 (*measured*,
`dclm-rowscales-2026-09-20.txt`). What it informs is whether the paid chain is proposed.

## Setup

Saved as `$HOME/q8b-dclm-2026-09-21/run.sh`, run with `bash`, launched detached. Any non-zero
exit stops the chain.

```bash
#!/usr/bin/env bash
set -euo pipefail
REPO=$HOME/Documents/Pro/workspace/poc/llvq; OUT=$HOME/q8b-dclm-2026-09-21
caffeinate -i -w $$ &                      # pmset sleep = 1 min: no idle sleep while this runs
if env | grep -q '^LLVQ_'; then echo 'refused: LLVQ_* inherited' >&2; exit 1; fi
cd "$REPO"; git diff --quiet 5d36d52 -- '*.rs' '*.toml' Cargo.lock '*.metal'
cargo build --release -p llvq-bench --bin rtbits          # the one binary dated before 5d36d52
mkdir -p "$OUT/bin"; cp target/release/{oracle,smoke,seal,ppl,rtbits} "$OUT/bin/"
BIN=$OUT/bin; shasum -a 256 "$BIN"/* | tee "$OUT/bin.sha256"   # the checkout is free from here
sysctl vm.swapusage | tee "$OUT/swap.txt"

nice -n 10 "$BIN/oracle" Qwen/Qwen3-8B 64 metal 2>&1 | tee "$OUT/oracle-metal.txt"
grep -q MATCH "$OUT/oracle-metal.txt"

export LLVQ_MODEL=Qwen/Qwen3-8B LLVQ_CALIB=dclm-edu LLVQ_INT4_TYPES=v_proj
export LLVQ_ARTIFACT=$HOME/q8b-dclm-2026-09-21.llvq LLVQ_THREADS=12
env | grep '^LLVQ_' | sort | tee "$OUT/env.txt"          # exactly these five
/usr/bin/time -l nice -n 10 "$BIN/smoke" 64 2048 12 4096 metal nogs tetra1 999 rot 2>&1 | tee "$OUT/smoke.txt"
sysctl vm.swapusage | tee -a "$OUT/swap.txt"

/usr/bin/time -l nice -n 10 "$BIN/seal" "$LLVQ_ARTIFACT" "$HOME/qwen3-8b-dclm.bin" 2>&1 | tee "$OUT/seal.txt"
shasum -a 256 "$LLVQ_ARTIFACT" "$HOME/qwen3-8b-dclm.bin" | tee "$OUT/files.sha256"
LLVQ_DTYPE=f16 nice -n 10 "$BIN/ppl" 4096 12 metal "$HOME/qwen3-8b-dclm.bin" 2>&1 | tee "$OUT/ppl-sealed-f16.txt"
"$BIN/rtbits" "$HOME/qwen3-8b-dclm.bin" 2>&1 | tee "$OUT/rtbits.txt"
```

The 4B recipe (`preregistration-dclm-4b-2026-09-18.md` §1 for `LLVQ_INT4_TYPES=v_proj`, the
`.state` of `~/q4b-dclm-2026-09-18.llvq` for the rest: `tetra1`, `dclm-edu`, 64 × 2048 from the
prefix, `0x110feed`, `nogs`, 1e-2, f32, Metal). Unset and at default: `h_shrink` 1, `gain_scale`
1, `LLVQ_SEQ_BLOCK`, which only `env.txt` records. The model changes, and so does the code: five
commits since that run touch the encode path, `4b40019` (row C, bit-identical at its default) and
four on `model.rs`, which the oracle checks. Against the bare 8B `Tetra` of 2026-09-06, the corpus
(C4 by its 4B twin of the same day, not recorded in its journal), the int4 and the code all
differ, and nothing here separates them. The Mac stays in use: the duration is wall time on a
shared machine, and if `swap.txt` shows swapping the operator is told (the 4B went from 178 to
393 s a block under swap, `dclm-v4-2026-09-18.txt`).

## Controls

If one of 1 to 6 fails, no number from this step is published and no paid job is proposed. 7 and 8
are records: a missing one is reported and the rest stands.

1. `oracle`, Metal, f32: `MATCH`, relative max |Δhidden| under 1e-4 (`oracle.rs:82`), value
   recorded. It read 0.000e0 on 2026-09-06; the forward has changed since (`85a7ec9`).
2. Header kinds `{Tetra, Int4G128}`; **36 int4 records out of 252**.
3. `verify_artifact` inside `smoke` reads back all **6,945,767,424** weights bit for bit.
4. Zero points outside Λ₂₄: the guard of `Tetra::encode` refuses at write time.
5. The sealed file opens; both sha256 in `files.sha256`.
6. `ppl` of the sealed file, f16 on Metal, same 12 windows: finite, within 1 % of the encoding
   perplexity. Encoding f32 against f16 moved −0.18 % to +0.06 % on seven precedents (*measured*,
   F5's three seeds, `leech0c13`, `dclm-2026-09-07`, the DCLM 4B base, the bare 8B).
7. `rtbits`: the `b/param WHOLE MODEL` row `f16 (served)`, column `embed q8`; both kernel lines.
8. `/usr/bin/time -l`: `peak memory footprint` and `maximum resident set size`, `smoke` and `seal`.

## Retention

`$OUT` goes to `docs/mesures/dclm-8b-2026-09-21-brut/`; the journal is
`docs/mesures/dclm-8b-2026-09-21.txt`; both are committed on the operator's go. The `.llvq`, its
`.state` and the sealed file stay in `$HOME`, named by sha256. Upload belongs to the first paid
prereg.

## What gets published, and what does not get compared

Published: R, the sealed f16 perplexity, the duration and phase profile, sizes, `rtbits`, memory.
Not compared: `Planes14` 8B as a format verdict (card-encoded, before 2026-08-26, another corpus;
the repository no longer reproduces its own 4B since then, most likely `4a3e5f0`, not proved);
the bare 8B as a corpus or int4 effect; any MMLU; the training; the kernel.

## Decision rule

`R` is quantized over baseline perplexity, both from `smoke`'s `exact-ppl` line
(`smoke.rs:1402`), same process, f32, to four decimals; the `degradation ×` line rounds to three
and is not read at a boundary. The bare 8B read ×1.2287 (*measured*, `tetra-8b-2026-09-06.txt:95`).
The 4B's calibration σ on perplexity is 5.2 % (*measured*, three seeds of `leech1c12` on C4,
`f5-graines-4b-2026-08-19.txt`; never measured on `Tetra`, at 8B or across corpora). One draw
above the bare figure is ×1.293, two ×1.360, one below ×1.168 (*computed*). No row kills: the
kill stays with the operator, on the trained arm's MMLU.

| result | reading | what follows |
|---|---|---|
| controls 1-6 pass, R < 1.168 | better than the bare 8B by more than one draw | check before reading (METHODE §7), then report; operator decides |
| controls 1-6 pass, 1.168 ≤ R ≤ 1.293 | within one draw of the bare 8B | report; propose the paid chain with its cost; operator sets a cap or declines |
| controls 1-6 pass, 1.293 < R ≤ 1.360 | worse than one draw, within two | report; operator decides whether the census is worth paying |
| controls 1-6 pass, R > 1.360 | worse than two draws | report with the phase profile and a proposed diagnosis; operator chooses |
| a control of 1-6 fails, R not finite, or the run dies | no usable file | nothing published; diagnose; a relaunch needs a new go |
| otherwise | not settled | operator decision |

×1.235 falls in row 2, ×1.31 in row 3, ×1.40 in row 4.

## Signed prediction

| quantity | point | interval |
|---|---|---|
| R | **×1.235** | [×1.19, ×1.29] |
| `smoke`'s `this segment ran` | **3 h 35** | [3 h 00, 4 h 45] |
| int4 records | **36** | exactly |
| sealed size | **4,364,205,777 B** | ± 0.01 % |
| kernel b/weight, tail f16 (tail f32) | **2.0944** (2.1393) | ± 0.0005 |
| b/param, tail f16, q8 (tail f32) | **3.0683** (3.1064) | ± 0.0005 |
| `smoke` peak memory footprint | **40 GB** | [33, 46] GB |
| `seal` peak memory footprint | **33 GB** | [28, 42] GB |
| sealed f16 perplexity against encoding | **−0.1 %** | within ± 1 % |

**R.** ×1.2287 times the 4B composite of the same two changes, 1.0055 (*computed*): the DCLM base
×1.3276, 16.2415 (`dclm-4b-2026-09-18.txt`) over 12.2336 (Metal f32, `~/tetra-q5-2026-09-09/
smoke.txt:82`, not committed), against the bare 4B's ×1.3203 (card f16,
`tetra-q5-encodage-2026-09-09.txt:49`). `v_proj` in int4 cost +1 % there and the corpus gave back
0.46 % (*measured*). Flaw: the composite mixes Metal f32 and card f16 (≤ 0.2 % apart); at 8B the
int4 allocation already paid about 27 % less per bit (`vod-8b-2026-09-18.txt`, repriced per type).

**Duration.** 6,408.6 s (4B DCLM, *measured*) × 1.908 (8B/4B bare, 16,830 / 8,820 s, *computed*)
= 12,228 s; 7,100 s (4B Q5 at 12 threads, *measured*) × 1.908 = 13,547 s. Point between.
Flaw: shared machine; neither precedent recorded its thread count.

**Size, kernel, b/param.** *Computed* from the writer and `rtbits`' own formula
(`format.rs:629-767, 1301-1379`, `rtbits.rs:377-381, 487-493`). One 8B `v_proj` is 1,118,268 B as
`Tetra` and 2,228,292 B as int4: +39,960,864 B on the bare 8B's 4,324,244,913 B. Kernel: 282,304,512
words × 48 + 19,464,192 tail × 16 (32) + 1,363,968 rows × 32 + 150,994,944 × 4.25 over
6,945,767,424. b/param: 3.0672 + 0.0392 for the int4 − 0.0380 for the tail at f16. The same
arithmetic returns the 4B's 24,035,616 B, 2.2030 / 2.1309 and 2.8126 / 2.7475 exactly.

**Memory.** f32 model 32.8 GB plus the larger transient: the ctx-4096 evaluation, f32 logits and
log-softmax, 3 × 2.49 GB, plus attention scores of 2.15 GB (`model.rs:1433, 1650`). `seal` holds
282,304,512 decoded blocks × 100 B = 28.2 GB (`seal.rs:71-82`). Flaw: never logged at this size.

One calibration draw: at the 4B that moves MMLU by 2.92 pp and perplexity by 5.2 %.
