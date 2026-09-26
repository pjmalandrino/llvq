# llvqtune

Trains the free parameters a quantized LLVQ artifact already holds.

This is not LoRA and not QLoRA. Rows 14 and 17 of `docs/ROADMAP-QUALITY.md`
add no parameter: they change the values of scales and tails the file already
carries, so the written artifact has the same width and the decoder stays
byte-identical. Row 20 is the LoRA-shaped lead and it does cost bits, which is
why `BitCost` is a return type and not a comment.

The paper's own note, transcribed in `docs/llvq-paper-notes.md`: "The
fine-tuning here is no more than learning the per-column scales
(< 0.001 bit/weight, ~52M tokens). It is not end-to-end training."

## Layering

| layer | knows | does not know |
|---|---|---|
| `domain/` | the loop, the rate, the schedule | torch, the `.llvq` format, the network |
| `ports/` | what the domain needs | who supplies it |
| `adapters/` | torch, parquet, the filesystem | which mode is running |
| `trainables/` | one mode each | the loop, the corpus, the journal |

## The three modes

| module | roadmap row | trains | cost |
|---|---|---|---|
| `row_scales.py` | 14 | 1,105,920 row scales | 0 b/param |
| `free_params.py` | 17, Q6a | scales and tails, ~18 M values | 0 b/param |
| `low_rank.py` | 20, Q6b | new factors `A @ B` | +0.131 at r=16, +0.263 at r=32 |

Adding a mode means one module and one line in `trainables/__init__.py`. The
loop, the objectives, the corpus and the journal are untouched.

## The corpus

`--corpus` decides the training text. Every published arm ran on `dclm`, which
stays the default.

| value | text | holds |
|---|---|---|
| `dclm` | DCLM-edu, the corpus the paper calibrates on | ~1.03 G tokens, twenty times any run's need |
| `mmlu-aux` | MMLU-format prompts, `cais/mmlu` `auxiliary_train` | ~24.75 M tokens, 0.79 of a 9,507-step run |
| `mix` | both, interleaved at `--mix-ratio` | the smaller of the two |

`mmlu-aux` exists because the DCLM arm brought perplexity to 1.0074 times f16
and left MMLU nine points under it: the generic-text objective has given what
it can. It reads the split MMLU ships for training, never the three the
harness scores — `dev` supplies the five worked examples of every prompt, and
`fetch_split` refuses all three by name. The overlap was measured before the
adapter was written: 0 items of `test`, `dev` or `validation` share question
and choices with any `auxiliary_train` row (`uv run ops/mmlu_aux_overlap.py`,
2026-09-23). Blocks are written character for character as `bin/mmlu.rs`
`block()` writes them, which one test holds literally.

That corpus is finite, unlike DCLM's shard, so it says how much it holds and
`check()` refuses a plan that outruns it. A run that quietly stops at step
6,300 of 9,507 writes a plausible sigma and a plausible journal.

## The invariant

The gradient never reaches the decoded directions. The Leech decoder is a
table lookup, not a differentiable function, so every mode receives frozen
directions and refuses a tensor that carries a gradient.
`tests/test_trainables.py` checks both halves, and mutating the guard away
fails the suite.

## Why the format is not read here

`llvq-llm --bin export` writes the sealed artifact as an f16 checkpoint, and
its own note records why that substitution is exact: the artifact decodes bit
for bit to those weights. A per-row multiplier on the exported tensor's coded
columns is a per-row multiplier on `row_scales`. So Python trains multipliers
and writes them out as plain data, and folding them back in stays in Rust
where the format lives.

`v_proj` is excluded by default. It is served as int4 g128 and an int4 record
holds no `row_scales`, the same exclusion `rhoapply` makes by construction.

## Running

```bash
uv run --project ops/llvqtune -m llvqtune \
  --student ~/qwen3-4b-export --teacher Qwen/Qwen3-4B \
  --mode row_scales --objective kl --corpus dclm \
  --steps 200 --out /tmp/sigma.json --dry-run
```

`--dry-run` wires everything, prints the rate the run would cost and stops
before the first batch.

```bash
cd ops/llvqtune && uv run --extra torch --with pytest python -m pytest tests -q
```

`--extra torch` brings `transformers` too; without it the tests of `main`
skip rather than fail.

`train.sh` is the card entry point. `EXPORT`, `OUT` and `TEACHER` are
required; `TEACHER` has no default since 2026-09-21, and `main` refuses a
teacher whose `hidden_size`, depth, width or vocabulary differs from the
student's before loading a weight. `STEPS` fixes the step count instead of
`BUDGET / rate`; the six-step probe runs either way and must close with a
rate and a device-memory gauge. `STAGE` copies the export to local disk
first, `MAX_TRAIN_SECONDS` and `MAX_FIRST_KL` refuse a run after the probe.
`CORPUS` and `MIX_RATIO` reach both the probe and the run: a rate measured on
other text prices the wrong run.
On cuda the journal records `max_memory_allocated`, `max_memory_reserved`
and `total_memory` at every checkpoint and in the closing summary.

## The write-back

`llvq-llm/src/bin/rowscale.rs` folds `sigma.json` into a sealed file:
`rowscale <in.bin> <out.bin> <sigma.json>`. Int4 records pass through, a
sigma of all ones writes a byte-identical file.
