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
  --mode row_scales --objective kl --steps 200 --out /tmp/sigma.json --dry-run
```

`--dry-run` wires everything, prints the rate the run would cost and stops
before the first batch.

```bash
cd ops/llvqtune && uv run --with pytest --with torch python -m pytest tests -q
```

## Not done here

The write-back. Nothing folds `sigma.json` into a `.llvq` yet.
`llvq-bench/examples/rhoapply.rs` already multiplies `row_scales` by one
scalar and its idempotence control at rho = 1 is byte-identical on the served
mixed file. Generalizing it from a scalar to a per-row vector is the missing
piece, and it is the only Rust work this module needs.
