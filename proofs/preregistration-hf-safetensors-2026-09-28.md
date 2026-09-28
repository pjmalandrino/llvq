# Prereg. Stage 0 of the transformers plan: the sealed file as safetensors

Status: written and committed on 2026-09-28, BEFORE the conversion is run.
Operator go given 2026-09-28 on stage 0 only. Cost: 0 $, Mac, no paid job.
Plan: `docs/plan-transformers.md`. Code under measurement: the commit that carries this file.

A timestamped prereg is no longer edited. Any departure goes in
`proofs/preregistration-hf-safetensors-2026-09-28-ECARTS.md`, beside it and never into it.

## 1. Question

Can the served 4B be written as safetensors, compressed, with nothing lost?

Nothing but this repository reads a `.llvq` today. `bin/export` bridges to `transformers` by
dequantizing to f16, 8 GB for a 4B, which is an interchange artifact and not a distribution
format. Stage 0 decides whether the compressed file has a faithful safetensors image at all.
Every later stage of the plan rests on the answer.

## 2. The object

`~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin`, 1,418,224,685 bytes, sha256
`886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8`. Format v5, kinds
Tetra+Int4G128. 168 Tetra records, 84 Int4G128 records (`v_proj`, `o_proj`, `down_proj` at
layers 12 to 23), the tied embedding carried int4 g64, the norms f16, `config.json` and
`tokenizer.json` as blobs (*measured*, `rtbits-sealed.txt` and `int4swap.txt` of that
directory).

## 3. The three format decisions, taken before the code

Taken by the operator on 2026-09-28, recorded here because each one is a fact about a
distribution format and not an implementation detail.

| decision | what is written | why not the other |
|---|---|---|
| naming | our own scheme: the artifact's `<prefix>.weight` becomes `<prefix>.<field>`, with `codes`, `row_scales`, `centroids`, `tail`, `qweight`, `scales`, `biases`; rotations under `llvq.rotations.<key>.signs` and `.small` | no in-tree method has Tetra's shape. aqlm and vptq store codes against a global codebook, we store a 47 bit map with no dictionary, and borrowing their names would misdescribe the content |
| code payload | the bytes the file stores: one 48 bit word per block, MSB first, dense, one continuous run per matrix | the served `tetra48` layout would let a CUDA kernel read the file with no transcode, at 2.148 against 2.000 b/weight on the codes, and would marry the distribution file to one kernel layout |
| rotation | the two tables, `signs` (f64, `d_in`) and `small` (f64, `k` by `k`), written once per distinct `(d_in, seed)` and deduplicated, the seed kept beside them as provenance | the seed alone would require a bit-exact port of `SplitMix64`, of the Gaussian draw and of Gram-Schmidt to Python. A last-bit disagreement there changes the weights and breaks nothing visibly |

## 4. Setup

```bash
# the converter, Rust, reading the sealed file through llvq-artifact
cargo run --release -p llvq-llm --bin hfpack -- \
  ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin ~/q4b-hf-2026-09-28

# the gate: an independent Python reader over the written directory
uv run ops/llvq_hf_check.py ~/q4b-hf-2026-09-28
```

`hfpack` writes `model.safetensors`, `config.json` with a `quantization_config` block,
`tokenizer.json` and `tokenizer_config.json`, plus `llvq-digest.json`. The digests in that
file are computed from the fields as `llvq-artifact` reads them out of the `.llvq`, which is
the reference implementation. The Python checker recomputes every one of them from the
safetensors bytes alone, through its own unpacking of the 48 bit words. The two paths share
no code, only the canonical byte convention this prereg names in section 5.

## 5. The canonical digest, defined once

sha256 over little-endian bit patterns, per field, per record:

| field | bytes hashed |
|---|---|
| `codes` | the code stream verbatim, as the record stores it |
| `row_scales`, `centroids`, rotation `signs` and `small` | f64 bit patterns, 8 bytes each |
| `tail` | f32 bit patterns, 4 bytes each |
| `qweight` | the packed nibbles verbatim, low nibble first |
| `scales`, `biases` | IEEE binary16 bit patterns, 2 bytes each |
| raw f16 tensors | IEEE binary16 bit patterns, 2 bytes each |
| blobs | the bytes verbatim |

Scalars are compared as values and not hashed: `d_out`, `d_in`, `shell_cap`, the code kind,
the rotation seed, `bits`, `group`, and every tensor shape.

## 6. Controls

If one of these fails, nothing is published and the stage does not advance.

1. `hfpack` re-reads `model.safetensors` after writing it and compares every tensor against
   the one in hand, as bit patterns, the way `bin/export` already does.
2. The record count matches the header: 252 records, 168 Tetra and 84 Int4G128.
3. Every weight of the sealed file is accounted for: 3,633,315,840 projection weights,
   388,956,160 embedding, 196,096 norm.
4. `config.json` and `tokenizer.json` come out byte for byte identical to the blobs.
5. A Ball record is refused by name. This converter has one map and the served files carry no
   Ball record; guessing a width for one is how a reader returns plausible wrong weights.
6. `cargo clippy --all-targets` is silent and `cargo test` is green in the fast loop.

## 7. Decision rule

| result | reading |
|---|---|
| every digest agrees and every control passes | stage 0 passes. Stage 1 is proposed to the operator with its own prereg |
| a digest disagrees | a byte order or a shape is wrong in the writer or in the Python reader. Defect, fixed, the whole check re-run. Not a tolerance to widen |
| a field of the sealed file has no faithful safetensors dtype | stage 0 fails on its own kill criterion. Written up, the stage stops, the operator decides whether an out-of-band sidecar is acceptable |
| the written directory exceeds 1.50 GB | the layout wastes what the format was built to save. The stage stops and the payload decision of section 3 is reopened |
| otherwise | not settled, operator decision |

## 8. Signed prediction

**No field fails for want of a dtype, and one field escapes the tensors.** safetensors carries
U8, F16, F32 and F64, which covers codes, scales, biases, tails, row scales and centroids. The
rotation seed is a u64 and candle's tensor dtypes stop at I64, so the seed goes in
`config.json` as a decimal string. That is a representation choice and not a loss, since the
tables the seed generates are written in full.

**The directory lands between 1.40 and 1.45 GB.** The payload is the same as the sealed file's,
1.407 GB of codes, side data, int4 records, embedding and norms (*computed* from
`rtbits-sealed.txt`). Added: the rotation tables, at most 5 MB, and a safetensors header of a
few tens of kilobytes. Removed: the sealed framing. `tokenizer.json`, about 11 MB, moves from a
blob to a file and stays in the total.

**The `quantization_config` block lands between 20 and 60 KB.** Five fields per record over 252
records, plus the rotation index.

The known flaw in this reasoning: the number of distinct rotations is not known before the file
is read. If the 4B holds one rotation per projection rather than one per group, the tables cost
36 layers times 7 projections, so up to 1,000 tables rather than about 100, and 30 MB rather
than 3. The prediction interval already absorbs that.

I have been wrong on signed predictions three times in this repository, once by a factor of
ten. This one is scored the same way.

## 9. What this stage cannot establish

- Nothing about tokens. No forward pass runs here, no perplexity, no MMLU. Identity of
  generation against `bin/run` is stage 1's gate.
- Nothing about speed, at any stage of this plan. `transformers` is not a throughput engine and
  no tok/s it prints is ever divided against ours or vLLM's (rule 5).
- Nothing about the 8B or the 14B. Both are the same format and neither is converted here.
- Nothing about the upstream PR of stage 6, which is the maintainers' call.
