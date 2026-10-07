# Prereg. Stage 1 of the transformers plan: a registered quantizer that answers

Status: written and committed on 2026-09-30, BEFORE the first load.
Operator go given 2026-09-30 on stage 1 only. Cost: 0 $, Mac, no paid job.
Plan: `docs/plan-transformers.md`. Stage 0: `proofs/preregistration-hf-safetensors-2026-09-28.md`,
journal `docs/mesures/hf-safetensors-4b-2026-09-28.txt`.

A timestamped prereg is no longer edited. Any departure goes in
`proofs/preregistration-hf-quantizer-2026-09-30-ECARTS.md`, beside it and never into it.

## 1. Question

Does `from_pretrained` on the packed 4B give the weights the artifact defines, and does the model
answer?

Stage 0 proved the directory carries every field. Nothing has read it as a model. What is unknown
is whether the Leech decode, the shape-gain reconstruction and the un-rotation can be rebuilt in
PyTorch to the last bit, and whether a registered quantizer can host them without patching the
model code.

## 2. The objects

Input: `~/q4b-hf-2026-09-28/`, written by `hfpack` at commit `0c4197f`. `model.safetensors` sha256
`a28348cafa82d23d...`, `config.json` sha256 `a64c2bca2e3cbb41...`. 252 records, 168 Tetra and 84
Int4G128, 96 rotations, the tied embedding int4 g64.

Reference for the weights: `llvq_artifact::decode_matrix`, the artifact's own decoder, through a new
`bin/hfdense` that writes one sha256 per record and nothing else.

Reference for the tokens: `bin/run`, the deliverable's own demo, on
`~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin`, sha256 `886391a8c03f66dc...`, the file
`~/q4b-hf-2026-09-28/` was packed from.

Environment: `transformers` 5.17.0, `torch` 2.14.0, resolved by `uv` on 2026-09-30. In this version
`create_quantized_param` no longer exists: a quantizer replaces its modules in
`_process_model_before_weight_loading` and the ordinary loader fills them, which is what
`quantizer_aqlm.py` does and what this follows.

## 3. Three decisions taken before the code

| decision | what is done | why not the other |
|---|---|---|
| the decode tables | the four universal tables of the Tetra map (`prefixes`, `branches`, `suffixes`, `rows`) plus `VALUES` and the trio order ship **inside the Python package**, dumped once from Rust by `bin/tetratables`, about 20 KB. The package refuses to load a file whose `tetra_fingerprint` disagrees | the map is a property of the codebook, not of a model, which is exactly why the format carries a fingerprint and no table (`llvq-artifact/src/lib.rs`). Putting it in every published model would duplicate a universal object; re-deriving Golay, the trio and the trellis in Python would be a second implementation to keep bit-exact |
| the rotation | folded into the dequantization at load, exactly as `decode_matrix` does: decode in the rotated basis, restore the tail, un-rotate, then narrow. No rotation runs in the forward pass | the served kernel keeps its weights rotated and rotates the activation, which is the `rot_apply` wall. Stage 1 does not need that and would not test it honestly |
| the loaded model | each module dequantizes into a dense weight at load and frees its code buffers | a per-call decode of 119 M blocks in PyTorch would make the gate a benchmark of our slowest path. What is compressed is the **file**, 1.42 GB; see §7 |

The consequence of the second decision is stated rather than buried: **the kill criterion the plan
wrote for stage 1 is void.** "The quantizer hook cannot host the per-group rotation without patching
the model code" cannot fire here, because folding the un-rotation into the dequantizer removes the
rotation from the forward pass. The question is real and moves to the stage where a kernel reads
rotated weights, stage 2 on Metal and stage 4 on CUDA. Nothing in this prereg claims it is settled.

The open decision the plan left for this stage, whether `quantization_config` stays pretty printed at
131 KB, is decided: it stays. A side file would be a second contract to hold in step with the first,
and `from_pretrained` reads `config.json` anyway.

## 4. Setup

```bash
# the two Rust references, 0 $, Mac
cargo run --release -p llvq-llm --bin tetratables -- llvq-hf/llvqhf/data/tetra-tables.safetensors
cargo run --release -p llvq-llm --bin hfdense -- ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin \
    ~/q4b-hf-2026-09-28/llvq-dense-digest.json

# gate A, the arithmetic
cd llvq-hf && uv run pytest && uv run python -m llvqhf.checkdense ~/q4b-hf-2026-09-28

# gate B, the model
uv run python -m llvqhf.gentokens ~/q4b-hf-2026-09-28 --dtype f32 --device cpu --new 64
LLVQ_DTYPE=f32 LLVQ_RUN_DUMP=/tmp/run-tokens.json \
    cargo run --release -p llvq-llm --bin run -- ~/q4b-sealed-2026-09-23/qwen3-4b-sealed.bin cpu 64
```

`LLVQ_RUN_DUMP` is added to `bin/run` by this lot: it writes the generated ids as JSON. The tokens
are compared as ids and not as decoded text, because a decoded string can hide a tokenizer
difference and this comparison is about the model.

## 5. The gates

**Gate A, the arithmetic.** For each of the 252 records and for the embedding, the sha256 of the
dequantized f32 weights, little-endian, equals the one `decode_matrix` gives. 253 of 253 or the stage
does not pass. This is the whole claim that a reader outside this repository rebuilds our weights,
and it is entirely inside our control.

**Gate B, the model.** `from_pretrained` loads the directory with no missing key and no unexpected
key, and the four prompts of `bin/run` give **64 identical greedy token ids** each, at f32 on the CPU
on both sides, same tokenizer, no special tokens.

Gate A is what makes a failure of gate B diagnosable: identical weights and different tokens is a
statement about two forward passes, not about this loader.

## 6. Controls

If one fails, nothing is published and the stage does not advance.

1. The package refuses a file whose `tetra_fingerprint` is not the one its tables were dumped under.
2. The decode is checked on the origin word and on the 47-bit label space by sampling: 100,000 words
   drawn from a fixed seed, decoded in Rust and in Python, compared as integers.
3. One mutant per gate, at least: a swapped section in the decode, a transposed rotation, a wrong
   gain bit. Each must be caught, and it is named in the journal.
4. `cargo clippy --all-targets` silent, `cargo test` green in the fast loop, `pytest` green.
5. The stage 0 gate still passes on the same directory, unchanged by this lot.

## 7. Signed predictions

**Gate A passes exactly, 253 of 253.** Every step of the chain is reproducible bit for bit in f64:
the decode is integer arithmetic and table lookups, the reconstruction is one multiply per
coordinate, the Walsh-Hadamard transform is a butterfly of single adds and subtractions in a fixed
pairing order, and the `k` by `k` mix is an accumulation whose order I will reproduce term by term
rather than hand to a matrix product. The prediction is exactness, not closeness. If it fails it will
fail on the mix, which is the one place a library could reassociate a sum.

**Gate B passes on all four prompts.** The weights are identical and greedy decoding has a wide
margin. This is the weaker of the two predictions: the two forward passes are different code, and
inside our own engine the 14B kernel and dense paths diverge at token 78 on identical weights. At the
4B, 256 tokens agreed.

**The dequantization of the whole model lands between 30 s and 5 minutes** on this Mac, for 119 M
blocks through numpy. Above 10 minutes the loader is unusable for the later stages, which is a fact
to report rather than a failure of a gate.

**The loaded model occupies 8 GB at f16 and 16 GB at f32**, because it is dense once loaded. No
memory claim is made at this stage.

I have been wrong on signed predictions four times in this repository, once by a factor of ten, and
the last one on 2026-09-28 by a factor of two. This one is scored the same way.

## 8. What this stage cannot establish

- Nothing about memory at inference. The model is dense once loaded, by decision §3.
- Nothing about speed, here or at any stage of this plan (rule 5).
- Nothing about the rotation under a kernel, which is stages 2 and 4.
- Nothing about the 8B and the 14B, same format, not loaded here.
- Nothing about quality. No MMLU, no perplexity, no GSM8K. Token identity against `bin/run` is an
  identity check and not a score.
- Nothing about `save_pretrained`. Writing a model back out is not in this stage.
