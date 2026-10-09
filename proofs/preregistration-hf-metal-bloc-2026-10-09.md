# Prereg. The Mac block: no toolchain, the 4-bit vocabulary, the rotation kernel, many rows a dispatch

Status: written and committed on 2026-10-09, BEFORE the first gate run of this block.
Operator go given 2026-10-09 on the Mac block. Cost: 0 $, Mac M3 Max, no paid job.
Plan: `docs/plan-transformers.md`. Earlier gates on this path: M1 and M2 (`proofs/preregistration-hf-metal-m1-2026-09-30.md`,
`proofs/preregistration-hf-metal-m2-2026-09-30.md`), and the in-tree check of 2026-10-07 (256 ids of 256 on MPS,
2.750 GB allocated).

A timestamped prereg is no longer edited. Any departure goes in
`proofs/preregistration-hf-metal-bloc-2026-10-09-ECARTS.md`, beside it and never into it.

## 1. Question

Can the compressed Metal path of `llvq-tetra` run with no compiler on the user's machine, keep the vocabulary in 4 bits,
rotate with a kernel, and take many rows a dispatch, while giving the same 256 tokens as `bin/run`? And how fast is it
then, in `transformers`, against what it is today?

Today the path needs ninja and the Xcode command line tools, holds the vocabulary dense in f32 (1.56 GB of the 2.750),
rotates in plain torch ops, and dispatches once per token. No speed was ever measured on it.

## 2. What was done before writing this

One feasibility probe, which is not a gate: the served `tv_tetra48_metal`, compiled by `torch.mps.compile_shader` with
its threadgroup buffer moved from the arguments into the body, gave the same f32 as the current Objective-C++ dispatch,
bit for bit, on 9 records of the 4B in four shapes. It decided that arm A is possible. It is not one of the gates below.

## 3. The arms

| arm | change |
|---|---|
| A | every Metal kernel through `torch.mps.compile_shader`: no `cpp_extension`, no ninja, no Xcode |
| B | the vocabulary stays 4-bit on the device: `emb_q4_gather_metal` for the input, an f32-input variant of `tv_emb_q4_metal` for the tied output layer |
| C | the rotation runs as a kernel, an f32-input variant of `rot_apply_rows_metal`, once per group of projections that share it |
| D | a dispatch takes several activation rows: a rows variant of the Tetra matvec and of the tiled 4-bit matvec, each block decoded once and applied to every row |

The served shaders stay byte for byte the copies the package ships. Each variant is made by a textual change to a
served kernel that a test names and checks, never by a second hand-written copy of its arithmetic.

## 4. The objects

- The model: `Pier-Jean/Qwen3-4B-LLVQ-Tetra`, the files TestPyPI and PyPI were checked against on 2026-10-07.
- The token reference: `bin/run`, f32, CPU, 64 new tokens on its four prompts, the dump of 2026-10-07 in
  `docs/mesures/hf-testpypi-4b-2026-10-07-brut/run-tokens-f32.json`.
- `transformers`: the in-tree branch `add-llvq-quantization` of 2026-10-07, which calls the package.

## 5. Gates

| gate | criterion |
|---|---|
| G-A | for each of the 252 records, one seeded random activation: the `compile_shader` output equals the current Objective-C++ output, bit for bit |
| G-B1 | the gathered vocabulary rows equal the rows the dense path dequantizes, bit for bit, on every row of the table |
| G-B2 | the output layer's logits against an f32 dense matmul of the same table: relative error under 1e-4 on every row of 64 seeded activations |
| G-C | the kernel rotation against the torch rotation it replaces: relative error under 1e-5, every rotation of the 4B, 8 seeded rows each |
| G-D | for R rows in one dispatch, each output row equals the single-row dispatch of that row, bit for bit, on every record |
| G-T | the 4B through `transformers`, `device_map="mps"`: no missing or unexpected key, 256 ids of 256 equal to the reference, and the device memory measured |

Each gate gets one mutant before it is called green: a change that should fail it, applied, counted, and caught.

## 6. The speed measurement

Arms, all in `transformers` on this Mac, same process conditions, interleaved round by round:

| arm | what |
|---|---|
| v0 | the path as of 2026-10-07: `llvq-tetra` 0.1.0 code, one dispatch a token, torch rotation, dense vocabulary |
| v1 | this block |
| fp16 | `Qwen/Qwen3-4B` in f16 on MPS, the dense reference of the same stack on the same card |

Two numbers per arm: decode, in tokens per second over 128 greedy tokens after a 16-token prompt; and prefill, the time
to the first token of a 1,024-token prompt. Five rounds, the median, and the range formed round by round (rule 7). v0
and v1 run the same 4B file; fp16 is reported beside them and divided only within this one stack and card (rule 5).

## 7. Decision rule

| result | verdict |
|---|---|
| a gate fails | that arm stops; the others are measured without it, and the failure is written down |
| every gate green | v1 is the new path of the package, to be released as 0.2.0 on the operator's go |
| speed, whatever it reads | published as measured; no speed result kills an arm, the operator decides |
| otherwise | not settled, operator decision |

## 8. Signed predictions

- G-A: green, from the probe. The flaw: the probe saw 9 records and no int4 one.
- G-T memory: 1.30 to 1.50 GB allocated, from 2.750 minus the 1.556 GB f32 table plus its 0.219 GB 4-bit form, 1.413
  *computed*.
- Decode: v1 at least 1.5 times v0, because the torch rotation costs 0.15 ms of a 0.25 ms matvec (M1) and q, k, v
  and gate, up each rotate the same vector today. The flaw: per-token Python overhead may dominate both arms.
- Prefill at 1,024 tokens: v1 at least 4 times faster than v0, from four rows a dispatch. The flaw: the rows kernel
  holds four activation tiles in threadgroup memory, which may cost occupancy.
- fp16 decodes faster than v1. The served Tetra matvec reads 49 GB/s on this card against 146 for the hardware (the
  shader's own measurement), and MPS's dense f16 matmul is tuned for it.
