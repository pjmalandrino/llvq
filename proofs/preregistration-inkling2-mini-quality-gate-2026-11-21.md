# Prereg: quality gate of the Tetra code on Inkling 2 Mini, looped core first (2026-11-21)

Written before any measurement. Local Mac only, $0. Operator mandate of 2026-11-21, evening: free hand for the night to
validate quality, the Mac's guard in force.

## Question

Does the lattice code of this repository, at its usual rate, keep Inkling 2 Mini's output as well as the public 2-bit
GGUF ladder does at the same bits, on the same layers?

## Objects

- Model: `thinkingmachines/Inkling-2-Mini` at revision `ed40e7da625f594fc9c4990f5f66be24873c22d7`. It is a looped
  Mixture-of-Experts: 24 layers, 128 experts and 8 routed a token, layers 6 to 17 applied three times. Its experts ship
  in NVFP4 (layers 0 to 19) and MXFP8 (layers 20 to 23). No f16 original exists; the native checkpoint is the reference.
- Rivals: `unsloth/Inkling-2-Mini-GGUF` at revision `ae9327c5fbc7333116ba32eee78bb5d2bbefae4f`, files `UD-IQ1_S`,
  `UD-IQ1_M`, `UD-IQ2_XXS` and `UD-Q2_K_XL`.

## Teacher

The modules of `transformers` at commit `046bdb8cc2`, with the experts dequantized from the native tensors to f32, run
by a layer-major executor (every sequence goes through one layer visit before the next visit starts). Two checks come
before any measurement, and a failed check stops the gate:

1. the executor against the library's own forward on one short sequence;
2. the dequantized expert values against the BF16 GGUF of `ggml-org/Inkling-2-Mini-GGUF`, read by a second code path.

## Calibration

- Corpus: dclm-edu, 64 sequences of 2,048 tokens of the model's own tokenizer, 131,072 tokens.
- If the teacher's forward runs under 25 tokens a second, the set becomes 64 sequences of 1,024 tokens, and the change
  is written as a deviation beside this file.
- One Hessian per (layer, expert) and per matrix input, built from the tokens routed to that expert, each token
  weighted by its squared routing weight. In layers 6 to 17 the sums run over the three passes.
- The inputs come from the teacher, not from a quantized prefix. This departs from the sealed Qwen3 recipe, which
  calibrates each block on the quantized blocks before it.

## Encoder

The quantizer of this repository through its per-matrix path, with the settings of the sealed 4B recipe, unchanged. No
row scales are trained. The exact settings, the bit accounting and the way the fused gate-and-up matrix is split are
written in an annex before the first encode.

## Arms

- Student S(L): the teacher whose experts in the layer set L are the reconstructions of the lattice code. Every other
  tensor is native.
- Rival R_q(L): the teacher whose experts in the same layers are the dequantized tensors of rung q of the GGUF ladder.

Layer sets, in this order: {6, 7, 8}, {6..11}, {6..17}. Then layers 0 to 5 and 18 to 23 if the night allows. The
encode is cumulative and stops where the night stops.

## Measures

- On the raw WikiText-2 test split, 8 windows of 2,048 tokens: mean KL(teacher ‖ arm) per position, top-1 agreement
  with the teacher, and the NLL paired by window. Intervals by bootstrap over windows.
- Bits: expert bits a weight on the layer set. The lattice arm reads the encoder's own accounting. A GGUF rung reads
  the bytes of its expert tensors.
- Census, no criterion: tokens routed to each (layer, visit, expert) cell, and the share of cells under the input
  width of 2,304.

## Criterion, read on the largest loop set the night completes

Let b_T be the lattice arm's expert bits and KL_T its KL.

- **PASS**: KL_T is at or under the ladder's KL interpolated at b_T, linear in log KL against bits between the two
  rungs that bracket b_T. If b_T is under the lowest rung, PASS when KL_T is at or under that rung's KL.
- **FAIL**: a rung with as many bits or fewer has a KL at or under KL_T.
- **NOT SETTLED**: neither.
- Floor: the student's top-1 agreement on the loop set is at least 90 %. Under it the gate reads FAIL whatever the
  ladder reads.

The gate informs. A kill is the operator's.

## Predictions, signed before any measurement

- KL_T on {6..17}: 0.02 to 0.05.
- Ladder on {6..17}: `UD-Q2_K_XL` 0.025 to 0.045, `UD-IQ2_XXS` 0.030 to 0.055, `UD-IQ1_S` 0.040 to 0.070.
- Reading: PASS with probability 0.45, NOT SETTLED 0.35, FAIL 0.20.
- Census: in one visit of a layer, 20 to 50 % of the (expert) cells see fewer than 2,304 tokens; after the sum over
  three passes, under 5 % of the cells of layers 6 to 17 do.

## What this gate does not read

The whole model, the student's GSM8K, any served speed or memory, the gain of row scales, the gain of a sequential
calibration, and the `LookAtMyQuant` file, whose runtime is not run here.
