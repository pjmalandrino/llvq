---
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Mini
base_model_relation: quantized
language:
  - en
library_name: llvq
pipeline_tag: text-generation
inference: false
tags:
  - inkling
  - inkling-2
  - inkling-2-mini
  - thinking-machines
  - mixture-of-experts
  - moe
  - quantization
  - quantized
  - 2-bit
  - low-bit
  - tool-calling
  - openai-compatible
  - ollama-compatible
  - rust
  - cuda
  - leech-lattice
  - vector-quantization
  - llvq
  - tetra
---

# Inkling 2 Mini, 2-bit quantized: a 26.4B mixture of experts in 8 GB of VRAM (LLVQ Tetra)

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), 26.4 billion parameters, with its 3,072
experts stored as Leech-lattice codes, 48 bits for a block of 24 weights. The weights stay compressed on the card:
one CUDA kernel decodes and multiplies them, and nothing is ever written out in full.

A 2-bit quantization of Thinking Machines' Inkling 2 Mini that loads in 8.4 GB of card memory, decodes at 72 tokens
per second on an L40S and 37 on an L4, and is served by one command with an OpenAI-compatible API and tool calls.

**This is a preview.** It is text only: the vision tower is not in this repository. The files are the ones our
engine reads today, not the final sealed format. Every figure below is ours, measured on the date given, on one
card, batch 1. Nothing was trained: the codes come from a Hessian-weighted search over calibration text.

## Run it

Linux with an NVIDIA card of the RTX 40 generation or newer, driver 550 or newer. Nothing to compile and no CUDA
toolkit to install:

```bash
curl -fsSL https://github.com/pjmalandrino/llvq-tetra/releases/latest/download/install.sh | sh
llvq run inkling2:mini
```

Tested on 2026-11-24 as a stranger would get it: on an NVIDIA L4, in a machine with no CUDA toolkit and no
compiler, the install command, an anonymous pull and `llvq run` answered, and the server passed its 13 checks, tool
calls included.

Other models of the same method, Qwen3 4B, 8B and 14B, are in the
[LLVQ Tetra - Qwen](https://huggingface.co/collections/Pier-Jean/llvq-tetra-qwen) collection.

`llvq serve` exposes the same model on an OpenAI-compatible route and an Ollama-compatible one, with tool calls.
The model writes a reasoning block before it answers, including at the lowest reasoning effort, so give it room:
an answer capped at a few dozen tokens can be all reasoning.

## Size

| | bytes | bits per parameter |
|---|---|---|
| this repository, what `llvq pull` downloads | 10.00 GB | 3.03, *computed* |
| the same model in its sealed form | 8.14 GB | 2.47, *computed* |
| the native model, text and vision | 19.53 GB | 5.92, *computed* |

The difference is a 16-bit copy of the input and output tables, 1.85 GB, that `model.safetensors` still carries
beside the 4-bit tables the engine reads. The sealed form drops it; the engine does not read the sealed form yet.
Bits per parameter count the whole text model, tables included.

## Memory and speed

*Measured* 2026-11-23 and 2026-11-24, one card, batch 1, the fused stack.

| | NVIDIA L40S | NVIDIA L4 |
|---|---|---|
| decode after a 32-token prompt | 72.2 tok/s | 36.9 tok/s |
| decode after a 9,000-token prompt | 62.1 tok/s | 31.2 tok/s |
| card memory after load | 8.39 GB | 8.13 GB |
| peak card memory reading a 9,000-token prompt | 12.0 GB | 11.5 GB |
| time to read a 9,000-token prompt | 23.9 s | 77.1 s |
| first token after a 32-token prompt | 0.76 s | not measured |

On an Apple M-series Mac through Metal: 27.7 tok/s after a short prompt, *measured* 2026-11-23.

A follow-up turn of a conversation reuses what the cache holds: 20 new tokens after 8,791 cached ones are read in
0.36 s on the L40S. The first read of a long prompt is the slow part, and it is what we are working on.

The numbers of the two cards are not to be divided by one another, and neither is comparable to a figure another
engine reports on another card.

## Quality

*Measured* in our own harness.

| | result |
|---|---|
| KL divergence against the native model, WikiText-2 test, 8 windows of 2,048 tokens | 0.0880 nats |
| GSM8K, first 200 problems, dense reference path | 191 of 200 |
| GSM8K, same 200 problems, through the fused kernel stack | 186 of 200 |

The two GSM8K rows are not separated: paired, p = 0.18. We have not run the native model through the same GSM8K
harness, so these counts say nothing yet about the loss against it. The KL does.

The fused stack is not bit for bit with the dense path: it reads the output table in 4 bits and a long prompt in
pieces of 512 tokens. Each was scored on perplexity against the path it replaces: +0.413 % for the 4-bit table,
and -0.042 % [-0.127, +0.044] for the pieces, on texts up to 4,096 tokens. No text longer than that was scored.

## What is not done

- Nothing here was run on a 16 GB consumer card. The peak figures above say it should fit; no run says it does.
- The vision tower and the multi-token-prediction head are absent.
- No comparison against other 2-bit files of this model was run through this engine.

## Files

`llvq-files.json` is the index the engine reads: it names every file below with its size and SHA-256.

- `config.json`, `model.safetensors`, `tokenizer.json`: the text model outside the experts.
- `tables/`: the input and output tables in 4 bits, groups of 64.
- `layers/L00.tar` to `layers/L23.tar`: the lattice codes of the 128 experts of each layer.

## Licence

Apache 2.0, as the native model. Use of these weights is also bound by Thinking Machines'
[Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/), which covers any modified
version of the model: read it before you use or serve this file.

The code that made and serves the file is at [pjmalandrino/llvq-tetra](https://github.com/pjmalandrino/llvq-tetra). The method
is LLVQ, vector quantization on the Leech lattice ([arXiv:2603.11021](https://arxiv.org/abs/2603.11021)). The
Tetra format and its fused kernel are described in *Tetra: Serving Leech-Lattice Quantized LLMs at 2.7 Bits per
Parameter* ([arXiv:2609.35465](https://arxiv.org/abs/2609.35465)). That paper measures Qwen3; the mixture-of-experts
support and every Inkling figure on this card came after it.
