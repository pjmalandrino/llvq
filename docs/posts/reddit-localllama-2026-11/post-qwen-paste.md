TITLE: Qwen3 4B / 8B / 14B squeezed to ~2.7 bits per weight: the 14B fits in about 5 GB and keeps 75.7 MMLU. Qwen 4 27B is next

>>> IMAGE ICI : 01_mmlu_vs_bits.png <<<

hi r/Qwen,

I've been quantizing Qwen3 models with a method from a recent Qualcomm paper, Leech lattice quantization ([arXiv:2603.11021](https://arxiv.org/abs/2603.11021)), in my own Rust implementation with my own CUDA kernel. Three sizes are done and public: Qwen3-4B, 8B and 14B, all around 2.7 bits per parameter for the whole model.

Short version: at this size, Qwen3 holds up better than I expected. Here's what it looks like, the good and the bad.

## How much of Qwen3 survives

| model | weights on the GPU | MMLU (FP16) | GSM8K (FP16) |
|---|---|---|---|
| Qwen3-4B | 1.38 GB | 63.4 (70.1) | 82.5 (92.1) |
| Qwen3-8B | 2.76 GB | 69.6 (75.1) | 88.6 (93.3) |
| Qwen3-14B | 5.04 GB | 75.7 (78.9) | 92.0 (95.3) |

MMLU is 5-shot on all 14,042 questions, GSM8K is all 1,319 problems, zero-shot, greedy. FP16 in brackets, run through vLLM on the same card.

>>> IMAGE ICI : 04_gap_to_fp16.png <<<

The loss shrinks as the model grows. At 4B it hurts (about 7 MMLU points, 10 on GSM8K). At 14B it's around 3 points on both. Bigger Qwens seem to take low bits better, which is part of why I want to try a bigger one next.

## Against other quants of the same Qwen3

At the same size, I compared against llama.cpp's IQ2_S (calibrated by me on the same data) and Unsloth's UD-IQ2_XXS. Same day, same L40S.

>>> IMAGE ICI : 02_same_bits_quality.png <<<

We come out ahead on MMLU at all three sizes (+9.1, +4.1, +2.9 points over IQ2_S), and on GSM8K at 4B and 8B. At 14B GSM8K it's a tie.

The more useful view for picking a model: at about the same memory, our 8B beats 4-bit AWQ's 4B on MMLU, and our 14B beats AWQ's 8B while using 1 GB less. On GSM8K both are ties.

>>> IMAGE ICI : 03_size_up.png <<<

## What's not great

- Speed: llama.cpp decodes faster, by 34 to 64 tok/s depending on size. We do 227 / 145 / 90 tok/s at batch 1 on an L40S. No consumer GPU measured yet.
- Long prompts: during an 8K-token prompt our memory peak is higher than llama.cpp's (5.95 GB against 3.52 at 4B), and the 4B takes 35 s to its first token.
- It's a custom format, so no llama.cpp, LM Studio or Ollama files. You need my runtime (it speaks the Ollama and OpenAI APIs), or `transformers` for the 4B.
- Greedy decoding only in the fast path, for now. Qwen recommends sampling, so keep that in mind.
- One calibration run per model. On the 4B, a different calibration sample moves MMLU by about 3 points.

## What's next

**Qwen 4 27B.** Not encoded yet, so no numbers. Back of the envelope: at ~2.7 b/param, that's roughly 9 GB of weights, which would bring a 27B within reach of a 12 GB card if the quality holds. That's the bet.

**Long prompts.** The work in progress right now: cutting the memory peak during long prompts (the prefill's attention first, then an int4 KV cache that would take the 4B's cache at 8K from 1.21 to about 0.34 GB), and a faster prefill that brings the 4B's first token at 8K from 37 s to 6.5 s on a bench. Neither is in the runtime yet.

## Try it

Files: [Qwen3-4B](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed), [Qwen3-8B](https://huggingface.co/Pier-Jean/Qwen3-8B-LLVQ-Tetra-sealed), [Qwen3-14B](https://huggingface.co/Pier-Jean/Qwen3-14B-LLVQ-Tetra-sealed)

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
cargo build --release -p llvq-runtime --features cuda    # --features metal on a Mac
target/release/llvq pull qwen3:4b
target/release/llvq run qwen3:4b "Name three primary colours."
```

4B in `transformers`: `pip install llvq-tetra`, then `from_pretrained("Pier-Jean/Qwen3-4B-LLVQ-Tetra")`. Import `llvq_tetra` first, otherwise transformers silently loads a random model.

Code, paper and every raw measurement: [github.com/pjmalandrino/llvq](https://github.com/pjmalandrino/llvq). Longer write-up with more charts on r/LocalLLaMA: [link]

If you run Qwen locally on a 3090, 4090 or 5090 and want to try it, I'd like to hear your speed. And if there's a Qwen size you'd rather see before the 27B, say so.
