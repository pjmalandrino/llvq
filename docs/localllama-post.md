<!--
Post r/LocalLLaMA, version finale apres panel de relecture (2026-08-29).
Remplacer HF_POST_URL par l'URL reelle du billet HF avant de poster.
NE PAS utiliser le lien docling-studio, c'est un autre article.
-->

TITLE:
Qwen3-4B at 2 bits: 2.60 GB VRAM, 87 tok/s, and it costs 14.7 MMLU points (Rust, from scratch)

BODY:

I got nerd sniped by a Qualcomm AI Research paper that quantizes LLM
weights on the Leech lattice (a 24 dimensional sphere packing, of all
things) and reports the best quality anyone has published at 2 bits per
weight. I'm not a researcher, I write software at a company in France. I
wanted to know if it holds up outside the paper, so I implemented it from
scratch in Rust. Took about a month.

Numbers on Qwen3-4B, measured on a rented L40S:

- 2.60 GB on the card vs 8.04 GB in fp16
- 1.41 GB on disk
- 87 tokens/s vs 43.5 on my own fp16 path (same code base, so not a vLLM comparison)
- same greedy tokens as the dense model up to one tie break at token 89

The bad news: the official 4-bit AWQ of this model loses 0.3 MMLU points.
Mine loses 14.7 (70.3 -> 55.6), and WikiText-2 perplexity goes
12.24 -> 16.94, about 38% worse. If 4-bit fits in your VRAM, run 4-bit.

The gap does shrink with size though: 14.7 points at 4B, 10.6 at 8B, 6.9
at 14B. That's three data points. I'm not calling it a scaling law.

Before anyone asks how to run it in llama.cpp: you can't yet. It's a
custom format, not GGUF/AWQ/safetensors, and the memory win needs
Linux + NVIDIA.

Repo: https://github.com/pjmalandrino/llvq
Write-up: HF_POST_URL
Paper: https://zenodo.org/records/22133607 (self deposited, not peer reviewed)

Is this worth porting somewhere people can actually use it? vLLM plugin
vs llama.cpp, which would you actually want?
