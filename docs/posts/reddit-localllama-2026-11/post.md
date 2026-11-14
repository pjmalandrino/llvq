**Title:** Qwen3 4B / 8B / 14B at ~2.7 bits per weight with Leech lattice quantization. Beats IQ2_S on quality at the same size, loses on speed [open source, Rust + CUDA]

---

[image: fig/01_mmlu_vs_bits.png]

hi r/LocalLLaMA,

I've been working for a while on an independent Rust implementation of LLVQ, the
Leech lattice quantization paper from Qualcomm ([arXiv:2603.11021](https://arxiv.org/abs/2603.11021)), with my own
CUDA kernel to actually run the result. Three Qwen3 models are done (4B, 8B, 14B), all around 2.7 bits per parameter,
and the files are on the Hub.

I'm posting because the quality numbers at this bit rate look good to me, and I'd like people to poke holes in them.
I'll also show where it's worse than llama.cpp, because there are real downsides.

## The idea in two paragraphs

Usual quants round weights one at a time, or in small groups, onto a grid. LLVQ takes 24 weights together and snaps
them to the nearest point of the Leech lattice, a very dense packing in 24 dimensions. Each block of 24 weights
becomes a 48-bit code, so 2 bits per weight for the codes. Add the scales and tables, plus the embedding in int4, and
you land at 2.70 to 2.73 bits per parameter for the whole model. The encoding runs GPTQ-style, once per model, offline.

The part I spent most of my time on is the decoder. The kernel reads the 48-bit codes directly during the matvec and
never builds a dense matrix. A whole token runs in one CUDA launch. Everything is in Rust, the core crates have no
dependencies, and the kernels are compiled at runtime with NVRTC.

## Same bits, same day, same card

This is the comparison I care about most. I calibrated llama.cpp's IQ2_S myself, with an imatrix built on the same
calibration set I use (dclm-edu), so it lands on the same bits we do: 2.69 to 2.82 b/param. I also added Unsloth's
UD-IQ2_XXS, which is a bit smaller. Everything ran on one L40S the same day. One caveat: at 14B the IQ2_S file sits 0.07 b/param below ours (about
0.12 GB smaller).

[image: fig/02_same_bits_quality.png]

- MMLU (5-shot, all 14,042 questions): +9.1, +4.1 and +2.9 points over IQ2_S at 4B, 8B and 14B.
- GSM8K (all 1,319 problems, greedy): +14.3 at 4B, +3.0 at 8B. At 14B we're tied.
- WikiText-2 perplexity: 12.58 / 9.71 / 8.44 against 16.78 / 10.75 / 9.47.

The gap shrinks as the model grows. My guess is that bigger models take low bits better whatever method you use, so
there's less to win.

About the Unsloth 4B: it loops under greedy decoding. 999 of its 1,319 GSM8K answers hit the token cap repeating
the same sentence. Qwen recommends against greedy decoding, so that number says more about the protocol than about the
file. At 8B and 14B the cap only catches 39 and 9 answers.

## Same memory, one size up

The point of going this low is fitting a bigger model in the same VRAM. So the other comparison is our model one size
up against 4-bit AWQ one size down, at about the same memory.

[image: fig/03_size_up.png]

On MMLU our 8B beats AWQ's 4B by 1.44 points, and our 14B beats AWQ's 8B by 1.87 points while using 1 GB less. On
GSM8K it's a tie both times: we lose enough on reasoning to cancel the size step. Full disclosure: I did this analysis
after a review, it wasn't planned in advance.

## What 2.7 bits costs

Against FP16 of the same model, we lose points. Same as above, the loss shrinks with size.

[image: fig/04_gap_to_fp16.png]

At 4B it hurts: 6.8 points of MMLU, 9.6 of GSM8K. At 14B it's down to about 3 points on both. Against 4-bit AWQ, which
uses roughly twice our bits, we're 4.8 / 4.2 / 2.5 MMLU points behind.

## Where it's worse

**Speed.** llama.cpp is faster, period. IQ2_S decodes 60, 64 and 34 tok/s faster than us after a short prompt, and
the gap stays at long context.

[image: fig/05_speed.png]

At batch 1 we hit 227 / 145 / 90 tok/s on the L40S. Put another way, at the same bits we pull 322 to 458 GB/s out of
the card and llama.cpp pulls 408 to 613. Also, the L40S is the only card where I can show a real gain. On an A100 our files run
slower than FP16 in vLLM, and I haven't measured any consumer card yet.

**Memory on long prompts.** At short context we're tied with IQ2_S (counting the embedding llama.cpp keeps in host
RAM). During an 8K-token prompt our peak goes much higher: 5.95 GB against 3.52 at 4B. That's not the weights, it's
how my prefill handles attention. It's also slow: the 4B takes 35 seconds to reach its first token after an 8K
prompt. That's the next thing I need to fix.

[image: fig/06_memory.png]

**Everything else:**
- Qwen3 dense only, 14B max. No MoE, and nothing bigger has been encoded yet.
- One calibration draw per model. On the 4B, changing the calibration sample moves MMLU by about 3 points (one
  standard deviation). So treat small differences with care.
- MMLU and perplexity were scored on the dense reconstruction of the file, GSM8K through the actual kernel. The kernel
  gives back the same tokens as the reference path, but it's not the same measurement.
- It's a custom format. llama.cpp can't read it. You need my runtime, or `transformers` for the 4B.

## Try it

Files: [4B](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed),
[8B](https://huggingface.co/Pier-Jean/Qwen3-8B-LLVQ-Tetra-sealed),
[14B](https://huggingface.co/Pier-Jean/Qwen3-14B-LLVQ-Tetra-sealed)

The `llvq` runtime speaks the Ollama and OpenAI APIs, so most clients just work:

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
cargo build --release -p llvq-runtime --features cuda    # --features metal on a Mac
target/release/llvq pull qwen3:4b
target/release/llvq run qwen3:4b "Name three primary colours."
```

For the 4B in `transformers`: `pip install llvq-tetra`, then `from_pretrained("Pier-Jean/Qwen3-4B-LLVQ-Tetra")`.
Import `llvq_tetra` first, otherwise transformers silently loads a random model.

Code, paper and every raw measurement log: [github.com/pjmalandrino/llvq](https://github.com/pjmalandrino/llvq).
Each number in this post comes from a log in `docs/mesures/`.

If you have a consumer card (3090, 4090, 5090) and some time, I'd really like to know what speed you get. And if you
think a comparison is unfair, tell me which one and why. I'd rather fix it now.
