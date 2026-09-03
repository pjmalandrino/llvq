# Running the published model

Qwen3-4B quantized to 2 bits on the Leech lattice fits in a 1.771 GB file (*measured*,
[docs/fiche-4b.md](docs/fiche-4b.md)) that starts on its own: no checkpoint, no Hugging Face cache,
no network. The current state of the project is in [docs/ETAT.md](docs/ETAT.md), the change history
in [docs/HISTORIQUE.md](docs/HISTORIQUE.md), the measurement rules in [docs/METHODE.md](docs/METHODE.md).

## 1. The file

`qwen3-4b-llvq.bin` is 1,770,527,533 bytes (*measured*, [docs/fiche-4b.md](docs/fiche-4b.md)) and
lives on the Hub: [huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-2bit).
Its sha256 is `9db213ef9fa9d7d7000789a8a529ce9459ce9ba6002ef5a72fd5a1c05c1c84b0`. A different hash is
a different file.

| section | content | bytes | provenance |
|---|---|---|---|
| matrices | 252 quantized projections, 3,633,315,840 weights, 47-bit index + 1 gain bit per block of 24 | 980,790,202 | *measured*, fiche-4b |
| f16 tensors | 146 carried tensors: embedding 388,956,160 values, norms 196,096 | 778,313,898 | *measured*, fiche-4b |
| blobs | `config.json` (726 B), `tokenizer.json` (11,422,654 B) | 11,423,433 | *measured*, fiche-4b |

The projections cost 2.1595 b/weight, tail included in the denominator (*computed*, fiche-4b;
2.1696 tail excluded, same file). The whole model costs 3.5213 b/param on disk, f16 embedding
included (*computed*, fiche-4b). The container carries the magic `LVQ2`. The format is defined by
[`llvq-artifact`](llvq-artifact/), a crate with zero dependencies. Its tree is 3 crates against 261
for the model side (*measured*, [README.md](README.md)).

This file is not a GGUF, not an AWQ, not a safetensors. `transformers`, `llama.cpp`, vLLM and TGI
do not read it. The only reader is this repository. No reader exists in another language; one would
also need `llvq-search` and `llvq-core` for the Leech index.

## 2. Download and run

Prerequisite: stable Rust. The download is the only step that touches the network.

```bash
git clone https://github.com/pjmalandrino/llvq && cd llvq
hf download Pier-Jean/Qwen3-4B-LLVQ-2bit qwen3-4b-llvq.bin --local-dir .
shasum -a 256 qwen3-4b-llvq.bin        # expected: 9db213ef…84b0, 1,770,527,533 bytes
```

Generate on a Mac (Metal) or on any machine (CPU) with `bin/run`, the dense path:

```bash
cargo run --release -p llvq-llm --features metal --bin run -- qwen3-4b-llvq.bin metal 24
```

On anything other than a Mac, drop `--features metal` and replace the `metal` argument with `cpu`.
The cargo feature and the argument are two distinct things; asking for `metal` without the feature is
an error. The last argument is the number of tokens. The program prints `252 quantized matrices + 146
carried tensors`, then four sampled prompts, so not reproducible to the letter.

`bin/run` decodes every weight to f16: the resident model is 8.045 GB whatever the file (*measured*,
fiche-4b). Peak RSS: 9.79 GB on `cpu`, 17.41 GB on Metal (*measured*, fiche-4b). Budget 10 GB of
free RAM on CPU and 17.4 GB on Metal. Throughput: 42.7 tok/s on an L40S with a KV cache (*measured*,
[docs/mesures/mini-2026-08-05.txt](docs/mesures/mini-2026-08-05.txt)); no Mac measurement since the
KV cache.

The fused path, on Linux + CUDA, is `bin/fusedrun`. It decodes and multiplies without going back
through a dense matrix. The served configuration v1 fits in three variables:

```bash
LLVQ_EMBED=q8 LLVQ_ROT_SHARE=1 LLVQ_FUSE=1 \
  cargo run --release -p llvq-llm --features cuda --bin fusedrun -- qwen3-4b-llvq.bin 128
```

| path | tok/s [range] | GB on card | greedy tokens | provenance |
|---|---|---|---|---|
| fused, config v1 (`planes14` + q8 + hoisted rotation + fusion) | 100.6 [99.9–100.7] | 2.57 | 128, diverges from the dense arm at token 89 | *measured*, [d1-fusion-servie-2026-08-24.txt](docs/mesures/d1-fusion-servie-2026-08-24.txt) |
| dense f16 (the companion arm) | 43.5 [43.4–43.5] | 8.04 | reference | *measured*, [b2-fusedrun-plages-2026-08-18.txt](docs/mesures/b2-fusedrun-plages-2026-08-18.txt) |

The raw ratio against this arm is never quoted alone: that arm copies 778 MB of vocabulary per token.
The same-head ratio, ×1.11 [1.11–1.11], measures the kernel (*computed* on the B2 medians). NVRTC
compiles the kernel at startup; `LLVQ_NVRTC_ARCH=compute_80` targets the A100, where the same kernel
returns 0.79× FP16 (*measured*, [f4-a100-2026-08-18.txt](docs/mesures/f4-a100-2026-08-18.txt)).

## 3. Validate, in order

Four checks, from fastest to longest.

| no. | question | command | expected | duration |
|---|---|---|---|---|
| 1 | is the mathematical core correct | `cargo test --release` | all green; the `ignored` tests are the archive sweeps | a few minutes (*estimated*) |
| 2 | is the file self-contained | `env -i HOME=/nonexistent PATH=/usr/bin:/bin ./target/release/run qwen3-4b-llvq.bin cpu 12` | an answer, with no HF cache and no network | 4 to 5 min (*estimated*; 255.7 s *measured* before the KV cache, [audit-publication-2026-08-03.md](docs/archive/audit-publication-2026-08-03.md)) |
| 3 | does the thesis hold | `cargo run --release -p llvq-metal --bin thesis -- qwen3-4b-llvq.bin` | 1,105,920 rows verified, worst error 3.4e-8·Σ\|w·x\|, ×2.03 [2.03–2.10] (*measured*, 7-arm bench, median of the round-by-round ratio, [k1-metal-2026-08-05.txt](docs/mesures/k1-metal-2026-08-05.txt)) | ~4 min (*estimated*), Mac, ~12 GB free |
| 4 | is the quality as announced | the two `ppl` commands below | 16.9415 against 12.2361 (*measured*, fiche-4b), fingerprint `3f1baca9033bf251` on both sides | ~15 min (*estimated*) plus the checkpoint the first time |

Step 1. The fast loop covers the Λ₂₄ invariants (196,560 kissing vectors, theta series) and exact
search against brute force. It also covers the bijectivity of the 48-bit index, the GPTQ loop
against an independent analytic minimizer, and the bit-for-bit round trips of the five runtime
formats. The archive sweeps run once the file is downloaded and take tens of minutes: 17 min without
finishing `llvq-artifact` on 2026-08-08 (*measured*, [CLAUDE.md](CLAUDE.md), section "Commands"),
10 min 51 s for its 45 tests alone (*measured* on 2026-08-08, no journal):

```bash
LLVQ_SEALED_ARTIFACT=$PWD/qwen3-4b-llvq.bin cargo test --release -- --include-ignored
```

Without the archive, these sweeps fail and name the file.

Step 2. A non-existent `HOME` and an empty environment: no Hugging Face cache is reachable. The
255.7 s measurement predates the KV cache of commit `9c24d26`; no measurement since.

Step 3. `bin/thesis` transcodes the 252 matrices to the kernel format, checks every output row
against an f64 CPU reference, then times one token of projections in both formats. It measures
`Slot32` on Metal: FP16 21.9 ms median for 7.27 GB read, LLVQ 10.7 ms for 2.50 GB (*measured*,
[k1-metal-2026-08-05.txt](docs/mesures/k1-metal-2026-08-05.txt)). The ratio is a range: the
milliseconds drift from 2.029× to 2.080× on an unchanged binary. This bench is not the served path,
which is `Planes14` on CUDA. With no argument it looks for `~/llvq-q4b.llvq`, an unpublished working
file.

Step 4. WikiText-2 perplexity, context 4096, 12 windows, f16 on both sides:

```bash
LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal qwen3-4b-llvq.bin
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_DTYPE=f16 cargo run --release -p llvq-llm --features metal --bin ppl -- 4096 12 metal
```

Expected: 16.9415 on the sealed file against 12.2361 on the checkpoint, a ratio of ×1.3846
(*measured*, fiche-4b). The two result lines must carry the same token fingerprint.

## 4. Against FP16 and against 4-bit

At 4B, LLVQ beats AWQ on disk and on served memory, and loses on quality. The useful reference is
4-bit: nobody chooses between 2 and 16 bits. The 4-bit measured here is Qwen's official AWQ, same
card, same harness, same token fingerprints.

| axis | LLVQ 2-bit | FP16 | official AWQ 4-bit | provenance |
|---|---|---|---|---|
| disk | 1.771 GB | 8.045 GB | 2.67 GB | 1.771 and 8.045 *measured*, fiche-4b; 2.67 *measured*, [campagne-finale-2026-08-07.md](docs/campagne-finale-2026-08-07.md); ×4.54 over FP16 (*computed*) |
| RAM of the dense path (`bin/run`) | 9.79 GB cpu, 17.41 GB Metal | 8.045 GB resident | not measured here | *measured*, fiche-4b |
| VRAM of the fused path, b/param whole model | 2.56 GB, 5.162 (`Planes14` + q8 without fusion); 2.57 GB in config v1, +3,686,400 B | 8.04 GB, 16.0 | 5.302 in its own engine | 2.56 GB *measured*, b2; 5.162 *computed* on measured bytes, [rtbits-planes-8b-2026-08-09.txt](docs/mesures/rtbits-planes-8b-2026-08-09.txt); 2.57 GB and +3,686,400 B *measured*, d1 |
| speed, L40S | 100.6 tok/s; ×1.11 same-head | 43.5 tok/s | 200.5 tok/s in vLLM, another stack, does not divide | *measured*, d1, b2, [awq-vllm-4b-2026-08-17.txt](docs/mesures/awq-vllm-4b-2026-08-17.txt) |
| WikiText-2 perplexity | 16.9415 (×1.385) | 12.2361 | ×1.105 | *measured*, fiche-4b, [a4-campagne-2026-08-06.txt](docs/mesures/a4-campagne-2026-08-06.txt) |
| MMLU 5-shot micro | 55.59 ± 1.35 | 70.32 ± 1.28 | 70.04 | *measured*, a4-campagne, fingerprint `65dcd53655e8bfa5` |

Memory: LLVQ comes in under AWQ in b/param over the whole model, 5.162 against 5.302. Every memory
comparison is stated in this accounting, embedding included; comparing a b/weight of projections
with a b/param of the whole model misleads. Quality: LLVQ loses 14.73 pp of MMLU where 4-bit loses
0.28. On a 4B, 4-bit dominates everywhere except disk. The gap shrinks with size (*measured*:
7.49 pp at 8B, [mmlupair-4b-8b-2026-08-13.txt](docs/mesures/mmlupair-4b-8b-2026-08-13.txt);
6.09 pp at 14B, [mmlupair-14b-2026-08-17.txt](docs/mesures/mmlupair-14b-2026-08-17.txt)), which does
not amount to a scaling law; see [docs/ETAT.md](docs/ETAT.md). The fused kernel is not wired into
`bin/run`: the portable demo stays dense, the speed gain lives in `bin/fusedrun`.

## 5. Rebuilding the model yourself

Quantizing the 4B takes 4.01 h on an M3 Max, 14,447 s (*measured*, fiche-4b). The run checks its own
file by decoding it and requiring the evaluated weights bit for bit:

```bash
LLVQ_MODEL=Qwen/Qwen3-4B LLVQ_CALIB=c4 LLVQ_ARTIFACT=q4b.llvq \
  cargo run --release -p llvq-llm --features metal,fast-linalg --bin smoke -- \
  64 2048 12 4096 metal nogs leech1c12 999 rot
```

Positionals: 64 calibration windows of 2048 tokens (131,072 tokens); 12 windows of 4096 for the
evaluation; backend `metal`. Then `nogs`, group scales disabled; codebook `leech1c12`, the Λ₂₄(12)
ball and 1 gain bit; seed 999; input rotation. `fast-linalg` is required in practice: without it the
factorization is 40× slower for a bit-identical result (*measured*, README). `LLVQ_THREADS=4` caps
the encoding pool.

The file produced carries only the projections. To make it self-contained, with embedding, norms,
config and tokenizer:

```bash
LLVQ_MODEL=Qwen/Qwen3-4B cargo run --release -p llvq-llm --bin seal -- q4b.llvq qwen3-4b-llvq.bin
```

This recipe reproduces the method without reproducing the bytes. Three deviations from the published
run, all documented in fiche-4b. The C4 calibration shard moved from `00000` to `00001` after the
run. The container magic moved from `LVQ2` to `LVQ4`, which also stores the codebook fingerprint.
`calib.rs` accumulates AᵀA in f32 on the accelerator, so a re-run on CUDA returns other weights, a
deviation not quantified. A re-run returns 1.771 GB at 2.1595 b/weight in a file that is not
identical. CI checks the code: clippy, tests, the zero-dependency guard on the five core crates
(`llvq-core`, `llvq-search`, `llvq-quant`, `llvq-artifact`, `llvq-bench`;
[ci.yml](.github/workflows/ci.yml)). It does not check the bytes of the artifact.

## 6. License

The model is under Apache 2.0, inherited from [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).
The code is under MIT OR Apache-2.0 ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)).
