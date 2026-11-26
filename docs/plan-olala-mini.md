# Plan: Olala v1.0 mini 7A1B, from a reconnaissance to a served object

Goal: a Tetra file of `OVHaiLLM/Olala-v1.0-mini-7A1B` near 2.6 bits per parameter, 2.2 GB against the 13.6 GB of the
bf16 checkpoint, measured against that checkpoint and against the public quantized files, then served.
State on 2026-11-25: a reconnaissance at $0 is done and nothing is encoded
([olala-reconnaissance](mesures/olala-reconnaissance-2026-11-25.txt)). Every later lot waits for the operator.
Never without him: a paid run, a format change, kernel work on a card, a publication, a served object.

## 1. The object

Olala v1.0 mini 7A1B is OVHcloud's first language model, released on 2026-11-19 under Apache 2.0 with its code, a
70-page technical report and three sibling checkpoints (base, SFT, and two specialised ones). The facts below come
from its model card, its `config.json` and its `modeling_olala.py`, read on 2026-11-25.

- **Size.** 6,773,124,360 parameters in 945 tensors, bf16, one safetensors file of 13.58 GB (*computed* from the
  header). About 1.0 B matrix parameters are read to make a token, 1.23 B with the output head (*computed*).
- **Backbone.** 36 layers of width 1,536: 29 Mamba-3 MIMO layers and 7 differential tensor-product attention layers,
  one in five. Only those 7 keep a KV cache, 15.7 kB a token in bf16 by OVH's count; the Mamba layers keep a state of
  constant size. Context is 32K tokens.
- **Feed-forward.** Every layer ends on a latent mixture of experts: a projection from 1,536 to 384, 256 routed
  experts of 384 to 768 to 384 with six active a token, a projection back to 1,536, and one shared expert of 1,536
  to 1,536 to 1,536. The activation is a squared ReLU. Routing is a sigmoid score plus a stored bias, top 6.
- **Residual.** Each branch output enters the stream through a geodesic update that keeps only its direction and a
  clamped angle, so the stream stays bounded whatever the size of the branch output.
- **Ends.** The embedding and the output head are untied, 151,936 by 1,536 each, on Qwen's tokenizer.

The routed experts are 80.3 % of the parameters, so their rate sets the file (*computed*, tensor table in the
journal, section 1). Every matrix width is a multiple of 24, so no row keeps a tail outside the lattice.

The model is unusual in three ways that matter here. No upstream engine loads it: llama.cpp, vLLM and MLX each
need OVH's fork. Its card scores twelve benchmarks under sampling at temperature 0.8 with reasoning on, and gives no
MMLU and no greedy score. And OVH states that generic quantization recipes return NaN on realistic inputs, because
a few branch outputs reach 1e7 to 1e16 and overflow any f16 scale.

## 2. The field

The smallest public file of this model is 3.72 GB, 4.39 bits per parameter, and OVH's own smallest is 5.99 GB,
7.08 bits per parameter. The table lists every quantized file found on the Hub on 2026-11-25, with the quality its
author reports. Bits per parameter are *computed* here as file bytes over 6.773 B parameters; the quality columns are
each author's own measurement, on their own protocol, and are not comparable across rows from different authors.

| file | GB | b/param | what its author reports |
|---|---|---|---|
| bf16, OVH | 13.58 | 16.04 | reference |
| GGUF q6_k, OVH | 6.31 | 7.45 | HumanEval+, MBPP+ and MGSM within noise of bf16 |
| GGUF q5_k_m, OVH | 5.99 | 7.08 | within noise of bf16; next token equal to bf16's 93.1 % of the time |
| NVFP4 on the routed experts only, `solfird` | 5.77 | 6.81 | 71.8 against 73.5 for bf16 on five tasks, a loss of 1.6 [0.0, 3.3] |
| MLX q4e8, OVH | 4.87 | 5.75 | not read |
| GGUF 3.7G, `ciselor59` | 3.72 | 4.39 | WikiText-2: perplexity 14.29 against 13.79, KLD 0.0587, top-1 88.8 % |

Three facts from these authors frame the target. OVH built a Q4_K_M file and withdrew it: it scored lower, reasoned
40 % longer and hit the context limit more than twice as often, and OVH's report says the fine-tuned model loses
quality already at 4 bits. OVH's recipes keep the 384-column projection `up_proj` in bf16. And `solfird` measured
the bf16 model at 234 tok/s for one request in OVH's vLLM fork on an RTX PRO 4500, against 85 tok/s for the Q5_K_M
file in OVH's llama.cpp fork on the same kind of pod.

A file at 2.6 bits per parameter would be 41 % smaller than the smallest public one and 63 % smaller than OVH's
smallest (*computed*). Nobody has published a file of this model under 4 bits.

## 3. What the reconnaissance settles

The reconnaissance ran OVH's own forward on a CPU and the repository's encoder on real tensors, at $0. Its numbers
are all in the journal; this section keeps the five that shape the plan.

**The weights suit the lattice.** Data-free, the Tetra encoder leaves the same relative error on Olala's matrices
as on Gaussian ones, 0.085, on 14 of 20 matrices tried and at most 0.102 on the others (*measured*, journal section
3). Layers 0 and 1 look pathological as stored, with 1 % of the rows holding 99 % of the energy, but a per-row
scale and the rotation make them Gaussian again, and the format has both. Without the rotation the error triples on
the output-side matrices.

**The routed experts are the cheap place to cut, and the dense path is not.** Isotropic noise of the encoder's size
on all routed experts costs +0.10 of NLL, a perplexity ratio of 1.11, with 80.9 % of next tokens unchanged
(*measured*, a noise proxy on 8,176 positions, journal section 6). Adding the same noise to the 0.80 B parameters of
the large dense matrices takes the cost from +0.07 to +0.82 on the one window where both were read. Among them the
shared expert and `mixer_proj` are the most sensitive, and the router the least.

**The latent projections must stay wide.** Noise of int4's size on `down_proj` and `up_proj` alone, 0.6 % of the
parameters, costs +0.015 of NLL on one window, where the same noise on all 5.4 B expert parameters costs +0.013 on
another (*measured*).
This agrees with OVH keeping `up_proj` in bf16.

**The branch outputs overflow f16.** The mixer output reaches 2.9e16 at layer 1 and the routed experts' output 1.1e7
at layer 0, as OVH reports, and layers 2 and 3 add 2.6e8 and 1.4e6 that the report does not name (*measured*,
journal section 4). A kernel must carry these paths in f32, and the Hessians of the matrices that read them are
dominated by a few directions.

**Routing is skewed but not starved.** Over 24,576 tokens one expert of 9,216 was never routed, the busiest expert
of a layer takes up to 28 % of the tokens, and a fifth of the experts carry 43 % of the traffic (*measured*, journal
section 5). At the 131,072 calibration tokens of the sealed recipe, about 8 % of the experts would see fewer tokens
than the 768 columns of their second matrix (*computed* from that census, on a text that is not the calibration
corpus). The Inkling encode met 28 % of such cells and took no fallback.

**A second word buys what a better code cannot.** At 48 bits a block the code is within 0.60 dB of any
24-dimension rival ([`CERVEAU.md`](CERVEAU.md) C7), so its error at 2 bits is close to final. A second Tetra word
on the residual takes the error from 0.085 to 0.0074 at 4.00 bits a weight, under int4 g128's 0.0100 at 4.25
(*measured*, [olala-second-mot](mesures/olala-second-mot-2026-11-26.txt)). Under the proxy, a second word on the
busiest 20 % of the experts halves the KL to bf16, from 0.125 to 0.061. The format holds one word a block today.

The noise proxy is not an encoded file. It ignores the Hessian, which is the whole of what GPTQ uses, and by
[`MODES.md`](MODES.md) a gain projected through a proxy is not a result. It ranks the parts of the model and
bounds nothing.

## 4. The recipe proposed

The proposed file puts the routed experts on the lattice, the large dense matrices in int4, and leaves the small
sensitive ones as shipped: 2.61 bits per parameter and 2.21 GB (*computed*). The table gives each part, and the
alternatives stand below it.

| part | parameters | type | GB |
|---|---|---|---|
| routed experts, 36 layers x 256 x 2 matrices | 5,435,817,984 | Tetra, 2.031 b/weight with its row scales | 1.380 |
| Mamba-3 `in_proj`, `mixer_proj`, shared expert, attention `c_q` and `gate_proj` | 798,990,336 | int4 g128 | 0.424 |
| embedding and output head | 466,747,392 | int4 g64 | 0.263 |
| latent projections, router, TPA factors, MIMO, norms, scalars | 71,568,648 | as shipped, 16 bits | 0.143 |
| whole model | 6,773,124,360 | | 2.21, 2.61 b/param |

| alternative | b/param | GB | standing |
|---|---|---|---|
| experts on the lattice, everything else bf16 | 4.79 | 4.05 | larger than the smallest public file |
| the Inkling recipe: experts on the lattice, ends in int4, the rest bf16 | 4.00 | 3.38 | a safe first arm |
| **the proposed recipe** | **2.61** | **2.21** | the target |
| the proposed recipe with a second Tetra word on the busiest 20 % of experts | 2.94 | 2.49 | the fallback if the gate fails; a format change |
| the same on the busiest 50 % | 3.43 | 2.90 | |
| the same on every routed expert | 4.24 | 3.59 | the size of the smallest public file |
| every matrix on the lattice | 2.35 | 1.99 | ruled out by the dense-path noise arm unless GPTQ changes the picture |

The embedding and the head were not put under noise, so their int4 form rests on the three Qwen3 files and on
Inkling, where the int4 head cost +0.41 % of perplexity (*measured*,
[inkling-cuda-stack-ppl](mesures/inkling-cuda-stack-ppl-2026-11-23.txt)).

## 5. Targets on each fundamental criterion

Each target states its basis. None is a prediction of the encoded file: the quality of a Tetra file of this model
has never been measured.

**Memory. Target: 2.61 b/param, 2.21 GB of file, under 4 GB on the card at 32K tokens.** The file size is
*computed* (section 4). On the card the weights are joined by the 7 attention caches, 0.51 GB at 32K tokens in bf16
by OVH's 15.7 kB a token, the Mamba state and the CUDA context; the sealed 4B adds 0.60 GB over its weights after
load (*measured*, [`ETAT.md`](ETAT.md) §2). OVH asks 24 GB or more for its vLLM path and 16 GB for its 6-bit file
on a Mac. This is the criterion on which the plan is most certain.

**Quality. Gate: match the smallest public file with 41 % fewer bytes.** On WikiText-2, in one engine, the 2.21 GB
file reads a KL to bf16 at or under the 3.72 GB file's and a top-1 agreement at or above it; that file's author
reports 0.0587 and 88.8 %. The honest reading tonight is that this gate is hard. The isotropic proxy puts the
experts alone at a KL near 0.16 on our text, and on Inkling the real encode of the experts at 2.04 bits read 0.083
on the whole model and did not beat the public ladder ([whole-model](mesures/inkling2-mini-whole-model-2026-11-22.txt)).
Three levers stand between the proxy and the gate, none measured on this model: GPTQ against real Hessians, trained
row scales, which gained 1.7 to 3.3 MMLU points on the Qwen3 files, and a second word on the busiest experts
(section 3). The fallback target is the same gate at 2.94 b/param, 2.49 GB, still 33 % under the smallest public
file. The task target comes after the
gate: within 3 points of bf16 on the five tasks `solfird` scored, where the NVFP4 build lost 1.6 and a 4-bit GGUF
4.2, with the share of answers that run out of context reported beside every score.

**Decode speed. Stage 1: 85 tok/s for one request on an L40S. Stage 2: 234 tok/s.** These are the two public
figures of section 2, taken as bars and not as ratios, since they were measured by others on another card. The
proposed file reads about 0.73 GB of weights a token against 2.46 GB in bf16 (*computed*), so bandwidth leaves
room. The risk is the launch count: Inkling's first fused stack decoded 72 tok/s on an L40S with 889 launches a
step and no kernel above 17 % ([inkling-cuda-kernels](mesures/inkling-cuda-kernels-2026-11-24.txt)), and Olala has
36 layers of mixer plus routed experts. Stage 2 needs the persistent kernel, which was not built for Inkling. On a
CPU the bar is OVH's own 46 tok/s on a desktop Ryzen with the Q5_K_M file; our CPU backend decodes the 4B at 8.44
tok/s on 8 vCPU ([llvq-cpu-x86](mesures/llvq-cpu-x86-2026-11-25.txt)), so the CPU is OVH's ground and not where
this plan expects to lead.

**Time to first token. No target yet.** Nothing in the repository reads a prompt through a Mamba-3 layer, and no
public figure isolates the prefill of this model on a GPU. OVH's llama.cpp fork reads 800 to 990 prompt tokens a
second on a MacBook Air M5. The first measurement sets the target.

**Cost of producing the object. Target: $0 for the encode, under $25 of card time to the task scores.** OVH's
forward runs 2,048 tokens a minute on 4 CPU threads, so the 131,072 calibration tokens are about an hour before the
Hessian sums (*computed* from the reconnaissance). The encode of the 9,216 experts is about 7 hours on those 4
threads, data-free timing (*computed*, journal section 3). The Hessians of the experts weigh 54 GB in f64 before
any symmetry is used (*computed*), so they are written layer by layer. Inkling's whole campaign spent $8.88 of card
time against a cap of $25 (`docs/data/jobs.csv`). Trained row scales need a card and are not costed for this model.

## 6. The lots

The lots are ordered so that the cheapest one that can stop the project runs first. O1 decides whether a file
under 3 bits of this model is worth an engine; nothing after it starts before it reads.

**O1. The quality gate, $0.** Collect the Hessians through OVH's forward on dclm-edu, 64 windows of 2,048, one
Hessian per expert and per input, weighted as for Inkling. Encode with `extq::encode_matrix`, which takes any width
that is a multiple of 24; `bin/extq.rs` hard-wires Inkling's 2,304 and 1,152 and has to read the shapes instead.
Rebuild the dense model, and read KL, top-1 and perplexity against bf16 on WikiText-2 in the same forward, arm by
arm: experts only, then the int4 parts, then the whole recipe. Read the 3.72 GB and Q5_K_M files in the same engine,
by converting our reconstruction to GGUF or by loading theirs into the PyTorch forward. The prereg states the gate
of section 5 before the first number. Two to four agent-days (*estimated*, with no measured basis).

**O2. The levers, if O1 misses.** In order of cost: a second word on the busiest experts, first as a dense
reconstruction and only then as a format; a floor for experts
under their column count; the spike layers, where the choice is between a data-free encode, f64 Hessians, and OVH's
own rewrite that normalises the branch output at layers 1, 7 and 23; trained row scales, which need an adapter in
`ops/llvqtune` and a card.

**O3. Task scores, paid.** The dense reconstruction through OVH's vLLM image on a card: the five tasks of the
`solfird` protocol and the overflow rate, bf16 and our file in the same job, paired. A greedy GSM8K and MMLU keep
the link with the Qwen3 tables. Needs a go and a cap.

**O4. A file a stranger can load.** `hfpack` and the `llvq-tetra` reader, decoding to dense at load inside OVH's
`transformers` code. This gives the download size and not the memory. Publication is the operator's.

**O5. Our own engine.** A Rust forward checked by `oracle` against the PyTorch one before anything else, then the
CPU backend, then CUDA. New against Inkling: the Mamba-3 MIMO step and its state, differential tensor-product
attention, the geodesic update, the shared expert (`config.rs` refuses one today), the latent projections, the
squared ReLU, and f32 on the spike paths. Reusable: the routed kernels of `moe_tetra.cu`, the top-k glue, the
expert records, the int4 head and embedding. This is the long pole: Inkling's plan put its whole programme at 43 to
91 agent-days, 11 to 19 of them for the track without an engine, and its persistent kernel was not reached.

**O6. Serve.** A family in `llvq-runtime`, the chat template with its three channels, the four reasoning efforts,
tool calls.

## 7. Traps

- **A new version is announced.** The card says a 128K-context version is coming soon. Every lot has to rerun on new
  weights from scripts, and a served object would follow a moving target.
- **Scores under sampling.** OVH and `solfird` score at temperature 0.8 with reasoning on, and their standard error
  is 2 to 3 points a task. The paired bar of this repository, 0.43 pp, does not transfer; a task verdict needs
  several samples a problem or a greedy protocol of our own.
- **Longer reasoning counts as a loss.** At 4 bits the public files fail by reasoning longer and overflowing the
  context, not by a lower perplexity. A perplexity gate alone would miss it.
- **Near-tie routing.** The gap between the 6th and 7th expert has a median of 2.5e-5 at layer 1, so any change
  upstream of the router flips experts there. Inkling met the same and logged a logit drift from one flip.
- **The Mamba state in bf16.** OVH's vLLM fork must be run with a float32 state or long generations loop. Our
  engine has to hold the state in f32 from the start.
- **Half the measurements of this document are on the wrong text.** The census and the noise arms used OVH's report
  and one of our documents, in windows of 512. O1 redoes them on the calibration corpus and on WikiText-2.

## 8. Decisions awaited

| decision | default if silent |
|---|---|
| open O1, the quality gate at $0 | not opened |
| the rate to aim at: 2.61 b/param, or 2.94 with a second word on the busiest experts from the start | 2.61, the fallback only after a miss |
| a second word a block as a format and kernel change | not built; read as a dense reconstruction only |
| a spend cap for O3 and for row scales | no paid job |
| wait for the 128K version before any paid work | no wait for O1, wait for the rest |
| Olala as a served family, O5 and O6 | not started |
| publication of any file or card | not published |
