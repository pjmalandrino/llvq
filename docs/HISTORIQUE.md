# History

The project's chronological thread, one entry per period, from 2026-09-26 on; the entries from 2026-07-24 to 2026-09-25
are in [archive/historique-2026-07-24-to-09-25.md](archive/historique-2026-07-24-to-09-25.md). The current state is in [ETAT.md](ETAT.md), the lab rules in [METHODE.md](METHODE.md), the writing rules in [STYLE.md](STYLE.md). Entries are appended at the bottom and the past is not rewritten.

## 2026-09-26. The repository catches up with its paper

Three journals the paper's provenance table names had never been committed: the 14B reference census, the 14B base
census and the 14B row-scale training. They were on the Mac, untracked. Every journal, dump and preregistration the
paper cites is now in the repository, and the four stamps attest the bytes committed.

The living documents were three weeks behind the object. `ETAT.md` still gave `Planes14` as the served configuration at
55.59 MMLU, `README.md` and `CLAUDE.md` still pointed at the earlier file. They are rewritten on the sealed Tetra
objects and brought back to the lengths `STYLE.md` sets: ETAT 566 lines to 134, ROADMAP 407 to 117, README 187 to 126,
CLAUDE 161 to 146. The entries of this file between 2026-09-05 and 09-17 were out of chronological order and are
reordered.

## 2026-09-26 and 27. A reasoning test: GSM8K through the served kernel

The three sealed files score **82.49, 88.63 and 92.04 on GSM8K** through the served kernel, against 92.12, 93.25 and
95.30 for FP16 and 89.01, 92.95 and 95.38 for AWQ in vLLM (*measured*,
[gsm8k-wave1](mesures/gsm8k-wave1-2026-09-26.txt), [gsm8k-wave2](mesures/gsm8k-wave2-2026-09-26.txt)). It is the
first quality score read on the served object over a whole test split. The paired gap to FP16 is 9.63 points at 4B,
1.4 times its MMLU gap. At 8B and 14B it is 4.62 and 3.26, and does not differ from the MMLU gap.

A new harness generates and grades through the kernel (`bin/gsm8k`), vLLM generates the references on the same prompt
ids (`ops/gsm8k_vllm.py`), and one grader scores every arm (`bin/gsm8kpair`). A free Metal pilot priced the campaign
first ([gsm8k-pilot](mesures/gsm8k-pilot-2026-09-26.txt)). The engine gate held: FP16 reads 91.51 in our dense path
against 92.12 in vLLM. The campaign cost 14.26 $.

## 2026-09-28. One size up at the same memory

The review of paper 2 found no comparison at equal memory. The committed dumps answer it for free: one size up, our
file beats AWQ on MMLU and ties it on GSM8K (*measured*, [sizeup](mesures/sizeup-2026-09-28.txt)). Our 8B scores 1.44
points above AWQ's 4B and our 14B 1.87 above AWQ's 8B; on GSM8K they tie at −0.38 and −0.91. The reasoning loss takes
back the larger model. Our 14B in 5.04 GB scores 5.52 MMLU points above FP16's 4B in 8.04 GB and ties it on GSM8K.
The 4B and 8B share one KV cache, so the step from 4B to 8B costs weights only. Paper 2 carries it as section 6.4 and
Table 5, marked as not preregistered.

## 2026-09-28. Paper 2, every number audited

1,132 numbers of paper 2 were checked against their CSVs, journals and code, by eight auditors,
eight skeptics and a blind sample; about a hundred corrections went in, each checked by hand
against its source (*computed*, [paper2-audit](mesures/paper2-audit-2026-09-28.txt)). The main
ones: the calibration spread of three draws is a range of 5.83 points, not "2.92", which is its
standard deviation; the 4B MMLU was scored one step before sealing; the cited LLVQ scores use the
0-gain-bit codebook; the cost of the Tetra split was measured, not only estimated. Four results
are new: the shell bound is m <= 26, found by an exhaustive pass that replaces a sweep which was
not a bound; Tetra against Planes14 on the 2,280 sample is +2.11 [-0.95, +5.17], not separated;
the size-up budgets move by one hundredth once summed from unrounded bytes; and the decode costs
about 150 instructions a block, not 141. The paper now runs to 19 pages. Two fresh readers
then checked every changed line and caught three more contradictions, among them the tail
width in the caption of Table 2.
The same day, the 24 cited works and the 67 provenance rows were checked against their
publications and files: 55 problems, one dead URL (the CUDA guide), and a dozen attributions
narrowed to what the cited work says, the chief one that the cited QTIP scores and the timed QTIP
kernel are two different codes. 20 pages.

## 2026-09-28. Paper 2, from 20 pages to 14

The same paper in 14 pages, with no number changed and three intervals added (FP16 minus AWQ, from paper2-gaps.csv).
Four readers proposed 212 cuts and four skeptics tried to refute each; five editors applied the kept ones and five
verifiers checked every diff. The provenance table (2.3 pages) moved to `paper2/PROVENANCE.md` and the commands to
`paper2/README.md`. The tile figure, the scale figure and panel (a) of Figure 2 went, their values kept in the text.
The defects paragraph, an erratum about a draft never distributed, the second training and four limitations that
restated a caveat of the body went too. Each caveat of the audit is stated once. The intro and the conclusion now name
section 6.4, marked as chosen after measuring. Appendix A and its QTIP row, the full predictions table and Figure 4
stay.

## 2026-09-28. Paper 2, four review points and a proofreading pass

A reviewer's four points, all taken, and the paper grows back to 15 pages. "Tie" becomes "not separated", defined once:
no equivalence test was run, and the GSM8K intervals of the one-size-up pairs reach down to -2.14 and -2.35 points. The
14B divergence at token 78 is reported as an observation: it moves to token 137 with f16 tables, the prefill gate shows
logit differences up to 1.35 between two paths of the kernel, and the margin at token 78 was never measured. The paper
now says the int4 choices were made on MMLU test questions, with no held-out split; the one measured case, `o_proj`,
reads +3.12 on the questions that chose it and +1.55 on the other 11,762 (*measured*,
[oproj-int4-full](mesures/oproj-int4-full-2026-09-17.txt)). GSM8K chose nothing. With AWQ's tables in 4 bits, our files
would need 64 to 65 % of its bits per parameter, not about half (*computed*, [sizeup](mesures/sizeup-2026-09-28.txt)).
A proofreading pass fixed six slips, among them the running header, whose small-caps ligature extracted as "qantized".

## 2026-09-28. Stage 0 of the transformers plan: the sealed 4B as safetensors

The served 4B has a faithful safetensors image, 1602 fields rebuilt bit for bit by an independent Python reader
(*measured*, [hf-safetensors-4b](mesures/hf-safetensors-4b-2026-09-28.txt)). The directory is 1.422 GB against the
sealed file's 1.418, so the object stays compressed: `bin/export` writes 8 GB of f16 and is an interchange artifact,
this one is a distribution candidate. Three format decisions were taken before the code and recorded in the prereg: our
own tensor naming, the disk's bytes as the code payload rather than the served `tetra48` layout, and the rotation
carried as its two tables rather than its seed. `code_stream` and `code_widths` are now public in `llvq-artifact`, and
`write_matrix_raw` calls them, so a packer cannot disagree with the format about a width or a byte order. The gate was
read three ways, one mutant each, all caught; the f16 fields caught a real defect first, the reader casting f16 to u16
instead of reading its bytes, which hashed a scale of 0.001 as a zero. One signed prediction missed: the
`quantization_config` block is 131 KB against a predicted 20 to 60, from pretty printing. Nothing loads the directory
through `from_pretrained` yet, which is stage 1.

## 2026-09-30. Stage 1: transformers reads the packed 4B and answers

`from_pretrained` on the packed directory gives the weights the artifact defines and the tokens our
engine gives: 253 of 253 dequantized weight digests identical to `decode_matrix`, and 256 of 256
greedy ids identical to `bin/run` over four prompts, f32 on the CPU on both sides (*measured*,
[hf-quantizer-4b](mesures/hf-quantizer-4b-2026-09-30.txt)). No missing key and no unexpected key.
The Leech decode is ported to numpy over the tables `bin/tetratables` dumps from
`llvq_search::tetra`, 19 KB shipped with the package rather than with every model, because the map
belongs to the codebook. Bit-exactness was predicted and held on the first run, including the one
place named as the risk: the `k` by `k` mix, accumulated term by term rather than through a matrix
product, which a library would be free to reassociate. Two facts about `transformers` 5.17 shaped
the code: `create_quantized_param` no longer exists, so a quantizer replaces its modules and the
ordinary loader fills them, and that loader keeps the dtype of an existing buffer, which is what
lets the f64 row scales survive a load at f16. Replacing the `nn.Embedding` broke the tie to
`lm_head`, since `tie_weights` runs before the quantizer's post-load hook, so the quantized
embedding goes through the conversion pipeline instead, three keys to one parameter. The Python
side lives in `llvq-hf/` and is extracted at stage 5.

## 2026-09-30. Stage 2: the served shader decodes the 4B under PyTorch

`torch.ops.llvq.tetra_decode` dispatches `tetra48_probe`, the shader's own decoder entry point, and
every one of the 118,665,216 blocks of the packed 4B decodes to the point the numpy path gives:
2,847,965,184 coordinates, exact, in 38.4 s (*measured*,
[hf-metal-decode-4b](mesures/hf-metal-decode-4b-2026-09-30.txt)). No MSL was written for it. The two
sides of that gate were already pinned to `llvq_search::tetra` before the stage began, one by
`llvq-metal/tests/tetra48_matches_rust.rs` and the other by stage 1's control 2, so the gate closes a
triangle. The plan's gate for this stage was unachievable and was narrowed rather than widened:
stage 1's dequantization is an f64 chain and Metal has no f64. The fused matvec moved to a stage 2
bis, blocked on `upload_int4` staging `d_in · 4` bytes against a 32 KB threadgroup limit, which
refuses all three sealed files on the Metal fused path. One defect was found and fixed on the first
dispatch: the transcode from the disk stream to the served layout is not a byte reversal, because the
gain bit moves from bit 0 of the disk value to bit 47 of the Tetra word.

## 2026-09-30. M1: the 4B answers with its Tetra weights never materialized

`tv_tetra48_metal`, the served shader, runs in a `transformers` forward pass: 256 greedy ids of 256
against `bin/run`, the 168 Tetra projections resident in 0.749 GB against 11.4 dense, 5.436 GB
allocated on the device against 16.1 computed for the dense arm, and the load down from 153.7 s to
6.5 because the fused path never dequantizes (*measured*,
[hf-metal-m1-4b](mesures/hf-metal-m1-4b-2026-09-30.txt)). What remains dense is the 21 % of int4
records, whose Metal kernel is blocked by a 32 KB staging limit, and the embedding. The rotation runs
in torch f32 rather than through `rot_apply_metal`, at 0.15 ms of the 0.25 a matvec costs.

The lot's real finding is about the method. Dropping the tail costs 8.79 % of the output on `k_proj`
and 3.53 % on `down_proj`, per row, and the token gate misses it over eight ids on all four prompts
and over all 64 on two of them. Stages 0 to 2 and M1 all gated on token identity; the per-row check
finds that same defect in 20 seconds with a 3,400-fold margin. Whether it becomes a gate is an open
decision.

## 2026-09-30. M2 and stage 4: no projection dense on Metal, the served CUDA kernel from torch

The 4B holds no projection dense on Metal any more. `tv_q4_metal_tiled` sits beside the served `tv_q4_metal` and gives
the same f32 values at tiles 256, 2,048 and 8,192 wherever the served kernel runs. The model gives the same 256 ids of
256, and the device allocation falls from 5.436 GB to 2.750, with all 252 projections resident in 1.158 GB
(*measured*, [hf-metal-m2-4b](mesures/hf-metal-m2-4b-2026-09-30.txt)). The four signed predictions held. One control
nearly went void: a mutation located by the kernel's name landed on both kernels, so the reference moved with the
subject. `checkq4guards.py` now closes the region at the next entry point.

On CUDA, `nvcc` compiles the served `llvq_tetra48.cuh` outside NVRTC and keeps its arithmetic: per-row 3.74e-04,
3.41e-04 and 3.53e-04 relative on three shapes against a 1e-2 bar, identical on two L4 instances, and 256 greedy ids
of 256 against `bin/run` (*measured*, [hf-cuda-4b](mesures/hf-cuda-4b-2026-09-30.txt)). The served header gained one
qualifier, `__device__` on `TETRA48_ORDER`, which changes no arithmetic. 168 of 252 projections are fused there,
because no torch op binds the int4 kernel. Seven launches cost $0.30, and two of the three build errors they found
are now held by tests on the Mac.

## 2026-10-01 to 10-02. The 4B published, and the package named

The out-of-tree route is complete and verifiable by a stranger. Two public repositories,
[Qwen3-4B-LLVQ-Tetra](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra) for the safetensors that
stay compressed and [Qwen3-4B-LLVQ-Tetra-sealed](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-sealed)
for the single file the Rust engine reads. **The Hub computed the sealed file's SHA-256 itself and it
is paper 2's**, `886391a8c03f66dc`: until 2026-10-02 that digest named a file on one laptop, so nobody
outside could check it. The published repository loads unauthenticated from an empty cache, 252
records replaced, no key missing or unexpected (*measured*,
[hf-cleanroom-4b](mesures/hf-cleanroom-4b-2026-10-01.txt)).

**Three defects found by asking what a stranger gets, not by reading the code.** `import llvqhf`
registered nothing, and failed soft: `transformers` warned "Unknown quantization type, got llvq ...
we will skip the quantization", loaded the file as dense and raised fifty lines later about a
corrupted checkpoint. Four stages of measurement had passed over it, because every script of the
package imported `.quantizer` by hand. `hfpack` writes a 64 byte `tokenizer_config.json`, a stub,
where the real one is 9,732 bytes and carries the chat template, so the repository as packed would
have shipped a tokenizer where `apply_chat_template` fails. And the wheel carried no CUDA source at
all: `_extension_cuda` looked for the repository two directories up, which under `site-packages` is
nonsense, so a pip user had no CUDA arm.

**A loadable fixture, and the fixture is the lesson.** No test in either language had ever called
`from_pretrained`; `fixtures/tiny` describes nothing on purpose and the real object is 1.4 GB.
`fixtures/mini` is a coherent one-layer Qwen3 of 148 KB whose every dimension is forced: 136 = 17 x 8
so the rotation's odd part is real where a power of two gives the trivial 1 by 1, two different
non-empty tails at 16 and 8, int4 only where `d_in` is a whole number of groups. The first version of
its test left `import llvqhf` out and, run alone, **passed its forward pass on random numbers** with
three tests of five green. A forward pass that runs proves nothing; the gate is `missing_keys` and
`unexpected_keys`.

**The package is `llvq-tetra` from 2026-10-02**, directory and module and distribution name, against
`llvq-hf` and `llvqhf` before. Preregistrations, journals and archived documents keep the old paths:
they are dated records and rule 2 forbids editing a stamped file. The six CUDA sources now ship
beside the two Metal shaders, in the repository's two-level shape so that `tv_tetra48_h.cu`'s own
relative includes resolve, and `the_shipped_cuda_closure_is_complete` recomputes the include closure
from the bytes rather than trusting the hand written list. Stage 4 had passed on a card for $0.30
over seven launches, five of which bought three build errors and one a queue of 3 h 47.

## 2026-10-03 and 04. A published model that loads as random, and a conversation

A caller that imports `transformers` alone gets the published 4B as a randomly initialized model, with no exception:
254 missing keys reinitialized at Qwen3's own std of 0.0200, 1,119 unexpected, and logits in an ordinary range
(*measured*, [hf-tripwire](mesures/hf-tripwire-2026-10-03.txt)). `transformers` registers quantization methods by
import, never by entry point. The next step planned was a quality figure through `lm_eval`, which does not import our
package, so it would have scored noise. `auto_map` cannot close the trap while `model_type` is `qwen3`. The operator
decided on 2026-10-03 to add it and keep the type: `trust_remote_code=True` now loads the model, verified from the Hub
with an empty cache, and a caller who passes nothing still gets the random model, as the card says.

`llvq_tetra.chat` talks to the 4B. On two prompts beside the FP16 checkpoint, greedy, FP16 answered visibly better
(an impression, [hf-chat-4b](mesures/hf-chat-4b-2026-10-04.txt)). The dense arm loads in 151.6 s against 4.5 for
FP16, because it rebuilds 252 matrices before the first token; the fused arm loads in 7.7 s and generates more slowly.
The upstream issue for `transformers` was drafted on 2026-10-04 and not posted
([`upstream/transformers-llvq-tetra/`](upstream/transformers-llvq-tetra/)).

## 2026-10-04 to 10-06. Review campaign, paper 2 version 2, release v0.0.2

An automated review of paper 2 contested no number. Answering it in full was priced at about $285. The operator
funded $9, and the campaign spent $4.57 (*measured*, `docs/data/jobs.csv`).

The three sealed files got a perplexity, 12.58, 9.71 and 8.44, all read on the L40S. The 4B FP16 and AWQ references
had been labelled as Apple-GPU readings; both were L40S readings of 2026-08-06, so no ratio crosses devices. The last
build step raises perplexity at 4B and 8B while it gains MMLU. On a quantized 8B file, Metal and the L40S agree to
0.01 % (*measured*, [ppl-bases-carte](mesures/ppl-bases-carte-2026-10-05.txt)).

The kernel bench moved from 216 to 252 matrices for `Tetra`, on the bare encode. Its time went from 3.424 to 3.734 ms,
its rate from 2.148 to 2.150 b/weight, and its tile gain from 128 to 64 on sm_89 from +16.1 % to +17.6 % (`Planes14`:
1.5 % to 1.1 %). A mixed int4 arm was written for the purpose, then dropped on the operator's instruction.

On an A100, `Tetra` beats our FP16 kernel 1.11× but not cuBLAS 1.14×, and the served files decode below FP16 in vLLM
(*measured*, [a100](mesures/a100-2026-10-06.txt)). The prediction that `Tetra` would lose to FP16 there was wrong
against our own FP16 kernel.

The 8B and 14B files went public on the Hub beside the 4B. The 4B card had said the two were deleted in a disk
cleanup, but they had sat in the job bucket all along. A first publish of the sm80 Space uploaded a local virtualenv,
19,618 files, and the Hub's secret scanner flagged numpy's own test names in it. The folder was removed, and `publish`
now refuses a perimeter over 1,000 files.

A second number audit of the paper found seven errors, none in a table
([paper2-audit](mesures/paper2-audit-2026-10-06.txt)), before a redundancy pass and a style pass. Merged into `main`
as `344499d` and released as `v0.0.2`, with the paper as a PDF.

## 2026-10-07. The transformers route comes to `main`

The operator decided to merge `hf-safetensors` into `main`, and `llvq-tetra/` stays a package of this repository. That
reverses the decision of 2026-09-30 to keep the branch off `main` and extract the package. Three facts of 2026-10-02
made extraction a source of drift: the wheel ships a copy of the served kernels made by `bin/tetratables`, the `mini`
fixture is written by a Rust test, and `hfpack`, which makes the published object, cannot leave the workspace. Only the
glue would go to `transformers` in tree, the way `aqlm`, `vptq`, `spqr` and `higgs` keep their own packages.

The registry gains the six stage 4 rows `main` lacked, $0.30: $246.75 over 227 priced jobs. The safetensors card moves
to `docs/hub/qwen3-4b-safetensors.md`, byte for byte the Hub's README. The upstream issue and its PR go out together,
once PyPI holds 0.1.0.

Five defects of the package were fixed before the first upload, because a version on PyPI cannot be changed. The
import registered "llvq" unconditionally while `transformers` raises on a name it already holds, so 0.1.0 would have
failed to import on the first `transformers` that carries the method in tree. The import guard swallowed any
`ImportError`, so a `transformers` too old for the package, one without `transformers.core_model_loading`, left the
method unregistered and `from_pretrained` loading a random model in silence. `torch` and `transformers` were an optional
extra, while the card's install line is `pip install llvq-tetra` alone. And the floors, `torch>=2.5` and
`transformers>=5.0`, admitted torch 2.5.1, which breaks against transformers 5.x; they are now the versions measured,
2.14 and 5.17. Two tests were added, 48 pass, and three mutants, one per decision of the guard, were each caught by
exactly one test. The fifth came from `twine check --strict` on the built wheel: two project URL labels held a comma,
which core metadata reads as the separator between label and URL.

The full suite ran twice. The first pass stopped at the first failing binary, so the second ran without fail-fast:
180 suites, 1,036 tests passed and 6 failed (*measured*). Three read the August archive `~/qwen3-4b-llvq.bin`, absent
from this machine: `real_rows_match_the_sealed_q8_file`, `transcode_of_the_sealed_model_matches_planes14` and
`rotation_keys_partition_the_sites`. The operator deleted them. Three CUDA rotation tests raced: four tests compiled
the same host driver into one fixed path while another one ran it. The driver is now built once per test binary, and
`golay70_decoder_matches_rust.rs`, which had the same pattern with two callers, got the same fix. The fix passed five
runs at 16 threads. The old behaviour also passed five runs on an idle machine, so the race cannot be forced here.

`llvq-tetra` 0.1.0 went to TestPyPI, the token read from a local `.env` that git ignores. TestPyPI serves the bytes
that were built, and installed from it into an empty environment, with nothing compiled, the package loads the
published 4B and gives the 256 ids of `bin/run` (*measured*, [hf-testpypi](mesures/hf-testpypi-4b-2026-10-07.txt)).
The check found a version the prereg had not seen: transformers 5.19.0, out the evening before, is what
`pip install llvq-tetra` resolves. It ran as a second arm, written beside the prereg before any load, and gave the
same 256 ids, so 0.1.0 needs no upper bound.

On the operator's go, the same bytes went to PyPI. PyPI serves the digests TestPyPI served, and
`pip install llvq-tetra` in an empty environment pulls torch 2.14.1 and transformers 5.19.0 and loads the fixture with
no missing or unexpected key (*measured*, [hf-pypi](mesures/hf-pypi-0.1.0-2026-10-07.txt)). The install line of
both Hub cards, and of the shim's `ImportError`, had promised that package since 2026-10-02: it is true now.

## 2026-10-30 to 11-02. The decode step trimmed, attention ungrouped from `repeat_kv`, and a field to serve it

Three jobs on the L40S, $1.55, all on the sealed 4B and its config. They answer the two cheapest questions a throughput
audit of 2026-10-29 raised: what the served step wastes on work it does not need, and what the KV cache costs at the
context the product promises.

**The trims.** Three removals, each bit for bit, give the 4B 122.4 tok/s [122.3-122.4] against 114.2 [113.6-115.2],
the path every published figure ran: the causal mask is not rebuilt when it is all zero in decode, `tv_q4_h` reads f16
where 84 conversions to f32 preceded it, and cudarc's event tracking is off on a stream that is alone (*measured*,
[trims-contexts](mesures/trims-contexts-2026-10-30.txt)). Singly they read 1.0283, 1.0230 and 1.0115; together 1.0716,
against 1.064 for their product. The same job measured the served decode at four prompt lengths for the first time:
122.5, 82.7, 61.4 and 33.2 tok/s at 5, 2,048, 4,096 and 8,192, with the first token at 0.03, 8.15, 16.91 and 34.86 s.
At 8,192 the cache takes 72.9 % of a decode token, which is what designated it as the next lever.

**Attention over the cached heads.** The shipped path concatenates the history every step and then copies each key and
value head to its four query heads, twice: `repeat_kv` in candle 0.9.2 is a `cat` of four copies. Two forms that copy
nothing were measured against it on a `Grow(256)` store, which grows by whole chunks and is written in place. The
grouped form, which stacks the query heads of one key head as rows of one matmul, reads 134.4 tok/s at a 5-token prompt
and 77.5 at 8,192, against 121.9 and 33.1, and returns the same 256 tokens at all four lengths (*measured*,
[gqa-ab](mesures/gqa-ab-2026-11-01.txt)). The shared form, which keeps the expanded matmul's shape, is slower
everywhere and was dropped.

**Neither unexpanded form is bit for bit on the card, and that changed a gate.** The shared form was bit for bit on the
Mac's CPU and is not on the L40S: cuBLAS rounds a batch of 8 key heads differently from 32 expanded ones, up to
6.6e-2 on a logit. Per-head shapes do not fix it. Both forms therefore fell under decision T8 of the audit, and the
operator took it as "tolerance plus control": a bounded tolerance replaces identity at the bit, provided a quality
control passes first. The control ran on WikiText-2 through the served kernel, 12 windows of 4,096, paired window by
window: +0.0000 % by prefill, bit for bit, and +0.0095 % [-0.0048 ; +0.0238] by decode, both inside the 0.12 % band of
[`ETAT.md`](ETAT.md) §4, with 99.788 % top-1 agreement over 49,140 steps (*measured*,
[gqa-ppl](mesures/gqa-ppl-2026-11-01.txt)). One prediction missed and is reported beside the decision, not inside it:
a few isolated decode steps move one logit by up to 13.13, where [0.04, 0.5] was predicted. Nothing explains it yet,
and the comparison that would read it, the published path against itself across its two paths, has not been run.

**Two traps.** The first grouped prefill failed on the card and not on the Mac, for $0.09: the first append to a `Cat`
cache returns the value projection as a transposed view, which candle's CUDA matmul refuses and its CPU gemm accepts
(`rows_contiguous`, commit `2f8bdd3`). A green test on a Mac proves nothing about a layout on a card. The second is
older and cost nothing this time: any new `KvStore` variant breaks the exhaustive matches of `fusedrun`'s A2 modes, and
the CUDA half of `llvq-llm` has no compiler on this machine, so the image build is where it shows, 17 minutes later.

**How it is served.** The operator chose a field of the served config over an engine default, on 2026-11-02. An
optional `attention` field names the form; absent means `expanded`, so the six configs that predate it keep their
logits and their digests, and `qwen3-4b-tetra-e4.json` still reads `098fe49c`, the digest the jobs check. A seventh
file, `configs/qwen3-4b-tetra-e4-grouped.json`, is the same object with `"attention": "grouped"`. The store is read
from the form and never written beside it, so no config can name a pair no measurement ran. `shared` is refused by
name. `grouped` is refused by name on Metal, where it has never been measured. `Served::apply` is the one place the
pair reaches a model, and the three served doors call it; `bin/gsm8k` applied `LLVQ_KV_PREALLOC` to its served arm
until then and now refuses that variable beside `LLVQ_CONFIG`.

**The door, measured.** A fifth job, $0.54 against $0.87 announced, ran the served door under each config and then
the context sweep beside each. The grouped config reads 133.3 tok/s against 121.5 after a 5-token prompt, and by the
sweep 134.5, 113.8, 98.0 and 77.5 against 122.5, 82.7, 61.4 and 33.2 at 5, 2,048, 4,096 and 8,192 (*measured*,
[served-grouped](mesures/served-grouped-2026-11-02.txt)). The two 256-token texts are the same bytes, so the grouped
config inherits the reference line `8784ffc4` and gate 4 does not change. The expanded arm reproduced the sweep of
2026-10-30 to every printed digit, in another job and another image, which is what says the served door and the bench
door measure one object. All fourteen predictions landed. The prefill is untouched, 35.02 s to the first token at
8,192 against 34.84, and it is now the largest single item of a long-context answer.

**The 4B now serves the grouped configuration**, decided the same day, and the publication perimeter was split to
allow it. Paper 2 cites three Hub repositories by URL and three file digests, and nothing else: those three cards and
the safetensors one are frozen, and a new repository,
[Qwen3-4B-LLVQ-Tetra-served](https://huggingface.co/Pier-Jean/Qwen3-4B-LLVQ-Tetra-served), carries the same bytes
under a card that is rewritten whenever a measurement replaces a figure. It went up the same day, commit `159afed4`,
seven files: the two configurations, the sealed file, and the `LICENSE`, `.gitattributes` and `SHA256SUMS` of the
cited repository so the structure matches. The Hub reports the file's digest as `886391a8`, the paper's, and the four
frozen repositories were read back at their prior commits afterwards. Only 267 MB crossed the wire: the Hub already
stored the object.

The switch took on a debt, written into ETAT §2 beside the number: the 4B's 133.3 tok/s is the grouped form, while
its 63.37 MMLU and 82.49 GSM8K are the expanded form's, and no task score has ever run under the grouped one. Its
whole licence is the paired perplexity inside ±0.12 %. Settling it costs about $2 for GSM8K through the kernel and
about $20 for an MMLU census.

## 2026-11-04: the 32B shared-memory wall, lifted in code

`docs/format-noyau.md` §8 said two ways out of the `rot_apply` wall were named and neither designed: an f16 staging of
the rotation, or a split into two kernels. Neither was taken. The rotation now stages in device memory where the
card refuses its shared staging (`rot_apply_dev`), and the int4 matvec stages its activation as f16 bits there
(`tv_q4_hs`). Both are byte-identical to the kernels they replace on the host at every width up to 31,744, and both
are selected only where those kernels are refused, so no served object moves. The f16 staging of the rotation was
dropped because it rounds every stage. No card has run either kernel yet; the identity job is drafted, not stamped.

2026-11-04, later the same day: lot 1 ran both kernels on the L40S. Each returned the host's bytes at every width
checked, 25,600 included, and `rot_apply_dev` took 45.43 µs at 25,600 (*measured*,
[lot1](mesures/lot1-4b-2026-11-04.txt), prereg `proofs/preregistration-bigmodel-2-2026-11-04.md`). The 32B is still
never encoded, and nothing has served it end to end.

## 2026-11-04. The grouped attention at 8B and 14B, and its GSM8K at three sizes

The grouped form serves all three sizes since 2026-11-04, and its GSM8K matches the expanded form's at each.

**GSM8K at 4B.** The grouped config scores 81.96 % (1,081 of 1,319) against 82.49 % for the expanded one, -0.53 pp
[-1.07 ; +0.00], McNemar p = 0.092, 10 discordant problems against it and 3 for it (*measured*,
[gsm8k-grouped](mesures/gsm8k-grouped-2026-11-04.txt)). Rule G1 holds, so the GSM8K half of the 2026-11-02 debt is
settled. The interval touches zero and leans against the grouped form.

**8B and 14B.** The grouped form keeps paired WikiText-2 perplexity inside ±0.12 % on both sealed files. By decode it
reads -0.0048 % [-0.0117 ; +0.0021] at 8B and +0.0023 % [-0.0062 ; +0.0109] at 14B; by prefill it is bit for bit
(*measured*, [gqa-8b](mesures/gqa-8b-2026-11-04.txt), [gqa-14b](mesures/gqa-14b-2026-11-04.txt)). At 8,192 it decodes
×2.156 and ×2.140 faster. A light MMLU of 285 questions picks the same answer on every question under both configs:
65.55 % at 8B, 74.13 % at 14B (*measured*, [served-grouped-8b](mesures/served-grouped-8b-2026-11-04.txt),
[served-grouped-14b](mesures/served-grouped-14b-2026-11-04.txt)). The operator granted the T8 at both sizes the same
day. The served doors read 105.8 and 62.5 tok/s against 98.3 and 58.6 for the expanded configs, with the same
256-token texts. They replace the 95.0 and 57.2 tok/s of 2026-09-25, measured before the trims.

**GSM8K at 8B and 14B**, run in lot 1 that evening: 88.55 % against 88.63 %, -0.08 pp [-0.61 ; +0.46], and 91.96 %
against 92.04 %, -0.08 pp [-0.22 ; +0.07], both p = 1.000 (*measured*,
[gsm8k-grouped-8b](mesures/gsm8k-grouped-8b-2026-11-04.txt),
[gsm8k-grouped-14b](mesures/gsm8k-grouped-14b-2026-11-04.txt)).

## 2026-11-04 and 05. A split-KV attention kernel, served from position 1,280

On the morning of 2026-11-05 the served 4B took its own decode-attention kernel from position 1,280, with the grouped
arithmetic below. 4E replaced that config the same afternoon.

**First readings.** The kernel returns the CPU replay's bits on every call checked. It reads the cache at η 0.406 of
864 GB/s at 8,192, where FlashInfer's decode kernel reads 0.810 on the same card (*measured*,
[attn-bench](mesures/attn-bench-2026-11-04.txt), [fi-decode-anchor](mesures/fi-decode-anchor-2026-11-04.txt)). In
the served 4B it gave 95.8 tok/s against 78.0 at 8,192 and ×0.93 after a 5-token prompt. A revised fold raised η to
0.545 and the 8,192 reading to 104.2 against 77.7, and left the short loss at ×0.9408 (*measured*,
[lot1](mesures/lot1-4b-2026-11-04.txt)). Its T8 control did not run: its condition at 5 tokens failed.

**The threshold.** Rows below position 1,280 now take the grouped arithmetic, bit for bit. The kernel config reads
134.7 against 135.0 tok/s after 5 tokens with the same tokens, and 104.0 against 77.5 at 8,192, ×1.3419 (*measured*,
[attn-t8-r2](mesures/attn-t8-r2-2026-11-04.txt)). Paired perplexity on the positions the kernel takes reads -0.0013 %
[-0.0074 ; +0.0047] at 4,096 and +0.0001 % [-0.0076 ; +0.0077] at 8,192. The operator granted that T8 on 2026-11-05,
and `configs/qwen3-4b-tetra-e4-kernel-from1280.json` became the 4B's served config that morning. Its door read 133.7
tok/s (*measured*, [served-4e](mesures/served-4e-2026-11-05.txt)).

**Below the threshold.** With every decode row on the kernel, paired perplexity reads +0.0318 % [-0.0151 ; +0.0786] on
32 windows of 1,280 and +0.0018 % [-0.0053 ; +0.0088] on 12 windows of 4,096 (*measured*,
[attn-t8-below-4b](mesures/attn-t8-below-4b-2026-11-05.txt)). The point at 1,280 missed its prediction. The operator
extended the T8 below 1,280 at 4B only, the licence 4E needs for a short prompt. No task score exists past 1,280,
because GSM8K never reaches that position.

**Beside it.** Lot 1's token-boundary gestures read ×1.0120 [1.0082-1.0128] together with the same tokens, and none
is served under its prereg's rules (lot1). `tv_tetra48_seg` returned the separate launches' bits on 72 of 72 groups
on the card ([seg-tetra](mesures/seg-tetra-2026-11-04.txt)). One more row on the batched path costs 3.93 ms at position
0 ([tk-sweep](mesures/tk-sweep-4b-2026-11-04.txt)).

## 2026-11-04 and 05. The persistent kernel 4E serves the 4B at 178.0 tok/s

The 4B decodes one token a cooperative launch since 2026-11-05. Its door reads 178.0 tok/s, against 133.3 for the
grouped config of 2026-11-02 (*measured*, [served-v4](mesures/served-v4-2026-11-05.txt),
[served-grouped](mesures/served-grouped-2026-11-02.txt)). The operator's target was 300.

**Route P.** One launch over the 4B's 252 matrices is bit-exact and takes 2.394 ms, against 3.009 for the 144 fused
launches, streaming 410 GB/s (*measured*, [pers-l1](mesures/pers-l1-2026-11-04.txt)). Rule H projected 222 to 258
tok/s for 4E from that rate (*estimated*). The rate came from a bench at 32 warps a SM, and 4E runs 16
([revue-debit](archive/revue-debit-2026-11-05.md) §6).

**First 4E readings.** The first was void: a timing round decoded different tokens
([pers-4e](mesures/pers-4e-2026-11-05.txt)). A stale argmax key was the cause (`3e186a0`). The retry decoded the same
tokens at ×0.993 of the eager path ([pers-4e-r2](mesures/pers-4e-r2-2026-11-05.txt)). Phase stamps then showed 4E's
own attention phase taking 2,355 µs of a short token and 72 % of a token at 8,192
([pers-4e-r3](mesures/pers-4e-r3-2026-11-05.txt)).

**v2 and v3.** v2 put attention on the split-KV kernel's mapping and prefetched in the row loop. It reads 178.7
against 132.1 tok/s at 5 tokens, ×1.353, and 131.8 against 103.5 at 8,192, ×1.273, same tokens (*measured*,
[pers-4e-v2](mesures/pers-4e-v2-2026-11-05.txt)). v3 read 179.2 and 135.4
([pers-4e-v3](mesures/pers-4e-v3-2026-11-05.txt)). Alone, v2's row loop streams 379 GB/s; inside 4E the same rows move
about 260 (*measured* and *computed*, [pers-rowloop](mesures/pers-rowloop-2026-11-05.txt)). The served door under the
v3 config read 169.0 tok/s against 133.7 for the served kernel config, with the same tokens at every length and 40 of
50 GSM8K pilot problems right under both (*measured*, [served-4e](mesures/served-4e-2026-11-05.txt)).

**The shaker, and v4 served.** One job timed v3, v4 and v5 with inner stamps. v4 over v3 reads ×1.060 [1.059-1.062]
at 5 tokens, 189.8 tok/s in the measurement form. v5 adds ×1.001 and stays a measurement, refused by name in a config
(*measured*, [shaker-4e](mesures/shaker-4e-2026-11-05.txt)). The operator served v4 on 2026-11-05 through
`configs/qwen3-4b-tetra-e4-4e-v4.json`. Its door reads 178.0 [178.0-178.3] tok/s in 1.38 GB. Its sweep reads 180.8,
169.5, 162.1 and 139.5 at 5, 2,048, 4,096 and 8,192, with the first token at 35.04 s at 8,192 (*measured*,
[served-v4](mesures/served-v4-2026-11-05.txt)). Its text equals v3's everywhere, and every prediction landed. No full
task census has run under 4E, and the 8B and 14B have never run it.

**v6.** Folding ln1, ln2 and the MLP down input into one round trip a thread gives 212.6 tok/s against 190.2 for v4,
×1.118 [1.117-1.118], and 152.0 against 138.9 at 8,192, same tokens (*measured*,
[pers-4e-v6](mesures/pers-4e-v6-2026-11-05.txt)). The down input fell from 796.0 to 254.6 µs a token and missed its
200 µs gate, so by its prereg v6 is not proposed.

The card work of 2026-11-04 and 05 took 25 jobs and $21.27. The registry reads $273.30 over 262 priced rows of 267
(*computed*, `docs/data/jobs.csv`).

## 2026-11-05 and 06. The Metal night: the sealed 4B fused on the Mac, then MLX

The sealed 4B decodes fused on the M3 Max at 52.07 tok/s with its one retained row-loop port, against 7.81 for the
dense reconstruction, with the same 256 tokens (*measured*, M2 of [metal-nuit](mesures/metal-nuit-2026-11-06.txt)).
That ratio is raw: the dense arm copies its f16 head table every token, and Metal has no same-head arm. Everything ran
locally at $0, and nothing is served on Metal.

**The int4 wall.** The fused Metal path was refused at load, because 12 int4 `down_proj` records exceed d_in 8,192.
They now route to a tiled int4 kernel at 2,048 columns a tile, and the served Metal kernels keep their bytes. All 27
mutants of the new kernels and host code were killed before the stamp.

**Attention forms.** On Metal the grouped form returned the expanded form's logits bit for bit on 49,140 decode steps
over 12 windows of 4,096 (*measured*, M5). It decodes ×1.750 faster at 2,048 and ×2.380 at 4,096 (M4). The Metal port
of the split-KV kernel reads ×0.864 after 5 tokens and a wash at 2,048 and 4,096, with paired perplexity +0.0051 % in
its T8 (M6). The Metal door still refuses both forms. Two NOT SERVED configs wait for the operator.

**Prefill.** Two prefill kernels cut the first token of a 4,096-token prompt from 59.73 s to 37.24 s, then from 34.56 s
to 31.53 s, with the same tokens (*measured*, M9 and M10). The two steps chain to about ×1.74 (*computed*). A decode
token is GPU-bound: the host takes 5.43 ms of 15.18 (M12).

**MLX.** mlx-lm's Qwen3-4B 4-bit decodes 123.27 tok/s against our 65.49 after a 5-token prompt (M14). The operator
chose MLX as the Mac's engine on 2026-11-06. A prototype runs our Tetra kernel inside mlx-lm's Qwen3. It decodes 93.92
tok/s after 5 tokens and 70.26 at 4,096, and reads a 4,096-token prompt in 4.11 s at a 3.20 GB peak (*measured*, M16).
mlx-lm reads 129.45, 98.38 and 3.96 s at 3.67 GB, at about 4.5 b/param against our 2.7. The MLX path is not bit-exact
with the Rust engine. On the 5-token house prompt its 256 tokens differ from the Rust engine's in a degenerate
repetition (M15). No perplexity has been read through it.

## 2026-11-07. 4E v6 written as a config for the 4B

Until 2026-11-07 the living state read 4E v6 as a measurement against its missed gate. Its lever left the MLP down
input at 254.6 µs against a prereg gate of 200, so by that prereg's rule v6 was not proposed for serving
([pers-4e-v6](mesures/pers-4e-v6-2026-11-05.txt)). On 2026-11-07 the operator gave the go to implement v6 on the 4B
regardless. `configs/qwen3-4b-tetra-e4-4e-v6.json` is the v4 config with `"rows": "v6"`, and `served.rs` accepts it.
v4 stays served until the door of `ops/jobs/served-v6.sh` reads.

The door read the same day for $0.41 (job 6aeeea70, 825 s of card). It gives 197.2 tok/s against 178.0 for v4, and
150.9 against 139.5 at 8,192, with v4's tokens at the door and every length and v4's 50 pilot completions
([served-v6](mesures/served-v6-2026-11-07.txt)). Every prediction held. Serving it is the operator's call.

## 2026-11-07. plan-300's step 1: no row lever

`docs/plan-300.md` was written on 2026-11-07 with 300 tok/s at the door reachable at the bit on its high bound, 306 to
322 tok/s, if a row lever existed. Step 1 ran the same day on the operator's go (job 6aef03fa, $0.25). No arm on the
Tetra row loop reached the ×1.15 bar; the best, a 128-bit tile layout, read ×1.002 over v2. By the prereg's rule B,
300 at the bit is out on every bound the plan anchors. The int4 tile layout read ×1.338 over 4E's int4 row and is a
lever; no counter beat the grid barrier by 0.3 µs ([step1-micro](mesures/step1-micro-2026-11-07.txt)). 4E v7 (K2) and
v8 (K3a) are coded and bit-exact on the host, not run.

## 2026-11-07. plan-300's step 2: v8 runs, v7 does not load

Step 2 ran the same day on the operator's go (job 6aef3f69, $0.22). v8, v7's staged constants plus a three-stage
cp.async ring in attention, kept the eager tokens and read ×1.076 over v6 at 5 prompt tokens and ×1.025 at 8,192,
227.9 against 211.8 tok/s in the measurement form. v7 alone did not install at 512 threads under the card's ptxas.
K2's gate, read on v8, held: the down input fell from 255 to 150 µs. K3a's gate failed: attention's slowest CTA over
mean reads 4.40, and the job spread is what remains. v8 stays a measurement
([step2-4e](mesures/step2-4e-2026-11-07.txt)).

## 2026-11-07. plan-300's step 3: the int4 tiles hold in situ, the route stops short of 300 at the bit

Step 3 ran the same day on the operator's go (job 6aef51d3, $0.28). v9, v8 with the int4 rows on step 1's tiled
words, kept the eager tokens and read ×1.035 over v8 and ×1.118 over v6 at 5 prompt tokens, 238.6 tok/s in the
measurement form, about 220 at the door (estimated). v6 under v8's shared memory read ×1.000: v8's slower rows on its
stamps were its stamp symbol's spill. The leads left at the bit give about 235 to 258 tok/s at the door (estimated);
300 needs speculation or fewer bytes ([step3-4e](mesures/step3-4e-2026-11-07.txt)).

## 2026-11-07. 4E v9 with one copy of the int4 bytes, the 8B and the 14B named out

The operator asked for v9 on main at 4B, 8B and 14B, with one copy of the int4 bytes. Step 3's measurement form had
held them twice: the served words for the eager path, a tiled copy for 4E, 0.385 GB more on the 4B. Under a config
whose rows are `v9` the loader now tiles the words on the host and uploads them once. The eager path reads them
through `tv_q4_ht` / `tv_q4_hht`, which keep every product and its order; only the addresses move. On the host they
return the served readers' bytes at every int4 width of the three sealed files, and 9 mutants of the tiled reader die.
Each load checks the tiled reader on the card against the served one on a file-order copy, then frees the copy. The
file and its b/param do not move. `configs/qwen3-4b-tetra-e4-4e-v9.json` is written; its door job,
`ops/jobs/served-v9.sh`, is prepared and not run.

4E did not hold on the other two sealed files at first. The 8B holds `o_proj` in Tetra and an untied head; the 14B fits
v9's shared memory (96,064 B of 101,376) and its int4 down rows, but its head is untied too. The operator asked the
same evening for all three sizes. 4E gained the head's own pointers and a Tetra o phase, the MLP's Tetra down at the
context's width; on the host the token loop with both equals the served chain for v4, v6, v8 and v9. The 8B and 14B
v9 configs and their doors, which also read the kernel form's perplexity band against the grouped form, are ready. While wiring this, a test of the v6 branch named `llvq_cuda` without the macOS-or-cuda gate
of pjmalandrino/llvq#12; it would have broken the Linux CI at the merge, and it carries the gate now.
`ops/check-linux-cuda.sh` type-checks the cuda half of `llvq-llm` on the Mac, which no tool did before.

## 2026-11-07. v9 served at 4B, 8B and 14B

The three doors ran the same evening on one image, for $2.65 (jobs 6aef7864, 6aef7ae2, 6aef7ae4). Every gate held:
4E returned the eager path's tokens at every length at each size, the 4B kept the v6 config's text and held the int4
words once (1.980 GB under v9 and under v6), and the kernel form from position 0 stayed inside the perplexity band
against the grouped form at 8B and 14B. At the door v9 reads 221.5, 142.5 and 86.3 tok/s against 197.4 for v6 and
105.9 and 62.8 for the grouped configs, in the same jobs ([served-v9](mesures/served-v9-2026-11-07.txt),
[-8b](mesures/served-v9-8b-2026-11-07.txt), [-14b](mesures/served-v9-14b-2026-11-07.txt)). On the operator's message
the v9 configs became the served ones at all three sizes. Before, the 4B served v4 at 178.0 tok/s and the 8B and 14B
their grouped configs at 105.8 and 62.5.

## 2026-11-07. The MLX door, on the operator's ruling of a tolerance on Metal

The operator ruled open decision 1 of the Metal plan on 2026-11-07: « Sur métal, on a validé mlx donc implem le
aussi ». Until then the plan recorded no ruling on tolerance against bit for bit on Metal, and the M15 prereg had
written "served under the M6 tolerance" as if it were ruled.

The prototype became a door. `ops/mlx_tetra/serve.py` reads `configs/mlx/qwen3-4b-tetra-e4-mlx.json` and refuses by
name every value it does not carry, the Rust keys, any other object, another mlx or mlx-lm, and any `LLVQ_*`
variable. `check.py` gained a reference outside MLX, the `llvq-tetra` decoder that matches Rust bit for bit, and on
the mini fixture a whole forward pass against `transformers`. Before, it compared the MLX kernels only with each
other, so a defect in the shared shaders could pass it. A first mutation run killed 22 of 23 mutants (*measured*). The survivor
gave k the gains of q in the stacked dispatch, invisible on a fixture whose records share their centroids; the check
now scales k's gains apart, and the second run killed 23 of 23 (*measured*, [mlx-door](mesures/mlx-door-2026-11-07.txt)).

The T8 reading ran the same morning on the operator's go, 16.5 minutes, $0. The pooled perplexity read +0.0078 % by
decode and +0.0133 % by prefill against the Rust Metal path, but windows 6 and 7 fell outside +-0.12 % by both paths,
at +0.27 % and -0.13 % (*measured*, [mlx-t8](mesures/mlx-t8-2026-11-07.txt)). By its prereg the door is not served. The same two
windows move by about 0.2 % between two Rust engines, so the gate itself went back to the operator.

The operator chose to keep MLX under a gate on the paired interval, stamped before a reading of windows 13 to 24,
which the first reading had not seen. Against the Rust dense reconstruction in f16 on Metal, the door read
-0.0103 % [-0.0847 ; +0.0642] by decode and -0.0099 % [-0.0842 ; +0.0644] by prefill, inside +-0.12 %
(*measured*, [mlx-pooled](mesures/mlx-pooled-2026-11-07.txt)). The MLX door has served the 4B on the Mac since that reading.

Both readings ran `configs/mlx/qwen3-4b-tetra-e4-mlx.json` at sha256 `d7bd5b5c…`, whose `note`, printed at every
launch, named the per-window gate that failed. A review before the merge rewrote the note alone to name the paired
gate that passed: sha256 `8542b7f9…`, every key the door reads unchanged. The same review moved every argument check
of `serve.py` before the model loads, including the reference file of a reading, and added four refusals (26 of 26
mutants killed, *measured*,
[mlx-door](mesures/mlx-door-2026-11-07.txt)).

## 2026-11-06 and 07. The `llvq` runtime, and a faster CI

`llvq-runtime` came to main (pjmalandrino/llvq#10). The `llvq` binary pulls the three sealed files against pinned
digests and serves them with Ollama's and OpenAI's routes, an OpenAPI document and a chat page. Since
pjmalandrino/llvq#15 it loads the v9 configs on CUDA, greedy only; on Metal it runs the expanded config. CI runs four
jobs in parallel in about 8 minutes, tests under a `ci` profile without LTO (pjmalandrino/llvq#12, #13).

## 2026-11-07. The documentation made light

The four HTML courses (`cours-comprendre-llvq`, `cours-tetra`, `cours-layouts-runtime`, `architecture-c4`) were
replaced by one Markdown document, `docs/how-it-works.md`. Removed with them: the two 27B plans of 2026-09-28, never
adopted; `plan-debit-noyau.md`, whose open leads `plan-300.md` and `ROADMAP.md` now carry; `modele-erreur.md`
(2026-09-15), `inference-cost-reduction-2026.md` (July 2026) and `runbook-rownorms-2026-09-22.md`. `ETAT.md`,
`ROADMAP.md`, `METHODE.md`, `STYLE.md`, `README.md` and `CLAUDE.md` were rewritten shorter on the state of the day.
`AGENTS.md`, a stale copy of `CLAUDE.md`, now points to it.
On the operator's go, 31 entries of `docs/archive/` that nothing outside the archive cites were removed: hand-overs,
drafts, posts and plans of July to November, 33 files and 6,765 lines.

## 2026-11-07. The runtime audit: eight bugs fixed, the text work 350 times cheaper at 8k

The audit of `llvq-runtime` fixed eight bugs, each with a test that failed first, and killed 12 of 12 mutants
([runtime-audit-2026-11](runtime-audit-2026-11.md)). A busy model stopped or deleted stayed in memory out of the cap,
and a second copy could load beside it. A request abandoned in the queue cut the live conversation's cache. Two pulls
of one file could corrupt it, and `llvq pull --from` hard-linked a source that a re-seal could change after its check.
The engine decoded the whole answer at every token, 679 µs a token at 7,936 tokens; the incremental decode takes 1.9 µs
for the same text (*measured*, [runtime-overhead](mesures/runtime-overhead-2026-11-07.txt)).

## 2026-11-07. A CPU backend for the runtime, in place of the dense reconstruction

Asked the same day whether a laptop without a dedicated GPU runs the models. Until then such a machine took
`sealed::load` on the CPU, which rebuilds every projection as f16. Measured for the first time on the 4B: 0.71 tok/s
in a 10.6 GB peak, candle's f16 matvec on one core (*measured*, [cpu-tetra](mesures/cpu-tetra-2026-11-07.txt)).

`llvq-llm` gained `tetra_cpu.rs` and `fused_cpu.rs`. The Tetra word is decoded through three lookups in a 128 KiB
table built from `llvq_search::tetra`'s public objects, not from the GPU kernels, and the activation is put in trio
order once a call. NEON on aarch64, AVX2 + FMA on x86_64, scalar otherwise, rows spread over a rayon pool of
`LLVQ_THREADS`. The int4 records and the int4 embedding are read the same way. `served::load_resolved` sends
`Device::Cpu` there in every build, and the runtime's `Backend::Dense` became `Backend::Cpu`, with a `cpu` config
per entry of `library.json`. Its tests hold the matvec within 2e-6 of `decode_matrix`'s product; 10 of 10 mutants
killed.

At f16 the first reading moved one 4,096-token window by -0.47 % of perplexity against the dense f16, past every
gap between two Rust engines. The same window at f32 read -0.0017 % against the dense f32: the matvec was right, and
candle's CPU f16 arithmetic around it was not. The CPU path runs at f32, which also decodes faster there: 8.6 tok/s at
12 threads and 5.1 at 4, in 1.86 GB, under the bench lock the Mac's sessions share, and -0.0021 % [-0.0022 ; -0.0019] of the dense f32 over 12 windows
(*measured*). Nothing ran on x86, and Windows was never built.

## 2026-11-07. A batching branch: the batched decode of the 4B and its tensor-core kernel

The operator opened a batching axis on 2026-11-07. AWQ in vLLM was measured first at B = 1 to 32 on the L40S: 203.0
to 5,128.7 tok/s aggregate, 541.9 to 427.9 GB/s (*measured*, [batch-awq](mesures/batch-awq-2026-11-07.txt), $0.08).

The branch added a batched step (B sequences, one cache slot each, one row each a step), a batched attention that is
the served kernel's body with one sequence per grid z, and `tv_batch_mma.cu`: Tetra blocks and int4 nibbles decoded
once into shared memory as exact integers, multiplied on the tensor cores, the rotated activation in two f16 halves.
Host probes held the kernel to a double reference (16 of 16 mutants killed) and the attention to the one-row bits (5
of 5). One job read both arms ($0.48): the existing multi-row path plateaus at 231.9 tok/s, the tensor-core kernel
reaches 1,256.5 at B = 32, both with the batch-1 tokens on every row (*measured*,
[batch-tetra](mesures/batch-tetra-2026-11-07.txt)). The gate passed. The kernel's step costs 15.6 ms at B = 1 against
AWQ's 4.9, which leaves Tetra four times below AWQ in aggregate.

## 2026-11-07. Speculative decoding with the 0.6B: the bits hold, the speed does not

- Operator's go: test the speculative decode with the 0.6B. Qwen3-0.6B sealed in Tetra on the Mac: 0.252 GB,
  3.099 b/param, perplexity 40.30 against 18.93 for its f16 checkpoint ([draft-0p6b](mesures/draft-0p6b-2026-11-07.txt)).
  `seal` no longer carries a head the model ties to its embedding.
- `pers_4ev.cu`, 4E's verify pass: up to 4 rows of the decode in one launch, each row the served chain's bits on the
  host (11 mutants killed). `speculative.rs`: greedy acceptance, both caches rewound. `bin/specrun`, the gate.
- On the L40S ([spec-gate](mesures/spec-gate-2026-11-07.txt), $2.37 with a first run void at the drafter's load):
  every speculative arm made the greedy tokens at the three sizes, four prompts, k = 1 to 3, and on the 50 GSM8K
  pilot problems. Decode at the best k, x0.595 (4B), x0.629 (8B), x0.625 (14B) of greedy; +0.24 to +0.27 GB on the
  card. A 2-row verify pass costs 1.95 to 2.25 greedy tokens (it spills, runs the prologues and attention per row),
  the drafter 2.5 ms a launch. Not proposed for serving.
