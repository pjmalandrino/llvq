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

Four jobs on the L40S, one of them a failed first attempt, $1.55, all on the sealed 4B and its config. They answer the two cheapest questions a throughput
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
AWQ's 4.9 in vLLM, which leaves Tetra below AWQ in aggregate: 1,256.5 tok/s against 5,128.7 at B = 32, 54.2 GB/s
against 427.9 (*measured*, same journal).

## 2026-11-08. Batching for the L40S: a staged GEMM, B to 128, continuous batching in the runtime

The operator made batching the project's priority on 2026-11-08 ("régler le batch pour défoncer la L40S"). The first
kernel read each weight row's block from global memory one lane a row, with no prefetch. `tv_batch_gemm.cu` stages
the words of a K chunk in shared memory two chunks ahead, folds every scale into f16 weights, runs one MMA pass, takes
up to 128 rows, and launches q, k, v and gate, up as one launch each (17 of 17 mutants killed on the host). The
attention gained a slot-indexed entry, and `llvq serve` a pool of slots under `LLVQ_BATCH`: a request joins a free
slot and leaves without stopping the others.

Two jobs ($1.48 of a $5 envelope) read the gate and the bench at three sizes. The GEMM is x1.67 to x1.81 the first
kernel, keeps every door row equal to batch 1 and perplexity within +-0.0053 %, and scores 40/50 on the GSM8K pilot.
It stays about two times below AWQ in aggregate at every size and B: 3,996.7 against 10,146.2 tok/s on the 4B at
B = 128 (*measured*, [batch-l40s](mesures/batch-l40s-2026-11-08.txt)). The 4B step costs 9.2 ms at B = 1, about 5 ms
of it eager launches, and grows 0.2 ms a row against AWQ's 0.06.

Three more jobs the same day ($1.99; branch $3.47 of $5). The activation in f16 and 128-row tiles gained 12 to 41 % at
B = 128 ([batch-l40s-v2](mesures/batch-l40s-v2-2026-11-08.txt)). The operator then asked for a lone request on 4E and
for two CTAs a SM. The GEMM learned to read 4E v9's tiled int4 words, so the served v9 configs carry the batch: alone,
a request decodes at 224.4 / 142.4 / 88.1 tok/s on the 4B / 8B / 14B, against 106 / 74 / 44 on the GEMM
([batch-l40s-v3](mesures/batch-l40s-v3-2026-11-08.txt)). Two CTAs a SM lost 5 to 17 % and the tiled order about 1 %,
read apart in one job ([batch-l40s-v4](mesures/batch-l40s-v4-2026-11-08.txt)); the occupancy lead is closed.

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

## 2026-11-08. The llama.cpp duel: IQ2_S at our bits, three sizes, one L40S

- Prereg stamped before the first job; four deviations, each written before the run it touched: our side on the
  batching Space (D1), run 1 stopped at its first bench by a VRAM line in the JSON (D2), a variable that never reached
  the retry (D3), and a `llama-server` slot starved at its default batch, reproduced on the Mac (D4).
- llama.cpp's nearest rung to our bits is IQ2_S at all three sizes. Against it Tetra wins perplexity and MMLU at
  every size (+9.08, +4.13, +2.89 points) and GSM8K at 4B and 8B; the 14B GSM8K is not separated. IQ2_S decodes 33.5
  to 63.9 tok/s faster after 5 tokens, and wins peak memory everywhere but the 4B after 5 tokens. unsloth's
  UD-IQ2_XXS loops under greedy decoding at 4B (9.17 GSM8K). $17.55
  ([duel-iq2](mesures/duel-iq2-2026-11-08.txt)).

## 2026-11-08. The prefill's card memory at 8,192 tokens: three score tensors, cut to one bit for bit

- The duel against llama.cpp IQ2_S measured 11.48 / 12.66 / 17.09 GB of card memory after an 8,192-token prompt at
  4B / 8B / 14B, against 3.52 / 4.71 / 6.90. The excess was the grouped prefill keeping three f16 score tensors alive
  (6.44 GB at 4B, *computed*), and the stream-ordered pool keeping the first chunk's under the second (*estimated*).
  That model lands on the three measures ([pic-vram-8k](mesures/pic-vram-8k-2026-11-08.txt)).
- The served path now writes the masked softmax back in place, builds the causal mask in f16, syncs the stream
  before each prefill chunk and after each cache growth, and frees the attention's locals before the MLP. The same
  logits bit for bit on the host and on Metal at the real 4B shape, 56.65 to 14.79 GB there (*measured*). On the
  L40S ($2.04): 5.95 / 7.39 / 10.54 GB, inside the computed 5.65 / 7.09 / 10.08 +-0.5, and the logits of the in-place
  softmax equal the whole tensor's (*measured*).
- `"prefill_attention": "kernel"`, NOT SERVED, sends the prompt rows to the decode-attention kernel: no score tensor.
  On the card 4.77 / 7.15 / 9.30 GB against 3.67 / 5.20 / 7.81 computed, 1.14 to 1.21 times slower to the first
  token, its logits the same in chunks of 512 and of 4,096. Not proposed; the excess is not traced.

## 2026-11-08. What the format's maths can buy in throughput

A study at $0 ([`recherche-debit-maths.md`](recherche-debit-maths.md)). The eager Tetra loop's time is linear in its
SASS count a block on the four F1 floor arms, within 2.5 % (*computed*, [sass-tetra-int8](mesures/sass-tetra-int8-2026-11-08.txt)).
Rows at DRAM speed would cap the doors at 278, 196 and 121 tok/s (*estimated*). The int8 activation with `dp4a` takes
the block from 237 to 152 instructions (*computed*), projected 248, 168 and 103 tok/s on the arithmetic (*estimated*),
not at the bit. Two exact levers give 234, 151 and 90 at most. Nothing ran on a card.

## 2026-11-08. Second pass on the throughput study

The first pass gave in-situ Tetra rows the eager loop's instruction share, 64.3 %, which the stream floor forbids:
bounded by the bytes at 671 to 864 GB/s the share is 36 to 51 % at 4B ([sass-tetra-lean](mesures/sass-tetra-lean-2026-11-08.txt)).
The int8 lever drops from 248 / 168 / 103 to 235 / 154 / 94 tok/s at the central figure, and the decode-free ceilings
from 278 / 196 / 121 to 260 / 174 / 105 (*estimated*). New: the word relabeled at load with the Golay patterns by
linearity (237 to 207 instructions, exact), and an int16 dot through `dp2a` (145, rounding 2.0e-5). The best lever,
both with lane rotation, projects 239 / 157 / 95. Turyn, hexacode, Walsh-Hadamard, table-of-dots and sublattice
decodes are closed by computation.

## 2026-11-08. Breaking the throughput ceiling: the budget of a token

Section 11 of [`recherche-debit-maths.md`](recherche-debit-maths.md). At batch 1 the hard limit is the weight bytes,
628 / 359 / 188 tok/s at 864 GB/s; the rest of the 4B's 4.51 ms door is implementation. The single-token floor of
this engine is about 370 / 222 / 124 tok/s (*estimated*). Past it, only multi-token verification: with B1's integer
dot shared across rows and a one-layer drafter, 454 / 288 / 171 at α = 0.7 (*estimated*), gated on a 2-row verify
under 1.15 greedy tokens (spec-gate measured 1.95). The sealed 4B's Tetra words carry 47.4 bits of entropy of 48
(*computed*, `bin/tetraentropy`): lossless coding above Tetra is closed, and so is self-speculation from the lattice.

## 2026-11-08. The integer-dot verify rows on the card: between the gates

[spec-micro](mesures/spec-micro-2026-11-08.txt), $0.48 over two runs, the first nul on a wiring panic. An extra verify
row with B1's integer dot costs 0.187 of 4E's Tetra row, against 0.381 for a second f32 row (*measured*): BETWEEN the
prereg's 0.15 and 0.26, about 0.34 ms in situ at 4B (*estimated*). The integer dot keeps 91.6 % of the rows identical
to the served ones, the rest one binary16 ulp away. Alone, none of B1's levers clears its bar: i16 x1.143 for x1.15,
lean x1.020, rotation x0.856. Projected doors with a one-layer EAGLE-3 head at α 0.6: 319 / 198 / 121 tok/s; with
prompt lookup at α 0.3 to 0.5: 251-300 / 159-187 / 96-113 (*estimated*, `plan-speculation.md`). Three Apache-2.0
EAGLE-3 heads exist for Qwen3 4B, 8B and 14B, +0.24 / 0.22 / 0.19 b/param in int4 (*computed*). The route goes on only
on the operator's call.

## 2026-11-08. Prompt lookup on the card: the copy-heavy prompts only

[spec-lookup](mesures/spec-lookup-2026-11-08.txt), the 4B alone under the operator's $0.75 cap, $0.49. Prompt lookup
keeps the greedy tokens on six prompts and 50 GSM8K problems. Tokens a verify at k = 3: 2.56 on a code edit and 3.04 on
a retrieval answer (α 0.71 and 0.82), 1.18 to 1.51 on chat, code, summary and GSM8K (α 0.15 to 0.35) (*measured*,
α *computed*). On today's verify the decode stays under greedy except on retrieval (x1.012). With the integer-dot
verify the copy-heavy prompts would reach 395 and 469 tok/s at 4B, free text about greedy's door (*estimated*).

The 8B and 14B the same day ($1.76 of the operator's $2.60): identity at both sizes. Tokens a verify at k = 3 on the
code edit 2.54 and 1.98, on the retrieval answer 3.52 and 2.61, 1.09 to 1.48 on free text (*measured*). Projected with
the integer-dot verify: 241 and 334 tok/s at 8B, 115 and 150 at 14B on those two prompts (*estimated*).

## 2026-11-08 to 11-09. The speculative night: closed on today's verify, for $1.04

The operator's go: finish speculation and its tests within $15. One card job and one Mac bench were enough.

- [eagle-alpha](mesures/eagle-alpha-2026-11-08.txt), $0. The Apache-2.0 head `AngelSlim/Qwen3-4B_eagle3`, on the
  sealed 4B through the MLX door, replayed on the greedy text. α 0.43 to 0.61 on free text, 0.67 to 0.81 on GSM8K and
  copies (*measured* / *computed*). On today's verify it projects x0.72 on chat (*computed*). The conditional card job
  J3 was not run.
- [spec-final](mesures/spec-final-2026-11-08.txt), $1.04. Identity at 4B, 8B and 14B on every prompt, phase launch and
  GSM8K completion. An extra verify row costs 0.74 / 0.89 / 0.98 of a 4E token, with 92 to 95 % in qkv, o, gateup and
  down; attention and RoPE take under 0.5 % (*measured*). Prompt lookup with no-draft rounds on 4E and a trigram probe
  policy reads x0.94 / 0.96 / 0.92 / 0.88 of greedy decode on chat, code, summary and GSM8K at 4B, x1.108 on
  retrieval; the house prompt reaches 250.4 tok/s at the door (*measured*). Rule B is not met: lookup closes.
- B1 was cut for the night: the integer dot changes the logits, so it needs a new served object (4E's single-token
  path and the verify, perplexity and MMLU gates at three sizes), weeks of work, not a test.
- The served decode stays greedy 4E v9. The next lever is the K-row loop, on the operator's go.

## 2026-11-09. Draft trees on the Mac: step A of the tree verify holds, $0

[eagle-tree](mesures/eagle-tree-2026-11-09.txt). The same head and greedy text as eagle-alpha, drafting EAGLE-2 trees
instead of a chain: 16 rows emit 2.28 tokens a pass on chat and 3.15 on GSM8K at 3 head steps, against 1.69 and 2.42
for the chain of 3 (*measured*). A tensor-core verify with each prologue run once a row would cost 1.27 [1.03-1.60]
greedy tokens at 16 rows (*estimated*), and the 4B would decode about 352 tok/s on chat [275-435] (*computed*). The
plan is [`plan-verify-tree.md`](plan-verify-tree.md); its first card job, about $0.30, is drafted and not
timestamped. The int4 head's run was interrupted when the Mac froze and was dropped: a quantized head's acceptance
stays unmeasured.

## 2026-11-09. Tree verify S0: the kernel passes its host gates, $0

[tree-verify-s0-host](mesures/tree-verify-s0-host-2026-11-09.txt). `pers_4et.cu` verifies a tree of up to 32 rows in
one launch: projections on the tensor cores with d_in slices folded in order, prologues once a row, the row's keys in
logical order, the accepted rows' K and V moved in the next launch. On the Mac, through `host_grid.h` and an MMA
modelled from the PTX fragments: rows keep their bits at R = 1 to 32 and under slot permutations, a tree row equals
the root it becomes, R = 1 sits within 1.1e-3 to 3.5e-3 relative RMS of 4E's logits with its argmax, and 10 mutants
of 10 die (*measured*). The host fixture's SiLU table became SiLU itself for the closeness gate. A fold-order mutant
first survived: the units were dealt slice after slice, so the last slice always arrived last; they are now dealt
block after block. Nothing has run on a card; S1 still needs its Rust mirror, loader and bench.

## 2026-11-09. Tree verify S1 on the card: the rows are nearly free, the pass is not; the route stops, $0.12

[verify-tree-s1](mesures/verify-tree-s1-2026-11-09.txt). One L40S job (6af20dab, 4 min billed, registry $307.25), on
the operator's go "Oui lance s1", launched from the session "Tetra décodage et vitesse de sortie". At the chat context
an extra row costs 0.0196 of a 4E token, inside its signed interval, but a one-row pass costs 2.80 4E tokens against a
signed 1.09, c_8 2.92 and c_16 3.09 (*measured*). The prereg's rule reads FAIL (c_16 > 1.60 and c_8 > 1.48): the route
stops. Identity holds and the one-row pass makes 4E's token at 64 of 64 positions. The bench stopped at 4E's stamped
token, which `p4e_phase_report` refuses below two tokens, so the rag context and the phase tables are missing
(deviation 1); a second context can only raise the larger value the rule reads. Where the pass's fixed cost goes is
not measured; the tree symbol's 736 B of local memory a thread and its 3 extra grid waits a layer are suspects by
reading. A rerun with two stamped tokens would locate it, and needs the operator's go.

## 2026-11-10 to 11-11. The tree pass repaired, the speculative loop set aside, the chain prefill read in place; nothing served changes, $13.96

On the operator's go of 2026-11-09 and its $25 cap: 27 card jobs, $13.96 of the cap, registry $321.21 (*computed*).
The milestones are in [jalons-debit](mesures/jalons-debit-2026-11-11.txt).

- Tree verify pass, N1 to N11 ([verify-tree-n11](mesures/verify-tree-n11-2026-11-10.txt)). A 16-row pass costs 1.73
  tokens of 4E at the chat context, from 3.07 (*measured*). Identity holds on every arm.
- Speculative loop, SP2 to SP8 ([spec-loop-sp8](mesures/spec-loop-sp8-2026-11-11.txt)). With the DFlash drafter the
  4B decodes 502 tok/s on GSM8K-200 (x2.30), 343 on the house set (x1.56), 242 in chat (x1.09); GSM8K 167 right
  against 166, p = 1 (*measured*). It peaks at 5.41 GB against 2.01 (*measured*,
  [spec-mem-m3](mesures/spec-mem-m3-2026-11-11.txt)), at 4.87 b/param with the f16 drafter (*computed*,
  [spec-memoire](mesures/spec-memoire-2026-11-11.txt)). The operator set it aside as a fast profile; the default stays
  at 2.73 b/param.
- Chain prefill: the prompt as known tokens through the tree pass, reading the served words in place, no copy. First
  token at 1,024 / 4,096 / 8,191 tokens: eager 3.951 / 17.641 / 37.037 s, 16 rows a pass 0.702 / 4.317 / 13.368, 32
  rows 0.449 / 2.720 / 7.679, 32 rows with the full attention tile 0.420 / 2.393 / 6.485 (*measured*,
  [large-prefill](mesures/large-prefill-2026-11-11.txt), [large-prefill-ft](mesures/large-prefill-ft-2026-11-11.txt)).
  The 32-row forms equal the 16-row form bit for bit (argmax, logits, K and V); paired perplexity of the 16-row form
  +0.0007 % against eager. A bench, not served.
- Refuted or failed: E1, 4E's L2 prefetch removed or moved, x1.003 and x0.978 ([large-4e-e1](mesures/large-4e-e1-2026-11-11.txt));
  D1, one K/V read for 4 rows, x1.302 at 8,191 for a x1.5 gate ([large-prefill-d1](mesures/large-prefill-d1-2026-11-11.txt));
  v9ft, the full tile in 4E, x1.004 at 8,192 for x1.05 ([large-attn-ft-4e](mesures/large-attn-ft-4e-2026-11-11.txt)).
- Nothing served changes: no config, no served module, no Hub file. 4E v9 stays the served decode.

## 2026-11-12. The sealed 4B loses its quality in the MLP; CERVEAU and MODES adopted; nothing served changes, $5.26

[ablation-qualite-4b](mesures/ablation-qualite-4b-2026-11-12.txt). One L40S job (6af550a4, 175.4 min) restored
projection groups of the sealed 4B to f16 from the checkpoint, in eight dense arms on the full MMLU split. It cost
$5.26 against $5.79 estimated (*computed*). Spend under the $25 cap of 2026-11-09 is $19.22, registry $326.47. The
rows of `docs/data/jobs.csv` sum to $326.22, $0.25 under the registry; the journal traces the gap to
[large-prefill-ft](mesures/large-prefill-ft-2026-11-11.txt), not chased.

- The MLP carries the gap. Restored to f16, gate/up recovers +3.38 [+2.75, +4.02] MMLU points, down +2.01 [+1.46,
  +2.54], the whole MLP +4.86 [+4.16, +5.57]; attention recovers +0.73 [+0.16, +1.29], qkv +0.55, o -0.13
  (*measured*). All 252 projections recover +6.72 and sit +0.04 [-0.23, +0.31] under FP16. All exceeds the MLP plus
  attention by +1.14 [+0.43, +1.85] (*computed*). Six controls are green.
- Nine of fourteen signed predictions are in. The five misses are all on attention or additivity: qkv, o and attn
  came in under their intervals, A_attn and S above theirs.
- 4E off the bit is a study, not a measurement:
  [large-4e-hors-poids-etude](mesures/large-4e-hors-poids-etude-2026-11-12.txt). Its best composed set takes the chat
  door from 224 to 241-281 tok/s (*estimated*), under x1.3 (291). Nothing was launched.
- The night of 2026-11-11 ran three rounds of ideation, and every idea ended under its bar (+3.4 MMLU, x1.3 of
  decode). The ideas were judged by estimates, not by a measurement or a verifier. The notes are
  `gains-criteres-*`, `recherche-*`, `innovation-*` and `refutation-k1-*` of 2026-11-11 in [`mesures/`](mesures/),
  with [innovation-tetra-v2-forme](mesures/innovation-tetra-v2-forme-2026-11-11.txt). The study notes
  [innovation-ecart-qualite](mesures/innovation-ecart-qualite-2026-11-12.txt) and
  [innovation-cout-encodage](mesures/innovation-cout-encodage-2026-11-12.txt) followed on 2026-11-12.
- [`CERVEAU.md`](CERVEAU.md) now holds what we believe about the subject, 59 claims each with its standing, and
  [`MODES.md`](MODES.md) the two working modes that replace the night's method. Operator's ruling: a per-model tweak
  and a law drawn from a few points are not ideas. Delegation became a standing rule ([`METHODE.md`](METHODE.md) §8).
- An independent review of the same day added the standing `proved` to `CERVEAU.md` and regraded nine claims
  against the definitions: 13 established, 2 proved, 35 supported, 6 conjecture, 3 refuted. The ablation journal now
  states both readings of the prereg's "interacting" row: to the letter it fires and the next step is global.
- Nothing served changes: no config, no served module, no Hub file.

## 2026-11-12. A research phase opens on decode speed; six speed claims of CERVEAU corrected against their journals

Mandate in [`CERVEAU.md`](CERVEAU.md) O11: decode speed, the sealed files untouchable, three days and $5 of card. The
reading step is [recherche-debit-r0-depot](mesures/recherche-debit-r0-depot-2026-11-12.txt) and
[recherche-debit-r0-litterature](mesures/recherche-debit-r0-litterature-2026-11-12.txt); the framing is
[recherche-debit-r1-cadrage](mesures/recherche-debit-r1-cadrage-2026-11-12.txt). Nothing was measured.

- The reading found six claims that said more than their journals. K2 is the bare encode, not the sealed file. K3's
  A100 ran the eager engine only. K4's ×1.002 is the row loop alone, not a 4E form. K8's +47 % of VRAM is *computed*.
  K9's factor of ten is the grouped form; the kernel form of 4E reaches 0.65 of the band at 4B.
- K10 said that a third of the Tetra rows' time was lost inside 4E, at about 260 GB/s against 379 alone. That was
  4E v2. At v9 gate/up reads 359 to 385 GB/s in the token (*computed* from stamps), so the rows are bound by their
  decode. The anomaly N5 is now the whole token at 297 GB/s against rows at 359 to 648.
- The first card run of the phase, [recherche-debit-banc](mesures/recherche-debit-banc-2026-11-12.txt) (6af5ce28,
  $0.10), fails its fidelity rule on one control of twelve, so it scores no candidate. As a description it measures
  the L40S's bare read at 751 GB/s, the sealed Tetra rows alone at 407, and the same words read with nothing decoded
  at 493 (*measured*). The decode is not what holds the rows; the way their words are read is. Six of ten signed
  predictions missed, all built on the decode-bound framing. The image recipe compiles three crates at a time since
  this run: at two the build no longer fits the Space builder's 30 minutes. Spend under the cap: $19.32 of $25,
  registry $326.57.
- The second card run, [recherche-debit-lecture](mesures/recherche-debit-lecture-2026-11-12.txt) (6af5e9e7, $0.10),
  fails its fidelity rule too, on the served window (394.0 GB/s against [368, 383]), so it kills no hypothesis to the
  letter. It describes: read as a flat stream by 128-bit loads with no side stream, the sealed Tetra words stream
  717 GB/s against 751 for the card and 480 for the served read (*measured*). The row boundary, the load width and
  the side streams each cost about x1.2. Spend under the cap: $19.42 of $25, registry $326.67.
- 2026-11-13, the third card run, [recherche-debit-corps](mesures/recherche-debit-corps-2026-11-13.txt) (6af6f074,
  $0.10), is the first whose fidelity rule passes: the served arm timed twice a round agrees to 0.5 %, and four
  journal controls sit in their windows. Seven whole bodies were scored on the sealed Tetra rows alone. The flat
  body, a thread a run of 8 blocks on aligned 128-bit loads with the row ends after a barrier, reads 470.5 GB/s,
  x1.185 [1.179-1.190] of the served row, inside a bound of 1e-5. The served row with its row end in a second pass
  reads x1.053 [1.050-1.056] bit for bit. A thread a row reads x0.35 to x0.37 (*measured*). No tok/s is
  established. Nine of twelve signed predictions held. Spend under the cap: $19.52 of $25, registry $326.77.
- The fourth card run, [recherche-debit-jeton](mesures/recherche-debit-jeton-2026-11-13.txt) (6af70da3, $0.46),
  put the two bodies in 4E's token as E1 arms and is void for its question: neither arm loads on the card. The
  driver's ptxas gives their timed symbol 8 B of local memory at 128 registers, where the devel nvcc gave 0, and
  the runtime refuses it. The run holds that the rule F1 passes in the token with a floor 0.06 % wide, and v9 at
  225.8 tok/s at 5 tokens (*measured*). The job had no load check first and ran 923 s for nothing more. Spend
  under the cap: $19.98 of $25, registry $327.23.
- The fifth card run, [recherche-debit-charge](mesures/recherche-debit-charge-2026-11-13.txt) (6af73aed, $0.16),
  was the load check of four lighter bodies (v9ep2, v9ep3, v9fl2, v9fl3, green on the Mac) and is void: the check
  compiles seven modules on seven threads, prints only at the end, and was killed by its 5 min timeout with no
  line out. No arm has a verdict. Spend under the cap: $20.14 of $25, registry $327.39.
- The sixth card run, [recherche-debit-charge2](mesures/recherche-debit-charge2-2026-11-14.txt) (6af82af3, $0.42,
  2026-11-14), is the same check with the arms one after the other and each line printed as known. Its gates pass.
  None of the four lighter bodies loads: each is refused for 8 B of local memory at 128 registers, as v9ep and
  v9fl, and v9 loads (*measured*). Shedding live values from the bodies did not move the local memory by a byte.
  The token measurement is not launched. Spend under the cap: $20.56 of $25, registry $327.81.
- The seventh card run, [recherche-debit-sondes](mesures/recherche-debit-sondes-2026-11-14.txt) (6af837d1, $0.27),
  compiles five load probes, the endpass body on one call site each. One site brings the 8 B: gate/up. The body on
  q/k, on the Tetra down or on the Tetra o alone loads at 0 B, and so does the inserted text alone (*measured*).
  Nothing is timed. Spend under the cap: $20.83 of $25, registry $328.08.
- The eighth card run, [recherche-debit-jeton3](mesures/recherche-debit-jeton3-2026-11-14.txt) (6af85b14, $0.32),
  checks three arms that leave the one-call gate/up site out: flat and endpass with gate/up in two calls, and flat
  with v9's own gate/up. None loads, each refused for the same 8 B at 128 registers (*measured*). The job stops
  after the check. Flat on q/k, down and o is refused even with v9's gate/up, so gate/up is not the only way to the
  8 B. Nothing is timed. Spend under the cap: $21.15 of $25, registry $328.40.
- The operator closed the phase on 2026-11-14, on a ceiling ([`CERVEAU.md`](CERVEAU.md) O11). No tok/s was gained
  and no candidate was timed in 4E's token. The gain on rows alone stands. Phase spend: $1.93 of $5. The registry
  figures above are this branch's; with the quality phase's rows the registry sums to $341.64 (*computed*).

## 2026-11-13 to 11-14. A research phase on quality opens and closes on a dead hypothesis; MODES allows two phases on two axes; nothing served changes, $13.49

The operator signed the mandate of a research phase on model quality, O12 of [`CERVEAU.md`](CERVEAU.md). It started
from the anomaly N1 and ran beside the phase on decode speed (O11), on five days and $10 of card under a cap of its
own, raised to $15. It closed the next day: judging gate/up on the final output, in its one-sided form, did not beat
the layer objective on the 4B. The bullets below are in the order things happened.

- [`MODES.md`](MODES.md) said until this day: "At most one research phase and one development phase run at a time:
  one card budget, one work machine." The operator replaced it. Two research phases now run together when each
  mandate carries its own card budget, the second loads no model on the work machine, and the two edit no file in
  common but `CERVEAU.md`.
- R0 of the phase is done the same day, by reading only: what the repository knows
  ([recherche-qualite-r0-depot](mesures/recherche-qualite-r0-depot-2026-11-13.txt)) and what is published
  ([recherche-qualite-r0-litterature](mesures/recherche-qualite-r0-litterature-2026-11-13.txt)). N1's Gaussian arm
  is noise on the four option logits of a dump, not on weights, and its replay on the sealed 4B needs dumps the
  repository holds. No published study answers N1 or N2. Eight claims of `CERVEAU.md` gained scope notes.
- R1's first measurement ran the same day under a stamped prereg (sha256 3323f313), on dumps, with no model:
  [recherche-qualite-n1](mesures/recherche-qualite-n1-2026-11-13.txt). N1 holds on the sealed 4B: 63.37 measured
  against 64.65 predicted by Gaussian noise of the same amplitude on the four option logits, −1.27 [−1.89, −0.65]
  (*computed* on *measured* dumps). The amplitude alone predicts 5.49 of the 6.77 points to FP16. AWQ 4B reads −0.88
  on the same split, so the gap is not LLVQ-only. Ten of thirteen signed predictions are in. `CERVEAU.md` Q7 is
  restated: its former sentence, "where it lands matters more than its size", does not hold as shares on this object.
- The operator then moved the phase's starting point from N1 to the amplitude share, the 5.49 points: where an error
  lands against chance cannot be known in advance. N1 is set aside. No budget is added yet.
- On the operator's go under $5, one card job, $0.44 (6af6f9da, prereg sha256 208751f9):
  [layer-vs-output-4b](mesures/layer-vs-output-4b-2026-11-13.txt). Restoring one projection group to FP16 in the
  sealed 4B recovers 36.29 % of the KL to FP16 for gate/up, 23.17 % for down, 8.74 % for qkv and 2.48 % for o
  (*measured*), the order of the MMLU restores. The relative layer-output error reads down > qkv > o > gate/up: it
  rates gate/up lowest. Rule D green, perplexity within 0.033 % of the candle path. Eight of seventeen signed
  predictions are in. Spend of the phase: $0.44 of $10; the rows of `docs/data/jobs.csv` sum to $326.66 (*computed*).
- The operator gave the go for a first re-encode of the 4B with gate/up judged on the final output, in its smallest
  form (gradient-weighted input Hessians, $1.65 to $1.89 *estimated*), and lifted the Mac clause of the mandate for
  its two encodes. Nothing is coded or launched yet.
- The code of that attempt passed the workspace fast loop (1,321 passed, 0 failed), clippy and ten mutants; its
  statistic job cost $0.09 (6af72f55). The operator moved the two encodes from the Mac to the card and raised the
  phase's budget to $15. Spent: $0.53.
- The attempt ran in full for $8.67: [guided-gateup-4b](mesures/guided-gateup-4b-2026-11-13.txt). Outcome LOSS under
  its prereg (sha256 dac1bca1, departures E1 and E2 stamped). The candidate, gate_proj and up_proj under
  gradient-weighted input Hessians, scores 57.11 on the full MMLU split against 58.62 for a baseline encoded the same
  day: −1.51 [−2.30, −0.70] (*measured*). Perplexity 18.08 against 16.46. Five of eight signed predictions are in; the
  lead predicted a gain. A card encode of the 4B took 77 and 80 billed minutes on `rtx-pro-6000`, $3.55 and $3.68
  (*computed* at $2.75/h). Read after the result and not preregistered: the per-token weights have an effective
  sample size of 1.58 % of the calibration tokens (*computed*). Spend of the phase: $9.11 of $15; the rows of
  `docs/data/jobs.csv` sum to $335.33 (*computed*).
- 2026-11-14, on the operator's go: one retry with each group's weights clipped at 10 times their mean
  (`LLVQ_GUIDED_CLIP`, cd70933; prereg sha256 83131dca, stamped before the encode), $4.38:
  [guided-clip-gateup-4b](mesures/guided-clip-gateup-4b-2026-11-14.txt). NOT RESOLVED: 57.95 against the baseline's
  58.62, −0.68 [−1.47, +0.14]; +0.83 [+0.03, +1.65] against the unclipped encode (*measured*). Perplexity 16.69. Seven
  of seven signed predictions are in. The gradient-weighted input-Hessian form is closed for the phase. Spend of the
  phase: $13.49 of $15; the rows of `docs/data/jobs.csv` sum to $339.71 (*computed*).
- 2026-11-14: the operator closes the phase. Outcome: a dead hypothesis, recorded in `CERVEAU.md` O12. No idea sheet.
- Nothing served changes: no config, no served module, no Hub file.

## 2026-11-14. A second research phase on quality opens on the lost contrast of the answer logits and closes on a dead hypothesis; nothing served changes, $2.48

The operator refused the lead's first three starting points for a new quality cycle, the calibration draw, probes of
gate/up and a fast verifier, as tinkering and not ideas. He then signed a mandate, O14 of [`CERVEAU.md`](CERVEAU.md)
(numbered O13 on its branch, before the long-context speed phase took that number on main),
on an anomaly read the same day from the dumps the repository holds: $10 of card, five days.

- The anomaly, N6: the sealed 4B's four answer logits regress on FP16's with slope 0.58, where AWQ 4B keeps 0.96
  ([recherche-qualite2-r0-contraste](mesures/recherche-qualite2-r0-contraste-2026-11-14.txt), *computed* on
  *measured* dumps). With every projection quantized the slope runs from 0.22 to 0.58 across 17 files and the noise
  from 1.20 to 1.77 a logit. The error-map correction and the fourfold calibration volume, two cases where perplexity
  improved and MMLU fell, lowered the slope more than the noise.
- At the sealed 4B's noise the Gaussian replay puts full contrast at 68.15 against 64.66 (*computed*, an upper bound
  of that model). [logit-snr-4b](mesures/logit-snr-4b-2026-09-12.txt) section 9 had read the attenuation in September
  and set it aside: only the ratio decides an argmax. The phase asks whether the contrast can be raised without
  raising the noise as much.
- One idea was killed at $0 the same day: coding each up_proj row from the gate_proj row of its neuron. Their mean
  absolute cosine is 0.109 on Qwen3-4B and a one-scalar prediction leaves 97.8 % of the energy.
- The literature half of R0 found the output shrink published (Wu et al., arXiv 2608.06564) and the idea as a whole
  nowhere ([recherche-qualite2-r0-litterature](mesures/recherche-qualite2-r0-litterature-2026-11-14.txt)).
- One card job, $0.46 (6af86d01, prereg sha256 cba0eb94, stamped before):
  [contrast-map-4b](mesures/contrast-map-4b-2026-11-14.txt). FP16 and the sealed 4B side by side on 570 MMLU prompts.
  The loss is complete when the answer is written, at block 25 of 36: lens accuracy 72.3 % against 63.9 %, and 73.7
  against 66.5 at the output (*measured*). The output slope is 0.81 at the write and falls to 0.61 across the last
  block by a rescale. At the answer position the quiet directions keep a slope of 0.47 at depth 12 under 2.5 times
  their energy in noise. Rule D green. Fifteen of eighteen signed predictions are in. The lead's idea of gains chosen
  for a coherent gain of 1 along the depth does not survive in that form. Spend of the phase: $0.46 of $10; the rows
  of `docs/data/jobs.csv` sum to $342.10 (*computed*).
- On the operator's go, three more card jobs, $1.71. Two restores of the sealed 4B on the full split
  ([depth-restore-4b](mesures/depth-restore-4b-2026-11-14.txt), 6af88648, $1.38): layers 25 to 35, 30.6 % of the
  projection weights, gain +0.43 [+0.11, +0.77] where the even rate of Q22 put +2.05; layers 0 to 24 gain +6.12
  [+5.41, +6.84] (*measured*). The map is supported, and Q25 is established on the sealed 4B. A reading of the
  originals alone ([lens-write-depth](mesures/lens-write-depth-2026-11-14.txt), $0.12 and $0.21): the 8B writes its
  answer at depth 25 of 36 and the 14B at depth 29 of 40, each in one block (Q26). Eighteen of eighteen signed
  predictions are in. Spend of the phase: $2.17 of $10; the rows of `docs/data/jobs.csv` sum to $343.81 (*computed*).
- GSM8K then read the same two windows ([gsm8k-depth-restore-4b](mesures/gsm8k-depth-restore-4b-2026-11-14.txt),
  6af89e7b, $0.31, three dense f16 files in vLLM): layers 25 to 35 gain +4.32 [+2.46, +6.18] where MMLU read +0.43;
  layers 0 to 24 gain +7.20 (*measured*). The MMLU pattern does not hold on generation (Q27), Q25 is bounded to MMLU,
  and no allocation of bits by depth is proposed. The lead had predicted +1.2. Spend of the phase: $2.48 of $10; the
  rows of `docs/data/jobs.csv` sum to $344.12 (*computed*).
- The operator closed the phase the same evening ("Dead on l'écrit"). Outcome: a dead hypothesis, recorded in
  `CERVEAU.md` O14, with two beliefs dropped and the anomaly N7. No idea sheet. A lens reading of generated tokens
  was proposed and not run.
- Nothing served changes.

## 2026-11-14 to 11-15. A research phase opens on decode speed at long context; an int8 KV cache holds the tolerance at 8,192 and buys 1 % of speed in the kernel; nothing served changes, $5.77

The operator closed O11 and opened a phase on decode speed after an 8,192-token prompt, O13 of
[`CERVEAU.md`](CERVEAU.md): five days and $10 of card.

- A capture mode (`LLVQ_KV_CAPTURE`) and an offline scorer were built to test a certified partial read of the cache
  (prereg sha256 c678161c). The operator judged the chance too low and the job was not run.
- A ladder of four quantized caches was scored as round trips against f16 on the sealed 4B, 12 windows of 8,192
  (`LLVQ_KV_LADDER`, prereg sha256 8913d843): [recherche-debit-kv-ladder](mesures/recherche-debit-kv-ladder-2026-11-14.txt).
  int8 keys and values +0.0126 % [−0.0485, +0.0737], inside ±0.12 %; int8 keys with int4 values +0.0206 %
  [−0.1359, +0.1774], not decided; int4 keys fail at +1.27 % and +141 % (*measured*).
- CERVEAU P4 is replaced. It read: "An int8 KV cache does not pass the served tolerance at long context: its own
  measured interval leaves the ±0.12 % band", from +0.0494 % [−0.0713, +0.1703] on 12 windows of 4,096 on Metal
  (kvq8-4b-2026-08-15).
- Five jobs were void for dataset pins written by the main session, $0.15.
- The int8 read was written as a research arm of 4E, `v9k8` (prereg sha256 4c7fe8cd), bit for bit against its
  reference on the Mac. On the card it is refused at its load check for 8 B of local memory at 128 registers, v9 at
  0 B beside it (*measured*, [recherche-debit-kv8](mesures/recherche-debit-kv8-2026-11-14.txt), $0.13). Nothing was
  timed.
- On 2026-11-15 the arm was timed as it is, the local-memory half of the load rule waived for it alone as a
  measurement mode (`+local`, prereg sha256 633c536d). x1.0109 [1.0108-1.0113] over v9 at 8,192, 164.4 against 162.6
  tok/s; x0.9668 [0.9664-0.9670] at 5 tokens, 219.4 against 226.9; v9's 256 tokens at both contexts (*measured*,
  [recherche-debit-kv8-local](mesures/recherche-debit-kv8-local-2026-11-15.txt), $0.33). It is the first token symbol
  timed with local memory.
- CERVEAU P3 changes standing, from conjecture to contradicted on one tile. It read: "At long context the decode
  attention is bound by its own bytes". The attention phase at 8,192 lost 1.8 % for 46.9 % fewer KV bytes.
- Spend of the phase: $5.77 of $10; the rows of `docs/data/jobs.csv` sum to $347.41 (*computed*).
- The operator closed the phase on 2026-11-15, outcome a dead hypothesis: a smaller KV cache does not buy decode
  time at long context. He keeps the int8 cache as a memory lead, 1,208 to 642 MB at 8,192 positions on the 4B
  (*computed*), not served. `ETAT.md` §5 and §6, `ROADMAP.md` §2.7 and `plan-300.md` §7 follow. ROADMAP §2.7 read:
  "An int4 KV cache [...] is the only lead that reaches 300 tok/s at that context".
- Nothing served changes: no config, no served module, no Hub file. No served tok/s is gained: the int8 read is a
  research arm, about 1 % faster at 8,192 and 3 % slower at 5.

## 2026-11-15. A development step: three served gains asked, two wired; the chain prefill measured at a door and served on the 4B, $1.35

The operator asked for three gains already measured on benches to be made servable: the chain prefill, the int8 KV
cache as a memory lead, and speculative decoding as an opt-in profile. Branch `dev/trois-gains-2026-11-15`.

- The chain prefill became a config field, `"prefill": "chain"` (`configs/qwen3-4b-tetra-e4-4e-v9-chain.json`), read
  by `fusedrun`, `gsm8k`, the REPL and `llvq`. Its tree modules load from a disk cache of their cubins, keyed by the
  kernel text, the NVRTC target and version, the driver and the card (`LLVQ_MODULE_CACHE`). 4B only.
- Its door job, [chain-prefill-served](mesures/chain-prefill-served-2026-11-15.txt) (6af99acc, $1.16): first token
  after 0.45 / 2.47 / 6.74 s at 1,024 / 4,096 / 8,192 tokens against 3.96 / 17.79 / 37.22 s for v9, the same 256
  tokens at each length, phase peak 3.633 GB against 6.047 at 8,192, GSM8K-200 165 against 165 with no discordant
  pair, load 509.2 s cold and 55.2 s cached (*measured*). The prereg read NOT SETTLED on one bound, a decode 2.9 and
  3.2 % faster than v9's at long context.
- The operator served it on the 4B on 2026-11-15. He accepted local memory in the tree modules alone (744 B), and a
  prefill that is not bit for bit with the eager one under the ±0.12 % band of the attention forms. `llvq` loads the
  chain config for the 4B on CUDA; `LLVQ_PROFILE=eager` is the way back.
- `llvq`'s own door was then read on a card (job 6af9b69f, $0.19, prereg sha256 07cf3e6d,
  [chain-runtime-door](mesures/chain-runtime-door-2026-11-15.txt)): it loads the chain config with no variable set,
  first content at 8,191 tokens after 6.448 s against 34.537 s for the way back, the way back and both refusals as
  written (*measured*). The prereg read FAILS on its control of repeatability, and the control failed on the eager
  server: one text with no prompt token cached, another with 3 cached. The chain gave one text in its four rounds,
  the eager server's timed one. The operator merged on that reading the same day; the departure from the prereg's
  row is in `proofs/preregistration-chain-runtime-door-2026-11-15-ECARTS.md`.
- `ETAT.md` §2 read: "On CUDA it loads the v9 configs". §4 read: "The 4B's first token at 8,192 takes 35.05 s". §5
  read: "The chain prefill, a bench, not served" and then "measured at a door, not served".
- The int8 cache got the form that frees the f16 cache at the hand-over, as a NOT SERVED config refused at every
  served door, for the 8 B of local memory in its token symbol. It is not run on a card; its prereg is a draft.
- Speculative decoding was stopped before any code: the loop returns the tree pass's greedy tokens, not 4E's, 177 of
  200 GSM8K prompts identical (spec-loop-sp8), and a loop identical to 4E needs the drafter's taps from a verify pass
  that does not produce them. The drafter's licence is recorded nowhere in the repository.

## 2026-11-15. A third research phase on quality opens on two anomalies and closes on a dead hypothesis; nothing served changes, $1.49

The operator opened a third quality phase on the two anomalies the second one left, N6 and N7, on the explanation
route: O15 of [`CERVEAU.md`](CERVEAU.md), $10 of card, five days. The second phase is O14 there; its journals say
O13, the number the long-context speed phase took on main the next day.

- R0, two agents, $0: what the repository holds
  ([recherche-qualite3-r0-depot](mesures/recherche-qualite3-r0-depot-2026-11-15.txt)) and what is published
  ([recherche-qualite3-r0-litterature](mesures/recherche-qualite3-r0-litterature-2026-11-15.txt)). The first found
  an error in a journal of the day before: the sealed 4B carries int4 records for o_proj at all 36 layers, not at
  layers 12 to 23. The journal has its erratum and Q11 its note.
- The frame ([recherche-qualite3-r1-cadrage](mesures/recherche-qualite3-r1-cadrage-2026-11-15.txt)) put N6 as one
  question: whether the MMLU deficit travels in the directions the calibration average calls cheap. Five rivals were
  written with what kills each.
- Readings at $0 on dumps already on disk, under a prereg stamped before (sha256 b2cdf2f9;
  [recherche-qualite3-dumps](mesures/recherche-qualite3-dumps-2026-11-15.txt), *computed* on *measured* dumps). Of the
  108 GSM8K problems the late restore repairs, 84 are also repaired by the early restore: nothing on disk makes the
  late blocks a place apart for generation (Q28). The first divergence from FP16's text turned out to read the
  opening word, 440 of 1,319 completions differing at their first character, and decides nothing. Eight of the
  lead's fifteen signed predictions were right.
- The patch map, one card job on the operator's go (6af9953a, $0.61, prereg sha256 0001bf6a, stamped before;
  [patch-map-4b](mesures/patch-map-4b-2026-11-15.txt), *measured*). FP16's stream is written into the sealed run on
  570 MMLU questions. At depth 12 the whole stream gives back 5.09 points [+2.10, +8.08] of a gap of 7.19. Its band
  of directions 513 to 2,560 gives back 3.33 points [+0.62, +6.05] alone, and 0.859 [0.734, 0.997] of the whole
  stream's share of shape agreement. The answer position alone gives back nothing. FP16's MLP outputs of blocks 0 to
  11 give back 0.850 [0.751, 0.935] of the depth-12 patch (Q29). One arm is not explained: FP16's attention outputs
  at the same blocks cost 12.63 points (N8). The rules were read on a share of shape agreement, after an instrument
  check on nine known restores, since an accuracy share resolves nothing on 570 questions.
- The lead drew a conjecture from it: the encoder misprices the quiet directions, and a noise laid out by the
  signal would cost less ([recherche-qualite3-r5-conjecture](mesures/recherche-qualite3-r5-conjecture-2026-11-15.txt)).
  Two agents who did not write it attacked it at $0, and it did not survive
  ([recherche-qualite3-r6-issue](mesures/recherche-qualite3-r6-issue-2026-11-15.txt)): what a band gives back
  follows the noise energy it held, 70.6, 79.3 and 70.2 a unit, and the lead's own arithmetic had counted a band
  twice. The lead had told the operator of a flaw of principle in the compressor before the attack, and corrected it
  the same day. No encode was run.
- The sensitivity map, a second card job on the operator's go (prereg sha256 1453e867, stamped before;
  [sensitivity-map-4b](mesures/sensitivity-map-4b-2026-11-15.txt), *measured*). Its first launch stopped on its own
  guard before any load (6af9be70, 79 s, $0.04): two candidate prompts of the Fisher carried the token hash of a
  scored prompt. The departure note is stamped, and the relaunch ran green (6af9c02c, $0.84). In the basis of a
  Fisher of running text a unit of deviation costs 2.34 [1.16, 3.72] times more in the 512 most sensitive
  directions. The rule reads NOT DECIDED. The Fisher's top 64 directions hold 0.516 of its trace and give back
  +0.026. The top-64 subspaces of running text and of MMLU answers overlap by 0.082 (Q30). The lead had put an
  uneven sensitivity at 55 %, and its stated reason was wrong.
- The operator closed the phase the same day ("ferme et fusionne"). Outcome: a dead hypothesis and one question not
  decided, recorded in `CERVEAU.md` O15, with one belief dropped and the anomaly N8. No idea sheet. Spend of the
  phase: $1.49 of $10; the rows of `docs/data/jobs.csv` sum to $352.73 (*computed*).
- Nothing served changes.

## 2026-11-17. The writing rules are rebased on two public skills, and paper 2 is revised; nothing served changes, $0

- `STYLE.md` rests on two MIT skills read at pinned commits: `Master-cai/Research-Paper-Writing-Skills` for
  paragraph and section structure, `blader/humanizer` for the tells of machine-written text. The rule "one idea a
  sentence" gives way to "one message a paragraph, sentences that connect". The project rules stay: no history, a
  number labelled once, no banners. The operator asked for the change.
- Paper 2's source is revised on branch `paper2/v3`: the persistent kernel and its served speeds, llama.cpp's IQ2_S
  at three sizes, long prompts, the 4B ablation, five new data figures. Its abstract, introduction, kernel and results sections are passed through the
  new rules; sections 2 to 4 are only scanned for the tells. Not released.


## 2026-11-20. The three sealed cards take a metadata pull request from the Hub; nothing served changes, $0

- A Hugging Face staff member opened the same pull request on the three sealed repositories. Each one changes
  `README.md` only: it adds `pipeline_tag: text-generation`, links paper 2 to its page
  (`huggingface.co/papers/2609.35465`), and reorders the front matter. The operator had them merged ("Oui, fusionne"),
  which lifts the freeze of 2026-11-02 for these three cards and for this change only.
- The repositories moved from `58e96ba3`, `2a6c85db` and `64fd81f7` to `0bfc3c36`, `21dc9980` and `dc0d94f7`. The
  three `.bin` files keep their sha256 (`886391a8`, `7bdb9a55`, `61db37fe`), read back from the Hub after the merge
  (*measured*). The revisions `llvq-runtime/library.json` pins are the earlier commits and still resolve to the same
  files. The safetensors repository (`acc5f663`) and the served one (`159afed4`) were not touched.
- `docs/hub/qwen3-4b-sealed.md`, `qwen3-8b-sealed.md` and `qwen3-14b-sealed.md` are the merged cards byte for byte:
  9,831, 5,625 and 5,659 bytes (*measured*). The 9,774 bytes `docs/plan-transformers.md` gives for the 4B card is its
  size on 2026-10-20.

## 2026-11-25 to 11-26. Olala v1.0 mini 7A1B read before any encode: a plan, a second word and a calibrated proxy; nothing served changes, $0

- OVHcloud released Olala v1.0 mini 7A1B on 2026-11-19 under Apache 2.0. The operator asked for an analysis and a
  plan to quantize it. Everything ran in a cloud container of 4 threads with no GPU: no paid job, no Hessian, no GPTQ
  loop on the model, no file, no task score.
- The reconnaissance read the object and its difficulty. The file holds 6,773,124,360 parameters, 80.3 % of them in
  the routed experts. Run data-free, the Tetra encoder leaves 0.0846 to 0.0854 on 14 of 20 matrices against 0.0848
  and 0.0852 on Gaussian ones, and OVH's massive activations are reproduced, 2.9e16 at the mixer branch of layer 1
  (*measured*, [olala-reconnaissance](mesures/olala-reconnaissance-2026-11-25.txt)).
- [`plan-olala-mini.md`](plan-olala-mini.md) proposes a recipe at 2.61 b/param, 2.21 GB (*computed*), a quality gate
  against the smallest public file, and six lots. Every lot waits for the operator.
- A second Tetra word on the residual takes the error from 0.0851 to 0.0074 at 4.00 b/weight, under int4 g128's
  0.0100 at 4.25 (*measured*, [olala-second-mot](mesures/olala-second-mot-2026-11-26.txt)).
- On the operator's go ("Tu peux sur cette machine ? Si oui go") the noise proxy of the reconnaissance was read
  against a real data-free encode of the routed experts of layers 6, 17 and 28, under a reading plan committed before
  the reads. The encode reads 1.037 and 0.952 times the proxy's KL at one word and two, and two words read 19 % less
  KL than int4 g128 with the same top-1 (*measured*,
  [olala-calibrage-proxy](mesures/olala-calibrage-proxy-2026-11-26.txt)). The KL levels the plan expected were wrong,
  0.0318 where 0.010 to 0.013 was written.
- `CERVEAU.md` gains C8, C9 and Q31, all three supported, and goes from 68 claims to 71. `ROADMAP.md` §4 points at
  the plan. The operator had the branch merged into main ("tu peut merge dans main pour alimenter la base de
  connaissance").


## 2026-11-26. The Tetra cut: four repairs read through a model, the lead closed; nothing served changes, $0

- The operator asked whether a change of geometry could avoid Tetra's 0.375 dB against the ball. Four repairs of
  the per-section caps were read on Gaussian blocks through a direction-density model calibrated on the ball and on
  the 4-profile selector (*computed*, [geometrie-tetra-proxy](mesures/geometrie-tetra-proxy-2026-11-26.txt)). Two earlier
  models missed their calibration and are kept beside the journal.
- None beats the selector by more than 0.03 dB: the best reads −0.21 dB. Pairing the six tetrads per block reads
  −0.16 dB with the rank table unchanged at 16 KiB, which is the one new fact. A sparse list of corrections is dead,
  because the excess is spread over blocks. Width profiles and the shared budget are known art (AMR-WB+, V.34).
- The operator closed the lead ("non ferme et ajoute dans la base de connaissance") and declined the read of the
  per-section saturation rate on a sealed file, which stays the reopening condition in `ETAT.md` §6. `CERVEAU.md`
  gains C10 as a conjecture and one dropped belief. No prereg, no job, no model load.


## 2026-11-26. Inkling 2 Mini on the Mac: served by the runtime, 30.1 tok/s, a prefix cache on disk; $0

- The operator opened a speed campaign on the Mac and centred it on Inkling ("je veux qu'il bombarde sur mac").
  Four lots ran the same day on one 64 GB M3 Max, each under a prereg stamped before its first measurement, and
  every change kept the logits bit for bit.
- `llvq serve --local` serves the model on Metal with the long-prompt settings: 13 of 13 checks
  (*measured*, [inkling-mac-serve](mesures/inkling-mac-serve-2026-11-26.txt)).
- The two CUDA levers of the prompt read were ported to Metal and stay off. Lattice launches go from 1,605,064 to
  3,456 at 9,000 tokens and the read does not follow: 0.816, 1.028 and 0.727 of the default over three alternated
  rounds (*measured*, [inkling-mac-prompt](mesures/inkling-mac-prompt-2026-11-26.txt)). The coordinating session
  had signed 70 to 110 s for that read; it stayed at 140 s [134, 149].
- The decode went from 26.66 to 30.12 tok/s [29.99, 30.18] after a 32-token prompt: `down` of the fused MoE call
  one simdgroup a slot, and 30 compute encoders a command buffer. The MoE rotations in threadgroup memory were
  refuted at the kernel bench (*measured*, [inkling-mac-decode](mesures/inkling-mac-decode-2026-11-26.txt)).
- The operator chose a prefix cache on disk for the long prompt ("Cache de préfixe"). A new session over an
  8,704-token static block has its first token after 3.90 s [3.62, 4.57] instead of 155 to 202 s, in a 1.41 GB
  file where 0.4 to 0.8 GB had been signed (*measured*,
  [inkling-mac-prefix](mesures/inkling-mac-prefix-2026-11-26.txt)). It is on by default on Metal only.
- The review before the merge found that the cache's key did not identify the weights, and the key now carries
  the files the model loads. The same 9,000-token read ranged from 109 to 305 s over the day with no cause found.
- `CERVEAU.md` gains A3 and A4, both supported, and the anomaly N9, and goes from 72 claims to 74. The operator had
  the branch pushed and merged ("Oui, pousse", "relis et merge").


## 2026-11-26 to 11-27. The Qwen3 files served on Metal grouped, the prefix cache on the CPU, a held prompt answered with no read; $0

- Three lots ran on one M3 Max under a prereg stamped before the first measurement
  ([`servi-au-bit`](../proofs/preregistration-servi-au-bit-2026-11-26.md)), at no cost.
- Part A. `llvq` serves the Qwen3 4B, 8B and 14B on Metal through `configs/qwen3-{4b,8b,14b}-tetra-e4-metal.json`,
  which carry the grouped attention and a new optional field, `metal_kernels`, naming the kernels `lutd+h+pq`. The
  matrices read 253 of 253 and 281 of 281 bit for bit, grouped against expanded max |dlogit| 0, the same tokens and
  the same text through the server. Against the expanded configs, round by round, decode after 5 tokens is x1.142,
  x1.097 and x1.077, after a long prompt x3.201, x1.689 and x1.525, and the 4B's first token at 4,096 tokens falls
  from 67.87 to 35.17 s (*measured*, [qwen-metal-servi](mesures/qwen-metal-servi-2026-11-26.txt)). The absolute
  rates of that night came from a loaded machine and are not quoted. The CPU keeps the expanded configs.
- Part B. The Inkling prefix cache is on by default on the CPU as on Metal. A restarted server gives the first token
  of a 2,316-token prompt after 20.33 s against 140.99 s cold, with the same answer (*measured*,
  [inkling-cpu-prefix-held](mesures/inkling-cpu-prefix-held-2026-11-26.txt)).
- Part C. A request whose whole prompt is held is answered from the logits kept at the end of its first read. On
  Metal the 8,791-token prompt sent again gets its first token after 0.159 s against 2.687 s, same journal.
- Two statements replaced in the living documents. `CLAUDE.md` said the grouped form is refused on Metal, where it
  had never been measured: it was measured bit for bit on 2026-11-06 and is served there since this lot, and
  `ETAT.md` §4 and §5 said the same until 2026-11-26. `ETAT.md` and `CERVEAU.md` A2 said the CPU backend was never
  run on x86: it ran on 2026-11-25, 5.80 and 6.53 tok/s on the 4B on one 8 vCPU instance at its last job
  ([llvq-cpu-x86](mesures/llvq-cpu-x86-2026-11-25.txt)).
- `CERVEAU.md` gains A5 to A8 and goes from 74 claims to 78.

## 2026-11-27. Two readings that serve nothing: Inkling's window arm on the Mac, the grouped attention on the CPU; $0

- Three lots ran the same night on one M3 Max, each under a prereg stamped before its first measurement
  ([`inkling-fenetre`](../proofs/preregistration-inkling-fenetre-2026-11-27.md),
  [`inkling-fenetre-ppl`](../proofs/preregistration-inkling-fenetre-ppl-2026-11-27.md),
  [`qwen-cpu-groupee`](../proofs/preregistration-qwen-cpu-groupee-2026-11-27.md)), at no cost. No default changed
  and the runtime was not touched.
- The window arm. `inkrun --attn-window narrow` scores each chunk of query rows of Inkling 2 Mini's sliding layers
  over the keys it can see, not over every key the slot holds. A 9,000-token prompt is read in 112.41 s against
  131.48 s, x0.855 [0.854, 0.856] over four alternated pairs, and a 2,000-token prompt x0.945 [0.943, 0.951]
  (*measured*, Metal, [inkling-mac-fenetre](mesures/inkling-mac-fenetre-2026-11-27.txt)). The logits move by up to
  6.41 against a largest logit of 29.49, with the same argmax and the same 32 tokens. The peak footprint is 0.24 GB
  higher at 9,000 tokens. The decode had nothing to gain: its score kernel already reads the window only.
- Its perplexity. Paired by position against the reference, the arm reads +0.022 % [-0.045, +0.089] on two joined
  WikiText-2 texts of 6,144 tokens and +0.015 % [-0.047, +0.076] on six windows of 2,048, inside the +-0.12 % band
  the pieces carry ([inkling-mac-fenetre-ppl](mesures/inkling-mac-fenetre-ppl-2026-11-27.txt)). No difference sits at
  a chunk or piece boundary. One window alone reads +0.169 % [+0.046, +0.292]. The arm stays off: serving it needs
  a reading on CUDA and on the CPU, a switch in the runtime, and the operator's word.
- The grouped attention on the CPU. The grouped configs load on the CPU backend as they are. After a 2,048-token
  prompt the 4B decodes x1.555 [1.550, 1.566] faster, 6.515 against 4.186 tok/s, and the 8B x1.378 [1.371, 1.385],
  4.363 against 3.165; after 5 tokens x1.033 and x1.038; the first token at 2,048 tokens does not move (*measured*, 6
  threads, f32, [qwen-cpu-groupee](mesures/qwen-cpu-groupee-2026-11-27.txt)). On the CPU the two forms are not bit
  for bit, where they are on Metal: max |dlogit| 8.1e-4 and 3.5e-4, no step whose argmax differs in 2,047, the same
  tokens. `cpurun` gains two measurement modes, `timed` and `lockstep`. The CPU keeps the expanded configs: a
  grouped one would need the paired perplexity of `ETAT.md` §4 on the CPU, about a night of this Mac (*estimated*).
- Predictions that missed, as signed in the preregs. The window arm's logit gap was predicted under 2e-3 and read
  6.41: the tiny model's drift does not carry to a model whose routing turns a small drift into other experts. Its
  peak was predicted not higher. The grouped form on the CPU was given one chance in two to be bit for bit, and its
  first token at 2,048 tokens was predicted x1.1 [1.0, 1.4] faster and read x0.997.
- `CERVEAU.md` gains A9 and A10 and goes from 78 claims to 80.
