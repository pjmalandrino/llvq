# Upstream proposal: LLVQ Tetra in `transformers`

Nothing is posted yet. [`ISSUE.md`](ISSUE.md) is the draft of the issue body, 103 lines. It can be edited until the
day it is posted. From then on it is an exact copy of what was posted and is not edited again, as in
[`../candle-broadcast-matmul/`](../candle-broadcast-matmul).

## Why the issue and the PR go out together

The two upstream guides disagree. `docs/source/en/quantization/contribute.md` lists ten code steps for a new
quantization method and does not mention an issue (read 2026-10-04). `CONTRIBUTING.md` asks for an issue before any
new feature, with the motivation, the details, a code snippet and the paper.

The operator decided on 2026-10-07 to post both at once. The issue gives the reasons and links the PR.

## Before posting

- [x] `pip install llvq-tetra` works. 0.1.0 has been on PyPI since 2026-10-07, and a clean environment installs it
      and loads a model (`docs/mesures/hf-pypi-0.1.0-2026-10-07.txt`).
- [ ] The PR is ready. The draft still says `PR #____`.
- [x] The snippet runs exactly as written. Checked on 2026-10-07 after a clean `pip install llvq-tetra`: exit 0, no
      warning, "The capital of France is Paris." (`docs/mesures/hf-snippet-2026-10-07.txt`).
- [x] Someone can try it with one command. True for the default dense path, which needs no GPU and no compiler. Not
      true for the fused path, which compiles a kernel at import, and the draft says so.
- [x] Every number in the draft is checked against its journal. Done on 2026-10-20. Three statements were wrong and
      are fixed: FP16 and AWQ on GSM8K were scored in vLLM, not in our engine; the dense load takes two to three
      minutes, not two and a half to five; the Metal runs used torch 2.14.0 as well as 2.14.1.
- [ ] If 0.2.0 ships before the post, rewrite the Metal line of "What is missing". It describes 0.1.0.

## What the draft says on purpose

It presents the random-model problem as a `transformers` problem. `transformers` finds quantization methods by import,
not by entry point. So a model published in any out-of-tree format loads as a random model, without raising. This is
measured (`docs/mesures/hf-tripwire-2026-10-03.txt`) and is not specific to LLVQ. It is the best argument for going in
tree. If the answer is no, the draft asks for an entry point for out-of-tree quantizers, or an error on an unknown
`quant_method`. Either would fix it for every method.

It lists what is not done, six items, including the weakness of our own token check. A maintainer would find them in
ten minutes. Saying them first makes the rest easier to believe.

## If the answer is no

The out-of-tree route already works and is published. A refusal costs us discoverability and the random-model fix,
not the feature itself.
