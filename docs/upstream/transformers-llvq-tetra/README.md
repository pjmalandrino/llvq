# Upstream proposal: LLVQ Tetra in `transformers`

**Nothing is posted.** [`ISSUE.md`](ISSUE.md) is a draft of the body, 102 lines,
and it stays editable until the day it is posted. On that day it becomes a
verbatim archive and stops being edited, which is the convention
[`../candle-broadcast-matmul/`](../candle-broadcast-matmul) already follows: the
point of the archive is that it matches what was posted.

## Why the issue and the PR go together

The two upstream documents disagree. `docs/source/en/quantization/contribute.md`
lists ten code steps for a new quantization method and never mentions an issue
(read 2026-10-04). `CONTRIBUTING.md` asks for an issue before a new feature,
with the motivation, the detail, a code snippet and the paper.

The operator decided on 2026-10-07 to post both at once: the issue says why, the
PR shows the code. The draft of 2026-10-04 asked first and built after; it was
rewritten on 2026-10-07, shorter, for a PR beside it.

## What has to be true before it is posted

- [x] **`pip install llvq-tetra` has to work.** Done on 2026-10-07: 0.1.0 is on
      PyPI, and an empty environment installs it and loads a model
      (`docs/mesures/hf-pypi-0.1.0-2026-10-07.txt`).
- [ ] **The PR is ready.** The operator decided on 2026-10-07 that the issue and
      the PR go out together. The draft carries `PR #____` until then.
- [x] **Run the snippet once, exactly as written.** Done on 2026-10-07 from a
      clean `pip install llvq-tetra`: exit 0, no warning, "The capital of France is
      Paris." (`docs/mesures/hf-snippet-2026-10-07.txt`).
- [x] **A reader has to be able to try it in one command.** True for the dense
      path, which needs no GPU and no compiler. Not for the fused path, which
      compiles a kernel at import, and the draft says so.
- [ ] Re-read every number against its journal. Each one in the draft has one,
      and a number in an issue is a number in public.

## What is deliberately in the draft

**The silent-load problem, stated as their problem.** `transformers` discovers
quantization methods by import and not by entry point, so a published model in
any out-of-tree format loads as a randomly initialized model without raising.
That is measured (`docs/mesures/hf-tripwire-2026-10-03.txt`), it is not specific
to this method, and it is the strongest argument for being in tree. If the answer
is no, the draft asks for an entry point for out-of-tree quantizers or an error on
an unknown `quant_method`. Either fixes it for every method, not only this one.

**Everything that is not done.** Six items, named, with the weakness of our own
gate among them. A maintainer finds those out in ten minutes; hearing them from us
first costs nothing and buys the only thing that matters in a review, which is
being believed on the rest.

## What a refusal would mean

The out-of-tree route already works and is published, so a no costs us
discoverability and the silent-load fix, not the feature. Paper 1 was
desk-rejected by TACO on scope with no technical objection, so the house position
on external gatekeepers is already written: the artifact stands on its own.
