# Upstream proposal: LLVQ Tetra in `transformers`

**Nothing is posted.** [`ISSUE.md`](ISSUE.md) is a draft of the body, 157 lines,
and it stays editable until the day it is posted. On that day it becomes a
verbatim archive and stops being edited, which is the convention
[`../candle-broadcast-matmul/`](../candle-broadcast-matmul) already follows: the
point of the archive is that it matches what was posted.

## Why an issue and not a pull request

The two upstream documents disagree, and the difference matters.

`docs/source/en/quantization/contribute.md` lists ten code steps for a new
quantization method, beginning with "have a look at another quantization method
such as Finegrained Fp8", and **never mentions an issue** (read 2026-10-04).

`CONTRIBUTING.md` asks, for a new feature, to "open an issue and describe: the
motivation ... the feature in as much detail as possible, a code snippet
demonstrating the feature's usage, a link to the relevant paper".

So the issue comes first, and that is to our advantage rather than a formality.
An in-tree method is a maintenance commitment for them. Better to be refused on
thirty lines than on eight hundred, and if the answer is yes we have it in
writing before the ten steps are taken.

## What has to be true before it is posted

- [x] **`pip install llvq-tetra` has to work.** Done on 2026-10-07: 0.1.0 is on
      PyPI, and an empty environment installs it and loads a model
      (`docs/mesures/hf-pypi-0.1.0-2026-10-07.txt`).
- [ ] **The PR is ready.** The operator decided on 2026-10-07 that the issue and
      the PR go out together, so the draft below is rewritten to point at it.
- [ ] **A reader has to be able to try it in one command.** That holds today for
      the dense path, which needs no GPU and no compiler. It does not hold for the
      fused path, which compiles a kernel at import.
- [ ] Re-read every number against its journal. Each one in the draft has one,
      and a number in an issue is a number in public.

## What is deliberately in the draft

**The silent-load problem, stated as their problem.** `transformers` discovers
quantization methods by import and not by entry point, so a published model in
any out-of-tree format loads as a randomly initialized model without raising.
That is measured (`docs/mesures/hf-tripwire-2026-10-03.txt`), it is not specific
to this method, and it is the strongest argument for being in tree. The draft
also asks, in case the answer is no, whether entry-point discovery is something
they would consider, which would be a better outcome for everyone than one more
method in tree.

**Everything that is not done.** Six items, named, with the weakness of our own
gate among them. A maintainer finds those out in ten minutes; hearing them from us
first costs nothing and buys the only thing that matters in a review, which is
being believed on the rest.

## What a refusal would mean

The out-of-tree route already works and is published, so a no costs us
discoverability and the silent-load fix, not the feature. Paper 1 was
desk-rejected by TACO on scope with no technical objection, so the house position
on external gatekeepers is already written: the artifact stands on its own.
