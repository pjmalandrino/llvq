# Deviations. Preregistration dclm-14b-rowscales-2026-09-22

The prereg is stamped (sha256 `adbbe3e92d8ed11a...`) and not edited.

## E1. The training exits 1: the decision rule stops the chain before the fold

The loop ran all 9,507 steps, wrote `sigma.json` (40,590,065 B) and `journal.jsonl` to the bucket,
and closed with `first_loss` 0.2229 and `last_loss` 0.1399. `llvqtune` still returned 1, because
`Outcome.improved` compares the mean of the last third of the losses with the mean of the first
third (`domain/loop.py:79-95`), and here the last third sits **2.5 % above** the first: by eighths,
0.1522, 0.1537, 0.1554, 0.1657, 0.1543, 0.1528, 0.1590, 0.1560 (*computed* from the journal). Row
"training exits 1 (KL not improved)" applies: **stop before the fold, report**. Nothing was folded,
uploaded or scored. Cost: $9.78 for 7,040 s of h200 running time.

Two facts the row does not carry, for the operator's decision:

- **The measure is confounded here, and the code says so.** `improved`'s own docstring: "Only
  meaningful when the corpus repeats a fixed batch. On a streaming corpus each step reads different
  text, so two losses differ by the text as much as by the parameters." The DCLM corpus streams;
  `RepeatCorpus` exists to remove that confound and is not used by `train.sh`. The 8B run passed the
  same test (−2.0 % by eighths) on the same streaming corpus.
- **The 14B base starts much closer to its teacher.** First loss 0.2229 against 0.2741 at 8B, and
  the whole 14B curve sits at about 0.152-0.166 where the 8B sat at 0.187-0.198. Less to repair,
  and the trend within the run is flat.

What would settle it costs $1.80: fold on the Mac ($0, with its all-ones control), then the FT
census on l40sx1, paired against arm A on the same 14,042 questions. That is the operator's call,
against the stamped row above.

## E2. The gate on arm A could not fire: the base census is still queued

The prereg's preconditions replaced the "arm A ≥ 62.0 before launch" gate with a live one, to be
enforced while the training ran. Arm A never arrived: `census-14b-base` entered the l40sx1 queue at
20:55 on 2026-09-22 and was still in `SCHEDULING` when the training ended at 00:07, four and a half
hours later. So the training ran to its end without that gate ever being testable, and the risk the
preconditions took was not repaid by the information it was taken for.

## E3. The fold and the FT census run anyway, on the operator's explicit go

On 2026-09-23 at 08:45 the operator answered E1's open question — "envoie le mmlu FT" — and the
chain resumed **against** the stamped row "training exits 1 (KL not improved) → stop before the
fold". The decision is his, recorded here before the fold ran, and it does not amend the prereg:
the row stays what it says, and this file says the row was overridden and by whom.

What that buys and what it costs: the FT census is the only measurement that can tell whether the
flat KL curve means "the training did nothing" or "the `improved` gauge is confounded by the
streaming corpus" (E1). The two readings make opposite predictions on the same number. Cost $1.80
on l40sx1, 14B chain at $30.22 of its $52 before it.

**The signed prediction stands as written in the prereg** and is not restated here, because the
result is not yet known at the time of this line: paired FT − base on the 14,042 questions, and the
prereg's interval is the one that arbitrates. The two readings above, in that interval's terms:
"the training did nothing" is a gap whose CI95 contains 0; "the gauge is confounded" is a gap that
looks like the 8B's +3.29 pp and the 4B's +3.15 pp.

The all-ones control ran before the fold, as at 4B and 8B: `sigma` with every scale set to 1.0 must
reproduce the base byte for byte, or nothing is folded.
