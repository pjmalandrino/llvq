# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow>=16", "huggingface-hub>=0.25"]
# ///
"""Does `auxiliary_train` overlap the splits the MMLU harness scores?

The task-format arm of the fine-tuning ladder (`ops/llvqtune`, `--corpus
mmlu-aux`) distils the teacher on multiple-choice prompts drawn from
`cais/mmlu`'s `auxiliary_train`. That is only a lead if it is not
contamination, and the difference between the two is measured here, not
argued.

Three splits are scored by `llvq-llm --bin mmlu`: `test` is the bar, `dev`
supplies the five worked examples every prompt carries, and `validation` is
held for tie-breaks. A question that sits in both `auxiliary_train` and any of
them makes the resulting MMLU number meaningless.

Two comparisons, because they answer different questions:

* **question text alone** — the strict reading, and it over-counts: MMLU is
  full of bare stems ("Which of the following is true?") shared by unrelated
  items;
* **question and its four choices** — the same item, in substance.

Measured 2026-09-23 on `cais/mmlu` revision c30699e:

```text
rows: aux 99,842 · test 14,042 · dev 285 · validation 1,531
test        question-only 14 (0.10 %)   question+choices 0
dev         question-only  0 (0.00 %)   question+choices 0
validation  question-only  1 (0.07 %)   question+choices 0
```

All 14 are bare stems under four unrelated options. So the training split is
disjoint in substance from every scored split, and a filter that dropped the
question-only matches would drop nothing that carries an answer.

Exit code 1 when any question+choices match is found, so this can gate a job.

    uv run ops/mmlu_aux_overlap.py
"""

from __future__ import annotations

import hashlib
import re
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "cais/mmlu"
TRAIN = "auxiliary_train"
SCORED = ("test", "dev", "validation")


def load(split: str) -> list[dict]:
    path = hf_hub_download(
        REPO, f"all/{split}-00000-of-00001.parquet", repo_type="dataset"
    )
    return pq.read_table(path).to_pylist()


def key(row: dict, with_choices: bool) -> str:
    text = row["question"]
    if with_choices:
        text += " || " + " | ".join(row["choices"])
    return hashlib.sha256(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()


def main() -> int:
    aux = load(TRAIN)
    questions = {key(r, False) for r in aux}
    items = {key(r, True) for r in aux}
    print(f"rows: {TRAIN} {len(aux):,}")

    worst = 0
    for split in SCORED:
        rows = load(split)
        q = sum(1 for r in rows if key(r, False) in questions)
        f = sum(1 for r in rows if key(r, True) in items)
        worst = max(worst, f)
        print(
            f"{split:<11} {len(rows):>6,} rows · question-only {q:>3} "
            f"({100 * q / len(rows):.2f} %) · question+choices {f}"
        )
    if worst:
        print("\nREFUSED: a scored item appears in the training split")
        return 1
    print("\nclean: no scored item appears in the training split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
