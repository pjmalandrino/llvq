"""Gate B of stage 1: two token dumps, compared as ids.

    uv run python -m llvqhf.comparetokens <a.json> <b.json>

One dump comes from `LLVQ_RUN_DUMP` on `bin/run`, the other from
`llvqhf.gentokens`. Both carry the prompt, its input ids and the generated ids.

Ids and not text. Greedy decoding is deterministic, so the two sequences must be
equal and not close; and a decoded string can hide a tokenizer difference, which
this comparison is not about.

When they differ, the position of the first difference is what matters: inside our
own engine the 14B kernel and dense paths agree for 77 tokens and part at 78, so
"they differ" is not a verdict on its own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    a, b = (json.loads(Path(p).read_text()) for p in argv[1:])
    print(f"a: {argv[1]}  dtype {a.get('dtype')}  n_new {a.get('n_new')}")
    print(f"b: {argv[2]}  dtype {b.get('dtype')}  n_new {b.get('n_new')}")
    if a.get("dtype") != b.get("dtype"):
        print("REFUSED: two dtypes are not one comparison", file=sys.stderr)
        return 2
    if len(a["prompts"]) != len(b["prompts"]):
        print(f"REFUSED: {len(a['prompts'])} prompts against {len(b['prompts'])}", file=sys.stderr)
        return 2
    bad = 0
    for pa, pb in zip(a["prompts"], b["prompts"]):
        if pa["prompt"] != pb["prompt"]:
            print(f"REFUSED: prompts differ, {pa['prompt']!r} against {pb['prompt']!r}",
                  file=sys.stderr)
            return 2
        if pa["input_ids"] != pb["input_ids"]:
            print(f"DIFFER on the tokenizer: {pa['prompt']!r}\n  a {pa['input_ids']}"
                  f"\n  b {pb['input_ids']}", file=sys.stderr)
            bad += 1
            continue
        x, y = pa["output_ids"], pb["output_ids"]
        if x == y:
            print(f"  same {len(x)} ids  {pa['prompt']!r}")
            continue
        first = next(i for i, (u, v) in enumerate(zip(x, y)) if u != v)
        print(f"DIFFER at token {first} of {len(x)}  {pa['prompt']!r}\n"
              f"  a {x[max(0, first - 3):first + 4]}\n  b {y[max(0, first - 3):first + 4]}",
              file=sys.stderr)
        bad += 1
    if bad:
        print(f"{bad} of {len(a['prompts'])} prompts differ", file=sys.stderr)
        return 1
    print(f"{len(a['prompts'])} prompts, every id identical")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
