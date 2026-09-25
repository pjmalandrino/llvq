# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub", "pyarrow", "transformers"]
# ///
"""Ship MMLU prompts to an engine that has no tokenizer: the jsonl `ops/gguf_mmlu_thin.py` reads.

The 2,280-question file of 2026-08-30 was produced by hand and never committed.
This file makes it again, for any reference dump, by the path that passed
2,280 / 2,280 qhash: `ops/vllm_score.py`'s `read_reference_dump` and
`build_prompts`, loaded by path as `ops/gguf_mmlu.py` does. It refuses to write
anything if a single qhash disagrees with the reference dump, so the shipped
prompts are the dump's questions byte for byte, not a re-selection.

    uv run ops/mmlu_prompts.py docs/data/mmlu-dumps/mmlu-4b-f16-FULL.csv out.jsonl

Each line: subject, index, qhash (16 hex), population, answer, prompt, the
fields and the order of the 2026-08-30 file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("vs", HERE / "vllm_score.py")
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    ref, out = Path(argv[0]), Path(argv[1])
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    wanted, head = vs.read_reference_dump(ref)
    test = vs.load_mmlu("test", "main")
    dev = vs.load_mmlu("dev", "main")
    populations: dict[str, int] = {}
    for it in test:
        populations[it["subject"]] = populations.get(it["subject"], 0) + 1

    prompts, items, bad = vs.build_prompts(wanted, test, dev, tok)
    if bad:
        print(f"REFUSED: {len(bad)} qhash disagree with {ref}; nothing written", file=sys.stderr)
        return 1
    with out.open("w", encoding="utf-8") as fh:
        for (subject, index, qhash), it, prompt in zip(wanted, items, prompts):
            row = {
                "subject": subject,
                "index": index,
                "qhash": f"{qhash:016x}",
                "population": populations[subject],
                "answer": it["answer"],
                "prompt": prompt,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"qhash checked {len(wanted)}/{len(wanted)} against {ref}")
    print(f"reference header: {head}")
    print(f"written {out}: {len(wanted)} prompts")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
