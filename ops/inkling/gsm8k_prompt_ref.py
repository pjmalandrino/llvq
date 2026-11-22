# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tokenizers",
#   "jinja2>=3.1.6",
#   "transformers @ git+https://github.com/huggingface/transformers@046bdb8cc2",
# ]
# ///
"""The prompt ids of GSM8K problems under Inkling 2 Mini's own chat template, as the reference for `bin/inkgsm`.

    uv run ops/inkling/gsm8k_prompt_ref.py [--native DIR] [--problems FILE] [--limit N] [--effort none] [--out FILE]

Defaults: DIR = ~/llvq-inkling-mini/native, FILE = ops/data/spec-gsm8k-200.jsonl, N = 3, effort `none`,
out = ~/llvq-inkling-mini/lot4-g8/gsm8k_prompt_ref.json.

Each problem is one user message whose content is the row's `prompt` (the question, a newline, the house
instruction), rendered by `apply_chat_template(..., reasoning_effort=EFFORT, tokenize=True,
sanitize_control_tokens=True)`, the way `ops/inkling/chat_ref.py` writes the reference of `llvq_llm::inkchat`. The
other arms of the GSM8K comparison (transformers, llama.cpp) take their prompt ids from this script, so every arm
is asked the same tokens. No model weights are read: tokenizer and template.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

D = Path(os.environ.get("INKLING_DATA", str(Path.home() / "llvq-inkling-mini")))
REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(ids):
    """`llvq_llm::eval::token_fingerprint`: FNV-1a 64 over the little-endian bytes of each u32 id."""
    h = 0xCBF29CE484222325
    for i in ids:
        for b in int(i).to_bytes(4, "little"):
            h ^= b
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", default=str(D / "native"))
    ap.add_argument("--problems", default=str(REPO / "ops" / "data" / "spec-gsm8k-200.jsonl"))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--effort", default="none")
    ap.add_argument("--out", default=str(D / "lot4-g8" / "gsm8k_prompt_ref.json"))
    a = ap.parse_args()

    import transformers
    from transformers import AutoTokenizer

    native = Path(a.native)
    tok = AutoTokenizer.from_pretrained(str(native))
    assert tok.chat_template == (native / "chat_template.jinja").read_text(), "chat_template.jinja not loaded"

    rows = [json.loads(l) for l in open(a.problems) if l.strip()]
    rows = [r for r in rows if "_comment" not in r][: a.limit]
    assert len(rows) == a.limit, f"{len(rows)} problems, {a.limit} asked"
    cases = []
    for k, r in enumerate(rows):
        assert r["id"] == f"gsm8k-test-{k:04d}", r["id"]
        msgs = [{"role": "user", "content": r["prompt"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, reasoning_effort=a.effort)
        ids = tok.apply_chat_template(msgs, tokenize=True, return_dict=False, sanitize_control_tokens=True,
                                      reasoning_effort=a.effort)
        ids = [int(i) for i in ids]
        cases.append({"index": k, "content": r["prompt"], "text": text, "ids": ids, "qhash": fingerprint(ids),
                      "gold": r["gold"]})
        print(f"{r['id']}: {len(ids)} ids, qhash {fingerprint(ids)}", file=sys.stderr)
    meta = {
        "transformers": transformers.__version__,
        "effort": a.effort,
        "chat_template_sha256": sha256(native / "chat_template.jinja"),
        "tokenizer_sha256": sha256(native / "tokenizer.json"),
        "problems_sha256": sha256(a.problems),
        "eos_token_id": tok.eos_token_id,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"meta": meta, "cases": cases}, f, ensure_ascii=False, indent=1)
    print(f"{len(cases)} cases -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
