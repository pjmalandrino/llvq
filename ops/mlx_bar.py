# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "pyarrow==26.0.1", "transformers==5.22.1"]
# ///
"""The rival on this Mac: mlx-lm's Qwen3-4B 4-bit, timed like `metalrun time`.

Same prompts as `metalrun time`: "The capital of France is" (5 tokens), or the
first N tokens of the WikiText-2 test split built the way `metalrun` builds
them (rows joined by a blank line, the first 12 N characters, tokenized without
special tokens, cut at N). The token digest is printed in `metalrun`'s form, so
the two prompts can be compared by their sha256.

Greedy, a fixed number of new tokens (no stop on EOS, as `metalrun`), a fresh
cache each round, 2 discarded rounds then the timed ones. Decode tok/s is the
tokens after the first over the time after the first, `metalrun`'s accounting.

    uv run ops/mlx_bar.py <model_dir> <n_new> <timed_rounds> [prompt_tokens=0]
"""
import hashlib
import os
import statistics
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

# The WikiText-2 test split at the revision `metalrun` reads, from the Hugging
# Face cache ($HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub).
# Nothing here downloads it: `wiki_parquet` names the command when it is missing.
WIKI_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKI_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def wiki_parquet() -> Path:
    hub = os.environ.get("HF_HUB_CACHE") or (
        str(Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None
    )
    hub = Path(hub) if hub else Path.home() / ".cache/huggingface/hub"
    path = hub / "datasets--Salesforce--wikitext" / "snapshots" / WIKI_REVISION / WIKI_FILE
    if not path.is_file():
        sys.exit(
            f"no WikiText-2 test split at {path}\n"
            f"It is the dataset Salesforce/wikitext, config wikitext-2-raw-v1, revision {WIKI_REVISION[:8]}.\n"
            f"Fetch it once into the Hugging Face cache:\n"
            f"  hf download Salesforce/wikitext --repo-type dataset --revision {WIKI_REVISION} "
            f"--include '{WIKI_FILE}'"
        )
    return path
PROMPT = "The capital of France is"
DISCARDED = 2


def prompt_ids(tok, n):
    if n == 0:
        return tok.encode(PROMPT, add_special_tokens=False)
    import pyarrow.parquet as pq

    rows = pq.read_table(wiki_parquet()).column("text").to_pylist()
    head = "\n\n".join(rows)[: n * 12]
    ids = tok.encode(head, add_special_tokens=False)
    assert len(ids) >= n, f"{n} prompt tokens asked, {len(ids)} found"
    return ids[:n]


def sha_u32(ids):
    return hashlib.sha256(b"".join(struct.pack("<I", i) for i in ids)).hexdigest()


def one_round(model, ids, n_new):
    prompt = mx.array(ids)
    out = []
    t0 = time.perf_counter()
    t1 = None
    for (tok, _), _ in zip(generate_step(prompt, model, max_tokens=n_new), range(n_new)):
        out.append(int(tok))
        if t1 is None:
            t1 = time.perf_counter()
    t2 = time.perf_counter()
    first = t1 - t0
    decode = (len(out) - 1) / (t2 - t1)
    door = len(out) / (t2 - t0)
    return first, decode, door, out


def main():
    model_dir, n_new, timed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    n_prompt = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    print(f"mlx {mx.__version__} | device {mx.default_device()} | model {model_dir}")
    t = time.perf_counter()
    model, tok = load(model_dir)
    print(f"loaded in {time.perf_counter() - t:.1f} s")
    ids = prompt_ids(tok, n_prompt)
    print(f"prompt {len(ids)} tokens, sha256 {sha_u32(ids)} (u32 little-endian); {n_new} new; "
          f"{DISCARDED} discarded + {timed} timed rounds")
    firsts, decodes, doors, ref = [], [], [], None
    for r in range(DISCARDED + timed):
        first, decode, door, out = one_round(model, ids, n_new)
        tag = "(discarded) " if r < DISCARDED else ""
        print(f"round {r} {tag}door {door:7.2f} tok/s | decode {decode:7.2f} tok/s | first token {first * 1e3:8.1f} ms")
        if r >= DISCARDED:
            firsts.append(first)
            decodes.append(decode)
            doors.append(door)
        if ref is None:
            ref = out
        elif out != ref:
            print("WARNING: tokens differ from round 0")
    med = statistics.median
    print(f"--- medians over {timed} timed rounds [min-max], measured on this Mac")
    print(f"mlx-lm  door {med(doors):7.2f} tok/s [{min(doors):.2f}-{max(doors):.2f}] | "
          f"decode {med(decodes):7.2f} tok/s [{min(decodes):.2f}-{max(decodes):.2f}] | "
          f"{1e3 / med(decodes):.2f} ms a decode token | first token {med(firsts) * 1e3:.1f} ms "
          f"[{min(firsts) * 1e3:.1f}-{max(firsts) * 1e3:.1f}]")
    print(f"peak memory {mx.get_peak_memory() / 1e9:.2f} GB")
    print(f"tokens sha256 {sha_u32(ref)}")
    print("text:", repr(tok.decode(ref)[:200]))


if __name__ == "__main__":
    main()
