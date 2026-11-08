"""GSM8K for a GGUF arm in its own engine, llama-server. Writes raw completions, grades nothing.

The llama.cpp twin of `ops/gsm8k_vllm.py`. The prompts are not rebuilt here:
`ops/duel/inputs.py` wrote them as ids into `gsm8k-prompts.jsonl`, each checked
against the qhash of our 4B census dump. They go to llama-server as a token array,
so the engine tokenizes nothing. Greedy (temperature 0, top_k 1), at most 1,024
new tokens, stopped on <|im_end|> or <|endoftext|>, as `bin/gsm8k` and
`gsm8k_vllm.py` run.

The dump has the format `gsm8k_vllm.py` writes (`"graded": false`), so
`bin/gsm8kpair` re-grades every row with the Rust rules before pairing: one
grader for every arm. The completion is decoded with the `tokenizers` library
from the model's own tokenizer.json, skip_special_tokens=False, as the vLLM arm
decodes.

llama-server runs its slots in parallel (continuous batching), like vLLM did for
the reference arms. A greedy chain can then differ from a batch-1 chain on a
few problems; the accuracy is what is compared, as for vLLM.

Usage:
    python3 gsm8k_llamacpp.py <prompts.jsonl> <tokenizer.json> <out.jsonl> <arm label> <url> <workers>
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MAX_NEW = 1024
DUMP_TAG, DUMP_VERSION = "llvq_gsm8k_dump", 1


def gen(url: str, ids: list[int]) -> dict:
    body = json.dumps({
        "prompt": ids,
        "n_predict": MAX_NEW,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 0,
        "return_tokens": True,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(url + "/completion", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def main(argv: list[str]) -> int:
    prompts_path, tok_path, out_path, arm, url, workers = argv[0], argv[1], argv[2], argv[3], argv[4], int(argv[5])
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    lines = [json.loads(l) for l in open(prompts_path, encoding="utf-8").read().split("\n") if l.strip()]
    head, rows = lines[0], lines[1:]
    stops = set(head["stops"])
    assert len(rows) == head["questions"], "prompts file truncated"
    print(f"gsm8k_llamacpp, arm {arm}: {len(rows)} problems, run fingerprint {head['fingerprint']}, {workers} slots")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(lambda r: gen(url, r["ids"]), rows))
    wall = time.time() - t0

    n_gen_total = 0
    with open(out_path, "w", encoding="utf-8") as w:
        w.write(json.dumps({
            DUMP_TAG: DUMP_VERSION,
            "model": arm,
            "arithmetic": f"llama.cpp {arm}",
            "engine": "llama.cpp",
            "device": "cuda",
            "dtype": "gguf",
            "kv": "f16",
            "max_new": MAX_NEW,
            "think": head["think"],
            "limit": "census",
            "questions": len(rows),
            "dataset": head["dataset"],
            "split": "test",
            "revision": head["revision"],
            "instruction": head["instruction"],
            "sample_seed": "0x65736d386b",
            "graded": False,
            "wall_s": round(wall, 3),
            "slots": workers,
        }) + "\n")
        for r, o in zip(rows, outs):
            ids = list(o.get("tokens") or [])
            st = o.get("stop_type")
            if ids and ids[-1] in stops:
                ids = ids[:-1]
            if st == "eos":
                stop = "eos"
            elif st == "limit":
                stop = "cap"
                if len(ids) != MAX_NEW:
                    raise SystemExit(f"problem {r['index']}: stop limit at {len(ids)} tokens")
            else:
                raise SystemExit(f"problem {r['index']}: stop_type {st!r}")
            n_gen_total += len(ids)
            w.write(json.dumps({
                "index": r["index"],
                "qhash": r["qhash"],
                "n_prompt": len(r["ids"]),
                "n_gen": len(ids),
                "stop": stop,
                "gold": r["gold"],
                "extracted": None,
                "source": "ungraded",
                "correct": False,
                "prefill_s": 0.0,
                "decode_s": 0.0,
                "completion": tok.decode(ids, skip_special_tokens=False),
            }) + "\n")
        w.write(json.dumps({"end": True, "fingerprint": head["fingerprint"], "questions": len(rows)}) + "\n")
    print(f"  generated {n_gen_total} tokens in {wall:.1f} s, {n_gen_total / max(wall, 1e-9):.0f} tok/s aggregate")
    print(f"  dump {out_path}: ungraded, score it with `gsm8kpair`")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
