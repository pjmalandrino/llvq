"""GSM8K-200 for an Inkling 2 Mini GGUF arm in llama-server, under the protocol of `bin/inkgsm`. Grades nothing.

    python3 -I ops/inkling/gsm8k_llamacpp.py <prompt_ref_200.json> <tokenizer.json> <out.jsonl> <arm label> <url>
            [--max-new 1536] [--limit 200]
    python3 -I ops/inkling/gsm8k_llamacpp.py --selftest

The prompts are not rebuilt here. `ops/inkling/gsm8k_prompt_ref.py --limit 200` wrote them as ids, rendered by the
checkpoint's own template at reasoning effort `none`; `tests/inkgsm.rs` holds `bin/inkgsm` to the same ids. They go
to llama-server as a token array (`/completion`, `prompt: [ids]`), so the engine renders and tokenizes nothing.

The protocol, as `bin/inkgsm` runs it:
- one request at a time (run llama-server with `-np 1`): batch 1, as our engine decodes;
- greedy (temperature 0, top_k 1);
- stop on `<|content_model_end_sampling|>` (200006), which the GGUF names its eos and eot (read in the headers of the
  two unsloth files, 2026-11-22), not counted;
- at most `--max-new` (1,536) generated tokens. `n_predict` is asked one higher, and a completion that holds
  `max_new` tokens is cut to them and marked `cap`, whatever followed: `bin/inkgsm` reads no token past the cap, so
  a stop arriving as token 1,537 is a cap on both sides.
The completion is decoded from the returned ids with the model's tokenizer.json, special tokens kept, as `inkgsm`
decodes. The dump has the `gsm8k::Row` fields with `extracted` null, `source` "ungraded" and `correct` false;
`inkgsm score` (or `gsm8kpair`) re-grades every row with `llvq_llm::gsm8k`, then applies the cap rule. The run
fingerprint is the FNV-1a of `llvq_llm::eval::token_fingerprint` over the concatenated prompt ids, the value `inkgsm`
writes, so the pairing refuses a dump asked other tokens. Resumes: problems already in `out.jsonl` are skipped.
Standard library only, plus `tokenizers` for the decode.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

DUMP_TAG, DUMP_VERSION = "llvq_gsm8k_dump", 1
STOP_ID = 200006
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def fingerprint(ids) -> str:
    h = 0xCBF29CE484222325
    for i in ids:
        for b in int(i).to_bytes(4, "little"):
            h ^= b
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def cut(tokens: list[int], max_new: int) -> tuple[list[int], str]:
    """The kept tokens and the stop, `bin/inkgsm`'s rule: stop id not kept, at most `max_new` kept."""
    out = []
    for t in tokens:
        if t == STOP_ID:
            return out, "eos"
        out.append(t)
        if len(out) >= max_new:
            return out, "cap"
    raise RuntimeError(f"the engine stopped after {len(out)} tokens with neither the stop id nor the cap")


def gen(url: str, ids: list[int], max_new: int) -> dict:
    body = json.dumps({
        "prompt": ids, "n_predict": max_new + 1, "temperature": 0.0, "top_k": 1, "seed": 0,
        "return_tokens": True, "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(url + "/completion", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def run(ref_path, tok_path, out_path, arm, url, max_new=1536, limit=200, decode=None, post=gen) -> int:
    ref = json.loads(Path(ref_path).read_text())
    assert ref["meta"]["effort"] == "none", "the reference prompts were not rendered at effort none"
    cases = ref["cases"][:limit]
    assert len(cases) == limit and [c["index"] for c in cases] == list(range(limit)), "not rows 0..limit-1"
    for c in cases:
        assert fingerprint(c["ids"]) == c["qhash"], f"problem {c['index']}: qhash"
        assert c["content"].endswith("\n" + INSTRUCTION)
    stream = [i for c in cases for i in c["ids"]]
    if decode is None:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tok_path))
        decode = lambda ids: tok.decode(ids, skip_special_tokens=False)  # noqa: E731
    header = {
        DUMP_TAG: DUMP_VERSION, "model": arm, "arithmetic": f"llama.cpp {arm}", "engine": "llama.cpp",
        "device": "cuda", "dtype": "gguf", "kv": "f16", "max_new": max_new, "think": False, "effort": "none",
        "limit": str(limit), "questions": limit, "selection": f"the first {limit} rows of the test split in parquet order",
        "dataset": "openai/gsm8k", "split": "test", "revision": "740312add88f781978c0658806c59bc2815b9866",
        "instruction": INSTRUCTION, "stop_ids": [STOP_ID], "graded": False, "prompt_stream": fingerprint(stream),
        "tokenizer_sha256": ref["meta"]["tokenizer_sha256"],
    }
    done = []
    out = Path(out_path)
    if out.exists() and out.read_text().strip():
        text = out.read_text()
        text = text[: text.rfind("\n") + 1]
        lines = [json.loads(l) for l in text.splitlines() if l.strip()]
        assert lines[0] == header, "the existing dump has another header: another arm or protocol"
        if any("end" in l for l in lines):
            print(f"{out}: finished")
            return 0
        done = lines[1:]
        out.write_text(text)
    else:
        out.write_text(json.dumps(header) + "\n")
    for k, r in enumerate(done):
        assert r["index"] == k and r["qhash"] == cases[k]["qhash"], f"dump row {k} is not problem {k}"
    with out.open("a") as w:
        for c in cases[len(done):]:
            t0 = time.time()
            res = post(url, c["ids"], max_new)
            wall = time.time() - t0
            kept, stop = cut(list(res["tokens"]), max_new)
            timings = res.get("timings", {})
            row = {
                "index": c["index"], "qhash": c["qhash"], "n_prompt": len(c["ids"]), "n_gen": len(kept),
                "stop": stop, "gold": c["gold"], "extracted": None, "source": "ungraded", "correct": False,
                "prefill_s": timings.get("prompt_ms", 0.0) / 1e3, "decode_s": timings.get("predicted_ms", wall * 1e3) / 1e3,
                "completion": decode(kept), "wall_s": wall,
            }
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            w.flush()
            print(f"  #{c['index']:<4} gen {len(kept):>4} {stop} {wall:.1f} s", file=sys.stderr)
        w.write(json.dumps({"end": True, "fingerprint": fingerprint(stream), "questions": limit}) + "\n")
    return 0


def selftest() -> int:
    """The cut rule and a whole run against a fake server, with no network and no model."""
    assert cut([5, 6, STOP_ID, 7], 10) == ([5, 6], "eos")
    assert cut([5, 6, 7, STOP_ID], 3) == ([5, 6, 7], "cap"), "a stop after the cap is a cap"
    assert cut([5, 6, STOP_ID], 3) == ([5, 6], "eos")
    assert cut([STOP_ID], 3) == ([], "eos")
    try:
        cut([5, 6], 3)
        raise AssertionError("an engine stop with neither stop id nor cap must be refused")
    except RuntimeError:
        pass
    assert fingerprint([1, 2]) == fingerprint([1, 2]) != fingerprint([2, 1])
    import tempfile
    d = Path(tempfile.mkdtemp())
    body = "Q?\n" + INSTRUCTION
    cases = [{"index": k, "content": body, "ids": [k + 1, 9], "qhash": fingerprint([k + 1, 9]), "gold": str(k)}
             for k in range(3)]
    (d / "ref.json").write_text(json.dumps({"meta": {"effort": "none", "tokenizer_sha256": "t"}, "cases": cases}))
    calls = []

    def post(url, ids, max_new):
        calls.append(ids)
        return {"tokens": [1, 2, STOP_ID] if ids[0] != 2 else [1] * (max_new + 1)}

    run(d / "ref.json", None, d / "out.jsonl", "fake", "http://x", max_new=4, limit=3,
        decode=lambda ids: " ".join(map(str, ids)), post=post)
    lines = [json.loads(l) for l in (d / "out.jsonl").read_text().splitlines()]
    assert [l.get("stop") for l in lines[1:4]] == ["eos", "cap", "eos"]
    assert lines[2]["n_gen"] == 4 and lines[-1]["fingerprint"] == fingerprint([1, 9, 2, 9, 3, 9])
    # Resume: a dump cut after its first row runs only the two others.
    text = (d / "out.jsonl").read_text().splitlines()
    (d / "out.jsonl").write_text("\n".join(text[:2]) + "\n" + text[2][:10])
    calls.clear()
    run(d / "ref.json", None, d / "out.jsonl", "fake", "http://x", max_new=4, limit=3,
        decode=lambda ids: " ".join(map(str, ids)), post=post)
    assert [c[0] for c in calls] == [2, 3], calls
    print("selftest: ok")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        sys.exit(selftest())
    opts = {"--max-new": 1536, "--limit": 200}
    pos = []
    while a:
        x = a.pop(0)
        if x in opts:
            opts[x] = int(a.pop(0))
        else:
            pos.append(x)
    if len(pos) != 5:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(run(*pos, max_new=opts["--max-new"], limit=opts["--limit"]))
