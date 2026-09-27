# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers>=0.21", "pyarrow>=15", "huggingface_hub>=0.24"]
# ///
# Two ways to run this file. On the Mac, `uv run ops/gsm8k_vllm.py --check ...`
# rebuilds the prompts and checks them against a dump of `bin/gsm8k`; the
# header above gives uv what that needs, and vLLM is never imported. In the
# vLLM job image, `python3 ops/gsm8k_vllm.py ...` generates; the header is a
# comment there and the image provides every import.
"""GSM8K for a reference arm (FP16, AWQ) **in its own engine**, vLLM.

## Why this file exists

`bin/gsm8k` scores our sealed files through the served kernel, batch 1. The
reference arms are not LLVQ files: their served engine is vLLM, the engine
their speed rows in `docs/ETAT.md` come from. This script generates their
answers there, on the same problems and the same prompt token ids.

## One grader, not two

This script writes raw completions and grades nothing. Its dump says
`"graded": false`, and `bin/gsm8kpair` re-grades every row of both dumps with
the Rust rules of `llvq_llm::gsm8k` before pairing. A second implementation of
the extraction would be a second thing to keep in step, and the first place it
would drift is the comparison it exists to make.

## Same prompts, proved rather than declared

The prompt is built from ids, as `llvq_llm::chatfmt` builds it, and handed to
vLLM as `TokensPrompt`, so no re-tokenization happens in the engine. Each
problem's `qhash` is the FNV-1a of `llvq_llm::eval::token_fingerprint`. With
`--check <dump>`, the prompts are compared to a dump of `bin/gsm8k` problem by
problem and the run refuses on a single mismatch. `bin/gsm8kpair` checks every
row again at pairing, and the run fingerprint with it.

## What transfers across engines

Not proved for a generation. MMLU transfers in level between candle and vLLM
(`ops/vllm_score.py`, gate green at 0.02 and 0.22 pp on 2026-08-30). A greedy
chain of 300 tokens diverges between engines within a few tokens, and only the
accuracy can transfer. That is what the FP16 arm run in both engines measures,
before any cross-engine gap is read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# --- constants shared with the Rust side ------------------------------------

# `llvq_llm::gsm8k::INSTRUCTION`.
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
# `llvq_llm::gsm8k::SAMPLE_SEED`.
SAMPLE_SEED = 0x0065_736D_386B
DUMP_TAG = "llvq_gsm8k_dump"
DUMP_VERSION = 1
GSM8K_REPO = "openai/gsm8k"
GSM8K_FILE = "main/test-00000-of-00001.parquet"

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
U64 = (1 << 64) - 1


class Refused(Exception):
    """Configuration refused. Nothing has been generated."""


def token_fingerprint(ids) -> int:
    """`llvq_llm::eval::token_fingerprint`: FNV-1a 64 over little-endian u32."""
    h = FNV_OFFSET
    for tid in ids:
        for b in int(tid).to_bytes(4, "little"):
            h = ((h ^ b) * FNV_PRIME) & U64
    return h


def splitmix64(state: int):
    """`llvq_core::SplitMix64`: yields the stream seeded by `state`."""
    while True:
        state = (state + 0x9E3779B97F4A7C15) & U64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & U64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & U64
        yield z ^ (z >> 31)


def select(n: int, limit: int) -> list[int]:
    """`llvq_llm::gsm8k::select`: census in order, else a seeded sample, sorted."""
    if limit >= n:
        return list(range(n))
    idx = list(range(n))
    rng = splitmix64(SAMPLE_SEED)
    for i in range(n - 1, 0, -1):
        j = next(rng) % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return sorted(idx[:limit])


def self_test() -> None:
    assert token_fingerprint([]) == FNV_OFFSET
    h = FNV_OFFSET
    for b in (1, 0, 0, 0):
        h = ((h ^ b) * FNV_PRIME) & U64
    assert token_fingerprint([1]) == h
    # SplitMix64's reference first output for seed 0 (Vigna, splitmix64.c).
    # The tie to the Rust generator is `--check`: a different stream draws a
    # different sample, and the index sets then refuse to match.
    assert next(splitmix64(0)) == 0xE220A8397B1DCDAF
    assert select(10, 20) == list(range(10))


# --- data and prompts ---------------------------------------------------------


def load_gsm8k(revision: str) -> tuple[list[dict], str]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(GSM8K_REPO, GSM8K_FILE, repo_type="dataset", revision=revision)
    table = pq.read_table(path)
    rows = table.to_pylist()
    for r in rows:
        if "question" not in r or "answer" not in r:
            raise Refused("gsm8k/test: a row is missing question or answer")
    parts = Path(path).parts
    commit = parts[parts.index("snapshots") + 1] if "snapshots" in parts else revision
    return rows, commit


class Marks:
    """`llvq_llm::chatfmt::Marks`, from a `tokenizers.Tokenizer`."""

    def __init__(self, tok):
        def tid(s: str) -> int:
            v = tok.token_to_id(s)
            if v is None:
                raise Refused(f"the tokenizer has no {s!r}; is this a Qwen3 tokenizer?")
            return v

        self.im_start = tid("<|im_start|>")
        self.im_end = tid("<|im_end|>")
        self.eot = tid("<|endoftext|>")
        self.nl = tok.encode("\n", add_special_tokens=False).ids[0]
        self.nl2 = tok.encode("\n\n", add_special_tokens=False).ids[0]
        o, c = tok.token_to_id("<think>"), tok.token_to_id("</think>")
        self.think = (o, c) if o is not None and c is not None else None


def prompt_ids(tok, m: Marks, question: str, think: bool) -> list[int]:
    """`turn(user, user_body(q))` then `open_assistant(think)`, as ids."""
    body = f"{question.strip()}\n{INSTRUCTION}"
    v = [m.im_start]
    v += tok.encode("user", add_special_tokens=False).ids
    v.append(m.nl)
    v += tok.encode(body, add_special_tokens=False).ids
    v += [m.im_end, m.nl]
    v.append(m.im_start)
    v += tok.encode("assistant", add_special_tokens=False).ids
    v.append(m.nl)
    if not think and m.think is not None:
        v += [m.think[0], m.nl2, m.think[1], m.nl2]
    return v


def read_dump(path: Path) -> tuple[dict, list[dict]]:
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    head = json.loads(lines[0])
    if head.get(DUMP_TAG) != DUMP_VERSION:
        raise Refused(f"{path} is not a GSM8K dump of version {DUMP_VERSION}")
    rows = [json.loads(l) for l in lines[1:]]
    if not rows or "end" not in rows[-1]:
        raise Refused(f"{path} has no trailer: the run did not finish")
    return head, rows[:-1]


def gold_of(answer: str) -> str:
    """The text after the last `####`, commas removed. `bin/gsm8kpair` reduces
    it with the Rust `canonical` and refuses a pair whose two golds differ."""
    return answer.rsplit("####", 1)[-1].strip().replace(",", "")


# --- main ---------------------------------------------------------------------


def main(argv: list[str]) -> int:
    self_test()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, help="f16 | awq_marlin, printed on every line")
    ap.add_argument("--model", required=True, help="repository, generation weights and tokenizer")
    ap.add_argument("--revision", required=True, help="pinned revision of --model")
    ap.add_argument("--quantization", default=None, help="passed to vLLM, e.g. awq_marlin")
    ap.add_argument("--dataset-rev", default="main", help="revision of openai/gsm8k")
    ap.add_argument("--limit", type=int, default=10**9, help="sample size, census by default")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--check", default=None, help="a bin/gsm8k dump to compare the prompts to")
    ap.add_argument("--check-only", action="store_true", help="build and check, generate nothing")
    ap.add_argument("--out", default=None, help="dump path (required unless --check-only)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args(argv)

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(
        hf_hub_download(args.model, "tokenizer.json", revision=args.revision)
    )
    marks = Marks(tok)
    items, dataset_commit = load_gsm8k(args.dataset_rev)
    picked = select(len(items), args.limit)
    prompts = [prompt_ids(tok, marks, items[i]["question"], args.think) for i in picked]
    stream = [t for p in prompts for t in p]
    fingerprint = f"{token_fingerprint(stream):016x}"
    print(f"gsm8k_vllm, arm {args.arm}, {args.model}@{args.revision}")
    print(f"  problems {len(picked)} of {len(items)}, dataset {dataset_commit}, tokens {fingerprint}")

    if args.check:
        head, rows = read_dump(Path(args.check))
        if head.get("max_new") != args.max_new or bool(head.get("think")) != args.think:
            raise Refused(
                f"--check dump has max_new {head.get('max_new')} think {head.get('think')}, "
                f"this run {args.max_new} {args.think}"
            )
        by_index = {i: p for i, p in zip(picked, prompts)}
        bad = []
        for r in rows:
            p = by_index.get(r["index"])
            got = None if p is None else f"{token_fingerprint(p):016x}"
            if got != r["qhash"]:
                bad.append((r["index"], r["qhash"], got))
        if bad:
            for i, want, got in bad[:5]:
                print(f"    problem {i}: dump {want}, rebuilt {got}")
            raise Refused(f"{len(bad)} of {len(rows)} prompts differ from {args.check}")
        print(f"  prompts checked       {len(rows)}/{len(rows)} identical to {args.check}")
        if len(rows) == len(picked):
            end = json.loads([l for l in Path(args.check).read_text().splitlines() if l.strip()][-1])
            if end.get("fingerprint") != fingerprint:
                raise Refused(f"run fingerprint {fingerprint} against {end.get('fingerprint')}")
            print("  run fingerprint       identical")
    if args.check_only:
        return 0
    if not args.out:
        raise Refused("--out is required to generate")

    from vllm import LLM, SamplingParams

    try:
        from vllm.inputs import TokensPrompt
    except ImportError:  # the dict form every vLLM version accepts
        def TokensPrompt(prompt_token_ids):  # noqa: N802
            return {"prompt_token_ids": prompt_token_ids}

    # The arguments of ops/awq_speed.py, which ran in this image: f16 on a bf16
    # checkpoint so the witness is an f16 witness, the tokenizer at the weights'
    # revision, no prefix cache, one card, no remote code.
    kwargs = dict(
        model=args.model,
        revision=args.revision,
        tokenizer_revision=args.revision,
        dtype="float16",
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
        seed=0,
        disable_log_stats=True,
        trust_remote_code=False,
    )
    if args.quantization:
        kwargs["quantization"] = args.quantization
    llm = LLM(**kwargs)
    stops = [marks.im_end, marks.eot]
    # `top_k` is not passed, as in awq_speed.py: its "disabled" value moved across
    # vLLM versions, and temperature 0 already selects the greedy path.
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        n=1,
        max_tokens=args.max_new,
        stop_token_ids=stops,
        skip_special_tokens=False,
    )
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
    wall = time.time() - t0

    out = Path(args.out)
    n_gen_total = 0
    with out.open("w") as w:
        head = {
            DUMP_TAG: DUMP_VERSION,
            "model": f"{args.model}@{args.revision} [vLLM {args.arm}]",
            "arithmetic": f"vllm {args.arm}",
            "engine": "vllm",
            "device": "cuda",
            "dtype": "f16",
            "kv": "vllm",
            "max_new": args.max_new,
            "think": args.think,
            "limit": "census" if len(picked) == len(items) else str(len(picked)),
            "questions": len(picked),
            "dataset": GSM8K_REPO,
            "split": "test",
            "revision": dataset_commit,
            "instruction": INSTRUCTION,
            "sample_seed": hex(SAMPLE_SEED),
            "graded": False,
            "wall_s": round(wall, 3),
        }
        w.write(json.dumps(head) + "\n")
        for i, p, o in zip(picked, prompts, outs):
            c = o.outputs[0]
            ids = list(c.token_ids)
            stop = "eos" if ids and ids[-1] in stops else "cap"
            if stop == "eos":
                ids = ids[:-1]
            if stop == "cap" and len(ids) < args.max_new:
                raise Refused(f"problem {i}: {len(ids)} tokens, no stop token, under the cap")
            n_gen_total += len(ids)
            row = {
                "index": i,
                "qhash": f"{token_fingerprint(p):016x}",
                "n_prompt": len(p),
                "n_gen": len(ids),
                "stop": stop,
                "gold": gold_of(items[i]["answer"]),
                "extracted": None,
                "source": "ungraded",
                "correct": False,
                "prefill_s": 0.0,
                "decode_s": 0.0,
                "completion": tok.decode(ids, skip_special_tokens=False),
            }
            w.write(json.dumps(row) + "\n")
        w.write(json.dumps({"end": True, "fingerprint": fingerprint, "questions": len(picked)}) + "\n")
    print(f"  generated {n_gen_total} tokens in {wall:.1f} s, {n_gen_total / max(wall, 1e-9):.0f} tok/s batched")
    print(f"  dump {out}: ungraded, score it with `gsm8kpair`")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)
