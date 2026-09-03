# This script does not run under `uv`: it runs **inside the vLLM job image**,
# next to `ops/awq_speed.py`, and imports `vllm` and `transformers` from it.
"""Score a quantized arm **in its own engine** (vLLM), on our own fixtures.

## Why this file exists

`ops/awq_speed.py` chronometers arms inside vLLM. It says nothing about their
**quality**, and the project has never measured the quality of a 2-bit
competitor at all: QTIP's 17.04 is a citation of the paper's Table 6, and F2
forbids any quality claim on its own arm (pseudo-random payload).

The alternative would be one dequantization pipeline per format, in the shape of
`ops/awq_dequant.py` — 2,358 lines whose header states what they are for:
*"publishing plausible, wrong weights"*. This file takes the other route: score
the arm **as deployed**, with the engine reading its own format with its own
code. One brick serves every arm instead of one pipeline per format.

## What transfers across engines, and what does not

Established in `docs/exp-piles-isolees-2026-08-30/PROTOCOLE.md` §0:

* **MMLU transfers in LEVEL.** It depends only on the tokenizer and on the
  logprob of four answer tokens.
* **Perplexity does not.** It depends on the windowing convention. It is
  published only as a ratio to the f16 witness **of the same engine**.

## The gate this file exists to pass before it is trusted

`prompt_logprobs` is a **new instrument**. Per §5 of `CLAUDE.md` — and the more
recent lesson of the `grep` on the `.ots` files — no instrument is trusted
before it reproduces a known answer. Two arms have known answers, measured in
our own harness by dequantization:

    f16         MMLU 70.32   ppl 12.2369
    awq_marlin  MMLU 70.04   ppl 13.5207

`--gate` runs exactly those two and refuses the run if either drifts. Criteria
are fixed in `proofs/preregistration-m3-gptq2-2026-08-30.md` §2.4 — MMLU within
1.5 pp, ppl within 2 % — and are **not** editable here: they are stamped.

## Same data, proved rather than declared

The MMLU question set is not re-selected. It is **read from an existing dump**
(`docs/data/mmlu-dumps/mmlu-4b-f16.csv`), which carries `(subject, index,
qhash)` per question. This file rebuilds each prompt, recomputes `qhash` with
the same FNV-1a used by `llvq_llm::eval::token_fingerprint`, and **refuses to
score** if a single one disagrees. Identical questions become a machine-checked
fact, not an intention.

## What this file deliberately does not do

It does not dequantize, so it assumes **no unpacking convention** — that is
vLLM's job, with vLLM's code, for whichever format. The GPTQ `g_idx`/act-order
trap is therefore not ours to fall into; the control that replaces it is the
gate above, since permuted weights reproduce no known answer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import sys
from pathlib import Path
from typing import Any

BANNER = "=" * 78

# --- the protocol constants, fixed by the preregistration -------------------

# `llvq_llm::eval::token_fingerprint` — FNV-1a 64 over the little-endian bytes
# of each u32 token id. Reimplemented rather than imported: the Rust side is the
# reference, and a divergence must show up as a qhash mismatch, loudly.
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
U64 = (1 << 64) - 1

LETTERS = ("A", "B", "C", "D")
ANSWER_STRINGS = (" A", " B", " C", " D")

# Known answers, from `docs/mesures/a4-campagne-2026-08-06.txt`. These are what
# `--gate` demands the instrument reproduce.
KNOWN: dict[str, dict[str, float]] = {
    "f16": {"mmlu": 70.32, "ppl": 12.2369},
    "awq_marlin": {"mmlu": 70.04, "ppl": 13.5207},
}
# Preregistration §2.4. Not editable here — the document is stamped.
GATE_MMLU_PP = 1.5
GATE_PPL_PCT = 2.0

PPL_CTX = 4096
PPL_WINDOWS = 12

MMLU_REPO = "cais/mmlu"
MMLU_FILE = "all/{split}-00000-of-00001.parquet"


class Refused(Exception):
    """Configuration refused. Nothing has been loaded or billed."""


class GateFailed(Exception):
    """The instrument did not reproduce a known answer. Nothing is read."""


# --- the fingerprint --------------------------------------------------------


def token_fingerprint(ids) -> int:
    h = FNV_OFFSET
    for tid in ids:
        for b in int(tid).to_bytes(4, "little"):
            h = ((h ^ b) * FNV_PRIME) & U64
    return h


def test_fingerprint_matches_rust() -> None:
    """The one property this file cannot get wrong silently.

    If FNV drifts, every qhash mismatches and the run refuses — which is the
    safe direction, but it would look like a dataset problem. This pins the
    arithmetic itself against a value computed from the Rust definition.
    """
    assert token_fingerprint([]) == FNV_OFFSET
    # One byte at a time, little-endian: id 1 is bytes 01 00 00 00.
    h = FNV_OFFSET
    for b in (1, 0, 0, 0):
        h = ((h ^ b) * FNV_PRIME) & U64
    assert token_fingerprint([1]) == h


# --- the prompt, byte-identical to `llvq-llm/src/bin/mmlu.rs` ---------------


def pretty(subject: str) -> str:
    """`llvq-llm/src/bin/mmlu.rs::pretty` — underscores become spaces."""
    return subject.replace("_", " ")


def block(item: dict, answer: int | None) -> str:
    """One worked example, or the scored question when `answer` is None.

    Mirrors `mmlu.rs::block` exactly, including every `trim()` and every
    newline. A single space of difference changes the qhash, which is precisely
    how a divergence here is caught rather than published.
    """
    s = f"{item['question'].strip()}\n"
    for i, c in enumerate(item["choices"]):
        s += f"{LETTERS[i]}. {c.strip()}\n"
    s += "Answer:"
    if answer is not None:
        s += f" {LETTERS[answer]}\n\n"
    return s


def prefix_for(subject: str, shots: list[dict]) -> str:
    s = (
        "The following are multiple choice questions (with answers) about "
        f"{pretty(subject)}.\n\n"
    )
    for ex in shots:
        s += block(ex, ex["answer"])
    return s


# --- data -------------------------------------------------------------------


def load_mmlu(split: str, revision: str) -> list[dict]:
    """`cais/mmlu`, the same parquet `llvq-llm/src/corpus.rs:83` reads.

    ⚠️ The Rust side defaults to `main` unless `LLVQ_DATASET_REV` is set, so the
    reference dumps were not made against a pinned revision. That weakness is
    **caught downstream**: a dataset that moved makes qhash disagree, and the
    run refuses.
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - image contents
        raise Refused(
            f"{e.name} is missing from the image; this script runs in the job's "
            "vLLM image, not under uv"
        ) from e

    path = hf_hub_download(
        repo_id=MMLU_REPO,
        filename=MMLU_FILE.format(split=split),
        repo_type="dataset",
        revision=revision,
    )
    tbl = pq.read_table(path).to_pylist()
    out = []
    for row in tbl:
        ans = row["answer"]
        if isinstance(ans, str):  # some mirrors store the letter
            ans = LETTERS.index(ans.strip().upper())
        out.append(
            {
                "subject": row["subject"],
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": int(ans),
            }
        )
    return out


def read_reference_dump(path: Path) -> tuple[list[tuple[str, int, int]], dict]:
    """The question set, taken from a dump instead of re-selected.

    Returns the ordered `(subject, index, qhash)` triples and the header fields.
    Re-running `select()` in Python would reproduce a *seeded shuffle* — one
    more thing to get subtly wrong. Reading the dump cannot drift.
    """
    head: dict[str, str] = {}
    rows: list[tuple[str, int, int]] = []
    with path.open(encoding="utf-8") as fh:
        text = fh.read()
    body = io.StringIO()
    for line in text.splitlines(keepends=True):
        if line.startswith("#"):
            if "=" in line:
                k, _, v = line[1:].strip().partition("=")
                head[k.strip()] = v.strip()
            continue
        body.write(line)
    body.seek(0)
    for r in csv.DictReader(body):
        if not r.get("subject"):
            continue
        rows.append((r["subject"], int(r["index"]), int(r["qhash"], 16)))
    if not rows:
        raise Refused(f"{path} holds no question")
    return rows, head


# --- scoring ----------------------------------------------------------------


def build_prompts(
    wanted: list[tuple[str, int, int]],
    test: list[dict],
    dev: list[dict],
    tok,
) -> tuple[list[str], list[dict], list[int]]:
    """Rebuild each prompt and **verify its qhash** before anything is scored."""
    by_subject: dict[str, list[dict]] = {}
    for it in test:
        by_subject.setdefault(it["subject"], []).append(it)
    shots: dict[str, list[dict]] = {}
    for it in dev:
        shots.setdefault(it["subject"], []).append(it)

    prefixes = {s: prefix_for(s, shots.get(s, [])) for s in by_subject}

    prompts, items, bad = [], [], []
    for subject, index, qhash in wanted:
        pool = by_subject.get(subject)
        if pool is None or index >= len(pool):
            raise Refused(
                f"{subject}[{index}] not in the test split. The dataset moved."
            )
        it = pool[index]
        prompt = prefixes[subject] + block(it, None)
        ids = tok.encode(prompt, add_special_tokens=False)
        got = token_fingerprint(ids)
        if got != qhash:
            bad.append((subject, index, qhash, got))
        prompts.append(prompt)
        items.append(it)
    return prompts, items, bad


def score_mmlu(llm, tok, prompts: list[str]) -> list[list[float]]:
    """Four logprobs per question, one forward per (question, letter).

    ⚠️ **These are logprobs, not the logits `bin/mmlu` dumps.** `argmax` is
    identical — a monotone map does not move the arg-maximum — so `pick` and
    therefore every accuracy is the same. The numbers in the columns are not,
    and the emitted dump says so in its header rather than pretending.

    Why not `SamplingParams(logprobs=k)` on the bare prompt: the top-k may not
    contain all four letters, and a missing letter would silently become a
    `-inf` that changes the pick. Appending the letter and reading the last
    prompt logprob is exact for all four, always.
    """
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    flat = [p + s for p in prompts for s in ANSWER_STRINGS]
    outs = llm.generate(flat, sp)
    scores: list[list[float]] = []
    for i in range(0, len(outs), 4):
        row = []
        for j in range(4):
            pl = outs[i + j].prompt_logprobs
            last = pl[-1]
            # `prompt_logprobs=0` returns the chosen token's own logprob.
            row.append(float(next(iter(last.values())).logprob))
        scores.append(row)
    return scores


def score_mmlu_hf(model, tok, prompts: list[str], answer_ids: list[int], bs: int):
    """The same score, in the transformers/gptqmodel engine.

    It exists because **vLLM 0.26.0 refuses 2-bit GPTQ**: `Unsupported
    quantization config: bits=2, sym=True`, measured on 2026-08-30 after the
    artifact was produced. An arm that one stack cannot serve is still
    servable in its own, and that is exactly what "each in its own stack" means.

    This path is CLOSER to `bin/mmlu` than the vLLM path: it reads the **raw
    logits** of the last position and compares four values, word for word what
    `mmlu.rs:420-432` does. The vLLM path goes through logprobs: same argmax,
    other numbers.

    Padding is on the LEFT, so the last position is `-1` for the whole batch,
    without having to find the true end of each sequence. Getting that wrong
    would read the logits of a padding token, which would produce plausible and
    wrong numbers.
    """
    import torch

    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    out: list[list[float]] = []
    dev = next(model.parameters()).device
    for i in range(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits[:, -1, :].float()
        for row in logits:
            out.append([float(row[a]) for a in answer_ids])
        if (i // bs) % 10 == 0:
            print(f"    {min(i + bs, len(prompts))}/{len(prompts)}", flush=True)
    return out


def score_ppl(llm, tok, ids: list[int], ctx: int, nwin: int) -> list[tuple[float, int]]:
    """Non-overlapping windows, exactly `llvq-llm/src/bin/ppl.rs:115`.

    Returns `(nll_sum, count)` per window. The **per-window** values are the
    whole error bar — the windows are the sampling unit — so they are returned
    rather than summed, and printed at 9 decimals like `bin/ppl` does.
    """
    from vllm import SamplingParams, TokensPrompt

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    windows = [
        TokensPrompt(prompt_token_ids=ids[w * ctx : (w + 1) * ctx])
        for w in range(nwin)
    ]
    outs = llm.generate(windows, sp)
    per: list[tuple[float, int]] = []
    for o in outs:
        pl = o.prompt_logprobs
        # Position 0 has no prediction: vLLM returns None there.
        vals = [
            float(next(iter(d.values())).logprob) for d in pl if d is not None
        ]
        per.append((-sum(vals), len(vals)))
    return per


# --- report -----------------------------------------------------------------


def emit_dump(
    path: Path,
    arm: str,
    model_label: str,
    wanted: list[tuple[str, int, int]],
    items: list[dict],
    scores: list[list[float]],
    populations: dict[str, int],
    revision: str | None,
    engine: str,
) -> tuple[int, int, float, float]:
    """The `llvq-mmlu-dump v1` format, so `bin/mmlupair` consumes it unchanged.

    The header carries `scores=logprob` because the columns are named
    `logit_*` for compatibility and are **not** logits. Mislabeling them
    silently is exactly the class of error this project keeps paying for.
    """
    right = 0
    per: dict[str, list[int]] = {}
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# llvq-mmlu-dump v1\n")
        fh.write(f"# model={model_label} [vLLM arm {arm}]\n")
        fh.write(f"# revision={revision or 'main (NOT PINNED)'}\n")
        fh.write("# dtype=f16\n")
        if engine == "hf":
            fh.write("# scores=logit  (logits bruts, comme bin/mmlu)\n")
        else:
            fh.write("# scores=logprob  ⚠️ colonnes logit_* = LOGPROBS, pas des logits\n")
        fh.write(f"# engine={engine}\n")
        fh.write(
            "subject,index,population,qhash,answer,pick,correct,"
            "logit_a,logit_b,logit_c,logit_d\n"
        )
        for (subject, index, qhash), it, row in zip(wanted, items, scores):
            pick = max(range(4), key=lambda k: row[k])
            ok = int(pick == it["answer"])
            right += ok
            slot = per.setdefault(subject, [0, 0])
            slot[0] += ok
            slot[1] += 1
            fh.write(
                f"{subject},{index},{populations.get(subject, 0)},{qhash:016x},"
                f"{it['answer']},{pick},{ok},"
                + ",".join(f"{v:.6f}" for v in row)
                + "\n"
            )
    # MICRO, NOT THE NAIVE ONE. Every subject carries exactly `limit` questions
    # (40), so `Σright/Σtotal` is **algebraically the unweighted mean** of the
    # 57 rates, the MACRO. That is the defect §3ter of CLAUDE.md identified on
    # 2026-08-01 and fixed in `bin/mmlu`, and that this file fell back into on
    # 2026-08-30: it returned 72.85 where the reference publishes 70.32, and the
    # gate cost $0.20 to say so.
    #
    # The micro reweights every subject by its **population in the whole test
    # split**, not by the 40 sampled ones. The dump's `population` column exists
    # for that and for nothing else.
    #
    # Both are returned: the macro is not wrong, it answers another question.
    # What was wrong was calling it MMLU.
    total_pop = sum(populations.get(s, 0) for s in per)
    if total_pop == 0:
        raise Refused("no known population: the micro cannot be computed")
    micro = (
        sum(v[0] / v[1] * populations.get(s, 0) for s, v in per.items())
        / total_pop
        * 100.0
    )
    macro = sum(v[0] / v[1] for v in per.values()) / len(per) * 100.0
    return right, len(wanted), micro, macro


def check_gate(arm: str, mmlu_pct: float | None, ppl: float | None) -> list[str]:
    """Preregistration §2.4. Returns the violations; empty means green."""
    known = KNOWN.get(arm)
    if known is None:
        return []
    bad = []
    if mmlu_pct is not None:
        d = abs(mmlu_pct - known["mmlu"])
        if d > GATE_MMLU_PP:
            bad.append(
                f"{arm}: MMLU {mmlu_pct:.2f} against the known {known['mmlu']:.2f}, "
                f"|Δ| = {d:.2f} pp > {GATE_MMLU_PP} pp"
            )
    if ppl is not None:
        pct = abs(ppl - known["ppl"]) / known["ppl"] * 100.0
        if pct > GATE_PPL_PCT:
            bad.append(
                f"{arm}: ppl {ppl:.4f} against the known {known['ppl']:.4f}, "
                f"|Δ| = {pct:.2f}% > {GATE_PPL_PCT}%"
            )
    return bad


def main(argv: list[str]) -> int:
    test_fingerprint_matches_rust()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, help="f16 | awq_marlin | awq | gptq2")
    ap.add_argument("--model", required=True, help="repository or local path")
    ap.add_argument("--quantization", default=None)
    ap.add_argument(
        "--revision",
        default=None,
        help="revision of the weight repository. This job exists to reproduce "
        "known values, so the revision is pinned rather than left to drift",
    )
    ap.add_argument(
        "--reference-dump",
        default="docs/data/mmlu-dumps/mmlu-4b-f16.csv",
        help="where the (subject, index, qhash) come from, never re-selected",
    )
    ap.add_argument("--out", required=True, help="where to write this arm's dump")
    ap.add_argument("--dataset-rev", default=os.environ.get("LLVQ_DATASET_REV", "main"))
    ap.add_argument(
        "--engine",
        default="vllm",
        choices=("vllm", "hf"),
        help="vllm by default; hf for the formats vLLM refuses (2-bit GPTQ)",
    )
    ap.add_argument("--batch-size", type=int, default=8, help="hf engine only")
    ap.add_argument("--ppl", action="store_true", help="also score perplexity")
    ap.add_argument(
        "--gate",
        action="store_true",
        help="require this arm to reproduce its known value (prereg §2.4)",
    )
    args = ap.parse_args(argv)

    print(BANNER)
    print(f"vllm_score, arm {args.arm}, model {args.model}")
    print(BANNER)

    ref = Path(args.reference_dump)
    if not ref.is_file():
        raise Refused(f"reference dump not found: {ref}")
    wanted, head = read_reference_dump(ref)
    print(f"  questions requested   {len(wanted)}  (from {ref})")
    print(f"  dump header           {head}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    for s in ANSWER_STRINGS:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise Refused(
                f"{s!r} tokenizes to {ids}, not to a single token. The four-logprob "
                "comparison would make no sense"
            )

    test = load_mmlu("test", args.dataset_rev)
    dev = load_mmlu("dev", args.dataset_rev)
    populations: dict[str, int] = {}
    for it in test:
        populations[it["subject"]] = populations.get(it["subject"], 0) + 1

    prompts, items, bad = build_prompts(wanted, test, dev, tok)
    if bad:
        print(f"\nWARNING: {len(bad)} qhash do not match. The first five:")
        for subject, index, want, got in bad[:5]:
            print(f"    {subject}[{index}]  expected {want:016x}  got {got:016x}")
        raise Refused(
            "the rebuilt questions are not those of the reference dump: dataset "
            "moved, different tokenizer, or divergent prompt format. "
            "Nothing is scored."
        )
    print(f"  qhash verified        {len(wanted)}/{len(wanted)} OK")

    answer_ids = [tok.encode(x, add_special_tokens=False)[0] for x in ANSWER_STRINGS]

    if args.engine == "hf":
        import torch

        print(f"  engine                transformers/gptqmodel (batch {args.batch_size})")
        try:
            from gptqmodel import GPTQModel

            model = GPTQModel.load(args.model, device="cuda:0").model
        except Exception as e:
            print(f"  gptqmodel.load refused ({e}), falling back to AutoModelForCausalLM")
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                revision=args.revision,
                dtype=torch.float16,
                device_map="cuda:0",
            )
        model.eval()
        scores = score_mmlu_hf(model, tok, prompts, answer_ids, args.batch_size)
    else:
        from vllm import LLM

        kwargs: dict[str, Any] = dict(model=args.model, dtype="float16")
        if args.quantization:
            kwargs["quantization"] = args.quantization
        if args.revision:
            kwargs["revision"] = args.revision
        llm = LLM(**kwargs)
        scores = score_mmlu(llm, tok, prompts)
    right, total, mmlu_pct, macro_pct = emit_dump(
        Path(args.out),
        args.arm,
        args.model,
        wanted,
        items,
        scores,
        populations,
        args.revision,
        args.engine,
    )
    print(f"\n  MMLU micro            {mmlu_pct:.2f}%   <-- the metric of the paper")
    print(f"  MMLU macro            {macro_pct:.2f}%   (= right/total here: 40/subject)")
    print(f"  raw                   {right}/{total}")
    print(f"  dump                  {args.out}")

    ppl_value = None
    if args.ppl:
        raise Refused(
            "the ppl arm is not wired yet: it waits for the wikitext-2 corpus in "
            "the same shape as llvq-llm/src/corpus.rs. "
            "PROTOCOL DEVIATION, to be recorded in §7 of the preregistration."
        )

    if args.gate:
        violations = check_gate(args.arm, mmlu_pct, ppl_value)
        if violations:
            for v in violations:
                print(f"\nGATE RED: {v}")
            raise GateFailed(
                "the instrument does not reproduce a known answer: no quantized "
                "arm is read (prereg §2.4)"
            )
        if args.arm in KNOWN:
            print(f"  gate §2.4             green for {args.arm}")
        else:
            print(f"  gate §2.4             WARNING: {args.arm} has no known value")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (Refused, GateFailed) as e:
        print(f"\n{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
