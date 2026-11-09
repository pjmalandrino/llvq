# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers>=0.21", "pyarrow>=15", "huggingface_hub>=0.24"]
# ///
"""Inputs of the llama.cpp duel, built on the Mac and checked against our own records.

Every input the llama.cpp arm reads is made here, as token ids, and each one is
proved equal to what our side read, by fingerprint, before it is shipped:

* `ppl-tokens.u32`: the 12 windows of 4,096 of raw WikiText-2 test that
  `bin/ppl` scores. Fingerprint 3f1baca9033bf251, the one
  `docs/mesures/ppl-scelles-2026-10-04.txt` prints for all three sealed files.
* `calib-dclm.txt`: the calibration text of the sealed files, the contiguous
  prefix of 64 x 2,048 = 131,072 tokens of the dclm-edu shard, read the way
  `corpus::parquet_text_bounded` reads it. The bounded read must reproduce the
  line every sealed encode printed: 825,408 chars, 127 rows, revision dbad8ad7.
  llama-imatrix takes text, so the text is the decode of those tokens, and the
  script checks that it re-encodes to the same ids.
* `gsm8k-prompts.jsonl`: the 1,319 census prompts as ids, built by
  `ops/gsm8k_vllm.py`, checked problem by problem against the qhash of our 4B
  census dump and against its run fingerprint.
* `mmlu-prompts-ids.jsonl`: the 14,042 census prompts of
  `paper-iq2-census-2026-09-25`, each with its ids; every line's ids must hash to
  its qhash, and the concatenation to the census run fingerprint a74a6d6213602979
  our sealed dumps end on.

Usage:
    uv run ops/duel/inputs.py <out_dir> <mmlu-prompts-FULL.jsonl>
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import gsm8k_vllm as g  # noqa: E402  (ops/gsm8k_vllm.py: prompts, fingerprint)

TOK_REPO = "Qwen/Qwen3-4B"
TOK_REV = "1cfa9a7208912126459214e8b04321603b3df60c"
# The other two bases, whose tokenizer.json must be byte-identical to the 4B's.
OTHER = {
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen/Qwen3-14B": "40c069824f4251a91eefaf281ebe4c544efd3e18",
}

WIKI_REPO, WIKI_FILE = "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet"
PPL_CTX, PPL_WINDOWS, PPL_FP = 4096, 12, 0x3F1BACA9033BF251

DCLM_REPO, DCLM_FILE = "HuggingFaceTB/dclm-edu", "data/000_00000.parquet"
DCLM_REV = "dbad8ad71224482740cd9c9d353591adbf62fe04"
N_CALIB, CALIB_LEN = 64, 2048
CALIB_CHARS = N_CALIB * CALIB_LEN * 6  # corpus::calib_chars
CALIB_WANT_CHARS, CALIB_WANT_ROWS = 825_408, 127  # encode-14b smoke.txt line 21

GSM8K_REV = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_REF = Path("docs/mesures/gsm8k-wave1-2026-09-26-brut/run1-served/gsm8k-4b-served.jsonl")
GSM8K_FP = "bfa9135ce4b55c75"

MMLU_N, MMLU_FP = 14_042, 0xA74A6D6213602979
MMLU_PSHA = "9bd7a441888424f7a8b207366e45ede34d4b150e11a2e28a4bc1be0043cf5556"


def fail(msg: str) -> None:
    print(f"REFUSED: {msg}", file=sys.stderr)
    sys.exit(2)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fnv_extend(h: int, ids) -> int:
    """Continue `token_fingerprint` over more ids (FNV-1a is a stream)."""
    for tid in ids:
        for b in int(tid).to_bytes(4, "little"):
            h = ((h ^ b) * g.FNV_PRIME) & g.U64
    return h


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 1
    out, mmlu_src = Path(argv[0]), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    import pyarrow.parquet as pq

    tj = Path(hf_hub_download(TOK_REPO, "tokenizer.json", revision=TOK_REV))
    for repo, rev in OTHER.items():
        o = Path(hf_hub_download(repo, "tokenizer.json", revision=rev))
        if sha256(o) != sha256(tj):
            fail(f"{repo}@{rev} tokenizer.json differs from the 4B's")
    tok = Tokenizer.from_file(str(tj))
    print(f"tokenizer   {TOK_REPO}@{TOK_REV[:8]}, sha256 {sha256(tj)[:16]}, same file at 8B and 14B")

    # --- perplexity windows ---------------------------------------------------
    wp = hf_hub_download(WIKI_REPO, WIKI_FILE, repo_type="dataset")
    rows = pq.read_table(wp, columns=["text"]).column("text").to_pylist()
    ids = tok.encode("\n\n".join(rows), add_special_tokens=False).ids
    n = PPL_CTX * PPL_WINDOWS
    fp = g.token_fingerprint(ids[:n])
    if fp != PPL_FP:
        fail(f"wikitext fingerprint {fp:016x}, ppl-scelles printed {PPL_FP:016x}")
    pp = out / "ppl-tokens.u32"
    pp.write_bytes(struct.pack(f"<{n}I", *ids[:n]))
    print(f"ppl         {len(ids)} tokens, first {PPL_WINDOWS} x {PPL_CTX}, fingerprint {fp:016x} (= ppl-scelles)")

    # --- calibration text -----------------------------------------------------
    dp = hf_hub_download(DCLM_REPO, DCLM_FILE, repo_type="dataset", revision=DCLM_REV)
    pf = pq.ParquetFile(dp)
    text, nrows = [], 0
    nbytes = 0
    for gi in range(pf.num_row_groups):
        if nbytes >= CALIB_CHARS:
            break
        for s in pf.read_row_group(gi, columns=["text"]).column("text").to_pylist():
            nrows += 1
            piece = s + "\n\n"
            text.append(piece)
            nbytes += len(piece.encode("utf-8"))
            if nbytes >= CALIB_CHARS:
                break
    if (nbytes, nrows) != (CALIB_WANT_CHARS, CALIB_WANT_ROWS):
        fail(f"bounded read {nbytes} chars / {nrows} rows, the sealed encodes read "
             f"{CALIB_WANT_CHARS} / {CALIB_WANT_ROWS}")
    cids = tok.encode("".join(text), add_special_tokens=False).ids
    want = N_CALIB * CALIB_LEN
    if len(cids) < want:
        fail(f"calibration text holds {len(cids)} tokens, {want} needed")
    cids = cids[:want]
    ctext = tok.decode(cids, skip_special_tokens=False)
    back = tok.encode(ctext, add_special_tokens=False).ids
    same = sum(a == b for a, b in zip(back, cids))
    cp = out / "calib-dclm.txt"
    cp.write_text(ctext, encoding="utf-8")
    print(f"calibration bounded read {nbytes} chars, {nrows} rows, revision {DCLM_REV[:8]} (= sealed encodes); "
          f"{want} tokens, fingerprint {g.token_fingerprint(cids):016x}; "
          f"re-encoded {len(back)} tokens, first {same} equal")
    if len(back) < want or back[: want - 8] != cids[: want - 8]:
        fail("the decoded calibration text does not re-encode to its ids (beyond the last 8)")

    # --- GSM8K ----------------------------------------------------------------
    items, commit = g.load_gsm8k(GSM8K_REV)
    if commit != GSM8K_REV:
        fail(f"gsm8k at {commit}, expected {GSM8K_REV}")
    picked = g.select(len(items), 10**9)
    marks = g.Marks(tok)
    prompts = [g.prompt_ids(tok, marks, items[i]["question"], False) for i in picked]
    gfp = f"{g.token_fingerprint([t for p in prompts for t in p]):016x}"
    _, ref = g.read_dump(GSM8K_REF)
    want_q = {r["index"]: r["qhash"] for r in ref}
    bad = [i for i, p in zip(picked, prompts) if f"{g.token_fingerprint(p):016x}" != want_q.get(i)]
    if bad or gfp != GSM8K_FP or len(ref) != len(picked):
        fail(f"gsm8k: {len(bad)} qhash differ, run fingerprint {gfp} against {GSM8K_FP}")
    gp = out / "gsm8k-prompts.jsonl"
    with gp.open("w") as w:
        w.write(json.dumps({"dataset": g.GSM8K_REPO, "revision": GSM8K_REV, "fingerprint": gfp,
                            "instruction": g.INSTRUCTION, "think": False, "questions": len(picked),
                            "stops": [marks.im_end, marks.eot]}) + "\n")
        for i, p in zip(picked, prompts):
            w.write(json.dumps({"index": i, "qhash": f"{g.token_fingerprint(p):016x}", "ids": p,
                                "gold": g.gold_of(items[i]["answer"])}) + "\n")
    print(f"gsm8k       {len(picked)} prompts, qhash {len(picked)}/{len(picked)} and run fingerprint {gfp} "
          f"= our 4B census dump")

    # --- MMLU -----------------------------------------------------------------
    if sha256(mmlu_src) != MMLU_PSHA:
        fail(f"{mmlu_src} is not the census prompts file (sha256 {MMLU_PSHA[:8]})")
    mrows = [json.loads(l) for l in mmlu_src.read_text(encoding="utf-8").split("\n") if l.strip()]
    h, bad = g.FNV_OFFSET, 0
    mp = out / "mmlu-prompts-ids.jsonl"
    with mp.open("w", encoding="utf-8") as w:
        for r in mrows:
            mi = tok.encode(r["prompt"], add_special_tokens=False).ids
            if f"{g.token_fingerprint(mi):016x}" != r["qhash"]:
                bad += 1
            h = fnv_extend(h, mi)
            r["ids"] = mi
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    if bad or len(mrows) != MMLU_N or h != MMLU_FP:
        fail(f"mmlu: {bad} qhash differ, {len(mrows)} rows, run fingerprint {h:016x} against {MMLU_FP:016x}")
    print(f"mmlu        {len(mrows)} prompts, qhash {len(mrows)}/{len(mrows)}, run fingerprint {h:016x} "
          f"= our census dumps")

    with (out / "inputs.sha256").open("w") as w:
        for p in (pp, cp, gp, mp):
            w.write(f"{sha256(p)}  {p.name}\n")
    print((out / "inputs.sha256").read_text(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
