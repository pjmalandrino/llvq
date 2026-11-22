"""Corpora of the gate, read as the repository's Rust code reads them (llvq-llm/src/corpus.rs).

Calibration, dclm-edu (`corpus::dclm_edu_calibration` + `calib_chars` + `window_starts(seed=None)` +
`bin/hcapture.rs`): shard `data/000_00000.parquet` of `HuggingFaceTB/dclm-edu`, column `text`, rows in file order,
each row followed by "\\n\\n", read until the UTF-8 byte count reaches `n_calib * calib_len * 6` (the last row kept
whole); the whole text is tokenized once with `add_special_tokens=False` (no BOS, as `encode(text, false)`), and
window w is tokens [w*len, (w+1)*len): a contiguous prefix from token 0.

Evaluation, WikiText-2 raw test (`corpus::wikitext2_test` + `bin/ppl.rs`): rows of
`wikitext-2-raw-v1/test-00000-of-00001.parquet` (Salesforce/wikitext) joined by "\\n\\n", tokenized with
`add_special_tokens=False`, non-overlapping windows of ctx tokens from token 0; a window's NLL is over its ctx-1
next-token predictions.

BOS: none, in both, as the repository does for every model; Inkling's `<|begin_of_text|>` is never prepended.
"""

import glob
import os

HUB = os.path.expanduser("~/.cache/huggingface/hub")


def _one(pattern):
    hits = sorted(glob.glob(os.path.join(HUB, pattern)))
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern}: {hits}")
    return hits[0]


def dclm_path():
    return _one("datasets--HuggingFaceTB--dclm-edu/snapshots/*/data/000_00000.parquet")


def wikitext_path():
    return _one("datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-00000-of-00001.parquet")


def dclm_text(max_chars):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(dclm_path())
    parts, n, rows = [], 0, 0
    for batch in pf.iter_batches(batch_size=1024, columns=["text"]):
        for s in batch.column(0).to_pylist():
            if s is None:
                continue
            parts.append(s)
            parts.append("\n\n")
            n += len(s.encode("utf-8")) + 2
            rows += 1
            if n >= max_chars:
                return "".join(parts), {"rows_read": rows, "bytes": n, "budget": max_chars, "file": dclm_path()}
    raise RuntimeError(f"dclm shard exhausted at {n} bytes < {max_chars}")


def calib_windows(tok, n_calib=64, calib_len=2048):
    text, stats = dclm_text(n_calib * calib_len * 6)
    ids = tok.encode(text, add_special_tokens=False).ids
    avail = len(ids) // calib_len
    if avail < n_calib:
        raise RuntimeError(f"only {avail} windows of {calib_len} in {len(ids)} tokens; {n_calib} asked")
    stats.update({"tokens": len(ids), "windows_available": avail})
    return [ids[w * calib_len : (w + 1) * calib_len] for w in range(n_calib)], stats


def wikitext_ids(tok):
    import pyarrow.parquet as pq

    rows = pq.read_table(wikitext_path(), columns=["text"]).column(0).to_pylist()
    return tok.encode("\n\n".join(rows), add_special_tokens=False).ids


def eval_windows(tok, n=8, ctx=2048):
    ids = wikitext_ids(tok)
    if len(ids) // ctx < n:
        raise RuntimeError("wikitext too short")
    return [ids[w * ctx : (w + 1) * ctx] for w in range(n)], {"tokens": len(ids), "file": wikitext_path()}
