"""Texts of prereg inkling-fenetre-ppl-2026-11-27, from the worktree root:
    HF_HUB_OFFLINE=1 python3 -I docs/mesures/inkling-mac-fenetre-ppl-2026-11-27-brut/windows.py
WikiText-2 test windows 0 to 5 of 2,048 tokens made as ops/inkling/data.py makes them (`eval_windows`, the corpus
and the tokenizer already on this Mac, read only). Windows 0 to 2 must equal the files this Mac holds
(lot2-d/wikitext_w0_2048.json, lot4-x/wikitext_w{1,2}_2048.json), or nothing is written. Writes w3, w4, w5 and the
two joined texts (w0+w1+w2, w3+w4+w5, 6,144 ids each) in this folder, in the files' format (key `ids`).
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "ops", "inkling"))
import data  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

D = os.path.expanduser("~/llvq-inkling-mini")
tok = Tokenizer.from_file(os.path.join(D, "native", "tokenizer.json"))
wins, info = data.eval_windows(tok, 6, 2048)
held = [f"{D}/lot2-d/wikitext_w0_2048.json", f"{D}/lot4-x/wikitext_w1_2048.json", f"{D}/lot4-x/wikitext_w2_2048.json"]
for w, p in enumerate(held):
    if json.load(open(p))["ids"] != wins[w]:
        sys.exit(f"window {w} made here differs from {p}: nothing written")
    print(f"window {w}: equal to {p}")
for w in (3, 4, 5):
    p = os.path.join(HERE, f"wikitext_w{w}_2048.json")
    json.dump({"ids": wins[w], "source": info, "window": w, "ctx": 2048, "ntok": 2048}, open(p, "w"))
    print(f"window {w}: {p}")
for name, ws in (("joined_w012_6144.json", (0, 1, 2)), ("joined_w345_6144.json", (3, 4, 5))):
    ids = [t for w in ws for t in wins[w]]
    json.dump({"ids": ids, "source": info, "windows": list(ws), "ctx": 2048, "ntok": len(ids)}, open(os.path.join(HERE, name), "w"))
    print(f"{name}: {len(ids)} ids")
