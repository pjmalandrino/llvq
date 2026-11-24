"""Write llvq-files.json, the index `llvq pull` reads for Inkling 2 Mini, over a directory laid out as the Hub repo.

The library entry `inkling2:mini` (llvq-runtime/library.json) pins this index by size and sha256; the index pins
every file. Standard library only. Publishing the directory is the operator's decision; this script uploads nothing.

The directory, as the engine reads it once pulled (llvq-runtime/src/inkling.rs, LocalModel):

    config.json                      the text model's config (textonly/config.json)
    model.safetensors                the text-only checkpoint (textonly/model.safetensors)
    tokenizer.json                   the checkpoint's tokenizer
    chat_template.jinja              optional; checked against the template inkchat renders
    tables/embed.q4g64.llvqraw       the sealed file's two int4 tables
    tables/unembed.q4g64.llvqraw
    layers/L00.tar ... layers/L23.tar  one plain tar a layer: L00/VERIFIED and L00/codes/e000..e127.llvqrec

A layer archive holds the codes only, never the w13_hat/w2_hat reconstructions beside them:

    tar --format ustar -cf <repo>/layers/L06.tar -C <data>/layers L06/VERIFIED L06/codes

One archive a layer keeps the pull at about 30 requests: 3,100 files one by one would pass the Hub's 3,000
resolve requests per 5 minutes for an anonymous client.

    python3 ops/inkling/files_index.py <repo dir> [--context 32768]
"""

import argparse
import hashlib
import json
import os
import re
import sys

LAYER = re.compile(r"^layers/L\d\d\.tar$")
PLAIN = {"config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja",
         "tables/embed.q4g64.llvqraw", "tables/unembed.q4g64.llvqraw"}
REQUIRED = PLAIN - {"chat_template.jinja"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--context", type=int, default=None)
    a = ap.parse_args()
    files, seen = [], set()
    for root, _, names in os.walk(a.repo):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), a.repo).replace(os.sep, "/")
            if rel == "llvq-files.json" or rel.startswith(".") or "/." in rel:
                continue
            if rel not in PLAIN and not LAYER.match(rel):
                sys.exit(f"refused: {rel} is neither a part the engine reads nor a layer archive")
            p = os.path.join(a.repo, rel)
            e = {"path": rel, "size": os.path.getsize(p), "sha256": sha256(p)}
            if LAYER.match(rel):
                e["unpack"] = "layers"
            files.append(e)
            seen.add(rel)
    missing = sorted(REQUIRED - seen)
    if missing:
        sys.exit(f"refused: missing {', '.join(missing)}")
    if not any(LAYER.match(r) for r in seen):
        sys.exit("refused: no layer archive")
    files.sort(key=lambda e: e["path"])
    local = {"config": "config.json", "checkpoint": "model.safetensors", "layers": "layers",
             "tokenizer": "tokenizer.json", "tables": "tables"}
    if a.context:
        local["context"] = a.context
    index = {"schema": 1, "family": "inkling2", "files": files, "local": local}
    out = os.path.join(a.repo, "llvq-files.json")
    text = json.dumps(index, indent=2) + "\n"
    with open(out, "w") as f:
        f.write(text)
    total = sum(e["size"] for e in files)
    print(f"{out}: {len(files)} files, {total} bytes to download")
    print(f"library pins: \"size\": {len(text.encode())}, \"sha256\": \"{hashlib.sha256(text.encode()).hexdigest()}\", "
          f"\"download_size\": {total}")


if __name__ == "__main__":
    main()
