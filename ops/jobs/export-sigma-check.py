#!/usr/bin/env python3
"""Check a llvqtune row_scales export before `rowscale` folds it, and summarize it
the way docs/mesures/dclm-rowscales-2026-09-20.txt does (mean, sd, p1, p99,
min, max, share of rows beyond 5 %, sd per projection type).

  export-sigma-check.py <sigma.json> <expected matrices> <expected rows>
  8B: export-sigma-check.py ~/dclm-8b-sigma.json 216 1363968
  4B reference: export-sigma-check.py ~/dclm-sigma.json 216 1069056

  ones <sigma.json> <out.json>   writes the same keys with every value 1.0
                                 (the idempotence control of rowscale.rs:31-35)

stdlib only. Refuses what rowscale.rs refuses (kind, empty, non-positive,
non-finite) so a bad export fails here, before a 4.4 GB file is written.
"""
import json
import math
import sys
from collections import defaultdict


def payload(path):
    doc = json.load(open(path))
    p = doc.get("result", doc)
    assert p.get("kind") == "row_scales", f"kind {p.get('kind')!r}"
    return doc, p


def main():
    if sys.argv[1] == "ones":
        doc, p = payload(sys.argv[2])
        p["sigma"] = {k: [1.0] * len(v) for k, v in p["sigma"].items()}
        json.dump(doc, open(sys.argv[3], "w"))
        print(f"wrote {sys.argv[3]}: {len(p['sigma'])} matrices, all 1.0")
        return
    path, want_m, want_r = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    _, p = payload(path)
    sig = p["sigma"]
    assert sig, "the export scales no matrix"
    assert not any("v_proj" in k for k in sig), "v_proj is int4 and holds no row_scales"
    vals, by_type = [], defaultdict(list)
    for k, v in sig.items():
        assert v, f"{k}: empty"
        for x in v:
            assert isinstance(x, (int, float)) and math.isfinite(x) and x > 0, f"{k}: {x}"
        vals += v
        by_type[k.rsplit(".", 1)[-1]] += v
    m, r = len(sig), len(vals)
    print(f"{m} matrices (want {want_m}), {r} rows (want {want_r})")
    assert (m, r) == (want_m, want_r), "count mismatch: wrong model, or wrong export"
    s = sorted(vals)
    mean = sum(vals) / r
    sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / r)
    q = lambda f: s[min(r - 1, int(f * r))]
    beyond = sum(abs(x - 1) > 0.05 for x in vals) / r
    print(f"sigma mean {mean:.5f} sd {sd:.5f} p1 {q(0.01):.4f} p99 {q(0.99):.4f} "
          f"min {s[0]:.4f} max {s[-1]:.4f} {100 * beyond:.1f} % beyond 5 %")
    for t in sorted(by_type):
        v = by_type[t]
        mu = sum(v) / len(v)
        print(f"  {t:<10} sd {math.sqrt(sum((x - mu) ** 2 for x in v) / len(v)):.4f}  rows {len(v)}")


if __name__ == "__main__":
    main()
