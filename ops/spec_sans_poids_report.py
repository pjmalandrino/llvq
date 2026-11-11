"""Tables for ops/spec_sans_poids.py (stdlib only).

    python3 ops/spec_sans_poids_report.py <rows.json> [<meta.json>]

Accepted a pass = tokens over passes minus one (sum over prompts), per kind; per-prompt median [range].
Projection (*estimated*): tok/s = (accepted + 1) / (verify + drafter), verify 8.53 ms GSM8K, 8.58 ms house set,
drafter 0.75 ms (top-k over 16 rows). Whole-model b/param: sealed 2.7320 + table bits over 4,022,468,096 parameters.
"""
import json
import statistics
import sys

PARAMS = 4_022_468_096
SEALED = 2.7320
VERIFY = {"gsm8k": 8.53}
DRAFT_MS = 0.75


def main():
    rows = json.load(open(sys.argv[1]))
    meta = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    arms = [k for k, v in rows[0].items() if isinstance(v, dict) and "hist" in v]
    kinds = []
    for r in rows:
        kd = "gsm8k" if r["set"] == "gsm" else r["kind"]
        r["_k"] = kd
        if kd not in kinds:
            kinds.append(kd)
    tb = (meta or {}).get("table_bytes", {})
    print(f"prompts {len(rows)}; arms {arms}")
    if meta:
        print(f"gate {json.dumps(meta['gate'])}; stats {json.dumps(meta['stats'])}")
    for kd in kinds:
        sub = [r for r in rows if r["_k"] == kd]
        v = VERIFY.get(kd, 8.58)
        print(f"\n{kd}  n {len(sub)}  verify {v} ms")
        print(f"  {'arm':<20} {'acc/pass':>8} {'median [range] per prompt':>28} {'tok/s est':>9} {'table B':>10} {'b/param':>8}")
        for a in arms:
            acc = sum(r[a]["accepted"] for r in sub) / sum(r[a]["passes"] for r in sub)
            per = [r[a]["accepted"] / r[a]["passes"] for r in sub]
            b = tb.get(a, 0)
            print(f"  {a:<20} {acc:8.3f} {statistics.median(per):8.3f} [{min(per):.3f}-{max(per):.3f}]{'':>6}"
                  f" {(acc + 1) / (v + DRAFT_MS) * 1000:9.1f} {b:10d} {SEALED + b * 8 / PARAMS:8.4f}")
    if len([k for k in kinds if k != "gsm8k"]) > 1:
        sub = [r for r in rows if r["set"] == "house"]
        print(f"\nhouse set pooled  n {len(sub)}")
        for a in arms:
            acc = sum(r[a]["accepted"] for r in sub) / sum(r[a]["passes"] for r in sub)
            print(f"  {a:<20} {acc:8.3f} {(acc + 1) / (8.58 + DRAFT_MS) * 1000:9.1f}")
    for kd in kinds:
        sub = [r for r in rows if r["_k"] == kd]
        print(f"\nhistogram of accepted a pass 0..15, {kd}")
        for a in arms:
            h = [sum(r[a]["hist"][i] for r in sub) for i in range(16)]
            print(f"  {a:<20} {h}")


if __name__ == "__main__":
    main()
