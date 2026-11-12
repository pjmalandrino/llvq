"""Tables for spec-brouillon-int4-mac-2026-11-11 from rows.json of ops/dflash_brouillon_int4.py (stdlib only).

    python3 ops/dflash_brouillon_int4_report.py <out_dir>/rows.json [<arbre32 rows.json>]

Accepted drafts a pass of the tree of 15 (sum accepted over sum passes, the card's aggregate) per arm, by kind and per
set; the relative change against the unquantized drafter ("ref"); the per-prompt paired difference; the gate of M2
(within 3 % of ref on GSM8K-200 and on the house set, a gain passes); sizes and whole-model b/param (computed).
With the arbre32 rows, checks that ref reproduces that run's tree of 15 prompt by prompt.
"""
import json
import statistics as st
import sys

TARGET_PARAMS = 4_022_468_096  # spec-memoire-2026-11-11.txt
TARGET_BPP = 2.7320  # the sealed file, docs/ETAT.md:32
LINEAR = 5 * (4096 * 2560 + 2 * 1024 * 2560 + 2560 * 4096 + 3 * 9728 * 2560)  # 504,627,200
FC = 2560 * 12800  # 32,768,000
NORMS = 5 * (2 * 2560 + 2 * 128) + 2 * 2560  # 32,000: ln1, ln2, q_norm, k_norm a layer; hidden_norm, norm
DRAFTER = LINEAR + FC + NORMS  # 537,427,200
GATE = 0.03

# arm -> (bits a linear weight with its scale and bias, fc quantized)
RATE = {"ref": (16.0, False), "f16": (16.0, False), "int8g64": (8 + 32 / 64, False),
        "int4g128": (4 + 32 / 128, False), "int4g64": (4 + 32 / 64, False), "int4g128+fc": (4 + 32 / 128, True),
        "int4g64+fc": (4 + 32 / 64, True), "int3g64": (3 + 32 / 64, False)}


def agg(rows, arm):
    p = sum(r[arm]["passes"] for r in rows)
    return sum(r[arm]["accepted"] for r in rows) / p if p else float("nan")


def per(r, arm):
    return r[arm]["accepted"] / r[arm]["passes"]


def size(arm):
    r, fcq = RATE[arm]
    bits = LINEAR * r + FC * (r if fcq else 16.0) + NORMS * 16.0
    return bits


def main(path, a32=None):
    rows = json.load(open(path))
    assert DRAFTER == 537_427_200, DRAFTER
    arms = [a for a in RATE if a in rows[0]]
    groups = {}
    for r in rows:
        groups.setdefault(r["kind"] if r["set"] == "house" else "gsm8k", []).append(r)
    sets = {"gsm8k-200": [r for r in rows if r["set"] == "gsm"], "house": [r for r in rows if r["set"] == "house"]}
    print(f"prompts: gsm8k {len(sets['gsm8k-200'])}, house {len(sets['house'])} "
          f"({', '.join(f'{k} {len(v)}' for k, v in groups.items() if k != 'gsm8k')})")

    if a32:
        old = {(r["set"], r["id"]): r for r in json.load(open(a32))}
        same = sum(1 for r in rows if old[(r["set"], r["id"])]["tree15"]["accepted"] == r["ref"]["accepted"]
                   and old[(r["set"], r["id"])]["tree15"]["passes"] == r["ref"]["passes"])
        print(f"ref against the arbre32 run's tree15: {same} of {len(rows)} prompts identical (accepted and passes)")

    print("\nACCEPTED DRAFTS A PASS, TREE OF 15 (measured); relative change against ref (computed)")
    kinds = [k for k in ("gsm8k", "math", "code", "rag", "chat") if k in groups]
    head = f"  {'arm':12s}" + "".join(f"{k:>16s}" for k in ["gsm8k-200", "house"] + kinds[1:])
    print(head)
    for a in arms:
        cells = []
        for g in [sets["gsm8k-200"], sets["house"]] + [groups[k] for k in kinds[1:]]:
            v, v0 = agg(g, a), agg(g, "ref")
            cells.append(f"{v:6.3f} {100 * (v / v0 - 1):+6.2f}%")
        print(f"  {a:12s}" + "".join(f"{c:>16s}" for c in cells))

    print("\nPAIRED PER PROMPT, arm - ref, accepted a pass (computed): mean, sd, median [min, max], prompts changed")
    for name, g in sets.items():
        print(f"  {name}")
        for a in arms[1:]:
            d = [per(r, a) - per(r, "ref") for r in g]
            ch = sum(1 for x in d if x != 0)
            print(f"    {a:12s} {st.mean(d):+.3f}  sd {st.pstdev(d):.3f}  median {st.median(d):+.3f} "
                  f"[{min(d):+.2f}, {max(d):+.2f}]  changed {ch}/{len(d)}")

    print(f"\nGATE (within {100 * GATE:.0f} % of ref on both sets; a gain passes)")
    for a in arms[1:]:
        rel = {n: agg(g, a) / agg(g, "ref") - 1 for n, g in sets.items()}
        ok = all(v >= -GATE for v in rel.values())
        print(f"  {a:12s} gsm8k {100 * rel['gsm8k-200']:+.2f} %  house {100 * rel['house']:+.2f} %  "
              f"{'PASS' if ok else 'FAIL'}")

    print("\nSIZE (computed): drafter stored, linears at the arm's rate with f16 scale and bias, the rest f16;")
    print("whole model = sealed target 2.7320 b/param + drafter bits over the target's 4,022,468,096 parameters")
    print(f"  {'arm':12s} {'drafter GB':>10s} {'b/drafter param':>16s} {'whole b/param':>14s} {'vs f16 GB':>10s}")
    for a in arms:
        bits = size(a)
        print(f"  {a:12s} {bits / 8e9:10.4f} {bits / DRAFTER:16.3f} {TARGET_BPP + bits / TARGET_PARAMS:14.3f} "
              f"{(bits - size('f16')) / 8e9:+10.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
