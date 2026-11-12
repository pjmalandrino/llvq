"""Tables for spec-arbre32-mac-2026-11-11 from rows.json of ops/dflash_arbre32.py (stdlib only).

    python3 ops/dflash_arbre32_report.py <out_dir>/rows.json

Accepted a pass by kind (sum of accepted over sum of passes, the card's aggregate), the median and range of the
per-prompt value, the histogram 0..15, the paired difference against the card's tree of 15 on the same prompts, and
the projected GSM8K tok/s: tokens a pass over (verify + drafter), verify 8.53 ms + 0.087 ms a row past 16, drafter
4.15 ms (SP8's measured loop costs; the per-row slope estimated from c_16 - c_8 on the card).
"""
import json
import statistics as st
import sys

ARMS = ["tree15", "tree23", "tree31", "tree16", "chain15"]
ROWS = {"tree15": 16, "tree16": 17, "tree23": 24, "tree31": 32, "chain15": 16}


def agg(rows, arm):
    a = sum(r[arm]["accepted"] for r in rows)
    p = sum(r[arm]["passes"] for r in rows)
    per = [r[arm]["accepted"] / r[arm]["passes"] for r in rows if r[arm]["passes"]]
    h = [sum(r[arm]["hist"][i] for r in rows) for i in range(16)]
    return a / p, st.median(per), min(per), max(per), h, p


def main(path):
    rows = json.load(open(path))
    groups = {}
    for r in rows:
        key = r["kind"] if r["set"] == "house" else "gsm8k"
        groups.setdefault(key, []).append(r)
    order = [k for k in ("math", "code", "rag", "chat", "gsm8k") if k in groups]
    print("accepted drafts a pass (tokens a pass = this + 1); per-prompt median [min-max]; text source")
    for k in order:
        g = groups[k]
        src = sorted({r["text"] for r in g})
        print(f"\n{k}: {len(g)} prompts, {sum(r['tokens'] for r in g)} tokens after first, text {src}")
        card = [r for r in g if "card_tree" in r]
        if card:
            ca = sum(r["card_tree"]["tokens"] for r in card) / sum(r["card_tree"]["passes"] for r in card) - 1
            cc = [r for r in card if "card_chain" in r]
            cline = f"  card tree15 {ca:.2f}"
            if cc:
                ch = sum(r["card_chain"]["tokens"] for r in cc) / sum(r["card_chain"]["passes"] for r in cc) - 1
                cline += f", card chain15 {ch:.2f}"
            print(cline + " (card's own loop, recomputed from its per-prompt tokens and passes)")
        for arm in ARMS:
            m, med, lo, hi, h, p = agg(g, arm)
            print(f"  {arm:8s} rows {ROWS[arm]:2d}: {m:5.2f}  median {med:.2f} [{lo:.2f}-{hi:.2f}]  passes {p}  hist {h}")
        if card:
            d = [r["tree15"]["accepted"] / r["tree15"]["passes"] - r["card_tree"]["acc"] for r in card]
            print(f"  paired Mac tree15 - card tree15, per prompt: median {st.median(d):+.2f} [{min(d):+.2f}-{max(d):+.2f}],"
                  f" mean {st.mean(d):+.2f}, sd {st.pstdev(d):.2f}, n {len(d)}")
        tf = [r["tf_agree"] for r in g]
        print(f"  teacher-forced agreement of the MLX door with the text: mean {st.mean(tf):.4f}, min {min(tf):.4f}")
    print("\nprojection, GSM8K (estimated): tok/s = (accepted + 1) / (verify + drafter)")
    if "gsm8k" in groups:
        for arm in ("tree15", "tree23", "tree31"):
            m = agg(groups["gsm8k"], arm)[0]
            v = 8.53 + 0.087 * (ROWS[arm] - 16)
            print(f"  {arm}: {m + 1:.2f} tokens a pass, verify {v:.3f} ms + drafter 4.15 ms -> {1000 * (m + 1) / (v + 4.15):.1f} tok/s")


if __name__ == "__main__":
    main(sys.argv[1])
