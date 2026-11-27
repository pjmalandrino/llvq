"""Gate D-iii of prereg inkling-fenetre-2026-11-27: the timed pairs, from the raw files of this folder.

usage: python3 -I summary.py      (from this folder)
For each length, pair 0 is discarded and pairs 1 to 4 are timed. Prints each run's read, decode, GB and time -l
peak footprint and the digest of its logits, each pair's ratio arm / reference, and the medians with their ranges.
"""
import hashlib, json, re, statistics


def run(label):
    r = json.load(open(f"{label}.json"))
    log = open(f"{label}.log").read()
    peak = int(re.search(r"(\d+)\s+peak memory footprint", log).group(1)) / 1e9
    rss = int(re.search(r"(\d+)\s+maximum resident set size", log).group(1)) / 1e9
    m = r["memory"]
    after = (m.get("after_prefill") or {}).get("used", 0) / 1e9
    sha = hashlib.sha256(open(f"{label}.logits.f32", "rb").read()).hexdigest()[:8]
    machine = open(f"{label}.machine").read().split(" ", 2)[2].strip()
    return r["seconds_prefill"], 1 / r["step_seconds_median"], after, peak, rss, sha, machine


def med(xs):
    return f"{statistics.median(xs):.2f} [{min(xs):.2f}, {max(xs):.2f}]"


for n in (9000, 2000):
    p = "t9" if n == 9000 else "t2"
    print(f"== {n} tokens")
    reads = {"ref": [], "arm": []}
    tok = {"ref": [], "arm": []}
    peaks = {"ref": [], "arm": []}
    ratios = []
    for i in range(5):
        rows = {}
        for arm in ("ref", "arm"):
            s, t, after, peak, rss, sha, machine = run(f"{p}-{arm}{i}")
            rows[arm] = s
            tag = "discarded" if i == 0 else "timed"
            print(f"  {p}-{arm}{i} ({tag}): read {s:.2f} s, decode {t:.2f} tok/s, GB after prompt {after:.2f}, "
                  f"peak footprint {peak:.2f} GB, max RSS {rss:.2f} GB, logits {sha}; before: {machine}")
            if i > 0:
                reads[arm].append(s)
                tok[arm].append(t)
                peaks[arm].append(peak)
        if i > 0:
            ratios.append(rows["arm"] / rows["ref"])
            print(f"  pair {i}: arm / reference {ratios[-1]:.4f}")
    print(f"  read, reference: {med(reads['ref'])} s; arm: {med(reads['arm'])} s")
    print(f"  pair ratios: median {statistics.median(ratios):.4f} [{min(ratios):.4f}, {max(ratios):.4f}]")
    print(f"  decode after the prompt, reference: {med(tok['ref'])} tok/s; arm: {med(tok['arm'])} tok/s")
    print(f"  peak footprint, reference: {med(peaks['ref'])} GB; arm: {med(peaks['arm'])} GB")
