#!/usr/bin/env python3
"""Stage 0 bis of the Tetra gain-decision plan, on real compensated GPTQ residues.

Usage: uv run --offline python ops/gain_disagree_real.py PILOT_DIR [--csv OUT.csv]
       uv run --offline python ops/gain_disagree_real.py PILOT_DIR --mutate cost

`gaindisagree` measures the served gain rule against the Euclidean optimum on a
Gaussian block source. This reads the same two rules off blocks the quantizer
actually saw: the shadow comparisons of the Schur pilot, whose `working` array
is compensated by `commit` after every block, against a row scale frozen before
the loop and centroids fitted on the rotated uncompensated weights exactly as
`llvq-llm/src/calib.rs` fits them.

It runs no model, encodes nothing and writes nothing outside its own outputs.
Every value below is read from dumps produced by commit 09e0f65, verified
against the pilot's own `outputs.sha256` before a single number is computed.
A file absent from that manifest, or whose digest moved, stops the run.
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys

DIM = 24
RATE_BITS_PER_DIM = 2.0
QS = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(root):
    """`outputs.sha256` keyed by path relative to the pilot directory."""
    path = root / "outputs.sha256"
    if not path.exists():
        sys.exit(f"{path} is missing: an unverified dump is not measured")
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        want, name = line.split(maxsplit=1)
        name = pathlib.Path(name.strip())
        try:
            out[str(name.relative_to(root))] = want
        except ValueError:
            # The manifest was written with absolute paths; a moved directory
            # keeps its tail, which is what the cells are addressed by.
            out[str(pathlib.Path(*name.parts[-3:]))] = want
    return out


def read_verified(path, root, manifest):
    for key in (str(path.relative_to(root)), str(pathlib.Path(*path.parts[-3:]))):
        want = manifest.get(key)
        if want is not None:
            break
    else:
        sys.exit(f"{path} is not in outputs.sha256")
    got = digest(path)
    if got != want:
        sys.exit(f"{path}: digest {got[:16]} against manifest {want[:16]}")
    return json.loads(path.read_text())


def load_blocks(root, manifest, mutate):
    """One record per shadow block: the two statistics the rules read, the two
    decoded costs the dump carries, and what each rule picked."""
    cells = sorted(p for p in root.glob("seed-*/layer-*-*") if p.is_dir())
    if not cells:
        sys.exit(f"no cell under {root}")
    blocks = []
    for cell in cells:
        bundle = read_verified(cell / "bundle.json", root, manifest)
        centroids = bundle["centroids"]
        if len(centroids) != 2:
            sys.exit(f"{cell}: {len(centroids)} centroids, this plan is one gain bit")
        seed = cell.parent.name
        layer = int(cell.name.split("-")[1])
        family = bundle["projection"].split(".")[-1]
        for rf in sorted(cell.glob("row-*.json")):
            d = read_verified(rf, root, manifest)["diagnostic"]
            scale = d["row_scale"]
            for c in d["shadow"]:
                norm = c["source_norm"]
                if mutate == "cost":
                    norm *= 1.0001
                served, euclid = c["choices_ABC"][0], c["choices_ABC"][1]
                if mutate == "rule":
                    # Swap what the two rules picked without touching either
                    # statistic, so control 1 stays green and only control 2
                    # can catch it.
                    served, euclid = euclid, served
                blocks.append(
                    dict(
                        seed=seed,
                        layer=layer,
                        family=family,
                        centroids=centroids,
                        scale=scale,
                        norm=norm,
                        projected=c["projected_gain"],
                        served=served,
                        euclid=euclid,
                        schur=c["choices_ABC"][2],
                        decoded=c["euclidean"],
                    )
                )
    return blocks


def squared_error(b, centroid):
    """`‖x − a u‖² = ‖x‖² − 2a⟨x,u⟩ + a²` at gain `a = centroid · row_scale`."""
    a = centroid * b["scale"]
    return b["norm"] ** 2 - 2.0 * a * b["projected"] + a * a


def nearest2(centroids, x):
    return 1 if abs(centroids[1] - x) < abs(centroids[0] - x) else 0


def quantiles(values):
    v = sorted(values)
    return [v[min(len(v) - 1, round(q * (len(v) - 1)))] for q in QS]


def show_quantiles(label, q):
    print(f"{label:<22}" + "  ".join(f"{x:.4f}" for x in q))


def arm_energy(blocks, shrink, euclidean_rule):
    """Total squared error of one (rule, centroid pair) arm. The direction never
    depends on the gain, so a shrunk pair is scored without re-encoding."""
    total = 0.0
    for b in blocks:
        centroids = [c * shrink for c in b["centroids"]]
        statistic = (b["projected"] if euclidean_rule else b["norm"]) / b["scale"]
        total += squared_error(b, centroids[nearest2(centroids, statistic)])
    return total


def lloyd_max(values, iters=40):
    """`fit_gain_centroids` at `k_bits = 1`: quantile init, then Lloyd sweeps.

    Reproduced here rather than imported, so a refit can be scored on the
    compensated statistics the dumps carry without re-running the encoder.
    """
    g = sorted(values)
    k = 2
    c = [g[(2 * j + 1) * len(g) // (2 * k)] for j in range(k)]
    for _ in range(iters):
        total = [0.0] * k
        count = [0] * k
        for v in g:
            j = min(range(k), key=lambda a: abs(v - c[a]))
            total[j] += v
            count[j] += 1
        for j in range(k):
            if count[j]:
                c[j] = total[j] / count[j]
        c.sort()
    return c


def best_scalar(blocks, euclidean_rule, base):
    """The shrink (or stretch) of the served pair that minimizes squared error.

    One parameter, so it is the estimate least exposed to the noise of a small
    sample, and it bounds what any rescaling of the centroids can be worth.
    """
    best = (float("inf"), None)
    step = 0.0005
    s = 0.90
    while s <= 1.15 + step / 2:
        total = arm_energy(blocks, s, euclidean_rule)
        if total < best[0]:
            best = (total, s)
        s += step
    return best[1], 100.0 * (best[0] - base) / base


def audit(blocks, cells, mean_cos):
    """Four ways the headline could be an artefact, and what each one reads.

    The 2x2 moves one scalar in one direction and sums over blocks of very
    different weight scales. That leaves four questions open, and a negative
    result on a lever is only worth as much as the search behind it.
    """
    base = arm_energy(blocks, 1.0, False)
    print()
    print("=== audit ===")

    print("A. is the mean cosine even the right scalar?")
    s_served, gain_served = best_scalar(blocks, False, base)
    print(
        f"   served rule, best scalar {s_served:.4f}   {gain_served:+.4f} %"
        f"   (mean cosine {mean_cos:.4f} gives "
        f"{100.0 * (arm_energy(blocks, mean_cos, False) - base) / base:+.4f} %)"
    )
    print(
        "   the correction goes UP, not down"
        if s_served > 1.0
        else "   the correction goes down"
    )

    print("B. a full refit, fitted and evaluated on disjoint seeds")
    seeds = sorted({b["seed"] for b in blocks})
    for fit_seed in seeds:
        fit = [b for b in blocks if b["seed"] == fit_seed]
        ev = [b for b in blocks if b["seed"] != fit_seed]
        if not fit or not ev:
            continue
        ev_base = arm_energy(ev, 1.0, False)
        by_norm, by_euclid = {}, {}
        for family, layer in cells:
            g = [b for b in fit if b["family"] == family and b["layer"] == layer]
            if g:
                by_norm[(family, layer)] = lloyd_max([b["norm"] / b["scale"] for b in g])
                by_euclid[(family, layer)] = lloyd_max(
                    [b["projected"] / b["scale"] for b in g]
                )
        cos_fit = sum(b["projected"] / b["norm"] for b in fit) / len(fit)

        def scored(pairs, euclidean_rule):
            total = 0.0
            for b in ev:
                centroids = pairs.get((b["family"], b["layer"]), b["centroids"])
                statistic = (b["projected"] if euclidean_rule else b["norm"]) / b["scale"]
                total += squared_error(b, centroids[nearest2(centroids, statistic)])
            return 100.0 * (total - ev_base) / ev_base

        print(f"   fit on {fit_seed}, evaluated on the rest (its cosine {cos_fit:.6f})")
        print(
            f"     its cosine as a scalar      "
            f"{100.0 * (arm_energy(ev, cos_fit, False) - ev_base) / ev_base:+.4f} %"
        )
        print(f"     Lloyd-Max refit on ||x||    {scored(by_norm, False):+.4f} %")
        print(f"     Lloyd-Max refit on <x,u>    {scored(by_euclid, False):+.4f} %")
        print(
            f"     euclid rule, served pair    "
            f"{100.0 * (arm_energy(ev, 1.0, True) - ev_base) / ev_base:+.4f} %"
        )

    print("C. every cell counts once, instead of by its weight scale")
    for name, shrink, euclidean_rule in (
        ("euclid rule", 1.0, True),
        ("mean-cosine scalar", mean_cos, False),
        ("best scalar", s_served, False),
    ):
        deltas = []
        for seed in seeds:
            for family, layer in cells:
                g = [
                    b
                    for b in blocks
                    if b["seed"] == seed and b["family"] == family and b["layer"] == layer
                ]
                if not g:
                    continue
                cell_base = arm_energy(g, 1.0, False)
                deltas.append(100.0 * (arm_energy(g, shrink, euclidean_rule) - cell_base) / cell_base)
        deltas.sort()
        mid = len(deltas) // 2
        median = deltas[mid] if len(deltas) % 2 else (deltas[mid - 1] + deltas[mid]) / 2
        print(
            f"   {name:<20} mean {sum(deltas) / len(deltas):+.4f} %   "
            f"median {median:+.4f} %   min {deltas[0]:+.4f}   max {deltas[-1]:+.4f}   "
            f"wins {sum(1 for d in deltas if d < 0)}/{len(deltas)}"
        )

    print("D. does the rule survive centroids placed for the OTHER rule?")
    s_euclid, gain_euclid = best_scalar(blocks, True, base)
    print(f"   served rule at its own best scalar   {gain_served:+.4f} %")
    print(f"   euclid rule at its own best scalar   {gain_euclid:+.4f} % (at {s_euclid:.4f})")
    print(f"   the gap, each rule at its best        {gain_euclid - gain_served:+.4f} points")
    wins = 0
    total_cells = 0
    for seed in seeds:
        for family, layer in cells:
            g = [
                b
                for b in blocks
                if b["seed"] == seed and b["family"] == family and b["layer"] == layer
            ]
            if not g:
                continue
            cell_base = arm_energy(g, 1.0, False)
            total_cells += 1
            if best_scalar(g, True, cell_base)[1] < best_scalar(g, False, cell_base)[1]:
                wins += 1
    print(f"   per cell, each rule at its own best scalar: euclid wins {wins}/{total_cells}")


def bias_map(blocks, cells):
    """Where each rule places an amplitude that is not the block's own norm.

    The 2026-09-15 arm measured that this bias is what costs perplexity: a rule
    shrinking every block by 2.93 % lost 4.026 %, where squared error said it
    should win. So the quantity worth mapping over the model is not where the
    rules disagree, it is where the reconstruction is off-centre and by how
    much. Reported per rule and per cell, as the mean of `placed / ‖x‖`.
    """
    print()
    print("=== bias map: mean placed amplitude over block norm ===")
    seeds = sorted({b["seed"] for b in blocks})

    def mean_bias(group, rule):
        return sum(
            b["centroids"][b[rule]] * b["scale"] / b["norm"] for b in group
        ) / len(group)

    for rule, label in (("served", "served rule"), ("euclid", "euclid rule"), ("schur", "Schur rule")):
        print(f"  {label:<14} whole sample {mean_bias(blocks, rule):.5f}"
              f"   ({100.0 * (mean_bias(blocks, rule) - 1.0):+.2f} %)")
    print()
    print(f"  {'cell':<26} {'served':>10} {'euclid':>10} {'Schur':>10}")
    for seed in seeds:
        for family, layer in cells:
            g = [
                b
                for b in blocks
                if b["seed"] == seed and b["family"] == family and b["layer"] == layer
            ]
            if not g:
                continue
            name = f"{seed} {family} L{layer}"
            print(
                f"  {name:<26} {mean_bias(g, 'served'):>10.5f} "
                f"{mean_bias(g, 'euclid'):>10.5f} {mean_bias(g, 'schur'):>10.5f}"
            )
    served = [
        mean_bias(
            [b for b in blocks if b["seed"] == s and b["family"] == f and b["layer"] == y],
            "served",
        )
        for s in seeds
        for f, y in cells
        if [b for b in blocks if b["seed"] == s and b["family"] == f and b["layer"] == y]
    ]
    print()
    print(
        f"  served-rule bias across cells: min {min(served):.5f}  max {max(served):.5f}"
        f"  spread {100.0 * (max(served) - min(served)):.2f} points"
    )
    print("  a spread this size is what decides whether one scalar can serve the whole model")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=pathlib.Path)
    parser.add_argument("--csv", type=pathlib.Path, help="per-cell table")
    parser.add_argument(
        "--bias-map",
        action="store_true",
        help="mean placed amplitude over block norm, per rule and per cell",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="four robustness checks on the 2x2 above",
    )
    parser.add_argument(
        "--mutate",
        choices=("cost", "rule"),
        help="break one control on purpose, to show it is not vacuous",
    )
    args = parser.parse_args()
    root = args.pilot_dir.resolve()
    manifest = load_manifest(root)
    blocks = load_blocks(root, manifest, args.mutate)
    n = len(blocks)

    print("=== gain_disagree_real ===")
    print(f"pilot           {root}")
    print(f"verified        {len(manifest)} entries in outputs.sha256")
    print(f"blocks          {n} shadow comparisons, compensated GPTQ residues")
    if args.mutate:
        print(f"MUTANT          {args.mutate}")

    # ---- control 1: the algebra against the costs the decoder wrote ----
    gap = max(
        abs(squared_error(b, b["centroids"][g]) - b["decoded"][g])
        for b in blocks
        for g in (0, 1)
    )
    largest = max(b["decoded"][g] for b in blocks for g in (0, 1))
    print(
        f"decoder check   worst |algebra - dumped cost| over {2 * n} costs: "
        f"{gap:.3e}  (largest cost {largest:.3e})"
    )
    # ---- control 2: both rules, recomputed from the raw statistics ----
    missA = sum(
        1 for b in blocks if nearest2(b["centroids"], b["norm"] / b["scale"]) != b["served"]
    )
    missB = sum(
        1
        for b in blocks
        if nearest2(b["centroids"], b["projected"] / b["scale"]) != b["euclid"]
    )
    print(f"rule check      A mismatches {missA}, B mismatches {missB} of {n}")
    assert gap < 1e-12, f"the cost model disagrees with the decoder by {gap:.3e}"
    assert missA == 0 and missB == 0, "a rule does not reproduce from its own statistic"
    print()

    # ---- the headline ----
    disagreed = [b for b in blocks if b["served"] != b["euclid"]]
    down = sum(1 for b in disagreed if b["euclid"] < b["served"])
    print("--- disagreement ---")
    print(
        f"served vs euclid  {len(disagreed)} of {n} blocks, "
        f"{100.0 * len(disagreed) / n:.3f} %"
    )
    print(f"  euclid picks lower level  {down}   higher level  {len(disagreed) - down}")
    def occupancy(key):
        return 100.0 * sum(1 for b in blocks if b[key] == 1) / n

    print(
        f"gain occupancy    served {occupancy('served'):.3f} % at level 1, "
        f"euclid {occupancy('euclid'):.3f} %"
    )
    # The Schur rule is read only: it needs the real Hessian factor the pilot
    # held, and it is reported here for its distance to the other two.
    for label, against in (("C vs A", "served"), ("C vs B", "euclid")):
        k = sum(1 for b in blocks if b["schur"] != b[against])
        print(f"{label:<17} {k} of {n} blocks, {100.0 * k / n:.3f} %")
    print()

    # ---- the same, per cell: a rate that is one family is not a rate ----
    print("--- by family and depth ---")
    cells = sorted({(b["family"], b["layer"]) for b in blocks})
    for family, layer in cells:
        g = [b for b in blocks if b["family"] == family and b["layer"] == layer]
        k = sum(1 for b in g if b["served"] != b["euclid"])
        cos = sum(b["projected"] / b["norm"] for b in g) / len(g)
        print(
            f"  {family:<10} layer {layer:>2}  n {len(g):>4}  "
            f"disagree {100.0 * k / len(g):6.3f} %   mean cos {cos:.6f}"
        )
    for seed in sorted({b["seed"] for b in blocks}):
        g = [b for b in blocks if b["seed"] == seed]
        k = sum(1 for b in g if b["served"] != b["euclid"])
        cos = sum(b["projected"] / b["norm"] for b in g) / len(g)
        print(
            f"  {seed:<10}          n {len(g):>4}  "
            f"disagree {100.0 * k / len(g):6.3f} %   mean cos {cos:.6f}"
        )
    print()

    # ---- the geometry that sets the band ----
    mean_cos = sum(b["projected"] / b["norm"] for b in blocks) / n
    print("--- geometry ---")
    show_quantiles("cos(x, u)", quantiles(b["projected"] / b["norm"] for b in blocks))
    show_quantiles("norm / row_scale", quantiles(b["norm"] / b["scale"] for b in blocks))
    print(
        f"mean cos          {mean_cos:.6f}   so the euclidean target sits "
        f"{100.0 * (1.0 - mean_cos):.3f} % below the norm"
    )
    print()

    # ---- the 2x2, at identical rate, format and decoder ----
    energy = sum(b["norm"] ** 2 for b in blocks)
    base = arm_energy(blocks, 1.0, False)
    print("--- 2x2, normalized mse and retention at 2.000 b/dim ---")
    print(f"source energy     {energy:.6e} over {n * DIM} coordinates")
    arms = (
        ("served rule, served centroids", 1.0, False),
        ("euclid rule, served centroids", 1.0, True),
        ("served rule, shrunk centroids", mean_cos, False),
        ("euclid rule, shrunk centroids", mean_cos, True),
    )
    for name, shrink, euclidean_rule in arms:
        total = arm_energy(blocks, shrink, euclidean_rule)
        nmse = total / energy
        retention = 100.0 * (-0.5 * math.log2(nmse)) / RATE_BITS_PER_DIM
        print(
            f"{name:<32} nmse {nmse:.6f}   retention {retention:.4f} %   "
            f"sq. error {100.0 * (total - base) / base:+.4f} %"
        )
    print()
    print(f"shrink scalar     {mean_cos:.6f}, the mean cosine of these blocks")

    if args.audit:
        audit(blocks, cells, mean_cos)

    if args.bias_map:
        bias_map(blocks, cells)

    if args.csv:
        # One row per (seed, family, depth) cell. The four arms are given as
        # squared error relative to that cell's own served arm, so a cell is
        # read without carrying the weight scale of its projection.
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w") as f:
            f.write(
                "seed,family,layer,blocks,disagree,disagree_pct,euclid_lower,"
                "occ_served_pct,occ_euclid_pct,mean_cos,"
                "euclid_served_centroids_pct,served_shrunk_centroids_pct,"
                "euclid_shrunk_centroids_pct\n"
            )
            for seed in sorted({b["seed"] for b in blocks}):
                for family, layer in cells:
                    g = [
                        b
                        for b in blocks
                        if b["seed"] == seed
                        and b["family"] == family
                        and b["layer"] == layer
                    ]
                    if not g:
                        continue
                    k = sum(1 for b in g if b["served"] != b["euclid"])
                    lower = sum(1 for b in g if b["euclid"] < b["served"])
                    cos = sum(b["projected"] / b["norm"] for b in g) / len(g)
                    cell_base = arm_energy(g, 1.0, False)
                    delta = [
                        100.0 * (arm_energy(g, shrink, rule) - cell_base) / cell_base
                        for shrink, rule in ((1.0, True), (mean_cos, False), (mean_cos, True))
                    ]
                    f.write(
                        f"{seed},{family},{layer},{len(g)},{k},{100.0 * k / len(g):.4f},"
                        f"{lower},"
                        f"{100.0 * sum(1 for b in g if b['served'] == 1) / len(g):.4f},"
                        f"{100.0 * sum(1 for b in g if b['euclid'] == 1) / len(g):.4f},"
                        f"{cos:.6f},"
                        + ",".join(f"{d:+.4f}" for d in delta)
                        + "\n"
                    )
        print(f"csv               {args.csv}")


if __name__ == "__main__":
    main()
