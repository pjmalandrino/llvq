# FlashInfer single-decode read-rate anchor on the L40S, Qwen3-4B decode geometry.
# Preregistered: proofs/preregistration-fi-decode-anchor-2026-11-04.md, stamped before any run.
# Job: ops/jobs/fi-decode-anchor.sh, image vllm/vllm-openai:v0.26.0 (digest in the job script),
# which ships flashinfer-python, flashinfer-cubin and flashinfer-jit-cache 0.6.14 on torch 2.11.0
# and CUDA 13.0, with nvcc for FlashInfer's runtime JIT. Nothing is installed by this script.
#
#   python3 fi_decode_bench.py --json out.json
#
# For each arm and each length L in {2048, 4096, 8192}:
#   - 36 distinct per-layer KV sets of the 4B geometry: 8 KV heads, head_dim 128, f16, layout HND
#     [8, L, 128]; one query of 8 * G = 32 heads. One pass = 36 calls of
#     flashinfer.single_decode_with_kv_cache, one per layer, and reuses no set;
#   - bytes of one pass = 36 layers x 2 (K and V) x 8 heads x L x 128 x 2 B = 147,456 x L,
#     0.302 GB at 2,048 and 1.208 GB at 8,192, against a 96 MB L2;
#   - the pass is captured once as a CUDA graph and timed with CUDA events around each replay;
#     a pass that cannot be captured is run eagerly, flagged, and read from the profiler's
#     kernel-only sum instead (prereg §3);
#   - before every timed replay a 256 MB buffer is read, so each pass starts with a cold L2;
#   - rounds: 2 discarded, 7 kept. In each round every (arm, L) runs REPS replays in one fixed
#     order, and the round's value is the median of its REPS replays;
#   - set 0 of every (arm, L) is checked against an f32 torch reference.
# eta = bytes of one pass / (t_pass x 864 GB/s). Medians over rounds, ranges from the slowest and
# fastest round; ratios between configurations are formed round by round.

import argparse
import json
import math
import statistics
import sys
import time

BW = 864e9                      # L40S peak DRAM bandwidth, B/s (spec)
N_LAYERS, N_KV, HD = 36, 8, 128  # Qwen3-4B
G = 4                           # 32 query heads over 8 KV heads
LENGTHS = (2048, 4096, 8192)
# P: FlashInfer's CUDA-core decode kernel (the arm the rules read first).
# T4: its tensor-core path at the same G (`use_tensor_cores=True`, the FA2 single-prefill kernel).
ARMS = (("P", False), ("T4", True))
ERR_BOUND = 2e-2                # max |out - ref| / max |ref|, f16; voids a broken arm only
SCRUB_BYTES = 256 << 20
SEED = 20261104


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="FlashInfer single-decode read rate on the Qwen3-4B decode geometry.")
    ap.add_argument("--json", required=True, help="output JSON path")
    ap.add_argument("--warmup", type=int, default=2, help="discarded rounds (default 2)")
    ap.add_argument("--rounds", type=int, default=7, help="kept rounds (default 7)")
    ap.add_argument("--reps", type=int, default=10,
                    help="replays of one (arm, L) inside a round (default 10)")
    a = ap.parse_args(argv)
    if a.rounds < 3 or a.warmup < 0 or a.reps < 1:
        ap.error("need --rounds >= 3, --warmup >= 0, --reps >= 1")
    return a


def pass_bytes(L):
    return N_LAYERS * 2 * N_KV * L * HD * 2


def eta(L, ms):
    return pass_bytes(L) / (ms * 1e-3) / BW


def main(argv=None):
    a = parse_args(argv)
    import torch          # imported here so that --help runs without torch
    import flashinfer

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    info = {"device": props.name, "sms": props.multi_processor_count,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "flashinfer": getattr(flashinfer, "__version__", "?"),
            "bw_peak": BW, "layers": N_LAYERS, "kv_heads": N_KV, "head_dim": HD, "g": G,
            "warmup": a.warmup, "rounds": a.rounds, "reps": a.reps, "seed": SEED}
    print(json.dumps(info), flush=True)

    torch.manual_seed(SEED)
    lmax = max(LENGTHS)
    # One pool per K and V, sliced per length: 36 x 8 x 8192 x 128 f16 = 604 MB each.
    n = N_LAYERS * N_KV * lmax * HD
    kpool = torch.randn(n, device=dev, dtype=torch.float16) * 0.5
    vpool = torch.randn(n, device=dev, dtype=torch.float16) * 0.5
    scrub = torch.ones(SCRUB_BYTES // 2, device=dev, dtype=torch.float16)
    acc = torch.zeros((), device=dev, dtype=torch.float32)

    def cold_l2():
        torch.sum(scrub, dim=0, dtype=torch.float32, out=acc)   # reads 256 MB, writes 4 B

    def sets(L):
        m = N_LAYERS * N_KV * L * HD
        return (kpool[:m].view(N_LAYERS, N_KV, L, HD), vpool[:m].view(N_LAYERS, N_KV, L, HD))

    def call(q, k, v, tc):
        return flashinfer.single_decode_with_kv_cache(
            q, k, v, kv_layout="HND", pos_encoding_mode="NONE", use_tensor_cores=tc)

    def reference(q, k, v):
        kf = k.float().repeat_interleave(G, dim=0)   # query head i reads KV head i // G
        vf = v.float().repeat_interleave(G, dim=0)
        s = torch.einsum("hd,hld->hl", q.float(), kf) / math.sqrt(HD)
        return torch.einsum("hl,hld->hd", torch.softmax(s, dim=-1), vf)

    def kernel_sum_us(q, K, V, tc):
        prof_kind = torch.profiler.ProfilerActivity.CUDA
        cold_l2()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[prof_kind]) as prof:
            for s in range(N_LAYERS):
                call(q, K[s], V[s], tc)
            torch.cuda.synchronize()
        kern = {}
        for ev in prof.key_averages():
            t = getattr(ev, "device_time_total", None)
            if t is None:
                t = getattr(ev, "cuda_time_total", 0)
            name = ev.key or ""
            low = name.lower()
            if t and "memcpy" not in low and "memset" not in low:
                kern[name] = {"us": round(float(t), 2), "calls": int(ev.count)}
        return kern

    results, runs = {}, []
    for L in LENGTHS:
        K, V = sets(L)
        for arm, tc in ARMS:
            key = f"{arm}-L{L}"
            q = torch.randn(N_KV * G, HD, device=dev, dtype=torch.float16)
            r = {"arm": arm, "tensor_cores": tc, "L": L, "bytes": pass_bytes(L), "round_ms": []}
            try:
                t0 = time.time()
                out = call(q, K[0], V[0], tc)           # JIT build or cache load on first use
                torch.cuda.synchronize()
                r["first_call_s"] = round(time.time() - t0, 2)
                ref = reference(q, K[0], V[0])
                r["rel_err"] = float((out.float() - ref).abs().max() / ref.abs().max())
                if not r["rel_err"] <= ERR_BOUND:
                    raise RuntimeError(f"set-0 error {r['rel_err']:.3g} above {ERR_BOUND}")
                for _ in range(2):                      # warm the allocator outside capture
                    for s in range(N_LAYERS):
                        call(q, K[s], V[s], tc)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        for s in range(N_LAYERS):
                            call(q, K[s], V[s], tc)
                    graph.replay()
                    torch.cuda.synchronize()
                    r["timing"] = "graph"
                    run = graph.replay
                except Exception as e:                  # eager fallback, flagged (prereg §3)
                    torch.cuda.synchronize()
                    r["timing"] = f"eager ({type(e).__name__}: {str(e)[:200]})"
                    graph = None

                    def run(q=q, K=K, V=V, tc=tc):
                        for s in range(N_LAYERS):
                            call(q, K[s], V[s], tc)
                r["kernels_one_pass"] = kernel_sum_us(q, K, V, tc)
                r["kernel_sum_us"] = round(sum(v["us"] for v in r["kernels_one_pass"].values()), 2)
                if graph is None:
                    r["round_kernel_sum_us"] = []
                runs.append((key, run, (q, K, V, tc), graph))
            except Exception as e:
                r["void"] = f"{type(e).__name__}: {str(e)[:400]}"
            results[key] = r
            show = {x: r[x] for x in r if x not in ("kernels_one_pass", "round_ms")}
            print(key, json.dumps(show), flush=True)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(a.reps)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(a.reps)]
    for rnd in range(a.warmup + a.rounds):
        for key, run, (q, K, V, tc), graph in runs:      # the same order every round
            for i in range(a.reps):
                cold_l2()
                starts[i].record()
                run()
                stops[i].record()
            torch.cuda.synchronize()
            if rnd < a.warmup:
                continue
            ms = statistics.median(s.elapsed_time(e) for s, e in zip(starts, stops))
            results[key]["round_ms"].append(ms)
            if graph is None:
                k = kernel_sum_us(q, K, V, tc)
                results[key]["round_kernel_sum_us"].append(sum(v["us"] for v in k.values()))
        print(f"round {rnd} {'discarded' if rnd < a.warmup else 'kept'}", flush=True)

    # Per configuration: the time the rules read is the event median for a graph, the
    # profiler's kernel-only sum for an eager pass (prereg §3).
    for key, r in results.items():
        if "void" in r:
            continue
        if r["timing"] == "graph":
            t = r["round_ms"]
        else:
            t = [us / 1e3 for us in r["round_kernel_sum_us"]]
        r["read_ms_by_round"] = t
        r["median_ms"] = statistics.median(t)
        r["range_ms"] = [min(t), max(t)]
        r["eta_median"] = eta(r["L"], r["median_ms"])
        r["eta_range"] = [eta(r["L"], max(t)), eta(r["L"], min(t))]
        r["eta_kernel_only"] = eta(r["L"], r["kernel_sum_us"] / 1e3)
        print(f"{key:10s} {r['median_ms']:8.4f} ms [{r['range_ms'][0]:.4f}-{r['range_ms'][1]:.4f}]"
              f"  eta {r['eta_median']:.3f} [{r['eta_range'][0]:.3f}-{r['eta_range'][1]:.3f}]"
              f"  kernel-only {r['eta_kernel_only']:.3f}  {r['timing']}", flush=True)

    # Derived quantities, round by round (prereg §2). None when an input is void.
    def per_round(f, *keys):
        rs = [results[k] for k in keys]
        if any("void" in r for r in rs):
            return None
        vals = [f(*xs) for xs in zip(*(r["read_ms_by_round"] for r in rs))]
        return {"median": statistics.median(vals), "range": [min(vals), max(vals)],
                "by_round": vals}

    derived = {
        "A8": per_round(lambda t: eta(8192, t), "P-L8192"),
        "F": per_round(lambda t2, t8: eta(2048, t2) / eta(8192, t8), "P-L2048", "P-L8192"),
        "Rtc": per_round(lambda tp, tt: tp / tt, "P-L8192", "T4-L8192"),
        # Marginal read rate from 2,048 to 8,192, the form of our own 0.25-0.26 slope.
        "S_P": per_round(lambda t2, t8: (pass_bytes(8192) - pass_bytes(2048))
                         / (max(t8 - t2, 1e-9) * 1e-3) / BW, "P-L2048", "P-L8192"),
    }
    for name, d in derived.items():
        if d is None:
            print(f"{name}: not formed (an input is void)", flush=True)
        else:
            print(f"{name}: {d['median']:.3f} [{d['range'][0]:.3f}-{d['range'][1]:.3f}]",
                  flush=True)

    with open(a.json, "w") as f:
        json.dump({"info": info, "results": results, "derived": derived}, f, indent=1)
    voids = [k for k, r in results.items() if "void" in r]
    print(f"done: {len(results) - len(voids)} configurations timed, {len(voids)} void: {voids}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
