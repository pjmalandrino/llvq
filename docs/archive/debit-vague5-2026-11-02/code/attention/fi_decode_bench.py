# DRAFT, never run. FlashInfer single-decode read-rate anchor on the L40S, for
# wave5-deep/prereg-attention-1.md. Not repository code; it would go to ops/fi_decode_bench.py
# only after the operator's go, with the prereg stamped first.
#
# Runs inside the pinned vLLM image, vllm/vllm-openai:v0.26.0, whose tag ships flashinfer 0.6.14
# (flashinfer-python, flashinfer-cubin, flashinfer-jit-cache) and cuda-nvcc for runtime JIT
# (docker/Dockerfile and requirements/cuda.txt at tag v0.26.0, read 2026-11-02). The job checks
# the versions it actually finds before anything is timed.
#
#   python3 fi_decode_bench.py --json out.json
#
# For each arm (cores, G, kv dtype) and each length L in {2048, 4096, 8192}:
#   - 36 distinct per-layer KV sets of the Qwen3-4B geometry: 8 KV heads, head_dim 128, layout HND
#     [n_kv, L, 128] (the Grow store's per-layer layout), one query of 8G heads. One pass reads
#     36 x 2 x 8 x L x 128 elements and reuses no set, 0.30 GB at L = 2048 in f16, above the
#     96 MB L2, so every pass starts cold;
#   - one pass = 36 calls of flashinfer.single_decode_with_kv_cache, captured once as a CUDA graph
#     and timed with CUDA events around each replay (eager fallback, flagged in the JSON);
#   - 2 discarded and 7 timed rounds; inside a round, every (L, arm) pair in one fixed order;
#   - one profiler pass per (L, arm) gives the kernel-only sum and the kernel names;
#   - set 0 is checked against an f32 torch reference; an arm above the error bound is void.
# eta = bytes of one pass / (t_pass x 864 GB/s). Medians and ranges are formed from the rounds.

import argparse
import json
import math
import statistics
import subprocess
import sys
import time

import torch
import flashinfer

BW = 864e9
N_KV, HD, SETS = 8, 128, 36
LENGTHS = (2048, 4096, 8192)
# FlashInfer 0.6.14's CUDA-core decode dispatches group sizes 1, 2, 3, 4 and 8 only
# (DISPATCH_GQA_GROUP_SIZE, include/flashinfer/utils.cuh at v0.6.14). The 14B's G = 5 is bracketed
# by 4 and 8 there; the tensor-core path (the FA2 prefill kernel) is tried at 5 directly.
ARMS = [
    ("cuda", 4, "f16"), ("cuda", 8, "f16"), ("cuda", 4, "e4m3"), ("cuda", 8, "e4m3"),
    ("tc", 4, "f16"), ("tc", 5, "f16"), ("tc", 8, "f16"), ("tc", 4, "e4m3"),
]
DT = {"f16": torch.float16, "e4m3": torch.float8_e4m3fn}
ERR_BOUND = {"f16": 2e-2, "e4m3": 5e-2}   # relative to max |ref|; voids broken arms only


def reference(q, k, v, g):
    """f32 softmax attention with FlashInfer's GQA map (query head i reads KV head i // g)."""
    kf = k.float().repeat_interleave(g, dim=0)          # [8g, L, 128]
    vf = v.float().repeat_interleave(g, dim=0)
    s = torch.einsum("hd,hld->hl", q.float(), kf) / math.sqrt(HD)
    return torch.einsum("hl,hld->hd", torch.softmax(s, dim=-1), vf)


def one_call(q, k, v, cores):
    return flashinfer.single_decode_with_kv_cache(
        q, k, v, kv_layout="HND", pos_encoding_mode="NONE", use_tensor_cores=(cores == "tc"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=7)
    a = ap.parse_args()

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    nvcc = subprocess.run(["bash", "-lc", "command -v nvcc && nvcc --version | tail -2"],
                          capture_output=True, text=True).stdout.strip()
    info = {"device": props.name, "sms": props.multi_processor_count,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "flashinfer": flashinfer.__version__, "nvcc": nvcc}
    print(json.dumps(info), flush=True)

    torch.manual_seed(20261102)
    lmax = max(LENGTHS)
    pool = {}
    for name, dt in DT.items():
        n = SETS * N_KV * lmax * HD
        kp = (torch.randn(n, device=dev, dtype=torch.float16) * 0.5).to(dt)
        vp = (torch.randn(n, device=dev, dtype=torch.float16) * 0.5).to(dt)
        pool[name] = (kp, vp)

    def sets(name, L):
        kp, vp = pool[name]
        m = SETS * N_KV * L * HD
        return kp[:m].view(SETS, N_KV, L, HD), vp[:m].view(SETS, N_KV, L, HD)

    configs, results = [], {}
    for L in LENGTHS:
        for cores, g, kvd in ARMS:
            key = f"{cores}-g{g}-{kvd}-L{L}"
            q = torch.randn(N_KV * g, HD, device=dev, dtype=torch.float16)
            K, V = sets(kvd, L)
            r = {"cores": cores, "g": g, "kv": kvd, "L": L,
                 "bytes": SETS * 2 * N_KV * L * HD * K.element_size()}
            try:
                t0 = time.time()
                out = one_call(q, K[0], V[0], cores)          # JIT or cache load on first use
                torch.cuda.synchronize()
                r["first_call_s"] = round(time.time() - t0, 2)
                ref = reference(q, K[0], V[0], g)
                r["rel_err"] = float((out.float() - ref).abs().max() / ref.abs().max())
                if not r["rel_err"] <= ERR_BOUND[kvd]:
                    raise RuntimeError(f"error {r['rel_err']:.3g} above {ERR_BOUND[kvd]}")
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        for s in range(SETS):
                            one_call(q, K[s], V[s], cores)
                    r["timing"] = "graph"
                    run = graph.replay
                except Exception as e:                      # eager fallback, flagged
                    r["timing"] = f"eager ({type(e).__name__})"
                    def run(q=q, K=K, V=V, cores=cores):
                        for s in range(SETS):
                            one_call(q, K[s], V[s], cores)
                run()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for s in range(SETS):
                        one_call(q, K[s], V[s], cores)
                    torch.cuda.synchronize()
                kern = {}
                for ev in prof.key_averages():
                    t = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
                    if t and ev.key and "memcpy" not in ev.key.lower():
                        kern[ev.key] = round(t, 1)
                r["kernels_us_per_pass"] = kern
                r["times_ms"] = []
                configs.append((key, run))
            except Exception as e:
                r["void"] = f"{type(e).__name__}: {e}"
            results[key] = r
            print(key, {x: r[x] for x in r if x not in ("kernels_us_per_pass",)}, flush=True)

    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for rnd in range(a.warmup + a.rounds):
        for key, run in configs:                           # same order every round
            start.record(); run(); stop.record(); stop.synchronize()
            if rnd >= a.warmup:
                results[key]["times_ms"].append(start.elapsed_time(stop))

    for key, r in results.items():
        if "void" in r:
            continue
        ts = r["times_ms"]
        r["median_ms"] = statistics.median(ts)
        r["range_ms"] = [min(ts), max(ts)]
        r["eta_median"] = r["bytes"] / (r["median_ms"] * 1e-3) / BW
        r["eta_range"] = [r["bytes"] / (max(ts) * 1e-3) / BW, r["bytes"] / (min(ts) * 1e-3) / BW]
        print(f"{key:28s} {r['median_ms']:8.3f} ms  eta {r['eta_median']:.3f} "
              f"[{r['eta_range'][0]:.3f}, {r['eta_range'][1]:.3f}]  {r['timing']}", flush=True)

    with open(a.json, "w") as f:
        json.dump({"info": info, "results": results}, f, indent=1)
    voids = [k for k, r in results.items() if "void" in r]
    print(f"done: {len(results) - len(voids)} arms timed, {len(voids)} void: {voids}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
