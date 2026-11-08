#!/usr/bin/env python3
"""Card-memory peak after an 8,192-token prompt, item by item, from the shapes.

    python3 ops/pic_vram_8k.py

Reads nothing: the geometries are Qwen3's configs, the measured peaks are the duel's
(docs/mesures/duel-iq2-2026-11-08-brut/tetra/vram.txt, peak minus idle, 1 GB = 1e9 B), and
every other line is arithmetic on the code paths named beside it. Three states:

  today     the served prefill as it ran in the duel (model.rs before 2026-11-08):
            chunks of 4,096 rows, grouped attention with three live f16 score tensors,
            the f32 mask uploaded then cast, no stream sync inside a prefill;
  exact     the same arithmetic bit for bit: the softmax written back in place block by
            block (one score tensor and two blocks live), the mask built in f16 on the
            host, the stream synchronised before each prefill chunk and after each
            cache growth, so the stream-ordered pool hands back what the last chunk freed;
  kernel    the prefill rows through the decode-attention kernel, 8 rows a call
            (config field "prefill_attention": "kernel"): no score tensor at all.

Labels: the measured base and peaks are *measured*; tensor sizes are *computed*; the pool
terms (what cuMemAllocAsync keeps between syncs) are *estimated*.
"""

GB = 1e9
F16 = 2

# name: hidden, layers, query heads, kv heads, head dim, parameters, MLP width
MODELS = {
    "4B": (2560, 36, 32, 8, 128, 4_022_468_096, 9728),
    "8B": (4096, 36, 32, 8, 128, 8_190_735_360, 12288),
    "14B": (5120, 40, 40, 8, 128, 14_768_307_200, 17408),
}
# Measured, duel 2026-11-08: peak minus idle after 5 and 8,192 tokens (MiB in vram.txt).
MIB = 1 << 20
BASE_P5 = {"4B": 1921 * MIB, "8B": 3265 * MIB, "14B": 5441 * MIB}
PEAK_8K = {"4B": 10951 * MIB, "8B": 12071 * MIB, "14B": 16295 * MIB}
LLAMA_8K = {"4B": 3.52e9, "8B": 4.71e9, "14B": 6.90e9}  # IQ2_S, same sampler

PROMPT, CHUNK, BLOCK_ROWS = 8192, 4096, 1024


def items(name):
    hidden, layers, nh, nkv, hd, _, inter = MODELS[name]
    l, big_l = CHUNK, PROMPT  # the second chunk: 4,096 rows over 8,192 keys
    kv = layers * 2 * nkv * PROMPT * hd * F16
    s2 = nh * l * big_l * F16  # one score tensor of the second chunk
    s1 = nh * l * l * F16  # one of the first chunk
    mask16 = l * big_l * F16
    mask32 = l * big_l * 4
    # Tensors alive at the second chunk's attention (forward_cached): the residual,
    # ln1, q, k, v out of the projections, q and k after their norms and after RoPE.
    q, kvrow = l * nh * hd * F16, l * nkv * hd * F16
    act = 2 * l * hidden * F16 + 3 * q + 5 * kvrow
    grow_frag = layers * 2 * nkv * CHUNK * hd * F16  # the chunk-1 buffers each layer drops
    block = BLOCK_ROWS * big_l * F16
    # The MLP of the same chunk (mlp_residual): gate, up, act(gate) and their product alive
    # together, f16 out of the rows kernels, beside the block's input, the new residual and ln2.
    # Below the score tensor's moment while one exists; the peak's moment once none does.
    mlp = 4 * l * inter * F16 + 3 * l * hidden * F16
    # The kernel prefill's own attention moment: the activations above, plus its 8-row
    # outputs and their concatenation.
    att_k = act + 2 * q
    return dict(kv=kv, s2=s2, s1=s1, mask16=mask16, mask32=mask32, act=act, frag=grow_frag, block=block,
                mlp=mlp, att_k=att_k)


def main():
    print("Peak after an 8,192-token prompt, GB (1e9 B), second prefill chunk (4,096 rows over 8,192 keys)\n")
    for name in MODELS:
        i = items(name)
        base = BASE_P5[name]
        params = MODELS[name][5]
        att_s = i["s2"] + 2 * i["block"] + i["act"]
        today_lo = base + i["kv"] + 3 * i["s2"] + i["s1"] + i["act"] + i["mask16"]
        today_hi = today_lo + i["frag"]
        exact = base + i["kv"] + max(att_s, i["mlp"]) + i["mask16"]
        kernel = base + i["kv"] + max(i["att_k"], i["mlp"]) + i["mask16"]
        print(f"== {name} ==")
        rows = [
            ("measured at 5 tokens: context, weights, 4E, fixed buffers", base, "measured"),
            ("KV cache, 8,192 positions, f16", i["kv"], "computed"),
            ("score tensors, grouped: 3 live x S", 3 * i["s2"], "computed"),
            ("  one score tensor S (heads x 4,096 x 8,192 x 2 B)", i["s2"], "computed"),
            ("chunk 1's score block kept by the pool (no sync)", i["s1"], "estimated"),
            ("activations alive at the attention, 4,096 rows", i["act"], "computed"),
            ("MLP of the chunk: gate, up, act, product, x, ln2", i["mlp"], "computed"),
            ("causal mask f16 (plus f32 transient before)", i["mask16"], "computed"),
            ("  f32 mask, uploaded then cast", i["mask32"], "computed"),
            ("chunk-1 cache buffers dropped at the growth (up to)", i["frag"], "estimated"),
            ("repeat_kv copies (grouped/kernel forms have none)", 0, "computed"),
            ("eager check in the duel (runs after, same path)", 0, "computed"),
        ]
        for label, v, tag in rows:
            print(f"  {label:<58} {v / GB:6.2f}  {tag}")
        print(f"  {'model today':<58} {today_lo / GB:6.2f}-{today_hi / GB:.2f}  against {PEAK_8K[name] / GB:.2f} measured")
        print(f"  {'exact fix (S + 2 blocks of 1,024 rows, syncs)':<58} {exact / GB:6.2f}")
        print(f"  {'kernel prefill (no score tensor; peak at the MLP)':<58} {kernel / GB:6.2f}")
        print(f"  {'llama.cpp IQ2_S, measured':<58} {LLAMA_8K[name] / GB:6.2f}")
        bp = lambda b: b * 8 / params
        print(
            f"  b/param, whole model at peak: today {bp(PEAK_8K[name]):.1f}, exact {bp(exact):.1f}, "
            f"kernel {bp(kernel):.1f}, llama.cpp {bp(LLAMA_8K[name]):.1f}\n"
        )


if __name__ == "__main__":
    main()
