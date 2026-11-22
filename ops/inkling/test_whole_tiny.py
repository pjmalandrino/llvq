# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "numpy",
#   "safetensors",
#   "tokenizers",
#   "pyarrow",
#   "transformers @ git+https://github.com/huggingface/transformers@046bdb8cc2",
#   "gguf @ git+https://github.com/ggml-org/llama.cpp@690b87d881#subdirectory=gguf-py",
# ]
# ///
"""Light tests of the whole-model sources on the tiny random Inkling2Loop (CPU, seconds).

    INKLING_THREADS=2 uv run test_whole_tiny.py [scratch_dir]

(1) `attention:<layers>=attn:<dir>` with the native matrices written in B's layout leaves the logits bit-identical,
and a key/value swap or a transposed read changes them; (2) `read_llvqraw` decodes an int4 g64 tensor written in the
`write_raw` framing to `f16(s) * q + f16(b)`, nibble order and group layout included, and an f16 tensor exactly;
(3) a table whose sealed name is not the parameter's is refused; (4) snapshot/restore puts the model back bit for bit.
"""

import _threads  # noqa: F401

import json
import os
import struct
import sys
import tempfile

import numpy as np

import executor  # noqa: E402,F401
from executor import Teacher, torch  # noqa: E402

import model_full  # noqa: E402
from test_lot1_tiny import TinySource, tiny_model  # noqa: E402

torch.set_grad_enabled(False)
INV = {v: k for k, v in model_full.ATTN_NATIVE.items()}


def run(model):
    t = Teacher(device="cpu", model=model)
    t.provider = TinySource(model.config)
    ids = torch.randint(0, 97, (1, 20), generator=torch.Generator().manual_seed(4))
    return t.run(ids, log=None)[0]


def test_attention(tmp):
    model = tiny_model()
    base = run(model)
    d = os.path.join(tmp, "attn")
    os.makedirs(d, exist_ok=True)
    params = dict(model.named_parameters())
    for L in (1, 3):
        for suffix, nat in INV.items():
            params[f"model.layers.{L}.{suffix}"].detach().numpy().tofile(
                os.path.join(d, f"model.layers.{L}.attn.{nat}.weight.f32"))
    names = model_full.tensor_names(model)
    plan = model_full.tensor_plan(model_full.parse_student(f"attention:1+3=attn:{d}"), names)
    assert len(plan) == 12, plan
    snap = model_full.snapshot(model, plan)
    model_full.apply_tensor_plan(model, plan, log=None)
    same = run(model)
    assert torch.equal(same, base)
    k = os.path.join(d, "model.layers.1.attn.wk_dv.weight.f32")
    v = os.path.join(d, "model.layers.1.attn.wv_dv.weight.f32")
    swap = model_full.tensor_plan(model_full.parse_student(
        f"model.layers.1.self_attn.k_proj.linear.weight=file:{v};model.layers.1.self_attn.v_proj.linear.weight=file:{k}"),
        names)
    model_full.apply_tensor_plan(model, swap, log=None)
    d_swap = float((run(model) - base).abs().max())
    model_full.restore(model, snap)
    assert torch.equal(run(model), base)
    q = os.path.join(d, "model.layers.3.attn.wq_du.weight.f32")
    tp = model_full.tensor_plan(model_full.parse_student(f"model.layers.3.self_attn.q_proj.linear.weight=fileT:{q}"),
                                names)
    model_full.apply_tensor_plan(model, tp, log=None)
    d_t = float((run(model) - base).abs().max())
    assert d_swap > 0 and d_t > 0
    try:
        model_full.tensor_plan(model_full.parse_student(f"attention:5=attn:{d}"), names)
        raise AssertionError("a layer without files was accepted")
    except FileNotFoundError:
        pass
    return {"bit_equal_native": True, "max_logit_change_swap": d_swap, "max_logit_change_transposed": d_t}


def write_raw_q4(path, name, dims, q, s16, b16):
    nb = name.encode()
    packed = (q[0::2] | (q[1::2] << 4)).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", 1, len(nb)) + nb + struct.pack("<I", len(dims)))
        f.write(b"".join(struct.pack("<Q", x) for x in dims))
        f.write(struct.pack("<IIQ", 4, 64, packed.size) + packed.tobytes())
        f.write(struct.pack("<Q", s16.size) + s16.astype("<f2").tobytes() + b16.astype("<f2").tobytes())


def test_llvqraw(tmp):
    rng = np.random.default_rng(0)
    rows, cols = 5, 128
    q = rng.integers(0, 16, rows * cols).astype(np.uint8)
    s16 = rng.uniform(0.001, 0.1, rows * 2).astype(np.float16)
    b16 = rng.uniform(-1, 0, rows * 2).astype(np.float16)
    p = os.path.join(tmp, "t.llvqraw")
    write_raw_q4(p, "model.embed.weight", [rows, cols], q, s16, b16)
    name, dims, a = model_full.read_llvqraw(p)
    want = np.empty(rows * cols, np.float32)
    for i in range(rows * cols):
        g = (i // cols) * 2 + (i % cols) // 64
        want[i] = np.float32(s16[g]) * np.float32(q[i]) + np.float32(b16[g])
    assert name == "model.embed.weight" and dims == [rows, cols]
    assert np.array_equal(a.ravel(), want)
    try:
        model_full._read_table(f"table:{p}", "lm_head.weight", (rows, cols))
        raise AssertionError("a mis-named table was accepted")
    except ValueError:
        pass
    got = model_full._read_table(f"table!:{p}", "lm_head.weight", (rows - 1, cols))
    assert got.shape == (rows - 1, cols)
    import gates_whole

    x16 = rng.standard_normal((3, 64)).astype(np.float16)
    p2 = os.path.join(tmp, "f16.llvqraw")
    gates_whole.write_f16_raw(p2, "model.unembed.weight", x16)
    _, _, b = model_full.read_llvqraw(p2)
    assert np.array_equal(b, x16.astype(np.float32))
    return {"q4_decode_exact": True, "f16_roundtrip_exact": True, "misnamed_refused": True}


def main():
    tmp = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="wholetiny-")
    os.makedirs(tmp, exist_ok=True)
    n_fail = 0
    for name, fn in (("attention", test_attention), ("llvqraw", test_llvqraw)):
        try:
            print(f"PASS {name}: {json.dumps(fn(tmp))}", flush=True)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:1500]}", flush=True)
    print(f"test result: {2 - n_fail} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
