# Free check: are the gate_proj and up_proj rows of one SwiGLU neuron aligned in the FP16 checkpoint?
# Reads two tensors a layer straight from the safetensors shards (header parse + mmap), no model load.
import sys, json, glob, struct, mmap, numpy as np
snap = sys.argv[1]; layers = [int(x) for x in sys.argv[2].split(",")]
index = {}
for f in sorted(glob.glob(snap + "/*.safetensors")):
    fh = open(f, "rb"); n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    for k, v in hdr.items():
        if k != "__metadata__": index[k] = (mm, 8 + n, v)
def load(name):
    mm, base, v = index[name]; a, b = v["data_offsets"]; raw = np.frombuffer(mm, dtype=np.uint16 if v["dtype"] in ("BF16", "F16") else np.float32, count=(b - a) // (2 if v["dtype"] in ("BF16", "F16") else 4), offset=base + a)
    if v["dtype"] == "BF16": x = (raw.astype(np.uint32) << 16).view(np.float32)
    elif v["dtype"] == "F16": x = raw.view(np.float16).astype(np.float32)
    else: x = raw
    return x.reshape(v["shape"]).astype(np.float64)
rng = np.random.default_rng(0)
print("layer  mean|cos|  median|cos|  p90|cos|  share>0.5  energy left after 1-scalar prediction  | shuffled-pair mean|cos|")
tot = []
for l in layers:
    g = load(f"model.layers.{l}.mlp.gate_proj.weight"); u = load(f"model.layers.{l}.mlp.up_proj.weight")
    gn = g / np.linalg.norm(g, axis=1, keepdims=True); un = u / np.linalg.norm(u, axis=1, keepdims=True)
    c = (gn * un).sum(1); ac = np.abs(c)
    left = ((1 - c ** 2) * (u ** 2).sum(1)).sum() / (u ** 2).sum()
    perm = rng.permutation(len(c)); cs = np.abs((gn * un[perm]).sum(1)).mean()
    tot.append((ac.mean(), left))
    print("%5d  %.3f      %.3f        %.3f     %.3f      %.3f                                   | %.3f" % (l, ac.mean(), np.median(ac), np.quantile(ac, 0.9), (ac > 0.5).mean(), left, cs))
t = np.array(tot); print("mean over layers: |cos| %.3f, energy left %.3f" % (t[:, 0].mean(), t[:, 1].mean()))
