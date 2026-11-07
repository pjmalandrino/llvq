# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm==0.28.3", "mlx==0.32.4", "numpy==2.5.4", "safetensors==0.8.0", "tokenizers==0.23.3",
#                 "pyarrow==26.0.1", "transformers==5.22.1", "llvq-tetra"]
# [tool.uv.sources]
# llvq-tetra = { path = "../../llvq-tetra" }
# ///
"""Do the MLX gates bite? Each mutant must turn `check.py` or `test_served.py` red.

    uv run ops/mlx_tetra/mutate.py

Works on a copy in a temporary directory, never on the working tree: the
engine, the door, the two Tetra shaders it compiles, the MLX config and the mini
fixture. The unmutated copy must pass first. Each mutant replaces one string
that must occur exactly once (a replacement that matches nothing would read as a
survivor), runs its gate on `fixtures/mini`, and is killed when the gate exits
non-zero. Exit 0 when every mutant is killed.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COPY = ["ops/mlx_tetra", "llvq-llm/kernels/llvq_tetra48.metal", "llvq-llm/kernels/llvq_tetra48_rows.metal",
        "configs/mlx", "llvq-tetra/tests/fixtures/mini"]
ENGINE = "ops/mlx_tetra/llvq_mlx.py"
UNIT = "llvq-llm/kernels/llvq_tetra48.metal"
ROWS = "llvq-llm/kernels/llvq_tetra48_rows.metal"
DOOR = "ops/mlx_tetra/served.py"
SERVE = "ops/mlx_tetra/serve.py"
PPL = "ops/mlx_tetra/ppl.py"

# (name, file, old, new, gate): gate "check" or "served".
MUTANTS = [
    ("rotation sign flip dropped", ENGINE, "s[i] = float(xr[base + i]) * signs[base + i];",
     "s[i] = float(xr[base + i]);", "check"),
    ("rotation 1/sqrt(m) dropped", ENGINE, "yr[base + i] = s[i] * inv;", "yr[base + i] = s[i];", "check"),
    ("rotation mix transposed", ENGINE, "small[g * uint(K) + t]", "small[t * uint(K) + g]", "check"),
    ("rotation block order dropped", ENGINE, "(q / 24u) * 24u + TETRA48_ORDER[q % 24u] : q;", "q : q;", "check"),
    ("stacked q+k: k reads q's gains", ENGINE, "uint go = row < uint(DQ) ? 0u : 2u;", "uint go = 0u;", "check"),
    ("8-row kernel: 7 rows", ENGINE, "uint nrows = min(8u, nr[0] - r0);", "uint nrows = min(7u, nr[0] - r0);",
     "check"),
    ("8-row kernel: gain bit ignored", ENGINE, "float gs = gscale[g];", "float gs = gscale[0];", "check"),
    ("dense prefill: row scale dropped", ENGINE, "float sc = gscale[g] * invnorm[m] * rscale[row];",
     "float sc = gscale[g] * invnorm[m];", "check"),
    ("dense prefill: tail zeroed", ENGINE, "wr[uint(NB) * 24u + t] = tail[row * uint(TW) + t];",
     "wr[uint(NB) * 24u + t] = half(0.0f);", "check"),
    ("fused MLP: SiLU dropped", ENGINE, "y[row] = half(gv / (1.0f + metal::exp(-gv)) * uv);",
     "y[row] = half(gv * uv);", "check"),
    ("fused MLP: up takes gate's row scales", ENGINE, "float uv = fma(au, ru[row], tu_);",
     "float uv = fma(au, rg[row], tu_);", "check"),
    ("rotation reuse: any input", ENGINE, "if self._last is not None and self._last[0] is x:\n            out = self._last[1]\n            self._hits",
     "if self._last is not None:\n            out = self._last[1]\n            self._hits", "check"),
    ("int4 biases zeroed", ENGINE, "    q.biases = mx.array(pm.tensor(f\"{p}.biases\").astype(np.float16))",
     "    q.biases = mx.zeros_like(q.scales)", "check"),
    ("shared epilogue: tail dropped", ROWS, "y[row] = half(fma(acc, rscale[row], tv));",
     "y[row] = half(acc * rscale[row]);", "check"),
    ("shared decoder: suffix bit ignored", UNIT, "b.c3 = t.suffixes[2u * s16 + b3];",
     "b.c3 = t.suffixes[2u * s16];", "check"),
    ("door: kv q8 accepted", DOOR, "        if raw[k] != want:", "        if raw[k] != want and k != \"kv\":",
     "served"),
    ("door: LLVQ_* not refused", DOOR, 'k.startswith("LLVQ_")', 'k.startswith("LLVQ_NONE_")', "served"),
    ("door: any object", DOOR, 'if raw["object"] not in OBJECTS:', "if False:", "served"),
    ("door: Rust keys unnamed", DOOR, "if k in RUST_KEYS:", "if False:", "served"),
    ("door: packed object unchecked", DOOR, "if got != cfg.object:", "if got is None:", "served"),
    ("door: mlx version unchecked", DOOR, "if mlx_version != cfg.mlx:", "if False:", "served"),
    ("serve: environment unchecked", SERVE, "    S.check_env(os.environ)\n", "    pass\n", "served"),
    ("serve: object unchecked", SERVE, '    S.check_object(cfg, json.loads(cfg_file.read_text()))\n', "    pass\n",
     "served"),
    ("serve: generate without a message", SERVE, "            if not rest:\n                raise ValueError(\"generate",
     "            if False:\n                raise ValueError(\"generate", "served"),
    ("serve: reference windows unchecked", PPL, "    if missing:\n        raise ValueError",
     "    if False:\n        raise ValueError", "served"),
    ("door: null quantization_config crashes", DOOR,
     'got = q.get("artifact_sha256") if isinstance(q, dict) else None', 'got = q.get("artifact_sha256")', "served"),
]


def gate(root: Path, which: str) -> tuple[int, str]:
    if which == "check":
        cmd = [sys.executable, str(root / "ops/mlx_tetra/check.py"), str(root / "llvq-tetra/tests/fixtures/mini")]
    else:
        cmd = [sys.executable, str(root / "ops/mlx_tetra/test_served.py")]
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("LLVQ_")}
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    out = [ln for ln in p.stdout.splitlines() if ln.strip()]
    err = [ln for ln in p.stderr.splitlines() if ln.strip() and "Loading weights" not in ln]
    # The gate's verdict line ("... comparisons ...", "... passed"), else the error that stopped it.
    if out and ("comparisons" in out[-1] or "passed" in out[-1]):
        return p.returncode, out[-1]
    return p.returncode, err[-1] if err else (out[-1] if out else "")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="llvq-mlx-mutate-") as tmp:
        root = Path(tmp)
        for rel in COPY:
            src, dst = REPO / rel, root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
        for which in ("check", "served"):
            code, last = gate(root, which)
            print(f"unmutated {which}: exit {code} | {last}")
            if code != 0:
                print("the unmutated copy must pass first")
                return 1
        survivors = 0
        for name, rel, old, new, which in MUTANTS:
            f = root / rel
            pristine = f.read_text()
            n = pristine.count(old)
            if n != 1:
                print(f"  BROKEN {name}: the string occurs {n} times in {rel}")
                survivors += 1
                continue
            f.write_text(pristine.replace(old, new))
            try:
                code, last = gate(root, which)
            finally:
                f.write_text(pristine)
            killed = code != 0
            survivors += not killed
            print(f"  {'killed  ' if killed else 'SURVIVED'} {name:<40} {which:<6} exit {code} | {last[:90]}",
                  flush=True)
        print(f"{len(MUTANTS) - survivors} of {len(MUTANTS)} mutants killed")
        return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
