# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers>=0.21", "pyarrow>=15", "huggingface_hub>=0.24"]
# ///
# The header is the one of ops/gsm8k_vllm.py: `uv run ops/gsm8k_vllm_local.py --check-only ...` works on the Mac,
# and in the vLLM job image `python3 ops/gsm8k_vllm_local.py ...` generates.
"""ops/gsm8k_vllm.py on a model directory on local disk, its code unchanged.

## Why this file exists

`ops/gsm8k_vllm.py` is cited by sha256 in a stamped prereg and is not edited. It takes the tokenizer from
`hf_hub_download(--model, "tokenizer.json", revision=--revision)`, which refuses a path (a path is not a repo id),
and hands vLLM `revision` and `tokenizer_revision`, which a local directory has no use for. The GSM8K depth restore
(ops/jobs/gsm8k-depth-restore-4b.sh) serves three directories that `ops/hybrid_checkpoint.py` builds on the job's
disk, so it needs both to point at the directory.

## What this changes, and nothing else

It imports `gsm8k_vllm` and runs its own `main`, with two substitutions made before `main` runs:
  1. `huggingface_hub.hf_hub_download(<the directory>, "tokenizer.json", ...)` returns `<the directory>/tokenizer.json`.
     Every other call (the GSM8K parquet) goes to the real function, untouched.
  2. `vllm.LLM(model=<the directory>, revision=..., tokenizer_revision=...)` is called with both revisions set to
     None. Every other argument (f16, no prefix cache, max_model_len, seed, no remote code) is `main`'s.
The prompts, the `--check` against a dump of `bin/gsm8k`, the sampling, the dump format: all `main`'s own code.
`--revision` is still required by `main` and lands in the dump header beside the path; the job passes the first 16
hex digits of the directory's model.safetensors sha256, so the header names the file that generated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsm8k_vllm as G  # noqa: E402


def local_dir(argv: list[str]) -> Path:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--model", required=True)
    ap.add_argument("--check-only", action="store_true")
    known, _ = ap.parse_known_args(argv)
    d = Path(known.model)
    if not d.is_absolute() or not d.is_dir():
        raise G.Refused(f"--model {known.model} is not an absolute path to a directory")
    for name in ("tokenizer.json", "config.json"):
        if not (d / name).is_file():
            raise G.Refused(f"{d} holds no {name}")
    return d


def patch(model_dir: Path, generate: bool) -> None:
    import huggingface_hub

    real = huggingface_hub.hf_hub_download

    def hf_hub_download(repo_id, filename, *args, **kwargs):
        if Path(str(repo_id)).resolve() == model_dir.resolve():
            if filename != "tokenizer.json":
                raise G.Refused(f"{filename} asked of the local directory; only tokenizer.json is served")
            return str(model_dir / filename)
        return real(repo_id, filename, *args, **kwargs)

    huggingface_hub.hf_hub_download = hf_hub_download
    if not generate:
        return
    import vllm

    real_llm = vllm.LLM

    def LLM(**kwargs):  # noqa: N802
        if Path(str(kwargs.get("model"))).resolve() != model_dir.resolve():
            raise G.Refused(f"vLLM asked for {kwargs.get('model')}, expected {model_dir}")
        kwargs["revision"] = None
        kwargs["tokenizer_revision"] = None
        print(f"  vLLM model {model_dir} (local directory, revisions cleared)")
        return real_llm(**kwargs)

    vllm.LLM = LLM


def main(argv: list[str]) -> int:
    d = local_dir(argv)
    patch(d, generate="--check-only" not in argv)
    return G.main(argv)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except G.Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)
