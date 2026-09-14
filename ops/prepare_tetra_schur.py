#!/usr/bin/env python3
"""Prepare the fixed local pilot inputs. Never run inference or download files."""
import argparse
import hashlib
import json
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--binary", type=pathlib.Path, default=pathlib.Path("target/release/tetra_schur"))
    parser.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/huggingface/hub")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    checkpoint = args.cache / "models--Qwen--Qwen3-0.6B/snapshots" / revision
    corpus = args.cache / "datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1"
    train = corpus / "train-00000-of-00001.parquet"
    validation = corpus / "validation-00000-of-00001.parquet"
    for path in [checkpoint / "config.json", checkpoint / "model.safetensors", checkpoint / "tokenizer.json", train, validation]:
        if not path.is_file():
            raise SystemExit(f"Required local input missing: {path}. No download attempted.")
    # Preserve the pinned snapshot name; resolving its individual symlinks is unnecessary.
    checkpoint = checkpoint.absolute()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    for seed in [1, 2]:
        directory = output / f"seed-{seed}"
        directory.mkdir()
        calibration = directory / "calibration.json"
        held_out = directory / "validation.json"
        for source, dest, count in [(train, calibration, 8), (validation, held_out, 2)]:
            subprocess.run([str(binary), "tokens", str(checkpoint), str(source), str(dest), str(count), "256", str(seed)], check=True)
        plan = dict(version=1, checkpoint=str(checkpoint), revision=revision,
                    calibration=str(calibration), validation=str(held_out), device="metal",
                    layers=[0, 13, 27], projections=["self_attn.q_proj", "mlp.gate_proj"],
                    rows_per_projection=4, snapshots_per_row=3,
                    damping=0.01, h_shrink=1.0, rotation_seed=0x11_0FEED)
        plan_file = directory / "plan.json"
        plan_file.write_text(json.dumps(plan, indent=2) + "\n")
        inspection = subprocess.run([str(binary), "inspect", str(plan_file)], check=True, capture_output=True, text=True)
        (directory / "inspection.json").write_text(inspection.stdout)
        print(plan_file)
        print(inspection.stdout)
    repository = pathlib.Path(__file__).resolve().parent.parent
    sources = [repository / name for name in [
        "Cargo.lock", "llvq-quant/src/schur.rs", "llvq-quant/src/quantizer.rs",
        "llvq-quant/src/gptq.rs", "llvq-quant/src/linalg.rs", "llvq-quant/src/rotation.rs",
        "llvq-llm/src/tetra_diag.rs", "llvq-llm/src/bin/tetra_schur.rs",
        "llvq-llm/src/model.rs", "llvq-llm/src/calib.rs", "llvq-llm/src/loader.rs",
        "ops/prepare_tetra_schur.py", "proofs/BROUILLON-preregistration-tetra-schur-2026-09-14.md",
    ]]
    files = sorted(output.glob("seed-*/*.json")) + sources + [binary, train, validation,
        checkpoint / "config.json", checkpoint / "tokenizer.json", checkpoint / "model.safetensors"]
    lines = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {path.absolute()}\n")
    (output / "preparation-manifest.sha256").write_text("".join(lines))
    print("Prepared two plans, token sets and a SHA-256 manifest. No capture, replay or model evaluation was launched.")


if __name__ == "__main__":
    main()
