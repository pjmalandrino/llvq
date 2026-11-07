# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""The MLX door's refusals, without a GPU and without MLX: `uv run ops/mlx_tetra/test_served.py`.

Every refusal must name what it refused. The shipped config must parse, and
must be the only accepted spelling of each key: changing any one value of it
must be refused. Exit 0 when every test passes.
"""
import json
import sys
from pathlib import Path

import served as S

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "mlx" / "qwen3-4b-tetra-e4-mlx.json"
OBJECT = "886391a8c03f66dc269cc65c3598c6627dbdcd259180aff36604ef10d37371b8"


def shipped() -> dict:
    return json.loads(CONFIG.read_text())


def refused(raw: dict, *names: str) -> None:
    try:
        S.parse(raw, Path("t.json"), "0" * 64)
    except S.Refused as e:
        for n in names:
            assert n in str(e), f"the refusal must name {n!r}: {e}"
        return
    raise AssertionError(f"accepted: {raw}")


def test_the_shipped_config_parses():
    cfg = S.load(CONFIG)
    assert cfg.object == OBJECT and cfg.mlx == "0.32.4" and cfg.mlx_lm == "0.28.3"
    assert "operator" in cfg.note and "Sur métal, on a validé mlx donc implem le aussi" in cfg.note
    assert cfg.line().startswith(f"served config: {CONFIG}")


def test_every_other_value_of_every_key_is_refused_by_name():
    others = {
        "engine": ["rust", "candle", ""],
        "layout": ["planes14", "slot32", "golay70", "Tetra48"],
        "embed": ["f16", "q8"],
        "rot_share": ["0", "true"],
        "kv": ["q8", "q4", "int8"],
        "activations": ["f32", "bf16"],
        "attention": ["expanded", "grouped", "kernel", "shared"],
    }
    assert set(others) == set(S.ACCEPTED), "a key without a refusal test"
    for key, values in others.items():
        for v in values:
            raw = shipped() | {key: v}
            refused(raw, key, repr(v))


def test_another_object_is_refused():
    for obj in ("7bdb9a55" + "0" * 56, "61db37fe" + "0" * 56, OBJECT.upper()):
        refused(shipped() | {"object": obj}, "object")


def test_rust_door_keys_are_refused_by_name():
    for key, v in (("fuse", "0"), ("decode", "persistent"), ("rows", "v4")):
        refused(shipped() | {key: v}, key, "Rust door")


def test_unknown_missing_and_non_string_keys_are_refused():
    refused(shipped() | {"embeding": "q4"}, "embeding")
    for key in S.REQUIRED:
        raw = shipped()
        del raw[key]
        refused(raw, key, "required")
    refused(shipped() | {"rot_share": 1}, "rot_share", "string")
    refused(shipped() | {"mlx": 0.32}, "mlx", "string")


def test_every_llvq_variable_is_refused_by_name():
    S.check_env({"HOME": "/x", "MLX_FOO": "1"})
    for env in ({"LLVQ_KV": "q8"}, {"LLVQ_MLX_ABLATE": "tetra"}, {"LLVQ_CONFIG": "x", "LLVQ_DTYPE": "f32"}):
        try:
            S.check_env(env)
        except S.Refused as e:
            for k in env:
                assert k in str(e), e
            continue
        raise AssertionError(f"accepted {env}")


def test_the_packed_object_must_be_the_configs():
    cfg = S.load(CONFIG)
    S.check_object(cfg, {"quantization_config": {"artifact_sha256": OBJECT}})
    mini = json.loads((CONFIG.parents[2] / "llvq-tetra/tests/fixtures/mini/config.json").read_text())
    for packed in (mini, {}, {"quantization_config": {}}):
        try:
            S.check_object(cfg, packed)
        except S.Refused as e:
            assert "artifact" in str(e), e
            continue
        raise AssertionError(f"accepted {str(packed)[:80]}")


def test_another_runtime_is_refused():
    cfg = S.load(CONFIG)
    S.check_runtime(cfg, "0.32.4", "0.28.3", "Device(gpu, 0)")
    for args, name in ((("0.29.3", "0.28.3", "Device(gpu, 0)"), "mlx 0.29.3"),
                       (("0.32.4", "0.28.4", "Device(gpu, 0)"), "mlx-lm 0.28.4"),
                       (("0.32.4", "0.28.3", "Device(cpu, 0)"), "cpu")):
        try:
            S.check_runtime(cfg, *args)
        except S.Refused as e:
            assert name in str(e), e
            continue
        raise AssertionError(f"accepted {args}")


def test_serve_checks_before_it_imports_mlx():
    """`serve.open_door` refuses before any MLX import: this file's environment has no MLX."""
    import os

    import serve

    mini = str(CONFIG.parents[2] / "llvq-tetra/tests/fixtures/mini")
    saved = dict(os.environ)
    try:
        os.environ["LLVQ_KV"] = "q8"
        for packed in (mini, "/nonexistent"):
            try:
                serve.open_door(str(CONFIG), packed)
            except S.Refused as e:
                assert "LLVQ_KV" in str(e), e
            else:
                raise AssertionError("LLVQ_KV accepted")
        del os.environ["LLVQ_KV"]
        for packed, name in ((mini, "artifact"), ("/nonexistent", "no packed model")):
            try:
                serve.open_door(str(CONFIG), packed)
            except S.Refused as e:
                assert name in str(e), e
            else:
                raise AssertionError(f"{packed} accepted")
    finally:
        os.environ.clear()
        os.environ.update(saved)


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    bad = 0
    for name, f in tests:
        try:
            f()
            print(f"  ok   {name}")
        except Exception as e:  # an import error from a mutant is a failure too
            bad += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - bad} of {len(tests)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
