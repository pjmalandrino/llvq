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
"""Light tests of `attn_slices.py` on the tiny random Inkling2Loop of `test_lot1_tiny.py` (CPU, seconds).

    INKLING_THREADS=2 uv run test_slices_tiny.py [scratch_dir]

(1) two slices of one sequence, merged, equal the single run of two sequences bit for bit; (2) four sequences in two
slices of two: within float64 reassociation of the single run, and the merge is reproducible bit for bit; (3) a dropped
slice is refused at merge; (4) a duplicated slice and an overlapping one are refused at fold; (5) torn partials (a
leftover tmp directory, a truncated file, a flipped byte) are removed and redone, and the merge then equals (1);
(6) a fold killed midway resumes to the same result; (7) a partial with other settings is refused.
"""

import _threads  # noqa: F401

import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np

import executor  # noqa: E402,F401
from executor import Teacher, torch  # noqa: E402

import attn_slices as S  # noqa: E402
import capture_attn  # noqa: E402
from test_lot1_tiny import TinySource, tiny_model  # noqa: E402

torch.set_grad_enabled(False)
LAYERS = list(range(8))


def setup(n):
    model = tiny_model()
    teacher = Teacher(device="cpu", model=model)
    teacher.provider = TinySource(model.config)
    windows = torch.randint(0, 97, (n, 20), generator=torch.Generator().manual_seed(3)).tolist()
    cfg = {k: getattr(model.config, k) for k in ("num_hidden_layers", "loop_layer_index_start",
                                                 "loop_layer_index_end", "loop_num_passes", "layer_types")}
    settings = {"config": cfg, "corpus_fingerprint": "tiny", "n": n, "len": 20, "layers": LAYERS,
                "streams": list(S.STREAMS), "dtype": "float32", "device": "cpu", "chunk": 7, "micro": 1,
                "model": "tiny"}
    return teacher, windows, settings


def single(teacher, windows, d):
    acc, _ = capture_attn.capture(teacher, torch.tensor(windows), LAYERS, group=len(windows), micro=1, chunk=7,
                                  log=None)
    os.makedirs(d, exist_ok=True)
    return {S.stem(L, s) + ".f64": g for (L, s), g in acc.gram.items()}


def merged(d):
    return {fn: np.fromfile(os.path.join(d, fn), np.float64) for fn in os.listdir(d) if fn.endswith(".f64")}


def expect_refused(fn, what):
    try:
        fn()
    except S.Refused as e:
        return str(e)
    raise AssertionError(f"{what} was not refused")


def test_bit_equal(tmp):
    teacher, windows, st = setup(2)
    ref = single(teacher, windows, os.path.join(tmp, "single"))
    out = os.path.join(tmp, "two")
    S.run_slices(teacher, out, [(0, 1), (1, 2)], windows, st, log=None)
    man = S.merge(out, log=None)
    got = merged(out)
    assert set(got) == set(ref)
    bad = [fn for fn in ref if not np.array_equal(got[fn], ref[fn].ravel())]
    assert not bad, bad
    ok, rep = capture_attn.check(out)
    assert ok, json.dumps(rep)[:2000]
    assert os.path.exists(os.path.join(out, "wh", "L06_q.json")) and man["n_tokens"] == 40
    return {"files_bit_equal": len(ref), "check": ok}


def test_tolerance_and_reproducible(tmp):
    teacher, windows, st = setup(4)
    ref = single(teacher, windows, os.path.join(tmp, "single4"))
    outs = []
    for name in ("r1", "r2"):
        out = os.path.join(tmp, name)
        S.run_slices(teacher, out, [(2, 4), (0, 2)], windows, st, log=None)  # out of order on purpose
        S.merge(out, log=None)
        outs.append(merged(out))
    worst = 0.0
    for fn, r in ref.items():
        r = r.ravel()
        scale = np.abs(r).max()
        worst = max(worst, float(np.abs(outs[0][fn] - r).max() / scale))
        assert np.array_equal(outs[0][fn], outs[1][fn]), fn
    assert worst <= 2 * 2.0**-52, worst
    return {"max_rel_diff_vs_single": worst, "reproducible": True}


def test_dropped(tmp):
    teacher, windows, st = setup(3)
    out = os.path.join(tmp, "dropped")
    S.run_slices(teacher, out, [(0, 1), (2, 3)], windows, st, log=None)
    return expect_refused(lambda: S.merge(out, log=None), "a merge with a missing slice")


def test_duplicate_and_overlap(tmp):
    teacher, windows, st = setup(3)
    out = os.path.join(tmp, "dup")
    S.run_slices(teacher, out, [(0, 1)], windows, st, log=None)
    keep = os.path.join(tmp, "dup_copy")
    shutil.copytree(S.pdir(out, 0, 1), keep)
    S.fold(out, log=None)
    shutil.copytree(keep, S.pdir(out, 0, 1))
    r1 = expect_refused(lambda: S.fold(out, log=None), "a duplicated slice")
    assert "duplicate" in r1, r1
    shutil.rmtree(S.pdir(out, 0, 1))
    S.run_slices(teacher, os.path.join(tmp, "dup_src"), [(0, 2)], windows, st, log=None)
    shutil.copytree(S.pdir(os.path.join(tmp, "dup_src"), 0, 2), S.pdir(out, 0, 2))
    r2 = expect_refused(lambda: S.fold(out, log=None), "an overlapping slice")
    return {"duplicate": r1, "overlap": r2}


def test_torn(tmp):
    teacher, windows, st = setup(2)
    ref = single(teacher, windows, os.path.join(tmp, "single_t"))
    out = os.path.join(tmp, "torn")
    S.run_slices(teacher, out, [(0, 1), (1, 2)], windows, st, log=None)
    # (a) a leftover tmp directory of a killed job
    os.rename(S.pdir(out, 0, 1), S.pdir(out, 0, 1) + ".tmp-999")
    # (b) a truncated file in a final partial
    f = sorted(glob.glob(os.path.join(S.pdir(out, 1, 2), "*.f64")))[0]
    with open(f, "r+b") as fh:
        fh.truncate(os.path.getsize(f) - 8)
    assert not S.is_done(out, 0, 1) and not S.is_done(out, 1, 2)
    redone = S.run_slices(teacher, out, [(0, 1), (1, 2)], windows, st, log=None)
    assert redone == [(0, 1), (1, 2)], redone
    # (c) one flipped byte, same size: caught by the sha256 at fold, removed, then redone
    f = sorted(glob.glob(os.path.join(S.pdir(out, 0, 1), "*.f64")))[0]
    with open(f, "r+b") as fh:
        fh.seek(100)
        b = fh.read(1)
        fh.seek(100)
        fh.write(bytes([b[0] ^ 1]))
    S.fold(out, log=None)
    assert not os.path.exists(S.pdir(out, 0, 1)) and read_led(out)["folded"] == []
    redone2 = S.run_slices(teacher, out, [(0, 1), (1, 2)], windows, st, log=None)
    assert redone2 == [(0, 1)], redone2
    S.merge(out, log=None)
    got = merged(out)
    bad = [fn for fn in ref if not np.array_equal(got[fn], ref[fn].ravel())]
    assert not bad, bad
    return {"redone_after_tmp_and_truncation": redone, "redone_after_flip": redone2, "merge_bit_equal": True}


def read_led(out):
    return S.read_ledger(out)


def test_interrupted_fold(tmp):
    teacher, windows, st = setup(2)
    ref = single(teacher, windows, os.path.join(tmp, "single_i"))
    out = os.path.join(tmp, "interrupt")
    S.run_slices(teacher, out, [(0, 1), (1, 2)], windows, st, log=None)
    S.fold(out, log=None)  # folds both
    out2 = os.path.join(tmp, "interrupt2")
    S.run_slices(teacher, out2, [(0, 1), (1, 2)], windows, st, log=None)
    orig, calls = S._write_array, {"n": 0}

    def dying(p, a):
        if ".k2.f64.tmp" in p:
            calls["n"] += 1
            if calls["n"] == 5:
                raise KeyboardInterrupt("killed mid-fold")
        return orig(p, a)

    S._write_array = dying
    try:
        S.fold(out2, log=None)
        raise AssertionError("the simulated kill did not happen")
    except KeyboardInterrupt:
        pass
    finally:
        S._write_array = orig
    assert read_led(out2)["pending"] == [1, 2]
    S.merge(out2, log=None)
    got = merged(out2)
    bad = [fn for fn in ref if not np.array_equal(got[fn], ref[fn].ravel())]
    assert not bad, bad
    return {"resumed_after_kill_bit_equal": True}


def test_settings_mismatch(tmp):
    teacher, windows, st = setup(2)
    out = os.path.join(tmp, "settings")
    S.run_slices(teacher, out, [(0, 1)], windows, st, log=None)
    S.fold(out, log=None)
    st2 = dict(st, chunk=5)
    S.run_slices(teacher, out, [(1, 2)], windows, st2, log=None)
    return expect_refused(lambda: S.fold(out, log=None), "a partial with other settings")


def main():
    tmp = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="slicetiny-")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    results = {}
    for name, fn in (("bit_equal", test_bit_equal), ("tolerance_and_reproducible", test_tolerance_and_reproducible),
                     ("dropped", test_dropped), ("duplicate_and_overlap", test_duplicate_and_overlap),
                     ("torn", test_torn), ("interrupted_fold", test_interrupted_fold),
                     ("settings_mismatch", test_settings_mismatch)):
        try:
            r = fn(tmp)
            results[name] = "PASS"
            print(f"PASS {name}: {json.dumps(r, default=str)[:400]}", flush=True)
        except Exception as e:  # noqa: BLE001
            results[name] = "FAIL"
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:1500]}", flush=True)
    n_fail = sum(1 for v in results.values() if v != "PASS")
    print(f"test result: {len(results) - n_fail} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
