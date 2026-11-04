# DRAFT, never run. Written read-only on 2026-11-02 for lead "bigmodel"
# (prereg-bigmodel-1.md §3 item 1). Not part of the repository: a sketch of
# the change to ops/vllm_score.py and ops/gguf_mmlu.py that the commit carrying
# the K32 prereg must hold. Nothing here was executed.
#
# The defect. `mmlupair` refuses any dump without the completion marker
#     # end fingerprint=<16 hex> questions=<n>
# (llvq-llm/src/bin/mmlupair.rs:185-196). `bin/mmlu` writes it (mmlu.rs:481-483,
# 854-856): FNV-1a over the token ids of every scored prompt, concatenated in
# scoring order. `vllm_score.emit_dump` (:375-412), which `gguf_mmlu.py` calls,
# writes no such line, so every GGUF and vLLM dump is refused today, the 4B
# engine-gate dump `mmlu-4b-gguf-f16.csv` included.
#
# The fix, three parts.
#
# 1. vllm_score.py: the run fingerprint, bin/mmlu's definition. FNV-1a is a
#    stream, so hashing the concatenation is hashing the prompts one after the
#    other; no list is built.

import itertools


def run_fingerprint(prompt_ids, token_fingerprint):
    """`bin/mmlu`'s run fingerprint: FNV over the concatenated ids, in plan order."""
    return token_fingerprint(itertools.chain.from_iterable(prompt_ids))


def read_trailer(path):
    """(fingerprint, questions) from a dump's `# end` line, or None if it has none."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("# end "):
                kv = dict(x.split("=", 1) for x in line[len("# end "):].split())
                return int(kv["fingerprint"], 16), int(kv["questions"])
    return None


# 2. vllm_score.emit_dump gains one argument, `fingerprint: int`, and writes the
#    trailer as its LAST line, after every row (a trailer is a completion
#    marker: written only when the loop finished):
#
#        fh.write(f"# end fingerprint={fingerprint:016x} questions={len(wanted)}\n")
#
#    Every caller passes it: gguf_mmlu.py (below) and vllm_score.py's own main,
#    which computes it the same way from the ids it already encodes.
#
# 3. gguf_mmlu.py, after `build_prompts` and its qhash check, BEFORE the first
#    request (so a wrong plan costs nothing):
#
#        ids = [tok.encode(p, add_special_tokens=False) for p in prompts]  # the call build_prompts makes
#        fp = vs.run_fingerprint(ids, vs.token_fingerprint)
#        ref = vs.read_trailer(Path(args.reference_dump))
#        if ref is not None and ref[1] == len(wanted) and ref[0] != fp:
#            raise SystemExit(
#                f"run fingerprint {fp:016x} != the reference run's {ref[0]:016x}: "
#                "same questions, other words or another order. NOTHING SCORED.")
#        print(f"  run fingerprint       {fp:016x}")
#        ...
#        vs.emit_dump(..., fingerprint=fp)
#
#    On the 2,280-question plan the expected value is 65dcd53655e8bfa5, the
#    trailer of both docs/data/mmlu-dumps/mmlu-4b-f16.csv and mmlu-14b-f16.csv.
#    The pilot's 100-row reference carries no trailer, so the check is skipped
#    there, and its dump's own trailer names its own 100 prompts.
#
# What this does and does not prove. The fingerprint hashes the ids the HF
# tokenizer gives for the prompts SENT. `gguf_mmlu.py` sends text, so
# llama-server tokenizes again on its side; the engine gate (gate 3, and P0)
# is what checks that both tokenizations agree in effect, not this line.
#
# Tests to add beside test_fingerprint_matches_rust (vllm_score.py):

def test_run_fingerprint_streams(token_fingerprint):
    """The run fingerprint of [[1], [2, 3]] is the fingerprint of [1, 2, 3]."""
    assert run_fingerprint([[1], [2, 3]], token_fingerprint) == token_fingerprint([1, 2, 3])
    assert run_fingerprint([], token_fingerprint) == token_fingerprint([])


# And one on the repository's own dumps, no network: read_trailer of
# mmlu-4b-f16.csv returns (0x65dcd53655e8bfa5, 2280), and read_trailer of
# mmlu-4b-gguf-f16.csv returns None (the defect this fixes, pinned).
#
# Readers that must keep working: vllm_score.read_reference_dump skips every
# `#` line (it already reads mmlu-4b-f16.csv, which has a trailer);
# k32_project.py parses the trailer; mmlupair requires it.
