#!/usr/bin/env python3
# NOTE: there is deliberately **no PEP 723 header** on this file, unlike
# `ops/awq_dequant.py`. This script does not run under `uv`: it runs inside the
# pinned `vllm/vllm-openai` image, where vLLM, torch and their whole CUDA stack
# are already installed and *are the thing being measured*. A dependency header
# would invite `uv` to resolve a second vLLM on top of the image's, and the
# version in the journal would then be a lie.
#
# Imports are therefore **stdlib + vllm only**. No `huggingface_hub`, no
# `transformers`, no `numpy`, and above all nothing from the LLVQ repository:
# this file is copied into a bucket and executed alone.
"""Chronometer the official AWQ 4-bit checkpoints **in their own engine** (vLLM).

Why this file exists
--------------------
The project's comparison tables have one empty cell: the AWQ 4-bit arm has no
tok/s, at any of the three sizes. Every LLVQ throughput figure comes from
`llvq-llm/src/bin/fusedrun.rs`, which cannot load an AWQ checkpoint, and the AWQ
kernel we *did* time (2026-08-10, six-arm bench) is a port of `mit-han-lab`'s
GEMV into our own harness — a kernel measurement, not a product measurement.

The operator's decision: measure AWQ **in vLLM**, and declare the engine
confounder rather than pretend it away.

What this script is NOT allowed to produce
------------------------------------------
A ratio that crosses the two stacks. It emits AWQ/f16 **within vLLM** and
nothing else; the LLVQ/f16 ratio stays where it was measured, in our harness.
See `proofs/preregistration-awq-vllm-2026-08-17.md` §3 and §4.

The protocol replicated from `fusedrun`
---------------------------------------
* prompt `"The capital of France is"` passed as **raw ids** `[785, 6722, 315,
  9625, 374]` — never re-tokenized, so no chat template can slip in;
* **128** tokens, greedy, `dtype=float16`;
* wall clock around the single `generate` call, **prefill included**,
  `rate = 128 / elapsed`.

Where it deliberately differs, and says so
------------------------------------------
* **2 discarded rounds then 5 timed** per arm, against `fusedrun`'s 1 + 1: the
  card ramps its clocks and vLLM captures its CUDA graphs on the first call;
* the ratio is the **median of 5 ratios formed round by round**, with its
  min–max range — never a quotient of two minima from rounds that never
  coexisted.

Exit codes (`cmd_bench` runs under `set -euo pipefail`, so a non-zero exit turns
the job ERROR instead of leaving it COMPLETED on an empty result)
-------------------------------------------------------------------------------
0  every arm measured, every guard green
2  configuration refused **before** anything is loaded or billed
3  load control failed: the f16 arm's first token is not the published one
4  §7 violation: token count, prefix caching, inter-round spread, unreadable
   engine config
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from typing import Any

# --- the objects, and the protocol constants --------------------------------

# `"The capital of France is"` under Qwen3's tokenizer, verified 2026-08-17 with
# `tokenizers` against the `tokenizer.json` of Qwen3-4B, Qwen3-4B-AWQ,
# Qwen3-8B-AWQ and Qwen3-14B-AWQ: the four files share one sha256
# (`aeb13307a71acd8f…`, the constant pinned in `ops/awq_dequant.py`), so the
# token fingerprint is common **by construction**, not by luck.
PROMPT_TEXT = "The capital of France is"
PROMPT_IDS = [785, 6722, 315, 9625, 374]

# The first token the published f16 arm emits at 4B, from
# `docs/mesures/planes14-fusedrun-2026-08-06.txt` (" Paris. (True or False?)…").
# The id is *derived*: `" Paris"` encodes to `[12095]` and 12095 decodes back to
# `" Paris"`, so it is the only token that can start that text. Kept alongside
# the text because an id mismatch with a text match is a tokenizer-version
# question, not a wrong-model question — and the two failures deserve different
# verdicts.
REF_FIRST_TOKEN_ID = 12095
REF_FIRST_TOKEN_TEXT = " Paris"
REF_TEXT_4B = (
    " Paris. (True or False?)\nThe statement \"The capital of France is Paris\" "
    "is **False**."
)

N_TOKENS = 128
WARMUPS = 2
ROUNDS = 5
SPREAD_MAX_PCT = 5.0  # §7.4

# Repositories and revisions. The 4B and 14B revisions are the ones pinned in
# `ops/awq_dequant.py`'s EXPECTED map — i.e. the ones its five structural
# controls were run against — and both were re-read on the Hub on 2026-08-17
# and found identical to `main`.
#
# 🚨 The 8B has **no EXPECTED entry anywhere in the repository**. Its SHAs below
# were read off the Hub on 2026-08-17 and have never passed `awq_dequant check`.
# That is why `pinned=False` refuses the size by default: a revision that nobody
# validated is not a pin, it is a snapshot.
SIZES: dict[str, dict[str, Any]] = {
    "4b": dict(
        awq_repo="Qwen/Qwen3-4B-AWQ",
        awq_rev="74d4bd2bd4bff9cafc9345221320bffb08b406a3",
        base_repo="Qwen/Qwen3-4B",
        base_rev="1cfa9a7208912126459214e8b04321603b3df60c",
        base_gb=8.05,
        awq_gb=2.67,
        pinned=True,
    ),
    "8b": dict(
        awq_repo="Qwen/Qwen3-8B-AWQ",
        awq_rev="4da05a8edb55c6046cce958586c33b61da07bb79",
        base_repo="Qwen/Qwen3-8B",
        base_rev="b968826d9c46dd6066d109eabc6255188de91218",
        base_gb=16.38,
        awq_gb=6.10,
        pinned=False,
    ),
    "14b": dict(
        awq_repo="Qwen/Qwen3-14B-AWQ",
        awq_rev="31c69efc29464b6bb0aee1398b5a7b50a99340c3",
        base_repo="Qwen/Qwen3-14B",
        base_rev="40c069824f4251a91eefaf281ebe4c544efd3e18",
        base_gb=29.54,
        awq_gb=9.98,
        pinned=True,
    ),
}

# name -> (which repo, what to pass as `quantization`)
#
# `awq_marlin` is the **default** path: on sm_89 vLLM detects that an AWQ
# checkpoint is convertible and repacks it into Marlin at load. It is therefore
# what a user actually gets, and it is the arm the published figure carries.
# `awq` forces the AutoAWQ GEMM, the kernel family our 2026-08-10 bench ported —
# it exists here to connect the two measurements, not to be published alone.
ARMS: dict[str, tuple[str, str | None]] = {
    "f16": ("base", None),
    "awq_marlin": ("awq", None),
    "awq": ("awq", "awq"),
}
DEFAULT_ARM_ORDER = ["f16", "awq_marlin", "awq"]

BANNER = "=" * 78


# --- guards that run before a single byte is loaded -------------------------


class Refused(Exception):
    """Configuration refused. Exit 2 — nothing has been loaded or billed yet."""


class Violation(Exception):
    """A §7 invalidation. Exit 4."""


class ControlFailed(Exception):
    """The load control diverged at token 1. Exit 3 — the job is false."""


def check_image_pin() -> tuple[str, str]:
    """The image tag cannot be discovered from inside the container.

    So the launcher passes it, and this refuses to run without it. §7.5 makes an
    unpinned image an invalidation, and a guard that only fires at report time
    would fire after the money is spent.
    """
    tag = os.environ.get("LLVQ_IMAGE_TAG", "").strip()
    digest = os.environ.get("LLVQ_IMAGE_DIGEST", "").strip()
    if not tag:
        raise Refused(
            "LLVQ_IMAGE_TAG missing. The container does not know its own tag:\n"
            "  the launcher must export it, otherwise the journal does not say\n"
            "  what was measured and the run is not replayable (§7.5)."
        )
    if "latest" in tag:
        raise Refused(
            f"LLVQ_IMAGE_TAG={tag!r}: `latest` is a moving tag.\n"
            "  Pin a version, and preferably its digest (§2.5)."
        )
    return tag, digest


def check_repo(repo: str, revision: str, pinned: bool, allow_unpinned: bool) -> None:
    """Two ways to measure the wrong thing, both silent, both caught here."""
    if repo.endswith("-deq") or "-awq-deq" in repo:
        raise Refused(
            f"{repo} is one of OUR dense f16 reconstructions: it holds NO nibble\n"
            "  at all. Timing it would produce an \"AWQ arm\" that is in fact an\n"
            "  f16 arm with degraded weights. Plausible, wrong, and invisible in\n"
            "  the logs."
        )
    if not revision:
        raise Refused(f"{repo}: no revision. A repository moves, a measurement does not.")
    if not pinned and not allow_unpinned:
        raise Refused(
            f"{repo}@{revision}: revision READ OFF the Hub, never validated by\n"
            "  `ops/awq_dequant.py check` (no EXPECTED entry). Force it with\n"
            "  --allow-unpinned-revision, and then say so in every published figure."
        )


# --- reading the engine back, defensively -----------------------------------


class _Missing:
    """Distinct from `None`, which is a *legitimate* value for `quantization`.

    Conflating "the field says no quantization" with "the field could not be
    found" is exactly how a guard reports green on a config it never read.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNREADABLE"


MISSING = _Missing()


def dig(root: Any, *paths: str) -> Any:
    """Follow the first attribute chain that resolves; `MISSING` if none does.

    vLLM has moved `model_config` and `cache_config` between `llm_engine` and
    `llm_engine.vllm_config` more than once. Guessing one path and asserting on
    it would turn a rename into a failed job; guessing none and *assuming* the
    flags took effect is worse — §7.3 requires the value to be read back, not
    supposed. So: try the known shapes, and if none resolves, say so loudly.
    """
    for path in paths:
        cur = root
        for part in path.split("."):
            cur = getattr(cur, part, MISSING)
            if cur is MISSING:
                break
        else:
            return cur
    return MISSING


def _s(v: Any) -> str:
    if v is MISSING:
        return "UNREADABLE"
    if v is None:
        return "none"
    return str(v)


def engine_facts(llm: Any) -> dict[str, Any]:
    quant = dig(
        llm,
        "llm_engine.vllm_config.model_config.quantization",
        "llm_engine.model_config.quantization",
        "vllm_config.model_config.quantization",
        "model_config.quantization",
    )
    prefix = dig(
        llm,
        "llm_engine.vllm_config.cache_config.enable_prefix_caching",
        "llm_engine.cache_config.enable_prefix_caching",
        "vllm_config.cache_config.enable_prefix_caching",
        "cache_config.enable_prefix_caching",
    )
    dtype = dig(
        llm,
        "llm_engine.vllm_config.model_config.dtype",
        "llm_engine.model_config.dtype",
        "vllm_config.model_config.dtype",
    )
    return {
        "quantization_resolved": _s(quant),
        "quantization_readable": quant is not MISSING,
        "prefix_caching": prefix if isinstance(prefix, bool) else None,
        "prefix_caching_readable": isinstance(prefix, bool),
        "dtype": _s(dtype),
    }


# --- capturing vLLM's own decision about the kernel -------------------------


class LogTap(logging.Handler):
    """Keep vLLM's log lines about quantization routing.

    This is a datum of the verdict, not noise: on sm_89 vLLM repacks AWQ into
    Marlin at load and *says so*. Which kernel was timed is exactly the thing a
    reader of the published number needs.
    """

    KEYS = ("marlin", "awq", "quantization", "quant_method", "kernel", "gptq")

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # a broken format string must not kill the bench
            return
        low = msg.lower()
        if any(k in low for k in self.KEYS):
            self.lines.append(f"[{record.name}] {msg}")

    def install(self) -> None:
        for name in ("", "vllm"):
            logging.getLogger(name).addHandler(self)
        logging.getLogger("vllm").setLevel(logging.INFO)

    def since(self, mark: int) -> list[str]:
        return self.lines[mark:]


# --- the measurement --------------------------------------------------------


def build_llm(spec: dict[str, Any], arm: str, args, util: float):
    from vllm import LLM

    which, quantization = ARMS[arm]
    repo = spec["awq_repo"] if which == "awq" else spec["base_repo"]
    rev = spec["awq_rev"] if which == "awq" else spec["base_rev"]

    kwargs = dict(
        model=repo,
        revision=rev,
        tokenizer_revision=rev,
        # §2.3 (1): the base checkpoints are bfloat16 on the Hub while both our
        # arms are f16. Without this, the "f16 control" would be a bf16 control.
        dtype="float16",
        # §2.3 (2): prefix caching is ON by default in V1. With it, rounds 2..n
        # skip the 5-token prefill and the min–max range stops describing the
        # card and starts describing a cache.
        enable_prefix_caching=False,
        # §2.3 (4): the 0.9 default preallocates ~43 GB of a 48 GB L40S. Three
        # engines could not coexist, and any memory reading would report a
        # reservation rather than an occupancy.
        gpu_memory_utilization=util,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
        seed=0,
        disable_log_stats=True,
        trust_remote_code=False,
    )
    if quantization is not None:
        kwargs["quantization"] = quantization
    return LLM(**kwargs)


def sampling_params():
    from vllm import SamplingParams

    # §2.3 (3): our `generate` does not know about EOS — it emits 128 tokens no
    # matter what. Without `ignore_eos`/`min_tokens`, vLLM stops at EOS and this
    # script would divide 128 by the time of fewer than 128 tokens.
    #
    # `top_k` is deliberately not passed: its "disabled" sentinel has changed
    # value across vLLM versions, and `temperature=0.0` already selects the
    # greedy path — which is what `fusedrun`'s argmax is.
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        n=1,
        max_tokens=N_TOKENS,
        min_tokens=N_TOKENS,
        ignore_eos=True,
    )


def make_prompt():
    try:
        from vllm.inputs import TokensPrompt

        return TokensPrompt(prompt_token_ids=list(PROMPT_IDS))
    except Exception:
        return {"prompt_token_ids": list(PROMPT_IDS)}


def one_generation(llm, prompt, sp) -> tuple[float, list[int], str]:
    t0 = time.perf_counter()
    outs = llm.generate([prompt], sp, use_tqdm=False)
    elapsed = time.perf_counter() - t0
    comp = outs[0].outputs[0]
    return elapsed, list(comp.token_ids), comp.text


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# --- report -----------------------------------------------------------------


def summarize(rates: list[float]) -> dict[str, float]:
    med = statistics.median(rates)
    lo, hi = min(rates), max(rates)
    return dict(
        median=med,
        min=lo,
        max=hi,
        spread_pct=(hi - lo) / med * 100.0 if med else float("inf"),
    )


def ratios_round_by_round(num: list[float], den: list[float]) -> dict[str, Any]:
    """The house rule n°2, applied literally.

    A ratio is formed **inside a round** and the median is taken over ratios —
    never the quotient of two aggregates, which would pair numbers from rounds
    that never coexisted.
    """
    per = [n / d for n, d in zip(num, den)]
    return dict(
        per_round=per,
        median=statistics.median(per),
        min=min(per),
        max=max(per),
    )


def run(args) -> int:
    tag, digest = check_image_pin()
    spec = SIZES[args.size]

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise Refused(f"unknown arm(s): {unknown}, known: {list(ARMS)}")
    if "f16" not in arms:
        raise Refused(
            "the f16 arm is the within-stack control: without it there is no\n"
            "  publishable ratio, only absolute tok/s (§3)."
        )
    # fixed dispatch order, independent of the order typed on the command line
    arms = [a for a in DEFAULT_ARM_ORDER if a in arms]

    for arm in arms:
        which, _ = ARMS[arm]
        check_repo(
            spec["awq_repo"] if which == "awq" else spec["base_repo"],
            spec["awq_rev"] if which == "awq" else spec["base_rev"],
            spec["pinned"],
            args.allow_unpinned_revision,
        )

    utils = {a: (args.gpu_util_f16 if a == "f16" else args.gpu_util_quant) for a in arms}
    total_util = sum(utils.values())
    interleaved = not args.one_arm
    if interleaved and total_util > 0.85:
        raise Refused(
            f"total utilization {total_util:.2f} > 0.85: three vLLM engines\n"
            "  will not fit in one process. Run one arm per process with\n"
            "  --one-arm <name> --json <file>, then --merge, and the report will\n"
            "  be labelled \"rounds NOT interleaved\" (§2.6)."
        )
    if args.one_arm:
        if args.one_arm not in arms:
            raise Refused(f"--one-arm {args.one_arm} not in --arms")
        arms = [args.one_arm]

    try:
        import vllm
    except ImportError as e:  # pragma: no cover - only outside the image
        raise Refused(
            f"vLLM not found ({e}). This script runs INSIDE the pinned\n"
            "  vllm/vllm-openai image, not in our image nor on the dev machine.\n"
            "  That is why it has no dependency on the LLVQ repository."
        ) from None

    tap = LogTap()
    tap.install()

    print(BANNER)
    print("AWQ SPEED BENCH, inside vLLM, its own engine")
    print(BANNER)
    print(f"  image                 {tag}")
    print(f"  digest                {digest or '(not provided)'}")
    print(f"  vllm                  {getattr(vllm, '__version__', '?')}")
    print(f"  size                  {args.size}")
    print(f"  arms (fixed order)    {arms}")
    print(f"  prompt ids            {PROMPT_IDS}  ({len(PROMPT_IDS)} tokens, not re-tokenized)")
    print(f"  tokens generated      {N_TOKENS} (min=max, ignore_eos)")
    print(f"  rounds                {args.warmups} discarded + {args.rounds} timed")
    print(f"  interleaving          {'YES' if interleaved else 'NO, rounds NOT interleaved'}")
    print(f"  gpu_memory_util       {utils}")
    print(f"  max_model_len         {args.max_model_len}")
    print(BANNER)
    if not interleaved:
        print("WARNING: sequential mode. The arms do not coexist. Any ratio formed")
        print("    by --merge is labelled \"rounds not interleaved\" and is not cited")
        print("    as an interleaved ratio (§2.6 of the preregistration).")
        print(BANNER)

    prompt = make_prompt()
    sp = sampling_params()
    engines: dict[str, Any] = {}
    facts: dict[str, dict[str, Any]] = {}

    for arm in arms:
        mark = len(tap.lines)
        t0 = time.perf_counter()
        engines[arm] = build_llm(spec, arm, args, utils[arm])
        load_s = time.perf_counter() - t0
        f = engine_facts(engines[arm])
        f["load_s"] = load_s
        f["gpu_memory_utilization"] = utils[arm]
        f["kernel_log"] = tap.since(mark)
        facts[arm] = f

        which, requested = ARMS[arm]
        print(f"\n--- arm {arm} loaded in {load_s:.1f} s ---")
        print(f"  repository             {spec['awq_repo'] if which == 'awq' else spec['base_repo']}")
        print(f"  revision               {spec['awq_rev'] if which == 'awq' else spec['base_rev']}")
        print(f"  quantization requested {requested!r}")
        print(f"  quantization resolved  {f['quantization_resolved']}")
        print(f"  dtype resolved         {f['dtype']}")
        print(f"  prefix caching         {f['prefix_caching']}")
        if f["kernel_log"]:
            print("  vLLM log on the selected kernel:")
            for line in f["kernel_log"]:
                print(f"    | {line}")
        else:
            print("  WARNING: no log line about routing. The resolved method above")
            print("     is then the ONLY evidence of which kernel runs.")

        # §7.3 — read back, never suppose. Checked on the FIRST arm, before the
        # others are built and before any generation: an unreadable config must
        # cost two minutes, not the whole job.
        if f["prefix_caching"] is True:
            raise Violation(
                f"arm {arm}: prefix caching ACTIVE despite the flag. Rounds 2..n\n"
                "  would skip the prefill and the range would become an artifact."
            )
        if not f["prefix_caching_readable"]:
            raise Violation(
                f"arm {arm}: `enable_prefix_caching` unreadable in the engine\n"
                "  config (known paths exhausted). §7.3 requires a READ BACK, not\n"
                "  a supposition: this vLLM version moved the field, and its path\n"
                "  must be added to `dig()` before anything is measured."
            )

    # --- rounds, interleaved in a fixed dispatch order ----------------------
    elapsed: dict[str, list[float]] = {a: [] for a in arms}
    texts: dict[str, str] = {}
    firsts: dict[str, list[int]] = {}
    ntok: dict[str, list[int]] = {a: [] for a in arms}

    print(f"\n{BANNER}")
    print("ROUNDS")
    print(BANNER)
    for r in range(args.warmups + args.rounds):
        kept = r >= args.warmups
        for arm in arms:
            dt, ids, text = one_generation(engines[arm], prompt, sp)
            n = len(ids)
            if kept:
                elapsed[arm].append(dt)
                ntok[arm].append(n)
                texts[arm] = text
                firsts[arm] = ids[:4]
            tagr = f"round {r - args.warmups + 1}" if kept else f"warmup {r + 1}"
            print(f"  {tagr:<10} {arm:<12} {dt:7.3f} s   {n:4d} tokens   "
                  f"{n / dt:7.2f} tok/s")
            # §7.2 — a short generation is not a slow one, it is a wrong one.
            if n != N_TOKENS:
                raise Violation(
                    f"arm {arm}, {tagr}: {n} tokens returned instead of {N_TOKENS}.\n"
                    "  Dividing 128 by the time of fewer than 128 tokens overstates\n"
                    "  the throughput by an unbounded factor."
                )

    # --- the load control ---------------------------------------------------
    control: dict[str, Any] = {"applicable": args.size == "4b" and "f16" in arms}
    if control["applicable"]:
        ids = firsts["f16"]
        text = texts["f16"]
        tok = engines["f16"].get_tokenizer()
        first_text = tok.decode(ids[:1])
        control.update(
            first_token_id=ids[0],
            first_token_text=first_text,
            expected_id=REF_FIRST_TOKEN_ID,
            expected_text=REF_FIRST_TOKEN_TEXT,
            common_prefix_chars=common_prefix_len(text, REF_TEXT_4B),
            ref_chars=len(REF_TEXT_4B),
        )
        print(f"\n{BANNER}")
        print("LOAD CONTROL, f16 vLLM against our journal of 2026-08-06")
        print(BANNER)
        print(f"  first token          id {ids[0]} {first_text!r}")
        print(f"  expected             id {REF_FIRST_TOKEN_ID} {REF_FIRST_TOKEN_TEXT!r}")
        print(f"  common prefix        {control['common_prefix_chars']} / "
              f"{control['ref_chars']} characters of the reference")
        if first_text.strip() != REF_FIRST_TOKEN_TEXT.strip():
            raise ControlFailed(
                f"the f16 arm starts on {first_text!r} instead of "
                f"{REF_FIRST_TOKEN_TEXT!r}.\n"
                "  A divergence AT THE FIRST TOKEN is not explained by an\n"
                "  accumulation order: it is a wrong model, a wrong tokenizer, or\n"
                "  non-greedy sampling. The job is FALSE (§7.1)."
            )
        if ids[0] != REF_FIRST_TOKEN_ID:
            print(f"  WARNING: text matches but id {ids[0]} ≠ {REF_FIRST_TOKEN_ID}: "
                  "different vocabulary?")
            control["id_mismatch"] = True
        if control["common_prefix_chars"] < control["ref_chars"]:
            print("  WARNING: divergence AFTER the first token, EXPECTED (two")
            print("     engines, two accumulation orders, one chain of argmax). It")
            print("     is reported and it invalidates nothing.")
    else:
        print("\n(load control: not applicable, a published reference text exists")
        print(" only at 4B)")

    # --- rates, spread, ratios ---------------------------------------------
    rates = {a: [N_TOKENS / dt for dt in elapsed[a]] for a in arms}
    stats = {a: summarize(rates[a]) for a in arms}

    print(f"\n{BANNER}")
    print("THROUGHPUTS, median and range over the kept rounds")
    print(BANNER)
    print(f"  {'arm':<12}{'med tok/s':>12}{'min':>10}{'max':>10}{'spread':>10}")
    for a in arms:
        s = stats[a]
        print(f"  {a:<12}{s['median']:>12.2f}{s['min']:>10.2f}{s['max']:>10.2f}"
              f"{s['spread_pct']:>9.2f}%")

    violations: list[str] = []
    for a in arms:
        if stats[a]["spread_pct"] > SPREAD_MAX_PCT:
            violations.append(
                f"arm {a}: inter-round spread {stats[a]['spread_pct']:.2f}% > "
                f"{SPREAD_MAX_PCT:.0f}% (§7.4)"
            )

    ratio_block: dict[str, Any] = {}
    if interleaved and "f16" in arms:
        print(f"\n{BANNER}")
        print("WITHIN-STACK RATIOS, formed ROUND BY ROUND, inside vLLM only")
        print(BANNER)
        for a in arms:
            if a == "f16":
                continue
            rb = ratios_round_by_round(rates[a], rates["f16"])
            ratio_block[f"{a}_vs_f16"] = rb
            print(f"  {a} / f16: ×{rb['median']:.3f} [{rb['min']:.3f}–{rb['max']:.3f}]"
                  "   (median of 5 ratios formed round by round)")
        print()
        print("  WARNING: DO NOT DIVIDE THESE RATIOS BY OURS.")
        print("     The in-house same-head ratio, ×1.12 at 4B (48.7/43.6) and")
        print("     ×1.30 at 8B (34.4/26.5), is a QUOTIENT OF TWO SINGLE POINTS,")
        print("     measured in candle, not in vLLM. The gap between the two stacks")
        print("     is dominated by the engine, not by the weight decoder (§4 i).")
    elif not interleaved:
        print("\n(no ratio formed: sequential mode, the arms did not coexist)")

    payload = {
        "schema": "llvq-awq-speed/1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preregistration": "proofs/preregistration-awq-vllm-2026-08-17.md",
        "image_tag": tag,
        "image_digest": digest,
        "vllm_version": getattr(vllm, "__version__", "?"),
        "accelerator": os.environ.get("ACCELERATOR", "?"),
        "job_id": os.environ.get("JOB_ID", "?"),
        "size": args.size,
        "interleaved": interleaved,
        "tokens": N_TOKENS,
        "warmups": args.warmups,
        "rounds": args.rounds,
        "prompt_token_ids": PROMPT_IDS,
        "prompt_text": PROMPT_TEXT,
        "arms": {
            a: dict(
                facts[a],
                repo=(spec["awq_repo"] if ARMS[a][0] == "awq" else spec["base_repo"]),
                revision=(spec["awq_rev"] if ARMS[a][0] == "awq" else spec["base_rev"]),
                quantization_requested=ARMS[a][1],
                elapsed_s=elapsed[a],
                tokps=rates[a],
                n_tokens_out=ntok[a],
                text=texts.get(a, ""),
                **stats[a],
            )
            for a in arms
        },
        "ratios": ratio_block,
        "control": control,
        "violations": violations,
    }
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)
        print(f"\nJSON written: {args.json}")

    if violations:
        raise Violation("\n  ".join(violations))

    print(f"\n{BANNER}")
    print("OK, every guard green")
    print(BANNER)
    return 0


def merge(args) -> int:
    """Combine per-arm JSONs from a sequential run — and label them as such."""
    parts = [json.load(open(p, encoding="utf-8")) for p in args.merge]
    arms: dict[str, Any] = {}
    for p in parts:
        arms.update(p["arms"])
    if "f16" not in arms:
        raise Refused("merge without an f16 arm: no within-stack ratio is possible")
    n = min(len(a["tokps"]) for a in arms.values())
    ratios = {}
    for name, a in arms.items():
        if name == "f16":
            continue
        ratios[f"{name}_vs_f16"] = ratios_round_by_round(
            a["tokps"][:n], arms["f16"]["tokps"][:n]
        )
    out = dict(parts[0])
    out["arms"] = arms
    out["ratios"] = ratios
    out["interleaved"] = False
    out["ratio_label"] = (
        "rounds NOT interleaved: the arms never coexisted. The ratio pairs "
        "round i of one with round i of the other and must be cited with that "
        "caveat (§2.6)"
    )
    print(BANNER)
    print("SEQUENTIAL MERGE, WARNING: ROUNDS NOT INTERLEAVED")
    print(BANNER)
    for k, v in ratios.items():
        print(f"  {k}: ×{v['median']:.3f} [{v['min']:.3f}–{v['max']:.3f}]  "
              "(arms not interleaved)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
        print(f"\nJSON written: {args.json}")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="AWQ 4-bit throughput in vLLM, fusedrun protocol, "
                    "within-stack ratios only.",
    )
    p.add_argument("--size", choices=sorted(SIZES), help="4b | 8b | 14b")
    p.add_argument("--arms", default=",".join(DEFAULT_ARM_ORDER),
                   help="subset of f16,awq_marlin,awq (fixed dispatch order)")
    p.add_argument("--one-arm", default=None,
                   help="sequential mode: measure only this arm, write its JSON")
    p.add_argument("--merge", nargs="+", default=None,
                   help="merge JSONs from sequential mode (labelled ratio)")
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--warmups", type=int, default=WARMUPS)
    p.add_argument("--gpu-util-f16", type=float, default=0.30)
    p.add_argument("--gpu-util-quant", type=float, default=0.13)
    p.add_argument("--max-model-len", type=int, default=2048,
                   help="bounds the KV budget; no effect on latency at batch 1")
    p.add_argument("--json", default=None, help="path of the machine journal")
    p.add_argument("--allow-unpinned-revision", action="store_true")
    args = p.parse_args(argv)

    try:
        if args.merge:
            return merge(args)
        if not args.size:
            raise Refused("--size is required (or --merge)")
        return run(args)
    except Refused as e:
        print(f"\nREFUSED (nothing was loaded): {e}", file=sys.stderr)
        return 2
    except ControlFailed as e:
        print(f"\nCONTROL FAILED, THE JOB IS FALSE: {e}", file=sys.stderr)
        return 3
    except Violation as e:
        print(f"\n§7 VIOLATION: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
