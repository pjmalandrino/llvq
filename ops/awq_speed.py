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
            "LLVQ_IMAGE_TAG absente. Le conteneur ne connaît pas son propre tag :\n"
            "  le lanceur doit l'exporter, sinon le journal ne dit pas ce qui a\n"
            "  été mesuré et le run n'est pas rejouable (§7.5)."
        )
    if "latest" in tag:
        raise Refused(
            f"LLVQ_IMAGE_TAG={tag!r} : `latest` est un tag mobile.\n"
            "  Épingler une version, et de préférence son digest (§2.5)."
        )
    return tag, digest


def check_repo(repo: str, revision: str, pinned: bool, allow_unpinned: bool) -> None:
    """Two ways to measure the wrong thing, both silent, both caught here."""
    if repo.endswith("-deq") or "-awq-deq" in repo:
        raise Refused(
            f"{repo} est une de NOS reconstructions f16 denses : elle ne contient\n"
            "  AUCUN quartet. La chronométrer produirait un « bras AWQ » qui est en\n"
            "  fait un bras f16 aux poids dégradés — plausible, faux, et invisible\n"
            "  dans les logs."
        )
    if not revision:
        raise Refused(f"{repo} : aucune révision. Un dépôt bouge, une mesure non.")
    if not pinned and not allow_unpinned:
        raise Refused(
            f"{repo}@{revision} : révision RELEVÉE au Hub, jamais validée par\n"
            "  `ops/awq_dequant.py check` (aucune entrée EXPECTED). Forcer avec\n"
            "  --allow-unpinned-revision, et alors le dire dans tout chiffre publié."
        )


# --- reading the engine back, defensively -----------------------------------


class _Missing:
    """Distinct from `None`, which is a *legitimate* value for `quantization`.

    Conflating "the field says no quantization" with "the field could not be
    found" is exactly how a guard reports green on a config it never read.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "ILLISIBLE"


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
        return "ILLISIBLE"
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
        raise Refused(f"bras inconnu(s) : {unknown} — connus : {list(ARMS)}")
    if "f16" not in arms:
        raise Refused(
            "le bras f16 est le témoin intra-pile : sans lui il n'y a aucun\n"
            "  rapport publiable, seulement des tok/s absolus (§3)."
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
            f"utilisations cumulées {total_util:.2f} > 0,85 : trois moteurs vLLM\n"
            "  ne tiendront pas dans un processus. Lancer un bras par processus\n"
            "  avec --one-arm <nom> --json <fichier>, puis --merge — et le rapport\n"
            "  sera étiqueté « rounds NON entrelacés » (§2.6)."
        )
    if args.one_arm:
        if args.one_arm not in arms:
            raise Refused(f"--one-arm {args.one_arm} absent de --arms")
        arms = [args.one_arm]

    try:
        import vllm
    except ImportError as e:  # pragma: no cover - only outside the image
        raise Refused(
            f"vLLM introuvable ({e}). Ce script tourne DANS l'image vllm/vllm-openai\n"
            "  épinglée, pas dans notre image ni sur la machine de dev — il n'a\n"
            "  aucune dépendance au dépôt LLVQ pour cette raison."
        ) from None

    tap = LogTap()
    tap.install()

    print(BANNER)
    print("BANC DE VITESSE AWQ — dans vLLM, son propre moteur")
    print(BANNER)
    print(f"  image                 {tag}")
    print(f"  digest                {digest or '(non fourni)'}")
    print(f"  vllm                  {getattr(vllm, '__version__', '?')}")
    print(f"  taille                {args.size}")
    print(f"  bras (ordre fixe)     {arms}")
    print(f"  prompt ids            {PROMPT_IDS}  ({len(PROMPT_IDS)} tokens, non re-tokenisés)")
    print(f"  tokens générés        {N_TOKENS} (min=max, ignore_eos)")
    print(f"  rounds                {args.warmups} jetés + {args.rounds} chronométrés")
    print(f"  entrelacement         {'OUI' if interleaved else 'NON — rounds NON entrelacés'}")
    print(f"  gpu_memory_util       {utils}")
    print(f"  max_model_len         {args.max_model_len}")
    print(BANNER)
    if not interleaved:
        print("⚠️  MODE SÉQUENTIEL : les bras ne coexistent pas. Tout rapport formé")
        print("    par --merge sera étiqueté « rounds non entrelacés » et ne se cite")
        print("    pas comme un rapport entrelacé (§2.6 du pré-enregistrement).")
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
        print(f"\n--- bras {arm} chargé en {load_s:.1f} s ---")
        print(f"  dépôt                 {spec['awq_repo'] if which == 'awq' else spec['base_repo']}")
        print(f"  révision              {spec['awq_rev'] if which == 'awq' else spec['base_rev']}")
        print(f"  quantization demandée {requested!r}")
        print(f"  quantization résolue  {f['quantization_resolved']}")
        print(f"  dtype résolu          {f['dtype']}")
        print(f"  prefix caching        {f['prefix_caching']}")
        if f["kernel_log"]:
            print("  journal vLLM sur le noyau retenu :")
            for line in f["kernel_log"]:
                print(f"    | {line}")
        else:
            print("  ⚠️ aucune ligne de log sur le routage : la méthode résolue")
            print("     ci-dessus est alors la SEULE preuve de quel noyau tourne.")

        # §7.3 — read back, never suppose. Checked on the FIRST arm, before the
        # others are built and before any generation: an unreadable config must
        # cost two minutes, not the whole job.
        if f["prefix_caching"] is True:
            raise Violation(
                f"bras {arm} : prefix caching ACTIF malgré le drapeau. Les rounds\n"
                "  2..n sauteraient le prefill et la plage deviendrait un artefact."
            )
        if not f["prefix_caching_readable"]:
            raise Violation(
                f"bras {arm} : `enable_prefix_caching` illisible dans la config du\n"
                "  moteur (chemins connus épuisés). Le §7.3 exige une RELECTURE, pas\n"
                "  une supposition : cette version de vLLM a déplacé le champ et il\n"
                "  faut ajouter son chemin à `dig()` avant de mesurer quoi que ce soit."
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
            tagr = f"round {r - args.warmups + 1}" if kept else f"jeté {r + 1}"
            print(f"  {tagr:<10} {arm:<12} {dt:7.3f} s   {n:4d} tokens   "
                  f"{n / dt:7.2f} tok/s")
            # §7.2 — a short generation is not a slow one, it is a wrong one.
            if n != N_TOKENS:
                raise Violation(
                    f"bras {arm}, {tagr} : {n} tokens rendus au lieu de {N_TOKENS}.\n"
                    "  Diviser 128 par le temps de moins de 128 tokens surestime le\n"
                    "  débit d'un facteur non borné."
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
        print("CONTRÔLE DE CHARGEMENT — f16 vLLM contre notre journal du 2026-08-06")
        print(BANNER)
        print(f"  premier token        id {ids[0]} {first_text!r}")
        print(f"  attendu              id {REF_FIRST_TOKEN_ID} {REF_FIRST_TOKEN_TEXT!r}")
        print(f"  préfixe commun       {control['common_prefix_chars']} / "
              f"{control['ref_chars']} caractères de la référence")
        if first_text.strip() != REF_FIRST_TOKEN_TEXT.strip():
            raise ControlFailed(
                f"le bras f16 démarre sur {first_text!r} au lieu de "
                f"{REF_FIRST_TOKEN_TEXT!r}.\n"
                "  Une divergence AU PREMIER TOKEN ne s'explique pas par un ordre\n"
                "  d'accumulation : c'est un mauvais modèle, un mauvais tokenizer ou\n"
                "  un échantillonnage non glouton. Le job est FAUX (§7.1)."
            )
        if ids[0] != REF_FIRST_TOKEN_ID:
            print(f"  ⚠️ texte conforme mais id {ids[0]} ≠ {REF_FIRST_TOKEN_ID} : "
                  "vocabulaire différent ?")
            control["id_mismatch"] = True
        if control["common_prefix_chars"] < control["ref_chars"]:
            print("  ⚠️ divergence APRÈS le premier token : ATTENDUE (deux moteurs,")
            print("     deux ordres d'accumulation, une chaîne d'argmax). Signalée,")
            print("     elle n'invalide rien.")
    else:
        print("\n(contrôle de chargement : non applicable — il n'existe de texte de")
        print(" référence publié qu'au 4B)")

    # --- rates, spread, ratios ---------------------------------------------
    rates = {a: [N_TOKENS / dt for dt in elapsed[a]] for a in arms}
    stats = {a: summarize(rates[a]) for a in arms}

    print(f"\n{BANNER}")
    print("DÉBITS — médiane et plage sur les rounds gardés")
    print(BANNER)
    print(f"  {'bras':<12}{'méd tok/s':>12}{'min':>10}{'max':>10}{'étendue':>10}")
    for a in arms:
        s = stats[a]
        print(f"  {a:<12}{s['median']:>12.2f}{s['min']:>10.2f}{s['max']:>10.2f}"
              f"{s['spread_pct']:>9.2f}%")

    violations: list[str] = []
    for a in arms:
        if stats[a]["spread_pct"] > SPREAD_MAX_PCT:
            violations.append(
                f"bras {a} : dispersion inter-round {stats[a]['spread_pct']:.2f} % > "
                f"{SPREAD_MAX_PCT:.0f} % (§7.4)"
            )

    ratio_block: dict[str, Any] = {}
    if interleaved and "f16" in arms:
        print(f"\n{BANNER}")
        print("RAPPORTS INTRA-PILE — formés ROUND PAR ROUND, dans vLLM uniquement")
        print(BANNER)
        for a in arms:
            if a == "f16":
                continue
            rb = ratios_round_by_round(rates[a], rates["f16"])
            ratio_block[f"{a}_vs_f16"] = rb
            print(f"  {a} / f16 : ×{rb['median']:.3f} [{rb['min']:.3f}–{rb['max']:.3f}]"
                  "   (médiane de 5 rapports formés round par round)")
        print()
        print("  🚨 CES RAPPORTS NE SE COMPARENT PAS PAR DIVISION AUX NÔTRES.")
        print("     Le rapport maison à tête identique — ×1,12 au 4B (48,7/43,6),")
        print("     ×1,30 au 8B (34,4/26,5) — est un QUOTIENT DE DEUX POINTS UNIQUES,")
        print("     mesuré dans candle, pas dans vLLM. L'écart entre les deux piles")
        print("     est dominé par le moteur, pas par le décodeur de poids (§4 i).")
    elif not interleaved:
        print("\n(aucun rapport formé : mode séquentiel, les bras n'ont pas coexisté)")

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
        print(f"\nJSON écrit : {args.json}")

    if violations:
        raise Violation("\n  ".join(violations))

    print(f"\n{BANNER}")
    print("OK — tous les gardes verts")
    print(BANNER)
    return 0


def merge(args) -> int:
    """Combine per-arm JSONs from a sequential run — and label them as such."""
    parts = [json.load(open(p, encoding="utf-8")) for p in args.merge]
    arms: dict[str, Any] = {}
    for p in parts:
        arms.update(p["arms"])
    if "f16" not in arms:
        raise Refused("fusion sans bras f16 : aucun rapport intra-pile possible")
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
        "rounds NON entrelacés — les bras n'ont jamais coexisté ; le rapport "
        "apparie le round i de l'un au round i de l'autre et doit être cité "
        "avec cette réserve (§2.6)"
    )
    print(BANNER)
    print("FUSION SÉQUENTIELLE — ⚠️ ROUNDS NON ENTRELACÉS")
    print(BANNER)
    for k, v in ratios.items():
        print(f"  {k} : ×{v['median']:.3f} [{v['min']:.3f}–{v['max']:.3f}]  "
              "(bras non entrelacés)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
        print(f"\nJSON écrit : {args.json}")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Débit AWQ 4 bits dans vLLM, protocole fusedrun, "
                    "rapports intra-pile uniquement.",
    )
    p.add_argument("--size", choices=sorted(SIZES), help="4b | 8b | 14b")
    p.add_argument("--arms", default=",".join(DEFAULT_ARM_ORDER),
                   help="sous-ensemble de f16,awq_marlin,awq (ordre de dispatch fixe)")
    p.add_argument("--one-arm", default=None,
                   help="mode séquentiel : ne mesurer que ce bras, écrire son JSON")
    p.add_argument("--merge", nargs="+", default=None,
                   help="fusionner des JSON de mode séquentiel (rapport étiqueté)")
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--warmups", type=int, default=WARMUPS)
    p.add_argument("--gpu-util-f16", type=float, default=0.30)
    p.add_argument("--gpu-util-quant", type=float, default=0.13)
    p.add_argument("--max-model-len", type=int, default=2048,
                   help="borne le budget KV ; sans effet sur la latence à batch 1")
    p.add_argument("--json", default=None, help="chemin du journal machine")
    p.add_argument("--allow-unpinned-revision", action="store_true")
    args = p.parse_args(argv)

    try:
        if args.merge:
            return merge(args)
        if not args.size:
            raise Refused("--size est obligatoire (ou --merge)")
        return run(args)
    except Refused as e:
        print(f"\nREFUS (rien n'a été chargé) : {e}", file=sys.stderr)
        return 2
    except ControlFailed as e:
        print(f"\nCONTRÔLE ÉCHOUÉ — LE JOB EST FAUX : {e}", file=sys.stderr)
        return 3
    except Violation as e:
        print(f"\nVIOLATION DU §7 : {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
