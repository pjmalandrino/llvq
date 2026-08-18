# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Resident-weight floor of every arm, by exact arithmetic on shapes and dtypes.

Of the three memory lines the campaign publishes (`docs/archive/plan-de-test-v2-cuda.md`
§3.3), only one is a property of the **format** rather than of the engine: what
the weights weigh once resident. It needs no GPU, no job, no download — a
`config.json` and integer arithmetic are enough. The other two lines (measured
peak and mean VRAM) belong to the engine and are deliberately not computed here.

## The headline this file exists to make visible

`llvq-llm/src/sealed.rs` decodes every matrix to a **dense** tensor at the
runtime dtype (`decode_matrix` → `Tensor::from_vec(Vec<f32>)` → `to_dtype`), and
the carried tensors follow the same path. So the 2-bit artifact — 2,1595
bits/weight on disk — holds **exactly the FP16 checkpoint's bytes** in memory:
8 044 936 192 o on Qwen3-4B. Today the file saves ×4,54 of disk and **zero**
bytes of memory. The gap between those two numbers is, to the byte, the kernel
that is not written.

## The rule this file obeys

**Sum the shapes that are actually produced; never apply the paper's formula.**
The project has already paid once for the other way round: a rate announced at
2,0653 bits/weight was really 2,7289, because "zero gain bits" was taken as a
property of the method instead of counted in the bytes (CLAUDE.md §G5). Every
rate below is rebuilt term by term from the enumerated tensors; `selftest` pins
the result to integers read off the published file, and `verify-shapes`
confronts the enumeration with the real safetensors headers on the Hub.

Two consequences of that rule are already visible in the output, and neither is
cosmetic:

* the `g_idx` term of the GPTQ w2g128 point is quoted in the plan (§2.3) on the
  **model-weight** denominator (4 022 468 096) while the rest of its rate is on
  the **projection** denominator (3 633 315 840). Rebuilt homogeneously the
  point sits at 2,1491 b/poids, not 2,1483 — the pre-registered margin against
  our 2,159506 moves from 0,52 % to 0,49 %, so the conclusion holds, but the
  abscissa of the main figure should carry the homogeneous number;
* the same §2.3 spells the decomposition `2 + 20/128 + g_idx` while its own
  total requires `18/128` (scales f16 + zeros packed at `bits`). The prose is
  wrong, the total is right;
* the 70B recap of `docs/fiche-4b.md` §5.6 mixes the two metrics **inside one
  table row**: its Slot32 (51,35 Go) is the broad metric of §6.7, its Grouped32
  (32,87 Go) is the narrow one — the broad metric gives 34,14. Everything here
  is broad, and `selftest` pins all three layouts to the Go figures §6.7
  publishes so the choice cannot drift back.

## What holds it up

`selftest` is the lock, and it is meant to be lethal: 33 of 34 hand-written
mutants die on it (the survivor is documented in `group_terms`, and it is an
equivalence on these widths, not a hole). Two habits earn most of that. First,
no assertion is taken on a **neutral** value — `ratio_vs_dense` is pinned on
Slot32 and on the sealed file, never on `a1_loaded` alone, where the ratio is
1,00 and `1/x = x` would swallow an inverted formula (CLAUDE.md §5, the λ = 0
motif). Second, the arithmetic is confronted with **four widths**, three of
them offline: Qwen3-8B, Qwen3-32B and Llama-3.1-70B each land on an integer a
real run reported. One model cannot tell a formula from a coincidence.

## What it establishes / what it does not

Establishes: what each format weighs in **weights**, and what a KV cache adds
per token. Does **not** establish: peak or mean memory of any engine (allocator,
activations and workspaces dominate at ctx 4096), nor that any format makes a
model fit that did not.

No dependency is declared, so `uv run ops/floor.py …` and plain
`python3 ops/floor.py …` are the same run.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co"


class FloorError(Exception):
    """A refusal. Printed in French, exits non-zero, never a silent fallback."""


# --- our format -------------------------------------------------------------
#
# The published Qwen3-4B artifact, `leech1c12`: angular search capped to the
# Λ₂₄(12) ball, so 47 index bits + 1 gain bit = 48 bits per block of 24 weights
# = 2,000000 b/poids of lattice code exactly. Everything above that is
# serialization: the `KeepExact` tail in f32 and the row scales in f64 (of the
# 1 105 920 row scales, **none** is representable in f32 — that is the price of
# the bit-for-bit decode proof, not an oversight). Sources: `docs/fiche-4b.md`
# §2.1 and §3.3, read off the file itself.
LLVQ_BLOCK = 24
LLVQ_INDEX_BITS = 47
LLVQ_GAIN_BITS = 1
LLVQ_GAIN_LEVELS = 2  # centroids per matrix, f64
LLVQ_CENTROID_BITS = 64
LLVQ_ROW_SCALE_BITS = 64  # f64, one per output row
LLVQ_TAIL_BITS = 32  # f32, `TailPolicy::KeepExact`

# Kernel layouts, **Metal only and not wired into `bin/run`** (fiche-4b §6.7).
# Broad metric: payload + addressing + f32 tail + row scales, over the
# projection weights — the same convention `bin/thesis` uses for its traffic.
# Slot32 is MESURE on the whole 4B; Flat32 and Grouped32 are MESURE on
# gate_proj and the 150 681 600 real blocks respectively. None of the three is
# derivable from shapes, so extrapolating them to another model is an
# assumption, and the report says so out loud.
RUNTIME_LAYOUTS = {
    "Slot32": 5.51,
    "Flat32": 4.6779,
    "Grouped32": 3.4982,
}

# --- the reference object ---------------------------------------------------
#
# Integers read off `/Users/pjmalandrino/qwen3-4b-llvq.bin` and off the Hub, not
# recomputed here. They are the target of `selftest`: the arithmetic below has
# to land on them exactly, from `config.json` alone. They attach **only** to the
# model whose enumerated counts identify it (see `is_reference_4b`), so a 70B
# run never inherits a 4B measurement.
QWEN3_4B = dict(
    quantized=3_633_315_840,  # projection weights, tail included
    carried=389_152_256,  # tied embedding 388 956 160 + 196 096 norms
    total=4_022_468_096,
    f16_floor=8_044_936_192,
    blocks=150_681_600,  # blocks of 24
    tail=16_957_440,  # KeepExact columns
    rows=1_105_920,  # output rows = row scales
    matrices=252,
    payload_bytes=980_770_752,
    artifact_bytes=1_770_527_533,  # the whole sealed .bin, sha 9db213ef…
    # matrices framing 19 450 + raw framing 9 386 + blobs 11 423 433
    artifact_overhead=11_452_269,
    slot32_proj_bytes=2_502_446_285,  # LLVQ traffic measured by `bin/thesis`
    awq_bytes=2_666_027_672,  # Qwen/Qwen3-4B-AWQ, model.safetensors
    awq_header=102_040,  # safetensors header of that file
    kv_bytes_per_token=147_456,
)

DTYPE_BITS = {"f16": 16, "bf16": 16, "f32": 32}

# --- three more widths, so the arithmetic is not tuned to one model ----------
#
# `docs/archive/plan-de-test-v2-cuda.md` §5 (ticket FL) asks for 4B/8B/32B/70B in test,
# and it is the only way to tell a formula from a coincidence: on Qwen3-4B
# `d_out·(d_in//g)` and `d_in·(d_out//g)` give the same number, `hidden/heads`
# happens to be twice `head_dim`, and every tail is non-zero. These three
# configs break each of those ties, and they are **inline and offline** so a
# `selftest` needs exactly one network call.
#
# They are not taken on faith: each one is pinned to an integer somebody
# measured, and the enumeration has to land on it to ten digits.
#
#   Qwen3-8B   projections   6 945 767 424  — CLAUDE.md, `verify_artifact` du run
#   Qwen3-32B  4 blocs       1 950 351 360  — CLAUDE.md, dé-risquage du 2026-08-03
#   Llama-3.1-70B  total    70 553 706 496  — fiche-4b §5.6, « 70,554 Md »
#   o/token f16      147 456 · 262 144 · 327 680  — plan §3.3, « Repères »
REFERENCE_WIDTHS = {
    "Qwen3-8B": dict(
        cfg=dict(
            model_type="qwen3", architectures=["Qwen3ForCausalLM"],
            num_hidden_layers=36, hidden_size=4096, intermediate_size=12288,
            num_attention_heads=32, num_key_value_heads=8, head_dim=128,
            vocab_size=151936, tie_word_embeddings=False,
        ),
        projection_weights=6_945_767_424,
        total_weights=8_190_735_360,
        # `tie_word_embeddings` false with hidden 4096: two full embeddings,
        # 15,2 % of the weights. That share is the whole reason CLAUDE.md
        # forbids publishing the 8B compression ratio.
        carried_share=0.152,
        kv_bytes_per_token=147_456,
        # 12288 = 24 × 512 exactly, so `down_proj` — a third of the weights —
        # has **no tail**. The 4B has one on every width; this is the config
        # that proves `d_in % 24` is read and not assumed.
        tailless_width=12288,
    ),
    "Qwen3-32B": dict(
        cfg=dict(
            model_type="qwen3", architectures=["Qwen3ForCausalLM"],
            num_hidden_layers=64, hidden_size=5120, intermediate_size=25600,
            num_attention_heads=64, num_key_value_heads=8, head_dim=128,
            vocab_size=151936, tie_word_embeddings=False,
        ),
        projection_weights=31_205_621_760,
        total_weights=32_762_123_264,
        carried_share=0.0475,
        kv_bytes_per_token=262_144,
        # 4 blocks out of 64 = the de-risking run, bit-for-bit verified.
        four_block_projections=1_950_351_360,
    ),
    "Llama-3.1-70B": dict(
        cfg=dict(
            model_type="llama", architectures=["LlamaForCausalLM"],
            num_hidden_layers=80, hidden_size=8192, intermediate_size=28672,
            num_attention_heads=64, num_key_value_heads=8, head_dim=128,
            vocab_size=128256, tie_word_embeddings=False,
        ),
        projection_weights=68_451_041_280,
        total_weights=70_553_706_496,
        carried_share=0.0298,
        kv_bytes_per_token=327_680,
        # No `q_norm`/`k_norm`: `has_qk_norm` must answer False here, and the
        # total is 10 digits wide, so a flag stuck at True cannot hide.
        qk_norm=False,
    ),
}

# The abscissa `docs/archive/plan-de-test-v2-cuda.md` §2.3 pre-registers for the GPTQ
# w2g128 point. Reproduced — not adopted — by `selftest`: see the module
# docstring for which denominator it mixes.
PLAN_GPTQ_W2G128 = 2.148255


# --- config -----------------------------------------------------------------


def load_config(ref: str, revision: str = "main") -> dict:
    """A `config.json`, from a local path or from a Hub repo id.

    Read-only and token-free, same shape as `run.py:fetch_config` — a path wins
    over a repo id when both would resolve, because a file on disk is the more
    specific intent.
    """
    p = Path(ref)
    if p.is_dir():
        p = p / "config.json"
    if p.is_file():
        return json.loads(p.read_text())
    if "/" not in ref:
        raise FloorError(f"ni fichier ni repo id : {ref}")
    url = f"{HUB}/{ref}/raw/{revision}/config.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310
            return json.load(r)
    except urllib.error.URLError as e:
        raise FloorError(f"lecture impossible de {url} : {e}") from e


def req(cfg: dict, key: str) -> int:
    """A `config.json` field this file cannot invent.

    A `KeyError` traceback out of a measurement script is worse than useless:
    it looks like a bug in the reader rather than a config the enumeration does
    not cover. Everything else in this file refuses in French and exits 2.
    """
    if key not in cfg:
        raise FloorError(
            f"`{key}` absent du config.json : l'énumération ne peut pas décrire "
            "ce modèle, et un plancher inventé serait pire que pas de plancher."
        )
    return int(cfg[key])


def head_dim(cfg: dict) -> int:
    """`head_dim`, explicit or derived.

    Qwen3 declares it (128 with hidden 2560, so it is **not** hidden/heads and
    deriving it blindly would be wrong). Older Llama configs omit it, where the
    derivation is the definition.
    """
    hd = cfg.get("head_dim")
    if hd:
        return int(hd)
    return req(cfg, "hidden_size") // req(cfg, "num_attention_heads")


def has_qk_norm(cfg: dict) -> bool:
    """Whether the family carries per-head `q_norm`/`k_norm` (Qwen3-specific).

    They are 128 weights each, 0,005 % of the model — but `selftest` demands an
    exact landing on 4 022 468 096, so a wrong flag fails loudly instead of
    hiding in a rounding.
    """
    mt = str(cfg.get("model_type", ""))
    archs = " ".join(cfg.get("architectures") or [])
    return mt.startswith("qwen3") or "Qwen3" in archs


def enumerate_tensors(cfg: dict) -> tuple[list, list]:
    """Every weight tensor the loader materializes: `(projections, carried)`.

    Each entry is `(name, shape)`. The names are the ones
    `llvq_llm::artifact::key()` builds and the ones a safetensors index carries,
    which is what lets `verify-shapes` confront this list with the real file
    rather than trust it.
    """
    mt = str(cfg.get("model_type", ""))
    moe = ("num_experts", "num_local_experts", "n_routed_experts")
    moe_keys = [k for k in moe if k in cfg]
    if mt.endswith("_moe") or moe_keys:
        raise FloorError(
            "modèle à experts (MoE) : l'énumération dense ci-dessous serait "
            f"fausse ({mt}, {moe_keys}). Refus plutôt qu'un plancher inventé."
        )

    layers = req(cfg, "num_hidden_layers")
    hidden = req(cfg, "hidden_size")
    inter = req(cfg, "intermediate_size")
    hd = head_dim(cfg)
    n_heads = req(cfg, "num_attention_heads")
    # Absent means MHA — that is the definition of GQA degenerating, not a
    # guess. Pre-GQA Llama-2 configs omit it, and `TheBloke/Llama-2-7B-GPTQ`
    # is exactly the kind of repo `verify-shapes` is pointed at.
    n_kv = int(cfg.get("num_key_value_heads") or n_heads)
    attn_out = hd * n_heads
    qk_norm = has_qk_norm(cfg)
    attn_bias = bool(cfg.get("attention_bias", False))
    mlp_bias = bool(cfg.get("mlp_bias", False))

    proj, carried = [], []
    for b in range(layers):
        pre = f"model.layers.{b}"
        proj += [
            (f"{pre}.self_attn.q_proj.weight", (attn_out, hidden)),
            (f"{pre}.self_attn.k_proj.weight", (n_kv * hd, hidden)),
            (f"{pre}.self_attn.v_proj.weight", (n_kv * hd, hidden)),
            (f"{pre}.self_attn.o_proj.weight", (hidden, attn_out)),
            (f"{pre}.mlp.gate_proj.weight", (inter, hidden)),
            (f"{pre}.mlp.up_proj.weight", (inter, hidden)),
            (f"{pre}.mlp.down_proj.weight", (hidden, inter)),
        ]
        carried += [
            (f"{pre}.input_layernorm.weight", (hidden,)),
            (f"{pre}.post_attention_layernorm.weight", (hidden,)),
        ]
        if qk_norm:
            carried += [
                (f"{pre}.self_attn.q_norm.weight", (hd,)),
                (f"{pre}.self_attn.k_norm.weight", (hd,)),
            ]
        # Declared biases only. A family that hard-codes a qkv bias without
        # saying so in `config.json` (Qwen2 does) is not covered here — that is
        # exactly what `verify-shapes` catches, and why it exists.
        if attn_bias:
            carried += [
                (f"{pre}.self_attn.q_proj.bias", (attn_out,)),
                (f"{pre}.self_attn.k_proj.bias", (n_kv * hd,)),
                (f"{pre}.self_attn.v_proj.bias", (n_kv * hd,)),
            ]
        if mlp_bias:
            carried += [
                (f"{pre}.mlp.gate_proj.bias", (inter,)),
                (f"{pre}.mlp.up_proj.bias", (inter,)),
                (f"{pre}.mlp.down_proj.bias", (hidden,)),
            ]

    vocab = int(cfg["vocab_size"])
    carried.append(("model.embed_tokens.weight", (vocab, hidden)))
    carried.append(("model.norm.weight", (hidden,)))
    if not cfg.get("tie_word_embeddings", False):
        # Untied is the trap: two full embedding tensors at 16 bits. On
        # Qwen3-32B that is 4,7 % of the weights but 27 % of the artifact.
        carried.append(("lm_head.weight", (vocab, hidden)))
    return proj, carried


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def counts(cfg: dict) -> dict:
    """Weight counts and the block accounting our format pays, from shapes only.

    `blocks`, `tail` and `rows` are what the encoder actually produces:
    `floor(d_in/24)` blocks per output row, the remaining `d_in mod 24` columns
    left exact, one row scale per output row. On Qwen3-4B they land on
    150 681 600 / 16 957 440 / 1 105 920 — the three integers the file reports
    and the bench prints.
    """
    proj, carried = enumerate_tensors(cfg)
    blocks = tail = rows = quantized = 0
    for _, (d_out, d_in) in proj:
        blocks += d_out * (d_in // LLVQ_BLOCK)
        tail += d_out * (d_in % LLVQ_BLOCK)
        rows += d_out
        quantized += d_out * d_in
    carried_n = sum(numel(s) for _, s in carried)
    return dict(
        matrices=len(proj),
        projection_weights=quantized,
        carried_weights=carried_n,
        total_weights=quantized + carried_n,
        blocks=blocks,
        block_weights=blocks * LLVQ_BLOCK,
        tail_weights=tail,
        rows=rows,
    )


def is_reference_4b(c: dict) -> bool:
    """Whether this is the very object the measured constants were taken on."""
    return (
        c["projection_weights"] == QWEN3_4B["quantized"]
        and c["total_weights"] == QWEN3_4B["total"]
    )


# --- rates ------------------------------------------------------------------


def llvq_payload_bits(c: dict) -> int:
    """Bits our artifact writes for the projections — the sum, not a formula."""
    return (
        c["blocks"] * (LLVQ_INDEX_BITS + LLVQ_GAIN_BITS)
        + c["rows"] * LLVQ_ROW_SCALE_BITS
        + c["matrices"] * LLVQ_GAIN_LEVELS * LLVQ_CENTROID_BITS
        + c["tail_weights"] * LLVQ_TAIL_BITS
    )


def group_terms(cfg: dict, group: int) -> dict:
    """Per-group bookkeeping shared by AWQ and GPTQ, checked against the shapes.

    `groups` is exact only if `group` divides every `d_in`. On Qwen3-4B it does
    (2560, 4096 and 9728 are all multiples of 128), which is the asymmetry to
    publish: the adversary has **no tail**, we pay 16 957 440 exact weights
    because 24 divides none of those widths.

    > 🕳️ Mutating this to `d_in * (d_out // group)` survives `selftest`, and it
    > is not a weak test: on all four reference widths `group` divides `d_out`
    > too, so the two spellings are the same integer. The grouping is along
    > `d_in` (`scales` is `[d_in/gs, d_out]`) — the equivalence is a property
    > of these models, not of the formula. Same category as the dead suffix
    > sweep of CLAUDE.md §4: a surviving mutant that says « equivalent here »,
    > not « untested ».
    """
    proj, _ = enumerate_tensors(cfg)
    bad = sorted({d_in for _, (_, d_in) in proj if d_in % group})
    groups = sum(d_out * (d_in // group) for _, (d_out, d_in) in proj)
    return dict(groups=groups, unaligned_widths=bad)


def gptq_gidx_bits(cfg: dict) -> int:
    """`g_idx`: one int32 per input channel, per matrix."""
    proj, _ = enumerate_tensors(cfg)
    return sum(d_in * 32 for _, (_, d_in) in proj)


def bits_to_bytes(bits: int) -> int:
    """Bits → bytes, rounding **up**.

    A file cannot hold a fraction of a byte, and `// 8` would quietly hand a
    format up to 7 bits it never paid for — the same species of error as the
    2,0653 that was really 2,7289. Every rate here is byte-aligned on Qwen3-4B
    and `selftest` requires it to stay so, which makes the ceiling a guard
    against a future width rather than a correction to a current number.
    """
    return -(-bits // 8)


def declared_rate(cfg: dict, c: dict, has_gidx: bool = False) -> dict | None:
    """The rate a repo's own `quantization_config` implies, or `None`.

    `bits` for the packed weights, one f16 scale and one `bits`-wide zero per
    (group, output) — so `bits + (16 + bits)/group` for both AWQ GEMM and GPTQ,
    plus one int32 `g_idx` per input channel **when the repo actually ships
    one** (`has_gidx`, read off the header, not guessed from the method name:
    `desc_act=False` does not imply the tensor is absent). Built from the
    repo's declaration so that `verify-shapes` can confront it with the bytes
    the same repo ships, which is the check `docs/archive/plan-de-test-v2-cuda.md` §3.1
    makes obligatory.

    Forgetting the `g_idx` term is not a rounding: on Qwen3-4B it is
    30 670 848 bits, and the confrontation would print « l'un des deux ment »
    about a tensor this script simply failed to count.
    """
    qc = cfg.get("quantization_config")
    if not qc or "bits" not in qc:
        return None
    bits = int(qc["bits"])
    group = int(qc.get("group_size") or 0)
    method = str(qc.get("quant_method", "?")).lower()
    if group <= 0:
        return None
    g = group_terms(cfg, group)
    total = c["projection_weights"] * bits + g["groups"] * (16 + bits)
    terms = f"{bits} + {16 + bits}/{group}"
    if has_gidx:
        total += gptq_gidx_bits(cfg)
        terms += " + 32/d_out (g_idx)"
    return dict(
        method=method,
        bits=bits,
        group=group,
        bits_total=total,
        rate=total / c["projection_weights"],
        terms=terms,
        unaligned=g["unaligned_widths"],
        has_gidx=has_gidx,
    )


def adversary_rates(cfg: dict, c: dict) -> dict:
    """Bits/weight on the projections, rebuilt term by term for each adversary.

    AWQ GEMM w4g128 (`Qwen/Qwen3-4B-AWQ`): `qweight` 4 b/w, `scales` f16 per
    (group, out), `qzeros` packed 4 bits per (group, out) → 4 + 20/128 =
    4,15625. Independent of the shapes as long as 128 divides `d_in`, and the
    file's own bytes confirm it to the header (see `selftest`).

    GPTQ w2g128 sym: `qweight` 2 b/w, `scales` f16 and `qzeros` packed at
    `bits`, so 2 + 18/128, plus `g_idx` int32 per input channel. The plan's
    `2 + 20/128` is a slip of the pen; its own total requires 18/128.

    bitsandbytes NF4, blocksize 64, no double quant: 4 b/w plus one f32 absmax
    per 64 elements → 4 + 32/64 = 4,5 exactly, whatever the shapes.
    """
    w = c["projection_weights"]
    g128 = group_terms(cfg, 128)
    gidx = gptq_gidx_bits(cfg)

    awq_bits = w * 4 + g128["groups"] * (16 + 4)
    gptq_bits = w * 2 + g128["groups"] * (16 + 2) + gidx
    nf4_bits = w * 4 + (w // 64) * 32

    return dict(
        awq_w4g128=dict(
            bits=awq_bits,
            rate=awq_bits / w,
            unaligned=g128["unaligned_widths"],
            terms="4 + 16/128 (scales f16) + 4/128 (zeros)",
        ),
        gptq_w2g128=dict(
            bits=gptq_bits,
            rate=gptq_bits / w,
            unaligned=g128["unaligned_widths"],
            terms="2 + 16/128 (scales f16) + 2/128 (zeros) + 32/d_out (g_idx)",
            gidx_bits=gidx,
            # The same rate with **only** the g_idx term over the model-weight
            # denominator. Not a rate anybody should publish — it is here to
            # reproduce the plan's 2,148255 on Qwen3-4B and so name the
            # discrepancy instead of silently correcting it.
            rate_gidx_on_model_weights=2 + 18 / 128 + gidx / c["total_weights"],
        ),
        bnb_nf4_bs64=dict(
            bits=nf4_bits,
            rate=nf4_bits / w,
            unaligned=[] if w % 64 == 0 else ["numel % 64"],
            terms="4 + 32/64 (absmax f32, sans double quant)",
        ),
    )


# --- the floor --------------------------------------------------------------


def arms(cfg: dict, c: dict, dtype: str = "f16") -> list[dict]:
    """One row per arm and per layout: resident bytes, b/poids, ratio vs f16.

    The carried half is 16 bits in **every** arm — no CUDA quantizer touches
    `nn.Embedding` or the `lm_head`, and neither do we. It is therefore a
    constant added to every row, and it dilutes every ratio. That is why it gets
    its own column instead of being folded in.
    """
    bits = DTYPE_BITS[dtype]
    proj_n = c["projection_weights"]
    carried_bytes = c["carried_weights"] * bits // 8
    dense_proj = proj_n * bits // 8
    ref = dense_proj + carried_bytes
    rates = adversary_rates(cfg, c)
    payload = llvq_payload_bits(c)

    def row(key, short, label, proj_bytes, status, note="", resident=True,
            total=None):
        # `total` is an override for the one row whose size is measured rather
        # than summed (the sealed file, framing and blobs included). It exists
        # so that row stays on the same three formulas as the others: a second
        # hand-written dict is a second place for the ratio to be inverted.
        total = proj_bytes + carried_bytes if total is None else total
        return dict(
            key=key,
            short=short,
            label=label,
            resident=resident,
            projection_bytes=proj_bytes,
            carried_bytes=carried_bytes,
            resident_bytes=total,
            projection_bits_per_weight=proj_bytes * 8 / proj_n,
            model_bits_per_weight=total * 8 / c["total_weights"],
            ratio_vs_dense=ref / total,
            status=status,
            note=note,
        )

    wide = "métrique large : charge utile + adressage + queue f32 + échelles"
    out = [
        row("c_dense", "C", f"C — checkpoint {dtype} dense", dense_proj,
            "CALCULE exact"),
        row(
            "a1_loaded",
            "A1 chargé",
            f"A1 — LLVQ 2 bits, TEL QUE CHARGÉ ({dtype} dense)",
            dense_proj,
            "CALCULE exact",
            "sealed.rs décode en tenseurs denses : identique au checkpoint",
        ),
    ]
    for name, rate in RUNTIME_LAYOUTS.items():
        out.append(
            row(
                f"a1_{name.lower()}",
                f"A1·{name}",
                f"A1 — si {name} (noyau Metal, NON branché)",
                round(proj_n * rate / 8),
                f"MESURE {gf(rate, 4)} b/poids"
                + ("" if is_reference_4b(c) else " sur le 4B — EXTRAPOLÉ ICI"),
                wide,
            )
        )
    for key, short, label in (
        ("awq_w4g128", "B0·AWQ", "B0 — AWQ w4g128, dans son moteur"),
        ("gptq_w2g128", "B1·GPTQ w2", "B1 — GPTQ w2g128 sym, dans son moteur"),
        ("bnb_nf4_bs64", "B3·NF4", "B3 — bitsandbytes NF4 bs64, dans son moteur"),
    ):
        r = rates[key]
        out.append(
            row(key, short, label, bits_to_bytes(r["bits"]), "CALCULE exact", r["terms"])
        )

    # Disk is not residency, and mixing the two is how "×4,54 de compression"
    # becomes "×4,54 de mémoire". Listed after a rule, flagged `resident=False`,
    # and never fed to the dilution arithmetic.
    out.append(
        row("a1_disk", "A1 disque", "A1 — sur DISQUE (pas résident)",
            bits_to_bytes(payload),
            "CALCULE exact", "charge utile + portés ; hors framing et blobs",
            resident=False)
    )
    if is_reference_4b(c) and dtype in ("f16", "bf16"):
        # The published file, so that the computed disk line above can never be
        # mistaken for it: the difference is 11 452 269 o of framing and blobs.
        out.append(
            row("a1_file", "A1 fichier", "A1 — fichier scellé publié (MESURE)",
                bits_to_bytes(payload), "MESURE (sha 9db213ef…)",
                "framing + config.json + tokenizer.json compris",
                resident=False, total=QWEN3_4B["artifact_bytes"])
        )
    return out


# --- KV cache ---------------------------------------------------------------


def kv_bytes_per_token(cfg: dict, dtype: str = "f16") -> int:
    """`2 · n_layers · n_kv_heads · head_dim · sizeof(dtype)`.

    The 2 is K and V, once each — counting it twice is the ~640 Ko/token of
    `docs/archive/pistes-battre-q4.md`, wrong in every reading. Qwen3-4B lands on
    147 456 o/token; 320 Kio/token is the correct Llama-3.1-70B figure quoted
    outside its model.
    """
    return (
        2
        * int(cfg["num_hidden_layers"])
        * int(cfg["num_key_value_heads"])
        * head_dim(cfg)
        * (DTYPE_BITS[dtype] // 8)
    )


def dilution(rows: list[dict], per_token: int, contexts: list[int],
             a_key: str, b_key: str, activations_bytes: int) -> dict:
    """How a format advantage shrinks as the context grows.

    The KV cache is **additive and common to all formats**, so any ratio between
    two formats converges to 1 as the context grows. This is deployment
    arithmetic on a hypothesis (a flat activation term), not a measurement, and
    it must never be printed in the same table as measured bytes.
    """
    by = {r["key"]: r for r in rows}
    for k in (a_key, b_key):
        if k not in by:
            raise FloorError(f"bras inconnu : {k} (connus : {', '.join(by)})")
        if not by[k]["resident"]:
            raise FloorError(
                f"{k} est une taille de FICHIER, pas un plancher résident : "
                "la diluer par un cache KV n'aurait aucun sens"
            )
    a, b = by[a_key], by[b_key]
    lines = []
    for ctx in contexts:
        kv = per_token * ctx
        add = kv + (activations_bytes if ctx else 0)
        ta, tb = a["resident_bytes"] + add, b["resident_bytes"] + add
        big, small = (ta, tb) if ta >= tb else (tb, ta)
        lines.append(
            dict(
                context=ctx,
                kv_bytes=kv,
                a_bytes=ta,
                b_bytes=tb,
                ratio=big / small,
                # The ratio is always ≥ 1; naming who it favours is what stops
                # it from being read backwards, which is how the plan's own
                # sentence ended up saying "l'avantage de Slot32 sur AWQ" about
                # a number that is AWQ's.
                favours=b["short"] if ta > tb else a["short"],
            )
        )
    return dict(
        a=a["label"],
        b=b["label"],
        a_short=a["short"],
        b_short=b["short"],
        activations_bytes=activations_bytes,
        lines=lines,
    )


# --- report -----------------------------------------------------------------


def report(ref: str, cfg: dict, dtype: str, contexts: list[int],
           pair: tuple[str, str], activations_bytes: int) -> dict:
    c = counts(cfg)
    rows = arms(cfg, c, dtype)
    per_token = kv_bytes_per_token(cfg, dtype)
    payload = llvq_payload_bits(c)
    rep = dict(
        model=ref,
        model_type=cfg.get("model_type"),
        dtype=dtype,
        is_reference_4b=is_reference_4b(c),
        counts=c,
        carried_share=c["carried_weights"] / c["total_weights"],
        # Two distinct objects, and conflating them is how a `--dtype f32` run
        # would print a ×-column against 16 Go while labelling the line "f16".
        # `dense_floor_bytes` is the reference every ratio here is taken
        # against; `f16_floor_bytes` is the constant the campaign publishes.
        dense_floor_bytes=c["total_weights"] * DTYPE_BITS[dtype] // 8,
        f16_floor_bytes=c["total_weights"] * 2,
        llvq_disk=dict(
            payload_bits=payload,
            payload_bytes=bits_to_bytes(payload),
            bits_per_projection_weight=payload / c["projection_weights"],
            bits_per_quantized_weight=payload / c["block_weights"],
        ),
        rates=adversary_rates(cfg, c),
        arms=rows,
        kv=dict(
            bytes_per_token=per_token,
            contexts={str(x): per_token * x for x in contexts},
        ),
        dilution=dilution(rows, per_token, contexts, pair[0], pair[1],
                          activations_bytes),
    )
    if rep["is_reference_4b"]:
        rep["measured"] = dict(QWEN3_4B)
    return rep


def gi(n: int) -> str:
    """1770527533 → '1 770 527 533'. The docs' convention, kept in the tables."""
    return f"{n:,}".replace(",", " ")


def gf(x: float, dec: int = 4) -> str:
    return f"{x:.{dec}f}".replace(".", ",")


def render(rep: dict) -> str:
    c = rep["counts"]
    bits = DTYPE_BITS[rep["dtype"]]
    out: list[str] = []
    w = out.append
    w("")
    w(f"{rep['model']} — plancher des poids résidents ({rep['dtype']})")
    w("  arithmétique exacte sur les formes ; 0 GPU, 0 $, aucun téléchargement")
    w("  contrôle externe : ops/floor.py verify-shapes <repo>")
    if not rep["is_reference_4b"]:
        w("  ⚠️  ce n'est pas Qwen3-4B : les layouts du noyau sont des MESURES "
          "faites sur le 4B, extrapolées ici")
    w("")
    w(f"  matrices de projection   {c['matrices']:>6}")
    w(f"  poids de projection      {gi(c['projection_weights']):>20}"
      f"   dont queue {gi(c['tail_weights'])}"
      f" ({gf(100 * c['tail_weights'] / c['projection_weights'], 4)} %)")
    w(f"  poids portés ({bits} bits)   {gi(c['carried_weights']):>20}"
      f"   {gf(100 * rep['carried_share'], 2)} % des poids")
    w(f"  total                    {gi(c['total_weights']):>20}")
    w(f"  blocs de {LLVQ_BLOCK}              {gi(c['blocks']):>20}"
      f"   lignes de sortie {gi(c['rows'])}")
    w(f"  plancher {rep['dtype']:<4}            {gi(rep['dense_floor_bytes']):>20} o"
      "   ← la référence de la colonne ×")
    if rep["dense_floor_bytes"] != rep["f16_floor_bytes"]:
        w(f"  plancher f16             {gi(rep['f16_floor_bytes']):>20} o"
          "   ← la référence que la campagne publie")
    w("")

    w(f"  PLANCHER DES POIDS RÉSIDENTS — les portés sont à {bits} bits dans TOUS "
      "les bras")
    w(f"  {'objet':<44}{'projection (o)':>17}{'total (o)':>17}"
      f"{'b/poids':>9}{'×' + rep['dtype']:>8}   statut")
    w("  " + "-" * 112)
    for i, r in enumerate(rep["arms"]):
        if not r["resident"] and (i == 0 or rep["arms"][i - 1]["resident"]):
            w("  " + "-" * 112 + "   ↓ TAILLE DE FICHIER, PAS DE LA MÉMOIRE")
        w(f"  {r['label']:<44}{gi(r['projection_bytes']):>17}"
          f"{gi(r['resident_bytes']):>17}"
          f"{gf(r['model_bits_per_weight'], 3):>9}"
          f"{'×' + gf(r['ratio_vs_dense'], 2):>8}   {r['status']}")
    w("")
    w("  Colonne « b/poids » = octets ÷ tous les poids du modèle. Sur les")
    w("  projections seules, les mêmes objets donnent :")
    for r in rep["arms"]:
        w(f"    {r['label']:<44}{gf(r['projection_bits_per_weight'], 4):>10} b/poids"
          f"   {r['note']}".rstrip())
    w("")
    w("  → Les poids portés valent "
      f"{gi(rep['arms'][0]['carried_bytes'])} o dans CHAQUE ligne — aucun")
    w("    quantifieur CUDA ne touche l'embedding, et nous non plus. C'est le")
    w("    terme qui dilue tout rapport de format sur le modèle entier.")
    w("")

    d = rep["llvq_disk"]
    w("  NOTRE FICHIER — charge utile, sommée sur les formes")
    w(f"    charge utile           {gi(d['payload_bytes']):>20} o "
      f"({gi(d['payload_bits'])} bits)")
    w(f"    b/poids de projection  {gf(d['bits_per_projection_weight'], 6):>20}"
      "   ← le chiffre homogène, celui de la carte HF")
    w(f"    b/poids quantifiés     {gf(d['bits_per_quantized_weight'], 6):>20}"
      "   ← variante conservatrice, dénominateur hors queue")
    w("")

    w("  CACHE KV — 2 · n_layers · n_kv_heads · head_dim · sizeof(dtype)")
    w(f"    {gi(rep['kv']['bytes_per_token'])} o/token")
    w(f"    {'contexte':>10}{'cache KV (o)':>18}{'Go':>10}")
    for ctx, b in rep["kv"]["contexts"].items():
        w(f"    {int(ctx):>10}{gi(b):>18}{gf(b / 1e9, 3):>10}")
    w("")

    dl = rep["dilution"]
    w("  DILUTION — le cache KV est ADDITIF et COMMUN aux formats")
    w(f"    A = {dl['a']}")
    w(f"    B = {dl['b']}")
    w(f"    activations supposées constantes : {gi(dl['activations_bytes'])} o "
      "(hypothèse de déploiement, PAS une mesure)")
    w(f"    {'contexte':>11}{'KV (Go)':>10}{'A (Go)':>10}{'B (Go)':>10}"
      f"{'rapport':>10}   en faveur de")
    for ln in dl["lines"]:
        label = "poids seuls" if ln["context"] == 0 else str(ln["context"])
        w(f"    {label:>11}{gf(ln['kv_bytes'] / 1e9, 3):>10}"
          f"{gf(ln['a_bytes'] / 1e9, 3):>10}{gf(ln['b_bytes'] / 1e9, 3):>10}"
          f"{'×' + gf(ln['ratio'], 4):>10}   {ln['favours']}")
    w("")
    w("  → Tout avantage de format se dilue mécaniquement quand le contexte")
    w("    grandit. La ligne « poids seuls » est de l'arithmétique exacte ; les")
    w("    autres sont de l'arithmétique de DÉPLOIEMENT HYPOTHÉTIQUE, à ne")
    w("    jamais publier dans la même table qu'un octet mesuré.")
    w("")

    g = rep["rates"]["gptq_w2g128"]
    if rep["is_reference_4b"]:
        # Only on the object the plan pre-registered. On any other width this
        # paragraph would be citing a constant that does not describe it.
        margin = 100 * (rep["llvq_disk"]["bits_per_projection_weight"] / g["rate"] - 1)
        w("  ⚠️  GPTQ w2g128 : reconstruit homogènement sur les projections, il "
          f"vaut {gf(g['rate'], 6)} b/poids.")
        w(f"      Le plan (§2.3) publie {gf(PLAN_GPTQ_W2G128, 6)}, que ce script "
          f"retrouve à {gf(g['rate_gidx_on_model_weights'], 6)}")
        w("      en rapportant le SEUL terme g_idx aux poids du MODÈLE quand le "
          "reste l'est aux")
        w("      projections. L'écart déplace la marge pré-enregistrée de "
          f"0,52 % à {gf(margin, 2)} % : la conclusion")
        w("      tient, mais l'abscisse de la figure principale doit porter le "
          "chiffre homogène.")
        w("      (Et §2.3 écrit « 2 + 20/128 » là où son propre total exige "
          "18/128.)")
        w("")
    if not rep["is_reference_4b"]:
        w("  ⚠️  Rappel : Slot32/Flat32/Grouped32 sont des mesures faites sur le "
          "4B publié, et")
        w("      rien ne garantit que la distribution des classes se transporte "
          "à cette largeur.")
        w("")
    return "\n".join(out)


# --- safetensors headers (the external control) -----------------------------


def _range(url: str, start: int, end: int) -> bytes:
    """One HTTP Range read. Refuses a 200 rather than stream gigabytes.

    A server that ignores `Range` answers 200 with the whole file — on a 4 Go
    shard that is a download nobody asked for. The body is never touched in that
    case.
    """
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            if r.status != 206:
                raise FloorError(
                    f"{url} : Range ignoré (status {r.status}). Refus de lire "
                    "le corps — ce serait le fichier entier."
                )
            return r.read(end - start + 1)
    except urllib.error.HTTPError as e:
        raise FloorError(f"{url} : HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise FloorError(f"{url} : {e}") from e


# safetensors dtype names → bytes. Deliberately a closed table: an unknown
# dtype is a refusal, not a guess, because guessing 2 where the file means 4
# would understate a floor by half.
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}


def safetensors_index(repo: str, revision: str, filename: str) -> dict:
    """`{name: (dtype, shape)}` from a shard's header, header bytes only."""
    url = f"{HUB}/{repo}/resolve/{revision}/{filename}"
    n = int.from_bytes(_range(url, 0, 7), "little")
    if not 0 < n < (64 << 20):
        raise FloorError(f"{filename} : taille d'en-tête aberrante ({n})")
    head = json.loads(_range(url, 8, 7 + n))
    return {
        k: (v["dtype"], tuple(v["shape"]))
        for k, v in head.items()
        if k != "__metadata__"
    }


def shard_list(repo: str, revision: str) -> list[str]:
    try:
        url = f"{HUB}/{repo}/raw/{revision}/model.safetensors.index.json"
        with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310
            idx = json.load(r)
        return sorted(set(idx["weight_map"].values()))
    except urllib.error.HTTPError:
        return ["model.safetensors"]


def cmd_verify_shapes(args) -> int:
    """Confront the enumeration with a repo's real safetensors headers.

    This is what turns the arithmetic above from a *model* of a checkpoint into
    a *reading* of one. Header bytes only — tens of kilobytes, no weights — and
    it is the only thing standing between a family-specific quirk (an undeclared
    qkv bias, a renamed norm) and a floor that looks right and is not.

    Two verdicts, because there are two kinds of repo in the campaign:

    * **dense** (`Qwen/Qwen3-4B`): every enumerated name must be there with its
      shape, nothing extra, and the weight count must land exactly. That is the
      guarantee the whole file rests on.
    * **packed** (`Qwen/Qwen3-4B-AWQ`): the projections are `.qweight`/`.qzeros`
      /`.scales` instead of `.weight`, so names cannot match. What is checked
      instead is stronger for our purpose — the carried tensors are identical
      (they are 16-bit in every arm), each of the 252 projections is present in
      packed form, and the **bytes the repo actually ships** are summed from the
      headers and confronted with the rate its own `quantization_config`
      declares. §3.1 of the plan makes that confrontation obligatory; this is it.
    """
    cfg = load_config(args.config or args.repo, args.revision)
    c = counts(cfg)
    proj, carried = enumerate_tensors(cfg)
    want_proj = {n: tuple(s) for n, s in proj}
    want_carried = {n: tuple(s) for n, s in carried}

    got: dict[str, tuple[str, tuple]] = {}
    for shard in shard_list(args.repo, args.revision):
        got.update(safetensors_index(args.repo, args.revision, shard))
        print(f"  en-tête lu : {shard} ({len(got)} tenseurs cumulés)")

    packed = any(k.endswith((".qweight", ".qzeros", ".absmax")) for k in got)
    ok = True

    def fail(msg: str) -> None:
        nonlocal ok
        ok = False
        print(f"  FAIL  {msg}")

    def show(label: str, names: list[str], detail=lambda n: "") -> None:
        if not names:
            return
        print(f"  {label} ({len(names)}) :")
        for n in names[:12]:
            print(f"    {n}{detail(n)}")
        if len(names) > 12:
            print(f"    … et {len(names) - 12} autres")

    print(f"\n  {'checkpoint empaqueté' if packed else 'checkpoint dense'}"
          f" · {len(got)} tenseurs")

    # The carried half is the same object in every arm, so it is checked the
    # same way in every repo — name and shape, no tolerance.
    miss_c = sorted(n for n in want_carried if n not in got)
    bad_c = sorted(n for n in want_carried if n in got and got[n][1] != want_carried[n])
    show("portés manquants", miss_c)
    show("portés de forme différente", bad_c,
         lambda n: f" énuméré {want_carried[n]} · fichier {got[n][1]}")
    if miss_c or bad_c:
        fail(f"{len(miss_c) + len(bad_c)} tenseurs portés ne correspondent pas")
    else:
        print(f"  ok    {len(want_carried)} tenseurs portés, nom et forme "
              "identiques à l'énumération")

    if packed:
        stems = {n.rsplit(".weight", 1)[0] for n in want_proj}
        absent = sorted(s for s in stems
                        if not any(k.startswith(s + ".") for k in got))
        show("projections absentes", absent)
        if absent:
            fail(f"{len(absent)} projections sur {len(stems)} introuvables")
        else:
            print(f"  ok    {len(stems)} projections présentes sous forme "
                  "empaquetée")
    else:
        miss_p = sorted(n for n in want_proj if n not in got)
        bad_p = sorted(n for n in want_proj if n in got and got[n][1] != want_proj[n])
        extra = sorted(set(got) - set(want_proj) - set(want_carried))
        show("projections manquantes", miss_p)
        show("projections de forme différente", bad_p,
             lambda n: f" énuméré {want_proj[n]} · fichier {got[n][1]}")
        show("tenseurs en trop", extra, lambda n: f" {got[n][0]} {got[n][1]}")
        if miss_p or bad_p or extra:
            fail("l'énumération ne décrit pas ce fichier")
        else:
            print(f"  ok    {len(want_proj)} projections, nom et forme "
                  "identiques à l'énumération")

    # --- bytes, summed from the real dtypes -------------------------------
    unknown = sorted({dt for dt, _ in got.values() if dt not in DTYPE_BYTES})
    if unknown:
        raise FloorError(f"dtypes safetensors inconnus : {unknown}")
    carried_bytes = sum(numel(s) * DTYPE_BYTES[dt]
                        for n, (dt, s) in got.items() if n in want_carried)
    total_bytes = sum(numel(s) * DTYPE_BYTES[dt] for dt, s in got.values())
    proj_bytes = total_bytes - carried_bytes
    hist: dict[str, int] = {}
    for dt, s in got.values():
        hist[dt] = hist.get(dt, 0) + numel(s)
    print("\n  dtypes : " + ", ".join(f"{k} {gi(v)} valeurs"
                                      for k, v in sorted(hist.items())))
    print(f"  octets de tenseurs : {gi(total_bytes)}"
          f"   (projections {gi(proj_bytes)} · portés {gi(carried_bytes)})")
    if packed:
        # Per-suffix, because the packed branch has no "tenseurs en trop" check
        # to lean on: a repo that ships a zero `bias` per projection, or an
        # `inv_freq`, adds bytes no `quantization_config` declares, and the
        # confrontation below would say « l'un des deux ment » about tensors
        # nobody has named. Naming them turns an accusation into a diagnosis.
        per: dict[str, tuple[int, int]] = {}
        for n, (dt, s) in got.items():
            if n in want_carried:
                continue
            suf = n.rsplit(".", 1)[-1]
            cnt, b = per.get(suf, (0, 0))
            per[suf] = (cnt + 1, b + numel(s) * DTYPE_BYTES[dt])
        print("  détail des projections, par suffixe :")
        for suf, (cnt, b) in sorted(per.items(), key=lambda kv: -kv[1][1]):
            print(f"    {suf:<12}{cnt:>5} tenseurs {gi(b):>18} o")
    implied = proj_bytes * 8 / c["projection_weights"]
    print(f"  débit implicite sur les projections : {gf(implied, 6)} b/poids")

    # The campaign's whole iso-perimeter claim (plan §2.4) is that the carried
    # half is 16 bits **in every arm**. `arms()` assumes it without ever
    # looking; here the bytes are in hand, so it gets asserted. A repo shipping
    # f32 norms would silently add 389 Mo to a floor this script publishes.
    carried_rate = carried_bytes * 8 / c["carried_weights"]
    if abs(carried_rate - 16.0) > 1e-9:
        fail(f"les tenseurs portés pèsent {gf(carried_rate, 6)} b/poids, pas 16 "
             "— l'iso-périmètre du plan §2.4 ne tient pas sur ce dépôt")
    else:
        print("  ok    portés à 16,000000 b/poids, comme dans tous les bras")

    if not packed:
        if total_bytes != c["total_weights"] * 2:
            fail("un checkpoint dense en 16 bits devrait peser "
                 f"{gi(c['total_weights'] * 2)} o")
        # Pins `projection_weights` itself against the header, which the total
        # above does not: a miscount that moved weights between the two halves
        # would keep the total right and the rate wrong.
        elif abs(implied - 16.0) > 1e-9:
            fail(f"projections à {gf(implied, 6)} b/poids sur un checkpoint 16 "
                 "bits — le compte de poids de projection ne décrit pas ce fichier")
        else:
            print("  ok    projections à 16,000000 b/poids : le compte de poids "
                  "de projection retombe sur les octets")
    else:
        has_gidx = any(k.endswith(".g_idx") for k in got)
        d = declared_rate(cfg, c, has_gidx)
        if d is None:
            print("  ⚠️  aucun `quantization_config` exploitable : rien à "
                  "confronter, le débit implicite ci-dessus est tout ce qu'on a")
        elif d["unaligned"]:
            # `group_terms` counts `d_in // group` groups per row; on a width
            # the group size does not divide, the real file pads and the count
            # is wrong. Refusing beats printing a confrontation whose two sides
            # are both this script's.
            fail(f"largeurs non divisibles par {d['group']} : {d['unaligned']} — "
                 "le compte de groupes serait faux, rien n'est confronté")
        else:
            print(f"  déclaré : {d['method']} w{d['bits']}g{d['group']}"
                  f" → {d['terms']} = {gf(d['rate'], 6)} b/poids"
                  + ("" if has_gidx else "   (aucun g_idx dans le fichier)"))
            if abs(implied - d["rate"]) > 1e-6:
                gap = proj_bytes - bits_to_bytes(d["bits_total"])
                fail(f"les octets du dépôt donnent {gf(implied, 6)} b/poids, sa "
                     f"déclaration {gf(d['rate'], 6)} — écart {gi(gap)} o, "
                     "à retrouver dans le détail par suffixe ci-dessus")
            else:
                print("  ok    les octets expédiés et la déclaration disent le "
                      "même débit, à 1e-6 près")

    print("\n  " + ("ok    ce dépôt est décrit exactement par l'arithmétique de "
                    "ce script" if ok else
                    "FAIL  écart non expliqué — aucun plancher publiable avant "
                    "correction"))
    return 0 if ok else 1


# --- commands ---------------------------------------------------------------


def parse_contexts(spec: str, cfg: dict) -> list[int]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "max":
            out.append(int(cfg.get("max_position_embeddings", 0)))
        else:
            out.append(int(tok))
    return sorted({x for x in out if x >= 0})


def cmd_floor(args) -> int:
    cfg = load_config(args.model, args.revision)
    contexts = parse_contexts(args.contexts, cfg)
    pair = tuple(x.strip() for x in args.pair.split(","))
    if len(pair) != 2:
        raise FloorError("--pair attend deux clés de bras séparées par une virgule")
    rep = report(args.model, cfg, args.dtype, contexts, pair,
                 int(args.activations_gb * 1e9))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        print(render(rep))
        if args.out:
            print(f"  JSON écrit : {args.out}\n")
    return 0


def cmd_selftest(args) -> int:
    """Confront the arithmetic with the integers read off the published objects.

    Not a smoke test. Every line below is an **exact** equality against a number
    somebody measured — the file's own counts, the bench's traffic, the AWQ
    file's bytes on the Hub — and the point is that a `config.json` plus shape
    arithmetic has to land on all of them at once. A rate that drifts, a norm
    tensor forgotten, a group term mis-modelled: any one of them breaks a line
    here rather than surfacing as an unexplained gigabyte in a paper.
    """
    cfg = load_config(args.config or "Qwen/Qwen3-4B", args.revision)
    c = counts(cfg)
    m = QWEN3_4B
    ok = True

    def shown(v) -> str:
        # `bool` is an `int` in Python, so the grouped-integer branch would
        # print `False` as « 0 » — unreadable next to a real count.
        if isinstance(v, bool):
            return "oui" if v else "non"
        return gi(v) if isinstance(v, int) else str(v)

    def check(label: str, got, want) -> None:
        nonlocal ok
        if got == want:
            print(f"ok    {label} = {shown(want)}")
        else:
            ok = False
            print(f"FAIL  {label} : {shown(got)}, attendu {shown(want)}")

    def close(label: str, got: float, want: float, tol: float, why: str = "") -> None:
        nonlocal ok
        tail = f" {why}" if why else ""
        if abs(got - want) <= tol:
            print(f"ok    {label} = {gf(got, 6)} (attendu {gf(want, 6)}){tail}")
        else:
            ok = False
            print(f"FAIL  {label} : {gf(got, 9)}, attendu {gf(want, 9)} ± {tol}")

    print("— les trois égalités exigées —")
    check("poids de projection", c["projection_weights"], m["quantized"])
    check("poids du modèle", c["total_weights"], m["total"])
    check("plancher f16 (o)", c["total_weights"] * 2, m["f16_floor"])
    if not ok:
        print("\nFAIL  ce config.json ne décrit pas Qwen3-4B : rien d'autre "
              "n'a de sens, arrêt.")
        return 1

    print("\n— la comptabilité de notre format, sommée sur les formes —")
    check("poids portés", c["carried_weights"], m["carried"])
    check("matrices", c["matrices"], m["matrices"])
    check("blocs de 24", c["blocks"], m["blocks"])
    check("poids de queue KeepExact", c["tail_weights"], m["tail"])
    check("lignes de sortie / échelles", c["rows"], m["rows"])
    check("poids réellement quantifiés", c["block_weights"],
          m["total"] - m["carried"] - m["tail"])
    payload = llvq_payload_bits(c)
    check("charge utile (o)", bits_to_bytes(payload), m["payload_bytes"])
    check(
        "queue lue en f32 par le noyau (o)",
        c["tail_weights"] * 4,
        67_829_760,
    )
    check(
        "écart charge utile+portés → fichier scellé (o)",
        m["artifact_bytes"] - (bits_to_bytes(payload) + c["carried_weights"] * 2),
        m["artifact_overhead"],
    )
    close("b/poids de projection", payload / c["projection_weights"], 2.159506, 5e-7)
    close("b/poids quantifiés", payload / c["block_weights"], 2.169632, 5e-7)

    print("\n— l'adversaire, confronté à ses propres octets —")
    r = adversary_rates(cfg, c)
    close("AWQ w4g128 b/poids", r["awq_w4g128"]["rate"], 4.15625, 0.0)
    # `arms()` converts each of these to bytes; if one were not a whole number
    # of bytes the conversion would round, and a rounded floor is a wrong floor.
    # The precondition is checked here, and the rounding **direction** is
    # checked on a value that exercises it — on Qwen3-4B every total is already
    # byte-aligned, so `bits_to_bytes` is at its neutral value everywhere else.
    check("les trois débits adverses tombent sur un octet entier",
          sorted({x["bits"] % 8 for x in r.values()}), [0])
    check("bits_to_bytes arrondit vers le HAUT (9 bits = 2 octets)",
          bits_to_bytes(9), 2)
    awq_resident = bits_to_bytes(r["awq_w4g128"]["bits"]) + c["carried_weights"] * 2
    check("AWQ résident (o)", awq_resident, m["awq_bytes"] - m["awq_header"])
    check("AWQ fichier (o) = résident + en-tête", awq_resident + m["awq_header"],
          m["awq_bytes"])
    close("bnb NF4 bs64 b/poids", r["bnb_nf4_bs64"]["rate"], 4.5, 0.0)
    close(
        "GPTQ w2g128 — chiffre du plan reproduit",
        r["gptq_w2g128"]["rate_gidx_on_model_weights"],
        PLAN_GPTQ_W2G128,
        1e-5,
        "(terme g_idx sur les poids du modèle, comme le plan)",
    )
    close(
        "GPTQ w2g128 — homogène sur les projections",
        r["gptq_w2g128"]["rate"],
        2 + 18 / 128 + gptq_gidx_bits(cfg) / c["projection_weights"],
        0.0,
    )
    # The line above is an identity between two spellings of the same terms; it
    # would survive a `gptq_gidx_bits` that is wrong on both sides. This one
    # pins the abscisse the campaign would actually print.
    close("GPTQ w2g128 — l'abscisse à publier", r["gptq_w2g128"]["rate"],
          2.149067, 5e-7)
    if not r["awq_w4g128"]["unaligned"]:
        print("ok    aucune largeur non alignée sur 128 : l'adversaire n'a pas "
              "de queue, nous en avons une")

    print("\n— le noyau et la dilution —")
    rows = arms(cfg, c, "f16")
    by = {x["key"]: x for x in rows}
    check("Slot32, projections (o) = trafic mesuré par thesis",
          by["a1_slot32"]["projection_bytes"], m["slot32_proj_bytes"])
    # The other two layouts have published Go figures in fiche-4b §6.7, and
    # without them nothing distinguishes the **large** metric this file uses
    # from the **narrow** one — the very confusion that makes fiche §5.6 quote
    # 32,87 Go where its own §6.7 implies 34,14. Two anchors, one per layout.
    check("Flat32, projections (o) = les 2,125 Go de fiche §6.7",
          by["a1_flat32"]["projection_bytes"], 2_124_536_021)
    check("Grouped32, projections (o) = les 1,589 Go de fiche §6.7",
          by["a1_grouped32"]["projection_bytes"], 1_588_758_184)
    check("A1 tel que chargé (o) = plancher f16", by["a1_loaded"]["resident_bytes"],
          m["f16_floor"])
    check("cache KV (o/token)", kv_bytes_per_token(cfg, "f16"), m["kv_bytes_per_token"])

    # The two columns a reader actually looks at, and neither was pinned:
    # `ratio_vs_dense` inverted, or `model_bits_per_weight` taken over the
    # projections instead of the model, both print a plausible table.
    # ⚠️ They are checked on rows where the value is **not** neutral. Pinning
    # them on `a1_loaded` alone would prove nothing: there the ratio is 1,00,
    # and 1/x = x. Same trap as the λ = 0 of CLAUDE.md §5.
    close("fichier scellé — b/poids sur le MODÈLE entier",
          by["a1_file"]["model_bits_per_weight"], 3.521276, 5e-7, "(fiche §3.3)")
    close("fichier scellé — compression contre le f16",
          by["a1_file"]["ratio_vs_dense"], 4.543807, 5e-7, "(fiche §3.3)")
    close("Slot32 — ×f16", by["a1_slot32"]["ratio_vs_dense"], 2.452163, 5e-7)
    close("AWQ — b/poids sur le MODÈLE entier",
          by["awq_w4g128"]["model_bits_per_weight"], 5.302069, 5e-7, "(plan §3.3)")
    close("A1 tel que chargé — ×f16", by["a1_loaded"]["ratio_vs_dense"], 1.0, 0.0,
          "(le résultat que ce fichier existe pour rendre visible)")

    # The plan publishes this row to two decimals with a 1 Go activation term.
    # Tolerance is 7e-3, not 5e-3, for one reason worth naming rather than
    # burying: the ctx 8192 entry is **truncated** (1,1261 → 1,12) where the
    # other three are rounded. A modelling error would be an order of magnitude
    # larger than that, so the check still bites.
    dl = dilution(rows, kv_bytes_per_token(cfg, "f16"), [0, 4096, 8192, 32768],
                  "a1_slot32", "awq_w4g128", int(1e9))
    for ln, want in zip(dl["lines"], (1.231, 1.14, 1.12, 1.07)):
        close(f"AWQ contre Slot32 à ctx {ln['context']}", ln["ratio"], want, 7e-3,
              "(plan §3.3)")
    print("      note : le plan écrit ×1,12 à ctx 8192 pour 1,1261 — tronqué, "
          "pas arrondi")

    # The plan's own sentence calls ×1,231 « l'avantage de Slot32 sur AWQ »,
    # about a number that is AWQ's — Slot32 is the BIGGER of the two. That is
    # the error `favours` exists to prevent, so it gets pinned, in both
    # argument orders: a ratio that reads `small/big` would agree with itself.
    check("la dilution est en faveur de", dl["lines"][0]["favours"], "B0·AWQ")
    rev = dilution(rows, kv_bytes_per_token(cfg, "f16"), [0],
                   "awq_w4g128", "a1_slot32", int(1e9))
    check("échanger A et B ne change ni le rapport ni le vainqueur",
          (rev["lines"][0]["ratio"], rev["lines"][0]["favours"]),
          (dl["lines"][0]["ratio"], "B0·AWQ"))

    print("\n— trois autres largeurs, hors réseau (plan §5, ticket FL) —")
    for name, ref in REFERENCE_WIDTHS.items():
        rc = counts(ref["cfg"])
        check(f"{name} — poids de projection", rc["projection_weights"],
              ref["projection_weights"])
        check(f"{name} — poids du modèle", rc["total_weights"], ref["total_weights"])
        check(f"{name} — cache KV (o/token)",
              kv_bytes_per_token(ref["cfg"], "f16"), ref["kv_bytes_per_token"])
        close(f"{name} — part des poids portés",
              rc["carried_weights"] / rc["total_weights"], ref["carried_share"],
              5e-4)
        # The measured constants are 4B constants. Attaching them to another
        # width would put `bin/thesis` traffic under a 70B report.
        check(f"{name} — n'hérite d'aucune mesure du 4B",
              is_reference_4b(rc), False)
        if "tailless_width" in ref:
            proj, _ = enumerate_tensors(ref["cfg"])
            tail = sum(d_out * (d_in % LLVQ_BLOCK)
                       for _, (d_out, d_in) in proj
                       if d_in == ref["tailless_width"])
            check(f"{name} — d_in = {ref['tailless_width']} = 24×512, donc queue nulle",
                  tail, 0)
        if "four_block_projections" in ref:
            four = dict(ref["cfg"], num_hidden_layers=4)
            check(f"{name} — 4 blocs sur 64 (dé-risquage)",
                  counts(four)["projection_weights"],
                  ref["four_block_projections"])
        if "qk_norm" in ref:
            check(f"{name} — q_norm/k_norm par tête",
                  has_qk_norm(ref["cfg"]), ref["qk_norm"])

    print("\n" + ("ok    toutes les égalités tiennent" if ok
                  else "FAIL  au moins une égalité est fausse"))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("plancher", help="plancher résident, par bras et par layout")
    f.add_argument("model", nargs="?", default="Qwen/Qwen3-4B",
                   help="repo id, dossier, ou chemin d'un config.json")
    f.add_argument("--revision", default="main")
    f.add_argument("--dtype", default="f16", choices=sorted(DTYPE_BITS),
                   help="précision des poids denses et du cache KV")
    f.add_argument("--contexts", default="0,1024,4096,8192,32768,max")
    f.add_argument("--pair", default="a1_slot32,awq_w4g128",
                   help="les deux bras dont on montre la dilution")
    f.add_argument("--activations-gb", type=float, default=1.0,
                   help="terme d'activations supposé, commun aux bras")
    f.add_argument("--json", action="store_true", help="JSON sur stdout, sans table")
    f.add_argument("--out", default=None, help="écrire aussi le JSON dans un fichier")
    f.set_defaults(fn=cmd_floor)

    s = sub.add_parser("selftest", help="confronter l'arithmétique aux objets mesurés")
    s.add_argument("--config", default=None,
                   help="config.json local, à défaut Qwen/Qwen3-4B depuis le Hub")
    s.add_argument("--revision", default="main")
    s.set_defaults(fn=cmd_selftest)

    v = sub.add_parser("verify-shapes",
                       help="confronter l'énumération aux en-têtes safetensors du Hub")
    v.add_argument("repo", help="ex. Qwen/Qwen3-4B")
    v.add_argument("--config", default=None,
                   help="config.json local si le repo n'en expose pas")
    v.add_argument("--revision", default="main")
    v.set_defaults(fn=cmd_verify_shapes)

    args = p.parse_args()
    try:
        return args.fn(args)
    except FloorError as e:
        print(f"\nrefus : {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
