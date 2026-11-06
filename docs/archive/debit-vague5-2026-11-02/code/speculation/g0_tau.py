# DRAFT, never compiled or run.
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.14,<2.15",
#     "transformers==5.18.0",  # Cache.crop semantics: see shim_cache_crop()
#     "accelerate>=1.0",
#     "safetensors>=0.4",
#     "numpy>=2.0",
#     "huggingface_hub>=1.0",
#     "llvq-tetra",
# ]
# [tool.uv.sources]
# llvq-tetra = { path = "/Users/pjmalandrino/Documents/Pro/workspace/poc/llvq/llvq-tetra" }
# ///
"""DRAFT, never compiled or run. Written read-only on 2026-11-02 as a design sketch for G0
(`prereg-speculation-1.md`), revised the same day after an adversarial review. Every line below is
unexecuted; expect bugs. Running it is a local run and needs the operator's go (CLAUDE.md rule 1) and a
stamped prereg (rule 2).

G0: tau of DeepSpec's DSpark, DFlash and EAGLE-3 drafters (block 7: 7 input slots = anchor + 6 mask ids,
7 proposals, 8 verify rows) and of prompt lookup, against the sealed Qwen3-4B read by llvq-tetra's dense
path, and against Qwen3-4B (BF16 checkpoint, run in f16).

    nice -n 10 uv run g0_tau.py <stage> --out ~/llvq-g0/ --corpora corpora.jsonl

Stages: check (L1), pilot (D0), v1, refs, lookup (4b), served (4c, optional), taps, replay, v2, agree,
probes, think, report. Machine load voids nothing: no output here depends on timing (a `ps` line is
logged per stage as information).

Design (speculation.md section 3.1):
* Greedy acceptance makes a speculative run a deterministic function of (the target's greedy
  continuation, the drafter). So each target decodes each prompt once, plainly, then one teacher-forced
  forward gives the 5 taps for every position, and the speculative loop is replayed offline: one target
  forward per prompt instead of one per round.
* DeepSpec is vendored at a pinned commit, sha256 per file, and its package __init__ files and
  deepspec.utils are replaced by stubs (the real ones import the optimizer and the distributed stack).
  eval.py is never used: it spawns torch.cuda.device_count() workers, zero on a Mac.
* Models load on the CPU, then move to mps: a direct device_map="mps" fp16 load segfaulted
  (preregistration-hf-metal-bloc-2026-10-09-ECARTS.md, section E4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import types
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------------------------------------
# Pins. The sha256 values are filled at L1 from the first fetch and copied into the prereg before
# it is stamped; a mismatch afterwards voids the run (prereg section 4, gate 3).
# ----------------------------------------------------------------------------------------------
DEEPSPEC_SHA = "005e03b81cec38b7da6399833d609ee89a2587f2"
VENDORED = {
    "deepspec/utils/sampling.py": None,
    "deepspec/data/parser.py": None,
    "deepspec/modeling/dspark/common.py": None,
    "deepspec/modeling/dspark/markov_head.py": None,
    "deepspec/modeling/dspark/qwen3/__init__.py": None,
    "deepspec/modeling/dspark/qwen3/config.py": None,
    "deepspec/modeling/dspark/qwen3/modeling.py": None,
    "deepspec/modeling/eagle3/common.py": None,
    "deepspec/modeling/eagle3/qwen3/modeling.py": None,
    "deepspec/eval/base_evaluator.py": None,
    "deepspec/eval/dspark/draft_ops.py": None,
    # Table 1's inputs (base_evaluator.py:31-55 reads turns[:1]; :513-525 shuffles GSM8K with
    # random.Random(980406) and keeps 500). V1 is judged against Table 1, so these are pinned too.
    "eval_datasets/mt-bench.jsonl": None,
    "eval_datasets/humaneval.jsonl": None,
    "eval_datasets/gsm8k.jsonl": None,
}
CORPORA_SHA256 = None   # corpora.jsonl, copied into prereg section 2 before stamping; a mismatch voids
DEEPSPEC_SEED = 980406
STUBS = {
    "deepspec/__init__.py": "",
    "deepspec/utils/__init__.py": (
        "import random, json\nimport numpy as np, torch\n"
        "def seed_all(seed):\n    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)\n"
        "def init_dist(*a, **k):\n    raise RuntimeError('G0 runs one process; init_dist is not used')\n"
        "def add_metric(*a, **k):\n    return None\n"
        "def jsonable(x):\n    return json.loads(json.dumps(x, default=str))\n"
        "class CustomJSONEncoder(json.JSONEncoder):\n    def default(self, o):\n        return str(o)\n"
    ),
    "deepspec/utils/metrics.py": "def add_metric(*a, **k):\n    return None\n",
    "deepspec/data/__init__.py": "",
    "deepspec/modeling/__init__.py": "",
    "deepspec/modeling/dspark/__init__.py": (
        "from .common import DSparkForwardOutput, extract_context_feature\n"
        "from .gemma4 import Gemma4DSparkModel\nfrom .qwen3 import Qwen3DSparkModel\n"
    ),
    # draft_ops.py builds `Qwen3DSparkModel | Gemma4DSparkModel` at import time: it needs a class.
    "deepspec/modeling/dspark/gemma4/__init__.py": "class Gemma4DSparkModel:\n    pass\n",
    "deepspec/modeling/eagle3/__init__.py": "from .common import extract_eagle3_context_feature\n",
    "deepspec/modeling/eagle3/qwen3/__init__.py": "from .modeling import Qwen3Eagle3Model\n",
    "deepspec/eval/__init__.py": "",
    "deepspec/eval/dspark/__init__.py": (
        "from .draft_ops import DSparkDraftProposal, build_dspark_proposal, forward_dspark_draft_block\n"
    ),
}

SEALED_REPO, SEALED_REV = "Pier-Jean/Qwen3-4B-LLVQ-Tetra", "acc5f663ea645886febe41c8abb39fea2b78c4a6"
SEALED_ARTIFACT_PREFIX = "886391a8"
DENSE_REPO, DENSE_REV = "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"
DRAFTERS = {
    "dspark": ("deepseek-ai/dspark_qwen3_4b_block7", "3457dff1417cb84927f6098a5fcb7cee85c934b7"),
    "dflash": ("deepseek-ai/dflash_qwen3_4b_block7", "02d530b7962ea1412beaf41a05c0b8e36d5f9b1d"),
    "eagle3": ("deepseek-ai/eagle3_qwen3_4b_ttt7", "b0b90fd15d052217c226be5e46d468d8d129e0cd"),
}
TAP_LAYERS = [1, 9, 17, 25, 33]
MAX_NEW = 512
PREFILL_CHUNK = 1024
STOP_IDS = (151643, 151645)   # <|endoftext|>, <|im_end|>: bin/gsm8k stops on both (gsm8k.rs:304-312)
MASK_ID = 151669              # DSpark/DFlash mask slots read embed_tokens at this row (draft_ops.py:37)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# S is reported over a grid of beta = (t(k)/t(1) - 1)/(k - 1); routing (D3) reads S at the beta that L1a
# measures (prereg-speculation-0.md section 5). The reference points are *estimated* (speculation.md 3.3):
# today's exact path 0.50 / 0.58 / 0.71 at 5 / 2,048 / 8,192 positions; after B2 0.28 / 0.39 / 0.59;
# after B2 + B3 0.19 / 0.16 / 0.11; flat verify 0.03. d = draft time / t(1), neural drafters only.
BETA_GRID = [0.03, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
D_NEURAL = {0.03: 0.25, 0.10: 0.25, 0.20: 0.30}   # flatter verify, cheaper drafter pass; else 0.70
T1_MS = {5: 7.43, 2048: 8.78, 8192: 12.90}       # grouped door, *measured* (served-grouped-2026-11-02)
KV_US_PER_POS = 0.668                             # grouped slope, *computed* from the same medians
B2_SAVING_MS = 0.55 + 0.19 + 0.875                # int4 rows, head rows, no regroup (*estimated*)


def t1_ms(pos: int) -> float:
    """The grouped step at a context position, on the fitted line anchored at 133.5 (requests.py)."""
    return 7.43 + KV_US_PER_POS * 1e-3 * (pos - 133.5)


def beta_exact(t_pf: dict, pos: int, launch_us: float = 3.5) -> float:
    """Exact-verify beta at `pos` from L1a's measured prefill-path t_pf[k] (ms), k = 1 and 4:
    (t_pf(4) - t_pf(1) - mask 0.23)/3 + head 0.32 + 216 attention launches + KV read, over t1(pos)."""
    row = (t_pf[4] - t_pf[1] - 0.23) / 3 + 0.32 + 216 * launch_us * 1e-3 + KV_US_PER_POS * 1e-3 * pos
    return row / t1_ms(pos)


# ----------------------------------------------------------------------------------------------
# Vendoring and the transformers shim
# ----------------------------------------------------------------------------------------------
def vendor_deepspec(root: Path) -> dict[str, str]:
    """Fetch the pinned files, check them, write stubs; return {path: sha256}."""
    got = {}
    for rel, want in VENDORED.items():
        dst = root / rel
        if not dst.exists():
            url = f"https://raw.githubusercontent.com/deepseek-ai/DeepSpec/{DEEPSPEC_SHA}/{rel}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as r:
                dst.write_bytes(r.read())
        digest = hashlib.sha256(dst.read_bytes()).hexdigest()
        if want is not None and digest != want:
            raise SystemExit(f"VOID: {rel} sha256 {digest} is not the prereg's {want}")
        got[rel] = digest
    for rel, text in STUBS.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    sys.path.insert(0, str(root))
    return got


class KeepCache:  # placeholder name, replaced in shim_cache_crop()
    pass


def shim_cache_crop() -> str:
    """DeepSpec calls cache.crop(start) meaning "keep `start` positions" (transformers 5.10.2). In 5.18.0,
    Cache.crop(tokens_to_remove) keeps that reading only on a deprecated path for positive values
    (cache_utils.py:175-190, :1631), announced for removal. Every DynamicCache the vendored modules create
    becomes this subclass, whose crop(n) keeps n positions whatever the transformers version does."""
    import transformers
    from transformers import DynamicCache

    class KeepPositionsCache(DynamicCache):
        def crop(self, keep: int) -> None:
            have = self.get_seq_length()
            if keep < have:
                super().crop(-(have - keep))   # the 5.18 non-deprecated form: remove that many

    global KeepCache
    KeepCache = KeepPositionsCache
    import deepspec.eval.base_evaluator as be
    import deepspec.eval.dspark.draft_ops as do
    be.DynamicCache = KeepPositionsCache          # generate_decoding_sample creates its own target cache
    do.DynamicCache = KeepPositionsCache
    return f"DynamicCache -> KeepPositionsCache (transformers {transformers.__version__})"


def shim_transformers() -> list[str]:
    """DeepSpec imports these names from modeling_qwen3 under transformers 5.10.2; 5.18 may have
    moved some. Each shim is printed and recorded as a deviation from verbatim behaviour."""
    import transformers.models.qwen3.modeling_qwen3 as mq

    fallbacks = {
        "ALL_ATTENTION_FUNCTIONS": ("transformers.modeling_utils", "ALL_ATTENTION_FUNCTIONS"),
        "FlashAttentionKwargs": ("transformers.modeling_flash_attention_utils", "FlashAttentionKwargs"),
        "GradientCheckpointingLayer": ("transformers.modeling_layers", "GradientCheckpointingLayer"),
    }
    shimmed = []
    for name in ["ALL_ATTENTION_FUNCTIONS", "FlashAttentionKwargs", "GradientCheckpointingLayer",
                 "Qwen3MLP", "Qwen3PreTrainedModel", "Qwen3RMSNorm", "Qwen3RotaryEmbedding",
                 "eager_attention_forward", "rotate_half"]:
        if hasattr(mq, name):
            continue
        if name not in fallbacks:
            raise SystemExit(f"VOID: modeling_qwen3 lacks {name} and no fallback is defined")
        mod, attr = fallbacks[name]
        setattr(mq, name, getattr(__import__(mod, fromlist=[attr]), attr))
        shimmed.append(name)
    print(f"transformers shims: {shimmed or 'none'}")
    return shimmed


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------
def load_target(which: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if which == "sealed":
        import llvq_tetra  # noqa: F401  (registers the "llvq" quantizer; without it the weights are random)

        repo, rev = SEALED_REPO, SEALED_REV
    else:
        repo, rev = DENSE_REPO, DENSE_REV
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    # No device_map: llvq-tetra rebuilds every record dense on the CPU (integration.py:153-171).
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=dtype).eval()
    if which == "sealed":
        sha = model.config.quantization_config.artifact_sha256
        if not str(sha).startswith(SEALED_ARTIFACT_PREFIX):
            raise SystemExit(f"VOID: artifact_sha256 {sha} is not the served 4B")
    return model.to(DEVICE), tok


HEAD_ARMS = ("own", "shared", "shared-bf16mask")


class F32TableHead(torch.nn.Module):
    """The shared head as f32 logits against the target's f16 table, never a bf16 copy of it: the bf16
    cast would round the sealed int4 values the target's own head uses. 1.56 GB of f32 on mps."""

    def __init__(self, table_f16: torch.Tensor):
        super().__init__()
        self.register_buffer("w", table_f16.detach().float(), persistent=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(h.float(), self.w)


def load_drafter(name: str, head: str, target):
    """head arms (prereg-speculation-1.md section 2):
    'own'             the published BF16 frozen copy of the target's table, in lm_head and embed_tokens;
    'shared'          the target's table (int4 g64 on the sealed target) in both, mask row 151669 included;
    'shared-bf16mask' the target's table, with row 151669 kept from the drafter's own BF16 table (5,120 B).
    In both shared arms the head computes f32 logits against the f16 table (F32TableHead). EAGLE-3 reads
    no mask row, so its 'shared-bf16mask' arm is 'shared' and is not run."""
    if head not in HEAD_ARMS:
        raise SystemExit(f"unknown head arm {head}")
    repo, rev = DRAFTERS[name]
    if name == "eagle3":
        from deepspec.modeling.eagle3.qwen3 import Qwen3Eagle3Model as cls
    else:
        from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel as cls
    d = cls.from_pretrained(repo, revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa")
    d = d.to(DEVICE).eval()
    if [int(x) for x in d.target_layer_ids] != TAP_LAYERS:
        raise SystemExit(f"VOID: {name} taps {d.target_layer_ids}")
    if name != "eagle3" and int(d.mask_token_id) != MASK_ID:
        raise SystemExit(f"VOID: {name} mask id {d.mask_token_id}")
    if head == "own":
        return d
    table = target.get_input_embeddings().weight.detach()          # f16 on mps, the sealed int4 values
    assert tuple(table.shape) == tuple(d.lm_head.weight.shape) == (151936, 2560)
    own_mask_row = d.embed_tokens.weight[MASK_ID].detach().clone()   # BF16, what the drafter trained on
    with torch.no_grad():
        d.embed_tokens.weight.copy_(table.to(torch.bfloat16))  # the body runs in bf16 anyway
        if head == "shared-bf16mask":
            d.embed_tokens.weight[MASK_ID].copy_(own_mask_row)
    d.lm_head = F32TableHead(table)   # compute_logits calls self.lm_head(h): f32 against the f16 table
    return d


def mask_row_report(target, drafter_own) -> dict:
    """L1 print, informational: row 151669 of the sealed table against the drafter's BF16 row."""
    a = target.get_input_embeddings().weight[MASK_ID].detach().float().cpu()
    b = drafter_own.embed_tokens.weight[MASK_ID].detach().float().cpu()
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    return dict(max_abs_delta=float((a - b).abs().max()), cosine=cos,
                norm_sealed=float(a.norm()), norm_bf16=float(b.norm()))


def v4_against_checkpoint(drafter, dense_repo: str = DENSE_REPO, dense_rev: str = DENSE_REV) -> float:
    """V4: the drafter's frozen tables against Qwen3-4B's BF16 checkpoint read in its OWN dtype with
    safetensors (an f16 cast is inexact below the f16 normal range). Returns max |delta|; must be 0."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    # Qwen3-4B is sharded; the embedding sits in one shard named by model.safetensors.index.json.
    idx = json.loads(Path(hf_hub_download(dense_repo, "model.safetensors.index.json",
                                          revision=dense_rev)).read_text())
    shard = idx["weight_map"]["model.embed_tokens.weight"]
    with safe_open(hf_hub_download(dense_repo, shard, revision=dense_rev), "pt") as f:
        ref = f.get_tensor("model.embed_tokens.weight")              # torch.bfloat16
    worst = 0.0
    for t in (drafter.embed_tokens.weight, drafter.lm_head.weight):
        worst = max(worst, float((t.detach().cpu().to(torch.bfloat16).float() - ref.float()).abs().max()))
    return worst


# ----------------------------------------------------------------------------------------------
# Reference continuations and taps
# ----------------------------------------------------------------------------------------------
def stop_ids(model, tok) -> list[int]:
    """The generation config's EOS ids, which must include both of bin/gsm8k's stops; E0 stops on the
    same two, so G0's references and E0's timed decodes cover the same text."""
    eos = getattr(model.generation_config, "eos_token_id", None) or tok.eos_token_id
    got = {int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos])}
    if not set(STOP_IDS) <= got:
        print(f"note: generation_config stops {sorted(got)}; adding {STOP_IDS}")
    return sorted(got | set(STOP_IDS))


@torch.inference_mode()
def greedy_reference(model, prompt_ids: list[int], stops: list[int], max_new: int = MAX_NEW) -> list[int]:
    """Plain greedy, one row a step after a chunked prefill: the decode the engine would do."""
    from transformers import DynamicCache

    cache, out = DynamicCache(), []
    logits = None
    for i in range(0, len(prompt_ids), PREFILL_CHUNK):
        part = torch.tensor([prompt_ids[i:i + PREFILL_CHUNK]], device=DEVICE)
        logits = model(input_ids=part, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
    nxt = int(logits[0, -1].float().argmax())
    while True:
        out.append(nxt)
        if nxt in stops or len(out) == max_new:
            return out
        logits = model(input_ids=torch.tensor([[nxt]], device=DEVICE), past_key_values=cache,
                       use_cache=True).logits
        nxt = int(logits[0, -1].float().argmax())


@torch.inference_mode()
def taps_of(model, ids: list[int]) -> torch.Tensor:
    """[1, n, 5*2560] raw decoder-layer outputs of layers 1, 9, 17, 25, 33 (hidden_states[i+1]),
    chunked so no score tensor spans the whole sequence."""
    from transformers import DynamicCache

    cache, parts = DynamicCache(), []
    for i in range(0, len(ids), PREFILL_CHUNK):
        part = torch.tensor([ids[i:i + PREFILL_CHUNK]], device=DEVICE)
        hs = model(input_ids=part, past_key_values=cache, use_cache=True,
                   output_hidden_states=True, logits_to_keep=1).hidden_states
        parts.append(torch.cat([hs[l + 1] for l in TAP_LAYERS], dim=-1))
    return torch.cat(parts, dim=1)


def loop_onset(cont: list[int], n: int = 16, times: int = 3) -> int | None:
    """Index of the third occurrence of any 16-gram of the continuation, else None."""
    seen: dict[tuple, int] = {}
    for i in range(len(cont) - n + 1):
        g = tuple(cont[i:i + n])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] == times:
            return i + n
    return None


def loop_truncated(ids: list[int], n_prompt: int) -> list[int]:
    """Prompt plus continuation cut at the loop onset. Prompt lookup always runs on this (rule D5): a
    loop is accepted almost whole by a lookup drafter and would inflate c and tau_d."""
    onset = loop_onset(ids[n_prompt:])
    return ids if onset is None else ids[:n_prompt + onset]


def by_parity(per_prompt: dict, parity: int) -> dict:
    """Prompt ids are integers; every choice made on data (gate, k, drafter, confidence temperature) is
    made on parity 0 (even ids) and scored on parity 1 (odd ids)."""
    return {k: v for k, v in per_prompt.items() if int(k) % 2 == parity}


# ----------------------------------------------------------------------------------------------
# Offline replays. Each returns rounds: dict(s, k, acc, accepted, pmax, conf). `acc` is DeepSpec's
# round length: accepted + 1, or accepted alone on a stop-token round (base_evaluator.py:415-423).
# ----------------------------------------------------------------------------------------------
@dataclass
class Round:
    s: int
    k: int
    accepted: int
    acc: int
    pmax: list
    conf: list | None


def _accept(props: list[int], ref: list[int], stops: list[int]) -> tuple[int, int, bool]:
    a = 0
    while a < len(props) and a < len(ref) and props[a] == ref[a]:
        a += 1
    for j in range(a):                      # an accepted stop token ends generation there
        if props[j] in stops:
            return j + 1, j + 1, True
    if a < len(ref) and ref[a] in stops:    # the bonus is a stop token
        return a, a + 1, True
    return a, a + 1, False


@torch.inference_mode()
def replay_dspark(d, taps: torch.Tensor, ids: list[int], n_prompt: int, stops: list[int]) -> list[Round]:
    """Mirrors Qwen3DSparkEvaluator._propose/_update with verification replaced by the reference.
    draft_ops.forward_dspark_draft_block is called unchanged; the proposal logic of
    build_dspark_proposal (draft_ops.py:96-153, threshold 0) is mirrored to also log logits."""
    from transformers import DynamicCache
    from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block

    block, n = int(d.block_size), len(ids)
    pos = torch.arange(n + block + 1, device=DEVICE).unsqueeze(0)
    cache, rounds, s, ctx_from = KeepCache(), [], n_prompt, 0
    while s < n - 1:
        draft_ids = torch.full((1, block), int(d.mask_token_id), dtype=torch.long, device=DEVICE)
        draft_ids[0, 0] = ids[s]
        hid = forward_dspark_draft_block(
            d, draft_input_ids=draft_ids, position_ids=pos, past_key_values_draft=cache,
            target_hidden_states=taps[:, ctx_from:s, :].to(torch.bfloat16), start=s, block_size=block)
        h = hid[:, :block, :]
        tokens, logits = d.sample_draft_tokens(d.compute_logits(h), first_prev_token_ids=draft_ids[:, 0],
                                               temperature=0.0, hidden_states=h)
        props = tokens[0].tolist()
        pmax = torch.softmax(logits[0].float(), dim=-1).max(dim=-1).values.tolist()
        conf = None
        if d.confidence_head is not None:
            prev = torch.cat([draft_ids[:, :1], tokens[:, :-1]], dim=1)
            conf = d.predict_confidence_step(h, prev_token_ids=prev).reshape(-1).tolist()
        accepted, acc, stop = _accept(props, ids[s + 1:s + 1 + block], stops)
        rounds.append(Round(s, block + 1, accepted, acc, pmax, conf))
        if stop:
            break
        ctx_from, s = s, s + accepted + 1   # the verify rows anchor..anchor+a become context
    return rounds


@torch.inference_mode()
def replay_eagle3(d, taps: torch.Tensor, ids: list[int], n_prompt: int, stops: list[int]) -> list[Round]:
    """Mirrors Qwen3Eagle3Evaluator (_init_context :61-96, _propose :98-146, _update :148-170): the
    draft cache pairs tap i with token i+1 at RoPE position i; one layer makes entries per-position."""
    from transformers import DynamicCache
    from deepspec.utils.sampling import sample_tokens

    n, ttt = len(ids), int(d.ttt_length)
    pos = torch.arange(n + ttt + 2, device=DEVICE).unsqueeze(0)
    x = lambda a, b: taps[:, a:b, :].to(torch.bfloat16)
    cache = KeepCache()
    hidden = d.extend_draft_cache(hidden_states=x(0, n_prompt),
                                  input_ids=torch.tensor([ids[1:n_prompt + 1]], device=DEVICE),
                                  position_ids=pos[:, :n_prompt], past_key_values=cache)
    rounds, s = [], n_prompt
    while s < n - 1:
        before, props, pmax, h, p = cache.get_seq_length(), [], [], hidden, s
        for _ in range(ttt):
            lg = d.compute_logits(h)
            t = sample_tokens(lg, temperature=0.0)
            props.append(int(t[0, -1]))
            pmax.append(float(torch.softmax(lg[0, -1].float(), -1).max()))
            if props[-1] in stops:
                break
            h = d(hidden_states=h, input_ids=t[:, -1:], position_ids=pos[:, p:p + 1],
                  past_key_values=cache, use_cache=True)
            p += 1
        accepted, acc, stop = _accept(props, ids[s + 1:s + 1 + len(props)], stops)
        rounds.append(Round(s, len(props) + 1, accepted, acc, pmax, None))
        if stop:
            break
        cache.crop(before)
        hidden = d.extend_draft_cache(hidden_states=x(s, s + accepted + 1),
                                      input_ids=torch.tensor([ids[s + 1:s + accepted + 2]], device=DEVICE),
                                      position_ids=pos[:, s:s + accepted + 1], past_key_values=cache)
        s += accepted + 1
    return rounds


def replay_prompt_lookup(ids: list[int], n_prompt: int, stops: list[int], k_max: int,
                         min_match: int, max_ngram: int = 4) -> list[Round]:
    """Longest 4/3/2-gram suffix of the HISTORY found earlier in it (most recent occurrence); propose
    what followed, from the history only. No match: a plain step (k = 1, acc = 1)."""
    rounds, s, n = [], n_prompt, len(ids)
    while s < n - 1:
        hist, props = ids[:s + 1], []
        for g in range(max_ngram, min_match - 1, -1):
            if len(hist) <= g:
                continue
            suffix = hist[-g:]
            for j in range(len(hist) - g - 1, -1, -1):   # never the suffix itself
                if hist[j:j + g] == suffix:
                    props = hist[j + g:j + g + k_max - 1]
                    break
            if props:
                break
        if not props:
            rounds.append(Round(s, 1, 0, 1, [], None))
            if ids[s + 1] in stops:
                break
            s += 1
            continue
        accepted, acc, stop = _accept(props, ids[s + 1:s + 1 + len(props)], stops)
        rounds.append(Round(s, len(props) + 1, accepted, acc, [], None))
        if stop:
            break
        s += accepted + 1
    return rounds


# ----------------------------------------------------------------------------------------------
# Agreement, S profiles, bootstrap
# ----------------------------------------------------------------------------------------------
@torch.inference_mode()
def agreement(model_a, model_b, ids: list[int], n_prompt: int) -> dict:
    """Teacher-forced top-1 agreement and KL(a || b) over continuation positions, f32 log-softmax.
    a = dense, b = sealed. Chunked: one [chunk, 151936] f32 pair at a time."""
    from transformers import DynamicCache

    ca, cb = DynamicCache(), DynamicCache()
    agree = kl = count = 0.0
    for i in range(0, len(ids) - 1, PREFILL_CHUNK):
        part = torch.tensor([ids[i:i + PREFILL_CHUNK]], device=DEVICE)
        la = model_a(input_ids=part, past_key_values=ca, use_cache=True).logits[0].float()
        lb = model_b(input_ids=part, past_key_values=cb, use_cache=True).logits[0].float()
        keep = torch.arange(i, i + part.shape[1], device=DEVICE)
        m = (keep >= n_prompt - 1) & (keep < len(ids) - 1)
        if m.any():
            pa, pb = torch.log_softmax(la[m], -1), torch.log_softmax(lb[m], -1)
            agree += float((pa.argmax(-1) == pb.argmax(-1)).sum())
            kl += float((pa.exp() * (pa - pb)).sum())
            count += float(m.sum())
    return dict(top1=agree / max(count, 1), kl=kl / max(count, 1), positions=int(count))


# At 14B (L3) two f16 targets do not fit one process (2 x 29.5 GB on 64 GB): each target runs alone and
# writes, per position, its argmax, its top-64 (ids, logits) and its logsumexp; full f16 log-softmax rows
# are kept only for a fixed 5 % sample of positions (seed 20261102). Agreement then reads the two caches:
# exact top-1, KL restricted to the union of top-64 supports (a lower bound), exact KL on the sample.
TOPK_CACHE = 64
FULL_ROW_SAMPLE = 0.05


@torch.inference_mode()
def cache_logits(model, ids: list[int], n_prompt: int, out: Path, key: str) -> None:
    rng = random.Random(f"20261102-{key}")
    from transformers import DynamicCache

    cache, rows = DynamicCache(), []
    for i in range(0, len(ids) - 1, PREFILL_CHUNK):
        part = torch.tensor([ids[i:i + PREFILL_CHUNK]], device=DEVICE)
        lg = model(input_ids=part, past_key_values=cache, use_cache=True).logits[0].float()
        lse = torch.logsumexp(lg, -1)
        top = lg.topk(TOPK_CACHE, -1)
        for j in range(lg.shape[0]):
            pos = i + j
            if pos < n_prompt - 1 or pos >= len(ids) - 1:
                continue
            row = dict(pos=pos, argmax=int(top.indices[j, 0]), lse=float(lse[j]),
                       top_ids=top.indices[j].tolist(), top_logits=top.values[j].tolist())
            if rng.random() < FULL_ROW_SAMPLE:
                row["full_f16"] = (lg[j] - lse[j]).half().cpu().numpy().tobytes().hex()
            rows.append(row)
    (out / f"logits-{key}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


# ----------------------------------------------------------------------------------------------
# Stage 4c (optional, $0): the served object's own GSM8K text. bin/gsm8k stored text, n_prompt, n_gen
# (stop excluded) and qhash per problem, no ids (run1-served, run4-f16dense of 2026-09-26). Rows must
# survive a re-tokenization round trip before they are used.
# ----------------------------------------------------------------------------------------------
def gsm8k_prompt_ids(tok, question: str) -> list[int]:
    """bin/gsm8k's prompt: one user turn of gsm8k::user_body(question), then the assistant opener with
    an empty think block (thinking off), chatfmt.rs. To be checked against `n_prompt` and `qhash`
    before any use; the exact user_body text is read from llvq-llm/src/gsm8k.rs at the pinned commit."""
    raise NotImplementedError("draft: port chatfmt.rs turn/open_assistant and gsm8k::user_body")


def token_fingerprint(ids: list[int]) -> str:
    """Port of llvq_llm::eval::token_fingerprint, checked on three dump rows before use."""
    raise NotImplementedError("draft: port eval::token_fingerprint")


def round_trip(tok, row: dict, question: str) -> list[int] | None:
    """Prompt + completion ids, or None when the row fails the round trip (counted, excluded)."""
    p = gsm8k_prompt_ids(tok, question)
    if len(p) != row["n_prompt"] or token_fingerprint(p) != row["qhash"]:
        return None
    c = tok.encode(row["completion"], add_special_tokens=False)
    if len(c) != row["n_gen"] or tok.decode(c, skip_special_tokens=False) != row["completion"]:
        return None
    return p + c


@torch.inference_mode()
def served_agreement(model, ids: list[int], n_prompt: int) -> dict:
    """Teacher-force the served text through the sealed rebuild: share of positions where the rebuild's
    argmax is the served token, and the first position where it is not (rule D7)."""
    from transformers import DynamicCache

    cache, hits, first, n = DynamicCache(), 0, None, 0
    for i in range(0, len(ids) - 1, PREFILL_CHUNK):
        part = torch.tensor([ids[i:i + PREFILL_CHUNK]], device=DEVICE)
        am = model(input_ids=part, past_key_values=cache, use_cache=True).logits[0].argmax(-1).tolist()
        for j, a in enumerate(am):
            pos = i + j
            if n_prompt - 1 <= pos < len(ids) - 1:
                ok = a == ids[pos + 1]
                hits, n = hits + ok, n + 1
                if not ok and first is None:
                    first = pos + 1 - n_prompt
    return dict(top1=hits / max(n, 1), first_divergence=first, positions=n)


def d0_pilot(q_chat: float, q_code: float) -> str:
    """Prereg rule D0, the declared interim look (DSpark, shared-bf16mask, 40 + 40 prompts)."""
    if q_chat >= 0.90 and q_code >= 0.90:
        return "propose D1 'both high' now; the full run continues for D2-D6 only on the operator's word"
    if q_chat <= 0.60 and q_code <= 0.60:
        return "propose D1 'any low' now; the full run shrinks to refs, lookup, 4c, agreement"
    return "full run"


def ps_snapshot(out: Path, stage: str) -> None:
    """Information only: machine load changes the wall clock, never tau, agreement or KL."""
    import subprocess

    line = subprocess.run(["ps", "-A", "-o", "%cpu=,comm="], capture_output=True, text=True).stdout
    busy = sorted((float(x.split(None, 1)[0]), x.split(None, 1)[1].strip())
                  for x in line.splitlines() if x.strip())[-5:]
    with (out / "ps.log").open("a") as f:
        f.write(json.dumps(dict(time=time.strftime("%H:%M:%S"), stage=stage, top=busy)) + "\n")


def speedup(rounds: list[Round], beta: float, d: float) -> float:
    """S = tokens / sum(r(k) + d) with r(k) = 1 + beta (k-1); a plain step (k = 1) costs 1, no d.
    Packet staircase ignored: K <= 4 stays in one 4-row packet; for K > 4 the caller adds a second
    weight pass (about +0.31 of a step, *estimated*) through `beta`'s choice, never silently."""
    tokens = sum(r.acc for r in rounds)
    cost = sum(1.0 if r.k == 1 else 1.0 + beta * (r.k - 1) + d for r in rounds)
    return tokens / max(cost, 1e-9)


def s_over_grid(rounds: list[Round], neural: bool) -> dict:
    """S at every beta of BETA_GRID (outcome of prereg section 5); d is paid by neural drafters only."""
    return {b: speedup(rounds, b, D_NEURAL.get(b, 0.70) if neural else 0.0) for b in BETA_GRID}


def route_d3(s_at: dict) -> str:
    """Prereg rule D3, given S of prompt lookup (gate and k chosen on even ids, scored on odd ids) at
    three betas: 'ex' (L1a's measured beta_exact), 'b2' (ex minus B2's estimated saving) and 'b3_8k'
    (the 6-8k slice under B3). A missing 'ex' means L1a has not run: report only."""
    if s_at.get("ex") is None:
        return "report over BETA_GRID; no route (L1a not run)"
    if s_at["ex"] >= 1.10:
        return "build B1 now and run E0"
    if s_at.get("b2") is not None and s_at["b2"] >= 1.10:
        return "B1 and B2 as one build"
    if s_at.get("b3_8k") is not None and s_at["b3_8k"] >= 1.20:
        return "zero-byte waits for B3"
    return "zero-byte waits for the brick"


def scheduled(rounds: list[Round], beta: float, d: float, temp: float = 1.0) -> float:
    """Confidence-scheduled K: per round choose k maximizing E[acc(k)] / (r(k) + d) from the slot
    survival estimates sigmoid(conf / temp); realized acc = min(accepted, k-1) + 1. The temperature
    is fitted on even prompt ids and this is scored on odd ones (caller's split)."""
    tokens = cost = 0.0
    for r in rounds:
        if not r.conf:
            tokens, cost = tokens + r.acc, cost + 1 + beta * (r.k - 1) + d
            continue
        p = [1 / (1 + math.exp(-c / temp)) for c in r.conf]
        best_k, best = 1, 1.0
        surv, e = 1.0, 1.0
        for j in range(len(p)):
            surv *= p[j]
            e += surv
            k = j + 2
            val = e / (1 + beta * (k - 1) + d)
            if val > best:
                best_k, best = k, val
        if best_k == 1:                     # the drafter ran for its confidences: d is paid
            tokens, cost = tokens + 1, cost + 1 + d
        else:
            tokens += min(r.accepted, best_k - 1) + 1
            cost += 1 + beta * (best_k - 1) + d
    return tokens / max(cost, 1e-9)


def corpus_tau(per_prompt: dict[str, list[Round]]) -> float:
    rs = [r for v in per_prompt.values() for r in v if r.k > 1]
    return sum(r.acc for r in rs) / max(len(rs), 1)


def paired_bootstrap_ratio(num: dict, den: dict, n: int = 10_000, seed: int = 20261102) -> tuple:
    """q_tau = corpus tau(num) / corpus tau(den), prompts resampled with replacement, paired."""
    keys = sorted(set(num) & set(den))
    rng = random.Random(seed)

    def tau_of(d, ks):
        rs = [r for k in ks for r in d[k] if r.k > 1]
        return sum(r.acc for r in rs) / max(len(rs), 1)

    point = tau_of(num, keys) / tau_of(den, keys)
    draws = sorted(tau_of(num, s) / tau_of(den, s) for s in ([rng.choice(keys) for _ in keys] for _ in range(n)))
    return point, draws[int(0.025 * n)], draws[int(0.975 * n)]


# ----------------------------------------------------------------------------------------------
# Online loop (V1 at T = 1, V2 at T = 0, thinking at T = 0.6): the vendored generate_decoding_sample
# with the DSpark evaluator's three callbacks (evaluator.py:85-147), copied, no torch.distributed.
# ----------------------------------------------------------------------------------------------
def online_dspark(target, d, input_ids: torch.Tensor, stops: list[int], temperature: float, max_new: int):
    from transformers import DynamicCache
    from deepspec.eval.base_evaluator import generate_decoding_sample
    from deepspec.eval.dspark.draft_ops import build_dspark_proposal, forward_dspark_draft_block
    from deepspec.modeling.dspark.common import extract_context_feature

    def init_context(*, initial_output, **_):
        return types.SimpleNamespace(past_key_values_draft=KeepCache(),
                                     target_hidden_states=extract_context_feature(
                                         initial_output.hidden_states, d.target_layer_ids).to(torch.bfloat16))

    def propose(*, context, output_ids, position_ids, start, stop_token_ids=None):
        block = int(d.block_size)
        di = torch.full((1, block), int(d.mask_token_id), dtype=torch.long, device=output_ids.device)
        di[:, 0] = output_ids[:, start]
        bh = forward_dspark_draft_block(d, draft_input_ids=di, position_ids=position_ids,
                                        past_key_values_draft=context.past_key_values_draft,
                                        target_hidden_states=context.target_hidden_states,
                                        start=start, block_size=block)
        return build_dspark_proposal(d, draft_input_ids=di, block_hidden=bh, block_size=block,
                                     temperature=temperature, confidence_threshold=0.0)

    def update(context, verification):
        h = extract_context_feature(verification.target_output.hidden_states, d.target_layer_ids)
        context.target_hidden_states = h[:, :verification.accepted_draft_tokens + 1, :].to(torch.bfloat16)

    return generate_decoding_sample(target_model=target, input_ids=input_ids, max_new_tokens=max_new,
                                    max_proposal_tokens=int(d.block_size), temperature=temperature,
                                    stop_token_ids=stops, init_context=init_context, propose=propose,
                                    update=update)


# ----------------------------------------------------------------------------------------------
# Driver (abridged: stage bodies call the functions above and write JSONL under --out)
# ----------------------------------------------------------------------------------------------
def banner(out: Path, vendored: dict, shims: list) -> None:
    import transformers

    info = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), torch=torch.__version__,
                transformers=transformers.__version__, device=DEVICE, threads=torch.get_num_threads(),
                nice=os.nice(0), deepspec=DEEPSPEC_SHA, vendored=vendored, shims=shims,
                sealed=(SEALED_REPO, SEALED_REV), dense=(DENSE_REPO, DENSE_REV), drafters=DRAFTERS)
    (out / "banner.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check", "pilot", "v1", "refs", "lookup", "served", "taps",
                                      "replay", "v2", "agree", "probes", "think", "report"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpora", type=Path, required=True,
                    help="JSONL: {id (int), corpus, messages, enable_thinking, source, license}")
    ap.add_argument("--l1a", type=Path, help="L1a's journal digest: {t_pf: {k: ms}} for rule D3")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)   # LLVQ_THREADS does not reach torch
    args.out.mkdir(parents=True, exist_ok=True)
    vendored = vendor_deepspec(args.out / "_vendor")
    got = hashlib.sha256(args.corpora.read_bytes()).hexdigest()
    if CORPORA_SHA256 is not None and got != CORPORA_SHA256:
        raise SystemExit(f"VOID: corpora.jsonl sha256 {got} is not the prereg's {CORPORA_SHA256}")
    shims = shim_transformers()
    shims.append(shim_cache_crop())
    banner(args.out, vendored, shims)
    ps_snapshot(args.out, args.stage)
    # check:  V3 (taps_of against forward hooks on model.model.layers[l], f32, torch.equal); V4
    #         (v4_against_checkpoint on each 'own' drafter, max |delta| == 0); mask_row_report printed.
    # pilot:  DSpark, shared-bf16mask, both targets, MT-Bench ids 0-39 and HumanEval ids 0-39: refs,
    #         taps, replay on those prompts only (reused later); d0_pilot on the two q_tau points.
    # refs:   greedy_reference per (prompt, target), f16, stops STOP_IDS; prompts.jsonl gets tokens,
    #         stop/cap, loop_onset.
    # lookup: replay_prompt_lookup on loop_truncated text, gates m in {2,3,4}, k in {2,3,4,8}; gate and
    #         k chosen on by_parity(.., 0), scored on by_parity(.., 1); route_d3 with beta_exact from
    #         --l1a at each prompt's mean position (B2: minus B2_SAVING_MS / t1; B3: B2's non-KV row).
    # served: round_trip on the run1-served and run4-f16dense dumps (sha256 pinned in the prereg);
    #         served_agreement on the sealed rebuild; the first 160 problems become math references.
    # taps + replay: per prompt, taps_of once per target, then every arm of prereg section 3 step 6
    #         (EAGLE-3 runs 'own' and 'shared' only); rounds.jsonl gets asdict(Round) plus (arm,
    #         target, head, corpus, prompt id).
    # agree:  agreement(dense, sealed, ...) at 4B in one process; at 14B cache_logits per target, one
    #         target per process, then agreement from the two caches.
    # report: corpus_tau and paired_bootstrap_ratio (q_tau) on odd ids for the drafter chosen on even
    #         ids; s_over_grid; prompt-lookup c and tau_d; loop-truncated variants; D0-D7.
    raise NotImplementedError("draft: stage bodies are described above, not written")


if __name__ == "__main__":
    sys.exit(main())
