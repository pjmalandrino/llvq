"""The teacher must be the dense twin of the student, and this checks it.

KL reads nothing but the two output distributions. Qwen3-4B and Qwen3-8B share
the vocabulary (151,936) and the depth (36 layers), so a 4B teacher bound to an
8B student produces logits of the same shape, a finite loss and a trained file.
Nothing downstream would notice. `train.sh` defaulted `TEACHER` to
`Qwen/Qwen3-4B` until 2026-09-21, which is exactly that run.

So the pairing is refused on the configs, before a single weight is loaded.
The configs are plain dicts here; reading them is the adapter's job.
"""

from __future__ import annotations

from typing import Mapping

from .loop import WiringError

ARCH_KEYS = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
)
"""What makes two checkpoints the same network. `hidden_size` is the one that
separates the 4B from the 8B: their depth and vocabulary are equal."""

_ABSENT = object()


def check_pairing(student: Mapping[str, object], teacher: Mapping[str, object]) -> None:
    """Refuse a teacher whose architecture differs from the student's.

    A key absent from both configs is skipped; absent from one side only is a
    difference, because the loader would then fill it with a default.
    """
    differ = []
    for key in ARCH_KEYS:
        s = student.get(key, _ABSENT)
        t = teacher.get(key, _ABSENT)
        if s is _ABSENT and t is _ABSENT:
            continue
        if s != t:
            shown_s = "absent" if s is _ABSENT else repr(s)
            shown_t = "absent" if t is _ABSENT else repr(t)
            differ.append(f"{key}: student {shown_s}, teacher {shown_t}")
    if differ:
        raise WiringError(
            "the teacher is not the student's dense twin; "
            + "; ".join(differ)
        )
