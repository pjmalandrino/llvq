"""Objectives, in torch. Both read the model's output and never a layer residual."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class KLDistillation:
    """KL(teacher || student) on the next-token distribution.

    The reference is the dense model. This is the objective that matches what
    the format is for: the quantized model should answer like the dense one,
    not merely score well on text.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._t = temperature

    @property
    def name(self) -> str:
        return f"kl_t{self._t:g}"

    @property
    def needs_teacher(self) -> bool:
        return True

    def __call__(self, student, teacher, targets=None):
        if teacher is None:
            raise ValueError("kl needs reference logits")
        t = self._t
        log_p = F.log_softmax(teacher.float() / t, dim=-1)
        log_q = F.log_softmax(student.float() / t, dim=-1)
        kl = torch.sum(log_p.exp() * (log_p - log_q), dim=-1)
        return kl.mean() * (t * t)

    def value(self, loss) -> float:
        return float(loss.detach())


class CrossEntropy:
    """Next-token cross entropy on the corpus. The control arm for KL."""

    @property
    def name(self) -> str:
        return "cross_entropy"

    @property
    def needs_teacher(self) -> bool:
        return False

    def __call__(self, student, teacher=None, targets=None):
        if targets is None:
            raise ValueError("cross entropy needs token ids")
        logits = student[:, :-1, :].float()
        labels = targets[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )

    def value(self, loss) -> float:
        return float(loss.detach())
