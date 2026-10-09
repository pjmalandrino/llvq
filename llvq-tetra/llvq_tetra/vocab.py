"""The 4-bit vocabulary kept 4-bit on the device: the input rows, and the tied output layer.

The file stores the embedding as `qweight`, `scales` and `biases` (int4 in groups of 64, `w =
scale·q + bias`). The dense path dequantizes it at load, 1.56 GB at f32 for Qwen3-4B. Here it stays
as stored, 0.22 GB, and two served shaders read it: `emb_q4_gather_metal` turns token ids into
rows, and `tv_emb_q4_metal`, in an f32-input variant, computes the logits from the same table.

The gathered rows are the dense path's rows bit for bit: the same `scale·q + bias` in f32, whose
product is exact. The logits sum in the kernel's order, not torch's, and gate G-B2 bounds that.
"""

from __future__ import annotations

import torch
from torch import nn

from . import mps


class Int4Vocab(nn.Module):
    """The table, under the names the file gives its three tensors, so the loader fills it."""

    def __init__(self, desc: dict):
        super().__init__()
        rows, d = desc["dims"]
        self.desc = desc
        self.num_embeddings, self.embedding_dim = rows, d
        self.gpr = desc["groups_per_row"]
        self.out_dtype = torch.float32
        self.register_buffer("qweight", torch.empty(rows, d // 2, dtype=torch.uint8))
        self.register_buffer("scales", torch.empty(rows, self.gpr, dtype=torch.float16))
        self.register_buffer("biases", torch.empty(rows, self.gpr, dtype=torch.float16))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        y = mps.vocab_gather(self.qweight, self.scales, self.biases, input_ids, self.embedding_dim, self.gpr)
        return y.to(self.out_dtype)

    def resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.qweight, self.scales, self.biases))

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}, int4 g{self.desc['group']}"


class Int4TiedHead(nn.Module):
    """The output layer of a model whose output and input embeddings are one table.

    It holds a reference to the table and not the table: registered as a submodule, the same
    buffers would appear twice in the state dict, and the loader would look for them twice.
    """

    def __init__(self, vocab: Int4Vocab):
        super().__init__()
        object.__setattr__(self, "_vocab", vocab)
        self.in_features, self.out_features = vocab.embedding_dim, vocab.num_embeddings

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        v = self._vocab
        flat = hidden.reshape(-1, self.in_features).to(torch.float32)
        y = mps.vocab_logits(v.qweight, v.scales, v.biases, flat, self.out_features, self.in_features, v.gpr)
        return y.reshape(*hidden.shape[:-1], self.out_features).to(hidden.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, tied to the 4-bit vocabulary"
