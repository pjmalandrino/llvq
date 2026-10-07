"""The carried tensors the file stores group-affine, converted while loading.

A quantized projection becomes a module that holds its own buffers, because the
module is what a kernel will later replace (stages 2 to 4). A carried tensor has
no module of its own: the embedding is an `nn.Embedding` whose `weight` the head
is tied to, and replacing it would break that tie before the weights exist.

So this arm goes through `transformers`' own conversion pipeline
(`get_weight_conversions`, `transformers/core_model_loading.py`): three checkpoint
keys collapse into one parameter during the load, and the model stays a stock
Qwen3 with a stock embedding.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers.core_model_loading import ConversionOps

from .dequant import dequantize_int4


class Int4Dequantize(ConversionOps):
    """`qweight`, `scales`, `biases` in, one dense tensor out.

    The arithmetic is `dequantize_int4`'s, so this path and the module path give
    the same values: `w = scale·q + bias`, in f32, nibbles low first.
    """

    def __init__(self, desc: dict, dtype: torch.dtype | None = None):
        self.desc = desc
        self.dtype = dtype

    def _one(self, input_dict: dict, pattern: str) -> np.ndarray:
        got = input_dict[pattern]
        t = got[0] if isinstance(got, (list, tuple)) else got
        return t.detach().cpu().numpy()

    @torch.no_grad()
    def convert(self, input_dict: dict, source_patterns: list[str], target_patterns: list[str],
                **kwargs) -> dict:
        d = self.desc
        q, s, b = (self._one(input_dict, p) for p in source_patterns)
        w = dequantize_int4(q, s, b, d["rows"], d["dims"][-1], d["group"])
        t = torch.from_numpy(w.reshape(tuple(d["dims"])))
        if self.dtype is not None:
            t = t.to(self.dtype)
        return {target_patterns[0]: t}

    @property
    def reverse_op(self):
        raise NotImplementedError("this format is written by `hfpack`, not by save_pretrained")
