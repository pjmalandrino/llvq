"""Device memory, read from torch's caching allocator.

`max_memory_allocated` is what the tensors held at their peak;
`max_memory_reserved` is what the allocator took from the card, fragmentation
included, and is the figure that decides whether a larger model fits.
`total_memory` is the card's own, so a journal states its margin without a
spec sheet.
"""

from __future__ import annotations

import torch


class CudaGauge:
    def __init__(self, device: str = "cuda") -> None:
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError(f"CudaGauge reads a CUDA device, not {device!r}")

    def __call__(self) -> dict[str, int]:
        return {
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(self._device)),
            "max_memory_reserved": int(torch.cuda.max_memory_reserved(self._device)),
            "total_memory": int(
                torch.cuda.get_device_properties(self._device).total_memory
            ),
        }
