"""The one concession the domain makes to the outside world.

`Tensor` is whatever the bound adapter uses. The domain never calls a method
on it. It passes tensors between ports and reads scalars back through
`Objective.value`. Keeping the alias here makes that boundary visible instead
of scattering `Any` through every signature.
"""

from __future__ import annotations

from typing import Any, TypeAlias

Tensor: TypeAlias = Any
