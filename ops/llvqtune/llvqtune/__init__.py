"""Trains the free parameters an LLVQ artifact already holds.

The layering is ports and adapters. `domain` knows no framework and no file
format; `ports` states what the domain needs; `adapters` supplies it; and
`trainables` is the point of variation, one module per row of
docs/ROADMAP-QUALITY.md.
"""

__all__ = ["domain", "ports", "adapters", "trainables"]
