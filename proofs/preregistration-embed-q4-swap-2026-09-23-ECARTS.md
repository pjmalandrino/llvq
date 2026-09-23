# Deviations from the q4-embedding swap preregistration

> The preregistration
> [`preregistration-embed-q4-swap-2026-09-23.md`](preregistration-embed-q4-swap-2026-09-23.md)
> (sha256 `c45f1a1a5124b177…`) is timestamped. It is not edited.

## É1. The package costs 2.7320 b/param, not 2.7138

**What the prereg computed** (§1): 2.7138 b/param and 2.5953 kernel b/weight
for the package, a file 16.9 MB lighter than the 61.11 object.

**What the sealed file measures.** `rtbits` over the bytes of
`qwen3-4b-sealed.bin` (1,418,224,685 B, sha256 `886391a8c03f66dc…`), the file
`int4swap` and `embedq q4` wrote from the 61.11 object on 2026-09-23 (*computed*,
`docs/mesures/embed-q4-swap-2026-09-23.txt`):

| accounting | 61.11 object | sealed package | delta |
|---|---|---|---|
| kernel b/weight, tail f16 (served) | 2.1309 | 2.5420 | +0.4111 |
| kernel b/weight, tail f32 | 2.2030 | 2.6065 | +0.4035 |
| b/param, tail f16, embedding q8 | 2.7475 | 3.1188 | |
| b/param, tail f16, embedding as served | 2.7475 (q8) | **2.7320** (q4) | −0.0155 |

The package is 7.8 MB lighter than the 61.11 object, not 16.9.

**The cause.** The prereg priced each of the 676,331,520 replaced weights at
Tetra's 2.1498 b/weight, so an int4 g128 weight added 2.1002 bits. In the file,
the 48 replaced lattice records cost less than 2.1498: the stream reads 1.99
b/weight, plus their share of tail and row scales. Each replaced weight
therefore adds about 2.21 bits, which is the 0.02 kernel b/weight between the
two deltas. The prereg also mixed two accountings: its 2.2044 is the f32-tail
kernel figure, its 2.7475 the f16-tail b/param.

**What it changes.** Nothing in §4. The rule reads the paired MMLU gain, and
the package stays under the 61.11 object in b/param and 0.458 kernel b/weight
under the triplet's b_max of 3.00. The sealed file's figure replaces the
prereg's in every living document.
