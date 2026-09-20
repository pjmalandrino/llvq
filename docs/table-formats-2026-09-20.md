# The five-row table: f16, AWQ, QTIP, IQ2_XXS, Tetra

Qwen3-4B throughout. Every cell carries its accounting and its journal. Cells marked **TBD**
are not measured and are listed with their cost in section 4.

## 1. The table

| | cold storage | VRAM read a pass | kernel median | ppl | MMLU |
|---|---|---|---|---|---|
| **f16** | 8.04 GB · 16.000 | 7.27 GB · 16.000 b/w | 11.109 ms · 654 GB/s | **12.2361** | **70.14** |
| **AWQ w4 g128** | 2.67 GB · 5.302 | 1.90 GB · 4.179 b/w | 3.249 ms · 585 GB/s | **13.5207** | 70.04 ⚠ |
| **QTIP 3INST** | TBD | 0.91 GB · 2.000 b/w | 2.246 ms · 405 GB/s 🕐 | 17.04 📄 | 57.4 📄 |
| **IQ2_XXS** | 1.26 GB · 2.4967 | TBD | TBD | TBD | 38.87 ⚠ |
| **Tetra, fine-tuned** | **1.79 GB · 2.7475** | **0.95 GB · 2.148 b/w** | **4.107 ms · 232 GB/s** | **12.3268** | **61.11** |

⚠ MMLU on the 2,280-question plan, not the 14,042 census. 📄 the paper's number, read in the
paper's harness, not ours. 🕐 measured 2026-08-21 in ANOTHER process; every other kernel cell
comes from the 2026-09-20 run, and `docs/data/README.md` forbids putting rounds from two
processes side by side. The common baseline does reproduce: Planes14 read 5.133 ms in August
and 5.135 in September, 0.04 % apart.

## 2. What each column means, and the traps in it

**Cold storage** is b/param over the whole model, embedding included, hard rule 6. Tetra's
1.79 GB is the sealed file's own byte count; its 2.7475 is the f16-tail accounting the card has
held since 2026-08-09, not the f32 accounting older figures print (2.8126).

**VRAM read a pass** is `gb_read` from the ten-arm bench, and `b/w` beside it is
`bpw_kernel`. This is the column the project exists for: `Planes14` stores 2 bits and reads
4.804, because its decoder is bought by expanding the representation before inference. Tetra
reads what it stores.

**Kernel median** is `med_ms` from the same bench, one process, one L40S. It is a projection
matvec, not a token. The end-to-end figures live in another unit and another stack: AWQ reads
200.49 tok/s against f16's 83.09 in vLLM, and hard rule 5 forbids dividing a ratio across
stacks, so those stay a note and never a column.

**ppl** is wikitext-2, 4096 context, 12 windows, f16, token fingerprint `3f1baca9033bf251`,
the protocol of every row of `docs/fiche-4b.md` section 5. QTIP's 17.04 is NOT in it: it comes
from the paper's Table 6 and its own pipeline.

**MMLU** is micro, stratified by subject population, the paper's own protocol. Two plans coexist
and they do not agree: bare Tetra reads 53.49 on the 2,280 plan and 54.64 on the 14,042 census,
a gap of +1.15 pp. Cells on different plans must not be subtracted.

## 3. Sources

| cell | journal |
|---|---|
| ten-arm bench, all `ms`/`GB`/`GB/s` | `docs/data/echelle-formats.csv`, rewritten from banc-tetra-2026-09-20; the QTIP row alone is from f2-p3-qtip-banc-2026-08-21 |
| f16 ppl 12.2361 · Planes14 16.9415 | `docs/fiche-4b.md` section 5 |
| AWQ ppl 13.5207, x1.105 | same |
| AWQ cold storage 2.67 GB, 5.302 | `rtbits-planes-8b-2026-08-09` |
| f16 MMLU 70.14 census | `f16-full-2026-09-18` |
| AWQ MMLU 70.04, f16 70.32, 2,280 plan | `mecanismes-perte-qualite-2026-09-12` |
| IQ2_XXS 1.26 GB, 2.4967, MMLU 38.87 | `iq2m-2026-09-19` |
| QTIP ppl 17.04 and MMLU 57.4 | `llvq-paper-notes` Table 6 |
| Tetra cold storage, ppl, MMLU | `dclm-rowscales-2026-09-20` |

## 4. What is missing, and what it costs

| # | cell | cost | why it matters |
|---|---|---|---|
| ~~1~~ | ~~Tetra kernel median~~ | ~~done~~ | **4.107 ms, 0.95 GB, 232 GB/s** ([banc-tetra](mesures/banc-tetra-2026-09-20.txt)) |
| 2 | QTIP in the SAME process as Tetra | ~$0.30 plus a flag | `arms.rs` has `HAS_KERNEL[qtip] = false` and refuses the name. Without this the QTIP row comes from another process, which `docs/data/README.md` forbids |
| 3 | IQ2_XXS ppl, VRAM, kernel median | ~$0.72 | three empty cells on the most widely deployed 2-bit format |
| 4 | QTIP cold storage | $0 to compute if a file exists | no QTIP Qwen3-4B is published; the paper's authors quantized it themselves |
| 5 | AWQ and IQ2 MMLU on the census | $0.72 each | to remove the two-plan footnote |

## 4 bis. The column that is NOT tokens a second

`med_ms` times 252 projection matvecs for one token. It is not a token: 48 % of a
token is outside the matmuls. Dividing 1000 by it gives 243.5 for Tetra, and the
measured end-to-end figure is **101.5 tok/s** (F1e section 0, in the model, L40S,
1.39 GB). The bench's 1.25x against Planes14 becomes **1.15x** end to end,
88.5 to 101.5 tok/s.

No end-to-end figure exists for QTIP. None has ever run in a model here.

And the floor: `nullk` reads no weight and takes 2.340 ms. QTIP's 2.246 ms sits
**below it**, which the August journal flags with a permanent reservation. A
speed verdict cannot rest on it unexplained.

## 5. The one thing the table does not say

Every quality cell here is a **dense reconstruction**, the protocol behind every published bar
in this repository.

**This file now also runs on a card**, through the served kernel, against its own dense arm in
the same process: **98.3 tok/s in 1.39 GB against 43.4 in 8.04**, x2.27 and /5.78, with 32
tokens identical and `oracle` MATCH
([dclm-ft-fusedrun](mesures/dclm-ft-fusedrun-2026-09-20.txt)). The object that scores 61.11 is
the object that serves.

Two reservations kept rather than rounded away. The check covered **32 tokens**, where F1e
section 0 covered 256 on the 2026-09-10 object; the 256-token pass waits for the authorized
throughput work and is done once, on the improved object. And that run reads 2.140 b/weight on
the card against the bench's 2.148, because the bench bills the tail in f32 while the card
holds it in binary16 — part of the 3 % the bench prereg left open.
