# Deviations from the M2 prereg of 2026-09-30

The prereg is stamped and is not edited. Every departure from it is written here, with its
reason. Journal: `docs/mesures/hf-metal-m2-4b-2026-09-30.txt`.

## 1. "A tile of 128" cannot be a mutant, it is a refusal

**What the prereg says.** Section 5, control 4: "Mutants, at least: the group index shifted by
one, the nibble order reversed, **a tile of 128**."

**Why it does not work.** A tile of 128 is not a wrong kernel, it is an illegal call, and the
op refuses it by name before any dispatch: "tile_cols 128 is not a multiple of 256". So it
cannot survive a gate, and it cannot be caught by one either. Writing it as a mutant was a
category error in the prereg.

**What was done instead.** A tile of 128 became a refusal probe under control 1, together with
a tile of 300, a `d_in` of 2,555 and a tile of 16,384, and the third mutant is **the lane
stride moved off 32** (`wi += 32u` to `wi += 31u`). That mutant attacks exactly the property
section 3 claims, the interleaving of the lanes, so it is the one the argument deserves. It was
caught at all three tiles.

## 2. The controls were discharged after the token run, not before it

The bit-identity gate and the 256 ids were taken first, then `llvqhf/checkq4guards.py` was
written and the refusals and mutants run. The order in the prereg reads the other way. Nothing
in the measurement changed, but the sequence is what it is and is recorded.

## 3. Control 3 is bit-equality on two shapes out of three, not a residue on each

**What the prereg says.** Control 3: "Per-row agreement with the dense reconstruction, one
matrix of each int4 shape, reported as a number and gated at 1e-2 relative."

**What was done.** For the two shapes the served kernel can take, the comparison is equality of
every f32 against that kernel, which is strictly stronger than a 1e-2 residue and leaves no
number to report. The residue is reported for the third shape, `down_proj` 2560 by 9,728, which
is past the served wall and has no reference but the dense reconstruction: **2.90e-07**, against
a gate of 1e-2.

## 4. One mutant was nearly void, and the fix is in the script

`tv_q4_metal_tiled` comes first in the shader and the served `tv_q4_metal` after it, so a
replacement "inside the tiled kernel" obtained by partitioning on its name lands on both
functions. The reference would have moved with the subject and the three mutants would have
read as survived while testing nothing. The site count assertion fired at 2 sites instead of 1.
`mutate()` now closes the region at the next entry point and asserts the served kernel is not
inside it. Not a deviation from the plan, but the kind of thing this file exists to carry.
