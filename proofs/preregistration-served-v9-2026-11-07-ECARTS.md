# Deviations from `preregistration-served-v9-2026-11-07.md`

Written on 2026-11-07, after the stamp and before any measurement. The prereg is not edited.

## 1. The 4E v9 module is not step3-4e's

**What changed.** The operator asked the same day for v9 at 8B and 14B too ("Tu me fait 4,8 et 14 en V9 et tu envoie les
tests point barre"). 4E gained two things those sizes need: the head read through pointers of its own (three
pointers appended to `Pers4eArgs`, 216 to 240 bytes), and a Tetra o phase named by a pointer appended to
`Pers4eLayer` (592 to 600 bytes). On the 4B the head pointers are the embedding's and every o is int4, so the
arithmetic of every row is the module step3-4e timed; the text is not.

**Effect on §3.** Gate 4's install line names module sha256 `1b422b62…` (252,757 B), pinned by
`the_4e_v9_module_is_the_one_the_doors_gate_on`, in place of `d9da09aa…`. The job also gates on the line
`4E: head tied to the embedding's table (151936 rows)`. The host proof that the new module keeps the served chain's
bits on the 4B shape is the unchanged `the_4e_v9_int4_tiles_are_the_served_chain` and its v4 to v8 siblings, green on
the new text; the 8B shape is `the_4e_tetra_o_and_untied_head_are_the_served_chain`.

**Effect on §5.** None expected: the timed path of the 4B moves by one uniform branch on a null pointer in the o
phase and in its prefetch. The predictions stand as stamped.

## 2. One image for three sizes

The image is rebuilt once, from the commit that carries the 8B and 14B work, and the 4B door runs on it. The prereg's
base commit is therefore a later commit of the same branch.
