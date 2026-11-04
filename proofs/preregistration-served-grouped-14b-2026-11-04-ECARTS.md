# Deviations of proofs/preregistration-served-grouped-14b-2026-11-04.md

## E1, written before the launch: the image is 9c5e935d, not ef658736

Between the stamp and the launch the Space was republished by the attention-kernel session from its branch
`claude/quizzical-kowalevski-38c5fc` (commit `d269f6b`), Space sha `9c5e935dba6e7aeb2df44b4dac250d0c9a96869a`, for its
own bench, which is running and asked that the Space not be republished until it ends. This run uses that image.

What that image changes against the base of this branch (`030747b`), read with `git diff 030747b d269f6b`: it adds a
third attention form, `AttnHeads::Kernel`, a `Block.attn_kernel` field that is `None` unless the config says
`"kernel"`, a device trait, and the kernel files. The `Expanded` and `Grouped` dispatch in `Block::forward` and the
stores they use are unchanged: `Grouped` still calls `grouped_attention`, `Expanded` still expands. Neither config of
this run names `"kernel"`. The MMLU comparison pairs the two configs inside one job on one image, so its rule reads the
same quantity. The served-door speeds are reported against this image's sha, never against ef658736's.
