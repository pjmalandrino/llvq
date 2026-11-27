# Deviations from the prereg of the grouped attention on the CPU (2026-11-27)

The prereg `preregistration-qwen-cpu-groupee-2026-11-27.md` is timestamped and not edited. What this lot did
differently from it is written here, before or as each part ran.

## E1. Two measurement modes added to `cpurun`

The CPU load already honours the `attention` field. `cpurun`'s `load_tetra` reads `LLVQ_CONFIG`, loads through
`served::load_resolved` on `Device::Cpu` and calls `Served::apply`, whose `check_device(false)` accepts every form
off Metal and whose `apply_form` sets the store (`Grow(256)` under `grouped`, `Cat` under `expanded`) and the form
on every block. The served `configs/qwen3-4b-tetra-e4-grouped.json` therefore loads on the CPU as it is, and no
`-NOT-SERVED` config was written.

`cpurun` had no lockstep and no long prompt: its `tetra` mode decodes four fixed prompts of 5 to 18 tokens. Two
modes were added, and nothing else in the crate changed:

- `timed <short|N> [n_new]`: one prompt, `short` the first of the four (5 tokens), N the first N tokens of
  WikiText-2 test; prints the time to the first token, the decode rate and the ids, and dumps them as JSON.
- `lockstep <config_b> [len] [window]`: the file loaded twice through the same CPU door, under `LLVQ_CONFIG` and
  under `config_b`, one window fed token by token to both, each on its own caches. Every step prints max
  |dlogit|, both argmaxes, the largest |logit| of each arm and whether the logits are equal in every bit.

Both load paths print the form and the store the loaded model reports (`loaded form:`), not the config's.

## E2. The lockstep runs two loaded models, not one model switched

`Qwen3::decode_nll_paired`, which `metalrun` used for M3 and M5, switches one model's store and form between the
two arms at every step. That bypasses the load: it would measure the grouped arithmetic, not what the CPU load
of the grouped config produces. The lockstep of E1 loads each arm through `Served::apply`, at the cost of a
second copy of the model in memory (about 1.9 GB).

## E3. The mutant sits in `Served::apply_form`, shared by every device

The prereg's mutant is "the measurement config's `attention` field ignored on the CPU". The CPU load has no
CPU-only site that reads the field: `Served::apply_form` is the one place every door sets the form. The mutant
replaces `model.set_attn_heads(self.attention)` by `model.set_attn_heads(AttnHeads::Expanded)` there, and the
test that reads it is `model::tests::a_served_config_sets_the_store_and_the_attention_form`, which applies a
grouped config on `Device::Cpu` and asserts the form the model reports.
