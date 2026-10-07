# Deviations from the stage 1 prereg of 2026-09-30

The prereg is `proofs/preregistration-hf-quantizer-2026-09-30.md`, sha256
`fbbf4b64317bce93`, timestamped before the first load and never edited. The journal is
`docs/mesures/hf-quantizer-4b-2026-09-30.txt`.

Two departures. Both were found while writing the code, neither after seeing a result.

## 1. The embedding has no module, it goes through a weight conversion

§3, third decision, says "each module dequantizes into a dense weight at load and frees its
code buffers". That is what the 252 projections do. The embedding cannot: replacing the
`nn.Embedding` breaks the tie to `lm_head` before a weight exists, because `tie_weights` runs
in `_finalize_model_loading`, which `from_pretrained` calls **before** the quantizer's post-load
hook. The first attempt failed there, with `model.embed_tokens.weight is neither a parameter,
buffer, nor extra state`.

So the quantized carried tensors go through `transformers`' own conversion pipeline instead
(`get_weight_conversions`, `transformers/core_model_loading.py`): the three checkpoint keys
collapse into one parameter during the load, and the model keeps a stock `nn.Embedding` with a
stock tie. The arithmetic is the same function the module path uses, and
`llvq-hf/tests/test_modules.py` holds the two paths against each other.

This is a departure in the mechanism and not in the claim: gate A covers the embedding
whichever path produced it, and it passed.

## 2. Gate B's mutant is not one of the three §6 names

§6 control 3 asks for "one mutant per gate, at least" and names three: a swapped section in the
decode, a transposed rotation, a wrong gain bit. All three are gate A mutants, and all three ran
and were caught, with a fourth, the butterfly's stages reversed.

Gate B needed a mutant of its own kind. The three named ones are all visible to gate A, which
hashes a record by name; what gate B adds is the **wiring**, and no digest can see it.

The first attempt swapped the descriptors of `gate_proj` and `up_proj` in the record table and
changed nothing: the first eight ids of all four prompts were unchanged. That is a property of
the design and not a hole in the gate. A module's buffers are named relative to its own path, so
the file's keys decide which module receives which weights, and the record table cannot misroute
them; the two descriptors are identical at every layer of the 4B, so swapping them is a no-op. A
mutant that changes nothing proves nothing, which `docs/METHODE.md` already records twice.

The mutant that counts injects the error where one can exist: after materialization,
`gate_proj`'s weight and `up_proj`'s are exchanged at every layer, so Qwen3's MLP computes
`act(up(x))·gate(x)`. Every digest of gate A still matches, every shape check still passes, and
gate B parts from the reference at token 0 of all four prompts.

## 3. No prediction missed

For the record, since the last two lots each carried one: the four signed predictions of §7 all
held. The interval on the dequantization time was the widest of them, and the exactness of gate A
was the one at risk.
