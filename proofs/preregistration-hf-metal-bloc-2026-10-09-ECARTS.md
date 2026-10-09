# Deviations from `preregistration-hf-metal-bloc-2026-10-09.md`

The prereg is timestamped (`.ots` beside it, sha256 `be12d562`) and is not edited. Its deviations are written here,
each before the measurement it changes.

## É1. The rows kernels are new code around served helpers

Section 3 says each variant is made by a textual change to a served kernel, never by a second hand-written copy of
its arithmetic. The two rows kernels of arm D break that in part. `tetra48_dot_q`, the per-block dot, is cut from the
served `tetra48_dot` by text edits, so the fma chain and the scaling are the served text. But the loops over rows, the
tile loads, the epilogue of the Tetra rows kernel (the tail `fma` and the final `fma`) and the inner loop of the 4-bit
rows kernel are written by hand, after `tv_tetra48_metal` and `tv_q4_metal_tiled`. What holds them to the served
arithmetic is gate G-D, bit for bit against the single-row kernels on every record, and its mutant.

## É2. G-C's reference is rebuilt from the kernel's own tables

The torch rotation G-C compares with is built from the sign bits and the padded mix the kernel object holds, not from
the f64 tables of the checkpoint. A defect in building those two would therefore move the reference with the subject,
and G-C could not see it. G-T sees it: a wrong sign or a wrong mix changes the tokens.

## É3. G-T failed on its first run, the arm was fixed, and G-T runs again

Measured on 2026-10-09: the 4B on `mps` loaded with no missing or unexpected key, in 2.9 s, with 1.393 GB allocated,
and gave **0 prompts of 4 identical**, the first difference at token 3, 3, 3 and 4.

The cause, found by a diagnosis run that turned the reuse off and got 256 ids of 256: the rotation kept for the next
caller was recognized by the storage address of its input, its version and its shape. The MPS allocator hands the
same address to the next token's activation, at version 0 and of the same shape, so a new activation received the
rotation of an old one. The address identifies memory, not a tensor.

The decision rule of section 7 says a failed gate stops its arm. This deviation keeps arm C instead, with the reuse
rebuilt: an input is recognized by the tensor object itself, through a weak reference, and its version counter, and
the kept rotation is dropped once every projection that shares it has read it. G-T runs again on the fixed arm, and
its mutant after it. Both runs, the failed one and the next, go into the journal.

## É4. The fp16 arm crashed when loaded onto the device, and the measurement runs again

Measured on 2026-10-09, first run of section 6, 18:26 to 18:34: v0 and v1 completed five rounds each, and the fp16
arm died in all five, exit 139, a segmentation fault in the first second of `from_pretrained(..., device_map="mps")`,
at "Loading weights 0/398". `faulthandler` could not print a stack before the process died. Loaded on the CPU and moved
with `.to("mps")`, the same checkpoint loads, 8.112 GB on the device in 4.1 s.

Change: the fp16 arm loads on the CPU and moves to `mps`, which is what the two LLVQ arms do internally, since their
quantizer loads on the CPU and moves the model at the end. Prefill and decode are timed on the device as before. The
whole section 6 runs a second time, all three arms, five rounds, the order rotated, so that fp16 is interleaved like
the others. The second run is the one the verdict reads. The first run's ten rows of v0 and v1 are kept in the journal
beside it.
