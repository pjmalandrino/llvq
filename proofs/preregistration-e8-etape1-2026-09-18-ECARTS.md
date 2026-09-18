# Deviations. E8 stage 1 prereg of 2026-09-18

The prereg is stamped (`7b75b465cf8be967be747f45983d480ccce0d470be74b69109d8d14be9114c54`,
four calendars) and is not edited. This document reads beside it.

**The primary question is void, and E3 is why.** The two numbers the run produced are real and
are reported in `docs/mesures/e8-etape1-2026-09-18.txt`, but neither isolates the lattice.

## E1. Control 3's premise was wrong

The prereg's control 3 asks that arm A's error, recomputed from the dump's `witness`, agree
with the `shadow` entries the replay recorded. Those are two different objects. `witness` is
the encoding of the **original** row; `shadow` describes the **compensated** blocks. They
coincide only at block 0, where nothing has been compensated yet, and the manual probe that
suggested the control would work had looked at exactly that block.

Replaced by two controls that hold: `|witness|` against `centroid * row_scale` block by block,
worst gap 1.110e-16, and block 0 against the replay's `euclidean` at the chosen level, worst
gap 3.469e-18. Both pass.

## E2. The prediction's referent was not arm A

The registered 1.09 is the ratio of the normalized second moments of E8 and Leech. Arm A is
not Leech's Voronoi quantizer: `Tetra` is a structured subset of the lattice searched by a
three-section trellis over two parities, chosen to be decodable in a kernel. So the prereg
predicted a quantity the experiment does not measure, and the measured 0.7155 is not evidence
against the second-moment bound.

## E3. Arm A is a GPTQ witness and arm B is a plain encode

This is the deviation that voids the primary result. `diagnose_row`
(`llvq-quant/src/schur.rs:195`) takes a `GptqFactor` and carries a `working` vector: the
witness is a **sequential encode with error feedback through the Schur factor of the cell's
Hessian**. The module header says so at `llvq-llm/src/tetra_diag.rs:4`.

Arm B, as specified in the prereg's section 2, is a plain per-block nearest-neighbour encode
with no compensation. The arms therefore differ in the codebook **and** in whether they know
H, and the primary ratio measures the second difference at least as much as the first.

That is the same confound this dossier criticized in the paper's own E8 row. It was
reproduced here, in the prereg, and it was found by the isotropic control of E6 rather than by
reading the protocol.

No claim is made about E8 against Leech. Stage 1 needs arm B inside the same GPTQ loop, which
is a new prereg.

## E4. The rate-matched arm went below Tetra, not above

The prereg's decision rule for a ratio under 1.00 asks for a re-run at a cap between shells 10
and 12. A cap there requires a partial shell, hence an arbitrary ordering rule inside it. The
norm-8 cap was run instead: 26,640 points, a 15-bit index, 45 bits plus one gain bit,
**1.9167 b/weight against Tetra's measured 1.9907**. Arm B therefore holds a 3.7 % bit
*disadvantage* rather than a 2.6 % advantage, which is strictly more conservative than what
the prereg asked for. Both rates are reported.

## E5. Control 2's tolerance was absolute and absurd

It was written as `|err| < 1e-24` on a quantity of order `picked^2 = 2.5e-3`, which is 21
orders of magnitude below machine epsilon. It passed at the norm-10 cap by arithmetic luck and
failed at norm 8 with a residue of 8.7e-19, that is 3.5e-16 relative. Changed to
`1e-12 * picked^2`. The prereg states the control without a tolerance, so this fixes the
script and not the protocol.

## E6. One control was added: the isotropic reference

Not in the prereg, and it is what found E3. For a residual of the same energy in a random
direction, `E[e'He] = |e|^2 tr(H)/n` exactly. Comparing each arm's `J_local` to its own
isotropic value separates *how much* error an arm makes from *where* it puts it. Arm A sits at
0.148 of isotropic and arm B at 0.996, which is the signature of compensation and not of a
codebook.
