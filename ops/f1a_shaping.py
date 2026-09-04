# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""F1a: exact shaping retention of a three-section E8 region, in closed form.

F1 codes the Leech lattice as a three-section E8 coset code (`docs/ROADMAP.md` §2.2).
Its shaping region is bounded section by section, so if nothing couples the three
sections the region is a *product of three 8-dimensional balls* instead of the single
24-dimensional ball the served `leech1c12` format uses. That substitution costs shaping
gain, and the F1b gate is written on retention, so the loss is computable before a line
of codebook exists. This script is the accounting behind the *computed* numbers in
`docs/ETAT.md` §5 bis; it reads no file and takes no argument.

Definitions, both taken from the repository and not reinvented here:

  * normalized second moment of an n-ball, G(n) = 1 / ((n+2) * V_n^(2/n)), the standard
    quantity whose ratio to the cube's 1/12 is the sphere shaping gain;
  * retention, `retention_pct(mse, rate) = 100 * (-0.5 * log2 mse) / rate`, the definition
    pinned in `docs/fiche-4b.md` and used by the G4 benchmark of `llvq-bench`.

The anchor MSE is the served shape-gain one, 0.077718 unrounded (`docs/fiche-4b.md`,
92.14% at 2.000 b/dim). The rate is 2.000 b/dim on both sides: an F1 word is 48 bits for
24 dimensions, exactly like `leech1c12` + 1 gain bit. Dividing by the 47 bits of the
lattice-point field alone would flatter F1 by 1.9 pp of retention and repeat the error of
2026-08-04, when a 92.24% retention was divided by a fractional rate no file pays
(`docs/HISTORIQUE.md`).
"""

import math

MSE_SERVED = 0.077718  # served shape-gain, unrounded (docs/fiche-4b.md)
RATE = 2.0  # b/dim: 48 bits per 24-dimensional block, both formats
KILL, ADOPT = 90.3, 91.0  # F1b gate (docs/ROADMAP.md §2.2)


def ball_nsm(n: int) -> float:
    """Normalized second moment of the n-dimensional ball."""
    ln_volume = (n / 2.0) * math.log(math.pi) - math.lgamma(n / 2.0 + 1.0)
    return 1.0 / ((n + 2.0) * math.exp(2.0 * ln_volume / n))


def shaping_gain_db(n: int) -> float:
    """Shaping gain of the n-ball over the cube, in dB."""
    return 10.0 * math.log10((1.0 / 12.0) / ball_nsm(n))


def retention_pct(mse: float, rate: float = RATE) -> float:
    return 100.0 * (-0.5 * math.log2(mse)) / rate


def mse_for_retention(pct: float, rate: float = RATE) -> float:
    """Inverse of retention_pct: the largest MSE that still reaches `pct`."""
    return 2.0 ** (-2.0 * (pct / 100.0) * rate)


def main() -> None:
    g8, g24 = shaping_gain_db(8), shaping_gain_db(24)
    loss_db = g24 - g8
    mse_product = MSE_SERVED * 10.0 ** (loss_db / 10.0)

    print(f"shaping gain, 8-ball          {g8:.4f} dB")
    print(f"shaping gain, 24-ball         {g24:.4f} dB")
    print(f"loss of the product form      {loss_db:.4f} dB  (MSE x{10**(loss_db/10):.4f},"
          f" +{100*(10**(loss_db/10)-1):.2f}%)")
    print()
    print(f"served, 24-ball               MSE {MSE_SERVED:.6f}"
          f"  retention {retention_pct(MSE_SERVED):.2f}%")
    print(f"F1 as a product of three      MSE {mse_product:.6f}"
          f"  retention {retention_pct(mse_product):.2f}%")
    print()

    # How much the state coupling must buy back for F1 to survive its own gate.
    for label, target in (("kill", KILL), ("adopt", ADOPT)):
        budget_db = 10.0 * math.log10(mse_for_retention(target) / MSE_SERVED)
        must_reach = g24 - budget_db
        share = (must_reach - g8) / loss_db
        print(f"{label:<5} {target:.1f}%  -> MSE <= {mse_for_retention(target):.6f}"
              f", loss budget {budget_db:.4f} dB")
        print(f"            region must reach {must_reach:.4f} dB"
              f" = {100*share:.1f}% of the product-to-ball gap")


if __name__ == "__main__":
    main()
