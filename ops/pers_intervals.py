"""The predicted intervals of the persistent preregs (L1, L2), derived and checked.

    python3 -I ops/pers_intervals.py

Every interval below is *estimated*; this script only checks that each one is
possible: no lower bound under the physical floor of its bytes at 864 GB/s, and
every derived interval (gains, rates, ratios) computed from its components'
bounds rather than written by hand. The review of 2026-11-02 found the drafts'
Q interval starting at 0.50, below what its own components allow, and a gain
interval that no tpersall value inside its interval could produce.

Measured inputs (journals): seg-tetra-2026-11-04 (tetra48 3.792 ms, tetra48seg
3.032 ms, 7 rounds, 2 discarded); banc-252-2026-10-06 sweep-64. Bytes are
computed from row formulas (llvq-cuda/src/pers.rs::bare_file_bytes).
"""

PEAK = 864e9  # B/s, L40S DRAM, data sheet
WORDS, F16_FAMILY, F32_FAMILY = 908_328_960, 0.9467e9, 0.9806e9


def floor_ms(nbytes):
    return nbytes / PEAK * 1e3


def check(name, lo, hi, floor):
    assert lo < hi, name
    assert lo >= floor, f"{name}: lower bound {lo} under the physical floor {floor:.3f}"
    print(f"  {name:<34} [{lo:.3f}, {hi:.3f}]  floor {floor:.3f}")


print("L1")
seg = (3.00, 3.07)  # tetra48seg, the measured 3.032 ms widened by the round-to-round spread seen in A4
tpa = (1.55, 2.60)  # tpersall, ms
check("tpersall ms", *tpa, floor_ms(F32_FAMILY))
gain = (100 * (1 - tpa[1] / seg[0]), 100 * (1 - tpa[0] / seg[1]))
print(f"  one-launch gain over tetra48seg %   [{gain[0]:.0f}, {gain[1]:.0f}]  (derived)")
bpa = (F32_FAMILY / tpa[1] / 1e6, F32_FAMILY / tpa[0] / 1e6)
print(f"  B_pa GB/s                          [{bpa[0]:.0f}, {bpa[1]:.0f}]  (derived)")
assert bpa[1] <= PEAK / 1e9

print("L2 (re-anchored 2026-11-05 on L1: tetra48 3.788 ms, tetra48seg 3.009, barrier 0.786 us; and on the LSU model)")
ref = (3.70, 3.90)  # tetra48h: the bench unit's tetra48 measured 3.788 [3.742-3.807] in L1, the same text
check("tetra48h ms", *ref, floor_ms(F16_FAMILY))
check("pword ms", 1.10, 1.75, floor_ms(WORDS))
check("pdec_l1 ms", 1.35, 1.95, floor_ms(WORDS))
check("pdec_smem ms", 1.10, 1.80, floor_ms(WORDS))
pl1, pl1_t28 = (2.05, 2.80), (1.75, 2.50)
check("pl1 ms", *pl1, floor_ms(F16_FAMILY))
check("pl1_t28 ms", *pl1_t28, floor_ms(F16_FAMILY))
check("psmem ms", 1.15, 2.10, floor_ms(F16_FAMILY))
dec = (1.12, 2.00)  # psmem_t28, and psmem_s25
check("psmem_t28 ms", *dec, floor_ms(F16_FAMILY))
h = (0.8e-3, 3.0e-3)  # h_exp, ms: the barrier alone measured 0.786 us at 142 x 768
n_sync = 145
chain_w = (dec[0] + n_sync * h[0], dec[1] + n_sync * h[1])
check("pchain_w ms (derived)", *chain_w, floor_ms(F16_FAMILY))
prol = (0.08, 0.35)
chain_wp = (chain_w[0] + prol[0], chain_w[1] + prol[1])
check("pchain_wp ms (derived)", *chain_wp, floor_ms(F16_FAMILY))
fus = (3.009 / 3.788 * 0.99, 3.009 / 3.788 * 1.01)  # tetra48seg / tetra48, measured in L1, +-1 %
q = (chain_wp[0] / ref[1] / fus[1], chain_wp[1] / ref[0] / fus[0])
print(f"  Q (derived)                        [{q[0]:.2f}, {q[1]:.2f}]")
assert q[0] > 0.40
b_t = (F16_FAMILY / dec[1] / 1e6, F16_FAMILY / dec[0] / 1e6)
print(f"  B_T of psmem_t28 GB/s              [{b_t[0]:.0f}, {b_t[1]:.0f}]  (derived)")
