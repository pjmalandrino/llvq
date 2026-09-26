#!/usr/bin/env python3
"""Small, deterministic checks used in the 2026-09-08 literature dossier.

These are algebra checks, not model benchmarks.  They deliberately use exact
rational arithmetic so that a paper's equivalence claim is tested without
floating-point tie-breaking or a GPU dependency.
"""

from fractions import Fraction as F
import json
import math
import random
from pathlib import Path


def mat(m, n, fill=F(0)):
    return [[fill for _ in range(n)] for _ in range(m)]


def copy(a):
    return [row[:] for row in a]


def round_nearest(x):
    # All test values avoid half-integer ties.
    return int(math.floor(float(x) + 0.5))


def kronq_compensation_cancels_g():
    """For E=[E_B,E_R], the optimum E_R is independent of left G."""
    hbr, hrr = F(2), F(5)
    # One row and two columns; H_BB is irrelevant to the suffix derivative.
    expected = -hbr / hrr
    vals = []
    for g in (F(1), F(7), F(101)):
        # d/d e_R of g*(e_B*hbr + e_R*hrr) is zero at this point.
        eb = F(1)
        derivative = g * (eb * hbr + expected * hrr)
        assert derivative == 0
        vals.append(str(expected))
    return {"expected_suffix": str(expected), "left_weights": [1, 7, 101], "all_derivatives_zero": True}


def schur_equals_trailing_factor():
    """Check S_B=U_BB^-1 U_BB^-T for an upper factor H^-1=U^T U."""
    # Upper triangular U; prefix is column 0, suffix is columns 1 and 2.
    U = [[F(2), F(1), F(3)], [F(0), F(2), F(1)], [F(0), F(0), F(3)]]
    # Compute H = (U^T U)^-1 with a tiny exact Gaussian helper.
    UtU = [[sum(U[k][i] * U[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    def inv(a):
        n = len(a)
        aug = [a[i][:] + [F(int(i == j)) for j in range(n)] for i in range(n)]
        for c in range(n):
            p = next(i for i in range(c, n) if aug[i][c])
            aug[c], aug[p] = aug[p], aug[c]
            q = aug[c][c]
            aug[c] = [x / q for x in aug[c]]
            for i in range(n):
                if i != c and aug[i][c]:
                    q = aug[i][c]
                    aug[i] = [x - q * y for x, y in zip(aug[i], aug[c])]
        return [r[n:] for r in aug]

    H = inv(UtU)
    # Schur complement on the current prefix B after eliminating the future R.
    # This is the block that appears in the conditional quadratic objective.
    B, R = (0, 1), (1, 2)
    Hbb = [[H[i][j] for j in B] for i in B]
    Hbr = [[H[i][j] for j in R] for i in B]
    Hrr = [[H[i][j] for j in R] for i in R]
    inv_hrr = inv(Hrr)
    S = [[Hbb[i][j] - sum(Hbr[i][a] * inv_hrr[a][b] * Hbr[j][b] for a in range(2) for b in range(2)) for j in range(1)] for i in range(1)]
    Ubb = [[U[i][j] for j in B] for i in B]
    inv_ubb = inv(Ubb)
    rhs = [[sum(inv_ubb[i][k] * inv_ubb[j][k] for k in range(1)) for j in range(1)] for i in range(1)]
    assert S == rhs, (S, rhs)
    return {"schur": [[str(x) for x in row] for row in S], "factor_product": [[str(x) for x in row] for row in rhs]}


def gptq2d_dense(L, U, X):
    m, n = len(X), len(X[0])
    Y = copy(X)
    Z = mat(m, n)
    for s in range(2, m + n + 1):
        for i in range(1, m + 1):
            j = s - i
            if not (1 <= j <= n):
                continue
            ii, jj = i - 1, j - 1
            z = round_nearest(Y[ii][jj])
            Z[ii][jj] = F(z)
            e = F(z) - Y[ii][jj]
            for k in range(ii, m):
                for l in range(jj, n):
                    Y[k][l] += L[k][ii] * e * U[jj][l]
    return Z


def gptq2d_lazy(L, U, X):
    m, n = len(X), len(X[0])
    Y, Z, C = copy(X), mat(m, n), mat(m, n)
    for s in range(2, m + n + 1):
        for i in range(1, m + 1):
            j = s - i
            if not (1 <= j <= n):
                continue
            ii, jj = i - 1, j - 1
            z = round_nearest(Y[ii][jj])
            Z[ii][jj] = F(z)
            e = F(z) - Y[ii][jj]
            for k in range(ii, m):
                C[k][jj] += L[k][ii] * e
            for k in range(ii + 1, m):
                Y[k][jj] += L[k][ii] * e
            for l in range(jj + 1, n):
                Y[ii][l] += C[ii][jj] * U[jj][l]
    return Z


def gptq2d_exact_trajectory():
    rng = random.Random(20260908)
    cases = 0
    for m in range(1, 6):
        for n in range(1, 6):
            # Unit triangular factors are sufficient to test the propagation.
            L = [[F(int(i == j) if i == j else (rng.randrange(-2, 3) if i > j else 0)) for j in range(m)] for i in range(m)]
            U = [[F(int(i == j) if i == j else (rng.randrange(-2, 3) if j > i else 0)) for j in range(n)] for i in range(n)]
            X = [[F(rng.randrange(-7, 8), 10) for _ in range(n)] for _ in range(m)]
            assert gptq2d_dense(L, U, X) == gptq2d_lazy(L, U, X), (m, n)
            cases += 1
    return {"exact_cases": cases, "dimensions": "1..5 squared", "trajectory_equal": True}


def full_output_coupling_counterexample():
    """Independent row rounding is not a solver for a dense output metric."""
    G = [[F(1), F(9, 10)], [F(9, 10), F(1)]]
    # One column; candidate errors for code (0,0) and (0,1).
    e00, e01 = [F(49, 100), F(49, 100)], [F(49, 100), F(-51, 100)]
    def cost(e):
        return sum(e[i] * G[i][j] * e[j] for i in range(2) for j in range(2))
    c00, c01 = cost(e00), cost(e01)
    assert c01 < c00
    return {"independent_cost": float(c00), "coupled_candidate_cost": float(c01), "coupled_wins": True}


def fisher_mean_field_can_reverse_order():
    """E[d^T E(H)d] can rank two errors differently from E[d^T E(H) d]."""
    H1, H2 = F(100), F(1)
    # A has error on token 2; B on token 1. Mean Hessian is 50.5.
    A = [F(0), F(2)]
    B = [F(1), F(0)]
    exact_A = A[0] * H1 * A[0] + A[1] * H2 * A[1]
    exact_B = B[0] * H1 * B[0] + B[1] * H2 * B[1]
    Hbar = (H1 + H2) / 2
    mean_A = sum(x * Hbar * x for x in A)
    mean_B = sum(x * Hbar * x for x in B)
    assert exact_A < exact_B and mean_A > mean_B
    return {"exact_A": float(exact_A), "exact_B": float(exact_B), "mean_field_A": float(mean_A), "mean_field_B": float(mean_B), "ranking_reversed": True}


def main():
    result = {
        "status": "pass",
        "tests": {
            "kronq_compensation": kronq_compensation_cancels_g(),
            "schur_factor": schur_equals_trailing_factor(),
            "gptq2d": gptq2d_exact_trajectory(),
            "output_coupling": full_output_coupling_counterexample(),
            "fisher_mean_field": fisher_mean_field_can_reverse_order(),
        },
    }
    out = Path(__file__).with_name("checks.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
