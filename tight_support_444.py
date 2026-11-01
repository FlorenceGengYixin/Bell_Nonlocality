"""Exact tight lower support for the 19-term [22][444][22] inequality.

Use S=c.T p with ||c||_1=1 and S>=-1/20; integer form is 20S>=-1.
No saved quantum strategy is needed. LP candidates are checked over Q.
"""
import argparse
import json
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import tripartite_seesaw_444 as model
from tight_support_324 import exact_checks

BASE = Path(__file__).resolve().parent
LABELS = model.LABELS
TERMS = [
    (1, ('PA', 0, 0)), (-1, ('PB', 0, 0)),
    (1, ('PB', 1, 0)), (1, ('PB', 2, 1)),
    (1, ('PAB', 0, 0, 0, 0)), (1, ('PAB', 0, 1, 0, 0)),
    (1, ('PAB', 0, 0, 1, 0)), (-1, ('PAB', 0, 1, 1, 0)),
    (-2, ('PAB', 0, 2, 0, 1)), (1, ('PAB', 0, 0, 1, 2)),
    (-1, ('PAB', 0, 2, 1, 2)), (1, ('PBC', 0, 0, 0, 0)),
    (-1, ('PBC', 1, 0, 0, 1)), (1, ('PBC', 2, 0, 1, 0)),
    (-1, ('PBC', 2, 0, 1, 1)), (-1, ('PBC', 0, 0, 2, 0)),
    (-1, ('PBC', 1, 0, 2, 0)), (1, ('PBC', 1, 0, 2, 1)),
    (1, ('PBC', 2, 0, 2, 1)),
]


def integer_inequality():
    c = np.zeros(49, dtype=np.int64)
    for value, label in TERMS:
        c[model.LABEL_TO_INDEX[label]] += value
    assert np.count_nonzero(c) == 19 and sum(abs(c)) == 20
    return c


def projection_model():
    A, d, R, _ = model.build_hi_system()
    for matrix in (A, d, R):
        if not np.array_equal(matrix, np.rint(matrix)):
            raise ValueError('Expected an integer HI parametrization')
    A, d, R = [np.asarray(M, dtype=np.int64) for M in (A, d, R)]
    rows = []
    for j in range(49):
        candidates = [i for i in np.flatnonzero(R[:, j] == 1)
                      if np.count_nonzero(R[i]) == 1 and d[i] == 0]
        if not candidates:
            raise RuntimeError('Missing homogeneous probability definition')
        rows.append(int(candidates[0]))
    Q = A[rows].copy()
    N = A-R @ Q
    keep = np.any(N != 0, axis=1) | (d != 0)
    return A, d, R, Q, N[keep], d[keep], rows


def check_support():
    A, original_d, R, Q, N, d, rows = projection_model()
    c = integer_inequality()/20
    alpha_exact = [Fraction(int(x), 20) for x in Q.T @ integer_inequality()]
    alpha = np.asarray(alpha_exact, dtype=float)
    result = linprog(
        alpha, A_eq=N, b_eq=d, bounds=(0, None),
        method='highs', options=model.LP_OPTIONS,
    )
    if not result.success:
        raise RuntimeError(result.message)
    w = result.x
    eta = result.eqlin.marginals
    exact, rational = exact_checks(
        N, d, alpha_exact, w, eta, Fraction(-1, 20), 1000000,
    )
    # Independent formulation retains the projected probabilities explicitly.
    joint = linprog(
        np.r_[np.zeros(A.shape[1]), c],
        A_eq=np.hstack([A, -R]), b_eq=original_d,
        bounds=[(0, None)]*A.shape[1]+[(None, None)]*49,
        method='highs', options=model.LP_OPTIONS,
    )
    if not joint.success:
        raise RuntimeError('Independent support LP failed')
    report = dict(
        scenario='[22][444][22]', projected_coordinates=49,
        hidden_weights=A.shape[1], lower_bound='-1/20',
        HI_minimum=float(result.fun), attained_value=float(c @ Q @ w),
        integer_scale_minimum=float(20*result.fun),
        independent_joint_LP_minimum=float(joint.fun),
        primal_dual_gap=abs(float(result.fun-d @ eta)),
        equality_residual=float(abs(N @ w-d).max()),
        **exact,
    )
    arrays = dict(
        coefficients=c, HI_bound=-1/20, saturating_point=Q @ w,
        hidden_weights=w, support_dual=eta, Q=Q, N=N, d=d,
    )
    return report, rational, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report, rational, arrays = check_support()
    output = args.output or BASE/'results_tight_support_444'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    output.mkdir(parents=True, exist_ok=False)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'rational_certificate.json').write_text(
        json.dumps(rational, indent=2)+'\n'
    )
    np.savez(output/'support_certificate.npz', **arrays)
    print(json.dumps(report, indent=2), flush=True)
    print('Saved to', output, flush=True)
    if not report['exact_tight_lower_bound']:
        raise SystemExit('No exact tight-support certificate')


if __name__ == '__main__':
    main()
