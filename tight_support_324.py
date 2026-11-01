"""Certify the tight lower HI bound of T/22 in [22][324][22].

Eliminate the projected probabilities: p=Qw, Nw=d, w>=0, alpha=Q.T c.
Minimize alpha.T w independently of any quantum strategy or claimed bound.
Check attainment, LP duality and rational reconstructions of both witnesses.
The exact conclusion is conditional on the supplied HI parametrization.
"""
import argparse
import json
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import tripartite_seesaw_324 as model

BASE = Path(__file__).resolve().parent


def projection_model():
    A, d, R = model.build_hi_system()
    for M in (A, d, R):
        if not np.array_equal(M, np.rint(M)):
            raise ValueError('Expected integer HI constraints')
    A, d, R = [np.asarray(M, dtype=np.int64) for M in (A, d, R)]
    rows = []
    for j in range(len(model.LABELS)):
        candidates = [i for i in np.flatnonzero(R[:, j] == 1)
                      if np.count_nonzero(R[i]) == 1]
        if not candidates:
            raise RuntimeError('No defining equation for probability coordinate')
        rows.append(candidates[0])
    if np.any(d[rows]):
        raise RuntimeError('Unexpected affine offset in p=Qw')
    Q = A[rows].copy()
    N = A-R @ Q
    active = np.any(N != 0, axis=1) | (d != 0)
    return A, d, R, Q, N[active], d[active], rows


def fraction_vector(values, denominator):
    return [Fraction(float(x)).limit_denominator(denominator) for x in values]


def exact_dot(row, vector):
    return sum((int(a)*b for a, b in zip(row, vector)), Fraction(0))


def exact_checks(N, d, alpha, weights, dual, claimed, denominator):
    w = fraction_vector(weights, denominator)
    eta = fraction_vector(dual, denominator)
    primal_feasible = all(x >= 0 for x in w) and all(
        exact_dot(row, w) == int(rhs) for row, rhs in zip(N, d)
    )
    slack = [a-exact_dot(col, eta) for a, col in zip(alpha, N.T)]
    dual_feasible = all(s >= 0 for s in slack)
    primal = sum((a*b for a, b in zip(alpha, w)), Fraction(0))
    dual_value = exact_dot(d, eta)
    exact = (primal_feasible and dual_feasible
             and primal == dual_value == claimed)
    report = dict(
        exact_primal_feasible=primal_feasible,
        exact_dual_feasible=dual_feasible,
        rational_primal_value=str(primal),
        rational_dual_value=str(dual_value),
        exact_tight_lower_bound=exact,
        rational_denominator_limit=denominator,
    )
    certificate = dict(
        hidden_weights=list(map(str, w)), dual=list(map(str, eta)),
        alpha=list(map(str, alpha)), slack=list(map(str, slack)),
        lower_bound=str(claimed),
    )
    return report, certificate


def validate_behavior(Q, weights):
    behaviors = []
    for y, outcomes in enumerate(model.BOB_OUTCOMES):
        P = np.zeros((2, outcomes, 2, 2, 2))
        for b in range(outcomes):
            for lam, (a, c) in enumerate(model.LAMBDA):
                for x in range(2):
                    for z in range(2):
                        P[a[x], b, c[z], x, z] += weights[
                            model.w_index(b, y, lam)
                        ]
        behaviors.append(P)
    normalization, ns = 0., 0.
    for P in behaviors:
        normalization = max(normalization, float(
            abs(P.sum(axis=(0, 1, 2))-1).max()
        ))
        AB, BC = P.sum(axis=2), P.sum(axis=0)
        ns = max(ns, float(abs(AB[:, :, :, 0]-AB[:, :, :, 1]).max()))
        ns = max(ns, float(abs(BC[:, :, 0, :]-BC[:, :, 1, :]).max()))
    for axes in ((1, 2), (0, 1), (1,)):
        marginals = [P.sum(axis=axes) for P in behaviors]
        ns = max(ns, max(float(abs(m-marginals[0]).max())
                         for m in marginals))
    point = []
    for label in model.LABELS:
        kind = label[0]
        if kind == 'PA':
            _, a, x = label
            value = behaviors[0][a, :, :, x, 0].sum()
        elif kind == 'PC':
            _, c, z = label
            value = behaviors[0][:, :, c, 0, z].sum()
        elif kind == 'PB':
            _, b, y = label
            value = behaviors[y][:, b, :, 0, 0].sum()
        elif kind == 'PAB':
            _, a, b, x, y = label
            value = behaviors[y][a, b, :, x, 0].sum()
        else:
            _, b, c, y, z = label
            value = behaviors[y][:, b, c, 0, z].sum()
        point.append(value)
    projection = float(abs(np.asarray(point)-Q @ weights).max())
    minimum = min(float(P.min()) for P in behaviors)
    return behaviors, dict(
        behavior_normalization_residual=normalization,
        behavior_nonsignaling_residual=ns,
        behavior_projection_residual=projection,
        minimum_behavior_probability=minimum,
    )


def check_support(claimed=Fraction(-1, 11), tolerance=1e-8,
                  denominator=1000000):
    A, original_d, R, Q, N, d, rows = projection_model()
    integer_c = model.explicit_T()
    c = integer_c/22
    alpha_integer = Q.T @ integer_c
    alpha_exact = [Fraction(int(a), 22) for a in alpha_integer]
    alpha = np.asarray(alpha_exact, dtype=float)
    result = linprog(
        alpha, A_eq=N, b_eq=d, bounds=(0, None),
        method='highs', options=model.LP_OPTIONS,
    )
    if not result.success:
        raise RuntimeError('Support LP failed: '+result.message)
    weights = result.x
    point = Q @ weights
    eta = result.eqlin.marginals
    slack = alpha-N.T @ eta
    primal = float(alpha @ weights)
    dual = float(d @ eta)
    residual = float(abs(N @ weights-d).max())
    dual_error = max(0., float(-slack.min()))
    # An independent formulation retains all 34 free probability coordinates.
    E = np.hstack([A, -R])
    objective = np.r_[np.zeros(model.NW), c]
    joint = linprog(
        objective, A_eq=E, b_eq=original_d,
        bounds=[(0, None)]*model.NW+[(None, None)]*34,
        method='highs', options=model.LP_OPTIONS,
    )
    if not joint.success:
        raise RuntimeError('Independent (w,p) support LP failed')
    behaviors, diagnostics = validate_behavior(Q, weights)
    exact, rational = exact_checks(
        N, d, alpha_exact, weights, eta, claimed, denominator,
    )
    numerical = (
        abs(primal-float(claimed)) <= tolerance
        and residual <= tolerance and weights.min() >= -tolerance
        and dual_error <= tolerance and abs(primal-dual) <= tolerance
        and abs(joint.fun-primal) <= tolerance
        and diagnostics['behavior_normalization_residual'] <= tolerance
        and diagnostics['behavior_nonsignaling_residual'] <= tolerance
        and diagnostics['behavior_projection_residual'] <= tolerance
        and diagnostics['minimum_behavior_probability'] >= -tolerance
    )
    report = dict(
        scenario='[22][324][22]', projected_dimension=34, hidden_weights=144,
        convention='min c.T p = min alpha.T w; valid inequality c.T p >= L',
        claimed_lower_bound=str(claimed), HI_minimum=primal,
        attained_value=float(c @ point), dual_lower_bound=dual,
        independent_joint_LP_minimum=float(joint.fun),
        integer_scale_minimum=22*primal, coefficient_l1=float(sum(abs(c))),
        equality_residual=residual, dual_feasibility_residual=dual_error,
        primal_dual_gap=abs(primal-dual),
        complementary_slackness_residual=float(abs(weights*slack).max()),
        minimum_hidden_weight=float(weights.min()),
        positive_hidden_weights=int(sum(weights > tolerance)),
        N_shape=list(N.shape), Q_shape=list(Q.shape),
        tight_within_tolerance=bool(numerical), tolerance=tolerance,
        solver_status=result.message, **diagnostics, **exact,
        limitation='Exact certificate is conditional on correctness of the '
                   'HI projection model; rational reconstruction is verified '
                   'exactly rather than accepted from rounding alone',
    )
    arrays = dict(
        coefficients=c, HI_bound=float(claimed), alpha=alpha,
        hidden_weights=weights, saturating_point=point, dual=eta, slack=slack,
        Q=Q, N=N, constants=d, A=A, original_constants=original_d, R=R,
        projection_rows=rows,
    )
    arrays.update({f'behavior_y{y}': P for y, P in enumerate(behaviors)})
    return report, arrays, rational


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bound', default='-1/11')
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--max-denominator', type=int, default=1000000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.tol <= 0 or args.max_denominator < 1:
        parser.error('Tolerance and denominator must be positive')
    report, arrays, rational = check_support(
        Fraction(args.bound), args.tol, args.max_denominator,
    )
    output = args.output or BASE/'results_tight_support_324'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output/'support_certificate.npz', **arrays)
    for filename, content in (
        ('report.json', report), ('rational_certificate.json', rational),
        ('labels.json', model.LABELS),
    ):
        (output/filename).write_text(json.dumps(content, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    print('Saved to', output, flush=True)
    if not report['tight_within_tolerance']:
        raise SystemExit('Claimed bound is not verified tight')


if __name__ == '__main__':
    main()
