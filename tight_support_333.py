"""Check the tight HI support of the freely discovered [22][333][22] witness.
Paper form: c.p >= L. Upper-support form: (-c).p <= -L.
The optimization uses only the HI constraints, not the quantum target point.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from hi_membership_benchmark import build_hi_system, LABELS, DIM, LP_OPTIONS

BASE = Path(__file__).resolve().parent
TOL = 1e-8


def load_witness(path):
    with np.load(path, allow_pickle=False) as data:
        coefficients = np.asarray(data['coefficients'], dtype=float).copy()
        bound = float(data['HI_bound'])
    if coefficients.shape != (DIM,) or not np.all(np.isfinite(coefficients)):
        raise ValueError('Expected a finite coefficient vector in benchmark coordinate order')
    if not np.isfinite(bound) or np.sum(abs(coefficients)) < 1e-14:
        raise ValueError('Invalid witness or bound')
    return coefficients, bound


def validate_behavior(p, weights):
    """Reconstruct all outcome probabilities from nonnegative hidden weights."""
    from hi_membership_benchmark import LAMBDA, w_index
    behavior = np.zeros((2, 3, 2, 2, 3, 2))
    for y in range(3):
        for b in range(3):
            for lam, (avec, cvec) in enumerate(LAMBDA):
                for x in range(2):
                    for z in range(2):
                        behavior[avec[x], b, cvec[z], x, y, z] += weights[w_index(b, y, lam)]
    from tripartite_benchmark import normalization_error, non_signaling_errors
    from hi_membership_benchmark import project_probability
    error = float(np.max(abs(project_probability(behavior) - p)))
    normalization = float(normalization_error(behavior))
    ns = list(map(float, non_signaling_errors(behavior)))
    if behavior.min() < -TOL or max(error, normalization, *ns) > TOL:
        raise RuntimeError('Reconstructed saturating behavior fails physical/NS checks')
    return behavior, dict(
        projection_residual=error,
        normalization_residual=normalization,
        non_signaling_residuals=ns,
        min_probability=float(behavior.min()),
    )


def check_support(coefficients, bound):
    A, constants, R, descriptions = build_hi_system()
    nw = A.shape[1]
    equality = np.hstack([A, -R])
    # u=-c. Max u.p is implemented as min -u.p = min c.p.
    upper_coefficients = -coefficients
    claimed_upper = -bound
    objective = np.r_[np.zeros(nw), -upper_coefficients]
    result = linprog(
        objective,
        A_eq=equality,
        b_eq=constants,
        bounds=[(0, None)] * nw + [(None, None)] * DIM,
        method='highs',
        options=LP_OPTIONS,
    )
    if not result.success:
        raise RuntimeError(f'HI support LP failed: {result.message}')
    weights, point = result.x[:nw], result.x[nw:]
    equality_error = float(np.max(abs(equality @ result.x - constants)))
    if equality_error > TOL or weights.min() < -TOL:
        raise RuntimeError('Support optimizer is not HI-feasible')
    # SciPy minimization dual: E.T lambda <= f on w>=0,
    # E.T lambda = f on free p. The objective is constants.T lambda.
    dual = np.asarray(result.eqlin.marginals)
    slack = objective - equality.T @ dual
    dual_error = max(
        float(np.max(abs(slack[nw:]))),
        max(0., float(-np.min(slack[:nw]))),
    )
    dual_minimum = float(constants @ dual)
    primal_minimum = float(coefficients @ point)
    support = float(upper_coefficients @ point)
    dual_gap = abs(primal_minimum - dual_minimum)
    behavior, diagnostics = validate_behavior(point, weights)
    tight = bool(
        abs(support - claimed_upper) <= TOL
        and dual_error <= TOL
        and dual_gap <= TOL
    )
    norm = float(np.sum(abs(coefficients)))
    scale = 27. / norm
    report = dict(
        convention='maximize u.p with u=-c, beta=-L from saved paper-form c.p>=L',
        claimed_upper_bound=claimed_upper,
        achieved_upper_support=support,
        dual_upper_bound=-dual_minimum,
        support_minus_claimed=support - claimed_upper,
        paper_form_claimed_lower_bound=bound,
        paper_form_HI_minimum=primal_minimum,
        coefficient_l1_norm=norm,
        normalized_upper_support=support / norm,
        rescaled_to_l1_27_support=scale * support,
        rescaled_to_l1_27_claimed_upper=scale * claimed_upper,
        equality_residual=equality_error,
        min_hidden_weight=float(weights.min()),
        dual_feasibility_residual=dual_error,
        primal_dual_gap=dual_gap,
        complementary_slackness_residual=float(np.max(abs(weights * slack[:nw]))),
        tight_within_tolerance=tight,
        tolerance=TOL,
        solver_status=result.message,
        **diagnostics,
    )
    arrays = dict(
        coefficients=coefficients, HI_bound=bound,
        upper_coefficients=upper_coefficients, claimed_upper_bound=claimed_upper,
        saturating_point=point, hidden_weights=weights, full_behavior=behavior,
        Aeq=A, constants=constants, p_coeffs=R, support_dual=dual,
    )
    return report, arrays, descriptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--certificate', type=Path, help='certificate.npz from hi_membership_seesaw.py')
    parser.add_argument('--output', type=Path, help='New output directory')
    args = parser.parse_args()
    certificate = args.certificate
    if certificate is None:
        candidates = list((BASE / 'results_hi_seesaw_333').glob('*/certificate.npz'))
        if not candidates:
            parser.error('Run hi_membership_seesaw.py first, or supply --certificate')
        certificate = max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))
    coefficients, bound = load_witness(certificate)
    report, arrays, descriptions = check_support(coefficients, bound)
    report['certificate_file'] = str(certificate.resolve())
    output = args.output or BASE / 'results_tight_support_333' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / 'support_certificate.npz', **arrays)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'labels.json').write_text(json.dumps(LABELS, indent=2) + '\n')
    (output / 'constraints.json').write_text(json.dumps(descriptions, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('Saved saturating HI behavior and dual certificate to', output)
    if not report['tight_within_tolerance']:
        raise SystemExit('The supplied bound was not verified tight; inspect report.json')


if __name__ == '__main__':
    main()
