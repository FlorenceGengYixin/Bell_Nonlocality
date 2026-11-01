"""Exact facet certificate for T/22 >= -1/11 in [22][324][22].

The 77-row membership system is A w = d + R p. Eliminate p=Qw;
N=A-RQ initially has 77 rows. Remove only identically zero equations.
A facet verdict requires a rational dual, exact zero slack, positive
rational full/face witnesses and all four ranks computed over QQ.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.optimize import linprog
from tight_support_324 import projection_model
from tripartite_seesaw_324 import explicit_T, LABELS, LP_OPTIONS
from certify_333_facet import rat, exact_rank, positive_witness

BASE = Path(__file__).resolve().parent


def load_inequality(certificate=None, c_file=None):
    expected = explicit_T()/22
    source = 'Exact 21-term T/22 inequality'
    error = 0.
    if certificate is not None:
        with np.load(certificate, allow_pickle=False) as data:
            c = np.asarray(data['coefficients'], dtype=float)
            bound = float(data['HI_bound'])
        norm = float(sum(abs(c)))
        if norm <= 0:
            raise ValueError('Zero inequality')
        c, bound = c/norm, bound/norm
        error = max(float(abs(c-expected).max()), abs(bound+1/11))
        source = str(certificate.resolve())
    elif c_file is not None:
        c = np.load(c_file, allow_pickle=False)
        if c.shape != (34,) or np.iscomplexobj(c):
            raise ValueError('Expected a real 34-entry coefficient vector')
        norm = float(sum(abs(c)))
        if norm <= 0:
            raise ValueError('Zero inequality')
        error = float(abs(c/norm-expected).max())
        source = str(c_file.resolve())
    if not np.isfinite(error) or error > 1e-9:
        raise ValueError('Saved coefficients/bound differ from T/22 >= -1/11')
    return sp.Matrix(list(map(int, explicit_T())))/22, source, error


def certify(coefficients, bound):
    A, original_d, R, Q, N, d, rows = projection_model()
    Nr, dr, Qr = sp.Matrix(N), sp.Matrix(d), sp.Matrix(Q)
    alpha = Qr.T*coefficients
    primal = linprog(
        np.asarray(alpha, dtype=float).ravel(),
        A_eq=N, b_eq=d, bounds=(0, None),
        method='highs', options=LP_OPTIONS,
    )
    if not primal.success:
        raise RuntimeError('HI support LP failed: '+primal.message)
    eta = sp.Matrix([rat(v) for v in primal.eqlin.marginals])
    delta = alpha-Nr.T*eta
    if any(s < 0 for s in delta) or dr.dot(eta) != bound:
        raise RuntimeError('No exact dual certificate for the supplied bound')
    J = [j for j, s in enumerate(delta) if s == 0]
    Nf, Qf = Nr[:, J], Qr[:, J]
    print('Exact dual verified; zero-slack columns:', len(J), flush=True)
    full_ok, full_w, full_message = positive_witness(Nr, dr)
    face_ok, face_w, face_message = positive_witness(Nf, dr)
    print('Exact positive witnesses:', full_ok, face_ok, flush=True)
    print('Computing exact rational ranks...', flush=True)
    rank_N = int(exact_rank(Nr))
    rank_NQ = int(exact_rank(Nr.col_join(Qr)))
    rank_NJ = int(exact_rank(Nf))
    rank_NJQJ = int(exact_rank(Nf.col_join(Qf)))
    dimension = rank_NQ-rank_N
    face_bound = rank_NJQJ-rank_NJ
    # Nonnegativity does not shrink the affine hull when a positive point exists.
    dimension_attained = bool(full_ok)
    face_attained = bool(face_ok)
    facet = dimension_attained and face_attained and face_bound == dimension-1
    full_projected = None
    face_projected = None
    padded = None
    if full_ok:
        full_projected = Qr*full_w
        assert Nr*full_w == dr and all(w > 0 for w in full_w)
        assert sp.Matrix(A)*full_w == (
            sp.Matrix(original_d)+sp.Matrix(R)*full_projected
        )
    if face_ok:
        padded = sp.zeros(A.shape[1], 1)
        for j, w in zip(J, face_w):
            padded[j] = w
        face_projected = Qr*padded
        assert Nf*face_w == dr and all(w > 0 for w in face_w)
        assert coefficients.dot(face_projected) == bound
        assert sp.Matrix(A)*padded == (
            sp.Matrix(original_d)+sp.Matrix(R)*face_projected
        )
        assert all(padded[j] == 0 for j in range(A.shape[1]) if j not in J)
    verdict = ('CERTIFIED FACET' if facet else
               'CERTIFIED NOT A FACET' if full_ok and face_ok else 'INCOMPLETE')
    report = dict(
        scenario='[22][324][22]', projected_coordinates=len(LABELS),
        hidden_weights=A.shape[1], membership_A_shape=list(A.shape),
        eliminated_N_before_zero_row_removal=list((A-R @ Q).shape),
        eliminated_N_shape=list(N.shape), Q_shape=list(Q.shape),
        rank_N=rank_N, rank_NQ=rank_NQ,
        rank_NJ=rank_NJ, rank_NJQJ=rank_NJQJ,
        HI_affine_dimension=dimension if full_ok else None,
        face_dimension=face_bound if face_ok else None,
        face_dimension_upper_bound=face_bound,
        zero_slack_columns=len(J), zero_slack_indices=J,
        exact_dual_valid=True, exact_lower_bound=str(bound),
        full_exact_strict_positivity=bool(full_ok),
        face_exact_strict_positivity=bool(face_ok),
        full_min_weight=str(min(full_w)) if full_ok else None,
        face_min_supported_weight=str(min(face_w)) if face_ok else None,
        full_message=full_message, face_message=face_message,
        facet_certified=bool(facet), verdict=verdict,
        proof_scope='Exact rational proof relative to the specified HI '
                    'weight and projection parametrization',
        note='J is computed from one exact feasible optimal dual. Strict '
             'positivity on J proves the rank difference is attained.',
    )
    proof = dict(
        A=A.tolist(), original_d=original_d.tolist(), R=R.tolist(),
        N=N.tolist(), d=d.tolist(), Q=Q.tolist(), projection_rows=list(map(int, rows)),
        coefficients=list(map(str, coefficients)), bound=str(bound),
        alpha=list(map(str, alpha)), dual=list(map(str, eta)),
        slack=list(map(str, delta)), J=J,
        full_positive_weights=list(map(str, full_w)) if full_ok else None,
        face_supported_positive_weights=(list(map(str, face_w))
                                         if face_ok else None),
        face_padded_weights=list(map(str, padded)) if face_ok else None,
        full_projected_point=(list(map(str, full_projected))
                              if full_ok else None),
        face_projected_point=(list(map(str, face_projected))
                              if face_ok else None),
    )
    return report, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--certificate', type=Path)
    group.add_argument('--c-file', type=Path)
    parser.add_argument('--bound', default='-1/11')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    coefficients, source, error = load_inequality(args.certificate, args.c_file)
    bound = sp.Rational(args.bound)
    report, proof = certify(coefficients, bound)
    report.update(inequality_source=source, reconstruction_error=error)
    output = args.output or BASE/'results_certify_324_facet'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (
        ('report.json', report), ('exact_certificate.json', proof),
        ('labels.json', LABELS),
    ):
        (output/name).write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    print('Saved exact witnesses and ranks to', output, flush=True)
    if not report['facet_certified']:
        raise SystemExit('No exact facet certificate: '+report['verdict'])


if __name__ == '__main__':
    main()
