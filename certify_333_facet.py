"""Exact facet certification for the projected [22][333][22] HI polytope.
Eliminate p to obtain Nw=r, p=Qw. N is NOT the original membership A.
Use exact rational dual slack, positive rational full/face witnesses and
exact QQ ranks. No floating-point rank or slack threshold proves a facet.
"""
import argparse
from datetime import datetime
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.optimize import linprog
from hi_membership_benchmark import build_hi_system, LABELS, DIM, paper_eq4

BASE = Path(__file__).resolve().parent

def rat(v, den=1000000):
    return sp.Rational(Fraction(float(v)).limit_denominator(den))

def exact_rank(mat):
    return mat.to_DM().convert_to(sp.QQ).rank()

def exact_interior_witness(N, r, x, eps=1e-11, den=100000000):
    """Repair numerical x to exact Nw=r using rational free coords.
    Uses pivot variables as exact dependent coords. Require all x > 0.
    """
    if x is None or np.min(x) <= eps:
        return (False, None, 'numerical witness not strictly positive')
    M = sp.Matrix(N)
    R, piv = M.rref()
    piv = list(piv)
    free = [j for j in range(M.cols) if j not in piv]
    v = sp.Matrix([rat(t, den) for t in x])
    aug = M.row_join(sp.Matrix(r))
    aug_rref, aug_piv = aug.rref()
    if M.cols in aug_piv:
        return (False, None, 'inconsistent exact linear equations')
    for row, j in enumerate(piv):
        v[j] = aug_rref[row, M.cols] - sum((aug_rref[row, k] * v[k] for k in free), sp.S.Zero)
    good = all((q > 0 for q in v)) and all((t == 0 for t in M * v - sp.Matrix(r)))
    return (good, v, 'exact repaired witness' if good else 'rational pivot repair lost positivity')

def strict_lp(N, r):
    n = N.shape[1]
    if n == 0:
        return None
    Aub = np.hstack([-np.eye(n), np.ones((n, 1))])
    Aeq = np.hstack([N, np.zeros((N.shape[0], 1))])
    res = linprog(np.r_[np.zeros(n), -1.0], A_ub=Aub, b_ub=np.zeros(n), A_eq=Aeq, b_eq=r, bounds=[(0, None)] * n + [(0, 1)], method='highs')
    return res

def integer_array(values):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)) or not np.array_equal(values, np.rint(values)):
        raise ValueError('Expected exact integer model coefficients')
    return np.asarray(values, dtype=np.int64)


def projection_model():
    A, constants, R, _ = build_hi_system()
    A, constants, R = map(integer_array, (A, constants, R))
    rows = []
    for j in range(DIM):
        candidates = [i for i in range(len(constants)) if R[i,j] == 1
                      and np.count_nonzero(R[i]) == 1 and constants[i] == 0]
        if not candidates:
            raise ValueError(f'No direct homogeneous projection for {LABELS[j]}')
        rows.append(candidates[0])
    Q = A[rows].copy()
    N = A - R @ Q
    keep = np.any(N != 0, axis=1) | (constants != 0)
    return N[keep], constants[keep], Q, A, R, rows


def rational_witness(certificate):
    coefficients, bound = paper_eq4()
    if certificate is None:
        return sp.Matrix(integer_array(coefficients)), sp.Integer(int(bound)), 'exact paper Eq.(4)'
    with np.load(certificate, allow_pickle=False) as data:
        c = np.asarray(data['coefficients'], dtype=float)
        L = float(data['HI_bound'])
    if c.shape != (DIM,) or not np.all(np.isfinite(c)) or not np.isfinite(L):
        raise ValueError('Invalid saved witness')
    norm = np.sum(abs(c))
    if norm <= 1e-14:
        raise ValueError('Zero witness')
    # Positive rescaling is harmless; record the precise rational inequality
    # being certified, not a claim that a float equals a rational exactly.
    scale = 27. / norm
    c, L = c*scale, L*scale
    cr = sp.Matrix([rat(v) for v in c])
    Lr = rat(L)
    error = max([abs(float(a)-b) for a,b in zip(cr,c)] + [abs(float(Lr)-L)])
    if error > 1e-9:
        raise ValueError('Witness rational reconstruction error exceeds tolerance')
    return cr, Lr, f'rational reconstruction of {certificate.resolve()}, scaled to L1=27; error={error:.3g}'


def positive_witness(N, r):
    result = strict_lp(np.asarray(N, dtype=float), np.asarray(r, dtype=float).ravel())
    if result is None or not result.success or result.x[-1] <= 1e-10:
        return False, None, 'No strictly positive numerical candidate; certification incomplete'
    return exact_interior_witness(N, r, result.x[:-1])


def certify(coefficients, bound):
    N, r, Q, A, R, rows = projection_model()
    Nr, rr, Qr = sp.Matrix(N), sp.Matrix(r), sp.Matrix(Q)
    alpha = Qr.T * coefficients
    solution = linprog(
        np.array(alpha, dtype=float).ravel(), A_eq=N, b_eq=r,
        bounds=(0, None), method='highs',
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    eta = sp.Matrix([rat(v) for v in solution.eqlin.marginals])
    slack = alpha - Nr.T * eta
    if rr.dot(eta) != bound or any(s < 0 for s in slack):
        raise RuntimeError('Exact dual reconstruction failed: no facet certificate issued')
    J = [j for j,s in enumerate(slack) if s == 0]
    Nf, Qf = Nr[:,J], Qr[:,J]
    print(f'Exact dual valid; {len(J)} zero-slack columns out of {N.shape[1]}', flush=True)
    full_ok, full_point, full_message = positive_witness(Nr, rr)
    face_ok, face_point, face_message = positive_witness(Nf, rr)
    # Strict positivity makes the rank difference an equality, not just an upper bound.
    rank_N = int(exact_rank(Nr))
    rank_NQ = int(exact_rank(Nr.col_join(Qr)))
    rank_face_N = int(exact_rank(Nf))
    rank_face_NQ = int(exact_rank(Nf.col_join(Qf)))
    dimension = rank_NQ - rank_N
    face_dimension_bound = rank_face_NQ - rank_face_N
    facet = bool(full_ok and face_ok and face_dimension_bound == dimension-1)
    verdict = 'CERTIFIED FACET' if facet else (
        'CERTIFIED NOT A FACET' if full_ok and face_ok else 'INCOMPLETE')
    report = dict(
        scenario='[22][333][22]', projected_coordinates=DIM,
        hidden_weights=N.shape[1], eliminated_N_shape=list(N.shape),
        rank_N=rank_N, rank_NQ=rank_NQ,
        rank_face_N=rank_face_N, rank_face_NQ=rank_face_NQ,
        HI_affine_dimension=dimension if full_ok else None,
        face_dimension=face_dimension_bound if face_ok else None,
        face_dimension_upper_bound=face_dimension_bound,
        zero_slack_columns=len(J), exact_dual_valid=True,
        lower_bound=str(bound), upper_form_bound=str(-bound),
        full_exact_strict_positivity=bool(full_ok),
        face_exact_strict_positivity=bool(face_ok),
        full_message=full_message, face_message=face_message,
        full_min_weight=str(min(full_point)) if full_ok else None,
        face_min_supported_weight=str(min(face_point)) if face_ok else None,
        facet_certified=facet, verdict=verdict,
        proof_scope='Exact rational proof relative to the specified HI weight/projection model',
    )
    proof = dict(
        N=N.tolist(), r=r.tolist(), Q=Q.tolist(), projection_rows=rows,
        lower_coefficients=[str(x) for x in coefficients], lower_bound=str(bound),
        lower_dual=[str(x) for x in eta], lower_slack=[str(x) for x in slack],
        J=J, full_positive_weights=[str(x) for x in full_point] if full_ok else None,
        face_supported_positive_weights=[str(x) for x in face_point] if face_ok else None,
    )
    # Recheck the original membership model for both exact rational witnesses.
    if full_ok:
        assert sp.Matrix(A)*full_point == sp.Matrix(integer_array(build_hi_system()[1])) + sp.Matrix(R)*Qr*full_point
    if face_ok:
        padded = sp.zeros(N.shape[1],1)
        for j,value in zip(J,face_point):
            padded[j] = value
        assert Nr*padded == rr
        assert coefficients.dot(Qr*padded) == bound
    return report, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--certificate', type=Path, help='Optional discovered witness certificate.npz; default exact Eq.(4)')
    parser.add_argument('--output', type=Path, help='New output directory')
    args = parser.parse_args()
    coefficients, bound, source = rational_witness(args.certificate)
    report, proof = certify(coefficients, bound)
    report['inequality_source'] = source
    output = args.output or BASE/'results_certify_333_facet'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'exact_certificate.json').write_text(json.dumps(proof, indent=2)+'\n')
    (output/'labels.json').write_text(json.dumps(LABELS, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    print('Exact witnesses and rational dual saved to', output)
    if not report['facet_certified']:
        raise SystemExit('No exact facet certificate: '+report['verdict'])


if __name__ == '__main__':
    main()
