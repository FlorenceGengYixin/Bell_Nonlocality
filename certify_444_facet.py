"""Exact facet certificate for 20S >= -1 in [22][444][22].

The 92-row membership system is A w = d + R p. Eliminate p=Qw;
N=A-RQ initially has 92 rows. Remove only identically zero equations.
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
from tight_support_444 import projection_model, integer_inequality, LABELS
from tripartite_seesaw_444 import LP_OPTIONS, term_indices
from tripartite_seesaw_324 import explicit_T as reduced_T
import itertools
from certify_333_facet import rat, exact_rank, positive_witness

BASE = Path(__file__).resolve().parent


def load_inequality(certificate=None, c_file=None):
    expected = integer_inequality()/20
    source = 'Exact 19-term integer 444 inequality'
    error = 0.
    if certificate is not None:
        with np.load(certificate, allow_pickle=False) as data:
            c = np.asarray(data['coefficients'], dtype=float)
            bound = float(data['HI_bound'])
        norm = float(sum(abs(c)))
        if norm <= 0:
            raise ValueError('Zero inequality')
        c, bound = c/norm, bound/norm
        error = max(float(abs(c-expected).max()), abs(bound+1/20))
        source = str(certificate.resolve())
    elif c_file is not None:
        c = np.load(c_file, allow_pickle=False)
        if c.shape != (49,) or np.iscomplexobj(c):
            raise ValueError('Expected a real 49-entry coefficient vector')
        norm = float(sum(abs(c)))
        if norm <= 0:
            raise ValueError('Zero inequality')
        error = float(abs(c/norm-expected).max())
        source = str(c_file.resolve())
    if not np.isfinite(error) or error > 1e-9:
        raise ValueError('Saved coefficients/bound differ from S >= -1/20')
    return sp.Matrix(list(map(int, integer_inequality()))), source, error


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
        scenario='[22][444][22]', projected_coordinates=len(LABELS),
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
    local, vertices, responses, values = local_face(coefficients, bound, face_bound)
    mapping, offset = coarse_graining()
    T = sp.Matrix(list(map(int, reduced_T())))
    assert sp.Matrix(mapping).T*T == coefficients
    assert T.dot(sp.Matrix(offset)) == -1
    face_image_rank = int(exact_rank(Nf.col_join(sp.Matrix(mapping)*Qf)))
    face_image_dimension = face_image_rank-rank_NJ
    maps, map_proof = verify_weight_maps(Nr, dr, Qr, mapping, offset)
    report.update(local_analysis=local, **maps,
                  coarse_face_image_dimension=face_image_dimension,
                  exact_affine_identity='20S_444 = 1 + T_324',
                  facet_to_facet_affine_dimensions=(
                      facet and face_image_dimension == 33
                  ))
    proof.update(
        weight_map_proof=map_proof,
        local_vertices=vertices, local_responses=responses,
        local_integer_values=values,
        coarse_graining_matrix=mapping.tolist(),
        coarse_graining_offset=offset.tolist(),
    )
    return report, proof


def verify_weight_maps(N, d, Q, mapping, offset):
    from tight_support_324 import projection_model as reduced_model
    from tripartite_seesaw_324 import MERGES, w_index
    _, _, _, q, n, r, _ = reduced_model()
    n, r, q = sp.Matrix(n), sp.Matrix(r), sp.Matrix(q)
    C = sp.zeros(144, 192)
    D = sp.zeros(192, 144)
    for y, groups in enumerate(MERGES):
        for b, group in enumerate(groups):
            for lam in range(16):
                row = w_index(b, y, lam)
                for old_b in group:
                    C[row, (4*y+old_b)*16+lam] = 1
                D[(4*y+group[0])*16+lam, row] = 1
    assert C*D == sp.eye(144)
    old_aug = N.row_join(-d)
    new_aug = n.row_join(-r)
    ranks = dict(
        coarse_feasibility=int(exact_rank(
            old_aug.col_join((n*C).row_join(-r))
        )),
        refinement_feasibility=int(exact_rank(
            new_aug.col_join((N*D).row_join(-d))
        )),
        probability_map=int(exact_rank(old_aug.col_join(
            (q*C-sp.Matrix(mapping)*Q).row_join(-sp.Matrix(offset))
        ))),
    )
    old_rank, new_rank = int(exact_rank(old_aug)), int(exact_rank(new_aug))
    assert ranks['coarse_feasibility'] == old_rank
    assert ranks['refinement_feasibility'] == new_rank
    assert ranks['probability_map'] == old_rank
    return dict(exact_coarse_map_surjective_on_HI=True), dict(
        coarse_weight_map=[[int(v) for v in row] for row in C.tolist()],
        refinement_weight_map=[[int(v) for v in row] for row in D.tolist()],
        implication_ranks=ranks, old_rank=old_rank, new_rank=new_rank,
    )


def local_face(coefficients, bound, hi_face_dimension):
    responses, vertices = [], []
    A = list(itertools.product(range(2), repeat=2))
    B = list(itertools.product(range(4), repeat=3))
    C = list(itertools.product(range(2), repeat=2))
    for response in itertools.product(A, B, C):
        responses.append(response)
        vertices.append([
            int(all(response[j][x] == a
                    for j, (x, a) in term_indices(label).items()))
            for label in LABELS
        ])
    P = sp.Matrix(vertices)
    values = P*coefficients
    minimum = min(values)
    tight = [i for i, v in enumerate(values) if v == bound]
    D = P[1:, :]-sp.ones(P.rows-1, 1)*P[0, :]
    F = P[tight, :]
    DF = F[1:, :]-sp.ones(F.rows-1, 1)*F[0, :]
    dimension = int(exact_rank(D))
    face_dimension = int(exact_rank(DF))
    report = dict(
        deterministic_strategies=len(vertices),
        saturating_vertices=len(tight),
        integer_local_bound=str(minimum),
        local_and_HI_bounds_coincide=bool(minimum == bound),
        local_dimension=dimension, local_face_dimension=face_dimension,
        local_facet=bool(face_dimension == dimension-1),
        extra_HI_face_dimensions=hi_face_dimension-face_dimension,
    )
    return report, vertices, responses, list(map(str, values))


def coarse_graining():
    from tripartite_seesaw_324 import LABELS as target_labels
    index = {label: i for i, label in enumerate(LABELS)}
    M = np.zeros((34, 49), dtype=np.int64)
    offset = np.zeros(34, dtype=np.int64)
    for i, label in enumerate(target_labels):
        kind = label[0]
        if kind in ('PA', 'PC'):
            M[i, index[label]] = 1
            continue
        y = label[2] if kind == 'PB' else label[4]
        if kind == 'PBC':
            y = label[3]
        if y != 1:
            M[i, index[label]] = 1
        elif kind == 'PB':
            M[i, index[('PB', 2, 1)]] = -1
            offset[i] = 1
        elif kind == 'PAB':
            _, a, b, x, y = label
            M[i, index[('PA', 0, x)]] = 1
            M[i, index[('PAB', 0, 2, x, 1)]] = -1
        else:
            _, b, c, y, z = label
            M[i, index[('PC', 0, z)]] = 1
            M[i, index[('PBC', 2, 0, 1, z)]] = -1
    return M, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--certificate', type=Path)
    group.add_argument('--c-file', type=Path)
    parser.add_argument('--bound', default='-1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    coefficients, source, error = load_inequality(args.certificate, args.c_file)
    bound = sp.Rational(args.bound)
    report, proof = certify(coefficients, bound)
    report.update(inequality_source=source, reconstruction_error=error)
    output = args.output or BASE/'results_certify_444_facet'/(
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
