"""Sample the HI-tight 324 face and diagnose its extra local-affine directions.

669 LP outputs by default: feasibility, 68 signed coordinate directions,
and 600 signed random directions. Repeated optimizers are retained and
counted separately from distinct points. Select the first point farthest
from aff(local face), then the second farthest from the enlarged hull.
Numerical selection is followed by exact rational feasibility, rank,
distance and bipartite locality/nonlocality witness checks.
These are HI behaviors, not asserted quantum-realizable correlations.
"""
import argparse
import itertools
import json
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.optimize import linprog
import tripartite_seesaw_324 as model
from tight_support_324 import projection_model, validate_behavior
from local_dimension_324 import analyze as local_vertices
from certify_333_facet import exact_rank
from hi_membership_benchmark import find_farkas_certificate

BASE = Path(__file__).resolve().parent


def rational_vector(values):
    return sp.Matrix([sp.Rational(Fraction(float(v)).limit_denominator(1000000))
                      for v in values])


def affine_basis(points):
    origin = points[0]
    _, singular, vh = np.linalg.svd(points[1:]-origin, full_matrices=False)
    rank = int(sum(singular > 1e-9))
    return origin, vh[:rank].T


def farthest(points, origin, basis):
    differences = points-origin
    residuals = differences-(differences @ basis) @ basis.T
    distances = np.linalg.norm(residuals, axis=1)
    maximum = float(distances.max())
    # Stable deterministic tie break; no marginal-locality preference.
    index = int(np.flatnonzero(distances >= maximum-1e-10)[0])
    return index, float(distances[index]), residuals[index]


def exact_distance_squared(points, point):
    origin = points[0, :].T
    differences = points[1:, :]-sp.ones(points.rows-1, 1)*points[0, :]
    null = differences.nullspace()
    if not null:
        return sp.S.Zero
    K = sp.Matrix.hstack(*null)
    v = point-origin
    return sp.factor((v.T*K*(K.T*K).inv()*K.T*v)[0])


def locality_matrix():
    """Full probabilities for a binary two-setting party versus Bob 324."""
    left = list(itertools.product(range(2), repeat=2))
    bob = list(itertools.product(*(range(o) for o in model.BOB_OUTCOMES)))
    responses = list(itertools.product(left, bob))
    labels = [(x, y, a, b) for x in range(2) for y in range(3)
              for a in range(2) for b in range(model.BOB_OUTCOMES[y])]
    M = [np.ones(len(responses), dtype=np.int64)]
    for x, y, a, b in labels:
        M.append([int(A[x] == a and B[y] == b) for A, B in responses])
    return np.asarray(M, dtype=np.int64), labels


def classify_locality(M, rhs, name, output):
    member = linprog(
        np.zeros(M.shape[1]), A_eq=M, b_eq=rhs, bounds=(0, None),
        method='highs', options=model.LP_OPTIONS,
    )
    rhs_r = rational_vector(rhs)
    if member.success:
        weights = rational_vector(member.x)
        exact = all(w >= 0 for w in weights) and sp.Matrix(M)*weights == rhs_r
        residual = float(abs(M @ member.x-rhs).max())
        if residual > 1e-8 or member.x.min() < -1e-8 or not exact:
            raise RuntimeError('Local decomposition not exactly recovered')
        proof = dict(local_weights=list(map(str, weights)))
        report = dict(local=True, exact_certificate=True,
                      primal_residual=residual, strategies=M.shape[1])
    elif member.status == 2:
        _, y = find_farkas_certificate(M, rhs)
        yr = rational_vector(y)
        lhs = sp.Matrix(M).T*yr
        value = rhs_r.dot(yr)
        exact = all(v <= 0 for v in lhs) and value > 0
        if not exact:
            raise RuntimeError('Exact nonlocality separator recovery failed')
        proof = dict(farkas_dual=list(map(str, yr)),
                     exact_separation=str(value))
        report = dict(local=False, exact_certificate=True,
                      exact_separation=str(value), strategies=M.shape[1])
    else:
        raise RuntimeError('Locality LP unresolved: '+member.message)
    proof.update(matrix=M.tolist(), rhs=list(map(str, rhs_r)))
    (output/(name+'.json')).write_text(json.dumps(proof, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-directions', type=int, default=300)
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.random_directions < 0:
        parser.error('Direction count must be nonnegative')
    A, original_d, R, Q, N, d, _ = projection_model()
    T = model.explicit_T()
    alpha = Q.T @ T
    face_matrix = np.vstack([N, alpha])
    face_rhs = np.r_[d, -2]
    rng = np.random.default_rng(args.seed)
    directions = [np.zeros(34)]
    for e in np.eye(34):
        directions.extend([e, -e])
    for _ in range(args.random_directions):
        direction = rng.normal(size=34)
        direction /= np.linalg.norm(direction)
        directions.extend([direction, -direction])
    points, weights = [], []
    max_residual = 0.
    for i, direction in enumerate(directions):
        result = linprog(
            Q.T @ direction, A_eq=face_matrix, b_eq=face_rhs,
            bounds=(0, None), method='highs', options=model.LP_OPTIONS,
        )
        if not result.success:
            raise RuntimeError('HI face optimization failed: '+result.message)
        error = float(abs(face_matrix @ result.x-face_rhs).max())
        if error > 1e-8 or result.x.min() < -1e-8:
            raise RuntimeError('Invalid sampled face optimizer')
        max_residual = max(max_residual, error)
        weights.append(result.x)
        points.append(Q @ result.x)
        if (i+1) % 100 == 0:
            print('Solved face LPs:', i+1, '/', len(directions), flush=True)
    points, weights = np.asarray(points), np.asarray(weights)
    local_report, local_proof, vertices, values, tight = local_vertices()
    local_face = vertices[tight]
    origin, basis = affine_basis(local_face)
    i1, d1, r1 = farthest(points, origin, basis)
    if d1 < 1e-9:
        raise RuntimeError('No sampled direction outside the local face hull')
    enlarged = np.column_stack([basis, r1/d1])
    i2, d2, _ = farthest(points, origin, enlarged)
    if d2 < 1e-9:
        raise RuntimeError('No second independent sampled extra direction')
    Nq, dq, Qq = sp.Matrix(N), sp.Matrix(d), sp.Matrix(Q)
    exact_w, exact_p = [], []
    for i in (i1, i2):
        w = rational_vector(weights[i])
        p = Qq*w
        if not all(v >= 0 for v in w) or Nq*w != dq:
            raise RuntimeError('Exact HI weight reconstruction failed')
        if (sp.Matrix(list(map(int, T))).dot(p)) != -2:
            raise RuntimeError('Selected point is not exactly HI-tight')
        assert sp.Matrix(A)*w == sp.Matrix(original_d)+sp.Matrix(R)*p
        exact_w.append(w)
        exact_p.append(p)
    L = sp.Matrix(local_face)
    L1 = L.col_join(exact_p[0].T)
    L2 = L1.col_join(exact_p[1].T)
    def dimension(M):
        D = M[1:, :]-sp.ones(M.rows-1, 1)*M[0, :]
        return int(exact_rank(D))
    dimensions = [dimension(M) for M in (L, L1, L2)]
    distances = [exact_distance_squared(L, exact_p[0]),
                 exact_distance_squared(L1, exact_p[1])]
    output = args.output or BASE/'results_bipartite_locality_324'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    output.mkdir(parents=True, exist_ok=False)
    M, pair_labels = locality_matrix()
    point_reports = []
    for number, (p, w, squared) in enumerate(zip(exact_p, exact_w, distances), 1):
        wf = np.asarray(w, dtype=float).ravel()
        pf = np.asarray(p, dtype=float).ravel()
        behaviors, diagnostics = validate_behavior(Q, wf)
        if max(diagnostics[k] for k in (
            'behavior_normalization_residual', 'behavior_nonsignaling_residual',
            'behavior_projection_residual',
        )) > 1e-8 or diagnostics['minimum_behavior_probability'] < -1e-8:
            raise RuntimeError('Reconstructed HI behavior failed checks')
        # Independent projected tripartite-local membership test.
        full_M = np.vstack([np.ones(len(vertices), dtype=int), vertices.T])
        projected_local = classify_locality(
            full_M, np.r_[1., pf], f'p{number}_projected_locality', output,
        )
        marginal_reports = {}
        for pair in ('AB', 'BC'):
            rhs = [1.]
            for x, y, a, b in pair_labels:
                if pair == 'AB':
                    rhs.append(behaviors[y][a, b, :, x, 0].sum())
                else:
                    rhs.append(behaviors[y][:, b, a, 0, x].sum())
            marginal_reports[pair] = classify_locality(
                M, np.asarray(rhs), f'p{number}_{pair}_locality', output,
            )
        point_reports.append(dict(
            label=f'p{number}', HI_membership_exact=True, HI_tight_exact=True,
            projected_locality=projected_local,
            distance_squared_exact=str(squared),
            distance=float(sp.sqrt(squared)),
            affine_dimension_after_inclusion=dimensions[number],
            **marginal_reports, **diagnostics,
        ))
    report = dict(
        scenario='[22][324][22]', seed=args.seed,
        LP_outputs=len(points), distinct_points_at_10_decimals=len(
            np.unique(np.round(points, 10), axis=0)
        ),
        signed_random_directions=2*args.random_directions,
        coordinate_directions=68, feasibility_LP=1,
        maximum_sampling_equality_residual=max_residual,
        selected_sample_indices=[i1, i2], numerical_distances=[d1, d2],
        local_face_vertices=len(local_face),
        local_polytope_dimension=local_report['local_affine_dimension'],
        exact_dimension_progression=dimensions, selected_points=point_reports,
        selection='Greedy farthest sampled point from preceding affine hull; '
                  'ties within 1e-10 resolved by earliest sample index',
        scope='Distances depend on the chosen 34-coordinate Euclidean metric. '
              'Farthest means among sampled optimizers, not globally proved. '
              'These HI points are not asserted quantum realizable.',
    )
    proof = dict(
        Q=Q.tolist(), N=N.tolist(), d=d.tolist(),
        T=T.tolist(), integer_bound=-2,
        local_face_vertices=local_face.tolist(),
        points=[list(map(str, p)) for p in exact_p],
        hidden_weights=[list(map(str, w)) for w in exact_w],
        squared_distances=list(map(str, distances)),
        dimension_progression=dimensions,
    )
    np.savez(output/'samples.npz', points=points, hidden_weights=weights,
             directions=np.asarray(directions), selected_indices=[i1, i2])
    for name, content in [('report.json', report), ('exact_points.json', proof)]:
        (output/name).write_text(json.dumps(content, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    print('Saved to', output, flush=True)


if __name__ == '__main__':
    main()
