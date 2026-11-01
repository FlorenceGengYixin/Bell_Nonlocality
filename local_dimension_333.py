"""Exact local-polytope dimension and saturating-face test for [22][333][22].

Enumerate all 4*27*4 deterministic strategies in the same 34-coordinate
chart as hi_membership_benchmark.py. Evaluate the original integer paper Eq.(4), S>=-2.
Affine dimensions use exact rational ranks, with no numerical rank cutoff.
"""
import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import sympy as sp
from hi_membership_benchmark import LABELS, paper_eq4

BOB_OUTCOMES = (3, 3, 3)
from certify_333_facet import exact_rank

BASE = Path(__file__).resolve().parent


def deterministic_point(a, b, c):
    values = []
    for label in LABELS:
        kind = label[0]
        if kind == 'PA':
            _, outcome, x = label
            value = a[x] == outcome
        elif kind == 'PB':
            _, outcome, y = label
            value = b[y] == outcome
        elif kind == 'PC':
            _, outcome, z = label
            value = c[z] == outcome
        elif kind == 'PAB':
            _, aa, bb, x, y = label
            value = a[x] == aa and b[y] == bb
        elif kind == 'PBC':
            _, bb, cc, y, z = label
            value = b[y] == bb and c[z] == cc
        else:
            raise ValueError(f"Unsupported coordinate: {label}")
        values.append(int(value))
    return values


def affine_dimension(points):
    if len(points) == 0:
        return -1
    differences = sp.Matrix(points[1:] - points[0])
    if len(points) == 1:
        return 0
    return int(exact_rank(differences))


def analyze():
    alice = list(itertools.product(range(2), repeat=2))
    bob = list(itertools.product(*(range(o) for o in BOB_OUTCOMES)))
    charlie = list(itertools.product(range(2), repeat=2))
    strategies, points = [], []
    for a, b, c in itertools.product(alice, bob, charlie):
        strategies.append((a, b, c))
        points.append(deterministic_point(a, b, c))
    points = np.asarray(points, dtype=np.int64)
    total = len(alice)*len(bob)*len(charlie)
    assert total == 432 and points.shape == (432, 34)
    raw_coefficients, raw_bound = paper_eq4()
    if not np.array_equal(raw_coefficients, np.rint(raw_coefficients)):
        raise ValueError("Expected integer Eq.(4) coefficients")
    coefficients = np.asarray(raw_coefficients, dtype=np.int64)
    if raw_bound != -2:
        raise ValueError("Unexpected Eq.(4) bound")
    coefficient_l1 = int(np.abs(coefficients).sum())
    values = points @ coefficients
    local_bound = int(values.min())
    claimed_HI_bound = -2
    if local_bound < claimed_HI_bound:
        raise RuntimeError('A local vertex violates the claimed HI inequality')
    saturating = np.flatnonzero(values == claimed_HI_bound)
    local_dimension = affine_dimension(points)
    face_dimension = affine_dimension(points[saturating])
    histogram = {str(int(v)): int(sum(values == v)) for v in np.unique(values)}
    is_facet = bool(
        local_bound == claimed_HI_bound
        and face_dimension == local_dimension-1
    )
    report = dict(
        scenario='[22][333][22]', projected_coordinates=len(LABELS),
        alice_deterministic_responses=len(alice),
        bob_deterministic_responses=len(bob),
        charlie_deterministic_responses=len(charlie),
        total_deterministic_strategies=total,
        distinct_projected_vertices=len(np.unique(points, axis=0)),
        integer_inequality='paper Eq.(4): S >= -2', integer_coefficient_l1=coefficient_l1,
        integer_local_minimum=local_bound,
        normalized_local_minimum=str(sp.Rational(local_bound, coefficient_l1)),
        integer_HI_bound=claimed_HI_bound,
        local_and_claimed_HI_bounds_coincide=(local_bound == claimed_HI_bound),
        saturating_vertices=len(saturating),
        local_affine_dimension=local_dimension,
        saturating_face_affine_dimension=face_dimension,
        local_face_codimension=(local_dimension-face_dimension
                                if face_dimension >= 0 else None),
        is_local_facet=is_facet, exact_value_histogram=histogram,
        rank_method='Exact QQ rank of differences from one vertex',
        note='The HI bound is the previously certified value; local bounds '
             'and dimensions here are independently obtained by enumeration.',
    )
    proof = dict(
        labels=LABELS, coefficients=coefficients.tolist(), integer_bound=-2,
        deterministic_responses=[dict(A=a, B=b, C=c)
                                 for a, b, c in strategies],
        projected_vertices=points.tolist(), values=values.tolist(),
        saturating_indices=list(map(int, saturating)),
        local_affine_base_vertex=0,
        face_affine_base_vertex=(int(saturating[0])
                                 if len(saturating) else None),
    )
    return report, proof, points, values, saturating


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report, proof, points, values, saturating = analyze()
    output = args.output or BASE/'results_local_dimension_333'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    output.mkdir(parents=True, exist_ok=False)
    np.savez(
        output/'vertices.npz', projected_vertices=points,
        integer_values=values, saturating_indices=saturating,
    )
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'exact_certificate.json').write_text(
        json.dumps(proof, indent=2)+'\n'
    )
    print(json.dumps(report, indent=2), flush=True)
    print('Saved exact vertices and face data to', output, flush=True)


if __name__ == '__main__':
    main()
