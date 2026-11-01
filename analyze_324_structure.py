"""Test standard Bob outcome liftings of paper Eq.(4) against the 444 witness.

Each map is a surjection from four new outcomes to three original outcomes.
Compare affine Bell functionals, including the bound, modulo normalization
and marginal identities. The residual uses a specified Euclidean quotient
metric; it is not an invariant physical distance. Alice/Charlie relabellings
and general transformations are outside this search.
"""
import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import tripartite_seesaw_444 as model
from hi_membership_benchmark import LABELS as OLD_LABELS, paper_eq4

BASE = Path(__file__).resolve().parent
FULL_LABELS = [label for label in model.LABELS
               if label[0] in ('PA', 'PC')]
for y in range(3):
    for b in range(4):
        FULL_LABELS.append(('PB', b, y))
for x in range(2):
    for y in range(3):
        for b in range(4):
            FULL_LABELS.append(('PAB', 0, b, x, y))
for y in range(3):
    for z in range(2):
        for b in range(4):
            FULL_LABELS.append(('PBC', b, 0, y, z))
INDEX = {label: i+1 for i, label in enumerate(FULL_LABELS)}
MAPS = [p for p in itertools.product(range(3), repeat=4)
        if len(set(p)) == 3]
assert len(MAPS) == 36 and len(FULL_LABELS) == 64


def quotient_basis():
    rows = []
    for y in range(3):
        row = np.zeros(65)
        row[0] = -1
        for b in range(4):
            row[INDEX[('PB', b, y)]] = 1
        rows.append(row)
    for kind in ('PAB', 'PBC'):
        for x in range(2):
            for y in range(3):
                row = np.zeros(65)
                marginal = 'PA' if kind == 'PAB' else 'PC'
                row[INDEX[(marginal, 0, x)]] = -1
                for b in range(4):
                    label = (('PAB', 0, b, x, y) if kind == 'PAB'
                             else ('PBC', b, 0, y, x))
                    row[INDEX[label]] = 1
                rows.append(row)
    E = np.asarray(rows)
    _, values, vh = np.linalg.svd(E, full_matrices=True)
    rank = int(sum(values > 1e-10))
    Q = vh[rank:].T
    assert np.max(abs(E @ Q)) < 1e-12
    return Q, rank


def lift_parts():
    old, bound = paper_eq4()
    global_part = np.zeros(65)
    global_part[0] = -bound
    pieces = np.zeros((3, 3, 36, 65))
    for alpha, label in zip(old, OLD_LABELS):
        if not alpha:
            continue
        kind = label[0]
        if kind in ('PA', 'PC'):
            global_part[INDEX[label]] += alpha
            continue
        old_y = label[2] if kind == 'PB' else label[4]
        if kind == 'PBC':
            old_y = label[3]
        old_b = label[2] if kind == 'PAB' else label[1]
        for new_y in range(3):
            for k, mapping in enumerate(MAPS):
                for new_b, image in enumerate(mapping):
                    if image != old_b:
                        continue
                    if kind == 'PB':
                        target = ('PB', new_b, new_y)
                    elif kind == 'PAB':
                        target = ('PAB', 0, new_b, label[3], new_y)
                    else:
                        target = ('PBC', new_b, 0, new_y, label[4])
                    pieces[old_y, new_y, k, INDEX[target]] += alpha
    return global_part, pieces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', type=Path)
    parser.add_argument('--tol', type=float, default=1e-9)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.tol <= 0:
        parser.error('Tolerance must be positive')
    path = args.strategy
    if path is None:
        files = list((BASE/'results_full_seesaw_444').glob('*/best.npz'))
        if not files:
            parser.error('Supply --strategy with a complete 444 NPZ')
        def gap(p):
            with np.load(p, allow_pickle=False) as d:
                return float(d['HI_bound']-d['coefficients'] @ d['point'])
        path = max(files, key=gap)
    with np.load(path, allow_pickle=False) as d:
        c, bound = d['coefficients'], float(d['HI_bound'])
    if c.shape != (49,):
        raise ValueError('Expected 49 coordinates of a 444 inequality')
    target = np.zeros(65)
    target[0] = -bound
    for alpha, label in zip(c, model.LABELS):
        target[INDEX[label]] += alpha
    Q, rank = quotient_basis()
    projected = target @ Q
    norm = np.linalg.norm(projected)
    if norm < 1e-14:
        raise ValueError('Zero affine functional in quotient')
    global_part, pieces = lift_parts()
    global_q, pieces_q = global_part @ Q, pieces @ Q
    combinations = np.array(list(itertools.product(range(36), repeat=3)))
    best, matches, checked = None, [], 0
    for perm in itertools.permutations(range(3)):
        candidates = global_q.copy()[None, :]
        candidates = candidates + pieces_q[0, perm[0], combinations[:, 0]]
        candidates += pieces_q[1, perm[1], combinations[:, 1]]
        candidates += pieces_q[2, perm[2], combinations[:, 2]]
        denominator = np.einsum('ij,ij->i', candidates, candidates)
        alpha = (candidates @ projected)/denominator
        positive = alpha > 0
        # Explicit differences avoid cancellation near a true match.
        delta = projected-alpha[:, None]*candidates
        residual = np.linalg.norm(delta, axis=1)/norm
        residual[~positive] = np.inf
        k = int(np.argmin(residual))
        if best is None or residual[k] < best['relative_residual']:
            best = dict(
                settings=perm,
                maps=[MAPS[i] for i in combinations[k]],
                positive_scale=float(alpha[k]),
                relative_residual=float(residual[k]),
            )
        for k in np.flatnonzero(residual < args.tol):
            matches.append(dict(
                settings=perm, maps=[MAPS[i] for i in combinations[k]],
                positive_scale=float(alpha[k]),
                relative_residual=float(residual[k]),
            ))
        checked += len(combinations)
        print('Checked', checked, 'of 279936', flush=True)
    assert checked == 279936
    # Positive control: the quotient test recognises a known scaled lift.
    known = global_part+sum(pieces[y, y, 0] for y in range(3))
    qknown = known @ Q
    identity_shift = np.zeros(65)
    identity_shift[0] = -1
    for b in range(4):
        identity_shift[INDEX[('PB', b, 0)]] = 1
    assert np.linalg.norm(identity_shift @ Q) < 1e-12
    test_target = (2*known+3*identity_shift) @ Q
    fitted = float(test_target @ qknown/(qknown @ qknown))
    assert fitted > 0 and abs(fitted-2) < 1e-12
    assert np.linalg.norm(test_target-fitted*qknown) < 1e-12
    assert float((-qknown) @ qknown/(qknown @ qknown)) < 0
    report = dict(
        source_strategy=str(path.resolve()),
        reference='paper Eq.(4), integer scale with bound -2',
        target_scenario='[22][444][22]', target_normalized_bound=bound,
        maps_per_setting=36, setting_permutations=6,
        candidate_count=checked, equality_rank=rank,
        quotient_dimension=Q.shape[1], tolerance=args.tol,
        matches=matches, match_count=len(matches), closest=best,
        metric='Orthogonal Euclidean projection of [constant, PA, PC, '
               'all PB, PAB, PBC coefficients] onto the complement of '
               'normalization/marginal identities; best positive scale',
        limitation='Only Bob surjective outcome liftings and setting '
                   'permutations; no Alice/Charlie relabelling search '
                   'or general equivalence classification',
    )
    output = args.output or BASE/'results_analyze_324_structure'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    output.mkdir(parents=True, exist_ok=False)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez(output/'quotient.npz', basis=Q, target=target)
    print(json.dumps(report, indent=2), flush=True)
    print('Saved to', output, flush=True)


if __name__ == '__main__':
    main()
