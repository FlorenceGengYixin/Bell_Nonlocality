"""Sample 61 quantum behaviors and compare independent HI separators.

Baseline plus 40 perturbed strategies and 20 unoptimised random strategies.
Both the state and POVMs are perturbed by default, matching multipoint_324.
Set --fixed-state to perturb POVMs only. Direct normalized coefficient and
bound agreement is tested, without quotienting by relabellings. No claim
is made about all nearby points, all optimal duals or global uniqueness.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import tripartite_seesaw_324 as model

BASE = Path(__file__).resolve().parent
SCALES = (1e-5, 1e-4, 1e-3, 1e-2, .05, .1, .2, .5)


def random_state(rng):
    psi = rng.normal(size=model.DIM)+1j*rng.normal(size=model.DIM)
    return psi/np.linalg.norm(psi)


def random_povm(rng, dim, outcomes):
    effects = []
    for _ in range(outcomes):
        X = rng.normal(size=(dim, dim))+1j*rng.normal(size=(dim, dim))
        effects.append(X @ X.conj().T)
    return model.normalize_povm(effects)


def random_parties(rng):
    counts = ((2, 2), (3, 2, 4), (2, 2))
    return [[random_povm(rng, d, o) for o in outcomes]
            for d, outcomes in zip(model.DIMS, counts)]


def perturb(rng, psi, parties, epsilon, fixed_state):
    new_psi = psi.copy()
    if not fixed_state:
        new_psi += epsilon*random_state(rng)
        new_psi /= np.linalg.norm(new_psi)
    random_P = random_parties(rng)
    new_P = [[(1-epsilon)*E+epsilon*R for E, R in zip(P, Q)]
             for P, Q in zip(parties, random_P)]
    return new_psi, new_P


def latest_strategy():
    files = list((BASE/'results_seesaw_324').glob('*/best.npz'))
    if not files:
        raise FileNotFoundError('Run tripartite_seesaw_324.py first')
    def gap(path):
        with np.load(path, allow_pickle=False) as d:
            return float(d['HI_bound']-d['coefficients'] @ d['point'])
    return max(files, key=gap)


def same_witness(c, L, other_c, other_L, tol):
    return max(float(abs(c-other_c).max()), abs(L-other_L)) < tol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', type=Path)
    parser.add_argument('--perturbed', type=int, default=40)
    parser.add_argument('--random', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--fixed-state', action='store_true')
    parser.add_argument('--dedup-tol', type=float, default=1e-7)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.perturbed, args.random) < 0 or args.dedup_tol <= 0:
        parser.error('Invalid counts or comparison tolerance')
    path = args.strategy or latest_strategy()
    with np.load(path, allow_pickle=False) as d:
        psi = d['state'].copy()
        parties = [d['A'].copy(),
                   [d[f'B_y{y}'].copy() for y in range(3)], d['C'].copy()]
    dim = parties[1][0].shape[-1]
    model.DIMS = (2, dim, 2)
    model.DIM = 4*dim
    model.IDENTITIES = [np.eye(d) for d in model.DIMS]
    model.validate_strategy(psi, parties)
    system = model.build_hi_system()
    Aeq, constants, R = system
    reference_c, reference_L = model.explicit_T()/22, -1/11
    rng = np.random.default_rng(args.seed)
    samples = [('baseline', None, psi, parties)]
    for i in range(args.perturbed):
        epsilon = SCALES[i % len(SCALES)]
        new_psi, new_P = perturb(
            rng, psi, parties, epsilon, args.fixed_state,
        )
        samples.append((f'perturbed_{i:03d}', epsilon, new_psi, new_P))
    for i in range(args.random):
        samples.append((f'random_{i:03d}', None,
                        random_state(rng), random_parties(rng)))
    output = args.output or BASE/'results_check_324_stability'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    output.mkdir(parents=True, exist_ok=False)
    rows, unique, certificates = [], [], []
    max_residual, min_eigenvalue = 0., 1.
    for i, (name, epsilon, state, P) in enumerate(samples):
        diag = model.validate_strategy(state, P)
        max_residual = max(max_residual, diag['max_povm_residual'])
        min_eigenvalue = min(min_eigenvalue, diag['min_effect_eigenvalue'])
        rho = np.outer(state, state.conj())
        point = model.projected_point(rho, P)
        rhs = constants+R @ point
        member = linprog(
            np.zeros(Aeq.shape[1]), A_eq=Aeq, b_eq=rhs,
            bounds=(0, None), method='highs', options=model.LP_OPTIONS,
        )
        row = dict(
            index=i, source=name, epsilon=epsilon,
            membership_status=member.status, candidate_class=None,
            reference_value=float(reference_c @ point),
            reference_gap=float(reference_L-reference_c @ point),
            separator_gap=None, matches_reference=None,
        )
        if member.success:
            error = float(abs(Aeq @ member.x-rhs).max())
            if error > 1e-8 or member.x.min() < -1e-8:
                raise RuntimeError('Invalid membership primal solution')
            row['membership'] = 'feasible'
            row['membership_residual'] = error
        elif member.status == 2:
            row['membership'] = 'infeasible'
            separator, status = model.extract_separator(system, point)
            row['certificate_status'] = status
            if separator is not None:
                c, L, y = separator
                support = model.witness_support(system, c)
                if abs(support-L) > 1e-8:
                    raise RuntimeError('Independent support check failed')
                gap = float(L-c @ point)
                if gap <= 1e-9:
                    raise RuntimeError('Separator does not resolve violation')
                candidate = next((j for j, (old_c, old_L) in enumerate(unique)
                                  if same_witness(c, L, old_c, old_L,
                                                  args.dedup_tol)), None)
                if candidate is None:
                    candidate = len(unique)
                    unique.append((c.copy(), float(L)))
                comparison = model.compare_witness(c, L, model.explicit_T(), -2)
                row.update(
                    candidate_class=candidate, separator_gap=gap,
                    matches_reference=comparison['equal_within_tolerance'],
                    max_coefficient_error=comparison[
                        'max_normalized_coefficient_error'
                    ],
                    bound_error=comparison['normalized_bound_error'],
                )
                np.savez(
                    output/f'certificate_{i:03d}.npz', coefficients=c,
                    HI_bound=L, farkas_y=y, point=point,
                )
                certificates.append(row.copy())
        else:
            row['membership'] = 'unresolved'
            row['message'] = member.message
        payload = dict(state=state, A=np.asarray(P[0]), C=np.asarray(P[2]),
                       point=point)
        payload.update({f'B_y{y}': By for y, By in enumerate(P[1])})
        np.savez(output/f'point_{i:03d}.npz', **payload)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(
        source_strategy=str(path.resolve()), seed=args.seed,
        total_points=len(samples), perturbed=args.perturbed, random=args.random,
        perturbation_scales=SCALES,
        perturb_state=not args.fixed_state,
        state_perturbation='normalize(psi + epsilon*random pure state)',
        measurement_perturbation='(1-epsilon)*E + epsilon*random Wishart POVM',
        feasible=sum(r['membership'] == 'feasible' for r in rows),
        infeasible=sum(r['membership'] == 'infeasible' for r in rows),
        unresolved=sum(r['membership'] == 'unresolved' for r in rows),
        valid_certificates=len(certificates), unique_direct_separators=len(unique),
        all_certificates_match_reference=bool(certificates) and all(
            r['matches_reference'] for r in certificates
        ),
        infeasible_random=sum(r['membership'] == 'infeasible'
                              and r['source'].startswith('random') for r in rows),
        violating_perturbation_scales=sorted(set(
            r['epsilon'] for r in certificates if r['epsilon'] is not None
        )),
        max_povm_residual=max_residual, min_effect_eigenvalue=min_eigenvalue,
        dedup_tolerance=args.dedup_tol,
        limitation='Finite sample and one solver-selected separator per point; '
                   'does not establish uniqueness of optimal duals, stability '
                   'on a neighborhood, or absence of other HI inequalities',
        points=rows,
    )
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    with (output/'points.csv').open('w', newline='') as f:
        fields = sorted(set().union(*(r.keys() for r in rows)))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in report.items() if k != 'points'},
                     indent=2), flush=True)
    print('Saved to', output, flush=True)


if __name__ == '__main__':
    main()
