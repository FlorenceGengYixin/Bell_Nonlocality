"""Alternate HI separation and quantum see-saw in [22][4444][22].

Imports the validated model and SDP updates from tripartite_seesaw_4444.
Farkas extraction optimizes its dual L1 norm, not directly the projected
coefficient L1 gap. All valid candidates are retained, including the seed.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import cvxpy as cp
from tripartite_seesaw_4444 import (
    LABELS, build_hi_system, projected_point, extract_separator,
    witness_support, validate_strategy, perturb_strategy, random_strategy,
    bell_value, effective_operators, normalize_povm, DIMS, SETTINGS, OUTCOMES,
    optimize_state,
)

BASE = Path(__file__).resolve().parent


def optimize_party(rho, coeff, parties, party):
    F = effective_operators(rho, coeff, parties, party)
    d, m, o = DIMS[party], SETTINGS[party], OUTCOMES[party]
    E = [[cp.Variable((d, d), hermitian=True) for _ in range(o)] for _ in range(m)]
    constraints = [X >> 0 for setting in E for X in setting]
    constraints += [sum(setting) == np.eye(d) for setting in E]
    objective = sum(cp.real(cp.trace(E[x][a] @ F[x, a])) for x in range(m) for a in range(o))
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    except cp.error.SolverError:
        pass
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        problem.solve(solver=cp.SCS, eps=1e-9, max_iters=50000, time_limit_secs=10)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SDP failed: {problem.status}")
    return np.asarray([normalize_povm([X.value for X in setting]) for setting in E])


def run_seesaw(coeff, rho, A, B, C, max_sweeps=80, tol=1e-10):
    parties = [np.array(p, copy=True) for p in (A, B, C)]
    previous = bell_value(rho, coeff, *parties)
    history = [previous]
    converged, rejected = False, False
    for _ in range(max_sweeps):
        if _ % 10 == 0:
            print(f'  inner sweep {_}, value={previous:.12g}', flush=True)
        before = [p.copy() for p in parties]
        rho_before = rho.copy()
        rho, psi, _ = optimize_state(coeff, *parties)
        for party in range(3):
            parties[party] = optimize_party(rho, coeff, parties, party)
        rho, psi, value = optimize_state(coeff, *parties)
        validate_strategy(psi, parties)
        if value > previous + 1e-10:
            parties, rho = before, rho_before
            _, vectors = np.linalg.eigh(rho)
            psi = vectors[:, -1]
            value, rejected = previous, True
            break
        history.append(value)
        if abs(previous - value) < tol:
            converged = True
            break
        previous = value
    return dict(rho=rho, psi=psi, A=parties[0], B=parties[1], C=parties[2],
                value=value, history=np.asarray(history), converged=converged,
                rejected_increasing_sweep=rejected, sweeps=len(history)-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', type=Path)
    parser.add_argument('--restarts', type=int, default=4)
    parser.add_argument('--outer', type=int, default=4)
    parser.add_argument('--max-sweeps', type=int, default=80)
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--witness-strength', type=float, default=1e-4)
    parser.add_argument('--perturbation', type=float, default=.001)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.restarts, args.outer, args.max_sweeps) < 1 or args.tol <= 0:
        parser.error('Counts and tolerance must be positive')
    if not 0 <= args.perturbation <= 1:
        parser.error('Perturbation must lie in [0,1]')
    if not 0 <= args.witness_strength <= 1:
        parser.error('Witness strength must lie in [0,1]')
    path = args.strategy
    if path is None:
        files = list((BASE/'results_seesaw_4444').glob('*/best.npz'))
        if not files:
            parser.error('Supply a saved four-dimensional strategy')
        path = max(files, key=lambda p: p.stat().st_mtime_ns)
    with np.load(path, allow_pickle=False) as data:
        state = data['state'].copy()
        parties = [data[k].copy() for k in 'ABC']
        base_c = data['coefficients'].copy()
        base_L = float(data['HI_bound'])
    if state.shape != (16,) or [p.shape for p in parties] != [
        (2,2,2,2), (4,4,4,4), (2,2,2,2)
    ]:
        raise ValueError('Expected a 2 x 4 x 2 strategy')
    if base_c.shape != (64,):
        raise ValueError('Expected 64 coefficients')
    norm = np.sum(abs(base_c))
    if norm <= 0:
        raise ValueError('Zero witness')
    base_c, base_L = base_c/norm, base_L/norm
    validate_strategy(state, parties)
    known = state, parties
    system = build_hi_system()
    support = witness_support(system, base_c)
    if abs(support-base_L) > 1e-8:
        raise ValueError('Seed witness support does not match HI bound')
    output = args.output or BASE/'results_full_seesaw_4444'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    (output/'labels.json').write_text(json.dumps(LABELS, indent=2)+'\n')
    rng = np.random.default_rng(args.seed)
    y3 = np.array([label[0] == 'PB' and label[2] == 3 or
                   label[0] == 'PAB' and label[4] == 3 or
                   label[0] == 'PBC' and label[3] == 3 for label in LABELS])
    records, best = [], -np.inf

    def save(tag, psi, P, c, L, extra):
        nonlocal best
        diagnostics = validate_strategy(psi, P)
        point = projected_point(np.outer(psi, psi.conj()), *P)
        value = float(c@point)
        if abs(value-bell_value(np.outer(psi, psi.conj()), c, *P)) > 1e-12:
            raise RuntimeError('Independent objective evaluations disagree')
        gap = float(L-value)
        payload = dict(state=psi, A=P[0], B=P[1], C=P[2], point=point,
                       coefficients=c, HI_bound=L)
        report = dict(normalized_gap=gap, value=value, bound=float(L),
                      y3_coefficient_l1=float(np.sum(abs(c[y3]))),
                      seed=args.seed, source_strategy=str(path.resolve()),
                      source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      **diagnostics, **extra)
        np.savez(output/(tag+'.npz'), **payload)
        if gap > best:
            checked = witness_support(system, c)
            if L > checked+1e-8:
                raise RuntimeError('Invalid candidate HI bound')
            report['independent_HI_support'] = checked
            best = gap
            np.savez(output/'best.npz', **payload)
            (output/'best.json').write_text(json.dumps(report, indent=2)+'\n')
        return gap

    baseline = save('initial', *known, base_c, base_L, dict(stage='seed'))
    for restart in range(args.restarts):
        if restart == 0:
            psi, P = known[0].copy(), [p.copy() for p in known[1]]
            kind = 'unperturbed_seed'
        else:
            psi, P = perturb_strategy(known, rng, strength=args.perturbation)
            _, random_P = random_strategy(rng)
            P[1][3] = random_P[1][3]
            kind = 'perturbed_seed_independent_B3'
        point = projected_point(np.outer(psi, psi.conj()), *P)
        separator, status = extract_separator(system, point)
        c, L = (separator[:2] if separator is not None else (base_c.copy(), base_L))
        if restart > 0:
            direction = np.zeros(64)
            direction[y3] = rng.normal(size=y3.sum())
            direction /= np.sum(abs(direction))
            c = c + args.witness_strength*direction
            c /= np.sum(abs(c))
            L = witness_support(system, c)
        for outer in range(args.outer):
            result = run_seesaw(c, np.outer(psi, psi.conj()), *P,
                               max_sweeps=args.max_sweeps, tol=args.tol)
            psi, P = result['psi'], [result[k] for k in 'ABC']
            if np.max(np.diff(result['history']), initial=0) > 1.1e-10:
                raise RuntimeError('Increasing inner objective')
            tag = f'r{restart:03d}_o{outer:03d}'
            gap = save(tag, psi, P, c, L, dict(stage='quantum', restart=restart, outer=outer))
            np.save(output/('history_'+tag+'.npy'), result['history'])
            point = projected_point(np.outer(psi, psi.conj()), *P)
            separator, status = extract_separator(system, point)
            record = dict(restart=restart, outer=outer, start=kind,
                          sweeps=result['sweeps'], converged=result['converged'],
                          rejected_increasing_sweep=result['rejected_increasing_sweep'],
                          fixed_gap=gap, fixed_y3_l1=float(np.sum(abs(c[y3]))),
                          membership_status=status)
            change = np.inf
            if separator is not None:
                new_c, new_L, y = separator
                np.save(output/('farkas_'+tag+'.npy'), y)
                new_gap = save('separator_'+tag, psi, P, new_c, new_L,
                               dict(stage='separation', restart=restart, outer=outer))
                record.update(separator_gap=new_gap,
                              separator_y3_l1=float(np.sum(abs(new_c[y3]))))
                change = max(np.max(abs(new_c-c)), abs(new_L-L))
                c, L = new_c, new_L
            records.append(record)
            print(json.dumps(record), flush=True)
            (output/'iterations.json').write_text(json.dumps(records, indent=2)+'\n')
            if separator is None or (change < 1e-8 and (result['converged'] or result['rejected_increasing_sweep'])):
                break
    summary = dict(baseline_gap=baseline, best_gap=best, ratio=best/baseline,
                   restarts=args.restarts, outer_limit=args.outer,
                   max_sweeps=args.max_sweeps, tol=args.tol,
                   iterations=len(records), output=str(output),
                   witness_strength=args.witness_strength, perturbation=args.perturbation,
                   caveat='Local search; Farkas dual normalization differs from projected coefficient normalization')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
