"""Numerically certified NPA outer bounds for T/22 in [22][324][22].

A,C are dichotomic observables (P0=(I+A)/2); Bob uses projectors with
outcome counts 3,2,4. Projectivity is without loss for dimension-unbounded
quantum bounds, not a requirement on the saved qutrit POVMs.

Use conditional Bob principal moment blocks <Wi* B_b|y Wj> with shared
AC moments. This is a valid NPA-type outer relaxation, NOT a full standard
numbered level. Real moments suffice by averaging a representation with its complex
conjugate, preserving this real Bell objective. Raw SDP primal values
are not certified quantum bounds.
A PSD-projected dual plus moment residual correction supplies a numerical
lower bound. The certificate is floating point, not interval arithmetic.
"""
import argparse
import itertools
import json
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import numpy as np
import cvxpy as cp
from scipy import sparse
import tripartite_seesaw_324 as model

BASE = Path(__file__).resolve().parent
BOB_OUTCOMES = (3, 2, 4)


@lru_cache(None)
def canonical(word):
    out = []
    for party in 'ABC':
        stack = []
        for op in word:
            if op[0] != party:
                continue
            if stack and stack[-1][1] == op[1]:
                if stack[-1][2] != op[2]:
                    return None
                if party in 'AC':
                    stack.pop()
                continue
            stack.append(op)
        out.extend(stack)
    return tuple(out)


def moment_key(word):
    reduced = canonical(word)
    if reduced is None:
        return None
    return min(reduced, canonical(tuple(reversed(reduced))))


def ac_words(depth):
    pools = []
    for party in 'AC':
        ops = [(party, x, 0) for x in range(2)]
        words = {()}
        for length in range(1, depth+1):
            words.update(canonical(w) for w in itertools.product(ops,
                                                               repeat=length))
        pools.append(words)
    return sorted({a+c for a, c in itertools.product(*pools)},
                  key=lambda w: (len(w), w))


def label_word(label):
    kind = label[0]
    if kind in ('PA', 'PB', 'PC'):
        return ((kind[1], label[2], label[1]),)
    if kind == 'PAB':
        _, a, b, x, y = label
        return (('A', x, a), ('B', y, b))
    _, b, c, y, z = label
    return (('B', y, b), ('C', z, c))


def objective_vector(keys):
    objective = np.zeros(len(keys))
    for coefficient, label in zip(model.explicit_T(), model.LABELS):
        if not coefficient:
            continue
        expansion = {(): 1.}
        for op in label_word(label):
            terms = [((), .5), ((op,), .5)] if op[0] in 'AC' else [((op,), 1.)]
            expansion = {w+t: v*a for w, v in expansion.items()
                         for t, a in terms}
        for word, factor in expansion.items():
            objective[keys[moment_key(word)]] += coefficient*factor
    return objective


def q2_words():
    singles = [(p, x, 0) for p in 'AC' for x in range(2)]
    singles += [('B', y, b) for y, o in enumerate(BOB_OUTCOMES)
                for b in range(o-1)]
    words = {()}
    for length in (1, 2):
        words.update(w for seq in itertools.product(singles, repeat=length)
                     if (w := canonical(seq)) is not None)
    return sorted(words, key=lambda w: (len(w), w))


def build_relaxation(depth, add_q2=False):
    words = ac_words(depth)
    n = len(words)
    keys = {(): 0}
    entries = []
    markers = [None]+[('B', y, b) for y, o in enumerate(BOB_OUTCOMES)
                      for b in range(o-1)]
    for marker in markers:
        rows, cols = [], []
        for j, right in enumerate(words):
            for i, left in enumerate(words):
                word = tuple(reversed(left))
                word += () if marker is None else (marker,)
                key = moment_key(word+right)
                if key is None:
                    continue
                if key not in keys:
                    keys[key] = len(keys)
                rows.append(i+n*j)
                cols.append(keys[key])
        entries.append((rows, cols))
    maps = [sparse.csc_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n*n, len(keys)),
    ) for rows, cols in entries]
    blocks, offset = [], 1
    for outcomes in BOB_OUTCOMES:
        explicit = maps[offset:offset+outcomes-1]
        offset += outcomes-1
        blocks.extend(explicit+[maps[0]-sum(explicit)])
    if add_q2:
        full = q2_words()
        size = len(full)
        rows, cols = [], []
        for j, right in enumerate(full):
            for i, left in enumerate(full):
                key = moment_key(tuple(reversed(left))+right)
                if key is None:
                    continue
                if key not in keys:
                    keys[key] = len(keys)
                rows.append(i+size*j)
                cols.append(keys[key])
        blocks = [sparse.hstack([
            M, sparse.csc_matrix((M.shape[0], len(keys)-M.shape[1])),
        ], format='csc') for M in blocks]
        blocks.append(sparse.csc_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(size*size, len(keys)),
        ))
    return words, keys, blocks, objective_vector(keys)


def corrected_bound(maps, objective, duals):
    adjoint = np.zeros(len(objective))
    projected = []
    for M, Z in zip(maps, duals):
        n = Z.shape[0]
        Z = (Z+Z.T)/2
        eigenvalues, U = np.linalg.eigh(Z)
        margin = 1e-12*max(1., float(np.linalg.norm(Z, 2)))
        Z = (U*np.maximum(eigenvalues, 0)) @ U.T+margin*np.eye(n)
        projected.append(Z)
        adjoint += np.asarray(M.T @ Z.reshape(-1, order='F')).ravel()
    residual = objective-adjoint
    penalty = float(sum(abs(residual[1:])))
    lower = float(residual[0]-penalty)
    return lower, projected, residual, penalty


def solve(depth, solver, tol, max_iters, verbose, time_limit=180,
          add_q2=False):
    start = time.perf_counter()
    words, keys, maps, objective = build_relaxation(depth, add_q2)
    n = len(words)
    moments = cp.Variable(len(keys))
    sizes = [int(round(np.sqrt(M.shape[0]))) for M in maps]
    cones = [cp.reshape(M @ moments, (size, size), order='F') >> 0
             for M, size in zip(maps, sizes)]
    normalization = moments[0] == 1
    problem = cp.Problem(cp.Minimize(objective @ moments),
                         cones+[normalization])
    print(f'Depth {depth}: PSD sizes {sizes}, '
          f'{len(keys)} shared moments', flush=True)
    if solver == 'CLARABEL':
        problem.solve(
            solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol,
            tol_feas=tol, max_iter=max_iters, max_threads=1,
            time_limit=time_limit, verbose=verbose,
        )
    else:
        problem.solve(
            solver=cp.SCS, eps=tol, max_iters=max_iters,
            time_limit_secs=time_limit, use_indirect=True, verbose=verbose,
        )
    if moments.value is None or any(c.dual_value is None for c in cones):
        raise RuntimeError('No primal/dual solution: '+str(problem.status))
    lower, duals, residual, penalty = corrected_bound(
        maps, objective, [np.asarray(c.dual_value) for c in cones],
    )
    grams = [np.asarray(M @ moments.value).reshape(size, size, order='F')
             for M, size in zip(maps, sizes)]
    primal = float(objective @ moments.value)
    dual = -float(normalization.dual_value)
    report = dict(
        scenario='[22][324][22]', local_AC_word_depth=depth,
        basis='AC words with local lengths at most (d,d); conditional '
              'Bob principal blocks, not a full numbered NPA level',
        containing_full_NPA_level=2*depth+1,
        words_per_block=n, PSD_blocks=len(maps), moments=len(keys),
        PSD_block_sizes=sizes, full_Q2_added=add_q2,
        solver=solver, status=problem.status,
        raw_T_primal=primal, raw_T_solver_dual=dual,
        raw_T_corrected_lower=lower, normalized_corrected_lower=lower/22,
        raw_primal_dual_gap=primal-dual,
        raw_dual_residual_l1_penalty=penalty,
        raw_dual_residual_max=float(abs(residual[1:]).max()),
        min_primal_block_eigenvalue=min(float(np.linalg.eigvalsh(G).min())
                                       for G in grams),
        min_corrected_dual_eigenvalue=min(float(np.linalg.eigvalsh(Z).min())
                                         for Z in duals),
        normalization_residual=float(abs(moments.value[0]-1)),
        iterations=int(problem.solver_stats.num_iters),
        seconds=time.perf_counter()-start,
        numerical_certificate='PSD projection with 1e-12 relative safety '
                              'margin; residual correction using |moment|<=1',
    )
    arrays = dict(
        moments=moments.value, objective=objective, residual=residual,
        depth=depth, add_q2=add_q2,
    )
    for i, (G, Z) in enumerate(zip(grams, duals)):
        arrays[f'primal_gram_{i}'] = G
        arrays[f'dual_Z_{i}'] = Z
    return report, arrays, words, keys


def strategy_upper(path):
    with np.load(path, allow_pickle=False) as data:
        psi = data['state'].copy()
        parties = [data['A'].copy(), [data[f'B_y{y}'].copy() for y in range(3)],
                   data['C'].copy()]
    dim = parties[1][0].shape[-1]
    model.DIMS, model.DIM = (2, dim, 2), 4*dim
    model.IDENTITIES = [np.eye(d) for d in model.DIMS]
    diagnostics = model.validate_strategy(psi, parties)
    rho = np.outer(psi, psi.conj())
    point = model.projected_point(rho, parties)
    value = float(model.explicit_T() @ point/22)
    if abs(value-model.bell_value(rho, model.explicit_T()/22, *parties)) > 1e-12:
        raise RuntimeError('Strategy objective checks disagree')
    return value, diagnostics


def self_test():
    from npa_333 import chsh_self_test
    result = chsh_self_test()
    assert canonical((('A', 0, 0), ('A', 1, 0))) != canonical(
        (('A', 1, 0), ('A', 0, 0))
    )
    assert canonical((('B', 2, 0), ('B', 2, 2))) is None
    assert canonical((('A', 0, 0), ('B', 1, 0), ('A', 0, 0))) == (
        ('B', 1, 0),
    )
    return result


def verify_certificate(path):
    with np.load(path, allow_pickle=False) as data:
        depth = int(data['depth'])
        add_q2 = bool(data['add_q2']) if 'add_q2' in data else False
        _, _, maps, objective = build_relaxation(depth, add_q2)
        residual = objective.copy()
        minimum = float('inf')
        for i, M in enumerate(maps):
            Z = data['dual_Z'][i] if 'dual_Z' in data else data[f'dual_Z_{i}']
            eigenvalue = float(np.linalg.eigvalsh((Z+Z.T)/2).min())
            if eigenvalue < 0:
                raise RuntimeError('Stored dual is not numerically positive')
            minimum = min(minimum, eigenvalue)
            residual -= np.asarray(M.T @ Z.reshape(-1, order='F')).ravel()
        lower = float(residual[0]-sum(abs(residual[1:])))
    return dict(
        certificate_file=str(path.resolve()), local_AC_word_depth=depth,
        full_Q2_added=add_q2, raw_T_corrected_lower=lower,
        normalized_corrected_lower=lower/22,
        min_corrected_dual_eigenvalue=minimum,
        scope='Independent floating-point check, not an exact interval proof',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depths', nargs='+', type=int, default=[3])
    parser.add_argument('--add-q2', action='store_true')
    parser.add_argument('--solver', choices=['CLARABEL', 'SCS'],
                        default='CLARABEL')
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--max-iters', type=int, default=500)
    parser.add_argument('--time-limit', type=float, default=180)
    parser.add_argument('--strategy', type=Path)
    parser.add_argument('--verify', type=Path, help='Recheck a saved SDP NPZ')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--self-test-only', action='store_true')
    args = parser.parse_args()
    if args.self_test_only:
        print(json.dumps(self_test()), flush=True)
        return
    if args.verify is not None:
        print(json.dumps(verify_certificate(args.verify), indent=2))
        return
    if min(args.depths) < 1 or max(args.depths) > 5:
        parser.error('Word depths must lie in [1,5]')
    if args.tol <= 0 or args.max_iters < 1 or args.time_limit <= 0:
        parser.error('Tolerance and iteration limit must be positive')
    path = args.strategy
    if path is None:
        files = list((BASE/'results_seesaw_324').glob('*/best.npz'))
        if files:
            path = min(files, key=lambda p: strategy_upper(p)[0])
    upper, diag = strategy_upper(path) if path else (None, None)
    output = args.output or BASE/'results_npa_324'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for depth in args.depths:
        report, arrays, words, keys = solve(
            depth, args.solver, args.tol, args.max_iters, args.verbose,
            args.time_limit, args.add_q2,
        )
        report.update(strategy_file=str(path) if path else None,
                      achieved_quantum_upper=upper, strategy_checks=diag)
        if upper is not None:
            interval = upper-report['normalized_corrected_lower']
            report['normalized_remaining_interval'] = interval
            if interval < -1e-8:
                raise RuntimeError('NPA lower exceeds a physical strategy value')
            report['tight_numerical_sandwich'] = bool(
                0 <= interval <= 1e-7
                and report['min_corrected_dual_eigenvalue'] >= -1e-10
                and report['normalization_residual'] < 1e-7
            )
        reports.append(report)
        np.savez(output/f'depth_{depth}.npz', **arrays)
        (output/f'words_{depth}.json').write_text(json.dumps(dict(
            AC=words, Q2=q2_words() if args.add_q2 else [],
        ))+'\n')
        (output/f'keys_{depth}.json').write_text(json.dumps(list(keys))+'\n')
        (output/'report.json').write_text(json.dumps(reports, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
    print('Saved to', output, flush=True)


if __name__ == '__main__':
    main()
