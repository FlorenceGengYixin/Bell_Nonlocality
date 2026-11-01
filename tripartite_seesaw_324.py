"""[22][324][22]: affine reduction, HI separation and quantum see-saw.

Start from one complete 444 strategy; do not mix old state and POVM files.
Check the derived affine inequality against the explicit 21-term formula,
then independently extract Farkas inequalities. Exact coefficient equality
and numerical agreement are reported separately. No target is imposed on LP.
"""
import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from tripartite_seesaw_333 import (
    hermitian, normalize_povm, term_indices,
)
from hi_membership_benchmark import (
    find_farkas_certificate, certificate_to_inequality, LP_OPTIONS,
)
import tripartite_seesaw_444 as source444

BASE = Path(__file__).resolve().parent
MA = MC = 2
OA = OC = 2
MB = 3
BOB_OUTCOMES = (3, 2, 4)
MERGES = (((0,), (1,), (2, 3)),
          ((0, 1, 3), (2,)),
          ((0,), (1,), (2,), (3,)))
DIMS = (2, 3, 2)
DIM = 12
IDENTITIES = [np.eye(d) for d in DIMS]
LAMBDA = list(itertools.product(
    itertools.product(range(2), repeat=2), repeat=2,
))
NLAMBDA = 16
OFFSETS = (0, 3, 5)
NW = 144


def make_labels():
    labels = [('PA', 0, x) for x in range(2)]
    labels += [('PB', b, y) for y in range(3)
               for b in range(BOB_OUTCOMES[y] - 1)]
    labels += [('PC', 0, z) for z in range(2)]
    labels += [('PAB', 0, b, x, y) for x in range(2)
               for y in range(3)
               for b in range(BOB_OUTCOMES[y] - 1)]
    labels += [('PBC', b, 0, y, z) for y in range(3)
               for z in range(2)
               for b in range(BOB_OUTCOMES[y] - 1)]
    return labels


LABELS = make_labels()
INDEX = {label: i for i, label in enumerate(LABELS)}
assert len(LABELS) == 34


def w_index(b, y, lam):
    return (OFFSETS[y] + b) * NLAMBDA + lam


def build_hi_system():
    rows = []
    constants = []
    p_coeffs = []

    def add_equation(row, constant=0.0, probability_terms=None):
        pc = np.zeros(len(LABELS))
        if probability_terms is not None:
            for label, value in probability_terms:
                pc[INDEX[label]] += value
        rows.append(row)
        constants.append(constant)
        p_coeffs.append(pc)
    for y in range(MB):
        row = np.zeros(NW)
        for b in range(BOB_OUTCOMES[y]):
            for lam in range(NLAMBDA):
                row[w_index(b, y, lam)] = 1.0
        add_equation(row, constant=1.0)
    for y in range(MB):
        for b in range(BOB_OUTCOMES[y] - 1):
            row = np.zeros(NW)
            for lam in range(NLAMBDA):
                row[w_index(b, y, lam)] = 1.0
            add_equation(row, probability_terms=[(('PB', b, y), 1.0)])
    for x in range(MA):
        for y in range(MB):
            for b in range(BOB_OUTCOMES[y] - 1):
                row = np.zeros(NW)
                for lam, (avec, cvec) in enumerate(LAMBDA):
                    if avec[x] == 0:
                        row[w_index(b, y, lam)] = 1.0
                add_equation(row, probability_terms=[(('PAB', 0, b, x, y), 1.0)])
    for y in range(MB):
        for z in range(MC):
            for b in range(BOB_OUTCOMES[y] - 1):
                row = np.zeros(NW)
                for lam, (avec, cvec) in enumerate(LAMBDA):
                    if cvec[z] == 0:
                        row[w_index(b, y, lam)] = 1.0
                add_equation(row, probability_terms=[(('PBC', b, 0, y, z), 1.0)])
    for x in range(MA):
        for y in range(MB):
            row = np.zeros(NW)
            for b in range(BOB_OUTCOMES[y]):
                for lam, (avec, cvec) in enumerate(LAMBDA):
                    if avec[x] == 0:
                        row[w_index(b, y, lam)] = 1.0
            add_equation(row, probability_terms=[(('PA', 0, x), 1.0)])
    for z in range(MC):
        for y in range(MB):
            row = np.zeros(NW)
            for b in range(BOB_OUTCOMES[y]):
                for lam, (avec, cvec) in enumerate(LAMBDA):
                    if cvec[z] == 0:
                        row[w_index(b, y, lam)] = 1.0
            add_equation(row, probability_terms=[(('PC', 0, z), 1.0)])
    for a in range(OA):
        for c in range(OC):
            for x in range(MA):
                for z in range(MC):
                    for y in range(1, MB):
                        row = np.zeros(NW)
                        for b in range(BOB_OUTCOMES[y]):
                            for lam, (avec, cvec) in enumerate(LAMBDA):
                                if avec[x] == a and cvec[z] == c:
                                    row[w_index(b, y, lam)] += 1.0
                        for b in range(BOB_OUTCOMES[0]):
                            for lam, (avec, cvec) in enumerate(LAMBDA):
                                if avec[x] == a and cvec[z] == c:
                                    row[w_index(b, 0, lam)] -= 1.0
                        add_equation(row)
    return (np.asarray(rows), np.asarray(constants), np.asarray(p_coeffs))

def bell_operator(coeff, A, B, C):
    parties = (A, B, C)
    H = np.zeros((DIM, DIM), dtype=complex)
    for value, label in zip(coeff, LABELS):
        factors = list(IDENTITIES)
        for party, (setting, outcome) in term_indices(label).items():
            factors[party] = parties[party][setting][outcome]
        H += value * np.kron(np.kron(factors[0], factors[1]), factors[2])
    return hermitian(H)

def bell_value(rho, coeff, A, B, C):
    return float(np.trace(rho @ bell_operator(coeff, A, B, C)).real)

def optimize_state(coeff, A, B, C):
    values, vectors = np.linalg.eigh(bell_operator(coeff, A, B, C))
    psi = vectors[:, 0]
    return np.outer(psi, psi.conj()), psi, float(values[0])


def projected_point(rho, parties):
    values = []
    for label in LABELS:
        factors = list(IDENTITIES)
        for party, (x, a) in term_indices(label).items():
            factors[party] = parties[party][x][a]
        operator = np.kron(np.kron(*factors[:2]), factors[2])
        values.append(float(np.trace(rho @ operator).real))
    return np.asarray(values)


def validate_strategy(psi, parties):
    if psi.shape != (DIM,) or abs(np.vdot(psi, psi) - 1) > 1e-8:
        raise ValueError('Invalid state')
    counts = ((2, 2), BOB_OUTCOMES, (2, 2))
    residual, minimum = 0., 1.
    for P, outcomes, d in zip(parties, counts, DIMS):
        if len(P) != len(outcomes):
            raise ValueError('Wrong number of settings')
        for setting, o in zip(P, outcomes):
            if np.shape(setting) != (o, d, d):
                raise ValueError('Wrong POVM dimensions')
            residual = max(residual, float(
                abs(sum(setting) - np.eye(d)).max()
            ))
            for E in setting:
                residual = max(residual, float(abs(E-E.conj().T).max()))
                minimum = min(minimum, float(
                    np.linalg.eigvalsh(hermitian(E)).min()
                ))
    if residual > 1e-8 or minimum < -1e-8:
        raise ValueError('Invalid POVM')
    return dict(max_povm_residual=residual, min_effect_eigenvalue=minimum)


def effective_operators(rho, coeff, parties, party):
    d = DIMS[party]
    F = [np.zeros_like(setting) for setting in parties[party]]
    tensor = rho.reshape(*DIMS, *DIMS)
    contractions = ('abcdef,eb,fc->ad', 'abcdef,da,fc->be',
                    'abcdef,da,eb->cf')
    for alpha, label in zip(coeff, LABELS):
        indices = term_indices(label)
        if party not in indices or abs(alpha) < 1e-15:
            continue
        factors = list(IDENTITIES)
        for j, (x, a) in indices.items():
            if j != party:
                factors[j] = parties[j][x][a]
        fixed = [factors[j] for j in range(3) if j != party]
        local = np.einsum(contractions[party], tensor, *fixed)
        x, a = indices[party]
        F[x][a] += alpha * hermitian(local)
    return F


def optimize_party(rho, coeff, parties, party):
    F = effective_operators(rho, coeff, parties, party)
    d = DIMS[party]
    effects = [[cp.Variable((d, d), hermitian=True) for _ in setting]
               for setting in parties[party]]
    constraints = [E >> 0 for setting in effects for E in setting]
    constraints += [sum(setting) == np.eye(d) for setting in effects]
    objective = sum(
        cp.real(cp.trace(E @ F[x][a]))
        for x, setting in enumerate(effects)
        for a, E in enumerate(setting)
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(
            solver=cp.CLARABEL, tol_gap_abs=1e-10,
            tol_gap_rel=1e-10, tol_feas=1e-10,
        )
    except cp.error.SolverError:
        pass
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        problem.solve(
            solver=cp.SCS, eps=1e-9,
            max_iters=50000, time_limit_secs=10,
        )
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError('SDP unresolved: ' + str(problem.status))
    return [normalize_povm([E.value for E in setting])
            for setting in effects]


def copy_parties(parties):
    return [[np.array(setting, copy=True) for setting in P]
            for P in parties]


def run_seesaw(c, psi, parties, max_sweeps, tol):
    parties = copy_parties(parties)
    rho = np.outer(psi, psi.conj())
    history = [bell_value(rho, c, *parties)]
    converged = rejected = False
    for _ in range(max_sweeps):
        before = psi.copy(), copy_parties(parties)
        rho, psi, _ = optimize_state(c, *parties)
        for party in range(3):
            parties[party] = optimize_party(rho, c, parties, party)
        rho, psi, value = optimize_state(c, *parties)
        validate_strategy(psi, parties)
        if value > history[-1] + 1e-10:
            psi, parties = before
            rejected = True
            break
        history.append(value)
        if abs(history[-1] - history[-2]) < tol:
            converged = True
            break
    return psi, parties, dict(
        history=history, sweeps=len(history)-1,
        converged=converged, rejected_increasing_sweep=rejected,
    )


def witness_support(system, coefficients):
    A,d,R = system
    n = A.shape[1]
    objective = np.r_[np.zeros(n),coefficients]
    E = np.hstack([A,-R])
    result = linprog(
        objective,A_eq=E,b_eq=d,
        bounds=[(0,None)]*n+[(None,None)]*len(LABELS),
        method='highs',options=LP_OPTIONS,
    )
    if not result.success:
        raise RuntimeError('HI witness support failed: '+result.message)
    eta = result.eqlin.marginals
    residual = objective-E.T@eta
    # Every hidden weight and probability coordinate lies in [0,1].
    # Correct the numerical dual objective conservatively for residuals.
    bound = float(d@eta+np.minimum(residual,0).sum())
    return bound


def extract_separator(system, point):
    A,d,R = system
    rhs=d+R@point
    member=linprog(np.zeros(A.shape[1]),A_eq=A,b_eq=rhs,bounds=(0,None),
                   method='highs',options=LP_OPTIONS)
    if member.success:
        return None, 'inside_HI'
    if member.status != 2:
        raise RuntimeError('Membership unresolved: '+member.message)
    _,y=find_farkas_certificate(A,rhs)
    if np.max(A.T@y)>1e-8 or np.sum(abs(y))>1+1e-8:
        raise RuntimeError('Farkas dual feasibility failed')
    if rhs@y<=1e-9:
        return None, 'infeasible_at_LP_tolerance_but_no_resolved_certificate'
    c,L=certificate_to_inequality(y,d,R)
    norm=np.sum(abs(c))
    if norm<1e-14:
        return None, 'zero_separator'
    return (c/norm,L/norm,y), 'resolved_outside_HI'


def affine_reduce(integer444):
    """Coarse-grain identical feature columns, then eliminate final outcomes."""
    old_index = source444.LABEL_TO_INDEX
    result = np.zeros(34, dtype=np.int64)
    for kind in ('PA', 'PC'):
        for x in range(2):
            result[INDEX[(kind, 0, x)]] = integer444[
                old_index[(kind, 0, x)]
            ]
    offset = 0
    for y, groups in enumerate(MERGES):
        profiles = np.zeros((4, 5), dtype=np.int64)
        for b in range(3):
            profiles[b, 0] = integer444[old_index[('PB', b, y)]]
            for x in range(2):
                profiles[b, 1+x] = integer444[
                    old_index[('PAB', 0, b, x, y)]
                ]
                profiles[b, 3+x] = integer444[
                    old_index[('PBC', b, 0, y, x)]
                ]
        for group in groups:
            if any(not np.array_equal(profiles[b], profiles[group[0]])
                   for b in group):
                raise ValueError('Merged outcomes have different coefficients')
        coarse = [profiles[group[0]] for group in groups]
        ref = coarse[-1]
        offset += int(ref[0])
        for x in range(2):
            result[INDEX[('PA', 0, x)]] += ref[1+x]
            result[INDEX[('PC', 0, x)]] += ref[3+x]
        for b, f in enumerate(coarse[:-1]):
            f = f-ref
            result[INDEX[('PB', b, y)]] += f[0]
            for x in range(2):
                result[INDEX[('PAB', 0, b, x, y)]] += f[1+x]
                result[INDEX[('PBC', b, 0, y, x)]] += f[3+x]
    return result, offset


def explicit_T():
    """Independent transcription of the 21-term expression in the text."""
    terms = [
        (-1, ('PA', 0, 0)), (-1, ('PB', 0, 0)),
        (1, ('PB', 1, 0)), (-1, ('PB', 0, 1)),
        (1, ('PC', 0, 0)), (-1, ('PC', 0, 1)),
        (1, ('PAB', 0, 0, 0, 0)), (1, ('PAB', 0, 1, 0, 0)),
        (2, ('PAB', 0, 0, 0, 1)), (1, ('PAB', 0, 0, 1, 0)),
        (-1, ('PAB', 0, 1, 1, 0)), (1, ('PAB', 0, 0, 1, 2)),
        (-1, ('PAB', 0, 2, 1, 2)), (1, ('PBC', 0, 0, 0, 0)),
        (-1, ('PBC', 1, 0, 0, 1)), (-1, ('PBC', 0, 0, 1, 0)),
        (1, ('PBC', 0, 0, 1, 1)), (-1, ('PBC', 0, 0, 2, 0)),
        (-1, ('PBC', 1, 0, 2, 0)), (1, ('PBC', 1, 0, 2, 1)),
        (1, ('PBC', 2, 0, 2, 1)),
    ]
    c = np.zeros(34, dtype=np.int64)
    for value, label in terms:
        c[INDEX[label]] += value
    return c


def compare_witness(c, L, reference, bound, tol=1e-8):
    norm = float(sum(abs(c)))
    cn, Ln = c/norm, L/norm
    rn = reference/sum(abs(reference))
    bn = bound/sum(abs(reference))
    error = max(float(abs(cn-rn).max()), abs(Ln-bn))
    reconstructed = np.rint(22*cn).astype(np.int64)
    reconstructed_bound = int(round(22*Ln))
    rational_error = max(
        float(abs(22*cn-reconstructed).max()),
        abs(22*Ln-reconstructed_bound),
    )
    return dict(
        equal_within_tolerance=bool(error < tol),
        max_normalized_coefficient_error=float(abs(cn-rn).max()),
        normalized_bound_error=abs(Ln-bn),
        exact_integer_reconstruction_matches=bool(
            rational_error < tol
            and np.array_equal(reconstructed, reference)
            and reconstructed_bound == bound
        ),
        integer_reconstruction_error=rational_error,
    )


def source_path():
    files = list((BASE/'results_full_seesaw_444').glob('*/best.npz'))
    files += list((BASE/'results_seesaw_444').glob('*/best.npz'))
    if not files:
        raise FileNotFoundError('Supply --strategy with a complete 444 NPZ')
    def gap(path):
        with np.load(path, allow_pickle=False) as d:
            return float(d['HI_bound'] - d['coefficients'] @ d['point'])
    return max(files, key=gap)


def save_candidate(path, psi, parties, c, L):
    validate_strategy(psi, parties)
    rho = np.outer(psi, psi.conj())
    point = projected_point(rho, parties)
    value = float(c @ point)
    if abs(value-bell_value(rho, c, *parties)) > 1e-12:
        raise RuntimeError('Probability and Bell objectives disagree')
    payload = dict(
        state=psi, A=parties[0], C=parties[2], point=point,
        coefficients=c, HI_bound=L,
    )
    for y, setting in enumerate(parties[1]):
        payload[f'B_y{y}'] = setting
    np.savez(path, **payload)
    return value, point


def main():
    global DIMS, DIM, IDENTITIES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', type=Path, help='Complete 444 NPZ')
    parser.add_argument('--outer', type=int, default=4)
    parser.add_argument('--max-sweeps', type=int, default=60)
    parser.add_argument('--tol', type=float, default=1e-10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.outer < 1 or args.max_sweeps < 1 or args.tol <= 0:
        parser.error('Iteration counts and tolerance must be positive')
    path = args.strategy or source_path()
    with np.load(path, allow_pickle=False) as data:
        psi = data['state'].copy()
        A, B, C = [data[k].copy() for k in 'ABC']
        c444 = data['coefficients'].copy()
        L444 = float(data['HI_bound'])
    source444.configure(B.shape[-1])
    source444.validate_strategy(psi, [A, B, C])
    DIMS, DIM = source444.DIMS, source444.DIM
    IDENTITIES = [np.eye(d) for d in DIMS]
    if B.shape[:2] != (3, 4) or c444.shape != (49,):
        raise ValueError('Expected a three-setting four-outcome source')
    norm = float(sum(abs(c444)))
    c444, L444 = c444/norm, L444/norm
    ints = np.rint(20*c444).astype(np.int64)
    if max(float(abs(20*c444-ints).max()), abs(20*L444+1)) > 1e-8:
        raise ValueError('Source is not the expected scale-20 inequality')
    T, offset = affine_reduce(ints)
    if offset != 1 or not np.array_equal(T, explicit_T()):
        raise RuntimeError('Affine reduction differs from the written T')
    bound = -1-offset
    scale = int(sum(abs(T)))
    assert np.count_nonzero(T) == 21 and scale == 22
    merged_B = [np.array([B[y, list(group)].sum(axis=0)
                         for group in groups])
                for y, groups in enumerate(MERGES)]
    parties = [A, merged_B, C]
    validate_strategy(psi, parties)
    rho = np.outer(psi, psi.conj())
    old_value = source444.bell_value(rho, ints, A, B, C)
    point = projected_point(rho, parties)
    reconstruction_error = abs(old_value - (T @ point + offset))
    if reconstruction_error > 1e-12:
        raise RuntimeError('Affine quantum value reconstruction failed')
    system = build_hi_system()
    c, L = T/scale, bound/scale
    support = witness_support(system, c)
    if abs(support-L) > 1e-9:
        raise RuntimeError('Independent support LP differs from -2/22')
    output = args.output or BASE/'results_seesaw_324'/(
        datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    output.mkdir(parents=True, exist_ok=False)
    (output/'labels.json').write_text(json.dumps(LABELS, indent=2)+'\n')
    initial, point = save_candidate(
        output/'initial.npz', psi, parties, c, L,
    )
    save_candidate(output/'best.npz', psi, parties, c, L)
    best, records = L-initial, []
    legacy = {}
    for filename in ('outer324_best_coefficients.npy',
                     'outer324_best_c.npy', 'merged324_c.npy'):
        candidate = BASE/filename
        if candidate.exists():
            old_c = np.load(candidate, allow_pickle=False).reshape(-1)
            if old_c.shape != (34,) or np.iscomplexobj(old_c):
                legacy[filename] = dict(
                    skipped=True, shape=list(old_c.shape),
                    reason='Not a real coefficient vector; possible c/C '
                           'filename collision on case-insensitive systems',
                )
                continue
            old_bound = witness_support(system, old_c)
            legacy[filename] = compare_witness(
                old_c, old_bound, T, bound,
            )
    for iteration in range(args.outer):
        separator, status = extract_separator(system, point)
        if separator is None:
            records.append(dict(iteration=iteration, membership=status))
            break
        next_c, certificate_bound, y = separator
        comparison = compare_witness(next_c, certificate_bound, T, bound)
        next_L = witness_support(system, next_c)
        if abs(next_L-certificate_bound) > 1e-8:
            raise RuntimeError('Farkas and independent support disagree')
        change = max(float(abs(next_c-c).max()), abs(next_L-L))
        c, L = next_c, next_L
        psi, parties, result = run_seesaw(
            c, psi, parties, args.max_sweeps, args.tol,
        )
        value, point = save_candidate(
            output/f'iteration_{iteration:03d}.npz',
            psi, parties, c, L,
        )
        np.save(output/f'farkas_{iteration:03d}.npy', y)
        np.save(output/f'history_{iteration:03d}.npy', result['history'])
        gap = L-value
        record = dict(
            iteration=iteration, membership=status, value=value,
            HI_bound=L, gap=gap, comparison_to_affine=comparison,
            **{k: v for k, v in result.items() if k != 'history'},
        )
        records.append(record)
        print(json.dumps(record), flush=True)
        if gap > best:
            best = gap
            save_candidate(output/'best.npz', psi, parties, c, L)
        stalled = result['converged'] or result['rejected_increasing_sweep']
        if change < 1e-8 and stalled:
            break
    report = dict(
        scenario='[22][324][22]', dimensions=list(DIMS),
        source_strategy=str(path.resolve()), projected_dimension=34,
        hidden_weights=144, integer_terms=21, integer_l1=22,
        affine_constant=offset, integer_bound=bound,
        explicit_T_matches_affine_exactly=True,
        quantum_reconstruction_error=reconstruction_error,
        initial_value=initial, initial_gap=bound/scale-initial,
        normalized_affine_bound=bound/scale, independent_HI_support=support,
        best_gap=best, legacy_comparisons=legacy, iterations=records,
        caveat='Finite local search; integer reconstruction is numerical, '
               'while the affine transformation is exact integer arithmetic',
    )
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    print('Saved to', output, flush=True)


if __name__ == '__main__':
    main()
