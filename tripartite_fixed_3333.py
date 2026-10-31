"""Fixed-strategy fourth-setting experiment, [22][3333][22].
No seesaw updates: keep state,A,C,B0,B1,B2 fixed; sample only B3.
Default 500 normalized complex-Wishart POVMs, reproducible RNG seed.
Report the observed gap/coefficient statistics, never assume no improvement.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from hi_membership_benchmark import (
    find_farkas_certificate, certificate_to_inequality, LP_OPTIONS,
    build_hi_system as build_three_setting_system,
    project_probability as project_three_setting_probability,
    compare_inequalities, paper_eq4, LABELS as THREE_LABELS,
)
from hi_membership_seesaw import load_strategy
from tripartite_benchmark import make_probability_table

BASE = Path(__file__).resolve().parent
MA, OA = 2, 2
MB, OB = 4, 3
MC, OC = 2, 2


def make_labels():
    labels = []
    for x in range(MA):
        labels.append(('PA', 0, x))
    for y in range(MB):
        for b in range(OB - 1):
            labels.append(('PB', b, y))
    for z in range(MC):
        labels.append(('PC', 0, z))
    for x in range(MA):
        for y in range(MB):
            for b in range(OB - 1):
                labels.append(('PAB', 0, b, x, y))
    for y in range(MB):
        for z in range(MC):
            for b in range(OB - 1):
                labels.append(('PBC', b, 0, y, z))
    return labels

LABELS = make_labels()
DIM = len(LABELS)
LABEL_TO_INDEX = {label:i for i,label in enumerate(LABELS)}
LAMBDA = list(itertools.product(itertools.product(range(2),repeat=2),repeat=2))
NLAMBDA = len(LAMBDA)
NW = MB * OB * NLAMBDA
assert (DIM,NW) == (44,192)


def w_index(b, y, lam):
    return (y * OB + b) * NLAMBDA + lam

def build_hi_system():
    A_rows = []
    constants = []
    p_coeffs = []
    descriptions = []

    def add_equation(row, constant=0.0, probability_terms=None, description=''):
        """
        RHS:

            constant
            +
            sum coefficient * p[label]
        """
        coeff = np.zeros(DIM)
        if probability_terms is not None:
            for label, value in probability_terms:
                coeff[LABEL_TO_INDEX[label]] += value
        A_rows.append(row)
        constants.append(constant)
        p_coeffs.append(coeff)
        descriptions.append(description)
    for y in range(MB):
        row = np.zeros(NW)
        for b in range(OB):
            for lam in range(NLAMBDA):
                row[w_index(b, y, lam)] = 1.0
        add_equation(row, constant=1.0, description=f'normalization y={y}')
    for y in range(MB):
        for b in range(OB - 1):
            row = np.zeros(NW)
            for lam in range(NLAMBDA):
                row[w_index(b, y, lam)] = 1.0
            label = ('PB', b, y)
            add_equation(row, probability_terms=[(label, 1.0)], description=str(label))
    for x in range(MA):
        for y in range(MB):
            for b in range(OB - 1):
                row = np.zeros(NW)
                for lam, (avec, cvec) in enumerate(LAMBDA):
                    if avec[x] == 0:
                        row[w_index(b, y, lam)] = 1.0
                label = ('PAB', 0, b, x, y)
                add_equation(row, probability_terms=[(label, 1.0)], description=str(label))
    for y in range(MB):
        for z in range(MC):
            for b in range(OB - 1):
                row = np.zeros(NW)
                for lam, (avec, cvec) in enumerate(LAMBDA):
                    if cvec[z] == 0:
                        row[w_index(b, y, lam)] = 1.0
                label = ('PBC', b, 0, y, z)
                add_equation(row, probability_terms=[(label, 1.0)], description=str(label))
    for x in range(MA):
        for y in range(MB):
            row = np.zeros(NW)
            for b in range(OB):
                for lam, (avec, cvec) in enumerate(LAMBDA):
                    if avec[x] == 0:
                        row[w_index(b, y, lam)] = 1.0
            label = ('PA', 0, x)
            add_equation(row, probability_terms=[(label, 1.0)], description=f'{label}, using y={y}')
    for z in range(MC):
        for y in range(MB):
            row = np.zeros(NW)
            for b in range(OB):
                for lam, (avec, cvec) in enumerate(LAMBDA):
                    if cvec[z] == 0:
                        row[w_index(b, y, lam)] = 1.0
            label = ('PC', 0, z)
            add_equation(row, probability_terms=[(label, 1.0)], description=f'{label}, using y={y}')
    for a in range(OA):
        for c in range(OC):
            for x in range(MA):
                for z in range(MC):
                    for y in range(1, MB):
                        row = np.zeros(NW)
                        for b in range(OB):
                            for lam, (avec, cvec) in enumerate(LAMBDA):
                                if avec[x] == a and cvec[z] == c:
                                    row[w_index(b, y, lam)] += 1.0
                        for b in range(OB):
                            for lam, (avec, cvec) in enumerate(LAMBDA):
                                if avec[x] == a and cvec[z] == c:
                                    row[w_index(b, 0, lam)] -= 1.0
                        add_equation(row, constant=0.0, description=f'AC-NS a={a},c={c},x={x},z={z},y={y}')
    return (np.asarray(A_rows), np.asarray(constants), np.asarray(p_coeffs), descriptions)

def random_povm(rng, family):
    if family == 'projective':
        X = rng.normal(size=(3,3)) + 1j*rng.normal(size=(3,3))
        U, R = np.linalg.qr(X)
        diagonal = np.diag(R)
        U = U @ np.diag(diagonal / np.abs(diagonal))
        return np.asarray([np.outer(U[:,b], U[:,b].conj()) for b in range(3)])
    effects = []
    for _ in range(3):
        X = rng.normal(size=(3,3)) + 1j*rng.normal(size=(3,3))
        effects.append(X @ X.conj().T)
    values, vectors = np.linalg.eigh(sum(effects))
    inv = (vectors * values**-0.5) @ vectors.conj().T
    return np.asarray([(inv @ E @ inv + (inv @ E @ inv).conj().T)/2 for E in effects])


def projected_point(rho, A, B, C):
    identities = [np.eye(2), np.eye(3), np.eye(2)]
    p = []
    for label in LABELS:
        factors = list(identities)
        kind = label[0]
        if kind in ('PA','PB','PC'):
            _, outcome, setting = label
            party = ('PA','PB','PC').index(kind)
            factors[party] = (A,B,C)[party][setting][outcome]
        elif kind == 'PAB':
            _, a,b,x,y = label
            factors[0], factors[1] = A[x,a], B[y,b]
        else:
            _, b,c,y,z = label
            factors[1], factors[2] = B[y,b], C[z,c]
        operator = np.kron(np.kron(factors[0],factors[1]),factors[2])
        p.append(float(np.trace(rho @ operator).real))
    return np.asarray(p)


def separate(system, point):
    A, constants, R, _ = system
    b = constants + R @ point
    member = linprog(
        np.zeros(A.shape[1]), A_eq=A, b_eq=b, bounds=(0,None),
        method='highs', options=LP_OPTIONS,
    )
    if member.status not in (0,2):
        raise RuntimeError('Membership unresolved: '+member.message)
    if member.success:
        return None
    _, y = find_farkas_certificate(A,b)
    if np.max(A.T@y) > 1e-8 or b@y <= 1e-9 or np.sum(abs(y)) > 1+1e-8:
        raise RuntimeError('Invalid upper-form Farkas certificate')
    # Convert to the paper lower-bound form, then normalize observable coefficients.
    c, L = certificate_to_inequality(y,constants,R)
    norm = np.sum(abs(c))
    if norm < 1e-14:
        raise RuntimeError('Zero separator')
    return c/norm, L/norm, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', type=Path, help='best.npz from tripartite_seesaw_333.py')
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261031)
    parser.add_argument('--family', choices=['wishart','projective'], default='wishart')
    parser.add_argument('--coefficient-tol', type=float, default=1e-10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.trials < 1 or args.coefficient_tol <= 0:
        parser.error('trials and coefficient-tol must be positive')
    path = args.strategy
    if path is None:
        candidates = list((BASE/'results_seesaw_333').glob('*/best.npz'))
        if not candidates:
            parser.error('Run tripartite_seesaw_333.py first or supply --strategy')
        path = max(candidates,key=lambda p:(p.stat().st_mtime_ns,str(p)))
    psi, (A,B0,C), _, diagnostics = load_strategy(path)
    frozen = [x.copy() for x in (psi,A,B0,C)]
    rho = np.outer(psi,psi.conj())
    baseline_point = project_three_setting_probability(make_probability_table(psi,A,B0,C))
    baseline = separate(build_three_setting_system(),baseline_point)
    if baseline is None:
        raise RuntimeError('Baseline strategy is inside HI')
    reference_c, reference_L, _ = baseline
    baseline_gap = float(reference_L-reference_c@baseline_point)
    raw, raw_L = paper_eq4()
    baseline_comparison = compare_inequalities(reference_c,reference_L,raw,raw_L)
    system = build_hi_system()
    output = args.output or BASE/'results_seesaw_3333'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    new_setting = np.array([
        (label[0] == 'PB' and label[2] == 3)
        or (label[0] == 'PAB' and label[4] == 3)
        or (label[0] == 'PBC' and label[3] == 3) for label in LABELS
    ])
    rows, B3s, coefficients, bounds, duals = [], [], [], [], []
    point_list = []
    old_indices = [LABEL_TO_INDEX[label] for label in THREE_LABELS]
    for trial in range(args.trials):
        B3 = random_povm(rng,args.family)
        complete = float(np.max(abs(B3.sum(axis=0)-np.eye(3))))
        minimum = float(min(np.linalg.eigvalsh(E).min() for E in B3))
        if complete > 1e-9 or minimum < -1e-9:
            raise RuntimeError('Random POVM is not physical')
        B = np.concatenate([B0,B3[None]],axis=0)
        point = projected_point(rho,A,B,C)
        separator = separate(system,point)
        if separator is None:
            raise RuntimeError('Extended point inside HI despite violating restriction')
        c,L,y = separator
        # Forget y=3 and compare the remaining coefficients with the baseline.
        gap = float(L-c@point)
        row = dict(
            trial=trial, normalized_gap=gap, gap_minus_baseline=gap-baseline_gap,
            y3_coefficient_l1=float(np.sum(abs(c[new_setting]))),
            y3_coefficient_max=float(np.max(abs(c[new_setting]))),
            old_coefficient_max_change=float(np.max(abs(c[old_indices]-reference_c))),
            bound_change=float(L-reference_L),
            min_B3_eigenvalue=minimum, B3_completeness_residual=complete,
        )
        rows.append(row)
        B3s.append(B3); coefficients.append(c); bounds.append(L); duals.append(y)
        point_list.append(point)
        if (trial+1) % 50 == 0 or trial+1 == args.trials:
            print(f'{trial+1}/{args.trials}: gap={gap:.15g}, y3 L1={row["y3_coefficient_l1"]:.3g}',flush=True)
    assert all(np.array_equal(a,b) for a,b in zip(frozen,(psi,A,B0,C)))
    changes = np.array([r['gap_minus_baseline'] for r in rows])
    report = dict(
        scenario='[22][3333][22]', projected_dimension=DIM, hidden_weights=NW,
        strategy_file=str(path.resolve()), strategy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        trials=args.trials, seed=args.seed, random_family=args.family,
        experiment='Only B3 varies; no seesaw/state or old-measurement updates',
        baseline_normalized_gap=baseline_gap, baseline_matches_Eq4=baseline_comparison,
        max_abs_gap_change=float(np.max(abs(changes))),
        max_gap_increase=float(np.max(changes)),
        max_y3_coefficient_l1=max(r['y3_coefficient_l1'] for r in rows),
        trials_with_y3_coefficients=sum(r['y3_coefficient_max']>args.coefficient_tol for r in rows),
        coefficient_tolerance=args.coefficient_tol, fixed_strategy_unchanged=True,
        caveat='One optimal Farkas solution per sample; dual degeneracy prevents claims about every optimal separator. No claim that four settings can never help.',
        fixed_strategy_diagnostics=diagnostics,
    )
    with (output/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'labels.json').write_text(json.dumps(LABELS,indent=2)+'\n')
    np.savez(output/'samples.npz',state=psi,A=A,B_original=B0,C=C,
             B3=np.asarray(B3s),coefficients=np.asarray(coefficients),HI_bound=np.asarray(bounds),
             farkas_y=np.asarray(duals),point=np.asarray(point_list))
    print(json.dumps(report,indent=2))
    print('Saved all 500 samples' if args.trials==500 else 'Saved all samples', 'to',output)


if __name__ == '__main__':
    main()
