"""Lift the optimized [22][333][22] strategy to [22][4444][22].
Extract one free HI/Farkas witness from the lift, then optimize all quantum
blocks for that FIXED witness. Optional perturbed restarts; no target gap
is imposed. Results and lift/certificate checks are saved per run.
"""
import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from tripartite_seesaw_333 import hermitian, normalize_povm, term_indices
from hi_membership_benchmark import (
    LABELS as THREE_LABELS, paper_eq4, find_farkas_certificate,
    certificate_to_inequality, LP_OPTIONS,
)
from hi_membership_seesaw import load_strategy

BASE = Path(__file__).resolve().parent
MA, MB, MC = 2, 4, 2
OA, OB, OC = 2, 4, 2
DIMS, SETTINGS, OUTCOMES = (2,4,2), (MA,MB,MC), (OA,OB,OC)
DIM = 16
IDENTITIES = [np.eye(d,dtype=complex) for d in DIMS]
MAX_SWEEPS, TOL = 100, 1e-10


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

LABELS=make_labels()
LABEL_TO_INDEX={label:i for i,label in enumerate(LABELS)}
LAMBDA=list(itertools.product(itertools.product(range(2),repeat=2),repeat=2))
NLAMBDA=len(LAMBDA)
NW=MB*OB*NLAMBDA
assert (len(LABELS),NW)==(64,256)


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
        coeff = np.zeros(len(LABELS))
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

def effective_operators(rho, coeff, parties, party):
    """F gives Tr(E F); contract the fixed parties out of the Born rule."""
    d = DIMS[party]
    F = np.zeros((SETTINGS[party], OUTCOMES[party], d, d), dtype=complex)
    tensor = rho.reshape(*DIMS, *DIMS)
    for value, label in zip(coeff, LABELS):
        indices = term_indices(label)
        if party not in indices or abs(value) < 1e-15:
            continue
        factors = list(IDENTITIES)
        for j, (setting, outcome) in indices.items():
            if j != party:
                factors[j] = parties[j][setting][outcome]
        if party == 0:
            local = np.einsum("abcdef,eb,fc->ad", tensor, factors[1], factors[2])
        elif party == 1:
            local = np.einsum("abcdef,da,fc->be", tensor, factors[0], factors[2])
        else:
            local = np.einsum("abcdef,da,eb->cf", tensor, factors[0], factors[1])
        setting, outcome = indices[party]
        F[setting, outcome] += value * hermitian(local)
    return F

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
    if problem.status != cp.OPTIMAL:
        problem.solve(solver=cp.SCS, eps=1e-9, max_iters=200000)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"SDP failed: {problem.status}")
    return np.asarray([normalize_povm([X.value for X in setting]) for setting in E])

def validate_strategy(psi, parties, tol=1e-8):
    if psi.shape != (DIM,) or abs(np.vdot(psi, psi) - 1) > tol:
        raise RuntimeError("Invalid state")
    residual, min_eig = 0., 1.
    for party, m, o, d in zip(parties, SETTINGS, OUTCOMES, DIMS):
        if np.shape(party) != (m, o, d, d):
            raise RuntimeError("Invalid measurement dimensions")
        for setting in party:
            residual = max(residual, float(np.max(np.abs(sum(setting) - np.eye(d)))))
            for E in setting:
                residual = max(residual, float(np.max(np.abs(E - E.conj().T))))
                min_eig = min(min_eig, float(np.linalg.eigvalsh(hermitian(E)).min()))
    if residual > tol or min_eig < -tol:
        raise RuntimeError("Invalid POVMs")
    return {"max_povm_residual": residual, "min_povm_eigenvalue": min_eig}

def random_strategy(rng):
    psi = rng.normal(size=DIM) + 1j * rng.normal(size=DIM)
    psi /= np.linalg.norm(psi)
    parties = []
    for m, o, d in zip(SETTINGS, OUTCOMES, DIMS):
        party = []
        for _ in range(m):
            effects = []
            for _ in range(o):
                X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
                effects.append(X @ X.conj().T)
            party.append(normalize_povm(effects))
        parties.append(np.asarray(party))
    return psi, parties

def perturb_strategy(known, rng, strength=0.05):
    psi, parties = known
    random_psi, random_parties = random_strategy(rng)
    psi = psi + strength * random_psi
    psi /= np.linalg.norm(psi)
    parties = [(1-strength)*P + strength*Q for P, Q in zip(parties, random_parties)]
    return psi, parties

def run_seesaw(coeff, rho, A, B, C, max_sweeps=MAX_SWEEPS, tol=TOL):
    parties = [np.array(p, copy=True) for p in (A, B, C)]
    previous = bell_value(rho, coeff, *parties)
    history = [previous]
    converged, rejected = False, False
    for _ in range(max_sweeps):
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

def witness_support(system, coefficients):
    A,d,R,_ = system
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
    A,d,R,_ = system
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

def projected_point(rho, A, B, C):
    identities = [np.eye(2), np.eye(4), np.eye(2)]
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

def lifted_strategy(path):
    old_state,(A,old_B,C),saved_S,_=load_strategy(path)
    state=np.zeros((2,4,2),dtype=complex)
    state[:,:3,:]=old_state.reshape(2,3,2)
    state=state.reshape(-1)
    B=np.zeros((4,4,4,4),dtype=complex)
    B[:3,:3,:3,:3]=old_B
    B[:3,3,3,3]=1
    B[3]=B[2].copy()
    parties=[A.copy(),B,C.copy()]
    validate_strategy(state,parties)
    point=projected_point(np.outer(state,state.conj()),*parties)
    raw,_=paper_eq4()
    indices=[LABEL_TO_INDEX[label] for label in THREE_LABELS]
    value=float(raw@point[indices])
    if saved_S is not None and abs(value-saved_S)>1e-10:
        raise RuntimeError('State/measurement lift failed Eq4 value preservation')
    return (state,parties),value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy',type=Path,help='A three-setting best.npz')
    parser.add_argument('--restarts',type=int,default=4)
    parser.add_argument('--max-sweeps',type=int,default=MAX_SWEEPS)
    parser.add_argument('--tol',type=float,default=TOL)
    parser.add_argument('--seed',type=int,default=20261031)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if min(args.restarts,args.max_sweeps)<1 or args.tol<=0:
        parser.error('Iteration counts and tolerance must be positive')
    path=args.strategy
    if path is None:
        candidates=list((BASE/'results_seesaw_333').glob('*/best.npz'))
        if not candidates:
            parser.error('Run tripartite_seesaw_333.py or supply --strategy')
        path=max(candidates,key=lambda p:(p.stat().st_mtime_ns,str(p)))
    known,old_S=lifted_strategy(path)
    state,parties=known
    system=build_hi_system()
    point=projected_point(np.outer(state,state.conj()),*parties)
    separator,status=extract_separator(system,point)
    if separator is None:
        raise RuntimeError('Lift did not give a resolved outside-HI certificate: '+status)
    c,L,y=separator
    support=witness_support(system,c)
    if abs(support-L)>1e-8:
        raise RuntimeError('Independent witness support check failed')
    initial=float(c@point)
    baseline_gap=(-2-old_S)/27
    output=args.output or BASE/'results_seesaw_4444'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    np.savez(output/'lift.npz',state=state,A=parties[0],B=parties[1],C=parties[2],
             point=point,coefficients=c,HI_bound=L,farkas_y=y)
    (output/'labels.json').write_text(json.dumps(LABELS,indent=2)+'\n')
    print(f'Lift: Eq4={old_S:.15g}, witness Q={initial:.15g}, L={L:.15g}, gap={L-initial:.12g}',flush=True)
    rng=np.random.default_rng(args.seed)
    records=[];best=float('inf')
    for restart in range(args.restarts):
        if restart==0:
            psi=known[0].copy();P=[p.copy() for p in known[1]];kind='exact_lift'
        else:
            psi,P=perturb_strategy(known,rng,strength=.02);kind='perturbed_lift'
        result=run_seesaw(c,np.outer(psi,psi.conj()),*P,
                          max_sweeps=args.max_sweeps,tol=args.tol)
        P=[result[p] for p in 'ABC'];psi=result['psi']
        diagnostics=validate_strategy(psi,P)
        value=bell_value(np.outer(psi,psi.conj()),c,*P)
        point=projected_point(np.outer(psi,psi.conj()),*P)
        if abs(value-c@point)>1e-12:
            raise RuntimeError('Bell-operator and projected-probability values disagree')
        record=dict(restart=restart,start=kind,sweeps=result['sweeps'],
                    converged=result['converged'],rejected_increasing_sweep=result['rejected_increasing_sweep'],
                    normalized_value=value,HI_bound=L,normalized_gap=L-value,
                    gap_ratio_over_333=(L-value)/baseline_gap,**diagnostics)
        records.append(record)
        np.save(output/f'history_{restart:03d}.npy',result['history'])
        print(json.dumps(record),flush=True)
        if value<best:
            best=value
            report=dict(record,scenario='[22][4444][22]',dimensions=list(DIMS),
                        projected_dimension=64,hidden_weights=256,source_strategy=str(path.resolve()),
                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        seed=args.seed,restarts_requested=args.restarts,tol=args.tol,
                        lifted_membership='infeasible',lifted_Eq4_value=old_S,
                        baseline_normalized_gap=baseline_gap,lifted_witness_value=initial,
                        lifted_witness_gap=L-initial,independent_HI_support=support,
                        quoted_gap_reference=2.1254098624572115e-5,
                        difference_from_quoted_gap=(L-value)-2.1254098624572115e-5,
                        method='One freely extracted lifted-point Farkas witness, fixed throughout seesaw')
            np.savez(output/'best.npz',state=psi,rho=np.outer(psi,psi.conj()),A=P[0],B=P[1],C=P[2],
                     coefficients=c,HI_bound=L,point=point,farkas_y=y,history=result['history'])
            (output/'best.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'restarts.json').write_text(json.dumps(records,indent=2)+'\n')
    print(f'Best normalized gap={L-best:.15g}; saved to {output}',flush=True)


if __name__=='__main__':
    main()
