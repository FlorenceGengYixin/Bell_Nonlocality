"""Joint HI/Farkas and quantum seesaw search for [22][3333][22].
All state and POVM blocks are updated, including Bob's fourth setting.
Perturbed fourth-setting witnesses and random strategies explore beyond
simply appending B3. This local search makes no global-optimum claim.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from tripartite_seesaw_333 import hermitian, normalize_povm, term_indices
from tripartite_fixed_3333 import (
    LABELS, LABEL_TO_INDEX, build_hi_system, projected_point, random_povm,
)
from hi_membership_benchmark import (
    LABELS as THREE_LABELS, paper_eq4, LP_OPTIONS,
    find_farkas_certificate, certificate_to_inequality,
)
from hi_membership_seesaw import load_strategy

BASE = Path(__file__).resolve().parent
DIMS, SETTINGS, OUTCOMES = (2,3,2), (2,4,2), (2,3,2)
DIM = 12
IDENTITIES = [np.eye(d,dtype=complex) for d in DIMS]
MAX_SWEEPS, TOL = 60, 1e-10

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy',type=Path,help='A three-setting best.npz; otherwise use latest saved strategy')
    parser.add_argument('--restarts',type=int,default=6)
    parser.add_argument('--outer',type=int,default=4)
    parser.add_argument('--max-sweeps',type=int,default=MAX_SWEEPS)
    parser.add_argument('--tol',type=float,default=TOL)
    parser.add_argument('--witness-strength',type=float,default=0.02)
    parser.add_argument('--seed',type=int,default=20261031)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if min(args.restarts,args.outer,args.max_sweeps)<1 or args.tol<=0 or args.witness_strength<0:
        parser.error('Invalid iteration count, tolerance or witness strength')
    path = args.strategy
    if path is None:
        files = list((BASE/'results_seesaw_333').glob('*/best.npz'))
        if not files:
            parser.error('Run tripartite_seesaw_333.py first or supply --strategy')
        path = max(files,key=lambda p:(p.stat().st_mtime_ns,str(p)))
    psi,(A,B,C),_,_ = load_strategy(path)
    rng = np.random.default_rng(args.seed)
    B = np.concatenate([B,random_povm(rng,'wishart')[None]])
    known = (psi,[A,B,C])
    validate_strategy(*known)
    system = build_hi_system()
    raw,Lraw = paper_eq4()
    base_c = np.zeros(len(LABELS))
    for label,value in zip(THREE_LABELS,raw/27):
        base_c[LABEL_TO_INDEX[label]]=value
    y3 = np.array([label not in THREE_LABELS for label in LABELS])
    output = args.output or BASE/'results_joint_seesaw_3333'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    records,best = [],-np.inf

    def save_candidate(tag,state,parties,c,L,extra):
        nonlocal best
        point = projected_point(np.outer(state,state.conj()),*parties)
        gap = float(L-c@point)
        diagnostics = validate_strategy(state,parties)
        payload = dict(state=state,A=parties[0],B=parties[1],C=parties[2],
                       coefficients=c,HI_bound=L,point=point)
        report = dict(normalized_gap=gap,value=float(c@point),bound=float(L),
                      seed=args.seed,source_strategy=str(path.resolve()),**diagnostics,**extra)
        np.savez(output/f'{tag}.npz',**payload)
        (output/f'{tag}.json').write_text(json.dumps(report,indent=2)+'\n')
        if gap>best:
            best=gap
            np.savez(output/'best.npz',**payload)
            (output/'best.json').write_text(json.dumps(report,indent=2)+'\n')
        return gap

    save_candidate('initial',*known,base_c,Lraw/27,dict(stage='lifted three-setting strategy'))
    for restart in range(args.restarts):
        if restart==0:
            state,parties = known[0].copy(),[p.copy() for p in known[1]]
            kind='lifted'
        elif restart<=args.restarts//2:
            state,parties = perturb_strategy(known,rng)
            kind='perturbed_all_blocks'
        else:
            state,parties = random_strategy(rng)
            kind='random_all_blocks'
        c=base_c.copy()
        # Without y3 coefficients, an Eq4-only seesaw cannot explore B3.
        # Give non-baseline restarts a genuinely B3-dependent VALID witness.
        if restart>0:
            direction=np.zeros(len(LABELS));direction[y3]=rng.normal(size=y3.sum())
            direction/=np.sum(abs(direction))
            c+=args.witness_strength*direction
            c/=np.sum(abs(c))
        L=witness_support(system,c)
        for outer in range(args.outer):
            before=projected_point(np.outer(state,state.conj()),*parties)
            result=run_seesaw(c,np.outer(state,state.conj()),*parties,
                             max_sweeps=args.max_sweeps,tol=args.tol)
            state=result['psi'];parties=[result[name] for name in 'ABC']
            fixed_gap=save_candidate(
                f'r{restart:03d}_o{outer:03d}',state,parties,c,L,
                dict(stage='joint quantum seesaw',restart=restart,outer=outer,
                     start=kind,sweeps=result['sweeps'],converged=result['converged']),
            )
            point=projected_point(np.outer(state,state.conj()),*parties)
            separator,membership_status=extract_separator(system,point)
            record=dict(restart=restart,outer=outer,start=kind,sweeps=result['sweeps'],
                        converged=result['converged'],fixed_witness_gap=fixed_gap,
                        fixed_witness_y3_l1=float(np.sum(abs(c[y3]))),
                        objective_improvement=float(c@before-c@point),
                        resolved_farkas_violation=separator is not None,
                        membership_status=membership_status)
            np.save(output/f'history_r{restart:03d}_o{outer:03d}.npy',result['history'])
            if separator is not None:
                new_c,new_L,y=separator
                np.save(output/f'farkas_r{restart:03d}_o{outer:03d}.npy',y)
                gap=save_candidate(
                    f'separator_r{restart:03d}_o{outer:03d}',state,parties,new_c,new_L,
                    dict(stage='independent Farkas separation',restart=restart,outer=outer),
                )
                record.update(new_separator_gap=gap,new_separator_y3_l1=float(np.sum(abs(new_c[y3]))))
                change=max(float(np.max(abs(new_c-c))),abs(new_L-L))
                c,L=new_c,new_L
            else:
                record.update(new_separator_gap=None,new_separator_y3_l1=None)
                change=0.
            records.append(record)
            print(json.dumps(record),flush=True)
            (output/'iterations.json').write_text(json.dumps(records,indent=2)+'\n')
            if separator is None or (outer>0 and change<1e-8):
                break
    print(f'Best normalized HI gap={best:.15g}; saved to {output}',flush=True)


if __name__=='__main__':
    main()
