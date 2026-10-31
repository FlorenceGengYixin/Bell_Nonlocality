"""Fixed-witness [22][444][22] see-saw, with independent HI support LP.
Default Bob dimension is 3, matching merge_4444_to_444/multirestart_444.
Use --bob-dim 4 for a four-dimensional realization. Results never overwrite
legacy numerical arrays. Imports only existing published helper modules.
"""
import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from tripartite_seesaw_333 import hermitian, normalize_povm, term_indices
from hi_membership_benchmark import (
    find_farkas_certificate,
    certificate_to_inequality, LP_OPTIONS,
)

BASE = Path(__file__).resolve().parent
MA, MB, MC = 2, 3, 2
OA, OB, OC = 2, 4, 2
DIMS, SETTINGS, OUTCOMES = (2,3,2), (MA,MB,MC), (OA,OB,OC)
DIM = 12
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
assert (len(LABELS),NW)==(49,192)


def w_index(b, y, lam):
    return (y * OB + b) * NLAMBDA + lam

def build_hi_system():
    A_rows = []
    constants = []
    p_coeffs = []
    descriptions = []

    def add_equation(row, constant=0.0, probability_terms=None, description=''):
        """Append one equality A w = constant + R p."""
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
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        problem.solve(solver=cp.SCS, eps=1e-9, max_iters=50000, time_limit_secs=10)
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
    psi = np.sqrt(1-strength**2) * psi + strength * random_psi
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
    identities = IDENTITIES
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

def configure(bob_dim):
    global DIMS, DIM, IDENTITIES
    if bob_dim not in (3,4):
        raise ValueError('Bob dimension must be 3 or 4')
    DIMS = (2,bob_dim,2)
    DIM = 4*bob_dim
    IDENTITIES = [np.eye(d,dtype=complex) for d in DIMS]


def merge_label(label):
    label = list(label)
    index = {'PB':2,'PAB':4,'PBC':3}.get(label[0])
    if index is not None:
        label[index] = min(label[index],2)
    return tuple(label)


def load_initial(strategy=None, witness=None):
    if strategy is None:
        prefix = BASE/'settingcontrol_B3eqB2'
        psi = np.load(str(prefix)+'_state.npy')
        P = [np.load(str(prefix)+'_'+k+'.npy') for k in 'ABC']
        c = np.load(witness or BASE/'seed4444_farkas_c.npy')
    else:
        with np.load(strategy,allow_pickle=False) as data:
            psi = data['state'].copy()
            P = [data[k].copy() for k in 'ABC']
            c = data['coefficients'].copy()
        if witness is not None:
            c = np.load(witness)
    P[1] = P[1][:3].copy()
    source_dim = P[1].shape[-1]
    if source_dim > DIMS[1]:
        raise ValueError('No automatic compression: use --bob-dim 4 for a 4D source')
    if source_dim < DIMS[1]:
        expanded = np.zeros((2,DIMS[1],2),complex)
        expanded[:,:source_dim] = psi.reshape(2,source_dim,2)
        psi = expanded.reshape(-1)
        B = np.zeros((3,4,DIMS[1],DIMS[1]),complex)
        B[:,:,:source_dim,:source_dim] = P[1]
        B[:,3,source_dim:,source_dim:] = np.eye(DIMS[1]-source_dim)
        P[1] = B
    if c.shape == (64,):
        labels4 = []
        for x in range(2): labels4.append(('PA',0,x))
        for y in range(4):
            for b in range(3): labels4.append(('PB',b,y))
        for z in range(2): labels4.append(('PC',0,z))
        for x in range(2):
            for y in range(4):
                for b in range(3): labels4.append(('PAB',0,b,x,y))
        for y in range(4):
            for z in range(2):
                for b in range(3): labels4.append(('PBC',b,0,y,z))
        merged = np.zeros(49)
        for value,label in zip(c,labels4):
            merged[LABEL_TO_INDEX[merge_label(label)]] += value
        c = merged
    if c.shape != (49,) or not np.isfinite(c).all() or np.sum(abs(c)) == 0:
        raise ValueError('Invalid witness coefficients')
    validate_strategy(psi,P)
    c /= np.sum(abs(c))
    return psi,P,c


def save_strategy(path,psi,P,c,L):
    point = projected_point(np.outer(psi,psi.conj()),*P)
    if abs(c@point-bell_value(np.outer(psi,psi.conj()),c,*P)) > 1e-12:
        raise RuntimeError('Born probability and Bell-operator objectives disagree')
    validate_strategy(psi,P)
    np.savez(path,state=psi,A=P[0],B=P[1],C=P[2],
             coefficients=c,HI_bound=L,point=point)
    return float(c@point)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy',type=Path)
    parser.add_argument('--witness',type=Path)
    parser.add_argument('--bob-dim',type=int,choices=[3,4],default=3)
    parser.add_argument('--perturbed',type=int,default=20)
    parser.add_argument('--random',type=int,default=80)
    parser.add_argument('--perturbation',type=float,default=.08)
    parser.add_argument('--max-sweeps',type=int,default=100)
    parser.add_argument('--tol',type=float,default=1e-10)
    parser.add_argument('--seed',type=int,default=20261002)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if min(args.perturbed,args.random)<0 or args.max_sweeps<1 or args.tol<=0 or not 0<=args.perturbation<=1:
        parser.error('Invalid count, tolerance, or perturbation')
    configure(args.bob_dim)
    psi,P,c = load_initial(args.strategy,args.witness)
    known = psi,P
    system = build_hi_system()
    L = witness_support(system,c)
    output = args.output or BASE/'results_seesaw_444'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    (output/'labels.json').write_text(json.dumps(LABELS,indent=2)+'\n')
    initial = save_strategy(output/'initial.npz',psi,P,c,L)
    best = initial
    save_strategy(output/'best.npz',psi,P,c,L)
    records = []
    for restart in range(args.perturbed+args.random):
        rng = np.random.default_rng(args.seed+restart)
        if restart < args.perturbed:
            psi,P = perturb_strategy(known,rng,strength=args.perturbation)
            kind = 'perturbed'
        else:
            psi,P = random_strategy(rng)
            kind = 'random'
        result = run_seesaw(c,np.outer(psi,psi.conj()),*P,
                           max_sweeps=args.max_sweeps,tol=args.tol)
        psi,P = result['psi'],[result[k] for k in 'ABC']
        value = save_strategy(output/f'restart_{restart:03d}.npz',psi,P,c,L)
        np.save(output/f'history_{restart:03d}.npy',result['history'])
        record = dict(restart=restart,seed=args.seed+restart,type=kind,value=value,
                      gap=L-value,sweeps=result['sweeps'],converged=result['converged'],
                      rejected_increasing_sweep=result['rejected_increasing_sweep'])
        records.append(record)
        if value < best:
            best = value
            save_strategy(output/'best.npz',psi,P,c,L)
        print(json.dumps(record),flush=True)
        (output/'restarts.json').write_text(json.dumps(records,indent=2)+'\n')
    summary = dict(scenario='[22][444][22]',dimensions=list(DIMS),projected_dimension=49,
                   hidden_weights=192,HI_bound=L,initial_value=initial,best_value=best,
                   best_gap=L-best,tol=args.tol,max_sweeps=args.max_sweeps,seed=args.seed,
                   perturbed=args.perturbed,random=args.random,perturbation=args.perturbation,
                   source_strategy=str(args.strategy or BASE/'settingcontrol_B3eqB2'),
                   output=str(output),**validate_strategy(*known))
    for kind in ('perturbed','random'):
        group = [r for r in records if r['type']==kind]
        summary[kind+'_within_1e7'] = sum(abs(r['value']-best)<=1e-7 for r in group)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
