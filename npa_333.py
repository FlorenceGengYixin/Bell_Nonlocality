"""NPA lower bounds for raw Eq.(4), [22][333][22].
Moment variables are shared by canonical words, rather than thousands of
redundant entry equalities. Real moments suffice for this real Bell objective
by averaging a feasible representation with its complex conjugate.
A,C use dichotomic observables A=2P-I, C=2P-I; Bob uses projectors.
Bob y=1 outcomes 0,1 are merged exactly because Eq4 weights them equally.
Projective measurements are WLOG in the dimension-unbounded relaxation
(Naimark dilation); do not impose projectivity on the qutrit seesaw POVMs.
"""
import argparse
from datetime import datetime
from functools import lru_cache
import itertools
import json
from pathlib import Path
import time
import cvxpy as cp
import numpy as np
from scipy import sparse
from hi_membership_benchmark import LABELS, paper_eq4

BASE = Path(__file__).resolve().parent
SINGLE = tuple((p,x,a) for p,m,o in [('A',2,2),('B',3,3),('C',2,2)] for x in range(m) for a in range(o-1) if not (p == 'B' and x == 1 and a == 1))


@lru_cache(None)
def canonical(word):
    """Commute ONLY different parties; A_x^2=C_z^2=I; Bob projectors are orthogonal."""
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
                    stack.pop()  # Dichotomic A_x^2=C_z^2=I.
                continue
            stack.append(op)
        out.extend(stack)
    return tuple(out)


def moment_key(word):
    w = canonical(word)
    if w is None:
        return None
    # Reversal is complex conjugation, not an extra commutation relation.
    return min(w, canonical(tuple(reversed(w))))


def word_basis(level):
    if level.startswith('local'):
        a_depth,b_depth,c_depth = map(int,level[5:].split(','))
        pools = []
        for party,depth in zip('ABC',(a_depth,b_depth,c_depth)):
            ops = [op for op in SINGLE if op[0] == party]
            pool = {()}
            for length in range(1,depth+1):
                pool.update(w for seq in itertools.product(ops,repeat=length) if (w:=canonical(seq)) is not None)
            pools.append(pool)
        return sorted({a+b+c for a,b,c in itertools.product(*pools)},key=lambda w:(len(w),w))
    depth = 1 if level.startswith('1') else 2 if level in ('2','2+ABC') else 3
    words = {()}
    for length in range(1, depth+1):
        for seq in itertools.product(SINGLE, repeat=length):
            word = canonical(seq)
            if word is not None:
                words.add(word)
    if level in ('1+AB+AC+BC','2+ABC'):
        for length in (2,3) if level == '2+ABC' else (2,):
            for seq in itertools.product(SINGLE, repeat=length):
                if len({op[0] for op in seq}) == length:
                    words.add(canonical(seq))
    return sorted(words, key=lambda w: (len(w),w))


def label_word(label):
    kind = label[0]
    if kind in ('PA','PB','PC'):
        return ((kind[1], label[2], label[1]),)
    if kind == 'PAB':
        _,a,b,x,y = label
        return (('A',x,a),('B',y,b))
    _,b,c,y,z = label
    return (('B',y,b),('C',z,c))


def build_relaxation(level):
    words = word_basis(level)
    n = len(words)
    keys = {():0}
    rows, cols = [], []
    for j,right in enumerate(words):
        for i,left in enumerate(words):
            key = moment_key(tuple(reversed(left))+right)
            if key is None:
                continue
            if key not in keys:
                keys[key] = len(keys)
            rows.append(i+j*n)
            cols.append(keys[key])
    mapping = sparse.csc_matrix((np.ones(len(rows)),(rows,cols)),shape=(n*n,len(keys)))
    objective = objective_vector(keys)
    return words,keys,mapping,objective


def objective_vector(keys):
    objective = np.zeros(len(keys))
    reference,_ = paper_eq4()
    reference_map = dict(zip(LABELS,reference))
    for coefficient,label in zip(reference,LABELS):
        # Bob y=1 outcomes 0 and 1 enter Eq4 with identical coefficients.
        b_pos,y_pos = (1,2) if label[0]=='PB' else (2,4) if label[0]=='PAB' else (1,3) if label[0]=='PBC' else (None,None)
        if y_pos is not None and label[y_pos]==1:
            if label[b_pos]==1:
                continue
            partner=list(label);partner[b_pos]=1
            assert coefficient == reference_map[tuple(partner)]
        expansion = {():1.}
        for op in label_word(label):
            terms = [((),.5),((op,),.5)] if op[0] in 'AC' else [((op,),1.)]
            expansion = {w+t:v*a for w,v in expansion.items() for t,a in terms}
        for word,factor in expansion.items():
            objective[keys[moment_key(word)]] += coefficient*factor
    return objective


def corrected_dual_bound(mapping,objective,Z,n):
    """Conservative residual correction using |<word>|<=1 for quantum moments.
    For any Z>=0: c.x >= c0-(F*Z)0 - sum_{k>0}|ck-(F*Z)k|.
    PSD-project the dual and add a numerical safety margin. This is a
    floating-point certificate check, not an interval/rational proof.
    """
    Z = (Z+Z.T)/2
    values,vectors = np.linalg.eigh(Z)
    margin = 1e-12*max(1.,np.linalg.norm(Z,2))
    Z = (vectors*np.maximum(values,0))@vectors.T + margin*np.eye(n)
    residual = objective - np.asarray(mapping.T@Z.reshape(-1,order='F')).ravel()
    penalty = float(np.sum(abs(residual[1:])))
    return float(residual[0]-penalty),Z,penalty,float(np.max(abs(residual[1:])))


def solve_level(level,solver='CLARABEL',tol=1e-9,max_iters=500,verbose=False):
    start = time.perf_counter()
    words,keys,mapping,objective = build_relaxation(level)
    n = len(words)
    x = cp.Variable(len(keys))
    G = cp.reshape(mapping@x,(n,n),order='F')
    psd,normalization = G >> 0, x[0] == 1
    problem = cp.Problem(cp.Minimize(objective@x),[psd,normalization])
    print(f'{level}: PSD {n}x{n}, {len(keys)} moment variables',flush=True)
    if solver == 'CLARABEL':
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=tol,tol_gap_rel=tol,
                      tol_feas=tol,max_iter=max_iters,max_threads=1,verbose=verbose)
    else:
        problem.solve(solver=cp.SCS,eps=tol,max_iters=max_iters,verbose=verbose)
    if x.value is None or psd.dual_value is None:
        raise RuntimeError(f'No primal/dual solution: {problem.status}')
    matrix = np.asarray(mapping@x.value).reshape((n,n),order='F')
    lower,Z,penalty,dual_residual = corrected_dual_bound(mapping,objective,np.asarray(psd.dual_value),n)
    primal = float(objective@x.value)
    dual = -float(normalization.dual_value)
    report = dict(level=level,solver=solver,status=problem.status,words=n,moments=len(keys),
                  basis_description=('products of local lengths (2,1,2), a valid submatrix of Q5; not a standard numbered NPA level' if level=='local2,1,2' else level),
                  primal_objective=primal,solver_dual_objective=dual,
                  primal_dual_gap=primal-dual,corrected_quantum_lower_bound=lower,
                  dual_residual_l1_penalty=penalty,dual_residual_max=dual_residual,
                  min_primal_eigenvalue=float(np.linalg.eigvalsh(matrix).min()),
                  min_corrected_dual_eigenvalue=float(np.linalg.eigvalsh(Z).min()),
                  normalization_residual=float(abs(x.value[0]-1)),
                  iterations=int(problem.solver_stats.num_iters),seconds=time.perf_counter()-start)
    report['residual_checks_passed'] = bool(
          abs(primal-dual)<1e-7 and report['min_primal_eigenvalue']>-1e-7 and
          report['normalization_residual']<1e-8)
    report['numerically_converged'] = bool(problem.status == cp.OPTIMAL and
          abs(primal-dual)<1e-7 and report['min_primal_eigenvalue']>-1e-7 and
          report['normalization_residual']<1e-8)
    print(json.dumps(report,indent=2),flush=True)
    return report,dict(Gamma=matrix,dual_Z=Z,moments=x.value,objective=objective),words,keys


def build_block_relaxation(depth):
    # W contains AC words only, with local lengths at most (depth,depth).
    # For each Bob outcome, Gamma_b[y]=<W_i^dag B_b|y W_j> >=0.
    # Last outcomes: Gamma_last[y]=Gamma_AC-sum_b Gamma_b[y] >=0.
    # These are valid principal NPA blocks (B_b W_i indexing), sharing all
    # canonical moments. We omit cross-setting Bob blocks: still an OUTER
    # approximation, not a full numbered NPA level or a hierarchy claim.
    words = [w for w in word_basis(f'local{depth},1,{depth}') if all(op[0]!='B' for op in w)]
    n = len(words)
    keys = {():0}
    entries = []
    for marker in [None]+[op for op in SINGLE if op[0]=='B']:
        rows,cols = [],[]
        for j,right in enumerate(words):
            for i,left in enumerate(words):
                key = moment_key(tuple(reversed(left))+(() if marker is None else (marker,))+right)
                if key is None: continue
                if key not in keys: keys[key]=len(keys)
                rows.append(i+n*j);cols.append(keys[key])
        entries.append((rows,cols))
    maps = [sparse.csc_matrix((np.ones(len(r)),(r,c)),shape=(n*n,len(keys))) for r,c in entries]
    blocks,offset = [],1
    for outcomes in (3,2,3):
        setting = maps[offset:offset+outcomes-1]
        offset += outcomes-1
        blocks.extend(setting+[maps[0]-sum(setting)])
    return words,keys,blocks,objective_vector(keys)


def solve_block(depth,solver='CLARABEL',tol=1e-9,max_iters=500,verbose=False):
    start = time.perf_counter()
    words,keys,maps,objective = build_block_relaxation(depth)
    n = len(words)
    x = cp.Variable(len(keys))
    cones = [cp.reshape(M@x,(n,n),order='F') >> 0 for M in maps]
    normalization = x[0] == 1
    problem = cp.Problem(cp.Minimize(objective@x),cones+[normalization])
    print(f'block{depth}: {len(cones)} PSD blocks {n}x{n}, {len(keys)} moments',flush=True)
    if solver == 'CLARABEL':
        problem.solve(solver=cp.CLARABEL,tol_gap_abs=tol,tol_gap_rel=tol,tol_feas=tol,
                      max_iter=max_iters,max_threads=1,verbose=verbose)
    else:
        problem.solve(solver=cp.SCS,eps=tol,max_iters=max_iters,verbose=verbose)
    if x.value is None or any(c.dual_value is None for c in cones):
        raise RuntimeError(f'No primal/dual solution: {problem.status}')
    adjoint = np.zeros(len(keys));duals=[];grams=[]
    for M,con in zip(maps,cones):
        Z = np.asarray(con.dual_value);Z=(Z+Z.T)/2
        values,vectors = np.linalg.eigh(Z)
        Z = (vectors*np.maximum(values,0))@vectors.T + 1e-12*max(1.,np.linalg.norm(Z,2))*np.eye(n)
        adjoint += np.asarray(M.T@Z.reshape(-1,order='F')).ravel()
        duals.append(Z)
        grams.append(np.asarray(M@x.value).reshape(n,n,order='F'))
    residual = objective-adjoint
    penalty = float(np.sum(abs(residual[1:])))
    lower = float(residual[0]-penalty)
    primal,dual = float(objective@x.value),-float(normalization.dual_value)
    mineig = min(float(np.linalg.eigvalsh(G).min()) for G in grams)
    report = dict(level=f'block{depth}',basis_description='conditional Bob principal NPA blocks; not a full numbered NPA level',
                  solver=solver,status=problem.status,words_per_block=n,blocks=len(maps),moments=len(keys),
                  primal_objective=primal,solver_dual_objective=dual,primal_dual_gap=primal-dual,
                  corrected_quantum_lower_bound=lower,dual_residual_l1_penalty=penalty,
                  dual_residual_max=float(np.max(abs(residual[1:]))),min_primal_eigenvalue=mineig,
                  min_corrected_dual_eigenvalue=min(float(np.linalg.eigvalsh(Z).min()) for Z in duals),
                  normalization_residual=float(abs(x.value[0]-1)),iterations=int(problem.solver_stats.num_iters),
                  seconds=time.perf_counter()-start)
    report['residual_checks_passed'] = bool(abs(primal-dual)<1e-7 and mineig>-1e-7 and report['normalization_residual']<1e-8)
    report['numerically_converged'] = bool(problem.status==cp.OPTIMAL and report['residual_checks_passed'])
    print(json.dumps(report,indent=2),flush=True)
    return report,dict(Gamma=np.asarray(grams),dual_Z=np.asarray(duals),moments=x.value,objective=objective),words,keys


def strategy_upper(path):
    from hi_membership_seesaw import load_strategy
    from tripartite_benchmark import make_probability_table,hidden_influence_S
    psi,parties,_,diagnostics = load_strategy(path)
    value = hidden_influence_S(make_probability_table(psi,*parties),verbose=False)
    return float(value),diagnostics


def chsh_self_test():
    words = [(),(('A',0,0),),(('A',1,0),),(('C',0,0),),(('C',1,0),)]
    keys = {():0}
    rows,cols = [],[]
    for j,r in enumerate(words):
        for i,l in enumerate(words):
            key = moment_key(tuple(reversed(l))+r)
            if key not in keys: keys[key]=len(keys)
            rows.append(i+5*j);cols.append(keys[key])
    F = sparse.csc_matrix((np.ones(len(rows)),(rows,cols)),shape=(25,len(keys)))
    c = np.zeros(len(keys))
    for x,z,value in [(0,0,-1),(0,1,-1),(1,0,-1),(1,1,1)]:
        c[keys[moment_key((('A',x,0),('C',z,0)))]] += value
    moments = cp.Variable(len(keys))
    G = cp.reshape(F@moments,(5,5),order='F')
    psd = G >> 0
    problem = cp.Problem(cp.Minimize(c@moments),[psd,moments[0]==1])
    problem.solve(solver=cp.CLARABEL,tol_gap_abs=1e-11,tol_gap_rel=1e-11,tol_feas=1e-11,max_threads=1)
    lower,_,_,_ = corrected_dual_bound(F,c,psd.dual_value,5)
    expected = -2*np.sqrt(2)
    assert abs(problem.value-expected)<1e-8 and abs(lower-expected)<1e-8
    assert canonical((('B',0,0),('B',0,1))) is None
    assert canonical((('A',0,0),('A',1,0))) != canonical((('A',1,0),('A',0,0)))
    assert canonical((('A',0,0),('B',0,0),('A',0,0))) == (('B',0,0),)
    print(json.dumps(dict(test='CHSH Q1',value=problem.value,corrected_lower=lower,expected=expected)),flush=True)
    return dict(CHSH_primal=problem.value,CHSH_corrected_lower=lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels',nargs='+',default=['1','1+AB+AC+BC','2','2+ABC','block3'],
                        choices=['1','1+AB+AC+BC','2','2+ABC','3','local2,1,2','local3,1,3','local2,2,2','block2','block3','block4'])
    parser.add_argument('--solver',choices=['CLARABEL','SCS'],default='CLARABEL')
    parser.add_argument('--tol',type=float,default=1e-10)
    parser.add_argument('--max-iters',type=int,default=500)
    parser.add_argument('--strategy',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--self-test-only',action='store_true',help='Check CHSH Tsirelson bound and operator algebra, then exit')
    args = parser.parse_args()
    if args.self_test_only:
        chsh_self_test()
        return
    if args.tol <= 0 or args.max_iters < 1:
        parser.error('tol and max-iters must be positive')
    output = args.output or BASE/'results_npa_333'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    path = args.strategy
    if path is None:
        candidates = list((BASE/'results_seesaw_333').glob('*/best.npz'))
        if candidates:
            path = min(candidates,key=lambda p:float(np.load(p,allow_pickle=False)['raw_S']))
    upper,diagnostics = strategy_upper(path) if path is not None else (None,None)
    # The strategy supplies a comparison only, never a restriction of the NPA feasible set.
    reports = []
    for level in args.levels:
        if level.startswith('block'):
            report,arrays,words,keys = solve_block(int(level[5:]),args.solver,args.tol,args.max_iters,args.verbose)
        else:
            report,arrays,words,keys = solve_level(level,args.solver,args.tol,args.max_iters,args.verbose)
        report.update(strategy_file=str(path) if path else None,seesaw_upper=upper)
        if upper is not None:
            report['remaining_interval'] = upper-report['corrected_quantum_lower_bound']
            if report['remaining_interval'] < -1e-7:
                raise RuntimeError('NPA lower bound is above an achievable strategy: check algebra/solver')
        reports.append(report)
        np.savez(output/f'level_{level}.npz',**arrays)
        (output/f'words_{level}.json').write_text(json.dumps(words))
        (output/f'moment_keys_{level}.json').write_text(json.dumps(list(keys)))
        (output/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('Saved:',output,flush=True)


if __name__ == '__main__':
    main()
