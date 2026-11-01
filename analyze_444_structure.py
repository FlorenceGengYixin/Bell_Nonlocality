"""Exact coefficient symmetries and numerical support of a [22][444][22] strategy.

Outcome relabelling is an affine change in the 49-coordinate chart because
Bob outcome 3 is eliminated. Test the full affine inequality, not a subset
of random behaviors. No positive rescaling or Alice/Charlie relabelling is
included in the Bob orbit. Numerical ranks are local support diagnostics,
not dimension witnesses for every realization of the behavior.
"""
import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import tripartite_seesaw_444 as model

BASE = Path(__file__).resolve().parent
LABELS = model.LABELS
INDEX = model.LABEL_TO_INDEX
SETTING_PERMS = list(itertools.permutations(range(3)))
OUTCOME_PERMS = list(itertools.permutations(range(4)))


def unpack(c):
    """Five features: PB, PAB(x=0,1), PBC(z=0,1), including outcome 3."""
    features = np.zeros((3,4,5),dtype=np.int64)
    for y in range(3):
        for b in range(3):
            features[y,b] = [c[INDEX[('PB',b,y)]],
                             c[INDEX[('PAB',0,b,0,y)]],
                             c[INDEX[('PAB',0,b,1,y)]],
                             c[INDEX[('PBC',b,0,y,0)]],
                             c[INDEX[('PBC',b,0,y,1)]]]
    return features


def canonicalize(features,c):
    """Eliminate outcome 3 using normalization and marginal consistency."""
    out = np.zeros(49,dtype=np.int64)
    for kind in ('PA','PC'):
        for setting in range(2):
            out[INDEX[(kind,0,setting)]] = c[INDEX[(kind,0,setting)]]
    constant = 0
    for y in range(3):
        ref = features[y,3]
        constant += int(ref[0])
        for x in range(2):
            out[INDEX[('PA',0,x)]] += ref[1+x]
            out[INDEX[('PC',0,x)]] += ref[3+x]
        for b in range(3):
            f = features[y,b]-ref
            out[INDEX[('PB',b,y)]] = f[0]
            for x in range(2):
                out[INDEX[('PAB',0,b,x,y)]] = f[1+x]
                out[INDEX[('PBC',b,0,y,x)]] = f[3+x]
    return out,constant


def relabel(c,setting_perm,outcome_perms):
    """Push old labels y,b to setting_perm[y],outcome_perms[y][b]."""
    features = unpack(c)
    transformed = np.zeros_like(features)
    for y in range(3):
        transformed[setting_perm[y],list(outcome_perms[y])] = features[y]
    return canonicalize(transformed,c)


def search_symmetries(c,bound):
    stabilizers,orbit = [],set()
    for setting_perm in SETTING_PERMS:
        for ops in itertools.product(OUTCOME_PERMS,repeat=3):
            cp,constant = relabel(c,setting_perm,ops)
            bp = bound-constant
            orbit.add(tuple(map(int,cp))+(int(bp),))
            if bp==bound and np.array_equal(cp,c):
                stabilizers.append(dict(settings=setting_perm,outcomes=ops))
    order = len(SETTING_PERMS)*len(OUTCOME_PERMS)**3
    assert order==82944 and len(orbit)*len(stabilizers)==order
    return dict(group_order=order,stabilizer_order=len(stabilizers),
                orbit_size=len(orbit),stabilizers=stabilizers,
                convention='Exact canonical affine inequalities; no rescaling quotient')


def ac_swap(c):
    out = np.zeros_like(c)
    for value,label in zip(c,LABELS):
        kind = label[0]
        if kind=='PA': target=('PC',label[1],label[2])
        elif kind=='PC': target=('PA',label[1],label[2])
        elif kind=='PB': target=label
        elif kind=='PAB':
            _,a,b,x,y=label;target=('PBC',b,a,y,x)
        else:
            _,b,a,y,x=label;target=('PAB',a,b,x,y)
        out[INDEX[target]] += value
    return out


def exact_affine_test(c):
    """Check a permutation involving eliminated outcome 3 on all local vertices.

    The 4*64*4=1024 deterministic local behaviors affinely span this
    projected no-signalling coordinate space. This also checks direction
    and constant conventions independently of canonicalize.
    """
    sp=(2,0,1);ops=((3,1,0,2),(2,3,1,0),(1,0,3,2))
    cp,k = relabel(c,sp,ops)
    def point(a,b,z):
        p=[]
        for label in LABELS:
            kind=label[0]
            if kind=='PA':v=a[label[2]]==label[1]
            elif kind=='PC':v=z[label[2]]==label[1]
            elif kind=='PB':v=b[label[2]]==label[1]
            elif kind=='PAB':
                _,aa,bb,x,y=label;v=a[x]==aa and b[y]==bb
            else:
                _,bb,cc,y,x=label;v=b[y]==bb and z[x]==cc
            p.append(int(v))
        return np.array(p,dtype=np.int64)
    for a in itertools.product(range(2),repeat=2):
        for b in itertools.product(range(4),repeat=3):
            # cp uses pushed coefficients; recover the old outcomes at new labels.
            inverse=tuple(ops[y].index(b[sp[y]]) for y in range(3))
            for z in itertools.product(range(2),repeat=2):
                assert int(cp@point(a,b,z)+k)==int(c@point(a,inverse,z))


def latest_strategy():
    candidates = list((BASE/'results_full_seesaw_444').glob('*/best.npz'))
    candidates += list((BASE/'results_seesaw_444').glob('*/best.npz'))
    best,path = -np.inf,None
    for candidate in sorted(candidates):
        with np.load(candidate,allow_pickle=False) as data:
            gap=float(data['HI_bound']-data['coefficients']@data['point'])
        if gap>best:
            best,path=gap,candidate
    if path is None:
        raise FileNotFoundError('Run a 444 search first or provide --strategy')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy',type=Path)
    parser.add_argument('--scale',type=int,default=20)
    parser.add_argument('--rank-tol',type=float,default=1e-9,
                        help='Absolute eigenvalue cutoff for trace-one reduced states')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.scale<1 or args.rank_tol<=0:
        parser.error('Scale and rank tolerance must be positive')
    path=args.strategy or latest_strategy()
    with np.load(path,allow_pickle=False) as data:
        psi=data['state'].copy();P=[data[k].copy() for k in 'ABC']
        c=data['coefficients'].copy();L=float(data['HI_bound'])
    if P[1].shape[:2]!=(3,4) or P[1].shape[-1] not in (3,4) or c.shape!=(49,):
        raise ValueError('Expected a three-setting four-outcome strategy')
    model.configure(P[1].shape[-1])
    diagnostics=model.validate_strategy(psi,P)
    norm=float(sum(abs(c)));c,L=c/norm,L/norm
    integer=np.rint(args.scale*c).astype(np.int64)
    bound=int(round(args.scale*L))
    error=max(float(np.max(abs(args.scale*c-integer))),abs(args.scale*L-bound))
    if error>1e-8:
        raise ValueError('Witness cannot be reconstructed at the requested scale')
    c,L=integer/args.scale,bound/args.scale
    support=model.witness_support(model.build_hi_system(),c)
    if abs(support-L)>1e-8:
        raise RuntimeError('Independent HI support does not match the saved bound')
    exact_affine_test(integer)
    symmetry=search_symmetries(integer,bound)
    features=unpack(integer)
    clusters=[]
    for y in range(3):
        groups={}
        for b in range(4):groups.setdefault(tuple(features[y,b]),[]).append(b)
        clusters.append(list(groups.values()))
    rho=np.outer(psi,psi.conj())
    point=model.projected_point(rho,*P)
    value=float(c@point)
    if abs(value-model.bell_value(rho,c,*P))>1e-12:
        raise RuntimeError('Bell and probability objectives disagree')
    merged_value = sum(integer[INDEX[('PA',0,x)]]*point[INDEX[('PA',0,x)]]
                       +integer[INDEX[('PC',0,x)]]*point[INDEX[('PC',0,x)]] for x in range(2))
    merged_residual = 0.
    for y,groups in enumerate(clusters):
        effects = [P[1][y,group].sum(axis=0) for group in groups]
        merged_residual = max(merged_residual,float(abs(sum(effects)-np.eye(model.DIMS[1])).max()))
        for group,E in zip(groups,effects):
            f = features[y,group[0]]
            operator = f[0]*np.kron(np.kron(np.eye(2),E),np.eye(2))
            for x in range(2):
                operator += f[1+x]*np.kron(np.kron(P[0][x,0],E),np.eye(2))
                operator += f[3+x]*np.kron(np.kron(np.eye(2),E),P[2][x,0])
            merged_value += float(np.trace(rho@operator).real)
    merged_value /= args.scale
    if abs(merged_value-value)>1e-12 or merged_residual>1e-10:
        raise RuntimeError('Outcome merging failed to preserve the objective or completeness')
    permutation_checks=[]
    identity=tuple(range(4))
    for sp in SETTING_PERMS:
        cp,k=relabel(integer,sp,(identity,)*3)
        B=np.empty_like(P[1])
        for y in range(3):B[sp[y]]=P[1][y]
        pp=model.projected_point(rho,P[0],B,P[2])
        v=float((cp@pp+k)/args.scale)
        beta=model.witness_support(model.build_hi_system(),cp/args.scale)+k/args.scale
        permutation_checks.append(dict(settings=sp,quantum_value=v,support=beta,
                                       identical_coefficients=np.array_equal(cp,integer)))
    tensor=psi.reshape(model.DIMS)
    spectra={}
    for i,name in enumerate('ABC'):
        matrix=np.moveaxis(tensor,i,0).reshape(model.DIMS[i],-1)
        eig=np.linalg.eigvalsh(matrix@matrix.conj().T)
        spectra[name]=dict(eigenvalues=eig.tolist(),rank=int(sum(eig>args.rank_tol)))
    report=dict(source_strategy=str(path.resolve()),scenario='[22][444][22]',
                dimensions=list(model.DIMS),projected_dimension=49,hidden_weights=192,
                integer_scale=args.scale,integer_bound=bound,integer_l1=int(sum(abs(integer))),
                nonzero_terms=int(np.count_nonzero(integer)),integer_reconstruction_error=error,
                normalized_HI_bound=L,independent_HI_support=support,quantum_value=value,gap=L-value,
                outcome_features=features.tolist(),feature_order=['PB','PAB_x0','PAB_x1','PBC_z0','PBC_z1'],
                identical_outcome_classes=clusters,reduced_outcomes=[len(g) for g in clusters],
                merged_quantum_value=merged_value,merged_value_error=abs(merged_value-value),
                merged_POVM_completeness_error=merged_residual,
                simple_AC_swap_invariant=bool(np.array_equal(ac_swap(integer),integer)),
                setting_checks=permutation_checks,
                setting_support_spread=float(np.ptp([r['support'] for r in permutation_checks])),
                setting_value_spread=float(np.ptp([r['quantum_value'] for r in permutation_checks])),
                reduced_state_spectra=spectra,rank_absolute_tolerance=args.rank_tol,
                exact_relabelling_test_vertices=1024,**diagnostics,**symmetry)
    output=args.output or BASE/'results_analyze_444_structure'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    print('Saved to',output,flush=True)


if __name__=='__main__':
    main()
