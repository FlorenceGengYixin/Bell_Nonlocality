"""Alternate quantum see-saw and HI/Farkas separation in [22][444][22].
The fixed-witness implementation and model live in tripartite_seesaw_444.
Farkas dual L1 normalization differs from the final coefficient L1 norm,
so outer iterations need not increase the normalized gap. Keep the best.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import tripartite_seesaw_444 as model

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy',type=Path)
    parser.add_argument('--witness',type=Path)
    parser.add_argument('--bob-dim',type=int,choices=[3,4],default=3)
    parser.add_argument('--perturbed',type=int,default=20)
    parser.add_argument('--random',type=int,default=80)
    parser.add_argument('--outer',type=int,default=4)
    parser.add_argument('--max-sweeps',type=int,default=100)
    parser.add_argument('--tol',type=float,default=1e-10)
    parser.add_argument('--perturbation',type=float,default=.08)
    parser.add_argument('--seed',type=int,default=20261002)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if min(args.perturbed,args.random)<0 or min(args.outer,args.max_sweeps)<1 or args.tol<=0 or not 0<=args.perturbation<=1:
        parser.error('Invalid count, tolerance, or perturbation')
    model.configure(args.bob_dim)
    psi,P,base_c = model.load_initial(args.strategy,args.witness)
    known = psi,P
    system = model.build_hi_system()
    base_L = model.witness_support(system,base_c)
    output = args.output or BASE/'results_full_seesaw_444'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    (output/'labels.json').write_text(json.dumps(model.LABELS,indent=2)+'\n')
    best,records = -np.inf,[]

    def save(tag,psi,P,c,L):
        nonlocal best
        value = model.save_strategy(output/(tag+'.npz'),psi,P,c,L)
        gap = float(L-value)
        if gap>best:
            support = model.witness_support(system,c)
            if L>support+1e-8:
                raise RuntimeError('Candidate bound exceeds independent HI minimum')
            best = gap
            model.save_strategy(output/'best.npz',psi,P,c,L)
            report = dict(gap=gap,value=value,HI_bound=float(L),
                          independent_HI_support=support,stage=tag,
                          **model.validate_strategy(psi,P))
            (output/'best.json').write_text(json.dumps(report,indent=2)+'\n')
        return gap

    initial_gap = save('initial',*known,base_c,base_L)
    # The unperturbed incumbent is searched in addition to requested restarts.
    for restart in range(-1,args.perturbed+args.random):
        rng = np.random.default_rng(args.seed+max(restart,0))
        if restart==-1:
            psi,P = known[0].copy(),[p.copy() for p in known[1]]
            kind = 'incumbent'
        elif restart<args.perturbed:
            psi,P = model.perturb_strategy(known,rng,strength=args.perturbation)
            kind = 'perturbed'
        else:
            psi,P = model.random_strategy(rng)
            kind = 'random'
        c,L = base_c.copy(),base_L
        for outer in range(args.outer):
            result = model.run_seesaw(c,np.outer(psi,psi.conj()),*P,
                                     max_sweeps=args.max_sweeps,tol=args.tol)
            psi,P = result['psi'],[result[k] for k in 'ABC']
            tag = f'r{restart+1:03d}_o{outer:03d}'
            gap = save(tag,psi,P,c,L)
            np.save(output/('history_'+tag+'.npy'),result['history'])
            point = model.projected_point(np.outer(psi,psi.conj()),*P)
            separator,status = model.extract_separator(system,point)
            record = dict(restart=restart,type=kind,outer=outer,fixed_gap=gap,
                          sweeps=result['sweeps'],converged=result['converged'],
                          rejected_increasing_sweep=result['rejected_increasing_sweep'],
                          membership_status=status)
            change = np.inf
            if separator is not None:
                new_c,new_L,y = separator
                np.save(output/('farkas_'+tag+'.npy'),y)
                new_gap = save('separator_'+tag,psi,P,new_c,new_L)
                record['separator_gap'] = new_gap
                change = max(np.max(abs(new_c-c)),abs(new_L-L))
                c,L = new_c,new_L
            records.append(record)
            print(json.dumps(record),flush=True)
            (output/'iterations.json').write_text(json.dumps(records,indent=2)+'\n')
            stalled = result['converged'] or result['rejected_increasing_sweep']
            if separator is None or (change<1e-8 and stalled):
                break
    summary = dict(scenario='[22][444][22]',dimensions=list(model.DIMS),
                   projected_dimension=49,hidden_weights=192,
                   initial_gap=initial_gap,best_gap=best,improvement=best-initial_gap,
                   perturbed=args.perturbed,random=args.random,outer_limit=args.outer,
                   max_sweeps=args.max_sweeps,tol=args.tol,seed=args.seed,
                   perturbation=args.perturbation,iterations=len(records),output=str(output))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
