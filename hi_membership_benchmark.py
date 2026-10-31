"""Recover and check paper Eq.(4) using Appendix A, without seesaw.
Scenario [22][333][22]. Farkas convention follows the supplied methods:
maximize b.T@y, A.T@y <= 0, ||y||_1 <= 1.
Reported paper convention is c.T@p >= L (negation of the upper-bound form).
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from tripartite_benchmark import (
    make_state, make_measurements, make_probability_table,
    hidden_influence_S, check_measurements, normalization_error, non_signaling_errors,
)
MA, OA = 2, 2
MB, OB = 3, 3
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

LABELS=make_labels()
DIM=len(LABELS)
LABEL_TO_INDEX={label:i for i,label in enumerate(LABELS)}
A_STRATEGIES=list(itertools.product(range(OA),repeat=MA))
C_STRATEGIES=list(itertools.product(range(OC),repeat=MC))
LAMBDA=list(itertools.product(A_STRATEGIES,C_STRATEGIES))
NLAMBDA=len(LAMBDA)
NW=MB*OB*NLAMBDA
assert (DIM,NLAMBDA,NW)==(34,16,144)


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

def project_probability(P):
    p = []
    for x in range(MA):
        p.append(np.sum(P[0, :, :, x, 0, 0]))
    for y in range(MB):
        for b in range(OB - 1):
            p.append(np.sum(P[:, b, :, 0, y, 0]))
    for z in range(MC):
        p.append(np.sum(P[:, :, 0, 0, 0, z]))
    for x in range(MA):
        for y in range(MB):
            for b in range(OB - 1):
                p.append(np.sum(P[0, b, :, x, y, 0]))
    for y in range(MB):
        for z in range(MC):
            for b in range(OB - 1):
                p.append(np.sum(P[:, b, 0, 0, y, z]))
    return np.asarray(p, dtype=float)

def label_to_string(label):
    kind = label[0]
    if kind == 'PA':
        _, a, x = label
        return f'PA({a}|{x})'
    if kind == 'PB':
        _, b, y = label
        return f'PB({b}|{y})'
    if kind == 'PC':
        _, c, z = label
        return f'PC({c}|{z})'
    if kind == 'PAB':
        _, a, b, x, y = label
        return f'PAB({a}{b}|{x}{y})'
    if kind == 'PBC':
        _, b, c, y, z = label
        return f'PBC({b}{c}|{y}{z})'
    return str(label)

LP_OPTIONS={"dual_feasibility_tolerance":1e-9,"primal_feasibility_tolerance":1e-9}

def membership_test(A,d,R,p):
    b=d+R@p
    result=linprog(np.zeros(A.shape[1]),A_eq=A,b_eq=b,bounds=(0,None),method="highs",options=LP_OPTIONS)
    if result.status not in (0,2): raise RuntimeError(f"Membership unresolved: {result.message}")
    return result,b


def find_farkas_certificate(A,b):
    # y=u-v; maximize b@y subject to A.T@y<=0 and sum(u+v)<=1.
    m=A.shape[0]
    ub=np.vstack([np.hstack([A.T,-A.T]),np.ones((1,2*m))])
    result=linprog(np.r_[-b,b],A_ub=ub,b_ub=np.r_[np.zeros(A.shape[1]),1.],bounds=(0,None),method="highs",options=LP_OPTIONS)
    if not result.success: raise RuntimeError(result.message)
    return result,result.x[:m]-result.x[m:]


def certificate_to_inequality(y,d,R):
    # Upper form: (R.T@y)@p <= -d@y. Negate to match paper Eq.(4).
    return -R.T@y,float(d@y)


def paper_eq4():
    # Independent transcription of Eq.(4), PDF page 4; never fed to free discovery.
    c=np.zeros(DIM)
    terms=[
      (("PA",0,0),-1),(("PA",0,1),-1),
      (("PB",0,0),-1),(("PB",1,0),1),(("PB",0,1),-1),(("PB",1,1),-1),
      (("PAB",0,0,0,0),1),(("PAB",0,0,1,0),1),(("PAB",0,1,0,0),1),
      (("PAB",0,1,1,0),-1),(("PAB",0,0,0,1),2),(("PAB",0,1,0,1),2),
      (("PAB",0,0,1,2),2),(("PAB",0,1,1,2),1),(("PC",0,0),1),
      (("PBC",0,0,0,0),1),(("PBC",0,0,1,0),-1),(("PBC",1,0,1,0),-1),
      (("PBC",0,0,2,0),-1),(("PBC",1,0,2,0),-1),(("PBC",1,0,0,1),-1),
      (("PBC",0,0,1,1),1),(("PBC",1,0,1,1),1),(("PBC",0,0,2,1),-1)]
    for label,value in terms:c[LABEL_TO_INDEX[label]]+=value
    assert np.sum(abs(c))==27
    return c,-2.


def compare_inequalities(c,L,reference,reference_L,tol=1e-7):
    norm=np.sum(abs(c));refnorm=np.sum(abs(reference))
    if norm<1e-14: raise RuntimeError("Zero separator")
    error=float(np.max(abs(c/norm-reference/refnorm)))
    bound_error=float(abs(L/norm-reference_L/refnorm))
    return {"same_up_to_positive_scale":bool(error<tol and bound_error<tol),
            "normalized_coefficient_max_error":error,"normalized_bound_error":bound_error,
            "scale_to_paper":float(refnorm/norm)}


def hi_bound(A,d,R,c):
    # Independently minimize over the projected HI polytope; no target point.
    objective=np.r_[np.zeros(A.shape[1]),c]
    result=linprog(objective,A_eq=np.hstack([A,-R]),b_eq=d,
                   bounds=[(0,None)]*A.shape[1]+[(None,None)]*DIM,
                   method="highs",options=LP_OPTIONS)
    if not result.success:raise RuntimeError(result.message)
    return float(result.fun),result.x[A.shape[1]:]


def eq4_dual_representation(A,d,R,c,L):
    # Separate verification only: represent known Eq.(4) in the Farkas cone.
    # This is NOT the unconstrained discovery result.
    result=linprog(np.zeros(A.shape[0]),A_ub=A.T,b_ub=np.zeros(A.shape[1]),
                   A_eq=np.vstack([-R.T,d.reshape(1,-1)]),b_eq=np.r_[c,L],
                   bounds=(None,None),method="highs",options=LP_OPTIONS)
    if not result.success:raise RuntimeError("Eq.(4) has no verified dual representation")
    y=result.x
    if np.max(A.T@y)>1e-7 or np.max(abs(-R.T@y-c))>1e-7 or abs(d@y-L)>1e-7:
        raise RuntimeError("Invalid Eq.(4) dual representation")
    return y


def physical_bob_completion(B):
    # Optional rounding repair, not literal Appendix A: nearest orthonormal
    # frame to each pair of published normalized rays (symmetric polar factor).
    repaired=[]
    for setting in B:
        rays=[]
        for E in setting[:2]:
            vals,U=np.linalg.eigh(E);rays.append(U[:,np.argmax(vals)])
        V=np.column_stack(rays);vals,U=np.linalg.eigh(V.conj().T@V)
        Q=V@(U*(vals**-.5))@U.conj().T
        first=[np.outer(Q[:,i],Q[:,i].conj()) for i in range(2)]
        repaired.append(first+[np.eye(3)-sum(first)])
    return repaired


def format_inequality(c,L):
    norm=np.sum(abs(c));scale=27/norm
    # Print paper scale only when actually equal to the independent reference.
    ref,refL=paper_eq4()
    if not compare_inequalities(c,L,ref,refL)["same_up_to_positive_scale"]:scale=1/norm
    terms=[f"{value*scale:+.10g} {label_to_string(label)}" for value,label in zip(c,LABELS) if abs(value)>1e-9]
    return " ".join(terms)+f" >= {L*scale:.12g}"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-bob",action="store_true",help="Explicitly repair rounding-induced nonorthogonality; changes Appendix A POVMs")
    parser.add_argument("--output",type=Path,help="Optional NEW directory for numerical certificate and report")
    args=parser.parse_args()
    psi=make_state();A_meas,B,C_meas=make_measurements()
    if args.physical_bob:B=physical_bob_completion(B)
    P=make_probability_table(psi,A_meas,B,C_meas)
    p=project_probability(P)
    herm,complete,mineig=check_measurements(A_meas,B,C_meas)
    ns=non_signaling_errors(P)
    report={"scenario":"[22][333][22]","input_mode":"physical_bob_rounding_repair" if args.physical_bob else "literal_published_rounded",
            "state_norm":float(np.linalg.norm(psi)),"povm_hermiticity_error":float(herm),
            "povm_completeness_error":float(complete),"min_povm_eigenvalue":float(mineig),
            "strictly_physical_within_1e-10":bool(mineig>=-1e-10),
            "min_joint_probability":float(P.min()),"normalization_error":float(normalization_error(P)),
            "non_signaling_errors":list(map(float,ns))}
    A,d,R,descriptions=build_hi_system()
    membership,b=membership_test(A,d,R,p)
    if membership.success:raise RuntimeError("Published target is inside HI at solver tolerance")
    result,y=find_farkas_certificate(A,b)
    if np.max(A.T@y)>1e-8 or b@y<=1e-9 or np.sum(abs(y))>1+1e-8:raise RuntimeError("Unverified Farkas certificate")
    c,L=certificate_to_inequality(y,d,R)
    ref,refL=paper_eq4()
    paper_value=float(ref@p)
    independent_value=float(hidden_influence_S(P,verbose=False))
    if abs(paper_value-independent_value)>1e-12:raise RuntimeError("Independent Eq.(4) implementations disagree")
    bound,point=hi_bound(A,d,R,ref)
    if abs(bound-refL)>1e-7:raise RuntimeError("HI bound differs from paper")
    dual_eq4=eq4_dual_representation(A,d,R,ref,refL)
    found_bound,_=hi_bound(A,d,R,c)
    if found_bound<L-1e-8:raise RuntimeError("Discovered inequality is not HI-valid")
    report.update({"matrix_shape":list(A.shape),"projected_dimension":DIM,
      "membership":"infeasible","farkas_objective_upper":float(b@y),"farkas_max_dual_slack":float(np.max(A.T@y)),
      "discovered_value":float(c@p),"discovered_bound":L,"discovered_gap":float(L-c@p),
      "discovered_normalized_gap":float((L-c@p)/np.sum(abs(c))),"discovered_HI_minimum":found_bound,
      "paper_value":paper_value,"paper_bound":refL,"paper_gap":refL-paper_value,
      "paper_normalized_gap":float((refL-paper_value)/27),"paper_independent_HI_minimum":bound,
      "paper_reported_value":-2.00015,"paper_value_minus_reported":paper_value+2.00015,
      "comparison":compare_inequalities(c,L,ref,refL),
      "eq4_dual_representation_max_slack":float(np.max(A.T@dual_eq4))})
    print(json.dumps(report,indent=2))
    if mineig < -1e-10:print("WARNING: Literal four-decimal Appendix A rays give a nonpositive Bob remainder; no silent repair was made.")
    print("Free Farkas discovery:",format_inequality(c,L))
    print("Matches paper Eq.(4) up to positive scale:",report["comparison"]["same_up_to_positive_scale"])
    print("Independent Eq.(4) validity check: HI minimum =",bound)
    if args.output:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        (args.output/"labels.json").write_text(json.dumps(LABELS,indent=2)+"\n")
        (args.output/"constraints.json").write_text(json.dumps(descriptions,indent=2)+"\n")
        np.savez(args.output/"certificate.npz",state=psi,A_meas=np.asarray(A_meas),B_meas=np.asarray(B),C_meas=np.asarray(C_meas),point=p,
                 Aeq=A,constants=d,p_coeffs=R,farkas_y=y,coefficients=c,HI_bound=L,
                 paper_coefficients=ref,paper_bound=refL,eq4_verification_y=dual_eq4,paper_HI_saturating_point=point)

if __name__=="__main__":
    main()
