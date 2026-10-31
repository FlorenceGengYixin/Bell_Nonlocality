"""Fixed Eq.(4) seesaw for [22][333][22], local dimensions 2 x 3 x 2.
Sweep: rho -> A -> B -> C -> rho. Values use ||c||_1=1; raw S=27*value.
No global-optimum claim. Results are saved in a new directory per run.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import cvxpy as cp
from tripartite_benchmark import make_state, make_measurements

DA, DB, DC = 2, 3, 2
MA, MB, MC = 2, 3, 2
OA, OB, OC = 2, 3, 2
DIM = DA * DB * DC
DIMS, SETTINGS, OUTCOMES = (DA, DB, DC), (MA, MB, MC), (OA, OB, OC)
MAX_SWEEPS, TOL = 100, 1e-10
IDENTITIES = [np.eye(d, dtype=complex) for d in DIMS]


def make_labels():
    labels = [("PA", 0, x) for x in range(MA)]
    labels += [("PB", b, y) for y in range(MB) for b in range(OB - 1)]
    labels += [("PC", 0, z) for z in range(MC)]
    labels += [("PAB", 0, b, x, y) for x in range(MA) for y in range(MB) for b in range(OB - 1)]
    labels += [("PBC", b, 0, y, z) for y in range(MB) for z in range(MC) for b in range(OB - 1)]
    return labels


LABELS = make_labels()
LABEL_INDEX = {label: i for i, label in enumerate(LABELS)}


def paper_inequality_coefficients():
    terms = [
        (("PA",0,0),-1), (("PA",0,1),-1), (("PB",0,0),-1),
        (("PB",1,0),1), (("PB",0,1),-1), (("PB",1,1),-1),
        (("PAB",0,0,0,0),1), (("PAB",0,0,1,0),1), (("PAB",0,1,0,0),1),
        (("PAB",0,1,1,0),-1), (("PAB",0,0,0,1),2), (("PAB",0,1,0,1),2),
        (("PAB",0,0,1,2),2), (("PAB",0,1,1,2),1), (("PC",0,0),1),
        (("PBC",0,0,0,0),1), (("PBC",0,0,1,0),-1), (("PBC",1,0,1,0),-1),
        (("PBC",0,0,2,0),-1), (("PBC",1,0,2,0),-1), (("PBC",1,0,0,1),-1),
        (("PBC",0,0,1,1),1), (("PBC",1,0,1,1),1), (("PBC",0,0,2,1),-1),
    ]
    c = np.zeros(len(LABELS))
    for label, value in terms:
        c[LABEL_INDEX[label]] += value
    assert np.sum(np.abs(c)) == 27
    return c / 27


def term_indices(label):
    kind = label[0]
    if kind in ("PA", "PB", "PC"):
        return {("PA", "PB", "PC").index(kind): (label[2], label[1])}
    if kind == "PAB":
        _, a, b, x, y = label
        return {0: (x, a), 1: (y, b)}
    if kind == "PBC":
        _, b, c, y, z = label
        return {1: (y, b), 2: (z, c)}
    raise ValueError(label)


def hermitian(M):
    return (M + M.conj().T) / 2


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


def normalize_povm(effects):
    """Remove tiny SDP residuals using PSD projection and a common congruence."""
    positive = []
    for effect in effects:
        values, vectors = np.linalg.eigh(hermitian(effect))
        if values.min() < -1e-6:
            raise RuntimeError("SDP positivity residual exceeds tolerance")
        positive.append((vectors * np.maximum(values, 0)) @ vectors.conj().T)
    values, vectors = np.linalg.eigh(sum(positive))
    if values.min() <= 1e-12:
        raise RuntimeError("Singular POVM sum")
    inv = (vectors * values**-0.5) @ vectors.conj().T
    return np.asarray([hermitian(inv @ E @ inv) for E in positive])


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


def initial_strategy():
    psi = make_state()
    A, B, C = make_measurements()
    # Published rounded Bob rays are slightly nonorthogonal: repair explicitly.
    for y in range(MB):
        V = np.column_stack([np.linalg.eigh(E)[1][:, -1] for E in B[y][:2]])
        U, _, Vh = np.linalg.svd(V, full_matrices=False)
        Q = U @ Vh
        first = [np.outer(Q[:, b], Q[:, b].conj()) for b in range(2)]
        B[y] = first + [np.eye(DB) - sum(first)]
    parties = [np.asarray(p) for p in (A, B, C)]
    validate_strategy(psi, parties)
    return psi, parties


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--max-sweeps", type=int, default=MAX_SWEEPS)
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.restarts < 1 or args.max_sweeps < 1 or args.tol <= 0:
        parser.error("restarts, max-sweeps and tol must be positive")
    output = args.output or Path(__file__).resolve().parent / "results_seesaw_333" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    coeff, rng = paper_inequality_coefficients(), np.random.default_rng(args.seed)
    known, records, best = initial_strategy(), [], None
    for restart in range(args.restarts):
        if restart == 0:
            (psi, parties), kind = known, "paper_polar_repaired"
        elif restart <= (args.restarts - 1)//2:
            (psi, parties), kind = perturb_strategy(known, rng), "perturbed_paper"
        else:
            (psi, parties), kind = random_strategy(rng), "random"
        result = run_seesaw(coeff, np.outer(psi, psi.conj()), *parties,
                            max_sweeps=args.max_sweeps, tol=args.tol)
        diagnostics = validate_strategy(result["psi"], [result[p] for p in "ABC"])
        record = dict(restart=restart, start=kind, normalized_value=result["value"],
                      raw_S=27*result["value"], raw_gap=-2-27*result["value"],
                      converged=result["converged"], sweeps=result["sweeps"],
                      rejected_increasing_sweep=result["rejected_increasing_sweep"], **diagnostics)
        records.append(record)
        print(json.dumps(record), flush=True)
        np.save(output/f"history_{restart:03d}.npy", result["history"])
        if best is None or result["value"] < best["value"]:
            best = result
            report = dict(record, seed=args.seed, restarts_requested=args.restarts,
                          max_sweeps=args.max_sweeps, tol=args.tol, scenario="[22][333][22]",
                          dimensions=list(DIMS), normalized_HI_bound=-2/27)
            np.savez(output/"best.npz", state=best["psi"], rho=best["rho"],
                     A=best["A"], B=best["B"], C=best["C"], coefficients=coeff,
                     history=best["history"], raw_S=27*best["value"])
            (output/"best.json").write_text(json.dumps(report, indent=2)+"\n")
        with (output/"restarts.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(records)
    print(f"Best raw S={27*best['value']:.15g}; saved to {output}", flush=True)


if __name__ == "__main__":
    main()
