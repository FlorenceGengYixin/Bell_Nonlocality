"""Extract a free HI/Farkas separator from an optimized [22][333][22] strategy.
Uses the same LP as hi_membership_benchmark.py, but loads the saved seesaw
state AND POVMs. Eq.(4) is a reference only, never a discovery constraint.
Upper form: R.T@y @ p <= -d.T@y; paper form: c@p >= L.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from hi_membership_benchmark import (
    LABELS, DIM, build_hi_system, membership_test, find_farkas_certificate,
    certificate_to_inequality, project_probability, paper_eq4,
    compare_inequalities, hi_bound, format_inequality,
)
from tripartite_benchmark import (
    make_probability_table, hidden_influence_S, normalization_error,
    non_signaling_errors, check_measurements,
)

BASE = Path(__file__).resolve().parent


def load_strategy(path):
    with np.load(path, allow_pickle=False) as data:
        psi = np.asarray(data["state"], dtype=complex).copy()
        parties = [np.asarray(data[name], dtype=complex).copy() for name in "ABC"]
        saved_S = float(data["raw_S"]) if "raw_S" in data else None
    if psi.shape != (12,) or not np.all(np.isfinite(psi)):
        raise ValueError("Expected a finite 12-component state")
    if abs(np.vdot(psi, psi) - 1) > 1e-9:
        raise ValueError("Saved state is not normalized; refusing silent changes")
    for party, shape in zip(parties, [(2,2,2,2), (3,3,3,3), (2,2,2,2)]):
        if party.shape != shape or not np.all(np.isfinite(party)):
            raise ValueError(f"Expected finite measurement array of shape {shape}")
    herm, complete, min_eig = check_measurements(*parties)
    if herm > 1e-8 or complete > 1e-8 or min_eig < -1e-8:
        raise ValueError("Saved POVMs fail physicality checks")
    return psi, parties, saved_S, dict(hermiticity_error=float(herm),
                                     completeness_error=float(complete),
                                     min_povm_eigenvalue=float(min_eig))


def analyze_strategy(path):
    psi, parties, saved_S, diagnostics = load_strategy(path)
    table = make_probability_table(psi, *parties)
    p = project_probability(table)
    ns = non_signaling_errors(table)
    if table.min() < -1e-8 or normalization_error(table) > 1e-8 or max(ns) > 1e-8:
        raise RuntimeError("Born probabilities fail positivity, normalization or NS")
    A, d, R, descriptions = build_hi_system()
    member, b = membership_test(A, d, R, p)
    if member.success:
        raise RuntimeError("Target belongs to HI; no separating inequality exists")
    _, y = find_farkas_certificate(A, b)
    dual_max, objective = float(np.max(A.T @ y)), float(b @ y)
    if dual_max > 1e-8 or objective <= 1e-9 or np.sum(abs(y)) > 1 + 1e-8:
        raise RuntimeError("Farkas certificate failed independent checks")
    c, L = certificate_to_inequality(y, d, R)
    hi_minimum, saturating = hi_bound(A, d, R, c)
    if hi_minimum < L - 1e-8:
        raise RuntimeError("Discovered separator is not valid over HI")
    reference, reference_L = paper_eq4()  # Comparison only, AFTER free discovery.
    S = float(reference @ p)
    if abs(S - hidden_influence_S(table, verbose=False)) > 1e-12:
        raise RuntimeError("Independent Eq.(4) evaluations disagree")
    if saved_S is not None and abs(saved_S - S) > 1e-9:
        raise RuntimeError("Saved objective disagrees with saved state and POVMs")
    paper_minimum, _ = hi_bound(A, d, R, reference)
    if abs(paper_minimum - reference_L) > 1e-8:
        raise RuntimeError("Independent paper HI-bound check failed")
    comparison = compare_inequalities(c, L, reference, reference_L)
    norm = float(np.sum(abs(c)))
    report = dict(
        scenario="[22][333][22]", strategy_file=str(path.resolve()),
        strategy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        matrix_shape=list(A.shape), projected_dimension=DIM,
        membership="infeasible", farkas_max_dual_slack=dual_max,
        farkas_objective_upper=objective, farkas_l1_norm=float(np.sum(abs(y))),
        separator_value=float(c @ p), separator_bound=L,
        separator_gap=float(L - c @ p),
        normalized_gap=float((L - c @ p) / norm),
        separator_HI_minimum=hi_minimum, comparison=comparison,
        paper_S=S, paper_HI_bound=reference_L, paper_HI_minimum=paper_minimum,
        paper_raw_gap=float(reference_L - S), saved_raw_S=saved_S,
        min_joint_probability=float(table.min()),
        normalization_error=float(normalization_error(table)),
        non_signaling_errors=list(map(float, ns)), **diagnostics,
    )
    arrays = dict(state=psi, A_meas=parties[0], B_meas=parties[1], C_meas=parties[2],
                  point=p, Aeq=A, constants=d, p_coeffs=R, farkas_y=y,
                  coefficients=c, HI_bound=L, paper_coefficients=reference,
                  paper_bound=reference_L, HI_saturating_point=saturating)
    return report, arrays, descriptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, help="best.npz saved by tripartite_seesaw_333.py")
    parser.add_argument("--output", type=Path, help="New output directory; existing results are never overwritten")
    args = parser.parse_args()
    path = args.strategy
    if path is None:
        candidates = list((BASE / "results_seesaw_333").glob("*/best.npz"))
        if not candidates:
            parser.error("Run tripartite_seesaw_333.py first, or supply --strategy /path/to/best.npz")
        path = max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))
    if not path.is_file():
        parser.error(f"Strategy file not found: {path}")
    report, arrays, descriptions = analyze_strategy(path)
    output = args.output or BASE / "results_hi_seesaw_333" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "certificate.npz", **arrays)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "labels.json").write_text(json.dumps(LABELS, indent=2) + "\n")
    (output / "constraints.json").write_text(json.dumps(descriptions, indent=2) + "\n")
    inequality = format_inequality(arrays["coefficients"], float(arrays["HI_bound"]))
    (output / "inequality.txt").write_text(inequality + "\n")
    print(json.dumps(report, indent=2))
    print("Free Farkas inequality:", inequality)
    print("Matches paper Eq.(4) up to positive scale:", report["comparison"]["same_up_to_positive_scale"])
    print("Saved certificate and checks to", output)


if __name__ == "__main__":
    main()
