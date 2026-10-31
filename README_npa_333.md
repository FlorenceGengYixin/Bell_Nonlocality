# NPA for raw Eq.(4): npa_333.py

The script minimizes the raw paper S (HI bound -2), not its L1-normalized version. The NPA relaxation provides a LOWER bound on the dimension-unbounded quantum minimum; a valid seesaw realization provides an UPPER bound. -2.0003 is a rounded comparison from the paper, never an objective constraint, fit target, or fixed moment.

## Improvements over the old scripts

- Canonical moments are shared through a sparse linear map instead of thousands of redundant entry equalities.
- Same-party distinct settings do NOT commute. Different parties commute; Bob projector idempotency and orthogonality are enforced by word reduction.
- A,C are represented by dichotomic observables A=2P-I and C=2P-I, so A_x^2=C_z^2=I. This improves conditioning versus nearly dependent projector words.
- Bob y=1 outcomes 0 and 1 have identical coefficients everywhere in Eq.(4). They are merged into D=P0+P1 exactly, giving the equivalent outcome scenario [22][323][22] for this fixed objective. This does not change its dimension-unbounded quantum optimum: merging and splitting POVM effects are both possible. Bob's other settings retain three outcomes.
- Real symmetric moments suffice for this real-coefficient objective by averaging a feasible complex representation with its complex conjugate. Reversal equality is a choice of real moments, not a claim that distinct same-party settings commute.
- Projective Bob measurements are without loss of generality for the dimension-unbounded relaxation via local dilation. The actual saved qutrit POVMs need not be projectors and are never forced into a projector Gram matrix.

## Word sets and the final block relaxation

Standard levels are generated mechanically: Q1, Q1+AB+AC+BC, Q2, Q2+ABC, Q3. Optional local2,1,2 uses products with local A/B/C lengths at most 2/1/2. It is a valid principal submatrix of Q5, not a full numbered NPA level.

The practical default additionally uses block3, an explicitly labelled conditional/principal-block NPA relaxation. Let W consist of all reduced AC words with local lengths at most 3 (49 words). For each Bob setting and outcome, impose

    Gamma_b[y]_{ij}=<W_i^dag B_b|y W_j> >= 0.

The final outcome is represented as Gamma_AC minus the other outcome matrices, and is also constrained PSD. There are eight 49x49 PSD blocks. Entries with the same canonical moment share one scalar variable, and every setting sums to the same AC moment matrix. All quantum strategies satisfy these constraints: with projectors these are principal blocks indexed by B_b|y W_i. Cross-setting Bob blocks are omitted, so this is an OUTER approximation, not the full numbered NPA hierarchy. Do not claim it is nested with every optional word list or that this particular reduced sequence is complete. The full numbered levels remain available.

## Numerical bound check

For each computed PSD dual Z, negative eigenvalues are projected away and a small positive safety margin is added. If r=c-sum_k F_k^*(Z_k), the code reports

    lower = r_0 - sum_{j>0}|r_j|.

This follows from positivity and |<word>|<=1 for products of bounded projectors/dichotomic observables. It corrects stationarity residuals instead of presenting an approximate primal objective as a bound. The check is in floating-point arithmetic, not an exact rational or interval-arithmetic proof. The raw solver status, primal/dual objectives, minimum eigenvalues, residual penalty and corrected lower bound are all retained.

At block3, tol=1e-10, CLARABEL returned optimal_inaccurate. The corrected lower bound is -2.0002703096954666. The validated saved seesaw strategy gives -2.000270300627504, leaving an interval of about 9.07e-9. Both round to the paper's -2.0003. The raw SDP primal objective -2.000270257178562 lies slightly above the achievable strategy and MUST NOT be used as a quantum lower bound.

For comparison, earlier low-level solves gave approximately Q1=-2.5279373, Q2=-2.1312943, Q2+ABC=-2.0054070. Their looseness was a relaxation-level issue, not something that could be fixed by forcing the answer toward the paper value.

## Run in VS Code / cdd-test

```sh
.venv/bin/python npa_333.py --self-test-only
.venv/bin/python npa_333.py --levels block3 --tol 1e-10
.venv/bin/python npa_333.py
```

The default compares several low levels with block3. Outputs are saved under a new results_npa_333 timestamp directory; old files are not overwritten. --strategy accepts a best.npz from tripartite_seesaw_333.py. Without it, the best saved raw_S under results_seesaw_333 is used for comparison only. No strategy-derived moments enter the SDP.

SCS is optional; use an appropriate iteration budget, e.g. --solver SCS --max-iters 200000. The default CLARABEL is substantially more effective here. An optimal_inaccurate or user_limit status is preserved, not relabelled as converged.

Validation includes the CHSH Q1 minimum -2*sqrt(2) within 1e-8, direct complex-projective Gram realizations, an independent Born evaluation of the merged objective, conditional-block positivity and the corrected-dual formula.

Primary reference: Navascues, Pironio and Acin, https://arxiv.org/abs/0803.4290 . CVXPY explicit PSD constraints expose their dual matrices through dual_value.
