import numpy as np
import cvxpy as cp

from tripartite_benchmark import (
    make_state,
    make_measurements,
)

# Scenario

DA = 2
DB = 3
DC = 2

MA = 2
MB = 4
MC = 2

OA = 2
OB = 3
OC = 2

DIM = DA * DB * DC

# Numerical parameters

MAX_SWEEPS = 50
TOL = 1e-8

SCS_EPS = 1e-8
SCS_MAX_ITERS = 100000

# Labels
# Same projected coordinates as before:
# PA(0|x)
# PB(b|y), b = 0,1
# PC(0|z)
# PAB(0,b|x,y), b = 0,1
# PBC(b,0|y,z), b = 0,1

def make_labels():
    labels = []

    for x in range(MA):
        labels.append(
            ("PA", 0, x)
        )

    for y in range(MB):
        for b in range(OB - 1):
            labels.append(
                ("PB", b, y)
            )

    for z in range(MC):
        labels.append(
            ("PC", 0, z)
        )

    for x in range(MA):
        for y in range(MB):
            for b in range(OB - 1):
                labels.append(
                    (
                        "PAB",
                        0,
                        b,
                        x,
                        y
                    )
                )

    for y in range(MB):
        for z in range(MC):
            for b in range(OB - 1):
                labels.append(
                    (
                        "PBC",
                        b,
                        0,
                        y,
                        z
                    )
                )

    return labels

LABELS = make_labels()

LABEL_INDEX = {
    label: i
    for i, label in enumerate(LABELS)
}

# Tensor helpers

IA = np.eye(DA, dtype=complex)
IB = np.eye(DB, dtype=complex)
IC = np.eye(DC, dtype=complex)

def kron3(A, B, C):
    return np.kron(
        np.kron(A, B),
        C
    )

# Convert projected inequality c into Bell operator
# S = c . p

def bell_operator(
    coeff,
    A,
    B,
    C
):
    Bell = np.zeros(
        (DIM, DIM),
        dtype=complex
    )

    for coefficient, label in zip(
        coeff,
        LABELS
    ):
        if abs(coefficient) < 1e-14:
            continue

        kind = label[0]

        # PA(a|x)

        if kind == "PA":
            _, a, x = label

            M = kron3(
                A[x][a],
                IB,
                IC
            )

        # PB(b|y)

        elif kind == "PB":
            _, b, y = label

            M = kron3(
                IA,
                B[y][b],
                IC
            )

        # PC(c|z)

        elif kind == "PC":
            _, c, z = label

            M = kron3(
                IA,
                IB,
                C[z][c]
            )

        # PAB(a,b|x,y)

        elif kind == "PAB":
            _, a, b, x, y = label

            M = kron3(
                A[x][a],
                B[y][b],
                IC
            )

        # PBC(b,c|y,z)

        elif kind == "PBC":
            _, b, c, y, z = label

            M = kron3(
                IA,
                B[y][b],
                C[z][c]
            )

        else:
            raise ValueError(
                f"Unknown label: {label}"
            )

        Bell += coefficient * M

    # remove tiny numerical non-Hermiticity

    Bell = (
        Bell
        +
        Bell.conj().T
    ) / 2

    return Bell

# Evaluate inequality

def bell_value(
    rho,
    coeff,
    A,
    B,
    C
):
    Bell = bell_operator(
        coeff,
        A,
        B,
        C
    )

    value = np.trace(
        rho @ Bell
    )

    return float(
        np.real(value)
    )

# State optimization
# min_rho Tr(rho Bell)
# = minimum eigenvalue of Bell

def optimize_state(
    coeff,
    A,
    B,
    C
):
    Bell = bell_operator(
        coeff,
        A,
        B,
        C
    )

    eigvals, eigvecs = np.linalg.eigh(
        Bell
    )

    index = np.argmin(
        eigvals
    )

    psi = eigvecs[:, index]

    psi = (
        psi
        /
        np.linalg.norm(psi)
    )

    rho = np.outer(
        psi,
        psi.conj()
    )

    return (
        rho,
        psi,
        float(
            eigvals[index]
        )
    )

# CVXPY trace helper
# Return real Tr(rho M)

def expectation_expr(
    rho,
    M
):
    return cp.real(
        cp.trace(
            rho @ M
        )
    )

# Alice SDP update
# Fix rho, B, C.
# Optimize all A[a|x].

def optimize_alice(
    rho,
    coeff,
    B,
    C
):
    Avar = []

    constraints = []

    for x in range(MA):
        setting = []

        for a in range(OA):
            X = cp.Variable(
                (DA, DA),
                hermitian=True
            )

            setting.append(X)

            constraints.append(
                X >> 0
            )

        constraints.append(
            sum(setting)
            ==
            np.eye(DA)
        )

        Avar.append(setting)

    objective_terms = []

    for coefficient, label in zip(
        coeff,
        LABELS
    ):
        if abs(coefficient) < 1e-14:
            continue

        kind = label[0]

        # Alice marginal

        if kind == "PA":
            _, a, x = label

            M = cp.kron(
                cp.kron(
                    Avar[x][a],
                    IB
                ),
                IC
            )

        # Terms independent of Alice
        # They are constant during this SDP and can be omitted.

        elif kind == "PB":
            continue

        elif kind == "PC":
            continue

        # AB

        elif kind == "PAB":
            _, a, b, x, y = label

            M = cp.kron(
                cp.kron(
                    Avar[x][a],
                    B[y][b]
                ),
                IC
            )

        # BC independent of Alice

        elif kind == "PBC":
            continue

        else:
            raise ValueError(label)

        objective_terms.append(
            coefficient
            *
            expectation_expr(
                rho,
                M
            )
        )

    objective = cp.Minimize(
        cp.sum(
            objective_terms
        )
    )

    problem = cp.Problem(
        objective,
        constraints
    )

    problem.solve(
        solver=cp.SCS,
        eps=SCS_EPS,
        max_iters=SCS_MAX_ITERS,
        verbose=False
    )

    if problem.status not in (
        "optimal",
        "optimal_inaccurate"
    ):
        raise RuntimeError(
            "Alice SDP failed: "
            + problem.status
        )

    Anew = []

    for x in range(MA):
        setting = []

        for a in range(OA):
            X = Avar[x][a].value

            X = (
                X
                +
                X.conj().T
            ) / 2

            setting.append(X)

        Anew.append(setting)

    return Anew

# Bob SDP update
# Fix rho, A, C.
# Bob:
# y = 0,1,2,3
# b = 0,1,2

def optimize_bob(
    rho,
    coeff,
    A,
    C
):
    Bvar = []

    constraints = []

    for y in range(MB):
        setting = []

        for b in range(OB):
            X = cp.Variable(
                (DB, DB),
                hermitian=True
            )

            setting.append(X)

            constraints.append(
                X >> 0
            )

        constraints.append(
            sum(setting)
            ==
            np.eye(DB)
        )

        Bvar.append(setting)

    objective_terms = []

    for coefficient, label in zip(
        coeff,
        LABELS
    ):
        if abs(coefficient) < 1e-14:
            continue

        kind = label[0]

        # PA independent of Bob

        if kind == "PA":
            continue

        # PB

        elif kind == "PB":
            _, b, y = label

            M = cp.kron(
                cp.kron(
                    IA,
                    Bvar[y][b]
                ),
                IC
            )

        # PC independent of Bob

        elif kind == "PC":
            continue

        # PAB

        elif kind == "PAB":
            _, a, b, x, y = label

            M = cp.kron(
                cp.kron(
                    A[x][a],
                    Bvar[y][b]
                ),
                IC
            )

        # PBC

        elif kind == "PBC":
            _, b, c, y, z = label

            M = cp.kron(
                cp.kron(
                    IA,
                    Bvar[y][b]
                ),
                C[z][c]
            )

        else:
            raise ValueError(label)

        objective_terms.append(
            coefficient
            *
            expectation_expr(
                rho,
                M
            )
        )

    objective = cp.Minimize(
        cp.sum(
            objective_terms
        )
    )

    problem = cp.Problem(
        objective,
        constraints
    )

    problem.solve(
        solver=cp.SCS,
        eps=SCS_EPS,
        max_iters=SCS_MAX_ITERS,
        verbose=False
    )

    if problem.status not in (
        "optimal",
        "optimal_inaccurate"
    ):
        raise RuntimeError(
            "Bob SDP failed: "
            + problem.status
        )

    Bnew = []

    for y in range(MB):
        setting = []

        for b in range(OB):
            X = Bvar[y][b].value

            X = (
                X
                +
                X.conj().T
            ) / 2

            setting.append(X)

        Bnew.append(setting)

    return Bnew

# Charlie SDP update

def optimize_charlie(
    rho,
    coeff,
    A,
    B
):
    Cvar = []

    constraints = []

    for z in range(MC):
        setting = []

        for c in range(OC):
            X = cp.Variable(
                (DC, DC),
                hermitian=True
            )

            setting.append(X)

            constraints.append(
                X >> 0
            )

        constraints.append(
            sum(setting)
            ==
            np.eye(DC)
        )

        Cvar.append(setting)

    objective_terms = []

    for coefficient, label in zip(
        coeff,
        LABELS
    ):
        if abs(coefficient) < 1e-14:
            continue

        kind = label[0]

        # PA independent of Charlie

        if kind == "PA":
            continue

        # PB independent of Charlie

        elif kind == "PB":
            continue

        # PC

        elif kind == "PC":
            _, c, z = label

            M = cp.kron(
                cp.kron(
                    IA,
                    IB
                ),
                Cvar[z][c]
            )

        # PAB independent of Charlie

        elif kind == "PAB":
            continue

        # PBC

        elif kind == "PBC":
            _, b, c, y, z = label

            M = cp.kron(
                cp.kron(
                    IA,
                    B[y][b]
                ),
                Cvar[z][c]
            )

        else:
            raise ValueError(label)

        objective_terms.append(
            coefficient
            *
            expectation_expr(
                rho,
                M
            )
        )

    objective = cp.Minimize(
        cp.sum(
            objective_terms
        )
    )

    problem = cp.Problem(
        objective,
        constraints
    )

    problem.solve(
        solver=cp.SCS,
        eps=SCS_EPS,
        max_iters=SCS_MAX_ITERS,
        verbose=False
    )

    if problem.status not in (
        "optimal",
        "optimal_inaccurate"
    ):
        raise RuntimeError(
            "Charlie SDP failed: "
            + problem.status
        )

    Cnew = []

    for z in range(MC):
        setting = []

        for c in range(OC):
            X = Cvar[z][c].value

            X = (
                X
                +
                X.conj().T
            ) / 2

            setting.append(X)

        Cnew.append(setting)

    return Cnew

# Random projective qutrit measurement

def random_qutrit_measurement(
    rng
):
    X = (
        rng.normal(
            size=(DB, DB)
        )
        +
        1j
        *
        rng.normal(
            size=(DB, DB)
        )
    )

    Q, _ = np.linalg.qr(X)

    return [
        np.outer(
            Q[:, b],
            Q[:, b].conj()
        )
        for b in range(DB)
    ]

# Old Eq. (4), normalized so ||c||_1 = 1
# This is our sanity-check inequality.
# Raw Eq.(4) coefficients have ||c||_1 = 27.

def paper_inequality_coefficients():
    c = np.zeros(
        len(LABELS)
    )

    def add(
        label,
        value
    ):
        c[
            LABEL_INDEX[label]
        ] += value

    # Eq. (4)

    add(
        ("PA", 0, 0),
        -1
    )

    add(
        ("PA", 0, 1),
        -1
    )

    add(
        ("PB", 0, 0),
        -1
    )

    add(
        ("PB", 1, 0),
        +1
    )

    add(
        ("PB", 0, 1),
        -1
    )

    add(
        ("PB", 1, 1),
        -1
    )

    add(
        ("PAB", 0, 0, 0, 0),
        +1
    )

    add(
        ("PAB", 0, 0, 1, 0),
        +1
    )

    add(
        ("PAB", 0, 1, 0, 0),
        +1
    )

    add(
        ("PAB", 0, 1, 1, 0),
        -1
    )

    add(
        ("PAB", 0, 0, 0, 1),
        +2
    )

    add(
        ("PAB", 0, 1, 0, 1),
        +2
    )

    add(
        ("PAB", 0, 0, 1, 2),
        +2
    )

    add(
        ("PAB", 0, 1, 1, 2),
        +1
    )

    add(
        ("PC", 0, 0),
        +1
    )

    add(
        ("PBC", 0, 0, 0, 0),
        +1
    )

    add(
        ("PBC", 0, 0, 1, 0),
        -1
    )

    add(
        ("PBC", 1, 0, 1, 0),
        -1
    )

    add(
        ("PBC", 0, 0, 2, 0),
        -1
    )

    add(
        ("PBC", 1, 0, 2, 0),
        -1
    )

    add(
        ("PBC", 1, 0, 0, 1),
        -1
    )

    add(
        ("PBC", 0, 0, 1, 1),
        +1
    )

    add(
        ("PBC", 1, 0, 1, 1),
        +1
    )

    add(
        ("PBC", 0, 0, 2, 1),
        -1
    )

    # Observable normalization

    norm = np.sum(
        np.abs(c)
    )

    c /= norm

    return c

# Full inner see-saw

def run_seesaw(
    coeff,
    rho,
    A,
    B,
    C,
    max_sweeps=MAX_SWEEPS,
    tol=TOL
):
    history = []

    initial_value = bell_value(
        rho,
        coeff,
        A,
        B,
        C
    )

    print(
        "\nInitial Bell value:",
        initial_value
    )

    previous = initial_value

    for sweep in range(
        max_sweeps
    ):
        # 1. State

        rho, psi, state_value = (
            optimize_state(
                coeff,
                A,
                B,
                C
            )
        )

        # 2. Alice

        A = optimize_alice(
            rho,
            coeff,
            B,
            C
        )

        # 3. Bob

        B = optimize_bob(
            rho,
            coeff,
            A,
            C
        )

        # 4. Charlie

        C = optimize_charlie(
            rho,
            coeff,
            A,
            B
        )

        # 5. Re-optimize state after all measurements

        rho, psi, current = (
            optimize_state(
                coeff,
                A,
                B,
                C
            )
        )

        history.append(
            current
        )

        improvement = (
            previous
            -
            current
        )

        print(
            f"sweep "
            f"{sweep + 1:3d}"
            f"   S = "
            f"{current:.12f}"
            f"   improvement = "
            f"{improvement:.3e}"
        )

        if abs(
            improvement
        ) < tol:
            print(
                "\nConverged."
            )

            break

        previous = current

    return {
        "rho": rho,
        "psi": psi,
        "A": A,
        "B": B,
        "C": C,
        "value": current,
        "history": np.asarray(
            history
        )
    }

# Main sanity test

def main():
    print(
        "=" * 72
    )

    print(
        "Tripartite full quantum see-saw"
    )

    print(
        "Scenario [22][3333][22]"
    )

    print(
        "Dimensions: 2 x 3 x 2"
    )

    print(
        "=" * 72
    )

    # Original paper strategy

    psi = make_state()

    A, B_original, C = (
        make_measurements()
    )

    rho = np.outer(
        psi,
        psi.conj()
    )

    # Add random B3

    rng = np.random.default_rng(
        12345
    )

    B = [
        [
            E.copy()
            for E in setting
        ]
        for setting
        in B_original
    ]

    B.append(
        random_qutrit_measurement(
            rng
        )
    )

    # Paper inequality

    coeff = (
        paper_inequality_coefficients()
    )

    print(
        "\n||c||_1 =",
        np.sum(
            np.abs(coeff)
        )
    )

    print(
        "\nImportant:"
    )

    print(
        "The paper inequality has zero coefficients "
        "for Bob y=3."
    )

    print(
        "Therefore this run is only a sanity check "
        "of the inner see-saw."
    )

    # Run

    result = run_seesaw(
        coeff,
        rho,
        A,
        B,
        C
    )

    # Final result

    print(
        "\n" + "=" * 72
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 72
    )

    print(
        "\nFinal normalized Bell value:"
    )

    print(
        result["value"]
    )

    # Raw Eq.(4) normalization
    raw_value = (
        27
        *
        result["value"]
    )

    print(
        "\nEquivalent raw Eq.(4) value:"
    )

    print(
        raw_value
    )

    print(
        "\nPaper explicit strategy:"
    )

    print(
        "-2.000152828651"
    )

    print(
        "\nPaper/NPA approximate optimum:"
    )

    print(
        "-2.0003"
    )

    # Save result

    np.save(
        "seesaw_best_state.npy",
        result["psi"]
    )

    np.save(
        "seesaw_best_A.npy",
        np.asarray(
            result["A"]
        )
    )

    np.save(
        "seesaw_best_B.npy",
        np.asarray(
            result["B"]
        )
    )

    np.save(
        "seesaw_best_C.npy",
        np.asarray(
            result["C"]
        )
    )

    np.save(
        "seesaw_history.npy",
        result["history"]
    )

    print(
        "\nSaved optimized strategy."
    )

if __name__ == "__main__":
    main()
