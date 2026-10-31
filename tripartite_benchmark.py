import numpy as np

# Tripartite hidden-influence benchmark
# Scenario:
# Alice:   x = 0,1       a = 0,1
# Bob:     y = 0,1,2     b = 0,1,2
# Charlie: z = 0,1       c = 0,1
# Hilbert space:
# C^2 tensor C^3 tensor C^2
# Tensor/basis order:
# |a,b,c>

# 1. Basic helper

def projector(v):
    """
    Return normalized rank-1 projector

        |v><v| / <v|v>
    """

    v = np.asarray(v, dtype=complex)
    v = v / np.linalg.norm(v)

    return np.outer(v, v.conj())

# 2. Quantum state -- Appendix A, Eq. (A1)

def make_state():
    # tensor structure:
    # psi[a,b,c]
    # a = Alice   0,1
    # b = Bob     0,1,2
    # c = Charlie 0,1

    psi = np.zeros((2, 3, 2), dtype=complex)

    psi[0, 0, 1] = -0.0003 - 0.0075j
    psi[0, 1, 0] =  0.0029 + 0.0093j
    psi[0, 2, 1] =  0.6769

    psi[1, 0, 0] = -0.1145 - 0.2881j
    psi[1, 1, 1] = -0.5782 - 0.3330j
    psi[1, 2, 0] = -0.0154 + 0.0055j

    # Convert 2 x 3 x 2 tensor into 12-dimensional vector
    psi = psi.reshape(-1)

    # Paper uses proportionality sign, so normalize
    psi = psi / np.linalg.norm(psi)

    return psi

# 3. Measurements -- Appendix A, Eq. (A2)
# Convention:
# A[x][a]
# B[y][b]
# C[z][c]

def make_measurements():
    I2 = np.eye(2, dtype=complex)
    I3 = np.eye(3, dtype=complex)

    # Alice

    A = [
        [None, None],
        [None, None]
    ]

    # x = 0
    # A^0_0 = 1/2 (I + sigma_z)
    #       = |0><0|

    A[0][0] = np.array([
        [1, 0],
        [0, 0]
    ], dtype=complex)

    # Remaining outcome from completeness
    A[0][1] = I2 - A[0][0]

    # x = 1
    # alpha^1_0 proportional to
    # [0.5289 - 0.4693 i,
    #  0.7071]

    A[1][0] = projector([
        0.5289 - 0.4693j,
        0.7071
    ])

    A[1][1] = I2 - A[1][0]

    # Bob

    B = [
        [None, None, None],
        [None, None, None],
        [None, None, None]
    ]

    # y = 0

    B[0][0] = projector([
        0.0368 - 0.2164j,
        0.7070,
        0.6584 + 0.1357j
    ])

    B[0][1] = projector([
        -0.0368 + 0.2164j,
        0.7072,
        -0.6583 - 0.1357j
    ])

    B[0][2] = (
        I3
        - B[0][0]
        - B[0][1]
    )

    # y = 1

    B[1][0] = projector([
        0.1466 - 0.0131j,
        0.9891,
        -0.0001 - 0.0027j
    ])

    B[1][1] = projector([
        0.9889,
        -0.1467 - 0.0131j,
        0.0006 - 0.0181j
    ])

    B[1][2] = (
        I3
        - B[1][0]
        - B[1][1]
    )

    # y = 2

    B[2][0] = projector([
        0.0002 + 0.0053j,
        0.6925 - 0.1428j,
        0.7072
    ])

    B[2][1] = projector([
        1.0000,
        0.0,
        -0.0003 + 0.0075j
    ])

    B[2][2] = (
        I3
        - B[2][0]
        - B[2][1]
    )

    # Charlie

    C = [
        [None, None],
        [None, None]
    ]

    # z = 0

    C[0][0] = projector([
        0.4800 - 0.8751j,
        0.0619
    ])

    C[0][1] = I2 - C[0][0]

    # z = 1

    C[1][0] = projector([
        0.0298 - 0.0543j,
        0.9981
    ])

    C[1][1] = I2 - C[1][0]

    return A, B, C

# 4. Born rule

def probability(
    psi,
    A, B, C,
    a, b, c,
    x, y, z
):
    """
    P(abc|xyz)

      = <psi|
          A[x][a] tensor
          B[y][b] tensor
          C[z][c]
        |psi>
    """

    measurement = np.kron(
        np.kron(
            A[x][a],
            B[y][b]
        ),
        C[z][c]
    )

    p = np.vdot(
        psi,
        measurement @ psi
    )

    # Probability should be real.
    # Tiny imaginary components can occur numerically.
    return float(np.real(p))

# 5. Generate complete probability table

def make_probability_table(
    psi,
    A, B, C
):
    # P[a,b,c,x,y,z]

    P = np.zeros(
        (2, 3, 2, 2, 3, 2),
        dtype=float
    )

    for a in range(2):
        for b in range(3):
            for c in range(2):
                for x in range(2):
                    for y in range(3):
                        for z in range(2):
                            P[
                                a, b, c,
                                x, y, z
                            ] = probability(
                                psi,
                                A, B, C,
                                a, b, c,
                                x, y, z
                            )

    return P

# 6. Marginal probabilities

def PA(P, a, x):
    # P_A(a|x)
    # fix y = 0, z = 0
    # because quantum distribution is non-signaling

    return np.sum(
        P[a, :, :, x, 0, 0]
    )

def PB(P, b, y):
    # P_B(b|y)

    return np.sum(
        P[:, b, :, 0, y, 0]
    )

def PC(P, c, z):
    # P_C(c|z)

    return np.sum(
        P[:, :, c, 0, 0, z]
    )

def PAB(P, a, b, x, y):
    # P_AB(ab|xy)
    # sum over Charlie
    # fix z = 0

    return np.sum(
        P[a, b, :, x, y, 0]
    )

def PBC(P, b, c, y, z):
    # P_BC(bc|yz)
    # sum over Alice
    # fix x = 0

    return np.sum(
        P[:, b, c, 0, y, z]
    )

def PAC(P, a, c, x, z):
    # Not needed in inequality,
    # but useful for checks.

    return np.sum(
        P[a, :, c, x, 0, z]
    )

# 7. Hidden-influence inequality -- Eq. (4)

def hidden_influence_S(
    P,
    verbose=True
):
    """
    Evaluate Eq. (4) of the paper.

    Hidden-influence bound:

        S >= -2
    """

    terms = [
        # single-party
        ("- PA(0|0)",
         -PA(P, 0, 0)),
        ("- PA(0|1)",
         -PA(P, 0, 1)),
        ("- PB(0|0)",
         -PB(P, 0, 0)),
        ("+ PB(1|0)",
         +PB(P, 1, 0)),
        ("- PB(0|1)",
         -PB(P, 0, 1)),
        ("- PB(1|1)",
         -PB(P, 1, 1)),
        # AB
        ("+ PAB(00|00)",
         +PAB(P, 0, 0, 0, 0)),
        ("+ PAB(00|10)",
         +PAB(P, 0, 0, 1, 0)),
        ("+ PAB(01|00)",
         +PAB(P, 0, 1, 0, 0)),
        ("- PAB(01|10)",
         -PAB(P, 0, 1, 1, 0)),
        ("+ 2 PAB(00|01)",
         +2 * PAB(P, 0, 0, 0, 1)),
        ("+ 2 PAB(01|01)",
         +2 * PAB(P, 0, 1, 0, 1)),
        ("+ 2 PAB(00|12)",
         +2 * PAB(P, 0, 0, 1, 2)),
        ("+ PAB(01|12)",
         +PAB(P, 0, 1, 1, 2)),
        # Charlie
        ("+ PC(0|0)",
         +PC(P, 0, 0)),
        # BC
        ("+ PBC(00|00)",
         +PBC(P, 0, 0, 0, 0)),
        ("- PBC(00|10)",
         -PBC(P, 0, 0, 1, 0)),
        ("- PBC(10|10)",
         -PBC(P, 1, 0, 1, 0)),
        ("- PBC(00|20)",
         -PBC(P, 0, 0, 2, 0)),
        ("- PBC(10|20)",
         -PBC(P, 1, 0, 2, 0)),
        ("- PBC(10|01)",
         -PBC(P, 1, 0, 0, 1)),
        ("+ PBC(00|11)",
         +PBC(P, 0, 0, 1, 1)),
        ("+ PBC(10|11)",
         +PBC(P, 1, 0, 1, 1)),
        ("- PBC(00|21)",
         -PBC(P, 0, 0, 2, 1)),
    ]

    if verbose:
        print("\n" + "=" * 60)

        print("Eq. (4) term-by-term")

        print("=" * 60)

        for name, value in terms:
            print(
                f"{name:24s}"
                f"{value:+.12f}"
            )

    S = sum(
        value
        for name, value in terms
    )

    return S

# 8. Sanity checks

def normalization_error(P):
    max_error = 0.0

    for x in range(2):
        for y in range(3):
            for z in range(2):
                total = np.sum(
                    P[:, :, :, x, y, z]
                )

                error = abs(
                    total - 1
                )

                max_error = max(
                    max_error,
                    error
                )

    return max_error

def non_signaling_errors(P):
    """
    Check:

        AB independent of z
        AC independent of y
        BC independent of x
    """

    err_AB = 0.0
    err_AC = 0.0
    err_BC = 0.0

    # AB independent of z

    for a in range(2):
        for b in range(3):
            for x in range(2):
                for y in range(3):
                    p_z0 = np.sum(
                        P[a, b, :, x, y, 0]
                    )

                    p_z1 = np.sum(
                        P[a, b, :, x, y, 1]
                    )

                    err_AB = max(
                        err_AB,
                        abs(p_z0 - p_z1)
                    )

    # AC independent of y

    for a in range(2):
        for c in range(2):
            for x in range(2):
                for z in range(2):
                    values = []

                    for y in range(3):
                        values.append(
                            np.sum(
                                P[
                                    a, :, c,
                                    x, y, z
                                ]
                            )
                        )

                    error = (
                        max(values)
                        -
                        min(values)
                    )

                    err_AC = max(
                        err_AC,
                        error
                    )

    # BC independent of x

    for b in range(3):
        for c in range(2):
            for y in range(3):
                for z in range(2):
                    p_x0 = np.sum(
                        P[:, b, c, 0, y, z]
                    )

                    p_x1 = np.sum(
                        P[:, b, c, 1, y, z]
                    )

                    err_BC = max(
                        err_BC,
                        abs(p_x0 - p_x1)
                    )

    return (
        err_AB,
        err_AC,
        err_BC
    )

# 9. Check POVMs

def check_measurements(A, B, C):
    max_hermiticity_error = 0.0

    max_completeness_error = 0.0

    minimum_eigenvalue = np.inf

    systems = [
        (A, 2, "Alice"),
        (B, 3, "Bob"),
        (C, 2, "Charlie")
    ]

    for measurements, dim, name in systems:
        identity = np.eye(
            dim,
            dtype=complex
        )

        for setting_index, setting in enumerate(measurements):
            total = np.zeros(
                (dim, dim),
                dtype=complex
            )

            for E in setting:
                # Hermiticity
                error = np.max(
                    np.abs(
                        E - E.conj().T
                    )
                )

                max_hermiticity_error = max(
                    max_hermiticity_error,
                    error
                )

                # Positivity
                E_herm = (
                    E + E.conj().T
                ) / 2

                eigs = np.linalg.eigvalsh(
                    E_herm
                )

                minimum_eigenvalue = min(
                    minimum_eigenvalue,
                    np.min(eigs)
                )

                total += E

            # Completeness:
            # sum_a A[a|x] = I

            error = np.max(
                np.abs(
                    total - identity
                )
            )

            max_completeness_error = max(
                max_completeness_error,
                error
            )

    return (
        max_hermiticity_error,
        max_completeness_error,
        minimum_eigenvalue
    )

# 10. Main

def main():
    print("=" * 72)

    print(
        "Tripartite hidden-influence benchmark"
    )

    print(
        "Scenario: [22][333][22]"
    )

    print("=" * 72)

    # Construct state

    psi = make_state()

    print(
        "\nState dimension:",
        len(psi)
    )

    print(
        "State norm:",
        np.linalg.norm(psi)
    )

    # Construct measurements

    A, B, C = make_measurements()

    (
        herm_error,
        completeness_error,
        min_povm_eig
    ) = check_measurements(
        A, B, C
    )

    print(
        "\nMax POVM Hermiticity error:",
        herm_error
    )

    print(
        "Max POVM completeness error:",
        completeness_error
    )

    print(
        "Minimum POVM eigenvalue:",
        min_povm_eig
    )

    # Born probabilities

    P = make_probability_table(
        psi,
        A, B, C
    )

    print(
        "\nProbability table shape:",
        P.shape
    )

    print(
        "Minimum probability:",
        np.min(P)
    )

    print(
        "Maximum probability:",
        np.max(P)
    )

    # Normalization

    norm_error = normalization_error(P)

    print(
        "\nMaximum normalization error:",
        norm_error
    )

    # Non-signaling

    (
        ns_AB,
        ns_AC,
        ns_BC
    ) = non_signaling_errors(P)

    print(
        "\nMaximum NS error AB vs z:",
        ns_AB
    )

    print(
        "Maximum NS error AC vs y:",
        ns_AC
    )

    print(
        "Maximum NS error BC vs x:",
        ns_BC
    )

    # Evaluate Eq. (4)

    S = hidden_influence_S(
        P,
        verbose=True
    )

    HI_BOUND = -2.0

    gap = HI_BOUND - S

    # Result

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
        "\nHidden-influence inequality:"
    )

    print(
        "S >= -2"
    )

    print(
        "\nComputed quantum value:"
    )

    print(
        f"S = {S:.12f}"
    )

    print(
        "\nViolation gap:"
    )

    print(
        f"Delta = -2 - S = {gap:.12e}"
    )

    if S < HI_BOUND:
        print(
            "\n>>> QUANTUM VIOLATION FOUND <<<"
        )

    else:
        print(
            "\nNo violation found with "
            "the rounded Appendix-A data."
        )

    print(
        "\nPaper reports approximately:"
    )

    print(
        "S = -2.00015"
    )

    print(
        "\nImportant:"
    )

    print(
        "Appendix A coefficients are "
        "given only to limited numerical precision."
    )

# Run

if __name__ == "__main__":
    main()
