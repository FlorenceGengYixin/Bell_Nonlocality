import numpy as np
from scipy.optimize import differential_evolution

# Pauli matrices
X = np.array([
    [0.0, 1.0],
    [1.0, 0.0]
])

Z = np.array([
    [1.0,  0.0],
    [0.0, -1.0]
])

# Jordan parameterization
def observables(alpha, beta):
    """
    Jordan's lemma reduces each party's two binary
    observables to a 2-dimensional block.

    Alice:
        A0 = Z
        A1 = cos(alpha) Z + sin(alpha) X

    Bob:
        B0 = -(Z+X)/np.sqrt(2)
        B1 = cos(beta) Z + sin(beta) X
    """

    A0 = Z

    A1 = (
        np.cos(alpha) * Z
        + np.sin(alpha) * X
    )

    B0 = -(Z+X)/np.sqrt(2)

    B1 = (
        np.cos(beta) * Z
        + np.sin(beta) * X
    )

    return A0, A1, B0, B1

# Construct the 4 x 4 CHSH operator
def chsh_operator(alpha, beta):
    A0, A1, B0, B1 = observables(
        alpha,
        beta
    )
    S = (
        np.kron(A0, B0)
        + np.kron(A0, B1)
        + np.kron(A1, B0)
        - np.kron(A1, B1)
    )

    return S
# ============================================================
# Maximum eigenvalue
#
# For fixed alpha,beta:
#
# max_{|psi|=1} <psi|S|psi>
# =
# lambda_max(S)
# ============================================================

def maximum_eigenvalue(alpha, beta):

    S = chsh_operator(
        alpha,
        beta
    )

    eigenvalues = np.linalg.eigvalsh(S)

    return eigenvalues[-1]


# ============================================================
# Optimization
#
# We want the largest possible eigenvalue:
#
# max_{alpha,beta} lambda_max(S)
# ============================================================
def objective(params):
    alpha, beta = params
    return -maximum_eigenvalue(alpha, beta)

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()
    print("====================================")
    print("CHSH via Jordan's lemma")
    print("====================================")
    print()

    # Relative angles can be restricted to [0, pi].

    bounds = [
        (0.0, np.pi),
        (0.0, np.pi)
    ]

    result = differential_evolution(
        objective,
        bounds,
        tol=1e-12,
        polish=True,
        seed=1234
    )

    alpha = result.x[0]
    beta = result.x[1]

    S = chsh_operator(
        alpha,
        beta
    )

    eigenvalues, eigenvectors = np.linalg.eigh(S)

    lambda_max = eigenvalues[-1]

    psi_max = eigenvectors[:, -1]

    print("Optimal parameters:")
    print()

    print(
        f"alpha = {alpha:.12f}"
    )

    print(
        f"beta  = {beta:.12f}"
    )

    print()

    print(
        f"alpha/pi = {alpha/np.pi:.12f}"
    )

    print(
        f"beta/pi  = {beta/np.pi:.12f}"
    )

    print()

    print("Optimal 4x4 CHSH operator:")
    print()

    print(S)

    print()

    print("Eigenvalues:")
    print()

    print(eigenvalues)

    print()

    print(
        "Maximum eigenvalue = "
        f"{lambda_max:.12f}"
    )

    print(
        "Expected 2sqrt(2) = "
        f"{2*np.sqrt(2):.12f}"
    )

    print()

    print(
        "Absolute quantum value = "
        f"{abs(lambda_max):.12f}"
    )

    print()

    print("Eigenvector for maximum eigenvalue:")
    print()

    print(psi_max)