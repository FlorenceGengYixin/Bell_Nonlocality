import cvxpy as cp
import numpy as np

#1.Operator algebra
I = ()
A0 = ("A0",)
A1 = ("A1",)
B0 = ("B0",)
B1 = ("B1",)

def dagger(word):
    """(A0 B1)^dagger = B1^dagger A0^dagger"""
    return tuple(reversed(word))

def canonicalize(word):
    """Ai^2 = I, Bi^2 = I, Ai Bi = Bi Ai [commute]"""
    #but Ai and Aj do not commute, Bi and Bj do not commute
    #so we can just sort the letters in the word
    alice = [x for x in word if x[0] == "A"]
    bob = [x for x in word if x[0] == "B"]

    def reduce_party(ops):
        stack = []
        for op in ops:
            if stack and stack[-1] == op:
                stack.pop()  # Ai^2 = I
            else:
                stack.append(op)
        return stack
    
    alice = reduce_party(alice)
    bob = reduce_party(bob)
    
    return tuple(alice + bob)

def moment_word(left, right):
    """Gamma_ij = < S_i^dagger S_j>"""
    return canonicalize(dagger(left) + right)

#2. Word lists for NPA hierarchy(2,2,2)
LEVEL_1 = [I, A0, A1, B0, B1]
LEVEL_1_AB = [I, A0, A1, B0, B1, ("A0", "B0"), ("A0", "B1"), ("A1", "B0"), ("A1", "B1")]
LEVEL_2 = [I, A0, A1, B0, B1, ("A0", "A1"), ("B0", "B1"), ("A0", "B0"), ("A0", "B1"), ("A1", "B0"), ("A1", "B1")]

#3. for printing
def word_to_str(word):
    if not word:
        return "I"
    return "".join(word)

#4.build one NPA relaxation matrix
def solve_npa(words, name):
    n = len(words)
    
    print("\n" + "="*40)
    print(name)
    print("="*40)
    print("Words:")
    for i, w in enumerate(words):
        print(f"{i}: {word_to_str(w)}")

    #Moment matrix [for CHSH can use a real symmetric matrix]
    Gamma = cp.Variable((n, n))
    constraints = []
    #Positive Semidefinite constraint
    constraints.append(Gamma >> 0)
    representatives = {}
    for i in range(n):
        for j in range(i, n):
            word = moment_word(words[i], words[j])
            if word not in representatives:
                representatives[word] = (i, j)
            else:
                rep_i, rep_j = representatives[word]
                constraints.append(Gamma[i, j] == Gamma[rep_i, rep_j])

    constraints.append(Gamma[0, 0] == 1)  # <I> = 1
    #CHSH objective: <A0 B0> + <A0 B1> + <A1 B0> - <A1 B1>
    idx = {w: i for i, w in enumerate(words)}
    a0 = idx[A0]
    a1 = idx[A1]
    b0 = idx[B0]
    b1 = idx[B1]
    CHSH = Gamma[a0, b0] + Gamma[a0, b1] + Gamma[a1, b0] - Gamma[a1, b1]
    objective = cp.Maximize(CHSH)
    problem = cp.Problem(objective, constraints)
    value = problem.solve(solver=cp.SCS,
                          eps = 1e-7,
                          max_iters = 100000,
                          verbose = False)
    
    #Results
    print("\nStatus:", problem.status)
    print("\nCHSH upper bound:", value)
    print("\n2 sqrt(2) =", 2*np.sqrt(2))
    print("\nDifference:", value - 2 * np.sqrt(2))
    if Gamma.value is not None:

        G = (Gamma.value + Gamma.value.T) / 2

        print("\nMoment matrix Gamma:")
        print(np.round(G, 5))

        eigs = np.linalg.eigvalsh(G)

        print("\nEigenvalues of Gamma:")
        print(np.round(eigs, 8))

        print("\nMinimum eigenvalue of Gamma:")
        print(np.min(eigs))

    print("\nNumber of words:", n)
    print("Moment matrix size:", n, "x", n)
    print("Number of constraints:", len(constraints))

    return value, Gamma.value

#5.run the 3 levels of NPA hierarchy for CHSH
if __name__ == "__main__":

    q1, gamma1 = solve_npa(
        LEVEL_1,
        "NPA Level 1 (Q1)"
    )

    q1ab, gamma1ab = solve_npa(
        LEVEL_1_AB,
        "NPA Level 1 + AB (Q1+AB)"
    )

    q2, gamma2 = solve_npa(
        LEVEL_2,
        "NPA Level 2 (Q2)"
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Q1       : {q1:.10f}")
    print(f"Q1 + AB  : {q1ab:.10f}")
    print(f"Q2       : {q2:.10f}")
    print(f"2sqrt(2) : {2*np.sqrt(2):.10f}")   



    