import sys
import itertools
import cdd

def bell_dimension(MA, MB, m): #= dim NS
    q = m - 1
    return MA*q + MB*q + MA*MB*q*q

def local_vertices(MA, MB, m):
    V = []
    for A in itertools.product(range(m), repeat=MA):
        for B in itertools.product(range(m), repeat=MB):
            row = [1]
            # Alice marginals
            row += [int(A[x] == a) for x in range(MA) for a in range(m-1)]
            # Bob marginals
            row += [int(B[y] == b) for y in range(MB) for b in range(m-1)]
            # Joint probabilities
            row += [int(A[x] == a and B[y] == b)
                    for x in range(MA) for y in range(MB)
                    for a in range(m-1) for b in range(m-1)]
            V.append(row)
    return V

def inequalities(V):
    mat = cdd.matrix_from_array(V, rep_type=cdd.RepType.GENERATOR)
    poly = cdd.polyhedron_from_matrix(mat)
    H = cdd.copy_inequalities(poly)
    return H.array

def save_matrix(filename, rows):
    with open(filename, "w") as f:
        for row in rows:
            f.write(" ".join(map(str, row)) + "\n")

def main():
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} MA MB m")
        sys.exit(1)
    MA, MB, m = map(int, sys.argv[1:])
    dim = bell_dimension(MA, MB, m)
    V = local_vertices(MA, MB, m)
    print(f"Bell scenario ({MA},{MB},{m})")
    print(f"Bell dimension = {dim}")
    print(f"Deterministic vertices = {len(V)}")
    print("\nGenerating H-representation...")
    H = inequalities(V)
    print(f"Number of inequalities = {len(H)}")
    vfile = f"bell_{MA}{MB}{m}_V.txt"
    hfile = f"bell_{MA}{MB}{m}_H.txt"
    save_matrix(vfile, V)
    save_matrix(hfile, H)
    print(f"\nInequalities:")
    for i, row in enumerate(H, 1):
        print(f"I{i}: {' '.join(map(str, row))} >= 0")
    print(f"\nSaved V-representation to {vfile}")
    print(f"Saved H-representation to {hfile}")

if __name__ == "__main__":
    main()