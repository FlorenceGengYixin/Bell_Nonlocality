import argparse
import csv
from fractions import Fraction
from functools import reduce
from itertools import product
from math import gcd, lcm
from pathlib import Path
import cdd

def vertices(MA, MB):
    rows = []
    for a in product(range(2), repeat=MA):
        for b in product(range(2), repeat=MB):
            rows.append([1] + [int(v == 0) for v in a] + [int(v == 0) for v in b]
                        + [int(a[x] == b[y] == 0) for x in range(MA) for y in range(MB)])
    return rows

def primitive(row):
    values = [Fraction(str(v)) for v in row]
    scale = lcm(*(v.denominator for v in values))
    ints = [int(v * scale) for v in values]
    divisor = reduce(gcd, ints)
    return tuple(v // divisor for v in ints)

def generators(MA, MB):
    return ([('swap_a', x) for x in range(MA - 1)]
            + [('swap_b', y) for y in range(MB - 1)]
            + [('flip_a', x) for x in range(MA)]
            + [('flip_b', y) for y in range(MB)])

def transform(v, generator, MA, MB):
    kind, i = generator
    w = list(v)
    a = lambda x: 1 + x
    b = lambda y: 1 + MA + y
    c = lambda x, y: 1 + MA + MB + x * MB + y
    if kind == 'swap_a':
        w[a(i)], w[a(i + 1)] = v[a(i + 1)], v[a(i)]
        for y in range(MB):
            w[c(i, y)], w[c(i + 1, y)] = v[c(i + 1, y)], v[c(i, y)]
    elif kind == 'swap_b':
        w[b(i)], w[b(i + 1)] = v[b(i + 1)], v[b(i)]
        for x in range(MA):
            w[c(x, i)], w[c(x, i + 1)] = v[c(x, i + 1)], v[c(x, i)]
    elif kind == 'flip_a':
        w[0] += v[a(i)]
        w[a(i)] = -v[a(i)]
        for y in range(MB):
            w[b(y)] += v[c(i, y)]
            w[c(i, y)] = -v[c(i, y)]
    else:
        w[0] += v[b(i)]
        w[b(i)] = -v[b(i)]
        for x in range(MA):
            w[a(x)] += v[c(x, i)]
            w[c(x, i)] = -v[c(x, i)]
    return primitive(w)

def classify(facets, MA, MB):
    lookup = {v: i for i, v in enumerate(facets)}
    if len(lookup) != len(facets):
        raise ValueError('Duplicate inequalities in H representation')
    seen, classes = set(), []
    for start in range(len(facets)):
        if start in seen:
            continue
        stack, orbit = [start], []
        seen.add(start)
        while stack:
            i = stack.pop()
            orbit.append(i)
            for g in generators(MA, MB):
                target = transform(facets[i], g, MA, MB)
                if target not in lookup:
                    raise ValueError(f'Symmetry image missing: facet {i}, generator {g}, vector {target}')
                j = lookup[target]
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        classes.append(sorted(orbit))
    return classes

def save_rows(path, rows):
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('MA', type=int)
    parser.add_argument('MB', type=int)
    parser.add_argument('m', type=int)
    args = parser.parse_args()
    MA, MB, m = args.MA, args.MB, args.m
    if MA < 1 or MB < 1 or m != 2:
        parser.error('Require MA >= 1, MB >= 1, m = 2')
    v = vertices(MA, MB)
    matrix = cdd.matrix_from_array(v, rep_type=cdd.RepType.GENERATOR)
    poly = cdd.polyhedron_from_matrix(matrix)
    h = cdd.copy_inequalities(poly)
    if h.lin_set:
        raise ValueError('H representation contains linearity rows')
    facets = [primitive(row) for row in h.array]
    classes = classify(facets, MA, MB)
    prefix = Path(f'bell_{MA}{MB}{m}')
    save_rows(f'{prefix}_V.csv', v)
    save_rows(f'{prefix}_H.csv', facets)
    with open(f'{prefix}_classes.txt', 'w') as f:
        for number, orbit in enumerate(classes, 1):
            lines = [f'Class {number}: size = {len(orbit)}', f'Representative (facet {orbit[0]}): {facets[orbit[0]]}', f'Facet indices: {orbit}', '']
            print('\n'.join(lines[:2]))
            f.write('\n'.join(lines) + '\n')
    print(f"\nSaved V-representation to {prefix}_V.csv")
    print(f"Saved H-representation to {prefix}_H.csv")
    print(f"Saved symmetry classes to {prefix}_classes.txt")
    print(f'Vertices: {len(v)}, facets: {len(facets)}, classes: {len(classes)}')

if __name__ == '__main__':
 
    main()
