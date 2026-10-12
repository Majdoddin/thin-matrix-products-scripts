"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and reductions that keep the whole
saving".  Supports Remark A.7 (the cases (2,2), (2,3), (3,3)) and Appendix A.1, paragraph "Border identities":
the border-rank bounds R(S(2,2,2)) >= 6, R(S(2,3,3)) >= 8, R(S(2,3,5)) >= 10 and R(S(2,2,3)) >= 7 (also
R(S(2,3,4)) >= 9), from exact ranks of Koszul flattenings [LO13] in rational arithmetic (no floating point, no modular arithmetic).

The Koszul flattening T_C^{wedge p}: Lambda^p C' (x) A* -> Lambda^{p+1} C' (x) B of the tensor, restricted on its
C factor to C' = image of an explicit integer matrix Pm (dim 2p+1; the identity if C has dimension 2p+1, else a
seeded random integer matrix), has rank at most (border rank) * C(2p, p).  Its rank is computed over Q by Gaussian
elimination on Python Fractions, so border rank >= ceil(rank / C(2p, p)).
Tensor convention: S(k,n,q') = sum x_i y_j z_ij + (sum_s p_s q_s + c y_1) z0 (A = x,p,c ; B = y,q ; C = z).
S(2,2,3) >= 7 is the branch of R(S(3,2,3)) >= 8 in Remark A.7 (the shape (2,3) with the shared variable among
the two left variables).
Controls: S(2,2,1) (the tensor of Lemma 3.1) and S(2,3,2), whose border ranks are 5 and 7, give exactly 5 and 7.
Usage: python3 koszul_exact.py      (standard library only; about a second)
"""
import itertools, random
from math import comb

def S_entries(k, n, q):
    A = [f'x{i}' for i in range(k)] + [f'p{s}' for s in range(q)] + ['c']
    B = [f'y{j}' for j in range(n)] + [f'q{s}' for s in range(q)]
    C = [f'z{i}{j}' for i in range(k) for j in range(n)] + ['z0']
    E = [(f'x{i}', f'y{j}', f'z{i}{j}') for i in range(k) for j in range(n)]
    E += [(f'p{s}', f'q{s}', 'z0') for s in range(q)] + [('c', 'y0', 'z0')]
    return A, B, C, E


def rank_frac(rows):
    from fractions import Fraction
    M = [[Fraction(v) for v in r] for r in rows]
    rank = 0; ncols = len(M[0])
    for col in range(ncols):
        piv = next((i for i in range(rank, len(M)) if M[i][col] != 0), None)
        if piv is None: continue
        M[rank], M[piv] = M[piv], M[rank]
        pv = M[rank][col]
        for i in range(len(M)):
            if i != rank and M[i][col] != 0:
                f = M[i][col] / pv
                M[i] = [a - f*b for a, b in zip(M[i], M[rank])]
        rank += 1
    return rank

def koszul_C(k, n, q, p, seed):
    A, B, C, E = S_entries(k, n, q)
    d = 2*p + 1
    rnd = random.Random(seed)
    Pm = [[1 if a == b else 0 for b in range(len(C))] for a in range(d)] if d == len(C) else \
         [[rnd.randint(-3, 3) for _ in C] for _ in range(d)]
    ci = {z: t for t, z in enumerate(C)}
    # T'(i, a, b) = sum_z Pm[i][z] T(a, b, z): wedge side i in C', domain A*, codomain B
    Sp = list(itertools.combinations(range(d), p)); Sp1 = list(itertools.combinations(range(d), p+1))
    i1 = {s: t for t, s in enumerate(Sp1)}; ai = {a: t for t, a in enumerate(A)}; bi = {b: t for t, b in enumerate(B)}
    rows = [[0]*(len(Sp)*len(A)) for _ in range(len(Sp1)*len(B))]
    for cs, S in enumerate(Sp):
        for i in range(d):
            if i in S: continue
            U = tuple(sorted(S + (i,))); sign = (-1)**sum(1 for s in S if s < i)
            for (a, b, z) in E:
                w = Pm[i][ci[z]]
                if w: rows[i1[U]*len(B) + bi[b]][cs*len(A) + ai[a]] += sign*w
    r = rank_frac(rows)
    return r, comb(2*p, p), -(-r // comb(2*p, p))

if __name__ == "__main__":
    for (k, n, q, p, seeds) in [(2, 2, 1, 2, [0]), (2, 2, 2, 2, [0]), (2, 3, 2, 2, [1, 2]), (2, 3, 3, 2, [1, 2, 3]),
                                (2, 3, 3, 3, [0]), (2, 3, 4, 2, [1, 2]), (2, 3, 5, 2, [1, 2]),
                                (2, 2, 3, 2, [0, 1])]:
        for sd in seeds:
            r, dv, bd = koszul_C(k, n, q, p, sd)
            print(f"S({k},{n},{q}), wedge on C, p={p}, seed {sd}: rank {r}, /{dv} -> border rank >= {bd}"
                  f"  (kn+1 = {k*n+1})", flush=True)
