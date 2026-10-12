"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and reductions that keep the whole
saving".  Supports Remark A.6 (a conditional improvement): "Koszul flattenings give [R(S(1,n,q')) >= n+q'+1] for
n = 2, q' <= 6 and for q' = 1, n <= 6" (mode base; the paper writes q_0' for q'), and re-checks independently the bounds of Remark A.7,
R(S(2,2,2)) >= 6, R(S(2,3,3)) >= 8, R(S(2,3,5)) >= 10, R(S(2,2,3)) >= 7 (mode excl), with controls of known border rank (mode ctrl).

S(k,n,q') = sum_{i,j} x_i y_j z_ij + (sum_{s<=q'} p_s q_s + c y_1) z0.

The flattening T^{wedge p}_W : Lambda^p W' (x) U* -> Lambda^{p+1} W' (x) V, with (U, V, W) a permutation of the
factors (A, B, C) and W' = image of W under a fixed integer projection of rank 2p+1, has rank at most
R_border(T) * C(2p, p).  The rank is computed modulo a large prime; rank mod p <= rank over Q = rank over C for an
integer matrix, so the bound printed is a rigorous lower bound (for the fixed projection used).
Usage: python3 koszul_modp.py base   (k = 1 tensors S(1,n,q'), n, q' = 1..6)
       python3 koszul_modp.py excl   (S(2,2,2), S(2,3,3), S(2,3,5), S(2,2,3), independent re-check)
       python3 koszul_modp.py ctrl   (controls of known border rank)
(needs numpy)
"""
import itertools, random, sys
from math import comb
import numpy as np

PR = 2147483629  # prime < 2^31

def S_triples(k, n, q):
    A = [('x', i) for i in range(k)] + [('p', s) for s in range(q)] + [('c',)]
    B = [('y', j) for j in range(n)] + [('q', s) for s in range(q)]
    C = [('z', i, j) for i in range(k) for j in range(n)] + [('z0',)]
    E = [(('x', i), ('y', j), ('z', i, j)) for i in range(k) for j in range(n)]
    E += [(('p', s), ('q', s), ('z0',)) for s in range(q)] + [(('c',), ('y', 0), ('z0',))]
    return [A, B, C], E

def rank_mod(M):
    M = M.copy() % PR
    r = 0; rows, cols = M.shape
    for col in range(cols):
        if r == rows: break
        nz = np.nonzero(M[r:, col])[0]
        if len(nz) == 0: continue
        piv = r + nz[0]
        if piv != r: M[[r, piv]] = M[[piv, r]]
        inv = pow(int(M[r, col]), PR - 2, PR)
        M[r] = (M[r] * inv) % PR
        others = np.nonzero(M[:, col])[0]
        others = others[others != r]
        if len(others):
            f = M[others, col].reshape(-1, 1)
            M[others] = (M[others] - f * M[r]) % PR
        r += 1
    return r

def koszul(spaces, E, wedge, p, seed):
    """wedge = index of the factor W; both orders (U, V) of the other two factors are tried."""
    others = [f for f in range(3) if f != wedge]
    best = (0, None)
    for (U, V) in (tuple(others), tuple(reversed(others))):
        W = spaces[wedge]; d = 2*p + 1
        if d > len(W): return best
        rnd = random.Random(seed)
        if d == len(W):
            Pm = [[1 if a == b else 0 for b in range(len(W))] for a in range(d)]
        else:
            Pm = [[rnd.randint(-5, 5) for _ in W] for _ in range(d)]
        wi = {w: t for t, w in enumerate(W)}
        ui = {u: t for t, u in enumerate(spaces[U])}; vi = {v: t for t, v in enumerate(spaces[V])}
        Sp = list(itertools.combinations(range(d), p)); Sp1 = list(itertools.combinations(range(d), p + 1))
        i1 = {s: t for t, s in enumerate(Sp1)}
        nU, nV = len(spaces[U]), len(spaces[V])
        M = np.zeros((len(Sp1)*nV, len(Sp)*nU), dtype=np.int64)
        for cs, S in enumerate(Sp):
            for i in range(d):
                if i in S: continue
                Ut = tuple(sorted(S + (i,))); sign = (-1)**sum(1 for s in S if s < i)
                for e in E:
                    w = Pm[i][wi[e[wedge]]]
                    if w: M[i1[Ut]*nV + vi[e[V]], cs*nU + ui[e[U]]] += sign*w
        r = rank_mod(M)
        bd = -(-r // comb(2*p, p))
        if bd > best[0]: best = (bd, f'wedge {"ABC"[wedge]} p={p} {"ABC"[U]}*->{"ABC"[V]} rank {r}/{comb(2*p,p)}')
    return best

def best_bound(k, n, q, pmax=4, seeds=(1, 2)):
    spaces, E = S_triples(k, n, q)
    best = (0, None)
    for wedge in range(3):
        for p in range(1, pmax + 1):
            if 2*p + 1 > len(spaces[wedge]): break
            for sd in seeds:
                b = koszul(spaces, E, wedge, p, sd)
                if b[0] > best[0]: best = b
                if 2*p + 1 == len(spaces[wedge]): break
    return best

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else 'base'
    if mode == 'base':
        print("Koszul bounds for S(1,n,q'): conciseness gives n+q'; is it n+q'+1?")
        for n in range(1, 7):
            for q in range(1, 7):
                b = best_bound(1, n, q)
                print(f"  S(1,{n},{q}): >= {b[0]}  (n+q'={n+q}, n+q'+1={n+q+1})  [{b[1]}]", flush=True)
    if mode == 'excl':
        # the exclusions of Remark A.7, recomputed independently
        for c in [(2, 2, 2), (2, 3, 3), (2, 3, 5), (2, 2, 3)]:
            print(f"  S{c}: >= {best_bound(*c)}", flush=True)
    if mode == 'ctrl':
        # controls: known border ranks must not be exceeded
        for (k, n, q, known) in [(2, 2, 1, 5), (2, 3, 2, 7), (1, 2, 0, 3), (1, 3, 0, 4)]:
            b = best_bound(k, n, q)
            print(f"  control S({k},{n},{q}): >= {b[0]}, known border rank {known}  [{b[1]}]", flush=True)
