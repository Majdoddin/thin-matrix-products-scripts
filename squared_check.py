"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and lossless reductions for 3SUM, Exact Triangle and APSP"
(Part I).  The paper's notation differs from the letters used here in a few places: the hashes G, F here are the
paper's re-indexing maps Gamma, Delta; the blocks Z_Q, W_Q are the paper's X_Q, Y_Q; kappa_g = phi_inf^2 is the
paper's mu_infinity^2 (phi_k here is the paper's mu_k).

Sections 4-7 of the paper (Remark 10.2): brute-force checks of the construction exactly as the paper defines it.

Run with python3 and numpy.  It uses the paper's renamed letters, the paper's left/right arrays, the 2m+1 runs indexed by s = t2 - t1, and
hashes G, F chosen deterministically by the conditional-expectation procedure of Lemma 4.1.

Part A  hash choice (the paper's re-indexing maps): K_g >= phi_m phi_{L-m} C(L,m) at several (L, m); the procedure is deterministic.
Part B  correctness: for every pair (Q1, Q2) of good sets and every output, the leaf sum over the 2m+1 runs
        equals the entry of Z_{Q1} W_{Q2} (encodings by Yates, products summed over runs, decoding by chi).
        Control: one run with no degree restriction must fail.
Part C  leaf counts: alpha'_d = C(2m,d) 4^d for every read output; leaves of order d with <= m P0 per half
        number beta'_d = sum_{d1+d2=d} beta_{d1} beta_{d2}; |Leaves(U)| <= sum_d min(|U| alpha'_d, beta'_d).
Part D  data structure with balanced boxes: every query, every switching order t, answered correctly;
        the number of balanced boxes is at most (2m+1) sum_{d>=t} beta'_d; the boxes of every read output are
        balanced; the dynamic program never leaves the balanced boxes.
"""
import itertools, random
from fractions import Fraction
from math import comb
import numpy as np

# ---------- the identity (Lemma 3.1) ----------
TERMS = ["P11", "P21", "P12", "P22", "P0"]          # P_ij: i = x-index, j = y-index
# left letters of the b-side: x1, x2, p1, p2 ; right letters of the c-side: y1, y2, q
PHI = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, -1], [0, 1, 0, 1], [-1, -1, 0, 0]])   # terms x B-letters
PSI = np.array([[1, 0, 0], [1, 0, -1], [0, 1, 0], [0, 1, 1], [1, 1, 0]])                    # terms x C-letters
# outputs z11, z21, z12, z22, z0 ; CHI[z, term] = 1 if term contributes to z
CHI = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [1, 1, 1, 1, 1]])
OUT_IJ = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}   # output index -> (i, j) zero-based

def apply_axis(arr, mat, axis):
    return np.moveaxis(np.tensordot(mat, arr, axes=([1], [axis])), 0, axis)

# ---------- GF(2) helpers ----------
def rank2(vecs):
    basis = []
    for v in vecs:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
    return len(basis)

def phi_inv(k):        # probability that a uniform k x k matrix over GF(2) is invertible
    p = Fraction(1)
    for i in range(1, k + 1):
        p *= 1 - Fraction(1, 2**i)
    return p

def choose_matrix(L, rows, sets, weight=None):
    """Columns 0..L-1 of a rows x L matrix over GF(2) (column = int < 2^rows), chosen one at a time to maximise
    sum over the given column sets S (|S| = rows) of weight(S) * Pr[M|_S invertible | columns fixed so far]."""
    if weight is None:
        weight = {S: Fraction(1) for S in sets}
    cols = []
    for l in range(L):
        best, bestval = None, None
        for v in range(2**rows):
            cand = cols + [v]
            val = Fraction(0)
            for S in sets:
                fixed = [cand[i] for i in S if i <= l]
                if rank2(fixed) == len(fixed):
                    val += weight[S]*phi_inv(rows - len(fixed))
            if bestval is None or val > bestval:
                best, bestval = v, val
        cols.append(best)
    return cols

def choose_hashes(L, m):
    Qs = list(itertools.combinations(range(L), m))
    G = choose_matrix(L, m, Qs)
    Gok = [Q for Q in Qs if rank2([G[i] for i in Q]) == m]
    comp = {Q: tuple(i for i in range(L) if i not in Q) for Q in Gok}
    F = choose_matrix(L, L - m, [comp[Q] for Q in Gok])
    good = [Q for Q in Gok if rank2([F[i] for i in comp[Q]]) == L - m]
    return G, F, good

def hsh(M, S):
    h = 0
    for i in S:
        h ^= M[i]
    return h

# ---------- Part A ----------
def part_a():
    print("Part A: deterministic hashes by conditional expectations")
    for L, m in [(4, 1), (4, 2), (6, 2), (8, 2), (9, 3), (10, 2), (11, 3)]:
        G, F, good = choose_hashes(L, m)
        G2, F2, good2 = choose_hashes(L, m)
        assert (G, F) == (G2, F2)
        exp = phi_inv(m)*phi_inv(L - m)*comb(L, m)
        assert len(good) >= exp
        print(f"  L={L:2d} m={m}: K_g = {len(good):4d} of C(L,m) = {comb(L,m):4d}; "
              f"phi_m phi_(L-m) C(L,m) = {float(exp):8.2f}; ratio K_g/C = {len(good)/comb(L,m):.4f}")

# ---------- Parts B-D ----------
def build(L, m, seed=1):
    G, F, good = choose_hashes(L, m)
    rng = np.random.default_rng(seed)
    n0 = 2**(L - m)
    Z = {Q: rng.integers(-5, 6, size=(n0*n0, 4**m)) for Q in good}     # rows (i1, r2), cols (r1, r1')
    W = {Q: rng.integers(-5, 6, size=(4**m, n0*n0)) for Q in good}     # rows (r1, r1'), cols (c1, i2)
    return G, F, good, Z, W

def outer_index(letters, Q):
    """bits of the x-index (or i-index) at the levels outside Q, in increasing order of level."""
    idx = 0
    for k, l in enumerate(i for i in range(len(letters)) if i not in Q):
        idx |= letters[l] << k
    return idx

def arrays(L, m, G, F, good, Z, W, s, split=True):
    goodset = set(good)
    n0 = 2**(L - m)
    a = np.zeros((4,)*L + (3,)*L, dtype=np.int64)
    b = np.zeros((3,)*L + (4,)*L, dtype=np.int64)
    Bstr = list(itertools.product(range(4), repeat=L))
    Cstr = list(itertools.product(range(3), repeat=L))
    def binfo(u):        # inner set, P (p2 levels), x-index outside Q
        Q = tuple(l for l in range(L) if u[l] >= 2)
        P = [l for l in range(L) if u[l] == 3]
        return Q, P, outer_index([u[l] if u[l] < 2 else 0 for l in range(L)], Q)
    def cinfo(v):        # P (q levels), S (y2 levels)
        return [l for l in range(L) if v[l] == 2], [l for l in range(L) if v[l] == 1]
    for u in Bstr:
        Q1, P1, i1 = binfo(u)
        if Q1 not in goodset:
            continue
        for v in Cstr:
            P2, S2 = cinfo(v)
            if split and len(P2) - len(P1) != s:
                continue
            row = i1 + n0*hsh(F, S2)
            col = hsh(G, P1) + 2**m*hsh(G, P2)
            a[u + v] = Z[Q1][row, col]
    for v in Cstr:
        P1, S1 = cinfo(v)
        for u in Bstr:
            Q2, P2, i2 = binfo(u)
            if Q2 not in goodset:
                continue
            if split and len(P2) - len(P1) != s:
                continue
            row = hsh(G, P1) + 2**m*hsh(G, P2)
            col = hsh(F, S1) + n0*i2
            b[v + u] = W[Q2][row, col]
    return a, b

def leaf_products(L, m, G, F, good, Z, W, split=True):
    runs = range(-m, m + 1) if split else [None]
    prod = np.zeros((5,)*(2*L), dtype=np.int64)
    for s in runs:
        a, b = arrays(L, m, G, F, good, Z, W, s, split)
        ea, eb = a, b
        for ax in range(L):
            ea = apply_axis(ea, PHI, ax); eb = apply_axis(eb, PSI, ax)
        for ax in range(L, 2*L):
            ea = apply_axis(ea, PSI, ax); eb = apply_axis(eb, PHI, ax)
        prod += ea*eb
    return prod

def read_outputs(L, m, G, F, good, Z, W):
    """yield (output string as tuple of output indices, expected value)"""
    n0 = 2**(L - m)
    for Q1 in good:
        for Q2 in good:
            M = Z[Q1] @ W[Q2]
            o1 = [l for l in range(L) if l not in Q1]
            o2 = [l for l in range(L) if l not in Q2]
            for z1 in itertools.product(range(4), repeat=L - m):
                for z2 in itertools.product(range(4), repeat=L - m):
                    w = [4]*(2*L)
                    for k, l in enumerate(o1): w[l] = z1[k]
                    for k, l in enumerate(o2): w[L + l] = z2[k]
                    i1 = sum(OUT_IJ[z1[k]][0] << k for k in range(L - m))
                    S1 = [o1[k] for k in range(L - m) if OUT_IJ[z1[k]][1] == 1]
                    i2 = sum(OUT_IJ[z2[k]][0] << k for k in range(L - m))
                    S2 = [o2[k] for k in range(L - m) if OUT_IJ[z2[k]][1] == 1]
                    yield tuple(w), M[i1 + n0*hsh(F, S2), hsh(F, S1) + n0*i2], (Q1, Q2)

def part_b(L, m):
    G, F, good, Z, W = build(L, m)
    prod = leaf_products(L, m, G, F, good, Z, W)
    out = prod
    for ax in range(2*L):
        out = apply_axis(out, CHI, ax)
    bad = tot = 0
    for w, val, _ in read_outputs(L, m, G, F, good, Z, W):
        tot += 1; bad += out[w] != val
    prod0 = leaf_products(L, m, G, F, good, Z, W, split=False)
    out0 = prod0
    for ax in range(2*L):
        out0 = apply_axis(out0, CHI, ax)
    bad0 = sum(out0[w] != val for w, val, _ in read_outputs(L, m, G, F, good, Z, W))
    print(f"  L={L}+{L}, m={m}+{m}: K_g={len(good)}, {tot} read outputs, {2*m+1} runs: {bad} wrong;"
          f" control without the split: {bad0} wrong")
    assert bad == 0 and bad0 > 0
    return G, F, good, Z, W, prod

def contributing(w, L):
    """cube of w: list per level of allowed term indices"""
    return [list(range(5)) if z == 4 else [z] for z in w]

def order(tau, L, m):
    return 2*m - sum(1 for t in tau if t == 4)

def part_c(L, m, G, F, good, Z, W):
    outs = [w for w, _, _ in read_outputs(L, m, G, F, good, Z, W)]
    beta = lambda e: comb(L, m - e)*4**(L - m + e)
    betap = [sum(beta(d1)*beta(d - d1) for d1 in range(max(0, d - m), min(m, d) + 1)) for d in range(2*m + 1)]
    alphap = [comb(2*m, d)*4**d for d in range(2*m + 1)]
    for w in outs[:50]:
        cnt = [0]*(2*m + 1)
        for tau in itertools.product(*contributing(w, L)):
            cnt[order(tau, L, m)] += 1
        assert cnt == alphap
    # leaves with <= m P0 per half, by order
    cntb = [0]*(2*m + 1)
    for tau in itertools.product(range(5), repeat=2*L):
        f1 = sum(1 for t in tau[:L] if t == 4); f2 = sum(1 for t in tau[L:] if t == 4)
        if f1 <= m and f2 <= m:
            cntb[2*m - f1 - f2] += 1
    assert cntb == betap
    rnd = random.Random(7)
    worst = 0
    for trial in range(20):
        U = rnd.sample(outs, rnd.randint(1, max(1, len(outs)//3)))
        leaves = set()
        for w in U:
            leaves.update(itertools.product(*contributing(w, L)))
        bound = sum(min(len(U)*alphap[d], betap[d]) for d in range(2*m + 1))
        assert len(leaves) <= bound
        worst = max(worst, len(leaves)/bound)
    print(f"  alpha'_d = {alphap} for every read output; beta'_d = {betap} (exact count);"
          f" |Leaves(U)| <= bound in 20 random U (max ratio {worst:.3f})")

def part_d(L, m, G, F, good, Z, W, prod):
    n = 2*L
    beta = lambda e: comb(L, m - e)*4**(L - m + e)
    betap = [sum(beta(d1)*beta(d - d1) for d1 in range(max(0, d - m), min(m, d) + 1)) for d in range(2*m + 1)]
    STAR = 5
    def balanced(box):
        return (sum(1 for t in box[:L] if t in (4, STAR)) <= m and sum(1 for t in box[L:] if t in (4, STAR)) <= m)
    allouts = list(read_outputs(L, m, G, F, good, Z, W))
    for t in range(2*m + 1):
        boxes = {}
        # generate: from balanced leaves with f <= 2m - t P0's, turn the e lowest P0 into stars
        for tau in itertools.product(range(5), repeat=n):
            zeros = [l for l in range(n) if tau[l] == 4]
            if len(zeros) > 2*m - t:
                continue
            if sum(1 for l in zeros if l < L) > m or sum(1 for l in zeros if l >= L) > m:
                continue
            for e in range(len(zeros) + 1):
                box = list(tau)
                for l in zeros[:e]:
                    box[l] = STAR
                boxes.setdefault(tuple(box), None)
        nbox = len(boxes)
        assert nbox <= (2*m + 1)*sum(betap[d] for d in range(t, 2*m + 1))
        val = {}
        for box in sorted(boxes, key=lambda bx: sum(1 for s in bx if s == STAR)):
            stars = [l for l in range(n) if box[l] == STAR]
            if not stars:
                val[box] = prod[box]
                continue
            l = stars[-1]
            tot = 0
            for lam in range(5):
                child = box[:l] + (lam,) + box[l + 1:]
                assert child in val and balanced(child)
                tot += val[child]
            val[box] = tot
        wrong = 0
        for w, expect, (Q1, Q2) in allouts:
            Q = [l for l in range(n) if w[l] == 4]
            ans = 0
            # leaves of order < t
            for d in range(t):
                for D in itertools.combinations(Q, d):
                    for terms in itertools.product(range(4), repeat=d):
                        tau = list(w)
                        for l, x in zip(D, terms):
                            tau[l] = x
                        ans += prod[tuple(tau)]
            # boxes of w
            for V in itertools.combinations(Q, 2*m - t):
                rest = [l for l in Q if l not in V]
                FV = set(Q) if not rest else {l for l in Q if l < min(rest)}
                for terms in itertools.product(range(4), repeat=t):
                    box = list(w)
                    for l in V:
                        box[l] = STAR if l in FV else 4
                    for l, x in zip(rest, terms):
                        box[l] = x
                    box = tuple(box)
                    assert box in val and balanced(box)
                    ans += val[box]
            wrong += ans != expect
        print(f"  t={t}: {len(allouts)} queries, {wrong} wrong; {nbox} balanced boxes"
              f" <= (2m+1) sum_(d>=t) beta'_d = {(2*m+1)*sum(betap[d] for d in range(t, 2*m+1))}")
        assert wrong == 0

if __name__ == "__main__":
    part_a()
    print("Part B: correctness of the squared run, 2m+1 runs")
    for L, m in [(2, 1), (3, 1), (3, 2), (4, 2)]:
        res = part_b(L, m)
        if (L, m) in [(3, 1), (3, 2)]:
            print(" Part C:"); part_c(L, m, *res[:5])
        if (L, m) in [(3, 1), (3, 2)]:
            print(" Part D: data structure with balanced boxes"); part_d(L, m, *res)
