"""Library for the Q2 computations: exact lattice tests (Smith normal form over
Python ints, written by hand) and the vectorised OV-gadget search.

Relations P ⊆ Σ^3 with Σ = {0..k-1} are given as a list of triples.
e(a,b,c) = e_{1,a}+e_{2,b}+e_{3,c} in Z^{3k};  L = Z-span{e(p)-e(p0)}.

Every verdict is certified independently of the correctness of the SNF code:
  * membership x in L (resp. Q-span, L + pZ^n) is certified by an explicit
    coefficient vector c (integer / rational / mod p) with c M = x;
  * non-membership is certified by the linear map x -> xV (V an integer
    matrix), after checking that it sends every generator into rowspan(D)
    (resp. its Q- or mod-p version) while x is not sent there.
"""
import itertools
from fractions import Fraction
import numpy as np


# ---------------------------------------------------------------- SNF ----
def snf(M, ncols):
    """Smith normal form.  M: list of m rows (lists of ints) of length ncols.
    Returns (U, D, V, d) with U M V = D (U m x m, V n x n, unimodular),
    d = nonzero diagonal entries d_0 | d_1 | ... (positive)."""
    m, n = len(M), ncols
    A = [row[:] for row in M]
    U = [[int(i == j) for j in range(m)] for i in range(m)]
    V = [[int(i == j) for j in range(n)] for i in range(n)]

    def swap_rows(i, j):
        A[i], A[j] = A[j], A[i]
        U[i], U[j] = U[j], U[i]

    def swap_cols(i, j):
        for R in A:
            R[i], R[j] = R[j], R[i]
        for R in V:
            R[i], R[j] = R[j], R[i]

    def add_row(dst, src, q):  # row dst += q*row src
        if q:
            Ad, As = A[dst], A[src]
            for c in range(n):
                Ad[c] += q * As[c]
            Ud, Us = U[dst], U[src]
            for c in range(m):
                Ud[c] += q * Us[c]

    def add_col(dst, src, q):  # col dst += q*col src
        if q:
            for R in A:
                R[dst] += q * R[src]
            for R in V:
                R[dst] += q * R[src]

    t = 0
    d = []
    while t < min(m, n):
        # pivot: smallest nonzero |entry| in the submatrix
        best = None
        for i in range(t, m):
            for j in range(t, n):
                v = A[i][j]
                if v and (best is None or abs(v) < best[0]):
                    best = (abs(v), i, j)
        if best is None:
            break
        _, i, j = best
        swap_rows(t, i)
        swap_cols(t, j)
        while True:
            p = A[t][t]
            dirty = False
            for i in range(t + 1, m):
                if A[i][t]:
                    add_row(i, t, -(A[i][t] // p))
                    if A[i][t]:
                        dirty = True
            for j in range(t + 1, n):
                if A[t][j]:
                    add_col(j, t, -(A[t][j] // p))
                    if A[t][j]:
                        dirty = True
            if dirty:
                # move smallest nonzero of row t / column t into pivot
                best = (abs(A[t][t]), t, t)
                for i in range(t + 1, m):
                    if A[i][t] and abs(A[i][t]) < best[0]:
                        best = (abs(A[i][t]), i, t)
                for j in range(t + 1, n):
                    if A[t][j] and abs(A[t][j]) < best[0]:
                        best = (abs(A[t][j]), t, j)
                _, i, j = best
                swap_rows(t, i)
                swap_cols(t, j)
                continue
            # divisibility of the rest by the pivot
            bad = None
            for i in range(t + 1, m):
                for j in range(t + 1, n):
                    if A[i][j] % p:
                        bad = i
                        break
                if bad is not None:
                    break
            if bad is None:
                break
            add_row(t, bad, 1)
        if A[t][t] < 0:
            A[t] = [-x for x in A[t]]
            U[t] = [-x for x in U[t]]
        d.append(A[t][t])
        t += 1
    return U, A, V, d


def matmul(X, Y):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*Y)] for row in X]


def det(M):
    """exact determinant (Fractions)."""
    A = [[Fraction(x) for x in r] for r in M]
    n = len(A)
    s = Fraction(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if A[r][c] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            s = -s
        s *= A[c][c]
        for r in range(c + 1, n):
            f = A[r][c] / A[c][c]
            if f:
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return s


# ---------------------------------------------------- representability ----
def primes_upto(N):
    return [p for p in range(2, N + 1) if all(p % q for q in range(2, p))]


def rep_info(P, k, primes=(2, 3, 5), certify=True, check_unimodular=False):
    """Representability of P ⊆ [k]^3.
    Returns dict: REP, REPZ, REP_p (dict p->bool), torsion (invariant factors > 1
    of A_P = Z^{3k}/L), free (free rank of A_P), closure sizes."""
    P = sorted(set(P))
    n = 3 * k
    allpts = list(itertools.product(range(k), repeat=3))
    Pset = set(P)
    out = dict()
    if not P:
        # empty relation: phi = 0, t != 0 works over Z and every F_p
        out.update(REP=True, REPZ=True, REP_p={p: True for p in primes},
                   torsion=[], free=n, rank=0)
        return out

    def e(s):
        v = [0] * n
        v[s[0]] += 1
        v[k + s[1]] += 1
        v[2 * k + s[2]] += 1
        return v

    p0 = P[0]
    e0 = e(p0)
    M = [[a - b for a, b in zip(e(p), e0)] for p in P[1:]]
    if M:
        U, D, V, d = snf(M, n)
    else:
        U, D, V, d = [], [], [[int(i == j) for j in range(n)] for i in range(n)], []
    r = len(d)
    if certify and M:
        assert matmul(matmul(U, M), V) == D
        if check_unimodular:
            assert abs(det(U)) == 1 and abs(det(V)) == 1
        MV = matmul(M, V)
        # the map x -> xV sends L into rowspan(D): columns >= r vanish,
        # column i < r divisible by d_i
        for row in MV:
            assert all(row[i] == 0 for i in range(r, n))
            assert all(row[i] % d[i] == 0 for i in range(r))
    allp = sorted(set(primes) | {q for x in d for q in primes_upto(x) if x % q == 0})
    REP, REPZ, REPp = True, True, {p: True for p in allp}
    witness = dict()
    for s in allpts:
        if s in Pset:
            continue
        x = [a - b for a, b in zip(e(s), e0)]
        y = [sum(x[i] * V[i][j] for i in range(n)) for j in range(n)]
        freezero = all(y[j] == 0 for j in range(r, n))
        inL = freezero and all(y[i] % d[i] == 0 for i in range(r))
        inQ = freezero
        if inL:
            REP = False
            witness.setdefault('L', s)
            if certify:
                z = [y[i] // d[i] for i in range(r)] + [0] * (len(M) - r)
                c = [sum(z[i] * U[i][j] for i in range(len(M))) for j in range(len(M))]
                assert [sum(c[i] * M[i][j] for i in range(len(M))) for j in range(n)] == x
        if inQ:
            REPZ = False
            witness.setdefault('Q', s)
            if certify:
                z = [Fraction(y[i], d[i]) for i in range(r)] + [0] * (len(M) - r)
                c = [sum(z[i] * U[i][j] for i in range(len(M))) for j in range(len(M))]
                assert [sum(c[i] * M[i][j] for i in range(len(M))) for j in range(n)] == x
        for p in allp:
            inp = all(y[j] % p == 0 for j in range(r, n)) and \
                all(y[i] % np.gcd(d[i], p) == 0 for i in range(r))
            if inp:
                REPp[p] = False
                if certify:
                    z = []
                    for i in range(r):
                        z.append(0 if d[i] % p == 0 else (y[i] * pow(d[i], -1, p)) % p)
                    z += [0] * (len(M) - r)
                    c = [sum(z[i] * U[i][j] for i in range(len(M))) % p for j in range(len(M))]
                    assert all((sum(c[i] * M[i][j] for i in range(len(M))) - x[j]) % p == 0
                               for j in range(n))
    out.update(REP=REP, REPZ=REPZ, REP_p=REPp, torsion=[x for x in d if x > 1],
               free=n - r, rank=r, witness=witness)
    return out


# -------------------------------------------------------- OV gadgets ----
SIDE_ASSIGNMENTS = [s for s in itertools.product('ABC', repeat=3) if 'A' in s and 'B' in s]


def gadget_spec(sides):
    """bit tuples (over the non-const positions, in position order) and the
    target set: A-bits all equal (=a), B-bits all equal (=b), not (a and b)."""
    active = [i for i in range(3) if sides[i] != 'C']
    tuples = list(itertools.product((0, 1), repeat=len(active)))
    target = 0
    for ti, bits in enumerate(tuples):
        A = {bits[q] for q, i in enumerate(active) if sides[i] == 'A'}
        B = {bits[q] for q, i in enumerate(active) if sides[i] == 'B'}
        if len(A) == 1 and len(B) == 1:
            a, = A
            b, = B
            if not (a and b):
                target |= 1 << ti
    return active, tuples, target


def gadget_search(tables, k):
    """tables: bool array (N, k^3), index a*k*k + b*k + c.
    For every relation and every side assignment, computes the set of masks
    achievable by one gadget coordinate (t = 1), and from it
      t1   : a t=1 gadget exists,
      t2   : a gadget with t <= 2 exists,
      tany : a gadget with some t exists (AND-closure of achievable masks
             that contain the target equals the target),
      tmin : minimal t (0 if none).
    Returns dict side-assignment -> dict of arrays, and summary arrays."""
    N = tables.shape[0]
    res = {}
    letters = range(k)
    for sides in SIDE_ASSIGNMENTS:
        active, tuples, target = gadget_spec(sides)
        nt = len(tuples)
        full = (1 << nt) - 1
        achieved = np.zeros((N, 1 << nt), dtype=bool)
        # choices per position: active -> pair (f(0), f(1)); const -> letter
        choices = [list(itertools.product(letters, repeat=2)) if sides[i] != 'C'
                   else [(v, v) for v in letters] for i in range(3)]
        rows = np.arange(N)
        for f in itertools.product(*choices):
            mask = np.zeros(N, dtype=np.int64)
            for ti, bits in enumerate(tuples):
                bit_of = {}
                for q, i in enumerate(active):
                    bit_of[i] = bits[q]
                cell = [f[i][bit_of.get(i, 0)] for i in range(3)]
                idx = cell[0] * k * k + cell[1] * k + cell[2]
                mask |= tables[:, idx].astype(np.int64) << ti
            achieved[rows, mask] = True
        supers = [m for m in range(1 << nt) if m & target == target]
        t1 = achieved[:, target].copy()
        closure = np.full(N, full, dtype=np.int64)
        for m in supers:
            closure = np.where(achieved[:, m], closure & m, closure)
        tany = closure == target
        t2 = t1.copy()
        sup_nontarget = [m for m in supers if m != target]
        for i1, m1 in enumerate(sup_nontarget):
            for m2 in sup_nontarget[i1:]:
                if m1 & m2 == target:
                    t2 |= achieved[:, m1] & achieved[:, m2]
        # minimal t by BFS over AND-reachable masks (only where tany and not t2)
        tmin = np.where(t1, 1, np.where(t2, 2, 0))
        for g in np.nonzero(tany & ~t2)[0]:
            ach = [m for m in supers if achieved[g, m]]
            frontier = set(ach)
            seen = set(ach)
            t = 1
            while target not in frontier:
                t += 1
                frontier = {a & m for a in frontier for m in ach} - seen
                seen |= frontier
            tmin[g] = t
        res[sides] = dict(t1=t1, t2=t2, tany=tany, tmin=tmin)
    t1 = np.any([v['t1'] for v in res.values()], axis=0)
    t2 = np.any([v['t2'] for v in res.values()], axis=0)
    tany = np.any([v['tany'] for v in res.values()], axis=0)
    tm = np.stack([v['tmin'] for v in res.values()])
    tm = np.where(tm == 0, 10 ** 6, tm).min(axis=0)
    tmin = np.where(tm == 10 ** 6, 0, tm)
    return res, dict(t1=t1, t2=t2, tany=tany, tmin=tmin)


def gadget_types(res, g):
    """side assignments with a t=1 gadget / any-t gadget for relation index g"""
    return ([''.join(s) for s, v in res.items() if v['t1'][g]],
            [''.join(s) + f"(t={v['tmin'][g]})" for s, v in res.items() if v['tany'][g]])


# ------------------------------------------------------------ orbits ----
def cell_perms(k, letter_perms_per_pos, pos_perms):
    """list of permutations of the k^3 cells (as index arrays: new[idx] = old[perm[idx]])"""
    out = []
    cells = list(itertools.product(range(k), repeat=3))
    for pp in pos_perms:
        for lp in itertools.product(*letter_perms_per_pos):
            # image of relation: s in P  ->  sigma(s) in P'; P'[sigma(s)] = P[s]
            perm = np.empty(k ** 3, dtype=np.int64)
            for s in cells:
                t = [None] * 3
                for i in range(3):
                    t[pp[i]] = lp[i][s[i]]
                perm[t[0] * k * k + t[1] * k + t[2]] = s[0] * k * k + s[1] * k + s[2]
            out.append(perm)
    return out


def canon_codes(tables, perms):
    """canonical code (min over group) for each relation; tables (N, k^3) bool, k^3 <= 62"""
    w = (np.int64(1) << np.arange(tables.shape[1], dtype=np.int64))
    best = None
    for perm in perms:
        code = (tables[:, perm].astype(np.int64) * w).sum(axis=1)
        best = code if best is None else np.minimum(best, code)
    return best
