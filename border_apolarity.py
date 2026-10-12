"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and lossless reductions for 3SUM,
Exact Triangle and APSP".  Supports Proposition 3.2: the border rank of G(2,2,2) = <2,1,2> (+) <1,2,1> is 6.

Weak border apolarity [BB21, Theorem 1.2 with the Fixed Ideal Theorem 4.3; computational form as in CHL23]:
if the border rank of T is at most r, then for every connected solvable group B fixing T there is a B-fixed
multigraded ideal I in Sym(A*) (x) Sym(B*) (x) Sym(C*), contained in the annihilator of T, with
codim I_D = min(r, dim S_D) in every multidegree D.  This script takes B = the maximal torus of the stabiliser
of T times its positive root subgroups, enumerates the B-fixed candidates in multidegrees (110), (101), (011)
when every torus weight there occurs once (a finite list without parameters), and applies the (111) test:
I_110 C* + I_101 B* + I_011 A* must have codimension >= r in A* (x) B* (x) C*.  Ranks are exact (Fractions).
If some weight occurs more than once, this script does not apply and says so; border_apolarity_family.py
(sympy) handles that case and runs the controls S(2,2,1) (border rank 5) and S(2,2,2) (border rank >= 6).

Tensor convention (Appendix A): A = x_i, p_s (, c); B = y_j, q_s; C = z_ij, z0.
G(k,n,q) = sum x_i y_j z_ij + sum_s p_s q_s z0;  S(k,n,q) = G(k,n,q) + c y_1 z0.

Usage: python3 border_apolarity.py [G222|G221] [r]   (default G222 5; standard library only; under 2 s)
  G222 5: no candidate pair survives the (111) test, so border rank > 5 (it is <= 6 by its 4 + 2 terms).
  Controls: G221 5 (Schoenhage's <2,1,2> (+) <1,1,1>, border rank 5) and G222 6 have survivors.
"""
import itertools, sys, random, json, time
from fractions import Fraction as F

# ---------------------------------------------------------------- tensors
def tensor(kind, k, n, q):
    A = [f'x{i}' for i in range(1, k+1)] + [f'p{s}' for s in range(1, q+1)] + (['c'] if kind == 'S' else [])
    B = [f'y{j}' for j in range(1, n+1)] + [f'q{s}' for s in range(1, q+1)]
    C = [f'z{i}{j}' for i in range(1, k+1) for j in range(1, n+1)] + ['z0']
    E = [(f'x{i}', f'y{j}', f'z{i}{j}') for i in range(1, k+1) for j in range(1, n+1)]
    E += [(f'p{s}', f'q{s}', 'z0') for s in range(1, q+1)]
    if kind == 'S':
        E += [('c', 'y1', 'z0')]
    T = {(A.index(a), B.index(b), C.index(c)): F(1) for a, b, c in E}
    return A, B, C, T

# ---------------------------------------------------------------- exact linear algebra
def rref(rows, ncols):
    M = [list(r) for r in rows]
    piv = []
    rk = 0
    for col in range(ncols):
        p = next((i for i in range(rk, len(M)) if M[i][col] != 0), None)
        if p is None:
            continue
        M[rk], M[p] = M[p], M[rk]
        pv = M[rk][col]
        M[rk] = [v / pv for v in M[rk]]
        for i in range(len(M)):
            if i != rk and M[i][col] != 0:
                f = M[i][col]
                M[i] = [a - f*b for a, b in zip(M[i], M[rk])]
        piv.append(col); rk += 1
    return M[:rk], piv

def rank(rows, ncols):
    if not rows:
        return 0
    return len(rref(rows, ncols)[0])

def nullspace(rows, ncols):
    R, piv = rref(rows, ncols)
    free = [c for c in range(ncols) if c not in piv]
    basis = []
    for f in free:
        v = [F(0)]*ncols; v[f] = F(1)
        for r, pc in zip(R, piv):
            v[pc] = -r[f]
        basis.append(v)
    return basis

# ---------------------------------------------------------------- symmetry
def stabilizer(A, B, C, T):
    """Lie algebra of the stabiliser of T in gl(A)+gl(B)+gl(C), as triples of matrices."""
    na, nb, nc = len(A), len(B), len(C)
    nun = na*na + nb*nb + nc*nc
    def ix(f, i, j):
        return [0, na*na, na*na+nb*nb][f] + i*[na, nb, nc][f] + j
    eqs = []
    for a in range(na):
        for b in range(nb):
            for c in range(nc):
                row = [F(0)]*nun
                # (X.T)[a,b,c] = sum_a' X[a,a'] T[a',b,c]
                for (a2, b2, c2), v in T.items():
                    if b2 == b and c2 == c: row[ix(0, a, a2)] += v
                    if a2 == a and c2 == c: row[ix(1, b, b2)] += v
                    if a2 == a and b2 == b: row[ix(2, c, c2)] += v
                eqs.append(row)
    return eqs, ix, nun

def torus_and_roots(A, B, C, T, seed=1):
    na, nb, nc = len(A), len(B), len(C)
    eqs, ix, nun = stabilizer(A, B, C, T)
    g = nullspace(eqs, nun)
    # diagonal torus: unknowns restricted to the diagonals
    diag = [ix(0, i, i) for i in range(na)] + [ix(1, i, i) for i in range(nb)] + [ix(2, i, i) for i in range(nc)]
    sub = [[r[d] for d in diag] for r in eqs]
    tor = nullspace(sub, len(diag))
    m = len(tor)
    # weight of basis vector e of factor f: vector (h_1(e),...,h_m(e))
    wt = {}
    for f, nf, off in [(0, na, 0), (1, nb, na), (2, nc, na+nb)]:
        for i in range(nf):
            wt[(f, i)] = tuple(tor[t][off+i] for t in range(m))
    # weight spaces of g
    def sub_w(w1, w2):
        return tuple(a - b for a, b in zip(w1, w2))
    groups = {}
    for f, nf in [(0, na), (1, nb), (2, nc)]:
        for i in range(nf):
            for j in range(nf):
                groups.setdefault(sub_w(wt[(f, i)], wt[(f, j)]), []).append(ix(f, i, j))
    rnd = random.Random(seed)
    ell = [rnd.randint(1000, 10**6) for _ in range(m)]
    roots = []
    zero = tuple(F(0) for _ in range(m))
    dims = {}
    for w, unk in groups.items():
        others = [u for u in range(nun) if u not in unk]
        # g_w = g intersect span(unk): solve eqs with others = 0
        subeqs = [[r[u] for u in unk] for r in eqs]
        ns = nullspace(subeqs, len(unk))
        dims[w] = len(ns)
        if w == zero or not ns:
            continue
        lv = sum(F(l)*x for l, x in zip(ell, w))
        assert lv != 0
        if lv > 0:
            for v in ns:
                full = [F(0)]*nun
                for u, x in zip(unk, v):
                    full[u] = x
                roots.append((w, full))
    def unpack(full):
        X = [[full[ix(0, i, j)] for j in range(na)] for i in range(na)]
        Y = [[full[ix(1, i, j)] for j in range(nb)] for i in range(nb)]
        Z = [[full[ix(2, i, j)] for j in range(nc)] for i in range(nc)]
        return X, Y, Z
    info = dict(dim_stab=len(g), dim_torus=m, dim_g0=dims.get(zero, 0),
                n_pos=len(roots), dim_check=sum(dims.values()))
    return wt, [(w, unpack(fl)) for w, fl in roots], info

# ---------------------------------------------------------------- polynomials on A*+B*+C*
# a monomial is a sorted tuple of variables (f, i) meaning the dual basis vector of factor f index i.
def mono_wt(mon, wt, m):
    s = [F(0)]*m
    for v in mon:
        for t in range(m):
            s[t] -= wt[v][t]        # dual vectors carry minus the weight
    return tuple(s)

def monomials(dims, deg):
    out = [()]
    for f in range(3):
        nxt = []
        for comb in itertools.combinations_with_replacement(range(dims[f]), deg[f]):
            for mo in out:
                nxt.append(mo + tuple((f, i) for i in comb))
        out = nxt
    return [tuple(sorted(mo)) for mo in out]

def act(root, poly):
    """Lie algebra element (X,Y,Z) acting on a polynomial in the dual variables: on factor f it
    acts on the dual basis by -M^T, i.e. alpha_j -> -sum_i M[j][i] alpha_i."""
    mats = root
    out = {}
    for mon, cf in poly.items():
        for pos, (f, j) in enumerate(mon):
            M = mats[f]
            for i in range(len(M)):
                if M[j][i] != 0:
                    newm = tuple(sorted(mon[:pos] + ((f, i),) + mon[pos+1:]))
                    out[newm] = out.get(newm, F(0)) - M[j][i]*cf
    return {k: v for k, v in out.items() if v != 0}

def mul(p1, p2):
    out = {}
    for m1, c1 in p1.items():
        for m2, c2 in p2.items():
            m = tuple(sorted(m1 + m2))
            out[m] = out.get(m, F(0)) + c1*c2
    return {k: v for k, v in out.items() if v != 0}

# ---------------------------------------------------------------- the setup
class Setup:
    def __init__(self, A, B, C, T, r):
        self.A, self.B, self.C, self.T, self.r = A, B, C, T, r
        self.dims = (len(A), len(B), len(C))
        self.wt, self.roots, self.info = torus_and_roots(A, B, C, T)
        self.m = len(next(iter(self.wt.values())))

    def name(self, v):
        return [self.A, self.B, self.C][v[0]][v[1]] + '*'

    def pstr(self, p):
        return ' + '.join(f'{c}*' + '.'.join(self.name(v) for v in mo) for mo, c in sorted(p.items()))

    def ann_weight_basis(self, deg):
        """Weight basis of Ann(T) in multidegree deg (only degrees with total 2 or 3 are relevant);
        returns dict weight -> list of polys (basis of that weight space)."""
        mons = monomials(self.dims, deg)
        byw = {}
        for mo in mons:
            byw.setdefault(mono_wt(mo, self.wt, self.m), []).append(mo)
        out = {}
        for w, ms in byw.items():
            # conditions: pairing with T contracted. For total degree 2 with two distinct factors
            # (f,g), Ann = the polys p with sum_{mon} p[mon] T(..) = 0 for each basis vector of
            # the remaining factor. For degree with a repeated factor or degree 3 (1,1,1) we only
            # need (1,1,1) where Ann = T-perp hyperplane-ish; we handle the cases used below.
            if sum(deg) == 2 and max(deg) == 1:
                f, g = [i for i in range(3) if deg[i] == 1]
                h = 3 - f - g
                eqs = []
                for k in range(self.dims[h]):
                    row = []
                    for mo in ms:
                        idx = [None]*3
                        for (ff, i) in mo: idx[ff] = i
                        idx[h] = k
                        row.append(self.T.get(tuple(idx), F(0)))
                    eqs.append(row)
                ns = nullspace(eqs, len(ms))
            elif sum(deg) == 3 and deg == (1, 1, 1):
                row = []
                for mo in ms:
                    idx = [None]*3
                    for (ff, i) in mo: idx[ff] = i
                    row.append(self.T.get(tuple(idx), F(0)))
                ns = nullspace([row], len(ms))
            else:
                # T has no component of this multidegree: everything annihilates T
                ns = [[F(1) if a == b else F(0) for b in range(len(ms))] for a in range(len(ms))]
            if ns:
                out[w] = [{mo: c for mo, c in zip(ms, v) if c != 0} for v in ns]
        return out, mons

    def fixed_subspaces(self, deg):
        """All B-fixed subspaces of Ann(T)_deg of dimension dim(S^deg) - min(r, dim S^deg).
        Requires 1-dimensional weight spaces of Ann(T)_deg, except for the forced case."""
        W, mons = self.ann_weight_basis(deg)
        total = len(mons)
        target = total - min(self.r, total)
        dimW = sum(len(v) for v in W.values())
        if target > dimW:
            return [], dict(total=total, target=target, dimW=dimW)
        if target == dimW:
            return [[p for v in W.values() for p in v]], dict(total=total, target=target, dimW=dimW, forced=True)
        mult = max(len(v) for v in W.values())
        if mult > 1:
            raise NotImplementedError(f'weight multiplicity {mult} in degree {deg}')
        ws = list(W.keys())
        vec = {w: W[w][0] for w in ws}
        # closure graph: w -> set of weights w' with root.vec[w] having nonzero component at w'
        edges = {w: set() for w in ws}
        for w in ws:
            for rw, root in self.roots:
                img = act(root, vec[w])
                if img:
                    w2 = tuple(a + b for a, b in zip(w, rw))  # alpha_k -> alpha_j, weight -wt(j) = -wt(k) + rw
                    # sanity: img must be a multiple of vec[w2]
                    assert w2 in vec, (deg, w, rw)
                    base = vec[w2]
                    k0 = next(iter(base))
                    ratio = img.get(k0, F(0)) / base[k0]
                    assert {k: v for k, v in img.items()} == {k: ratio*v for k, v in base.items()}, 'not multiple'
                    edges[w].add(w2)
        res = []
        for sub in itertools.combinations(range(len(ws)), target):
            S = {ws[i] for i in sub}
            if all(edges[w] <= S for w in S):
                res.append([vec[w] for w in ws if w in S])
        return res, dict(total=total, target=target, dimW=dimW, n=len(res))

    def test(self, deg, parts):
        """parts: list of (list of polys, factor f) meaning polys * (dual basis of factor f).
        Returns (dim of span, ambient dim, needed max = ambient - min(r, ambient))."""
        mons = monomials(self.dims, deg)
        mi = {mo: t for t, mo in enumerate(mons)}
        byw = {}
        for polys, f in parts:
            for p in polys:
                for i in range(self.dims[f]):
                    q = mul(p, {((f, i),): F(1)})
                    w = mono_wt(next(iter(q)), self.wt, self.m)
                    byw.setdefault(w, []).append(q)
        total = 0
        for w, qs in byw.items():
            ms = sorted({mo for q in qs for mo in q})
            loc = {mo: t for t, mo in enumerate(ms)}
            rows = []
            for q in qs:
                row = [F(0)]*len(ms)
                for mo, c in q.items(): row[loc[mo]] = c
                rows.append(row)
            total += rank(rows, len(ms))
        amb = len(mons)
        return total, amb, amb - min(self.r, amb)


def run(kind, k, n, q, r, out):
    t0 = time.time()
    A, B, C, T = tensor(kind, k, n, q)
    S = Setup(A, B, C, T, r)
    log = lambda *a: (print(*a), out.write(' '.join(map(str, a)) + '\n'))
    log(f'== {kind}({k},{n},{q}), r = {r}; A={A} B={B} C={C}')
    log('symmetry:', S.info)
    for w, (X, Y, Z) in S.roots:
        terms = []
        for f, M, names in [(0, X, A), (1, Y, B), (2, Z, C)]:
            for i in range(len(M)):
                for j in range(len(M)):
                    if M[i][j] != 0: terms.append(f'{M[i][j]}*{names[i]}<-{names[j]}')
        log('  positive root vector:', ', '.join(terms))
    cands = {}
    def get(deg):
        if deg not in cands:
            try:
                res, inf = S.fixed_subspaces(deg)
            except NotImplementedError as e:
                log(f'  {deg}: {e}'); res, inf = None, None
            cands[deg] = res
            if res is not None:
                log(f'  candidates {deg}: {len(res)}  {inf}')
        return cands[deg]
    def unit(f): return tuple(1 if i == f else 0 for i in range(3))
    def add(a, b): return tuple(x+y for x, y in zip(a, b))
    # stage 1: (111) test on triples of (110),(101),(011) candidates
    L1 = [(1, 1, 0), (1, 0, 1), (0, 1, 1)]
    for e in L1:
        if get(e) is None:
            log(f'  degree {e} is not multiplicity-free: this script does not apply (use border_apolarity_family.py)'); return S, cands, None
        if not get(e):
            log(f'  no candidate in degree {e}: border rank > {r}'); return S, cands, []
    survivors = []
    for choice in itertools.product(*[range(len(cands[e])) for e in L1]):
        sel = dict(zip(L1, choice))
        parts = [(cands[e][sel[e]], f) for e in L1 for f in range(3) if add(e, unit(f)) == (1, 1, 1)]
        dim, amb, mx = S.test((1, 1, 1), parts)
        log(f'   (111) test {choice}: dim {dim}, allowed <= {mx}', 'PASS' if dim <= mx else 'fail')
        if dim <= mx:
            survivors.append(sel)
    log(f'  stage 1 survivors: {len(survivors)}')
    # stage 2: (210),(120),(201),(021),(102),(012) tests with the degree-2 pure components
    stage2 = []
    for sel in survivors:
        for d, e2, f2 in [((2, 1, 0), (2, 0, 0), 1), ((1, 2, 0), (0, 2, 0), 0), ((2, 0, 1), (2, 0, 0), 2),
                          ((0, 2, 1), (0, 2, 0), 2), ((1, 0, 2), (0, 0, 2), 0), ((0, 1, 2), (0, 0, 2), 1)]:
            mixed = [(cands[e][sel[e]], f) for e in L1 for f in range(3) if add(e, unit(f)) == d]
            dim0, amb, mx = S.test(d, mixed)
            c2 = get(e2)
            if c2 is None:
                log(f'   {d}: mixed part alone dim {dim0} <= {mx}?', dim0 <= mx, '(pure part not enumerated)')
                continue
            ok = [i for i, P in enumerate(c2) if S.test(d, mixed + [(P, f2)])[0] <= mx]
            log(f'   {d}: mixed part alone dim {dim0}, allowed {mx}; pure {e2} candidates passing: {len(ok)}/{len(c2)}')
        stage2.append(sel)
    log(f'  time {time.time()-t0:.1f}s')
    return S, cands, survivors


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'G222'
    r = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    kind, k, n, q = which[0], int(which[1]), int(which[2]), int(which[3])
    import io
    run(kind, k, n, q, r, io.StringIO())
