"""Ancillary file of the paper (see border_apolarity.py); supports the controls of Proposition 3.2.
Extension of border_apolarity.py for weight spaces of multiplicity 2 (needed only for the controls
S(2,2,1) and S(2,2,2)): a B-fixed subspace may meet a 2-dimensional weight space W_w in a line
<v1 + t v2> (or <v2>), a P^1-family.  Closure under the raising operators is solved for t exactly;
a test on a family is decided exactly: the generic rank, and the finitely many t where the rank
drops (roots of the gcd of determinants of random square projections of each t-dependent weight
block; each root is then checked exactly)."""
import itertools, random
from fractions import Fraction as F
import sympy as sp
from border_apolarity import Setup, act, mul, monomials, mono_wt, rank, tensor

t = sp.Symbol('t')

def has_t(polys):
    return any(isinstance(c, sp.Basic) and c.has(t) for p in polys for c in p.values())

def subs(polys, val):
    out = []
    for p in polys:
        q = {}
        for mo, c in p.items():
            v = sp.sympify(c).subs(t, val) if isinstance(c, sp.Basic) else c
            v = F(int(sp.numer(v)), int(sp.denom(v))) if isinstance(v, sp.Basic) else v
            if v != 0: q[mo] = v
        out.append(q)
    return out

def fixed_subspaces_general(S, deg):
    W, mons = S.ann_weight_basis(deg)
    total = len(mons); target = total - min(S.r, total)
    dimW = sum(len(v) for v in W.values())
    if target > dimW: return [], dict(total=total, target=target, dimW=dimW)
    if target == dimW: return [[p for v in W.values() for p in v]], dict(forced=True)
    ws = list(W.keys())
    assert max(len(W[w]) for w in ws) <= 2
    res = []
    choices = [range(len(W[w]) + 1) for w in ws]
    for dims in itertools.product(*choices):
        if sum(dims) != target: continue
        # options for the line in each multiplicity-2 weight with dim 1
        line_ws = [w for w, d in zip(ws, dims) if d == 1 and len(W[w]) == 2]
        assert len(line_ws) <= 1
        opts = [None] if not line_ws else ['t', 'inf']
        for opt in opts:
            U = {}
            for w, d in zip(ws, dims):
                if d == len(W[w]): U[w] = list(W[w])
                elif d == 0: U[w] = []
                else:
                    v1, v2 = W[w]
                    if opt == 't':
                        U[w] = [{mo: v1.get(mo, 0) + t*v2.get(mo, 0) for mo in set(v1) | set(v2)}]
                    else:
                        U[w] = [v2]
            # closure conditions: for each u in U_w and root e, e.u in U_{w+rw}
            conds = []
            ok = True
            for w in ws:
                for u in U[w]:
                    for rw, root in S.roots:
                        img = act(root, u)
                        img = {k: sp.expand(v) if isinstance(v, sp.Basic) else v for k, v in img.items()}
                        img = {k: v for k, v in img.items() if v != 0}
                        if not img: continue
                        w2 = tuple(a + b for a, b in zip(w, rw))
                        if w2 not in U: ok = False; break
                        tgt = U[w2]
                        if len(tgt) == len(W[w2]): continue
                        if not tgt:
                            conds += list(img.values()); continue
                        base = tgt[0]
                        keys = sorted(set(img) | set(base))
                        k0 = keys
                        # img parallel to base: all 2x2 minors vanish
                        for a, b in itertools.combinations(keys, 2):
                            conds.append(sp.expand(sp.sympify(img.get(a, 0))*base.get(b, 0) - sp.sympify(img.get(b, 0))*base.get(a, 0)))
                    if not ok: break
                if not ok: break
            if not ok: continue
            conds = [c for c in (sp.sympify(c) for c in conds) if c != 0]
            polys = [p for w in ws for p in U[w]]
            if not conds:
                res.append(polys)
            elif opt == 't':
                sols = sp.solve(conds, t, dict=True)
                for s in sols:
                    val = s[t]
                    assert val.is_rational, val
                    res.append(subs(polys, val))
            # opt 'inf' or None with nonzero (constant) conds: excluded
    return res, dict(total=total, target=target, dimW=dimW, n=len(res),
                     families=sum(1 for c in res if has_t(c)))

def test_general(S, deg, parts, rnd=random.Random(7)):
    """Returns (pass_generic, generic_dim, mx, list of (t value, dim) where the dim drops)."""
    allp = [p for polys, f in parts for p in polys]
    if not has_t(allp):
        dim, amb, mx = S.test(deg, parts)
        return dim <= mx, dim, mx, []
    mons = monomials(S.dims, deg)
    amb = len(mons); mx = amb - min(S.r, amb)
    byw = {}
    for polys, f in parts:
        for p in polys:
            for i in range(S.dims[f]):
                q = mul(p, {((f, i),): F(1)})
                w = mono_wt(next(iter(q)), S.wt, S.m)
                byw.setdefault(w, []).append(q)
    const = 0; tblocks = []
    for w, qs in byw.items():
        ms = sorted({mo for q in qs for mo in q})
        M = sp.Matrix([[sp.expand(sp.sympify(q.get(mo, 0))) for mo in ms] for q in qs])
        if M.has(t): tblocks.append(M)
        else: const += M.rank()
    gen = 0; roots = set()
    for M in tblocks:
        val = sp.Rational(rnd.randint(10**5, 10**6), rnd.randint(1, 1000))
        g = M.subs(t, val).rank(); gen += g
        if g == 0: continue
        G = None
        for _ in range(3):
            P = sp.Matrix(g, M.rows, lambda i, j: rnd.randint(-5, 5))
            Q = sp.Matrix(M.cols, g, lambda i, j: rnd.randint(-5, 5))
            d = sp.Poly(sp.expand((P*M*Q).det(method='berkowitz')), t)
            G = d if G is None else sp.gcd(G, d)
        for rt in sp.roots(G, multiple=True) if G.degree() > 0 else []:
            roots.add(rt)
    drops = []
    for rt in roots:
        if not rt.is_rational:
            drops.append((rt, 'irrational root; not checked')); continue
        dim = const + sum(M.subs(t, rt).rank() for M in tblocks)
        drops.append((rt, dim))
    total = const + gen
    return total <= mx, total, mx, drops


def run_general(kind, k, n, q, r, log=print):
    A, B, C, T = tensor(kind, k, n, q)
    S = Setup(A, B, C, T, r)
    log(f'== {kind}({k},{n},{q}), r = {r} (general enumeration with P^1 families)')
    log('symmetry:', S.info)
    L1 = [(1, 1, 0), (1, 0, 1), (0, 1, 1)]
    cands = {}
    for e in L1:
        cands[e], inf = fixed_subspaces_general(S, e)
        log(f'  candidates {e}: {len(cands[e])} {inf}')
        if not cands[e]:
            log(f'  no candidate: border rank > {r}'); return 0
    unit = lambda f: tuple(1 if i == f else 0 for i in range(3))
    add = lambda a, b: tuple(x+y for x, y in zip(a, b))
    nsurv = 0
    for choice in itertools.product(*[range(len(cands[e])) for e in L1]):
        parts = [(cands[e][c], f) for e, c in zip(L1, choice) for f in range(3) if add(e, unit(f)) == (1, 1, 1)]
        ok, dim, mx, drops = test_general(S, (1, 1, 1), parts)
        passing_t = [rt for rt, d in drops if not isinstance(d, str) and d <= mx]
        unchecked = [rt for rt, d in drops if isinstance(d, str)]
        status = 'PASS (generic)' if ok else ('PASS at t in %s' % passing_t if passing_t else 'fail')
        if unchecked: status += f' [unchecked irrational t: {unchecked}]'
        log(f'   (111) {choice}: generic dim {dim}, allowed <= {mx}, drops {drops}: {status}')
        if ok or passing_t or unchecked: nsurv += 1
    log(f'  (111) survivors: {nsurv}')
    return nsurv

if __name__ == '__main__':
    import sys
    w = sys.argv[1] if len(sys.argv) > 1 else 'S222'; r = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_general(w[0], int(w[1]), int(w[2]), int(w[3]), r, log=print)
