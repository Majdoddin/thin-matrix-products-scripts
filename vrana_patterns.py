"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and reductions that keep the whole
saving".  Supports Appendix A.1 (the details of Section 11), paragraph "Other tensors": for Vrana's Example 4.10 [Vra24], "none of the support
patterns that the flattening ranks allow for a five-term decomposition (..., which give 464 distinct leaf counts)
gives a saving (numerical)" (mode enum).

The profile model of profile_model.py, extended to bases with several inner letters per level.
Per level: outer letters (length-1 outputs), P inner letters (inner products of length q each); a run selects the
output strings with exactly m inner levels: C = C(L,m) O^(L-m) P^m, D = q^m.  A term covering o selected outer and
i selected inner letters gives s(tau) = [t^m] prod (o + i t).  Classes (mult, o, i); classes with o = i = 0 cover
nothing selected and are dropped (their leaves contribute to no output).
  Gamma/m = max_x count(x) - lnC + min(0, s(x) - kappa ln q),  gamma = -Gamma/ln q,
  eps = ln q / max(lnC/2, gamma ln q + c ln R - lnC/2)       (R counts all terms, also dropped ones).
Modes:
  check    the model reproduces the scores 0.003935 (Schoenhage (3,3)) and 0.005825 (the tensor of Lemma 3.1),
           and Gamma for Blaeser's form of the Bini-Capovani-Romani-Lotti decomposition;
  vrana    Vrana's decomposition (eq. (42) of [Vra24]) in its three orientations;
  subsets  the same, reading only some outer or inner letters;
  bini     the original Bini-Capovani-Romani-Lotti decomposition [BCRL79], two orientations;
  binisub  the same, reading only some letters;
  enum     every 5-term letter-support pattern that the flattening ranks of the restrictions of Vrana's tensor allow:
           prints the number of distinct leaf profiles (464) and that none gives a saving.
Usage: python3 vrana_patterns.py [check|vrana|subsets|bini|binisub|enum]
       (needs numpy and profile_model.py beside it; enum uses 8 worker processes, about 15 minutes)
"""
import sys
from math import log
from itertools import combinations_with_replacement, product
import numpy as np
sys.path.insert(0, '.')
from profile_model import H, xlx, nm

def s_rate(x, cls):
    lo = sum(xc for xc, (mu, o, i) in zip(x, cls) if o == 0)
    hi = sum(xc for xc, (mu, o, i) in zip(x, cls) if i > 0)
    if lo > 1 + 1e-9 or hi < 1 - 1e-9: return -np.inf
    def val(u):
        v = -u
        for xc, (mu, o, i) in zip(x, cls):
            if xc <= 0: continue
            v += xc*(u + log(i)) if o == 0 else (xc*log(o) if i == 0 else xc*log(o + i*np.exp(u)))
        return v
    def der(u):
        t = np.exp(u)
        return sum(xc*(i*t)/(o + i*t) for xc, (mu, o, i) in zip(x, cls) if i > 0 and xc > 0) - 1
    a, b = -60.0, 60.0
    if der(a) >= 0: return val(a)
    if der(b) <= 0: return val(b)
    for _ in range(60):
        mid = (a + b)/2
        if der(mid) < 0: a = mid
        else: b = mid
    return val((a + b)/2)

def Gamma(cls, lnC, q, c, kappa=0.5, starts=4, seed=1):
    k = len(cls)
    # if no class is mixed (o > 0 and i > 0), the levels of the o = 0 classes are exactly the m inner levels:
    # parametrize the two groups separately (sum 1 and sum c - 1) instead of penalising an equality constraint
    split = not any(o > 0 and i > 0 for mu, o, i in cls)
    g0 = np.array([o == 0 for mu, o, i in cls])
    def tox(z):
        if not split:
            w = np.exp(z - z.max()); return c*w/w.sum()
        x = np.zeros(k)
        for grp, tot in ((g0, 1.0), (~g0, c - 1.0)):
            if grp.any():
                zz = z[grp]; w = np.exp(zz - zz.max()); x[grp] = tot*w/w.sum()
        return x
    def obj(z):
        if split and (not g0.any() or g0.all()): return -1e9
        x = tox(z)
        lo = sum(xc for xc, (mu, o, i) in zip(x, cls) if o == 0)
        hi = sum(xc for xc, (mu, o, i) in zip(x, cls) if i > 0)
        if lo > 1 + 1e-9 or hi < 1 - 1e-9: return -50.0 - 100*(max(0, lo - 1) + max(0, 1 - hi))
        cnt = xlx(c) - sum(xlx(xc) for xc in x) + sum(xc*log(mu) for xc, (mu, o, i) in zip(x, cls))
        return cnt - lnC + min(0.0, s_rate(x, cls) - kappa*log(q))
    rng = np.random.default_rng(seed); best = -1e18
    for st in range(starts):
        z0 = rng.normal(size=k) if st else np.log(np.array([mu for mu, o, i in cls], float))
        z, v = nm(obj, z0, 0.7); z, v = nm(obj, z, 0.05); best = max(best, v)
    return best

def score(cls, O, P, q, R, cs, kappa=0.5):
    cls = [cl for cl in cls if cl[1] + cl[2] > 0]
    best = (-1.0, None, None, None)
    for c in cs:
        lnC = c*H(1/c) + (c - 1)*log(O) + log(P)
        G = Gamma(cls, lnC, q, c, kappa)
        g = -G/log(q)
        if g <= 0: continue
        eps = log(q)/max(lnC/2, g*log(q) + c*log(R) - lnC/2)
        if eps*g > best[0]: best = (eps*g, c, g, eps)
    return best

def refine(cls, O, P, q, R, lo=None, hi=None, n=30):
    lo = lo or R + 0.5; hi = hi or 4*R + 8
    b = score(cls, O, P, q, R, np.linspace(lo, hi, n))
    if b[1] is None: return b
    span = (hi - lo)/n
    return score(cls, O, P, q, R, np.linspace(max(1.05, b[1] - 2*span), b[1] + 2*span, 21))

def maxGamma_over_c(cls, O, P, q, cs):
    """for gamma<0 cases: min over c of Gamma/m (best case), to report how far from a saving."""
    cls2 = [cl for cl in cls if cl[1] + cl[2] > 0]
    return min((Gamma(cls2, c*H(1/c) + (c - 1)*log(O) + log(P), q, c), c) for c in cs)

def fmt(b): return "no saving (gamma <= 0 at every c)" if b[1] is None else f"eps*gamma = {b[0]:.6f} at c = {b[1]:.2f}, gamma = {b[2]:.4f}, eps = {b[3]:.4f}"

# Vrana's eq. (42): output-letter sets of the 5 terms per orientation; outer = ii, inner = ij
V42 = {'x': [{'x12'}, {'x12', 'x21'}, {'x21'}, {'x12', 'x22'}, {'x11', 'x12'}],
       'y': [{'y12'}, {'y12', 'y21'}, {'y11', 'y12'}, {'y21'}, {'y12', 'y22'}],
       'z': [{'z12'}, {'z12', 'z21'}, {'z12', 'z22'}, {'z11', 'z12'}, {'z21'}]}

def profile(terms, outer, inner):
    from collections import Counter
    cnt = Counter((len(t & outer), len(t & inner)) for t in terms)
    return [(mu, o, i) for (o, i), mu in sorted(cnt.items())]

SCREEN_C = [3, 5, 7, 10, 14, 20, 30]
def _screen(cls):
    cl = [x for x in cls if x[1] + x[2] > 0]
    return min(Gamma(cl, c*H(1/c) + (c - 1)*log(2) + log(2), 2, c, starts=3) for c in SCREEN_C)
def _fine(cls):
    cl = [x for x in cls if x[1] + x[2] > 0]
    return min((Gamma(cl, c*H(1/c) + (c - 1)*log(2) + log(2), 2, c), c) for c in np.linspace(1.2, 45, 40))
def _refine(cls): return refine(list(cls), 2, 2, 2, 5, n=30)

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else 'vrana'
    if task == 'check':
        print("Schoenhage (3,3):", fmt(refine([(9, 1, 1), (1, 0, 1)], 9, 1, 4, 10)), " [0.003935 expected]")
        print("Coppersmith 1982:", fmt(refine([(4, 1, 1), (1, 0, 1)], 4, 1, 2, 5)), " [0.005825 expected]")
        bl = [(2, 1, 1), (2, 0, 1), (1, 1, 2)]   # Blaeser's form of the BCLR decomposition, (mult, #short, #long)
        for c in [1.5, 4, 16]:
            G = Gamma(bl, c*H(1/c) + c*log(2), 2, c)
            print(f"Blaeser BCLR c={c}: Gamma/m = {G:+.4f}, gamma = {-G/log(2):+.4f}   (expected: gamma -0.61 at 1.5, -8.6 at 16)")
    if task == 'vrana':
        for o in 'xyz':
            terms = V42[o]; outer = {f'{o}11', f'{o}22'}; inner = {f'{o}12', f'{o}21'}
            cls = profile(terms, outer, inner)
            print(f"{o}-orientation, classes (mult, outer, inner) = {cls}:", fmt(refine(cls, 2, 2, 2, 5)))
            G, c = maxGamma_over_c(cls, 2, 2, 2, np.linspace(1.2, 40, 25))
            print(f"    best Gamma/m over c in [1.2, 40] = {G:+.4f} at c = {c:.2f}  (gamma = {-G/log(2):+.4f})")
    if task == 'subsets':
        o = 'x'; terms = V42[o]
        for A in [{'x11', 'x22'}, {'x11'}, {'x22'}]:
            for B in [{'x12', 'x21'}, {'x12'}, {'x21'}]:
                cls = profile(terms, A, B)
                b = refine(cls, len(A), len(B), 2, 5)
                G, c = maxGamma_over_c(cls, len(A), len(B), 2, np.linspace(1.2, 40, 25))
                print(f"outer {sorted(A)}, inner {sorted(B)}: classes {cls}: {fmt(b)}; min Gamma/m {G:+.4f} at c={c:.1f}")
    if task == 'bini':
        # original Bini et al. 1979 decomposition: letter sets per term r1..r5
        # w-orientation (outputs): w1 long, w2 short, w3 long, w4 short
        W = [{'w1', 'w4'}, {'w1'}, {'w1', 'w3'}, {'w1', 'w2'}, {'w3', 'w4'}]
        Yo = [{'y1', 'y4'}, {'y2', 'y4'}, {'y4'}, {'y1', 'y2'}, {'y3', 'y4'}]
        for name, terms, outer, inner in [('w', W, {'w2', 'w4'}, {'w1', 'w3'}),
                                          ('y', Yo, {'y1', 'y3'}, {'y2', 'y4'})]:
            cls = profile(terms, outer, inner)
            b = refine(cls, 2, 2, 2, 5)
            print(f"Bini original, {name}-orientation, classes {cls}: {fmt(b)}")
            for c in [1.5, 2, 4, 8, 16, 32]:
                G = Gamma([cl for cl in cls if cl[1]+cl[2] > 0], c*H(1/c) + c*log(2), 2, c)
                print(f"    c={c:5.1f}: Gamma/m = {G:+.4f}  gamma = {-G/log(2):+.4f}")
    if task == 'binisub':
        Yo = [{'y1', 'y4'}, {'y2', 'y4'}, {'y4'}, {'y1', 'y2'}, {'y3', 'y4'}]
        for A in [{'y1', 'y3'}, {'y1'}, {'y3'}]:
            for B in [{'y2', 'y4'}, {'y2'}, {'y4'}]:
                cls = profile(Yo, A, B)
                G, c = maxGamma_over_c(cls, len(A), len(B), 2, np.linspace(1.2, 40, 25))
                print(f"Bini original y: outer {sorted(A)}, inner {sorted(B)}: classes {cls}: min Gamma/m {G:+.4f} at c={c:.1f}", flush=True)
    if task == 'enum':
        # every 5-term letter-support pattern that a border decomposition of M_Lambda (x-orientation) could have,
        # as far as flattening ranks of restrictions tell (FLAT_LB below).  Score each distinct class profile.
        lets = ['x11', 'x22', 'x12', 'x21']
        # for every set A of x-letters, the terms meeting A give a border decomposition of M restricted to A
        # (other x-letters set to 0), so their number is >= the largest flattening rank of that restriction
        LAMB = [(1, 1, 2), (1, 2, 1), (2, 1, 1), (2, 2, 1), (2, 1, 2), (1, 2, 2)]
        FLAT_LB = {}
        for r in range(1, 5):
            for A in combinations_with_replacement(lets, r):
                if len(set(A)) < r: continue
                mons = [(i, j, k) for i, j, k in LAMB if f'x{i}{j}' in A]
                T = np.zeros((4, 4, 4))
                for i, j, k in mons: T[2*(i-1)+j-1, 2*(j-1)+k-1, 2*(k-1)+i-1] = 1
                FLAT_LB[frozenset(A)] = max(np.linalg.matrix_rank(T.reshape(4, 16)),
                                            np.linalg.matrix_rank(T.transpose(1, 0, 2).reshape(4, 16)),
                                            np.linalg.matrix_rank(T.transpose(2, 0, 1).reshape(4, 16)))
        print("flattening lower bounds for the restrictions:", {tuple(sorted(A)): v for A, v in FLAT_LB.items()}, flush=True)
        subsets = [frozenset(s) for r in range(1, 5) for s in combinations_with_replacement(lets, r) if len(set(s)) == r]
        subsets = list(set(subsets))
        profs = {}
        for pat in combinations_with_replacement(range(len(subsets)), 5):
            T = [subsets[p] for p in pat]
            if any(sum(1 for t in T if t & A) < need for A, need in FLAT_LB.items()): continue
            cls = tuple(profile(T, {'x11', 'x22'}, {'x12', 'x21'}))
            profs.setdefault(cls, T)
        print(f"{len(profs)} distinct class profiles", flush=True)
        # screen: Gamma/m at a few c (8 processes); refine only the profiles with Gamma < 0 somewhere
        from multiprocessing import Pool
        items = list(profs.items())
        with Pool(8) as pool:
            scr = pool.map(_screen, [cls for cls, T in items])
        cand = [(cls, T) for (cls, T), g in zip(items, scr) if g < 0]
        top = sorted(zip(scr, items), key=lambda r: r[0])[:6]
        print("smallest screened Gamma/m (closest to a saving), re-checked on 40 values of c in [1.2, 45]:", flush=True)
        with Pool(8) as pool:
            fine = pool.map(_fine, [cls for g, (cls, T) in top])
        for (g, (cls, T)), (gf, cf) in zip(top, fine):
            print(f"  {cls}: screen {g:+.4f}, fine min {gf:+.4f} at c = {cf:.1f}   e.g. {[sorted(t) for t in T]}", flush=True)
        print(f"{len(cand)} profiles with Gamma/m < 0 at some c in {SCREEN_C}", flush=True)
        with Pool(8) as pool:
            bs = pool.map(_refine, [cls for cls, T in cand])
        res = [(b[0], cls, b, [sorted(t) for t in T]) for (cls, T), b in zip(cand, bs)]
        res += [(-1.0, cls, (-1.0, None, None, None), T) for (cls, T), g in zip(items, scr) if g >= 0]
        res.sort(key=lambda r: -r[0])
        for v, cls, b, T in res:
            if b[1] is not None: print(f"  {cls}: {fmt(b)}   e.g. {T}")
        print(f"  ... {sum(1 for r in res if r[2][1] is None)} profiles with no saving")
