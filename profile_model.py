"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and reductions that keep the whole
saving".  Shared library for the scores of Section 11 ("Why not other identities") and Appendix A.1; imported by
border_identity_scores.py, shared_variable_variants.py, pure_inner_cases.py, vrana_patterns.py and
cw_decompositions.py.  Run on its own it prints the score of Schoenhage's
identity for (k,n) = (3,3), the value 0.003935 of Appendix A.1, paragraph "Exact identities".

What it computes: the score epsilon*gamma (the rate value of epsilon*gamma in the exact leaf count of
Remark 6.2) of a base identity in the framework of [ZT] (arXiv:2610.06783), Section 2: tiling plus
pruned recursion, with the leaf count of [ZT, Lemma 11] counted exactly.

PROFILE MODEL.  A base identity is described per level by
  - O  : number of outer output variables (outputs of the 'outer' part, one per entry),
  - q  : inner-product length of the inner part (one inner output z0 per level),
  - term classes (mult, o, i): mult terms, each covering o outer outputs and (i = 1) or not (i = 0)
         the inner output z0.
The run selects the output strings with exactly m inner levels (K = C(L, m) products of shape
O^{(L-m)/2} x q^m x O^{(L-m)/2} after balancing; C = K O^{L-m} outputs, D = q^m).
A leaf tau with n_c levels of class c contributes to
  s(tau) = [t^m] prod_c (o_c + i_c t)^{n_c}       selected outputs,
and with wanted density rho = D^{-kappa} the pruned recursion visits ([ZT, Lemma 11] with classes, which
equals the random-W expectation up to constants)
  sum_tau min(1, rho s(tau))  leaves.
All rates per m (L = c m), natural logs, poly(m) dropped:
  count(x) = c ln c - sum x ln x + sum x ln mult         (x_c = n_c / m, sum x = c)
  s(x)     = min_t  sum x_c ln(o_c + i_c t) - ln t       (saddle point; exact rate)
  lnC      = c H(1/c) + (c-1) ln O    (+ lnK_extra if K differs from C(L,m))
  Gamma    = max_x  count(x) - lnC + min(0, s(x) - kappa ln q);   gamma = -Gamma / ln q
  eps      = ln q / max( lnC/2 , gamma ln q + c ln R - lnC/2 )   (tile fit and encodings, [ZT, Section 2.4])
For Schoenhage (3,3): classes [(9,1,1),(1,0,1)], O = 9, q = 4: score 0.003935.
The maximisations are numerical (Nelder-Mead over the class frequencies, a grid over c).

Usage: python3 profile_model.py      (needs numpy)
"""
import sys
from math import log, lgamma, comb
import numpy as np

def H(x):
    return 0.0 if x <= 0 or x >= 1 else -x*log(x) - (1-x)*log(1-x)

def xlx(x):
    return 0.0 if x <= 0 else x*log(x)

def s_rate(x, cls):
    """exponential rate of [t^1] prod_c (o_c + i_c t)^{x_c}  (per m); -inf if not attainable."""
    lo = sum(xc for xc, (mu, o, i) in zip(x, cls) if o == 0 and i == 1)
    hi = sum(xc for xc, (mu, o, i) in zip(x, cls) if i == 1)
    if lo > 1 + 1e-12 or hi < 1 - 1e-12:
        return -np.inf
    def deriv(u):  # d/du of sum x ln(o + i e^u) - u
        t = np.exp(u)
        return sum(xc*(i*t)/(o + i*t) for xc, (mu, o, i) in zip(x, cls) if i == 1 and xc > 0) - 1
    def val(u):
        t = np.exp(u)
        v = -u
        for xc, (mu, o, i) in zip(x, cls):
            if xc <= 0: continue
            if o == 0: v += xc*(u + log(i))
            else: v += xc*log(o + i*t)
        return v
    a, b = -60.0, 60.0
    if deriv(a) >= 0: return val(a)
    if deriv(b) <= 0: return val(b)
    for _ in range(200):
        mid = (a+b)/2
        if deriv(mid) < 0: a = mid
        else: b = mid
    return val((a+b)/2)

def count_rate(x, cls, c):
    return xlx(c) - sum(xlx(xc) for xc in x) + sum(xc*log(mu) for xc, (mu, o, i) in zip(x, cls))

def nm(fun, x0, step, iters=1500, tol=1e-11):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + np.eye(n)[k]*step for k in range(n)]
    vals = [fun(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)[::-1]; pts = [pts[k] for k in order]; vals = [vals[k] for k in order]
        if abs(vals[0] - vals[-1]) < tol: break
        cen = sum(pts[:-1])/n
        xr = cen + (cen - pts[-1]); fr = fun(xr)
        if fr > vals[0]:
            xe = cen + 2*(cen - pts[-1]); fe = fun(xe)
            pts[-1], vals[-1] = (xe, fe) if fe > fr else (xr, fr)
        elif fr > vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5*(pts[-1] - cen); fc = fun(xc)
            if fc > vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5*(p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [fun(p) for p in pts[1:]]
    k = int(np.argmax(vals)); return pts[k], vals[k]

def Gamma(cls, O, q, c, kappa=0.5, lnK_extra=0.0, starts=6, seed=1):
    """max over class frequencies of the per-m log of (visited leaves)/C."""
    lnC = c*H(1/c) + (c-1)*log(O) + lnK_extra
    k = len(cls)
    def obj(z):
        w = np.exp(z - z.max()); x = c*w/w.sum()
        lo = sum(xc for xc, (mu, o, i) in zip(x, cls) if o == 0 and i == 1)
        hi = sum(xc for xc, (mu, o, i) in zip(x, cls) if i == 1)
        if lo > 1 or hi < 1: return -50.0 - 100*(max(0, lo-1) + max(0, 1-hi))
        s = s_rate(x, cls)
        return count_rate(x, cls, c) - lnC + min(0.0, s - kappa*log(q))
    rng = np.random.default_rng(seed)
    best = (-1e18, None)
    for st in range(starts):
        z0 = rng.normal(size=k) if st else np.log(np.array([mu for mu, o, i in cls], float))
        z, v = nm(obj, z0, 0.7)
        z, v = nm(obj, z, 0.05)
        if v > best[0]:
            w = np.exp(z - z.max()); best = (v, c*w/w.sum())
    return best

def score(cls, O, q, kappa=0.5, cs=None, lnK_extra_fn=None, enc=True):
    """best eps*gamma over c; returns (eg, c, gamma, eps, x*)."""
    R = sum(mu for mu, o, i in cls)
    if cs is None: cs = np.linspace(R + 0.5, 4*R + 8, 120)
    best = (-1, None, None, None, None)
    for c in cs:
        ext = lnK_extra_fn(c) if lnK_extra_fn else 0.0
        G, x = Gamma(cls, O, q, c, kappa, ext, starts=3)
        g = -G/log(q)
        if g <= 0: continue
        lnC = c*H(1/c) + (c-1)*log(O) + ext
        tile = lnC/2
        encr = g*log(q) + c*log(R) - lnC/2 if enc else -1
        eps = log(q)/max(tile, encr)
        if eps*g > best[0]: best = (eps*g, c, g, eps, x)
    return best

def refine(cls, O, q, kappa=0.5, enc=True, lnK_extra_fn=None):
    b = score(cls, O, q, kappa, enc=enc, lnK_extra_fn=lnK_extra_fn)
    if b[1] is None: return b
    R = sum(mu for mu, o, i in cls)
    span = (4*R + 8 - R)/120
    cs = np.linspace(max(1.01, b[1] - 2*span), b[1] + 2*span, 41)
    return score(cls, O, q, kappa, cs=cs, enc=enc, lnK_extra_fn=lnK_extra_fn)

if __name__ == "__main__":
    print("== Schoenhage (3,3) ==")
    v = refine([(9, 1, 1), (1, 0, 1)], 9, 4)
    print(f"eps*gamma={v[0]:.6f} c={v[1]:.2f} gamma={v[2]:.4f} eps={v[3]:.4f}")
