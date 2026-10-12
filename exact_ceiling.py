"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and lossless reductions for 3SUM, Exact Triangle and APSP"
(Part I).  Supports the last paragraph of Section 8: with W known in advance and the exact leaf count of
Remark 6.2, the range of D <= N^eps extends to about eps = 0.1784 for the squared construction (Schoenhage's
identity: 0.1292).  These are numerical roots (kind N), not theorems, and they concern only the non-adaptive
theorem (W given in advance).

Notation.  Rates are per m, in natural logarithms, for one half of the squared run; every rate of the squared run
doubles, so gamma and eps equal the half's values.  Here q is the inner length per level (the paper's q_0 of
Proposition 8.1), O the number of non-private terms, R the number of terms, c = L/m:
  Schoenhage's identity:      O = 9, q = 4, R = 10   (D = 4^m)
  squared construction, half: O = 4, q = 2, R = 5    (D = 4^m)
  alpha_d = C(m,d)(R-1)^d,  beta_d = C(L,m-d)(R-1)^(L-m+d),  kappa = 1/2 (|W| <= N^2/sqrt D).
Constraints (the paper's (m:eq:encrange) in rate form; the function T below is the tile-fit rate, not the paper's
set T(.) of Section 5):
  tile fit      ln N >= T(c) m,   T(c) = (c H(1/c) + (c-1) ln O)/2
  encodings     ln N >= (A(c) + gamma ln q) m,   A(c) = c ln R - T(c)
  so 1/eps = max(T, A + gamma ln q)/ln q.
Exact leaf count (Lemma 6.1(c), sum_d min(|U| alpha_d, beta_d)): gamma_exact(c) = -max_y min(f(y), b(y))/ln q with
  f(y) = H(y) + y ln(R-1) - kappa ln q        (rate of |U| alpha_d / M, d = y m),
  b(y) = c[H((1-y)/c) - H(1/c)] + y ln(R-1)   (rate of beta_d / M).
gamma_exact(c) > 0 iff the positive root y0(c) of b lies below the root y_f of f.  Below the threshold c_min the
leaves of some order are both more numerous than the tile's entries and needed by the wanted ones, so gamma <= 0;
the ceiling is eps at c_min with gamma = 0 (a supremum, not attained).  Unlike the split form, which needs c > R,
the exact count works down to c_min < R.

How to read the output: the line "eps at c_min, gamma=0" gives 0.17843 (squared) and 0.12923 (Schoenhage), against
the split-form limits 0.17227 and 0.12041 of Proposition 8.1; the last lines show gamma_exact and eps just above
c_min and at c = R.

Run: python3 exact_ceiling.py      (numpy, scipy; about 12 s)
"""
from math import log
import numpy as np
from scipy.optimize import brentq

PROF = {"schoenhage": dict(O=9, q=4, R=10), "squared": dict(O=4, q=2, R=5)}
KAPPA = 0.5


def H(x):
    return 0.0 if x <= 0 or x >= 1 else -x*log(x) - (1 - x)*log(1 - x)


def T(c, O, q, R): return 0.5*(c*H(1/c) + (c - 1)*log(O))
def A(c, O, q, R): return c*log(R) - T(c, O, q, R)


def eps_of(c, g, O, q, R):
    return log(q)/max(T(c, O, q, R), A(c, O, q, R) + g*log(q))


def gamma_exact_any(c, O, q, R, kappa=KAPPA):
    f = lambda y: H(y) + y*log(R - 1) - kappa*log(q)
    b = lambda y: c*(H((1 - y)/c) - H(1/c)) + y*log(R - 1)
    hi = (R - 1)/R
    # b is concave (b'' = -1/(1-y) - 1/(c-1+y) < 0); f increasing on [0,hi]; max of min(f,b):
    # scan finely, then refine at the crossing
    ys = np.linspace(1e-9, hi, 200001)
    fv = np.array([f(y) for y in ys]); bv = np.array([b(y) for y in ys])
    v = np.minimum(fv, bv); k = int(v.argmax()); G = v[k]
    return -G/log(q), ys[k]


for name in ("schoenhage", "squared"):
    P = PROF[name]; O, q, R = P["O"], P["q"], P["R"]
    yf = brentq(lambda y: H(y) + y*log(R - 1) - KAPPA*log(q), 1e-12, (R - 1)/R)
    g = lambda c: gamma_exact_any(c, O, q, R)[0]
    cmin = brentq(g, 1.5, R + 0.5, xtol=1e-10)
    print(f"{name}: y_f={yf:.5f}  c_min (gamma_exact=0) = {cmin:.5f}  (split form needs c > {R})")
    print(f"   eps at c_min, gamma=0: {eps_of(cmin, 0, O, q, R):.5f}   vs split ceiling 2 ln q/(R ln R) = {2*log(q)/(R*log(R)):.5f}")
    print(f"   A(c_min)/ln q = {A(cmin, O, q, R)/log(q):.4f}  T/ln q = {T(cmin, O, q, R)/log(q):.4f}")
    for c in (cmin + 0.01, cmin + 0.1, R - 0.5, R, R + 0.5):
        gg = g(c); print(f"   c={c:.4f}: gamma_exact={gg:.5f} eps={eps_of(c, gg, O, q, R):.5f}")
