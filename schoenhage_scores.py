"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and reductions that keep the whole
saving".  Supports Appendix A.1 (the details of Section 11), paragraph "Exact identities": "Schoenhage's identities attain the bound, and
among those computed (k,n) = (3,3) has the largest score, epsilon*gamma = 0.003935".

The score (exact leaf count, rate value as m -> infinity) of Schoenhage's identity [ZT, Lemma 6] with general
k, n: k*n outer product + inner product of length (k-1)(n-1), kn+1 terms.  D = ((k-1)(n-1))^m,
N0 = k^{L-m} x n^{L-m} (bands balanced so a tile is square of side sqrt(C)), C = K (kn)^{L-m},
alpha_d = C(m,d)(kn)^d, beta_d = C(L,m-d)(kn)^{L-m+d}, encodings (kn+1)^L per band.
Assumes the error term stays harmless and z_ij still has a single term (true for Schoenhage's identity).
"ET saving" is half the score, the Exact Triangle saving through [ZT, Theorem 19] (exponent 3 - eps*gamma/2).
Usage: python3 schoenhage_scores.py      (standard library only)
"""
from math import log
def H(x): return 0.0 if x <= 0 or x >= 1 else -x*log(x)-(1-x)*log(1-x)
def xlx(x): return 0.0 if x <= 0 else x*log(x)
def run(k, n, kappa=0.5):
    T, lD = k*n, log((k-1)*(n-1))
    f = lambda d: H(d) + d*log(T) - kappa*lD
    b = lambda d, c: d*log(T) - xlx(1-d) - xlx(c-1+d) + xlx(c-1)
    cmin = T + 1  # need ratio T*m/(L-m) < 1  -> c > T+1
    best = (0,)
    c = cmin + 0.01
    while c < 8*cmin:
        lo, hi = 0.0, T/(T+1)
        for _ in range(100):
            mid = (lo+hi)/2
            if f(mid) - b(mid, c) < 0: lo = mid
            else: hi = mid
        g = -b(lo, c)/lD
        tile = 0.5*(c*H(1/c) + (c-1)*log(T))
        enc = g*lD + c*log(T+1) - tile
        eps = lD/max(tile, enc)
        if eps*g > best[0]: best = (eps*g, c, g, eps)
        c += 0.02
    return best
for k, n in [(3,3),(3,4),(4,4),(3,5),(4,5),(5,5),(6,6),(8,8),(2,3),(2,5)]:
    v = run(k, n)
    print(f"k={k} n={n}: eps*gamma={v[0]:.6f}  c={v[1]:.2f}  gamma={v[2]:.4f}  eps={v[3]:.4f}  ET saving={v[0]/2:.6f}")
