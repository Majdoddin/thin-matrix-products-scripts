"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and lossless reductions for 3SUM, Exact Triangle and APSP"
(Part I).  The paper's notation differs from the letters used here in a few places: the hashes G, F here are the
paper's re-indexing maps Gamma, Delta; the blocks Z_Q, W_Q are the paper's X_Q, Y_Q; kappa_g = phi_inf^2 is the
paper's mu_infinity^2 (phi_k here is the paper's mu_k).

Sections 1, 7-9 and 16 of the paper: the explicit constants of Corollary 9.1, the rate values of Table 4 (rows
N >= D^18, D^14, D^12, D^10, D^8, D^6), the split-form optima of eps*gamma of Remark 6.2 and
Section 16 (output lines E3), the preprint's numbers at N >= D^14 used in Table 1 (output lines E1, E2), and the
downstream exponents of Section 16.

Run with a python3 that has numpy and scipy (scipy for brentq).  Self-contained; it also reproduces the preprint's own numbers.

Notation (natural logs; D = 4^m; per half L = ceil(c m) levels; the squared run has 2L levels, 2m inner levels).
  kappa_g := prod_{i>=1} (1 - 2^-i)^2 = 0.083398...  (K_g >= kappa_g C(L,m) good inner sets, Lemma 4.1)
  Lambda  := (25/4)^c e^{-c H(1/c)} 4^{1+gamma}
  encodings + tile fit:  D^gamma 25^L / (K_g 4^{L-m}) <= (25/4)/kappa_g (cm+2) Lambda^m <= 75 (cm+2) Lambda^m
  with N >= 4^{(m-1)/eps} this holds once (4^{1/eps}/Lambda)^m >= 4^{1/eps} 75 (cm+2).
  boxes  : t = ceil(2 theta m), rho <= rho_c = 4/(c-1), gamma_box = theta log2((c-1)/4)
  queries: sum_{d<=t} C(2m,d) 4^d <= 4x D^q, x = (1-theta)/theta, q = (H(theta) + theta ln 4)/ln 2.
"""
from math import log, exp, ceil, floor, comb, lgamma, inf
import numpy as np
from scipy.optimize import brentq

def H(x):
    return 0.0 if x <= 0 or x >= 1 else -x*log(x) - (1 - x)*log(1 - x)

KAPPA = 1.0
for i in range(1, 200):
    KAPPA *= (1 - 2.0**-i)
KAPPA = KAPPA**2
CONST = 6.25/KAPPA        # (25/4)/kappa_g

# ---------------- the two identities in the split form (rates; D = 4^m in both) ----------------
SCH = dict(name="Schoenhage", R=10, thmax=0.9,
           A=lambda c: (c*log(10) - 0.5*c*H(1/c) - (c - 1)*log(3))/log(4),
           gbox=lambda c, th: th*log((c - 1)/9)/log(4),
           q=lambda th: (H(th) + th*log(9))/log(4))
SQ = dict(name="squared", R=5, thmax=0.8,
          A=lambda c: (2*c*log(5) - c*H(1/c) - (c - 1)*log(4))/log(4),
          gbox=lambda c, th: 2*th*log((c - 1)/4)/log(4),
          q=lambda th: (H(th) + th*log(4))/log(2))

def best_c(P, th, inv_eps):
    """sup over c > R of min(gamma_box(c, th), 1/eps - A(c)): gamma_box increases, the encoding bound decreases."""
    f = lambda c: P["gbox"](c, th) - (inv_eps - P["A"](c))
    lo, hi = P["R"] + 1e-9, 200.0
    if f(lo) >= 0:
        return max(0.0, min(P["gbox"](lo, th), inv_eps - P["A"](lo))), lo
    c = brentq(f, lo, hi, xtol=1e-12)
    return P["gbox"](c, th), c

def gamma_ds(P, q, inv_eps):          # data structure, query exponent q
    th = brentq(lambda t: P["q"](t) - q, 1e-12, P["thmax"] - 1e-12)
    return best_c(P, th, inv_eps)[0]

def gamma_W(P, kappa, inv_eps):       # |W| <= N^2/D^kappa
    g = lambda th: min(best_c(P, th, inv_eps)[0], kappa - P["q"](th))
    # best_c increases in th, kappa - q decreases: maximise by golden search on a fine grid then refine
    ths = np.linspace(1e-4, P["thmax"] - 1e-4, 4000)
    vals = [g(t) for t in ths]
    k = int(np.argmax(vals))
    a, b = ths[max(k - 1, 0)], ths[min(k + 1, len(ths) - 1)]
    for _ in range(80):
        m1, m2 = a + (b - a)/3, b - (b - a)/3
        if g(m1) < g(m2): a = m1
        else: b = m2
    return max(0.0, g((a + b)/2))

def eg_opt(P, x0):
    """the split-form optimum of eps*gamma (rate value): gamma = min(gamma_box, 1/2 - q) for |W| <= N^2/sqrt D
    (kappa = 1/2), eps = 1/(A(c) + gamma) from the encodings; maximised over the ratio c and theta."""
    from scipy.optimize import minimize
    def f(v):
        c, th = v
        if c <= P["R"] or th <= 0 or th >= P["thmax"]:
            return 1.0
        g = min(P["gbox"](c, th), 0.5 - P["q"](th))
        return -g/(P["A"](c) + g)
    r = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14, maxiter=10000))
    return -r.fun, r.x[0], r.x[1]

def rdown(x, d=4):
    return floor(x*10**d + 1e-12)/10**d

# ---------------- explicit corollaries ----------------
def explicit_sq(c, th, inv_eps):
    rc = 4/(c - 1)
    gb = SQ["gbox"](c, th); q = SQ["q"](th)
    Lam = (25/4)**c*exp(-c*H(1/c))*4**(1 + gb)
    base = 4**inv_eps
    if Lam >= base:
        return None
    r = base/Lam
    m0 = 1
    while r**m0 < base*75*(c*m0 + 2):
        m0 += 1
    return dict(c=c, th=th, gbox=gb, q=q, rc=rc, Lam=Lam, r=r, m0=m0, x=(1 - th)/th)

def exact_m0(c, th, inv_eps, mmax=3000):
    """smallest m0 such that, with exact binomials and K_g >= phi_m phi_{L-m} C(L,m) >= KAPPA C(L,m),
    D^g 25^L/(K_g 4^{L-m}) <= 4^{(m-1)/eps} for all m0 <= m <= mmax."""
    gb = SQ["gbox"](c, th)
    lb = lambda n, k: lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    last_bad = 0
    for m in range(1, mmax + 1):
        L = ceil(c*m)
        lhs = gb*m*log(4) + L*log(25) - (log(KAPPA) + lb(L, m) + (L - m)*log(4))
        if lhs > (m - 1)*inv_eps*log(4):
            last_bad = m
    return last_bad + 1

def check_bound(c, th, inv_eps):
    d = explicit_sq(c, th, inv_eps)
    lb = lambda n, k: lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    out = []
    for m in (10, d["m0"], 200, 1001):
        L = ceil(c*m)
        lhs = d["gbox"]*m*log(4) + L*log(25) - (log(KAPPA) + lb(L, m) + (L - m)*log(4))
        rhs = log(75*(c*m + 2)) + m*log(d["Lam"])
        out.append((m, lhs, rhs, lhs <= rhs, (m - 1)*inv_eps*log(4) >= rhs))
    return out

def query_check(th, ms=(73, 200, 1000)):
    q = SQ["q"](th); x = (1 - th)/th
    res = []
    for m in ms:
        t = ceil(2*th*m)
        s = sum(comb(2*m, d)*4**d for d in range(t + 1))
        res.append((m, log(s)/(m*log(4)), (log(4*x) + q*m*log(4))/(m*log(4)), s <= 4*x*4**(q*m)))
    return res

def explicit_sch(c, th, inv_eps):
    """the preprint's own inequality (eq. encrange), with K_0 = floor(sqrt K) >= sqrt(K)/2 and L <= cm+1."""
    gb = SCH["gbox"](c, th); q = SCH["q"](th)
    Lam = 10**c*exp(-0.5*c*H(1/c))*3**(-(c - 1))*4**gb
    base = 4**inv_eps
    if Lam >= base:
        return None
    r = base/Lam; m0 = 1
    while r**m0 < base*2*(10/3)*(c*m0 + 2)**0.5:
        m0 += 1
    return dict(c=c, th=th, gbox=gb, q=q, Lam=Lam, r=r, m0=m0)

if __name__ == "__main__":
    print(f"kappa_g = phi_inf^2 = {KAPPA:.7f};  (25/4)/kappa_g = {CONST:.3f}  (<= 75: {CONST <= 75})")

    print("\n== calibration: the preprint's Corollary 26 (c = 21, theta = 1/9)")
    g = log(20/9)/(9*log(4)); q = log(72*(9/8)**9)/(9*log(4))
    Lam = 10**21*(20/9)**(1/9)/((21**21/20**20)**0.5*3**20)
    r = 4**18/Lam; m0 = 1
    while r**m0 < 4**18*(21*m0 + 1)**0.5: m0 += 1
    print(f"  gamma = {g:.5f}, q = {q:.5f}, Lambda = {Lam:.4e}, 4^18/Lambda = {r:.4f}, m0 = {m0} (preprint: m >= 60)")
    print(f"  rate value at c=21, q=0.43 (the preprint's Table 2): {rdown(gamma_ds(SCH, q, 1/((SCH['A'](21) + g)**-1*1)) ,4)}"
          f"  [should be 0.0640 at eps = R_21(gamma)]")
    eps21 = 1/(SCH["A"](21) + g)
    print(f"  R_21(gamma) = {eps21:.4f} (the preprint's Table 2: 0.056); best gamma at that eps with q = 0.4277: {gamma_ds(SCH, q, 1/eps21):.4f}")

    print("\n== explicit squared corollary at N >= D^18 (c = 14.5, theta = 1/19)")
    d = explicit_sq(14.5, 1/19, 18)
    print(f"  gamma_box = {d['gbox']:.5f}, q = {d['q']:.5f}, 1/2 - q = {0.5 - d['q']:.5f}, rho_c = {d['rc']:.4f},"
          f" 1/(1-rho_c) = {1/(1 - d['rc']):.4f}")
    print(f"  Lambda = {d['Lam']:.5e}, 4^18 = {4**18:.5e}, margin 4^18/Lambda = {d['r']:.4f}, m0 (bound) = {d['m0']},"
          f" m0 (exact binomials) = {exact_m0(14.5, 1/19, 18)}")
    for row in check_bound(14.5, 1/19, 18):
        print(f"   m={row[0]:5d}: ln LHS (exact binomials) = {row[1]:9.2f} <= ln bound = {row[2]:9.2f}: {row[3]};"
              f"  ln 4^((m-1)18) >= bound: {row[4]}")
    print(f"  query: 4x = {4*d['x']:.0f};  sum_(d<=t) C(2m,d)4^d vs 4x D^q:")
    for m, e1, e2, ok in query_check(1/19):
        print(f"   m={m}: log_D(sum) = {e1:.5f} <= log_D(4x D^q) = {e2:.5f}: {ok}")
    assert 0.092 < d['gbox'] and 0.403 > d['q'] and 0.092 + 0.403 <= 0.5
    print(f"  stated exponents: preprocessing 0.092 < {d['gbox']:.5f}, query 0.403 > {d['q']:.5f};"
          f" 0.092 + 0.403 = {0.092 + 0.403:.3f} <= 1/2")

    print("\n== best explicit (split-form) instances per threshold, c in steps of 0.05, theta = 1/n, m0 <= 150")
    best_eg = None
    for inv in (10, 11, 12, 13, 14, 15, 16, 18):
        cands = []
        for c in np.arange(5.05, 40, 0.05):
            for n in range(6, 80):
                dd = explicit_sq(round(c, 2), 1/n, inv)
                if dd and dd["m0"] <= 150 and dd["gbox"] < 0.5 - dd["q"]:
                    cands.append(dd)
        cands.sort(key=lambda z: -z["gbox"])
        z = cands[0]
        gstated = floor(z["gbox"]*1000)/1000
        if gstated == z["gbox"]: gstated -= 0.001
        qstated = ceil(z["q"]*1000 + 1)/1000
        eg = gstated/inv
        print(f"  N>=D^{inv:2d}: c={z['c']:.2f} theta=1/{round(1/z['th'])}: gamma_box={z['gbox']:.5f} q={z['q']:.4f}"
              f" margin={z['r']:.4f} m0={z['m0']} (exact {exact_m0(z['c'], z['th'], inv)}) ->"
              f" stated gamma {gstated:.3f}, eps*gamma = {eg:.6f}; rate frontier {gamma_W(SQ, 0.5, inv):.4f}")
        if best_eg is None or eg > best_eg[0]:
            best_eg = (eg, inv, z, gstated)
    print(f"  best stated eps*gamma: {best_eg[0]:.6f} at N >= D^{best_eg[1]}")

    print("\n== explicit squared corollary at N >= D^14 (instance (b): c = 11.5, theta = 1/18), the one used downstream")
    d = explicit_sq(11.5, 1/18, 14)
    print(f"  gamma_box = {d['gbox']:.5f}, q = {d['q']:.5f}, 1/2 - q = {0.5 - d['q']:.5f}, rho_c = {d['rc']:.4f},"
          f" 1/(1-rho_c) = {1/(1 - d['rc']):.4f}")
    print(f"  Lambda = {d['Lam']:.5e}, 4^14 = {4**14:.5e}, margin 4^14/Lambda = {d['r']:.4f}, m0 (bound) = {d['m0']},"
          f" m0 (exact binomials) = {exact_m0(11.5, 1/18, 14)}")
    for row in check_bound(11.5, 1/18, 14):
        print(f"   m={row[0]:5d}: ln LHS (exact binomials) = {row[1]:9.2f} <= ln bound = {row[2]:9.2f}: {row[3]};"
              f"  ln 4^((m-1)14) >= bound: {row[4]}")
    print(f"  query: 4x = {4*d['x']:.0f};  sum_(d<=t) C(2m,d)4^d vs 4x D^q:")
    for m, e1, e2, ok in query_check(1/18, (d['m0'], 200, 1000)):
        print(f"   m={m}: log_D(sum) = {e1:.5f} <= log_D(4x D^q) = {e2:.5f}: {ok}")
    assert 0.077 < d['gbox'] and 0.421 > d['q'] and 0.077 + 0.421 <= 0.5
    print(f"  stated exponents: preprocessing 0.077 < {d['gbox']:.5f}, query 0.421 > {d['q']:.5f};"
          f" 0.077 + 0.421 = {0.077 + 0.421:.3f} <= 1/2")
    eps_p = 0.077/28
    print(f"  Exact Triangle (Cor. fg(a)): g = D^(0.077/2) = D^{0.077/2:.4f}, D = n^(1/14): eps' = 0.077/28 = {eps_p:.6f},"
          f" 3 - eps' = {3 - eps_p:.5f} (stated O(n^2.9973) absorbs the log); 3SUM 2 - eps'/2 = {2 - eps_p/2:.6f},"
          f" APSP 3 - eps'/3 = {3 - eps_p/3:.6f}")

    print("\n== downstream exponents (exponents rounded up in the last digit)")
    def downstream(eps, g, label):
        eg = eps*g
        print(f"  {label}: eps*gamma = {eg:.6f}")
        print(f"    preprint's reductions: Exact Triangle n^(3 - eg/2) = n^{3 - eg/2:.5f},"
              f" 3SUM n^(2 - eg/4) = n^{2 - eg/4:.5f}, APSP n^(3 - eg/6) = n^{3 - eg/6:.5f}")
        print(f"    Part II of the paper: 3SUM n^{2 - eg:.5f}+o(1), Exact Triangle and APSP n^{3 - eg:.5f}+o(1)")
    downstream(1/18, 0.063, "preprint Cor 26 (1/18, 0.063)")
    downstream(1/18, 0.092, "explicit squared (1/18, 0.092)")
    downstream(1/14, 0.077, "explicit squared, instance (b) (1/14, 0.077)")
    print(f"  side cost check (Theorem 19's choice of p, n^(log2 7 + 1.5 eps)): eps=1/14: {log(7, 2) + 1.5/14:.4f};"
          f" eps=1/18: {log(7, 2) + 1.5/18:.4f}")
    eg_sq = eg_opt(SQ, (10.5, 0.057))[0]
    print(f"  rate values: split form {eg_sq:.6f} -> 3SUM n^{2 - eg_sq:.5f}; exact form 0.005825 -> n^{2 - 0.005825:.5f}"
          f"  (the exact-form value is computed by shared_variable_variants.py, line H)")

    print("\n== ceiling: eps* = ln4/(5 ln5) =", f"{log(4)/(5*log(5)):.6f}", " vs ln4/(5 ln10) =", f"{log(4)/(5*log(10)):.6f}",
          "; H(1/5) + (4/5) ln 4 - ln 5 =", f"{H(0.2) + 0.8*log(4) - log(5):.2e}")
    print("  Cop82: alpha = 2 log 2/(5 log 5) =", f"{2*log(2)/(5*log(5)):.6f}")
    for c in (5.001, 5.1, 6, 8, 10):
        print(f"   c={c}: R'_c(0) = {1/SQ['A'](c):.5f}")
    print(f"  2 - eps* = {2 - log(4)/(5*log(5)):.4f} (HLS24 remark), 2 - 0.1204 = {2 - 0.12041:.4f}")

    print("\n== the preprint's numbers at N >= D^14 (it states no corollary for this threshold)")
    g15 = SCH["gbox"](15, 1/9)
    print(f"  E1 its Table 2, row c = 15, theta = 1/9: gamma = {g15:.5f} (printed 0.0354), q = {SCH['q'](1/9):.4f},"
          f" eps = R_15(gamma) = {1/(SCH['A'](15) + g15):.4f} (printed 0.079) >= 1/14 = {1/14:.4f}")
    th43 = brentq(lambda t: SCH["q"](t) - 0.43, 1e-12, SCH["thmax"] - 1e-12)
    g14, c14 = best_c(SCH, th43, 14)
    print(f"  E2 re-optimised over c at N >= D^14, q = 0.43: gamma = {g14:.5f} at c = {c14:.2f}"
          f"  (the D^14 row of the rate table below)")

    print("\n== optimum of eps*gamma in the split form, |W| <= N^2/sqrt D (rate values, Remark 6.2)")
    for P, x0 in ((SQ, (10.5, 0.057)), (SCH, (22, 0.11))):
        eg, c, th = eg_opt(P, x0)
        print(f"  E3 {P['name']}: eps*gamma = {eg:.6f} at c = {c:.2f}, theta = {th:.4f}")
    print(f"  ratio {eg_opt(SQ, (10.5, 0.057))[0]/eg_opt(SCH, (22, 0.11))[0]:.3f}")

    print("\n== rate table (split form, values rounded down), Schoenhage / squared")
    qs = (0.25, 0.43, 0.50); ks = (0.50, 1.00)
    hdr = "  threshold | " + " | ".join(f"q={q:.2f}" for q in qs) + " | " + " | ".join(f"kappa={k:.2f}" for k in ks)
    print(hdr)
    for inv in (18, 14, 12, 10, 8, 6):
        cells = []
        for q in qs:
            a = gamma_ds(SCH, q, inv); b = gamma_ds(SQ, q, inv)
            cells.append(f"{rdown(a):.4f}/{rdown(b):.4f}" if a > 0 else f"  -   /{rdown(b):.4f}")
        for k in ks:
            a = gamma_W(SCH, k, inv); b = gamma_W(SQ, k, inv)
            cells.append(f"{rdown(a):.4f}/{rdown(b):.4f}" if a > 0 else f"  -   /{rdown(b):.4f}")
        print(f"  N>=D^{inv:<2d}   | " + " | ".join(cells))
    print("  (the q = 0.4277 query exponent of Corollary 26: Schoenhage", f"{rdown(gamma_ds(SCH, 0.4277, 18)):.4f},",
          f"squared {rdown(gamma_ds(SQ, 0.4277, 18)):.4f} at N >= D^18)")
