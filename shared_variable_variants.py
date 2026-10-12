"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and reductions that keep the whole
saving".  Supports Appendix A.1 (the details of Section 11), paragraph "Border identities": "that a single product per run gives no saving is
computed" (the line "K1 one product per run: no gamma > 0").  It also prints the score 0.005825 of the squared
construction (line H; Remark 6.2 and Appendix A.1) and the lower score 0.004903 of a single run against one shared
right matrix per tile (line A; Section 5, "its rate is lower (epsilon*gamma = 0.0049 ...)").

eps*gamma for the tensor of Lemma 3.1 (Schoenhage's five-term tensor in the form of [Cop82]) in the framework of
[ZT] (arXiv:2610.06783), Section 2, under several ways of handling its shared variable.  Imports profile_model.py
(profile model, rates per m, natural logs).

Orientation: the a-variables are the outputs (Coppersmith's 'Algorithm 2' orientation).  Per level
  inner output a11 = b11 c11 + b12 c12   (inner length 2, every one of the 5 terms contributes),
  outer outputs a12,a13,a22,a23 = (b21,b31) x (c11,c21), each in exactly ONE term,
so the leaf profile is Schoenhage-like: classes [(4,1,1),(1,0,1)], O = 4, q = 2, R = 5.  The catch:
c11 is both an inner and an outer right variable, so the right inputs of the K = C(L,m) products
overlap.  Variants (all with inner j-weight fixed to theta*m, so D = C(m, theta m) and the error
terms vanish exactly; theta = 1/2 unless optimised):
  H  products treated as independent K x K grid.  Not valid for ONE run as stated, but realised
     exactly (same rates) by the squared construction (Sections 4-5 of the paper), so H is the
     achievable value, 0.005825;
  A  shared right block: all K products of a tile use the same column block, made consistent by
     re-indexing the right array through GF(2)-linear hashes (valid for a constant fraction of the
     Q); tile = K row blocks x 1 column block, so N >= K N0 and one right encoding per column block;
  B  packing: only a family of Q with pairwise |Q cap Q'| < theta m, so that the right inputs are
     disjoint (GV existence: rigorous; 'ideal' = the counting upper bound C(L,tm)/C(m,tm), a ceiling);
  K1 one product per run.
Leaf count for a sub-family of density delta: by concavity of min(1, .) the class-average
s -> s + ln(delta) gives an UPPER bound on the visited leaves (Jensen), so gamma below is a lower bound
for the family; for H and A the family is all Q (up to a constant), as in profile_model.py.
Usage: python3 shared_variable_variants.py      (needs numpy and profile_model.py beside it)
"""
import sys
from math import log, lgamma
import numpy as np
sys.path.insert(0, '.')
from profile_model import H, s_rate, count_rate, nm, refine

CLS = [(4, 1, 1), (1, 0, 1)]
R = 5
LN2 = log(2)

def gamma_rate(c, lnq, lnd=0.0, kappa=0.5, starts=4, seed=1):
    """Gamma per m = max_x count - lnC_full - lnd + min(0, s + lnd - kappa lnq); returns gamma, lnC."""
    lnC = c*H(1/c) + (c-1)*log(4)
    def obj(z):
        w = np.exp(z - z.max()); x = c*w/w.sum()
        if x[1] > 1: return -50.0 - 100*(x[1]-1)
        s = s_rate(x, CLS)
        return count_rate(x, CLS, c) - lnC - lnd + min(0.0, s + lnd - kappa*lnq)
    rng = np.random.default_rng(seed); best = -1e18
    for st in range(starts):
        z0 = rng.normal(size=2) if st else np.log(np.array([4.0, 1.0]))
        z, v = nm(obj, z0, 0.7); z, v = nm(obj, z, 0.05)
        best = max(best, v)
    return -best/lnq, lnC + lnd

def eps_square(g, lnq, lnC, c):          # tiling of [ZT]: N >= sqrt(C) and shared encodings
    return lnq/max(lnC/2, g*lnq + c*log(R) - lnC/2)

def eps_shared(g, lnq, c):                 # variant A: tile = K row blocks x one column block
    lnK, lnN0 = c*H(1/c), (c-1)*LN2
    return lnq/max(lnK + lnN0, g*lnq + c*log(R) - lnN0)

def lnV(c, th):        # rate of #{Q' : |Q' cap Q| >= th m}
    taus = np.linspace(th, 1, 400)
    return max(H(t) + (c-1)*H((1-t)/(c-1)) for t in taus)

def scan(name, cs, fn):
    best = (-1, None)
    for c in cs:
        r = fn(c)
        if r and r[0] > best[0]: best = (r[0], (c,) + tuple(r[1:]))
    return best

def run():
    cs = np.linspace(2.5, 40, 151)
    out = {}
    def H_(c):
        g, lnC = gamma_rate(c, LN2)
        return (eps_square(g, LN2, lnC, c)*g, g, eps_square(g, LN2, lnC, c)) if g > 0 else None
    def A_(c):
        g, _ = gamma_rate(c, LN2)
        return (eps_shared(g, LN2, c)*g, g, eps_shared(g, LN2, c)) if g > 0 else None
    def K1_(c):
        lnd = -c*H(1/c); g, lnC = gamma_rate(c, LN2, lnd)
        return (eps_square(g, LN2, lnC, c)*g, g, eps_square(g, LN2, lnC, c)) if g > 0 else None
    def B_(c, ideal=False):
        best = None
        for th in np.linspace(0.5, 0.95, 10):
            lnq = H(th)
            lnd = (c*H(th/c) - H(th) - c*H(1/c)) if ideal else -lnV(c, th)
            lnd = min(lnd, 0.0)
            g, lnC = gamma_rate(c, lnq, lnd, starts=2)
            if g <= 0: continue
            e = eps_square(g, lnq, lnC, c)
            if best is None or e*g > best[0]: best = (e*g, g, e, th, lnd)
        return best
    for name, fn, grid in [("H  independent grid = squared construction S", H_, cs),
                           ("A  shared right block per tile (hash)", A_, cs),
                           ("K1 one product per run", K1_, cs),
                           ("B  packing, GV (rigorous)", B_, np.linspace(2.5, 30, 56)),
                           ("B  packing, ideal (ceiling)", lambda c: B_(c, True), np.linspace(2.5, 30, 56))]:
        v = scan(name, grid, fn)
        if v[1] is None: print(f"{name}: no gamma > 0", flush=True); continue
        # refine c
        c0 = v[1][0]; step = grid[1]-grid[0]
        v = scan(name, np.linspace(max(2.05, c0-2*step), c0+2*step, 41), fn)
        print(f"{name}: eps*gamma={v[0]:.6f}  c={v[1][0]:.2f} gamma={v[1][1]:.4f} eps={v[1][2]:.4f}"
              + (f" theta={v[1][3]:.2f} ln(delta)/m={v[1][4]:.3f}" if len(v[1]) > 3 else ""), flush=True)

if __name__ == "__main__":
    print("check (profile_model.py, Schoenhage 3,3):", "%.6f" % refine([(9, 1, 1), (1, 0, 1)], 9, 4)[0])
    run()
