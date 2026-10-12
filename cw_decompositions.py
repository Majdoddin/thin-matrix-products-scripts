"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and reductions that keep the whole
saving".  Supports Appendix A.1 (the details of Section 11), paragraph "Other tensors": for the Coppersmith-Winograd tensor T_CW(q) [CW90],
"other decompositions we tried give no saving either" (the numerical part of that statement).

Border decompositions of T_CW(q) other than the standard one, in the profile model of profile_model.py
(support-based leaf count, K = C(L,m) products per run):
  grouped(g): outer indices split into q/g groups of size g; each group has its own lambda*_G
     (covers the g outer outputs of G and z0); the lambda0's merge: profile [(q,1,1),(q/g,g,1),(1,0,1)],
     R = q + q/g + 1 (g = q is the standard decomposition, g = 1 gives every z_i two private-outer terms).
  robust: the grouped decompositions without the pure-inner term lambda0 (the only term that input pruning
     can remove).
  W-style: [(q,1,1),(q,1,0)], R = 2q.
  scan: profiles meeting only the necessary conditions of a CW decomposition
     (every z_i in >= 2 terms: sum mu*o >= 2q; z0 in >= q terms; R >= q+2),
     built as q private mixed terms (q,1,1) + a class of broad terms (mu_b, o_b, i_b) + r0 pure-inner terms.
Prints min over c of Gamma/m, the per-m log of (visited leaves)/(outputs); > 0 means no saving at any c.
Usage: python3 cw_decompositions.py [grouped|robust|wstyle|scan] [q ...]
       (needs numpy and profile_model.py beside it; e.g. "python3 cw_decompositions.py scan 2 4 8")
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math import log
import numpy as np
from profile_model import Gamma, refine

def minGamma(cls, O, q, cs=None):
    R = sum(mu for mu, o, i in cls)
    if cs is None: cs = np.concatenate([np.linspace(1.05, 6, 30), np.linspace(6.5, 3*R + 12, 40)])
    best = None
    for c in cs:
        G, x = Gamma(cls, O, q, c, starts=3)
        if best is None or G < best[0]: best = (G, c)
    return best

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "grouped"
    QS = [int(a) for a in sys.argv[2:]]
    if task == "grouped":
        for q in (QS or [2, 4, 6, 8, 12, 16]):
            for g in [d for d in range(1, q+1) if q % d == 0]:
                cls = [(q, 1, 1), (q//g, g, 1), (1, 0, 1)]
                G, c = minGamma(cls, q, q)
                print(f"grouped q={q:2d} g={g:2d} R={q+q//g+1:3d}  min Gamma/m={G:+.4f} at c={c:.2f}", flush=True)
    elif task == "robust":
        for q in (QS or [2, 4, 8, 16]):
            for g in [d for d in range(1, q+1) if q % d == 0]:
                G, c = minGamma([(q, 1, 1), (q//g, g, 1)], q, q)
                print(f"robust (no lambda0) q={q:2d} g={g:2d}  min Gamma/m={G:+.4f} at c={c:.2f}", flush=True)
    elif task == "wstyle":
        for q in (QS or [2, 4, 8, 16]):
            G, c = minGamma([(q, 1, 1), (q, 1, 0)], q, q)
            print(f"W-style q={q:2d}  min Gamma/m={G:+.4f} at c={c:.2f}", flush=True)
    elif task == "scan":
        rows = []
        for q in (QS or [2, 3, 4, 6, 8, 12, 16]):
            for ob in sorted(set([1, 2, 3, 4, q//2, q]) - {0}):
                if ob > q: continue
                for mub in sorted(set([1, 2, (q + ob - 1)//ob, 2*((q + ob - 1)//ob)])):
                    for ib in [0, 1]:
                        for r0 in [0, 1, 2]:
                            cls = [(q, 1, 1), (mub, ob, ib)] + ([(r0, 0, 1)] if r0 else [])
                            R = q + mub + r0
                            cov = q + mub*ob
                            z0 = q + (mub if ib else 0) + r0
                            if cov < 2*q or z0 < q or R < q + 2: continue
                            G, c = minGamma(cls, q, q)
                            print(f"scan q={q:2d} broad=({mub},{ob},{ib}) r0={r0} R={R:3d}  min Gamma/m={G:+.4f} at c={c:.2f}", flush=True)
