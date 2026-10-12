"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and reductions that keep the whole
saving".  Supports Appendix A.1, paragraph "Border identities", and the closing paragraph of Section 11: "the cases R0 >= 2
not excluded by the private-leaf condition (kn <= 30)" give no saving.

Scores (profile model of profile_model.py) of the sparse class with one shared variable at the largest effective
inner length that Proposition A.5 (border identities with one shared variable) allows, q = (k-1)(n-1) + R0 + 1:
kn terms each covering one outer output and z0, and R0 pure-inner terms.  Cases: R0 = 1 on seven larger shapes,
and R0 = 2..5 on every shape with kn <= 30 that the private-leaf condition R0^2 < q (Lemma A.1) leaves open.
"no saving" means that the optimiser found, at every c of the grid, a class distribution whose visited leaves are
at least the outputs (Gamma >= 0).

Usage: python3 pure_inner_cases.py      (needs numpy and profile_model.py beside it; 7 worker processes, about 20 minutes)
"""
import sys
sys.path.insert(0, '.')
from multiprocessing import Pool
from profile_model import refine

def cases():
    out = []
    for (k, n) in [(2, 8), (2, 9), (2, 10), (3, 6), (3, 7), (4, 5), (5, 5)]:
        out.append((k, n, 1))
    for R0 in (2, 3, 4, 5):
        for k in range(2, 7):
            for n in range(k, 40):
                q = (k-1)*(n-1) + R0 + 1
                if R0*R0 >= q: continue
                if k*n > 30: break
                out.append((k, n, R0))
    return out

def run(c):
    k, n, R0 = c
    q = (k-1)*(n-1) + R0 + 1; O = k*n
    v = refine([(O, 1, 1), (R0, 0, 1)], O, q)
    return c, q, v

if __name__ == "__main__":
    cs = cases()
    print(f"{len(cs)} cases", flush=True)
    with Pool(7) as P:
        for (k, n, R0), q, v in P.imap(run, cs):
            print(f"(k,n)=({k},{n}) R0={R0} q={q}: " + ("no saving" if v[1] is None else
                  f"eps*gamma={v[0]:.6f} (x{v[0]/0.003935:.2f}) c={v[1]:.2f} gamma={v[2]:.4f} eps={v[3]:.4f}"),
                  flush=True)
