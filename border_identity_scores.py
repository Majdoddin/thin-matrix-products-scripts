"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and reductions that keep the whole
saving".  Supports Appendix A.1 (the details of Section 11), paragraphs "Exact identities" and "Border identities": the scores (epsilon*gamma,
profile model of profile_model.py)
  0.003935  Schoenhage's identity, (k,n) = (3,3), q = 4;
  0.005825  (2,2), q = 2: the tensor of Lemma 3.1 (the member k = n = 2 of the family below);
  0.005489, 0.004771, 0.004720  the family <k,1,n> plus an inner product of length (k-1)(n-1)+1 in kn+1 terms,
            for (2,3), (3,3), (2,4);
  0.005730  the open shape 2x4 at effective inner length 5 (and 0.004667 for 2x5 at 6): "at most 0.00573";
and further cases at the inner lengths the border-rank bounds of Appendix A leave open or exclude.
Profile: kn mixed terms (each covering one outer output and z0) and R0 pure-inner terms; O = kn outer outputs;
inner length q (a shared variable counted in q).
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 border_identity_scores.py   (needs numpy and profile_model.py beside it)
"""
import sys
sys.path.insert(0, '.')
from profile_model import refine

CASES = [  # (k, n, q, R0, note)
    (3, 3, 4, 1, "Schoenhage (3,3), reference"),
    (2, 2, 2, 1, "q=(k-1)(n-1)+1: Coppersmith / disjoint bound"),
    (2, 2, 3, 1, "shared letter, q'=(k-1)(n-1)+1"),
    (2, 3, 3, 1, "q=(k-1)(n-1)+1"),
    (2, 3, 4, 1, "shared letter, q'=(k-1)(n-1)+1"),
    (2, 4, 4, 1, "q=(k-1)(n-1)+1"),
    (2, 4, 5, 1, "open shape 2x4, effective length 5"),
    (2, 5, 6, 1, "open shape 2x5, effective length 6"),
    (3, 3, 5, 1, "q=(k-1)(n-1)+1"),
    (3, 3, 6, 1, "shared letter, q'=(k-1)(n-1)+1"),
    (2, 2, 3, 2, "R0=2 at border bound"),
    (2, 3, 4, 2, "R0=2 at border bound"),
    (2, 3, 5, 2, "R0=2, shared"),
    (3, 3, 6, 2, "R0=2 at border bound"),
    (3, 3, 7, 2, "R0=2, shared"),
    (2, 4, 5, 2, "R0=2 at border bound"),
    (3, 4, 8, 2, "R0=2 at border bound"),
]
if __name__ == "__main__":
    for k, n, q, R0, note in CASES:
        O = k*n
        v = refine([(O, 1, 1), (R0, 0, 1)], O, q)
        if v[1] is None:
            print(f"(k,n)=({k},{n}) q={q} R0={R0}: no saving   [{note}]", flush=True)
        else:
            print(f"(k,n)=({k},{n}) q={q} R0={R0}: eps*gamma={v[0]:.6f} (x{v[0]/0.003935:.2f}) c={v[1]:.2f} "
                  f"gamma={v[2]:.4f} eps={v[3]:.4f}   [{note}]", flush=True)
