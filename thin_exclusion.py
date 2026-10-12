"""thin_exclusion.py -- the exact integer check behind Corollary "the 145 decompositions" of Appendix A
(subsection "Thin full products").

For a decomposition of the matrix multiplication tensor <o1,b,o2>, read with the output factor over the
n = o1*o2 output entries, let kappa = min(o1,o2). The paper proves (Lemma "forest bound", Lemma "rank bound",
Proposition "criterion") that every decomposition of the shape, of any rank and over any field, satisfies
sum_l S_l^t >= n*b^(t/2) for all t in [0,1], hence gives no saving, whenever kappa <= b or
kappa^(2(b-1)) >= b^(kappa-1). This script checks the integer condition EXACTLY
(a) for the 81 distinct thin shapes (b < min(o1,o2), written o1 <= o2) met among the 96 thin rotations of the 145
computer-search decompositions (93 AlphaTensor, 37 Kauers-Moosbauer flip graphs, 15 AlphaEvolve), listed below as
data (the list was extracted from those files once; no decomposition file is needed here); every other rotation
has kappa <= b; and
(b) for every shape <o1,b,o2> with b >= 2 and all dimensions <= DMAX (default 12), reporting the ones it misses,
and (c) the condition with the forest cap f = o1*o2/(o1+o2-1) in place of kappa (f <= b or f^(2(b-1)) >= b^(f-1),
also exact), for the shapes missed in (b).

Sanity checks (floating point, not part of the proof): the two-point inequality s^t >= (1-c) + c*min(s,k),
c = (k^t-1)/(k-1), for s = 1..n on a t-grid, and 1 + (b-1)c >= b^(t/2) on the same grid.
Standard library only. Run time: well under a second.
Usage: python3 thin_exclusion.py [DMAX]   (exit status 0 iff all 81 shapes of the 145 pass)
"""
import sys

THIN145 = [(3, 2, 3), (3, 2, 4), (3, 2, 5), (3, 2, 6), (4, 2, 4), (4, 2, 5), (4, 2, 6), (4, 2, 7), (4, 2, 8), (4, 3, 4), (4, 3, 5), (4, 3, 6), (4, 3, 7), (4, 3, 8), (4, 3, 11), (5, 2, 5), (5, 2, 6), (5, 3, 5), (5, 3, 6), (5, 3, 7), (5, 3, 9), (5, 4, 5), (5, 4, 6), (5, 4, 9), (5, 4, 10), (5, 4, 11), (6, 2, 6), (6, 3, 6), (6, 4, 6), (6, 5, 6), (7, 5, 9), (7, 5, 10), (7, 5, 11), (7, 6, 9), (7, 6, 10), (7, 6, 11), (8, 5, 9), (8, 5, 10), (8, 5, 11), (8, 6, 10), (8, 6, 11), (8, 7, 9), (8, 7, 10), (8, 7, 11), (8, 7, 12), (9, 3, 11), (9, 4, 10), (9, 4, 11), (9, 5, 9), (9, 5, 10), (9, 5, 11), (9, 5, 12), (9, 6, 9), (9, 6, 10), (9, 6, 11), (9, 7, 9), (9, 7, 10), (9, 7, 11), (9, 7, 12), (9, 8, 10), (9, 8, 11), (9, 8, 12), (10, 7, 10), (10, 7, 11), (10, 8, 10), (10, 8, 11), (10, 8, 12), (10, 9, 10), (10, 9, 11), (10, 9, 12), (11, 4, 11), (11, 4, 12), (11, 7, 11), (11, 8, 11), (11, 8, 12), (11, 9, 11), (11, 9, 12), (11, 10, 11), (11, 10, 12), (12, 10, 12), (12, 11, 12)]


def condition(k, b):
    """Exact: k^(2(b-1)) >= b^(k-1)  <=>  (b-1) ln k / (k-1) >= (ln b)/2."""
    return k ** (2 * (b - 1)) >= b ** (k - 1)


def condition_forest(o1, b, o2):
    """Exact: f <= b or f^(2(b-1)) >= b^(f-1), with f = p/q = o1*o2/(o1+o2-1) (the forest cap).
    For rational f the second inequality is p^(2(b-1)q) >= q^(2(b-1)q) * b^(p-q)."""
    p, q = o1 * o2, o1 + o2 - 1
    if p <= b * q:
        return True
    e = 2 * (b - 1) * q
    return p ** e >= q ** e * b ** (p - q)


def sanity(o1, b, o2, steps=200):
    n, k = o1 * o2, min(o1, o2)
    worst = float('inf')
    for i in range(steps + 1):
        t = i / steps
        c = (k ** t - 1) / (k - 1) if k > 1 else 0.0
        for s in range(1, n + 1):
            assert s ** t >= (1 - c) + c * min(s, k) - 1e-12, (o1, b, o2, t, s)
        lhs = 1 + (b - 1) * c if k >= b else b * k ** (t - 1)
        worst = min(worst, lhs - b ** (t / 2))
    return worst


def main(dmax):
    print("(a) thin shapes of the 145 computer-search decompositions:", len(THIN145))
    fail145 = []
    for (o1, b, o2) in THIN145:
        k = min(o1, o2)
        ok = condition(k, b)
        w = sanity(o1, b, o2, steps=50)
        print(f"  <{o1},{b},{o2}>  n={o1*o2:3d}  k={k:2d}  k^(2(b-1))={k**(2*(b-1))} >= b^(k-1)={b**(k-1)}: {ok}"
              f"   [float: min_t (1+(b-1)c - b^(t/2)) = {w:.4f}]")
        if not ok or w < -1e-12:
            fail145.append((o1, b, o2))
    print("  not excluded:", fail145 if fail145 else "none")
    missed = [(o1, b, o2) for o1 in range(1, dmax + 1) for b in range(2, dmax + 1) for o2 in range(o1, dmax + 1)
              if not condition(min(o1, o2), b) and min(o1, o2) > b]
    print(f"(b) all shapes with dims <= {dmax}: thin shapes NOT covered by the condition on kappa:", missed)
    missed_f = [s for s in missed if not condition_forest(*s)]
    print(f"(c) of these, NOT covered by the condition on the forest cap f either:", missed_f if missed_f else "none")
    return not fail145


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 12) else 1)
