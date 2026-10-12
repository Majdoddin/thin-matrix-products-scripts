"""thin_value.py -- what a thin shape outside the proved region could be worth (Appendix A, subsection "Thin full
products"; numerical, a t-grid, not a proof).

A single-type base of <o1,b,o2> run at L levels computes <o1^L, b^L, o2^L>, so D = b^L needs N >= max(o1,o2)^L,
eps = ln b / ln max(o1,o2), and in the leaf-count model eps*gamma = -Gamma / ln max(o1,o2) when Gamma < 0, where
Gamma = min_t [ln sum_l S_l^t - (t/2) ln b] - ln n. The forest bound and the rank bound give, for every decomposition,
sum_l S_l^t >= n((1-c) r + c b), c = (f^t-1)/(f-1), f = o1 o2/(o1+o2-1), r = 1 + (b-1)/min(o1,o2); so
eps*gamma <= max(0, -Gb)/ln max(o1,o2) with Gb = min_t [ln((1-c) r + c b) - (t/2) ln b]  (t on a grid).
Modes:
  squares  per b, the smallest square <k,b,k> whose bound allows eps*gamma > 0.005825 (the paper's rate value)
  rect     per b, the smallest o1 such that some o2 >= o1 (sampled) allows eps*gamma > 0.005825
  sup      per b, the largest allowed eps*gamma over square shapes (k sampled geometrically up to about 2*10^5)
Standard library only.  Usage: python3 thin_value.py [squares|rect|sup]   (default: all three)
"""
import math, sys

RATE = 0.005825


def Gb(o1, b, o2, steps=600):
    m = o1 * o2 / (o1 + o2 - 1); r = 1 + (b - 1) / min(o1, o2); best = 0.0
    for i in range(1, steps):
        t = i / steps; c = (m ** t - 1) / (m - 1)
        best = min(best, math.log((1 - c) * r + c * b) - t / 2 * math.log(b))
    return best


def allowed(o1, b, o2, steps=600):
    return max(0.0, -Gb(o1, b, o2, steps)) / math.log(max(o1, o2))


def squares():
    for b in [2, 3, 4, 5, 6, 8, 10, 16, 32]:
        k = b + 1
        while allowed(k, b, k) <= RATE:
            k += 1
        print('squares: b', b, 'first k with allowed eps*gamma > %.6f: %d  (value %.5f)' % (RATE, k, allowed(k, b, k)), flush=True)


def rect():
    for b in [2, 3, 4, 6, 10]:
        o1 = b + 1; found = None
        while not found:
            for o2 in sorted(set([o1 + j for j in range(40)] + [int(o1 * 1.15 ** j) for j in range(1, 80)])):
                v = allowed(o1, b, o2, 400)
                if v > RATE:
                    found = (o1, o2, v); break
            o1 += 1
        print('rect: b', b, 'smallest o1 with some o2 allowing > %.6f: o1=%d (o2=%d, value %.5f)' % ((RATE,) + found), flush=True)


def sup():
    best = (0, None)
    ks = sorted(set(int(round(1.08 ** j)) for j in range(1, 160)))
    for b in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 512,
              640, 800, 1000, 1300, 1600, 2000, 2600, 3200, 4000, 5000]:
        row = [(allowed(k, b, k), k) for k in ks if k > b]
        if row:
            v, k = max(row)
            print('sup: b', b, 'max allowed eps*gamma %.5f at k=%d' % (v, k), flush=True)
            if v > best[0]:
                best = (v, (k, b, k))
    print('sup: overall %.5f at %s (k is capped by the sample, so the supremum is at least this)' % best, flush=True)


if __name__ == '__main__':
    modes = sys.argv[1:] or ['squares', 'rect', 'sup']
    for mode in modes:
        {'squares': squares, 'rect': rect, 'sup': sup}[mode]()
