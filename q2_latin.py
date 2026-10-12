"""Q2, Latin-square relations P_g = {(a,b,g(a,b))}, g a quasigroup.
Order 4: all 576 Latin squares; orbits under isotopy (independent letter
permutations per position) and transposition; REP / REP-Z / REP-p, gadgets.
Order 5: the non-associative loop below (not isotopic to Z_5) and Z_5 itself.
Run: python3 q2_latin.py
"""
import itertools, time
from collections import Counter
import numpy as np
from q2lib import rep_info, gadget_search, gadget_types

T0 = time.time()


def latin_squares(k):
    out = []
    rows = list(itertools.permutations(range(k)))
    def rec(sq):
        if len(sq) == k:
            out.append(tuple(sq))
            return
        for r in rows:
            if all(r[j] != s[j] for s in sq for j in range(k)):
                rec(sq + [r])
    rec([])
    return out


def rel(sq):
    k = len(sq)
    return [(a, b, sq[a][b]) for a in range(k) for b in range(k)]


def tables_of(sqs, k):
    T = np.zeros((len(sqs), k ** 3), dtype=bool)
    for n, sq in enumerate(sqs):
        for a, b, c in rel(sq):
            T[n, a * k * k + b * k + c] = True
    return T


def report(sqs, k, primes, label):
    infos = [rep_info(rel(sq), k, primes=primes) for sq in sqs]
    res, summ = gadget_search(tables_of(sqs, k), k)
    return infos, res, summ


# ---- order 4
k = 4
sqs = latin_squares(4)
print(len(sqs), 'Latin squares of order 4')
G = np.array([[x for row in sq for x in row] for sq in sqs])  # (576,16), index a*4+b
S4 = list(itertools.permutations(range(4)))
w = 4 ** np.arange(16, dtype=np.int64)
best = None
for tr in (False, True):
    for pa in S4:
        for pb in S4:
            # g'(pa[a], pb[b]) = pc[g(a,b)]  (transposed: g'(pa[b], pb[a]))
            idx = np.empty(16, dtype=np.int64)
            for a in range(4):
                for b in range(4):
                    na, nb = (pa[a], pb[b]) if not tr else (pa[b], pb[a])
                    idx[na * 4 + nb] = a * 4 + b
            Gp = G[:, idx]
            for pc in S4:
                code = (np.array(pc)[Gp] * w).sum(axis=1)
                best = code if best is None else np.minimum(best, code)
orbits = Counter(best.tolist())
print('orbits under isotopy+transpose:', len(orbits), 'sizes', sorted(orbits.values()))
infos, res, summ = report(sqs, 4, (2, 3, 5), 'order 4')
cls = Counter()
for n in range(len(sqs)):
    i = infos[n]
    cls[(int(best[n]), i['REP'], i['REPZ'], i['REP_p'][2], i['REP_p'][3], tuple(i['torsion']),
         i['free'], bool(summ['t1'][n]), bool(summ['t2'][n]), bool(summ['tany'][n]))] += 1
print('(orbit code, REP, REPZ, REP2, REP3, torsion, free, t1, t<=2, any t): count')
for kk, v in sorted(cls.items()):
    print('  ', kk, v)
for code in orbits:
    n = int(np.nonzero(best == code)[0][0])
    print('representative:', sqs[n], 'gadget types', gadget_types(res, n))
print(f'order 4 done {time.time() - T0:.1f} s')

# ---- order 5
L5 = ((0, 1, 2, 3, 4),
      (1, 0, 3, 4, 2),
      (2, 4, 0, 1, 3),
      (3, 2, 4, 0, 1),
      (4, 3, 1, 2, 0))
Z5 = tuple(tuple((a + b) % 5 for b in range(5)) for a in range(5))
assert L5 in latin_squares(5) if False else all(sorted(r) == list(range(5)) for r in L5) and \
    all(sorted(c) == list(range(5)) for c in zip(*L5))
assoc = all(L5[L5[a][b]][c] == L5[a][L5[b][c]] for a in range(5) for b in range(5) for c in range(5))
print('L5 is a loop with identity 0:', all(L5[0][x] == x and L5[x][0] == x for x in range(5)),
      '; associative:', assoc, '(a loop isotopic to a group is isomorphic to it, so L5 is not isotopic to Z5)')
infos5, res5, summ5 = report([L5, Z5], 5, (2, 3, 5, 7), 'order 5')
for name, n in (('L5', 0), ('Z5', 1)):
    i = infos5[n]
    print(name, 'REP', i['REP'], 'REPZ', i['REPZ'], 'REP_p', i['REP_p'], 'torsion', i['torsion'],
          'free', i['free'], 'witness', i['witness'], 'gadget t1', bool(summ5['t1'][n]),
          't<=2', bool(summ5['t2'][n]), 'any t', bool(summ5['tany'][n]), gadget_types(res5, n))
print(f'runtime {time.time() - T0:.1f} s')
