"""Q2, Boolean part: all 256 relations P ⊆ {0,1}^3.
REP / REP-Z / REP-2 / REP-3 via q2lib.rep_info (certified), OV-gadgets via
q2lib.gadget_search, orbits under S_3 (positions) x flips.  Cross-check of the
lattice test against a brute-force search of integer combinations with
coefficients in [-B, B].   Run: python3 q2_boolean.py
"""
import itertools, time
import numpy as np
from q2lib import rep_info, gadget_search, gadget_types, cell_perms, canon_codes

T0 = time.time()
k = 2
PTS = list(itertools.product(range(2), repeat=3))
tables = np.array([[bool(m >> (a * 4 + b * 2 + c) & 1) for (a, b, c) in PTS] for m in range(256)])
rels = [[p for p in PTS if tables[m][p[0] * 4 + p[1] * 2 + p[2]]] for m in range(256)]

infos = [rep_info(P, 2, primes=(2, 3), check_unimodular=True) for P in rels]

# brute-force cross-check of L-membership
B = 2
def e(s):
    v = [0] * 6
    v[s[0]] += 1; v[2 + s[1]] += 1; v[4 + s[2]] += 1
    return np.array(v)
mismatch = 0
for m, P in enumerate(rels):
    if len(P) < 2:
        continue
    G = np.array([e(p) - e(P[0]) for p in P[1:]])
    coeffs = np.array(list(itertools.product(range(-B, B + 1), repeat=len(G))))
    combos = {tuple(r) for r in coeffs @ G}
    bf_closure = {s for s in PTS if tuple(e(s) - e(P[0])) in combos}
    bf_rep = bf_closure == set(P)
    if bf_rep != infos[m]['REP']:
        mismatch += 1
        print('MISMATCH', m, P, bf_closure, infos[m])
print('brute-force cross-check mismatches:', mismatch)

res, summ = gadget_search(tables, 2)

perms = cell_perms(2, [list(itertools.permutations(range(2)))] * 3,
                   list(itertools.permutations(range(3))))
codes = canon_codes(tables, perms)
orbits = {}
for m in range(256):
    orbits.setdefault(int(codes[m]), []).append(m)
print(len(orbits), 'orbits')

both = neither = 0
for m in range(256):
    r = infos[m]['REP']
    g = bool(summ['t1'][m])
    if r and g: both += 1
    if not r and not g: neither += 1
print('relations with REP and t=1 gadget:', both, '; with neither:', neither)
print('REP but neither REP-Z nor REP-2:',
      sum(1 for i in infos if i['REP'] and not i['REPZ'] and not i['REP_p'][2]))
print('gadget any t but not t=1:', int((summ['tany'] & ~summ['t1']).sum()))

for code, mem in sorted(orbits.items(), key=lambda kv: (len(rels[kv[1][0]]), kv[0])):
    m = mem[0]
    i = infos[m]
    P = ' '.join(''.join(map(str, p)) for p in rels[m]) or '∅'
    t1, tany = gadget_types(res, m)
    print(f"{P:32s} orbit={len(mem):2d} REP={i['REP']!s:5} Z={i['REPZ']!s:5} "
          f"F2={i['REP_p'][2]!s:5} F3={i['REP_p'][3]!s:5} tors={i['torsion']} free={i['free']} "
          f"gadget_t1={t1}")
print(f'runtime {time.time() - T0:.1f} s')
