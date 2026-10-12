"""Q2, functional relations over Σ = {0,1,2}: P_g = {(a,b,g(a,b))}, all 3^9 maps g.
For each g: REP, REP-Z, REP-2, REP-3 (certified), invariant factors of A_P,
OV-gadgets (t=1, t<=2, any t).  Orbits under independent letter permutations in
each position (216) and transposing positions 1,2.   Run: python3 q2_functional3.py
"""
import itertools, time, pickle
from collections import Counter
from multiprocessing import Pool
import numpy as np
from q2lib import rep_info, gadget_search, gadget_types, cell_perms, canon_codes

k = 3
GS = list(itertools.product(range(3), repeat=9))  # g[a*3+b]


def rel(g):
    return [(a, b, g[a * 3 + b]) for a in range(3) for b in range(3)]


def work(g):
    i = rep_info(rel(g), 3, primes=(2, 3))
    i.pop('witness')
    return i


def table(g):
    return '\n'.join('    ' + ' '.join(str(g[a * 3 + b]) for b in range(3)) for a in range(3))


if __name__ == '__main__':
    T0 = time.time()
    with Pool(4) as pool:
        infos = pool.map(work, GS, chunksize=200)
    T1 = time.time()
    print(f'REP computation {T1 - T0:.1f} s')
    tables = np.zeros((len(GS), 27), dtype=bool)
    for n, g in enumerate(GS):
        for a, b, c in rel(g):
            tables[n, a * 9 + b * 3 + c] = True
    res, summ = gadget_search(tables, 3)
    T2 = time.time()
    print(f'gadget search {T2 - T1:.1f} s')
    S3 = list(itertools.permutations(range(3)))
    perms = cell_perms(3, [S3] * 3, [(0, 1, 2), (1, 0, 2)])
    codes = canon_codes(tables, perms)
    T3 = time.time()
    print(f'orbits {T3 - T2:.1f} s')

    def cls(n):
        i = infos[n]
        return (i['REP'], i['REPZ'], i['REP_p'][2], i['REP_p'][3], tuple(i['torsion']), i['free'],
                bool(summ['t1'][n]), bool(summ['t2'][n]), bool(summ['tany'][n]), int(summ['tmin'][n]))

    orbits = {}
    for n in range(len(GS)):
        orbits.setdefault(int(codes[n]), []).append(n)
    # invariance check: classification constant on orbits
    bad = sum(1 for mem in orbits.values() if len({cls(n) for n in mem}) > 1)
    print(len(orbits), 'orbits; orbits with non-constant classification:', bad)

    print('counts over all 19683 g:')
    print('  REP', sum(i['REP'] for i in infos), ' REP-Z', sum(i['REPZ'] for i in infos),
          ' REP-2', sum(i['REP_p'][2] for i in infos), ' REP-3', sum(i['REP_p'][3] for i in infos))
    print('  gadget t=1', int(summ['t1'].sum()), ' t<=2', int(summ['t2'].sum()),
          ' any t', int(summ['tany'].sum()))
    print('  tmin distribution', Counter(summ['tmin'].tolist()))
    print('  REP and gadget(any t):', int(sum(1 for n in range(len(GS)) if infos[n]['REP'] and summ['tany'][n])))
    print('  torsion of A_P:', Counter((tuple(i['torsion']), i['free']) for i in infos))
    other_p = Counter(p for i in infos for p, v in i['REP_p'].items() if p not in (2, 3) and v and not i['REPZ'])
    print('  REP-p for p>3 without REP-Z:', other_p)

    joint = Counter()
    for n in range(len(GS)):
        i = infos[n]
        joint[(i['REP'], i['REPZ'], i['REP_p'][2], i['REP_p'][3], bool(summ['t1'][n]),
               bool(summ['t2'][n]), bool(summ['tany'][n]))] += 1
    print('joint (REP, REPZ, REP2, REP3, t1, t<=2, any t): #g')
    for kk, v in sorted(joint.items()):
        print('   ', kk, v)

    print('\nall orbits (rep = min code element):')
    third = []
    for code, mem in sorted(orbits.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        n = mem[0]
        i = infos[n]
        g = GS[n]
        t1, tany = gadget_types(res, n)
        tag = ''
        anyp = i['REPZ'] or any(i['REP_p'].values())
        if not summ['t2'][n] and not anyp:
            tag = '  <== THIRD CLASS'
            third.append(n)
        print(f"g={''.join(map(str, g))} size={len(mem):5d} REP={i['REP']!s:5} Z={i['REPZ']!s:5} "
              f"F2={i['REP_p'][2]!s:5} F3={i['REP_p'][3]!s:5} tors={i['torsion']} free={i['free']} "
              f"t1={t1} any={tany}{tag}")
    print(f'\n{len(third)} third-class orbits')
    for n in third:
        print(f"g={''.join(map(str, GS[n]))} orbit size {len(orbits[int(codes[n])])}:")
        print(table(GS[n]))
        print('   ', {kk: v for kk, v in infos[n].items()})
    pickle.dump(dict(infos=infos, summ=summ, codes=codes), open('q2_functional3.pkl', 'wb'))
    print(f'runtime {time.time() - T0:.1f} s')
