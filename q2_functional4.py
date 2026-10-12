"""Q2, beyond Theorems 20.11 and 20.12: where the dichotomy stops (Proposition 20.14(d)-(f)).

Modes (run from inside anc/:  python3 q2_functional4.py [MODE]):

  functional4 (default)  all 4^16 maps g: {0..3}^2 -> {0..3}, P_g = {(a,b,g(a,b))}.
      Orbit representatives under letter permutations in each position and transposition of
      positions 1, 2 (group order 27648) by orderly generation (C helper enum_functional.c,
      compiled here with cc), checked against Burnside's lemma and against sum of orbit sizes
      = 4^16; OV gadgets of every size with q2lib.gadget_search; certified lattice tests with
      q2lib.rep_info (Z, F_p for p = 2, 3, 5, 7 and every prime dividing an invariant factor) and,
      for representable P, over groups of exponent dividing 4 (L_P + 4Z^12, Lemma 20.3(c)),
      certified the same way; the Boolean-minor criterion of Proposition 20.14(d) on every orbit.
      Prints the counts per class, the groups the representable orbits need, and the
      third-class orbits.  About 35 min with 4 processes.
  nonfunctional3  all 2^27 relations P over {0,1,2}: the relations all of whose Boolean minors
      are representable (backtracking, pure Python), the gadget search on them (none has a
      gadget) and on random other relations, certified lattice tests, and the orbits (group
      order 1296) of those that are not representable.  About 1 min.
  local4  evidence, not a proof (kind N): Burnside counts of all relations over 3 and 4 letters;
      the number of relations over {0,1,2,3} all of whose Boolean minors are representable
      (C helper locrep.c), and the representability of a uniform sample of 20000 of them.
      About 3 min.
"""
import itertools, math, os, subprocess, sys, time
from collections import Counter, defaultdict
from math import gcd
from multiprocessing import Pool
import numpy as np
from q2lib import rep_info, gadget_search, snf, matmul, SIDE_ASSIGNMENTS, cell_perms, canon_codes

HERE = os.path.dirname(os.path.abspath(__file__))
NPROC = 4


def build(name):
    exe = os.path.join(HERE, name)
    src = exe + '.c'
    if not os.path.exists(exe) or os.path.getmtime(exe) < os.path.getmtime(src):
        subprocess.run(['cc', '-O2', '-o', exe, src], check=True)
    return exe


# ------------------------------------------------------------- Burnside ----
def cycles(perm):
    seen, out = set(), []
    for s in range(len(perm)):
        if s not in seen:
            L, x = 0, s
            while x not in seen:
                seen.add(x)
                x = perm[x]
                L += 1
            out.append(L)
    return out


def burnside_functional(k):
    S = list(itertools.permutations(range(k)))
    tot = 0
    for tr in (0, 1):
        for pa in S:
            for pb in S:
                perm = [0] * (k * k)
                for a in range(k):
                    for b in range(k):
                        na, nb = (pa[a], pb[b]) if not tr else (pa[b], pb[a])
                        perm[a * k + b] = na * k + nb
                cyc = cycles(perm)
                for gam in S:
                    f = 1
                    for L in cyc:
                        x = 0
                        for v in range(k):   # letters fixed by gam^L
                            y = v
                            for _ in range(L):
                                y = gam[y]
                            x += y == v
                        f *= x
                    tot += f
    order = math.factorial(k) ** 3 * 2
    assert tot % order == 0
    return tot // order, order


def burnside_relations(k):
    S = list(itertools.permutations(range(k)))
    cells = list(itertools.product(range(k), repeat=3))
    idx = {c: i for i, c in enumerate(cells)}
    tot = 0
    for pp in itertools.permutations(range(3)):
        for lp in itertools.product(S, repeat=3):
            perm = [0] * len(cells)
            for s in cells:
                t = [None] * 3
                for i in range(3):
                    t[pp[i]] = lp[i][s[i]]
                perm[idx[s]] = idx[tuple(t)]
            tot += 2 ** len(cycles(perm))
    order = math.factorial(k) ** 3 * 6
    assert tot % order == 0
    return tot // order, order


# ------------------------------------------------- Boolean minors ----
def boolean_rep_masks():
    """bitmask m (bit a*4+b*2+c) -> representable, for all 256 Boolean relations"""
    pts = list(itertools.product(range(2), repeat=3))
    rep = np.zeros(256, dtype=bool)
    for m in range(256):
        P = [p for p in pts if m >> (p[0] * 4 + p[1] * 2 + p[2]) & 1]
        rep[m] = rep_info(P, 2, primes=(2,))['REP']
    return rep


def minors(k):
    """for each choice of a pair of letters per position, the 8 cells, bit b1*4+b2*2+b3"""
    pairs = list(itertools.combinations(range(k), 2))
    out = []
    for x, y, z in itertools.product(pairs, repeat=3):
        out.append([x[b >> 2 & 1] * k * k + y[b >> 1 & 1] * k + z[b & 1] for b in range(8)])
    return out


def all_minors_rep(tables, k, rep):
    ok = np.ones(tables.shape[0], dtype=bool)
    for cells in minors(k):
        code = np.zeros(tables.shape[0], dtype=np.int64)
        for b, c in enumerate(cells):
            code |= tables[:, c].astype(np.int64) << b
        ok &= rep[code]
    return ok


# ------------------------------------------------ lattice, exponent m ----
def rep_exponent(P, k, m):
    """Is P representable over an abelian group of exponent dividing m (Lemma 20.3(c): iff no s
    outside P has e(s)-e(s0) in L_P + mZ^{3k})?  Certified: membership by an explicit coefficient
    vector mod m, non-membership by the map x -> xV after checking that it sends L_P into the
    row space of diag(d_i)."""
    P = sorted(set(P))
    n = 3 * k

    def e(s):
        v = [0] * n
        v[s[0]] += 1
        v[k + s[1]] += 1
        v[2 * k + s[2]] += 1
        return v
    e0 = e(P[0])
    M = [[a - b for a, b in zip(e(p), e0)] for p in P[1:]]
    U, D, V, d = snf(M, n)
    r = len(d)
    assert matmul(matmul(U, M), V) == D
    for row in matmul(M, V):
        assert all(row[i] == 0 for i in range(r, n)) and all(row[i] % d[i] == 0 for i in range(r))
    Pset = set(P)
    for s in itertools.product(range(k), repeat=3):
        if s in Pset:
            continue
        x = [u - w for u, w in zip(e(s), e0)]
        y = [sum(x[i] * V[i][j] for i in range(n)) for j in range(n)]
        if all(y[j] % m == 0 for j in range(r, n)) and all(y[i] % gcd(d[i], m) == 0 for i in range(r)):
            z = []
            for i in range(r):
                h = gcd(d[i], m)
                mm = m // h
                z.append(((y[i] // h) * pow(d[i] // h, -1, mm)) % mm if mm > 1 else 0)
            z += [0] * (len(M) - r)
            c = [sum(z[i] * U[i][j] for i in range(len(M))) for j in range(len(M))]
            assert all((sum(c[i] * M[i][j] for i in range(len(M))) - x[j]) % m == 0 for j in range(n))
            return False
    return True


# ----------------------------------------------------------- functional4 ----
def rel(g, k):
    return [(a, b, g[a * k + b]) for a in range(k) for b in range(k)]


def tables_of(gs, k):
    T = np.zeros((len(gs), k ** 3), dtype=bool)
    for n, g in enumerate(gs):
        for a, b, c in rel(g, k):
            T[n, a * k * k + b * k + c] = True
    return T


def gadget_chunk(args):
    gs, k = args
    res, summ = gadget_search(tables_of(gs, k), k)
    return np.stack([res[s]['tmin'] for s in SIDE_ASSIGNMENTS])  # (12, chunk), 0 = no gadget


def lattice_work(g):
    P = rel(g, 4)
    i = rep_info(P, 4, primes=(2, 3, 5, 7))
    if i['REP']:
        i['exp4'] = rep_exponent(P, 4, 4)
    return i


def enumerate_reps(k, workers):
    exe = build('enum_functional')
    procs = [subprocess.Popen([exe, str(k), str(w), str(workers)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for w in range(workers)]
    reps = []
    for p in procs:
        out, err = p.communicate()
        assert p.returncode == 0, err
        for line in out.splitlines():
            gs, st, im, osz = line.split()
            reps.append((tuple(int(ch) for ch in gs), int(osz)))
    reps.sort()
    return reps


def fmt(g, k=4):
    return '/'.join(''.join(map(str, g[a * k:a * k + k])) for a in range(k))


def functional4():
    T0 = time.time()
    k = 4
    for kk in (2, 3):   # small cases of the enumeration: 4 and 84 orbits (Theorems 20.11, 20.12)
        r = enumerate_reps(kk, 1)
        nb, _ = burnside_functional(kk)
        print(f'enumeration k={kk}: {len(r)} representatives, Burnside {nb}, '
              f'sum of orbit sizes {sum(o for _, o in r)} (= {kk}^{kk * kk}: {sum(o for _, o in r) == kk ** (kk * kk)})')
    nb, order = burnside_functional(k)
    reps = enumerate_reps(k, NPROC)
    total = sum(o for _, o in reps)
    print(f'A  k=4: group order {order}; Burnside {nb} orbits; enumerated {len(reps)} representatives; '
          f'sum of orbit sizes {total} (= 4^16: {total == 4 ** 16}); agree: {nb == len(reps)}')
    assert nb == len(reps) and total == 4 ** 16
    T1 = time.time()
    print(f'   enumeration {T1 - T0:.1f} s', flush=True)

    gs = [g for g, _ in reps]
    chunks = [(gs[i:i + 4000], k) for i in range(0, len(gs), 4000)]
    with Pool(NPROC) as pool:
        tm = np.concatenate(pool.map(gadget_chunk, chunks), axis=1)   # (12, N)
    T2 = time.time()
    print(f'   gadget search {T2 - T1:.1f} s', flush=True)
    sides = [''.join(s) for s in SIDE_ASSIGNMENTS]
    hasg = (tm > 0).any(axis=0)
    tmin = np.where(tm > 0, tm, 10 ** 6).min(axis=0)
    slice_ = np.array([any(tm[j, n] == 1 for j, s in enumerate(sides) if 'C' in s) for n in range(len(gs))])

    with Pool(NPROC) as pool:
        infos = pool.map(lattice_work, gs, chunksize=200)
    T3 = time.time()
    print(f'   lattice tests {T3 - T2:.1f} s', flush=True)

    rep = boolean_rep_masks()
    loc = all_minors_rep(tables_of(gs, k), k, rep)
    print(f'B  Boolean relations representable: {int(rep.sum())} of 256; Proposition 20.14(d) on every orbit '
          f'(gadget of some size <=> some Boolean minor not representable): '
          f'{int((loc == ~hasg).sum())} of {len(gs)} agree')

    cls, clsm = Counter(), Counter()
    both = 0
    kinds = Counter()
    tmaps = Counter()
    for n, (g, osz) in enumerate(reps):
        i = infos[n]
        both += bool(i['REP'] and hasg[n])
        c = 'REP' if i['REP'] else ('gadget' if hasg[n] else 'neither')
        cls[c] += 1
        clsm[c] += osz
        if hasg[n]:
            kinds[(int(tmin[n]), 'slice' if slice_[n] else 'identification only')] += 1
            tmaps[int(tmin[n])] += osz
    print(f'C  orbits per class: {dict(cls)}')
    print(f'D  maps per class:   {dict(clsm)}')
    print(f'E  representable and with a gadget: {both}')
    print(f'F  gadget orbits by least size and kind: {dict(kinds)}; maps by least size: {dict(tmaps)}')

    def groups(i):
        s = ['Z'] if i['REPZ'] else []
        s += [f'F{p}' for p, v in sorted(i['REP_p'].items()) if v]
        if i.get('exp4'):
            s.append('exp4')
        return s

    def kind(i):
        gsn = groups(i)
        if 'Z' in gsn:
            return 'Z' + (', F2' if 'F2' in gsn else '') + (', F3' if 'F3' in gsn else '')
        if 'F2' in gsn:
            return 'F2 only'
        if 'F3' in gsn:
            return 'F3 only'
        if 'exp4' in gsn:
            return 'exponent 4 only'
        return 'other: ' + ','.join(gsn)
    kc, km = Counter(), Counter()
    for (g, osz), i in zip(reps, infos):
        if i['REP']:
            kc[kind(i)] += 1
            km[kind(i)] += osz
    print('G  representable orbits by the groups they need (orbits, maps):')
    for key in sorted(kc):
        print(f'     {key:28s} {kc[key]:3d} {km[key]:7d}')
    print('H  representable orbits (representative g rows a=0..3, orbit size, torsion and free rank of A_P, groups):')
    rows = [(kind(i), -osz, g, osz, i) for (g, osz), i in zip(reps, infos) if i['REP']]
    for kd, _, g, osz, i in sorted(rows):
        print(f'     {fmt(g)}  {osz:5d}  torsion {i["torsion"]} free {i["free"]}  {",".join(groups(i))}')
    third = [n for n in range(len(reps)) if not infos[n]['REP'] and not hasg[n]]
    print(f'I  third-class orbits (not representable, no gadget of any size): {len(third)}, '
          f'{sum(reps[n][1] for n in third)} maps')
    for n in sorted(third, key=lambda n: -reps[n][1]):
        g, osz = reps[n]
        i = infos[n]
        print(f'     {fmt(g)}  orbit size {osz}  torsion {i["torsion"]} free {i["free"]}  '
              f'closure witness {i["witness"]["L"]}')
    print(f'runtime {time.time() - T0:.1f} s')


# -------------------------------------------------------- nonfunctional3 ----
def local_relations(k, rep):
    """all P subset [k]^3 (bitmask over cells a*k*k+b*k+c) whose Boolean minors are all
    representable; backtracking over the cells with pruning by partly assigned minors"""
    N = k ** 3
    mins = minors(k)
    feas = np.zeros((256, 256), dtype=bool)   # feas[mask][val]: some rep r with r & mask == val
    reps = np.nonzero(rep)[0]
    for m in range(256):
        for r in reps:
            feas[m, r & m] = True
    feas = feas.tolist()
    at = defaultdict(list)
    for mi, cells in enumerate(mins):
        for b, c in enumerate(cells):
            at[c].append((mi, 1 << b))
    amask = [0] * len(mins)
    aval = [0] * len(mins)
    out = []

    def rec(c, P):
        if c == N:
            out.append(P)
            return
        for v in (0, 1):
            done = []
            ok = True
            for mi, bit in at[c]:
                amask[mi] |= bit
                if v:
                    aval[mi] |= bit
                done.append((mi, bit))
                if not feas[amask[mi]][aval[mi]]:
                    ok = False
                    break
            if ok:
                rec(c + 1, P | (v << c))
            for mi, bit in done:
                amask[mi] &= ~bit
                aval[mi] &= ~bit
    rec(0, 0)
    return out


def nonfunctional3():
    T0 = time.time()
    k, N = 3, 27
    rep = boolean_rep_masks()
    loc = local_relations(k, rep)
    print(f'A  relations over 3 letters with all 27 Boolean minors representable: {len(loc)}')
    tables = ((np.array(loc, dtype=np.int64)[:, None] >> np.arange(N)) & 1).astype(bool)
    _, s = gadget_search(tables, k)
    print(f'B  of these, with an OV gadget of some size: {int(s["tany"].sum())}')
    rng = np.random.default_rng(1)
    locset = set(loc)
    rnd = [m for m in rng.integers(0, 1 << N, 3000).tolist() if m not in locset][:2000]
    rt = ((np.array(rnd, dtype=np.int64)[:, None] >> np.arange(N)) & 1).astype(bool)
    _, s2 = gadget_search(rt, k)
    print(f'C  random relations with a non-representable Boolean minor: {len(rnd)}, '
          f'with a gadget of size 1: {int(s2["t1"].sum())}')
    cells = list(itertools.product(range(k), repeat=3))
    bad = []
    for m in loc:
        P = [cells[i] for i in range(N) if m >> i & 1]
        if not rep_info(P, k, certify=True)['REP']:
            bad.append(m)
    S3 = list(itertools.permutations(range(3)))
    perms = cell_perms(k, [S3] * 3, list(itertools.permutations(range(3))))
    bt = ((np.array(bad, dtype=np.int64)[:, None] >> np.arange(N)) & 1).astype(bool)
    codes = canon_codes(bt, perms)
    orb = defaultdict(list)
    for m, c in zip(bad, codes.tolist()):
        orb[c].append(m)
    print(f'D  not representable, no gadget (third class): {len(bad)} relations in {len(orb)} orbits '
          f'(group order {len(perms)})')
    for c, ms in sorted(orb.items(), key=lambda x: (bin(x[1][0]).count('1'), x[0])):
        m = min(ms)
        P = [cells[i] for i in range(N) if m >> i & 1]
        r = rep_info(P, k)
        print(f'     orbit size {len(ms):4d}  |P| = {len(P)}  P = {" ".join("".join(map(str, p)) for p in P)}'
              f'  torsion {r["torsion"]} free {r["free"]}  closure witness {r["witness"]["L"]}')
    z3 = [p for p in cells if sum(p) % 3 == 0 and p != (0, 0, 0)]
    r = rep_info(z3, k)
    m = sum(1 << (a * 9 + b * 3 + c) for a, b, c in z3)
    _, s3 = gadget_search(((np.array([m])[:, None] >> np.arange(N)) & 1).astype(bool), k)
    print(f'E  {{a+b+c = 0 mod 3}} minus {{000}}: REP {r["REP"]}, closure witness {r["witness"]["L"]}, '
          f'gadget of some size {bool(s3["tany"][0])}, in the third class above {m in set(bad)}')
    print(f'runtime {time.time() - T0:.1f} s')


# ----------------------------------------------------------------- local4 ----
def local4():
    T0 = time.time()
    for k in (3, 4):
        n, o = burnside_relations(k)
        print(f'A  all relations over {k} letters: 2^{k ** 3}, group order {o}, orbits {n} (about {float(n):.3g})')
    exe = build('locrep')
    dump = os.path.join(HERE, 'locrep4_sample.bin')
    out = subprocess.run([exe, '4', dump, 'sample'], capture_output=True, text=True, check=True)
    print('B  locrep:', out.stderr.strip().replace('\n', '; '), ';', out.stdout.strip())
    masks = np.fromfile(dump, dtype=np.uint64)
    os.remove(dump)
    cells = list(itertools.product(range(4), repeat=3))
    st = Counter()
    for m in masks.tolist():
        P = [cells[i] for i in range(64) if m >> i & 1]
        r = rep_info(P, 4, certify=False)
        st['REP' if r['REP'] else 'not REP'] += 1
    print(f'C  uniform sample of {len(masks)} of them (evidence only): {dict(st)}')
    print(f'runtime {time.time() - T0:.1f} s')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'functional4'
    {'functional4': functional4, 'nonfunctional3': nonfunctional3, 'local4': local4}[mode]()
