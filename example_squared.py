"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and lossless reductions for 3SUM, Exact Triangle and APSP".

Generates every number of the worked example of the squared strings in Section 5 ("Strings", "The products",
Theorem 5.1) for L = 3, m = 1, and checks Theorem 5.1 on it by brute force.

  1. Hashes (the paper's re-indexing maps): Gamma is 1 x 3 and Delta is 2 x 3 over F_2.  The script takes the first pair, in lexicographic
     order of their entries, for which all three sets Q = {1}, {2}, {3} are good (Section 4).
  2. The strings of the example (levels 1, 2, 3 in each half):
        w' = z21 z0 z12,  w'' = z0 z11 z22   (output string w = w'w'')
        u' = x2 p2 x1,    u'' = q y1 y2      (left string u = u'u'')
        v' = y1 q y2,     v'' = p2 x1 x2     (right string v = v'v'')
     and their sets Q, P, S and index strings iota as defined in Section 5 (the paper writes T for P), the run s of u and of v, the row,
     column and inner indices at which a_s[u] and b_s[v] read X_Q and Y_Q, and the product of the level
     coefficients gamma_l (the coefficients in G + E of Lemma 3.1).
  3. Brute force: with random integer blocks X_Q, Y_Q for the good sets, the arrays a_s, b_s of the definition,
     and Mult(a_s, b_s)[w] = sum_{u,v} prod_l gamma_l(u_l, v_l, w_l) a_s[u] b_s[v] (Lemma 8 of the preprint),
     with nonzero entries (so that a summand vanishes only by the definitions), the sum over s = -1, 0, 1 equals the entry of X_{Q(w')} Y_{Q(w'')} named by Theorem 5.1, for every read
     output string w; and the nonzero summands for the example's w are listed.

Run: python3 example_squared.py   (standard library only).
"""
import itertools, random

L, m = 3, 1
XL = ['x1', 'x2', 'p1', 'p2']          # letters of an x-string
YL = ['y1', 'y2', 'q']                 # letters of a y-string
ZL = ['z11', 'z21', 'z12', 'z22', 'z0']

# G + E of Lemma 3.1, as {(x-letter, y-letter, z-letter): coefficient}
GE = {}
for i in (1, 2):
    for j in (1, 2):
        GE[(f'x{i}', f'y{j}', f'z{i}{j}')] = 1
GE[('p1', 'y1', 'z0')] = 1
GE[('p2', 'q', 'z0')] = 1
for mono, c in [(('p1', 'y1', 'z11'), 1), (('p2', 'y2', 'z12'), -1), (('p2', 'y2', 'z22'), 1),
                (('p2', 'q', 'z22'), 1), (('x2', 'q', 'z21'), -1), (('x2', 'q', 'z22'), 1)]:
    GE[mono] = GE.get(mono, 0) + c

def gamma_first(u, v, w):   # first half: left letter is an x-letter, right letter a y-letter
    return GE.get((u, v, w), 0)

def gamma_second(u, v, w):  # second half: left and right exchanged
    return GE.get((v, u, w), 0)

# ---------- F_2 helpers ----------
def colsum(M, P):
    """sum of the columns of M (list of rows) at the levels P (1-based), as a tuple over F_2"""
    return tuple(sum(row[l - 1] for l in P) % 2 for row in M)

def is_basis(M, P):
    cols = [tuple(row[l - 1] for row in M) for l in sorted(P)]
    span = {tuple(0 for _ in M)}
    for c in cols:
        new = {tuple((a + b) % 2 for a, b in zip(s, c)) for s in span}
        if new <= span:
            return False
        span |= new
    return len(span) == 2 ** len(M)

def good(Gam, Del, Q):
    return is_basis(Gam, Q) and is_basis(Del, set(range(1, L + 1)) - set(Q))

# ---------- 1. hashes ----------
def matrices(rows):
    for bits in itertools.product((0, 1), repeat=rows * L):
        yield [list(bits[r * L:(r + 1) * L]) for r in range(rows)]

ALLQ = [frozenset(c) for c in itertools.combinations(range(1, L + 1), m)]
Gam = Del = None
for G_ in matrices(m):
    for D_ in matrices(L - m):
        if all(good(G_, D_, Q) for Q in ALLQ):
            Gam, Del = G_, D_
            break
    if Gam:
        break
Qg = [Q for Q in ALLQ if good(Gam, Del, Q)]
print("1. hashes")
print(f"   Gamma = {Gam}   Delta = {Del}   (rows over F_2)")
print(f"   good sets: {[sorted(Q) for Q in Qg]}")

# ---------- string statistics (Section 5, 'Strings') ----------
def Qx(xi): return frozenset(l + 1 for l, a in enumerate(xi) if a in ('p1', 'p2'))
def Px(xi): return frozenset(l + 1 for l, a in enumerate(xi) if a == 'p2')
def iotax(xi): return tuple(int(a[1]) for a in xi if a in ('x1', 'x2'))
def Py(eta): return frozenset(l + 1 for l, a in enumerate(eta) if a == 'q')
def Sy(eta): return frozenset(l + 1 for l, a in enumerate(eta) if a == 'y2')
def Qz(ze): return frozenset(l + 1 for l, a in enumerate(ze) if a == 'z0')
def iotaz(ze): return tuple(int(a[1]) for a in ze if a != 'z0')
def Sz(ze): return frozenset(l + 1 for l, a in enumerate(ze) if a != 'z0' and a[2] == '2')

def fmt(S): return '{' + ','.join(map(str, sorted(S))) + '}'

# ---------- the arrays a_s, b_s ----------
rng = random.Random(1)
Xb, Yb = {}, {}
rows = [(i, d) for i in itertools.product((1, 2), repeat=L - m) for d in itertools.product((0, 1), repeat=L - m)]
inner = [(g1, g2) for g1 in itertools.product((0, 1), repeat=m) for g2 in itertools.product((0, 1), repeat=m)]
cols = [(d, i) for d in itertools.product((0, 1), repeat=L - m) for i in itertools.product((1, 2), repeat=L - m)]
for Q in Qg:
    Xb[Q] = {(r, c): rng.randint(1, 9) for r in rows for c in inner}
    Yb[Q] = {(c, k): rng.randint(1, 9) for c in inner for k in cols}

def a_s(s, u1, u2):
    Q = Qx(u1)
    if Q not in Qg or len(Py(u2)) - len(Px(u1)) != s:
        return 0
    return Xb[Q][((iotax(u1), colsum(Del, Sy(u2))), (colsum(Gam, Px(u1)), colsum(Gam, Py(u2))))]

def b_s(s, v1, v2):
    Q = Qx(v2)
    if Q not in Qg or len(Px(v2)) - len(Py(v1)) != s:
        return 0
    return Yb[Q][((colsum(Gam, Py(v1)), colsum(Gam, Px(v2))), (colsum(Del, Sy(v1)), iotax(v2)))]

XS = list(itertools.product(XL, repeat=L))
YS = list(itertools.product(YL, repeat=L))

def summands(w1, w2):
    out = []
    for u1 in XS:
        for v1 in YS:
            g1 = 1
            for l in range(L):
                g1 *= gamma_first(u1[l], v1[l], w1[l])
                if not g1: break
            if not g1: continue
            for u2 in YS:
                for v2 in XS:
                    g = g1
                    for l in range(L):
                        g *= gamma_second(u2[l], v2[l], w2[l])
                        if not g: break
                    if not g: continue
                    for s in range(-m, m + 1):
                        val = g * a_s(s, u1, u2) * b_s(s, v1, v2)
                        if val:
                            out.append((s, u1, u2, v1, v2, g, val))
    return out

# ---------- 2. the example ----------
w1, w2 = ('z21', 'z0', 'z12'), ('z0', 'z11', 'z22')
u1, u2 = ('x2', 'p2', 'x1'), ('q', 'y1', 'y2')
v1, v2 = ('y1', 'q', 'y2'), ('p2', 'x1', 'x2')
print("2. the example")
print(f"   w' = {' '.join(w1)}: Q = {fmt(Qz(w1))}, iota = {iotaz(w1)}, S = {fmt(Sz(w1))}")
print(f"   w'' = {' '.join(w2)}: Q = {fmt(Qz(w2))}, iota = {iotaz(w2)}, S = {fmt(Sz(w2))}")
print(f"   u' = {' '.join(u1)}: Q = {fmt(Qx(u1))}, P = {fmt(Px(u1))}, iota = {iotax(u1)}")
print(f"   u'' = {' '.join(u2)}: P = {fmt(Py(u2))}, S = {fmt(Sy(u2))}")
print(f"   v' = {' '.join(v1)}: P = {fmt(Py(v1))}, S = {fmt(Sy(v1))}")
print(f"   v'' = {' '.join(v2)}: Q = {fmt(Qx(v2))}, P = {fmt(Px(v2))}, iota = {iotax(v2)}")
su = len(Py(u2)) - len(Px(u1)); sv = len(Px(v2)) - len(Py(v1))
print(f"   run of u: |P(u'')| - |P(u')| = {su};  run of v: |P(v'')| - |P(v')| = {sv}")
print(f"   a_s[u] reads X_Q, Q = Q(u') = {fmt(Qx(u1))}, at row (iota(u'), Delta[S(u'')]) = "
      f"({iotax(u1)}, {colsum(Del, Sy(u2))}) and column (Gamma[P(u')], Gamma[P(u'')]) = "
      f"({colsum(Gam, Px(u1))}, {colsum(Gam, Py(u2))})")
print(f"   b_s[v] reads Y_Q, Q = Q(v'') = {fmt(Qx(v2))}, at row (Gamma[P(v')], Gamma[P(v'')]) = "
      f"({colsum(Gam, Py(v1))}, {colsum(Gam, Px(v2))}) and column (Delta[S(v')], iota(v'')) = "
      f"({colsum(Del, Sy(v1))}, {iotax(v2)})")
print(f"   Theorem 5.1 position for w: row (iota(w'), Delta[S(w'')]) = ({iotaz(w1)}, {colsum(Del, Sz(w2))}),"
      f" column (Delta[S(w')], iota(w'')) = ({colsum(Del, Sz(w1))}, {iotaz(w2)})")
coef = 1
for l in range(L):
    coef *= gamma_first(u1[l], v1[l], w1[l]) * gamma_second(u2[l], v2[l], w2[l])
print(f"   product of the six level coefficients gamma_l(u_l, v_l, w_l): {coef}")
S = summands(w1, w2)
print(f"   nonzero summands of sum_s Mult(a_s, b_s)[w]: {len(S)} (one per inner index, 2^m * 2^m = {4 ** m})")
for (s, a1, a2, b1, b2, g, val) in S:
    print(f"     s = {s:+d}: u = {' '.join(a1)} | {' '.join(a2)},  v = {' '.join(b1)} | {' '.join(b2)},  coeff {g},"
          f"  inner index (Gamma[P(u')], Gamma[P(u'')]) = ({colsum(Gam, Px(a1))}, {colsum(Gam, Py(a2))})")
assert any((a1, a2, b1, b2) == (u1, u2, v1, v2) and s == 0 for (s, a1, a2, b1, b2, g, val) in S)

# ---------- 3. Theorem 5.1 on all read outputs ----------
ZS = list(itertools.product(ZL, repeat=L))
bad = tot = 0
for w1_ in ZS:
    if Qz(w1_) not in Qg: continue
    for w2_ in ZS:
        if Qz(w2_) not in Qg: continue
        tot += 1
        lhs = sum(val for (*_, val) in summands(w1_, w2_))
        I = (iotaz(w1_), colsum(Del, Sz(w2_))); J = (colsum(Del, Sz(w1_)), iotaz(w2_))
        rhs = sum(Xb[Qz(w1_)][(I, c)] * Yb[Qz(w2_)][(c, J)] for c in inner)
        bad += lhs != rhs
print(f"3. Theorem 5.1 by brute force at L = {L}, m = {m}: {tot} read output strings, {bad} wrong")
assert bad == 0
