"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and lossless reductions for 3SUM, Exact Triangle and APSP"
(Part I).  The paper's notation differs from the letters used here in a few places: the hashes G, F here are the
paper's re-indexing maps Gamma, Delta; the blocks Z_Q, W_Q are the paper's X_Q, Y_Q; kappa_g = phi_inf^2 is the
paper's mu_infinity^2 (phi_k here is the paper's mu_k).

Section 3 of the paper: the identity (Lemma 3.1) in the paper's notation, and Remark 5.2.

Checks, with sympy (python3):
  1. Cop82's five terms (p. 467; two signs lost in the scanned text are determined in check 5) sum to xi^2 T + O(xi^3).
  2. Under the renaming of Section 3 (a-variables are the outputs)
        x1 = b21, x2 = b31, p1 = b11, p2 = b12     (left)
        y1 = c11, y2 = c12, q = c21                (right)
        z11 = a12, z21 = a13, z12 = a22, z22 = a23, z0 = a11   (outputs)
     the five terms at xi = 1 are  phi_P * psi_P * chi_P  with
        phi_{P_ij} = x_i + phat_ij, psi_{P_ij} = y_j + qhat_ij, chi_{P_ij} = z_ij + z0,
        phi_{P_0} = -(x1+x2),       psi_{P_0} = y1 + y2,        chi_{P_0} = z0,
        phat = [[p1, -p2], [0, p2]],  qhat = [[0, 0], [-q, q]],
     and their sum is G + E with
        G = sum x_i y_j z_ij + (p1 y1 + p2 q) z0,
        E = sum_ij (x_i qhat_ij + phat_ij y_j + phat_ij qhat_ij) z_ij   (six monomials, listed).
  3. Every monomial of E contains an outer output z_ij and an inner input (p1, p2 or q), and has
     weight >= 3 in the grading p1:2, p2:1, q:1, z_ij:2 (others 0), while every monomial of G has weight 2.
  4. "Contributes to": z0 is in all five chi, each z_ij in exactly one (and these four terms are distinct).
"""
import sympy as sp

xi = sp.Symbol('xi')
a11, a12, a13, a22, a23 = sp.symbols('a11 a12 a13 a22 a23')
b11, b12, b21, b31 = sp.symbols('b11 b12 b21 b31')
c11, c12, c21 = sp.symbols('c11 c12 c21')

# 1. Cop82 p. 467 (trace convention a_ij b_jk c_ki)
cop = ((a11 + xi**2*a12)*(b21 + xi**2*b11)*c11 + (a11 + xi**2*a13)*b31*(c11 - xi*c21)
       + (a11 + xi**2*a22)*(b21 - xi*b12)*c12 + (a11 + xi**2*a23)*(b31 + xi*b12)*(c12 + xi*c21)
       - a11*(b21 + b31)*(c11 + c12))
T = a11*b11*c11 + a11*b12*c21 + a12*b21*c11 + a13*b31*c11 + a22*b21*c12 + a23*b31*c12
P = sp.Poly(sp.expand(cop), xi)
low = [P.coeff_monomial(xi**k) for k in range(3)]
assert sp.expand(low[0]) == 0 and sp.expand(low[1]) == 0 and sp.expand(low[2] - T) == 0
print("1. Cop82's five terms: coefficients of xi^0, xi^1 vanish, coefficient of xi^2 is T (6 monomials). OK")

# 2. renamed form at xi = 1
x1, x2, p1, p2, y1, y2, q = sp.symbols('x1 x2 p1 p2 y1 y2 q')
z11, z21, z12, z22, z0 = sp.symbols('z11 z21 z12 z22 z0')
ren = {b21: x1, b31: x2, b11: p1, b12: p2, c11: y1, c12: y2, c21: q,
       a12: z11, a13: z21, a22: z12, a23: z22, a11: z0}
x = [x1, x2]; y = [y1, y2]; z = [[z11, z12], [z21, z22]]
phat = [[p1, -p2], [0, p2]]
qhat = [[0, 0], [-q, q]]
terms = {}
for i in range(2):
    for j in range(2):
        terms[f"P{i+1}{j+1}"] = (x[i] + phat[i][j], y[j] + qhat[i][j], z[i][j] + z0)
terms["P0"] = (-(x1 + x2), y1 + y2, z0)
S = sp.expand(sum(f*g*h for f, g, h in terms.values()))
cop1 = sp.expand(cop.subs(xi, 1).subs(ren, simultaneous=True))
assert sp.expand(S - cop1) == 0
G = sum(x[i]*y[j]*z[i][j] for i in range(2) for j in range(2)) + (p1*y1 + p2*q)*z0
E = sum((x[i]*qhat[i][j] + phat[i][j]*y[j] + phat[i][j]*qhat[i][j])*z[i][j] for i in range(2) for j in range(2))
assert sp.expand(S - G - E) == 0
print("2. Renamed terms at xi=1 equal Cop82's terms at xi=1, and their sum is G + E. OK")
print("   column sums of phat:", [sp.simplify(phat[0][j] + phat[1][j]) for j in range(2)],
      " row sums of qhat:", [sp.simplify(qhat[i][0] + qhat[i][1]) for i in range(2)])
Ep = sp.Poly(sp.expand(E), x1, x2, p1, p2, y1, y2, q, z11, z21, z12, z22, z0)
print("   E =", sp.expand(E), f"({len(Ep.terms())} monomials)")

# 3. weights
wt = {p1: 2, p2: 1, q: 1, z11: 2, z21: 2, z12: 2, z22: 2}
gens = Ep.gens
def weight(mon):
    return sum(e*wt.get(g, 0) for g, e in zip(gens, mon))
Gp = sp.Poly(sp.expand(G), *gens)
assert all(weight(mn) == 2 for mn, _ in Gp.terms())
assert all(weight(mn) >= 3 for mn, _ in Ep.terms())
inner_in = {p1, p2, q}; outer_out = {z11, z21, z12, z22}
for mn, _ in Ep.terms():
    used = {g for g, e in zip(gens, mn) if e}
    assert used & inner_in and used & outer_out
print("3. G: all monomials weight 2; E: weights", sorted(weight(mn) for mn, _ in Ep.terms()),
      "; every E monomial has an outer output and an inner input. OK")

# 4. contributes-to
for name, (f, g, h) in terms.items():
    outs = [v for v in (z11, z21, z12, z22, z0) if sp.expand(h).coeff(v) != 0]
    print(f"   {name}: phi = {f}, psi = {g}, chi = {h}; contributes to {outs}")
assert all(sp.expand(h).coeff(z0) == 1 for _, _, h in terms.values())
print("4. z0 in all five terms with coefficient 1; z_ij only in P_ij. OK")

# 5. the two signs that the scan's text layer loses ((c11 ? x c21) and (b21 ? x b12)): only (-, -) works
ok = []
for s1 in (1, -1):
    for s2 in (1, -1):
        alt = ((a11 + xi**2*a12)*(b21 + xi**2*b11)*c11 + (a11 + xi**2*a13)*b31*(c11 + s1*xi*c21)
               + (a11 + xi**2*a22)*(b21 + s2*xi*b12)*c12 + (a11 + xi**2*a23)*(b31 + xi*b12)*(c12 + xi*c21)
               - a11*(b21 + b31)*(c11 + c12))
        Pa = sp.Poly(sp.expand(alt), xi)
        if all(sp.expand(Pa.coeff_monomial(xi**k)) == 0 for k in range(2)) and sp.expand(Pa.coeff_monomial(xi**2) - T) == 0:
            ok.append((s1, s2))
assert ok == [(-1, -1)]
print("5. of the four sign choices for the two signs illegible in the scanned copy of Cop82, only (-, -) gives xi^2 T + O(xi^3). OK")
