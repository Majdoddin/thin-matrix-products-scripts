"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and lossless reductions for 3SUM,
Exact Triangle and APSP" (Remark 10.3).

Brute-force check of the two index identities behind Remark 10.3 (Conjecture 5.12 of van den Brand, Nanongkai
and Saranurak on a larger region), over the Boolean semiring, computed over Z and then thresholded, with repeated
indices in I and J allowed.  For U (n x t1), N (n x n), V (t2 x n), I in [n]^t1, J in [n]^t2:
  first region:  (U N_{I,J} V)[i,j] = OR_b ( [(U N_{I,[n]})[i, J_b] > 0] and V[b,j] )   (t2 queries)
  second region: (U N_{I,J} V)[i,j] = OR_a ( U[i,a] and [(N_{I,J} V)[a,j] > 0] )        (t1 queries)

Run: python3 omv512_check.py      (standard library only; a few seconds)
"""
import random


def mm(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def pos(A):
    return [[1 if x > 0 else 0 for x in r] for r in A]


if __name__ == "__main__":
    random.seed(1)
    trials = 3000
    for _ in range(trials):
        n = random.randint(2, 8); t1 = random.randint(1, 5); t2 = random.randint(1, 5)
        R = lambda a, b: [[random.randint(0, 1) for _ in range(b)] for _ in range(a)]
        U, N, V = R(n, t1), R(n, n), R(t2, n)
        I = [random.randrange(n) for _ in range(t1)]; J = [random.randrange(n) for _ in range(t2)]
        NIJ = [[N[a][b] for b in J] for a in I]
        truth = pos(mm(mm(U, NIJ), V))
        P = mm(U, [N[a] for a in I])                     # first region: U N_{I,[n]}, inner dimension t1
        first = [[1 if any(P[i][J[b]] > 0 and V[b][j] for b in range(t2)) else 0 for j in range(n)]
                 for i in range(n)]
        Q = mm(NIJ, V)                                   # second region: N_{I,J} V, inner dimension t2
        second = [[1 if any(U[i][a] and Q[a][j] > 0 for a in range(t1)) else 0 for j in range(n)]
                  for i in range(n)]
        assert truth == first == second
    print(f"both identities hold on {trials} random Boolean instances (repeated indices allowed)")
