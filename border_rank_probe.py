"""Ancillary file of the paper "Thin matrix products up to D <= N^0.172 and lossless reductions for 3SUM,
Exact Triangle and APSP" (Proposition 3.2, an illustration; the proof is border_apolarity.py).

Numerical border-rank probe (heuristic, not a proof).  It asks whether the disjoint tensor
G(2,2,2) = <2,1,2> (+) <1,2,1> (an outer product <2,1,2> and an inner product of length 2 with inner variables
disjoint from the outer ones) has border rank at most 5, i.e. whether a border identity of length 2 in five terms
with disjoint inner variables exists.  Proposition A.4 does not exclude it.

Method: Levenberg-Marquardt minimisation of || T - sum_{l<=r} a_l (x) b_l (x) c_l ||^2 from 12 random starts.  The
set of tensors of border rank <= r is the closure of the rank-<= r tensors, so the infimum is 0 exactly when the
border rank is <= r; for a border identity it is approached with growing parameters.  Controls of known status
are run alongside: G(2,2,1) (Schoenhage's partial product <2,1,2> (+) <1,1,1>, rank 5), S(2,2,1) (the tensor of
Part I, border rank 5, approached with growing parameters), S(2,2,2) (border rank >= 6 by Remark A.7) and
G(2,2,3) (border rank >= 6 by Proposition A.4).  The output shows, per tensor, the four smallest squared residuals
with the largest parameter in brackets.  A residual that stops at a positive plateau (1.0 or 2.0) with bounded
parameters, from every start, is the behaviour of the infeasible controls; it is evidence, not a proof, that the
border rank exceeds r.

Tensors as arrays T[a, b, c]: a = left variables (x_1..x_k, p_1..p_q[, c]), b = right variables
(y_1..y_n, q_1..q_q), c = outputs (z_11..z_kn, z_0).

Run: python3 border_rank_probe.py      (numpy; about 75 s)
"""
import numpy as np


def G(k, n, q):
    """<k,1,n> (+) <1,q,1>: outer product plus a disjoint inner product of length q."""
    T = np.zeros((k + q, n + q, k*n + 1), dtype=np.int64)
    for i in range(k):
        for j in range(n):
            T[i, j, i*n + j] = 1
    for s in range(q):
        T[k + s, n + s, k*n] = 1
    return T


def S_tensor(k, n, q):
    """S(k,n,q) of Appendix A: inner form sum_s p_s q_s + c y_1, with y_1 shared."""
    T = np.zeros((k + q + 1, n + q, k*n + 1), dtype=np.int64)
    for i in range(k):
        for j in range(n):
            T[i, j, i*n + j] = 1
    for s in range(q):
        T[k + s, n + s, k*n] = 1
    T[k + q, 0, k*n] = 1
    return T


def lm(T, r, seed, iters=3000):
    A, B, C = T.shape
    rng = np.random.default_rng(seed)
    th = rng.normal(size=r*(A + B + C))
    Tf = T.astype(float).reshape(-1)

    def unpack(t):
        return t[:r*A].reshape(r, A), t[r*A:r*(A+B)].reshape(r, B), t[r*(A+B):].reshape(r, C)

    def fj(t):
        a, b, c = unpack(t)
        res = np.einsum('la,lb,lc->abc', a, b, c).reshape(-1) - Tf
        Ja = np.einsum('ad,lb,lc->abcld', np.eye(A), b, c).reshape(A*B*C, r*A)
        Jb = np.einsum('la,bd,lc->abcld', a, np.eye(B), c).reshape(A*B*C, r*B)
        Jc = np.einsum('la,lb,cd->abcld', a, b, np.eye(C)).reshape(A*B*C, r*C)
        return res, np.hstack([Ja, Jb, Jc])

    f, J = fj(th); val = f @ f; lam = 1e-2
    for _ in range(iters):
        g = J.T @ f; Hm = J.T @ J
        ok = False
        for _ in range(30):
            step = np.linalg.solve(Hm + lam*np.diag(np.diag(Hm) + 1e-12), -g)
            f2, J2 = fj(th + step); v2 = f2 @ f2
            if v2 < val:
                th, f, J, val = th + step, f2, J2, v2; lam = max(lam/3, 1e-15); ok = True; break
            lam *= 4
        if not ok:
            break
    return val, np.abs(th).max()


if __name__ == "__main__":
    cases = [("G(2,2,1) Schoenhage, border 5 (control yes)", G(2, 2, 1), 5),
             ("S(2,2,1) Part I tensor, border 5 (control yes)", S_tensor(2, 2, 1), 5),
             ("S(2,2,2), Remark A.7: >= 6 (control no)", S_tensor(2, 2, 2), 5),
             ("G(2,2,3), Proposition A.4: >= 6 (control no)", G(2, 2, 3), 5),
             ("G(2,2,2), open: border 5?", G(2, 2, 2), 5)]
    for name, T, r in cases:
        res = sorted(lm(T, r, s) for s in range(12))
        print(f"{name}: best residuals^2 (max|param|): " +
              ", ".join(f"{v:.1e} ({m:.0f})" for v, m in res[:4]), flush=True)
