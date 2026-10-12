/* Orbit representatives of maps g: [K]^2 -> [K] under letter permutations of a, b, c
   and transposition of a, b (group order (K!)^3 * 2).
   g is the string g[a*K+b], a,b < K.  Canonical form: the lexicographically least string
   among RGS(g o sigma), sigma ranging over the (K!)^2*2 cell permutations, where RGS
   relabels values in order of first occurrence (the least relabelling under value
   permutations).  We enumerate all RGS strings (values in order of first occurrence,
   at most K values) and keep those equal to their canonical form (orderly generation).
   Output per representative:  g  |Stab_sigma|  |im g|  orbitsize
   orbitsize = (K!)^3*2 / (|{sigma : RGS(g o sigma) = g}| * (K-|im g|)!).
   Usage: enum_functional K worker nworkers > out   (work split by a prefix counter)
   Helper of q2_functional4.py; build: cc -O2 -o enum_functional enum_functional.c */
#include <stdio.h>
#include <stdlib.h>

static int K, N, NS;
static int sig[1152][16];
static int g[16];
static long long nrep = 0, ncand = 0, orbsum = 0;
static int worker, nworkers;
static long long split = 0;
#define SPLITDEPTH 7

static int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }

static void perms(int k, int out[][4], int *cnt) {
    int p[4] = {0, 1, 2, 3}, c = 0;
    /* all permutations of 0..k-1 by brute force */
    for (int a = 0; a < k; a++) for (int b = 0; b < k; b++) for (int d = 0; d < (k > 2 ? k : 1); d++)
    for (int e = 0; e < (k > 3 ? k : 1); e++) {
        int q[4] = {a, b, d, e}, ok = 1;
        for (int i = 0; i < k; i++) for (int j = 0; j < i; j++) if (q[i] == q[j]) ok = 0;
        if (!ok) continue;
        for (int i = 0; i < k; i++) out[c][i] = q[i];
        c++;
    }
    (void)p;
    *cnt = c;
}

/* returns -1 if g is not canonical, else number of sigma with RGS(g o sigma) = g */
static int canon(void) {
    int st = 0;
    for (int s = 0; s < NS; s++) {
        int map[4] = {-1, -1, -1, -1}, nx = 0, cmp = 0;
        const int *S = sig[s];
        for (int i = 0; i < N; i++) {
            int v = g[S[i]];
            if (map[v] < 0) map[v] = nx++;
            int w = map[v];
            if (w < g[i]) { cmp = -1; break; }
            if (w > g[i]) { cmp = 1; break; }
        }
        if (cmp < 0) return -1;
        if (cmp == 0) st++;
    }
    return st;
}

static void rec(int i, int mx) {
    if (i == SPLITDEPTH) {
        if ((split++ % nworkers) != worker) return;
    }
    if (i == N) {
        ncand++;
        int st = canon();
        if (st > 0) {
            int m = mx + 1;
            long long order = (long long)fact(K) * fact(K) * fact(K) * 2;
            long long stab = (long long)st * fact(K - m);
            long long osz = order / stab;
            if (osz * stab != order) { fprintf(stderr, "bad stab\n"); exit(1); }
            for (int j = 0; j < N; j++) putchar('0' + g[j]);
            printf(" %d %d %lld\n", st, m, osz);
            nrep++;
            orbsum += osz;
        }
        return;
    }
    int top = mx + 1 < K - 1 ? mx + 1 : K - 1;
    for (int v = 0; v <= top; v++) {
        g[i] = v;
        rec(i + 1, v > mx ? v : mx);
    }
}

int main(int argc, char **argv) {
    K = atoi(argv[1]);
    worker = argc > 2 ? atoi(argv[2]) : 0;
    nworkers = argc > 3 ? atoi(argv[3]) : 1;
    N = K * K;
    int P[24][4], np;
    perms(K, P, &np);
    NS = 0;
    /* put the identity first? order irrelevant for correctness */
    for (int tr = 0; tr < 2; tr++)
        for (int x = 0; x < np; x++)
            for (int y = 0; y < np; y++) {
                for (int a = 0; a < K; a++)
                    for (int b = 0; b < K; b++)
                        sig[NS][a * K + b] = tr ? P[x][b] * K + P[y][a] : P[x][a] * K + P[y][b];
                NS++;
            }
    g[0] = 0;
    rec(1, 0);
    fprintf(stderr, "K=%d worker %d/%d: candidates %lld, representatives %lld, sum of orbit sizes %lld\n",
            K, worker, nworkers, ncand, nrep, orbsum);
    return 0;
}
