"""thin_region.py -- the region of thin shapes covered by Proposition "criterion" of Appendix A (subsection
"Thin full products"), checked numerically by interval subdivision (floating point, relative margin 1e-12; not a
symbolic proof).

Claim checked for a thin <o1,b,o2> (b < o1 <= o2): for all t in [0,1],
    g(t) := (1-c) r + c b >= b^(t/2),   c = (f^t-1)/(f-1),  f = o1 o2/(o1+o2-1),  r = 1 + (b-1)/o1,
where r n = n + (b-1) max(o1,o2) is the rank bound. Then every decomposition has sum_l S_l^t >= n b^(t/2): no saving.
Since r < b, g increases with c, and c decreases in f; f < o1 and f increases with o2, so the case f = o1 covers
every o2 >= o1.  On [t0,t1], g >= r + c(t0)(b-r) and b^(t/2) <= b^(t1/2); bisect until the lower bound beats the
upper bound (depth <= 40).
Output, per b = 2..12: K_all (every <o1,b,o2> with b < o1 <= K_all and every o2 >= o1 is covered), K_sq (every square
<k,b,k> with k <= K_sq is covered), and the three open shapes with the fewest outputs; then every open shape
with at most 1190 outputs for any b (only <34,2,35>).
Standard library only; about 10 minutes (b = 11, 12 take most of it).  Usage: python3 thin_region.py
"""
import math, sys
def ok_interval(b,m,rho,t0=0.0,t1=1.0,depth=0):
    if m<=b: return True
    if rho>=math.sqrt(b): return True
    c0=(m**t0-1)/(m-1)
    lo=(rho+c0*(b-rho))*(1-1e-12); hi=b**(t1/2)*(1+1e-12)
    if lo>hi: return True
    if depth>40: return False
    tm=(t0+t1)/2
    return ok_interval(b,m,rho,t0,tm,depth+1) and ok_interval(b,m,rho,tm,t1,depth+1)
def covered(o1,b,o2):
    o1,o2=min(o1,o2),max(o1,o2)
    return ok_interval(b,o1*o2/(o1+o2-1),1+(b-1)/o1)
def covered_all_o2(o1,b):          # limit m -> o1
    return ok_interval(b,float(o1),1+(b-1)/o1)
if __name__=="__main__":
    print("b  K_all (every <o1,b,o2>, b<o1<=K_all, any o2>=o1)  K_sq (every square <k,b,k>, k<=K_sq)  smallest-n open shapes")
    for b in range(2,13):
        K=b
        while covered_all_o2(K+1,b): K+=1
        Ks=b
        while covered(Ks+1,b,Ks+1): Ks+=1
        # check region is an interval (no holes below thresholds)
        holes=[o for o in range(b+1,K+1) if not covered_all_o2(o,b)]
        opens=[]
        for o1 in range(K+1,Ks+2):
            o2=o1
            while covered(o1,b,o2): o2+=1
            opens.append((o1*o2,(o1,b,o2)))
        opens.sort()
        print(b,K,Ks,'holes' if holes else '',[s for _,s in opens[:3]],flush=True)
    # smallest open shape over all b: a thin shape has o1 o2 >= (b+1)^2 > 1190 for b >= 34, so b <= 33 suffices
    small=[(o1*o2,(o1,b,o2)) for b in range(2,34) for o1 in range(b+1,1191)
           for o2 in range(o1,1190//o1+1) if not covered(o1,b,o2)]
    print("open shapes with at most 1190 outputs, any b:",sorted(small),flush=True)
