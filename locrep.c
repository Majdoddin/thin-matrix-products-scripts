/* Count (and optionally dump) the relations P subset [k]^3 all of whose Boolean minors
   (2 letters per position, C(k,2)^3 of them) are representable, i.e. lie in one of the
   10 REP orbits of Table 6 of the paper (60 relations).  By Proposition 20.14(d) these are
   exactly the relations without an OV gadget of any size.
   usage: locrep k [dumpfile [sample]]  (with a third argument: a uniform reservoir sample
   of 20000 of them, fixed seed, is written instead of all; relations as uint64 bitmasks
   over the cells a*k*k+b*k+c).  Helper of q2_functional4.py; build: cc -O2 -o locrep locrep.c */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static int k, N, nmin;
static unsigned char rep[256];          /* rep[r]=1 iff Boolean relation r representable */
static unsigned char feas[256][256];    /* feas[mask][val]: some rep r with r&mask==val */
static int mincells[1000][8];           /* cell index for bit (b1 b2 b3) of minor */
static int cellmin[64][64][2], ncellmin[64]; /* minors containing cell, and bit pos */
static int amask[1000], aval[1000];
static int P[64];
static long long cnt = 0; static FILE *dump = NULL;
#define S 20000
static uint64_t res[S]; static int sample=0; static uint64_t rs=88172645463325252ULL;
static uint64_t xr(void){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;return rs;}
static int bit(int a,int b,int c){return (a<<2)|(b<<1)|c;}
static void addrep(int *pts,int n){
  for(int pi=0;pi<6;pi++) for(int fl=0;fl<8;fl++){
    static const int perm[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    int r=0;
    for(int j=0;j<n;j++){int x[3]={pts[j]>>2&1,pts[j]>>1&1,pts[j]&1},y[3];
      for(int i=0;i<3;i++) y[i]=x[perm[pi][i]]^((fl>>i)&1);
      r|=1<<bit(y[0],y[1],y[2]);}
    rep[r]=1;}
}
static void rec(int cell){
  if(cell==N){cnt++; if(dump){uint64_t m=0; for(int i=0;i<N;i++) if(P[i]) m|=1ULL<<i;
    if(!sample) fwrite(&m,8,1,dump);
    else if(cnt<=S) res[cnt-1]=m; else {uint64_t j=xr()%cnt; if(j<S) res[j]=m;}} return;}
  for(int v=0;v<2;v++){
    int ok=1, j;
    for(j=0;j<ncellmin[cell];j++){int mi=cellmin[cell][j][0],b=cellmin[cell][j][1];
      amask[mi]|=1<<b; if(v) aval[mi]|=1<<b;
      if(!feas[amask[mi]][aval[mi]]){ok=0; j++; break;}}
    if(ok){P[cell]=v; rec(cell+1);}
    for(int jj=0;jj<j;jj++){int mi=cellmin[cell][jj][0],b=cellmin[cell][jj][1];
      amask[mi]&=~(1<<b); aval[mi]&=~(1<<b);}
  }
}
int main(int argc,char**argv){
  k=atoi(argv[1]); N=k*k*k; if(argc>2) dump=fopen(argv[2],"wb"); if(argc>3) sample=1;
  int R[10][8]={{0},{0},{0,1},{1,2},{3,4},{1,2,4},{0,1,2,3},{2,3,4,5},{0,3,5,6},{0,1,2,3,4,5,6,7}};
  int Rn[10]={0,1,2,2,2,3,4,4,4,8};
  for(int i=0;i<10;i++) addrep(R[i],Rn[i]);
  int nr=0; for(int r=0;r<256;r++) nr+=rep[r]; fprintf(stderr,"representable Boolean relations: %d\n",nr);
  for(int m=0;m<256;m++) for(int v=0;v<256;v++){ if(v&~m) continue;
    for(int r=0;r<256;r++) if(rep[r]&&((r&m)==v)){feas[m][v]=1;break;}}
  int pr[16][2],np=0; for(int a=0;a<k;a++) for(int b=a+1;b<k;b++){pr[np][0]=a;pr[np][1]=b;np++;}
  nmin=0;
  for(int x=0;x<np;x++)for(int y=0;y<np;y++)for(int z=0;z<np;z++){
    for(int b=0;b<8;b++){int c=pr[x][b>>2&1]*k*k+pr[y][b>>1&1]*k+pr[z][b&1];
      mincells[nmin][b]=c; cellmin[c][ncellmin[c]][0]=nmin; cellmin[c][ncellmin[c]][1]=b; ncellmin[c]++;}
    nmin++;}
  fprintf(stderr,"k=%d minors=%d\n",k,nmin);
  rec(0);
  printf("k=%d locally representable relations: %lld\n",k,cnt);
  if(dump&&sample) fwrite(res,8,cnt<S?cnt:S,dump);
  if(dump) fclose(dump);
  return 0;
}
