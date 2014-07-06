#include<vector>
#include<algorithm>

struct A {
  int j;
};

inline bool lessA(const A & a, const A & b) {  return a.j<b.j; }


constexpr int N=1024;
int seq[N];

constexpr int incr=2;


void fill0(int K) {
  for (int j=0, i=0; i!=K; ++i, j+=incr)
    seq[i]=j;
}


int sortA(std::vector<A> & va) {
  const int N=va.size();
  int index[N];
  for (int i=0; i!=N; ++i)
    index[i]=i;

  std::sort(index,index+N,[&va](int i, int j) { return lessA(va[i],va[j]);});
  // std::sort(std::begin(index),std::end(index),[&va](int i, int j) { return lessA(va[i],va[j]);});

  return index[0];

}


/*
void fill1(int K) {
  seq[0]=0;
  for (int i=1; i!=K; ++i)
    seq[i]=seq[i-1]+incr;
}


void fillV(int K) {
  constexpr int incrV[8]={8*incr,8*incr,8*incr,8*incr,8*incr,8*incr,8*incr,8*incr};
  seq[0]=0;
  for(int j=1;j!=8;++j) seq[j]=j*incr;
  for (int i=8; i!=K/8; i+=8) 
    for(int j=0;j!=8;++j)
      seq[i+j]=seq[(i-8)+j]+incrV[j];
}

void fillV2(int K) {
  constexpr int incrV[8]={8*incr,8*incr,8*incr,8*incr,8*incr,8*incr,8*incr,8*incr};
  seq[0]=0;
  for(int j=1;j!=8;++j) seq[j]=j*incr;
  for (int i=8; i!=K/8; i+=8) {
    int * __restrict__ p = seq+i; int * __restrict__ t = p-8;
    for(int j=0;j!=8;++j)
      p[j]=t[j]+incrV[j];
  }
}
*/
