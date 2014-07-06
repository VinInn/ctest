#include<cmath>
struct Loc {
  float x;
  int   index;
  float mdist=9999999;
  int   mindex=-1;

  float dist(Loc const & rh) const { return std::abs(x-rh.x);}
  
#pragma omp declare simd
  Loc & closest(Loc const & rh) { 
    auto d = dist(rh);
    if (d<mdist) {
      mdist=d;
      mindex=rh.index;
    }
    return *this;
  }

#pragma omp declare simd
  Loc & reduce(Loc const & rh) { 
    if(rh.mdist<mdist) {
      mdist=rh.mdist;
      mindex=rh.mindex;
    }
    return *this;
  }
};

#pragma omp declare reduction (foo:struct Loc: omp_out.reduce(omp_in))

Loc loc[1024];


void nn(Loc & me) {
  Loc nn=me;
#pragma omp simd  reduction(foo:nn)
  for (int i=0; i<1024; ++i) {
    nn.closest(loc[i]);
  }
  me = nn;

}




