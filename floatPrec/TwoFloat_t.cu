#include "TwoFloat.h"
#include <cstdio>

#ifdef __NVCC__
__global__ 
#endif
void go(int k) {

  using namespace detailsTwoFloat;

 float h = std::sqrt(2.f);
 float l = 1.e-4*std::sqrt(3.f);
 TwoFloat<float> f(h,l, fromSum());
 TwoFloat<double> d(h,l, fromSum());

if (k==0) {
  printf("%a,%a\n",f.hi(),f.lo());;
  printf("%a\n", double(f.hi())+double(f.lo()));
  printf("%a,%a\n", d.hi() , d.lo() );
}

  auto f1 = f;
  TwoFloat<float> f2(-1.e-3*std::sqrt(3.f),1.e-6*std::sqrt(2.f),  fromSum());
  TwoFloat<float> f2n(1.e-3*std::sqrt(3.f),-1.e-6*std::sqrt(2.f), fromSum());
  double d1 = double(f.hi())+double(f.lo());
  double d2 = double(f2.hi())+double(f2.lo());
  double d2n = double(f2n.hi())+double(f2n.lo());

if (k==1) {
  printf("%a,%a\n", f2.hi() , f2.lo() );
  printf("%a\n", d2 );
}

  auto sf =  f1+f2;
  auto sd = d1 + d2;
if (k==2) {
  printf("sum\n" );
  printf("%a,%a\n", sf.hi() , sf.lo() );
  printf("%a\n", double(sf.hi()) + double(sf.lo()) );
  printf("%a\n", sd );
}
  auto sfn =  f1-f2n;
  auto sdn = d1 - d2n;

if (k==2) {
  printf("sub\n" );
  printf("%a,%a\n", sfn.hi() , sfn.lo() );
  printf("%a\n", double(sfn.hi()) + double(sfn.lo()) );
  printf("%a\n", sdn );
}

if (k==3) {
  printf("mul\n" );
  auto mf =  f1*f2.hi();
  auto md = d1 * f2.hi();
  printf("%a\n", f1.hi()*f2.hi() );
  printf("%a,%a\n", mf.hi() , mf.lo() );
  printf("%a\n", md );
}

if (k==4) {
  auto mf =  f1*f2;
  auto md = d1 * d2;
  printf("%a\n", f1.hi()*f2.hi() );
  printf("%a,%a\n", mf.hi() , mf.lo() );
  printf("%a\n", double(mf.hi()) + double(mf.lo()) );
  printf("%a\n", md );
}


if (k==5) {
  printf("div\n");
  auto mf =  f1/f2.hi();
  double md = d1/f2.hi();
  printf("%a\n", f1.hi()/f2.hi() );
  printf("%a,%a\n", mf.hi() , mf.lo() );
  printf("%a\n", md );
}

if (k==6) {
  auto mf =  f1/f2;
  auto md = d1/d2;
  printf("%a\n", f1.hi()/f2.hi() );
  printf("%a,%a\n", mf.hi() , mf.lo() );
  printf("%a\n", double(mf.hi()) + double(mf.lo()) );
  printf("%a\n", md );
}

}

int main(){
  for (int k=0; k<7; ++k) {
#ifdef __NVCC__
    go<<<1,1,0,0>>>(k);
    cudaDeviceSynchronize();
#else
    go(k);
#endif
  }
  return 0;

}
