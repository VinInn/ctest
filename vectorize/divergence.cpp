#include<cmath>
#include<iostream>
#include<memory>

void set(double * v, int n, int flag) {
  for (int i=0; i<n; ++i) {
    v[i]=0;
    if (1==flag && 0==i%128) v[i]=M_PI;
    if (2==flag && 0==i%16) v[i]=M_PI;
    if (3==flag && 0==i%4) v[i]=M_PI;
    if (4==flag) v[i]=M_PI;
  }

}

void compute(double const * __restrict__ v, int n, double * __restrict__ res) {
  for (int i=0; i<n; ++i) {
     *res += (v[i]>0) ? 1./std::sqrt(v[i]) : 0;
  }
}




int main(int argc, char**) {

  std::cout << "flag " << argc-1 << std::endl;


  int size=4*1024;
  std::unique_ptr<double[]> v((double*)malloc(size*sizeof(double)));

  double res=0;
  set(v.get(),size,argc-1);  

  for (int i=0; i<500000; ++i) {
    compute(v.get(),size,&res);
  }

  std::cout << res << std::endl;
  return 0;

}
