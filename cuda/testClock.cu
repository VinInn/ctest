// nvcc -gencode arch=compute_75,code=sm_75 -O3 --expt-relaxed-constexpr -std=c++23 testClock.cu -DNT=512 -DNB=4 -DMX=10000
#include "clockCuda.h"


struct Q {
   constexpr float operator()(float x) { return x+x*x;}
};

struct G {
  constexpr float operator()(int i) { return i*0.125e-4;}
};




int main() {

   doClock<G,Q,float>();

   return 0;
}
